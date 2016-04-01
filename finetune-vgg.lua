
require 'torch'
require 'nn'
require 'nngraph'
require 'loadcaffe'
require 'cunn'
require 'cutorch'
require 'cudnn'
require 'SpatialGlimpse'
require 'optim'
cudnn.benchmark = true
cutorch.manualSeed(123)
cutorch.setDevice(1)

---------------------------------
--1.load the data
---------------------------------
trsize = 300
vasize = 20           
tesize = 200
wi = 32
hi = 32

--traning data and validationset
trainset = {
	data = torch.Tensor(10000, 3*wi*hi),
	labels = torch.Tensor(10000),
	size = function() return trsize end
}
validationset = {
	data = torch.Tensor(1500, 3*wi*hi),
	labels = torch.Tensor(1500),
	size = function() return vasize end
}

for i = 0, 0 do
	print('loading the trainset: ' .. 'cifar-10-batches-t7/data_batch_' .. (i+1) .. '.t7')
	subset = torch.load('cifar-10-batches-t7/data_batch_' .. (i+1) .. '.t7','ascii')
	trainset.data[{ {i*10000+1, i*10000+10000} }] = subset.data:t()
	trainset.labels[{ {i*10000+1, i*10000+10000} }] = subset.labels
end

validationset.data[{ {1,1500} }] = trainset.data[{ {8501,10000} }]
validationset.labels[{ {1,1500} }] = trainset.labels[{ {8501,10000} }]

validationset.labels = validationset.labels + 1
trainset.labels = trainset.labels + 1

---test data
print('loading the testset: ' .. 'cifar-10-batches-t7/test_batch.t7')
subset = torch.load('cifar-10-batches-t7/test_batch.t7', 'ascii')
testset = {
	data = subset.data:t():double(),
	labels = subset.labels[1]:double(),
	size = function() return tesize end
}
testset.labels = testset.labels + 1

-- resize dataset (if just using small set)
trainset.data = trainset.data[{ {1,trsize} }]
trainset.labels = trainset.labels[{ {1,trsize} }]

validationset.data = validationset.data[{ {1,vasize} }]
validationset.labels = validationset.labels[{ {1,vasize} }]

testset.data = testset.data[{ {1,tesize} }]
testset.labels = testset.labels[{ {1,tesize} }]

---reshape data
trainset.data = trainset.data:reshape(trsize,3,wi,hi)
validationset.data = validationset.data:reshape(vasize,3,wi,hi)
testset.data = testset.data:reshape(tesize,3,wi,hi)


-------------------------------------------------
--2.load the vggnet
-------------------------------------------------

cnn_proto = 'VGG_ILSVRC_16_layers_deploy.prototxt'
cnn_model = 'VGG_ILSVRC_16_layers.caffemodel'
cnn_backend = 'cudnn'           --nn|cudnn
output_size = 1000
layer_num = 38


function build_cnn(cnn)
	require 'cudnn'
	local cnn_part = nn.Sequential()   --copy the vggnet to cnn_part
	for i =1,layer_num do
		local layer = cnn:get(i)
		if i == 1 then
			-- convert kernels in first conv layer into RGB format instead of BGR,
      			-- which is the order in which it was trained in Caffe
			local w = layer.weight:clone()
			-- swap weights to R and B channels
			print('converting first layer conv filters from BGR to RGB...')
			layer.weight[{ {}, 1, {}, {} }]:copy(w[{ {}, 3, {}, {} }])
			layer.weight[{ {}, 3, {}, {} }]:copy(w[{ {}, 1, {}, {} }])
		end
		cnn_part:add(layer)
	end
	print('finish building cnn')

	-- cnn_part:add(nn.Linear(4096,encoding_size))
	-- cnn_part:add(cnn_backend.ReLU(true))
	return cnn_part
end

local protos = {}                --store for the model
cnn_raw = loadcaffe.load(cnn_proto,cnn_model,cnn_backend)
cnn_tower1 = build_cnn(cnn_raw)
cnn_tower2 = cnn_tower1:clone('weight','bias','gradWeight','gradBias')
cnn_tower3 = cnn_tower1:clone('weight','bias','gradWeight','gradBias')
collectgarbage() 

--------------------------------
--3.combine the new models
--------------------------------

--first combination
s0 = nn.Sequential()  --input:depth*batch*channel*224*224
d0 = nn.Parallel(1,2)
d0:add(cnn_tower1)
d0:add(cnn_tower2)
d0:add(cnn_tower3)
s0:add(d0)
s0:add(nn.Linear(4096*3,output_size))
s0:add(nn.LogSoftMax())

--second combination
s1 = nn.Sequential()
d1 = nn.Parallel(1,2)
d1:add(cnn_tower1)
d1:add(cnn_tower1)
s1:add(d1)
s1:add(nn.Linear(4096*2,output_size))
s1:add(nn.LogSoftMax())

--3rd combination
s2 = nn.Sequential()
d2 = nn.Parallel(1,2)
d2:add(cnn_tower1)
d2:add(cnn_tower1)
s2:add(d2)
s2:add(nn.Linear(4096*2,output_size))
s2:add(nn.LogSoftMax())

--4rd combination
s3 = nn.Sequential()
d3 = nn.Parallel(1,2)
d3:add(cnn_tower1)
d3:add(cnn_tower1)
s3:add(d3)
s3:add(nn.Linear(4096*2,output_size))
s3:add(nn.LogSoftMax())

---------------------------------------
--5.train the model
---------------------------------------
max_iter = 5
threshold = 1
batch_size = 7
gsize = 8             --glimpse size
depth = 3

model = {s0 , s1, s2, s3 }             --gather the model

criterion = nn.ClassNLLCriterion():cuda()

sgd_params = {
	learningRate = 0.01,
	learningRateDecay =1e-4,
	weightDecay = 1e-3,
	momentum = 1e-4
}

for loop = 1,3 do
	for index = 1,1 do
		print(string.format('current loop %d , index %d...', loop, index))
		x, dl_dx = model[index]:cuda():getParameters()

		step = function (idx)
			model[idx]:training()
			local current_loss = 0
			local count = 0
			local shuffle = torch.randperm(trsize)  
			for t = 1,trsize,batch_size do    --get a minibatch data
				local size = math.min(t + batch_size - 1, trsize) -t +1
				local original_inputs = torch.Tensor(size, 3, wi, hi)
				local targets = torch.Tensor(size)
				for i = 0,size-1 do
					local input = trainset.data[shuffle[i+t]]
					local output = trainset.labels[shuffle[i+t]]
					original_inputs[i+1] = input
					targets[i+1] = output
				end
				targets = targets:cuda()

				------get a glimpse into 3 scale image---------------------------------
				loc = torch.Tensor(size,2)
				for i = 1,size do
					--location between (-1,1)
					loc[i][1] = torch.rand(1) - torch.rand(1)  
					loc[i][2] = torch.rand(1) - torch.rand(1)
				end
				glimpse = nn.SpatialGlimpse(gsize)
				inp = {original_inputs,loc}
				original_inputs = glimpse:forward(inp)
				-----------------------------------------------------------------------


				------resize the image size to fit into the vggnet for different model---------------------
				local inputs
				if idx == 1 then       --low,medium,high resolution combine
					inputs = torch.Tensor(depth,size,3,224,224):zero()   --the vggnet input 3*224*224
					for i=1,depth do
						for j=1,size do				
							local center = inputs[i][j]:narrow(2,96,gsize):narrow(3,96,gsize)
							center:copy(original_inputs[i][j])
						end
					end
					inputs = inputs:cuda()    --used for GPU
				end
				
				if idx == 2 then        --low and medium resolution combine
					inputs = torch.Tensor(2,size,3,224,224):zero()   --the vggnet input 3*224*224
					for i=1,2 do
						for j=1,size do				
							local center = inputs[i][j]:narrow(2,96,gsize):narrow(3,96,gsize)
							center:copy(original_inputs[i][j])
						end
					end
					inputs = inputs:cuda()    --used for GPU
				end
				
				if idx == 3 then      --medium and high resolution combine
					inputs = torch.Tensor(2,size,3,224,224):zero()   --the vggnet input 3*224*224
					for i=2,3 do
						for j=1,size do				
							local center = inputs[i-1][j]:narrow(2,96,gsize):narrow(3,96,gsize)
							center:copy(original_inputs[i][j])
						end
					end
					inputs = inputs:cuda()    --used for GPU
				end

				if idx == 4 then     ---low and high resolution combine
					inputs = torch.Tensor(2,size,3,224,224):zero()   --the vggnet input 3*224*224
					for j=1,size do
						local center1 = inputs[1][j]:narrow(2,96,gsize):narrow(3,96,gsize)
						local center2 = inputs[2][j]:narrow(2,96,gsize):narrow(3,96,gsize)
						center1:copy(original_inputs[1][j])
						center2:copy(original_inputs[3][j])
					end
					inputs = inputs:cuda()    --used for GPU
				end
				----------------------------------------------------------------------------------------

				collectgarbage() 
				local feval = function(x_new)         --the function to update output and gradinput for sgd
					if x ~= x_new then x:copy(x_new) end
					dl_dx:zero()

					local loss = criterion:forward(model[idx]:forward(inputs), targets)
					model[idx]:backward(inputs, criterion:backward(model[idx].output, targets))
					return loss, dl_dx
				end

				_ , fs = optim.sgd(feval, x, sgd_params)
				collectgarbage() 
				count = count +1
				current_loss = current_loss +fs[1]
				print('batch' .. t .. ' current_loss: ' .. current_loss)
			end

			return current_loss/count
		end

		eval = function(idx)                 --eval for the validationset
			model[idx]:evaluate()
			local count = 0
			for t = 1, vasize , batch_size do
				local size = math.min(t + batch_size - 1, vasize) - t +1
				local original_inputs = validationset.data[{{t,t + size - 1}}]
				local targets = validationset.labels[{{t,t + size -1}}]:long()
				targets = targets:cuda()
				
				-----------get a glimpse-------------
				loc = torch.Tensor(size,2)
				for i = 1,size do
					loc[i][1] = torch.rand(1) - torch.rand(1)
					loc[i][2] = torch.rand(1) - torch.rand(1)
				end
				glimpse = nn.SpatialGlimpse(gsize)
				inp = {original_inputs,loc}
				original_inputs = glimpse:forward(inp)


				------resize the image size to fit into the vggnet for different model---------------------
				local inputs
				if idx == 1 then       --low,medium,high resolution combine
					inputs = torch.Tensor(depth,size,3,224,224):zero()   --the vggnet input 3*224*224
					for i=1,depth do
						for j=1,size do				
							local center = inputs[i][j]:narrow(2,96,gsize):narrow(3,96,gsize)
							center:copy(original_inputs[i][j])
						end
					end
					inputs = inputs:cuda()    --used for GPU
				end
				
				if idx == 2 then        --low and medium resolution combine
					inputs = torch.Tensor(2,size,3,224,224):zero()   --the vggnet input 3*224*224
					for i=1,2 do
						for j=1,size do				
							local center = inputs[i][j]:narrow(2,96,gsize):narrow(3,96,gsize)
							center:copy(original_inputs[i][j])
						end
					end
					inputs = inputs:cuda()    --used for GPU
				end
				
				if idx == 3 then      --medium and high resolution combine
					inputs = torch.Tensor(2,size,3,224,224):zero()   --the vggnet input 3*224*224
					for i=2,3 do
						for j=1,size do				
							local center = inputs[i-1][j]:narrow(2,96,gsize):narrow(3,96,gsize)
							center:copy(original_inputs[i][j])
						end
					end
					inputs = inputs:cuda()    --used for GPU
				end

				if idx == 4 then     ---low and high resolution combine
					inputs = torch.Tensor(2,size,3,224,224):zero()   --the vggnet input 3*224*224
					for j=1,size do
						local center1 = inputs[1][j]:narrow(2,96,gsize):narrow(3,96,gsize)
						local center2 = inputs[2][j]:narrow(2,96,gsize):narrow(3,96,gsize)
						center1:copy(original_inputs[1][j])
						center2:copy(original_inputs[3][j])
					end
					inputs = inputs:cuda()    --used for GPU
				end
				----------------------------------------------------------------------------------------

				local outputs = model[idx]:forward(inputs)
				local _, indices = torch.max(outputs, 2)
				local right = indices:eq(targets):sum()
				count = count + right
			end

			return count / vasize
		end

		do
			local last_accuracy = 0
			local decreasing = 0
			for z = 1, max_iter do
				print(string.format('step %d begin...',z))
				local loss = step(index)
				print(string.format('Epoch: %d Current loss: %4f', z, loss))
				local accuracy = eval(index)
				print(string.format('Accuracy on the validation set: %4f', accuracy))
				if accuracy < last_accuracy then
					if decreasing > threshold then break end
					decreasing = decreasing + 1
				else
					decreasing = 0
				end
				last_accuracy = accuracy
			end
		end
	end
end


---------------------------------------
--6.save the model
---------------------------------------
save_path = 'Finetune-VggNet.t7'
torch.save(save_path,s0)





