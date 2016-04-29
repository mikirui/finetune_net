collectgarbage()
require 'torch'
require 'dp'
require 'SpatialGlimpse2'
require 'inn'
require 'nn'
require 'cutorch'
require 'cunn'
require 'cudnn'
require 'dpnn'
require 'optim'
require 'norm_cuda'

cudnn.benchmark = true
cutorch.manualSeed(123)
cutorch.setDevice(1)

cmd = torch.CmdLine()
cmd:text()
cmd:text('Finetune-GoogleNet using ImageNet')
cmd:text('Example:')
cmd:text('$> th finetune-googlenet-imagenet.lua --batchSize 128 --momentum 0.5')
cmd:text('Options:')
cmd:option('--dataPath',paths.concat(dp.DATA_DIR,'ImageNet'),'path to ImageNet')
cmd:option('--trainPath', '', 'Path to train set. Defaults to --dataPath/ILSVRC2012_img_train')
cmd:option('--validPath', '', 'Path to valid set. Defaults to --dataPath/ILSVRC2012_img_val')
cmd:option('--metaPath', '', 'Path to metadata. Defaults to --dataPath/metadata')
cmd:option('--overwrite', false, 'overwrite the cache (useful for debugging the ImageNet DataSource')
cmd:option('--learningRate1', 0.01, 'learning rate at t=0')
cmd:option('--learningRate2', 0.1, 'learning rate at t =0')
cmd:option('--schedule1', '{[1]=1e-2,[19]=5e-3,[30]=1e-3,[44]=5e-4,[53]=1e-4', 'learning rate schedule1')
cmd:option('--schedule2', '{[1]=1e-1,[19]=5e-2,[30]=1e-2,[44]=5e-3,[53]=1e-3}', 'learning rate schedule2')
cmd:option('--maxOutNorm', -1, 'max norm each layers output neuron weights')
cmd:option('--weightDecay', 5e-4, 'weight decay') 
cmd:option('--momentum', 0.9, 'momentum') 
cmd:option('--batchSize', 30, 'number of examples per batch')
cmd:option('--trainEpochSize', -1, 'number of train examples seen between each epoch')
cmd:option('--maxEpoch', 100, 'maximum number of epochs to run')
cmd:option('--maxTries', 30, 'maximum number of epochs to try to find a better local minima for early-stopping')
cmd:option('--accUpdate', false, 'accumulate gradients inplace')
cmd:option('--verbose', false, 'print verbose messages')
cmd:option('--progress', true, 'print progress bar')
cmd:option('--nThread', 2, 'allocate threads for loading images from disk. Requires threads-ffi.')
cmd:option('--LCN', false, 'use Local Constrast Normalization as in the original paper. Requires inn (imagine-nn)')
cmd:option('--resume',false,'continue experiment on a saved model')
cmd:text()

opt = cmd:parse(arg or {})

opt.trainPath = (opt.trainPath == '') and paths.concat(opt.dataPath, 'trainsubset') or opt.trainPath
opt.validPath = (opt.validPath == '') and paths.concat(opt.dataPath, 'valsubset') or opt.validPath
opt.metaPath = (opt.metaPath == '') and paths.concat(opt.dataPath, 'metadata') or opt.metaPath
table.print(opt)


if opt.LCN then
   assert(opt.cuda, "LCN only works with CUDA")
   require "inn"
end

--[[data]]--
ds = dp.ImageNet{
   train_path=opt.trainPath, valid_path=opt.validPath, 
   meta_path=opt.metaPath, verbose=opt.verbose,
   cache_mode = opt.overwrite and 'overwrite' or nil,
   load_size = {3,500,500},
   sample_size = {3,384,384}
}

-- preprocessing function 
--ppf = ds:normalizePPF()


--[[Model]]--
--load the googlenet--
collectgarbage()
gsize = 96   --glimpse size

--if opt.resume then
--	print('loading the saved experiment')
--	model = torch.load('F-G-I_stride.t7')
if opt.resume then
	print('loading the saved experiment for change lr')
	exp = torch.load('/home/qianlima/save/Exia:1460821117:1.dat')
	model = exp._model
else
--combine the new models
d0 = torch.load('one.t7')
--first combination
model = nn.Sequential()   --input: batch*channel*wi*hi
model:add(nn.norm_cuda())
d = cudnn.SpatialAveragePooling(4,4,4,4)
d.accGradParameters = function() return end
model:add(d)
--core = nn.Sequential()
--d0.accGradParameters = function() return end
--core = nn.Sequential()
--core:add(d0)
--core:add(d0)
--core:add(nn.NaN(nn.Linear(2496,4096)))
d1 = nn.NaN(nn.Linear(832,1000))
--d2 = nn.NaN(nn.Linear(4096,1000))
--d2 = nn.NaN(nn.LogSoftMax())
--d1:add(nn.NaN(nn.Linear(7488,4096)))
--d1:add(nn.NaN(nn.Linear(4096,1000)))
--d1:add(nn.NaN(nn.LogSoftMax()))
model:add(d0)
model:add(d1)
--model:add(d2)
model:add(nn.NaN(nn.LogSoftMax()))
model = model:cuda()
collectgarbage()
end

--[[Propagators]]--
train = dp.Optimizer{
	acc_update = opt.accUpdate,
	loss = nn.ModuleCriterion(nn.ClassNLLCriterion(),nil,nn.Convert()),
	callback = function(model,report)
		opt.learningRate1 = opt.schedule1[report.epoch] or opt.learningRate1
		opt.learningRate2 = opt.schedule2[report.epoch] or opt.learningRate2
		if opt.accUpdate then
		--	model:accUpdateGradParameters(model.dpnn_input, model.output, opt.learningRate2)
			d0:accUpdateGradParameters(d0.dpnn_input, d0.output, opt.learningRate1)
			d1:accUpdateGradParameters(d1.dpnn_input, d1.output, opt.learningRate2)
		--	d2:accUpdateGradParameters(d2.dpnn_input, d2.output, opt.learningRate2)
		--	d2:accUpdateGradParameters(d3.dpnn_input, d3.output, opt.learningRate2)
		else
			model:updateGradParameters(opt.momentum)
			--d0:updateGradParameters(opt.momentum)
			--d1:updateGradParameters(opt.momentum)
			model:weightDecay(opt.weightDecay)
			--d2:weightDecay(opt.weightDecay)
			--d3:weightDecay(opt.weightDecay)
		--	model:updateParameters(opt.learningRate2)
			d0:updateParameters(opt.learningRate1)
			d1:updateParameters(opt.learningRate2)
		--	d2:updateParameters(opt.learningRate2)
		end
		model:maxParamNorm(opt.maxOutNorm)
		--d2:maxParamNorm(opt.maxOutNorm)
		--d3:maxParamNorm(opt.maxOutNorm)
		model:zeroGradParameters()
	end,
	feedback = dp.Confusion(),
   	sampler = dp.RandomSampler{
      		batch_size=opt.batchSize, epoch_size=opt.trainEpochSize   	},
	progress = opt.progress
}

valid = dp.Evaluator{
	feedback = dp.TopCrop{n_top={1,5,10},n_crop=10,center=2}, 
	sampler = dp.Sampler{
		batch_size=math.round(opt.batchSize/10)
	}
}

--[[Experiment]]--

xp = dp.Experiment{
   model = model,
   optimizer = train,
   validator = valid,
   observer = {
      dp.FileLogger(),
      dp.EarlyStopper{
         error_report = {'validator','feedback','topcrop','all',5},
         maximize = true,
         max_epochs = opt.maxTries
      }
   },
   random_seed = os.time(),
   max_epoch = opt.maxEpoch
}
--xp = torch.load('/home/qianlima/save/Exia:1460821117:1.dat')
xp:cuda()


print"Model :"
print(model)
--print "last optimizer:"
--print(xp:report().optimizer)
--print(xp:report().validator)


save_path1 = '/home/qianlima/torch-test/one_cut_norm.t7'
xp:run(ds,save_path1)

--save the model--
--save_path = 'context_network4.t7'
--torch.save(save_path,core)	

----save !!!----
torch.save(save_path1,model)


