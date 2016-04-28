require 'torch'
require 'nn'
require 'inn'
require 'nn'
require 'cutorch'
require 'cunn'
require 'cudnn'
require 'hdf5'
require 'dp'
require 'dpnn'
require 'optim'
require 'norm_cuda'
require 'norm_cuda_con'

--h5_file_path = '/home/qianlima/ylh_test/resize_img/cocotalk384.h5'
--ft_matrix_path = '/home/qianlima/ylh_test/resize_img/cocotalk384_ft_matrix.t7'
h5_file_path = '/home/qianlima/ylh_test/process_flickr/flickr8ktalk384.h5'
ft_matrix_path = '/home/qianlima/ylh_test/process_flickr/flickr8ktalk384_ft_matrix.t7'
--h5_file_path = '/home/qianlima/ylh_test/process_flickr/flickr30k/flickr30ktalk384.h5'
--ft_matrix_path = '/home/qianlima/ylh_test/process_flickr/flickr30k/flickr30ktalk384_ft_matrix.t7'
model_path = '/home/qianlima/torch-test/raw_googlenet.t7'

---1.read h5 file to get image file information---
h5_file = hdf5.open(h5_file_path, 'r')
local file_size = h5_file:read('/images'):dataspaceSize()
local num_images = file_size[1]
local num_channels = file_size[2]
local image_size = file_size[3]
local batch_size = 30
local cur = 1
ft_matrix = torch.CudaTensor(num_images,1024):fill(0)   --feature matrix,each images 1024 features


---2.define the model(sample & googlenet)------
cnn = torch.load(model_path)
model = nn.Sequential()
model:add(nn.norm_cuda_con())    --norm_cuda
model:add(cudnn.SpatialAveragePooling(4,4,4,4))    --SpatialAveragePooling
model:add(cnn)    --googlenet

model = model:cuda()
model:evaluate()

--output = model:forward(input:cuda())


---3.main loop to process the data------
for i = 1,num_images,batch_size
do
	ed = math.min(cur + batch_size - 1,num_images)
	local imgs = h5_file:read('/images'):partial({cur,ed},{1,num_channels},
                            {1,image_size},{1,image_size})
	ft = model:forward(imgs:cuda())    ---get the feature through googlenet
	ft_matrix[{{cur,ed}}] = ft
	print(string.format('solved images from %d to %d\n',cur,ed))
	cur = ed + 1
end

torch.save(ft_matrix_path,ft_matrix)




	






















