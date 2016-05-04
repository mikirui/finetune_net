require 'torch'
require 'nn'
require 'hdf5'
require 'nnx'

--h5_file_path = '/home/qianlima/torch-test/DRAM/dataset/MSCOCO/coco256/cocotalk256.h5'
--ft_matrix_path = '/home/qianlima/torch-test/DRAM/dataset/MSCOCO/coco256/cocotalk256_ft_matrix.t7'
h5_file_path = '/home/qianlima/torch-test/DRAM/dataset/Flickr8k/f8k256/flickr8ktalk256.h5'
ft_matrix_path = '/home/qianlima/torch-test/DRAM/dataset/Flickr8k/f8k256/flickr8ktalk256_ft_matrix.t7'
--h5_file_path = '/home/qianlima/torch-test/DRAM/dataset/Flickr30k/f30k256/flickr30ktalk256.h5'
--ft_matrix_path = '/home/qianlima/torch-test/DRAM/dataset/Flickr30k/f30k256/flickr30ktalk256_ft_matrix.t7'
model_path = '/home/qianlima/torch-test/raw_googlenet.t7'

---1.read h5 file to get image file information---
h5_file = hdf5.open(h5_file_path, 'r')
local file_size = h5_file:read('/images'):dataspaceSize()
local num_images = file_size[1]
local num_channels = file_size[2]
local image_size = file_size[3]
local batch_size = 30
local cur = 1

---2.define the model(sample & googlenet)------
res = nn.SpatialReSampling{oheight=96,owidth=96}   --SpatialAveragePooling

require 'inn'
require 'cutorch'
require 'cunn'
require 'cudnn'
require 'dp'
require 'dpnn'
require 'optim'
require 'norm_cuda'
require 'norm_cuda_con'

cnn = torch.load(model_path)
model = nn.Sequential()
model:add(nn.norm_cuda_con())    --norm_cuda
model:add(cnn)    --googlenet

model = model:cuda()
model:evaluate()

ft_matrix = torch.CudaTensor(num_images,1024):fill(0)

---3.main loop to process the data------
for i = 1,num_images,batch_size
do
	ed = math.min(cur + batch_size - 1,num_images)
	local imgs = h5_file:read('/images'):partial({cur,ed},{1,num_channels},
                            {1,image_size},{1,image_size})
	sa = res:forward(imgs:double())             --resize to 96*96
	ft = model:forward(sa:cuda())    ---get the feature through googlenet
	ft_matrix[{{cur,ed}}] = ft
	print(string.format('solved images from %d to %d\n',cur,ed))
	cur = ed + 1
end

torch.save(ft_matrix_path,ft_matrix)
