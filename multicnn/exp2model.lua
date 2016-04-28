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

--[[
exp_path = '/home/qianlima/save/Exia:1461484089:1.dat'

---1.define the model(sample & googlenet)------
exp = torch.load(exp_path)
whole = exp._model
seq = whole:get(1)
model = seq:get(3)
v1 = model:get(1)
local w = v1.weight:clone()
-- swap weights to R and B channels
print('converting first layer conv filters from BGR to RGB...')
v1.weight[{ {}, 1, {}, {} }]:copy(w[{ {}, 3, {}, {} }])
v1.weight[{ {}, 3, {}, {} }]:copy(w[{ {}, 1, {}, {} }])

torch.save('raw_googlenet.t7',model)

--]]

exp_path = '/home/qianlima/save/Exia:1461513056:1.dat'

---1.define the model(sample & googlenet)------
exp = torch.load(exp_path)
whole = exp._model
seq = whole:get(1)
model = seq:get(3)
v1 = model:get(1)
local w = v1.weight:clone()
-- swap weights to R and B channels
print('converting first layer conv filters from BGR to RGB...')
v1.weight[{ {}, 1, {}, {} }]:copy(w[{ {}, 3, {}, {} }])
v1.weight[{ {}, 3, {}, {} }]:copy(w[{ {}, 1, {}, {} }])

torch.save('raw_googlenet_cut.t7',model)
