require 'torch'
require 'dp'
require 'SpatialGlimpse2'
require 'inn'
require 'nn'
require 'cutorch'
require 'cunn'
require 'cudnn'
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
cmd:option('--learningRate', 0.01, 'learning rate at t=0')
cmd:option('--schedule', '{[1]=1e-2,[19]=5e-3,[30]=1e-3,[44]=5e-4,[53]=1e-4}', 'learning rate schedule')
cmd:option('--maxOutNorm', -1, 'max norm each layers output neuron weights')
cmd:option('--weightDecay', 5e-4, 'weight decay') 
cmd:option('--momentum', 0.9, 'momentum') 
cmd:option('--batchSize', 10, 'number of examples per batch')
cmd:option('--trainEpochSize', -1, 'number of train examples seen between each epoch')
cmd:option('--maxEpoch', 2, 'maximum number of epochs to run')
cmd:option('--maxTries', 2, 'maximum number of epochs to try to find a better local minima for early-stopping')
cmd:option('--accUpdate', false, 'accumulate gradients inplace')
cmd:option('--verbose', false, 'print verbose messages')
cmd:option('--progress', true, 'print progress bar')
cmd:option('--nThread', 2, 'allocate threads for loading images from disk. Requires threads-ffi.')
cmd:option('--LCN', false, 'use Local Constrast Normalization as in the original paper. Requires inn (imagine-nn)')
cmd:text()
opt = cmd:parse(arg or {})

opt.trainPath = (opt.trainPath == '') and paths.concat(opt.dataPath, 'ILSVRC2012_img_train') or opt.trainPath
opt.validPath = (opt.validPath == '') and paths.concat(opt.dataPath, 'ILSVRC2012_img_val') or opt.validPath
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
   cache_mode = opt.overwrite and 'overwrite' or nil
}

-- preprocessing function 
ppf = ds:normalizePPF()


--[[Model]]--
--load the googlenet--
cnn_tower1 = torch.load('googlenet.t7')
collectgarbage()
gsize = 96   --glimpse size

--combine the new models

--first combination
model = nn.Sequential()   --input: batch*channel*wi*hi
model:add(nn.SpatialGlimpse2(gsize))
core = nn.Sequential()
d0 = nn.Parallel(1,2)   --input:depth*batch*channel*gsize*gsize
d0:add(cnn_tower1)
d0:add(cnn_tower1)
d0:add(cnn_tower1)
core:add(d0)
core:add(nn.Linear(89856,4096))
model:add(core)
model:add(nn.Linear(4096,1000))
model:add(nn.LogSoftMax())
model = model:cuda()
collectgarbage()

--[[Propagators]]--
train = dp.Optimizer{
	acc_update = opt.accUpdate,
	loss = nn.ModuleCriterion(nn.ClassNLLCriterion(),nil,nn.Convert()),
	callback = function(model,report)
		opt.learningRate = opt.schedule[report.epoch] or opt.learningRate
		if opt.accUpdate then
			model:accUpdateGradParameters(model.dpnn_input, model.output, opt.learningRate)
		else
			model:updateGradParameters(opt.momentum)
			model:weightDecay(opt.weightDecay)
			model:updateParameters(opt.learningRate)
		end
		model:maxParamNorm(opt.maxOutNorm)
		model:zeroGradParameters()
	end,
	feedback = dp.Confusion(),
   	sampler = dp.RandomSampler{
      		batch_size=opt.batchSize, epoch_size=opt.trainEpochSize, ppf=ppf
   	},
	progress = opt.progress
}

valid = dp.Evaluator{
	feedback = dp.TopCrop{n_top={1,5,10},n_crop=10,center=2}, 
	sampler = dp.Sampler{
		batch_size=math.round(opt.batchSize/10), ppf=ppf
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

xp:cuda()

print"Model :"
print(model)

xp:run(ds)

--save the model--
save_path = 'Finetune-GoogleNet-ImageNet.t7'
torch.save(save_path,core)	




