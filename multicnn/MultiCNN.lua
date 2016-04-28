require 'nn'
require 'cudnn'
require 'inn'
require 'dp'
require 'dpnn'
require 'norm_cuda2'

googlenet_path = '/home/qianlima/torch-test/raw_googlenet.t7'

local layer, parent = torch.class('nn.MultiCNN', 'nn.Module')

function layer:__init(opt)
	print(opt)
	parent.__init(self)
	self.comb = opt.comb or 3        --the number of googlenet to combine
	self.cnn_path = opt.cnn_path or googlenet_path
	self.cnn_learningrate = opt.cnn_learningrate or 0    --to decide train cnn or not
	self.linear_learningrate = opt.linear_learningrate or 0.1
	self:build_model()
end

function layer:build_model()
	self.model = nn.Sequential()
	self.norm = nn.norm_cuda2()
	self.core = nn.Parallel(1,2)
	self.cnn = torch.load(self.cnn_path)
	self.linear = nn.Linear(1024*self.comb, 2048)
	if(self.comb == 3) then
		self.core:add(self.cnn)
		self.core:add(self.cnn)
		self.core:add(self.cnn)
	elseif(self.comb == 2) then
		self.core:add(self.cnn)
		self.core:add(self.cnn)
	elseif(self.comb == 1) then
		self.core:add(self.cnn)
	end
	self.model:add(self.norm)
	self.model:add(self.core)
	self.model:add(self.linear)
	self.model = self.model:cuda()
	if(self.cnn_learningrate == 0) then
		self.core.accGradParameters = function() return end
	end
end

function layer:updateOutput(input)
	print(input:size())
	self.output = self.model:forward(input)
	print(self.output:size())
	return self.output
end

function layer:accGradParameters(input, gradOutput, lr)
	if(self.cnn_learningrate ~= 0) then
		self.core:accGradParameters(self.norm.output,self.core.output,self.cnn_learningrate)
	end
	self.linear:accGradParameters(self.core.output,self.linear.output,self.linear_learningrate)
end

function layer:updateGradInput(input,gradOutput)
	return self.model:updateGradInput(input,gradOutput)
end

function layer:parameters()
	return self.model:parameters()
end

function layer:accUpdateGradParameters(input, gradOutput, lr)
	self.model:accUpdateGradParameters(input, gradOutput, lr)
end

function layer:zeroGradParameters()
	self.model:zeroGradParameters()
end

function layer:updateParameters(learningRate)
	if(self.cnn_learningrate ~= 0) then
		self.core:updateParameters(self.cnn_learningrate)
	end
	self.linear:updateParameters(self.linear_learningrate)
end

function layer:training()
	self.model.train = true
end

function layer:evaluate()
	self.model.train = false
end

function layer:cuda(...)
	return self.model:cuda()
end

function layer:getParameters()
	return self.model:getParameters()
end

function layer:listModules()
	return self.model:listModules()
end
