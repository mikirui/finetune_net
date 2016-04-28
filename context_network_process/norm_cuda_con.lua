local norm_cuda, parent = torch.class("nn.norm_cuda_con", "nn.Module") -- NOTES: return metatables

function norm_cuda:__init()
   self.size = 224
end

-- a bandwidth limited sensor which focuses on a location.
-- locations index the x,y coord of the center of the output glimpse
function norm_cuda:updateOutput(input)
   self.output = input:cuda()
   
   local mean = torch.CudaTensor(input:size())
   mean[{ {}, {1}, {}, {}}] = -123
   mean[{ {}, {2}, {}, {}}] = -117
   mean[{ {}, {3}, {}, {}}] = -104
   self.output:add(mean)
   --print(self.output:size())
   return self.output
end

function norm_cuda:updateGradInput(inputTable, gradOutput)
--   local input = inputTable
end
