------------------------------------------------------------------------
--[[ SpatialGlimpse ]]--
-- Ref A.: http://papers.nips.cc/paper/5542-recurrent-models-of-visual-attention.pdf
-- a glimpse is the concatenation of down-scaled cropped images of 
-- increasing scale around a given location in a given image.
-- input is a pair of Tensors: {image, location}
-- locations are x,y coordinates of the center of cropped patches. 
-- Coordinates are between -1,-1 (top-left) and 1,1 (bottom right)
-- output is a batch of glimpses taken in image at location (x,y)
-- size specifies width = height of glimpses
-- depth is number of patches to crop per glimpse (one patch per scale)
-- Each successive patch is scale x size of the previous patch
------------------------------------------------------------------------
local SpatialGlimpse, parent = torch.class("nn.SpatialGlimpse", "nn.Module") -- NOTES: return metatables

function SpatialGlimpse:__init(size, depth, scale)
   require 'nnx'
   self.size = size -- height == width
   self.depth = depth or 3
   self.scale = scale or 2
   
   assert(torch.type(self.size) == 'number')
   assert(torch.type(self.depth) == 'number')
   assert(torch.type(self.scale) == 'number')
   parent.__init(self)
   self.gradInput = {torch.Tensor(), torch.Tensor()}
   if self.scale == 2 then
      self.module = nn.SpatialAveragePooling(2,2,2,2)
   else
      self.module = nn.SpatialReSampling{oheight=size,owidth=size} -- NOTES: performs bilinear resampling of a 3D or 4D input image. Zoom the picture, here zoom out.
   end
   self.modules = {self.module}
end

-- a bandwidth limited sensor which focuses on a location.
-- locations index the x,y coord of the center of the output glimpse
function SpatialGlimpse:updateOutput(inputTable)
   assert(torch.type(inputTable) == 'table')
   assert(#inputTable >= 2)
   local input, location = unpack(inputTable)
   --input, location = self:toBatch(input, 3), self:toBatch(location, 1)-- ???WHAT DOES toBatch MEAN? CAN'T FIND IT IN THE DOC NOW.
   assert(input:dim() == 4 and location:dim() == 2)
   
   self.output:resize(input:size(1), self.depth, input:size(2), self.size, self.size)-- NOTES: batch depth chanel size size?
   
   self._crop = self._crop or self.output.new()
   self._pad = self._pad or input.new() 
   
   for sampleIdx=1,self.output:size(1) do --NOTES: loop a batch
      local outputSample = self.output[sampleIdx]
      local inputSample = input[sampleIdx]
      local xy = location[sampleIdx]
      -- (-1,-1) top left corner, (1,1) bottom right corner of image
      local x, y = xy:select(1,1), xy:select(1,2) --NOTES: select dim1's index 1 and likewise.
      -- (0,0), (1,1)
      x, y = (x+1)/2, (y+1)/2
      
      -- for each depth of glimpse : pad, crop, downscale
      local glimpseSize = self.size
      for depth=1,self.depth do 
         local dst = outputSample[depth]
         if depth > 1 then
            glimpseSize = glimpseSize*self.scale
         end
         
         -- add zero padding (glimpse could be partially out of bounds)
         local padSize = math.floor((glimpseSize-1)/2) -- NOTES: condition that center at the bounds
         self._pad:resize(input:size(2), input:size(3)+padSize*2, input:size(4)+padSize*2):zero() -- NOTES: initialize a zero matrix to store the zero padding image
         local center = self._pad:narrow(2,padSize+1,input:size(3)):narrow(3,padSize+1,input:size(4)) -- NOTES: narrow dim2 from index padSize+1 to padSize+input:size(3), will change the content in the memory
         center:copy(inputSample) --NOTES: since narrow will change the memory, self._pad has now been fulfilled with the corresponding pixil values.
         
         -- crop it
         local h, w = self._pad:size(2)-glimpseSize, self._pad:size(3)-glimpseSize
         local x, y = math.min(h,math.max(0,x*h)),  math.min(w,math.max(0,y*w))-- NOTES: to map from coordinate to pixel space and judge if it's overbounded
         
         if depth == 1 then
            dst:copy(self._pad:narrow(2,x+1,glimpseSize):narrow(3,y+1,glimpseSize))-- NOTES: since it has added zero padding, it can choose the patches centered at the x,y coordinates by simply calling copy(self._pad:narrow(2,x+1,glimpseSize):narrow(3,y+1,glimpseSize))
         else
            self._crop:resize(input:size(2), glimpseSize, glimpseSize)
            self._crop:copy(self._pad:narrow(2,x+1,glimpseSize):narrow(3,y+1,glimpseSize))
         
            if torch.type(self.module) == 'nn.SpatialAveragePooling' then --NOTES: downscale by calling self.module's method
               local poolSize = glimpseSize/self.size
               assert(poolSize % 2 == 0)
               self.module.kW = poolSize
               self.module.kH = poolSize
               self.module.dW = poolSize
               self.module.dH = poolSize
            end
            dst:copy(self.module:updateOutput(self._crop))
         end
      end
   end
   
   self.output:resize(input:size(1), self.depth,input:size(2), self.size, self.size) -- NOTES: batch depth*channel size size
   self.output = self.output:transpose(1,2)
   --self.output = self:fromBatch(self.output, 1)
   return self.output
end

function SpatialGlimpse:updateGradInput(inputTable, gradOutput)
   local input, location = unpack(inputTable)
   local gradInput, gradLocation = unpack(self.gradInput)
   input, location = self:toBatch(input, 3), self:toBatch(location, 1)
   gradOutput = self:toBatch(gradOutput, 3)
   
   gradInput:resizeAs(input):zero()
   gradLocation:resizeAs(location):zero() -- no backprop through location
   
   gradOutput = gradOutput:view(input:size(1), self.depth, input:size(2), self.size, self.size) -- NOTES: view() resize the tensor, if one of the arg is -1, inferred. Can't change the memory.
   
   for sampleIdx=1,gradOutput:size(1) do -- NOTES: from 1 to batchSize
      local gradOutputSample = gradOutput[sampleIdx]
      local gradInputSample = gradInput[sampleIdx]
      local xy = location[sampleIdx] -- height, width
      -- (-1,-1) top left corner, (1,1) bottom right corner of image
      local x, y = xy:select(1,1), xy:select(1,2)
      -- (0,0), (1,1)
      x, y = (x+1)/2, (y+1)/2
      
      -- for each depth of glimpse : pad, crop, downscale?
      local glimpseSize = self.size
      for depth=1,self.depth do 
         local src = gradOutputSample[depth]
         if depth > 1 then
            glimpseSize = glimpseSize*self.scale
         end
         
         -- add zero padding (glimpse could be partially out of bounds)
         local padSize = math.floor((glimpseSize-1)/2)
         self._pad:resize(input:size(2), input:size(3)+padSize*2, input:size(4)+padSize*2):zero()
         
         -- crop it
         local h, w = self._pad:size(2)-glimpseSize, self._pad:size(3)-glimpseSize
         local x, y = math.min(h,math.max(0,x*h)),  math.min(w,math.max(0,y*w))
         local pad = self._pad:narrow(2, x+1, glimpseSize):narrow(3, y+1, glimpseSize)
         
         -- upscale glimpse for different depths
         if depth == 1 then
            pad:copy(src)
         else
            self._crop:resize(input:size(2), glimpseSize, glimpseSize)
            
            if torch.type(self.module) == 'nn.SpatialAveragePooling' then
               local poolSize = glimpseSize/self.size
               assert(poolSize % 2 == 0)
               self.module.kW = poolSize
               self.module.kH = poolSize
               self.module.dW = poolSize
               self.module.dH = poolSize
            end
            
            pad:copy(self.module:updateGradInput(self._crop, src))
         end
        
         -- copy into gradInput tensor (excluding padding)
         gradInputSample:add(self._pad:narrow(2, padSize+1, input:size(3)):narrow(3, padSize+1, input:size(4)))
      end
   end
   
   self.gradInput[1] = self:fromBatch(gradInput, 1) --NOTES: WHAT DOES FROMBATCH METHOD'S ARGUMENTS MEAN? 1??
   self.gradInput[2] = self:fromBatch(gradLocation, 1)
   
   return self.gradInput
end
