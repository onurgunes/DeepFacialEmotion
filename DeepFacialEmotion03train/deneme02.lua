require 'nn'
require 'optim'
require 'image'

local cv = require 'cv'
require 'cv.imgcodecs' -- reading/writing images
require 'cv.imgproc' -- image processing
require 'cv.highgui' -- GUI
require 'cv.videoio' -- Video input/output

-- cv.ml and cv.flann return a separate table,
-- while other submodules just update 'cv' table
cv.ml = require 'cv.ml' -- Machine Learning


-- MODEL VE DATASET İŞLEMLERİ TAMAMLANMIŞ


trsize = 6

trainData = {
data = torch.Tensor(trsize,3,32,32),  -- PARAMETRE 2 = CHANNEL -> GRAY 1 , RGB 3
labels = torch.Tensor(trsize),
size = function() return trsize end
}

-- print(trainData.data[1]:size())


-- classes: GLOBAL var!
classes = {'face','asd'}

confusion = optim.ConfusionMatrix(classes)

-- print(confusion)



for f=0,5 do
  trainData.data[f+1][1] = image.load('0neutral/'..f..'.png') 
  trainData.data[f+1][2] = image.load('0neutral/'..f..'.png') 
  trainData.data[f+1][3] = image.load('0neutral/'..f..'.png') 
  trainData.labels[f+1] = 1 -- 1 = faces
end

a = image.drawText(trainData.data[5], "no", 1, 1,{color = {0, 0, 255}, size = 2})
-- image.display(a)

normalization = nn.SpatialContrastiveNormalization(1, image.gaussian1D(7))
for i = 1,trainData:size() do
   -- rgb -> yuv
   local rgb = trainData.data[i]
   local yuv = image.rgb2yuv(rgb)
   -- normalize y locally:
   yuv[1] = normalization(yuv[{{1}}])
   trainData.data[i] = yuv
end


a = image.drawText(trainData.data[5], "y", 1, 1,{color = {0, 0, 255}, size = 2})
-- image.display(a)


-- normalize u globally:
mean_u = trainData.data[{ {},2,{},{} }]:mean()
std_u = trainData.data[{ {},2,{},{} }]:std()
trainData.data[{ {},2,{},{} }]:add(-mean_u)
trainData.data[{ {},2,{},{} }]:div(-std_u)


a = image.drawText(trainData.data[5], "u", 1, 1,{color = {0, 0, 255}, size = 2})
-- image.display(a)



-- normalize v globally:
mean_v = trainData.data[{ {},3,{},{} }]:mean()
std_v = trainData.data[{ {},3,{},{} }]:std()
trainData.data[{ {},3,{},{} }]:add(-mean_v)
trainData.data[{ {},3,{},{} }]:div(-std_v)

a = image.drawText(trainData.data[5], "v", 1, 1,{color = {0, 0, 255}, size = 2})
-- image.display(a)






 -- cv.cvtColor{src=rgb, dst=rgb, code=cv.COLOR_GRAY2RGB}




