
require 'nn'
require 'optim'
require 'image'
local cv = require 'cv'
require 'cv.imgcodecs' -- reading/writing images
require 'cv.imgproc' -- image processing
require 'cv.highgui' -- GUI
require 'cv.videoio' -- Video input/output
cv.ml = require 'cv.ml' -- Machine Learning


-- IMAGE CHANNEL İŞLEMLERİ

toplamImage = 2

trainData = {
data = torch.Tensor(toplamImage,3,32,32),
size = function() return toplamImage end
}


for f=0,(toplamImage-1) do
  trainData.data[f+1][1] = image.load('testImages/'..f..'.png') 
  trainData.data[f+1][2] = image.load('testImages/'..f..'.png') 
  trainData.data[f+1][3] = image.load('testImages/'..f..'.png') 
  image.save('testImages/'..f..'.png', trainData.data[f+1])   -- KAYIT
end

print('tamamlandı...')







