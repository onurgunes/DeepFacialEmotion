
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

toplamImage = 620

trainData = {
data = torch.Tensor(toplamImage,3,32,32),
size = function() return toplamImage end
}


for f=0,(toplamImage-1) do
  trainData.data[f+1][1] = image.load('1angerofke/'..f..'.png') 
  trainData.data[f+1][2] = image.load('1angerofke/'..f..'.png') 
  trainData.data[f+1][3] = image.load('1angerofke/'..f..'.png') 
  image.save('1angerofke2/'..f..'.png', trainData.data[f+1])   -- KAYIT
end

print('tamamlandı...')







