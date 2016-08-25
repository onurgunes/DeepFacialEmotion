
require 'image'
local cv = require 'cv'
require 'cv.imgcodecs' -- reading/writing images
require 'cv.imgproc' -- image processing
require 'cv.highgui' -- GUI


-- IMAGE CHANNEL İŞLEMLERİ (1 Channel to 3 Channel)
-- (gri foto hataları için gerekli)

toplamImage1 = 432
toplamImage2 = 620
toplamImage3 = 160
toplamImage = toplamImage1 + toplamImage2 + toplamImage3
trainData = {
data = torch.Tensor(toplamImage,3,32,32),
size = function() return toplamImage end
}

for f=1,(toplamImage1) do
  trainData.data[f][1] = image.load('0neutral/'..f..'.png') 
  trainData.data[f][2] = image.load('0neutral/'..f..'.png') 
  trainData.data[f][3] = image.load('0neutral/'..f..'.png') 
  image.save('0neutral/'..f..'.png', trainData.data[f])   -- KAYIT
end
for f=1,(toplamImage2) do
  trainData.data[f][1] = image.load('1angerofke/'..f..'.png') 
  trainData.data[f][2] = image.load('1angerofke/'..f..'.png') 
  trainData.data[f][3] = image.load('1angerofke/'..f..'.png') 
  image.save('1angerofke/'..f..'.png', trainData.data[f])   -- KAYIT
end
for f=1,(toplamImage3) do
  trainData.data[f][1] = image.load('2happymutlu/'..f..'.png') 
  trainData.data[f][2] = image.load('2happymutlu/'..f..'.png') 
  trainData.data[f][3] = image.load('2happymutlu/'..f..'.png') 
  image.save('2happymutlu/'..f..'.png', trainData.data[f])   -- KAYIT
end

print('tamamlandı...')







