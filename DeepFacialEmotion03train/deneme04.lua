require 'nn'
require 'optim'
require 'image'

local cv = require 'cv'
require 'cv.imgcodecs' -- reading/writing images
require 'cv.imgproc' -- image processing
require 'cv.highgui' -- GUI
require 'cv.videoio' -- Video input/output
cv.ml = require 'cv.ml' -- Machine Learning


------------------------------------------
-- t7 PAKETLEME
------------------------------------------


emotion01 = 432
emotion02 = 620
emotion02 = 160

-- classes
classes = {'0neutral','1angerofke','2happymutlu'}



trsize = 1212 -- EĞİTİM TOPLAM IMAGE SAYISI
trainData = {
data = torch.Tensor(trsize,3,32,32),  -- PARAMETRE 2 = CHANNEL -> GRAY 1 , RGB 3
labels = torch.Tensor(trsize),
size = function() return trsize end
}


for f=1,432 do
  trainData.data[f] = image.load('0neutral/'..f..'.png') 
  trainData.labels[f] = 1 -- 1 = 0neutral
end
for f=1,620 do
  trainData.data[f+432] = image.load('1angerofke/'..f..'.png') 
  trainData.labels[f+432] = 2 -- 2 = 1angerofke
end
for f=1,160 do
  trainData.data[f+1052] = image.load('2happymutlu/'..f..'.png') 
  trainData.labels[f+1052] = 3 -- 3 = 2happymutlu
end


  --save created dataset:
torch.save('traindeneme.t7',trainData)
   
print('Kaydedildi')

deneme = torch.load('traindeneme.t7')

-- KAYIT SONRASI TEST
print(deneme.data:size())
print(deneme.data:type())
print(deneme.labels[1212])
image.display(deneme.data[1212])




