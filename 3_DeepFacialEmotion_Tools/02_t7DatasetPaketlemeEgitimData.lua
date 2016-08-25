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
-- t7 PAKETLEME EGITIM DATASET
------------------------------------------

emotion01 = 1463
emotion02 = 1410
emotion03 = 1193
emotion04 = 1134
emotion05 = 1102

-- classes
classes = {'1imageNotr','2imageMutlu','3imageUzgun','4imageSaskin','5imageSinirli'}

-- EĞİTİM TOPLAM IMAGE SAYISI
trsize = emotion01 + emotion02 + emotion03 + emotion04 + emotion05 

trainData = {
data = torch.Tensor(trsize,3,32,32),
labels = torch.Tensor(trsize),
size = function() return trsize end
}


for f=1,emotion01 do
  trainData.data[f] = image.load('MUGdataset5classFaces/1imageNotr/'..f..'.png') 
  trainData.labels[f] = 1 -- 1 = 1imageNotr
end
for f=1,emotion02 do
  trainData.data[f+emotion01] = image.load('MUGdataset5classFaces/2imageMutlu/'..f..'.png') 
  trainData.labels[f+emotion01] = 2 -- 2 = 2imageMutlu
end
for f=1,emotion03 do
  trainData.data[f+emotion01+emotion02] = image.load('MUGdataset5classFaces/3imageUzgun/'..f..'.png') 
  trainData.labels[f+emotion01+emotion02] = 3 -- 3 = 3imageUzgun
end
for f=1,emotion04 do
  trainData.data[f+emotion01+emotion02+emotion03] = image.load('MUGdataset5classFaces/4imageSaskin/'..f..'.png') 
  trainData.labels[f+emotion01+emotion02+emotion03] = 4 -- 4 = 4imageSaskin
end
for f=1,emotion05 do
  trainData.data[f+emotion01+emotion02+emotion03+emotion04] = image.load('MUGdataset5classFaces/5imageSinirli/'..f..'.png') 
  trainData.labels[f+emotion01+emotion02+emotion03+emotion04] = 5 -- 5 = 5imageSinirli
end

  --save created dataset:
torch.save('trainDataset.t7',trainData)
   
print('Kaydedildi')

deneme = torch.load('trainDataset.t7')

-- KAYIT SONRASI TEST
print(deneme.data:size())
print(deneme.data:type())
print(deneme.labels[trsize])
image.display(deneme.data[trsize])

