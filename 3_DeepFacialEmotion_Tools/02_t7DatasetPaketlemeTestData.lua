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
-- t7 PAKETLEME TEST DATASET
------------------------------------------

emotion01 = 10
emotion02 = 10
emotion03 = 10
emotion04 = 10
emotion05 = 10

-- classes
classes = {'1imageNotr','2imageMutlu','3imageUzgun','4imageSaskin','5imageSinirli'}

-- EĞİTİM TOPLAM IMAGE SAYISI
tesize = emotion01 + emotion02 + emotion03 + emotion04 + emotion05 

testData = {
data = torch.Tensor(tesize,3,32,32),
labels = torch.Tensor(tesize),
size = function() return tesize end
}


for f=1,emotion01 do
  testData.data[f] = image.load('MUGdataset5classFaces/1imageNotr/'..f..'.png') 
  testData.labels[f] = 1 -- 1 = 1imageNotr
end
for f=1,emotion02 do
  testData.data[f+emotion01] = image.load('MUGdataset5classFaces/2imageMutlu/'..f..'.png') 
  testData.labels[f+emotion01] = 2 -- 2 = 2imageMutlu
end
for f=1,emotion03 do
  testData.data[f+emotion01+emotion02] = image.load('MUGdataset5classFaces/3imageUzgun/'..f..'.png') 
  testData.labels[f+emotion01+emotion02] = 3 -- 3 = 3imageUzgun
end
for f=1,emotion04 do
  testData.data[f+emotion01+emotion02+emotion03] = image.load('MUGdataset5classFaces/4imageSaskin/'..f..'.png') 
  testData.labels[f+emotion01+emotion02+emotion03] = 4 -- 4 = 4imageSaskin
end
for f=1,emotion05 do
  testData.data[f+emotion01+emotion02+emotion03+emotion04] = image.load('MUGdataset5classFaces/5imageSinirli/'..f..'.png') 
  testData.labels[f+emotion01+emotion02+emotion03+emotion04] = 5 -- 5 = 5imageSinirli
end


  --save created dataset:
torch.save('testDataset.t7',testData)
   
print('Kaydedildi')

deneme = torch.load('testDataset.t7')

-- KAYIT SONRASI TEST
print(deneme.data:size())
print(deneme.data:type())
print(deneme.labels[tesize])
image.display(deneme.data[tesize])

