-- User Model Test GUI

-- include required packages
require 'xlua'	
require 'qt'
require 'qtwidget'
require 'qtuiloader'
xrequire('nnx',true)
xrequire('camera',true)

require 'optim'
require 'pl'
require 'trepl'
require 'torch'   -- torch
require 'image'   -- to visualize the dataset
require 'nn'      -- provides all sorts of trainable modules/layers

local cv = require 'cv'
require 'cv.objdetect' -- CascadeClassifier 
require 'cv.imgcodecs' -- reading/writing images
require 'cv.imgproc' -- image processing
require 'cv.highgui' -- GUI
require 'cv.videoio' -- Video input/output

cv.ml = require 'cv.ml' -- Machine Learning

local cascade_path = 'GUIandData/haarcascade_frontalface_default.xml'
local face_cascade = cv.CascadeClassifier{filename=cascade_path}

-- setup UI
widget = qtuiloader.load('GUIandData/guiTest.ui')
win1 = qt.QtLuaPainter(widget.frame)
win2 = qt.QtLuaPainter(widget.frame2) -- Frame for Cropped Image
win3 = qt.QtLuaPainter(widget.frame3) -- Frame for Cropped Image YUV

model = torch.load('trainedmodel/model.net') -- Default Model (Dump file, do not remove this file)

classes = {'1Neutral','2Happiness','3Sadness','4Surprise','5Anger'}

confusion = optim.ConfusionMatrix(classes)
criterion = nn.ClassNLLCriterion()

tesize = 1
testData = {
data = torch.Tensor(tesize,3,32,32),
labels = torch.Tensor(tesize),
   size = function() return tesize end
}

normalization = nn.SpatialContrastiveNormalization(1, image.gaussian1D(7))

-- initializing the camera
  local cap = cv.VideoCapture{device=0}
  local _, frame = cap:read{}

function display()
   
   local fx = 0.20  -- rescale factor
   local w = frame:size(2)
   local h = frame:size(1)
   local im2 = cv.resize{src=frame, fx=fx, fy=fx}
   cv.cvtColor{src=im2, dst=im2, code=cv.COLOR_BGR2GRAY}
   local faces = face_cascade:detectMultiScale{image = im2}
   for i=1,faces.size do
      local f = faces.data[i]
      local x = f.x/fx
      local y = f.y/fx
      local w = f.width/fx
      local h = f.height/fx
      cv.rectangle{img=frame, pt1={x, y}, pt2={x + w, y + h}, color={255,0,255,0}, thickness=2}
      cv.putText{
         img=frame,
         text = 'face: ',
         org={x, y-10},
         fontFace=cv.FONT_HERSHEY_PLAIN,
         fontScale=1,
         color={255, 255, 0},
         thickness=1
         }
      
      -- cropping face and save a 32x32 tensor
      crop = cv.getRectSubPix{
        image=frame,
        patchSize={w, h},
        center={x + w/2, y + h/2},
      }
      if crop then
      
        local im = cv.resize{src=crop, dsize={32,32}}:float()
        
        if testwithImage == 1 then
          rgbTensorCroped = image.load(selectedFileName)
          rgbTensorCroped = image.scale(rgbTensorCroped, 32, 32)
        else
          rgbTensorCroped = convertBGRtoRGB(im)
        end
      
        image.display{image = rgbTensorCroped, win = win2, zoom = 1}  -- show cropped image
              
        testData.data[1] = rgbTensorCroped
        
        -- preprocess testSet
        for i = 1,testData:size() do
           -- rgb -> yuv
           local rgb = testData.data[i]
           local yuv = image.rgb2yuv(rgb)
           -- normalize y locally:
           yuv[1] = normalization(yuv[{{1}}])
           testData.data[i] = yuv
        end
        -- normalize u globally:
        mean_u = testData.data[{ {},2,{},{} }]:mean()
        std_u = testData.data[{ {},2,{},{} }]:std()
        testData.data[{ {},2,{},{} }]:add(-mean_u)
        testData.data[{ {},2,{},{} }]:div(-std_u)
        -- normalize v globally:
        mean_v = testData.data[{ {},3,{},{} }]:mean()
        std_v = testData.data[{ {},3,{},{} }]:std()
        testData.data[{ {},3,{},{} }]:add(-mean_v)
        testData.data[{ {},3,{},{} }]:div(-std_v)
        
        image.display{image = testData.data[1], win = win3, zoom = 1}  -- show cropped image YUV
  
        input = testData.data[1]
        pred = model:forward(input)
        pred:exp()
        
        --[[ 
        -- print predictions
        for i=1,pred:size(1) do
          print(classes[i], pred[i])
        end
        ]]--            
      
      end

   end
   
      rgbTensor = convertBGRtoRGB(frame) -- convert image from OpenCV to Torch
      
      image.display{image = rgbTensor, win = win1, zoom = 1}
      cap:read{image=frame}
      
      if crop then
        widget.progressBar:setValue(pred[1]*100);
        widget.progressBar_2:setValue(pred[2]*100);
        widget.progressBar_3:setValue(pred[3]*100);
        widget.progressBar_4:setValue(pred[4]*100);
        widget.progressBar_7:setValue(pred[5]*100);
      end
end

-- BGR to RGB (OpenCV to Torch)
function convertBGRtoRGB(frame)
      forQTimage = frame:transpose(2,3):transpose(1,2)
      local b,g,r = forQTimage[1],forQTimage[2],forQTimage[3]
      local temp = forQTimage[1];
      local rgbTensor = forQTimage:clone()
      rgbTensor[1] = rgbTensor[3]
      rgbTensor[3] = temp
      return rgbTensor
end

-- fotograf button callback
qt.connect(qt.QtLuaListener(widget.btnFoto),
          'sigMousePress(int,int,QByteArray,QByteArray,QByteArray)',
          function ()
            selectedFileName1 = qt.QFileDialog.getOpenFileName(this,'Select Image')
            selectedFileName = qt.QString.tostring(selectedFileName1)
            testwithImage = 1
            testwithWebcam  = 0
            -- print(selectedFileName)
          end);
-- webcam button callback
qt.connect(qt.QtLuaListener(widget.btnCam),
          'sigMousePress(int,int,QByteArray,QByteArray,QByteArray)',
          function ()
            testwithImage = 0
            testwithWebcam  = 1
          end);
-- Model Sec button callback
qt.connect(qt.QtLuaListener(widget.btnModelSec),
          'sigMousePress(int,int,QByteArray,QByteArray,QByteArray)',
          function ()
            modelFileName1 = qt.QFileDialog.getOpenFileName(this,'Select Trained Model')
            -- print(modelFileName1)
            modelName = qt.QString.tostring(modelFileName1)
            model = torch.load(modelName)
          end);

-- timer
timer = qt.QTimer()
timer.interval = 50
timer.singleShot = true
qt.connect(timer,
  'timeout()',
  function() 
    display()
    collectgarbage()
    timer:start()
  end)

widget.windowTitle = 'DeepFED Deep Facial Emotion Detection - User Model Test'
widget:show()
timer:start()
