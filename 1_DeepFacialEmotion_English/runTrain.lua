--- User Screen: Create Image & Package & Train

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
widget = qtuiloader.load('GUIandData/guiTrain.ui')
win1 = qt.QtLuaPainter(widget.frame)
win2 = qt.QtLuaPainter(widget.frame2) -- show cropped image frame
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
         
      -- crop : 32x32 cropped face tensor
      crop = cv.getRectSubPix{
        image=frame,
        patchSize={w, h},
        center={x + w/2, y + h/2},
      }
      if crop then
          local im = cv.resize{src=crop, dsize={32,32}}:float()
          imageCroppedCV = im
          rgbTensorCroped = convertBGRtoRGB(im)
          image.display{image = rgbTensorCroped, win = win2, zoom = 1}  -- show crop
      end
   end
   rgbTensor = convertBGRtoRGB(frame) -- OpenCV to Torch tensor
   image.display{image = rgbTensor, win = win1, zoom = 1}
   cap:read{image=frame}  
end

-- Functions
-----------------------------------------------------------------------
-- BGR to RGB Function
function convertBGRtoRGB(frame)
      forQTimage = frame:transpose(2,3):transpose(1,2) -- OpenCV to Torch tensor
      local b,g,r = forQTimage[1],forQTimage[2],forQTimage[3]
      local temp = forQTimage[1];
      local rgbTensor = forQTimage:clone()
      rgbTensor[1] = rgbTensor[3]
      rgbTensor[3] = temp
      return rgbTensor
end
-----------------------------------------------------------------------
-- Create Path Function
function createImagePath()
      os.execute('mkdir -p ' .. 'images')
      os.execute('mkdir -p ' .. 'images/1imageNotr')
      os.execute('mkdir -p ' .. 'images/2imageMutlu')
      os.execute('mkdir -p ' .. 'images/3imageUzgun')
      os.execute('mkdir -p ' .. 'images/4imageSaskin')
      os.execute('mkdir -p ' .. 'images/5imageSinirli')
      print('Created Path...')
end
-----------------------------------------------------------------------
-- PACKAGE Function
function paketle(im1, im2, im3, im4, im5)
      totalTrainImages = im1 + im2 + im3 + im4 + im5
      -- classes
      classes = {'1imageNotr','2imageMutlu','3imageUzgun','4imageSaskin','5imageSinirli'}
      trsize = totalTrainImages -- Train total image number
      trainData = {
      data = torch.Tensor(trsize,3,32,32),
      labels = torch.Tensor(trsize),
      size = function() return trsize end
      }
      for f=1,im1 do
        trainData.data[f] = image.load('images/1imageNotr/'..f..'.png') 
        trainData.labels[f] = 1 -- 1 = 1imageNotr
      end
      for f=1,im2 do
        trainData.data[f+im1] = image.load('images/2imageMutlu/'..f..'.png') 
        trainData.labels[f+im1] = 2 -- 2 = 2imageMutlu
      end
      for f=1,im3 do
        trainData.data[f+im1+im2] = image.load('images/3imageUzgun/'..f..'.png')         
        trainData.labels[f+im1+im2] = 3 -- 3 = 3imageUzgun
      end
      for f=1,im4 do
        trainData.data[f+im1+im2+im3] = image.load('images/4imageSaskin/'..f..'.png')
        trainData.labels[f+im1+im2+im3] = 4 -- 4 = 4imageSaskin
      end
      for f=1,im5 do
        trainData.data[f+im1+im2+im3+im4] = image.load('images/5imageSinirli/'..f..'.png') 
        trainData.labels[f+im1+im2+im3+im4] = 5 -- 5 = 5imageSinirli
      end
        --save created dataset:
      torch.save('trainDataset.t7',trainData)
      print('Saved')
      deneme = torch.load('trainDataset.t7')
      
      -- Test after save
      -- print(deneme.data:size())
      -- print(deneme.data:type())
      -- print(deneme.labels[totalTrainImages])
      -- image.display(deneme.data[totalTrainImages])

end
-----------------------------------------------------------------------

-- Default Values
imageNotr    = 0 
imageMutlu   = 0
imageUzgun   = 0
imageSaskin  = 0
imageSinirli = 0


-- Callbacks 
-----------------------------------------------------------------------

-- 1 button callback 
qt.connect(qt.QtLuaListener(widget.btnChange1),
          'sigMousePress(int,int,QByteArray,QByteArray,QByteArray)',
          function ()
		        widget.btnChange1.enabled = false  -- error block for quick click
		        createImagePath()
		        imageNotr = imageNotr + 1
            pathimagesave = "images/1imageNotr/"..imageNotr..".png"
            cv.imwrite{pathimagesave, imageCroppedCV}
            widget.lineEdit1.text = imageNotr
            widget.btnChange1.enabled = true   -- error block for quick click
            widget.labelProcess.text = "Saved image..."
          end);
-- 2 button callback 
qt.connect(qt.QtLuaListener(widget.btnChange2),
          'sigMousePress(int,int,QByteArray,QByteArray,QByteArray)',
          function ()
            widget.btnChange2.enabled = false  -- error block for quick click
            createImagePath()
            imageMutlu = imageMutlu + 1
            pathimagesave = "images/2imageMutlu/"..imageMutlu..".png"
            cv.imwrite{pathimagesave, imageCroppedCV}
            widget.lineEdit2.text = imageMutlu
            widget.btnChange2.enabled = true   -- error block for quick click
            widget.labelProcess.text = "Saved image..."
          end);
-- 3 button callback 
qt.connect(qt.QtLuaListener(widget.btnChange3),
          'sigMousePress(int,int,QByteArray,QByteArray,QByteArray)',
          function ()
            widget.btnChange3.enabled = false  -- error block for quick click
            createImagePath()
            imageUzgun = imageUzgun + 1
            pathimagesave = "images/3imageUzgun/"..imageUzgun..".png"
            cv.imwrite{pathimagesave, imageCroppedCV}
            widget.lineEdit3.text = imageUzgun
            widget.btnChange3.enabled = true  -- error block for quick click 
            widget.labelProcess.text = "Saved image..."
          end);
-- 4 button callback 
qt.connect(qt.QtLuaListener(widget.btnChange4),
          'sigMousePress(int,int,QByteArray,QByteArray,QByteArray)',
          function ()
            widget.btnChange4.enabled = false  -- error block for quick click
            createImagePath()
            imageSaskin = imageSaskin + 1
            pathimagesave = "images/4imageSaskin/"..imageSaskin..".png"
            cv.imwrite{pathimagesave, imageCroppedCV}
            widget.lineEdit4.text = imageSaskin
            widget.btnChange4.enabled = true  -- error block for quick click
            widget.labelProcess.text = "Saved image..."
          end);
-- 5 button callback 
qt.connect(qt.QtLuaListener(widget.btnChange5),
          'sigMousePress(int,int,QByteArray,QByteArray,QByteArray)',
          function ()
            widget.btnChange5.enabled = false  -- error block for quick click
            createImagePath()
            imageSinirli = imageSinirli + 1
            pathimagesave = "images/5imageSinirli/"..imageSinirli..".png"
            cv.imwrite{pathimagesave, imageCroppedCV}
            widget.lineEdit5.text = imageSinirli
            widget.btnChange5.enabled = true  -- error block for quick click
            widget.labelProcess.text = "Saved image..."
          end);
-- package button callback 
qt.connect(qt.QtLuaListener(widget.buttonPaketle),
          'sigMousePress(int,int,QByteArray,QByteArray,QByteArray)',
          function ()
            widget.labelProcess.text = "Running Packaging..."
            widget.buttonPaketle.enabled = false  -- error block for quick click
            imageNotr    = qt.QString.tonumber(widget.lineEdit1.text)
            imageMutlu   = qt.QString.tonumber(widget.lineEdit2.text)
            imageUzgun   = qt.QString.tonumber(widget.lineEdit3.text)
            imageSaskin  = qt.QString.tonumber(widget.lineEdit4.text)
            imageSinirli = qt.QString.tonumber(widget.lineEdit5.text)
            paketle(imageNotr, imageMutlu, imageUzgun, imageSaskin, imageSinirli)
            widget.buttonPaketle.enabled = true  -- error block for quick click
            widget.labelProcess.text = "Package completed..."
          end);
-- starttrain button callback 
qt.connect(qt.QtLuaListener(widget.buttonEgitimBaslat),
          'sigMousePress(int,int,QByteArray,QByteArray,QByteArray)',
          function ()
            
            modelName =qt.QString.tostring(widget.modelAdi.text) -- Model Name
            
            widget.labelProcess.text = "Training..."
            widget.buttonEgitimBaslat.enabled = false  -- error block for quick click
            
            epochNumber = qt.QString.tonumber(widget.epochNumber.text)
            startTraining  = require 'startTraining'
            TrainResult = trainStart(epochNumber, modelName)
            
            if TrainResult == 111 then
            widget.labelProcess.text = "Completed train. Ready for use..."
            end
            widget.buttonEgitimBaslat.enabled = true  -- error block for quick click
          end);
-----------------------------------------------------------------------

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

widget.windowTitle = 'DeepFED Deep Facial Emotion Detection - User Model Create Image & Dataset & Train'
widget:show()
timer:start()
