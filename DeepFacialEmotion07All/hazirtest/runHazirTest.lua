
-- include required packages
require 'xlua'	
require 'qt'
require 'qtwidget'
require 'qtuiloader'
xrequire('nnx',true)
xrequire('camera',true)


require 'nn'
require 'optim'
require 'image'
model = torch.load('egitilmisModel100e5c.net')

classes = {'1imageNotr','2imageMutlu','3imageUzgun','4imageSaskin','5imageSinirli'}
confusion = optim.ConfusionMatrix(classes)
criterion = nn.ClassNLLCriterion()

tesize = 1
testData = {
data = torch.Tensor(tesize,3,32,32),
labels = torch.Tensor(tesize),
   size = function() return tesize end
}

normalization = nn.SpatialContrastiveNormalization(1, image.gaussian1D(7))


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

local cascade_path = 'haarcascade_frontalface_default.xml'
local face_cascade = cv.CascadeClassifier{filename=cascade_path}

-- setup UI
widget = qtuiloader.load('guiHazirTest.ui')
win1 = qt.QtLuaPainter(widget.frame)
win2 = qt.QtLuaPainter(widget.frame2) -- CROPLU IMAGE GOSTEREN FRAME
win3 = qt.QtLuaPainter(widget.frame3) -- CROPLU IMAGE GOSTEREN FRAME YUV

-- initializing the camera
  local cap = cv.VideoCapture{device=0}
  local _, frame = cap:read{}
  -- camera = image.Camera{}

function display()

   local fx = 0.20  -- rescale factor -- 1 olunca hata artar
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
      cv.rectangle{img=frame, pt1={x, y}, pt2={x + w, y + h}, color={255,0,0,0}, thickness=2}
      cv.putText{
         img=frame,
         text = 'yuz: ',
         org={x, y-10}, -- yazı konumu
         fontFace=cv.FONT_HERSHEY_PLAIN,
         fontScale=1,
         color={0, 0, 255},
         thickness=1
         }
      
      ------------------------------------------------------------------------
      -- crop : CAM içerisinde yüz varsa keser 32x32 tensor olarak alır
      crop = cv.getRectSubPix{
        image=frame,
        patchSize={w, h},
        center={x + w/2, y + h/2},
      }
      if crop then
        local im = cv.resize{src=crop, dsize={32,32}}:float()
      
        if fotoileTest == 1 then
          rgbTensorCroped = image.load(secilecekdosyaadi)
          rgbTensorCroped = image.scale(rgbTensorCroped, 32, 32)
        else
          rgbTensorCroped = convertBRGtoRGB(im)
        end
      
      -- image.display(rgbTensorCroped)
      
      -- rgbTensorCroped = image.load('test1.png') -- TEST IMAGE EĞİTİMDE KULLANILAN
      
      image.display{image = rgbTensorCroped, win = win2, zoom = 1}  -- CROP GOSTER
      
              
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
              
              image.display{image = testData.data[1], win = win3, zoom = 1}  -- CROP GOSTER YUV
              
                    input = testData.data[1]
                    pred = model:forward(input)
                    pred:exp()
                    
                    for i=1,pred:size(1) do
                      print(classes[i], pred[i])
                    end
                    
      
      
      end
      -------------------------------------------------------------------------



      
   end
   
      rgbTensor = convertBRGtoRGB(frame) -- OpenCV formatından Torch'a cevirir.
      
      -- cv.imshow{winname="Yuz Bulur", image=frame} Eski OpenCV formatı pencere
      image.display{image = rgbTensor, win = win1, zoom = 1}  --yeni format
      cap:read{image=frame}
      
      
	   if crop then -- crop olursa değerleri değiştir
  	   widget.progressBar:setValue(pred[1]*100);
  	   widget.progressBar_2:setValue(pred[2]*100);
  	   widget.progressBar_3:setValue(pred[3]*100);
  	   widget.progressBar_4:setValue(pred[4]*100);
  	   widget.progressBar_7:setValue(pred[5]*100);
	   end
end


-----------------------------------------------------------------------
-- BRG to RGB
function convertBRGtoRGB(frame)
      forQTimage = frame:transpose(2,3):transpose(1,2) -- OpenCV formatından Torch'a cevirir.
      --Torch a cevrilen tensor BGR formatindan oldugu icin RGB ye cevrilir.
      local b,g,r = forQTimage[1],forQTimage[2],forQTimage[3]
      local temp = forQTimage[1];
      local rgbTensor = forQTimage:clone()
      rgbTensor[1] = rgbTensor[3]
      rgbTensor[3] = temp
      --Bu aşamadan sonra rgbTensor formati Torch a uygun
      return rgbTensor
end
-----------------------------------------------------------------------

-- fotograf button callback
qt.connect(qt.QtLuaListener(widget.btnFoto),
          'sigMousePress(int,int,QByteArray,QByteArray,QByteArray)',
          function ()
            secilecekdosyaadi1 = qt.QFileDialog.getOpenFileName(this,'Foto Seç')
            print(secilecekdosyaadi1)
            secilecekdosyaadi = qt.QString.tostring(secilecekdosyaadi1)
            fotoileTest = 1
            webcamileTest  = 0
            -- print(secilecekdosyaadi)
          end);
-- webcam button callback
qt.connect(qt.QtLuaListener(widget.btnCam),
          'sigMousePress(int,int,QByteArray,QByteArray,QByteArray)',
          function ()
            fotoileTest = 0
            webcamileTest  = 1
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

widget.windowTitle = 'Deep Facial Emotion Detector'
widget:show()
timer:start()
