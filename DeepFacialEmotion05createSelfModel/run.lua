
-- KULLANICI IMAGE PAKETLEME &  GUI



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

-- Önceki format için farklı bir pencelere opencv formatında foro gösterimi vardı
-- Bunun yerine Torch tensor tarzına dönüştürüp window içerisinde gösterir

-- OPENCV icin (gereksizler silinecek)

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

-- OPENCV icin (gereksizler silinecek)

local cascade_path = 'haarcascade_frontalface_default.xml'
local face_cascade = cv.CascadeClassifier{filename=cascade_path}

-- setup UI
widget = qtuiloader.load('gui.ui')
win1 = qt.QtLuaPainter(widget.frame)
win2 = qt.QtLuaPainter(widget.frame2) -- CROPLU IMAGE GOSTEREN FRAME

-- initializing the camera
  local cap = cv.VideoCapture{device=0}
  local _, frame = cap:read{}
  -- camera = image.Camera{}

function display()
  -- frame = torch.Tensor(100,100)
	-- frame = camera:forward()	-- takes image from camera
   
   local fx = 0.50  -- rescale factor -- 1 olunca hata artar
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
         text = 'yuz: ',
         org={x, y-10}, -- yazı konumu
         fontFace=cv.FONT_HERSHEY_PLAIN,
         fontScale=1,
         color={255, 255, 0},
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
      imageCroppedCV = im
      rgbTensorCroped = convertBRGtoRGB(im)
      image.display{image = rgbTensorCroped, win = win2, zoom = 1}  -- CROP GOSTER
      
      end
      ---------------------
   end
   
      rgbTensor = convertBRGtoRGB(frame) -- OpenCV formatından Torch'a cevirir.
      image.display{image = rgbTensor, win = win1, zoom = 1}  --yeni format
      cap:read{image=frame}
      
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
-- PATH OLUSTUR
function createImagePath()

      os.execute('mkdir -p ' .. 'images')
      os.execute('mkdir -p ' .. 'images/1imageNotr')
      os.execute('mkdir -p ' .. 'images/2imageMutlu')
      os.execute('mkdir -p ' .. 'images/3imageUzgun')
      os.execute('mkdir -p ' .. 'images/4imageSaskin')
      os.execute('mkdir -p ' .. 'images/5imageSinirli')
      print('Path Oluşturuldu...')
end
-----------------------------------------------------------------------
-- PAKETLEME
function paketle(im1, im2, im3, im4, im5)
      
      totalTrainImages = im1 + im2 + im3 + im4 + im5
      
      -- classes
      classes = {'1imageNotr','2imageMutlu','3imageUzgun','4imageSaskin','5imageSinirli'}
      
      trsize = totalTrainImages -- EĞİTİM TOPLAM IMAGE SAYISI
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
         
      print('Kaydedildi')
      
      deneme = torch.load('trainDataset.t7')
      
      -- KAYIT SONRASI TEST
      -- print(deneme.data:size())
      -- print(deneme.data:type())
      -- print(deneme.labels[totalTrainImages])
      -- image.display(deneme.data[totalTrainImages])

end
-----------------------------------------------------------------------

imageNotr    = 0 -- BAŞLANGIÇ DEGERLERİ
imageMutlu   = 0
imageUzgun   = 0
imageSaskin  = 0
imageSinirli = 0

-----------------------------------------------------------------------
-- 1 button callback 
qt.connect(qt.QtLuaListener(widget.btnChange1),
          'sigMousePress(int,int,QByteArray,QByteArray,QByteArray)',
          function ()
		        --print('button 1 basıldı')
		        widget.btnChange1.enabled = false  -- SERİ TIKLAMA HATA ENGELLEME İÇİN 
		        createImagePath()
		        imageNotr = imageNotr + 1
            pathimagesave = "images/1imageNotr/"..imageNotr..".png"
            cv.imwrite{pathimagesave, imageCroppedCV}
            widget.lineEdit1.text = imageNotr
            widget.btnChange1.enabled = true   -- SERİ TIKLAMA HATA ENGELLEME İÇİN 
            widget.labelProcess.text = "Image Kaydedildi..."
          end);
-- 2 button callback 
qt.connect(qt.QtLuaListener(widget.btnChange2),
          'sigMousePress(int,int,QByteArray,QByteArray,QByteArray)',
          function ()
            --print('button 2 basıldı')
            widget.btnChange2.enabled = false  -- SERİ TIKLAMA HATA ENGELLEME İÇİN 
            createImagePath()
            imageMutlu = imageMutlu + 1
            pathimagesave = "images/2imageMutlu/"..imageMutlu..".png"
            cv.imwrite{pathimagesave, imageCroppedCV}
            widget.lineEdit2.text = imageMutlu
            widget.btnChange2.enabled = true   -- SERİ TIKLAMA HATA ENGELLEME İÇİN 
            widget.labelProcess.text = "Image Kaydedildi..."
          end);
-- 3 button callback 
qt.connect(qt.QtLuaListener(widget.btnChange3),
          'sigMousePress(int,int,QByteArray,QByteArray,QByteArray)',
          function ()
            --print('button 3 basıldı')
            widget.btnChange3.enabled = false  -- SERİ TIKLAMA HATA ENGELLEME İÇİN 
            createImagePath()
            imageUzgun = imageUzgun + 1
            pathimagesave = "images/3imageUzgun/"..imageUzgun..".png"
            cv.imwrite{pathimagesave, imageCroppedCV}
            widget.lineEdit3.text = imageUzgun
            widget.btnChange3.enabled = true  -- SERİ TIKLAMA HATA ENGELLEME İÇİN 
            widget.labelProcess.text = "Image Kaydedildi..."
          end);
-- 4 button callback 
qt.connect(qt.QtLuaListener(widget.btnChange4),
          'sigMousePress(int,int,QByteArray,QByteArray,QByteArray)',
          function ()
            --print('button 4 basıldı')
            widget.btnChange4.enabled = false  -- SERİ TIKLAMA HATA ENGELLEME İÇİN 
            createImagePath()
            imageSaskin = imageSaskin + 1
            pathimagesave = "images/4imageSaskin/"..imageSaskin..".png"
            cv.imwrite{pathimagesave, imageCroppedCV}
            widget.lineEdit4.text = imageSaskin
            widget.btnChange4.enabled = true  -- SERİ TIKLAMA HATA ENGELLEME İÇİN 
            widget.labelProcess.text = "Image Kaydedildi..."
          end);
-- 5 button callback 
qt.connect(qt.QtLuaListener(widget.btnChange5),
          'sigMousePress(int,int,QByteArray,QByteArray,QByteArray)',
          function ()
            --print('button 5 basıldı')
            widget.btnChange5.enabled = false  -- SERİ TIKLAMA HATA ENGELLEME İÇİN 
            createImagePath()
            imageSinirli = imageSinirli + 1
            pathimagesave = "images/5imageSinirli/"..imageSinirli..".png"
            cv.imwrite{pathimagesave, imageCroppedCV}
            widget.lineEdit5.text = imageSinirli
            widget.btnChange5.enabled = true  -- SERİ TIKLAMA HATA ENGELLEME İÇİN 
            widget.labelProcess.text = "Image Kaydedildi..."
          end);
-- paketle button callback 
qt.connect(qt.QtLuaListener(widget.buttonPaketle),
          'sigMousePress(int,int,QByteArray,QByteArray,QByteArray)',
          function ()
            widget.labelProcess.text = "Paketleme İşlemde"
            widget.buttonPaketle.enabled = false  -- SERİ TIKLAMA HATA ENGELLEME İÇİN 
            imageNotr    = qt.QString.tonumber(widget.lineEdit1.text)
            imageMutlu   = qt.QString.tonumber(widget.lineEdit2.text)
            imageUzgun   = qt.QString.tonumber(widget.lineEdit3.text)
            imageSaskin  = qt.QString.tonumber(widget.lineEdit4.text)
            imageSinirli = qt.QString.tonumber(widget.lineEdit5.text)
            paketle(imageNotr, imageMutlu, imageUzgun, imageSaskin, imageSinirli)
            widget.buttonPaketle.enabled = true  -- SERİ TIKLAMA HATA ENGELLEME İÇİN 
            widget.labelProcess.text = "Paketleme Tamamlandı"
          end);
-- egitimbaslat button callback 
qt.connect(qt.QtLuaListener(widget.buttonEgitimBaslat),
          'sigMousePress(int,int,QByteArray,QByteArray,QByteArray)',
          function ()
            widget.labelProcess.text = "Eğitimde"
            widget.buttonEgitimBaslat.enabled = false  -- SERİ TIKLAMA HATA ENGELLEME İÇİN 
            
            epochSayisi = qt.QString.tonumber(widget.epochNumber.text)
            egitimbaslat  = require 'egitimbaslat'
            egitimSonuc = trainStart(epochSayisi)
            
            if egitimSonuc == 111 then
            widget.labelProcess.text = "Eğitim bitti. Kullanıma Hazır"
            end
            widget.buttonEgitimBaslat.enabled = true  -- SERİ TIKLAMA HATA ENGELLEME İÇİN 
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

widget.windowTitle = 'IMAGE OLUSTUR & PAKETLE & EĞİTİM YAP'
widget:show()
timer:start()
