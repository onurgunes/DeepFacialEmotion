
-- include required packages
require 'xlua'	
require 'qt'
require 'qtwidget'
require 'qtuiloader'
xrequire('nnx',true)
xrequire('camera',true)

-- setup UI
widget = qtuiloader.load('gui.ui')
win1 = qt.QtLuaPainter(widget.frame)

-- initializing the camera
camera = image.Camera{}

function display()
  cam = torch.Tensor(100,100)
	cam = camera:forward()	-- takes image from camera
	image.display{image = cam, win = win1, zoom = 0.75}	--original image
	
	if changeValues == true then
	   --  progressBar_2
	   --widget.label:setText("Degisti");
	   widget.progressBar:setValue(math.random(0,100));
	   widget.progressBar_2:setValue(math.random(0,100));
	   widget.progressBar_3:setValue(math.random(0,100));
	   --print("tiklandi")
	   changeValues = false
	end 
end

-- train network
function train(dataset)

end

-- test network
function test(dataset)

end

-- button callback
qt.connect(qt.QtLuaListener(widget.btnChange),
          'sigMousePress(int,int,QByteArray,QByteArray,QByteArray)',
          function ()
            changeValues = true
          end);
--function (...) changeValues = true end

-- timer
timer = qt.QTimer()
timer.interval = 10
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
