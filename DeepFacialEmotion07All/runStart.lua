
-- KULLANICI GİRİŞ EKRANI

-- include required packages
require 'xlua'	
require 'qt'
require 'qtwidget'
require 'qtuiloader'

-- setup UI
widget = qtuiloader.load('guiStart.ui')
win1 = qt.QtLuaPainter(widget.frame)

------------------------------------------------------------------------
-- 1 button callback 
qt.connect(qt.QtLuaListener(widget.button1),
          'sigMousePress(int,int,QByteArray,QByteArray,QByteArray)',
          function ()
		        os.execute('qlua runTrain.lua') 
          end);
-- 2 button callback 
qt.connect(qt.QtLuaListener(widget.button2),
          'sigMousePress(int,int,QByteArray,QByteArray,QByteArray)',
          function ()
           print('button 2 basıldı')
           os.execute('qlua runTest.lua') 
          end);
-- 3 button callback 
qt.connect(qt.QtLuaListener(widget.button3),
          'sigMousePress(int,int,QByteArray,QByteArray,QByteArray)',
          function ()
            print('button 3 basıldı')
            os.execute('cd hazirtest;qlua runHazirTest.lua')
          end);
-----------------------------------------------------------------------

widget.windowTitle = 'DeepFED Yüzden Duygu Tanıma'
widget:show()
