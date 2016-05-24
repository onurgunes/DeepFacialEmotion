

-- include required packages
require 'xlua'  
require 'qt'
require 'qtwidget'
require 'qtuiloader'
xrequire('nnx',true)
xrequire('camera',true)

require 'ffmpeg'

-- setup UI
widget = qtuiloader.load('deneme.ui')
win1 = qt.QtLuaPainter(widget.frame)

videoYol = 'video.mpg'
videoWidth   = 320
videoHeight  = 240
videoSeconds = 4     -- kac saniye
videoFps     = 25    -- saniyede kac frame
videoToplam  = videoSeconds * videoFps
videoBasla   = 1 

function videoprocess()
   videoFrame = video:forward()
   videoBasla = videoBasla +1
end
function videoDisplay()
   image.display{image={videoFrame}, win=win1, zoom=1}
end

widget.btnVideo.enabled = false
-- Video button callback
qt.connect(qt.QtLuaListener(widget.btnVideo),
          'sigMousePress(int,int,QByteArray,QByteArray,QByteArray)',
          function ()
            videoBasla   = 1 
            video = ffmpeg.Video{path=videoYol, width=videoWidth, height=videoHeight, 
                                  fps=videoFps, length=videoSeconds, delete=false, destFolder='temp'}
            timer1 = qt.QTimer()
            timer1.interval = 40
            timer1.singleShot = true
            qt.connect(timer1,
                       'timeout()',
                       function()
                          videoprocess()
                          videoDisplay()
                          timer1:start()
                          if videoBasla > videoToplam then
                            timer1:stop()
                            widget.btnVideoSec.enabled = true
                            widget.btnVideo.enabled = false
                          end
                       end)
            timer1:start()
          end);
          
-- Video Sec button callback
qt.connect(qt.QtLuaListener(widget.btnVideoSec),
          'sigMousePress(int,int,QByteArray,QByteArray,QByteArray)',
          function ()
            videoYol1 = qt.QFileDialog.getOpenFileName(this,'Foto Seç')
            print(videoYol1)
            videoYol = qt.QString.tostring(videoYol1)
            widget.btnVideoSec.enabled = false
            widget.btnVideo.enabled = true
          end);



widget:show()


