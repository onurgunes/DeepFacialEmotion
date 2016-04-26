require 'nn'
require 'optim'
require 'image'


-- t7 PAKETLEME eski

   --save created dataset:
   -- torch.save('traindeneme.t7',trainData)
   
   
   print('Kaydedildi')
   deneme = torch.load('traindeneme.t7')
   
   print(deneme.data:size())
   print(deneme.data:type())
   print(deneme.labels[1])
   image.display(deneme.data[1])



-- image.display(trainData.data[1])


