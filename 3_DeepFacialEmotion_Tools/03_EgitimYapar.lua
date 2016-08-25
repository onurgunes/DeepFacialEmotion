
require 'nn'
require 'optim'
require 'image'

-- EĞITİM YAPAR

----------------------------------------------------------------------
-- parse command-line options
--
dname,fname = sys.fpath()
fname = "egitilmisModel"
cmd = torch.CmdLine()
cmd:text()
cmd:text('CIFAR Training')
cmd:text()
cmd:text('Options:')
cmd:option('-save', fname, 'kayıt ve log için alt dizin adı')
cmd:option('-network', '', 'reload pretrained network')  -- KULLANILMADI
cmd:option('-model', 'convnet', 'eğitimdeki kullanılacak model tipi: convnet | mlp | linear')
cmd:option('-full', false, 'use full dataset (50,000 samples)')  -- KULLANILMADI
cmd:option('-visualize', true, 'eğitimde input data ve ağırlıkların görsellemesi')
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
cmd:option('-optimization', 'SGD', 'optimization method: SGD | ASGD | CG | LBFGS')
cmd:option('-learningRate', 1e-3, 'öğrenme oranı t=0')
cmd:option('-batchSize', 32, 'mini-batch size (1 = pure stochastic)')
cmd:option('-weightDecay', 0, 'weight decay (SGD only)')
cmd:option('-momentum', 0, 'momentum (SGD only)')
cmd:option('-t0', 1, 'start averaging at t0 (ASGD only), in nb of epochs')
cmd:option('-maxIter', 5, 'maximum nb of iterations for CG and LBFGS')
cmd:option('-threads', 1, 'nb of threads to use')
cmd:text()
opt = cmd:parse(arg)

-- fix seed
torch.manualSeed(opt.seed)

-- threads
torch.setnumthreads(opt.threads)
print('<torch> set nb of threads to ' .. opt.threads)

----------------------------------------------------------------------
-- define model to train
-- on the 10-class classification problem
--
--classes = {'airplane', 'automobile', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck'}

-- classes
classes = {'1imageNotr','2imageMutlu','3imageUzgun','4imageSaskin','5imageSinirli'}

if opt.network == '' then
   -- define model to train
   model = nn.Sequential()

   if opt.model == 'convnet' then
      ------------------------------------------------------------
      -- convolutional network
      ------------------------------------------------------------
      -- stage 1 : mean+std normalization -> filter bank -> squashing -> max pooling
      model:add(nn.SpatialConvolutionMap(nn.tables.random(3,16,1), 5, 5))
      model:add(nn.Tanh())
      model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
      -- stage 2 : filter bank -> squashing -> max pooling
      model:add(nn.SpatialConvolutionMap(nn.tables.random(16, 256, 4), 5, 5))
      model:add(nn.Tanh())
      model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
      -- stage 3 : standard 2-layer neural network
      model:add(nn.Reshape(256*5*5))
      model:add(nn.Linear(256*5*5, 128))
      model:add(nn.Tanh())
      model:add(nn.Linear(128,#classes))
      ------------------------------------------------------------
   else
      print('Unknown model type')
      cmd:text()
      error()
   end
else
   print('<trainer> reloading previously trained network')
   model = nn.Sequential()
   model:read(torch.DiskFile(opt.network))
end

-- retrieve parameters and gradients
parameters,gradParameters = model:getParameters()

-- verbose
print('<cifar> kullanılacak model:')
print(model)

----------------------------------------------------------------------
-- loss function: negative log-likelihood
--
model:add(nn.LogSoftMax())
criterion = nn.ClassNLLCriterion()

----------------------------------------------------------------------
-- get/create dataset
trsize = 6302 -- EĞİTİM TOPLAM IMAGE SAYISI
trainData = {
   size = function() return trsize end
}

trainData = torch.load('trainDataset.t7')

----------------------------------------------------------------------
-- preprocess/normalize train/test sets
--

print '<trainer> preprocessing data (color space + normalization)'
collectgarbage()

-- preprocess trainSet
normalization = nn.SpatialContrastiveNormalization(1, image.gaussian1D(7))
for i = 1,trainData:size() do
   -- rgb -> yuv
   local rgb = trainData.data[i]
   local yuv = image.rgb2yuv(rgb)
   -- normalize y locally:
   yuv[1] = normalization(yuv[{{1}}])
   trainData.data[i] = yuv
end

-- normalize u globally:
mean_u = trainData.data[{ {},2,{},{} }]:mean()
std_u = trainData.data[{ {},2,{},{} }]:std()
trainData.data[{ {},2,{},{} }]:add(-mean_u)
trainData.data[{ {},2,{},{} }]:div(-std_u)

-- normalize v globally:
mean_v = trainData.data[{ {},3,{},{} }]:mean()
std_v = trainData.data[{ {},3,{},{} }]:std()
trainData.data[{ {},3,{},{} }]:add(-mean_v)
trainData.data[{ {},3,{},{} }]:div(-std_v)


tesize = 50
testData = {
   size = function() return tesize end
}
testData = torch.load('testDataset.t7')


-- preprocess testSet
for i = 1,testData:size() do
   -- rgb -> yuv
   local rgb = testData.data[i]
   local yuv = image.rgb2yuv(rgb)
   -- normalize y locally:
   yuv[{1}] = normalization(yuv[{{1}}])
   testData.data[i] = yuv
end
-- normalize u globally:
testData.data[{ {},2,{},{} }]:add(-mean_u)
testData.data[{ {},2,{},{} }]:div(-std_u)
-- normalize v globally:
testData.data[{ {},3,{},{} }]:add(-mean_v)
testData.data[{ {},3,{},{} }]:div(-std_v)



collectgarbage()


----------------------------------------------------------------------
-- define training and testing functions
--

-- this matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

-- print(confusion)


-- log results to files
accLogger = optim.Logger(paths.concat(opt.save, 'dogruluk.log'))
errLogger = optim.Logger(paths.concat(opt.save, 'hata.log'   ))

-- display function
function display(input)
   iter = iter or 0
   require 'image'
   win_input = image.display{image=input, win=win_input, zoom=2, legend='input'}
   if iter % 10 == 0 then
      if opt.model == 'convnet' then
         win_w1 = image.display{
            image=model:get(1).weight, zoom=4, nrow=10,
            min=-1, max=1,
            win=win_w1, legend='Aşama 1: Ağırlıklar', padding=1
         }
         win_w2 = image.display{
            image=model:get(4).weight, zoom=2, nrow=30,
            min=-1, max=1,
            win=win_w2, legend='Aşama 2: Ağırlıklar', padding=1
         }
      elseif opt.model == 'mlp' then
         local W1 = torch.Tensor(model:get(2).weight):resize(2048,1024)
         win_w1 = image.display{
            image=W1, zoom=0.5, min=-1, max=1,
            win=win_w1, legend='W1 Ağırlıklar'
         }
         local W2 = torch.Tensor(model:get(2).weight):resize(10,2048)
         win_w2 = image.display{
            image=W2, zoom=0.5, min=-1, max=1,
            win=win_w2, legend='W2 Ağırlıklar'
         }
      end
   end
   iter = iter + 1
end

-- training function
function train(dataset)
   -- epoch tracker
   epoch = epoch or 1

   -- local vars
   local time = sys.clock()
   local trainError = 0

   -- do one epoch
   print('<trainer> eğitim başlangıcı:')
   print("<trainer> epoch sayısı # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
   for t = 1,dataset:size(),opt.batchSize do
      -- disp progress
      xlua.progress(t, dataset:size())

      -- create mini batch
      local inputs = {}
      local targets = {}
      for i = t,math.min(t+opt.batchSize-1,dataset:size()) do
         -- load new sample
         local input = dataset.data[i]
         local target = dataset.labels[i]
         table.insert(inputs, input)
         table.insert(targets, target)
      end

      -- create closure to evaluate f(X) and df/dX
      local feval = function(x)
         -- get new parameters
         if x ~= parameters then
            parameters:copy(x)
         end
         -- reset gradients
         gradParameters:zero()

         -- f is the average of all criterions
         local f = 0

         -- evaluate function for complete mini batch
         for i = 1,#inputs do
            -- estimate f
            local output = model:forward(inputs[i])
            local err = criterion:forward(output, targets[i])
            f = f + err

            -- estimate df/dW
            local df_do = criterion:backward(output, targets[i])
            model:backward(inputs[i], df_do)

            -- update confusion
            confusion:add(output, targets[i])
            

            -- visualize?
            if opt.visualize then
               display(inputs[i])
            end
         end

         -- normalize gradients and f(X)
         gradParameters:div(#inputs)
         f = f/#inputs
         trainError = trainError + f

         -- return f and df/dX
         return f,gradParameters
      end

      -- optimize on current mini-batch
      if opt.optimization == 'CG' then
         config = config or {maxIter = opt.maxIter}
         optim.cg(feval, parameters, config)

      elseif opt.optimization == 'LBFGS' then
         config = config or {learningRate = opt.learningRate,
                             maxIter = opt.maxIter,
                             nCorrection = 10}
         optim.lbfgs(feval, parameters, config)

      elseif opt.optimization == 'SGD' then
      
         config = config or {learningRate = opt.learningRate,
                             weightDecay = opt.weightDecay,
                             momentum = opt.momentum,
                             learningRateDecay = 5e-7}
         optim.sgd(feval, parameters, config)

      elseif opt.optimization == 'ASGD' then
         config = config or {eta0 = opt.learningRate,
                             t0 = nbTrainingPatches * opt.t0}
         _,_,average = optim.asgd(feval, parameters, config)

      else
         error('unknown optimization method')
      end
   end

   -- train error
   trainError = trainError / math.floor(dataset:size()/opt.batchSize)

   -- time taken
   time = sys.clock() - time
   time = time / dataset:size()
   print("<trainer> 1 sample öğrenme süresi = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)
   local trainAccuracy = confusion.totalValid * 100
   confusion:zero()

   -- save/log current net
   local filename = paths.concat(opt.save, 'egitilmisModel.net')
   os.execute('mkdir -p ' .. paths.dirname(filename))
   if paths.filep(filename) then
      os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
   end
   print('<trainer> ağ kaydediliyor '..filename)
   torch.save(filename, model)

   -- next epoch
   epoch = epoch + 1

   return trainAccuracy, trainError
end

-- test function
function test(dataset)
 -- local vars
   local testError = 0
   local time = sys.clock()

   -- averaged param use?
   if average then
      cachedparams = parameters:clone()
      parameters:copy(average)
   end

   -- test over given dataset
   print('<trainer> test başlangıcı:')
   for t = 1,dataset:size() do
      -- disp progress
      xlua.progress(t, dataset:size())

      -- get new sample
      local input = dataset.data[t]
      local target = dataset.labels[t]

      -- test sample
      local pred = model:forward(input)
      confusion:add(pred, target)

      -- compute error
      err = criterion:forward(pred, target)
      testError = testError + err
   end

   -- timing
   time = sys.clock() - time
   time = time / dataset:size()
   print("<trainer> 1 sample test süresi = " .. (time*1000) .. 'ms')

   -- testing error estimation
   testError = testError / dataset:size()

   -- print confusion matrix
   print(confusion)
   local testAccuracy = confusion.totalValid * 100
   confusion:zero()

   -- averaged param use?
   if average then
      -- restore parameters
      parameters:copy(cachedparams)
   end

   return testAccuracy, testError
end




----------------------------------------------------------------------
-- and train!
--
--
--

while true do
   -- train/test
   trainAcc, trainErr = train(trainData)
   testAcc,  testErr  = test (testData)

   -- update logger
   accLogger:add{['% egitimde dogruluk'] = trainAcc, ['% testte dogruluk'] = testAcc}
   errLogger:add{['% egitimde hata']    = trainErr, ['% testte hata']    = testErr}

   -- plot logger
   accLogger:style{['% egitimde dogruluk'] = '-', ['% testte dogruluk'] = '-'}
   errLogger:style{['% egitimde hata']    = '-', ['% testte hata']    = '-'}
   accLogger:plot()
   errLogger:plot()
end


