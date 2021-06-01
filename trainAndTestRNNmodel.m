% --------------------------------------------------------------
%   trainAndTestRNNmodel.m
%
%   Written by Jari Korhonen, Shenzhen University
%
%   This function trains the RNN model to predict quality of 
%   the large resolution test images.
%
%   Usage: trainAndTestRNNmodel(XTrain,YTrain,XTest,YTest,cpugpu)
%   Inputs: 
%       XTrain:     Training feature vector sequences
%       YTrain:     Training ground truth MOS vector
%       XTest:      Testing feature vector sequences
%       YTest:      Testing ground truth MOS vector
%       cpugpu:     For using CPU, set to 'cpu', and for using 
%                   GPU, set to 'gpu'
%   Output: [srocc plcc rmse], where
%       srocc:      Spearman rank order correlation coefficient 
%       plcc:       Pearson linear correlation coefficient 
%       rmse:       Root mean squared error
%   

function res = trainAndTestRNNmodel(XTrain,YTrain,XTest,YTest,cpugpu)

    % Define network layers
    numFeatures = size(XTrain{1},1);
    layers = [ sequenceInputLayer(numFeatures,'Normalization','zerocenter')                
               fullyConnectedLayer(numFeatures,'WeightsInitializer','glorot') 
               reluLayer
               dropoutLayer(0.25)
               gruLayer(256,'OutputMode','sequence')
               gruLayer(128,'OutputMode','last')
               gruLayer(64,'OutputMode','last')
               gruLayer(32,'OutputMode','last')
               fullyConnectedLayer(1,'WeightsInitializer','glorot')
               myHuberRegressionLayer('reg',1/9)];

    % Define learning parameters
    options = trainingOptions('adam', ...
                              'MaxEpochs',5, ...
                              'MiniBatchSize',16, ...
                              'SequenceLength','longest', ...
                              'SequencePaddingDirection','right', ...
                              'GradientDecayFactor',0.95, ...
                              'SquaredGradientDecayFactor',0.9, ...
                              'LearnRateSchedule','piecewise', ...
                              'LearnRateDropPeriod', 1, ...
                              'LearnRateDropFactor',0.5, ...
                              'InitialLearnRate',0.0002, ...
                              'L2Regularization',0.00001, .....
                              'ExecutionEnvironment',cpugpu, ...
                              'ValidationData',{XTest,YTest}, ...
                              'ValidationFrequency', 200, ...
                              'Shuffle','every-epoch', ...
                              'plots','training-progress', ...
                              'Verbose',0);

    % Train the network
    net = trainNetwork(XTrain,YTrain,layers,options);

    % Test the network
    YPred = predict(net,XTest,'ExecutionEnvironment','gpu','SequenceLength','longest','SequencePaddingDirection','right')';
    res = [corr(YTest, YPred','type','Spearman') ...
           corr(YTest, YPred','type','Pearson') ...
           sqrt(mse(YTest, YPred'))];
end
       
