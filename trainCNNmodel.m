% --------------------------------------------------------------
%   trainCNNmodel.m
%
%   Written by Jari Korhonen, Shenzhen University
%
%   This function trains the CNN feature extractor.
%
%   Inputs: 
%       path:       Path to the training patches
%       model_file: Name of the file where to save the obtained
%                   CNN model
%       cpugpu:     For using CPU, set to 'cpu', and for using 
%                   GPU, set to 'gpu'
%
%   Output: dummy
%   
function res = trainCNNmodel(path,model_file,cpugpu)

    % Load probabilistic representations for quality scores
    load([path '\\LiveC_prob.mat'],'LiveC_prob');

    % Loop through all the training patches to obtain source paths
    % and the respective ground truth outputs
    filenames = {};
    outputs = [];
    sqlen = length(LiveC_prob(:,1));
    for i=1:sqlen
        for j=1:36
            filenames = [filenames; sprintf('%s\\%04d_%02d.png',path,i,j)];
            outputs = [outputs; LiveC_prob(i,:)];
        end
    end
    T = table(filenames, outputs);
   
    % Get pre-trained ResNet50 model
    net = resnet50; 

    % Modify the model for quality prediction
    newLayer1 = fullyConnectedLayer(5,'WeightLearnRateFactor',2,'BiasLearnRateFactor',2,'Name','fc_output','WeightsInitializer','narrow-normal');
    newLayer2 = mySoftmaxLayer('softmax');
    newLayer3 = myCrossentropyRegressionLayer('output');
    lgraph = layerGraph(net);
    lgraph = replaceLayer(lgraph,'fc1000',newLayer1);
    lgraph = replaceLayer(lgraph,'fc1000_softmax',newLayer2);
    lgraph = replaceLayer(lgraph,'ClassificationLayer_fc1000',newLayer3);
    layers = lgraph.Layers;
    connections = lgraph.Connections;

    % Freeze layers 1-36
    layers(1:36) = freezeWeights(layers(1:36));
    lgraph = createLgraphUsingConnections(layers,connections);

    % Define training options
    options = trainingOptions('sgdm', ...
        'MiniBatchSize',16, ...        
        'MaxEpochs',2, ...
        'L2Regularization',0.01, ...
        'InitialLearnRate',0.0005, ...
        'Shuffle','every-epoch', ...
        'ExecutionEnvironment',cpugpu, ...
        'Verbose',false, ...
        'Plots','training-progress');

    % Train the model
    model = trainNetwork(T,'outputs',lgraph,options);
   
    % Save the model in ONNX format
    lgraph = layerGraph(model);
    lgraph = removeLayers(lgraph,'fc_output');
    lgraph = removeLayers(lgraph,'softmax');
    lgraph = replaceLayer(lgraph,'output',regressionLayer('Name','output'));
    lgraph = connectLayers(lgraph, 'avg_pool', 'output');
    model = assembleNetwork(lgraph);
    exportONNXNetwork(model,model_file);
end
