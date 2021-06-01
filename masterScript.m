%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  masterScript.m
%
%  Written by Jari Korhonen, Shenzhen University
%
%  This script runs scripts to process LIVE Challenge database,
%  train CNN feature extractor, make SPAQ-768 dataset, extract
%  CNN features from KonIQ-10k and SPAQ-768 datasets, and finally
%  train and test RNN model to predict MOS in KoNIQ-10k/SPAQ
%  cross-dataset scenario.
%
%  inputs: 
%          livec_path: path to the LIVE Challenge image quality  
%          database (e.g. 'd:\\live_challenge')
%
%          koniq_path: path to the KoNIQ-10k image quality  
%          database (e.g. 'd:\\koniq10k')
%
%          spaq_path: path to the SPAQ image quality 
%          database (e.g. 'd:\\spaq')
%
%          cpugpu: defined if CPU or GPU is used for training
%          and testing the CNN model, use either 'cpu' or 'gpu'
%
%  outputs: 
%          Displays SCC, PCC and RMSE results for cross-dataset
%          test on the RNN model (KoNIQ-10k / SPAQ)
%
function out = masterScript(livec_path, ...
                            koniq_path, ...
                            spaq_path, ...
                            cpugpu)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Phase 1: setup 
%
                        
cnn_model_file = '.\\CNN_model.onnx';
livec_patches_path =  '.\\livec_patches';
spaq768_path = [spaq_path '\\spaq768'];
out = 0;

% % Make CLIVE patches
fprintf('Generating patches from LIVE Challenge database...\n');
processLiveChallenge(livec_path, livec_patches_path);

% Make SPAQ-768 dataset
fprintf('Making SPAQ-768 dataset...\n');
resizeImages([spaq_path '\\TestImage\\'], spaq768_path, 768);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Phase 2: train the CNN feature extractor

fprintf('Training CNN feature extractor...\n');
trainCNNmodel(livec_patches_path, cnn_model_file, cpugpu);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Phase 3: computing feature sequences for image datasets

% Read saved ONNX model
model = importONNXNetwork(cnn_model_file,'OutputLayerType','regression');

% Compute features for all the images in KoNIQ-10k
fprintf('Extracting features for KoNIQ-10k...\n');
metadata = readtable([koniq_path '\\koniq10k_scores_and_distributions.csv']);
seqlen = size(metadata,1);
for i=1:seqlen
    filename = sprintf('%s\\%s',koniq_path,char(metadata{i,1}));
    computeCNNfeatures(filename, model, cpugpu);
    if mod(i,500)==0
        fprintf('Extracted features for %d/%d images\n',i,seqlen);
    end
end

% Compute features for all the images in SPAQ
fprintf('Extracting features for SPAQ-768...\n');
metadata = readmatrix([spaq_path '\\mos_spaq.xlsx']);
seqlen = size(metadata,1);
for i=1:seqlen
    filename = sprintf('%s\\%05d.png',spaq768_path,i);
    computeCNNfeatures(filename, model, cpugpu);
    if mod(i,500)==0
        fprintf('Extracted features for %d/%d images\n',i,seqlen);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Phase 4: training and testing the RNN quality model

% Read features from files; case 1: train with KoNIQ-10k, test with SPAQ
XTrain = {};
YTrain = [];
metadata = readtable([koniq_path '\\koniq10k_scores_and_distributions.csv']);
seqlen = size(metadata,1);
fprintf('Reading features from file for KoNIQ-10k...\n');
for i=1:seqlen
    filename = sprintf('%s\\%s',koniq_path,char(metadata{i,1}));
    filename(end-2:end)=['c' 's' 'v'];
    XTrain{i} = readmatrix(filename)';
    YTrain(i,:) = metadata{i,10}./100;
    if mod(i,1000)==0
        fprintf('Read features for %d/%d images\n',i,seqlen);
    end
end

XTest = {};
YTest = [];
fprintf('Reading features from file for SPAQ-768...\n');
metadata = readmatrix([spaq_path '\\mos_spaq.xlsx']);
seqlen = size(metadata,1);
for i=1:seqlen
    filename = sprintf('%s\\%05d.csv',spaq768_path,i);
    XTest{i} = readmatrix(filename)';
    YTest(i,:) = metadata(i,2)./100;
    if mod(i,1000)==0
        fprintf('Read features for %d/%d images\n',i,seqlen);
    end
end

% Train and test RNN model
result = trainAndTestRNNmodel(XTrain, YTrain, XTest, YTest, cpugpu);
fprintf('Results when training on KoNIQ-10k and testing on SPAQ:\n');
fprintf('SCC: %1.3f PCC: %1.3f RMSE: %2.1f\n', result(1), result(2), result(3)*100);

% Swap training and testing data
XTemp = XTest;
YTemp = YTest;
XTest = XTrain;
YTest = YTrain;
XTrain = XTemp;
YTrain = YTemp;

% Train and test RNN model
result = trainAndTestRNNmodel(XTrain, YTrain, XTest, YTest, cpugpu);
fprintf('Results when training on SPAQ and testing on KoNIQ-10k:\n');
fprintf('SCC: %1.3f PCC: %1.3f RMSE: %2.1f\n', result(1), result(2), result(3)*100);

out = 0;
end

% End of file %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%