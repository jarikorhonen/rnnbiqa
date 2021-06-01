%-------------------------------------------------------------------------
%
%  computeCNNfeatures.m
%
%  Written by Jari Korhonen, Shenzhen University
%  
%  Use this function to compute sequence of features vectors 
%  for input image. 
%
%  Note that C++/OpenCV implementation does not produce identical 
%  results with Matlab implementation.
%
%
%  Input: 
%           test_image:    Path to the test video file (e.g. png, jpg)
%           cnn:           Convolutional neural network for spatial
%                          feature extraction
%           cpugpu:        For using CPU, set to 'cpu', and for using 
%                          GPU, set to 'gpu'
%
%  Output:
%           features:      Resulting sequence of feature vectors 
%

function features = computeCNNfeatures(test_image, cnn, cpugpu)

    % Open image
    img = imread(test_image);
    img = cast(img,'double');

    % Make sure the image is large enough for at least one patch
    patch_size = [224 224];
    [height,width,~] = size(img);
    if height<patch_size(1) || width<patch_size(2)
        img = imresize(img, patch_size);
    end
    img_small = imresize(img,0.5,'method','box');
    [height,width,~] = size(img_small);
    if height<patch_size(1) || width<patch_size(2)
        img_small = imresize(img_small, patch_size);
    end
    
    % Initializations
    features = [];
    
    % Extrat patches and spatial activity vector
    [patches,si_vec] = extract_patches(img_small);
    [~,idx] = sort(si_vec,'ascend');
    for i1=1:length(si_vec)
        features(i1,:) = predict(cnn,patches(:,:,:,idx(i1)),...
                                 'ExecutionEnvironment',cpugpu);
    end 
    [patches,si_vec] = extract_patches(img);
    [~,idx] = sort(si_vec,'ascend');
    for i2=1:length(si_vec)
        features(i1+i2,:) = predict(cnn,patches(:,:,:,idx(i2)),...
                                    'ExecutionEnvironment',cpugpu);
    end 
    
    csv_file = test_image;
    csv_file(end-2:end) = 'csv';
    writematrix(features, csv_file);
end
    
function [im_patches,si_vec] = extract_patches(img)

    % Make Sobel filter -based spatial activity map
    im_act = im2gray(cast(img,'double')./255);
    h = [-1 -2 -1; 0 0 0; 1 2 1]./8;  
    im_act = sqrt(imfilter(im_act(:,:,1),h).^2 + ...
                  imfilter(im_act(:,:,1),h').^2); 
    im_act_z = 0.*im_act;
    im_act_z(3:end-2,3:end-2)=im_act(3:end-2,3:end-2);
    im_act = im_act_z; 
    [height,width,~] = size(img);
    patch_size = [224 224];
                
    % Split image in patches
    x_numb = ceil(width/patch_size(2));
    y_numb = ceil(height/patch_size(1));
    x_step = 1;
    y_step = 1;
    if x_numb>1 && y_numb>1
        x_step = floor((width-patch_size(1))/(x_numb-1));
        y_step = floor((height-patch_size(2))/(y_numb-1));
    end
    
    im_patches = [];
    si_vec = [];
    num_patches = 0;
    
    % Loop through all patches  
    for i=1:x_step:width-patch_size(2)+1
        for j=1:y_step:height-patch_size(2)+1      
            y_range = j:j+patch_size(2)-1;
            x_range = i:i+patch_size(1)-1;
            activity_patch = im_act(y_range, x_range);
            activity = std2(activity_patch);
            si_vec = [si_vec activity];
            num_patches = num_patches + 1;
            im_patches(:,:,:,num_patches) = img(y_range, x_range,:);          
        end
    end 
end
