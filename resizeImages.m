% --------------------------------------------------------------
%   resizeImages.m
%
%   This function resizes SPAQ dataset images to target resolution.
%   The code is based on the implementation from the authors of SPAQ.
%
%   Usage: resizeImages(img_path, save_path, minSize)
%   Inputs: 
%       img_path:   Path to the original images
%       save_path:  Path to the resized images
%       minSize:    Size of the smallest side after resizing
%   Output: 
%       res:        dummy
%   

function res = resizeImages(img_path, save_path, minSize)

if ~exist(save_path, 'dir')
    mkdir(save_path);
end

for i=1:numel(dir(img_path)) - 2                                                                                                                                                                                                                                                                                                                                                                                                                                       
    image = imread([img_path num2str(i, '%05d') '.jpg']);
    [h, w, ~] = size(image);
    if h >= w && w > minSize
        sampleFactor = w / minSize;
        result = imresize(image, [floor(h / sampleFactor), minSize], 'bilinear');
    elseif h < w && h > minSize
        sampleFactor = h / minSize;
        result = imresize(image, [minSize, floor(w / sampleFactor)], 'bilinear');
    else
        result = image;
    end
    imwrite(result, [save_path '\\' num2str(i, '%05d') '.png'])
end

res = 0;

end