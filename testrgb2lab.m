function [ output_args ] = testrgb2lab( input_args )
%TESTRGB2LAB Summary of this function goes here
%   Detailed explanation goes here
    [fileName,pathName] = uigetfile('*.*', 'Select an image');
    img = imread(strcat(pathName,fileName));
    
    colorTransform = makecform('srgb2lab');
    imgLab = applycform(img, colorTransform);
    
    L_Image = imgLab(:, :, 1);  % Extract the L image.
    A_Image = imgLab(:, :, 2);  % Extract the A image.
    B_Image = imgLab(:, :, 3);  % Extract the B image.
end

