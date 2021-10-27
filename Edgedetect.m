function [ output_args ] = Edgedetect(type)
%EDGEDETECT Summary of this function goes here
%   Detailed explanation goes here
[fileName,pathName] = uigetfile('*.*', 'Select an image');
curImg = imread(strcat(pathName,fileName));
imgBW = rgb2gray(curImg);

edgeImg = edge(imgBW, type);

figure, imshow(edgeImg);
end

