function TestFeatures()
%TESTFEATURES Summary of this function goes here
%   Detailed explanation goes here
        blockSize = 8;
        
        [fileName,pathName] = uigetfile({'*.*'},'Select an image');
        img = imread(strcat(pathName,fileName));
        
        imgGray = rgb2gray(img);
        
        blockX = 1;
        blockY = 1;
        sizeImg = size(imgGray);
        map = zeros(sizeImg(1) - 4, sizeImg(2) - 4);

        while blockX <= sizeImg(1) - blockSize
            while blockY <= sizeImg(2) - blockSize
                blockImg = double(imgGray(blockX:blockX + blockSize, blockY:blockY + blockSize));
                
                [Gx, Gy]=gradient(blockImg);
                
                S=sqrt(Gx.*Gx+Gy.*Gy);
                sharpness=sum(sum(S))./(numel(Gx));
                blockVal = sharpness/100;
                
                map(blockX:blockX + 4, blockY:blockY + 4) = blockVal;
                
                blockY = blockY + 4;
            end
            
            blockY = 1;
            blockX = blockX + 4;
        end
        
        figure, imshow(map);
end

