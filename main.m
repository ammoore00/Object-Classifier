function main()
%MAIN Senior Project Object Recognition
%   Detailed explanation goes here
    clear;
    clc;
    close all;
    
    fig_main = figure('Units', 'Normalized', 'Outerposition', [0, 0, 1, 1], 'MenuBar', 'none', 'Name', 'Main', 'NumberTitle', 'off', 'ToolBar', 'none', 'Visible', 'off');
    fig_capImg = figure('Units', 'Normalized', 'Outerposition', [0, 0, 1, 1], 'MenuBar', 'none', 'Name', 'Get Image', 'NumberTitle', 'off', 'ToolBar', 'none', 'Visible', 'off');
    fig_procImg = figure('Units', 'Normalized', 'Outerposition', [0, 0, 1, 1], 'MenuBar', 'none', 'Name', 'Process Image', 'NumberTitle', 'off', 'ToolBar', 'none', 'Visible', 'off');
    fig_addData = figure('Units', 'Normalized', 'Outerposition', [0, 0, 1, 1], 'MenuBar', 'none', 'Name', 'Add Images to Database', 'NumberTitle', 'off', 'ToolBar', 'none', 'Visible', 'off');
    
    fig_output = figure('Units', 'Normalized', 'Outerposition', [0, 0, 1, 1], 'Visible', 'off');
    
    curImg = 0;
    ax_main = 0;
    ax_capImg = 0;
    ax_procImg = 0;
    
    blockSize = 8;
    
    url = 'http://192.168.254.159:8080/shot.jpg';
    %url = 'http://10.235.79.211:8080/shot.jpg';
    
    
    setGUI();

    %sets the GUI for all figures in the program
    function setGUI()
        %Main figure GUI
        uicontrol(fig_main, 'Style', 'pushbutton', 'String', 'Get Image', 'Units', 'Normalized', 'Position', [.2, .3, .2, .1], 'Callback', @getImage);
        uicontrol(fig_main, 'Style', 'pushbutton', 'String', 'Process Image', 'Units', 'Normalized', 'Position', [.6, .3, .2, .1], 'Callback', @processImage);
        uicontrol(fig_main, 'Style', 'pushbutton', 'String', 'Add Images to Database', 'Units', 'Normalized', 'Position', [.2, .15, .2, .1], 'Callback', @addToDatabase);
        uicontrol(fig_main, 'Style', 'pushbutton', 'String', 'Exit', 'Units', 'Normalized', 'Position', [.6, .15, .2, .1], 'Callback', @exit);
        ax_main = axes('Parent', fig_main, 'Units', 'Normalized', 'Position',[.25,.45,.5,.5]);
        ax_main.set('xtick', [], 'ytick', []);
        fig_main.set('Visible', 'on');
        
        %Capture Image figure GUI
        uicontrol(fig_capImg, 'Style', 'pushbutton', 'String', 'From Camera', 'Units', 'Normalized', 'Position', [.1, .2, .2, .1], 'Callback', @fromCamera);
        uicontrol(fig_capImg, 'Style', 'pushbutton', 'String', 'From File', 'Units', 'Normalized', 'Position', [.4, .2, .2, .1], 'Callback', @fromFile);
        uicontrol(fig_capImg, 'Style', 'pushbutton', 'String', 'Back', 'Units', 'Normalized', 'Position', [.7, .2, .2, .1], 'Callback', @backToMain);
        ax_capImg = axes('Parent', fig_capImg, 'Units', 'Normalized', 'Position',[.25,.4,.5,.5]);
        ax_capImg.set('xtick', [], 'ytick', []);
        
        %Process Image figure GUI
        uicontrol(fig_procImg, 'Style', 'pushbutton', 'String', 'Process Image', 'Units', 'Normalized', 'Position', [.2, .2, .2, .1], 'Callback', @procImg);
        uicontrol(fig_procImg, 'Style', 'pushbutton', 'String', 'Back', 'Units', 'Normalized', 'Position', [.6, .2, .2, .1], 'Callback', @backToMain);
        ax_procImg = axes('Parent', fig_procImg, 'Units', 'Normalized', 'Position',[.25,.4,.5,.5]);
        ax_procImg.set('xtick', [], 'ytick', []);
        
        %Add to Database figure GUI
        uicontrol(fig_addData, 'Style', 'pushbutton', 'String', 'Add Images to Database', 'Units', 'Normalized', 'Position', [.2, .2, .2, .1], 'Callback', @addData);
        uicontrol(fig_addData, 'Style', 'pushbutton', 'String', 'Back', 'Units', 'Normalized', 'Position', [.6, .2, .2, .1], 'Callback', @backToMain);
    end
    
    %Figure - captures image for processing
    function getImage(~,~,~)
        fig_main.set('Visible', 'off');
        fig_capImg.set('Visible', 'on');
    end
        
    %Gets an image wirelessly from phone camera
    function fromCamera(~,~,~)
        curImg = webread(url);
        image(curImg, 'Parent', ax_capImg);
        image(curImg, 'Parent', ax_procImg);
        image(curImg, 'Parent', ax_main);
    end
    
    %Gets an image from a file
    function fromFile(~,~,~)
        [fileName,pathName] = uigetfile({'*.*'},'Select an image');
        curImg = imread(strcat(pathName,fileName));
        image(curImg, 'Parent', ax_capImg);
        image(curImg, 'Parent', ax_procImg);
        image(curImg, 'Parent', ax_main);
    end
    
    %Figure - processes captured image
    function processImage(~,~,~)
        fig_main.set('Visible', 'off');
        fig_procImg.set('Visible', 'on');
    end

    %Processes the image
    function procImg(~,~,~)
        %Finds and isolates the main subject
        [imgMask, MSDerode] = detectSubject(curImg, 'true');
        %Uses shape context on the edge-detected image
        %to create a histogram of point locations
        histogram = findHistogram(imgMask, MSDerode, 'true');
        
        %Retrieves all example histograms
        fnames = dir('Histograms\*.png');
        numEx = length(fnames);
        histExList = cell(1,numEx);
        similarities = zeros(numEx);
        for i = 1:numEx
            cd('Histograms');
            curEx = imread(fnames(i).name);
            curEx = double(curEx)/255;
            histExList{i} = curEx;
            similarities(i) = compareHist(histogram, curEx);
            cd('../');
        end
        similarities = similarities(:,1);
        [~, indices] = sort(similarities, 'ascend');
        out = strcat('Closest match is:', fnames(indices(1)).name(1:end-4), ', next closest are:', fnames(indices(2)).name(1:end-4), ' and:', fnames(indices(3)).name(1:end-4));
        fig_out = figure('Units', 'Normalized', 'MenuBar', 'none', 'Name', 'Result', 'NumberTitle', 'off', 'ToolBar', 'none');
        uicontrol(fig_out, 'Style', 'text', 'String', out, 'Units', 'Normalized', 'Position', [0, .5, 1, .5]);
    end

    function [imgMask, MSDerode] = detectSubject(img, bOutput)
        numMSDIter = 1;
        rect = [1, 1, 0, 0];
        %sizeImg = size(curImg);
        %centroid = [sizeImg(1)/2, sizeImg(2)/2];
        
        for i = 1:numMSDIter
            %Get each feature map
            map_lightDist = getLightDistMap(img, rect);
            map_colorDist = getColorDistMap(img, rect);
            map_contrast = getContrastMap(img);
            map_sharpness = getSharpnessMap(img);
            map_edgeStrength = getEdgeStrengthMap(img);
        
            mapArray = {map_lightDist, map_colorDist, map_contrast, map_sharpness, map_edgeStrength};
            
            for j = 1:5
                %Normalizes the map to be in the range [0,1]
                map = mapArray{j};
                maxVal = max(max(max(map)));
                minVal = min(min(min(map)));
                mapArray{j} = (map - minVal) ./ (maxVal - minVal);
            end
            
            %mapArray = modCenterweight(mapArray, centroid);
            
            %Weigh and combine feature maps
            [map_MSD, densityList] = weighMapImportance(mapArray, 'least');
            
            %props = regionprops(map_MSD, 'Centroid');
            %centroid = props.Centroid;
            
            %Displays the output
            if bOutput
                figure(fig_output);
                fig_output.set('Visible','on');
                subplot(3,3,6*i-5), imshow(mapArray{1}), title(densityList(1));
                subplot(3,3,6*i-4), imshow(mapArray{2}), title(densityList(2));
                subplot(3,3,6*i-3), imshow(mapArray{3}), title(densityList(3));
                subplot(3,3,6*i-2), imshow(mapArray{4}), title(densityList(4));
                subplot(3,3,6*i-1), imshow(mapArray{5}), title(densityList(5));
                subplot(3,3,6*i), imshow(map_MSD);
            end
            
            %props = regionprops(MSDerode, 'BoundingBox');
            %bb = props.BoundingBox;
            %centroid = [bb(1) + bb(3)/2, bb(2) + bb(4)/2];
        end
        
        %Creates a mask from the MSD map to use on the original image
        imgMean = mean2(map_MSD);
        
        MSDBW = im2bw(map_MSD, 1.5*imgMean);
        MSDBW = bwareaopen(MSDBW, 500);
        
        se = strel('disk',12);
        MSDerode = imdilate(MSDBW, se);
        
        %Crops the image to be the size of the map, 
        %4 pixels smaller in width and height
        imgSize = size(curImg);
        cropRect = [3, 3, imgSize(2)-5, imgSize(1)-5];
        imgCrop = imcrop(curImg, 'sRGB', cropRect);
        
        imgR = imgCrop(:,:,1);
        imgG = imgCrop(:,:,2);
        imgB = imgCrop(:,:,3);
        
        %Uses the mask to remove the background of the image
        MSDerode = imfill(MSDerode, 'holes');
        MSDerode = extractNLargestBlobs(MSDerode, 1);
        
        RMask = bsxfun(@times, im2double(imgR), MSDerode);
        GMask = bsxfun(@times, im2double(imgG), MSDerode);
        BMask = bsxfun(@times, im2double(imgB), MSDerode);
        
        imgMask = cat(3, RMask, GMask, BMask);
        
        %Displays the output
        if bOutput
            subplot(3,3,7), imshow(MSDerode);
            subplot(3,3,8), imshow(curImg);
            subplot(3,3,9), imshow(imgMask);
        end
    end

    function histogram = findHistogram(img, MSDerode, bOutput)
        %Gets the edges of the main subject, filtering extraneous data
        imgMaskGray = rgb2gray(img);
        imgEdge = edge(imgMaskGray, 'prewitt');
        se2 = strel('disk', 2);
        edgeMask = imerode(MSDerode, se2);
        imgEdge = bsxfun(@times, im2double(imgEdge), edgeMask);
        imgEdge = bwareaopen(imgEdge, 20);
        
        %Finds the centroid of the edge-detected image
        sizeEdge = size(imgEdge);
        x = 1;
        y = 1;
        weight = [0, 0];
        numPts = 0;
        xList = [0,0];
        yList = [0, 0];
        
        while x < sizeEdge(1)
            while y < sizeEdge(2)
                if imgEdge(x, y) == 1
                    weight(1) = weight(1) + x;
                    weight(2) = weight(2) + y;
                    numPts = numPts + 1;
                    xList(numPts) = x;
                    yList(numPts) = y;
                end
                
                y = y + 1;
            end
            
            x = x + 1;
            y = 1;
        end
        
        edgeCentroid = floor(weight./numPts);
        
        %Determines the scale to use for binning based on
        %the maximum distance of the points, excluding the
        %top 5% to account for outliers
        distList = [0, 0];
        
        for i = 1:numPts
            pt = [xList(i), yList(i)];
            dist = sqrt((pt(1) - edgeCentroid(1)).^2 + (pt(2) - edgeCentroid(2)).^2);
            distList(i) = dist;
        end
        
        distList = sort(distList, 'ascend');
        distListTrunc = distList(1:floor(numPts*.95));
        distListTrunc = sort(distListTrunc, 'descend');
        
        scaleTotal = distListTrunc(1);
        
        %Sorts all points on the edges into bins based on a
        %log-polar grid, with 5 'r' bins and 12 'theta' bins
        histogram = zeros(12, 5);
        ptnum = 1;
        numRBins = 5;
        numThetaBins = 12;
        
        %Divides the log scale used for sorting based on powers of e
        n = 0;
        logScales = zeros(5);
        
        for i = 1:numRBins
            n = n + exp((i - 1));
        end
        
        scale = scaleTotal/n;
        %scale = scaleTotal/numRBins;
        
        for i = 1:numRBins
            if i == 1
                logScales(i) = scale;
            else
                logScales(i) = logScales(i - 1) + scale * exp((i - 1));
                %logScales(i) = logScales(i - 1) + scale;
            end
        end
        
        %Displays edge detected image with scale
        rgbEdge = floor(imgEdge(:,:,[1,1,1]) * 255);
        %xStart = edgeCentroid(1) - 2;
        %yStart = edgeCentroid(2) - 2;
        %rgbEdge(xStart:xStart+4, yStart:yStart+4,1) = 1;
        
        if bOutput
            figure, imshow(rgbEdge);
        end
        
        for theta = 0:30:330
            [x1,y1] = pol2cart(theta / 180 * pi , scaleTotal);
            line([edgeCentroid(2), x1 + edgeCentroid(2)], [edgeCentroid(1), y1 + edgeCentroid(1)], 'Color', 'r');
        end
        
        for i = 1:numRBins
            r = logScales(i);
            pos = [edgeCentroid(2) - r, edgeCentroid(1) - r, 2*r, 2*r];
            rectangle('Position', pos, 'Curvature', 1, 'EdgeColor', 'r');
        end
        
        %Determines which bin each point will go to and adds 1 to that bin
        while ptnum < numPts
            %Changes each point to polar coordinates
            %relative to the centroid
            pt = [xList(ptnum), yList(ptnum)];
            relCoords = edgeCentroid - pt;
            [theta, rho] = cart2pol(relCoords(1), relCoords(2));
            
            %Determines which bin the point is in
            thetaVal = floor(theta/(2*pi / numThetaBins));
            logR = 1;
            while rho > logScales(logR)
                if logR < 5
                    logR = logR + 1;
                else
                    break;
                end
            end
            histCoords = [0, 0];
            
            if thetaVal + 7 == 13
                thetaVal = 5;
            end
            
            histCoords(1) = thetaVal + 7;
            histCoords(2) = logR;
            
            histogram(histCoords(1), histCoords(2)) = histogram(histCoords(1), histCoords(2)) + 1;
            
            ptnum = ptnum + 1;
        end
        
        %Reduces the histogram to the range [0, 1] to remove
        %differences cause by differing numbers of points
        %among vaious images
        histogram = histogram./max(max(histogram));
        histogram = 1 - histogram;
        
        if bOutput
            figure, imshow(histogram);
        end
    end
    
    %Compares 2 histograms to find their similarity
    %In order to account for subject rotation each histogram
    %is shifted circularly in case the pattern appears in a different
    %position in the histograms
    function similarity = compareHist(hist1, hist2)
        sim = zeros(12,1);
        for i=1:12
            %sim(i) = sum(sum(pdist2(hist1, hist2)))
            x = 1;
            y = 1;
            while x <= 5
                while y <= 12
                    sim(i) = sim(i) + abs(hist1(y, x) - hist2(y, x));
                    y = y + 1;
                end
                y = 1;
                x = x + 1;
            end
            circshift(hist1, [1, 0]);
        end
        sim = sort(sim, 'descend');
        similarity = sim(1);
    end
    
    %Modifies each map to put more weight on the estimated center
    %of the main subject (assumed to be the center of the image
    %for the first iteration)
    %UNUSED
    function mapArray = modCenterweight(mapArray, centroid)
        for i = 1:5
            sizeImg = size(mapArray{i});
            centermap = zeros(sizeImg(1), sizeImg(2));
            
            x = 1;
            y = 1;
            while x < sizeImg(1)
                while y < sizeImg(2)
                    ptdist = sqrt((x - centroid(1)).^2 + (y - centroid(2)).^2);
                    imgdist = sqrt((sizeImg(1)/2).^2 + (sizeImg(2)/2).^2);
                    
                    if ptdist <= imgdist
                        centermap(x:x+4, y:y+4) = 1 - ptdist/imgdist;
                    else
                        centermap(x:x+4, y:y+4) = 0;
                    end
                    
                    y = y + 4;
                end
                
                x = x + 4;
                y = 1;
            end
            
            centermap = imcrop(centermap, [1,1,sizeImg(2) - 1,sizeImg(1) - 1]);
            
            mapArray{i} = bsxfun(@times, mapArray{i}, centermap);
        end
    end
    
    %Weighs each feature map based on cluster density,
    %then combines them into a single map
    %Order: 'most', 'least'
    function [MSDmap, mapDensityList] = weighMapImportance(mapArray, order)
        mapDensityList = [0, 0, 0, 0, 0];
        
        for i = 1:5
            mapMean = mean2(mapArray{i});
            
            imgBW = im2bw(mapArray{i}, 1.5*mapMean);
            sizeImg = size(imgBW);
            
            imgBW = bwareaopen(imgBW, 500);
            
            mapX = 1;
            mapY = 1;
            densitySum = 0;
            numPixAboveThresh = 0;
            
            weightedVals = [0,0];
            
            %Finds the centroid of all values above .5 in the image, as
            %well as counts the number of them
            while mapX < sizeImg(1)
                while mapY < sizeImg(2)
                    if imgBW(mapX, mapY) == 1
                        weightedVals(1) = weightedVals(1) + mapX;
                        weightedVals(2) = weightedVals(2) + mapY;
                        numPixAboveThresh = numPixAboveThresh + 1;
                    end
                    
                    mapY = mapY + 4;
                end
                    
                mapY = 1;
                mapX = mapX + 4;
            end
            
            centroid = weightedVals./numPixAboveThresh;
            
            mapX = 1;
            mapY = 1;
            
            %Finds cluster density of the image
            while mapX < sizeImg(1)
                while mapY < sizeImg(2)
                    if imgBW(mapX, mapY) == 1
                        densitySum = densitySum + sqrt((centroid(1) - mapX).^2 + (centroid(2) - mapY).^2);
                    end
                    
                    mapY = mapY + 4;
                end
                    
                mapY = 1;
                mapX = mapX + 4;
            end
            %Stores cluster density in the list
            if numPixAboveThresh > 0
                %mapDensityList(i) = densitySum/sqrt(numPixAboveThresh);
                mapDensityList(i) = densitySum/(numPixAboveThresh.^1.5);
            else
                if strcmp(order, 'most')
                    mapDensityList(i) = 0;
                else
                    if strcmp(order, 'least')
                        mapDensityList(i) = intmax;
                    end
                end
            end
        end
        
        %Weighted sum of each feature map
        [~, indices] = sort(mapDensityList);
        if strcmp(order, 'least');
            MSDmap = .5*(mapArray{indices(1)} + (2/3)*mapArray{indices(2)} + (1/3)*mapArray{indices(3)});
        else if strcmp(order, 'most')
                MSDmap = .5*(mapArray{indices(5)} + (2/3)*mapArray{indices(4)} + (1/3)*mapArray{indices(3)});
            end
        end
        
        %Normalizes the map to be in the range [0,1]
        maxVal = max(max(max(MSDmap)));
        minVal = min(min(min(MSDmap)));
        MSDmap = (MSDmap - minVal) ./ (maxVal - minVal);
    end
    
    %Gets light distance map
    %Light Distance = difference in lightness of an area from the
    %background
    function map = getLightDistMap(img, rect)
        imgLab = convert2Lab(img);
        blockX = 1;
        blockY = 1;
        sizeImg = size(imgLab);
        map = zeros(sizeImg(1) - 4, sizeImg(2) - 4);
        
        if rect == [1, 1, 0, 0]
            L_bg = mean2(imgLab(:,:,1));
        else
            L_bglist = [0, 0, 0, 0];
            L_bglist(1) = mean2(imgLab(1:rect(1),:,1));
            L_bglist(2) = mean2(imgLab(rect(1):(rect(1) + rect(2)),1:rect(2),1));
            L_bglist(3) = mean2(imgLab((rect(1) + rect(3)):sizeImg(1),:,1));
            L_bglist(4) = mean2(imgLab(rect(1):(rect(1) + rect(2)),(rect(2) + rect(4)):sizeImg(2),1));
            
            for i = 1:4
                if isnan(L_bglist(i))
                    L_bglist(i) = 0;
                end
            end
            
            L_bg = mean2(L_bglist);
        end
        
        while blockX <= sizeImg(1) - blockSize
            while blockY <= sizeImg(2) - blockSize
                L_block = mean2(imgLab(blockX:blockX + blockSize, blockY:blockY + blockSize, 1));
                blockVal = abs(L_block - L_bg)/100;
            
                map(blockX:blockX + 4, blockY:blockY + 4) = blockVal;
                blockY = blockY + 4;
            end
            
            blockY = 1;
            blockX = blockX + 4;
        end
        
        %figure, imshow(map);
    end

    %Gets the color distance
    %Color Distance = difference in color of an area from the
    %background (2D distance along a* and b* axes)
    function  map = getColorDistMap(img, rect)
        imgLab = convert2Lab(img);
        blockX = 1;
        blockY = 1;
        sizeImg = size(imgLab);
        map = zeros(sizeImg(1) - 4, sizeImg(2) - 4);
        
        if rect == [1, 1, 0, 0]
            a_bg = mean2(imgLab(:,:,2));
            b_bg = mean2(imgLab(:,:,3));
        else
            a_bglist = [0, 0, 0, 0];
            a_bglist(1) = mean2(imgLab(1:rect(1),:,2));
            a_bglist(2) = mean2(imgLab(rect(1):(rect(1) + rect(2)),1:rect(2),2));
            a_bglist(3) = mean2(imgLab((rect(1) + rect(3)):sizeImg(1),:,2));
            a_bglist(4) = mean2(imgLab(rect(1):(rect(1) + rect(2)),(rect(2) + rect(4)):sizeImg(2),2));
            
            b_bglist = [0, 0, 0, 0];
            b_bglist(1) = mean2(imgLab(1:rect(1),:,3));
            b_bglist(2) = mean2(imgLab(rect(1):(rect(1) + rect(2)),1:rect(2),3));
            b_bglist(3) = mean2(imgLab((rect(1) + rect(3)):sizeImg(1),:,3));
            b_bglist(4) = mean2(imgLab(rect(1):(rect(1) + rect(2)),(rect(2) + rect(4)):sizeImg(2),3));
            
            for i = 1:4
                if isnan(a_bglist(i))
                    a_bglist(i) = 0;
                end
                if isnan(b_bglist(i))
                    b_bglist(i) = 0;
                end
            end
            
            a_bg = mean2(a_bglist);
            b_bg = mean2(b_bglist);
        end

        while blockX <= sizeImg(1) - blockSize
            while blockY <= sizeImg(2) - blockSize
                a_block = mean2(imgLab(blockX:blockX + blockSize, blockY:blockY + blockSize, 2));
                b_block = mean2(imgLab(blockX:blockX + blockSize, blockY:blockY + blockSize, 3));
                blockVal = sqrt((a_block - a_bg)*(a_block - a_bg) + (b_block - b_bg)*(b_block - b_bg))/100;
            
                map(blockX:blockX + 4, blockY:blockY + 4) = (blockVal).^2;
                blockY = blockY + 4;
            end
            
            blockY = 1;
            blockX = blockX + 4;
        end
        
        %figure, imshow(map);
    end
    
    %Gets local contrast map
    function  map = getContrastMap(img)
        imgGray = rgb2gray(img);
        
        blockX = 1;
        blockY = 1;
        sizeImg = size(imgGray);
        map = zeros(sizeImg(1) - 4, sizeImg(2) - 4);

        while blockX <= sizeImg(1) - blockSize
            while blockY <= sizeImg(2) - blockSize
                blockImg = imgGray(blockX:blockX + blockSize, blockY:blockY + blockSize);
                blockLuminanceMap = power((.7297 + .037644*blockImg), 2);
                lumStdDev = std2(blockLuminanceMap);
                lumMean = mean2(blockLuminanceMap);
                blockVal = lumStdDev/lumMean;
            
                map(blockX:blockX + 4, blockY:blockY + 4) = blockVal;
                blockY = blockY + 4;
            end
            
            blockY = 1;
            blockX = blockX + 4;
        end
        
        %figure, imshow(map);
    end
    
    %Gets local sharpness map
    function map = getSharpnessMap(img)
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
                blockVal = sqrt(sharpness/100);
                
                map(blockX:blockX + 4, blockY:blockY + 4) = blockVal;
                
                blockY = blockY + 4;
            end
            
            blockY = 1;
            blockX = blockX + 4;
        end
        
        %figure, imshow(map);
    end

    %Gets edge strength map
    function  map = getEdgeStrengthMap(img)
        imgGray = rgb2gray(img);
        imgEdge = edge(imgGray, 'Roberts');
        
        blockX = 1;
        blockY = 1;
        sizeImg = size(imgEdge);
        map = zeros(sizeImg(1) - 4, sizeImg(2) - 4);

        while blockX <= sizeImg(1) - blockSize
            while blockY <= sizeImg(2) - blockSize
                edgeStr = mean2(imgEdge(blockX:blockX + blockSize, blockY:blockY + blockSize));
                blockVal = sqrt(sqrt(edgeStr));
            
                map(blockX:blockX + 4, blockY:blockY + 4) = blockVal;
                blockY = blockY + 4;
            end
            
            blockY = 1;
            blockX = blockX + 4;
        end
        
        %figure, imshow(map);
    end
    
    %Function for adding example data
    function addData(~,~,~)
        fig_data = figure('Units', 'Normalized', 'OuterPosition', [0, 0, 1, 1], 'MenuBar', 'none', 'Name', 'Add Examples', 'NumberTitle', 'off', 'ToolBar', 'none');
        ex = 0;
        hist = 0;
        bMakeNewEx = false;
        name = '';
        exName = '';
        uicontrol(fig_data, 'Style', 'pushbutton', 'String', 'Get Image', 'Units', 'Normalized', 'Position', [.2, .35, .2, .1], 'Callback', @getImg);
        uicontrol(fig_data, 'Style', 'pushbutton', 'String', 'Create New', 'Units', 'Normalized', 'Position', [.6, .35, .2, .1], 'Callback', @newEx);
        uicontrol(fig_data, 'Style', 'pushbutton', 'String', 'Confirm', 'Units', 'Normalized', 'Position', [.2, .15, .2, .1], 'Callback', @confirmEx);
        uicontrol(fig_data, 'Style', 'pushbutton', 'String', 'Back', 'Units', 'Normalized', 'Position', [.6, .15, .2, .1], 'Callback', @back);

        function getImg(~,~,~)
            [fileName,pathName] = uigetfile({'*.*'},'Select an image');
            curImg = imread(strcat(pathName, fileName));
            
            [imgMask, MSDerode] = detectSubject(curImg, false);
            hist = findHistogram(imgMask, MSDerode, false);
            
            subplot(2,2,1), imshow(curImg);
            subplot(2,2,2), imshow(hist);
        end
        
        function newEx(~,~,~)
            fig_newEx = figure('Units', 'Normalized', 'MenuBar', 'none', 'Name', 'New Example', 'NumberTitle', 'off', 'ToolBar', 'none');
            uicontrol(fig_newEx, 'Style', 'pushbutton', 'String', 'Ok', 'Units', 'Normalized', 'Position', [.2, .2, .2, .1], 'Callback', @confNewEx);
            uicontrol(fig_newEx, 'Style', 'pushbutton', 'String', 'Cancel', 'Units', 'Normalized', 'Position', [.6, .2, .2, .1], 'Callback', @back);
            text = uicontrol(fig_newEx, 'Style', 'edit', 'String', '', 'Units', 'Normalized', 'Position', [.2, .4, .6, .1]);
            
            function confNewEx(~,~,~)
                name = text.String;
                back();
                bMakeNewEx = true;
            end
        end
        
        function confirmEx(~,~,~)
            if bMakeNewEx
                %imwrite(hist, name, 'png');
                cd('Histograms');
                imwrite(hist, strcat(name, '.png'), 'png');
                cd('../');
            elseif size(ex) ~= 0
                numInEx = exName;
                ex = ex * (numInEx / (numInEx + 1)) + hist;
            end
        end
        
        function back(~,~,~)
            close gcf;
        end
    end

    %Converts an image from rgb to L*a*b*
    function imgLab = convert2Lab(img)
        colorTransform = makecform('srgb2lab');
        imgLab = applycform(img, colorTransform);
    end

    function binaryImage = extractNLargestBlobs(binaryImage, numberToExtract)
        try
            % Get all the blob properties.  Can only pass in originalImage in version R2008a and later.
        	[labeledImage, ~] = bwlabel(binaryImage);
            blobMeasurements = regionprops(labeledImage, 'area');
        	% Get all the areas
            allAreas = [blobMeasurements.Area];
            % For positive numbers, sort in order of largest to smallest.
            % Sort them.
            [~, sortIndexes] = sort(allAreas, 'descend');
            % Extract the "numberToExtract" largest blob(a)s using ismember().
            biggestBlob = ismember(labeledImage, sortIndexes(1:numberToExtract));
            % Convert from integer labeled image into binary (logical) image.
            binaryImage = biggestBlob > 0;
        catch ME
            errorMessage = sprintf('Error in function ExtractNLargestBlobs().\n\nError Message:\n%s', ME.message);
            fprintf(1, '%s\n', errorMessage);
            uiwait(warndlg(errorMessage));
        end
    end
    
    %Figure - adds images to the database for comparison
    function addToDatabase (~,~,~)
        fig_main.set('Visible', 'off');
        fig_addData.set('Visible', 'on');
    end

    %Returns to the main figure
    function backToMain(~,~,~)
        fig_capImg.set('Visible', 'off');
        fig_procImg.set('Visible', 'off');
        fig_addData.set('Visible', 'off');
        
        fig_main.set('Visible', 'on');
    end

    %Exits the program
    function exit(~,~,~)
        close(fig_main);
    end
end