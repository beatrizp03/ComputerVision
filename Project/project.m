clear all, close all;

% ----- USER INPUT -----
disp('Choose an operation:');
disp('1 - Display Ground-Truth Bounding Boxes');
disp('2 - Perform Pedestrian Tracking');
disp('3 - Perform Trajectory Tracking');
disp('4 - Perform Pedestrian Tracking with Consistent Labels');
disp('5 - Perform Heatmap Computing');
disp('6 - Perform EM Analysis of Pedestrian Trajectories');
disp('7 - Perform Pedestrian Tracking Algorithm Evaluation');
choice = input('Enter your choice (number of choice): ');

% ----- COMMON VARIABLES -----

% change "Your_path" to the actual path to the directories
framesDir = "Your_path\\Time_12-34\\View_001\\";
gtFile = "Your_path\\PETS-S2L1\\gt\\gt.txt";

totalFrames = 794;
totalFramesForBackground = 50;
thresholdValue = 40;
minArea = 200;
se = strel('disk', 2);
maxTrajectoryLength = 25;
maxDisappeared = 10; 

% ----- BACKGROUND COMPUTATION -----
frameIndices = round(linspace(1, 794, totalFramesForBackground)); 
sampleFrame = imread(sprintf("%sframe_%04d.jpg", framesDir, 1));
[rows, cols, channels] = size(sampleFrame);
allFrames = zeros(rows, cols, channels, totalFramesForBackground, 'uint8');

for i = 1:totalFramesForBackground
    frameFilename = sprintf("%sframe_%04d.jpg", framesDir, frameIndices(i));
    allFrames(:, :, :, i) = imread(frameFilename);
end

cleanBackground = median(allFrames, 4);

trajectoryData = struct('ID', {}, 'Trajectory', {});
trajectoryDataMerged = struct('ID', {}, 'Centroid', {}, 'BoundingBox', {}, 'Trajectory', {}, 'Disappeared', {}, 'Merged', {});
colors = containers.Map('KeyType', 'double', 'ValueType', 'any');
colorsMerged = containers.Map('KeyType', 'char', 'ValueType', 'any'); % Store colors with string IDs
globalID = 1;

% ----- SWITCH CASE -----
switch choice
    case 1
        displayGroundTruth(gtFile, framesDir, totalFrames);
        
    case 2
        basicPedestrianTracking(framesDir, totalFrames, cleanBackground, thresholdValue, minArea, se, gtFile);
        
    case 3
        advancedTrajectoryTracking(framesDir, totalFrames, cleanBackground, thresholdValue, minArea, se, maxTrajectoryLength, rows, cols,trajectoryData,colors,globalID);
        
    case 4
        consistentLabeling(framesDir, totalFrames, cleanBackground, thresholdValue, minArea, se, maxTrajectoryLength, rows, cols, trajectoryDataMerged, colorsMerged, globalID, maxDisappeared, gtFile);
    
    case 5
        heatmapComputing(framesDir, totalFrames, cleanBackground, thresholdValue, minArea, se, gtFile, cols, rows);

    case 6
        emAnalysis(gtFile);

    case 7
        evaluatePerformance(framesDir, totalFrames, cleanBackground, thresholdValue, minArea, se, gtFile);
        
    otherwise
        disp('Invalid choice. Please enter 1, 2, 3, 4, 5, 6 or 7.');
end

function displayGroundTruth(gtFile, framesDir, totalFrames)
    gtData = readmatrix(gtFile);
    frameIdxCol = 1;
    bbLeftCol = 3;
    bbTopCol = 4;
    bbWidthCol = 5;
    bbHeightCol = 6;

    for frameNumber = 1:totalFrames
        frameFilename = sprintf('%sframe_%04d.jpg', framesDir, frameNumber);
        img = imread(frameFilename);
        boxes = gtData(gtData(:, frameIdxCol) == frameNumber, :);

        figure(1); imshow(img); hold on;
        for i = 1:size(boxes, 1)
            rectangle('Position', [boxes(i, bbLeftCol), boxes(i, bbTopCol), boxes(i, bbWidthCol), boxes(i, bbHeightCol)], ...
                      'EdgeColor', 'g', 'LineWidth', 2);
        end
        title(['Frame: ' num2str(frameNumber)]);
        hold off;
        pause(0.05);
    end
end

function basicPedestrianTracking(framesDir, totalFrames, cleanBackground, thresholdValue, minArea, se, gtFile)
    gtData = readmatrix(gtFile);
    frameIdxCol = 1;
    bbLeftCol = 3;
    bbTopCol = 4;
    bbWidthCol = 5;
    bbHeightCol = 6;

    figure;
    for frameNumber = 1:totalFrames
        frameFilename = sprintf("%sframe_%04d.jpg", framesDir, frameNumber);
        currentFrame = imread(frameFilename);

        % display ground truth
        subplot(1,2,1); imshow(currentFrame); hold on;
        boxes = gtData(gtData(:, frameIdxCol) == frameNumber, :);
        for i = 1:size(boxes, 1)
            rectangle('Position', [boxes(i, bbLeftCol), boxes(i, bbTopCol), boxes(i, bbWidthCol), boxes(i, bbHeightCol)], ...
                      'EdgeColor', 'g', 'LineWidth', 2);
        end
        title('Ground Truth');
        hold off;

        % Pedestrian Detection on Right
        diffImage = (abs(double(cleanBackground(:,:,1)) - double(currentFrame(:,:,1))) > thresholdValue) | ...
                    (abs(double(cleanBackground(:,:,2)) - double(currentFrame(:,:,2))) > thresholdValue) | ...
                    (abs(double(cleanBackground(:,:,3)) - double(currentFrame(:,:,3))) > thresholdValue);

        binaryMask = imclose(diffImage, se);
        binaryMask = imerode(binaryMask, se);

        [labeledImage, numBlobs] = bwlabel(binaryMask);
        regionProps = regionprops(labeledImage, 'Area', 'BoundingBox', 'Centroid');
        validRegions = find([regionProps.Area] > minArea);
        regionNum = length(validRegions);

        subplot(1,2,2); imshow(currentFrame); hold on;
        title('Detections');

        labelCounter = 1;
        for idx = 1:regionNum
            [line, column] = find(labeledImage == validRegions(idx));
            upLeftPoint = min([line column]);
            dWindow  = max([line column]) - upLeftPoint + 1;

            rectangle('Position',[fliplr(upLeftPoint) fliplr(dWindow)],'EdgeColor',[1 1 0], 'linewidth',2);
    
            label = sprintf('%d', labelCounter);
            text(upLeftPoint(2), upLeftPoint(1) - 30, label, 'Color', 'yellow', 'FontSize', 12);
            labelCounter = labelCounter + 1;
        end
        hold off;
        drawnow;
    end
end

function advancedTrajectoryTracking(framesDir, totalFrames, cleanBackground, thresholdValue, minArea, se, maxTrajectoryLength, rows, cols, trajectoryData,colors,globalID)
     % ----- PROCESSING AND PLOTTING -----
    figure;
    for frameNumber = 1:totalFrames
        frameFilename = sprintf("%sframe_%04d.jpg", framesDir, frameNumber);
        currentFrame = imread(frameFilename);
        
        % Image Processing to Detect Motion
        diffImage = (abs(double(cleanBackground(:,:,1)) - double(currentFrame(:,:,1))) > thresholdValue) | ...
                    (abs(double(cleanBackground(:,:,2)) - double(currentFrame(:,:,2))) > thresholdValue) | ...
                    (abs(double(cleanBackground(:,:,3)) - double(currentFrame(:,:,3))) > thresholdValue);
        
        binaryMask = imclose(diffImage, se);
        binaryMask = imerode(binaryMask, se);
        
        [labeledImage, ~] = bwlabel(binaryMask);
        regionProps = regionprops(labeledImage, 'Area', 'BoundingBox', 'Centroid');
        validRegions = find([regionProps.Area] > minArea);
        
        currentIDs = [];
        subplot(1,2,1); imshow(currentFrame); title('Detections');
        
        % --- Assign IDs Dynamically Based on Current Frame ---
        for idx = 1:length(validRegions)
            centroid = regionProps(validRegions(idx)).Centroid;
            position = centroid(1) + 1i * (rows - centroid(2)); 
            
            % Check for proximity to existing trajectory to maintain ID
            foundMatch = false;
            for i = 1:length(trajectoryData)
                lastPosition = trajectoryData(i).Trajectory(end);
                if ~isnan(real(lastPosition)) && abs(position - lastPosition) < 30
                    % Update trajectory
                    trajectoryData(i).Trajectory = [trajectoryData(i).Trajectory(2:end), position];
                    currentIDs = [currentIDs, trajectoryData(i).ID];
                    foundMatch = true;
                    break;
                end
            end
            
            if ~foundMatch
                % Create a new ID for a newly detected object
                trajectoryData(end+1) = struct('ID', globalID, ...
                                               'Trajectory', [repmat(NaN + 1i*NaN, 1, maxTrajectoryLength-1), position]);
                colors(globalID) = rand(1, 3);
                currentIDs = [currentIDs, globalID];
                globalID = globalID + 1;
            end
        end
        
        % Remove trajectories not detected in the current frame
        trajectoryData = trajectoryData(ismember([trajectoryData.ID], currentIDs));
        
        % Draw bounding boxes and IDs dynamically for the current frame
        for idx = 1:length(validRegions)
            centroid = regionProps(validRegions(idx)).Centroid;
            position = centroid(1) + 1i * (rows - centroid(2));
            
            % Find corresponding trajectory for drawing
            matchID = [];
            for i = 1:length(trajectoryData)
                if abs(trajectoryData(i).Trajectory(end) - position) < 1e-3
                    matchID = trajectoryData(i).ID;
                    break;
                end
            end
            
            if ~isempty(matchID)
                boundingBox = regionProps(validRegions(idx)).BoundingBox;
                rectangle('Position', boundingBox, 'EdgeColor', colors(matchID), 'LineWidth', 2);
                text(boundingBox(1), boundingBox(2) - 10, sprintf('%d', idx), ...
                    'Color', colors(matchID), 'FontSize', 12);
            end
        end
        
        % --- Plot dynamic trajectories ---
        subplot(1,2,2); cla;
        for i = 1:length(trajectoryData)
            validPositions = trajectoryData(i).Trajectory(~isnan(real(trajectoryData(i).Trajectory)));
            if ~isempty(validPositions)
                scatter(real(validPositions), imag(validPositions), 10, colors(trajectoryData(i).ID), 'filled'); hold on;
            end
        end
        xlim([0 cols]); ylim([0 rows]); hold off;
        
        drawnow;
    end
end

function consistentLabeling(framesDir, totalFrames, cleanBackground, thresholdValue, minArea, se, maxTrajectoryLength, rows, cols, trajectoryDataMerged, colorsMerged, globalID, maxDisappeared, gtFile)

    % Load Ground Truth Data
    gtData = readmatrix(gtFile);  
    groundTruthLabels = containers.Map('KeyType', 'char', 'ValueType', 'any'); % Store GT histograms
    matchingLabels = containers.Map('KeyType', 'double', 'ValueType', 'char'); % Our predicted labels
    
    cmap = lines(100); % Generate 100 unique colors
    figure; % Create figure window

    for frameNumber = 1:totalFrames
        frameFilename = sprintf("%sframe_%04d.jpg", framesDir, frameNumber);
        currentFrame = imread(frameFilename);

        % ---- Foreground Extraction ----
        diffImage = (abs(double(cleanBackground(:,:,1)) - double(currentFrame(:,:,1))) > thresholdValue) | ...
                    (abs(double(cleanBackground(:,:,2)) - double(currentFrame(:,:,2))) > thresholdValue) | ...
                    (abs(double(cleanBackground(:,:,3)) - double(currentFrame(:,:,3))) > thresholdValue);

        binaryMask = imclose(diffImage, se);
        binaryMask = imerode(binaryMask, se);

        % ---- Extract Ground Truth Labels ----
        gtLabelsInFrame = gtData(gtData(:, 1) == frameNumber, :);
        
        % **Skip iteration if no GT labels exist for this frame**
        if isempty(gtLabelsInFrame) || size(gtLabelsInFrame, 2) < 6
            continue;
        end
        
        gtBoundingBoxes = gtLabelsInFrame(:, 3:6);  % Extract bounding boxes
        gtIDs = gtLabelsInFrame(:, 2);  % Extract ground truth labels
        
        for i = 1:size(gtLabelsInFrame, 1)
            gtLabelID = num2str(gtIDs(i)); % Convert GT ID to string
            
            % **Ensure bounding box indices are valid**
            bb = gtBoundingBoxes(i, :); 
            
            % **Extract GT Pedestrian & Compute Histogram**
            croppedPedestrian = imcrop(currentFrame, bb);
            if isempty(croppedPedestrian) % Ensure valid cropping
                continue;
            end
            grayPedestrian = rgb2gray(croppedPedestrian);
            histValues = imhist(grayPedestrian);
            histValues = histValues / sum(histValues); % Normalize histogram
            
            % **Store in Ground Truth Database**
            groundTruthLabels(gtLabelID) = histValues;
        end

        % ---- Object Detection (Same as Function 3) ----
        [labeledImage, ~] = bwlabel(binaryMask);
        regionProps = regionprops(labeledImage, 'Area', 'BoundingBox', 'Centroid');
        validRegions = find([regionProps.Area] > minArea);

        currentIDs = [];

        % ---- Subplots: 1x2 Layout ----
        subplot(1,2,1); cla; imshow(currentFrame); title('Ground Truth Labels'); hold on;
        subplot(1,2,2); cla; imshow(currentFrame); title('Final Matching Labels'); hold on;

        for idx = 1:length(validRegions)
            centroid = regionProps(validRegions(idx)).Centroid;
            boundingBox = regionProps(validRegions(idx)).BoundingBox;

            % **Extract pedestrian region**
            croppedPedestrian = imcrop(currentFrame, boundingBox);
            if isempty(croppedPedestrian) % Ensure valid cropping
                continue;
            end
            grayPedestrian = rgb2gray(croppedPedestrian);
            histValues = imhist(grayPedestrian);
            histValues = histValues / sum(histValues); % Normalize histogram

            % ---- Find Best Match in Ground Truth ----
            minDiff = inf;
            bestMatch = '';

            for gtKey = keys(groundTruthLabels)
                gtHist = groundTruthLabels(gtKey{1});
                dist = sqrt(sum((gtHist - histValues).^2)); % Euclidean Distance
                
                if dist < minDiff
                    minDiff = dist;
                    bestMatch = gtKey{1}; % Assign GT label
                end
            end

            % **Store the matched label**
            currentID = str2double(bestMatch);
            currentIDs = [currentIDs, currentID];
            matchingLabels(currentID) = bestMatch;

            % ---- Assign Unique Colors for Each Label ----
            if ~isKey(colorsMerged, bestMatch)
                colorsMerged(bestMatch) = cmap(mod(str2double(bestMatch), 100) + 1, :);
            end
            labelColor = colorsMerged(bestMatch);

            % **Find the correct GT label for this bounding box**
            [~, minIdx] = min(vecnorm(gtBoundingBoxes - boundingBox, 2, 2)); % Find closest bounding box
            correctGTLabel = num2str(gtIDs(minIdx));  % Get the correct label

            % **Ground Truth Label Plotting (Subplot 1)**
            subplot(1,2,1); hold on;
            rectangle('Position', gtBoundingBoxes(minIdx, :), 'EdgeColor', 'b', 'LineWidth', 2);
            text(gtBoundingBoxes(minIdx, 1), gtBoundingBoxes(minIdx, 2) - 20, correctGTLabel, 'Color', 'blue', 'FontSize', 12, 'FontWeight', 'bold');
            
            % **Final Matching Labels Plotting (Subplot 2)**
            subplot(1,2,2); hold on;
            rectangle('Position', boundingBox, 'EdgeColor', labelColor, 'LineWidth', 2);
            text(boundingBox(1), boundingBox(2) - 20, bestMatch, 'Color', labelColor, 'FontSize', 12, 'FontWeight', 'bold');
        end

        drawnow; % Update figure
        pause(0.1); % Ensure the plot updates
    end
end

function heatmapComputing(framesDir, totalFrames, cleanBackground, thresholdValue, minArea, se, gtFile, cols, rows)
    % Initialize heatmap structure
    heatmapCols = 50; heatmapColsConstant = heatmapCols / cols;
    heatmapRows = 25; heatmapRowsConstant = heatmapRows / rows;
    heatmapMatrix = zeros(heatmapRows, heatmapCols);
    % Matrix with positions for Gaussian distance computing
    [heatmapX, heatmapY] = meshgrid(1:heatmapCols, 1:heatmapRows);
    heatmapMatrixPositions = heatmapY * 1i + heatmapX;
    sigma = 1.0;
    % Dynamic heatmap structure
    maxHeatmapMemory = 300;
    dynamicHeatmap = zeros(maxHeatmapMemory, heatmapRows, heatmapCols);
    dynamicHeatmapIndex = 0;

    for frameNumber = 1:totalFrames
        figure(1);
        frameFilename = sprintf("%sframe_%04d.jpg", framesDir, frameNumber);
        currentFrame = imread(frameFilename);

        % display original image
        subplot(2,2,1); imshow(currentFrame);

        % Pedestrian Detection on Right
        diffImage = (abs(double(cleanBackground(:,:,1)) - double(currentFrame(:,:,1))) > thresholdValue) | ...
                    (abs(double(cleanBackground(:,:,2)) - double(currentFrame(:,:,2))) > thresholdValue) | ...
                    (abs(double(cleanBackground(:,:,3)) - double(currentFrame(:,:,3))) > thresholdValue);

        binaryMask = imclose(diffImage, se);
        binaryMask = imerode(binaryMask, se);

        [labeledImage, numBlobs] = bwlabel(binaryMask);
        regionProps = regionprops(labeledImage, 'Area', 'BoundingBox', 'Centroid');
        validRegions = find([regionProps.Area] > minArea);
        regionNum = length(validRegions);

        subplot(2,2,2); cla;
        imshow(currentFrame); hold on;
        title('Detections');

        labelCounter = 1;
        for idx = 1:regionNum
            subplot(2,2,2);
            [line, column] = find(labeledImage == validRegions(idx));
            upLeftPoint = min([line column]);
            dWindow  = max([line column]) - upLeftPoint + 1;

            rectangle('Position',[fliplr(upLeftPoint) fliplr(dWindow)],'EdgeColor',[1 1 0], 'linewidth',2);
    
            label = sprintf('%d', labelCounter);
            text(upLeftPoint(2), upLeftPoint(1) - 30, label, 'Color', 'yellow', 'FontSize', 12);
            labelCounter = labelCounter + 1;

            subplot(2,2,3);
            centroidX = regionProps(validRegions(idx)).Centroid(1);
            centroidY = rows - regionProps(validRegions(idx)).Centroid(2);
            
            % Add info to heatmap
            heatmapY = round((-centroidY + rows) * heatmapRowsConstant);
            heatmapX = round(centroidX * heatmapColsConstant);
            centroidPositionMatrix = ones(heatmapRows, heatmapCols) * (heatmapX + heatmapY*1i);
            heatmapIncrement = exp(-(abs(heatmapMatrixPositions - centroidPositionMatrix).^2) / (2 * sigma^2));
            heatmapMatrix = heatmapMatrix + heatmapIncrement;
            dynamicHeatmapIndex = mod(dynamicHeatmapIndex, maxHeatmapMemory) + 1;
            dynamicHeatmap(dynamicHeatmapIndex,:,:) = heatmapIncrement;
        end

        subplot(2,2,3); imagesc(heatmapMatrix); title('Static Heatmap');
        subplot(2,2,4); imagesc(squeeze(sum(dynamicHeatmap, 1))); title('Dynamic Heatmap');
        colormap hot;
        %drawnow;
    end
end

function emAnalysis(gtFile)
    data = readmatrix(gtFile);
    % Extract ground truth bounding box data
    ID = data(:,2);       
    bbox_left = data(:,3); 
    bbox_top = data(:,4);  
    bbox_width = data(:,5); 
    bbox_height = data(:,6); 
    
    % Compute bounding box centers
    bbox_center_x = bbox_left + bbox_width / 2;
    bbox_center_y = bbox_top + bbox_height / 2;
    
    % Organize data by pedestrian ID
    uniqueIDs = unique(ID);
    displacements = [];
    labels = [];
    
    % Collect displacements for each pedestrian
    for i = 1:length(uniqueIDs)
        idx = (ID == uniqueIDs(i));
        x_i = bbox_center_x(idx);
        y_i = bbox_center_y(idx);
        
        if length(x_i) > 1
            dx = diff(x_i);
            dy = diff(y_i);
    
            displacements = [displacements; dx, dy];
            
            for j = 1:length(dx)
                if abs(dx(j)) > abs(dy(j))
                    if dx(j) > 0
                        labels = [labels; "L-R"];
                    else
                        labels = [labels; "R-L"];
                    end
                else
                    if dy(j) > 0
                        labels = [labels; "B-T"];
                    else
                        labels = [labels; "T-B"];
                    end
                end
            end
        end
    end
    
    % Define the direction groups
    directionGroups = {'L-R', 'R-L', 'B-T', 'T-B'};
    
    % EM Algorithm Parameters
    maxIter = 50;
    tol = 1e-6;
    
    X = displacements;
    
    % Number of clusters
    K = 4;  
    numData = size(X, 1);
    dim = size(X, 2);
    
    % Initialize parameters
    mu =  3*[cos(linspace(0, 2*pi, K+1))', sin(linspace(0, 2*pi, K+1))'];
    mu = mu(1:end-1,:);
    sigma = repmat(eye(dim), [1, 1, K]);
    
    % EM Algorithm
    log_likelihood_old = -inf;
    for iter = 1:maxIter
        % E-step: Calculate responsibilities (Wmn)
        Wmn = zeros(numData, K);
        
        for k = 1:K
            invSigma = pageinv(sigma(:,:,k));
            dif = X - mu(k, :);
            coeff = 1 / ((2 * pi)^(dim / 2) * sqrt(det(sigma(:,:,k))));
            Wmn(:, k) = coeff * exp(-0.5 * diag(dif * invSigma * dif'));
        end
        
        Wmn = Wmn ./ sum(Wmn, 2);
    
        % M-step: Update parameters
        Nk = sum(Wmn,1);
        for k = 1:K
            mu(k, :) = sum(Wmn(:,k) .* X, 1) / Nk(k);
            dif = X - mu(k, :);
            sigma(:,:,k) = (dif' * (dif .* Wmn(:,k))) / Nk(k);
            sigma(:,:,k) = sigma(:,:,k) + 1e-6 * eye(dim);
        end
        pi_k = Nk / numData;
        log_likelihood = sum(log(sum(Wmn, 2)));
        if abs(log_likelihood - log_likelihood_old) < tol
            break;
        end
        log_likelihood_old = log_likelihood;
    end
    
    % Assign points to clusters
    [~, idx] = max(Wmn, [], 2);
    
    % Plotting
    figure; hold on;
    colors = lines(K);
    
    % Scatter the points by cluster
    for k = 1:K
        scatter(X(idx==k,1), X(idx==k,2), 20, colors(k,:), 'filled');
    end
    scatter(mu(:,1), mu(:,2), 100, 'kx', 'LineWidth', 2);
    
    % Plotting Axes
    plot([0 0], ylim, 'k--', 'LineWidth', 1);
    plot(xlim, [0 0], 'k--', 'LineWidth', 1);
    legend("Cluster" + (1:K));
    
    title('EM Clustering of Displacement Trajectories by Direction');
    xlabel('X Displacement');
    ylabel('Y Displacement');
    hold off;
    
end

function evaluatePerformance(framesDir, totalFrames, cleanBackground, thresholdValue, minArea, se, gtFile) 
    gtData = readmatrix(gtFile);
    frameIdxCol = 1;
    bbLeftCol = 3;
    bbTopCol = 4;
    bbWidthCol = 5;
    bbHeightCol = 6;
    
    % Bounding box Threshold parameters
    minimumThreshold = 0.1;
    maximumThreshold = 0.9;
    incrementThreshold = 0.1;
    nThresholdValues = ((maximumThreshold - minimumThreshold) / incrementThreshold)+1;
    percentageCoveredFrames = zeros(1, nThresholdValues);
    percentageFalsePositives = zeros(1, nThresholdValues);
    percentageFalseNegatives = zeros(1, nThresholdValues);
    
    figure;
    for threshold = minimumThreshold:incrementThreshold:maximumThreshold
        % Binary statistical data
        statisticalData = repmat(struct("falsePositives", 0, ...
                                     "falseNegatives", 0, ...
                                     "detectedLabels", 0,...
                                     "splits", 0, ...
                                     "merges", 0), ...
                                     1, totalFrames);
        for frameNumber = 1:totalFrames
            frameFilename = sprintf("%sframe_%04d.jpg", framesDir, frameNumber);
            currentFrame = imread(frameFilename);
        
            % display ground truth
            subplot(1,2,1); imshow(currentFrame); hold on;
            boxes = gtData(gtData(:, frameIdxCol) == frameNumber, :);
        
            gtBoxes = zeros(size(boxes,1), 4);
            for i = 1:size(boxes, 1)
                rectangle('Position', [boxes(i, bbLeftCol), boxes(i, bbTopCol), boxes(i, bbWidthCol), boxes(i, bbHeightCol)], ...
                          'EdgeColor', 'g', 'LineWidth', 2);
                gtBoxes(i,:) = [boxes(i, bbLeftCol), boxes(i, bbTopCol), boxes(i, bbWidthCol), boxes(i, bbHeightCol)];
            end
            title('Ground Truth');
            hold off;
        
            % Pedestrian Detection on Right
            diffImage = (abs(double(cleanBackground(:,:,1)) - double(currentFrame(:,:,1))) > thresholdValue) | ...
                        (abs(double(cleanBackground(:,:,2)) - double(currentFrame(:,:,2))) > thresholdValue) | ...
                        (abs(double(cleanBackground(:,:,3)) - double(currentFrame(:,:,3))) > thresholdValue);
        
            binaryMask = imclose(diffImage, se);
            binaryMask = imerode(binaryMask, se);
        
            [labeledImage, numBlobs] = bwlabel(binaryMask);
            regionProps = regionprops(labeledImage, 'Area', 'BoundingBox', 'Centroid');
            validRegions = find([regionProps.Area] > minArea);
            regionNum = length(validRegions);
        
            subplot(1,2,2); imshow(currentFrame); hold on;
            title('Detections');
        
            labelCounter = 1;
            detectedBoxes = zeros(regionNum, 4);
            for idx = 1:regionNum
                [line, column] = find(labeledImage == validRegions(idx));
                upLeftPoint = min([line column]);
                dWindow  = max([line column]) - upLeftPoint + 1;
        
                rectangle('Position',[fliplr(upLeftPoint) fliplr(dWindow)],'EdgeColor',[1 1 0], 'linewidth',2);
        
                label = sprintf('%d', labelCounter);
                text(upLeftPoint(2), upLeftPoint(1) - 30, label, 'Color', 'yellow', 'FontSize', 12);
                labelCounter = labelCounter + 1;
                detectedBoxes(idx,:) = [fliplr(upLeftPoint) fliplr(dWindow)];
            end
            hold off;
            drawnow;
        
            % Data analysis
            intersectionMatrix = zeros(size(gtBoxes,1), size(detectedBoxes,1));
            for idx = 1:size(gtBoxes,1)
                for idy = 1:size(detectedBoxes,1)
                    intersectionMatrix(idx,idy) = overlapPercentage(gtBoxes(idx,:), detectedBoxes(idy,:)) >= threshold;
                end
            end
            
            statisticalData(frameNumber).falsePositives = any(all(intersectionMatrix == 0, 1));
            statisticalData(frameNumber).falseNegatives = any(all(intersectionMatrix == 0, 2));
            statisticalData(frameNumber).detectedLabels = (sum(any(intersectionMatrix == 1, 1)) / size(gtBoxes,1)) >= threshold;
            statisticalData(frameNumber).splits = any(sum(intersectionMatrix == 1, 2) > 1); 
            statisticalData(frameNumber).merges = any(sum(intersectionMatrix == 1, 1) > 1);
        end
        
        % Threshold statistics
        index = round(threshold / incrementThreshold);
        percentageCoveredFrames(index) = (sum([statisticalData.detectedLabels]) / totalFrames) * 100;
        percentageFalsePositives(index) = (sum([statisticalData.falsePositives]) / totalFrames) * 100;
        percentageFalseNegatives(index) = (sum([statisticalData.falseNegatives]) / totalFrames) * 100;
    end
    
    % Plot statistics
    thresholdInterval = minimumThreshold:incrementThreshold:maximumThreshold;
    figure; title("Success Plot (%)"); hold on;
    plot(thresholdInterval, percentageCoveredFrames, 'b.', 'LineWidth', 2);
    
    figure; title("False Positives and False Negatives (%)"); hold on;
    plot(thresholdInterval, percentageFalsePositives, 'b.', 'LineWidth', 2, "DisplayName", "False Positives");
    plot(thresholdInterval, percentageFalseNegatives, 'r.', 'LineWidth', 2, "DisplayName", "False Negatives");
    legend("Location", "southeast");
    
    function IoU = overlapPercentage(gtBox, detectedBox)
        x1 = gtBox(1); y1 = gtBox(2); w1 = gtBox(3); h1 = gtBox(4);
        x2 = detectedBox(1); y2 = detectedBox(2); w2 = detectedBox(3); h2 = detectedBox(4);
        
        % No overlap
        if x1 + w1 <= x2 || x2 + w2 <= x1 || y1 + h1 <= y2 || y2 + h2 <= y1
            IoU = 0;
            return;
        end
    
        % Overlap region dimensions
        xLeft = max(x1, x2);
        xRight = min(x1 + w1, x2 + w2);
        yBottom = max(y1, y2);
        yTop = min(y1 + h1, y2 + h2);
        
        % Overlap area
        overlapWidth = xRight - xLeft;
        overlapHeight = yTop - yBottom;
        overlapArea = overlapWidth * overlapHeight;
    
        % Individual boxes area
        area1 = w1 * h1;
        area2 = w2 * h2;
        unionArea = (area1 + area2) - overlapArea;
    
        IoU = overlapArea / unionArea;
    end
end
