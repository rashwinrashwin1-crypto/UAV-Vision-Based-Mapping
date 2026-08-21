function pipelineResult = reconstructionPipeline(imageDir, outputDir, detector)
% RECONSTRUCTIONPIPELINE Main pipeline for drone image stitching
% Processes images from imageDir and saves outputs to outputDir
%   detector - (optional) feature algorithm: 'AKAZE' (default), 'SURF', 'KAZE', etc.

    if nargin < 3
        detector = 'AKAZE';
    end

    disp('Starting Drone Image Reconstruction Pipeline...');
    
    % --- 1. Perception Module ---
    disp('Loading and preprocessing images...');
    images = loadDroneImages(imageDir);
    if isempty(images)
        error('No images found in the specified directory.');
    end
    images = preprocessImages(images);
    images = distortionCorrection(images);
    
    numImages = numel(images);
    disp(['Successfully loaded ' num2str(numImages) ' images.']);
    
    % --- 2. Processing Module ---
    fprintf('Extracting features using %s and estimating homographies...\n', detector);
    features = cell(1, numImages);
    points = cell(1, numImages);
    
    for i = 1:numImages
        [points{i}, features{i}] = extractDescriptors(images{i}, detector);
    end
    
    % Match features between all pairs to build a robust spanning tree
    disp('Matching features across all image pairs to find optimal connections...');
    tforms(numImages) = projective2d(eye(3));
    matchScores = zeros(numImages, numImages);
    matchesI = cell(numImages, numImages);
    matchesJ = cell(numImages, numImages);

    % Compute pairwise matches
    for i = 1:numImages
        for j = i+1:numImages
            [~, ptsI, ptsJ] = matchFeaturesNN(features{i}, features{j}, points{i}, points{j});
            numMatches = size(ptsI, 1);
            matchScores(i, j) = numMatches;
            matchScores(j, i) = numMatches;
            matchesI{i, j} = ptsI;
            matchesJ{i, j} = ptsJ;
            matchesI{j, i} = ptsJ;
            matchesJ{j, i} = ptsI;
        end
    end

    % --- Auto-correct mirrored/flipped/rotated images ---
    for i = 1:numImages
        % If the maximum match score is suspiciously low, test exhaustive geometric states
        if max(matchScores(i, :)) < 350
            disp(['Warning: Image ' num2str(i) ' has low max matches (' num2str(max(matchScores(i, :))) '). Testing 5 geometric states...']);
            
            bestTransformScore = max(matchScores(i, :));
            bestTransformImg = images{i};
            bestTransformPoints = points{i};
            bestTransformFeatures = features{i};
            bestTransformMatchesI = matchesI(i, :);
            bestTransformMatchesJ = matchesJ(i, :);
            bestTransformMatchScores = matchScores(i, :);
            transformMade = false;
            bestTransformName = '';
            
            transforms = {@(x) fliplr(x), @(x) flipud(x), @(x) rot90(x, 2), @(x) rot90(x, 1), @(x) rot90(x, -1)};
            transformNames = {'Flipped Horizontally', 'Flipped Vertically', 'Rotated 180°', 'Rotated 90°', 'Rotated -90°'};
            
            for tIdx = 1:length(transforms)
                transImg = transforms{tIdx}(images{i});
                [ptsT, featsT] = extractDescriptors(transImg, detector);
                
                tempScores = zeros(1, numImages);
                tempMatchesI = cell(1, numImages);
                tempMatchesJ = cell(1, numImages);
                
                for j = 1:numImages
                    if i == j; continue; end
                    [~, pT, pJ] = matchFeaturesNN(featsT, features{j}, ptsT, points{j});
                    tempScores(j) = size(pT, 1);
                    tempMatchesI{j} = pT;
                    tempMatchesJ{j} = pJ;
                end
                
                if max(tempScores) > bestTransformScore * 1.5 && max(tempScores) > 15
                    bestTransformScore = max(tempScores);
                    bestTransformImg = transImg;
                    bestTransformPoints = ptsT;
                    bestTransformFeatures = featsT;
                    bestTransformMatchScores = tempScores;
                    bestTransformMatchesI = tempMatchesI;
                    bestTransformMatchesJ = tempMatchesJ;
                    transformMade = true;
                    bestTransformName = transformNames{tIdx};
                end
            end
            
            if transformMade
                disp([' -> Image ' num2str(i) ' [' bestTransformName '] found strong matches! (Score ' num2str(bestTransformScore) '). Permanently correcting geometry.']);
                
                images{i} = bestTransformImg;
                points{i} = bestTransformPoints;
                features{i} = bestTransformFeatures;
                
                for j = 1:numImages
                    if i == j; continue; end
                    matchScores(i, j) = bestTransformMatchScores(j);
                    matchScores(j, i) = bestTransformMatchScores(j);
                    matchesI{i, j} = bestTransformMatchesI{j};
                    matchesJ{i, j} = bestTransformMatchesJ{j};
                    matchesI{j, i} = bestTransformMatchesJ{j};
                    matchesJ{j, i} = bestTransformMatchesI{j};
                end
            else
                disp([' -> No geometric transformation yielded significantly better matches for Image ' num2str(i) '. Keeping original.']);
            end
        end
    end
    
    % Build Maximum Spanning Tree to connect images resiliently
    added = false(1, numImages);
    validTransformsIdx = true(1, numImages);
    
    [maxVal, linearIdx] = max(matchScores(:));
    if maxVal == 0 || isempty(maxVal)
        warning('No valid matches found across any images!');
        added(1) = true; % Fallback
    else
        [idx1, ~] = ind2sub(size(matchScores), linearIdx);
        added(idx1) = true;
    end
    
    if sum(added) == 0
        added(1) = true;
    end
    
    % Iteratively add the best matching unadded image using global backtracking aggregation
    for step = 2:numImages
        bestScore = -1;
        bestUnadded = -1;
        
        for j = 1:numImages
            if ~added(j)
                scoreJ = 0;
                for i = 1:numImages
                    if added(i)
                        scoreJ = scoreJ + matchScores(i, j);
                    end
                end
                if scoreJ > bestScore
                    bestScore = scoreJ;
                    bestUnadded = j;
                end
            end
        end
        
        fprintf('Connecting image %d to panorama globally (Total Score: %d)\n', bestUnadded, bestScore);
        
        allGlobalPts = [];
        allUnaddedPts = [];
        
        for i = 1:numImages
            if added(i) && matchScores(i, bestUnadded) >= 4
                ptsI = matchesI{i, bestUnadded};
                ptsJ = matchesJ{i, bestUnadded};
                
                [xG, yG] = transformPointsForward(tforms(i), double(ptsI.Location(:,1)), double(ptsI.Location(:,2)));
                allGlobalPts = [allGlobalPts; double(xG), double(yG)]; %#ok<AGROW>
                allUnaddedPts = [allUnaddedPts; double(ptsJ.Location)]; %#ok<AGROW>
            end
        end
        
        % Estimate homography to the newly combined global points map!
        [tformStep, isValid] = estimateHomography(allGlobalPts, allUnaddedPts, projective2d(eye(3)));
        
        tforms(bestUnadded) = tformStep;
        validTransformsIdx(bestUnadded) = isValid;
        
        if ~isValid
            warning('Failed to reliably estimate mapping for image %d. It will be excluded from the final panorama.', bestUnadded);
        end
        
        added(bestUnadded) = true;
    end
    
    % Filter out invalid images to avoid a jumbled output entirely
    validImages = images(validTransformsIdx);
    validTforms = tforms(validTransformsIdx);
    
    if isempty(validImages)
        error('All images failed to align properly.');
    end
    
    % --- 3. Reconstruction Module ---
    disp('Aligning images and generating panorama...');
    [panorama, ~] = alignImages(validImages, validTforms);
    
    % Resize proportionally to prevent aspect ratio distortion (squashing), but cap the max dimension
    disp('Applying sharpening and proportional resizing to preserve natural geometry...');
    [h, w, ~] = size(panorama);
    maxDim = 3840; % Allow up to 4K resolution rather than forcing 1080p
    if max(h, w) > maxDim
        scaleFactor = maxDim / max(h, w);
        panorama = imresize(panorama, scaleFactor, 'lanczos3');
    end
    panorama = imsharpen(panorama, 'Radius', 2, 'Amount', 1.0, 'Threshold', 0.02);
    
    % Save result
    outputFile = fullfile(outputDir, 'stitched_panorama.png');
    imwrite(panorama, outputFile);
    
    pipelineResult = struct('status', 'success', 'outputFile', outputFile, 'numImages', numImages);
    disp(['Pipeline completed successfully. Output saved to ' outputFile]);
end
