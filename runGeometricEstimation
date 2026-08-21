function [images, points, features, matchScores, matchesI, matchesJ, clusterBins, numClusters, edgeCounts] = runGeometricEstimation(images, points, features, numImages, config, tui)
% RUNGEOMETRICESTIMATION Executes Phase 3: Computes global match scores and auto-corrects geometry
    tui.phase('3/5', 'HOMOGRAPHY ESTIMATION & CLUSTERING');
    t3 = tic;

    % --- Step 3a: Exhaustive global match graph ---
    matchScores = zeros(numImages, numImages);
    matchesI = cell(numImages, numImages);
    matchesJ = cell(numImages, numImages);

    for i = 1:numImages
        for j = i+1:numImages
            [~, ptsI, ptsJ] = matchFeaturesNN(features{i}, features{j}, points{i}, points{j});
            n = size(ptsI, 1);
            matchScores(i, j) = n;
            matchScores(j, i) = n;
            matchesI{i, j} = ptsI;
            matchesJ{i, j} = ptsJ;
            matchesI{j, i} = ptsJ;
            matchesJ{j, i} = ptsI;
            
            if n >= 4
                tui.log('MATCH', sprintf('Global: Image %d ↔ %d matched (%d features)', i, j, n));
            end
        end
    end

    % --- Auto-correct mirrored/flipped/rotated images ---
    for i = 1:numImages
        if max(matchScores(i, :)) < 350
            tui.log('WARN', sprintf('Image %d has low max matches (%d). Testing 5 geometric states...', i, max(matchScores(i, :))));
            
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
                [ptsT, featsT] = extractDescriptors(transImg, config.detector);
                
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
                tui.log('MATCH', sprintf('  ✔ Image %d [%s] found strong global matches! (Score %d). Permanently correcting geometry.', ...
                    i, bestTransformName, bestTransformScore));
                
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
                tui.log('MATCH', '  ✘ No geometric transformation yielded significantly better matches. Keeping original.');
            end
        end
    end

    % --- Step 3b: Find connected components (clusters) ---
    [srcNodes, dstNodes] = find(matchScores >= 4);
    validEdges = srcNodes < dstNodes;
    srcNodes = srcNodes(validEdges);
    dstNodes = dstNodes(validEdges);
    edgeCounts = arrayfun(@(s, d) matchScores(s, d), srcNodes, dstNodes);

    if isempty(srcNodes)
        tui.log('WARN', 'No valid matches found across any image pair. Each image is its own cluster.');
        clusterBins = 1:numImages;
    else
        G = graph(srcNodes, dstNodes, edgeCounts, numImages);
        clusterBins = conncomp(G);  % clusterBins(i) = cluster index for image i
    end

    numClusters = max(clusterBins);
    tui.log('CLUSTER', sprintf('Identified %d image cluster(s):', numClusters));
    for k = 1:numClusters
        clusterIdx = find(clusterBins == k);
        tui.log('CLUSTER', sprintf('  Cluster %d: images [%s]', k, num2str(clusterIdx)));
    end

    tui.timing('Homography Estimation & Clustering', toc(t3));
end
