function [points, features] = runFeatureExtraction(images, numImages, config, tui)
% RUNFEATUREEXTRACTION Executes Phase 2: Extracts features using the specific detector
    tui.phase('2/5', 'FEATURE EXTRACTION');
    t2 = tic;
    
    features = cell(1, numImages);
    points   = cell(1, numImages);

    for i = 1:numImages
        tui.log('FEATURES', sprintf('Image %d/%d — extracting %s descriptors...', ...
            i, numImages, config.detector));
        [points{i}, features{i}] = extractDescriptors(images{i}, config.detector);
        tui.log('FEATURES', sprintf('  -> %d keypoints detected', points{i}.Count));
    end

    tui.timing('Feature Extraction', toc(t2));
end
