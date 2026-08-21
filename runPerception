function [images, numImages] = runPerception(imageDir, config, tui)
% RUNPERCEPTION Executes Phase 1: Loads and preprocesses images
    tui.phase('1/5', 'IMAGE PERCEPTION');
    t1 = tic;
    
    tui.log('LOAD', 'Scanning for drone images...');
    images = loadDroneImages(imageDir);

    if isempty(images) || all(cellfun(@isempty, images))
        tui.log('ERROR', 'No valid images found in directory.');
        tui.log('HINT', 'Supported formats: .jpg .jpeg .png .tif .tiff .bmp');
        numImages = 0;
        return;
    end

    numImages = numel(images);
    tui.log('LOAD', sprintf('Found %d images', numImages));

    % Apply image cap if configured
    if config.maxImages > 0 && numImages > config.maxImages
        tui.log('LOAD', sprintf('Limiting to %d images (config.maxImages)', config.maxImages));
        images = images(1:config.maxImages);
        numImages = config.maxImages;
    end

    if numImages < 2
        tui.log('ERROR', 'Need at least 2 images for reconstruction.');
        return;
    end

    tui.log('PREPROCESS', 'Normalizing resolution, contrast, and noise...');
    images = preprocessImages(images);

    tui.log('DISTORTION', 'Applying lens distortion correction...');
    images = distortionCorrection(images);

    tui.timing('Perception', toc(t1));
end
