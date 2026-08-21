function main(varargin)
% MAIN Entry point for the Drone Image Reconstruction System
%   Runs the entire pipeline from a single MATLAB command.
%   All output is displayed in a dedicated TUI log window and
%   results are shown in separate figure windows.
%
% Usage:
%   main()                          — Interactive folder picker
%   main('path/to/images')          — Direct path
%   main('path/to/images', config)  — Custom config struct
%
% Example:
%   main('C:\drone_captures\flight_001')
%   main('storage/raw_images', struct('detector','SURF','blend','multiband'))

    % =====================================================================
    %  BOOTSTRAP — Resolve paths, add all modules to path
    % =====================================================================
    matlabDir = fileparts(mfilename('fullpath'));
    projectRoot = fileparts(matlabDir);           % one level up from matlab/
    addpath(genpath(matlabDir));

    % =====================================================================
    %  CONFIGURATION — Defaults + user overrides
    % =====================================================================
    config = defaultConfig(matlabDir);

    if nargin >= 2 && isstruct(varargin{2})
        userCfg = varargin{2};
        fnames = fieldnames(userCfg);
        for k = 1:numel(fnames)
            config.(fnames{k}) = userCfg.(fnames{k});
        end
    end

    % =====================================================================
    %  TUI — Launch Log Window
    % =====================================================================
    tui = TUILogger('Drone Reconstruction Engine');

    tui.header();
    tui.log('SYSTEM', 'Pipeline engine initialized');
    tui.log('SYSTEM', sprintf('MATLAB %s  |  %s', version, computer));
    maxImgStr = 'all';
    if config.maxImages > 0
        maxImgStr = num2str(config.maxImages);
    end
    tui.log('CONFIG', sprintf('Detector: %s  |  Blend: %s  |  Max Images: %s', ...
        config.detector, config.blendMethod, maxImgStr));
    tui.separator();

    % =====================================================================
    %  INPUT — Resolve image directory
    % =====================================================================
    if nargin >= 1 && ~isempty(varargin{1})
        imageDir = varargin{1};
        if ~isfolder(imageDir)
            % Try relative to project root
            imageDir = fullfile(projectRoot, varargin{1});
        end
    else
        tui.log('INPUT', 'No path provided — opening folder picker...');
        imageDir = uigetdir(fullfile(projectRoot, 'storage', 'raw_images'), ...
                            'Select Drone Image Directory');
        if imageDir == 0
            tui.log('ERROR', 'No folder selected. Aborting.');
            return;
        end
    end

    if ~isfolder(imageDir)
        tui.log('ERROR', sprintf('Directory not found: %s', imageDir));
        return;
    end
    tui.log('INPUT', sprintf('Source: %s', imageDir));

    % Ensure output directories exist
    outputDirs = {config.outputDir, config.depthDir, config.logDir, config.processedDir};
    for d = 1:numel(outputDirs)
        if ~isfolder(outputDirs{d})
            mkdir(outputDirs{d});
        end
    end

    % =====================================================================
    %  PIPELINE EXECUTION
    % =====================================================================
    totalTimer = tic;

    try
        % ------ PHASE 1 : PERCEPTION ------
        [images, numImages] = runPerception(imageDir, config, tui);
        if numImages < 2
            return;
        end

        % ------ PHASE 2 : FEATURE EXTRACTION ------
        [points, features] = runFeatureExtraction(images, numImages, config, tui);

        % ------ PHASE 3 : HOMOGRAPHY ESTIMATION & CLUSTERING ------
        [images, points, features, matchScores, matchesI, matchesJ, clusterBins, numClusters, edgeCounts] = runGeometricEstimation(images, points, features, numImages, config, tui);

        % ------ PHASE 4 & 5 : RECONSTRUCTION + DEPTH (per cluster) ------
        [panoramaFiles, depthFiles, totalEdgeMatches] = runReconstruction(images, points, features, matchScores, matchesI, matchesJ, clusterBins, numClusters, edgeCounts, config, tui);


        % =====================================================================
        %  SUMMARY
        % =====================================================================
        totalTime = toc(totalTimer);
        tui.separator();
        tui.log('DONE', '═══ PIPELINE COMPLETE ═══');
        tui.log('STATS', sprintf('Images processed:    %d', numImages));
        tui.log('STATS', sprintf('Clusters found:      %d', numClusters));
        tui.log('STATS', sprintf('Total edge matches:  %d', totalEdgeMatches));
        tui.log('STATS', sprintf('Total time:          %.2f seconds', totalTime));
        for pf = 1:numel(panoramaFiles)
            tui.log('OUTPUT', sprintf('Panorama %d: %s', pf, panoramaFiles{pf}));
        end
        for df = 1:numel(depthFiles)
            tui.log('OUTPUT', sprintf('Depth    %d: %s', df, depthFiles{df}));
        end
        tui.separator();

        % Save run log
        logFile = fullfile(config.logDir, ...
            sprintf('run_%s.log', string(datetime('now', 'Format', 'yyyyMMdd_HHmmss'))));
        tui.saveLog(logFile);
        tui.log('LOG', sprintf('Full log saved: %s', logFile));

    catch ex
        tui.log('FATAL', sprintf('Pipeline crashed: %s', ex.message));
        tui.log('FATAL', sprintf('In: %s (line %d)', ex.stack(1).name, ex.stack(1).line));
        for s = 1:min(5, numel(ex.stack))
            tui.log('TRACE', sprintf('  %s:%d', ex.stack(s).name, ex.stack(s).line));
        end
        tui.separator();
    end
end


% =========================================================================
%  DEFAULT CONFIGURATION
% =========================================================================
function config = defaultConfig(matlabDir)
    config.detector     = 'AKAZE';      % Feature detector: AKAZE | SURF | KAZE | FAST | HARRIS | ORB
    config.blendMethod  = 'multiband';  % Blend method: multiband | simple
    config.enableDepth  = true;         % Enable depth estimation
    config.maxImages    = 0;            % Max images to process (0 = all)
    config.maxWorkers   = 4;            % Max parallel workers (reserved for future parfor)
    
    % Ensure files save explicitly inside matlab/storage/
    baseStorageDir      = fullfile(matlabDir, 'storage');
    config.outputDir    = fullfile(baseStorageDir, 'panoramas');
    config.depthDir     = fullfile(baseStorageDir, 'depth_maps');
    config.logDir       = fullfile(baseStorageDir, 'logs');
    config.processedDir = fullfile(baseStorageDir, 'processed_images');
end
