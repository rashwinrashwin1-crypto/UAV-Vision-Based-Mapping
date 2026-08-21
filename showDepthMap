function showDepthMap(depthMap, titleText)
% SHOWDEPTHMAP Renders a colorized depth map visualization
%   Uses the 'turbo' colormap for intuitive near/far representation.
%
% Inputs:
%    depthMap  - 2D float matrix (values in [0, 1] range expected)
%    titleText - (optional) custom title

    if nargin < 2
        titleText = 'Relative Depth Map — Multi-View Estimation';
    end

    figure('Name', 'Depth Map Viewer', ...
           'NumberTitle', 'off', ...
           'Color', [0.08 0.08 0.1], ...
           'Position', [100, 100, 1000, 700]);

    % Normalize if not already in [0, 1]
    if max(depthMap(:)) > 1
        depthMap = depthMap / max(depthMap(:));
    end

    imagesc(depthMap);
    axis image off;
    colormap('turbo');

    cb = colorbar;
    cb.Color = [0.8 0.8 0.8];
    cb.Label.String = 'Relative Depth (0 = far, 1 = near)';
    cb.Label.FontSize = 11;
    cb.Label.Color = [0.8 0.8 0.8];

    title(titleText, ...
          'Color', 'w', ...
          'FontSize', 14, ...
          'FontWeight', 'bold');

    % Stats annotation
    meanD = mean(depthMap(:));
    stdD  = std(depthMap(:));
    annotation('textbox', [0.01 0.01 0.4 0.05], ...
        'String', sprintf('Mean: %.3f  |  Std: %.3f  |  Range: [%.3f, %.3f]', ...
                           meanD, stdD, min(depthMap(:)), max(depthMap(:))), ...
        'Color', [0.6 0.6 0.6], ...
        'FontSize', 9, ...
        'EdgeColor', 'none');

    drawnow;
    fprintf('Depth map displayed: %d × %d\n', size(depthMap, 2), size(depthMap, 1));
end
