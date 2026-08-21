function showFeatureMatches(img1, img2, matchedPts1, matchedPts2, titleText)
% SHOWFEATUREMATCHES Visualizes matched keypoints between two drone images
%   Side-by-side display with colored lines connecting matched features.
%
% Inputs:
%    img1, img2             - the two source images (RGB or grayscale)
%    matchedPts1, matchedPts2 - matched point coordinates (Nx2 or feature objects)
%    titleText              - (optional) figure title

    if nargin < 5
        titleText = 'Feature Correspondences';
    end

    % Ensure RGB for display
    if size(img1, 3) == 1
        img1 = repmat(img1, [1 1 3]);
    end
    if size(img2, 3) == 1
        img2 = repmat(img2, [1 1 3]);
    end

    figure('Name', 'Feature Matches', ...
           'NumberTitle', 'off', ...
           'Color', [0.05 0.05 0.08], ...
           'Position', [50, 100, 1600, 600]);

    % Use Computer Vision Toolbox showMatchedFeatures if available
    try
        showMatchedFeatures(img1, img2, matchedPts1, matchedPts2, ...
            'montage', 'PlotOptions', {'g+', 'r+', 'y-'});
    catch
        % Fallback: manual side-by-side rendering
        manualMatchDisplay(img1, img2, matchedPts1, matchedPts2);
    end

    title(titleText, ...
          'Color', 'w', ...
          'FontSize', 14, ...
          'FontWeight', 'bold');

    % Extract numeric count
    if isobject(matchedPts1) && isprop(matchedPts1, 'Count')
        n = matchedPts1.Count;
    else
        n = size(matchedPts1, 1);
    end

    fprintf('Displaying %d feature matches.\n', n);
    drawnow;
end


function manualMatchDisplay(img1, img2, pts1, pts2)
% Manual fallback when showMatchedFeatures is not available

    % Extract numeric coordinates
    if isobject(pts1) && ismethod(pts1, 'Location')
        p1 = pts1.Location;
    elseif isobject(pts1)
        p1 = double(pts1.Location);
    else
        p1 = double(pts1);
    end

    if isobject(pts2) && ismethod(pts2, 'Location')
        p2 = pts2.Location;
    elseif isobject(pts2)
        p2 = double(pts2.Location);
    else
        p2 = double(pts2);
    end

    % Create side-by-side montage
    [h1, w1, ~] = size(img1);
    [h2, w2, ~] = size(img2);
    maxH = max(h1, h2);

    canvas = zeros(maxH, w1 + w2 + 20, 3, 'uint8'); % 20px gap
    canvas(1:h1, 1:w1, :) = img1;
    canvas(1:h2, w1+21:end, :) = img2;

    imshow(canvas, 'Border', 'tight');
    hold on;

    % Draw match lines
    numMatches = min(size(p1, 1), 200); % Cap for readability
    colors = hsv(numMatches);

    for i = 1:numMatches
        x1 = p1(i, 1); y1 = p1(i, 2);
        x2 = p2(i, 1) + w1 + 20; y2 = p2(i, 2);

        plot([x1, x2], [y1, y2], '-', 'Color', colors(i,:), 'LineWidth', 0.5);
        plot(x1, y1, 'g+', 'MarkerSize', 4);
        plot(x2, y2, 'r+', 'MarkerSize', 4);
    end

    hold off;
end
