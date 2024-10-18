%ICE

% function ice = computeICE(image1, T1, T2)
%     refGrid = imref3d(size(image1));
% 
%     % Apply forward transformation and then the inverse
%     transformedForward = imwarp(image1, T1, 'OutputView', refGrid);
%     transformedInverse = imwarp(transformedForward, T2, 'OutputView', refGrid);
% 
%     % Compute the difference between the original image and the double-transformed image
%     difference = image1 - transformedInverse;
%     ice = sqrt(mean(difference(:).^2)); % RMSE as the ICE metric
% end

function ice = computeICE(img, vx, vy, vz)
    [dimX, dimY, dimZ] = size(img);

    % Create meshgrid for image coordinates
    [X, Y, Z] = ndgrid(1:dimX, 1:dimY, 1:dimZ);

    % Forward transform
    Xf = X + vx; Yf = Y + vy; Zf = Z + vz;

    % Clamp to valid range
    Xf = min(max(Xf, 1), dimX);
    Yf = min(max(Yf, 1), dimY);
    Zf = min(max(Zf, 1), dimZ);

    % Inverse transform (assuming symmetric field)
    Xi = Xf - vx; Yi = Yf - vy; Zi = Zf - vz;

    % Clamp inverse coordinates
    Xi = min(max(Xi, 1), dimX);
    Yi = min(max(Yi, 1), dimY);
    Zi = min(max(Zi, 1), dimZ);

    % Interpolate values
    transformed = iminterpolate(img, Xi, Yi, Zi);

    % Compute RMSE as ICE metric
    difference = img - transformed;
    ice = sqrt(mean(difference(:).^2));
end
