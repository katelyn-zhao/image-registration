%%Jacobian
function [jacobianMean, jacobianStd] = computeJacobianStats(sx,sy,sz)
    % Gradients
    [gx_y,gx_x,gx_z] = gradient(sx);
    [gy_y,gy_x,gy_z] = gradient(sy);
    [gz_y,gz_x,gz_z] = gradient(sz);
    
    % Add identity
    gx_x = gx_x + 1;
    gy_y = gy_y + 1;
    gz_z = gz_z + 1;
    
    % Determinant
    det_J = gx_x.*gy_y.*gz_z + ...
            gy_x.*gz_y.*gx_z + ...
            gz_x.*gx_y.*gy_z - ...
            gz_x.*gy_y.*gx_z - ...
            gy_x.*gx_y.*gz_z - ...
            gx_x.*gz_y.*gy_z;

    % Compute mean and standard deviation
    jacobianMean = mean(det_J(:));
    jacobianStd = std(det_J(:));
end
