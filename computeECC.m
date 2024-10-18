%%ECC
function ecc = computeECC(image1, image2)
    % Normalize images
    image1 = double(image1) / max(image1(:));
    image2 = double(image2) / max(image2(:));
   
    % Compute means
    mean1 = mean(image1(:));
    mean2 = mean(image2(:));
   
    % Compute covariance
    covariance = sum((image1(:) - mean1) .* (image2(:) - mean2));
   
    % Compute variances
    variance1 = sum((image1(:) - mean1).^2);
    variance2 = sum((image2(:) - mean2).^2);
   
    % Compute ECC
    ecc = covariance / sqrt(variance1 * variance2);
end