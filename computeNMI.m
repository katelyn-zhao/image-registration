%%NMI
function nmi = computeNMI(image1, image2)
    % Compute histogram of both images
    jointHist = histcounts2(double(image1(:)), double(image2(:)), 256);
    jointHist = jointHist / sum(jointHist(:)); % Normalize to get joint probabilities
    marginal1 = sum(jointHist, 2); % Marginal for image1
    marginal2 = sum(jointHist, 1); % Marginal for image2

    % Calculate entropies
    H1 = -sum(marginal1 .* log2(marginal1 + eps));
    H2 = -sum(marginal2 .* log2(marginal2 + eps));
    H12 = -sum(jointHist(:) .* log2(jointHist(:) + eps));

    % Compute NMI
    nmi = (H1 + H2) / H12;
end