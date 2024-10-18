% Load the fixed and moving images (NIfTI format)
fixedImage = double(niftiread("precontrast_23-TR-0002_20110731_900.nii"));

movingImage = double(niftiread("portalvenous_23-TR-0002_20110731_10.nii"));

fixedImage = fixedImage / max(fixedImage(:));
movingImage = movingImage / max(movingImage(:));

% Set up optimizer and metric
[optimizer, metric] = imregconfig('monomodal');
optimizer.MaximumStepLength = 0.001;

% Perform registration
tform = imregtform(movingImage, fixedImage, 'rigid', optimizer, metric);
registered = imwarp(movingImage, tform, 'OutputView', imref3d(size(fixedImage)));

%Save the transformation object
save('portalvenous_affine.mat', 'tform');

% Compute metrics
nmi_value = computeNMI(fixedImage, registered);
ecc_value = computeECC(fixedImage, registered);
tform_inverse = imregtform(registered, fixedImage, 'rigid', optimizer, metric);
ice_value = computeICE(movingImage, tform, tform_inverse);


% Display metrics
fprintf('NMI: %.4f\n', nmi_value);
fprintf('ECC: %.4f\n', ecc_value);
fprintf('ICE: %.4f\n', ice_value);

% Visualize Overlayed
figure;
imshowpair(squeeze(fixedImage(:,:,end/2)), squeeze(registered(:,:,end/2)));
title('Fixed vs Registered Image');

% Visualize Side by Side
figure;
subplot(1, 2, 1);
imshow(squeeze(fixedImage(:,:,end/2)), []);
title('Fixed Image');
subplot(1, 2, 2);
imshow(squeeze(registered(:,:,end/2)), []);
title('Registered Image');
