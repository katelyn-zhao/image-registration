% Load the fixed and moving images (NIfTI format)
fixedImage = double(niftiread("23-TR-0002/PreTreatment Imaging/precontrast_23-TR-0002_20110731_900 tumor sphere seg.nii"));

movingImage = double(niftiread("23-TR-0002/PreTreatment Imaging/delayed_23-TR-0002_20110731_12 tumor sphere seg.nii"));

fixedImage = fixedImage / max(fixedImage(:));
movingImage = movingImage / max(movingImage(:));

% Set up optimizer and metric
[optimizer, metric] = imregconfig('monomodal');

% Perform registration
tform = imregtform(movingImage, fixedImage, 'rigid', optimizer, metric);
registered = imwarp(movingImage, tform, 'OutputView', imref3d(size(fixedImage)));

% Compute metrics
nmi_value = computeNMI(fixedImage, registered);
ecc_value = computeECC(fixedImage, registered);
tform_inverse = imregtform(registered, fixedImage, 'rigid', optimizer, metric);
ice_value = computeICE(movingImage, tform, tform_inverse);


% Display metrics
fprintf('NMI: %.4f\n', nmi_value);
fprintf('ECC: %.4f\n', ecc_value);
fprintf('ICE: %.4f\n', ice_value);

% Visualize
figure;
imshowpair(squeeze(fixedImage(:,:,end/2)), squeeze(registered(:,:,end/2)));
title('Fixed vs Registered Image');