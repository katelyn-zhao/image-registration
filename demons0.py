import numpy as np
from sklearn.metrics import mutual_info_score
from scipy.stats import entropy
from skimage.metrics import normalized_mutual_information as skimage_nmi
import os
import SimpleITK as sitk
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import map_coordinates

def calculate_nmi(image1, image2):
    """
    Calculate the Normalized Mutual Information (NMI) between two images.
    """
    # Convert SimpleITK images to NumPy arrays
    image1_np = sitk.GetArrayFromImage(image1)
    image2_np = sitk.GetArrayFromImage(image2)

    # Compute 2D histogram
    hist_2d, _, _ = np.histogram2d(image1_np.ravel(), image2_np.ravel())

    # Joint and marginal probabilities
    joint_prob = hist_2d / np.sum(hist_2d)
    p1 = np.sum(joint_prob, axis=1)  # Marginal probabilities for image1
    p2 = np.sum(joint_prob, axis=0)  # Marginal probabilities for image2

    # Calculate entropies
    H1 = -np.sum(p1 * np.log(p1 + 1e-10))
    H2 = -np.sum(p2 * np.log(p2 + 1e-10))
    H12 = -np.sum(joint_prob * np.log(joint_prob + 1e-10))

    # NMI formula
    nmi = (H1 + H2) / H12
    return nmi

def calculate_ecc(image1, image2):
    """
    Calculate the Entropy Correlation Coefficient (ECC) between two images.
    """
    # Convert SimpleITK images to NumPy arrays
    image1_np = sitk.GetArrayFromImage(image1)
    image2_np = sitk.GetArrayFromImage(image2)
    
    hist_2d, _, _ = np.histogram2d(image1_np.ravel(), image2_np.ravel())

    joint_prob = hist_2d / np.sum(hist_2d)
    p1 = np.sum(joint_prob, axis=1)  # Marginal probabilities for image1
    p2 = np.sum(joint_prob, axis=0)  # Marginal probabilities for image2

    # Calculate entropy
    H1 = -np.sum(p1 * np.log(p1 + 1e-10))
    H2 = -np.sum(p2 * np.log(p2 + 1e-10))
    H12 = -np.sum(joint_prob * np.log(joint_prob + 1e-10))

    # ECC formula
    ecc = 2 * (H1 + H2 - H12) / (H1 + H2)
    return ecc


def calculate_jacobian_stats(sx, sy, sz):
     # Compute gradients
    gx_y, gx_x, gx_z = np.gradient(sx)
    gy_y, gy_x, gy_z = np.gradient(sy)
    gz_y, gz_x, gz_z = np.gradient(sz)

    # Add identity to diagonal elements
    gx_x += 1
    gy_y += 1
    gz_z += 1

    # Calculate determinant of the Jacobian matrix
    det_J = (
        gx_x * gy_y * gz_z +
        gy_x * gz_y * gx_z +
        gz_x * gx_y * gy_z -
        gz_x * gy_y * gx_z -
        gy_x * gx_y * gz_z -
        gx_x * gz_y * gy_z
    )

    # Compute mean and standard deviation
    jacobian_mean = np.mean(det_J)
    jacobian_std = np.std(det_J)

    return jacobian_mean, jacobian_std


def calculate_ice(img, vx, vy, vz):
    """
    Calculate the Inverse Consistency Error (ICE) using displacement fields.
    """
    # Convert SimpleITK image to NumPy array
    img_np = sitk.GetArrayFromImage(img)  # Convert SimpleITK Image to NumPy array
    dimX, dimY, dimZ = img_np.shape

    # Create meshgrid for image coordinates
    X, Y, Z = np.meshgrid(np.arange(dimX), np.arange(dimY), np.arange(dimZ), indexing='ij')

    # Forward transform
    Xf = X + vx
    Yf = Y + vy
    Zf = Z + vz

    # Clamp to valid range
    Xf = np.clip(Xf, 0, dimX - 1)
    Yf = np.clip(Yf, 0, dimY - 1)
    Zf = np.clip(Zf, 0, dimZ - 1)

    # Inverse transform (assuming symmetric field)
    Xi = Xf - vx
    Yi = Yf - vy
    Zi = Zf - vz

    # Clamp inverse coordinates
    Xi = np.clip(Xi, 0, dimX - 1)
    Yi = np.clip(Yi, 0, dimY - 1)
    Zi = np.clip(Zi, 0, dimZ - 1)

    # Reshape coordinates for map_coordinates
    coordinates = np.vstack([Xi.ravel(), Yi.ravel(), Zi.ravel()])

    # Interpolate values using map_coordinates
    transformed = map_coordinates(img_np, coordinates, order=1, mode='nearest').reshape(img_np.shape)

    # Compute RMSE as ICE metric
    difference = img_np - transformed
    ice = np.sqrt(np.mean(difference**2))
    
    return ice

def pad_image_to_match(image, reference_image):
    """
    Pad 'image' to match the size of 'reference_image' by adding zero padding.
    """
    # Get the sizes of both images
    image_size = image.GetSize()
    reference_size = reference_image.GetSize()

    # Calculate the difference in size (pad only z-axis)
    pad_diff = [ref_size - img_size for ref_size, img_size in zip(reference_size, image_size)]
    
    # Ensure only z-axis needs padding, and pad if necessary
    lower_pad = [0, 0, 0]  # No padding in x and y dimensions
    upper_pad = [0, 0, max(pad_diff[2], 0)]  # Pad z-axis to match size

    # Apply padding
    padded_image = sitk.ConstantPad(image, lower_pad, upper_pad, constant=0)
    
    return padded_image

def demons(fixed_image, moving_image):
    fixed_image = pad_image_to_match(fixed_image, moving_image)

    demons_filter = sitk.DemonsRegistrationFilter()
    demons_filter.SetNumberOfIterations(100)  # Increase if needed
    demons_filter.SetSmoothDisplacementField(True)  # Optional, for regularization
    demons_filter.SetStandardDeviations(1.0)  # Smoothness of update field

    displacement_field = demons_filter.Execute(fixed_image, moving_image)

    sx = sitk.VectorIndexSelectionCast(displacement_field, 0)  # X-component
    sy = sitk.VectorIndexSelectionCast(displacement_field, 1)  # Y-component
    sz = sitk.VectorIndexSelectionCast(displacement_field, 2)  # Z-component

    # Convert to NumPy arrays
    sx_array = sitk.GetArrayFromImage(sx)
    sy_array = sitk.GetArrayFromImage(sy)
    sz_array = sitk.GetArrayFromImage(sz)


    # Convert the displacement field into a transform for warping
    transform = sitk.DisplacementFieldTransform(displacement_field)

    # Apply the transformation to the moving image
    warped_image = sitk.Resample(moving_image, fixed_image, transform,
                                sitk.sitkLinear, 0.0, moving_image.GetPixelID())
    print("Image Registration Done!")

    file_name = os.path.basename(moving_image).split('.')[0]


    sitk.WriteImage(warped_image, f'C:/Users/User/Desktop/img_reg_data/new_patients/demons/{file_name}_registered.nii')
    print('Registered Image Saved.')

    nmi_value = calculate_nmi(fixed_image, warped_image)
    print("NMI: ", nmi_value)
    ecc_value = calculate_ecc(fixed_image, warped_image)
    print("ECC: ", ecc_value)
    ice_value = calculate_ice(moving_image, sx_array, sy_array, sz_array)
    print("ICE: ", ice_value)

    f = open(f"C:/Users/User/Desktop/img_reg_data/new_patients/demons_metrics.txt", "a")
    print(moving_image_path, file=f)
    print("NMI: ", nmi_value, file=f)
    print("ECC: ", ecc_value, file=f)
    print("ICE: ", ice_value, file=f)
    f.close()

fixed_image_path = 'C:/Users/User/Desktop/img_reg_data/new_patients/fixed/'
moving_image_path = 'C:/Users/User/Desktop/img_reg_data/new_patients/moving/'

fixed_image_paths = []
moving_image_paths = []

for i in range(3):
    fixed = sorted(os.listdir(fixed_image_path))
    for i, image_path in enumerate(fixed):
        fixed_image_paths.append(image_path)

moving = sorted(os.listdir(moving_image_path))
for i, image_path in enumerate(moving):
    moving_image_paths.append(image_path)

for i in range(len(fixed_image_paths)):
    print(fixed_image_paths[i])
    print(moving_image_paths[i])
    demons(f"C:/Users/User/Desktop/img_reg_data/new_patients/{fixed_image_paths[i]}", f"C:/Users/User/Desktop/img_reg_data/new_patients/{moving_image_paths[i]}")
