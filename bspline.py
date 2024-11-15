import SimpleITK as sitk
import numpy as np

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

def b_spline_registration(fixed_image_path, moving_image_path):
    # Read the images
    fixed_image = sitk.ReadImage(fixed_image_path, sitk.sitkFloat32)
    moving_image = sitk.ReadImage(moving_image_path, sitk.sitkFloat32)

    print("Images Loaded")

    # Initialize the B-spline registration
    registration_method = sitk.ImageRegistrationMethod()

    # Set the metric (e.g., Mean Squares)
    registration_method.SetMetricAsMeanSquares()

    # Set the optimizer (e.g., Gradient Descent)
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100)

    # Set the interpolator
    registration_method.SetInterpolator(sitk.sitkLinear)

    # Set the initial transformation (identity)
    initial_transform = sitk.BSplineTransformInitializer(fixed_image, [8, 8, 8], 3)
    registration_method.SetInitialTransform(initial_transform)

    print("Starting Registration...")

    # Execute the registration
    final_transform = registration_method.Execute(fixed_image, moving_image)

    # Apply the transform to the moving image
    moving_resampled = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())

    print("Registration completed.")

    return fixed_image, moving_image, moving_resampled, final_transform

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

def compute_displacement_field(transform, image):
    width, height, depth = image.GetWidth(), image.GetHeight(), image.GetDepth()
    displacement_field = np.zeros((width, height, depth, 3))

    for x in range(width):
        for y in range(height):
            for z in range(depth):
                original_point = [x, y, z]
                transformed_point = transform.TransformPoint(original_point)
                displacement_field[x, y, z] = np.array(transformed_point) - np.array(original_point)

    return displacement_field

def jacobian_determinant(displacement_field):
    grad_x = np.gradient(displacement_field[..., 0], axis=0)
    grad_y = np.gradient(displacement_field[..., 1], axis=1)
    grad_z = np.gradient(displacement_field[..., 2], axis=2)
    
    jacobian = (grad_x[..., 0] * (grad_y[..., 1] * grad_z[..., 2] - grad_y[..., 2] * grad_z[..., 1]) -
                 grad_x[..., 1] * (grad_y[..., 0] * grad_z[..., 2] - grad_y[..., 2] * grad_z[..., 0]) +
                 grad_x[..., 2] * (grad_y[..., 0] * grad_z[..., 1] - grad_y[..., 1] * grad_z[..., 0]))
    
    return jacobian

def jacobian_statistics(jacobian):
    mean = np.mean(jacobian)
    std_dev = np.std(jacobian)
    return mean, std_dev

def calculate_ice(fixed_image, moving_image, final_transform):
    # Resample the moving image using the final transform
    moving_resampled = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())

    # To approximate ICE without inverse, we can measure the difference directly
    difference_image = sitk.Abs(fixed_image - moving_resampled)

    # Compute the ICE (mean absolute error can be used as a measure of consistency)
    ice = sitk.GetArrayFromImage(difference_image).mean()

    return ice

fixed_image_path = 'C:/Users/Mittal/Desktop/img_reg_data/registered fixed/'
moving_image_path = 'C:/Users/Mittal/Desktop/img_reg_data/new moving/'

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
    fixed_image, moving_image, moving_resampled, transform = b_spline_registration(f'C:/Users/Mittal/Desktop/img_reg_data/registered fixed/{fixed_image_paths[i]}',f'C:/Users/Mittal/Desktop/img_reg_data/new moving/{moving_image_paths[i]}')

    nmi_value = calculate_nmi(fixed_image, moving_resampled)
    print("NMI: ", nmi_value)
    ecc_value = calculate_ecc(fixed_image, moving_resampled)
    print("ECC: ", ecc_value)
    # displacement_field = compute_displacement_field(transform, fixed_image)
    # jacobian = jacobian_determinant(displacement_field)
    # jacobian_mean, jacobian_std = jacobian_statistics(jacobian)
    # print("Jacobian Mean: ", jacobian_mean)
    # print("Jacobian Std: ", jacobian_std)
    ice_value = calculate_ice(fixed_image, moving_image, transform)
    print("ICE: ", ice_value)

    f = open(f"C:/Users/Mittal/Desktop/img_reg_data/bspline_metrics.txt", "a")
    print(moving_image_paths[i], file=f)
    print("NMI: ", nmi_value, file=f)
    print("ECC: ", ecc_value, file=f)
    # print("Jacobian Mean: ", jacobian_mean, file=f)
    # print("Jacobian Std: ", jacobian_std, file=f)
    print("ICE: ", ice_value, file=f)
    f.close()

# Example usage
fixed_image_path = 'C:/Users/Mittal/Desktop/img_reg_data/precontrast.nii'
moving_image_path = 'C:/Users/Mittal/Desktop/img_reg_data/arterial.nii'

fixed_image, moving_image, moving_resampled, transform = b_spline_registration(fixed_image_path, moving_image_path)

f = open(f"C:/Users/Mittal/Desktop/img_reg_data/bspline_metrics.txt", "a")
print("Arterial:", file=f)
f.close()

nmi_value = calculate_nmi(fixed_image, moving_resampled)
print("NMI: ", nmi_value)
ecc_value = calculate_ecc(fixed_image, moving_resampled)
print("ECC: ", ecc_value)
displacement_field = compute_displacement_field(transform, fixed_image)
jacobian = jacobian_determinant(displacement_field)
jacobian_mean, jacobian_std = jacobian_statistics(jacobian)
print("Jacobian Mean: ", jacobian_mean)
print("Jacobian Std: ", jacobian_std)
ice_value = calculate_ice(fixed_image, moving_image, transform)
print("ICE: ", ice_value)

f = open(f"C:/Users/Mittal/Desktop/img_reg_data/bspline_metrics.txt", "a")
print("NMI: ", nmi_value, file=f)
print("ECC: ", ecc_value, file=f)
print("Jacobian Mean: ", jacobian_mean, file=f)
print("Jacobian Std: ", jacobian_std, file=f)
print("ICE: ", ice_value, file=f)
f.close()

fixed_image_path = 'C:/Users/Mittal/Desktop/img_reg_data/precontrast.nii'
moving_image_path = 'C:/Users/Mittal/Desktop/img_reg_data/delayed.nii'

fixed_image, moving_image, moving_resampled, transform = b_spline_registration(fixed_image_path, moving_image_path)

f = open(f"C:/Users/Mittal/Desktop/img_reg_data/bspline_metrics.txt", "a")
print("Delayed:", file=f)
f.close()

nmi_value = calculate_nmi(fixed_image, moving_resampled)
print("NMI: ", nmi_value)
ecc_value = calculate_ecc(fixed_image, moving_resampled)
print("ECC: ", ecc_value)
displacement_field = compute_displacement_field(transform, fixed_image)
jacobian = jacobian_determinant(displacement_field)
jacobian_mean, jacobian_std = jacobian_statistics(jacobian)
print("Jacobian Mean: ", jacobian_mean)
print("Jacobian Std: ", jacobian_std)
ice_value = calculate_ice(fixed_image, moving_image, transform)
print("ICE: ", ice_value)

f = open(f"C:/Users/Mittal/Desktop/img_reg_data/bspline_metrics.txt", "a")
print("NMI: ", nmi_value, file=f)
print("ECC: ", ecc_value, file=f)
print("Jacobian Mean: ", jacobian_mean, file=f)
print("Jacobian Std: ", jacobian_std, file=f)
print("ICE: ", ice_value, file=f)
f.close()

fixed_image_path = 'C:/Users/Mittal/Desktop/img_reg_data/precontrast.nii'
moving_image_path = 'C:/Users/Mittal/Desktop/img_reg_data/portalvenous.nii'

fixed_image, moving_image, moving_resampled, transform = b_spline_registration(fixed_image_path, moving_image_path)

f = open(f"C:/Users/Mittal/Desktop/img_reg_data/bspline_metrics.txt", "a")
print("Portal Venous:", file=f)
f.close()

nmi_value = calculate_nmi(fixed_image, moving_resampled)
print("NMI: ", nmi_value)
ecc_value = calculate_ecc(fixed_image, moving_resampled)
print("ECC: ", ecc_value)
displacement_field = compute_displacement_field(transform, fixed_image)
jacobian = jacobian_determinant(displacement_field)
jacobian_mean, jacobian_std = jacobian_statistics(jacobian)
print("Jacobian Mean: ", jacobian_mean)
print("Jacobian Std: ", jacobian_std)
ice_value = calculate_ice(fixed_image, moving_image, transform)
print("ICE: ", ice_value)

f = open(f"C:/Users/Mittal/Desktop/img_reg_data/bspline_metrics.txt", "a")
print("NMI: ", nmi_value, file=f)
print("ECC: ", ecc_value, file=f)
print("Jacobian Mean: ", jacobian_mean, file=f)
print("Jacobian Std: ", jacobian_std, file=f)
print("ICE: ", ice_value, file=f)
f.close()
