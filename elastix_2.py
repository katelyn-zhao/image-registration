import SimpleITK as sitk

import numpy as np
import matplotlib.pyplot as plt

# Load fixed and moving images
fixed_image = sitk.ReadImage("C:/Users/Mittal/Desktop/img_reg_data/precontrast.nii")
moving_image = sitk.ReadImage("C:/Users/Mittal/Desktop/img_reg_data/portalvenous.nii")
# Set up the registration method
registration_method = sitk.ImageRegistrationMethod()

# Set the metric
registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)

# Set the optimizer
registration_method.SetOptimizerAsGradientDescentLineSearch(learningRate=1.0, numberOfIterations=100)

# Set the transform
initial_transform = sitk.BSplineTransformInitializer(fixed_image, transformDomainMeshSize=[10, 10, 10], order=3)
registration_method.SetInitialTransform(initial_transform, inPlace=False)

# Set the interpolator
registration_method.SetInterpolator(sitk.sitkBSpline)

# Set multi-resolution strategy
registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])


## either use above code or upload parameters.txt using bottom code
"""# Set up the registration method
registration_method = sitk.ImageRegistrationMethod()

# Load parameters from the file
parameter_file = 'parameter.txt'  # Ensure this path is correct
registration_method.SetParameterMap(sitk.ReadParameterFile(parameter_file))

# Enable logging for debugging
registration_method.LogToConsoleOn()
registration_method.LogToFileOn("elastix_log.txt")

# Execute registration
final_transform = registration_method.Execute(fixed_image, moving_image)
"""

print("before")
# Execute registration
final_transform = registration_method.Execute(fixed_image, moving_image)
print("after")

def visualize_registration_slices(fixed_image_path, registered_image_path, slice_index=None):
    """
    Visualizes a specific slice from the fixed and registered moving images.

    Parameters:
    - fixed_image_path: Path to the fixed image (NIfTI format).
    - registered_image_path: Path to the registered moving image (NIfTI format).
    - slice_index: Index of the slice to visualize. If None, the middle slice will be used.
    """
    # Load the images
    fixed_image = sitk.ReadImage(fixed_image_path)
    moving_resampled = sitk.ReadImage(registered_image_path)

    # Convert to NumPy arrays
    fixed_array = sitk.GetArrayFromImage(fixed_image)
    moving_array = sitk.GetArrayFromImage(moving_resampled)

    # Determine the slice index
    if slice_index is None:
        slice_index = fixed_array.shape[0] // 2  # Use the middle slice

    # Extract the slices
    fixed_slice = fixed_array[slice_index, :, :]
    moving_slice = moving_array[slice_index, :, :]

    # Create a figure to display the slices
    plt.figure(figsize=(12, 6))

    # Plot the fixed image slice
    plt.subplot(1, 2, 1)
    plt.title('Fixed Image Slice')
    plt.imshow(fixed_slice, cmap='gray')
    plt.axis('off')

    # Plot the registered moving image slice
    plt.subplot(1, 2, 2)
    plt.title('Registered Moving Image Slice')
    plt.imshow(moving_slice, cmap='gray')
    plt.axis('off')

    # Show the plots
    plt.show()

visualize_registration_slices('C:/Users/Mittal/Desktop/img_reg_data/precontrast.nii', 'registered_image.nii')


