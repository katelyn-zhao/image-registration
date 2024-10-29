import ants
import numpy as np
from sklearn.metrics import mutual_info_score
from scipy.stats import entropy
from skimage.metrics import normalized_mutual_information as skimage_nmi

# 1. Normalized Mutual Information (NMI)
def normalized_mutual_information(image1, image2, bins=32):
    hist_2d, _, _ = np.histogram2d(image1.ravel(), image2.ravel(), bins=bins)
    joint_entropy = -np.sum(hist_2d * np.log(hist_2d + 1e-9))
    entropy1 = -np.sum(np.histogram(image1, bins=bins)[0] * np.log(np.histogram(image1, bins=bins)[0] + 1e-9))
    entropy2 = -np.sum(np.histogram(image2, bins=bins)[0] * np.log(np.histogram(image2, bins=bins)[0] + 1e-9))
    return (entropy1 + entropy2) / joint_entropy

# Alternative using skimage
def nmi_skimage(image1, image2):
    return skimage_nmi(image1, image2)

# 2. Entropy Correlation Coefficient (ECC)
def entropy_correlation_coefficient(image1, image2, bins=32):
    hist_2d, _, _ = np.histogram2d(image1.ravel(), image2.ravel(), bins=bins)
    joint_entropy = entropy(hist_2d.ravel())
    entropy1 = entropy(np.histogram(image1, bins=bins)[0])
    entropy2 = entropy(np.histogram(image2, bins=bins)[0])
    ecc = 2 * (entropy1 + entropy2 - joint_entropy) / (entropy1 + entropy2)
    return ecc

# 3. Jacobian Statistics (Mean and Standard Deviation)
def jacobian_determinant(displacement_field):
    grad_x = np.gradient(displacement_field[..., 0], axis=0)
    grad_y = np.gradient(displacement_field[..., 1], axis=1)
    grad_z = np.gradient(displacement_field[..., 2], axis=2)
    jacobian = grad_x[..., 0] * (grad_y[..., 1] * grad_z[..., 2] - grad_y[..., 2] * grad_z[..., 1]) - \
               grad_x[..., 1] * (grad_y[..., 0] * grad_z[..., 2] - grad_y[..., 2] * grad_z[..., 0]) + \
               grad_x[..., 2] * (grad_y[..., 0] * grad_z[..., 1] - grad_y[..., 1] * grad_z[..., 0])
    return jacobian

def jacobian_statistics(jacobian):
    mean = np.mean(jacobian)
    std_dev = np.std(jacobian)
    return mean, std_dev

# 4. Inverse Consistency Error (ICE)
def inverse_consistency_error(forward_transform, inverse_transform):
    ice = np.mean(np.linalg.norm(forward_transform + inverse_transform, axis=-1))
    return ice

# Main function to load images, register, and calculate metrics
def main(fixed_image_path, moving_image_path):
    # Load images using ANTsPy
    fixed_image = ants.image_read(fixed_image_path)
    moving_image = ants.image_read(moving_image_path)
    
    print("Loaded Images")

    # Perform registration
    registered_image = ants.registration(fixed=fixed_image, moving=moving_image, type_of_transform='SyN')

    # Extract the registered image
    moving_registered = registered_image['warpedmovout']
    forward_displacement_field = registered_image['fwdtransforms'][-1]
    
    print("Image Registration Complete")

    if save_transform_path:
        for i, transform_path in enumerate(registered_image['fwdtransforms']):
            ants.write_transform(transform_path, f"{save_transform_path}_fwd_transform_{i}.mat")
        for i, transform_path in enumerate(registered_image['invtransforms']):
            ants.write_transform(transform_path, f"{save_transform_path}_inv_transform_{i}.mat")
        print(f"Transformations saved at {save_transform_path}")

    # Compute NMI
    nmi_value = normalized_mutual_information(fixed_image.numpy(), moving_registered.numpy())
    nmi_skimage_value = nmi_skimage(fixed_image.numpy(), moving_registered.numpy())
    print(f"NMI (Custom): {nmi_value}")
    print(f"NMI (Skimage): {nmi_skimage_value}")

    # Compute ECC
    ecc_value = entropy_correlation_coefficient(fixed_image.numpy(), moving_registered.numpy())
    print(f"ECC: {ecc_value}")

    # Jacobian calculations (Assuming you have a displacement field, you may need to modify this part)
    displacement_field_path = registered_image['fwdtransforms'][0]
    displacement_field = ants.image_read(displacement_field_path)  # Load the transform as an image

#     # Compute the Jacobian determinant
#     jacobian = jacobian_determinant(displacement_field.numpy())
#     jacobian_mean, jacobian_std_dev = jacobian_statistics(jacobian)
#     print(f"Jacobian Mean: {jacobian_mean}")
#     print(f"Jacobian Std Dev: {jacobian_std_dev}")
#     if displacement_field is not None:
#         jacobian = jacobian_determinant(displacement_field.numpy())
#         jacobian_mean, jacobian_std_dev = jacobian_statistics(jacobian)
#         print(f"Jacobian Mean: {jacobian_mean}")
#         print(f"Jacobian Std Dev: {jacobian_std_dev}")
        
    #Compute ICE
    backward_registration = ants.registration(fixed=moving_image, moving=fixed_image, type_of_transform='SyN')
    backward_displacement_field = backward_registration['fwdtransforms'][-1]  # Get the backward displacement field
    
    inverse_mapped_image = ants.apply_transforms(fixed=moving_image, moving=moving_registered, transformlist=backward_displacement_field)
    
    ice = np.mean((fixed_image.numpy() - inverse_mapped_image.numpy())**2)
    print("ICE: ", ice)

    # ICE calculation would require forward and inverse transforms
    # For demonstration, we'll skip this unless you have actual transforms
    # forward_transform = ...
    # inverse_transform = ...
    # ice_value = inverse_consistency_error(forward_transform, inverse_transform)
    # print(f"ICE: {ice_value}")

# Example usage call
fixed_image_path = "precontrast.nii"  # Replace with your actual image path
moving_image_path = "arterial.nii"  # Replace with your actual image path

main(fixed_image_path, moving_image_path)

# Example usage call
fixed_image_path = "precontrast.nii"  # Replace with your actual image path
moving_image_path = "delayed.nii"  # Replace with your actual image path

main(fixed_image_path, moving_image_path)

# Example usage call
fixed_image_path = "precontrast.nii"  # Replace with your actual image path
moving_image_path = "precontrast.nii"  # Replace with your actual image path

main(fixed_image_path, moving_image_path)
