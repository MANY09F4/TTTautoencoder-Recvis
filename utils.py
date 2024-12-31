#Utils 
import matplotlib.pyplot as plt
import os

# def display_images(original, masked, reconstructed,save_dir,file_name,rec_loss,class_loss,step_iteration):
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)

#     fig, axs = plt.subplots(1, 3, figsize=(15, 5))
#     axs[0].imshow(original.permute(1, 2, 0))
#     axs[0].set_title('Image Originale')
#     axs[0].axis('off')

#     axs[1].imshow(masked.permute(1, 2, 0))
#     axs[1].set_title('Image Masquée')
#     axs[1].axis('off')

#     img = reconstructed.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
#     img = (img - img.min()) / (img.max() - img.min())
#     axs[2].imshow(img)
#     axs[2].set_title('Image Reconstruite')
#     axs[2].axis('off')

#     plt.savefig(os.path.join(save_dir, file_name))
#     plt.close(fig)

def display_images(original, masked, reconstructed_list, save_dir, file_name, rec_losses, class_losses, steps):
    """
    Displays and saves the evolution of reconstructed images with losses at different steps.

    Args:
    - original: Original image (Tensor).
    - masked: Masked image (Tensor).
    - reconstructed_list: List of reconstructed images at different steps (List of Tensors).
    - save_dir: Directory to save the output image.
    - file_name: Name of the output file.
    - rec_losses: List of reconstruction losses at different steps.
    - class_losses: List of classification losses at different steps.
    - steps: List of step numbers corresponding to the images.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    num_reconstructions = len(reconstructed_list)

    # Create a grid for the plot
    fig, axs = plt.subplots(1, 2 + num_reconstructions, figsize=(6 + 3 * num_reconstructions, 5))

    # Original Image
    axs[0].imshow(original.permute(1, 2, 0))
    axs[0].set_title('Original Image')
    axs[0].axis('off')

    # Masked Image
    axs[1].imshow(masked.permute(1, 2, 0))
    axs[1].set_title('Masked Image')
    axs[1].axis('off')

    # Reconstructions with losses
    for i, (reconstructed, rec_loss, class_loss, step) in enumerate(zip(reconstructed_list, rec_losses, class_losses, steps)):
        img = reconstructed.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
        img = (img - img.min()) / (img.max() - img.min())
        axs[2 + i].imshow(img)
        axs[2 + i].set_title(f"Reconstruction: {rec_loss:.2f}\nClassification: {class_loss:.2f}\nStep {step}", color='red')
        axs[2 + i].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, file_name))
    plt.close(fig)

def apply_mask_to_image(image, mask, patch_size):
    """ Applique un masque de patch à une copie de l'image entière. """
    C, H, W = image.shape
    # Transformer l'image en patches
    image_patches = image.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    # Transformer l'image en patches et redimensionner pour la manipulation
    image_patches = image_patches.contiguous().view(C, -1, patch_size*patch_size)
    
    # Ajuster le masque pour correspondre à la dimension des patches
    mask = mask.flatten()
    
    # Appliquer le masque sur les patches
    image_patches[:, mask == 1, :] = 0  # Mettre à zéro les patches masqués
    
    # Reconstruire l'image à partir des patches
    image_patches = image_patches.view(C, H // patch_size, W // patch_size, patch_size, patch_size)
    image_reconstructed = image_patches.permute(0, 1, 3, 2, 4).contiguous().view(C, H, W)
    
    return image_reconstructed