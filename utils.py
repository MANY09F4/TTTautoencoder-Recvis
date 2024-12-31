#Utils 
import matplotlib.pyplot as plt
import os

def display_images(original, masked, reconstructed,save_dir,file_name):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(original.permute(1, 2, 0))
    axs[0].set_title('Image Originale')
    axs[0].axis('off')

    axs[1].imshow(masked.permute(1, 2, 0))
    axs[1].set_title('Image Masquée')
    axs[1].axis('off')

    img = reconstructed.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
    img = (img - img.min()) / (img.max() - img.min())
    axs[2].imshow(img)
    axs[2].set_title('Image Reconstruite')
    axs[2].axis('off')

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