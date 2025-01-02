import os
import random
import shutil
from pathlib import Path

def create_imagenet_c_subset(base_dir, output_dir, num_images_per_class=10):
    """
    Create a reduced subset of ImageNet-C with a fixed number of images per class
    and ensure the same images are selected across all corruption types and severity levels.
    """
    random.seed(42)  # Ensure reproducibility

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # List corruption types (folders in the base directory)
    corruptions = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

    # Sample image names based on the first corruption type and first severity level
    first_corruption = corruptions[0]
    first_severity_dir = os.path.join(base_dir, first_corruption, "1")
    classes = [d for d in os.listdir(first_severity_dir) if os.path.isdir(os.path.join(first_severity_dir, d))]

    sampled_images_per_class = {}

    for cls in classes:
        class_path = os.path.join(first_severity_dir, cls)
        images = os.listdir(class_path)
        sampled_images_per_class[cls] = random.sample(images, min(len(images), num_images_per_class))

    # Copy the sampled images for all corruption types and severity levels
    for corruption in corruptions:
        for severity_level in range(1, 6):  # Levels are 1 to 5
            input_dir = os.path.join(base_dir, corruption, str(severity_level))
            output_corruption_dir = os.path.join(output_dir, corruption, str(severity_level))
            os.makedirs(output_corruption_dir, exist_ok=True)

            for cls, sampled_images in sampled_images_per_class.items():
                class_input_dir = os.path.join(input_dir, cls)
                class_output_dir = os.path.join(output_corruption_dir, cls)
                os.makedirs(class_output_dir, exist_ok=True)

                for image in sampled_images:
                    src = os.path.join(class_input_dir, image)
                    dst = os.path.join(class_output_dir, image)
                    if os.path.exists(src):
                        shutil.copy(src, dst)

    print(f"Subset created at: {output_dir}")

if __name__ == "__main__":
    base_dir = "/home/toniomirri/datasets/Imagenet-C"  # Path to the original ImageNet-C dataset
    output_dir = "/home/toniomirri/datasets/Imagenet-C-Reduced-100"  # Path to save the reduced dataset
    num_images_per_class = 100  # Number of images per class

    create_imagenet_c_subset(base_dir, output_dir, num_images_per_class)
