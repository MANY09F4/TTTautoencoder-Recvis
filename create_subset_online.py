import os
import random
import shutil

def create_imagenet_c_subset(base_dir, output_dir, num_images=500):
    """
    Create a subset of ImageNet-C with the same selected images across all corruption types and severity levels.
    """
    random.seed(23)  # Fix the seed for reproducibility

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # List all corruption types
    corruptions = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

    # Reference: the first corruption and severity to choose the 500 images
    first_corruption = corruptions[0]
    first_severity_dir = os.path.join(base_dir, first_corruption, "1")  # Use severity level 1
    all_files = []

    # Collect all images from all classes in the first corruption/severity
    for cls in os.listdir(first_severity_dir):
        class_dir = os.path.join(first_severity_dir, cls)
        if os.path.isdir(class_dir):
            files = [os.path.join(cls, img) for img in os.listdir(class_dir) if img.endswith(('.JPEG', '.jpg', '.png'))]
            all_files.extend(files)

    # Randomly select 500 images
    selected_files = random.sample(all_files, min(len(all_files), num_images))

    # Copy the same selected images for each corruption and severity level
    for corruption in corruptions:
        for severity in range(1, 6):  # Severity levels 1 to 5
            corruption_dir = os.path.join(base_dir, corruption, str(severity))
            output_corruption_dir = os.path.join(output_dir, corruption, str(severity))
            os.makedirs(output_corruption_dir, exist_ok=True)

            for selected_file in selected_files:
                cls, filename = os.path.split(selected_file)  # Extract class and filename
                src = os.path.join(corruption_dir, cls, filename)
                dest_dir = os.path.join(output_corruption_dir, cls)
                os.makedirs(dest_dir, exist_ok=True)
                dest = os.path.join(dest_dir, filename)

                # Copy the file if it exists
                if os.path.exists(src):
                    shutil.copy(src, dest)

    print(f"Subset created at: {output_dir} with {num_images} images per corruption level.")


if __name__ == "__main__":
    base_dir = "/home/toniomirri/datasets/Imagenet-C"  # Path to the original ImageNet-C dataset
    output_dir = "/home/toniomirri/datasets/Imagenet-C-Reduced-500"  # Path to save the reduced dataset
    num_images = 500  # Total number of images to select

    create_imagenet_c_subset(base_dir, output_dir, num_images)
