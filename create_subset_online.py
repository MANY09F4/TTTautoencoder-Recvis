import os
import random
import shutil

def create_random_imagenet_c_subset(base_dir, output_dir, num_images_per_level=500, num_corruptions=10):
    """
    Create a reduced subset of ImageNet-C with a fixed number of random images per corruption level,
    limited to a subset of corruption types.
    """
    random.seed(42)  # Ensure reproducibility

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # List all corruption types (folders in the base directory)
    all_corruptions = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    selected_corruptions = random.sample(all_corruptions, num_corruptions)  # Pick 10 corruption types

    print(f"Selected corruptions: {selected_corruptions}")

    # Process each corruption type
    for corruption in selected_corruptions:
        for severity_level in range(1, 6):  # Severity levels 1 to 5
            input_dir = os.path.join(base_dir, corruption, str(severity_level))
            output_dir_level = os.path.join(output_dir, corruption, str(severity_level))
            os.makedirs(output_dir_level, exist_ok=True)

            # Collect all images across all classes
            all_images = []
            for cls in os.listdir(input_dir):
                class_dir = os.path.join(input_dir, cls)
                if os.path.isdir(class_dir):
                    all_images += [os.path.join(class_dir, img) for img in os.listdir(class_dir)]

            # Randomly sample 500 images
            sampled_images = random.sample(all_images, min(len(all_images), num_images_per_level))

            # Copy sampled images to the output directory
            for src in sampled_images:
                # Preserve class directory structure in the output
                relative_path = os.path.relpath(src, input_dir)
                dst = os.path.join(output_dir_level, relative_path)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy(src, dst)

    print(f"Random subset created at: {output_dir}")

if __name__ == "__main__":
    base_dir = "/home/toniomirri/datasets/Imagenet-C"  # Path to ImageNet-C
    output_dir = "/home/toniomirri/datasets/Imagenet-C-subset-online"  # Output directory for the subset
    create_random_imagenet_c_subset(base_dir, output_dir, num_images_per_level=500, num_corruptions=10)
