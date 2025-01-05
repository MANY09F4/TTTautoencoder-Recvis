import os
import shutil

def create_online_subfolders(base_dir, levels_to_merge=[1, 3, 5]):
    """
    Crée un sous-dossier "online" dans chaque dossier de corruption.
    Regroupe les images des niveaux spécifiés dans levels_to_merge.

    :param base_dir: Chemin du dataset réduit (e.g., Imagenet-C-Reduced-500).
    :param levels_to_merge: Liste des niveaux de corruption à inclure dans "online".
    """
    corruptions = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

    for corruption in corruptions:
        corruption_path = os.path.join(base_dir, corruption)
        online_path = os.path.join(corruption_path, "online")
        os.makedirs(online_path, exist_ok=True)

        print(f"Processing corruption: {corruption}")

        for level in levels_to_merge:
            level_path = os.path.join(corruption_path, str(level))
            if not os.path.exists(level_path):
                print(f"Warning: Level {level} does not exist in {corruption_path}")
                continue

            classes = os.listdir(level_path)

            ordered_cls = sorted(classes)

            for class_name in ordered_cls:
                class_source_path = os.path.join(level_path, class_name)
                class_name_online_level = class_name[:1] + f'{level}' + class_name[2:]
                class_target_path = os.path.join(online_path, class_name_online_level)
                os.makedirs(class_target_path, exist_ok=True)

                if not os.path.isdir(class_source_path):
                    continue

                for img_file in os.listdir(class_source_path):
                    src = os.path.join(class_source_path, img_file)
                    dst = os.path.join(class_target_path, img_file)

                    if not os.path.exists(dst):
                        shutil.copy(src, dst)

        print(f"Finished processing {corruption}, 'online' folder created.")

if __name__ == "__main__":
    base_dir = "/home/toniomirri/datasets/Imagenet-C-Reduced-500"  # Chemin vers le dataset réduit
    create_online_subfolders(base_dir)
