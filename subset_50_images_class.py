import os
import shutil
import random


def create_subset_imagenet_c():
    """
    Crée un sous-dataset ImageNet-C avec des classes spécifiques et un nombre limité d'images par classe.
    Les sous-dossiers vides des classes non sélectionnées sont également créés pour éviter les bugs.
    """
    # Chemins fixes
    base_dir = "/home/toniomirri/datasets/Imagenet-C"
    output_dir = "/home/toniomirri/datasets/Imagenet-C-Reduced-3-classes"
    class_ids = ["n01484850", "n01770393", "n02114367"]  # Classes spécifiques
    num_images = 50  # Nombre maximum d'images par classe et par niveau

    random.seed(42)  # Fixer la seed pour la reproductibilité

    # Liste des types de corruptions dans ImageNet-C
    corruptions = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

    for corruption in corruptions:
        print(f"Traitement de la corruption : {corruption}")

        for severity in range(1, 6):  # Niveaux de corruption de 1 à 5
            corruption_level_dir = os.path.join(base_dir, corruption, str(severity))
            output_corruption_level_dir = os.path.join(output_dir, corruption, str(severity))
            os.makedirs(output_corruption_level_dir, exist_ok=True)

            # Récupérer toutes les classes disponibles dans ce niveau de corruption
            all_classes = [
                d for d in os.listdir(corruption_level_dir)
                if os.path.isdir(os.path.join(corruption_level_dir, d))
            ]

            for class_id in all_classes:
                class_dir = os.path.join(corruption_level_dir, class_id)
                output_class_dir = os.path.join(output_corruption_level_dir, class_id)
                os.makedirs(output_class_dir, exist_ok=True)

                # Si la classe fait partie des classes sélectionnées
                if class_id in class_ids:
                    if os.path.exists(class_dir):
                        # Liste des fichiers dans la classe
                        files = [f for f in os.listdir(class_dir) if f.endswith(('.JPEG', '.jpg', '.png'))]

                        # Sélection aléatoire de num_images fichiers
                        selected_files = random.sample(files, min(len(files), num_images))

                        for file_name in selected_files:
                            src = os.path.join(class_dir, file_name)
                            dest = os.path.join(output_class_dir, file_name)
                            shutil.copy(src, dest)

    print(f"Sous-dataset créé avec succès dans : {output_dir}")


if __name__ == "__main__":
    create_subset_imagenet_c()
