from torchvision import datasets
from typing import Optional, Callable, Tuple, Any
import torch
import numpy as np
from data.imagenet_r import ImageFolderSafe


class ExtendedImageFolder(ImageFolderSafe):
    def __init__(self, root: str, batch_size: int = 1, steps_per_example: int = 1, minimizer = None, transform: Optional[Callable] = None, single_crop: bool = False, start_index: int = 0):
        super().__init__(root=root, transform=transform)
        self.batch_size = batch_size
        self.minimizer = minimizer
        self.steps_per_example = steps_per_example
        self.single_crop = single_crop
        self.start_index = start_index

    def __len__(self):
        mult = self.steps_per_example * self.batch_size
        mult *= (super().__len__() if self.minimizer is None else len(self.minimizer))
        return mult


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        real_index = (index // self.steps_per_example) + self.start_index
        if self.minimizer is not None:
            real_index = self.minimizer[real_index]
        path, target = self.samples[real_index]
        sample = self.loader(path)
        if self.transform is not None and not self.single_crop:
            samples = torch.stack([self.transform(sample) for i in range(self.batch_size)], axis=0)
        elif self.transform and self.single_crop:
            s = self.transform(sample)
            samples = torch.stack([s for i in range(self.batch_size)], axis=0)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return samples, target

class ExtendedImageFolder_online(ImageFolderSafe):
    def __init__(self, root: str, batch_size: int = 1, initial_steps: int = 250, subsequent_steps: int = 1, minimizer=None, transform: Optional[Callable] = None, single_crop: bool = False, start_index: int = 0):
        super().__init__(root=root, transform=transform)
        self.batch_size = batch_size
        self.minimizer = minimizer
        self.initial_steps = initial_steps
        self.subsequent_steps = subsequent_steps
        self.single_crop = single_crop
        self.start_index = start_index
        self.steps_per_example = [self.initial_steps] + [self.subsequent_steps] * (len(self.samples) - 1)

    def __len__(self):
        # Calculate total length considering varying steps per example
        if self.minimizer is not None:
            total_steps = sum(self.steps_per_example[idx] for idx in self.minimizer)
        else:
            total_steps = sum(self.steps_per_example)
        return total_steps * self.batch_size

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        # Determine the actual image index based on the cumulative sum of steps
        cumulative_steps = 0
        real_index = 0
        for steps in self.steps_per_example:
            cumulative_steps += steps
            if index < cumulative_steps:
                break
            real_index += 1

        if self.minimizer is not None:
            real_index = self.minimizer[real_index]

        path, target = self.samples[real_index]
        sample = self.loader(path)
        if self.transform is not None and not self.single_crop:
            samples = torch.stack([self.transform(sample) for i in range(self.batch_size)], axis=0)
        elif self.transform and self.single_crop:
            s = self.transform(sample)
            samples = torch.stack([s for i in range(self.batch_size)], axis=0)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return samples, target

# class ExtendedImageFolder_online_shuffle(datasets.ImageFolder):
#     def __init__(self, root: str, batch_size: int = 1, initial_steps: int = 250, subsequent_steps: int = 1, transform: Optional[Callable] = None, single_crop: bool = False, start_index: int = 0):
#         super().__init__(root=root, transform=transform)
#         self.batch_size = batch_size
#         self.initial_steps = initial_steps
#         self.subsequent_steps = subsequent_steps
#         self.single_crop = single_crop
#         self.start_index = start_index

#         # Création de l'ordre mélangé des indices
#         self.indices = np.arange(len(self.samples))
#         np.random.shuffle(self.indices)

#         # Assigner le nombre de pas par exemple
#         self.steps_per_example = [self.initial_steps] + [self.subsequent_steps] * (len(self.samples) - 1)

#     def __len__(self):
#         # Calculer la longueur totale en tenant compte du nombre de pas par exemple
#         total_steps = np.sum(np.array(self.steps_per_example)[self.indices])  # Utilisation des indices mélangés
#         return total_steps * self.batch_size

#     def __getitem__(self, index: int) -> Tuple[Any, Any]:
#         # Déterminer l'index de l'image réelle basé sur la somme cumulative des pas
#         cumulative_steps = 0
#         real_index = 0
#         for i, steps in enumerate(self.steps_per_example):
#             cumulative_steps += steps
#             if index < cumulative_steps:
#                 real_index = i
#                 break

#         # Utiliser l'indice mélangé pour obtenir le vrai index
#         shuffled_real_index = self.indices[real_index]

#         path, target = self.samples[shuffled_real_index]
#         sample = self.loader(path)
#         if self.transform is not None and not self.single_crop:
#             samples = torch.stack([self.transform(sample) for _ in range(self.batch_size)], axis=0)
#         elif self.transform and self.single_crop:
#             s = self.transform(sample)
#             samples = torch.stack([s for _ in range(self.batch_size)], axis=0)

#         return samples, target

class ExtendedImageFolder_online_shuffle(datasets.ImageFolder):
    def __init__(self, root: str, batch_size: int = 1, initial_steps: int = 250, subsequent_steps: int = 1,
                 shuffle_seed: Optional[int] = None, transform: Optional[Callable] = None,
                 single_crop: bool = False, start_index: int = 0):
        super().__init__(root=root, transform=transform)
        self.batch_size = batch_size
        self.initial_steps = initial_steps
        self.subsequent_steps = subsequent_steps
        self.single_crop = single_crop
        self.start_index = start_index

        # Shuffle indices using the provided shuffle seed
        rng = np.random.default_rng(shuffle_seed)
        self.indices = rng.permutation(len(self.samples))

        # Assign steps per example
        self.steps_per_example = [self.initial_steps] + [self.subsequent_steps] * (len(self.samples) - 1)

        # Compute cumulative steps for efficient index mapping
        self.cumulative_steps = np.cumsum(np.array(self.steps_per_example)[self.indices])

    def __len__(self):
        """
        Total number of steps across all shuffled indices.
        """
        return self.cumulative_steps[-1] * self.batch_size

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Get the sample and target corresponding to the given step index.

        Args:
            index (int): The global index for the dataset, considering steps.

        Returns:
            Tuple[Any, Any]: Transformed sample and its target class.
        """
        # Find the index of the cumulative step
        real_index = np.searchsorted(self.cumulative_steps, index // self.batch_size, side="right")
        shuffled_real_index = self.indices[real_index]

        # Load the image and target
        path, target = self.samples[shuffled_real_index]
        sample = self.loader(path)

        # Apply transformations
        if self.transform is not None and not self.single_crop:
            samples = torch.stack([self.transform(sample) for _ in range(self.batch_size)], axis=0)
        elif self.transform and self.single_crop:
            s = self.transform(sample)
            samples = torch.stack([s for _ in range(self.batch_size)], axis=0)

        return samples, target


class ExtendedSplitImageFolder(ExtendedImageFolder):
    def __init__(self, root: str, batch_size: int = 1, steps_per_example: int = 1, split: int = 0, minimizer = None,
                 transform: Optional[Callable] = None, single_crop: bool = False, start_index: int = 0):
        super().__init__(root=root, batch_size=batch_size, steps_per_example=steps_per_example, minimizer=minimizer,
                         transform=transform, single_crop=single_crop, start_index=start_index)
        self.new_samples = []
        for i, sample in enumerate(self.samples):
            if i % 20 == split:
                self.new_samples.append(sample)
        self.samples = self.new_samples
