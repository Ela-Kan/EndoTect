"""
Author: Ela Kanani
Dataset: Gynecologic Laparoscopy Endometriosis Dataset (GLENDA v1.5): A. Leibetseder, S. Kletz, K. Schoeffmann, S. Keckstein and J. Keckstein. 2020. GLENDA: Gynecologic Laparoscopy Endometriosis Dataset. In Proceedings of the 26th International Conference on Multimedia Modeling, MMM 2020. Springer, Cham.

"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pycocotools.coco import COCO
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

class GlendaDataset(Dataset):
    def __init__(self, coco_json_path, im_size=512, is_training=False, task='classification'):
        """
        Args:
            coco_json_path (str): Path to split coco json file
            im_size (int): Dimension to resize images/masks to 
            is_training (bool): If True, apply training augmentations
            task (str): 'classification', 'segmentation', or 'both'
        """
        self.coco = COCO(coco_json_path)
        self.ids = self.coco.getImgIds()
        self.im_size = im_size
        self.task = task
        self.is_training = is_training
        
        self.category_ids = self.coco.getCatIds()

        if self.is_training:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.CLAHE(p=0.5),
                A.Blur(p=0.5),
                A.Perspective(p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                A.Resize(self.im_size, self.im_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(self.im_size, self.im_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
            current_id = self.ids[idx]
            current_image_info = self.coco.loadImgs([current_id])[0]
            image_path = current_image_info["coco_url"]
            
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f"Failed to load image at: {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            annotation_ids = self.coco.getAnnIds(imgIds=[current_id], catIds=self.category_ids, iscrowd=None)
            annotations = self.coco.loadAnns(annotation_ids)
        
            is_pathological = 1.0 if len(annotations) > 0 else 0.0

            transformed = self.transform(image=image)
            return transformed['image'], torch.tensor(is_pathological, dtype=torch.float32)


class HealthyDataset(Dataset):
    def __init__(self, folder_path, im_size=512, max_images=None):
        """
        Reads raw, unannotated healthy frames from folder  and maps them as negatives (0.0).
        Args:
            max_images : int, number of maximum frames to use for training to prevent imbalance
        """
        self.folder_path = folder_path
        self.im_size = im_size
        
        # Gather all images inside folder recursively
        self.image_paths = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(root, file))
        
        # Limit images to prevent extreme imbalance
        if max_images and len(self.image_paths) > max_images:
            np.random.seed(1311)
            self.image_paths = list(np.random.choice(self.image_paths, max_images, replace=False))

        self.transform = A.Compose([
            A.Resize(self.im_size, self.im_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to load healthy image at: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        image_tensor = self.transform(image=image)['image']
        
        # Target returns: (image, is_pathological=0.0, multi_label=[0, 0, 0, 0])
        return image_tensor, torch.tensor(0.0), torch.tensor([0.0, 0.0, 0.0, 0.0])
