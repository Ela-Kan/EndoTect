"""
Stage 1 Quantitative Evaluation Script.
Computes and reports metrics at the optimised decision threshold.

Author: Ela Kanani
"""

import torch
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset
import timm
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, precision_score, recall_score, accuracy_score

from glenda_dataset import GlendaClassificationDataset, HealthyDataset

class UnseenHealthyDataset(HealthyDataset):
    """Slices the healthy dataset to exclude training files. (i.e. the first 1000 images)"""
    def __init__(self, folder_path, im_size=224, max_images=300):
        super().__init__(folder_path, im_size=im_size)
        self.image_paths.sort()
        if len(self.image_paths) > 1000:
            self.image_paths = self.image_paths[1000:]
        if max_images and len(self.image_paths) > max_images:
            self.image_paths = self.image_paths[:max_images]

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Targeting device: {device}")

    im_size = 224
    test_pathology_ds = GlendaClassificationDataset('coco_test.json', im_size=im_size, is_training=False)
    test_healthy_ds = UnseenHealthyDataset('no_pathology/', im_size=im_size, max_images=300)

    test_dataset = ConcatDataset([test_pathology_ds, test_healthy_ds])
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)
    print(f"Loaded {len(test_dataset)} unseen test samples ({len(test_pathology_ds)} pathology, {len(test_healthy_ds)} healthy).")

    model = timm.create_model('convnextv2_nano', pretrained=False, num_classes=1)
    model.load_state_dict(torch.load('best_stage1_model.pth', map_location=device))
    model = model.to(device)
    model.eval()

    raw_probabilities = []
    all_targets = []

    print("Running model inference...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy()
            raw_probabilities.extend(probs.flatten())
            all_targets.extend(labels.numpy())

    raw_probabilities = np.array(raw_probabilities)
    all_targets = np.array(all_targets)

    best_threshold = 0.2118 # found by investigating the levels of threshold to minimise false negatives
    all_preds = (raw_probabilities >= best_threshold).astype(np.float32)

    tn, fp, fn, tp = confusion_matrix(all_targets, all_preds).ravel()

    accuracy = accuracy_score(all_targets, all_preds)
    sensitivity = recall_score(all_targets, all_preds) 
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = precision_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds)
    auc_roc = roc_auc_score(all_targets, raw_probabilities)

    print(f"Decision Boundary:                {best_threshold:.4f}")
    print(f"Classification Accuracy:          {accuracy*100:.2f}%")
    print(f"Sensitivity (Recall):             {sensitivity*100:.2f}%  (Detecting Endometriosis)")
    print(f"Specificity:                      {specificity*100:.2f}%  (Clearing Healthy Tissue)")
    print(f"Precision (PPV):                  {precision*100:.2f}%")
    print(f"F1-Score:                         {f1:.4f}")
    print(f"Area Under ROC Curve (AUC):       {auc_roc*100:.2f}%")
    print("CONFUSION MATRIX:")
    print(f"True Negatives (Healthy cleared):     {tn}")
    print(f"False Positives (Healthy flagged):    {fp}")
    print(f"False Negatives (Pathology missed):   {fn}")
    print(f"True Positives (Pathology detected):  {tp}")

if __name__ == "__main__":
    main()