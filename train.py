import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler
import timm
from glendaDataset import GlendaDataset, HealthyDataset

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    im_size = 224
    pathology_dataset = GlendaDataset('coco_train.json', im_size=im_size, task='classification', is_training=True)
    val_pathology_dataset = GlendaDataset('coco_val.json', im_size=im_size, task='classification', is_training=False)
    healthy_dataset = HealthyDataset('no_pathology/frames/', im_size=im_size, max_images=1000)

    train_dataset = ConcatDataset([pathology_dataset, healthy_dataset])
    print(f"Total training samples: {len(train_dataset)}")

    # Class-balancing 
    labels = [1] * len(pathology_dataset) + [0] * len(healthy_dataset)
    class_counts = [len(healthy_dataset), len(pathology_dataset)]
    class_weights = [1.0 / count for count in class_counts]
    sample_weights = [class_weights[label] for label in labels]
    sample_weights = torch.DoubleTensor(sample_weights)

    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_pathology_dataset, batch_size=16, shuffle=False, num_workers=0)

    model = timm.create_model('convnextv2_nano', pretrained=True, num_classes=1)
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=10)

    epochs = 10
    best_val_loss = float('inf')

    print("Beginning training...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for images, is_pathological, _ in train_loader:
            images = images.to(device)
            targets = is_pathological.to(device).unsqueeze(1)

            optimiser.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimiser.step()

            train_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(outputs) >= 0.5).float()
            correct_train += (preds == targets).sum().item()
            total_train += targets.size(0)

        scheduler.step()
        epoch_train_loss = train_loss / len(train_dataset)
        epoch_train_acc = correct_train / total_train

        # Validation
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, is_pathological, _ in val_loader:
                images = images.to(device)
                targets = is_pathological.to(device).unsqueeze(1)

                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * images.size(0)

                preds = (torch.sigmoid(outputs) >= 0.5).float()
                correct_val += (preds == targets).sum().item()
                total_val += targets.size(0)

        epoch_val_loss = val_loss / len(val_pathology_dataset)
        epoch_val_acc = correct_val / total_val if total_val > 0 else 0

        print(f"Epoch {epoch+1:02d} | Train Loss (BCE): {epoch_train_loss:.4f} - Acc (Accuracy): {epoch_train_acc:.4f} | Val Loss: {epoch_val_loss:.4f} - Acc (Accuracy): {epoch_val_acc:.4f}")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), 'best_stage1_model.pth')
            print("Saved new best model weights.")

if __name__ == "__main__":
    main()