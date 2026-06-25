import argparse
import glob
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from pose_format import Pose
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from model import SignTransformer


class PoseDataset(Dataset):
    def __init__(self, data_dir, is_train=True):
        self.data_dir = data_dir
        self.is_train = is_train
        self.file_paths = []
        self.labels = []
        self.classes = sorted(
            d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))
        )
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        for cls in self.classes:
            cls_dir = os.path.join(data_dir, cls)
            files = glob.glob(os.path.join(cls_dir, "*.pose"))
            self.file_paths.extend(files)
            self.labels.extend([self.class_to_idx[cls]] * len(files))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        with open(file_path, "rb") as f:
            pose = Pose.read(f.read())

        if self.is_train:
            pose.augment2d(rotation_std=0.2, scale_std=0.2, shear_std=0.2)

        data = pose.body.data
        data = np.nan_to_num(data)
        data = data.squeeze(1)

        mean = data.mean()
        std = data.std()
        if std > 1e-6:
            data = (data - mean) / std

        frames, points, dims = data.shape
        data = data.reshape(frames, points * dims)

        x = torch.tensor(data, dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


def train_model(data_dir, output_path, num_epochs=50, batch_size=32, lr=1e-4):
    dataset = PoseDataset(data_dir, is_train=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    num_classes = len(dataset.classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SignTransformer(input_dim=126, num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        acc = correct / total
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader):.4f}, Acc: {acc:.4f}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"Training finished. Model saved to {output_path}")


def main():
    from paths import DEFAULT_WEIGHTS

    parser = argparse.ArgumentParser(description="Train ISL alphabet SignTransformer")
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Directory of .pose files grouped by class (A-Z folders)",
    )
    parser.add_argument("--output", type=str, default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    train_model(
        data_dir=args.data_dir,
        output_path=Path(args.output),
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
