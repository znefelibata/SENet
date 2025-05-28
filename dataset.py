import argparse
import better_exceptions
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class FaceDataset(Dataset):
    def __init__(self, data_dir, data_type, img_size=224, augment=False):
        assert data_type in ("train", "valid", "test")
        self.img_dir = Path(data_dir) / data_type
        self.img_size = img_size
        self.augment = augment

        # 标准图像归一化参数 (ImageNet)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        # 设置 transform
        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])

        self.x = []
        self.y = []

        for img_path in self.img_dir.glob("*.jpg"):
            try:
                age = int(img_path.stem.split("_")[0])
                self.x.append(str(img_path))
                self.y.append(age)
            except Exception as e:
                print(f"跳过无法解析的文件: {img_path.name}, 错误: {e}")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        img_path = self.x[idx]
        age = self.y[idx]

        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        age_label = np.clip(age, 0, 100)

        return img, age_label


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_dir", type=str, required=True)
    args = parser.parse_args()

    for split in ["train", "valid", "test"]:
        dataset = FaceDataset(args.data_dir, split, augment=(split == "train"))
        print(f"{split} dataset len: {len(dataset)}")


if __name__ == '__main__':
    main()
