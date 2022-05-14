import os
from torch.utils.data import Dataset
from PIL import Image


class MyDataset(Dataset):
    def __init__(self, root, annot, cls_mapping, transform):
        self.root = root
        with open(os.path.join(root, annot), 'r') as f:
            self.img_path = f.readlines()
        self.cls_mapping = cls_mapping
        self.transform = transform

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img_path = self.img_path[idx].strip()
        img = Image.open(os.path.join(self.root, img_path))
        img = self.transform(img)
        label = img_path.split('/')[-2]
        return img, self.cls_mapping[label]
