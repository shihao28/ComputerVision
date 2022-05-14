import os
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import labelme


class MyDataset(Dataset):
    def __init__(self, root, annot, transform):
        self.root = root
        with open(os.path.join(root, annot), 'r') as f:
            self.img_path = f.readlines()
        self.transform = transform
        self.class_name_to_id = {}
        for i, line in enumerate(open(os.path.join(root, "labels.txt")).readlines()):
            class_id = i - 1  # starts with -1
            class_name = line.strip()
            if class_id == -1:
                assert class_name == "__ignore__"
                continue
            elif class_id == 0:
                assert class_name == "_background_"
            self.class_name_to_id[class_name] = class_id

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img_path = self.img_path[idx].strip()
        img = Image.open(img_path)
        label_path = self.img_path[idx].replace("jpg", "json").strip()
        label_file = labelme.LabelFile(filename=label_path)
        label, _ = labelme.utils.shapes_to_label(
            img_shape=np.asarray(img).shape[:-1],
            shapes=label_file.shapes,
            label_name_to_value=self.class_name_to_id)
        img = self.transform(img)
        label = label.astype(np.uint8)
        label = cv2.resize(label, tuple(img.shape[1:]), cv2.INTER_AREA)
        return img_path, img, label


if __name__ == "__main__":
    data_transforms = transforms.Compose([
        transforms.Resize((600, 600)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225])
        ])

    image_datasets = MyDataset(
        root="data/input/pipe_seg_v1_test",
        annot="train.txt",
        transform=data_transforms)

    for (img, label) in image_datasets:
        print(img.size(), label.size())
