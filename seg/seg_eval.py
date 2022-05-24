import os
import sys
import torch
from torch import nn
from torchvision import transforms
from sklearn.metrics import classification_report
import logging
import yaml
import numpy as np
import cv2
sys.path.insert(0, os.getcwd())

from cls.utils import AverageMeter
from seg.seg_accuracy import accuracy


def validate(dataloader_eval, model, criterion, device,
             num_classes, print_cls_report=False, viz=False):

    model.eval()
    val_epoch_loss = AverageMeter()
    val_epoch_accuracy = AverageMeter()
    img_path_all = []
    labels_all = []
    preds_all = []
    
    # Generate class-specific color for mask
    if viz:
        color = np.zeros((1, 3), dtype=np.uint8)
        # color_tmp = np.random.randint(
        #     256, size=(max(1, num_classes - 1), 3), 
        #     dtype=np.uint8)
        color_tmp = np.array([[255, 255, 255]], dtype=np.uint8)
        color = np.concatenate([color, color_tmp], 0)

    for img_path, inputs, labels in dataloader_eval:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            logits = model(inputs)

            if num_classes == 1:
                # Here 1 means the output is (B, 1, H, W)
                preds = torch.sigmoid(logits.squeeze(1))
                preds = torch.where(preds > 0.5, 1, 0)
            else:
                _, preds = torch.max(logits, 1)               
            
            val_batch_loss = criterion(logits, labels)

        val_epoch_loss.update(val_batch_loss, inputs.size(0))
        acc1 = accuracy(logits, labels.data)[0]
        val_epoch_accuracy.update(acc1.item(), inputs.size(0))

        if print_cls_report or viz:
            img_path_all.extend(list(img_path))
            labels_all.append(labels)
            preds_all.append(preds)

    if print_cls_report or viz:
        labels_all = torch.concat(labels_all, 0).cpu().numpy()
        preds_all = torch.concat(preds_all, 0).cpu().numpy()

    if print_cls_report:
        cls_report = classification_report(
            y_true=labels_all.reshape(-1), y_pred=preds_all.reshape(-1),
            target_names=dataloader_eval.dataset.class_name_to_id,
            digits=6)
        logging.info(f"\n{cls_report}")
    
    if viz:
        for (img_path, labels, preds) in zip(img_path_all, labels_all, preds_all):
            # preds = labels

            H, W = preds.shape
            mask = np.ones((H, W, 3)) * np.expand_dims(preds, -1)

            for cls_ in np.unique(labels_all)[1:]:
                mask[(mask == cls_).all(-1)] = color[cls_]
            mask = mask.astype(np.uint8)

            img_path = os.path.join(
                "data/output", os.path.basename(img_path))
            if os.path.exists(img_path):
                os.remove(img_path)
            cv2.imwrite(img_path, mask)
            # cv2.imshow(img_path, mask)
            # cv2.waitKey()
            # cv2.destroyWindow(img_path)

    return val_epoch_loss.avg, val_epoch_accuracy.avg


if __name__ == '__main__':
    from seg import *

    # Set log level
    logging.basicConfig(
        level=logging.DEBUG,
        format="(%(asctime)s) | %(levelname)-8s | %(module)s: %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S')

    # Input
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 8
    MODEL_NAME = "src/car_seg/exp/model0"
    VIZ = True

    # Load config
    with open(f"{MODEL_NAME}.yml", "r") as stream:
        config = yaml.safe_load(stream)

    # create transform
    data_transforms = {
        "val": transforms.Compose([
            transforms.Resize((600, 600)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])])}

    # create dataset
    image_datasets = {"val": MyDataset(
        root=config['root'],
        annot=config['train_test_annot'].split(',')[1],
        transform=data_transforms["val"])}
    dataloaders = {"val": torch.utils.data.DataLoader(
        image_datasets["val"], batch_size=BATCH_SIZE, shuffle=True,
        num_workers=os.cpu_count(), drop_last=False)}
    dataset_sizes = {"val": len(image_datasets["val"])}

    # create model
    num_classes = len(image_datasets["val"].class_name_to_id)
    if "binary_cross_entropy" in config["loss"]:
        num_classes = 1
    if config["model"] == "UNet":
        model = UNet(num_classes=num_classes, pretrained=True)
    elif config["model"] == "UResNet":
        model = UResNet(num_classes=num_classes, pretrained=True)
    elif config["model"] == "SegNet":
        model = SegNet(num_classes=num_classes, pretrained=False)
    elif config["model"] == "PipeNet":
        model = SegResNet(num_class=num_classes, pretrained=False)
    model.load_state_dict(torch.load(
        f"{MODEL_NAME}.pth", map_location=torch.device("cpu")))
    model = model.to(DEVICE)

    criterion = eval(config["loss"])

    # eval
    val_epoch_loss, val_epoch_accuracy = validate(
        dataloaders['val'], model, criterion, DEVICE,
        num_classes, True, VIZ)
