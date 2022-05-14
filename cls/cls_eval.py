import os
import sys
import torch
from torch import nn
from torchvision import transforms
from sklearn.metrics import classification_report
import logging
import yaml
sys.path.insert(0, os.getcwd())

from cls.utils import AverageMeter
from cls.cls_accuracy import accuracy


def validate(dataloader_eval, model, criterion, device, cls_mapping,
             print_cls_report=False):

    model.eval()
    val_epoch_loss = AverageMeter()
    val_epoch_accuracy = AverageMeter()
    labels_all = []
    preds_all = []
    for inputs, labels in dataloader_eval:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            _, logits = model(inputs)
            if len(cls_mapping) > 2:
                _, preds = torch.max(logits, 1)
            else:
                preds = torch.sigmoid(logits)
                preds = torch.where(preds > 0.5, 1, 0)
            if len(cls_mapping) > 2:
                if isinstance(criterion, nn.CrossEntropyLoss):
                    val_batch_loss = criterion(logits, labels)
                elif isinstance(criterion, nn.BCEWithLogitsLoss):
                    labels_one_hot = torch.nn.functional.one_hot(
                        labels, num_classes=len(cls_mapping))
                    val_batch_loss = criterion(
                        logits, labels_one_hot.type(torch.float32))
            else:
                val_batch_loss = criterion(
                    logits, labels.unsqueeze(-1).type(torch.float32))

        labels_all.append(labels)
        preds_all.append(preds)

        val_epoch_loss.update(val_batch_loss, inputs.size(0))
        acc1 = accuracy(logits, labels.data)[0]
        val_epoch_accuracy.update(acc1.item(), inputs.size(0))

    labels_all = torch.concat(labels_all, 0).cpu().numpy()
    preds_all = torch.concat(preds_all, 0).cpu().numpy()
    if print_cls_report:
        cls_report = classification_report(
            y_true=labels_all, y_pred=preds_all,
            target_names=dict(
                sorted(cls_mapping.items(), key=lambda item: item[1])).keys(),
            digits=6)
        logging.info(f"\n{cls_report}")
    return val_epoch_loss.avg, val_epoch_accuracy.avg


if __name__ == '__main__':
    from cls import MyDataset, PipeNet, ResNet18

    # Set log level
    logging.basicConfig(
        level=logging.DEBUG,
        format="(%(asctime)s) | %(levelname)-8s | %(module)s: %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S')

    # Input
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 64
    MODEL_NAME = "src/pipe_cls/exp/model22"

    # Load config
    with open(f"{MODEL_NAME}.yml", "r") as stream:
        config = yaml.safe_load(stream)

    # create transform
    data_transforms = {
        "val": transforms.Compose([
            transforms.Resize(56),
            transforms.CenterCrop(56),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])])}

    # create dataset
    image_datasets = {"val": MyDataset(
        root=config['root'],
        annot=config['train_test_annot'].split(',')[1],
        cls_mapping=config["cls_mapping"],
        transform=data_transforms["val"])}
    dataloaders = {"val": torch.utils.data.DataLoader(
        image_datasets["val"], batch_size=BATCH_SIZE, shuffle=True,
        num_workers=os.cpu_count(), drop_last=False)}
    dataset_sizes = {"val": len(image_datasets["val"])}

    # create model
    if config["model"] == "ResNet18":
        model = ResNet18(num_class=len(config["cls_mapping"]))
    elif config["model"] == "PipeNet":
        model = PipeNet(num_class=len(config["cls_mapping"]))
    model.load_state_dict(torch.load(
        f"{MODEL_NAME}.pth", map_location=torch.device("cpu")))
    model = model.to(DEVICE)

    # create loss
    if len(config["cls_mapping"]) > 2:
        criterion = nn.CrossEntropyLoss(label_smoothing=0)
    else:
        criterion = nn.BCEWithLogitsLoss()

    # eval
    val_epoch_loss, val_epoch_accuracy = validate(
        dataloaders['val'], model, criterion, DEVICE,
        config["cls_mapping"], True)
