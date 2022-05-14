import os
import sys
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torchvision import transforms
import copy
import logging
import pandas as pd
import yaml
sys.path.insert(0, os.getcwd())

from seg import *


# Set log level
logging.basicConfig(
    level=logging.DEBUG,
    format="(%(asctime)s) | %(levelname)-8s | %(module)s: %(message)s",
    datefmt='%Y-%m-%d %H:%M:%S')


def train_one_epoch(dataloader_train, model, criterion, 
                    optimizer_, device):

    model.train()
    train_epoch_loss = AverageMeter()
    train_epoch_accuracy = AverageMeter()
    for _, inputs, labels in dataloader_train:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer_.zero_grad()

        with torch.set_grad_enabled(True):
            logits = model(inputs)
            train_batch_loss = criterion(logits, labels)

            train_batch_loss.backward()
            optimizer_.step()

        train_epoch_loss.update(train_batch_loss, inputs.size(0))
        acc1 = accuracy(logits, labels.data)[0]
        train_epoch_accuracy.update(acc1.item(), inputs.size(0))
    return model, train_epoch_loss.avg, train_epoch_accuracy.avg


def train(dataloaders, model, criterion, optimizer_, scheduler_, 
          num_classes, num_epochs, device, model_name):

    logging.info("Training starts...")
    best_state_dict = copy.deepcopy(model.state_dict())
    best_accuracy = 0
    train_loss, train_accuracy, val_loss, val_accuracy, lr = [], [], [], [], []
    for epoch in range(num_epochs):

        # Train
        model, train_epoch_loss, train_epoch_accuracy =\
            train_one_epoch(
                dataloaders['train'], model,
                criterion, optimizer_, device)
        train_loss.append(train_epoch_loss.item())
        train_accuracy.append(train_epoch_accuracy)
        logging.info(
            f"Epoch {epoch:3d}/{num_epochs-1:3d} {'Train':5s}, "
            f"Loss: {train_epoch_loss:.4f}, "
            f"Acc: {train_epoch_accuracy:.4f}")

        # Eval
        val_epoch_loss, val_epoch_accuracy = validate(
            dataloaders['val'], model, criterion, device,
            num_classes)
        val_loss.append(val_epoch_loss.item())
        val_accuracy.append(val_epoch_accuracy)
        logging.info(
            f"Epoch {epoch:3d}/{num_epochs-1:3d} {'Val':5s}, "
            f"Loss: {val_epoch_loss:.4f}, "
            f"Acc: {val_epoch_accuracy:.4f}")

        lr.append(scheduler_.get_last_lr()[0])

        if val_epoch_accuracy > best_accuracy:
            best_accuracy = val_epoch_accuracy
            best_state_dict = copy.deepcopy(model.state_dict())

        scheduler_.step()

    logging.info('Best Val Acc: {:4f}'.format(best_accuracy))

    # Load best model
    model.load_state_dict(best_state_dict)

    # Classification report
    val_epoch_loss, val_epoch_accuracy = validate(
        dataloaders['val'], model, criterion, device, 
        num_classes, True, True)

    # Save best model
    torch.save(model.state_dict(), f"{model_name}.pth")

    pd.DataFrame({
        'Epochs': range(num_epochs), 'Learning Rate': lr,
        'Training Loss': train_loss, 'Training Accuracy': train_accuracy,
        'Validation Loss': val_loss, 'Validation Accuracy': val_accuracy
        }).to_csv(f"{model_name}.csv", index=False)
    logging.info("Training run successfully...")
    return None


if __name__ == '__main__':
    # Input
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    EPOCHS = 50
    BATCH_SIZE = 8
    LEARNING_RATE = 0.0001
    DATA_DIR = "data/input/pipe_seg_v1"
    ANNOT = ['train.txt', 'val.txt']
    MODEL = "SegNet"  # Check src/pipe_seg/seg_models for more models
    MODEL_NAME = "src/pipe_seg/exp/model1"
    CRITERION = cross_entropy # Check seg_losses for more losses

    # create transform
    data_transforms = {
        "train": transforms.Compose([
            # TODO: Random crop
            transforms.Resize((600, 600)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([
            transforms.Resize((600, 600)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])])}

    # create dataset
    image_datasets = {x: MyDataset(
        root=DATA_DIR,
        annot=annot,
        transform=data_transforms[x])
        for x, annot in zip(["train", "val"], ANNOT)}
    dataloaders = {x: torch.utils.data.DataLoader(
        image_datasets[x], batch_size=BATCH_SIZE, shuffle=True,
        num_workers=os.cpu_count(), drop_last=False) for x in ["train", "val"]}
    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}

    # create model
    num_classes = len(image_datasets["train"].class_name_to_id)
    if "binary_cross_entropy" in CRITERION.__name__:
        num_classes = 1
    if MODEL == "UNet":
        model = UNet(num_classes=num_classes, pretrained=True)
    elif MODEL == "UResNet":
        model = UResNet(num_classes=num_classes, pretrained=True)
    elif MODEL == "SegNet":
        model = SegNet(num_classes=num_classes, pretrained=True)
    elif MODEL == "SegResNet":
        model = SegResNet(num_classes=num_classes, pretrained=True)
    model = model.to(DEVICE)

    # Calculate model parameters
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    logging.info(f'Model Parameters (M): {params/1e6}')

    # create optimizer and lr scheduler
    # optimizer_ = optim.sgd(
    #     model.parameters(), lr=LEARNING_RATE,
    #     momentum=0.9, weight_decay=5e-4)
    optimizer_ = optim.Adam(
        model.parameters(), lr=LEARNING_RATE,
        weight_decay=5e-4)
    lr_scheduler_ = optim.lr_scheduler.StepLR(
        optimizer_, step_size=EPOCHS, gamma=0.1)
    # lr_scheduler_ = optim.lr.scheduler.MultiStepLR(
    #     optimizer_, milestones=[15, 80], gamma=0.1)
    # lr_scheduler_ = optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer_, T_max=EPOCHS*dataset_sizes['train']/BATCH_SIZE,
    #     verbose=False)
    # lr_scheduler_ = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #     optimizer_, T_0=20, T_mult=1, verbose=False)

    # Save training details
    config = dict(
        root=DATA_DIR,
        train_test_annot=','.join(ANNOT),
        model_name=f"{MODEL_NAME}.pth",
        model=MODEL,
        loss=CRITERION.__name__)
    with open(f"{MODEL_NAME}.yml", "w") as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    # Train
    model = train(
        dataloaders, model, CRITERION, optimizer_, lr_scheduler_,
        num_classes, EPOCHS, DEVICE, MODEL_NAME)
