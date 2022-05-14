import torch
import torch.nn.functional as F


def binary_cross_entropy(preds, labels, class_weight=None):
    assert preds.dim() == 4 and preds.size(1) == 1, "Pred must have shape [N, 1, H, W]"
    assert labels.dim() == 3, "Label must have shape [N, H, W]"
    preds = preds.squeeze(1)
    loss = F.binary_cross_entropy_with_logits(
        preds, labels.float(), pos_weight=class_weight)

    return loss


def cross_entropy(preds, labels, class_weight=None, ignore_index=255):
    assert preds.dim() == 4 and preds.size(1) > 1, "Pred must have shape [N, C, H, W]"
    assert labels.dim() == 3, "Label must have shape [N, H, W]"
    loss = F.cross_entropy(
        preds, labels.long(), weight=class_weight, ignore_index=ignore_index)

    return loss


def __binary_dice_loss(preds, labels, valid_mask, smooth=1, exponent=2, **kwards):
    """
    To be called by dice loss only, not for external use
    """
    assert preds.shape[0] == labels.shape[0]
    preds = preds.reshape(preds.shape[0], -1)
    labels = labels.reshape(labels.shape[0], -1)
    valid_mask = valid_mask.reshape(valid_mask.shape[0], -1)

    num = torch.sum(torch.mul(preds, labels) * valid_mask, dim=1) * 2 + smooth
    den = torch.sum(preds.pow(exponent) + labels.pow(exponent), dim=1) + smooth

    return 1 - num / den


def dice_loss(preds,
              labels,
              smooth=1,
              exponent=2,
              class_weight=None,
              ignore_index=255):
    """
    Dice Loss is hard to converge. It is proposed in `V-Net: Fully Convolutional Neural Networks for
    Volumetric Medical Image Segmentation <https://arxiv.org/abs/1606.04797>`_.
    """
    assert preds.dim() == 4 and preds.size(1) > 1, "Pred must have shape [N, C, H, W]"
    assert labels.dim() == 3, "Label must have shape [N, H, W]"

    preds = F.softmax(preds, dim=1)
    num_classes = preds.shape[1]
    valid_mask = (labels != ignore_index).long()
    labels = F.one_hot(
        torch.clamp(labels.long(), 0, num_classes - 1),
        num_classes=num_classes)

    total_loss = 0
    for i in range(num_classes):
        dice_loss = __binary_dice_loss(
            preds[:, i],
            labels[..., i],
            valid_mask=valid_mask,
            smooth=smooth,
            exponent=exponent)
        if class_weight is not None:
            dice_loss *= class_weight[i]
        total_loss += dice_loss
    return (total_loss / num_classes).mean()


def focal_loss(preds, labels, gamma=2):
    assert preds.dim() == 4, "Pred must have shape [N, 1, H, W] or [N, C, H, W]"
    assert labels.dim() == 3, "Label must have shape [N, H, W]"

    if preds.size(1) == 1:
        preds = preds.squeeze()
        preds_sigmoid = preds.sigmoid()
        labels = labels.type_as(preds)
        one_minus_pt = (1 - preds_sigmoid) * labels + preds_sigmoid * (1 - labels)
        focal_weight = one_minus_pt.pow(gamma)

        loss = F.binary_cross_entropy_with_logits(
            preds, labels, reduction='none') * focal_weight

    elif preds.size(1) > 1:
        preds_softmax = F.softmax(preds, 1)
        preds_softmax_log = preds_softmax.log()
        focal_weight = (1 - preds_softmax).pow(gamma)
        labels_onehot = F.one_hot(
            labels.long(), num_classes=max(2, preds.size(1))).transpose(1, 3)
        loss = (focal_weight * preds_softmax_log * labels_onehot * -1)

        return loss.mean()


def lovasz_loss(preds, labels):
    raise NotImplementedError("To be implemented...")


if __name__ == "__main__":
    preds = torch.rand((4, 3, 7, 7))
    labels = torch.randint(0, 3, (4, 7, 7))
    cross_entropy(preds, labels)
