import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from .sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from .aspp import build_aspp
from .decoder import build_decoder
from .backbone import build_backbone


class DeepLabv3Plus(nn.Module):
    """
    https://paperswithcode.com/paper/encoder-decoder-with-atrous-separable#code
    https://github.com/jfzhang95/pytorch-deeplab-xception
    """
    def __init__(
        self, pretrained=True, pretrained_backbone='resnet101',
        output_stride=16, num_classes=21, sync_bn=True, freeze_bn=False, **kwargs):
        super(DeepLabv3Plus, self).__init__()
        if pretrained_backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(pretrained_backbone, output_stride, BatchNorm, pretrained)
        self.aspp = build_aspp(pretrained_backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, pretrained_backbone, BatchNorm)

        self.freeze_bn = freeze_bn

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p


def deeplabv3plus(pretrained, num_classes, pretrained_backbone, **kwargs):
    model = DeepLabv3Plus(
        pretrained=pretrained, num_classes=num_classes,
        pretrained_backbone=pretrained_backbone, *kwargs)
    
    # Load pretrained weight
    if pretrained_backbone == "resnet101":
        pretrained_state_dict_path = "src/pipe_seg/seg_models/deeplabv3plus/deeplab-resnet.pth.tar"
    elif pretrained_backbone == "mobilenet":
        pretrained_state_dict_path = "src/pipe_seg/seg_models/deeplabv3plus/deeplab-mobilenet.pth.tar"
    elif pretrained_backbone == "drn":
        pretrained_state_dict_path = "src/pipe_seg/seg_models/deeplabv3plus/deeplab-drn.pth.tar"
    if pretrained and os.path.exists(pretrained_state_dict_path):
        # Load pretrained state dict
        pretrained_state_dict = torch.load(pretrained_state_dict_path, map_location="cpu")['state_dict']
        
        # Remove last layer from pretrained state dict
        pretrained_state_dict.pop("decoder.last_conv.8.weight")
        pretrained_state_dict.pop("decoder.last_conv.8.bias")

        # Load state dict
        model_state_dict = model.state_dict()
        model_state_dict.update(pretrained_state_dict)
        model.load_state_dict(model_state_dict)
    
    return model


if __name__ == "__main__":
    model = DeepLabv3Plus(pretrained_backbone='resnet101', output_stride=16)
    model.eval()
    input = torch.rand(1, 3, 1024, 1024)
    output = model(input)
    print(output.size())
