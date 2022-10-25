import torch
from torch import nn
import onnx
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import tensorrt as trt
from torchvision import models
from torchvision.transforms import *
# import cv2
from collections import OrderedDict, namedtuple
from PIL import Image

from triton import TritonRemoteModel


class CustomCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(CustomCNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = self.layers(x)
        return x


def preprocess_image(img_path, img_size):
    # transformations for the input data
    transforms = Compose([
        Resize((img_size, img_size)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # read input image
    input_img = Image.open(img_path)
    # do transformations
    input_data = transforms(input_img)

    batch_data = torch.unsqueeze(input_data, 0)

    return batch_data


# Input
device = 'cuda'
num_classes = 1000  # number of classes for cnn
dummy_input = torch.rand(1, 3, 224, 224).to(device)  # dummy input image
onnx_file_path = "customcnn.onnx"  # file path for .onnx
img_path = 'data/street.jpg'  # image for inference using .engine model
dynamic = False  # dynamic for torch.onnx.export
half = False  # whether to convert to fp16 when exporting to .engine
img_size = 224  # image size for inference
use_trt = True  # Set True to use trt for inference else Triton


# Need not change
input_names = ["input1"]  # input layer name used by torch.onnx.export
output_names = ["output1"]    # output layer name used by torch.onnx.export
workspace = 4  # used during export to .engine
trt_file_path = onnx_file_path.replace('onnx', 'engine')  # file path for .engine

"""
Export model to onnx and trt
# https://github.com/ultralytics/yolov5/blob/master/export.py
"""
# Load pytorch model and export to onnx
model_pt = models.resnet18(pretrained=True)
fc_in = model_pt.fc.in_features
model_pt.fc = nn.Linear(fc_in, num_classes)
# model = CustomCNN(num_classes)
model_pt.eval().to(device)
torch.onnx.export(
    model_pt.cpu() if dynamic else model_pt,  # --dynamic only compatible with cpu
    dummy_input.cpu() if dynamic else dummy_input,  # --dynamic only compatible with cpu
    onnx_file_path,
    verbose=True,
    opset_version=13,
    training=torch.onnx.TrainingMode.EVAL,
    do_constant_folding=True,
    input_names=input_names,
    output_names=output_names,
    dynamic_axes={
        input_names[0]: {
            0: 'batch',
            2: 'height',
            3: 'width'},
        output_names[0]: {
            0: 'batch'}
    } if dynamic else None)

# Check if model is properly saved in onnx
model_onnx = onnx.load(onnx_file_path)  # load onnx model
onnx.checker.check_model(model_onnx)  # check onnx model

# Export onnx to trt
logger = trt.Logger(trt.Logger.INFO)
builder = trt.Builder(logger)
config = builder.create_builder_config()
config.max_workspace_size = workspace * 1 << 30
# config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace << 30)  # fix TRT 8.4 deprecation notice

flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
network = builder.create_network(flag)
parser = trt.OnnxParser(network, logger)
if not parser.parse_from_file(str(onnx_file_path)):
    raise RuntimeError(f'failed to load ONNX file: {onnx_file_path}')

inputs = [network.get_input(i) for i in range(network.num_inputs)]
outputs = [network.get_output(i) for i in range(network.num_outputs)]
# LOGGER.info(f'{prefix} Network Description:')
# for inp in inputs:
#     LOGGER.info(f'{prefix}\tinput "{inp.name}" with shape {inp.shape} and dtype {inp.dtype}')
# for out in outputs:
#     LOGGER.info(f'{prefix}\toutput "{out.name}" with shape {out.shape} and dtype {out.dtype}')

if dynamic:
    # if dummy_input.shape[0] <= 1:
    #     LOGGER.warning(f"{prefix}WARNING: --dynamic model requires maximum --batch-size argument")
    profile = builder.create_optimization_profile()
    for inp in inputs:
        profile.set_shape(inp.name, (
            1, *dummy_input.shape[1:]), (max(1, dummy_input.shape[0] // 2),
            *dummy_input.shape[1:]), dummy_input.shape)
    config.add_optimization_profile(profile)

# LOGGER.info(f'{prefix} building FP{16 if builder.platform_has_fast_fp16 and half else 32} engine in {f}')
if builder.platform_has_fast_fp16 and half:
    config.set_flag(trt.BuilderFlag.FP16)
with builder.build_engine(network, config) as engine, open(trt_file_path, 'wb') as t:
    t.write(engine.serialize())


"""
Inference using trt
# https://github.com/ultralytics/yolov5/blob/e4398cf179601d47207e9f526cf0760b82058930/models/common.py#L311
"""
if use_trt:
    # Load trt model
    Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
    logger = trt.Logger(trt.Logger.INFO)
    with open(trt_file_path, 'rb') as f, trt.Runtime(logger) as runtime:
        model = runtime.deserialize_cuda_engine(f.read())
    context = model.create_execution_context()
    bindings = OrderedDict()
    fp16 = False  # default updated below
    dynamic = False
    for index in range(model.num_bindings):
        name = model.get_binding_name(index)
        dtype = trt.nptype(model.get_binding_dtype(index))
        if model.binding_is_input(index):
            if -1 in tuple(model.get_binding_shape(index)):  # dynamic
                dynamic = True
                context.set_binding_shape(index, tuple(model.get_profile_shape(0, index)[2]))
            if dtype == np.float16:
                fp16 = True
        shape = tuple(context.get_binding_shape(index))
        im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
        bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
    binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
    batch_size = bindings[input_names[0]].shape[0]

    # Forward
    # from datetime import datetime
    # start_time = datetime.now()
    im = preprocess_image(img_path, img_size).to(device)
    # im = torch.cat([im, im], 0)
    if dynamic and im.shape != bindings[input_names[0]].shape:
        i_in, i_out = (model.get_binding_index(x) for x in (input_names[0], output_names[0]))  # check images, output
        context.set_binding_shape(i_in, im.shape)  # reshape if dynamic
        bindings[input_names[0]] = bindings[input_names[0]]._replace(shape=im.shape)
        # bindings[output_names[0]].data.resize_(tuple(context.get_binding_shape(i_out)))
        # change output shape according to batch size
        bindings[output_names[0]].data.resize_((im.shape[0], num_classes))  # this is only valid for cls model
    s = bindings[input_names[0]].shape
    assert im.shape == s, f"input size {im.shape} {'>' if dynamic else 'not equal to'} max model size {s}"
    binding_addrs[input_names[0]] = int(im.data_ptr())
    context.execute_v2(list(binding_addrs.values()))
    y = bindings[output_names[0]].data
    print(y)

    # check the result by model_pt
    model_pt.to(device)
    print(model_pt(im))

else:
    model = TritonRemoteModel(url=w)
    nhwc = model.runtime.startswith("tensorflow")
