import os
import sys
import torch 
import random
from pynvml import *
from fast_neural_style.config import *
from fast_neural_style.helpers import *
from fast_neural_style.neural_style.transformer_net import TransformerNet

def init_pytorch_model():
    """ style transfer global """
    global style_model, gpuHandle
    gpuHandle = []
    # initialize gpu handles
    nvmlInit()
    for gpuId in range(GPU_COUNT):
        gpuHandle.append(nvmlDeviceGetHandleByIndex(gpuId))
    # check CUDA availability
    print('CUDA availabiliy = {}'.format(torch.cuda.is_available()))
    # initialize network architecture
    style_model = TransformerNet()
