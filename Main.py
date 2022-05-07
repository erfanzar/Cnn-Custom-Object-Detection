from torch.nn.modules.loss import L1Loss
import torch 
import torch.nn as nn
import torch.optim as optim 
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms
import pandas as pd
from torchsummary import summary
from torch.utils.data import Dataset , DataLoader
from torch.nn import Conv2d , Linear , Dropout2d , BatchNorm2d , MaxPool2d , L1Loss , MSELoss , SiLU , Dropout , init , ReLU , CrossEntropyLoss ,Softmax , Upsample , LeakyReLU
import time
from torch.nn import Sequential



# MIT License

# Copyright (c) 2022 Erano-

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
