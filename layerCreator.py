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
from .common import PredictionNeurons , NeuronBlock , BConvBlock, ResidualBlock,ScalePrediction,ConvBlock
from .config import *


class MultiLayerCreator(nn.Module):

  def __init__ (self,config,resulution,num_classes,in_channel=3,information=False):
    
    super(MultiLayerCreator,self).__init__()
    self.information=information
    self.resulotion = resulution
    self.num_classes = num_classes
    self.config = config
    self.in_channel = in_channel
    self.layers = self.layer_creator()
  

  def layer_creator(self):
    layers = nn.ModuleList()

    for conf in self.config:
      if isinstance(conf,list):
        out_c , kernel , sp  = conf 
        layers.append(

                ConvBlock(
                    self.in_channel,
                    out_c,
                    kernel_size=kernel,
                    padding=1 if kernel == 3 else 0,
                    stride=sp
                    ).to(device)

        )
        self.in_channel = out_c
      
      elif isinstance(conf,tuple):
        types , time = conf
        if types == 'B':  
          layers +=[
            ResidualBlock( self.in_channel,num_repeat=time)
          ]
          
        elif types == 'M':
          for i in range(time):
            layers.append(
              ResidualBlock( self.in_channel,time)
            )
        elif types == "S":

          layers += [
            ResidualBlock(self.in_channel,use_residual=False,num_repeat=1),
            ConvBlock(self.in_channel,self.in_channel//2,kernel_size=1),
            ScalePrediction(self.in_channel//2,num_classes=self.num_classes,stride=time)
          ]
          self.in_channel //=2


      elif isinstance(conf , str):

        if conf == "O":
          layers.append(
              NeuronBlock(self.in_channel*84.5, self.num_classes,act=False).to(device)
          )
        elif conf == "P":
          layers.append(
              PredictionNeurons(self.in_channel*3, self.num_classes,act=False).to(device)
          )
        elif conf == "S":

          layers += [
                ResidualBlock(self.in_channel,use_residual=False,num_repeat=1),
                ConvBlock(self.in_channel,self.in_channel//2,kernel_size=1),
                ScalePrediction(self.in_channel//2,num_classes=self.num_classes)
          ]
          self.in_channel //=2
          
        elif conf == "U":
          layers.append(nn.Upsample(scale_factor=1))
          self.in_channel*=9

    return layers

  def forward(self,x):


    output = []


    route_connections = []

    i=0
    for layer in self.layers:

      if isinstance(layer , ScalePrediction):


        if  self.information:
          print('[INFO] Running On ScalePrediction Layer')
          stp = time.time()


        output.append((layer.forward(x).to(device)))

        if  self.information:
          # print(f'[INFO] Output From Scale Prediction : {x.shape}')
          ttp = (time.time()-stp)
          print('[INFO] Finished On ScalePrediction Layer in {} Sec'.format(ttp) )

        continue
      if  self.information:
        print(f'[INFO] befor the predict : {x.shape}')
      x = layer.forward(x)
      if  self.information:
        print(f"[INFO] Layer is runing on {i} index")
        if isinstance(layer,nn.Upsample) != True:
          print(f"[INFO] Layer Information : in channels : {layer.c_in} ")
        else:
          print("[INFO] Upsample Layer Is Running ")
        print(f'[INFO] after the predict : {x.shape}\n-------------')

      i+=1
      if isinstance(layer,ResidualBlock) and layer.num_repeat==2:



        if  self.information:

          print('[INFO] Repeats ON this Layer : {}'.format(layer.num_repeat))
          strb = time.time()
          print('[INFO] Running On ResidualBlock Layer')


        route_connections.append(x)
        
        # if  self.information:
        #   routecopy = torch.stack(route_connections).to('cpu').detach().numpy()
        #   vrb = np.array(routecopy)
        #   print('[INFO] Shape  On route connections : {} '.format(vrb.shape))
      
        if  self.information:
          ttrb = (time.time()-strb)
          print('[INFO] Finished On ResidualBlock Layer in {} Sec'.format(ttrb))




      elif isinstance(layer,nn.Upsample):


        if  self.information:
          print('[INFO] Running On Upsample Layer')
          stu = time.time()


        x = torch.cat((x,route_connections[-1]),dim=1)
        route_connections.pop()


        if  self.information:
          ttu = (time.time()-stu)
          print('[INFO] Finished On Upsample Layer in {} Sec'.format(ttu))

    return output      
