import torch
import torch.nn as nn
import time
from common import PredictionNeurons, NeuronBlock, BConvBlock, ResidualBlock, ScalePrediction, ConvBlock
from config import config, config_v2, config_v3



class MultiLayerCreator(nn.Module):
      

  def __init__ (self,config,resulution,num_classes,in_channel=3,information=False):
    self.device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
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
                    ).to(self.device)

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
              NeuronBlock(self.in_channel*84.5, self.num_classes,act=False).to(self.device)
          )
        elif conf == "P":
          layers.append(
              PredictionNeurons(self.in_channel*3, self.num_classes,act=False).to(self.device)
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


        output.append((layer.forward(x).to(self.device)))

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

  
  

if __name__ == '__main__':
  
  device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
  
  model = MultiLayerCreator(config = config_v2,in_channel=3,num_classes=4,resulution=416,information=True)
  
  model.to(device)
  
  x = torch.randn((2,3,416,416)).to(device)
  
  out = model.forward(x)

  print(out[0].shape)

  print(out[1].shape)

  print(out[2].shape)

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
