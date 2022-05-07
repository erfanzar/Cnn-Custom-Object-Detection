import torch.nn as nn
import torch
from torch.nn import Conv2d, Linear, BatchNorm2d, SiLU, Dropout, ReLU, LeakyReLU
from torch.nn import Sequential

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class PredictionNeurons(nn.Module):
  def __init__(self,num_in,num_classes , act=True):
    super().__init__()

    self.act = act

    self.num_in = int(num_in)
    
    self.num_classes = num_classes

    self.dromp_input = 1

    self.LeakyReLU1 = LeakyReLU(0.02).to(device)

    self.LeakyReLU2 = LeakyReLU(0.02).to(device)

    self.LeakyReLU3 = LeakyReLU(0.02).to(device)

    self.T_num = 300
    
    self.fc0 = Linear(self.T_num,self.T_num//8).to(device)
    
    self.fc1 = Linear(self.T_num//8 , self.num_classes).to(device)
    
    self.ReLU = ReLU().to(device)

    self.softmax = nn.Softmax()

  def forward(self,x):
    x1 = x[0][self.num_classes+5]
    x2 = x[0][self.num_classes*2+5]
    print('x1 shape {} x2 shape {}'.format(x1.shape , x2.shape))
    x = torch.cat((x1,x2),1)
    x = x.view(1,-1)

    print(x.shape)

    return self.ReLU(self.fc1(self.LeakyReLU1(self.fc0(x))))

class NeuronBlock(nn.Module):
  def __init__(self,num_in,num_classes , act=True):
    super(NeuronBlock,self).__init__()

    self.act = act
    self.num_in = int(num_in)
    self.num_classes = num_classes

    self.T_num = 410
    
    self.dropout1 = Dropout(0.001).to(device)

    self.LeakyReLU1 = LeakyReLU(0.02).to(device)

    self.LeakyReLU2 = LeakyReLU(0.02).to(device)

    self.fc0 = Linear(self.T_num,self.T_num//9).to(device)

    self.ReLU = ReLU().to(device)

    self.init_Weights()

  
  def forward(self,x):

    x = x.view(-1,x.size)
    
    return self.ReLU(self.dropout1(self.LeakyReLU1(self.neuron1(x))))
        
  
  def init_Weights(self):

    if self.act == True:
      torch.nn.init.kaiming_normal(self.neuron1).to(device)
      torch.nn.init.kaiming_normal(self.neuron2).to(device)
      torch.nn.init.kaiming_normal(self.neuron3).to(device)


    
    

class BConvBlock(nn.Module):
  def __init__(self,c_in,c_out,**kwargs):
    
    super(BConvBlock,self).__init__()
    self.conv = None
    self.c_in = c_in
    self.c_out = c_out
    
  def forward(self,x):
    self.conv = Sequential(
        ConvBlock(self.c_in,self.c_out,kernel_size=6,stride=1),
        ConvBlock(self.c_out,self.c_out//2,kernel_size=6,stride=1),
    )
    return self.maxpool(self.conv(x))

class ConvBlock(nn.Module):
  def __init__(self,c_in,c_out,act=False,**kwargs):
    super(ConvBlock,self).__init__()
    self.act = act
    self.c_in = c_in
    self.c_out = c_out
    self.conv = Conv2d(self.c_in,self.c_out,bias=not act,**kwargs).to(device)
    self.batch = BatchNorm2d(self.c_out).to(device)
    self.SiLU = SiLU().to(device)

  def forward(self,x):
    if self.act==True:
      return self.SiLU(self.batch(self.conv(x)))
    else:
      return self.conv(x)


class ResidualBlock(nn.Module):
  def __init__(self,c_in,use_residual=True,num_repeat = 1,**kwargs):
    super().__init__()
    self.c_in = c_in
    self.num_repeat = num_repeat
    self.use_residual = use_residual
    self.layer = nn.ModuleList()
    for repeat in range(num_repeat):
      self.layer += [
          Sequential(
            ConvBlock(c_in,c_in//2,kernel_size=1),
            ConvBlock(c_in//2,c_in,kernel_size=3, padding=1)
          )
        ]
  def forward(self,x):
    for layer in self.layer:

      x = layer(x) + x if self.use_residual else layer(x)

    return x


class ScalePrediction(nn.Module):
  def __init__(self,c_in,num_classes,**kwargs):
    super().__init__()
    self.c_in = c_in
    self.num_classes = num_classes
    self.pred = Sequential(

        ConvBlock(c_in,c_in*2,kernel_size=3,padding=1,**kwargs),
        ConvBlock(2*c_in,(num_classes+5)*3,False,kernel_size=1)#why plus 5 ? cause we want to also predict the x y h w and probeblity of theres an object in frame or no [po,x,y,w,h]
    )
  def forward(self,x):

    """ 
      N , 3,13,13,num_classes +5
      N is for number of Batchs
      3 is for anchor boxes
      13 is the shape of image in width
      13 is the shape of image in height
      num_classes + 5 cause we want to have each class output + x,y,h,w,po

    """
    
    # return (
    #       self.pred(x)
    #       .reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3])
    #       .permute(0, 1, 3, 4, 2)
    #   )

    x = self.pred(x)
    print(f'[INFO] being Reshape : {x.shape}')
    x = x.reshape(x.shape[0],3,(self.num_classes +5),x.shape[2],x.shape[3])
    print(f'[INFO] after Reshape : {x.shape}')
    x = x.permute(0,1,3,4,2)
    print(f'[INFO] after permute : {x.shape}')
    return x

