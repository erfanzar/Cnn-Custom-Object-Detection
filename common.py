import torch.nn as nn

from autopad import *


class CNN(nn.Module):

    def __init__(self,c1,c2,act=True,**kwargs):
        
        super().__init__()

        self.conv = nn.Conv2d(c1,c2,bias= True if act == True else False , **kwargs)


        self.batch = nn.BatchNorm2d(c2)

        
        self.activation = nn.SiLU() if act == True else nn.Identity()


        self.act = act


    def forward(self , x):

        if self.act :
            
            return self.activation(self.batch(self.conv(x)))

        else :

            return self.conv(x)




class Residual(nn.Module):

    def __init__(self , c , use=True , num=1):

        super().__init__()

        self.layers = nn.ModuleList()

        for repeat in range(num) :

            self.layers += [
                nn.Sequential(
                    CNN(c , c * 2 , kernel_size=1),
                    CNN(c * 2 , c , kernel_size=3 , padding=1)
                )
            ] 


        self.num = num

        self.use = use


    def forward(self , x):

        for layer in self.layers :


            x = layer(x) + x if self.use == True else layer(x)


        return x





class ScalePrediction(nn.Module):
    
    def __init__(self ,c ,num):
       
        super().__init__()

        self.pred = nn.Sequential(
            CNN(c , c*2 , kernel_size=3 , padding=1),
            CNN(c*2 ,3 * (num+5), act=False , kernel_size=1)
        
        )  

        self.num = num

        self.c = c

    def forward(self , x):

        return (
            self.pred(x)
            .reshape(x.shape[0] , 3 , self.num + 5 , x.shape[2] , x.shape[3])
            .permute(0,1,3,4 , 2)
        )
