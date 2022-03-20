
import torch.nn as nn

import torch

from layerCreator import layer_creator

from common import CNN , Residual , ScalePrediction

from autopad import *


class Detector(nn.Module):

    def __init__(self , numclass , in_c = 1):
        
        super().__init__()
        
        
        
        self.classes = numclass

        self.in_c = in_c

        self.fullylayer= layer_creator(3,2)

    def forward(self , x):

       

        out = []

        route_connections = []


        for layer in self.fullylayer:


            if isinstance(layer , ScalePrediction):

                out.append(layer(x))

                continue

            x = layer(x)

            if isinstance(layer , Residual) and layer.num_rep == 8 :

                route_connections.append(x)

            elif isinstance(layer , nn.Upsample):
                
                x = torch.cat([x,route_connections[-1]] , dim=1)

                route_connections.pop()





#### example

if __name__ == "__main__":
    model = Detector(2)
    print(model.fullylayer)


    ## for train 
    
    ##model.forward(input)

    o = open('Layers.txt' , 'a')

    with o as w :
        w.write(str(model.fullylayer))


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