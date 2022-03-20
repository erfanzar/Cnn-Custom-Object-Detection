import torch.nn as nn

from config import config

from config import config

from common import CNN , Residual , ScalePrediction

from autopad import autopad

def layer_creator(in_c , classes):
    
        layers = nn.ModuleList()

        in_c
        
        for module in config:
        
            if isinstance(module , tuple):

                out_c , kernel_size , padding = module

                layers.append(
                    CNN(in_c , out_c , kernel_size=kernel_size , padding=padding)
                    )
                
                print(in_c)
                in_c = out_c

            elif isinstance(module , list):

                num_rep = module[1]

                layers.append(Residual(in_c,num=num_rep))
        
            elif isinstance(module , str):
                if module == "S":

                    layers +=[
                    
                        nn.Sequential(

                            Residual(in_c , use=False, num=1),
                        
                            CNN(in_c , in_c//2 ,kernel_size=1),
                        
                            ScalePrediction(in_c//2 , num=classes)

                        )
                    ]

                    in_c = in_c // 2


                elif module == "U":

                    layers.append(nn.Upsample(scale_factor=2))

                    in_c = in_c *3

        return layers

