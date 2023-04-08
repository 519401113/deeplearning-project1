import os
import numpy as np
from BP import *
from matplotlib import pyplot as plt

def visulization(model):
    for i in model.Layers:
        for j in i.w:
            h,w = j.shape
            pathsize = 16 if w%16==0 else 5
            rew = j.reshape(16, h//16, pathsize, w//pathsize).transpose(0,2,1,3)
            rew = rew.reshape(16*pathsize, -1)
            v_image = plt.imshow(rew, cmap='viridis')
            plt.colorbar(v_image, fraction=0.046, pad=0.04)
            plt.title('parameters for {}x{} matrix'.format(h,w))
            plt.show()



if __name__=='__main__':
    twolayer = MLP(784, 10, [256, 256], convolution_layer=None, lr=0.001, momentum=0.9, regularization=0)
    twolayer.load()
    _ = Verify(twolayer, test_dataloader)

    visulization(model=twolayer)
