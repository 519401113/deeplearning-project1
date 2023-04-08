import os
import numpy as np
from BP import *


if __name__=='__main__':
    twolayer = MLP(784, 10, [256, 256], convolution_layer=None, lr=0.001, momentum=0.9, regularization=0)
    best_model = train(twolayer, loss, train_dataloader, epoch=100000)
    _ = Verify(twolayer, test_dataloader)


