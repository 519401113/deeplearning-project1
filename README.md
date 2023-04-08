# deeplearning-project1

0. data preperation
```
Go to the http://yann.lecun.com/exdb/mnist/ and download 
the train 60k and test 10k images and labels.
Decompress then, and put then in the './data'. After that you should have
|-data
|   t10k-images.idx3-ubyte
|   t10k-images.idx3-ubyte
|   train-images.idx3-ubyte
|   train-labels.idx1-ubyte
| BP.py
| ...
```
1. train
```
python train.py
```
2. load trained model and test
```
python load_and_test.py
## It will load 'best_model.npy', test the model, and visulize the reshaped parameters 
```
