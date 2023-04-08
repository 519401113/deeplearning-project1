import numpy as np
from Class import *
import random
seed = 1000
np.random.seed(seed)
random.seed(seed)
##########################################################################
# 各个参数对应实验结果
##########################################################################
# cnn = Convolution(cout=32, pool_size=3)
# a = MLP(256,10,[32,32,64,64,32,16], convolution_layer=cnn)
## best acc: 0.979   ----  best iteration: 44999  ---- seed=1000
## best acc: 0.978   ----  best iteration: 32499  ---- seed=950
## best acc: 0.976   ----  best iteration: 32499  ---- seed=950 ---- bias=True
## best acc: 0.974   ----  best iteration: 29999  ---- seed=950 ---- pool_size=2
## best acc: 0.979   ----  best iteration: 47499  ---- seed=950 ---- reg=0.01
## best acc: 0.981   ----  best iteration: 39999  ---- seed=1000 ---- cout=64

# model_1 = MLP(256,10,[20,20,10,10,20,20]) # acc = 0.918

##########################################################################
# 激活函数： y=x ， Leaky Relu
##########################################################################
## fx = x
class Identity(Activation_Function):
    def get_gradient(self, y):
        return np.ones(shape=y.shape)
    def h(self, y):
        return y

## Leaky Relu : a=0.01
class ReLu(Activation_Function):
    def get_gradient(self,y):
        ## 计算 h'(y)
        def fun(x):
            if x>=0:
                return 1
            else:
                return 0.01
        vec_fun = np.vectorize(fun)
        return vec_fun(y)

    def h(self,y):
        ## 计算 h(y)

        def fun_out(x):
            if x>=0:
                return x
            else:
                return 0.01*x
        vec_fun = np.vectorize(fun_out)
        return vec_fun(y)


##########################################################################
## 池化层：maxpooling
##########################################################################
class MaxPool():
    def __init__(self, size=2):
        self.size = size

    def set_size(self, size=3):
        self.size = size

    def get_out(self, input):
        cout, batch_size, a, b = input.shape
        x, y = int(a/self.size), int(b/self.size)
        out = np.zeros(shape=[cout, batch_size, x, y])
        index_1 = out+0
        index_2 = out+0
        for i in range(cout):
            for j in range(batch_size):
                input_slice = input[i, j]
                out_slice = out[i, j]
                index_1_slice = index_1[i, j]
                index_2_slice = index_2[i, j]
                for k in range(x):
                    for l in range(y):
                        out_slice[k, l] = np.max(input_slice[k*self.size: (k+1)*self.size, l*self.size: (l+1)*self.size])
                        shift = np.argmax(input_slice[k*self.size: (k+1)*self.size, l*self.size: (l+1)*self.size])
                        index_1_slice[k, l] = k*self.size + int(shift/self.size)
                        index_2_slice[k, l] = l*self.size + shift%self.size
        self.out = out
        self.index_1 = index_1
        self.index_2 = index_2
        return out

    def get_gradient(self, grad):
        cout, batch_size, x, y = grad.shape
        a, b = x*self.size, y*self.size
        res = np.zeros(shape=[cout, batch_size, a, b])
        for i in range(cout):
            for j in range(batch_size):
                res_slice = res[i, j]
                grad_slice = grad[i, j]
                index_1_slice = self.index_1[i, j].astype(np.int)
                index_2_slice = self.index_2[i, j].astype(np.int)

                res_slice[index_1_slice, index_2_slice] = grad_slice

        return res


##########################################################################
## 卷积层和全连接层实现
##########################################################################
## 卷积层
class Convolution(Layer):
    def __init__(self, img_size=[16, 16], cout=3, kernel_size=5, pool_size=3, activation=ReLu(), pooling=MaxPool(),
                 dropout=0):
        self.size = img_size
        self.cout = cout
        self.activation = activation
        self.pooling = pooling
        self.pooling.set_size(pool_size)
        self.dropout = dropout
        self.kernel_size = kernel_size
        self.r = int(self.kernel_size / 2)  # radias

        self.outlen = int((img_size[0]-2*self.r)/pool_size * (img_size[1]-2*self.r)/pool_size * cout)

        delta = 1

        self.kernel = delta * np.random.normal(size=[cout, kernel_size, kernel_size], scale=(2/(0.01**2+1)/(cout*kernel_size*kernel_size))**0.5)
        self.b = np.zeros(shape=[cout])

        self.last_kernel = self.kernel + 0
        self.last_b = self.b + 0

        self.out = [0, 0, 0]  # conv_out, h(x)_out, pool_out
        self.grad_kernel = np.zeros(shape=self.kernel.shape)
        self.grad_out = [0, 0, 0]  # input, conv_out, h(x)_out
        self.grad_b = np.zeros(shape=[cout])

    def get_output(self, input):
        batch_size = input.shape[0]
        input = input.reshape([batch_size, self.size[0], self.size[1]])
        self.in_size = input.shape
        r = self.r
        transf_in = np.zeros(shape=[self.kernel_size, self.kernel_size, batch_size, self.size[0] - 2 * r, self.size[1] - 2 * r])
        a = self.size[0] - 2 * r
        b = self.size[1] - 2 * r
        ## 用爱因斯坦求和以加快卷积操作
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                transf_in[i, j] = input[:, i:i+a, j:j+b]
        self.out[0] = np.einsum('ijkln,hij->hkln', transf_in, self.kernel)+self.b.reshape([self.cout,1,1,1])
        self.out[1] = self.activation.h(self.out[0])
        self.out[2] = self.pooling.get_out(self.out[1])
        res = self.out[2].transpose((1,0,2,3))
        self.outshape = res.shape
        res = res.reshape([batch_size, self.outlen])
        self.output = res

        return res

    def get_gradient(self, input, out_grad):
        input = input.reshape(self.in_size)

        out_grad = out_grad.reshape(self.outshape)
        out_grad = out_grad.transpose((1,0,2,3))

        self.grad_out[2] = self.pooling.get_gradient(out_grad)
        self.grad_out[1] = self.activation.get_gradient(self.out[1])*self.grad_out[2]

        cout, batch_size, a, b = self.grad_out[1].shape
        for i in range(cout):
            grad_slice = self.grad_out[1][i]
            input_slice = input
            self.grad_b[i] = np.sum(grad_slice)
            for j in range(self.kernel_size):
                for k in range(self.kernel_size):
                    self.grad_kernel[i, j, k] = np.sum(grad_slice * input_slice[:, j:j + a, k:k + b])
        return 0

    def update(self, lr=0.001, momentum=0, regularization=0):
        lr = lr

        delta_kernel = self.kernel - self.last_kernel
        delta_b = self.b - self.last_b
        self.kernel = (1 - lr * regularization) * self.kernel - lr * self.grad_kernel + momentum * delta_kernel
        self.b = self.b - lr * self.grad_b + momentum * delta_b

        self.last_kernel = self.kernel
        self.last_b = self.b

    def save(self):
        return {'kernel': self.kernel, 'b': self.b}
    def load(self, param):
        self.kernel = param['kernel']
        self.b = param['b']

    def change_weight_to_fix_dropout(self):
        pass
    def change_weight_to_train(self):
        pass

## 全连接层
class LinearLayer(Layer):
    # 由于有batch的存在，input的维度是batchsize*inlen，w的维度应为inlen*outlen
    def __init__(self, inputlen=256, hidden=[10], activation=ReLu(), dropout=0, monment=0, bias=False):
        # h(x*w+b) x默认为列向量，导数也为列向量
        self.hidden = hidden
        self.dropout = dropout
        self.moment = monment
        self.activation = activation
        self.bias = bias

        self.l = len(hidden)
        self.w = []  # 各层权重
        self.b = []

        self.grad_w = [0 for _ in range(self.l)] # 各层w的导数
        self.grad_x = [0 for _ in range(self.l)] # 各层输出的导数
        self.grad_b = [0 for _ in range(self.l)] # 各层偏移的导数

        self.last_w = [] # t-1时刻w
        # self.b = np.random.normal(size=[len(hidden)]) # 各层偏移量

        self.last_b = [] # t-1时刻b
        self.out = [0 for _ in range(len(hidden))] #各层输出

        delta = 0.1

        self.bias_mat = []
        m = np.ones(shape=[inputlen,1])
        m[0,0] = 0
        self.bias_mat.append(m)

        self.w.append(np.random.normal(size=[inputlen, hidden[0]], scale=(2/(inputlen*(1+0.01**2)))**0.5))
        self.last_w.append(self.w[0]+0)
        self.b.append(delta*np.zeros(shape=[hidden[0]]))
        self.last_b.append(self.b[0]+0)
        for i in range(len(hidden)-1):
            m = np.ones(shape=[hidden[i],1])
            m[0,0] = 0
            self.bias_mat.append(m)

            self.w.append(np.random.normal(size=[hidden[i], hidden[i+1]], scale=(2/(hidden[i]*(1+0.01**2)))**0.5))
            self.last_w.append(self.w[-1]+0)
            self.b.append(delta*np.zeros(shape=[hidden[1+i]]))
            self.last_b.append(self.b[-1]+0)

        self.ifdrop = True
        if (dropout > 0):
            self.create_dropout()

    def get_output(self, input):
        if self.dropout > 0 and self.ifdrop:
            # ifdrop指是在预测还是在训练

            self.create_dropout() ## 随机得到dropout的神经元向量
            self.out[0] = self.activation.h(np.dot(input*self.dropout_vec[0], self.w[0]) + self.b[0])
            for i in range(self.l - 1):
                self.out[i] = self.out[i]*self.dropout_vec[i+1]
                yi = np.dot(self.out[i], self.w[i + 1]) + self.b[i + 1]
                self.out[i + 1] = self.activation.h(yi)
            self.output = self.out[self.l - 1]
            return self.output
        else:
            # 无dropout
            self.out[0] = self.activation.h(np.dot(input, self.w[0]) + self.b[0])
            for i in range(self.l - 1):
                yi = np.dot(self.out[i], self.w[i + 1]) + self.b[i + 1]
                self.out[i + 1] = self.activation.h(yi)
            self.output = self.out[self.l - 1]
            return self.output



    def get_gradient(self, input, out_grad):
        if self.dropout > 0:
            grad = out_grad
            # 更新后l-1层
            for i in range(self.l - 1):
                j = self.l - i - 1
                grad = grad * self.activation.get_gradient(self.out[j])
                self.grad_w[j] = np.dot(self.out[j - 1].T, grad)
                shape = self.out[j - 1].shape
                # self.grad_b[j] = np.dot(np.ones(shape).T, grad)
                self.grad_b[j] = np.dot(grad.T, np.ones(shape=[shape[0], 1])).T
                self.grad_x[j] = np.dot(grad, self.w[j].T)*self.dropout_vec[j]
                grad = self.grad_x[j]
            # 更新第一层
            grad = grad * self.activation.get_gradient(self.out[0])
            self.grad_w[0] = np.dot((input*self.dropout_vec[0]).T, grad)
            shape = self.out[0].shape
            self.grad_b[0] = np.dot(grad.T, np.ones(shape=[shape[0], 1])).T
            self.grad_x[0] = np.dot(grad, self.w[0].T)*self.dropout_vec[0]
            return self.grad_w
        else:
            # 无dropout模式
            grad = out_grad
            # 更新后l-1层
            for i in range(self.l - 1):
                j = self.l - i - 1
                grad = grad * self.activation.get_gradient(self.out[j])
                self.grad_w[j] = np.dot(self.out[j - 1].T, grad)
                shape = self.out[j - 1].shape
                # self.grad_b[j] = np.dot(np.ones(shape).T, grad)
                self.grad_b[j] = np.dot(grad.T, np.ones(shape=[shape[0], 1])).T
                self.grad_x[j] = np.dot(grad, self.w[j].T)
                grad = self.grad_x[j]
            # 更新第一层
            grad = grad * self.activation.get_gradient(self.out[0])
            self.grad_w[0] = np.dot(input.T, grad)
            shape = self.out[0].shape
            self.grad_b[0] = np.dot(grad.T, np.ones(shape=[shape[0], 1])).T
            self.grad_x[0] = np.dot(grad, self.w[0].T)

            return self.grad_w



    def get_input_gradient(self):
        return self.grad_x[0]

    def update(self, lr=0.001, momentum=0, regularization=0):
        for i in range(len(self.w)):
            delta_w = self.w[i]-self.last_w[i]
            delta_b = self.b[i]-self.last_b[i]
            if self.bias is False:
                self.w[i] = self.w[i] - lr * self.grad_w[i] + momentum * delta_w - lr * regularization * self.w[i]
                self.b[i] = self.b[i] - lr * self.grad_b[i] + momentum * delta_b
            else:
                self.b[i] = self.b[i] +\
                            (- lr * self.grad_b[i] + momentum * delta_b)*1
                self.w[i] = self.w[i] +\
                            (- lr * self.grad_w[i] + momentum * delta_w - lr * regularization * self.w[i])*self.bias_mat[i]

            self.last_w[i] = self.w[i]
            self.last_b[i] = self.b[i]
    def save(self):
        return {'w':self.w,'b':self.b}

    def load(self, param):
        self.w = param['w']
        self.b = param['b']

    def create_dropout(self):
        ## 随机每层dropout的神经元向量
        dropout = 1-self.dropout
        inputlen,_ = self.w[0].shape
        self.dropout_vec = []
        self.dropout_vec.append(np.random.binomial(1,dropout,size=[inputlen,]))
        for i in range(self.l-1):
            length,_ = self.w[i+1].shape
            self.dropout_vec.append(np.random.binomial(1,dropout,size=[length,]))

    def change_weight_to_fix_dropout(self):
        ## 训练完后改变权重以适应dropout模型的预测
        p = 1-self.dropout
        for i in range(self.l):
            self.w[i] = self.w[i]*p
            self.b[i] = self.b[i]*p
        self.ifdrop = False

    def change_weight_to_train(self):
        ## 改回参数为训练模式
        p = 1-self.dropout
        for i in range(self.l):
            self.w[i] = self.w[i]/p
            self.b[i] = self.b[i]/p
        self.ifdrop = True


##########################################################################
## Loss： MSE和交叉熵实现
##########################################################################
class MSE(Loss):
    def get_gradient(self):
        y = self.y
        input = self.input
        # give the label y and compute the gradient of the last layer
        batch_size, n = input.shape
        self.g = input-y
        self.g = self.g/batch_size
        return self.g


    def get_loss_value(self, y, input):
        self.y = y
        self.input = input
        # give the input and label to calculate the loss value
        batch_size, n = input.shape # n是label种类个数
        delta = input-y
        return 0.5*np.sum(delta**2)/batch_size

class Cross_Entropy(Loss):
    # 默认y为one-hot形式
    def get_gradient(self):
        # give the label y and compute the gradient of the last layer
        y = self.y
        input = self.input
        p = (y*self.p).sum(axis=1, keepdims=True)
        batch_size, _ = y.shape
        self.grad = -(y-self.p)/batch_size

        return self.grad

    def get_loss_value(self, y, input):
        # give the input and label to calculate the loss value
        self.y = y
        self.input = input
        batch_size, _ = y.shape
        e_ip = np.exp(input)
        value = e_ip/e_ip.sum(axis=1, keepdims=True)
        self.p = value
        value = np.log(value)
        loss = np.sum(value*y)/batch_size
        self.loss = -loss

        return self.loss


##########################################################################
## 模型定义
##########################################################################
class MLP(Model):
    # 模型：全连接神经网络，可以选择第一层是否加入卷积层
    # 可以选择是否dropout， 是否将每一层的某一个神经元置为常数（参数bias=True）


    def __init__(self, inputlen, outputlen, hidden, act_fun=ReLu(), dropout=0, bias=False, convolution_layer=None,
                 lr=0.001, momentum=0.9, regularization=0.00):
        if convolution_layer is not None:
            inputlen = convolution_layer.outlen
        # 共有3层，依次是卷积层，全连接隐层，输出层，输出层默认没有激活函数
        self.hidden_layer = LinearLayer(inputlen, hidden, activation=act_fun, dropout=dropout, bias=bias)
        self.output_layer = LinearLayer(hidden[-1], [outputlen], activation=Identity(), dropout=dropout, bias=bias)
        self.cnn = convolution_layer

        self.lr = lr
        self.momentum = momentum
        self.regularization=regularization


        if convolution_layer is not None:
            self.Layers = [convolution_layer, self.hidden_layer, self.output_layer]
        else:
            self.Layers = [self.hidden_layer, self.output_layer]

    def save(self, path='./model.npy'):

        param = [l.save() for l in self.Layers]
        opt_param = [self.lr, self.momentum, self.regularization]
        np.save(path, {'param': param, 'opt_param': opt_param})

    def load(self, path='./model.npy'):
        pkl = np.load(path, allow_pickle=True)[()]
        model_param = pkl['param']
        opt_param = pkl['opt_param']
        for i, param in enumerate(model_param):
            self.Layers[i].load(param)
        self.lr, self.momentum, self.regularization = opt_param

    def forward(self, input):
        ## 前馈，并计算得到每一层的输出
        self.input = input
        in_data = input
        for layer in self.Layers:
            out_data = layer.get_output(in_data)
            in_data = out_data
        return out_data

    def backprop(self, grad_of_loss):
        ## 得到传入的梯度计算每一层的梯度
        l = len(self.Layers)
        grad_out = grad_of_loss
        for i in range(l-1):
            layer = self.Layers[l-i-1]
            before_layer = self.Layers[l-i-2]
            _ = layer.get_gradient(before_layer.output ,grad_out)
            grad_out = layer.get_input_gradient()
        first_layer = self.Layers[0]
        _ = first_layer.get_gradient(self.input, grad_out)

    def update(self, lr=None, momentum=0, regularization=0):
        if lr is None:
            for layer in self.Layers:
                layer.update(self.lr, self.momentum, self.regularization)
        else:
            ## 更新每一层的参数
            for layer in self.Layers:
                layer.update(lr, momentum, regularization)


    def update_only_last_layer(self, lr=0.001, momentum=0, regularization=0):
        ## 只更新最后一层的参数
        self.Layers[-1].update(lr, momentum, regularization)

    def train_over(self):
        for i in self.Layers:
            i.change_weight_to_fix_dropout()

    def train_begin(self):
        for i in self.Layers:
            i.change_weight_to_train()

    def update_lr(self, rate):
        self.lr = self.lr*rate



##########################################################################
## Dataloader
##########################################################################
class Dataloader:
    ## 装载数据，默认shuffle，batchsize在最初指定
    def __init__(self, data, batchsize):
        self.batchsize = batchsize
        self.img = data[0]
        self.label = data[1]
        self.index = 0
        a, b = self.img.shape
        self.size = a
        ## 对数据进行标准化
        mu = np.mean(self.img, axis=1, keepdims=True)
        sigma = np.std(self.img, axis=1, keepdims=True)
        self.img = (self.img-mu)/sigma
        ## 数据随机排序
        shuffle = np.arange(self.img.shape[0])
        np.random.shuffle(shuffle)
        self.img = self.img[shuffle]
        self.label = self.label[shuffle]


    def get_data(self):
        ## 获取一个batch的数据
        list_all = 0
        if self.index + self.batchsize > self.size:
            end = self.size
            list_all = 1
        else:
            end = self.index+self.batchsize
        data = self.img[self.index:end, :]
        label = self.label[self.index:end, :]
        self.index = end % self.size
        if list_all == 1:
            shuffle = np.arange(self.img.shape[0])
            np.random.shuffle(shuffle)
            self.img = self.img[shuffle]
            self.label = self.label[shuffle]

        return data, label



##########################################################################
## 数据曾广（拉伸，旋转，平移），标签模糊
## 训练集，测试集，验证集Dataloader定义
##########################################################################
import scipy.io as scio
import cv2 as cv
import os
# datafile = "basisData.mat"
# digits = "digits.mat"
# matrix = scio.loadmat(digits)

def load_data(mode='train'):
    labels_path = os.path.join('data', '%s-labels.idx1-ubyte' % mode)
    images_path = os.path.join('data', '%s-images.idx3-ubyte' % mode)
    with open(labels_path, 'rb') as lbpath:
        y = np.fromfile(file=lbpath, dtype=np.uint8)[8:]+1
    with open(images_path, 'rb') as imgpath:
        X = np.fromfile(file=imgpath, dtype=np.uint8)[16:].reshape(len(y), 784)[:,:]
    return X, y


##  标签模糊化
def vague(label, a=0.01):
    res = label + a
    res = res/(1+10*a)
    return res

## 图片拉伸
def stretch(data, label, size=28):

    batch_size = data.shape[0]
    new_label = label + 0
    new_data = np.zeros(shape=[batch_size, size**2])
    for i in range(batch_size):
        slice = data[i]
        img = slice.reshape([size, size])
        percent_l = random.uniform(0.7, 1)
        percent_w = random.uniform(0.7, 1)
        n_l = int(percent_l*size)
        n_w = int(percent_w*size)
        new_img = cv.resize(img, (n_l, n_w))
        delta_l = i % (size-n_l)
        delta_w = i % (size-n_w)
        ss = np.zeros(shape=[size, size])
        ss[delta_w:delta_w+n_w, delta_l:delta_l+n_l] = new_img
        s = ss.flatten()
        new_data[i] = s

    return new_data, new_label
## 图片旋转
def rotation(data, size=28):
    batch_size = data.shape[0]
    res = data.reshape([batch_size, size, size])
    res = res.transpose([0,2,1])
    res = res.reshape([batch_size, size**2])
    return res

def one_hot(x):
    ohx = np.zeros((len(x), 10))
    ohx[range(len(x)), x] = 1
    return ohx

X,y = load_data(mode='train')
Xtest,ytest = load_data(mode='t10k')
Xvalid, yvalid = load_data(mode='t10k')

# y = matrix['y']
# y = y.reshape((5000,))


y = y-1
y = one_hot(y)

# y = vague(y,a=0.000)

# X = matrix['X']


## 数据拓宽
##########
s_X, s_y = stretch(X, y)
X = np.r_[X, s_X]
y = np.r_[y, s_y]
##########
# X_rot = rotation(X)
# y_rot = y + 0
# X_train = np.r_[X, X_rot]
# y_train = np.r_[y, y_rot]


train_data = [X, y]


# Xvalid = matrix['Xvalid']
# yvalid = matrix['yvalid']
# yvalid = yvalid.reshape((5000,))
yvalid = yvalid-1
yvalid = one_hot(yvalid)
valid_dataloader = Dataloader([Xvalid, yvalid], 1)

# Xtest = matrix['Xtest']
# ytest = matrix['ytest']
# ytest = ytest.reshape((1000,))
ytest = ytest-1
ytest = one_hot(ytest)
test_dataloader = Dataloader([Xtest,ytest], 1)



##########################################################################
## 模型实例化
##########################################################################
cnn = Convolution(cout=64, pool_size=3)
a = MLP(784,10,[32,32,64,64,32,16], convolution_layer=cnn)
loss = Cross_Entropy()
# a = MLP(256,10,[10], convolution_layer=None)
# loss = MSE()
train_dataloader = Dataloader(train_data, 8)




##########################################################################
## 训练和预测函数
##########################################################################

# 测试模型在数据集上的预测误差
def Verify(model=a, dataloader=valid_dataloader, size=10000):
    correct = 0
    model.train_over()
    for i in range(size):
        input, label = dataloader.get_data()
        # 前向算法
        model_out = model.forward(input)
        model_out = model_out.reshape((10,))
        label = label.reshape(10,)
        model_out = model_out.tolist()
        label = label.tolist()
        max_index = model_out.index(max(model_out))
        if label[max_index] == 1:
            correct += 1
    print('Accuracy:',correct/size)
    model.train_begin()
    return correct/size

# 模型训练函数
import copy
from matplotlib import pyplot as plt
def train(model, loss_function, dataloader, epoch=100000, lr=0.001, momentum=0.0, regularization=0.0):
    train_loss = 0
    train_batch = 100
    best_error = 0
    best_model = copy.deepcopy(model)
    best_itr = 0
    fine_tuning = 1000000

    loss_list = []
    loss_x = []
    acc_list = []
    acc_x = []

    for i in range(epoch):
        input, label = dataloader.get_data()
        # 前向算法
        model_out = model.forward(input)
        loss_value = loss_function.get_loss_value(label, model_out)
        # 反向传播并训练
        grad_of_loss = loss_function.get_gradient()
        model.backprop(grad_of_loss)
        model.update()

        # model.update(lr=lr, momentum=momentum, regularization=regularization)


        train_loss += loss_value
        if (i+1) % train_batch == 0:

            print((i+1),':', train_loss/train_batch)

            loss_list.append(train_loss/train_batch)
            loss_x.append(i+1)

            train_loss = 0


        if (i+1) % 1000 == 0:
            print("vaild:")
            vaild = Verify(model=model, size=1000)
            print("test:")
            test_error = Verify(model=model, dataloader=test_dataloader, size=10000)
            if test_error > best_error:
                best_error = test_error
                best_model = copy.deepcopy(model)
                model.save()
                best_itr = i
            print("best acc:", best_error,"  ----  best iteration:", best_itr)
            acc_list.append(test_error)
            acc_x.append(i+1)

        ## 学习率变动
        if (i+1) % 5000 == 0:
            # lr = lr*0.9
            model.update_lr(rate=0.9)

        if (i+1)% 20000 == 0:
            plt.plot(loss_x,loss_list)
            plt.xlabel('iter')
            plt.ylabel('loss')
            plt.show()

            plt.plot(acc_x,acc_list)
            plt.xlabel('iter')
            plt.ylabel('accuracy')
            plt.show()


    return best_model



##########################################################################
## 实验
##########################################################################
print('Loading complete')

if __name__=='__main__':
    twolayer = MLP(784, 10, [256, 256], convolution_layer=None, lr=0.001, momentum=0.9, regularization=0)


    # twolayer.load()
    # _ = Verify(twolayer, test_dataloader)

    # best_model = train(a, loss, train_dataloader, epoch=1000000,lr=0.001, momentum=0.9, regularization=0.00)
    # _ = Verify(a,test_dataloader)

    best_model = train(twolayer, loss, train_dataloader, epoch=1000000)
    _ = Verify(twolayer, test_dataloader)








