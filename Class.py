class Layer:
    def get_output(self, input):
        # give a input and calculate the output
        pass
    def get_gradient(self, input, out_grad):
        # give the input and the gradient of the output to get the gradient of the parameter
        pass
    def get_input_gradient(self):
        # return the gradient of fore layer's output
        pass
    def update(self, lr=0.001, momentum=0, regularization=0):
        # update the parameter of the layer
        pass

    def preserve(self):
        pass

    def load(self, param):
        pass


class Model:
    def forward(self, input):
        # give a list of Layer and input to compute the output
        pass
    def backprop(self, grad_of_loss):
        # update all parameter
        pass

class Loss:
    def get_gradient(self):
        # give the label y and compute the gradient of the last layer
        pass
    def get_input_gradient(self):
        # return the gradient of fore layer's output
        pass
    def get_loss_value(self, y, input):
        # give the input and label to calculate the loss value
        pass


class Activation_Function:
    # h(x*w+b), x*w+b=y
    def get_gradient(self, y):
        # dh/dy
        pass
    def h(self, y):
        # calculate h(y)
        pass


