from .op import *
import pickle

class Model_MLP(Layer):
    """
    A simple Multi-Layer Perceptron (MLP) model.
    """
    def __init__(self, size_list=None, act_func=None, lambda_list=None):
    # def __init__(self, size_list=None, act_func=None, lambda_list=None, dropout_p=0.5):
        super().__init__()
        self.size_list = size_list
        self.act_func = act_func
        self.layers = []

        if size_list is not None and act_func is not None:
            for i in range(len(size_list) - 1):
                layer = Linear(in_dim=size_list[i], out_dim=size_list[i + 1])
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
                self.layers.append(layer)

                if i < len(size_list) - 2:
                    if act_func == 'ReLU':
                        self.layers.append(ReLU())
                    # Add dropout after each layer except the last one
                    # self.layers.append(Dropout(p=dropout_p))

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        assert self.size_list is not None and self.act_func is not None, 'Model not initialized. Use `load_model()` or specify `size_list` and `act_func`.'
        # 确保输入是2D的
        if len(X.shape) > 2:
            X = X.reshape(X.shape[0], -1)
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def clear_grad(self):
        for layer in self.layers:
            if layer.optimizable:
                layer.clear_grad()

    def load_model(self, param_path):
        with open(param_path, 'rb') as f:
            param_list = pickle.load(f)
        self.size_list, self.act_func = param_list[:2]

        self.layers = []
        for i in range(len(self.size_list) - 1):
            layer = Linear(in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
            layer.W = param_list[i + 2]['W']
            layer.b = param_list[i + 2]['b']
            layer.params['W'] = layer.W
            layer.params['b'] = layer.b
            layer.weight_decay = param_list[i + 2]['weight_decay']
            layer.weight_decay_lambda = param_list[i + 2]['lambda']
            self.layers.append(layer)

            if i < len(self.size_list) - 2:
                if self.act_func == 'ReLU':
                    self.layers.append(ReLU())

    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({'W': layer.params['W'], 'b': layer.params['b'], 'weight_decay': layer.weight_decay, 'lambda': layer.weight_decay_lambda})
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)

class Model_CNN(Layer):
    """
    定义一个卷积神经网络模型类，继承自 Layer（基础层类）
    """
    def __init__(self, layers=None):
        super().__init__()
        if layers is None:
            raise ValueError("You must provide a list of layers to Model_CNN.")
        self.layers = layers # 接收一个包含神经网络各层的列表

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        out = X
        for layer in self.layers:
            out = layer(out) # 每一层处理后的输出作为下一层的输入
        return out

    def train(self):
        self.training = True
        for layer in self.layers:
            layer.train()

    def eval(self):
        self.training = False
        for layer in self.layers:
            layer.eval()
            
    def backward(self, loss_grad):
        grads = loss_grad # 初始的梯度来自损失函数
        for layer in reversed(self.layers): # 反向遍历所有层
            grads = layer.backward(grads) # 每一层根据当前梯度计算并传递给前一层
        return grads

    def load_model(self, param_path):
        with open(param_path, 'rb') as f:
            param_list = pickle.load(f)
        cnt = 0
        for layer in self.layers:
            if hasattr(layer, 'optimizable') and layer.optimizable:
                layer.W = param_list[cnt]['params']['W']
                layer.b = param_list[cnt]['params']['b']
                layer.params['W'] = layer.W
                layer.params['b'] = layer.b
                layer.weight_decay = param_list[cnt]['weight_decay']
                layer.weight_decay_lambda = param_list[cnt]['lambda']
                cnt += 1

    def save_model(self, save_path):
        param_list = []
        for layer in self.layers:
            if hasattr(layer, 'optimizable') and layer.optimizable:
                param_list.append({
                    'type': layer.__class__.__name__,  # 保存层的类型，例如 'Linear'
                    'params': {
                        'W': layer.params['W'],
                        'b': layer.params['b']
                    },
                    'weight_decay': layer.weight_decay,
                    'lambda': layer.weight_decay_lambda
                })
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)