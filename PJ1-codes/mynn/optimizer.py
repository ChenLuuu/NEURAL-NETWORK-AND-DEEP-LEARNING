from abc import abstractmethod
import numpy as np


class Optimizer:
    def __init__(self, init_lr, model) -> None:
        self.init_lr = init_lr
        self.model = model

    @abstractmethod
    def step(self):
        pass


class SGD(Optimizer):
    def __init__(self, init_lr, model):
        super().__init__(init_lr, model)
    
    def step(self):
        for layer in self.model.layers:
            if layer.optimizable:
                for key in layer.params.keys():
                    if layer.weight_decay:
                        layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda)
                    layer.params[key] -= self.init_lr * layer.grads[key]
                
                if 'W' in layer.params:
                    layer.W = layer.params['W']
                if 'b' in layer.params:
                    layer.b = layer.params['b']

class MomentGD(Optimizer):
    def __init__(self, init_lr, model, mu):
        super().__init__(init_lr, model)
        self.mu = mu  # 动量系数
        self.velocity = {}  # 用于存储每个参数的速度
        
        # 为每个可优化层初始化速度
        for layer in self.model.layers:
            if layer.optimizable:
                self.velocity[layer] = {}
                for key in layer.params.keys():
                    self.velocity[layer][key] = np.zeros_like(layer.params[key])
    
    def step(self):
        for layer in self.model.layers:
            if layer.optimizable:
                for key in layer.params.keys():
                    # 更新速度
                    self.velocity[layer][key] = self.mu * self.velocity[layer][key] - self.init_lr * layer.grads[key]
                    
                   # 如果需要，应用权重衰减
                    if layer.weight_decay:
                        layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda)
                    
                    # 使用速度更新参数
                    layer.params[key] += self.velocity[layer][key]
                
                # 更新层的参数
                if 'W' in layer.params:
                    layer.W = layer.params['W']
                if 'b' in layer.params:
                    layer.b = layer.params['b']