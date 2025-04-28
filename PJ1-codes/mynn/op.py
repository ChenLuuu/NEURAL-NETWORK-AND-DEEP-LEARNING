from abc import abstractmethod
import numpy as np

class Layer():
    """
    所有层的基类，包括训练模式与评估模式的切换，以及参数优化标记
    """
    def __init__(self):
        self.training = True # 标记该层是否具有可优化的参数
        self.optimizable = True

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

def he_init(size):
    """He初始化（用于ReLU激活）"""
    fan_in = size[0]  # 输入维度，用于控制方差
    return np.random.randn(*size) * np.sqrt(2. / fan_in)

class Linear(Layer):
    """
    全连接层：包含前向传播、反向传播和权重更新
    """
    def __init__(self, in_dim, out_dim, initialize_method=he_init, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.W = initialize_method(size=(in_dim, out_dim))  # 权重矩阵初始化，尺寸为 [in_dim, out_dim]
        self.b = initialize_method(size=(1, out_dim))       # 偏置项初始化，尺寸为 [1, out_dim]
        self.grads = {'W': None, 'b': None}
        self.input = None  # 用于保存前向传播时的输入

        self.params = {'W': self.W, 'b': self.b}

        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda
        self.optimizable = True

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        前向传播：计算输出
        输入：X，[batch_size, in_dim]
        输出：Y，[batch_size, out_dim]
        """
        self.input = X  # 缓存输入，用于反向传播时计算梯度
        output = np.dot(X, self.W) + self.b  # Broadcasting handles bias
        return output

    def backward(self, grad: np.ndarray):
        """
        反向传播：计算梯度并返回输入的梯度
        grad：来自下一层的梯度，[batch_size, out_dim]
        返回：输入的梯度，[batch_size, in_dim]
        """
        batch_size = self.input.shape[0]

        # Gradients w.r.t parameters
        grad_W = np.dot(self.input.T, grad) / batch_size  # 权重的梯度，计算方式为 X.T 和 grad 的点积，维度为 [in_dim, out_dim]
        grad_b = np.sum(grad, axis=0, keepdims=True) / batch_size  # 偏置的梯度，按 batch 维度求和，维度为 [1, out_dim]

        # Weight decay (L2 regularization)
        # 如果启用权重衰减，则在权重的梯度中加入 L2 正则项
        if self.weight_decay:
            grad_W += self.weight_decay_lambda * self.W

        self.grads['W'] = grad_W
        self.grads['b'] = grad_b

        # Gradients w.r.t input to pass back
        grad_input = np.dot(grad, self.W.T)  # [batch_size, in_dim]
        return grad_input

    def clear_grad(self):
        self.grads = {'W': None, 'b': None}

class Flatten(Layer):
    """
    展平层：将输入张量 [B, C, H, W] 转换为 [B, C * H * W]
    """
    def __init__(self):
        super().__init__()
        self.input_shape = None
        self.optimizable = False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        """
        前向传播：将输入张量展平
        输入：X，[batch_size, C, H, W]
        输出：展平后的输出，[batch_size, C * H * W]
        """
        self.input_shape = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, grads):
        """
        反向传播：将梯度恢复成输入的形状
        grads：来自下一层的梯度
        返回：恢复后的梯度
        """
        return grads.reshape(self.input_shape)

class MaxPool2D(Layer):
    """
    最大池化层：对每个滑动窗口取最大值
    输入：[B, C, H, W]
    输出：[B, C, H_out, W_out]
    """
    def __init__(self, kernel_size=2, stride=2):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.input = None
        self.argmax_mask = None
        self.optimizable = False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        """
        前向传播：计算最大池化
        """
        self.input = X
        B, C, H, W = X.shape
        k = self.kernel_size
        s = self.stride

        out_H = (H - k) // s + 1
        out_W = (W - k) // s + 1

        # 将 X 划分成滑动窗口
        shape = (B, C, out_H, out_W, k, k)
        strides = (
            X.strides[0],
            X.strides[1],
            X.strides[2] * s,
            X.strides[3] * s,
            X.strides[2],
            X.strides[3],
        )
        patches = np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)
        # shape: [B, C, H_out, W_out, k, k]

        # 记录最大值 & mask
        reshaped = patches.reshape(B, C, out_H, out_W, -1)
        out = reshaped.max(axis=-1)  # [B, C, H_out, W_out]
        # 只在训练模式下保存 mask
        if self.training:
            max_mask = reshaped == out[..., None]  # [B, C, H_out, W_out, k*k]
            self.argmax_mask = max_mask.reshape(patches.shape)

        return out
    
    def backward(self, grads):
        """
        反向传播：根据最大值的掩码将梯度分配到对应的位置
        """
        B, C, H, W = self.input.shape
        k = self.kernel_size
        s = self.stride
        out_H = (H - k) // s + 1
        out_W = (W - k) // s + 1

        dX = np.zeros_like(self.input)
        mask = self.argmax_mask.reshape(B, C, out_H, out_W, k * k)
        grads_expand = grads[..., None]  # [B, C, H_out, W_out, 1]

        grads_broadcasted = grads_expand * mask  # 只对最大位置反传梯度
        grads_broadcasted = grads_broadcasted.reshape(B, C, out_H, out_W, k, k)

        # 将反向传播的梯度填回到原始输入
        for i in range(out_H):
            for j in range(out_W):
                h_start = i * s
                w_start = j * s
                dX[:, :, h_start:h_start+k, w_start:w_start+k] += grads_broadcasted[:, :, i, j, :, :]

        return dX

class conv2D(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 initialize_method=np.random.normal,
                 weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda
        self.optimizable = True

        # 参数初始化
        self.W = initialize_method(size=(out_channels, in_channels, *self.kernel_size))
        self.b = np.zeros((out_channels, 1))

        self.params = {'W': self.W, 'b': self.b}
        self.grads = {'W': None, 'b': None}
        self.input = None

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def im2col(self, X, out_H, out_W, kH, kW):
        """
        将输入数据 X 转换为列形式，用于卷积运算
        输入：[B, C, H, W]
        输出：展开后的列 [B, C*kH*kW, H_out*W_out]
        """
        B, C, H, W = X.shape
        cols = []
        for i in range(out_H):
            for j in range(out_W):
                h_start = i * self.stride
                w_start = j * self.stride
                patch = X[:, :, h_start:h_start + kH, w_start:w_start + kW] # 获取当前 patch
                cols.append(patch.reshape(B, -1)) # 展平成 [B, C*kH*kW]
        X_col = np.stack(cols, axis=-1)  # [B, C*kH*kW, H_out*W_out]
        return X_col

    def forward(self, X):
        """
        前向传播：进行卷积操作，计算输出
        输入：[B, C, H, W]
        输出：[B, K, H_out, W_out]
        """
        if self.training:
            self.input = X

        B, C, H, W = X.shape
        K, _, kH, kW = self.W.shape

        out_H = (H + 2 * self.padding - kH) // self.stride + 1
        out_W = (W + 2 * self.padding - kW) // self.stride + 1

        if self.padding > 0:
            X = np.pad(X, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')

        X_col = self.im2col(X, out_H, out_W, kH, kW)
        W_col = self.W.reshape(K, -1)

        # 计算卷积输出
        out = np.einsum('kc,bcp->bkp', W_col, X_col) + self.b.reshape(1, K, 1)
        out = out.reshape(B, K, out_H, out_W)

        if self.training:
            self.X_padded = X
            self.X_col = X_col
            self.W_col = W_col  # 缓存 W_col

        return out

    def backward(self, grads):
        """
        反向传播：计算梯度并返回输入的梯度
        输入：[B, K, H_out, W_out]
        输入：[B, C, H, W]

        """
        B, K, H_out, W_out = grads.shape
        kH, kW = self.kernel_size

        X_col = self.X_col
        W_col = self.W_col

        grads_reshaped = grads.reshape(B, K, -1)

        dW = np.einsum('bkp,bcp->kc', grads_reshaped, X_col) / B
        dW = dW.reshape(self.W.shape)

        db = np.sum(grads_reshaped, axis=(0, 2), keepdims=True).reshape(self.b.shape) / B

        dX_col = np.einsum('kc,bkp->bcp', W_col, grads_reshaped)

        B, C, H_padded, W_padded = self.X_padded.shape
        dX_padded = np.zeros((B, C, H_padded, W_padded))

        out_idx = 0
        for i in range(H_out):
            for j in range(W_out):
                h_start = i * self.stride
                w_start = j * self.stride
                patch = dX_col[:, :, out_idx].reshape(B, C, kH, kW)
                dX_padded[:, :, h_start:h_start + kH, w_start:w_start + kW] += patch
                out_idx += 1

        if self.padding > 0:
            dX = dX_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            dX = dX_padded

        if self.weight_decay:
            dW += self.weight_decay_lambda * self.W

        self.grads['W'] = dW
        self.grads['b'] = db

        return dX

    def clear_grad(self):
        self.grads = {'W': None, 'b': None}

class ReLU(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None

        self.optimizable =False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X<0, 0, X)
        return output
    
    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)
        return output


class Dropout(Layer):
    """
    Dropout layer for regularization.
    """
    def __init__(self, p=0.5) -> None:
        super().__init__()
        self.p = p  # probability of dropping out
        self.mask = None
        self.optimizable = False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        if self.training:
            # Create dropout mask
            self.mask = (np.random.rand(*X.shape) > self.p) / (1 - self.p)
            return X * self.mask
        else:
            return X
    
    def backward(self, grads):
        if self.training:
            return grads * self.mask
        else:
            return grads

class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it,
    which could be cancelled by method cancel_softmax
    """
    def __init__(self, model=None, max_classes=10) -> None:
        self.model = model
        self.max_classes = max_classes
        self.has_softmax = True  # 默认为带 softmax
        self.preds = None
        self.labels = None
        self.grads = None

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)

    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels : [batch_size, ]
        """
        self.labels = labels
        if self.has_softmax:
            # 做 softmax
            exps = np.exp(predicts - np.max(predicts, axis=1, keepdims=True))
            self.preds = exps / np.sum(exps, axis=1, keepdims=True)
        else:
            self.preds = predicts  # 认为用户已传入 softmax 后的概率

        batch_size = predicts.shape[0]
        # 防止数值错误，添加 epsilon
        epsilon = 1e-12
        # 确保 labels 是整数类型
        labels = labels.astype(np.int64)
        log_probs = -np.log(self.preds[np.arange(batch_size), labels] + epsilon)
        loss = np.mean(log_probs)
        return loss

    def backward(self):
        """
        Backpropagates the gradient of the loss to the model.
        """
        batch_size = self.preds.shape[0]
        self.grads = self.preds.copy()
        self.grads[np.arange(batch_size), self.labels.astype(np.int64)] -= 1
        self.grads /= batch_size

        self.model.backward(self.grads)

    def cancel_soft_max(self):
        self.has_softmax = False
        return self
    
class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """
    def __init__(self, lambda_reg=1e-4):
        super().__init__()
        self.lambda_reg = lambda_reg
        self.optimizable = False
        self.model = None
        self.l2_loss = 0  # 存储当前的L2损失

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        """
        前向传播：直接传递输入，同时计算L2正则化损失
        X: 前一层的输出
        """
        if self.model is None:
            raise ValueError("Model not set for L2Regularization layer. Please set model using set_model() method.")
        
        # 计算L2正则化损失
        self.l2_loss = 0
        for layer in self.model.layers:
            if hasattr(layer, 'params') and layer.optimizable:
                for param in layer.params.values():
                    self.l2_loss += 0.5 * self.lambda_reg * np.sum(param ** 2)
            
        return X  # 直接返回输入，不改变数据

    def backward(self, grads):
        """
        反向传播：计算L2正则化的梯度
        grads: 来自下一层的梯度
        """
        if self.model is None:
            raise ValueError("Model not set for L2Regularization layer. Please set model using set_model() method.")
            
        # 计算L2正则化的梯度
        for layer in self.model.layers:
            if hasattr(layer, 'params') and layer.optimizable:
                for param_name, param in layer.params.items():
                    # 计算L2正则化的梯度
                    l2_grad = self.lambda_reg * param
                    # 将L2梯度添加到原始梯度中
                    if layer.grads[param_name] is not None:
                        layer.grads[param_name] += l2_grad
                    else:
                        layer.grads[param_name] = l2_grad
                        
        return grads  # 直接返回输入的梯度

    def set_model(self, model):
        """
        设置模型引用
        model: 要应用L2正则化的模型
        """
        self.model = model

    def get_l2_loss(self):
        """
        获取当前的L2正则化损失
        """
        return self.l2_loss

def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition