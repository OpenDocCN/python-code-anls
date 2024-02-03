# `numpy-ml\numpy_ml\tests\nn_torch_models.py`

```py
# 禁用 flake8 检查
# 导入 PyTorch 库
import torch
import torch.nn as nn
import torch.nn.functional as F

# 导入 TensorFlow 库
import tensorflow as tf

# 导入 NumPy 库
import numpy as np

#######################################################################
#       用于测试自定义层的黄金标准实现                               #
#                       (需要 PyTorch)                               #
#######################################################################

# 将输入转换为 PyTorch 变量
def torchify(var, requires_grad=True):
    return torch.autograd.Variable(torch.FloatTensor(var), requires_grad=requires_grad)

# 生成 PyTorch 梯度计算器
def torch_gradient_generator(fn, **kwargs):
    def get_grad(z):
        z1 = torch.autograd.Variable(torch.FloatTensor(z), requires_grad=True)
        z2 = fn(z1, **kwargs).sum()
        z2.backward()
        grad = z1.grad.numpy()
        return grad

    return get_grad

# 计算交叉熵损失函数的梯度
def torch_xe_grad(y, z):
    z = torch.autograd.Variable(torch.FloatTensor(z), requires_grad=True)
    y = torch.LongTensor(y.argmax(axis=1))
    loss = F.cross_entropy(z, y, reduction="sum")
    loss.backward()
    grad = z.grad.numpy()
    return grad

# 计算均方误差损失函数的梯度
def torch_mse_grad(y, z, act_fn):
    y = torch.FloatTensor(y)
    z = torch.autograd.Variable(torch.FloatTensor(z), requires_grad=True)
    y_pred = act_fn(z)
    loss = F.mse_loss(y_pred, y, reduction="sum")  # size_average=False).sum()
    loss.backward()
    grad = z.grad.numpy()
    return grad

# PyTorch VAE 损失函数类
class TorchVAELoss(nn.Module):
    def __init__(self):
        super(TorchVAELoss, self).__init__()
    # 从输入数据中提取梯度信息
    def extract_grads(self, X, X_recon, t_mean, t_log_var):
        # 定义一个极小的浮点数，用于处理梯度计算中的数值稳定性
        eps = np.finfo(float).eps
        # 将输入数据转换为 Torch 张量，并设置不需要梯度信息
        X = torchify(X, requires_grad=False)
        # 将重构后的输入数据转换为 Torch 张量，并进行数值裁剪，避免出现极端值
        X_recon = torchify(np.clip(X_recon, eps, 1 - eps))
        # 将均值数据转换为 Torch 张量
        t_mean = torchify(t_mean)
        # 将对数方差数据转换为 Torch 张量
        t_log_var = torchify(t_log_var)

        # 计算重构误差，使用二元交叉熵损失函数
        BCE = torch.sum(F.binary_cross_entropy(X_recon, X, reduction="none"), dim=1)

        # 计算 KL 散度，参考 VAE 论文的附录 B
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + t_log_var - t_mean.pow(2) - t_log_var.exp(), dim=1)

        # 计算总损失，包括重构误差和 KL 散度
        loss = torch.mean(BCE + KLD)
        # 反向传播计算梯度
        loss.backward()

        # 将损失值和各个梯度信息保存到字典中并返回
        grads = {
            "loss": loss.detach().numpy(),
            "dX_recon": X_recon.grad.numpy(),
            "dt_mean": t_mean.grad.numpy(),
            "dt_log_var": t_log_var.grad.numpy(),
        }
        return grads
# 定义一个 TorchWGANGPLoss 类，继承自 nn.Module
class TorchWGANGPLoss(nn.Module):
    # 初始化函数，接受一个 lambda_ 参数，默认值为 10
    def __init__(self, lambda_=10):
        # 将 lambda_ 转换为张量形式
        self.lambda_ = torchify([lambda_])
        # 调用父类的初始化函数
        super(TorchWGANGPLoss, self).__init__()

    # 前向传播函数，接受 Y_real, Y_fake, gradInterp 三个参数
    def forward(self, Y_real, Y_fake, gradInterp):
        # 复制 Y_fake 到 GY_fake
        GY_fake = Y_fake.copy()
        # 将 Y_real, Y_fake, GY_fake, gradInterp 转换为张量形式
        self.Y_real = torchify(Y_real)
        self.Y_fake = torchify(Y_fake)
        self.GY_fake = torchify(GY_fake)
        self.gradInterp = torchify(gradInterp)

        # 计算梯度惩罚
        norm = self.gradInterp.norm(2, dim=1)
        self.norm1 = torch.sqrt(torch.sum(self.gradInterp.pow(2), dim=1))
        # 断言两种计算方式得到的结果应该非常接近
        assert torch.allclose(norm, self.norm1)

        # 计算梯度惩罚项
        self.gpenalty = self.lambda_ * ((self.norm1 - 1).pow(2)).mean()
        # 计算 C_loss 和 G_loss
        self.C_loss = self.Y_fake.mean() - self.Y_real.mean() + self.gpenalty
        self.G_loss = -self.GY_fake.mean()

    # 提取梯度信息函数，接受 Y_real, Y_fake, gradInterp 三个参数
    def extract_grads(self, Y_real, Y_fake, gradInterp):
        # 调用前向传播函数
        self.forward(Y_real, Y_fake, gradInterp)

        # 计算 C_loss 和 G_loss 的梯度
        self.C_loss.backward()
        self.G_loss.backward()

        # 将各个梯度信息转换为 numpy 数组形式，存储在字典中并返回
        grads = {
            "Y_real": self.Y_real.detach().numpy(),
            "Y_fake": self.Y_fake.detach().numpy(),
            "gradInterp": self.gradInterp.detach().numpy(),
            "GP": self.gpenalty.detach().numpy(),
            "C_loss": self.C_loss.detach().numpy(),
            "G_loss": self.G_loss.detach().numpy(),
            "C_dY_real": self.Y_real.grad.numpy(),
            "C_dGradInterp": self.gradInterp.grad.numpy(),
            "C_dY_fake": self.Y_fake.grad.numpy(),
            "G_dY_fake": self.GY_fake.grad.numpy(),
        }
        return grads

# 定义一个 TorchLinearActivation 类，继承自 nn.Module
class TorchLinearActivation(nn.Module):
    # 初始化函数
    def __init__(self):
        # 调用父类的初始化函数
        super(TorchLinearActivation, self).__init__()
        pass

    # 静态方法，实现前向传播
    @staticmethod
    def forward(input):
        return input

    # 静态方法，实现反向传播
    @staticmethod
    def backward(grad_output):
        return torch.ones_like(grad_output)

# 定义一个 TorchBatchNormLayer 类，继承自 nn.Module
class TorchBatchNormLayer(nn.Module):
    # 初始化批量归一化层对象
    def __init__(self, n_in, params, mode, momentum=0.9, epsilon=1e-5):
        # 调用父类的初始化方法
        super(TorchBatchNormLayer, self).__init__()

        # 从参数中获取缩放因子和截距
        scaler = params["scaler"]
        intercept = params["intercept"]

        # 根据模式选择不同维度的批量归一化层
        if mode == "1D":
            self.layer1 = nn.BatchNorm1d(
                num_features=n_in, momentum=1 - momentum, eps=epsilon, affine=True
            )
        elif mode == "2D":
            self.layer1 = nn.BatchNorm2d(
                num_features=n_in, momentum=1 - momentum, eps=epsilon, affine=True
            )

        # 设置批量归一化层的权重和偏置
        self.layer1.weight = nn.Parameter(torch.FloatTensor(scaler))
        self.layer1.bias = nn.Parameter(torch.FloatTensor(intercept))

    # 前向传播函数
    def forward(self, X):
        # 调整输入张量的维度顺序，从(N, H, W, C)到(N, C, H, W)
        if X.ndim == 4:
            X = np.moveaxis(X, [0, 1, 2, 3], [0, -2, -1, -3])

        # 如果输入不是torch张量，则转换为torch张量
        if not isinstance(X, torch.Tensor):
            X = torchify(X)

        # 保存输入张量和经过批量归一化层后的输出张量
        self.X = X
        self.Y = self.layer1(self.X)
        # 保留输出张量的梯度信息
        self.Y.retain_grad()
    # 从神经网络中提取梯度信息
    def extract_grads(self, X, Y_true=None):
        # 进行前向传播
        self.forward(X)

        # 如果真实标签是 NumPy 数组
        if isinstance(Y_true, np.ndarray):
            # 调整真实标签的维度顺序
            Y_true = np.moveaxis(Y_true, [0, 1, 2, 3], [0, -2, -1, -3])
            # 计算损失函数
            self.loss1 = (
                0.5 * F.mse_loss(self.Y, torchify(Y_true), size_average=False).sum()
            )
        else:
            # 如果没有真实标签，直接将输出求和作为损失
            self.loss1 = self.Y.sum()

        # 反向传播计算梯度
        self.loss1.backward()

        # 将张量转换为 NumPy 数组
        X_np = self.X.detach().numpy()
        Y_np = self.Y.detach().numpy()
        dX_np = self.X.grad.numpy()
        dY_np = self.Y.grad.numpy()

        # 如果输入数据的维度为4
        if self.X.dim() == 4:
            orig, X_swap = [0, 1, 2, 3], [0, -1, -3, -2]
            # 调整真实标签的维度顺序
            if isinstance(Y_true, np.ndarray):
                Y_true = np.moveaxis(Y_true, orig, X_swap)
            X_np = np.moveaxis(X_np, orig, X_swap)
            Y_np = np.moveaxis(Y_np, orig, X_swap)
            dX_np = np.moveaxis(dX_np, orig, X_swap)
            dY_np = np.moveaxis(dY_np, orig, X_swap)

        # 构建梯度字典
        grads = {
            "loss": self.loss1.detach().numpy(),
            "X": X_np,
            "momentum": 1 - self.layer1.momentum,
            "epsilon": self.layer1.eps,
            "intercept": self.layer1.bias.detach().numpy(),
            "scaler": self.layer1.weight.detach().numpy(),
            "running_mean": self.layer1.running_mean.detach().numpy(),
            "running_var": self.layer1.running_var.detach().numpy(),
            "y": Y_np,
            "dLdy": dY_np,
            "dLdIntercept": self.layer1.bias.grad.numpy(),
            "dLdScaler": self.layer1.weight.grad.numpy(),
            "dLdX": dX_np,
        }
        # 如果真实标签是 NumPy 数组，将其加入梯度字典
        if isinstance(Y_true, np.ndarray):
            grads["Y_true"] = Y_true
        # 返回梯度字典
        return grads
# 定义一个继承自 nn.Module 的 TorchLayerNormLayer 类
class TorchLayerNormLayer(nn.Module):
    # 初始化方法，接受特征维度、参数、模式和 epsilon 参数
    def __init__(self, feat_dims, params, mode, epsilon=1e-5):
        super(TorchLayerNormLayer, self).__init__()

        # 创建 LayerNorm 层，指定特征维度、epsilon 值和是否启用 elementwise_affine
        self.layer1 = nn.LayerNorm(
            normalized_shape=feat_dims, eps=epsilon, elementwise_affine=True
        )

        # 从参数中获取 scaler 和 intercept
        scaler = params["scaler"]
        intercept = params["intercept"]

        # 如果模式为 "2D"，则调整 scaler 和 intercept 的维度
        if mode == "2D":
            scaler = np.moveaxis(scaler, [0, 1, 2], [-2, -1, -3])
            intercept = np.moveaxis(intercept, [0, 1, 2], [-2, -1, -3])

        # 断言 scaler 和 intercept 的形状与 LayerNorm 层的权重和偏置形状相同
        assert scaler.shape == self.layer1.weight.shape
        assert intercept.shape == self.layer1.bias.shape

        # 将 scaler 和 intercept 转换为 nn.Parameter 类型，并赋值给 LayerNorm 层的权重和偏置
        self.layer1.weight = nn.Parameter(torch.FloatTensor(scaler))
        self.layer1.bias = nn.Parameter(torch.FloatTensor(intercept))

    # 前向传播方法，接受输入 X
    def forward(self, X):
        # 如果输入 X 的维度为 4，则调整维度顺序为 (N, C, H, W)
        if X.ndim == 4:
            X = np.moveaxis(X, [0, 1, 2, 3], [0, -2, -1, -3])

        # 如果输入 X 不是 torch.Tensor 类型，则转换为 torch.Tensor
        if not isinstance(X, torch.Tensor):
            X = torchify(X)

        # 将输入 X 保存在 self.X 中，并通过 LayerNorm 层得到输出 self.Y
        self.X = X
        self.Y = self.layer1(self.X)
        # 保留 self.Y 的梯度信息
        self.Y.retain_grad()
    # 从输入数据 X 中提取梯度信息，如果提供了真实标签 Y_true，则计算损失
    def extract_grads(self, X, Y_true=None):
        # 进行前向传播
        self.forward(X)

        # 如果 Y_true 是 numpy 数组，则调整其维度顺序
        if isinstance(Y_true, np.ndarray):
            Y_true = np.moveaxis(Y_true, [0, 1, 2, 3], [0, -2, -1, -3])
            # 计算损失函数
            self.loss1 = (
                0.5 * F.mse_loss(self.Y, torchify(Y_true), size_average=False).sum()
            )
        else:
            # 如果没有提供 Y_true，则将损失设为 Y 的总和
            self.loss1 = self.Y.sum()

        # 反向传播计算梯度
        self.loss1.backward()

        # 将张量转换为 numpy 数组
        X_np = self.X.detach().numpy()
        Y_np = self.Y.detach().numpy()
        dX_np = self.X.grad.numpy()
        dY_np = self.Y.grad.numpy()
        intercept_np = self.layer1.bias.detach().numpy()
        scaler_np = self.layer1.weight.detach().numpy()
        dIntercept_np = self.layer1.bias.grad.numpy()
        dScaler_np = self.layer1.weight.grad.numpy()

        # 如果输入数据 X 的维度为 4，则调整维度顺序
        if self.X.dim() == 4:
            orig, X_swap = [0, 1, 2, 3], [0, -1, -3, -2]
            orig_p, p_swap = [0, 1, 2], [-1, -3, -2]
            if isinstance(Y_true, np.ndarray):
                Y_true = np.moveaxis(Y_true, orig, X_swap)
            X_np = np.moveaxis(X_np, orig, X_swap)
            Y_np = np.moveaxis(Y_np, orig, X_swap)
            dX_np = np.moveaxis(dX_np, orig, X_swap)
            dY_np = np.moveaxis(dY_np, orig, X_swap)
            scaler_np = np.moveaxis(scaler_np, orig_p, p_swap)
            intercept_np = np.moveaxis(intercept_np, orig_p, p_swap)
            dScaler_np = np.moveaxis(dScaler_np, orig_p, p_swap)
            dIntercept_np = np.moveaxis(dIntercept_np, orig_p, p_swap)

        # 构建梯度字典
        grads = {
            "loss": self.loss1.detach().numpy(),
            "X": X_np,
            "epsilon": self.layer1.eps,
            "intercept": intercept_np,
            "scaler": scaler_np,
            "y": Y_np,
            "dLdy": dY_np,
            "dLdIntercept": dIntercept_np,
            "dLdScaler": dScaler_np,
            "dLdX": dX_np,
        }
        # 如果提供了 Y_true，则将其加入梯度字典
        if isinstance(Y_true, np.ndarray):
            grads["Y_true"] = Y_true
        # 返回梯度字典
        return grads
class TorchAddLayer(nn.Module):
    # 定义 TorchAddLayer 类，继承自 nn.Module
    def __init__(self, act_fn, **kwargs):
        # 初始化函数，接受激活函数 act_fn 和其他关键字参数
        super(TorchAddLayer, self).__init__()
        # 调用父类的初始化函数
        self.act_fn = act_fn
        # 设置实例变量 act_fn 为传入的激活函数

    def forward(self, Xs):
        # 前向传播函数，接受输入 Xs
        self.Xs = []
        # 初始化实例变量 Xs 为空列表
        x = Xs[0].copy()
        # 复制输入列表中的第一个元素
        if not isinstance(x, torch.Tensor):
            # 如果 x 不是 torch.Tensor 类型
            x = torchify(x)
            # 将 x 转换为 torch.Tensor 类型

        self.sum = x.clone()
        # 克隆 x 并赋值给实例变量 sum
        x.retain_grad()
        # 保留 x 的梯度信息
        self.Xs.append(x)
        # 将 x 添加到 Xs 列表中

        for i in range(1, len(Xs)):
            # 遍历输入列表中的其他元素
            x = Xs[i]
            # 获取当前元素
            if not isinstance(x, torch.Tensor):
                # 如果 x 不是 torch.Tensor 类型
                x = torchify(x)
                # 将 x 转换为 torch.Tensor 类型

            x.retain_grad()
            # 保留 x 的梯度信息
            self.Xs.append(x)
            # 将 x 添加到 Xs 列表中
            self.sum += x
            # 将 x 加到 sum 中

        self.sum.retain_grad()
        # 保留 sum 的梯度信息
        self.Y = self.act_fn(self.sum)
        # 计算 sum 的激活值并赋值给实例变量 Y
        self.Y.retain_grad()
        # 保留 Y 的梯度信息
        return self.Y
        # 返回 Y

    def extract_grads(self, X):
        # 提取梯度信息函数，接受输入 X
        self.forward(X)
        # 调用前向传播函数
        self.loss = self.Y.sum()
        # 计算损失值并赋值给实例变量 loss
        self.loss.backward()
        # 反向传播计算梯度
        grads = {
            # 定义梯度字典
            "Xs": X,
            # 输入 X
            "Sum": self.sum.detach().numpy(),
            # sum 的值
            "Y": self.Y.detach().numpy(),
            # Y 的值
            "dLdY": self.Y.grad.numpy(),
            # Y 的梯度
            "dLdSum": self.sum.grad.numpy(),
            # sum 的梯度
        }
        grads.update(
            # 更新梯度字典
            {"dLdX{}".format(i + 1): xi.grad.numpy() for i, xi in enumerate(self.Xs)}
            # 遍历 Xs 列表，获取每个元素的梯度信息
        )
        return grads
        # 返回梯度字典


class TorchMultiplyLayer(nn.Module):
    # 定义 TorchMultiplyLayer 类，继承自 nn.Module
    def __init__(self, act_fn, **kwargs):
        # 初始化函数，接受激活函数 act_fn 和其他关键字参数
        super(TorchMultiplyLayer, self).__init__()
        # 调用父类的初始化函数
        self.act_fn = act_fn
        # 设置实例变量 act_fn 为传入的激活函数

    def forward(self, Xs):
        # 前向传播函数，接受输入 Xs
        self.Xs = []
        # 初始化实例变量 Xs 为空列表
        x = Xs[0].copy()
        # 复制输入列表中的第一个元素
        if not isinstance(x, torch.Tensor):
            # 如果 x 不是 torch.Tensor 类型
            x = torchify(x)
            # 将 x 转换为 torch.Tensor 类型

        self.prod = x.clone()
        # 克隆 x 并赋值给实例变量 prod
        x.retain_grad()
        # 保留 x 的梯度信息
        self.Xs.append(x)
        # 将 x 添加到 Xs 列表中

        for i in range(1, len(Xs)):
            # 遍历输入列表中的其他元素
            x = Xs[i]
            # 获取当前元素
            if not isinstance(x, torch.Tensor):
                # 如果 x 不是 torch.Tensor 类型
                x = torchify(x)
                # 将 x 转换为 torch.Tensor 类型

            x.retain_grad()
            # 保留 x 的梯度信息
            self.Xs.append(x)
            # 将 x 添加到 Xs 列表中
            self.prod *= x
            # 将 x 乘到 prod 中

        self.prod.retain_grad()
        # 保留 prod 的梯度信息
        self.Y = self.act_fn(self.prod)
        # 计算 prod 的激活值并赋值给实例变量 Y
        self.Y.retain_grad()
        # 保留 Y 的梯度信息
        return self.Y
        # 返回 Y
    # 定义一个方法用于提取梯度信息
    def extract_grads(self, X):
        # 调用神经网络的前向传播方法
        self.forward(X)
        # 计算损失值，将所有元素求和
        self.loss = self.Y.sum()
        # 反向传播计算梯度
        self.loss.backward()
        # 构建包含各个梯度信息的字典
        grads = {
            "Xs": X,  # 输入数据
            "Prod": self.prod.detach().numpy(),  # 中间变量 prod 的值
            "Y": self.Y.detach().numpy(),  # 神经网络输出的值
            "dLdY": self.Y.grad.numpy(),  # 损失函数对 Y 的梯度
            "dLdProd": self.prod.grad.numpy(),  # 损失函数对 prod 的梯度
        }
        # 更新字典，包含每个输入数据对应的梯度信息
        grads.update(
            {"dLdX{}".format(i + 1): xi.grad.numpy() for i, xi in enumerate(self.Xs)}
        )
        # 返回包含梯度信息的字典
        return grads
class TorchSkipConnectionIdentity(nn.Module):
    # 定义一个 TorchSkipConnectionIdentity 类，继承自 nn.Module
    def forward(self, X):
        # 定义 forward 方法，接受输入 X
        if not isinstance(X, torch.Tensor):
            # 如果 X 不是 torch.Tensor 类型
            # 将 X 的维度从 (N, H, W, C) 调整为 (N, C, H, W)
            X = np.moveaxis(X, [0, 1, 2, 3], [0, -2, -1, -3])
            # 将调整后的 X 转换为 torch.Tensor 类型
            X = torchify(X)

        self.X = X
        # 保留 X 的梯度信息
        self.X.retain_grad()

        self.conv1_out = self.conv1(self.X)
        # 保留 conv1_out 的梯度信息
        self.conv1_out.retain_grad()

        self.act_fn1_out = self.act_fn(self.conv1_out)
        # 保留 act_fn1_out 的梯度信息
        self.act_fn1_out.retain_grad()

        self.batchnorm1_out = self.batchnorm1(self.act_fn1_out)
        # 保留 batchnorm1_out 的梯度信息
        self.batchnorm1_out.retain_grad()

        self.conv2_out = self.conv2(self.batchnorm1_out)
        # 保留 conv2_out 的梯度信息
        self.conv2_out.retain_grad()

        self.batchnorm2_out = self.batchnorm2(self.conv2_out)
        # 保留 batchnorm2_out 的梯度信息
        self.batchnorm2_out.retain_grad()

        self.layer3_in = self.batchnorm2_out + self.X
        # 保留 layer3_in 的梯度信息
        self.layer3_in.retain_grad()

        self.Y = self.act_fn(self.layer3_in)
        # 保留 Y 的梯度信息
        self.Y.retain_grad()

class TorchCausalConv1d(torch.nn.Conv1d):
    """https://github.com/pytorch/pytorch/issues/1333

    NB: this is only ensures that the convolution out length is the same as
    the input length IFF stride = 1. Otherwise, in/out lengths will differ.
    """
    # 定义 TorchCausalConv1d 类，继承自 torch.nn.Conv1d
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        groups=1,
        bias=True,
    ):
        # 初始化方法，接受输入参数
        self.__padding = (kernel_size - 1) * dilation

        super(TorchCausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.__padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, input):
        # 定义 forward 方法，接受输入 input
        result = super(TorchCausalConv1d, self).forward(input)
        if self.__padding != 0:
            return result[:, :, : -self.__padding]
        return result

class TorchWavenetModule(nn.Module):
    # 定义 TorchWavenetModule 类，继承自 nn.Module
    # 初始化 TorchWavenetModule 类，接受参数 params, hparams, conv_1x1_pad
    def __init__(self, params, hparams, conv_1x1_pad):
        # 调用父类的初始化方法
        super(TorchWavenetModule, self).__init__()
        
        # 创建 TorchCausalConv1d 对象，用于实现膨胀卷积
        self.conv_dilation = TorchCausalConv1d(
            in_channels=hparams["components"]["conv_dilation"]["in_ch"],
            out_channels=hparams["components"]["conv_dilation"]["out_ch"],
            kernel_size=hparams["components"]["conv_dilation"]["kernel_width"],
            stride=hparams["components"]["conv_dilation"]["stride"],
            dilation=hparams["components"]["conv_dilation"]["dilation"] + 1,
            bias=True,
        )

        # 创建 nn.Conv1d 对象，用于实现 1x1 卷积
        self.conv_1x1 = nn.Conv1d(
            in_channels=hparams["components"]["conv_1x1"]["in_ch"],
            out_channels=hparams["components"]["conv_1x1"]["out_ch"],
            kernel_size=hparams["components"]["conv_1x1"]["kernel_width"],
            stride=hparams["components"]["conv_1x1"]["stride"],
            padding=conv_1x1_pad,
            dilation=hparams["components"]["conv_1x1"]["dilation"] + 1,
            bias=True,
        )

        # 初始化膨胀卷积的权重和偏置
        W = params["components"]["conv_dilation"]["W"]
        b = params["components"]["conv_dilation"]["b"]
        W = np.moveaxis(W, [0, 1, 2], [-1, -2, -3])  # 调整权重的维度顺序
        self.conv_dilation.weight = nn.Parameter(torch.FloatTensor(W))
        self.conv_dilation.bias = nn.Parameter(torch.FloatTensor(b.flatten()))
        assert self.conv_dilation.weight.shape == W.shape
        assert self.conv_dilation.bias.shape == b.flatten().shape

        # 初始化 1x1 卷积的权重和偏置
        W = params["components"]["conv_1x1"]["W"]
        b = params["components"]["conv_1x1"]["b"]
        W = np.moveaxis(W, [0, 1, 2], [-1, -2, -3])  # 调整权重的维度顺序
        self.conv_1x1.weight = nn.Parameter(torch.FloatTensor(W))
        self.conv_1x1.bias = nn.Parameter(torch.FloatTensor(b.flatten()))
        assert self.conv_1x1.weight.shape == W.shape
        assert self.conv_1x1.bias.shape == b.flatten().shape
    def forward(self, X_main, X_skip):
        # 将输入数据的维度顺序从(N, W, C)转换为(N, C, W)
        self.X_main = np.moveaxis(X_main, [0, 1, 2], [0, -1, -2])
        # 将转换后的数据转换为torch张量
        self.X_main = torchify(self.X_main)
        # 保留梯度信息
        self.X_main.retain_grad()

        # 使用卷积扩张操作处理转换后的数据
        self.conv_dilation_out = self.conv_dilation(self.X_main)
        self.conv_dilation_out.retain_grad()

        # 对卷积扩张输出进行tanh和sigmoid激活函数处理
        self.tanh_out = torch.tanh(self.conv_dilation_out)
        self.sigm_out = torch.sigmoid(self.conv_dilation_out)

        # 保留梯度信息
        self.tanh_out.retain_grad()
        self.sigm_out.retain_grad()

        # 将tanh和sigmoid输出相乘
        self.multiply_gate_out = self.tanh_out * self.sigm_out
        self.multiply_gate_out.retain_grad()

        # 使用1x1卷积处理相乘结果
        self.conv_1x1_out = self.conv_1x1(self.multiply_gate_out)
        self.conv_1x1_out.retain_grad()

        # 初始化X_skip为与conv_1x1_out相同形状的全零张量
        self.X_skip = torch.zeros_like(self.conv_1x1_out)
        # 如果X_skip不为空，则将其转换为torch张量
        if X_skip is not None:
            self.X_skip = torchify(np.moveaxis(X_skip, [0, 1, 2], [0, -1, -2]))
        self.X_skip.retain_grad()

        # 计算Y_skip和Y_main
        self.Y_skip = self.X_skip + self.conv_1x1_out
        self.Y_main = self.X_main + self.conv_1x1_out

        # 保留梯度信息
        self.Y_skip.retain_grad()
        self.Y_main.retain_grad()
class TorchSkipConnectionConv(nn.Module):
    def __init__(
        self, act_fn, pad1, pad2, pad_skip, params, hparams, momentum=0.9, epsilon=1e-5
    # 初始化函数，定义了跳跃连接卷积层的参数和超参数
    def forward(self, X):
        # 检查输入是否为 torch.Tensor 类型，如果不是则进行转换
        if not isinstance(X, torch.Tensor):
            # 将输入的维度顺序从 (N, H, W, C) 调整为 (N, C, H, W)
            X = np.moveaxis(X, [0, 1, 2, 3], [0, -2, -1, -3])
            X = torchify(X)

        self.X = X
        self.X.retain_grad()

        # 对输入进行第一次卷积操作
        self.conv1_out = self.conv1(self.X)
        self.conv1_out.retain_grad()

        # 对第一次卷积结果应用激活函数
        self.act_fn1_out = self.act_fn(self.conv1_out)
        self.act_fn1_out.retain_grad()

        # 对激活函数输出进行批归一化
        self.batchnorm1_out = self.batchnorm1(self.act_fn1_out)
        self.batchnorm1_out.retain_grad()

        # 对批归一化结果进行第二次卷积操作
        self.conv2_out = self.conv2(self.batchnorm1_out)
        self.conv2_out.retain_grad()

        # 对第二次卷积结果进行批归一化
        self.batchnorm2_out = self.batchnorm2(self.conv2_out)
        self.batchnorm2_out.retain_grad()

        # 对输入进行跳跃连接卷积操作
        self.c_skip_out = self.conv_skip(self.X)
        self.c_skip_out.retain_grad()

        # 对跳跃连接卷积结果进行批归一化
        self.bn_skip_out = self.batchnorm_skip(self.c_skip_out)
        self.bn_skip_out.retain_grad()

        # 将第二次卷积结果和跳跃连接卷积结果相加作为第三层的输入
        self.layer3_in = self.batchnorm2_out + self.bn_skip_out
        self.layer3_in.retain_grad()

        # 对第三层的输入应用激活函数
        self.Y = self.act_fn(self.layer3_in)
        self.Y.retain_grad()

class TorchBidirectionalLSTM(nn.Module):
    def forward(self, X):
        # 将输入的维度顺序从 (batch, input_size, seq_len) 调整为 (seq_len, batch, input_size)
        self.X = np.moveaxis(X, [0, 1, 2], [-2, -1, -3])

        # 检查输入是否为 torch.Tensor 类型，如果不是则进行转换
        if not isinstance(self.X, torch.Tensor):
            self.X = torchify(self.X)

        self.X.retain_grad()

        # 初始化隐藏状态为 0
        n_ex, n_in, n_timesteps = self.X.shape
        n_out, n_out = self.layer1.weight_hh_l0.shape

        # 前向传播
        self.A, (At, Ct) = self.layer1(self.X)
        self.A.retain_grad()
        return self.A

class TorchPool2DLayer(nn.Module):
    # 初始化 TorchPool2DLayer 类，设置输入通道数和超参数
    def __init__(self, in_channels, hparams, **kwargs):
        # 调用父类的初始化方法
        super(TorchPool2DLayer, self).__init__()

        # 根据超参数中的模式选择不同的池化层
        if hparams["mode"] == "max":
            self.layer1 = nn.MaxPool2d(
                kernel_size=hparams["kernel_shape"],
                padding=hparams["pad"],
                stride=hparams["stride"],
            )
        elif hparams["mode"] == "average":
            self.layer1 = nn.AvgPool2d(
                kernel_size=hparams["kernel_shape"],
                padding=hparams["pad"],
                stride=hparams["stride"],
            )

    # 前向传播函数
    def forward(self, X):
        # 将输入数据的维度顺序从 (N, H, W, C) 调整为 (N, C, H, W)
        self.X = np.moveaxis(X, [0, 1, 2, 3], [0, -2, -1, -3])
        # 如果输入数据不是 torch.Tensor 类型，则转换为 torch.Tensor
        if not isinstance(self.X, torch.Tensor):
            self.X = torchify(self.X)

        # 保留输入数据的梯度
        self.X.retain_grad()
        # 对输入数据进行池化操作，得到输出数据
        self.Y = self.layer1(self.X)
        # 保留输出数据的梯度
        self.Y.retain_grad()
        # 返回输出数据
        return self.Y

    # 提取梯度信息函数
    def extract_grads(self, X):
        # 运行前向传播函数得到输出数据
        self.forward(X)
        # 计算损失函数为输出数据的和
        self.loss = self.Y.sum()
        # 反向传播计算梯度
        self.loss.backward()

        # 调整梯度信息的维度顺序，以适应不同的表示方式
        orig, X_swap = [0, 1, 2, 3], [0, -1, -3, -2]
        grads = {
            "X": np.moveaxis(self.X.detach().numpy(), orig, X_swap),
            "y": np.moveaxis(self.Y.detach().numpy(), orig, X_swap),
            "dLdY": np.moveaxis(self.Y.grad.numpy(), orig, X_swap),
            "dLdX": np.moveaxis(self.X.grad.numpy(), orig, X_swap),
        }
        # 返回梯度信息字典
        return grads
# 定义一个 TorchConv2DLayer 类，继承自 nn.Module 类
class TorchConv2DLayer(nn.Module):
    # 初始化函数，接受输入通道数、输出通道数、激活函数、参数、超参数等参数
    def __init__(self, in_channels, out_channels, act_fn, params, hparams, **kwargs):
        # 调用父类的初始化函数
        super(TorchConv2DLayer, self).__init__()

        # 从参数中获取权重 W 和偏置 b
        W = params["W"]
        b = params["b"]
        # 保存激活函数
        self.act_fn = act_fn

        # 创建一个卷积层，设置输入通道数、输出通道数、卷积核形状、填充、步长、膨胀等参数
        self.layer1 = nn.Conv2d(
            in_channels,
            out_channels,
            hparams["kernel_shape"],
            padding=hparams["pad"],
            stride=hparams["stride"],
            dilation=hparams["dilation"] + 1,
            bias=True,
        )

        # 调整权重 W 的维度顺序，使其与卷积层的权重维度匹配
        W = np.moveaxis(W, [0, 1, 2, 3], [-2, -1, -3, -4])
        # 断言卷积层的权重形状与调整后的 W 的形状相同
        assert self.layer1.weight.shape == W.shape
        # 断言卷积层的偏置形状与展平后的 b 的形状相同
        assert self.layer1.bias.shape == b.flatten().shape

        # 将调整后的 W 转换为 PyTorch 的参数形式，并赋值给卷积层的权重
        self.layer1.weight = nn.Parameter(torch.FloatTensor(W))
        # 将展平后的 b 转换为 PyTorch 的参数形式，并赋值给卷积层的偏置
        self.layer1.bias = nn.Parameter(torch.FloatTensor(b.flatten()))

    # 前向传播函数，接受输入 X，进行卷积操作和激活函数操作，并返回结果
    def forward(self, X):
        # 调整输入 X 的维度顺序，使其与卷积层的输入维度匹配
        self.X = np.moveaxis(X, [0, 1, 2, 3], [0, -2, -1, -3])
        # 如果输入 X 不是 torch.Tensor 类型，则转换为 torch.Tensor 类型
        if not isinstance(self.X, torch.Tensor):
            self.X = torchify(self.X)

        # 保留输入 X 的梯度信息
        self.X.retain_grad()

        # 对输入 X 进行卷积操作，保存结果并保留梯度信息
        self.Z = self.layer1(self.X)
        self.Z.retain_grad()

        # 对卷积结果进行激活函数操作，保存结果并保留梯度信息
        self.Y = self.act_fn(self.Z)
        self.Y.retain_grad()
        # 返回激活函数的结果
        return self.Y
    # 提取梯度信息
    def extract_grads(self, X):
        # 进行前向传播
        self.forward(X)
        # 计算损失值
        self.loss = self.Y.sum()
        # 反向传播计算梯度
        self.loss.backward()

        # 定义坐标转换规则
        orig, X_swap, W_swap = [0, 1, 2, 3], [0, -1, -3, -2], [-1, -2, -4, -3]
        # 提取各个梯度信息并进行坐标转换
        grads = {
            "X": np.moveaxis(self.X.detach().numpy(), orig, X_swap),
            "W": np.moveaxis(self.layer1.weight.detach().numpy(), orig, W_swap),
            "b": self.layer1.bias.detach().numpy().reshape(1, 1, 1, -1),
            "y": np.moveaxis(self.Y.detach().numpy(), orig, X_swap),
            "dLdY": np.moveaxis(self.Y.grad.numpy(), orig, X_swap),
            "dLdZ": np.moveaxis(self.Z.grad.numpy(), orig, X_swap),
            "dLdW": np.moveaxis(self.layer1.weight.grad.numpy(), orig, W_swap),
            "dLdB": self.layer1.bias.grad.numpy().reshape(1, 1, 1, -1),
            "dLdX": np.moveaxis(self.X.grad.numpy(), orig, X_swap),
        }
        # 返回梯度信息字典
        return grads
class TorchConv1DLayer(nn.Module):
    # 定义一个继承自 nn.Module 的 TorchConv1DLayer 类
    def __init__(self, in_channels, out_channels, act_fn, params, hparams, **kwargs):
        # 初始化函数，接受输入通道数、输出通道数、激活函数、参数、超参数等参数

        # 调用父类的初始化函数
        super(TorchConv1DLayer, self).__init__()

        # 从参数中获取权重 W 和偏置 b
        W = params["W"]
        b = params["b"]
        self.act_fn = act_fn

        # 创建一个一维卷积层，设置输入通道数、输出通道数、卷积核宽度、填充、步长、膨胀等参数
        self.layer1 = nn.Conv1d(
            in_channels,
            out_channels,
            hparams["kernel_width"],
            padding=hparams["pad"],
            stride=hparams["stride"],
            dilation=hparams["dilation"] + 1,
            bias=True,
        )

        # 调整权重 W 的维度顺序
        W = np.moveaxis(W, [0, 1, 2], [-1, -2, -3])
        # 断言卷积层的权重形状与调整后的 W 的形状相同
        assert self.layer1.weight.shape == W.shape
        # 断言卷积层的偏置形状与展平后的 b 的形状相同
        assert self.layer1.bias.shape == b.flatten().shape

        # 将调整后的 W 赋值给卷积层的权重
        self.layer1.weight = nn.Parameter(torch.FloatTensor(W))
        # 将展平后的 b 赋值给卷积层的偏置
        self.layer1.bias = nn.Parameter(torch.FloatTensor(b.flatten()))

    def forward(self, X):
        # 前向传播函数，接受输入 X

        # 调整输入 X 的维度顺序
        self.X = np.moveaxis(X, [0, 1, 2], [0, -1, -2])
        # 如果输入 X 不是 torch.Tensor 类型，则转换为 torch.Tensor 类型
        if not isinstance(self.X, torch.Tensor):
            self.X = torchify(self.X)

        # 保留输入 X 的梯度信息
        self.X.retain_grad()

        # 对输入 X 进行卷积操作，得到 Z
        self.Z = self.layer1(self.X)
        # 保留 Z 的梯度信息
        self.Z.retain_grad()

        # 对 Z 应用激活函数，得到 Y
        self.Y = self.act_fn(self.Z)
        # 保留 Y 的梯度信息
        self.Y.retain_grad()
        # 返回 Y
        return self.Y
    # 提取梯度信息
    def extract_grads(self, X):
        # 进行前向传播
        self.forward(X)
        # 计算损失值
        self.loss = self.Y.sum()
        # 反向传播计算梯度
        self.loss.backward()

        # 定义坐标转换规则
        orig, X_swap, W_swap = [0, 1, 2], [0, -1, -2], [-1, -2, -3]
        # 提取各个梯度信息并进行坐标转换
        grads = {
            "X": np.moveaxis(self.X.detach().numpy(), orig, X_swap),
            "W": np.moveaxis(self.layer1.weight.detach().numpy(), orig, W_swap),
            "b": self.layer1.bias.detach().numpy().reshape(1, 1, -1),
            "y": np.moveaxis(self.Y.detach().numpy(), orig, X_swap),
            "dLdY": np.moveaxis(self.Y.grad.numpy(), orig, X_swap),
            "dLdZ": np.moveaxis(self.Z.grad.numpy(), orig, X_swap),
            "dLdW": np.moveaxis(self.layer1.weight.grad.numpy(), orig, W_swap),
            "dLdB": self.layer1.bias.grad.numpy().reshape(1, 1, -1),
            "dLdX": np.moveaxis(self.X.grad.numpy(), orig, X_swap),
        }
        # 返回梯度信息字典
        return grads
# 定义一个 TorchDeconv2DLayer 类，继承自 nn.Module
class TorchDeconv2DLayer(nn.Module):
    # 初始化函数，接受输入通道数、输出通道数、激活函数、参数、超参数等参数
    def __init__(self, in_channels, out_channels, act_fn, params, hparams, **kwargs):
        # 调用父类的初始化函数
        super(TorchDeconv2DLayer, self).__init__()

        # 从参数中获取权重和偏置
        W = params["W"]
        b = params["b"]
        self.act_fn = act_fn

        # 创建一个反卷积层，设置输入通道数、输出通道数、卷积核形状、填充、步幅等参数
        self.layer1 = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            hparams["kernel_shape"],
            padding=hparams["pad"],
            stride=hparams["stride"],
            dilation=1,
            bias=True,
        )

        # 调整权重的维度顺序，使其与反卷积层的权重维度匹配
        W = np.moveaxis(W, [0, 1, 2, 3], [-2, -1, -4, -3])
        # 断言反卷积层的权重形状与调整后的权重形状相同
        assert self.layer1.weight.shape == W.shape
        # 断言反卷积层的偏置形状与调整后的偏置形状相同
        assert self.layer1.bias.shape == b.flatten().shape

        # 将调整后的权重设置为反卷积层的权重参数
        self.layer1.weight = nn.Parameter(torch.FloatTensor(W))
        # 将调整后的偏置设置为反卷积层的偏置参数
        self.layer1.bias = nn.Parameter(torch.FloatTensor(b.flatten()))

    # 前向传播函数，接受输入数据 X，返回激活后的输出数据 Y
    def forward(self, X):
        # 调整输入数据的维度顺序，使其与反卷积层的输入数据维度匹配
        self.X = np.moveaxis(X, [0, 1, 2, 3], [0, -2, -1, -3])
        # 如果输入数据不是 torch.Tensor 类型，则转换为 torch.Tensor 类型
        if not isinstance(self.X, torch.Tensor):
            self.X = torchify(self.X)

        # 保留输入数据的梯度信息
        self.X.retain_grad()

        # 将输入数据传入反卷积层，得到输出数据 Z，并保留输出数据的梯度信息
        self.Z = self.layer1(self.X)
        self.Z.retain_grad()

        # 对输出数据 Z 应用激活函数，得到最终输出数据 Y，并保留输出数据的梯度信息
        self.Y = self.act_fn(self.Z)
        self.Y.retain_grad()
        # 返回最终输出数据 Y
        return self.Y
    # 提取梯度信息
    def extract_grads(self, X):
        # 进行前向传播
        self.forward(X)
        # 计算损失值
        self.loss = self.Y.sum()
        # 反向传播计算梯度
        self.loss.backward()

        # 定义坐标转换规则
        orig, X_swap, W_swap = [0, 1, 2, 3], [0, -1, -3, -2], [-2, -1, -4, -3]
        # 提取各个梯度信息并进行坐标转换
        grads = {
            "X": np.moveaxis(self.X.detach().numpy(), orig, X_swap),
            "W": np.moveaxis(self.layer1.weight.detach().numpy(), orig, W_swap),
            "b": self.layer1.bias.detach().numpy().reshape(1, 1, 1, -1),
            "y": np.moveaxis(self.Y.detach().numpy(), orig, X_swap),
            "dLdY": np.moveaxis(self.Y.grad.numpy(), orig, X_swap),
            "dLdZ": np.moveaxis(self.Z.grad.numpy(), orig, X_swap),
            "dLdW": np.moveaxis(self.layer1.weight.grad.numpy(), orig, W_swap),
            "dLdB": self.layer1.bias.grad.numpy().reshape(1, 1, 1, -1),
            "dLdX": np.moveaxis(self.X.grad.numpy(), orig, X_swap),
        }
        # 返回梯度信息字典
        return grads
# 定义一个继承自 nn.Module 的 TorchLSTMCell 类
class TorchLSTMCell(nn.Module):
    # 初始化方法，接受输入维度、输出维度、参数字典和其他关键字参数
    def __init__(self, n_in, n_out, params, **kwargs):
        # 调用父类的初始化方法
        super(TorchLSTMCell, self).__init__()

        # 从参数字典中获取权重矩阵，并转置
        Wiu = params["Wu"][n_out:, :].T
        Wif = params["Wf"][n_out:, :].T
        Wic = params["Wc"][n_out:, :].T
        Wio = params["Wo"][n_out:, :].T
        # 将权重矩阵堆叠成输入权重矩阵
        W_ih = np.vstack([Wiu, Wif, Wic, Wio])

        # 从参数字典中获取权重矩阵，并转置
        Whu = params["Wu"][:n_out, :].T
        Whf = params["Wf"][:n_out, :].T
        Whc = params["Wc"][:n_out, :].T
        Who = params["Wo"][:n_out, :].T
        # 将权重矩阵堆叠成隐藏状态权重矩阵
        W_hh = np.vstack([Whu, Whf, Whc, Who])

        # 创建一个 LSTMCell 层，设置输入维度、输出维度和是否包含偏置
        self.layer1 = nn.LSTMCell(input_size=n_in, hidden_size=n_out, bias=True)
        # 断言输入权重矩阵的形状与 LSTMCell 层的输入权重矩阵形状相同
        assert self.layer1.weight_ih.shape == W_ih.shape
        # 断言隐藏状态权重矩阵的形状与 LSTMCell 层的隐藏状态权重矩阵形状相同
        assert self.layer1.weight_hh.shape == W_hh.shape
        # 将输入权重矩阵转换为可训练参数并赋值给 LSTMCell 层的输入权重矩阵
        self.layer1.weight_ih = nn.Parameter(torch.FloatTensor(W_ih))
        # 将隐藏状态权重矩阵转换为可训练参数并赋值给 LSTMCell 层的隐藏状态权重矩阵

        self.layer1.weight_hh = nn.Parameter(torch.FloatTensor(W_hh))

        # 将偏置参数从参数字典中提取并拼接成一个一维数组
        b = np.concatenate(
            [params["bu"], params["bf"], params["bc"], params["bo"]], axis=-1
        ).flatten()
        # 断言输入偏置参数的形状与 LSTMCell 层的输入偏置参数形状相同
        assert self.layer1.bias_ih.shape == b.shape
        # 断言隐藏状态偏置参数的形状与 LSTMCell 层的隐藏状态偏置参数形状相同
        assert self.layer1.bias_hh.shape == b.shape
        # 将偏置参数转换为可训练参数并赋值给 LSTMCell 层的输入偏置参数
        self.layer1.bias_ih = nn.Parameter(torch.FloatTensor(b))
        # 将偏置参数转换为可训练参数并赋值给 LSTMCell 层的隐藏状态偏置参数
        self.layer1.bias_hh = nn.Parameter(torch.FloatTensor(b))
    # 定义一个前向传播函数，接受输入 X
    def forward(self, X):
        # 将输入 X 存储在对象中
        self.X = X
        # 如果输入 X 不是 torch.Tensor 类型，则将其转换为 torch.Tensor 类型
        if not isinstance(self.X, torch.Tensor):
            self.X = torchify(self.X)

        # 保留输入 X 的梯度信息
        self.X.retain_grad()

        # 初始化隐藏状态为 0
        n_ex, n_in, n_timesteps = self.X.shape
        n_out, n_out = self.layer1.weight_hh.shape

        # 初始化隐藏状态 a0 和 c0
        a0 = torchify(np.zeros((n_ex, n_out)))
        c0 = torchify(np.zeros((n_ex, n_out)))
        a0.retain_grad()
        c0.retain_grad()

        # 执行前向传播
        A, C = [], []
        at = a0
        ct = c0
        for t in range(n_timesteps):
            A.append(at)
            C.append(ct)
            at1, ct1 = self.layer1(self.X[:, :, t], (at, ct))
            at.retain_grad()
            ct.retain_grad()
            at = at1
            ct = ct1

        at.retain_grad()
        ct.retain_grad()
        A.append(at)
        C.append(ct)

        # 不包括 a0 在输出中
        self.A = A[1:]
        self.C = C[1:]
        # 返回隐藏状态 A 和 C
        return self.A, self.C
class TorchRNNCell(nn.Module):
    # 定义 TorchRNNCell 类，继承自 nn.Module
    def __init__(self, n_in, n_hid, params, **kwargs):
        # 初始化方法
        super(TorchRNNCell, self).__init__()
        # 创建一个 RNNCell 层，输入维度为 n_in，隐藏层维度为 n_hid，包含偏置，激活函数为 tanh
        self.layer1 = nn.RNNCell(n_in, n_hid, bias=True, nonlinearity="tanh")

        # 设置权重和偏置以匹配 RNNCell 的权重和偏置
        # 注意：我们将 RNNCell 的权重和偏置的转置传递给 pytorch，这意味着我们需要针对权重的转置检查我们的输出的转置
        self.layer1.weight_ih = nn.Parameter(torch.FloatTensor(params["Wax"].T))
        self.layer1.weight_hh = nn.Parameter(torch.FloatTensor(params["Waa"].T))
        self.layer1.bias_ih = nn.Parameter(torch.FloatTensor(params["bx"].T))
        self.layer1.bias_hh = nn.Parameter(torch.FloatTensor(params["ba"].T))

    def forward(self, X):
        # 前向传播方法
        self.X = X
        if not isinstance(self.X, torch.Tensor):
            self.X = torchify(self.X)

        self.X.retain_grad()

        # 初始隐藏状态为 0
        n_ex, n_in, n_timesteps = self.X.shape
        n_out, n_out = self.layer1.weight_hh.shape

        # 初始化隐藏状态
        a0 = torchify(np.zeros((n_ex, n_out)))
        a0.retain_grad()

        # 前向传播
        A = []
        at = a0
        for t in range(n_timesteps):
            A += [at]
            at1 = self.layer1(self.X[:, :, t], at)
            at.retain_grad()
            at = at1

        at.retain_grad()
        A += [at]

        # 不包括 a0 在我们的输出中
        self.A = A[1:]
        return self.A
    # 定义一个方法用于提取梯度信息
    def extract_grads(self, X):
        # 运行前向传播
        self.forward(X)
        # 计算损失值并将所有损失值叠加在一起
        self.loss = torch.stack(self.A).sum()
        # 反向传播计算梯度
        self.loss.backward()
        # 提取并保存各个参数的梯度信息到字典中
        grads = {
            "X": self.X.detach().numpy(),
            "ba": self.layer1.bias_hh.detach().numpy(),
            "bx": self.layer1.bias_ih.detach().numpy(),
            "Wax": self.layer1.weight_ih.detach().numpy(),
            "Waa": self.layer1.weight_hh.detach().numpy(),
            "y": torch.stack(self.A).detach().numpy(),
            "dLdA": np.array([a.grad.numpy() for a in self.A]),
            "dLdWaa": self.layer1.weight_hh.grad.numpy(),
            "dLdWax": self.layer1.weight_ih.grad.numpy(),
            "dLdBa": self.layer1.bias_hh.grad.numpy(),
            "dLdBx": self.layer1.bias_ih.grad.numpy(),
            "dLdX": self.X.grad.numpy(),
        }
        # 返回保存梯度信息的字典
        return grads
class TorchFCLayer(nn.Module):
    # 定义一个全连接层的类
    def __init__(self, n_in, n_hid, act_fn, params, **kwargs):
        # 初始化函数，接受输入维度、隐藏层维度、激活函数、参数等参数
        super(TorchFCLayer, self).__init__()
        # 调用父类的初始化函数
        self.layer1 = nn.Linear(n_in, n_hid)
        # 创建一个线性层，输入维度为n_in，输出维度为n_hid

        # explicitly set weights and bias
        # 明确设置权重和偏置
        # NB: we pass the *transpose* of the weights to pytorch, meaning
        # we'll need to check against the *transpose* of our outputs for
        # any function of the weights
        # 注意：我们将权重的转置传递给pytorch，这意味着我们需要检查权重的输出的转置
        self.layer1.weight = nn.Parameter(torch.FloatTensor(params["W"].T))
        # 设置权重为参数中W的转置
        self.layer1.bias = nn.Parameter(torch.FloatTensor(params["b"]))
        # 设置偏置为参数中b

        self.act_fn = act_fn
        # 设置激活函数
        self.model = nn.Sequential(self.layer1, self.act_fn)
        # 创建一个包含线性层和激活函数的序列模型

    def forward(self, X):
        # 前向传播函数
        self.X = X
        # 保存输入数据
        if not isinstance(X, torch.Tensor):
            self.X = torchify(X)
        # 如果输入数据不是torch张量，则转换为torch张量

        self.z1 = self.layer1(self.X)
        # 计算线性层的输出
        self.z1.retain_grad()
        # 保留梯度信息

        self.out1 = self.act_fn(self.z1)
        # 计算激活函数的输出
        self.out1.retain_grad()
        # 保留梯度信息

    def extract_grads(self, X):
        # 提取梯度信息的函数
        self.forward(X)
        # 调用前向传播函数
        self.loss1 = self.out1.sum()
        # 计算损失值
        self.loss1.backward()
        # 反向传播计算梯度
        grads = {
            "X": self.X.detach().numpy(),
            "b": self.layer1.bias.detach().numpy(),
            "W": self.layer1.weight.detach().numpy(),
            "y": self.out1.detach().numpy(),
            "dLdy": self.out1.grad.numpy(),
            "dLdZ": self.z1.grad.numpy(),
            "dLdB": self.layer1.bias.grad.numpy(),
            "dLdW": self.layer1.weight.grad.numpy(),
            "dLdX": self.X.grad.numpy(),
        }
        # 保存梯度信息到字典中
        return grads
        # 返回梯度信息字典


class TorchEmbeddingLayer(nn.Module):
    # 定义一个嵌入层的类
    def __init__(self, vocab_size, n_out, params, **kwargs):
        # 初始化函数，接受词汇表大小、输出维度、参数等参数
        super(TorchEmbeddingLayer, self).__init__()
        # 调用父类的初始化函数
        self.layer1 = nn.Embedding(vocab_size, n_out)
        # 创建一个嵌入层，词汇表大小为vocab_size，输出维度为n_out

        # explicitly set embedding weights
        # 明确设置嵌入权重
        self.layer1.weight = nn.Parameter(torch.FloatTensor(params["W"]))
        # 设置嵌入层的权重为参数中的W
        self.model = nn.Sequential(self.layer1)
        # 创建一个包含嵌入层的序列模型
    # 定义一个前向传播函数，接受输入 X
    def forward(self, X):
        # 将输入 X 存储在对象中
        self.X = X
        # 如果输入 X 不是 torch.Tensor 类型，则将其转换为 torch.Tensor 类型
        if not isinstance(X, torch.Tensor):
            self.X = torch.from_numpy(X)

        # 将输入 X 传递给第一层神经网络，并存储输出
        self.out1 = self.layer1(self.X)
        # 保留输出的梯度信息
        self.out1.retain_grad()

    # 定义一个提取梯度信息的函数，接受输入 X
    def extract_grads(self, X):
        # 调用前向传播函数
        self.forward(X)
        # 计算损失函数 loss1，为输出的和
        self.loss1 = self.out1.sum()
        # 反向传播计算梯度
        self.loss1.backward()
        # 提取并返回梯度信息
        grads = {
            "X": self.X.detach().numpy(),
            "W": self.layer1.weight.detach().numpy(),
            "y": self.out1.detach().numpy(),
            "dLdy": self.out1.grad.numpy(),
            "dLdW": self.layer1.weight.grad.numpy(),
        }
        return grads
class TorchSDPAttentionLayer(nn.Module):
    # 定义一个基于PyTorch的自注意力层
    def __init__(self):
        super(TorchSDPAttentionLayer, self).__init__()

    def forward(self, Q, K, V, mask=None):
        # 将输入的查询、键、值保存到对象中
        self.Q = Q
        self.K = K
        self.V = V

        # 如果查询、键、值不是PyTorch张量，则转换为张量
        if not isinstance(self.Q, torch.Tensor):
            self.Q = torchify(self.Q)
        if not isinstance(self.K, torch.Tensor):
            self.K = torchify(self.K)
        if not isinstance(self.V, torch.Tensor):
            self.V = torchify(self.V)

        # 保留查询、键、值的梯度信息
        self.Q.retain_grad()
        self.K.retain_grad()
        self.V.retain_grad()

        # 获取键值对应的维度
        self.d_k = self.Q.size(-1)
        # 计算注意力分数
        self.scores = torch.matmul(self.Q, self.K.transpose(-2, -1)) / np.sqrt(self.d_k)
        # 如果存在掩码，则将分数中对应位置的值替换为负无穷
        if mask is not None:
            self.scores = self.scores.masked_fill(mask == 0, -1e9)
        self.scores.retain_grad()

        # 计算注意力权重
        self.weights = F.softmax(self.scores, dim=-1)
        self.weights.retain_grad()
        # 计算加权后的值
        self.Y = torch.matmul(self.weights, self.V)
        self.Y.retain_grad()
        # 返回加权后的值和注意力权重
        return self.Y, self.weights

    def extract_grads(self, Q, K, V, mask=None):
        # 调用前向传播计算梯度
        self.forward(Q, K, V, mask=mask)
        # 计算损失值
        self.loss1 = self.Y.sum()
        # 反向传播计算梯度
        self.loss1.backward()
        # 提取并返回各个参数的梯度信息
        grads = {
            "Q": self.Q.detach().numpy(),
            "K": self.K.detach().numpy(),
            "V": self.V.detach().numpy(),
            "d_k": self.d_k,
            "scores": self.scores.detach().numpy(),
            "weights": self.weights.detach().numpy(),
            "Y": self.Y.detach().numpy(),
            "dLdV": self.V.grad.numpy(),
            "dWeights": self.weights.grad.numpy(),
            "dScores": self.scores.grad.numpy(),
            "dLdQ": self.Q.grad.numpy(),
            "dLdK": self.K.grad.numpy(),
        }
        return grads


class TorchMultiHeadedAttentionModule(nn.Module):
    # 初始化多头注意力模块，接受参数和超参数
    def __init__(self, params, hparams):
        # 调用父类的初始化方法
        super(TorchMultiHeadedAttentionModule, self).__init__()
        # 确保每个头的维度能够整除总维度
        assert hparams["kqv_dim"] % hparams["n_heads"] == 0
        # 设置头的数量
        self.n_heads = hparams["n_heads"]
        # 计算每个头的潜在维度
        self.latent_dim = hparams["kqv_dim"] // hparams["n_heads"]
        # 设置丢弃概率
        self.p_dropout = hparams["dropout_p"]
        # 初始化投影矩阵
        self.projections = {
            "Q": nn.Linear(hparams["kqv_dim"], hparams["kqv_dim"]),
            "K": nn.Linear(hparams["kqv_dim"], hparams["kqv_dim"]),
            "V": nn.Linear(hparams["kqv_dim"], hparams["kqv_dim"]),
            "O": nn.Linear(hparams["kqv_dim"], hparams["kqv_dim"]),
        }
        # 设置投影矩阵的权重和偏置
        self.projections["Q"].weight = nn.Parameter(
            torch.FloatTensor(params["components"]["Q"]["W"].T)
        )
        self.projections["Q"].bias = nn.Parameter(
            torch.FloatTensor(params["components"]["Q"]["b"])
        )
        self.projections["K"].weight = nn.Parameter(
            torch.FloatTensor(params["components"]["K"]["W"].T)
        )
        self.projections["K"].bias = nn.Parameter(
            torch.FloatTensor(params["components"]["K"]["b"])
        )
        self.projections["V"].weight = nn.Parameter(
            torch.FloatTensor(params["components"]["V"]["W"].T)
        )
        self.projections["V"].bias = nn.Parameter(
            torch.FloatTensor(params["components"]["V"]["b"])
        )
        self.projections["O"].weight = nn.Parameter(
            torch.FloatTensor(params["components"]["O"]["W"].T)
        )
        self.projections["O"].bias = nn.Parameter(
            torch.FloatTensor(params["components"]["O"]["b"])
        )

        # 初始化注意力和丢弃层
        self.attn = None
        self.dropout = nn.Dropout(p=hparams["dropout_p"])
    # 定义前向传播函数，接收查询(Q)、键(K)、值(V)和掩码(mask)作为输入
    def forward(self, Q, K, V, mask=None):
        # 将输入的查询(Q)、键(K)、值(V)保存到当前对象中
        self.Q = Q
        self.K = K
        self.V = V

        # 如果查询(Q)不是torch.Tensor类型，则将其转换为torch.Tensor类型
        if not isinstance(self.Q, torch.Tensor):
            self.Q = torchify(self.Q)
        # 如果键(K)不是torch.Tensor类型，则将其转换为torch.Tensor类型
        if not isinstance(self.K, torch.Tensor):
            self.K = torchify(self.K)
        # 如果值(V)不是torch.Tensor类型，则将其转换为torch.Tensor类型
        if not isinstance(self.V, torch.Tensor):
            self.V = torchify(self.V)

        # 保留查询(Q)、键(K)、值(V)的梯度信息
        self.Q.retain_grad()
        self.K.retain_grad()
        self.V.retain_grad()

        # 如果存在掩码(mask)，则将其扩展维度
        if mask is not None:
            mask = mask.unsqueeze(1)
        # 获取输入查询(Q)的样本数量
        n_ex = self.Q.size(0)

        # 对查询(Q)、键(K)、值(V)进行线性变换并重塑维度，然后转置
        self.Q_proj = (
            self.projections["Q"](self.Q)
            .view(n_ex, -1, self.n_heads, self.latent_dim)
            .transpose(1, 2)
        )

        self.K_proj = (
            self.projections["K"](self.K)
            .view(n_ex, -1, self.n_heads, self.latent_dim)
            .transpose(1, 2)
        )

        self.V_proj = (
            self.projections["V"](self.V)
            .view(n_ex, -1, self.n_heads, self.latent_dim)
            .transpose(1, 2)
        )

        # 保留查询(Q)、键(K)、值(V)的梯度信息
        self.Q_proj.retain_grad()
        self.K_proj.retain_grad()
        self.V_proj.retain_grad()

        # 2) 在批处理中对所有投影向量应用注意力机制
        self.attn_out, self.attn = TorchSDPAttentionLayer().forward(
            self.Q_proj, self.K_proj, self.V_proj, mask=mask
        )
        # 保留注意力权重和输出的梯度信息
        self.attn.retain_grad()
        self.attn_out.retain_grad()

        # 3) 使用视图(view)进行“连接”并应用最终的线性变换
        self.attn_out_reshaped = (
            self.attn_out.transpose(1, 2)
            .contiguous()
            .view(n_ex, -1, self.n_heads * self.latent_dim)
        )
        # 保留连接后的输出的梯度信息
        self.attn_out_reshaped.retain_grad()
        print(self.attn_out_reshaped.shape)
        # 对连接后的输出应用最终的线性变换
        self.Y = self.projections["O"](self.attn_out_reshaped)
        print(self.Y.shape)
        # 保留最终输出的梯度信息
        self.Y.retain_grad()
# 定义全局变量_params和_param_aliases，用于存储参数和参数别名
_params = {}
_param_aliases = {}

# 定义param函数，用于创建共享参数变量
def param(name, *args, **kwargs):
    """
    A wrapper for `tf.Variable` which enables parameter sharing in models.

    Creates and returns theano shared variables similarly to `tf.Variable`,
    except if you try to create a param with the same name as a
    previously-created one, `param(...)` will just return the old one instead of
    making a new one.

    This constructor also adds a `param` attribute to the shared variables it
    creates, so that you can easily search a graph for all params.
    """

    # 如果参数名不在_params中，则创建新的参数并添加到_params中
    if name not in _params:
        kwargs["name"] = name
        param = tf.Variable(*args, **kwargs)
        param.param = True
        _params[name] = param
    # 如果参数名已存在于_params中，则直接返回已存在的参数
    result = _params[name]
    i = 0
    # 处理参数别名
    while result in _param_aliases:
        i += 1
        result = _param_aliases[result]
    return result

# 根据参数名查找所有包含该名称的参数
def params_with_name(name):
    return [p for n, p in _params.items() if name in n]

# 定义ReLULayer函数，实现ReLU激活函数的全连接层
def ReLULayer(name, n_in, n_out, inputs, w_initialization):
    if isinstance(w_initialization, np.ndarray):
        weight_values = w_initialization.astype("float32")

    # 创建权重参数W，并进行矩阵乘法运算
    W = param(name + ".W", weight_values)
    result = tf.matmul(inputs, W)
    # 添加偏置并进行ReLU激活
    output = tf.nn.bias_add(
        result, param(name + ".b", np.zeros((n_out,), dtype="float32"))
    )
    output = tf.nn.relu(output)
    return output, W

# 定义LinearLayer函数，实现线性全连接层
def LinearLayer(name, n_in, n_out, inputs, w_initialization):
    if isinstance(w_initialization, np.ndarray):
        weight_values = w_initialization.astype("float32")

    # 创建权重参数W，并进行矩阵乘法运算
    W = param(name + ".W", weight_values)
    result = tf.matmul(inputs, W)
    # 添加偏置
    output = tf.nn.bias_add(
        result, param(name + ".b", np.zeros((n_out,), dtype="float32"))
    )
    # 返回 output 和 W 两个变量
    return output, W
# 生成器函数，用于生成数据
def Generator(n_samples, X_real, params=None):
    # 设置特征数为2
    n_feats = 2
    # 初始化权重矩阵
    W1 = W2 = W3 = W4 = "he"
    # 生成噪声数据
    noise = tf.random.normal([n_samples, 2])
    # 如果参数不为空，则使用参数中的值
    if params is not None:
        # 转换噪声数据为张量
        noise = tf.convert_to_tensor(params["noise"], dtype="float32")
        # 获取生成器的权重矩阵
        W1 = params["generator"]["FC1"]["W"]
        W2 = params["generator"]["FC2"]["W"]
        W3 = params["generator"]["FC3"]["W"]
        W4 = params["generator"]["FC4"]["W"]
        # 获取隐藏层维度和输入特征数
        DIM = params["g_hidden"]
        n_feats = params["n_in"]

    # 初始化输出字典和权重字典
    outs = {}
    weights = {}
    # 第一层全连接层
    output, W = ReLULayer("Generator.1", n_feats, DIM, noise, w_initialization=W1)
    outs["FC1"] = output
    weights["FC1"] = W
    # 第二层全连接层
    output, W = ReLULayer("Generator.2", DIM, DIM, output, w_initialization=W2)
    outs["FC2"] = output
    weights["FC2"] = W
    # 第三层全连接层
    output, W = ReLULayer("Generator.3", DIM, DIM, output, w_initialization=W3)
    outs["FC3"] = output
    weights["FC3"] = W
    # 第四层全连接层
    output, W = LinearLayer("Generator.4", DIM, n_feats, output, w_initialization=W4)
    outs["FC4"] = output
    weights["FC4"] = W
    # 返回输出、输出字典和权重字典
    return output, outs, weights

# 判别器函数，用于判别数据真伪
def Discriminator(inputs, params=None):
    # 设置特征数为2
    n_feats = 2
    # 初始化权重矩阵
    W1 = W2 = W3 = W4 = "he"
    # 如果参数不为空，则使用参数中的值
    if params is not None:
        # 获取判别器的权重矩阵
        W1 = params["critic"]["FC1"]["W"]
        W2 = params["critic"]["FC2"]["W"]
        W3 = params["critic"]["FC3"]["W"]
        W4 = params["critic"]["FC4"]["W"]
        # 获取隐藏层维度和输入特征数
        DIM = params["g_hidden"]
        n_feats = params["n_in"]

    # 初始化输出字典和权重字典
    outs = {}
    weights = {}
    # 第一层全连接层
    output, W = ReLULayer("Discriminator.1", n_feats, DIM, inputs, w_initialization=W1)
    outs["FC1"] = output
    weights["FC1"] = W
    # 第二层全连接层
    output, W = ReLULayer("Discriminator.2", DIM, DIM, output, w_initialization=W2)
    outs["FC2"] = output
    weights["FC2"] = W
    # 第三层全连接层
    output, W = ReLULayer("Discriminator.3", DIM, DIM, output, w_initialization=W3)
    outs["FC3"] = output
    weights["FC3"] = W
    # 第四层全连接层
    output, W = LinearLayer("Discriminator.4", DIM, 1, output, w_initialization=W4)
    outs["FC4"] = output
    weights["FC4"] = W
    # 获取偏置项
    # 遍历参数列表中包含名称为"Discriminator"的参数
    for var in params_with_name("Discriminator"):
        # 如果参数名称中包含"1.b:"，将该参数存入权重字典中的"FC1_b"键
        if "1.b:" in var.name:
            weights["FC1_b"] = var
        # 如果参数名称中包含"2.b:"，将该参数存入权重字典中的"FC2_b"键
        elif "2.b:" in var.name:
            weights["FC2_b"] = var
        # 如果参数名称中包含"3.b:"，将该参数存入权重字典中的"FC3_b"键
        elif "3.b:" in var.name:
            weights["FC3_b"] = var
        # 如果参数名称中包含"4.b:"，将该参数存入权重字典中的"FC4_b"键
        elif "4.b:" in var.name:
            weights["FC4_b"] = var

    # 将输出结果进行重塑，将其形状变为一维数组
    return tf.reshape(output, [-1]), outs, weights
# 定义 WGAN-GP 模型的 TensorFlow 函数
def WGAN_GP_tf(X, lambda_, params, batch_size):
    # 禁用即时执行模式
    tf.compat.v1.disable_eager_execution()

    # 获取输入数据的批量大小
    batch_size = X.shape[0]

    # 获取超参数
    n_steps = params["n_steps"]
    c_updates_per_epoch = params["c_updates_per_epoch"]
    alpha = tf.convert_to_tensor(params["alpha"], dtype="float32")

    # 定义真实数据的占位符
    X_real = tf.compat.v1.placeholder(tf.float32, shape=[None, params["n_in"]])

    # 生成器生成假数据，获取生成器输出和权重
    X_fake, G_out_X_fake, G_weights = Generator(batch_size, X_real, params)

    # 判别器对真实数据进行判别，获取判别器输出和权重
    Y_real, C_out_Y_real, C_Y_real_weights = Discriminator(X_real, params)
    # 判别器对假数据进行判别，获取判别器输出和权重
    Y_fake, C_out_Y_fake, C_Y_fake_weights = Discriminator(X_fake, params)

    # 计算 WGAN 损失
    mean_fake = tf.reduce_mean(Y_fake)
    mean_real = tf.reduce_mean(Y_real)
    C_loss = tf.reduce_mean(Y_fake) - tf.reduce_mean(Y_real)
    G_loss = -tf.reduce_mean(Y_fake)

    # 计算 WGAN 梯度惩罚
    X_interp = alpha * X_real + ((1 - alpha) * X_fake)
    Y_interp, C_out_Y_interp, C_Y_interp_weights = Discriminator(X_interp, params)
    gradInterp = tf.gradients(Y_interp, [X_interp])[0]
    norm_gradInterp = tf.sqrt(
        tf.compat.v1.reduce_sum(tf.square(gradInterp), reduction_indices=[1])
    )
    gradient_penalty = tf.reduce_mean((norm_gradInterp - 1) ** 2)
    C_loss += lambda_ * gradient_penalty

    # 提取判别器对插值数据的梯度
    C_bwd_Y_interp = {}
    for k, v in C_out_Y_interp.items():
        C_bwd_Y_interp[k] = tf.gradients(Y_interp, [v])[0]

    # 提取判别器权重的梯度
    C_bwd_W = {}
    for k, v in C_Y_interp_weights.items():
        C_bwd_W[k] = tf.gradients(C_loss, [v])[0]

    # 获取梯度
    dC_Y_fake = tf.gradients(C_loss, [Y_fake])[0]
    dC_Y_real = tf.gradients(C_loss, [Y_real])[0]
    dC_gradInterp = tf.gradients(C_loss, [gradInterp])[0]
    dG_Y_fake = tf.gradients(G_loss, [Y_fake])[0]

    # 返回梯度
    return grads


# 定义 TensorFlow 的负采样交叉熵损失函数
def TFNCELoss(X, target_word, L):
    from tensorflow.python.ops.nn_impl import _compute_sampled_logits
    from tensorflow.python.ops.nn_impl import sigmoid_cross_entropy_with_logits
    # 禁用 TensorFlow 2.x 中的即时执行模式
    tf.compat.v1.disable_eager_execution()
    
    # 创建占位符，用于接收输入数据
    in_embed = tf.compat.v1.placeholder(tf.float32, shape=X.shape)
    in_bias = tf.compat.v1.placeholder(tf.float32, shape=L.parameters["b"].flatten().shape)
    in_weights = tf.compat.v1.placeholder(tf.float32, shape=L.parameters["W"].shape)
    in_target_word = tf.compat.v1.placeholder(tf.int64)
    in_neg_samples = tf.compat.v1.placeholder(tf.int32)
    in_target_prob = tf.compat.v1.placeholder(tf.float32)
    in_neg_samp_prob = tf.compat.v1.placeholder(tf.float32)
    
    # 创建 feed 字典，将输入数据传入对应的占位符
    feed = {
        in_embed: X,
        in_weights: L.parameters["W"],
        in_target_word: target_word,
        in_bias: L.parameters["b"].flatten(),
        in_neg_samples: L.derived_variables["noise_samples"][0],
        in_target_prob: L.derived_variables["noise_samples"][1],
        in_neg_samp_prob: L.derived_variables["noise_samples"][2],
    }
    
    # 使用负采样计算 NCE 损失
    nce_unreduced = tf.nn.nce_loss(
        weights=in_weights,
        biases=in_bias,
        labels=in_target_word,
        inputs=in_embed,
        sampled_values=(in_neg_samples, in_target_prob, in_neg_samp_prob),
        num_sampled=L.num_negative_samples,
        num_classes=L.n_classes,
    )
    
    # 计算总损失
    loss = tf.reduce_sum(nce_unreduced)
    # 计算损失对权重的梯度
    dLdW = tf.gradients(loss, [in_weights])[0]
    # 计算损失对偏置的梯度
    dLdb = tf.gradients(loss, [in_bias])[0]
    # 计算损失对输入数据的梯度
    dLdX = tf.gradients(loss, [in_embed])[0]
    # 计算采样后的logits和labels
    sampled_logits, sampled_labels = _compute_sampled_logits(
        weights=in_weights,  # 输入权重
        biases=in_bias,  # 输入偏置
        labels=in_target_word,  # 目标词标签
        inputs=in_embed,  # 输入嵌入
        sampled_values=(in_neg_samples, in_target_prob, in_neg_samp_prob),  # 采样值
        num_sampled=L.num_negative_samples,  # 负采样数量
        num_classes=L.n_classes,  # 类别数量
        num_true=1,  # 真实样本数量
        subtract_log_q=True,  # 是否减去log(q)
    )
    
    # 计算采样后的损失
    sampled_losses = sigmoid_cross_entropy_with_logits(
        labels=sampled_labels,  # 采样标签
        logits=sampled_logits  # 采样logits
    )
    
    # 创建一个会话
    with tf.compat.v1.Session() as session:
        # 初始化全局变量
        session.run(tf.compat.v1.global_variables_initializer())
        # 运行会话，获取损失和相关变量
        (
            _final_loss,
            _nce_unreduced,
            _dLdW,
            _dLdb,
            _dLdX,
            _sampled_logits,
            _sampled_labels,
            _sampled_losses,
        ) = session.run(
            [
                loss,
                nce_unreduced,
                dLdW,
                dLdb,
                dLdX,
                sampled_logits,
                sampled_labels,
                sampled_losses,
            ],
            feed_dict=feed,  # 喂入数据
        )
    
    # 重置默认图
    tf.compat.v1.reset_default_graph()
    
    # 返回结果字典
    return {
        "final_loss": _final_loss,  # 最终损失
        "nce_unreduced": _nce_unreduced,  # 未减少的nce
        "dLdW": _dLdW,  # dL/dW
        "dLdb": _dLdb,  # dL/db
        "dLdX": _dLdX,  # dL/dX
        "out_logits": _sampled_logits,  # 输出logits
        "out_labels": _sampled_labels,  # 输出标签
        "sampled_loss": _sampled_losses,  # 采样损失
    }
```