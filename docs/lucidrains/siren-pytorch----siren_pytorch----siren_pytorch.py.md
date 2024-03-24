# `.\lucidrains\siren-pytorch\siren_pytorch\siren_pytorch.py`

```py
# 导入数学库和PyTorch库
import math
import torch
# 从torch库中导入神经网络模块
from torch import nn
# 从torch.nn.functional中导入函数F
import torch.nn.functional as F
# 从einops库中导入rearrange函数
from einops import rearrange

# 辅助函数

# 判断值是否存在的函数
def exists(val):
    return val is not None

# 将值转换为元组的函数
def cast_tuple(val, repeat = 1):
    return val if isinstance(val, tuple) else ((val,) * repeat)

# 正弦激活函数

class Sine(nn.Module):
    def __init__(self, w0 = 1.):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)

# Siren层

class Siren(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        w0 = 1.,
        c = 6.,
        is_first = False,
        use_bias = True,
        activation = None,
        dropout = 0.
    ):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first

        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out) if use_bias else None
        self.init_(weight, bias, c = c, w0 = w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None
        self.activation = Sine(w0) if activation is None else activation
        self.dropout = nn.Dropout(dropout)

    def init_(self, weight, bias, c, w0):
        dim = self.dim_in

        w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
        weight.uniform_(-w_std, w_std)

        if exists(bias):
            bias.uniform_(-w_std, w_std)

    def forward(self, x):
        out =  F.linear(x, self.weight, self.bias)
        out = self.activation(out)
        out = self.dropout(out)
        return out

# Siren网络

class SirenNet(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        num_layers,
        w0 = 1.,
        w0_initial = 30.,
        use_bias = True,
        final_activation = None,
        dropout = 0.
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dim_hidden = dim_hidden

        self.layers = nn.ModuleList([])
        for ind in range(num_layers):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden

            layer = Siren(
                dim_in = layer_dim_in,
                dim_out = dim_hidden,
                w0 = layer_w0,
                use_bias = use_bias,
                is_first = is_first,
                dropout = dropout
            )

            self.layers.append(layer)

        final_activation = nn.Identity() if not exists(final_activation) else final_activation
        self.last_layer = Siren(dim_in = dim_hidden, dim_out = dim_out, w0 = w0, use_bias = use_bias, activation = final_activation)

    def forward(self, x, mods = None):
        mods = cast_tuple(mods, self.num_layers)

        for layer, mod in zip(self.layers, mods):
            x = layer(x)

            if exists(mod):
                x *= rearrange(mod, 'd -> () d')

        return self.last_layer(x)

# 调制前馈

class Modulator(nn.Module):
    def __init__(self, dim_in, dim_hidden, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([])

        for ind in range(num_layers):
            is_first = ind == 0
            dim = dim_in if is_first else (dim_hidden + dim_in)

            self.layers.append(nn.Sequential(
                nn.Linear(dim, dim_hidden),
                nn.ReLU()
            ))

    def forward(self, z):
        x = z
        hiddens = []

        for layer in self.layers:
            x = layer(x)
            hiddens.append(x)
            x = torch.cat((x, z))

        return tuple(hiddens)

# 包装器

class SirenWrapper(nn.Module):
    # 初始化函数，接受神经网络、图像宽度、图像高度和潜在维度作为参数
    def __init__(self, net, image_width, image_height, latent_dim = None):
        # 调用父类的初始化函数
        super().__init__()
        # 断言网络类型为 SirenNet
        assert isinstance(net, SirenNet), 'SirenWrapper must receive a Siren network'

        # 初始化网络、图像宽度和图像高度
        self.net = net
        self.image_width = image_width
        self.image_height = image_height

        # 初始化调制器为 None，如果传入了潜在维度，则创建 Modulator 对象
        self.modulator = None
        if exists(latent_dim):
            self.modulator = Modulator(
                dim_in = latent_dim,
                dim_hidden = net.dim_hidden,
                num_layers = net.num_layers
            )

        # 创建坐标张量
        tensors = [torch.linspace(-1, 1, steps = image_height), torch.linspace(-1, 1, steps = image_width)]
        mgrid = torch.stack(torch.meshgrid(*tensors, indexing = 'ij'), dim=-1)
        mgrid = rearrange(mgrid, 'h w c -> (h w) c')
        # 将坐标张量注册为缓冲区
        self.register_buffer('grid', mgrid)

    # 前向传播函数，接受图像或潜在向量作为参数
    def forward(self, img = None, *, latent = None):
        # 判断是否需要调制
        modulate = exists(self.modulator)
        # 断言只有在初始化时传入了潜在向量才能提供潜在向量
        assert not (modulate ^ exists(latent)), 'latent vector must be only supplied if `latent_dim` was passed in on instantiation'

        # 如果需要调制，则计算调制结果
        mods = self.modulator(latent) if modulate else None

        # 复制坐标张量并设置为需要梯度
        coords = self.grid.clone().detach().requires_grad_()
        # 将坐标张量输入网络得到输出
        out = self.net(coords, mods)
        out = rearrange(out, '(h w) c -> () c h w', h = self.image_height, w = self.image_width)

        # 如果提供了图像，则计算均方误差损失
        if exists(img):
            return F.mse_loss(img, out)

        # 返回输出结果
        return out
```