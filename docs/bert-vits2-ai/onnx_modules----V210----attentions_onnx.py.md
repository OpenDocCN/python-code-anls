# `Bert-VITS2\onnx_modules\V210\attentions_onnx.py`

```

# 导入数学库和PyTorch库
import math
import torch
from torch import nn
from torch.nn import functional as F

# 导入自定义的commons模块和logging模块
import commons
import logging

# 获取logger对象
logger = logging.getLogger(__name__)

# 定义LayerNorm类，继承自nn.Module
class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps

        # 初始化gamma和beta参数
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    # 前向传播函数
    def forward(self, x):
        # 转置操作
        x = x.transpose(1, -1)
        # 使用F.layer_norm进行LayerNorm操作
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        # 再次进行转置操作
        return x.transpose(1, -1)

# 定义Encoder类，继承自nn.Module
class Encoder(nn.Module):
    def __init__(
        self,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size=1,
        p_dropout=0.0,
        window_size=4,
        isflow=True,
        **kwargs
    ):
        super().__init__()
        # 初始化各种参数
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.window_size = window_size
        self.cond_layer_idx = self.n_layers
        # 如果kwargs中包含"gin_channels"，则进行相应的初始化操作
        if "gin_channels" in kwargs:
            self.gin_channels = kwargs["gin_channels"]
            if self.gin_channels != 0:
                self.spk_emb_linear = nn.Linear(self.gin_channels, self.hidden_channels)
                self.cond_layer_idx = (
                    kwargs["cond_layer_idx"] if "cond_layer_idx" in kwargs else 2
                )
                logging.debug(self.gin_channels, self.cond_layer_idx)
                assert (
                    self.cond_layer_idx < self.n_layers
                ), "cond_layer_idx should be less than n_layers"
        # 初始化Dropout层
        self.drop = nn.Dropout(p_dropout)
        # 初始化多头注意力层、LayerNorm层和前馈神经网络层
        self.attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()
        for i in range(self.n_layers):
            self.attn_layers.append(
                MultiHeadAttention(
                    hidden_channels,
                    hidden_channels,
                    n_heads,
                    p_dropout=p_dropout,
                    window_size=window_size,
                )
            )
            self.norm_layers_1.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(
                FFN(
                    hidden_channels,
                    hidden_channels,
                    filter_channels,
                    kernel_size,
                    p_dropout=p_dropout,
                )
            )
            self.norm_layers_2.append(LayerNorm(hidden_channels))

    # 前向传播函数
    def forward(self, x, x_mask, g=None):
        # 生成注意力掩码
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        x = x * x_mask
        for i in range(self.n_layers):
            # 如果当前层是条件层并且g不为None，则进行相应的操作
            if i == self.cond_layer_idx and g is not None:
                g = self.spk_emb_linear(g.transpose(1, 2))
                g = g.transpose(1, 2)
                x = x + g
                x = x * x_mask
            # 多头注意力层的前向传播
            y = self.attn_layers[i](x, x, attn_mask)
            y = self.drop(y)
            x = self.norm_layers_1[i](x + y)

            # 前馈神经网络层的前向传播
            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)
            x = self.norm_layers_2[i](x + y)
        x = x * x_mask
        return x

# 定义FFN类，继承自nn.Module
class FFN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        filter_channels,
        kernel_size,
        p_dropout=0.0,
        activation=None,
        causal=False,
    ):
        super().__init__()
        # 初始化各种参数
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.activation = activation
        self.causal = causal

        # 根据是否是因果卷积选择不同的padding函数
        if causal:
            self.padding = self._causal_padding
        else:
            self.padding = self._same_padding

        # 初始化卷积层和Dropout层
        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size)
        self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size)
        self.drop = nn.Dropout(p_dropout)

    # 前向传播函数
    def forward(self, x, x_mask):
        # 第一个卷积层
        x = self.conv_1(self.padding(x * x_mask))
        # 激活函数
        if self.activation == "gelu":
            x = x * torch.sigmoid(1.702 * x)
        else:
            x = torch.relu(x)
        x = self.drop(x)
        # 第二个卷积层
        x = self.conv_2(self.padding(x * x_mask))
        return x * x_mask

    # 因果卷积的padding函数
    def _causal_padding(self, x):
        if self.kernel_size == 1:
            return x
        pad_l = self.kernel_size - 1
        pad_r = 0
        padding = [[0, 0], [0, 0], [pad_l, pad_r]]
        x = F.pad(x, commons.convert_pad_shape(padding))
        return x

    # 非因果卷积的padding函数
    def _same_padding(self, x):
        if self.kernel_size == 1:
            return x
        pad_l = (self.kernel_size - 1) // 2
        pad_r = self.kernel_size // 2
        padding = [[0, 0], [0, 0], [pad_l, pad_r]]
        x = F.pad(x, commons.convert_pad_shape(padding))
        return x

```