# `Bert-VITS2\attentions.py`

```

# 导入 math、torch、nn、functional 模块
import math
import torch
from torch import nn
from torch.nn import functional as F

# 导入自定义的 commons 模块和 logging 模块
import commons
import logging

# 获取 logger 对象
logger = logging.getLogger(__name__)

# 定义 LayerNorm 类，继承自 nn.Module
class LayerNorm(nn.Module):
    # 初始化函数，接受 channels 和 eps 两个参数
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        # 初始化 gamma 和 beta 为可学习参数
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    # 前向传播函数，接受输入 x
    def forward(self, x):
        # 转置输入 x
        x = x.transpose(1, -1)
        # 对输入 x 进行 LayerNorm 操作
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        # 再次转置输入 x
        return x.transpose(1, -1)

# 定义 Encoder 类，继承自 nn.Module
class Encoder(nn.Module):
    # 初始化函数，接受多个参数
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
        # 初始化各个参数
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.window_size = window_size
        self.cond_layer_idx = self.n_layers
        # 如果传入了特定参数，则进行相应的初始化操作
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
        # 初始化 Dropout 层
        self.drop = nn.Dropout(p_dropout)
        # 初始化多个注意力层、LayerNorm 层和 FFN 层
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

    # 前向传播函数，接受输入 x、x_mask 和 g
    def forward(self, x, x_mask, g=None):
        # 计算注意力掩码
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        x = x * x_mask
        # 循环进行多层 Encoder 操作
        for i in range(self.n_layers):
            if i == self.cond_layer_idx and g is not None:
                g = self.spk_emb_linear(g.transpose(1, 2))
                g = g.transpose(1, 2)
                x = x + g
                x = x * x_mask
            y = self.attn_layers[i](x, x, attn_mask)
            y = self.drop(y)
            x = self.norm_layers_1[i](x + y)

            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)
            x = self.norm_layers_2[i](x + y)
        x = x * x_mask
        return x

# 定义 Decoder 类，继承自 nn.Module
class Decoder(nn.Module):
    # 初始化函数，接受多个参数
    def __init__(
        self,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size=1,
        p_dropout=0.0,
        proximal_bias=False,
        proximal_init=True,
        **kwargs
    ):
        super().__init__()
        # 初始化各个参数
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.proximal_bias = proximal_bias
        self.proximal_init = proximal_init
        # 初始化 Dropout 层
        self.drop = nn.Dropout(p_dropout)
        # 初始化多个注意力层、LayerNorm 层和 FFN 层
        self.self_attn_layers = nn.ModuleList()
        self.norm_layers_0 = nn.ModuleList()
        self.encdec_attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()
        for i in range(self.n_layers):
            self.self_attn_layers.append(
                MultiHeadAttention(
                    hidden_channels,
                    hidden_channels,
                    n_heads,
                    p_dropout=p_dropout,
                    proximal_bias=proximal_bias,
                    proximal_init=proximal_init,
                )
            )
            self.norm_layers_0.append(LayerNorm(hidden_channels))
            self.encdec_attn_layers.append(
                MultiHeadAttention(
                    hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout
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
                    causal=True,
                )
            )
            self.norm_layers_2.append(LayerNorm(hidden_channels))

    # 前向传播函数，接受输入 x、x_mask、h 和 h_mask
    def forward(self, x, x_mask, h, h_mask):
        """
        x: decoder input
        h: encoder output
        """
        # 计算自注意力掩码和编码-解码注意力掩码
        self_attn_mask = commons.subsequent_mask(x_mask.size(2)).to(
            device=x.device, dtype=x.dtype
        )
        encdec_attn_mask = h_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        x = x * x_mask
        # 循环进行多层 Decoder 操作
        for i in range(self.n_layers):
            y = self.self_attn_layers[i](x, x, self_attn_mask)
            y = self.drop(y)
            x = self.norm_layers_0[i](x + y)

            y = self.encdec_attn_layers[i](x, h, encdec_attn_mask)
            y = self.drop(y)
            x = self.norm_layers_1[i](x + y)

            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)
            x = self.norm_layers_2[i](x + y)
        x = x * x_mask
        return x

# 定义 FFN 类，继承自 nn.Module
class FFN(nn.Module):
    # 初始化函数，接受多个参数
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
        # 初始化各个参数
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.activation = activation
        self.causal = causal
        # 根据是否是因果卷积选择不同的填充方式
        if causal:
            self.padding = self._causal_padding
        else:
            self.padding = self._same_padding
        # 初始化卷积层和 Dropout 层
        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size)
        self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size)
        self.drop = nn.Dropout(p_dropout)

    # 前向传播函数，接受输入 x 和 x_mask
    def forward(self, x, x_mask):
        x = self.conv_1(self.padding(x * x_mask))
        if self.activation == "gelu":
            x = x * torch.sigmoid(1.702 * x)
        else:
            x = torch.relu(x)
        x = self.drop(x)
        x = self.conv_2(self.padding(x * x_mask))
        return x * x_mask

    # 定义因果卷积的填充方式
    def _causal_padding(self, x):
        if self.kernel_size == 1:
            return x
        pad_l = self.kernel_size - 1
        pad_r = 0
        padding = [[0, 0], [0, 0], [pad_l, pad_r]]
        x = F.pad(x, commons.convert_pad_shape(padding))
        return x

    # 定义普通卷积的填充方式
    def _same_padding(self, x):
        if self.kernel_size == 1:
            return x
        pad_l = (self.kernel_size - 1) // 2
        pad_r = self.kernel_size // 2
        padding = [[0, 0], [0, 0], [pad_l, pad_r]]
        x = F.pad(x, commons.convert_pad_shape(padding))
        return x

```