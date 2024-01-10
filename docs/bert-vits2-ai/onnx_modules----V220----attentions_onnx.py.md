# `Bert-VITS2\onnx_modules\V220\attentions_onnx.py`

```
# 导入 math 模块
import math
# 导入 torch 模块
import torch
# 从 torch 模块中导入 nn 模块
from torch import nn
# 从 torch.nn 模块中导入 functional 模块，并重命名为 F
from torch.nn import functional as F

# 导入自定义的 commons 模块
import commons
# 导入 logging 模块
import logging

# 获取当前模块的 logger 对象
logger = logging.getLogger(__name__)


# 定义 LayerNorm 类，继承自 nn.Module 类
class LayerNorm(nn.Module):
    # 初始化方法，接受 channels 和 eps 两个参数
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        # 将 channels 和 eps 分别赋值给 self.channels 和 self.eps
        self.channels = channels
        self.eps = eps

        # 创建可学习的参数 gamma 和 beta
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    # 前向传播方法，接受输入 x
    def forward(self, x):
        # 将输入 x 进行维度转置
        x = x.transpose(1, -1)
        # 对输入 x 进行 Layer Normalization 操作
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        # 再次将结果进行维度转置，并返回
        return x.transpose(1, -1)


# 使用 torch.jit.script 装饰器，定义融合的 add、tanh、sigmoid 和 multiply 操作
@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    # 将 n_channels 转换为整数
    n_channels_int = n_channels[0]
    # 执行 add 操作
    in_act = input_a + input_b
    # 执行 tanh 操作
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    # 执行 sigmoid 操作
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    # 执行 multiply 操作
    acts = t_act * s_act
    # 返回结果
    return acts


# 定义 Encoder 类，继承自 nn.Module 类
class Encoder(nn.Module):
    # 初始化方法，接受多个参数
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
        # 省略部分初始化代码

    # 前向传播方法，接受输入 x, x_mask, g
    def forward(self, x, x_mask, g=None):
        # 创建注意力掩码
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        # 对输入 x 进行掩码处理
        x = x * x_mask
        # 循环执行多层编码器操作
        for i in range(self.n_layers):
            # 如果���前层是条件层且 g 不为空
            if i == self.cond_layer_idx and g is not None:
                # 对 g 进行线性变换
                g = self.spk_emb_linear(g.transpose(1, 2))
                g = g.transpose(1, 2)
                # 将 g 加到 x 上
                x = x + g
                # 再次对 x 进行掩码处理
                x = x * x_mask
            # 执行注意力层操作
            y = self.attn_layers[i](x, x, attn_mask)
            y = self.drop(y)
            x = self.norm_layers_1[i](x + y)

            # 执行前馈神经网络层操作
            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)
            x = self.norm_layers_2[i](x + y)
        # 再次对 x 进行掩码处理
        x = x * x_mask
        # 返回结果
        return x


# 定义 MultiHeadAttention 类，继承自 nn.Module 类
class MultiHeadAttention(nn.Module):
    # 省略部分代码
    # 初始化函数，定义了模型的参数和层
    def __init__(
        self,
        channels,
        out_channels,
        n_heads,
        p_dropout=0.0,
        window_size=None,
        heads_share=True,
        block_length=None,
        proximal_bias=False,
        proximal_init=False,
    ):
        # 调用父类的初始化函数
        super().__init__()
        # 断言通道数能够被注意力头数整除
        assert channels % n_heads == 0

        # 初始化模型的参数
        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.p_dropout = p_dropout
        self.window_size = window_size
        self.heads_share = heads_share
        self.block_length = block_length
        self.proximal_bias = proximal_bias
        self.proximal_init = proximal_init
        self.attn = None

        # 计算每个注意力头的通道数
        self.k_channels = channels // n_heads
        # 定义卷积层，用于计算查询、键、值和输出
        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        self.conv_o = nn.Conv1d(channels, out_channels, 1)
        # 定义 dropout 层
        self.drop = nn.Dropout(p_dropout)

        # 如果定义了窗口大小，则初始化相对位置编码的参数
        if window_size is not None:
            n_heads_rel = 1 if heads_share else n_heads
            rel_stddev = self.k_channels**-0.5
            self.emb_rel_k = nn.Parameter(
                torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels)
                * rel_stddev
            )
            self.emb_rel_v = nn.Parameter(
                torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels)
                * rel_stddev
            )

        # 使用 Xavier 初始化方法初始化卷积层的权重
        nn.init.xavier_uniform_(self.conv_q.weight)
        nn.init.xavier_uniform_(self.conv_k.weight)
        nn.init.xavier_uniform_(self.conv_v.weight)
        # 如果启用了近似初始化，则将键的权重和偏置初始化为查询的权重和偏置
        if proximal_init:
            with torch.no_grad():
                self.conv_k.weight.copy_(self.conv_q.weight)
                self.conv_k.bias.copy_(self.conv_q.bias)
    # 对输入进行前向传播，计算注意力权重并应用到值向量上
    def forward(self, x, c, attn_mask=None):
        # 使用卷积层计算查询向量
        q = self.conv_q(x)
        # 使用卷积层计算键向量
        k = self.conv_k(c)
        # 使用卷积层计算值向量
        v = self.conv_v(c)

        # 使用注意力机制计算输出向量，并返回注意力权重
        x, self.attn = self.attention(q, k, v, mask=attn_mask)

        # 使用卷积层计算输出向量
        x = self.conv_o(x)
        return x

    # 执行矩阵乘法，计算相对值
    def _matmul_with_relative_values(self, x, y):
        """
        x: [b, h, l, m]
        y: [h or 1, m, d]
        ret: [b, h, l, d]
        """
        ret = torch.matmul(x, y.unsqueeze(0))
        return ret

    # 执行矩阵乘法，计算相对键
    def _matmul_with_relative_keys(self, x, y):
        """
        x: [b, h, l, d]
        y: [h or 1, m, d]
        ret: [b, h, l, m]
        """
        ret = torch.matmul(x, y.unsqueeze(0).transpose(-2, -1))
        return ret

    # 获取相对位置嵌入
    def _get_relative_embeddings(self, relative_embeddings, length):
        max_relative_position = 2 * self.window_size + 1
        # 在进行切片之前进行填充，以避免使用条件操作
        pad_length = max(length - (self.window_size + 1), 0)
        slice_start_position = max((self.window_size + 1) - length, 0)
        slice_end_position = slice_start_position + 2 * length - 1
        if pad_length > 0:
            # 对相对位置嵌入进行填充
            padded_relative_embeddings = F.pad(
                relative_embeddings,
                commons.convert_pad_shape([[0, 0], [pad_length, pad_length], [0, 0]]),
            )
        else:
            padded_relative_embeddings = relative_embeddings
        # 获取使用的相对位置嵌入
        used_relative_embeddings = padded_relative_embeddings[
            :, slice_start_position:slice_end_position
        ]
        return used_relative_embeddings
    # 将相对位置转换为绝对位置
    def _relative_position_to_absolute_position(self, x):
        """
        x: [b, h, l, 2*l-1]
        ret: [b, h, l, l]
        """
        # 获取输入张量的维度信息
        batch, heads, length, _ = x.size()
        # 在列上添加填充，以从相对索引转换为绝对索引
        x = F.pad(x, commons.convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, 1]]))

        # 添加额外的元素，使其加起来形成形状 (len+1, 2*len-1)
        x_flat = x.view([batch, heads, length * 2 * length])
        x_flat = F.pad(
            x_flat, commons.convert_pad_shape([[0, 0], [0, 0], [0, length - 1]])
        )

        # 重塑并切片出填充的元素
        x_final = x_flat.view([batch, heads, length + 1, 2 * length - 1])[
            :, :, :length, length - 1 :
        ]
        return x_final

    # 将绝对位置转换为相对位置
    def _absolute_position_to_relative_position(self, x):
        """
        x: [b, h, l, l]
        ret: [b, h, l, 2*l-1]
        """
        # 获取输入张量的维度信息
        batch, heads, length, _ = x.size()
        # 沿列添加填充
        x = F.pad(
            x, commons.convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, length - 1]])
        )
        x_flat = x.view([batch, heads, length**2 + length * (length - 1)])
        # 在重塑后的元素之前添加 0，这将使元素偏移
        x_flat = F.pad(x_flat, commons.convert_pad_shape([[0, 0], [0, 0], [length, 0]]))
        x_final = x_flat.view([batch, heads, length, 2 * length])[:, :, :, 1:]
        return x_final

    # 用于自注意力的偏置，以鼓励关注接近的位置
    def _attention_bias_proximal(self, length):
        """Bias for self-attention to encourage attention to close positions.
        Args:
          length: an integer scalar.
        Returns:
          a Tensor with shape [1, 1, length, length]
        """
        # 创建一个长度为 length 的浮点数张量
        r = torch.arange(length, dtype=torch.float32)
        # 计算位置之间的差异
        diff = torch.unsqueeze(r, 0) - torch.unsqueeze(r, 1)
        # 返回一个形状为 [1, 1, length, length] 的张量
        return torch.unsqueeze(torch.unsqueeze(-torch.log1p(torch.abs(diff)), 0), 0)
class FFN(nn.Module):
    # 定义一个前馈神经网络模型
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
        # 初始化函数，设置模型的各种参数
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.activation = activation
        self.causal = causal

        if causal:
            # 如果是因果卷积，则使用因果卷积的填充方式
            self.padding = self._causal_padding
        else:
            # 否则使用普通卷积的填充方式
            self.padding = self._same_padding

        # 定义第一个卷积层
        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size)
        # 定义第二个卷积层
        self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size)
        # 定义 dropout 层
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x, x_mask):
        # 前向传播函数
        x = self.conv_1(self.padding(x * x_mask))
        if self.activation == "gelu":
            # 如果激活函数是 gelu，则使用 gelu 函数
            x = x * torch.sigmoid(1.702 * x)
        else:
            # 否则使用 relu 函数
            x = torch.relu(x)
        x = self.drop(x)
        x = self.conv_2(self.padding(x * x_mask))
        return x * x_mask

    def _causal_padding(self, x):
        # 定义因果卷积的填充方式
        if self.kernel_size == 1:
            return x
        pad_l = self.kernel_size - 1
        pad_r = 0
        padding = [[0, 0], [0, 0], [pad_l, pad_r]]
        x = F.pad(x, commons.convert_pad_shape(padding))
        return x

    def _same_padding(self, x):
        # 定义普通卷积的填充方式
        if self.kernel_size == 1:
            return x
        pad_l = (self.kernel_size - 1) // 2
        pad_r = self.kernel_size // 2
        padding = [[0, 0], [0, 0], [pad_l, pad_r]]
        x = F.pad(x, commons.convert_pad_shape(padding))
        return x
```