# `Bert-VITS2\attentions.py`

```py
# 导入数学库
import math
# 导入 PyTorch 库
import torch
# 从 PyTorch 库中导入 nn 模块
from torch import nn
# 从 PyTorch 的 nn 模块中导入 functional 模块，并重命名为 F
from torch.nn import functional as F
# 导入自定义的 commons 模块
import commons
# 导入日志模块
import logging

# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)

# 定义 LayerNorm 类，继承自 nn.Module
class LayerNorm(nn.Module):
    # 初始化方法
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        # 初始化 gamma 参数
        self.gamma = nn.Parameter(torch.ones(channels))
        # 初始化 beta 参数
        self.beta = nn.Parameter(torch.zeros(channels))

    # 前向传播方法
    def forward(self, x):
        # 转置张量 x
        x = x.transpose(1, -1)
        # 对 x 进行 Layer Normalization
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        # 再次转置张量 x
        return x.transpose(1, -1)

# 使用 Torch 的脚本装饰器进行脚本化
@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    # 获取通道数的整数值
    n_channels_int = n_channels[0]
    # 对输入张量进行加法操作
    in_act = input_a + input_b
    # 对加法结果的前半部分进行双曲正切函数操作
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    # 对加法结果的后半部分进行 Sigmoid 函数操作
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    # 对双曲正切函数和 Sigmoid 函数的结果进行逐元素相乘
    acts = t_act * s_act
    return acts

# 定义 Encoder 类，继承自 nn.Module
class Encoder(nn.Module):
    # 初始化方法
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
    # 前向传播方法
    def forward(self, x, x_mask, g=None):
        # 生成注意力遮罩
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        # 对输入张量进行遮罩操作
        x = x * x_mask
        # 循环遍历每一层
        for i in range(self.n_layers):
            # 如果当前层是条件层且存在条件输入 g
            if i == self.cond_layer_idx and g is not None:
                # 对条件输入 g 进行线性变换
                g = self.spk_emb_linear(g.transpose(1, 2))
                g = g.transpose(1, 2)
                # 将条件输入 g 加到输入张量 x 上
                x = x + g
                # 对结果再次进行遮罩操作
                x = x * x_mask
            # 使用注意力层进行处理
            y = self.attn_layers[i](x, x, attn_mask)
            y = self.drop(y)
            # 对输入张量进行 Layer Normalization
            x = self.norm_layers_1[i](x + y)

            # 使用前馈神经网络层进行处理
            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)
            # 对输入张量再次进行 Layer Normalization
            x = self.norm_layers_2[i](x + y)
        # 对最终结果再次进行遮罩操作
        x = x * x_mask
        return x

# 定义 Decoder 类，继承自 nn.Module
class Decoder(nn.Module):
    # 初始化函数，设置模型的各种参数
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
        # 调用父类的初始化函数
        super().__init__()
        # 设置模型的隐藏通道数
        self.hidden_channels = hidden_channels
        # 设置模型的过滤通道数
        self.filter_channels = filter_channels
        # 设置模型的注意力头数
        self.n_heads = n_heads
        # 设置模型的层数
        self.n_layers = n_layers
        # 设置模型的卷积核大小
        self.kernel_size = kernel_size
        # 设置模型的丢弃率
        self.p_dropout = p_dropout
        # 设置模型的近端偏置
        self.proximal_bias = proximal_bias
        # 设置模型的近端初始化
        self.proximal_init = proximal_init

        # 创建丢弃层
        self.drop = nn.Dropout(p_dropout)
        # 创建自注意力层列表
        self.self_attn_layers = nn.ModuleList()
        # 创建第一个层规范化层列表
        self.norm_layers_0 = nn.ModuleList()
        # 创建编码-解码注意力层列表
        self.encdec_attn_layers = nn.ModuleList()
        # 创建第二个层规范化层列表
        self.norm_layers_1 = nn.ModuleList()
        # 创建前馈神经网络层列表
        self.ffn_layers = nn.ModuleList()
        # 创建第三个层规范化层列表
        self.norm_layers_2 = nn.ModuleList()
        # 循环创建每一层的自注意力层、规范化层、编码-解码注意力层、规范化层和前馈神经网络层
        for i in range(self.n_layers):
            # 添加自注意力层
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
            # 添加第一个层规范化层
            self.norm_layers_0.append(LayerNorm(hidden_channels))
            # 添加编码-解码注意力层
            self.encdec_attn_layers.append(
                MultiHeadAttention(
                    hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout
                )
            )
            # 添加第二个层规范化层
            self.norm_layers_1.append(LayerNorm(hidden_channels))
            # 添加前馈神经网络层
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
            # 添加第三个层规范化层
            self.norm_layers_2.append(LayerNorm(hidden_channels))
    # 定义一个方法，用于实现编码器-解码器模型的前向传播
    def forward(self, x, x_mask, h, h_mask):
        """
        x: decoder input
        h: encoder output
        """
        # 生成自注意力机制的掩码，用于屏蔽未来位置的信息
        self_attn_mask = commons.subsequent_mask(x_mask.size(2)).to(
            device=x.device, dtype=x.dtype
        )
        # 生成编码器-解码器注意力机制的掩码，用于屏蔽填充位置的信息
        encdec_attn_mask = h_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        # 对输入进行掩码处理
        x = x * x_mask
        # 循环执行多层的自注意力机制、编码器-解码器注意力机制和前馈神经网络
        for i in range(self.n_layers):
            # 自注意力机制
            y = self.self_attn_layers[i](x, x, self_attn_mask)
            y = self.drop(y)
            x = self.norm_layers_0[i](x + y)

            # 编码器-解码器注意力机制
            y = self.encdec_attn_layers[i](x, h, encdec_attn_mask)
            y = self.drop(y)
            x = self.norm_layers_1[i](x + y)

            # 前馈神经网络
            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)
            x = self.norm_layers_2[i](x + y)
        # 再次对输入进行掩码处理
        x = x * x_mask
        # 返回处理后的结果
        return x
class MultiHeadAttention(nn.Module):
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
        super().__init__()
        assert channels % n_heads == 0  # 断言确保通道数可以被头数整除

        self.channels = channels  # 设置输入通道数
        self.out_channels = out_channels  # 设置输出通道数
        self.n_heads = n_heads  # 设置注意力头数
        self.p_dropout = p_dropout  # 设置丢弃概率
        self.window_size = window_size  # 设置窗口大小
        self.heads_share = heads_share  # 设置头是否共享
        self.block_length = block_length  # 设置块长度
        self.proximal_bias = proximal_bias  # 设置邻近偏置
        self.proximal_init = proximal_init  # 设置邻近初始化
        self.attn = None  # 初始化注意力

        self.k_channels = channels // n_heads  # 计算每个头的通道数
        self.conv_q = nn.Conv1d(channels, channels, 1)  # 创建查询卷积层
        self.conv_k = nn.Conv1d(channels, channels, 1)  # 创建键卷积层
        self.conv_v = nn.Conv1d(channels, channels, 1)  # 创建值卷积层
        self.conv_o = nn.Conv1d(channels, out_channels, 1)  # 创建输出卷积层
        self.drop = nn.Dropout(p_dropout)  # 创建丢弃层

        if window_size is not None:  # 如果窗口大小不为空
            n_heads_rel = 1 if heads_share else n_heads  # 计算相对位置编码的头数
            rel_stddev = self.k_channels**-0.5  # 计算相对位置编码的标准差
            self.emb_rel_k = nn.Parameter(  # 创建相对位置编码的键参数
                torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels)
                * rel_stddev
            )
            self.emb_rel_v = nn.Parameter(  # 创建相对位置编码的值参数
                torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels)
                * rel_stddev
            )

        nn.init.xavier_uniform_(self.conv_q.weight)  # 初始化查询卷积层权重
        nn.init.xavier_uniform_(self.conv_k.weight)  # 初始化键卷积层权重
        nn.init.xavier_uniform_(self.conv_v.weight)  # 初始化值卷积层权重
        if proximal_init:  # 如果使用邻近初始化
            with torch.no_grad():  # 关闭梯度计算
                self.conv_k.weight.copy_(self.conv_q.weight)  # 复制查询卷积层权重到键卷积层
                self.conv_k.bias.copy_(self.conv_q.bias)  # 复制查询卷积层偏置到键卷积层
    # 对输入进行前向传播，计算注意力权重并应用到值向量上
    def forward(self, x, c, attn_mask=None):
        # 使用卷积层计算查询向量
        q = self.conv_q(x)
        # 使用卷积层计算键向量
        k = self.conv_k(c)
        # 使用卷积层计算值向量
        v = self.conv_v(c)

        # 在注意力机制中计算输出向量，并返回注意力权重
        x, self.attn = self.attention(q, k, v, mask=attn_mask)

        # 使用卷积层计算输出向量
        x = self.conv_o(x)
        return x

    # 计算相对值的矩阵乘法
    def _matmul_with_relative_values(self, x, y):
        """
        x: [b, h, l, m]
        y: [h or 1, m, d]
        ret: [b, h, l, d]
        """
        ret = torch.matmul(x, y.unsqueeze(0))
        return ret

    # 计算相对键的矩阵乘法
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
        # 计算窗口大小
        window_size = 2 * self.window_size + 1
        # 在长度不足时进行填充
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
        # 在列上填充0，以实现从相对索引到绝对索引的转换
        x = F.pad(x, commons.convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, 1]]))

        # 在张量末尾添加额外的元素，使其形状达到 (len+1, 2*len-1)
        x_flat = x.view([batch, heads, length * 2 * length])
        x_flat = F.pad(
            x_flat, commons.convert_pad_shape([[0, 0], [0, 0], [0, length - 1]])
        )

        # 重新调整形状并切片掉填充的元素
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
        # 沿着列方向填充0
        x = F.pad(
            x, commons.convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, length - 1]])
        )
        x_flat = x.view([batch, heads, length**2 + length * (length - 1)])
        # 在重塑后的元素之前添加0，以使其偏移
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
        # 生成一个长度为length的浮点数张量
        r = torch.arange(length, dtype=torch.float32)
        # 计算位置之间的差值
        diff = torch.unsqueeze(r, 0) - torch.unsqueeze(r, 1)
        # 对差值取绝对值并取对数，然后添加负号
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
            # 如果激活函数是 gelu，则使用 gelu 激活函数
            x = x * torch.sigmoid(1.702 * x)
        else:
            # 否则使用 relu 激活函数
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