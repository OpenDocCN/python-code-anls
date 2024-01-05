# `d:/src/tocomm/Bert-VITS2\onnx_modules\V210\attentions_onnx.py`

```
import math  # 导入 math 模块，用于数学运算
import torch  # 导入 torch 模块，用于构建神经网络
from torch import nn  # 从 torch 模块中导入 nn 模块，用于构建神经网络层
from torch.nn import functional as F  # 从 torch.nn 模块中导入 functional 模块，并重命名为 F

import commons  # 导入自定义的 commons 模块
import logging  # 导入 logging 模块，用于记录日志信息

logger = logging.getLogger(__name__)  # 创建一个 logger 对象，用于记录当前模块的日志信息


class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()  # 调用父类的构造函数
        self.channels = channels  # 初始化 channels 属性
        self.eps = eps  # 初始化 eps 属性

        self.gamma = nn.Parameter(torch.ones(channels))  # 创建一个可学习的参数 gamma，初始值为 1，用于归一化
        self.beta = nn.Parameter(torch.zeros(channels))  # 创建一个可学习的参数 beta，初始值为 0，用于归一化
    def forward(self, x):  # 定义一个前向传播函数，接受输入 x
        x = x.transpose(1, -1)  # 对输入 x 进行维度转置操作
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)  # 对输入 x 进行 layer normalization 操作
        return x.transpose(1, -1)  # 再次对结果进行维度转置操作并返回

@torch.jit.script  # 使用 torch.jit.script 进行脚本化
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):  # 定义一个融合了加法、双曲正切、Sigmoid 和乘法的函数
    n_channels_int = n_channels[0]  # 获取 n_channels 的第一个元素
    in_act = input_a + input_b  # 对输入 input_a 和 input_b 进行加法操作
    t_act = torch.tanh(in_act[:, :n_channels_int, :])  # 对加法结果的前 n_channels_int 个通道进行双曲正切操作
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])  # 对加法结果的后 n_channels_int 个通道进行 Sigmoid 操作
    acts = t_act * s_act  # 将双曲正切和 Sigmoid 的结果进行逐元素相乘
    return acts  # 返回最终结果

class Encoder(nn.Module):  # 定义一个名为 Encoder 的类，继承自 nn.Module
    def __init__(  # 定义初始化函数，接受一系列参数
        self,
        hidden_channels,  # 隐藏通道数
        filter_channels,  # 过滤器通道数
        n_heads,  # 多头注意力机制的头数
        n_layers,  # 网络层数
        kernel_size=1,  # 卷积核大小，默认为1
        p_dropout=0.0,  # dropout概率，默认为0.0
        window_size=4,  # 窗口大小，默认为4
        isflow=True,  # 是否为流模式，默认为True
        **kwargs  # 其他参数
    ):
        super().__init__()  # 调用父类的构造函数
        self.hidden_channels = hidden_channels  # 隐藏层通道数
        self.filter_channels = filter_channels  # 过滤器通道数
        self.n_heads = n_heads  # 多头注意力机制的头数
        self.n_layers = n_layers  # 网络层数
        self.kernel_size = kernel_size  # 卷积核大小
        self.p_dropout = p_dropout  # dropout概率
        self.window_size = window_size  # 窗口大小
        # if isflow:  # 如果是流模式
        #  cond_layer = torch.nn.Conv1d(256, 2*hidden_channels*n_layers, 1)  # 定义条件层
        #  self.cond_pre = torch.nn.Conv1d(hidden_channels, 2*hidden_channels, 1)  # 定义条件前处理层
        #  self.cond_layer = weight_norm(cond_layer, name='weight')
        #  self.gin_channels = 256
        # 设置条件层的索引为网络层数
        self.cond_layer_idx = self.n_layers
        # 如果参数中包含"gin_channels"，则将self.gin_channels设置为参数中的值
        if "gin_channels" in kwargs:
            self.gin_channels = kwargs["gin_channels"]
            # 如果self.gin_channels不为0，则创建一个线性层，将输入维度设置为self.gin_channels，输出维度设置为self.hidden_channels
            if self.gin_channels != 0:
                self.spk_emb_linear = nn.Linear(self.gin_channels, self.hidden_channels)
                # 如果参数中包含"cond_layer_idx"，则将self.cond_layer_idx设置为参数中的值，否则设置为2
                self.cond_layer_idx = (
                    kwargs["cond_layer_idx"] if "cond_layer_idx" in kwargs else 2
                )
                # 记录日志，输出self.gin_channels和self.cond_layer_idx的值
                logging.debug(self.gin_channels, self.cond_layer_idx)
                # 断言self.cond_layer_idx小于self.n_layers，如果不成立则抛出异常
                assert (
                    self.cond_layer_idx < self.n_layers
                ), "cond_layer_idx should be less than n_layers"
        # 创建一个丢弃层，丢弃概率为p_dropout
        self.drop = nn.Dropout(p_dropout)
        # 初始化注意力层、第一个归一化层、前馈神经网络层、第二个归一化层为ModuleList
        self.attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()
        for i in range(self.n_layers):  # 循环n_layers次，为每一层添加注意力层和前馈神经网络层
            self.attn_layers.append(  # 将MultiHeadAttention实例添加到attn_layers列表中
                MultiHeadAttention(  # 创建MultiHeadAttention实例
                    hidden_channels,  # 隐藏层通道数
                    hidden_channels,  # 隐藏层通道数
                    n_heads,  # 注意力头的数量
                    p_dropout=p_dropout,  # 丢弃概率
                    window_size=window_size,  # 窗口大小
                )
            )
            self.norm_layers_1.append(LayerNorm(hidden_channels))  # 将LayerNorm实例添加到norm_layers_1列表中
            self.ffn_layers.append(  # 将FFN实例添加到ffn_layers列表中
                FFN(  # 创建FFN实例
                    hidden_channels,  # 隐藏层通道数
                    hidden_channels,  # 隐藏层通道数
                    filter_channels,  # 过滤器通道数
                    kernel_size,  # 卷积核大小
                    p_dropout=p_dropout,  # 丢弃概率
                )
            )
        self.norm_layers_2.append(LayerNorm(hidden_channels))  # 将 LayerNorm 模块添加到 norm_layers_2 列表中

    def forward(self, x, x_mask, g=None):
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)  # 创建注意力掩码
        x = x * x_mask  # 对输入进行掩码处理
        for i in range(self.n_layers):  # 遍历每个层
            if i == self.cond_layer_idx and g is not None:  # 如果当前层是条件层并且存在条件信息 g
                g = self.spk_emb_linear(g.transpose(1, 2))  # 对条件信息进行线性变换
                g = g.transpose(1, 2)  # 转置条件信息
                x = x + g  # 将条件信息加到输入上
                x = x * x_mask  # 对结果进行掩码处理
            y = self.attn_layers[i](x, x, attn_mask)  # 使用注意力层处理输入
            y = self.drop(y)  # 对输出进行 dropout
            x = self.norm_layers_1[i](x + y)  # 对输出进行 LayerNorm 处理

            y = self.ffn_layers[i](x, x_mask)  # 使用前馈神经网络层处理输入
            y = self.drop(y)  # 对输出进行 dropout
            x = self.norm_layers_2[i](x + y)  # 对输出进行 LayerNorm 处理
        x = x * x_mask  # 最终输出结果再次进行掩码处理
        return x  # 返回处理后的结果
class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        channels,  # 输入通道数
        out_channels,  # 输出通道数
        n_heads,  # 多头注意力机制的头数
        p_dropout=0.0,  # 丢弃概率，默认为0
        window_size=None,  # 窗口大小
        heads_share=True,  # 是否共享多头注意力机制的权重
        block_length=None,  # 块长度
        proximal_bias=False,  # 是否使用邻近偏置
        proximal_init=False,  # 是否使用邻近初始化
    ):
        super().__init__()
        assert channels % n_heads == 0  # 断言输入通道数能够被头数整除

        self.channels = channels  # 初始化输入通道数
        self.out_channels = out_channels  # 初始化输出通道数
        # 设置注意力头的数量
        self.n_heads = n_heads
        # 设置丢弃概率
        self.p_dropout = p_dropout
        # 设置窗口大小
        self.window_size = window_size
        # 设置是否共享注意力头
        self.heads_share = heads_share
        # 设置块长度
        self.block_length = block_length
        # 设置近似偏差
        self.proximal_bias = proximal_bias
        # 设置近似初始化
        self.proximal_init = proximal_init
        # 初始化注意力
        self.attn = None

        # 计算每个注意力头的通道数
        self.k_channels = channels // n_heads
        # 创建卷积层用于计算查询
        self.conv_q = nn.Conv1d(channels, channels, 1)
        # 创建卷积层用于计算键
        self.conv_k = nn.Conv1d(channels, channels, 1)
        # 创建卷积层用于计算值
        self.conv_v = nn.Conv1d(channels, channels, 1)
        # 创建卷积层用于计算输出
        self.conv_o = nn.Conv1d(channels, out_channels, 1)
        # 创建丢弃层
        self.drop = nn.Dropout(p_dropout)

        # 如果存在窗口大小
        if window_size is not None:
            # 如果共享注意力头，则设置相对注意力头数量为1，否则为n_heads
            n_heads_rel = 1 if heads_share else n_heads
            # 计算相对标准差
            rel_stddev = self.k_channels**-0.5
            # 创建相对键的参数
            self.emb_rel_k = nn.Parameter(
        torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels)
        * rel_stddev
```
创建一个大小为 (n_heads_rel, window_size * 2 + 1, self.k_channels) 的张量，张量中的值是从均值为 0，标准差为 rel_stddev 的正态分布中随机抽取的。

```
    self.emb_rel_v = nn.Parameter(
        torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels)
        * rel_stddev
    )
```
创建一个大小为 (n_heads_rel, window_size * 2 + 1, self.k_channels) 的张量，并将其封装为一个可训练的参数。

```
nn.init.xavier_uniform_(self.conv_q.weight)
nn.init.xavier_uniform_(self.conv_k.weight)
nn.init.xavier_uniform_(self.conv_v.weight)
```
使用 Xavier 初始化方法对 self.conv_q.weight, self.conv_k.weight, self.conv_v.weight 进行初始化。

```
if proximal_init:
    with torch.no_grad():
        self.conv_k.weight.copy_(self.conv_q.weight)
        self.conv_k.bias.copy_(self.conv_q.bias)
```
如果 proximal_init 为 True，则使用无梯度计算的方式将 self.conv_q.weight 和 self.conv_q.bias 复制给 self.conv_k.weight 和 self.conv_k.bias。

```
def forward(self, x, c, attn_mask=None):
    q = self.conv_q(x)
    k = self.conv_k(c)
    v = self.conv_v(c)
```
定义了一个前向传播的方法，其中使用 self.conv_q 对输入 x 进行卷积操作得到 q，使用 self.conv_k 对输入 c 进行卷积操作得到 k，使用 self.conv_v 对输入 c 进行卷积操作得到 v。
        x, self.attn = self.attention(q, k, v, mask=attn_mask)  # 使用注意力机制计算输出 x，并将注意力权重保存在 self.attn 中

        x = self.conv_o(x)  # 使用卷积层 conv_o 处理输出 x
        return x  # 返回处理后的输出 x

    def attention(self, query, key, value, mask=None):
        # reshape [b, d, t] -> [b, n_h, t, d_k]
        b, d, t_s, t_t = (*key.size(), query.size(2))  # 获取 query 和 key 的维度信息
        query = query.view(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)  # 重塑 query 的形状并进行转置
        key = key.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)  # 重塑 key 的形状并进行转置
        value = value.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)  # 重塑 value 的形状并进行转置

        scores = torch.matmul(query / math.sqrt(self.k_channels), key.transpose(-2, -1))  # 计算注意力分数
        if self.window_size is not None:  # 如果存在窗口大小限制
            assert (
                t_s == t_t
            ), "Relative attention is only available for self-attention."  # 断言 t_s 和 t_t 相等，相对注意力只适用于自注意力
            key_relative_embeddings = self._get_relative_embeddings(self.emb_rel_k, t_s)  # 获取相对位置编码
            rel_logits = self._matmul_with_relative_keys(
# 如果存在局部相对位置编码，则计算局部相对位置编码
if self.use_local_relative_position:
    rel_logits = self._compute_local_relative_position_logits(
        query / math.sqrt(self.k_channels), key_relative_embeddings
    )
    scores_local = self._relative_position_to_absolute_position(rel_logits)
    scores = scores + scores_local

# 如果存在近距离偏置，则添加近距离偏置到得分中
if self.proximal_bias:
    assert t_s == t_t, "Proximal bias is only available for self-attention."
    scores = scores + self._attention_bias_proximal(t_s).to(
        device=scores.device, dtype=scores.dtype
    )

# 如果存在掩码，则用较大的负数填充掩码位置
if mask is not None:
    scores = scores.masked_fill(mask == 0, -1e4)
    # 如果存在块长度限制，则生成块掩码
    if self.block_length is not None:
        assert (
            t_s == t_t
        ), "Local attention is only available for self-attention."
        block_mask = (
            torch.ones_like(scores)
            .triu(-self.block_length)
            .tril(self.block_length)
        )
        scores = scores.masked_fill(block_mask == 0, -1e4)  # 使用 block_mask 对 scores 进行填充，将为 0 的位置填充为 -1e4
        p_attn = F.softmax(scores, dim=-1)  # 对 scores 进行 softmax 操作，得到注意力权重 p_attn，维度为 [b, n_h, t_t, t_s]
        p_attn = self.drop(p_attn)  # 对 p_attn 进行 dropout 操作
        output = torch.matmul(p_attn, value)  # 使用注意力权重 p_attn 对 value 进行加权求和，得到 output
        if self.window_size is not None:  # 如果存在窗口大小限制
            relative_weights = self._absolute_position_to_relative_position(p_attn)  # 将绝对位置编码转换为相对位置编码
            value_relative_embeddings = self._get_relative_embeddings(self.emb_rel_v, t_s)  # 获取相对位置编码的嵌入
            output = output + self._matmul_with_relative_values(relative_weights, value_relative_embeddings)  # 使用相对位置编码对 output 进行加权求和
        output = output.transpose(2, 3).contiguous().view(b, d, t_t)  # 对 output 进行维度变换，得到最终的输出结果，维度为 [b, d, t_t]
        return output, p_attn  # 返回输出结果和注意力权重

    def _matmul_with_relative_values(self, x, y):
        """
        x: [b, h, l, m]
        y: [m, d]
        """
        y: [h or 1, m, d]  # 定义变量y的形状为[h或1, m, d]

        ret: [b, h, l, d]  # 定义变量ret的形状为[b, h, l, d]

        """
        ret = torch.matmul(x, y.unsqueeze(0))  # 使用torch.matmul函数计算x和y.unsqueeze(0)的矩阵乘法，结果赋值给ret
        return ret  # 返回ret作为函数的输出结果

    def _matmul_with_relative_keys(self, x, y):
        """
        x: [b, h, l, d]  # 定义变量x的形状为[b, h, l, d]
        y: [h or 1, m, d]  # 定义变量y的形状为[h或1, m, d]
        ret: [b, h, l, m]  # 定义变量ret的形状为[b, h, l, m]
        """
        ret = torch.matmul(x, y.unsqueeze(0).transpose(-2, -1))  # 使用torch.matmul函数计算x和y.unsqueeze(0).transpose(-2, -1)的矩阵乘法，结果赋值给ret
        return ret  # 返回ret作为函数的输出结果

    def _get_relative_embeddings(self, relative_embeddings, length):
        max_relative_position = 2 * self.window_size + 1  # 计算最大相对位置
        # Pad first before slice to avoid using cond ops.
        pad_length = max(length - (self.window_size + 1), 0)  # 计算pad的长度，避免使用条件操作
        slice_start_position = max((self.window_size + 1) - length, 0)  # 计算切片的起始位置，避免使用条件操作
        # 计算切片结束位置
        slice_end_position = slice_start_position + 2 * length - 1
        # 如果填充长度大于0，则对相对位置嵌入进行填充
        if pad_length > 0:
            padded_relative_embeddings = F.pad(
                relative_embeddings,
                commons.convert_pad_shape([[0, 0], [pad_length, pad_length], [0, 0]]),
            )
        else:
            # 否则直接使用相对位置嵌入
            padded_relative_embeddings = relative_embeddings
        # 从填充后的相对位置嵌入中提取出需要使用的部分
        used_relative_embeddings = padded_relative_embeddings[
            :, slice_start_position:slice_end_position
        ]
        # 返回提取出的相对位置嵌入
        return used_relative_embeddings

    def _relative_position_to_absolute_position(self, x):
        """
        x: [b, h, l, 2*l-1]
        ret: [b, h, l, l]
        """
        # 获取输入张量的维度信息
        batch, heads, length, _ = x.size()
        # 将填充的列连接起来，以从相对位置转换为绝对位置索引
        x = F.pad(x, commons.convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, 1]]))
        # 使用 F.pad 函数对输入的张量 x 进行填充，使其形状变为 [batch, heads, length, 2*length]

        # Concat extra elements so to add up to shape (len+1, 2*len-1).
        x_flat = x.view([batch, heads, length * 2 * length])
        # 将 x 进行展平操作，变为形状 [batch, heads, length * 2 * length]
        x_flat = F.pad(
            x_flat, commons.convert_pad_shape([[0, 0], [0, 0], [0, length - 1]])
        )
        # 对展平后的 x 进行填充，使其形状变为 [batch, heads, length * 2 * length + length - 1]

        # Reshape and slice out the padded elements.
        x_final = x_flat.view([batch, heads, length + 1, 2 * length - 1])[:, :, :length, length - 1 :]
        # 将填充后的 x 进行形状变换和切片操作，得到最终的输出 x_final，形状为 [batch, heads, length, 2*length-1]
        return x_final

    def _absolute_position_to_relative_position(self, x):
        """
        x: [b, h, l, l]
        ret: [b, h, l, 2*l-1]
        """
        batch, heads, length, _ = x.size()
        # 获取输入张量 x 的形状信息
        # 在列上填充
        x = F.pad(
            x, commons.convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, length - 1]])
        )
        # 将 x 展平为 [batch, heads, length**2 + length * (length - 1)] 的形状
        x_flat = x.view([batch, heads, length**2 + length * (length - 1)])
        # 在开始处添加 0，这将使重塑后的元素产生偏移
        x_flat = F.pad(x_flat, commons.convert_pad_shape([[0, 0], [0, 0], [length, 0]]))
        # 将 x_flat 重塑为 [batch, heads, length, 2 * length] 的形状，并且取出偏移后的部分
        x_final = x_flat.view([batch, heads, length, 2 * length])[:, :, :, 1:]
        return x_final

    def _attention_bias_proximal(self, length):
        """鼓励自注意力关注接近位置的偏置。
        Args:
          length: 一个整数标量。
        Returns:
          一个形状为 [1, 1, length, length] 的张量
        """
        r = torch.arange(length, dtype=torch.float32)
        diff = torch.unsqueeze(r, 0) - torch.unsqueeze(r, 1)
        return torch.unsqueeze(torch.unsqueeze(-torch.log1p(torch.abs(diff)), 0), 0)
class FFN(nn.Module):  # 定义一个名为FFN的类，继承自nn.Module
    def __init__(  # 初始化方法
        self,  # 第一个参数为self，表示实例对象本身
        in_channels,  # 输入通道数
        out_channels,  # 输出通道数
        filter_channels,  # 过滤器通道数
        kernel_size,  # 卷积核大小
        p_dropout=0.0,  # 丢弃概率，默认为0.0
        activation=None,  # 激活函数，默认为None
        causal=False,  # 是否是因果卷积，默认为False
    ):
        super().__init__()  # 调用父类的初始化方法
        self.in_channels = in_channels  # 设置实例对象的输入通道数属性
        self.out_channels = out_channels  # 设置实例对象的输出通道数属性
        self.filter_channels = filter_channels  # 设置实例对象的过滤器通道数属性
        self.kernel_size = kernel_size  # 设置实例对象的卷积核大小属性
        self.p_dropout = p_dropout  # 设置实例对象的丢弃概率属性
        self.activation = activation  # 设置实例对象的激活函数属性
        self.causal = causal  # 设置类属性 causal 为传入的参数值

        if causal:  # 如果 causal 为 True
            self.padding = self._causal_padding  # 设置 padding 方法为 _causal_padding 方法
        else:  # 如果 causal 为 False
            self.padding = self._same_padding  # 设置 padding 方法为 _same_padding 方法

        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size)  # 创建第一个卷积层
        self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size)  # 创建第二个卷积层
        self.drop = nn.Dropout(p_dropout)  # 创建 Dropout 层

    def forward(self, x, x_mask):  # 定义前向传播方法，接受输入 x 和掩码 x_mask
        x = self.conv_1(self.padding(x * x_mask))  # 对输入进行卷积和填充操作
        if self.activation == "gelu":  # 如果激活函数为 gelu
            x = x * torch.sigmoid(1.702 * x)  # 使用 gelu 激活函数
        else:  # 如果激活函数不是 gelu
            x = torch.relu(x)  # 使用 ReLU 激活函数
        x = self.drop(x)  # 对结果进行 Dropout 操作
        x = self.conv_2(self.padding(x * x_mask))  # 对结果进行第二次卷积和填充操作
        return x * x_mask  # 返回结果乘以掩码
    def _causal_padding(self, x):  # 定义一个名为_causal_padding的方法，接受参数x
        if self.kernel_size == 1:  # 如果卷积核大小为1
            return x  # 返回x
        pad_l = self.kernel_size - 1  # 计算左侧填充大小
        pad_r = 0  # 右侧填充大小为0
        padding = [[0, 0], [0, 0], [pad_l, pad_r]]  # 构建填充参数
        x = F.pad(x, commons.convert_pad_shape(padding))  # 对输入x进行填充操作
        return x  # 返回填充后的x

    def _same_padding(self, x):  # 定义一个名为_same_padding的方法，接受参数x
        if self.kernel_size == 1:  # 如果卷积核大小为1
            return x  # 返回x
        pad_l = (self.kernel_size - 1) // 2  # 计算左侧填充大小
        pad_r = self.kernel_size // 2  # 计算右侧填充大小
        padding = [[0, 0], [0, 0], [pad_l, pad_r]]  # 构建填充参数
        x = F.pad(x, commons.convert_pad_shape(padding))  # 对输入x进行填充操作
        return x  # 返回填充后的x
```