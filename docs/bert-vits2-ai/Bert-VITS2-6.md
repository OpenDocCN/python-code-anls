# Bert-VITS2 源码解析 6

# `D:\src\Bert-VITS2\onnx_modules\V210\models_onnx.py`

```python
import math
import torch
from torch import nn
from torch.nn import functional as F

import commons
import modules
from . import attentions_onnx
from vector_quantize_pytorch import VectorQuantize

from torch.nn import Conv1d, ConvTranspose1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from commons import init_weights, get_padding
from .text import symbols, num_tones, num_languages


class DurationDiscriminator(nn.Module):  # vits2
    def __init__(
        self, in_channels, filter_channels, kernel_size, p_dropout, gin_channels=0
    ):
        super().__init__()

        self.in_channels = in_channels  # input channels
        self.filter_channels = filter_channels  # filter channels
        self.kernel_size = kernel_size  # kernel size
        self.p_dropout = p_dropout  # dropout probability
        self.gin_channels = gin_channels  # gin channels

        self.drop = nn.Dropout(p_dropout)  # dropout layer
        self.conv_1 = nn.Conv1d(  # 1D convolutional layer
            in_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_1 = modules.LayerNorm(filter_channels)  # layer normalization
        self.conv_2 = nn.Conv1d(  # 1D convolutional layer
            filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_2 = modules.LayerNorm(filter_channels)  # layer normalization
        self.dur_proj = nn.Conv1d(1, filter_channels, 1)  # 1D convolutional layer

        self.pre_out_conv_1 = nn.Conv1d(  # 1D convolutional layer
            2 * filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.pre_out_norm_1 = modules.LayerNorm(filter_channels)  # layer normalization
        self.pre_out_conv_2 = nn.Conv1d(  # 1D convolutional layer
            filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.pre_out_norm_2 = modules.LayerNorm(filter_channels)  # layer normalization

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, in_channels, 1)  # 1D convolutional layer

        self.output_layer = nn.Sequential(nn.Linear(filter_channels, 1), nn.Sigmoid())  # linear layer followed by sigmoid activation function

    def forward_probability(self, x, x_mask, dur, g=None):
        dur = self.dur_proj(dur)  # projection of duration
        x = torch.cat([x, dur], dim=1)  # concatenation of input and duration
        x = self.pre_out_conv_1(x * x_mask)  # 1D convolutional layer
        x = torch.relu(x)  # ReLU activation function
        x = self.pre_out_norm_1(x)  # layer normalization
        x = self.drop(x)  # dropout
        x = self.pre_out_conv_2(x * x_mask)  # 1D convolutional layer
        x = torch.relu(x)  # ReLU activation function
        x = self.pre_out_norm_2(x)  # layer normalization
        x = self.drop(x)  # dropout
        x = x * x_mask  # element-wise multiplication
        x = x.transpose(1, 2)  # transpose
        output_prob = self.output_layer(x)  # output layer
        return output_prob  # return output probability

    def forward(self, x, x_mask, dur_r, dur_hat, g=None):
        x = torch.detach(x)  # detach tensor
        if g is not None:
            g = torch.detach(g)  # detach tensor
            x = x + self.cond(g)  # addition
        x = self.conv_1(x * x_mask)  # 1D convolutional layer
        x = torch.relu(x)  # ReLU activation function
        x = self.norm_1(x)  # layer normalization
        x = self.drop(x)  # dropout
        x = self.conv_2(x * x_mask)  # 1D convolutional layer
        x = torch.relu(x)  # ReLU activation function
        x = self.norm_2(x)  # layer normalization
        x = self.drop(x)  # dropout

        output_probs = []
        for dur in [dur_r, dur_hat]:
            output_prob = self.forward_probability(x, x_mask, dur, g)  # forward probability
            output_probs.append(output_prob)  # append output probability

        return output_probs  # return output probabilities
```

# `D:\src\Bert-VITS2\onnx_modules\V210\__init__.py`

```python
from .text.symbols import symbols  # 从text模块中的symbols文件中导入symbols变量
from .models_onnx import SynthesizerTrn  # 从models_onnx模块中导入SynthesizerTrn类

__all__ = ["symbols", "SynthesizerTrn"]  # 定义__all__变量，包含symbols和SynthesizerTrn，用于模块导入时指定可导入的内容
```

# `D:\src\Bert-VITS2\onnx_modules\V210\text\symbols.py`

```python
punctuation = ["!", "?", "…", ",", ".", "'", "-"]  # 创建一个包含标点符号的列表
pu_symbols = punctuation + ["SP", "UNK"]  # 创建一个包含标点符号和特殊标记的列表
pad = "_"  # 创建一个填充标记

# chinese
zh_symbols = [  # 创建一个包含中文音节的列表
    "E",
    "En",
    ...
    "OO",
]
num_zh_tones = 6  # 设置中文音节的音调数量为6

# japanese
ja_symbols = [  # 创建一个包含日文音节的列表
    "N",
    "a",
    ...
    "zy",
]
num_ja_tones = 2  # 设置日文音节的音调数量为2

# English
en_symbols = [  # 创建一个包含英文音素的列表
    "aa",
    "ae",
    ...
    "zh",
]
num_en_tones = 4  # 设置英文音素的音调数量为4

# combine all symbols
normal_symbols = sorted(set(zh_symbols + ja_symbols + en_symbols))  # 合并所有音节并去重排序
symbols = [pad] + normal_symbols + pu_symbols  # 创建一个包含填充标记、所有音节和特殊标记的列表
sil_phonemes_ids = [symbols.index(i) for i in pu_symbols]  # 获取特殊标记在列表中的索引

# combine all tones
num_tones = num_zh_tones + num_ja_tones + num_en_tones  # 计算所有音节的总音调数量

# language maps
language_id_map = {"ZH": 0, "JP": 1, "EN": 2}  # 创建一个语言到ID的映射字典
num_languages = len(language_id_map.keys())  # 获取语言数量

language_tone_start_map = {  # 创建一个语言到音调起始位置的映射字典
    "ZH": 0,
    "JP": num_zh_tones,
    "EN": num_zh_tones + num_ja_tones,
}

if __name__ == "__main__":
    a = set(zh_symbols)  # 创建一个包含中文音节的集合
    b = set(en_symbols)  # 创建一个包含英文音素的集合
    print(sorted(a & b))  # 打印中英文共有的音节
```

# `D:\src\Bert-VITS2\onnx_modules\V210\text\__init__.py`

```python
from .symbols import *  # 从symbols模块中导入所有的变量和函数
```

# `D:\src\Bert-VITS2\onnx_modules\V210_OnnxInference\__init__.py`

```python
import numpy as np  # 导入numpy库
import onnxruntime as ort  # 导入onnxruntime库

def convert_pad_shape(pad_shape):  # 定义函数convert_pad_shape
    layer = pad_shape[::-1]  # 反转pad_shape
    pad_shape = [item for sublist in layer for item in sublist]  # 将layer展开成一维数组
    return pad_shape  # 返回pad_shape

def sequence_mask(length, max_length=None):  # 定义函数sequence_mask
    if max_length is None:  # 如果max_length为空
        max_length = length.max()  # max_length取length的最大值
    x = np.arange(max_length, dtype=length.dtype)  # 生成一个长度为max_length的数组x
    return np.expand_dims(x, 0) < np.expand_dims(length, 1)  # 返回x和length的比较结果

def generate_path(duration, mask):  # 定义函数generate_path
    """
    duration: [b, 1, t_x]
    mask: [b, 1, t_y, t_x]
    """
    b, _, t_y, t_x = mask.shape  # 获取mask的形状
    cum_duration = np.cumsum(duration, -1)  # 对duration进行累加

    cum_duration_flat = cum_duration.reshape(b * t_x)  # 将cum_duration展开成一维数组
    path = sequence_mask(cum_duration_flat, t_y)  # 生成path
    path = path.reshape(b, t_x, t_y)  # 调整path的形状
    path = path ^ np.pad(path, ((0, 0), (1, 0), (0, 0)))[:, :-1]  # 对path进行异或运算
    path = np.expand_dims(path, 1).transpose(0, 1, 3, 2)  # 调整path的形状
    return path  # 返回path

class OnnxInferenceSession:  # 定义类OnnxInferenceSession
    def __init__(self, path, Providers=["CPUExecutionProvider"]):  # 定义初始化方法
        self.enc = ort.InferenceSession(path["enc"], providers=Providers)  # 初始化self.enc
        self.emb_g = ort.InferenceSession(path["emb_g"], providers=Providers)  # 初始化self.emb_g
        self.dp = ort.InferenceSession(path["dp"], providers=Providers)  # 初始化self.dp
        self.sdp = ort.InferenceSession(path["sdp"], providers=Providers)  # 初始化self.sdp
        self.flow = ort.InferenceSession(path["flow"], providers=Providers)  # 初始化self.flow
        self.dec = ort.InferenceSession(path["dec"], providers=Providers)  # 初始化self.dec

    def __call__(  # 定义__call__方法
        self,
        seq,
        tone,
        language,
        bert_zh,
        bert_jp,
        bert_en,
        vqidx,
        sid,
        seed=114514,
        seq_noise_scale=0.8,
        sdp_noise_scale=0.6,
        length_scale=1.0,
        sdp_ratio=0.0,
    ):
        if seq.ndim == 1:  # 如果seq的维度为1
            seq = np.expand_dims(seq, 0)  # 将seq扩展为二维数组
        if tone.ndim == 1:  # 如果tone的维度为1
            tone = np.expand_dims(tone, 0)  # 将tone扩展为二维数组
        if language.ndim == 1:  # 如果language的维度为1
            language = np.expand_dims(language, 0)  # 将language扩展为二维数组
        assert (seq.ndim == 2, tone.ndim == 2, language.ndim == 2)  # 断言seq、tone、language的维度为2
        g = self.emb_g.run(  # 运行self.emb_g
            None,
            {
                "sid": sid.astype(np.int64),
            },
        )[0]  # 获取结果
        g = np.expand_dims(g, -1)  # 将g扩展为三维数组
        enc_rtn = self.enc.run(  # 运行self.enc
            None,
            {
                "x": seq.astype(np.int64),
                "t": tone.astype(np.int64),
                "language": language.astype(np.int64),
                "bert_0": bert_zh.astype(np.float32),
                "bert_1": bert_jp.astype(np.float32),
                "bert_2": bert_en.astype(np.float32),
                "g": g.astype(np.float32),
                "vqidx": vqidx.astype(np.int64),
                "sid": sid.astype(np.int64),
            },
        )  # 获取结果
        x, m_p, logs_p, x_mask = enc_rtn[0], enc_rtn[1], enc_rtn[2], enc_rtn[3]  # 获取结果
        np.random.seed(seed)  # 设置随机种子
        zinput = np.random.randn(x.shape[0], 2, x.shape[2]) * sdp_noise_scale  # 生成zinput
        logw = self.sdp.run(  # 运行self.sdp
            None, {"x": x, "x_mask": x_mask, "zin": zinput.astype(np.float32), "g": g}
        )[0] * (sdp_ratio) + self.dp.run(None, {"x": x, "x_mask": x_mask, "g": g})[
            0
        ] * (
            1 - sdp_ratio
        )  # 获取结果
        w = np.exp(logw) * x_mask * length_scale  # 计算w
        w_ceil = np.ceil(w)  # 对w进行向上取整
        y_lengths = np.clip(np.sum(w_ceil, (1, 2)), a_min=1.0, a_max=100000).astype(
            np.int64
        )  # 计算y_lengths
        y_mask = np.expand_dims(sequence_mask(y_lengths, None), 1)  # 生成y_mask
        attn_mask = np.expand_dims(x_mask, 2) * np.expand_dims(y_mask, -1)  # 生成attn_mask
        attn = generate_path(w_ceil, attn_mask)  # 生成attn
        m_p = np.matmul(attn.squeeze(1), m_p.transpose(0, 2, 1)).transpose(
            0, 2, 1
        )  # 计算m_p
        logs_p = np.matmul(attn.squeeze(1), logs_p.transpose(0, 2, 1)).transpose(
            0, 2, 1
        )  # 计算logs_p

        z_p = (
            m_p
            + np.random.randn(m_p.shape[0], m_p.shape[1], m_p.shape[2])
            * np.exp(logs_p)
            * seq_noise_scale
        )  # 计算z_p

        z = self.flow.run(  # 运行self.flow
            None,
            {
                "z_p": z_p.astype(np.float32),
                "y_mask": y_mask.astype(np.float32),
                "g": g,
            },
        )[0]  # 获取结果

        return self.dec.run(None, {"z_in": z.astype(np.float32), "g": g})[0]  # 返回结果
```

# `D:\src\Bert-VITS2\onnx_modules\V220\attentions_onnx.py`

```python
import math  # 导入数学库
import torch  # 导入PyTorch
from torch import nn  # 从PyTorch中导入神经网络模块
from torch.nn import functional as F  # 从PyTorch中导入函数模块

import commons  # 导入自定义的commons模块
import logging  # 导入日志模块

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


class LayerNorm(nn.Module):  # 定义LayerNorm类，继承自nn.Module
    def __init__(self, channels, eps=1e-5):  # 初始化函数，channels为通道数，eps为epsilon值
        super().__init__()  # 调用父类的初始化函数
        self.channels = channels  # 设置通道数
        self.eps = eps  # 设置epsilon值

        self.gamma = nn.Parameter(torch.ones(channels))  # 初始化gamma参数
        self.beta = nn.Parameter(torch.zeros(channels))  # 初始化beta参数

    def forward(self, x):  # 前向传播函数，x为输入
        x = x.transpose(1, -1)  # 转置x
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)  # 对x进行Layer Norm
        return x.transpose(1, -1)  # 返回转置后的x


@torch.jit.script  # 使用Torch Script装饰器
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):  # 定义函数fused_add_tanh_sigmoid_multiply
    n_channels_int = n_channels[0]  # 获取n_channels的第一个元素
    in_act = input_a + input_b  # 计算input_a和input_b的和
    t_act = torch.tanh(in_act[:, :n_channels_int, :])  # 计算tanh
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])  # 计算sigmoid
    acts = t_act * s_act  # 计算t_act和s_act的乘积
    return acts  # 返回结果


class Encoder(nn.Module):  # 定义Encoder类，继承自nn.Module
    def __init__(  # 初始化函数
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
        super().__init__()  # 调用父类的初始化函数
        self.hidden_channels = hidden_channels  # 设置隐藏通道数
        self.filter_channels = filter_channels  # 设置过滤通道数
        self.n_heads = n_heads  # 设置头数
        self.n_layers = n_layers  # 设置层数
        self.kernel_size = kernel_size  # 设置卷积核大小
        self.p_dropout = p_dropout  # 设置dropout概率
        self.window_size = window_size  # 设置窗口大小
        self.cond_layer_idx = self.n_layers  # 设置条件层索引为n_layers
        if "gin_channels" in kwargs:  # 如果gin_channels在kwargs中
            self.gin_channels = kwargs["gin_channels"]  # 设置gin_channels
            if self.gin_channels != 0:  # 如果gin_channels不为0
                self.spk_emb_linear = nn.Linear(self.gin_channels, self.hidden_channels)  # 初始化线性层
                self.cond_layer_idx = (  # 设置条件层索引
                    kwargs["cond_layer_idx"] if "cond_layer_idx" in kwargs else 2
                )
                logging.debug(self.gin_channels, self.cond_layer_idx)  # 记录日志
                assert (  # 断言条件
                    self.cond_layer_idx < self.n_layers
                ), "cond_layer_idx should be less than n_layers"  # 如果条件不满足，抛出异常
        self.drop = nn.Dropout(p_dropout)  # 初始化Dropout层
        self.attn_layers = nn.ModuleList()  # 初始化注意力层列表
        self.norm_layers_1 = nn.ModuleList()  # 初始化规范化层1列表
        self.ffn_layers = nn.ModuleList()  # 初始化前馈神经网络层列表
        self.norm_layers_2 = nn.ModuleList()  # 初始化规范化层2列表
        for i in range(self.n_layers):  # 遍历层数
            self.attn_layers.append(  # 向注意力层列表中添加元素
                MultiHeadAttention(  # 创建多头注意力层
                    hidden_channels,
                    hidden_channels,
                    n_heads,
                    p_dropout=p_dropout,
                    window_size=window_size,
                )
            )
            self.norm_layers_1.append(LayerNorm(hidden_channels))  # 向规范化层1列表中添加元素
            self.ffn_layers.append(  # 向前馈神经网络层列表中添加元素
                FFN(
                    hidden_channels,
                    hidden_channels,
                    filter_channels,
                    kernel_size,
                    p_dropout=p_dropout,
                )
            )
            self.norm_layers_2.append(LayerNorm(hidden_channels))  # 向规范化层2列表中添加元素

    def forward(self, x, x_mask, g=None):  # 前向传播函数，x为输入，x_mask为掩码，g为条件
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)  # 计算注意力掩码
        x = x * x_mask  # 对x进行掩码
        for i in range(self.n_layers):  # 遍历层数
            if i == self.cond_layer_idx and g is not None:  # 如果i等于条件层索引且g不为空
                g = self.spk_emb_linear(g.transpose(1, 2))  # 计算说话人嵌入
                g = g.transpose(1, 2)  # 转置g
                x = x + g  # 更新x
                x = x * x_mask  # 对x进行掩码
            y = self.attn_layers[i](x, x, attn_mask)  # 计算注意力层输出
            y = self.drop(y)  # 使用Dropout层
            x = self.norm_layers_1[i](x + y)  # 规范化
            y = self.ffn_layers[i](x, x_mask)  # 计算前馈神经网络层输出
            y = self.drop(y)  # 使用Dropout层
            x = self.norm_layers_2[i](x + y)  # 规范化
        x = x * x_mask  # 对x进行掩码
        return x  # 返回结果


class MultiHeadAttention(nn.Module):  # 定义MultiHeadAttention类，继承自nn.Module
    def __init__(  # 初始化函数
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
        super().__init__()  # 调用父类的初始化函数
        assert channels % n_heads == 0  # 断言条件

        self.channels = channels  # 设置通道数
        self.out_channels = out_channels  # 设置输出通道数
        self.n_heads = n_heads  # 设置头数
        self.p_dropout = p_dropout  # 设置dropout概率
        self.window_size = window_size  # 设置窗口大小
        self.heads_share = heads_share  # 设置头共享
        self.block_length = block_length  # 设置块长度
        self.proximal_bias = proximal_bias  # 设置近端偏置
        self.proximal_init = proximal_init  # 设置近端初始化
        self.attn = None  # 初始化注意力

        self.k_channels = channels // n_heads  # 计算k通道数
        self.conv_q = nn.Conv1d(channels, channels, 1)  # 初始化卷积层q
        self.conv_k = nn.Conv1d(channels, channels, 1)  # 初始化卷积层k
        self.conv_v = nn.Conv1d(channels, channels, 1)  # 初始化卷积层v
        self.conv_o = nn.Conv1d(channels, out_channels, 1)  # 初始化卷积层o
        self.drop = nn.Dropout(p_dropout)  # 初始化Dropout层

        if window_size is not None:  # 如果窗口大小不为空
            n_heads_rel = 1 if heads_share else n_heads  # 计算相对头数
            rel_stddev = self.k_channels**-0.5  # 计算相对标准差
            self.emb_rel_k = nn.Parameter(  # 初始化相对嵌入k
                torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels)
                * rel_stddev
            )
            self.emb_rel_v = nn.Parameter(  # 初始化相对嵌入v
                torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels)
                * rel_stddev
            )

        nn.init.xavier_uniform_(self.conv_q.weight)  # 初始化卷积层q的权重
        nn.init.xavier_uniform_(self.conv_k.weight)  # 初始化卷积层k的权重
        nn.init.xavier_uniform_(self.conv_v.weight)  # 初始化卷积层v的权重
        if proximal_init:  # 如果近端初始化为真
            with torch.no_grad():  # 不计算梯度
                self.conv_k.weight.copy_(self.conv_q.weight)  # 复制卷积层q的权重到卷积层k
                self.conv_k.bias.copy_(self.conv_q.bias)  # 复制卷积层q的偏置到卷积层k

    def forward(self, x, c, attn_mask=None):  # 前向传播函数，x为输入，c为条件，attn_mask为注意力掩码
        q = self.conv_q(x)  # 计算卷积层q的输出
        k = self.conv_k(c)  # 计算卷积层k的输出
        v = self.conv_v(c)  # 计算卷积层v的输出

        x, self.attn = self.attention(q, k, v, mask=attn_mask)  # 计算注意力

        x = self.conv_o(x)  # 计算卷积层o的输出
        return x  # 返回结果

    def attention(self, query, key, value, mask=None):  # 定义注意力函数，query为查询，key为键，value为值，mask为掩码
        # reshape [b, d, t] -> [b, n_h, t, d_k]
        b, d, t_s, t_t = (*key.size(), query.size(2))  # 获取维度
        query = query.view(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)  # 重塑query
        key = key.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)  # 重塑key
        value = value.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)  # 重塑value

        scores = torch.matmul(query / math.sqrt(self.k_channels), key.transpose(-2, -1))  # 计算得分
        if self.window_size is not None:  # 如果窗口大小不为空
            assert (  # 断言条件
                t_s == t_t
            ), "Relative attention is only available for self-attention."  # 如果条件不满足，抛出异常
            key_relative_embeddings = self._get_relative_embeddings(self.emb_rel_k, t_s)  # 获取相对嵌入
            rel_logits = self._matmul_with_relative_keys(  # 计算相对logits
                query / math.sqrt(self.k_channels), key_relative_embeddings
            )
            scores_local = self._relative_position_to_absolute_position(rel_logits)  # 计算相对位置到绝对位置
            scores = scores + scores_local  # 更新得分
        if self.proximal_bias:  # 如果近端偏置为真
            assert t_s == t_t, "Proximal bias is only available for self-attention."  # 如果条件不满足，抛出异常
            scores = scores + self._attention_bias_proximal(t_s).to(  # 更新得分
                device=scores.device, dtype=scores.dtype
            )
        if mask is not None:  # 如果掩码不为空
            scores = scores.masked_fill(mask == 0, -1e4)  # 对得分进行掩码
            if self.block_length is not None:  # 如果块长度不为空
                assert (  # 断言条件
                    t_s == t_t
                ), "Local attention is only available for self-attention."  # 如果条件不满足，抛出异常
                block_mask = (  # 计算块掩码
                    torch.ones_like(scores)
                    .triu(-self.block_length)
                    .tril(self.block_length)
                )
                scores = scores.masked_fill(block_mask == 0, -1e4)  # 对得分进行块掩码
        p_attn = F.softmax(scores, dim=-1)  # 计算注意力权重
        p_attn = self.drop(p_attn)  # 使用Dropout层
        output = torch.matmul(p_attn, value)  # 计算输出
        if self.window_size is not None:  # 如果窗口大小不为空
            relative_weights = self._absolute_position_to_relative_position(p_attn)  # 计算相对权重
            value_relative_embeddings = self._get_relative_embeddings(  # 获取相对嵌入
                self.emb_rel_v, t_s
            )
            output = output + self._matmul_with_relative_values(  # 计算相对值
                relative_weights, value_relative_embeddings
            )
        output = (  # 重塑输出
            output.transpose(2, 3).contiguous().view(b, d, t_t)
        )  # [b, n_h, t_t, d_k] -> [b, d, t_t]
        return output, p_attn  # 返回结果

    def _matmul_with_relative_values(self, x, y):  # 定义_matmul_with_relative_values函数
        """
        x: [b, h, l, m]
        y: [h or 1, m, d]
        ret: [b, h, l, d]
        """
        ret = torch.matmul(x, y.unsqueeze(0))  # 矩阵相乘
        return ret  # 返回结果

    def _matmul_with_relative_keys(self, x, y):  # 定义_matmul_with_relative_keys函数
        """
        x: [b, h, l, d]
        y: [h or 1, m, d]
        ret: [b, h, l, m]
        """
        ret = torch.matmul(x, y.unsqueeze(0).transpose(-2, -1))  # 矩阵相乘
        return ret  # 返回结果

    def _get_relative_embeddings(self, relative_embeddings, length):  # 定义_get_relative_embeddings函数
        max_relative_position = 2 * self.window_size + 1  # 计算最大相对位置
        # Pad first before slice to avoid using cond ops.
        pad_length = max(length - (self.window_size + 1), 0)  # 计算填充长度
        slice_start_position = max((self.window_size + 1) - length, 0)  # 计算切片起始位置
        slice_end_position = slice_start_position + 2 * length - 1  # 计算切片结束位置
        if pad_length > 0:  # 如果填充长度大于0
            padded_relative_embeddings = F.pad(  # 对相对嵌入进行填充
                relative_embeddings,
                commons.convert_pad_shape([[0, 0], [pad_length, pad_length], [0, 0]]),
            )
        else:  # 否则
            padded_relative_embeddings = relative_embeddings  # 不填充
        used_relative_embeddings = padded_relative_embeddings[  # 获取使用的相对嵌入
            :, slice_start_position:slice_end_position
        ]
        return used_relative_embeddings  # 返回结果

    def _relative_position_to_absolute_position(self, x):  # 定义_relative_position_to_absolute_position函数
        """
        x: [b, h, l, 2*l-1]
        ret: [b, h, l, l]
        """
        batch, heads, length, _ = x.size()  # 获取维度
        # Concat columns of pad to shift from relative to absolute indexing.
        x = F.pad(x, commons.convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, 1]]))  # 对x进行填充

        # Concat extra elements so to add up to shape (len+1, 2*len-1).
        x_flat = x.view([batch, heads, length * 2 * length])  # 重塑x
        x_flat = F.pad(  # 对x进行填
```

# `D:\src\Bert-VITS2\onnx_modules\V220\models_onnx.py`

```python
import math
import torch
from torch import nn
from torch.nn import functional as F

import commons
import modules
from . import attentions_onnx
from vector_quantize_pytorch import VectorQuantize

from torch.nn import Conv1d, ConvTranspose1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from commons import init_weights, get_padding
from .text import symbols, num_tones, num_languages


class DurationDiscriminator(nn.Module):  # vits2
    def __init__(
        self, in_channels, filter_channels, kernel_size, p_dropout, gin_channels=0
    ):
        super().__init__()

        self.in_channels = in_channels  # input channels
        self.filter_channels = filter_channels  # filter channels
        self.kernel_size = kernel_size  # kernel size
        self.p_dropout = p_dropout  # dropout probability
        self.gin_channels = gin_channels  # gin channels

        self.drop = nn.Dropout(p_dropout)  # dropout layer
        self.conv_1 = nn.Conv1d(  # 1D convolutional layer
            in_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_1 = modules.LayerNorm(filter_channels)  # layer normalization
        self.conv_2 = nn.Conv1d(  # 1D convolutional layer
            filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_2 = modules.LayerNorm(filter_channels)  # layer normalization
        self.dur_proj = nn.Conv1d(1, filter_channels, 1)  # 1D convolutional layer

        self.pre_out_conv_1 = nn.Conv1d(  # 1D convolutional layer
            2 * filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.pre_out_norm_1 = modules.LayerNorm(filter_channels)  # layer normalization
        self.pre_out_conv_2 = nn.Conv1d(  # 1D convolutional layer
            filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.pre_out_norm_2 = modules.LayerNorm(filter_channels)  # layer normalization

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, in_channels, 1)  # 1D convolutional layer

        self.output_layer = nn.Sequential(nn.Linear(filter_channels, 1), nn.Sigmoid())  # linear layer followed by sigmoid activation function

    def forward_probability(self, x, x_mask, dur, g=None):
        dur = self.dur_proj(dur)  # projection of duration
        x = torch.cat([x, dur], dim=1)  # concatenation of tensors
        x = self.pre_out_conv_1(x * x_mask)  # convolutional layer
        x = torch.relu(x)  # ReLU activation function
        x = self.pre_out_norm_1(x)  # layer normalization
        x = self.drop(x)  # dropout
        x = self.pre_out_conv_2(x * x_mask)  # convolutional layer
        x = torch.relu(x)  # ReLU activation function
        x = self.pre_out_norm_2(x)  # layer normalization
        x = self.drop(x)  # dropout
        x = x * x_mask  # element-wise multiplication
        x = x.transpose(1, 2)  # transpose
        output_prob = self.output_layer(x)  # output layer
        return output_prob  # return output probability

    def forward(self, x, x_mask, dur_r, dur_hat, g=None):
        x = torch.detach(x)  # detach tensor
        if g is not None:
            g = torch.detach(g)  # detach tensor
            x = x + self.cond(g)  # add conditional tensor
        x = self.conv_1(x * x_mask)  # convolutional layer
        x = torch.relu(x)  # ReLU activation function
        x = self.norm_1(x)  # layer normalization
        x = self.drop(x)  # dropout
        x = self.conv_2(x * x_mask)  # convolutional layer
        x = torch.relu(x)  # ReLU activation function
        x = self.norm_2(x)  # layer normalization
        x = self.drop(x)  # dropout

        output_probs = []
        for dur in [dur_r, dur_hat]:
            output_prob = self.forward_probability(x, x_mask, dur, g)  # forward probability
            output_probs.append(output_prob)  # append output probability

        return output_probs  # return output probabilities
```

# `D:\src\Bert-VITS2\onnx_modules\V220\__init__.py`

```python
from .text.symbols import symbols  # 从text模块中的symbols文件中导入symbols变量
from .models_onnx import SynthesizerTrn  # 从models_onnx模块中导入SynthesizerTrn类

__all__ = ["symbols", "SynthesizerTrn"]  # 定义__all__变量，包含symbols和SynthesizerTrn，用于模块导入时指定可导入的内容
```

# `D:\src\Bert-VITS2\onnx_modules\V220\text\symbols.py`

```python
punctuation = ["!", "?", "…", ",", ".", "'", "-"]  # 创建一个包含标点符号的列表
pu_symbols = punctuation + ["SP", "UNK"]  # 创建一个包含标点符号和特殊标记的列表
pad = "_"  # 创建一个填充标记

# chinese
zh_symbols = [  # 创建一个包含中文音节的列表
    "E",
    "En",
    ...
    "OO",
]
num_zh_tones = 6  # 设置中文音节的音调数量为6

# japanese
ja_symbols = [  # 创建一个包含日文音节的列表
    "N",
    "a",
    ...
    "zy",
]
num_ja_tones = 2  # 设置日文音节的音调数量为2

# English
en_symbols = [  # 创建一个包含英文音素的列表
    "aa",
    "ae",
    ...
    "zh",
]
num_en_tones = 4  # 设置英文音素的音调数量为4

# combine all symbols
normal_symbols = sorted(set(zh_symbols + ja_symbols + en_symbols))  # 合并所有音节并去重排序
symbols = [pad] + normal_symbols + pu_symbols  # 创建一个包含填充标记、所有音节和特殊标记的列表
sil_phonemes_ids = [symbols.index(i) for i in pu_symbols]  # 获取特殊标记在列表中的索引

# combine all tones
num_tones = num_zh_tones + num_ja_tones + num_en_tones  # 计算所有音节的总音调数量

# language maps
language_id_map = {"ZH": 0, "JP": 1, "EN": 2}  # 创建一个语言到ID的映射字典
num_languages = len(language_id_map.keys())  # 获取语言数量

language_tone_start_map = {  # 创建一个语言到音调起始位置的映射字典
    "ZH": 0,
    "JP": num_zh_tones,
    "EN": num_zh_tones + num_ja_tones,
}

if __name__ == "__main__":
    a = set(zh_symbols)  # 创建一个包含中文音节的集合
    b = set(en_symbols)  # 创建一个包含英文音素的集合
    print(sorted(a & b))  # 打印中英文共有的音节
```

# `D:\src\Bert-VITS2\onnx_modules\V220\text\__init__.py`

```python
from .symbols import *  # 从symbols模块中导入所有的变量和函数
```

# `D:\src\Bert-VITS2\onnx_modules\V220_novq_dev\attentions_onnx.py`

```python
import math  # 导入数学库
import torch  # 导入PyTorch
from torch import nn  # 从PyTorch中导入神经网络模块
from torch.nn import functional as F  # 从PyTorch中导入函数模块

import commons  # 导入自定义的commons模块
import logging  # 导入日志模块

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


class LayerNorm(nn.Module):  # 定义LayerNorm类，继承自nn.Module
    def __init__(self, channels, eps=1e-5):  # 初始化函数
        super().__init__()  # 调用父类的初始化函数
        self.channels = channels  # 设置channels属性
        self.eps = eps  # 设置eps属性

        self.gamma = nn.Parameter(torch.ones(channels))  # 初始化gamma参数
        self.beta = nn.Parameter(torch.zeros(channels))  # 初始化beta参数

    def forward(self, x):  # 前向传播函数
        x = x.transpose(1, -1)  # 调整x的维度
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)  # 对x进行Layer Norm
        return x.transpose(1, -1)  # 调整x的维度


@torch.jit.script  # 使用Torch Script装饰器
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):  # 定义fused_add_tanh_sigmoid_multiply函数
    n_channels_int = n_channels[0]  # 获取n_channels的第一个元素
    in_act = input_a + input_b  # 计算input_a和input_b的和
    t_act = torch.tanh(in_act[:, :n_channels_int, :])  # 计算tanh激活函数
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])  # 计算sigmoid激活函数
    acts = t_act * s_act  # 计算t_act和s_act的乘积
    return acts  # 返回结果


class Encoder(nn.Module):  # 定义Encoder类，继承自nn.Module
    def __init__(  # 初始化函数
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
        super().__init__()  # 调用父类的初始化函数
        self.hidden_channels = hidden_channels  # 设置hidden_channels属性
        self.filter_channels = filter_channels  # 设置filter_channels属性
        self.n_heads = n_heads  # 设置n_heads属性
        self.n_layers = n_layers  # 设置n_layers属性
        self.kernel_size = kernel_size  # 设置kernel_size属性
        self.p_dropout = p_dropout  # 设置p_dropout属性
        self.window_size = window_size  # 设置window_size属性
        self.cond_layer_idx = self.n_layers  # 设置cond_layer_idx属性
        if "gin_channels" in kwargs:  # 如果gin_channels在kwargs中
            self.gin_channels = kwargs["gin_channels"]  # 设置gin_channels属性
            if self.gin_channels != 0:  # 如果gin_channels不为0
                self.spk_emb_linear = nn.Linear(self.gin_channels, self.hidden_channels)  # 初始化spk_emb_linear
                self.cond_layer_idx = (  # 设置cond_layer_idx属性
                    kwargs["cond_layer_idx"] if "cond_layer_idx" in kwargs else 2
                )
                logging.debug(self.gin_channels, self.cond_layer_idx)  # 记录日志
                assert (  # 断言
                    self.cond_layer_idx < self.n_layers
                ), "cond_layer_idx should be less than n_layers"  # 如果不满足条件则抛出异常
        self.drop = nn.Dropout(p_dropout)  # 初始化Dropout层
        self.attn_layers = nn.ModuleList()  # 初始化注意力层列表
        self.norm_layers_1 = nn.ModuleList()  # 初始化LayerNorm层列表
        self.ffn_layers = nn.ModuleList()  # 初始化FeedForward层列表
        self.norm_layers_2 = nn.ModuleList()  # 初始化LayerNorm层列表
        for i in range(self.n_layers):  # 遍历n_layers
            self.attn_layers.append(  # 向attn_layers列表中添加元素
                MultiHeadAttention(  # 创建MultiHeadAttention实例
                    hidden_channels,
                    hidden_channels,
                    n_heads,
                    p_dropout=p_dropout,
                    window_size=window_size,
                )
            )
            self.norm_layers_1.append(LayerNorm(hidden_channels))  # 向norm_layers_1列表中添加元素
            self.ffn_layers.append(  # 向ffn_layers列表中添加元素
                FFN(  # 创建FFN实例
                    hidden_channels,
                    hidden_channels,
                    filter_channels,
                    kernel_size,
                    p_dropout=p_dropout,
                )
            )
            self.norm_layers_2.append(LayerNorm(hidden_channels))  # 向norm_layers_2列表中添加元素

    def forward(self, x, x_mask, g=None):  # 前向传播函数
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)  # 计算注意力掩码
        x = x * x_mask  # 对x进行掩码
        for i in range(self.n_layers):  # 遍历n_layers
            if i == self.cond_layer_idx and g is not None:  # 如果i等于cond_layer_idx并且g不为None
                g = self.spk_emb_linear(g.transpose(1, 2))  # 计算g
                g = g.transpose(1, 2)  # 调整g的维度
                x = x + g  # 更新x
                x = x * x_mask  # 对x进行掩码
            y = self.attn_layers[i](x, x, attn_mask)  # 计算注意力层输出
            y = self.drop(y)  # Dropout
            x = self.norm_layers_1[i](x + y)  # LayerNorm
            y = self.ffn_layers[i](x, x_mask)  # 计算FeedForward层输出
            y = self.drop(y)  # Dropout
            x = self.norm_layers_2[i](x + y)  # LayerNorm
        x = x * x_mask  # 对x进行掩码
        return x  # 返回结果


class MultiHeadAttention(nn.Module):  # 定义MultiHeadAttention类，继承自nn.Module
    def __init__(  # 初始化函数
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
        super().__init__()  # 调用父类的初始化函数
        assert channels % n_heads == 0  # 断言

        self.channels = channels  # 设置channels属性
        self.out_channels = out_channels  # 设置out_channels属性
        self.n_heads = n_heads  # 设置n_heads属性
        self.p_dropout = p_dropout  # 设置p_dropout属性
        self.window_size = window_size  # 设置window_size属性
        self.heads_share = heads_share  # 设置heads_share属性
        self.block_length = block_length  # 设置block_length属性
        self.proximal_bias = proximal_bias  # 设置proximal_bias属性
        self.proximal_init = proximal_init  # 设置proximal_init属性
        self.attn = None  # 初始化attn属性

        self.k_channels = channels // n_heads  # 计算k_channels
        self.conv_q = nn.Conv1d(channels, channels, 1)  # 初始化conv_q
        self.conv_k = nn.Conv1d(channels, channels, 1)  # 初始化conv_k
        self.conv_v = nn.Conv1d(channels, channels, 1)  # 初始化conv_v
        self.conv_o = nn.Conv1d(channels, out_channels, 1)  # 初始化conv_o
        self.drop = nn.Dropout(p_dropout)  # 初始化Dropout层

        if window_size is not None:  # 如果window_size不为None
            n_heads_rel = 1 if heads_share else n_heads  # 计算n_heads_rel
            rel_stddev = self.k_channels**-0.5  # 计算rel_stddev
            self.emb_rel_k = nn.Parameter(  # 初始化emb_rel_k
                torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels)
                * rel_stddev
            )
            self.emb_rel_v = nn.Parameter(  # 初始化emb_rel_v
                torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels)
                * rel_stddev
            )

        nn.init.xavier_uniform_(self.conv_q.weight)  # 初始化conv_q的权重
        nn.init.xavier_uniform_(self.conv_k.weight)  # 初始化conv_k的权重
        nn.init.xavier_uniform_(self.conv_v.weight)  # 初始化conv_v的权重
        if proximal_init:  # 如果proximal_init为True
            with torch.no_grad():  # 关闭梯度计算
                self.conv_k.weight.copy_(self.conv_q.weight)  # 复制conv_q的权重到conv_k
                self.conv_k.bias.copy_(self.conv_q.bias)  # 复制conv_q的偏置到conv_k

    def forward(self, x, c, attn_mask=None):  # 前向传播函数
        q = self.conv_q(x)  # 计算q
        k = self.conv_k(c)  # 计算k
        v = self.conv_v(c)  # 计算v

        x, self.attn = self.attention(q, k, v, mask=attn_mask)  # 计算注意力

        x = self.conv_o(x)  # 计算输出
        return x  # 返回结果

    def attention(self, query, key, value, mask=None):  # 定义attention函数
        # reshape [b, d, t] -> [b, n_h, t, d_k]
        b, d, t_s, t_t = (*key.size(), query.size(2))  # 获取维度信息
        query = query.view(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)  # 调整query的维度
        key = key.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)  # 调整key的维度
        value = value.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)  # 调整value的维度

        scores = torch.matmul(query / math.sqrt(self.k_channels), key.transpose(-2, -1))  # 计算得分
        if self.window_size is not None:  # 如果window_size不为None
            assert (  # 断言
                t_s == t_t
            ), "Relative attention is only available for self-attention."  # 如果不满足条件则抛出异常
            key_relative_embeddings = self._get_relative_embeddings(self.emb_rel_k, t_s)  # 获取相对位置编码
            rel_logits = self._matmul_with_relative_keys(  # 计算相对位置编码
                query / math.sqrt(self.k_channels), key_relative_embeddings
            )
            scores_local = self._relative_position_to_absolute_position(rel_logits)  # 计算相对位置编码
            scores = scores + scores_local  # 更新得分
        if self.proximal_bias:  # 如果proximal_bias为True
            assert (  # 断言
                t_s == t_t
            ), "Proximal bias is only available for self-attention."  # 如果不满足条件则抛出异常
            scores = scores + self._attention_bias_proximal(t_s).to(  # 添加近似偏置
                device=scores.device, dtype=scores.dtype
            )
        if mask is not None:  # 如果mask不为None
            scores = scores.masked_fill(mask == 0, -1e4)  # 对得分进行掩码
            if self.block_length is not None:  # 如果block_length不为None
                assert (  # 断言
                    t_s == t_t
                ), "Local attention is only available for self-attention."  # 如果不满足条件则抛出异常
                block_mask = (  # 创建块掩码
                    torch.ones_like(scores)
                    .triu(-self.block_length)
                    .tril(self.block_length)
                )
                scores = scores.masked_fill(block_mask == 0, -1e4)  # 对得分进行掩码
        p_attn = F.softmax(scores, dim=-1)  # 计算注意力权重
        p_attn = self.drop(p_attn)  # Dropout
        output = torch.matmul(p_attn, value)  # 计算输出
        if self.window_size is not None:  # 如果window_size不为None
            relative_weights = self._absolute_position_to_relative_position(p_attn)  # 计算相对权重
            value_relative_embeddings = self._get_relative_embeddings(  # 获取相对位置编码
                self.emb_rel_v, t_s
            )
            output = output + self._matmul_with_relative_values(  # 计算相对位置编码
                relative_weights, value_relative_embeddings
            )
        output = (  # 调整输出��维度
            output.transpose(2, 3).contiguous().view(b, d, t_t)
        )  # [b, n_h, t_t, d_k] -> [b, d, t_t]
        return output, p_attn  # 返回结果

    def _matmul_with_relative_values(self, x, y):  # 定义_matmul_with_relative_values函数
        """
        x: [b, h, l, m]
        y: [h or 1, m, d]
        ret: [b, h, l, d]
        """
        ret = torch.matmul(x, y.unsqueeze(0))  # 矩阵相乘
        return ret  # 返回结果

    def _matmul_with_relative_keys(self, x, y):  # 定义_matmul_with_relative_keys函数
        """
        x: [b, h, l, d]
        y: [h or 1, m, d]
        ret: [b, h, l, m]
        """
        ret = torch.matmul(x, y.unsqueeze(0).transpose(-2, -1))  # 矩阵相乘
        return ret  # 返回结果

    def _get_relative_embeddings(self, relative_embeddings, length):  # 定义_get_relative_embeddings函数
        max_relative_position = 2 * self.window_size + 1  # 计算最大相对位置
        # Pad first before slice to avoid using cond ops.
        pad_length = max(length - (self.window_size + 1), 0)  # 计算填充长度
        slice_start_position = max((self.window_size + 1) - length, 0)  # 计算切片起始位置
        slice_end_position = slice_start_position + 2 * length - 1  # 计算切片结束位置
        if pad_length > 0:  # 如果pad_length大于0
            padded_relative_embeddings = F.pad(  # 对relative_embeddings进行填充
                relative_embeddings,
                commons.convert_pad_shape([[0, 0], [pad_length, pad_length], [0, 0]]),
            )
        else:  # 否则
            padded_relative_embeddings = relative_embeddings  # 不进行填充
        used_relative_embeddings = padded_relative_embeddings[  # 获取使用的相对位置编码
            :, slice_start_position:slice_end_position
        ]
        return used_relative_embeddings  # 返回结果

    def _relative_position_to_absolute_position(self, x):  # 定义_relative_position_to_absolute_position函数
        """
        x: [b, h, l, 2*l-1]
        ret: [b, h, l, l]
        """
        batch, heads, length, _ = x.size()  # 获取维度信息
        # Concat columns of pad to shift from relative to absolute indexing.
        x = F.pad(x, commons.convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, 1]]))  # 对x进行填充

        # Concat extra elements so to add up to shape (len+1, 2*len-1).
        x_flat = x.view([batch, heads, length * 2 * length])  # 调整x的维度
        x_flat = F.pad(  # 对x_flat进行填充
            x_flat, commons.convert_pad_shape([[0, 0], [0, 0], [0, length - 1]])
        )

        # Reshape and slice out the padded elements.
        x_final = x_flat.view([batch, heads, length + 1, 2 * length - 1])[
            :, :, :length, length - 1 :
        ]  # 调整x_final的维度
        return x_final  # 返回结果

    def _absolute_position_to_relative_position(self, x):  # 定义_absolute_position_to_relative_position函数
        """
        x: [b, h, l, l]
        ret: [b, h, l, 2*l-1]
        """
        batch, heads, length, _ = x.size()  # 获取维度信息
        # padd along column
        x = F.pad
```

# `D:\src\Bert-VITS2\onnx_modules\V220_novq_dev\models_onnx.py`

```python
import math
import torch
from torch import nn
from torch.nn import functional as F

import commons
import modules
from . import attentions_onnx

from torch.nn import Conv1d, ConvTranspose1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from commons import init_weights, get_padding
from .text import symbols, num_tones, num_languages


class DurationDiscriminator(nn.Module):  # vits2
    def __init__(
        self, in_channels, filter_channels, kernel_size, p_dropout, gin_channels=0
    ):
        super().__init__()

        self.in_channels = in_channels  # input channels
        self.filter_channels = filter_channels  # filter channels
        self.kernel_size = kernel_size  # kernel size
        self.p_dropout = p_dropout  # dropout probability
        self.gin_channels = gin_channels  # gin channels

        self.drop = nn.Dropout(p_dropout)  # dropout layer
        self.conv_1 = nn.Conv1d(  # 1D convolutional layer
            in_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_1 = modules.LayerNorm(filter_channels)  # layer normalization
        self.conv_2 = nn.Conv1d(  # 1D convolutional layer
            filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_2 = modules.LayerNorm(filter_channels)  # layer normalization
        self.dur_proj = nn.Conv1d(1, filter_channels, 1)  # 1D convolutional layer

        self.pre_out_conv_1 = nn.Conv1d(  # 1D convolutional layer
            2 * filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.pre_out_norm_1 = modules.LayerNorm(filter_channels)  # layer normalization
        self.pre_out_conv_2 = nn.Conv1d(  # 1D convolutional layer
            filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.pre_out_norm_2 = modules.LayerNorm(filter_channels)  # layer normalization

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, in_channels, 1)  # 1D convolutional layer

        self.output_layer = nn.Sequential(nn.Linear(filter_channels, 1), nn.Sigmoid())  # linear layer followed by sigmoid activation function

    def forward_probability(self, x, x_mask, dur, g=None):
        dur = self.dur_proj(dur)  # projection of duration
        x = torch.cat([x, dur], dim=1)  # concatenation of input and duration
        x = self.pre_out_conv_1(x * x_mask)  # 1D convolutional layer
        x = torch.relu(x)  # ReLU activation function
        x = self.pre_out_norm_1(x)  # layer normalization
        x = self.drop(x)  # dropout
        x = self.pre_out_conv_2(x * x_mask)  # 1D convolutional layer
        x = torch.relu(x)  # ReLU activation function
        x = self.pre_out_norm_2(x)  # layer normalization
        x = self.drop(x)  # dropout
        x = x * x_mask  # masking
        x = x.transpose(1, 2)  # transpose
        output_prob = self.output_layer(x)  # output layer
        return output_prob  # return output probability

    def forward(self, x, x_mask, dur_r, dur_hat, g=None):
        x = torch.detach(x)  # detach tensor
        if g is not None:
            g = torch.detach(g)  # detach tensor
            x = x + self.cond(g)  # add conditional tensor
        x = self.conv_1(x * x_mask)  # 1D convolutional layer
        x = torch.relu(x)  # ReLU activation function
        x = self.norm_1(x)  # layer normalization
        x = self.drop(x)  # dropout
        x = self.conv_2(x * x_mask)  # 1D convolutional layer
        x = torch.relu(x)  # ReLU activation function
        x = self.norm_2(x)  # layer normalization
        x = self.drop(x)  # dropout

        output_probs = []
        for dur in [dur_r, dur_hat]:
            output_prob = self.forward_probability(x, x_mask, dur, g)  # forward probability
            output_probs.append(output_prob)  # append output probability

        return output_probs  # return output probabilities
```

# `D:\src\Bert-VITS2\onnx_modules\V220_novq_dev\__init__.py`

```python
from .text.symbols import symbols  # 从text模块中的symbols文件中导入symbols变量
from .models_onnx import SynthesizerTrn  # 从models_onnx模块中导入SynthesizerTrn类

__all__ = ["symbols", "SynthesizerTrn"]  # 定义__all__变量，包含symbols和SynthesizerTrn，用于模块导入时指定可导入的内容
```

# `D:\src\Bert-VITS2\onnx_modules\V220_novq_dev\text\symbols.py`

```python
punctuation = ["!", "?", "…", ",", ".", "'", "-"]  # 创建一个包含标点符号的列表
pu_symbols = punctuation + ["SP", "UNK"]  # 创建一个包含标点符号和特殊标记的列表
pad = "_"  # 创建一个填充标记

# chinese
zh_symbols = [  # 创建一个包含中文音节的列表
    "E",
    "En",
    ...
    "OO",
]
num_zh_tones = 6  # 设置中文音节的音调数量为6

# japanese
ja_symbols = [  # 创建一个包含日文音节的列表
    "N",
    "a",
    ...
    "zy",
]
num_ja_tones = 2  # 设置日文音节的音调数量为2

# English
en_symbols = [  # 创建一个包含英文音素的列表
    "aa",
    "ae",
    ...
    "zh",
]
num_en_tones = 4  # 设置英文音素的音调数量为4

# combine all symbols
normal_symbols = sorted(set(zh_symbols + ja_symbols + en_symbols))  # 合并所有音节并去重排序
symbols = [pad] + normal_symbols + pu_symbols  # 创建一个包含填充标记、所有音节和特殊标记的列表
sil_phonemes_ids = [symbols.index(i) for i in pu_symbols]  # 获取特殊标记在列表中的索引

# combine all tones
num_tones = num_zh_tones + num_ja_tones + num_en_tones  # 计算所有音节的总音调数量

# language maps
language_id_map = {"ZH": 0, "JP": 1, "EN": 2}  # 创建一个语言到ID的映射字典
num_languages = len(language_id_map.keys())  # 获取语言数量

language_tone_start_map = {  # 创建一个语言到音调起始位置的映射字典
    "ZH": 0,
    "JP": num_zh_tones,
    "EN": num_zh_tones + num_ja_tones,
}

if __name__ == "__main__":
    a = set(zh_symbols)  # 创建一个包含中文音节的集合
    b = set(en_symbols)  # 创建一个包含英文音素的集合
    print(sorted(a & b))  # 打印中英文共有的音节
```

# `D:\src\Bert-VITS2\onnx_modules\V220_novq_dev\text\__init__.py`

```python
from .symbols import *  # 从symbols模块中导入所有的变量和函数
```

# `D:\src\Bert-VITS2\onnx_modules\V220_OnnxInference\__init__.py`

```python
import numpy as np  # 导入numpy库
import onnxruntime as ort  # 导入onnxruntime库

def convert_pad_shape(pad_shape):  # 定义函数convert_pad_shape
    layer = pad_shape[::-1]  # 反转pad_shape
    pad_shape = [item for sublist in layer for item in sublist]  # 将layer展开
    return pad_shape  # 返回pad_shape

def sequence_mask(length, max_length=None):  # 定义函数sequence_mask
    if max_length is None:  # 如果max_length为空
        max_length = length.max()  # max_length取length的最大值
    x = np.arange(max_length, dtype=length.dtype)  # 生成长度为max_length的数组x
    return np.expand_dims(x, 0) < np.expand_dims(length, 1)  # 返回x和length的比较结果

def generate_path(duration, mask):  # 定义函数generate_path
    """
    duration: [b, 1, t_x]
    mask: [b, 1, t_y, t_x]
    """
    b, _, t_y, t_x = mask.shape  # 获取mask的形状
    cum_duration = np.cumsum(duration, -1)  # 对duration进行累加

    cum_duration_flat = cum_duration.reshape(b * t_x)  # 将cum_duration展平
    path = sequence_mask(cum_duration_flat, t_y)  # 生成path
    path = path.reshape(b, t_x, t_y)  # 调整path的形状
    path = path ^ np.pad(path, ((0, 0), (1, 0), (0, 0)))[:, :-1]  # 对path进行异或操作
    path = np.expand_dims(path, 1).transpose(0, 1, 3, 2)  # 调整path的形状
    return path  # 返回path

class OnnxInferenceSession:  # 定义类OnnxInferenceSession
    def __init__(self, path, Providers=["CPUExecutionProvider"]):  # 定义初始化方法
        self.enc = ort.InferenceSession(path["enc"], providers=Providers)  # 初始化self.enc
        self.emb_g = ort.InferenceSession(path["emb_g"], providers=Providers)  # 初始化self.emb_g
        self.dp = ort.InferenceSession(path["dp"], providers=Providers)  # 初始化self.dp
        self.sdp = ort.InferenceSession(path["sdp"], providers=Providers)  # 初始化self.sdp
        self.flow = ort.InferenceSession(path["flow"], providers=Providers)  # 初始化self.flow
        self.dec = ort.InferenceSession(path["dec"], providers=Providers)  # 初始化self.dec

    def __call__(  # 定义__call__方法
        self,
        seq,
        tone,
        language,
        bert_zh,
        bert_jp,
        bert_en,
        emo,
        sid,
        seed=114514,
        seq_noise_scale=0.8,
        sdp_noise_scale=0.6,
        length_scale=1.0,
        sdp_ratio=0.0,
    ):
        if seq.ndim == 1:  # 如果seq的维度为1
            seq = np.expand_dims(seq, 0)  # 将seq扩展为二维数组
        if tone.ndim == 1:  # 如果tone的维度为1
            tone = np.expand_dims(tone, 0)  # 将tone扩展为二维数组
        if language.ndim == 1:  # 如果language的维度为1
            language = np.expand_dims(language, 0)  # 将language扩展为二维数组
        assert (seq.ndim == 2, tone.ndim == 2, language.ndim == 2)  # 断言seq、tone、language的维度为2
        g = self.emb_g.run(  # 运行self.emb_g
            None,
            {
                "sid": sid.astype(np.int64),  # 传入参数sid
            },
        )[0]  # 获取返回值
        g = np.expand_dims(g, -1)  # 将g扩展为三维数组
        enc_rtn = self.enc.run(  # 运行self.enc
            None,
            {
                "x": seq.astype(np.int64),  # 传入参数seq
                "t": tone.astype(np.int64),  # 传入参数tone
                "language": language.astype(np.int64),  # 传入参数language
                "bert_0": bert_zh.astype(np.float32),  # 传入参数bert_zh
                "bert_1": bert_jp.astype(np.float32),  # 传入参数bert_jp
                "bert_2": bert_en.astype(np.float32),  # 传入参数bert_en
                "emo": emo.astype(np.float32),  # 传入参数emo
                "g": g.astype(np.float32),  # 传入参数g
            },
        )
        x, m_p, logs_p, x_mask = enc_rtn[0], enc_rtn[1], enc_rtn[2], enc_rtn[3]  # 获取enc_rtn的返回值
        np.random.seed(seed)  # 设置随机种子
        zinput = np.random.randn(x.shape[0], 2, x.shape[2]) * sdp_noise_scale  # 生成zinput
        logw = self.sdp.run(  # 运行self.sdp
            None, {"x": x, "x_mask": x_mask, "zin": zinput.astype(np.float32), "g": g}
        )[0] * (sdp_ratio) + self.dp.run(None, {"x": x, "x_mask": x_mask, "g": g})[  # 获取logw
            0
        ] * (
            1 - sdp_ratio
        )
        w = np.exp(logw) * x_mask * length_scale  # 计算w
        w_ceil = np.ceil(w)  # 对w进行向上取整
        y_lengths = np.clip(np.sum(w_ceil, (1, 2)), a_min=1.0, a_max=100000).astype(  # 计算y_lengths
            np.int64
        )
        y_mask = np.expand_dims(sequence_mask(y_lengths, None), 1)  # 生成y_mask
        attn_mask = np.expand_dims(x_mask, 2) * np.expand_dims(y_mask, -1)  # 生成attn_mask
        attn = generate_path(w_ceil, attn_mask)  # 生成attn
        m_p = np.matmul(attn.squeeze(1), m_p.transpose(0, 2, 1)).transpose(  # 计算m_p
            0, 2, 1
        )  # [b, t', t], [b, t, d] -> [b, d, t']
        logs_p = np.matmul(attn.squeeze(1), logs_p.transpose(0, 2, 1)).transpose(  # 计算logs_p
            0, 2, 1
        )  # [b, t', t], [b, t, d] -> [b, d, t']

        z_p = (  # 计算z_p
            m_p
            + np.random.randn(m_p.shape[0], m_p.shape[1], m_p.shape[2])
            * np.exp(logs_p)
            * seq_noise_scale
        )

        z = self.flow.run(  # 运行self.flow
            None,
            {
                "z_p": z_p.astype(np.float32),  # 传入参数z_p
                "y_mask": y_mask.astype(np.float32),  # 传入参数y_mask
                "g": g,  # 传入参数g
            },
        )[0]  # 获取返回值

        return self.dec.run(None, {"z_in": z.astype(np.float32), "g": g})[0]  # 运行self.dec并返回结果
```

# `D:\src\Bert-VITS2\onnx_modules\V230\attentions_onnx.py`

```python
import math  # 导入数学库
import torch  # 导入PyTorch
from torch import nn  # 从PyTorch中导入神经网络模块
from torch.nn import functional as F  # 从PyTorch中导入函数模块

import commons  # 导入自定义的commons模块
import logging  # 导入日志模块

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


class LayerNorm(nn.Module):  # 定义LayerNorm类，继承自nn.Module
    def __init__(self, channels, eps=1e-5):  # 初始化函数
        super().__init__()  # 调用父类的初始化函数
        self.channels = channels  # 设置channels属性
        self.eps = eps  # 设置eps属性

        self.gamma = nn.Parameter(torch.ones(channels))  # 初始化gamma参数
        self.beta = nn.Parameter(torch.zeros(channels))  # 初始化beta参数

    def forward(self, x):  # 前向传播函数
        x = x.transpose(1, -1)  # 调整x的维度
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)  # 对x进行Layer Norm
        return x.transpose(1, -1)  # 调整x的维度


@torch.jit.script  # 使用Torch Script装饰器
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):  # 定义fused_add_tanh_sigmoid_multiply函数
    n_channels_int = n_channels[0]  # 获取n_channels的第一个元素
    in_act = input_a + input_b  # 计算input_a和input_b的和
    t_act = torch.tanh(in_act[:, :n_channels_int, :])  # 计算tanh激活函数
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])  # 计算sigmoid激活函数
    acts = t_act * s_act  # 计算t_act和s_act的乘积
    return acts  # 返回结果


class Encoder(nn.Module):  # 定义Encoder类，继承自nn.Module
    def __init__(  # 初始化函数
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
        super().__init__()  # 调用父类的初始化函数
        self.hidden_channels = hidden_channels  # 设置hidden_channels属性
        self.filter_channels = filter_channels  # 设置filter_channels属性
        self.n_heads = n_heads  # 设置n_heads属性
        self.n_layers = n_layers  # 设置n_layers属性
        self.kernel_size = kernel_size  # 设置kernel_size属性
        self.p_dropout = p_dropout  # 设置p_dropout属性
        self.window_size = window_size  # 设置window_size属性
        self.cond_layer_idx = self.n_layers  # 设置cond_layer_idx属性
        if "gin_channels" in kwargs:  # 如果gin_channels在kwargs中
            self.gin_channels = kwargs["gin_channels"]  # 设置gin_channels属性
            if self.gin_channels != 0:  # 如果gin_channels不为0
                self.spk_emb_linear = nn.Linear(self.gin_channels, self.hidden_channels)  # 初始化spk_emb_linear
                self.cond_layer_idx = (  # 设置cond_layer_idx属性
                    kwargs["cond_layer_idx"] if "cond_layer_idx" in kwargs else 2
                )
                logging.debug(self.gin_channels, self.cond_layer_idx)  # 记录日志
                assert (  # 断言
                    self.cond_layer_idx < self.n_layers
                ), "cond_layer_idx should be less than n_layers"  # 如果不满足条件则抛出异常
        self.drop = nn.Dropout(p_dropout)  # 初始化Dropout层
        self.attn_layers = nn.ModuleList()  # 初始化注意力层列表
        self.norm_layers_1 = nn.ModuleList()  # 初始化LayerNorm层列表
        self.ffn_layers = nn.ModuleList()  # 初始化FeedForward层列表
        self.norm_layers_2 = nn.ModuleList()  # 初始化LayerNorm层列表
        for i in range(self.n_layers):  # 遍历n_layers
            self.attn_layers.append(  # 向attn_layers列表中添加元素
                MultiHeadAttention(  # 创建MultiHeadAttention实例
                    hidden_channels,
                    hidden_channels,
                    n_heads,
                    p_dropout=p_dropout,
                    window_size=window_size,
                )
            )
            self.norm_layers_1.append(LayerNorm(hidden_channels))  # 向norm_layers_1列表中添加元素
            self.ffn_layers.append(  # 向ffn_layers列表中添加元素
                FFN(  # 创建FFN实例
                    hidden_channels,
                    hidden_channels,
                    filter_channels,
                    kernel_size,
                    p_dropout=p_dropout,
                )
            )
            self.norm_layers_2.append(LayerNorm(hidden_channels))  # 向norm_layers_2列表中添加元素

    def forward(self, x, x_mask, g=None):  # 前向传播函数
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)  # 计算注意力掩码
        x = x * x_mask  # 对x进行掩码
        for i in range(self.n_layers):  # 遍历n_layers
            if i == self.cond_layer_idx and g is not None:  # 如果i等于cond_layer_idx并且g不为None
                g = self.spk_emb_linear(g.transpose(1, 2))  # 计算g
                g = g.transpose(1, 2)  # 调整g的维度
                x = x + g  # 更新x
                x = x * x_mask  # 对x进行掩码
            y = self.attn_layers[i](x, x, attn_mask)  # 计算注意力层输出
            y = self.drop(y)  # Dropout
            x = self.norm_layers_1[i](x + y)  # LayerNorm
            y = self.ffn_layers[i](x, x_mask)  # 计算FeedForward层输出
            y = self.drop(y)  # Dropout
            x = self.norm_layers_2[i](x + y)  # LayerNorm
        x = x * x_mask  # 对x进行掩码
        return x  # 返回结果


class MultiHeadAttention(nn.Module):  # 定义MultiHeadAttention类，继承自nn.Module
    def __init__(  # 初始化函数
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
        super().__init__()  # 调用父类的初始化函数
        assert channels % n_heads == 0  # 断言

        self.channels = channels  # 设置channels属性
        self.out_channels = out_channels  # 设置out_channels属性
        self.n_heads = n_heads  # 设置n_heads属性
        self.p_dropout = p_dropout  # 设置p_dropout属性
        self.window_size = window_size  # 设置window_size属性
        self.heads_share = heads_share  # 设置heads_share属性
        self.block_length = block_length  # 设置block_length属性
        self.proximal_bias = proximal_bias  # 设置proximal_bias属性
        self.proximal_init = proximal_init  # 设置proximal_init属性
        self.attn = None  # 初始化attn属性

        self.k_channels = channels // n_heads  # 计算k_channels
        self.conv_q = nn.Conv1d(channels, channels, 1)  # 初始化conv_q
        self.conv_k = nn.Conv1d(channels, channels, 1)  # 初始化conv_k
        self.conv_v = nn.Conv1d(channels, channels, 1)  # 初始化conv_v
        self.conv_o = nn.Conv1d(channels, out_channels, 1)  # 初始化conv_o
        self.drop = nn.Dropout(p_dropout)  # 初始化Dropout层

        if window_size is not None:  # 如果window_size不为None
            n_heads_rel = 1 if heads_share else n_heads  # 计算n_heads_rel
            rel_stddev = self.k_channels**-0.5  # 计算rel_stddev
            self.emb_rel_k = nn.Parameter(  # 初始化emb_rel_k
                torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels)
                * rel_stddev
            )
            self.emb_rel_v = nn.Parameter(  # 初始化emb_rel_v
                torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels)
                * rel_stddev
            )

        nn.init.xavier_uniform_(self.conv_q.weight)  # 初始化conv_q的权重
        nn.init.xavier_uniform_(self.conv_k.weight)  # 初始化conv_k的权重
        nn.init.xavier_uniform_(self.conv_v.weight)  # 初始化conv_v的权重
        if proximal_init:  # 如果proximal_init为True
            with torch.no_grad():  # 关闭梯度计算
                self.conv_k.weight.copy_(self.conv_q.weight)  # 复制conv_q的权重到conv_k
                self.conv_k.bias.copy_(self.conv_q.bias)  # 复制conv_q的偏置到conv_k

    def forward(self, x, c, attn_mask=None):  # 前向传播函数
        q = self.conv_q(x)  # 计算q
        k = self.conv_k(c)  # 计算k
        v = self.conv_v(c)  # 计算v

        x, self.attn = self.attention(q, k, v, mask=attn_mask)  # 计算注意力

        x = self.conv_o(x)  # 计算输出
        return x  # 返回结果

    def attention(self, query, key, value, mask=None):  # 定义attention函数
        # reshape [b, d, t] -> [b, n_h, t, d_k]
        b, d, t_s, t_t = (*key.size(), query.size(2))  # 获取维度信息
        query = query.view(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)  # 调整query的维度
        key = key.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)  # 调整key的维度
        value = value.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)  # 调整value的维度

        scores = torch.matmul(query / math.sqrt(self.k_channels), key.transpose(-2, -1))  # 计算得分
        if self.window_size is not None:  # 如果window_size不为None
            assert (  # 断言
                t_s == t_t
            ), "Relative attention is only available for self-attention."  # 如果不满足条件则抛出异常
            key_relative_embeddings = self._get_relative_embeddings(self.emb_rel_k, t_s)  # 获取相对位置编码
            rel_logits = self._matmul_with_relative_keys(  # 计算相对位置编码
                query / math.sqrt(self.k_channels), key_relative_embeddings
            )
            scores_local = self._relative_position_to_absolute_position(rel_logits)  # 计算相对位置编码
            scores = scores + scores_local  # 更新得分
        if self.proximal_bias:  # 如果proximal_bias为True
            assert (  # 断言
                t_s == t_t
            ), "Proximal bias is only available for self-attention."  # 如果不满足条件则抛出异常
            scores = scores + self._attention_bias_proximal(t_s).to(  # 添加近似偏置
                device=scores.device, dtype=scores.dtype
            )
        if mask is not None:  # 如果mask不为None
            scores = scores.masked_fill(mask == 0, -1e4)  # 对得分进行掩码
            if self.block_length is not None:  # 如果block_length不为None
                assert (  # 断言
                    t_s == t_t
                ), "Local attention is only available for self-attention."  # 如果不满足条件则抛出异常
                block_mask = (  # 创建块掩码
                    torch.ones_like(scores)
                    .triu(-self.block_length)
                    .tril(self.block_length)
                )
                scores = scores.masked_fill(block_mask == 0, -1e4)  # 对得分进行掩码
        p_attn = F.softmax(scores, dim=-1)  # 计算注意力权重
        p_attn = self.drop(p_attn)  # Dropout
        output = torch.matmul(p_attn, value)  # 计算输出
        if self.window_size is not None:  # 如果window_size不为None
            relative_weights = self._absolute_position_to_relative_position(p_attn)  # 计算相对权重
            value_relative_embeddings = self._get_relative_embeddings(  # 获取相对位置编码
                self.emb_rel_v, t_s
            )
            output = output + self._matmul_with_relative_values(  # 计算相对位置编码
                relative_weights, value_relative_embeddings
            )
        output = (  # 调整输出��维度
            output.transpose(2, 3).contiguous().view(b, d, t_t)
        )  # [b, n_h, t_t, d_k] -> [b, d, t_t]
        return output, p_attn  # 返回结果

    def _matmul_with_relative_values(self, x, y):  # 定义_matmul_with_relative_values函数
        """
        x: [b, h, l, m]
        y: [h or 1, m, d]
        ret: [b, h, l, d]
        """
        ret = torch.matmul(x, y.unsqueeze(0))  # 矩阵相乘
        return ret  # 返回结果

    def _matmul_with_relative_keys(self, x, y):  # 定义_matmul_with_relative_keys函数
        """
        x: [b, h, l, d]
        y: [h or 1, m, d]
        ret: [b, h, l, m]
        """
        ret = torch.matmul(x, y.unsqueeze(0).transpose(-2, -1))  # 矩阵相乘
        return ret  # 返回结果

    def _get_relative_embeddings(self, relative_embeddings, length):  # 定义_get_relative_embeddings函数
        max_relative_position = 2 * self.window_size + 1  # 计算最大相对位置
        # Pad first before slice to avoid using cond ops.
        pad_length = max(length - (self.window_size + 1), 0)  # 计算填充长度
        slice_start_position = max((self.window_size + 1) - length, 0)  # 计算切片起始位置
        slice_end_position = slice_start_position + 2 * length - 1  # 计算切片结束位置
        if pad_length > 0:  # 如果pad_length大于0
            padded_relative_embeddings = F.pad(  # 对relative_embeddings进行填充
                relative_embeddings,
                commons.convert_pad_shape([[0, 0], [pad_length, pad_length], [0, 0]]),
            )
        else:  # 否则
            padded_relative_embeddings = relative_embeddings  # 不进行填充
        used_relative_embeddings = padded_relative_embeddings[  # 获取使用的相对位置编码
            :, slice_start_position:slice_end_position
        ]
        return used_relative_embeddings  # 返回结果

    def _relative_position_to_absolute_position(self, x):  # 定义_relative_position_to_absolute_position函数
        """
        x: [b, h, l, 2*l-1]
        ret: [b, h, l, l]
        """
        batch, heads, length, _ = x.size()  # 获取维度信息
        # Concat columns of pad to shift from relative to absolute indexing.
        x = F.pad(x, commons.convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, 1]]))  # 对x进行填充

        # Concat extra elements so to add up to shape (len+1, 2*len-1).
        x_flat = x.view([batch, heads, length * 2 * length])  # 调整x的维度
        x_flat = F.pad(  # 对x_flat进行填充
            x_flat, commons.convert_pad_shape([[0, 0], [0, 0], [0, length - 1]])
        )

        # Reshape and slice out the padded elements.
        x_final = x_flat.view([batch, heads, length + 1, 2 * length - 1])[
            :, :, :length, length - 1 :
        ]  # 调整x_final的维度
        return x_final  # 返回结果

    def _absolute_position_to_relative_position(self, x):  # 定义_absolute_position_to_relative_position函数
        """
        x: [b, h, l, l]
        ret: [b, h, l, 2*l-1]
        """
        batch, heads, length, _ = x.size()  # 获取维度信息
        # padd along column
        x = F.pad
```