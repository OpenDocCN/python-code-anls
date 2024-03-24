# `.\lucidrains\vit-pytorch\vit_pytorch\cct.py`

```py
# 导入 torch 库
import torch
# 从 torch 库中导入 nn 模块和 einsum 函数
from torch import nn, einsum
# 从 torch.nn 模块中导入 F 函数
import torch.nn.functional as F

# 从 einops 库中导入 rearrange 和 repeat 函数
from einops import rearrange, repeat

# 定义辅助函数

# 判断变量是否存在的函数
def exists(val):
    return val is not None

# 如果变量存在则返回该变量，否则返回默认值的函数
def default(val, d):
    return val if exists(val) else d

# 将输入转换为元组的函数
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# CCT 模型

# 定义导出的 CCT 模型名称列表
__all__ = ['cct_2', 'cct_4', 'cct_6', 'cct_7', 'cct_8', 'cct_14', 'cct_16']

# 定义创建不同层数 CCT 模型的函数
def cct_2(*args, **kwargs):
    return _cct(num_layers=2, num_heads=2, mlp_ratio=1, embedding_dim=128,
                *args, **kwargs)

def cct_4(*args, **kwargs):
    return _cct(num_layers=4, num_heads=2, mlp_ratio=1, embedding_dim=128,
                *args, **kwargs)

def cct_6(*args, **kwargs):
    return _cct(num_layers=6, num_heads=4, mlp_ratio=2, embedding_dim=256,
                *args, **kwargs)

def cct_7(*args, **kwargs):
    return _cct(num_layers=7, num_heads=4, mlp_ratio=2, embedding_dim=256,
                *args, **kwargs)

def cct_8(*args, **kwargs):
    return _cct(num_layers=8, num_heads=4, mlp_ratio=2, embedding_dim=256,
                *args, **kwargs)

def cct_14(*args, **kwargs):
    return _cct(num_layers=14, num_heads=6, mlp_ratio=3, embedding_dim=384,
                *args, **kwargs)

def cct_16(*args, **kwargs):
    return _cct(num_layers=16, num_heads=6, mlp_ratio=3, embedding_dim=384,
                *args, **kwargs)

# 创建 CCT 模型的内部函数
def _cct(num_layers, num_heads, mlp_ratio, embedding_dim,
         kernel_size=3, stride=None, padding=None,
         *args, **kwargs):
    # 计算默认的步长和填充值
    stride = default(stride, max(1, (kernel_size // 2) - 1))
    padding = default(padding, max(1, (kernel_size // 2)))

    # 返回 CCT 模型
    return CCT(num_layers=num_layers,
               num_heads=num_heads,
               mlp_ratio=mlp_ratio,
               embedding_dim=embedding_dim,
               kernel_size=kernel_size,
               stride=stride,
               padding=padding,
               *args, **kwargs)

# 位置编码

# 创建正弦位置编码的函数
def sinusoidal_embedding(n_channels, dim):
    pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                            for p in range(n_channels)])
    pe[:, 0::2] = torch.sin(pe[:, 0::2])
    pe[:, 1::2] = torch.cos(pe[:, 1::2])
    return rearrange(pe, '... -> 1 ...')

# 模块

# 定义注意力机制模块
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, attention_dropout=0.1, projection_dropout=0.1):
        super().__init__()
        self.heads = num_heads
        head_dim = dim // self.heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(projection_dropout)

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        q = q * self.scale

        attn = einsum('b h i d, b h j d -> b h i j', q, k)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = einsum('b h i j, b h j d -> b h i d', attn, v)
        x = rearrange(x, 'b h n d -> b n (h d)')

        return self.proj_drop(self.proj(x))

# 定义 Transformer 编码器层模块
class TransformerEncoderLayer(nn.Module):
    """
    Inspired by torch.nn.TransformerEncoderLayer and
    rwightman's timm package.
    """
    # 初始化函数，定义了 Transformer Encoder 层的结构
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 attention_dropout=0.1, drop_path_rate=0.1):
        # 调用父类的初始化函数
        super().__init__()

        # 对输入进行 Layer Normalization
        self.pre_norm = nn.LayerNorm(d_model)
        # 定义自注意力机制
        self.self_attn = Attention(dim=d_model, num_heads=nhead,
                                   attention_dropout=attention_dropout, projection_dropout=dropout)

        # 第一个线性层
        self.linear1  = nn.Linear(d_model, dim_feedforward)
        # 第一个 Dropout 层
        self.dropout1 = nn.Dropout(dropout)
        # 第一个 Layer Normalization 层
        self.norm1    = nn.LayerNorm(d_model)
        # 第二个线性层
        self.linear2  = nn.Linear(dim_feedforward, d_model)
        # 第二个 Dropout 层
        self.dropout2 = nn.Dropout(dropout)

        # DropPath 模块
        self.drop_path = DropPath(drop_path_rate)

        # 激活函数为 GELU
        self.activation = F.gelu

    # 前向传播函数
    def forward(self, src, *args, **kwargs):
        # 使用自注意力机制处理输入，并加上 DropPath 模块
        src = src + self.drop_path(self.self_attn(self.pre_norm(src)))
        # 对结果进行 Layer Normalization
        src = self.norm1(src)
        # 第一个线性层、激活函数、Dropout、第二个线性层的组合
        src2 = self.linear2(self.dropout1(self.activation(self.linear1(src))))
        # 将结果与 DropPath 模块处理后的结果相加
        src = src + self.drop_path(self.dropout2(src2))
        # 返回处理后的结果
        return src
class DropPath(nn.Module):
    # 初始化 DropPath 类
    def __init__(self, drop_prob=None):
        # 调用父类的初始化方法
        super().__init__()
        # 将传入的 drop_prob 转换为浮点数
        self.drop_prob = float(drop_prob)

    # 前向传播方法
    def forward(self, x):
        # 获取输入 x 的批次大小、drop_prob、设备和数据类型
        batch, drop_prob, device, dtype = x.shape[0], self.drop_prob, x.device, x.dtype

        # 如果 drop_prob 小于等于 0 或者不处于训练模式，则直接返回输入 x
        if drop_prob <= 0. or not self.training:
            return x

        # 计算保留概率
        keep_prob = 1 - self.drop_prob
        # 构建形状元组
        shape = (batch, *((1,) * (x.ndim - 1)))

        # 生成保留掩码
        keep_mask = torch.zeros(shape, device=device).float().uniform_(0, 1) < keep_prob
        # 对输入 x 进行 DropPath 操作
        output = x.div(keep_prob) * keep_mask.float()
        return output

class Tokenizer(nn.Module):
    # 初始化 Tokenizer 类
    def __init__(self,
                 kernel_size, stride, padding,
                 pooling_kernel_size=3, pooling_stride=2, pooling_padding=1,
                 n_conv_layers=1,
                 n_input_channels=3,
                 n_output_channels=64,
                 in_planes=64,
                 activation=None,
                 max_pool=True,
                 conv_bias=False):
        # 调用父类的初始化方法
        super().__init()

        # 构建卷积层的通道数列表
        n_filter_list = [n_input_channels] + \
                        [in_planes for _ in range(n_conv_layers - 1)] + \
                        [n_output_channels]

        # 构建通道数列表的配对
        n_filter_list_pairs = zip(n_filter_list[:-1], n_filter_list[1:])

        # 构建卷积层序列
        self.conv_layers = nn.Sequential(
            *[nn.Sequential(
                nn.Conv2d(chan_in, chan_out,
                          kernel_size=(kernel_size, kernel_size),
                          stride=(stride, stride),
                          padding=(padding, padding), bias=conv_bias),
                nn.Identity() if not exists(activation) else activation(),
                nn.MaxPool2d(kernel_size=pooling_kernel_size,
                             stride=pooling_stride,
                             padding=pooling_padding) if max_pool else nn.Identity()
            )
                for chan_in, chan_out in n_filter_list_pairs
            ])

        # 对模型参数进行初始化
        self.apply(self.init_weight)

    # 计算序列长度
    def sequence_length(self, n_channels=3, height=224, width=224):
        return self.forward(torch.zeros((1, n_channels, height, width))).shape[1]

    # 前向传播方法
    def forward(self, x):
        # 对卷积层的输出进行重排列
        return rearrange(self.conv_layers(x), 'b c h w -> b (h w) c')

    # 初始化权重方法
    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)


class TransformerClassifier(nn.Module):
    # 初始化函数，设置模型的各种参数
    def __init__(self,
                 seq_pool=True,  # 是否使用序列池化
                 embedding_dim=768,  # 嵌入维度
                 num_layers=12,  # 编码器层数
                 num_heads=12,  # 注意力头数
                 mlp_ratio=4.0,  # MLP 扩展比例
                 num_classes=1000,  # 类别数
                 dropout_rate=0.1,  # Dropout 比例
                 attention_dropout=0.1,  # 注意力 Dropout 比例
                 stochastic_depth_rate=0.1,  # 随机深度比例
                 positional_embedding='sine',  # 位置编码类型
                 sequence_length=None,  # 序列长度
                 *args, **kwargs):  # 其他参数
        super().__init__()  # 调用父类的初始化函数
        assert positional_embedding in {'sine', 'learnable', 'none'}  # 断言位置编码类型合法

        dim_feedforward = int(embedding_dim * mlp_ratio)  # 计算前馈网络维度
        self.embedding_dim = embedding_dim  # 设置嵌入维度
        self.sequence_length = sequence_length  # 设置序列长度
        self.seq_pool = seq_pool  # 设置是否使用序列池化

        assert exists(sequence_length) or positional_embedding == 'none', \  # 断言序列长度存在或位置编码为'none'
            f"Positional embedding is set to {positional_embedding} and" \  # 打印位置编码设置信息
            f" the sequence length was not specified."

        if not seq_pool:  # 如果不使用序列池化
            sequence_length += 1  # 序列长度加一
            self.class_emb = nn.Parameter(torch.zeros(1, 1, self.embedding_dim), requires_grad=True)  # 创建类别嵌入参数
        else:
            self.attention_pool = nn.Linear(self.embedding_dim, 1)  # 创建注意力池化层

        if positional_embedding == 'none':  # 如果位置编码为'none'
            self.positional_emb = None  # 不使用位置编码
        elif positional_embedding == 'learnable':  # 如果位置编码为'learnable'
            self.positional_emb = nn.Parameter(torch.zeros(1, sequence_length, embedding_dim),  # 创建可学习位置编码参数
                                               requires_grad=True)
            nn.init.trunc_normal_(self.positional_emb, std=0.2)  # 对位置编码参数进行初始化
        else:
            self.positional_emb = nn.Parameter(sinusoidal_embedding(sequence_length, embedding_dim),  # 创建正弦位置编码参数
                                               requires_grad=False)

        self.dropout = nn.Dropout(p=dropout_rate)  # 创建 Dropout 层

        dpr = [x.item() for x in torch.linspace(0, stochastic_depth_rate, num_layers)]  # 计算随机深度比例列表

        self.blocks = nn.ModuleList([  # 创建 Transformer 编码器层列表
            TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads,
                                    dim_feedforward=dim_feedforward, dropout=dropout_rate,
                                    attention_dropout=attention_dropout, drop_path_rate=layer_dpr)
            for layer_dpr in dpr])

        self.norm = nn.LayerNorm(embedding_dim)  # 创建 LayerNorm 层

        self.fc = nn.Linear(embedding_dim, num_classes)  # 创建全连接层
        self.apply(self.init_weight)  # 应用初始化权重函数

    # 前向传播函数
    def forward(self, x):
        b = x.shape[0]  # 获取 batch 大小

        if not exists(self.positional_emb) and x.size(1) < self.sequence_length:  # 如果位置编码不存在且序列长度小于指定长度
            x = F.pad(x, (0, 0, 0, self.n_channels - x.size(1)), mode='constant', value=0)  # 对输入进行填充

        if not self.seq_pool:  # 如果不使用序列池化
            cls_token = repeat(self.class_emb, '1 1 d -> b 1 d', b = b)  # 重复类别嵌入
            x = torch.cat((cls_token, x), dim=1)  # 拼接类别嵌入和输入

        if exists(self.positional_emb):  # 如果位置编码存在
            x += self.positional_emb  # 加上位置编码

        x = self.dropout(x)  # Dropout

        for blk in self.blocks:  # 遍历编码器层
            x = blk(x)  # 应用编码器层

        x = self.norm(x)  # LayerNorm

        if self.seq_pool:  # 如果使用序列池化
            attn_weights = rearrange(self.attention_pool(x), 'b n 1 -> b n')  # 注意力权重计算
            x = einsum('b n, b n d -> b d', attn_weights.softmax(dim = 1), x)  # 加权池化
        else:
            x = x[:, 0]  # 取第一个位置的输出作为结果

        return self.fc(x)  # 全连接层输出结果

    # 初始化权重函数
    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Linear):  # 如果是线性层
            nn.init.trunc_normal_(m.weight, std=.02)  # 初始化权重
            if isinstance(m, nn.Linear) and exists(m.bias):  # 如果是线性层且存在偏置
                nn.init.constant_(m.bias, 0)  # 初始化偏置为0
        elif isinstance(m, nn.LayerNorm):  # 如果是 LayerNorm 层
            nn.init.constant_(m.bias, 0)  # 初始化偏置为0
            nn.init.constant_(m.weight, 1.0)  # 初始化权重为1.0
# 定义 CCT 类，继承自 nn.Module
class CCT(nn.Module):
    # 初始化函数，设置各种参数
    def __init__(
        self,
        img_size=224,
        embedding_dim=768,
        n_input_channels=3,
        n_conv_layers=1,
        kernel_size=7,
        stride=2,
        padding=3,
        pooling_kernel_size=3,
        pooling_stride=2,
        pooling_padding=1,
        *args, **kwargs
    ):
        # 调用父类的初始化函数
        super().__init__()
        # 获取图像的高度和宽度
        img_height, img_width = pair(img_size)

        # 初始化 Tokenizer 对象
        self.tokenizer = Tokenizer(n_input_channels=n_input_channels,
                                   n_output_channels=embedding_dim,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   pooling_kernel_size=pooling_kernel_size,
                                   pooling_stride=pooling_stride,
                                   pooling_padding=pooling_padding,
                                   max_pool=True,
                                   activation=nn.ReLU,
                                   n_conv_layers=n_conv_layers,
                                   conv_bias=False)

        # 初始化 TransformerClassifier 对象
        self.classifier = TransformerClassifier(
            sequence_length=self.tokenizer.sequence_length(n_channels=n_input_channels,
                                                           height=img_height,
                                                           width=img_width),
            embedding_dim=embedding_dim,
            seq_pool=True,
            dropout_rate=0.,
            attention_dropout=0.1,
            stochastic_depth=0.1,
            *args, **kwargs)

    # 前向传播函数
    def forward(self, x):
        # 对输入数据进行编码
        x = self.tokenizer(x)
        # 使用 Transformer 进行分类
        return self.classifier(x)
```