# `.\lucidrains\vit-pytorch\vit_pytorch\cct_3d.py`

```py
import torch  # 导入 PyTorch 库
from torch import nn, einsum  # 从 PyTorch 库中导入 nn 模块和 einsum 函数
import torch.nn.functional as F  # 从 PyTorch 库中导入 F 模块

from einops import rearrange, repeat  # 从 einops 库中导入 rearrange 和 repeat 函数

# helpers

def exists(val):
    return val is not None  # 判断变量是否存在的辅助函数

def default(val, d):
    return val if exists(val) else d  # 如果变量存在则返回变量，否则返回默认值的辅助函数

def pair(t):
    return t if isinstance(t, tuple) else (t, t)  # 如果输入是元组则返回输入，否则返回包含输入两次的元组

# CCT Models

__all__ = ['cct_2', 'cct_4', 'cct_6', 'cct_7', 'cct_8', 'cct_14', 'cct_16']  # 定义导出的模型名称列表

# 定义不同层数的 CCT 模型函数

def cct_2(*args, **kwargs):
    return _cct(num_layers=2, num_heads=2, mlp_ratio=1, embedding_dim=128,
                *args, **kwargs)  # 返回 2 层 CCT 模型

def cct_4(*args, **kwargs):
    return _cct(num_layers=4, num_heads=2, mlp_ratio=1, embedding_dim=128,
                *args, **kwargs)  # 返回 4 层 CCT 模型

def cct_6(*args, **kwargs):
    return _cct(num_layers=6, num_heads=4, mlp_ratio=2, embedding_dim=256,
                *args, **kwargs)  # 返回 6 层 CCT 模型

def cct_7(*args, **kwargs):
    return _cct(num_layers=7, num_heads=4, mlp_ratio=2, embedding_dim=256,
                *args, **kwargs)  # 返回 7 层 CCT 模型

def cct_8(*args, **kwargs):
    return _cct(num_layers=8, num_heads=4, mlp_ratio=2, embedding_dim=256,
                *args, **kwargs)  # 返回 8 层 CCT 模型

def cct_14(*args, **kwargs):
    return _cct(num_layers=14, num_heads=6, mlp_ratio=3, embedding_dim=384,
                *args, **kwargs)  # 返回 14 层 CCT 模型

def cct_16(*args, **kwargs):
    return _cct(num_layers=16, num_heads=6, mlp_ratio=3, embedding_dim=384,
                *args, **kwargs)  # 返回 16 层 CCT 模型

# 定义 CCT 模型函数

def _cct(num_layers, num_heads, mlp_ratio, embedding_dim,
         kernel_size=3, stride=None, padding=None,
         *args, **kwargs):
    stride = default(stride, max(1, (kernel_size // 2) - 1))  # 设置默认的步长
    padding = default(padding, max(1, (kernel_size // 2)))  # 设置默认的填充大小

    return CCT(num_layers=num_layers,
               num_heads=num_heads,
               mlp_ratio=mlp_ratio,
               embedding_dim=embedding_dim,
               kernel_size=kernel_size,
               stride=stride,
               padding=padding,
               *args, **kwargs)  # 返回 CCT 模型

# positional

def sinusoidal_embedding(n_channels, dim):
    pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                            for p in range(n_channels)])  # 计算正弦余弦位置编码
    pe[:, 0::2] = torch.sin(pe[:, 0::2])  # 偶数列使用正弦函数
    pe[:, 1::2] = torch.cos(pe[:, 1::2])  # 奇数列使用余弦函数
    return rearrange(pe, '... -> 1 ...')  # 重新排列位置编码的维度

# modules

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, attention_dropout=0.1, projection_dropout=0.1):
        super().__init__()
        self.heads = num_heads  # 设置注意力头数
        head_dim = dim // self.heads  # 计算每个头的维度
        self.scale = head_dim ** -0.5  # 缩放因子

        self.qkv = nn.Linear(dim, dim * 3, bias=False)  # 线性变换层
        self.attn_drop = nn.Dropout(attention_dropout)  # 注意力丢弃层
        self.proj = nn.Linear(dim, dim)  # 投影层
        self.proj_drop = nn.Dropout(projection_dropout)  # 投影丢弃层

    def forward(self, x):
        B, N, C = x.shape  # 获取输入张量的形状

        qkv = self.qkv(x).chunk(3, dim = -1)  # 将线性变换后的张量切分为 Q、K、V
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)  # 重排 Q、K、V 的维度

        q = q * self.scale  # 缩放 Q

        attn = einsum('b h i d, b h j d -> b h i j', q, k)  # 计算注意力分数
        attn = attn.softmax(dim=-1)  # 对注意力分数进行 softmax
        attn = self.attn_drop(attn)  # 使用注意力丢弃层

        x = einsum('b h i j, b h j d -> b h i d', attn, v)  # 计算加权后的 V
        x = rearrange(x, 'b h n d -> b n (h d)')  # 重排输出张量的维度

        return self.proj_drop(self.proj(x))  # 使用投影丢弃层进行投影

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
        # 使用自注意力机制对输入进行处理，并加上 DropPath 模块
        src = src + self.drop_path(self.self_attn(self.pre_norm(src)))
        # 对结果进行 Layer Normalization
        src = self.norm1(src)
        # 第二个线性层的计算过程
        src2 = self.linear2(self.dropout1(self.activation(self.linear1(src))))
        # 将第二个线性层的结果加上 DropPath 模块
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
        # 获取输入 x 的批量大小、drop_prob、设备和数据类型
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
        # 对输入 x 进行处理并返回输出
        output = x.div(keep_prob) * keep_mask.float()
        return output

class Tokenizer(nn.Module):
    # 初始化 Tokenizer 类
    def __init__(
        self,
        frame_kernel_size,
        kernel_size,
        stride,
        padding,
        frame_stride=1,
        frame_pooling_stride=1,
        frame_pooling_kernel_size=1,
        pooling_kernel_size=3,
        pooling_stride=2,
        pooling_padding=1,
        n_conv_layers=1,
        n_input_channels=3,
        n_output_channels=64,
        in_planes=64,
        activation=None,
        max_pool=True,
        conv_bias=False
    ):
        # 调用父类的初始化方法
        super().__init__()

        # 构建卷积层的通道数列表
        n_filter_list = [n_input_channels] + \
                        [in_planes for _ in range(n_conv_layers - 1)] + \
                        [n_output_channels]

        # 构建通道数列表的配对
        n_filter_list_pairs = zip(n_filter_list[:-1], n_filter_list[1:])

        # 构建卷积层序列
        self.conv_layers = nn.Sequential(
            *[nn.Sequential(
                nn.Conv3d(chan_in, chan_out,
                          kernel_size=(frame_kernel_size, kernel_size, kernel_size),
                          stride=(frame_stride, stride, stride),
                          padding=(frame_kernel_size // 2, padding, padding), bias=conv_bias),
                nn.Identity() if not exists(activation) else activation(),
                nn.MaxPool3d(kernel_size=(frame_pooling_kernel_size, pooling_kernel_size, pooling_kernel_size),
                             stride=(frame_pooling_stride, pooling_stride, pooling_stride),
                             padding=(frame_pooling_kernel_size // 2, pooling_padding, pooling_padding)) if max_pool else nn.Identity()
            )
                for chan_in, chan_out in n_filter_list_pairs
            ])

        # 对模型进行权重初始化
        self.apply(self.init_weight)

    # 计算序列长度
    def sequence_length(self, n_channels=3, frames=8, height=224, width=224):
        return self.forward(torch.zeros((1, n_channels, frames, height, width))).shape[1]

    # 前向传播方法
    def forward(self, x):
        # 对输入 x 进行卷积操作并返回重排后的输出
        x = self.conv_layers(x)
        return rearrange(x, 'b c f h w -> b (f h w) c')

    # 初始化权重方法
    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight)


class TransformerClassifier(nn.Module):
    # 初始化 TransformerClassifier 类
    def __init__(
        self,
        seq_pool=True,
        embedding_dim=768,
        num_layers=12,
        num_heads=12,
        mlp_ratio=4.0,
        num_classes=1000,
        dropout_rate=0.1,
        attention_dropout=0.1,
        stochastic_depth_rate=0.1,
        positional_embedding='sine',
        sequence_length=None,
        *args, **kwargs
    ):
        # 调用父类的构造函数
        super().__init__()
        # 断言位置编码在{'sine', 'learnable', 'none'}中
        assert positional_embedding in {'sine', 'learnable', 'none'}

        # 计算前馈网络的维度
        dim_feedforward = int(embedding_dim * mlp_ratio)
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.seq_pool = seq_pool

        # 断言序列长度存在或者位置编码为'none'
        assert exists(sequence_length) or positional_embedding == 'none', \
            f"Positional embedding is set to {positional_embedding} and" \
            f" the sequence length was not specified."

        # 如果不使用序列池化
        if not seq_pool:
            sequence_length += 1
            self.class_emb = nn.Parameter(torch.zeros(1, 1, self.embedding_dim))
        else:
            self.attention_pool = nn.Linear(self.embedding_dim, 1)

        # 根据位置编码类型初始化位置编码
        if positional_embedding == 'none':
            self.positional_emb = None
        elif positional_embedding == 'learnable':
            self.positional_emb = nn.Parameter(torch.zeros(1, sequence_length, embedding_dim))
            nn.init.trunc_normal_(self.positional_emb, std=0.2)
        else:
            self.register_buffer('positional_emb', sinusoidal_embedding(sequence_length, embedding_dim))

        # 初始化Dropout层
        self.dropout = nn.Dropout(p=dropout_rate)

        # 生成随机Drop Path率
        dpr = [x.item() for x in torch.linspace(0, stochastic_depth_rate, num_layers)]

        # 创建Transformer编码器层
        self.blocks = nn.ModuleList([
            TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads,
                                    dim_feedforward=dim_feedforward, dropout=dropout_rate,
                                    attention_dropout=attention_dropout, drop_path_rate=layer_dpr)
            for layer_dpr in dpr])

        # 初始化LayerNorm层
        self.norm = nn.LayerNorm(embedding_dim)

        # 初始化全连接层
        self.fc = nn.Linear(embedding_dim, num_classes)
        # 应用初始化权重函数
        self.apply(self.init_weight)

    @staticmethod
    def init_weight(m):
        # 初始化线性层的权重
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            # 如果是线性层且存在偏置项，则初始化偏置项
            if isinstance(m, nn.Linear) and exists(m.bias):
                nn.init.constant_(m.bias, 0)
        # 初始化LayerNorm层的权重和偏置项
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # 获取批量大小
        b = x.shape[0]

        # 如果位置编码不存在且输入序列长度小于设定的序列长度，则进行填充
        if not exists(self.positional_emb) and x.size(1) < self.sequence_length:
            x = F.pad(x, (0, 0, 0, self.n_channels - x.size(1)), mode='constant', value=0)

        # 如果不使用序列池化，则在输入序列前添加类别标记
        if not self.seq_pool:
            cls_token = repeat(self.class_emb, '1 1 d -> b 1 d', b=b)
            x = torch.cat((cls_token, x), dim=1)

        # 如果位置编码存在，则加上位置编码
        if exists(self.positional_emb):
            x += self.positional_emb

        # Dropout层
        x = self.dropout(x)

        # 遍历Transformer编码器层
        for blk in self.blocks:
            x = blk(x)

        # LayerNorm层
        x = self.norm(x)

        # 如果使用序列池化，则计算注意力权重并进行加权求和
        if self.seq_pool:
            attn_weights = rearrange(self.attention_pool(x), 'b n 1 -> b n')
            x = einsum('b n, b n d -> b d', attn_weights.softmax(dim=1), x)
        else:
            x = x[:, 0]

        # 全连接层
        return self.fc(x)
# 定义 CCT 类，继承自 nn.Module
class CCT(nn.Module):
    # 初始化函数，设置模型参数
    def __init__(
        self,
        img_size=224,  # 图像大小，默认为 224
        num_frames=8,  # 帧数，默认为 8
        embedding_dim=768,  # 嵌入维度，默认为 768
        n_input_channels=3,  # 输入通道数，默认为 3
        n_conv_layers=1,  # 卷积层数，默认为 1
        frame_stride=1,  # 帧步长，默认为 1
        frame_kernel_size=3,  # 帧卷积核大小，默认为 3
        frame_pooling_kernel_size=1,  # 帧池化核大小，默认为 1
        frame_pooling_stride=1,  # 帧池化步长，默认为 1
        kernel_size=7,  # 卷积核大小，默认为 7
        stride=2,  # 步长，默认为 2
        padding=3,  # 填充，默认为 3
        pooling_kernel_size=3,  # 池化核大小，默认为 3
        pooling_stride=2,  # 池化步长，默认为 2
        pooling_padding=1,  # 池化填充，默认为 1
        *args, **kwargs  # 其他参数
    ):
        super().__init__()  # 调用父类的初始化函数

        img_height, img_width = pair(img_size)  # 获取图像的高度和宽度

        # 初始化 Tokenizer 对象
        self.tokenizer = Tokenizer(
            n_input_channels=n_input_channels,
            n_output_channels=embedding_dim,
            frame_stride=frame_stride,
            frame_kernel_size=frame_kernel_size,
            frame_pooling_stride=frame_pooling_stride,
            frame_pooling_kernel_size=frame_pooling_kernel_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            pooling_kernel_size=pooling_kernel_size,
            pooling_stride=pooling_stride,
            pooling_padding=pooling_padding,
            max_pool=True,
            activation=nn.ReLU,
            n_conv_layers=n_conv_layers,
            conv_bias=False
        )

        # 初始化 TransformerClassifier 对象
        self.classifier = TransformerClassifier(
            sequence_length=self.tokenizer.sequence_length(
                n_channels=n_input_channels,
                frames=num_frames,
                height=img_height,
                width=img_width
            ),
            embedding_dim=embedding_dim,
            seq_pool=True,
            dropout_rate=0.,
            attention_dropout=0.1,
            stochastic_depth=0.1,
            *args, **kwargs
        )

    # 前向传播函数
    def forward(self, x):
        x = self.tokenizer(x)  # 对输入数据进行编码
        return self.classifier(x)  # 对编码后的���据进行分类
```