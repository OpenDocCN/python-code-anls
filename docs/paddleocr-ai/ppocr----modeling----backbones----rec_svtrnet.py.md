# `.\PaddleOCR\ppocr\modeling\backbones\rec_svtrnet.py`

```py
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 均按“原样”分发，不附带任何明示或暗示的担保或条件
# 请查看许可证以获取特定语言的权限和限制

# 导入所需的库
from paddle import ParamAttr
from paddle.nn.initializer import KaimingNormal
import numpy as np
import paddle
import paddle.nn as nn
from paddle.nn.initializer import TruncatedNormal, Constant, Normal

# 初始化不同类型的参数
trunc_normal_ = TruncatedNormal(std=.02)
normal_ = Normal
zeros_ = Constant(value=0.)
ones_ = Constant(value=1.)

# 定义 drop_path 函数，用于在训练时对输入进行随机丢弃
def drop_path(x, drop_prob=0., training=False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ...
    """
    # 如果丢弃概率为 0 或者不处于训练状态，则直接返回输入
    if drop_prob == 0. or not training:
        return x
    # 计算保留概率
    keep_prob = paddle.to_tensor(1 - drop_prob, dtype=x.dtype)
    # 生成与输入相同形状的随机张量
    shape = (paddle.shape(x)[0], ) + (1, ) * (x.ndim - 1)
    random_tensor = keep_prob + paddle.rand(shape, dtype=x.dtype)
    random_tensor = paddle.floor(random_tensor)  # 将随机张量二值化
    # 对输入进行随机丢弃
    output = x.divide(keep_prob) * random_tensor
    return output

# 定义 ConvBNLayer 类
class ConvBNLayer(nn.Layer):
    # 初始化卷积层，包括输入通道数、输出通道数、卷积核大小、步长、填充、是否包含偏置、分组数和激活函数
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 bias_attr=False,
                 groups=1,
                 act=nn.GELU):
        # 调用父类的初始化方法
        super().__init__()
        # 创建卷积层对象，设置输入通道数、输出通道数、卷积核大小、步长、填充、分组数、权重属性和偏置属性
        self.conv = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            weight_attr=paddle.ParamAttr(
                initializer=nn.initializer.KaimingUniform()),
            bias_attr=bias_attr)
        # 创建批归一化层对象，设置输出通道数
        self.norm = nn.BatchNorm2D(out_channels)
        # 创建激活函数对象
        self.act = act()
    
    # 前向传播函数，接收输入数据，经过卷积、批归一化和激活函数处理后返回结果
    def forward(self, inputs):
        # 输入数据经过卷积层处理
        out = self.conv(inputs)
        # 卷积结果经过批归一化处理
        out = self.norm(out)
        # 批归一化结果经过激活函数处理
        out = self.act(out)
        # 返回处理后的结果
        return out
class DropPath(nn.Layer):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        # 初始化 DropPath 类，设置 drop_prob 属性
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        # 在前向传播中应用 drop_path 函数
        return drop_path(x, self.drop_prob, self.training)


class Identity(nn.Layer):
    def __init__(self):
        # 初始化 Identity 类
        super(Identity, self).__init__()

    def forward(self, input):
        # 返回输入的恒等映射
        return input


class Mlp(nn.Layer):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        # 初始化 Mlp 类，设置线性层和激活函数等属性
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # 前向传播过程，包括线性层、激活函数和 dropout 操作
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ConvMixer(nn.Layer):
    def __init__(
            self,
            dim,
            num_heads=8,
            HW=[8, 25],
            local_k=[3, 3], ):
        # 初始化 ConvMixer 类，设置卷积参数和局部混合器
        super().__init__()
        self.HW = HW
        self.dim = dim
        self.local_mixer = nn.Conv2D(
            dim,
            dim,
            local_k,
            1, [local_k[0] // 2, local_k[1] // 2],
            groups=num_heads,
            weight_attr=ParamAttr(initializer=KaimingNormal()))

    def forward(self, x):
        # 前向传播过程，包括数据重排、局部混合器和数据恢复
        h = self.HW[0]
        w = self.HW[1]
        x = x.transpose([0, 2, 1]).reshape([0, self.dim, h, w])
        x = self.local_mixer(x)
        x = x.flatten(2).transpose([0, 2, 1])
        return x


class Attention(nn.Layer):
    # Attention 类未完整定义，需要继续补充
    # 初始化函数，设置注意力机制的参数
    def __init__(self,
                 dim,
                 num_heads=8,
                 mixer='Global',
                 HW=None,
                 local_k=[7, 11],
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.):
        # 调用父类的初始化函数
        super().__init__()
        # 设置注意力头的数量
        self.num_heads = num_heads
        # 设置输入维度
        self.dim = dim
        # 计算每个注意力头的维度
        self.head_dim = dim // num_heads
        # 设置缩放因子
        self.scale = qk_scale or self.head_dim**-0.5

        # 创建线性层，用于计算查询、键、值
        self.qkv = nn.Linear(dim, dim * 3, bias_attr=qkv_bias)
        # 创建用于注意力机制的 dropout 层
        self.attn_drop = nn.Dropout(attn_drop)
        # 创建用于投影的线性层
        self.proj = nn.Linear(dim, dim)
        # 创建用于投影的 dropout 层
        self.proj_drop = nn.Dropout(proj_drop)
        # 设置输入的高度和宽度
        self.HW = HW
        # 如果输入的高度和宽度不为空
        if HW is not None:
            # 获取高度和宽度
            H = HW[0]
            W = HW[1]
            # 计算输入的总数和维度
            self.N = H * W
            self.C = dim
        # 如果混合器为局部且输入的高度和宽度不为空
        if mixer == 'Local' and HW is not None:
            # 获取局部注意力的 k 值
            hk = local_k[0]
            wk = local_k[1]
            # 创建掩码矩阵
            mask = paddle.ones([H * W, H + hk - 1, W + wk - 1], dtype='float32')
            # 设置掩码矩阵的值
            for h in range(0, H):
                for w in range(0, W):
                    mask[h * W + w, h:h + hk, w:w + wk] = 0.
            # 将掩码矩阵转换为 PaddlePaddle 张量
            mask_paddle = mask[:, hk // 2:H + hk // 2, wk // 2:W + wk // 2].flatten(1)
            # 创建全为负无穷的张量
            mask_inf = paddle.full([H * W, H * W], '-inf', dtype='float32')
            # 根据条件设置掩码张量
            mask = paddle.where(mask_paddle < 1, mask_paddle, mask_inf)
            # 将掩码张量添加维度
            self.mask = mask.unsqueeze([0, 1])
        # 设置混合器类型
        self.mixer = mixer
    # 前向传播函数，接收输入 x
    def forward(self, x):
        # 使用 self.qkv 对象对输入 x 进行处理，并重塑形状
        qkv = self.qkv(x).reshape(
            (0, -1, 3, self.num_heads, self.head_dim)).transpose(
                (2, 0, 3, 1, 4))
        # 将处理后的结果分别赋值给 q, k, v
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]

        # 计算注意力矩阵 attn
        attn = (q.matmul(k.transpose((0, 1, 3, 2))))
        # 如果使用局部混合器，则在 attn 上添加掩码
        if self.mixer == 'Local':
            attn += self.mask
        # 对 attn 进行 softmax 操作
        attn = nn.functional.softmax(attn, axis=-1)
        # 对 attn 进行丢弃操作
        attn = self.attn_drop(attn)

        # 计算最终输出 x
        x = (attn.matmul(v)).transpose((0, 2, 1, 3)).reshape((0, -1, self.dim))
        # 使用 self.proj 对 x 进行处理
        x = self.proj(x)
        # 对处理后的 x 进行丢弃操作
        x = self.proj_drop(x)
        # 返回最终结果 x
        return x
# 定义一个名为 Block 的类，继承自 nn.Layer
class Block(nn.Layer):
    # 初始化函数，接受多个参数
    def __init__(self,
                 dim,
                 num_heads,
                 mixer='Global',
                 local_mixer=[7, 11],
                 HW=None,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer='nn.LayerNorm',
                 epsilon=1e-6,
                 prenorm=True):
        # 调用父类的初始化函数
        super().__init__()
        # 根据 norm_layer 的类型，创建不同的 norm1 对象
        if isinstance(norm_layer, str):
            self.norm1 = eval(norm_layer)(dim, epsilon=epsilon)
        else:
            self.norm1 = norm_layer(dim)
        # 根据 mixer 的类型选择不同的混合器
        if mixer == 'Global' or mixer == 'Local':
            self.mixer = Attention(
                dim,
                num_heads=num_heads,
                mixer=mixer,
                HW=HW,
                local_k=local_mixer,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop)
        elif mixer == 'Conv':
            self.mixer = ConvMixer(
                dim, num_heads=num_heads, HW=HW, local_k=local_mixer)
        else:
            raise TypeError("The mixer must be one of [Global, Local, Conv]")

        # 根据 drop_path 的值选择是否添加 DropPath 层
        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()
        # 根据 norm_layer 的类型，创建不同的 norm2 对象
        if isinstance(norm_layer, str):
            self.norm2 = eval(norm_layer)(dim, epsilon=epsilon)
        else:
            self.norm2 = norm_layer(dim)
        # 计算 MLP 的隐藏层维度
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_ratio = mlp_ratio
        # 创建 MLP 层
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)
        self.prenorm = prenorm
    # 定义前向传播函数，接受输入 x
    def forward(self, x):
        # 如果使用预归一化
        if self.prenorm:
            # 先将输入 x 经过混合层和随机丢弃路径，然后加上输入 x，再经过归一化层
            x = self.norm1(x + self.drop_path(self.mixer(x)))
            # 再将上一步结果经过 MLP 层和随机丢弃路径，然后加上上一步结果 x，再经过归一化层
            x = self.norm2(x + self.drop_path(self.mlp(x)))
        else:
            # 先将输入 x 经过归一化层，再经过混合层和随机丢弃路径，然后加上输入 x
            x = x + self.drop_path(self.mixer(self.norm1(x)))
            # 再将上一步结果经过归一化层，再经过 MLP 层和随机丢弃路径，然后加上上一步结果 x
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        # 返回处理后的结果 x
        return x
class PatchEmbed(nn.Layer):
    """ Image to Patch Embedding
    """

    # 定义 PatchEmbed 类，用于将图像转换为补丁嵌入表示

    def forward(self, x):
        # 前向传播函数，接收输入 x
        B, C, H, W = x.shape
        # 获取输入 x 的形状信息
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]}."
        # 断言输入图像的大小与模型期望的大小是否匹配
        x = self.proj(x).flatten(2).transpose((0, 2, 1))
        # 对输入进行投影操作，然后展平并转置
        return x
        # 返回处理后的结果


class SubSample(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 types='Pool',
                 stride=[2, 1],
                 sub_norm='nn.LayerNorm',
                 act=None):
        super().__init__()
        # 初始化函数，定义不同类型的下采样操作

        self.types = types
        # 设置下采样类型

        if types == 'Pool':
            # 如果类型是池化
            self.avgpool = nn.AvgPool2D(
                kernel_size=[3, 5], stride=stride, padding=[1, 2])
            self.maxpool = nn.MaxPool2D(
                kernel_size=[3, 5], stride=stride, padding=[1, 2])
            self.proj = nn.Linear(in_channels, out_channels)
            # 初始化平均池化、最大池化和投影操作
        else:
            self.conv = nn.Conv2D(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                weight_attr=ParamAttr(initializer=KaimingNormal()))
            # 初始化卷积操作
        self.norm = eval(sub_norm)(out_channels)
        # 根据给定的规范化类型初始化规范化操作
        if act is not None:
            self.act = act()
        else:
            self.act = None
        # 如果存在激活函数，则初始化激活函数，否则为 None

    def forward(self, x):
        # 前向传播函数，接收输入 x

        if self.types == 'Pool':
            # 如果类型是池化
            x1 = self.avgpool(x)
            x2 = self.maxpool(x)
            x = (x1 + x2) * 0.5
            # 对输入进行平均池化和最大池化，然后取平均值
            out = self.proj(x.flatten(2).transpose((0, 2, 1)))
            # 对处理后的结果进行投影操作
        else:
            x = self.conv(x)
            # 如果类型不是池化，则进行卷积操作
            out = x.flatten(2).transpose((0, 2, 1))
            # 对处理后的结果进行展平并转置
        out = self.norm(out)
        # 对结果进行规范化操作
        if self.act is not None:
            out = self.act(out)
        # 如果存在激活函数，则对结果进行激活

        return out
        # 返回处理后的结果


class SVTRNet(nn.Layer):
    # 定义 SVTRNet 类
    # 初始化神经网络权重
    def _init_weights(self, m):
        # 如果是线性层
        if isinstance(m, nn.Linear):
            # 对权重进行截断正态分布初始化
            trunc_normal_(m.weight)
            # 如果是线性层且有偏置项
            if isinstance(m, nn.Linear) and m.bias is not None:
                # 将偏置项初始化为零
                zeros_(m.bias)
        # 如果是 LayerNorm 层
        elif isinstance(m, nn.LayerNorm):
            # 将偏置项初始化为零
            zeros_(m.bias)
            # 将权重初始化为全为1
            ones_(m.weight)

    # 前向传播函数，处理特征
    def forward_features(self, x):
        # 对输入进行 patch embedding
        x = self.patch_embed(x)
        # 加上位置编码
        x = x + self.pos_embed
        # 对位置编码进行 dropout
        x = self.pos_drop(x)
        # 遍历第一组块
        for blk in self.blocks1:
            x = blk(x)
        # 如果存在 patch merging 操作
        if self.patch_merging is not None:
            # 进行下采样操作
            x = self.sub_sample1(
                x.transpose([0, 2, 1]).reshape(
                    [0, self.embed_dim[0], self.HW[0], self.HW[1]]))
        # 遍历第二组块
        for blk in self.blocks2:
            x = blk(x)
        # 如果存在 patch merging 操作
        if self.patch_merging is not None:
            # 进行下采样操作
            x = self.sub_sample2(
                x.transpose([0, 2, 1]).reshape(
                    [0, self.embed_dim[1], self.HW[0] // 2, self.HW[1]]))
        # 遍历第三组块
        for blk in self.blocks3:
            x = blk(x)
        # 如果不使用预归一化
        if not self.prenorm:
            # 对输出进行归一化
            x = self.norm(x)
        # 返回处理后的特征
        return x

    # 整体前向传播函数
    def forward(self, x):
        # 处理特征
        x = self.forward_features(x)
        # 如果使用长度头
        if self.use_lenhead:
            # 对特征进行平均池化
            len_x = self.len_conv(x.mean(1))
            # 对长度信息进行 dropout 和激活函数处理
            len_x = self.dropout_len(self.hardswish_len(len_x))
        # 如果是最后一个阶段
        if self.last_stage:
            # 如果存在 patch merging 操作
            if self.patch_merging is not None:
                h = self.HW[0] // 4
            else:
                h = self.HW[0]
            # 进行平均池化操作
            x = self.avg_pool(
                x.transpose([0, 2, 1]).reshape(
                    [0, self.embed_dim[2], h, self.HW[1]]))
            # 最后一层卷积操作
            x = self.last_conv(x)
            # 激活函数处理
            x = self.hardswish(x)
            # dropout 操作
            x = self.dropout(x)
        # 如果使用长度头，返回特征和长度信息
        if self.use_lenhead:
            return x, len_x
        # 否则只返回特征
        return x
```