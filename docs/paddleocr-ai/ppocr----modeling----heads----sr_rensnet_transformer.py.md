# `.\PaddleOCR\ppocr\modeling\heads\sr_rensnet_transformer.py`

```py
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“原样”分发的，
# 没有任何明示或暗示的保证或条件。
# 请查看许可证以获取特定语言的权限和限制
"""
此代码参考自:
https://github.com/FudanVI/FudanOCR/blob/main/text-gestalt/loss/transformer_english_decomposition.py
"""
import copy
import math

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

# 生成一个方形的序列掩码。掩码位置填充为 float('-inf')。
# 未掩码的位置填充为 float(0.0)。
def subsequent_mask(size):
    mask = paddle.ones([1, size, size], dtype='float32')
    mask_inf = paddle.triu(
        paddle.full(
            shape=[1, size, size], dtype='float32', fill_value='-inf'),
        diagonal=1)
    mask = mask + mask_inf
    padding_mask = paddle.equal(mask, paddle.to_tensor(1, dtype=mask.dtype))
    return padding_mask

# 复制 N 个相同的模块
def clones(module, N):
    return nn.LayerList([copy.deepcopy(module) for _ in range(N)])

# 根据掩码填充张量
def masked_fill(x, mask, value):
    y = paddle.full(x.shape, value, x.dtype)
    return paddle.where(mask, y, x)

# 注意力机制函数
def attention(query, key, value, mask=None, dropout=None, attention_map=None):
    d_k = query.shape[-1]
    scores = paddle.matmul(query,
                           paddle.transpose(key, [0, 1, 3, 2])) / math.sqrt(d_k)

    if mask is not None:
        scores = masked_fill(scores, mask == 0, float('-inf'))
    else:
        pass

    p_attn = F.softmax(scores, axis=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)
    # 返回矩阵乘法结果和注意力权重
    return paddle.matmul(p_attn, value), p_attn
# 定义一个多头注意力机制的类
class MultiHeadedAttention(nn.Layer):
    # 初始化函数，设置多头数、模型维度、dropout率和是否压缩注意力
    def __init__(self, h, d_model, dropout=0.1, compress_attention=False):
        super(MultiHeadedAttention, self).__init__()
        # 确保模型维度可以整除多头数
        assert d_model % h == 0
        # 计算每个头的维度
        self.d_k = d_model // h
        self.h = h
        # 使用线性层复制4次，用于计算注意力
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        # 添加dropout层
        self.dropout = nn.Dropout(p=dropout, mode="downscale_in_infer")
        self.compress_attention = compress_attention
        # 压缩注意力的线性层
        self.compress_attention_linear = nn.Linear(h, 1)

    # 前向传播函数，接收查询、键、值、掩码和注意力映射
    def forward(self, query, key, value, mask=None, attention_map=None):
        # 如果存在掩码，则在第二维度上增加一个维度
        if mask is not None:
            mask = mask.unsqueeze(1)
        # 获取批次大小
        nbatches = paddle.shape(query)[0]

        # 对查询、键、值进行线性变换、重塑和转置
        query, key, value = \
            [paddle.transpose(l(x).reshape([nbatches, -1, self.h, self.d_k]), [0,2,1,3])
             for l, x in zip(self.linears, (query, key, value))]

        # 调用注意力函数计算注意力
        x, attention_map = attention(
            query,
            key,
            value,
            mask=mask,
            dropout=self.dropout,
            attention_map=attention_map)

        # 重塑输出并返回
        x = paddle.reshape(
            paddle.transpose(x, [0, 2, 1, 3]),
            [nbatches, -1, self.h * self.d_k])

        return self.linears[-1](x), attention_map


# 定义一个ResNet类
class ResNet(nn.Layer):
    # 初始化 ResNet 类，传入输入通道数、基本块类型和每个阶段的层数
    def __init__(self, num_in, block, layers):
        # 调用父类的初始化方法
        super(ResNet, self).__init__()

        # 第一层卷积层，输入通道数为 num_in，输出通道数为 64，卷积核大小为 3x3，步长为 1，填充为 1
        self.conv1 = nn.Conv2D(num_in, 64, kernel_size=3, stride=1, padding=1)
        # 第一层批归一化层，输入通道数为 64，使用全局统计信息
        self.bn1 = nn.BatchNorm2D(64, use_global_stats=True)
        # 第一层激活函数为 ReLU
        self.relu1 = nn.ReLU()
        # 最大池化层，池化大小为 2x2，步长为 2
        self.pool = nn.MaxPool2D((2, 2), (2, 2))

        # 第二层卷积层，输入通道数为 64，输出通道数为 128，卷积核大小为 3x3，步长为 1，填充为 1
        self.conv2 = nn.Conv2D(64, 128, kernel_size=3, stride=1, padding=1)
        # 第二层批归一化层，输入通道数为 128，使用全局统计信息
        self.bn2 = nn.BatchNorm2D(128, use_global_stats=True)
        # 第二层激活函数为 ReLU
        self.relu2 = nn.ReLU()

        # 第一阶段池化层，池化大小为 2x2，步长为 2
        self.layer1_pool = nn.MaxPool2D((2, 2), (2, 2))
        # 创建第一阶段残差块，输入通道数为 128，输出通道数为 256，层数为 layers[0]
        self.layer1 = self._make_layer(block, 128, 256, layers[0])
        # 第一阶段卷积层，输入通道数为 256，输出通道数为 256，卷积核大小为 3x3，步长为 1，填充为 1
        self.layer1_conv = nn.Conv2D(256, 256, 3, 1, 1)
        # 第一阶段批归一化层，输入通道数为 256，使用全局统计信息
        self.layer1_bn = nn.BatchNorm2D(256, use_global_stats=True)
        # 第一阶段激活函数为 ReLU
        self.layer1_relu = nn.ReLU()

        # 第二阶段池化层，池化大小为 2x2，步长为 2
        self.layer2_pool = nn.MaxPool2D((2, 2), (2, 2))
        # 创建第二阶段残差块，输入通道数为 256，输出通道数为 256，层数为 layers[1]
        self.layer2 = self._make_layer(block, 256, 256, layers[1])
        # 第二阶段卷积层，输入通道数为 256，输出通道数为 256，卷积核大小为 3x3，步长为 1，填充为 1
        self.layer2_conv = nn.Conv2D(256, 256, 3, 1, 1)
        # 第二阶段批归一化层，输入通道数为 256，使用全局统计信息
        self.layer2_bn = nn.BatchNorm2D(256, use_global_stats=True)
        # 第二阶段激活函数为 ReLU
        self.layer2_relu = nn.ReLU()

        # 第三阶段池化层，池化大小为 2x2，步长为 2
        self.layer3_pool = nn.MaxPool2D((2, 2), (2, 2))
        # 创建第三阶段残差块，输入通道数为 256，输出通道数为 512，层数为 layers[2]
        self.layer3 = self._make_layer(block, 256, 512, layers[2])
        # 第三阶段卷积层，输入通道数为 512，输出通道数为 512，卷积核大小为 3x3，步长为 1，填充为 1
        self.layer3_conv = nn.Conv2D(512, 512, 3, 1, 1)
        # 第三阶段批归一化层，输入通道数为 512，使用全局统计信息
        self.layer3_bn = nn.BatchNorm2D(512, use_global_stats=True)
        # 第三阶段激活函数为 ReLU
        self.layer3_relu = nn.ReLU()

        # 第四阶段池化层，池化大小为 2x2，步长为 2
        self.layer4_pool = nn.MaxPool2D((2, 2), (2, 2))
        # 创建第四阶段残差块，输入通道数为 512，输出通道数为 512，层数为 layers[3]
        self.layer4 = self._make_layer(block, 512, 512, layers[3])
        # 第四阶段第二个卷积层，输入通道数为 512，输出通道数为 1024，卷积核大小为 3x3，步长为 1，填充为 1
        self.layer4_conv2 = nn.Conv2D(512, 1024, 3, 1, 1)
        # 第四阶段第二个批归一化层，输入通道数为 1024，使用全局统计信息
        self.layer4_conv2_bn = nn.BatchNorm2D(1024, use_global_stats=True)
        # 第四阶段第二个激活函数为 ReLU
        self.layer4_conv2_relu = nn.ReLU()
    # 创建 ResNet 的一个层，包含多个 block
    def _make_layer(self, block, inplanes, planes, blocks):

        # 如果输入通道数不等于输出通道数
        if inplanes != planes:
            # 创建下采样层，包含卷积层和批归一化层
            downsample = nn.Sequential(
                nn.Conv2D(inplanes, planes, 3, 1, 1),
                nn.BatchNorm2D(
                    planes, use_global_stats=True), )
        else:
            downsample = None
        layers = []
        # 添加第一个 block
        layers.append(block(inplanes, planes, downsample))
        # 添加剩余的 block
        for i in range(1, blocks):
            layers.append(block(planes, planes, downsample=None))

        # 返回包含所有 block 的序列
        return nn.Sequential(*layers)

    # 前向传播函数
    def forward(self, x):
        # 第一层卷积
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool(x)

        # 第二层卷积
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        # 第一个层的池化
        x = self.layer1_pool(x)
        x = self.layer1(x)
        x = self.layer1_conv(x)
        x = self.layer1_bn(x)
        x = self.layer1_relu(x)

        # 第二个层
        x = self.layer2(x)
        x = self.layer2_conv(x)
        x = self.layer2_bn(x)
        x = self.layer2_relu(x)

        # 第三个层
        x = self.layer3(x)
        x = self.layer3_conv(x)
        x = self.layer3_bn(x)
        x = self.layer3_relu(x)

        # 第四个层
        x = self.layer4(x)
        x = self.layer4_conv2(x)
        x = self.layer4_conv2_bn(x)
        x = self.layer4_conv2_relu(x)

        # 返回结果
        return x
class Bottleneck(nn.Layer):
    # 定义一个 Bottleneck 类，用于神经网络的瓶颈层
    def __init__(self, input_dim):
        # 初始化函数，接受输入维度参数
        super(Bottleneck, self).__init__()
        # 调用父类的初始化函数
        self.conv1 = nn.Conv2D(input_dim, input_dim, 1)
        # 创建一个卷积层，输入和输出维度均为 input_dim，卷积核大小为 1
        self.bn1 = nn.BatchNorm2D(input_dim, use_global_stats=True)
        # 创建一个二维批归一化层，输入维度为 input_dim，使用全局统计信息
        self.relu = nn.ReLU()
        # 创建一个 ReLU 激活函数层

        self.conv2 = nn.Conv2D(input_dim, input_dim, 3, 1, 1)
        # 创建一个卷积层，输入和输出维度均为 input_dim，卷积核大小为 3，步长为 1，填充为 1
        self.bn2 = nn.BatchNorm2D(input_dim, use_global_stats=True)
        # 创建一个二维批归一化层，输入维度为 input_dim，使用全局统计信息

    def forward(self, x):
        # 定义前向传播函数，接受输入 x
        residual = x
        # 将输入作为残差连接的一部分

        out = self.conv1(x)
        # 对输入进行第一个卷积操作
        out = self.bn1(out)
        # 对卷积结果进行批归一化
        out = self.relu(out)
        # 对批归一化结果进行 ReLU 激活

        out = self.conv2(out)
        # 对上一步结果进行第二个卷积操作
        out = self.bn2(out)
        # 对卷积结果进行批归一化

        out += residual
        # 将残差连接添加到当前结果中
        out = self.relu(out)
        # 对结果进行 ReLU 激活

        return out
        # 返回处理后的结果


class PositionalEncoding(nn.Layer):
    # 定义一个 PositionalEncoding 类，用于位置编码

    def __init__(self, dropout, dim, max_len=5000):
        # 初始化函数，接受丢弃率、维度和最大长度参数
        super(PositionalEncoding, self).__init__()
        # 调用父类的初始化函数
        self.dropout = nn.Dropout(p=dropout, mode="downscale_in_infer")
        # 创建一个丢弃层，用于在推理时缩放

        pe = paddle.zeros([max_len, dim])
        # 创建一个形状为 [max_len, dim] 的全零张量
        position = paddle.arange(0, max_len, dtype=paddle.float32).unsqueeze(1)
        # 创建一个位置张量
        div_term = paddle.exp(
            paddle.arange(0, dim, 2).astype('float32') *
            (-math.log(10000.0) / dim))
        # 计算除数项
        pe[:, 0::2] = paddle.sin(position * div_term)
        # 计算 sin 位置编码
        pe[:, 1::2] = paddle.cos(position * div_term)
        # 计算 cos 位置编码
        pe = paddle.unsqueeze(pe, 0)
        # 在第 0 维度上增加一个维度
        self.register_buffer('pe', pe)
        # 注册位置编码张量为模型的缓冲区

    def forward(self, x):
        # 定义前向传播函数，接受输入 x
        x = x + self.pe[:, :paddle.shape(x)[1]]
        # 将位置编码添加到输入中
        return self.dropout(x)
        # 返回添加位置编码后的结果


class PositionwiseFeedForward(nn.Layer):
    # 定义一个 PositionwiseFeedForward 类，用于位置前馈网络

    def __init__(self, d_model, d_ff, dropout=0.1):
        # 初始化函数，接受模型维度、前馈网络维度和丢弃率参数
        super(PositionwiseFeedForward, self).__init__()
        # 调用父类的初始化函数
        self.w_1 = nn.Linear(d_model, d_ff)
        # 创建一个线性层，输入维度为 d_model，输出维度为 d_ff
        self.w_2 = nn.Linear(d_ff, d_model)
        # 创建一个线性层，输入维度为 d_ff，输出维度为 d_model
        self.dropout = nn.Dropout(dropout, mode="downscale_in_infer")
        # 创建一个丢弃层，用于在推理时缩放

    def forward(self, x):
        # 定义前向传播函数，接受输入 x
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
        # 返回前馈网络的结果


class Generator(nn.Layer):
    # 定义一个 Generator 类，用于生成器
    # 初始化生成器对象，传入模型维度和词汇表大小
    def __init__(self, d_model, vocab):
        # 调用父类的初始化方法
        super(Generator, self).__init__()
        # 创建一个线性层，将输入维度转换为词汇表大小
        self.proj = nn.Linear(d_model, vocab)
        # 创建一个ReLU激活函数
        self.relu = nn.ReLU()

    # 前向传播函数，接收输入x并返回输出
    def forward(self, x):
        # 将输入x传入线性层进行转换
        out = self.proj(x)
        # 返回转换后的输出
        return out
class Embeddings(nn.Layer):
    # 定义 Embeddings 类，用于将输入序列中的每个 token 映射为对应的 d_model 维度的向量
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        # 创建 Embedding 层，将词汇表中的每个词映射为 d_model 维度的向量
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        # 将输入序列 x 映射为对应的向量，并乘以 math.sqrt(self.d_model)
        embed = self.lut(x) * math.sqrt(self.d_model)
        return embed


class LayerNorm(nn.Layer):
    # 构建一个 LayerNorm 模块
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        # 创建参数 a_2 和 b_2，用于归一化
        self.a_2 = self.create_parameter(
            shape=[features],
            default_initializer=paddle.nn.initializer.Constant(1.0))
        self.b_2 = self.create_parameter(
            shape=[features],
            default_initializer=paddle.nn.initializer.Constant(0.0))
        self.eps = eps

    def forward(self, x):
        # 计算输入 x 的均值和标准差
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        # 返回 LayerNorm 后的结果
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class Decoder(nn.Layer):
    # 定义 Decoder 类，用于解码器部分的处理
    def __init__(self):
        super(Decoder, self).__init__()

        # 定义多头注意力机制和 LayerNorm 模块
        self.mask_multihead = MultiHeadedAttention(
            h=16, d_model=1024, dropout=0.1)
        self.mul_layernorm1 = LayerNorm(1024)

        # 定义多头注意力机制和 LayerNorm 模块
        self.multihead = MultiHeadedAttention(h=16, d_model=1024, dropout=0.1)
        self.mul_layernorm2 = LayerNorm(1024)

        # 定义位置前馈神经网络和 LayerNorm 模块
        self.pff = PositionwiseFeedForward(1024, 2048)
        self.mul_layernorm3 = LayerNorm(1024)
    # 定义一个前向传播函数，接受文本、卷积特征和注意力图作为输入
    def forward(self, text, conv_feature, attention_map=None):
        # 获取文本的最大长度
        text_max_length = text.shape[1]
        # 生成一个掩码，用于遮挡未来信息
        mask = subsequent_mask(text_max_length)
        # 初始化结果为文本
        result = text
        # 对文本进行自注意力操作，并添加残差连接和层归一化
        result = self.mul_layernorm1(result + self.mask_multihead(
            text, text, text, mask=mask)[0])
        # 获取卷积特征的形状信息
        b, c, h, w = conv_feature.shape
        # 将卷积特征重塑为[b, h*w, c]的形状
        conv_feature = paddle.transpose(
            conv_feature.reshape([b, c, h * w]), [0, 2, 1])
        # 对文本和卷积特征进行多头注意力操作
        word_image_align, attention_map = self.multihead(
            result,
            conv_feature,
            conv_feature,
            mask=None,
            attention_map=attention_map)
        # 添加残差连接和层归一化
        result = self.mul_layernorm2(result + word_image_align)
        # 对结果进行前馈神经网络操作，并添加残差连接和层归一化
        result = self.mul_layernorm3(result + self.pff(result))

        # 返回处理后的结果和注意力图
        return result, attention_map
class BasicBlock(nn.Layer):
    # 定义基本的残差块类
    def __init__(self, inplanes, planes, downsample):
        # 初始化函数，接受输入通道数、输出通道数和下采样函数
        super(BasicBlock, self).__init__()
        # 调用父类的初始化函数
        self.conv1 = nn.Conv2D(
            inplanes, planes, kernel_size=3, stride=1, padding=1)
        # 创建卷积层1
        self.bn1 = nn.BatchNorm2D(planes, use_global_stats=True)
        # 创建批归一化层1
        self.relu = nn.ReLU()
        # 创建ReLU激活函数
        self.conv2 = nn.Conv2D(
            planes, planes, kernel_size=3, stride=1, padding=1)
        # 创建卷积层2
        self.bn2 = nn.BatchNorm2D(planes, use_global_stats=True)
        # 创建批归一化层2
        self.downsample = downsample
        # 设置下采样函数

    def forward(self, x):
        # 定义前向传播函数
        residual = x
        # 将输入赋值给残差变量

        out = self.conv1(x)
        # 使用卷积层1处理输入
        out = self.bn1(out)
        # 使用批归一化层1处理输出
        out = self.relu(out)
        # 使用ReLU激活函数处理输出

        out = self.conv2(out)
        # 使用卷积层2处理输出
        out = self.bn2(out)
        # 使用批归一化层2处理输出

        if self.downsample != None:
            # 如果存在下采样函数
            residual = self.downsample(residual)
            # 则对残差进行下采样处理

        out += residual
        # 将残差与输出相加
        out = self.relu(out)
        # 使用ReLU激活函数处理输出

        return out
        # 返回输出结果


class Encoder(nn.Layer):
    # 定义编码器类
    def __init__(self):
        # 初始化函数
        super(Encoder, self).__init__()
        # 调用父类的初始化函数
        self.cnn = ResNet(num_in=1, block=BasicBlock, layers=[1, 2, 5, 3])
        # 创建ResNet模型，指定输入通道数、残差块类型和每个阶段的残差块数量

    def forward(self, input):
        # 定义前向传播函数
        conv_result = self.cnn(input)
        # 使用ResNet模型处理输入数据
        return conv_result
        # 返回处理结果


class Transformer(nn.Layer):
    # 定义Transformer类
    def __init__(self, in_channels=1, alphabet='0123456789'):
        # 初始化函数，接受输入通道数和字母表
        super(Transformer, self).__init__()
        # 调用父类的初始化函数
        self.alphabet = alphabet
        # 设置字母表
        word_n_class = self.get_alphabet_len()
        # 获取字母表长度
        self.embedding_word_with_upperword = Embeddings(512, word_n_class)
        # 创建嵌入层
        self.pe = PositionalEncoding(dim=512, dropout=0.1, max_len=5000)
        # 创建位置编码层

        self.encoder = Encoder()
        # 创建编码器
        self.decoder = Decoder()
        # 创建解码器
        self.generator_word_with_upperword = Generator(1024, word_n_class)
        # 创建生成器

        for p in self.parameters():
            # 遍历模型参数
            if p.dim() > 1:
                # 如果参数维度大于1
                nn.initializer.XavierNormal(p)
                # 使用Xavier正态分布初始化参数

    def get_alphabet_len(self):
        # 定义获取字母表长度的函数
        return len(self.alphabet)
        # 返回字母表的长度
    # 前向传播函数，接收图像、文本长度、文本输入和注意力图作为输入
    def forward(self, image, text_length, text_input, attention_map=None):
        # 如果图像通道数为3，将RGB通道分离并转换为灰度图像
        if image.shape[1] == 3:
            R = image[:, 0:1, :, :]
            G = image[:, 1:2, :, :]
            B = image[:, 2:3, :, :]
            image = 0.299 * R + 0.587 * G + 0.114 * B

        # 使用编码器对图像进行特征提取
        conv_feature = self.encoder(image)  # batch, 1024, 8, 32
        # 获取文本序列的最大长度
        max_length = max(text_length)
        text_input = text_input[:, :max_length]

        # 对文本输入进行词嵌入
        text_embedding = self.embedding_word_with_upperword(
            text_input)  # batch, text_max_length, 512
        # 生成位置编码
        postion_embedding = self.pe(
            paddle.zeros(text_embedding.shape))  # batch, text_max_length, 512
        # 将词嵌入和位置编码拼接在一起
        text_input_with_pe = paddle.concat([text_embedding, postion_embedding],
                                           2)  # batch, text_max_length, 1024
        batch, seq_len, _ = text_input_with_pe.shape

        # 使用解码器对文本输入进行解码
        text_input_with_pe, word_attention_map = self.decoder(
            text_input_with_pe, conv_feature)

        # 生成词级别的结果
        word_decoder_result = self.generator_word_with_upperword(
            text_input_with_pe)

        # 如果处于训练状态
        if self.training:
            # 计算总文本长度
            total_length = paddle.sum(text_length)
            # 初始化概率结果
            probs_res = paddle.zeros([total_length, self.get_alphabet_len()])
            start = 0

            # 遍历每个文本序列，将解码结果填充到概率结果中
            for index, length in enumerate(text_length):
                length = int(length.numpy())
                probs_res[start:start + length, :] = word_decoder_result[
                    index, 0:0 + length, :]

                start = start + length

            # 返回概率结果、词级别注意力图和空值
            return probs_res, word_attention_map, None
        else:
            # 返回词级别的结果
            return word_decoder_result
```