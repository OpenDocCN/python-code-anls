# `.\PaddleOCR\ppocr\modeling\heads\table_master_head.py`

```
# 版权声明，告知代码版权归属于 PaddlePaddle 作者
#
# 根据 Apache 许可证 2.0 版本授权使用此文件
# 只有在遵守许可证的情况下才能使用此文件
# 可以在以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 是基于“按原样”分发的，没有任何明示或暗示的保证或条件
# 请查看许可证以获取有关权限和限制的具体语言
"""
"""
此代码参考自:
https://github.com/JiaquanYe/TableMASTER-mmocr/blob/master/mmocr/models/textrecog/decoders/master_decoder.py
"""

# 导入所需的库
import copy
import math
import paddle
from paddle import nn
from paddle.nn import functional as F

# 定义一个类 TableMasterHead，用于分割最后一层的两个 Transformer 头部
# Cls_layer 用于结构化标记分类
# Bbox_layer 用于回归边界框坐标
class TableMasterHead(nn.Layer):
    # 初始化函数，设置模型的参数
    def __init__(self,
                 in_channels,
                 out_channels=30,
                 headers=8,
                 d_ff=2048,
                 dropout=0,
                 max_text_length=500,
                 loc_reg_num=4,
                 **kwargs):
        # 调用父类的初始化函数
        super(TableMasterHead, self).__init__()
        # 获取输入通道的最后一个维度作为隐藏层大小
        hidden_size = in_channels[-1]
        # 克隆两个解码器层作为主要层
        self.layers = clones(
            DecoderLayer(headers, hidden_size, dropout, d_ff), 2)
        # 克隆一个解码器层作为分类层
        self.cls_layer = clones(
            DecoderLayer(headers, hidden_size, dropout, d_ff), 1)
        # 克隆一个解码器层作为边界框层
        self.bbox_layer = clones(
            DecoderLayer(headers, hidden_size, dropout, d_ff), 1)
        # 创建分类层的全连接层
        self.cls_fc = nn.Linear(hidden_size, out_channels)
        # 创建边界框层的全连接层和 Sigmoid 激活函数
        self.bbox_fc = nn.Sequential(
            # nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, loc_reg_num),
            nn.Sigmoid())
        # 创建 LayerNorm 层
        self.norm = nn.LayerNorm(hidden_size)
        # 创建嵌入层
        self.embedding = Embeddings(d_model=hidden_size, vocab=out_channels)
        # 创建位置编码层
        self.positional_encoding = PositionalEncoding(d_model=hidden_size)

        # 定义起始符号和填充符号的索引
        self.SOS = out_channels - 3
        self.PAD = out_channels - 1
        self.out_channels = out_channels
        self.loc_reg_num = loc_reg_num
        self.max_text_length = max_text_length

    # 创建用于自注意力的掩码
    def make_mask(self, tgt):
        """
        Make mask for self attention.
        :param src: [b, c, h, l_src]
        :param tgt: [b, l_tgt]
        :return:
        """
        # 创建填充掩码
        trg_pad_mask = (tgt != self.PAD).unsqueeze(1).unsqueeze(3)

        # 获取目标序列的长度
        tgt_len = paddle.shape(tgt)[1]
        # 创建下三角矩阵作为子掩码
        trg_sub_mask = paddle.tril(
            paddle.ones(
                ([tgt_len, tgt_len]), dtype=paddle.float32))

        # 将填充掩码和子掩码合并作为最终的目标掩码
        tgt_mask = paddle.logical_and(
            trg_pad_mask.astype(paddle.float32), trg_sub_mask)
        return tgt_mask.astype(paddle.float32)
    # 解码器的主要处理过程，接收输入、特征、源掩码和目标掩码
    def decode(self, input, feature, src_mask, tgt_mask):
        # 使用嵌入层将输入转换为张量 x: 1*x*512, feature: 1*3600,512
        x = self.embedding(input)  
        # 对输入进行位置编码
        x = self.positional_encoding(x)

        # 原始的 transformer 层
        for i, layer in enumerate(self.layers):
            # 逐层进行 transformer 操作
            x = layer(x, feature, src_mask, tgt_mask)

        # cls 头部
        cls_x = x
        for layer in self.cls_layer:
            # 逐层进行 cls 头部操作
            cls_x = layer(x, feature, src_mask, tgt_mask)
        cls_x = self.norm(cls_x)

        # bbox 头部
        bbox_x = x
        for layer in self.bbox_layer:
            # 逐层进行 bbox 头部操作
            bbox_x = layer(x, feature, src_mask, tgt_mask)
        bbox_x = self.norm(bbox_x)
        # 返回 cls 头部和 bbox 头部的结果
        return self.cls_fc(cls_x), self.bbox_fc(bbox_x)

    # 贪婪解码过程，接收起始符号 SOS 和特征
    def greedy_forward(self, SOS, feature):
        input = SOS
        # 初始化输出和 bbox 输出
        output = paddle.zeros(
            [input.shape[0], self.max_text_length + 1, self.out_channels])
        bbox_output = paddle.zeros(
            [input.shape[0], self.max_text_length + 1, self.loc_reg_num])
        max_text_length = paddle.to_tensor(self.max_text_length)
        # 循环生成文本序列
        for i in range(max_text_length + 1):
            # 生成目标掩码
            target_mask = self.make_mask(input)
            # 解码器解码
            out_step, bbox_output_step = self.decode(input, feature, None,
                                                     target_mask)
            # 对输出进行 softmax 操作
            prob = F.softmax(out_step, axis=-1)
            # 获取下一个词的索引
            next_word = prob.argmax(axis=2, dtype="int64")
            # 更新输入序列
            input = paddle.concat(
                [input, next_word[:, -1].unsqueeze(-1)], axis=1)
            # 如果达到最大文本长度，则更新输出和 bbox 输出
            if i == self.max_text_length:
                output = out_step
                bbox_output = bbox_output_step
        # 返回输出和 bbox 输出
        return output, bbox_output
    # 定义训练过程的前向传播函数，接收编码器输出和目标标签作为输入
    def forward_train(self, out_enc, targets):
        # x 是标签的 token
        # feat 是经过骨干网络处理后位置编码之前的特征
        # out_enc 是位置编码后的特征
        # 获取填充后的目标标签
        padded_targets = targets[0]
        # 初始化源序列掩码为 None
        src_mask = None
        # 生成目标序列掩码
        tgt_mask = self.make_mask(padded_targets[:, :-1])
        # 解码器解码，得到输出和边界框输出
        output, bbox_output = self.decode(padded_targets[:, :-1], out_enc,
                                          src_mask, tgt_mask)
        # 返回结构概率和位置预测
        return {'structure_probs': output, 'loc_preds': bbox_output}

    # 定义测试过程的前向传播函数，接收编码器输出作为输入
    def forward_test(self, out_enc):
        # 获取批量大小
        batch_size = out_enc.shape[0]
        # 初始化起始符号 SOS
        SOS = paddle.zeros([batch_size, 1], dtype='int64') + self.SOS
        # 贪婪解码，得到输出和边界框输出
        output, bbox_output = self.greedy_forward(SOS, out_enc)
        # 对输出进行 softmax 处理
        output = F.softmax(output)
        # 返回结构概率和位置预测
        return {'structure_probs': output, 'loc_preds': bbox_output}

    # 定义整体前向传播函数，接收特征和目标标签作为输入
    def forward(self, feat, targets=None):
        # 获取最后一个特征
        feat = feat[-1]
        # 获取特征的形状信息
        b, c, h, w = feat.shape
        # 将特征展平为二维特征图
        feat = feat.reshape([b, c, h * w])
        # 转置特征维度
        feat = feat.transpose((0, 2, 1))
        # 对特征进行位置编码
        out_enc = self.positional_encoding(feat)
        # 如果处于训练状态，则调用训练过程的前向传播函数
        if self.training:
            return self.forward_train(out_enc, targets)
        # 否则调用测试过程的前向传播函数
        return self.forward_test(out_enc)
class DecoderLayer(nn.Layer):
    """
    Decoder is made of self attention, srouce attention and feed forward.
    """

    def __init__(self, headers, d_model, dropout, d_ff):
        # 初始化 DecoderLayer 类
        super(DecoderLayer, self).__init__()
        # 创建 self attention、source attention 和 feed forward 层
        self.self_attn = MultiHeadAttention(headers, d_model, dropout)
        self.src_attn = MultiHeadAttention(headers, d_model, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.sublayer = clones(SubLayerConnection(d_model, dropout), 3)

    def forward(self, x, feature, src_mask, tgt_mask):
        # 使用 self attention 层
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        # 使用 source attention 层
        x = self.sublayer[1](
            x, lambda x: self.src_attn(x, feature, feature, src_mask))
        # 返回 feed forward 层的结果
        return self.sublayer[2](x, self.feed_forward)


class MultiHeadAttention(nn.Layer):
    def __init__(self, headers, d_model, dropout):
        # 初始化 MultiHeadAttention 类
        super(MultiHeadAttention, self).__init__()

        assert d_model % headers == 0
        # 计算每个头的维度
        self.d_k = int(d_model / headers)
        self.headers = headers
        # 创建线性层
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        B = query.shape[0]

        # 1) 批量进行线性投影，从 d_model => h x d_k
        query, key, value = \
            [l(x).reshape([B, 0, self.headers, self.d_k]).transpose([0, 2, 1, 3])
             for l, x in zip(self.linears, (query, key, value))]
        # 2) 在批量中应用注意力机制
        x, self.attn = self_attention(
            query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose([0, 2, 1, 3]).reshape([B, 0, self.headers * self.d_k])
        return self.linears[-1](x)


class FeedForward(nn.Layer):
    # 定义一个名为 FeedForward 的类，继承自 nn.Module 类
    def __init__(self, d_model, d_ff, dropout):
        # 调用父类的构造函数
        super(FeedForward, self).__init__()
        # 创建一个线性变换层，输入维度为 d_model，输出维度为 d_ff
        self.w_1 = nn.Linear(d_model, d_ff)
        # 创建另一个线性变换层，输入维度为 d_ff，输出维度为 d_model
        self.w_2 = nn.Linear(d_ff, d_model)
        # 创建一个丢弃层，用于随机丢弃部分神经元以防止过拟合
        self.dropout = nn.Dropout(dropout)

    # 定义前向传播函数
    def forward(self, x):
        # 先将输入 x 经过第一个线性变换层 w_1，然后经过激活函数 ReLU，再经过丢弃层
        # 最后再经过第二个线性变换层 w_2，得到最终的输出
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
class SubLayerConnection(nn.Layer):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        # 初始化函数，包含层归一化和 dropout 操作
        super(SubLayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # 实现残差连接和层归一化
        return x + self.dropout(sublayer(self.norm(x)))


def masked_fill(x, mask, value):
    # 根据掩码填充张量
    mask = mask.astype(x.dtype)
    return x * paddle.logical_not(mask).astype(x.dtype) + mask * value


def self_attention(query, key, value, mask=None, dropout=None):
    """
    Compute 'Scale Dot Product Attention'
    """
    d_k = value.shape[-1]

    # 计算注意力分数
    score = paddle.matmul(query, key.transpose([0, 1, 3, 2]) / math.sqrt(d_k))
    if mask is not None:
        # 使用掩码填充注意力分数
        score = masked_fill(score, mask == 0, -6.55e4)  # for fp16

    # 计算注意力权重
    p_attn = F.softmax(score, axis=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)
    return paddle.matmul(p_attn, value), p_attn


def clones(module, N):
    """ Produce N identical layers """
    # 复制 N 个相同的层
    return nn.LayerList([copy.deepcopy(module) for _ in range(N)])


class Embeddings(nn.Layer):
    def __init__(self, d_model, vocab):
        # 初始化函数，包含词嵌入层
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, *input):
        x = input[0]
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Layer):
    """ Implement the PE function. """
    # 实现位置编码函数
    # 初始化函数，设置位置编码的参数
    def __init__(self, d_model, dropout=0., max_len=5000):
        # 调用父类的初始化函数
        super(PositionalEncoding, self).__init__()
        # 初始化丢弃率
        self.dropout = nn.Dropout(p=dropout)

        # 在对数空间中计算位置编码
        # 创建一个形状为[max_len, d_model]的全零张量
        pe = paddle.zeros([max_len, d_model])
        # 生成位置信息，从0到max_len，然后增加一个维度，转换为float32类型
        position = paddle.arange(0, max_len).unsqueeze(1).astype('float32')
        # 计算除数项，用于计算正弦和余弦值
        div_term = paddle.exp(
            paddle.arange(0, d_model, 2) * -math.log(10000.0) / d_model)
        # 计算正弦值和余弦值，填充到位置编码张量中
        pe[:, 0::2] = paddle.sin(position * div_term)
        pe[:, 1::2] = paddle.cos(position * div_term)
        # 增加一个维度，将位置编码张量变为形状为[1, max_len, d_model]
        pe = pe.unsqueeze(0)
        # 将位置编码张量注册为模型的缓冲区
        self.register_buffer('pe', pe)

    # 前向传播函数，用于计算位置编码后的特征
    def forward(self, feat, **kwargs):
        # 将输入特征与位置编码相加，只取位置编码的前feat的长度部分
        feat = feat + self.pe[:, :paddle.shape(feat)[1]]  # pe 1*5000*512
        # 对特征进行丢弃操作
        return self.dropout(feat)
```