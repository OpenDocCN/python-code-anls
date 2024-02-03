# `.\PaddleOCR\ppocr\modeling\heads\rec_nrtr_head.py`

```
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 均基于“原样”分发，没有任何明示或暗示的保证
# 请查看许可证以获取特定语言的权限和限制

# 导入所需的库
import math
import paddle
from paddle import nn
import paddle.nn.functional as F
from paddle.nn import LayerList
# from paddle.nn.initializer import XavierNormal as xavier_uniform_
from paddle.nn import Dropout, Linear, LayerNorm
import numpy as np
from ppocr.modeling.backbones.rec_svtrnet import Mlp, zeros_, ones_
from paddle.nn.initializer import XavierNormal as xavier_normal_

# 定义 Transformer 类，基于论文“Attention Is All You Need”构建
class Transformer(nn.Layer):
    """A transformer model. User is able to modify the attributes as needed. The architechture
    is based on the paper "Attention Is All You Need". Ashish Vaswani, Noam Shazeer,
    Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and
    Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information
    Processing Systems, pages 6000-6010.
    Args:
        d_model: 编码器/解码器输入中预期特征的数量（默认为512）。
        nhead: 多头注意力模型中的头数（默认为8）。
        num_encoder_layers: 编码器中子编码器层的数量（默认为6）。
        num_decoder_layers: 解码器中子解码器层的数量（默认为6）。
        dim_feedforward: 前馈网络模型的维度（默认为2048）。
        dropout: 丢弃率（默认为0.1）。
        custom_encoder: 自定义编码器（默认为None）。
        custom_decoder: 自定义解码器（默认为None）。
    """

    def _init_weights(self, m):
        # 初始化权重
        if isinstance(m, nn.Linear):
            xavier_normal_(m.weight)  # 使用Xavier正态分布初始化权重
            if m.bias is not None:
                zeros_(m.bias)  # 初始化偏置为0

    def forward_train(self, src, tgt):
        tgt = tgt[:, :-1]  # 去掉目标序列的最后一个元素

        tgt = self.embedding(tgt)  # 嵌入目标序列
        tgt = self.positional_encoding(tgt)  # 添加位置编码
        tgt_mask = self.generate_square_subsequent_mask(tgt.shape[1])  # 生成目标序列的掩码

        if self.encoder is not None:
            src = self.positional_encoding(src)  # 添加位置编码到源序列
            for encoder_layer in self.encoder:
                src = encoder_layer(src)  # 编码器层处理源序列
            memory = src  # B N C
        else:
            memory = src  # B N C
        for decoder_layer in self.decoder:
            tgt = decoder_layer(tgt, memory, self_mask=tgt_mask)  # 解码器层处理目标序列
        output = tgt
        logit = self.tgt_word_prj(output)  # 输出层
        return logit
    # 定义一个方法用于处理输入的源序列和目标序列
    def forward(self, src, targets=None):
        """Take in and process masked source/target sequences.
        Args:
            src: the sequence to the encoder (required).  # 源序列，传递给编码器的序列（必需）
            tgt: the sequence to the decoder (required).  # 目标序列，传递给解码器的序列（必需）
        Shape:
            - src: :math:`(B, sN, C)`.  # 源序列的形状
            - tgt: :math:`(B, tN, C)`.  # 目标序列的形状
        Examples:
            >>> output = transformer_model(src, tgt)  # 示例调用transformer_model函数
        """

        # 如果处于训练模式
        if self.training:
            # 获取目标序列中的最大长度
            max_len = targets[1].max()
            # 截取目标序列的长度为2+max_len
            tgt = targets[0][:, :2 + max_len]
            # 调用forward_train方法处理源序列和截取后的目标序列
            return self.forward_train(src, tgt)
        else:
            # 如果不处于训练模式
            if self.beam_size > 0:
                # 如果beam_size大于0，则调用forward_beam方法处理源序列
                return self.forward_beam(src)
            else:
                # 如果beam_size等于0，则调用forward_test方法处理源序列
                return self.forward_test(src)
    # 执行前向推理测试，输入为src
    def forward_test(self, src):
        # 获取batch size
        bs = paddle.shape(src)[0]
        # 如果存在编码器
        if self.encoder is not None:
            # 对输入进行位置编码
            src = self.positional_encoding(src)
            # 遍历编码器层
            for encoder_layer in self.encoder:
                src = encoder_layer(src)
            # 将编码结果作为记忆
            memory = src  # B N C
        else:
            # 如果没有编码器，则记忆为输入
            memory = src
        # 初始化解码序列为起始符号
        dec_seq = paddle.full((bs, 1), 2, dtype=paddle.int64)
        # 初始化解码概率为1
        dec_prob = paddle.full((bs, 1), 1., dtype=paddle.float32)
        # 循环生成解码序列
        for len_dec_seq in range(1, paddle.to_tensor(self.max_len)):
            # 对解码序列进行嵌入
            dec_seq_embed = self.embedding(dec_seq)
            # 对解码序列进行位置编码
            dec_seq_embed = self.positional_encoding(dec_seq_embed)
            # 生成目标掩码
            tgt_mask = self.generate_square_subsequent_mask(
                paddle.shape(dec_seq_embed)[1])
            # 将解码序列嵌入作为目标
            tgt = dec_seq_embed
            # 遍历解码器层
            for decoder_layer in self.decoder:
                tgt = decoder_layer(tgt, memory, self_mask=tgt_mask)
            # 获取解码输出
            dec_output = tgt
            # 取解码输出的最后一个位置作为当前预测
            dec_output = dec_output[:, -1, :]
            # 对预测结果进行softmax得到单词概率
            word_prob = F.softmax(self.tgt_word_prj(dec_output), axis=-1)
            # 获取预测的单词索引
            preds_idx = paddle.argmax(word_prob, axis=-1)
            # 如果预测为终止符号，则结束解码
            if paddle.equal_all(
                    preds_idx,
                    paddle.full(
                        paddle.shape(preds_idx), 3, dtype='int64')):
                break
            # 获取预测的概率
            preds_prob = paddle.max(word_prob, axis=-1)
            # 将预测的单词索引添加到解码序列中
            dec_seq = paddle.concat(
                [dec_seq, paddle.reshape(preds_idx, [-1, 1])], axis=1)
            # 将预测的概率添加到解码概率中
            dec_prob = paddle.concat(
                [dec_prob, paddle.reshape(preds_prob, [-1, 1])], axis=1)
        # 返回解码序列和解码概率
        return [dec_seq, dec_prob]
    # 生成一个方形的序列掩码。被掩盖的位置填充为负无穷大。
    # 未被掩盖的位置填充为0.0。
    def generate_square_subsequent_mask(self, sz):
        # 创建一个形状为(sz, sz)的全零张量，数据类型为float32
        mask = paddle.zeros([sz, sz], dtype='float32')
        # 创建一个上三角矩阵，对角线以上的元素填充为负无穷大
        mask_inf = paddle.triu(
            # 创建一个形状为(sz, sz)的张量，数据类型为float32，填充值为负无穷大
            paddle.full(
                shape=[sz, sz], dtype='float32', fill_value='-inf'),
            diagonal=1)
        # 将原始掩码和负无穷大掩码相加
        mask = mask + mask_inf
        # 在第0和第1维度上增加一个维度
        return mask.unsqueeze([0, 1])
class MultiheadAttention(nn.Layer):
    """Allows the model to jointly attend to information
    from different representation subspaces.
    See reference: Attention Is All You Need

    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O
        \text{where} head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)

    Args:
        embed_dim: total dimension of the model
        num_heads: parallel attention layers, or heads

    """

    def __init__(self, embed_dim, num_heads, dropout=0., self_attn=False):
        # 初始化 MultiheadAttention 类
        super(MultiheadAttention, self).__init__()
        # 设置总维度
        self.embed_dim = embed_dim
        # 设置并行注意力层或头的数量
        self.num_heads = num_heads
        # 设置每个头的维度
        self.head_dim = embed_dim // num_heads
        # 确保 embed_dim 必须能被 num_heads 整除
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        # 设置缩放因子
        self.scale = self.head_dim**-0.5
        # 设置是否为自注意力机制
        self.self_attn = self_attn
        # 如果是自注意力机制
        if self_attn:
            # 创建线性层 qkv，用于计算 Q、K、V
            self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        else:
            # 创建线性层 q，用于计算 Q
            self.q = nn.Linear(embed_dim, embed_dim)
            # 创建线性层 kv，用于计算 K、V
            self.kv = nn.Linear(embed_dim, embed_dim * 2)
        # 创建 Dropout 层，用于注意力权重的随机失活
        self.attn_drop = nn.Dropout(dropout)
        # 创建线性层 out_proj，用于最终输出
        self.out_proj = nn.Linear(embed_dim, embed_dim)
    # 定义一个前向传播函数，接受查询(query)、键(key)和注意力掩码(attn_mask)作为输入
    def forward(self, query, key=None, attn_mask=None):

        # 获取查询(query)的序列长度
        qN = query.shape[1]

        # 如果是自注意力机制
        if self.self_attn:
            # 使用查询(query)经过全连接层(qkv)得到qkv张量，然后重塑形状
            qkv = self.qkv(query).reshape(
                (0, qN, 3, self.num_heads, self.head_dim)).transpose(
                    (2, 0, 3, 1, 4))
            q, k, v = qkv[0], qkv[1], qkv[2]
        else:
            # 获取键(key)的序列长度
            kN = key.shape[1]
            # 使用查询(query)经过全连接层(q)得到q张量，然后重塑形状
            q = self.q(query).reshape(
                [0, qN, self.num_heads, self.head_dim]).transpose([0, 2, 1, 3])
            # 使用键(key)经过全连接层(kv)得到kv张量，然后重塑形状
            kv = self.kv(key).reshape(
                (0, kN, 2, self.num_heads, self.head_dim)).transpose(
                    (2, 0, 3, 1, 4))
            k, v = kv[0], kv[1]

        # 计算注意力分数
        attn = (q.matmul(k.transpose((0, 1, 3, 2)))) * self.scale

        # 如果存在注意力掩码，则将其加到注意力分数上
        if attn_mask is not None:
            attn += attn_mask

        # 对注意力分数进行 softmax 操作
        attn = F.softmax(attn, axis=-1)
        # 对注意力分数进行 dropout 操作
        attn = self.attn_drop(attn)

        # 计算加权后的值
        x = (attn.matmul(v)).transpose((0, 2, 1, 3)).reshape(
            (0, qN, self.embed_dim))
        # 通过输出投影层得到最终输出
        x = self.out_proj(x)

        # 返回输出结果
        return x
# 定义 Transformer 模型中的一个 TransformerBlock 类，用于实现一个 Transformer 模块
class TransformerBlock(nn.Layer):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 attention_dropout_rate=0.0,
                 residual_dropout_rate=0.1,
                 with_self_attn=True,
                 with_cross_attn=False,
                 epsilon=1e-5):
        # 调用父类的初始化方法
        super(TransformerBlock, self).__init__()
        # 设置是否包含自注意力机制
        self.with_self_attn = with_self_attn
        if with_self_attn:
            # 如果包含自注意力机制，则创建 MultiheadAttention 层
            self.self_attn = MultiheadAttention(
                d_model,
                nhead,
                dropout=attention_dropout_rate,
                self_attn=with_self_attn)
            # 创建 LayerNorm 层
            self.norm1 = LayerNorm(d_model, epsilon=epsilon)
            # 创建 Dropout 层
            self.dropout1 = Dropout(residual_dropout_rate)
        # 设置是否包含跨注意力机制
        self.with_cross_attn = with_cross_attn
        if with_cross_attn:
            # 如果包含跨注意力机制，则创建 MultiheadAttention 层
            self.cross_attn = MultiheadAttention(  #for self_attn of encoder or cross_attn of decoder
                d_model,
                nhead,
                dropout=attention_dropout_rate)
            # 创建 LayerNorm 层
            self.norm2 = LayerNorm(d_model, epsilon=epsilon)
            # 创建 Dropout 层
            self.dropout2 = Dropout(residual_dropout_rate)

        # 创建 Mlp 层
        self.mlp = Mlp(in_features=d_model,
                       hidden_features=dim_feedforward,
                       act_layer=nn.ReLU,
                       drop=residual_dropout_rate)

        # 创建 LayerNorm 层
        self.norm3 = LayerNorm(d_model, epsilon=epsilon)

        # 创建 Dropout 层
        self.dropout3 = Dropout(residual_dropout_rate)

    # 定义前向传播方法
    def forward(self, tgt, memory=None, self_mask=None, cross_mask=None):
        # 如果包含自注意力机制
        if self.with_self_attn:
            # 使用自注意力机制处理输入数据
            tgt1 = self.self_attn(tgt, attn_mask=self_mask)
            # 对处理后的数据进行 LayerNorm 和 Dropout 处理
            tgt = self.norm1(tgt + self.dropout1(tgt1))

        # 如果包含跨注意力机制
        if self.with_cross_attn:
            # 使用跨注意力机制处理输入数据
            tgt2 = self.cross_attn(tgt, key=memory, attn_mask=cross_mask)
            # 对处理后的数据进行 LayerNorm 和 Dropout 处理
            tgt = self.norm2(tgt + self.dropout2(tgt2))
        
        # 对处理后的数据进行 LayerNorm、Dropout 和 MLP 处理
        tgt = self.norm3(tgt + self.dropout3(self.mlp(tgt)))
        # 返回处理后的数据
        return tgt


# 定义 PositionalEncoding 类
class PositionalEncoding(nn.Layer):
    """Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    # 定义一个类 PositionalEncoding，用于为序列中的标记注入关于其相对或绝对位置的一些信息
    class PositionalEncoding(nn.Layer):
        
        # 初始化函数，设置位置编码的参数
        def __init__(self, dropout, dim, max_len=5000):
            super(PositionalEncoding, self).__init__()
            self.dropout = nn.Dropout(p=dropout)

            # 创建一个形状为 [max_len, dim] 的零张量 pe
            pe = paddle.zeros([max_len, dim])
            # 创建一个表示位置的张量 position，范围为 [0, max_len)，并增加一个维度
            position = paddle.arange(0, max_len, dtype=paddle.float32).unsqueeze(1)
            # 计算除数项 div_term，用于计算正弦和余弦值
            div_term = paddle.exp(
                paddle.arange(0, dim, 2).astype('float32') *
                (-math.log(10000.0) / dim))
            # 计算正弦值和余弦值，并将它们分别赋值给 pe 的偶数列和奇数列
            pe[:, 0::2] = paddle.sin(position * div_term)
            pe[:, 1::2] = paddle.cos(position * div_term)
            # 增加一个维度，并转置 pe 的维度顺序
            pe = paddle.unsqueeze(pe, 0)
            pe = paddle.transpose(pe, [1, 0, 2])
            # 将 pe 注册为模型的缓冲区
            self.register_buffer('pe', pe)

        # 前向传播函数，用于计算位置编码后的输出
        def forward(self, x):
            """Inputs of forward function
            Args:
                x: the sequence fed to the positional encoder model (required).
            Shape:
                x: [sequence length, batch size, embed dim]
                output: [sequence length, batch size, embed dim]
            Examples:
                >>> output = pos_encoder(x)
            """
            # 转置输入张量 x 的维度顺序
            x = x.transpose([1, 0, 2])
            # 将位置编码加到输入张量 x 上
            x = x + self.pe[:paddle.shape(x)[0], :]
            # 对加了位置编码的张量进行 dropout 处理，并再次转置维度顺序
            return self.dropout(x).transpose([1, 0, 2])
class PositionalEncoding_2d(nn.Layer):
    """Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, dropout, dim, max_len=5000):
        # 初始化 PositionalEncoding_2d 类
        super(PositionalEncoding_2d, self).__init__()
        # 设置 dropout 层
        self.dropout = nn.Dropout(p=dropout)

        # 创建一个形状为 [max_len, dim] 的全零张量 pe
        pe = paddle.zeros([max_len, dim])
        # 创建一个表示位置的张量 position
        position = paddle.arange(0, max_len, dtype=paddle.float32).unsqueeze(1)
        # 计算 div_term，用于后续计算 sin 和 cos
        div_term = paddle.exp(
            paddle.arange(0, dim, 2).astype('float32') *
            (-math.log(10000.0) / dim))
        # 计算 sin 和 cos，并将结果存入 pe 张量
        pe[:, 0::2] = paddle.sin(position * div_term)
        pe[:, 1::2] = paddle.cos(position * div_term)
        # 转置 pe 张量，并在第 0 维度上增加一个维度
        pe = paddle.transpose(paddle.unsqueeze(pe, 0), [1, 0, 2])
        # 将 pe 注册为模型的缓冲区
        self.register_buffer('pe', pe)

        # 创建自适应平均池化层 avg_pool_1
        self.avg_pool_1 = nn.AdaptiveAvgPool2D((1, 1))
        # 创建线性层 linear1
        self.linear1 = nn.Linear(dim, dim)
        # 初始化 linear1 的权重为全 1
        self.linear1.weight.data.fill_(1.)
        # 创建自适应平均池化层 avg_pool_2
        self.avg_pool_2 = nn.AdaptiveAvgPool2D((1, 1))
        # 创建线性层 linear2
        self.linear2 = nn.Linear(dim, dim)
        # 初始化 linear2 的权重为全 1
        self.linear2.weight.data.fill_(1.)
    # 定义前向传播函数，接受输入 x
    def forward(self, x):
        """Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        # 从位置编码矩阵中获取与输入 x 相关的部分
        w_pe = self.pe[:paddle.shape(x)[-1], :]
        # 对输入 x 进行平均池化并压缩，然后通过线性层 linear1 处理并增加一个维度
        w1 = self.linear1(self.avg_pool_1(x).squeeze()).unsqueeze(0)
        # 对位置编码矩阵 w_pe 和 w1 进行逐元素相乘
        w_pe = w_pe * w1
        # 调整 w_pe 的维度顺序
        w_pe = paddle.transpose(w_pe, [1, 2, 0])
        # 在 w_pe 的第二个维度上增加一个维度
        w_pe = paddle.unsqueeze(w_pe, 2)

        # 从位置编码矩阵中获取与输入 x 相关的部分
        h_pe = self.pe[:paddle.shape(x).shape[-2], :]
        # 对输入 x 进行平均池化并压缩，然后通过线性层 linear2 处理并增加一个维度
        w2 = self.linear2(self.avg_pool_2(x).squeeze()).unsqueeze(0)
        # 对位置编码矩阵 h_pe 和 w2 进行逐元素相乘
        h_pe = h_pe * w2
        # 调整 h_pe 的维度顺序
        h_pe = paddle.transpose(h_pe, [1, 2, 0])
        # 在 h_pe 的第三个维度上增加一个维度
        h_pe = paddle.unsqueeze(h_pe, 3)

        # 将输入 x、w_pe 和 h_pe 相加
        x = x + w_pe + h_pe
        # 调整 x 的维度顺序并将其形状重塑
        x = paddle.transpose(
            paddle.reshape(x,
                           [x.shape[0], x.shape[1], x.shape[2] * x.shape[3]]),
            [2, 0, 1])

        # 对 x 进行 dropout 操作后返回结果
        return self.dropout(x)
class Embeddings(nn.Layer):
    # 定义 Embeddings 类，用于处理词嵌入
    def __init__(self, d_model, vocab, padding_idx=None, scale_embedding=True):
        # 初始化函数，接受词嵌入维度、词汇量、填充索引和是否缩放词嵌入等参数
        super(Embeddings, self).__init__()
        # 调用父类的初始化函数
        self.embedding = nn.Embedding(vocab, d_model, padding_idx=padding_idx)
        # 创建 Embedding 层，指定词汇量和词嵌入维度
        w0 = np.random.normal(0.0, d_model**-0.5,
                              (vocab, d_model)).astype(np.float32)
        # 生成随机初始化的词嵌入权重
        self.embedding.weight.set_value(w0)
        # 将生成的权重设置给 Embedding 层
        self.d_model = d_model
        # 保存词嵌入维度
        self.scale_embedding = scale_embedding
        # 保存是否缩放词嵌入的标志

    def forward(self, x):
        # 前向传播函数，接受输入 x
        if self.scale_embedding:
            # 如果需要缩放词嵌入
            x = self.embedding(x)
            # 获取词嵌入
            return x * math.sqrt(self.d_model)
            # 返回缩放后的词嵌入
        return self.embedding(x)
        # 返回词嵌入

class Beam():
    """ Beam search """
    # 定义 Beam 类，用于实现 Beam Search 算法

    def __init__(self, size, device=False):
        # 初始化函数，接受 Beam 大小和设备参数
        self.size = size
        # 保存 Beam 大小
        self._done = False
        # 初始化完成标志为 False
        self.scores = paddle.zeros((size, ), dtype=paddle.float32)
        # 初始化得分数组
        self.all_scores = []
        # 保存所有得分
        self.prev_ks = []
        # 保存每个时间步的回溯指针
        self.next_ys = [paddle.full((size, ), 0, dtype=paddle.int64)]
        # 初始化输出序列
        self.next_ys[0][0] = 2
        # 设置初始输出序列的第一个元素为特殊标记 2

    def get_current_state(self):
        # 获取当前时间步的输出
        return self.get_tentative_hypothesis()

    def get_current_origin(self):
        # 获取当前时间步的回溯指针
        return self.prev_ks[-1]

    @property
    def done(self):
        # 返回完成标志
        return self._done
    def advance(self, word_prob):
        "Update beam status and check if finished or not."
        # 获取词汇概率矩阵的列数
        num_words = word_prob.shape[1]

        # 计算当前步骤的分数
        if len(self.prev_ks) > 0:
            # 将前一步的分数与当前词汇概率相加
            beam_lk = word_prob + self.scores.unsqueeze(1).expand_as(word_prob)
        else:
            beam_lk = word_prob[0]

        # 将二维矩阵展平为一维
        flat_beam_lk = beam_lk.reshape([-1])
        # 获取分数最高的前 self.size 个分数及其对应的索引
        best_scores, best_scores_id = flat_beam_lk.topk(self.size, 0, True, True)  # 1st sort
        # 将当前分数添加到历史分数列表中
        self.all_scores.append(self.scores)
        self.scores = best_scores
        # 将最高分数的索引转换为二维索引
        prev_k = best_scores_id // num_words
        self.prev_ks.append(prev_k)
        self.next_ys.append(best_scores_id - prev_k * num_words)
        # 判断是否结束条件为最高分数对应的词为 EOS
        if self.next_ys[-1][0] == 3:
            self._done = True
            self.all_scores.append(self.scores)

        return self._done

    def sort_scores(self):
        "Sort the scores."
        # 对分数进行排序
        return self.scores, paddle.to_tensor(
            [i for i in range(int(self.scores.shape[0]))], dtype='int32')

    def get_the_best_score_and_idx(self):
        "Get the score of the best in the beam."
        # 获取最高分数及其索引
        scores, ids = self.sort_scores()
        return scores[1], ids[1]

    def get_tentative_hypothesis(self):
        "Get the decoded sequence for the current timestep."
        if len(self.next_ys) == 1:
            # 如果只有一个步骤，则直接获取当前步骤的解码序列
            dec_seq = self.next_ys[0].unsqueeze(1)
        else:
            _, keys = self.sort_scores()
            # 获取所有候选序列
            hyps = [self.get_hypothesis(k) for k in keys]
            hyps = [[2] + h for h in hyps]
            dec_seq = paddle.to_tensor(hyps, dtype='int64')
        return dec_seq
    # 获取假设序列，根据当前位置 k 回溯构建完整的假设序列
    def get_hypothesis(self, k):
        """ Walk back to construct the full hypothesis. """
        # 初始化假设序列
        hyp = []
        # 从最后一个位置开始向前遍历
        for j in range(len(self.prev_ks) - 1, -1, -1):
            # 将当前位置的预测结果添加到假设序列中
            hyp.append(self.next_ys[j + 1][k])
            # 更新当前位置 k 到前一个位置 k 的映射
            k = self.prev_ks[j][k]
        # 返回构建好的假设序列，将结果转换为列表形式
        return list(map(lambda x: x.item(), hyp[::-1]))
```