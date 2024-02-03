# `.\PaddleOCR\ppocr\modeling\heads\self_attention.py`

```py
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"原样"的基础分发的，
# 没有任何明示或暗示的保证或条件
# 请查看许可证以获取特定语言的权限和限制

# 导入必要的库
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

# 导入 Paddle 库
import paddle
from paddle import ParamAttr, nn
from paddle import nn, ParamAttr
from paddle.nn import functional as F
import numpy as np

# 设置梯度裁剪的阈值
gradient_clip = 10

# 定义一个包装编码器的类
class WrapEncoderForFeature(nn.Layer):
    def __init__(self,
                 src_vocab_size,
                 max_length,
                 n_layer,
                 n_head,
                 d_key,
                 d_value,
                 d_model,
                 d_inner_hid,
                 prepostprocess_dropout,
                 attention_dropout,
                 relu_dropout,
                 preprocess_cmd,
                 postprocess_cmd,
                 weight_sharing,
                 bos_idx=0):
        super(WrapEncoderForFeature, self).__init__()

        # 准备编码器，包括词汇表大小、模型维度、最大长度等参数
        self.prepare_encoder = PrepareEncoder(
            src_vocab_size,
            d_model,
            max_length,
            prepostprocess_dropout,
            bos_idx=bos_idx,
            word_emb_param_name="src_word_emb_table")
        
        # 编码器，包括层数、头数、键值维度、模型维度等参数
        self.encoder = Encoder(n_layer, n_head, d_key, d_value, d_model,
                               d_inner_hid, prepostprocess_dropout,
                               attention_dropout, relu_dropout, preprocess_cmd,
                               postprocess_cmd)
    # 定义一个前向传播函数，接受编码器输入
    def forward(self, enc_inputs):
        # 解包编码器输入，包括卷积特征、源位置、自注意力偏置
        conv_features, src_pos, src_slf_attn_bias = enc_inputs
        # 准备编码器输入，将卷积特征和源位置信息进行处理
        enc_input = self.prepare_encoder(conv_features, src_pos)
        # 将处理后的编码器输入传入编码器模型，得到编码器输出
        enc_output = self.encoder(enc_input, src_slf_attn_bias)
        # 返回编码器输出
        return enc_output
# 定义一个包含嵌入器和编码器的类
class WrapEncoder(nn.Layer):
    """
    embedder + encoder
    """

    def __init__(self,
                 src_vocab_size,
                 max_length,
                 n_layer,
                 n_head,
                 d_key,
                 d_value,
                 d_model,
                 d_inner_hid,
                 prepostprocess_dropout,
                 attention_dropout,
                 relu_dropout,
                 preprocess_cmd,
                 postprocess_cmd,
                 weight_sharing,
                 bos_idx=0):
        # 调用父类的构造函数
        super(WrapEncoder, self).__init__()

        # 创建一个准备解码器对象
        self.prepare_decoder = PrepareDecoder(
            src_vocab_size,
            d_model,
            max_length,
            prepostprocess_dropout,
            bos_idx=bos_idx)
        # 创建一个编码器对象
        self.encoder = Encoder(n_layer, n_head, d_key, d_value, d_model,
                               d_inner_hid, prepostprocess_dropout,
                               attention_dropout, relu_dropout, preprocess_cmd,
                               postprocess_cmd)

    # 定义前向传播函数
    def forward(self, enc_inputs):
        # 解包编码器输入
        src_word, src_pos, src_slf_attn_bias = enc_inputs
        # 准备编码器输入
        enc_input = self.prepare_decoder(src_word, src_pos)
        # 编码器进行编码
        enc_output = self.encoder(enc_input, src_slf_attn_bias)
        # 返回编码结果
        return enc_output


# 定义一个编码器类
class Encoder(nn.Layer):
    """
    encoder
    """
    # 初始化编码器对象，设置各种参数
    def __init__(self,
                 n_layer,
                 n_head,
                 d_key,
                 d_value,
                 d_model,
                 d_inner_hid,
                 prepostprocess_dropout,
                 attention_dropout,
                 relu_dropout,
                 preprocess_cmd="n",
                 postprocess_cmd="da"):

        # 调用父类的初始化方法
        super(Encoder, self).__init__()

        # 初始化编码器层列表
        self.encoder_layers = list()
        # 循环创建编码器层对象并添加到编码器层列表中
        for i in range(n_layer):
            self.encoder_layers.append(
                self.add_sublayer(
                    "layer_%d" % i,
                    EncoderLayer(n_head, d_key, d_value, d_model, d_inner_hid,
                                 prepostprocess_dropout, attention_dropout,
                                 relu_dropout, preprocess_cmd,
                                 postprocess_cmd)))
        # 创建预处理后处理层对象
        self.processer = PrePostProcessLayer(preprocess_cmd, d_model,
                                             prepostprocess_dropout)

    # 编码器前向传播方法
    def forward(self, enc_input, attn_bias):
        # 遍历编码器层列表，对输入进行编码
        for encoder_layer in self.encoder_layers:
            enc_output = encoder_layer(enc_input, attn_bias)
            enc_input = enc_output
        # 对编码后的输出进行处理
        enc_output = self.processer(enc_output)
        # 返回处理后的输出
        return enc_output
class EncoderLayer(nn.Layer):
    """
    EncoderLayer
    """

    def __init__(self,
                 n_head,
                 d_key,
                 d_value,
                 d_model,
                 d_inner_hid,
                 prepostprocess_dropout,
                 attention_dropout,
                 relu_dropout,
                 preprocess_cmd="n",
                 postprocess_cmd="da"):

        # 调用父类的构造函数
        super(EncoderLayer, self).__init__()
        # 创建预处理层1对象
        self.preprocesser1 = PrePostProcessLayer(preprocess_cmd, d_model,
                                                 prepostprocess_dropout)
        # 创建多头注意力层对象
        self.self_attn = MultiHeadAttention(d_key, d_value, d_model, n_head,
                                            attention_dropout)
        # 创建后处理层1对象
        self.postprocesser1 = PrePostProcessLayer(postprocess_cmd, d_model,
                                                  prepostprocess_dropout)

        # 创建预处理层2对象
        self.preprocesser2 = PrePostProcessLayer(preprocess_cmd, d_model,
                                                 prepostprocess_dropout)
        # 创建前馈神经网络对象
        self.ffn = FFN(d_inner_hid, d_model, relu_dropout)
        # 创建后处理层2对象
        self.postprocesser2 = PrePostProcessLayer(postprocess_cmd, d_model,
                                                  prepostprocess_dropout)

    def forward(self, enc_input, attn_bias):
        # 使用预处理层1对输入进行预处理
        attn_output = self.self_attn(
            self.preprocesser1(enc_input), None, None, attn_bias)
        # 使用后处理层1对注意力输出进行后处理
        attn_output = self.postprocesser1(attn_output, enc_input)
        # 使用预处理层2对注意力输出进行预处理
        ffn_output = self.ffn(self.preprocesser2(attn_output))
        # 使用后处理层2对前馈神经网络输出进行后处理
        ffn_output = self.postprocesser2(ffn_output, attn_output)
        # 返回前馈神经网络输出
        return ffn_output


class MultiHeadAttention(nn.Layer):
    """
    Multi-Head Attention
    """
    # 初始化多头注意力机制模块
    def __init__(self, d_key, d_value, d_model, n_head=1, dropout_rate=0.):
        # 调用父类的初始化方法
        super(MultiHeadAttention, self).__init__()
        # 设置多头注意力机制的头数
        self.n_head = n_head
        # 设置键的维度
        self.d_key = d_key
        # 设置值的维度
        self.d_value = d_value
        # 设置模型的维度
        self.d_model = d_model
        # 设置dropout的比率
        self.dropout_rate = dropout_rate
        # 创建一个线性层，用于计算查询的线性变换
        self.q_fc = paddle.nn.Linear(
            in_features=d_model, out_features=d_key * n_head, bias_attr=False)
        # 创建一个线性层，用于计算键的线性变换
        self.k_fc = paddle.nn.Linear(
            in_features=d_model, out_features=d_key * n_head, bias_attr=False)
        # 创建一个线性层，用于计算值的线性变换
        self.v_fc = paddle.nn.Linear(
            in_features=d_model, out_features=d_value * n_head, bias_attr=False)
        # 创建一个线性层，用于将多头注意力机制的输出映射回原始维度
        self.proj_fc = paddle.nn.Linear(
            in_features=d_value * n_head, out_features=d_model, bias_attr=False)
    # 准备查询、键、值的操作，用于自注意力或交叉注意力
    def _prepare_qkv(self, queries, keys, values, cache=None):
        # 如果键为空，则为自注意力
        if keys is None:  
            keys, values = queries, queries
            static_kv = False
        else:  # 否则为交叉注意力
            static_kv = True

        # 对查询进行全连接操作
        q = self.q_fc(queries)
        # 重塑查询的形状
        q = paddle.reshape(x=q, shape=[0, 0, self.n_head, self.d_key])
        # 转置查询的维度
        q = paddle.transpose(x=q, perm=[0, 2, 1, 3])

        # 如果存在缓存且为静态键值对，并且缓存中有静态键
        if cache is not None and static_kv and "static_k" in cache:
            # 用于推理中的编码器-解码器注意力且已缓存
            k = cache["static_k"]
            v = cache["static_v"]
        else:
            # 对键进行全连接操作
            k = self.k_fc(keys)
            # 对值进行全连接操作
            v = self.v_fc(values)
            # 重塑键的形状
            k = paddle.reshape(x=k, shape=[0, 0, self.n_head, self.d_key])
            # 转置键的维度
            k = paddle.transpose(x=k, perm=[0, 2, 1, 3])
            # 重塑值的形状
            v = paddle.reshape(x=v, shape=[0, 0, self.n_head, self.d_value])
            # 转置值的维度
            v = paddle.transpose(x=v, perm=[0, 2, 1, 3])

        # 如果存在缓存
        if cache is not None:
            if static_kv and not "static_k" in cache:
                # 用于推理中的编码器-解码器注意力且未缓存
                cache["static_k"], cache["static_v"] = k, v
            elif not static_kv:
                # 用于推理中的解码器自注意力
                cache_k, cache_v = cache["k"], cache["v"]
                # 拼接缓存的键和当前的键
                k = paddle.concat([cache_k, k], axis=2)
                # 拼接缓存的值和当前的值
                v = paddle.concat([cache_v, v], axis=2)
                cache["k"], cache["v"] = k, v

        # 返回查询、键、值
        return q, k, v
    # 前向传播函数，接收查询、键、值、注意力偏置和缓存作为输入
    def forward(self, queries, keys, values, attn_bias, cache=None):
        # 如果没有传入键，则将查询作为键
        keys = queries if keys is None else keys
        # 如果没有传入值，则将键作为值
        values = keys if values is None else values
        # 准备查询、键、值
        q, k, v = self._prepare_qkv(queries, keys, values, cache)

        # 缩放点积注意力
        product = paddle.matmul(x=q, y=k, transpose_y=True)
        product = product * self.d_model**-0.5
        # 如果存在注意力偏置，则加上偏置
        if attn_bias is not None:
            product += attn_bias
        # 计算注意力权重
        weights = F.softmax(product)
        # 如果存在 dropout_rate，则对权重进行 dropout
        if self.dropout_rate:
            weights = F.dropout(
                weights, p=self.dropout_rate, mode="downscale_in_infer")
        # 计算输出
        out = paddle.matmul(weights, v)

        # 合并多头注意力的结果
        out = paddle.transpose(out, perm=[0, 2, 1, 3])
        out = paddle.reshape(x=out, shape=[0, 0, out.shape[2] * out.shape[3]])

        # 投影到输出维度
        out = self.proj_fc(out)

        # 返回输出结果
        return out
class PrePostProcessLayer(nn.Layer):
    """
    PrePostProcessLayer
    """

    def __init__(self, process_cmd, d_model, dropout_rate):
        # 初始化 PrePostProcessLayer 类
        super(PrePostProcessLayer, self).__init__()
        # 存储处理命令、模型维度和 dropout 率
        self.process_cmd = process_cmd
        self.functors = []
        # 遍历处理命令列表
        for cmd in self.process_cmd:
            # 如果命令是 "a"，添加残差连接的函数到 functors 列表中
            if cmd == "a":  
                self.functors.append(lambda x, y: x + y if y is not None else x)
            # 如果命令是 "n"，添加层归一化的函数到 functors 列表中
            elif cmd == "n":  
                self.functors.append(
                    self.add_sublayer(
                        "layer_norm_%d" % len(self.sublayers()),
                        paddle.nn.LayerNorm(
                            normalized_shape=d_model,
                            weight_attr=paddle.ParamAttr(
                                initializer=paddle.nn.initializer.Constant(1.)),
                            bias_attr=paddle.ParamAttr(
                                initializer=paddle.nn.initializer.Constant(0.)))))
            # 如果命令是 "d"，添加 dropout 的函数到 functors 列表中
            elif cmd == "d":  
                self.functors.append(lambda x: F.dropout(
                    x, p=dropout_rate, mode="downscale_in_infer")
                                     if dropout_rate else x)

    def forward(self, x, residual=None):
        # 遍历处理命令列表
        for i, cmd in enumerate(self.process_cmd):
            # 如果命令是 "a"，使用残差连接函数处理输入
            if cmd == "a":
                x = self.functors[i](x, residual)
            # 否则，使用对应的函数处理输入
            else:
                x = self.functors[i](x)
        # 返回处理后的结果
        return x


class PrepareEncoder(nn.Layer):
    # 初始化函数，设置编码器的参数
    def __init__(self,
                 src_vocab_size,
                 src_emb_dim,
                 src_max_len,
                 dropout_rate=0,
                 bos_idx=0,
                 word_emb_param_name=None,
                 pos_enc_param_name=None):
        # 调用父类的初始化函数
        super(PrepareEncoder, self).__init__()
        # 设置源语言词汇表大小
        self.src_emb_dim = src_emb_dim
        # 设置源语言最大长度
        self.src_max_len = src_max_len
        # 创建一个嵌入层，用于将输入的词索引转换为词向量
        self.emb = paddle.nn.Embedding(
            num_embeddings=self.src_max_len, embedding_dim=self.src_emb_dim)
        # 设置 dropout 的比率
        self.dropout_rate = dropout_rate

    # 前向传播函数，实现编码器的计算过程
    def forward(self, src_word, src_pos):
        # 将源语言词索引转换为词向量
        src_word_emb = src_word
        # 将词向量转换为 float32 类型
        src_word_emb = paddle.cast(src_word_emb, 'float32')
        # 对词向量进行缩放，以减小训练时的梯度爆炸问题
        src_word_emb = paddle.scale(x=src_word_emb, scale=self.src_emb_dim**0.5)
        # 去除位置编码的多余维度
        src_pos = paddle.squeeze(src_pos, axis=-1)
        # 使用嵌入层将位置索引转换为位置编码
        src_pos_enc = self.emb(src_pos)
        # 设置位置编码的梯度不回传
        src_pos_enc.stop_gradient = True
        # 将词向量和位置编码相加作为编码器的输入
        enc_input = src_word_emb + src_pos_enc
        # 如果设置了 dropout 比率，则对输入进行 dropout 处理
        if self.dropout_rate:
            out = F.dropout(
                x=enc_input, p=self.dropout_rate, mode="downscale_in_infer")
        else:
            out = enc_input
        # 返回编码器的输出
        return out
class PrepareDecoder(nn.Layer):
    def __init__(self,
                 src_vocab_size,
                 src_emb_dim,
                 src_max_len,
                 dropout_rate=0,
                 bos_idx=0,
                 word_emb_param_name=None,
                 pos_enc_param_name=None):
        super(PrepareDecoder, self).__init__()
        self.src_emb_dim = src_emb_dim
        """
        # 创建一个 Embedding 层，用于将输入的词索引映射为词向量
        self.emb0 = paddle.nn.Embedding(
            num_embeddings=src_vocab_size,
            embedding_dim=self.src_emb_dim,
            padding_idx=bos_idx,
            weight_attr=paddle.ParamAttr(
                name=word_emb_param_name,
                initializer=nn.initializer.Normal(0., src_emb_dim**-0.5)))
        # 创建另一个 Embedding 层，用于将输入的位置索引映射为位置编码
        self.emb1 = paddle.nn.Embedding(
            num_embeddings=src_max_len,
            embedding_dim=self.src_emb_dim,
            weight_attr=paddle.ParamAttr(name=pos_enc_param_name))
        self.dropout_rate = dropout_rate

    def forward(self, src_word, src_pos):
        # 将输入的词索引转换为 int64 类型
        src_word = paddle.cast(src_word, 'int64')
        src_word = paddle.squeeze(src_word, axis=-1)
        # 通过词嵌入层将词索引转换为词向量
        src_word_emb = self.emb0(src_word)
        # 对词向量进行缩放
        src_word_emb = paddle.scale(x=src_word_emb, scale=self.src_emb_dim**0.5)
        # 将输入的位置索引转换为 int64 类型
        src_pos = paddle.squeeze(src_pos, axis=-1)
        # 通过位置编码层将位置索引转换为位置编码
        src_pos_enc = self.emb1(src_pos)
        # 停止位置编码的梯度传播
        src_pos_enc.stop_gradient = True
        # 将词向量和位置编码相加作为编码器的输入
        enc_input = src_word_emb + src_pos_enc
        # 如果有设置 dropout_rate，则对输入进行 dropout 处理
        if self.dropout_rate:
            out = F.dropout(
                x=enc_input, p=self.dropout_rate, mode="downscale_in_infer")
        else:
            out = enc_input
        return out


class FFN(nn.Layer):
    """
    Feed-Forward Network
    """
    # 初始化 FFN 类，设置隐藏层维度、输入维度和 dropout 率
    def __init__(self, d_inner_hid, d_model, dropout_rate):
        # 调用父类的初始化方法
        super(FFN, self).__init__()
        # 保存 dropout 率
        self.dropout_rate = dropout_rate
        # 创建全连接层 fc1，输入维度为 d_model，输出维度为 d_inner_hid
        self.fc1 = paddle.nn.Linear(
            in_features=d_model, out_features=d_inner_hid)
        # 创建全连接层 fc2，输入维度为 d_inner_hid，输出维度为 d_model

        self.fc2 = paddle.nn.Linear(
            in_features=d_inner_hid, out_features=d_model)

    # 前向传播函数
    def forward(self, x):
        # 使用全连接层 fc1 进行前向传播
        hidden = self.fc1(x)
        # 使用 ReLU 激活函数
        hidden = F.relu(hidden)
        # 如果有 dropout_rate，则进行 dropout 操作
        if self.dropout_rate:
            hidden = F.dropout(
                hidden, p=self.dropout_rate, mode="downscale_in_infer")
        # 使用全连接层 fc2 进行前向传播
        out = self.fc2(hidden)
        # 返回输出结果
        return out
```