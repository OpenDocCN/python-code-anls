# `.\PaddleOCR\ppocr\modeling\heads\rec_visionlan_head.py`

```
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“原样”分发的，
# 没有任何明示或暗示的保证或条件
# 请查看许可证以获取有关权限和限制的详细信息
"""
# 引用来源
# https://github.com/wangyuxin87/VisionLAN

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 导入 paddle 库
import paddle
# 从 paddle 中导入 ParamAttr
from paddle import ParamAttr
# 从 paddle.nn 中导入 nn
import paddle.nn as nn
# 从 paddle.nn 中导入 functional 模块
import paddle.nn.functional as F
# 从 paddle.nn.initializer 中导入 Normal 和 XavierNormal
from paddle.nn.initializer import Normal, XavierNormal
# 导入 numpy 库
import numpy as np

# 定义一个名为 PositionalEncoding 的类，继承自 nn.Layer
class PositionalEncoding(nn.Layer):
    # 初始化函数，接受隐藏层维度 d_hid 和位置编码的长度 n_position
    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()
        # 注册一个缓冲区，存储正弦位置编码表
        self.register_buffer(
            'pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    # 定义一个获取正弦位置编码表的函数
    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''

        # 定义一个函数，用于计算位置角度向量
        def get_position_angle_vec(position):
            return [
                position / np.power(10000, 2 * (hid_j // 2) / d_hid)
                for hid_j in range(d_hid)
            ]

        # 生成正弦位置编码表
        sinusoid_table = np.array(
            [get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        # 计算正弦和余弦值
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        # 将 numpy 数组转换为 paddle 张量
        sinusoid_table = paddle.to_tensor(sinusoid_table, dtype='float32')
        # 在第 0 维度增加一个维度
        sinusoid_table = paddle.unsqueeze(sinusoid_table, axis=0)
        return sinusoid_table
    # 定义一个前向传播函数，接受输入 x
    def forward(self, x):
        # 返回 x 与 self.pos_table 的部分矩阵相加，同时进行克隆并分离计算图
        return x + self.pos_table[:, :x.shape[1]].clone().detach()
class ScaledDotProductAttention(nn.Layer):
    "Scaled Dot-Product Attention"

    def __init__(self, temperature, attn_dropout=0.1):
        # 初始化 Scaled Dot-Product Attention 类
        super(ScaledDotProductAttention, self).__init__()
        # 设置温度参数
        self.temperature = temperature
        # 设置注意力机制的 dropout 概率
        self.dropout = nn.Dropout(attn_dropout)
        # 创建 Softmax 层，用于计算注意力权重
        self.softmax = nn.Softmax(axis=2)

    def forward(self, q, k, v, mask=None):
        # 将 k 矩阵进行转置
        k = paddle.transpose(k, perm=[0, 2, 1])
        # 计算注意力权重
        attn = paddle.bmm(q, k)
        # 对注意力权重进行缩放
        attn = attn / self.temperature
        # 如果存在 mask，则进行 mask 操作
        if mask is not None:
            # 将注意力权重中 mask 的位置替换为一个很小的负数
            attn = attn.masked_fill(mask, -1e9)
            # 根据 mask 的维度进行处理
            if mask.dim() == 3:
                mask = paddle.unsqueeze(mask, axis=1)
            elif mask.dim() == 2:
                mask = paddle.unsqueeze(mask, axis=1)
                mask = paddle.unsqueeze(mask, axis=1)
            # 将 mask 扩展到与注意力权重相同的维度
            repeat_times = [
                attn.shape[1] // mask.shape[1], attn.shape[2] // mask.shape[2]
            ]
            mask = paddle.tile(mask, [1, repeat_times[0], repeat_times[1], 1])
            # 将 mask 为 0 的位置替换为一个很小的负数
            attn[mask == 0] = -1e9
        # 对注意力权重进行 Softmax 操作
        attn = self.softmax(attn)
        # 对注意力权重进行 dropout 操作
        attn = self.dropout(attn)
        # 计算最终输出
        output = paddle.bmm(attn, v)
        return output


class MultiHeadAttention(nn.Layer):
    " Multi-Head Attention module"
    # 初始化多头注意力机制模块
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        # 调用父类的初始化方法
        super(MultiHeadAttention, self).__init__()
        # 设置多头注意力机制的头数
        self.n_head = n_head
        # 设置查询向量的维度
        self.d_k = d_k
        # 设置数值向量的维度
        self.d_v = d_v
        # 创建线性变换层，用于将输入的维度转换为 n_head * d_k 的维度
        self.w_qs = nn.Linear(
            d_model,
            n_head * d_k,
            weight_attr=ParamAttr(initializer=Normal(
                mean=0, std=np.sqrt(2.0 / (d_model + d_k))))
        )
        # 创建线性变换层，用于将输入的维度转换为 n_head * d_k 的维度
        self.w_ks = nn.Linear(
            d_model,
            n_head * d_k,
            weight_attr=ParamAttr(initializer=Normal(
                mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        )
        # 创建线性变换层，用于将输入的维度转换为 n_head * d_v 的维度
        self.w_vs = nn.Linear(
            d_model,
            n_head * d_v,
            weight_attr=ParamAttr(initializer=Normal(
                mean=0, std=np.sqrt(2.0 / (d_model + d_v)))
        )

        # 初始化缩放点积注意力机制
        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        # 初始化层归一化模块
        self.layer_norm = nn.LayerNorm(d_model)
        # 创建线性变换层，用于将 n_head * d_v 的维度转换为 d_model 的维度
        self.fc = nn.Linear(
            n_head * d_v,
            d_model,
            weight_attr=ParamAttr(initializer=XavierNormal())
        )
        # 初始化 dropout 模块
        self.dropout = nn.Dropout(dropout)
    # 定义一个前向传播函数，接受查询(q)、键(k)、值(v)和掩码(mask)作为输入
    def forward(self, q, k, v, mask=None):
        # 获取键值的维度和注意力头数
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        # 获取查询的形状
        sz_b, len_q, _ = q.shape
        # 获取键的形状
        sz_b, len_k, _ = k.shape
        # 获取值的形状
        sz_b, len_v, _ = v.shape
        # 保存残差连接
        residual = q

        # 对查询进行线性变换
        q = self.w_qs(q)
        # 重塑查询的形状为(n*b) x lq x n_head x d_k
        q = paddle.reshape(
            q, shape=[-1, len_q, n_head, d_k])  # 4*21*512 ---- 4*21*8*64
        # 对键进行线性变换
        k = self.w_ks(k)
        # 重塑键的形状为(n*b) x lk x n_head x d_k
        k = paddle.reshape(k, shape=[-1, len_k, n_head, d_k])
        # 对值进行线性变换
        v = self.w_vs(v)
        # 重塑值的形状为(n*b) x lv x n_head x d_v
        v = paddle.reshape(v, shape=[-1, len_v, n_head, d_v])

        # 调换查询的维度顺序
        q = paddle.transpose(q, perm=[2, 0, 1, 3])
        # 重塑查询的形状为(n*b) x len_q x d_k
        q = paddle.reshape(q, shape=[-1, len_q, d_k])  # (n*b) x lq x dk
        # 调换键的维度顺序
        k = paddle.transpose(k, perm=[2, 0, 1, 3])
        # 重塑键的形状为(n*b) x len_k x d_k
        k = paddle.reshape(k, shape=[-1, len_k, d_k])  # (n*b) x lk x dk
        # 调换值的维度顺序
        v = paddle.transpose(v, perm=[2, 0, 1, 3])
        # 重塑值的形状为(n*b) x len_v x d_v
        v = paddle.reshape(v, shape=[-1, len_v, d_v])  # (n*b) x lv x dv

        # 复制掩码以匹配注意力头数
        mask = paddle.tile(
            mask,
            [n_head, 1, 1]) if mask is not None else None  # (n*b) x .. x ..
        # 使用注意力函数计算输出
        output = self.attention(q, k, v, mask=mask)
        # 重塑输出的形状为n_head x (n*b) x len_q x d_v
        output = paddle.reshape(output, shape=[n_head, -1, len_q, d_v])
        # 调换输出的维度顺序
        output = paddle.transpose(output, perm=[1, 2, 0, 3])
        # 重塑输出的形状为(n*b) x len_q x (n*d_v)
        output = paddle.reshape(
            output, shape=[-1, len_q, n_head * d_v])  # b x lq x (n*dv)
        # 使用全连接层和Dropout进行线性变换和正则化
        output = self.dropout(self.fc(output))
        # 添加残差连接并进行层归一化
        output = self.layer_norm(output + residual)
        # 返回输出
        return output
class PositionwiseFeedForward(nn.Layer):
    # 定义位置编码前馈神经网络类
    def __init__(self, d_in, d_hid, dropout=0.1):
        # 初始化函数，接受输入维度、隐藏层维度和dropout参数
        super(PositionwiseFeedForward, self).__init__()
        # 调用父类构造函数
        self.w_1 = nn.Conv1D(d_in, d_hid, 1)  # position-wise
        # 定义第一个卷积层，用于位置编码
        self.w_2 = nn.Conv1D(d_hid, d_in, 1)  # position-wise
        # 定义第二个卷积层，用于位置编码
        self.layer_norm = nn.LayerNorm(d_in)
        # 定义LayerNorm层，用于归一化
        self.dropout = nn.Dropout(dropout)
        # 定义Dropout层，用于防止过拟合

    def forward(self, x):
        # 前向传播函数
        residual = x
        # 保存输入的残差连接
        x = paddle.transpose(x, perm=[0, 2, 1])
        # 调整输入的维度顺序
        x = self.w_2(F.relu(self.w_1(x)))
        # 经过两个卷积层和ReLU激活函数
        x = paddle.transpose(x, perm=[0, 2, 1])
        # 调整输出的维度顺序
        x = self.dropout(x)
        # 对输出进行Dropout操作
        x = self.layer_norm(x + residual)
        # 添加残差连接并进行LayerNorm
        return x


class EncoderLayer(nn.Layer):
    ''' Compose with two layers '''
    # 编码器层，由两个子层组成

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        # 初始化函数，接受模型维度、内部维度、注意力头数、注意力维度和dropout参数
        super(EncoderLayer, self).__init__()
        # 调用父类构造函数
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        # 定义自注意力层
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout)
        # 定义位置编码前馈神经网络层

    def forward(self, enc_input, slf_attn_mask=None):
        # 前向传播函数，接受编码器输入和自注意力掩码
        enc_output = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        # 经过自注意力层
        enc_output = self.pos_ffn(enc_output)
        # 经过位置编码前馈神经网络层
        return enc_output


class Transformer_Encoder(nn.Layer):
    # 定义Transformer编码器类
    # 初始化 Transformer 编码器类，设置各种参数
    def __init__(self,
                 n_layers=2,
                 n_head=8,
                 d_word_vec=512,
                 d_k=64,
                 d_v=64,
                 d_model=512,
                 d_inner=2048,
                 dropout=0.1,
                 n_position=256):
        # 调用父类的初始化方法
        super(Transformer_Encoder, self).__init__()
        # 创建位置编码对象
        self.position_enc = PositionalEncoding(
            d_word_vec, n_position=n_position)
        # 创建 Dropout 层
        self.dropout = nn.Dropout(p=dropout)
        # 创建编码器层列表
        self.layer_stack = nn.LayerList([
            EncoderLayer(
                d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)
        ])
        # 创建 LayerNorm 层
        self.layer_norm = nn.LayerNorm(d_model, epsilon=1e-6)
    
    # 定义前向传播方法
    def forward(self, enc_output, src_mask, return_attns=False):
        # 对编码器输出进行 Dropout 和位置编码
        enc_output = self.dropout(
            self.position_enc(enc_output))  # position embeding
        # 遍历编码器层列表，对每一层进行处理
        for enc_layer in self.layer_stack:
            enc_output = enc_layer(enc_output, slf_attn_mask=src_mask)
        # 对编码器输出进行 LayerNorm 处理
        enc_output = self.layer_norm(enc_output)
        # 返回编码器输出
        return enc_output
class PP_layer(nn.Layer):
    # 定义一个PP_layer类，继承自nn.Layer
    def __init__(self, n_dim=512, N_max_character=25, n_position=256):
        # 初始化函数，接受三个参数：n_dim, N_max_character, n_position
        super(PP_layer, self).__init__()
        # 调用父类的初始化函数
        self.character_len = N_max_character
        # 设置类属性character_len为N_max_character
        self.f0_embedding = nn.Embedding(N_max_character, n_dim)
        # 创建一个Embedding层，参数为N_max_character, n_dim
        self.w0 = nn.Linear(N_max_character, n_position)
        # 创建一个Linear层，参数为N_max_character, n_position
        self.wv = nn.Linear(n_dim, n_dim)
        # 创建一个Linear层，参数为n_dim, n_dim
        self.we = nn.Linear(n_dim, N_max_character)
        # 创建一个Linear层，参数为n_dim, N_max_character
        self.active = nn.Tanh()
        # 创建一个Tanh激活函数
        self.softmax = nn.Softmax(axis=2)
        # 创建一个Softmax层，指定axis为2

    def forward(self, enc_output):
        # 定义前向传播函数，接受enc_output作为输入
        # enc_output: b,256,512
        reading_order = paddle.arange(self.character_len, dtype='int64')
        # 创建一个长度为character_len的整数序列
        reading_order = reading_order.unsqueeze(0).expand(
            [enc_output.shape[0], self.character_len])  # (S,) -> (B, S)
        # 将序列扩展为形状为[enc_output.shape[0], self.character_len]的张量
        reading_order = self.f0_embedding(reading_order)  # b,25,512
        # 使用Embedding层将序列转换为张量

        # calculate attention
        reading_order = paddle.transpose(reading_order, perm=[0, 2, 1])
        # 转置张量的维度
        t = self.w0(reading_order)  # b,512,256
        # 使用Linear层进行线性变换
        t = self.active(
            paddle.transpose(
                t, perm=[0, 2, 1]) + self.wv(enc_output))  # b,256,512
        # 进行激活函数处理和加法操作
        t = self.we(t)  # b,256,25
        # 使用Linear层进行线性变换
        t = self.softmax(paddle.transpose(t, perm=[0, 2, 1]))  # b,25,256
        # 使用Softmax层进行处理
        g_output = paddle.bmm(t, enc_output)  # b,25,512
        # 进行矩阵乘法操作
        return g_output
        # 返回结果张量


class Prediction(nn.Layer):
    # 定义一个Prediction类，继承自nn.Layer
    def __init__(self,
                 n_dim=512,
                 n_position=256,
                 N_max_character=25,
                 n_class=37):
        # 初始化函数，接受四个参数：n_dim, n_position, N_max_character, n_class
        super(Prediction, self).__init__()
        # 调用父类的初始化函数
        self.pp = PP_layer(
            n_dim=n_dim, N_max_character=N_max_character, n_position=n_position)
        # 创建一个PP_layer对象，传入指定参数
        self.pp_share = PP_layer(
            n_dim=n_dim, N_max_character=N_max_character, n_position=n_position)
        # 创建另一个PP_layer对象，传入指定参数
        self.w_vrm = nn.Linear(n_dim, n_class)  # output layer
        # 创建一个Linear层，参数为n_dim, n_class，用于输出
        self.w_share = nn.Linear(n_dim, n_class)  # output layer
        # 创建另一个Linear层，参数为n_dim, n_class，用于输出
        self.nclass = n_class
        # 设置类属性nclass为n_class
    # 定义一个前向传播函数，接受卷积神经网络特征、f_res、f_sub作为输入，train_mode和use_mlm作为可选参数
    def forward(self, cnn_feature, f_res, f_sub, train_mode=False,
                use_mlm=True):
        # 如果处于训练模式
        if train_mode:
            # 如果不使用MLM
            if not use_mlm:
                # 对cnn_feature进行预处理
                g_output = self.pp(cnn_feature)  # b,25,512
                # 对g_output进行处理
                g_output = self.w_vrm(g_output)
                # 重置f_res和f_sub
                f_res = 0
                f_sub = 0
                # 返回处理后的结果
                return g_output, f_res, f_sub
            # 对cnn_feature进行预处理
            g_output = self.pp(cnn_feature)  # b,25,512
            # 对f_res和f_sub进行共享处理
            f_res = self.pp_share(f_res)
            f_sub = self.pp_share(f_sub)
            # 对g_output进行处理
            g_output = self.w_vrm(g_output)
            # 对f_res和f_sub进行共享处理
            f_res = self.w_share(f_res)
            f_sub = self.w_share(f_sub)
            # 返回处理后的结果
            return g_output, f_res, f_sub
        # 如果不处于训练模式
        else:
            # 对cnn_feature进行预处理
            g_output = self.pp(cnn_feature)  # b,25,512
            # 对g_output进行处理
            g_output = self.w_vrm(g_output)
            # 返回处理后的结果
            return g_output
class MLM(nn.Layer):
    "Architecture of MLM"

    def __init__(self, n_dim=512, n_position=256, max_text_length=25):
        # 初始化MLM类，设置默认参数n_dim=512, n_position=256, max_text_length=25
        super(MLM, self).__init__()
        # 调用父类的初始化方法
        self.MLM_SequenceModeling_mask = Transformer_Encoder(
            n_layers=2, n_position=n_position)
        # 创建MLM的序列建模遮罩，使用Transformer_Encoder，设置层数为2，位置数为n_position
        self.MLM_SequenceModeling_WCL = Transformer_Encoder(
            n_layers=1, n_position=n_position)
        # 创建MLM的序列建模WCL，使用Transformer_Encoder，设置层数为1，位置数为n_position
        self.pos_embedding = nn.Embedding(max_text_length, n_dim)
        # 创建位置嵌入层，使用Embedding，设置最大文本长度为max_text_length，维度为n_dim
        self.w0_linear = nn.Linear(1, n_position)
        # 创建线性层w0_linear，输入维度为1，输出维度为n_position
        self.wv = nn.Linear(n_dim, n_dim)
        # 创建线性层wv，输入维度为n_dim，输出维度为n_dim
        self.active = nn.Tanh()
        # 创建Tanh激活函数
        self.we = nn.Linear(n_dim, 1)
        # 创建线性层we，输入维度为n_dim，输出维度为1
        self.sigmoid = nn.Sigmoid()
        # 创建Sigmoid激活函数

    def forward(self, x, label_pos):
        # 前向传播函数
        # transformer unit for generating mask_c
        feature_v_seq = self.MLM_SequenceModeling_mask(x, src_mask=None)
        # 使用MLM的序列建模遮罩生成特征v序列
        # position embedding layer
        label_pos = paddle.to_tensor(label_pos, dtype='int64')
        # 将label_pos转换为张量，数据类型为int64
        pos_emb = self.pos_embedding(label_pos)
        # 获取位置嵌入
        pos_emb = self.w0_linear(paddle.unsqueeze(pos_emb, axis=2))
        # 使用w0_linear对位置嵌入进行处理
        pos_emb = paddle.transpose(pos_emb, perm=[0, 2, 1])
        # 转置位置嵌入
        # fusion position embedding with features V & generate mask_c
        att_map_sub = self.active(pos_emb + self.wv(feature_v_seq))
        # 将位置嵌入与特征V融合并生成mask_c
        att_map_sub = self.we(att_map_sub)  # b,256,1
        att_map_sub = paddle.transpose(att_map_sub, perm=[0, 2, 1])
        # 转置att_map_sub
        att_map_sub = self.sigmoid(att_map_sub)  # b,1,256
        # 使用Sigmoid激活函数处理att_map_sub
        # WCL
        ## generate inputs for WCL
        att_map_sub = paddle.transpose(att_map_sub, perm=[0, 2, 1])
        # 转置att_map_sub
        f_res = x * (1 - att_map_sub)  # second path with remaining string
        # 计算第二路径的结果
        f_sub = x * att_map_sub  # first path with occluded character
        # 计算第一路径的结果
        ## transformer units in WCL
        f_res = self.MLM_SequenceModeling_WCL(f_res, src_mask=None)
        # 使用MLM的序列建模WCL处理第二路径结果
        f_sub = self.MLM_SequenceModeling_WCL(f_sub, src_mask=None)
        # 使用MLM的序列建模WCL处理第一路径结果
        return f_res, f_sub, att_map_sub
        # 返回结果

def trans_1d_2d(x):
    b, w_h, c = x.shape  # b, 256, 512
    # 获取x的形状信息
    x = paddle.transpose(x, perm=[0, 2, 1])
    # 转置x
    # 将输入张量 x 重塑为指定形状 [-1, c, 32, 8]
    x = paddle.reshape(x, [-1, c, 32, 8])
    # 对输入张量 x 进行转置操作，交换指定维度的顺序，得到形状为 [b, c, 8, 32] 的张量
    x = paddle.transpose(x, perm=[0, 1, 3, 2])  
    # 返回转置后的张量 x
    return x
class MLM_VRM(nn.Layer):
    """
    MLM+VRM, MLM is only used in training.
    ratio controls the occluded number in a batch.
    The pipeline of VisionLAN in testing is very concise with only a backbone + sequence modeling(transformer unit) + prediction layer(pp layer).
    x: input image
    label_pos: character index
    training_step: LF or LA process
    output
    text_pre: prediction of VRM
    test_rem: prediction of remaining string in MLM
    text_mas: prediction of occluded character in MLM
    mask_c_show: visualization of Mask_c
    """

    def __init__(self,
                 n_layers=3,
                 n_position=256,
                 n_dim=512,
                 max_text_length=25,
                 nclass=37):
        super(MLM_VRM, self).__init__()
        # 初始化 MLM 模型
        self.MLM = MLM(n_dim=n_dim,
                       n_position=n_position,
                       max_text_length=max_text_length)
        # 初始化序列建模模块
        self.SequenceModeling = Transformer_Encoder(
            n_layers=n_layers, n_position=n_position)
        # 初始化预测层
        self.Prediction = Prediction(
            n_dim=n_dim,
            n_position=n_position,
            N_max_character=max_text_length +
            1,  # N_max_character = 1 eos + 25 characters
            n_class=nclass)
        self.nclass = nclass
        self.max_text_length = max_text_length

class VLHead(nn.Layer):
    """
    Architecture of VisionLAN
    """

    def __init__(self,
                 in_channels,
                 out_channels=36,
                 n_layers=3,
                 n_position=256,
                 n_dim=512,
                 max_text_length=25,
                 training_step='LA'):
        super(VLHead, self).__init__()
        # 初始化 MLM_VRM 模型
        self.MLM_VRM = MLM_VRM(
            n_layers=n_layers,
            n_position=n_position,
            n_dim=n_dim,
            max_text_length=max_text_length,
            nclass=out_channels + 1)
        self.training_step = training_step
    # 定义一个前向传播函数，接受特征和目标作为输入
    def forward(self, feat, targets=None):
        # 如果处于训练模式
        if self.training:
            # 获取目标中倒数第二个元素作为标签位置
            label_pos = targets[-2]
            # 调用MLM_VRM方法，传入特征、标签位置、训练步骤和训练模式，返回预测文本、测试剩余、文本掩码和掩码映射
            text_pre, test_rem, text_mas, mask_map = self.MLM_VRM(
                feat, label_pos, self.training_step, train_mode=True)
            # 返回预测文本、测试剩余、文本掩码和掩码映射
            return text_pre, test_rem, text_mas, mask_map
        # 如果不处于训练模式
        else:
            # 调用MLM_VRM方法，传入特征、目标、训练步骤和训练模式为False，返回预测文本和x
            text_pre, x = self.MLM_VRM(
                feat, targets, self.training_step, train_mode=False)
            # 返回预测文本和x
            return text_pre, x
```