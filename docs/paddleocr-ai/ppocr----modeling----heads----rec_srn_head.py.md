# `.\PaddleOCR\ppocr\modeling\heads\rec_srn_head.py`

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
# 没有任何明示或暗示的保证或条件
# 请查看许可证以获取特定语言的权限和限制

# 导入必要的库
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import paddle
from paddle import nn, ParamAttr
from paddle.nn import functional as F
import numpy as np
# 导入自定义的 self_attention 模块中的 WrapEncoderForFeature 和 WrapEncoder 类
from .self_attention import WrapEncoderForFeature
from .self_attention import WrapEncoder
from paddle.static import Program
from ppocr.modeling.backbones.rec_resnet_fpn import ResNetFPN

# 导入 OrderedDict 类
from collections import OrderedDict
# 设置梯度裁剪的阈值为 10
gradient_clip = 10

# 定义 PVAM 类，继承自 nn.Layer
class PVAM(nn.Layer):
    # 初始化 PVAM 模型，设置输入通道数、字符数量、最大文本长度、注意力头数、编码器层数和隐藏层维度
    def __init__(self, in_channels, char_num, max_text_length, num_heads,
                 num_encoder_tus, hidden_dims):
        # 调用父类的初始化方法
        super(PVAM, self).__init__()
        # 设置字符数量
        self.char_num = char_num
        # 设置最大文本长度
        self.max_length = max_text_length
        # 设置注意力头数
        self.num_heads = num_heads
        # 设置编码器 Transformer 单元数
        self.num_encoder_TUs = num_encoder_tus
        # 设置隐藏层维度
        self.hidden_dims = hidden_dims
        # 初始化 Transformer 编码器
        t = 256
        c = 512
        self.wrap_encoder_for_feature = WrapEncoderForFeature(
            src_vocab_size=1,
            max_length=t,
            n_layer=self.num_encoder_TUs,
            n_head=self.num_heads,
            d_key=int(self.hidden_dims / self.num_heads),
            d_value=int(self.hidden_dims / self.num_heads),
            d_model=self.hidden_dims,
            d_inner_hid=self.hidden_dims,
            prepostprocess_dropout=0.1,
            attention_dropout=0.1,
            relu_dropout=0.1,
            preprocess_cmd="n",
            postprocess_cmd="da",
            weight_sharing=True)

        # 初始化 PVAM 模型的各个层
        self.flatten0 = paddle.nn.Flatten(start_axis=0, stop_axis=1)
        self.fc0 = paddle.nn.Linear(
            in_features=in_channels,
            out_features=in_channels, )
        self.emb = paddle.nn.Embedding(
            num_embeddings=self.max_length, embedding_dim=in_channels)
        self.flatten1 = paddle.nn.Flatten(start_axis=0, stop_axis=2)
        self.fc1 = paddle.nn.Linear(
            in_features=in_channels, out_features=1, bias_attr=False)
    # 定义前向传播函数，接受输入数据、编码器词位置和GSRM词位置
    def forward(self, inputs, encoder_word_pos, gsrm_word_pos):
        # 获取输入数据的形状信息
        b, c, h, w = inputs.shape
        # 重塑输入数据的形状
        conv_features = paddle.reshape(inputs, shape=[-1, c, h * w])
        # 调换维度顺序
        conv_features = paddle.transpose(conv_features, perm=[0, 2, 1])
        
        # transformer encoder
        # 获取转换后的特征数据的形状信息
        b, t, c = conv_features.shape
        # 将特征数据传入编码器进行处理
        enc_inputs = [conv_features, encoder_word_pos, None]
        word_features = self.wrap_encoder_for_feature(enc_inputs)
        
        # pvam
        # 获取处理后的特征数据的形状信息
        b, t, c = word_features.shape
        # 将特征数据通过全连接层fc0进行处理
        word_features = self.fc0(word_features)
        # 重塑特征数据的形状
        word_features_ = paddle.reshape(word_features, [-1, 1, t, c])
        word_features_ = paddle.tile(word_features_, [1, self.max_length, 1, 1])
        # 获取GSRM词位置的嵌入特征
        word_pos_feature = self.emb(gsrm_word_pos)
        word_pos_feature_ = paddle.reshape(word_pos_feature, [-1, self.max_length, 1, c])
        word_pos_feature_ = paddle.tile(word_pos_feature_, [1, 1, t, 1])
        # 将词位置特征和词特征相加，并通过tanh激活函数
        y = word_pos_feature_ + word_features_
        y = F.tanh(y)
        # 通过全连接层fc1获取注意力权重
        attention_weight = self.fc1(y)
        attention_weight = paddle.reshape(attention_weight, shape=[-1, self.max_length, t])
        # 对注意力权重进行softmax处理
        attention_weight = F.softmax(attention_weight, axis=-1)
        # 使用注意力权重对词特征进行加权求和
        pvam_features = paddle.matmul(attention_weight, word_features)  #[b, max_length, c]
        # 返回pvam特征
        return pvam_features
# 定义 GSRM 类，继承自 nn.Layer
class GSRM(nn.Layer):
    # 初始化函数，接受输入通道数、字符数量、最大文本长度、注意力头数、编码器层数、解码器层数和隐藏维度
    def __init__(self, in_channels, char_num, max_text_length, num_heads,
                 num_encoder_tus, num_decoder_tus, hidden_dims):
        # 调用父类的初始化函数
        super(GSRM, self).__init__()
        # 初始化类的属性
        self.char_num = char_num
        self.max_length = max_text_length
        self.num_heads = num_heads
        self.num_encoder_TUs = num_encoder_tus
        self.num_decoder_TUs = num_decoder_tus
        self.hidden_dims = hidden_dims

        # 创建全连接层，输入特征数为输入通道数，输出特征数为字符数量
        self.fc0 = paddle.nn.Linear(
            in_features=in_channels, out_features=self.char_num)
        
        # 创建 WrapEncoder 对象 wrap_encoder0
        self.wrap_encoder0 = WrapEncoder(
            src_vocab_size=self.char_num + 1,
            max_length=self.max_length,
            n_layer=self.num_decoder_TUs,
            n_head=self.num_heads,
            d_key=int(self.hidden_dims / self.num_heads),
            d_value=int(self.hidden_dims / self.num_heads),
            d_model=self.hidden_dims,
            d_inner_hid=self.hidden_dims,
            prepostprocess_dropout=0.1,
            attention_dropout=0.1,
            relu_dropout=0.1,
            preprocess_cmd="n",
            postprocess_cmd="da",
            weight_sharing=True)

        # 创建 WrapEncoder 对象 wrap_encoder1
        self.wrap_encoder1 = WrapEncoder(
            src_vocab_size=self.char_num + 1,
            max_length=self.max_length,
            n_layer=self.num_decoder_TUs,
            n_head=self.num_heads,
            d_key=int(self.hidden_dims / self.num_heads),
            d_value=int(self.hidden_dims / self.num_heads),
            d_model=self.hidden_dims,
            d_inner_hid=self.hidden_dims,
            prepostprocess_dropout=0.1,
            attention_dropout=0.1,
            relu_dropout=0.1,
            preprocess_cmd="n",
            postprocess_cmd="da",
            weight_sharing=True)

        # 定义矩阵乘法操作，用于计算矩阵相乘
        self.mul = lambda x: paddle.matmul(x=x,
                                           y=self.wrap_encoder0.prepare_decoder.emb0.weight,
                                           transpose_y=True)
    # 定义一个前向传播函数，接受输入数据、GSRM词位置、GSRM自注意力偏置1、GSRM自注意力偏置2作为参数
    def forward(self, inputs, gsrm_word_pos, gsrm_slf_attn_bias1,
                gsrm_slf_attn_bias2):
        # ===== GSRM 视觉到语义嵌入块 =====
        # 获取输入数据的形状信息
        b, t, c = inputs.shape
        # 将输入数据重塑为二维张量
        pvam_features = paddle.reshape(inputs, [-1, c])
        # 通过全连接层进行特征转换
        word_out = self.fc0(pvam_features)
        # 对输出进行 softmax 操作，并取最大值作为词的索引
        word_ids = paddle.argmax(F.softmax(word_out), axis=1)
        # 将词的索引重塑为三维张量
        word_ids = paddle.reshape(x=word_ids, shape=[-1, t, 1])

        #===== GSRM 语义推理块 =====
        """
        通过双向transformer实现的模块，
        ngram_feature1是前向的，ngram_feature2是后向的
        """
        # 定义填充索引
        pad_idx = self.char_num

        # 处理第一个词的张量
        word1 = paddle.cast(word_ids, "float32")
        word1 = F.pad(word1, [1, 0], value=1.0 * pad_idx, data_format="NLC")
        word1 = paddle.cast(word1, "int64")
        word1 = word1[:, :-1, :]
        # 处理第二个词的张量
        word2 = word_ids

        # 构建第一个词的编码器输入
        enc_inputs_1 = [word1, gsrm_word_pos, gsrm_slf_attn_bias1]
        # 构建第二个词的编码器输入
        enc_inputs_2 = [word2, gsrm_word_pos, gsrm_slf_attn_bias2]

        # 对第一个词进行编码
        gsrm_feature1 = self.wrap_encoder0(enc_inputs_1)
        # 对第二个词进行编码
        gsrm_feature2 = self.wrap_encoder1(enc_inputs_2)

        # 对第二个词进行填充
        gsrm_feature2 = F.pad(gsrm_feature2, [0, 1],
                              value=0.,
                              data_format="NLC")
        gsrm_feature2 = gsrm_feature2[:, 1:, ]
        # 合并两个词的特征
        gsrm_features = gsrm_feature1 + gsrm_feature2

        # 对合并后的特征进行乘法操作
        gsrm_out = self.mul(gsrm_features)

        # 获取输出的形状信息
        b, t, c = gsrm_out.shape
        # 将输出重塑为二维张量
        gsrm_out = paddle.reshape(gsrm_out, [-1, c])

        # 返回结果
        return gsrm_features, word_out, gsrm_out
# 定义一个名为VSFD的类，继承自nn.Layer类
class VSFD(nn.Layer):
    # 初始化方法，设置输入通道数、PVAM通道数和字符数量
    def __init__(self, in_channels=512, pvam_ch=512, char_num=38):
        super(VSFD, self).__init__()
        self.char_num = char_num
        # 创建一个全连接层，输入特征为in_channels*2，输出特征为pvam_ch
        self.fc0 = paddle.nn.Linear(
            in_features=in_channels * 2, out_features=pvam_ch)
        # 创建一个全连接层，输入特征为pvam_ch，输出特征为字符数量char_num
        self.fc1 = paddle.nn.Linear(
            in_features=pvam_ch, out_features=self.char_num)

    # 前向传播方法，接收PVAM特征和GSRM特征作为输入
    def forward(self, pvam_feature, gsrm_feature):
        # 获取PVAM特征的形状信息
        b, t, c1 = pvam_feature.shape
        # 获取GSRM特征的形状信息
        b, t, c2 = gsrm_feature.shape
        # 将PVAM特征和GSRM特征在第二个维度上拼接起来
        combine_feature_ = paddle.concat([pvam_feature, gsrm_feature], axis=2)
        # 将拼接后的特征重塑为二维张量
        img_comb_feature_ = paddle.reshape(
            combine_feature_, shape=[-1, c1 + c2])
        # 将重塑后的特征通过全连接层fc0得到映射结果
        img_comb_feature_map = self.fc0(img_comb_feature_)
        # 对映射结果进行Sigmoid激活函数处理
        img_comb_feature_map = F.sigmoid(img_comb_feature_map)
        # 将处理后的映射结果再次重塑为三维张量
        img_comb_feature_map = paddle.reshape(
            img_comb_feature_map, shape=[-1, t, c1])
        # 根据映射结果对PVAM特征和GSRM特征进行加权融合
        combine_feature = img_comb_feature_map * pvam_feature + (
            1.0 - img_comb_feature_map) * gsrm_feature
        # 将融合后的特征再次重塑为二维张量
        img_comb_feature = paddle.reshape(combine_feature, shape=[-1, c1])
        # 将融合后的特征通过全连接层fc1得到最终输出结果
        out = self.fc1(img_comb_feature)
        # 返回输出结果
        return out


class SRNHead(nn.Layer):
    # 初始化 SRNHead 类，设置输入通道数、输出通道数、最大文本长度、注意力头数、编码器和解码器的 Transformer 单元数、隐藏层维度等参数
    def __init__(self, in_channels, out_channels, max_text_length, num_heads,
                 num_encoder_TUs, num_decoder_TUs, hidden_dims, **kwargs):
        # 调用父类的初始化方法
        super(SRNHead, self).__init__()
        # 设置输出通道数
        self.char_num = out_channels
        # 设置最大文本长度
        self.max_length = max_text_length
        # 设置注意力头数
        self.num_heads = num_heads
        # 设置编码器 Transformer 单元数
        self.num_encoder_TUs = num_encoder_TUs
        # 设置解码器 Transformer 单元数
        self.num_decoder_TUs = num_decoder_TUs
        # 设置隐藏层维度
        self.hidden_dims = hidden_dims

        # 初始化 PVAM 模块
        self.pvam = PVAM(
            in_channels=in_channels,
            char_num=self.char_num,
            max_text_length=self.max_length,
            num_heads=self.num_heads,
            num_encoder_tus=self.num_encoder_TUs,
            hidden_dims=self.hidden_dims)

        # 初始化 GSRM 模块
        self.gsrm = GSRM(
            in_channels=in_channels,
            char_num=self.char_num,
            max_text_length=self.max_length,
            num_heads=self.num_heads,
            num_encoder_tus=self.num_encoder_TUs,
            num_decoder_tus=self.num_decoder_TUs,
            hidden_dims=self.hidden_dims)
        
        # 初始化 VSFD 模块
        self.vsfd = VSFD(in_channels=in_channels, char_num=self.char_num)

        # 将编码器 0 的准备解码器的嵌入层赋值给编码器 1 的准备解码器的嵌入层
        self.gsrm.wrap_encoder1.prepare_decoder.emb0 = self.gsrm.wrap_encoder0.prepare_decoder.emb0
    # 前向传播函数，接受输入和目标数据
    def forward(self, inputs, targets=None):
        # 从目标数据中获取最后四个元素
        others = targets[-4:]
        # 从最后四个元素中获取编码器词位置信息
        encoder_word_pos = others[0]
        # 从最后四个元素中获取 GSRM 词位置信息
        gsrm_word_pos = others[1]
        # 从最后四个元素中获取 GSRM 自注意力偏置1
        gsrm_slf_attn_bias1 = others[2]
        # 从最后四个元素中获取 GSRM 自注意力偏置2
        gsrm_slf_attn_bias2 = others[3]

        # 使用 PVAM 模块处理输入数据和编码器词位置信息
        pvam_feature = self.pvam(inputs, encoder_word_pos, gsrm_word_pos)

        # 使用 GSRM 模块处理 PVAM 特征、GSRM 词位置信息、GSRM 自注意力偏置1和2
        gsrm_feature, word_out, gsrm_out = self.gsrm(
            pvam_feature, gsrm_word_pos, gsrm_slf_attn_bias1,
            gsrm_slf_attn_bias2)

        # 使用 VSFD 模块处理 PVAM 特征和 GSRM 特征，得到最终输出
        final_out = self.vsfd(pvam_feature, gsrm_feature)
        # 如果不是训练阶段，则对最终输出进行 softmax 处理
        if not self.training:
            final_out = F.softmax(final_out, axis=1)

        # 获取最终输出中概率最高的类别
        _, decoded_out = paddle.topk(final_out, k=1)

        # 将预测结果和各个模块的特征输出整理为有序字典
        predicts = OrderedDict([
            ('predict', final_out),
            ('pvam_feature', pvam_feature),
            ('decoded_out', decoded_out),
            ('word_out', word_out),
            ('gsrm_out', gsrm_out),
        ])

        # 返回预测结果
        return predicts
```