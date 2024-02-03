# `.\PaddleOCR\ppocr\modeling\heads\rec_sar_head.py`

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
# 均基于“按原样”分发，没有任何明示或暗示的保证或条件
# 请查看许可证以获取有关权限和限制的详细信息
"""
# 代码参考自:
# https://github.com/open-mmlab/mmocr/blob/main/mmocr/models/textrecog/encoders/sar_encoder.py
# https://github.com/open-mmlab/mmocr/blob/main/mmocr/models/textrecog/decoders/sar_decoder.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 导入必要的库
import math
import paddle
from paddle import ParamAttr
import paddle.nn as nn
import paddle.nn.functional as F

# 定义 SAREncoder 类，用于文本识别的编码器
class SAREncoder(nn.Layer):
    """
    Args:
        enc_bi_rnn (bool): If True, use bidirectional RNN in encoder.
        enc_drop_rnn (float): Dropout probability of RNN layer in encoder.
        enc_gru (bool): If True, use GRU, else LSTM in encoder.
        d_model (int): Dim of channels from backbone.
        d_enc (int): Dim of encoder RNN layer.
        mask (bool): If True, mask padding in RNN sequence.
    """
    # 初始化函数，设置编码器的参数
    def __init__(self,
                 enc_bi_rnn=False,  # 是否使用双向 RNN，默认为 False
                 enc_drop_rnn=0.1,  # RNN 的 dropout 概率，默认为 0.1
                 enc_gru=False,     # 是否使用 GRU，默认为 False
                 d_model=512,       # 模型维度，默认为 512
                 d_enc=512,         # 编码器维度，默认为 512
                 mask=True,         # 是否使用掩码，默认为 True
                 **kwargs):         # 其他参数

        super().__init__()  # 调用父类的初始化函数

        # 断言确保参数的类型和取值范围
        assert isinstance(enc_bi_rnn, bool)
        assert isinstance(enc_drop_rnn, (int, float))
        assert 0 <= enc_drop_rnn < 1.0
        assert isinstance(enc_gru, bool)
        assert isinstance(d_model, int)
        assert isinstance(d_enc, int)
        assert isinstance(mask, bool)

        # 设置编码器的参数
        self.enc_bi_rnn = enc_bi_rnn
        self.enc_drop_rnn = enc_drop_rnn
        self.mask = mask

        # LSTM 编码器
        if enc_bi_rnn:
            direction = 'bidirectional'
        else:
            direction = 'forward'
        
        # 根据参数设置 RNN 的参数
        kwargs = dict(
            input_size=d_model,
            hidden_size=d_enc,
            num_layers=2,
            time_major=False,
            dropout=enc_drop_rnn,
            direction=direction)
        
        # 根据是否使用 GRU 初始化 RNN 编码器
        if enc_gru:
            self.rnn_encoder = nn.GRU(**kwargs)
        else:
            self.rnn_encoder = nn.LSTM(**kwargs)

        # 全局特征转换
        encoder_rnn_out_size = d_enc * (int(enc_bi_rnn) + 1)
        self.linear = nn.Linear(encoder_rnn_out_size, encoder_rnn_out_size)
    # 前向传播函数，接受特征 feat 和图像元信息 img_metas
    def forward(self, feat, img_metas=None):
        # 如果图像元信息不为空，则确保第一个元信息的长度与特征的第一维大小相同
        if img_metas is not None:
            assert len(img_metas[0]) == paddle.shape(feat)[0]

        # 初始化有效比例为 None
        valid_ratios = None
        # 如果图像元信息不为空且需要进行遮罩处理
        if img_metas is not None and self.mask:
            # 获取最后一个元信息作为有效比例
            valid_ratios = img_metas[-1]

        # 获取特征的高度
        h_feat = feat.shape[2]  # bsz c h w
        # 在特征的高度维度上进行最大池化操作
        feat_v = F.max_pool2d(
            feat, kernel_size=(h_feat, 1), stride=1, padding=0)
        # 压缩特征的高度维度
        feat_v = feat_v.squeeze(2)  # bsz * C * W
        # 调换特征的维度顺序
        feat_v = paddle.transpose(feat_v, perm=[0, 2, 1])  # bsz * W * C
        # 使用 RNN 编码器处理特征，获取整体特征
        holistic_feat = self.rnn_encoder(feat_v)[0]  # bsz * T * C

        # 如果存在有效比例
        if valid_ratios is not None:
            # 初始化有效的整体特征列表
            valid_hf = []
            # 获取整体特征的时间步数
            T = paddle.shape(holistic_feat)[1]
            # 遍历有效比例
            for i in range(paddle.shape(valid_ratios)[0]):
                # 计算有效步数
                valid_step = paddle.minimum(
                    T, paddle.ceil(valid_ratios[i] * T).astype('int32')) - 1
                # 将有效的整体特征添加到列表中
                valid_hf.append(holistic_feat[i, valid_step, :])
            # 在维度 0 上堆叠有效的整体特征
            valid_hf = paddle.stack(valid_hf, axis=0)
        else:
            # 如果不存在有效比例，则直接获取最后一个时间步的整体特征
            valid_hf = holistic_feat[:, -1, :]  # bsz * C
        # 使用线性层处理有效的整体特征，得到最终整体特征
        holistic_feat = self.linear(valid_hf)  # bsz * C

        # 返回最终整体特征
        return holistic_feat
class BaseDecoder(nn.Layer):
    # 定义基础解码器类
    def __init__(self, **kwargs):
        # 初始化函数，接受任意关键字参数
        super().__init__()
        # 调用父类的初始化函数

    def forward_train(self, feat, out_enc, targets, img_metas):
        # 定义训练时的前向传播函数，接受特征、编码器输出、目标、图像元数据作为输入
        raise NotImplementedError
        # 抛出未实现错误

    def forward_test(self, feat, out_enc, img_metas):
        # 定义测试时的前向传播函数，接受特征、编码器输出、图像元数据作为输入
        raise NotImplementedError
        # 抛出未实现错误

    def forward(self,
                feat,
                out_enc,
                label=None,
                img_metas=None,
                train_mode=True):
        # 定义前向传播函数，接受特征、编码器输出、标签、图像元数据和训练模式作为输入
        self.train_mode = train_mode
        # 设置训练模式标志

        if train_mode:
            # 如果是训练模式
            return self.forward_train(feat, out_enc, label, img_metas)
            # 调用训练时的前向传播函数
        return self.forward_test(feat, out_enc, img_metas)
        # 否则调用测试时的前向传播函数


class ParallelSARDecoder(BaseDecoder):
    """
    Args:
        out_channels (int): Output class number.
        enc_bi_rnn (bool): If True, use bidirectional RNN in encoder.
        dec_bi_rnn (bool): If True, use bidirectional RNN in decoder.
        dec_drop_rnn (float): Dropout of RNN layer in decoder.
        dec_gru (bool): If True, use GRU, else LSTM in decoder.
        d_model (int): Dim of channels from backbone.
        d_enc (int): Dim of encoder RNN layer.
        d_k (int): Dim of channels of attention module.
        pred_dropout (float): Dropout probability of prediction layer.
        max_seq_len (int): Maximum sequence length for decoding.
        mask (bool): If True, mask padding in feature map.
        start_idx (int): Index of start token.
        padding_idx (int): Index of padding token.
        pred_concat (bool): If True, concat glimpse feature from
            attention with holistic feature and hidden state.
    """
    # 并行SAR解码器类，继承自基础解码器类，包含一系列参数
    # 定义前向传播函数，接受特征、编码器输出、标签和图像元信息作为输入
    def forward_train(self, feat, out_enc, label, img_metas):
        '''
        img_metas: [label, valid_ratio]
        '''
        # 如果图像元信息不为空，则确保标签的数量与特征的数量相同
        if img_metas is not None:
            assert paddle.shape(img_metas[0])[0] == paddle.shape(feat)[0]

        valid_ratios = None
        # 如果图像元信息不为空且模型需要使用掩码，则获取有效比例
        if img_metas is not None and self.mask:
            valid_ratios = img_metas[-1]

        # 将标签通过嵌入层转换为标签嵌入向量
        lab_embedding = self.embedding(label)
        # 将编码器输出扩展一个维度，转换为与标签嵌入向量相同的数据类型
        out_enc = out_enc.unsqueeze(1).astype(lab_embedding.dtype)
        # 将编码器输出和标签嵌入向量在指定维度上拼接起来
        in_dec = paddle.concat((out_enc, lab_embedding), axis=1)
        # 使用二维注意力机制生成解码器输出
        out_dec = self._2d_attention(
            in_dec, feat, out_enc, valid_ratios=valid_ratios)

        # 返回解码器输出的第二个元素开始的部分，即去除了起始标记的输出
        return out_dec[:, 1:, :]  # bsz * seq_len * num_classes
    # 执行前向测试，生成输出序列
    def forward_test(self, feat, out_enc, img_metas):
        # 如果图像元信息不为空，则确保第一个元信息的长度与特征的行数相同
        if img_metas is not None:
            assert len(img_metas[0]) == feat.shape[0]

        valid_ratios = None
        # 如果图像元信息不为空且模型需要掩码，则获取最后一个元信息作为有效比例
        if img_metas is not None and self.mask:
            valid_ratios = img_metas[-1]

        # 获取最大序列长度和批大小
        seq_len = self.max_seq_len
        bsz = feat.shape[0]
        # 创建起始标记张量，填充值为起始索引，数据类型为int64
        start_token = paddle.full(
            (bsz, ), fill_value=self.start_idx, dtype='int64')
        # 将起始标记张量进行嵌入
        start_token = self.embedding(start_token)
        # 获取嵌入维度
        emb_dim = start_token.shape[1]
        start_token = start_token.unsqueeze(1)
        start_token = paddle.expand(start_token, shape=[bsz, seq_len, emb_dim])
        # 将out_enc张量在第一维度上进行扩展
        out_enc = out_enc.unsqueeze(1)
        # 将out_enc和起始标记张量在第二维度上进行拼接
        decoder_input = paddle.concat((out_enc, start_token), axis=1)

        # 初始化输出列表
        outputs = []
        # 遍历序列长度
        for i in range(1, seq_len + 1):
            # 使用二维注意力机制获取解码器输出
            decoder_output = self._2d_attention(
                decoder_input, feat, out_enc, valid_ratios=valid_ratios)
            # 获取当前字符的输出
            char_output = decoder_output[:, i, :]  # bsz * num_classes
            # 对字符输出进行softmax操作
            char_output = F.softmax(char_output, -1)
            outputs.append(char_output)
            # 获取概率最大的字符索引
            max_idx = paddle.argmax(char_output, axis=1, keepdim=False)
            # 获取概率最大字符的嵌入表示
            char_embedding = self.embedding(max_idx)  # bsz * emb_dim
            # 如果未到达序列长度上限，则将字符嵌入添加到解码器输入中
            if i < seq_len:
                decoder_input[:, i + 1, :] = char_embedding

        # 将输出列表堆叠为张量
        outputs = paddle.stack(outputs, 1)  # bsz * seq_len * num_classes

        return outputs
class SARHead(nn.Layer):
    # 定义 SARHead 类，继承自 nn.Layer
    def __init__(self,
                 in_channels,
                 out_channels,
                 enc_dim=512,
                 max_text_length=30,
                 enc_bi_rnn=False,
                 enc_drop_rnn=0.1,
                 enc_gru=False,
                 dec_bi_rnn=False,
                 dec_drop_rnn=0.0,
                 dec_gru=False,
                 d_k=512,
                 pred_dropout=0.1,
                 pred_concat=True,
                 **kwargs):
        # 初始化函数，接受多个参数
        super(SARHead, self).__init__()
        # 调用父类的初始化函数

        # encoder module
        self.encoder = SAREncoder(
            enc_bi_rnn=enc_bi_rnn,
            enc_drop_rnn=enc_drop_rnn,
            enc_gru=enc_gru,
            d_model=in_channels,
            d_enc=enc_dim)
        # 创建 SAREncoder 对象作为 encoder 模块

        # decoder module
        self.decoder = ParallelSARDecoder(
            out_channels=out_channels,
            enc_bi_rnn=enc_bi_rnn,
            dec_bi_rnn=dec_bi_rnn,
            dec_drop_rnn=dec_drop_rnn,
            dec_gru=dec_gru,
            d_model=in_channels,
            d_enc=enc_dim,
            d_k=d_k,
            pred_dropout=pred_dropout,
            max_text_length=max_text_length,
            pred_concat=pred_concat)
        # 创建 ParallelSARDecoder 对象作为 decoder 模块

    def forward(self, feat, targets=None):
        '''
        img_metas: [label, valid_ratio]
        '''
        # 前向传播函数，接受 feat 和 targets 作为输入
        holistic_feat = self.encoder(feat, targets)  # bsz c
        # 使用 encoder 处理 feat 和 targets，得到 holistic_feat

        if self.training:
            label = targets[0]  # label
            final_out = self.decoder(
                feat, holistic_feat, label, img_metas=targets)
        else:
            final_out = self.decoder(
                feat,
                holistic_feat,
                label=None,
                img_metas=targets,
                train_mode=False)
            # (bsz, seq_len, num_classes)

        return final_out
        # 返回 final_out
```