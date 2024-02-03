# `.\PaddleOCR\ppocr\modeling\heads\rec_robustscanner_head.py`

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
"""
"""
此代码参考自:
https://github.com/open-mmlab/mmocr/blob/main/mmocr/models/textrecog/encoders/channel_reduction_encoder.py
https://github.com/open-mmlab/mmocr/blob/main/mmocr/models/textrecog/decoders/robust_scanner_decoder.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import paddle
from paddle import ParamAttr
import paddle.nn as nn
import paddle.nn.functional as F


class BaseDecoder(nn.Layer):
    def __init__(self, **kwargs):
        super().__init__()

    def forward_train(self, feat, out_enc, targets, img_metas):
        raise NotImplementedError

    def forward_test(self, feat, out_enc, img_metas):
        raise NotImplementedError

    def forward(self,
                feat,
                out_enc,
                label=None,
                valid_ratios=None,
                word_positions=None,
                train_mode=True):
        self.train_mode = train_mode

        if train_mode:
            return self.forward_train(feat, out_enc, label, valid_ratios,
                                      word_positions)
        return self.forward_test(feat, out_enc, valid_ratios, word_positions)


class ChannelReductionEncoder(nn.Layer):
    """Change the channel number with a one by one convoluational layer.
    Args:
        in_channels (int): 输入通道数。
        out_channels (int): 输出通道数。
    """

    def __init__(self, in_channels, out_channels, **kwargs):
        super(ChannelReductionEncoder, self).__init__()

        # 创建一个卷积层，将输入通道数转换为输出通道数
        self.layer = nn.Conv2D(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            weight_attr=nn.initializer.XavierNormal())

    def forward(self, feat):
        """
        Args:
            feat (Tensor): 具有形状为 :math:`(N, C_{in}, H, W)` 的图像特征。

        Returns:
            Tensor: 形状为 :math:`(N, C_{out}, H, W)` 的张量。
        """
        # 将特征传递给卷积层进行前向传播
        return self.layer(feat)
# 定义一个函数，根据给定的掩码和数值，将输入张量中的部分值替换为指定数值
def masked_fill(x, mask, value):
    # 创建一个形状和数据类型与输入张量相同的张量，填充指定数值
    y = paddle.full(x.shape, value, x.dtype)
    # 根据掩码条件，将指定数值填充到输入张量中
    return paddle.where(mask, y, x)

# 定义一个注意力层类，用于计算点积注意力
class DotProductAttentionLayer(nn.Layer):
    def __init__(self, dim_model=None):
        super().__init__()
        
        # 初始化缩放因子，根据模型维度进行缩放
        self.scale = dim_model**-0.5 if dim_model is not None else 1.

    def forward(self, query, key, value, h, w, valid_ratios=None):
        # 调整输入张量的维度顺序
        query = paddle.transpose(query, (0, 2, 1))
        # 计算点积注意力得分
        logits = paddle.matmul(query, key) * self.scale
        n, c, t = logits.shape
        # 重塑得分张量的形状为 (n, c, h, w)
        logits = paddle.reshape(logits, [n, c, h, w])
        if valid_ratios is not None:
            # 计算注意力权重的掩码
            with paddle.base.framework._stride_in_no_check_dy2st_diff():
                for i, valid_ratio in enumerate(valid_ratios):
                    valid_width = min(w, int(w * valid_ratio + 0.5))
                    if valid_width < w:
                        # 将超出有效宽度范围的部分设置为负无穷
                        logits[i, :, :, valid_width:] = float('-inf')

        # 重塑得分张量的形状为 (n, c, t)
        logits = paddle.reshape(logits, [n, c, t])
        # 计算注意力权重
        weights = F.softmax(logits, axis=2)
        # 调整值张量的维度顺序
        value = paddle.transpose(value, (0, 2, 1))
        # 计算注意力瞥视结果
        glimpse = paddle.matmul(weights, value)
        glimpse = paddle.transpose(glimpse, (0, 2, 1))
        return glimpse

# 定义一个序列注意力解码器类，用于RobustScanner
class SequenceAttentionDecoder(BaseDecoder):
    """Sequence attention decoder for RobustScanner.

    RobustScanner: `RobustScanner: Dynamically Enhancing Positional Clues for
    Robust Text Recognition <https://arxiv.org/abs/2007.07542>`_
    # 定义一个解码器类，用于将模型的输出转换为文本序列
    class AttentionDecoder(nn.Module):
        # 初始化函数，设置解码器的参数
        def __init__(self, num_classes, rnn_layers, dim_input, dim_model, max_seq_len, start_idx, mask, padding_idx, dropout, return_feature, encode_value):
            # 设置输出类别数目
            num_classes (int): Number of output classes :math:`C`.
            # 设置RNN层数
            rnn_layers (int): Number of RNN layers.
            # 设置输入向量的维度
            dim_input (int): Dimension :math:`D_i` of input vector ``feat``.
            # 设置模型的维度
            dim_model (int): Dimension :math:`D_m` of the model. Should also be the same as encoder output vector ``out_enc``.
            # 设置最大输出序列长度
            max_seq_len (int): Maximum output sequence length :math:`T`.
            # 设置起始索引
            start_idx (int): The index of `<SOS>`.
            # 是否根据`img_meta['valid_ratio']`对输入特征进行掩码处理
            mask (bool): Whether to mask input features according to ``img_meta['valid_ratio']``.
            # 设置填充索引
            padding_idx (int): The index of `<PAD>`.
            # 设置丢弃率
            dropout (float): Dropout rate.
            # 是否返回特征或logits作为结果
            return_feature (bool): Return feature or logits as the result.
            # 是否使用编码器`out_enc`的输出作为注意力层的`value`
            encode_value (bool): Whether to use the output of encoder ``out_enc`` as `value` of attention layer. If False, the original feature ``feat`` will be used.
    
        # 警告信息，解码器不会预测假设为`<PAD>`的最终类别，因此其输出大小始终为`C-1`。`<PAD>`也会被损失函数忽略
        Warning:
            This decoder will not predict the final class which is assumed to be `<PAD>`. Therefore, its output size is always :math:`C - 1`. `<PAD>` is also ignored by loss as specified in :obj:`mmocr.models.textrecog.recognizer.EncodeDecodeRecognizer`.
    # 初始化神经网络模型的参数
    def __init__(self,
                 num_classes=None,  # 类别数量
                 rnn_layers=2,  # RNN 层的数量
                 dim_input=512,  # 输入维度
                 dim_model=128,  # 模型维度
                 max_seq_len=40,  # 最大序列长度
                 start_idx=0,  # 起始索引
                 mask=True,  # 是否使用掩码
                 padding_idx=None,  # 填充索引
                 dropout=0,  # 丢弃率
                 return_feature=False,  # 是否返回特征
                 encode_value=False):  # 是否编码值
        super().__init__()

        # 初始化模型参数
        self.num_classes = num_classes
        self.dim_input = dim_input
        self.dim_model = dim_model
        self.return_feature = return_feature
        self.encode_value = encode_value
        self.max_seq_len = max_seq_len
        self.start_idx = start_idx
        self.mask = mask

        # 初始化嵌入层
        self.embedding = nn.Embedding(
            self.num_classes, self.dim_model, padding_idx=padding_idx)

        # 初始化序列层
        self.sequence_layer = nn.LSTM(
            input_size=dim_model,
            hidden_size=dim_model,
            num_layers=rnn_layers,
            time_major=False,
            dropout=dropout)

        # 初始化注意力层
        self.attention_layer = DotProductAttentionLayer()

        # 初始化预测层
        self.prediction = None
        if not self.return_feature:
            pred_num_classes = num_classes - 1
            self.prediction = nn.Linear(dim_model if encode_value else
                                        dim_input, pred_num_classes)
    def forward_train(self, feat, out_enc, targets, valid_ratios):
        """
        Args:
            feat (Tensor): Tensor of shape :math:`(N, D_i, H, W)`.
            out_enc (Tensor): Encoder output of shape
                :math:`(N, D_m, H, W)`.
            targets (Tensor): a tensor of shape :math:`(N, T)`. Each element is the index of a
                character.
            valid_ratios (Tensor): valid length ratio of img.
        Returns:
            Tensor: A raw logit tensor of shape :math:`(N, T, C-1)` if
            ``return_feature=False``. Otherwise it would be the hidden feature
            before the prediction projection layer, whose shape is
            :math:`(N, T, D_m)`.
        """

        # 使用目标字符的索引进行嵌入
        tgt_embedding = self.embedding(targets)

        # 获取输出编码器的形状信息
        n, c_enc, h, w = out_enc.shape
        assert c_enc == self.dim_model
        # 获取输入特征的形状信息
        _, c_feat, _, _ = feat.shape
        assert c_feat == self.dim_input
        # 获取目标嵌入的形状信息
        _, len_q, c_q = tgt_embedding.shape
        assert c_q == self.dim_model
        assert len_q <= self.max_seq_len

        # 对目标嵌入进行序列层处理
        query, _ = self.sequence_layer(tgt_embedding)
        # 调整维度顺序
        query = paddle.transpose(query, (0, 2, 1))
        # 将输出编码器重塑为二维形状
        key = paddle.reshape(out_enc, [n, c_enc, h * w])
        # 根据编码器输出和输入特征是否编码值来确定值
        if self.encode_value:
            value = key
        else:
            value = paddle.reshape(feat, [n, c_feat, h * w])

        # 使用注意力层处理查询、键和值
        attn_out = self.attention_layer(query, key, value, h, w, valid_ratios)
        # 调整维度顺序
        attn_out = paddle.transpose(attn_out, (0, 2, 1))

        # 如果需要返回特征，则直接返回注意力输出
        if self.return_feature:
            return attn_out

        # 否则，通过预测层处理注意力输出
        out = self.prediction(attn_out)

        return out
    # 定义一个前向推理函数，用于生成输出序列
    def forward_test(self, feat, out_enc, valid_ratios):
        """
        Args:
            feat (Tensor): Tensor of shape :math:`(N, D_i, H, W)`.
            out_enc (Tensor): Encoder output of shape
                :math:`(N, D_m, H, W)`.
            valid_ratios (Tensor): valid length ratio of img.

        Returns:
            Tensor: The output logit sequence tensor of shape
            :math:`(N, T, C-1)`.
        """
        # 获取最大序列长度和批量大小
        seq_len = self.max_seq_len
        batch_size = feat.shape[0]

        # 初始化解码序列，全部为起始索引
        decode_sequence = (paddle.ones(
            (batch_size, seq_len), dtype='int64') * self.start_idx)

        # 存储每个时间步的输出
        outputs = []
        # 遍历每个时间步
        for i in range(seq_len):
            # 调用前向推理的单步函数，生成当前时间步的输出
            step_out = self.forward_test_step(feat, out_enc, decode_sequence, i,
                                              valid_ratios)
            outputs.append(step_out)
            # 获取当前时间步的最大值索引
            max_idx = paddle.argmax(step_out, axis=1, keepdim=False)
            # 如果不是最后一个时间步，则更新解码序列
            if i < seq_len - 1:
                decode_sequence[:, i + 1] = max_idx

        # 将所有时间步的输出堆叠在一起
        outputs = paddle.stack(outputs, 1)

        # 返回输出序列
        return outputs
    # 定义前向测试步骤函数，用于模型前向推断
    def forward_test_step(self, feat, out_enc, decode_sequence, current_step,
                          valid_ratios):
        """
        Args:
            feat (Tensor): Tensor of shape :math:`(N, D_i, H, W)`.
            out_enc (Tensor): Encoder output of shape
                :math:`(N, D_m, H, W)`.
            decode_sequence (Tensor): Shape :math:`(N, T)`. The tensor that
                stores history decoding result.
            current_step (int): Current decoding step.
            valid_ratios (Tensor): valid length ratio of img

        Returns:
            Tensor: Shape :math:`(N, C-1)`. The logit tensor of predicted
            tokens at current time step.
        """

        # 将历史解码结果转换为嵌入向量
        embed = self.embedding(decode_sequence)

        # 获取 Encoder 输出的形状信息
        n, c_enc, h, w = out_enc.shape
        assert c_enc == self.dim_model
        # 获取输入特征的形状信息
        _, c_feat, _, _ = feat.shape
        assert c_feat == self.dim_input
        # 获取嵌入向量的形状信息
        _, _, c_q = embed.shape
        assert c_q == self.dim_model

        # 对嵌入向量进行序列层处理
        query, _ = self.sequence_layer(embed)
        # 调整 query 的维度顺序
        query = paddle.transpose(query, (0, 2, 1))
        # 将 Encoder 输出重塑为 key
        key = paddle.reshape(out_enc, [n, c_enc, h * w])
        # 根据是否使用 value 编码，选择相应的 value
        if self.encode_value:
            value = key
        else:
            value = paddle.reshape(feat, [n, c_feat, h * w])

        # 使用注意力层计算注意力输出
        # [n, c, l]
        attn_out = self.attention_layer(query, key, value, h, w, valid_ratios)
        # 获取当前步骤的输出
        out = attn_out[:, :, current_step]

        # 如果需要返回特征，则直接返回输出
        if self.return_feature:
            return out

        # 对输出进行预测
        out = self.prediction(out)
        # 对预测结果进行 softmax 处理
        out = F.softmax(out, dim=-1)

        return out
class PositionAwareLayer(nn.Layer):
    # 定义一个位置感知层，继承自 nn.Layer
    def __init__(self, dim_model, rnn_layers=2):
        # 初始化函数，接受模型维度和 RNN 层数作为参数
        super().__init__()

        self.dim_model = dim_model
        # 设置模型维度

        self.rnn = nn.LSTM(
            input_size=dim_model,
            hidden_size=dim_model,
            num_layers=rnn_layers,
            time_major=False)
        # 创建一个 LSTM 层，输入维度为模型维度，隐藏层维度为模型维度，层数为指定的 RNN 层数

        self.mixer = nn.Sequential(
            nn.Conv2D(
                dim_model, dim_model, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2D(
                dim_model, dim_model, kernel_size=3, stride=1, padding=1))
        # 创建一个混合层，包含两个卷积层和一个 ReLU 激活函数

    def forward(self, img_feature):
        # 定义前向传播函数，接受图像特征作为输入
        n, c, h, w = img_feature.shape
        # 获取图像特征的形状信息
        rnn_input = paddle.transpose(img_feature, (0, 2, 3, 1))
        # 调整输入数据的维度顺序
        rnn_input = paddle.reshape(rnn_input, (n * h, w, c))
        # 重新调整输入数据的形状
        rnn_output, _ = self.rnn(rnn_input)
        # 将输入数据传入 LSTM 层进行计算
        rnn_output = paddle.reshape(rnn_output, (n, h, w, c))
        # 调整 LSTM 输出的形状
        rnn_output = paddle.transpose(rnn_output, (0, 3, 1, 2))
        # 调整 LSTM 输出的维度顺序
        out = self.mixer(rnn_output)
        # 将 LSTM 输出传入混合层进行计算
        return out
        # 返回计算结果


class PositionAttentionDecoder(BaseDecoder):
    """Position attention decoder for RobustScanner.

    RobustScanner: `RobustScanner: Dynamically Enhancing Positional Clues for
    Robust Text Recognition <https://arxiv.org/abs/2007.07542>`_
    # 位置注意力解码器，用于 RobustScanner，参考 RobustScanner 论文链接
    # 定义一个函数，用于初始化解码器的参数
    Args:
        num_classes (int): 输出类别的数量 C
        rnn_layers (int): RNN 层的数量
        dim_input (int): 输入向量 feat 的维度 Di
        dim_model (int): 模型的维度 Dm，应该与编码器输出向量 out_enc 的维度相同
        max_seq_len (int): 输出序列的最大长度 T
        mask (bool): 是否根据 img_meta['valid_ratio'] 对输入特征进行掩码处理
        return_feature (bool): 返回特征或逻辑值作为结果
        encode_value (bool): 是否将编码器的输出 out_enc 用作注意力层的值。如果为 False，则使用原始特征 feat
    
    Warning:
        该解码器不会预测假设为 '<PAD>' 的最终类别。因此，其输出大小始终为 C - 1。'<PAD>' 也会被损失函数忽略
    # 初始化函数，设置模型参数和层
    def __init__(self,
                 num_classes=None,
                 rnn_layers=2,
                 dim_input=512,
                 dim_model=128,
                 max_seq_len=40,
                 mask=True,
                 return_feature=False,
                 encode_value=False):
        # 调用父类的初始化函数
        super().__init__()

        # 初始化模型参数
        self.num_classes = num_classes
        self.dim_input = dim_input
        self.dim_model = dim_model
        self.max_seq_len = max_seq_len
        self.return_feature = return_feature
        self.encode_value = encode_value
        self.mask = mask

        # 创建词嵌入层
        self.embedding = nn.Embedding(self.max_seq_len + 1, self.dim_model)

        # 创建位置感知模块
        self.position_aware_module = PositionAwareLayer(self.dim_model,
                                                        rnn_layers)

        # 创建注意力层
        self.attention_layer = DotProductAttentionLayer()

        # 初始化预测层
        self.prediction = None
        if not self.return_feature:
            pred_num_classes = num_classes - 1
            self.prediction = nn.Linear(dim_model if encode_value else
                                        dim_input, pred_num_classes)

    # 获取位置索引函数
    def _get_position_index(self, length, batch_size):
        position_index_list = []
        for i in range(batch_size):
            # 生成指定长度的位置索引
            position_index = paddle.arange(0, end=length, step=1, dtype='int64')
            position_index_list.append(position_index)
        # 将位置索引列表堆叠成张量
        batch_position_index = paddle.stack(position_index_list, axis=0)
        return batch_position_index
    # 前向传播训练函数，接受特征 feat，编码器输出 out_enc，目标 targets，有效长度比例 valid_ratios，位置索引 position_index 作为参数
    def forward_train(self, feat, out_enc, targets, valid_ratios,
                      position_index):
        """
        Args:
            feat (Tensor): Tensor of shape :math:`(N, D_i, H, W)`. 输入特征的张量，形状为(N, D_i, H, W)。
            out_enc (Tensor): Encoder output of shape :math:`(N, D_m, H, W)`. 编码器输出的张量，形状为(N, D_m, H, W)。
            targets (dict): A dict with the key ``padded_targets``, a
                tensor of shape :math:`(N, T)`. Each element is the index of a
                character. 目标字典，包含键为``padded_targets``的张量，形状为(N, T)，每个元素是一个字符的索引。
            valid_ratios (Tensor): valid length ratio of img. 图像的有效长度比例。
            position_index (Tensor): The position of each word. 每个单词的位置。

        Returns:
            Tensor: A raw logit tensor of shape :math:`(N, T, C-1)` if
            ``return_feature=False``. Otherwise it will be the hidden feature
            before the prediction projection layer, whose shape is
            :math:`(N, T, D_m)`. 返回值为张量，如果``return_feature=False``，形状为(N, T, C-1)的原始对数张量。否则，它将是预测投影层之前的隐藏特征，形状为(N, T, D_m)。

        """
        # 获取编码器输出的形状信息
        n, c_enc, h, w = out_enc.shape
        # 断言编码器输出的通道数等于模型的维度
        assert c_enc == self.dim_model
        # 获取输入特征的形状信息
        _, c_feat, _, _ = feat.shape
        # 断言输入特征的通道数等于输入维度
        assert c_feat == self.dim_input
        # 获取目标的形状信息
        _, len_q = targets.shape
        # 断言目标长度不超过最大序列长度
        assert len_q <= self.max_seq_len

        # 使用位置感知模块处理编码器输出
        position_out_enc = self.position_aware_module(out_enc)

        # 使用嵌入层获取查询信息，并进行转置
        query = self.embedding(position_index)
        query = paddle.transpose(query, (0, 2, 1))
        # 将编码器输出重塑为键
        key = paddle.reshape(position_out_enc, (n, c_enc, h * w))
        # 根据是否编码值，选择值为编码器输出或输入特征
        if self.encode_value:
            value = paddle.reshape(out_enc, (n, c_enc, h * w))
        else:
            value = paddle.reshape(feat, (n, c_feat, h * w))

        # 使用注意力层计算注意力输出
        attn_out = self.attention_layer(query, key, value, h, w, valid_ratios)
        attn_out = paddle.transpose(attn_out, (0, 2, 1))  # [n, len_q, dim_v]

        # 如果需要返回特征，则直接返回注意力输出
        if self.return_feature:
            return attn_out

        # 否则，通过预测层进行预测并返回结果
        return self.prediction(attn_out)
    def forward_test(self, feat, out_enc, valid_ratios, position_index):
        """
        Args:
            feat (Tensor): Tensor of shape :math:`(N, D_i, H, W)`.
            out_enc (Tensor): Encoder output of shape
                :math:`(N, D_m, H, W)`.
            valid_ratios (Tensor): valid length ratio of img
            position_index (Tensor): The position of each word.

        Returns:
            Tensor: A raw logit tensor of shape :math:`(N, T, C-1)` if
            ``return_feature=False``. Otherwise it would be the hidden feature
            before the prediction projection layer, whose shape is
            :math:`(N, T, D_m)`.
        """
        # 获取 Encoder 输出的维度信息
        n, c_enc, h, w = out_enc.shape
        # 断言 Encoder 输出的通道数等于模型的维度
        assert c_enc == self.dim_model
        # 获取输入特征的维度信息
        _, c_feat, _, _ = feat.shape
        # 断言输入特征的通道数等于输入维度
        assert c_feat == self.dim_input

        # 使用位置感知模块处理 Encoder 输出
        position_out_enc = self.position_aware_module(out_enc)

        # 通过嵌入层获取查询向量
        query = self.embedding(position_index)
        query = paddle.transpose(query, (0, 2, 1))
        # 将位置感知后的 Encoder 输出重塑为 key
        key = paddle.reshape(position_out_enc, (n, c_enc, h * w))
        # 根据是否编码值选择 value
        if self.encode_value:
            value = paddle.reshape(out_enc, (n, c_enc, h * w))
        else:
            value = paddle.reshape(feat, (n, c_feat, h * w))

        # 使用注意力层计算注意力输出
        attn_out = self.attention_layer(query, key, value, h, w, valid_ratios)
        attn_out = paddle.transpose(attn_out, (0, 2, 1))  # [n, len_q, dim_v]

        # 如果需要返回特征，则直接返回注意力输出
        if self.return_feature:
            return attn_out

        # 否则通过预测层得到最终输出
        return self.prediction(attn_out)
class RobustScannerFusionLayer(nn.Layer):
    # 定义 RobustScannerFusionLayer 类，继承自 nn.Layer
    def __init__(self, dim_model, dim=-1):
        # 初始化函数，接受 dim_model 和 dim 两个参数
        super(RobustScannerFusionLayer, self).__init__()
        # 调用父类的初始化函数

        self.dim_model = dim_model
        # 设置对象属性 dim_model 为传入的 dim_model
        self.dim = dim
        # 设置对象属性 dim 为传入的 dim
        self.linear_layer = nn.Linear(dim_model * 2, dim_model * 2)
        # 创建一个线性层，输入维度为 dim_model * 2，输出维度为 dim_model * 2

    def forward(self, x0, x1):
        # 定义前向传播函数，接受两个输入 x0 和 x1
        assert x0.shape == x1.shape
        # 断言 x0 和 x1 的形状相同
        fusion_input = paddle.concat([x0, x1], self.dim)
        # 将 x0 和 x1 沿着维度 self.dim 进行拼接
        output = self.linear_layer(fusion_input)
        # 将拼接后的输入传入线性层进行计算
        output = F.glu(output, self.dim)
        # 使用门控线性单元（GLU）激活函数对输出进行处理
        return output
        # 返回处理后的输出


class RobustScannerDecoder(BaseDecoder):
    # 定义 RobustScannerDecoder 类，继承自 BaseDecoder
    """Decoder for RobustScanner.

    RobustScanner: `RobustScanner: Dynamically Enhancing Positional Clues for
    Robust Text Recognition <https://arxiv.org/abs/2007.07542>`_

    Args:
        num_classes (int): Number of output classes :math:`C`.
        dim_input (int): Dimension :math:`D_i` of input vector ``feat``.
        dim_model (int): Dimension :math:`D_m` of the model. Should also be the
            same as encoder output vector ``out_enc``.
        max_seq_len (int): Maximum output sequence length :math:`T`.
        start_idx (int): The index of `<SOS>`.
        mask (bool): Whether to mask input features according to
            ``img_meta['valid_ratio']``.
        padding_idx (int): The index of `<PAD>`.
        encode_value (bool): Whether to use the output of encoder ``out_enc``
            as `value` of attention layer. If False, the original feature
            ``feat`` will be used.

    Warning:
        This decoder will not predict the final class which is assumed to be
        `<PAD>`. Therefore, its output size is always :math:`C - 1`. `<PAD>`
        is also ignored by loss as specified in
        :obj:`mmocr.models.textrecog.recognizer.EncodeDecodeRecognizer`.
    """
    # RobustScanner 的解码器，包含了一些参数和注意事项的说明
    # 初始化类的构造函数，设置各种参数的默认值
    def __init__(self,
                 num_classes=None,  # 类别数量，默认为None
                 dim_input=512,  # 输入维度，默认为512
                 dim_model=128,  # 模型维度，默认为128
                 hybrid_decoder_rnn_layers=2,  # 混合解码器的RNN层数，默认为2
                 hybrid_decoder_dropout=0,  # 混合解码器的dropout，默认为0
                 position_decoder_rnn_layers=2,  # 位置解码器的RNN层数，默认为2
                 max_seq_len=40,  # 最大序列长度，默认为40
                 start_idx=0,  # 起始索引，默认为0
                 mask=True,  # 是否使用mask，默认为True
                 padding_idx=None,  # 填充索引，默认为None
                 encode_value=False):  # 是否编码值，默认为False
        # 调用父类的构造函数
        super().__init__()
        # 初始化各个参数
        self.num_classes = num_classes
        self.dim_input = dim_input
        self.dim_model = dim_model
        self.max_seq_len = max_seq_len
        self.encode_value = encode_value
        self.start_idx = start_idx
        self.padding_idx = padding_idx
        self.mask = mask

        # 初始化混合解码器
        self.hybrid_decoder = SequenceAttentionDecoder(
            num_classes=num_classes,
            rnn_layers=hybrid_decoder_rnn_layers,
            dim_input=dim_input,
            dim_model=dim_model,
            max_seq_len=max_seq_len,
            start_idx=start_idx,
            mask=mask,
            padding_idx=padding_idx,
            dropout=hybrid_decoder_dropout,
            encode_value=encode_value,
            return_feature=True)

        # 初始化位置解码器
        self.position_decoder = PositionAttentionDecoder(
            num_classes=num_classes,
            rnn_layers=position_decoder_rnn_layers,
            dim_input=dim_input,
            dim_model=dim_model,
            max_seq_len=max_seq_len,
            mask=mask,
            encode_value=encode_value,
            return_feature=True)

        # 初始化融合模块
        self.fusion_module = RobustScannerFusionLayer(
            self.dim_model if encode_value else dim_input)

        # 预测类别数量
        pred_num_classes = num_classes - 1
        # 初始化预测层
        self.prediction = nn.Linear(dim_model if encode_value else dim_input,
                                    pred_num_classes)
    # 定义一个方法用于前向传播训练，接受特征、编码器输出、目标、有效比例和单词位置作为参数
    def forward_train(self, feat, out_enc, target, valid_ratios,
                      word_positions):
        """
        Args:
            feat (Tensor): Tensor of shape :math:`(N, D_i, H, W)`.
            out_enc (Tensor): Encoder output of shape
                :math:`(N, D_m, H, W)`.
            target (dict): A dict with the key ``padded_targets``, a
                tensor of shape :math:`(N, T)`. Each element is the index of a
                character.
            valid_ratios (Tensor): 
            word_positions (Tensor): The position of each word.

        Returns:
            Tensor: A raw logit tensor of shape :math:`(N, T, C-1)`.
        """
        # 使用混合解码器进行前向传播训练，得到混合瞥视结果
        hybrid_glimpse = self.hybrid_decoder.forward_train(feat, out_enc,
                                                           target, valid_ratios)
        # 使用位置解码器进行前向传播训练，得到位置瞥视结果
        position_glimpse = self.position_decoder.forward_train(
            feat, out_enc, target, valid_ratios, word_positions)

        # 使用融合模块融合混合瞥视和位置瞥视结果
        fusion_out = self.fusion_module(hybrid_glimpse, position_glimpse)

        # 使用预测模块对融合结果进行预测
        out = self.prediction(fusion_out)

        # 返回预测结果
        return out
    # 定义前向推理函数，接受特征 feat、编码器输出 out_enc、有效比例 valid_ratios 和单词位置 word_positions 作为输入
    def forward_test(self, feat, out_enc, valid_ratios, word_positions):
        """
        Args:
            feat (Tensor): Tensor of shape :math:`(N, D_i, H, W)`.
            out_enc (Tensor): Encoder output of shape
                :math:`(N, D_m, H, W)`.
            valid_ratios (Tensor): 
            word_positions (Tensor): The position of each word.
        Returns:
            Tensor: The output logit sequence tensor of shape
            :math:`(N, T, C-1)`.
        """
        # 获取最大序列长度和批量大小
        seq_len = self.max_seq_len
        batch_size = feat.shape[0]

        # 初始化解码序列，全部填充为起始索引
        decode_sequence = (paddle.ones(
            (batch_size, seq_len), dtype='int64') * self.start_idx)

        # 获取位置解码器的位置信息
        position_glimpse = self.position_decoder.forward_test(
            feat, out_enc, valid_ratios, word_positions)

        # 初始化输出列表
        outputs = []
        # 遍历序列长度
        for i in range(seq_len):
            # 获取混合解码器的输出
            hybrid_glimpse_step = self.hybrid_decoder.forward_test_step(
                feat, out_enc, decode_sequence, i, valid_ratios)

            # 融合混合解码器输出和位置信息
            fusion_out = self.fusion_module(hybrid_glimpse_step,
                                            position_glimpse[:, i, :])

            # 获取字符预测输出
            char_out = self.prediction(fusion_out)
            # 对字符预测输出进行 softmax 处理
            char_out = F.softmax(char_out, -1)
            # 将字符预测输出添加到输出列表中
            outputs.append(char_out)
            # 获取概率最大的字符索引
            max_idx = paddle.argmax(char_out, axis=1, keepdim=False)
            # 更新解码序列
            if i < seq_len - 1:
                decode_sequence[:, i + 1] = max_idx

        # 将输出列表堆叠成张量
        outputs = paddle.stack(outputs, 1)

        # 返回输出张量
        return outputs
class RobustScannerHead(nn.Layer):
    # 定义 RobustScannerHead 类，用于实现扫描器的头部模块
    def __init__(
            self,
            out_channels,  # 输出通道数，包括90个已知类别、未知类别、起始标记和填充标记
            in_channels,  # 输入通道数
            enc_outchannles=128,  # 编码器输出通道数，默认为128
            hybrid_dec_rnn_layers=2,  # 混合解码器的 RNN 层级数，默认为2
            hybrid_dec_dropout=0,  # 混合解码器的 Dropout 比例，默认为0
            position_dec_rnn_layers=2,  # 位置解码器的 RNN 层级数，默认为2
            start_idx=0,  # 起始标记的索引，默认为0
            max_text_length=40,  # 最大文本长度，默认为40
            mask=True,  # 是否使用掩码，默认为True
            padding_idx=None,  # 填充标记的索引，默认为None
            encode_value=False,  # 是否编码值，默认为False
            **kwargs):
        super(RobustScannerHead, self).__init__()

        # encoder module
        # 创建 ChannelReductionEncoder 实例作为编码器模块
        self.encoder = ChannelReductionEncoder(
            in_channels=in_channels, out_channels=enc_outchannles)

        # decoder module
        # 创建 RobustScannerDecoder 实例作为解码器模块
        self.decoder = RobustScannerDecoder(
            num_classes=out_channels,
            dim_input=in_channels,
            dim_model=enc_outchannles,
            hybrid_decoder_rnn_layers=hybrid_dec_rnn_layers,
            hybrid_decoder_dropout=hybrid_dec_dropout,
            position_decoder_rnn_layers=position_dec_rnn_layers,
            max_seq_len=max_text_length,
            start_idx=start_idx,
            mask=mask,
            padding_idx=padding_idx,
            encode_value=encode_value)
    # 定义一个前向传播函数，接受输入和目标参数
    def forward(self, inputs, targets=None):
        '''
        targets: [label, valid_ratio, word_positions]
        '''
        # 使用编码器对输入进行编码
        out_enc = self.encoder(inputs)
        # 从目标参数中获取单词位置信息
        word_positions = targets[-1]

        # 如果目标参数的长度大于1
        if len(targets) > 1:
            # 从目标参数中获取有效比例信息
            valid_ratios = targets[-2]

        # 如果处于训练模式
        if self.training:
            # 从目标参数中获取标签信息
            label = targets[0]  # label
            # 将标签转换为张量类型
            label = paddle.to_tensor(label, dtype='int64')
            # 使用解码器进行解码
            final_out = self.decoder(inputs, out_enc, label, valid_ratios,
                                     word_positions)
        # 如果不处于训练模式
        if not self.training:
            # 使用解码器进行解码，不传入标签信息
            final_out = self.decoder(
                inputs,
                out_enc,
                label=None,
                valid_ratios=valid_ratios,
                word_positions=word_positions,
                train_mode=False)
        # 返回最终输出结果
        return final_out
```