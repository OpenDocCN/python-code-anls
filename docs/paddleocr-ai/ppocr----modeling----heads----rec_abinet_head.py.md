# `.\PaddleOCR\ppocr\modeling\heads\rec_abinet_head.py`

```py
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权;
# 除非符合许可证的规定，否则不得使用此文件。
# 您可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"原样"的基础分发的，
# 没有任何明示或暗示的保证或条件。
# 请查看许可证以获取特定语言的权限和限制。
"""
# 代码来源于：
# https://github.com/FangShancheng/ABINet/tree/main/modules

# 导入所需的库
import math
import paddle
from paddle import nn
import paddle.nn.functional as F
from paddle.nn import LayerList
from ppocr.modeling.heads.rec_nrtr_head import TransformerBlock, PositionalEncoding

# 定义 BCNLanguage 类，继承自 nn.Layer
class BCNLanguage(nn.Layer):
    # 初始化函数，设置模型参数和层数
    def __init__(self,
                 d_model=512,
                 nhead=8,
                 num_layers=4,
                 dim_feedforward=2048,
                 dropout=0.,
                 max_length=25,
                 detach=True,
                 num_classes=37):
        # 调用父类的初始化函数
        super().__init__()

        # 设置模型参数
        self.d_model = d_model
        self.detach = detach
        self.max_length = max_length + 1  # additional stop token
        # 创建一个线性层，用于将类别数映射到模型维度
        self.proj = nn.Linear(num_classes, d_model, bias_attr=False)
        # 创建一个位置编码器，用于编码输入的标记
        self.token_encoder = PositionalEncoding(
            dropout=0.1, dim=d_model, max_len=self.max_length)
        # 创建一个位置编码器，用于编码位置信息
        self.pos_encoder = PositionalEncoding(
            dropout=0, dim=d_model, max_len=self.max_length)

        # 创建多层TransformerBlock组成的解码器
        self.decoder = nn.LayerList([
            TransformerBlock(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                attention_dropout_rate=dropout,
                residual_dropout_rate=dropout,
                with_self_attn=False,
                with_cross_attn=True) for i in range(num_layers)
        ])

        # 创建一个线性层，用于将模型输出映射到类别数
        self.cls = nn.Linear(d_model, num_classes)

    # 前向传播函数
    def forward(self, tokens, lengths):
        """
        Args:
            tokens: (B, N, C) where N is length, B is batch size and C is classes number
            lengths: (B,)
        """
        # 如果detach为True，则将tokens进行detach操作
        if self.detach: tokens = tokens.detach()
        # 将输入tokens映射到模型维度
        embed = self.proj(tokens)  # (B, N, C)
        # 对映射后的数据进行位置编码
        embed = self.token_encoder(embed)  # (B, N, C)
        # 获取填充掩码
        padding_mask = _get_mask(lengths, self.max_length)
        zeros = paddle.zeros_like(embed)  # (B, N, C)
        qeury = self.pos_encoder(zeros)
        # 遍历解码器中的每一层，并进行解码操作
        for decoder_layer in self.decoder:
            qeury = decoder_layer(qeury, embed, cross_mask=padding_mask)
        output = qeury  # (B, N, C)

        # 将解码后的输出映射到类别数
        logits = self.cls(output)  # (B, N, C)

        return output, logits
# 定义一个编码器层，包括卷积层、批归一化层和ReLU激活函数
def encoder_layer(in_c, out_c, k=3, s=2, p=1):
    return nn.Sequential(
        nn.Conv2D(in_c, out_c, k, s, p), nn.BatchNorm2D(out_c), nn.ReLU())

# 定义一个解码器层，包括上采样层、卷积层、批归一化层和ReLU激活函数
def decoder_layer(in_c,
                  out_c,
                  k=3,
                  s=1,
                  p=1,
                  mode='nearest',
                  scale_factor=None,
                  size=None):
    align_corners = False if mode == 'nearest' else True
    return nn.Sequential(
        nn.Upsample(
            size=size,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners),
        nn.Conv2D(in_c, out_c, k, s, p),
        nn.BatchNorm2D(out_c),
        nn.ReLU())

# 定义一个位置注意力类，包括编码器、解码器、位置编码和线性变换
class PositionAttention(nn.Layer):
    def __init__(self,
                 max_length,
                 in_channels=512,
                 num_channels=64,
                 h=8,
                 w=32,
                 mode='nearest',
                 **kwargs):
        super().__init__()
        self.max_length = max_length
        # 编码器部分，包括多个编码器层
        self.k_encoder = nn.Sequential(
            encoder_layer(
                in_channels, num_channels, s=(1, 2)),
            encoder_layer(
                num_channels, num_channels, s=(2, 2)),
            encoder_layer(
                num_channels, num_channels, s=(2, 2)),
            encoder_layer(
                num_channels, num_channels, s=(2, 2)))
        # 解码器部分，包括多个解码器层
        self.k_decoder = nn.Sequential(
            decoder_layer(
                num_channels, num_channels, scale_factor=2, mode=mode),
            decoder_layer(
                num_channels, num_channels, scale_factor=2, mode=mode),
            decoder_layer(
                num_channels, num_channels, scale_factor=2, mode=mode),
            decoder_layer(
                num_channels, in_channels, size=(h, w), mode=mode))

        # 位置编码部分
        self.pos_encoder = PositionalEncoding(
            dropout=0, dim=in_channels, max_len=max_length)
        # 线性变换
        self.project = nn.Linear(in_channels, in_channels)
    # 前向传播函数，接受输入张量 x
    def forward(self, x):
        # 获取输入张量 x 的形状信息
        B, C, H, W = x.shape
        # 初始化 key 和 value 为输入张量 x
        k, v = x, x

        # 计算 key 向量
        features = []
        # 对 k 进行编码
        for i in range(0, len(self.k_encoder)):
            k = self.k_encoder[i](k)
            features.append(k)
        # 对 k 进行解码
        for i in range(0, len(self.k_decoder) - 1):
            k = self.k_decoder[i](k)
            # k 与对应的编码结果相加
            k = k + features[len(self.k_decoder) - 2 - i]
        k = self.k_decoder[-1](k)

        # 计算 query 向量
        # TODO: 完善 q=f(q,k) 的计算
        # 创建全零张量作为输入 query
        zeros = paddle.zeros(
            (B, self.max_length, C), dtype=x.dtype)  # (T, N, C)
        q = self.pos_encoder(zeros)  # (B, N, C)
        q = self.project(q)  # (B, N, C)

        # 计算注意力分数
        attn_scores = q @ k.flatten(2)  # (B, N, (H*W))
        attn_scores = attn_scores / (C**0.5)
        attn_scores = F.softmax(attn_scores, axis=-1)

        # 对 value 进行变形和转置
        v = v.flatten(2).transpose([0, 2, 1])  # (B, (H*W), C)
        # 计算注意力向量
        attn_vecs = attn_scores @ v  # (B, N, C)

        # 返回注意力向量和注意力分数，并将注意力分数重塑为与输入张量相同形状
        return attn_vecs, attn_scores.reshape([0, self.max_length, H, W])
# 定义一个名为ABINetHead的类，继承自nn.Layer
class ABINetHead(nn.Layer):
    # 初始化函数，接受多个参数
    def __init__(self,
                 in_channels,
                 out_channels,
                 d_model=512,
                 nhead=8,
                 num_layers=3,
                 dim_feedforward=2048,
                 dropout=0.1,
                 max_length=25,
                 use_lang=False,
                 iter_size=1):
        # 调用父类的初始化函数
        super().__init__()
        # 设置最大长度为max_length+1
        self.max_length = max_length + 1
        # 创建位置编码器对象，设置dropout为0.1，维度为d_model，最大长度为8*32
        self.pos_encoder = PositionalEncoding(
            dropout=0.1, dim=d_model, max_len=8 * 32)
        # 创建编码器对象列表，包含num_layers个TransformerBlock对象
        self.encoder = nn.LayerList([
            TransformerBlock(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                attention_dropout_rate=dropout,
                residual_dropout_rate=dropout,
                with_self_attn=True,
                with_cross_attn=False) for i in range(num_layers)
        ])
        # 创建解码器对象PositionAttention，设置最大长度为max_length+1，模式为'nearest'
        self.decoder = PositionAttention(
            max_length=max_length + 1,  # additional stop token
            mode='nearest', )
        # 设置输出通道数为out_channels
        self.out_channels = out_channels
        # 创建线性层对象cls，输入维度为d_model，输出维度为out_channels
        self.cls = nn.Linear(d_model, self.out_channels)
        # 设置是否使用语言模型的标志
        self.use_lang = use_lang
        # 如果使用语言模型
        if use_lang:
            # 设置迭代次数为iter_size
            self.iter_size = iter_size
            # 创建BCNLanguage对象language，设置参数
            self.language = BCNLanguage(
                d_model=d_model,
                nhead=nhead,
                num_layers=4,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                max_length=max_length,
                num_classes=self.out_channels)
            # 创建线性层对象w_att_align，输入维度为2*d_model，输出维度为d_model
            self.w_att_align = nn.Linear(2 * d_model, d_model)
            # 创建线性层对象cls_align，输入维度为d_model，输出维度为out_channels
            self.cls_align = nn.Linear(d_model, self.out_channels)
    # 前向传播函数，接受输入 x 和目标 targets
    def forward(self, x, targets=None):
        # 将输入 x 进行维度转换，调整维度顺序为 [batch, height, width, channels]
        x = x.transpose([0, 2, 3, 1])
        # 获取输入 x 的形状信息
        _, H, W, C = x.shape
        # 将输入 x 展平为一维特征向量
        feature = x.flatten(1, 2)
        # 对特征向量进行位置编码
        feature = self.pos_encoder(feature)
        # 遍历所有编码器层，对特征进行编码
        for encoder_layer in self.encoder:
            feature = encoder_layer(feature)
        # 将特征向量重新调整为原始形状
        feature = feature.reshape([0, H, W, C]).transpose([0, 3, 1, 2])
        # 将特征向量输入解码器，获取视觉特征和注意力分数
        v_feature, attn_scores = self.decoder(feature)  # (B, N, C), (B, C, H, W)
        # 将视觉特征输入分类器，获取视觉特征的预测结果
        vis_logits = self.cls(v_feature)  # (B, N, C)
        logits = vis_logits
        # 获取视觉特征的长度信息
        vis_lengths = _get_length(vis_logits)
        # 如果使用语言模型
        if self.use_lang:
            # 初始化对齐的预测结果和长度信息
            align_logits = vis_logits
            align_lengths = vis_lengths
            all_l_res, all_a_res = [], []
            # 迭代多次进行对齐操作
            for i in range(self.iter_size):
                # 对对齐的预测结果进行 softmax 操作
                tokens = F.softmax(align_logits, axis=-1)
                lengths = align_lengths
                # 对长度信息进行裁剪
                lengths = paddle.clip(
                    lengths, 2, self.max_length)  # TODO:move to langauge model
                # 使用语言模型对 tokens 进行编码
                l_feature, l_logits = self.language(tokens, lengths)

                # 对齐操作
                all_l_res.append(l_logits)
                fuse = paddle.concat((l_feature, v_feature), -1)
                f_att = F.sigmoid(self.w_att_align(fuse))
                output = f_att * v_feature + (1 - f_att) * l_feature
                align_logits = self.cls_align(output)  # (B, N, C)

                align_lengths = _get_length(align_logits)
                all_a_res.append(align_logits)
            # 如果处于训练状态，返回对齐结果、语言结果和视觉结果
            if self.training:
                return {
                    'align': all_a_res,
                    'lang': all_l_res,
                    'vision': vis_logits
                }
            # 否则，更新 logits 为对齐结果
            else:
                logits = align_logits
        # 如果处于训练状态，返回 logits
        if self.training:
            return logits
        # 否则，对 logits 进行 softmax 操作并返回
        else:
            return F.softmax(logits, -1)
# 从logit中获取长度的贪婪解码器
def _get_length(logit):
    # 获取logit中最大值所在位置是否为0的布尔值数组
    out = (logit.argmax(-1) == 0)
    # 检查是否有任何True值存在
    abn = out.any(-1)
    # 将布尔数组转换为int32类型
    out_int = out.cast('int32')
    # 计算累积和，判断是否为1，并且为True的位置
    out = (out_int.cumsum(-1) == 1) & out
    # 将布尔数组转换为int32类型
    out = out.cast('int32')
    # 获取最大值所在位置
    out = out.argmax(-1)
    # 加1得到长度
    out = out + 1
    # 创建与out相同形状的全零张量，并赋值为logit的第二维度大小
    len_seq = paddle.zeros_like(out) + logit.shape[1]
    # 根据abn的值选择out或len_seq
    out = paddle.where(abn, out, len_seq)
    return out

# 生成一个用于序列的方形掩码。掩盖的位置填充为float('-inf')，未掩盖的位置填充为float(0.0)
def _get_mask(length, max_length):
    # 在最后一个维度上增加一个维度
    length = length.unsqueeze(-1)
    # 获取批次大小
    B = paddle.shape(length)[0]
    # 生成0到max_length的网格张量
    grid = paddle.arange(0, max_length).unsqueeze(0).tile([B, 1])
    # 创建全零张量
    zero_mask = paddle.zeros([B, max_length], dtype='float32')
    # 创建全-inf张量
    inf_mask = paddle.full([B, max_length], '-inf', dtype='float32')
    # 创建对角线为-inf的对角矩阵
    diag_mask = paddle.diag(
        paddle.full(
            [max_length], '-inf', dtype=paddle.float32),
        offset=0,
        name=None)
    # 根据grid和length的大小关系选择inf_mask或zero_mask
    mask = paddle.where(grid >= length, inf_mask, zero_mask)
    # 在第二个维度上增加一个维度，并加上对角线为-inf的对角矩阵
    mask = mask.unsqueeze(1) + diag_mask
    return mask.unsqueeze(1)
```