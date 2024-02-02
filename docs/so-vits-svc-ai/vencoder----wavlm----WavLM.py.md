# `so-vits-svc\vencoder\wavlm\WavLM.py`

```py
# 导入所需的库
import logging
import math
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm

# 导入自定义模块
from vencoder.wavlm.modules import (
    Fp32GroupNorm,
    Fp32LayerNorm,
    GLU_Linear,
    GradMultiply,
    MultiheadAttention,
    SamePad,
    TransposeLast,
    get_activation_fn,
    init_bert_params,
)

# 获取日志记录器
logger = logging.getLogger(__name__)

# 定义函数，用于计算随机掩码跨度
def compute_mask_indices(
    shape: Tuple[int, int],  # 输入形状的元组
    padding_mask: Optional[torch.Tensor],  # 可选的填充掩码张量
    mask_prob: float,  # 掩码概率
    mask_length: int,  # 掩码长度
    mask_type: str = "static",  # 掩码类型，默认为静态
    mask_other: float = 0.0,  # 其他掩码概率，默认为0.0
    min_masks: int = 0,  # 最小掩码数，默认为0
    no_overlap: bool = False,  # 是否不重叠，默认为False
    min_space: int = 0,  # 最小间隔，默认为0
) -> np.ndarray:  # 返回一个NumPy数组
    """
    Computes random mask spans for a given shape
    计算给定形状的随机掩码跨度
    """
    # 定义一个函数，用于计算给定形状的掩码
    # shape: 要计算掩码的形状，应该是大小为2的列表，第一个元素是批量大小，第二个是时间步长
    # padding_mask: 可选的与形状相同大小的填充掩码，用于防止掩盖填充元素
    # mask_prob: 每个标记被选择为掩盖的起始概率。这将乘以时间步长除以掩码跨度的长度，以掩盖大约这个百分比的所有元素。但是由于重叠，实际数量会更少（除非no_overlap为True）
    # mask_type: 如何计算掩码长度
    #   static = 固定大小
    #   uniform = 从均匀分布[mask_other, mask_length*2]中抽样
    #   normal = 从均值为mask_length，标准差为mask_other的正态分布中抽样。掩码至少包含1个元素
    #   poisson = 从lambda=mask length的泊松分布中抽样
    # min_masks: 掩码的最小数量
    # no_overlap: 如果为False，将切换到一个替代的递归算法，以防止跨度重叠
    # min_space: 仅在no_overlap为True时使用，这是在跨度之间保持未掩盖的元素数量
    
    bsz, all_sz = shape
    # 创建一个大小为(bsz, all_sz)的布尔数组，用于存储掩码
    mask = np.full((bsz, all_sz), False)
    
    # 计算所有掩码的数量，使用概率舍入
    all_num_mask = int(
        mask_prob * all_sz / float(mask_length)
        + np.random.rand()
    )
    
    # 确保掩码数量不低于最小数量
    all_num_mask = max(min_masks, all_num_mask)
    
    # 创建一个空列表，用于存储掩码的索引
    mask_idcs = []
    
    # 计算最小长度
    min_len = min([len(m) for m in mask_idcs])
    
    # 遍历掩码索引列表，将长度大于最小长度的掩码索引随机缩减到最小长度
    for i, mask_idc in enumerate(mask_idcs):
        if len(mask_idc) > min_len:
            mask_idc = np.random.choice(mask_idc, min_len, replace=False)
        # 将掩码索引对应的位置设置为True
        mask[i, mask_idc] = True
    
    # 返回计算得到的掩码
    return mask
# 定义 WavLMConfig 类，用于配置 WavLM 模型
class WavLMConfig:
    # 更新配置参数
    def update(self, cfg: dict):
        self.__dict__.update(cfg)

# 定义 WavLM 类，继承自 nn.Module
class WavLM(nn.Module):
    # 初始化方法
    def __init__(
        self,
        cfg: WavLMConfig,
    ) -> None:
        # 调用父类的初始化方法
        super().__init__()
        # 打印 WavLM 配置信息
        logger.info(f"WavLM Config: {cfg.__dict__}")

        # 将配置参数赋值给 self.cfg
        self.cfg = cfg
        # 解析配置中的卷积特征层
        feature_enc_layers = eval(cfg.conv_feature_layers)
        # 获取嵌入层的维度
        self.embed = feature_enc_layers[-1][0]

        # 创建特征提取模型
        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers=feature_enc_layers,
            dropout=0.0,
            mode=cfg.extractor_mode,
            conv_bias=cfg.conv_bias,
        )

        # 根据嵌入层的维度和配置中的编码器嵌入维度创建线性层
        self.post_extract_proj = (
            nn.Linear(self.embed, cfg.encoder_embed_dim)
            if self.embed != cfg.encoder_embed_dim
            else None
        )

        # 设置掩码概率和选择方式
        self.mask_prob = cfg.mask_prob
        self.mask_selection = cfg.mask_selection
        self.mask_other = cfg.mask_other
        self.mask_length = cfg.mask_length
        self.no_mask_overlap = cfg.no_mask_overlap
        self.mask_min_space = cfg.mask_min_space

        # 设置通道掩码概率和选择方式
        self.mask_channel_prob = cfg.mask_channel_prob
        self.mask_channel_selection = cfg.mask_channel_selection
        self.mask_channel_other = cfg.mask_channel_other
        self.mask_channel_length = cfg.mask_channel_length
        self.no_mask_channel_overlap = cfg.no_mask_channel_overlap
        self.mask_channel_min_space = cfg.mask_channel_min_space

        # 创建输入层和特征层的丢弃层
        self.dropout_input = nn.Dropout(cfg.dropout_input)
        self.dropout_features = nn.Dropout(cfg.dropout_features)

        # 设置特征梯度倍数
        self.feature_grad_mult = cfg.feature_grad_mult

        # 创建编码器的掩码嵌入层
        self.mask_emb = nn.Parameter(
            torch.FloatTensor(cfg.encoder_embed_dim).uniform_()
        )

        # 创建 Transformer 编码器
        self.encoder = TransformerEncoder(cfg)
        # 创建嵌入层的 LayerNorm 层
        self.layer_norm = LayerNorm(self.embed)
    # 对输入的张量 x 应用掩码，根据 padding_mask 进行填充
    def apply_mask(self, x, padding_mask):
        # 获取张量 x 的维度信息
        B, T, C = x.shape
        # 如果 mask_prob 大于 0，则计算掩码的索引并应用掩码
        if self.mask_prob > 0:
            mask_indices = compute_mask_indices(
                (B, T),
                padding_mask,
                self.mask_prob,
                self.mask_length,
                self.mask_selection,
                self.mask_other,
                min_masks=2,
                no_overlap=self.no_mask_overlap,
                min_space=self.mask_min_space,
            )
            mask_indices = torch.from_numpy(mask_indices).to(x.device)
            x[mask_indices] = self.mask_emb
        else:
            mask_indices = None

        # 如果 mask_channel_prob 大于 0，则计算通道掩码的索引并应用掩码
        if self.mask_channel_prob > 0:
            mask_channel_indices = compute_mask_indices(
                (B, C),
                None,
                self.mask_channel_prob,
                self.mask_channel_length,
                self.mask_channel_selection,
                self.mask_channel_other,
                no_overlap=self.no_mask_channel_overlap,
                min_space=self.mask_channel_min_space,
            )
            mask_channel_indices = (
                torch.from_numpy(mask_channel_indices)
                .to(x.device)
                .unsqueeze(1)
                .expand(-1, T, -1)
            )
            x[mask_channel_indices] = 0

        # 返回应用掩码后的张量 x 和掩码索引
        return x, mask_indices

    # 对填充掩码进行前向处理
    def forward_padding_mask(
            self, features: torch.Tensor, padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        # 计算额外的填充数量，并对 padding_mask 进行相应的处理
        extra = padding_mask.size(1) % features.size(1)
        if extra > 0:
            padding_mask = padding_mask[:, :-extra]
        # 将 padding_mask 重新调整形状，以匹配 features 的维度
        padding_mask = padding_mask.view(
            padding_mask.size(0), features.size(1), -1
        )
        # 对 padding_mask 进行逻辑与操作，得到最终的填充掩码
        padding_mask = padding_mask.all(-1)
        # 返回处理后的填充掩码
        return padding_mask
    # 定义提取特征的方法，接受输入源和填充掩码等参数
    def extract_features(
        self,
        source: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = False,
        ret_conv: bool = False,
        output_layer: Optional[int] = None,
        ret_layer_results: bool = False,
    ):

        # 如果特征梯度乘数大于0，则提取特征
        if self.feature_grad_mult > 0:
            features = self.feature_extractor(source)
            # 如果特征梯度乘数不等于1.0，则对特征进行梯度乘法操作
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            # 否则，使用torch.no_grad()上下文管理器，不计算梯度提取特征
            with torch.no_grad():
                features = self.feature_extractor(source)

        # 转置特征张量的维度
        features = features.transpose(1, 2)
        # 对特征进行层归一化
        features = self.layer_norm(features)

        # 如果存在填充掩码，则调用forward_padding_mask方法生成填充掩码
        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(features, padding_mask)

        # 如果存在后提取投影层，则对特征进行投影
        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        # 对特征进行输入层的dropout操作
        features = self.dropout_input(features)

        # 如果需要进行掩码操作，则调用apply_mask方法
        if mask:
            x, mask_indices = self.apply_mask(
                features, padding_mask
            )
        else:
            x = features

        # 对特征进行编码器操作，返回编码结果和层结果
        x, layer_results = self.encoder(
            x,
            padding_mask=padding_mask,
            layer=None if output_layer is None else output_layer - 1
        )

        # 将结果存储在字典中
        res = {"x": x, "padding_mask": padding_mask, "features": features, "layer_results": layer_results}

        # 如果需要返回卷积特征，则返回卷积特征，否则返回编码结果
        feature = res["features"] if ret_conv else res["x"]
        # 如果需要返回层结果，则将结果和层结果一起返回
        if ret_layer_results:
            feature = (feature, res["layer_results"])
        return feature, res["padding_mask"]
class ConvFeatureExtractionModel(nn.Module):
    def __init__(
            self,
            conv_layers: List[Tuple[int, int, int]],
            dropout: float = 0.0,
            mode: str = "default",
            conv_bias: bool = False,
            conv_type: str = "default"
    # 初始化函数，定义了卷积特征提取模型的结构和参数
    def forward(self, x, mask=None):

        # BxT -> BxCxT
        # 将输入张量 x 在第一维度上增加一个维度，从 BxT 变为 BxCxT
        x = x.unsqueeze(1)
        # 如果卷积类型为自定义
        if self.conv_type == "custom":
            # 遍历卷积层列表
            for conv in self.conv_layers:
                # 如果当前卷积层是 LayerNorm
                if isinstance(conv, nn.LayerNorm):
                    # 将 x 在第1和第2维度上进行转置
                    x = x.transpose(1, 2)
                    # 对 x 进行 LayerNorm 操作，并将结果再次转置
                    x = conv(x).transpose(1, 2)
                else:
                    # 对 x 进行卷积操作
                    x = conv(x)
            # 将 x 在第2和第3维度上进行转置，并保证内存连续性
            x = x.transpose(2, 3).contiguous()
            # 将 x 进行形状变换，保留第1维度，将其余维度展平
            x = x.view(x.size(0), -1, x.size(-1))
        else:
            # 如果卷积类型不是自定义
            for conv in self.conv_layers:
                # 对 x 进行卷积操作
                x = conv(x)
            # 如果卷积类型是 conv2d
            if self.conv_type == "conv2d":
                # 获取 x 的形状信息
                b, c, t, f = x.size()
                # 将 x 在第2和第3维度上进行转置，并保证内存连续性，然后展平
                x = x.transpose(2, 3).contiguous().view(b, c * f, t)
        # 返回处理后的 x
        return x


class TransformerEncoder(nn.Module):
    # 定义了 Transformer 编码器的前向传播函数
    def forward(self, x, padding_mask=None, streaming_mask=None, layer=None):
        # 调用 extract_features 函数提取特征
        x, layer_results = self.extract_features(x, padding_mask, streaming_mask, layer)

        # 如果 layer_norm_first 为真且未指定层编号
        if self.layer_norm_first and layer is None:
            # 对 x 进行 LayerNorm 操作
            x = self.layer_norm(x)

        # 返回处理后的 x 和层结果
        return x, layer_results
    # 定义一个方法，用于提取特征
    def extract_features(self, x, padding_mask=None, streaming_mask=None, tgt_layer=None):

        # 如果存在填充掩码，则将对应位置的值设为0
        if padding_mask is not None:
            x[padding_mask] = 0

        # 对输入进行位置卷积
        x_conv = self.pos_conv(x.transpose(1, 2))
        x_conv = x_conv.transpose(1, 2)
        # 将位置卷积的结果与原始输入相加
        x = x + x_conv

        # 如果不是首次进行层归一化，则进行层归一化操作
        if not self.layer_norm_first:
            x = self.layer_norm(x)

        # 对输入进行dropout操作
        x = F.dropout(x, p=self.dropout, training=self.training)

        # 将输入的维度顺序从 B x T x C 转换为 T x B x C
        x = x.transpose(0, 1)

        # 初始化一个空列表用于存储每个层的结果
        layer_results = []
        z = None
        # 如果指定了目标层，则将当前层的结果添加到layer_results中
        if tgt_layer is not None:
            layer_results.append((x, z))
        r = None
        pos_bias = None
        # 遍历每一层进行处理
        for i, layer in enumerate(self.layers):
            # 生成一个随机概率用于dropout
            dropout_probability = np.random.random()
            # 如果不是训练状态或者随机概率大于layerdrop，则进行当前层的处理
            if not self.training or (dropout_probability > self.layerdrop):
                x, z, pos_bias = layer(x, self_attn_padding_mask=padding_mask, need_weights=False,
                                       self_attn_mask=streaming_mask, pos_bias=pos_bias)
            # 如果指定了目标层，则将当前层的结果添加到layer_results中
            if tgt_layer is not None:
                layer_results.append((x, z))
            # 如果当前层是目标层，则将结果赋值给r，并跳出循环
            if i == tgt_layer:
                r = x
                break

        # 如果r不为空，则将x赋值为r
        if r is not None:
            x = r

        # 将输入的维度顺序从 T x B x C 转换为 B x T x C
        x = x.transpose(0, 1)

        # 返回处理后的结果和每个层的结果
        return x, layer_results
class TransformerSentenceEncoderLayer(nn.Module):
    """
    实现了用于BERT/XLM风格预训练模型中的Transformer编码器层。
    """

    def __init__(
            self,
            embedding_dim: float = 768,  # 嵌入维度，默认为768
            ffn_embedding_dim: float = 3072,  # 前馈神经网络嵌入维度，默认为3072
            num_attention_heads: float = 8,  # 注意力头的数量，默认为8
            dropout: float = 0.1,  # 丢弃概率，默认为0.1
            attention_dropout: float = 0.1,  # 注意力丢弃概率，默认为0.1
            activation_dropout: float = 0.1,  # 激活函数丢弃概率，默认为0.1
            activation_fn: str = "relu",  # 激活函数类型，默认为"relu"
            layer_norm_first: bool = False,  # 是否先进行层归一化，默认为False
            has_relative_attention_bias: bool = False,  # 是否具有相对注意力偏置，默认为False
            num_buckets: int = 0,  # 桶的数量，默认为0
            max_distance: int = 0,  # 最大距离，默认为0
            rescale_init: bool = False,  # 是否重新缩放初始化，默认为False
            gru_rel_pos: bool = False,  # 是否使用GRU相对位置编码，默认为False
    # 定义一个初始化函数，设置模型的参数
    def __init__(self, embedding_dim: int, dropout: float, activation_dropout: float, activation_fn: str, num_attention_heads: int, attention_dropout: float, has_relative_attention_bias: bool, num_buckets: int, max_distance: int, rescale_init: float, gru_rel_pos: bool, ffn_embedding_dim: int) -> None:
        # 调用父类的初始化函数
        super().__init__()
        # 初始化参数
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout

        # 初始化模块
        self.activation_name = activation_fn
        self.activation_fn = get_activation_fn(activation_fn)
        # 初始化多头注意力机制
        self.self_attn = MultiheadAttention(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            self_attention=True,
            has_relative_attention_bias=has_relative_attention_bias,
            num_buckets=num_buckets,
            max_distance=max_distance,
            rescale_init=rescale_init,
            gru_rel_pos=gru_rel_pos,
        )

        # 初始化三个不同的 dropout 模块
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(self.activation_dropout)
        self.dropout3 = nn.Dropout(dropout)

        # 是否在第一层使用 layer normalization
        self.layer_norm_first = layer_norm_first

        # 与自注意力层相关联的 layer normalization
        self.self_attn_layer_norm = LayerNorm(self.embedding_dim)

        # 如果激活函数为 "glu"，则使用 GLU_Linear 模块，否则使用普通的线性层
        if self.activation_name == "glu":
            self.fc1 = GLU_Linear(self.embedding_dim, ffn_embedding_dim, "swish")
        else:
            self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)

        # 与位置前馈神经网络相关联的 layer normalization
        self.final_layer_norm = LayerNorm(self.embedding_dim)

    # 前向传播函数
    def forward(
            self,
            x: torch.Tensor,
            self_attn_mask: torch.Tensor = None,
            self_attn_padding_mask: torch.Tensor = None,
            need_weights: bool = False,
            pos_bias=None
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer imlementation.
        """
        residual = x

        if self.layer_norm_first:
            # 如果先进行 LayerNorm，则对输入进行 self-attention 层的 LayerNorm
            x = self.self_attn_layer_norm(x)
            # 调用 self-attention 模块，计算注意力权重和输出，同时传入位置偏置
            x, attn, pos_bias = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=False,
                attn_mask=self_attn_mask,
                position_bias=pos_bias
            )
            # 对输出进行 dropout
            x = self.dropout1(x)
            # 将输出与残差连接
            x = residual + x

            residual = x
            # 对输出进行最终的 LayerNorm
            x = self.final_layer_norm(x)
            # 根据激活函数的类型进行不同的处理
            if self.activation_name == "glu":
                x = self.fc1(x)
            else:
                x = self.activation_fn(self.fc1(x))
            # 对输出进行 dropout
            x = self.dropout2(x)
            # 经过全连接层
            x = self.fc2(x)
            # 对输出进行 dropout
            x = self.dropout3(x)
            # 将输出与残差连接
            x = residual + x
        else:
            # 如果先进行 self-attention，则直接调用 self-attention 模块，计算注意力权重和输出，同时传入位置偏置
            x, attn, pos_bias = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=need_weights,
                attn_mask=self_attn_mask,
                position_bias=pos_bias
            )

            # 对输出进行 dropout
            x = self.dropout1(x)
            # 将输出与残差连接
            x = residual + x

            # 对连接后的输出进行 LayerNorm
            x = self.self_attn_layer_norm(x)

            residual = x
            # 根据激活函数的类型进行不同的处理
            if self.activation_name == "glu":
                x = self.fc1(x)
            else:
                x = self.activation_fn(self.fc1(x))
            # 对输出进行 dropout
            x = self.dropout2(x)
            # 经过全连接层
            x = self.fc2(x)
            # 对输出进行 dropout
            x = self.dropout3(x)
            # 将输出与残差连接
            x = residual + x
            # 对连接后的输出进行最终的 LayerNorm
            x = self.final_layer_norm(x)

        return x, attn, pos_bias
```