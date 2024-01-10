# `so-vits-svc\vencoder\dphubert\model.py`

```
"""Speech SSL models supporting pruning.

Originally from:
https://github.com/pytorch/audio/blob/main/torchaudio/models/wav2vec2/model.py

"""

import math  # 导入数学库
from typing import List, Optional, Tuple  # 导入类型提示相关的库

import torch  # 导入 PyTorch 库
import torch.nn.functional as F  # 导入 PyTorch 中的函数模块
from torch import Tensor  # 从 PyTorch 中导入张量
from torch.nn import Module  # 从 PyTorch 中导入神经网络模块

from . import components  # 从当前目录中导入 components 模块


class Wav2Vec2Model(Module):
    """Acoustic model used in *wav2vec 2.0* :cite:`baevski2020wav2vec`.

    Note:
        To build the model, please use one of the factory functions.
        :py:func:`wav2vec2_model`, :py:func:`wav2vec2_base`, :py:func:`wav2vec2_large`,
        :py:func:`wav2vec2_large_lv60k`, :py:func:`hubert_base`, :py:func:`hubert_large`,
        and :py:func:`hubert_xlarge`.

    See Also:
        * :class:`torchaudio.pipelines.Wav2Vec2Bundle`: Pretrained models (without fine-tuning)
        * :class:`torchaudio.pipelines.Wav2Vec2ASRBundle`: ASR pipelines with pretrained models.

    Args:
        feature_extractor (torch.nn.Module):
            Feature extractor that extracts feature vectors from raw audio Tensor.

        encoder (torch.nn.Module):
            Encoder that converts the audio features into the sequence of probability
            distribution (in negative log-likelihood) over labels.

        aux (torch.nn.Module or None, optional):
            Auxiliary module. If provided, the output from encoder is passed to this module.
    """  # noqa: E501

    def __init__(
        self,
        normalize_waveform: bool,
        feature_extractor: Module,
        encoder: Module,
        aux: Optional[Module] = None,
    ):
        super().__init__()  # 调用父类的构造函数
        self.normalize_waveform = normalize_waveform  # 初始化属性 normalize_waveform
        self.feature_extractor = feature_extractor  # 初始化属性 feature_extractor
        self.encoder = encoder  # 初始化属性 encoder
        self.aux = aux  # 初始化属性 aux

    @torch.jit.export
    def extract_features(
        self,
        waveforms: Tensor,
        lengths: Optional[Tensor] = None,
        num_layers: Optional[int] = None,
    # 计算当前模型的参数数量
    def get_num_params(self):
        """Calculate the current size."""
        # 获取特征提取器和编码器的参数数量和最终输出通道数
        feature_extractor_size, encoder_in_features = self.feature_extractor.get_num_params_and_final_out_channels()
        # 获取编码器的参数数量
        encoder_size = self.encoder.get_num_params(encoder_in_features)
        # 返回特征提取器和编码器参数数量之和
        return feature_extractor_size + encoder_size
    
    # 对模型进行剪枝
    def prune(self):
        # 将模型设置为评估模式
        self.eval()     # must be in eval mode
        # 对特征提取器进行剪枝，返回剪枝后的卷积配置和卷积输出索引
        conv_config, conv_out_index = self.feature_extractor.prune()    # [(output_channel, kernel_size, stride), ...]
        # 对编码器进行剪枝，返回变换器配置
        transformer_config = self.encoder.prune(conv_out_index)     # NOTE: this is a defaultdict(list)
        # 获取是否使用注意力机制
        use_attention = transformer_config["use_attention"]
        # 获取是否使用前馈神经网络
        use_feed_forward = transformer_config["use_feed_forward"]
        # 获取注意力头的数量
        num_heads = transformer_config["num_heads"]     # can be []
        # 获取剩余的注意力头数量
        remaining_heads = transformer_config["remaining_heads"]     # can be []
        # 获取前馈神经网络中间特征的数量
        ff_interm_features = transformer_config["ff_interm_features"]

        # 返回卷积配置、是否使用注意力机制、是否使用前馈神经网络、注意力头数量、剩余的注意力头数量、前馈神经网络中间特征数量
        return conv_config, use_attention, use_feed_forward, num_heads, remaining_heads, ff_interm_features

    # 前向传播函数
    def forward(
        self,
        waveforms: Tensor,
        lengths: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Compute the sequence of probability distribution over labels.

        Args:
            waveforms (Tensor): Audio tensor of shape `(batch, frames)`.
            lengths (Tensor or None, optional):
                Indicates the valid length of each audio in the batch.
                Shape: `(batch, )`.
                When the ``waveforms`` contains audios with different durations,
                by providing ``lengths`` argument, the model will compute
                the corresponding valid output lengths and apply proper mask in
                transformer attention layer.
                If ``None``, it is assumed that all the audio in ``waveforms``
                have valid length. Default: ``None``.

        Returns:
            (Tensor, Optional[Tensor]):
            Tensor
                The sequences of probability distribution (in logit) over labels.
                Shape: `(batch, frames, num labels)`.
            Tensor or None
                If ``lengths`` argument was provided, a Tensor of shape `(batch, )`
                is returned.
                It indicates the valid length in time axis of the output Tensor.
        """
        # 如果需要对音频数据进行归一化处理
        if self.normalize_waveform:
            # 如果提供了长度信息
            if lengths is not None:
                # 对每个音频数据进行 layer normalization，并进行填充
                waveforms = [
                    F.layer_norm(wave[:length], (length,)) for wave, length in zip(waveforms, lengths)
                ]
                waveforms = torch.nn.utils.rnn.pad_sequence(waveforms, batch_first=True)
            # 如果未提供长度信息
            else:
                # 对整个音频数据进行 layer normalization
                waveforms = F.layer_norm(waveforms, waveforms.shape[-1:])

        # 提取特征
        x, lengths = self.feature_extractor(waveforms, lengths)
        # 编码特征
        x = self.encoder(x, lengths)
        # 如果存在辅助模块，则对特征进行处理
        if self.aux is not None:
            x = self.aux(x)
        # 返回处理后的特征及长度信息
        return x, lengths
# 定义一个函数，用于创建 Wav2Vec2Model 对象，根据传入的参数不同，可能会返回不同的对象
def wav2vec2_model(**configs) -> Wav2Vec2Model:
    """Wraps the original wav2vec2_model and wavlm_model."""

    # 如果传入的参数中包含 "encoder_remaining_heads"，则调用 wavlm_model 函数
    if "encoder_remaining_heads" in configs:
        return wavlm_model(**configs)
    
    # 否则调用 wav2vec2_model_original 函数
    return wav2vec2_model_original(**configs)


# 定义一个函数，用于创建原始的 Wav2Vec2Model 对象
def wav2vec2_model_original(
    extractor_mode: str,
    extractor_conv_layer_config: Optional[List[Tuple[int, int, int]]],
    extractor_conv_bias: bool,
    encoder_embed_dim: int,
    encoder_projection_dropout: float,
    encoder_pos_conv_kernel: int,
    encoder_pos_conv_groups: int,
    encoder_num_layers: int,
    encoder_use_attention: List[bool],
    encoder_use_feed_forward: List[bool],
    encoder_num_heads: List[int],
    encoder_head_dim: int,
    encoder_attention_dropout: float,
    encoder_ff_interm_features: List[int],
    encoder_ff_interm_dropout: float,
    encoder_dropout: float,
    encoder_layer_norm_first: bool,
    encoder_layer_drop: float,
    aux_num_out: Optional[int],
    normalize_waveform: bool,
    extractor_prune_conv_channels: bool = False,
    encoder_prune_attention_heads: bool = False,
    encoder_prune_attention_layer: bool = False,
    encoder_prune_feed_forward_intermediate: bool = False,
    encoder_prune_feed_forward_layer: bool = False,
) -> Wav2Vec2Model:
    """Builds custom :class:`~torchaudio.models.Wav2Vec2Model`.

    Note:
        The "feature extractor" below corresponds to
        `ConvFeatureExtractionModel <https://github.com/pytorch/fairseq/blob/dd3bd3c0497ae9a7ae7364404a6b0a4c501780b3/fairseq/models/wav2vec/wav2vec2.py#L736>`__
        in the original ``fairseq`` implementation.
        This is referred as "(convolutional) feature encoder" in the *wav2vec 2.0*
        :cite:`baevski2020wav2vec` paper.

        The "encoder" below corresponds to `TransformerEncoder <https://github.com/pytorch/fairseq/blob/dd3bd3c0497ae9a7ae7364404a6b0a4c501780b3/fairseq/models/wav2vec/wav2vec2.py#L817>`__,
        and this is referred as "Transformer" in the paper.
    Returns:
        Wav2Vec2Model:
            The resulting model.
    """  # noqa: E501
    # 如果未提供特征提取器的卷积层配置，则使用默认配置
    if extractor_conv_layer_config is None:
        extractor_conv_layer_config = [(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] * 2

    # 获取特征提取器
    feature_extractor = components._get_feature_extractor(
        extractor_mode, extractor_conv_layer_config, extractor_conv_bias, 
        prune_conv_channels=extractor_prune_conv_channels,
    )
    # 获取编码器
    encoder = components._get_encoder(
        in_features=extractor_conv_layer_config[-1][0],
        embed_dim=encoder_embed_dim,
        dropout_input=encoder_projection_dropout,
        pos_conv_kernel=encoder_pos_conv_kernel,
        pos_conv_groups=encoder_pos_conv_groups,
        num_layers=encoder_num_layers,
        use_attention=encoder_use_attention,
        use_feed_forward=encoder_use_feed_forward,
        num_heads=encoder_num_heads,
        head_dim=encoder_head_dim,
        attention_dropout=encoder_attention_dropout,
        ff_interm_features=encoder_ff_interm_features,
        ff_interm_dropout=encoder_ff_interm_dropout,
        dropout=encoder_dropout,
        layer_norm_first=encoder_layer_norm_first,
        layer_drop=encoder_layer_drop,
        prune_attention_heads=encoder_prune_attention_heads,
        prune_attention_layer=encoder_prune_attention_layer,
        prune_feed_forward_intermediate=encoder_prune_feed_forward_intermediate,
        prune_feed_forward_layer=encoder_prune_feed_forward_layer,
    )
    aux = None
    # 如果提供了辅助输出的输出维度，则创建辅助输出层
    if aux_num_out is not None:
        aux = torch.nn.Linear(in_features=encoder_embed_dim, out_features=aux_num_out)
    # 返回 Wav2Vec2Model 模型
    return Wav2Vec2Model(normalize_waveform, feature_extractor, encoder, aux)
def wav2vec2_base(
    encoder_projection_dropout: float = 0.1,  # 设置编码器投影层的丢弃率
    encoder_attention_dropout: float = 0.1,  # 设置编码器注意力层的丢弃率
    encoder_ff_interm_dropout: float = 0.1,  # 设置编码器前馈中间层的丢弃率
    encoder_dropout: float = 0.1,  # 设置编码器的丢弃率
    encoder_layer_drop: float = 0.1,  # 设置编码器层的丢弃率
    aux_num_out: Optional[int] = None,  # 辅助输出的数量，可选参数
    extractor_prune_conv_channels: bool = False,  # 是否修剪提取器的卷积通道
    encoder_prune_attention_heads: bool = False,  # 是否修剪编码器的注意力头
    encoder_prune_attention_layer: bool = False,  # 是否修剪编码器的注意力层
    encoder_prune_feed_forward_intermediate: bool = False,  # 是否修剪编码器的前馈中间层
    encoder_prune_feed_forward_layer: bool = False,  # 是否修剪编码器的前馈层
) -> Wav2Vec2Model:  # 函数返回类型为 Wav2Vec2Model
    """Builds "base" :class:`~torchaudio.models.Wav2Vec2Model` from *wav2vec 2.0* :cite:`baevski2020wav2vec`

    Args:
        encoder_projection_dropout (float):
            See :py:func:`wav2vec2_model`.  # 查看 wav2vec2_model 函数以了解更多信息
        encoder_attention_dropout (float):
            See :py:func:`wav2vec2_model`.  # 查看 wav2vec2_model 函数以了解更多信息
        encoder_ff_interm_dropout (float):
            See :py:func:`wav2vec2_model`.  # 查看 wav2vec2_model 函数以了解更多信息
        encoder_dropout (float):
            See :py:func:`wav2vec2_model`.  # 查看 wav2vec2_model 函数以了解更多信息
        encoder_layer_drop (float):
            See :py:func:`wav2vec2_model`.  # 查看 wav2vec2_model 函数以了解更多信息
        aux_num_out (int or None, optional):
            See :py:func:`wav2vec2_model`.  # 查看 wav2vec2_model 函数以了解更多信息

    Returns:
        Wav2Vec2Model:
            The resulting model.  # 返回构建的模型
    """  # noqa: E501
    # 返回一个 wav2vec2 模型
    return wav2vec2_model(
        # 提取器模式为 group_norm
        extractor_mode="group_norm",
        # 提取器卷积层配置为 None
        extractor_conv_layer_config=None,
        # 提取器卷积层偏置为 False
        extractor_conv_bias=False,
        # 编码器嵌入维度为 768
        encoder_embed_dim=768,
        # 编码器投影丢弃率为 encoder_projection_dropout
        encoder_projection_dropout=encoder_projection_dropout,
        # 编码器位置卷积核为 128
        encoder_pos_conv_kernel=128,
        # 编码器位置卷积分组为 16
        encoder_pos_conv_groups=16,
        # 编码器层数为 12
        encoder_num_layers=12,
        # 编码器注意力头数为 12
        encoder_num_heads=12,
        # 编码器注意力丢弃率为 encoder_attention_dropout
        encoder_attention_dropout=encoder_attention_dropout,
        # 编码器前馈中间特征为 3072
        encoder_ff_interm_features=3072,
        # 编码器前馈中间丢弃率为 encoder_ff_interm_dropout
        encoder_ff_interm_dropout=encoder_ff_interm_dropout,
        # 编码器丢弃率为 encoder_dropout
        encoder_dropout=encoder_dropout,
        # 编码器层标准化是否在首位为 False
        encoder_layer_norm_first=False,
        # 编码器层丢弃率为 encoder_layer_drop
        encoder_layer_drop=encoder_layer_drop,
        # 辅助输出数量为 aux_num_out
        aux_num_out=aux_num_out,
        # 提取器修剪卷积通道为 extractor_prune_conv_channels
        extractor_prune_conv_channels=extractor_prune_conv_channels,
        # 编码器修剪注意力头数为 encoder_prune_attention_heads
        encoder_prune_attention_heads=encoder_prune_attention_heads,
        # 编码器修剪注意力层为 encoder_prune_attention_layer
        encoder_prune_attention_layer=encoder_prune_attention_layer,
        # 编码器修剪前馈中间特征为 encoder_prune_feed_forward_intermediate
        encoder_prune_feed_forward_intermediate=encoder_prune_feed_forward_intermediate,
        # 编码器修剪前馈层为 encoder_prune_feed_forward_layer
        encoder_prune_feed_forward_layer=encoder_prune_feed_forward_layer,
    )
def wav2vec2_large(
    encoder_projection_dropout: float = 0.1,  # 设置编码器投影层的丢弃率
    encoder_attention_dropout: float = 0.1,  # 设置编码器注意力层的丢弃率
    encoder_ff_interm_dropout: float = 0.1,  # 设置编码器前馈中间层的丢弃率
    encoder_dropout: float = 0.1,  # 设置编码器的丢弃率
    encoder_layer_drop: float = 0.1,  # 设置编码器层的丢弃率
    aux_num_out: Optional[int] = None,  # 辅助输出的数量，可选参数
    extractor_prune_conv_channels: bool = False,  # 是否修剪提取器的卷积通道
    encoder_prune_attention_heads: bool = False,  # 是否修剪编码器的注意力头
    encoder_prune_attention_layer: bool = False,  # 是否修剪编码器的注意力层
    encoder_prune_feed_forward_intermediate: bool = False,  # 是否修剪编码器的前馈中间层
    encoder_prune_feed_forward_layer: bool = False,  # 是否修剪编码器的前馈层
) -> Wav2Vec2Model:  # 函数返回类型为 Wav2Vec2Model
    """Builds "large" :class:`~torchaudio.models.Wav2Vec2Model` from *wav2vec 2.0* :cite:`baevski2020wav2vec`

    Args:
        encoder_projection_dropout (float):
            See :py:func:`wav2vec2_model`.  # 查看 wav2vec2_model 函数以了解更多信息
        encoder_attention_dropout (float):
            See :py:func:`wav2vec2_model`.  # 查看 wav2vec2_model 函数以了解更多信息
        encoder_ff_interm_dropout (float):
            See :py:func:`wav2vec2_model`.  # 查看 wav2vec2_model 函数以了解更多信息
        encoder_dropout (float):
            See :py:func:`wav2vec2_model`.  # 查看 wav2vec2_model 函数以了解更多信息
        encoder_layer_drop (float):
            See :py:func:`wav2vec2_model`.  # 查看 wav2vec2_model 函数以了解更多信息
        aux_num_out (int or None, optional):
            See :py:func:`wav2vec2_model`.  # 查看 wav2vec2_model 函数以了解更多信息

    Returns:
        Wav2Vec2Model:
            The resulting model.  # 返回构建的模型
    """  # noqa: E501
    # 返回一个 wav2vec2 模型
    return wav2vec2_model(
        # 提取器模式为 group_norm
        extractor_mode="group_norm",
        # 提取器卷积层配置为 None
        extractor_conv_layer_config=None,
        # 提取器卷积层偏置为 False
        extractor_conv_bias=False,
        # 编码器嵌入维度为 1024
        encoder_embed_dim=1024,
        # 编码器投影丢弃率为 encoder_projection_dropout
        encoder_projection_dropout=encoder_projection_dropout,
        # 编码器位置卷积核为 128
        encoder_pos_conv_kernel=128,
        # 编码器位置卷积分组为 16
        encoder_pos_conv_groups=16,
        # 编码器层数为 24
        encoder_num_layers=24,
        # 编码器注意力头数为 16
        encoder_num_heads=16,
        # 编码器注意力丢弃率为 encoder_attention_dropout
        encoder_attention_dropout=encoder_attention_dropout,
        # 编码器前馈中间特征为 4096
        encoder_ff_interm_features=4096,
        # 编码器前馈中间丢弃率为 encoder_ff_interm_dropout
        encoder_ff_interm_dropout=encoder_ff_interm_dropout,
        # 编码器丢弃率为 encoder_dropout
        encoder_dropout=encoder_dropout,
        # 编码器层标准化是否在首位为 False
        encoder_layer_norm_first=False,
        # 编码器层丢弃率为 encoder_layer_drop
        encoder_layer_drop=encoder_layer_drop,
        # 辅助输出数量为 aux_num_out
        aux_num_out=aux_num_out,
        # 提取器修剪卷积通道为 extractor_prune_conv_channels
        extractor_prune_conv_channels=extractor_prune_conv_channels,
        # 编码器修剪注意力头数为 encoder_prune_attention_heads
        encoder_prune_attention_heads=encoder_prune_attention_heads,
        # 编码器修剪注意力层为 encoder_prune_attention_layer
        encoder_prune_attention_layer=encoder_prune_attention_layer,
        # 编码器修剪前馈中间特征为 encoder_prune_feed_forward_intermediate
        encoder_prune_feed_forward_intermediate=encoder_prune_feed_forward_intermediate,
        # 编码器修剪前馈层为 encoder_prune_feed_forward_layer
        encoder_prune_feed_forward_layer=encoder_prune_feed_forward_layer,
    )
def wav2vec2_large_lv60k(
    encoder_projection_dropout: float = 0.1,  # 设置编码器投影层的丢弃率
    encoder_attention_dropout: float = 0.0,  # 设置编码器注意力层的丢弃率
    encoder_ff_interm_dropout: float = 0.1,  # 设置编码器前馈中间层的丢弃率
    encoder_dropout: float = 0.0,  # 设置编码器的丢弃率
    encoder_layer_drop: float = 0.1,  # 设置编码器层的丢弃率
    aux_num_out: Optional[int] = None,  # 辅助输出的数量，可选参数
    extractor_prune_conv_channels: bool = False,  # 是否修剪提取器的卷积通道
    encoder_prune_attention_heads: bool = False,  # 是否修剪编码器的注意力头
    encoder_prune_attention_layer: bool = False,  # 是否修剪编码器的注意力层
    encoder_prune_feed_forward_intermediate: bool = False,  # 是否修剪编码器的前馈中间层
    encoder_prune_feed_forward_layer: bool = False,  # 是否修剪编码器的前馈层
) -> Wav2Vec2Model:  # 函数返回类型为 Wav2Vec2Model
    """Builds "large lv-60k" :class:`~torchaudio.models.Wav2Vec2Model` from *wav2vec 2.0* :cite:`baevski2020wav2vec`

    Args:
        encoder_projection_dropout (float):
            See :py:func:`wav2vec2_model`.  # 查看 wav2vec2_model 函数的文档以了解更多信息
        encoder_attention_dropout (float):
            See :py:func:`wav2vec2_model`.  # 查看 wav2vec2_model 函数的文档以了解更多信息
        encoder_ff_interm_dropout (float):
            See :py:func:`wav2vec2_model`.  # 查看 wav2vec2_model 函数的文档以了解更多信息
        encoder_dropout (float):
            See :py:func:`wav2vec2_model`.  # 查看 wav2vec2_model 函数的文档以了解更多信息
        encoder_layer_drop (float):
            See :py:func:`wav2vec2_model`.  # 查看 wav2vec2_model 函数的文档以了解更多信息
        aux_num_out (int or None, optional):
            See :py:func:`wav2vec2_model`.  # 查看 wav2vec2_model 函数的文档以了解更多信息

    Returns:
        Wav2Vec2Model:
            The resulting model.  # 返回构建的模型
    """  # noqa: E501
    # 返回一个 wav2vec2 模型
    return wav2vec2_model(
        # 提取器模式为"layer_norm"
        extractor_mode="layer_norm",
        # 提取器卷积层配置为 None
        extractor_conv_layer_config=None,
        # 提取器卷积层偏置为 True
        extractor_conv_bias=True,
        # 编码器嵌入维度为 1024
        encoder_embed_dim=1024,
        # 编码器投影丢弃率为 encoder_projection_dropout
        encoder_projection_dropout=encoder_projection_dropout,
        # 编码器位置卷积核为 128
        encoder_pos_conv_kernel=128,
        # 编码器位置卷积分组为 16
        encoder_pos_conv_groups=16,
        # 编码器层数为 24
        encoder_num_layers=24,
        # 编码器注意力头数为 16
        encoder_num_heads=16,
        # 编码器注意力丢弃率为 encoder_attention_dropout
        encoder_attention_dropout=encoder_attention_dropout,
        # 编码器前馈中间特征为 4096
        encoder_ff_interm_features=4096,
        # 编码器前馈中间丢弃率为 encoder_ff_interm_dropout
        encoder_ff_interm_dropout=encoder_ff_interm_dropout,
        # 编码器丢弃率为 encoder_dropout
        encoder_dropout=encoder_dropout,
        # 编码器层归一化在前
        encoder_layer_norm_first=True,
        # 编码器层丢弃率为 encoder_layer_drop
        encoder_layer_drop=encoder_layer_drop,
        # 辅助输出数量为 aux_num_out
        aux_num_out=aux_num_out,
        # 提取器修剪卷积通道为 extractor_prune_conv_channels
        extractor_prune_conv_channels=extractor_prune_conv_channels,
        # 编码器修剪注意力头数为 encoder_prune_attention_heads
        encoder_prune_attention_heads=encoder_prune_attention_heads,
        # 编码器修剪注意力层为 encoder_prune_attention_layer
        encoder_prune_attention_layer=encoder_prune_attention_layer,
        # 编码器修剪前馈中间特征为 encoder_prune_feed_forward_intermediate
        encoder_prune_feed_forward_intermediate=encoder_prune_feed_forward_intermediate,
        # 编码器修剪前馈层为 encoder_prune_feed_forward_layer
        encoder_prune_feed_forward_layer=encoder_prune_feed_forward_layer,
    )
def hubert_base(
    encoder_projection_dropout: float = 0.1,  # 设置编码器投影层的dropout比例
    encoder_attention_dropout: float = 0.1,  # 设置编码器注意力层的dropout比例
    encoder_ff_interm_dropout: float = 0.0,  # 设置编码器前馈中间层的dropout比例
    encoder_dropout: float = 0.1,  # 设置编码器的dropout比例
    encoder_layer_drop: float = 0.05,  # 设置编码器层的dropout比例
    aux_num_out: Optional[int] = None,  # 辅助输出的数量，可选参数
    extractor_prune_conv_channels: bool = False,  # 是否对提取器卷积通道进行修剪
    encoder_prune_attention_heads: bool = False,  # 是否对编码器注意力头进行修剪
    encoder_prune_attention_layer: bool = False,  # 是否对编码器注意力层进行修剪
    encoder_prune_feed_forward_intermediate: bool = False,  # 是否对编码器前馈中间层进行修剪
    encoder_prune_feed_forward_layer: bool = False,  # 是否对编码器前馈层进行修剪
) -> Wav2Vec2Model:  # 返回类型为Wav2Vec2Model
    """Builds "base" :class:`HuBERT <torchaudio.models.Wav2Vec2Model>` from *HuBERT* :cite:`hsu2021hubert`

    Args:
        encoder_projection_dropout (float):  # 编码器投影层的dropout比例
            See :py:func:`wav2vec2_model`.  # 参考wav2vec2_model函数
        encoder_attention_dropout (float):  # 编码器注意力层的dropout比例
            See :py:func:`wav2vec2_model`.  # 参考wav2vec2_model函数
        encoder_ff_interm_dropout (float):  # 编码器前馈中间层的dropout比例
            See :py:func:`wav2vec2_model`.  # 参考wav2vec2_model函数
        encoder_dropout (float):  # 编码器的dropout比例
            See :py:func:`wav2vec2_model`.  # 参考wav2vec2_model函数
        encoder_layer_drop (float):  # 编码器层的dropout比例
            See :py:func:`wav2vec2_model`.  # 参考wav2vec2_model函数
        aux_num_out (int or None, optional):  # 辅助输出的数量，可选参数
            See :py:func:`wav2vec2_model`.  # 参考wav2vec2_model函数

    Returns:
        Wav2Vec2Model:  # 返回类型为Wav2Vec2Model
            The resulting model.  # 返回生成的模型
    """  # noqa: E501
    # 返回一个 wav2vec2 模型
    return wav2vec2_model(
        # 使用 group_norm 作为特征提取器的模式
        extractor_mode="group_norm",
        # 不使用特定的卷积层配置
        extractor_conv_layer_config=None,
        # 不使用特征提取器的卷积层偏置
        extractor_conv_bias=False,
        # 编码器的嵌入维度为 768
        encoder_embed_dim=768,
        # 编码器投影层的丢弃率
        encoder_projection_dropout=encoder_projection_dropout,
        # 编码器位置卷积的卷积核大小
        encoder_pos_conv_kernel=128,
        # 编码器位置卷积的分组数
        encoder_pos_conv_groups=16,
        # 编码器的层数为 12
        encoder_num_layers=12,
        # 编码器每层是否使用注意力机制
        encoder_use_attention=[True] * 12,
        # 编码器每层是否使用前馈网络
        encoder_use_feed_forward=[True] * 12,
        # 编码器每层的注意力头数
        encoder_num_heads=[12] * 12,
        # 编码器注意力机制的头维度
        encoder_head_dim=64,
        # 编码器注意力机制的丢弃率
        encoder_attention_dropout=encoder_attention_dropout,
        # 编码器前馈网络中间特征的维度
        encoder_ff_interm_features=[3072] * 12,
        # 编码器前馈网络中间特征的丢弃率
        encoder_ff_interm_dropout=encoder_ff_interm_dropout,
        # 编码器的丢弃率
        encoder_dropout=encoder_dropout,
        # 编码器是否先进行层归一化
        encoder_layer_norm_first=False,
        # 编码器层的丢弃率
        encoder_layer_drop=encoder_layer_drop,
        # 辅助输出的数量
        aux_num_out=aux_num_out,
        # 特征提取器裁剪卷积通道数
        extractor_prune_conv_channels=extractor_prune_conv_channels,
        # 编码器裁剪注意力头数
        encoder_prune_attention_heads=encoder_prune_attention_heads,
        # 编码器裁剪注意力层
        encoder_prune_attention_layer=encoder_prune_attention_layer,
        # 编码器裁剪前馈网络中间特征
        encoder_prune_feed_forward_intermediate=encoder_prune_feed_forward_intermediate,
        # 编码器裁剪前馈网络层
        encoder_prune_feed_forward_layer=encoder_prune_feed_forward_layer,
    )
# 定义一个名为hubert_large的函数，用于构建"large" HuBERT模型
def hubert_large(
    encoder_projection_dropout: float = 0.0,  # 编码器投影丢失率
    encoder_attention_dropout: float = 0.0,  # 编码器注意力丢失率
    encoder_ff_interm_dropout: float = 0.0,  # 编码器前馈中间丢失率
    encoder_dropout: float = 0.0,  # 编码器丢失率
    encoder_layer_drop: float = 0.0,  # 编码器层丢失率
    aux_num_out: Optional[int] = None,  # 辅助输出数量（可选）
    extractor_prune_conv_channels: bool = False,  # 剪枝卷积通道
    encoder_prune_attention_heads: bool = False,  # 剪枝注意力头
    encoder_prune_attention_layer: bool = False,  # 剪枝注意力层
    encoder_prune_feed_forward_intermediate: bool = False,  # 剪枝前馈中间层
    encoder_prune_feed_forward_layer: bool = False,  # 剪枝前馈层
) -> Wav2Vec2Model:  # 返回类型为Wav2Vec2Model
    """Builds "large" :class:`HuBERT <torchaudio.models.Wav2Vec2Model>` from *HuBERT* :cite:`hsu2021hubert`

    Args:
        encoder_projection_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_attention_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_ff_interm_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_layer_drop (float):
            See :py:func:`wav2vec2_model`.
        aux_num_out (int or None, optional):
            See :py:func:`wav2vec2_model`.

    Returns:
        Wav2Vec2Model:
            The resulting model.
    """  # noqa: E501
    # 返回一个 wav2vec2 模型
    return wav2vec2_model(
        # 提取器模式为"layer_norm"
        extractor_mode="layer_norm",
        # 提取器卷积层配置为None
        extractor_conv_layer_config=None,
        # 提取器卷积层偏置为False
        extractor_conv_bias=False,
        # 编码器嵌入维度为1024
        encoder_embed_dim=1024,
        # 编码器投影丢弃率为encoder_projection_dropout
        encoder_projection_dropout=encoder_projection_dropout,
        # 编码器位置卷积核为128
        encoder_pos_conv_kernel=128,
        # 编码器位置卷积分组为16
        encoder_pos_conv_groups=16,
        # 编码器层数为24
        encoder_num_layers=24,
        # 编码器注意力头数为16
        encoder_num_heads=16,
        # 编码器注意力丢弃率为encoder_attention_dropout
        encoder_attention_dropout=encoder_attention_dropout,
        # 编码器前馈中间特征为4096
        encoder_ff_interm_features=4096,
        # 编码器前馈中间丢弃率为encoder_ff_interm_dropout
        encoder_ff_interm_dropout=encoder_ff_interm_dropout,
        # 编码器丢弃率为encoder_dropout
        encoder_dropout=encoder_dropout,
        # 编码器层归一化优先为True
        encoder_layer_norm_first=True,
        # 编码器层丢弃率为encoder_layer_drop
        encoder_layer_drop=encoder_layer_drop,
        # 辅助输出数量为aux_num_out
        aux_num_out=aux_num_out,
        # 提取器修剪卷积通道为extractor_prune_conv_channels
        extractor_prune_conv_channels=extractor_prune_conv_channels,
        # 编码器修剪注意力头数为encoder_prune_attention_heads
        encoder_prune_attention_heads=encoder_prune_attention_heads,
        # 编码器修剪注意力层为encoder_prune_attention_layer
        encoder_prune_attention_layer=encoder_prune_attention_layer,
        # 编码器修剪前馈中间特征为encoder_prune_feed_forward_intermediate
        encoder_prune_feed_forward_intermediate=encoder_prune_feed_forward_intermediate,
        # 编码器修剪前馈层为encoder_prune_feed_forward_layer
        encoder_prune_feed_forward_layer=encoder_prune_feed_forward_layer,
    )
def hubert_xlarge(
    encoder_projection_dropout: float = 0.0,  # 定义编码器投影层的丢弃率，默认为0.0
    encoder_attention_dropout: float = 0.0,  # 定义编码器注意力层的丢弃率，默认为0.0
    encoder_ff_interm_dropout: float = 0.0,  # 定义编码器前馈中间层的丢弃率，默认为0.0
    encoder_dropout: float = 0.0,  # 定义编码器的丢弃率，默认为0.0
    encoder_layer_drop: float = 0.0,  # 定义编码器层的丢弃率，默认为0.0
    aux_num_out: Optional[int] = None,  # 定义辅助输出的数量，可选参数，默认为None
    extractor_prune_conv_channels: bool = False,  # 定义是否修剪提取器卷积通道的布尔值，默认为False
    encoder_prune_attention_heads: bool = False,  # 定义是否修剪编码器注意力头的布尔值，默认为False
    encoder_prune_attention_layer: bool = False,  # 定义是否修剪编码器注意力层的布尔值，默认为False
    encoder_prune_feed_forward_intermediate: bool = False,  # 定义是否修剪编码器前馈中间层的布尔值，默认为False
    encoder_prune_feed_forward_layer: bool = False,  # 定义是否修剪编码器前馈层的布尔值，默认为False
) -> Wav2Vec2Model:  # 指定函数返回类型为Wav2Vec2Model
    """Builds "extra large" :class:`HuBERT <torchaudio.models.Wav2Vec2Model>` from *HuBERT* :cite:`hsu2021hubert`

    Args:
        encoder_projection_dropout (float):
            See :py:func:`wav2vec2_model`.  # 查看wav2vec2_model函数以了解更多信息
        encoder_attention_dropout (float):
            See :py:func:`wav2vec2_model`.  # 查看wav2vec2_model函数以了解更多信息
        encoder_ff_interm_dropout (float):
            See :py:func:`wav2vec2_model`.  # 查看wav2vec2_model函数以了解更多信息
        encoder_dropout (float):
            See :py:func:`wav2vec2_model`.  # 查看wav2vec2_model函数以了解更多信息
        encoder_layer_drop (float):
            See :py:func:`wav2vec2_model`.  # 查看wav2vec2_model函数以了解更多信息
        aux_num_out (int or None, optional):
            See :py:func:`wav2vec2_model`.  # 查看wav2vec2_model函数以了解更多信息

    Returns:
        Wav2Vec2Model:
            The resulting model.  # 返回生成的模型
    """  # noqa: E501
    # 返回一个 wav2vec2 模型
    return wav2vec2_model(
        # 提取器模式为"layer_norm"
        extractor_mode="layer_norm",
        # 提取器卷积层配置为None
        extractor_conv_layer_config=None,
        # 提取器卷积层偏置为False
        extractor_conv_bias=False,
        # 编码器嵌入维度为1280
        encoder_embed_dim=1280,
        # 编码器投影丢弃率为encoder_projection_dropout
        encoder_projection_dropout=encoder_projection_dropout,
        # 编码器位置卷积核为128
        encoder_pos_conv_kernel=128,
        # 编码器位置卷积分组为16
        encoder_pos_conv_groups=16,
        # 编码器层数为48
        encoder_num_layers=48,
        # 编码器注意力头数为16
        encoder_num_heads=16,
        # 编码器注意力丢弃率为encoder_attention_dropout
        encoder_attention_dropout=encoder_attention_dropout,
        # 编码器前馈中间特征为5120
        encoder_ff_interm_features=5120,
        # 编码器前馈中间丢弃率为encoder_ff_interm_dropout
        encoder_ff_interm_dropout=encoder_ff_interm_dropout,
        # 编码器丢弃率为encoder_dropout
        encoder_dropout=encoder_dropout,
        # 编码器层归一化在前
        encoder_layer_norm_first=True,
        # 编码器层丢弃率为encoder_layer_drop
        encoder_layer_drop=encoder_layer_drop,
        # 辅助输出数量为aux_num_out
        aux_num_out=aux_num_out,
        # 提取器修剪卷积通道为extractor_prune_conv_channels
        extractor_prune_conv_channels=extractor_prune_conv_channels,
        # 编码器修剪注意力头为encoder_prune_attention_heads
        encoder_prune_attention_heads=encoder_prune_attention_heads,
        # 编码器修剪注意力层为encoder_prune_attention_layer
        encoder_prune_attention_layer=encoder_prune_attention_layer,
        # 编码器修剪前馈中间为encoder_prune_feed_forward_intermediate
        encoder_prune_feed_forward_intermediate=encoder_prune_feed_forward_intermediate,
        # 编码器修剪前馈层为encoder_prune_feed_forward_layer
        encoder_prune_feed_forward_layer=encoder_prune_feed_forward_layer,
    )
# 初始化 Hubert 预训练模型的参数
def _init_hubert_pretrain_model(module):
    # 如果是 LayerNorm 类型的模块，使用 kaiming_normal_ 方法初始化权重
    if isinstance(module, components.LayerNorm):
        torch.nn.init.kaiming_normal_(module.conv.weight)
    # 如果是 ConvolutionalPositionalEmbedding 类型的模块
    elif isinstance(module, components.ConvolutionalPositionalEmbedding):
        # 标准差计算
        std = math.sqrt(4.0 / (module.embed_dim * module.kernel_size))
        # 使用 normal_ 方法初始化权重和常数项
        torch.nn.init.normal_(module.conv.weight, mean=0.0, std=std)
        torch.nn.init.constant_(module.conv.bias, 0.0)
    # 如果是 SelfAttention 类型的模块
    elif isinstance(module, components.SelfAttention):
        # 使用 xavier_uniform_ 方法初始化权重和常数项
        torch.nn.init.xavier_uniform_(module.k_proj.weight, gain=1 / math.sqrt(2))
        torch.nn.init.xavier_uniform_(module.v_proj.weight, gain=1 / math.sqrt(2))
        torch.nn.init.xavier_uniform_(module.q_proj.weight, gain=1 / math.sqrt(2))
        torch.nn.init.xavier_uniform_(module.out_proj.weight)
        torch.nn.init.constant_(module.out_proj.bias, 0.0)
    # 如果是 Transformer 类型的模块
    elif isinstance(module, components.Transformer):
        # 调用 _init_transformer_params 方法初始化参数
        module.apply(components._init_transformer_params)
    # 其他情况，不做任何操作
    else:
        pass

# 定义 wavlm_model 函数，接收多个参数
def wavlm_model(
    extractor_mode: str,
    extractor_conv_layer_config: Optional[List[Tuple[int, int, int]]],
    extractor_conv_bias: bool,
    encoder_embed_dim: int,
    encoder_projection_dropout: float,
    encoder_pos_conv_kernel: int,
    encoder_pos_conv_groups: int,
    encoder_num_layers: int,
    encoder_use_attention: List[bool],
    encoder_use_feed_forward: List[bool],
    encoder_total_num_heads: List[int],
    encoder_remaining_heads: List[List[int]],
    encoder_num_buckets: int,
    encoder_max_distance: int,
    encoder_attention_dropout: float,
    encoder_ff_interm_features: List[int],
    encoder_ff_interm_dropout: float,
    encoder_dropout: float,
    encoder_layer_norm_first: bool,
    encoder_layer_drop: float,
    aux_num_out: Optional[int],
    normalize_waveform: bool,
    extractor_prune_conv_channels: bool = False,
    # 设置编码器剪枝注意力头的标志，默认为 False
    encoder_prune_attention_heads: bool = False,
    # 设置编码器剪枝注意力层的标志，默认为 False
    encoder_prune_attention_layer: bool = False,
    # 设置编码器剪枝前馈网络中间层的标志，默认为 False
    encoder_prune_feed_forward_intermediate: bool = False,
    # 设置编码器剪枝前馈网络层的标志，默认为 False
    encoder_prune_feed_forward_layer: bool = False,
def build_wavelm_model(
    extractor_mode: str,
    extractor_conv_layer_config: Optional[List[Tuple[int, int]]],
    extractor_conv_bias: bool,
    encoder_embed_dim: int,
    encoder_projection_dropout: float,
    encoder_pos_conv_kernel: int,
    encoder_pos_conv_groups: int,
    encoder_num_layers: int,
    encoder_num_heads: int,
    encoder_num_buckets: int,
    encoder_max_distance: int,
    encoder_attention_dropout: float,
    encoder_ff_interm_features: int,
    encoder_ff_interm_dropout: float,
    encoder_dropout: float,
    encoder_layer_norm_first: bool,
    encoder_layer_drop: float,
    aux_num_out: Optional[int]
) -> Wav2Vec2Model:
    """
    Builds custom WaveLM model :cite:`chen2022wavlm`. The architecture is compatible
    with Wav2Vec2 model :cite:`baevski2020wav2vec`, and so the output object is
    :class:`~torchaudio.models.Wav2Vec2Model`. Most of the arguments have the same meaning
    as in :py:func:`wav2vec2_model` so please refer there for documentation.

    Args:
        extractor_mode (str): Operation mode of feature extractor.
            See :py:func:`wav2vec2_model`.

        extractor_conv_layer_config (list of integer tuples or None):
            See :py:func:`wav2vec2_model`.

        extractor_conv_bias (bool):
            See :py:func:`wav2vec2_model`.

        encoder_embed_dim (int):
            See :py:func:`wav2vec2_model`.

        encoder_projection_dropout (float):
            See :py:func:`wav2vec2_model`.

        encoder_pos_conv_kernel (int):
            See :py:func:`wav2vec2_model`.

        encoder_pos_conv_groups (int):
            See :py:func:`wav2vec2_model`.

        encoder_num_layers (int):
            See :py:func:`wav2vec2_model`.

        encoder_num_heads (int):
            See :py:func:`wav2vec2_model`.

        encoder_num_buckets (int):
            Number of buckets for relative position embedding.

        encoder_max_distance (int):
            Maximum distance for relative position embedding.

        encoder_attention_dropout (float):
            See :py:func:`wav2vec2_model`.

        encoder_ff_interm_features (int):
            See :py:func:`wav2vec2_model`.

        encoder_ff_interm_dropout (float):
            See :py:func:`wav2vec2_model`.

        encoder_dropout (float):
            See :py:func:`wav2vec2_model`.

        encoder_layer_norm_first (bool):
            See :py:func:`wav2vec2_model`.

        encoder_layer_drop (float):
            See :py:func:`wav2vec2_model`.

        aux_num_out (int or None):
            See :py:func:`wav2vec2_model`.
    """
    Returns:
        Wav2Vec2Model:
            The resulting model.
    """
    # 如果未提供特征提取器的卷积层配置，则使用默认配置
    if extractor_conv_layer_config is None:
        extractor_conv_layer_config = [(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] * 2

    # 获取特征提取器
    feature_extractor = components._get_feature_extractor(
        extractor_mode, extractor_conv_layer_config, extractor_conv_bias,
        prune_conv_channels=extractor_prune_conv_channels,
    )
    # 获取 WavLM 编码器
    encoder = components._get_wavlm_encoder(
        in_features=extractor_conv_layer_config[-1][0],
        embed_dim=encoder_embed_dim,
        dropout_input=encoder_projection_dropout,
        pos_conv_kernel=encoder_pos_conv_kernel,
        pos_conv_groups=encoder_pos_conv_groups,
        num_layers=encoder_num_layers,
        use_attention=encoder_use_attention,
        use_feed_forward=encoder_use_feed_forward,
        total_num_heads=encoder_total_num_heads,
        remaining_heads=encoder_remaining_heads,
        num_buckets=encoder_num_buckets,
        max_distance=encoder_max_distance,
        attention_dropout=encoder_attention_dropout,
        ff_interm_features=encoder_ff_interm_features,
        ff_interm_dropout=encoder_ff_interm_dropout,
        dropout=encoder_dropout,
        layer_norm_first=encoder_layer_norm_first,
        layer_drop=encoder_layer_drop,
        prune_attention_heads=encoder_prune_attention_heads,
        prune_attention_layer=encoder_prune_attention_layer,
        prune_feed_forward_intermediate=encoder_prune_feed_forward_intermediate,
        prune_feed_forward_layer=encoder_prune_feed_forward_layer,
    )
    # 如果有辅助输出的数量，则创建辅助输出层
    aux = None
    if aux_num_out is not None:
        aux = torch.nn.Linear(in_features=encoder_embed_dim, out_features=aux_num_out)
    # 返回 Wav2Vec2Model 模型
    return Wav2Vec2Model(normalize_waveform, feature_extractor, encoder, aux)
def wavlm_base(
    encoder_projection_dropout: float = 0.1,  # 设置编码器投影层的dropout比例
    encoder_attention_dropout: float = 0.1,  # 设置编码器注意力层的dropout比例
    encoder_ff_interm_dropout: float = 0.1,  # 设置编码器前馈中间层的dropout比例
    encoder_dropout: float = 0.1,  # 设置编码器的dropout比例
    encoder_layer_drop: float = 0.1,  # 设置编码器层的dropout比例
    aux_num_out: Optional[int] = None,  # 设置辅助输出的数量，可选参数
) -> Wav2Vec2Model:  # 返回类型为Wav2Vec2Model的模型
    """Builds "base" WaveLM model :cite:`chen2022wavlm`. The architecture is compatible
    with Wav2Vec2 model :cite:`baevski2020wav2vec`, and so the output class is
    :class:`~torchaudio.models.Wav2Vec2Model`.

    Args:
        encoder_projection_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_attention_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_ff_interm_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_layer_drop (float):
            See :py:func:`wav2vec2_model`.
        aux_num_out (int, optional):
            See :py:func:`wav2vec2_model`.

    Returns:
        Wav2Vec2Model:
            The resulting model.
    """
    return wavlm_model(
        extractor_mode="group_norm",  # 设置特征提取器的模式为"group_norm"
        extractor_conv_layer_config=None,  # 设置特征提取器的卷积层配置为None
        extractor_conv_bias=False,  # 设置特征提取器的卷积层偏置为False
        encoder_embed_dim=768,  # 设置编码器的嵌入维度为768
        encoder_projection_dropout=encoder_projection_dropout,  # 设置编码器投影层的dropout比例
        encoder_pos_conv_kernel=128,  # 设置编码器位置卷积的卷积核大小为128
        encoder_pos_conv_groups=16,  # 设置编码器位置卷积的分组数为16
        encoder_num_layers=12,  # 设置编码器的层数为12
        encoder_num_heads=12,  # 设置编码器的注意力头数为12
        encoder_num_buckets=320,  # 设置编码器的桶数为320
        encoder_max_distance=800,  # 设置编码器的最大距离为800
        encoder_attention_dropout=encoder_attention_dropout,  # 设置编码器注意力层的dropout比例
        encoder_ff_interm_features=3072,  # 设置编码器前馈中间层的特征数为3072
        encoder_ff_interm_dropout=encoder_ff_interm_dropout,  # 设置编码器前馈中间层的dropout比例
        encoder_dropout=encoder_dropout,  # 设置编码器的dropout比例
        encoder_layer_norm_first=False,  # 设置编码器的层归一化是否在前面为False
        encoder_layer_drop=encoder_layer_drop,  # 设置编码器层的dropout比例
        aux_num_out=aux_num_out,  # 设置辅助输出的数量
    )


def wavlm_large(
    encoder_projection_dropout: float = 0.1,  # 设置编码器投影层的dropout比例
    encoder_attention_dropout: float = 0.1,  # 设置编码器注意力层的dropout比例
    encoder_ff_interm_dropout: float = 0.0,  # 设置编码器前馈中间层的dropout比例为0.0
    # 设置编码器的丢弃率为0.1
    encoder_dropout: float = 0.1,
    # 设置编码器层的丢弃率为0.1
    encoder_layer_drop: float = 0.1,
    # 设置辅助输出的数量为可选的整数，如果没有指定则为None
    aux_num_out: Optional[int] = None,
def build_large_wavelm_model(
    encoder_projection_dropout: float,
    encoder_attention_dropout: float,
    encoder_ff_interm_dropout: float,
    encoder_dropout: float,
    encoder_layer_drop: float,
    aux_num_out: int = None
) -> Wav2Vec2Model:
    """Builds "large" WaveLM model :cite:`chen2022wavlm`. The architecture is compatible
    with Wav2Vec2 model :cite:`baevski2020wav2vec`, and so the output class is
    :class:`~torchaudio.models.Wav2Vec2Model`.

    Args:
        encoder_projection_dropout (float):
            Dropout probability for the encoder projection layer.
        encoder_attention_dropout (float):
            Dropout probability for the encoder attention layer.
        encoder_ff_interm_dropout (float):
            Dropout probability for the encoder feed-forward intermediate layer.
        encoder_dropout (float):
            Dropout probability for the encoder layer.
        encoder_layer_drop (float):
            Dropout probability for the encoder layer.
        aux_num_out (int, optional):
            Number of auxiliary outputs.

    Returns:
        Wav2Vec2Model:
            The resulting model.
    """
    return wavlm_model(
        extractor_mode="layer_norm",
        extractor_conv_layer_config=None,
        extractor_conv_bias=False,
        encoder_embed_dim=1024,
        encoder_projection_dropout=encoder_projection_dropout,
        encoder_pos_conv_kernel=128,
        encoder_pos_conv_groups=16,
        encoder_num_layers=24,
        encoder_num_heads=16,
        encoder_num_buckets=320,
        encoder_max_distance=800,
        encoder_attention_dropout=encoder_attention_dropout,
        encoder_ff_interm_features=4096,
        encoder_ff_interm_dropout=encoder_ff_interm_dropout,
        encoder_dropout=encoder_dropout,
        encoder_layer_norm_first=True,
        encoder_layer_drop=encoder_layer_drop,
        aux_num_out=aux_num_out,
    )
```