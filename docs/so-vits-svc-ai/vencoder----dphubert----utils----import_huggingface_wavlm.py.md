# `so-vits-svc\vencoder\dphubert\utils\import_huggingface_wavlm.py`

```py
# 导入日志模块
import logging
# 导入类型提示模块
from typing import Any, Dict
# 从 torch.nn 模块中导入 Module 类
from torch.nn import Module
# 从当前目录下的 model 模块中导入 Wav2Vec2Model、wav2vec2_model、wavlm_model 类
from ..model import Wav2Vec2Model, wav2vec2_model, wavlm_model

# 获取当前模块的日志记录器
_LG = logging.getLogger(__name__)

# 定义一个函数，用于获取配置信息
def _get_config(cfg):
    # 构建配置字典
    config = {
        "extractor_mode": f"{cfg.feat_extract_norm}_norm",
        "extractor_conv_layer_config": list(zip(cfg.conv_dim, cfg.conv_kernel, cfg.conv_stride)),
        "extractor_conv_bias": cfg.conv_bias,
        "encoder_embed_dim": cfg.hidden_size,
        "encoder_projection_dropout": cfg.feat_proj_dropout,
        "encoder_pos_conv_kernel": cfg.num_conv_pos_embeddings,
        "encoder_pos_conv_groups": cfg.num_conv_pos_embedding_groups,
        "encoder_num_layers": cfg.num_hidden_layers,
        "encoder_num_heads": cfg.num_attention_heads,
        "encoder_attention_dropout": cfg.attention_dropout,
        "encoder_ff_interm_features": cfg.intermediate_size,
        "encoder_ff_interm_dropout": cfg.activation_dropout,
        "encoder_dropout": cfg.hidden_dropout,
        "encoder_layer_norm_first": cfg.do_stable_layer_norm,
        "encoder_layer_drop": cfg.layerdrop,
    }
    # 返回配置字典
    return config

# 定义一个函数，用于获取 Wavlm 配置信息
def _get_config_wavlm(cfg):
    # 定义配置字典，包含各种模型参数和设置
    config = {
        # 提取器模式，使用特征提取的归一化方式
        "extractor_mode": f"{cfg.feat_extract_norm}_norm",
        # 提取器卷积层配置，包括维度、卷积核大小和步长
        "extractor_conv_layer_config": list(zip(cfg.conv_dim, cfg.conv_kernel, cfg.conv_stride)),
        # 提取器卷积层偏置
        "extractor_conv_bias": cfg.conv_bias,
        # 编码器嵌入维度
        "encoder_embed_dim": cfg.hidden_size,
        # 编码器投影层丢弃率
        "encoder_projection_dropout": cfg.feat_proj_dropout,
        # 编码器位置卷积核大小
        "encoder_pos_conv_kernel": cfg.num_conv_pos_embeddings,
        # 编码器位置卷积分组数
        "encoder_pos_conv_groups": cfg.num_conv_pos_embedding_groups,
        # 编码器层数
        "encoder_num_layers": cfg.num_hidden_layers,
        # 编码器是否使用注意力机制
        "encoder_use_attention": [True] * cfg.num_hidden_layers,
        # 编码器是否使用前馈网络
        "encoder_use_feed_forward": [True] * cfg.num_hidden_layers,
        # 编码器总注意力头数
        "encoder_total_num_heads": [cfg.num_attention_heads for _ in range(cfg.num_hidden_layers)],
        # 编码器剩余注意力头数
        "encoder_remaining_heads": [list(range(cfg.num_attention_heads)) for _ in range(cfg.num_hidden_layers)],
        # 编码器桶数
        "encoder_num_buckets": cfg.num_buckets,
        # 编码器最大距离
        "encoder_max_distance": cfg.max_bucket_distance,
        # 编码器注意力丢弃率
        "encoder_attention_dropout": cfg.attention_dropout,
        # 编码器前馈网络中间层特征维度
        "encoder_ff_interm_features": [cfg.intermediate_size for _ in range(cfg.num_hidden_layers)],
        # 编码器前馈网络中间层丢弃率
        "encoder_ff_interm_dropout": cfg.activation_dropout,
        # 编码器丢弃率
        "encoder_dropout": cfg.hidden_dropout,
        # 编码器是否首先进行稳定层归一化
        "encoder_layer_norm_first": cfg.do_stable_layer_norm,
        # 编码器层丢弃率
        "encoder_layer_drop": cfg.layerdrop,
        # 是否对波形进行归一化
        "normalize_waveform": cfg.feat_extract_norm == "layer",
    }
    # 返回配置字典
    return config
# 根据配置和原始模型构建新模型
def _build(config, original):
    # 判断原始模型是否用于 CTC
    is_for_ctc = original.__class__.__name__ in ["Wav2Vec2ForCTC", "WavLMForCTC"]
    if is_for_ctc:
        # 如果是用于 CTC，则获取原始模型的词汇表大小和 wav2vec2 对象
        aux_num_out = original.config.vocab_size
        wav2vec2 = original.wav2vec2
    else:
        # 如果不是用于 CTC，则记录警告信息，并将 aux_num_out 和 wav2vec2 设置为 None 和原始模型
        _LG.warning(
            "The model is not an instance of Wav2Vec2ForCTC or WavLMForCTC. " '"lm_head" module is not imported.'
        )
        aux_num_out = None
        wav2vec2 = original
    # 判断原始模型是否为 WavLM
    is_wavlm = original.__class__.__name__ in ["WavLMModel", "WavLMForCTC"]
    if is_wavlm:
        # 如果是 WavLM，则根据配置和 aux_num_out 构建导入模型
        imported = wavlm_model(**config, aux_num_out=aux_num_out)
    else:
        # 如果不是 WavLM，则根据配置和 aux_num_out 构建导入模型
        imported = wav2vec2_model(**config, aux_num_out=aux_num_out)
    # 打印导入模型的特征提取器和编码器的状态字典加载情况
    print(imported.feature_extractor.load_state_dict(wav2vec2.feature_extractor.state_dict(), strict=False))
    print(imported.encoder.feature_projection.load_state_dict(wav2vec2.feature_projection.state_dict(), strict=False))
    # 获取编码器的状态字典
    encoder_state_dict = wav2vec2.encoder.state_dict()
    if is_wavlm:  # 为了与 HF 模型兼容，重命名线性变换的参数
        transform_wavlm_encoder_state(encoder_state_dict, config["encoder_num_layers"])
    # 打印导入模型的编码器的状态字典加载情况
    print(imported.encoder.transformer.load_state_dict(encoder_state_dict, strict=False))
    if is_for_ctc:
        # 如果是用于 CTC，则加载原始模型的 lm_head 状态字典到导入模型的 aux
        imported.aux.load_state_dict(original.lm_head.state_dict())
    # 返回导入模型
    return imported


# 转换 WavLM 编码器状态的函数
def transform_wavlm_encoder_state(state: Dict[str, Any], encoder_num_layers: int):
    """Converts WavLM encoder state from HuggingFace format. In particular, concatenates linear projection weights and
    biases to align with the structure of ``torch.nn.MultiheadAttention``.
    """
    pass
    

# 从 Transformers 中导入模型的函数
def import_huggingface_model(original: Module) -> Wav2Vec2Model:
    """Builds :class:`Wav2Vec2Model` from the corresponding model object of
    `Transformers <https://huggingface.co/transformers/>`_.

    Args:
        original (torch.nn.Module): An instance of ``Wav2Vec2ForCTC`` from ``transformers``.

    Returns:
        Wav2Vec2Model: Imported model.
    # 导入所需的模块和函数
    Example
        >>> from torchaudio.models.wav2vec2.utils import import_huggingface_model
        >>>
        # 从预训练模型中加载原始模型
        >>> original = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        # 导入原始模型
        >>> model = import_huggingface_model(original)
        >>>
        # 加载音频文件
        >>> waveforms, _ = torchaudio.load("audio.wav")
        # 使用模型对音频进行推理，得到预测的logits
        >>> logits, _ = model(waveforms)
    """
    # 输出日志信息，表示正在导入模型
    _LG.info("Importing model.")
    # 输出日志信息，表示正在加载模型配置
    _LG.info("Loading model configuration.")
    # 判断原始模型是否为语言模型
    is_wavlm = original.__class__.__name__ in ["WavLMModel", "WavLMForCTC"]
    # 如果是语言模型，则获取语言模型的配置
    if is_wavlm:
        config = _get_config_wavlm(original.config)
    # 如果不是语言模型，则获取一般模型的配置
    else:
        config = _get_config(original.config)
    # 输出调试日志，显示模型的配置信息
    _LG.debug("  - config: %s", config)
    # 输出日志信息，表示正在构建模型
    _LG.info("Building model.")
    # 根据配置和原始模型构建导入的模型
    imported = _build(config, original)
    # 返回导入的模型
    return imported
```