# `.\models\flaubert\configuration_flaubert.py`

```py
# coding=utf-8
# 版权声明，包括版权所有者和许可协议信息
# 此代码用于Flaubert配置
from collections import OrderedDict  # 导入OrderedDict类
from typing import Mapping  # 导入Mapping类型

from ...configuration_utils import PretrainedConfig  # 导入PretrainedConfig类
from ...onnx import OnnxConfig  # 导入OnnxConfig类
from ...utils import logging  # 导入logging模块

logger = logging.get_logger(__name__)  # 获取logger对象

# Flaubert预训练模型的配置文件映射
FLAUBERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "flaubert/flaubert_small_cased": "https://huggingface.co/flaubert/flaubert_small_cased/resolve/main/config.json",
    "flaubert/flaubert_base_uncased": "https://huggingface.co/flaubert/flaubert_base_uncased/resolve/main/config.json",
    "flaubert/flaubert_base_cased": "https://huggingface.co/flaubert/flaubert_base_cased/resolve/main/config.json",
    "flaubert/flaubert_large_cased": "https://huggingface.co/flaubert/flaubert_large_cased/resolve/main/config.json",
}


class FlaubertConfig(PretrainedConfig):
    """
    这是一个配置类，用于存储[`FlaubertModel`]或[`TFFlaubertModel`]的配置。
    根据指定的参数实例化FlauBERT模型，定义模型架构。
    使用默认值实例化配置将产生与FlauBERT[flaubert/flaubert_base_uncased](https://huggingface.co/flaubert/flaubert_base_uncased)架构类似的配置。

    配置对象继承自[`PretrainedConfig`]，可用于控制模型输出。阅读[`PretrainedConfig`]的文档以获取更多信息。
    """

    model_type = "flaubert"  # 模型类型为Flaubert
    attribute_map = {
        "hidden_size": "emb_dim",  # 隐藏层大小映射为emb_dim
        "num_attention_heads": "n_heads",  # 注意力头数映射为n_heads
        "num_hidden_layers": "n_layers",  # 隐藏层数映射为n_layers
        "n_words": "vocab_size",  # 词汇量大小映射为vocab_size，向后兼容
    }
    # 初始化 FlaubertConfig 对象
    def __init__(
        self,
        pre_norm=False, # 是否进行预标准化
        layerdrop=0.0, # layerdrop 参数
        vocab_size=30145, # 词汇量大小
        emb_dim=2048, # 嵌入维度大小
        n_layers=12, # 层数
        n_heads=16, # 多头注意力头数
        dropout=0.1, # dropout 概率
        attention_dropout=0.1, # 注意力dropout 概率
        gelu_activation=True, # 是否使用gelu激活函数
        sinusoidal_embeddings=False, # 是否使用正弦嵌入
        causal=False, # 是否使用因果关系
        asm=False, # 是否使用自注意力机制
        n_langs=1, # 语言数量
        use_lang_emb=True, # 是否使用语言嵌入
        max_position_embeddings=512, # 最大位置嵌入
        embed_init_std=2048**-0.5, # 嵌入初始化标准差
        layer_norm_eps=1e-12, # layer normalization 系数
        init_std=0.02, # 初始化标准差
        bos_index=0, # bos 索引
        eos_index=1, # eos 索引
        pad_index=2, # pad 索引
        unk_index=3, # 未知单词索引
        mask_index=5, # mask 索引
        is_encoder=True, # 是否为编码器
        summary_type="first", # 摘要类型
        summary_use_proj=True, # 是否使用投影进行摘要
        summary_activation=None, # 摘要激活函数
        summary_proj_to_labels=True, # 是否将摘要投影到标签
        summary_first_dropout=0.1, # 摘要前dropout概率
        start_n_top=5, # 开始top值
        end_n_top=5, # 结束top值
        mask_token_id=0, # mask标记id
        lang_id=0, # 语言id
        pad_token_id=2, # pad标记id
        bos_token_id=0, # bos标记id
        **kwargs, # 其他参数
    ):
        """Constructs FlaubertConfig."""
        self.pre_norm = pre_norm # 设置预标准化
        self.layerdrop = layerdrop # 设置 layerdrop 参数
        self.vocab_size = vocab_size # 设置词汇量大小
        self.emb_dim = emb_dim # 设置嵌入维度大小
        self.n_layers = n_layers # 设置层数
        self.n_heads = n_heads # 设置多头注意力头数
        self.dropout = dropout # 设置dropout概率
        self.attention_dropout = attention_dropout # 设置注意力dropout概率
        self.gelu_activation = gelu_activation # 设置是否使用gelu激活函数
        self.sinusoidal_embeddings = sinusoidal_embeddings # 设置是否使用正弦嵌入
        self.causal = causal # 设置是否使用因果关系
        self.asm = asm # 设置是否使用自注意力机制
        self.n_langs = n_langs # 设置语言数量
        self.use_lang_emb = use_lang_emb # 设置是否使用语言嵌入
        self.layer_norm_eps = layer_norm_eps # 设置layer normalization 系数
        self.bos_index = bos_index # 设置bos索引
        self.eos_index = eos_index # 设置eos索引
        self.pad_index = pad_index # 设置pad索引
        self.unk_index = unk_index # 设置未知单词索引
        self.mask_index = mask_index # 设置mask索引
        self.is_encoder = is_encoder # 设置是否为编码器
        self.max_position_embeddings = max_position_embeddings # 设置最大位置嵌入
        self.embed_init_std = embed_init_std # 设置嵌入初始化标准差
        self.init_std = init_std # 设置初始化标准差
        self.summary_type = summary_type # 设置摘要类型
        self.summary_use_proj = summary_use_proj # 设置是否使用投影进行摘要
        self.summary_activation = summary_activation # 设置摘要激活函数
        self.summary_proj_to_labels = summary_proj_to_labels # 设置是否将摘要投影到标签
        self.summary_first_dropout = summary_first_dropout # 设置摘要前dropout概率
        self.start_n_top = start_n_top # 设置开始top值
        self.end_n_top = end_n_top # 设置结束top值
        self.mask_token_id = mask_token_id # 设置mask标记id
        self.lang_id = lang_id # 设置语言id
    
        if "n_words" in kwargs:
            self.n_words = kwargs["n_words"] # 如果kwargs中包含'n_words'参数，则设置self.n_words为kwargs["n_words"]
    
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, **kwargs) # 调用父类的初始化方法，传入参数pad_token_id和bos_token_id，以及其他kwargs参数
# 定义一个名为FlaubertOnnxConfig的类，它继承自OnnxConfig类
class FlaubertOnnxConfig(OnnxConfig):
    # inputs属性是一个只读属性，返回一个映射关系，其中键为字符串，值为映射关系
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # 如果任务为"multiple-choice"，则动态轴为{0: "batch", 1: "choice", 2: "sequence"}
        if self.task == "multiple-choice":
            dynamic_axis = {0: "batch", 1: "choice", 2: "sequence"}
        # 否则，动态轴为{0: "batch", 1: "sequence"}
        else:
            dynamic_axis = {0: "batch", 1: "sequence"}
        # 返回一个有序字典，包含输入的名称和对应的动态轴
        return OrderedDict(
            [
                ("input_ids", dynamic_axis),
                ("attention_mask", dynamic_axis),
            ]
        )
```