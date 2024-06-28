# `.\models\flaubert\configuration_flaubert.py`

```
# coding=utf-8
# Copyright 2019-present CNRS, Facebook Inc. and the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" Flaubert configuration"""

# 从 collections 模块导入 OrderedDict 类
from collections import OrderedDict
# 从 typing 模块导入 Mapping 类型
from typing import Mapping

# 导入预训练配置的基类 PretrainedConfig
from ...configuration_utils import PretrainedConfig
# 导入 ONNX 配置
from ...onnx import OnnxConfig
# 导入日志工具
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 预训练模型配置文件的映射表，将模型名称映射到配置文件的 URL
FLAUBERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "flaubert/flaubert_small_cased": "https://huggingface.co/flaubert/flaubert_small_cased/resolve/main/config.json",
    "flaubert/flaubert_base_uncased": "https://huggingface.co/flaubert/flaubert_base_uncased/resolve/main/config.json",
    "flaubert/flaubert_base_cased": "https://huggingface.co/flaubert/flaubert_base_cased/resolve/main/config.json",
    "flaubert/flaubert_large_cased": "https://huggingface.co/flaubert/flaubert_large_cased/resolve/main/config.json",
}


class FlaubertConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`FlaubertModel`] or a [`TFFlaubertModel`]. It is
    used to instantiate a FlauBERT model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the FlauBERT
    [flaubert/flaubert_base_uncased](https://huggingface.co/flaubert/flaubert_base_uncased) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    """

    # 模型类型标识为 "flaubert"
    model_type = "flaubert"
    # 属性映射，将一些通用名称映射到 Flaubert 模型的特定参数名称
    attribute_map = {
        "hidden_size": "emb_dim",
        "num_attention_heads": "n_heads",
        "num_hidden_layers": "n_layers",
        "n_words": "vocab_size",  # For backward compatibility
    }
    # 初始化函数，用于构建 FlaubertConfig 对象
    def __init__(
        self,
        pre_norm=False,  # 是否进行预归一化，默认为 False
        layerdrop=0.0,  # 层随机丢弃概率，默认为 0.0
        vocab_size=30145,  # 词汇表大小，默认为 30145
        emb_dim=2048,  # 词嵌入维度，默认为 2048
        n_layers=12,  # 层数，默认为 12
        n_heads=16,  # 注意力头数，默认为 16
        dropout=0.1,  # 全连接层和注意力层的 dropout 概率，默认为 0.1
        attention_dropout=0.1,  # 注意力机制中的 dropout 概率，默认为 0.1
        gelu_activation=True,  # 是否使用 GeLU 激活函数，默认为 True
        sinusoidal_embeddings=False,  # 是否使用正弦位置编码，默认为 False
        causal=False,  # 是否是因果模型，默认为 False
        asm=False,  # 是否使用异步条带模型，默认为 False
        n_langs=1,  # 支持的语言数量，默认为 1
        use_lang_emb=True,  # 是否使用语言嵌入，默认为 True
        max_position_embeddings=512,  # 最大位置编码数量，默认为 512
        embed_init_std=2048**-0.5,  # 嵌入初始化的标准差，默认为 2048 的负半幂
        layer_norm_eps=1e-12,  # 层归一化的 epsilon，默认为 1e-12
        init_std=0.02,  # 初始化标准差，默认为 0.02
        bos_index=0,  # 开始词索引，默认为 0
        eos_index=1,  # 结束词索引，默认为 1
        pad_index=2,  # 填充词索引，默认为 2
        unk_index=3,  # 未知词索引，默认为 3
        mask_index=5,  # 掩码词索引，默认为 5
        is_encoder=True,  # 是否是编码器，默认为 True
        summary_type="first",  # 摘要类型，默认为 "first"
        summary_use_proj=True,  # 是否对摘要进行投影，默认为 True
        summary_activation=None,  # 摘要激活函数，默认为 None
        summary_proj_to_labels=True,  # 是否对摘要投影到标签，默认为 True
        summary_first_dropout=0.1,  # 第一次摘要投影的 dropout 概率，默认为 0.1
        start_n_top=5,  # 开始的 top-N 概率，默认为 5
        end_n_top=5,  # 结束的 top-N 概率，默认为 5
        mask_token_id=0,  # 掩码的 token id，默认为 0
        lang_id=0,  # 语言 id，默认为 0
        pad_token_id=2,  # 填充的 token id，默认为 2
        bos_token_id=0,  # 开始的 token id，默认为 0
        **kwargs,  # 其余参数作为关键字参数传递
    ):
        """Constructs FlaubertConfig."""
        self.pre_norm = pre_norm  # 初始化对象的预归一化属性
        self.layerdrop = layerdrop  # 初始化对象的层随机丢弃概率属性
        self.vocab_size = vocab_size  # 初始化对象的词汇表大小属性
        self.emb_dim = emb_dim  # 初始化对象的词嵌入维度属性
        self.n_layers = n_layers  # 初始化对象的层数属性
        self.n_heads = n_heads  # 初始化对象的注意力头数属性
        self.dropout = dropout  # 初始化对象的全连接层和注意力层的 dropout 概率属性
        self.attention_dropout = attention_dropout  # 初始化对象的注意力机制中的 dropout 概率属性
        self.gelu_activation = gelu_activation  # 初始化对象的 GeLU 激活函数属性
        self.sinusoidal_embeddings = sinusoidal_embeddings  # 初始化对象的正弦位置编码属性
        self.causal = causal  # 初始化对象的因果模型属性
        self.asm = asm  # 初始化对象的异步条带模型属性
        self.n_langs = n_langs  # 初始化对象的支持的语言数量属性
        self.use_lang_emb = use_lang_emb  # 初始化对象的语言嵌入属性
        self.layer_norm_eps = layer_norm_eps  # 初始化对象的层归一化 epsilon 属性
        self.bos_index = bos_index  # 初始化对象的开始词索引属性
        self.eos_index = eos_index  # 初始化对象的结束词索引属性
        self.pad_index = pad_index  # 初始化对象的填充词索引属性
        self.unk_index = unk_index  # 初始化对象的未知词索引属性
        self.mask_index = mask_index  # 初始化对象的掩码词索引属性
        self.is_encoder = is_encoder  # 初始化对象的是否是编码器属性
        self.max_position_embeddings = max_position_embeddings  # 初始化对象的最大位置编码数量属性
        self.embed_init_std = embed_init_std  # 初始化对象的嵌入初始化标准差属性
        self.init_std = init_std  # 初始化对象的初始化标准差属性
        self.summary_type = summary_type  # 初始化对象的摘要类型属性
        self.summary_use_proj = summary_use_proj  # 初始化对象的摘要是否使用投影属性
        self.summary_activation = summary_activation  # 初始化对象的摘要激活函数属性
        self.summary_proj_to_labels = summary_proj_to_labels  # 初始化对象的摘要是否投影到标签属性
        self.summary_first_dropout = summary_first_dropout  # 初始化对象的第一次摘要投影的 dropout 概率属性
        self.start_n_top = start_n_top  # 初始化对象的开始的 top-N 概率属性
        self.end_n_top = end_n_top  # 初始化对象的结束的 top-N 概率属性
        self.mask_token_id = mask_token_id  # 初始化对象的掩码的 token id 属性
        self.lang_id = lang_id  # 初始化对象的语言 id 属性

        if "n_words" in kwargs:  # 如果关键字参数中包含 'n_words'，则将其作为属性存储在对象中
            self.n_words = kwargs["n_words"]

        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, **kwargs)  # 调用父类初始化方法，传递填充 token id 和开始 token id，以及其他关键字参数
# 定义一个名为 FlaubertOnnxConfig 的类，继承自 OnnxConfig 类
class FlaubertOnnxConfig(OnnxConfig):
    
    # 定义一个属性 inputs，返回一个映射，其键为字符串，值为映射（整数到字符串）
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        
        # 如果任务为 "multiple-choice"，则动态轴包含 3 个维度
        if self.task == "multiple-choice":
            dynamic_axis = {0: "batch", 1: "choice", 2: "sequence"}
        # 否则，动态轴包含 2 个维度
        else:
            dynamic_axis = {0: "batch", 1: "sequence"}
        
        # 返回一个有序字典，包含两个条目：input_ids 和 attention_mask，其值为动态轴
        return OrderedDict(
            [
                ("input_ids", dynamic_axis),
                ("attention_mask", dynamic_axis),
            ]
        )
```