# `.\transformers\models\xlm\configuration_xlm.py`

```
# 设定代码文件的编码为 UTF-8
# 版权声明，著作权归Facebook公司、HuggingFace Inc.团队所有
# 在遵守Apache License 2.0的情况下方可使用本文件
# 可以在以下链接获取License的一份副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除了遵循适用法律或书面同意外，本软件分发遵循"AS IS"基础，没有任何明示或暗示的担保或条件
# 请查看License了解具体语言规定的权限和限制
"""XLM配置"""
# 导入必要的库
from collections import OrderedDict
from typing import Mapping

# 导入预训练配置和Onnx配置
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 预训练模型的配置文件映射
XLM_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "xlm-mlm-en-2048": "https://huggingface.co/xlm-mlm-en-2048/resolve/main/config.json",
    "xlm-mlm-ende-1024": "https://huggingface.co/xlm-mlm-ende-1024/resolve/main/config.json",
    "xlm-mlm-enfr-1024": "https://huggingface.co/xlm-mlm-enfr-1024/resolve/main/config.json",
    "xlm-mlm-enro-1024": "https://huggingface.co/xlm-mlm-enro-1024/resolve/main/config.json",
    "xlm-mlm-tlm-xnli15-1024": "https://huggingface.co/xlm-mlm-tlm-xnli15-1024/resolve/main/config.json",
    "xlm-mlm-xnli15-1024": "https://huggingface.co/xlm-mlm-xnli15-1024/resolve/main/config.json",
    "xlm-clm-enfr-1024": "https://huggingface.co/xlm-clm-enfr-1024/resolve/main/config.json",
    "xlm-clm-ende-1024": "https://huggingface.co/xlm-clm-ende-1024/resolve/main/config.json",
    "xlm-mlm-17-1280": "https://huggingface.co/xlm-mlm-17-1280/resolve/main/config.json",
    "xlm-mlm-100-1280": "https://huggingface.co/xlm-mlm-100-1280/resolve/main/config.json",
}

# XLM配置类，继承自PretrainedConfig
class XLMConfig(PretrainedConfig):
    """
    这是用于存储[`XLMModel`]或[`TFXLMModel`]配置的类。它用于根据指定的参数实例化XLM模型，定义模型架构。
    设置默认参数实例化一个与[xlm-mlm-en-2048](https://huggingface.co/xlm-mlm-en-2048)架构类似的配置对象。

    配置对象从[`PretrainedConfig`]继承，可用于控制模型输出。
    阅读[`PretrainedConfig`]的文档以获取更多信息。

    例子：

    ```python
    >>> from transformers import XLMConfig, XLMModel

    >>> # 初始化XLM配置
    >>> configuration = XLMConfig()

    >>> # 从配置实例化一个模型（带有随机权重）
    >>> model = XLMModel(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config
    ```
    """

    # 模型类型为"xlm"
    model_type = "xlm"
    # 将属性名映射为配置字典中的键，用于处理不同命名的属性
    attribute_map = {
        "hidden_size": "emb_dim",
        "num_attention_heads": "n_heads",
        "num_hidden_layers": "n_layers",
        "n_words": "vocab_size",  # 用于向后兼容
    }

    def __init__(
        self,
        vocab_size=30145,  # 词汇表大小，默认值为30145
        emb_dim=2048,  # 嵌入维度，默认值为2048
        n_layers=12,  # 隐藏层层数，默认值为12
        n_heads=16,  # 注意力头数，默认值为16
        dropout=0.1,  # 丢弃率，默认值为0.1
        attention_dropout=0.1,  # 注意力丢弃率，默认值为0.1
        gelu_activation=True,  # 是否使用GELU激活函数，默认值为True
        sinusoidal_embeddings=False,  # 是否使用正弦嵌入，默认值为False
        causal=False,  # 是否使用因果注意力，默认值为False
        asm=False,  # 是否使用自注意力机制，默认值为False
        n_langs=1,  # 语言数量，默认值为1
        use_lang_emb=True,  # 是否使用语言嵌入，默认值为True
        max_position_embeddings=512,  # 最大位置嵌入数量，默认值为512
        embed_init_std=2048**-0.5,  # 嵌入初始化标准差，默认值为2048的负0.5次方
        layer_norm_eps=1e-12,  # 层归一化的epsilon，默认值为1e-12
        init_std=0.02,  # 初始化标准差，默认值为0.02
        bos_index=0,  # 起始标记的索引，默认值为0
        eos_index=1,  # 终止标记的索引，默认值为1
        pad_index=2,  # 填充标记的索引，默认值为2
        unk_index=3,  # 未知标记的索引，默认值为3
        mask_index=5,  # 掩码标记的索引，默认值为5
        is_encoder=True,  # 是否为编码器，默认值为True
        summary_type="first",  # 摘要类型，默认值为"first"
        summary_use_proj=True,  # 是否投影到标签的摘要，默认值为True
        summary_activation=None,  # 摘要激活函数，默认为None
        summary_proj_to_labels=True,  # 是否投影到标签，默认值为True
        summary_first_dropout=0.1,  # 摘要的首个丢弃率，默认值为0.1
        start_n_top=5,  # 起始top值，默认值为5
        end_n_top=5,  # 终止top值，默认为5
        mask_token_id=0,  # 掩码标记的ID，默认值为0
        lang_id=0,  # 语言ID，默认值为0
        pad_token_id=2,  # 填充标记的ID，默认值为2
        bos_token_id=0,  # 起始标记的ID，默认值为0
        **kwargs,  # 多余的关键字参数
    ):
        """构造XLMConfig。"""
        # 初始化XLMConfig属性
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.gelu_activation = gelu_activation
        self.sinusoidal_embeddings = sinusoidal_embeddings
        self.causal = causal
        self.asm = asm
        self.n_langs = n_langs
        self.use_lang_emb = use_lang_emb
        self.max_position_embeddings = max_position_embeddings
        self.embed_init_std = embed_init_std
        self.layer_norm_eps = layer_norm_eps
        self.bos_index = bos_index
        self.eos_index = eos_index
        self.pad_index = pad_index
        self.unk_index = unk_index
        self.mask_index = mask_index
        self.is_encoder = is_encoder
        self.init_std = init_std
        self.summary_type = summary_type
        self.summary_use_proj = summary_use_proj
        self.summary_activation = summary_activation
        self.summary_proj_to_labels = summary_proj_to_labels
        self.summary_first_dropout = summary_first_dropout
        self.start_n_top = start_n_top
        self.end_n_top = end_n_top
        self.mask_token_id = mask_token_id
        self.lang_id = lang_id

        # 如果kwargs中包含'n_words'，则将其赋值给self.n_words属性
        if "n_words" in kwargs:
            self.n_words = kwargs["n_words"]

        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, **kwargs)  # 调用父类的初始化方法
# 从transformers.models.bert.configuration_bert.BertOnnxConfig中复制XLMOnnxConfig类
class XLMOnnxConfig(OnnxConfig):
    # 定义inputs属性，返回输入的键值对
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # 如果任务是"multiple-choice"，则设置动态轴为{0: "batch", 1: "choice", 2: "sequence"}
        if self.task == "multiple-choice":
            dynamic_axis = {0: "batch", 1: "choice", 2: "sequence"}
        # 否则设置动态轴为{0: "batch", 1: "sequence"}
        else:
            dynamic_axis = {0: "batch", 1: "sequence"}
        # 返回一个有序字典，包含键值对："input_ids"、"attention_mask"、"token_type_ids"，值为动态轴
        return OrderedDict(
            [
                ("input_ids", dynamic_axis),
                ("attention_mask", dynamic_axis),
                ("token_type_ids", dynamic_axis),
            ]
        )

```  
```