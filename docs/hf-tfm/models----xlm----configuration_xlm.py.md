# `.\models\xlm\configuration_xlm.py`

```
# 导入所需的模块和类
from collections import OrderedDict  # 导入有序字典模块
from typing import Mapping  # 导入 Mapping 类型提示

from ...configuration_utils import PretrainedConfig  # 导入预训练配置基类
from ...onnx import OnnxConfig  # 导入 ONNX 配置
from ...utils import logging  # 导入日志工具

# 获取 logger 对象，用于记录日志
logger = logging.get_logger(__name__)

# 预训练模型配置文件映射表，将模型名称映射到其配置文件的 URL
XLM_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "FacebookAI/xlm-mlm-en-2048": "https://huggingface.co/FacebookAI/xlm-mlm-en-2048/resolve/main/config.json",
    "FacebookAI/xlm-mlm-ende-1024": "https://huggingface.co/FacebookAI/xlm-mlm-ende-1024/resolve/main/config.json",
    "FacebookAI/xlm-mlm-enfr-1024": "https://huggingface.co/FacebookAI/xlm-mlm-enfr-1024/resolve/main/config.json",
    "FacebookAI/xlm-mlm-enro-1024": "https://huggingface.co/FacebookAI/xlm-mlm-enro-1024/resolve/main/config.json",
    "FacebookAI/xlm-mlm-tlm-xnli15-1024": "https://huggingface.co/FacebookAI/xlm-mlm-tlm-xnli15-1024/resolve/main/config.json",
    "FacebookAI/xlm-mlm-xnli15-1024": "https://huggingface.co/FacebookAI/xlm-mlm-xnli15-1024/resolve/main/config.json",
    "FacebookAI/xlm-clm-enfr-1024": "https://huggingface.co/FacebookAI/xlm-clm-enfr-1024/resolve/main/config.json",
    "FacebookAI/xlm-clm-ende-1024": "https://huggingface.co/FacebookAI/xlm-clm-ende-1024/resolve/main/config.json",
    "FacebookAI/xlm-mlm-17-1280": "https://huggingface.co/FacebookAI/xlm-mlm-17-1280/resolve/main/config.json",
    "FacebookAI/xlm-mlm-100-1280": "https://huggingface.co/FacebookAI/xlm-mlm-100-1280/resolve/main/config.json",
}

class XLMConfig(PretrainedConfig):
    """
    XLM 模型的配置类，用于存储 [`XLMModel`] 或 [`TFXLMModel`] 的配置信息。根据指定参数实例化一个 XLM 模型配置，定义模型的架构。
    使用默认值实例化配置将得到与 [FacebookAI/xlm-mlm-en-2048](https://huggingface.co/FacebookAI/xlm-mlm-en-2048) 架构类似的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型输出。阅读 [`PretrainedConfig`] 的文档以获取更多信息。

    Examples:

    ```python
    >>> from transformers import XLMConfig, XLMModel

    >>> # 初始化一个 XLM 配置
    >>> configuration = XLMConfig()

    >>> # 从配置初始化一个模型（随机权重）

    ```
    """
    # 创建一个 XLMModel 的实例，使用给定的 configuration 参数
    >>> model = XLMModel(configuration)

    # 访问模型配置信息
    >>> # Accessing the model configuration
    >>> configuration = model.config

    # 定义一个表示 XLM 模型类型的字符串变量
    model_type = "xlm"

    # 定义一个字典，将 XLM 模型属性名映射为别名
    attribute_map = {
        "hidden_size": "emb_dim",
        "num_attention_heads": "n_heads",
        "num_hidden_layers": "n_layers",
        "n_words": "vocab_size",  # 为了向后兼容性
    }

    # XLMConfig 类的构造函数，初始化 XLM 模型的各项配置参数
    def __init__(
        self,
        vocab_size=30145,
        emb_dim=2048,
        n_layers=12,
        n_heads=16,
        dropout=0.1,
        attention_dropout=0.1,
        gelu_activation=True,
        sinusoidal_embeddings=False,
        causal=False,
        asm=False,
        n_langs=1,
        use_lang_emb=True,
        max_position_embeddings=512,
        embed_init_std=2048**-0.5,
        layer_norm_eps=1e-12,
        init_std=0.02,
        bos_index=0,
        eos_index=1,
        pad_index=2,
        unk_index=3,
        mask_index=5,
        is_encoder=True,
        summary_type="first",
        summary_use_proj=True,
        summary_activation=None,
        summary_proj_to_labels=True,
        summary_first_dropout=0.1,
        start_n_top=5,
        end_n_top=5,
        mask_token_id=0,
        lang_id=0,
        pad_token_id=2,
        bos_token_id=0,
        **kwargs,
    ):
        """Constructs XLMConfig."""
        # 初始化 XLMConfig 对象的各个配置参数
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
        self.layer_norm_eps = layer_norm_eps
        self.bos_index = bos_index
        self.eos_index = eos_index
        self.pad_index = pad_index
        self.unk_index = unk_index
        self.mask_index = mask_index
        self.is_encoder = is_encoder
        self.max_position_embeddings = max_position_embeddings
        self.embed_init_std = embed_init_std
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

        # 如果 kwargs 中包含 'n_words' 参数，将其赋值给 self.n_words
        if "n_words" in kwargs:
            self.n_words = kwargs["n_words"]

        # 调用父类的构造函数，初始化基类的一些参数，如 pad_token_id 和 bos_token_id
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, **kwargs)
# 从 transformers.models.bert.configuration_bert.BertOnnxConfig 复制过来的 XLMOnnxConfig 类
class XLMOnnxConfig(OnnxConfig):
    # 定义 inputs 属性，返回一个映射类型，其键为字符串，值为映射类型，映射类型的键为整数，值为字符串
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # 如果任务类型是 "multiple-choice"
        if self.task == "multiple-choice":
            # 设置动态轴的映射，0 对应 "batch"，1 对应 "choice"，2 对应 "sequence"
            dynamic_axis = {0: "batch", 1: "choice", 2: "sequence"}
        else:
            # 否则设置动态轴的映射，0 对应 "batch"，1 对应 "sequence"
            dynamic_axis = {0: "batch", 1: "sequence"}
        # 返回一个有序字典，包含三个键值对，键为字符串，值为 dynamic_axis 映射
        return OrderedDict(
            [
                ("input_ids", dynamic_axis),         # 键为 "input_ids"，值为 dynamic_axis 映射
                ("attention_mask", dynamic_axis),    # 键为 "attention_mask"，值为 dynamic_axis 映射
                ("token_type_ids", dynamic_axis),    # 键为 "token_type_ids"，值为 dynamic_axis 映射
            ]
        )
```