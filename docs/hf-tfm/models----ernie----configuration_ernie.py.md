# `.\models\ernie\configuration_ernie.py`

```py
# 导入必要的模块和类
from collections import OrderedDict  # 导入OrderedDict类，用于有序字典
from typing import Mapping  # 导入Mapping类，用于类型提示

# 从transformers库中导入所需的配置类和模块
from ...configuration_utils import PretrainedConfig  # 导入预训练配置类
from ...onnx import OnnxConfig  # 导入ONNX配置类
from ...utils import logging  # 导入日志模块

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# ERNIE预训练模型配置的映射表，每个模型名称对应其配置文件的URL
ERNIE_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "nghuyong/ernie-1.0-base-zh": "https://huggingface.co/nghuyong/ernie-1.0-base-zh/resolve/main/config.json",
    "nghuyong/ernie-2.0-base-en": "https://huggingface.co/nghuyong/ernie-2.0-base-en/resolve/main/config.json",
    "nghuyong/ernie-2.0-large-en": "https://huggingface.co/nghuyong/ernie-2.0-large-en/resolve/main/config.json",
    "nghuyong/ernie-3.0-base-zh": "https://huggingface.co/nghuyong/ernie-3.0-base-zh/resolve/main/config.json",
    "nghuyong/ernie-3.0-medium-zh": "https://huggingface.co/nghuyong/ernie-3.0-medium-zh/resolve/main/config.json",
    "nghuyong/ernie-3.0-mini-zh": "https://huggingface.co/nghuyong/ernie-3.0-mini-zh/resolve/main/config.json",
    "nghuyong/ernie-3.0-micro-zh": "https://huggingface.co/nghuyong/ernie-3.0-micro-zh/resolve/main/config.json",
    "nghuyong/ernie-3.0-nano-zh": "https://huggingface.co/nghuyong/ernie-3.0-nano-zh/resolve/main/config.json",
    "nghuyong/ernie-gram-zh": "https://huggingface.co/nghuyong/ernie-gram-zh/resolve/main/config.json",
    "nghuyong/ernie-health-zh": "https://huggingface.co/nghuyong/ernie-health-zh/resolve/main/config.json",
}

# 定义ERINE配置类，继承自PretrainedConfig类
class ErnieConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ErnieModel`] or a [`TFErnieModel`]. It is used to
    instantiate a ERNIE model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the ERNIE
    [nghuyong/ernie-3.0-base-zh](https://huggingface.co/nghuyong/ernie-3.0-base-zh) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    """

    # 示例用法
    # 实例化一个与[nghuyong/ernie-3.0-base-zh](https://huggingface.co/nghuyong/ernie-3.0-base-zh)类似的配置
    # 参数设置为默认值将产生与ERNIE [nghuyong/ernie-3.0-base-zh](https://huggingface.co/nghuyong/ernie-3.0-base-zh)架构类似的配置

    # 示例代码
    # ```
    # >>> from transformers import ErnieConfig, ErnieModel
    #
    # >>> # Initializing a ERNIE nghuyong/ernie-3.0-base-zh style configuration
    # >>> configuration = ErnieConfig()
    # ```
    # 设置模型类型为ERNIE
    model_type = "ernie"
    
    # 定义ERNIE模型类
    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        task_type_vocab_size=3,
        use_task_id=False,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        position_embedding_type="absolute",
        use_cache=True,
        classifier_dropout=None,
        **kwargs,
    ):
        # 调用父类的构造函数，初始化模型
        super().__init__(pad_token_id=pad_token_id, **kwargs)
    
        # 初始化模型参数
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.task_type_vocab_size = task_type_vocab_size
        self.use_task_id = use_task_id
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout
class ErnieOnnxConfig(OnnxConfig):
    # 定义 Ernie 模型的配置类，继承自 OnnxConfig 类

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # 定义 inputs 属性，返回一个映射，其键为字符串，值为映射的映射，其中键为整数，值为字符串

        if self.task == "multiple-choice":
            # 如果任务类型是多项选择
            dynamic_axis = {0: "batch", 1: "choice", 2: "sequence"}
        else:
            # 否则
            dynamic_axis = {0: "batch", 1: "sequence"}

        return OrderedDict(
            [
                ("input_ids", dynamic_axis),  # 返回包含 input_ids 的动态轴映射
                ("attention_mask", dynamic_axis),  # 返回包含 attention_mask 的动态轴映射
                ("token_type_ids", dynamic_axis),  # 返回包含 token_type_ids 的动态轴映射
                ("task_type_ids", dynamic_axis),  # 返回包含 task_type_ids 的动态轴映射
            ]
        )
```