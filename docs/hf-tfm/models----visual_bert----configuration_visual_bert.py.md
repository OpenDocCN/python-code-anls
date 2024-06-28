# `.\models\visual_bert\configuration_visual_bert.py`

```
# 设置文件编码为 UTF-8

# 版权声明和许可信息，声明代码版权归 HuggingFace Inc. 团队所有，并遵循 Apache License, Version 2.0
# 根据许可证，除非符合许可协议，否则不得使用此文件

# 引入 VisualBERT 模型的配置类 PretrainedConfig 和日志记录工具 logging
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取当前模块的日志记录器对象
logger = logging.get_logger(__name__)

# 定义 VisualBERT 预训练模型及其对应的配置文件下载地址映射表
VISUAL_BERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "uclanlp/visualbert-vqa": "https://huggingface.co/uclanlp/visualbert-vqa/resolve/main/config.json",
    "uclanlp/visualbert-vqa-pre": "https://huggingface.co/uclanlp/visualbert-vqa-pre/resolve/main/config.json",
    "uclanlp/visualbert-vqa-coco-pre": (
        "https://huggingface.co/uclanlp/visualbert-vqa-coco-pre/resolve/main/config.json"
    ),
    "uclanlp/visualbert-vcr": "https://huggingface.co/uclanlp/visualbert-vcr/resolve/main/config.json",
    "uclanlp/visualbert-vcr-pre": "https://huggingface.co/uclanlp/visualbert-vcr-pre/resolve/main/config.json",
    "uclanlp/visualbert-vcr-coco-pre": (
        "https://huggingface.co/uclanlp/visualbert-vcr-coco-pre/resolve/main/config.json"
    ),
    "uclanlp/visualbert-nlvr2": "https://huggingface.co/uclanlp/visualbert-nlvr2/resolve/main/config.json",
    "uclanlp/visualbert-nlvr2-pre": "https://huggingface.co/uclanlp/visualbert-nlvr2-pre/resolve/main/config.json",
    "uclanlp/visualbert-nlvr2-coco-pre": (
        "https://huggingface.co/uclanlp/visualbert-nlvr2-coco-pre/resolve/main/config.json"
    ),
    # 所有 VisualBERT 模型的列表可在 https://huggingface.co/models?filter=visual_bert 查看
}

# VisualBertConfig 类，继承自 PretrainedConfig 类
class VisualBertConfig(PretrainedConfig):
    r"""
    这是 VisualBERT 模型的配置类，用于存储 [`VisualBertModel`] 的配置信息。根据指定的参数实例化一个 VisualBERT 模型，
    定义模型架构。使用默认配置实例化一个配置对象将产生类似于 VisualBERT [uclanlp/visualbert-vqa-coco-pre]
    (https://huggingface.co/uclanlp/visualbert-vqa-coco-pre) 架构的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型的输出。阅读 [`PretrainedConfig`] 的文档以获取更多信息。

    Example:

    ```python
    >>> from transformers import VisualBertConfig, VisualBertModel

    >>> # 初始化一个 VisualBERT visualbert-vqa-coco-pre 风格的配置
    >>> configuration = VisualBertConfig.from_pretrained("uclanlp/visualbert-vqa-coco-pre")
    ```

    """
    # 初始化一个 VisualBertModel 模型，使用给定的配置参数
    model = VisualBertModel(configuration)

    # 访问模型的配置信息
    configuration = model.config
```