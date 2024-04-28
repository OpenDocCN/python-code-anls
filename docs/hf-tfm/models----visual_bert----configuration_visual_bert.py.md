# `.\transformers\models\visual_bert\configuration_visual_bert.py`

```py
# 设置文件编码为 utf-8
# 版权声明
# 根据 Apache 许可证 2.0 版本，授权使用此文件
# 只有在遵守许可证的情况下才能使用此文件
# 可以在以下网址获取许可证的副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"AS IS"的基础，没有任何明示或暗示的担保或条件
# 请查看许可证以获取有关特定语言的权限和限制

""" VisualBERT 模型配置"""

# 导入必要的模块和类
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 预训练配置文件映射
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
    # 查看所有 VisualBERT 模型 https://huggingface.co/models?filter=visual_bert
}

# VisualBERT 配置类，继承自 PretrainedConfig
class VisualBertConfig(PretrainedConfig):
    r"""
    这是用于存储 [`VisualBertModel`] 配置的类。用于根据指定的参数实例化 VisualBERT 模型，定义模型架构。
    使用默认值实例化配置将产生类似于 VisualBERT [uclanlp/visualbert-vqa-coco-pre](https://huggingface.co/uclanlp/visualbert-vqa-coco-pre) 架构的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型输出。阅读 [`PretrainedConfig`] 的文档以获取更多信息。

    示例:

    ```python
    >>> from transformers import VisualBertConfig, VisualBertModel

    >>> # 初始化一个 VisualBERT visualbert-vqa-coco-pre 风格的配置
    >>> configuration = VisualBertConfig.from_pretrained("uclanlp/visualbert-vqa-coco-pre")
    ```py
    # 初始化一个模型（带有随机权重），使用 visualbert-vqa-coco-pre 风格的配置
    model = VisualBertModel(configuration)
    
    # 访问模型配置
    configuration = model.config
```