# `.\models\mask2former\configuration_mask2former.py`

```py
# 设置文件编码为 UTF-8
# 版权声明，指明版权归属 Meta Platforms, Inc. 和 The HuggingFace Inc. team，保留所有权利
#
# 根据 Apache 许可证 2.0 版本授权，除非符合许可证规定，否则不得使用此文件
# 可以在以下链接获取许可证副本：http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按"原样"分发本软件
# 没有任何形式的明示或暗示担保或条件。详情请参阅许可证
""" Mask2Former 模型配置"""

# 从 typing 导入所需的类型注解
from typing import Dict, List, Optional

# 导入预训练配置类 PretrainedConfig
from ...configuration_utils import PretrainedConfig
# 导入日志记录工具
from ...utils import logging
# 从自动模块中导入配置映射
from ..auto import CONFIG_MAPPING

# Mask2Former 预训练配置映射表，包含预训练模型及其配置文件的链接
MASK2FORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/mask2former-swin-small-coco-instance": (
        "https://huggingface.co/facebook/mask2former-swin-small-coco-instance/blob/main/config.json"
    )
    # 查看所有 Mask2Former 模型的链接：https://huggingface.co/models?filter=mask2former
}

# 获取日志记录器
logger = logging.get_logger(__name__)

# Mask2FormerConfig 类继承自 PretrainedConfig 类，用于存储 Mask2Former 模型的配置信息
class Mask2FormerConfig(PretrainedConfig):
    r"""
    这是用于存储 [`Mask2FormerModel`] 配置的配置类。根据指定的参数实例化 Mask2Former 模型，定义模型架构。
    使用默认参数实例化配置将生成类似于 Mask2Former [facebook/mask2former-swin-small-coco-instance] 
    (https://huggingface.co/facebook/mask2former-swin-small-coco-instance) 架构的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型的输出。阅读 [`PretrainedConfig`] 的文档以获取更多信息。

    当前，Mask2Former 仅支持 [Swin Transformer](swin) 作为主干。

    示例：

    ```
    >>> from transformers import Mask2FormerConfig, Mask2FormerModel

    >>> # 初始化 Mask2Former facebook/mask2former-swin-small-coco-instance 配置
    >>> configuration = Mask2FormerConfig()

    >>> # 使用配置初始化模型（带有随机权重），使用 facebook/mask2former-swin-small-coco-instance 风格的配置
    >>> model = Mask2FormerModel(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config
    ```

    """
    # 模型类型为 "mask2former"
    model_type = "mask2former"
    # 支持的主干为 ["swin"]
    backbones_supported = ["swin"]
    # 属性映射表，将 "hidden_size" 映射到 "hidden_dim"
    attribute_map = {"hidden_size": "hidden_dim"}
    @classmethod
    def from_backbone_config(cls, backbone_config: PretrainedConfig, **kwargs):
        """从预训练的骨干模型配置中实例化一个 [`Mask2FormerConfig`]（或其派生类）对象。

        Args:
            backbone_config ([`PretrainedConfig`]):
                骨干模型的配置对象。

        Returns:
            [`Mask2FormerConfig`]: 返回一个配置对象的实例
        """
        # 使用给定的骨干模型配置实例化一个新的 `Mask2FormerConfig` 对象，并传递额外的关键字参数
        return cls(
            backbone_config=backbone_config,
            **kwargs,
        )
```