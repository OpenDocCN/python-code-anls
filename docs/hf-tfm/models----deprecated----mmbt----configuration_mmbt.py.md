# `.\models\deprecated\mmbt\configuration_mmbt.py`

```py
# 设置编码格式为 UTF-8
# 版权声明：来自 Facebook, Inc. 及其附属公司和 HuggingFace Inc. 团队
# 根据 Apache 许可证 2.0 版本获得许可
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非法律要求或书面同意，或者软件在 "AS IS" 基础上分发
# 没有任何形式的担保或条件，无论是明示或默示的
# 请参阅许可证以了解详细的权限和限制
"""MMBT 配置"""

from ....utils import logging

# 获取 logger 实例
logger = logging.get_logger(__name__)


# MMBT 配置类
class MMBTConfig(object):
    """
    这是配置类，用于存储 [`MMBTModel`] 的配置。根据指定的参数实例化 MMBT 模型，定义模型架构。

    Args:
        config ([`PreTrainedConfig`]): 潜在 Transformer 模型的配置。将其值复制以使用单个配置。
        num_labels (`int`, 可选): 用于分类的最终线性层的大小。
        modal_hidden_size (`int`, 可选`, 默认为 2048): 非文本模态编码器的嵌入维度。
    """

    # 初始化方法
    def __init__(self, config, num_labels=None, modal_hidden_size=2048):
        # 将当前对象的字典设置为配置对象的字典
        self.__dict__ = config.__dict__
        # 设置 modal_hidden_size 属性
        self.modal_hidden_size = modal_hidden_size
        # 如果存在 num_labels，则设置 num_labels 属性
        if num_labels:
            self.num_labels = num_labels
```