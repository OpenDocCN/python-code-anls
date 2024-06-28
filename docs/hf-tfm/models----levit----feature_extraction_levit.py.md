# `.\models\levit\feature_extraction_levit.py`

```py
# coding=utf-8
# 版权 2022 年 Meta Platforms, Inc. 和 HuggingFace Inc. 团队。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）许可；
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据“许可证”分发的软件是基于“按原样提供”的基础上分发的，
# 没有任何明示或暗示的保证或条件。
# 有关具体语言的详细信息，请参阅许可证。
"""LeViT 的特征提取器类。"""

# 导入警告模块
import warnings

# 从当前包的 utils 模块中导入 logging 功能
from ...utils import logging
# 从本地模块中导入 LevitImageProcessor 类
from .image_processing_levit import LevitImageProcessor

# 获取 logger 对象
logger = logging.get_logger(__name__)

# 定义 LevitFeatureExtractor 类，继承自 LevitImageProcessor 类
class LevitFeatureExtractor(LevitImageProcessor):
    
    # 初始化方法，接受任意位置参数和关键字参数，并发出警告
    def __init__(self, *args, **kwargs) -> None:
        # 发出警告，提醒 LevitFeatureExtractor 类即将在 Transformers 的第五个版本中被移除，建议使用 LevitImageProcessor 替代
        warnings.warn(
            "The class LevitFeatureExtractor is deprecated and will be removed in version 5 of Transformers. Please"
            " use LevitImageProcessor instead.",
            FutureWarning,
        )
        # 调用父类（LevitImageProcessor）的初始化方法
        super().__init__(*args, **kwargs)
```