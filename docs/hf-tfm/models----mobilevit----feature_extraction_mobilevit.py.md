# `.\models\mobilevit\feature_extraction_mobilevit.py`

```
# 设置文件编码为 UTF-8
# 版权声明和许可证信息，声明代码版权归 HuggingFace Inc. 团队所有
#
# 根据 Apache License, Version 2.0 许可证使用本文件
# 除非符合许可证要求，否则不得使用本文件
# 可以从以下链接获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按"原样"分发软件
# 没有任何明示或暗示的担保或条件
# 请查阅许可证了解具体的法律权利和限制
"""MobileViT 的特征提取器类。"""

# 导入警告模块
import warnings

# 从 utils 模块导入 logging
from ...utils import logging
# 从 image_processing_mobilevit 模块导入 MobileViTImageProcessor 类
from .image_processing_mobilevit import MobileViTImageProcessor

# 获取 logger 对象用于记录日志
logger = logging.get_logger(__name__)


class MobileViTFeatureExtractor(MobileViTImageProcessor):
    def __init__(self, *args, **kwargs) -> None:
        # 发出警告，表明 MobileViTFeatureExtractor 类已被弃用，将在 Transformers 版本 5 中移除，请使用 MobileViTImageProcessor 替代
        warnings.warn(
            "The class MobileViTFeatureExtractor is deprecated and will be removed in version 5 of Transformers."
            " Please use MobileViTImageProcessor instead.",
            FutureWarning,
        )
        # 调用父类 MobileViTImageProcessor 的构造函数，传递所有位置参数和关键字参数
        super().__init__(*args, **kwargs)
```