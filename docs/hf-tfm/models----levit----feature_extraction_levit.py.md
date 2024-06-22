# `.\transformers\models\levit\feature_extraction_levit.py`

```py
# 设置文件编码为 UTF-8
# 版权声明信息
# 版权所有©2022 Meta Platforms, Inc. 和 The HuggingFace Inc. 团队。保留所有权利。
#
# 根据 Apache 许可证第 2.0 版 (“许可证”) 授权;
# 你只能在符合许可证的情况下使用此文件。
# 你可以从以下网址获得许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 是基于“原样”分发的，没有任何明示或暗示的保修或条件。
# 有关特定语言的权限和限制，请参见许可证。
"""LeViT 的特征提取器类。"""

# 导入警告模块
import warnings

# 导入日志记录模块
from ...utils import logging
# 从 image_processing_levit.py 中导入 LevitImageProcessor 类
from .image_processing_levit import LevitImageProcessor

# 获取 logger 对象
logger = logging.get_logger(__name__)

# 定义 LevitFeatureExtractor 类，它继承自 LevitImageProcessor 类
class LevitFeatureExtractor(LevitImageProcessor):
    # 初始化方法，接收任意参数和关键字参数
    def __init__(self, *args, **kwargs) -> None:
        # 发出警告，说明 LevitFeatureExtractor 类已被弃用，并将在 Transformers 的第 5 版中删除。建议使用 LevitImageProcessor 代替。
        warnings.warn(
            "The class LevitFeatureExtractor is deprecated and will be removed in version 5 of Transformers. Please"
            " use LevitImageProcessor instead.",
            FutureWarning,
        )
        # 调用父类 LevitImageProcessor 的初始化方法，并传入所有参数
        super().__init__(*args, **kwargs)
```