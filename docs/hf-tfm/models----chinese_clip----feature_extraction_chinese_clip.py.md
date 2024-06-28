# `.\models\chinese_clip\feature_extraction_chinese_clip.py`

```py
# coding=utf-8
# 版权所有 2021 年 OFA-Sys 团队作者和 HuggingFace 团队。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）许可；
# 除非符合许可证的要求，否则您不能使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据“许可证”分发的软件是基于“原样”提供的，
# 不提供任何形式的明示或暗示的担保或条件。
# 有关详细信息，请参阅许可证。
"""Chinese-CLIP 的特征提取器类。"""

import warnings

from ...utils import logging
from .image_processing_chinese_clip import ChineseCLIPImageProcessor

# 获取 logger 对象
logger = logging.get_logger(__name__)

class ChineseCLIPFeatureExtractor(ChineseCLIPImageProcessor):
    def __init__(self, *args, **kwargs) -> None:
        # 发出未来警告，表明 ChineseCLIPFeatureExtractor 类将在 Transformers 的第五个版本中移除
        warnings.warn(
            "The class ChineseCLIPFeatureExtractor is deprecated and will be removed in version 5 of Transformers."
            " Please use ChineseCLIPImageProcessor instead.",
            FutureWarning,
        )
        # 调用父类的初始化方法
        super().__init__(*args, **kwargs)
```