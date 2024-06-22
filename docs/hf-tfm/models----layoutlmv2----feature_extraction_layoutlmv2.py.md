# `.\models\layoutlmv2\feature_extraction_layoutlmv2.py`

```py
# 设定脚本编码为 UTF-8
# 版权归 2021 年 HuggingFace Inc. 团队所有
#
# 根据 Apache 许可证 2.0 版本，您不得使用此文件，除非符合许可证的规定
# 您可以在以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证软件是基于“原样”分发的，
# 没有任何明示或暗示的担保或条件
# 请查看许可证，了解特定语言的权限和限制

# 导入必要的模块和库，并忽略警告信息
import warnings

from ...utils import logging
from .image_processing_layoutlmv2 import LayoutLMv2ImageProcessor

# 获取日志记录器的实例
logger = logging.get_logger(__name__)

# LayoutLMv2FeatureExtractor 类继承自 LayoutLMv2ImageProcessor 类
class LayoutLMv2FeatureExtractor(LayoutLMv2ImageProcessor):
    # 初始化方法，接受任意位置和关键字参数
    def __init__(self, *args, **kwargs) -> None:
        # 发出警告信息，表明 LayoutLMv2FeatureExtractor 类将在 Transformers 的第 5 版中被移除
        # 请改用 LayoutLMv2ImageProcessor 类
        warnings.warn(
            "The class LayoutLMv2FeatureExtractor is deprecated and will be removed in version 5 of Transformers."
            " Please use LayoutLMv2ImageProcessor instead.",
            FutureWarning,
        )
        # 调用父类的初始化方法
        super().__init__(*args, **kwargs)
```