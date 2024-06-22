# `.\models\imagegpt\feature_extraction_imagegpt.py`

```py
# 设置文件编码为 UTF-8
# 版权声明，版权归 The HuggingFace Inc. 团队所有
# 根据 Apache 许可证 2.0 版本授权使用此文件
# 除非符合许可证要求，否则不得使用此文件
# 可以在以下网址获取许可证副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“按原样”分发的，没有任何明示或暗示的担保或条件
# 请查看许可证以获取有关特定语言的权限和限制
"""ImageGPT 的特征提取器类。"""

# 导入警告模块
import warnings

# 导入日志记录工具
from ...utils import logging
# 从 image_processing_imagegpt 模块中导入 ImageGPTImageProcessor 类
from .image_processing_imagegpt import ImageGPTImageProcessor

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义 ImageGPTFeatureExtractor 类，继承自 ImageGPTImageProcessor 类
class ImageGPTFeatureExtractor(ImageGPTImageProcessor):
    # 初始化方法
    def __init__(self, *args, **kwargs) -> None:
        # 发出警告，提示 ImageGPTFeatureExtractor 类已弃用，将在 Transformers 版本 5 中移除，请使用 ImageGPTImageProcessor 替代
        warnings.warn(
            "The class ImageGPTFeatureExtractor is deprecated and will be removed in version 5 of Transformers."
            " Please use ImageGPTImageProcessor instead.",
            FutureWarning,
        )
        # 调用父类的初始化方法
        super().__init__(*args, **kwargs)
```