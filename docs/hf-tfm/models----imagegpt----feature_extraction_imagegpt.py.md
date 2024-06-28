# `.\models\imagegpt\feature_extraction_imagegpt.py`

```
# 指定编码格式为 UTF-8，确保文件中的所有字符能够正确地被解析和处理
# 版权声明，指出 HuggingFace Inc. 团队保留所有权利
#
# 根据 Apache 许可证 2.0 版本的规定使用此文件
# 您只能在遵守许可证的前提下使用本文件
# 您可以从以下链接获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，本软件是基于"原样"提供的，不提供任何形式的担保或条件
# 无论是明示的还是暗示的，包括但不限于对特定用途的适销性和适用性的暗示担保。
# 有关许可证的详细信息，请参阅许可证文档。
"""ImageGPT 的特征提取器类。"""

# 引入警告模块，用于发出关于类过时的警告
import warnings

# 从 utils 模块中引入日志记录功能
from ...utils import logging

# 从 image_processing_imagegpt 模块导入 ImageGPTImageProcessor 类
from .image_processing_imagegpt import ImageGPTImageProcessor

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)


class ImageGPTFeatureExtractor(ImageGPTImageProcessor):
    def __init__(self, *args, **kwargs) -> None:
        # 发出警告：ImageGPTFeatureExtractor 类已弃用，将在 Transformers 版本 5 中移除，请使用 ImageGPTImageProcessor 替代。
        warnings.warn(
            "The class ImageGPTFeatureExtractor is deprecated and will be removed in version 5 of Transformers."
            " Please use ImageGPTImageProcessor instead.",
            FutureWarning,
        )
        # 调用父类 ImageGPTImageProcessor 的构造函数，传递所有位置参数和关键字参数
        super().__init__(*args, **kwargs)
```