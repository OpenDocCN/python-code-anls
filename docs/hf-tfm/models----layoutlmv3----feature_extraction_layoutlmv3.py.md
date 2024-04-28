# `.\models\layoutlmv3\feature_extraction_layoutlmv3.py`

```py
# 设置文件编码为 UTF-8，确保可以正确处理中文等字符
# 版权声明，使用 Apache License 2.0 授权
# 引入警告模块
# 引入日志记录模块
# 引入 LayoutLMv3 的图像处理模块

import warnings  # 导入警告模块
from ...utils import logging  # 从内部模块中引入日志记录模块
from .image_processing_layoutlmv3 import LayoutLMv3ImageProcessor  # 从 LayoutLMv3 图像处理模块引入 LayoutLMv3ImageProcessor 类

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

class LayoutLMv3FeatureExtractor(LayoutLMv3ImageProcessor):
    def __init__(self, *args, **kwargs) -> None:
        # 发出警告，指示 LayoutLMv3FeatureExtractor 类将在 Transformers 的第 5 版中被弃用并移除，建议使用 LayoutLMv3ImageProcessor 代替
        warnings.warn(
            "The class LayoutLMv3FeatureExtractor is deprecated and will be removed in version 5 of Transformers."
            " Please use LayoutLMv3ImageProcessor instead.",
            FutureWarning,
        )
        # 调用父类的初始化方法，传递参数
        super().__init__(*args, **kwargs)
```