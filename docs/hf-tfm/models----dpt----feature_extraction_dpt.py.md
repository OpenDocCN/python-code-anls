# `.\models\dpt\feature_extraction_dpt.py`

```py
# 设置编码格式为 utf-8

# 版权声明及许可证信息，保留版权声明，提供 Apache License, Version 2.0 的获取链接

# 导入警告模块
import warnings

# 导入 logging 模块，用于记录日志
from ...utils import logging

# 从 .image_processing_dpt 模块中导入 DPTImageProcessor 类
from .image_processing_dpt import DPTImageProcessor

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 定义 DPTFeatureExtractor 类，继承自 DPTImageProcessor 类
class DPTFeatureExtractor(DPTImageProcessor):
    # 初始化方法
    def __init__(self, *args, **kwargs) -> None:
        # 发出警告，提示 DPTFeatureExtractor 类已弃用，并将在 Transformers 版本 5 中移除
        warnings.warn(
            "The class DPTFeatureExtractor is deprecated and will be removed in version 5 of Transformers. Please"
            " use DPTImageProcessor instead.",
            FutureWarning,
        )
        # 调用父类的初始化方法
        super().__init__(*args, **kwargs)
```