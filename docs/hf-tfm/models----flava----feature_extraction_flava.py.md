# `.\models\flava\feature_extraction_flava.py`

```py
# 导入警告模块
import warnings

# 导入日志记录模块
from ...utils import logging

# 从图像处理模块中导入 FLAVA 图像处理器类
from .image_processing_flava import FlavaImageProcessor

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义 FLAVA 特征提取器类，继承自 FLAVA 图像处理器类
class FlavaFeatureExtractor(FlavaImageProcessor):
    
    # 初始化方法
    def __init__(self, *args, **kwargs) -> None:
        
        # 发出警告，提示 FlavaFeatureExtractor 类已被弃用，并将在 Transformers 版本 5 中移除，建议使用 FlavaImageProcessor 替代
        warnings.warn(
            "The class FlavaFeatureExtractor is deprecated and will be removed in version 5 of Transformers. Please"
            " use FlavaImageProcessor instead.",
            FutureWarning,
        )
        
        # 调用父类的初始化方法
        super().__init__(*args, **kwargs)
```