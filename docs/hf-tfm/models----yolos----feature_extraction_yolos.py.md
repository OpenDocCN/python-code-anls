# `.\transformers\models\yolos\feature_extraction_yolos.py`

```
# 指定编码格式为utf-8
# 版权声明
"""Feature extractor class for YOLOS."""

# 导入警告模块
import warnings

# 从特定模块中导入特定函数
# 将rgb_to_id重命名为_rgb_to_id
from ...image_transforms import rgb_to_id as _rgb_to_id
# 导入日志记录模块
from ...utils import logging
# 从.image_processing_yolos模块中导入YolosImageProcessor类
from .image_processing_yolos import YolosImageProcessor

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义函数rgb_to_id，传入参数x
def rgb_to_id(x):
    # 发出警告
    warnings.warn(
        "rgb_to_id has moved and will not be importable from this module from v5. "
        "Please import from transformers.image_transforms instead.",
        FutureWarning,
    )
    return _rgb_to_id(x)

# 定义类YolosFeatureExtractor，继承自YolosImageProcessor类
class YolosFeatureExtractor(YolosImageProcessor):
    # 初始化方法，接受任意位置参数和关键词参数
    def __init__(self, *args, **kwargs) -> None:
        # 发出警告
        warnings.warn(
            "The class YolosFeatureExtractor is deprecated and will be removed in version 5 of Transformers. Please"
            " use YolosImageProcessor instead.",
            FutureWarning,
        )
        # 调用父类的初始化方法
        super().__init__(*args, **kwargs)
```