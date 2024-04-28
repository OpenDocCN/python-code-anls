# `.\models\conditional_detr\feature_extraction_conditional_detr.py`

```
# 设置文件编码为utf-8
# 版权声明及许可证的声明
# 条件DETR功能提取器的类
import warnings  # 导入警告模块
from ...image_transforms import rgb_to_id as _rgb_to_id  # 从图像转换中导入rgb_to_id函数并重命名为_rgb_to_id
from ...utils import logging  # 从工具包中导入日志模块

# 获取logger实例
logger = logging.get_logger(__name__)

# 定义rgb_to_id函数并发出FutureWarning警告
def rgb_to_id(x):
    warnings.warn(
        "rgb_to_id has moved and will not be importable from this module from v5. "
        "Please import from transformers.image_transforms instead.",
        FutureWarning,
    )
    return _rgb_to_id(x)

# 条件DETR功能提取器类，继承自ConditionalDetrImageProcessor类
class ConditionalDetrFeatureExtractor(ConditionalDetrImageProcessor):
    # 初始化方法，发出FutureWarning警告
    def __init__(self, *args, **kwargs) -> None:
        warnings.warn(
            "The class ConditionalDetrFeatureExtractor is deprecated and will be removed in version 5 of Transformers."
            " Please use ConditionalDetrImageProcessor instead.",
            FutureWarning,
        )
        super().__init__(*args, **kwargs)  # 调用父类的初始化方法
```