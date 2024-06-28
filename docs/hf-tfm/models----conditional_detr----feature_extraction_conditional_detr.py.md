# `.\models\conditional_detr\feature_extraction_conditional_detr.py`

```py
# 设置编码格式为 UTF-8
# 版权声明和许可条款，指明此代码的版权归 HuggingFace Inc. 团队所有，遵循 Apache License, Version 2.0
# 除非符合许可证要求，否则不得使用此文件
# 可以在以下网址获取许可证副本：http://www.apache.org/licenses/LICENSE-2.0
# 根据适用法律或书面同意，软件按"原样"提供，不提供任何明示或暗示的保证或条件
# 请查阅许可证了解具体的语言和限制
"""
Feature extractor class for Conditional DETR.
"""

# 导入警告模块
import warnings

# 从相关模块中导入 rgb_to_id 函数并重命名为 _rgb_to_id
from ...image_transforms import rgb_to_id as _rgb_to_id
# 从 utils 模块导入 logging 功能
from ...utils import logging
# 从当前目录下的 image_processing_conditional_detr 模块中导入 ConditionalDetrImageProcessor 类
from .image_processing_conditional_detr import ConditionalDetrImageProcessor

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 定义一个函数 rgb_to_id，用于将 RGB 图像转换为 ID（标识符）
def rgb_to_id(x):
    # 发出警告，说明 rgb_to_id 函数已移动，从 v5 版本开始将不再从此模块导入
    warnings.warn(
        "rgb_to_id has moved and will not be importable from this module from v5. "
        "Please import from transformers.image_transforms instead.",
        FutureWarning,
    )
    # 调用 _rgb_to_id 函数，执行 RGB 到 ID 的转换操作
    return _rgb_to_id(x)

# 定义一个特征提取器类 ConditionalDetrFeatureExtractor，继承自 ConditionalDetrImageProcessor 类
class ConditionalDetrFeatureExtractor(ConditionalDetrImageProcessor):
    # 初始化方法，接受任意位置参数和关键字参数
    def __init__(self, *args, **kwargs) -> None:
        # 发出警告，说明 ConditionalDetrFeatureExtractor 类已弃用，将在 Transformers 版本 5 中移除
        # 建议使用 ConditionalDetrImageProcessor 替代
        warnings.warn(
            "The class ConditionalDetrFeatureExtractor is deprecated and will be removed in version 5 of Transformers."
            " Please use ConditionalDetrImageProcessor instead.",
            FutureWarning,
        )
        # 调用父类 ConditionalDetrImageProcessor 的初始化方法，传入所有接收到的参数
        super().__init__(*args, **kwargs)
```