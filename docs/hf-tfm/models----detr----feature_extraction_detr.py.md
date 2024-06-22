# `.\models\detr\feature_extraction_detr.py`

```py
# 设置编码格式为 UTF-8
# 版权声明
# 版权所有 2021 年 HuggingFace Inc. 团队。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）获得许可;
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则软件
# 根据许可证“按原样”分发,
# 没有任何明示或暗示的保证或条件。
# 请参阅许可证以获取特定语言的许可证
# 简化和具体的权限。
"""DETR 的特征提取器类。"""

# 导入警告模块
import warnings

# 从 ...image_transforms 模块导入 rgb_to_id 函数并命名为 _rgb_to_id
from ...image_transforms import rgb_to_id as _rgb_to_id
# 导入日志模块
from ...utils import logging
# 从 .image_processing_detr 模块导入 DetrImageProcessor 类
from .image_processing_detr import DetrImageProcessor

# 获取日志记录器
logger = logging.get_logger(__name__)


# 定义 rgb_to_id 函数
def rgb_to_id(x):
    # 发出未来警告，告知 rgb_to_id 函数已经移动，并且从版本 5 开始无法从此模块导入。
    warnings.warn(
        "rgb_to_id has moved and will not be importable from this module from v5. "
        "Please import from transformers.image_transforms instead.",
        FutureWarning,
    )
    # 返回 _rgb_to_id 函数的调用结果
    return _rgb_to_id(x)


# 定义 DetrFeatureExtractor 类，继承自 DetrImageProcessor 类
class DetrFeatureExtractor(DetrImageProcessor):
    # 初始化方法
    def __init__(self, *args, **kwargs) -> None:
        # 发出未来警告，告知 DetrFeatureExtractor 类将在 Transformers 的版本 5 中被弃用并移除，请使用 DetrImageProcessor 替代。
        warnings.warn(
            "The class DetrFeatureExtractor is deprecated and will be removed in version 5 of Transformers."
            " Please use DetrImageProcessor instead.",
            FutureWarning,
        )
        # 调用父类的初始化方法，并传入参数
        super().__init__(*args, **kwargs)
```