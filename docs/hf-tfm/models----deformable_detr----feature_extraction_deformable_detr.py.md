# `.\models\deformable_detr\feature_extraction_deformable_detr.py`

```py
# 设置 Python 文件的编码格式为 UTF-8
# 版权声明，声明此代码版权归 HuggingFace Inc. 团队所有
#
# 根据 Apache 许可证 2.0 版本，除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件是基于"原样"提供的，不提供任何形式的担保或条件，
# 无论是明示的还是暗示的。有关详细信息，请参阅许可证。
"""Deformable DETR 的特征提取器类。"""

# 导入警告模块
import warnings

# 从本地模块中导入 rgb_to_id 函数，并重命名为 _rgb_to_id
from ...image_transforms import rgb_to_id as _rgb_to_id
# 导入日志记录工具
from ...utils import logging
# 从本地模块中导入 DeformableDetrImageProcessor 类
from .image_processing_deformable_detr import DeformableDetrImageProcessor

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)


def rgb_to_id(x):
    # 发出警告，提醒用户从版本 5 开始，不再从当前模块中导入 rgb_to_id 函数
    warnings.warn(
        "rgb_to_id has moved and will not be importable from this module from v5. "
        "Please import from transformers.image_transforms instead.",
        FutureWarning,
    )
    # 调用本地模块中的 _rgb_to_id 函数，执行颜色转换操作
    return _rgb_to_id(x)


class DeformableDetrFeatureExtractor(DeformableDetrImageProcessor):
    def __init__(self, *args, **kwargs) -> None:
        # 发出警告，提示 DeformableDetrFeatureExtractor 类在 Transformers 版本 5 中将被移除
        # 建议使用 DeformableDetrImageProcessor 类代替
        warnings.warn(
            "The class DeformableDetrFeatureExtractor is deprecated and will be removed in version 5 of Transformers."
            " Please use DeformableDetrImageProcessor instead.",
            FutureWarning,
        )
        # 调用父类的初始化方法，传入所有位置参数和关键字参数
        super().__init__(*args, **kwargs)
```