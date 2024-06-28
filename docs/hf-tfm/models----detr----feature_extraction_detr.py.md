# `.\models\detr\feature_extraction_detr.py`

```
# 设置文件编码为 UTF-8
# 版权声明，指明版权归 HuggingFace Inc. 团队所有，保留所有权利
#
# 根据 Apache 许可证 2.0 版本，除非符合许可证要求，否则不得使用此文件
# 可在以下链接获取许可证副本：http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据此许可证分发的软件是基于 "原样" 提供的，
# 不附带任何明示或暗示的担保或条件。详见许可证文本获取更多信息。
"""DETR 的特征提取器类。"""

# 导入警告模块
import warnings

# 导入特定的图像转换函数 rgb_to_id，作为 _rgb_to_id 别名
from ...image_transforms import rgb_to_id as _rgb_to_id
# 导入日志工具
from ...utils import logging
# 导入 DETR 图像处理模块中的类 DetrImageProcessor
from .image_processing_detr import DetrImageProcessor

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)


def rgb_to_id(x):
    # 发出警告，指出 rgb_to_id 函数已经移动，从版本 5 开始将无法从当前模块导入
    warnings.warn(
        "rgb_to_id has moved and will not be importable from this module from v5. "
        "Please import from transformers.image_transforms instead.",
        FutureWarning,
    )
    # 调用 _rgb_to_id 函数并返回结果
    return _rgb_to_id(x)


class DetrFeatureExtractor(DetrImageProcessor):
    def __init__(self, *args, **kwargs) -> None:
        # 发出警告，指出 DetrFeatureExtractor 类在 Transformers 版本 5 中将被移除，建议使用 DetrImageProcessor 替代
        warnings.warn(
            "The class DetrFeatureExtractor is deprecated and will be removed in version 5 of Transformers."
            " Please use DetrImageProcessor instead.",
            FutureWarning,
        )
        # 调用父类 DetrImageProcessor 的构造函数，传入所有位置参数和关键字参数
        super().__init__(*args, **kwargs)
```