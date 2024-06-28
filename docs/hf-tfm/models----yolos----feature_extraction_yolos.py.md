# `.\models\yolos\feature_extraction_yolos.py`

```py
# 设置文件编码为 UTF-8
# 版权声明，版权归 The HuggingFace Inc. 团队所有，保留所有权利
#
# 根据 Apache 许可证 2.0 版本进行许可
# 除非符合许可证的要求，否则不得使用此文件
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件按“原样”提供，
# 没有任何明示或暗示的担保或条件
# 请参阅许可证了解具体的法律条款和条件

"""YOLOS 的特征提取器类。"""

# 导入警告模块
import warnings

# 导入 RGB 到 ID 转换函数，并重命名为 _rgb_to_id
from ...image_transforms import rgb_to_id as _rgb_to_id
# 导入日志模块
from ...utils import logging
# 导入 YolosImageProcessor 类，用于图像处理
from .image_processing_yolos import YolosImageProcessor

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)


def rgb_to_id(x):
    # 发出警告，提醒用户 rgb_to_id 函数已经移动，
    # 从版本 5 开始将不再从当前模块中导入，
    # 请从 transformers.image_transforms 中导入
    warnings.warn(
        "rgb_to_id has moved and will not be importable from this module from v5. "
        "Please import from transformers.image_transforms instead.",
        FutureWarning,
    )
    return _rgb_to_id(x)


class YolosFeatureExtractor(YolosImageProcessor):
    def __init__(self, *args, **kwargs) -> None:
        # 发出警告，提醒用户 YolosFeatureExtractor 类已经废弃，
        # 将在版本 5 中移除，请使用 YolosImageProcessor 代替
        warnings.warn(
            "The class YolosFeatureExtractor is deprecated and will be removed in version 5 of Transformers. Please"
            " use YolosImageProcessor instead.",
            FutureWarning,
        )
        # 调用父类 YolosImageProcessor 的初始化方法
        super().__init__(*args, **kwargs)
```