# `.\models\convnext\feature_extraction_convnext.py`

```py
# coding=utf-8
# 定义文件的编码格式为 UTF-8

# 版权声明，声明代码版权归 The HuggingFace Inc. 团队所有，保留所有权利。
# 根据 Apache 许可证版本 2.0 进行许可，除非符合许可证的要求，否则不得使用此文件。
# 可以在以下网址获取许可证的副本：http://www.apache.org/licenses/LICENSE-2.0

# 如果不符合适用法律或书面同意的要求，本软件是基于“按原样”提供，没有任何明示或暗示的担保或条件。
# 请参阅许可证了解具体的法律规定。

"""ConvNeXT 的特征提取器类。"""

# 导入警告模块
import warnings

# 导入日志工具
from ...utils import logging

# 从 image_processing_convnext 模块导入 ConvNextImageProcessor 类
from .image_processing_convnext import ConvNextImageProcessor

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)


class ConvNextFeatureExtractor(ConvNextImageProcessor):
    def __init__(self, *args, **kwargs) -> None:
        # 发出警告信息，提示 ConvNextFeatureExtractor 类已经弃用，并将在 Transformers 的第五个版本中删除。
        # 建议使用 ConvNextImageProcessor 替代。
        warnings.warn(
            "The class ConvNextFeatureExtractor is deprecated and will be removed in version 5 of Transformers."
            " Please use ConvNextImageProcessor instead.",
            FutureWarning,
        )
        # 调用父类的初始化方法，传递所有位置参数和关键字参数
        super().__init__(*args, **kwargs)
```