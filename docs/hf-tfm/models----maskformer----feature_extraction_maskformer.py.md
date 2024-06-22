# `.\transformers\models\maskformer\feature_extraction_maskformer.py`

```py
# 设置文件编码为utf-8

# 版权声明，版权归The HuggingFace Inc.团队所有
#
# 根据Apache许可证2.0版本（“许可证”）授权
# 您不得使用此文件，除非遵守许可证
# 您可以在以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则软件在“原样”基础上分发
# 没有任何种类的保证或条件，无论是明示还是默示
# 有关特定语言管理权限和
# 限制在许可证下。

"""MaskFormer的特征提取器类。"""

# 引入必要的库
import warnings

from ...utils import logging
from .image_processing_maskformer import MaskFormerImageProcessor

# 获取logger对象
logger = logging.get_logger(__name__)

# 定义特征提取器类，继承自MaskFormerImageProcessor类
class MaskFormerFeatureExtractor(MaskFormerImageProcessor):
    def __init__(self, *args, **kwargs) -> None:
        # 发出警告，提示MaskFormerFeatureExtractor类已弃用，并将在Transformers的第5版中移除
        warnings.warn(
            "The class MaskFormerFeatureExtractor is deprecated and will be removed in version 5 of Transformers."
            " Please use MaskFormerImageProcessor instead.",
            FutureWarning,
        )
        # 调用父类的初始化方法
        super().__init__(*args, **kwargs)
```