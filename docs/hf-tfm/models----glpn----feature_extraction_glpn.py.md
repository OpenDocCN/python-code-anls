# `.\models\glpn\feature_extraction_glpn.py`

```
# 设置文件编码为 UTF-8
# 版权声明信息
# 版权所有 © 2022 HuggingFace Inc. 团队。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）进行许可；
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证的副本
# http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是按“原样”基础分发的，没有任何保证或条件，无论是明示的还是暗示的。
# 请查看许可证以获取特定语言的参数和约束。
"""GLPN 的特征提取器类。"""

# 导入警告模块
import warnings

# 导入日志记录工具
from ...utils import logging
# 从 image_processing_glpn 模块中导入 GLPNImageProcessor 类
from .image_processing_glpn import GLPNImageProcessor

# 获取记录器对象
logger = logging.get_logger(__name__)

# GLPNFeatureExtractor 类继承自 GLPNImageProcessor 类
class GLPNFeatureExtractor(GLPNImageProcessor):
    def __init__(self, *args, **kwargs) -> None:
        # 发出警告信息，指出GLPNFeatureExtractor类已弃用，并将在 Transformers 的版本5中被移除，建议使用GLPNImageProcessor代替
        warnings.warn(
            "The class GLPNFeatureExtractor is deprecated and will be removed in version 5 of Transformers. Please"
            " use GLPNImageProcessor instead.",
            FutureWarning,
        )
        # 调用父类的初始化方法，并传入参数
        super().__init__(*args, **kwargs)
```