# `.\models\glpn\feature_extraction_glpn.py`

```py
# 设置文件编码为 UTF-8
# 版权声明和许可协议
# 版权所有 2022 年 HuggingFace Inc. 团队。保留所有权利。
#
# 根据 Apache 许可证版本 2.0 进行许可；
# 除非符合许可协议的规定，否则不得使用此文件。
# 您可以在以下网址获取许可协议的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按“原样”分发本软件，
# 没有任何明示或暗示的担保或条件。
# 有关许可协议的详细信息，请参阅许可协议。
"""GLPN 的特征提取器类。"""

# 引入警告模块
import warnings

# 引入日志模块
from ...utils import logging
# 从本地模块中引入 GLPNImageProcessor 类
from .image_processing_glpn import GLPNImageProcessor

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# GLPNFeatureExtractor 类继承自 GLPNImageProcessor 类
class GLPNFeatureExtractor(GLPNImageProcessor):
    def __init__(self, *args, **kwargs) -> None:
        # 发出警告信息，表明 GLPNFeatureExtractor 类在 Transformers 版本 5 中将被弃用并移除，建议使用 GLPNImageProcessor 代替
        warnings.warn(
            "The class GLPNFeatureExtractor is deprecated and will be removed in version 5 of Transformers. Please"
            " use GLPNImageProcessor instead.",
            FutureWarning,
        )
        # 调用父类的初始化方法
        super().__init__(*args, **kwargs)
```