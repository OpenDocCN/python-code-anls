# `.\models\owlvit\feature_extraction_owlvit.py`

```
# 设置编码为 UTF-8，确保脚本能够正确处理 Unicode 字符串
# 版权声明，标明 HuggingFace Inc. 团队保留所有权利
#
# 根据 Apache 许可证版本 2.0 进行许可，除非符合许可证条款，否则不得使用此文件
# 可以在以下网址获取许可证副本：http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则不得根据此许可证分发本软件
# 本软件基于"按原样"基础提供，不提供任何明示或暗示的担保或条件
# 有关特定语言的权限，请参阅许可证
"""OwlViT 的特征提取器类。"""

# 导入警告模块，用于标记类已经被弃用
import warnings

# 从 utils 模块中导入 logging 功能
from ...utils import logging

# 从 image_processing_owlvit 模块中导入 OwlViTImageProcessor 类
from .image_processing_owlvit import OwlViTImageProcessor

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# OwlViTFeatureExtractor 类继承自 OwlViTImageProcessor 类
class OwlViTFeatureExtractor(OwlViTImageProcessor):
    def __init__(self, *args, **kwargs) -> None:
        # 发出警告，表明 OwlViTFeatureExtractor 类已被弃用，并将在 Transformers 的版本 5 中移除
        warnings.warn(
            "The class OwlViTFeatureExtractor is deprecated and will be removed in version 5 of Transformers. Please"
            " use OwlViTImageProcessor instead.",
            FutureWarning,
        )
        # 调用父类 OwlViTImageProcessor 的初始化方法
        super().__init__(*args, **kwargs)
```