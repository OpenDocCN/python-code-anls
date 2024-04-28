# `.\transformers\models\vit\feature_extraction_vit.py`

```
# 设置文件编码为 UTF-8
# 版权声明，版权归 The HuggingFace Inc. 团队所有
# 根据 Apache 许可证 2.0 版本，除非符合许可证规定，否则不得使用此文件
# 可以在以下网址获取许可证的副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“原样”分发的，没有任何明示或暗示的担保或条件
# 请查看许可证以获取有关特定语言的权限和限制
"""ViT 的特征提取器类。"""

# 导入警告模块
import warnings

# 导入日志记录工具
from ...utils import logging
# 从 image_processing_vit 模块导入 ViTImageProcessor 类
from .image_processing_vit import ViTImageProcessor

# 获取 logger 对象
logger = logging.get_logger(__name__)

# ViTFeatureExtractor 类继承自 ViTImageProcessor 类
class ViTFeatureExtractor(ViTImageProcessor):
    # 初始化方法
    def __init__(self, *args, **kwargs) -> None:
        # 发出警告，提示 ViTFeatureExtractor 类已弃用，将在 Transformers 的第 5 版中移除，请使用 ViTImageProcessor 替代
        warnings.warn(
            "The class ViTFeatureExtractor is deprecated and will be removed in version 5 of Transformers. Please"
            " use ViTImageProcessor instead.",
            FutureWarning,
        )
        # 调用父类的初始化方法
        super().__init__(*args, **kwargs)
```