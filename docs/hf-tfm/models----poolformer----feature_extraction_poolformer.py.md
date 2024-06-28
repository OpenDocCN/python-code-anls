# `.\models\poolformer\feature_extraction_poolformer.py`

```py
# 设置文件编码为 UTF-8
# 版权声明，版权归 HuggingFace Inc. 团队所有，保留所有权利
#
# 根据 Apache 许可证版本 2.0 许可，除非符合许可，否则不得使用此文件
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件是基于“按原样提供”的基础分发的，无任何明示或暗示的担保或条件
# 请查阅许可证了解详细信息
"""PoolFormer 的特征提取器类。"""

# 导入警告模块
import warnings

# 导入日志工具
from ...utils import logging
# 从 image_processing_poolformer 模块中导入 PoolFormerImageProcessor 类
from .image_processing_poolformer import PoolFormerImageProcessor

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# PoolFormerFeatureExtractor 类继承自 PoolFormerImageProcessor 类
class PoolFormerFeatureExtractor(PoolFormerImageProcessor):
    # 初始化方法，接受任意参数和关键字参数
    def __init__(self, *args, **kwargs) -> None:
        # 发出未来警告，表明 PoolFormerFeatureExtractor 类将在 Transformers 版本 5 中被删除
        warnings.warn(
            "The class PoolFormerFeatureExtractor is deprecated and will be removed in version 5 of Transformers."
            " Please use PoolFormerImageProcessor instead.",
            FutureWarning,
        )
        # 调用父类的初始化方法
        super().__init__(*args, **kwargs)
```