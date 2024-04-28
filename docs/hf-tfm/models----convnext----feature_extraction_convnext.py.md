# `.\models\convnext\feature_extraction_convnext.py`

```
# 指定文件编码为 UTF-8

# 版权声明
# 2022年 HuggingFace Inc. 团队保留所有权利。
#
# 根据 Apache 许可证 2.0 版（“许可证”）进行许可；
# 除非符合许可证，否则不能使用此文件。
# 可以在以下网址获得许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，按“原样”分发的软件
# 没有任何明示或暗示担保或条件。
# 请参阅许可证以了解特定语言的管理权限和限制事项。
"""ConvNeXT 的特征提取器类。"""

# 引入警告模块
import warnings

# 引入 logging 模块
from ...utils import logging
# 引入 ConvNextImageProcessor 类
from .image_processing_convnext import ConvNextImageProcessor

# 获取 logger 对象
logger = logging.get_logger(__name__)


# 定义 ConvNextFeatureExtractor 类，继承自 ConvNextImageProcessor 类
class ConvNextFeatureExtractor(ConvNextImageProcessor):
    def __init__(self, *args, **kwargs) -> None:
        # 发出警告，提醒用户该类已被弃用，并在 Transformers 版本 5 中将被删除
        warnings.warn(
            "The class ConvNextFeatureExtractor is deprecated and will be removed in version 5 of Transformers."
            " Please use ConvNextImageProcessor instead.",
            FutureWarning,
        )
        
        # 调用父类的初始化方法
        super().__init__(*args, **kwargs)
```