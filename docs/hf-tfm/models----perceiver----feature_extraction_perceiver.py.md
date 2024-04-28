# `.\transformers\models\perceiver\feature_extraction_perceiver.py`

```
# 设置文件编码为 UTF-8
# 版权声明，版权归 The HuggingFace Inc. 团队所有，保留所有权利
#
# 根据 Apache 许可证，版本 2.0 进行许可
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件根据“原样”分发
# 没有任何形式的担保或条件，明示或暗示
# 有关特定语言的权限，请参阅许可证
import warnings  # 导入警告模块

from ...utils import logging  # 从工具包中导入日志模块
from .image_processing_perceiver import PerceiverImageProcessor  # 从图像处理的 Perceiver 模块导入 PerceiverImageProcessor 类


logger = logging.get_logger(__name__)  # 获取日志记录器


# PerceiverFeatureExtractor 类继承自 PerceiverImageProcessor 类
class PerceiverFeatureExtractor(PerceiverImageProcessor):
    # 构造函数，接受任意数量的位置参数和关键字参数
    def __init__(self, *args, **kwargs) -> None:
        # 发出警告，指示 PerceiverFeatureExtractor 类已被弃用，并将在 Transformers 版本 5 中移除，建议使用 PerceiverImageProcessor 类
        warnings.warn(
            "The class PerceiverFeatureExtractor is deprecated and will be removed in version 5 of Transformers."
            " Please use PerceiverImageProcessor instead.",
            FutureWarning,
        )
        # 调用父类 PerceiverImageProcessor 的构造函数
        super().__init__(*args, **kwargs)
```