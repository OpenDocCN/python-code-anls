# `.\transformers\models\vilt\feature_extraction_vilt.py`

```py
# 设置编码为 UTF-8
# 版权声明，版权归 The HuggingFace Inc. 团队所有
# 根据 Apache 许可证版本 2.0 授权，除非符合许可证的规定，否则不得使用此文件
# 可以在以下网址获取许可证的副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则按"原样"分发软件
# 本软件按"原样"分发，不提供任何形式的明示或暗示的保证，包括但不限于适销性和特定用途的适用性保证。
# 有关详细信息，请参阅许可证

# 导入警告模块
import warnings
# 从当前目录下的 utils 模块中导入 logging 函数
from ...utils import logging
# 从 image_processing_vilt 模块中导入 ViltImageProcessor 类
from .image_processing_vilt import ViltImageProcessor

# 获取日志记录器对象
logger = logging.get_logger(__name__)

# 定义 ViltFeatureExtractor 类，继承自 ViltImageProcessor 类
class ViltFeatureExtractor(ViltImageProcessor):
    # 初始化函数，接受任意位置参数和关键字参数
    def __init__(self, *args, **kwargs) -> None:
        # 发出警告，提示 ViltFeatureExtractor 类已弃用，将在 Transformers 的第 5 版中移除，建议使用 ViltImageProcessor 代替
        warnings.warn(
            "The class ViltFeatureExtractor is deprecated and will be removed in version 5 of Transformers. Please"
            " use ViltImageProcessor instead.",
            FutureWarning,
        )
        # 调用父类 ViltImageProcessor 的初始化方法，并传入参数
        super().__init__(*args, **kwargs)
```