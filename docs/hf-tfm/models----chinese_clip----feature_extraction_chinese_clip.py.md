# `.\transformers\models\chinese_clip\feature_extraction_chinese_clip.py`

```
# 设置文件编码为 UTF-8
# 版权声明，版权归 OFA-Sys 团队作者和 HuggingFace 团队所有
# 根据 Apache 许可证 2.0 版本，除非符合许可证规定，否则不得使用此文件
# 您可以在以下网址获取许可证副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“原样”基础分发的，没有任何明示或暗示的担保或条件
# 请查看许可证以获取有关权限和限制的具体语言
"""用于中文-CLIP的特征提取器类。"""

# 导入警告模块
import warnings

# 导入日志记录工具
from ...utils import logging
# 导入中文-CLIP图像处理模块
from .image_processing_chinese_clip import ChineseCLIPImageProcessor

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义中文-CLIP特征提取器类，继承自中文-CLIP图像处理器类
class ChineseCLIPFeatureExtractor(ChineseCLIPImageProcessor):
    # 初始化方法
    def __init__(self, *args, **kwargs) -> None:
        # 发出警告，提示 ChineseCLIPFeatureExtractor 类已被弃用，并将在 Transformers 的第 5 版中移除
        # 请使用 ChineseCLIPImageProcessor 替代
        warnings.warn(
            "The class ChineseCLIPFeatureExtractor is deprecated and will be removed in version 5 of Transformers."
            " Please use ChineseCLIPImageProcessor instead.",
            FutureWarning,
        )
        # 调用父类的初始化方法
        super().__init__(*args, **kwargs)
```