# `.\models\vilt\feature_extraction_vilt.py`

```
# 设置脚本的编码格式为 UTF-8
# 版权声明，版权归 HuggingFace Inc. 团队所有，保留所有权利
#
# 根据 Apache 许可证 2.0 版本授权：
# 您可以在符合许可证的情况下使用此文件。
# 您可以从以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件根据“原样”分发，
# 不附带任何明示或暗示的担保或条件。
# 请参阅许可证获取更多信息。
"""ViLT 的特征提取器类。"""

# 导入警告模块
import warnings

# 导入日志记录模块
from ...utils import logging
# 导入 ViLT 图像处理模块中的 ViltImageProcessor 类
from .image_processing_vilt import ViltImageProcessor

# 获取当前模块的日志记录器对象
logger = logging.get_logger(__name__)

# 定义 ViLT 特征提取器类，继承自 ViltImageProcessor 类
class ViltFeatureExtractor(ViltImageProcessor):
    # 初始化方法，接受任意位置参数和关键字参数
    def __init__(self, *args, **kwargs) -> None:
        # 发出警告，提示 ViltFeatureExtractor 类即将在 Transformers 版本 5 中被移除，建议使用 ViltImageProcessor 代替
        warnings.warn(
            "The class ViltFeatureExtractor is deprecated and will be removed in version 5 of Transformers. Please"
            " use ViltImageProcessor instead.",
            FutureWarning,
        )
        # 调用父类 ViltImageProcessor 的初始化方法，传递所有接收到的位置参数和关键字参数
        super().__init__(*args, **kwargs)
```