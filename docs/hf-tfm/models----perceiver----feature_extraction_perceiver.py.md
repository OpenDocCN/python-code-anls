# `.\models\perceiver\feature_extraction_perceiver.py`

```py
# 设置文件编码为 UTF-8
# 版权声明和许可信息，告知代码使用者遵循 Apache 许可证版本 2.0 使用，禁止未经许可的复制和修改
# 获取 Apache 许可证版本 2.0 的详细信息的链接
# 根据适用法律或书面同意，按“现状”分发软件，不提供任何形式的担保或条件
# 引入警告模块，用于将来可能删除的类的警告
# 引入日志模块，用于记录和输出日志信息
# 从 image_processing_perceiver 模块中导入 PerceiverImageProcessor 类
import warnings
from ...utils import logging
from .image_processing_perceiver import PerceiverImageProcessor

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# PerceiverFeatureExtractor 类继承自 PerceiverImageProcessor 类
class PerceiverFeatureExtractor(PerceiverImageProcessor):
    # 初始化方法，接受任意位置参数和关键字参数
    def __init__(self, *args, **kwargs) -> None:
        # 发出未来警告，提醒用户 PerceiverFeatureExtractor 类将在 Transformers 版本 5 中删除，建议使用 PerceiverImageProcessor 替代
        warnings.warn(
            "The class PerceiverFeatureExtractor is deprecated and will be removed in version 5 of Transformers."
            " Please use PerceiverImageProcessor instead.",
            FutureWarning,
        )
        # 调用父类 PerceiverImageProcessor 的初始化方法，传递所有位置参数和关键字参数
        super().__init__(*args, **kwargs)
```