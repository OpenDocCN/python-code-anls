# `.\sagemaker\trainer_sm.py`

```py
# 导入警告模块，用于在特定情况下发出警告
import warnings

# 从上级目录中导入 Trainer 类
from ..trainer import Trainer

# 从上级目录中的 utils 模块中导入 logging 工具
from ..utils import logging

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 定义 SageMakerTrainer 类，继承自 Trainer 类
class SageMakerTrainer(Trainer):
    def __init__(self, args=None, **kwargs):
        # 发出警告，提示用户 SageMakerTrainer 类将在 Transformers v5 版本中被移除，建议使用 Trainer 类
        warnings.warn(
            "`SageMakerTrainer` is deprecated and will be removed in v5 of Transformers. You can use `Trainer` "
            "instead.",
            FutureWarning,
        )
        # 调用父类 Trainer 的初始化方法，传递参数 args 和其他关键字参数
        super().__init__(args=args, **kwargs)
```