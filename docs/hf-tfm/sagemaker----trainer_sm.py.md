# `.\transformers\sagemaker\trainer_sm.py`

```py
# 导入警告模块
import warnings

# 导入 Trainer 类
from ..trainer import Trainer
# 导入日志工具
from ..utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义 SageMakerTrainer 类，继承自 Trainer 类
class SageMakerTrainer(Trainer):
    # 初始化方法
    def __init__(self, args=None, **kwargs):
        # 发出警告，指出 SageMakerTrainer 类已弃用，并将在 Transformers 的 v5 版本中移除，建议使用 Trainer 类代替
        warnings.warn(
            "`SageMakerTrainer` is deprecated and will be removed in v5 of Transformers. You can use `Trainer` "
            "instead.",
            FutureWarning,
        )
        # 调用父类 Trainer 的初始化方法
        super().__init__(args=args, **kwargs)
```  
```