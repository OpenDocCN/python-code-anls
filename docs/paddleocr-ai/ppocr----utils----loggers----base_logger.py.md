# `.\PaddleOCR\ppocr\utils\loggers\base_logger.py`

```py
# 导入 os 模块
import os
# 从 abc 模块中导入 ABC 和 abstractmethod
from abc import ABC, abstractmethod

# 定义一个抽象基类 BaseLogger
class BaseLogger(ABC):
    # 初始化方法，接收保存目录参数
    def __init__(self, save_dir):
        # 将保存目录参数保存到实例属性中
        self.save_dir = save_dir
        # 创建保存目录，如果目录已存在则不做任何操作
        os.makedirs(self.save_dir, exist_ok=True)

    # 抽象方法，用于记录指标数据
    @abstractmethod
    def log_metrics(self, metrics, prefix=None):
        pass

    # 抽象方法，用于关闭日志记录器
    @abstractmethod
    def close(self):
        pass
```