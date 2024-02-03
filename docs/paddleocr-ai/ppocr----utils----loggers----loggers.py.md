# `.\PaddleOCR\ppocr\utils\loggers\loggers.py`

```py
# 导入WandbLogger类
from .wandb_logger import WandbLogger

# 定义Loggers类
class Loggers(object):
    # 初始化Loggers对象，接受loggers参数
    def __init__(self, loggers):
        super().__init__()
        # 将loggers参数赋值给实例变量loggers
        self.loggers = loggers

    # 记录指标的方法，接受metrics、prefix和step参数
    def log_metrics(self, metrics, prefix=None, step=None):
        # 遍历loggers列表中的每个logger对象
        for logger in self.loggers:
            # 调用logger对象的log_metrics方法，传入metrics、prefix和step参数
            logger.log_metrics(metrics, prefix=prefix, step=step)
    
    # 记录模型的方法，接受is_best、prefix和metadata参数
    def log_model(self, is_best, prefix, metadata=None):
        # 遍历loggers列表中的每个logger对象
        for logger in self.loggers:
            # 调用logger对象的log_model方法，传入is_best、prefix和metadata参数
            logger.log_model(is_best=is_best, prefix=prefix, metadata=metadata)
    
    # 关闭所有logger对象的方法
    def close(self):
        # 遍历loggers列表中的每个logger对象
        for logger in self.loggers:
            # 调用logger对象的close方法
            logger.close()
```