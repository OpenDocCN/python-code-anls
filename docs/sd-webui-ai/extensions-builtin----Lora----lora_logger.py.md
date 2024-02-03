# `stable-diffusion-webui\extensions-builtin\Lora\lora_logger.py`

```
# 导入必要的模块
import sys
import copy
import logging

# 自定义的带颜色的日志格式类
class ColoredFormatter(logging.Formatter):
    # 定义不同日志级别对应的颜色
    COLORS = {
        "DEBUG": "\033[0;36m",  # CYAN
        "INFO": "\033[0;32m",  # GREEN
        "WARNING": "\033[0;33m",  # YELLOW
        "ERROR": "\033[0;31m",  # RED
        "CRITICAL": "\033[0;37;41m",  # WHITE ON RED
        "RESET": "\033[0m",  # RESET COLOR
    }

    # 重写 format 方法，实现日志级别颜色化
    def format(self, record):
        colored_record = copy.copy(record)
        levelname = colored_record.levelname
        seq = self.COLORS.get(levelname, self.COLORS["RESET"])
        colored_record.levelname = f"{seq}{levelname}{self.COLORS['RESET']}"
        return super().format(colored_record)

# 获取名为 "lora" 的 logger 对象，并设置不传播
logger = logging.getLogger("lora")
logger.propagate = False

# 如果 logger 没有处理器，则添加一个输出到控制台的处理器
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        ColoredFormatter("[%(name)s]-%(levelname)s: %(message)s")
    )
    logger.addHandler(handler)
```