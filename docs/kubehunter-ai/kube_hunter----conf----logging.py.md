# `kubehunter\kube_hunter\conf\logging.py`

```
# 导入 logging 模块
import logging

# 设置默认日志级别为 INFO
DEFAULT_LEVEL = logging.INFO
# 获取默认日志级别的名称
DEFAULT_LEVEL_NAME = logging.getLevelName(DEFAULT_LEVEL)
# 设置日志格式
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"

# 禁止 scapy 模块的日志输出
logging.getLogger("scapy.runtime").setLevel(logging.CRITICAL)
logging.getLogger("scapy.loading").setLevel(logging.CRITICAL)

# 设置日志记录器的级别
def setup_logger(level_name):
    # 移除已存在的日志处理器
    # 在 Python 3.8 中不必要，因为 `logging.basicConfig` 有 `force` 参数
    for h in logging.getLogger().handlers[:]:
        h.close()
        logging.getLogger().removeHandler(h)

    # 如果日志级别为 "NONE"，则禁用日志记录
    if level_name.upper() == "NONE":
        logging.disable(logging.CRITICAL)
    else:
        # 获取指定名称对应的日志级别
        log_level = getattr(logging, level_name.upper(), None)
        # 如果找不到对应的日志级别，则设置为 None
        log_level = log_level if type(log_level) is int else None
        # 设置日志记录器的级别和格式
        logging.basicConfig(level=log_level or DEFAULT_LEVEL, format=LOG_FORMAT)
        # 如果指定的日志级别不存在，则发出警告并使用默认级别
        if not log_level:
            logging.warning(f"Unknown log level '{level_name}', using {DEFAULT_LEVEL_NAME}")
```