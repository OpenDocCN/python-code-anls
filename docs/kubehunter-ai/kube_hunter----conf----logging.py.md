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

    # 如果传入的日志级别名称为 "NONE"
# 禁用临界级别以下的所有日志记录
logging.disable(logging.CRITICAL)
# 如果未禁用日志记录，则根据给定的日志级别名称获取对应的日志级别
log_level = getattr(logging, level_name.upper(), None)
# 如果获取的日志级别不是整数类型，则将其设为 None
log_level = log_level if type(log_level) is int else None
# 配置日志记录的级别和格式，如果未指定日志级别，则使用默认级别和格式
logging.basicConfig(level=log_level or DEFAULT_LEVEL, format=LOG_FORMAT)
# 如果未指定日志级别，则发出警告并使用默认级别
if not log_level:
    logging.warning(f"Unknown log level '{level_name}', using {DEFAULT_LEVEL_NAME}")
```