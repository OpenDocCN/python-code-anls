# `stable-diffusion-webui\modules\logging_config.py`

```py
# 导入必要的模块
import os
import logging

# 尝试导入 tqdm 模块，如果导入失败则设置 TQDM_IMPORTED 为 False
try:
    from tqdm.auto import tqdm

    # 自定义 logging.Handler 类，用于将日志输出到 tqdm 进度条
    class TqdmLoggingHandler(logging.Handler):
        def __init__(self, level=logging.INFO):
            super().__init__(level)

        # 重写 emit 方法，将日志消息写入 tqdm 进度条
        def emit(self, record):
            try:
                msg = self.format(record)
                tqdm.write(msg)
                self.flush()
            except Exception:
                self.handleError(record)

    # 标记 tqdm 是否成功导入
    TQDM_IMPORTED = True
except ImportError:
    # 如果导入失败，设置 TQDM_IMPORTED 为 False，并记录原因
    # 在首次启动前 tqdm 不存在，将在 UI 完成设置环境并重新加载后导入
    TQDM_IMPORTED = False

# 设置日志记录级别
def setup_logging(loglevel):
    # 如果未指定日志级别，则从环境变量中获取
    if loglevel is None:
        loglevel = os.environ.get("SD_WEBUI_LOG_LEVEL")

    # 初始化日志处理器列表
    loghandlers = []

    # 如果成功导入 tqdm 模块，则添加 TqdmLoggingHandler 到日志处理器列表
    if TQDM_IMPORTED:
        loghandlers.append(TqdmLoggingHandler())

    # 如果指定了日志级别，则根据级别设置日志记录级别，否则默认为 INFO 级别
    if loglevel:
        log_level = getattr(logging, loglevel.upper(), None) or logging.INFO
        # 配置日志记录格式和处理器
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s %(levelname)s [%(name)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=loghandlers
        )
```