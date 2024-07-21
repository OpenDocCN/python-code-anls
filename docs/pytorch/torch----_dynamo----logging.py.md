# `.\pytorch\torch\_dynamo\logging.py`

```py
# mypy: allow-untyped-defs
# 引入 itertools 和 logging 模块
import itertools
import logging

# 从 torch.hub 中导入 _Faketqdm 和 tqdm
from torch.hub import _Faketqdm, tqdm

# 默认禁用进度条，避免在 dynamo 配置中循环导入的问题
disable_progress = True

# 返回所有与 torchdynamo/torchinductor 相关的日志记录器
def get_loggers():
    return [
        logging.getLogger("torch.fx.experimental.symbolic_shapes"),
        logging.getLogger("torch._dynamo"),
        logging.getLogger("torch._inductor"),
    ]

# 创建一个日志函数，以步骤号为前缀记录消息。
# get_step_logger 应该在运行时懒加载，而不是在模块加载时。
# 这样可以确保步骤号在初始化时正确设置。例如：
# 
# @functools.lru_cache(None)
# def _step_logger():
#     return get_step_logger(logging.getLogger(...))
# 
# def fn():
#     _step_logger()(logging.INFO, "msg")

# itertools.count 用于生成连续的整数计数器，从1开始
_step_counter = itertools.count(1)

# 更新 num_steps 如果添加了更多阶段：Dynamo、AOT、Backend
# 这非常依赖于 Inductor
# _inductor.utils.has_triton() 在此处会导致循环导入错误

# 如果未禁用进度条，则尝试引入 triton 模块来设置 num_steps
if not disable_progress:
    try:
        import triton  # noqa: F401

        num_steps = 3
    except ImportError:
        num_steps = 2
    
    # 初始化进度条对象 tqdm，显示为 "torch.compile()"，延迟为 0
    pbar = tqdm(total=num_steps, desc="torch.compile()", delay=0)

# 返回具有步骤号的日志记录器函数
def get_step_logger(logger):
    if not disable_progress:
        # 更新进度条
        pbar.update(1)
        # 如果 pbar 不是 _Faketqdm 的实例，则设置后缀字符串为 logger 的名称
        if not isinstance(pbar, _Faketqdm):
            pbar.set_postfix_str(f"{logger.name}")

    # 获取下一个步骤号
    step = next(_step_counter)

    # 定义日志函数，记录特定级别的消息，并附带步骤号
    def log(level, msg, **kwargs):
        logger.log(level, "Step %s: %s", step, msg, **kwargs)

    return log
```