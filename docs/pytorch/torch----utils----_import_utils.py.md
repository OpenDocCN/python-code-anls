# `.\pytorch\torch\utils\_import_utils.py`

```
# 引入类型提示允许未定义的函数
mypy: allow-untyped-defs

# 导入 functools 模块，用于缓存函数的结果
import functools

# 导入 importlib.util 模块，用于动态导入模块
import importlib.util

# 导入 torch 模块，深度学习框架
import torch


# 检查是否存在名为 name 的顶层模块，但不进行实际导入。这比在 `import X` 周围使用 try-catch 块更安全。
# 避免第三方库在导入时破坏某些测试的假设，例如设置多进程启动方法。
def _check_module_exists(name: str) -> bool:
    r"""Returns if a top-level module with :attr:`name` exists *without**
    importing it. This is generally safer than try-catch block around a
    `import X`. It avoids third party libraries breaking assumptions of some of
    our tests, e.g., setting multiprocessing start method when imported
    (see librosa/#747, torchvision/#544).
    """
    try:
        # 使用 importlib.util.find_spec 查找模块的规范
        spec = importlib.util.find_spec(name)
        # 如果找到了模块规范，则返回 True
        return spec is not None
    except ImportError:
        # 如果 ImportError 异常发生，返回 False
        return False


# 使用 functools.lru_cache 装饰器，缓存函数的结果
def dill_available():
    return (
        # 检查 dill 模块是否存在
        _check_module_exists("dill")
        # 并且确保不在 torchdeploy 环境下运行
        and not torch._running_with_deploy()
    )


# 使用 functools.lru_cache 装饰器，缓存函数的结果
def import_dill():
    # 如果 dill 模块不可用，则返回 None
    if not dill_available():
        return None

    # 导入 dill 模块
    import dill

    # XXX: 默认情况下，dill 将 Pickler 分发表写入以注入其逻辑。这会全局影响标准库的 Pickler 行为，
    # 任何依赖于该模块的用户都会受到影响！为了避免全局修改 Pickler 行为，需要撤消此扩展。
    dill.extend(use_dill=False)
    return dill
```