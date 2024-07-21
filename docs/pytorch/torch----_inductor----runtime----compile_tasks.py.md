# `.\pytorch\torch\_inductor\runtime\compile_tasks.py`

```py
# mypy: allow-untyped-defs
# 引入未类型化的函数定义支持
from __future__ import annotations

# 引入 functools 模块，用于函数式编程支持
import functools
# 引入 os 模块，提供操作系统相关的功能
import os
# 引入 sys 模块，提供 Python 解释器相关的功能
import sys
# 引入 warnings 模块，用于警告相关的功能
import warnings
# 从 types 模块中导入 ModuleType 类型，用于创建新的模块对象
from types import ModuleType
# 从 typing 模块中导入 Any, Callable, Dict 类型，用于类型标注
from typing import Any, Callable, Dict


def _reload_triton_kernel_in_subproc(reload_module, kernel_name):
    # 在子进程中重新加载 Triton 内核
    return _module_to_triton_kernel(reload_module(), kernel_name)


def _module_to_triton_kernel(mod, kernel_name):
    # 将模块转换为 Triton 内核对象
    kernel = getattr(mod, kernel_name)
    # 部分函数化地定义 Triton 内核的子进程重新加载
    kernel._reload_in_subproc = functools.partial(
        _reload_triton_kernel_in_subproc,
        mod._reload_in_subproc,
        kernel_name,
    )
    return kernel


def _reload_python_module_in_subproc(key, path):
    # 在子进程中重新加载 Python 模块
    codecache = sys.modules.get("torch._inductor.codecache")
    if codecache:
        return codecache.PyCodeCache.load_by_key_path(key, path)
    else:
        return _reload_python_module(key, path)


def _reload_python_module(key, path):
    # 使用给定的路径重新加载 Python 模块
    with open(path) as f:
        try:
            # 编译文件内容为代码对象
            code = compile(f.read(), path, "exec", dont_inherit=True)
        except Exception as e:
            # 如果编译出错，则抛出运行时异常
            raise RuntimeError(
                f"Failed to import {path}\n{type(e).__name__}: {e}"
            ) from None
        # 创建一个新的模块对象
        mod = ModuleType(f"{__name__}.{key}")
        # 设置模块的文件路径
        mod.__file__ = path
        # 设置模块的键值（key）属性
        mod.key = key  # type: ignore[attr-defined]
        # 在模块字典内执行编译后的代码
        exec(code, mod.__dict__, mod.__dict__)
        # 将新创建的模块添加到 sys.modules 中
        sys.modules[mod.__name__] = mod
        return mod


@functools.lru_cache(None)
def _set_triton_ptxas_path() -> None:
    # 设置 Triton 的 ptxas 路径
    if os.environ.get("TRITON_PTXAS_PATH") is not None:
        return
    # 拼接 ptxas 路径
    ptxas_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "bin", "ptxas")
    )
    # 如果 ptxas 路径不存在，则返回
    if not os.path.exists(ptxas_path):
        return
    # 如果 ptxas 路径是一个文件且可执行，则设置环境变量 TRITON_PTXAS_PATH
    if os.path.isfile(ptxas_path) and os.access(ptxas_path, os.X_OK):
        os.environ["TRITON_PTXAS_PATH"] = ptxas_path
    else:
        # 否则发出警告
        warnings.warn(f"{ptxas_path} exists but is not an executable")


def _worker_compile_triton(load_kernel: Callable[[], Any], extra_env: Dict[str, str]):
    # Triton 编译工作进程
    _set_triton_ptxas_path()
    # 更新操作系统环境变量
    os.environ.update(extra_env)
    # 载入内核并预编译，仅使用缓存热启动
    load_kernel().precompile(warm_cache_only=True)
```