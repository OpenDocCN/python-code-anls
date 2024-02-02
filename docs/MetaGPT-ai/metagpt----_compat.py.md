# `MetaGPT\metagpt\_compat.py`

```py

# 导入 platform 模块，用于获取系统信息
import platform
# 导入 sys 模块，用于访问 Python 解释器的相关信息
import sys
# 导入 warnings 模块，用于处理警告
import warnings

# 检查 Python 解释器是否为 CPython，并且操作系统为 Windows
if sys.implementation.name == "cpython" and platform.system() == "Windows":
    # 导入 asyncio 模块，用于异步编程
    import asyncio

    # 检查 Python 版本是否为 3.9
    if sys.version_info[:2] == (3, 9):
        # 导入 _ProactorBasePipeTransport 类，用于处理异步 I/O
        from asyncio.proactor_events import _ProactorBasePipeTransport

        # 重写 _ProactorBasePipeTransport 类的 __del__ 方法，用于处理资源释放
        def pacth_del(self, _warn=warnings.warn):
            if self._sock is not None:
                _warn(f"unclosed transport {self!r}", ResourceWarning, source=self)
                self._sock.close()

        _ProactorBasePipeTransport.__del__ = pacth_del

    # 检查 Python 版本是否大于等于 3.9.0
    if sys.version_info >= (3, 9, 0):
        # 导入 sk_function 函数，用于语义内核编排
        from semantic_kernel.orchestration import sk_function as _  # noqa: F401

        # 设置 Windows 平台的事件循环策略为 Proactor
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

```