# `.\pytorch\torch\distributed\elastic\control_plane.py`

```py
# 导入必要的模块和库
import os
from contextlib import contextmanager, ExitStack
from typing import Generator

# 从torch.distributed.elastic.multiprocessing.errors模块导入record函数
from torch.distributed.elastic.multiprocessing.errors import record

# 定义模块中公开的所有符号
__all__ = [
    "worker_main",
]

# 环境变量名称，用于指定工作进程服务器的套接字路径
TORCH_WORKER_SERVER_SOCKET = "TORCH_WORKER_SERVER_SOCKET"

# 定义一个上下文管理器，用于创建工作进程服务器
@contextmanager
def _worker_server(socket_path: str) -> Generator[None, None, None]:
    # 导入 _WorkerServer 类来创建工作进程服务器
    from torch._C._distributed_c10d import _WorkerServer

    # 创建 _WorkerServer 对象，绑定到指定的套接字路径
    server = _WorkerServer(socket_path)
    try:
        yield
    finally:
        # 在退出上下文时关闭工作进程服务器
        server.shutdown()

# 定义一个上下文管理器，用于主工作进程的入口点
@contextmanager
@record
def worker_main() -> Generator[None, None, None]:
    """
    这是一个上下文管理器，用于包装您的主入口函数。它结合了现有的 ``errors.record`` 逻辑，
    以及一个新的 ``_WorkerServer``，通过环境变量 ``TORCH_WORKER_SERVER_SOCKET`` 指定的 Unix 套接字暴露处理程序。

    示例

    ::

     @worker_main()
     def main():
         pass

     if __name__=="__main__":
        main()

    """
    with ExitStack() as stack:
        # 从环境变量中获取套接字路径
        socket_path = os.environ.get(TORCH_WORKER_SERVER_SOCKET)
        # 如果套接字路径不为None，则进入 _worker_server 上下文
        if socket_path is not None:
            stack.enter_context(_worker_server(socket_path))

        # 执行 yield 之前的操作
        yield
```