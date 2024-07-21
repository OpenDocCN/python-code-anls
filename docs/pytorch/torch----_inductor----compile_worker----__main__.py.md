# `.\pytorch\torch\_inductor\compile_worker\__main__.py`

```
# mypy: allow-untyped-defs
# 导入必要的模块：命令行参数解析、日志记录、操作系统相关功能、类型提示
import argparse
import logging
import os
import sys
import typing

# 导入异步编译的预设置函数和相关模块
from torch._inductor.async_compile import pre_fork_setup
from torch._inductor.compile_worker.subproc_pool import Pipe, SubprocMain
from torch._inductor.compile_worker.watchdog import _async_compile_initializer
from torch._inductor.runtime.compile_tasks import _set_triton_ptxas_path

# 获取当前模块的日志记录器
log = logging.getLogger(__name__)

# 设置 Triton 的 PTXAS 路径
_set_triton_ptxas_path()

# 尝试导入 Triton 模块，如果导入失败则忽略
try:
    import triton

    assert triton is not None  # 在父进程中预加载 Triton
except ImportError:
    pass

# 主函数入口
def main():
    try:
        # 创建命令行参数解析器
        parser = argparse.ArgumentParser()
        # 添加命令行参数：工作进程数目
        parser.add_argument("--workers", type=int)
        # 添加命令行参数：父进程 ID
        parser.add_argument("--parent", type=int)
        # 解析命令行参数
        args = parser.parse_args()
        
        # 如果当前进程的父进程 ID 不等于指定的父进程 ID，则退出程序
        if os.getppid() != args.parent:
            sys.exit(0)
        
        # 复制标准输出的文件描述符，并封装成管道对象
        write_fd = typing.cast(Pipe, os.fdopen(os.dup(sys.stdout.fileno()), "wb"))
        # 复制标准输入的文件描述符，并封装成管道对象
        read_fd = typing.cast(Pipe, os.fdopen(os.dup(sys.stdin.fileno()), "rb"))

        # 关闭标准输入，确保没有其它程序可以读取它
        sys.stdin.close()

        # 将工作进程的输出重定向到标准错误
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())

        # 执行工作进程的预设置操作
        pre_fork_setup()

        # 异步编译初始化设置
        _async_compile_initializer(args.parent)
        
        # 创建并启动工作进程的主控制器
        SubprocMain(args.workers, read_fd, write_fd).main()
    except Exception:
        # 捕获并记录异常信息
        log.exception("Uncaught exception in compile_worker subprocess")


# 如果当前脚本作为主程序运行，则调用 main 函数
if __name__ == "__main__":
    main()
```