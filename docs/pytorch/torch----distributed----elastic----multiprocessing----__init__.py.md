# `.\pytorch\torch\distributed\elastic\multiprocessing\__init__.py`

```py
#!/usr/bin/env python3
"""
Library that launches and manages ``n`` copies of worker subprocesses either specified by a function or a binary.

For functions, it uses ``torch.multiprocessing`` (and therefore python
``multiprocessing``) to spawn/fork worker processes. For binaries it uses python
``subprocessing.Popen`` to create worker processes.
"""

from typing import Callable, Dict, Optional, Tuple, Union

from torch.distributed.elastic.multiprocessing.api import (
    _validate_full_rank,
    DefaultLogsSpecs,
    LogsDest,
    LogsSpecs,
    MultiprocessContext,
    PContext,
    ProcessFailure,
    RunProcsResult,
    SignalException,
    Std,
    SubprocessContext,
    to_map,
)
from torch.distributed.elastic.utils.logging import get_logger

__all__ = [
    "start_processes",
    "MultiprocessContext",
    "PContext",
    "ProcessFailure",
    "RunProcsResult",
    "SignalException",
    "Std",
    "LogsDest",
    "LogsSpecs",
    "DefaultLogsSpecs",
    "SubprocessContext",
    "to_map",
]

def start_processes(
    name: str,
    entrypoint: Union[Callable, str],
    args: Dict[int, Tuple],
    envs: Dict[int, Dict[str, str]],
    logs_specs: LogsSpecs,
    log_line_prefixes: Optional[Dict[int, str]] = None,
    start_method: str = "spawn",
) -> PContext:
    """
    Start ``n`` copies of ``entrypoint`` processes with the provided options.

    :param name: Name of the process group.
    :param entrypoint: Function or binary to launch as subprocesses.
    :param args: Dictionary mapping process indices to arguments passed to each process.
    :param envs: Dictionary mapping process indices to environment variables for each process.
    :param logs_specs: Specification for logging destinations and behavior.
    :param log_line_prefixes: Optional mapping from process indices to log line prefixes.
    :param start_method: Method used to start subprocesses ("spawn", "fork", etc.).

    :return: A context object representing the launched processes and their state.
    """
    pass
    # `entrypoint` 可以是 `Callable`（函数）或 `str`（二进制文件路径）。
    # 复制的次数由 `args` 和 `envs` 参数的条目数量决定，这两个参数需要具有相同的键集合。
    # 
    # `args` 和 `env` 参数是要传递给入口点的参数和环境变量，由副本索引（本地排名）映射。
    # 必须考虑所有本地排名，因此键集合应为 `{0,1,...,(nprocs-1)}`。
    # 
    # .. note:: 当 `entrypoint` 是二进制文件路径（`str`）时，`args` 只能是字符串。
    #           如果给定其他类型的参数，则会被转换为字符串表示（例如 `str(arg1)`）。
    #           此外，只有在主函数标注有 `torch.distributed.elastic.multiprocessing.errors.record` 时，
    #           二进制文件路径的失败才会写入 `error.json` 错误文件。对于函数启动，默认情况下会自动进行此操作，
    #           无需手动使用 `@record` 注释。
    # 
    # `redirects` 和 `tee` 是比特掩码，指定要重定向到 `log_dir` 中日志文件的标准流。
    # 有效的掩码值在 `Std` 中定义。要仅重定向/打印某些本地排名的输出，将 `redirects` 传递为一个映射，
    # 键为本地排名，指定重定向行为。任何缺失的本地排名将默认为 `Std.NONE`。
    # 
    # `tee` 类似于 Unix 的 "tee" 命令，即重定向并同时输出到控制台。
    # 若要避免工作进程的标准输出/标准错误输出到控制台，可以使用 `redirects` 参数。
    # 
    # 每个进程的 `log_dir` 将包含以下内容：
    # 
    # 1. `{local_rank}/error.json`: 如果进程失败，则包含错误信息的文件
    # 2. `{local_rank}/stdout.json`: 如果 `redirect & STDOUT == STDOUT`，则包含标准输出的文件
    # 3. `{local_rank}/stderr.json`: 如果 `redirect & STDERR == STDERR`，则包含标准错误的文件
    # 
    # .. note:: 预期 `log_dir` 存在且为空，并且是一个目录。
    # 
    # 示例：
    # ::
    # 
    #  log_dir = "/tmp/test"
    # 
    #  # 正确；两个 `foo` 的副本：foo("bar0"), foo("bar1")
    #  start_processes(
    #     name="trainer",
    #     entrypoint=foo,
    #     args:{0:("bar0",), 1:("bar1",),
    #     envs:{0:{}, 1:{}},
    #     log_dir=log_dir
    #  )
    # 
    #  # 无效；本地排名 1 缺少环境变量
    #  start_processes(
    #     name="trainer",
    #     entrypoint=foo,
    #     args:{0:("bar0",), 1:("bar1",),
    #     envs:{0:{}},
    #     log_dir=log_dir
    #  )
    # 
    #  # 正确；两个 `/usr/bin/touch` 的副本：touch file1, touch file2
    #  start_processes(
    #     name="trainer",
    #     entrypoint="/usr/bin/touch",
    #     args:{0:("file1",), 1:("file2",),
    #     envs:{0:{}, 1:{}},
    #     log_dir=log_dir
    #   )
    # 
    #  # 警告；参数被转换为字符串，运行：
    #  # echo "1" "2" "3" 和 echo "[1, 2, 3]"
    #  start_processes(
    #     name="trainer",
    #     entrypoint="/usr/bin/echo",
    #     args:{0:(1,2,3), 1:([1,2,3],),
    #     envs:{0:{}, 1:{}},
    #     log_dir=log_dir
    #   )
    Args:
        name: a human readable short name that describes what the processes are
              (used as header when tee'ing stdout/stderr outputs)
        entrypoint: either a ``Callable`` (function) or ``cmd`` (binary)
        args: arguments to each replica
        envs: env vars to each replica
        log_dir: directory used to write log files
        start_method: multiprocessing start method (spawn, fork, forkserver)
                      ignored for binaries
        redirects: which std streams to redirect to a log file
        tee: which std streams to redirect + print to console
        local_ranks_filter: which ranks' logs to print to console
    """
    # 计算进程数量
    nprocs = len(args)
    # 验证参数列表的完整性
    _validate_full_rank(args, nprocs, "args")
    # 验证环境变量列表的完整性
    _validate_full_rank(envs, nprocs, "envs")

    # 定义上下文对象
    context: PContext
    # 如果 entrypoint 是字符串类型，使用 SubprocessContext 创建上下文
    if isinstance(entrypoint, str):
        context = SubprocessContext(
            name=name,
            entrypoint=entrypoint,
            args=args,
            envs=envs,
            logs_specs=logs_specs,
            log_line_prefixes=log_line_prefixes,
        )
    # 否则，使用 MultiprocessContext 创建上下文
    else:
        context = MultiprocessContext(
            name=name,
            entrypoint=entrypoint,
            args=args,
            envs=envs,
            log_line_prefixes=log_line_prefixes,
            start_method=start_method,
            logs_specs=logs_specs,
        )

    try:
        # 启动上下文
        context.start()
        # 返回上下文对象
        return context
    except Exception:
        # 发生异常时关闭上下文
        context.close()
        # 重新抛出异常
        raise
```