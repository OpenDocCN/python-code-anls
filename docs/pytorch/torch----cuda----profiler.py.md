# `.\pytorch\torch\cuda\profiler.py`

```py
# 指定不要检查未定义的函数和方法
mypy: allow-untyped-defs

# 导入上下文管理器和临时文件模块
import contextlib
import tempfile

# 导入 Torch 库
import torch
# 导入本地模块 check_error 和 cudart
from . import check_error, cudart

# 暴露给外部使用的函数和方法
__all__ = ["init", "start", "stop", "profile"]

# 默认的 CUDA 分析标志列表
DEFAULT_FLAGS = [
    "gpustarttimestamp",
    "gpuendtimestamp",
    "gridsize3d",
    "threadblocksize",
    "streamid",
    "enableonstart 0",
    "conckerneltrace",
]


# 初始化 CUDA 分析器
def init(output_file, flags=None, output_mode="key_value"):
    # 获取 cudart 模块
    rt = cudart()
    # 如果 cudart 模块不支持 cudaOutputMode 属性，抛出异常
    if not hasattr(rt, "cudaOutputMode"):
        raise AssertionError("HIP does not support profiler initialization!")
    # 如果 Torch 版本支持 CUDA 且版本号大于等于 12，抛出异常
    if (
        hasattr(torch.version, "cuda")
        and torch.version.cuda is not None
        and int(torch.version.cuda.split(".")[0]) >= 12
    ):
        # 检查 https://github.com/pytorch/pytorch/pull/91118
        # CUDA 12+ 不再需要初始化 CUDA 分析器
        raise AssertionError("CUDA12+ does not need profiler initialization!")
    # 如果未指定标志，则使用默认标志
    flags = DEFAULT_FLAGS if flags is None else flags
    # 根据输出模式设置输出模式枚举值
    if output_mode == "key_value":
        output_mode_enum = rt.cudaOutputMode.KeyValuePair
    elif output_mode == "csv":
        output_mode_enum = rt.cudaOutputMode.CSV
    else:
        raise RuntimeError(
            "supported CUDA profiler output modes are: key_value and csv"
        )
    # 使用临时文件对象写入标志并刷新
    with tempfile.NamedTemporaryFile(delete=True) as f:
        f.write(b"\n".join(f.encode("ascii") for f in flags))
        f.flush()
        # 检查 CUDA 初始化，并抛出可能的错误
        check_error(rt.cudaProfilerInitialize(f.name, output_file, output_mode_enum))


# 启动 CUDA 分析器数据收集
def start():
    r"""Starts cuda profiler data collection.

    .. warning::
        Raises CudaError in case of it is unable to start the profiler.
    """
    check_error(cudart().cudaProfilerStart())


# 停止 CUDA 分析器数据收集
def stop():
    r"""Stops cuda profiler data collection.

    .. warning::
        Raises CudaError in case of it is unable to stop the profiler.
    """
    check_error(cudart().cudaProfilerStop())


# 上下文管理器，用于启用分析
@contextlib.contextmanager
def profile():
    """
    Enable profiling.

    Context Manager to enabling profile collection by the active profiling tool from CUDA backend.
    Example:
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)
        >>> import torch
        >>> model = torch.nn.Linear(20, 30).cuda()
        >>> inputs = torch.randn(128, 20).cuda()
        >>> with torch.cuda.profiler.profile() as prof:
        ...     model(inputs)
    """
    try:
        # 启动 CUDA 分析器
        start()
        # 执行代码块
        yield
    finally:
        # 停止 CUDA 分析器
        stop()
```