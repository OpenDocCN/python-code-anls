# `.\pytorch\torch\cuda\tunable.py`

```py
r"""
This module exposes a TunableOp interface.

Some operations, such as GEMMs, could be implemented using more than one library
or more than one technique. For example, a GEMM could be implemented for CUDA or
ROCm using either the blas or blasLt libraries. Further, ROCm's rocblas and
hipblaslt libraries allow the user to query for all possible algorithms and then
choose one. How does one know which implementation is the fastest and should be
chosen? That's what TunableOp provides.

Enabling TunableOp and Tuning Separately
========================================

The TunableOp feature is enabled separately from enabling the tuning phase
itself. Enabling TunableOp means that PyTorch will replace any standard
operators with their Tunable implementations. Any call to a TunableOp first
checks whether it has already been tuned for the given operator inputs. If so,
it will immediately call the tuned operation; no further tuning will take place
even when the tuning setting is enabled. Instead if no tuning result is found,
and tuning is enabled, the TunableOp will benchmark every registered
implementation of that operator for the given set of inputs and select the
fastest.

File Input and Output
=====================

The first time any TunableOp is invoked, the internal database of tuned
operations will be prepared by attempting to read the results from the given
file. The default filename is 'tunableop_results.csv'. To support tuning when
multiple GPUs are used across multiple processes, the GPU device ordinal is
automatically inserted into the filename to avoid multiple processes overwriting
the same file.

If tuning is enabled and new tunings are discovered during the course of your
workload, it will also write out to this same filename with all tunings, both
the ones it read in at startup as well as the new ones found at runtime. This
can be used, for example, to build up a tunings file across many workloads by
reusing the same file. The output file is automatically created when the
application terminates. This behavior can be controlled by the C++ and Python
APIs but not the environment variables.

Assuming you specified a filename, you'll end up with a CSV file with contents
like so::

  Validator,PT_VERSION,2.2.0
  Validator,ROCM_VERSION,6.0.0.0-12969-1544e39
  Validator,HIPBLASLT_VERSION,0.6.0-a9c5cc7
  Validator,ROCBLAS_VERSION,4.0.0-72e57364-dirty
  GemmTunableOp_float_NT,nt_25088_4096_64,1219,1.262
  GemmTunableOp_float_NT,nt_4096_4096_64,1216,0.033

Note the "Validator" lines. If you change a library verison, or ROCm version, or
PyTorch version, TunableOp will detect this and reject the tunings file because
the prior tunings are likely affected by other software changes.

The remaining lines are the tuned solutions for each TunableOp encountered
during your execution. Each line consists of 4 comma-separated fields: operator
name, operator parameters, solution name, and average execution time. The
"""

注释：
"""
Define a list of public symbols to be exported when using 'from module import *'
"""
from typing import Optional, Tuple
"""
Import necessary modules from the standard library
"""
import torch

"""
List of symbols that will be accessible when using 'from module import *'
"""
__all__ = [
    "enable",
    "is_enabled",
    "tuning_enable",
    "tuning_is_enabled",
    "set_max_tuning_duration",

"""
Function to enable TunableOp functionality.
"""
def enable():
    """
    Enable TunableOp functionality by setting appropriate environment variables or API calls.
    """
    pass

"""
Function to check if TunableOp functionality is enabled.
"""
def is_enabled() -> bool:
    """
    Check whether TunableOp functionality is currently enabled.
    Returns True if enabled, False otherwise.
    """
    pass

"""
Function to enable tuning mode for TunableOp.
"""
def tuning_enable():
    """
    Enable tuning mode for TunableOp, allowing optimization of operators.
    """
    pass

"""
Function to check if tuning mode is enabled for TunableOp.
"""
def tuning_is_enabled() -> bool:
    """
    Check whether tuning mode is currently enabled for TunableOp.
    Returns True if tuning mode is enabled, False otherwise.
    """
    pass

"""
Function to set the maximum duration for tuning operations.
"""
def set_max_tuning_duration(duration: float):
    """
    Set the maximum duration (in seconds) that tuning operations are allowed to run.
    
    Parameters:
    - duration: Maximum duration in seconds for tuning operations.
    """
    pass
    "get_max_tuning_duration",  # 返回最大调优持续时间的方法名
    "set_max_tuning_iterations",  # 设置最大调优迭代次数的方法名
    "get_max_tuning_iterations",  # 返回最大调优迭代次数的方法名
    "set_filename",  # 设置文件名的方法名
    "get_filename",  # 返回文件名的方法名
    "get_results",  # 返回结果的方法名
    "get_validators",  # 返回验证器列表的方法名
    "write_file_on_exit",  # 在退出时写文件的方法名
    "write_file",  # 写文件的方法名
    "read_file",  # 读文件的方法名
# 定义一个函数，用于启用或禁用所有 TunableOp 实现的开关。
def enable(val: bool = True) -> None:
    # 调用 Torch 库中的底层 C 函数，设置 TunableOp 的启用状态
    torch._C._cuda_tunableop_enable(val)  # type: ignore[attr-defined]


# 定义一个函数，用于检查 TunableOp 功能是否已启用。
def is_enabled() -> bool:
    # 调用 Torch 库中的底层 C 函数，返回 TunableOp 的启用状态
    return torch._C._cuda_tunableop_is_enabled()  # type: ignore[attr-defined]


# 定义一个函数，用于启用或禁用 TunableOp 实现的调优功能。
def tuning_enable(val: bool = True) -> None:
    # 调用 Torch 库中的底层 C 函数，设置 TunableOp 的调优功能启用状态
    torch._C._cuda_tunableop_tuning_enable(val)  # type: ignore[attr-defined]


# 定义一个函数，用于检查 TunableOp 实现的调优功能是否已启用。
def tuning_is_enabled() -> bool:
    # 调用 Torch 库中的底层 C 函数，返回 TunableOp 的调优功能启用状态
    return torch._C._cuda_tunableop_tuning_is_enabled()  # type: ignore[attr-defined]


# 定义一个函数，设置调优过程的最大持续时间（毫秒）。
def set_max_tuning_duration(duration: int) -> None:
    # 调用 Torch 库中的底层 C 函数，设置 TunableOp 的最大调优持续时间
    torch._C._cuda_tunableop_set_max_tuning_duration(duration)  # type: ignore[attr-defined]


# 定义一个函数，获取调优过程的最大持续时间（毫秒）。
def get_max_tuning_duration() -> int:
    # 调用 Torch 库中的底层 C 函数，获取 TunableOp 的最大调优持续时间
    return torch._C._cuda_tunableop_get_max_tuning_duration()  # type: ignore[attr-defined]


# 定义一个函数，设置调优过程的最大迭代次数。
def set_max_tuning_iterations(iterations: int) -> None:
    # 调用 Torch 库中的底层 C 函数，设置 TunableOp 的最大调优迭代次数
    torch._C._cuda_tunableop_set_max_tuning_iterations(iterations)  # type: ignore[attr-defined]


# 定义一个函数，获取调优过程的最大迭代次数。
def get_max_tuning_iterations() -> int:
    # 调用 Torch 库中的底层 C 函数，获取 TunableOp 的最大调优迭代次数
    return torch._C._cuda_tunableop_get_max_tuning_iterations()  # type: ignore[attr-defined]


# 定义一个函数，设置用于调优结果输入/输出的文件名。
def set_filename(filename: str, insert_device_ordinal: bool = False) -> None:
    # 调用 Torch 库中的底层 C 函数，设置 TunableOp 的结果文件名
    torch._C._cuda_tunableop_set_filename(filename, insert_device_ordinal)  # type: ignore[attr-defined]


# 定义一个函数，获取用于调优结果输入/输出的文件名。
def get_filename() -> str:
    # 调用 Torch 库中的底层 C 函数，获取 TunableOp 的结果文件名
    return torch._C._cuda_tunableop_get_filename()  # type: ignore[attr-defined]


# 定义一个函数，获取所有 TunableOp 的结果。
def get_results() -> Tuple[str, str, str, float]:
    # 调用 Torch 库中的底层 C 函数，返回所有 TunableOp 的结果
    return torch._C._cuda_tunableop_get_results()  # type: ignore[attr-defined]


# 定义一个函数，获取 TunableOp 的验证器。
def get_validators() -> Tuple[str, str]:
    # 调用 Torch 库中的底层 C 函数，返回 TunableOp 的验证器
    return torch._C._cuda_tunableop_get_validators()  # type: ignore[attr-defined]
def write_file_on_exit(val: bool) -> None:
    r"""During Tuning Context destruction, write file to disk.

    This is useful as a final flush of your results to disk if your application
    terminates as result of normal operation or an error. Manual flushing of
    your results can be achieved by manually calling ``write_file()``."""
    # 调用 torch._C._cuda_tunableop_write_file_on_exit 函数，将 val 写入文件系统
    torch._C._cuda_tunableop_write_file_on_exit(val)  # type: ignore[attr-defined]


def write_file(filename: Optional[str] = None) -> bool:
    r"""Write results to a CSV file.

    If :attr:`filename` is not given, ``get_filename()`` is called.
    """
    # 如果未提供 filename，则调用 get_filename() 获取文件名
    if filename is None:
        filename = get_filename()
    # 调用 torch._C._cuda_tunableop_write_file 函数，将结果写入 CSV 文件，返回写入是否成功的布尔值
    return torch._C._cuda_tunableop_write_file(filename)  # type: ignore[attr-defined]


def read_file(filename: Optional[str] = None) -> bool:
    r"""Read results from a TunableOp CSV file.

    If :attr:`filename` is not given, ``get_filename()`` is called.
    """
    # 如果未提供 filename，则调用 get_filename() 获取文件名
    if filename is None:
        filename = get_filename()
    # 调用 torch._C._cuda_tunableop_read_file 函数，从 CSV 文件中读取结果，返回读取是否成功的布尔值
    return torch._C._cuda_tunableop_read_file(filename)  # type: ignore[attr-defined]
```