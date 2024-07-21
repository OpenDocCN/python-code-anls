# `.\pytorch\torch\backends\cudnn\__init__.py`

```
# mypy: allow-untyped-defs
# 引入必要的模块和库
import os  # 操作系统接口
import sys  # 系统相关的参数和功能
import warnings  # 警告相关的功能
from contextlib import contextmanager  # 上下文管理相关功能
from typing import Optional  # 类型提示相关

import torch  # PyTorch 深度学习框架
from torch.backends import __allow_nonbracketed_mutation, ContextProp, PropModule  # 导入相关子模块

try:
    from torch._C import _cudnn  # 尝试导入 _cudnn 模块
except ImportError:
    _cudnn = None  # 如果导入失败，将 _cudnn 设置为 None

# 全局设置，禁用 CuDNN/MIOpen
# 设置 torch.backends.cudnn.enabled = False 来全局禁用 CuDNN/MIOpen

__cudnn_version: Optional[int] = None  # 初始化 __cudnn_version 变量为可选的整数类型

if _cudnn is not None:

    def _init():
        global __cudnn_version
        if __cudnn_version is None:
            # 获取 cuDNN 版本信息
            __cudnn_version = _cudnn.getVersionInt()
            runtime_version = _cudnn.getRuntimeVersion()
            compile_version = _cudnn.getCompileVersion()
            runtime_major, runtime_minor, _ = runtime_version
            compile_major, compile_minor, _ = compile_version
            # 不同的主要版本总是不兼容
            # 从 cuDNN 7 开始，次要版本是向后兼容的
            # 对于 MIOpen (ROCm)，不确定，因此始终进行严格检查
            if runtime_major != compile_major:
                cudnn_compatible = False
            elif runtime_major < 7 or not _cudnn.is_cuda:
                cudnn_compatible = runtime_minor == compile_minor
            else:
                cudnn_compatible = runtime_minor >= compile_minor
            if not cudnn_compatible:
                # 检查是否需要跳过 cuDNN 兼容性检查
                if os.environ.get("PYTORCH_SKIP_CUDNN_COMPATIBILITY_CHECK", "0") == "1":
                    return True
                base_error_msg = (
                    f"cuDNN version incompatibility: "
                    f"PyTorch was compiled  against {compile_version} "
                    f"but found runtime version {runtime_version}. "
                    f"PyTorch already comes bundled with cuDNN. "
                    f"One option to resolving this error is to ensure PyTorch "
                    f"can find the bundled cuDNN. "
                )

                # 检查环境变量中的 LD_LIBRARY_PATH
                if "LD_LIBRARY_PATH" in os.environ:
                    ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
                    if any(
                        substring in ld_library_path for substring in ["cuda", "cudnn"]
                    ):
                        raise RuntimeError(
                            f"{base_error_msg}"
                            f"Looks like your LD_LIBRARY_PATH contains incompatible version of cudnn. "
                            f"Please either remove it from the path or install cudnn {compile_version}"
                        )
                    else:
                        raise RuntimeError(
                            f"{base_error_msg}"
                            f"one possibility is that there is a "
                            f"conflicting cuDNN in LD_LIBRARY_PATH."
                        )
                else:
                    raise RuntimeError(base_error_msg)

        return True

else:
    # 如果 _cudnn 为 None，返回 False
    def _init():
        return False
# 返回 cuDNN 的版本号
def version():
    # 如果 cuDNN 没有初始化成功，则返回 None
    if not _init():
        return None
    # 返回 cuDNN 的版本号
    return __cudnn_version


# 定义 cuDNN 支持的张量数据类型
CUDNN_TENSOR_DTYPES = {
    torch.half,
    torch.float,
    torch.double,
}


# 返回一个布尔值，指示当前是否可用 cuDNN
def is_available():
    # 返回一个布尔值，指示是否有 cuDNN
    return torch._C._has_cudnn


# 检查张量是否可接受 cuDNN
def is_acceptable(tensor):
    # 如果 cuDNN 未启用，则返回 False
    if not torch._C._get_cudnn_enabled():
        return False
    # 如果张量不在 CUDA 设备上或数据类型不在 cuDNN 支持的数据类型中，则返回 False
    if tensor.device.type != "cuda" or tensor.dtype not in CUDNN_TENSOR_DTYPES:
        return False
    # 如果 cuDNN 不可用，则发出警告
    if not is_available():
        warnings.warn(
            "PyTorch was compiled without cuDNN/MIOpen support. To use cuDNN/MIOpen, rebuild "
            "PyTorch making sure the library is visible to the build system."
        )
        return False
    # 如果 cuDNN 未初始化成功，则发出警告
    if not _init():
        warnings.warn(
            "cuDNN/MIOpen library not found. Check your {libpath}".format(
                libpath={"darwin": "DYLD_LIBRARY_PATH", "win32": "PATH"}.get(
                    sys.platform, "LD_LIBRARY_PATH"
                )
            )
        )
        return False
    return True


# 设置 cuDNN 的标志
def set_flags(
    _enabled=None,
    _benchmark=None,
    _benchmark_limit=None,
    _deterministic=None,
    _allow_tf32=None,
):
    # 获取当前 cuDNN 的标志
    orig_flags = (
        torch._C._get_cudnn_enabled(),
        torch._C._get_cudnn_benchmark(),
        None if not is_available() else torch._C._cuda_get_cudnn_benchmark_limit(),
        torch._C._get_cudnn_deterministic(),
        torch._C._get_cudnn_allow_tf32(),
    )
    # 如果指定了参数，则设置对应的 cuDNN 标志
    if _enabled is not None:
        torch._C._set_cudnn_enabled(_enabled)
    if _benchmark is not None:
        torch._C._set_cudnn_benchmark(_benchmark)
    if _benchmark_limit is not None and is_available():
        torch._C._cuda_set_cudnn_benchmark_limit(_benchmark_limit)
    if _deterministic is not None:
        torch._C._set_cudnn_deterministic(_deterministic)
    if _allow_tf32 is not None:
        torch._C._set_cudnn_allow_tf32(_allow_tf32)
    return orig_flags


# 上下文管理器，用于设置 cuDNN 的标志
@contextmanager
def flags(
    enabled=False,
    benchmark=False,
    benchmark_limit=10,
    deterministic=False,
    allow_tf32=True,
):
    # 设置 cuDNN 的标志，并返回原始标志
    with __allow_nonbracketed_mutation():
        orig_flags = set_flags(
            enabled, benchmark, benchmark_limit, deterministic, allow_tf32
        )
    try:
        yield
    finally:
        # 恢复之前的 cuDNN 标志
        with __allow_nonbracketed_mutation():
            set_flags(*orig_flags)


# 允许拦截代码如下：
# torch.backends.<cudnn|mkldnn>.enabled = True


# 定义 CudnnModule 类，用于处理 cuDNN 模块的属性
class CudnnModule(PropModule):
    def __init__(self, m, name):
        super().__init__(m, name)

    # 定义 enabled 属性，用于获取和设置 cuDNN 是否启用
    enabled = ContextProp(torch._C._get_cudnn_enabled, torch._C._set_cudnn_enabled)
    # 定义 deterministic 属性，用于获取和设置 cuDNN 是否确定性
    deterministic = ContextProp(
        torch._C._get_cudnn_deterministic, torch._C._set_cudnn_deterministic
    )
    # 定义 benchmark 属性，用于获取和设置 cuDNN 是否基准测试
    benchmark = ContextProp(
        torch._C._get_cudnn_benchmark, torch._C._set_cudnn_benchmark
    )
    # 定义 benchmark_limit 属性，暂未定义
    benchmark_limit = None
    # 如果 CUDA 可用
    if is_available():
        # 设置 cuDNN 的 benchmark 限制上下文属性
        benchmark_limit = ContextProp(
            torch._C._cuda_get_cudnn_benchmark_limit,
            torch._C._cuda_set_cudnn_benchmark_limit,
        )
    # 设置 cuDNN 是否允许使用 TF32 精度计算的上下文属性
    allow_tf32 = ContextProp(
        torch._C._get_cudnn_allow_tf32, torch._C._set_cudnn_allow_tf32
    )
# 使用 sys.modules 替换技巧，详情见 Stack Overflow 上的解释：
# https://stackoverflow.com/questions/2447353/getattr-on-a-module/7668273#7668273
# 将当前模块（__name__）替换为 CudnnModule 的实例，传入当前模块和模块名称作为参数
sys.modules[__name__] = CudnnModule(sys.modules[__name__], __name__)

# 为以下变量添加类型注解

# 表示是否启用 CUDNN 加速的标志，类型为布尔值
enabled: bool

# 表示是否启用 CUDNN 的确定性运行模式的标志，类型为布尔值
deterministic: bool

# 表示是否启用 CUDNN 的基准模式的标志，类型为布尔值
benchmark: bool

# 表示是否允许使用 TF32 精度的标志，类型为布尔值
allow_tf32: bool

# 用于控制 CUDNN 基准模式运行的限制数值，类型为整数
benchmark_limit: int
```