# `.\pytorch\torch\backends\cuda\__init__.py`

```py
# mypy: allow-untyped-defs
# 导入上下文管理模块
import contextlib

# 导入类型提示
from typing import Union
from typing_extensions import deprecated

# 导入 PyTorch 库
import torch

# 定义模块中公开的对象列表
__all__ = [
    "is_built",
    "cuFFTPlanCacheAttrContextProp",
    "cuFFTPlanCache",
    "cuFFTPlanCacheManager",
    "cuBLASModule",
    "preferred_linalg_library",
    "preferred_blas_library",
    "cufft_plan_cache",
    "matmul",
    "SDPAParams",
    "enable_cudnn_sdp",
    "cudnn_sdp_enabled",
    "enable_flash_sdp",
    "flash_sdp_enabled",
    "enable_mem_efficient_sdp",
    "mem_efficient_sdp_enabled",
    "math_sdp_enabled",
    "enable_math_sdp",
    "can_use_flash_attention",
    "can_use_efficient_attention",
    "sdp_kernel",
]


def is_built():
    r"""
    返回 PyTorch 是否构建了 CUDA 支持。

    注意，这并不意味着 CUDA 是可用的；只是如果在支持 CUDA 的设备上运行该 PyTorch 二进制文件，
    我们将能够使用 CUDA 支持。
    """
    return torch._C._has_cuda


class cuFFTPlanCacheAttrContextProp:
    # 类似于普通的 ContextProp，但使用调用对象的 `.device_index` 属性作为 getter 和 setter 的第一个参数。
    def __init__(self, getter, setter):
        self.getter = getter
        self.setter = setter

    def __get__(self, obj, objtype):
        return self.getter(obj.device_index)

    def __set__(self, obj, val):
        if isinstance(self.setter, str):
            raise RuntimeError(self.setter)
        self.setter(obj.device_index, val)


class cuFFTPlanCache:
    r"""
    表示特定 `device_index` 的 cuFFT 计划缓存。

    属性 `size` 和 `max_size`，以及 `clear` 方法，可以获取和/或更改 C++ cuFFT 计划缓存的属性。
    """

    def __init__(self, device_index):
        self.device_index = device_index

    # `size` 属性使用 `torch._cufft_get_plan_cache_size` 作为 getter，用于获取计划缓存中当前计划的数量。
    size = cuFFTPlanCacheAttrContextProp(
        torch._cufft_get_plan_cache_size,
        ".size is a read-only property showing the number of plans currently in the "
        "cache. To change the cache capacity, set cufft_plan_cache.max_size.",
    )

    # `max_size` 属性使用 `torch._cufft_get_plan_cache_max_size` 作为 getter，
    # 使用 `torch._cufft_set_plan_cache_max_size` 作为 setter，用于获取和设置计划缓存的最大容量。
    max_size = cuFFTPlanCacheAttrContextProp(
        torch._cufft_get_plan_cache_max_size, torch._cufft_set_plan_cache_max_size
    )

    # 清空当前设备的 cuFFT 计划缓存。
    def clear(self):
        return torch._cufft_clear_plan_cache(self.device_index)


class cuFFTPlanCacheManager:
    r"""
    表示所有 cuFFT 计划缓存，根据索引返回给定设备的 cuFFTPlanCache。

    当直接使用该对象作为 `cuFFTPlanCache` 对象（例如，设置 `.max_size` 属性）时，
    使用当前设备的 cuFFT 计划缓存。
    """

    # 类级别的标志，表示是否已初始化。
    __initialized = False

    def __init__(self):
        self.caches = []  # 存储所有的 cuFFT 计划缓存
        self.__initialized = True  # 标记已经初始化
    # 定义一个特殊方法，用于按索引访问对象的属性
    def __getitem__(self, device):
        # 获取设备的索引
        index = torch.cuda._utils._get_device_index(device)
        # 如果索引小于0或者大于等于设备数量，则引发运行时错误
        if index < 0 or index >= torch.cuda.device_count():
            raise RuntimeError(
                f"cufft_plan_cache: expected 0 <= device index < {torch.cuda.device_count()}, but got "
                f"device with index {index}"
            )
        # 如果缓存列表为空，则初始化缓存列表
        if len(self.caches) == 0:
            self.caches.extend(
                cuFFTPlanCache(index) for index in range(torch.cuda.device_count())
            )
        # 返回对应索引的缓存对象
        return self.caches[index]

    # 定义一个特殊方法，用于获取对象的属性
    def __getattr__(self, name):
        # 调用父对象的同名方法
        return getattr(self[torch.cuda.current_device()], name)

    # 定义一个特殊方法，用于设置对象的属性
    def __setattr__(self, name, value):
        # 如果对象已经初始化，则设置当前设备的属性值
        if self.__initialized:
            return setattr(self[torch.cuda.current_device()], name, value)
        # 否则调用父类的设置属性方法
        else:
            return super().__setattr__(name, value)
class cuBLASModule:
    # 定义一个自定义类 cuBLASModule，用于处理 cuBLAS 相关的属性获取和设置

    def __getattr__(self, name):
        # 当试图获取属性时调用，name 为属性名
        if name == "allow_tf32":
            # 如果属性名为 "allow_tf32"，返回 torch._C._get_cublas_allow_tf32() 的结果
            return torch._C._get_cublas_allow_tf32()
        elif name == "allow_fp16_reduced_precision_reduction":
            # 如果属性名为 "allow_fp16_reduced_precision_reduction"，返回 torch._C._get_cublas_allow_fp16_reduced_precision_reduction() 的结果
            return torch._C._get_cublas_allow_fp16_reduced_precision_reduction()
        elif name == "allow_bf16_reduced_precision_reduction":
            # 如果属性名为 "allow_bf16_reduced_precision_reduction"，返回 torch._C._get_cublas_allow_bf16_reduced_precision_reduction() 的结果
            return torch._C._get_cublas_allow_bf16_reduced_precision_reduction()
        # 如果属性名未知，抛出 AttributeError 异常
        raise AttributeError("Unknown attribute " + name)

    def __setattr__(self, name, value):
        # 当试图设置属性时调用，name 为属性名，value 为属性值
        if name == "allow_tf32":
            # 如果属性名为 "allow_tf32"，调用 torch._C._set_cublas_allow_tf32(value)
            return torch._C._set_cublas_allow_tf32(value)
        elif name == "allow_fp16_reduced_precision_reduction":
            # 如果属性名为 "allow_fp16_reduced_precision_reduction"，调用 torch._C._set_cublas_allow_fp16_reduced_precision_reduction(value)
            return torch._C._set_cublas_allow_fp16_reduced_precision_reduction(value)
        elif name == "allow_bf16_reduced_precision_reduction":
            # 如果属性名为 "allow_bf16_reduced_precision_reduction"，调用 torch._C._set_cublas_allow_bf16_reduced_precision_reduction(value)
            return torch._C._set_cublas_allow_bf16_reduced_precision_reduction(value)
        # 如果属性名未知，抛出 AttributeError 异常
        raise AttributeError("Unknown attribute " + name)


_LinalgBackends = {
    "default": torch._C._LinalgBackend.Default,
    "cusolver": torch._C._LinalgBackend.Cusolver,
    "magma": torch._C._LinalgBackend.Magma,
}
_LinalgBackends_str = ", ".join(_LinalgBackends.keys())

def preferred_linalg_library(
    backend: Union[None, str, torch._C._LinalgBackend] = None
) -> torch._C._LinalgBackend:
    r"""
    Override the heuristic PyTorch uses to choose between cuSOLVER and MAGMA for CUDA linear algebra operations.

    .. warning:: This flag is experimental and subject to change.

    When PyTorch runs a CUDA linear algebra operation it often uses the cuSOLVER or MAGMA libraries,
    and if both are available it decides which to use with a heuristic.
    This flag (a :class:`str`) allows overriding those heuristics.

    * If `"cusolver"` is set then cuSOLVER will be used wherever possible.
    * If `"magma"` is set then MAGMA will be used wherever possible.
    * If `"default"` (the default) is set then heuristics will be used to pick between
      cuSOLVER and MAGMA if both are available.
    * When no input is given, this function returns the currently preferred library.
    * User may use the environment variable TORCH_LINALG_PREFER_CUSOLVER=1 to set the preferred library to cuSOLVER
      globally.
      This flag only sets the initial value of the preferred library and the preferred library
      may still be overridden by this function call later in your script.

    Note: When a library is preferred other libraries may still be used if the preferred library
    doesn't implement the operation(s) called.
    This flag may achieve better performance if PyTorch's heuristic library selection is incorrect
    for your application's inputs.

    Currently supported linalg operators:

    * :func:`torch.linalg.inv`
    * :func:`torch.linalg.inv_ex`
    * :func:`torch.linalg.cholesky`
    * :func:`torch.linalg.cholesky_ex`
    * :func:`torch.cholesky_solve`
    * :func:`torch.cholesky_inverse`
    * :func:`torch.linalg.lu_factor`
    # 根据指定的后端设置首选的线性代数库
    def _set_linalg_backend(backend=None):
        """
        * :func:`torch.linalg.lu`
        * :func:`torch.linalg.lu_solve`
        * :func:`torch.linalg.qr`
        * :func:`torch.linalg.eigh`
        * :func:`torch.linalg.eighvals`
        * :func:`torch.linalg.svd`
        * :func:`torch.linalg.svdvals`
        """
    
        # 如果未指定后端，则不做任何操作
        if backend is None:
            pass
        # 如果指定的后端是一个字符串
        elif isinstance(backend, str):
            # 检查该字符串是否为已知的线性代数后端之一
            if backend not in _LinalgBackends:
                raise RuntimeError(
                    "Unknown input value. " f"Choose from: {_LinalgBackends_str}."
                )
            # 设置 Torch 的首选线性代数后端为指定的字符串所对应的后端
            torch._C._set_linalg_preferred_backend(_LinalgBackends[backend])
        # 如果指定的后端是一个 torch._C._LinalgBackend 类型的对象
        elif isinstance(backend, torch._C._LinalgBackend):
            # 设置 Torch 的首选线性代数后端为该对象
            torch._C._set_linalg_preferred_backend(backend)
        # 如果输入的后端类型未知，则抛出运行时错误
        else:
            raise RuntimeError("Unknown input value type.")
    
        # 返回当前 Torch 首选的线性代数后端
        return torch._C._get_linalg_preferred_backend()
# 定义一个字典，将字符串表示的 BLAS 后端映射到对应的 Torch C++ BLAS 后端类
_BlasBackends = {
    "cublas": torch._C._BlasBackend.Cublas,
    "cublaslt": torch._C._BlasBackend.Cublaslt,
    "hipblaslt": torch._C._BlasBackend.Cublaslt,  # 别名
}
# 将字典中的键以字符串形式连接起来，用于错误信息中的选择提示
_BlasBackends_str = ", ".join(_BlasBackends.keys())


def preferred_blas_library(
    backend: Union[None, str, torch._C._BlasBackend] = None
) -> torch._C._BlasBackend:
    r"""
    重写 PyTorch 用于 BLAS 操作的库的选择，可选 cuBLAS 和 cuBLASLt。

    .. warning:: 此标志是实验性的，可能会发生变化。

    当 PyTorch 执行 CUDA BLAS 操作时，默认使用 cuBLAS，即使 cuBLAS 和 cuBLASLt 都可用。
    对于为 ROCm 构建的 PyTorch，hipBLAS 和 hipBLASLt 可能提供不同的性能。
    此标志（一个 :class:`str`）允许覆盖使用的 BLAS 库的选择。

    * 如果设置为 `"cublas"`，则尽可能使用 cuBLAS。
    * 如果设置为 `"cublaslt"`，则尽可能使用 cuBLASLt。
    * 当未提供输入时，此函数返回当前首选库。
    * 用户可以使用环境变量 TORCH_BLAS_PREFER_CUBLASLT=1 全局设置首选库为 cuBLASLt。
      此标志仅设置首选库的初始值，首选库仍可能在脚本中后续调用此函数时被覆盖。

    注意：当首选库不实现调用的操作时，仍可能使用其他库。
    如果 PyTorch 的库选择对应用程序输入不正确，此标志可能会实现更好的性能。

    """
    # 如果 backend 为 None，则不做任何操作，返回当前首选库
    if backend is None:
        pass
    # 如果 backend 是字符串类型，则根据字符串选择对应的 BLAS 后端并设置
    elif isinstance(backend, str):
        if backend not in _BlasBackends:
            raise RuntimeError(
                "Unknown input value. " f"Choose from: {_BlasBackends_str}."
            )
        torch._C._set_blas_preferred_backend(_BlasBackends[backend])
    # 如果 backend 是 torch._C._BlasBackend 类型，则直接设置为首选库
    elif isinstance(backend, torch._C._BlasBackend):
        torch._C._set_blas_preferred_backend(backend)
    # 如果 backend 类型未知，则引发运行时错误
    else:
        raise RuntimeError("Unknown input value type.")

    # 返回当前首选的 BLAS 后端
    return torch._C._get_blas_preferred_backend()


from torch._C import _SDPAParams as SDPAParams, _SDPBackend as SDPBackend

# 设置 SDPAParams 类的 __module__ 属性
SDPAParams.__module__ = "torch.backends.cuda"
# 设置 SDPAParams 类的 __name__ 属性
SDPAParams.__name__ = "SDPAParams"


def flash_sdp_enabled():
    r"""
    .. warning:: 此标志是 beta 版本，可能会发生变化。

    返回是否启用了 flash scaled dot product attention。
    """
    return torch._C._get_flash_sdp_enabled()


def enable_flash_sdp(enabled: bool):
    r"""
    .. warning:: 此标志是 beta 版本，可能会发生变化。

    启用或禁用 flash scaled dot product attention。
    """
    torch._C._set_sdp_use_flash(enabled)


def mem_efficient_sdp_enabled():
    r"""
    .. warning:: 此标志是 beta 版本，可能会发生变化。

    返回是否启用了 memory efficient scaled dot product attention。
    """
    # 调用 torch 库的 C++ 扩展接口，获取内存效率相关的配置是否启用
    return torch._C._get_mem_efficient_sdp_enabled()
def enable_mem_efficient_sdp(enabled: bool):
    r"""
    .. warning:: This flag is beta and subject to change.

    Enables or disables memory efficient scaled dot product attention.
    """
    # 调用 PyTorch C++ 接口，设置是否启用内存高效的缩放点积注意力
    torch._C._set_sdp_use_mem_efficient(enabled)


def math_sdp_enabled():
    r"""
    .. warning:: This flag is beta and subject to change.

    Returns whether math scaled dot product attention is enabled or not.
    """
    # 返回当前是否启用数学缩放点积注意力的状态
    return torch._C._get_math_sdp_enabled()


def enable_math_sdp(enabled: bool):
    r"""
    .. warning:: This flag is beta and subject to change.

    Enables or disables math scaled dot product attention.
    """
    # 调用 PyTorch C++ 接口，设置是否启用数学缩放点积注意力
    torch._C._set_sdp_use_math(enabled)


def can_use_flash_attention(params: SDPAParams, debug: bool = False) -> bool:
    r"""Check if FlashAttention can be utilized in scaled_dot_product_attention.

    Args:
        params: An instance of SDPAParams containing the tensors for query,
                key, value, an optional attention mask, dropout rate, and
                a flag indicating if the attention is causal.
        debug: Whether to logging.warn debug information as to why FlashAttention could not be run.
            Defaults to False.

    Returns:
        True if FlashAttention can be used with the given parameters; otherwise, False.

    Note:
        This function is dependent on a CUDA-enabled build of PyTorch. It will return False
        in non-CUDA environments.
    """
    # 调用 PyTorch C++ 接口，检查是否可以使用 FlashAttention
    return torch._C._can_use_flash_attention(params, debug)


def can_use_efficient_attention(params: SDPAParams, debug: bool = False) -> bool:
    r"""Check if efficient_attention can be utilized in scaled_dot_product_attention.

    Args:
        params: An instance of SDPAParams containing the tensors for query,
                key, value, an optional attention mask, dropout rate, and
                a flag indicating if the attention is causal.
        debug: Whether to logging.warn with information as to why efficient_attention could not be run.
            Defaults to False.

    Returns:
        True if efficient_attention can be used with the given parameters; otherwise, False.

    Note:
        This function is dependent on a CUDA-enabled build of PyTorch. It will return False
        in non-CUDA environments.
    """
    # 调用 PyTorch C++ 接口，检查是否可以使用高效注意力机制
    return torch._C._can_use_mem_efficient_attention(params, debug)


def cudnn_sdp_enabled():
    r"""
    .. warning:: This flag is beta and subject to change.

    Returns whether cuDNN scaled dot product attention is enabled or not.
    """
    # 返回当前是否启用 cuDNN 缩放点积注意力的状态
    return torch._C._get_cudnn_sdp_enabled()


def enable_cudnn_sdp(enabled: bool):
    r"""
    .. warning:: This flag is beta and subject to change.

    Enables or disables cuDNN scaled dot product attention.
    """
    # 调用 PyTorch C++ 接口，设置是否启用 cuDNN 缩放点积注意力
    torch._C._set_sdp_use_cudnn(enabled)


@contextlib.contextmanager
@deprecated(
    (
        "`torch.backends.cuda.sdp_kernel()` is deprecated. "
        "In the future, this context manager will be removed. "
        "Please see `torch.nn.attention.sdpa_kernel()` for the new context manager, "
        "with updated signature."
    ),
    category=FutureWarning,
    
    
    
    # 发出未来警告，指出 `torch.backends.cuda.sdp_kernel()` 已弃用，并说明未来将移除此上下文管理器
    (
        "`torch.backends.cuda.sdp_kernel()` is deprecated. "
        "In the future, this context manager will be removed. "
        "Please see `torch.nn.attention.sdpa_kernel()` for the new context manager, "
        "with updated signature."
    ),
    # 将警告归类为“未来警告”，以提醒用户将来可能会发生的变化
    category=FutureWarning,
)
def sdp_kernel(
    enable_flash: bool = True,
    enable_math: bool = True,
    enable_mem_efficient: bool = True,
    enable_cudnn: bool = True,
):
    r"""
    .. warning:: This flag is beta and subject to change.

    This context manager can be used to temporarily enable or disable any of the three backends for scaled dot product attention.
    Upon exiting the context manager, the previous state of the flags will be restored.
    """
    # 导入 scaled dot product attention 的核心函数 sdpa_kernel
    from torch.nn.attention import sdpa_kernel

    # 创建一个空列表，用于存储选择的后端类型
    backend_list = []

    # 根据 enable_flash 的值决定是否启用 FLASH_ATTENTION 后端
    if enable_flash:
        backend_list.append(SDPBackend.FLASH_ATTENTION)
    
    # 根据 enable_mem_efficient 的值决定是否启用 EFFICIENT_ATTENTION 后端
    if enable_mem_efficient:
        backend_list.append(SDPBackend.EFFICIENT_ATTENTION)
    
    # 根据 enable_math 的值决定是否启用 MATH 后端
    if enable_math:
        backend_list.append(SDPBackend.MATH)
    
    # 根据 enable_cudnn 的值决定是否启用 CUDNN_ATTENTION 后端
    if enable_cudnn:
        backend_list.append(SDPBackend.CUDNN_ATTENTION)

    # 进入 sdpa_kernel 上下文管理器，传入选择的后端列表
    with sdpa_kernel(backend_list) as context:
        try:
            # 使用 yield 语句返回上下文 context
            yield context
        finally:
            # 在 finally 块中不执行任何操作，保留原样以恢复上下文状态

# 创建 cuFFTPlanCacheManager 实例 cufft_plan_cache
cufft_plan_cache = cuFFTPlanCacheManager()

# 创建 cuBLASModule 实例 matmul
matmul = cuBLASModule()
```