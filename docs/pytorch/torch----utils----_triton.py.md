# `.\pytorch\torch\utils\_triton.py`

```
# mypy: allow-untyped-defs
# 引入 functools 模块，用于实现函数的缓存
import functools
# 引入 hashlib 模块，用于计算哈希值
import hashlib

# 使用 functools 提供的装饰器 lru_cache，缓存函数的返回值，避免重复计算
@functools.lru_cache(None)
# 检查是否安装了 Triton 软件包，返回布尔值
def has_triton_package() -> bool:
    try:
        # 尝试导入 triton 库
        import triton

        # 如果导入成功，返回 triton 是否不为 None
        return triton is not None
    except ImportError:
        # 导入失败则返回 False
        return False


@functools.lru_cache(None)
# 检查是否具有 Triton 支持的硬件设备，返回布尔值
def has_triton() -> bool:
    # 从 torch 库中导入 _dynamo 的 device_interface 模块的 get_interface_for_device 函数
    from torch._dynamo.device_interface import get_interface_for_device

    # 定义一个检查 CUDA 设备是否支持 Triton 的函数
    def cuda_extra_check(device_interface):
        return device_interface.Worker.get_device_properties().major >= 7

    # 返回始终为 True 的函数
    def _return_true(device_interface):
        return True

    # Triton 支持的设备及其对应的检查函数
    triton_supported_devices = {"cuda": cuda_extra_check, "xpu": _return_true}

    # 检查当前设备是否兼容 Triton
    def is_device_compatible_with_triton():
        for device, extra_check in triton_supported_devices.items():
            device_interface = get_interface_for_device(device)
            if device_interface.is_available() and extra_check(device_interface):
                return True
        return False

    # 返回当前设备是否兼容 Triton 且已安装 Triton 软件包的结果
    return is_device_compatible_with_triton() and has_triton_package()


@functools.lru_cache(None)
# 返回 Triton 后端对象
def triton_backend():
    # 从 torch 库中导入 triton 模块
    import torch

    # 如果是 HIP 版本的 torch，则不支持 Triton，返回 None
    if torch.version.hip:
        return None

    # 从 triton 编译器中导入 make_backend 函数和运行时驱动 driver
    from triton.compiler.compiler import make_backend
    from triton.runtime.driver import driver

    # 获取当前活动的目标设备
    target = driver.active.get_current_target()
    # 使用当前目标设备创建并返回 Triton 后端对象
    return make_backend(target)


@functools.lru_cache(None)
# 返回基于 Triton 后端的哈希值
def triton_hash_with_backend():
    # 从 torch 库中导入 triton 模块
    import torch

    # 如果是 HIP 版本的 torch，则不支持 Triton，返回 None
    if torch.version.hip:
        return None

    # 从 triton 编译器中导入 triton_key 函数
    from triton.compiler.compiler import triton_key

    # 获取 Triton 后端对象
    backend = triton_backend()
    # 创建包含 Triton 键和后端哈希值的字符串
    key = f"{triton_key()}-{backend.hash()}"

    # 将字符串编码为 UTF-8，计算其 SHA-256 哈希值，并转换为大写形式
    return hashlib.sha256(key.encode("utf-8")).hexdigest().upper()


# 定义函数，将 dtype 转换为对应的字符串表示形式
def dtype_to_string(dtype):
    # 如果 dtype 的名称以 "fp" 开头，则构造对应的浮点数字符串
    if dtype.name.startswith("fp"):
        suffix = "float" + dtype.name[2:]
    # 如果 dtype 的名称以 "bf" 开头，则构造对应的 bfloat 字符串
    elif dtype.name.startswith("bf"):
        suffix = "bfloat" + dtype.name[2:]
    else:
        # 否则直接使用 dtype 的名称
        suffix = dtype.name
    # 返回以 "triton.language." 开头的字符串表示形式
    return "triton.language." + suffix


@functools.lru_cache(None)
# 修改 Triton dtype 的 repr 方法，使其能够生成可评估的表达式
def patch_triton_dtype_repr():
    # 从 triton 库中导入 triton 模块
    import triton

    # 重定义 triton.language.dtype 的 __repr__ 方法，使其返回可评估的字符串表示形式
    triton.language.dtype.__repr__ = lambda self: dtype_to_string(self)
```