# `stable-diffusion-webui\modules\devices.py`

```py
# 导入必要的模块
import sys
import contextlib
from functools import lru_cache

import torch
from modules import errors, shared

# 如果操作系统是 macOS，则导入 macOS 特定模块
if sys.platform == "darwin":
    from modules import mac_specific

# 如果使用了 IPEx，则导入 XPU 特定模块
if shared.cmd_opts.use_ipex:
    from modules import xpu_specific

# 检查是否有 XPU 设备可用
def has_xpu() -> bool:
    return shared.cmd_opts.use_ipex and xpu_specific.has_xpu

# 检查是否有 MPS 设备可用
def has_mps() -> bool:
    if sys.platform != "darwin":
        return False
    else:
        return mac_specific.has_mps

# 获取 CUDA 设备字符串
def get_cuda_device_string():
    if shared.cmd_opts.device_id is not None:
        return f"cuda:{shared.cmd_opts.device_id}"

    return "cuda"

# 获取最佳设备名称
def get_optimal_device_name():
    if torch.cuda.is_available():
        return get_cuda_device_string()

    if has_mps():
        return "mps"

    if has_xpu():
        return xpu_specific.get_xpu_device_string()

    return "cpu"

# 获取最佳设备
def get_optimal_device():
    return torch.device(get_optimal_device_name())

# 为任务获取设备
def get_device_for(task):
    if task in shared.cmd_opts.use_cpu or "all" in shared.cmd_opts.use_cpu:
        return cpu

    return get_optimal_device()

# 执行 Torch 的垃圾回收
def torch_gc():

    # 如果 CUDA 可用，则清空 CUDA 缓存和 IPC 收集
    if torch.cuda.is_available():
        with torch.cuda.device(get_cuda_device_string()):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    # 如果有 MPS 设备，则执行 MPS 的 Torch 垃圾回收
    if has_mps():
        mac_specific.torch_mps_gc()

    # 如果有 XPU 设备，则执行 XPU 的 Torch 垃圾回收
    if has_xpu():
        xpu_specific.torch_xpu_gc()

# 启用 TF32
def enable_tf32():
    # 检查是否有可用的 CUDA 设备
    if torch.cuda.is_available():

        # 启用 benchmark 选项似乎可以使一系列卡在其他情况下无法进行 fp16 运算
        # 参考 https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/4407
        # 确定设备 ID，如果未指定则默认为 0，或者根据当前设备获取设备 ID
        device_id = (int(shared.cmd_opts.device_id) if shared.cmd_opts.device_id is not None and shared.cmd_opts.device_id.isdigit() else 0) or torch.cuda.current_device()
        
        # 如果设备是 NVIDIA GeForce GTX 16 系列且计算能力为 7.5，则启用 cudnn 的 benchmark
        if torch.cuda.get_device_capability(device_id) == (7, 5) and torch.cuda.get_device_name(device_id).startswith("NVIDIA GeForce GTX 16"):
            torch.backends.cudnn.benchmark = True

        # 允许 CUDA 进行 tf32 运算
        torch.backends.cuda.matmul.allow_tf32 = True
        # 允许 cudnn 进行 tf32 运算
        torch.backends.cudnn.allow_tf32 = True
# 运行错误处理函数，启用 TF32
errors.run(enable_tf32, "Enabling TF32")

# 定义 CPU 设备
cpu: torch.device = torch.device("cpu")
# 定义设备变量，初始值为 None
device: torch.device = None
device_interrogate: torch.device = None
device_gfpgan: torch.device = None
device_esrgan: torch.device = None
device_codeformer: torch.device = None
# 定义数据类型变量，初始值为 torch.float16
dtype: torch.dtype = torch.float16
dtype_vae: torch.dtype = torch.float16
dtype_unet: torch.dtype = torch.float16
# 定义是否需要升级数据类型的标志变量，初始值为 False
unet_needs_upcast = False

# 条件转换函数，根据 unet_needs_upcast 变量决定是否将输入转换为 dtype_unet 数据类型
def cond_cast_unet(input):
    return input.to(dtype_unet) if unet_needs_upcast else input

# 条件转换函数，根据 unet_needs_upcast 变量决定是否将输入转换为 float 数据类型
def cond_cast_float(input):
    return input.float() if unet_needs_upcast else input

# 定义 nv_rng 变量，初始值为 None
nv_rng = None

# 自动类型转换函数，根据 disable 参数决定是否启用自动类型转换
def autocast(disable=False):
    if disable:
        return contextlib.nullcontext()

    if dtype == torch.float32 or shared.cmd_opts.precision == "full":
        return contextlib.nullcontext()

    return torch.autocast("cuda")

# 关闭自动类型转换函数，根据 disable 参数决定是否关闭自动类型转换
def without_autocast(disable=False):
    return torch.autocast("cuda", enabled=False) if torch.is_autocast_enabled() and not disable else contextlib.nullcontext()

# 定义 NansException 异常类
class NansException(Exception):
    pass

# 检查是否存在 NaN 值的函数
def test_for_nans(x, where):
    if shared.cmd_opts.disable_nan_check:
        return

    if not torch.all(torch.isnan(x)).item():
        return

    if where == "unet":
        message = "A tensor with all NaNs was produced in Unet."

        if not shared.cmd_opts.no_half:
            message += " This could be either because there's not enough precision to represent the picture, or because your video card does not support half type. Try setting the \"Upcast cross attention layer to float32\" option in Settings > Stable Diffusion or using the --no-half commandline argument to fix this."

    elif where == "vae":
        message = "A tensor with all NaNs was produced in VAE."

        if not shared.cmd_opts.no_half and not shared.cmd_opts.no_half_vae:
            message += " This could be because there's not enough precision to represent the picture. Try adding --no-half-vae commandline argument to fix this."
    # 如果条件不满足，说明产生了全为 NaN 的张量
    else:
        # 设置消息为“产生了全为 NaN 的张量”
        message = "A tensor with all NaNs was produced."

    # 在消息后面添加一条建议，使用命令行参数 --disable-nan-check 来禁用此检查
    message += " Use --disable-nan-check commandline argument to disable this check."

    # 抛出自定义异常 NansException，并传入消息
    raise NansException(message)
# 使用 lru_cache 装饰器，缓存函数的结果，避免重复计算
@lru_cache
def first_time_calculation():
    """
    第一次计算时，使用 PyTorch 层进行任何计算 - 这将分配大约 700MB 的内存，并且在 NVidia 设备上大约需要 2.7 秒。
    """

    # 创建一个形状为 (1, 1) 的零张量，并将其移动到指定设备上，并指定数据类型
    x = torch.zeros((1, 1)).to(device, dtype)
    # 创建一个线性层，并将其移动到指定设备上，并指定数据类型
    linear = torch.nn.Linear(1, 1).to(device, dtype)
    # 将输入张量传递给线性层进行计算

    linear(x)

    # 创建一个形状为 (1, 1, 3, 3) 的零张量，并将其移动到指定设备上，并指定数据类型
    x = torch.zeros((1, 1, 3, 3)).to(device, dtype)
    # 创建一个二维卷积层，并将其移动到指定设备上，并指定数据类型
    conv2d = torch.nn.Conv2d(1, 1, (3, 3)).to(device, dtype)
    # 将输入张量传递给卷积层进行计算
    conv2d(x)
```