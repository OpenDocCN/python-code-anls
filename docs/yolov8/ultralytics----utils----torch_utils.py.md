# `.\yolov8\ultralytics\utils\torch_utils.py`

```py
# Ultralytics YOLO , AGPL-3.0 license

import gc  # 导入垃圾回收模块
import math  # 导入数学模块
import os  # 导入操作系统模块
import random  # 导入随机数模块
import time  # 导入时间模块
from contextlib import contextmanager  # 导入上下文管理器模块
from copy import deepcopy  # 导入深拷贝函数
from datetime import datetime  # 导入日期时间模块
from pathlib import Path  # 导入路径模块
from typing import Union  # 导入类型注解

import numpy as np  # 导入NumPy库
import torch  # 导入PyTorch库
import torch.distributed as dist  # 导入PyTorch分布式训练模块
import torch.nn as nn  # 导入PyTorch神经网络模块
import torch.nn.functional as F  # 导入PyTorch函数模块

from ultralytics.utils import (  # 导入Ultralytics工具函数
    DEFAULT_CFG_DICT,  # 默认配置字典
    DEFAULT_CFG_KEYS,  # 默认配置键列表
    LOGGER,  # 日志记录器
    NUM_THREADS,  # 线程数
    PYTHON_VERSION,  # Python版本
    TORCHVISION_VERSION,  # TorchVision版本
    __version__,  # Ultralytics版本
    colorstr,  # 字符串颜色化函数
)
from ultralytics.utils.checks import check_version  # 导入版本检查函数

try:
    import thop  # 尝试导入thop库
except ImportError:
    thop = None  # 如果导入失败，设为None

# Version checks (all default to version>=min_version)
TORCH_1_9 = check_version(torch.__version__, "1.9.0")  # 检查PyTorch版本是否>=1.9.0
TORCH_1_13 = check_version(torch.__version__, "1.13.0")  # 检查PyTorch版本是否>=1.13.0
TORCH_2_0 = check_version(torch.__version__, "2.0.0")  # 检查PyTorch版本是否>=2.0.0
TORCHVISION_0_10 = check_version(TORCHVISION_VERSION, "0.10.0")  # 检查TorchVision版本是否>=0.10.0
TORCHVISION_0_11 = check_version(TORCHVISION_VERSION, "0.11.0")  # 检查TorchVision版本是否>=0.11.0
TORCHVISION_0_13 = check_version(TORCHVISION_VERSION, "0.13.0")  # 检查TorchVision版本是否>=0.13.0
TORCHVISION_0_18 = check_version(TORCHVISION_VERSION, "0.18.0")  # 检查TorchVision版本是否>=0.18.0


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """Ensures all processes in distributed training wait for the local master (rank 0) to complete a task first."""
    initialized = dist.is_available() and dist.is_initialized()  # 检查是否启用了分布式训练且是否已初始化
    if initialized and local_rank not in {-1, 0}:  # 如果初始化且当前进程不是主进程（rank 0）
        dist.barrier(device_ids=[local_rank])  # 等待本地主节点（rank 0）完成任务
    yield  # 执行上下文管理器的主体部分
    if initialized and local_rank == 0:  # 如果初始化且当前进程是主进程（rank 0）
        dist.barrier(device_ids=[0])  # 确保所有进程在继续之前都等待主进程完成


def smart_inference_mode():
    """Applies torch.inference_mode() decorator if torch>=1.9.0 else torch.no_grad() decorator."""
    
    def decorate(fn):
        """Applies appropriate torch decorator for inference mode based on torch version."""
        if TORCH_1_9 and torch.is_inference_mode_enabled():
            return fn  # 如果已启用推断模式，直接返回函数
        else:
            return (torch.inference_mode if TORCH_1_9 else torch.no_grad)()(fn)  # 根据版本选择合适的推断模式装饰器

    return decorate


def autocast(enabled: bool, device: str = "cuda"):
    """
    Get the appropriate autocast context manager based on PyTorch version and AMP setting.

    This function returns a context manager for automatic mixed precision (AMP) training that is compatible with both
    older and newer versions of PyTorch. It handles the differences in the autocast API between PyTorch versions.

    Args:
        enabled (bool): Whether to enable automatic mixed precision.
        device (str, optional): The device to use for autocast. Defaults to 'cuda'.

    Returns:
        (torch.amp.autocast): The appropriate autocast context manager.

    Note:
        - For PyTorch versions 1.13 and newer, it uses `torch.amp.autocast`.
        - For older versions, it uses `torch.cuda.autocast`.

    Example:
        ```py
        with autocast(amp=True):
            # Your mixed precision operations here
            pass
        ```py
    """
    # 如果 TORCH_1_13 变量为真，使用 torch.amp.autocast 方法开启自动混合精度模式
    if TORCH_1_13:
        return torch.amp.autocast(device, enabled=enabled)
    # 如果 TORCH_1_13 变量为假，使用 torch.cuda.amp.autocast 方法开启自动混合精度模式
    else:
        return torch.cuda.amp.autocast(enabled)
def get_cpu_info():
    """Return a string with system CPU information, i.e. 'Apple M2'."""
    import cpuinfo  # 导入cpuinfo库，用于获取CPU信息，需使用pip安装py-cpuinfo

    k = "brand_raw", "hardware_raw", "arch_string_raw"  # 按优先顺序列出信息键（并非所有键始终可用）
    info = cpuinfo.get_cpu_info()  # 获取CPU信息的字典
    string = info.get(k[0] if k[0] in info else k[1] if k[1] in info else k[2], "unknown")  # 提取CPU信息字符串
    return string.replace("(R)", "").replace("CPU ", "").replace("@ ", "")  # 处理特殊字符后返回CPU信息字符串


def select_device(device="", batch=0, newline=False, verbose=True):
    """
    Selects the appropriate PyTorch device based on the provided arguments.

    The function takes a string specifying the device or a torch.device object and returns a torch.device object
    representing the selected device. The function also validates the number of available devices and raises an
    exception if the requested device(s) are not available.

    Args:
        device (str | torch.device, optional): Device string or torch.device object.
            Options are 'None', 'cpu', or 'cuda', or '0' or '0,1,2,3'. Defaults to an empty string, which auto-selects
            the first available GPU, or CPU if no GPU is available.
        batch (int, optional): Batch size being used in your model. Defaults to 0.
        newline (bool, optional): If True, adds a newline at the end of the log string. Defaults to False.
        verbose (bool, optional):
    elif device:  # 非 CPU 设备请求时执行以下操作
        if device == "cuda":
            device = "0"
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        os.environ["CUDA_VISIBLE_DEVICES"] = device  # 设置环境变量，必须在检查可用性之前设置
        if not (torch.cuda.is_available() and torch.cuda.device_count() >= len(device.split(","))):
            LOGGER.info(s)  # 记录信息到日志
            install = (
                "See https://pytorch.org/get-started/locally/ for up-to-date torch install instructions if no "
                "CUDA devices are seen by torch.\n"
                if torch.cuda.device_count() == 0
                else ""
            )
            raise ValueError(
                f"Invalid CUDA 'device={device}' requested."
                f" Use 'device=cpu' or pass valid CUDA device(s) if available,"
                f" i.e. 'device=0' or 'device=0,1,2,3' for Multi-GPU.\n"
                f"\ntorch.cuda.is_available(): {torch.cuda.is_available()}"
                f"\ntorch.cuda.device_count(): {torch.cuda.device_count()}"
                f"\nos.environ['CUDA_VISIBLE_DEVICES']: {visible}\n"
                f"{install}"
            )

    if not cpu and not mps and torch.cuda.is_available():  # 如果可用且未请求 CPU 或 MPS
        devices = device.split(",") if device else "0"  # 定义设备列表，默认为 "0"
        n = len(devices)  # 设备数量
        if n > 1:  # 多 GPU 情况
            if batch < 1:
                raise ValueError(
                    "AutoBatch with batch<1 not supported for Multi-GPU training, "
                    "please specify a valid batch size, i.e. batch=16."
                )
            if batch >= 0 and batch % n != 0:  # 检查 batch_size 是否可以被设备数量整除
                raise ValueError(
                    f"'batch={batch}' must be a multiple of GPU count {n}. Try 'batch={batch // n * n}' or "
                    f"'batch={batch // n * n + n}', the nearest batch sizes evenly divisible by {n}."
                )
        space = " " * (len(s) + 1)  # 创建空格串
        for i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / (1 << 20):.0f}MiB)\n"  # 字符串拼接 GPU 信息
        arg = "cuda:0"  # 设置 CUDA 设备为默认值 "cuda:0"
    elif mps and TORCH_2_0 and torch.backends.mps.is_available():
        # 如果支持 MPS 并且满足条件，则优先选择 MPS
        s += f"MPS ({get_cpu_info()})\n"  # 添加 MPS 信息到字符串
        arg = "mps"  # 设置设备类型为 "mps"
    else:  # 否则，默认使用 CPU
        s += f"CPU ({get_cpu_info()})\n"  # 添加 CPU 信息到字符串
        arg = "cpu"  # 设置设备类型为 "cpu"

    if arg in {"cpu", "mps"}:
        torch.set_num_threads(NUM_THREADS)  # 设置 CPU 训练的线程数
    if verbose:
        LOGGER.info(s if newline else s.rstrip())  # 如果需要详细输出，则记录详细信息到日志
    return torch.device(arg)  # 返回对应的 Torch 设备对象
# 返回当前系统时间，确保在使用 PyTorch 时精确同步时间
def time_sync():
    """PyTorch-accurate time."""
    # 如果 CUDA 可用，同步 CUDA 计算的时间
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    # 返回当前时间戳
    return time.time()


# 将 Conv2d() 和 BatchNorm2d() 层融合，实现优化 https://tehnokv.com/posts/fusing-batchnorm-and-conv/
def fuse_conv_and_bn(conv, bn):
    """Fuse Conv2d() and BatchNorm2d() layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/."""
    # 创建融合后的卷积层对象
    fusedconv = (
        nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=True,
        )
        .requires_grad_(False)  # 禁用梯度追踪，不需要反向传播训练
        .to(conv.weight.device)  # 将融合后的卷积层移到与输入卷积层相同的设备上
    )

    # 准备卷积层的权重
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    # 计算融合后的权重
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # 准备空间偏置项
    b_conv = torch.zeros(conv.weight.shape[0], device=conv.weight.device) if conv.bias is None else conv.bias
    # 计算融合后的偏置项
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv


# 将 ConvTranspose2d() 和 BatchNorm2d() 层融合
def fuse_deconv_and_bn(deconv, bn):
    """Fuse ConvTranspose2d() and BatchNorm2d() layers."""
    # 创建融合后的反卷积层对象
    fuseddconv = (
        nn.ConvTranspose2d(
            deconv.in_channels,
            deconv.out_channels,
            kernel_size=deconv.kernel_size,
            stride=deconv.stride,
            padding=deconv.padding,
            output_padding=deconv.output_padding,
            dilation=deconv.dilation,
            groups=deconv.groups,
            bias=True,
        )
        .requires_grad_(False)  # 禁用梯度追踪，不需要反向传播训练
        .to(deconv.weight.device)  # 将融合后的反卷积层移到与输入反卷积层相同的设备上
    )

    # 准备反卷积层的权重
    w_deconv = deconv.weight.clone().view(deconv.out_channels, -1)
    # 计算融合后的权重
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fuseddconv.weight.copy_(torch.mm(w_bn, w_deconv).view(fuseddconv.weight.shape))

    # 准备空间偏置项
    b_conv = torch.zeros(deconv.weight.shape[1], device=deconv.weight.device) if deconv.bias is None else deconv.bias
    # 计算融合后的偏置项
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fuseddconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fuseddconv


# 输出模型的信息，包括参数数量、梯度数量和层的数量
def model_info(model, detailed=False, verbose=True, imgsz=640):
    """
    Model information.

    imgsz may be int or list, i.e. imgsz=640 or imgsz=[640, 320].
    """
    # 如果不需要详细信息，则直接返回
    if not verbose:
        return
    # 获取模型的参数数量
    n_p = get_num_params(model)  # number of parameters
    # 获取模型的梯度数量
    n_g = get_num_gradients(model)  # number of gradients
    # 获取模型的层数量
    n_l = len(list(model.modules()))  # number of layers
    # 如果 detailed 参数为 True，则输出详细的模型参数信息
    if detailed:
        # 使用 LOGGER 记录模型参数的详细信息表头，包括层编号、名称、梯度是否计算、参数数量、形状、平均值、标准差和数据类型
        LOGGER.info(
            f"{'layer':>5} {'name':>40} {'gradient':>9} {'parameters':>12} {'shape':>20} {'mu':>10} {'sigma':>10}"
        )
        # 遍历模型的所有命名参数，并给每个参数分配一个序号 i
        for i, (name, p) in enumerate(model.named_parameters()):
            # 去除参数名中的 "module_list." 字符串
            name = name.replace("module_list.", "")
            # 使用 LOGGER 记录每个参数的详细信息，包括序号、名称、是否需要梯度、参数数量、形状、平均值、标准差和数据类型
            LOGGER.info(
                "%5g %40s %9s %12g %20s %10.3g %10.3g %10s"
                % (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std(), p.dtype)
            )

    # 计算模型的浮点运算量（FLOPs）
    flops = get_flops(model, imgsz)
    # 检查模型是否支持融合计算，如果支持，则添加 " (fused)" 到输出中
    fused = " (fused)" if getattr(model, "is_fused", lambda: False)() else ""
    # 如果计算得到的 FLOPs 不为空，则添加到输出中
    fs = f", {flops:.1f} GFLOPs" if flops else ""
    # 获取模型的 YAML 文件路径或者直接从模型属性中获取 YAML 文件路径，并去除路径中的 "yolo" 替换为 "YOLO"，或默认为 "Model"
    yaml_file = getattr(model, "yaml_file", "") or getattr(model, "yaml", {}).get("yaml_file", "")
    model_name = Path(yaml_file).stem.replace("yolo", "YOLO") or "Model"
    # 使用 LOGGER 记录模型的总结信息，包括模型名称、层数量、参数数量、梯度数量和计算量信息
    LOGGER.info(f"{model_name} summary{fused}: {n_l:,} layers, {n_p:,} parameters, {n_g:,} gradients{fs}")
    # 返回模型的层数量、参数数量、梯度数量和计算量
    return n_l, n_p, n_g, flops
# 返回 YOLO 模型中的总参数数量
def get_num_params(model):
    return sum(x.numel() for x in model.parameters())


# 返回 YOLO 模型中具有梯度的参数总数
def get_num_gradients(model):
    return sum(x.numel() for x in model.parameters() if x.requires_grad)


# 为日志记录器返回包含有用模型信息的字典
def model_info_for_loggers(trainer):
    if trainer.args.profile:  # 如果需要进行 ONNX 和 TensorRT 的性能分析
        from ultralytics.utils.benchmarks import ProfileModels

        # 使用 ProfileModels 进行模型性能分析，获取结果
        results = ProfileModels([trainer.last], device=trainer.device).profile()[0]
        results.pop("model/name")  # 移除结果中的模型名称
    else:  # 否则仅返回最近验证的 PyTorch 时间信息
        results = {
            "model/parameters": get_num_params(trainer.model),  # 计算模型参数数量
            "model/GFLOPs": round(get_flops(trainer.model), 3),  # 计算模型的 GFLOPs
        }
    results["model/speed_PyTorch(ms)"] = round(trainer.validator.speed["inference"], 3)  # 记录 PyTorch 推理速度
    return results


# 返回 YOLO 模型的 FLOPs（浮点运算数）
def get_flops(model, imgsz=640):
    if not thop:
        return 0.0  # 如果 thop 包未安装，返回 0.0 GFLOPs

    try:
        model = de_parallel(model)  # 取消模型的并行化
        p = next(model.parameters())
        if not isinstance(imgsz, list):
            imgsz = [imgsz, imgsz]  # 如果 imgsz 是 int 或 float，扩展为列表

        try:
            stride = max(int(model.stride.max()), 32) if hasattr(model, "stride") else 32  # 获取输入张量的步幅大小
            im = torch.empty((1, p.shape[1], stride, stride), device=p.device)  # 创建输入图像张量
            flops = thop.profile(deepcopy(model), inputs=[im], verbose=False)[0] / 1e9 * 2  # 使用 thop 计算 GFLOPs
            return flops * imgsz[0] / stride * imgsz[1] / stride  # 计算基于图像尺寸的 GFLOPs
        except Exception:
            im = torch.empty((1, p.shape[1], *imgsz), device=p.device)  # 创建输入图像张量
            return thop.profile(deepcopy(model), inputs=[im], verbose=False)[0] / 1e9 * 2  # 计算基于图像尺寸的 GFLOPs
    except Exception:
        return 0.0  # 发生异常时返回 0.0 GFLOPs


# 使用 Torch 分析器计算模型的 FLOPs（thop 包的替代方案，但速度通常较慢 2-10 倍）
def get_flops_with_torch_profiler(model, imgsz=640):
    if not TORCH_2_0:  # 如果 Torch 版本低于 2.0，返回 0.0
        return 0.0
    model = de_parallel(model)  # 取消模型的并行化
    p = next(model.parameters())
    if not isinstance(imgsz, list):
        imgsz = [imgsz, imgsz]  # 如果 imgsz 是 int 或 float，扩展为列表
    try:
        # 使用模型的步幅大小来确定输入张量的步幅
        stride = (max(int(model.stride.max()), 32) if hasattr(model, "stride") else 32) * 2  # 最大步幅
        # 创建一个空的张量作为输入图像，格式为BCHW
        im = torch.empty((1, p.shape[1], stride, stride), device=p.device)
        with torch.profiler.profile(with_flops=True) as prof:
            # 对模型进行推理，记录性能指标
            model(im)
        # 计算模型的浮点运算量（FLOPs）
        flops = sum(x.flops for x in prof.key_averages()) / 1e9
        # 根据输入图像大小调整计算的FLOPs，例如 640x640 GFLOPs
        flops = flops * imgsz[0] / stride * imgsz[1] / stride
    except Exception:
        # 对于RTDETR模型，使用实际图像大小作为输入张量的大小
        im = torch.empty((1, p.shape[1], *imgsz), device=p.device)  # 输入图像为BCHW格式
        with torch.profiler.profile(with_flops=True) as prof:
            # 对模型进行推理，记录性能指标
            model(im)
        # 计算模型的浮点运算量（FLOPs）
        flops = sum(x.flops for x in prof.key_averages()) / 1e9
    # 返回计算得到的FLOPs
    return flops
def initialize_weights(model):
    """Initialize model weights to random values."""
    # Iterate over all modules in the model
    for m in model.modules():
        t = type(m)
        # Check if the module is a 2D convolutional layer
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        # Check if the module is a 2D batch normalization layer
        elif t is nn.BatchNorm2d:
            # Set epsilon (eps) and momentum parameters
            m.eps = 1e-3
            m.momentum = 0.03
        # Check if the module is one of the specified activation functions
        elif t in {nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU}:
            # Enable inplace operation for the activation function
            m.inplace = True


def scale_img(img, ratio=1.0, same_shape=False, gs=32):
    """Scales and pads an image tensor of shape img(bs,3,y,x) based on given ratio and grid size gs, optionally
    retaining the original shape.
    """
    # If ratio is 1.0, return the original image tensor
    if ratio == 1.0:
        return img
    # Retrieve height and width from the image tensor shape
    h, w = img.shape[2:]
    # Compute the new scaled size based on the given ratio
    s = (int(h * ratio), int(w * ratio))  # new size
    # Resize the image tensor using bilinear interpolation
    img = F.interpolate(img, size=s, mode="bilinear", align_corners=False)  # resize
    # If not retaining the original shape, pad or crop the image tensor
    if not same_shape:
        # Calculate the padded height and width based on the ratio and grid size
        h, w = (math.ceil(x * ratio / gs) * gs for x in (h, w))
    # Pad the image tensor to match the calculated dimensions
    return F.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)  # value = imagenet mean


def copy_attr(a, b, include=(), exclude=()):
    """Copies attributes from object 'b' to object 'a', with options to include/exclude certain attributes."""
    # Iterate through attributes in object 'b'
    for k, v in b.__dict__.items():
        # Skip attributes based on conditions: not in include list, starts with '_', or in exclude list
        if (len(include) and k not in include) or k.startswith("_") or k in exclude:
            continue
        else:
            # Set attribute 'k' in object 'a' to the value 'v' from object 'b'
            setattr(a, k, v)


def get_latest_opset():
    """Return the second-most recent ONNX opset version supported by this version of PyTorch, adjusted for maturity."""
    # Check if using PyTorch version 1.13 or newer
    if TORCH_1_13:
        # Dynamically compute the second-most recent ONNX opset version supported
        return max(int(k[14:]) for k in vars(torch.onnx) if "symbolic_opset" in k) - 1
    # For PyTorch versions <= 1.12, return predefined opset versions
    version = torch.onnx.producer_version.rsplit(".", 1)[0]  # i.e. '2.3'
    return {"1.12": 15, "1.11": 14, "1.10": 13, "1.9": 12, "1.8": 12}.get(version, 12)


def intersect_dicts(da, db, exclude=()):
    """Returns a dictionary of intersecting keys with matching shapes, excluding 'exclude' keys, using da values."""
    # Create a dictionary comprehension to filter keys based on conditions
    return {k: v for k, v in da.items() if k in db and all(x not in k for x in exclude) and v.shape == db[k].shape}


def is_parallel(model):
    """Returns True if model is of type DP or DDP."""
    # Check if the model is an instance of DataParallel or DistributedDataParallel
    return isinstance(model, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel))


def de_parallel(model):
    """De-parallelize a model: returns single-GPU model if model is of type DP or DDP."""
    # Return the underlying module of a DataParallel or DistributedDataParallel model
    return model.module if is_parallel(model) else model


def one_cycle(y1=0.0, y2=1.0, steps=100):
    """Returns a lambda function for sinusoidal ramp from y1 to y2 https://arxiv.org/pdf/1812.01187.pdf."""
    # Generate a lambda function that implements a sinusoidal ramp
    return lambda x: max((1 - math.cos(x * math.pi / steps)) / 2, 0) * (y2 - y1) + y1


def init_seeds(seed=0, deterministic=False):
    """Initialize random number generator seeds."""
    # This function initializes seeds for random number generators
    # It is intended to be implemented further, but the current snippet does not contain the complete implementation.
    pass
    # 初始化随机数生成器（RNG）种子，以确保实验的可复现性 https://pytorch.org/docs/stable/notes/randomness.html.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 用于多GPU情况下的种子设置，确保异常安全性
    # torch.backends.cudnn.benchmark = True  # AutoBatch问题 https://github.com/ultralytics/yolov5/issues/9287
    # 如果需要确定性行为，则执行以下操作
    if deterministic:
        if TORCH_2_0:
            # 使用确定性算法，并在不可确定时发出警告
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.deterministic = True
            # 设置CUBLAS工作空间大小的配置
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            os.environ["PYTHONHASHSEED"] = str(seed)
        else:
            # 提示升级到torch>=2.0.0以实现确定性训练
            LOGGER.warning("WARNING ⚠️ Upgrade to torch>=2.0.0 for deterministic training.")
    else:
        # 关闭确定性算法，允许非确定性行为
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
class ModelEMA:
    """
    Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models. Keeps a moving
    average of everything in the model state_dict (parameters and buffers)

    For EMA details see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage

    To disable EMA set the `enabled` attribute to `False`.
    """

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        """Initialize EMA for 'model' with given arguments."""
        self.ema = deepcopy(de_parallel(model)).eval()  # FP32 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.enabled = True

    def update(self, model):
        """Update EMA parameters."""
        if self.enabled:
            self.updates += 1
            d = self.decay(self.updates)

            msd = de_parallel(model).state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:  # true for FP16 and FP32
                    v *= d
                    v += (1 - d) * msd[k].detach()
                    # assert v.dtype == msd[k].dtype == torch.float32, f'{k}: EMA {v.dtype},  model {msd[k].dtype}'

    def update_attr(self, model, include=(), exclude=("process_group", "reducer")):
        """Updates attributes and saves stripped model with optimizer removed."""
        if self.enabled:
            copy_attr(self.ema, model, include, exclude)


def strip_optimizer(f: Union[str, Path] = "best.pt", s: str = "") -> None:
    """
    Strip optimizer from 'f' to finalize training, optionally save as 's'.

    Args:
        f (str): file path to model to strip the optimizer from. Default is 'best.pt'.
        s (str): file path to save the model with stripped optimizer to. If not provided, 'f' will be overwritten.

    Returns:
        None

    Example:
        ```py
        from pathlib import Path
        from ultralytics.utils.torch_utils import strip_optimizer

        for f in Path('path/to/model/checkpoints').rglob('*.pt'):
            strip_optimizer(f)
        ```py

    Note:
        Use `ultralytics.nn.torch_safe_load` for missing modules with `x = torch_safe_load(f)[0]`
    """
    try:
        x = torch.load(f, map_location=torch.device("cpu"))
        assert isinstance(x, dict), "checkpoint is not a Python dictionary"
        assert "model" in x, "'model' missing from checkpoint"
    except Exception as e:
        LOGGER.warning(f"WARNING ⚠️ Skipping {f}, not a valid Ultralytics model: {e}")
        return

    updates = {
        "date": datetime.now().isoformat(),
        "version": __version__,
        "license": "AGPL-3.0 License (https://ultralytics.com/license)",
        "docs": "https://docs.ultralytics.com",
    }

    # Update model
    # 如果字典 x 中有 "ema" 键，则将 "model" 键的值设为 "ema" 的值，替换模型为 EMA 模型
    if x.get("ema"):
        x["model"] = x["ema"]  # replace model with EMA
    
    # 如果 "model" 对象具有 "args" 属性，将其转换为字典类型，从 IterableSimpleNamespace 转换为 dict
    if hasattr(x["model"], "args"):
        x["model"].args = dict(x["model"].args)  # convert from IterableSimpleNamespace to dict
    
    # 如果 "model" 对象具有 "criterion" 属性，将其设置为 None，去除损失函数的标准
    if hasattr(x["model"], "criterion"):
        x["model"].criterion = None  # strip loss criterion
    
    # 将模型转换为半精度浮点数表示，即 FP16
    x["model"].half()  # to FP16
    
    # 将模型的所有参数设置为不需要梯度计算
    for p in x["model"].parameters():
        p.requires_grad = False

    # 更新字典中的其他键
    args = {**DEFAULT_CFG_DICT, **x.get("train_args", {})}  # 将 DEFAULT_CFG_DICT 和 x 中的 "train_args" 合并为一个字典
    for k in "optimizer", "best_fitness", "ema", "updates":  # 遍历指定的键
        x[k] = None  # 将字典 x 中指定键的值设为 None
    x["epoch"] = -1  # 将 epoch 键的值设为 -1
    # 创建一个新字典，其中仅包含 DEFAULT_CFG_KEYS 中存在的键值对，并将其赋给 "train_args"
    x["train_args"] = {k: v for k, v in args.items() if k in DEFAULT_CFG_KEYS}  # strip non-default keys
    # x['model'].args = x['train_args']  # 此行代码被注释掉了，不再使用

    # 将 updates 和 x 中的内容合并为一个字典，并保存到文件中，不使用 dill 序列化
    torch.save({**updates, **x}, s or f, use_dill=False)  # combine dicts (prefer to the right)
    
    # 获取文件的大小，并将其转换为兆字节（MB）
    mb = os.path.getsize(s or f) / 1e6  # file size
    
    # 记录日志，显示优化器已从文件中剥离，同时显示文件名和文件大小
    LOGGER.info(f"Optimizer stripped from {f},{f' saved as {s},' if s else ''} {mb:.1f}MB")
# 将给定优化器的状态字典转换为FP16格式，重点在于转换'torch.Tensor'类型的数据
def convert_optimizer_state_dict_to_fp16(state_dict):
    # 遍历优化器状态字典中的'state'键对应的所有状态
    for state in state_dict["state"].values():
        # 遍历每个状态的键值对
        for k, v in state.items():
            # 排除键为"step"且值为'torch.Tensor'类型且数据类型为torch.float32的情况
            if k != "step" and isinstance(v, torch.Tensor) and v.dtype is torch.float32:
                # 将符合条件的Tensor类型数据转换为半精度（FP16）
                state[k] = v.half()

    # 返回转换后的状态字典
    return state_dict


# Ultralytics速度、内存和FLOPs（浮点运算数）分析器
def profile(input, ops, n=10, device=None):
    # 结果存储列表
    results = []
    # 如果设备参数不是torch.device类型，则选择设备
    if not isinstance(device, torch.device):
        device = select_device(device)
    # 打印日志信息，包括各项参数
    LOGGER.info(
        f"{'Params':>12s}{'GFLOPs':>12s}{'GPU_mem (GB)':>14s}{'forward (ms)':>14s}{'backward (ms)':>14s}"
        f"{'input':>24s}{'output':>24s}"
    )
    for x in input if isinstance(input, list) else [input]:
        # 如果输入是列表，则遍历列表中的每个元素；否则将输入放入列表中并遍历
        x = x.to(device)
        # 将当前元素移动到指定的设备上（如GPU）
        x.requires_grad = True
        # 设置当前元素的梯度跟踪为True

        for m in ops if isinstance(ops, list) else [ops]:
            # 如果操作是列表，则遍历列表中的每个操作；否则将操作放入列表中并遍历
            m = m.to(device) if hasattr(m, "to") else m
            # 如果操作具有"to"方法，则将其移动到指定的设备上；否则保持不变
            m = m.half() if hasattr(m, "half") and isinstance(x, torch.Tensor) and x.dtype is torch.float16 else m
            # 如果操作具有"half"方法，并且输入是torch.Tensor类型且数据类型为torch.float16，则将操作转换为半精度（float16）；否则保持不变
            tf, tb, t = 0, 0, [0, 0, 0]
            # 初始化时间记录变量：前向传播时间，反向传播时间，时间记录列表

            try:
                flops = thop.profile(m, inputs=[x], verbose=False)[0] / 1e9 * 2 if thop else 0
                # 使用thop库对操作进行浮点操作计算（FLOPs），并将结果转换为GFLOPs（十亿次浮点操作每秒）
            except Exception:
                flops = 0
                # 如果计算FLOPs出现异常，则将FLOPs设置为0

            try:
                for _ in range(n):
                    t[0] = time_sync()
                    # 记录前向传播开始时间
                    y = m(x)
                    # 执行操作的前向传播
                    t[1] = time_sync()
                    # 记录前向传播结束时间
                    try:
                        (sum(yi.sum() for yi in y) if isinstance(y, list) else y).sum().backward()
                        # 计算输出y的总和，并对总和进行反向传播
                        t[2] = time_sync()
                        # 记录反向传播结束时间
                    except Exception:  # no backward method
                        t[2] = float("nan")
                        # 如果没有反向传播方法，则将反向传播时间设置为NaN
                    tf += (t[1] - t[0]) * 1000 / n
                    # 计算每个操作的平均前向传播时间（毫秒）
                    tb += (t[2] - t[1]) * 1000 / n
                    # 计算每个操作的平均反向传播时间（毫秒）
                mem = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
                # 如果CUDA可用，则计算当前GPU上的内存使用量（单位：GB）
                s_in, s_out = (tuple(x.shape) if isinstance(x, torch.Tensor) else "list" for x in (x, y))
                # 获取输入x和输出y的形状信息
                p = sum(x.numel() for x in m.parameters()) if isinstance(m, nn.Module) else 0
                # 计算操作m中的参数数量
                LOGGER.info(f"{p:12}{flops:12.4g}{mem:>14.3f}{tf:14.4g}{tb:14.4g}{str(s_in):>24s}{str(s_out):>24s}")
                # 将结果记录到日志中，包括参数数量、FLOPs、内存占用、时间等信息
                results.append([p, flops, mem, tf, tb, s_in, s_out])
                # 将结果添加到结果列表中
            except Exception as e:
                LOGGER.info(e)
                # 记录异常信息到日志中
                results.append(None)
                # 将空结果添加到结果列表中
            gc.collect()
            # 尝试释放未使用的内存
            torch.cuda.empty_cache()
            # 清空CUDA缓存

    return results
    # 返回所有操作的结果列表
class EarlyStopping:
    """Early stopping class that stops training when a specified number of epochs have passed without improvement."""

    def __init__(self, patience=50):
        """
        Initialize early stopping object.

        Args:
            patience (int, optional): Number of epochs to wait after fitness stops improving before stopping.
        """
        self.best_fitness = 0.0  # 初始化最佳适应度为0.0，即最佳平均精度（mAP）
        self.best_epoch = 0  # 初始化最佳轮次为0
        self.patience = patience or float("inf")  # 设置等待适应度停止提高的轮次数，若未提供则设为无穷大
        self.possible_stop = False  # 是否可能在下一个轮次停止训练的标志

    def __call__(self, epoch, fitness):
        """
        Check whether to stop training.

        Args:
            epoch (int): Current epoch of training
            fitness (float): Fitness value of current epoch

        Returns:
            (bool): True if training should stop, False otherwise
        """
        if fitness is None:  # 检查适应度是否为None（当val=False时会发生）
            return False

        if fitness >= self.best_fitness:  # 如果当前适应度大于或等于最佳适应度
            self.best_epoch = epoch  # 更新最佳轮次为当前轮次
            self.best_fitness = fitness  # 更新最佳适应度为当前适应度
        delta = epoch - self.best_epoch  # 计算未改善的轮次数
        self.possible_stop = delta >= (self.patience - 1)  # 更新可能在下一个轮次停止训练的标志
        stop = delta >= self.patience  # 若未改善的轮次数超过设定的等待轮次数，则停止训练
        if stop:
            prefix = colorstr("EarlyStopping: ")  # 设置输出前缀
            LOGGER.info(
                f"{prefix}Training stopped early as no improvement observed in last {self.patience} epochs. "
                f"Best results observed at epoch {self.best_epoch}, best model saved as best.pt.\n"
                f"To update EarlyStopping(patience={self.patience}) pass a new patience value, "
                f"i.e. `patience=300` or use `patience=0` to disable EarlyStopping."
            )  # 输出停止训练信息
        return stop  # 返回是否停止训练的标志
```