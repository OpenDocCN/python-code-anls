# `stable-diffusion-webui\modules\mac_specific.py`

```
# 导入 logging 模块
import logging

# 导入 torch 模块及其子模块 Tensor
import torch
from torch import Tensor

# 导入 platform 模块
import platform

# 导入 sd_hijack_utils 模块中的 CondFunc 类
from modules.sd_hijack_utils import CondFunc

# 导入 version 函数
from packaging import version

# 导入 shared 模块
from modules import shared

# 获取当前模块的日志记录器
log = logging.getLogger(__name__)

# 检查是否支持 MPS（Memory Persistence Service），在 torch 版本 1.13 之前，只有在夜间构建的 PyTorch 和 macOS 12.3+ 上才可用，
# 使用 getattr 检查并尝试兼容性。
# 在 torch 版本 1.13 中，引入了 backends.mps.is_available() 和 backends.mps.is_built() 来检查 MPS 的可用性，
# 自 torch 2.0.1+ 夜间构建版本开始，getattr(torch, 'has_mps', False) 已被弃用，参见 https://github.com/pytorch/pytorch/pull/103279
def check_for_mps() -> bool:
    if version.parse(torch.__version__) <= version.parse("2.0.1"):
        if not getattr(torch, 'has_mps', False):
            return False
        try:
            torch.zeros(1).to(torch.device("mps"))
            return True
        except Exception:
            return False
    else:
        return torch.backends.mps.is_available() and torch.backends.mps.is_built()

# 检查是否支持 MPS，并将结果存储在 has_mps 变量中
has_mps = check_for_mps()

# 执行 MPS 垃圾回收
def torch_mps_gc() -> None:
    try:
        # 如果 shared.state.current_latent 不为 None，则跳过 MPS 垃圾回收
        if shared.state.current_latent is not None:
            log.debug("`current_latent` is set, skipping MPS garbage collection")
            return
        # 导入 empty_cache 函数并执行 MPS 垃圾回收
        from torch.mps import empty_cache
        empty_cache()
    except Exception:
        log.warning("MPS garbage collection failed", exc_info=True)

# 用于解决 https://github.com/pytorch/pytorch/issues/89784 的 MPS 工作区
def cumsum_fix(input, cumsum_func, *args, **kwargs):
    # 如果输入张量的设备类型为 'mps'
    if input.device.type == 'mps':
        # 获取输出数据类型
        output_dtype = kwargs.get('dtype', input.dtype)
        # 如果输出数据类型为 torch.int64
        if output_dtype == torch.int64:
            # 将输入张量转移到 CPU 上执行累加操作，然后转回原设备
            return cumsum_func(input.cpu(), *args, **kwargs).to(input.device)
        # 如果输出数据类型为 torch.bool 或者 cumsum_needs_int_fix 为真且输出数据类型为 torch.int8 或 torch.int16
        elif output_dtype == torch.bool or cumsum_needs_int_fix and (output_dtype == torch.int8 or output_dtype == torch.int16):
            # 将输入张量转为 torch.int32 类型执行累加操作，然后转为 torch.int64 类型
            return cumsum_func(input.to(torch.int32), *args, **kwargs).to(torch.int64)
    # 对输入张量执行累加操作
    return cumsum_func(input, *args, **kwargs)
# 为了解决 https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/14046 中的问题，对 MPS 进行了一些处理
def interpolate_with_fp32_fallback(orig_func, *args, **kwargs) -> Tensor:
    try:
        # 尝试调用原始函数
        return orig_func(*args, **kwargs)
    except RuntimeError as e:
        # 捕获运行时错误
        if "not implemented for" in str(e) and "Half" in str(e):
            # 如果错误信息中包含 "not implemented for" 和 "Half"
            input_tensor = args[0]
            # 将输入张量转换为 torch.float32 类型，再次调用原始函数，最后将结果转换回原始输入张量的数据类型
            return orig_func(input_tensor.to(torch.float32), *args[1:], **kwargs).to(input_tensor.dtype)
        else:
            # 如果出现意外的运行时错误，则打印错误信息
            print(f"An unexpected RuntimeError occurred: {str(e)}")

# 如果支持 MPS
if has_mps:
    # 如果运行在 macOS 版本为 13.2.
    if platform.mac_ver()[0].startswith("13.2."):
        # 对 torch.nn.functional.linear 函数进行条件处理，根据条件返回不同的函数
        # 这是为了解决 https://github.com/pytorch/pytorch/issues/95188 中的问题，感谢 danieldk (https://github.com/explosion/curated-transformers/pull/124)
        CondFunc('torch.nn.functional.linear', lambda _, input, weight, bias: (torch.matmul(input, weight.t()) + bias) if bias is not None else torch.matmul(input, weight.t()), lambda _, input, weight, bias: input.numel() > 10485760)
    # 检查 PyTorch 版本是否小于 1.13，如果是则需要进行以下修复，因为 PyTorch 1.13 存在性能问题和训练回归问题
    # MPS 的问题解决方案，用于 https://github.com/pytorch/pytorch/issues/79383
    CondFunc('torch.Tensor.to', lambda orig_func, self, *args, **kwargs: orig_func(self.contiguous(), *args, **kwargs),
                                                      lambda _, self, *args, **kwargs: self.device.type != 'mps' and (args and isinstance(args[0], torch.device) and args[0].type == 'mps' or isinstance(kwargs.get('device'), torch.device) and kwargs['device'].type == 'mps'))
    # MPS 的问题解决方案，用于 https://github.com/pytorch/pytorch/issues/80800
    CondFunc('torch.nn.functional.layer_norm', lambda orig_func, *args, **kwargs: orig_func(*([args[0].contiguous()] + list(args[1:])), **kwargs),
                                                                                    lambda _, *args, **kwargs: args and isinstance(args[0], torch.Tensor) and args[0].device.type == 'mps')
    # MPS 的问题解决方案，用于 https://github.com/pytorch/pytorch/issues/90532
    CondFunc('torch.Tensor.numpy', lambda orig_func, self, *args, **kwargs: orig_func(self.detach(), *args, **kwargs), lambda _, self, *args, **kwargs: self.requires_grad)
    # 检查 PyTorch 版本是否大于 1.13.1
    elif version.parse(torch.__version__) > version.parse("1.13.1"):
        # 检查是否需要修复累加函数的整数问题
        cumsum_needs_int_fix = not torch.Tensor([1,2]).to(torch.device("mps")).equal(torch.ShortTensor([1,1]).to(torch.device("mps")).cumsum(0))
        # 定义修复累加函数的 lambda 函数
        cumsum_fix_func = lambda orig_func, input, *args, **kwargs: cumsum_fix(input, orig_func, *args, **kwargs)
        # 条件函数，用于修复 torch.cumsum 函数
        CondFunc('torch.cumsum', cumsum_fix_func, None)
        # 条件函数，用于修复 torch.Tensor.cumsum 函数
        CondFunc('torch.Tensor.cumsum', cumsum_fix_func, None)
        # 条件函数，用于修复 torch.narrow 函数
        CondFunc('torch.narrow', lambda orig_func, *args, **kwargs: orig_func(*args, **kwargs).clone(), None)

        # MPS 的问题解决方案，用于修复 torch.nn.functional.layer_norm 函数
        CondFunc('torch.nn.functional.layer_norm', lambda orig_func, x, normalized_shape, weight, bias, eps, **kwargs: orig_func(x.float(), normalized_shape, weight.float() if weight is not None else None, bias.float() if bias is not None else bias, eps).to(x.dtype), lambda _, input, *args, **kwargs: len(args) == 4 and input.device.type == 'mps')

        # MPS 的问题解决方案，用于修复 torch.nn.functional.interpolate 函数
        CondFunc('torch.nn.functional.interpolate', interpolate_with_fp32_fallback, None)

        # MPS 的问题解决方案，用于修复 torch.argmax 和 torch.Tensor.argmax 函数
        if platform.processor() == 'i386':
            for funcName in ['torch.argmax', 'torch.Tensor.argmax']:
                CondFunc(funcName, lambda _, input, *args, **kwargs: torch.max(input.float() if input.dtype == torch.int64 else input, *args, **kwargs)[1], lambda _, input, *args, **kwargs: input.device.type == 'mps')
```