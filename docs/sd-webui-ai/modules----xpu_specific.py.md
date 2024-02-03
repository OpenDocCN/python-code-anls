# `stable-diffusion-webui\modules\xpu_specific.py`

```py
# 从 modules 模块中导入 shared 对象
from modules import shared
# 从 modules.sd_hijack_utils 模块中导入 CondFunc 类
from modules.sd_hijack_utils import CondFunc

# 初始化变量 has_ipex 为 False
has_ipex = False
# 尝试导入 torch 和 intel_extension_for_pytorch 模块，如果导入成功则将 has_ipex 置为 True
try:
    import torch
    import intel_extension_for_pytorch as ipex # noqa: F401
    has_ipex = True
except Exception:
    pass

# 检查是否存在 XPU 设备
def check_for_xpu():
    return has_ipex and hasattr(torch, 'xpu') and torch.xpu.is_available()

# 获取 XPU 设备字符串
def get_xpu_device_string():
    # 如果命令行参数中指定了设备 ID，则返回格式化后的 XPU 设备字符串
    if shared.cmd_opts.device_id is not None:
        return f"xpu:{shared.cmd_opts.device_id}"
    # 否则返回默认的 XPU 设备字符串
    return "xpu"

# 清理 XPU 设备的缓存
def torch_xpu_gc():
    # 使用指定的 XPU 设备字符串创建 XPU 设备上下文，清理 XPU 设备缓存
    with torch.xpu.device(get_xpu_device_string()):
        torch.xpu.empty_cache()

# 检查是否存在 XPU 设备
has_xpu = check_for_xpu()

# 如果存在 XPU 设备，则执行以下操作
if has_xpu:
    # 为 torch.Generator API 添加条件函数，解决 XPU 设备不支持的问题
    CondFunc('torch.Generator',
        lambda orig_func, device=None: torch.xpu.Generator(device),
        lambda orig_func, device=None: device is not None and device.type == "xpu")

    # 为一些无法处理不同输入数据类型的操作添加条件函数
    CondFunc('torch.nn.functional.layer_norm',
        lambda orig_func, input, normalized_shape=None, weight=None, *args, **kwargs:
        orig_func(input.to(weight.data.dtype), normalized_shape, weight, *args, **kwargs),
        lambda orig_func, input, normalized_shape=None, weight=None, *args, **kwargs:
        weight is not None and input.dtype != weight.data.dtype)
    CondFunc('torch.nn.modules.GroupNorm.forward',
        lambda orig_func, self, input: orig_func(self, input.to(self.weight.data.dtype)),
        lambda orig_func, self, input: input.dtype != self.weight.data.dtype)
    CondFunc('torch.nn.modules.linear.Linear.forward',
        lambda orig_func, self, input: orig_func(self, input.to(self.weight.data.dtype)),
        lambda orig_func, self, input: input.dtype != self.weight.data.dtype)
    # 为指定的函数添加条件执行功能，当输入数据类型与权重数据类型不同时，将输入数据转换为权重数据类型后再执行原始函数
    CondFunc('torch.nn.modules.conv.Conv2d.forward',
        lambda orig_func, self, input: orig_func(self, input.to(self.weight.data.dtype)),
        lambda orig_func, self, input: input.dtype != self.weight.data.dtype)
    # 为指定的函数添加条件执行功能，当输入数据类型与矩阵数据类型不同时，将输入数据转换为矩阵数据类型后再执行原始函数
    CondFunc('torch.bmm',
        lambda orig_func, input, mat2, out=None: orig_func(input.to(mat2.dtype), mat2, out=out),
        lambda orig_func, input, mat2, out=None: input.dtype != mat2.dtype)
    # 为指定的函数添加条件执行功能，当输入张量中存在数据类型与第一个张量数据类型不同的张量时，将所有张量转换为第一个张量数据类型后再执行原始函数
    CondFunc('torch.cat',
        lambda orig_func, tensors, dim=0, out=None: orig_func([t.to(tensors[0].dtype) for t in tensors], dim=dim, out=out),
        lambda orig_func, tensors, dim=0, out=None: not all(t.dtype == tensors[0].dtype for t in tensors))
    # 为指定的函数添加条件执行功能，当查询、键、值的数据类型与查询数据类型不同时，将键和值转换为查询数据类型后再执行原始函数
    CondFunc('torch.nn.functional.scaled_dot_product_attention',
        lambda orig_func, query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False: orig_func(query, key.to(query.dtype), value.to(query.dtype), attn_mask, dropout_p, is_causal),
        lambda orig_func, query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False: query.dtype != key.dtype or query.dtype != value.dtype)
```