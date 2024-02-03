# `stable-diffusion-webui\modules\sd_hijack_unet.py`

```
# 导入 torch 库
import torch
# 从 packaging 库中导入 version 模块
from packaging import version

# 从 modules 中导入 devices 模块
from modules import devices
# 从 modules.sd_hijack_utils 中导入 CondFunc 类
from modules.sd_hijack_utils import CondFunc

# 定义 TorchHijackForUnet 类
class TorchHijackForUnet:
    """
    This is torch, but with cat that resizes tensors to appropriate dimensions if they do not match;
    this makes it possible to create pictures with dimensions that are multiples of 8 rather than 64
    """

    # 定义 __getattr__ 方法
    def __getattr__(self, item):
        # 如果 item 为 'cat'，返回 self.cat 方法
        if item == 'cat':
            return self.cat

        # 如果 torch 中存在 item 属性，返回对应属性
        if hasattr(torch, item):
            return getattr(torch, item)

        # 抛出 AttributeError 异常
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{item}'")

    # 定义 cat 方法
    def cat(self, tensors, *args, **kwargs):
        # 如果 tensors 中有两个张量
        if len(tensors) == 2:
            a, b = tensors
            # 如果 a 的最后两个维度不匹配 b 的最后两个维度，使用最近邻插值将 a 调整为与 b 相同的维度
            if a.shape[-2:] != b.shape[-2:]:
                a = torch.nn.functional.interpolate(a, b.shape[-2:], mode="nearest")

            tensors = (a, b)

        # 返回拼接后的张量
        return torch.cat(tensors, *args, **kwargs)


# 创建 TorchHijackForUnet 类的实例 th
th = TorchHijackForUnet()

# 以下是用于启用将 float16 UNet 上采样为 float32 采样的 monkey patches
# 定义 apply_model 函数
def apply_model(orig_func, self, x_noisy, t, cond, **kwargs):
    # 如果 cond 是字典类型
    if isinstance(cond, dict):
        # 遍历 cond 中的键值对
        for y in cond.keys():
            # 如果 cond[y] 是列表类型
            if isinstance(cond[y], list):
                # 将 cond[y] 中的张量转换为 devices.dtype_unet 类型
                cond[y] = [x.to(devices.dtype_unet) if isinstance(x, torch.Tensor) else x for x in cond[y]]
            else:
                # 将 cond[y] 转换为 devices.dtype_unet 类型
                cond[y] = cond[y].to(devices.dtype_unet) if isinstance(cond[y], torch.Tensor) else cond[y]

    # 使用 devices.autocast 上下文管理器
    with devices.autocast():
        # 调用原始函数 orig_func，并将输入数据转换为 devices.dtype_unet 类型后返回 float 类型结果
        return orig_func(self, x_noisy.to(devices.dtype_unet), t.to(devices.dtype_unet), cond, **kwargs).float()

# 定义 GELUHijack 类，继承自 torch.nn.GELU 和 torch.nn.Module
class GELUHijack(torch.nn.GELU, torch.nn.Module):
    def __init__(self, *args, **kwargs):
        # 调用父类的 __init__ 方法
        torch.nn.GELU.__init__(self, *args, **kwargs)
    def forward(self, x):
        # 如果 devices.unet_needs_upcast 为真
        if devices.unet_needs_upcast:
            # 将输入数据和模型转换为 float 类型，并将结果转换为 devices.dtype_unet 类型后返回
            return torch.nn.GELU.forward(self.float(), x.float()).to(devices.dtype_unet)
        else:
            # 否则直接调用父类的 forward 方法
            return torch.nn.GELU.forward(self, x)

# 定义 ddpm_edit_hijack 变量为 None
ddpm_edit_hijack = None
# 定义 hijack_ddpm_edit 函数
def hijack_ddpm_edit():
    # 声明全局变量 ddpm_edit_hijack
    global ddpm_edit_hijack
    # 如果 ddpm_edit_hijack 为假，则执行以下代码
    if not ddpm_edit_hijack:
        # 调用 CondFunc 函数，传入函数名 'modules.models.diffusion.ddpm_edit.LatentDiffusion.decode_first_stage'，first_stage_sub 和 first_stage_cond 作为参数
        CondFunc('modules.models.diffusion.ddpm_edit.LatentDiffusion.decode_first_stage', first_stage_sub, first_stage_cond)
        # 调用 CondFunc 函数，传入函数名 'modules.models.diffusion.ddpm_edit.LatentDiffusion.encode_first_stage'，first_stage_sub 和 first_stage_cond 作为参数
        CondFunc('modules.models.diffusion.ddpm_edit.LatentDiffusion.encode_first_stage', first_stage_sub, first_stage_cond)
        # 将 CondFunc 函数返回值赋给 ddpm_edit_hijack，传入函数名 'modules.models.diffusion.ddpm_edit.LatentDiffusion.apply_model'，apply_model 和 unet_needs_upcast 作为参数
        ddpm_edit_hijack = CondFunc('modules.models.diffusion.ddpm_edit.LatentDiffusion.apply_model', apply_model, unet_needs_upcast)
# 定义一个 lambda 函数，用于确定是否需要将数据类型升级为 unet_needs_upcast
unet_needs_upcast = lambda *args, **kwargs: devices.unet_needs_upcast
# 将 apply_model 函数应用到 'ldm.models.diffusion.ddpm.LatentDiffusion.apply_model' 上，并根据 unet_needs_upcast 确定是否需要数据类型升级
CondFunc('ldm.models.diffusion.ddpm.LatentDiffusion.apply_model', apply_model, unet_needs_upcast)
# 如果 torch 版本小于等于 "1.13.2" 或者有可用的 CUDA，则执行以下条件函数
if version.parse(torch.__version__) <= version.parse("1.13.2") or torch.cuda.is_available():
    # 将 orig_func 应用到 'ldm.modules.diffusionmodules.util.GroupNorm32.forward' 上，并根据 unet_needs_upcast 确定是否需要数据类型升级
    CondFunc('ldm.modules.diffusionmodules.util.GroupNorm32.forward', lambda orig_func, self, *args, **kwargs: orig_func(self.float(), *args, **kwargs), unet_needs_upcast)
    # 将 orig_func 应用到 'ldm.modules.attention.GEGLU.forward' 上，并根据 unet_needs_upcast 确定是否需要数据类型升级
    CondFunc('ldm.modules.attention.GEGLU.forward', lambda orig_func, self, x: orig_func(self.float(), x.float()).to(devices.dtype_unet), unet_needs_upcast)
    # 将 'act_layer' 更新为 GELUHijack 并返回 False，或者保持原样返回 orig_func(*args, **kwargs)，根据条件判断是否需要执行
    CondFunc('open_clip.transformer.ResidualAttentionBlock.__init__', lambda orig_func, *args, **kwargs: kwargs.update({'act_layer': GELUHijack}) and False or orig_func(*args, **kwargs), lambda _, *args, **kwargs: kwargs.get('act_layer') is None or kwargs['act_layer'] == torch.nn.GELU)
# 定义一个 lambda 函数，用于确定是否需要在第一阶段进行数据类型升级
first_stage_cond = lambda _, self, *args, **kwargs: devices.unet_needs_upcast and self.model.diffusion_model.dtype == torch.float16
# 定义一个 lambda 函数，用于在第一阶段对输入数据进行数据类型升级
first_stage_sub = lambda orig_func, self, x, **kwargs: orig_func(self, x.to(devices.dtype_vae), **kwargs)
# 将 first_stage_sub 函数应用到 'ldm.models.diffusion.ddpm.LatentDiffusion.decode_first_stage' 上，并根据 first_stage_cond 确定是否需要数据类型升级
CondFunc('ldm.models.diffusion.ddpm.LatentDiffusion.decode_first_stage', first_stage_sub, first_stage_cond)
# 将 first_stage_sub 函数应用到 'ldm.models.diffusion.ddpm.LatentDiffusion.encode_first_stage' 上，并根据 first_stage_cond 确定是否需要数据类型升级
CondFunc('ldm.models.diffusion.ddpm.LatentDiffusion.encode_first_stage', first_stage_sub, first_stage_cond)
# 将 orig_func 应用到 'ldm.models.diffusion.ddpm.LatentDiffusion.get_first_stage_encoding' 上，并根据 first_stage_cond 确定是否需要数据类型升级
CondFunc('ldm.models.diffusion.ddpm.LatentDiffusion.get_first_stage_encoding', lambda orig_func, *args, **kwargs: orig_func(*args, **kwargs).float(), first_stage_cond)
# 将 apply_model 函数应用到 'sgm.modules.diffusionmodules.wrappers.OpenAIWrapper.forward' 上，并根据 unet_needs_upcast 确定是否需要数据类型升级
CondFunc('sgm.modules.diffusionmodules.wrappers.OpenAIWrapper.forward', apply_model, unet_needs_upcast)
# 调用 CondFunc 函数，传入模块路径和一个 lambda 函数作为参数
CondFunc('sgm.modules.diffusionmodules.openaimodel.timestep_embedding', 
    # lambda 函数接受原始函数 orig_func、时间步长 timesteps 和其他参数 *args、**kwargs
    lambda orig_func, timesteps, *args, **kwargs: 
        # 调用原始函数 orig_func，并根据条件将返回结果转换为 torch.float32 或 devices.dtype_unet 类型
        orig_func(timesteps, *args, **kwargs).to(torch.float32 if timesteps.dtype == torch.int64 else devices.dtype_unet), 
    # unet_needs_upcast 作为最后一个参数传入 CondFunc 函数
    unet_needs_upcast)
```