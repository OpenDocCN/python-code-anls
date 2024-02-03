# `stable-diffusion-webui\modules\sd_hijack_optimizations.py`

```
# 导入必要的模块
from __future__ import annotations
import math
import psutil
import platform

import torch
from torch import einsum

from ldm.util import default
from einops import rearrange

# 导入自定义模块
from modules import shared, errors, devices, sub_quadratic_attention
from modules.hypernetworks import hypernetwork

# 导入不同模块的相同名称的类
import ldm.modules.attention
import ldm.modules.diffusionmodules.model

import sgm.modules.attention
import sgm.modules.diffusionmodules.model

# 保存 ldm.modules.diffusionmodules.model.AttnBlock.forward 的引用
diffusionmodules_model_AttnBlock_forward = ldm.modules.diffusionmodules.model.AttnBlock.forward
# 保存 sgm.modules.diffusionmodules.model.AttnBlock.forward 的引用
sgm_diffusionmodules_model_AttnBlock_forward = sgm.modules.diffusionmodules.model.AttnBlock.forward

# 定义 SdOptimization 类
class SdOptimization:
    name: str = None
    label: str | None = None
    cmd_opt: str | None = None
    priority: int = 0

    # 返回标题
    def title(self):
        if self.label is None:
            return self.name

        return f"{self.name} - {self.label}"

    # 判断优化是否可用
    def is_available(self):
        return True

    # 应用优化
    def apply(self):
        pass

    # 撤销优化
    def undo(self):
        # 恢复 ldm.modules.attention.CrossAttention.forward 的原始值
        ldm.modules.attention.CrossAttention.forward = hypernetwork.attention_CrossAttention_forward
        # 恢复 ldm.modules.diffusionmodules.model.AttnBlock.forward 的原始值
        ldm.modules.diffusionmodules.model.AttnBlock.forward = diffusionmodules_model_AttnBlock_forward

        # 恢复 sgm.modules.attention.CrossAttention.forward 的原始值
        sgm.modules.attention.CrossAttention.forward = hypernetwork.attention_CrossAttention_forward
        # 恢复 sgm.modules.diffusionmodules.model.AttnBlock.forward 的原始值
        sgm.modules.diffusionmodules.model.AttnBlock.forward = sgm_diffusionmodules_model_AttnBlock_forward

# 定义 SdOptimizationXformers 类，继承自 SdOptimization
class SdOptimizationXformers(SdOptimization):
    name = "xformers"
    cmd_opt = "xformers"
    priority = 100

    # 判断 xformers 优化是否可用
    def is_available(self):
        return shared.cmd_opts.force_enable_xformers or (shared.xformers_available and torch.cuda.is_available() and (6, 0) <= torch.cuda.get_device_capability(shared.device) <= (9, 0))
    # 重写ldm和sgm模块中的CrossAttention类的forward方法，使用xformers_attention_forward函数
    ldm.modules.attention.CrossAttention.forward = xformers_attention_forward
    sgm.modules.attention.CrossAttention.forward = xformers_attention_forward
    # 重写ldm和sgm模块中的AttnBlock类的forward方法，使用xformers_attnblock_forward函数
    ldm.modules.diffusionmodules.model.AttnBlock.forward = xformers_attnblock_forward
    sgm.modules.diffusionmodules.model.AttnBlock.forward = xformers_attnblock_forward
# 定义一个继承自 SdOptimization 的类 SdOptimizationSdpNoMem，用于优化 scaled dot product without memory efficient attention
class SdOptimizationSdpNoMem(SdOptimization):
    # 定义类属性 name 为 "sdp-no-mem"
    name = "sdp-no-mem"
    # 定义类属性 label 为 "scaled dot product without memory efficient attention"
    label = "scaled dot product without memory efficient attention"
    # 定义类属性 cmd_opt 为 "opt_sdp_no_mem_attention"
    cmd_opt = "opt_sdp_no_mem_attention"
    # 定义类属性 priority 为 80
    priority = 80

    # 定义方法 is_available，用于检查是否存在 torch.nn.functional.scaled_dot_product_attention 方法
    def is_available(self):
        return hasattr(torch.nn.functional, "scaled_dot_product_attention") and callable(torch.nn.functional.scaled_dot_product_attention)

    # 定义方法 apply，用于应用优化
    def apply(self):
        # 重写 ldm.modules.attention.CrossAttention.forward 方法为 scaled_dot_product_no_mem_attention_forward
        ldm.modules.attention.CrossAttention.forward = scaled_dot_product_no_mem_attention_forward
        # 重写 ldm.modules.diffusionmodules.model.AttnBlock.forward 方法为 sdp_no_mem_attnblock_forward
        ldm.modules.diffusionmodules.model.AttnBlock.forward = sdp_no_mem_attnblock_forward
        # 重写 sgm.modules.attention.CrossAttention.forward 方法为 scaled_dot_product_no_mem_attention_forward
        sgm.modules.attention.CrossAttention.forward = scaled_dot_product_no_mem_attention_forward
        # 重写 sgm.modules.diffusionmodules.model.AttnBlock.forward 方法为 sdp_no_mem_attnblock_forward


# 定义一个继承自 SdOptimizationSdpNoMem 的类 SdOptimizationSdp，用于优化 scaled dot product
class SdOptimizationSdp(SdOptimizationSdpNoMem):
    # 定义类属性 name 为 "sdp"
    name = "sdp"
    # 定义类属性 label 为 "scaled dot product"
    label = "scaled dot product"
    # 定义类属性 cmd_opt 为 "opt_sdp_attention"
    cmd_opt = "opt_sdp_attention"
    # 定义类属性 priority 为 70
    priority = 70

    # 重写 apply 方法，用于应用优化
    def apply(self):
        # 重写 ldm.modules.attention.CrossAttention.forward 方法为 scaled_dot_product_attention_forward
        ldm.modules.attention.CrossAttention.forward = scaled_dot_product_attention_forward
        # 重写 ldm.modules.diffusionmodules.model.AttnBlock.forward 方法为 sdp_attnblock_forward
        ldm.modules.diffusionmodules.model.AttnBlock.forward = sdp_attnblock_forward
        # 重写 sgm.modules.attention.CrossAttention.forward 方法为 scaled_dot_product_attention_forward
        sgm.modules.attention.CrossAttention.forward = scaled_dot_product_attention_forward
        # 重写 sgm.modules.diffusionmodules.model.AttnBlock.forward 方法为 sdp_attnblock_forward


# 定义一个继承自 SdOptimization 的类 SdOptimizationSubQuad，用于优化 sub-quadratic
class SdOptimizationSubQuad(SdOptimization):
    # 定义类属性 name 为 "sub-quadratic"
    name = "sub-quadratic"
    # 定义类属性 cmd_opt 为 "opt_sub_quad_attention"

    # 定义 priority 属性为属性方法，根据 shared.device.type 的值返回不同的优先级
    @property
    def priority(self):
        return 1000 if shared.device.type == 'mps' else 10

    # 重写 apply 方法，用于应用优化
    def apply(self):
        # 重写 ldm.modules.attention.CrossAttention.forward 方法为 sub_quad_attention_forward
        ldm.modules.attention.CrossAttention.forward = sub_quad_attention_forward
        # 重写 ldm.modules.diffusionmodules.model.AttnBlock.forward 方法为 sub_quad_attnblock_forward
        ldm.modules.diffusionmodules.model.AttnBlock.forward = sub_quad_attnblock_forward
        # 重写 sgm.modules.attention.CrossAttention.forward 方法为 sub_quad_attention_forward
        sgm.modules.attention.CrossAttention.forward = sub_quad_attention_forward
        # 重写 sgm.modules.diffusionmodules.model.AttnBlock.forward 方法为 sub_quad_attnblock_forward


# 定义一个继承自 SdOptimization 的类 SdOptimizationV1，用于优化 original v1
class SdOptimizationV1(SdOptimization):
    # 定义类属性 name 为 "V1"
    name = "V1"
    # 定义类属性 label 为 "original v1"
    label = "original v1"
    # 定义类属性 cmd_opt 为 "opt_split_attention_v1"
    # 设置优先级为10
    priority = 10

    # 重写ldm模块中的CrossAttention类的forward方法为split_cross_attention_forward_v1函数
    ldm.modules.attention.CrossAttention.forward = split_cross_attention_forward_v1
    # 重写sgm模块中的CrossAttention类的forward方法为split_cross_attention_forward_v1函数
    sgm.modules.attention.CrossAttention.forward = split_cross_attention_forward_v1
# 定义一个名为 SdOptimizationInvokeAI 的类，继承自 SdOptimization 类
class SdOptimizationInvokeAI(SdOptimization):
    # 设置类属性 name 为 "InvokeAI"
    name = "InvokeAI"
    # 设置类属性 cmd_opt 为 "opt_split_attention_invokeai"
    cmd_opt = "opt_split_attention_invokeai"

    # 定义一个名为 priority 的属性方法
    @property
    def priority(self):
        # 如果设备类型不是 'mps' 并且没有可用的 CUDA 设备，则返回 1000，否则返回 10
        return 1000 if shared.device.type != 'mps' and not torch.cuda.is_available() else 10

    # 定义一个名为 apply 的方法
    def apply(self):
        # 将 ldm 模块中的 attention.CrossAttention.forward 方法设置为 split_cross_attention_forward_invokeAI
        ldm.modules.attention.CrossAttention.forward = split_cross_attention_forward_invokeAI
        # 将 sgm 模块中的 attention.CrossAttention.forward 方法设置为 split_cross_attention_forward_invokeAI


# 定义一个名为 SdOptimizationDoggettx 的类，继承自 SdOptimization 类
class SdOptimizationDoggettx(SdOptimization):
    # 设置类属性 name 为 "Doggettx"
    name = "Doggettx"
    # 设置类属性 cmd_opt 为 "opt_split_attention"
    cmd_opt = "opt_split_attention"
    # 设置类属性 priority 为 90
    priority = 90

    # 定义一个名为 apply 的方法
    def apply(self):
        # 将 ldm 模块中的 attention.CrossAttention.forward 方法设置为 split_cross_attention_forward
        ldm.modules.attention.CrossAttention.forward = split_cross_attention_forward
        # 将 ldm 模块中的 diffusionmodules.model.AttnBlock.forward 方法设置为 cross_attention_attnblock_forward
        ldm.modules.diffusionmodules.model.AttnBlock.forward = cross_attention_attnblock_forward
        # 将 sgm 模块中的 attention.CrossAttention.forward 方法设置为 split_cross_attention_forward
        sgm.modules.attention.CrossAttention.forward = split_cross_attention_forward
        # 将 sgm 模块中的 diffusionmodules.model.AttnBlock.forward 方法设置为 cross_attention_attnblock_forward


# 定义一个名为 list_optimizers 的函数，接受一个参数 res
def list_optimizers(res):
    # 将以下优化器实例添加到 res 列表中
    res.extend([
        SdOptimizationXformers(),
        SdOptimizationSdpNoMem(),
        SdOptimizationSdp(),
        SdOptimizationSubQuad(),
        SdOptimizationV1(),
        SdOptimizationInvokeAI(),
        SdOptimizationDoggettx(),
    ])


# 如果 shared.cmd_opts.xformers 或 shared.cmd_opts.force_enable_xformers 为真
if shared.cmd_opts.xformers or shared.cmd_opts.force_enable_xformers:
    # 尝试导入 xformers.ops 模块
    try:
        import xformers.ops
        # 设置 shared.xformers_available 为 True
        shared.xformers_available = True
    # 如果导入出现异常
    except Exception:
        # 报告错误信息 "Cannot import xformers"，并输出异常信息
        errors.report("Cannot import xformers", exc_info=True)


# 定义一个名为 get_available_vram 的函数
def get_available_vram():
    # 如果设备类型为 'cuda'
    if shared.device.type == 'cuda':
        # 获取 CUDA 设备的内存统计信息
        stats = torch.cuda.memory_stats(shared.device)
        # 获取当前活跃内存
        mem_active = stats['active_bytes.all.current']
        # 获取当前保留内存
        mem_reserved = stats['reserved_bytes.all.current']
        # 获取 CUDA 设备的空闲内存和总内存
        mem_free_cuda, _ = torch.cuda.mem_get_info(torch.cuda.current_device())
        # 计算 Torch 保留内存的空闲内存
        mem_free_torch = mem_reserved - mem_active
        # 计算总的空闲内存
        mem_free_total = mem_free_cuda + mem_free_torch
        # 返回总的空闲内存
        return mem_free_total
    # 如果设备类型不是 'cuda'
    else:
        # 返回系统虚拟内存的可用空间
        return psutil.virtual_memory().available
# 定义一个方法，用于执行分割交叉注意力的前向传播
def split_cross_attention_forward_v1(self, x, context=None, mask=None, **kwargs):
    # 获取头数
    h = self.heads

    # 将输入 x 转换为查询向量 q
    q_in = self.to_q(x)
    # 如果没有提供上下文，则使用输入 x 作为上下文
    context = default(context, x)

    # 使用超网络应用超网络获取上下文的键和值
    context_k, context_v = hypernetwork.apply_hypernetworks(shared.loaded_hypernetworks, context)
    # 将上下文的键转换为键向量 k，将上下文的值转换为值向量 v
    k_in = self.to_k(context_k)
    v_in = self.to_v(context_v)
    # 删除不再需要的变量，以释放内存
    del context, context_k, context_v, x

    # 重新排列查询向量 q、键向量 k 和值向量 v，以便进行计算
    q, k, v = (rearrange(t, 'b n (h d) -> (b h) n d', h=h) for t in (q_in, k_in, v_in))
    # 删除不再需要的变量，以释放内存
    del q_in, k_in, v_in

    # 获取查询、键、值向量的数据类型
    dtype = q.dtype
    # 如果需要升级注意力计算的精度，则将查询、键、值向量转换为浮点数类型
    if shared.opts.upcast_attn:
        q, k, v = q.float(), k.float(), v.float()

    # 在不使用自动混合精度的情况下执行以下代码块
    with devices.without_autocast(disable=not shared.opts.upcast_attn):
        # 初始化结果张量 r1
        r1 = torch.zeros(q.shape[0], q.shape[1], v.shape[2], device=q.device, dtype=q.dtype)
        # 遍历查询向量 q 的每个批次
        for i in range(0, q.shape[0], 2):
            end = i + 2
            # 计算注意力分数 s1
            s1 = einsum('b i d, b j d -> b i j', q[i:end], k[i:end])
            s1 *= self.scale

            # 对注意力分数 s1 进行 softmax 操作
            s2 = s1.softmax(dim=-1)
            del s1

            # 计算加权值并更新结果张量 r1
            r1[i:end] = einsum('b i j, b j d -> b i d', s2, v[i:end])
            del s2
        # 删除不再需要的变量，以释放内存
        del q, k, v

    # 将结果张量 r1 转换为指定的数据类型
    r1 = r1.to(dtype)

    # 重新排列结果张量 r1，以便返回输出
    r2 = rearrange(r1, '(b h) n d -> b n (h d)', h=h)
    # 删除不再需要的变量，以释放内存
    del r1

    # 返回输出结果
    return self.to_out(r2)


# 从指定链接获取代码，并进行修改后定义一个方法，用于执行分割交叉注意力的前向传播
def split_cross_attention_forward(self, x, context=None, mask=None, **kwargs):
    # 获取头数
    h = self.heads

    # 将输入 x 转换为查询向量 q
    q_in = self.to_q(x)
    # 如果没有提供上下文，则使用输入 x 作为上下文
    context = default(context, x)

    # 使用超网络应用超网络获取上下文的键和值
    context_k, context_v = hypernetwork.apply_hypernetworks(shared.loaded_hypernetworks, context)
    # 将上下文的键转换为键向量 k，将上下文的值转换为值向量 v
    k_in = self.to_k(context_k)
    v_in = self.to_v(context_v)

    # 获取查询向量 q 的数据类型
    dtype = q_in.dtype
    # 如果需要升级注意力计算的精度，则将查询、键、值向量转换为浮点数类型
    if shared.opts.upcast_attn:
        q_in, k_in, v_in = q_in.float(), k_in.float(), v_in if v_in.device.type == 'mps' else v_in.float()
    # 禁用自动类型转换，根据条件设置是否禁用自动类型转换
    with devices.without_autocast(disable=not shared.opts.upcast_attn):
        # 将输入的 k_in 乘以缩放因子
        k_in = k_in * self.scale

        # 删除 context, x 变量
        del context, x

        # 重新排列输入的 q_in, k_in, v_in 张量的维度
        q, k, v = (rearrange(t, 'b n (h d) -> (b h) n d', h=h) for t in (q_in, k_in, v_in))
        # 删除 q_in, k_in, v_in 变量
        del q_in, k_in, v_in

        # 创建一个全零张量 r1，用于存储计算结果
        r1 = torch.zeros(q.shape[0], q.shape[1], v.shape[2], device=q.device, dtype=q.dtype)

        # 获取可用的显存总量
        mem_free_total = get_available_vram()

        # 定义常量
        gb = 1024 ** 3
        # 计算张量的大小
        tensor_size = q.shape[0] * q.shape[1] * k.shape[1] * q.element_size()
        # 根据元素大小计算修正因子
        modifier = 3 if q.element_size() == 2 else 2.5
        # 计算所需的内存大小
        mem_required = tensor_size * modifier
        # 初始化步数
        steps = 1

        # 如果所需内存大于可用内存
        if mem_required > mem_free_total:
            # 计算步数
            steps = 2 ** (math.ceil(math.log(mem_required / mem_free_total, 2)))
            # 打印信息
            # print(f"Expected tensor size:{tensor_size/gb:0.1f}GB, cuda free:{mem_free_cuda/gb:0.1f}GB "
            #       f"torch free:{mem_free_torch/gb:0.1f} total:{mem_free_total/gb:0.1f} steps:{steps}")

        # 如果步数大于64
        if steps > 64:
            # 计算最大分辨率
            max_res = math.floor(math.sqrt(math.sqrt(mem_free_total / 2.5)) / 8) * 64
            # 抛出异常
            raise RuntimeError(f'Not enough memory, use lower resolution (max approx. {max_res}x{max_res}). '
                               f'Need: {mem_required / 64 / gb:0.1f}GB free, Have:{mem_free_total / gb:0.1f}GB free')

        # 计算每个切片的大小
        slice_size = q.shape[1] // steps
        # 遍历每个切片
        for i in range(0, q.shape[1], slice_size):
            end = min(i + slice_size, q.shape[1])
            # 计算 s1
            s1 = einsum('b i d, b j d -> b i j', q[:, i:end], k)

            # 计算 s2，并进行 softmax 操作
            s2 = s1.softmax(dim=-1, dtype=q.dtype)
            # 删除 s1 变量
            del s1

            # 计算最终结果 r1
            r1[:, i:end] = einsum('b i j, b j d -> b i d', s2, v)
            # 删除 s2 变量
            del s2

        # 删除 q, k, v 变量
        del q, k, v

    # 将 r1 转换为指定的数据类型
    r1 = r1.to(dtype)

    # 重新排列 r1 的维度
    r2 = rearrange(r1, '(b h) n d -> b n (h d)', h=h)
    # 删除 r1 变量
    del r1

    # 返回处理后的结果
    return self.to_out(r2)
# 计算系统总内存大小（单位：GB）
mem_total_gb = psutil.virtual_memory().total // (1 << 30)

# 根据输入的查询（q）、键（k）、值（v）计算注意力分布
def einsum_op_compvis(q, k, v):
    # 计算注意力分布矩阵
    s = einsum('b i d, b j d -> b i j', q, k)
    # 对注意力分布进行 softmax 归一化
    s = s.softmax(dim=-1, dtype=s.dtype)
    # 根据注意力分布计算加权和
    return einsum('b i j, b j d -> b i d', s, v)

# 对输入的查询（q）、键（k）、值（v）进行分块计算，每块大小为 slice_size
def einsum_op_slice_0(q, k, v, slice_size):
    # 初始化结果张量
    r = torch.zeros(q.shape[0], q.shape[1], v.shape[2], device=q.device, dtype=q.dtype)
    # 循环处理每个分块
    for i in range(0, q.shape[0], slice_size):
        end = i + slice_size
        r[i:end] = einsum_op_compvis(q[i:end], k[i:end], v[i:end])
    return r

# 对输入的查询（q）、键（k）、值（v）进行分块计算，每块大小为 slice_size
def einsum_op_slice_1(q, k, v, slice_size):
    # 初始化结果张量
    r = torch.zeros(q.shape[0], q.shape[1], v.shape[2], device=q.device, dtype=q.dtype)
    # 循环处理每个分块
    for i in range(0, q.shape[1], slice_size):
        end = i + slice_size
        r[:, i:end] = einsum_op_compvis(q[:, i:end], k, v)
    return r

# 根据输入的查询（q）、键（k）、值（v）大小选择不同的计算方式
def einsum_op_mps_v1(q, k, v):
    if q.shape[0] * q.shape[1] <= 2**16: # (512x512) max q.shape[1]: 4096
        return einsum_op_compvis(q, k, v)
    else:
        # 计算分块大小
        slice_size = math.floor(2**30 / (q.shape[0] * q.shape[1]))
        if slice_size % 4096 == 0:
            slice_size -= 1
        return einsum_op_slice_1(q, k, v, slice_size)

# 根据输入的查询（q）、键（k）、值（v）大小和系统总内存选择不同的计算方式
def einsum_op_mps_v2(q, k, v):
    if mem_total_gb > 8 and q.shape[0] * q.shape[1] <= 2**16:
        return einsum_op_compvis(q, k, v)
    else:
        return einsum_op_slice_0(q, k, v, 1)

# 根据输入的查询（q）、键（k）、值（v）大小和最大张量大小选择不同的计算方式
def einsum_op_tensor_mem(q, k, v, max_tensor_mb):
    # 计算张量大小（单位：MB）
    size_mb = q.shape[0] * q.shape[1] * k.shape[1] * q.element_size() // (1 << 20)
    if size_mb <= max_tensor_mb:
        return einsum_op_compvis(q, k, v)
    # 计算分块大小
    div = 1 << int((size_mb - 1) / max_tensor_mb).bit_length()
    if div <= q.shape[0]:
        return einsum_op_slice_0(q, k, v, q.shape[0] // div)
    return einsum_op_slice_1(q, k, v, max(q.shape[1] // div, 1))

# 获取当前 CUDA 设备的内存使用情况
def einsum_op_cuda(q, k, v):
    stats = torch.cuda.memory_stats(q.device)
    mem_active = stats['active_bytes.all.current']
    # 获取当前系统中已经分配但未使用的内存大小
    mem_reserved = stats['reserved_bytes.all.current']
    # 获取当前 CUDA 设备上的空闲内存大小
    mem_free_cuda, _ = torch.cuda.mem_get_info(q.device)
    # 计算 Torch 可用内存大小
    mem_free_torch = mem_reserved - mem_active
    # 计算总的可用内存大小
    mem_free_total = mem_free_cuda + mem_free_torch
    # 将总的可用内存大小除以3.3再除以2^20，作为安全因子，考虑拷贝和碎片化
    return einsum_op_tensor_mem(q, k, v, mem_free_total / 3.3 / (1 << 20))
# 定义一个函数，根据输入的查询、键、值计算输出
def einsum_op(q, k, v):
    # 如果查询张量在 CUDA 设备上，则调用 einsum_op_cuda 函数处理
    if q.device.type == 'cuda':
        return einsum_op_cuda(q, k, v)

    # 如果查询张量在 MPS 设备上
    if q.device.type == 'mps':
        # 如果总内存大于等于32GB且查询张量的第一个维度不能被32整除且查询张量的维度乘积小于2^18
        if mem_total_gb >= 32 and q.shape[0] % 32 != 0 and q.shape[0] * q.shape[1] < 2**18:
            # 调用 einsum_op_mps_v1 处理
            return einsum_op_mps_v1(q, k, v)
        # 否则调用 einsum_op_mps_v2 处理
        return einsum_op_mps_v2(q, k, v)

    # 较小的切片由于 L2/L3/SLC 缓存更快
    # 在具有8MB L3缓存的i7上进行了测试
    return einsum_op_tensor_mem(q, k, v, 32)


# 定义一个函数，执行交叉注意力的前向传播
def split_cross_attention_forward_invokeAI(self, x, context=None, mask=None, **kwargs):
    # 头数
    h = self.heads

    # 将输入 x 转换为查询张量 q
    q = self.to_q(x)
    # 如果未提供上下文，则使用输入 x 作为上下文
    context = default(context, x)

    # 使用超网络应用超网络获取上下文的键和值
    context_k, context_v = hypernetwork.apply_hypernetworks(shared.loaded_hypernetworks, context)
    # 将上下文的键转换为 k，上下文的值转换为 v
    k = self.to_k(context_k)
    v = self.to_v(context_v)
    # 释放内存
    del context, context_k, context_v, x

    # 数据类型为查询张量的数据类型
    dtype = q.dtype
    # 如果选项中开启了上转换
    if shared.opts.upcast_attn:
        # 将 q、k 转换为 float 类型，如果 v 在 MPS 设备上，则保持不变，否则转换为 float 类型
        q, k, v = q.float(), k.float(), v if v.device.type == 'mps' else v.float()

    # 禁用自动转换
    with devices.without_autocast(disable=not shared.opts.upcast_attn):
        # 对 k 乘以缩放因子
        k = k * self.scale

        # 重排 q、k、v 张量的维度
        q, k, v = (rearrange(t, 'b n (h d) -> (b h) n d', h=h) for t in (q, k, v))
        # 调用 einsum_op 函数计算输出
        r = einsum_op(q, k, v)
    # 将输出转换为指定数据类型
    r = r.to(dtype)
    # 将输出重排为原始形状
    return self.to_out(rearrange(r, '(b h) n d -> b n (h d)', h=h))


# 基于 Birch-san 修改的 sub-quadratic attention 实现
# sub_quad_attention_forward 函数在网页UI界面的许可证部分列出的 MIT 许可证下
def sub_quad_attention_forward(self, x, context=None, mask=None, **kwargs):
    # 断言未实现注意力掩码
    assert mask is None, "attention-mask not currently implemented for SubQuadraticCrossAttnProcessor."

    # 头数
    h = self.heads

    # 将输入 x 转换为查询张量 q
    q = self.to_q(x)
    # 如果未提供上下文，则使用输入 x 作为上下文
    context = default(context, x)

    # 使用超网络应用超网络获取上下文的键和值
    context_k, context_v = hypernetwork.apply_hypernetworks(shared.loaded_hypernetworks, context)
    # 将上下文的键转换为 k
    k = self.to_k(context_k)
    # 将上下文向量转换为 V 矩阵
    v = self.to_v(context_v)
    # 删除不再需要的变量，释放内存
    del context, context_k, context_v, x

    # 对查询、键、值进行形状变换，以便进行注意力计算
    q = q.unflatten(-1, (h, -1)).transpose(1,2).flatten(end_dim=1)
    k = k.unflatten(-1, (h, -1)).transpose(1,2).flatten(end_dim=1)
    v = v.unflatten(-1, (h, -1)).transpose(1,2).flatten(end_dim=1)

    # 如果查询张量在 MPS 设备上，则需要进行内存连续性处理
    if q.device.type == 'mps':
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()

    # 获取查询张量的数据类型
    dtype = q.dtype
    # 如果需要升级注意力计算的数据类型
    if shared.opts.upcast_attn:
        q, k = q.float(), k.float()

    # 调用子块级别的注意力计算函数
    x = sub_quad_attention(q, k, v, q_chunk_size=shared.cmd_opts.sub_quad_q_chunk_size, kv_chunk_size=shared.cmd_opts.sub_quad_kv_chunk_size, chunk_threshold=shared.cmd_opts.sub_quad_chunk_threshold, use_checkpoint=self.training)

    # 将计算结果转换为指定数据类型
    x = x.to(dtype)

    # 对计算结果进行形状变换，以便进行输出投影
    x = x.unflatten(0, (-1, h)).transpose(1,2).flatten(start_dim=2)

    # 获取输出投影和 dropout 操作
    out_proj, dropout = self.to_out
    # 对计算结果进行输出投影
    x = out_proj(x)
    # 对输出结果进行 dropout 操作
    x = dropout(x)

    # 返回最终的计算结果
    return x
# 定义一个函数，实现子二次注意力机制，接受查询、键、值张量作为输入，还可以设置查询和键值的分块大小，默认为1024
def sub_quad_attention(q, k, v, q_chunk_size=1024, kv_chunk_size=None, kv_chunk_size_min=None, chunk_threshold=None, use_checkpoint=True):
    # 计算每个标记的字节数
    bytes_per_token = torch.finfo(q.dtype).bits//8
    # 获取查询张量的形状信息
    batch_x_heads, q_tokens, _ = q.shape
    # 获取键张量的形状信息
    _, k_tokens, _ = k.shape
    # 计算查询-键矩阵乘法的字节数
    qk_matmul_size_bytes = batch_x_heads * bytes_per_token * q_tokens * k_tokens

    # 如果未设置分块阈值
    if chunk_threshold is None:
        # 如果设备类型为'mps'
        if q.device.type == 'mps':
            # 根据处理器类型设置分块阈值字节数
            chunk_threshold_bytes = 268435456 * (2 if platform.processor() == 'i386' else bytes_per_token)
        else:
            # 根据可用显存的70%设置分块阈值字节数
            chunk_threshold_bytes = int(get_available_vram() * 0.7)
    # 如果分块阈值为0
    elif chunk_threshold == 0:
        chunk_threshold_bytes = None
    else:
        # 根据设定的分块阈值百分比和可用显存计算分块阈值字节数
        chunk_threshold_bytes = int(0.01 * chunk_threshold * get_available_vram())

    # 如果未设置最小键值分块大小并且存在分块阈值字节数
    if kv_chunk_size_min is None and chunk_threshold_bytes is not None:
        # 根据分块阈值字节数和张量形状信息计算最小键值分块大小
        kv_chunk_size_min = chunk_threshold_bytes // (batch_x_heads * bytes_per_token * (k.shape[2] + v.shape[2]))
    # 如果最小键值分块大小为0
    elif kv_chunk_size_min == 0:
        kv_chunk_size_min = None

    # 如果存在分块阈值字节数并且查询-键矩阵乘法字节数小于等于分块阈值字节数
    if chunk_threshold_bytes is not None and qk_matmul_size_bytes <= chunk_threshold_bytes:
        # 大矩阵乘法适合我们的内存限制；在一个分块中完成所有操作，即使用未分块的快速路径
        kv_chunk_size = k_tokens

    # 禁用自动转换时，执行以下代码块
    with devices.without_autocast(disable=q.dtype == v.dtype):
        # 调用efficient_dot_product_attention函数，传入查询、键、值张量，以及其他参数
        return sub_quadratic_attention.efficient_dot_product_attention(
            q,
            k,
            v,
            query_chunk_size=q_chunk_size,
            kv_chunk_size=kv_chunk_size,
            kv_chunk_size_min = kv_chunk_size_min,
            use_checkpoint=use_checkpoint,
        )

# 定义一个函数，获取xformers_flash_attention_op操作
def get_xformers_flash_attention_op(q, k, v):
    # 如果未设置xformers_flash_attention标志，则返回None
    if not shared.cmd_opts.xformers_flash_attention:
        return None
    # 尝试导入 FlashAttentionOp 模块
    try:
        flash_attention_op = xformers.ops.MemoryEfficientAttentionFlashAttentionOp
        # 尝试获取 FlashAttentionOp 的前向和后向操作
        fw, bw = flash_attention_op
        # 如果前向操作支持给定的输入参数，则返回 FlashAttentionOp
        if fw.supports(xformers.ops.fmha.Inputs(query=q, key=k, value=v, attn_bias=None)):
            return flash_attention_op
    # 捕获任何异常并显示错误信息
    except Exception as e:
        errors.display_once(e, "enabling flash attention")

    # 如果以上代码块未返回 FlashAttentionOp，则返回 None
    return None
# 定义 Transformer 模型的注意力前向传播函数，接受输入 x，上下文 context，掩码 mask 和其他关键字参数
def xformers_attention_forward(self, x, context=None, mask=None, **kwargs):
    # 获取头数
    h = self.heads
    # 将输入 x 转换为查询向量 q
    q_in = self.to_q(x)
    # 如果没有指定上下文，则使用输入 x 作为上下文
    context = default(context, x)

    # 使用超网络应用于上下文，获取上下文的键和值
    context_k, context_v = hypernetwork.apply_hypernetworks(shared.loaded_hypernetworks, context)
    # 将上下文的键转换为键向量 k，将上下文的值转换为值向量 v
    k_in = self.to_k(context_k)
    v_in = self.to_v(context_v)

    # 将查询向量 q、键向量 k 和值向量 v 重排为多头注意力的形式
    q, k, v = (rearrange(t, 'b n (h d) -> b n h d', h=h) for t in (q_in, k_in, v_in))
    # 释放不再需要的变量
    del q_in, k_in, v_in

    # 获取查询、键、值的数据类型
    dtype = q.dtype
    # 如果需要向上转型，则将查询、键、值转换为浮点数类型
    if shared.opts.upcast_attn:
        q, k, v = q.float(), k.float(), v.float()

    # 使用 memory_efficient_attention 函数计算注意力，得到输出 out
    out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=get_xformers_flash_attention_op(q, k, v))

    # 将输出 out 转换为原始数据类型
    out = out.to(dtype)

    # 将输出 out 重排为原始形状
    out = rearrange(out, 'b n h d -> b n (h d)', h=h)
    # 返回输出 out 经过输出转换的结果
    return self.to_out(out)


# 基于 Diffusers 对缩放点积注意力的使用，参考 https://github.com/huggingface/diffusers/blob/c7da8fd23359a22d0df2741688b5b4f33c26df21/src/diffusers/models/cross_attention.py
# scaled_dot_product_attention_forward 函数包含了 Apache-2.0 许可下的代码部分，详见网页 UI 界面的许可证部分的缩放点积注意力
def scaled_dot_product_attention_forward(self, x, context=None, mask=None, **kwargs):
    # 获取输入 x 的批量大小、序列长度和内部维度
    batch_size, sequence_length, inner_dim = x.shape

    # 如果存在掩码，则准备注意力掩码
    if mask is not None:
        mask = self.prepare_attention_mask(mask, sequence_length, batch_size)
        mask = mask.view(batch_size, self.heads, -1, mask.shape[-1])

    # 获取头数
    h = self.heads
    # 将输入 x 转换为查询向量 q
    q_in = self.to_q(x)
    # 如果没有指定上下文，则使用输入 x 作为上下文
    context = default(context, x)

    # 使用超网络应用于上下文，获取上下文的键和值
    context_k, context_v = hypernetwork.apply_hypernetworks(shared.loaded_hypernetworks, context)
    # 将上下文的键转换为键向量 k，将上下文的值转换为值向量 v
    k_in = self.to_k(context_k)
    v_in = self.to_v(context_v)

    # 计算每个头的维度
    head_dim = inner_dim // h
    # 将查询、键、值重排为多头注意力的形式
    q = q_in.view(batch_size, -1, h, head_dim).transpose(1, 2)
    k = k_in.view(batch_size, -1, h, head_dim).transpose(1, 2)
    v = v_in.view(batch_size, -1, h, head_dim).transpose(1, 2)

    # 释放不再需要的变量
    del q_in, k_in, v_in

    # 获取查询的数据类型
    dtype = q.dtype
    # 如果设置了 upcast_attn 标志，则将 q、k、v 转换为 float 类型
    if shared.opts.upcast_attn:
        q, k, v = q.float(), k.float(), v.float()

    # 使用 scaled_dot_product_attention 函数计算注意力矩阵，输出形状为(batch, num_heads, seq_len, head_dim)
    hidden_states = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False
    )

    # 调整 hidden_states 的维度顺序，然后重塑为(batch_size, -1, h * head_dim)
    hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, h * head_dim)
    # 将 hidden_states 转换为指定的数据类型
    hidden_states = hidden_states.to(dtype)

    # 线性投影
    hidden_states = self.to_out[0](hidden_states)
    # dropout
    hidden_states = self.to_out[1](hidden_states)
    # 返回处理后的 hidden_states
    return hidden_states
# 使用 scaled dot product attention 进行前向传播，不使用内存优化
def scaled_dot_product_no_mem_attention_forward(self, x, context=None, mask=None, **kwargs):
    # 启用 CUDA SDP 内核，关闭内存优化，进行 scaled dot product attention 前向传播
    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=False):
        return scaled_dot_product_attention_forward(self, x, context, mask)


# 实现交叉注意力机制的前向传播
def cross_attention_attnblock_forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q1 = self.q(h_)
        k1 = self.k(h_)
        v = self.v(h_)

        # 计算注意力
        b, c, h, w = q1.shape

        q2 = q1.reshape(b, c, h*w)
        del q1

        q = q2.permute(0, 2, 1)   # 调整维度顺序
        del q2

        k = k1.reshape(b, c, h*w) # 调整维度顺序
        del k1

        h_ = torch.zeros_like(k, device=q.device)

        # 获取可用的显存
        mem_free_total = get_available_vram()

        tensor_size = q.shape[0] * q.shape[1] * k.shape[2] * q.element_size()
        mem_required = tensor_size * 2.5
        steps = 1

        # 根据内存需求和可用内存计算步数
        if mem_required > mem_free_total:
            steps = 2**(math.ceil(math.log(mem_required / mem_free_total, 2)))

        slice_size = q.shape[1] // steps if (q.shape[1] % steps) == 0 else q.shape[1]
        for i in range(0, q.shape[1], slice_size):
            end = i + slice_size

            w1 = torch.bmm(q[:, i:end], k)     # 计算注意力权重
            w2 = w1 * (int(c)**(-0.5))
            del w1
            w3 = torch.nn.functional.softmax(w2, dim=2, dtype=q.dtype)
            del w2

            # 对值进行注意力聚合
            v1 = v.reshape(b, c, h*w)
            w4 = w3.permute(0, 2, 1)   # 调整维度顺序
            del w3

            h_[:, :, i:end] = torch.bmm(v1, w4)     # 注意力聚合
            del v1, w4

        h2 = h_.reshape(b, c, h, w)
        del h_

        h3 = self.proj_out(h2)
        del h2

        h3 += x

        return h3


def xformers_attnblock_forward(self, x):
    # 尝试执行以下代码块，如果出现异常则执行 except 代码块
    try:
        # 将输入 x 赋值给 h_
        h_ = x
        # 对 h_ 进行归一化处理
        h_ = self.norm(h_)
        # 计算查询向量 q
        q = self.q(h_)
        # 计算键向量 k
        k = self.k(h_)
        # 计算数值向量 v
        v = self.v(h_)
        # 获取 q 的形状信息
        b, c, h, w = q.shape
        # 对查询、键、值向量进行形状重排
        q, k, v = (rearrange(t, 'b c h w -> b (h w) c') for t in (q, k, v))
        # 获取查询向量的数据类型
        dtype = q.dtype
        # 如果需要升级注意力机制的数据类型
        if shared.opts.upcast_attn:
            # 将查询、键向量转换为 float 类型
            q, k = q.float(), k.float()
        # 使查询、键、值向量连续存储
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        # 调用 memory_efficient_attention 函数进行注意力计算
        out = xformers.ops.memory_efficient_attention(q, k, v, op=get_xformers_flash_attention_op(q, k, v))
        # 将输出转换为指定数据类型
        out = out.to(dtype)
        # 对输出进行形状重排
        out = rearrange(out, 'b (h w) c -> b c h w', h=h)
        # 对输出进行投影操作
        out = self.proj_out(out)
        # 返回输入 x 与输出 out 的和
        return x + out
    # 如果出现 NotImplementedError 异常
    except NotImplementedError:
        # 调用 cross_attention_attnblock_forward 函数处理异常情况
        return cross_attention_attnblock_forward(self, x)
# 定义自注意力块的前向传播函数，接收输入张量 x
def sdp_attnblock_forward(self, x):
    # 将输入张量赋值给 h_
    h_ = x
    # 对 h_ 进行归一化处理
    h_ = self.norm(h_)
    # 分别计算查询、键、值
    q = self.q(h_)
    k = self.k(h_)
    v = self.v(h_)
    # 获取查询张量的形状信息
    b, c, h, w = q.shape
    # 重排查询、键、值张量的维度
    q, k, v = (rearrange(t, 'b c h w -> b (h w) c') for t in (q, k, v))
    # 获取查询张量的数据类型
    dtype = q.dtype
    # 如果需要升级注意力机制的数据类型
    if shared.opts.upcast_attn:
        q, k, v = q.float(), k.float(), v.float()
    # 使查询、键、值张量连续
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    # 使用缩放点积注意力机制计算输出
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
    # 将输出转换为指定数据类型
    out = out.to(dtype)
    # 重排输出张量的维度
    out = rearrange(out, 'b (h w) c -> b c h w', h=h)
    # 对输出进行投影
    out = self.proj_out(out)
    # 返回输入张量与输出的和
    return x + out


# 定义无记忆自注意力块的前向传播函数，接收输入张量 x
def sdp_no_mem_attnblock_forward(self, x):
    # 使用 CUDA 加速的 SDP 内核计算
    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=False):
        # 调用自注意力块的前向传播函数
        return sdp_attnblock_forward(self, x)


# 定义子四方注意力块的前向传播函数，接收输入张量 x
def sub_quad_attnblock_forward(self, x):
    # 将输入张量赋值给 h_
    h_ = x
    # 对 h_ 进行归一化处理
    h_ = self.norm(h_)
    # 分别计算查询、键、值
    q = self.q(h_)
    k = self.k(h_)
    v = self.v(h_)
    # 获取查询张量的形状信息
    b, c, h, w = q.shape
    # 重排查询、键、值张量的维度
    q, k, v = (rearrange(t, 'b c h w -> b (h w) c') for t in (q, k, v))
    # 使查询、键、值张量连续
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    # 使用子四方注意力机制计算输出
    out = sub_quad_attention(q, k, v, q_chunk_size=shared.cmd_opts.sub_quad_q_chunk_size, kv_chunk_size=shared.cmd_opts.sub_quad_kv_chunk_size, chunk_threshold=shared.cmd_opts.sub_quad_chunk_threshold, use_checkpoint=self.training)
    # 重排输出张量的维度
    out = rearrange(out, 'b (h w) c -> b c h w', h=h)
    # 对输出进行投影
    out = self.proj_out(out)
    # 返回输入张量与输出的和
    return x + out
```