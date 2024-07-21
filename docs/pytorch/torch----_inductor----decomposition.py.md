# `.\pytorch\torch\_inductor\decomposition.py`

```
# 设置 mypy，允许未类型化的定义
# 导入必要的模块和库
import functools  # 提供高阶函数的实用工具，如函数包装器
import logging  # 提供日志记录功能
import math  # 数学函数库
import sys  # 系统相关的参数和功能
import typing  # Python 类型提示模块
from typing import Optional  # 引入可选类型提示

import torch  # PyTorch 深度学习库
import torch._decomp as decomp  # PyTorch 内部分解模块
import torch._prims_common as utils  # PyTorch 通用原语模块
import torch.ao.quantization.fx._decomposed  # PyTorch 量化分解模块
from torch._decomp import (  # 导入特定函数和类
    core_aten_decompositions,  # 核心 ATen 函数的分解
    get_decompositions,  # 获取分解表
    remove_decompositions,  # 移除指定分解
)
from torch._decomp.decompositions import (  # 导入特定的分解函数
    _grid_sampler_2d as decomp_grid_sampler_2d,  # 网格采样二维分解
    pw_cast_for_opmath,  # 运算数学的类型转换
)
from torch._decomp.decompositions_for_rng import extra_random_decomps  # 用于 RNG 的额外随机分解
from torch._dynamo.utils import counters  # 动态添加功能的计数器工具
from torch._higher_order_ops.out_dtype import out_dtype  # 输出数据类型
from torch._inductor.utils import pad_listlike  # 填充列表的实用工具
from torch._prims_common import (  # 导入通用原语函数
    elementwise_dtypes,  # 元素级操作的数据类型
    ELEMENTWISE_TYPE_PROMOTION_KIND,  # 数据类型提升种类
    type_to_dtype,  # 类型到数据类型的转换
)
from torch.fx.experimental.symbolic_shapes import definitely_true, guard_size_oblivious  # 符号形状实验性模块

from . import config, inductor_prims  # 导入当前目录下的特定模块和包
from .utils import (  # 导入当前目录下的实用工具函数
    is_gpu,  # 检查是否使用 GPU
    needs_fallback_due_to_atomic_add_limitations,  # 检查是否需要回退到原子加限制
    use_scatter_fallback,  # 使用散射回退
)

log = logging.getLogger(__name__)  # 获取当前模块的日志记录器对象
aten = torch.ops.aten  # 获取 PyTorch ATen 原语操作对象
prims = torch.ops.prims  # 获取 PyTorch 原语操作对象
quantized = torch.ops.quantized  # 获取 PyTorch 量化操作对象
quantized_decomposed = torch.ops.quantized_decomposed  # 获取 PyTorch 量化分解操作对象

# 从 ATen 中获取适配器的分解列表，并加入 Inductor 部分的自定义分解
inductor_decompositions = get_decompositions(
    [
        aten._adaptive_avg_pool2d_backward,  # 自适应平均池化反向传播
        aten.arange,  # 生成等差数列
        aten.bitwise_and_,  # 按位与操作
        aten.bitwise_or_,  # 按位或操作
        aten.clamp_min_,  # 最小值截断
        aten.dist,  # 计算张量之间的距离
        aten.empty_like,  # 返回一个形状相同的空张量
        aten.flip,  # 翻转张量的指定维度
        aten.gelu,  # GeLU 激活函数
        aten.hardtanh,  # HardTanh 激活函数
        aten.index_select,  # 根据索引从输入张量中选择子张量
        aten.lcm,  # 计算最小公倍数
        aten.leaky_relu,  # LeakyReLU 激活函数
        aten.linalg_vector_norm,  # 计算矩阵的向量范数
        aten._log_softmax,  # 对数 Softmax 函数
        aten.max_pool2d_with_indices_backward,  # 最大池化反向传播
        aten._native_batch_norm_legit,  # 批量归一化
        aten._native_batch_norm_legit_functional,  # 功能批量归一化
        aten._native_batch_norm_legit_no_training,  # 无训练批量归一化
        aten._batch_norm_with_update,  # 带更新的批量归一化
        aten._batch_norm_with_update_functional,  # 功能带更新的批量归一化
        aten._batch_norm_no_update,  # 无更新的批量归一化
        aten.batch_norm_backward,  # 批量归一化反向传播
        aten.native_batch_norm,  # 原生批量归一化
        aten.native_group_norm,  # 原生分组归一化
        aten.native_layer_norm,  # 原生层归一化
        aten.nll_loss2d_backward,  # 二维负对数似然损失反向传播
        aten._softmax,  # Softmax 函数
        aten.sin_,  # 正弦函数
        aten.sqrt_,  # 平方根函数
        out_dtype,  # 输出数据类型
        aten._to_copy,  # 复制函数
        aten.tril_indices,  # 下三角索引
        aten.triu_indices,  # 上三角索引
        aten.upsample_bilinear2d.vec,  # 双线性插值向量化
        quantized.linear_dynamic_fp16_unpacked_weight,  # 量化动态 FP16 解包权重
    ]
)

# 合并核心 ATen 的分解和 Inductor 部分的分解，形成总的分解字典
decompositions = {**core_aten_decompositions(), **inductor_decompositions}

# 移除从核心 ATen 分解中包含的不需要的分解
decomps_to_exclude = [
    aten._unsafe_index,  # 不安全的索引操作
    aten._unsafe_masked_index,  # 不安全的掩码索引操作
    aten._unsafe_masked_index_put_accumulate,  # 不安全的掩码索引累加操作
    aten._scaled_dot_product_flash_attention_for_cpu.default,  # CPU 闪存注意力的默认缩放点积
    aten._softmax_backward_data,  # Softmax 反向传播数据
    aten.clamp_max,  # 最大值截断
    aten.clamp_min,  # 最小值截断
    aten.glu,  # GLU 激活函数，Inductor 直接降低这个
]
    aten.select_scatter,  # 必须在 ATen 图中，以便使其能够在重新插入过程中正常工作
    aten.slice_scatter,  # 必须在 ATen 图中，以便使其能够在重新插入过程中正常工作
    aten.split.Tensor,  # 感应器直接降低此操作
    aten.squeeze,  # 感应器直接降低此操作
    aten.sum,  # 感应器直接降低此操作
    aten.unbind,  # 感应器直接降低此操作
# 移除已排除的分解函数
remove_decompositions(decompositions, decomps_to_exclude)

# 注册分解函数，如果 ops 是可调用的则遍历 ops，检查是否存在重复的分解函数
def register_decomposition(ops):
    for op in [ops] if callable(ops) else ops:
        if op in decompositions:
            log.warning("duplicate decomp: %s", ops)
    return decomp.register_decomposition(ops, decompositions)

# TODO: 暂时不处理断言语句，因为在图中条件为符号 -> 张量。
# 注册 aten._assert_async.msg 函数的分解实现
@register_decomposition([aten._assert_async.msg])
def assert_async_msg_decomp(tensor, msg):
    return

# 根据 assert_async_msg_decomp，实现 aten._functional_assert_async.msg 函数的非操作分解
@register_decomposition([aten._functional_assert_async.msg])
def functional_assert_async_msg_decomp(tensor, msg):
    return

# 注册 aten.sym_constrain_range_for_size.default 函数的分解实现
@register_decomposition([aten.sym_constrain_range_for_size.default])
def sym_constrain_range_for_size(symbol, *, min=None, max=None):
    return

# 注册 aten.clamp 函数的分解实现，并在数学操作中进行强制类型转换
@register_decomposition([aten.clamp])
@pw_cast_for_opmath
def clamp(x, min=None, max=None):
    if min is not None:
        x = x.clamp_min(min)
    if max is not None:
        x = x.clamp_max(max)
    return x

# 注册 aten.full 函数的分解实现
@register_decomposition([aten.full])
def full(size, fill_value, **kwargs):
    dtype = kwargs.get("dtype")
    if dtype is None:
        kwargs["dtype"] = type_to_dtype(type(fill_value))
        return torch.full(size, fill_value, **kwargs)
    return NotImplemented

# 在主库中不确定如何处理这个，PrimTorch 希望 empty_permuted 转到 prim，
# 通常用户不希望分解到 empty_strided（但是感应器可以接受，因为我们对步幅很了解，并且所有东西都会到 empty_strided）
@register_decomposition([aten.empty_permuted.default])
def empty_permuted(size, physical_layout, **kwargs):
    perm = [0] * len(size)
    for p, l in enumerate(physical_layout):
        perm[l] = p
    return torch.empty([size[l] for l in physical_layout], **kwargs).permute(perm)

# 注册 aten.convolution_backward 函数的分解实现
@register_decomposition([aten.convolution_backward])
def convolution_backward(
    grad_output,
    input,
    weight,
    bias_sizes,
    stride,
    padding,
    dilation,
    transposed,
    output_padding,
    groups,
    output_mask,
):
    if not output_mask[2] or not is_gpu(grad_output.device.type):
        return NotImplemented
    grad_bias = aten.sum(grad_output, [0] + list(range(2, grad_output.dim())))
    grad_inp, grad_weight, _ = aten.convolution_backward(
        grad_output,
        input,
        weight,
        bias_sizes,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
        [output_mask[0], output_mask[1], False],
    )
    return (grad_inp, grad_weight, grad_bias)

# 注册 aten.round.decimals 函数的分解实现
@register_decomposition([aten.round.decimals])
def round_dec(x, decimals=0):
    ten_pow_decimals = 10.0**decimals
    return aten.round(x * ten_pow_decimals) * (1.0 / ten_pow_decimals)

# 注册 aten.bmm 函数的分解实现，并在操作数学中进行强制类型转换
@register_decomposition([aten.bmm])
@pw_cast_for_opmath
def bmm(self, batch2):
    # 如果配置指定了使用坐标下降调优
    if config.coordinate_descent_tuning:
        # 如果 self 的第二维度为 1 或者 batch2 的第三维度为 1，则执行以下操作
        if guard_size_oblivious(self.shape[1] == 1) or guard_size_oblivious(
            batch2.shape[2] == 1
        ):
            # 将 self 和 batch2 进行维度扩展后相乘，并在第二维度上求和，得到 out
            out = (self.unsqueeze(-1) * batch2.unsqueeze(1)).sum(dim=2)
            return out
    
    # 如果 self 在 CPU 设备上
    if self.device.type == "cpu":
        # 如果 self 的第一维度为 1 且 batch2 的最后一维度为 1，则执行以下操作
        if guard_size_oblivious(self.size(1) == 1) and guard_size_oblivious(
            batch2.size(-1) == 1
        ):
            # 计数器记录操作次数
            counters["inductor"]["decompose_bmm"] += 1
            # 将 self 和 batch2 在指定维度上进行压缩后相乘，并在第一维度上求和，保持维度为 1
            return torch.sum(
                self.squeeze(1) * batch2.squeeze(-1), dim=1, keepdim=True
            ).unsqueeze(1)
    
    # 如果以上条件均不满足，返回 NotImplemented
    return NotImplemented
# 将该函数注册为 addmm 操作的分解函数，即对特定操作进行函数注册
# 并且标记为操作数学转换的 PW_CAST
@register_decomposition([aten.addmm])
@pw_cast_for_opmath
def addmm(self, mat1, mat2, beta=1, alpha=1):
    # 如果当前设备为 CPU
    if self.device.type == "cpu":
        # 如果 mat1 的第一维度大小为 1 并且 mat2 的最后一维度大小为 1
        if guard_size_oblivious(mat1.size(0) == 1) and guard_size_oblivious(
            mat2.size(-1) == 1
        ):
            # 增加对 decompose_addmm 计数器的计数
            counters["inductor"]["decompose_addmm"] += 1
            # 计算 mat1 去掉第一维度后的张量与 mat2 去掉最后一维度后的张量的乘积，按行求和并保持维度
            out = torch.sum(
                mat1.squeeze(0) * mat2.squeeze(-1), dim=0, keepdim=True
            ).unsqueeze(0)
            # 返回加权和 alpha*out + beta*self
            return alpha * out + beta * self
        # 如果 mat1 的第一维度大小为 1，并且 mat2 的尺寸小于等于 16x16
        if (
            guard_size_oblivious(mat1.size(0) == 1)
            and definitely_true(mat2.size(0) <= 16)
            and definitely_true(mat2.size(1) <= 16)
        ):
            # 增加对 decompose_addmm 计数器的计数
            counters["inductor"]["decompose_addmm"] += 1
            # 计算 mat1 的转置与 mat2 的乘积，按行求和并保持维度
            out = (mat1.T * mat2).sum(dim=0, keepdim=True)
            # 返回加权和 alpha*out + beta*self
            return alpha * out + beta * self
    # 如果不满足以上条件，则返回 NotImplemented
    return NotImplemented


# 将该函数注册为 mm 操作的分解函数，即对特定操作进行函数注册
# 并且标记为操作数学转换的 PW_CAST
@register_decomposition([aten.mm])
@pw_cast_for_opmath
def mm(self, input2):
    # 当启用 coordinate descent tuning 时，优化矩阵向量乘法以达到最大带宽
    if config.coordinate_descent_tuning:
        # 如果 self 的第一维度大小为 1 或者 input2 的第二维度大小为 1
        if guard_size_oblivious(self.shape[0] == 1) or guard_size_oblivious(
            input2.shape[1] == 1
        ):
            # 返回 self 和 input2 的乘积，按指定维度求和
            return (self.unsqueeze(2) * input2.unsqueeze(0)).sum(dim=1)
    # 如果当前设备为 CPU
    if self.device.type == "cpu":
        # 如果 self 的最后一维度大小为 1，并且 self 的第一维度大于 0
        # 并且 input2 的第一维度大小为 1，并且 self 和 input2 的数据类型相同
        # 并且总元素数小于等于 32
        if (
            guard_size_oblivious(self.size(-1) == 1)
            and guard_size_oblivious(self.size(0) > 0)
            and guard_size_oblivious(input2.size(0) == 1)
            and (self.dtype == input2.dtype)
            and definitely_true((torch.numel(self) + torch.numel(input2)) <= 32)
        ):
            # 增加对 decompose_mm 计数器的计数
            counters["inductor"]["decompose_mm"] += 1
            # 对 self 的每一行进行乘积，然后按行连接起来
            return torch.cat([self[i, :] * input2 for i in range(self.size(0))])
        # 如果 self 的第一维度大小为 1，并且 input2 的最后一维度大小为 1
        if guard_size_oblivious(self.size(0) == 1) and guard_size_oblivious(
            input2.size(-1) == 1
        ):
            # 增加对 decompose_mm 计数器的计数
            counters["inductor"]["decompose_mm"] += 1
            # 计算 self 去掉第一维度后的张量与 input2 去掉最后一维度后的张量的乘积，按行求和并保持维度
            return torch.sum(
                self.squeeze(0) * input2.squeeze(-1), dim=0, keepdim=True
            ).unsqueeze(0)
    # 如果不满足以上条件，则返回 NotImplemented
    return NotImplemented


# 这个函数的作用是对 cat 操作进行分解
# 该函数主要有两个作用：
# - 在只有一个张量输入时消除 cat 操作
# - 规范化 cat 调用，以删除遗留的空 1-D 张量（注意：我们不会删除所有空张量，只删除不当的）
@register_decomposition([aten.cat.default])
def cat(tensors, dim=0):
    from torch.fx.experimental.symbolic_shapes import guard_size_oblivious
    def non_empty_tensor(x):
        # 判断张量是否非空的函数
        # 如果张量的维度不为1或者第一个维度大于0，则返回True，否则返回False
        return len(x.shape) != 1 or guard_size_oblivious(x.shape[0] > 0)

    # 对张量列表进行过滤，只保留非空张量
    filtered_tensors = list(filter(non_empty_tensor, tensors))

    if len(filtered_tensors) == 1:
        # 如果过滤后的张量列表只有一个张量，则返回该张量的克隆
        return filtered_tensors[0].clone()
    elif 1 < len(filtered_tensors) < len(tensors):
        # 如果过滤后的张量列表个数大于1且小于原始张量列表个数，
        # 则递归调用默认的cat操作，并指定维度dim
        return aten.cat.default(filtered_tensors, dim)
    # 当没有进行任何过滤时，抛出NotImplemented以防止无限递归（不需要进一步分解）
    return NotImplemented
@register_decomposition([aten.angle])
def angle(x):
    # 如果输入张量 x 是复数类型
    if x.is_complex():
        # 返回 arctan2(y, x)，处理复数的角度计算
        return torch.where(
            torch.isnan(x.real), float("nan"), torch.atan2(x.imag, x.real)
        )

    # 当 x 是实数时
    _, dtype = elementwise_dtypes(
        x,
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
    )
    # 获取 pi 常数张量，与输入 x 的数据类型和设备相匹配
    pi = torch.scalar_tensor(math.pi, dtype=dtype, device=x.device)
    # 根据 x 的值，返回对应的角度
    ret = torch.where(x < 0, pi, 0.0)
    return torch.where(torch.isnan(x), float("nan"), ret)


@register_decomposition([aten.add])
def add(x, y, *, alpha=None):
    # 要求 x 和 y 都是复数张量
    x_is_complex_tensor = torch.is_tensor(x) and x.is_complex()
    y_is_complex_tensor = torch.is_tensor(y) and y.is_complex()
    if not x_is_complex_tensor or not y_is_complex_tensor:
        return NotImplemented
    z = y
    if alpha is not None:
        z = alpha * y
    # 推广 x 和 y 的数据类型为复数类型
    complex_type = torch.promote_types(x.dtype, y.dtype)

    # 对于复数类型的 x，使用 x.view(x.real.dtype) 可能会使最后一个维度翻倍，可能会导致广播加法问题
    def reshape_tensor_complex(tensor):
        """将张量从 [*initial_dims, last_dim] 重塑为 [*initial_dims, last_dim/2, 2]"""
        # 获取张量的当前形状
        *initial_dims, last_dim = tensor.shape

        # 检查最后一个维度是否为偶数。由于复数的处理方式，不应该出现奇数维度的情况。
        if last_dim % 2 != 0:
            raise AssertionError(
                "The size of the last dimension must be even to reshape it to [..., last_dim/2, 2]"
            )

        # 重塑张量
        new_shape = (*initial_dims, last_dim // 2, 2)
        reshaped_tensor = tensor.view(new_shape)
        return reshaped_tensor

    # 对输入张量 x 和 z 进行复数类型的重塑处理
    x_reshaped = reshape_tensor_complex(x.view(x.real.dtype))
    z_reshaped = reshape_tensor_complex(z.view(y.real.dtype))
    # 执行广播加法，并将结果展平，最后将结果的数据类型设为复数类型
    result = torch.flatten(x_reshaped + z_reshaped, start_dim=-2).view(complex_type)
    return result


@register_decomposition([aten.conj_physical])
def conj_physical(self):
    # 确保 self 不是复数类型张量，需要实现此功能
    assert not self.is_complex(), "TODO: implement this"
    return self


@register_decomposition([aten.lift, aten.detach_])
def lift(self):
    # 返回原始张量 self
    return self


@register_decomposition([aten.bernoulli.default])
def bernoulli(self, *, generator=None):
    # 断言 generator 参数为 None
    assert generator is None
    # 返回一个与 self 相同形状的布尔张量，元素值为按照 self 的概率生成的伯努利随机变量
    return (torch.rand_like(self, dtype=torch.float32) < self).to(self.dtype)


@register_decomposition([aten.fmin, prims.fmin])
def fmin(self, other):
    # 返回一个张量，其中每个元素为 self 和 other 中对应位置较小的值
    return torch.where(torch.isnan(other) | (other > self), self, other)


@register_decomposition([aten.fmax, prims.fmax])
def fmax(self, other):
    # 返回一个张量，其中每个元素为 self 和 other 中对应位置较大的值
    return torch.where(torch.isnan(other) | (other < self), self, other)


@register_decomposition(aten.amax)
def amax(self, dim=None, keepdim=False):
    # 此处是未完成的函数定义，需要实现对张量的最大值计算
    pass
    # 如果张量的数据类型是 torch.bool 类型
    if self.dtype == torch.bool:
        # 调用 torch.any 函数，计算张量在指定维度上的任意元素是否为 True，并保持维度
        return torch.any(self, dim=dim, keepdim=keepdim)
    
    # 如果张量的数据类型不是 torch.bool 类型，则返回 NotImplemented 表示不支持该操作
    return NotImplemented
# 注册函数，将 torch.amin 函数注册为当前函数的分解函数
@register_decomposition(aten.amin)
def amin(self, dim=None, keepdim=False):
    # 如果张量的数据类型为 torch.bool 类型，则调用 torch.all 函数对张量进行逻辑与操作
    if self.dtype == torch.bool:
        return torch.all(self, dim=dim, keepdim=keepdim)
    # 否则返回 Not Implemented
    return NotImplemented


# 注册函数，将 torch.narrow_copy 函数注册为当前函数的分解函数
@register_decomposition([aten.narrow_copy])
def narrow_copy(self, dim, start, length):
    # 调用 torch.narrow 函数进行张量的按维度缩小操作，然后进行克隆（复制）
    return torch.narrow(self, dim, start, length).clone()


# 注册函数，将 torch.expand_copy 函数注册为当前函数的分解函数
@register_decomposition([aten.expand_copy])
def expand_copy(self, size, *, implicit=False):
    # 调用 aten.expand 函数对张量进行扩展操作，然后进行克隆（复制）
    return aten.expand(self, size, implicit=implicit).clone()


# 注册函数，将 torch.view_copy.default 函数注册为当前函数的分解函数
@register_decomposition([aten.view_copy.default])
def view_copy_default(self, size):
    # 调用 aten.view 函数对张量进行视图变换操作，然后进行克隆（复制）
    return aten.view(self, size).clone()


# 注册函数，将 torch.view_copy.dtype 函数注册为当前函数的分解函数
@register_decomposition([aten.view_copy.dtype])
def view_copy_dtype(self, dtype):
    # 将张量转换为指定数据类型，然后进行克隆（复制）
    return self.to(dtype).clone()


# 定义函数 get_like_layout，用于获取与给定张量相似的内存格式
def get_like_layout(
    tensor: torch.Tensor, memory_format: Optional[torch.memory_format]
) -> torch.memory_format:
    # 如果内存格式是保留原格式或者为 None，则调用 utils.suggest_memory_format 函数建议内存格式
    if memory_format is torch.preserve_format or memory_format is None:
        return utils.suggest_memory_format(tensor)
    else:
        return memory_format


# 注册函数，将 torch.rand_like 函数注册为当前函数的分解函数
@register_decomposition(aten.rand_like)
def rand_like(self, *, dtype=None, device=None, memory_format=None, **kwargs):
    # 生成与当前张量形状相同的随机张量，并指定数据类型、设备和内存格式
    return torch.rand(
        [*self.size()],
        dtype=dtype or self.dtype,
        device=device or self.device,
        **kwargs,
    ).to(memory_format=get_like_layout(self, memory_format))


# 注册函数，将 torch.randn_like 函数注册为当前函数的分解函数
@register_decomposition(aten.randn_like)
def randn_like(self, *, dtype=None, device=None, memory_format=None, **kwargs):
    # 生成与当前张量形状相同的随机正态分布张量，并指定数据类型、设备和内存格式
    return torch.randn(
        [*self.size()],
        dtype=dtype or self.dtype,
        device=device or self.device,
        **kwargs,
    ).to(memory_format=get_like_layout(self, memory_format))


# 注册函数，将 torch.full_like 函数注册为当前函数的分解函数
@register_decomposition(aten.full_like)
def full_like(
    self,
    fill_value,
    *,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=False,
    requires_grad=False,
    memory_format=torch.preserve_format,
):
    # 生成与当前张量形状相同的指定填充值的张量，并指定数据类型、布局、设备等参数
    return torch.full(
        [*self.size()],
        fill_value,
        dtype=dtype or self.dtype,
        layout=layout or self.layout,
        device=device or self.device,
        requires_grad=requires_grad,
    ).to(memory_format=get_like_layout(self, memory_format))


# 注册函数，将 torch.randint_like.default 函数注册为当前函数的分解函数
@register_decomposition(aten.randint_like.default)
def randint_like(self, high, *, dtype=None, device=None, memory_format=None, **kwargs):
    # 生成与当前张量形状相同的指定范围内随机整数张量，并指定数据类型、设备和内存格式
    return aten.randint.low(
        0,
        high,
        [*self.size()],
        dtype=dtype or self.dtype,
        device=device or self.device,
        **kwargs,
    ).to(memory_format=get_like_layout(self, memory_format))


# 注册函数，将 torch.randint_like.low_dtype 函数注册为当前函数的分解函数
@register_decomposition(aten.randint_like.low_dtype)
def randint_like_low(
    self, low, high, *, dtype=None, device=None, memory_format=None, **kwargs
):
    # 生成与当前张量形状相同的指定范围内随机整数张量，并指定数据类型、设备和内存格式
    return aten.randint.low(
        low,
        high,
        [*self.size()],
        dtype=dtype or self.dtype,
        device=device or self.device,
        **kwargs,
    ).to(memory_format=get_like_layout(self, memory_format))
# 注册函数的装饰器，用于注册特定操作的分解函数
@register_decomposition(aten.randint.default)
def randint(high, size, **kwargs):
    # 调用底层函数 aten.randint.low，生成随机整数张量
    return aten.randint.low(0, high, size, **kwargs)


@register_decomposition(quantized.linear_dynamic_fp16_unpacked_weight.default)
def linear_dynamic_fp16_unpacked_weight(
    input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    # 使用 wrapped_fbgemm_pack_gemm_matrix_fp16 函数对权重张量进行打包
    packed_weight = torch.ops._quantized.wrapped_fbgemm_pack_gemm_matrix_fp16(weight)
    # 使用 wrapped_fbgemm_linear_fp16_weight 函数进行量化动态全连接操作
    return torch.ops._quantized.wrapped_fbgemm_linear_fp16_weight(
        input, packed_weight, bias, weight.size()[0]
    )


@register_decomposition(torch.ops.quantized.embedding_bag_byte_unpack)
def q_embedding_bag_byte_unpack_decomp(packed):
    def bitcast_u8_to_f32(u8):
        # 将输入的无符号8位整数张量转换为单精度浮点数张量
        x, y, z, w = (u8[..., n].to(torch.int32) for n in (0, 1, 2, 3))
        if sys.byteorder == "little":
            return (x + (y << 8) + (z << 16) + (w << 24)).view(torch.float32)[..., None]
        else:
            return ((x << 24) + (y << 16) + (z << 8) + w).view(torch.float32)[..., None]

    # 解包函数的实现，将 packed 张量解压为浮点数张量
    scales = bitcast_u8_to_f32(packed[..., -8:-4])
    offsets = bitcast_u8_to_f32(packed[..., -4:])
    return packed[..., :-8].to(torch.float32) * scales + offsets


@register_decomposition([aten.grid_sampler_2d])
@pw_cast_for_opmath
def grid_sampler_2d(
    a: torch.Tensor,
    grid: torch.Tensor,
    interpolation_mode: int = 0,
    padding_mode: int = 0,
    align_corners: bool = False,
) -> torch.Tensor:
    # 根据性能考虑，在 CPU 上不对网格进行扩展，以优化 bilinear 插值模式
    _expand_grid = not (
        a.device == torch.device("cpu")
        and interpolation_mode == 0
        and a.is_contiguous(memory_format=torch.contiguous_format)
    )

    # 调用 decomp_grid_sampler_2d 函数执行二维网格采样操作
    output = decomp_grid_sampler_2d(
        a,
        grid=grid,
        interpolation_mode=interpolation_mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
        _expand_grid=_expand_grid,
    )
    return output


@register_decomposition(aten._foreach_addcmul.Scalar)
def _foreach_addcmul_scalar(self, left_tensors, right_tensors, scalar=1):
    # 对每个张量进行 element-wise 的 addcmul 操作，带有标量因子
    return aten._foreach_add.List(
        self, aten._foreach_mul.List(left_tensors, right_tensors), alpha=scalar
    )


@register_decomposition(aten._foreach_addcdiv.Scalar)
def _foreach_addcdiv_scalar(self, left_tensors, right_tensors, scalar=1):
    # 对每个张量进行 element-wise 的 addcdiv 操作，带有标量因子
    return aten._foreach_add.List(
        self, aten._foreach_div.List(left_tensors, right_tensors), alpha=scalar
    )


@register_decomposition(aten._foreach_lerp.Scalar)
def _foreach_lerp_scalar(start_tensors, end_tensors, weight):
    # 对每个张量进行 element-wise 的 lerp 插值操作，带有权重因子
    # 返回一个列表对象，使用 _foreach_add.List 方法生成
    return aten._foreach_add.List(
        # 作为 _foreach_add.List 方法的第一个参数，传入 start_tensors 列表
        start_tensors,
        # 作为 _foreach_add.List 方法的第二个参数，传入一个标量值，使用 _foreach_mul.Scalar 方法生成
        aten._foreach_mul.Scalar(
            # _foreach_mul.Scalar 方法的参数是一个列表，使用 _foreach_sub.List 方法生成
            aten._foreach_sub.List(
                # _foreach_sub.List 方法的第一个参数是 end_tensors 列表
                end_tensors,
                # _foreach_sub.List 方法的第二个参数是 start_tensors 列表
                start_tensors
            ),
            # _foreach_mul.Scalar 方法的第二个参数是 weight 变量
            weight
        )
    )
# 定义 miopen_batch_norm 函数，用于实现 MiOpen 批量归一化的操作
@aten.miopen_batch_norm.default.py_impl(torch._C.DispatchKey.Autograd)
@register_decomposition(aten.miopen_batch_norm)
def miopen_batch_norm(
    input: torch.Tensor,  # 输入张量
    weight: torch.Tensor,  # 权重张量
    bias: typing.Optional[torch.Tensor],  # 可选的偏置张量
    running_mean: typing.Optional[torch.Tensor],  # 可选的运行均值张量
    running_var: typing.Optional[torch.Tensor],  # 可选的运行方差张量
    training: bool,  # 是否处于训练模式
    exponential_average_factor: float,  # 指数移动平均因子
    epsilon: float,  # 用于数值稳定性的小值
):
    # 调用 native_batch_norm 函数进行批量归一化
    a, b, c = aten.native_batch_norm(
        input,
        weight,
        bias,
        running_mean,
        running_var,
        training,
        exponential_average_factor,
        epsilon,
    )

    # 如果处于训练模式，返回批量归一化的结果 (a, b, c)
    if training:
        return (a, b, c)
    # 否则返回 a，并使用 weight 创建形状为 (0,) 的全零张量作为 b 和 c
    return (
        a,
        weight.new_zeros((0,)),
        weight.new_zeros((0,)),
    )


# 使用 functools.lru_cache 进行缓存的 fast_random_decomps 函数
@functools.lru_cache(None)
def fast_random_decomps():
    return {**decompositions, **extra_random_decomps}


# 根据配置选择合适的分解表格的函数 select_decomp_table
def select_decomp_table():
    """decomps can change based on config"""
    # 如果配置指定使用随机分解表，则返回 fast_random_decomps 函数的结果
    if config.fallback_random:
        return decompositions
    # 否则返回默认的 decompositions 分解表
    return fast_random_decomps()


# 注册对 aten.masked_scatter 操作进行分解的函数
@register_decomposition(aten.masked_scatter)
def masked_scatter(self, mask, source):
    from .codegen.common import BackendFeature, has_backend_feature

    # 如果设备支持带索引的掩码散射操作，则执行以下代码块
    if has_backend_feature(self.device, BackendFeature.MASKED_SCATTER_WITH_INDEX):
        # 将 self 和 mask 进行广播以匹配形状
        self, mask = aten.broadcast_tensors([self, mask])
        # 计算 mask 的扁平版本的累积索引
        source_idx = mask.reshape(-1).cumsum(0) - 1
        # 将 self、mask 和 source 扁平化
        self_flat, mask_flat, source_flat = (x.flatten() for x in (self, mask, source))
        # 使用 _unsafe_masked_index 函数执行不安全的带索引掩码操作
        result = aten._unsafe_masked_index(source_flat, mask_flat, [source_idx], 0)
        # 根据 mask 选择结果或者保持 self 的值并重塑成 self 的形状
        return torch.where(mask_flat, result, self_flat).view(self.shape)
    # 如果设备不支持，则返回 Not Implemented
    return NotImplemented


# 注册对 quantized_decomposed.choose_qparams.tensor 操作进行分解的函数
@register_decomposition(quantized_decomposed.choose_qparams.tensor)
def choose_qparams_tensor(
    input: torch.Tensor,  # 输入张量
    quant_min: int,  # 量化的最小值
    quant_max: int,  # 量化的最大值
    eps: float,  # 用于数值稳定性的小值
    dtype: torch.dtype  # 输出张量的数据类型
):
    # 计算输入张量的最小值和最大值
    min_val, max_val = torch.aminmax(input)
    # 计算缩放比例
    scale = (max_val - min_val) / float(quant_max - quant_min)
    # 将 scale 与 eps 比较，选择较大值
    scale = torch.max(scale, torch.Tensor([eps]))
    # 计算零点值
    zero_point = quant_min - torch.round(min_val / scale).to(torch.int)
    # 将零点值限制在 quant_min 和 quant_max 之间
    zero_point = torch.clamp(zero_point, quant_min, quant_max)
    # 返回缩放比例和零点值的浮点数和整数版本
    return scale.to(torch.float64), zero_point.to(torch.int64)


# 注册对 aten.put 操作进行分解的函数
@register_decomposition(aten.put)
def put(self, index, source, accumulate=False):
    # 将 self 张量扁平化
    flattened = self.flatten()
    # 使用 index 将 source 放入 flattened 中
    flattened = torch.index_put(
        flattened, [index], source.reshape(index.shape), accumulate
    )
    # 将扁平化后的张量重新塑形成原始形状并返回
    return flattened.reshape(self.shape)


# 注册对 aten.put_ 操作进行分解的函数
@register_decomposition(aten.put_)
def put_(self, index, source, accumulate=False):
    # 调用 aten.put 函数执行操作，并将结果赋值给 out
    out = aten.put(self, index, source, accumulate=accumulate)
    # 将操作结果复制到 self 上并返回 self
    return self.copy_(out)


# 注册对 aten._softmax_backward_data.default 操作进行分解的函数
@pw_cast_for_opmath
@register_decomposition(aten._softmax_backward_data.default)
def _softmax_backward_data(grad_output, output, dim, input_dtype):
    # 计算新的梯度输出，即 grad_output 乘以 output
    new_grad_output = grad_output * output
    # 返回新的梯度输出作为结果
    return new_grad_output
    # 计算 new_grad_output 按指定维度 dim 求和，保持维度不变
    sum_new_grad = torch.sum(new_grad_output, dim=dim, keepdim=True)
    
    # 计算梯度输入 grad_input，使用 fma 函数进行操作，等价于 new_grad_output - output * sum_new_grad
    grad_input = inductor_prims.fma(-output, sum_new_grad, new_grad_output)

    # 如果梯度输出的数据类型与输入数据的数据类型不同，则将 grad_input 转换为输入数据类型
    if grad_output.dtype != input_dtype:
        grad_input = grad_input.to(input_dtype)
    
    # 返回一个连续的（contiguous）版本的 grad_input
    return grad_input.contiguous()
# 注册函数 `index_reduce` 到 `aten.index_reduce` 的分解器
@register_decomposition(aten.index_reduce)
def index_reduce(
    self, dim: int, index, src, reduction_type: str, *, include_self: bool = True
):
    # 如果 reduction_type 是 "mean" 并且不需要因为原子加限制而回退
    if reduction_type == "mean" and not needs_fallback_due_to_atomic_add_limitations(
        self.dtype
    ):
        # 真除法条件，根据数据类型判断是否浮点数或复数
        true_division = self.dtype.is_floating_point or self.dtype.is_complex
        # 创建与 src 相同形状的全为1的张量 ones
        ones = torch.ones_like(src)
        if include_self:
            # 如果 include_self 为真，则 out 是 self，并且计数是在指定维度上索引添加 ones 后的结果
            out = self
            counts = torch.ones_like(self).index_add(dim, index, ones)
        else:
            # 如果 include_self 不为真，则 out 是在指定维度上用 0 填充索引后的结果，并且计数是在指定维度上索引添加 ones 后的结果
            out = self.index_fill(dim, index, 0)
            counts = torch.zeros_like(self).index_add(dim, index, ones)
            # 对计数小于1的元素进行掩码替换为1
            counts = counts.masked_fill(counts < 1, 1)
        # 在指定维度上将 src 添加到 out 上
        out = out.index_add(dim, index, src)
        # 如果是真除法则返回 out 除以 counts，否则返回 out 整除 counts
        return out / counts if true_division else out // counts

    # 如果需要使用 scatter 回退
    if use_scatter_fallback(
        aten.scatter_reduce_.two,
        reduction_type,
        self.dtype,
        src.dtype,
        src.device.type,
        True,
    ):
        # 返回未实现
        return NotImplemented

    # 计算重复次数，用于 scatter 操作
    repeats = self.shape[dim + 1 :].numel() * self.shape[:dim].numel()
    # 索引形状计算
    index_shape = (index.numel(), *self.shape[dim + 1 :], *self.shape[:dim])
    # 置换维度顺序
    perm = (*range(self.ndim - dim, self.ndim), 0, *range(1, self.ndim - dim))
    # 创建 scatter_index 张量，重复索引并调整形状和维度顺序
    scatter_index = (
        index.to(torch.int64)
        .repeat_interleave(repeats)
        .reshape(index_shape)
        .permute(perm)
    )
    # 返回 scatter_reduce 操作的结果
    return self.scatter_reduce(
        dim,
        scatter_index,
        src,
        reduction_type,
        include_self=include_self,
    )


# 注册函数 `max_pool2d_with_indices` 到 `aten.max_pool2d_with_indices` 的分解器
@register_decomposition(aten.max_pool2d_with_indices)
def max_pool2d_with_indices(
    x, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False
):
    # 如果 dilation 是 1，则设为 [1, 1]
    if dilation == 1:
        dilation = [1, 1]

    # 如果 padding 是 0，则设为 [0, 0]
    if padding == 0:
        padding = [0, 0]

    # 如果 stride 未指定，则设为 kernel_size
    if not stride:
        stride = kernel_size

    # 统一将 kernel_size、dilation、padding、stride 转为长度为 2 的列表
    kernel_size = pad_listlike(kernel_size, 2)
    dilation = pad_listlike(dilation, 2)
    padding = pad_listlike(padding, 2)
    stride = pad_listlike(stride, 2)

    # 计算窗口大小
    window_size = kernel_size[0] * kernel_size[1]
    # 当使用非默认 dilation 或者窗口大小过大时，回退到传统方法
    if (
        torch._inductor.lowering.should_fallback_max_pool2d_with_indices(
            kernel_size, dilation
        )
        or window_size > torch.iinfo(torch.int8).max
    ):
        # 返回未实现
        return NotImplemented

    # 使用低内存模式计算最大池化的值和偏移
    vals, offsets = prims._low_memory_max_pool2d_with_offsets(
        x,
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
    )
    # 使用偏移计算出最大池化的索引
    indices = prims._low_memory_max_pool2d_offsets_to_indices(
        offsets,
        kernel_size[1],
        x.size(-1),
        stride,
        padding,
    )
    # 返回最大池化的值和索引
    return vals, indices
```