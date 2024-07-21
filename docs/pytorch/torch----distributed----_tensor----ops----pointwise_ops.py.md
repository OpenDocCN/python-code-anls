# `.\pytorch\torch\distributed\_tensor\ops\pointwise_ops.py`

```py
# Copyright (c) Meta Platforms, Inc. and affiliates
from typing import List, Sequence, Tuple

import torch
from torch.distributed._tensor._op_schema import (
    _is_inplace_op,
    _is_out_variant_op,
    OpSchema,
    OpStrategy,
    PlacementStrategy,
    RuntimeSchemaInfo,
    StrategyType,
    TupleStrategy,
)
from torch.distributed._tensor.ops.utils import (
    generate_redistribute_costs,
    infer_broadcast_dims_map,
    map_placements_after_broadcast,
    normalize_dim,
    register_op_strategy,
)
from torch.distributed._tensor.placement_types import (
    DTensorSpec,
    Partial,
    Placement,
    Replicate,
    Shard,
)
from torch.distributed.device_mesh import DeviceMesh

# 使用 torch 模块下的 aten 命名空间的操作符
aten = torch.ops.aten

# leave the remaining pointwise_ops list here for convenience,
# Below ops are some pointwise ops that are yet to be supported,
# they might not be a complete list.
# pointwise_ops = [
#     "fake_quantize_per_channel_affine",
#     "fake_quantize_per_tensor_affine",
#     "floor_divide",  # floor_divide is deprecated
#     "frexp",  # multiple output pointwise op, need to add support
#     "gradient",  #  need investigation on this op
#     "imag",  # complex data type only
#     "quantized_batch_norm",
#     "quantized_max_pool1d",
#     "quantized_max_pool2d",
#     "real",  # complex data type only
# ]

# 包含线性操作的 pointwise_ops 列表
linear_pointwise_ops = [
    aten.div.Scalar,  # this op is linear on the first argument, and the second argument is scalar, so it fits as a linear op.
    aten.div_.Scalar,  # this op is linear on the first argument, and the second argument is scalar, so it fits as a linear op.
    aten.to.dtype,
    aten.add.Tensor,
    aten.add_.Tensor,
]

# 包含各种点对点操作的 pointwise_ops 列表
pointwise_ops = [
    # please keep the entries below alphabetically sorted
    aten._conj.default,
    aten.abs.default,
    aten.abs.out,
    aten.abs_.default,
    aten.acos.default,
    aten.acos.out,
    aten.acos_.default,
    aten.acosh.default,
    aten.acosh.out,
    aten.acosh_.default,
    aten.add.Scalar,
    aten.add.out,
    aten.add_.Scalar,
    aten.addcdiv.default,
    aten.addcdiv.out,
    aten.addcdiv_.default,
    aten.addcmul.default,
    aten.addcmul.out,
    aten.addcmul_.default,
    aten.angle.default,
    aten.angle.out,
    aten.asin.default,
    aten.asin.out,
    aten.asin_.default,
    aten.asinh.default,
    aten.asinh.out,
    aten.asinh_.default,
    aten.atan.default,
    aten.atan.out,
    aten.atan2.default,
    aten.atan2.out,
    aten.atan2_.default,
    aten.atan_.default,
    aten.atanh.default,
    aten.atanh.out,
    aten.atanh_.default,
    aten.bitwise_and.Scalar,
    aten.bitwise_and.Scalar_Tensor,
    aten.bitwise_and.Scalar_out,
    aten.bitwise_and.Tensor,
    aten.bitwise_and.Tensor_out,
    aten.bitwise_and_.Scalar,
    aten.bitwise_and_.Tensor,
    aten.bitwise_left_shift.Scalar_Tensor,
    aten.bitwise_left_shift.Tensor,
    aten.bitwise_left_shift.Tensor_Scalar,
    aten.bitwise_left_shift.Tensor_Scalar_out,
]
    aten.bitwise_left_shift.Tensor_out,  # torch.Tensor 的按位左移操作，将结果输出到给定的张量
    aten.bitwise_left_shift_.Tensor,     # torch.Tensor 的原地按位左移操作
    aten.bitwise_left_shift_.Tensor_Scalar,  # torch.Tensor 的原地按位左移操作，与标量进行位运算
    aten.bitwise_not.default,            # torch.Tensor 的按位取反操作
    aten.bitwise_not.out,                # torch.Tensor 的按位取反操作，并将结果输出到给定的张量
    aten.bitwise_not_.default,           # torch.Tensor 的原地按位取反操作
    aten.bitwise_or.Scalar,              # torch.Tensor 的按位或操作，与标量进行位运算
    aten.bitwise_or.Scalar_Tensor,       # torch.Tensor 与标量的按位或操作
    aten.bitwise_or.Scalar_out,          # torch.Tensor 与标量的按位或操作，并将结果输出到给定的张量
    aten.bitwise_or.Tensor,              # torch.Tensor 的按位或操作
    aten.bitwise_or.Tensor_out,          # torch.Tensor 的按位或操作，并将结果输出到给定的张量
    aten.bitwise_or_.Scalar,             # torch.Tensor 的原地按位或操作，与标量进行位运算
    aten.bitwise_or_.Tensor,             # torch.Tensor 的原地按位或操作
    aten.bitwise_right_shift.Scalar_Tensor,  # torch.Tensor 的按位右移操作，与标量进行位运算
    aten.bitwise_right_shift.Tensor,     # torch.Tensor 的按位右移操作
    aten.bitwise_right_shift.Tensor_Scalar,  # torch.Tensor 的按位右移操作，与标量进行位运算
    aten.bitwise_right_shift.Tensor_Scalar_out,  # torch.Tensor 的按位右移操作，并将结果输出到给定的张量
    aten.bitwise_right_shift.Tensor_out, # torch.Tensor 的按位右移操作，并将结果输出到给定的张量
    aten.bitwise_right_shift_.Tensor,    # torch.Tensor 的原地按位右移操作
    aten.bitwise_right_shift_.Tensor_Scalar,  # torch.Tensor 的原地按位右移操作，与标量进行位运算
    aten.bitwise_xor.Scalar,             # torch.Tensor 的按位异或操作，与标量进行位运算
    aten.bitwise_xor.Scalar_Tensor,      # torch.Tensor 与标量的按位异或操作
    aten.bitwise_xor.Scalar_out,         # torch.Tensor 与标量的按位异或操作，并将结果输出到给定的张量
    aten.bitwise_xor.Tensor,             # torch.Tensor 的按位异或操作
    aten.bitwise_xor.Tensor_out,         # torch.Tensor 的按位异或操作，并将结果输出到给定的张量
    aten.bitwise_xor_.Scalar,            # torch.Tensor 的原地按位异或操作，与标量进行位运算
    aten.bitwise_xor_.Tensor,            # torch.Tensor 的原地按位异或操作
    aten.ceil.default,                   # torch.Tensor 的向上取整操作
    aten.ceil.out,                       # torch.Tensor 的向上取整操作，并将结果输出到给定的张量
    aten.ceil_.default,                  # torch.Tensor 的原地向上取整操作
    aten.clamp.default,                  # torch.Tensor 的值夹紧操作，限制在指定范围内
    aten.clamp.out,                      # torch.Tensor 的值夹紧操作，并将结果输出到给定的张量
    aten.clamp_.default,                 # torch.Tensor 的原地值夹紧操作
    aten.clip.default,                   # torch.Tensor 的剪裁操作，限制在指定范围内
    aten.clip.out,                       # torch.Tensor 的剪裁操作，并将结果输出到给定的张量
    aten.clip_.default,                  # torch.Tensor 的原地剪裁操作
    aten.conj_physical.default,          # torch.Tensor 的物理共轭操作
    aten.conj_physical.out,              # torch.Tensor 的物理共轭操作，并将结果输出到给定的张量
    aten.conj_physical_.default,         # torch.Tensor 的原地物理共轭操作
    aten.copysign.Scalar,                # torch.Tensor 的符号复制操作，与标量进行运算
    aten.copysign.Scalar_out,            # torch.Tensor 的符号复制操作，并将结果输出到给定的张量
    aten.copysign.Tensor,                # torch.Tensor 的符号复制操作
    aten.copysign.out,                   # torch.Tensor 的符号复制操作，并将结果输出到给定的张量
    aten.copysign_.Scalar,               # torch.Tensor 的原地符号复制操作，与标量进行运算
    aten.copysign_.Tensor,               # torch.Tensor 的原地符号复制操作
    aten.cos.default,                    # torch.Tensor 的余弦函数操作
    aten.cos.out,                        # torch.Tensor 的余弦函数操作，并将结果输出到给定的张量
    aten.cos_.default,                   # torch.Tensor 的原地余弦函数操作
    aten.cosh.default,                   # torch.Tensor 的双曲余弦函数操作
    aten.cosh.out,                       # torch.Tensor 的双曲余弦函数操作，并将结果输出到给定的张量
    aten.cosh_.default,                  # torch.Tensor 的原地双曲余弦函数操作
    aten.deg2rad.default,                # 角度到弧度的转换操作
    aten.deg2rad.out,                    # 角度到弧度的转换操作，并将结果输出到给定的张量
    aten.deg2rad_.default,               # 原地角度到弧度的转换操作
    aten.digamma.default,                # torch.Tensor 的Ψ（digamma）函数操作
    aten.digamma.out,                    # torch.Tensor 的Ψ（digamma）函数操作，并将结果输出到给定的张量
    aten.digamma_.default,               # torch.Tensor 的原地Ψ（digamma）函数操作
    aten.div.Tensor,                     # torch.Tensor 的除法操作
    aten.div.Tensor_mode,                # torch.Tensor 的除法操作，指定计算模式
    aten.div.out,                        # torch.Tensor 的除法操作，并将结果输出到给定的张量
    aten.div.out_mode,                   # torch.Tensor 的除法操作，指定计算模式，并将结果输出到给定的张量
    aten.div_.Tensor,                    # torch.Tensor 的原地除法操作
    aten.div_.Tensor_mode,               # torch.Tensor 的原地除法操作，指定计算模式
    aten.eq.Tensor,                      # torch.Tensor 的相等比较操作
    aten.eq.Tensor_out,                  # torch.Tensor 的相等比较操作，并将结果输出到给定的张量
    aten.eq.Scalar,                      # torch.Tensor 与标量的相等比较操作
    aten.eq.Scalar_out,                  # torch.Tensor 与标量的相等比较操作，并将结果输出到给定的张量
    aten.erf.default,                    # torch.Tensor 的误差函数操作
    aten.erf.out,                        # torch.Tensor 的误差函数操作，并将结果输出到给定的张量
    aten.erf_.default,                   # torch.Tensor 的原地误差函数操作
    aten.erfc.default,                   # torch.Tensor 的互补误差函数操作
    aten.erfc.out,                       # torch.Tensor 的互补误差函数操作，并将结果输出到给定的张量
    aten.erfc_.default,                  # torch.Tensor 的原地互补误差函数操作
    aten.erfinv.default,                 # torch.Tensor 的逆误差函数操作
    aten.erfinv.out,                     # torch.Tensor 的逆误差函数操作，并将结果输出到给定的张量
    aten.erfinv_.default,                # torch.Tensor 的原地逆误差函数操作
    aten.exp.default,                    # torch.Tensor 的指数函数操作
    aten.exp.out,                        # torch.Tensor 的指数函数操作，并将结果输出到给定的张量
    aten.exp2.default,                   # torch.Tensor 的2的指数次幂函数操作
    aten.exp2.out,                       # torch.Tensor 的2的指数次幂函数操作，并将结果输出到
    # 下面是一系列 PyTorch 中的张量运算函数，按名称列出
    
    aten.gt.Scalar,  # 检查张量中每个元素是否大于标量
    aten.gt.Tensor,  # 检查张量中每对元素是否满足大于关系
    aten.hypot.default,  # 计算两个张量的直角三角形的斜边长度（元素级）
    aten.hypot.out,  # 计算两个张量的直角三角形的斜边长度（输出到指定张量）
    aten.hypot_.default,  # 计算两个张量的直角三角形的斜边长度（原地操作）
    aten.i0.default,  # 计算修正的零阶贝塞尔函数
    aten.i0.out,  # 计算修正的零阶贝塞尔函数（输出到指定张量）
    aten.i0_.default,  # 计算修正的零阶贝塞尔函数（原地操作）
    aten.igamma.default,  # 计算不完全伽玛函数
    aten.igamma.out,  # 计算不完全伽玛函数（输出到指定张量）
    aten.igamma_.default,  # 计算不完全伽玛函数（原地操作）
    aten.igammac.default,  # 计算归一化不完全伽玛函数
    aten.igammac.out,  # 计算归一化不完全伽玛函数（输出到指定张量）
    aten.igammac_.default,  # 计算归一化不完全伽玛函数（原地操作）
    aten.isnan.default,  # 检查张量中每个元素是否为NaN
    aten.ldexp.default,  # 计算 2 的整数次幂乘以浮点数
    aten.ldexp.out,  # 计算 2 的整数次幂乘以浮点数（输出到指定张量）
    aten.ldexp_.default,  # 计算 2 的整数次幂乘以浮点数（原地操作）
    aten.lt.Tensor,  # 检查张量中每对元素是否满足小于关系
    aten.lt.Tensor_out,  # 检查张量中每对元素是否满足小于关系（输出到指定张量）
    aten.lt.Scalar,  # 检查张量中每个元素是否小于标量
    aten.lt.Scalar_out,  # 检查张量中每个元素是否小于标量（输出到指定张量）
    aten.le.Scalar,  # 检查张量中每个元素是否小于等于标量
    aten.le.Tensor,  # 检查张量中每对元素是否满足小于等于关系
    aten.lerp.Scalar,  # 执行线性插值（标量版本）
    aten.lerp.Scalar_out,  # 执行线性插值（标量版本，输出到指定张量）
    aten.lerp.Tensor,  # 执行线性插值（张量版本）
    aten.lerp.Tensor_out,  # 执行线性插值（张量版本，输出到指定张量）
    aten.lerp_.Scalar,  # 执行线性插值（原地操作，标量版本）
    aten.lerp_.Tensor,  # 执行线性插值（原地操作，张量版本）
    aten.lgamma.default,  # 计算对数伽玛函数
    aten.lgamma.out,  # 计算对数伽玛函数（输出到指定张量）
    aten.lgamma_.default,  # 计算对数伽玛函数（原地操作）
    aten.log.default,  # 计算自然对数
    aten.log.out,  # 计算自然对数（输出到指定张量）
    aten.log10.default,  # 计算以 10 为底的对数
    aten.log10.out,  # 计算以 10 为底的对数（输出到指定张量）
    aten.log10_.default,  # 计算以 10 为底的对数（原地操作）
    aten.log1p.default,  # 计算 log(1 + x)
    aten.log1p.out,  # 计算 log(1 + x)（输出到指定张量）
    aten.log1p_.default,  # 计算 log(1 + x)（原地操作）
    aten.log2.default,  # 计算以 2 为底的对数
    aten.log2.out,  # 计算以 2 为底的对数（输出到指定张量）
    aten.log2_.default,  # 计算以 2 为底的对数（原地操作）
    aten.log_.default,  # 计算对数（默认底数）
    aten.logaddexp.default,  # 计算 log(exp(x) + exp(y))
    aten.logaddexp.out,  # 计算 log(exp(x) + exp(y))（输出到指定张量）
    aten.logaddexp2.default,  # 计算 log2(2^x + 2^y)
    aten.logaddexp2.out,  # 计算 log2(2^x + 2^y)（输出到指定张量）
    aten.logical_and.default,  # 执行逻辑与操作
    aten.logical_and.out,  # 执行逻辑与操作（输出到指定张量）
    aten.logical_and_.default,  # 执行逻辑与操作（原地操作）
    aten.logical_not.default,  # 执行逻辑非操作
    aten.logical_not.out,  # 执行逻辑非操作（输出到指定张量）
    aten.logical_not_.default,  # 执行逻辑非操作（原地操作）
    aten.logical_or.default,  # 执行逻辑或操作
    aten.logical_or.out,  # 执行逻辑或操作（输出到指定张量）
    aten.logical_or_.default,  # 执行逻辑或操作（原地操作）
    aten.logical_xor.default,  # 执行逻辑异或操作
    aten.logical_xor.out,  # 执行逻辑异或操作（输出到指定张量）
    aten.logical_xor_.default,  # 执行逻辑异或操作（原地操作）
    aten.logit.default,  # 计算 logit 函数
    aten.logit.out,  # 计算 logit 函数（输出到指定张量）
    aten.logit_.default,  # 计算 logit 函数（原地操作）
    aten.masked_fill.Scalar,  # 使用标量值填充掩码位置
    aten.maximum.out,  # 求张量中每对元素的最大值（输出到指定张量）
    aten.mul.Scalar,  # 对张量每个元素乘以标量
    aten.mul.Tensor,  # 对两个张量逐元素相乘
    aten.mul.out,  # 对两个张量逐元素相乘（输出到指定张量）
    aten.mul_.Scalar,  # 对张量每个元素乘以标量（原地操作）
    aten.mul_.Tensor,  # 对两个张量逐元素相乘（原地操作）
    aten.mvlgamma.default,  # 计算多维伽玛函数的对数
    aten.mvlgamma.out,  # 计算多维伽玛函数的对数（输出到指定张量）
    aten.mvlgamma_.default,  # 计算多维伽玛函数的对数（原地操作）
    aten.native_dropout_backward.default,  # 计算原生 dropout 的反向传播
    aten.native_dropout_backward.out,  # 计算原生 dropout 的反向传播（输出到指定张量）
    aten.nan_to_num.default,  # 将 NaN 替换为指定值
    aten.nan_to_num.out,  # 将 NaN 替换为指定值（输出到指定张量）
    aten.nan_to_num_.default,  # 将 NaN 替换为指定值（原地操作）
    aten.ne.Scalar,  # 检查张量中每个元素是否不等于标量
    aten.neg.default,  # 对张量每个元素取负
    aten.neg.out,  # 对张量每个元素取负（输出到指定张量）
    aten.neg_.default,  # 对张量每个元素取负（原地操作）
    aten.nextafter.default,  # 计算浮点数的下一个相邻值
    aten.nextafter.out,
    # 下面是一系列的张量操作函数，这些函数通常用于深度学习框架中，如PyTorch或TensorFlow，用于张量的数学运算和梯度计算。
    
    aten.rsqrt.default,                 # 计算张量的倒数的平方根
    aten.rsqrt.out,                     # 将计算结果存储到指定张量中
    aten.rsqrt_.default,                # 原地计算张量的倒数的平方根
    aten.rsub.Scalar,                   # 用标量减去张量的右侧元素
    aten.sgn.default,                   # 计算张量中元素的符号
    aten.sgn.out,                       # 将计算结果存储到指定张量中
    aten.sgn_.default,                  # 原地计算张量中元素的符号
    aten.sigmoid.default,               # 计算张量的sigmoid函数
    aten.sigmoid.out,                   # 将计算结果存储到指定张量中
    aten.sigmoid_.default,              # 原地计算张量的sigmoid函数
    aten.sign.default,                  # 计算张量的符号函数
    aten.sign.out,                      # 将计算结果存储到指定张量中
    aten.sign_.default,                 # 原地计算张量的符号函数
    aten.signbit.default,               # 计算张量中每个元素的符号位
    aten.signbit.out,                   # 将计算结果存储到指定张量中
    aten.silu.default,                  # 计算张量的SiLU（Sigmoid-Weighted Linear Unit）函数
    aten.silu.out,                      # 将计算结果存储到指定张量中
    aten.sin.default,                   # 计算张量的正弦函数
    aten.sin.out,                       # 将计算结果存储到指定张量中
    aten.sin_.default,                  # 原地计算张量的正弦函数
    aten.sinc.default,                  # 计算张量的sinc函数
    aten.sinc.out,                      # 将计算结果存储到指定张量中
    aten.sinc_.default,                 # 原地计算张量的sinc函数
    aten.sinh.default,                  # 计算张量的双曲正弦函数
    aten.sinh.out,                      # 将计算结果存储到指定张量中
    aten.sinh_.default,                 # 原地计算张量的双曲正弦函数
    aten.sqrt.default,                  # 计算张量的平方根
    aten.sqrt.out,                      # 将计算结果存储到指定张量中
    aten.sqrt_.default,                 # 原地计算张量的平方根
    aten.square.default,                # 计算张量的平方
    aten.square.out,                    # 将计算结果存储到指定张量中
    aten.square_.default,               # 原地计算张量的平方
    aten.sub.Scalar,                    # 用标量减去张量的每个元素
    aten.sub.Tensor,                    # 张量之间的减法操作
    aten.sub.out,                       # 将计算结果存储到指定张量中
    aten.sub_.Scalar,                   # 原地用标量减去张量的每个元素
    aten.sub_.Tensor,                   # 原地计算张量之间的减法操作
    aten.tan.default,                   # 计算张量的正切函数
    aten.tan.out,                       # 将计算结果存储到指定张量中
    aten.tan_.default,                  # 原地计算张量的正切函数
    aten.tanh.default,                  # 计算张量的双曲正切函数
    aten.tanh.out,                      # 将计算结果存储到指定张量中
    aten.tanh_.default,                 # 原地计算张量的双曲正切函数
    aten.true_divide.Tensor,            # 张量之间的真除操作
    aten.trunc.default,                 # 计算张量的截断值
    aten.trunc.out,                     # 将计算结果存储到指定张量中
    aten.trunc_.default,                # 原地计算张量的截断值
    aten.where.self,                    # 返回符合条件的张量元素
    aten.where.self_out,                # 将符合条件的张量元素存储到指定张量中
    aten.xlogy.OutScalar_Self,          # 计算自然对数乘以另一标量的张量
    aten.xlogy.OutScalar_Other,         # 计算自然对数乘以另一标量的张量
    aten.xlogy.OutTensor,               # 计算自然对数乘以另一张量的张量
    aten.xlogy.Scalar_Other,            # 计算自然对数乘以另一标量的张量
    aten.xlogy.Scalar_Self,             # 计算自然对数乘以另一标量的张量
    aten.xlogy.Tensor,                  # 计算自然对数乘以另一张量的张量
    aten.xlogy_.Scalar_Other,           # 原地计算自然对数乘以另一标量的张量
    aten.xlogy_.Tensor,                 # 原地计算自然对数乘以另一张量的张量
    
    # 反向传播的逐点操作
    # 请保持以下条目按字母顺序排序
    aten.gelu_backward.default,         # GELU函数的反向传播
    aten.sigmoid_backward.default,      # sigmoid函数的反向传播
    aten.silu_backward.default,         # SiLU函数的反向传播
    aten.tanh_backward.default,         # 双曲正切函数的反向传播
    aten.threshold_backward.default,    # 阈值函数的反向传播
# 点对点操作的策略选择函数，根据操作模式和参数策略选择最优操作策略
def pointwise_strategy(
    mesh: DeviceMesh, op_schema: OpSchema, linearity: bool = False
) -> OpStrategy:
    # 初始化最大分片策略的索引和分片数
    max_shards_strategy_index = -1
    max_shards = -1

    # 如果是原地操作，应该遵循第一个参数的策略
    if _is_inplace_op(op_schema.op):
        followed_strategy = op_schema.args_schema[0]
    # 如果是输出变体操作，应该遵循输出参数的策略
    elif _is_out_variant_op(op_schema.op):
        followed_strategy = op_schema.kwargs_schema["out"]
    else:
        # 对于普通的点对点操作，选择遵循包含最大分片数的参数策略，以便进行重新分片
        for idx, arg_strategy in enumerate(op_schema.args_schema):
            if not isinstance(arg_strategy, OpStrategy):
                continue

            arg_max_shards = arg_strategy.max_num_shards()
            if arg_max_shards > max_shards:
                max_shards_strategy_index = idx
                max_shards = arg_max_shards

        followed_strategy = op_schema.args_schema[max_shards_strategy_index]

    # 确保选择的策略是 OpStrategy 类型
    assert isinstance(
        followed_strategy, OpStrategy
    ), f"no strategy to follow for {op_schema}!"
    
    # 调用通用的点对点操作策略函数，返回最终选择的操作策略
    return common_pointwise_strategy(
        mesh, op_schema.args_schema, followed_strategy, linearity
    )


# 通用的点对点操作策略函数，处理广播和返回操作策略
def common_pointwise_strategy(
    mesh: DeviceMesh,
    args_schema: Sequence[object],
    followed_strategy: OpStrategy,
    linearity: bool,
) -> OpStrategy:
    # 处理广播，确定参数的公共形状
    common_shape = torch.broadcast_shapes(
        *[arg.shape for arg in args_schema if isinstance(arg, OpStrategy)]
    )
    # 创建并返回点对点操作的策略对象
    pointwise_strategy = OpStrategy([])
    # 遍历 followed_strategy 中的每个放置策略
    for placement_strategy in followed_strategy.strategies:
        # 获取当前放置策略的输出规格
        spec_to_follow = placement_strategy.output_spec
        # 初始化一个空列表，用于存放输出的放置位置
        out_placements: List[Placement] = []

        # 遍历当前放置策略中的每个放置
        for placement in spec_to_follow.placements:
            if isinstance(placement, Shard):
                # 如果是 Shard 放置，标准化其维度并计算新的维度
                shard_dim = normalize_dim(placement.dim, len(spec_to_follow.shape))
                common_ndim = len(common_shape)
                new_shard_dim = common_ndim - len(spec_to_follow.shape) + shard_dim
                out_placements.append(Shard(new_shard_dim))
            elif isinstance(placement, Partial) and not linearity:
                # 如果是 Partial 放置，并且不支持线性，则清除该部分的放置
                # 默认情况下复制该部分，需要确认这是否对所有情况都最优
                out_placements.append(Replicate())
            else:
                # 否则直接添加当前放置
                out_placements.append(placement)

        # 初始化输入规格和重新分配成本的空列表
        input_specs: List[DTensorSpec] = []
        redistribute_costs: List[List[float]] = []

        # 遍历参数架构中的每个参数
        for idx, input_arg in enumerate(args_schema):
            if isinstance(input_arg, OpStrategy):
                # 如果是 OpStrategy 类型的参数，获取其第一个策略的输出规格
                input_arg_spec = input_arg.strategies[0].output_spec
                # 推断广播维度映射关系
                input_arg_dims_map = infer_broadcast_dims_map(
                    common_shape, input_arg_spec.shape
                )
                # 根据广播后的维度映射，映射输出放置位置
                input_target_placements = map_placements_after_broadcast(
                    tuple(out_placements),
                    common_shape,
                    input_arg_dims_map,
                )
                # 创建目标输入规格
                input_arg_target_spec = DTensorSpec(
                    mesh=mesh,
                    placements=input_target_placements,
                    tensor_meta=input_arg_spec.tensor_meta,
                )
                # 将目标输入规格添加到输入规格列表中
                input_specs.append(input_arg_target_spec)
                # 生成重新分配成本并添加到重新分配成本列表中
                redistribute_costs.append(
                    generate_redistribute_costs(input_arg, input_arg_target_spec)
                )

        # 创建一个新的放置策略并添加到 pointwise_strategy 中
        pointwise_strategy.strategies.append(
            PlacementStrategy(
                output_specs=DTensorSpec(
                    mesh=mesh,
                    placements=tuple(out_placements),
                ),
                input_specs=input_specs,
                redistribute_cost=redistribute_costs,
            )
        )

    # 返回 pointwise_strategy
    return pointwise_strategy
# 对于线性逐点操作，使用线性策略来生成操作策略
def linear_pointwise_strategy(mesh: DeviceMesh, op_schema: OpSchema) -> StrategyType:
    """
    Linear pointwise operators can propagate pending reductions.
    For example, c = add(a, b); if a is pending sum, then c will be
    pending sum as well without any communication overhead.
    """
    # 调用pointwise_strategy函数，设置linearity参数为True，返回生成的操作策略
    return pointwise_strategy(mesh, op_schema, linearity=True)


# 遍历线性逐点操作集合，为每个操作注册对应的策略
for op in linear_pointwise_ops:
    # 使用register_op_strategy函数注册操作策略，设置静态关键字参数为["out"]
    register_op_strategy(op, schema_info=RuntimeSchemaInfo(static_kwargkey=["out"]))(
        linear_pointwise_strategy
    )

# 遍历逐点操作集合，为每个操作注册对应的策略
for op in pointwise_ops:
    # 使用register_op_strategy函数注册操作策略，设置静态关键字参数为["out"]
    register_op_strategy(op, schema_info=RuntimeSchemaInfo(static_kwargkey=["out"]))(
        pointwise_strategy
    )


# TODO: add all for_each ops
# 定义所有逐元素操作的集合，用于后续注册操作策略
for_each_ops = [
    aten._foreach_abs.default,
    aten._foreach_abs_.default,
    aten._foreach_addcdiv_.Scalar,
    aten._foreach_addcdiv_.ScalarList,
    aten._foreach_addcdiv_.Tensor,
    aten._foreach_addcmul.Scalar,
    aten._foreach_addcmul_.Scalar,
    aten._foreach_addcmul_.ScalarList,
    aten._foreach_addcmul_.Tensor,
    aten._foreach_clamp_max_.Scalar,
    aten._foreach_clamp_min_.Scalar,
    aten._foreach_div_.List,
    aten._foreach_div_.ScalarList,
    aten._foreach_lerp_.Scalar,
    aten._foreach_maximum_.List,
    aten._foreach_mul.Scalar,
    aten._foreach_mul.List,
    aten._foreach_mul_.Scalar,
    aten._foreach_mul_.ScalarList,
    aten._foreach_mul_.Tensor,
    aten._foreach_mul_.List,
    aten._foreach_neg.default,
    aten._foreach_neg_.default,
    aten._foreach_reciprocal_.default,
    aten._foreach_sub.List,
    aten._foreach_sub_.Scalar,
    aten._foreach_sqrt.default,
    aten._foreach_sqrt_.default,
    aten._foreach_zero_.default,
]

# 定义所有线性逐元素操作的集合，用于后续注册操作策略
for_each_linearity_ops = [
    aten._foreach_add.Scalar,
    aten._foreach_add_.Scalar,
    aten._foreach_add_.ScalarList,
    aten._foreach_add.List,
    aten._foreach_add_.List,
]


def list_pointwise_strategy(
    mesh: DeviceMesh, op_schema: OpSchema, linearity: bool = False
) -> StrategyType:
    """
    Apply the pointwise strategy to the zipped arguments. For example, if we
    run a foreach add of two lists l1 and l2, then we apply the pointwise
    strategy on each pair (l1[i], l2[i]). If the first argument is a list but
    the second (or later) one is a tensor, then we broadcast the tensor by
    replicating it into a list with the length of the first argument.

    Args:
        mesh (DeviceMesh): device mesh for pointwise ops
        op_schema (OpSchema): schema of the operator to generate strategy for
        linearity (bool): specify whether op(a) + op(b) = op(a + b)

    Returns:
        OpStrategy: generated strategy
    """
    # 应用逐点策略到传入的参数，根据linearity参数决定是否应用线性操作策略
    # 返回生成的操作策略
    return OpStrategy(...)
    # 定义一个函数，用于处理参数模式为元组的情况，并返回一个操作策略的列表
    def args_tuple_strategies(args_schema: Tuple[object, ...]) -> List[TupleStrategy]:
        # 获取第一个参数
        first_arg = args_schema[0]
        # 断言第一个参数是 TupleStrategy 类型
        assert isinstance(first_arg, TupleStrategy)
        # 确定策略长度，即每个元组策略的子元素数量
        strategy_len = len(first_arg.childs)
        # 初始化一个空列表用于存放元组策略
        tuple_strategies: List[TupleStrategy] = []
        # 遍历参数列表中的每个参数及其索引
        for arg_idx, arg in enumerate(args_schema):
            # 如果参数是 TupleStrategy 类型
            if isinstance(arg, TupleStrategy):
                # 断言其子元素数量与第一个参数相同
                assert len(arg.childs) == strategy_len
                # 将该参数加入策略列表
                tuple_strategies.append(arg)
            # 如果参数是 OpStrategy 类型
            elif isinstance(arg, OpStrategy):
                # 如果不是第一个参数，则隐式广播该操作策略
                if arg_idx > 0:
                    tuple_strategies.append(
                        TupleStrategy([arg for _ in range(strategy_len)])
                    )
                else:
                    # 抛出异常，因为列表操作仅支持元组策略
                    raise RuntimeError(
                        f"list op only supports tuple strategy! {op_schema}"
                    )
        # 返回生成的元组策略列表
        return tuple_strategies

    # 调用函数以获取参数的策略列表
    args_strategies = args_tuple_strategies(op_schema.args_schema)
    # 获取第一个策略作为后续策略
    follow_strategy: TupleStrategy = args_strategies[0]
    # 初始化一个空列表，用于存放操作策略
    list_strategy: List[OpStrategy] = []
    # 遍历后续策略的子元素
    for child_idx, child_strtgy in enumerate(follow_strategy.childs):
        # 断言子元素是 OpStrategy 类型
        assert isinstance(child_strtgy, OpStrategy)
        # 从所有参数的策略中提取当前子元素的策略列表
        args_schema: List[StrategyType] = [
            arg_strategy.childs[child_idx] for arg_strategy in args_strategies
        ]
        # 计算共同点操作策略
        pointwise_strategy: OpStrategy = common_pointwise_strategy(
            mesh, args_schema, child_strtgy, linearity
        )
        # 将计算得到的点操作策略添加到列表中
        list_strategy.append(pointwise_strategy)
    # 返回一个新的元组策略，其中包含计算得到的列表策略
    return TupleStrategy(list_strategy)
# 定义函数，列出每个支持线性性质的点操作策略
def list_linear_pointwise_strategy(
    mesh: DeviceMesh, op_schema: OpSchema
) -> StrategyType:
    """
    对每个支持线性性质的列表操作策略进行处理
    """
    # 调用函数，返回使用线性标记的点操作策略
    return list_pointwise_strategy(mesh, op_schema, linearity=True)


# 对于每个在 for_each_ops 中的操作，注册操作策略
for op in for_each_ops:
    # 使用运行时模式信息注册操作策略，并指定需要 PyTree
    register_op_strategy(op, schema_info=RuntimeSchemaInfo(needs_pytree=True))(
        list_pointwise_strategy
    )

# 对于每个在 for_each_linearity_ops 中的操作，注册线性点操作策略
for op in for_each_linearity_ops:
    # 使用运行时模式信息注册操作策略，并指定需要 PyTree
    register_op_strategy(op, schema_info=RuntimeSchemaInfo(needs_pytree=True))(
        list_linear_pointwise_strategy
    )

# 预定义融合操作列表
fused_ops = [
    aten._fused_adam_.default,
    aten._fused_adam.default,
    aten._fused_adam.tensor_lr,
    aten._fused_adam_.tensor_lr,
    aten._fused_adamw_.default,
    aten._fused_adamw.default,
    aten._fused_adamw.tensor_lr,
    aten._fused_adamw_.tensor_lr,
]

# 对于每个预定义的融合操作，注册点操作策略
for op in fused_ops:
    # 使用运行时模式信息注册操作策略，并指定需要 PyTree
    register_op_strategy(op, schema_info=RuntimeSchemaInfo(needs_pytree=True))(
        list_pointwise_strategy
    )
```