# `.\pytorch\torch\_inductor\mkldnn_lowerings.py`

```
# mypy: allow-untyped-defs
# 引入所需的模块和类型
from typing import List, Optional

# 导入 PyTorch 库
import torch
import torch.utils._pytree as pytree
from torch._inductor.kernel.mm_common import mm_args
# 从当前目录导入特定模块
from . import ir, mkldnn_ir
# 导入 C++ gemm 模板生成器
from .codegen.cpp_gemm_template import CppPackedGemmTemplate
# 导入张量盒子类
from .ir import TensorBox
# 导入降低模块
from .lowering import (
    add,
    add_needs_realized_inputs,
    aten,
    permute,
    register_lowering,
    to_dtype,
    view,
)
# 导入算法选择模块
from .select_algorithm import (
    autotune_select_algorithm,
    ChoiceCaller,
    ExternKernelChoice,
)
# 导入工具函数
from .utils import use_aten_gemm_kernels, use_cpp_packed_gemm_template, use_max_autotune
# 导入虚拟化操作和 V 类
from .virtualized import ops, V

# 创建带有属性的结尾部分函数
def create_epilogue_with_attr(input_buffer, attr, **kwargs):
    # 创建输入加载器
    input_loader = input_buffer.make_loader()
    # 获取数据类型
    dtype = input_buffer.get_dtype()
    # 如果属性是 "relu"
    if attr == "relu":
        # 定义内部函数，用于处理每个索引
        def inner_fn(index):
            # 获取输入数据
            input = input_loader(index)
            # 创建零常数张量
            zero = ops.constant(0, dtype)
            # 返回输入张量与零张量的元素级最大值
            return ops.maximum(input, zero)
    
    # 如果属性是 "gelu"
    elif attr == "gelu":
        # 断言 "algorithm" 在 kwargs 中
        assert "algorithm" in kwargs
        # 如果算法是 "none"
        if kwargs["algorithm"] == "none":
            # 定义内部函数，用于处理每个索引
            def inner_fn(index):
                # 获取输入数据
                input = input_loader(index)
                # 如果数据类型不是 torch.float，转换为 torch.float
                if dtype != torch.float:
                    input = ops.to_dtype(input, torch.float)
                # 创建常数张量
                half = ops.constant(0.5, torch.float)
                one = ops.constant(1.0, torch.float)
                const = ops.constant(0.7071067811865476, torch.float)
                # 计算 GELU 函数
                result = input * half * (ops.erf(input * const) + one)
                # 如果数据类型不是原始类型，转换为原始类型
                if dtype != torch.float:
                    result = ops.to_dtype(result, dtype)
                return result
        
        # 如果算法是 "tanh"
        else:
            assert kwargs["algorithm"] == "tanh"
            # 定义内部函数，用于处理每个索引
            def inner_fn(index):
                # 获取输入数据
                input = input_loader(index)
                # 如果数据类型不是 torch.float，转换为 torch.float
                if dtype != torch.float:
                    input = ops.to_dtype(input, torch.float)
                # 创建常数张量
                half = ops.constant(0.5, torch.float)
                one = ops.constant(1.0, torch.float)
                const1 = ops.constant(0.7978845608028654, torch.float)
                const2 = ops.constant(0.044715, torch.float)
                # 计算 tanh 算法
                result = (
                    half
                    * input
                    * (
                        one
                        + ops.tanh(const1 * (input + const2 * input * input * input))
                    )
                )
                # 如果数据类型不是原始类型，转换为原始类型
                if dtype != torch.float:
                    result = ops.to_dtype(result, dtype)
                return result
    
    # 如果属性是 "swish"
    elif attr == "swish":
        # 定义内部函数，用于处理每个索引
        def inner_fn(index):
            # 获取输入数据
            input = input_loader(index)
            # 计算 Swish 函数
            result = input * ops.sigmoid(input)
            return result
    
    # 如果属性是 "sigmoid"
    elif attr == "sigmoid":
        # 定义内部函数，用于处理每个索引
        def inner_fn(index):
            # 返回输入数据的 sigmoid 函数结果
            return ops.sigmoid(input_loader(index))
    
    # 如果属性是 "tanh"
    elif attr == "tanh":
        # 定义内部函数，用于处理每个索引
        def inner_fn(index):
            # 返回输入数据的 tanh 函数结果
            return ops.tanh(input_loader(index))
    # 如果属性是 "hardswish" 或者 "hardsigmoid"，则进入以下逻辑分支
    elif attr == "hardswish" or attr == "hardsigmoid":

        # 定义 hardsigmoid_float 函数，计算硬 sigmoid 函数
        def hardsigmoid_float(input):
            # 定义常量 zero、six、three、one_over_six
            zero = ops.constant(0, torch.float)
            six = ops.constant(6, torch.float)
            three = ops.constant(3, torch.float)
            one_over_six = ops.constant(0.16666666666666666, torch.float)
            # 计算 input + 3 和 0 的最大值
            max = ops.maximum(input + three, zero)
            # 取 max 和 6 的最小值
            min = ops.minimum(max, six)
            return min * one_over_six

        # 定义 inner_fn 函数，用于处理每个索引
        def inner_fn(index):
            # 调用 input_loader 函数获取输入
            input = input_loader(index)
            # 如果数据类型不是 torch.float，则转换为 torch.float
            if dtype != torch.float:
                input = ops.to_dtype(input, torch.float)
            # 计算 hardsigmoid_float 函数的结果
            result = hardsigmoid_float(input)
            # 如果属性是 "hardswish"，则对结果应用 hardswish 函数
            if attr == "hardswish":
                result = input * result
            # 如果数据类型不是 torch.float，则再次转换为指定的数据类型
            if dtype != torch.float:
                result = ops.to_dtype(result, dtype)
            return result

    # 如果属性是 "leaky_relu"，则进入以下逻辑分支
    elif attr == "leaky_relu":
        # 确保 kwargs 字典中存在键 "scalars"
        assert "scalars" in kwargs
        # 确保 kwargs["scalars"] 是一个长度为 1 的列表
        assert len(kwargs["scalars"]) == 1
        # 获取 negative_slope 参数
        negative_slope = kwargs["scalars"][0]

        # 定义 inner_fn 函数，用于处理每个索引
        def inner_fn(index):
            # 调用 input_loader 函数获取输入
            input = input_loader(index)
            # 如果数据类型不是 torch.float，则转换为 torch.float
            if dtype != torch.float:
                input = ops.to_dtype(input, torch.float)
            # 定义常量 zero
            zero = ops.constant(0, torch.float)
            # 根据输入值和 negative_slope 计算 leaky ReLU 函数的结果
            result = ops.where(
                input > zero, input, input * ops.constant(negative_slope, torch.float)
            )
            # 如果数据类型不是 torch.float，则再次转换为指定的数据类型
            if dtype != torch.float:
                result = ops.to_dtype(result, dtype)
            return result

    # 如果属性是 "hardtanh"，则进入以下逻辑分支
    elif attr == "hardtanh":
        # 确保 kwargs 字典中存在键 "scalars"
        assert "scalars" in kwargs
        # 确保 kwargs["scalars"] 是一个长度为 2 的列表
        assert len(kwargs["scalars"]) == 2
        # 获取 min_value 和 max_value 参数
        min_value = kwargs["scalars"][0]
        max_value = kwargs["scalars"][1]

        # 定义 inner_fn 函数，用于处理每个索引
        def inner_fn(index):
            # 调用 input_loader 函数获取输入
            input = input_loader(index)
            # 如果数据类型不是 torch.float，则转换为 torch.float
            if dtype != torch.float:
                input = ops.to_dtype(input, torch.float)
            # 根据输入值、min_value 和 max_value 计算 hardtanh 函数的结果
            result = ops.minimum(
                ops.maximum(input, ops.constant(min_value, torch.float)),
                ops.constant(max_value, torch.float),
            )
            # 如果数据类型不是 torch.float，则再次转换为指定的数据类型
            if dtype != torch.float:
                result = ops.to_dtype(result, dtype)
            return result

    # 如果属性是 "add" 或者 "sub"，则进入以下逻辑分支
    elif attr == "add" or attr == "sub":
        # 确保 kwargs 字典中存在键 "other"
        assert "other" in kwargs
        # 获取 other 参数，并创建其加载器 other_loader
        other = kwargs["other"]
        other_loader = other.make_loader()

        # 定义 inner_fn 函数，用于处理每个索引
        def inner_fn(index):
            # 根据操作符 attr 获取对应的操作函数 op
            op = getattr(ops, attr)
            # 返回 input_loader(index) 和 other_loader(index) 的操作结果
            return op(input_loader(index), other_loader(index))

    # 如果属性不在支持的属性列表中，则抛出异常
    else:
        raise ValueError(f"Unsupported epilogue attribute: {attr}")

    # 返回一个 Pointwise 对象，指定设备、数据类型、处理函数和输入缓冲区大小
    return ir.Pointwise(
        device=input_buffer.get_device(),
        dtype=dtype,
        inner_fn=inner_fn,
        ranges=input_buffer.get_size(),
    )
def register_onednn_fusion_ops():
    # 这里是一个函数定义，定义了一个名为 register_onednn_fusion_ops 的函数
    else:
        # 这里是一个不完整的条件语句，没有与之匹配的 if 或 elif，因此会导致语法错误
        pass
```