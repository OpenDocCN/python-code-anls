# `.\pytorch\torch\fx\experimental\shape_inference\infer_shape.py`

```py
"""
This is the function that runs shape inference. It will modify the input graph module so that shapes are annotated.
"""

# 引入必要的库和模块
import copy  # 导入深拷贝功能
from collections import defaultdict  # 导入默认字典

import torch  # 导入PyTorch库
from torch._dynamo.source import LocalSource  # 导入本地数据源
from torch._subclasses import FakeTensorMode  # 导入伪张量模式
from torch.fx.experimental.proxy_tensor import make_fx  # 导入创建代理张量函数
from torch.fx.experimental.shape_inference.infer_symbol_values import (
    infer_symbol_values,  # 导入符号值推断函数
)
from torch.fx.experimental.symbolic_shapes import DimDynamic, ShapeEnv  # 导入动态维度和形状环境
from torch.utils import _pytree  # 导入_pytree模块

def infer_shape(gm, input_tensors):
    # 准备形状推断所需的环境
    shape_env = ShapeEnv()
    fake_mode = FakeTensorMode(shape_env=shape_env, allow_non_fake_inputs=True)

    # 扁平化输入张量列表，并获取规格
    flatten_inputs, spec = _pytree.tree_flatten(input_tensors)
    dim_count = 1
    # 计算所有输入张量的维度总数
    for input_tensor in flatten_inputs:
        dim_count += input_tensor.dim() - 1

    # 创建一个样例字典，用于初始化符号整数
    sample = {f"s{i}": 2 for i in range(dim_count)}
    init_symints = [
        mksym(shape_env, v, LocalSource(k), DimDynamic.DYNAMIC)  # 创建符号整数并使用动态维度
        for k, v in sample.items()
    ]
    symints = copy.deepcopy(init_symints)  # 深拷贝初始化的符号整数列表
    symbol_to_idx_dict = {f"s{i}": i for i in range(dim_count)}  # 符号到索引的映射字典
    padding_constraints = defaultdict(list)  # 声明一个默认字典用于存储填充约束

    complete_flag = False  # 完成标志位初始化为False
    allowed_try_times = dim_count * 2  # 允许的尝试次数初始化为维度总数的两倍

    # 开始循环进行形状推断
    while not complete_flag and allowed_try_times > 0:
        # 创建符号输入张量
        with fake_mode:
            sym_tensors = []
            i = 1
            for input_tensor in flatten_inputs:
                curr_dim = input_tensor.dim()
                desired_size = [symints[0]] + [
                    symints[ii] for ii in range(i, i + curr_dim - 1)
                ]
                sym_tensor = torch.randn(desired_size)  # 使用符号整数创建随机张量
                sym_tensors.append(sym_tensor)
                i += curr_dim - 1
            sym_tensors = _pytree.tree_unflatten(sym_tensors, spec)  # 还原成原始形式的输入张量列表

        try:
            with fake_mode:
                make_fx(
                    gm,
                    tracing_mode="symbolic",
                    _allow_non_fake_inputs=True,
                    pre_dispatch=True,
                    _allow_fake_constant=True,
                )(*sym_tensors)  # 调用make_fx函数进行符号推断
            complete_flag = True  # 标志位设置为True，完成形状推断
            return (gm, input_tensors, fake_mode, symints[0])  # 返回形状推断结果及相关对象
        except RuntimeError as e:
            if e:
                # 异常处理：根据异常信息推断符号整数的值
                infer_symbol_values(
                    symints,
                    init_symints,
                    symbol_to_idx_dict,
                    padding_constraints,
                    str(e),
                )
                allowed_try_times -= 1  # 尝试次数减一
        except ValueError as e:
            if e:
                # 异常处理：根据异常信息推断符号整数的值
                infer_symbol_values(
                    symints,
                    init_symints,
                    symbol_to_idx_dict,
                    padding_constraints,
                    str(e),
                )
                allowed_try_times -= 1  # 尝试次数减一
# 定义一个函数 mksym，用于创建符号整数节点
def mksym(shape_env, value, source, dynamic_dim):
    # 调用 shape_env 对象的 create_symbol 方法，创建一个符号对象，传入 value 作为符号的值
    # 设置符号的来源为 source，动态维度为 dynamic_dim
    return shape_env.create_symintnode(
        shape_env.create_symbol(
            value,
            source=source,
            dynamic_dim=dynamic_dim,
        ),
        # 将 value 设置为创建的符号整数节点的提示信息
        hint=value,
        # 设定符号整数节点的来源为 source
        source=source,
    )
```