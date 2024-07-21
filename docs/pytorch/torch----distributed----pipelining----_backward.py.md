# `.\pytorch\torch\distributed\pipelining\_backward.py`

```
# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates

# 引入 List 和 Optional 类型提示
from typing import List, Optional

# 引入 PyTorch 库
import torch

# 从 _debug 模块中导入 map_debug_info 函数
from ._debug import map_debug_info

# 定义 stage_backward 函数，用于反向传播的辅助函数
def stage_backward(
    stage_output,  # 存储阶段输出的变量或张量
    output_grads,  # 存储输出梯度的变量或张量
    input_values,  # 存储输入值的变量或张量
    outputs_with_grads_idxs: Optional[List[int]] = None,  # 已弃用，运行时不使用
):
    """
    This is a helper function to:
    1. compute the gradients for the stage inputs, and
    2. accumulate gradients for the stage module's parameters.

    Given the input value(s) and the corresponding gradient for the output
    value(s), compute and accumulate gradients for all parameter values (leaves
    in the autograd trace) as well as return a list of the gradients for the
    input values
    """
    # 如果 outputs_with_grads_idxs 不为 None，则根据索引重新设置 stage_output 和 output_grads
    if outputs_with_grads_idxs is not None:
        # 已弃用的参数，在运行时调用中不使用，只存在于编译器中
        stage_output = [stage_output[i] for i in outputs_with_grads_idxs]
        output_grads = [output_grads[i] for i in outputs_with_grads_idxs]
    try:
        # stage_output 可能是一个复合数据类型，比如字典。这里提取所有单独的张量数值
        stage_output_tensors = []
        output_grad_tensors = []

        def extract_tensors_with_grads(output_val, grad_val):
            # 如果 output_val 是 torch.Tensor 类型
            if isinstance(output_val, torch.Tensor):
                # 如果 output_val 不需要梯度且没有梯度函数
                if not output_val.requires_grad and output_val.grad_fn is None:
                    return
                # grad_val 应该是 torch.Tensor 类型或者 None
                assert isinstance(
                    grad_val, (torch.Tensor, type(None))
                ), f"Expected Tensor or None gradient but got {type(grad_val)}"
                # 将 output_val 加入到 stage_output_tensors 中
                stage_output_tensors.append(output_val)
                # 将 grad_val 加入到 output_grad_tensors 中
                output_grad_tensors.append(grad_val)
            # 如果 output_val 是 tuple 或者 list 类型
            elif isinstance(output_val, (tuple, list)):
                # 如果 grad_val 是 None，则直接返回
                if grad_val is None:
                    return
                # grad_val 应该是 tuple 或者 list 类型，并且长度应与 output_val 相同
                assert isinstance(
                    grad_val, (tuple, list)
                ), f"grad_value expected to have type {type(output_val)} but got {type(grad_val)}"
                assert len(output_val) == len(grad_val)
                # 逐个处理 output_val 和 grad_val 中的每个元素
                for ov, gv in zip(output_val, grad_val):
                    extract_tensors_with_grads(ov, gv)
            # 如果 output_val 是 dict 类型
            elif isinstance(output_val, dict):
                # 如果 grad_val 是 None，则直接返回
                if grad_val is None:
                    return
                # grad_val 应该是 dict 类型，并且其键集合应与 output_val 相同
                assert isinstance(grad_val, dict)
                assert set(output_val.keys()) == set(grad_val.keys())
                # 逐个处理 output_val 和 grad_val 中的每个键值对
                for k in output_val.keys():
                    extract_tensors_with_grads(output_val[k], grad_val[k])
            else:
                # 输出是非张量类型，直接忽略
                pass

        # 调用函数，提取带有梯度的张量
        extract_tensors_with_grads(stage_output, output_grads)

        # 使用 autograd.backward 计算梯度
        torch.autograd.backward(
            stage_output_tensors, grad_tensors=output_grad_tensors  # type: ignore[arg-type]
        )

        # 提取关于输入值的梯度
        grad_inputs = []
        for val in input_values:
            # 如果输入值是 torch.Tensor 类型，则加入到 grad_inputs 中
            if isinstance(val, torch.Tensor):
                grad_inputs.append(val.grad)
            else:
                grad_inputs.append(None)

    except Exception as e:
        # 如果出现异常，生成详细的异常信息
        exc_msg = f"""
        Failed to run stage backward:
        Stage output: {map_debug_info(stage_output)}
        Output gradient: {map_debug_info(output_grads)}
        Input: {map_debug_info(input_values)}
        """
        # 抛出运行时异常，并附带详细信息
        raise RuntimeError(exc_msg) from e

    # 返回计算得到的梯度列表
    return grad_inputs
# 定义一个函数 _null_coalesce_accumulate，用于合并两个值，即使其中一个值为 null，返回非空的值。
# 参数 lhs 和 rhs 分别表示左右两个待合并的值。

def _null_coalesce_accumulate(lhs, rhs):
    """
    Coalesce two values, even if one of them is null, returning the non-null
    value.
    """
    # 如果左值 lhs 为 None，则返回右值 rhs
    if lhs is None:
        return rhs
    # 如果右值 rhs 为 None，则返回左值 lhs
    elif rhs is None:
        return lhs
    else:
        # 如果 lhs 和 rhs 都不为 None，则返回它们的和，使用 torch.add 实现
        return torch.add(lhs, rhs)
```