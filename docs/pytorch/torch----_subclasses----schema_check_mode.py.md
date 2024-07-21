# `.\pytorch\torch\_subclasses\schema_check_mode.py`

```py
# mypy: ignore-errors

# Import necessary modules and classes
from collections import namedtuple
from copy import deepcopy
from itertools import combinations

import torch
from torch.fx.operator_schemas import normalize_function
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map

# Named tuple definitions for SchemaCheckMode
Mutation = namedtuple("Mutation", ["op_name", "arg_name"])
Aliasing = namedtuple("Aliasing", ["op_name", "arg_name", "output_number"])

# Simplified naming for C++ classes used in Torch
SchemaArgument = torch._C._SchemaArgument
SchemaArgType = torch._C._SchemaArgType
SchemaInfo = torch._C._SchemaInfo

# This subclass of TorchDispatchMode verifies op schemas
class SchemaCheckMode(TorchDispatchMode):
    def __init__(self):
        # Initialize lists to record operations, mutations, and aliasing
        self.ops = []       # Records called ops
        self.mutated = []   # Records mutations on inputs
        self.aliasing = []  # Records aliasing on inputs

    def reset_cache(self):
        # Clear recorded information lists
        self.ops.clear()
        self.mutated.clear()
        self.aliasing.clear()

    def display_ops(self):
        # Print all recorded ops
        print(*self.ops, sep=",")

# move these 2 functions here to avoid numpy dependency in testing/_internal/common_utils.py

def is_iterable_of_tensors(iterable):
    # Check if the iterable is a tensor itself
    if isinstance(iterable, torch.Tensor):
        return False
    try:
        # Iterate through elements to verify if all are tensors
        if len(iterable) == 0:
            return False
        for t in iter(iterable):
            if not isinstance(t, torch.Tensor):
                return False
    except TypeError as te:
        return False
    return True

def clone_inputs(args):
    # Create a deep copy of each argument in args
    inputs = []
    for arg in args:
        if isinstance(arg, torch.Tensor):
            # If argument is a tensor, detach and clone it
            inputs.append(arg.detach().clone())
        elif is_iterable_of_tensors(arg):
            # If argument is an iterable containing tensors, clone each tensor
            inputs.append([t.detach().clone() for t in arg])
        else:
            # Otherwise, directly append the argument
            inputs.append(arg)
    return inputs
        # 检查每个输出是否已经发生了突变
        for j, name in enumerate(func._schema.returns):
            log.debug(
                f"However, we found that `outputs[{str(j)}] is {name}`"
            )
            # 检查在给定的变异前后是否有任何突变
            if any(
                has_mutated(a, b, c)
                for a, b, c in zip(
                    pytree.tree_leaves(before), pytree.tree_leaves(after), md
                )
            ):
                # 如果输出被检测到有突变但未定义为可变，则抛出异常
                if not schema_info.is_mutable(
                    SchemaArgument(SchemaArgType.input, i)
                ):
                    raise RuntimeError(
                        f"Argument {name} is not defined as mutable but was mutated"
                    )
                else:
                    # 如果输出被检测到有突变且定义为可变，则记录变异信息
                    self.mutated.append(Mutation(func._schema.name, name))

        # 检查输出之间是否存在别名
        for i, j in combinations(range(len(func._schema.returns)), 2):
            if has_aliased(tuple_out[i], tuple_out[j]):
                # 如果未预期地发现输出之间存在别名，则抛出异常
                if not schema_info.may_contain_alias(
                    SchemaArgument(SchemaArgType.output, i),
                    SchemaArgument(SchemaArgType.output, j),
                ):
                    raise RuntimeError(f"Outputs {i} and {j} alias unexpectedly")

        # 返回计算结果
        return out
```