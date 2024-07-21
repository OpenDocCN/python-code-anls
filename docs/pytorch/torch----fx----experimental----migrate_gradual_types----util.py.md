# `.\pytorch\torch\fx\experimental\migrate_gradual_types\util.py`

```
# mypy: allow-untyped-defs
# 从 torch.fx.experimental.migrate_gradual_types.constraint 模块导入所需的符号
from torch.fx.experimental.migrate_gradual_types.constraint import TVar, DVar, BinConstraintD, \
    BVar
# 从 torch.fx.experimental.migrate_gradual_types.operation 模块导入 op_leq 函数
from torch.fx.experimental.migrate_gradual_types.operation import op_leq


def gen_tvar(curr):
    """
    Generate a tensor variable
    :param curr: The current counter
    :return: a tensor variable and the updated counter
    """
    curr += 1
    return TVar(curr), curr


def gen_dvar(curr):
    """
    Generate a dimension variable
    :param curr: the current counter
    :return: a dimension variable and an updated counter
    """
    curr += 1
    return DVar(curr), curr


def gen_bvar(curr):
    """
    Generate a boolean variable
    :param curr: the current counter
    :return: a boolean variable and an updated counter
    """
    curr += 1
    return BVar(curr), curr


def gen_tensor_dims(n, curr):
    """
    Generate a list of tensor dimensions
    :param n:  the number of dimensions
    :param curr: the current counter
    :return: a list of dimension variables and an updated counter
    """
    dims = []
    for _ in range(n):
        # Generate a new dimension variable and update the counter
        dvar, curr = gen_dvar(curr)
        dims.append(dvar)
    return dims, curr


def gen_nat_constraints(list_of_dims):
    """
    Generate natural number constraints for dimensions
    :param list_of_dims: list of dimension variables
    :return: list of binary constraints (less than or equal to zero) for each dimension
    """
    # Generate a binary constraint (less than or equal to zero) for each dimension variable
    return [BinConstraintD(0, d, op_leq) for d in list_of_dims]
```