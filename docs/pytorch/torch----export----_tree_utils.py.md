# `.\pytorch\torch\export\_tree_utils.py`

```
# 导入所需的类型和函数
from typing import Any, Callable, Dict, Optional
# 导入torch.utils._pytree模块中的Context和TreeSpec类

from torch.utils._pytree import Context, TreeSpec


def reorder_kwargs(user_kwargs: Dict[str, Any], spec: TreeSpec) -> Dict[str, Any]:
    """重新排序用户提供的关键字参数，使其与`spec`中的顺序匹配。`spec`应该是导出程序的in_spec，
    即从`(args, kwargs)`展开得到的spec。

    我们需要这个函数来提供一致的输入顺序，这样用户可以传入foo(a=a, b=b)或foo(b=b, a=a)并获得相同的结果。
    """
    # 确保spec实际上是(args, kwargs)形状
    assert spec.type is tuple
    assert spec.num_children == 2
    kwargs_spec = spec.children_specs[1]
    # 确保kwargs_spec是dict类型
    assert kwargs_spec.type is dict

    if set(user_kwargs) != set(kwargs_spec.context):
        raise ValueError(
            f"kwarg key mismatch: "
            f"Got {list(user_kwargs)} but expected {kwargs_spec.context}"
        )

    reordered_kwargs = {}
    for kw in kwargs_spec.context:
        reordered_kwargs[kw] = user_kwargs[kw]

    return reordered_kwargs


def is_equivalent(
    spec1: TreeSpec,
    spec2: TreeSpec,
    equivalence_fn: Callable[[Optional[type], Context, Optional[type], Context], bool],
) -> bool:
    """定制化的两个TreeSpecs的等价性检查。

    参数:
        spec1: 要比较的第一个TreeSpec
        spec2: 要比较的第二个TreeSpec
        equivalence_fn: 一个函数，通过检查它们的类型和context来确定两个TreeSpecs的等价性。将按以下方式调用：

                equivalence_fn(spec1.type, spec1.context, spec2.type, spec2.context)

            此函数将递归应用于所有子项。

    返回:
        如果两个TreeSpecs等价则返回True，否则返回False。
    """
    if not equivalence_fn(spec1.type, spec1.context, spec2.type, spec2.context):
        return False

    # 递归处理子项
    if len(spec1.children_specs) != len(spec2.children_specs):
        return False

    for child_spec1, child_spec2 in zip(spec1.children_specs, spec2.children_specs):
        if not is_equivalent(child_spec1, child_spec2, equivalence_fn):
            return False

    return True
```