# `.\pytorch\torch\nn\modules\utils.py`

```
# mypy: allow-untyped-defs
# 导入必要的模块和库
import collections
from itertools import repeat
from typing import Any, Dict, List

# 模块中公开的函数名列表
__all__ = ["consume_prefix_in_state_dict_if_present"]

# 定义一个生成指定元组大小的函数
def _ntuple(n, name="parse"):
    def parse(x):
        # 如果输入是可迭代对象，则转换为元组
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        # 否则重复元素构成元组
        return tuple(repeat(x, n))
    
    parse.__name__ = name
    return parse

# 定义不同大小元组生成函数的具体实现
_single = _ntuple(1, "_single")
_pair = _ntuple(2, "_pair")
_triple = _ntuple(3, "_triple")
_quadruple = _ntuple(4, "_quadruple")

# 反转元组并重复每个元素指定次数的函数
def _reverse_repeat_tuple(t, n):
    r"""Reverse the order of `t` and repeat each element for `n` times.

    This can be used to translate padding arg used by Conv and Pooling modules
    to the ones used by `F.pad`.
    """
    return tuple(x for x in reversed(t) for _ in range(n))

# 返回带有默认值的输出大小列表
def _list_with_default(out_size: List[int], defaults: List[int]) -> List[int]:
    import torch

    # 如果输出大小是单个整数或者 SymInt 类型，则直接返回
    if isinstance(out_size, (int, torch.SymInt)):
        return out_size
    # 如果默认值列表长度小于等于输出大小列表，则抛出异常
    if len(defaults) <= len(out_size):
        raise ValueError(f"Input dimension should be at least {len(out_size) + 1}")
    # 否则根据输出大小和默认值构造新的列表
    return [
        v if v is not None else d for v, d in zip(out_size, defaults[-len(out_size):])
    ]

# 在状态字典中消除前缀（如果存在）的函数
def consume_prefix_in_state_dict_if_present(
    state_dict: Dict[str, Any],
    prefix: str,
) -> None:
    r"""Strip the prefix in state_dict in place, if any.

    ..note::
        Given a `state_dict` from a DP/DDP model, a local model can load it by applying
        `consume_prefix_in_state_dict_if_present(state_dict, "module.")` before calling
        :meth:`torch.nn.Module.load_state_dict`.

    Args:
        state_dict (OrderedDict): a state-dict to be loaded to the model.
        prefix (str): prefix.
    """
    # 获取状态字典中的所有键
    keys = list(state_dict.keys())
    # 遍历键列表
    for key in keys:
        # 如果键以指定前缀开头
        if key.startswith(prefix):
            # 新键名为去除前缀后的名称
            newkey = key[len(prefix):]
            # 将原键值对弹出，并以新键名重新插入字典中
            state_dict[newkey] = state_dict.pop(key)

    # 如果状态字典有元数据属性，则也在元数据中消除前缀
    if hasattr(state_dict, "_metadata"):
        # 获取元数据字典的所有键
        keys = list(state_dict._metadata.keys())
        # 遍历键列表
        for key in keys:
            # 对于元数据字典，键可能是空字符串、'module'或者'module.xx.xx'
            if len(key) == 0:
                continue
            # 处理 'module' 或者 'module.' 开头的情况
            if key == prefix.replace(".", "") or key.startswith(prefix):
                # 新键名为去除前缀后的名称
                newkey = key[len(prefix):]
                # 将原键值对弹出，并以新键名重新插入元数据字典中
                state_dict._metadata[newkey] = state_dict._metadata.pop(key)
```