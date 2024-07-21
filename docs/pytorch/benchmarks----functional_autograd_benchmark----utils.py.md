# `.\pytorch\benchmarks\functional_autograd_benchmark\utils.py`

```py
# 引入默认字典集合模块
from collections import defaultdict
# 引入类型提示模块中的 Callable, Dict, List, Tuple, Union 类
from typing import Callable, Dict, List, Tuple, Union

# 引入 PyTorch 深度学习框架
import torch

# 从 torch 模块中引入 nn（神经网络）模块和 Tensor（张量）类
from torch import nn, Tensor

# Type helpers
# InputsType 类型为 Tensor 或者 Tensor 元组
InputsType = Union[Tensor, Tuple[Tensor, ...]]
# GetterReturnType 类型为包含 Callable 对象和 InputsType 对象的元组
GetterReturnType = Tuple[Callable[..., Tensor], InputsType]
# GetterType 类型为接收 torch.device 参数并返回 GetterReturnType 对象的可调用类型
GetterType = Callable[[torch.device], GetterReturnType]
# VType 类型为 None、Tensor 或 Tensor 元组的联合类型，通常用于 vjp、jvp、vhp 或 hvp 中的 v
VType = Union[None, Tensor, Tuple[Tensor, ...]]
# TimingResultType 类型为嵌套字典，用于存储计时结果，包含模型名、任务名和性能结果的元组
TimingResultType = Dict[str, Dict[str, Tuple[float, ...]]]


# Utilities to make nn.Module "functional"
# In particular the goal is to be able to provide a function that takes as input
# the parameters and evaluate the nn.Module using fixed inputs.

# _del_nested_attr 函数删除指定名称列表的对象属性
def _del_nested_attr(obj: nn.Module, names: List[str]) -> None:
    """
    Deletes the attribute specified by the given list of names.
    For example, to delete the attribute obj.conv.weight,
    use _del_nested_attr(obj, ['conv', 'weight'])
    """
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        _del_nested_attr(getattr(obj, names[0]), names[1:])


# _set_nested_attr 函数将指定名称列表的对象属性设置为给定值
def _set_nested_attr(obj: nn.Module, names: List[str], value: Tensor) -> None:
    """
    Set the attribute specified by the given list of names to value.
    For example, to set the attribute obj.conv.weight,
    use _del_nested_attr(obj, ['conv', 'weight'], value)
    """
    if len(names) == 1:
        setattr(obj, names[0], value)
    else:
        _set_nested_attr(getattr(obj, names[0]), names[1:], value)


# extract_weights 函数从模型中提取所有参数，并返回参数的元组及其原始属性名称列表
def extract_weights(mod: nn.Module) -> Tuple[Tuple[Tensor, ...], List[str]]:
    """
    This function removes all the Parameters from the model and
    return them as a tuple as well as their original attribute names.
    The weights must be re-loaded with `load_weights` before the model
    can be used again.
    Note that this function modifies the model in place and after this
    call, mod.parameters() will be empty.
    """
    # 获取模型的原始参数元组
    orig_params = tuple(mod.parameters())
    names = []
    # 遍历模型中的命名参数列表，删除模型中的所有参数
    for name, p in list(mod.named_parameters()):
        _del_nested_attr(mod, name.split("."))
        names.append(name)

    # 将原始参数元组中的每个参数转换为普通张量，并保留其梯度历史
    params = tuple(p.detach().requires_grad_() for p in orig_params)
    return params, names


# load_weights 函数重新加载一组参数，以便模型可以再次用于执行前向传播
def load_weights(mod: nn.Module, names: List[str], params: Tuple[Tensor, ...]) -> None:
    """
    Reload a set of weights so that `mod` can be used again to perform a forward pass.
    Note that the `params` are regular Tensors (that can have history) and so are left
    as Tensors. This means that mod.parameters() will still be empty after this call.
    """
    # 对names和params列表中的每一对元素进行迭代，分别赋值给name和p
    for name, p in zip(names, params):
        # 调用_set_nested_attr函数，将name按"."分割为列表，并将p作为最后一个参数传递给该函数
        _set_nested_attr(mod, name.split("."), p)
# Utilities to read/write markdown table-like content.
# 将结果以 Markdown 表格的形式输出
def to_markdown_table(res: TimingResultType, header: Tuple[str, ...] = None) -> str:
    if header is None:
        header = ("model", "task", "mean", "var")
    out = ""

    def write_line(*args):
        nonlocal out
        out += f"| {' | '.join(str(a) for a in args)} |\n"

    # 构建 Markdown 表格的表头
    write_line(*header)
    # 添加分隔线
    write_line(*["--"] * len(header))
    # 遍历结果字典，填充表格内容
    for model, tasks in res.items():
        for task, line in tasks.items():
            write_line(*(model, task) + line)

    return out


# 从 Markdown 表格的字符串中解析出结果字典
def from_markdown_table(data: str) -> TimingResultType:
    out = data.strip().split("\n")
    out = out[2:]  # 忽略表头行

    res: TimingResultType
    res = defaultdict(defaultdict)

    # 解析每一行，填充结果字典
    for line in out:
        model, task, mean, var = (f.strip() for f in line.strip().split("|") if f)
        res[model][task] = (float(mean), float(var))

    return res


# 检查是否安装了 functorch 库
def check_for_functorch():
    try:
        import functorch  # noqa: F401  # 尝试导入 functorch 库
        return True  # 导入成功则返回 True
    except ImportError:
        return False  # 导入失败则返回 False
```