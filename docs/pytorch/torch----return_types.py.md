# `.\pytorch\torch\return_types.py`

```
import inspect  # 导入inspect模块，用于检查对象的类型和内部结构

import torch  # 导入PyTorch库
from torch.utils._pytree import register_pytree_node, SequenceKey  # 从torch.utils._pytree导入register_pytree_node和SequenceKey

__all__ = ["pytree_register_structseq", "all_return_types"]  # 定义一个列表__all__，包含公开的符号名称

all_return_types = []  # 初始化空列表all_return_types，用于存储所有的返回类型

# error: Module has no attribute "_return_types"
return_types = torch._C._return_types  # type: ignore[attr-defined]
# 从torch._C._return_types模块导入return_types，用于描述返回类型，如果出现错误则忽略其类型检查

def pytree_register_structseq(cls):
    # 定义一个函数structseq_flatten，用于将结构化序列展平为列表，上下文为None
    def structseq_flatten(structseq):
        return list(structseq), None

    # 定义一个函数structseq_flatten_with_keys，将结构化序列展平为带有索引键的列表
    def structseq_flatten_with_keys(structseq):
        values, context = structseq_flatten(structseq)
        return [(SequenceKey(i), v) for i, v in enumerate(values)], context

    # 定义一个函数structseq_unflatten，用于将展平的值重新构建为结构化序列
    def structseq_unflatten(values, context):
        return cls(values)

    # 注册pytree节点，将cls作为结构化序列的一部分
    register_pytree_node(
        cls,
        structseq_flatten,
        structseq_unflatten,
        flatten_with_keys_fn=structseq_flatten_with_keys,
    )


# 遍历return_types模块中的所有属性名称
for name in dir(return_types):
    # 如果属性名称以双下划线开头，则跳过
    if name.startswith("__"):
        continue

    # 获取return_types模块中名称为name的属性对象
    _attr = getattr(return_types, name)
    # 将属性对象赋值给全局命名空间中的name变量
    globals()[name] = _attr

    # 如果属性名称不以下划线开头，则将其添加到__all__列表中，并添加到all_return_types列表中
    if not name.startswith("_"):
        __all__.append(name)
        all_return_types.append(_attr)

    # 打印说明文本，表明torch.return_types中的所有内容都是结构化序列（structseq）
    # 当不再是这种情况时，需要修改这部分代码
    # 注意：不知道如何检查是否是“structseq”，因此进行模糊检查是否为元组（tuple）
    if inspect.isclass(_attr) and issubclass(_attr, tuple):
        pytree_register_structseq(_attr)
```