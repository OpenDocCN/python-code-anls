# `.\quantizers\quantizers_utils.py`

```
# 导入必要的类型
from typing import Any, Tuple

# 定义一个函数，接受一个模块和一个字符串作为参数，返回一个元组
def get_module_from_name(module, tensor_name: str) -> Tuple[Any, str]:
    # 如果字符串中包含"."，则按"."分割字符串
    if "." in tensor_name:
        splits = tensor_name.split(".")
        # 遍历分割后的字符串列表，除了最后一个元素
        for split in splits[:-1]:
            # 获取模块中的属性
            new_module = getattr(module, split)
            # 如果获取的属性为None，则抛出异常
            if new_module is None:
                raise ValueError(f"{module} has no attribute {split}.")
            # 更新模块为获取到的属性
            module = new_module
        # 更新张量名称为分割后列表的最后一个元素
        tensor_name = splits[-1]
    # 返回更新后的模块和张量名称组成的元组
    return module, tensor_name
```