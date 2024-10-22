# `.\diffusers\utils\outputs.py`

```py
# 版权所有 2024 HuggingFace 团队。保留所有权利。
#
# 根据 Apache 许可证，第 2.0 版（“许可证”）进行许可；
# 除非遵循许可证，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面协议另有规定，软件
# 按“现状”分发，不附带任何明示或暗示的担保或条件。
# 请参阅许可证以了解管理权限和
# 限制的具体语言。
"""
通用工具函数
"""

# 从有序字典模块导入 OrderedDict
from collections import OrderedDict
# 从数据类模块导入字段和是否为数据类的检查
from dataclasses import fields, is_dataclass
# 导入 Any 和 Tuple 类型
from typing import Any, Tuple

# 导入 NumPy 库
import numpy as np

# 从本地导入工具模块，检查 PyTorch 是否可用及其版本
from .import_utils import is_torch_available, is_torch_version


def is_tensor(x) -> bool:
    """
    测试 `x` 是否为 `torch.Tensor` 或 `np.ndarray`。
    """
    # 如果 PyTorch 可用
    if is_torch_available():
        # 导入 PyTorch 库
        import torch

        # 检查 x 是否为 torch.Tensor 类型
        if isinstance(x, torch.Tensor):
            return True

    # 检查 x 是否为 np.ndarray 类型
    return isinstance(x, np.ndarray)


class BaseOutput(OrderedDict):
    """
    所有模型输出的基类，作为数据类。具有一个 `__getitem__` 方法，允许通过整数或切片（像元组）或字符串（像字典）进行索引，并会忽略 `None` 属性。
    否则像常规 Python 字典一样工作。

    <提示 警告={true}>
    
    不能直接解包 [`BaseOutput`]。请先使用 [`~utils.BaseOutput.to_tuple`] 方法将其转换为元组。
    
    </提示>
    """

    def __init_subclass__(cls) -> None:
        """将子类注册为 pytree 节点。

        这对于在使用 `torch.nn.parallel.DistributedDataParallel` 和
        `static_graph=True` 时同步梯度是必要的，尤其是对于输出 `ModelOutput` 子类的模块。
        """
        # 如果 PyTorch 可用
        if is_torch_available():
            # 导入 PyTorch 的 pytree 工具
            import torch.utils._pytree

            # 检查 PyTorch 版本是否小于 2.2
            if is_torch_version("<", "2.2"):
                # 注册 pytree 节点，使用字典扁平化和解扁平化
                torch.utils._pytree._register_pytree_node(
                    cls,
                    torch.utils._pytree._dict_flatten,
                    lambda values, context: cls(**torch.utils._pytree._dict_unflatten(values, context)),
                )
            else:
                # 注册 pytree 节点，使用字典扁平化和解扁平化
                torch.utils._pytree.register_pytree_node(
                    cls,
                    torch.utils._pytree._dict_flatten,
                    lambda values, context: cls(**torch.utils._pytree._dict_unflatten(values, context)),
                )
    # 定义数据类的后处理初始化方法
        def __post_init__(self) -> None:
            # 获取当前数据类的所有字段
            class_fields = fields(self)
    
            # 安全性和一致性检查
            if not len(class_fields):
                # 如果没有字段，抛出错误
                raise ValueError(f"{self.__class__.__name__} has no fields.")
    
            # 获取第一个字段的值
            first_field = getattr(self, class_fields[0].name)
            # 检查除了第一个字段外，其他字段是否均为 None
            other_fields_are_none = all(getattr(self, field.name) is None for field in class_fields[1:])
    
            # 如果其他字段均为 None 且第一个字段为字典，进行赋值
            if other_fields_are_none and isinstance(first_field, dict):
                for key, value in first_field.items():
                    # 将字典内容赋值到当前对象
                    self[key] = value
            else:
                # 遍历所有字段并赋值非 None 的字段
                for field in class_fields:
                    v = getattr(self, field.name)
                    if v is not None:
                        # 将非 None 的字段值赋值到当前对象
                        self[field.name] = v
    
        # 定义删除项的方法
        def __delitem__(self, *args, **kwargs):
            # 不允许删除项，抛出异常
            raise Exception(f"You cannot use ``__delitem__`` on a {self.__class__.__name__} instance.")
    
        # 定义设置默认值的方法
        def setdefault(self, *args, **kwargs):
            # 不允许设置默认值，抛出异常
            raise Exception(f"You cannot use ``setdefault`` on a {self.__class__.__name__} instance.")
    
        # 定义弹出项的方法
        def pop(self, *args, **kwargs):
            # 不允许弹出项，抛出异常
            raise Exception(f"You cannot use ``pop`` on a {self.__class__.__name__} instance.")
    
        # 定义更新项的方法
        def update(self, *args, **kwargs):
            # 不允许更新项，抛出异常
            raise Exception(f"You cannot use ``update`` on a {self.__class__.__name__} instance.")
    
        # 定义获取项的方法
        def __getitem__(self, k: Any) -> Any:
            # 如果键是字符串
            if isinstance(k, str):
                # 将当前项转换为字典并返回对应值
                inner_dict = dict(self.items())
                return inner_dict[k]
            else:
                # 如果键不是字符串，返回对应的元组值
                return self.to_tuple()[k]
    
        # 定义设置属性的方法
        def __setattr__(self, name: Any, value: Any) -> None:
            # 如果属性名在键中且值不为 None
            if name in self.keys() and value is not None:
                # 不调用 self.__setitem__ 以避免递归错误
                super().__setitem__(name, value)
            # 设置属性值
            super().__setattr__(name, value)
    
        # 定义设置项的方法
        def __setitem__(self, key, value):
            # 将键值对设置到当前对象中
            super().__setitem__(key, value)
            # 不调用 self.__setattr__ 以避免递归错误
            super().__setattr__(key, value)
    
        # 定义序列化的方法
        def __reduce__(self):
            # 如果当前对象不是数据类
            if not is_dataclass(self):
                # 调用父类的序列化方法
                return super().__reduce__()
            # 获取可调用对象和参数
            callable, _args, *remaining = super().__reduce__()
            # 生成字段的元组
            args = tuple(getattr(self, field.name) for field in fields(self))
            # 返回可调用对象、参数及其他信息
            return callable, args, *remaining
    
        # 定义转换为元组的方法
        def to_tuple(self) -> Tuple[Any, ...]:
            """
            将当前对象转换为一个包含所有非 `None` 属性/键的元组。
            """
            # 返回包含所有键的值的元组
            return tuple(self[k] for k in self.keys())
```