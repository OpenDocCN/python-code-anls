# `.\DB-GPT-src\dbgpt\util\module_utils.py`

```py
# 从 importlib 模块中导入 import_module 函数
# 从 typing 模块中导入 Type 类型
from importlib import import_module
from typing import Type

# 根据字符串导入模块
def import_from_string(module_path: str, ignore_import_error: bool = False):
    # 尝试将模块路径和类名分开
    try:
        module_path, class_name = module_path.rsplit(".", 1)
    except ValueError:
        raise ImportError(f"{module_path} doesn't look like a module path")
    # 导入模块
    module = import_module(module_path)

    # 尝试获取类对象
    try:
        return getattr(module, class_name)
    except AttributeError:
        # 如果忽略导入错误，则返回 None
        if ignore_import_error:
            return None
        # 抛出 ImportError 异常
        raise ImportError(
            f'Module "{module_path}" does not define a "{class_name}" attribute/class'
        )

# 根据字符串导入模块，并检查是否为指定父类的子类
def import_from_checked_string(module_path: str, supper_cls: Type):
    # 根据字符串导入类
    cls = import_from_string(module_path)
    # 检查导入的类是否为指定父类的子类
    if not issubclass(cls, supper_cls):
        raise ImportError(
            f'Module "{module_path}" does not the subclass of {str(supper_cls)}'
        )
    return cls
```