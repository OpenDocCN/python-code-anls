# `.\diffusers\utils\deprecation_utils.py`

```py
# 导入 inspect 模块，用于获取栈帧信息
import inspect
# 导入 warnings 模块，用于发出警告
import warnings
# 从 typing 模块导入所需类型，用于类型注解
from typing import Any, Dict, Optional, Union

# 从 packaging 模块导入 version，用于版本比较
from packaging import version


# 定义 deprecate 函数，用于处理函数参数和属性的弃用警告
def deprecate(*args, take_from: Optional[Union[Dict, Any]] = None, standard_warn=True, stacklevel=2):
    # 导入当前模块的版本
    from .. import __version__

    # 将 take_from 赋值给 deprecated_kwargs，用于存储要检查的参数或属性
    deprecated_kwargs = take_from
    # 初始化一个元组用于存储返回值
    values = ()
    # 如果第一个参数不是元组，将其转换为元组
    if not isinstance(args[0], tuple):
        args = (args,)

    # 遍历每个弃用的属性、版本名和警告信息
    for attribute, version_name, message in args:
        # 检查当前版本是否大于等于弃用版本
        if version.parse(version.parse(__version__).base_version) >= version.parse(version_name):
            # 抛出 ValueError，提示弃用的元组应被移除
            raise ValueError(
                f"The deprecation tuple {(attribute, version_name, message)} should be removed since diffusers'"
                f" version {__version__} is >= {version_name}"
            )

        # 初始化警告消息为 None
        warning = None
        # 如果 deprecated_kwargs 是字典且包含当前属性
        if isinstance(deprecated_kwargs, dict) and attribute in deprecated_kwargs:
            # 从字典中弹出属性值并添加到 values 元组
            values += (deprecated_kwargs.pop(attribute),)
            # 设置警告信息
            warning = f"The `{attribute}` argument is deprecated and will be removed in version {version_name}."
        # 如果 deprecated_kwargs 有该属性
        elif hasattr(deprecated_kwargs, attribute):
            # 获取属性值并添加到 values 元组
            values += (getattr(deprecated_kwargs, attribute),)
            # 设置警告信息
            warning = f"The `{attribute}` attribute is deprecated and will be removed in version {version_name}."
        # 如果 deprecated_kwargs 为 None
        elif deprecated_kwargs is None:
            # 设置警告信息
            warning = f"`{attribute}` is deprecated and will be removed in version {version_name}."

        # 如果有警告信息，发出警告
        if warning is not None:
            # 根据标准警告标志决定是否添加空格
            warning = warning + " " if standard_warn else ""
            # 发出 FutureWarning 警告
            warnings.warn(warning + message, FutureWarning, stacklevel=stacklevel)

    # 如果 deprecated_kwargs 是字典且仍有未处理的项
    if isinstance(deprecated_kwargs, dict) and len(deprecated_kwargs) > 0:
        # 获取当前帧的外部帧信息
        call_frame = inspect.getouterframes(inspect.currentframe())[1]
        # 获取文件名、行号和函数名
        filename = call_frame.filename
        line_number = call_frame.lineno
        function = call_frame.function
        # 获取第一个未处理的键值对
        key, value = next(iter(deprecated_kwargs.items()))
        # 抛出 TypeError，提示收到意外的关键字参数
        raise TypeError(f"{function} in {filename} line {line_number-1} got an unexpected keyword argument `{key}`")

    # 如果没有值返回
    if len(values) == 0:
        return
    # 如果只有一个值返回该值
    elif len(values) == 1:
        return values[0]
    # 如果有多个值返回值元组
    return values
```