# `.\DB-GPT-src\dbgpt\util\annotations.py`

```py
# 引入类型提示中的 Optional 类型
from typing import Optional

# 定义 PublicAPI 装饰器函数，用于标记函数或类作为公共 API
def PublicAPI(*args, **kwargs):
    """Decorator to mark a function or class as a public API.

    Args:
        stability: The stability of the API. Can be "alpha", "beta" or "stable".
            If "alpha", the API is in alpha may come breaking changes before becoming beta.
            If "beta", the API is in beta and may change before becoming stable.
            If "stable", the API will remain backwards compatible with the current major version.
            Defaults to "stable".

    Examples:
        >>> from dbgpt.util.annotations import PublicAPI
        >>> @PublicAPI
        ... def foo():
        ...     pass

        >>> @PublicAPI(stability="beta")
        ... def bar():
        ...     pass

    """
    # 如果只有一个参数且为可调用对象，则返回修饰后的 PublicAPI 装饰器
    if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
        return PublicAPI(stability="stable")(args[0])
    
    # 从 kwargs 中获取 stability 参数的值，若不存在则默认为 "stable"
    stability = None
    if "stability" in kwargs:
        stability = kwargs["stability"]
    if not stability:
        stability = "stable"
    
    # 确保 stability 值在合法的选项中
    assert stability in ["alpha", "beta", "stable"]

    # 定义装饰器函数，根据 stability 修改文档字符串和注解
    def decorator(obj):
        if stability in ["alpha", "beta"]:
            _modify_docstring(
                obj,
                f"**PublicAPI ({stability}):** This API is in {stability} and may change before becoming stable.",
            )
            _modify_annotation(obj, stability)
        return obj

    return decorator


# 定义 DeveloperAPI 装饰器函数，用于标记函数或类作为开发者 API
def DeveloperAPI(*args, **kwargs):
    """Decorator to mark a function or class as a developer API.

    Developer APIs are low-level APIs for advanced users and may change cross major versions.

    Examples:
        >>> from dbgpt.util.annotations import DeveloperAPI
        >>> @DeveloperAPI
        ... def foo():
        ...     pass

    """
    # 如果只有一个参数且为可调用对象，则返回修饰后的 DeveloperAPI 装饰器
    if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
        return DeveloperAPI()(args[0])

    # 定义装饰器函数，修改文档字符串为开发者 API 的描述
    def decorator(obj):
        _modify_docstring(
            obj,
            "**DeveloperAPI:** This API is for advanced users and may change cross major versions.",
        )
        return obj

    return decorator


# 定义 mutable 装饰器函数，标记实例方法会修改实例状态
def mutable(func):
    """Decorator to mark a method of an instance will change the instance state.

    Examples:
        >>> from dbgpt.util.annotations import mutable
        >>> class Foo:
        ...     def __init__(self):
        ...         self.a = 1
        ...
        ...     @mutable
        ...     def change_a(self):
        ...         self.a = 2
        ...

    """
    # 调用 _modify_mutability 函数，将实例方法标记为可修改状态
    _modify_mutability(func, mutability=True)
    return func


# 定义 immutable 装饰器函数，标记实例方法不会修改实例状态
def immutable(func):
    """Decorator to mark a method of an instance will not change the instance state.

    Examples:
        >>> from dbgpt.util.annotations import immutable
        >>> class Foo:
        ...     def __init__(self):
        ...         self.a = 1
        ...
        ...     @immutable
        ...     def get_a(self):
        ...         return self.a
        ...

    """
    # 调用 _modify_mutability 函数，将实例方法标记为不可修改状态
    _modify_mutability(func, mutability=False)
    return func
# 修改对象的文档字符串，将给定消息添加到现有文档字符串的开头
def _modify_docstring(obj, message: Optional[str] = None):
    # 如果没有提供消息，则不进行任何操作，直接返回
    if not message:
        return
    # 如果对象没有现有的文档字符串，则初始化为空字符串
    if not obj.__doc__:
        obj.__doc__ = ""
    # 保存原始的文档字符串内容
    original_doc = obj.__doc__

    # 将文档字符串分割成行
    lines = original_doc.splitlines()

    # 初始化最小缩进为无穷大
    min_indent = float("inf")
    # 遍历文档字符串的每一行（从第二行开始）
    for line in lines[1:]:
        # 去除行首空白后的内容
        stripped = line.lstrip()
        # 如果该行不是全空白行
        if stripped:
            # 计算该行的缩进空格数
            min_indent = min(min_indent, len(line) - len(stripped))

    # 如果没有找到非空白行，则最小缩进设置为0
    if min_indent == float("inf"):
        min_indent = 0
    # 将消息添加到文档字符串的开头，并保持原始文档字符串的缩进格式
    indented_message = message.rstrip() + "\n" + (" " * min_indent)
    # 更新对象的文档字符串
    obj.__doc__ = indented_message + original_doc


# 修改对象的注解，设置对象的公共稳定性和注解属性
def _modify_annotation(obj, stability) -> None:
    # 如果提供了稳定性参数，则设置对象的公共稳定性属性
    if stability:
        obj._public_stability = stability
    # 如果对象具有 "__name__" 属性，则将对象自身的名称赋给 "_annotated" 属性
    if hasattr(obj, "__name__"):
        obj._annotated = obj.__name__


# 修改对象的可变性，设置对象的可变性属性
def _modify_mutability(obj, mutability) -> None:
    # 设置对象的可变性属性为提供的可变性参数值
    obj._mutability = mutability
```