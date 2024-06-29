# `D:\src\scipysrc\matplotlib\lib\matplotlib\_api\__init__.pyi`

```
# 导入必要的模块和类
from collections.abc import Callable, Generator, Mapping, Sequence
from typing import Any, Iterable, TypeVar, overload
from numpy.typing import NDArray

# 导入需要的函数和类别名
from .deprecation import (
    deprecated as deprecated,  # 标记为已弃用的函数或类的别名
    warn_deprecated as warn_deprecated,  # 发出已弃用警告的函数的别名
    rename_parameter as rename_parameter,  # 重命名参数的函数的别名
    delete_parameter as delete_parameter,  # 删除参数的函数的别名
    make_keyword_only as make_keyword_only,  # 将参数强制成关键字参数的函数的别名
    deprecate_method_override as deprecate_method_override,  # 标记方法覆盖为已弃用的函数的别名
    deprecate_privatize_attribute as deprecate_privatize_attribute,  # 标记私有属性为已弃用的函数的别名
    suppress_matplotlib_deprecation_warning as suppress_matplotlib_deprecation_warning,  # 抑制 Matplotlib 已弃用警告的函数的别名
    MatplotlibDeprecationWarning as MatplotlibDeprecationWarning,  # Matplotlib 已弃用警告的别名
)

# 定义一个类型变量 _T
_T = TypeVar("_T")

# 定义一个类属性装饰器 classproperty
class classproperty(Any):
    def __init__(
        self,
        fget: Callable[[_T], Any],  # 用于获取属性值的方法
        fset: None = ...,  # 未指定设置方法
        fdel: None = ...,  # 未指定删除方法
        doc: str | None = None,  # 文档字符串，默认为空
    ): ...

    # 当 Python 版本 >= 3.9 时，使用 Self 替代 return
    @overload
    def __get__(self, instance: None, owner: None) -> classproperty: ...

    @overload
    def __get__(self, instance: object, owner: type[object]) -> Any: ...

    @property
    def fget(self) -> Callable[[_T], Any]:  # 返回获取属性值的方法
        ...

# 检查对象是否为指定类型或类型元组的实例
def check_isinstance(
    types: type | tuple[type | None, ...],  # 要检查的类型或类型元组
    /,  # 强制位置参数的分隔符
    **kwargs: Any  # 其他关键字参数
) -> None: ...

# 检查值是否在指定序列中
def check_in_list(
    values: Sequence[Any],  # 要检查的值的序列
    /,  # 强制位置参数的分隔符
    *,  # 强制关键字参数的分隔符
    _print_supported_values: bool = ...,  # 是否打印支持的值
    **kwargs: Any  # 其他关键字参数
) -> None: ...

# 检查数组的形状是否符合要求
def check_shape(shape: tuple[int | None, ...],  # 要检查的数组形状
    /,  # 强制位置参数的分隔符
    **kwargs: NDArray  # 其他关键字参数
) -> None: ...

# 检查映射对象中的键是否存在并返回对应的值
def check_getitem(mapping: Mapping[Any, Any],  # 要检查的映射对象
    /,  # 强制位置参数的分隔符
    **kwargs: Any  # 其他关键字参数
) -> Any: ...

# 返回一个函数，用于获取指定类的指定属性值
def caching_module_getattr(cls: type) -> Callable[[str], Any]: ...

# 定义一个函数装饰器，用于为类定义别名
@overload
def define_aliases(
    alias_d: dict[str, list[str]],  # 别名字典，键为原始名称，值为别名列表
    cls: None = ...  # 要定义别名的类（可选）
) -> Callable[[type[_T]], type[_T]]: ...

@overload
def define_aliases(
    alias_d: dict[str, list[str]],  # 别名字典，键为原始名称，值为别名列表
    cls: type[_T]  # 要定义别名的类
) -> type[_T]: ...

# 选择与给定参数匹配的函数签名并返回结果
def select_matching_signature(
    funcs: list[Callable],  # 要选择的函数列表
    *args: Any,  # 位置参数
    **kwargs: Any  # 关键字参数
) -> Any: ...

# 报告参数数量错误
def nargs_error(name: str,  # 参数名
    takes: int | str,  # 预期参数数量或描述
    given: int  # 实际参数数量
) -> TypeError: ...

# 报告关键字参数错误
def kwarg_error(name: str,  # 参数名
    kw: str | Iterable[str]  # 期望的关键字参数或描述
) -> TypeError: ...

# 生成给定类的递归子类生成器
def recursive_subclasses(cls: type) -> Generator[type, None, None]: ...

# 发出外部警告
def warn_external(
    message: str | Warning,  # 警告消息或警告对象
    category: type[Warning] | None = ...  # 警告类型（可选）
) -> None: ...
```