# `D:\src\scipysrc\matplotlib\lib\matplotlib\_docstring.pyi`

```
# 导入必要的类型和函数签名定义
from typing import Any, Callable, TypeVar, overload

# 定义类型变量 _T
_T = TypeVar('_T')

# 装饰器函数 kwarg_doc，用于为函数添加关键字参数文档
def kwarg_doc(text: str) -> Callable[[_T], _T]: ...

# 定义 Substitution 类，支持多态构造函数和函数调用
class Substitution:
    @overload
    def __init__(self, *args: str): ...
    @overload
    def __init__(self, **kwargs: str): ...
    def __call__(self, func: _T) -> _T: ...
    def update(self, *args, **kwargs): ...  # type: ignore[no-untyped-def]

# 定义 _ArtistKwdocLoader 类，继承自 dict[str, str]，用于处理缺失的键值查找
class _ArtistKwdocLoader(dict[str, str]):
    def __missing__(self, key: str) -> str: ...

# 定义 _ArtistPropertiesSubstitution 类，继承自 Substitution 类
class _ArtistPropertiesSubstitution(Substitution):
    def __init__(self) -> None: ...
    def __call__(self, obj: _T) -> _T: ...

# 函数 copy，接受任意类型的参数 source，返回一个装饰器
def copy(source: Any) -> Callable[[_T], _T]: ...

# 定义全局变量 dedent_interpd 和 interpd，类型为 _ArtistPropertiesSubstitution
dedent_interpd: _ArtistPropertiesSubstitution
interpd: _ArtistPropertiesSubstitution
```