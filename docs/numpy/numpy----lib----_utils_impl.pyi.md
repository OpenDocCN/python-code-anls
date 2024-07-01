# `.\numpy\numpy\lib\_utils_impl.pyi`

```py
# 导入必要的模块和类型声明
from typing import (
    Any,
    TypeVar,
    Protocol,
)

# 导入 numpy 库中的 issubdtype 函数并重命名为 issubdtype
from numpy._core.numerictypes import (
    issubdtype as issubdtype,
)

# 定义一个逆变（contravariant=True）类型变量 _T_contra
_T_contra = TypeVar("_T_contra", contravariant=True)

# 定义一个协议（Protocol），表示支持写入操作的文件类对象
class _SupportsWrite(Protocol[_T_contra]):
    # 声明支持接受字符串写入的 write 方法
    def write(self, s: _T_contra, /) -> Any: ...

# 定义模块中公开的标识符列表 __all__，限制模块的导入时仅导入这些标识符
__all__: list[str]

# 返回当前平台相关的包含文件的目录路径
def get_include() -> str: ...

# 打印对象的信息到指定输出，允许控制最大输出宽度和顶级对象
def info(
    object: object = ...,
    maxwidth: int = ...,
    output: None | _SupportsWrite[str] = ...,
    toplevel: str = ...,
) -> None: ...

# 将对象的源代码输出到指定位置，如果未指定输出，则输出到默认位置
def source(
    object: object,
    output: None | _SupportsWrite[str] = ...,
) -> None: ...

# 显示当前运行时的信息，通常用于调试和性能分析
def show_runtime() -> None: ...
```