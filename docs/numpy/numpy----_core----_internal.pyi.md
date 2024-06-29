# `D:\src\scipysrc\numpy\numpy\_core\_internal.pyi`

```py
# 从 typing 模块中导入需要的类型：Any（任意类型）、TypeVar（类型变量）、overload（函数重载）、Generic（泛型）
import ctypes as ct

# 从 numpy.typing 模块中导入 NDArray 类型
from numpy.typing import NDArray
# 从 numpy.ctypeslib 模块中导入 c_intp 类型
from numpy.ctypeslib import c_intp

# 创建一个类型变量 _CastT，其上界为 ct._CanCastTo 类型，这是从 ctypes.cast 处复制而来的
_CastT = TypeVar("_CastT", bound=ct._CanCastTo)

# 创建一个类型变量 _CT，其上界为 ct._CData 类型
_CT = TypeVar("_CT", bound=ct._CData)

# 创建一个类型变量 _PT，其上界为 int 类型
_PT = TypeVar("_PT", bound=int)

# 定义一个泛型类 _ctypes，使用了类型变量 _PT
class _ctypes(Generic[_PT]):
    
    # 以下是函数重载的定义，为了不同的输入参数类型返回不同的类型
    @overload
    def __new__(cls, array: NDArray[Any], ptr: None = ...) -> _ctypes[None]: ...
    
    @overload
    def __new__(cls, array: NDArray[Any], ptr: _PT) -> _ctypes[_PT]: ...
    
    # 属性定义：返回 _PT 类型的数据
    @property
    def data(self) -> _PT: ...
    
    # 属性定义：返回 c_intp 类型的数组，表示形状
    @property
    def shape(self) -> ct.Array[c_intp]: ...
    
    # 属性定义：返回 c_intp 类型的数组，表示步幅
    @property
    def strides(self) -> ct.Array[c_intp]: ...
    
    # 属性定义：返回 ct.c_void_p 类型的对象，作为参数使用
    @property
    def _as_parameter_(self) -> ct.c_void_p: ...
    
    # 方法定义：将当前对象转换为 obj 指定的类型 _CastT
    def data_as(self, obj: type[_CastT]) -> _CastT: ...
    
    # 方法定义：将当前对象的形状转换为 obj 指定的类型 _CT 的数组
    def shape_as(self, obj: type[_CT]) -> ct.Array[_CT]: ...
    
    # 方法定义：将当前对象的步幅转换为 obj 指定的类型 _CT 的数组
    def strides_as(self, obj: type[_CT]) -> ct.Array[_CT]: ...
```