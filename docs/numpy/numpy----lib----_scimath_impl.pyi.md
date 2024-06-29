# `D:\src\scipysrc\numpy\numpy\lib\_scimath_impl.pyi`

```
# 导入必要的类型提示和函数定义
from typing import overload, Any

# 导入复数浮点数相关模块
from numpy import complexfloating

# 导入类型提示相关模块
from numpy._typing import (
    NDArray,
    _ArrayLikeFloat_co,
    _ArrayLikeComplex_co,
    _ComplexLike_co,
    _FloatLike_co,
)

# 指定在此模块中可导出的公共接口列表
__all__: list[str]

# 平方根函数的函数重载，根据不同输入类型返回不同类型的值
@overload
def sqrt(x: _FloatLike_co) -> Any: ...

@overload
def sqrt(x: _ComplexLike_co) -> complexfloating[Any, Any]: ...

@overload
def sqrt(x: _ArrayLikeFloat_co) -> NDArray[Any]: ...

@overload
def sqrt(x: _ArrayLikeComplex_co) -> NDArray[complexfloating[Any, Any]]: ...

# 对数函数的函数重载，根据不同输入类型返回不同类型的值
@overload
def log(x: _FloatLike_co) -> Any: ...

@overload
def log(x: _ComplexLike_co) -> complexfloating[Any, Any]: ...

@overload
def log(x: _ArrayLikeFloat_co) -> NDArray[Any]: ...

@overload
def log(x: _ArrayLikeComplex_co) -> NDArray[complexfloating[Any, Any]]: ...

# 以10为底的对数函数的函数重载，根据不同输入类型返回不同类型的值
@overload
def log10(x: _FloatLike_co) -> Any: ...

@overload
def log10(x: _ComplexLike_co) -> complexfloating[Any, Any]: ...

@overload
def log10(x: _ArrayLikeFloat_co) -> NDArray[Any]: ...

@overload
def log10(x: _ArrayLikeComplex_co) -> NDArray[complexfloating[Any, Any]]: ...

# 以2为底的对数函数的函数重载，根据不同输入类型返回不同类型的值
@overload
def log2(x: _FloatLike_co) -> Any: ...

@overload
def log2(x: _ComplexLike_co) -> complexfloating[Any, Any]: ...

@overload
def log2(x: _ArrayLikeFloat_co) -> NDArray[Any]: ...

@overload
def log2(x: _ArrayLikeComplex_co) -> NDArray[complexfloating[Any, Any]]: ...

# 自定义底数的对数函数的函数重载，根据不同输入类型返回不同类型的值
@overload
def logn(n: _FloatLike_co, x: _FloatLike_co) -> Any: ...

@overload
def logn(n: _ComplexLike_co, x: _ComplexLike_co) -> complexfloating[Any, Any]: ...

@overload
def logn(n: _ArrayLikeFloat_co, x: _ArrayLikeFloat_co) -> NDArray[Any]: ...

@overload
def logn(n: _ArrayLikeComplex_co, x: _ArrayLikeComplex_co) -> NDArray[complexfloating[Any, Any]]: ...

# 幂函数的函数重载，根据不同输入类型返回不同类型的值
@overload
def power(x: _FloatLike_co, p: _FloatLike_co) -> Any: ...

@overload
def power(x: _ComplexLike_co, p: _ComplexLike_co) -> complexfloating[Any, Any]: ...

@overload
def power(x: _ArrayLikeFloat_co, p: _ArrayLikeFloat_co) -> NDArray[Any]: ...

@overload
def power(x: _ArrayLikeComplex_co, p: _ArrayLikeComplex_co) -> NDArray[complexfloating[Any, Any]]: ...

# 反余弦函数的函数重载，根据不同输入类型返回不同类型的值
@overload
def arccos(x: _FloatLike_co) -> Any: ...

@overload
def arccos(x: _ComplexLike_co) -> complexfloating[Any, Any]: ...

@overload
def arccos(x: _ArrayLikeFloat_co) -> NDArray[Any]: ...

@overload
def arccos(x: _ArrayLikeComplex_co) -> NDArray[complexfloating[Any, Any]]: ...

# 反正弦函数的函数重载，根据不同输入类型返回不同类型的值
@overload
def arcsin(x: _FloatLike_co) -> Any: ...

@overload
def arcsin(x: _ComplexLike_co) -> complexfloating[Any, Any]: ...

@overload
def arcsin(x: _ArrayLikeFloat_co) -> NDArray[Any]: ...

@overload
def arcsin(x: _ArrayLikeComplex_co) -> NDArray[complexfloating[Any, Any]]: ...

# 反双曲正切函数的函数重载，根据不同输入类型返回不同类型的值
@overload
def arctanh(x: _FloatLike_co) -> Any: ...

@overload
def arctanh(x: _ComplexLike_co) -> complexfloating[Any, Any]: ...

@overload
def arctanh(x: _ArrayLikeFloat_co) -> NDArray[Any]: ...

@overload
def arctanh(x: _ArrayLikeComplex_co) -> NDArray[complexfloating[Any, Any]]: ...
```