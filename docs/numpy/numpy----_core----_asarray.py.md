# `.\numpy\numpy\_core\_asarray.py`

```py
"""
Functions in the ``as*array`` family that promote array-likes into arrays.

`require` fits this category despite its name not matching this pattern.
"""
# 从overrides模块导入必要的函数和类
from .overrides import (
    array_function_dispatch,
    set_array_function_like_doc,
    set_module,
)
# 从multiarray模块导入array和asanyarray函数
from .multiarray import array, asanyarray

# 定义模块公开的函数名列表
__all__ = ["require"]

# 定义可能的标志和其对应的规范化标志
POSSIBLE_FLAGS = {
    'C': 'C', 'C_CONTIGUOUS': 'C', 'CONTIGUOUS': 'C',
    'F': 'F', 'F_CONTIGUOUS': 'F', 'FORTRAN': 'F',
    'A': 'A', 'ALIGNED': 'A',
    'W': 'W', 'WRITEABLE': 'W',
    'O': 'O', 'OWNDATA': 'O',
    'E': 'E', 'ENSUREARRAY': 'E'
}

# 使用装饰器设定文档字符串的数组函数行为
@set_array_function_like_doc
# 设置模块名为'numpy'
@set_module('numpy')
def require(a, dtype=None, requirements=None, *, like=None):
    """
    Return an ndarray of the provided type that satisfies requirements.

    This function is useful to be sure that an array with the correct flags
    is returned for passing to compiled code (perhaps through ctypes).

    Parameters
    ----------
    a : array_like
       The object to be converted to a type-and-requirement-satisfying array.
    dtype : data-type
       The required data-type. If None preserve the current dtype. If your
       application requires the data to be in native byteorder, include
       a byteorder specification as a part of the dtype specification.
    requirements : str or sequence of str
       The requirements list can be any of the following

       * 'F_CONTIGUOUS' ('F') - ensure a Fortran-contiguous array
       * 'C_CONTIGUOUS' ('C') - ensure a C-contiguous array
       * 'ALIGNED' ('A')      - ensure a data-type aligned array
       * 'WRITEABLE' ('W')    - ensure a writable array
       * 'OWNDATA' ('O')      - ensure an array that owns its own data
       * 'ENSUREARRAY', ('E') - ensure a base array, instead of a subclass
    ${ARRAY_FUNCTION_LIKE}

        .. versionadded:: 1.20.0

    Returns
    -------
    out : ndarray
        Array with specified requirements and type if given.

    See Also
    --------
    asarray : Convert input to an ndarray.
    asanyarray : Convert to an ndarray, but pass through ndarray subclasses.
    ascontiguousarray : Convert input to a contiguous array.
    asfortranarray : Convert input to an ndarray with column-major
                     memory order.
    ndarray.flags : Information about the memory layout of the array.

    Notes
    -----
    The returned array will be guaranteed to have the listed requirements
    by making a copy if needed.

    Examples
    --------
    >>> x = np.arange(6).reshape(2,3)
    >>> x.flags
      C_CONTIGUOUS : True
      F_CONTIGUOUS : False
      OWNDATA : False
      WRITEABLE : True
      ALIGNED : True
      WRITEBACKIFCOPY : False

    >>> y = np.require(x, dtype=np.float32, requirements=['A', 'O', 'W', 'F'])
    >>> y.flags
      C_CONTIGUOUS : False
      F_CONTIGUOUS : True
      OWNDATA : True
      WRITEABLE : True
      ALIGNED : True
      WRITEBACKIFCOPY : False

    """
    # 如果参数 like 不为 None，则调用 _require_with_like 函数，并传入相关参数
    if like is not None:
        return _require_with_like(
            like,
            a,
            dtype=dtype,
            requirements=requirements,
        )
    
    # 如果 requirements 为空，则将 a 转换为任意数组，使用指定的数据类型 dtype
    if not requirements:
        return asanyarray(a, dtype=dtype)
    
    # 将 requirements 转换为大写，并从 POSSIBLE_FLAGS 字典中获取相应的标志集合
    requirements = {POSSIBLE_FLAGS[x.upper()] for x in requirements}
    
    # 如果集合中包含 'E'，则移除 'E' 并设置 subok 为 False，否则设置为 True
    if 'E' in requirements:
        requirements.remove('E')
        subok = False
    else:
        subok = True
    
    # 默认设置数组的存储顺序为 'A'（任意顺序）
    order = 'A'
    
    # 如果 requirements 中包含 {'C', 'F'}，则抛出 ValueError 异常，因为不能同时指定 "C" 和 "F" 顺序
    if requirements >= {'C', 'F'}:
        raise ValueError('Cannot specify both "C" and "F" order')
    elif 'F' in requirements:
        # 如果 requirements 中包含 'F'，则设置数组的存储顺序为 'F'，并移除 'F'
        order = 'F'
        requirements.remove('F')
    elif 'C' in requirements:
        # 如果 requirements 中包含 'C'，则设置数组的存储顺序为 'C'，并移除 'C'
        order = 'C'
        requirements.remove('C')
    
    # 使用指定的数据类型 dtype 和存储顺序 order，创建数组 arr，并根据 subok 参数决定是否允许子类化
    arr = array(a, dtype=dtype, order=order, copy=None, subok=subok)
    
    # 遍历 requirements 中的每个属性，如果 arr 不满足该属性的标志，则返回 arr 的副本，并设置存储顺序 order
    for prop in requirements:
        if not arr.flags[prop]:
            return arr.copy(order)
    
    # 如果所有 requirements 中的属性都被满足，则返回原始的 arr 数组
    return arr
# 使用array_function_dispatch()函数装饰require函数，并将结果赋值给_require_with_like变量
_require_with_like = array_function_dispatch()(require)
```