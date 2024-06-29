# `.\numpy\numpy\_core\_dtype_ctypes.py`

```py
"""
Conversion from ctypes to dtype.

In an ideal world, we could achieve this through the PEP3118 buffer protocol,
something like::

    def dtype_from_ctypes_type(t):
        # needed to ensure that the shape of `t` is within memoryview.format
        class DummyStruct(ctypes.Structure):
            _fields_ = [('a', t)]

        # empty to avoid memory allocation
        ctype_0 = (DummyStruct * 0)()
        mv = memoryview(ctype_0)

        # convert the struct, and slice back out the field
        return _dtype_from_pep3118(mv.format)['a']

Unfortunately, this fails because:

* ctypes cannot handle length-0 arrays with PEP3118 (bpo-32782)
* PEP3118 cannot represent unions, but both numpy and ctypes can
* ctypes cannot handle big-endian structs with PEP3118 (bpo-32780)
"""

# We delay-import ctypes for distributions that do not include it.
# While this module is not used unless the user passes in ctypes
# members, it is eagerly imported from numpy/_core/__init__.py.
import numpy as np


def _from_ctypes_array(t):
    """
    Convert a ctypes array to a numpy dtype.

    Args:
        t: ctypes array type

    Returns:
        numpy dtype corresponding to the ctypes array type
    """
    return np.dtype((dtype_from_ctypes_type(t._type_), (t._length_,)))


def _from_ctypes_structure(t):
    """
    Convert a ctypes structure to a numpy dtype.

    Args:
        t: ctypes structure type

    Raises:
        TypeError: If ctypes bitfields are encountered

    Returns:
        numpy dtype corresponding to the ctypes structure type
    """
    for item in t._fields_:
        if len(item) > 2:
            raise TypeError(
                "ctypes bitfields have no dtype equivalent")

    if hasattr(t, "_pack_"):
        import ctypes
        formats = []
        offsets = []
        names = []
        current_offset = 0
        for fname, ftyp in t._fields_:
            names.append(fname)
            formats.append(dtype_from_ctypes_type(ftyp))
            # Each type has a default offset, this is platform dependent
            # for some types.
            effective_pack = min(t._pack_, ctypes.alignment(ftyp))
            current_offset = (
                (current_offset + effective_pack - 1) // effective_pack
            ) * effective_pack
            offsets.append(current_offset)
            current_offset += ctypes.sizeof(ftyp)

        return np.dtype(dict(
            formats=formats,
            offsets=offsets,
            names=names,
            itemsize=ctypes.sizeof(t)))
    else:
        fields = []
        for fname, ftyp in t._fields_:
            fields.append((fname, dtype_from_ctypes_type(ftyp)))

        # by default, ctypes structs are aligned
        return np.dtype(fields, align=True)


def _from_ctypes_scalar(t):
    """
    Convert a ctypes scalar type to a numpy dtype.

    Args:
        t: ctypes scalar type

    Returns:
        numpy dtype corresponding to the ctypes scalar type
    """
    if getattr(t, '__ctype_be__', None) is t:
        return np.dtype('>' + t._type_)
    elif getattr(t, '__ctype_le__', None) is t:
        return np.dtype('<' + t._type_)
    else:
        return np.dtype(t._type_)


def _from_ctypes_union(t):
    """
    Convert a ctypes union to a numpy dtype.

    Args:
        t: ctypes union type
    """
    import ctypes
    formats = []
    offsets = []
    names = []
    for fname, ftyp in t._fields_:
        names.append(fname)
        formats.append(dtype_from_ctypes_type(ftyp))
        offsets.append(0)  # Union fields are offset to 0
    # 返回一个 NumPy 数据类型对象，该对象基于传入的参数构建
    return np.dtype(dict(
        formats=formats,    # 设置数据类型的格式列表
        offsets=offsets,    # 设置数据类型的偏移量列表
        names=names,        # 设置数据类型的字段名列表
        itemsize=ctypes.sizeof(t)))  # 设置数据类型的字节大小为给定类型 t 的大小
# 根据给定的 ctypes 类型 t 构造一个 dtype 对象
def dtype_from_ctypes_type(t):
    # 导入 _ctypes 模块，用于处理 ctypes 相关操作
    import _ctypes
    # 如果 t 是 _ctypes.Array 的子类，则调用 _from_ctypes_array 函数处理
    if issubclass(t, _ctypes.Array):
        return _from_ctypes_array(t)
    # 如果 t 是 _ctypes._Pointer 的子类，则抛出 TypeError
    elif issubclass(t, _ctypes._Pointer):
        raise TypeError("ctypes pointers have no dtype equivalent")
    # 如果 t 是 _ctypes.Structure 的子类，则调用 _from_ctypes_structure 函数处理
    elif issubclass(t, _ctypes.Structure):
        return _from_ctypes_structure(t)
    # 如果 t 是 _ctypes.Union 的子类，则调用 _from_ctypes_union 函数处理
    elif issubclass(t, _ctypes.Union):
        return _from_ctypes_union(t)
    # 如果 t 的 _type_ 属性是字符串，则调用 _from_ctypes_scalar 函数处理
    elif isinstance(getattr(t, '_type_', None), str):
        return _from_ctypes_scalar(t)
    # 如果 t 不属于以上任何一种类型，则抛出 NotImplementedError
    else:
        raise NotImplementedError(
            "Unknown ctypes type {}".format(t.__name__))
```