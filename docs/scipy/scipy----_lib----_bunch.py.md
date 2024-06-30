# `D:\src\scipysrc\scipy\scipy\_lib\_bunch.py`

```
import sys as _sys
from keyword import iskeyword as _iskeyword


def _validate_names(typename, field_names, extra_field_names):
    """
    Ensure that all the given names are valid Python identifiers that
    do not start with '_'.  Also check that there are no duplicates
    among field_names + extra_field_names.
    """
    # Iterate over typename, field_names, and extra_field_names to validate each name
    for name in [typename] + field_names + extra_field_names:
        # Check if the name is not a string, raise TypeError if it's not
        if not isinstance(name, str):
            raise TypeError('typename and all field names must be strings')
        # Check if the name is a valid Python identifier, raise ValueError if it's not
        if not name.isidentifier():
            raise ValueError('typename and all field names must be valid '
                             f'identifiers: {name!r}')
        # Check if the name is a Python keyword, raise ValueError if it is
        if _iskeyword(name):
            raise ValueError('typename and all field names cannot be a '
                             f'keyword: {name!r}')

    seen = set()
    # Check for duplicates and names starting with underscore in field_names and extra_field_names
    for name in field_names + extra_field_names:
        # Raise ValueError if the name starts with underscore
        if name.startswith('_'):
            raise ValueError('Field names cannot start with an underscore: '
                             f'{name!r}')
        # Raise ValueError if the name is already seen (duplicate)
        if name in seen:
            raise ValueError(f'Duplicate field name: {name!r}')
        seen.add(name)


# Note: This code is adapted from CPython:Lib/collections/__init__.py
def _make_tuple_bunch(typename, field_names, extra_field_names=None,
                      module=None):
    """
    Create a namedtuple-like class with additional attributes.

    This function creates a subclass of tuple that acts like a namedtuple
    and that has additional attributes.

    The additional attributes are listed in `extra_field_names`.  The
    values assigned to these attributes are not part of the tuple.

    The reason this function exists is to allow functions in SciPy
    that currently return a tuple or a namedtuple to returned objects
    that have additional attributes, while maintaining backwards
    compatibility.

    This should only be used to enhance *existing* functions in SciPy.
    New functions are free to create objects as return values without
    having to maintain backwards compatibility with an old tuple or
    namedtuple return value.

    Parameters
    ----------
    typename : str
        The name of the type.
    field_names : list of str
        List of names of the values to be stored in the tuple. These names
        will also be attributes of instances, so the values in the tuple
        can be accessed by indexing or as attributes.  At least one name
        is required.  See the Notes for additional restrictions.
    extra_field_names : list of str, optional
        List of names of values that will be stored as attributes of the
        object.  See the notes for additional restrictions.

    Returns
    -------
    cls : type
        The new class.

    Notes
    -----
    There are restrictions on the names that may be used in `field_names`
    and `extra_field_names`:

    * The names must be unique--no duplicates allowed.

    """
    # Function implementation to create a named tuple-like class with extra attributes
    pass  # Placeholder for actual implementation, not shown in this snippet
    # 如果字段名列表为空，则抛出值错误异常
    if len(field_names) == 0:
        raise ValueError('field_names must contain at least one name')

    # 如果额外字段名列表为None，则将其设为空列表
    if extra_field_names is None:
        extra_field_names = []
    
    # 校验类型名、字段名和额外字段名的合法性
    _validate_names(typename, field_names, extra_field_names)

    # 将类型名转换为内部标识符对象
    typename = _sys.intern(str(typename))
    # 将字段名列表转换为内部标识符对象的元组
    field_names = tuple(map(_sys.intern, field_names))
    # 将额外字段名列表转换为内部标识符对象的元组
    extra_field_names = tuple(map(_sys.intern, extra_field_names))

    # 合并字段名和额外字段名列表，形成所有字段名的列表
    all_names = field_names + extra_field_names
    # 构建用于方法参数的字符串表示
    arg_list = ', '.join(field_names)
    # 构建用于完整字段名列表的字符串表示
    full_list = ', '.join(all_names)
    # 构建用于对象表示形式的字符串格式化模板
    repr_fmt = ''.join(('(',
                        ', '.join(f'{name}=%({name})r' for name in all_names),
                        ')'))
    # 获取内置tuple类的实例化方法
    tuple_new = tuple.__new__
    # 设置内置数据结构对象的别名
    _dict, _tuple, _zip = dict, tuple, zip

    # 创建要添加到类命名空间的所有命名元组方法
# 定义特殊方法 __new__，用于创建新的命名元组实例
def __new__(_cls, {arg_list}, **extra_fields):
    return _tuple_new(_cls, ({arg_list},))

# 定义特殊方法 __init__，用于初始化命名元组实例
def __init__(self, {arg_list}, **extra_fields):
    # 检查额外字段是否缺失必要的关键字参数
    for key in self._extra_fields:
        if key not in extra_fields:
            raise TypeError("missing keyword argument '%s'" % (key,))
    # 检查额外字段是否包含未预期的关键字参数
    for key, val in extra_fields.items():
        if key not in self._extra_fields:
            raise TypeError("unexpected keyword argument '%s'" % (key,))
        self.__dict__[key] = val

# 定义特殊方法 __setattr__，用于设置命名元组实例的属性
def __setattr__(self, key, val):
    if key in {repr(field_names)}:
        # 如果属性名在字段名中，则抛出属性错误
        raise AttributeError("can't set attribute %r of class %r"
                             % (key, self.__class__.__name__))
    else:
        self.__dict__[key] = val

# 删除 arg_list 变量
del arg_list

# 构建 exec 函数所需的命名空间
namespace = {'_tuple_new': tuple_new,
             '__builtins__': dict(TypeError=TypeError,
                                  AttributeError=AttributeError),
             '__name__': f'namedtuple_{typename}'}

# 执行字符串 s 中的代码，并将结果存储到命名空间中
exec(s, namespace)

# 获取并设置特殊方法的引用
__new__ = namespace['__new__']
__new__.__doc__ = f'Create new instance of {typename}({full_list})'
__init__ = namespace['__init__']
__init__.__doc__ = f'Instantiate instance of {typename}({full_list})'
__setattr__ = namespace['__setattr__']

# 定义特殊方法 __repr__，返回命名元组实例的格式化字符串表示
def __repr__(self):
    'Return a nicely formatted representation string'
    return self.__class__.__name__ + repr_fmt % self._asdict()

# 定义特殊方法 _asdict，返回命名元组实例的字段名与值的映射字典
def _asdict(self):
    'Return a new dict which maps field names to their values.'
    out = _dict(_zip(self._fields, self))
    out.update(self.__dict__)
    return out

# 定义特殊方法 __getnewargs_ex__，返回命名元组实例的纯元组形式和其 __dict__ 属性
def __getnewargs_ex__(self):
    'Return self as a plain tuple.  Used by copy and pickle.'
    return _tuple(self), self.__dict__

# 修改特殊方法的限定名称，帮助进行内省和调试
for method in (__new__, __repr__, _asdict, __getnewargs_ex__):
    method.__qualname__ = f'{typename}.{method.__name__}'

# 构建类的命名空间字典，并使用 type() 函数创建结果类
class_namespace = {
    '__doc__': f'{typename}({full_list})',
    '_fields': field_names,
    '__new__': __new__,
    '__init__': __init__,
    '__repr__': __repr__,
    '__setattr__': __setattr__,
    '_asdict': _asdict,
    '_extra_fields': extra_field_names,
    '__getnewargs_ex__': __getnewargs_ex__,
}

# 创建属性名到索引值的映射关系
for index, name in enumerate(field_names):
    def _get(self, index=index):
        return self[index]
    class_namespace[name] = property(_get)

# 对额外字段创建属性访问器
for name in extra_field_names:
    def _get(self, name=name):
        return self.__dict__[name]
    class_namespace[name] = property(_get)

# 使用 type() 函数创建最终的命名元组类
result = type(typename, (tuple,), class_namespace)

# 为了使 pickle 正常工作，需要设置 __module__ 变量为创建命名元组的帧
# 在不支持 sys._getframe 的环境中跳过此步骤（例如 Jython）
    # 如果 module 参数为 None，则尝试获取调用者的模块名，用作 result 对象的 __module__ 属性
    if module is None:
        try:
            # 获取调用者的模块名，如果获取失败则忽略异常继续执行
            module = _sys._getframe(1).f_globals.get('__name__', '__main__')
        except (AttributeError, ValueError):
            pass
    
    # 如果 module 不为 None，则将 result 对象和 __new__ 方法的 __module__ 属性设置为 module 的值
    if module is not None:
        result.__module__ = module
        __new__.__module__ = module

    # 返回 result 对象
    return result
```