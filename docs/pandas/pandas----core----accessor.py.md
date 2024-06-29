# `D:\src\scipysrc\pandas\pandas\core\accessor.py`

```
"""
accessor.py contains base classes for implementing accessor properties
that can be mixed into or pinned onto other pandas classes.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    final,
)
import warnings

from pandas.util._decorators import doc
from pandas.util._exceptions import find_stack_level

if TYPE_CHECKING:
    from collections.abc import Callable

    from pandas._typing import TypeT

    from pandas import Index
    from pandas.core.generic import NDFrame


class DirNamesMixin:
    _accessors: set[str] = set()
    _hidden_attrs: frozenset[str] = frozenset()

    @final
    def _dir_deletions(self) -> set[str]:
        """
        Delete unwanted __dir__ for this object.
        """
        return self._accessors | self._hidden_attrs

    def _dir_additions(self) -> set[str]:
        """
        Add additional __dir__ for this object.
        """
        return {accessor for accessor in self._accessors if hasattr(self, accessor)}

    def __dir__(self) -> list[str]:
        """
        Provide method name lookup and completion.

        Notes
        -----
        Only provide 'public' methods.
        """
        rv = set(super().__dir__())
        rv = (rv - self._dir_deletions()) | self._dir_additions()
        return sorted(rv)


class PandasDelegate:
    """
    Abstract base class for delegating methods/properties.
    """

    def _delegate_property_get(self, name: str, *args, **kwargs):
        raise TypeError(f"You cannot access the property {name}")

    def _delegate_property_set(self, name: str, value, *args, **kwargs) -> None:
        raise TypeError(f"The property {name} cannot be set")

    def _delegate_method(self, name: str, *args, **kwargs):
        raise TypeError(f"You cannot call method {name}")

    @classmethod
    def _add_delegate_accessors(
        cls,
        delegate,
        accessors: list[str],
        typ: str,
        overwrite: bool = False,
        accessor_mapping: Callable[[str], str] = lambda x: x,
        raise_on_missing: bool = True,
    ) -> None:
        """
        Add delegate accessors to a class.

        Parameters
        ----------
        delegate : object
            The object to which the accessors will be added.
        accessors : list of str
            List of accessor names.
        typ : str
            Type of accessor to add.
        overwrite : bool, optional
            Whether to overwrite existing accessors, by default False.
        accessor_mapping : Callable[[str], str], optional
            Function to map accessor names, by default lambda x: x.
        raise_on_missing : bool, optional
            Whether to raise error on missing accessors, by default True.
        """
        # Implementation intentionally left blank as it's abstract
    def add_delegates(cls, delegate, accessors: list[str], typ: str, overwrite: bool = False,
                      accessor_mapping: Callable[[str], str] = lambda x: x,
                      raise_on_missing: bool = True) -> None:
        """
        Add accessors to cls from the delegate class.
    
        Parameters
        ----------
        cls
            Class to add the methods/properties to.
        delegate
            Class to get methods/properties and doc-strings.
        accessors : list of str
            List of accessors to add.
        typ : {'property', 'method'}
            Type of accessor to create.
        overwrite : bool, default False
            Overwrite the method/property in the target class if it exists.
        accessor_mapping: Callable, default lambda x: x
            Callable to map the delegate's function to the cls' function.
        raise_on_missing: bool, default True
            Raise if an accessor does not exist on delegate.
            False skips the missing accessor.
        """
    
        def _create_delegator_property(name: str):
            """
            Create a property delegator for the given name.
    
            Parameters
            ----------
            name : str
                Name of the property.
    
            Returns
            -------
            property
                Property object delegating to the delegate class.
            """
            def _getter(self):
                """
                Getter function for the delegated property.
    
                Returns
                -------
                Any
                    Value of the delegated property.
                """
                return self._delegate_property_get(name)
    
            def _setter(self, new_value):
                """
                Setter function for the delegated property.
    
                Parameters
                ----------
                new_value : Any
                    New value to set for the delegated property.
                """
                return self._delegate_property_set(name, new_value)
    
            _getter.__name__ = name
            _setter.__name__ = name
    
            return property(
                fget=_getter,
                fset=_setter,
                doc=getattr(delegate, accessor_mapping(name)).__doc__,
            )
    
        def _create_delegator_method(name: str):
            """
            Create a method delegator for the given name.
    
            Parameters
            ----------
            name : str
                Name of the method.
    
            Returns
            -------
            Callable
                Method delegator function.
            """
            def f(self, *args, **kwargs):
                """
                Delegated method function.
    
                Parameters
                ----------
                *args
                    Positional arguments for the delegated method.
                **kwargs
                    Keyword arguments for the delegated method.
    
                Returns
                -------
                Any
                    Return value of the delegated method.
                """
                return self._delegate_method(name, *args, **kwargs)
    
            f.__name__ = name
            f.__doc__ = getattr(delegate, accessor_mapping(name)).__doc__
    
            return f
    
        for name in accessors:
            if (
                not raise_on_missing
                and getattr(delegate, accessor_mapping(name), None) is None
            ):
                continue
    
            if typ == "property":
                f = _create_delegator_property(name)
            else:
                f = _create_delegator_method(name)
    
            # don't overwrite existing methods/properties
            if overwrite or not hasattr(cls, name):
                setattr(cls, name, f)
# 定义一个装饰器函数，用于向类添加委托访问器的功能
def delegate_names(
    delegate,
    accessors: list[str],
    typ: str,
    overwrite: bool = False,
    accessor_mapping: Callable[[str], str] = lambda x: x,
    raise_on_missing: bool = True,
):
    """
    Add delegated names to a class using a class decorator.  This provides
    an alternative usage to directly calling `_add_delegate_accessors`
    below a class definition.

    Parameters
    ----------
    delegate : object
        要获取方法/属性及其文档字符串的类。
    accessors : Sequence[str]
        要添加的访问器列表。
    typ : {'property', 'method'}
        类型，可以是 'property' 或 'method'。
    overwrite : bool, default False
       如果目标类中存在同名方法/属性，是否覆盖。
    accessor_mapping: Callable, default lambda x: x
        将委托对象的函数映射到目标类函数的可调用对象。
    raise_on_missing: bool, default True
        如果委托对象上不存在某个访问器，是否抛出异常。设为 False 则跳过缺失的访问器。

    Returns
    -------
    callable
        一个类装饰器。

    Examples
    --------
    @delegate_names(Categorical, ["categories", "ordered"], "property")
    class CategoricalAccessor(PandasDelegate):
        [...]
    """

    # 内部函数，向类添加委托访问器的具体实现
    def add_delegate_accessors(cls):
        cls._add_delegate_accessors(
            delegate,
            accessors,
            typ,
            overwrite=overwrite,
            accessor_mapping=accessor_mapping,
            raise_on_missing=raise_on_missing,
        )
        return cls

    return add_delegate_accessors


class Accessor:
    """
    自定义的类属性对象。

    用于访问器的描述符。

    Parameters
    ----------
    name : str
        要访问的命名空间，例如 ``df.foo``。
    accessor : cls
        具有扩展方法的类。

    Notes
    -----
    对于访问器，类的 __init__ 方法假设 ``Series``、``DataFrame`` 或 ``Index`` 作为单个参数 ``data``。
    """

    def __init__(self, name: str, accessor) -> None:
        self._name = name
        self._accessor = accessor

    def __get__(self, obj, cls):
        if obj is None:
            # 访问类的属性，例如 Dataset.geo
            return self._accessor
        return self._accessor(obj)


# 在下游库中保留的别名
# TODO: 由于名称现在具有误导性，应当废弃
CachedAccessor = Accessor


@doc(klass="", examples="", others="")
def _register_accessor(
    name: str, cls: type[NDFrame | Index]
) -> Callable[[TypeT], TypeT]:
    """
    在 {klass} 对象上注册自定义访问器。

    Parameters
    ----------
    name : str
        要注册访问器的名称。如果此名称与现有属性冲突，将发出警告。

    Returns
    -------
    callable
        一个类装饰器。

    See Also
    --------
    register_dataframe_accessor : 在 DataFrame 对象上注册自定义访问器。
    """
    """
    register_series_accessor : Register a custom accessor on Series objects.
    register_index_accessor : Register a custom accessor on Index objects.

    Notes
    -----
    This function allows you to register a custom-defined accessor class for {klass}.
    The requirements for the accessor class are as follows:

    * Must contain an init method that:
      * accepts a single {klass} object
      * raises an AttributeError if the {klass} object does not have correctly
        matching inputs for the accessor

    * Must contain a method for each access pattern.
      * The methods should be able to take any argument signature.
      * Accessible using the @property decorator if no additional arguments are
        needed.

    Examples
    --------
    {examples}
    """

    # 定义一个装饰器函数，接受一个类型为 TypeT 的参数 accessor，并返回同样类型的值
    def decorator(accessor: TypeT) -> TypeT:
        # 如果 cls（类对象）已经具有名为 name 的属性
        if hasattr(cls, name):
            # 发出警告，指出注册的 accessor 正在覆盖同名的现有属性
            warnings.warn(
                f"registration of accessor {accessor!r} under name "
                f"{name!r} for type {cls!r} is overriding a preexisting "
                f"attribute with the same name.",
                UserWarning,
                stacklevel=find_stack_level(),  # 获取调用栈级别
            )
        
        # 将 Accessor 类的实例赋值给 cls 的名为 name 的属性
        setattr(cls, name, Accessor(name, accessor))
        # 将 name 添加到 cls 的 _accessors 集合中
        cls._accessors.add(name)
        # 返回 accessor 参数
        return accessor

    # 返回装饰器函数 decorator
    return decorator
# 示例代码展示了如何定义一个 Pandas 扩展，允许注册自定义的 DataFrame 访问器。
# 该函数接受一个名称作为参数，并返回一个装饰器函数 _register_accessor。
# 装饰器函数将传入的 DataFrame 类型注册到指定名称的访问器。

_register_df_examples = """
An accessor that only accepts integers could
have a class defined like this:

>>> @pd.api.extensions.register_dataframe_accessor("int_accessor")
... class IntAccessor:
...     def __init__(self, pandas_obj):
...         if not all(pandas_obj[col].dtype == 'int64' for col in pandas_obj.columns):
...             raise AttributeError("All columns must contain integer values only")
...         self._obj = pandas_obj
...
...     def sum(self):
...         return self._obj.sum()
...
>>> df = pd.DataFrame([[1, 2], ['x', 'y']])
>>> df.int_accessor
Traceback (most recent call last):
...
AttributeError: All columns must contain integer values only.
>>> df = pd.DataFrame([[1, 2], [3, 4]])
>>> df.int_accessor.sum()
0    4
1    6
dtype: int64"""

# 注册 DataFrame 访问器的函数，接受一个名称字符串参数，并返回一个装饰器函数。
@doc(_register_accessor, klass="DataFrame", examples=_register_df_examples)
def register_dataframe_accessor(name: str) -> Callable[[TypeT], TypeT]:
    from pandas import DataFrame

    return _register_accessor(name, DataFrame)


# 示例代码展示了如何定义一个 Pandas 扩展，允许注册自定义的 Series 访问器。
# 该函数接受一个名称作为参数，并返回一个装饰器函数 _register_accessor。
# 装饰器函数将传入的 Series 类型注册到指定名称的访问器。

_register_series_examples = """
An accessor that only accepts integers could
have a class defined like this:

>>> @pd.api.extensions.register_series_accessor("int_accessor")
... class IntAccessor:
...     def __init__(self, pandas_obj):
...         if not pandas_obj.dtype == 'int64':
...             raise AttributeError("The series must contain integer data only")
...         self._obj = pandas_obj
...
...     def sum(self):
...         return self._obj.sum()
...
>>> df = pd.Series([1, 2, 'x'])
>>> df.int_accessor
Traceback (most recent call last):
...
AttributeError: The series must contain integer data only.
>>> df = pd.Series([1, 2, 3])
>>> df.int_accessor.sum()
6"""

# 注册 Series 访问器的函数，接受一个名称字符串参数，并返回一个装饰器函数。
@doc(_register_accessor, klass="Series", examples=_register_series_examples)
def register_series_accessor(name: str) -> Callable[[TypeT], TypeT]:
    from pandas import Series

    return _register_accessor(name, Series)


# 示例代码展示了如何定义一个 Pandas 扩展，允许注册自定义的 Index 访问器。
# 该函数接受一个名称作为参数，并返回一个装饰器函数 _register_accessor。
# 装饰器函数将传入的 Index 类型注册到指定名称的访问器。

_register_index_examples = """
An accessor that only accepts integers could
have a class defined like this:

>>> @pd.api.extensions.register_index_accessor("int_accessor")
... class IntAccessor:
...     def __init__(self, pandas_obj):
...         if not all(isinstance(x, int) for x in pandas_obj):
...             raise AttributeError("The index must only be an integer value")
...         self._obj = pandas_obj
...
...     def even(self):
...         return [x for x in self._obj if x % 2 == 0]
>>> df = pd.DataFrame.from_dict(
...     {"row1": {"1": 1, "2": "a"}, "row2": {"1": 2, "2": "b"}}, orient="index"
... )
>>> df.index.int_accessor
Traceback (most recent call last):
...
AttributeError: The index must only be an integer value.
>>> df = pd.DataFrame(
...     {"col1": [1, 2, 3, 4], "col2": ["a", "b", "c", "d"]}, index=[1, 2, 5, 8]
... )
>>> df.index.int_accessor.even()
[2, 8]"""

# 注册 Index 访问器的函数，接受一个名称字符串参数，并返回一个装饰器函数。
@doc(_register_accessor, klass="Index", examples=_register_index_examples)
def register_index_accessor(name: str) -> Callable[[TypeT], TypeT]:
    from pandas import Index
    # 调用函数 _register_accessor，并返回其结果
    return _register_accessor(name, Index)
```