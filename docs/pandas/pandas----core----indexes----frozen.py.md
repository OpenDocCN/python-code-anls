# `D:\src\scipysrc\pandas\pandas\core\indexes\frozen.py`

```
"""
frozen (immutable) data structures to support MultiIndexing

These are used for:

- .names (FrozenList)

"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    NoReturn,
)

from pandas.core.base import PandasObject

from pandas.io.formats.printing import pprint_thing

if TYPE_CHECKING:
    from pandas._typing import Self


class FrozenList(PandasObject, list):
    """
    Container that doesn't allow setting item *but*
    because it's technically hashable, will be used
    for lookups, appropriately, etc.
    """

    # Side note: This has to be of type list. Otherwise,
    #            it messes up PyTables type checks.

    def union(self, other) -> FrozenList:
        """
        Returns a FrozenList with other concatenated to the end of self.

        Parameters
        ----------
        other : array-like
            The array-like whose elements we are concatenating.

        Returns
        -------
        FrozenList
            The collection difference between self and other.
        """
        # Convert other to list if it's a tuple
        if isinstance(other, tuple):
            other = list(other)
        # Concatenate self with other and return a new FrozenList instance
        return type(self)(super().__add__(other))

    def difference(self, other) -> FrozenList:
        """
        Returns a FrozenList with elements from other removed from self.

        Parameters
        ----------
        other : array-like
            The array-like whose elements we are removing from self.

        Returns
        -------
        FrozenList
            The collection difference between self and other.
        """
        # Convert other to a set for faster lookup
        other = set(other)
        # Create a new list `temp` excluding elements in `other`
        temp = [x for x in self if x not in other]
        # Return a new FrozenList instance containing `temp`
        return type(self)(temp)

    # TODO: Consider deprecating these in favor of `union` (xref gh-15506)
    # error: Incompatible types in assignment (expression has type
    # "Callable[[FrozenList, Any], FrozenList]", base class "list" defined the
    # type as overloaded function)
    # Override __add__ and __iadd__ to use union method
    __add__ = __iadd__ = union  # type: ignore[assignment]

    def __getitem__(self, n):
        """
        Override __getitem__ to handle slicing and return a new FrozenList instance.

        Parameters
        ----------
        n : int or slice
            Index or slice to retrieve from the list.

        Returns
        -------
        FrozenList
            New FrozenList instance containing the sliced elements.
        """
        if isinstance(n, slice):
            return type(self)(super().__getitem__(n))
        return super().__getitem__(n)

    def __radd__(self, other) -> Self:
        """
        Override __radd__ to handle right-side addition with tuples and return a new FrozenList instance.

        Parameters
        ----------
        other : tuple or list
            Tuple or list to concatenate with self.

        Returns
        -------
        Self
            New instance of the FrozenList with other concatenated.
        """
        if isinstance(other, tuple):
            other = list(other)
        return type(self)(other + list(self))

    def __eq__(self, other: object) -> bool:
        """
        Override __eq__ to compare FrozenList with tuple or another FrozenList.

        Parameters
        ----------
        other : object
            Object to compare with self.

        Returns
        -------
        bool
            True if equal, False otherwise.
        """
        if isinstance(other, (tuple, FrozenList)):
            other = list(other)
        return super().__eq__(other)

    __req__ = __eq__

    def __mul__(self, other) -> Self:
        """
        Override __mul__ to multiply elements of FrozenList and return a new FrozenList instance.

        Parameters
        ----------
        other : int
            Multiplier for the elements.

        Returns
        -------
        Self
            New instance of the FrozenList with elements multiplied.
        """
        return type(self)(super().__mul__(other))

    __imul__ = __mul__

    def __reduce__(self):
        """
        Override __reduce__ to provide serialization support for FrozenList.

        Returns
        -------
        tuple
            Tuple containing type and serialized data (list).
        """
        return type(self), (list(self),)

    # error: Signature of "__hash__" incompatible with supertype "list"
    def __hash__(self) -> int:  # type: ignore[override]
        """
        Override __hash__ to compute hash value for FrozenList.

        Returns
        -------
        int
            Hash value of the FrozenList.
        """
        return hash(tuple(self))
    # 定义一个方法 _disabled，接受任意参数但不返回值，用于在对象不可变时禁用可变操作
    def _disabled(self, *args, **kwargs) -> NoReturn:
        # 抛出类型错误，说明对象不支持可变操作
        raise TypeError(f"'{type(self).__name__}' does not support mutable operations.")

    # 重写 __str__ 方法，返回对象的格式化字符串表示，引用字符串并转义特定字符
    def __str__(self) -> str:
        return pprint_thing(self, quote_strings=True, escape_chars=("\t", "\r", "\n"))

    # 重写 __repr__ 方法，返回对象的规范字符串表示，包括对象类型和其字符串表示
    def __repr__(self) -> str:
        return f"{type(self).__name__}({self!s})"

    # 将 __setitem__ 和 __setslice__ 方法指向 _disabled 方法，忽略类型检查
    __setitem__ = __setslice__ = _disabled  # type: ignore[assignment]
    
    # 将 __delitem__ 和 __delslice__ 方法指向 _disabled 方法，禁用这些操作
    __delitem__ = __delslice__ = _disabled
    
    # 将 pop、append、extend 方法指向 _disabled 方法，禁用列表的修改操作
    pop = append = extend = _disabled
    
    # 将 remove、sort、insert 方法指向 _disabled 方法，禁用列表的修改操作
    remove = sort = insert = _disabled  # type: ignore[assignment]
```