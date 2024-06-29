# `D:\src\scipysrc\pandas\pandas\core\flags.py`

```
# 引入 __future__ 模块中的 annotations 特性，使得在类型提示中可以使用字符串形式的类型声明
from __future__ import annotations

# 引入 TYPE_CHECKING，用于在类型提示中检查类型，避免循环导入问题
from typing import TYPE_CHECKING
import weakref

# 如果 TYPE_CHECKING 为 True，则表示当前处于类型检查状态，执行下面的导入
if TYPE_CHECKING:
    # 从 pandas.core.generic 模块导入 NDFrame 类型
    from pandas.core.generic import NDFrame


class Flags:
    """
    Flags that apply to pandas objects.

    “Flags” differ from “metadata”. Flags reflect properties of the pandas
    object (the Series or DataFrame). Metadata refer to properties of the
    dataset, and should be stored in DataFrame.attrs.

    Parameters
    ----------
    obj : Series or DataFrame
        The object these flags are associated with.
    allows_duplicate_labels : bool, default True
        Whether to allow duplicate labels in this object. By default,
        duplicate labels are permitted. Setting this to ``False`` will
        cause an :class:`errors.DuplicateLabelError` to be raised when
        `index` (or columns for DataFrame) is not unique, or any
        subsequent operation on introduces duplicates.
        See :ref:`duplicates.disallow` for more.

        .. warning::

           This is an experimental feature. Currently, many methods fail to
           propagate the ``allows_duplicate_labels`` value. In future versions
           it is expected that every method taking or returning one or more
           DataFrame or Series objects will propagate ``allows_duplicate_labels``.

    See Also
    --------
    DataFrame.attrs : Dictionary of global attributes of this dataset.
    Series.attrs : Dictionary of global attributes of this dataset.

    Examples
    --------
    Attributes can be set in two ways:

    >>> df = pd.DataFrame()
    >>> df.flags
    <Flags(allows_duplicate_labels=True)>
    >>> df.flags.allows_duplicate_labels = False
    >>> df.flags
    <Flags(allows_duplicate_labels=False)>

    >>> df.flags["allows_duplicate_labels"] = True
    >>> df.flags
    <Flags(allows_duplicate_labels=True)>
    """

    # _keys 属性，定义了 Flags 类中有效的属性键集合，目前只有 "allows_duplicate_labels"
    _keys: set[str] = {"allows_duplicate_labels"}

    def __init__(self, obj: NDFrame, *, allows_duplicate_labels: bool) -> None:
        # 初始化 Flags 对象的实例
        self._allows_duplicate_labels = allows_duplicate_labels  # 设置是否允许重复标签的属性值
        self._obj = weakref.ref(obj)  # 使用弱引用保存传入的 pandas 对象的引用

    @property
    def allows_duplicate_labels(self) -> bool:
        """
        Whether this object allows duplicate labels.

        Setting ``allows_duplicate_labels=False`` ensures that the
        index (and columns of a DataFrame) are unique. Most methods
        that accept and return a Series or DataFrame will propagate
        the value of ``allows_duplicate_labels``.

        See :ref:`duplicates` for more.

        See Also
        --------
        DataFrame.attrs : Set global metadata on this object.
        DataFrame.set_flags : Set global flags on this object.

        Examples
        --------
        >>> df = pd.DataFrame({"A": [1, 2]}, index=["a", "a"])
        >>> df.flags.allows_duplicate_labels
        True
        >>> df.flags.allows_duplicate_labels = False
        Traceback (most recent call last):
            ...
        pandas.errors.DuplicateLabelError: Index has duplicates.
              positions
        label
        a        [0, 1]
        """
        return self._allows_duplicate_labels

    @allows_duplicate_labels.setter
    def allows_duplicate_labels(self, value: bool) -> None:
        """
        Setter method for allows_duplicate_labels attribute.

        Parameters
        ----------
        value : bool
            The value to set for allows_duplicate_labels.

        Raises
        ------
        ValueError
            If the object associated with this flag has been deleted.

        Notes
        -----
        If value is False, it checks uniqueness for each axis in the object.
        """
        value = bool(value)
        obj = self._obj()
        if obj is None:
            raise ValueError("This flag's object has been deleted.")

        if not value:
            for ax in obj.axes:
                ax._maybe_check_unique()

        self._allows_duplicate_labels = value

    def __getitem__(self, key: str):
        """
        Retrieve the value associated with the given key.

        Parameters
        ----------
        key : str
            The key to retrieve the value for.

        Returns
        -------
        object
            The value associated with the key.

        Raises
        ------
        KeyError
            If the key is not found in _keys.
        """
        if key not in self._keys:
            raise KeyError(key)

        return getattr(self, key)

    def __setitem__(self, key: str, value) -> None:
        """
        Set the value associated with the given key.

        Parameters
        ----------
        key : str
            The key to set the value for.
        value : object
            The value to associate with the key.

        Raises
        ------
        ValueError
            If the key is not recognized (not in _keys).
        """
        if key not in self._keys:
            raise ValueError(f"Unknown flag {key}. Must be one of {self._keys}")
        setattr(self, key, value)

    def __repr__(self) -> str:
        """
        Return a string representation of the Flags object.

        Returns
        -------
        str
            A string representation of the Flags object.
        """
        return f"<Flags(allows_duplicate_labels={self.allows_duplicate_labels})>"

    def __eq__(self, other: object) -> bool:
        """
        Compare this Flags object with another for equality.

        Parameters
        ----------
        other : object
            The object to compare with.

        Returns
        -------
        bool
            True if both objects are of the same type and have the same allows_duplicate_labels value, False otherwise.
        """
        if isinstance(other, type(self)):
            return self.allows_duplicate_labels == other.allows_duplicate_labels
        return False
```