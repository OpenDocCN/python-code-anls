# `D:\src\scipysrc\pandas\pandas\core\ops\common.py`

```
"""
Boilerplate functions used in defining binary operations.
"""

from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING

from pandas._libs.lib import item_from_zerodim
from pandas._libs.missing import is_matching_na

from pandas.core.dtypes.generic import (
    ABCIndex,
    ABCSeries,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from pandas._typing import F


def unpack_zerodim_and_defer(name: str) -> Callable[[F], F]:
    """
    Boilerplate for pandas conventions in arithmetic and comparison methods.

    Parameters
    ----------
    name : str
        The name of the method being decorated.

    Returns
    -------
    decorator
        A decorator function for the method.
    """

    def wrapper(method: F) -> F:
        """
        Inner function to wrap the method.

        Parameters
        ----------
        method : F
            The method to be wrapped.

        Returns
        -------
        F
            The wrapped method.
        """
        return _unpack_zerodim_and_defer(method, name)

    return wrapper


def _unpack_zerodim_and_defer(method: F, name: str) -> F:
    """
    Boilerplate for pandas conventions in arithmetic and comparison methods.

    Ensure method returns NotImplemented when operating against "senior"
    classes.  Ensure zero-dimensional ndarrays are always unpacked.

    Parameters
    ----------
    method : binary method
        The method to be wrapped.
    name : str
        The name of the method.

    Returns
    -------
    method
        The wrapped method.
    """
    stripped_name = name.removeprefix("__").removesuffix("__")
    is_cmp = stripped_name in {"eq", "ne", "lt", "le", "gt", "ge"}

    @wraps(method)
    def new_method(self, other):
        """
        Wrapper function that modifies behavior based on method and operands.

        Parameters
        ----------
        self : object
            The instance of the class invoking the method.
        other : object
            The operand for the method.

        Returns
        -------
        method
            The modified method.
        """
        if is_cmp and isinstance(self, ABCIndex) and isinstance(other, ABCSeries):
            # For comparison ops, Index does *not* defer to Series
            pass
        else:
            prio = getattr(other, "__pandas_priority__", None)
            if prio is not None:
                if prio > self.__pandas_priority__:
                    # e.g. other is DataFrame while self is Index/Series/EA
                    return NotImplemented

        other = item_from_zerodim(other)

        return method(self, other)

    # error: Incompatible return value type (got "Callable[[Any, Any], Any]",
    # expected "F")
    return new_method  # type: ignore[return-value]


def get_op_result_name(left, right):
    """
    Find the appropriate name to pin to an operation result.  This result
    should always be either an Index or a Series.

    Parameters
    ----------
    left : {Series, Index}
        The left operand of the operation.
    right : object
        The right operand of the operation.

    Returns
    -------
    name : object
        Usually a string representing the name of the result.
    """
    if isinstance(right, (ABCSeries, ABCIndex)):
        name = _maybe_match_name(left, right)
    else:
        name = left.name
    return name


def _maybe_match_name(a, b):
    """
    Try to find a name to attach to the result of an operation between
    a and b.  If only one of these has a `name` attribute, return that
    name.  Otherwise return a consensus name if they match or None if
    they have different names.

    Parameters
    ----------
    a : object
    b : object

    Returns
    -------
    name : str or None
        The name derived from the operation result.
    """
    # 检查对象 a 是否有属性 "name"
    a_has = hasattr(a, "name")
    # 检查对象 b 是否有属性 "name"
    b_has = hasattr(b, "name")
    # 如果 a 和 b 都有 "name" 属性
    if a_has and b_has:
        try:
            # 如果 a 和 b 的名称相同，则返回 a 的名称
            if a.name == b.name:
                return a.name
            # 如果名称不同但符合特定的匹配条件（例如都是 np.nan），也返回 a 的名称
            elif is_matching_na(a.name, b.name):
                # 例如都是 np.nan
                return a.name
            else:
                # 其他情况返回 None
                return None
        except TypeError:
            # 如果出现 TypeError，通常是因为 a.name 或 b.name 是 pd.NA
            # 在这种情况下，如果名称符合特定匹配条件，返回 a 的名称
            if is_matching_na(a.name, b.name):
                return a.name
            # 否则返回 None
            return None
        except ValueError:
            # 如果出现 ValueError，通常是因为 a.name 和 b.name 的类型不匹配
            # 例如 np.int64(1) vs (np.int64(1), np.int64(2))
            return None
    # 如果只有 a 有 "name" 属性，则返回 a 的名称
    elif a_has:
        return a.name
    # 如果只有 b 有 "name" 属性，则返回 b 的名称
    elif b_has:
        return b.name
    # 如果 a 和 b 都没有 "name" 属性，则返回 None
    return None
```