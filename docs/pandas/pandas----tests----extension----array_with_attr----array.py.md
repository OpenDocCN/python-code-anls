# `D:\src\scipysrc\pandas\pandas\tests\extension\array_with_attr\array.py`

```
"""
Test extension array that has custom attribute information (not stored on the dtype).

"""

from __future__ import annotations

import numbers
from typing import TYPE_CHECKING

import numpy as np

from pandas.core.dtypes.base import ExtensionDtype

import pandas as pd
from pandas.core.arrays import ExtensionArray

if TYPE_CHECKING:
    from pandas._typing import type_t


class FloatAttrDtype(ExtensionDtype):
    type = float
    name = "float_attr"
    na_value = np.nan

    @classmethod
    def construct_array_type(cls) -> type_t[FloatAttrArray]:
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
            The type of the associated array.
        """
        return FloatAttrArray


class FloatAttrArray(ExtensionArray):
    dtype = FloatAttrDtype()
    __array_priority__ = 1000

    def __init__(self, values, attr=None) -> None:
        """
        Initialize a FloatAttrArray.

        Parameters
        ----------
        values : np.ndarray
            An array of float64 dtype.
        attr : any, optional
            Custom attribute associated with the array, by default None.

        Raises
        ------
        TypeError
            If `values` is not a numpy array of float64 dtype.
        """
        if not isinstance(values, np.ndarray):
            raise TypeError("Need to pass a numpy array of float64 dtype as values")
        if not values.dtype == "float64":
            raise TypeError("Need to pass a numpy array of float64 dtype as values")
        self.data = values
        self.attr = attr

    @classmethod
    def _from_sequence(cls, scalars, *, dtype=None, copy=False):
        """
        Construct a FloatAttrArray from a sequence of scalars.

        Parameters
        ----------
        scalars : sequence
            Sequence of scalars to be converted into the array.
        dtype : str or dtype, optional
            Desired dtype of the array, by default None.
        copy : bool, optional
            Whether to copy the data, by default False.

        Returns
        -------
        FloatAttrArray
            A new instance of FloatAttrArray.
        """
        if not copy:
            data = np.asarray(scalars, dtype="float64")
        else:
            data = np.array(scalars, dtype="float64", copy=copy)
        return cls(data)

    def __getitem__(self, item):
        """
        Retrieve items from the array.

        Parameters
        ----------
        item : int, slice, or array-like
            Index or slice to retrieve.

        Returns
        -------
        FloatAttrArray
            A new instance of FloatAttrArray containing selected items.
        """
        if isinstance(item, numbers.Integral):
            return self.data[item]
        else:
            item = pd.api.indexers.check_array_indexer(self, item)
            return type(self)(self.data[item], self.attr)

    def __len__(self) -> int:
        """
        Return the length of the array.

        Returns
        -------
        int
            Length of the array.
        """
        return len(self.data)

    def isna(self):
        """
        Check for NaN values in the array.

        Returns
        -------
        np.ndarray
            Boolean array indicating NaN values.
        """
        return np.isnan(self.data)

    def take(self, indexer, allow_fill=False, fill_value=None):
        """
        Take elements from the array using an indexer.

        Parameters
        ----------
        indexer : array-like
            Indices to take from the array.
        allow_fill : bool, optional
            Whether to allow filling missing values, by default False.
        fill_value : any, optional
            Value to fill missing values with, by default None.

        Returns
        -------
        FloatAttrArray
            A new instance of FloatAttrArray containing selected items.
        """
        from pandas.api.extensions import take

        data = self.data
        if allow_fill and fill_value is None:
            fill_value = self.dtype.na_value

        result = take(data, indexer, fill_value=fill_value, allow_fill=allow_fill)
        return type(self)(result, self.attr)

    def copy(self):
        """
        Create a copy of the array.

        Returns
        -------
        FloatAttrArray
            A new instance of FloatAttrArray that is a copy of the original.
        """
        return type(self)(self.data.copy(), self.attr)

    @classmethod
    def _concat_same_type(cls, to_concat):
        """
        Concatenate multiple instances of FloatAttrArray.

        Parameters
        ----------
        to_concat : list of FloatAttrArray
            Arrays to concatenate.

        Returns
        -------
        FloatAttrArray
            A new instance of FloatAttrArray containing concatenated data.
        """
        data = np.concatenate([x.data for x in to_concat])
        attr = to_concat[0].attr if len(to_concat) else None
        return cls(data, attr)
```