# `D:\src\scipysrc\pandas\pandas\_libs\arrays.pyx`

```
"""
Cython implementations for internal ExtensionArrays.
"""
# 导入Cython模块
cimport cython

# 导入NumPy模块
import numpy as np

# 从Cython中导入NumPy相关的类型和函数
cimport numpy as cnp
from cpython cimport PyErr_Clear
from numpy cimport ndarray

# 调用cnp.import_array()以确保正确导入NumPy C API
cnp.import_array()

# 使用cython.freelist优化的Cython类NDArrayBacked
@cython.freelist(16)
cdef class NDArrayBacked:
    """
    Implementing these methods in cython improves performance quite a bit.

    import pandas as pd

    from pandas._libs.arrays import NDArrayBacked as cls

    dti = pd.date_range("2016-01-01", periods=3)
    dta = dti._data
    arr = dta._ndarray

    obj = cls._simple_new(arr, arr.dtype)

    # for foo in [arr, dta, obj]: ...

    %timeit foo.copy()
    299 ns ± 30 ns per loop     # <-- arr underlying ndarray (for reference)
    530 ns ± 9.24 ns per loop   # <-- dta with cython NDArrayBacked
    1.66 µs ± 46.3 ns per loop  # <-- dta without cython NDArrayBacked
    328 ns ± 5.29 ns per loop   # <-- obj with NDArrayBacked.__cinit__
    371 ns ± 6.97 ns per loop   # <-- obj with NDArrayBacked._simple_new

    %timeit foo.T
    125 ns ± 6.27 ns per loop   # <-- arr underlying ndarray (for reference)
    226 ns ± 7.66 ns per loop   # <-- dta with cython NDArrayBacked
    911 ns ± 16.6 ns per loop   # <-- dta without cython NDArrayBacked
    215 ns ± 4.54 ns per loop   # <-- obj with NDArrayBacked._simple_new

    """
    # TODO: implement take in terms of cnp.PyArray_TakeFrom
    # TODO: implement concat_same_type in terms of cnp.PyArray_Concatenate

    # 定义两个只读属性
    cdef:
        readonly ndarray _ndarray
        readonly object _dtype

    # 类的初始化方法，接受一个ndarray和一个dtype对象作为参数
    def __init__(self, ndarray values, object dtype):
        self._ndarray = values
        self._dtype = dtype

    # 类方法，用于创建一个新的NDArrayBacked实例
    @classmethod
    def _simple_new(cls, ndarray values, object dtype):
        cdef:
            NDArrayBacked obj
        obj = NDArrayBacked.__new__(cls)
        obj._ndarray = values
        obj._dtype = dtype
        return obj

    # 公共CPython方法，用于根据给定的values创建一个新的NDArrayBacked实例
    cpdef NDArrayBacked _from_backing_data(self, ndarray values):
        """
        Construct a new ExtensionArray `new_array` with `arr` as its _ndarray.

        This should round-trip:
            self == self._from_backing_data(self._ndarray)
        """
        # TODO: re-reuse simple_new if/when it can be cpdef
        cdef:
            NDArrayBacked obj
        obj = NDArrayBacked.__new__(type(self))
        obj._ndarray = values
        obj._dtype = self._dtype
        return obj
    # 定义一个 Cython 的 cdef 函数，用于反序列化对象状态
    cpdef __setstate__(self, state):
        # 如果状态是字典类型，则处理
        if isinstance(state, dict):
            # 如果状态中包含 "_data" 键，则取出其值作为数据
            if "_data" in state:
                data = state.pop("_data")
            # 否则如果包含 "_ndarray" 键，则取出其值作为数据
            elif "_ndarray" in state:
                data = state.pop("_ndarray")
            else:
                # 如果状态中既没有 "_data" 也没有 "_ndarray"，则引发值错误异常
                raise ValueError  # pragma: no cover
            # 将取出的数据赋值给对象的 _ndarray 属性
            self._ndarray = data
            # 弹出状态中的 "_dtype" 键，并将其值赋给对象的 _dtype 属性
            self._dtype = state.pop("_dtype")

            # 遍历状态中的其余键值对，使用 setattr 方法设置对象的属性
            for key, val in state.items():
                setattr(self, key, val)
        # 如果状态是元组类型，则处理
        elif isinstance(state, tuple):
            # 如果元组长度不为 3，则根据条件处理
            if len(state) != 3:
                # 如果长度为 1，且第一个元素是字典，则递归调用 __setstate__ 方法处理
                if len(state) == 1 and isinstance(state[0], dict):
                    self.__setstate__(state[0])
                    return
                # 否则引发未实现错误，并传入当前状态对象
                raise NotImplementedError(state)  # pragma: no cover

            # 提取元组的前两个元素作为数据和数据类型
            data, dtype = state[:2]
            # 如果数据类型是 np.ndarray，则交换数据和数据类型的位置
            if isinstance(dtype, np.ndarray):
                dtype, data = data, dtype
            # 将提取的数据赋值给对象的 _ndarray 属性
            self._ndarray = data
            # 将提取的数据类型赋值给对象的 _dtype 属性
            self._dtype = dtype

            # 如果状态的第三个元素是字典，则遍历其键值对并设置为对象的属性
            if isinstance(state[2], dict):
                for key, val in state[2].items():
                    setattr(self, key, val)
            else:
                # 否则引发未实现错误，并传入当前状态对象
                raise NotImplementedError(state)  # pragma: no cover
        else:
            # 如果状态既不是字典也不是元组类型，则引发未实现错误，并传入当前状态对象
            raise NotImplementedError(state)  # pragma: no cover

    # 返回对象中 _ndarray 的长度作为整数
    def __len__(self) -> int:
        return len(self._ndarray)

    # 返回对象中 _ndarray 的形状
    @property
    def shape(self):
        # object cast bc _ndarray.shape is npy_intp*
        return (<object>(self._ndarray)).shape

    # 返回对象中 _ndarray 的维度作为整数
    @property
    def ndim(self) -> int:
        return self._ndarray.ndim

    # 返回对象中 _ndarray 的元素数目作为整数
    @property
    def size(self) -> int:
        return self._ndarray.size

    # 返回对象中 _ndarray 的字节大小
    @property
    def nbytes(self) -> int:
        return cnp.PyArray_NBYTES(self._ndarray)

    # 复制对象中 _ndarray 的数据，指定顺序
    def copy(self, order="C"):
        # 定义 Cython 的 cdef 变量和一个整数变量 success
        cdef:
            cnp.NPY_ORDER order_code
            int success

        # 调用 cnp.PyArray_OrderConverter 函数，将 order 转换为 order_code
        success = cnp.PyArray_OrderConverter(order, &order_code)
        # 如果转换失败，则清除异常并抛出值错误异常，显示错误消息
        if not success:
            PyErr_Clear()
            msg = f"order must be one of 'C', 'F', 'A', or 'K' (got '{order}')"
            raise ValueError(msg)

        # 调用 cnp.PyArray_NewCopy 函数，以指定的顺序复制 _ndarray 的数据
        res_values = cnp.PyArray_NewCopy(self._ndarray, order_code)
        # 返回新创建的数据对象，调用 _from_backing_data 方法
        return self._from_backing_data(res_values)

    # 删除对象中 _ndarray 沿指定轴的元素
    def delete(self, loc, axis=0):
        # 调用 numpy 的 np.delete 函数，删除 _ndarray 指定位置的元素
        res_values = np.delete(self._ndarray, loc, axis=axis)
        # 返回删除元素后的数据对象，调用 _from_backing_data 方法
        return self._from_backing_data(res_values)

    # 交换对象中 _ndarray 的两个轴
    def swapaxes(self, axis1, axis2):
        # 调用 cnp.PyArray_SwapAxes 函数，交换 _ndarray 的指定两个轴
        res_values = cnp.PyArray_SwapAxes(self._ndarray, axis1, axis2)
        # 返回交换轴后的数据对象，调用 _from_backing_data 方法
        return self._from_backing_data(res_values)

    # 使用 cnp.PyArray_Repeat 函数，沿指定轴重复对象中 _ndarray 的元素
    # TODO: pass NPY_MAXDIMS equiv to axis=None?
    def repeat(self, repeats, axis: int | np.integer = 0):
        # 如果 axis 为 None，则将其设为 0
        if axis is None:
            axis = 0
        # 调用 cnp.PyArray_Repeat 函数，重复 _ndarray 的元素
        res_values = cnp.PyArray_Repeat(self._ndarray, repeats, <int>axis)
        # 返回重复元素后的数据对象，调用 _from_backing_data 方法
        return self._from_backing_data(res_values)

    # 重新定义对象中 _ndarray 的形状
    def reshape(self, *args, **kwargs):
        # 调用 _ndarray 的 reshape 方法，重新定义其形状
        res_values = self._ndarray.reshape(*args, **kwargs)
        # 返回重新定义形状后的数据对象，调用 _from_backing_data 方法
        return self._from_backing_data(res_values)
    def ravel(self, order="C"):
        """
        将数组展平为一维数组，并按指定顺序重新排列。

        Args:
            order (str): 排列顺序，默认为'C'（按行优先）。

        Returns:
            self._from_backing_data(res_values): 返回重新排列后的一维数组对象。
        """
        # cnp.PyArray_OrderConverter(PyObject* obj, NPY_ORDER* order)
        # 调用C函数将Python对象转换为NumPy数组的排列顺序
        # res_values = cnp.PyArray_Ravel(self._ndarray, order)
        res_values = self._ndarray.ravel(order)
        return self._from_backing_data(res_values)

    @property
    def T(self):
        """
        返回数组的转置视图。

        Returns:
            self._from_backing_data(res_values): 返回数组的转置视图。
        """
        res_values = self._ndarray.T
        return self._from_backing_data(res_values)

    def transpose(self, *axes):
        """
        返回根据指定轴重新排列后的数组。

        Args:
            *axes: 要传递给transpose方法的轴参数。

        Returns:
            self._from_backing_data(res_values): 返回重新排列后的数组对象。
        """
        res_values = self._ndarray.transpose(*axes)
        return self._from_backing_data(res_values)

    @classmethod
    def _concat_same_type(cls, to_concat, axis=0):
        """
        合并同类型的对象数组为一个新的数组对象。

        Args:
            to_concat (list): 要合并的对象数组列表。
            axis (int): 沿着哪个轴进行合并，默认为0。

        Returns:
            to_concat[0]._from_backing_data(new_arr): 返回合并后的新数组对象。
        """
        # NB: We are assuming at this point that dtypes all match
        # 提示：我们在此假设所有对象的数据类型都匹配
        new_values = [obj._ndarray for obj in to_concat]
        # 调用C函数将多个NumPy数组对象在指定轴上合并为一个新的NumPy数组对象
        new_arr = cnp.PyArray_Concatenate(new_values, axis)
        return to_concat[0]._from_backing_data(new_arr)
```