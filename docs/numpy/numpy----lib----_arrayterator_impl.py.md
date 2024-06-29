# `.\numpy\numpy\lib\_arrayterator_impl.py`

```
"""
A buffered iterator for big arrays.

This module solves the problem of iterating over a big file-based array
without having to read it into memory. The `Arrayterator` class wraps
an array object, and when iterated it will return sub-arrays with at most
a user-specified number of elements.

"""
from operator import mul
from functools import reduce

__all__ = ['Arrayterator']


class Arrayterator:
    """
    Buffered iterator for big arrays.

    `Arrayterator` creates a buffered iterator for reading big arrays in small
    contiguous blocks. The class is useful for objects stored in the
    file system. It allows iteration over the object *without* reading
    everything in memory; instead, small blocks are read and iterated over.

    `Arrayterator` can be used with any object that supports multidimensional
    slices. This includes NumPy arrays, but also variables from
    Scientific.IO.NetCDF or pynetcdf for example.

    Parameters
    ----------
    var : array_like
        The object to iterate over.
    buf_size : int, optional
        The buffer size. If `buf_size` is supplied, the maximum amount of
        data that will be read into memory is `buf_size` elements.
        Default is None, which will read as many element as possible
        into memory.

    Attributes
    ----------
    var : array_like
        The object being iterated over.
    buf_size : int or None
        The buffer size for reading elements into memory.
    start : list
        List of starting indices for iterating over the object.
    stop : list
        List of stopping indices for iterating over the object.
    step : list
        List of step sizes for iterating over the object.
    shape : tuple
        Shape of the object being iterated over.
    flat : iterator
        Flat iterator over the object.

    See Also
    --------
    numpy.ndenumerate : Multidimensional array iterator.
    numpy.flatiter : Flat array iterator.
    numpy.memmap : Create a memory-map to an array stored
                   in a binary file on disk.

    Notes
    -----
    The algorithm works by first finding a "running dimension", along which
    the blocks will be extracted. Given an array of dimensions
    ``(d1, d2, ..., dn)``, e.g. if `buf_size` is smaller than ``d1``, the
    first dimension will be used. If, on the other hand,
    ``d1 < buf_size < d1*d2`` the second dimension will be used, and so on.
    Blocks are extracted along this dimension, and when the last block is
    returned the process continues from the next dimension, until all
    elements have been read.

    Examples
    --------
    >>> a = np.arange(3 * 4 * 5 * 6).reshape(3, 4, 5, 6)
    >>> a_itor = np.lib.Arrayterator(a, 2)
    >>> a_itor.shape
    (3, 4, 5, 6)

    Now we can iterate over ``a_itor``, and it will return arrays of size
    two. Since `buf_size` was smaller than any dimension, the first
    dimension will be iterated over first:

    >>> for subarr in a_itor:
    ...     if not subarr.all():
    ...         print(subarr, subarr.shape) # doctest: +SKIP
    >>> # [[[[0 1]]]] (1, 1, 1, 2)

    """

    def __init__(self, var, buf_size=None):
        # Initialize `Arrayterator` with the object to iterate over `var`
        self.var = var
        # Set the buffer size for reading elements into memory
        self.buf_size = buf_size

        # Initialize start indices for each dimension
        self.start = [0 for dim in var.shape]
        # Initialize stop indices for each dimension
        self.stop = [dim for dim in var.shape]
        # Initialize step sizes for each dimension
        self.step = [1 for dim in var.shape]

    def __getattr__(self, attr):
        # Delegate attribute access to the underlying `var` object
        return getattr(self.var, attr)
    def __getitem__(self, index):
        """
        Return a new array iterator based on the given index.

        """
        # 修正索引，处理省略号和不完整的切片
        if not isinstance(index, tuple):
            index = (index,)
        fixed = []
        length, dims = len(index), self.ndim
        for slice_ in index:
            if slice_ is Ellipsis:
                # 处理省略号，将未指定的维度填充为完整切片
                fixed.extend([slice(None)] * (dims - length + 1))
                length = len(fixed)
            elif isinstance(slice_, int):
                # 将整数索引转换为切片对象
                fixed.append(slice(slice_, slice_ + 1, 1))
            else:
                fixed.append(slice_)
        index = tuple(fixed)
        if len(index) < dims:
            # 补充缺失的维度为完整切片
            index += (slice(None),) * (dims - len(index))

        # 返回一个新的数组迭代器对象
        out = self.__class__(self.var, self.buf_size)
        for i, (start, stop, step, slice_) in enumerate(
                zip(self.start, self.stop, self.step, index)):
            # 计算新迭代器对象的起始、结束和步长
            out.start[i] = start + (slice_.start or 0)
            out.step[i] = step * (slice_.step or 1)
            out.stop[i] = start + (slice_.stop or stop - start)
            out.stop[i] = min(stop, out.stop[i])
        return out

    def __array__(self, dtype=None, copy=None):
        """
        Return the corresponding data as a numpy array.

        """
        # 根据当前的切片信息创建切片元组
        slice_ = tuple(slice(*t) for t in zip(
                self.start, self.stop, self.step))
        # 返回对应的数据数组
        return self.var[slice_]

    @property
    def flat(self):
        """
        A 1-D flat iterator for Arrayterator objects.

        This iterator returns elements of the array to be iterated over in
        `~lib.Arrayterator` one by one. 
        It is similar to `flatiter`.

        See Also
        --------
        lib.Arrayterator
        flatiter

        Examples
        --------
        >>> a = np.arange(3 * 4 * 5 * 6).reshape(3, 4, 5, 6)
        >>> a_itor = np.lib.Arrayterator(a, 2)

        >>> for subarr in a_itor.flat:
        ...     if not subarr:
        ...         print(subarr, type(subarr))
        ...
        0 <class 'numpy.int64'>

        """
        # 返回一个迭代器，用于按顺序访问数组中的每个元素
        for block in self:
            yield from block.flat

    @property
    def shape(self):
        """
        The shape of the array to be iterated over.

        For an example, see `Arrayterator`.

        """
        # 返回被迭代的数组的形状信息
        return tuple(((stop - start - 1) // step + 1) for start, stop, step in
                zip(self.start, self.stop, self.step))
    def __iter__(self):
        # 跳过具有退化维度的数组
        if [dim for dim in self.shape if dim <= 0]:
            return
        
        # 复制起始、停止和步长数组
        start = self.start[:]
        stop = self.stop[:]
        step = self.step[:]
        ndims = self.var.ndim  # 获取变量的维度数

        while True:
            count = self.buf_size or reduce(mul, self.shape)

            # 迭代每个维度，寻找正在运行的维度（即从中构建块的维度）
            rundim = 0
            for i in range(ndims-1, -1, -1):
                # 如果 count 为零，表示沿更高维度已经没有要读取的元素了，因此只读取单个位置
                if count == 0:
                    stop[i] = start[i] + 1
                elif count <= self.shape[i]:
                    # 在这个维度上的限制
                    stop[i] = start[i] + count * step[i]
                    rundim = i
                else:
                    # 沿这个维度读取所有内容
                    stop[i] = self.stop[i]
                stop[i] = min(self.stop[i], stop[i])
                count = count // self.shape[i]

            # 生成一个数据块
            slice_ = tuple(slice(*t) for t in zip(start, stop, step))
            yield self.var[slice_]

            # 更新起始位置，处理到其他维度的溢出
            start[rundim] = stop[rundim]  # 从我们停止的位置开始
            for i in range(ndims-1, 0, -1):
                if start[i] >= self.stop[i]:
                    start[i] = self.start[i]
                    start[i-1] += self.step[i-1]
            if start[0] >= self.stop[0]:
                return
```