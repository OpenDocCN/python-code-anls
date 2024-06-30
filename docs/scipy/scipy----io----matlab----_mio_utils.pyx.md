# `D:\src\scipysrc\scipy\scipy\io\matlab\_mio_utils.pyx`

```
# -*- python -*- like file
''' Utilities for generic processing of return arrays from read
'''

# 导入 numpy 库，以便使用其中的功能
import numpy as np
# 导入 cnp 模块，使得可以使用 C 编写的 numpy 函数
cimport numpy as cnp

# 导入 C 语言中的 numpy 库
cnp.import_array()

# 定义一个 Cython 函数，用于将输入的数组 arr 进行 squeeze 操作，并返回结果
cpdef object squeeze_element(cnp.ndarray arr):
    ''' Return squeezed element

    The returned object may not be an ndarray - for example if we do
    ``arr.item`` to return a ``mat_struct`` object from a struct array '''
    # 如果 arr 的大小为 0，则返回一个空的 numpy 数组，数据类型与 arr 相同
    if not arr.size:
        return np.array([], dtype=arr.dtype)
    # 使用 np.squeeze 函数对 arr 进行 squeeze 操作，并赋值给 arr2
    cdef cnp.ndarray arr2 = np.squeeze(arr)
    # 我们希望对 0 维的数组进行 squeeze，除非它们是记录数组
    if arr2.ndim == 0 and arr2.dtype.kind != 'V':
        # 如果 arr2 的维度为 0，并且数据类型的种类不是 'V'，则返回 arr2 的元素
        return arr2.item()
    # 否则返回 arr2
    return arr2

# 定义一个 Cython 函数，将输入数组 in_arr 的最后一个轴转换为字符串类型
cpdef cnp.ndarray chars_to_strings(in_arr):
    ''' Convert final axis of char array to strings

    Parameters
    ----------
    in_arr : array
       dtype of 'U1'

    Returns
    -------
    str_arr : array
       dtype of 'UN' where N is the length of the last dimension of
       ``arr``
    '''
    # 将输入的 in_arr 赋值给 arr
    cdef cnp.ndarray arr = in_arr
    # 获取 arr 的维度数量，并赋值给 ndim
    cdef int ndim = arr.ndim
    # 获取 arr 的形状，并赋值给 dims
    cdef cnp.npy_intp *dims = arr.shape
    # 获取 arr 的最后一个维度的长度，并赋值给 last_dim
    cdef cnp.npy_intp last_dim = dims[ndim-1]
    # 定义 new_dt_str 和 out_shape 变量
    cdef object new_dt_str, out_shape
    # 如果最后一个维度的长度为 0，则处理空数组的情况
    if last_dim == 0:
        # 获取 arr 的数据类型字符串，并赋值给 new_dt_str
        new_dt_str = arr.dtype.str
        # 如果 ndim 为 2，则 out_shape 为 (0,)
        if ndim == 2:
            out_shape = (0,)
        else:
            # 否则，将 in_arr 的形状除去最后两个维度，并在末尾添加 0
            out_shape = in_arr.shape[:-2] + (0,)
    else:
        # 否则，根据最后一个维度的长度更新数据类型字符串，将 N 追加到字符串末尾
        new_dt_str = arr.dtype.str[:-1] + str(last_dim)
        # 更新 out_shape 为 in_arr 去掉最后一个维度
        out_shape = in_arr.shape[:-1]
    
    # 将 arr 转换为连续存储的数组，以处理 F 序列的数组
    arr = np.ascontiguousarray(arr)
    # 将 arr 视图转换为新的数据类型 new_dt_str
    arr = arr.view(new_dt_str)
    # 将 arr 重新整形为 out_shape 的形状并返回
    return arr.reshape(out_shape)
```