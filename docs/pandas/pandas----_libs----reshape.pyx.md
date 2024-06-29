# `D:\src\scipysrc\pandas\pandas\_libs\reshape.pyx`

```
# 导入 Cython 模块
cimport cython
# 从 Cython 中导入 Py_ssize_t 类型
from cython cimport Py_ssize_t
# 从 numpy 中导入特定类型
from numpy cimport (
    int64_t,
    ndarray,
    uint8_t,
)

# 导入 numpy 库并重命名为 np
import numpy as np

# 从 Cython 的 numpy 模块中导入 cnp
cimport numpy as cnp
# 从 C 标准库中导入 NAN 常量
from libc.math cimport NAN

# 调用 cnp 模块的 import_array 函数
cnp.import_array()

# 从 pandas 库的内部 _libs.dtypes 中导入 numeric_object_t 类型
from pandas._libs.dtypes cimport numeric_object_t
# 从 pandas 库的内部 _libs.lib 中导入 c_is_list_like 函数
from pandas._libs.lib cimport c_is_list_like

# 使用 Cython 的装饰器设置 wraparound 和 boundscheck 为 False
@cython.wraparound(False)
@cython.boundscheck(False)
def unstack(const numeric_object_t[:, :] values, const uint8_t[:] mask,
            Py_ssize_t stride, Py_ssize_t length, Py_ssize_t width,
            numeric_object_t[:, :] new_values, uint8_t[:, :] new_mask) -> None:
    """
    Transform long values to wide new_values.

    Parameters
    ----------
    values : typed ndarray
        输入的二维数组，类型为 numeric_object_t
    mask : np.ndarray[bool]
        布尔类型的掩码数组
    stride : int
        步长
    length : int
        长度
    width : int
        宽度
    new_values : np.ndarray[bool]
        结果数组，类型为 numeric_object_t
    new_mask : np.ndarray[bool]
        结果掩码数组，类型为 uint8_t
    """
    cdef:
        Py_ssize_t i, j, w, nulls, s, offset

    if numeric_object_t is not object:
        # 在编译时评估
        with nogil:
            for i in range(stride):
                nulls = 0
                for j in range(length):
                    for w in range(width):
                        offset = j * width + w
                        if mask[offset]:
                            s = i * width + w
                            new_values[j, s] = values[offset - nulls, i]
                            new_mask[j, s] = 1
                        else:
                            nulls += 1
    else:
        # 对象类型为 object，与上述相同但不能使用 nogil
        for i in range(stride):
            nulls = 0
            for j in range(length):
                for w in range(width):
                    offset = j * width + w
                    if mask[offset]:
                        s = i * width + w
                        new_values[j, s] = values[offset - nulls, i]
                        new_mask[j, s] = 1
                    else:
                        nulls += 1

@cython.wraparound(False)
@cython.boundscheck(False)
def explode(object[:] values):
    """
    Transform array list-likes to long form.
    Preserve non-list entries.

    Parameters
    ----------
    values : ndarray[object]
        输入的对象数组

    Returns
    -------
    ndarray[object]
        结果数组
    ndarray[int64_t]
        计数数组
    """
    cdef:
        Py_ssize_t i, j, count, n
        object v
        ndarray[object] result
        ndarray[int64_t] counts

    # 计算输入数组的长度
    n = len(values)
    # 初始化计数数组为零
    counts = np.zeros(n, dtype="int64")
    for i in range(n):
        v = values[i]
        # 判断对象是否类似列表
        if c_is_list_like(v, True):
            if len(v):
                counts[i] += len(v)
            else:
                # 空列表，使用 NaN 标记
                counts[i] += 1
        else:
            counts[i] += 1

    # 根据计数数组的总和创建结果数组
    result = np.empty(counts.sum(), dtype="object")
    count = 0
    # 遍历范围为 n 的索引，处理每个值
    for i in range(n):
        # 获取当前索引处的值
        v = values[i]

        # 检查当前值是否类似于列表（list-like），允许为空列表
        if c_is_list_like(v, True):
            # 如果 v 是非空列表
            if len(v):
                # 将 v 转换为列表形式
                v = list(v)
                # 遍历列表 v 的每个元素
                for j in range(len(v)):
                    # 将列表 v 的元素放入结果数组中，并增加计数
                    result[count] = v[j]
                    count += 1
            else:
                # 如果 v 是空的列表类似结构，使用 NaN（非数字）作为标记放入结果数组，并增加计数
                result[count] = NAN
                count += 1
        else:
            # 如果 v 不是列表类似结构，直接将其作为标量放入结果数组中，并增加计数
            result[count] = v
            count += 1

    # 返回处理后的结果数组和计数器
    return result, counts
```