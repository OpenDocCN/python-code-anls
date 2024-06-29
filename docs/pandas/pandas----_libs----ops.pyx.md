# `D:\src\scipysrc\pandas\pandas\_libs\ops.pyx`

```
import operator  # 导入标准库中的 operator 模块，用于获取比较操作符

cimport cython  # 导入 Cython 中的 cimport 模块
from cpython.object cimport (  # 从 CPython 中的 object 模块导入特定的对象
    Py_EQ,  # 等于操作符的标志
    Py_GE,  # 大于等于操作符的标志
    Py_GT,  # 大于操作符的标志
    Py_LE,  # 小于等于操作符的标志
    Py_LT,  # 小于操作符的标志
    Py_NE,  # 不等于操作符的标志
    PyObject_RichCompareBool,  # 用于对象比较的函数
)
from cython cimport Py_ssize_t  # 从 Cython 中导入 Py_ssize_t 类型

import numpy as np  # 导入 NumPy 库，并用 np 作为别名

from numpy cimport (  # 从 NumPy 中导入特定对象
    import_array,  # 导入数组对象的函数
    ndarray,  # 导入多维数组对象
    uint8_t,  # 无符号 8 位整数类型
)

import_array()  # 初始化 NumPy 数组

from pandas._libs.missing cimport checknull  # 从 Pandas 库中导入 checknull 函数
from pandas._libs.util cimport is_nan  # 从 Pandas 库中导入 is_nan 函数


@cython.wraparound(False)  # 禁用 Cython 中数组的负索引
@cython.boundscheck(False)  # 禁用 Cython 中数组的边界检查
def scalar_compare(ndarray[object] values, object val, object op) -> ndarray:
    """
    Compare each element of `values` array with the scalar `val`, with
    the comparison operation described by `op`.

    Parameters
    ----------
    values : ndarray[object]
        包含待比较元素的数组
    val : object
        用于比较的标量值
    op : {operator.eq, operator.ne,
          operator.le, operator.lt,
          operator.ge, operator.gt}
        比较操作符，可选值为标准库 operator 模块中的比较操作符函数

    Returns
    -------
    result : ndarray[bool]
        包含比较结果的布尔类型数组
    """
    cdef:
        Py_ssize_t i, n = len(values)  # 定义 Cython 变量 i 和 n，分别为索引和数组长度
        ndarray[uint8_t, cast=True] result  # 定义结果数组，类型为 uint8_t 的数组
        bint isnull_val  # 定义用于检查 val 是否为 null 的布尔变量
        int flag  # 定义整型变量 flag，用于存储比较操作的标志
        object x  # 定义对象 x，用于临时存储数组元素

    if op is operator.lt:  # 如果比较操作是小于
        flag = Py_LT  # 设置 flag 为 Py_LT，表示小于的比较操作标志
    elif op is operator.le:  # 如果比较操作是小于等于
        flag = Py_LE  # 设置 flag 为 Py_LE，表示小于等于的比较操作标志
    elif op is operator.gt:  # 如果比较操作是大于
        flag = Py_GT  # 设置 flag 为 Py_GT，表示大于的比较操作标志
    elif op is operator.ge:  # 如果比较操作是大于等于
        flag = Py_GE  # 设置 flag 为 Py_GE，表示大于等于的比较操作标志
    elif op is operator.eq:  # 如果比较操作是等于
        flag = Py_EQ  # 设置 flag 为 Py_EQ，表示等于的比较操作标志
    elif op is operator.ne:  # 如果比较操作是不等于
        flag = Py_NE  # 设置 flag 为 Py_NE，表示不等于的比较操作标志
    else:
        raise ValueError("Unrecognized operator")  # 如果操作符无法识别，抛出 ValueError 异常

    result = np.empty(n, dtype=bool).view(np.uint8)  # 创建一个空的布尔类型数组，并将其视图转换为 uint8 类型
    isnull_val = checknull(val)  # 检查 val 是否为 null，并将结果存储在 isnull_val 中

    if flag == Py_NE:  # 如果比较操作是不等于
        for i in range(n):  # 遍历数组中的每一个元素
            x = values[i]  # 获取数组中的当前元素
            if checknull(x):  # 如果当前元素为 null
                result[i] = True  # 将结果数组中对应位置设置为 True
            elif isnull_val:  # 如果 val 也为 null
                result[i] = True  # 将结果数组中对应位置设置为 True
            else:
                try:
                    result[i] = PyObject_RichCompareBool(x, val, flag)  # 使用 PyObject_RichCompareBool 比较当前元素和 val
                except TypeError:
                    result[i] = True  # 捕获到 TypeError 异常时，将结果数组中对应位置设置为 True
    elif flag == Py_EQ:  # 如果比较操作是等于
        for i in range(n):  # 遍历数组中的每一个元素
            x = values[i]  # 获取数组中的当前元素
            if checknull(x):  # 如果当前元素为 null
                result[i] = False  # 将结果数组中对应位置设置为 False
            elif isnull_val:  # 如果 val 也为 null
                result[i] = False  # 将结果数组中对应位置设置为 False
            else:
                try:
                    result[i] = PyObject_RichCompareBool(x, val, flag)  # 使用 PyObject_RichCompareBool 比较当前元素和 val
                except TypeError:
                    result[i] = False  # 捕获到 TypeError 异常时，将结果数组中对应位置设置为 False

    else:  # 对于其他的比较操作
        for i in range(n):  # 遍历数组中的每一个元素
            x = values[i]  # 获取数组中的当前元素
            if checknull(x):  # 如果当前元素为 null
                result[i] = False  # 将结果数组中对应位置设置为 False
            elif isnull_val:  # 如果 val 也为 null
                result[i] = False  # 将结果数组中对应位置设置为 False
            else:
                result[i] = PyObject_RichCompareBool(x, val, flag)  # 使用 PyObject_RichCompareBool 比较当前元素和 val

    return result.view(bool)  # 将结果数组的视图转换为布尔类型数组并返回


@cython.wraparound(False)  # 禁用 Cython 中数组的负索引
@cython.boundscheck(False)  # 禁用 Cython 中数组的边界检查
def vec_compare(ndarray[object] left, ndarray[object] right, object op) -> ndarray:
    """
    Compare the elements of `left` with the elements of `right` pointwise,
    with the comparison operation described by `op`.

    Parameters
    ----------
    left : ndarray[object]
        包含左侧数组中的元素
    right : ndarray[object]
        包含右侧数组中的元素
    op : {operator.eq, operator.ne,
          operator.le, operator.lt,
          operator.ge, operator.gt}
        比较操作符，可选值为标准库 operator 模块中的比较操作符函数

    Returns
    -------
    result : ndarray[bool]
        包含比较结果的布尔类型数组
    """
    op : {operator.eq, operator.ne,
          operator.le, operator.lt,
          operator.ge, operator.gt}
    """
    定义一个包含所有比较操作符的集合 op，用于选择比较操作的标志

    Returns
    -------
    result : ndarray[bool]
    """
    声明函数返回一个布尔类型的 NumPy 数组 result

    cdef:
        Py_ssize_t i, n = len(left)
        声明两个 C 语言风格的变量 i 和 n，其中 n 为 left 数组的长度
        ndarray[uint8_t, cast=True] result
        声明一个 NumPy 数组 result，数据类型为 uint8_t（无符号整数型），通过 cast=True 明确类型转换
        int flag
        声明一个整型变量 flag

    if n != <Py_ssize_t>len(right):
        如果 left 和 right 数组长度不同，抛出 ValueError 异常
        raise ValueError(f"Arrays were different lengths: {n} vs {len(right)}")

    if op is operator.lt:
        如果操作符为 operator.lt（小于），设置 flag 为 Py_LT
        flag = Py_LT
    elif op is operator.le:
        如果操作符为 operator.le（小于等于），设置 flag 为 Py_LE
        flag = Py_LE
    elif op is operator.gt:
        如果操作符为 operator.gt（大于），设置 flag 为 Py_GT
        flag = Py_GT
    elif op is operator.ge:
        如果操作符为 operator.ge（大于等于），设置 flag 为 Py_GE
        flag = Py_GE
    elif op is operator.eq:
        如果操作符为 operator.eq（等于），设置 flag 为 Py_EQ
        flag = Py_EQ
    elif op is operator.ne:
        如果操作符为 operator.ne（不等于），设置 flag 为 Py_NE
        flag = Py_NE
    else:
        如果操作符未被识别，抛出 ValueError 异常
        raise ValueError("Unrecognized operator")

    result = np.empty(n, dtype=bool).view(np.uint8)
    创建一个长度为 n 的空 NumPy 数组 result，数据类型为 bool，并将其视图转换为 uint8 类型

    if flag == Py_NE:
        如果操作标志为 Py_NE（不等于）
        for i in range(n):
            遍历范围为 n 的索引 i
            x = left[i]
            获取 left 数组的第 i 个元素赋值给 x
            y = right[i]
            获取 right 数组的第 i 个元素赋值给 y

            if checknull(x) or checknull(y):
                如果 x 或 y 为 null，将 result[i] 设为 True
                result[i] = True
            else:
                否则使用 PyObject_RichCompareBool 函数比较 x 和 y 的值，根据 flag 设定 result[i]
                result[i] = PyObject_RichCompareBool(x, y, flag)
    else:
        否则，即 flag 不为 Py_NE（其他比较操作）
        for i in range(n):
            遍历范围为 n 的索引 i
            x = left[i]
            获取 left 数组的第 i 个元素赋值给 x
            y = right[i]
            获取 right 数组的第 i 个元素赋值给 y

            if checknull(x) or checknull(y):
                如果 x 或 y 为 null，将 result[i] 设为 False
                result[i] = False
            else:
                否则使用 PyObject_RichCompareBool 函数比较 x 和 y 的值，根据 flag 设定 result[i]
                result[i] = PyObject_RichCompareBool(x, y, flag)

    return result.view(bool)
    返回 result 数组的布尔视图
@cython.wraparound(False)
@cython.boundscheck(False)
def scalar_binop(object[:] values, object val, object op) -> ndarray:
    """
    Apply the given binary operator `op` between each element of the array
    `values` and the scalar `val`.

    Parameters
    ----------
    values : ndarray[object]
        输入的数组，包含对象类型的元素
    val : object
        用于执行操作的标量值
    op : binary operator
        二元操作符函数

    Returns
    -------
    result : ndarray[object]
        包含操作结果的对象数组
    """
    cdef:
        Py_ssize_t i, n = len(values)  # 获取数组长度
        object[::1] result  # 结果数组的声明
        object x  # 临时变量

    result = np.empty(n, dtype=object)  # 创建对象数组，用于存储结果
    if val is None or is_nan(val):  # 检查是否需要处理 NaN 值
        result[:] = val  # 将结果数组填充为 val
        return result.base  # 返回底层的 np.ndarray 对象

    for i in range(n):
        x = values[i]  # 获取数组中的元素
        if x is None or is_nan(x):  # 检查当前元素是否为 None 或 NaN
            result[i] = x  # 若是，则将结果数组对应位置设为当前元素
        else:
            result[i] = op(x, val)  # 使用给定的操作符函数对元素和标量值进行操作

    return maybe_convert_bool(result.base)[0]  # 转换布尔值并返回底层 np.ndarray


@cython.wraparound(False)
@cython.boundscheck(False)
def vec_binop(object[:] left, object[:] right, object op) -> ndarray:
    """
    Apply the given binary operator `op` pointwise to the elements of
    arrays `left` and `right`.

    Parameters
    ----------
    left : ndarray[object]
        包含对象类型元素的左操作数数组
    right : ndarray[object]
        包含对象类型元素的右操作数数组
    op : binary operator
        二元操作符函数

    Returns
    -------
    result : ndarray[object]
        包含操作结果的对象数组
    """
    cdef:
        Py_ssize_t i, n = len(left)  # 获取数组长度
        object[::1] result  # 结果数组的声明

    if n != <Py_ssize_t>len(right):  # 检查左右数组长度是否相等
        raise ValueError(f"Arrays were different lengths: {n} vs {len(right)}")

    result = np.empty(n, dtype=object)  # 创建对象数组，用于存储结果

    for i in range(n):
        x = left[i]  # 获取左数组的当前元素
        y = right[i]  # 获取右数组的当前元素
        try:
            result[i] = op(x, y)  # 尝试对当前元素执行给定的操作符函数
        except TypeError:
            if x is None or is_nan(x):  # 如果左侧元素是 None 或 NaN
                result[i] = x  # 将结果数组对应位置设为左侧元素
            elif y is None or is_nan(y):  # 如果右侧元素是 None 或 NaN
                result[i] = y  # 将结果数组对应位置设为右侧元素
            else:
                raise  # 抛出异常，表示无法执行操作

    return maybe_convert_bool(result.base)[0]  # 转换布尔值并返回底层 np.ndarray


def maybe_convert_bool(ndarray[object] arr,
                       true_values=None,
                       false_values=None,
                       convert_to_masked_nullable=False
                       ) -> tuple[np.ndarray, np.ndarray | None]:
    cdef:
        Py_ssize_t i, n  # 定义整数变量
        ndarray[uint8_t] result  # 布尔结果数组
        ndarray[uint8_t] mask  # 掩码数组
        object val  # 通用对象变量
        set true_vals, false_vals  # 存储真值和假值的集合
        bint has_na = False  # 表示是否存在 NaN 的布尔值

    n = len(arr)  # 获取数组长度
    result = np.empty(n, dtype=np.uint8)  # 创建布尔结果数组
    mask = np.zeros(n, dtype=np.uint8)  # 创建全零掩码数组
    # 设置默认的真值和假值
    true_vals = {"True", "TRUE", "true"}
    false_vals = {"False", "FALSE", "false"}

    if true_values is not None:
        true_vals = true_vals | set(true_values)  # 将用户提供的真值添加到集合中

    if false_values is not None:
        false_vals = false_vals | set(false_values)  # 将用户提供的假值添加到集合中
    # 遍历数组 arr 中的元素，索引从 0 到 n-1
    for i in range(n):
        # 获取数组 arr 中索引为 i 的元素
        val = arr[i]

        # 检查当前元素是否为布尔类型
        if isinstance(val, bool):
            # 如果当前元素为 True，则在结果数组的相同位置设置为 1
            if val is True:
                result[i] = 1
            # 如果当前元素为 False，则在结果数组的相同位置设置为 0
            else:
                result[i] = 0
        # 如果当前元素在 true_vals 集合中，则在结果数组的相同位置设置为 1
        elif val in true_vals:
            result[i] = 1
        # 如果当前元素在 false_vals 集合中，则在结果数组的相同位置设置为 0
        elif val in false_vals:
            result[i] = 0
        # 如果当前元素为 NaN 或者 None，则在 mask 数组中相同位置设置为 1，
        # 在结果数组的相同位置设置为 0（该值不重要，将被 NaN 替换）
        elif is_nan(val) or val is None:
            mask[i] = 1
            result[i] = 0  # 这里的值并不重要，将被 NaN 替换
            has_na = True
        # 如果当前元素不满足以上条件，则返回原始数组 arr 和 None
        else:
            return (arr, None)

    # 如果存在缺失值（NaN 或者 None）
    if has_na:
        # 如果需要转换为带掩码的可空数组，则返回结果数组和掩码数组的视图
        if convert_to_masked_nullable:
            return (result.view(np.bool_), mask.view(np.bool_))
        # 否则，将结果数组视图转换为对象类型数组，并将 mask 应用于数组的 NaN 值
        else:
            arr = result.view(np.bool_).astype(object)
            np.putmask(arr, mask, np.nan)
            return (arr, None)
    # 如果不存在缺失值，则返回结果数组的视图和 None
    else:
        return (result.view(np.bool_), None)
```