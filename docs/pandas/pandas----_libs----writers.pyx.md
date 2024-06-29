# `D:\src\scipysrc\pandas\pandas\_libs\writers.pyx`

```
# 导入Cython模块
cimport cython
# 从Cython模块中导入Py_ssize_t
from cython cimport Py_ssize_t
# 导入NumPy模块
import numpy as np

# 从Cython模块中导入CPython相关内容
from cpython cimport (
    PyBytes_GET_SIZE,
    PyUnicode_GET_LENGTH,
)
# 从NumPy模块中导入ndarray和uint8_t
from numpy cimport (
    ndarray,
    uint8_t,
)

# 定义pandas_string类型的别名
ctypedef fused pandas_string:
    str
    bytes

# 使用Cython装饰器设置边界检查为False
@cython.boundscheck(False)
# 使用Cython装饰器设置数组访问越界检查为False
@cython.wraparound(False)
# 定义写入CSV行的函数
def write_csv_rows(
    list data,
    ndarray data_index,
    Py_ssize_t nlevels,
    ndarray cols,
    object writer
) -> None:
    """
    将给定数据写入writer对象，尽可能进行预分配以提高性能。

    参数
    ----------
    data : list[ArrayLike]
    data_index : ndarray
    nlevels : int
    cols : ndarray
    writer : _csv.writer
    """
    # 在粗略测试中，N>100几乎没有边际改进
    cdef:
        Py_ssize_t i, j = 0, k = len(data_index), N = 100, ncols = len(cols)
        list rows

    # 预分配行
    rows = [[None] * (nlevels + ncols) for _ in range(N)]

    # 根据nlevels的不同情况进行不同的处理
    if nlevels == 1:
        for j in range(k):
            row = rows[j % N]
            row[0] = data_index[j]
            for i in range(ncols):
                row[1 + i] = data[i][j]

            if j >= N - 1 and j % N == N - 1:
                writer.writerows(rows)
    elif nlevels > 1:
        for j in range(k):
            row = rows[j % N]
            row[:nlevels] = list(data_index[j])
            for i in range(ncols):
                row[nlevels + i] = data[i][j]

            if j >= N - 1 and j % N == N - 1:
                writer.writerows(rows)
    else:
        for j in range(k):
            row = rows[j % N]
            for i in range(ncols):
                row[i] = data[i][j]

            if j >= N - 1 and j % N == N - 1:
                writer.writerows(rows)

    if j >= 0 and (j < N - 1 or (j % N) != N - 1):
        writer.writerows(rows[:((j + 1) % N)])

# 使用Cython装饰器设置边界检查为False
@cython.boundscheck(False)
# 使用Cython装饰器设置数组访问越界检查为False
@cython.wraparound(False)
# 定义将JSON转换为行的函数
def convert_json_to_lines(arr: str) -> str:
    """
    将逗号分隔的JSON替换为换行符，特别注意引号和括号
    """
    cdef:
        # 定义变量和数组
        Py_ssize_t i = 0, num_open_brackets_seen = 0, length
        bint in_quotes = False, is_escaping = False
        ndarray[uint8_t, ndim=1] narr
        unsigned char val, newline, comma, left_bracket, right_bracket, quote
        unsigned char backslash

    # 定义换行符、逗号、左括号、右括号、引号和反斜杠的ASCII码
    newline = ord("\n")
    comma = ord(",")
    left_bracket = ord("{")
    right_bracket = ord("}")
    quote = ord('"')
    backslash = ord("\\")

    # 将字符串转换为UTF-8编码的字节数组
    narr = np.frombuffer(arr.encode("utf-8"), dtype="u1").copy()
    length = narr.shape[0]
    # 遍历输入数组中的每一个元素
    for i in range(length):
        # 获取当前索引位置的元素值
        val = narr[i]
        # 检查是否遇到引号，并且不是转义状态下
        if val == quote and i > 0 and not is_escaping:
            # 切换引号状态（进入/退出引号）
            in_quotes = ~in_quotes
        # 检查是否遇到反斜杠或者处于转义状态
        if val == backslash or is_escaping:
            # 切换转义状态
            is_escaping = ~is_escaping
        # 检查是否遇到逗号，如果需要替换为换行符
        if val == comma:  # commas that should be \n
            # 检查是否处于括号外且不在引号内，然后替换逗号为换行符
            if num_open_brackets_seen == 0 and not in_quotes:
                narr[i] = newline
        # 检查是否遇到左括号，如果不在引号内则增加括号计数
        elif val == left_bracket:
            if not in_quotes:
                num_open_brackets_seen += 1
        # 检查是否遇到右括号，如果不在引号内则减少括号计数
        elif val == right_bracket:
            if not in_quotes:
                num_open_brackets_seen -= 1

    # 将处理后的数组转换为字节序列，并按照 UTF-8 编码解码成字符串后返回，末尾追加换行符
    return narr.tobytes().decode("utf-8") + "\n"  # GH:36888
# 定义一个 Cython 函数，用于计算一维字符串数组中最长字符串的长度
@cython.boundscheck(False)
@cython.wraparound(False)
def max_len_string_array(pandas_string[:] arr) -> Py_ssize_t:
    """
    Return the maximum size of elements in a 1-dim string array.
    返回一维字符串数组中元素的最大长度。
    """
    cdef:
        Py_ssize_t i, m = 0, wlen = 0, length = arr.shape[0]
        pandas_string val

    for i in range(length):
        val = arr[i]  # 获取数组中的当前元素
        wlen = word_len(val)  # 调用 word_len 函数计算当前元素的长度

        if wlen > m:
            m = wlen  # 更新当前最大长度

    return m  # 返回最大长度


# 定义一个 Cython 内联函数，用于获取字符串或字节串的长度
cpdef inline Py_ssize_t word_len(object val):
    """
    Return the maximum length of a string or bytes value.
    返回字符串或字节串的最大长度。
    """
    cdef:
        Py_ssize_t wlen = 0

    if isinstance(val, str):
        wlen = PyUnicode_GET_LENGTH(val)  # 获取字符串的长度
    elif isinstance(val, bytes):
        wlen = PyBytes_GET_SIZE(val)  # 获取字节串的长度

    return wlen  # 返回长度


# ------------------------------------------------------------------
# PyTables 辅助函数


# 定义一个 Cython 函数，用于将数组中的特定值替换为 np.nan
@cython.boundscheck(False)
@cython.wraparound(False)
def string_array_replace_from_nan_rep(
    ndarray[object, ndim=1] arr,
    object nan_rep,
) -> None:
    """
    Replace the values in the array with np.nan if they are nan_rep.
    如果数组中的值等于 nan_rep，则将其替换为 np.nan。
    """
    cdef:
        Py_ssize_t length = len(arr), i = 0

    for i in range(length):
        if arr[i] == nan_rep:
            arr[i] = np.nan  # 将当前位置的值替换为 np.nan
```