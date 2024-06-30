# `D:\src\scipysrc\scipy\scipy\ndimage\src\_cytest.pxd`

```
# 导入特定的 C 扩展类型 `npy_intp` 作为 `intp`
from numpy cimport npy_intp as intp

# 定义一个 C 语言级别的函数 `_filter1d`，该函数接受以下参数：
# - `input_line`: 指向 `double` 类型数组的指针，表示输入线性数据
# - `input_length`: `intp` 类型，表示输入数据的长度
# - `output_line`: 指向 `double` 类型数组的指针，表示输出线性数据
# - `output_length`: `intp` 类型，表示输出数据的长度
# - `callback_data`: 指向任意类型数据的指针，用于回调函数的附加数据
# 函数是无异常抛出的
cdef int _filter1d(double *input_line, intp input_length, double *output_line,
               intp output_length, void *callback_data) noexcept

# 定义另一个 C 语言级别的函数 `_filter2d`，该函数接受以下参数：
# - `buffer`: 指向 `double` 类型数组的指针，表示输入的二维缓冲区
# - `filter_size`: `intp` 类型，表示滤波器的大小
# - `res`: 指向 `double` 类型数组的指针，表示输出结果
# - `callback_data`: 指向任意类型数据的指针，用于回调函数的附加数据
# 函数是无异常抛出的
cdef int _filter2d(double *buffer, intp filter_size, double *res,
               void *callback_data) noexcept

# 定义另一个 C 语言级别的函数 `_transform`，该函数接受以下参数：
# - `output_coordinates`: 指向 `intp` 类型数组的指针，表示输出坐标
# - `input_coordinates`: 指向 `double` 类型数组的指针，表示输入坐标
# - `output_rank`: `int` 类型，表示输出数据的秩（维度）
# - `input_rank`: `int` 类型，表示输入数据的秩（维度）
# - `callback_data`: 指向任意类型数据的指针，用于回调函数的附加数据
# 函数是无异常抛出的
cdef int _transform(intp *output_coordinates, double *input_coordinates,
                int output_rank, int input_rank, void *callback_data) noexcept
```