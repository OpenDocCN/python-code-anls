# `.\numpy\numpy\_core\src\multiarray\textreading\str_to_int.h`

```py
/*
 * 定义头文件防止重复包含
 */
#ifndef NUMPY_CORE_SRC_MULTIARRAY_TEXTREADING_STR_TO_INT_H_
#define NUMPY_CORE_SRC_MULTIARRAY_TEXTREADING_STR_TO_INT_H_

/*
 * 定义宏以避免使用已废弃的 NumPy API，并指定使用的 API 版本
 */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

/*
 * 定义宏 _MULTIARRAYMODULE，用于标识本模块
 */
#define _MULTIARRAYMODULE

/*
 * 包含 NumPy 数组数据类型相关的头文件
 */
#include "numpy/ndarraytypes.h"

/*
 * 包含文本解析器配置文件的头文件
 */
#include "textreading/parser_config.h"


/*
 * 下面的两个字符串转换函数在 Pandas 中基本等效。
 * 它们在这里的头文件中定义，以确保它们可以在其他函数中轻松内联。
 * 与 Pandas 不同，传入结束指针（不依赖于 \0）并返回 0 或 -1。
 *
 * 实际函数在下面的宏模板定义中。
 */

/*
 * 将字符串转换为 int64_t 类型的整数。
 */
NPY_FINLINE int
str_to_int64(
        const Py_UCS4 *p_item, const Py_UCS4 *p_end,
        int64_t int_min, int64_t int_max, int64_t *result)
{
    const Py_UCS4 *p = (const Py_UCS4 *)p_item;
    bool isneg = 0;
    int64_t number = 0;

    // 跳过前导空格。
    while (Py_UNICODE_ISSPACE(*p)) {
        ++p;
    }

    // 处理符号。
    if (*p == '-') {
        isneg = true;
        ++p;
    }
    else if (*p == '+') {
        p++;
    }

    // 检查是否有第一个数字。
    if (!isdigit(*p)) {
        return -1;
    }

    if (isneg) {
        // 如果数字大于 pre_min，至少还可以处理一个更多的数字而不会溢出。
        int dig_pre_min = -(int_min % 10);
        int64_t pre_min = int_min / 10;

        // 处理数字。
        int d = *p;
        while (isdigit(d)) {
            if ((number > pre_min) || ((number == pre_min) && (d - '0' <= dig_pre_min))) {
                number = number * 10 - (d - '0');
                d = *++p;
            }
            else {
                return -1;
            }
        }
    }
    else {
        // 如果数字小于 pre_max，至少还可以处理一个更多的数字而不会溢出。
        int64_t pre_max = int_max / 10;
        int dig_pre_max = int_max % 10;

        // 处理数字。
        int d = *p;
        while (isdigit(d)) {
            if ((number < pre_max) || ((number == pre_max) && (d - '0' <= dig_pre_max))) {
                number = number * 10 + (d - '0');
                d = *++p;
            }
            else {
                return -1;
            }
        }
    }

    // 跳过尾随空格。
    while (Py_UNICODE_ISSPACE(*p)) {
        ++p;
    }

    // 是否使用了所有字符？
    if (p != p_end) {
        return -1;
    }

    // 将结果存入 result，并返回 0 表示成功。
    *result = number;
    return 0;
}

/*
 * 将字符串转换为 uint64_t 类型的无符号整数。
 */
NPY_FINLINE int
str_to_uint64(
        const Py_UCS4 *p_item, const Py_UCS4 *p_end,
        uint64_t uint_max, uint64_t *result)
{
    const Py_UCS4 *p = (const Py_UCS4 *)p_item;
    uint64_t number = 0;
    int d;

    // 跳过前导空格。
    while (Py_UNICODE_ISSPACE(*p)) {
        ++p;
    }

    // 处理符号。
    if (*p == '-') {
        return -1;
    }
    if (*p == '+') {
        p++;
    }

    // 检查是否有第一个数字。
    if (!isdigit(*p)) {
        return -1;
    }
    // 如果数字小于 pre_max，意味着还可以处理至少一个更多的数字而不会溢出。
    uint64_t pre_max = uint_max / 10;
    int dig_pre_max = uint_max % 10;

    // 处理数字。
    d = *p;
    while (isdigit(d)) {
        // 如果满足条件：数字小于 pre_max，或者数字等于 pre_max 且当前数字不超过 dig_pre_max，则继续处理。
        if ((number < pre_max) || ((number == pre_max) && (d - '0' <= dig_pre_max))) {
            number = number * 10 + (d - '0');
            d = *++p;
        }
        else {
            return -1;  // 如果超过了预设的最大值范围，返回错误。
        }
    }

    // 跳过尾部的空格。
    while (Py_UNICODE_ISSPACE(*p)) {
        ++p;
    }

    // 是否已经使用了所有的字符？
    if (p != p_end) {
        return -1;  // 如果没有使用完所有字符，返回错误。
    }

    *result = number;  // 将解析出的数字存入 result 指针指向的位置。
    return 0;  // 返回成功状态。
# 定义一个宏函数，用于声明将字符串转换为特定整数类型的函数原型

#define DECLARE_TO_INT_PROTOTYPE(intw)                                  \
    #ifndef NPY_NO_EXPORT
    // 如果 NPY_NO_EXPORT 未定义，则定义为 int
    NPY_NO_EXPORT int                                                   \
    #else
    // 否则，空行
    #endif
    // 定义一个函数原型 npy_to_##intw，接受描述符 descr、Unicode 字符串 str、结束位置 end、数据指针 dataptr，以及解析器配置 pconfig
    npy_to_##intw(PyArray_Descr *descr,                                     \
            const Py_UCS4 *str, const Py_UCS4 *end, char *dataptr,      \
            parser_config *pconfig);
// 使用宏 DECLARE_TO_INT_PROTOTYPE 分别声明以下整型转换函数的原型：

DECLARE_TO_INT_PROTOTYPE(int8)
DECLARE_TO_INT_PROTOTYPE(int16)
DECLARE_TO_INT_PROTOTYPE(int32)
DECLARE_TO_INT_PROTOTYPE(int64)

DECLARE_TO_INT_PROTOTYPE(uint8)
DECLARE_TO_INT_PROTOTYPE(uint16)
DECLARE_TO_INT_PROTOTYPE(uint32)
DECLARE_TO_INT_PROTOTYPE(uint64)

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_TEXTREADING_STR_TO_INT_H_ */
```