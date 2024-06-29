# `.\numpy\numpy\_core\src\multiarray\textreading\str_to_int.c`

```py
#include <Python.h>

# 包含 Python.h 头文件，这是与 Python 解释器通信的主要头文件


#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#include "lowlevel_strided_loops.h"

# 定义 NPY_NO_DEPRECATED_API 宏，设置为当前 NumPy API 版本，用以禁用已弃用的 API
# 定义 _MULTIARRAYMODULE 宏，可能用于指定编译模块的一些特定行为
# 包含 lowlevel_strided_loops.h 头文件，该头文件似乎提供了低级别的循环操作支持


#include <string.h>
#include "textreading/str_to_int.h"
#include "textreading/parser_config.h"
#include "conversions.h"  /* For the deprecated parse-via-float path */

# 包含 string.h 头文件，提供字符串操作函数的声明
# 包含 textreading/str_to_int.h 头文件，可能定义了字符串到整数的转换函数
# 包含 textreading/parser_config.h 头文件，可能定义了解析器的配置信息
# 包含 conversions.h 头文件，用于废弃的通过浮点数解析路径，可能包含一些转换相关的函数声明


const char *deprecation_msg = (
        "loadtxt(): Parsing an integer via a float is deprecated.  To avoid "
        "this warning, you can:\n"
        "    * make sure the original data is stored as integers.\n"
        "    * use the `converters=` keyword argument.  If you only use\n"
        "      NumPy 1.23 or later, `converters=float` will normally work.\n"
        "    * Use `np.loadtxt(...).astype(np.int64)` parsing the file as\n"
        "      floating point and then convert it.  (On all NumPy versions.)\n"
        "  (Deprecated NumPy 1.23)");

# 定义字符串常量 deprecation_msg，内容为一条关于 loadtxt() 函数的警告信息，提示整数通过浮点数解析已废弃


#define DECLARE_TO_INT(intw, INT_MIN, INT_MAX, byteswap_unaligned)          \
    NPY_NO_EXPORT int                                                       \
    npy_to_##intw(PyArray_Descr *descr,                                     \
            const Py_UCS4 *str, const Py_UCS4 *end, char *dataptr,          \
            parser_config *pconfig)                                         \

# 定义 DECLARE_TO_INT 宏，可能用于声明一个函数 npy_to_##intw，该函数转换 PyArray_Descr 描述符指定的数据类型为整数
# 该函数接受一些参数，包括描述符、字符串起始和结束位置、数据指针和解析器配置对象```c
    NPY_NO_EXPORT int                                                       \

# 使用 NPY_NO_EXPORT 宏，指示该函数在编译时不导出，可能用于内部实现


    npy_to_##intw(PyArray_Descr *descr,                                     \

# 定义函数 npy_to_##intw，接受 PyArray_Descr 描述符指针 descr 参数，用于描述数据类型


            const Py_UCS4 *str, const Py_UCS4 *end, char *dataptr,          \

# 接受 const Py_UCS4 类型的 str 和 end 参数，用于指定字符串的起始和结束位置，这些参数可能是 Unicode 字符串


            parser_config *pconfig)                                         \

# 接受 parser_config 指针类型的 pconfig 参数，可能用于配置解析器的行为Here are the annotated comments for each line of the provided C code snippet:


#include <Python.h>

# Include the Python.h header file, essential for interacting with the Python interpreter.


#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#include "lowlevel_strided_loops.h"

# Define NPY_NO_DEPRECATED_API to the current NumPy API version, disabling deprecated APIs.
# Define _MULTIARRAYMODULE, likely used for module-specific compilation behaviors.
# Include the lowlevel_strided_loops.h header file, which supports low-level strided operations.


#include <string.h>
#include "textreading/str_to_int.h"
#include "textreading/parser_config.h"
#include "conversions.h"  /* For the deprecated parse-via-float path */

# Include string.h for string manipulation functions.
# Include textreading/str_to_int.h, which likely provides functions for converting strings to integers.
# Include textreading/parser_config.h, possibly defining configuration for a parser.
# Include conversions.h, which handles conversions, including deprecated methods like parse-via-float.


const char *deprecation_msg = (
        "loadtxt(): Parsing an integer via a float is deprecated.  To avoid "
        "this warning, you can:\n"
        "    * make sure the original data is stored as integers.\n"
        "    * use the `converters=` keyword argument.  If you only use\n"
        "      NumPy 1.23 or later, `converters=float` will normally work.\n"
        "    * Use `np.loadtxt(...).astype(np.int64)` parsing the file as\n"
        "      floating point and then convert it.  (On all NumPy versions.)\n"
        "  (Deprecated NumPy 1.23)");

# Define deprecation_msg as a constant string, containing a deprecation warning message for the loadtxt() function regarding parsing integers via float, and providing alternatives to avoid the warning.


#define DECLARE_TO_INT(intw, INT_MIN, INT_MAX, byteswap_unaligned)          \
    NPY_NO_EXPORT int                                                       \
    npy_to_##intw(PyArray_Descr *descr,                                     \
            const Py_UCS4 *str, const Py_UCS4 *end, char *dataptr,          \
            parser_config *pconfig)                                         \

# Define DECLARE_TO_INT macro, likely for declaring a function named npy_to_##intw, which converts data described by PyArray_Descr *descr into integers.
# This function accepts parameters such as descr (descriptor), str and end (start and end of string), dataptr (data pointer), and pconfig (parser configuration).


    NPY_NO_EXPORT int                                                       \

# Utilize NPY_NO_EXPORT macro to ensure the function is not exported externally during compilation.


    npy_to_##intw(PyArray_Descr *descr,                                     \

# Define npy_to_##intw function, accepting PyArray_Descr *descr to describe the data type.


            const Py_UCS4 *str, const Py_UCS4 *end, char *dataptr,          \

# Accept const Py_UCS4 *str and *end as parameters to specify the start and end of a Unicode string.


            parser_config *pconfig)                                         \

# Accept parser_config *pconfig to configure parser behavior.
    {
        // 声明一个变量用于存储解析后的整数值
        int64_t parsed;
        // 声明一个变量 x，用于存储最终转换后的整数值，类型为 intw##_t
        intw##_t x;

        // 检查字符串转换为整数是否失败
        if (NPY_UNLIKELY(
                str_to_int64(str, end, INT_MIN, INT_MAX, &parsed) < 0)) {
            /* DEPRECATED 2022-07-03, NumPy 1.23 */
            // 如果字符串转换为整数失败，则尝试将其解析为浮点数
            double fval;
            // 创建一个双精度浮点数的 NumPy 数组描述符对象
            PyArray_Descr *d_descr = PyArray_DescrFromType(NPY_DOUBLE);
            Py_DECREF(d_descr);  /* borrowed */

            // 如果将字符串转换为浮点数失败，则返回错误
            if (npy_to_double(d_descr, str, end, (char *)&fval, pconfig) < 0) {
                return -1;
            }

            // 如果之前没有发出浮点数转整数的警告，则发出警告
            if (!pconfig->gave_int_via_float_warning) {
                pconfig->gave_int_via_float_warning = true;
                // 发出 DeprecationWarning，指明相关信息
                if (PyErr_WarnEx(PyExc_DeprecationWarning,
                        deprecation_msg, 3) < 0) {
                    return -1;
                }
            }
            pconfig->gave_int_via_float_warning = true;

            // 将浮点数转换为 intw##_t 类型的整数，并赋给变量 x
            x = (intw##_t)fval;
        }
        else {
            // 如果字符串成功转换为整数，则直接赋值给变量 x
            x = (intw##_t)parsed;
        }

        // 将 x 的值复制到 dataptr 指向的内存中
        memcpy(dataptr, &x, sizeof(x));

        // 如果描述符的字节顺序不是本地字节顺序，则进行字节交换
        if (!PyArray_ISNBO(descr->byteorder)) {
            byteswap_unaligned(dataptr);
        }

        // 返回操作成功的标志
        return 0;
    }
/*
#define DECLARE_TO_UINT(uintw, UINT_MAX, byteswap_unaligned)                \
    NPY_NO_EXPORT int                                                       \
    npy_to_##uintw(PyArray_Descr *descr,                                    \
            const Py_UCS4 *str, const Py_UCS4 *end, char *dataptr,          \
            parser_config *pconfig)                                         \
    {                                                                       \
        uint64_t parsed;                                                    \
        uintw##_t x;                                                        \
                                                                            \
        if (NPY_UNLIKELY(                                                   \
                str_to_uint64(str, end, UINT_MAX, &parsed) < 0)) {          \
            // DEPRECATED 2022-07-03, NumPy 1.23                              \
            double fval;                                                    \
            PyArray_Descr *d_descr = PyArray_DescrFromType(NPY_DOUBLE);     \
            Py_DECREF(d_descr);  /* borrowed */                             \
            if (npy_to_double(d_descr, str, end, (char *)&fval, pconfig) < 0) { \
                return -1;                                                  \
            }                                                               \
            if (!pconfig->gave_int_via_float_warning) {                     \
                pconfig->gave_int_via_float_warning = true;                 \
                if (PyErr_WarnEx(PyExc_DeprecationWarning,                  \
                        deprecation_msg, 3) < 0) {                          \
                    return -1;                                              \
                }                                                           \
            }                                                               \
            pconfig->gave_int_via_float_warning = true;                     \
            x = (uintw##_t)fval;                                            \
        }                                                                   \
        else {                                                              \
            x = (uintw##_t)parsed;                                          \
        }                                                                   \
        memcpy(dataptr, &x, sizeof(x));                                     \
        if (!PyArray_ISNBO(descr->byteorder)) {                             \
            byteswap_unaligned(dataptr);                                    \
        }                                                                   \
        return 0;                                                           \
    }
*/

/*
#define byteswap_nothing(ptr)
*/

// 定义一个将字符串转换为指定类型整数的函数
DECLARE_TO_INT(int8, INT8_MIN, INT8_MAX, byteswap_nothing)
// 定义一个将字符串转换为16位有符号整数的函数，包括字节交换
DECLARE_TO_INT(int16, INT16_MIN, INT16_MAX, npy_bswap2_unaligned)
# 定义一个宏，用于将给定变量转换为特定类型的整数类型(int32)，并进行范围检查，使用指定的字节交换操作函数(npy_bswap4_unaligned)
DECLARE_TO_INT(int32, INT32_MIN, INT32_MAX, npy_bswap4_unaligned)

# 定义一个宏，用于将给定变量转换为特定类型的整数类型(int64)，并进行范围检查，使用指定的字节交换操作函数(npy_bswap8_unaligned)
DECLARE_TO_INT(int64, INT64_MIN, INT64_MAX, npy_bswap8_unaligned)

# 定义一个宏，用于将给定变量转换为特定类型的无符号整数类型(uint8)，并进行范围检查，不使用字节交换操作函数(byteswap_nothing)
DECLARE_TO_UINT(uint8, UINT8_MAX, byteswap_nothing)

# 定义一个宏，用于将给定变量转换为特定类型的无符号整数类型(uint16)，并进行范围检查，使用指定的字节交换操作函数(npy_bswap2_unaligned)
DECLARE_TO_UINT(uint16, UINT16_MAX, npy_bswap2_unaligned)

# 定义一个宏，用于将给定变量转换为特定类型的无符号整数类型(uint32)，并进行范围检查，使用指定的字节交换操作函数(npy_bswap4_unaligned)
DECLARE_TO_UINT(uint32, UINT32_MAX, npy_bswap4_unaligned)

# 定义一个宏，用于将给定变量转换为特定类型的无符号整数类型(uint64)，并进行范围检查，使用指定的字节交换操作函数(npy_bswap8_unaligned)
DECLARE_TO_UINT(uint64, UINT64_MAX, npy_bswap8_unaligned)
```