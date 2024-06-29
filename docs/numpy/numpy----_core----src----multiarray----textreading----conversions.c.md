# `.\numpy\numpy\_core\src\multiarray\textreading\conversions.c`

```py
/*
 * Include Python.h header file to interface with Python API.
 */
#include <Python.h>

/*
 * Include necessary standard C library headers.
 */
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>

/*
 * Define NPY_NO_DEPRECATED_API to specify the version of NumPy API to use.
 */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

/*
 * Include the lowlevel_strided_loops.h header file for low-level strided loops.
 */
#include "lowlevel_strided_loops.h"

/*
 * Include NumPy headers for mathematical functions (npy_math.h),
 * type conversions (conversions.h), and string to integer conversion (str_to_int.h).
 */
#include "numpy/npy_math.h"
#include "conversions.h"
#include "str_to_int.h"

/*
 * Include array_coercion.h for handling array coercion operations.
 */
#include "array_coercion.h"


/*
 * Coercion to boolean is done via integer right now.
 */
NPY_NO_EXPORT int
npy_to_bool(PyArray_Descr *NPY_UNUSED(descr),
        const Py_UCS4 *str, const Py_UCS4 *end, char *dataptr,
        parser_config *NPY_UNUSED(pconfig))
{
    int64_t res;
    /*
     * Convert UCS4 string to int64 using str_to_int64 function,
     * and store the result in res.
     */
    if (str_to_int64(str, end, INT64_MIN, INT64_MAX, &res) < 0) {
        return -1;
    }
    /*
     * Store the boolean value (cast from res) in dataptr.
     */
    *dataptr = (char)(res != 0);
    return 0;
}


/*
 * In order to not pack a whole copy of a floating point parser, we copy the
 * result into ascii and call the Python one.  Float parsing isn't super quick
 * so this is not terrible, but avoiding it would speed up things.
 *
 * Also note that parsing the first float of a complex will copy the whole
 * string to ascii rather than just the first part.
 * TODO: A tweak of the break might be a simple mitigation there.
 *
 * @param str The UCS4 string to parse
 * @param end Pointer to the end of the string
 * @param skip_trailing_whitespace If false does not skip trailing whitespace
 *        (used by the complex parser).
 * @param result Output stored as double value.
 */
static inline int
double_from_ucs4(
        const Py_UCS4 *str, const Py_UCS4 *end,
        bool strip_whitespace, double *result, const Py_UCS4 **p_end)
{
    /* skip leading whitespace */
    if (strip_whitespace) {
        while (Py_UNICODE_ISSPACE(*str)) {
            str++;
        }
    }
    /*
     * If the string is empty or consists only of whitespace, return -1
     * indicating it's not a valid floating point number.
     */
    if (str == end) {
        return -1;
    }

    /* We convert to ASCII for the Python parser, use stack if small: */
    char stack_buf[128];
    char *heap_buf = NULL;
    char *ascii = stack_buf;

    /*
     * Determine the length of the UCS4 string to allocate buffer.
     * If length exceeds 128, allocate heap memory; otherwise, use stack buffer.
     */
    size_t str_len = end - str + 1;
    if (str_len > 128) {
        heap_buf = PyMem_MALLOC(str_len);
        if (heap_buf == NULL) {
            PyErr_NoMemory();
            return -1;
        }
        ascii = heap_buf;
    }
    char *c = ascii;
    /*
     * Convert UCS4 characters to ASCII characters.
     * If encountering non-ASCII characters, ignore them.
     */
    for (; str < end; str++, c++) {
        if (NPY_UNLIKELY(*str >= 128)) {
            /* Character cannot be used, stop conversion and update end pointer. */
            end = str;
            break;
        }
        *c = (char)(*str);
    }
    *c = '\0';

    /*
     * Use PyOS_string_to_double to parse ASCII string to double.
     * Store the parsed value in result, and update end_parsed for tracking
     * the position of the last parsed character.
     */
    char *end_parsed;
    *result = PyOS_string_to_double(ascii, &end_parsed, NULL);
    /* Rewind `end` to the first UCS4 character not parsed: */
    end = end - (c - end_parsed);

    /*
     * Free heap buffer if allocated.
     */
    PyMem_FREE(heap_buf);

    /*
     * Check if PyOS_string_to_double encountered an error.
     */
    if (*result == -1. && PyErr_Occurred()) {
        return -1;
    }

    /*
     * If strip_whitespace is true, skip any remaining whitespace characters in end.
     */
    if (strip_whitespace) {
        while (Py_UNICODE_ISSPACE(*end)) {
            end++;
        }
    }
    *p_end = end;
    return 0;
}
// 将传入的字符串解析为浮点数，并存储为单精度浮点数格式到指定位置
npy_to_float(PyArray_Descr *descr,
        const Py_UCS4 *str, const Py_UCS4 *end, char *dataptr,
        parser_config *NPY_UNUSED(pconfig))
{
    double double_val;
    const Py_UCS4 *p_end;

    // 使用 double_from_ucs4 函数将字符串解析为双精度浮点数
    if (double_from_ucs4(str, end, true, &double_val, &p_end) < 0) {
        return -1;  // 解析失败则返回 -1
    }

    // 检查解析结束位置是否和字符串末尾相同，如果不同则返回 -1
    if (p_end != end) {
        return -1;
    }

    // 将双精度浮点数转换为单精度浮点数，并复制到指定位置
    float val = (float)double_val;
    memcpy(dataptr, &val, sizeof(float));

    // 如果字节顺序不是网络字节顺序，则进行字节交换以适应网络字节顺序
    if (!PyArray_ISNBO(descr->byteorder)) {
        npy_bswap4_unaligned(dataptr);
    }

    return 0;  // 返回成功标志
}


// 将传入的字符串解析为双精度浮点数，并存储到指定位置
NPY_NO_EXPORT int
npy_to_double(PyArray_Descr *descr,
        const Py_UCS4 *str, const Py_UCS4 *end, char *dataptr,
        parser_config *NPY_UNUSED(pconfig))
{
    double val;
    const Py_UCS4 *p_end;

    // 使用 double_from_ucs4 函数将字符串解析为双精度浮点数
    if (double_from_ucs4(str, end, true, &val, &p_end) < 0) {
        return -1;  // 解析失败则返回 -1
    }

    // 检查解析结束位置是否和字符串末尾相同，如果不同则返回 -1
    if (p_end != end) {
        return -1;
    }

    // 将双精度浮点数直接复制到指定位置
    memcpy(dataptr, &val, sizeof(double));

    // 如果字节顺序不是网络字节顺序，则进行字节交换以适应网络字节顺序
    if (!PyArray_ISNBO(descr->byteorder)) {
        npy_bswap8_unaligned(dataptr);
    }

    return 0;  // 返回成功标志
}


// 解析包含实部和虚部的复数字符串，存储实部和虚部到指定位置
static bool
to_complex_int(
        const Py_UCS4 *item, const Py_UCS4 *token_end,
        double *p_real, double *p_imag,
        Py_UCS4 imaginary_unit, bool allow_parens)
{
    const Py_UCS4 *p_end;
    bool unmatched_opening_paren = false;

    // 跳过可能的起始空白字符，处理可能的开头 '('
    while (Py_UNICODE_ISSPACE(*item)) {
        ++item;
    }
    if (allow_parens && (*item == '(')) {
        unmatched_opening_paren = true;
        ++item;

        // 允许括号内的空白字符，例如 "( 1j)"
        while (Py_UNICODE_ISSPACE(*item)) {
            ++item;
        }
    }

    // 解析实部的双精度浮点数
    if (double_from_ucs4(item, token_end, false, p_real, &p_end) < 0) {
        return false;  // 解析失败则返回 false
    }

    // 检查是否没有虚部，如果没有，则将虚部设为 0
    if (p_end == token_end) {
        *p_imag = 0.0;
        return !unmatched_opening_paren;  // 如果有未匹配的括号则返回 false
    }

    // 如果虚部部分以虚数单位开始，解析虚部
    if (*p_end == imaginary_unit) {
        *p_imag = *p_real;
        *p_real = 0.0;
        ++p_end;  // 跳过虚数单位字符
    }
    else if (*p_end == '+' || *p_end == '-') {
        // 如果虚部部分以正负号开始，解析剩余的虚部
        if (*p_end == '+') {
            ++p_end;  // 支持 '+-' 和 '++'
        }
        if (double_from_ucs4(p_end, token_end, false, p_imag, &p_end) < 0) {
            return false;  // 解析失败则返回 false
        }
        if (*p_end != imaginary_unit) {
            return false;  // 虚部部分未以虚数单位结束，返回 false
        }
        ++p_end;  // 跳过虚数单位字符
    }
    else {
        *p_imag = 0;  // 没有虚部
    }

    // 如果有未匹配的括号，允许括号内的空白字符
    if (unmatched_opening_paren) {
        while (Py_UNICODE_ISSPACE(*p_end)) {
            ++p_end;
        }
        if (*p_end == ')') {
            ++p_end;  // 如果匹配到右括号则跳过
        }
        else {
            return false;  // 括号未闭合，返回 false
        }
    }

    // 跳过可能的结尾空白字符
    while (Py_UNICODE_ISSPACE(*p_end)) {
        ++p_end;
    }

    // 检查是否已经解析到 token_end 结尾
    return p_end == token_end;
}
/*
 * 将 numpy 数据描述符 `descr` 中的字符串或 Unicode 数据转换为复数浮点数。
 * 
 * Parameters:
 * - descr: NumPy 数据描述符指针，描述要转换的数据类型和字节顺序
 * - str: 要转换的输入字符串或 Unicode 数据的起始位置
 * - end: 输入数据的结束位置
 * - dataptr: 指向输出数据缓冲区的指针
 * - pconfig: 解析器配置结构体指针，包含有关复数单位的信息
 * 
 * Returns:
 * - 成功返回 0，失败返回 -1
 */

NPY_NO_EXPORT int
npy_to_cfloat(PyArray_Descr *descr,
        const Py_UCS4 *str, const Py_UCS4 *end, char *dataptr,
        parser_config *pconfig)
{
    double real;
    double imag;

    // 调用辅助函数 to_complex_int 尝试将输入解析为复数整数
    bool success = to_complex_int(
            str, end, &real, &imag,
            pconfig->imaginary_unit, true);

    // 如果解析失败，返回 -1
    if (!success) {
        return -1;
    }

    // 创建 numpy 的复数浮点数结构 npy_complex64
    npy_complex64 val;
    npy_csetrealf(&val, (float) real);  // 设置实部
    npy_csetimagf(&val, (float) imag);  // 设置虚部

    // 将复数数据复制到输出缓冲区
    memcpy(dataptr, &val, sizeof(npy_complex64));

    // 如果数据字节顺序不是本地字节顺序，则进行字节交换
    if (!PyArray_ISNBO(descr->byteorder)) {
        npy_bswap4_unaligned(dataptr);       // 交换前四个字节
        npy_bswap4_unaligned(dataptr + 4);   // 交换后四个字节
    }

    // 返回转换结果
    return 0;
}


/*
 * 将 numpy 数据描述符 `descr` 中的字符串或 Unicode 数据转换为双精度复数浮点数。
 * 
 * Parameters:
 * - descr: NumPy 数据描述符指针，描述要转换的数据类型和字节顺序
 * - str: 要转换的输入字符串或 Unicode 数据的起始位置
 * - end: 输入数据的结束位置
 * - dataptr: 指向输出数据缓冲区的指针
 * - pconfig: 解析器配置结构体指针，包含有关复数单位的信息
 * 
 * Returns:
 * - 成功返回 0，失败返回 -1
 */

NPY_NO_EXPORT int
npy_to_cdouble(PyArray_Descr *descr,
        const Py_UCS4 *str, const Py_UCS4 *end, char *dataptr,
        parser_config *pconfig)
{
    double real;
    double imag;

    // 调用辅助函数 to_complex_int 尝试将输入解析为复数整数
    bool success = to_complex_int(
            str, end, &real, &imag, pconfig->imaginary_unit, true);

    // 如果解析失败，返回 -1
    if (!success) {
        return -1;
    }

    // 创建 numpy 的双精度复数浮点数结构 npy_complex128
    npy_complex128 val;
    npy_csetreal(&val, real);   // 设置实部
    npy_csetimag(&val, imag);   // 设置虚部

    // 将复数数据复制到输出缓冲区
    memcpy(dataptr, &val, sizeof(npy_complex128));

    // 如果数据字节顺序不是本地字节顺序，则进行字节交换
    if (!PyArray_ISNBO(descr->byteorder)) {
        npy_bswap8_unaligned(dataptr);       // 交换前八个字节
        npy_bswap8_unaligned(dataptr + 8);   // 交换后八个字节
    }

    // 返回转换结果
    return 0;
}


/*
 * 将 numpy 数据描述符 `descr` 中的字符串或 Unicode 数据转换为字节字符串。
 * 
 * Parameters:
 * - descr: NumPy 数据描述符指针，描述要转换的数据类型和字节顺序
 * - str: 要转换的输入字符串或 Unicode 数据的起始位置
 * - end: 输入数据的结束位置
 * - dataptr: 指向输出数据缓冲区的指针
 * - unused: 未使用的解析器配置结构体指针
 * 
 * Returns:
 * - 成功返回 0，失败返回 -1
 */

NPY_NO_EXPORT int
npy_to_string(PyArray_Descr *descr,
        const Py_UCS4 *str, const Py_UCS4 *end, char *dataptr,
        parser_config *NPY_UNUSED(unused))
{
    const Py_UCS4* c = str;
    size_t length = descr->elsize;

    // 遍历输入字符串或 Unicode 数据，逐字符转换为 Latin1 编码的字节
    for (size_t i = 0; i < length; i++) {
        if (c < end) {
            // 如果字符超出 Latin1 范围，返回解析错误
            if (NPY_UNLIKELY(*c > 255)) {
                /* TODO: Was UnicodeDecodeError, is unspecific error good? */
                return -1;
            }
            dataptr[i] = (Py_UCS1)(*c);  // 将 Unicode 字符转换为 Latin1 字节
            c++;
        }
        else {
            dataptr[i] = '\0';  // 如果输入结束，填充剩余空间为 '\0'
        }
    }
    // 返回转换结果
    return 0;
}


/*
 * 将 numpy 数据描述符 `descr` 中的字符串或 Unicode 数据转换为 Unicode 字符串。
 * 
 * Parameters:
 * - descr: NumPy 数据描述符指针，描述要转换的数据类型和字节顺序
 * - str: 要转换的输入字符串或 Unicode 数据的起始位置
 * - end: 输入数据的结束位置
 * - dataptr: 指向输出数据缓冲区的指针
 * - unused: 未使用的解析器配置结构体指针
 * 
 * Returns:
 * - 成功返回 0，失败返回 -1
 */

NPY_NO_EXPORT int
npy_to_unicode(PyArray_Descr *descr,
        const Py_UCS4 *str, const Py_UCS4 *end, char *dataptr,
        parser_config *NPY_UNUSED(unused))
{
    int length = descr->elsize / 4;  // 计算要复制的 Unicode 字符数

    // 如果输入长度不超过给定结束位置到起始位置的距离，直接复制
    if (length <= end - str) {
        memcpy(dataptr, str, length * 4);
    }
    else {
        size_t given_len = end - str;
        memcpy(dataptr, str, given_len * 4);  // 复制已给定长度的数据
        memset(dataptr + given_len * 4, '\0', (length - given_len) * 4);  // 填充剩余空间为 '\0'
    }

    // 如果数据字节顺序不是本地字节顺序，则进行字节交换
    if (!PyArray_ISNBO(descr->byteorder)) {
        for (int i = 0; i < length; i++) {
            npy_bswap4_unaligned(dataptr);   // 逐个字符进行字节交换
            dataptr += 4;
        }
    }

    // 返回转换结果
    return 0;
}
    }
    // 如果需要进行字节转换操作
    if (byte_converters) {
        // 将字符串对象转换为 latin1 编码的字节串对象
        Py_SETREF(s, PyUnicode_AsEncodedString(s, "latin1", NULL));
        // 检查转换后的结果是否为 NULL
        if (s == NULL) {
            // 如果转换失败，返回 NULL
            return NULL;
        }
    }
    // 如果 func 为 NULL，则直接返回字符串对象 s
    if (func == NULL) {
        return s;
    }
    // 否则，调用指定的 Python 函数 func，并传入字符串对象 s 作为参数
    PyObject *result = PyObject_CallFunctionObjArgs(func, s, NULL);
    // 释放字符串对象 s 的引用
    Py_DECREF(s);
    // 返回函数调用的结果对象
    return result;
# 使用 NPY_NO_EXPORT 宏来限定函数的可见性，这通常用于隐藏函数，使其在模块外不可见

NPY_NO_EXPORT int
# 定义一个名为 npy_to_generic_with_converter 的函数，接受以下参数：
# - descr: PyArray_Descr 结构指针，描述数组元素的数据类型
# - str: 指向 Py_UCS4 类型的指针，表示输入数据的起始位置
# - end: 指向 Py_UCS4 类型的指针，表示输入数据的结束位置
# - dataptr: 指向 char 类型的指针，用于指向数据将要写入的位置
# - config: 指向 parser_config 结构的指针，包含解析器的配置信息
# - func: PyObject 指针，表示一个可选的转换函数

npy_to_generic_with_converter(PyArray_Descr *descr,
        const Py_UCS4 *str, const Py_UCS4 *end, char *dataptr,
        parser_config *config, PyObject *func)
{
    bool use_byte_converter;
    # 如果 func 为 NULL，则使用 C 字节转换器
    if (func == NULL) {
        use_byte_converter = config->c_byte_converters;
    }
    # 否则，使用 Python 字节转换器
    else {
        use_byte_converter = config->python_byte_converters;
    }

    /* Converts to unicode and calls custom converter (if set) */
    # 调用转换函数，将输入数据转换为 Unicode，并调用自定义转换器（如果设置了）
    PyObject *converted = call_converter_function(
            func, str, (size_t)(end - str), use_byte_converter);
    
    # 如果转换失败，返回 -1
    if (converted == NULL) {
        return -1;
    }

    # 使用 PyArray_Pack 函数将转换后的数据打包到指定的数组描述符中
    int res = PyArray_Pack(descr, dataptr, converted);
    
    # 释放转换后的 Python 对象的引用计数
    Py_DECREF(converted);
    
    # 返回 PyArray_Pack 的执行结果
    return res;
}


NPY_NO_EXPORT int
# 使用 NPY_NO_EXPORT 宏来限定函数的可见性，这通常用于隐藏函数，使其在模块外不可见

npy_to_generic(PyArray_Descr *descr,
        const Py_UCS4 *str, const Py_UCS4 *end, char *dataptr,
        parser_config *config)
{
    # 调用 npy_to_generic_with_converter 函数，传递 NULL 作为转换函数参数
    return npy_to_generic_with_converter(descr, str, end, dataptr, config, NULL);
}
```