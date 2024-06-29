# `.\numpy\numpy\_core\src\multiarray\textreading\readtext.c`

```py
// 引入必要的头文件和库

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <stdio.h>
#include <stdbool.h>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#include "numpy/arrayobject.h"
#include "npy_argparse.h"
#include "common.h"
#include "conversion_utils.h"

// 引入文本解析所需的自定义头文件
#include "textreading/parser_config.h"
#include "textreading/stream_pyobject.h"
#include "textreading/field_types.h"
#include "textreading/rows.h"
#include "textreading/str_to_int.h"

// `_readtext_from_stream` 函数定义，用于从流中读取文本数据
static PyObject *
_readtext_from_stream(stream *s,
        parser_config *pc, Py_ssize_t num_usecols, Py_ssize_t usecols[],
        Py_ssize_t skiplines, Py_ssize_t max_rows,
        PyObject *converters, PyObject *dtype)
{
    PyArrayObject *arr = NULL;
    PyArray_Descr *out_dtype = NULL;
    field_type *ft = NULL;

    /*
     * 如果 `dtype` 是结构化的，那么将其转换为 `PyArray_Descr` 类型，
     * 并增加其引用计数。
     */
    out_dtype = (PyArray_Descr *)dtype;
    Py_INCREF(out_dtype);

    // 创建字段类型描述，并返回字段数量
    Py_ssize_t num_fields = field_types_create(out_dtype, &ft);
    if (num_fields < 0) {
        goto finish;  // 如果创建失败，跳转到结束标签
    }
    // 检查数据是否同构，即是否只有一个字段且类型相同
    bool homogeneous = num_fields == 1 && ft[0].descr == out_dtype;

    // 如果数据不同构且 `usecols` 不为空，且其数量不等于字段数量，则引发错误
    if (!homogeneous && usecols != NULL && num_usecols != num_fields) {
        PyErr_Format(PyExc_TypeError,
                "If a structured dtype is used, the number of columns in "
                "`usecols` must match the effective number of fields. "
                "But %zd usecols were given and the number of fields is %zd.",
                num_usecols, num_fields);
        goto finish;  // 跳转到结束标签
    }

    // 调用 `read_rows` 函数从流中读取数据行，并返回一个 `PyArrayObject` 对象
    arr = read_rows(
            s, max_rows, num_fields, ft, pc,
            num_usecols, usecols, skiplines, converters,
            NULL, out_dtype, homogeneous);
    if (arr == NULL) {
        goto finish;  // 如果读取失败，跳转到结束标签
    }

  finish:
    // 释放 `out_dtype` 的引用
    Py_XDECREF(out_dtype);
    // 清理字段类型描述数组
    field_types_xclear(num_fields, ft);
    // 返回 `arr` 对象
    return (PyObject *)arr;
}

// `parse_control_character` 函数定义，用于解析控制字符
static int
parse_control_character(PyObject *obj, Py_UCS4 *character)
{
    // 如果传入的对象是 `Py_None`，则设置字符为超出 Unicode 范围的值
    if (obj == Py_None) {
        *character = (Py_UCS4)-1;  /* character beyond unicode range */
        return 1;  // 返回解析成功标志
    }
    # 如果对象不是单个 Unicode 字符串或长度不为1，则报错
    if (!PyUnicode_Check(obj) || PyUnicode_GetLength(obj) != 1) {
        # 抛出类型错误异常，说明期望的控制字符必须是单个 Unicode 字符或者 None
        PyErr_Format(PyExc_TypeError,
                "Text reading control character must be a single unicode "
                "character or None; but got: %.100R", obj);
        # 返回 0 表示失败
        return 0;
    }
    # 将 Unicode 对象中的第一个字符赋值给指定的字符变量
    *character = PyUnicode_READ_CHAR(obj, 0);
    # 返回 1 表示成功
    return 1;
/*
 * A (somewhat verbose) check that none of the control characters match or are
 * newline.  Most of these combinations are completely fine, just weird or
 * surprising.
 * (I.e. there is an implicit priority for control characters, so if a comment
 * matches a delimiter, it would just be a comment.)
 * In theory some `delimiter=None` paths could have a "meaning", but let us
 * assume that users are better off setting one of the control chars to `None`
 * for clarity.
 *
 * This also checks that the control characters cannot be newlines.
 */
static int
error_if_matching_control_characters(
        Py_UCS4 delimiter, Py_UCS4 quote, Py_UCS4 comment)
{
    char *control_char1;  // 声明一个指向控制字符名称的指针
    char *control_char2 = NULL;  // 初始化第二个控制字符名称的指针为 NULL
    if (comment != (Py_UCS4)-1) {  // 如果注释字符不是特殊值 -1
        control_char1 = "comment";  // 设置第一个控制字符名称为 "comment"
        if (comment == '\r' || comment == '\n') {  // 如果注释字符是回车或换行
            goto error;  // 跳转到错误处理部分
        }
        else if (comment == quote) {  // 如果注释字符等于引号字符
            control_char2 = "quotechar";  // 设置第二个控制字符名称为 "quotechar"
            goto error;  // 跳转到错误处理部分
        }
        else if (comment == delimiter) {  // 如果注释字符等于分隔符字符
            control_char2 = "delimiter";  // 设置第二个控制字符名称为 "delimiter"
            goto error;  // 跳转到错误处理部分
        }
    }
    if (quote != (Py_UCS4)-1) {  // 如果引号字符不是特殊值 -1
        control_char1 = "quotechar";  // 设置第一个控制字符名称为 "quotechar"
        if (quote == '\r' || quote == '\n') {  // 如果引号字符是回车或换行
            goto error;  // 跳转到错误处理部分
        }
        else if (quote == delimiter) {  // 如果引号字符等于分隔符字符
            control_char2 = "delimiter";  // 设置第二个控制字符名称为 "delimiter"
            goto error;  // 跳转到错误处理部分
        }
    }
    if (delimiter != (Py_UCS4)-1) {  // 如果分隔符字符不是特殊值 -1
        control_char1 = "delimiter";  // 设置第一个控制字符名称为 "delimiter"
        if (delimiter == '\r' || delimiter == '\n') {  // 如果分隔符字符是回车或换行
            goto error;  // 跳转到错误处理部分
        }
    }
    /* The above doesn't work with delimiter=None, which means "whitespace" */
    if (delimiter == (Py_UCS4)-1) {  // 如果分隔符字符是特殊值 -1，表示“空白符”
        control_char1 = "delimiter";  // 设置第一个控制字符名称为 "delimiter"
        if (Py_UNICODE_ISSPACE(comment)) {  // 如果注释字符是空白符
            control_char2 = "comment";  // 设置第二个控制字符名称为 "comment"
            goto error;  // 跳转到错误处理部分
        }
        else if (Py_UNICODE_ISSPACE(quote)) {  // 如果引号字符是空白符
            control_char2 = "quotechar";  // 设置第二个控制字符名称为 "quotechar"
            goto error;  // 跳转到错误处理部分
        }
    }
    return 0;  // 没有发现控制字符匹配问题，返回 0 表示无错误

  error:
    if (control_char2 != NULL) {
        PyErr_Format(PyExc_TypeError,
                "The values for control characters '%s' and '%s' are "
                "incompatible",
                control_char1, control_char2);  // 报告两个控制字符值不兼容的错误
    }
    else {
        PyErr_Format(PyExc_TypeError,
                "control character '%s' cannot be a newline (`\\r` or `\\n`).",
                control_char1);  // 报告控制字符不能是换行符的错误
    }
    return -1;  // 返回 -1 表示发生了错误
}


/*
 * This function loads data from a file-like object, processing various
 * parameters and options related to data loading.
 */
NPY_NO_EXPORT PyObject *
_load_from_filelike(PyObject *NPY_UNUSED(mod),
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    PyObject *file;  // 声明一个文件对象的指针
    Py_ssize_t skiplines = 0;  // 初始化跳过行数为 0
    Py_ssize_t max_rows = -1;  // 初始化最大行数为 -1
    PyObject *usecols_obj = Py_None;  // 初始化列使用对象为 None
    PyObject *converters = Py_None;  // 初始化转换器对象为 None

    PyObject *dtype = Py_None;  // 初始化数据类型为 None
    PyObject *encoding_obj = Py_None;  // 初始化编码对象为 None
    const char *encoding = NULL;  // 初始化编码字符串为 NULL
}
    // 定义并初始化一个解析器配置结构体 pc，包括分隔符、引号、注释符、是否忽略前导空白、
    // 分隔符是否为空白、虚数单位、Python 字节转换器、C 字节转换器、是否提供整数通过浮点数警告
    parser_config pc = {
        .delimiter = ',',
        .quote = '"',
        .comment = '#',
        .ignore_leading_whitespace = false,
        .delimiter_is_whitespace = false,
        .imaginary_unit = 'j',
        .python_byte_converters = false,
        .c_byte_converters = false,
        .gave_int_via_float_warning = false,
    };
    // 定义并初始化一个布尔值变量 filelike，表示是否类似文件
    bool filelike = true;

    // 初始化 PyObject 指针变量 arr，赋值为 NULL
    PyObject *arr = NULL;

    // 准备 NumPy 参数解析器宏的调用
    NPY_PREPARE_ARGPARSER;

    // 解析传入参数，如果解析失败则返回空指针
    if (npy_parse_arguments("_load_from_filelike", args, len_args, kwnames,
            "file", NULL, &file,
            "|delimiter", &parse_control_character, &pc.delimiter,
            "|comment", &parse_control_character, &pc.comment,
            "|quote", &parse_control_character, &pc.quote,
            "|imaginary_unit", &parse_control_character, &pc.imaginary_unit,
            "|usecols", NULL, &usecols_obj,
            "|skiplines", &PyArray_IntpFromPyIntConverter, &skiplines,
            "|max_rows", &PyArray_IntpFromPyIntConverter, &max_rows,
            "|converters", NULL, &converters,
            "|dtype", NULL, &dtype,
            "|encoding", NULL, &encoding_obj,
            "|filelike", &PyArray_BoolConverter, &filelike,
            "|byte_converters", &PyArray_BoolConverter, &pc.python_byte_converters,
            "|c_byte_converters", PyArray_BoolConverter, &pc.c_byte_converters,
            NULL, NULL, NULL) < 0) {
        return NULL;
    }

    // 拒绝匹配的控制字符，因为它们通常没有意义
    if (error_if_matching_control_characters(
            pc.delimiter, pc.quote, pc.comment) < 0) {
        return NULL;
    }

    // 如果分隔符为特定值 (Py_UCS4)-1，则将 delimiter_is_whitespace 设置为 true，
    // 并忽略前导空白以匹配 `string.split(None)`
    if (pc.delimiter == (Py_UCS4)-1) {
        pc.delimiter_is_whitespace = true;
        pc.ignore_leading_whitespace = true;
    }

    // 如果 dtype 不是有效的 NumPy 数据类型描述符，则设置类型错误并返回空指针
    if (!PyArray_DescrCheck(dtype)) {
        PyErr_SetString(PyExc_TypeError,
                "internal error: dtype must be provided and be a NumPy dtype");
        return NULL;
    }

    // 如果 encoding_obj 不是 None，则检查它是否为 Unicode 字符串，
    // 如果不是则设置类型错误并返回空指针；否则将其转换为 UTF-8 编码
    if (encoding_obj != Py_None) {
        if (!PyUnicode_Check(encoding_obj)) {
            PyErr_SetString(PyExc_TypeError,
                    "encoding must be a unicode string.");
            return NULL;
        }
        encoding = PyUnicode_AsUTF8(encoding_obj);
        if (encoding == NULL) {
            return NULL;
        }
    }

    /*
     * 解析 usecols，因为 NumPy 中没有明确的辅助函数，所以在这里手动处理。
     */
    // 初始化 num_usecols 为 -1，usecols 为 NULL
    Py_ssize_t num_usecols = -1;
    Py_ssize_t *usecols = NULL;
    if (usecols_obj != Py_None) {
        // 获取 usecols_obj 序列的长度
        num_usecols = PySequence_Length(usecols_obj);
        if (num_usecols < 0) {
            return NULL;
        }
        /* Calloc just to not worry about overflow */
        // 分配足够大小的内存，以存储 usecols 序列
        usecols = PyMem_Calloc(num_usecols, sizeof(Py_ssize_t));
        if (usecols == NULL) {
            PyErr_NoMemory();
            return NULL;
        }
        // 遍历 usecols_obj 序列，将每个元素转换为 Py_ssize_t 存入 usecols
        for (Py_ssize_t i = 0; i < num_usecols; i++) {
            PyObject *tmp = PySequence_GetItem(usecols_obj, i);
            if (tmp == NULL) {
                PyMem_FREE(usecols);
                return NULL;
            }
            // 将 tmp 转换为 Py_ssize_t 类型存入 usecols[i]
            usecols[i] = PyNumber_AsSsize_t(tmp, PyExc_OverflowError);
            if (error_converting(usecols[i])) {
                if (PyErr_ExceptionMatches(PyExc_TypeError)) {
                    // 若转换出错，生成相应的类型错误异常
                    PyErr_Format(PyExc_TypeError,
                            "usecols must be an int or a sequence of ints but "
                            "it contains at least one element of type '%s'",
                            Py_TYPE(tmp)->tp_name);
                }
                Py_DECREF(tmp);
                PyMem_FREE(usecols);
                return NULL;
            }
            Py_DECREF(tmp);
        }
    }

    // 根据 filelike 参数选择相应的流处理函数
    stream *s;
    if (filelike) {
        s = stream_python_file(file, encoding);
    }
    else {
        s = stream_python_iterable(file, encoding);
    }
    // 若流处理函数返回 NULL，则释放 usecols 内存并返回 NULL
    if (s == NULL) {
        PyMem_FREE(usecols);
        return NULL;
    }

    // 调用 _readtext_from_stream 函数处理流数据，返回处理结果
    arr = _readtext_from_stream(
            s, &pc, num_usecols, usecols, skiplines, max_rows, converters, dtype);
    // 关闭流处理函数
    stream_close(s);
    // 释放 usecols 内存
    PyMem_FREE(usecols);
    // 返回处理结果
    return arr;
}


注释：


# 结束一个代码块，这里对应着某个函数、循环、条件语句或类定义的结束
```