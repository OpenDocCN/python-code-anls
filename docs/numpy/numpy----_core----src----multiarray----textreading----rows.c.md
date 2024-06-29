# `.\numpy\numpy\_core\src\multiarray\textreading\rows.c`

```
#include <Python.h> 
// 包含 Python 头文件

#include "numpy/arrayobject.h"
#include "numpy/npy_3kcompat.h"
#include "alloc.h"
// 包含 numpy 相关头文件

#include <string.h>
#include <stdbool.h>
// 包含标准库头文件

#include "textreading/stream.h"
#include "textreading/tokenize.h"
#include "textreading/conversions.h"
#include "textreading/field_types.h"
#include "textreading/rows.h"
#include "textreading/growth.h"
// 包含自定义头文件

/*
 * Minimum size to grow the allocation by (or 25%). The 8KiB means the actual
 * growths is within `8 KiB <= size < 16 KiB` (depending on the row size).
 */
#define MIN_BLOCK_SIZE (1 << 13)
// 定义最小块大小为 8KB，表示在 `8 KiB <= size < 16 KiB` 之间变动（取决于行大小）

/*
 *  Create the array of converter functions from the Python converters.
 */
static PyObject **
create_conv_funcs(
        PyObject *converters, Py_ssize_t num_fields, const Py_ssize_t *usecols)
{
    assert(converters != Py_None);
    // 断言 converters 不为 Py_None

    PyObject **conv_funcs = PyMem_Calloc(num_fields, sizeof(PyObject *));
    // 使用 PyMem_Calloc 为函数创建 PyObject 指针数组
    if (conv_funcs == NULL) {
        PyErr_NoMemory();
        return NULL;
    }

    if (PyCallable_Check(converters)) {
        /* a single converter used for all columns individually */
        // 单个转换器用于每一列
        for (Py_ssize_t i = 0; i < num_fields; i++) {
            Py_INCREF(converters);
            conv_funcs[i] = converters;
        }
        return conv_funcs;
    }
    else if (!PyDict_Check(converters)) {
        PyErr_SetString(PyExc_TypeError,
                "converters must be a dictionary mapping columns to converter "
                "functions or a single callable.");
        goto error;
        // 报错：converters 必须是一个将列映射到转换器函数的字典或单个可调用对象
    }

    PyObject *key, *value;
    Py_ssize_t pos = 0;
    // 定义变量 key, value, pos
    while (PyDict_Next(converters, &pos, &key, &value)) {
        // 从 converters 字典中依次获取每对键值对
        Py_ssize_t column = PyNumber_AsSsize_t(key, PyExc_IndexError);
        // 将 key 转换为 Py_ssize_t 类型，表示列索引

        if (column == -1 && PyErr_Occurred()) {
            // 如果转换失败或者 column 为 -1 且有错误发生，则报错并跳转到 error 标签
            PyErr_Format(PyExc_TypeError,
                    "keys of the converters dictionary must be integers; "
                    "got %.100R", key);
            goto error;
        }

        if (usecols != NULL) {
            /*
             * 这段代码用于查找对应的 usecols。它与传统的 usecols 代码相同，但有两个弱点：
             * 1. 对于重复的 usecols 只设置第一个的转换器。
             * 2. 如果 usecols 使用负索引而 converters 没有使用，会出错。
             *    （这是一个特性，因为它允许我们在这里正确规范化转换器到结果列。）
             */
            Py_ssize_t i = 0;
            for (; i < num_fields; i++) {
                // 遍历 num_fields（字段数量），查找匹配的列索引
                if (column == usecols[i]) {
                    column = i;
                    break;
                }
            }
            if (i == num_fields) {
                continue;  /* 忽略未使用的转换器 */
            }
        }
        else {
            // 如果 usecols 为 NULL，则处理列索引的边界情况
            if (column < -num_fields || column >= num_fields) {
                PyErr_Format(PyExc_ValueError,
                        "converter specified for column %zd, which is invalid "
                        "for the number of fields %zd.", column, num_fields);
                goto error;
            }
            if (column < 0) {
                column += num_fields;
            }
        }

        if (!PyCallable_Check(value)) {
            // 检查 value 是否可调用，如果不是则报错
            PyErr_Format(PyExc_TypeError,
                    "values of the converters dictionary must be callable, "
                    "but the value associated with key %R is not", key);
            goto error;
        }

        Py_INCREF(value);
        // 增加 value 的引用计数，确保其在后续使用过程中不会被释放
        conv_funcs[column] = value;
        // 将 value 存入 conv_funcs 数组对应的列索引位置
    }

    return conv_funcs;

error:
    // 处理错误时的清理工作
    for (Py_ssize_t i = 0; i < num_fields; i++) {
        Py_XDECREF(conv_funcs[i]);
        // 逐个释放 conv_funcs 数组中的引用
    }
    PyMem_FREE(conv_funcs);
    // 释放 conv_funcs 数组的内存空间
    return NULL;
}

/**
 * Read a file into the provided array, or create (and possibly grow) an
 * array to read into.
 *
 * @param s The stream object/struct providing reading capabilities used by
 *        the tokenizer.
 * @param max_rows The number of rows to read, or -1.  If negative
 *        all rows are read.
 * @param num_field_types The number of field types stored in `field_types`.
 * @param field_types Information about the dtype for each column (or one if
 *        `homogeneous`).
 * @param pconfig Pointer to the parser config object used by both the
 *        tokenizer and the conversion functions.
 * @param num_usecols The number of columns in `usecols`.
 * @param usecols An array of length `num_usecols` or NULL.  If given indicates
 *        which column is read for each individual row (negative columns are
 *        accepted).
 * @param skiplines The number of lines to skip, these lines are ignored.
 * @param converters Python dictionary of converters.  Finalizing converters
 *        is difficult without information about the number of columns.
 * @param data_array An array to be filled or NULL.  In either case a new
 *        reference is returned (the reference to `data_array` is not stolen).
 * @param out_descr The dtype used for allocating a new array.  This is not
 *        used if `data_array` is provided.  Note that the actual dtype of the
 *        returned array can differ for strings.
 * @param num_cols Pointer in which the actual (discovered) number of columns
 *        is returned.  This is only relevant if `homogeneous` is true.
 * @param homogeneous Whether the datatype of the array is not homogeneous,
 *        i.e. not structured.  In this case the number of columns has to be
 *        discovered an the returned array will be 2-dimensional rather than
 *        1-dimensional.
 *
 * @returns Returns the result as an array object or NULL on error.  The result
 *          is always a new reference (even when `data_array` was passed in).
 */
NPY_NO_EXPORT PyArrayObject *
read_rows(stream *s,
        npy_intp max_rows, Py_ssize_t num_field_types, field_type *field_types,
        parser_config *pconfig, Py_ssize_t num_usecols, Py_ssize_t *usecols,
        Py_ssize_t skiplines, PyObject *converters,
        PyArrayObject *data_array, PyArray_Descr *out_descr,
        bool homogeneous)
{
    // 指向数据的指针，初始化为 NULL
    char *data_ptr = NULL;
    // 当前字段的数量
    Py_ssize_t current_num_fields;
    // 行的大小，由输出描述符的元素大小决定
    npy_intp row_size = out_descr->elsize;
    // 是否需要初始化数据
    bool needs_init = PyDataType_FLAGCHK(out_descr, NPY_NEEDS_INIT);

    // 数组的维度，如果数据类型不同构，则为 2，否则为 1
    int ndim = homogeneous ? 2 : 1;
    // 结果数组的形状，初始为 {0, 1}
    npy_intp result_shape[2] = {0, 1};

    // 数据数组是否已分配的标志
    bool data_array_allocated = data_array == NULL;
    /* 确保我们对错误处理目的拥有 `data_array` 的所有权 */
    // 增加 `data_array` 的引用计数
    Py_XINCREF(data_array);
    // 每块中的行数，根据行大小而增加
    size_t rows_per_block = 1;
    // 已分配的数据行数，初始为 0
    npy_intp data_allocated_rows = 0;

    /* 如果 max_rows 被使用并且遇到空行，则发出警告 */
    # 检查是否应该给出空行警告，如果最大行数大于等于零则为真
    bool give_empty_row_warning = max_rows >= 0;

    # 初始化结果变量为零，并创建分词器状态对象
    int ts_result = 0;
    tokenizer_state ts;
    if (npy_tokenizer_init(&ts, pconfig) < 0) {
        goto error;
    }

    /* 如果已知字段数量，设置实际字段数；否则设置为 -1 */
    Py_ssize_t actual_num_fields = -1;
    if (usecols != NULL) {
        assert(homogeneous || num_field_types == num_usecols);
        actual_num_fields = num_usecols;
    }
    else if (!homogeneous) {
        assert(usecols == NULL || num_field_types == num_usecols);
        actual_num_fields = num_field_types;
    }

    # 跳过指定行数的循环，处理每行数据直到结束或发生错误
    for (Py_ssize_t i = 0; i < skiplines; i++) {
        ts.state = TOKENIZE_GOTO_LINE_END;
        ts_result = npy_tokenize(s, &ts, pconfig);
        if (ts_result < 0) {
            goto error;
        }
        else if (ts_result != 0) {
            /* 少于指定行数是可以接受的 */
            break;
        }
    }

    # 实际处理的行数
    Py_ssize_t row_count = 0;  /* 实际处理的行数 */

    # 清理分词器状态
    npy_tokenizer_clear(&ts);

    # 如果存在类型转换函数，释放它们的引用
    if (conv_funcs != NULL) {
        for (Py_ssize_t i = 0; i < actual_num_fields; i++) {
            Py_XDECREF(conv_funcs[i]);
        }
        PyMem_FREE(conv_funcs);
    }

    # 如果数据数组为空，根据情况重新分配
    if (data_array == NULL) {
        assert(row_count == 0 && result_shape[0] == 0);
        if (actual_num_fields == -1) {
            /*
             * 如果找不到行数，则必须猜测有一个元素
             * 注意：可以考虑将此移到外部以优化必要的行为
             */
            result_shape[1] = 1;
        }
        else {
            result_shape[1] = actual_num_fields;
        }
        Py_INCREF(out_descr);
        data_array = (PyArrayObject *)PyArray_Empty(
                ndim, result_shape, out_descr, 0);
    }

    /*
     * 注意，如果没有数据，`data_array` 可能仍为 NULL，row_count 为 0。
     * 在这种情况下，始终重新分配以防万一。
     */
    if (data_array_allocated && data_allocated_rows != row_count) {
        size_t size = row_count * row_size;
        char *new_data = PyDataMem_UserRENEW(
                PyArray_BYTES(data_array), size ? size : 1,
                PyArray_HANDLER(data_array));
        if (new_data == NULL) {
            Py_DECREF(data_array);
            PyErr_NoMemory();
            return NULL;
        }
        ((PyArrayObject_fields *)data_array)->data = new_data;
        ((PyArrayObject_fields *)data_array)->dimensions[0] = row_count;
    }

    return data_array;

  error:
    # 处理错误时释放资源
    if (conv_funcs != NULL) {
        for (Py_ssize_t i = 0; i < actual_num_fields; i++) {
            Py_XDECREF(conv_funcs[i]);
        }
        PyMem_FREE(conv_funcs);
    }
    npy_tokenizer_clear(&ts);
    Py_XDECREF(data_array);
    return NULL;
}



# 这行代码表示一个函数或类定义的结尾。在这里，} 表示代码块的结束，可能是一个函数、类或其他代码块的结尾。
```