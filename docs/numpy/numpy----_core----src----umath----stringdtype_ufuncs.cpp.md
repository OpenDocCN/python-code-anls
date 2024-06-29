# `.\numpy\numpy\_core\src\umath\stringdtype_ufuncs.cpp`

```
/* Ufunc implementations for the StringDType class */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#define _UMATHMODULE

#include "numpy/arrayobject.h"
#include "numpy/ndarraytypes.h"
#include "numpy/npy_math.h"
#include "numpy/ufuncobject.h"

#include "numpyos.h"
#include "gil_utils.h"
#include "dtypemeta.h"
#include "abstractdtypes.h"
#include "dispatching.h"
#include "string_ufuncs.h"
#include "stringdtype_ufuncs.h"
#include "string_buffer.h"
#include "string_fastsearch.h"
#include "templ_common.h" /* for npy_mul_size_with_overflow_size_t */

#include "stringdtype/static_string.h"
#include "stringdtype/dtype.h"
#include "stringdtype/utf8_utils.h"

/* Define macro LOAD_TWO_INPUT_STRINGS(CONTEXT) for loading and checking two input strings */
#define LOAD_TWO_INPUT_STRINGS(CONTEXT)                                            \
        const npy_packed_static_string *ps1 = (npy_packed_static_string *)in1;     \
        npy_static_string s1 = {0, NULL};                                          \
        int s1_isnull = NpyString_load(s1allocator, ps1, &s1);                     \
        const npy_packed_static_string *ps2 = (npy_packed_static_string *)in2;     \
        npy_static_string s2 = {0, NULL};                                          \
        int s2_isnull = NpyString_load(s2allocator, ps2, &s2);                     \
        if (s1_isnull == -1 || s2_isnull == -1) {                                  \
            npy_gil_error(PyExc_MemoryError, "Failed to load string in %s",        \
                          CONTEXT);                                                \
            goto fail;                                                             \
        }                                                                          \

/* Implementation of a function to resolve descriptors for multiplication */
static NPY_CASTING
multiply_resolve_descriptors(
        struct PyArrayMethodObject_tag *NPY_UNUSED(method),
        PyArray_DTypeMeta *const dtypes[], PyArray_Descr *const given_descrs[],
        PyArray_Descr *loop_descrs[], npy_intp *NPY_UNUSED(view_offset))
{
    PyArray_Descr *ldescr = given_descrs[0];  // Left input descriptor
    PyArray_Descr *rdescr = given_descrs[1];  // Right input descriptor
    PyArray_StringDTypeObject *odescr = NULL; // Output string dtype descriptor
    PyArray_Descr *out_descr = NULL;          // Output descriptor

    // Determine which dtype (left or right) corresponds to PyArray_StringDType
    if (dtypes[0] == &PyArray_StringDType) {
        odescr = (PyArray_StringDTypeObject *)ldescr;
    }
    else {
        odescr = (PyArray_StringDTypeObject *)rdescr;
    }

    // If the third descriptor is not provided, create a new instance of string dtype
    if (given_descrs[2] == NULL) {
        out_descr = (PyArray_Descr *)new_stringdtype_instance(
                odescr->na_object, odescr->coerce);
        if (out_descr == NULL) {
            return (NPY_CASTING)-1;  // Return error if creation fails
        }
    }
    else {
        Py_INCREF(given_descrs[2]);
        out_descr = given_descrs[2];
    }

    // Increment references to input descriptors and assign to loop descriptors
    Py_INCREF(ldescr);
    loop_descrs[0] = ldescr;
    Py_INCREF(rdescr);
    loop_descrs[1] = rdescr;
    loop_descrs[2] = out_descr;  // Assign output descriptor to loop descriptors array

    return NPY_NO_CASTING;  // Return no-casting flag
}
# 定义一个静态函数，用于在循环中执行字符串数组的乘法操作
static int multiply_loop_core(
        npy_intp N, char *sin, char *iin, char *out,
        npy_intp s_stride, npy_intp i_stride, npy_intp o_stride,
        PyArray_StringDTypeObject *idescr, PyArray_StringDTypeObject *odescr)
{
    # 创建一个描述符数组，包含输入和输出的字符串数据类型描述符
    PyArray_Descr *descrs[2] =
            {(PyArray_Descr *)idescr, (PyArray_Descr *)odescr};
    # 申请字符串分配器的内存，获取用于字符串操作的分配器
    npy_string_allocator *allocators[2] = {};
    NpyString_acquire_allocators(2, descrs, allocators);
    # 分别获取输入和输出字符串的分配器
    npy_string_allocator *iallocator = allocators[0];
    npy_string_allocator *oallocator = allocators[1];
    # 检查输入描述符是否有空对象
    int has_null = idescr->na_object != NULL;
    # 检查输入描述符是否具有 NaN 或 NA 值
    int has_nan_na = idescr->has_nan_na;
    # 检查输入描述符是否具有字符串类型的 NA 值
    int has_string_na = idescr->has_string_na;
    # 获取默认字符串指针
    const npy_static_string *default_string = &idescr->default_string;
    // 循环 N 次，处理输入和输出的字符串数据
    while (N--) {
        // 从输入源读取一个静态压缩字符串结构到 ips
        const npy_packed_static_string *ips =
                (npy_packed_static_string *)sin;
        // 初始化输出的静态字符串结构 is
        npy_static_string is = {0, NULL};
        // 从输出源读取一个静态压缩字符串结构到 ops
        npy_packed_static_string *ops = (npy_packed_static_string *)out;
        // 载入 ips 所指向的数据到 is，如果失败返回 is_isnull
        int is_isnull = NpyString_load(iallocator, ips, &is);
        // 如果载入过程中出现内存错误，报错并跳转到 fail 标签
        if (is_isnull == -1) {
            npy_gil_error(PyExc_MemoryError,
                          "Failed to load string in multiply");
            goto fail;
        }
        // 如果载入结果为空值
        else if (is_isnull) {
            // 如果允许 NaN 或 NA 值存在
            if (has_nan_na) {
                // 将空值打包到 ops 中，如果失败则报错并跳转到 fail 标签
                if (NpyString_pack_null(oallocator, ops) < 0) {
                    npy_gil_error(PyExc_MemoryError,
                                  "Failed to deallocate string in multiply");
                    goto fail;
                }
                // 调整输入和输出指针以及步长，并继续下一次循环
                sin += s_stride;
                iin += i_stride;
                out += o_stride;
                continue;
            }
            // 如果存在字符串的 NA 值或者不允许空值存在
            else if (has_string_na || !has_null) {
                // 使用默认字符串作为 is 的值
                is = *(npy_static_string *)default_string;
            }
            // 否则报类型错误并跳转到 fail 标签
            else {
                npy_gil_error(PyExc_TypeError,
                              "Cannot multiply null that is not a nan-like "
                              "value");
                goto fail;
            }
        }
        // 从输入中读取因子值到 factor
        T factor = *(T *)iin;
        // 计算新的字符串大小，检查是否溢出
        size_t cursize = is.size;
        size_t newsize;
        int overflowed = npy_mul_with_overflow_size_t(
                &newsize, cursize, factor);
        // 如果溢出，报内存错误并跳转到 fail 标签
        if (overflowed) {
            npy_gil_error(PyExc_MemoryError,
                          "Failed to allocate string in string multiply");
            goto fail;
        }

        char *buf = NULL;
        npy_static_string os = {0, NULL};
        // 如果描述符相同，执行原地操作
        if (descrs[0] == descrs[1]) {
            // 分配新的缓冲区以存储扩展后的字符串
            buf = (char *)PyMem_RawMalloc(newsize);
            // 如果分配失败，报内存错误并跳转到 fail 标签
            if (buf == NULL) {
                npy_gil_error(PyExc_MemoryError,
                              "Failed to allocate string in multiply");
                goto fail;
            }
        }
        // 如果描述符不同，加载新的字符串到 os 中
        else {
            // 加载新字符串到 ops 中，如果失败则跳转到 fail 标签
            if (load_new_string(
                        ops, &os, newsize,
                        oallocator, "multiply") == -1) {
                goto fail;
            }
            // 将新缓冲区初始化为 os.buf
            /* explicitly discard const; initializing new buffer */
            buf = (char *)os.buf;
        }

        // 将 is.buf 的内容复制 factor 次到 buf 中
        for (size_t i = 0; i < (size_t)factor; i++) {
            /* multiply can't overflow because cursize * factor */
            /* has already been checked and doesn't overflow */
            memcpy((char *)buf + i * cursize, is.buf, cursize);
        }

        // 如果描述符相同，进行原地打包操作
        if (descrs[0] == descrs[1]) {
            // 将 buf 中的数据打包到 ops 中，如果失败则报内存错误并跳转到 fail 标签
            if (NpyString_pack(oallocator, ops, buf, newsize) < 0) {
                npy_gil_error(PyExc_MemoryError,
                              "Failed to pack string in multiply");
                goto fail;
            }
            // 释放临时缓冲区 buf
            PyMem_RawFree(buf);
        }

        // 调整输入和输出指针以及步长，准备下一轮循环
        sin += s_stride;
        iin += i_stride;
        out += o_stride;
    }
    // 调用 NpyString_release_allocators 函数，释放内存分配器资源，参数为 2 和 allocators 数组
    NpyString_release_allocators(2, allocators);
    // 返回整数值 0，表示函数执行成功
    return 0;
fail:
    # 释放两个分配器，这里使用了 NpyString_release_allocators 函数
    NpyString_release_allocators(2, allocators);
    # 返回 -1，表示函数执行失败
    return -1;
}

template <typename T>
static int multiply_right_strided_loop(
        PyArrayMethod_Context *context, char *const data[],
        npy_intp const dimensions[], npy_intp const strides[],
        NpyAuxData *NPY_UNUSED(auxdata))
{
    // 获取输入和输出的数据类型描述符
    PyArray_StringDTypeObject *idescr =
            (PyArray_StringDTypeObject *)context->descriptors[0];
    PyArray_StringDTypeObject *odescr =
            (PyArray_StringDTypeObject *)context->descriptors[2];
    // 获取数组的长度
    npy_intp N = dimensions[0];
    // 获取输入数组的指针
    char *sin = data[0];
    char *iin = data[1];
    char *out = data[2];
    // 获取输入数组的步长
    npy_intp in1_stride = strides[0];
    npy_intp in2_stride = strides[1];
    npy_intp out_stride = strides[2];

    // 调用模板函数 multiply_loop_core 进行核心的乘法运算
    return multiply_loop_core<T>(
            N, sin, iin, out, in1_stride, in2_stride, out_stride,
            idescr, odescr);
}

template <typename T>
static int multiply_left_strided_loop(
        PyArrayMethod_Context *context, char *const data[],
        npy_intp const dimensions[], npy_intp const strides[],
        NpyAuxData *NPY_UNUSED(auxdata))
{
    // 获取输入和输出的数据类型描述符
    PyArray_StringDTypeObject *idescr =
            (PyArray_StringDTypeObject *)context->descriptors[1];
    PyArray_StringDTypeObject *odescr =
            (PyArray_StringDTypeObject *)context->descriptors[2];
    // 获取数组的长度
    npy_intp N = dimensions[0];
    // 获取输入数组的指针
    char *iin = data[0];
    char *sin = data[1];
    char *out = data[2];
    // 获取输入数组的步长
    npy_intp in1_stride = strides[0];
    npy_intp in2_stride = strides[1];
    npy_intp out_stride = strides[2];

    // 调用模板函数 multiply_loop_core 进行核心的乘法运算
    return multiply_loop_core<T>(
            N, sin, iin, out, in2_stride, in1_stride, out_stride,
            idescr, odescr);
}

static NPY_CASTING
binary_resolve_descriptors(struct PyArrayMethodObject_tag *NPY_UNUSED(method),
                           PyArray_DTypeMeta *const NPY_UNUSED(dtypes[]),
                           PyArray_Descr *const given_descrs[],
                           PyArray_Descr *loop_descrs[],
                           npy_intp *NPY_UNUSED(view_offset))
{
    // 获取给定的输入数据类型描述符
    PyArray_StringDTypeObject *descr1 = (PyArray_StringDTypeObject *)given_descrs[0];
    PyArray_StringDTypeObject *descr2 = (PyArray_StringDTypeObject *)given_descrs[1];
    // 判断是否需要强制转换输出
    int out_coerce = descr1->coerce && descr1->coerce;
    PyObject *out_na_object = NULL;

    // 检查字符串类型描述符的兼容性，如果不兼容则返回 -1
    if (stringdtype_compatible_na(
                descr1->na_object, descr2->na_object, &out_na_object) == -1) {
        return (NPY_CASTING)-1;
    }

    // 增加输入描述符的引用计数，并设置循环描述符的值
    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];
    // 增加输入描述符的引用计数，并设置循环描述符的值
    Py_INCREF(given_descrs[1]);
    loop_descrs[1] = given_descrs[1];

    PyArray_Descr *out_descr = NULL;

    // 如果没有给定输出描述符，则创建一个新的字符串类型实例
    if (given_descrs[2] == NULL) {
        out_descr = (PyArray_Descr *)new_stringdtype_instance(
                out_na_object, out_coerce);

        // 如果创建失败，则返回 -1
        if (out_descr == NULL) {
            return (NPY_CASTING)-1;
        }
    }
    else {
        // 增加给定输出描述符的引用计数，并设置循环描述符的值
        Py_INCREF(given_descrs[2]);
        loop_descrs[2] = given_descrs[2];
    }

    // 设置输出描述符的值
    loop_descrs[2] = out_descr;

    // 返回不需要任何强制转换
    return NPY_NO_CASTING;
}
# 定义名为 add_strided_loop 的静态函数，接受 PyArrayMethod_Context 结构体指针和多个数组参数
static int
add_strided_loop(PyArrayMethod_Context *context, char *const data[],
                 npy_intp const dimensions[], npy_intp const strides[],
                 NpyAuxData *NPY_UNUSED(auxdata))
{
    # 获取第一个输入数组的字符串数据类型描述符
    PyArray_StringDTypeObject *s1descr = (PyArray_StringDTypeObject *)context->descriptors[0];
    # 获取第二个输入数组的字符串数据类型描述符
    PyArray_StringDTypeObject *s2descr = (PyArray_StringDTypeObject *)context->descriptors[1];
    # 获取输出数组的字符串数据类型描述符
    PyArray_StringDTypeObject *odescr = (PyArray_StringDTypeObject *)context->descriptors[2];
    # 检查 s1descr 中是否包含空值对象
    int has_null = s1descr->na_object != NULL;
    # 检查 s1descr 是否具有 NaN 和 NA 值
    int has_nan_na = s1descr->has_nan_na;
    # 检查 s1descr 是否包含字符串 NA 值
    int has_string_na = s1descr->has_string_na;
    # 获取 s1descr 的默认字符串
    const npy_static_string *default_string = &s1descr->default_string;
    # 获取输入数组的第一维度大小
    npy_intp N = dimensions[0];
    # 获取第一个输入数组的起始地址
    char *in1 = data[0];
    # 获取第二个输入数组的起始地址
    char *in2 = data[1];
    # 获取输出数组的起始地址
    char *out = data[2];
    # 获取第一个输入数组的步长
    npy_intp in1_stride = strides[0];
    # 获取第二个输入数组的步长
    npy_intp in2_stride = strides[1];
    # 获取输出数组的步长
    npy_intp out_stride = strides[2];

    # 创建一个长度为 3 的指针数组 allocators，用于存储字符串分配器
    npy_string_allocator *allocators[3] = {};
    # 调用 NpyString_acquire_allocators 函数获取字符串分配器
    NpyString_acquire_allocators(3, context->descriptors, allocators);
    # 分别获取第一个、第二个输入数组以及输出数组的字符串分配器
    npy_string_allocator *s1allocator = allocators[0];
    npy_string_allocator *s2allocator = allocators[1];
    npy_string_allocator *oallocator = allocators[2];
    // 循环执行 N 次，每次执行一组操作
    while (N--) {
        // 宏定义：加载两个输入字符串，操作为"add"
        LOAD_TWO_INPUT_STRINGS("add")
        
        // 初始化变量
        char *buf = NULL;
        npy_static_string os = {0, NULL};
        size_t newsize = 0;
        npy_packed_static_string *ops = (npy_packed_static_string *)out;
        
        // 检查是否存在空值
        if (NPY_UNLIKELY(s1_isnull || s2_isnull)) {
            // 如果存在 NaN 或 NA，将空字符串打包到输出
            if (has_nan_na) {
                if (NpyString_pack_null(oallocator, ops) < 0) {
                    // 内存错误处理：打包字符串失败
                    npy_gil_error(PyExc_MemoryError,
                                  "Failed to deallocate string in add");
                    goto fail;
                }
                // 跳转到下一步骤
                goto next_step;
            }
            // 如果存在字符串 NA 或没有空值
            else if (has_string_na || !has_null) {
                // 如果 s1 为空，则使用默认字符串
                if (s1_isnull) {
                    s1 = *default_string;
                }
                // 如果 s2 为空，则使用默认字符串
                if (s2_isnull) {
                    s2 = *default_string;
                }
            }
            // 否则，出现不支持的空值情况，抛出值错误
            else {
                npy_gil_error(PyExc_ValueError,
                              "Cannot add null that is not a nan-like value");
                goto fail;
            }
        }

        // 检查是否会溢出
        newsize = s1.size + s2.size;
        if (newsize < s1.size) {
            // 内存错误处理：分配字符串失败
            npy_gil_error(PyExc_MemoryError, "Failed to allocate string in add");
            goto fail;
        }

        // 如果是原地操作
        if (s1descr == odescr || s2descr == odescr) {
            // 分配内存
            buf = (char *)PyMem_RawMalloc(newsize);

            if (buf == NULL) {
                // 内存错误处理：分配字符串失败
                npy_gil_error(PyExc_MemoryError,
                          "Failed to allocate string in add");
                goto fail;
            }
        }
        // 否则，加载新字符串并初始化新缓冲区
        else {
            if (load_new_string(ops, &os, newsize, oallocator, "add") == -1) {
                goto fail;
            }
            // 显式丢弃 const；初始化新缓冲区
            buf = (char *)os.buf;
        }

        // 复制 s1 和 s2 的内容到 buf
        memcpy(buf, s1.buf, s1.size);
        memcpy(buf + s1.size, s2.buf, s2.size);

        // 清理临时的原地缓冲区
        if (s1descr == odescr || s2descr == odescr) {
            // 将 buf 打包到输出字符串
            if (NpyString_pack(oallocator, ops, buf, newsize) < 0) {
                // 内存错误处理：打包输出字符串失败
                npy_gil_error(PyExc_MemoryError,
                          "Failed to pack output string in add");
                goto fail;
            }

            // 释放内存
            PyMem_RawFree(buf);
        }

    next_step:
        // 更新输入和输出指针位置
        in1 += in1_stride;
        in2 += in2_stride;
        out += out_stride;
    }
    // 释放所有分配器
    NpyString_release_allocators(3, allocators);
    // 返回成功状态
    return 0;
// 根据输入的上下文和数据执行字符串比较操作的循环
static int
string_comparison_strided_loop(PyArrayMethod_Context *context, char *const data[],
                            npy_intp const dimensions[],
                            npy_intp const strides[],
                            NpyAuxData *NPY_UNUSED(auxdata))
{
    // 获取调用该函数的通用函数对象的名称
    const char *ufunc_name = ((PyUFuncObject *)context->caller)->name;
    // 从静态数据中提取等于操作的结果标志
    npy_bool res_for_eq = ((npy_bool *)context->method->static_data)[0];
    // 从静态数据中提取小于操作的结果标志
    npy_bool res_for_lt = ((npy_bool *)context->method->static_data)[1];

    // 获取第一个输入参数的字符串数据类型描述符
    PyArray_StringDTypeObject *in1_descr =
            ((PyArray_StringDTypeObject *)context->descriptors[0]);
    // 获取第二个输入参数的字符串数据类型描述符
    PyArray_StringDTypeObject *in2_descr =
            ((PyArray_StringDTypeObject *)context->descriptors[1]);

    // 获取当前处理的维度大小
    npy_intp N = dimensions[0];
    // 获取第一个输入参数的起始地址
    char *in1 = data[0];
    // 获取第二个输入参数的起始地址
    char *in2 = data[1];
    // 获取输出参数的起始地址
    char *out = data[2];
    // 获取第一个输入参数的步长
    npy_intp in1_stride = strides[0];
    // 获取第二个输入参数的步长
    npy_intp in2_stride = strides[1];
    // 获取输出参数的步长
    npy_intp out_stride = strides[2];

    // 创建用于存储字符串分配器的数组
    npy_string_allocator *allocators[3] = {};
    // 获取字符串分配器，用于输入和输出参数的内存分配
    NpyString_acquire_allocators(3, context->descriptors, allocators);
    // 获取第一个输入参数的字符串分配器
    npy_string_allocator *in1_allocator = allocators[0];
    // 获取第二个输入参数的字符串分配器
    npy_string_allocator *in2_allocator = allocators[1];
    // 获取输出参数的字符串分配器
    npy_string_allocator *out_allocator = allocators[2];

    // 循环处理每个元素
    while (N--) {
        // 将第一个输入参数转换为静态字符串
        const npy_packed_static_string *sin1 = (npy_packed_static_string *)in1;
        // 将第二个输入参数转换为静态字符串
        const npy_packed_static_string *sin2 = (npy_packed_static_string *)in2;
        // 将输出参数转换为静态字符串
        npy_packed_static_string *sout = (npy_packed_static_string *)out;
        
        // 执行字符串的比较操作
        int cmp = _compare(in1, in2, in1_descr, in2_descr);
        
        // 如果字符串相等且输出等于输入之一，则跳到下一步
        if (cmp == 0 && (in1 == out || in2 == out)) {
            goto next_step;
        }
        
        // 根据比较结果和操作标志执行复制操作或避免释放后使用
        if ((cmp < 0) ^ res_for_lt) {
            // 如果输入1不等于输出，则执行复制操作
            if (in1 != out) {
                if (free_and_copy(in1_allocator, out_allocator, sin1, sout,
                                  ufunc_name) == -1) {
                    goto fail;
                }
            }
        }
        else {
            // 如果输入2不等于输出，则执行复制操作
            if (in2 != out) {
                if (free_and_copy(in2_allocator, out_allocator, sin2, sout,
                                  ufunc_name) == -1) {
                    goto fail;
                }
            }
        }

      next_step:
        // 更新输入和输出的指针位置
        in1 += in1_stride;
        in2 += in2_stride;
        out += out_stride;
    }

    // 释放字符串分配器
    NpyString_release_allocators(3, allocators);
    // 返回成功完成操作的标志
    return 0;

fail:
    // 在操作失败时释放字符串分配器
    NpyString_release_allocators(3, allocators);
    // 返回操作失败的标志
    return -1;
}
    // 从上下文中获取第三个静态数据的布尔值作为大于结果的标志
    npy_bool res_for_gt = ((npy_bool *)context->method->static_data)[2];
    // 计算不等于结果的布尔值
    npy_bool res_for_ne = !res_for_eq;
    // 检查是否等于或不等于的结果是相同的
    npy_bool eq_or_ne = res_for_lt == res_for_gt;
    // 从上下文中获取第一个描述符并转换为字符串数据类型对象
    PyArray_StringDTypeObject *descr1 = (PyArray_StringDTypeObject *)context->descriptors[0];
    // 检查描述符中是否有 NULL 值
    int has_null = descr1->na_object != NULL;
    // 检查描述符中是否有 NaN 或 NA 值
    int has_nan_na = descr1->has_nan_na;
    // 检查描述符中是否有字符串 NA 值
    int has_string_na = descr1->has_string_na;
    // 获取默认字符串并指向描述符的默认字符串
    const npy_static_string *default_string = &descr1->default_string;
    // 获取第一个维度的大小
    npy_intp N = dimensions[0];
    // 获取输入数据数组的第一个指针
    char *in1 = data[0];
    // 获取输入数据数组的第二个指针
    char *in2 = data[1];
    // 获取输出数据数组的指针，并转换为布尔类型指针
    npy_bool *out = (npy_bool *)data[2];
    // 获取输入数据数组的第一个步长
    npy_intp in1_stride = strides[0];
    // 获取输入数据数组的第二个步长
    npy_intp in2_stride = strides[1];
    // 获取输出数据数组的步长
    npy_intp out_stride = strides[2];

    // 分配两个字符串分配器的数组并初始化为 NULL
    npy_string_allocator *allocators[2] = {};
    // 调用函数以获取字符串分配器
    NpyString_acquire_allocators(2, context->descriptors, allocators);
    // 获取第一个字符串分配器
    npy_string_allocator *s1allocator = allocators[0];
    // 获取第二个字符串分配器
    npy_string_allocator *s2allocator = allocators[1];

    // 循环处理每个元素
    while (N--) {
        int cmp;
        // 载入两个输入字符串，宏定义的函数
        LOAD_TWO_INPUT_STRINGS(ufunc_name);
        // 检查是否有 NULL 或 NaN NA 值
        if (NPY_UNLIKELY(s1_isnull || s2_isnull)) {
            // 如果有 NaN NA 值
            if (has_nan_na) {
                // s1 或 s2 是 NA
                *out = NPY_FALSE;
                // 跳转到下一步骤
                goto next_step;
            }
            // 如果有 NULL 值但没有字符串 NA 值
            else if (has_null && !has_string_na) {
                // 如果是等于或不等于操作
                if (eq_or_ne) {
                    // 如果两个都是 NULL
                    if (s1_isnull && s2_isnull) {
                        *out = res_for_eq;
                    }
                    else {
                        *out = res_for_ne;
                    }
                }
                else {
                    // 抛出异常，不支持在非 NaN 类型或字符串中的空值
                    npy_gil_error(PyExc_ValueError,
                                  "'%s' not supported for null values that are not "
                                  "nan-like or strings.", ufunc_name);
                    // 跳转到失败处理
                    goto fail;
                }
            }
            // 否则，处理默认字符串
            else {
                if (s1_isnull) {
                    s1 = *default_string;
                }
                if (s2_isnull) {
                    s2 = *default_string;
                }
            }
        }
        // 比较两个字符串
        cmp = NpyString_cmp(&s1, &s2);
        // 根据比较结果设置输出值
        if (cmp == 0) {
            *out = res_for_eq;
        }
        else if (cmp < 0) {
            *out = res_for_lt;
        }
        else {
            *out = res_for_gt;
        }

    next_step:
        // 更新输入和输出指针
        in1 += in1_stride;
        in2 += in2_stride;
        out += out_stride;
    }

    // 释放字符串分配器
    NpyString_release_allocators(2, allocators);

    // 返回成功状态
    return 0;
fail:
    # 释放两个分配器的资源
    NpyString_release_allocators(2, allocators);

    # 返回错误状态码
    return -1;
}

static NPY_CASTING
string_comparison_resolve_descriptors(
        struct PyArrayMethodObject_tag *NPY_UNUSED(method),
        PyArray_DTypeMeta *const NPY_UNUSED(dtypes[]),
        PyArray_Descr *const given_descrs[],
        PyArray_Descr *loop_descrs[], npy_intp *NPY_UNUSED(view_offset))
{
    # 将给定的描述符转换为字符串数据类型对象
    PyArray_StringDTypeObject *descr1 = (PyArray_StringDTypeObject *)given_descrs[0];
    PyArray_StringDTypeObject *descr2 = (PyArray_StringDTypeObject *)given_descrs[1];

    # 检查是否两个字符串数据类型对象的特性兼容，如果不兼容则返回错误状态
    if (stringdtype_compatible_na(descr1->na_object, descr2->na_object, NULL) == -1) {
        return (NPY_CASTING)-1;
    }

    # 增加给定描述符的引用计数，并将其赋值给循环描述符数组
    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];
    Py_INCREF(given_descrs[1]);
    loop_descrs[1] = given_descrs[1];
    # 将循环描述符数组的第三个位置设置为布尔类型的描述符，操作不可能失败
    loop_descrs[2] = PyArray_DescrFromType(NPY_BOOL);

    # 返回无需转换的状态
    return NPY_NO_CASTING;
}

static int
string_isnan_strided_loop(PyArrayMethod_Context *context, char *const data[],
                          npy_intp const dimensions[],
                          npy_intp const strides[],
                          NpyAuxData *NPY_UNUSED(auxdata))
{
    # 获取字符串数据类型对象的描述符并检查是否包含 NaN 或 NA
    PyArray_StringDTypeObject *descr = (PyArray_StringDTypeObject *)context->descriptors[0];
    int has_nan_na = descr->has_nan_na;

    # 获取第一维度的大小
    npy_intp N = dimensions[0];
    # 输入数据的起始地址
    char *in = data[0];
    # 输出数据的起始地址
    npy_bool *out = (npy_bool *)data[1];
    # 输入数据和输出数据的步幅
    npy_intp in_stride = strides[0];
    npy_intp out_stride = strides[1];

    # 遍历每个元素并检查是否为 NaN 或 NA
    while (N--) {
        const npy_packed_static_string *s = (npy_packed_static_string *)in;
        if (has_nan_na && NpyString_isnull(s)) {
            *out = NPY_TRUE;
        }
        else {
            *out = NPY_FALSE;
        }

        in += in_stride;
        out += out_stride;
    }

    # 返回成功状态
    return 0;
}

static NPY_CASTING
string_bool_output_resolve_descriptors(
        struct PyArrayMethodObject_tag *NPY_UNUSED(method),
        PyArray_DTypeMeta *const NPY_UNUSED(dtypes[]),
        PyArray_Descr *const given_descrs[],
        PyArray_Descr *loop_descrs[], npy_intp *NPY_UNUSED(view_offset))
{
    # 增加给定描述符的引用计数，并将其赋值给循环描述符数组
    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];
    # 将循环描述符数组的第二个位置设置为布尔类型的描述符，操作不可能失败
    loop_descrs[1] = PyArray_DescrFromType(NPY_BOOL);

    # 返回无需转换的状态
    return NPY_NO_CASTING;
}

static NPY_CASTING
string_intp_output_resolve_descriptors(
        struct PyArrayMethodObject_tag *NPY_UNUSED(method),
        PyArray_DTypeMeta *const NPY_UNUSED(dtypes[]),
        PyArray_Descr *const given_descrs[],
        PyArray_Descr *loop_descrs[], npy_intp *NPY_UNUSED(view_offset))
{
    # 增加给定描述符的引用计数，并将其赋值给循环描述符数组
    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];
    # 将循环描述符数组的第二个位置设置为整数类型的描述符，操作不可能失败
    loop_descrs[1] = PyArray_DescrFromType(NPY_INTP);

    # 返回无需转换的状态
    return NPY_NO_CASTING;
}

using utf8_buffer_method = bool (Buffer<ENCODING::UTF8>::*)();

static int
// 获取调用者的 ufunc 名称
const char *ufunc_name = ((PyUFuncObject *)context->caller)->name;
// 获取 static_data 中的 utf8_buffer_method 结构体
utf8_buffer_method is_it = *(utf8_buffer_method *)(context->method->static_data);
// 获取第一个描述符的字符串数据类型对象
PyArray_StringDTypeObject *descr = (PyArray_StringDTypeObject *)context->descriptors[0];
// 获取字符串分配器
npy_string_allocator *allocator = NpyString_acquire_allocator(descr);
// 检查描述符是否包含字符串 NA
int has_string_na = descr->has_string_na;
// 检查描述符是否包含 NaN NA
int has_nan_na = descr->has_nan_na;
// 获取默认字符串
const npy_static_string *default_string = &descr->default_string;
// 获取第一个维度的大小
npy_intp N = dimensions[0];
// 获取输入数据指针
char *in = data[0];
// 获取输出数据指针
char *out = data[1];
// 获取输入步长
npy_intp in_stride = strides[0];
// 获取输出步长
npy_intp out_stride = strides[1];

// 循环处理每个元素
while (N--) {
    // 将输入数据解析为 npy_packed_static_string 结构体
    const npy_packed_static_string *ps = (npy_packed_static_string *)in;

    // 初始化一个空的静态字符串 s
    npy_static_string s = {0, NULL};
    // 初始化 buffer 和 size 为 NULL
    const char *buffer = NULL;
    size_t size = 0;
    // 初始化一个 Buffer<ENCODING::UTF8> 对象 buf
    Buffer<ENCODING::UTF8> buf;

    // 载入字符串到 s 中，返回是否为 null
    int is_null = NpyString_load(allocator, ps, &s);

    // 如果载入失败，抛出内存错误并跳转到 fail 标签处
    if (is_null == -1) {
        npy_gil_error(PyExc_MemoryError, "Failed to load string in %s", ufunc_name);
        goto fail;
    }

    // 如果字符串为 null
    if (is_null) {
        // 如果描述符允许 NaN NA，则将输出设为 NPY_FALSE 并跳转到 next_step
        if (has_nan_na) {
            *out = NPY_FALSE;
            goto next_step;
        }
        // 如果不允许字符串 NA，则抛出值错误并跳转到 fail 标签处
        else if (!has_string_na) {
            npy_gil_error(PyExc_ValueError,
                          "Cannot use the %s function with a null that is "
                          "not a nan-like value", ufunc_name);
            goto fail;
        }
        // 否则使用默认字符串的数据
        buffer = default_string->buf;
        size = default_string->size;
    }
    else {
        // 否则使用载入的字符串数据
        buffer = s.buf;
        size = s.size;
    }

    // 初始化 buf 为包含 buffer 和 size 的 UTF8 编码的 Buffer 对象
    buf = Buffer<ENCODING::UTF8>((char *)buffer, size);
    // 将输出数据转换为 npy_bool 并存储 buf 执行 is_it 操作的结果
    *(npy_bool *)out = (buf.*is_it)();

next_step:
    // 更新输入和输出指针
    in += in_stride;
    out += out_stride;
}

// 释放字符串分配器
NpyString_release_allocator(allocator);

return 0;

fail:
// 失败时释放字符串分配器并返回 -1
NpyString_release_allocator(allocator);
return -1;
}



// 使用字符串长度函数的循环处理函数
static int
string_strlen_strided_loop(PyArrayMethod_Context *context, char *const data[],
                           npy_intp const dimensions[],
                           npy_intp const strides[],
                           NpyAuxData *auxdata)
{
    // 获取第一个描述符的字符串数据类型对象
    PyArray_StringDTypeObject *descr = (PyArray_StringDTypeObject *)context->descriptors[0];
    // 获取字符串分配器
    npy_string_allocator *allocator = NpyString_acquire_allocator(descr);
    // 检查描述符是否包含字符串 NA
    int has_string_na = descr->has_string_na;
    // 获取默认字符串
    const npy_static_string *default_string = &descr->default_string;

    // 获取第一个维度的大小
    npy_intp N = dimensions[0];
    // 获取输入数据指针
    char *in = data[0];
    // 获取输出数据指针
    char *out = data[1];
    // 获取输入步长
    npy_intp in_stride = strides[0];
    // 获取输出步长
    npy_intp out_stride = strides[1];
    // 循环，执行N次
    while (N--) {
        // 将输入的指针解释为np_packed_static_string类型，并赋给ps
        const npy_packed_static_string *ps = (npy_packed_static_string *)in;

        // 定义并初始化npy_static_string结构体变量s
        npy_static_string s = {0, NULL};
        
        // 初始化buffer为空指针，size为0
        const char *buffer = NULL;
        size_t size = 0;
        
        // 创建一个Buffer对象，模板参数为ENCODING::UTF8
        Buffer<ENCODING::UTF8> buf;
        
        // 调用NpyString_load函数加载字符串数据，返回值表示字符串是否为空
        int is_null = NpyString_load(allocator, ps, &s);

        // 如果加载失败，抛出内存错误，并跳转到fail标签处
        if (is_null == -1) {
            npy_gil_error(PyExc_MemoryError, "Failed to load string in str_len");
            goto fail;
        }

        // 如果字符串为空
        if (is_null) {
            // 如果未设置has_string_na标志，抛出数值错误并跳转到next_step标签处
            if (!has_string_na) {
                npy_gil_error(PyExc_ValueError,
                              "The length of a null string is undefined");
                goto next_step;
            }
            // 否则使用默认字符串的缓冲区和大小
            buffer = default_string->buf;
            size = default_string->size;
        }
        else {
            // 否则使用加载的字符串的缓冲区和大小
            buffer = s.buf;
            size = s.size;
        }
        
        // 将buffer和size传递给Buffer对象buf进行初始化
        buf = Buffer<ENCODING::UTF8>((char *)buffer, size);
        
        // 将buf的字符数赋给输出指针指向的npy_intp类型变量（假设out是npy_intp*类型）
        *(npy_intp *)out = buf.num_codepoints();

      next_step:
        // 更新输入指针，移动到下一个字符串位置
        in += in_stride;
        // 更新输出指针，移动到下一个输出位置
        out += out_stride;
    }

    // 释放分配器allocator所分配的内存资源
    NpyString_release_allocator(allocator);

    // 返回0表示成功执行
    return 0;
fail:
    # 调用 NpyString_release_allocator 函数释放分配器
    NpyString_release_allocator(allocator);
    # 返回 -1 表示失败
    return -1;
}

static int
string_findlike_promoter(PyObject *NPY_UNUSED(ufunc),
        PyArray_DTypeMeta *const op_dtypes[],
        PyArray_DTypeMeta *const signature[],
        PyArray_DTypeMeta *new_op_dtypes[])
{
    # 设置新操作的数据类型为字符串类型
    new_op_dtypes[0] = NPY_DT_NewRef(&PyArray_StringDType);
    new_op_dtypes[1] = NPY_DT_NewRef(&PyArray_StringDType);
    # 设置新操作的数据类型为 int64 类型
    new_op_dtypes[2] = NPY_DT_NewRef(&PyArray_Int64DType);
    new_op_dtypes[3] = NPY_DT_NewRef(&PyArray_Int64DType);
    # 设置新操作的数据类型为默认整数类型
    new_op_dtypes[4] = PyArray_DTypeFromTypeNum(NPY_DEFAULT_INT);
    # 返回 0 表示成功
    return 0;
}

static NPY_CASTING
string_findlike_resolve_descriptors(
        struct PyArrayMethodObject_tag *NPY_UNUSED(method),
        PyArray_DTypeMeta *const NPY_UNUSED(dtypes[]),
        PyArray_Descr *const given_descrs[],
        PyArray_Descr *loop_descrs[],
        npy_intp *NPY_UNUSED(view_offset))
{
    # 强制转换给定描述符为字符串类型对象
    PyArray_StringDTypeObject *descr1 = (PyArray_StringDTypeObject *)given_descrs[0];
    PyArray_StringDTypeObject *descr2 = (PyArray_StringDTypeObject *)given_descrs[1];

    # 检查给定的字符串类型是否兼容
    if (stringdtype_compatible_na(descr1->na_object, descr2->na_object, NULL) == -1) {
        return (NPY_CASTING)-1;
    }

    # 增加对给定描述符的引用计数，并将其赋给循环描述符
    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];
    Py_INCREF(given_descrs[1]);
    loop_descrs[1] = given_descrs[1];
    Py_INCREF(given_descrs[2]);
    loop_descrs[2] = given_descrs[2];
    Py_INCREF(given_descrs[3]);
    loop_descrs[3] = given_descrs[3];
    # 如果第四个描述符为空，则使用默认整数类型
    if (given_descrs[4] == NULL) {
        loop_descrs[4] = PyArray_DescrFromType(NPY_DEFAULT_INT);
    }
    else {
        Py_INCREF(given_descrs[4]);
        loop_descrs[4] = given_descrs[4];
    }

    # 返回不需要类型转换的值
    return NPY_NO_CASTING;
}

static int
string_startswith_endswith_promoter(
        PyObject *NPY_UNUSED(ufunc),
        PyArray_DTypeMeta *const op_dtypes[],
        PyArray_DTypeMeta *const signature[],
        PyArray_DTypeMeta *new_op_dtypes[])
{
    # 设置新操作的数据类型为字符串类型
    new_op_dtypes[0] = NPY_DT_NewRef(&PyArray_StringDType);
    new_op_dtypes[1] = NPY_DT_NewRef(&PyArray_StringDType);
    # 设置新操作的数据类型为 int64 类型
    new_op_dtypes[2] = NPY_DT_NewRef(&PyArray_Int64DType);
    new_op_dtypes[3] = NPY_DT_NewRef(&PyArray_Int64DType);
    # 设置新操作的数据类型为布尔类型
    new_op_dtypes[4] = PyArray_DTypeFromTypeNum(NPY_BOOL);
    # 返回 0 表示成功
    return 0;
}

static NPY_CASTING
string_startswith_endswith_resolve_descriptors(
        struct PyArrayMethodObject_tag *NPY_UNUSED(method),
        PyArray_DTypeMeta *const NPY_UNUSED(dtypes[]),
        PyArray_Descr *const given_descrs[],
        PyArray_Descr *loop_descrs[],
        npy_intp *NPY_UNUSED(view_offset))
{
    # 强制转换给定描述符为字符串类型对象
    PyArray_StringDTypeObject *descr1 = (PyArray_StringDTypeObject *)given_descrs[0];
    PyArray_StringDTypeObject *descr2 = (PyArray_StringDTypeObject *)given_descrs[1];

    # 检查给定的字符串类型是否兼容
    if (stringdtype_compatible_na(descr1->na_object, descr2->na_object, NULL) == -1) {
        return (NPY_CASTING)-1;
    }

    # 增加对给定描述符的引用计数，并将其赋给循环描述符
    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];
    Py_INCREF(given_descrs[1]);
    loop_descrs[1] = given_descrs[1];
    # 使用默认整数类型描述符
    # 增加给定描述符数组索引为 2 的引用计数
    Py_INCREF(given_descrs[2]);
    # 将给定描述符数组索引为 2 的元素赋值给循环描述符数组的索引为 2 的位置
    loop_descrs[2] = given_descrs[2];
    # 增加给定描述符数组索引为 3 的引用计数
    Py_INCREF(given_descrs[3]);
    # 将给定描述符数组索引为 3 的元素赋值给循环描述符数组的索引为 3 的位置
    loop_descrs[3] = given_descrs[3];
    # 如果给定描述符数组索引为 4 的元素为空指针
    if (given_descrs[4] == NULL) {
        # 将循环描述符数组的索引为 4 的位置设置为布尔类型的数组描述符
        loop_descrs[4] = PyArray_DescrFromType(NPY_BOOL);
    }
    else {
        # 增加给定描述符数组索引为 4 的引用计数
        Py_INCREF(given_descrs[4]);
        # 将给定描述符数组索引为 4 的元素赋值给循环描述符数组的索引为 4 的位置
        loop_descrs[4] = given_descrs[4];
    }
    
    # 返回值 NPY_NO_CASTING，表示不进行任何类型转换
    return NPY_NO_CASTING;
static int
string_findlike_strided_loop(PyArrayMethod_Context *context,
                         char *const data[],
                         npy_intp const dimensions[],
                         npy_intp const strides[],
                         NpyAuxData *auxdata)
{
    // 获取调用该函数的 ufunc 的名字
    const char *ufunc_name = ((PyUFuncObject *)context->caller)->name;
    
    // 从静态数据中获取 find_like_function 函数指针
    find_like_function *function = *(find_like_function *)(context->method->static_data);
    
    // 获取第一个输入的字符串描述符
    PyArray_StringDTypeObject *descr1 = (PyArray_StringDTypeObject *)context->descriptors[0];

    // 检查第一个输入是否包含空值
    int has_null = descr1->na_object != NULL;
    // 检查第一个输入是否包含字符串 NA
    int has_string_na = descr1->has_string_na;
    // 获取默认字符串
    const npy_static_string *default_string = &descr1->default_string;

    // 分配字符串的内存分配器
    npy_string_allocator *allocators[2] = {};
    NpyString_acquire_allocators(2, context->descriptors, allocators);
    // 获取第一个和第二个输入的字符串内存分配器
    npy_string_allocator *s1allocator = allocators[0];
    npy_string_allocator *s2allocator = allocators[1];

    // 获取输入数据的指针
    char *in1 = data[0];  // 输入字符串 1 的指针
    char *in2 = data[1];  // 输入字符串 2 的指针
    char *in3 = data[2];  // 输入起始位置的指针
    char *in4 = data[3];  // 输入结束位置的指针
    char *out = data[4];  // 输出结果的指针

    // 获取数据维度的大小
    npy_intp N = dimensions[0];

    // 迭代处理每一个数据点
    while (N--) {
        // 加载两个输入字符串
        LOAD_TWO_INPUT_STRINGS(ufunc_name);
        
        // 如果输入字符串中有空值
        if (NPY_UNLIKELY(s1_isnull || s2_isnull)) {
            // 如果支持空值且不支持字符串 NA
            if (has_null && !has_string_na) {
                // 报告错误并跳转到失败标签
                npy_gil_error(PyExc_ValueError,
                              "'%s' not supported for null values that are not "
                              "strings.", ufunc_name);
                goto fail;
            }
            else {
                // 如果输入字符串 1 是空值，使用默认字符串
                if (s1_isnull) {
                    s1 = *default_string;
                }
                // 如果输入字符串 2 是空值，使用默认字符串
                if (s2_isnull) {
                    s2 = *default_string;
                }
            }
        }

        // 获取起始和结束位置
        npy_int64 start = *(npy_int64 *)in3;
        npy_int64 end = *(npy_int64 *)in4;

        // 将输入字符串转换为 Buffer 对象
        Buffer<ENCODING::UTF8> buf1((char *)s1.buf, s1.size);
        Buffer<ENCODING::UTF8> buf2((char *)s2.buf, s2.size);

        // 调用 find_like_function 函数计算结果
        npy_intp pos = function(buf1, buf2, start, end);
        // 如果返回特定错误标志，跳转到失败标签
        if (pos == -2) {
            goto fail;
        }
        
        // 将结果写入输出数组
        *(npy_intp *)out = pos;

        // 更新输入和输出指针位置
        in1 += strides[0];
        in2 += strides[1];
        in3 += strides[2];
        in4 += strides[3];
        out += strides[4];
    }

    // 释放字符串的内存分配器
    NpyString_release_allocators(2, allocators);

    // 返回成功
    return 0;

fail:
    // 在失败时释放字符串的内存分配器并返回失败
    NpyString_release_allocators(2, allocators);
    return -1;
}
    // 获取第一个描述符，并将其转换为字符串数据类型对象
    PyArray_StringDTypeObject *descr1 = (PyArray_StringDTypeObject *)context->descriptors[0];

    // 检查是否存在空值
    int has_null = descr1->na_object != NULL;
    // 检查是否存在字符串类型的空值
    int has_string_na = descr1->has_string_na;
    // 检查是否存在NaN类型的空值
    int has_nan_na = descr1->has_nan_na;
    // 获取默认字符串指针
    const npy_static_string *default_string = &descr1->default_string;

    // 分配两个字符串分配器的空间，并获取描述符中的字符串分配器
    npy_string_allocator *allocators[2] = {};
    NpyString_acquire_allocators(2, context->descriptors, allocators);
    // 分配器1用于第一个描述符
    npy_string_allocator *s1allocator = allocators[0];
    // 分配器2用于第二个描述符
    npy_string_allocator *s2allocator = allocators[1];

    // 获取输入数据指针
    char *in1 = data[0];
    char *in2 = data[1];
    char *in3 = data[2];
    char *in4 = data[3];
    // 获取输出数据指针
    char *out = data[4];

    // 获取第一维度的大小
    npy_intp N = dimensions[0];

    // 进入主循环，逐行处理数据
    while (N--) {
        // 加载两个输入字符串
        LOAD_TWO_INPUT_STRINGS(ufunc_name);
        
        // 如果其中一个字符串为null
        if (NPY_UNLIKELY(s1_isnull || s2_isnull)) {
            // 如果支持null，并且不支持字符串类型的空值
            if (has_null && !has_string_na) {
                // 如果支持NaN类型的空值
                if (has_nan_na) {
                    // 对于此操作，null始终为假
                    *(npy_bool *)out = 0;
                    // 跳转到下一步骤
                    goto next_step;
                }
                // 否则，抛出值错误异常
                else {
                    npy_gil_error(PyExc_ValueError,
                                  "'%s' not supported for null values that "
                                  "are not nan-like or strings.", ufunc_name);
                    // 跳转到失败处理
                    goto fail;
                }
            }
            // 否则，如果第一个字符串为null，使用默认字符串
            else {
                if (s1_isnull) {
                    s1 = *default_string;
                }
                // 如果第二个字符串为null，使用默认字符串
                if (s2_isnull) {
                    s2 = *default_string;
                }
            }
        }
        {
            // 从输入数据中获取起始和结束位置
            npy_int64 start = *(npy_int64 *)in3;
            npy_int64 end = *(npy_int64 *)in4;

            // 使用UTF8编码创建缓冲区1和缓冲区2
            Buffer<ENCODING::UTF8> buf1((char *)s1.buf, s1.size);
            Buffer<ENCODING::UTF8> buf2((char *)s2.buf, s2.size);

            // 执行UTF8编码的尾部匹配操作
            npy_bool match = tailmatch<ENCODING::UTF8>(buf1, buf2, start, end,
                                                       startposition);
            // 将匹配结果存储到输出数据中
            *(npy_bool *)out = match;
        }

      next_step:

        // 更新输入和输出数据指针
        in1 += strides[0];
        in2 += strides[1];
        in3 += strides[2];
        in4 += strides[3];
        out += strides[4];
    }

    // 释放两个字符串分配器
    NpyString_release_allocators(2, allocators);

    // 返回0表示成功
    return 0;
    // 释放之前获取的字符串分配器的资源
    NpyString_release_allocators(2, allocators);

    // 返回错误码 -1
    return -1;
}

static int
all_strings_promoter(PyObject *NPY_UNUSED(ufunc),
                     PyArray_DTypeMeta *const op_dtypes[],
                     PyArray_DTypeMeta *const signature[],
                     PyArray_DTypeMeta *new_op_dtypes[])
{
    // 将所有新操作数据类型设置为对字符串数据类型的新引用
    new_op_dtypes[0] = NPY_DT_NewRef(&PyArray_StringDType);
    new_op_dtypes[1] = NPY_DT_NewRef(&PyArray_StringDType);
    new_op_dtypes[2] = NPY_DT_NewRef(&PyArray_StringDType);
    // 返回成功码 0
    return 0;
}

NPY_NO_EXPORT int
string_lrstrip_chars_strided_loop(
        PyArrayMethod_Context *context, char *const data[],
        npy_intp const dimensions[],
        npy_intp const strides[],
        NpyAuxData *auxdata)
{
    // 获取调用者的名称作为字符串类型
    const char *ufunc_name = ((PyUFuncObject *)context->caller)->name;
    // 获取方法静态数据中的剥离类型
    STRIPTYPE striptype = *(STRIPTYPE *)context->method->static_data;
    // 获取第一个描述符作为字符串数据类型对象
    PyArray_StringDTypeObject *s1descr = (PyArray_StringDTypeObject *)context->descriptors[0];
    // 检查第一个描述符是否有 NULL 对象
    int has_null = s1descr->na_object != NULL;
    // 检查第一个描述符是否包含字符串 NA 值
    int has_string_na = s1descr->has_string_na;
    // 检查第一个描述符是否包含 NaN NA 值
    int has_nan_na = s1descr->has_nan_na;

    // 获取默认字符串的静态指针
    const npy_static_string *default_string = &s1descr->default_string;
    // 获取数据的第一维度大小
    npy_intp N = dimensions[0];
    // 获取输入数据数组的指针
    char *in1 = data[0];
    char *in2 = data[1];
    char *out = data[2];

    // 定义字符串分配器数组，并获取三个描述符的字符串分配器
    npy_string_allocator *allocators[3] = {};
    NpyString_acquire_allocators(3, context->descriptors, allocators);
    npy_string_allocator *s1allocator = allocators[0];
    npy_string_allocator *s2allocator = allocators[1];
    npy_string_allocator *oallocator = allocators[2];
    while (N--) {
        LOAD_TWO_INPUT_STRINGS(ufunc_name);
        // 从输入数组中加载两个字符串
        npy_packed_static_string *ops = (npy_packed_static_string *)out;
        // 将输出指针类型转换为静态字符串结构体指针

        if (NPY_UNLIKELY(s1_isnull || s2_isnull)) {
            // 如果 s1 或 s2 是空值
            if (has_string_na || !has_null) {
                // 如果存在字符串类型的空值或者没有空值
                if (s1_isnull) {
                    s1 = *default_string;
                    // 如果 s1 是空值，则使用默认字符串
                }
                if (s2_isnull) {
                    s2 = *default_string;
                    // 如果 s2 是空值，则使用默认字符串
                }
            }
            else if (has_nan_na) {
                // 如果存在 NaN 类型的空值
                if (s2_isnull) {
                    npy_gil_error(PyExc_ValueError,
                                  "Cannot use a null string that is not a "
                                  "string as the %s delimiter", ufunc_name);
                    // 报错，不能使用非字符串类型的空字符串作为分隔符
                }
                if (s1_isnull) {
                    if (NpyString_pack_null(oallocator, ops) < 0) {
                        npy_gil_error(PyExc_MemoryError,
                                      "Failed to deallocate string in %s",
                                      ufunc_name);
                        // 打印内存错误信息
                        goto fail;
                    }
                    goto next_step;
                    // 跳转到下一步
                }
            }
            else {
                npy_gil_error(PyExc_ValueError,
                              "Can only strip null values that are strings "
                              "or NaN-like values");
                // 报错，只能去除字符串类型或类似 NaN 的空值
                goto fail;
                // 跳转到错误处理
            }
        }
        {
            char *new_buf = (char *)PyMem_RawCalloc(s1.size, 1);
            // 分配新的内存缓冲区
            Buffer<ENCODING::UTF8> buf1((char *)s1.buf, s1.size);
            Buffer<ENCODING::UTF8> buf2((char *)s2.buf, s2.size);
            Buffer<ENCODING::UTF8> outbuf(new_buf, s1.size);
            // 创建 UTF-8 编码的缓冲区对象

            size_t new_buf_size = string_lrstrip_chars
                    (buf1, buf2, outbuf, striptype);
            // 调用字符串去除函数，计算新缓冲区的大小

            if (NpyString_pack(oallocator, ops, new_buf, new_buf_size) < 0) {
                npy_gil_error(PyExc_MemoryError, "Failed to pack string in %s",
                              ufunc_name);
                // 打印内存错误信息
                PyMem_RawFree(new_buf);
                // 释放内存
                goto fail;
                // 跳转到错误处理
            }

            PyMem_RawFree(new_buf);
            // 释放内存
        }
      next_step:
        // 下一步标签

        in1 += strides[0];
        in2 += strides[1];
        out += strides[2];
        // 更新输入输出指针的位置
    }

    NpyString_release_allocators(3, allocators);
    // 释放分配器
    return 0;
fail:
    // 调用NpyString_release_allocators函数释放分配的字符串分配器
    NpyString_release_allocators(3, allocators);
    // 返回-1，表示函数执行失败
    return -1;
}



static NPY_CASTING
strip_whitespace_resolve_descriptors(
        struct PyArrayMethodObject_tag *NPY_UNUSED(method),
        PyArray_DTypeMeta *const NPY_UNUSED(dtypes[]),
        PyArray_Descr *const given_descrs[],
        PyArray_Descr *loop_descrs[],
        npy_intp *NPY_UNUSED(view_offset))
{
    // 增加给定描述符的引用计数
    Py_INCREF(given_descrs[0]);
    // 将给定的描述符赋值给循环描述符数组的第一个元素
    loop_descrs[0] = given_descrs[0];

    // 初始化输出描述符指针为NULL
    PyArray_Descr *out_descr = NULL;

    // 如果第二个给定描述符为NULL
    if (given_descrs[1] == NULL) {
        // 创建一个新的字符串类型实例作为输出描述符
        out_descr = (PyArray_Descr *)new_stringdtype_instance(
                ((PyArray_StringDTypeObject *)given_descrs[0])->na_object,
                ((PyArray_StringDTypeObject *)given_descrs[0])->coerce);

        // 如果无法创建输出描述符实例，返回-1表示失败
        if (out_descr == NULL) {
            return (NPY_CASTING)-1;
        }
    }
    else {
        // 增加第二个给定描述符的引用计数
        Py_INCREF(given_descrs[1]);
        // 将第二个给定描述符赋值给输出描述符
        out_descr = given_descrs[1];
    }

    // 将输出描述符赋值给循环描述符数组的第二个元素
    loop_descrs[1] = out_descr;

    // 返回NPY_NO_CASTING，表示无需类型转换
    return NPY_NO_CASTING;
}



static int
string_lrstrip_whitespace_strided_loop(
        PyArrayMethod_Context *context,
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    // 获取调用者的ufunc名称
    const char *ufunc_name = ((PyUFuncObject *)context->caller)->name;
    // 获取strip类型，这是一个枚举值
    STRIPTYPE striptype = *(STRIPTYPE *)context->method->static_data;
    // 获取第一个描述符作为字符串类型对象
    PyArray_StringDTypeObject *descr = (PyArray_StringDTypeObject *)context->descriptors[0];
    // 检查是否存在na_object
    int has_null = descr->na_object != NULL;
    // 检查是否具有string_na标志
    int has_string_na = descr->has_string_na;
    // 检查是否具有nan_na标志
    int has_nan_na = descr->has_nan_na;
    // 获取默认字符串的指针
    const npy_static_string *default_string = &descr->default_string;

    // 分配两个字符串分配器的数组
    npy_string_allocator *allocators[2] = {};
    // 获取两个描述符的字符串分配器
    NpyString_acquire_allocators(2, context->descriptors, allocators);
    // 第一个分配器用于输入数据，第二个分配器用于输出数据
    npy_string_allocator *allocator = allocators[0];
    npy_string_allocator *oallocator = allocators[1];

    // 输入数据的指针
    char *in = data[0];
    // 输出数据的指针
    char *out = data[1];

    // 数据的维度大小
    npy_intp N = dimensions[0];

    // 这里是函数的主要逻辑，后续的代码应该继续注释，但超出了最大字符数限制，因此结束。
    // 循环，N 逐步减少直到为 0
    while (N--) {
        // 将输入的指针视为 npy_packed_static_string 类型的静态字符串指针
        const npy_packed_static_string *ps = (npy_packed_static_string *)in;
        // 定义并初始化一个静态字符串 s
        npy_static_string s = {0, NULL};
        // 调用 NpyString_load 函数加载字符串 s，并检查是否为空
        int s_isnull = NpyString_load(allocator, ps, &s);

        // 如果加载过程中出现错误，抛出内存错误并跳转到失败处理标签
        if (s_isnull == -1) {
            npy_gil_error(PyExc_MemoryError, "Failed to load string in %s", ufunc_name);
            goto fail;
        }

        // 将输出指针视为 npy_packed_static_string 类型的静态字符串指针
        npy_packed_static_string *ops = (npy_packed_static_string *)out;

        // 如果 s 为空
        if (NPY_UNLIKELY(s_isnull)) {
            // 如果允许字符串为空或者不存在空值，使用默认字符串替换 s
            if (has_string_na || !has_null) {
                s = *default_string;
            }
            // 如果存在 NaN 类型的空值
            else if (has_nan_na) {
                // 使用空值填充 ops，并检查是否出错
                if (NpyString_pack_null(oallocator, ops) < 0) {
                    npy_gil_error(PyExc_MemoryError, "Failed to deallocate string in %s", ufunc_name);
                    goto fail;
                }
                // 跳转到下一步处理标签
                goto next_step;
            }
            // 其他情况，抛出数值错误，说明只能去除字符串类型的空值或 NaN 类型的空值
            else {
                npy_gil_error(PyExc_ValueError, "Can only strip null values that are strings or NaN-like values");
                goto fail;
            }
        }

        {
            // 分配新的缓冲区，用于处理字符串操作
            char *new_buf = (char *)PyMem_RawCalloc(s.size, 1);
            // 创建输入和输出的 UTF-8 编码缓冲区
            Buffer<ENCODING::UTF8> buf((char *)s.buf, s.size);
            Buffer<ENCODING::UTF8> outbuf(new_buf, s.size);
            // 对字符串进行去除空白字符处理，得到新的缓冲区大小
            size_t new_buf_size = string_lrstrip_whitespace(buf, outbuf, striptype);

            // 将处理后的字符串打包到 ops，检查是否出错
            if (NpyString_pack(oallocator, ops, new_buf, new_buf_size) < 0) {
                npy_gil_error(PyExc_MemoryError, "Failed to pack string in %s", ufunc_name);
                goto fail;
            }

            // 释放新分配的缓冲区
            PyMem_RawFree(new_buf);
        }

      next_step:
        // 更新输入和输出指针的位置
        in += strides[0];
        out += strides[1];
    }

    // 释放所有的分配器资源
    NpyString_release_allocators(2, allocators);

    // 返回成功状态码
    return 0;

  fail:
    // 释放所有的分配器资源
    NpyString_release_allocators(2, allocators);

    // 返回失败状态码
    return -1;
static int
string_replace_promoter(PyObject *NPY_UNUSED(ufunc),
                        PyArray_DTypeMeta *const op_dtypes[],
                        PyArray_DTypeMeta *const signature[],
                        PyArray_DTypeMeta *new_op_dtypes[])
{
    // 设置新的操作数据类型为字符串类型的引用
    new_op_dtypes[0] = NPY_DT_NewRef(&PyArray_StringDType);
    new_op_dtypes[1] = NPY_DT_NewRef(&PyArray_StringDType);
    new_op_dtypes[2] = NPY_DT_NewRef(&PyArray_StringDType);
    new_op_dtypes[3] = NPY_DT_NewRef(&PyArray_Int64DType);
    new_op_dtypes[4] = NPY_DT_NewRef(&PyArray_StringDType);
    // 返回操作成功
    return 0;
}

static NPY_CASTING
replace_resolve_descriptors(struct PyArrayMethodObject_tag *NPY_UNUSED(method),
                            PyArray_DTypeMeta *const NPY_UNUSED(dtypes[]),
                            PyArray_Descr *const given_descrs[],
                            PyArray_Descr *loop_descrs[],
                            npy_intp *NPY_UNUSED(view_offset))
{
    // 获取给定描述符的字符串类型对象
    PyArray_StringDTypeObject *descr1 = (PyArray_StringDTypeObject *)given_descrs[0];
    PyArray_StringDTypeObject *descr2 = (PyArray_StringDTypeObject *)given_descrs[1];
    PyArray_StringDTypeObject *descr3 = (PyArray_StringDTypeObject *)given_descrs[2];
    // 判断是否需要强制转换
    int out_coerce = descr1->coerce && descr2->coerce && descr3->coerce;
    PyObject *out_na_object = NULL;

    // 检查字符串类型的可兼容性
    if (stringdtype_compatible_na(
                descr1->na_object, descr2->na_object, &out_na_object) == -1) {
        return (NPY_CASTING)-1;
    }

    // 继续检查另一组字符串类型的兼容性
    if (stringdtype_compatible_na(
                out_na_object, descr3->na_object, &out_na_object) == -1) {
        return (NPY_CASTING)-1;
    }

    // 增加给定描述符的引用计数，并将其复制到循环描述符中
    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];
    Py_INCREF(given_descrs[1]);
    loop_descrs[1] = given_descrs[1];
    Py_INCREF(given_descrs[2]);
    loop_descrs[2] = given_descrs[2];
    Py_INCREF(given_descrs[3]);
    loop_descrs[3] = given_descrs[3];

    PyArray_Descr *out_descr = NULL;

    // 如果第四个描述符为空，则创建新的字符串类型实例
    if (given_descrs[4] == NULL) {
        out_descr = (PyArray_Descr *)new_stringdtype_instance(
                out_na_object, out_coerce);

        // 如果创建失败，则返回错误状态
        if (out_descr == NULL) {
            return (NPY_CASTING)-1;
        }
    }
    else {
        // 否则增加给定描述符的引用计数，并直接使用
        Py_INCREF(given_descrs[4]);
        out_descr = given_descrs[4];
    }

    // 将输出描述符放入循环描述符中
    loop_descrs[4] = out_descr;

    // 返回不需要强制转换的状态
    return NPY_NO_CASTING;
}


static int
string_replace_strided_loop(
        PyArrayMethod_Context *context,
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    // 获取输入和输出数据的指针
    char *in1 = data[0];
    char *in2 = data[1];
    char *in3 = data[2];
    char *in4 = data[3];
    char *out = data[4];

    // 获取数据的维度大小
    npy_intp N = dimensions[0];

    // 获取描述符对象并检查其属性
    PyArray_StringDTypeObject *descr0 =
            (PyArray_StringDTypeObject *)context->descriptors[0];
    int has_null = descr0->na_object != NULL;
    int has_string_na = descr0->has_string_na;
    int has_nan_na = descr0->has_nan_na;
    const npy_static_string *default_string = &descr0->default_string;


这些注释详细说明了每行代码的功能和作用，遵循了提供的示例和要求。
    # 创建一个包含5个元素的指针数组，初始都为空指针
    npy_string_allocator *allocators[5] = {};

    # 调用函数，获取字符串分配器的数组，存储在allocators数组中
    NpyString_acquire_allocators(5, context->descriptors, allocators);

    # 获取特定位置的字符串分配器
    npy_string_allocator *i1allocator = allocators[0];
    npy_string_allocator *i2allocator = allocators[1];
    npy_string_allocator *i3allocator = allocators[2];

    # allocators[3] 是空指针
    npy_string_allocator *oallocator = allocators[4];

    # 释放字符串分配器的数组
    NpyString_release_allocators(5, allocators);

    # 返回成功状态
    return 0;

  fail:
    # 如果出现错误，释放字符串分配器的数组
    NpyString_release_allocators(5, allocators);

    # 返回错误状态
    return -1;
}


static NPY_CASTING expandtabs_resolve_descriptors(
        struct PyArrayMethodObject_tag *NPY_UNUSED(method),
        PyArray_DTypeMeta *const NPY_UNUSED(dtypes[]),
        PyArray_Descr *const given_descrs[],
        PyArray_Descr *loop_descrs[],
        npy_intp *NPY_UNUSED(view_offset))
{
    // 增加给定描述符的引用计数，并将其赋给循环描述符数组中的第一个位置
    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];
    // 增加给定描述符的引用计数，并将其赋给循环描述符数组中的第二个位置
    Py_INCREF(given_descrs[1]);
    loop_descrs[1] = given_descrs[1];
    // 初始化输出描述符为 NULL
    PyArray_Descr *out_descr = NULL;
    // 将给定描述符数组中的第一个描述符转换为字符串类型描述符对象
    PyArray_StringDTypeObject *idescr =
            (PyArray_StringDTypeObject *)given_descrs[0];

    // 如果第三个给定描述符为空
    if (given_descrs[2] == NULL) {
        // 创建新的字符串类型实例作为输出描述符
        out_descr = (PyArray_Descr *)new_stringdtype_instance(
                idescr->na_object, idescr->coerce);
        // 如果创建失败，则返回错误标志
        if (out_descr == NULL) {
            return (NPY_CASTING)-1;
        }
    }
    else {
        // 如果第三个给定描述符不为空，则增加其引用计数并直接赋给输出描述符
        Py_INCREF(given_descrs[2]);
        out_descr = given_descrs[2];
    }
    // 将输出描述符赋给循环描述符数组中的第三个位置
    loop_descrs[2] = out_descr;
    // 返回无需转换的标志
    return NPY_NO_CASTING;
}


static int
string_expandtabs_strided_loop(PyArrayMethod_Context *context,
                               char *const data[],
                               npy_intp const dimensions[],
                               npy_intp const strides[],
                               NpyAuxData *NPY_UNUSED(auxdata))
{
    // 获取输入字符串数组的指针
    char *in1 = data[0];
    char *in2 = data[1];
    // 获取输出字符串数组的指针
    char *out = data[2];

    // 获取数组的长度
    npy_intp N = dimensions[0];

    // 将描述符数组中的第一个描述符转换为字符串类型描述符对象
    PyArray_StringDTypeObject *descr0 =
            (PyArray_StringDTypeObject *)context->descriptors[0];
    // 检查第一个描述符对象是否包含字符串 NA
    int has_string_na = descr0->has_string_na;
    // 获取默认字符串
    const npy_static_string *default_string = &descr0->default_string;


    // 分配字符串操作所需的分配器数组
    npy_string_allocator *allocators[3] = {};
    NpyString_acquire_allocators(3, context->descriptors, allocators);
    npy_string_allocator *iallocator = allocators[0];
    // allocators[1] 是 NULL，因为这里没有第二个输入
    npy_string_allocator *oallocator = allocators[2];
    // 使用 while 循环，循环执行 N 次，每次处理一个字符串
    while (N--) {
        // 将输入指针强制转换为 npy_packed_static_string 类型，赋给 ips
        const npy_packed_static_string *ips = (npy_packed_static_string *)in1;
        // 将输出指针强制转换为 npy_packed_static_string 类型，赋给 ops
        npy_packed_static_string *ops = (npy_packed_static_string *)out;
        // 创建并初始化 npy_static_string 结构体 is
        npy_static_string is = {0, NULL};
        // 从输入中读取 tabsize，赋给 tabsize
        npy_int64 tabsize = *(npy_int64 *)in2;

        // 调用 NpyString_load 函数加载字符串 is
        int isnull = NpyString_load(iallocator, ips, &is);

        // 如果加载失败，抛出内存错误异常，并跳转到 fail 标签处
        if (isnull == -1) {
            npy_gil_error(
                    PyExc_MemoryError, "Failed to load string in expandtabs");
            goto fail;
        }
        // 如果加载的字符串为空
        else if (isnull) {
            // 如果不支持字符串 NA（缺失值），抛出值错误异常，并跳转到 fail 标签处
            if (!has_string_na) {
                npy_gil_error(PyExc_ValueError,
                              "Null values are not supported arguments for "
                              "expandtabs");
                goto fail;
            }
            // 否则，使用默认字符串替代 is
            else {
                is = *default_string;
            }
        }

        // 使用 Buffer 类型的 buf 封装 is 的数据和大小
        Buffer<ENCODING::UTF8> buf((char *)is.buf, is.size);
        // 计算扩展制表符后的新字符串大小
        npy_intp new_buf_size = string_expandtabs_length(buf, tabsize);

        // 如果新字符串大小小于 0，跳转到 fail 标签处
        if (new_buf_size < 0) {
            goto fail;
        }

        // 分配 new_buf_size 大小的内存，并初始化为 0，赋给 new_buf
        char *new_buf = (char *)PyMem_RawCalloc(new_buf_size, 1);
        // 使用 Buffer 类型的 outbuf 封装 new_buf 和 new_buf_size
        Buffer<ENCODING::UTF8> outbuf(new_buf, new_buf_size);

        // 执行字符串扩展制表符处理
        string_expandtabs(buf, tabsize, outbuf);

        // 将处理后的字符串打包到 ops 中，如果失败，抛出内存错误异常，并跳转到 fail 标签处
        if (NpyString_pack(oallocator, ops, new_buf, new_buf_size) < 0) {
            npy_gil_error(
                    PyExc_MemoryError, "Failed to pack string in expandtabs");
            goto fail;
        }

        // 释放 new_buf 的内存
        PyMem_RawFree(new_buf);

        // 更新输入和输出指针位置
        in1 += strides[0];
        in2 += strides[1];
        out += strides[2];
    }

    // 释放所有分配器
    NpyString_release_allocators(3, allocators);
    // 成功执行，返回 0
    return 0;

  fail:
    // 发生失败，释放所有分配器，并返回 -1
    NpyString_release_allocators(3, allocators);
    return -1;
}

static NPY_CASTING
center_ljust_rjust_resolve_descriptors(
        struct PyArrayMethodObject_tag *NPY_UNUSED(method),
        PyArray_DTypeMeta *const dtypes[], PyArray_Descr *const given_descrs[],
        PyArray_Descr *loop_descrs[], npy_intp *NPY_UNUSED(view_offset))
{
    PyArray_StringDTypeObject *input_descr = (PyArray_StringDTypeObject *)given_descrs[0];
    // 获取输入字符串类型描述符
    PyArray_StringDTypeObject *fill_descr = (PyArray_StringDTypeObject *)given_descrs[2];
    // 获取填充字符串类型描述符
    int out_coerce = input_descr->coerce && fill_descr->coerce;
    // 检查是否需要强制转换输出
    PyObject *out_na_object = NULL;

    if (stringdtype_compatible_na(
                input_descr->na_object, fill_descr->na_object, &out_na_object) == -1) {
        // 检查输入和填充描述符的兼容性，设置输出 NA 对象
        return (NPY_CASTING)-1;
        // 若兼容性检查失败则返回错误码
    }

    Py_INCREF(given_descrs[0]);
    // 增加输入描述符的引用计数
    loop_descrs[0] = given_descrs[0];
    // 设置循环描述符数组的第一个元素为输入描述符
    Py_INCREF(given_descrs[1]);
    // 增加输入描述符的引用计数
    loop_descrs[1] = given_descrs[1];
    // 设置循环描述符数组的第二个元素为输入描述符
    Py_INCREF(given_descrs[2]);
    // 增加输入描述符的引用计数
    loop_descrs[2] = given_descrs[2];
    // 设置循环描述符数组的第三个元素为输入描述符

    PyArray_Descr *out_descr = NULL;

    if (given_descrs[3] == NULL) {
        // 若输出描述符为空
        out_descr = (PyArray_Descr *)new_stringdtype_instance(
                out_na_object, out_coerce);
        // 根据输出 NA 对象和是否需要强制转换创建新的字符串类型实例

        if (out_descr == NULL) {
            // 若创建实例失败
            return (NPY_CASTING)-1;
            // 返回错误码
        }
    }
    else {
        // 若输出描述符不为空
        Py_INCREF(given_descrs[3]);
        // 增加输出描述符的引用计数
        out_descr = given_descrs[3];
        // 设置输出描述符为给定的输出描述符
    }

    loop_descrs[3] = out_descr;
    // 设置循环描述符数组的第四个元素为输出描述符

    return NPY_NO_CASTING;
    // 返回无需转换的标志
}


static int
center_ljust_rjust_strided_loop(PyArrayMethod_Context *context,
                                char *const data[],
                                npy_intp const dimensions[],
                                npy_intp const strides[],
                                NpyAuxData *NPY_UNUSED(auxdata))
{
    PyArray_StringDTypeObject *s1descr = (PyArray_StringDTypeObject *)context->descriptors[0];
    // 获取第一个字符串描述符
    int has_null = s1descr->na_object != NULL;
    // 检查第一个描述符是否有 NA 对象
    int has_nan_na = s1descr->has_nan_na;
    // 检查第一个描述符是否有 NaN 或 NA
    int has_string_na = s1descr->has_string_na;
    // 检查第一个描述符是否有字符串 NA
    const npy_static_string *default_string = &s1descr->default_string;
    // 获取第一个描述符的默认字符串
    npy_intp N = dimensions[0];
    // 获取维度的大小
    char *in1 = data[0];
    // 获取输入数据数组的第一个元素
    char *in2 = data[1];
    // 获取输入数据数组的第二个元素
    char *in3 = data[2];
    // 获取输入数据数组的第三个元素
    char *out = data[3];
    // 获取输出数据数组的第四个元素
    npy_intp in1_stride = strides[0];
    // 获取输入数据数组的第一个元素的步长
    npy_intp in2_stride = strides[1];
    // 获取输入数据数组的第二个元素的步长
    npy_intp in3_stride = strides[2];
    // 获取输入数据数组的第三个元素的步长
    npy_intp out_stride = strides[3];
    // 获取输出数据数组的第四个元素的步长

    npy_string_allocator *allocators[4] = {};
    // 创建字符串分配器数组
    NpyString_acquire_allocators(4, context->descriptors, allocators);
    // 获取字符串分配器

    npy_string_allocator *s1allocator = allocators[0];
    // 获取第一个字符串分配器
    // allocators[1] is NULL
    npy_string_allocator *s2allocator = allocators[2];
    // 获取第三个字符串分配器
    npy_string_allocator *oallocator = allocators[3];
    // 获取第四个字符串分配器

    JUSTPOSITION pos = *(JUSTPOSITION *)(context->method->static_data);
    // 获取 JUSTPOSITION 结构体
    const char* ufunc_name = ((PyUFuncObject *)context->caller)->name;
    // 获取 UFunc 的名称

    }

    NpyString_release_allocators(4, allocators);
    // 释放字符串分配器
    return 0;

 fail:
    NpyString_release_allocators(4, allocators);
    // 释放字符串分配器
    return -1;
}

static int
# 定义字符串分割循环函数，接受一些输入参数和执行环境
zfill_strided_loop(PyArrayMethod_Context *context,
                   char *const data[], npy_intp const dimensions[],
                   npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    # 获取描述符数组中的字符串类型描述符
    PyArray_StringDTypeObject *idescr =
            (PyArray_StringDTypeObject *)context->descriptors[0];
    # 获取第一个维度的大小
    npy_intp N = dimensions[0];
    # 输入数组的第一个，第二个和输出数组的指针
    char *in1 = data[0];
    char *in2 = data[1];
    char *out = data[2];
    # 输入数组的步幅
    npy_intp in1_stride = strides[0];
    npy_intp in2_stride = strides[1];
    npy_intp out_stride = strides[2];

    # 字符串分配器数组，用于存储字符串分配器的指针
    npy_string_allocator *allocators[3] = {};
    # 获取字符串分配器
    NpyString_acquire_allocators(3, context->descriptors, allocators);
    npy_string_allocator *iallocator = allocators[0];
    # allocators[1] 是 NULL
    npy_string_allocator *oallocator = allocators[2];
    # 是否具有空值标志
    int has_null = idescr->na_object != NULL;
    # 是否具有NaN或NA标志
    int has_nan_na = idescr->has_nan_na;
    # 是否具有字符串NA标志
    int has_string_na = idescr->has_string_na;
    # 默认字符串
    const npy_static_string *default_string = &idescr->default_string;

    # 释放字符串分配器资源
    }

    # 在失败时释放字符串分配器资源，并返回失败状态
    NpyString_release_allocators(3, allocators);
    return 0;

fail:
    NpyString_release_allocators(3, allocators);
    return -1;
}


# 解析字符串分割函数描述符，用于字符串数据类型对象的处理
static NPY_CASTING
string_partition_resolve_descriptors(
        PyArrayMethodObject *self,
        PyArray_DTypeMeta *const NPY_UNUSED(dtypes[3]),
        PyArray_Descr *const given_descrs[3],
        PyArray_Descr *loop_descrs[3],
        npy_intp *NPY_UNUSED(view_offset))
{
    # 如果给定的描述符中包含不支持的 'out' 关键字，则引发类型错误异常
    if (given_descrs[2] || given_descrs[3] || given_descrs[4]) {
        PyErr_Format(PyExc_TypeError, "The StringDType '%s' ufunc does not "
                     "currently support the 'out' keyword", self->name);
        return (NPY_CASTING)-1;
    }

    # 获取给定的字符串类型描述符
    PyArray_StringDTypeObject *descr1 = (PyArray_StringDTypeObject *)given_descrs[0];
    PyArray_StringDTypeObject *descr2 = (PyArray_StringDTypeObject *)given_descrs[1];
    # 确定是否需要强制类型转换
    int out_coerce = descr1->coerce && descr2->coerce;
    PyObject *out_na_object = NULL;

    # 检查是否兼容NA对象
    if (stringdtype_compatible_na(
                descr1->na_object, descr2->na_object, &out_na_object) == -1) {
        return (NPY_CASTING)-1;
    }

    # 增加引用计数并复制描述符
    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];
    Py_INCREF(given_descrs[1]);
    loop_descrs[1] = given_descrs[1];

    # 对于索引从2到4的循环描述符
    for (int i=2; i<5; i++) {
        # 创建新的字符串数据类型实例
        loop_descrs[i] = (PyArray_Descr *)new_stringdtype_instance(
                out_na_object, out_coerce);
        # 如果创建失败，则返回错误状态
        if (loop_descrs[i] == NULL) {
            return (NPY_CASTING)-1;
        }
    }

    # 返回无需类型转换的标志
    return NPY_NO_CASTING;
}

# 字符串分割的分段循环函数
NPY_NO_EXPORT int
string_partition_strided_loop(
        PyArrayMethod_Context *context,
        char *const data[],
        npy_intp const dimensions[],
        npy_intp const strides[],
        NpyAuxData *NPY_UNUSED(auxdata))
{
    # 获取字符串分割方法的起始位置
    STARTPOSITION startposition = *(STARTPOSITION *)(context->method->static_data);
    # 确定快速搜索方向
    int fastsearch_direction =
            startposition == STARTPOSITION::FRONT ? FAST_SEARCH : FAST_RSEARCH;

    # 获取第一个维度的大小
    npy_intp N = dimensions[0];

    # 输入数组的指针
    char *in1 = data[0];
    char *in2 = data[1];
    # 获取输入和输出数据指针，从传入的数据数组中取出特定索引处的指针
    char *out1 = data[2];
    char *out2 = data[3];
    char *out3 = data[4];

    # 获取输入和输出数据的步长（strides）
    npy_intp in1_stride = strides[0];
    npy_intp in2_stride = strides[1];
    npy_intp out1_stride = strides[2];
    npy_intp out2_stride = strides[3];
    npy_intp out3_stride = strides[4];

    # 分配字符串处理所需的内存分配器
    npy_string_allocator *allocators[5] = {};
    NpyString_acquire_allocators(5, context->descriptors, allocators);
    npy_string_allocator *in1allocator = allocators[0];
    npy_string_allocator *in2allocator = allocators[1];
    npy_string_allocator *out1allocator = allocators[2];
    npy_string_allocator *out2allocator = allocators[3];
    npy_string_allocator *out3allocator = allocators[4];

    # 获取字符串类型描述符对象，并从中获取特定信息
    PyArray_StringDTypeObject *idescr =
            (PyArray_StringDTypeObject *)context->descriptors[0];
    int has_string_na = idescr->has_string_na;
    const npy_static_string *default_string = &idescr->default_string;

    # 释放分配的内存分配器资源
    NpyString_release_allocators(5, allocators);
    # 返回成功标志
    return 0;

  fail:

    # 在失败时释放已分配的内存分配器资源
    NpyString_release_allocators(5, allocators);
    # 返回失败标志
    return -1;
}

NPY_NO_EXPORT int
string_inputs_promoter(
        PyObject *ufunc_obj, PyArray_DTypeMeta *const op_dtypes[],
        PyArray_DTypeMeta *const signature[],
        PyArray_DTypeMeta *new_op_dtypes[],
        PyArray_DTypeMeta *final_dtype,
        PyArray_DTypeMeta *result_dtype)
{
    PyUFuncObject *ufunc = (PyUFuncObject *)ufunc_obj;
    /* set all input operands to final_dtype */
    for (int i = 0; i < ufunc->nin; i++) {
        PyArray_DTypeMeta *tmp = final_dtype;
        if (signature[i]) {
            tmp = signature[i]; /* never replace a fixed one. */
        }
        Py_INCREF(tmp);
        new_op_dtypes[i] = tmp;
    }
    /* don't touch output dtypes if they are set */
    for (int i = ufunc->nin; i < ufunc->nargs; i++) {
        if (op_dtypes[i] != NULL) {
            Py_INCREF(op_dtypes[i]);
            new_op_dtypes[i] = op_dtypes[i];
        }
        else {
            Py_INCREF(result_dtype);
            new_op_dtypes[i] = result_dtype;
        }
    }

    return 0;
}

static int
string_object_bool_output_promoter(
        PyObject *ufunc, PyArray_DTypeMeta *const op_dtypes[],
        PyArray_DTypeMeta *const signature[],
        PyArray_DTypeMeta *new_op_dtypes[])
{
    return string_inputs_promoter(
            ufunc, op_dtypes, signature,
            new_op_dtypes, &PyArray_ObjectDType, &PyArray_BoolDType);
}

static int
string_unicode_bool_output_promoter(
        PyObject *ufunc, PyArray_DTypeMeta *const op_dtypes[],
        PyArray_DTypeMeta *const signature[],
        PyArray_DTypeMeta *new_op_dtypes[])
{
    return string_inputs_promoter(
            ufunc, op_dtypes, signature,
            new_op_dtypes, &PyArray_StringDType, &PyArray_BoolDType);
}

static int
is_integer_dtype(PyArray_DTypeMeta *DType)
{
    if (DType == &PyArray_PyLongDType) {
        return 1;
    }
    else if (DType == &PyArray_Int8DType) {
        return 1;
    }
    else if (DType == &PyArray_Int16DType) {
        return 1;
    }
    else if (DType == &PyArray_Int32DType) {
        return 1;
    }
    // int64 already has a loop registered for it,
    // so don't need to consider it
#if NPY_SIZEOF_BYTE == NPY_SIZEOF_SHORT
    else if (DType == &PyArray_ByteDType) {
        return 1;
    }
#endif
#if NPY_SIZEOF_SHORT == NPY_SIZEOF_INT
    else if (DType == &PyArray_ShortDType) {
        return 1;
    }
#endif
#if NPY_SIZEOF_INT == NPY_SIZEOF_LONG
    else if (DType == &PyArray_IntDType) {
        return 1;
    }
#endif
#if NPY_SIZEOF_LONGLONG == NPY_SIZEOF_LONG
    else if (DType == &PyArray_LongLongDType) {
        return 1;
    }
#endif
    else if (DType == &PyArray_UInt8DType) {
        return 1;
    }
    else if (DType == &PyArray_UInt16DType) {
        return 1;
    }
    else if (DType == &PyArray_UInt32DType) {
        return 1;
    }
    // uint64 already has a loop registered for it,
    // so don't need to consider it
#if NPY_SIZEOF_BYTE == NPY_SIZEOF_SHORT
    else if (DType == &PyArray_UByteDType) {
        return 1;
    }
#endif

    // No match found, not an integer dtype
    return 0;
}
#ifdef
#if NPY_SIZEOF_SHORT == NPY_SIZEOF_INT
    // 如果短整型和整型的字节大小相同，返回1
    else if (DType == &PyArray_UShortDType) {
        return 1;
    }
#endif
#if NPY_SIZEOF_INT == NPY_SIZEOF_LONG
    // 如果整型和长整型的字节大小相同，返回1
    else if (DType == &PyArray_UIntDType) {
        return 1;
    }
#endif
#if NPY_SIZEOF_LONGLONG == NPY_SIZEOF_LONG
    // 如果长长整型和长整型的字节大小相同，返回1
    else if (DType == &PyArray_ULongLongDType) {
        return 1;
    }
#endif
    // 默认情况下返回0
    return 0;
}


static int
string_multiply_promoter(PyObject *ufunc_obj,
                         PyArray_DTypeMeta *const op_dtypes[],
                         PyArray_DTypeMeta *const signature[],
                         PyArray_DTypeMeta *new_op_dtypes[])
{
    PyUFuncObject *ufunc = (PyUFuncObject *)ufunc_obj;
    // 遍历输入参数
    for (int i = 0; i < ufunc->nin; i++) {
        PyArray_DTypeMeta *tmp = NULL;
        // 如果有签名类型，则使用签名类型
        if (signature[i]) {
            tmp = signature[i];
        }
        // 如果是整数类型，则使用64位整型
        else if (is_integer_dtype(op_dtypes[i])) {
            tmp = &PyArray_Int64DType;
        }
        // 如果存在操作类型，则使用该类型
        else if (op_dtypes[i]) {
            tmp = op_dtypes[i];
        }
        // 默认使用字符串类型
        else {
            tmp = &PyArray_StringDType;
        }
        // 增加临时类型的引用计数
        Py_INCREF(tmp);
        new_op_dtypes[i] = tmp;
    }
    /* don't touch output dtypes if they are set */
    // 如果输出类型已经设置，则不修改输出类型
    for (int i = ufunc->nin; i < ufunc->nargs; i++) {
        if (op_dtypes[i]) {
            // 增加输出类型的引用计数
            Py_INCREF(op_dtypes[i]);
            new_op_dtypes[i] = op_dtypes[i];
        }
        else {
            // 默认使用字符串类型
            Py_INCREF(&PyArray_StringDType);
            new_op_dtypes[i] = &PyArray_StringDType;
        }
    }
    return 0;
}

// Register a ufunc.
//
// Pass NULL for resolve_func to use the default_resolve_descriptors.
int
init_ufunc(PyObject *umath, const char *ufunc_name, PyArray_DTypeMeta **dtypes,
           PyArrayMethod_ResolveDescriptors *resolve_func,
           PyArrayMethod_StridedLoop *loop_func, int nin, int nout,
           NPY_CASTING casting, NPY_ARRAYMETHOD_FLAGS flags,
           void *static_data)
{
    PyObject *ufunc = PyObject_GetAttrString(umath, ufunc_name);
    if (ufunc == NULL) {
        return -1;
    }
    char loop_name[256] = {0};

    // 根据ufunc_name创建循环名称
    snprintf(loop_name, sizeof(loop_name), "string_%s", ufunc_name);

    // 定义PyArrayMethod_Spec结构体
    PyArrayMethod_Spec spec;
    spec.name = loop_name;
    spec.nin = nin;
    spec.nout = nout;
    spec.casting = casting;
    spec.flags = flags;
    spec.dtypes = dtypes;
    spec.slots = NULL;

    // 设置PyType_Slot resolve_slots数组
    PyType_Slot resolve_slots[] = {
            {NPY_METH_resolve_descriptors, (void *)resolve_func},
            {NPY_METH_strided_loop, (void *)loop_func},
            {_NPY_METH_static_data, static_data},
            {0, NULL}};

    // 将resolve_slots数组赋值给spec.slots
    spec.slots = resolve_slots;

    // 将循环规范添加到ufunc中
    if (PyUFunc_AddLoopFromSpec_int(ufunc, &spec, 1) < 0) {
        Py_DECREF(ufunc);
        return -1;
    }

    // 减少ufunc的引用计数
    Py_DECREF(ufunc);
    return 0;
}

int
add_promoter(PyObject *numpy, const char *ufunc_name,
             PyArray_DTypeMeta *dtypes[], size_t n_dtypes,
             PyArrayMethod_PromoterFunction *promoter_impl)
{
    // 从numpy对象中获取ufunc对象
    PyObject *ufunc = PyObject_GetAttrString((PyObject *)numpy, ufunc_name);
    if (ufunc == NULL) {
        return -1;
    }
    // 剩余代码未提供，不进行注释
    # 如果传入的 ufunc 指针为空，则直接返回错误码 -1
    if (ufunc == NULL) {
        return -1;
    }

    # 创建一个包含 n_dtypes 个元素的 Python 元组对象 DType_tuple
    PyObject *DType_tuple = PyTuple_New(n_dtypes);

    # 如果创建元组失败，则释放之前分配的 ufunc 对象，并返回错误码 -1
    if (DType_tuple == NULL) {
        Py_DECREF(ufunc);
        return -1;
    }

    # 遍历 dtypes 数组，对每个 PyObject 指针增加引用计数并添加到 DType_tuple 中
    for (size_t i=0; i<n_dtypes; i++) {
        Py_INCREF((PyObject *)dtypes[i]);
        PyTuple_SET_ITEM(DType_tuple, i, (PyObject *)dtypes[i]);
    }

    # 创建一个 PyCapsule 对象 promoter_capsule，用于包装 promoter_impl 函数指针
    PyObject *promoter_capsule = PyCapsule_New((void *)promoter_impl,
                                               "numpy._ufunc_promoter", NULL);

    # 如果创建 PyCapsule 失败，则释放之前分配的对象，并返回错误码 -1
    if (promoter_capsule == NULL) {
        Py_DECREF(ufunc);
        Py_DECREF(DType_tuple);
        return -1;
    }

    # 将 DType_tuple 和 promoter_capsule 传递给 PyUFunc_AddPromoter 函数，将数据类型和
    # 优化器函数注册到 ufunc 中。如果添加失败，则释放相关对象并返回错误码 -1
    if (PyUFunc_AddPromoter(ufunc, DType_tuple, promoter_capsule) < 0) {
        Py_DECREF(promoter_capsule);
        Py_DECREF(DType_tuple);
        Py_DECREF(ufunc);
        return -1;
    }

    # 成功添加优化器后，释放 promoter_capsule、DType_tuple 和 ufunc 对象的引用计数
    Py_DECREF(promoter_capsule);
    Py_DECREF(DType_tuple);
    Py_DECREF(ufunc);

    # 返回成功标志码 0
    return 0;
#define INIT_MULTIPLY(typename, shortname)                                 \
    // 创建一个用于右操作数为指定类型的乘法通用函数类型数组
    PyArray_DTypeMeta *multiply_right_##shortname##_types[] = {            \
        &PyArray_StringDType, &PyArray_##typename##DType,                  \
        &PyArray_StringDType};                                             \
                                                                           \
    // 初始化右操作数为指定类型的乘法通用函数
    if (init_ufunc(umath, "multiply", multiply_right_##shortname##_types,  \
                   &multiply_resolve_descriptors,                          \
                   &multiply_right_strided_loop<npy_##shortname>, 2, 1,    \
                   NPY_NO_CASTING, (NPY_ARRAYMETHOD_FLAGS) 0, NULL) < 0) { \
        return -1;                                                         \
    }                                                                      \
                                                                           \
    // 创建一个用于左操作数为指定类型的乘法通用函数类型数组
    PyArray_DTypeMeta *multiply_left_##shortname##_types[] = {             \
            &PyArray_##typename##DType, &PyArray_StringDType,              \
            &PyArray_StringDType};                                         \
                                                                           \
    // 初始化左操作数为指定类型的乘法通用函数
    if (init_ufunc(umath, "multiply", multiply_left_##shortname##_types,   \
                   &multiply_resolve_descriptors,                          \
                   &multiply_left_strided_loop<npy_##shortname>, 2, 1,     \
                   NPY_NO_CASTING, (NPY_ARRAYMETHOD_FLAGS) 0, NULL) < 0) { \
        return -1;                                                         \
    }
    // 定义一个静态常量字符指针数组，包含6个字符串，表示比较操作的函数名
    static const char *comparison_ufunc_names[6] = {
            "equal", "not_equal",
            "less", "less_equal", "greater_equal", "greater",
    };

    // 定义一个 PyArray_DTypeMeta 指针数组，包含3个数据类型元信息对象，用于比较操作
    PyArray_DTypeMeta *comparison_dtypes[] = {
            &PyArray_StringDType,
            &PyArray_StringDType, &PyArray_BoolDType};

    // 定义一个静态布尔数组，表示在 string_cmp_strided_loop 函数中，eq 和 ne 的比较结果
    static npy_bool comparison_ufunc_eq_lt_gt_results[6*3] = {
        NPY_TRUE, NPY_FALSE, NPY_FALSE, // eq: results for eq, lt, gt
        NPY_FALSE, NPY_TRUE, NPY_TRUE,  // ne
        NPY_FALSE, NPY_TRUE, NPY_FALSE, // lt
        NPY_TRUE, NPY_TRUE, NPY_FALSE,  // le
        NPY_TRUE, NPY_FALSE, NPY_TRUE,  // gt
        NPY_FALSE, NPY_FALSE, NPY_TRUE, // ge
    };

    // 循环初始化比较操作的通用函数
    for (int i = 0; i < 6; i++) {
        // 初始化通用函数 umath 中的比较函数，如果失败则返回 -1
        if (init_ufunc(umath, comparison_ufunc_names[i], comparison_dtypes,
                       &string_comparison_resolve_descriptors,
                       &string_comparison_strided_loop, 2, 1, NPY_NO_CASTING,
                       (NPY_ARRAYMETHOD_FLAGS) 0,
                       &comparison_ufunc_eq_lt_gt_results[i*3]) < 0) {
            return -1;
        }

        // 将对象和 Unicode 提升器添加到 umath 中的比较函数中，如果失败则返回 -1
        if (add_object_and_unicode_promoters(
                    umath, comparison_ufunc_names[i],
                    &string_unicode_bool_output_promoter,
                    &string_object_bool_output_promoter) < 0) {
            return -1;
        }
    }

    // 定义一个布尔输出数据类型的 PyArray_DTypeMeta 指针数组
    PyArray_DTypeMeta *bool_output_dtypes[] = {
        &PyArray_StringDType,
        &PyArray_BoolDType
    };

    // 初始化 umath 中的 "isnan" 函数，如果失败则返回 -1
    if (init_ufunc(umath, "isnan", bool_output_dtypes,
                   &string_bool_output_resolve_descriptors,
                   &string_isnan_strided_loop, 1, 1, NPY_NO_CASTING,
                   (NPY_ARRAYMETHOD_FLAGS) 0, NULL) < 0) {
        return -1;
    }

    // 定义一个一元操作的函数名字符串数组
    const char *unary_loop_names[] = {
        "isalpha", "isdecimal", "isdigit", "isnumeric", "isspace",
        "isalnum", "istitle", "isupper", "islower",
    };

    // 定义一组 utf8_buffer_method 指针数组，指向一元操作函数成员方法
    static utf8_buffer_method unary_loop_buffer_methods[] = {
        &Buffer<ENCODING::UTF8>::isalpha,
        &Buffer<ENCODING::UTF8>::isdecimal,
        &Buffer<ENCODING::UTF8>::isdigit,
        &Buffer<ENCODING::UTF8>::isnumeric,
        &Buffer<ENCODING::UTF8>::isspace,
        &Buffer<ENCODING::UTF8>::isalnum,
        &Buffer<ENCODING::UTF8>::istitle,
        &Buffer<ENCODING::UTF8>::isupper,
        &Buffer<ENCODING::UTF8>::islower,
    };

    // 循环初始化一元操作的通用函数
    for (int i=0; i<9; i++) {
        // 初始化通用函数 umath 中的一元操作函数，如果失败则返回 -1
        if (init_ufunc(umath, unary_loop_names[i], bool_output_dtypes,
                       &string_bool_output_resolve_descriptors,
                       &string_bool_output_unary_strided_loop, 1, 1, NPY_NO_CASTING,
                       (NPY_ARRAYMETHOD_FLAGS) 0,
                       &unary_loop_buffer_methods[i]) < 0) {
            return -1;
        }
    }
    // 定义一个包含两种数据类型元信息的数组，分别为字符串和整数指针
    PyArray_DTypeMeta *intp_output_dtypes[] = {
        &PyArray_StringDType,
        &PyArray_IntpDType
    };

    // 初始化一个ufunc函数，用于计算字符串长度，设置输出数据类型，解析描述符，并设置循环函数
    if (init_ufunc(umath, "str_len", intp_output_dtypes,
                   &string_intp_output_resolve_descriptors,
                   &string_strlen_strided_loop, 1, 1, NPY_NO_CASTING,
                   (NPY_ARRAYMETHOD_FLAGS) 0, NULL) < 0) {
        return -1;
    }

    // 定义一个包含三种字符串数据类型元信息的数组
    PyArray_DTypeMeta *binary_dtypes[] = {
            &PyArray_StringDType,
            &PyArray_StringDType,
            &PyArray_StringDType,
    };

    // 定义包含字符串数组的名称
    const char* minimum_maximum_names[] = {"minimum", "maximum"};

    // 定义静态布尔数组，用于指示最小和最大的反转状态
    static npy_bool minimum_maximum_invert[2] = {NPY_FALSE, NPY_TRUE};

    // 循环处理最小和最大名称，为每个名称初始化一个ufunc函数
    for (int i = 0; i < 2; i++) {
        if (init_ufunc(umath, minimum_maximum_names[i],
                       binary_dtypes, binary_resolve_descriptors,
                       &minimum_maximum_strided_loop, 2, 1, NPY_NO_CASTING,
                       (NPY_ARRAYMETHOD_FLAGS) 0,
                       &minimum_maximum_invert[i]) < 0) {
            return -1;
        }
    }

    // 初始化一个ufunc函数，用于执行加法操作，设置输入输出数据类型和相关处理函数
    if (init_ufunc(umath, "add", binary_dtypes, binary_resolve_descriptors,
                   &add_strided_loop, 2, 1, NPY_NO_CASTING,
                   (NPY_ARRAYMETHOD_FLAGS) 0, NULL) < 0) {
        return -1;
    }

    // 定义包含三种数据类型元信息的数组，用于所有字符串推广操作
    PyArray_DTypeMeta *rall_strings_promoter_dtypes[] = {
        &PyArray_StringDType,
        &PyArray_UnicodeDType,
        &PyArray_StringDType,
    };

    // 添加所有字符串推广操作的ufunc函数，并指定数据类型和推广函数
    if (add_promoter(umath, "add", rall_strings_promoter_dtypes, 3,
                     all_strings_promoter) < 0) {
        return -1;
    }

    // 定义包含三种数据类型元信息的数组，用于左侧所有字符串推广操作
    PyArray_DTypeMeta *lall_strings_promoter_dtypes[] = {
        &PyArray_UnicodeDType,
        &PyArray_StringDType,
        &PyArray_StringDType,
    };

    // 添加左侧所有字符串推广操作的ufunc函数，并指定数据类型和推广函数
    if (add_promoter(umath, "add", lall_strings_promoter_dtypes, 3,
                     all_strings_promoter) < 0) {
        return -1;
    }

    // 初始化 Int64 和 UInt64 的乘法操作的宏定义
    INIT_MULTIPLY(Int64, int64);
    INIT_MULTIPLY(UInt64, uint64);

    // 定义包含三种数据类型元信息的数组，用于字符串乘法操作
    PyArray_DTypeMeta *rdtypes[] = {
        &PyArray_StringDType,
        &PyArray_IntAbstractDType,
        &PyArray_StringDType};

    // 添加字符串乘法操作的ufunc函数，并指定数据类型和推广函数
    if (add_promoter(umath, "multiply", rdtypes, 3, string_multiply_promoter) < 0) {
        return -1;
    }

    // 定义包含三种数据类型元信息的数组，用于左侧字符串乘法操作
    PyArray_DTypeMeta *ldtypes[] = {
        &PyArray_IntAbstractDType,
        &PyArray_StringDType,
        &PyArray_StringDType};

    // 添加左侧字符串乘法操作的ufunc函数，并指定数据类型和推广函数
    if (add_promoter(umath, "multiply", ldtypes, 3, string_multiply_promoter) < 0) {
        return -1;
    }

    // 定义包含五种数据类型元信息的数组，用于 find-like 操作
    PyArray_DTypeMeta *findlike_dtypes[] = {
        &PyArray_StringDType, &PyArray_StringDType,
        &PyArray_Int64DType, &PyArray_Int64DType,
        &PyArray_DefaultIntDType,
    };

    // 定义包含五种字符串操作名称的数组
    const char* findlike_names[] = {
        "find", "rfind", "index", "rindex", "count",
    };

    // 定义包含五种数据类型元信息的数组，用于所有字符串推广操作
    PyArray_DTypeMeta *findlike_promoter_dtypes[] = {
        &PyArray_StringDType, &PyArray_UnicodeDType,
        &PyArray_IntAbstractDType, &PyArray_IntAbstractDType,
        &PyArray_DefaultIntDType,
    };
    // 定义一个函数指针数组，包含查找类函数的指针
    find_like_function *findlike_functions[] = {
        // 使用UTF-8编码的字符串查找函数
        string_find<ENCODING::UTF8>,
        // 使用UTF-8编码的字符串反向查找函数
        string_rfind<ENCODING::UTF8>,
        // 使用UTF-8编码的字符串索引函数
        string_index<ENCODING::UTF8>,
        // 使用UTF-8编码的字符串反向索引函数
        string_rindex<ENCODING::UTF8>,
        // 使用UTF-8编码的字符串计数函数
        string_count<ENCODING::UTF8>,
    };

    // 遍历函数指针数组
    for (int i=0; i<5; i++) {
        // 初始化通用函数，设置查找函数的解析描述符和循环处理函数
        if (init_ufunc(umath, findlike_names[i], findlike_dtypes,
                       &string_findlike_resolve_descriptors,
                       &string_findlike_strided_loop, 4, 1, NPY_NO_CASTING,
                       (NPY_ARRAYMETHOD_FLAGS) 0,
                       (void *)findlike_functions[i]) < 0) {
            return -1;
        }

        // 添加字符串查找类函数的类型提升器
        if (add_promoter(umath, findlike_names[i],
                         findlike_promoter_dtypes,
                         5, string_findlike_promoter) < 0) {
            return -1;
        }
    }

    // 定义开始和结束字符串操作的数据类型数组
    PyArray_DTypeMeta *startswith_endswith_dtypes[] = {
        &PyArray_StringDType, &PyArray_StringDType,
        &PyArray_Int64DType, &PyArray_Int64DType,
        &PyArray_BoolDType,
    };

    // 开始和结束字符串操作的函数名数组
    const char* startswith_endswith_names[] = {
        "startswith", "endswith",
    };

    // 开始和结束字符串操作的类型提升器数据类型数组
    PyArray_DTypeMeta *startswith_endswith_promoter_dtypes[] = {
        &PyArray_StringDType, &PyArray_UnicodeDType,
        &PyArray_IntAbstractDType, &PyArray_IntAbstractDType,
        &PyArray_BoolDType,
    };

    // 开始和结束字符串操作的起始位置数组
    static STARTPOSITION startswith_endswith_startposition[] = {
        STARTPOSITION::FRONT,
        STARTPOSITION::BACK,
    };

    // 遍历开始和结束字符串操作的名称数组
    for (int i=0; i<2; i++) {
        // 初始化通用函数，设置开始和结束字符串操作的解析描述符和循环处理函数
        if (init_ufunc(umath, startswith_endswith_names[i], startswith_endswith_dtypes,
                       &string_startswith_endswith_resolve_descriptors,
                       &string_startswith_endswith_strided_loop,
                       4, 1, NPY_NO_CASTING, (NPY_ARRAYMETHOD_FLAGS) 0,
                       &startswith_endswith_startposition[i]) < 0) {
            return -1;
        }

        // 添加开始和结束字符串操作的类型提升器
        if (add_promoter(umath, startswith_endswith_names[i],
                         startswith_endswith_promoter_dtypes,
                         5, string_startswith_endswith_promoter) < 0) {
            return -1;
        }
    }

    // 定义去除字符串两端空白字符操作的数据类型数组
    PyArray_DTypeMeta *strip_whitespace_dtypes[] = {
        &PyArray_StringDType, &PyArray_StringDType
    };

    // 去除字符串两端空白字符操作的函数名数组
    const char *strip_whitespace_names[] = {
        "_lstrip_whitespace", "_rstrip_whitespace", "_strip_whitespace",
    };

    // 去除字符串两端空白字符操作的类型数组
    static STRIPTYPE strip_types[] = {
        STRIPTYPE::LEFTSTRIP,
        STRIPTYPE::RIGHTSTRIP,
        STRIPTYPE::BOTHSTRIP,
    };

    // 遍历去除字符串两端空白字符操作的名称数组
    for (int i=0; i<3; i++) {
        // 初始化通用函数，设置去除字符串两端空白字符的解析描述符和循环处理函数
        if (init_ufunc(umath, strip_whitespace_names[i], strip_whitespace_dtypes,
                       &strip_whitespace_resolve_descriptors,
                       &string_lrstrip_whitespace_strided_loop,
                       1, 1, NPY_NO_CASTING, (NPY_ARRAYMETHOD_FLAGS) 0,
                       &strip_types[i]) < 0) {
            return -1;
        }
    }
    # 定义包含三个字符类型的数组，用于字符串操作
    PyArray_DTypeMeta *strip_chars_dtypes[] = {
        &PyArray_StringDType, &PyArray_StringDType, &PyArray_StringDType
    };

    # 定义包含三个字符类型名称的数组
    const char *strip_chars_names[] = {
        "_lstrip_chars", "_rstrip_chars", "_strip_chars",
    };

    # 循环处理三个字符操作
    for (int i=0; i<3; i++) {
        # 初始化通用函数，处理去除字符操作
        if (init_ufunc(umath, strip_chars_names[i], strip_chars_dtypes,
                       &binary_resolve_descriptors,
                       &string_lrstrip_chars_strided_loop,
                       2, 1, NPY_NO_CASTING, (NPY_ARRAYMETHOD_FLAGS) 0,
                       &strip_types[i]) < 0) {
            return -1;
        }

        # 添加字符串推广器，处理右侧去除字符操作
        if (add_promoter(umath, strip_chars_names[i],
                         rall_strings_promoter_dtypes, 3,
                         all_strings_promoter) < 0) {
            return -1;
        }

        # 添加字符串推广器，处理左侧去除字符操作
        if (add_promoter(umath, strip_chars_names[i],
                         lall_strings_promoter_dtypes, 3,
                         all_strings_promoter) < 0) {
            return -1;
        }
    }

    # 定义包含五个替换操作类型的数组
    PyArray_DTypeMeta *replace_dtypes[] = {
        &PyArray_StringDType, &PyArray_StringDType, &PyArray_StringDType,
        &PyArray_Int64DType, &PyArray_StringDType,
    };

    # 初始化通用函数，处理字符串替换操作
    if (init_ufunc(umath, "_replace", replace_dtypes,
                   &replace_resolve_descriptors,
                   &string_replace_strided_loop, 4, 1,
                   NPY_NO_CASTING,
                   (NPY_ARRAYMETHOD_FLAGS) 0, NULL) < 0) {
        return -1;
    }

    # 定义包含五个替换操作推广器类型的数组（Python整数类型）
    PyArray_DTypeMeta *replace_promoter_pyint_dtypes[] = {
        &PyArray_StringDType, &PyArray_UnicodeDType, &PyArray_UnicodeDType,
        &PyArray_IntAbstractDType, &PyArray_StringDType,
    };

    # 添加字符串推广器，处理Python整数类型的替换操作
    if (add_promoter(umath, "_replace", replace_promoter_pyint_dtypes, 5,
                     string_replace_promoter) < 0) {
        return -1;
    }

    # 定义包含五个替换操作推广器类型的数组（Int64类型）
    PyArray_DTypeMeta *replace_promoter_int64_dtypes[] = {
        &PyArray_StringDType, &PyArray_UnicodeDType, &PyArray_UnicodeDType,
        &PyArray_Int64DType, &PyArray_StringDType,
    };

    # 添加字符串推广器，处理Int64类型的替换操作
    if (add_promoter(umath, "_replace", replace_promoter_int64_dtypes, 5,
                     string_replace_promoter) < 0) {
        return -1;
    }

    # 定义包含三个扩展制表符操作类型的数组
    PyArray_DTypeMeta *expandtabs_dtypes[] = {
        &PyArray_StringDType,
        &PyArray_Int64DType,
        &PyArray_StringDType,
    };

    # 初始化通用函数，处理扩展制表符操作
    if (init_ufunc(umath, "_expandtabs", expandtabs_dtypes,
                   &expandtabs_resolve_descriptors,
                   &string_expandtabs_strided_loop, 2, 1,
                   NPY_NO_CASTING,
                   (NPY_ARRAYMETHOD_FLAGS) 0, NULL) < 0) {
        return -1;
    }

    # 定义包含三个扩展制表符操作推广器类型的数组
    PyArray_DTypeMeta *expandtabs_promoter_dtypes[] = {
        &PyArray_StringDType,
        (PyArray_DTypeMeta *)Py_None,
        &PyArray_StringDType
    };

    # 添加字符串推广器，处理扩展制表符操作
    if (add_promoter(umath, "_expandtabs", expandtabs_promoter_dtypes,
                     3, string_multiply_promoter) < 0) {
        return -1;
    }
    # 定义一个数组，包含三种不同的 PyArray_DTypeMeta 类型的指针，用于处理 center、ljust、rjust 操作
    PyArray_DTypeMeta *center_ljust_rjust_dtypes[] = {
        &PyArray_StringDType,
        &PyArray_Int64DType,
        &PyArray_StringDType,
        &PyArray_StringDType,
    };
    
    # 定义一个静态数组，包含三个字符串常量，分别表示 center、ljust、rjust 操作的名称
    static const char* center_ljust_rjust_names[3] = {
        "_center", "_ljust", "_rjust"
    };
    
    # 定义一个静态数组，包含三个 JUSTPOSITION 枚举类型的元素，表示 center、ljust、rjust 操作的对齐位置
    static JUSTPOSITION positions[3] = {
        JUSTPOSITION::CENTER, JUSTPOSITION::LEFT, JUSTPOSITION::RIGHT
    };
    
    # 循环遍历三个操作名称
    for (int i=0; i<3; i++) {
        # 初始化通用函数对象（ufunc），注册 center、ljust、rjust 操作
        if (init_ufunc(umath, center_ljust_rjust_names[i],
                       center_ljust_rjust_dtypes,
                       &center_ljust_rjust_resolve_descriptors,
                       &center_ljust_rjust_strided_loop, 3, 1, NPY_NO_CASTING,
                       (NPY_ARRAYMETHOD_FLAGS) 0, &positions[i]) < 0) {
            return -1;
        }
    
        # 定义一个数组，包含四种不同的 PyArray_DTypeMeta 类型的指针，用于处理整数推广操作
        PyArray_DTypeMeta *int_promoter_dtypes[] = {
            &PyArray_StringDType,
            (PyArray_DTypeMeta *)Py_None,
            &PyArray_StringDType,
            &PyArray_StringDType,
        };
    
        # 将整数推广操作添加到通用函数对象（ufunc）中
        if (add_promoter(umath, center_ljust_rjust_names[i],
                         int_promoter_dtypes, 4,
                         string_multiply_promoter) < 0) {
            return -1;
        }
    
        # 定义一个数组，包含四种不同的 PyArray_DTypeMeta 类型的指针，用于处理Unicode字符串推广操作
        PyArray_DTypeMeta *unicode_promoter_dtypes[] = {
            &PyArray_StringDType,
            (PyArray_DTypeMeta *)Py_None,
            &PyArray_UnicodeDType,
            &PyArray_StringDType,
        };
    
        # 将Unicode字符串推广操作添加到通用函数对象（ufunc）中
        if (add_promoter(umath, center_ljust_rjust_names[i],
                         unicode_promoter_dtypes, 4,
                         string_multiply_promoter) < 0) {
            return -1;
        }
    }
    
    # 定义一个数组，包含三种不同的 PyArray_DTypeMeta 类型的指针，用于处理 zfill 操作
    PyArray_DTypeMeta *zfill_dtypes[] = {
        &PyArray_StringDType,
        &PyArray_Int64DType,
        &PyArray_StringDType,
    };
    
    # 初始化通用函数对象（ufunc），注册 zfill 操作
    if (init_ufunc(umath, "_zfill", zfill_dtypes, multiply_resolve_descriptors,
                   zfill_strided_loop, 2, 1, NPY_NO_CASTING,
                   (NPY_ARRAYMETHOD_FLAGS) 0, NULL) < 0) {
        return -1;
    }
    
    # 定义一个数组，包含三种不同的 PyArray_DTypeMeta 类型的指针，用于处理整数推广操作
    PyArray_DTypeMeta *int_promoter_dtypes[] = {
            &PyArray_StringDType,
            (PyArray_DTypeMeta *)Py_None,
            &PyArray_StringDType,
    };
    
    # 将整数推广操作添加到通用函数对象（ufunc）中
    if (add_promoter(umath, "_zfill", int_promoter_dtypes, 3,
                     string_multiply_promoter) < 0) {
        return -1;
    }
    
    # 定义一个数组，包含五种不同的 PyArray_DTypeMeta 类型的指针，用于处理 partition 操作
    PyArray_DTypeMeta *partition_dtypes[] = {
        &PyArray_StringDType,
        &PyArray_StringDType,
        &PyArray_StringDType,
        &PyArray_StringDType,
        &PyArray_StringDType
    };
    
    # 定义一个静态数组，包含两个字符串常量，表示 partition 和 rpartition 操作的名称
    const char *partition_names[] = {"_partition", "_rpartition"};
    
    # 定义一个静态数组，包含两个 STARTPOSITION 枚举类型的元素，表示 partition 和 rpartition 操作的起始位置
    static STARTPOSITION partition_startpositions[] = {
        STARTPOSITION::FRONT, STARTPOSITION::BACK
    };
    # 循环两次，初始化分区函数
    for (int i=0; i<2; i++) {
        # 调用初始化通用函数 init_ufunc，初始化第 i 个分区
        if (init_ufunc(umath, partition_names[i], partition_dtypes,
                       string_partition_resolve_descriptors,
                       string_partition_strided_loop, 2, 3, NPY_NO_CASTING,
                       (NPY_ARRAYMETHOD_FLAGS) 0, &partition_startpositions[i]) < 0) {
            # 如果初始化失败，返回 -1
            return -1;
        }
    }

    # 初始化完成，返回 0 表示成功
    return 0;
}


注释：


# 这是一个单独的右大括号 '}'，用于闭合之前的代码块或函数定义。
# 在这里没有上下文，无法具体说明其用途，需要结合前面的代码来理解。
```