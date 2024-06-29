# `.\numpy\numpy\_core\src\umath\ufunc_object.c`

```
/*
 * Python Universal Functions Object -- Math for all types, plus fast
 * arrays math
 *
 * Full description
 *
 * This supports mathematical (and Boolean) functions on arrays and other python
 * objects.  Math on large arrays of basic C types is rather efficient.
 *
 * Travis E. Oliphant  2005, 2006 oliphant@ee.byu.edu (oliphant.travis@ieee.org)
 * Brigham Young University
 *
 * based on the
 *
 * Original Implementation:
 * Copyright (c) 1995, 1996, 1997 Jim Hugunin, hugunin@mit.edu
 *
 * with inspiration and code from
 * Numarray
 * Space Science Telescope Institute
 * J. Todd Miller
 * Perry Greenfield
 * Rick White
 *
 */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#define _UMATHMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <stddef.h>

#include "npy_config.h"
#include "npy_pycompat.h"
#include "npy_argparse.h"

#include "numpy/arrayobject.h"
#include "numpy/ufuncobject.h"
#include "numpy/arrayscalars.h"
#include "lowlevel_strided_loops.h"
#include "ufunc_type_resolution.h"
#include "reduction.h"
#include "mem_overlap.h"
#include "npy_hashtable.h"
#include "conversion_utils.h"

#include "ufunc_object.h"
#include "override.h"
#include "npy_import.h"
#include "extobj.h"

#include "arrayobject.h"
#include "arraywrap.h"
#include "common.h"
#include "ctors.h"
#include "dtypemeta.h"
#include "numpyos.h"
#include "dispatching.h"
#include "convert_datatype.h"
#include "legacy_array_method.h"
#include "abstractdtypes.h"
#include "mapping.h"
#include "npy_static_data.h"
#include "multiarraymodule.h"

/* TODO: Only for `NpyIter_GetTransferFlags` until it is public */
#define NPY_ITERATOR_IMPLEMENTATION_CODE
#include "nditer_impl.h"

/********** PRINTF DEBUG TRACING **************/
#define NPY_UF_DBG_TRACING 0

#if NPY_UF_DBG_TRACING
#define NPY_UF_DBG_PRINT(s) {printf("%s", s);fflush(stdout);}
#define NPY_UF_DBG_PRINT1(s, p1) {printf((s), (p1));fflush(stdout);}
#define NPY_UF_DBG_PRINT2(s, p1, p2) {printf(s, p1, p2);fflush(stdout);}
#define NPY_UF_DBG_PRINT3(s, p1, p2, p3) {printf(s, p1, p2, p3);fflush(stdout);}
#else
#define NPY_UF_DBG_PRINT(s)
#define NPY_UF_DBG_PRINT1(s, p1)
#define NPY_UF_DBG_PRINT2(s, p1, p2)
#define NPY_UF_DBG_PRINT3(s, p1, p2, p3)
#endif
/**********************************************/

typedef struct {
    PyObject *in;   /* The input arguments to the ufunc, a tuple */
    PyObject *out;  /* The output arguments, a tuple. If no non-None outputs are
                       provided, then this is NULL. */
} ufunc_full_args;


/* ---------------------------------------------------------------- */

static PyObject *
prepare_input_arguments_for_outer(PyObject *args, PyUFuncObject *ufunc);

static int
resolve_descriptors(int nop,
        PyUFuncObject *ufunc, PyArrayMethodObject *ufuncimpl,
        PyArrayObject *operands[], PyArray_Descr *dtypes[],
        PyArray_DTypeMeta *signature[], PyObject *inputs_tup,
        NPY_CASTING casting);


/*UFUNC_API*/
NPY_NO_EXPORT int


/*
 * Python Universal Functions Object -- Math for all types, plus fast
 * arrays math
 *
 * Full description
 *
 * This supports mathematical (and Boolean) functions on arrays and other python
 * objects.  Math on large arrays of basic C types is rather efficient.
 *
 * Travis E. Oliphant  2005, 2006 oliphant@ee.byu.edu (oliphant.travis@ieee.org)
 * Brigham Young University
 *
 * based on the
 *
 * Original Implementation:
 * Copyright (c) 1995, 1996, 1997 Jim Hugunin, hugunin@mit.edu
 *
 * with inspiration and code from
 * Numarray
 * Space Science Telescope Institute
 * J. Todd Miller
 * Perry Greenfield
 * Rick White
 *
 */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#define _UMATHMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <stddef.h>

#include "npy_config.h"
#include "npy_pycompat.h"
#include "npy_argparse.h"

#include "numpy/arrayobject.h"
#include "numpy/ufuncobject.h"
#include "numpy/arrayscalars.h"
#include "lowlevel_strided_loops.h"
#include "ufunc_type_resolution.h"
#include "reduction.h"
#include "mem_overlap.h"
#include "npy_hashtable.h"
#include "conversion_utils.h"

#include "ufunc_object.h"
#include "override.h"
#include "npy_import.h"
#include "extobj.h"

#include "arrayobject.h"
#include "arraywrap.h"
#include "common.h"
#include "ctors.h"
#include "dtypemeta.h"
#include "numpyos.h"
#include "dispatching.h"
#include "convert_datatype.h"
#include "legacy_array_method.h"
#include "abstractdtypes.h"
#include "mapping.h"
#include "npy_static_data.h"
#include "multiarraymodule.h"

/* TODO: Only for `NpyIter_GetTransferFlags` until it is public */
#define NPY_ITERATOR_IMPLEMENTATION_CODE
#include "nditer_impl.h"

/********** PRINTF DEBUG TRACING **************/
#define NPY_UF_DBG_TRACING 0

#if NPY_UF_DBG_TRACING
#define NPY_UF_DBG_PRINT(s) {printf("%s", s);fflush(stdout);}
#define NPY_UF_DBG_PRINT1(s, p1) {printf((s), (p1));fflush(stdout);}
#define NPY_UF_DBG_PRINT2(s, p1, p2) {printf(s, p1, p2);fflush(stdout);}
#define NPY_UF_DBG_PRINT3(s, p1, p2, p3) {printf(s, p1, p2, p3);fflush(stdout);}
#else
#define NPY_UF_DBG_PRINT(s)
#define NPY_UF_DBG_PRINT1(s, p1)
#define NPY_UF_DBG_PRINT2(s, p1, p2)
#define NPY_UF_DBG_PRINT3(s, p1, p2, p3)
#endif
/**********************************************/

typedef struct {
    PyObject *in;   /* The input arguments to the ufunc, a tuple */
    PyObject *out;  /* The output arguments, a tuple. If no non-None outputs are
                       provided, then this is NULL. */
} ufunc_full_args;


/* ---------------------------------------------------------------- */

static PyObject *
prepare_input_arguments_for_outer(PyObject *args, PyUFuncObject *ufunc);
/*
 * Prepare input arguments for outer product computation using the given
 * `args` tuple and `ufunc` object.
 */

static int
resolve_descriptors(int nop,
        PyUFuncObject *ufunc, PyArrayMethodObject *ufuncimpl,
        PyArrayObject *operands[], PyArray_Descr *dtypes[],
        PyArray_DTypeMeta *signature[], PyObject *inputs_tup,
        NPY_CASTING casting);
/*
 * Resolve the descriptors for the ufunc operation with `nop` number of operands,
 * using `ufunc`, `ufuncimpl`, `operands`, `dtypes`, `signature`, `inputs_tup`, and
 * `casting`.
 */


/*UFUNC_API*/
NPY_NO_EXPORT int
/*
 * 获取浮点错误状态，此函数在1.9版之前不会清除浮点错误状态，
 * 保留清除操作以防第三方代码依赖于状态清除。
 */
PyUFunc_getfperr(void)
{
    /*
     * non-clearing get was only added in 1.9 so this function always cleared
     * keep it so just in case third party code relied on the clearing
     */
    char param = 0;
    return npy_clear_floatstatus_barrier(&param);
}


/* 检查状态标志会清除它 */
/*UFUNC_API*/
NPY_NO_EXPORT void
PyUFunc_clearfperr()
{
    char param = 0;
    npy_clear_floatstatus_barrier(&param);
}


#define NPY_UFUNC_DEFAULT_INPUT_FLAGS \
    NPY_ITER_READONLY | \
    NPY_ITER_ALIGNED | \
    NPY_ITER_OVERLAP_ASSUME_ELEMENTWISE

#define NPY_UFUNC_DEFAULT_OUTPUT_FLAGS \
    NPY_ITER_ALIGNED | \
    NPY_ITER_ALLOCATE | \
    NPY_ITER_NO_BROADCAST | \
    NPY_ITER_NO_SUBTYPE | \
    NPY_ITER_OVERLAP_ASSUME_ELEMENTWISE

/* 在模块初始化时调用，设置矩阵乘积（matmul）ufunc的输出标志 */
NPY_NO_EXPORT int
set_matmul_flags(PyObject *d)
{
    PyObject *matmul = NULL;
    int result = PyDict_GetItemStringRef(d, "matmul", &matmul);
    if (result <= 0) {
        // 如果错误未被调用者设置，则返回错误
        return -1;
    }
    /*
     * 默认的输出标志 NPY_ITER_OVERLAP_ASSUME_ELEMENTWISE 允许完全重叠的输入和输出（原地操作）。
     * 尽管对于常见的数学操作是正确的，但在一般情况下以及特别是矩阵乘积的情况下，这种假设是不正确的。
     *
     * NPY_ITER_UPDATEIFCOPY 在 PyUFunc_GeneralizedFunction 中默认添加，
     * 这是为具有签名的广义ufunc调用的变体。
     *
     * 启用 NPY_ITER_WRITEONLY 可以在某些情况下避免复制操作。
     */
    ((PyUFuncObject *)matmul)->op_flags[2] = (NPY_ITER_WRITEONLY |
                                         NPY_ITER_UPDATEIFCOPY |
                                         NPY_UFUNC_DEFAULT_OUTPUT_FLAGS) &
                                         ~NPY_ITER_OVERLAP_ASSUME_ELEMENTWISE;
    Py_DECREF(matmul);
    return 0;
}


/*
 * 根据所需的输入或输出标志设置每个操作数的标志。
 * 对于输入（由ufunc->nin确定），op_flags[i]将与op_in_flags合并，
 * 可能会覆盖之前阶段设置的每个操作数的标志。
 * 对于输出（由ufunc->nout确定），op_flags[i]将仅在之前未设置的情况下设置为op_out_flags。
 * 输入标志行为保留向后兼容性，而输出标志行为则是最大灵活性的“正确”行为。
 */
NPY_NO_EXPORT void
_ufunc_setup_flags(PyUFuncObject *ufunc, npy_uint32 op_in_flags,
                   npy_uint32 op_out_flags, npy_uint32 *op_flags)
{
    int nin = ufunc->nin;
    int nout = ufunc->nout;
    int nop = nin + nout, i;
    /* 设置标志 */
}
    # 遍历输入操作数的范围 [0, nin)
    for (i = 0; i < nin; ++i) {
        # 将当前操作数的标志设置为 ufunc 操作标志和输入操作标志的按位或结果
        op_flags[i] = ufunc->op_flags[i] | op_in_flags;
        
        /*
         * 如果当前操作数被设置为 READWRITE 或者 WRITEONLY，
         * 则清除默认的 READONLY 标志
         */
        if (op_flags[i] & (NPY_ITER_READWRITE | NPY_ITER_WRITEONLY)) {
            op_flags[i] &= ~NPY_ITER_READONLY;
        }
    }
    
    # 遍历剩余的操作数范围 [nin, nop)
    for (i = nin; i < nop; ++i) {
        # 如果 ufunc 的操作标志 ufunc->op_flags[i] 存在，则使用它；否则使用输出操作标志
        op_flags[i] = ufunc->op_flags[i] ? ufunc->op_flags[i] : op_out_flags;
    }
/*
 * Return the position of next non-white-space char in the string
 */
static int
_next_non_white_space(const char* str, int offset)
{
    int ret = offset;
    // 循环直到找到非空格非制表符的字符
    while (str[ret] == ' ' || str[ret] == '\t') {
        ret++;
    }
    return ret;
}

/*
 * Check if the character is an alphabetic character or underscore
 */
static int
_is_alpha_underscore(char ch)
{
    // 检查字符是否为字母或者下划线
    return (ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z') || ch == '_';
}

/*
 * Check if the character is alphanumeric or underscore
 */
static int
_is_alnum_underscore(char ch)
{
    // 检查字符是否为字母、数字或下划线
    return _is_alpha_underscore(ch) || (ch >= '0' && ch <= '9');
}

/*
 * Convert a string into a number
 */
static npy_intp
_get_size(const char* str)
{
    char *stop;
    // 将字符串转换为长整型数值
    npy_longlong size = NumPyOS_strtoll(str, &stop, 10);

    if (stop == str || _is_alpha_underscore(*stop)) {
        /* not a well formed number */
        // 如果转换失败或者包含非法字符，返回-1
        return -1;
    }
    if (size >= NPY_MAX_INTP || size <= NPY_MIN_INTP) {
        /* len(str) too long */
        // 如果数值超出范围，返回-1
        return -1;
    }
    return size;
}

/*
 * Return the ending position of a variable name including optional modifier
 */
static int
_get_end_of_name(const char* str, int offset)
{
    int ret = offset;
    // 找到变量名结束位置，包括可选的修饰符
    while (_is_alnum_underscore(str[ret])) {
        ret++;
    }
    if (str[ret] == '?') {
        ret ++;
    }
    return ret;
}

/*
 * Returns 1 if the dimension names pointed by s1 and s2 are the same,
 * otherwise returns 0.
 */
static int
_is_same_name(const char* s1, const char* s2)
{
    // 检查两个字符串指向的维度名是否相同
    while (_is_alnum_underscore(*s1) && _is_alnum_underscore(*s2)) {
        if (*s1 != *s2) {
            return 0;
        }
        s1++;
        s2++;
    }
    // 如果两个字符串都结束于字母数字字符之后，返回1；否则返回0
    return !_is_alnum_underscore(*s1) && !_is_alnum_underscore(*s2);
}

/*
 * Sets the following fields in the PyUFuncObject 'ufunc':
 *
 * Field             Type                     Array Length
 * core_enabled      int (effectively bool)   N/A
 * core_num_dim_ix   int                      N/A
 * core_dim_flags    npy_uint32 *             core_num_dim_ix
 * core_dim_sizes    npy_intp *               core_num_dim_ix
 * core_num_dims     int *                    nargs (i.e. nin+nout)
 * core_offsets      int *                    nargs
 * core_dim_ixs      int *                    sum(core_num_dims)
 * core_signature    char *                   strlen(signature) + 1
 *
 * The function assumes that the values that are arrays have not
 * been set already, and sets these pointers to memory allocated
 * with PyArray_malloc.  These are freed when the ufunc dealloc
 * method is called.
 *
 * Returns 0 unless an error occurred.
 */
static int
_parse_signature(PyUFuncObject *ufunc, const char *signature)
{
    size_t len;
    char const **var_names;
    int nd = 0;             /* number of dimension of the current argument */
    int cur_arg = 0;        /* index into core_num_dims&core_offsets */
    int cur_core_dim = 0;   /* index into core_dim_ixs */
    int i = 0;
    char *parse_error = NULL;

    if (signature == NULL) {
        PyErr_SetString(PyExc_RuntimeError,
                        "_parse_signature with NULL signature");
        return -1;
    }
    // 继续添加注释...
}
    # 计算签名字符串的长度
    len = strlen(signature);
    # 为核心函数签名分配足够的内存空间，包括字符串末尾的空字符 '\0'
    ufunc->core_signature = PyArray_malloc(sizeof(char) * (len+1));
    if (ufunc->core_signature) {
        # 将签名字符串复制到分配的内存空间中
        strcpy(ufunc->core_signature, signature);
    }
    /* 分配足够的内存空间来存储所有维度名称的指针 */
    var_names = PyArray_malloc(sizeof(char const*) * len);
    if (var_names == NULL) {
        # 内存分配失败，引发内存错误异常并返回-1
        PyErr_NoMemory();
        return -1;
    }

    ufunc->core_enabled = 1;
    ufunc->core_num_dim_ix = 0;
    # 为核心函数参数的维度数分配内存空间
    ufunc->core_num_dims = PyArray_malloc(sizeof(int) * ufunc->nargs);
    ufunc->core_offsets = PyArray_malloc(sizeof(int) * ufunc->nargs);
    /* 下面三个项目稍后将被缩减 */
    # 为维度索引、维度大小和维度标志分别分配内存空间
    ufunc->core_dim_ixs = PyArray_malloc(sizeof(int) * len);
    ufunc->core_dim_sizes = PyArray_malloc(sizeof(npy_intp) * len);
    ufunc->core_dim_flags = PyArray_malloc(sizeof(npy_uint32) * len);

    if (ufunc->core_num_dims == NULL || ufunc->core_dim_ixs == NULL ||
        ufunc->core_offsets == NULL ||
        ufunc->core_dim_sizes == NULL ||
        ufunc->core_dim_flags == NULL) {
        # 如果内存分配失败，引发内存错误异常并跳转到失败处理标签
        PyErr_NoMemory();
        goto fail;
    }
    for (size_t j = 0; j < len; j++) {
        # 初始化维度标志数组为0
        ufunc->core_dim_flags[j] = 0;
    }

    i = _next_non_white_space(signature, 0);
    /* 如果当前参数数量不等于ufunc的总参数数量，则出现解析错误，跳转到失败处理标签 */
    if (cur_arg != ufunc->nargs) {
        parse_error = "incomplete signature: not all arguments found";
        goto fail;
    }
    # 重新分配核心维度索引、维度大小和维度标志的内存空间
    ufunc->core_dim_ixs = PyArray_realloc(ufunc->core_dim_ixs,
            sizeof(int) * cur_core_dim);
    ufunc->core_dim_sizes = PyArray_realloc(
            ufunc->core_dim_sizes,
            sizeof(npy_intp) * ufunc->core_num_dim_ix);
    ufunc->core_dim_flags = PyArray_realloc(
            ufunc->core_dim_flags,
            sizeof(npy_uint32) * ufunc->core_num_dim_ix);

    /* 检查是否是简单的核心签名，例如 "(),()->()" */
    if (cur_core_dim == 0) {
        # 如果核心维度数为0，则禁用核心函数
        ufunc->core_enabled = 0;
    }
    # 释放变量名称的内存空间
    PyArray_free((void*)var_names);
    # 返回0，表示函数成功执行
    return 0;
fail:
    // 释放变量名数组的内存
    PyArray_free((void*)var_names);
    // 如果存在解析错误，设置一个带有详细位置信息的 ValueError 异常
    if (parse_error) {
        PyErr_Format(PyExc_ValueError,
                     "%s at position %d in \"%s\"",
                     parse_error, i, signature);
    }
    // 返回 -1 表示函数执行失败
    return -1;
}

/*
 * 检查 'obj' 是否是 ufunc 的有效输出数组，即它要么是 None 或可写的数组，
 * 增加它的引用计数并将指针存储在 'store' 中。成功返回 0，失败设置异常并返回 -1。
 */
static int
_set_out_array(PyObject *obj, PyArrayObject **store)
{
    // 如果 obj 是 None，则将其转换为 NULL
    if (obj == Py_None) {
        return 0;
    }
    // 如果 obj 是一个数组
    if (PyArray_Check(obj)) {
        // 如果数组不可写，则返回 -1 表示失败
        if (PyArray_FailUnlessWriteable((PyArrayObject *)obj,
                                        "output array") < 0) {
            return -1;
        }
        // 增加对象的引用计数
        Py_INCREF(obj);
        // 将 obj 强制转换为 PyArrayObject，并存储在 store 中
        *store = (PyArrayObject *)obj;

        return 0;
    }
    // 如果 obj 不是有效的数组类型，设置类型错误异常
    PyErr_SetString(PyExc_TypeError, "return arrays must be of ArrayType");

    return -1;
}

/********* GENERIC UFUNC USING ITERATOR *********/

/*
 * 为 ufunc 生成一个名称（如果名称未设置）
 * 在 PyUFunc_handlefperr 机制中使用，在错误消息中也使用
 */
NPY_NO_EXPORT const char*
ufunc_get_name_cstr(PyUFuncObject *ufunc) {
    // 返回 ufunc 的名称，如果没有设置，则返回 "<unnamed ufunc>"
    return ufunc->name ? ufunc->name : "<unnamed ufunc>";
}


/*
 * 用于解析关键字参数的转换器。
 */
static int
_subok_converter(PyObject *obj, npy_bool *subok)
{
    // 如果 obj 是布尔类型
    if (PyBool_Check(obj)) {
        // 将布尔值转换为 npy_bool 类型，并存储在 subok 中
        *subok = (obj == Py_True);
        return NPY_SUCCEED;
    }
    else {
        // 如果 obj 不是布尔类型，设置类型错误异常
        PyErr_SetString(PyExc_TypeError,
                        "'subok' must be a boolean");
        return NPY_FAIL;
    }
}

static int
_keepdims_converter(PyObject *obj, int *keepdims)
{
    // 如果 obj 是布尔类型
    if (PyBool_Check(obj)) {
        // 将布尔值转换为整数，并存储在 keepdims 中
        *keepdims = (obj == Py_True);
        return NPY_SUCCEED;
    }
    else {
        // 如果 obj 不是布尔类型，设置类型错误异常
        PyErr_SetString(PyExc_TypeError,
                        "'keepdims' must be a boolean");
        return NPY_FAIL;
    }
}

static int
_wheremask_converter(PyObject *obj, PyArrayObject **wheremask)
{
    /*
     * 优化：where=True 等同于没有 where 参数。
     * 这让我们将 True 作为默认值。
     */
    // 如果 obj 是 True
    if (obj == Py_True) {
        return NPY_SUCCEED;
    }
    else {
        // 创建一个描述符为 NPY_BOOL 的数组，用于 wheremask
        PyArray_Descr *dtype = PyArray_DescrFromType(NPY_BOOL);
        if (dtype == NULL) {
            return NPY_FAIL;
        }
        // 创建一个 PyArrayObject 对象，使用 obj 来转换为数组，使用 dtype 描述符
        // PyArray_FromAny 即使失败也会将 dtype 的引用计数交给 *wheremask
        *wheremask = (PyArrayObject *)PyArray_FromAny(obj, dtype, 0, 0, 0, NULL);
        if ((*wheremask) == NULL) {
            return NPY_FAIL;
        }
        return NPY_SUCCEED;
    }
}
/*
 * 由于数组重写，仅在此步骤中执行实际的参数转换。
 * 此函数接收引用对象并将它们解析为所需的值。
 * 此函数在错误发生时进行清理并将引用置空，
 * 但调用者必须确保 `out_op[0:nargs]` 和 `out_wheremask` 被 NULL 初始化。
 */
static int
convert_ufunc_arguments(PyUFuncObject *ufunc,
        ufunc_full_args full_args, PyArrayObject *out_op[],
        PyArray_DTypeMeta *out_op_DTypes[],
        npy_bool *force_legacy_promotion, npy_bool *allow_legacy_promotion,
        npy_bool *promoting_pyscalars,
        PyObject *order_obj, NPY_ORDER *out_order,
        PyObject *casting_obj, NPY_CASTING *out_casting,
        PyObject *subok_obj, npy_bool *out_subok,
        PyObject *where_obj, PyArrayObject **out_wheremask, /* PyArray of bool */
        PyObject *keepdims_obj, int *out_keepdims)
{
    int nin = ufunc->nin;
    int nout = ufunc->nout;
    int nop = ufunc->nargs;
    PyObject *obj;

    /* 转换并填充输入参数 */
    npy_bool all_scalar = NPY_TRUE;
    npy_bool any_scalar = NPY_FALSE;
    *allow_legacy_promotion = NPY_TRUE;
    *force_legacy_promotion = NPY_FALSE;
    *promoting_pyscalars = NPY_FALSE;

    /* 如果允许遗留升级且存在非全标量和任意标量的情况，确定是否应使用最小标量 */
    if (*allow_legacy_promotion && (!all_scalar && any_scalar)) {
        *force_legacy_promotion = should_use_min_scalar(nin, out_op, 0, NULL);
    }

    /* 转换并填充输出参数 */
    memset(out_op_DTypes + nin, 0, nout * sizeof(*out_op_DTypes));
    if (full_args.out != NULL) {
        for (int i = 0; i < nout; i++) {
            obj = PyTuple_GET_ITEM(full_args.out, i);
            /* 设置输出数组对象 */
            if (_set_out_array(obj, out_op + i + nin) < 0) {
                goto fail;
            }
            /* 获取并增加输出数据类型的引用计数 */
            if (out_op[i] != NULL) {
                out_op_DTypes[i + nin] = NPY_DTYPE(PyArray_DESCR(out_op[i]));
                Py_INCREF(out_op_DTypes[i + nin]);
            }
        }
    }

    /*
     * 大多数参数在此手动转换，因为先解析为对象更容易处理 ufunc 覆盖。
     */
    
    /* 如果存在 where_obj，将其转换为 PyArrayObject 类型的 out_wheremask */
    if (where_obj && !_wheremask_converter(where_obj, out_wheremask)) {
        goto fail;
    }

    /* 如果存在 keepdims_obj，将其转换为整数类型的 out_keepdims */
    if (keepdims_obj && !_keepdims_converter(keepdims_obj, out_keepdims)) {
        goto fail;
    }

    /* 如果存在 casting_obj，将其转换为 NPY_CASTING 类型的 out_casting */
    if (casting_obj && !PyArray_CastingConverter(casting_obj, out_casting)) {
        goto fail;
    }

    /* 如果存在 order_obj，将其转换为 NPY_ORDER 类型的 out_order */
    if (order_obj && !PyArray_OrderConverter(order_obj, out_order)) {
        goto fail;
    }

    /* 如果存在 subok_obj，将其转换为布尔值类型的 out_subok */
    if (subok_obj && !_subok_converter(subok_obj, out_subok)) {
        goto fail;
    }

    /* 成功转换，返回 0 */
    return 0;

fail:
    /* 失败处理：清理和释放资源 */
    if (out_wheremask != NULL) {
        Py_XSETREF(*out_wheremask, NULL);
    }
    for (int i = 0; i < nop; i++) {
        Py_XSETREF(out_op[i], NULL);
    }
    return -1;
}
/*
 * This function checks whether a trivial loop can be used for a given
 * ufunc implementation and its operands, potentially making copies of
 * scalar and one-dimensional operands to facilitate this.
 *
 * Returns:
 *   1  - if a trivial loop is feasible
 *   0  - if a trivial loop is not feasible
 *  -1  - if an error occurs during the checks
 */
static int
check_for_trivial_loop(PyArrayMethodObject *ufuncimpl,
                       PyArrayObject **op, PyArray_Descr **dtypes,
                       NPY_CASTING casting, npy_intp buffersize)
{
    int force_cast_input = ufuncimpl->flags & _NPY_METH_FORCE_CAST_INPUTS;
    int i, nin = ufuncimpl->nin, nop = nin + ufuncimpl->nout;

    // Loop through all operands (inputs and outputs)
    for (i = 0; i < nop; ++i) {
        /*
         * If the operand pointer is NULL, skip to the next operand.
         * This typically happens for outputs that are not yet allocated.
         */
        if (op[i] == NULL) {
            continue;
        }

        // Check if the operand is not aligned
        int must_copy = !PyArray_ISALIGNED(op[i]);

        // Check if the data types do not match
        if (dtypes[i] != PyArray_DESCR(op[i])) {
            npy_intp view_offset;
            // Check if casting from op[i]'s dtype to dtypes[i] is safe
            npy_intp is_safe = PyArray_SafeCast(PyArray_DESCR(op[i]), dtypes[i], &view_offset, casting, 0);
            if (is_safe < 0 && PyErr_Occurred()) {
                /* A proper error during a cast check, should be rare */
                return -1;
            }
            // Check if there is a non-zero view offset
            if (view_offset != 0) {
                /* NOTE: Could possibly implement non-zero view offsets */
                must_copy = 1;
            }

            // Check if force casting input is enabled and if the operand is an input
            if (force_cast_input && i < nin) {
                /*
                 * ArrayMethod flagged to ignore casting (logical funcs
                 * can force cast to bool)
                 */
            }
            else if (is_safe != 1) {
                return 0;  /* there was a cast error or cast is not safe enough */
            }
        }

        // If must_copy is true, make a copy of the operand
        if (must_copy) {
            /*
             * If op[i] is a scalar or a small one-dimensional array input,
             * make a copy to keep the opportunity for a trivial loop.
             * Outputs are not copied here.
             */
            if (i < nin && (PyArray_NDIM(op[i]) == 0
                            || (PyArray_NDIM(op[i]) == 1
                                && PyArray_DIM(op[i], 0) <= buffersize))) {
                PyArrayObject *tmp;
                Py_INCREF(dtypes[i]);
                // Create a copy of op[i] with dtype dtypes[i]
                tmp = (PyArrayObject *)PyArray_CastToType(op[i], dtypes[i], 0);
                if (tmp == NULL) {
                    return -1;
                }
                // Replace op[i] with the copied array
                Py_DECREF(op[i]);
                op[i] = tmp;
            }
            else {
                return 0;
            }
        }
    }

    return 1;
}
/*
 * Check whether a trivial loop is possible and call the innerloop if it is.
 * A trivial loop is defined as one where a single strided inner-loop call
 * is possible.
 *
 * This function only supports a single output (due to the overlap check).
 * It always accepts 0-D arrays and will broadcast them.  The function
 * cannot broadcast any other array (as it requires a single stride).
 * The function accepts all 1-D arrays, and N-D arrays that are either all
 * C- or all F-contiguous.
 * NOTE: Broadcast outputs are implicitly rejected in the overlap detection.
 *
 * Returns -2 if a trivial loop is not possible, 0 on success and -1 on error.
 */
static int
try_trivial_single_output_loop(PyArrayMethod_Context *context,
        PyArrayObject *op[], NPY_ORDER order,
        int errormask)
{
    // 获取输入操作数的数量
    int nin = context->method->nin;
    // 获取总操作数的数量（包括输出）
    int nop = nin + 1;
    // 确保只有一个输出数组
    assert(context->method->nout == 1);

    /* The order of all N-D contiguous operands, can be fixed by `order` */
    // 操作的数组顺序和内存布局（是否连续）由 `order` 参数确定
    int operation_order = 0;
    if (order == NPY_CORDER) {
        operation_order = NPY_ARRAY_C_CONTIGUOUS; // C 连续布局
    }
    else if (order == NPY_FORTRANORDER) {
        operation_order = NPY_ARRAY_F_CONTIGUOUS; // Fortran 连续布局
    }

    // 初始化操作的维度数量
    int operation_ndim = 0;
    // 初始化操作的形状数组
    npy_intp *operation_shape = NULL;
    // 初始化固定步幅数组，最多支持 NPY_MAXARGS 个操作数
    npy_intp fixed_strides[NPY_MAXARGS];
    for (int iop = 0; iop < nop; iop++) {
        if (op[iop] == NULL) {
            /* 输出参数可能为 NULL（只有一个）；稍后填充 */
            assert(iop == nin);
            continue;
        }

        int op_ndim = PyArray_NDIM(op[iop]);

        /* 处理 0 维的特殊情况，因为可以使用 0 步幅进行广播 */
        if (op_ndim == 0 && iop < nin) {
            fixed_strides[iop] = 0;
            continue;
        }

        /* 第一个非 0 维的操作：固定维度和形状（顺序稍后固定） */
        if (operation_ndim == 0) {
            operation_ndim = op_ndim;
            operation_shape = PyArray_SHAPE(op[iop]);
        }
        else if (op_ndim != operation_ndim) {
            return -2;  /* 维度不匹配（除非是 0 维输入操作） */
        }
        else if (!PyArray_CompareLists(
                operation_shape, PyArray_DIMS(op[iop]), op_ndim)) {
            return -2;  /* 形状不匹配 */
        }

        if (op_ndim == 1) {
            fixed_strides[iop] = PyArray_STRIDES(op[iop])[0];
        }
        else {
            fixed_strides[iop] = PyArray_ITEMSIZE(op[iop]);  /* 连续的 */

            /* 此操作必须与操作顺序匹配（并且是连续的） */
            int op_order = (PyArray_FLAGS(op[iop]) &
                            (NPY_ARRAY_C_CONTIGUOUS|NPY_ARRAY_F_CONTIGUOUS));
            if (op_order == 0) {
                return -2;  /* N 维操作必须是连续的 */
            }
            else if (operation_order == 0) {
                operation_order = op_order;  /* 操作固定顺序 */
            }
            else if (operation_order != op_order) {
                return -2;
            }
        }
    }

    if (op[nin] == NULL) {
        Py_INCREF(context->descriptors[nin]);
        op[nin] = (PyArrayObject *) PyArray_NewFromDescr(&PyArray_Type,
                context->descriptors[nin], operation_ndim, operation_shape,
                NULL, NULL, operation_order==NPY_ARRAY_F_CONTIGUOUS, NULL);
        if (op[nin] == NULL) {
            return -1;
        }
        fixed_strides[nin] = context->descriptors[nin]->elsize;
    }
    else {
        /* 如果任何输入与输出重叠，我们使用完整路径。 */
        for (int iop = 0; iop < nin; iop++) {
            if (!PyArray_EQUIVALENTLY_ITERABLE_OVERLAP_OK(
                    op[iop], op[nin],
                    PyArray_TRIVIALLY_ITERABLE_OP_READ,
                    PyArray_TRIVIALLY_ITERABLE_OP_NOREAD)) {
                return -2;
            }
        }
        /* 检查自重叠（非 1 维的是连续的，完全重叠是可以的） */
        if (operation_ndim == 1 &&
                PyArray_STRIDES(op[nin])[0] < PyArray_ITEMSIZE(op[nin]) &&
                PyArray_STRIDES(op[nin])[0] != 0) {
            return -2;
        }
    }

    /*
     * 我们可以使用简单的优化（单个内部循环调用），`fixed_strides` 包含了那次调用的步幅。
     */
    char *data[NPY_MAXARGS];
    // 计算操作形状的总元素个数
    npy_intp count = PyArray_MultiplyList(operation_shape, operation_ndim);
    // 如果计数为0，则无需执行任何操作，直接返回0
    if (count == 0) {
        /* Nothing to do */
        return 0;
    }
    // 定义线程状态变量
    NPY_BEGIN_THREADS_DEF;

    // 定义跨步循环方法及辅助数据
    PyArrayMethod_StridedLoop *strided_loop;
    NpyAuxData *auxdata = NULL;
    NPY_ARRAYMETHOD_FLAGS flags = 0;
    // 获取跨步循环方法，若失败则返回-1
    if (context->method->get_strided_loop(context,
            1, 0, fixed_strides,
            &strided_loop, &auxdata, &flags) < 0) {
        return -1;
    }
    // 将操作数组的数据指针存储在data数组中
    for (int iop=0; iop < nop; iop++) {
        data[iop] = PyArray_BYTES(op[iop]);
    }

    // 如果方法标志不包含 NPY_METH_NO_FLOATINGPOINT_ERRORS，则清除浮点错误状态
    if (!(flags & NPY_METH_NO_FLOATINGPOINT_ERRORS)) {
        npy_clear_floatstatus_barrier((char *)context);
    }
    // 如果方法标志不包含 NPY_METH_REQUIRES_PYAPI，则开启线程
    if (!(flags & NPY_METH_REQUIRES_PYAPI)) {
        NPY_BEGIN_THREADS_THRESHOLDED(count);
    }

    // 调用跨步循环方法执行操作，并获取返回值
    int res = strided_loop(context, data, &count, fixed_strides, auxdata);

    // 结束线程状态管理
    NPY_END_THREADS;
    // 释放辅助数据
    NPY_AUXDATA_FREE(auxdata);

    /*
     * 如果已经设置了异常（PyErr_Occurred()返回真），则将结果置为-1，
     * 这对于旧式ufunc（如`power`释放了GIL但手动设置异常）可能不是严格正确的。
     */
    if (PyErr_Occurred()) {
        res = -1;
    }

    // 如果操作成功且方法标志不包含 NPY_METH_NO_FLOATINGPOINT_ERRORS，则检查浮点错误
    if (res == 0 && !(flags & NPY_METH_NO_FLOATINGPOINT_ERRORS)) {
        /* 注意：即使 `res < 0` 时也可以检查浮点错误 */
        const char *name = ufunc_get_name_cstr((PyUFuncObject *)context->caller);
        res = _check_ufunc_fperr(errormask, name);
    }
    // 返回执行结果
    return res;
/*
 * Check casting: It would be nice to just move this into the iterator
 * or pass in the full cast information.  But this can special case
 * the logical functions and prints a better error message.
 */
static inline int
validate_casting(PyArrayMethodObject *method, PyUFuncObject *ufunc,
        PyArrayObject *ops[], PyArray_Descr *const descriptors_const[],
        NPY_CASTING casting)
{
    /* Cast away const to not change old public `PyUFunc_ValidateCasting`. */
    PyArray_Descr **descriptors = (PyArray_Descr **)descriptors_const;
    // 如果使用了旧版的类型解析函数，直接返回成功，无需进一步验证
    if (method->resolve_descriptors == &wrapped_legacy_resolve_descriptors) {
        return 0;
    }
    // 如果设置了强制类型转换输入的标志，调用验证输出类型转换函数
    if (method->flags & _NPY_METH_FORCE_CAST_INPUTS) {
        if (PyUFunc_ValidateOutCasting(ufunc, casting, ops, descriptors) < 0) {
            return -1;
        }
    }
    // 否则调用一般类型转换验证函数
    else {
        if (PyUFunc_ValidateCasting(ufunc, casting, ops, descriptors) < 0) {
            return -1;
        }
    }
    // 返回成功标志
    return 0;
}


/*
 * The ufunc loop implementation for both normal ufunc calls and masked calls
 * when the iterator has to be used.
 *
 * See `PyUFunc_GenericFunctionInternal` for more information (where this is
 * called from).
 */
static int
execute_ufunc_loop(PyArrayMethod_Context *context, int masked,
        PyArrayObject **op, NPY_ORDER order, npy_intp buffersize,
        NPY_CASTING casting,
        npy_uint32 *op_flags, int errormask)
{
    PyUFuncObject *ufunc = (PyUFuncObject *)context->caller;
    int nin = context->method->nin, nout = context->method->nout;
    int nop = nin + nout;

    // 验证类型转换是否有效，如果无效则直接返回错误
    if (validate_casting(context->method,
            ufunc, op, context->descriptors, casting) < 0) {
        return -1;
    }

    // 如果是带掩码的操作
    if (masked) {
        assert(PyArray_TYPE(op[nop]) == NPY_BOOL);

        /*
         * NOTE: In the masked version, we consider the output read-write,
         *       this gives a best-effort of preserving the input, but does
         *       not always work.  It could allow the operand to be copied
         *       due to copy-if-overlap, but only if it was passed in.
         */
        // 对于从第 nin 个操作数开始的所有输出操作数标记为读写权限，如果操作数不为 NULL
        for (int i = nin; i < nop; ++i) {
            op_flags[i] |= (op[i] != NULL ? NPY_ITER_READWRITE : NPY_ITER_WRITEONLY);
        }
        // 最后一个操作数（掩码）标记为只读数组掩码
        op_flags[nop] = NPY_ITER_READONLY | NPY_ITER_ARRAYMASK;  /* mask */
    }

    // 打印调试信息，表示正在创建迭代器
    NPY_UF_DBG_PRINT("Making iterator\n");

    // 设置迭代器的标志，包括外部循环、允许引用、允许零大小、缓冲区化、内部增长、延迟缓冲区分配、重叠时复制等
    npy_uint32 iter_flags = ufunc->iter_flags |
                 NPY_ITER_EXTERNAL_LOOP |
                 NPY_ITER_REFS_OK |
                 NPY_ITER_ZEROSIZE_OK |
                 NPY_ITER_BUFFERED |
                 NPY_ITER_GROWINNER |
                 NPY_ITER_DELAY_BUFALLOC |
                 NPY_ITER_COPY_IF_OVERLAP;
    /*
     * 分配迭代器。因为输入的类型已经被检查过，我们使用 'unsafe' 强制转换规则，这样计算速度更快。
     */
    NpyIter *iter = NpyIter_AdvancedNew(nop + masked, op,
                        iter_flags,
                        order, NPY_UNSAFE_CASTING,
                        op_flags, (PyArray_Descr **)context->descriptors,
                        -1, NULL, NULL, buffersize);
    if (iter == NULL) {
        return -1;
    }

    NPY_UF_DBG_PRINT("Made iterator\n");

    /* 将新分配的数组设置为输出 */
    PyArrayObject **op_it = NpyIter_GetOperandArray(iter);
    for (int i = 0; i < nout; ++i) {
        if (op[nin + i] == NULL) {
            op[nin + i] = op_it[nin + i];
            Py_INCREF(op[nin + i]);
        }
    }

    /* 只有在迭代大小非零时才执行循环 */
    npy_intp full_size = NpyIter_GetIterSize(iter);
    if (full_size == 0) {
        if (!NpyIter_Deallocate(iter)) {
            return -1;
        }
        return 0;
    }

    /*
     * 获取内部循环，根据固定步长进行特化。
     */
    PyArrayMethod_StridedLoop *strided_loop;
    NpyAuxData *auxdata;
    npy_intp fixed_strides[NPY_MAXARGS];

    NpyIter_GetInnerFixedStrideArray(iter, fixed_strides);
    NPY_ARRAYMETHOD_FLAGS flags = 0;
    if (masked) {
        if (PyArrayMethod_GetMaskedStridedLoop(context,
                1, fixed_strides, &strided_loop, &auxdata, &flags) < 0) {
            NpyIter_Deallocate(iter);
            return -1;
        }
    }
    else {
        if (context->method->get_strided_loop(context,
                1, 0, fixed_strides, &strided_loop, &auxdata, &flags) < 0) {
            NpyIter_Deallocate(iter);
            return -1;
        }
    }

    /* 获取循环所需的变量 */
    NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter, NULL);
    if (iternext == NULL) {
        NPY_AUXDATA_FREE(auxdata);
        NpyIter_Deallocate(iter);
        return -1;
    }
    char **dataptr = NpyIter_GetDataPtrArray(iter);
    npy_intp *strides = NpyIter_GetInnerStrideArray(iter);
    npy_intp *countptr = NpyIter_GetInnerLoopSizePtr(iter);

    NPY_BEGIN_THREADS_DEF;

    flags = PyArrayMethod_COMBINED_FLAGS(flags, NpyIter_GetTransferFlags(iter));

    if (!(flags & NPY_METH_NO_FLOATINGPOINT_ERRORS)) {
        npy_clear_floatstatus_barrier((char *)context);
    }
    if (!(flags & NPY_METH_REQUIRES_PYAPI)) {
        NPY_BEGIN_THREADS_THRESHOLDED(full_size);
    }

    /* 重置迭代器，可能会复制第一个缓冲区块，可能引起浮点错误 */
    if (NpyIter_Reset(iter, NULL) != NPY_SUCCEED) {
        NPY_AUXDATA_FREE(auxdata);
        NpyIter_Deallocate(iter);
        return -1;
    }

    NPY_UF_DBG_PRINT("Actual inner loop:\n");
    /* 执行循环 */
    int res;
    do {
        // 打印迭代器循环计数，使用调试宏，格式化输出
        NPY_UF_DBG_PRINT1("iterator loop count %d\n", (int)*countptr);
        // 调用 strided_loop 函数进行迭代处理
        res = strided_loop(context, dataptr, countptr, strides, auxdata);
    } while (res == 0 && iternext(iter));

    // 结束线程并等待
    NPY_END_THREADS;
    // 释放辅助数据
    NPY_AUXDATA_FREE(auxdata);

    // 如果结果为 0 并且未设置 NPY_METH_NO_FLOATINGPOINT_ERRORS 标志
    if (res == 0 && !(flags & NPY_METH_NO_FLOATINGPOINT_ERRORS)) {
        /* NOTE: We could check float errors even when `res < 0` */
        // 获取当前 ufunc 的名称
        const char *name = ufunc_get_name_cstr((PyUFuncObject *)context->caller);
        // 检查浮点错误
        res = _check_ufunc_fperr(errormask, name);
    }

    // 如果迭代器未成功释放
    if (!NpyIter_Deallocate(iter)) {
        // 返回错误码 -1
        return -1;
    }
    // 返回处理结果 res
    return res;
/*
 * Validate that operands have enough dimensions, accounting for
 * possible flexible dimensions that may be absent.
 */
static int
_validate_num_dims(PyUFuncObject *ufunc, PyArrayObject **op,
                   npy_uint32 *core_dim_flags,
                   int *op_core_num_dims) {
    int i, j;
    int nin = ufunc->nin;    // 获取 ufunc 的输入数量
    int nop = ufunc->nargs;  // 获取 ufunc 的操作数数量

    for (i = 0; i < nop; i++) {  // 遍历所有操作数
        if (op[i] != NULL) {    // 检查操作数是否为空
            int op_ndim = PyArray_NDIM(op[i]);  // 获取操作数的维度数

            if (op_ndim < op_core_num_dims[i]) {  // 如果操作数的维度少于要求的核心维度数
                int core_offset = ufunc->core_offsets[i];  // 获取核心偏移量
                /* We've too few, but some dimensions might be flexible */
                for (j = core_offset;
                     j < core_offset + ufunc->core_num_dims[i]; j++) {
                    int core_dim_index = ufunc->core_dim_ixs[j];  // 获取核心维度的索引
                    if ((core_dim_flags[core_dim_index] &
                         UFUNC_CORE_DIM_CAN_IGNORE)) {
                        int i1, j1, k;
                        /*
                         * Found a dimension that can be ignored. Flag that
                         * it is missing, and unflag that it can be ignored,
                         * since we are doing so already.
                         */
                        core_dim_flags[core_dim_index] |= UFUNC_CORE_DIM_MISSING;  // 将核心维度标记为缺失
                        core_dim_flags[core_dim_index] ^= UFUNC_CORE_DIM_CAN_IGNORE;  // 取消核心维度可忽略的标记
                        /*
                         * Reduce the number of core dimensions for all
                         * operands that use this one (including ours),
                         * and check whether we're now OK.
                         */
                        for (i1 = 0, k=0; i1 < nop; i1++) {
                            for (j1 = 0; j1 < ufunc->core_num_dims[i1]; j1++) {
                                if (ufunc->core_dim_ixs[k++] == core_dim_index) {
                                    op_core_num_dims[i1]--;  // 减少所有使用此核心维度的操作数的核心维度数量
                                }
                            }
                        }
                        if (op_ndim == op_core_num_dims[i]) {  // 如果操作数的维度数与核心维度要求数相等，则跳出循环
                            break;
                        }
                    }
                }
                if (op_ndim < op_core_num_dims[i]) {
                    PyErr_Format(PyExc_ValueError,
                         "%s: %s operand %d does not have enough "
                         "dimensions (has %d, gufunc core with "
                         "signature %s requires %d)",
                         ufunc_get_name_cstr(ufunc),
                         i < nin ? "Input" : "Output",
                         i < nin ? i : i - nin, PyArray_NDIM(op[i]),
                         ufunc->core_signature, op_core_num_dims[i]);
                    return -1;  // 报错并返回 -1
                }
            }
        }
    }
    return 0;  // 操作数维度验证通过，返回 0
}

/*
 * Check whether any of the outputs of a gufunc has core dimensions.
 */
static int
_has_output_coredims(PyUFuncObject *ufunc) {
    int i;
    // 遍历所有输出
    for (i = ufunc->nin; i < ufunc->nargs; i++) {
        // 如果输出的核心维度数量大于 0，则返回 1 表示有核心维度
        if (ufunc->core_num_dims[i] > 0) {
            return 1;
        }
    }
    // 所有输出的核心维度数量都为 0，则返回 0 表示没有核心维度
    return 0;
}
    # 遍历从输入参数到输出参数的索引范围
    for (i = ufunc->nin; i < ufunc->nin + ufunc->nout; ++i) {
        # 检查当前参数的核心维度数是否大于零
        if (ufunc->core_num_dims[i] > 0) {
            # 如果找到任意一个参数的核心维度大于零，返回1表示真
            return 1;
        }
    }
    # 如果所有参数的核心维度均为零，则返回0表示假
    return 0;
/*
 * Check whether the gufunc can be used with axis, i.e., that there is only
 * a single, shared core dimension (which means that operands either have
 * that dimension, or have no core dimensions).  Returns 0 if all is fine,
 * and sets an error and returns -1 if not.
 */
static int
_check_axis_support(PyUFuncObject *ufunc) {
    // 检查核心维度的索引是否不为1，如果不是，则抛出类型错误并返回-1
    if (ufunc->core_num_dim_ix != 1) {
        PyErr_Format(PyExc_TypeError,
                     "%s: axis can only be used with a single shared core "
                     "dimension, not with the %d distinct ones implied by "
                     "signature %s.",
                     ufunc_get_name_cstr(ufunc),
                     ufunc->core_num_dim_ix,
                     ufunc->core_signature);
        return -1;
    }
    // 返回0表示一切正常
    return 0;
}

/*
 * Check whether the gufunc can be used with keepdims, i.e., that all its
 * input arguments have the same number of core dimension, and all output
 * arguments have no core dimensions. Returns 0 if all is fine, and sets
 * an error and returns -1 if not.
 */
static int
_check_keepdims_support(PyUFuncObject *ufunc) {
    int i;
    int nin = ufunc->nin, nout = ufunc->nout;
    int input_core_dims = ufunc->core_num_dims[0];
    // 遍历所有输入和输出参数，检查核心维度是否满足条件
    for (i = 1; i < nin + nout; i++) {
        if (ufunc->core_num_dims[i] != (i < nin ? input_core_dims : 0)) {
            PyErr_Format(PyExc_TypeError,
                "%s does not support keepdims: its signature %s requires "
                "%s %d to have %d core dimensions, but keepdims can only "
                "be used when all inputs have the same number of core "
                "dimensions and all outputs have no core dimensions.",
                ufunc_get_name_cstr(ufunc),
                ufunc->core_signature,
                i < nin ? "input" : "output",
                i < nin ? i : i - nin,
                ufunc->core_num_dims[i]);
            return -1;
        }
    }
    // 返回0表示一切正常
    return 0;
}

/*
 * Interpret a possible axes keyword argument, using it to fill the remap_axis
 * array which maps default to actual axes for each operand, indexed as
 * as remap_axis[iop][iaxis]. The default axis order has first all broadcast
 * axes and then the core axes the gufunc operates on.
 *
 * Returns 0 on success, and -1 on failure
 */
static int
_parse_axes_arg(PyUFuncObject *ufunc, int op_core_num_dims[], PyObject *axes,
                PyArrayObject **op, int broadcast_ndim, int **remap_axis) {
    int nin = ufunc->nin;
    int nop = ufunc->nargs;
    int iop, list_size;

    // 检查 axes 是否为列表类型，如果不是则抛出类型错误并返回-1
    if (!PyList_Check(axes)) {
        PyErr_SetString(PyExc_TypeError, "axes should be a list.");
        return -1;
    }
    // 获取列表的大小
    list_size = PyList_Size(axes);
    # 检查列表大小是否与操作数数量相等，若不相等则执行以下条件语句
    if (list_size != nop) {
        # 若列表大小不等于输入数量或者有输出核心维度，则执行以下条件语句
        if (list_size != nin || _has_output_coredims(ufunc)) {
            # 抛出值错误异常，指出轴应该是一个包含所有输入和输出的条目的列表；
            # 如果没有输出具有核心轴，则可以省略输出的条目。
            PyErr_Format(PyExc_ValueError,
                         "axes should be a list with an entry for all "
                         "%d inputs and outputs; entries for outputs can only "
                         "be omitted if none of them has core axes.",
                         nop);
            # 返回-1表示出现错误
            return -1;
        }
        # 对于超出输入数目的每个输出，将其重新映射的轴设置为NULL
        for (iop = nin; iop < nop; iop++) {
            remap_axis[iop] = NULL;
        }
    }
    # 结束对操作数循环的注释
    } /* end of for(iop) loop over operands */
    # 返回0表示成功执行
    return 0;
/*
 * Simplified version of the above, using axis to fill the remap_axis
 * array, which maps default to actual axes for each operand, indexed as
 * as remap_axis[iop][iaxis]. The default axis order has first all broadcast
 * axes and then the core axes the gufunc operates on.
 *
 * Returns 0 on success, and -1 on failure
 */
static int
_parse_axis_arg(PyUFuncObject *ufunc, const int core_num_dims[], PyObject *axis,
                PyArrayObject **op, int broadcast_ndim, int **remap_axis) {
    int nop = ufunc->nargs;  /* Number of operands for the universal function */
    int iop, axis_int;       /* Loop counter for operands and integer axis */

    /* Convert the Python object 'axis' to an integer */
    axis_int = PyArray_PyIntAsInt(axis);
    if (error_converting(axis_int)) {  /* Check if conversion failed */
        return -1;  /* Return -1 on failure */
    }

    /* Loop over each operand */
    for (iop = 0; iop < nop; ++iop) {
        int axis, op_ndim, op_axis;

        /* Ensure core_num_dims is 0 or 1 for the current operand */
        if (core_num_dims[iop] == 0) {
            remap_axis[iop] = NULL;  /* Set remap_axis to NULL if no core dimensions */
            continue;
        }

        /* Determine the number of dimensions of the current operand */
        if (op[iop]) {
            op_ndim = PyArray_NDIM(op[iop]);
        }
        else {
            op_ndim = broadcast_ndim + 1;  /* Set op_ndim for broadcast dimensions */
        }

        op_axis = axis_int;  /* Ensure axis_int remains unchanged */
        
        /* Check and adjust the axis value within the valid range */
        if (check_and_adjust_axis(&op_axis, op_ndim) < 0) {
            return -1;  /* Return -1 on failure from check_and_adjust_axis */
        }

        /* If the axis is the last axis, no remapping is needed */
        if (op_axis == op_ndim - 1) {
            remap_axis[iop] = NULL;
            continue;
        }

        /* Map the default to actual axes using remap_axis array */
        remap_axis[iop][op_ndim - 1] = op_axis;  /* Set the last axis */
        
        /* Fill remap_axis for axes before and after op_axis */
        for (axis = 0; axis < op_axis; axis++) {
            remap_axis[iop][axis] = axis;
        }
        for (axis = op_axis; axis < op_ndim - 1; axis++) {
            remap_axis[iop][axis] = axis + 1;
        }
    } /* end of for(iop) loop over operands */

    return 0;  /* Return 0 on success */
}

/*
 * Validate the core dimensions of all the operands, and collect all of
 * the labelled core dimensions into 'core_dim_sizes'.
 *
 * Returns 0 on success, and -1 on failure
 *
 * The behavior has been changed in NumPy 1.16.0, and the following
 * requirements must be fulfilled or an error will be raised:
 *  * Arguments, both input and output, must have at least as many
 *    dimensions as the corresponding number of core dimensions. In
 *    versions before 1.10, 1's were prepended to the shape as needed.
 *  * Core dimensions with same labels must have exactly matching sizes.
 *    In versions before 1.10, core dimensions of size 1 would broadcast
 *    against other core dimensions with the same label.
 *  * All core dimensions must have their size specified by a passed in
 *    input or output argument. In versions before 1.10, core dimensions in
 *    an output argument that were not specified in an input argument,
 *    and whose size could not be inferred from a passed in output
 *    argument, would have their size set to 1.
 *  * Core dimensions may be fixed, new in NumPy 1.16
 */
static int
_get_coredim_sizes(PyUFuncObject *ufunc, PyArrayObject **op,
                   const int *op_core_num_dims, npy_uint32 *core_dim_flags,
                   npy_intp *core_dim_sizes, int **remap_axis) {
    int i;
    int nin = ufunc->nin;                      // 获取ufunc的输入数量
    int nout = ufunc->nout;                    // 获取ufunc的输出数量
    int nop = nin + nout;                      // 计算操作数总数

    for (i = 0; i < nop; ++i) {                // 循环处理每个操作数
        if (op[i] != NULL) {                   // 检查操作数是否存在
            int idim;
            int dim_offset = ufunc->core_offsets[i];  // 获取核心偏移量
            int core_start_dim = PyArray_NDIM(op[i]) - op_core_num_dims[i];  // 计算起始核心维度
            int dim_delta = 0;

            /* checked before this routine gets called */
            assert(core_start_dim >= 0);        // 断言核心起始维度非负

            /*
             * Make sure every core dimension exactly matches all other core
             * dimensions with the same label. Note that flexible dimensions
             * may have been removed at this point, if so, they are marked
             * with UFUNC_CORE_DIM_MISSING.
             */
            for (idim = 0; idim < ufunc->core_num_dims[i]; ++idim) {
                int core_index = dim_offset + idim;  // 计算核心索引
                int core_dim_index = ufunc->core_dim_ixs[core_index];  // 获取核心维度索引
                npy_intp core_dim_size = core_dim_sizes[core_dim_index];  // 获取核心维度大小
                npy_intp op_dim_size;

                /* can only happen if flexible; dimension missing altogether */
                if (core_dim_flags[core_dim_index] & UFUNC_CORE_DIM_MISSING) {
                    op_dim_size = 1;
                    dim_delta++;  // 对于索引在维度中的调整
                }
                else {
                    op_dim_size = PyArray_DIM(op[i],
                             REMAP_AXIS(i, core_start_dim + idim - dim_delta));  // 获取操作数的维度大小
                }
                if (core_dim_sizes[core_dim_index] < 0) {
                    core_dim_sizes[core_dim_index] = op_dim_size;  // 更新核心维度大小
                }
                else if (op_dim_size != core_dim_size) {  // 如果维度大小不匹配则报错
                    PyErr_Format(PyExc_ValueError,
                            "%s: %s operand %d has a mismatch in its "
                            "core dimension %d, with gufunc "
                            "signature %s (size %zd is different "
                            "from %zd)",
                            ufunc_get_name_cstr(ufunc), i < nin ? "Input" : "Output",
                            i < nin ? i : i - nin, idim - dim_delta,
                            ufunc->core_signature, op_dim_size,
                            core_dim_sizes[core_dim_index]);
                    return -1;  // 返回错误码
                }
            }
        }
    }

    /*
     * Make sure no core dimension is unspecified.
     */
    for (i = nin; i < nop; ++i) {
        // 迭代处理从 nin 到 nop 的操作数索引
        int idim;
        // 获取当前操作数的偏移量
        int dim_offset = ufunc->core_offsets[i];

        for (idim = 0; idim < ufunc->core_num_dims[i]; ++idim) {
            // 获取核心维度索引
            int core_dim_index = ufunc->core_dim_ixs[dim_offset + idim];

            /* 检查所有尚未设置尺寸的情况 */
            if (core_dim_sizes[core_dim_index] < 0) {
                /*
                 * 噢，这个维度从未被指定过
                 * （只有在没有输出操作数的情况下才会发生）
                 */
                PyErr_Format(PyExc_ValueError,
                        "%s: 输出操作数 %d 的核心维度 %d 未指定，使用的gufunc签名为 %s",
                        ufunc_get_name_cstr(ufunc), i - nin, idim,
                        ufunc->core_signature);
                // 返回错误代码 -1
                return -1;
            }
        }
    }

    // 执行成功，返回代码 0
    return 0;
/*
 * 返回一个新的 ufunc 标识符的引用。注意，这个标识符仅仅是存储在 ufunc 上的
 * 默认标识值，实际的标识值由 ufunc 循环（ArrayMethod）查询得到。
 *
 * TODO: 将一个引用存储在 ufunc 对象本身，而不是每次构造时都创建一个新的
 */
NPY_NO_EXPORT PyObject *
PyUFunc_GetDefaultIdentity(PyUFuncObject *ufunc, npy_bool *reorderable)
{
    switch(ufunc->identity) {
    case PyUFunc_One:
        *reorderable = 1;
        return PyLong_FromLong(1);

    case PyUFunc_Zero:
        *reorderable = 1;
        return PyLong_FromLong(0);

    case PyUFunc_MinusOne:
        *reorderable = 1;
        return PyLong_FromLong(-1);

    case PyUFunc_ReorderableNone:
        *reorderable = 1;
        Py_RETURN_NONE;

    case PyUFunc_None:
        *reorderable = 0;
        Py_RETURN_NONE;

    case PyUFunc_IdentityValue:
        *reorderable = 1;
        Py_INCREF(ufunc->identity_value);
        return ufunc->identity_value;

    default:
        PyErr_Format(PyExc_ValueError,
                "ufunc %s has an invalid identity", ufunc_get_name_cstr(ufunc));
        return NULL;
    }
}

/*
 * 复制 ufunc 结构中可能在执行过程中需要改变的部分。成功时返回 0；否则返回 -1。
 */
static int
_initialize_variable_parts(PyUFuncObject *ufunc,
                           int op_core_num_dims[],
                           npy_intp core_dim_sizes[],
                           npy_uint32 core_dim_flags[]) {
    int i;

    for (i = 0; i < ufunc->nargs; i++) {
        op_core_num_dims[i] = ufunc->core_num_dims[i];
    }
    for (i = 0; i < ufunc->core_num_dim_ix; i++) {
        core_dim_sizes[i] = ufunc->core_dim_sizes[i];
        core_dim_flags[i] = ufunc->core_dim_flags[i];
    }
    return 0;
}

/*
 * 内部通用函数实现，用于处理广义的 ufunc。
 */
static int
PyUFunc_GeneralizedFunctionInternal(PyUFuncObject *ufunc,
        PyArrayMethodObject *ufuncimpl, PyArray_Descr *operation_descrs[],
        PyArrayObject *op[], NPY_CASTING casting, NPY_ORDER order,
        PyObject *axis, PyObject *axes, int keepdims)
{
    int nin, nout;
    int i, j, idim, nop;
    const char *ufunc_name;
    int retval;
    int needs_api = 0;

    /* Use remapped axes for generalized ufunc */
    int broadcast_ndim, iter_ndim;
    int op_core_num_dims[NPY_MAXARGS];
    int op_axes_arrays[NPY_MAXARGS][NPY_MAXDIMS];
    int *op_axes[NPY_MAXARGS];
    npy_uint32 core_dim_flags[NPY_MAXARGS];

    npy_uint32 op_flags[NPY_MAXARGS];
    npy_intp iter_shape[NPY_MAXARGS];
    NpyIter *iter = NULL;
    npy_uint32 iter_flags;
    npy_intp total_problem_size;

    /* These parameters come from a TLS global */
    int buffersize = 0, errormask = 0;

    /* The dimensions which get passed to the inner loop */
    npy_intp inner_dimensions[NPY_MAXDIMS+1];
    /* The strides which get passed to the inner loop */
    npy_intp *inner_strides = NULL;
    /* Auxiliary data allocated by the ufuncimpl (ArrayMethod) */
}
    // 定义一个指向 NpyAuxData 结构的指针，初始为 NULL
    NpyAuxData *auxdata = NULL;

    /* 核心维度的大小（# 条目等于 ufunc->core_num_dim_ix）*/
    // core_dim_sizes 指向 inner_dimensions + 1 的地址
    npy_intp *core_dim_sizes = inner_dimensions + 1;
    // core_dim_ixs_size 的值尚未初始化
    int core_dim_ixs_size;
    /* 轴的交换 */
    // remap_axis_memory 初始为 NULL
    int *remap_axis_memory = NULL;
    // remap_axis 初始为 NULL
    int **remap_axis = NULL;

    // 输入、输出、操作数的数量
    nin = ufunc->nin;
    nout = ufunc->nout;
    nop = nin + nout;

    // 获取 ufunc 的名称为字符串
    ufunc_name = ufunc_get_name_cstr(ufunc);

    // 打印调试信息，评估 ufunc 的名称
    NPY_UF_DBG_PRINT1("\nEvaluating ufunc %s\n", ufunc_name);

    // 验证操作的类型转换是否有效
    if (validate_casting(ufuncimpl,
            ufunc, op, operation_descrs, casting) < 0) {
        // 如果验证失败则返回 -1
        return -1;
    }

    /* 初始化可能变化的部分，使用 ufunc 中的值 */
    retval = _initialize_variable_parts(ufunc, op_core_num_dims,
                                        core_dim_sizes, core_dim_flags);
    // 如果初始化失败则跳转到失败标签
    if (retval < 0) {
        goto fail;
    }

    /*
     * 如果 keepdims 被传入（因此从初始值更改），检查 gufunc 是否适用，
     * 即其输入共享相同数量的核心维度，其输出没有。
     */
    if (keepdims != -1) {
        // 检查是否支持 keepdims
        retval = _check_keepdims_support(ufunc);
        if (retval < 0) {
            goto fail;
        }
    }
    // 如果 axis 不为 NULL，检查是否支持 axis
    if (axis != NULL) {
        retval = _check_axis_support(ufunc);
        if (retval < 0) {
            goto fail;
        }
    }
    /*
     * 如果 keepdims 被设置为 true，表示所有输入维度相同，
     * 则说明所有输出维度也将相同。
     */
    if (keepdims == 1) {
        // 将输出的核心维度调整为与输入相同
        int num_dims = op_core_num_dims[0];
        for (i = nin; i < nop; ++i) {
            op_core_num_dims[i] = num_dims;
        }
    }
    else {
        /* keepdims 未设置或为 false，不需要调整 */
        keepdims = 0;
    }
    /*
     * 检查操作数是否具有所需的最小维度。
     * （只检查核心维度；广播维度由迭代器测试。）
     */
    retval = _validate_num_dims(ufunc, op, core_dim_flags,
                                op_core_num_dims);
    // 如果维度验证失败则跳转到失败标签
    if (retval < 0) {
        goto fail;
    }
    /*
     * 计算迭代维度的数量，这是所有非核心维度的广播结果。
     * （如果给定的话，我们允许输出广播输入，这与普通的 ufunc 行为一致。）
     */
    broadcast_ndim = 0;
    for (i = 0; i < nop; ++i) {
        if (op[i] == NULL) {
            continue;
        }
        // 计算非核心维度的数量
        int n = PyArray_NDIM(op[i]) - op_core_num_dims[i];
        // 更新广播维度的最大值
        if (n > broadcast_ndim) {
            broadcast_ndim = n;
        }
    }

    /* 可能需要重新映射轴。 */
    # 检查 axes 和 axis 是否至少有一个不为 NULL
    if (axes != NULL || axis != NULL) {
        # 使用断言确保 axes 和 axis 不同时为非 NULL
        assert(!(axes != NULL && axis != NULL));

        # 分配 remap_axis 和 remap_axis_memory 的内存空间
        remap_axis = PyArray_malloc(sizeof(remap_axis[0]) * nop);
        remap_axis_memory = PyArray_malloc(sizeof(remap_axis_memory[0]) *
                                                  nop * NPY_MAXDIMS);
        # 检查内存分配是否成功
        if (remap_axis == NULL || remap_axis_memory == NULL) {
            # 内存分配失败，触发内存错误异常
            PyErr_NoMemory();
            goto fail;
        }
        # 将 remap_axis_memory 划分成 nop 个 NPY_MAXDIMS 维度的数组
        for (i=0; i < nop; i++) {
            remap_axis[i] = remap_axis_memory + i * NPY_MAXDIMS;
        }
        # 根据是否存在 axis 参数来调用相应的解析函数
        if (axis) {
            retval = _parse_axis_arg(ufunc, op_core_num_dims, axis, op,
                                     broadcast_ndim, remap_axis);
        }
        else {
            retval = _parse_axes_arg(ufunc, op_core_num_dims, axes, op,
                                     broadcast_ndim, remap_axis);
        }
        # 检查解析函数的返回值，小于 0 则跳转到失败处理部分
        if(retval < 0) {
            goto fail;
        }
    }

    /* 收集标记核心维度的长度 */
    retval = _get_coredim_sizes(ufunc, op, op_core_num_dims, core_dim_flags,
                                core_dim_sizes, remap_axis);
    # 检查收集函数的返回值，小于 0 则跳转到失败处理部分
    if(retval < 0) {
        goto fail;
    }
    /*
     * 计算迭代器创建的维度数量，包括广播维度和所有输出的核心维度，
     * 以便迭代器可以按照 order='F' 的规则分配这些输出维度。
     */
    iter_ndim = broadcast_ndim;
    for (i = nin; i < nop; ++i) {
        iter_ndim += op_core_num_dims[i];
    }
    # 检查迭代器创建的维度数量是否超过 NPY_MAXDIMS，超过则触发值错误异常
    if (iter_ndim > NPY_MAXDIMS) {
        PyErr_Format(PyExc_ValueError,
                    "too many dimensions for generalized ufunc %s",
                    ufunc_name);
        retval = -1;
        goto fail;
    }

    /* 初始化 iter_shape 数组的前部分为 -1 */
    for (idim = 0; idim < broadcast_ndim; ++idim) {
        iter_shape[idim] = -1;
    }

    /* 填充所有操作数的 op_axes */
    j = broadcast_ndim;
    for (i = 0; i < nop; ++i) {
        int n;

        if (op[i]) {
            // 计算当前操作数 op[i] 的维度差值
            n = PyArray_NDIM(op[i]) - op_core_num_dims[i];
        }
        else {
            // 如果 op[i] 为空，使用广播维度的值
            n = broadcast_ndim;
        }
        /* Broadcast all the unspecified dimensions normally */
        // 对所有未指定的维度进行广播处理
        for (idim = 0; idim < broadcast_ndim; ++idim) {
            if (idim >= broadcast_ndim - n) {
                // 如果是核心维度，则重新映射为 REMAP_AXIS 返回的值
                op_axes_arrays[i][idim] =
                    REMAP_AXIS(i, idim - (broadcast_ndim - n));
            }
            else {
                // 否则标记为 -1，表示不应用于当前操作
                op_axes_arrays[i][idim] = -1;
            }
        }

        /*
         * Any output core dimensions shape should be ignored, so we add
         * it as a Reduce dimension (which can be broadcast with the rest).
         * These will be removed before the actual iteration for gufuncs.
         */
        // 忽略任何输出的核心维度形状，因此将其添加为 Reduce 维度（可以与其余部分广播）
        // 在 gufuncs 实际迭代之前将会移除这些维度
        for (idim = broadcast_ndim; idim < iter_ndim; ++idim) {
            op_axes_arrays[i][idim] = NPY_ITER_REDUCTION_AXIS(-1);
        }

        /* Except for when it belongs to this output */
        // 除非它属于当前输出，否则执行以下操作
        if (i >= nin) {
            int dim_offset = ufunc->core_offsets[i];
            int num_removed = 0;
            /*
             * Fill in 'iter_shape' and 'op_axes' for the core dimensions
             * of this output. Here, we have to be careful: if keepdims
             * was used, then the axes are not real core dimensions, but
             * are being added back for broadcasting, so their size is 1.
             * If the axis was removed, we should skip altogether.
             */
            // 为当前输出的核心维度填充 'iter_shape' 和 'op_axes'。
            // 如果使用了 keepdims，那么轴不是真正的核心维度，而是为了广播而添加的，因此它们的大小为 1。
            // 如果轴被移除，我们应该完全跳过。
            if (keepdims) {
                for (idim = 0; idim < op_core_num_dims[i]; ++idim) {
                    iter_shape[j] = 1;
                    op_axes_arrays[i][j] = REMAP_AXIS(i, n + idim);
                    ++j;
                }
            }
            else {
                for (idim = 0; idim < ufunc->core_num_dims[i]; ++idim) {
                    int core_index = dim_offset + idim;
                    int core_dim_index = ufunc->core_dim_ixs[core_index];
                    if ((core_dim_flags[core_dim_index] &
                         UFUNC_CORE_DIM_MISSING)) {
                        /* skip it */
                        // 如果核心维度标志指示缺失，则跳过
                        num_removed++;
                        continue;
                    }
                    iter_shape[j] = core_dim_sizes[ufunc->core_dim_ixs[core_index]];
                    op_axes_arrays[i][j] = REMAP_AXIS(i, n + idim - num_removed);
                    ++j;
                }
            }
        }

        op_axes[i] = op_axes_arrays[i];
    }
#if NPY_UF_DBG_TRACING
    // 如果定义了 NPY_UF_DBG_TRACING 宏，则打印迭代器的形状信息
    printf("iter shapes:");
    for (j=0; j < iter_ndim; j++) {
        // 逐个打印迭代器的每个维度大小
        printf(" %ld", iter_shape[j]);
    }
    // 打印完所有维度大小后换行
    printf("\n");
#endif

    /* Get the buffersize and errormask */
    // 调用函数获取缓冲区大小和错误掩码
    if (_get_bufsize_errmask(&buffersize, &errormask) < 0) {
        // 如果获取失败，则设置返回值为 -1，并跳转到错误处理标签 fail
        retval = -1;
        goto fail;
    }

    // 打印调试信息，表示正在查找内部循环
    NPY_UF_DBG_PRINT("Finding inner loop\n");

    /*
     * We don't write to all elements, and the iterator may make
     * UPDATEIFCOPY temporary copies. The output arrays (unless they are
     * allocated by the iterator itself) must be considered READWRITE by the
     * iterator, so that the elements we don't write to are copied to the
     * possible temporary array.
     */
    // 设置通用函数的标志位，指定输入和输出数组的属性
    _ufunc_setup_flags(ufunc, NPY_ITER_COPY | NPY_UFUNC_DEFAULT_INPUT_FLAGS,
                       NPY_ITER_UPDATEIFCOPY |
                       NPY_ITER_WRITEONLY |
                       NPY_UFUNC_DEFAULT_OUTPUT_FLAGS,
                       op_flags);
    /*
     * Set up the iterator per-op flags.  For generalized ufuncs, we
     * can't do buffering, so must COPY or UPDATEIFCOPY.
     */
    // 设置迭代器的操作标志位，用于指定迭代器的行为特性
    iter_flags = ufunc->iter_flags |
                 NPY_ITER_MULTI_INDEX |
                 NPY_ITER_REFS_OK |
                 NPY_ITER_ZEROSIZE_OK |
                 NPY_ITER_COPY_IF_OVERLAP |
                 NPY_ITER_DELAY_BUFALLOC;

    /* Create the iterator */
    // 使用指定的参数创建高级迭代器对象
    iter = NpyIter_AdvancedNew(nop, op, iter_flags,
                           order, NPY_UNSAFE_CASTING, op_flags,
                           operation_descrs, iter_ndim,
                           op_axes, iter_shape, 0);
    if (iter == NULL) {
        // 如果创建迭代器失败，则设置返回值为 -1，并跳转到错误处理标签 fail
        retval = -1;
        goto fail;
    }

    /* Fill in any allocated outputs */
    {
        // 获取迭代器的操作数数组
        PyArrayObject **operands = NpyIter_GetOperandArray(iter);
        // 遍历操作数数组中的未初始化输出数组，并进行初始化
        for (i = nin; i < nop; ++i) {
            if (op[i] == NULL) {
                op[i] = operands[i];
                // 增加引用计数，确保正确的内存管理
                Py_INCREF(op[i]);
            }
        }
    }
    /*
     * Set up the inner strides array. Because we're not doing
     * buffering, the strides are fixed throughout the looping.
     */
    // 计算内部步长数组的大小
    core_dim_ixs_size = 0;
    for (i = 0; i < nop; ++i) {
        core_dim_ixs_size += ufunc->core_num_dims[i];
    }
    // 分配内部步长数组的内存空间
    inner_strides = (npy_intp *)PyArray_malloc(
                        NPY_SIZEOF_INTP * (nop+core_dim_ixs_size));
    if (inner_strides == NULL) {
        // 如果内存分配失败，则引发内存错误异常，并设置返回值为 -1，并跳转到错误处理标签 fail
        PyErr_NoMemory();
        retval = -1;
        goto fail;
    }
    // 复制步长信息到内部步长数组中
    /* Copy the strides after the first nop */
    idim = nop;
    for (i = 0; i < nop; ++i) {
        /*
         * 需要使用迭代器中的数组而不是 op 数组，因为可能会复制一个不同大小的类型。
         */
        PyArrayObject *arr = NpyIter_GetOperandArray(iter)[i];
        // 获取当前数组的形状（shape）
        npy_intp *shape = PyArray_SHAPE(arr);
        // 获取当前数组的步长（strides）
        npy_intp *strides = PyArray_STRIDES(arr);
        /*
         * 如果使用了灵活的维度，则可能为负数，但对于 keepdims，因为这些维度在 arr 中分配。
         */
        int core_start_dim = PyArray_NDIM(arr) - op_core_num_dims[i];
        int num_removed = 0;
        int dim_offset = ufunc->core_offsets[i];

        for (j = 0; j < ufunc->core_num_dims[i]; ++j) {
            int core_dim_index = ufunc->core_dim_ixs[dim_offset + j];
            /*
             * 当形状为 1 时（通常是缺失的维度），强制步长为 0，以便广播功能正常工作。
             */
            if (core_dim_flags[core_dim_index] & UFUNC_CORE_DIM_MISSING) {
                num_removed++;
                inner_strides[idim++] = 0;
            }
            else {
                int remapped_axis = REMAP_AXIS(i, core_start_dim + j - num_removed);
                if (shape[remapped_axis] != 1) {
                    inner_strides[idim++] = strides[remapped_axis];
                } else {
                    inner_strides[idim++] = 0;
                }
            }
        }
    }

    total_problem_size = NpyIter_GetIterSize(iter);
    if (total_problem_size < 0) {
        /*
         * 仅用于线程处理，如果为负数（表示在轴移除之前超出了 ssize_t），则假定实际问题足够大以便有用地使用线程。
         */
        total_problem_size = 1000;
    }

    /* 从迭代器中移除所有核心输出维度 */
    for (i = broadcast_ndim; i < iter_ndim; ++i) {
        if (NpyIter_RemoveAxis(iter, broadcast_ndim) != NPY_SUCCEED) {
            retval = -1;
            goto fail;
        }
    }
    if (NpyIter_RemoveMultiIndex(iter) != NPY_SUCCEED) {
        retval = -1;
        goto fail;
    }
    if (NpyIter_EnableExternalLoop(iter) != NPY_SUCCEED) {
        retval = -1;
        goto fail;
    }

    /*
     * 前 nop 个步长是内部循环的步长（但只有在移除核心轴之后才能复制它们）。
     * 如果迭代器不是缓冲的，步长将不会改变（它们实际上是固定的）。
     * 支持缓冲可能是有意义的，但可能必须在内部循环本身完成（而不是迭代器）。
     */
    assert(!NpyIter_IsBuffered(iter));
    memcpy(inner_strides, NpyIter_GetInnerStrideArray(iter),
                                    NPY_SIZEOF_INTP * nop);

    /* 最后准备数组方法调用 */
    PyArrayMethod_Context context = {
        .caller = (PyObject *)ufunc,
        .method = ufuncimpl,
        .descriptors = operation_descrs,
    };

    // 指向函数指针结构体的指针
    PyArrayMethod_StridedLoop *strided_loop;
    // 数组方法的标志，初始化为0
    NPY_ARRAYMETHOD_FLAGS flags = 0;

    // 获取循环执行函数指针及相关数据
    if (ufuncimpl->get_strided_loop(&context, 1, 0, inner_strides,
            &strided_loop, &auxdata, &flags) < 0) {
        // 如果获取失败则跳转到错误处理标签
        goto fail;
    }

    // 检查是否需要 Python API
    needs_api = (flags & NPY_METH_REQUIRES_PYAPI) != 0;
    // 检查迭代器是否需要 Python API
    needs_api |= NpyIter_IterationNeedsAPI(iter);

    // 如果不是无浮点数错误标志
    if (!(flags & NPY_METH_NO_FLOATINGPOINT_ERRORS)) {
        /* Start with the floating-point exception flags cleared */
        // 清除浮点数异常标志
        npy_clear_floatstatus_barrier((char*)&iter);
    }

    // 调试输出：执行内部循环
    NPY_UF_DBG_PRINT("Executing inner loop\n");

    // 如果迭代器的大小不为0
    if (NpyIter_GetIterSize(iter) != 0) {
        /* Do the ufunc loop */
        // 获取迭代器的下一个函数
        NpyIter_IterNextFunc *iternext;
        // 数据指针数组
        char **dataptr;
        // 内部循环大小指针
        npy_intp *count_ptr;
        // 定义多线程开始标志
        NPY_BEGIN_THREADS_DEF;

        // 获取循环所需的变量
        iternext = NpyIter_GetIterNext(iter, NULL);
        // 如果获取失败则返回-1并跳转到错误处理标签
        if (iternext == NULL) {
            retval = -1;
            goto fail;
        }
        // 获取数据指针数组
        dataptr = NpyIter_GetDataPtrArray(iter);
        // 获取内部循环大小指针
        count_ptr = NpyIter_GetInnerLoopSizePtr(iter);

        // 如果不需要 Python API
        if (!needs_api) {
            // 多线程处理阈值
            NPY_BEGIN_THREADS_THRESHOLDED(total_problem_size);
        }

        // 执行循环
        do {
            // 设置内部维度
            inner_dimensions[0] = *count_ptr;
            // 调用strided_loop执行内部循环
            retval = strided_loop(&context,
                    dataptr, inner_dimensions, inner_strides, auxdata);
        } while (retval == 0 && iternext(iter));

        // 如果不需要 Python API 并且迭代器不需要 Python API
        if (!needs_api && !NpyIter_IterationNeedsAPI(iter)) {
            // 结束多线程
            NPY_END_THREADS;
        }
    }

    // 如果返回值为0并且不是无浮点数错误标志
    if (retval == 0 && !(flags & NPY_METH_NO_FLOATINGPOINT_ERRORS)) {
        /* NOTE: We could check float errors even when `res < 0` */
        // 检查浮点数错误
        retval = _check_ufunc_fperr(errormask, ufunc_name);
    }

    // 释放内部步长数组
    PyArray_free(inner_strides);
    // 释放辅助数据
    NPY_AUXDATA_FREE(auxdata);
    // 如果无法释放迭代器则返回-1
    if (!NpyIter_Deallocate(iter)) {
        retval = -1;
    }

    // 释放重映射轴内存
    PyArray_free(remap_axis_memory);
    // 释放重映射轴
    PyArray_free(remap_axis);

    // 调试输出：返回代码
    NPY_UF_DBG_PRINT1("Returning code %d\n", retval);

    // 返回最终结果值
    return retval;
fail:
    // 打印调试信息，指示返回失败代码的原因
    NPY_UF_DBG_PRINT1("Returning failure code %d\n", retval);
    // 释放内部步幅数组的内存
    PyArray_free(inner_strides);
    // 释放辅助数据的内存
    NPY_AUXDATA_FREE(auxdata);
    // 释放迭代器资源
    NpyIter_Deallocate(iter);
    // 释放轴重映射内存
    PyArray_free(remap_axis_memory);
    // 释放轴重映射数组
    PyArray_free(remap_axis);
    // 返回操作的结果代码
    return retval;
}


static int
PyUFunc_GenericFunctionInternal(PyUFuncObject *ufunc,
        PyArrayMethodObject *ufuncimpl, PyArray_Descr *operation_descrs[],
        PyArrayObject *op[], NPY_CASTING casting, NPY_ORDER order,
        PyArrayObject *wheremask)
{
    int nin = ufunc->nin, nout = ufunc->nout, nop = nin + nout;

    npy_intp default_op_out_flags;
    npy_uint32 op_flags[NPY_MAXARGS];

    /* These parameters come from a TLS global */
    int buffersize = 0, errormask = 0;

    // 调试打印，显示当前评估的ufunc名称
    NPY_UF_DBG_PRINT1("\nEvaluating ufunc %s\n", ufunc_get_name_cstr(ufunc));

    /* Get the buffersize and errormask */
    // 获取缓冲区大小和错误掩码
    if (_get_bufsize_errmask(&buffersize, &errormask) < 0) {
        return -1;
    }

    if (wheremask != NULL) {
        /* Set up the flags. */
        // 设置操作输出标志
        default_op_out_flags = NPY_ITER_NO_SUBTYPE |
                               NPY_ITER_WRITEMASKED |
                               NPY_UFUNC_DEFAULT_OUTPUT_FLAGS;
        // 配置ufunc的标志
        _ufunc_setup_flags(ufunc, NPY_UFUNC_DEFAULT_INPUT_FLAGS,
                           default_op_out_flags, op_flags);
    }
    else {
        /* Set up the flags. */
        // 设置操作输出标志（不包括掩码）
        default_op_out_flags = NPY_ITER_WRITEONLY |
                               NPY_UFUNC_DEFAULT_OUTPUT_FLAGS;
        // 配置ufunc的标志
        _ufunc_setup_flags(ufunc, NPY_UFUNC_DEFAULT_INPUT_FLAGS,
                           default_op_out_flags, op_flags);
    }

    /* Final preparation of the arraymethod call */
    // 准备数组方法调用的上下文
    PyArrayMethod_Context context = {
        .caller = (PyObject *)ufunc,
        .method = ufuncimpl,
        .descriptors = operation_descrs,
    };

    /* Do the ufunc loop */
    if (wheremask != NULL) {
        // 执行带掩码的内部循环
        NPY_UF_DBG_PRINT("Executing masked inner loop\n");

        if (nop + 1 > NPY_MAXARGS) {
            // 如果操作数过多（包括where参数），则返回错误
            PyErr_SetString(PyExc_ValueError,
                    "Too many operands when including where= parameter");
            return -1;
        }
        // 设置操作数数组的掩码
        op[nop] = wheremask;
        // 将掩码位置的操作描述符设为NULL
        operation_descrs[nop] = NULL;

        // 执行ufunc循环操作
        return execute_ufunc_loop(&context, 1,
                op, order, buffersize, casting,
                op_flags, errormask);
    }
    else {
        NPY_UF_DBG_PRINT("Executing normal inner loop\n");
        // 打印调试信息，表示正在执行正常的内部循环

        /*
         * This checks whether a trivial loop is ok, making copies of
         * scalar and one dimensional operands if that should help.
         */
        // 检查是否可以使用简单循环，如果可以的话，复制标量和一维操作数可能会有所帮助

        int trivial_ok = check_for_trivial_loop(ufuncimpl,
                op, operation_descrs, casting, buffersize);
        // 调用函数检查是否可以使用简单循环，并将结果存储在trivial_ok中
        if (trivial_ok < 0) {
            // 如果检查出错，返回-1
            return -1;
        }
        if (trivial_ok && context.method->nout == 1) {
            // 如果可以使用简单循环且输出只有一个，尝试在不使用迭代器的情况下处理所有操作
            int retval = try_trivial_single_output_loop(&context,
                    op, order, errormask);
            // 调用函数尝试简单单输出循环，并将结果存储在retval中
            if (retval != -2) {
                // 如果返回值不是-2，直接返回该值
                return retval;
            }
        }

        // 调用函数执行通用的ufunc循环
        return execute_ufunc_loop(&context, 0,
                op, order, buffersize, casting, op_flags, errormask);
    }
/*
 * Promote and resolve a reduction-like operation for NumPy ufuncs.
 *
 * @param ufunc The ufunc object defining the operation.
 * @param arr The input array for the operation.
 * @param out The output array or NULL if not provided. Note: NumPy interprets
 *            out to mean the same as `dtype=out.dtype` and never passes the
 *            array itself to type-resolution.
 * @param signature The DType signature, potentially set by user dtype or
 *                  special cases like "add" or "multiply". May be modified.
 * @param enforce_uniform_args If true, enforces fully uniform dtypes/descriptors
 *                             required for accumulate and reduceat.
 * @param out_descrs New references to resolved descriptors (on success).
 * @param casting The casting rule to be applied during the operation.
 * @param method The ufunc method ("reduce", "reduceat", or "accumulate").
 *
 * @returns ufuncimpl The ArrayMethod implementation to use, or NULL on error.
 */
static PyArrayMethodObject *
reducelike_promote_and_resolve(PyUFuncObject *ufunc,
        PyArrayObject *arr, PyArrayObject *out,
        PyArray_DTypeMeta *signature[3],
        npy_bool enforce_uniform_args, PyArray_Descr *out_descrs[3],
        NPY_CASTING casting, char *method)
{
     /*
      * If no dtype is specified and out is not specified, we override the
      * integer and bool dtype used for add and multiply reduction to avoid overflow.
      */
    if (signature[0] == NULL && out == NULL) {
        /*
         * For integer types — ensure at least a long is used for add and multiply
         * reduction to prevent overflow.
         */
        int typenum = PyArray_TYPE(arr);
        if ((PyTypeNum_ISBOOL(typenum) || PyTypeNum_ISINTEGER(typenum))
                && ((strcmp(ufunc->name, "add") == 0)
                    || (strcmp(ufunc->name, "multiply") == 0))) {
            if (PyTypeNum_ISBOOL(typenum)) {
                typenum = NPY_INTP;
            }
            else if ((size_t)PyArray_ITEMSIZE(arr) < sizeof(npy_intp)) {
                if (PyTypeNum_ISUNSIGNED(typenum)) {
                    typenum = NPY_UINTP;
                }
                else {
                    typenum = NPY_INTP;
                }
            }
            signature[0] = PyArray_DTypeFromTypeNum(typenum);
        }
    }
    assert(signature[2] == NULL);  /* Ensure third signature element is initially NULL */
    Py_XINCREF(signature[0]);  // Increment reference count for the first signature element
    signature[2] = signature[0];  // Set the third signature element to be the first

    /*
     * Note that the `ops` is not really correct. But legacy resolution
     * cannot quite handle the correct ops (e.g., a NULL first item if `out`
     * is NULL), so we pass `arr` instead in that case.
     */
    PyArrayObject *ops[3] = {out ? out : arr, arr, out};  // Define the array objects for operation
    /*
     * 设置一个变量，用于执行一个有风险的操作。这个操作通过依赖于全局解释锁(GIL)，
     * 对输出维度进行变异，以确保即使在需要使用传统提升的情况下，reduce-likes 函数
     * 也能在没有值基础提升的未来生效。
     * 这个操作非常危险，信任于类型解析器不会做出疯狂的行为。
     */
    npy_bool evil_ndim_mutating_hack = NPY_FALSE;
    if (out != NULL && PyArray_NDIM(out) == 0 && PyArray_NDIM(arr) != 0) {
        evil_ndim_mutating_hack = NPY_TRUE;
        ((PyArrayObject_fields *)out)->nd = 1;
    }
    
    /*
     * 如果未提供 `out`，则 `initial` 可能定义第一个数据类型（也可能定义输出类型）。
     * 这样，`np.add.reduce([1, 2, 3], initial=3.4)` 将返回一个浮点数值。在版本 1.20 中，
     * 它返回一个整数，因此应该首先引发错误或警告。
     */
    PyArray_DTypeMeta *operation_DTypes[3] = {
            NULL, NPY_DTYPE(PyArray_DESCR(arr)), NULL};
    Py_INCREF(operation_DTypes[1]);
    
    if (out != NULL) {
        operation_DTypes[0] = NPY_DTYPE(PyArray_DESCR(out));
        Py_INCREF(operation_DTypes[0]);
        operation_DTypes[2] = operation_DTypes[0];
        Py_INCREF(operation_DTypes[2]);
    }
    
    PyArrayMethodObject *ufuncimpl = promote_and_get_ufuncimpl(ufunc,
            ops, signature, operation_DTypes, NPY_FALSE, NPY_TRUE,
            NPY_FALSE, NPY_TRUE);
    if (evil_ndim_mutating_hack) {
        ((PyArrayObject_fields *)out)->nd = 0;
    }
    /* 可能会在回退中填充 DTypes，并处理错误时进行 XDECREF 操作： */
    Py_XDECREF(operation_DTypes[0]);
    Py_XDECREF(operation_DTypes[1]);
    Py_XDECREF(operation_DTypes[2]);
    if (ufuncimpl == NULL) {
        return NULL;
    }
    
    /*
     * 查找操作的正确描述符。出于历史原因，我们使用不安全的转换：
     * Ufunc 逻辑要求将所有内容转换为布尔值。但是，我们现在特别处理逻辑 Ufunc，
     * 因此在原则上转换的安全性可能被设置为默认的同种类型。
     * （尽管这应该通过弃用发生）
     */
    if (resolve_descriptors(3, ufunc, ufuncimpl,
            ops, out_descrs, signature, NULL, casting) < 0) {
        return NULL;
    }
    
    /*
     * 第一个操作数和输出应该是相同的数组，因此它们应该是相同的。对于 reduce，
     * 第二个参数可以是不同的，但对于 accumulate 和 reduceat，应检查它们是相同的。
     * 理想情况下，类型解析器确保所有都相同，但我们这里并不严格执行。
     * 否则，正确处理字节顺序更改（或元数据）需要非常小心；参见 gh-20699。
     */
    # 检查解析后的数据类型是否兼容，如果不兼容则生成类型错误异常
    if (!PyArray_EquivTypes(out_descrs[0], out_descrs[2]) || (
            enforce_uniform_args && !PyArray_EquivTypes(
                    out_descrs[0], out_descrs[1]))) {
        PyErr_Format(PyExc_TypeError,
                "the resolved dtypes are not compatible with %s.%s. "
                "Resolved (%R, %R, %R)",
                ufunc_get_name_cstr(ufunc), method,
                out_descrs[0], out_descrs[1], out_descrs[2]);
        // 跳转到异常处理标签
        goto fail;
    }

    /*
     * 确认它们等价后，强制使用用户定义的输出描述符中的第三个（应由用户定义）。这对于字符串数据类型是必要的。
     */
    // 增加第三个输出描述符的引用计数，以确保它不会被释放
    Py_INCREF(out_descrs[2]);
    // 设置第一个输出描述符指向第三个输出描述符的引用
    Py_SETREF(out_descrs[0], out_descrs[2]);

    /* TODO: This really should _not_ be unsafe casting (same above)! */
    // 如果验证类型转换失败，则跳转到异常处理标签
    if (validate_casting(ufuncimpl, ufunc, ops, out_descrs, casting) < 0) {
        goto fail;
    }

    // 返回计算实现的函数指针
    return ufuncimpl;

  fail:
    // 清理失败时生成的输出描述符数组中的每个元素
    for (int i = 0; i < 3; ++i) {
        Py_CLEAR(out_descrs[i]);
    }
    // 返回空指针，表示操作失败
    return NULL;
static int
reduce_loop(PyArrayMethod_Context *context,
        PyArrayMethod_StridedLoop *strided_loop, NpyAuxData *auxdata,
        NpyIter *iter, char **dataptrs, npy_intp const *strides,
        npy_intp const *countptr, NpyIter_IterNextFunc *iternext,
        int needs_api, npy_intp skip_first_count)
{
    int retval = 0;
    char *dataptrs_copy[4];     // 声明一个指针数组，用于存储数据指针的副本
    npy_intp strides_copy[4];   // 声明一个整型数组，用于存储步长的副本
    npy_bool masked;            // 声明一个布尔值，用于标记是否使用了掩码

    NPY_BEGIN_THREADS_DEF;      // 定义多线程开始标记

    /* 获取操作数的数量，以确定是否使用了 "where" */
    masked = (NpyIter_GetNOp(iter) == 3);

    if (!needs_api) {
        NPY_BEGIN_THREADS_THRESHOLDED(NpyIter_GetIterSize(iter));  // 根据迭代器的大小设置多线程阈值
    }

    if (skip_first_count > 0) {
        assert(!masked);  /* 当前路径对于使用掩码的情况尚不可用 */   // 断言，确保当前路径在使用掩码时不可用
        while (1) {
            npy_intp count = *countptr;

            /* 跳过任何首次访问的元素 */
            if (NpyIter_IsFirstVisit(iter, 0)) {
                if (strides[0] == 0) {
                    --count;
                    --skip_first_count;
                    dataptrs[1] += strides[1];   // 更新数据指针数组的第二个指针位置
                }
                else {
                    skip_first_count -= count;
                    count = 0;
                }
            }
            if (count > 0) {
                /* 将两个项扩展为三个，用于内部循环 */
                dataptrs_copy[0] = dataptrs[0];
                dataptrs_copy[1] = dataptrs[1];
                dataptrs_copy[2] = dataptrs[0];
                strides_copy[0] = strides[0];
                strides_copy[1] = strides[1];
                strides_copy[2] = strides[0];

                retval = strided_loop(context,
                        dataptrs_copy, &count, strides_copy, auxdata);   // 执行步进循环操作
                if (retval < 0) {
                    goto finish_loop;   // 如果返回值小于0，跳转至循环结束标记
                }
            }

            /* 推进循环，并在错误（或完成）时中止 */
            if (!iternext(iter)) {
                goto finish_loop;   // 如果迭代器无法继续推进，则跳转至循环结束标记
            }

            /* 当跳过完成时，中断并继续更快的循环 */
            if (skip_first_count == 0) {
                break;   // 如果跳过计数为0，跳出循环
            }
        }
    }

    do {
        /* 将两个项扩展为三个，用于内部循环 */
        dataptrs_copy[0] = dataptrs[0];
        dataptrs_copy[1] = dataptrs[1];
        dataptrs_copy[2] = dataptrs[0];
        strides_copy[0] = strides[0];
        strides_copy[1] = strides[1];
        strides_copy[2] = strides[0];
        if (masked) {
            dataptrs_copy[3] = dataptrs[2];
            strides_copy[3] = strides[2];
        }

        retval = strided_loop(context,
                dataptrs_copy, countptr, strides_copy, auxdata);   // 执行步进循环操作
        if (retval < 0) {
            goto finish_loop;   // 如果返回值小于0，跳转至循环结束标记
        }

    } while (iternext(iter));   // 当迭代器仍能继续推进时继续循环

finish_loop:
    NPY_END_THREADS;   // 结束多线程

    return retval;   // 返回循环操作的最终返回值
}
/*
 * The following function implements the PyUFunc_Accumulate operation, which
 * accumulates results of a universal function (ufunc) along a specified axis.
 * It sets up necessary parameters and uses an iterator for efficient looping
 * over the array elements.
 */
static PyObject *
PyUFunc_Accumulate(PyUFuncObject *ufunc, PyArrayObject *arr, PyArrayObject *out,
                   int axis, PyArray_DTypeMeta *signature[3])
{
    PyArrayObject *op[2];
    int op_axes_arrays[2][NPY_MAXDIMS];
    int *op_axes[2] = {op_axes_arrays[0], op_axes_arrays[1]};
    npy_uint32 op_flags[2];
    int idim, ndim;
    int needs_api, need_outer_iterator;
    int res = 0;
#if NPY_UF_DBG_TRACING
    const char *ufunc_name = ufunc_get_name_cstr(ufunc);
#endif

    PyArrayMethod_StridedLoop *strided_loop;
    NpyAuxData *auxdata = NULL;

    NpyIter *iter = NULL;

    /* These parameters come from a TLS global */
    int buffersize = 0, errormask = 0;

    NPY_BEGIN_THREADS_DEF;
    # 调用宏 NPY_UF_DBG_PRINT1，用于打印调试信息，格式化输出 ufunc_name 的累积评估消息。
    NPY_UF_DBG_PRINT1("\nEvaluating ufunc %s.accumulate\n", ufunc_name);
#if 0
    // 输出正在执行的操作名称及数组的数据类型描述符
    printf("Doing %s.accumulate on array with dtype :  ", ufunc_name);
    PyObject_Print((PyObject *)PyArray_DESCR(arr), stdout, 0);
    printf("\n");
#endif

// 获取缓冲区大小和错误掩码
if (_get_bufsize_errmask(&buffersize, &errormask) < 0) {
    return NULL;
}

/* 对输出对象增加一个引用，以便稍后返回 */
Py_XINCREF(out);

// 创建 PyArray_Descr 结构体数组
PyArray_Descr *descrs[3];
// 使用 reducelike_promote_and_resolve 函数进行类型提升和解析
PyArrayMethodObject *ufuncimpl = reducelike_promote_and_resolve(ufunc,
        arr, out, signature, NPY_TRUE, descrs, NPY_UNSAFE_CASTING,
        "accumulate");
if (ufuncimpl == NULL) {
    return NULL;
}

/*
 * 以下代码假设所有描述符都是可以互换的，尽管它们可能不是严格相同的（但通常应该是相同的）
 */
assert(PyArray_EquivTypes(descrs[0], descrs[1])
       && PyArray_EquivTypes(descrs[0], descrs[2]));

// 检查描述符是否是引用检查类型且不是对象类型
if (PyDataType_REFCHK(descrs[2]) && descrs[2]->type_num != NPY_OBJECT) {
    /* 这部分可以移除，但需要修复初始元素的复制问题 */
    PyErr_SetString(PyExc_TypeError,
            "accumulation currently only supports `object` dtype with "
            "references");
    goto fail;
}

// 设置 PyArrayMethod_Context 上下文对象
PyArrayMethod_Context context = {
    .caller = (PyObject *)ufunc,
    .method = ufuncimpl,
    .descriptors = descrs,
};

// 获取数组的维度
ndim = PyArray_NDIM(arr);

#if NPY_UF_DBG_TRACING
// 输出找到的内部循环的数据类型描述符
printf("Found %s.accumulate inner loop with dtype :  ", ufunc_name);
PyObject_Print((PyObject *)descrs[0], stdout, 0);
printf("\n");
#endif

/* 设置外部循环的操作轴 */
for (idim = 0; idim < ndim; ++idim) {
    op_axes_arrays[0][idim] = idim;
    op_axes_arrays[1][idim] = idim;
}

/* 外部循环每个操作数的标志 */
op_flags[0] = NPY_ITER_READWRITE |
              NPY_ITER_NO_BROADCAST |
              NPY_ITER_ALLOCATE |
              NPY_ITER_NO_SUBTYPE;
op_flags[1] = NPY_ITER_READONLY;

op[0] = out;
op[1] = arr;

// 判断是否需要外部迭代器
need_outer_iterator = (ndim > 1);
// 如果数组不对齐或者输入输出对象类型不相等，则需要外部迭代器
if (!PyArray_ISALIGNED(arr) || (out && !PyArray_ISALIGNED(out)) ||
        !PyArray_EquivTypes(descrs[1], PyArray_DESCR(arr)) ||
        (out &&
         !PyArray_EquivTypes(descrs[0], PyArray_DESCR(out)))) {
    need_outer_iterator = 1;
}
// 如果输入和输出在内存中重叠，则需要外部迭代器来判断
else if (out != NULL && solve_may_share_memory(out, arr, NPY_MAY_SHARE_BOUNDS) != 0) {
    need_outer_iterator = 1;
}
    // 如果需要外部迭代器
    if (need_outer_iterator) {
        int ndim_iter = 0;  // 初始化迭代器的维度数为0
        npy_uint32 flags = NPY_ITER_ZEROSIZE_OK |  // 允许零大小迭代
                           NPY_ITER_REFS_OK |  // 允许引用迭代
                           NPY_ITER_COPY_IF_OVERLAP;  // 如果有重叠则复制数据

        /*
         * 由于 accumulate 的设置，无法进行缓冲，
         * 因此必要时进行复制。
         */
        ndim_iter = ndim;  // 设置迭代器的维度为当前维度
        flags |= NPY_ITER_MULTI_INDEX;  // 启用多索引迭代

        /*
         * 添加更多标志。
         *
         * 积累的外部循环是对数组的“逐元素”操作，因此启用 NPY_ITER_OVERLAP_ASSUME_ELEMENTWISE。
         * 这意味着， inplace 操作 accumulate(x, out=x) 可以安全执行，无需临时复制。
         */
        op_flags[0] |= NPY_ITER_UPDATEIFCOPY | NPY_ITER_ALIGNED | NPY_ITER_OVERLAP_ASSUME_ELEMENTWISE;  // 更新操作的标志
        op_flags[1] |= NPY_ITER_COPY | NPY_ITER_ALIGNED | NPY_ITER_OVERLAP_ASSUME_ELEMENTWISE;  // 复制操作的标志

        NPY_UF_DBG_PRINT("Allocating outer iterator\n");  // 调试输出信息

        // 创建高级迭代器对象，用于操作两个操作数
        iter = NpyIter_AdvancedNew(2, op, flags,
                                   NPY_KEEPORDER, NPY_UNSAFE_CASTING,
                                   op_flags, descrs,
                                   ndim_iter, op_axes, NULL, 0);
        if (iter == NULL) {
            goto fail;  // 如果创建迭代器失败，则跳转到失败处理标签
        }

        // 如果存在复制或更新操作，更新操作数数组
        op[0] = NpyIter_GetOperandArray(iter)[0];
        op[1] = NpyIter_GetOperandArray(iter)[1];

        // 移除指定轴
        if (NpyIter_RemoveAxis(iter, axis) != NPY_SUCCEED) {
            goto fail;  // 移除指定轴失败则跳转到失败处理标签
        }
        // 移除多重索引
        if (NpyIter_RemoveMultiIndex(iter) != NPY_SUCCEED) {
            goto fail;  // 移除多重索引失败则跳转到失败处理标签
        }
    }

    // 如果未提供输出数组，则从迭代器获取输出
    if (out == NULL) {
        if (iter) {
            // 从迭代器获取第一个操作数并增加其引用计数
            op[0] = out = NpyIter_GetOperandArray(iter)[0];
            Py_INCREF(out);
        } else {
            // 如果没有迭代器，则根据描述符创建新的数组
            PyArray_Descr *dtype = descrs[0];
            Py_INCREF(dtype);
            // 根据描述符创建新的数组对象
            op[0] = out = (PyArrayObject *)PyArray_NewFromDescr(
                                    &PyArray_Type, dtype,
                                    ndim, PyArray_DIMS(op[1]), NULL, NULL,
                                    0, NULL);
            if (out == NULL) {
                goto fail;  // 如果创建数组对象失败，则跳转到失败处理标签
            }
        }
    }

    npy_intp fixed_strides[3];
    // 如果需要外部迭代器，则获取内部固定步长数组
    if (need_outer_iterator) {
        NpyIter_GetInnerFixedStrideArray(iter, fixed_strides);
    } else {
        // 否则获取指定轴的固定步长
        fixed_strides[0] = PyArray_STRIDES(op[0])[axis];
        fixed_strides[1] = PyArray_STRIDES(op[1])[axis];
        fixed_strides[2] = fixed_strides[0];
    }

    NPY_ARRAYMETHOD_FLAGS flags = 0;
    // 获取 UFunc 实现的步进循环函数及其标志
    if (ufuncimpl->get_strided_loop(&context,
            1, 0, fixed_strides, &strided_loop, &auxdata, &flags) < 0) {
        goto fail;  // 如果获取步进循环函数失败，则跳转到失败处理标签
    }
    needs_api = (flags & NPY_METH_REQUIRES_PYAPI) != 0;  // 检查是否需要 Python API
    if (!(flags & NPY_METH_NO_FLOATINGPOINT_ERRORS)) {
        /* 从清除浮点异常标志开始 */
        npy_clear_floatstatus_barrier((char*)&iter);  // 清除浮点异常标志
    }
    /*
     * 如果减少轴的大小为零，则根据 UFUNC_REDUCE 返回减少单位，
     * 或根据 UFUNC_ACCUMULATE 返回大小为零的输出数组。
     */
    if (PyArray_DIM(op[1], axis) == 0) {
        // 如果第二个操作数在指定轴上的维度为零，则跳转到完成标签
        goto finish;
    }
    else if (PyArray_SIZE(op[0]) == 0) {
        // 如果第一个操作数的总大小为零，则跳转到完成标签
        goto finish;
    }

    if (iter && NpyIter_GetIterSize(iter) != 0) {
        char *dataptr_copy[3];
        npy_intp stride_copy[3];
        npy_intp count_m1, stride0, stride1;

        NpyIter_IterNextFunc *iternext;
        char **dataptr;

        int itemsize = descrs[0]->elsize;

        /* 获取循环所需的变量 */
        iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            // 如果获取迭代器下一个函数失败，则跳转到失败标签
            goto fail;
        }
        dataptr = NpyIter_GetDataPtrArray(iter);
        needs_api |= NpyIter_IterationNeedsAPI(iter);

        /* 使用仅外部迭代器执行循环 */
        count_m1 = PyArray_DIM(op[1], axis)-1;
        stride0 = 0, stride1 = PyArray_STRIDE(op[1], axis);

        NPY_UF_DBG_PRINT("UFunc: Reduce loop with just outer iterator\n");

        stride0 = PyArray_STRIDE(op[0], axis);

        stride_copy[0] = stride0;
        stride_copy[1] = stride1;
        stride_copy[2] = stride0;

        if (!needs_api) {
            // 如果不需要使用 Python API，则开启线程
            NPY_BEGIN_THREADS_THRESHOLDED(NpyIter_GetIterSize(iter));
        }

        do {
            dataptr_copy[0] = dataptr[0];
            dataptr_copy[1] = dataptr[1];
            dataptr_copy[2] = dataptr[0];

            /*
             * 复制第一个元素以开始减少。
             *
             * 输出（dataptr[0]）和输入（dataptr[1]）可能指向同一内存，
             * 例如 np.add.accumulate(a, out=a)。
             */
            if (descrs[2]->type_num == NPY_OBJECT) {
                /*
                 * 在减少引用计数之前增加引用计数，以避免引用计数
                 * 短暂地为零的可能性。
                 */
                Py_XINCREF(*(PyObject **)dataptr_copy[1]);
                Py_XDECREF(*(PyObject **)dataptr_copy[0]);
                *(PyObject **)dataptr_copy[0] =
                                    *(PyObject **)dataptr_copy[1];
            }
            else {
                memmove(dataptr_copy[0], dataptr_copy[1], itemsize);
            }

            if (count_m1 > 0) {
                /* 将两个项目转换为三个，用于内部循环 */
                dataptr_copy[1] += stride1;
                dataptr_copy[2] += stride0;
                NPY_UF_DBG_PRINT1("iterator loop count %d\n",
                                                (int)count_m1);
                // 执行带有上下文的步进循环
                res = strided_loop(&context,
                        dataptr_copy, &count_m1, stride_copy, auxdata);
            }
        } while (res == 0 && iternext(iter));

        NPY_END_THREADS;
    }
    else if (iter == NULL) {
        // 如果迭代器为空，则执行以下操作

        // 复制 dataptr 的三个指针，用于内部循环
        char *dataptr_copy[3];

        // 获取数组元素大小
        int itemsize = descrs[0]->elsize;

        /* 执行没有迭代器的循环 */
        // 获取数组 op[1] 在指定轴上的维度大小作为循环次数
        npy_intp count = PyArray_DIM(op[1], axis);
        // 计算 op[1] 在指定轴上的步长
        npy_intp stride0 = 0, stride1 = PyArray_STRIDE(op[1], axis);

        // 调试打印信息
        NPY_UF_DBG_PRINT("UFunc: Reduce loop with no iterators\n");

        // 检查 op[0] 和 op[1] 的维度是否相同，并且各维度大小是否一致
        if (PyArray_NDIM(op[0]) != PyArray_NDIM(op[1]) ||
                !PyArray_CompareLists(PyArray_DIMS(op[0]),
                                      PyArray_DIMS(op[1]),
                                      PyArray_NDIM(op[0]))) {
            // 如果维度不匹配，设置错误信息并跳转到失败标签
            PyErr_SetString(PyExc_ValueError,
                    "provided out is the wrong size "
                    "for the accumulation.");
            goto fail;
        }
        // 获取 op[0] 在指定轴上的步长
        stride0 = PyArray_STRIDE(op[0], axis);

        /* 将两个元素复制为三个用于内部循环 */
        dataptr_copy[0] = PyArray_BYTES(op[0]);
        dataptr_copy[1] = PyArray_BYTES(op[1]);
        dataptr_copy[2] = PyArray_BYTES(op[0]);

        /*
         * 复制第一个元素以开始归约。
         *
         * 输出 (dataptr[0]) 和 输入 (dataptr[1]) 可能指向同一内存，
         * 例如 np.add.accumulate(a, out=a) 的情况。
         */
        if (descrs[2]->type_num == NPY_OBJECT) {
            /*
             * 在减少引用之前增加引用，以避免引用计数暂时为零的可能性。
             */
            Py_XINCREF(*(PyObject **)dataptr_copy[1]);
            Py_XDECREF(*(PyObject **)dataptr_copy[0]);
            *(PyObject **)dataptr_copy[0] =
                                *(PyObject **)dataptr_copy[1];
        }
        else {
            // 使用 memmove 将内存块从 dataptr_copy[1] 复制到 dataptr_copy[0]
            memmove(dataptr_copy[0], dataptr_copy[1], itemsize);
        }

        // 如果 count 大于 1，进行迭代处理
        if (count > 1) {
            --count;
            // 更新 dataptr_copy[1] 和 dataptr_copy[2] 的指针位置
            dataptr_copy[1] += stride1;
            dataptr_copy[2] += stride0;

            // 调试打印循环计数信息
            NPY_UF_DBG_PRINT1("iterator loop count %d\n", (int)count);

            // 检查是否需要 API（对描述符进行引用计数检查）
            needs_api = PyDataType_REFCHK(descrs[0]);

            // 如果不需要 API，启动线程
            if (!needs_api) {
                NPY_BEGIN_THREADS_THRESHOLDED(count);
            }

            // 调用 strided_loop 函数执行循环操作
            res = strided_loop(&context,
                    dataptr_copy, &count, fixed_strides, auxdata);

            // 结束线程
            NPY_END_THREADS;
        }
    }
finish:
    // 释放辅助数据
    NPY_AUXDATA_FREE(auxdata);
    // 减少第一个描述符的引用计数
    Py_DECREF(descrs[0]);
    // 减少第二个描述符的引用计数
    Py_DECREF(descrs[1]);
    // 减少第三个描述符的引用计数
    Py_DECREF(descrs[2]);

    // 如果迭代器没有成功释放，则设置结果为-1
    if (!NpyIter_Deallocate(iter)) {
        res = -1;
    }

    // 如果结果为0且未设置不检查浮点错误标志，则检查浮点错误
    if (res == 0 && !(flags & NPY_METH_NO_FLOATINGPOINT_ERRORS)) {
        /* 注意：即使 `res < 0` 时也可以检查浮点错误 */
        res = _check_ufunc_fperr(errormask, "accumulate");
    }

    // 如果结果小于0，释放输出对象并返回NULL
    if (res < 0) {
        Py_DECREF(out);
        return NULL;
    }

    // 返回输出对象的PyObject指针类型
    return (PyObject *)out;

fail:
    // 在失败标签下，释放输出对象的引用
    Py_XDECREF(out);

    // 释放辅助数据
    NPY_AUXDATA_FREE(auxdata);
    // 释放第一个描述符的引用
    Py_XDECREF(descrs[0]);
    // 释放第二个描述符的引用
    Py_XDECREF(descrs[1]);
    // 释放第三个描述符的引用
    Py_XDECREF(descrs[2]);

    // 释放迭代器
    NpyIter_Deallocate(iter);

    // 返回NULL，表示操作失败
    return NULL;
}



/*
 * Reduceat performs a reduce over an axis using the indices as a guide
 *
 * op.reduceat(array,indices)  computes
 * op.reduce(array[indices[i]:indices[i+1]]
 * for i=0..end with an implicit indices[i+1]=len(array)
 * assumed when i=end-1
 *
 * if indices[i+1] <= indices[i]+1
 * then the result is array[indices[i]] for that value
 *
 * op.accumulate(array) is the same as
 * op.reduceat(array,indices)[::2]
 * where indices is range(len(array)-1) with a zero placed in every other sample
 * indices = zeros(len(array)*2-1)
 * indices[1::2] = range(1,len(array))
 *
 * output shape is based on the size of indices
 *
 * TODO: Reduceat duplicates too much code from accumulate!
 */
static PyObject *
PyUFunc_Reduceat(PyUFuncObject *ufunc, PyArrayObject *arr, PyArrayObject *ind,
                 PyArrayObject *out, int axis, PyArray_DTypeMeta *signature[3])
{
    PyArrayObject *op[3];
    int op_axes_arrays[3][NPY_MAXDIMS];
    int *op_axes[3] = {op_axes_arrays[0], op_axes_arrays[1],
                            op_axes_arrays[2]};
    npy_uint32 op_flags[3];
    int idim, ndim;
    int needs_api, need_outer_iterator = 0;

    int res = 0;

    NpyIter *iter = NULL;

    PyArrayMethod_StridedLoop *strided_loop;
    NpyAuxData *auxdata = NULL;

    /* The reduceat indices - ind must be validated outside this call */
    npy_intp *reduceat_ind;
    npy_intp i, ind_size, red_axis_size;

    const char *ufunc_name = ufunc_get_name_cstr(ufunc);
    char *opname = "reduceat";

    /* These parameters comefrom a TLS global */
    int buffersize = 0, errormask = 0;

    NPY_BEGIN_THREADS_DEF;

    reduceat_ind = (npy_intp *)PyArray_DATA(ind);
    ind_size = PyArray_DIM(ind, 0);
    red_axis_size = PyArray_DIM(arr, axis);

    /* Check for out-of-bounds values in indices array */
    for (i = 0; i < ind_size; ++i) {
        if (reduceat_ind[i] < 0 || reduceat_ind[i] >= red_axis_size) {
            PyErr_Format(PyExc_IndexError,
                "index %" NPY_INTP_FMT " out-of-bounds in %s.%s [0, %" NPY_INTP_FMT ")",
                reduceat_ind[i], ufunc_name, opname, red_axis_size);
            return NULL;
        }
    }

    NPY_UF_DBG_PRINT2("\nEvaluating ufunc %s.%s\n", ufunc_name, opname);

#if 0
    printf("Doing %s.%s on array with dtype :  ", ufunc_name, opname);
    # 使用PyObject_Print函数打印PyArray_DESCR(arr)的内容到标准输出(stdout)
    PyObject_Print((PyObject *)PyArray_DESCR(arr), stdout, 0);
    # 打印换行符到标准输出(stdout)
    printf("\n");
    # 打印字符串"Index size is %d"及其后接的ind_size的整数值到标准输出(stdout)
    printf("Index size is %d\n", (int)ind_size);
#endif

    if (_get_bufsize_errmask(&buffersize, &errormask) < 0) {
        // 检查缓冲区大小和错误掩码，如果小于0则返回空指针
        return NULL;
    }

    /* Take a reference to out for later returning */
    // 增加 out 的引用计数，以便稍后返回
    Py_XINCREF(out);

    PyArray_Descr *descrs[3];
    // 调用 reducelike_promote_and_resolve 函数，解析并推断 ufunc 的方法对象
    // 并返回描述符数组及实现对象
    PyArrayMethodObject *ufuncimpl = reducelike_promote_and_resolve(ufunc,
            arr, out, signature, NPY_TRUE, descrs, NPY_UNSAFE_CASTING,
            "reduceat");
    if (ufuncimpl == NULL) {
        // 如果 ufuncimpl 为空指针，则返回空指针
        return NULL;
    }

    /*
     * The below code assumes that all descriptors are interchangeable, we
     * allow them to not be strictly identical (but they typically should be)
     */
    // 断言前三个描述符应当是等效的，可能不是严格相同但通常应当如此
    assert(PyArray_EquivTypes(descrs[0], descrs[1])
           && PyArray_EquivTypes(descrs[0], descrs[2]));

    if (PyDataType_REFCHK(descrs[2]) && descrs[2]->type_num != NPY_OBJECT) {
        /* This can be removed, but the initial element copy needs fixing */
        // 如果第三个描述符的数据类型需要引用检查且不是对象类型，则设置类型错误并跳转到失败处理
        PyErr_SetString(PyExc_TypeError,
                "reduceat currently only supports `object` dtype with "
                "references");
        goto fail;
    }

    PyArrayMethod_Context context = {
        .caller = (PyObject *)ufunc,
        .method = ufuncimpl,
        .descriptors = descrs,
    };

    ndim = PyArray_NDIM(arr);

#if NPY_UF_DBG_TRACING
    // 如果定义了 NPY_UF_DBG_TRACING 宏，则输出调试信息
    printf("Found %s.%s inner loop with dtype :  ", ufunc_name, opname);
    PyObject_Print((PyObject *)descrs[0], stdout, 0);
    printf("\n");
#endif

    /* Set up the op_axes for the outer loop */
    // 为外部循环设置操作轴
    for (idim = 0; idim < ndim; ++idim) {
        // 使用第 idim 维度来匹配 ind
        if (idim == axis) {
            op_axes_arrays[0][idim] = axis;
            op_axes_arrays[1][idim] = -1;
            op_axes_arrays[2][idim] = 0;
        }
        else {
            op_axes_arrays[0][idim] = idim;
            op_axes_arrays[1][idim] = idim;
            op_axes_arrays[2][idim] = -1;
        }
    }

    op[0] = out;
    op[1] = arr;
    op[2] = ind;

    if (out != NULL || ndim > 1 || !PyArray_ISALIGNED(arr) ||
            !PyArray_EquivTypes(descrs[0], PyArray_DESCR(arr))) {
        // 如果 out 不为空，或者 ndim 大于1，或者 arr 不是对齐的，
        // 或者第一个描述符与 arr 的描述符不等效，则需要外部迭代器
        need_outer_iterator = 1;
    }
    if (need_outer_iterator) {
        // 如果需要外部迭代器

        // 定义操作数的数据类型数组
        PyArray_Descr *op_dtypes[3] = {descrs[0], descrs[1], NULL};

        // 定义迭代器的标志
        npy_uint32 flags = NPY_ITER_ZEROSIZE_OK|
                           NPY_ITER_REFS_OK|
                           NPY_ITER_MULTI_INDEX|
                           NPY_ITER_COPY_IF_OVERLAP;

        /*
         * The way reduceat is set up, we can't do buffering,
         * so make a copy instead when necessary using
         * the UPDATEIFCOPY flag
         */

        /* The per-operand flags for the outer loop */
        // 外部循环的每个操作数的标志
        op_flags[0] = NPY_ITER_READWRITE|
                      NPY_ITER_NO_BROADCAST|
                      NPY_ITER_ALLOCATE|
                      NPY_ITER_NO_SUBTYPE|
                      NPY_ITER_UPDATEIFCOPY|
                      NPY_ITER_ALIGNED;
        op_flags[1] = NPY_ITER_READONLY|
                      NPY_ITER_COPY|
                      NPY_ITER_ALIGNED;
        op_flags[2] = NPY_ITER_READONLY;

        // 将第二个操作数的数据类型设置为第一个操作数的数据类型
        op_dtypes[1] = op_dtypes[0];

        // 打印调试信息
        NPY_UF_DBG_PRINT("Allocating outer iterator\n");

        // 创建高级迭代器对象
        iter = NpyIter_AdvancedNew(3, op, flags,
                                   NPY_KEEPORDER, NPY_UNSAFE_CASTING,
                                   op_flags, op_dtypes,
                                   ndim, op_axes, NULL, 0);
        if (iter == NULL) {
            goto fail;
        }

        /* Remove the inner loop axis from the outer iterator */
        // 从外部迭代器中移除内部循环轴
        if (NpyIter_RemoveAxis(iter, axis) != NPY_SUCCEED) {
            goto fail;
        }
        // 移除多重索引标志
        if (NpyIter_RemoveMultiIndex(iter) != NPY_SUCCEED) {
            goto fail;
        }

        /* In case COPY or UPDATEIFCOPY occurred */
        // 如果发生了COPY或UPDATEIFCOPY操作

        // 获取迭代器的操作数数组
        op[0] = NpyIter_GetOperandArray(iter)[0];
        op[1] = NpyIter_GetOperandArray(iter)[1];
        op[2] = NpyIter_GetOperandArray(iter)[2];

        // 如果输出为空，则将输出设置为第一个操作数，并增加其引用计数
        if (out == NULL) {
            out = op[0];
            Py_INCREF(out);
        }
    }
    else {
        // 如果不需要外部迭代器

        /*
         * Allocate the output for when there's no outer iterator, we always
         * use the outer_iteration path when `out` is passed.
         */
        // 分配输出内存，当没有外部迭代器时，总是使用外部迭代路径，如果传入了`out`

        // 断言输出为空
        assert(out == NULL);

        // 增加第一个描述符的引用计数
        Py_INCREF(descrs[0]);

        // 从描述符创建新的数组对象
        op[0] = out = (PyArrayObject *)PyArray_NewFromDescr(
                                    &PyArray_Type, descrs[0],
                                    1, &ind_size, NULL, NULL,
                                    0, NULL);
        if (out == NULL) {
            goto fail;
        }
    }

    // 固定步长数组
    npy_intp fixed_strides[3];

    // 如果需要外部迭代器，获取内部固定步长数组
    if (need_outer_iterator) {
        NpyIter_GetInnerFixedStrideArray(iter, fixed_strides);
    }
    else {
        // 否则，设置第二个操作数的固定步长为指定轴上的数组步长
        fixed_strides[1] = PyArray_STRIDES(op[1])[axis];
    }

    // 第一个操作数的固定步长设为0，第三个操作数的固定步长设为0
    fixed_strides[0] = 0;
    fixed_strides[2] = 0;

    // 数组方法的标志
    NPY_ARRAYMETHOD_FLAGS flags = 0;

    // 如果获取步长循环失败，则跳转到错误处理部分
    if (ufuncimpl->get_strided_loop(&context,
            1, 0, fixed_strides, &strided_loop, &auxdata, &flags) < 0) {
        goto fail;
    }

    // 需要API调用
    needs_api = (flags & NPY_METH_REQUIRES_PYAPI) != 0;
    # 如果没有设置 NPY_METH_NO_FLOATINGPOINT_ERRORS 标志位
    if (!(flags & NPY_METH_NO_FLOATINGPOINT_ERRORS)) {
        # 清除浮点异常标志位
        npy_clear_floatstatus_barrier((char*)&iter);
    }

    # 如果输出数组 op[0] 的元素个数为 0
    '''
     * 如果输出数组 op[0] 的元素个数为 0，现在返回。
     '''
    if (PyArray_SIZE(op[0]) == 0) {
        # 跳转到完成处理的标签
        goto finish;
    }
    # 检查迭代器是否存在且迭代器大小不为零
    if (iter && NpyIter_GetIterSize(iter) != 0) {
        # 创建指针数组和步长数组的副本
        char *dataptr_copy[3];
        npy_intp stride_copy[3];

        # 迭代器的下一个迭代函数和数据指针数组
        NpyIter_IterNextFunc *iternext;
        char **dataptr;
        npy_intp count_m1;
        npy_intp stride0, stride1;
        npy_intp stride0_ind = PyArray_STRIDE(op[0], axis);

        # 计算元素大小并检查是否需要 API 支持
        int itemsize = descrs[0]->elsize;
        needs_api |= NpyIter_IterationNeedsAPI(iter);

        /* Get the variables needed for the loop */
        # 获取用于循环的变量
        iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            goto fail;
        }
        dataptr = NpyIter_GetDataPtrArray(iter);

        /* Execute the loop with just the outer iterator */
        # 使用外部迭代器执行循环
        count_m1 = PyArray_DIM(op[1], axis)-1;
        stride0 = 0;
        stride1 = PyArray_STRIDE(op[1], axis);

        NPY_UF_DBG_PRINT("UFunc: Reduce loop with just outer iterator\n");

        # 复制步长以备使用
        stride_copy[0] = stride0;
        stride_copy[1] = stride1;
        stride_copy[2] = stride0;

        # 如果不需要 API 支持，启动多线程执行
        if (!needs_api) {
            NPY_BEGIN_THREADS_THRESHOLDED(NpyIter_GetIterSize(iter));
        }

        do {
            # 对于每个索引进行迭代
            for (i = 0; i < ind_size; ++i) {
                npy_intp start = reduceat_ind[i],
                        end = (i == ind_size-1) ? count_m1+1 :
                                                  reduceat_ind[i+1];
                npy_intp count = end - start;

                # 设置数据指针的副本
                dataptr_copy[0] = dataptr[0] + stride0_ind*i;
                dataptr_copy[1] = dataptr[1] + stride1*start;
                dataptr_copy[2] = dataptr[0] + stride0_ind*i;

                /*
                 * 复制第一个元素以启动归约。
                 *
                 * 输出（dataptr[0]）和输入（dataptr[1]）可能指向同一内存，
                 * 例如 np.add.reduceat(a, np.arange(len(a)), out=a)。
                 */
                if (descrs[2]->type_num == NPY_OBJECT) {
                    /*
                     * 在减少引用之前增加引用，以避免引用计数暂时为零的可能性。
                     */
                    Py_XINCREF(*(PyObject **)dataptr_copy[1]);
                    Py_XDECREF(*(PyObject **)dataptr_copy[0]);
                    *(PyObject **)dataptr_copy[0] =
                                        *(PyObject **)dataptr_copy[1];
                }
                else {
                    memmove(dataptr_copy[0], dataptr_copy[1], itemsize);
                }

                if (count > 1) {
                    /* 内部循环类似于 REDUCE */
                    --count;
                    dataptr_copy[1] += stride1;
                    NPY_UF_DBG_PRINT1("iterator loop count %d\n",
                                                    (int)count);
                    res = strided_loop(&context,
                            dataptr_copy, &count, stride_copy, auxdata);
                }
            }
        } while (res == 0 && iternext(iter));

        # 结束多线程
        NPY_END_THREADS;
    }
    else if (iter == NULL) {
        // 复制数据指针的数组，用于操作数组元素
        char *dataptr_copy[3];

        // 获取第一个数组元素的大小
        int itemsize = descrs[0]->elsize;

        // 计算第一个操作数在指定轴上的步长
        npy_intp stride0_ind = PyArray_STRIDE(op[0], axis);
        // 计算第二个操作数在指定轴上的步长
        npy_intp stride1 = PyArray_STRIDE(op[1], axis);

        // 调试信息：没有迭代器的情况下进行归约循环
        NPY_UF_DBG_PRINT("UFunc: Reduce loop with no iterators\n");

        // 如果不需要 API 调用，则开始多线程处理
        if (!needs_api) {
            NPY_BEGIN_THREADS;
        }

        // 遍历每个归约索引
        for (i = 0; i < ind_size; ++i) {
            // 计算当前归约索引的起始位置和结束位置
            npy_intp start = reduceat_ind[i],
                    end = (i == ind_size-1) ? PyArray_DIM(arr,axis) :
                                              reduceat_ind[i+1];
            // 计算当前归约的元素个数
            npy_intp count = end - start;

            // 设置数据指针副本的位置
            dataptr_copy[0] = PyArray_BYTES(op[0]) + stride0_ind*i;
            dataptr_copy[1] = PyArray_BYTES(op[1]) + stride1*start;
            dataptr_copy[2] = PyArray_BYTES(op[0]) + stride0_ind*i;

            /*
             * 复制第一个元素以开始归约。
             *
             * 输出（dataptr[0]）和输入（dataptr[1]）可能指向相同的内存，
             * 例如 np.add.reduceat(a, np.arange(len(a)), out=a)。
             */
            if (descrs[2]->type_num == NPY_OBJECT) {
                /*
                 * 在减少引用之前增加引用，以避免临时引用计数为零的可能性。
                 */
                Py_XINCREF(*(PyObject **)dataptr_copy[1]);
                Py_XDECREF(*(PyObject **)dataptr_copy[0]);
                *(PyObject **)dataptr_copy[0] =
                                    *(PyObject **)dataptr_copy[1];
            }
            else {
                // 使用内存移动复制数据
                memmove(dataptr_copy[0], dataptr_copy[1], itemsize);
            }

            // 如果元素个数大于1，则执行内部循环类似于 REDUCE 操作
            if (count > 1) {
                --count;
                dataptr_copy[1] += stride1;
                // 调试信息：迭代器循环次数
                NPY_UF_DBG_PRINT1("iterator loop count %d\n",
                                                (int)count);
                // 执行步进循环操作
                res = strided_loop(&context,
                        dataptr_copy, &count, fixed_strides, auxdata);
                // 如果返回值不为0，则中断循环
                if (res != 0) {
                    break;
                }
            }
        }

        // 结束多线程处理
        NPY_END_THREADS;
    }
finish:
    // 释放辅助数据
    NPY_AUXDATA_FREE(auxdata);
    // 减少对象的引用计数
    Py_DECREF(descrs[0]);
    Py_DECREF(descrs[1]);
    Py_DECREF(descrs[2]);

    // 释放迭代器内存，并检查是否成功
    if (!NpyIter_Deallocate(iter)) {
        res = -1;
    }

    // 如果处理成功且未设置 NPY_METH_NO_FLOATINGPOINT_ERRORS 标志，则检查浮点错误
    if (res == 0 && !(flags & NPY_METH_NO_FLOATINGPOINT_ERRORS)) {
        /* NOTE: We could check float errors even when `res < 0` */
        res = _check_ufunc_fperr(errormask, "reduceat");
    }

    // 如果处理出现错误，则释放输出对象并返回空
    if (res < 0) {
        Py_DECREF(out);
        return NULL;
    }

    // 处理成功则返回输出对象
    return (PyObject *)out;

fail:
    // 处理失败时释放输出对象
    Py_XDECREF(out);

    // 释放辅助数据
    NPY_AUXDATA_FREE(auxdata);
    // 减少对象的引用计数
    Py_XDECREF(descrs[0]);
    Py_XDECREF(descrs[1]);
    Py_XDECREF(descrs[2]);

    // 释放迭代器内存
    NpyIter_Deallocate(iter);

    // 返回空指针表示失败
    return NULL;
}


static npy_bool
tuple_all_none(PyObject *tup) {
    // 检查元组中所有元素是否都为 None
    npy_intp i;
    for (i = 0; i < PyTuple_GET_SIZE(tup); ++i) {
        if (PyTuple_GET_ITEM(tup, i) != Py_None) {
            return NPY_FALSE;
        }
    }
    return NPY_TRUE;
}


static int
_set_full_args_out(int nout, PyObject *out_obj, ufunc_full_args *full_args)
{
    // 设置完整参数中的输出对象
    if (PyTuple_CheckExact(out_obj)) {
        // 如果输出对象是元组，则检查元素数量是否与 nout 相符
        if (PyTuple_GET_SIZE(out_obj) != nout) {
            PyErr_SetString(PyExc_ValueError,
                            "The 'out' tuple must have exactly "
                            "one entry per ufunc output");
            return -1;
        }
        // 如果元组中所有元素均为 None，则返回成功
        if (tuple_all_none(out_obj)) {
            return 0;
        }
        else {
            // 增加输出对象的引用计数并存储在完整参数中
            Py_INCREF(out_obj);
            full_args->out = out_obj;
        }
    }
    else if (nout == 1) {
        // 如果输出对象是 None，则返回成功
        if (out_obj == Py_None) {
            return 0;
        }
        /* 如果只有一个输出，则可以是数组 */
        // 将单个输出对象打包成元组
        full_args->out = PyTuple_Pack(1, out_obj);
        if (full_args->out == NULL) {
            return -1;
        }
    }
    else {
        // 如果输出对象不符合要求，则设置相应的类型错误并返回失败
        PyErr_SetString(PyExc_TypeError,
                        nout > 1 ? "'out' must be a tuple of arrays" :
                        "'out' must be an array or a tuple with "
                        "a single array");
        return -1;
    }
    return 0;
}


/* forward declaration */
// 前向声明：获取数据类型元对象的函数
static PyArray_DTypeMeta * _get_dtype(PyObject *dtype_obj);

/*
 * This code handles reduce, reduceat, and accumulate
 * (accumulate and reduce are special cases of the more general reduceat
 * but they are handled separately for speed)
 */
// 这段代码处理 reduce、reduceat 和 accumulate 函数
static PyObject *
PyUFunc_GenericReduction(PyUFuncObject *ufunc,
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames, int operation)
{
    int i, naxes=0, ndim;
    int axes[NPY_MAXDIMS];

    ufunc_full_args full_args = {NULL, NULL};
    PyObject *axes_obj = NULL;
    PyArrayObject *mp = NULL, *wheremask = NULL, *ret = NULL;
    PyObject *op = NULL;
    PyArrayObject *indices = NULL;
    PyArray_DTypeMeta *signature[3] = {NULL, NULL, NULL};
    PyArrayObject *out = NULL;
    int keepdims = 0;
    PyObject *initial = NULL;
    npy_bool out_is_passed_by_position;


    static char *_reduce_type[] = {"reduce", "accumulate", "reduceat", NULL};
    /*
     * 如果传入的 ufunc 是空指针，则设置一个值错误异常并返回空指针。
     * 这表示函数不支持空指针对应的操作。
     */
    if (ufunc == NULL) {
        PyErr_SetString(PyExc_ValueError, "function not supported");
        return NULL;
    }

    /*
     * 如果 ufunc 的 core_enabled 属性为真，则抛出一个运行时错误异常。
     * 这表示在带有签名的 ufunc 上未定义此操作。
     */
    if (ufunc->core_enabled) {
        PyErr_Format(PyExc_RuntimeError,
                     "Reduction not defined on ufunc with signature");
        return NULL;
    }

    /*
     * 如果 ufunc 的输入参数个数 nin 不等于 2，则抛出一个值错误异常。
     * 这表示该操作仅支持二元函数。
     */
    if (ufunc->nin != 2) {
        PyErr_Format(PyExc_ValueError,
                     "%s only supported for binary functions",
                     _reduce_type[operation]);
        return NULL;
    }

    /*
     * 如果 ufunc 的输出参数个数 nout 不等于 1，则抛出一个值错误异常。
     * 这表示该操作仅支持返回单个值的函数。
     */
    if (ufunc->nout != 1) {
        PyErr_Format(PyExc_ValueError,
                     "%s only supported for functions "
                     "returning a single value",
                     _reduce_type[operation]);
        return NULL;
    }

    /*
     * 执行参数解析，但只提取参数。这是为了保留 __array_ufunc__ 的行为，
     * 不对参数执行任何检查，我们可以针对某些参数更改此行为。
     */
    PyObject *otype_obj = NULL, *out_obj = NULL, *indices_obj = NULL;
    PyObject *keepdims_obj = NULL, *wheremask_obj = NULL;

    /*
     * 如果操作是 UFUNC_REDUCEAT，则准备解析参数。
     * 在这里，我们解析 "reduceat" 操作的特定参数。
     */
    if (operation == UFUNC_REDUCEAT) {
        NPY_PREPARE_ARGPARSER;

        if (npy_parse_arguments("reduceat", args, len_args, kwnames,
                "array", NULL, &op,
                "indices", NULL, &indices_obj,
                "|axis", NULL, &axes_obj,
                "|dtype", NULL, &otype_obj,
                "|out", NULL, &out_obj,
                NULL, NULL, NULL) < 0) {
            goto fail;
        }
        /* 为 PyUfunc_CheckOverride 准备输入参数 */
        full_args.in = PyTuple_Pack(2, op, indices_obj);
        if (full_args.in == NULL) {
            goto fail;
        }
        out_is_passed_by_position = len_args >= 5;
    }
    /*
     * 如果操作是 UFUNC_ACCUMULATE，则准备解析参数。
     * 在这里，我们解析 "accumulate" 操作的特定参数。
     */
    else if (operation == UFUNC_ACCUMULATE) {
        NPY_PREPARE_ARGPARSER;

        if (npy_parse_arguments("accumulate", args, len_args, kwnames,
                "array", NULL, &op,
                "|axis", NULL, &axes_obj,
                "|dtype", NULL, &otype_obj,
                "|out", NULL, &out_obj,
                NULL, NULL, NULL) < 0) {
            goto fail;
        }
        /* 为 PyUfunc_CheckOverride 准备输入参数 */
        full_args.in = PyTuple_Pack(1, op);
        if (full_args.in == NULL) {
            goto fail;
        }
        out_is_passed_by_position = len_args >= 4;
    }
    else {
        NPY_PREPARE_ARGPARSER;
        
        if (npy_parse_arguments("reduce", args, len_args, kwnames,
                "array", NULL, &op,
                "|axis", NULL, &axes_obj,
                "|dtype", NULL, &otype_obj,
                "|out", NULL, &out_obj,
                "|keepdims", NULL, &keepdims_obj,
                "|initial", &_not_NoValue, &initial,
                "|where", NULL, &wheremask_obj,
                NULL, NULL, NULL) < 0) {
            goto fail;
        }
        /* 准备输入参数以供 PyUfunc_CheckOverride 使用 */
        full_args.in = PyTuple_Pack(1, op);
        if (full_args.in == NULL) {
            goto fail;
        }
        // 检查是否通过位置传递了输出参数
        out_is_passed_by_position = len_args >= 4;
    }
    
    /* 规范化输出参数以供 PyUFunc_CheckOverride 和类型转换使用。 */
    if (out_is_passed_by_position) {
        /* 在这个分支中，输出参数总是封装在一个元组中。 */
        if (out_obj != Py_None) {
            full_args.out = PyTuple_Pack(1, out_obj);
            if (full_args.out == NULL) {
                goto fail;
            }
        }
    }
    else if (out_obj) {
        // 将输出参数设置到 full_args 结构体中
        if (_set_full_args_out(1, out_obj, &full_args) < 0) {
            goto fail;
        }
        /* 确保 out_obj 是数组而不是元组 */
        if (full_args.out != NULL) {
            out_obj = PyTuple_GET_ITEM(full_args.out, 0);
        }
    }

    /* 现在我们有了检查覆盖的所有信息 */
    PyObject *override = NULL;
    int errval = PyUFunc_CheckOverride(ufunc, _reduce_type[operation],
            full_args.in, full_args.out, wheremask_obj, args, len_args, kwnames, &override);
    if (errval) {
        return NULL;
    }
    else if (override) {
        Py_XDECREF(full_args.in);
        Py_XDECREF(full_args.out);
        return override;
    }

    /* 完成所有参数的解析（无论是哪种减少类型） */
    if (indices_obj) {
        // 从 indices_obj 中获取整数类型的描述符
        PyArray_Descr *indtype = PyArray_DescrFromType(NPY_INTP);

        indices = (PyArrayObject *)PyArray_FromAny(indices_obj,
                indtype, 1, 1, NPY_ARRAY_CARRAY, NULL);
        if (indices == NULL) {
            goto fail;
        }
    }
    if (otype_obj && otype_obj != Py_None) {
        /* 使用 `_get_dtype` 因为 `otype_obj` 是类型而不是实例 */
        signature[0] = _get_dtype(otype_obj);
        if (signature[0] == NULL) {
            goto fail;
        }
    }
    if (out_obj && !PyArray_OutputConverter(out_obj, &out)) {
        goto fail;
    }
    if (keepdims_obj && !PyArray_PythonPyIntFromInt(keepdims_obj, &keepdims)) {
        goto fail;
    }
    if (wheremask_obj && !_wheremask_converter(wheremask_obj, &wheremask)) {
        goto fail;
    }

    /* 确保输入是一个数组 */
    mp = (PyArrayObject *)PyArray_FromAny(op, NULL, 0, 0, 0, NULL);
    if (mp == NULL) {
        goto fail;
    }

    ndim = PyArray_NDIM(mp);

    /* 将 'axis' 参数转换为轴列表 */
    # 如果 axes_obj 是空指针 NULL
    if (axes_obj == NULL) {
        /* 应用默认设置 */
        # 如果 ndim（数组的维数）为0，则没有轴
        if (ndim == 0) {
            naxes = 0;
        }
        else {
            # 否则只有一个轴，默认为0
            naxes = 1;
            axes[0] = 0;
        }
    }
    # 如果 axes_obj 是 Py_None，表示要使用所有的轴
    else if (axes_obj == Py_None) {
        /* 将 'None' 转换为所有的轴 */
        naxes = ndim;  # 轴的数量等于数组的维数
        for (i = 0; i < naxes; ++i) {
            # 逐个将轴索引赋值为对应的整数值
            axes[i] = i;
        }
    }
    # 如果 axes_obj 是一个元组
    else if (PyTuple_Check(axes_obj)) {
        # 获取元组中的轴的数量
        naxes = PyTuple_Size(axes_obj);
        if (naxes < 0 || naxes > NPY_MAXDIMS) {
            PyErr_SetString(PyExc_ValueError,
                    "too many values for 'axis'");
            goto fail;  // 如果超出最大维数范围，抛出值错误异常
        }
        for (i = 0; i < naxes; ++i) {
            # 从元组中获取每个轴对象
            PyObject *tmp = PyTuple_GET_ITEM(axes_obj, i);
            # 将轴对象转换为整数
            int axis = PyArray_PyIntAsInt(tmp);
            # 检查转换是否出错
            if (error_converting(axis)) {
                goto fail;  // 如果转换出错，跳转到错误处理
            }
            # 检查并调整轴的值，确保在有效范围内
            if (check_and_adjust_axis(&axis, ndim) < 0) {
                goto fail;  // 如果轴值不合法，跳转到错误处理
            }
            axes[i] = (int)axis;  // 将调整后的轴值存入 axes 数组中
        }
    }
    else {
        /* 尝试将 axes_obj 解释为一个整数 */
        int axis = PyArray_PyIntAsInt(axes_obj);
        /* TODO: 这里最好使用 PyNumber_Index */
        # 检查是否转换出错
        if (error_converting(axis)) {
            goto fail;  // 如果转换出错，跳转到错误处理
        }
        /*
         * 对于 'sum'、'prod' 等操作，即使对标量进行缩减也允许，
         * 虽然这在技术上是不正确的，为了向后兼容性而特殊处理。
         */
        if (ndim == 0 && (axis == 0 || axis == -1)) {
            naxes = 0;  // 如果数组维数为0且轴为0或-1，则没有轴
        }
        else if (check_and_adjust_axis(&axis, ndim) < 0) {
            goto fail;  // 检查并调整轴的值，确保在有效范围内，若不合法则跳转到错误处理
        }
        else {
            axes[0] = (int)axis;  // 将调整后的轴值存入 axes 数组中
            naxes = 1;  // 轴的数量为1
        }
    }

    switch(operation) {
    case UFUNC_REDUCE:
        # 执行 UFUNC_REDUCE 操作
        ret = PyUFunc_Reduce(ufunc,
                mp, out, naxes, axes, signature, keepdims, initial, wheremask);
        Py_XSETREF(wheremask, NULL);  // 清空 wheremask 引用
        break;
    case UFUNC_ACCUMULATE:
        # 如果数组的维数为0，无法对标量进行累积操作
        if (ndim == 0) {
            PyErr_SetString(PyExc_TypeError, "cannot accumulate on a scalar");
            goto fail;  // 抛出类型错误异常，跳转到错误处理
        }
        # 如果轴的数量不为1，累积操作不允许多个轴
        if (naxes != 1) {
            PyErr_SetString(PyExc_ValueError,
                        "accumulate does not allow multiple axes");
            goto fail;  // 抛出值错误异常，跳转到错误处理
        }
        # 执行 UFUNC_ACCUMULATE 操作
        ret = (PyArrayObject *)PyUFunc_Accumulate(ufunc,
                mp, out, axes[0], signature);
        break;
    case UFUNC_REDUCEAT:
        # 如果数组的维数为0，无法对标量进行 reduceat 操作
        if (ndim == 0) {
            PyErr_SetString(PyExc_TypeError, "cannot reduceat on a scalar");
            goto fail;  // 抛出类型错误异常，跳转到错误处理
        }
        # 如果轴的数量不为1，reduceat 操作不允许多个轴
        if (naxes != 1) {
            PyErr_SetString(PyExc_ValueError,
                        "reduceat does not allow multiple axes");
            goto fail;  // 抛出值错误异常，跳转到错误处理
        }
        # 执行 UFUNC_REDUCEAT 操作
        ret = (PyArrayObject *)PyUFunc_Reduceat(ufunc,
                mp, indices, out, axes[0], signature);
        Py_SETREF(indices, NULL);  // 清空 indices 引用
        break;
    }
    # 如果操作执行失败，跳转到错误处理
    if (ret == NULL) {
        goto fail;
    }

    # 释放 signature 数组的引用
    Py_DECREF(signature[0]);
    Py_DECREF(signature[1]);
    Py_DECREF(signature[2]);
    // 减少对 signature 数组中第三个元素的引用计数，释放其内存

    Py_DECREF(mp);
    // 减少对 mp 对象的引用计数，释放其内存
    Py_XDECREF(full_args.in);
    // 如果 full_args.in 不为 NULL，则减少其引用计数，释放其内存
    Py_XDECREF(full_args.out);
    // 如果 full_args.out 不为 NULL，则减少其引用计数，释放其内存

    /* Wrap and return the output */
    // 包装并返回输出结果

    PyObject *wrap, *wrap_type;
    // 声明两个 PyObject 指针 wrap 和 wrap_type

    if (npy_find_array_wrap(1, &op, &wrap, &wrap_type) < 0) {
        // 如果 npy_find_array_wrap 函数返回值小于 0，则表示包装失败
        Py_DECREF(ret);
        // 减少对 ret 对象的引用计数，释放其内存
        return NULL;
        // 返回空指针表示失败
    }

    /* TODO: Data is mutated, so force_wrap like a normal ufunc call does */
    // 数据已经被修改，因此像正常的 ufunc 调用一样进行强制包装

    PyObject *wrapped_result = npy_apply_wrap(
            (PyObject *)ret, out_obj, wrap, wrap_type, NULL,
            PyArray_NDIM(ret) == 0, NPY_FALSE);
    // 使用 npy_apply_wrap 函数对 ret 进行包装，返回包装后的结果
    Py_DECREF(ret);
    // 减少对 ret 对象的引用计数，释放其内存
    Py_DECREF(wrap);
    // 减少对 wrap 对象的引用计数，释放其内存
    Py_DECREF(wrap_type);
    // 减少对 wrap_type 对象的引用计数，释放其内存
    return wrapped_result;
    // 返回经过包装后的结果对象
fail:
    // 释放引用计数，避免内存泄漏
    Py_XDECREF(signature[0]);
    Py_XDECREF(signature[1]);
    Py_XDECREF(signature[2]);

    // 释放引用计数，避免内存泄漏
    Py_XDECREF(mp);
    // 释放引用计数，避免内存泄漏
    Py_XDECREF(wheremask);
    // 释放引用计数，避免内存泄漏
    Py_XDECREF(indices);
    // 释放引用计数，避免内存泄漏
    Py_XDECREF(full_args.in);
    // 释放引用计数，避免内存泄漏
    Py_XDECREF(full_args.out);
    // 返回空指针，表示函数执行失败
    return NULL;
}


/*
 * 对 `dtype`、`sig` 和 `signature` 进行基本检查，因为只能设置其中一个。
 * 如果使用了 `sig`，则将其写入 `out_signature`（应设置为 `signature_obj`，
 * 因此后续代码只需处理 `signature_obj`）。
 *
 * 注意：此处不增加引用计数！仅复制在参数解析期间获取的借用引用。
 *
 * 此函数不会对输入的 dtype 元组进行任何标准化处理，
 * 目前在数组-ufunc 覆盖检查之后进行。
 */
static int
_check_and_copy_sig_to_signature(
        PyObject *sig_obj, PyObject *signature_obj, PyObject *dtype,
        PyObject **out_signature)
{
    // 初始化 out_signature 为 NULL
    *out_signature = NULL;
    // 如果存在 signature_obj，则将其赋给 out_signature
    if (signature_obj != NULL) {
        *out_signature = signature_obj;
    }

    // 如果存在 sig_obj，则根据情况赋给 out_signature
    if (sig_obj != NULL) {
        if (*out_signature != NULL) {
            // 如果已经有值，不能同时指定 'sig' 和 'signature'，设置错误信息并返回错误标志
            PyErr_SetString(PyExc_TypeError,
                    "cannot specify both 'sig' and 'signature'");
            *out_signature = NULL;
            return -1;
        }
        *out_signature = sig_obj;
    }

    // 如果存在 dtype，则进行相关检查
    if (dtype != NULL) {
        if (*out_signature != NULL) {
            // 如果已经有值，不能同时指定 'signature' 和 'dtype'，设置错误信息并返回错误标志
            PyErr_SetString(PyExc_TypeError,
                    "cannot specify both 'signature' and 'dtype'");
            return -1;
        }
        // dtype 需要转换，但延迟至覆盖检查之后进行
        /* dtype needs to be converted, delay after the override check */
    }
    // 返回成功标志
    return 0;
}


/*
 * 注意：此函数当前允许 DType 类通过，但一般情况下，类（而不是描述符实例）
 * 是首选输入，因此解析应逐渐调整以优先使用类和可能已弃用的实例。
 * （用户通常不会注意到太多，因为 `np.float64` 或 "float64" 通常表示 DType 类，
 * 而不是实例。）
 */
static PyArray_DTypeMeta *
_get_dtype(PyObject *dtype_obj) {
    // 如果是 PyArrayDTypeMeta 类的实例，则增加引用计数并返回该对象的类型
    if (PyObject_TypeCheck(dtype_obj, &PyArrayDTypeMeta_Type)) {
        Py_INCREF(dtype_obj);
        return (PyArray_DTypeMeta *)dtype_obj;
    }
    // 其他情况返回 NULL
    //（注：这里并未完整处理所有可能的输入情况，可能需要进一步扩展）
    return NULL;
}
    else {
        // 如果代码执行到这里，说明不是预期的情况，需要处理自定义数据类型错误
        PyArray_Descr *descr = NULL;
        // 尝试将 Python 对象转换为 NumPy 数据类型描述符
        if (!PyArray_DescrConverter(dtype_obj, &descr)) {
            // 转换失败，返回空指针表示错误
            return NULL;
        }
        // 获取描述符对应的数据类型元信息
        PyArray_DTypeMeta *out = NPY_DTYPE(descr);
        // 如果返回的数据类型不是旧版类型，报类型错误并释放描述符
        if (NPY_UNLIKELY(!NPY_DT_is_legacy(out))) {
            /* TODO: this path was unreachable when added. */
            // 当前分支预计不会执行到，留下备忘以供后续修改
            PyErr_SetString(PyExc_TypeError,
                    "Cannot pass a new user DType instance to the `dtype` or "
                    "`signature` arguments of ufuncs. Pass the DType class "
                    "instead.");
            Py_DECREF(descr);
            return NULL;
        }
        // 如果返回的单例不是描述符本身，检查类型等价性
        else if (NPY_UNLIKELY(out->singleton != descr)) {
            // 此处不警告 `metadata`，但 `units` 是重要的
            if (out->singleton == NULL
                    || !PyArray_EquivTypes(out->singleton, descr)) {
                // 报类型错误，指出 `dtype` 和 `signature` 参数只选择通用的数据类型，不包括字节顺序或时间单位等细节
                PyErr_SetString(PyExc_TypeError,
                        "The `dtype` and `signature` arguments to "
                        "ufuncs only select the general DType and not details "
                        "such as the byte order or time unit. "
                        "You can avoid this error by using the scalar types "
                        "`np.float64` or the dtype string notation.");
                Py_DECREF(descr);
                return NULL;
            }
        }
        // 增加返回对象的引用计数，并释放描述符
        Py_INCREF(out);
        Py_DECREF(descr);
        // 返回成功处理的数据类型元信息对象
        return out;
    }
/*
 * 完成对 DType 签名的转换解析。NumPy 总是仅仅根据类型号来处理传入的描述符或者 DType。
 * `dtype` 参数被解释为第一个输出的 DType（而不是描述符）。
 * 与 `out` 数组的 dtype 不同，它会影响循环选择！
 *
 * 在调用此函数之前，清理并将 `signature` 设为 NULL 是调用者的责任。
 */
static int
_get_fixed_signature(PyUFuncObject *ufunc,
        PyObject *dtype_obj, PyObject *signature_obj,
        PyArray_DTypeMeta **signature)
{
    if (dtype_obj == NULL && signature_obj == NULL) {
        return 0;
    }

    int nin = ufunc->nin, nout = ufunc->nout, nop = nin + nout;

    if (dtype_obj != NULL) {
        if (dtype_obj == Py_None) {
            /* 如果传入 `dtype=None`，则无需执行任何操作 */
            return 0;
        }
        if (nout == 0) {
            /* 可能允许这种情况（NumPy 不支持这样做）？ */
            PyErr_SetString(PyExc_TypeError,
                    "Cannot provide `dtype` when a ufunc has no outputs");
            return -1;
        }
        // 获取 `dtype_obj` 对应的 PyArray_DTypeMeta 结构
        PyArray_DTypeMeta *dtype = _get_dtype(dtype_obj);
        if (dtype == NULL) {
            return -1;
        }
        // 将 dtype 复制给所有输出的 signature 元数据
        for (int i = nin; i < nop; i++) {
            Py_INCREF(dtype);
            signature[i] = dtype;
        }
        Py_DECREF(dtype);
        return 0;
    }

    // 确保 `signature_obj` 不为 NULL
    assert(signature_obj != NULL);
    /* 从元组或字符串 `signature_obj` 填充指定类型到 signature 中 */
}
    # 检查 signature_obj 是否是 PyTuple 类型的对象
    if (PyTuple_Check(signature_obj)) {
        # 获取 PyTuple 对象的大小
        Py_ssize_t n = PyTuple_GET_SIZE(signature_obj);
        # 如果元组大小为 1 且 nop 不等于 1，则进行特殊处理
        if (n == 1 && nop != 1) {
            /*
             * 特殊处理，因为我们已经不推荐使用这个路径了。这个路径可能主要存在是因为
             * `dtype=obj` 被传递为 `(obj,)` 并稍后解析。
             */
            # 如果元组的第一个元素是 None，则设置类型错误并返回 -1
            if (PyTuple_GET_ITEM(signature_obj, 0) == Py_None) {
                PyErr_SetString(PyExc_TypeError,
                        "a single item type tuple cannot contain None.");
                return -1;
            }
            # 如果使用了已经不推荐的路径，返回警告信息，并返回 -1
            if (DEPRECATE("The use of a length 1 tuple for the ufunc "
                          "`signature` is deprecated. Use `dtype` or fill the"
                          "tuple with `None`s.") < 0) {
                return -1;
            }
            /* 使用与 `dtype=` 相同的逻辑 */
            # 调用 _get_fixed_signature 函数处理元组中的第一个元素
            return _get_fixed_signature(ufunc,
                    PyTuple_GET_ITEM(signature_obj, 0), NULL, signature);
        }
        # 如果元组大小不等于 nop，则设置值错误并返回 -1
        if (n != nop) {
            PyErr_Format(PyExc_ValueError,
                    "a type-tuple must be specified of length %d for ufunc '%s'",
                    nop, ufunc_get_name_cstr(ufunc));
            return -1;
        }
        # 遍历元组中的每个元素
        for (int i = 0; i < nop; ++i) {
            # 获取元组中的第 i 个元素
            PyObject *item = PyTuple_GET_ITEM(signature_obj, i);
            # 如果元素是 None，则继续下一个循环
            if (item == Py_None) {
                continue;
            }
            else {
                # 否则，根据元素获取其对应的数据类型
                signature[i] = _get_dtype(item);
                # 如果获取数据类型失败，则返回 -1
                if (signature[i] == NULL) {
                    return -1;
                }
                # 如果当前元素索引小于 nin，并且获取的数据类型是抽象的，则设置类型错误并返回 -1
                else if (i < nin && NPY_DT_is_abstract(signature[i])) {
                    /*
                     * 目前我们不接受抽象的输入签名。这些可能可以通过找到与实际输入的公共数据类型来定义，
                     * 并使用其结果进行提升。
                     */
                    PyErr_SetString(PyExc_TypeError,
                            "Input DTypes to the signature must not be "
                            "abstract.  The behaviour may be defined in the "
                            "future.");
                    return -1;
                }
            }
        }
    }
    else if (PyBytes_Check(signature_obj) || PyUnicode_Check(signature_obj)) {
        // 检查 signature_obj 是否是字节串或者 Unicode 对象

        PyObject *str_object = NULL;

        if (PyBytes_Check(signature_obj)) {
            // 如果 signature_obj 是字节串，将其转换为 Unicode 对象
            str_object = PyUnicode_FromEncodedObject(signature_obj, NULL, NULL);
            if (str_object == NULL) {
                return -1;
            }
        }
        else {
            // 如果 signature_obj 是 Unicode 对象，则增加其引用计数
            Py_INCREF(signature_obj);
            str_object = signature_obj;
        }

        Py_ssize_t length;
        // 获取 Unicode 对象的 UTF-8 编码及其长度
        const char *str = PyUnicode_AsUTF8AndSize(str_object, &length);
        if (str == NULL) {
            Py_DECREF(str_object);
            return -1;
        }

        // 检查字符串长度是否为 1，或者长度是否为 nin + nout + 2 并且符合 "->" 的格式
        if (length != 1 && (length != nin+nout + 2 ||
                            str[nin] != '-' || str[nin+1] != '>')) {
            PyErr_Format(PyExc_ValueError,
                    "a type-string for %s, %d typecode(s) before and %d after "
                    "the -> sign", ufunc_get_name_cstr(ufunc), nin, nout);
            Py_DECREF(str_object);
            return -1;
        }
        // 如果字符串长度为 1，且 nin+nout 不为 1，则发出警告，并返回特定签名的结果
        if (length == 1 && nin+nout != 1) {
            Py_DECREF(str_object);
            if (DEPRECATE("The use of a length 1 string for the ufunc "
                          "`signature` is deprecated. Use `dtype` attribute or "
                          "pass a tuple with `None`s.") < 0) {
                return -1;
            }
            /* `signature="l"` is the same as `dtype="l"` */
            return _get_fixed_signature(ufunc, str_object, NULL, signature);
        }
        else {
            // 解析字符串中的每个字符，获取对应的 NumPy 数据类型描述符
            for (int i = 0; i < nin+nout; ++i) {
                npy_intp istr = i < nin ? i : i+2;
                PyArray_Descr *descr = PyArray_DescrFromType(str[istr]);
                if (descr == NULL) {
                    Py_DECREF(str_object);
                    return -1;
                }
                // 将获取到的描述符转换为 NumPy 数据类型对象，存入 signature 数组
                signature[i] = NPY_DTYPE(descr);
                Py_INCREF(signature[i]);
                Py_DECREF(descr);
            }
            Py_DECREF(str_object);
        }
    }
    else {
        // 如果 signature_obj 不是字节串或 Unicode 对象，则抛出类型错误
        PyErr_SetString(PyExc_TypeError,
                "the signature object to ufunc must be a string or a tuple.");
        return -1;
    }
    return 0;
/*
 * Fill in the actual descriptors used for the operation.  This function
 * supports falling back to the legacy `ufunc->type_resolver`.
 *
 * We guarantee the array-method that all passed in descriptors are of the
 * correct DType instance (i.e. a string can just fetch the length, it doesn't
 * need to "cast" to string first).
 */
static int
resolve_descriptors(int nop,
        PyUFuncObject *ufunc, PyArrayMethodObject *ufuncimpl,
        PyArrayObject *operands[], PyArray_Descr *dtypes[],
        PyArray_DTypeMeta *signature[], PyObject *inputs_tup,
        NPY_CASTING casting)
{
    int retval = -1;
    NPY_CASTING safety;
    PyArray_Descr *original_dtypes[NPY_MAXARGS];

    NPY_UF_DBG_PRINT("Resolving the descriptors\n");

    if (NPY_UNLIKELY(ufuncimpl->resolve_descriptors_with_scalars != NULL)) {
        /*
         * Allow a somewhat more powerful approach which:
         * 1. Has access to scalars (currently only ever Python ones)
         * 2. Can in principle customize `PyArray_CastDescrToDType()`
         *    (also because we want to avoid calling it for the scalars).
         */
        int nin = ufunc->nin;
        PyObject *input_scalars[NPY_MAXARGS];
        
        // Iterate over operands to prepare original_dtypes and input_scalars
        for (int i = 0; i < nop; i++) {
            if (operands[i] == NULL) {
                original_dtypes[i] = NULL;
            }
            else {
                // Store the current operand's dtype in original_dtypes
                original_dtypes[i] = PyArray_DTYPE(operands[i]);
                Py_INCREF(original_dtypes[i]);  // Increase reference count
            }
            
            // Determine if the current input is a scalar of the expected type
            if (i < nin && inputs_tup != NULL) {
                PyObject *input = PyTuple_GET_ITEM(inputs_tup, i);
                input_scalars[i] = signature[i]->scalar_type == Py_TYPE(input) ?
                    input : NULL;
            }
            else {
                input_scalars[i] = NULL;
            }
        }

        npy_intp view_offset = NPY_MIN_INTP;  // Offset for views (currently not used)

        // Call the ufunc's specialized descriptor resolution function
        safety = ufuncimpl->resolve_descriptors_with_scalars(
            ufuncimpl, signature, original_dtypes, input_scalars,
            dtypes, &view_offset
        );

        // Jump to safety checking after descriptor resolution
        goto check_safety;
    }

    // ...
    for (int i = 0; i < nop; ++i) {
        // 检查操作数是否为 NULL
        if (operands[i] == NULL) {
            // 如果是 NULL，则将原始数据类型设置为 NULL
            original_dtypes[i] = NULL;
        }
        else {
            /*
             * 如果数据类型与签名不匹配，则在调用解析之前需要进行调整。
             * 获取操作数 i 的数据类型描述符。
             */
            PyArray_Descr *descr = PyArray_DTYPE(operands[i]);
            // 将数据类型描述符转换为与签名指定的类型匹配的数据类型
            original_dtypes[i] = PyArray_CastDescrToDType(descr, signature[i]);
            // 如果转换失败，则只初始化到 i 位置处，然后跳转到结束标签
            if (original_dtypes[i] == NULL) {
                nop = i;  /* 只有这么多已经初始化 */
                goto finish;
            }
        }
    }

    // 检查是否需要使用 legacy 解析器
    if (ufuncimpl->resolve_descriptors != &wrapped_legacy_resolve_descriptors) {
        /* 默认情况：按照 `ufuncimpl` 的自然意图使用 */
        npy_intp view_offset = NPY_MIN_INTP;  /* 当前被忽略的 */

        // 调用指定的解析器来解析数据类型描述符
        safety = ufuncimpl->resolve_descriptors(ufuncimpl,
                signature, original_dtypes, dtypes, &view_offset);
        // 跳转到安全性检查标签
        goto check_safety;
    }
    else {
        /*
         * 退回到 legacy 解析器，使用 `operands`，仅用于 datetime64/timedelta64
         * 和自定义 ufuncs（在 pyerfa/astropy 中使用）。
         */
        // 调用 ufunc 的类型解析器来解析数据类型
        retval = ufunc->type_resolver(ufunc, casting, operands, NULL, dtypes);
        // 跳转到结束标签
        goto finish;
    }

 check_safety:
    // 检查安全性值是否小于 0
    if (safety < 0) {
        // 跳转到结束标签
        goto finish;
    }
    // 如果安全性不符合要求，则抛出类型错误
    if (NPY_UNLIKELY(PyArray_MinCastSafety(safety, casting) != casting)) {
        /* TODO: 目前不可能到达的部分（特定不安全的循环） */
        PyErr_Format(PyExc_TypeError,
                "The ufunc implementation for %s with the given dtype "
                "signature is not possible under the casting rule %s",
                ufunc_get_name_cstr(ufunc), npy_casting_to_string(casting));
        // 跳转到结束标签
        goto finish;
    }
    // 所有操作成功完成，设置返回值为 0
    retval = 0;

  finish:
    // 释放所有原始数据类型的引用计数
    for (int i = 0; i < nop; i++) {
        Py_XDECREF(original_dtypes[i]);
    }
    // 返回结果值
    return retval;
/**
 * Wraps all outputs and returns the result (which may be NULL on error).
 *
 * Use __array_wrap__ on all outputs
 * if present on one of the input arguments.
 * If present for multiple inputs:
 * use __array_wrap__ of input object with largest
 * __array_priority__ (default = 0.0)
 *
 * Exception:  we should not wrap outputs for items already
 * passed in as output-arguments.  These items should either
 * be left unwrapped or wrapped by calling their own __array_wrap__
 * routine.
 *
 * For each output argument, wrap will be either
 * NULL --- call PyArray_Return() -- default if no output arguments given
 * None --- array-object passed in don't call PyArray_Return
 * method --- the __array_wrap__ method to call.
 *
 * @param ufunc The universal function object
 * @param full_args Original inputs and outputs wrapped in a structure
 * @param subok Whether subclasses are allowed
 * @param result_arrays The array objects for ufunc results (references are stolen)
 */
static PyObject *
replace_with_wrapped_result_and_return(PyUFuncObject *ufunc,
        ufunc_full_args full_args, npy_bool subok,
        PyArrayObject *result_arrays[])
{
    PyObject *result = NULL;
    PyObject *wrap, *wrap_type;

    if (!subok) {
        /* subok=False ignores input wrapping (but not output) */
        Py_INCREF(Py_None);
        wrap = Py_None;
        Py_INCREF(&PyArray_Type);
        wrap_type = (PyObject *)&PyArray_Type;
    }
    else if (npy_find_array_wrap(
            ufunc->nin, PySequence_Fast_ITEMS(full_args.in),
            &wrap, &wrap_type) < 0) {
        goto fail;
    }

    /* wrap outputs */
    NpyUFuncContext context = {
            .ufunc = (PyObject *)ufunc,
            .in = full_args.in, .out = full_args.out};

    if (ufunc->nout != 1) {
        result = PyTuple_New(ufunc->nout);
        if (result == NULL) {
            goto fail;
        }
    }

    for (int out_i = 0; out_i < ufunc->nout; out_i++) {
        context.out_i = out_i;
        PyObject *original_out = NULL;
        if (full_args.out) {
            original_out = PyTuple_GET_ITEM(full_args.out, out_i);
        }

        PyObject *ret_i = npy_apply_wrap(
                (PyObject *)result_arrays[out_i], original_out, wrap, wrap_type,
                /* Always try to return a scalar right now: */
                &context, PyArray_NDIM(result_arrays[out_i]) == 0, NPY_TRUE);
        Py_CLEAR(result_arrays[out_i]);
        if (ret_i == NULL) {
            goto fail;
        }
        /* When we are not returning a tuple, this is the result: */
        if (result == NULL) {
            result = ret_i;
            goto finish;
        }
        PyTuple_SET_ITEM(result, out_i, ret_i);
    }

  finish:
    Py_DECREF(wrap);
    Py_DECREF(wrap_type);
    return result;
  fail:
    /* Fail path ensures result_arrays are fully cleared */
    Py_XDECREF(result);
    Py_DECREF(wrap);
    Py_DECREF(wrap_type);
    for (int i = 0; i < ufunc->nout; i++) {
        Py_CLEAR(result_arrays[i]);
    }
    return NULL;
}
/*
 * Main ufunc call implementation.
 *
 * This implementation makes use of the "fastcall" way of passing keyword
 * arguments and is called directly from `ufunc_generic_vectorcall` when
 * Python has `tp_vectorcall` (Python 3.8+).
 * If `tp_vectorcall` is not available, the dictionary `kwargs` are unpacked in
 * `ufunc_generic_call` with fairly little overhead.
 */
static PyObject *
ufunc_generic_fastcall(PyUFuncObject *ufunc,
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames,
        npy_bool outer)
{
    int errval;
    int nin = ufunc->nin, nout = ufunc->nout, nop = ufunc->nargs;

    /* All following variables are cleared in the `fail` error path */
    ufunc_full_args full_args;
    PyArrayObject *wheremask = NULL;

    PyArray_DTypeMeta *signature[NPY_MAXARGS];
    PyArrayObject *operands[NPY_MAXARGS];
    PyArray_DTypeMeta *operand_DTypes[NPY_MAXARGS];
    PyArray_Descr *operation_descrs[NPY_MAXARGS];
    /* Initialize all arrays (we usually only need a small part) */
    memset(signature, 0, nop * sizeof(*signature));
    memset(operands, 0, nop * sizeof(*operands));
    memset(operand_DTypes, 0, nop * sizeof(*operation_descrs));
    memset(operation_descrs, 0, nop * sizeof(*operation_descrs));

    /*
     * Note that the input (and possibly output) arguments are passed in as
     * positional arguments. We extract these first and check for `out`
     * passed by keyword later.
     * Outputs and inputs are stored in `full_args.in` and `full_args.out`
     * as tuples (or NULL when no outputs are passed).
     */

    /* Check number of arguments */
    if (NPY_UNLIKELY((len_args < nin) || (len_args > nop))) {
        PyErr_Format(PyExc_TypeError,
                "%s() takes from %d to %d positional arguments but "
                "%zd were given",
                ufunc_get_name_cstr(ufunc) , nin, nop, len_args);
        return NULL;
    }

    /* Fetch input arguments. */
    full_args.in = PyArray_TupleFromItems(ufunc->nin, args, 0);
    if (full_args.in == NULL) {
        return NULL;
    }

    /*
     * If there are more arguments, they define the out args. Otherwise
     * full_args.out is NULL for now, and the `out` kwarg may still be passed.
     */
    npy_bool out_is_passed_by_position = len_args > nin;
    if (out_is_passed_by_position) {
        npy_bool all_none = NPY_TRUE;

        full_args.out = PyTuple_New(nout);
        if (full_args.out == NULL) {
            goto fail;
        }
        for (int i = nin; i < nop; i++) {
            PyObject *tmp;
            if (i < (int)len_args) {
                tmp = args[i];
                if (tmp != Py_None) {
                    all_none = NPY_FALSE;
                }
            }
            else {
                tmp = Py_None;
            }
            Py_INCREF(tmp);
            PyTuple_SET_ITEM(full_args.out, i-nin, tmp);
        }
        if (all_none) {
            Py_SETREF(full_args.out, NULL);
        }
    }


注释：
    else {
        // 如果没有输出参数，则设置为 NULL
        full_args.out = NULL;
    }

    /*
     * 我们已经提取了输入参数（但尚未转换）。
     * 为了简化覆盖，提取所有其他参数（仅作为对象）。
     */
    PyObject *out_obj = NULL, *where_obj = NULL;
    PyObject *axes_obj = NULL, *axis_obj = NULL;
    PyObject *keepdims_obj = NULL, *casting_obj = NULL, *order_obj = NULL;
    PyObject *subok_obj = NULL, *signature_obj = NULL, *sig_obj = NULL;
    PyObject *dtype_obj = NULL;

    /* 如果存在关键字参数，则跳过解析，没有其他事情可做 */
    if (kwnames != NULL) {
        // 如果核心未启用，则准备参数解析器
        if (!ufunc->core_enabled) {
            NPY_PREPARE_ARGPARSER;

            // 解析参数，如果失败则跳转到失败处理标签
            if (npy_parse_arguments(ufunc->name, args + len_args, 0, kwnames,
                    "$out", NULL, &out_obj,
                    "$where", NULL, &where_obj,
                    "$casting", NULL, &casting_obj,
                    "$order", NULL, &order_obj,
                    "$subok", NULL, &subok_obj,
                    "$dtype", NULL, &dtype_obj,
                    "$signature", NULL, &signature_obj,
                    "$sig", NULL, &sig_obj,
                    NULL, NULL, NULL) < 0) {
                goto fail;
            }
        }
        else {
            NPY_PREPARE_ARGPARSER;

            // 解析参数，如果失败则跳转到失败处理标签
            if (npy_parse_arguments(ufunc->name, args + len_args, 0, kwnames,
                    "$out", NULL, &out_obj,
                    "$axes", NULL, &axes_obj,
                    "$axis", NULL, &axis_obj,
                    "$keepdims", NULL, &keepdims_obj,
                    "$casting", NULL, &casting_obj,
                    "$order", NULL, &order_obj,
                    "$subok", NULL, &subok_obj,
                    "$dtype", NULL, &dtype_obj,
                    "$signature", NULL, &signature_obj,
                    "$sig", NULL, &sig_obj,
                    NULL, NULL, NULL) < 0) {
                goto fail;
            }
            // 如果同时指定了 'axis' 和 'axes'，则抛出类型错误并跳转到失败处理标签
            if (NPY_UNLIKELY((axes_obj != NULL) && (axis_obj != NULL))) {
                PyErr_SetString(PyExc_TypeError,
                        "cannot specify both 'axis' and 'axes'");
                goto fail;
            }
        }

        /* 处理通过关键字传递的 `out` 参数 */
        if (out_obj != NULL) {
            // 如果 `out` 既作为位置参数又作为关键字参数传递，则抛出类型错误并跳转到失败处理标签
            if (out_is_passed_by_position) {
                PyErr_SetString(PyExc_TypeError,
                        "cannot specify 'out' as both a "
                        "positional and keyword argument");
                goto fail;
            }
            // 设置完整的输出参数
            if (_set_full_args_out(nout, out_obj, &full_args) < 0) {
                goto fail;
            }
        }
        /*
         * 只应传递 signature、sig 或 dtype 中的一个。如果传递了 `sig`，
         * 则将其复制到 `signature_obj` 中（这些是借用引用）。
         */
        if (_check_and_copy_sig_to_signature(
                sig_obj, signature_obj, dtype_obj, &signature_obj) < 0) {
            goto fail;
        }
    }
    // 声明一个指向字符的指针变量 method
    char *method;
    // 如果 outer 为假（即外部调用不需要特定方法），则 method 指向 "__call__"
    if (!outer) {
        method = "__call__";
    }
    // 否则，method 指向字符串 "outer"
    else {
        method = "outer";
    }
    /* 现在我们已经获取了检查是否有重写所需的所有信息 */
    // 声明一个指向 PyObject 的指针变量 override，并初始化为 NULL
    PyObject *override = NULL;
    // 调用 PyUFunc_CheckOverride 函数检查是否有重写，传入相应的参数
    errval = PyUFunc_CheckOverride(ufunc, method,
            full_args.in, full_args.out, where_obj,
            args, len_args, kwnames, &override);
    // 如果出现错误（errval 非零），跳转到 fail 标签处
    if (errval) {
        goto fail;
    }
    // 如果 override 不为空，则释放 full_args.in，并返回 override
    else if (override) {
        Py_DECREF(full_args.in);
        Py_XDECREF(full_args.out);
        return override;
    }

    // 如果 outer 为真，则需要对输入参数进行特殊准备（如扩展维度）
    if (outer) {
        /* Outer 使用特殊的输入准备（扩展维度） */
        // 调用 prepare_input_arguments_for_outer 函数，处理 full_args.in，并将结果赋给 new_in
        PyObject *new_in = prepare_input_arguments_for_outer(full_args.in, ufunc);
        // 如果处理失败（new_in 为空），跳转到 fail 标签处
        if (new_in == NULL) {
            goto fail;
        }
        // 设置 full_args.in 指向 new_in
        Py_SETREF(full_args.in, new_in);
    }

    /*
     * 将传递的 dtype 或 signature 解析为包含 PyArray_DTypeMeta 和/或 None 的数组。
     */
    // 如果 _get_fixed_signature 函数返回值小于 0，跳转到 fail 标签处
    if (_get_fixed_signature(ufunc,
            dtype_obj, signature_obj, signature) < 0) {
        goto fail;
    }

    // 初始化一些变量
    NPY_ORDER order = NPY_KEEPORDER;
    NPY_CASTING casting = NPY_DEFAULT_ASSIGN_CASTING;
    npy_bool subok = NPY_TRUE;
    int keepdims = -1;  /* 需要知道是否传递了该参数 */
    npy_bool force_legacy_promotion;
    npy_bool allow_legacy_promotion;
    npy_bool promoting_pyscalars;
    // 如果 convert_ufunc_arguments 函数返回值小于 0，跳转到 fail 标签处
    if (convert_ufunc_arguments(ufunc,
            /* 提取操作数相关信息： */
            full_args, operands,
            operand_DTypes,
            &force_legacy_promotion, &allow_legacy_promotion,
            &promoting_pyscalars,
            /* 提取一般信息： */
            order_obj, &order,
            casting_obj, &casting,
            subok_obj, &subok,
            where_obj, &wheremask,
            keepdims_obj, &keepdims) < 0) {
        goto fail;
    }

    /*
     * 注意，推广的一部分是完成签名
     * （到这里为止，它只表示固定部分并且通常是 NULL）。
     *
     * 在推广之后，我们可以将以下逻辑推入 ArrayMethod 的未来。
     * 现在，我们在这里执行。类型解析步骤可以在 ufunc 和 gufunc 代码之间共享。
     */
    // 调用 promote_and_get_ufuncimpl 函数推广并获取 ufuncimpl 对象
    PyArrayMethodObject *ufuncimpl = promote_and_get_ufuncimpl(ufunc,
            operands, signature,
            operand_DTypes, force_legacy_promotion, allow_legacy_promotion,
            promoting_pyscalars, NPY_FALSE);
    // 如果 ufuncimpl 为空，跳转到 fail 标签处
    if (ufuncimpl == NULL) {
        goto fail;
    }

    /* 查找操作的正确描述符 */
    // 如果 resolve_descriptors 函数返回值小于 0，跳转到 fail 标签处
    if (resolve_descriptors(nop, ufunc, ufuncimpl,
            operands, operation_descrs, signature, full_args.in, casting) < 0) {
        goto fail;
    }
    if (promoting_pyscalars) {
        /*
         * Python integers need to be cast specially.  For other python
         * scalars it does not hurt either.  It would be nice to never create
         * the array in this case, but that is difficult until value-based
         * promotion rules are gone.  (After that, we may get away with using
         * dummy arrays rather than real arrays for the legacy resolvers.)
         */
        // 遍历操作数列表，处理Python整数和其他标量的类型提升
        for (int i = 0; i < nin; i++) {
            // 获取操作数的原始标志位
            int orig_flags = PyArray_FLAGS(operands[i]);
            // 如果操作数不是Python字面量，则跳过
            if (!(orig_flags & NPY_ARRAY_WAS_PYTHON_LITERAL)) {
                continue;
            }
            /*
             * If descriptor matches, no need to convert, but integers may
             * have been too large.
             */
            // 如果描述符匹配且未替换整数类型，则跳过
            if (!(orig_flags & NPY_ARRAY_WAS_INT_AND_REPLACED)
                    && PyArray_EquivTypes(
                        PyArray_DESCR(operands[i]), operation_descrs[i])) {
                continue;
            }
            /* Otherwise, replace the operand with a new array */
            // 否则，使用新数组替换操作数
            PyArray_Descr *descr = operation_descrs[i];
            Py_INCREF(descr);
            PyArrayObject *new = (PyArrayObject *)PyArray_NewFromDescr(
                    &PyArray_Type, descr, 0, NULL, NULL, NULL, 0, NULL);
            Py_SETREF(operands[i], new);
            // 检查操作数是否成功替换
            if (operands[i] == NULL) {
                goto fail;
            }

            // 获取完整参数元组中的值
            PyObject *value = PyTuple_GET_ITEM(full_args.in, i);
            // 将值设置到新数组中
            if (PyArray_SETITEM(new, PyArray_BYTES(operands[i]), value) < 0) {
                goto fail;
            }
        }
    }

    /*
     * Do the final preparations and call the inner-loop.
     */
    // 如果未启用核心函数，则调用通用函数内部实现
    if (!ufunc->core_enabled) {
        errval = PyUFunc_GenericFunctionInternal(ufunc, ufuncimpl,
                operation_descrs, operands, casting, order,
                wheremask);
    }
    // 否则，调用广义函数内部实现
    else {
        errval = PyUFunc_GeneralizedFunctionInternal(ufunc, ufuncimpl,
                operation_descrs, operands, casting, order,
                axis_obj, axes_obj, keepdims);
    }
    // 检查函数调用是否成功
    if (errval < 0) {
        goto fail;
    }

    /*
     * Clear all variables which are not needed any further.
     * (From here on, we cannot `goto fail` any more.)
     */
    // 释放不再需要的所有变量
    Py_XDECREF(wheremask);
    for (int i = 0; i < nop; i++) {
        Py_DECREF(signature[i]);
        Py_XDECREF(operand_DTypes[i]);
        Py_DECREF(operation_descrs[i]);
        if (i < nin) {
            Py_DECREF(operands[i]);
        }
    }
    /* The following steals the references to the outputs: */
    // 替换结果并返回包装后的结果对象
    PyObject *result = replace_with_wrapped_result_and_return(ufunc,
            full_args, subok, operands+nin);
    // 释放完整参数中的输入和输出对象
    Py_XDECREF(full_args.in);
    Py_XDECREF(full_args.out);

    // 返回操作的结果对象
    return result;
fail:
    Py_XDECREF(full_args.in);  // 释放输入参数中的'in'对象引用
    Py_XDECREF(full_args.out);  // 释放输入参数中的'out'对象引用
    Py_XDECREF(wheremask);      // 释放 wheremask 对象引用
    for (int i = 0; i < ufunc->nargs; i++) {
        Py_XDECREF(operands[i]);            // 释放操作数数组中的每个对象引用
        Py_XDECREF(signature[i]);           // 释放签名数组中的每个对象引用
        Py_XDECREF(operand_DTypes[i]);      // 释放操作数数据类型数组中的每个对象引用
        Py_XDECREF(operation_descrs[i]);    // 释放操作描述符数组中的每个对象引用
    }
    return NULL;  // 返回空指针，标志着函数执行失败
}


/*
 * Implement vectorcallfunc which should be defined with Python 3.8+.
 * In principle this could be backported, but the speed gain seems moderate
 * since ufunc calls often do not have keyword arguments and always have
 * a large overhead. The only user would potentially be cython probably.
 */
static PyObject *
ufunc_generic_vectorcall(PyObject *ufunc,
        PyObject *const *args, size_t len_args, PyObject *kwnames)
{
    /*
     * Unlike METH_FASTCALL, `len_args` may have a flag to signal that
     * args[-1] may be (temporarily) used. So normalize it here.
     */
    return ufunc_generic_fastcall((PyUFuncObject *)ufunc,
            args, PyVectorcall_NARGS(len_args), kwnames, NPY_FALSE);
}


/*UFUNC_API*/
NPY_NO_EXPORT int
PyUFunc_ReplaceLoopBySignature(PyUFuncObject *func,
                               PyUFuncGenericFunction newfunc,
                               const int *signature,
                               PyUFuncGenericFunction *oldfunc)
{
    int i, j;
    int res = -1;
    /* Find the location of the matching signature */
    for (i = 0; i < func->ntypes; i++) {
        for (j = 0; j < func->nargs; j++) {
            if (signature[j] != func->types[i*func->nargs+j]) {
                break;
            }
        }
        if (j < func->nargs) {
            continue;  // 如果找到的签名不匹配，继续查找下一个
        }
        if (oldfunc != NULL) {
            *oldfunc = func->functions[i];  // 将原始函数指针存储到 oldfunc 中
        }
        func->functions[i] = newfunc;  // 用新函数指针替换原始函数指针
        res = 0;  // 标志函数替换成功
        break;
    }
    return res;  // 返回替换结果
}

/*UFUNC_API*/
NPY_NO_EXPORT PyObject *
PyUFunc_FromFuncAndData(PyUFuncGenericFunction *func, void *const *data,
                        const char *types, int ntypes,
                        int nin, int nout, int identity,
                        const char *name, const char *doc, int unused)
{
    return PyUFunc_FromFuncAndDataAndSignature(func, data, types, ntypes,
        nin, nout, identity, name, doc, unused, NULL);
}

/*UFUNC_API*/
NPY_NO_EXPORT PyObject *
PyUFunc_FromFuncAndDataAndSignature(PyUFuncGenericFunction *func, void *const *data,
                                     const char *types, int ntypes,
                                     int nin, int nout, int identity,
                                     const char *name, const char *doc,
                                     int unused, const char *signature)
{
    return PyUFunc_FromFuncAndDataAndSignatureAndIdentity(
        func, data, types, ntypes, nin, nout, identity, name, doc,
        unused, signature, NULL);
}

/*UFUNC_API*/
NPY_NO_EXPORT PyObject *
PyUFunc_FromFuncAndDataAndSignatureAndIdentity(PyUFuncGenericFunction *func, void *const *data,
                                     const char *types, int ntypes,
                                     int nin, int nout, int identity,
                                     const char *name, const char *doc,
                                     const int unused, const char *signature,
                                     PyObject *identity_value)
{
    // 检查输入和输出的总数是否超过最大允许值
    if (nin + nout > NPY_MAXARGS) {
        // 报错，返回空指针
        PyErr_Format(PyExc_ValueError,
                     "Cannot construct a ufunc with more than %d operands "
                     "(requested number were: inputs = %d and outputs = %d)",
                     NPY_MAXARGS, nin, nout);
        return NULL;
    }

    // 分配内存并初始化一个新的 PyUFuncObject 结构
    ufunc = PyObject_GC_New(PyUFuncObject, &PyUFunc_Type);

    /*
     * 我们在这里使用 GC_New 来创建 ufunc->obj，但是不调用 GC_Track，
     * 因为在函数结束时 ufunc->obj 仍然是 NULL。
     * 参见 ufunc_frompyfunc 函数，在那里设置 ufunc->obj 并调用 GC_Track。
     */
    if (ufunc == NULL) {
        return NULL;
    }

    // 设置 ufunc 结构体的各个字段
    ufunc->nin = nin;
    ufunc->nout = nout;
    ufunc->nargs = nin + nout;
    ufunc->identity = identity;

    // 如果 ufunc 的 identity 是 PyUFunc_IdentityValue，则增加引用计数并设置 identity_value
    if (ufunc->identity == PyUFunc_IdentityValue) {
        Py_INCREF(identity_value);
        ufunc->identity_value = identity_value;
    } else {
        ufunc->identity_value = NULL;
    }

    ufunc->functions = func;
    ufunc->data = data;
    ufunc->types = types;
    ufunc->ntypes = ntypes;
    ufunc->core_signature = NULL;
    ufunc->core_enabled = 0;
    ufunc->obj = NULL;
    ufunc->core_num_dims = NULL;
    ufunc->core_num_dim_ix = 0;
    ufunc->core_offsets = NULL;
    ufunc->core_dim_ixs = NULL;
    ufunc->core_dim_sizes = NULL;
    ufunc->core_dim_flags = NULL;
    ufunc->userloops = NULL;
    ufunc->ptr = NULL;
    ufunc->vectorcall = &ufunc_generic_vectorcall;
    ufunc->reserved1 = 0;
    ufunc->iter_flags = 0;

    // 设置类型解析器和内部循环选择函数
    ufunc->type_resolver = &PyUFunc_DefaultTypeResolver;

    ufunc->op_flags = NULL;
    ufunc->_loops = NULL;

    // 如果输入和输出的总数不为零，则创建 _dispatch_cache
    if (nin + nout != 0) {
        ufunc->_dispatch_cache = PyArrayIdentityHash_New(nin + nout);
        if (ufunc->_dispatch_cache == NULL) {
            Py_DECREF(ufunc);
            return NULL;
        }
    } else {
        // 否则设置 _dispatch_cache 为 NULL
        ufunc->_dispatch_cache = NULL;
    }

    // 创建一个空的 _loops 列表
    ufunc->_loops = PyList_New(0);
    if (ufunc->_loops == NULL) {
        Py_DECREF(ufunc);
        return NULL;
    }

    // 设置 ufunc 的名称和文档字符串
    if (name == NULL) {
        ufunc->name = "?";
    } else {
        ufunc->name = name;
    }
    ufunc->doc = doc;

    // 分配内存以保存操作标志位
    ufunc->op_flags = PyArray_malloc(sizeof(npy_uint32) * ufunc->nargs);
    if (ufunc->op_flags == NULL) {
        Py_DECREF(ufunc);
        return PyErr_NoMemory();
    }
    // 将 ufunc 结构体中的 op_flags 数组初始化为零，长度为 nargs
    memset(ufunc->op_flags, 0, sizeof(npy_uint32)*ufunc->nargs);

    // 如果给定了函数签名 signature，则解析该签名并设置给 ufunc
    if (signature != NULL) {
        // 解析函数签名，如果解析失败，则释放 ufunc 并返回空指针
        if (_parse_signature(ufunc, signature) != 0) {
            Py_DECREF(ufunc);
            return NULL;
        }
    }

    // 获取 ufunc 结构体中的 types 字符串的指针
    const char *curr_types = ufunc->types;
    // 迭代 ntypes * (nin + nout) 次，每次增加 nin + nout
    for (int i = 0; i < ntypes * (nin + nout); i += nin + nout) {
        /*
         * 在此处添加所有旧版包装循环。通常情况下这是不必要的，
         * 但是有时有意义。这也有助于或者是必需的，以避免出现二义性循环，
         * 例如：`OO->?` 和 `OO->O`，理论上可能会选择错误的循环。
         */
        PyObject *info;
        PyArray_DTypeMeta *op_dtypes[NPY_MAXARGS];
        // 对于每个参数，根据类型号获取 PyArray_DTypeMeta 结构
        for (int arg = 0; arg < nin + nout; arg++) {
            op_dtypes[arg] = PyArray_DTypeFromTypeNum(curr_types[arg]);
            /* 这些 DTypes 是不可变的并且已经增加了 INCREFs，因此借用它们 */
            Py_DECREF(op_dtypes[arg]);
        }
        // 移动到下一个类型组的 types 字符串
        curr_types += nin + nout;

        // 添加并返回一个旧版包装的 ufunc 循环信息
        info = add_and_return_legacy_wrapping_ufunc_loop(ufunc, op_dtypes, 1);
        // 如果添加循环失败，则释放 ufunc 并返回空指针
        if (info == NULL) {
            Py_DECREF(ufunc);
            return NULL;
        }
    }
    /*
     * TODO: 我尝试在这里添加一个默认的类型提升器（对于某些特殊情况可能是全部对象，
     *       或者全部同类）。这些是合理的默认值，但是缩短了一个被弃用的 SciPy 循环，
     *       其中同类循环 `ddd->d` 被弃用，但非同类循环 `dld->d` 应当被选择。
     *       默认的提升器确实是一个合理的默认值，但是改变了该行为。
     *       另一个问题由于日期时间类型解析错误而出现，这导致 `timedelta.sum(dtype="f8")`
     *       返回日期时间（而不是浮点数或错误），这可能是错误的，但是...
     */
    // 返回 ufunc 的 PyObject 指针类型转换结果
    return (PyObject *)ufunc;
/*
 * This structure defines a basic object with a void pointer.
 * It is part of a C API and extends PyObject_HEAD.
 */
typedef struct {
    PyObject_HEAD
    void *c_obj;
} _simple_cobj;

/*
 * Macro to set the void pointer in _simple_cobj structure.
 * This macro simplifies setting the c_obj field in the structure.
 */
#define _SETCPTR(cobj, val) ((_simple_cobj *)(cobj))->c_obj = (val)

/*
 * Compare two arrays of integer types element-wise.
 * Returns 1 if arg1 > arg2, 0 if arg1 == arg2, and -1 if arg1 < arg2.
 */
static int
cmp_arg_types(int *arg1, int *arg2, int n)
{
    for (; n > 0; n--, arg1++, arg2++) {
        if (PyArray_EquivTypenums(*arg1, *arg2)) {
            continue;
        }
        if (PyArray_CanCastSafely(*arg1, *arg2)) {
            return -1;
        }
        return 1;
    }
    return 0;
}

/*
 * Free a linked-list structure of PyUFunc_Loop1d.
 * This function is used to deallocate memory when the structure is destroyed.
 */
static inline void
_free_loop1d_list(PyUFunc_Loop1d *data)
{
    int i;

    while (data != NULL) {
        PyUFunc_Loop1d *next = data->next;
        PyArray_free(data->arg_types);

        if (data->arg_dtypes != NULL) {
            for (i = 0; i < data->nargs; i++) {
                Py_DECREF(data->arg_dtypes[i]);
            }
            PyArray_free(data->arg_dtypes);
        }

        PyArray_free(data);
        data = next;
    }
}

/*
 * Free function for PyCapsule object holding PyUFunc_Loop1d data.
 * It extracts the pointer from the capsule and calls _free_loop1d_list to free the data.
 */
static void
_loop1d_list_free(PyObject *ptr)
{
    PyUFunc_Loop1d *data = (PyUFunc_Loop1d *)PyCapsule_GetPointer(ptr, NULL);
    _free_loop1d_list(data);
}

/*
 * Register a 1-d loop function for a specific user-defined dtype with a ufunc object.
 * This function allows a 1-d loop to be associated with PyArray_Descr objects.
 * It is similar to RegisterLoopForType but works with structured array dtypes or custom dtypes.
 *
 * Parameters:
 * ufunc      - ufunc object created from PyUFunc_FromFuncAndData
 * user_dtype - dtype that the ufunc will be registered with
 * function   - 1-d loop function pointer
 * arg_dtypes - array of dtype objects describing the ufunc operands
 * data       - arbitrary data pointer passed to the loop function
 *
 * Returns 0 on success, -1 for failure.
 */
/*UFUNC_API*/
NPY_NO_EXPORT int
PyUFunc_RegisterLoopForDescr(PyUFuncObject *ufunc,
                            PyArray_Descr *user_dtype,
                            PyUFuncGenericFunction function,
                            PyArray_Descr **arg_dtypes,
                            void *data)
{
    int i;
    int result = 0;
    int *arg_typenums;
    PyObject *key, *cobj;

    // Check if user_dtype is NULL
    if (user_dtype == NULL) {
        PyErr_SetString(PyExc_TypeError,
            "unknown user defined struct dtype");
        return -1;
    }

    // Create a key from the type number of user_dtype
    key = PyLong_FromLong((long) user_dtype->type_num);
    if (key == NULL) {
        return -1;
    }
    # 分配内存以存储参数类型数的数组
    arg_typenums = PyArray_malloc(ufunc->nargs * sizeof(int));
    # 检查内存分配是否成功
    if (arg_typenums == NULL) {
        Py_DECREF(key);  # 减少对 key 的引用计数
        PyErr_NoMemory();  # 抛出内存错误异常
        return -1;  # 返回错误代码
    }
    # 如果用户提供了参数数据类型
    if (arg_dtypes != NULL) {
        # 将每个参数的类型数存储到 arg_typenums 数组中
        for (i = 0; i < ufunc->nargs; i++) {
            arg_typenums[i] = arg_dtypes[i]->type_num;
        }
    }
    # 如果未提供参数数据类型
    else {
        # 使用用户定义的数据类型的类型数填充 arg_typenums 数组
        for (i = 0; i < ufunc->nargs; i++) {
            arg_typenums[i] = user_dtype->type_num;
        }
    }

    # 注册用户定义函数的循环类型
    result = PyUFunc_RegisterLoopForType(ufunc, user_dtype->type_num,
        function, arg_typenums, data);

    # 如果注册成功
    if (result == 0) {
        # 获取用户循环的对象
        cobj = PyDict_GetItemWithError(ufunc->userloops, key);
        # 如果获取对象时出错
        if (cobj == NULL && PyErr_Occurred()) {
            result = -1;  # 返回错误代码
        }
        # 如果未找到对象
        else if (cobj == NULL) {
            PyErr_SetString(PyExc_KeyError,
                "userloop for user dtype not found");  # 设置关键字错误异常
            result = -1;  # 返回错误代码
        }
        # 如果找到对象
        else {
            int cmp = 1;  # 初始化比较结果为 1
            PyUFunc_Loop1d *current = PyCapsule_GetPointer(cobj, NULL);  # 获取当前循环对象的指针
            # 如果获取失败
            if (current == NULL) {
                result = -1;  # 返回错误代码
                goto done;  # 跳转到完成标签
            }
            # 循环查找匹配的循环对象
            while (current != NULL) {
                cmp = cmp_arg_types(current->arg_types,
                    arg_typenums, ufunc->nargs);  # 比较参数类型
                # 如果找到匹配且未指定参数数据类型
                if (cmp >= 0 && current->arg_dtypes == NULL) {
                    break;  # 跳出循环
                }
                current = current->next;  # 继续查找下一个循环对象
            }
            # 如果找到匹配的循环且未指定参数数据类型
            if (cmp == 0 && current != NULL && current->arg_dtypes == NULL) {
                # 分配内存以存储参数数据类型的数组
                current->arg_dtypes = PyArray_malloc(ufunc->nargs *
                    sizeof(PyArray_Descr*));
                # 如果内存分配失败
                if (current->arg_dtypes == NULL) {
                    PyErr_NoMemory();  # 抛出内存错误异常
                    result = -1;  # 返回错误代码
                    goto done;  # 跳转到完成标签
                }
                # 如果提供了参数数据类型
                else if (arg_dtypes != NULL) {
                    # 将每个参数的数据类型复制到当前循环对象的数组中
                    for (i = 0; i < ufunc->nargs; i++) {
                        current->arg_dtypes[i] = arg_dtypes[i];
                        Py_INCREF(current->arg_dtypes[i]);  # 增加数据类型对象的引用计数
                    }
                }
                # 如果未提供参数数据类型
                else {
                    # 使用用户定义的数据类型复制到当前循环对象的数组中
                    for (i = 0; i < ufunc->nargs; i++) {
                        current->arg_dtypes[i] = user_dtype;
                        Py_INCREF(current->arg_dtypes[i]);  # 增加数据类型对象的引用计数
                    }
                }
                current->nargs = ufunc->nargs;  # 设置参数数量
            }
            # 如果已注册循环
            else {
                PyErr_SetString(PyExc_RuntimeError,
                    "loop already registered");  # 设置运行时错误异常
                result = -1;  # 返回错误代码
            }
        }
    }
    // 释放 arg_typenums 指向的内存
    PyArray_free(arg_typenums);

    // 减少对 key 的引用计数
    Py_DECREF(key);

    // 返回 result 结果
    return result;
}

/*UFUNC_API*/
NPY_NO_EXPORT int
PyUFunc_RegisterLoopForType(PyUFuncObject *ufunc,
                            int usertype,
                            PyUFuncGenericFunction function,
                            const int *arg_types,
                            void *data)
{
    PyArray_Descr *descr;
    PyUFunc_Loop1d *funcdata;
    PyObject *key, *cobj;
    PyArray_DTypeMeta *signature[NPY_MAXARGS];
    PyObject *signature_tuple = NULL;
    int i;
    int *newtypes=NULL;

    // 根据 usertype 获取其对应的描述符
    descr = PyArray_DescrFromType(usertype);
    // 检查 usertype 是否合法
    if ((usertype < NPY_USERDEF && usertype != NPY_VOID) || (descr==NULL)) {
        // 设置错误信息并返回 -1
        PyErr_SetString(PyExc_TypeError, "unknown user-defined type");
        return -1;
    }
    // 减少描述符的引用计数
    Py_DECREF(descr);

    // 如果 ufunc->userloops 为空，创建一个新的字典对象
    if (ufunc->userloops == NULL) {
        ufunc->userloops = PyDict_New();
    }

    // 创建一个长整型对象作为字典的键值
    key = PyLong_FromLong((long) usertype);
    if (key == NULL) {
        return -1;
    }

    // 分配内存以存储 PyUFunc_Loop1d 结构体
    funcdata = PyArray_malloc(sizeof(PyUFunc_Loop1d));
    if (funcdata == NULL) {
        goto fail;
    }

    // 分配内存以存储新类型的数组
    newtypes = PyArray_malloc(sizeof(int)*ufunc->nargs);
    if (newtypes == NULL) {
        goto fail;
    }

    // 如果提供了 arg_types，则使用它们创建签名；否则使用 usertype 创建签名
    if (arg_types != NULL) {
        for (i = 0; i < ufunc->nargs; i++) {
            newtypes[i] = arg_types[i];
            signature[i] = PyArray_DTypeFromTypeNum(arg_types[i]);
            Py_DECREF(signature[i]);  /* DType can't be deleted... */
        }
    }
    else {
        for (i = 0; i < ufunc->nargs; i++) {
            newtypes[i] = usertype;
            signature[i] = PyArray_DTypeFromTypeNum(usertype);
            Py_DECREF(signature[i]);  /* DType can't be deleted... */
        }
    }

    // 创建一个元组，其中包含所有签名的对象引用
    signature_tuple = PyArray_TupleFromItems(
            ufunc->nargs, (PyObject **)signature, 0);
    if (signature_tuple == NULL) {
        goto fail;
    }

    /*
     * 将循环添加到所有循环和提升的列表中。如果等效循环已添加，则跳过。
     * 即使如此，ufunc 仍然被修改：遗留的 ArrayMethod 已经从 ufunc 中查找内部循环
     * （并且在下面被替换！）。
     * 如果现有的循环不是遗留的 ArrayMethod，则当前会引发：
     * 不应该用旧式循环替换新式循环。
     */
    int add_new_loop = 1;
    for (Py_ssize_t j = 0; j < PyList_GET_SIZE(ufunc->_loops); j++) {
        // 获取 ufunc 对象的 _loops 列表的第 j 个元素
        PyObject *item = PyList_GET_ITEM(ufunc->_loops, j);
        // 获取 item 元组的第一个元素
        PyObject *existing_tuple = PyTuple_GET_ITEM(item, 0);

        // 比较 existing_tuple 和 signature_tuple 是否相等
        int cmp = PyObject_RichCompareBool(existing_tuple, signature_tuple, Py_EQ);
        // 如果比较失败，跳转到错误处理部分
        if (cmp < 0) {
            goto fail;
        }
        // 如果不相等，继续下一次循环
        if (!cmp) {
            continue;
        }
        // 获取 item 元组的第二个元素
        PyObject *registered = PyTuple_GET_ITEM(item, 1);
        // 检查 registered 是否为 PyArrayMethod_Type 类型，并且其 get_strided_loop 函数指针不等于 &get_wrapped_legacy_ufunc_loop
        if (!PyObject_TypeCheck(registered, &PyArrayMethod_Type) || (
                (PyArrayMethodObject *)registered)->get_strided_loop !=
                        &get_wrapped_legacy_ufunc_loop) {
            // 抛出类型错误异常，指示已为 ufunc 和特定数据类型注册了不兼容的循环
            PyErr_Format(PyExc_TypeError,
                    "A non-compatible loop was already registered for "
                    "ufunc %s and DTypes %S.",
                    ufunc_get_name_cstr(ufunc), signature_tuple);
            goto fail;
        }
        /* The loop was already added */
        // 标记新循环不需要添加
        add_new_loop = 0;
        // 跳出循环
        break;
    }
    // 如果需要添加新循环
    if (add_new_loop) {
        // 调用函数添加并返回 legacy wrapping ufunc loop 的信息
        PyObject *info = add_and_return_legacy_wrapping_ufunc_loop(
                ufunc, signature, 0);
        // 如果添加失败，跳转到错误处理部分
        if (info == NULL) {
            goto fail;
        }
    }
    /* Clearing sets it to NULL for the error paths */
    // 清空 signature_tuple，以便在错误路径中设为 NULL
    Py_CLEAR(signature_tuple);

    // 设置 funcdata 的成员变量
    funcdata->func = function;
    funcdata->arg_types = newtypes;
    funcdata->data = data;
    funcdata->next = NULL;
    funcdata->arg_dtypes = NULL;
    funcdata->nargs = 0;

    /* Get entry for this user-defined type*/
    // 从 ufunc->userloops 字典中获取 key 对应的值
    cobj = PyDict_GetItemWithError(ufunc->userloops, key);
    // 如果获取失败并且设置了异常，则跳转到错误处理部分
    if (cobj == NULL && PyErr_Occurred()) {
        goto fail;
    }
    /* If it's not there, then make one and return. */
    // 如果找不到对应的值，创建一个新的 PyCapsule 对象并将其设置为 ufunc->userloops[key] 的值
    else if (cobj == NULL) {
        cobj = PyCapsule_New((void *)funcdata, NULL, _loop1d_list_free);
        // 如果创建失败，跳转到错误处理部分
        if (cobj == NULL) {
            goto fail;
        }
        PyDict_SetItem(ufunc->userloops, key, cobj);
        Py_DECREF(cobj);
        Py_DECREF(key);
        // 返回 0 表示成功
        return 0;
    }
    else {
        // 指针声明，用于遍历 PyUFunc_Loop1d 链表
        PyUFunc_Loop1d *current, *prev = NULL;
        // 比较结果，默认为1
        int cmp = 1;
        /*
         * 已经存在至少一个循环。将当前循环按字典顺序插入。
         * 如果下一个循环的签名与当前完全相同，则直接替换。
         * 否则，插入新的循环。
         */
        // 获取当前 cobj 指向的 PyUFunc_Loop1d 结构体指针
        current = PyCapsule_GetPointer(cobj, NULL);
        // 如果获取失败，则跳转到失败处理
        if (current == NULL) {
            goto fail;
        }
        // 遍历链表，按字典顺序比较循环的参数类型
        while (current != NULL) {
            cmp = cmp_arg_types(current->arg_types, newtypes, ufunc->nargs);
            // 如果当前循环的参数类型大于等于新循环的参数类型，退出循环
            if (cmp >= 0) {
                break;
            }
            // 保留前一个循环的指针
            prev = current;
            // 移动到下一个循环
            current = current->next;
        }
        // 如果参数类型完全相同
        if (cmp == 0) {
            /* 直接替换为新的函数和数据 */
            current->func = function;
            current->data = data;
            // 释放内存
            PyArray_free(newtypes);
            PyArray_free(funcdata);
        }
        else {
            /*
             * 在当前循环之前插入新循环，通过修改 cobject 的内部结构
             * 替换函数指针 --- 无法使用 CObject API 因为设置了析构函数。
             */
            // 将新循环插入到链表中
            funcdata->next = current;
            // 如果 prev 为 NULL，则将新循环置于链表开头
            if (prev == NULL) {
                /* 放置在链表前端 */
                _SETCPTR(cobj, funcdata);
            }
            else {
                // 否则，插入到 prev 之后
                prev->next = funcdata;
            }
        }
    }
    // 释放 key 的引用计数
    Py_DECREF(key);
    // 成功返回 0
    return 0;

 fail:
    // 失败时清理资源并返回 -1
    Py_DECREF(key);
    Py_XDECREF(signature_tuple);
    PyArray_free(funcdata);
    PyArray_free(newtypes);
    // 如果没有设置错误，则抛出内存不足的异常
    if (!PyErr_Occurred()) PyErr_NoMemory();
    return -1;
}

#undef _SETCPTR


static void
ufunc_dealloc(PyUFuncObject *ufunc)
{
    // 解除 Python 垃圾回收器的跟踪
    PyObject_GC_UnTrack((PyObject *)ufunc);
    
    // 释放 ufunc 结构体中的各个核心数组
    PyArray_free(ufunc->core_num_dims);
    PyArray_free(ufunc->core_dim_ixs);
    PyArray_free(ufunc->core_dim_sizes);
    PyArray_free(ufunc->core_dim_flags);
    PyArray_free(ufunc->core_offsets);
    PyArray_free(ufunc->core_signature);
    PyArray_free(ufunc->ptr);
    PyArray_free(ufunc->op_flags);
    
    // 释放用户自定义循环的引用
    Py_XDECREF(ufunc->userloops);
    
    // 如果 ufunc 的 identity 是 PyUFunc_IdentityValue，则释放其引用的对象
    if (ufunc->identity == PyUFunc_IdentityValue) {
        Py_DECREF(ufunc->identity_value);
    }
    
    // 释放 ufunc 的 obj 和 _loops 属性的引用
    Py_XDECREF(ufunc->obj);
    Py_XDECREF(ufunc->_loops);
    
    // 如果 _dispatch_cache 不为 NULL，则释放其内存
    if (ufunc->_dispatch_cache != NULL) {
        PyArrayIdentityHash_Dealloc(ufunc->_dispatch_cache);
    }
    
    // 使用 Python 垃圾回收器删除 ufunc 对象
    PyObject_GC_Del(ufunc);
}

static PyObject *
ufunc_repr(PyUFuncObject *ufunc)
{
    // 返回 ufunc 对象的字符串表示形式
    return PyUnicode_FromFormat("<ufunc '%s'>", ufunc->name);
}

static int
ufunc_traverse(PyUFuncObject *self, visitproc visit, void *arg)
{
    // 对 ufunc 对象进行遍历，调用 visit 函数
    Py_VISIT(self->obj);
    
    // 如果 ufunc 的 identity 是 PyUFunc_IdentityValue，则调用 visit 函数
    if (self->identity == PyUFunc_IdentityValue) {
        Py_VISIT(self->identity_value);
    }
    
    // 返回遍历成功
    return 0;
}

/******************************************************************************
 ***                          UFUNC METHODS                                 ***
 *****************************************************************************/

/*
 * op.outer(a,b) is equivalent to op(a[:,NewAxis,NewAxis,etc.],b)
 * where a has b.ndim NewAxis terms appended.
 *
 * The result has dimensions a.ndim + b.ndim
 */
static PyObject *
ufunc_outer(PyUFuncObject *ufunc,
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    // 如果 ufunc 的 core_enabled 属性为真，抛出类型错误异常
    if (ufunc->core_enabled) {
        PyErr_Format(PyExc_TypeError,
                     "method outer is not allowed in ufunc with non-trivial"\
                     " signature");
        return NULL;
    }
    
    // 如果 ufunc 的 nin 不等于 2，抛出值错误异常
    if (ufunc->nin != 2) {
        PyErr_SetString(PyExc_ValueError,
                        "outer product only supported "\
                        "for binary functions");
        return NULL;
    }
    
    // 如果传入的参数个数不等于 2，抛出类型错误异常
    if (len_args != 2) {
        PyErr_SetString(PyExc_TypeError, "exactly two arguments expected");
        return NULL;
    }
    
    // 调用 ufunc_generic_fastcall 处理函数，返回其结果
    return ufunc_generic_fastcall(ufunc, args, len_args, kwnames, NPY_TRUE);
}

static PyObject *
prepare_input_arguments_for_outer(PyObject *args, PyUFuncObject *ufunc)
{
    PyArrayObject *ap1 = NULL;
    PyObject *tmp;
    npy_cache_import("numpy", "matrix", &npy_thread_unsafe_state.numpy_matrix);

    const char *matrix_deprecation_msg = (
            "%s.outer() was passed a numpy matrix as %s argument. "
            "Special handling of matrix is deprecated and will result in an "
            "error in most cases. Please convert the matrix to a NumPy "
            "array to retain the old behaviour. You can use `matrix.A` "
            "to achieve this.");

    tmp = PyTuple_GET_ITEM(args, 0);


注释：以上是一段 C 语言的代码，主要涉及 Python 的 C 扩展模块中的 ufunc 对象的生命周期管理和方法实现。
    if (PyObject_IsInstance(tmp, npy_thread_unsafe_state.numpy_matrix)) {
        /* 检查是否为旧版 NumPy 矩阵对象，此功能已于 2020-05-13 废弃，从 NumPy 1.20 开始不建议使用 */
        if (PyErr_WarnFormat(PyExc_DeprecationWarning, 1,
                matrix_deprecation_msg, ufunc->name, "first") < 0) {
            return NULL;
        }
        // 将 tmp 转换为 PyArrayObject 对象，类型为 NPY_NOTYPE，不复制数据
        ap1 = (PyArrayObject *) PyArray_FromObject(tmp, NPY_NOTYPE, 0, 0);
    }
    else {
        // 将 tmp 转换为 PyArrayObject 对象，自动推断数据类型
        ap1 = (PyArrayObject *) PyArray_FROM_O(tmp);
    }
    if (ap1 == NULL) {
        return NULL;
    }

    PyArrayObject *ap2 = NULL;
    tmp = PyTuple_GET_ITEM(args, 1);
    if (PyObject_IsInstance(tmp, npy_thread_unsafe_state.numpy_matrix)) {
        /* 检查是否为旧版 NumPy 矩阵对象，此功能已于 2020-05-13 废弃，从 NumPy 1.20 开始不建议使用 */
        if (PyErr_WarnFormat(PyExc_DeprecationWarning, 1,
                matrix_deprecation_msg, ufunc->name, "second") < 0) {
            // 发出警告失败时释放 ap1 并返回 NULL
            Py_DECREF(ap1);
            return NULL;
        }
        // 将 tmp 转换为 PyArrayObject 对象，类型为 NPY_NOTYPE，不复制数据
        ap2 = (PyArrayObject *) PyArray_FromObject(tmp, NPY_NOTYPE, 0, 0);
    }
    else {
        // 将 tmp 转换为 PyArrayObject 对象，自动推断数据类型
        ap2 = (PyArrayObject *) PyArray_FROM_O(tmp);
    }
    if (ap
# 结束静态函数 ufunc_reduce，使用 PyUFunc_GenericReduction 执行通用归约操作
static PyObject *
ufunc_reduce(PyUFuncObject *ufunc,
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    return PyUFunc_GenericReduction(
            ufunc, args, len_args, kwnames, UFUNC_REDUCE);
}

# 结束静态函数 ufunc_accumulate，使用 PyUFunc_GenericReduction 执行通用累积操作
static PyObject *
ufunc_accumulate(PyUFuncObject *ufunc,
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    return PyUFunc_GenericReduction(
            ufunc, args, len_args, kwnames, UFUNC_ACCUMULATE);
}

# 结束静态函数 ufunc_reduceat，使用 PyUFunc_GenericReduction 执行通用 reduceat 操作
static PyObject *
ufunc_reduceat(PyUFuncObject *ufunc,
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    return PyUFunc_GenericReduction(
            ufunc, args, len_args, kwnames, UFUNC_REDUCEAT);
}

/* Helper for ufunc_at, below */
# 创建一个新的 PyArrayObject，用于操作数组，并增加描述符的引用计数
static inline PyArrayObject *
new_array_op(PyArrayObject *op_array, char *data)
{
    npy_intp dims[1] = {1};
    Py_INCREF(PyArray_DESCR(op_array));  /* NewFromDescr steals a reference */
    PyObject *r = PyArray_NewFromDescr(&PyArray_Type, PyArray_DESCR(op_array),
                                       1, dims, NULL, data,
                                       NPY_ARRAY_WRITEABLE, NULL);
    return (PyArrayObject *)r;
}

/*
 * 使用索引循环执行工作
 * 如果成功返回 0
 */
static int
trivial_at_loop(PyArrayMethodObject *ufuncimpl, NPY_ARRAYMETHOD_FLAGS flags,
                    PyArrayMapIterObject *iter,
                    PyArrayObject *op1_array, PyArrayObject *op2_array,
                    PyArrayMethod_Context *context)
{
    int buffersize=0, errormask = 0;
    int res;
    char *args[3];
    npy_intp steps[4];
    args[0] = (char *) iter->baseoffset;
    steps[0] = iter->fancy_strides[0];
    if (ufuncimpl->nin == 1) {
        args[2] = NULL;
        steps[2] = 0;
    } else {
        args[2] = (char *)PyArray_DATA(op2_array);
        if (PyArray_NDIM(op2_array) == 0
            || PyArray_DIM(op2_array, 0) <= 1) {
            steps[2] = 0;
        } else {
            steps[2] = PyArray_STRIDE(op2_array, 0);
        }
    }

    if (!(flags & NPY_METH_NO_FLOATINGPOINT_ERRORS)) {
        npy_clear_floatstatus_barrier((char *)context);
    }

    do {
        npy_intp *inner_size = NpyIter_GetInnerLoopSizePtr(iter->outer);
        npy_intp * indxP = (npy_intp *)iter->outer_ptrs[0];
        args[1] = (char *)indxP;
        steps[1] = iter->outer_strides[0];
        /*
         * 内循环中，将 iter->fancy_dims[0] 的值添加到负索引中
         */
        steps[3] = iter->fancy_dims[0];

        res = ufuncimpl->contiguous_indexed_loop(
                context, args, inner_size, steps, NULL);

        if (args[2] != NULL) {
            args[2] += (*inner_size) * steps[2];
        }
    } while (res == 0 && iter->outer_next(iter->outer));
    # 如果 res 等于 0 并且 flags 没有设置 NPY_METH_NO_FLOATINGPOINT_ERRORS 标志位时执行以下操作
    if (res == 0 && !(flags & NPY_METH_NO_FLOATINGPOINT_ERRORS)) {
        # 获取调用者的 ufunc 名称作为 C 字符串
        const char * ufunc_name =
                        ufunc_get_name_cstr((PyUFuncObject *)context->caller);
        # 获取缓冲区大小和错误掩码，如果获取失败则返回 -1
        if (_get_bufsize_errmask(&buffersize, &errormask) < 0) {
            return -1;
        }
        # 检查 ufunc 的浮点错误，将结果存储在 res 中
        res = _check_ufunc_fperr(errormask, ufunc_name);
    }
    # 返回 res 的值
    return res;
}

static int
ufunc_at__fast_iter(PyUFuncObject *ufunc, NPY_ARRAYMETHOD_FLAGS flags,
                    PyArrayMapIterObject *iter, PyArrayIterObject *iter2,
                    PyArrayObject *op1_array, PyArrayObject *op2_array,
                    PyArrayMethod_StridedLoop *strided_loop,
                    PyArrayMethod_Context *context,
                    NpyAuxData *auxdata
                    )
{
    int buffersize;
    int errormask = 0;
    int res = 0;
    NPY_BEGIN_THREADS_DEF;

    // 获取缓冲区大小和错误掩码
    if (_get_bufsize_errmask(&buffersize, &errormask) < 0) {
        return -1;
    }
    int needs_api = (flags & NPY_METH_REQUIRES_PYAPI) != 0;
    // 如果未禁用浮点错误，清除浮点异常标志
    if (!(flags & NPY_METH_NO_FLOATINGPOINT_ERRORS)) {
        /* Start with the floating-point exception flags cleared */
        npy_clear_floatstatus_barrier((char*)&iter);
    }

    // 如果不需要 Python API 调用，开始多线程计算
    if (!needs_api) {
        NPY_BEGIN_THREADS;
    }

    npy_intp strides[3] = {0, 0, 0};
    /*
     * 遍历第一个和第二个操作数，并为每对输入调用 ufunc
     */
    for (npy_intp i = iter->size; i > 0; i--)
    {
        char *dataptr[3];
        /* 每次一个元素，无需步长，但由内循环读取 */
        npy_intp count = 1;

        /*
         * 设置数据指针，用于一个或两个输入操作数。
         * 输出数据指针指向第一个操作数的数据。
         */
        dataptr[0] = iter->dataptr;
        if (iter2 != NULL) {
            dataptr[1] = PyArray_ITER_DATA(iter2);
            dataptr[2] = iter->dataptr;
        }
        else {
            dataptr[1] = iter->dataptr;
            dataptr[2] = NULL;
        }

        // 调用 strided_loop 函数执行循环计算
        res = strided_loop(context, dataptr, &count, strides, auxdata);
        if (res != 0) {
            break;
        }

        // 迭代到下一个位置
        PyArray_MapIterNext(iter);
        if (iter2 != NULL) {
            PyArray_ITER_NEXT(iter2);
        }
    }

    // 结束多线程计算
    NPY_END_THREADS;

    // 如果没有错误，并且未禁用浮点错误，检查浮点错误
    if (res == 0 && !(flags & NPY_METH_NO_FLOATINGPOINT_ERRORS)) {
        /* NOTE: We could check float errors even when `res < 0` */
        res = _check_ufunc_fperr(errormask, "at");
    }
    return res;
}

static int
ufunc_at__slow_iter(PyUFuncObject *ufunc, NPY_ARRAYMETHOD_FLAGS flags,
                    PyArrayMapIterObject *iter, PyArrayIterObject *iter2,
                    PyArrayObject *op1_array, PyArrayObject *op2_array,
                    PyArray_Descr *operation_descrs[3],
                    PyArrayMethod_StridedLoop *strided_loop,
                    PyArrayMethod_Context *context,
                    NpyAuxData *auxdata
                    )
{
    NpyIter *iter_buffer = NULL;
    PyArrayObject *array_operands[3] = {NULL, NULL, NULL};
    int buffersize;
    int errormask = 0;
    int res = 0;
    int nop = 0;
    NpyIter_IterNextFunc *iternext;
    char * err_msg = NULL;
    NPY_BEGIN_THREADS_DEF;

    // 获取缓冲区大小和错误掩码
    if (_get_bufsize_errmask(&buffersize, &errormask) < 0) {
        return -1;
    }
    // 创建新的操作数组对象
    array_operands[0] = new_array_op(op1_array, iter->dataptr);
    // 检查 iter2 是否为 NULL，确定操作数数组和操作数的个数
    if (iter2 != NULL) {
        // 使用 op2_array 和 iter2 的数据指针创建新的数组操作对象
        array_operands[1] = new_array_op(op2_array, PyArray_ITER_DATA(iter2));
        // 使用 op1_array 和 iter 的数据指针创建新的数组操作对象
        array_operands[2] = new_array_op(op1_array, iter->dataptr);
        // 操作数的个数设为 3
        nop = 3;
    }
    else {
        // 使用 op1_array 和 iter 的数据指针创建新的数组操作对象
        array_operands[1] = new_array_op(op1_array, iter->dataptr);
        // 第二个操作数设为 NULL
        array_operands[2] = NULL;
        // 操作数的个数设为 2
        nop = 2;
    }
    /* 设置操作标志 */
    npy_uint32 op_flags[3];
    // 第一个操作标志：只读和内存对齐
    op_flags[0] = NPY_ITER_READONLY|
                  NPY_ITER_ALIGNED;

    if (iter2 != NULL) {
        // 第二个操作标志：只读和内存对齐
        op_flags[1] = NPY_ITER_READONLY|
                      NPY_ITER_ALIGNED;
        // 第三个操作标志：写入、内存对齐、分配内存、不进行广播、不接受子类型
        op_flags[2] = NPY_ITER_WRITEONLY|
                      NPY_ITER_ALIGNED|
                      NPY_ITER_ALLOCATE|
                      NPY_ITER_NO_BROADCAST|
                      NPY_ITER_NO_SUBTYPE;
    }
    else {
        // 第二个操作标志：写入、内存对齐、分配内存、不进行广播、不接受子类型
        op_flags[1] = NPY_ITER_WRITEONLY|
                      NPY_ITER_ALIGNED|
                      NPY_ITER_ALLOCATE|
                      NPY_ITER_NO_BROADCAST|
                      NPY_ITER_NO_SUBTYPE;
    }
    /*
     * 创建 NpyIter 对象，用于"迭代"每个输入操作数的单个元素。
     * 这是重用 NpyIter 逻辑处理特定情况（如操作数的正确数据类型转换）的简便方法。
     * 在上面创建的 MapIterArray 对象的每次迭代中，我们将使用该对象的当前数据指针重置此 NpyIter 对象，
     * 然后触发缓冲区复制。NpyIter 对象的缓冲区数据指针将传递给内部循环函数。
     */
    iter_buffer = NpyIter_AdvancedNew(nop, array_operands,
                        NPY_ITER_EXTERNAL_LOOP|
                        NPY_ITER_REFS_OK|
                        NPY_ITER_ZEROSIZE_OK|
                        NPY_ITER_BUFFERED|
                        NPY_ITER_GROWINNER|
                        NPY_ITER_DELAY_BUFALLOC,
                        NPY_KEEPORDER, NPY_UNSAFE_CASTING,
                        op_flags, operation_descrs,
                        -1, NULL, NULL, buffersize);

    if (iter_buffer == NULL) {
        /* 只在内存分配错误时失败 */
        // 释放所有数组操作对象
        for (int i = 0; i < 3; i++) {
            Py_XDECREF(array_operands[i]);
        }
        // 返回错误代码 -1
        return -1;
    }

    // 获取 NpyIter 对象的迭代器
    iternext = NpyIter_GetIterNext(iter_buffer, NULL);
    if (iternext == NULL) {
        /* 实际上不可能发生，iter_buffer 的创建受到严格控制 */
        // 释放 NpyIter 对象
        NpyIter_Deallocate(iter_buffer);
        // 释放所有数组操作对象
        for (int i = 0; i < 3; i++) {
            Py_XDECREF(array_operands[i]);
        }
        // 返回错误代码 -1
        return -1;
    }

    // 根据标志检查是否需要 Python API
    int needs_api = (flags & NPY_METH_REQUIRES_PYAPI) != 0;
    needs_api |= NpyIter_IterationNeedsAPI(iter_buffer);
    if (!(flags & NPY_METH_NO_FLOATINGPOINT_ERRORS)) {
        /* 从清除浮点异常标志开始 */
        // 清除浮点异常标志
        npy_clear_floatstatus_barrier((char*)&iter);
    }

    // 初始化步长数组
    npy_intp strides[3] = {0, 0, 0};
    if (!needs_api) {
        // 开始线程
        NPY_BEGIN_THREADS;
    }
    /*
     * Iterate over first and second operands and call ufunc
     * for each pair of inputs
     */
    for (npy_intp i = iter->size; i > 0; i--)
    {
        char *dataptr[3];
        char **buffer_dataptr;
        /* one element at a time, no stride required but read by innerloop */
        npy_intp count = 1;

        /*
         * Set up data pointers for either one or two input operands.
         * The output data pointer points to the first operand data.
         */
        dataptr[0] = iter->dataptr;
        if (iter2 != NULL) {
            // Set data pointers for two input operands
            dataptr[1] = PyArray_ITER_DATA(iter2);
            dataptr[2] = iter->dataptr;  // Output data pointer
        }
        else {
            // Set data pointers for one input operand
            dataptr[1] = iter->dataptr;
            dataptr[2] = NULL;  // No output data pointer
        }

        /* Reset NpyIter data pointers which will trigger a buffer copy */
        // Reset the base pointers of the iterator buffer
        NpyIter_ResetBasePointers(iter_buffer, dataptr, &err_msg);
        if (err_msg) {
            // If error message is set, return -1
            res = -1;
            break;
        }

        buffer_dataptr = NpyIter_GetDataPtrArray(iter_buffer);

        // Execute strided loop operation
        res = strided_loop(context, buffer_dataptr, &count, strides, auxdata);
        if (res != 0) {
            // If strided loop operation fails, break loop
            break;
        }

        /*
         * Call to iternext triggers copy from buffer back to output array
         * after innerloop puts result in buffer.
         */
        // Move to the next iteration in the iterator buffer
        iternext(iter_buffer);

        // Move to the next iteration in the main iterator
        PyArray_MapIterNext(iter);
        if (iter2 != NULL) {
            // Move to the next iteration in the second iterator
            PyArray_ITER_NEXT(iter2);
        }
    }

    NPY_END_THREADS;

    if (res != 0 && err_msg) {
        // If there was an error and an error message exists, raise ValueError
        PyErr_SetString(PyExc_ValueError, err_msg);
    }
    if (res == 0 && !(flags & NPY_METH_NO_FLOATINGPOINT_ERRORS)) {
        /* NOTE: We could check float errors even when `res < 0` */
        // Check for floating point errors if the result is successful
        res = _check_ufunc_fperr(errormask, "at");
    }
    // Deallocate the iterator buffer
    NpyIter_Deallocate(iter_buffer);
    // Release references to array operands
    for (int i = 0; i < 3; i++) {
        Py_XDECREF(array_operands[i]);
    }
    // Return the result status
    return res;
/*
 * Call ufunc only on selected array items and store result in first operand.
 * For add ufunc, method call is equivalent to op1[idx] += op2 with no
 * buffering of the first operand.
 * Arguments:
 * op1 - First operand to ufunc
 * idx - Indices that are applied to first operand. Equivalent to op1[idx].
 * op2 - Second operand to ufunc (if needed). Must be able to broadcast
 *       over first operand.
 */
static PyObject *
ufunc_at(PyUFuncObject *ufunc, PyObject *args)
{
    PyObject *op1 = NULL;               // 第一个操作数
    PyObject *idx = NULL;               // 应用于第一个操作数的索引，相当于 op1[idx]
    PyObject *op2 = NULL;               // 第二个操作数
    PyArrayObject *op1_array = NULL;    // 第一个操作数的数组对象
    PyArrayObject *op2_array = NULL;    // 第二个操作数的数组对象（如果有）
    PyArrayMapIterObject *iter = NULL;  // 数组映射迭代器对象
    PyArrayIterObject *iter2 = NULL;    // 数组迭代器对象
    PyArray_Descr *operation_descrs[3] = {NULL, NULL, NULL};  // 操作描述符数组

    int nop;  // 操作数个数

    /* override vars */
    int errval;           // 错误值
    PyObject *override = NULL;  // 覆盖对象
    int res = -1;         // 初始设置为失败状态，以便 "goto fail" 会出错

    PyArrayMethod_StridedLoop *strided_loop;  // 数组方法的跨步循环
    NpyAuxData *auxdata = NULL;               // 辅助数据

    // 如果核心功能启用，不支持复杂签名的ufunc
    if (ufunc->core_enabled) {
        PyErr_Format(PyExc_TypeError,
            "%s.at does not support ufunc with non-trivial signature: %s has signature %s.",
            ufunc->name, ufunc->name, ufunc->core_signature);
        return NULL;
    }

    // 仅支持一元和二元ufunc
    if (ufunc->nin > 2) {
        PyErr_SetString(PyExc_ValueError,
            "Only unary and binary ufuncs supported at this time");
        return NULL;
    }

    // 仅支持单输出的ufunc
    if (ufunc->nout != 1) {
        PyErr_SetString(PyExc_ValueError,
            "Only single output ufuncs supported at this time");
        return NULL;
    }

    // 解析参数，必须是 op1, idx, op2 (可选)
    if (!PyArg_ParseTuple(args, "OO|O:at", &op1, &idx, &op2)) {
        return NULL;
    }

    // 如果是二元ufunc但是没有提供第二个操作数，报错
    if (ufunc->nin == 2 && op2 == NULL) {
        PyErr_SetString(PyExc_ValueError,
                        "second operand needed for ufunc");
        return NULL;
    }

    // 如果是一元ufunc但提供了第二个操作数，报错
    if (ufunc->nin == 1 && op2 != NULL) {
        PyErr_SetString(PyExc_ValueError,
                        "second operand provided when ufunc is unary");
        return NULL;
    }

    // 检查是否有覆盖实现
    errval = PyUFunc_CheckOverride(ufunc, "at",
            args, NULL, NULL, NULL, 0, NULL, &override);

    if (errval) {
        return NULL;
    }
    else if (override) {
        return override;
    }

    // 第一个操作数必须是数组
    if (!PyArray_Check(op1)) {
        PyErr_SetString(PyExc_TypeError,
                        "first operand must be array");
        return NULL;
    }

    op1_array = (PyArrayObject *)op1;

    // 如果没有提供第二个操作数，设置操作数个数为2，否则为3，并从第二个操作数创建数组对象
    if (op2 == NULL) {
        nop = 2;
    }
    else {
        nop = 3;
        op2_array = (PyArrayObject *)PyArray_FromAny(op2, NULL,
                                0, 0, 0, NULL);
        if (op2_array == NULL) {
            goto fail;
        }
    }

    PyArrayMethodObject *ufuncimpl = NULL;
    {
        /* 处理 dtype 并找到正确的 ufunc 实现 */

        PyArrayObject *tmp_operands[3] = {NULL, NULL, NULL};
        PyArray_DTypeMeta *signature[3] = {NULL, NULL, NULL};
        PyArray_DTypeMeta *operand_DTypes[3] = {NULL, NULL, NULL};
        /*
         * 创建一个 dtypes 数组，用于一个或两个输入操作数。
         * 与 `convert_ufunc_arguments` 中的逻辑进行比较。
         * TODO: 可能需要回顾一些行为，因为操作数数组是特殊的（它是可写的），类似于减少操作。
         *       在这里使用不安全的强制转换可能不是理想的做法。
         */
        tmp_operands[0] = op1_array;
        operand_DTypes[0] = NPY_DTYPE(PyArray_DESCR(op1_array));
        Py_INCREF(operand_DTypes[0]);
        int force_legacy_promotion = 0;
        int allow_legacy_promotion = NPY_DT_is_legacy(operand_DTypes[0]);

        if (op2_array != NULL) {
            tmp_operands[1] = op2_array;
            operand_DTypes[1] = NPY_DTYPE(PyArray_DESCR(op2_array));
            Py_INCREF(operand_DTypes[1]);
            allow_legacy_promotion &= NPY_DT_is_legacy(operand_DTypes[1]);
            tmp_operands[2] = tmp_operands[0];
            operand_DTypes[2] = operand_DTypes[0];
            Py_INCREF(operand_DTypes[2]);

            if (allow_legacy_promotion && ((PyArray_NDIM(op1_array) == 0)
                                           != (PyArray_NDIM(op2_array) == 0))) {
                    /* 如果两者都是 legacy 并且只有一个是 0-D：强制使用 legacy */
                    force_legacy_promotion = should_use_min_scalar(2, tmp_operands, 0, NULL);
                }
        }
        else {
            tmp_operands[1] = tmp_operands[0];
            operand_DTypes[1] = operand_DTypes[0];
            Py_INCREF(operand_DTypes[1]);
            tmp_operands[2] = NULL;
        }

        // 推广并获取 ufunc 的实现
        ufuncimpl = promote_and_get_ufuncimpl(ufunc, tmp_operands, signature,
                        operand_DTypes, force_legacy_promotion,
                        allow_legacy_promotion, NPY_FALSE, NPY_FALSE);
        if (ufuncimpl == NULL) {
            for (int i = 0; i < 3; i++) {
                Py_XDECREF(signature[i]);
                Py_XDECREF(operand_DTypes[i]);
            }
            goto fail;
        }

        /* 找到操作的正确 operation_descrs */
        int resolve_result = resolve_descriptors(nop, ufunc, ufuncimpl,
                tmp_operands, operation_descrs, signature, NULL, NPY_UNSAFE_CASTING);
        for (int i = 0; i < 3; i++) {
            Py_XDECREF(signature[i]);
            Py_XDECREF(operand_DTypes[i]);
        }
        if (resolve_result < 0) {
            goto fail;
        }
    }

    iter = (PyArrayMapIterObject *)PyArray_MapIterArrayCopyIfOverlap(
        op1_array, idx, 1, op2_array);
    if (iter == NULL) {
        goto fail;
    }
    op1_array = iter->array;  /* 如果重叠可能会更新 */
    # 检查第二操作数数组是否非空
    if (op2_array != NULL) {
        """
         * 可能需要交换轴，以确保第二操作数能够正确迭代
         """
        # 如果存在子空间且连续性标志为真，则交换迭代器的轴
        if ((iter->subspace != NULL) && (iter->consec)) {
            PyArray_MapIterSwapAxes(iter, &op2_array, 0);
            # 如果内存分配失败，则跳转到失败标签
            if (op2_array == NULL) {
                /* 仅在内存分配失败时执行 */
                goto fail;
            }
        }

        """
         * 创建第二操作数的数组迭代器对象，使其与第一操作数的
         * "匹配"。这样我们就可以同时迭代第一和第二操作数，
         * 而无需担心选择每个操作数的正确元素来应用ufunc。
         """
        # 如果创建第二操作数的广播形状对象失败，则跳转到失败标签
        if ((iter2 = (PyArrayIterObject *)\
             PyArray_BroadcastToShape((PyObject *)op2_array,
                                        iter->dimensions, iter->nd))==NULL) {
            goto fail;
        }
    }

    # 设置操作上下文结构体
    PyArrayMethod_Context context = {
            .caller = (PyObject *)ufunc,
            .method = ufuncimpl,
            .descriptors = operation_descrs,
    };

    """
     * 使用连续的步长；如果存在这样的循环，可能会更快
     """
    # 初始化步长数组
    npy_intp strides[3] = {
            operation_descrs[0]->elsize, operation_descrs[1]->elsize, 0};
    # 如果操作数个数为3，则设置第三个步长
    if (nop == 3) {
        strides[2] = operation_descrs[2]->elsize;
    }

    # 初始化数组方法标志
    NPY_ARRAYMETHOD_FLAGS flags;
    # 调用ufunc实现获取步进循环，若失败则跳转到失败标签
    if (ufuncimpl->get_strided_loop(&context, 1, 0, strides,
            &strided_loop, &auxdata, &flags) < 0) {
        goto fail;
    }
    # 初始化快速路径标志为真
    int fast_path = 1;
    """
     * 检查无需类型转换和对齐
     """
    # 如果第一操作数的数据类型描述符与第一个操作的描述符不匹配，则快速路径标志为假
    if (PyArray_DESCR(op1_array) != operation_descrs[0]) {
        fast_path = 0;
    }
    # 如果第一操作数的数据类型描述符与最后一个操作的描述符不匹配，则快速路径标志为假
    if (PyArray_DESCR(op1_array) != operation_descrs[nop - 1]) {
        """ 
        Who	Had استigating amber alert Scents
    # 如果 fast_path 等于 1，则执行以下代码块
    if (fast_path == 1) {
        """
         * 尝试使用简单的循环（一维、无类型转换、对齐）：
         * - 匹配信息具有索引循环
         * - idx 必须是正好一个整数索引数组
         * - 所有操作数都是一维的
         * 未来的增强可以通过在 trivial_at_loop 内部添加迭代循环来放宽对一维操作数的限制
         """
        # 如果 ufuncimpl 的 contiguous_indexed_loop 不为 NULL，并且：
        # - op1_array 是一维数组
        # - op2_array 为 NULL 或者 op2_array 的维数不超过 1
        # - iter 的 subspace_iter 为 NULL
        # - iter 的 num_fancy 为 1
        if ((ufuncimpl->contiguous_indexed_loop != NULL) &&
                (PyArray_NDIM(op1_array) == 1)  &&
                (op2_array == NULL || PyArray_NDIM(op2_array) <= 1) &&
                (iter->subspace_iter == NULL) && (iter->num_fancy == 1)) {
            # 调用 trivial_at_loop 函数处理
            res = trivial_at_loop(ufuncimpl, flags, iter, op1_array,
                        op2_array, &context);

        }
        else {
            # 无法使用最快的路径，转而使用更快的路径
            res = ufunc_at__fast_iter(ufunc, flags, iter, iter2, op1_array,
                        op2_array, strided_loop, &context, auxdata);
        }
    } else {
        # fast_path 不为 1，则使用慢速迭代器处理
        res = ufunc_at__slow_iter(ufunc, flags, iter, iter2, op1_array,
                        op2_array, operation_descrs, strided_loop, &context,
                        auxdata);
    }
fail:
    // 释放 NPY_AUXDATA
    NPY_AUXDATA_FREE(auxdata);

    // 释放 Python 对象 op2_array
    Py_XDECREF(op2_array);
    // 释放 Python 对象 iter
    Py_XDECREF(iter);
    // 释放 Python 对象 iter2
    Py_XDECREF(iter2);
    // 循环释放操作描述符列表中的 Python 对象
    for (int i = 0; i < nop; i++) {
        Py_XDECREF(operation_descrs[i]);
    }

    /*
     * 只有在 res 不等于 0 或者出现了异常时才返回 NULL。
     * 对于旧式的 ufunc（例如 `power`），这种情况下不严格正确，
     * 因为它会释放 GIL 但手动设置异常。
     */
    if (res != 0 || PyErr_Occurred()) {
        /*
         * 如果 op1_array 的标志中包含 NPY_ARRAY_WRITEBACKIFCOPY，
         * 则调用 PyArray_DiscardWritebackIfCopy 函数来丢弃写回副本。
         */
        if (PyArray_FLAGS(op1_array) & NPY_ARRAY_WRITEBACKIFCOPY) {
            PyArray_DiscardWritebackIfCopy(op1_array);
        }
        // 返回 NULL
        return NULL;
    }
    else {
        // 返回 Py_None
        Py_RETURN_NONE;
    }
}


typedef struct {
    // 指向 PyArrayMethod_StridedLoop 函数的指针
    PyArrayMethod_StridedLoop *strided_loop;
    // 指向 PyArrayMethod_Context 结构的指针
    PyArrayMethod_Context *context;
    // 指向 NpyAuxData 结构的指针
    NpyAuxData *auxdata;
    // 是否需要 Python API
    npy_bool requires_pyapi;
    // 是否不允许浮点错误
    npy_bool no_floatingpoint_errors;
    // 完整的 PyArrayMethod_Context 结构
    PyArrayMethod_Context _full_context;
    // PyArray_Descr 类型的指针数组
    PyArray_Descr *_descrs[];
} ufunc_call_info;


void
free_ufunc_call_info(PyObject *self)
{
    // 获取指向 ufunc_call_info 结构的指针
    ufunc_call_info *call_info = PyCapsule_GetPointer(
            self, "numpy_1.24_ufunc_call_info");

    // 获取 call_info 中的 context 指针
    PyArrayMethod_Context *context = call_info->context;

    // 获取参数的数量
    int nargs = context->method->nin + context->method->nout;
    // 循环释放 context 中的描述符数组中的对象
    for (int i = 0; i < nargs; i++) {
        Py_DECREF(context->descriptors[i]);
    }
    // 释放 context 的 caller 对象
    Py_DECREF(context->caller);
    // 释放 context 的 method 对象
    Py_DECREF(context->method);
    // 释放 call_info 的 auxdata
    NPY_AUXDATA_FREE(call_info->auxdata);

    // 释放 PyObject 内存
    PyObject_Free(call_info);
}


/*
 * Python 入口点，用于 ufunc 的类型提升和 dtype/descr 解析。
 *
 * 此函数执行了几乎所有执行 ufunc 所需的工作，但实际上并未执行它。
 * 对于重新实现 NumPy 功能的下游库（如 Numba 或 Dask），这非常有用。
 */
static PyObject *
py_resolve_dtypes_generic(PyUFuncObject *ufunc, npy_bool return_context,
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    NPY_PREPARE_ARGPARSER;

    // 描述符元组
    PyObject *descrs_tuple;
    // 签名对象
    PyObject *signature_obj = NULL;
    // 转换类型
    NPY_CASTING casting = NPY_DEFAULT_ASSIGN_CASTING;
    // 是否为 reduction
    npy_bool reduction = NPY_FALSE;

    // 解析参数
    if (npy_parse_arguments("resolve_dtypes", args, len_args, kwnames,
            "", NULL, &descrs_tuple,
            "$signature", NULL, &signature_obj,
            "$casting", &PyArray_CastingConverter, &casting,
            "$reduction", &PyArray_BoolConverter, &reduction,
            NULL, NULL, NULL) < 0) {
        return NULL;
    }

    // 如果是 reduction 并且 ufunc 的输入输出数量不匹配，则设置错误并返回 NULL
    if (reduction && (ufunc->nin != 2 || ufunc->nout != 1)) {
        PyErr_SetString(PyExc_ValueError,
                "ufunc is not compatible with reduction operations.");
        return NULL;
    }
    /*
     * Legacy type resolvers expect NumPy arrays as input.  Until NEP 50 is
     * adopted, it is most convenient to ensure that we have an "array" object
     * before calling the type promotion.  Eventually, this hack may be moved
     * into the legacy type resolution code itself (probably after NumPy stops
     * using legacy type resolution itself for the most part).
     *
     * We make the pretty safe assumptions here that:
     * - Nobody will actually do anything with the array objects besides
     *   checking the descriptor or calling CanCast.
     * - No type resolver will cause weird paths that mess with our promotion
     *   state (or mind us messing with it).
     */

    // 初始化变量
    PyObject *result = NULL;  // 结果对象初始化为空
    PyObject *result_dtype_tuple = NULL;  // 结果数据类型元组初始化为空

    PyArrayObject *dummy_arrays[NPY_MAXARGS] = {NULL};  // 用于占位的 NumPy 数组对象数组
    PyArray_DTypeMeta *DTypes[NPY_MAXARGS] = {NULL};  // 数据类型元数据数组
    PyArray_DTypeMeta *signature[NPY_MAXARGS] = {NULL};  // 签名数据类型元数据数组
    PyArray_Descr *operation_descrs[NPY_MAXARGS] = {NULL};  // 操作描述符数组

    /* This entry-point to promotion lives in the NEP 50 future: */
    // 获取当前 NumPy 提升状态，并设置为弱提升状态
    int original_promotion_state = get_npy_promotion_state();
    set_npy_promotion_state(NPY_USE_WEAK_PROMOTION);

    // 初始化标志变量
    npy_bool promoting_pyscalars = NPY_FALSE;  // 是否提升 Python 标量的标志
    npy_bool allow_legacy_promotion = NPY_TRUE;  // 是否允许使用旧式类型提升的标志

    // 获取固定签名数据类型元数据
    if (_get_fixed_signature(ufunc, NULL, signature_obj, signature) < 0) {
        goto finish;  // 如果获取失败，跳转到结束标签
    }

    // 检查描述符是否为元组并且长度与 ufunc->nargs 相符
    if (!PyTuple_CheckExact(descrs_tuple)
            || PyTuple_Size(descrs_tuple) != ufunc->nargs)  {
        PyErr_SetString(PyExc_TypeError,
                "resolve_dtypes: The dtypes must be a tuple of "
                "`ufunc.nargs` length.");
        goto finish;  // 如果检查失败，设置错误信息并跳转到结束标签
    }

    PyArrayMethodObject *ufuncimpl;
    // 如果不是缩减操作，执行类型提升并获取相应的函数实现对象
    if (!reduction) {
        ufuncimpl = promote_and_get_ufuncimpl(ufunc,
                dummy_arrays, signature, DTypes, NPY_FALSE,
                allow_legacy_promotion, promoting_pyscalars, NPY_FALSE);
        if (ufuncimpl == NULL) {
            goto finish;  // 如果获取失败，跳转到结束标签
        }

        // 解析操作的正确描述符
        if (resolve_descriptors(ufunc->nargs, ufunc, ufuncimpl,
                dummy_arrays, operation_descrs, signature,
                NULL, casting) < 0) {
            goto finish;  // 如果解析失败，跳转到结束标签
        }

        // 验证类型转换的有效性
        if (validate_casting(
                ufuncimpl, ufunc, dummy_arrays, operation_descrs, casting) < 0) {
            goto finish;  // 如果验证失败，跳转到结束标签
        }
    }
    else {  /* reduction */
        // 如果是减少操作，则执行以下代码块

        if (signature[2] != NULL) {
            // 如果签名中第三个元素不是NULL，则抛出值错误异常
            PyErr_SetString(PyExc_ValueError,
                    "Reduction signature must end with None, instead pass "
                    "the first DType in the signature.");
            goto finish;
        }

        if (dummy_arrays[2] != NULL) {
            // 如果输出的dummy数组不为NULL，则抛出类型错误异常
            PyErr_SetString(PyExc_TypeError,
                    "Output dtype must not be passed for reductions, "
                    "pass the first input instead.");
            goto finish;
        }

        // 调用reducelike_promote_and_resolve函数，解析并促进约简操作
        ufuncimpl = reducelike_promote_and_resolve(ufunc,
                dummy_arrays[1], dummy_arrays[0], signature, NPY_FALSE,
                operation_descrs, casting, "resolve_dtypes");

        // 如果ufuncimpl为空，则跳转到finish标签
        if (ufuncimpl == NULL) {
            goto finish;
        }
    }

    // 从操作描述数组创建一个Python元组
    result = PyArray_TupleFromItems(
            ufunc->nargs, (PyObject **)operation_descrs, 0);

    // 如果结果为NULL或者不需要返回上下文信息，则跳转到finish标签
    if (result == NULL || !return_context) {
        goto finish;
    }
    /* Result will be (dtype_tuple, call_info), so move it and clear result */
    // 结果将是一个元组(result_dtype_tuple, capsule)，移动它并清除result
    result_dtype_tuple = result;
    result = NULL;

    // 可能需要返回上下文信息
    ufunc_call_info *call_info;
    // 分配内存给call_info结构体
    call_info = PyObject_Malloc(sizeof(ufunc_call_info)
                              + ufunc->nargs * sizeof(PyArray_Descr *));
    // 如果分配失败，则抛出内存不足异常，跳转到finish标签
    if (call_info == NULL) {
        PyErr_NoMemory();
        goto finish;
    }
    call_info->strided_loop = NULL;
    call_info->auxdata = NULL;
    call_info->context = &call_info->_full_context;

    /*
     * 创建一个胶囊对象，使用"numpy_1.24_ufunc_call_info"作为名称，
     * 这个胶囊表明它可能在版本更新中发生变化（但不一定会变化）。
     * 这个胶囊在`ufunc._resolve_dtypes_and_context`的文档字符串中有描述。
     */
    // 创建一个PyCapsule对象capsule，封装call_info结构体，使用自定义释放函数free_ufunc_call_info
    PyObject *capsule = PyCapsule_New(
            call_info, "numpy_1.24_ufunc_call_info", &free_ufunc_call_info);
    // 如果创建失败，则释放call_info内存并跳转到finish标签
    if (capsule == NULL) {
        PyObject_Free(call_info);
        goto finish;
    }

    // 获取call_info的context字段
    PyArrayMethod_Context *context = call_info->context;

    // 增加ufunc的引用计数，并将其设置为context的caller
    Py_INCREF(ufunc);
    context->caller = (PyObject *)ufunc;
    // 增加ufuncimpl的引用计数，并将其设置为context的method
    Py_INCREF(ufuncimpl);
    context->method = ufuncimpl;
    // 设置context的descriptors字段为call_info结构体中的_descrs字段
    context->descriptors = call_info->_descrs;
    // 遍历操作描述数组，增加其引用计数，并设置到context的descriptors数组中
    for (int i=0; i < ufunc->nargs; i++) {
        Py_INCREF(operation_descrs[i]);
        ((PyArray_Descr **)context->descriptors)[i] = operation_descrs[i];
    }

    // 将结果打包成一个Python元组，包括result_dtype_tuple和capsule
    result = PyTuple_Pack(2, result_dtype_tuple, capsule);
    /* cleanup and return */
    // 清理并返回

    // 减少capsule的引用计数
    Py_DECREF(capsule);

  finish:
    // 恢复原始的NumPy促进状态
    set_npy_promotion_state(original_promotion_state);

    // 释放result_dtype_tuple的引用
    Py_XDECREF(result_dtype_tuple);
    // 释放signature、dummy_arrays、operation_descrs和DTypes数组的引用
    for (int i = 0; i < ufunc->nargs; i++) {
        Py_XDECREF(signature[i]);
        Py_XDECREF(dummy_arrays[i]);
        Py_XDECREF(operation_descrs[i]);
        Py_XDECREF(DTypes[i]);
    }

    // 返回结果
    return result;
}


static PyObject *
py_resolve_dtypes(PyUFuncObject *ufunc,
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    // 调用通用的 dtype 解析函数，不考虑上下文
    return py_resolve_dtypes_generic(ufunc, NPY_FALSE, args, len_args, kwnames);
}


static PyObject *
py_resolve_dtypes_and_context(PyUFuncObject *ufunc,
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    // 调用通用的 dtype 解析函数，并考虑上下文
    return py_resolve_dtypes_generic(ufunc, NPY_TRUE, args, len_args, kwnames);
}


static PyObject *
py_get_strided_loop(PyUFuncObject *ufunc,
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    // 准备参数解析器
    NPY_PREPARE_ARGPARSER;

    PyObject *call_info_obj;
    PyObject *fixed_strides_obj = Py_None;
    npy_intp fixed_strides[NPY_MAXARGS];

    // 解析参数
    if (npy_parse_arguments("_get_strided_loop", args, len_args, kwnames,
            "", NULL, &call_info_obj,
            "$fixed_strides", NULL, &fixed_strides_obj,
            NULL, NULL, NULL) < 0) {
        return NULL;
    }

    // 获取 ufunc_call_info 结构体指针
    ufunc_call_info *call_info = PyCapsule_GetPointer(
                call_info_obj, "numpy_1.24_ufunc_call_info");
    if (call_info == NULL) {
        /* 无法处理包含 NULL 的上下文... */
        assert(PyErr_Occurred());
        return NULL;
    }
    // 检查 strided_loop 是否已经被填充或使用
    if (call_info->strided_loop != NULL) {
        PyErr_SetString(PyExc_TypeError,
                "ufunc call_info has already been filled/used!");
        return NULL;
    }

    // 检查调用上下文是否与 ufunc 匹配
    if (call_info->context->caller != (PyObject *)ufunc) {
        PyErr_SetString(PyExc_TypeError,
                "calling get_strided_loop with incompatible context");
        return NULL;
    }

    /*
     * 严格转换 fixed_strides，可以是 None 或者 int 的元组
     */
    if (fixed_strides_obj == Py_None) {
        for (int i = 0; i < ufunc->nargs; i++) {
            fixed_strides[i] = NPY_MAX_INTP;
        }
    }
    else if (PyTuple_CheckExact(fixed_strides_obj)
            && PyTuple_Size(fixed_strides_obj) == ufunc->nargs) {
        for (int i = 0; i < ufunc->nargs; i++) {
            PyObject *stride = PyTuple_GET_ITEM(fixed_strides_obj, i);
            if (PyLong_CheckExact(stride)) {
                fixed_strides[i] = PyLong_AsSsize_t(stride);
                if (error_converting(fixed_strides[i])) {
                    return NULL;
                }
            }
            else if (stride == Py_None) {
                fixed_strides[i] = NPY_MAX_INTP;
            }
            else {
                PyErr_SetString(PyExc_TypeError,
                    "_get_strided_loop(): fixed_strides tuple must contain "
                    "Python ints or None");
                return NULL;
            }
        }
    }
    else {
        PyErr_SetString(PyExc_TypeError,
            "_get_strided_loop(): fixed_strides must be a tuple or None");
        return NULL;
    }

    // 设置数组方法的标志
    NPY_ARRAYMETHOD_FLAGS flags;
    # 如果获取步进循环信息的函数调用失败（返回值小于0），则返回空指针
    if (call_info->context->method->get_strided_loop(call_info->context,
            1, 0, fixed_strides, &call_info->strided_loop, &call_info->auxdata,
            &flags) < 0) {
        return NULL;
    }

    # 设置调用信息结构体中的 requires_pyapi 标志，根据 flags 中的位掩码 NPY_METH_REQUIRES_PYAPI 确定
    call_info->requires_pyapi = flags & NPY_METH_REQUIRES_PYAPI;
    
    # 设置调用信息结构体中的 no_floatingpoint_errors 标志，根据 flags 中的位掩码 NPY_METH_NO_FLOATINGPOINT_ERRORS 确定
    call_info->no_floatingpoint_errors = (
            flags & NPY_METH_NO_FLOATINGPOINT_ERRORS);

    # 返回 Python 中的 None 对象
    Py_RETURN_NONE;
static struct PyMethodDef ufunc_methods[] = {
    {"reduce",
        (PyCFunction)ufunc_reduce,
        METH_FASTCALL | METH_KEYWORDS, NULL },
    {"accumulate",
        (PyCFunction)ufunc_accumulate,
        METH_FASTCALL | METH_KEYWORDS, NULL },
    {"reduceat",
        (PyCFunction)ufunc_reduceat,
        METH_FASTCALL | METH_KEYWORDS, NULL },
    {"outer",
        (PyCFunction)ufunc_outer,
        METH_FASTCALL | METH_KEYWORDS, NULL},
    {"at",
        (PyCFunction)ufunc_at,
        METH_VARARGS, NULL},
    /* Lower level methods: */
    {"resolve_dtypes",
        (PyCFunction)py_resolve_dtypes,
        METH_FASTCALL | METH_KEYWORDS, NULL},
    /*
     * The following two functions are public API, but underscored since they
     * are C-user specific and allow direct access to the core of ufunc loops.
     * (See their documentation for API stability.)
     */
    {"_resolve_dtypes_and_context",
        (PyCFunction)py_resolve_dtypes_and_context,
        METH_FASTCALL | METH_KEYWORDS, NULL},
    {"_get_strided_loop",
        (PyCFunction)py_get_strided_loop,
        METH_FASTCALL | METH_KEYWORDS, NULL},
    {NULL, NULL, 0, NULL}           /* sentinel */
};
/*
 * 定义了一个静态的 PyMethodDef 结构体数组，用于定义一系列 Python C 函数。
 * 每个数组元素包括函数名、函数指针、调用方式（方法调用类型）以及可选的文档字符串。
 * 数组的最后一个元素 {NULL, NULL, 0, NULL} 作为哨兵用于表示数组的结尾。
 */


/******************************************************************************
 ***                           UFUNC GETSET                                 ***
 *****************************************************************************/


static char
_typecharfromnum(int num) {
    PyArray_Descr *descr;
    char ret;

    descr = PyArray_DescrFromType(num);
    ret = descr->type;
    Py_DECREF(descr);
    return ret;
}
/*
 * 根据给定的数据类型编码（num），返回对应的字符表示。
 * 使用 PyArray_DescrFromType 函数获取数据类型描述符，从中提取类型字符后释放描述符。
 */


static PyObject *
ufunc_get_doc(PyUFuncObject *ufunc, void *NPY_UNUSED(ignored))
{
    PyObject *doc;

    npy_cache_import(
        "numpy._core._internal",
        "_ufunc_doc_signature_formatter",
        &npy_thread_unsafe_state._ufunc_doc_signature_formatter);

    if (npy_thread_unsafe_state._ufunc_doc_signature_formatter == NULL) {
        return NULL;
    }

    /*
     * Put docstring first or FindMethod finds it... could so some
     * introspection on name and nin + nout to automate the first part
     * of it the doc string shouldn't need the calling convention
     */
    doc = PyObject_CallFunctionObjArgs(npy_thread_unsafe_state._ufunc_doc_signature_formatter,
                                       (PyObject *)ufunc, NULL);
    if (doc == NULL) {
        return NULL;
    }
    if (ufunc->doc != NULL) {
        Py_SETREF(doc, PyUnicode_FromFormat("%S\n\n%s", doc, ufunc->doc));
    }
    return doc;
}
/*
 * 获取给定 PyUFuncObject 对象的文档字符串。
 * 使用内部函数 npy_cache_import 加载 _ufunc_doc_signature_formatter。
 * 调用 _ufunc_doc_signature_formatter 函数生成文档字符串，如果存在的话，添加到 ufunc->doc 后返回。
 */


static PyObject *
ufunc_get_nin(PyUFuncObject *ufunc, void *NPY_UNUSED(ignored))
{
    return PyLong_FromLong(ufunc->nin);
}
/*
 * 获取给定 PyUFuncObject 对象的输入参数数量（nin）并封装成 Python 的长整型对象返回。
 */


static PyObject *
ufunc_get_nout(PyUFuncObject *ufunc, void *NPY_UNUSED(ignored))
{
    return PyLong_FromLong(ufunc->nout);
}
/*
 * 获取给定 PyUFuncObject 对象的输出参数数量（nout）并封装成 Python 的长整型对象返回。
 */


static PyObject *
ufunc_get_nargs(PyUFuncObject *ufunc, void *NPY_UNUSED(ignored))
{
    return PyLong_FromLong(ufunc->nargs);
}
/*
 * 获取给定 PyUFuncObject 对象的总参数数量（nargs）并封装成 Python 的长整型对象返回。
 */

static PyObject *
static PyObject *
ufunc_get_ntypes(PyUFuncObject *ufunc, void *NPY_UNUSED(ignored))
{
    // 返回一个包含 ufunc 的 ntypes 属性的 Python 整数对象
    return PyLong_FromLong(ufunc->ntypes);
}

static PyObject *
ufunc_get_types(PyUFuncObject *ufunc, void *NPY_UNUSED(ignored))
{
    /* 返回一个列表，其中包含按照输入到输出分组的类型 */
    PyObject *list;
    PyObject *str;
    int k, j, n, nt = ufunc->ntypes;
    int ni = ufunc->nin;
    int no = ufunc->nout;
    char *t;
    list = PyList_New(nt);
    if (list == NULL) {
        return NULL;
    }
    t = PyArray_malloc(no+ni+2);
    n = 0;
    for (k = 0; k < nt; k++) {
        for (j = 0; j<ni; j++) {
            t[j] = _typecharfromnum(ufunc->types[n]);
            n++;
        }
        t[ni] = '-';
        t[ni+1] = '>';
        for (j = 0; j < no; j++) {
            t[ni + 2 + j] = _typecharfromnum(ufunc->types[n]);
            n++;
        }
        str = PyUnicode_FromStringAndSize(t, no + ni + 2);
        PyList_SET_ITEM(list, k, str);
    }
    PyArray_free(t);
    return list;
}

static PyObject *
ufunc_get_name(PyUFuncObject *ufunc, void *NPY_UNUSED(ignored))
{
    // 返回一个包含 ufunc 的 name 属性的 Python 字符串对象
    return PyUnicode_FromString(ufunc->name);
}

static PyObject *
ufunc_get_identity(PyUFuncObject *ufunc, void *NPY_UNUSED(ignored))
{
    npy_bool reorderable;
    // 返回一个包含 ufunc 的默认 identity 的 Python 对象
    return PyUFunc_GetDefaultIdentity(ufunc, &reorderable);
}

static PyObject *
ufunc_get_signature(PyUFuncObject *ufunc, void *NPY_UNUSED(ignored))
{
    if (!ufunc->core_enabled) {
        // 如果 ufunc 的 core_enabled 属性为假，返回 None
        Py_RETURN_NONE;
    }
    // 否则返回一个包含 ufunc 的 core_signature 属性的 Python 字符串对象
    return PyUnicode_FromString(ufunc->core_signature);
}

#undef _typecharfromnum

static PyGetSetDef ufunc_getset[] = {
    {"__doc__",
        (getter)ufunc_get_doc,
        NULL, NULL, NULL},
    {"nin",
        (getter)ufunc_get_nin,
        NULL, NULL, NULL},
    {"nout",
        (getter)ufunc_get_nout,
        NULL, NULL, NULL},
    {"nargs",
        (getter)ufunc_get_nargs,
        NULL, NULL, NULL},
    {"ntypes",
        (getter)ufunc_get_ntypes,
        NULL, NULL, NULL},
    {"types",
        (getter)ufunc_get_types,
        NULL, NULL, NULL},
    {"__name__",
        (getter)ufunc_get_name,
        NULL, NULL, NULL},
    {"identity",
        (getter)ufunc_get_identity,
        NULL, NULL, NULL},
    {"signature",
        (getter)ufunc_get_signature,
        NULL, NULL, NULL},
    {NULL, NULL, NULL, NULL, NULL},  /* Sentinel */
};
    .tp_flags = Py_TPFLAGS_DEFAULT |  // 设置对象的默认标志
        _Py_TPFLAGS_HAVE_VECTORCALL |  // 启用向量调用特性
        Py_TPFLAGS_HAVE_GC,  // 启用垃圾回收特性
    .tp_traverse = (traverseproc)ufunc_traverse,  // 设置对象的遍历函数为 ufunc_traverse
    .tp_methods = ufunc_methods,  // 设置对象的方法集合为 ufunc_methods
    .tp_getset = ufunc_getset,  // 设置对象的属性访问器为 ufunc_getset
};

/* End of code for ufunc objects */
```