# `.\numpy\numpy\_core\src\umath\wrapping_array_method.c`

```py
/*
 * This file defines most of the machinery in order to wrap an existing ufunc
 * loop for use with a different set of dtypes.
 *
 * There are two approaches for this, one is to teach the NumPy core about
 * the possibility that the loop descriptors do not match exactly the result
 * descriptors.
 * The other is to handle this fully by "wrapping", so that NumPy core knows
 * nothing about this going on.
 * The slight difficulty here is that `context` metadata needs to be mutated.
 * It also adds a tiny bit of overhead, since we have to "fix" the descriptors
 * and unpack the auxdata.
 *
 * This means that this currently needs to live within NumPy, as it needs both
 * extensive API exposure to do it outside, as well as some thoughts on how to
 * expose the `context` without breaking ABI forward compatibility.
 * (I.e. we probably need to allocate the context and provide a copy function
 * or so.)
 */

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#define _UMATHMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "numpy/ndarraytypes.h"

#include "npy_pycompat.h"
#include "common.h"
#include "array_method.h"
#include "legacy_array_method.h"
#include "dtypemeta.h"
#include "dispatching.h"

/*
 * Function: wrapping_method_resolve_descriptors
 * ---------------------------------------------
 * Resolve descriptors for a wrapped ufunc method, handling dtype translation
 * and context mutation.
 *
 * Parameters:
 * - self: PyArrayMethodObject instance representing the wrapped ufunc method
 * - dtypes: Array of PyArray_DTypeMeta pointers
 * - given_descrs: Array of given descriptors
 * - loop_descrs: Output array for loop descriptors
 * - view_offset: View offset as npy_intp pointer
 *
 * Returns:
 * - NPY_CASTING value indicating the casting behavior
 */
static NPY_CASTING
wrapping_method_resolve_descriptors(
        PyArrayMethodObject *self,
        PyArray_DTypeMeta *const dtypes[],
        PyArray_Descr *const given_descrs[],
        PyArray_Descr *loop_descrs[],
        npy_intp *view_offset)
{
    int nin = self->nin, nout = self->nout, nargs = nin + nout;
    PyArray_Descr *orig_given_descrs[NPY_MAXARGS];
    PyArray_Descr *orig_loop_descrs[NPY_MAXARGS];

    // Translate given descriptors using wrapped method's types
    if (self->translate_given_descrs(
            nin, nout, self->wrapped_dtypes,
            given_descrs, orig_given_descrs) < 0) {
        return -1;
    }

    // Resolve descriptors using wrapped method, obtain casting behavior
    NPY_CASTING casting = self->wrapped_meth->resolve_descriptors(
            self->wrapped_meth, self->wrapped_dtypes,
            orig_given_descrs, orig_loop_descrs, view_offset);

    // Release original given descriptors
    for (int i = 0; i < nargs; i++) {
        Py_XDECREF(orig_given_descrs[i]);
    }

    // Return immediately if resolve_descriptors failed
    if (casting < 0) {
        return -1;
    }

    // Translate loop descriptors for the wrapped method
    int res = self->translate_loop_descrs(
            nin, nout, dtypes, given_descrs, orig_loop_descrs, loop_descrs);

    // Release original loop descriptors
    for (int i = 0; i < nargs; i++) {
        Py_DECREF(orig_loop_descrs[i]);
    }

    // Return immediately if translate_loop_descrs failed
    if (res < 0) {
        return -1;
    }

    return casting;
}

/*
 * Structure: wrapping_auxdata
 * ---------------------------
 * Auxiliary data structure for wrapping ufunc method, encapsulating original
 * context, loop, auxdata, and descriptors.
 */
typedef struct {
    NpyAuxData base;
    PyArrayMethod_Context orig_context;  // Original method context
    PyArrayMethod_StridedLoop *orig_loop;  // Original strided loop
    NpyAuxData *orig_auxdata;  // Original auxiliary data
    PyArray_Descr *descriptors[NPY_MAXARGS];  // Array of descriptors
} wrapping_auxdata;

#define WRAPPING_AUXDATA_FREELIST_SIZE 5
static int wrapping_auxdata_freenum = 0;
static wrapping_auxdata *wrapping_auxdata_freelist[WRAPPING_AUXDATA_FREELIST_SIZE] = {NULL};

/*
 * Function: wrapping_auxdata_free
 * -------------------------------
 * Free function for releasing wrapping_auxdata resources.
 *
 * Parameters:
 * - wrapping_auxdata: Pointer to wrapping_auxdata instance to be freed
 */
static void
wrapping_auxdata_free(wrapping_auxdata *wrapping_auxdata)
{
    /* Free auxdata, everything else is borrowed: */

    /* Free the wrapping_auxdata instance */

    /* Release the base NpyAuxData */
    NPY_AUXDATA_FREE((NpyAuxData *)wrapping_auxdata);

    /* Reset the original context and loop pointers to NULL */
    wrapping_auxdata->orig_context = NULL;
    wrapping_auxdata->orig_loop = NULL;

    /* Release the original auxiliary data if it exists */
    if (wrapping_auxdata->orig_auxdata != NULL) {
        NPY_AUXDATA_FREE(wrapping_auxdata->orig_auxdata);
        wrapping_auxdata->orig_auxdata = NULL;
    }

    /* Reset descriptors pointers to NULL */
    for (int i = 0; i < NPY_MAXARGS; i++) {
        wrapping_auxdata->descriptors[i] = NULL;
    }

    /* If there is space in the freelist, store this instance for reuse */
    if (wrapping_auxdata_freenum < WRAPPING_AUXDATA_FREELIST_SIZE) {
        wrapping_auxdata_freelist[wrapping_auxdata_freenum++] = wrapping_auxdata;
    } else {
        /* Otherwise, free the memory directly */
        PyMem_FREE(wrapping_auxdata);
    }
}


注释：
    # 释放 wrapping_auxdata 结构体中 orig_auxdata 指针指向的内存块
    NPY_AUXDATA_FREE(wrapping_auxdata->orig_auxdata);
    # 将 wrapping_auxdata 结构体的 orig_auxdata 指针置为 NULL，防止悬空引用
    wrapping_auxdata->orig_auxdata = NULL;

    # 检查 wrapping_auxdata_freelist 是否还有空间存储 wrapping_auxdata
    if (wrapping_auxdata_freenum < WRAPPING_AUXDATA_FREELIST_SIZE) {
        # 将 wrapping_auxdata 添加到 wrapping_auxdata_freelist 中
        wrapping_auxdata_freelist[wrapping_auxdata_freenum] = wrapping_auxdata;
        # 更新 wrapping_auxdata_freenum 指示下一个可用位置
        wrapping_auxdata_freenum++;
    }
    else {
        # 如果 wrapping_auxdata_freelist 已满，则释放 wrapping_auxdata 占用的内存
        PyMem_Free(wrapping_auxdata);
    }
}

/*
 * 获取可重用的包装辅助数据结构
 * 如果有空闲的辅助数据结构可用，则返回一个；否则分配一个新的
 */
static wrapping_auxdata *
get_wrapping_auxdata(void)
{
    wrapping_auxdata *res;
    if (wrapping_auxdata_freenum > 0) {
        // 如果有空闲的辅助数据结构可用，则从空闲列表中取出一个
        wrapping_auxdata_freenum--;
        res = wrapping_auxdata_freelist[wrapping_auxdata_freenum];
    }
    else {
        // 否则，分配一个新的辅助数据结构
        res = PyMem_Calloc(1, sizeof(wrapping_auxdata));
        if (res < 0) {
            // 内存分配失败时，设置错误状态并返回空指针
            PyErr_NoMemory();
            return NULL;
        }
        // 设置辅助数据结构的释放函数
        res->base.free = (void *)wrapping_auxdata_free;
        // 将原始上下文的描述符设置为辅助数据结构的描述符数组
        res->orig_context.descriptors = res->descriptors;
    }

    return res;
}

/*
 * 包装方法的分块循环函数
 * 如果上下文中有更多的东西被存储，可能需要在这里复制它们。但当前情况下不需要。
 */
static int
wrapping_method_strided_loop(PyArrayMethod_Context *NPY_UNUSED(context),
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], wrapping_auxdata *auxdata)
{
    return auxdata->orig_loop(
            &auxdata->orig_context, data, dimensions, strides,
            auxdata->orig_auxdata);
}

/*
 * 获取包装方法的循环函数
 * 设置包装方法的循环函数及其传输数据
 */
static int
wrapping_method_get_loop(
        PyArrayMethod_Context *context,
        int aligned, int move_references, const npy_intp *strides,
        PyArrayMethod_StridedLoop **out_loop, NpyAuxData **out_transferdata,
        NPY_ARRAYMETHOD_FLAGS *flags)
{
    assert(move_references == 0);  /* 仅在内部用于“decref”函数 */

    int nin = context->method->nin, nout = context->method->nout;

    // 获取可重用的包装辅助数据结构
    wrapping_auxdata *auxdata = get_wrapping_auxdata();
    if (auxdata == NULL) {
        return -1;
    }

    // 设置原始上下文的方法为被包装的方法
    auxdata->orig_context.method = context->method->wrapped_meth;
    auxdata->orig_context.caller = context->caller;

    // 将描述符从被包装方法的格式转换回原始格式
    if (context->method->translate_given_descrs(
            nin, nout, context->method->wrapped_dtypes, context->descriptors,
            (PyArray_Descr **)auxdata->orig_context.descriptors) < 0) {
        NPY_AUXDATA_FREE((NpyAuxData *)auxdata);
        return -1;
    }

    // 获取被包装方法的分块循环
    if (context->method->wrapped_meth->get_strided_loop(
            &auxdata->orig_context, aligned, 0, strides,
            &auxdata->orig_loop, &auxdata->orig_auxdata,
            flags) < 0) {
        NPY_AUXDATA_FREE((NpyAuxData *)auxdata);
        return -1;
    }

    // 返回包装方法的分块循环及其传输数据
    *out_loop = (PyArrayMethod_StridedLoop *)&wrapping_method_strided_loop;
    *out_transferdata = (NpyAuxData *)auxdata;
    return 0;
}

/*
 * 包装原始身份函数，需要将描述符翻译回原始的描述符，并提供一个“原始”上下文（与get_loop完全相同）
 * 我们再次假设描述符翻译是快速的。
 */
static int
wrapping_method_get_identity_function(
        PyArrayMethod_Context *context, npy_bool reduction_is_empty,
        char *item)
{
    /* 复制上下文，并替换描述符 */
    PyArrayMethod_Context orig_context = *context;
    PyArray_Descr *orig_descrs[NPY_MAXARGS];
    orig_context.descriptors = orig_descrs;
    orig_context.method = context->method->wrapped_meth;
    // 获取输入和输出参数的数量
    int nin = context->method->nin, nout = context->method->nout;
    // 获取数据类型元信息数组的指针
    PyArray_DTypeMeta **dtypes = context->method->wrapped_dtypes;

    // 调用方法对象的 translate_given_descrs 函数，将描述符转换为原始描述符
    if (context->method->translate_given_descrs(
            nin, nout, dtypes, context->descriptors, orig_descrs) < 0) {
        // 如果转换失败，返回 -1
        return -1;
    }

    // 调用方法对象的 wrapped_meth 成员的 get_reduction_initial 函数，获取初始值
    int res = context->method->wrapped_meth->get_reduction_initial(
            &orig_context, reduction_is_empty, item);

    // 循环释放原始描述符的引用计数
    for (int i = 0; i < nin + nout; i++) {
        Py_DECREF(orig_descrs);
    }

    // 返回初始值获取的结果
    return res;
    /*UFUNC_API
     * 允许在现有的ufunc循环周围创建一个相当轻量级的包装器。
     * 主要用于单位，因为它目前有些限制，即强制您不能使用另一个ufunc的循环。
     *
     * @param ufunc_obj
     * @param new_dtypes
     * @param wrapped_dtypes
     * @param translate_given_descrs 参见typedef注释
     * @param translate_loop_descrs 参见typedef注释
     * @return 成功返回0，失败返回-1
     */
    NPY_NO_EXPORT int
    PyUFunc_AddWrappingLoop(PyObject *ufunc_obj,
            PyArray_DTypeMeta *new_dtypes[], PyArray_DTypeMeta *wrapped_dtypes[],
            PyArrayMethod_TranslateGivenDescriptors *translate_given_descrs,
            PyArrayMethod_TranslateLoopDescriptors *translate_loop_descrs)
    {
        int res = -1;
        PyUFuncObject *ufunc = (PyUFuncObject *)ufunc_obj;
        PyObject *wrapped_dt_tuple = NULL;
        PyObject *new_dt_tuple = NULL;
        PyArrayMethodObject *meth = NULL;

        if (!PyObject_TypeCheck(ufunc_obj, &PyUFunc_Type)) {
            PyErr_SetString(PyExc_TypeError,
                    "ufunc object passed is not a ufunc!");
            return -1;
        }

        wrapped_dt_tuple = PyArray_TupleFromItems(
                ufunc->nargs, (PyObject **)wrapped_dtypes, 1);
        if (wrapped_dt_tuple == NULL) {
            goto finish;
        }

        PyArrayMethodObject *wrapped_meth = NULL;
        PyObject *loops = ufunc->_loops;
        Py_ssize_t length = PyList_Size(loops);
        for (Py_ssize_t i = 0; i < length; i++) {
            PyObject *item = PyList_GetItemRef(loops, i);
            PyObject *cur_DType_tuple = PyTuple_GetItem(item, 0);
            Py_DECREF(item);
            int cmp = PyObject_RichCompareBool(cur_DType_tuple, wrapped_dt_tuple, Py_EQ);
            if (cmp < 0) {
                goto finish;
            }
            if (cmp == 0) {
                continue;
            }
            wrapped_meth = (PyArrayMethodObject *)PyTuple_GET_ITEM(item, 1);
            if (!PyObject_TypeCheck(wrapped_meth, &PyArrayMethod_Type)) {
                PyErr_SetString(PyExc_TypeError,
                        "Matching loop was not an ArrayMethod.");
                goto finish;
            }
            break;
        }
        if (wrapped_meth == NULL) {
            PyErr_Format(PyExc_TypeError,
                    "Did not find the to-be-wrapped loop in the ufunc with given "
                    "DTypes. Received wrapping types: %S", wrapped_dt_tuple);
            goto finish;
        }

        PyType_Slot slots[] = {
            {NPY_METH_resolve_descriptors, &wrapping_method_resolve_descriptors},
            {NPY_METH_get_loop, &wrapping_method_get_loop},
            {NPY_METH_get_reduction_initial,
                &wrapping_method_get_identity_function},
            {0, NULL}
        };

        PyArrayMethod_Spec spec = {
            .name = "wrapped-method",
            .nin = wrapped_meth->nin,
            .nout = wrapped_meth->nout,
            .casting = wrapped_meth->casting,
            .flags = wrapped_meth->flags,
            .dtypes = new_dtypes,
            .slots = slots,
        };
        PyBoundArrayMethodObject *bmeth = PyArrayMethod_FromSpec_int(&spec, 1);
    # 如果 bmeth 为 NULL，则跳转到 finish 标签处，结束函数
    if (bmeth == NULL) {
        goto finish;
    }

    # 增加 bmeth->method 的引用计数，并将其赋值给 meth
    Py_INCREF(bmeth->method);
    meth = bmeth->method;
    # 将 bmeth 置为 NULL
    Py_SETREF(bmeth, NULL);

    /* 完成新 ArrayMethod 的“包装”部分 */
    # 分配内存以存储 wrapped_dtypes 数组，大小为 ufunc->nargs 个 PyArray_DTypeMeta* 元素的空间
    meth->wrapped_dtypes = PyMem_Malloc(ufunc->nargs * sizeof(PyArray_DTypeMeta *));
    if (meth->wrapped_dtypes == NULL) {
        goto finish;
    }

    # 增加 wrapped_meth 的引用计数，并将其赋值给 meth->wrapped_meth
    Py_INCREF(wrapped_meth);
    meth->wrapped_meth = wrapped_meth;
    # 设置 meth 的 translate_given_descrs 和 translate_loop_descrs 属性
    meth->translate_given_descrs = translate_given_descrs;
    meth->translate_loop_descrs = translate_loop_descrs;
    # 复制 wrapped_dtypes 数组的每个元素到 meth->wrapped_dtypes 数组中
    for (int i = 0; i < ufunc->nargs; i++) {
        Py_XINCREF(wrapped_dtypes[i]);
        meth->wrapped_dtypes[i] = wrapped_dtypes[i];
    }

    # 从 new_dtypes 中创建一个包含 ufunc->nargs 个元素的元组 new_dt_tuple
    new_dt_tuple = PyArray_TupleFromItems(
            ufunc->nargs, (PyObject **)new_dtypes, 1);
    # 如果创建元组失败，则跳转到 finish 标签处，结束函数
    if (new_dt_tuple == NULL) {
        goto finish;
    }

    # 创建一个元组 info，包含 new_dt_tuple 和 meth 两个元素
    PyObject *info = PyTuple_Pack(2, new_dt_tuple, meth);
    # 如果创建元组 info 失败，则跳转到 finish 标签处，结束函数
    if (info == NULL) {
        goto finish;
    }

    # 将 info 作为参数调用 PyUFunc_AddLoop 函数，并将结果赋值给 res
    res = PyUFunc_AddLoop(ufunc, info, 0);
    # 减少 info 的引用计数
    Py_DECREF(info);

  finish:
    # 递减并释放 wrapped_dt_tuple 的引用计数
    Py_XDECREF(wrapped_dt_tuple);
    # 递减并释放 new_dt_tuple 的引用计数
    Py_XDECREF(new_dt_tuple);
    # 递减并释放 meth 的引用计数
    Py_XDECREF(meth);
    # 返回 res 变量
    return res;
}



# 这行代码是一个单独的右花括号 '}'，用于结束一个代码块或数据结构的定义。
# 在很多编程语言中，花括号用于界定代码块的范围，例如函数、循环、条件语句等。
# 在这段代码中，它是代码块的结尾，可能是一个函数、类、循环或条件语句的末尾。
```