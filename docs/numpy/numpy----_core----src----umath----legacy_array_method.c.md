# `.\numpy\numpy\_core\src\umath\legacy_array_method.c`

```py
/*
 * This file defines most of the machinery in order to wrap legacy style
 * ufunc loops into new style arraymethods.
 */

#define NPY_NO_DEPRECATED_API NPY_API_VERSION  // Define to use only non-deprecated NumPy API
#define _MULTIARRAYMODULE  // Define to include multiarray module functionality
#define _UMATHMODULE  // Define to include umath module functionality

#define PY_SSIZE_T_CLEAN  // Define to use Python 3's clean API for Py_ssize_t
#include <Python.h>  // Include Python's C API header

#include "numpy/ndarraytypes.h"  // Include NumPy's ndarraytypes header

#include "convert_datatype.h"  // Include header for datatype conversion
#include "array_method.h"  // Include header for array methods
#include "array_coercion.h"  // Include header for array coercion
#include "dtype_transfer.h"  // Include header for dtype transfer
#include "legacy_array_method.h"  // Include header for legacy array methods
#include "dtypemeta.h"  // Include header for dtype metadata

#include "ufunc_object.h"  // Include header for ufunc object
#include "ufunc_type_resolution.h"  // Include header for ufunc type resolution

// Define a structure for auxiliary data used in legacy array methods
typedef struct {
    NpyAuxData base;  // Base structure for auxiliary data
    PyUFuncGenericFunction loop;  // Legacy ufunc loop function pointer
    void *user_data;  // Additional user data associated with the loop
    int pyerr_check;  // Flag indicating whether to check PyErr_Occurred()
} legacy_array_method_auxdata;

// Use a free list for caching auxiliary data, only if GIL is enabled
#ifndef Py_GIL_DISABLED
#define NPY_LOOP_DATA_CACHE_SIZE 5  // Size of the loop data cache
static int loop_data_num_cached = 0;
static legacy_array_method_auxdata *loop_data_cache[NPY_LOOP_DATA_CACHE_SIZE];
#else
#define NPY_LOOP_DATA_CACHE_SIZE 0
#endif

// Free function for legacy array method auxiliary data
static void
legacy_array_method_auxdata_free(NpyAuxData *data)
{
#if NPY_LOOP_DATA_CACHE_SIZE > 0
    if (loop_data_num_cached < NPY_LOOP_DATA_CACHE_SIZE) {
        loop_data_cache[loop_data_num_cached] = (
                (legacy_array_method_auxdata *)data);
        loop_data_num_cached++;
    }
    else
#endif
    {
        PyMem_Free(data);  // Free the memory allocated for auxiliary data
    }
}

// Function to allocate new loop data
NpyAuxData *
get_new_loop_data(
        PyUFuncGenericFunction loop, void *user_data, int pyerr_check)
{
    legacy_array_method_auxdata *data;
#if NPY_LOOP_DATA_CACHE_SIZE > 0
    if (NPY_LIKELY(loop_data_num_cached > 0)) {
        loop_data_num_cached--;
        data = loop_data_cache[loop_data_num_cached];
    }
    else
#endif
    {
        data = PyMem_Malloc(sizeof(legacy_array_method_auxdata));  // Allocate memory for new auxiliary data
        if (data == NULL) {
            return NULL;  // Return NULL if memory allocation fails
        }
        data->base.free = legacy_array_method_auxdata_free;  // Set free function for the auxiliary data
        data->base.clone = NULL;  // No need for cloning (at least for now)
    }
    data->loop = loop;  // Assign legacy ufunc loop function pointer
    data->user_data = user_data;  // Assign additional user data
    data->pyerr_check = pyerr_check;  // Set flag for PyErr_Occurred() checking
    return (NpyAuxData *)data;  // Return the allocated auxiliary data
}

#undef NPY_LOOP_DATA_CACHE_SIZE

/*
 * This is a thin wrapper around the legacy loop signature.
 */
static int
generic_wrapped_legacy_loop(PyArrayMethod_Context *NPY_UNUSED(context),
        char *const *data, const npy_intp *dimensions, const npy_intp *strides,
        NpyAuxData *auxdata)
{
    legacy_array_method_auxdata *ldata = (legacy_array_method_auxdata *)auxdata;

    ldata->loop((char **)data, dimensions, strides, ldata->user_data);  // Call the legacy ufunc loop
    if (ldata->pyerr_check && PyErr_Occurred()) {  // Check for Python errors if flag is set
        return -1;  // Return -1 on error
    }
    return 0;  // Return 0 indicating success
}
/*
 * Signal that the old type-resolution function must be used to resolve
 * the descriptors (mainly/only used for datetimes due to the unit).
 *
 * ArrayMethod's are expected to implement this, but it is too tricky
 * to support properly.  So we simply set an error that should never be seen.
 */
NPY_NO_EXPORT NPY_CASTING
wrapped_legacy_resolve_descriptors(PyArrayMethodObject *NPY_UNUSED(self),
        PyArray_DTypeMeta *const NPY_UNUSED(dtypes[]),
        PyArray_Descr *const NPY_UNUSED(given_descrs[]),
        PyArray_Descr *NPY_UNUSED(loop_descrs[]),
        npy_intp *NPY_UNUSED(view_offset))
{
    // 设置运行时错误，指出不能使用旧的 ArrayMethod 包装而不调用 ufunc 本身
    PyErr_SetString(PyExc_RuntimeError,
            "cannot use legacy wrapping ArrayMethod without calling the ufunc "
            "itself.  If this error is hit, the solution will be to port the "
            "legacy ufunc loop implementation to the new API.");
    return -1; // 返回错误代码 -1
}

/*
 * Much the same as the default type resolver, but tries a bit harder to
 * preserve metadata.
 */
static NPY_CASTING
simple_legacy_resolve_descriptors(
        PyArrayMethodObject *method,
        PyArray_DTypeMeta *const *dtypes,
        PyArray_Descr *const *given_descrs,
        PyArray_Descr **output_descrs,
        npy_intp *NPY_UNUSED(view_offset))
{
    int i = 0;
    int nin = method->nin; // 获取输入数量
    int nout = method->nout; // 获取输出数量

    if (nin == 2 && nout == 1 && given_descrs[2] != NULL
            && dtypes[0] == dtypes[2]) {
        /*
         * Could be a reduction, which requires `descr[0] is descr[2]`
         * (identity) at least currently. This is because `op[0] is op[2]`.
         * (If the output descriptor is not passed, the below works.)
         */
        // 确保给定的描述符是规范的
        output_descrs[2] = NPY_DT_CALL_ensure_canonical(given_descrs[2]);
        if (output_descrs[2] == NULL) {
            Py_CLEAR(output_descrs[2]);
            return -1; // 失败，返回错误代码 -1
        }
        Py_INCREF(output_descrs[2]); // 增加引用计数
        output_descrs[0] = output_descrs[2]; // 设置第一个输出描述符为第三个描述符的规范版本
        if (dtypes[1] == dtypes[2]) {
            /* Same for the second one (accumulation is stricter) */
            Py_INCREF(output_descrs[2]); // 增加引用计数
            output_descrs[1] = output_descrs[2]; // 设置第二个输出描述符为第三个描述符的规范版本
        }
        else {
            // 确保给定的描述符是规范的
            output_descrs[1] = NPY_DT_CALL_ensure_canonical(given_descrs[1]);
            if (output_descrs[1] == NULL) {
                i = 2;
                goto fail; // 失败，跳转到失败处理标签
            }
        }
        return NPY_NO_CASTING; // 返回无需转换
    }

    for (; i < nin + nout; i++) {
        if (given_descrs[i] != NULL) {
            // 确保给定的描述符是规范的
            output_descrs[i] = NPY_DT_CALL_ensure_canonical(given_descrs[i]);
        }
        else if (dtypes[i] == dtypes[0] && i > 0) {
            /* Preserve metadata from the first operand if same dtype */
            Py_INCREF(output_descrs[0]); // 增加引用计数
            output_descrs[i] = output_descrs[0]; // 使用第一个操作数的元数据
        }
        else {
            // 使用默认的描述符
            output_descrs[i] = NPY_DT_CALL_default_descr(dtypes[i]);
        }
        if (output_descrs[i] == NULL) {
            goto fail; // 失败，跳转到失败处理标签
        }
    }

    return NPY_NO_CASTING; // 返回无需转换

  fail:
    // 失败处理
    # 逆向遍历循环，从 i 的初始值开始，每次递减直到 i 大于等于 0
    for (; i >= 0; i--) {
        # 使用 Py_CLEAR 宏来清空 output_descrs 数组中索引为 i 的元素，释放其引用计数
        Py_CLEAR(output_descrs[i]);
    }
    # 返回 -1，表示函数执行出错或异常
    return -1;
/*
 * This function retrieves the legacy inner-loop for a ufunc. If performance is
 * slow, caching might be considered.
 */
NPY_NO_EXPORT int
get_wrapped_legacy_ufunc_loop(PyArrayMethod_Context *context,
        int aligned, int move_references,
        const npy_intp *NPY_UNUSED(strides),
        PyArrayMethod_StridedLoop **out_loop,
        NpyAuxData **out_transferdata,
        NPY_ARRAYMETHOD_FLAGS *flags)
{
    assert(aligned);  // Ensure 'aligned' flag is true
    assert(!move_references);  // Ensure 'move_references' flag is false

    if (context->caller == NULL ||
            !PyObject_TypeCheck(context->caller, &PyUFunc_Type)) {
        PyErr_Format(PyExc_RuntimeError,
                "cannot call %s without its ufunc as caller context.",
                context->method->name);
        return -1;  // Return error if caller is not a valid PyUFunc_Type
    }

    PyUFuncObject *ufunc = (PyUFuncObject *)context->caller;
    void *user_data;
    int needs_api = 0;

    PyUFuncGenericFunction loop = NULL;
    /* Note that `needs_api` is not reliable (it was in fact unused normally) */
    // Attempt to select the default legacy inner loop for the ufunc
    if (PyUFunc_DefaultLegacyInnerLoopSelector(ufunc,
            context->descriptors, &loop, &user_data, &needs_api) < 0) {
        return -1;  // Return error if selecting the loop fails
    }
    *flags = context->method->flags & NPY_METH_RUNTIME_FLAGS;
    if (needs_api) {
        *flags |= NPY_METH_REQUIRES_PYAPI;  // Set flag if API access is needed
    }

    *out_loop = &generic_wrapped_legacy_loop;  // Assign the generic legacy loop
    // Obtain transfer data for the loop
    *out_transferdata = get_new_loop_data(
            loop, user_data, (*flags & NPY_METH_REQUIRES_PYAPI) != 0);
    if (*out_transferdata == NULL) {
        PyErr_NoMemory();  // Return error if memory allocation fails
        return -1;
    }
    return 0;  // Return success
}



/*
 * This function copies the cached initial value for the ufunc method.
 * It assumes all internal numeric types are trivially copied.
 */
static int
copy_cached_initial(
        PyArrayMethod_Context *context, npy_bool NPY_UNUSED(reduction_is_empty),
        void *initial)
{
    // Copy the legacy initial value from method context
    memcpy(initial, context->method->legacy_initial,
           context->descriptors[0]->elsize);
    return 1;  // Return success
}


/*
 * This function attempts to obtain the initial value from the calling ufunc.
 * It is only called when necessary, particularly for internal numeric dtypes.
 */
static int
get_initial_from_ufunc(
        PyArrayMethod_Context *context, npy_bool reduction_is_empty,
        void *initial)
{
    if (context->caller == NULL
            || !PyObject_TypeCheck(context->caller, &PyUFunc_Type)) {
        /* Impossible in NumPy 1.24;  guard in case it becomes possible. */
        PyErr_SetString(PyExc_ValueError,
                "getting initial failed because it can only done for legacy "
                "ufunc loops when the ufunc is provided.");
        return -1;  // Return error if caller is not a valid PyUFunc_Type
    }
    // Check if reordering is possible for the dtype
    npy_bool reorderable;
    // 获取当前上下文中的默认标识对象，用于指定的通用函数调用
    PyObject *identity_obj = PyUFunc_GetDefaultIdentity(
            (PyUFuncObject *)context->caller, &reorderable);
    // 如果获取失败，则返回错误代码
    if (identity_obj == NULL) {
        return -1;
    }
    // 如果默认标识对象是 Py_None，表示通用函数没有默认标识（理论上不应发生）
    if (identity_obj == Py_None) {
        /* UFunc has no identity (should not happen) */
        // 释放对 Py_None 的引用并返回 0
        Py_DECREF(identity_obj);
        return 0;
    }
    // 如果输入类型是无符号整数并且标识对象是 PyLong 对象
    if (PyTypeNum_ISUNSIGNED(context->descriptors[1]->type_num)
            && PyLong_CheckExact(identity_obj)) {
        /*
         * 这是一个小技巧，直到我们有真正的循环特定标识为止。
         * Python 中 -1 不能转换为无符号整数，因此将其转换为 NumPy 标量，
         * 但对于位运算函数，我们使用 -1 表示全部为 1。
         * （内置标识在这里不会溢出，尽管我们可能会不必要地转换 0 和 1。）
         */
        // 将标识对象转换为 NumPy 的数组标量对象
        Py_SETREF(identity_obj, PyObject_CallFunctionObjArgs(
                     (PyObject *)&PyLongArrType_Type, identity_obj, NULL));
        // 如果转换失败，则返回错误代码
        if (identity_obj == NULL) {
            return -1;
        }
    }
    // 如果输入类型是对象类型且归约不是空的
    else if (context->descriptors[0]->type_num == NPY_OBJECT
            && !reduction_is_empty) {
        /* 允许 `sum([object()])` 起作用，但在空时使用 0。 */
        // 释放对标识对象的引用并返回 0
        Py_DECREF(identity_obj);
        return 0;
    }

    // 将初始值和标识对象打包到数组中
    int res = PyArray_Pack(context->descriptors[0], initial, identity_obj);
    // 释放对标识对象的引用
    Py_DECREF(identity_obj);
    // 如果打包失败，则返回错误代码
    if (res < 0) {
        return -1;
    }

    // 如果输入类型是数字，可以缓存以避免通过 Python int 转换
    if (PyTypeNum_ISNUMBER(context->descriptors[0]->type_num)) {
        /* 对于数字，我们可以缓存以避免通过 Python int 转换 */
        // 将初始值的一部分复制到方法的缓存中，并设置获取初始值的方法
        memcpy(context->method->legacy_initial, initial,
               context->descriptors[0]->elsize);
        context->method->get_reduction_initial = &copy_cached_initial;
    }

    // 归约可以使用初始值
    // 返回 1 表示成功
    return 1;
/*
 * 结束函数 `PyArray_NewLegacyWrappingArrayMethod` 的定义。
 */
}


/*
 * 获取封装了 ufunc 实例的未绑定 ArrayMethod。
 * 注意，此函数将结果存储在 ufunc 上，然后只返回相同的结果。
 */
NPY_NO_EXPORT PyArrayMethodObject *
PyArray_NewLegacyWrappingArrayMethod(PyUFuncObject *ufunc,
        PyArray_DTypeMeta *signature[])
{
    // 创建方法名称字符串，格式为 "legacy_ufunc_wrapper_for_{ufunc->name}"，长度限制为 100
    char method_name[101];
    const char *name = ufunc->name ? ufunc->name : "<unknown>";
    snprintf(method_name, 100, "legacy_ufunc_wrapper_for_%s", name);

    /*
     * 假设我们在任何（传统）dtype 标记时需要 Python API。
     */
    int any_output_flexible = 0;
    NPY_ARRAYMETHOD_FLAGS flags = 0;
    if (ufunc->nargs == 3 &&
            signature[0]->type_num == NPY_BOOL &&
            signature[1]->type_num == NPY_BOOL &&
            signature[2]->type_num == NPY_BOOL && (
                strcmp(ufunc->name, "logical_or") == 0 ||
                strcmp(ufunc->name, "logical_and") == 0 ||
                strcmp(ufunc->name, "logical_xor") == 0)) {
        /*
         * 这是一个逻辑 ufunc，并且`??->?`循环。始终可以将任何输入强制转换为布尔值，
         * 因为这种转换由真值定义。
         * 这使得我们可以确保两件事：
         * 1. `np.all`/`np.any` 知道强制转换输入是可以的
         *    （它们必须这样做，因为没有 `?l->?` 等循环）
         * 2. 逻辑函数自动适用于任何实现到布尔值的 DType。
         */
        flags = _NPY_METH_FORCE_CAST_INPUTS;
    }

    // 获取归约初始值的函数指针
    PyArrayMethod_GetReductionInitial *get_reduction_intial = NULL;
    if (ufunc->nin == 2 && ufunc->nout == 1) {
        npy_bool reorderable = NPY_FALSE;
        // 获取默认的标识对象，可能返回 NULL
        PyObject *identity_obj = PyUFunc_GetDefaultIdentity(
                ufunc, &reorderable);
        if (identity_obj == NULL) {
            return NULL;
        }
        /*
         * TODO: 对于对象，"reorderable" 是必需的吗？因为否则我们会禁用多轴归约 `arr.sum(0, 1)`。
         *       但是对于 `arr = array([["a", "b"], ["c", "d"]], dtype="object")`，
         *       它实际上不可重新排序（顺序更改结果）。
         */
        if (reorderable) {
            flags |= NPY_METH_IS_REORDERABLE;
        }
        if (identity_obj != Py_None) {
            get_reduction_intial = &get_initial_from_ufunc;
        }
    }

    // 遍历所有输入和输出的签名，设置相应的标志
    for (int i = 0; i < ufunc->nin+ufunc->nout; i++) {
        if (signature[i]->singleton->flags & (
                NPY_ITEM_REFCOUNT | NPY_ITEM_IS_POINTER | NPY_NEEDS_PYAPI)) {
            flags |= NPY_METH_REQUIRES_PYAPI;
        }
        if (NPY_DT_is_parametric(signature[i])) {
            any_output_flexible = 1;
        }
    }

    // 定义 PyType_Slot 数组，指定不同的功能方法和对应的函数指针
    PyType_Slot slots[4] = {
        {NPY_METH_get_loop, &get_wrapped_legacy_ufunc_loop},
        {NPY_METH_resolve_descriptors, &simple_legacy_resolve_descriptors},
        {NPY_METH_get_reduction_initial, get_reduction_intial},
        {0, NULL},
    };
    // 如果存在任何输出灵活性，执行以下操作
    if (any_output_flexible) {
        // 设置函数指针以使用包装后的传统描述符解析器
        slots[1].pfunc = &wrapped_legacy_resolve_descriptors;
    }

    // 定义 PyArrayMethod_Spec 结构体并初始化
    PyArrayMethod_Spec spec = {
        .name = method_name,           // 方法名称
        .nin = ufunc->nin,             // 输入数量
        .nout = ufunc->nout,           // 输出数量
        .casting = NPY_NO_CASTING,     // 禁用类型转换
        .flags = flags,                // 标志位
        .dtypes = signature,           // 签名数据类型
        .slots = slots,                // 方法结构体数组
    };

    // 通过 PyArrayMethod_FromSpec_int 函数创建 PyBoundArrayMethodObject 对象
    PyBoundArrayMethodObject *bound_res = PyArrayMethod_FromSpec_int(&spec, 1);
    // 如果创建失败，返回空指针
    if (bound_res == NULL) {
        return NULL;
    }

    // 获取 PyArrayMethodObject 对象
    PyArrayMethodObject *res = bound_res->method;
    // 增加对象的引用计数
    Py_INCREF(res);
    // 减少 PyBoundArrayMethodObject 对象的引用计数
    Py_DECREF(bound_res);

    // 返回 PyArrayMethodObject 对象
    return res;
}



# 这是一个代码块的结束符号，表示一个代码块的结尾。
```