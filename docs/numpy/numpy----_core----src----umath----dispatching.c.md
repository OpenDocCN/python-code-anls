# `.\numpy\numpy\_core\src\umath\dispatching.c`

```
/*
 * This file implements universal function dispatching and promotion (which
 * is necessary to happen before dispatching).
 * This is part of the UFunc object.  Promotion and dispatching uses the
 * following things:
 *
 * - operand_DTypes:  The datatypes as passed in by the user.
 * - signature: The DTypes fixed by the user with `dtype=` or `signature=`.
 * - ufunc._loops: A list of all ArrayMethods and promoters, it contains
 *   tuples `(dtypes, ArrayMethod)` or `(dtypes, promoter)`.
 * - ufunc._dispatch_cache: A cache to store previous promotion and/or
 *   dispatching results.
 * - The actual arrays are used to support the old code paths where necessary.
 *   (this includes any value-based casting/promotion logic)
 *
 * In general, `operand_DTypes` is always overridden by `signature`.  If a
 * DType is included in the `signature` it must match precisely.
 *
 * The process of dispatching and promotion can be summarized in the following
 * steps:
 *
 * 1. Override any `operand_DTypes` from `signature`.
 * 2. Check if the new `operand_Dtypes` is cached (if it is, go to 4.)
 * 3. Find the best matching "loop".  This is done using multiple dispatching
 *    on all `operand_DTypes` and loop `dtypes`.  A matching loop must be
 *    one whose DTypes are superclasses of the `operand_DTypes` (that are
 *    defined).  The best matching loop must be better than any other matching
 *    loop.  This result is cached.
 * 4. If the found loop is a promoter: We call the promoter. It can modify
 *    the `operand_DTypes` currently.  Then go back to step 2.
 *    (The promoter can call arbitrary code, so it could even add the matching
 *    loop first.)
 * 5. The final `ArrayMethod` is found, its registered `dtypes` is copied
 *    into the `signature` so that it is available to the ufunc loop.
 *
 */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#define _UMATHMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <convert_datatype.h>

#include "numpy/ndarraytypes.h"
#include "numpy/npy_3kcompat.h"
#include "common.h"
#include "npy_pycompat.h"

#include "dispatching.h"
#include "dtypemeta.h"
#include "npy_hashtable.h"
#include "legacy_array_method.h"
#include "ufunc_object.h"
#include "ufunc_type_resolution.h"


#define PROMOTION_DEBUG_TRACING 0


/* forward declaration */
static inline PyObject *
promote_and_get_info_and_ufuncimpl(PyUFuncObject *ufunc,
        PyArrayObject *const ops[],
        PyArray_DTypeMeta *signature[],
        PyArray_DTypeMeta *op_dtypes[],
        npy_bool allow_legacy_promotion);


/**
 * Function to add a new loop to the ufunc.  This mainly appends it to the
 * list (as it currently is just a list).
 *
 * @param ufunc The universal function to add the loop to.
 * @param info The tuple (dtype_tuple, ArrayMethod/promoter).
 * @param ignore_duplicate If 1 and a loop with the same `dtype_tuple` is
 *        found, the function does nothing.
 */
NPY_NO_EXPORT int
/*
 * PyUFunc_AddLoop: Add a loop implementation to a ufunc object based on provided info.
 * Validates the info object to ensure it meets expected format and content.
 */
PyUFunc_AddLoop(PyUFuncObject *ufunc, PyObject *info, int ignore_duplicate)
{
    /*
     * Validate the info object, this should likely move to a different
     * entry-point in the future (and is mostly unnecessary currently).
     */
    if (!PyTuple_CheckExact(info) || PyTuple_GET_SIZE(info) != 2) {
        PyErr_SetString(PyExc_TypeError,
                "Info must be a tuple: "
                "(tuple of DTypes or None, ArrayMethod or promoter)");
        return -1;
    }

    // Extract the DType tuple from info
    PyObject *DType_tuple = PyTuple_GetItem(info, 0);

    // Check if the length of DType tuple matches the number of operands in ufunc
    if (PyTuple_GET_SIZE(DType_tuple) != ufunc->nargs) {
        PyErr_SetString(PyExc_TypeError,
                "DType tuple length does not match ufunc number of operands");
        return -1;
    }

    // Validate each item in DType tuple
    for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(DType_tuple); i++) {
        PyObject *item = PyTuple_GET_ITEM(DType_tuple, i);
        if (item != Py_None
                && !PyObject_TypeCheck(item, &PyArrayDTypeMeta_Type)) {
            PyErr_SetString(PyExc_TypeError,
                    "DType tuple may only contain None and DType classes");
            return -1;
        }
    }

    // Validate the second argument (meth_or_promoter) in info
    PyObject *meth_or_promoter = PyTuple_GET_ITEM(info, 1);
    if (!PyObject_TypeCheck(meth_or_promoter, &PyArrayMethod_Type)
            && !PyCapsule_IsValid(meth_or_promoter, "numpy._ufunc_promoter")) {
        PyErr_SetString(PyExc_TypeError,
                "Second argument to info must be an ArrayMethod or promoter");
        return -1;
    }

    // Initialize ufunc->_loops if it's NULL
    if (ufunc->_loops == NULL) {
        ufunc->_loops = PyList_New(0);
        if (ufunc->_loops == NULL) {
            return -1;
        }
    }

    // Get the current list of loops registered in ufunc
    PyObject *loops = ufunc->_loops;
    Py_ssize_t length = PyList_Size(loops);

    // Iterate through existing loops to check for duplicates
    for (Py_ssize_t i = 0; i < length; i++) {
        PyObject *item = PyList_GetItemRef(loops, i);
        PyObject *cur_DType_tuple = PyTuple_GetItem(item, 0);
        Py_DECREF(item);

        // Compare current DType tuple with the one being added
        int cmp = PyObject_RichCompareBool(cur_DType_tuple, DType_tuple, Py_EQ);
        if (cmp < 0) {
            return -1;
        }
        if (cmp == 0) {
            continue;
        }
        // If ignore_duplicate is enabled, return success
        if (ignore_duplicate) {
            return 0;
        }
        // Otherwise, raise an error for duplicate registration
        PyErr_Format(PyExc_TypeError,
                "A loop/promoter has already been registered with '%s' for %R",
                ufunc_get_name_cstr(ufunc), DType_tuple);
        return -1;
    }

    // Append the new info (tuple) to the list of loops for ufunc
    if (PyList_Append(loops, info) < 0) {
        return -1;
    }

    // Return success
    return 0;
}
    // 检查传入的对象是否为 ufunc 对象，如果不是则设置类型错误并返回 -1
    if (!PyObject_TypeCheck(ufunc, &PyUFunc_Type)) {
        PyErr_SetString(PyExc_TypeError,
                "ufunc object passed is not a ufunc!");
        return -1;
    }
    // 从给定的规格和私有数据创建一个 PyBoundArrayMethodObject 对象
    PyBoundArrayMethodObject *bmeth =
            (PyBoundArrayMethodObject *)PyArrayMethod_FromSpec_int(spec, priv);
    // 如果创建失败，则返回 -1
    if (bmeth == NULL) {
        return -1;
    }
    // 计算参数的个数，包括输入和输出参数
    int nargs = bmeth->method->nin + bmeth->method->nout;
    // 根据给定的数据类型数组创建一个元组对象
    PyObject *dtypes = PyArray_TupleFromItems(
            nargs, (PyObject **)bmeth->dtypes, 1);
    // 如果创建元组失败，则返回 -1
    if (dtypes == NULL) {
        return -1;
    }
    // 使用 dtypes 和 bmeth->method 创建一个元组对象
    PyObject *info = PyTuple_Pack(2, dtypes, bmeth->method);
    // 减少 bmeth 和 dtypes 的引用计数
    Py_DECREF(bmeth);
    Py_DECREF(dtypes);
    // 如果创建元组失败，则返回 -1
    if (info == NULL) {
        return -1;
    }
    // 调用 PyUFunc_AddLoop 将 info 传递给 ufunc 对象的添加循环方法，并返回结果
    return PyUFunc_AddLoop((PyUFuncObject *)ufunc, info, 0);
/**
 * Resolves the implementation to use, this uses typical multiple dispatching
 * methods of finding the best matching implementation or resolver.
 * (Based on `isinstance()`, the knowledge that non-abstract DTypes cannot
 * be subclassed is used, however.)
 *
 * NOTE: This currently does not take into account output dtypes which do not
 *       have to match.  The possible extension here is that if an output
 *       is given (and thus an output dtype), but not part of the signature
 *       we could ignore it for matching, but *prefer* a loop that matches
 *       better.
 *       Why is this not done currently?  First, it seems a niche feature that
 *       loops can only be distinguished based on the output dtype.  Second,
 *       there are some nasty theoretical things because:
 *
 *            np.add(f4, f4, out=f8)
 *            np.add(f4, f4, out=f8, dtype=f8)
 *
 *       are different, the first uses the f4 loop, the second the f8 loop.
 *       The problem is, that the current cache only uses the op_dtypes and
 *       both are `(f4, f4, f8)`.  The cache would need to store also which
 *       output was provided by `dtype=`/`signature=`.
 *
 * @param ufunc The universal function object representing the function to resolve.
 * @param op_dtypes The array of DTypeMeta objects representing operand types.
 * @param only_promoters Flag indicating whether to consider only promoters.
 * @param out_info Output parameter returning the best implementation information.
 *        WARNING: This is a borrowed reference!
 * @returns -1 on error, 0 on success. Note that the output can be NULL on success if nothing is found.
 */
static int
resolve_implementation_info(PyUFuncObject *ufunc,
        PyArray_DTypeMeta *op_dtypes[], npy_bool only_promoters,
        PyObject **out_info)
{
    int nin = ufunc->nin, nargs = ufunc->nargs;
    Py_ssize_t size = PySequence_Length(ufunc->_loops);
    PyObject *best_dtypes = NULL;
    PyObject *best_resolver_info = NULL;

#if PROMOTION_DEBUG_TRACING
    printf("Promoting for '%s' promoters only: %d\n",
            ufunc->name ? ufunc->name : "<unknown>", (int)only_promoters);
    printf("    DTypes: ");
    PyObject *tmp = PyArray_TupleFromItems(ufunc->nargs, op_dtypes, 1);
    PyObject_Print(tmp, stdout, 0);
    Py_DECREF(tmp);
    printf("\n");
#endif

    // Logic to resolve the best implementation goes here

    if (best_dtypes == NULL) {
        /* The non-legacy lookup failed */
        *out_info = NULL;
        return 0;
    }

    *out_info = best_resolver_info;
    return 0;
}
/*
 * 调用推广器并递归处理
 * 
 * ufunc: 要操作的通用函数对象
 * info: 一个元组，包含有关推广的信息；若为NULL，则表示进行减少特殊路径
 * op_dtypes: 操作数的数据类型元数据数组
 * signature: 签名的数据类型元数据数组
 * operands: 操作数数组
 * 返回值: 推广和解析后的信息及ufunc实现的解析结果，或者在出错时返回NULL
 */
call_promoter_and_recurse(PyUFuncObject *ufunc, PyObject *info,
        PyArray_DTypeMeta *op_dtypes[], PyArray_DTypeMeta *signature[],
        PyArrayObject *const operands[])
{
    int nargs = ufunc->nargs; // 获取ufunc对象的参数个数
    PyObject *resolved_info = NULL; // 初始化解析后的信息为NULL

    int promoter_result; // 推广器的结果
    PyArray_DTypeMeta *new_op_dtypes[NPY_MAXARGS]; // 新的操作数数据类型元数据数组

    if (info != NULL) { // 如果信息不为NULL
        PyObject *promoter = PyTuple_GET_ITEM(info, 1); // 获取元组中第二个元素作为推广器

        if (PyCapsule_CheckExact(promoter)) { // 检查推广器是否为PyCapsule类型
            /* 我们也可以选择包装Python函数... */
            PyArrayMethod_PromoterFunction *promoter_function = PyCapsule_GetPointer(
                    promoter, "numpy._ufunc_promoter"); // 获取推广器函数指针
            if (promoter_function == NULL) {
                return NULL; // 如果无法获取推广器函数指针，返回NULL
            }
            promoter_result = promoter_function((PyObject *)ufunc,
                    op_dtypes, signature, new_op_dtypes); // 调用推广器函数
        }
        else {
            PyErr_SetString(PyExc_NotImplementedError,
                    "Calling python functions for promotion is not implemented.");
            return NULL; // 如果推广器不是PyCapsule类型，返回NULL并设置错误信息
        }
        if (promoter_result < 0) { // 如果推广结果小于0，返回NULL
            return NULL;
        }

        /*
         * 如果没有数据类型发生变化，我们将会无限递归，终止。
         * （当然，仍然可能无限递归。）
         *
         * TODO: 我们可以允许用户直接信号这一点，并且还可以移动
         *       调用以几乎立即进行。这有时会不必要地调用它，
         *       但可能会增加灵活性。
         */
        int dtypes_changed = 0; // 标记数据类型是否发生变化
        for (int i = 0; i < nargs; i++) {
            if (new_op_dtypes[i] != op_dtypes[i]) { // 检查每个操作数的数据类型是否发生变化
                dtypes_changed = 1;
                break;
            }
        }
        if (!dtypes_changed) { // 如果数据类型没有发生变化，跳转到finish标签处
            goto finish;
        }
    }
    else {
        /* 减少特殊路径 */
        new_op_dtypes[0] = NPY_DT_NewRef(op_dtypes[1]); // 复制第二个操作数的数据类型为第一个
        new_op_dtypes[1] = NPY_DT_NewRef(op_dtypes[1]); // 复制第二个操作数的数据类型为第二个
        Py_XINCREF(op_dtypes[2]); // 增加第三个操作数的引用计数
        new_op_dtypes[2] = op_dtypes[2]; // 第三个操作数的数据类型保持不变
    }

    /*
     * 进行递归调用，推广函数必须确保新元组严格更精确
     * （从而保证最终完成）
     */
    if (Py_EnterRecursiveCall(" during ufunc promotion.") != 0) { // 进入递归调用检查
        goto finish; // 如果递归深度超过限制，跳转到finish标签处
    }
    resolved_info = promote_and_get_info_and_ufuncimpl(ufunc,
            operands, signature, new_op_dtypes,
            /* no legacy promotion */ NPY_FALSE); // 执行推广和获取信息及ufunc实现的解析

    Py_LeaveRecursiveCall(); // 离开递归调用

  finish: // 结束标签，释放资源
    for (int i = 0; i < nargs; i++) {
        Py_XDECREF(new_op_dtypes[i]); // 释放新操作数数据类型的引用
    }
    return resolved_info; // 返回解析后的信息
}

/*
 * 将DType 'signature'转换为旧ufunc类型解析器在'ufunc_type_resolution.c'中使用的描述符元组。
 *
 * 注意，当我们使用类型解析的旧路径而不是推广时，我们不需要传递类型元组，
 * 因为在这种情况下签名始终是正确的。
 */
/*
 * 创建一个新的类型元组，根据给定的签名数组和元组指针进行填充。
 * 如果创建过程中发生错误，将返回-1，否则返回0。
 */
static int
_make_new_typetup(
        int nop, PyArray_DTypeMeta *signature[], PyObject **out_typetup) {
    *out_typetup = PyTuple_New(nop);  // 创建一个新的元组，长度为nop，存储在out_typetup中
    if (*out_typetup == NULL) {  // 如果元组创建失败，返回-1
        return -1;
    }

    int none_count = 0;  // 记录为None的元素数量
    for (int i = 0; i < nop; i++) {  // 遍历每个元素
        PyObject *item;
        if (signature[i] == NULL) {  // 如果签名中的元素为空
            item = Py_None;  // 使用Python中的None来填充item
            none_count++;  // 计数加一
        }
        else {
            if (!NPY_DT_is_legacy(signature[i])) {  // 如果不是遗留类型
                /*
                 * 遗留类型解析无法处理这些情况。
                 * 在将来，这条路径将返回`None`或类似的值，
                 * 如果使用了遗留类型解析，则稍后设置错误。
                 */
                PyErr_SetString(PyExc_RuntimeError,
                        "Internal NumPy error: new DType in signature not yet "
                        "supported. (This should be unreachable code!)");
                Py_SETREF(*out_typetup, NULL);  // 将out_typetup设置为NULL
                return -1;  // 返回-1表示错误
            }
            item = (PyObject *)signature[i]->singleton;  // 否则使用签名对应的单例对象填充item
        }
        Py_INCREF(item);  // 增加item的引用计数
        PyTuple_SET_ITEM(*out_typetup, i, item);  // 将item放入元组的第i个位置
    }
    if (none_count == nop) {
        /* 整个签名都是None，简单地忽略类型元组 */
        Py_DECREF(*out_typetup);  // 释放类型元组的引用
        *out_typetup = NULL;  // 将类型元组设置为NULL
    }
    return 0;  // 返回0表示成功
}


/*
 * 使用遗留类型解析器填充操作的数据类型数组，并使用借用引用。这可能会更改内容，
 * 因为它将使用遗留类型解析，可以特殊处理0维数组（使用基于值的逻辑）。
 */
static int
legacy_promote_using_legacy_type_resolver(PyUFuncObject *ufunc,
        PyArrayObject *const *ops, PyArray_DTypeMeta *signature[],
        PyArray_DTypeMeta *operation_DTypes[], int *out_cacheable,
        npy_bool check_only)
{
    int nargs = ufunc->nargs;  // 获取操作数的数量
    PyArray_Descr *out_descrs[NPY_MAXARGS] = {NULL};  // 创建描述符数组，初始化为NULL

    PyObject *type_tuple = NULL;  // 创建一个Python对象指针，初始化为NULL
    if (_make_new_typetup(nargs, signature, &type_tuple) < 0) {  // 创建新的类型元组
        return -1;  // 如果创建失败，直接返回-1
    }

    /*
     * 我们使用不安全的强制转换。当然，这不准确，但在这里没关系，
     * 因为对于提升/分派来说，强制转换的安全性没有影响。
     * 实际操作数是否可以强制转换必须在类型解析步骤中检查（这可能也会调用此函数！）。
     */
    if (ufunc->type_resolver(ufunc,
            NPY_UNSAFE_CASTING, (PyArrayObject **)ops, type_tuple,
            out_descrs) < 0) {
        Py_XDECREF(type_tuple);  // 释放类型元组的引用
        /* 不是所有的遗留解析器在失败时都会清理： */
        for (int i = 0; i < nargs; i++) {
            Py_CLEAR(out_descrs[i]);  // 清理描述符数组中的每个元素
        }
        return -1;  // 返回-1表示失败
    }
    Py_XDECREF(type_tuple);  // 释放类型元组的引用
    if (NPY_UNLIKELY(check_only)) {
        /*
         * 当启用警告时，我们不替换数据类型，而只检查旧结果是否与新结果相同。
         * 由于噪音的原因，我们只在*输出*数据类型上执行此操作，忽略浮点精度变化，
         * 如 `np.float32(3.1) < 3.1` 的比较。
         */
        for (int i = ufunc->nin; i < ufunc->nargs; i++) {
            /*
             * 如果提供了输出并且新的数据类型匹配，则可能会略微损失精度，例如:
             * `np.true_divide(float32_arr0d, 1, out=float32_arr0d)`
             * （在此之前操作的是 float64，尽管这种情况可能很少见）
             */
            if (ops[i] != NULL
                    && PyArray_EquivTypenums(
                            operation_DTypes[i]->type_num,
                            PyArray_DESCR(ops[i])->type_num)) {
                continue;
            }
            /* 否则，如果数据类型不匹配，则发出警告 */
            if (!PyArray_EquivTypenums(
                    operation_DTypes[i]->type_num, out_descrs[i]->type_num)) {
                if (PyErr_WarnFormat(PyExc_UserWarning, 1,
                        "result dtype changed due to the removal of value-based "
                        "promotion from NumPy. Changed from %S to %S.",
                        out_descrs[i], operation_DTypes[i]->singleton) < 0) {
                    return -1;
                }
                return 0;
            }
        }
        return 0;
    }

    for (int i = 0; i < nargs; i++) {
        Py_XSETREF(operation_DTypes[i], NPY_DTYPE(out_descrs[i]));
        Py_INCREF(operation_DTypes[i]);
        Py_DECREF(out_descrs[i]);
    }
    /*
     * 日期时间的传统解析器忽略了签名，这在使用时应该是警告/异常（如果使用）。
     * 在这种情况下，签名被（错误地）修改，缓存是不可能的。
     */
    for (int i = 0; i < nargs; i++) {
        if (signature[i] != NULL && signature[i] != operation_DTypes[i]) {
            Py_INCREF(operation_DTypes[i]);
            Py_SETREF(signature[i], operation_DTypes[i]);
            *out_cacheable = 0;
        }
    }
    return 0;
/*
 * 注意，此函数返回一个对 info 的借用引用，因为它将 info 添加到循环中。
 */
NPY_NO_EXPORT PyObject *
add_and_return_legacy_wrapping_ufunc_loop(PyUFuncObject *ufunc,
        PyArray_DTypeMeta *operation_dtypes[], int ignore_duplicate)
{
    // 创建包含操作数据类型元组的 Python 元组对象
    PyObject *DType_tuple = PyArray_TupleFromItems(ufunc->nargs,
            (PyObject **)operation_dtypes, 0);
    if (DType_tuple == NULL) {
        return NULL;
    }

    // 创建新的遗留包装数组方法对象
    PyArrayMethodObject *method = PyArray_NewLegacyWrappingArrayMethod(
            ufunc, operation_dtypes);
    if (method == NULL) {
        Py_DECREF(DType_tuple);
        return NULL;
    }

    // 打包 DType 元组和方法对象为一个元组，作为 info
    PyObject *info = PyTuple_Pack(2, DType_tuple, method);
    Py_DECREF(DType_tuple);
    Py_DECREF(method);
    if (info == NULL) {
        return NULL;
    }

    // 将 info 添加到 ufunc 的循环列表中
    if (PyUFunc_AddLoop(ufunc, info, ignore_duplicate) < 0) {
        Py_DECREF(info);
        return NULL;
    }
    Py_DECREF(info);  /* 现在从 ufunc 的循环列表中借用引用 */
    return info;
}


/*
 * 这是查找正确的 DType 签名和 ArrayMethod 的主要实现函数，用于 ufunc。此函数可以
 * 在 `do_legacy_fallback` 设置为 False 时递归调用。
 *
 * 如果需要基于值的提升，会提前由 `promote_and_get_ufuncimpl` 处理。
 */
static inline PyObject *
promote_and_get_info_and_ufuncimpl(PyUFuncObject *ufunc,
        PyArrayObject *const ops[],
        PyArray_DTypeMeta *signature[],
        PyArray_DTypeMeta *op_dtypes[],
        npy_bool allow_legacy_promotion)
{
    /*
     * 获取分发信息，其中包含实现和 DType 签名元组。具体步骤如下：
     *
     * 1. 检查缓存。
     * 2. 检查所有注册的循环/提升器，找到最佳匹配。
     * 3. 如果找不到匹配项，则回退到遗留实现。
     */
    PyObject *info = PyArrayIdentityHash_GetItem(ufunc->_dispatch_cache,
                (PyObject **)op_dtypes);
    if (info != NULL && PyObject_TypeCheck(
            PyTuple_GET_ITEM(info, 1), &PyArrayMethod_Type)) {
        /* 找到了 ArrayMethod 并且不是提升器：返回它 */
        return info;
    }

    /*
     * 如果 `info == NULL`，从缓存加载失败，使用 `resolve_implementation_info`
     * 进行完整解析（成功时缓存其结果）。
     */
    if (info == NULL) {
        if (resolve_implementation_info(ufunc,
                op_dtypes, NPY_FALSE, &info) < 0) {
            return NULL;
        }
        if (info != NULL && PyObject_TypeCheck(
                PyTuple_GET_ITEM(info, 1), &PyArrayMethod_Type)) {
            /*
             * 找到了 ArrayMethod 并且不是提升器。在返回之前，将其添加到缓存
             * 中，以便将来更快地查找。
             */
            if (PyArrayIdentityHash_SetItem(ufunc->_dispatch_cache,
                    (PyObject **)op_dtypes, info, 0) < 0) {
                return NULL;
            }
            return info;
        }
    /*
     * 此时，如果没有匹配的循环，则 `info` 为 NULL；如果有匹配的循环，它是一个需要使用/调用的推广器。
     * TODO: 可能需要找到更好的减少解决方案，但这种方式是一个真正的后备（未注册，因此优先级最低）。
     */
    if (info != NULL || op_dtypes[0] == NULL) {
        // 调用推广器并递归处理
        info = call_promoter_and_recurse(ufunc,
                info, op_dtypes, signature, ops);
        if (info == NULL && PyErr_Occurred()) {
            return NULL;
        }
        else if (info != NULL) {
            /* 将结果使用原始类型添加到缓存中： */
            if (PyArrayIdentityHash_SetItem(ufunc->_dispatch_cache,
                    (PyObject **)op_dtypes, info, 0) < 0) {
                return NULL;
            }
            return info;
        }
    }

    /*
     * 即使使用推广器也找不到循环。
     * 推广失败，这通常应该是一个错误。
     * 但是，我们需要在这里给传统实现一个机会（它将修改 `op_dtypes`）。
     */
    if (!allow_legacy_promotion || ufunc->type_resolver == NULL ||
            (ufunc->ntypes == 0 && ufunc->userloops == NULL)) {
        /* 已经尝试过或不是“传统”的 ufunc（未找到循环，返回） */
        return NULL;
    }

    PyArray_DTypeMeta *new_op_dtypes[NPY_MAXARGS] = {NULL};
    int cacheable = 1;  /* TODO: 只有比较过时才需要这个 */
    if (legacy_promote_using_legacy_type_resolver(ufunc,
            ops, signature, new_op_dtypes, &cacheable, NPY_FALSE) < 0) {
        return NULL;
    }
    // 推广并获取信息和 ufunc 实现
    info = promote_and_get_info_and_ufuncimpl(ufunc,
            ops, signature, new_op_dtypes, NPY_FALSE);
    if (info == NULL) {
        /*
         * NOTE: This block exists solely to support numba's DUFuncs which add
         * new loops dynamically, so our list may get outdated.  Thus, we
         * have to make sure that the loop exists.
         *
         * Before adding a new loop, ensure that it actually exists. There
         * is a tiny chance that this would not work, but it would require an
         * extension additionally have a custom loop getter.
         * This check should ensure a the right error message, but in principle
         * we could try to call the loop getter here.
         */
        
        // 获取当前 ufunc 的类型字符串
        const char *types = ufunc->types;
        // 初始化循环是否存在的标志
        npy_bool loop_exists = NPY_FALSE;
        
        // 遍历 ufunc 的所有类型
        for (int i = 0; i < ufunc->ntypes; ++i) {
            loop_exists = NPY_TRUE;  /* 假设循环存在，如果不存在则中断 */
            
            // 检查每个参数类型是否与新操作的类型匹配
            for (int j = 0; j < ufunc->nargs; ++j) {
                if (types[j] != new_op_dtypes[j]->type_num) {
                    loop_exists = NPY_FALSE;
                    break;
                }
            }
            
            // 如果循环存在，则退出循环
            if (loop_exists) {
                break;
            }
            
            // 移动到下一组参数的类型字符串
            types += ufunc->nargs;
        }

        // 如果循环存在，则添加并返回旧的包装 ufunc 循环信息
        if (loop_exists) {
            info = add_and_return_legacy_wrapping_ufunc_loop(
                    ufunc, new_op_dtypes, 0);
        }
    }

    // 释放新操作数据类型的引用计数
    for (int i = 0; i < ufunc->nargs; i++) {
        Py_XDECREF(new_op_dtypes[i]);
    }

    /* 使用原始类型将此项添加到缓存中： */
    // 如果可缓存且添加到缓存失败，则返回 NULL
    if (cacheable && PyArrayIdentityHash_SetItem(ufunc->_dispatch_cache,
            (PyObject **)op_dtypes, info, 0) < 0) {
        return NULL;
    }
    
    // 返回操作信息
    return info;
/**
 * The central entry-point for the promotion and dispatching machinery.
 *
 * It currently may work with the operands (although it would be possible to
 * only work with DType (classes/types).  This is because it has to ensure
 * that legacy (value-based promotion) is used when necessary.
 *
 * NOTE: The machinery here currently ignores output arguments unless
 *       they are part of the signature.  This slightly limits unsafe loop
 *       specializations, which is important for the `ensure_reduce_compatible`
 *       fallback mode.
 *       To fix this, the caching mechanism (and dispatching) can be extended.
 *       When/if that happens, the `ensure_reduce_compatible` could be
 *       deprecated (it should never kick in because promotion kick in first).
 *
 * @param ufunc The ufunc object, used mainly for the fallback.
 * @param ops The array operands (used only for the fallback).
 * @param signature As input, the DType signature fixed explicitly by the user.
 *        The signature is *filled* in with the operation signature we end up
 *        using.
 * @param op_dtypes The operand DTypes (without casting) which are specified
 *        either by the `signature` or by an `operand`.
 *        (outputs and the second input can be NULL for reductions).
 *        NOTE: In some cases, the promotion machinery may currently modify
 *        these including clearing the output.
 * @param force_legacy_promotion If set, we have to use the old type resolution
 *        to implement value-based promotion/casting.
 * @param promoting_pyscalars Indication that some of the initial inputs were
 *        int, float, or complex.  In this case weak-scalar promotion is used
 *        which can lead to a lower result precision even when legacy promotion
 *        does not kick in: `np.int8(1) + 1` is the example.
 *        (Legacy promotion is skipped because `np.int8(1)` is also scalar)
 * @param ensure_reduce_compatible Must be set for reductions, in which case
 *        the found implementation is checked for reduce-like compatibility.
 *        If it is *not* compatible and `signature[2] != NULL`, we assume its
 *        output DType is correct (see NOTE above).
 *        If removed, promotion may require information about whether this
 *        is a reduction, so the more likely case is to always keep fixing this
 *        when necessary, but push down the handling so it can be cached.
 */
NPY_NO_EXPORT PyArrayMethodObject *
promote_and_get_ufuncimpl(PyUFuncObject *ufunc,
        PyArrayObject *const ops[],
        PyArray_DTypeMeta *signature[],
        PyArray_DTypeMeta *op_dtypes[],
        npy_bool force_legacy_promotion,
        npy_bool allow_legacy_promotion,
        npy_bool promoting_pyscalars,
        npy_bool ensure_reduce_compatible)
{
    // 获取 ufunc 的输入数量和参数数量
    int nin = ufunc->nin, nargs = ufunc->nargs;

    /*
     * Get the actual DTypes we operate with by setting op_dtypes[i] from
     * signature[i].
     */
    for (int i = 0; i < nargs; i++) {
        // 如果签名中的操作数不为空
        if (signature[i] != NULL) {
            /*
             * 忽略操作数输入，不能覆盖签名，因为它是固定的（不能被提升！）
             */
            Py_INCREF(signature[i]);  // 增加对签名对象的引用计数
            Py_XSETREF(op_dtypes[i], signature[i]);  // 设置操作数的数据类型为签名中对应的数据类型
            assert(i >= ufunc->nin || !NPY_DT_is_abstract(signature[i]));  // 断言，确保在输入数量之外或者签名不是抽象的
        }
        // 如果签名中的操作数为空，并且索引大于等于输入数量
        else if (i >= nin) {
            /*
             * 如果不在签名中，我们暂时忽略输出，这总是会得到正确的结果
             * （限制注册包含类型转换的专用循环）
             * （见 resolve_implementation_info 中的注释）
             */
            Py_CLEAR(op_dtypes[i]);  // 清除操作数的数据类型引用
        }
    }

    int current_promotion_state = get_npy_promotion_state();

    // 如果强制使用传统提升，并且当前提升状态是 NPY_USE_LEGACY_PROMOTION，并且有注册的类型或者用户自定义循环
    if (force_legacy_promotion
            && current_promotion_state == NPY_USE_LEGACY_PROMOTION
            && (ufunc->ntypes != 0 || ufunc->userloops != NULL)) {
        /*
         * 对于基于值的逻辑，我们必须使用传统提升。
         * 首先一次性调用旧解析器以获取“实际”循环数据类型。
         * 在此（额外的）提升之后，我们甚至可以使用正常的缓存。
         */
        int cacheable = 1;  // 未使用，因为我们修改了原始的 `op_dtypes`
        if (legacy_promote_using_legacy_type_resolver(ufunc,
                ops, signature, op_dtypes, &cacheable, NPY_FALSE) < 0) {
            goto handle_error;
        }
    }

    /* 暂停警告并始终使用“新”路径 */
    set_npy_promotion_state(NPY_USE_WEAK_PROMOTION);
    // 提升并获取信息及 ufuncimpl 对象
    PyObject *info = promote_and_get_info_and_ufuncimpl(ufunc,
            ops, signature, op_dtypes, allow_legacy_promotion);
    set_npy_promotion_state(current_promotion_state);

    // 如果获取信息失败，则处理错误
    if (info == NULL) {
        goto handle_error;
    }

    // 获取方法对象和所有数据类型元组
    PyArrayMethodObject *method = (PyArrayMethodObject *)PyTuple_GET_ITEM(info, 1);
    PyObject *all_dtypes = PyTuple_GET_ITEM(info, 0);

    /* 如果有必要，检查旧结果是否会有不同 */
    if (NPY_UNLIKELY(current_promotion_state == NPY_USE_WEAK_PROMOTION_AND_WARN)
            && (force_legacy_promotion || promoting_pyscalars)
            && npy_give_promotion_warnings()) {
        PyArray_DTypeMeta *check_dtypes[NPY_MAXARGS];
        for (int i = 0; i < nargs; i++) {
            check_dtypes[i] = (PyArray_DTypeMeta *)PyTuple_GET_ITEM(
                    all_dtypes, i);
        }
        /* 在调用传统提升之前，假装那是状态： */
        set_npy_promotion_state(NPY_USE_LEGACY_PROMOTION);
        // 调用传统类型解析器进行提升
        int res = legacy_promote_using_legacy_type_resolver(ufunc,
                ops, signature, check_dtypes, NULL, NPY_TRUE);
        /* 重置提升状态： */
        set_npy_promotion_state(NPY_USE_WEAK_PROMOTION_AND_WARN);
        if (res < 0) {
            goto handle_error;
        }
    }
    /*
     * 在某些情况下（仅逻辑 ufunc），我们找到的循环可能不兼容于 reduce 操作。
     * 因为机制无法区分带输出的 reduce 操作和普通的 ufunc 调用，
     * 所以我们必须假设结果的 DType 是正确的，并为输入强制指定（如果尚未强制）。
     * 注意：这假设所有的循环都是“安全的”，参见本注释中的注意事项。
     * 这一点可以放宽，如果放宽了，则可能需要缓存调用是否用于 reduce 操作。
     */
    if (ensure_reduce_compatible && signature[0] == NULL &&
            PyTuple_GET_ITEM(all_dtypes, 0) != PyTuple_GET_ITEM(all_dtypes, 2)) {
        signature[0] = (PyArray_DTypeMeta *)PyTuple_GET_ITEM(all_dtypes, 2);
        Py_INCREF(signature[0]);
        return promote_and_get_ufuncimpl(ufunc,
                ops, signature, op_dtypes,
                force_legacy_promotion, allow_legacy_promotion,
                promoting_pyscalars, NPY_FALSE);
    }

    // 遍历所有参数，确保签名与传入的 dtype 相匹配
    for (int i = 0; i < nargs; i++) {
        // 如果当前参数的签名为空，则使用传入的 dtype 作为签名
        if (signature[i] == NULL) {
            signature[i] = (PyArray_DTypeMeta *)PyTuple_GET_ITEM(all_dtypes, i);
            Py_INCREF(signature[i]);
        }
        // 否则，如果签名不匹配传入的 dtype，则处理错误
        else if ((PyObject *)signature[i] != PyTuple_GET_ITEM(all_dtypes, i)) {
            /*
             * 如果签名被强制，缓存中可能包含一个通过促进找到的不兼容循环（未强制签名）。
             * 在这种情况下，拒绝它。
             */
            goto handle_error;
        }
    }

    // 处理正常情况，返回方法
    return method;

  handle_error:
    /* 只在此处设置“未找到循环错误” */
    if (!PyErr_Occurred()) {
        raise_no_loop_found_error(ufunc, (PyObject **)op_dtypes);
    }
    /*
     * 否则，如果发生了错误，但如果错误是 DTypePromotionError，
     * 则将其链接，因为 DTypePromotionError 实际上意味着没有可用的循环。
     * （我们通过使用促进未能找到循环。）
     */
    else if (PyErr_ExceptionMatches(npy_static_pydata.DTypePromotionError)) {
        PyObject *err_type = NULL, *err_value = NULL, *err_traceback = NULL;
        PyErr_Fetch(&err_type, &err_value, &err_traceback);
        raise_no_loop_found_error(ufunc, (PyObject **)op_dtypes);
        npy_PyErr_ChainExceptionsCause(err_type, err_value, err_traceback);
    }
    // 返回空指针表示出现错误
    return NULL;
/*
 * Generic promoter used by as a final fallback on ufuncs.  Most operations are
 * homogeneous, so we can try to find the homogeneous dtype on the inputs
 * and use that.
 * We need to special case the reduction case, where op_dtypes[0] == NULL
 * is possible.
 */
NPY_NO_EXPORT int
default_ufunc_promoter(PyObject *ufunc,
        PyArray_DTypeMeta *op_dtypes[], PyArray_DTypeMeta *signature[],
        PyArray_DTypeMeta *new_op_dtypes[])
{
    /* If nin < 2 promotion is a no-op, so it should not be registered */
    PyUFuncObject *ufunc_obj = (PyUFuncObject *)ufunc;
    assert(ufunc_obj->nin > 1);  // Ensure the number of inputs is more than 1

    // Check if op_dtypes[0] is NULL, indicating a reduction operation
    if (op_dtypes[0] == NULL) {
        assert(ufunc_obj->nin == 2 && ufunc_obj->nout == 1);  /* must be reduction */
        
        // Increase reference count and set new_op_dtypes to op_dtypes[1]
        Py_INCREF(op_dtypes[1]);
        new_op_dtypes[0] = op_dtypes[1];
        Py_INCREF(op_dtypes[1]);
        new_op_dtypes[1] = op_dtypes[1];
        Py_INCREF(op_dtypes[1]);
        new_op_dtypes[2] = op_dtypes[1];

        return 0;  // Return success
    }

    PyArray_DTypeMeta *common = NULL;

    /*
     * If a signature is used and homogeneous in its outputs use that
     * (Could/should likely be rather applied to inputs also, although outs
     * only could have some advantage and input dtypes are rarely enforced.)
     */
    for (int i = ufunc_obj->nin; i < ufunc_obj->nargs; i++) {
        if (signature[i] != NULL) {
            // Set 'common' to the first non-NULL signature dtype
            if (common == NULL) {
                Py_INCREF(signature[i]);
                common = signature[i];
            } else if (common != signature[i]) {
                Py_CLEAR(common);  /* Not homogeneous, unset common */
                break;
            }
        }
    }

    /* Otherwise, use the common DType of all input operands */
    if (common == NULL) {
        // Find a common dtype among input operands
        common = PyArray_PromoteDTypeSequence(ufunc_obj->nin, op_dtypes);
        if (common == NULL) {
            if (PyErr_ExceptionMatches(PyExc_TypeError)) {
                PyErr_Clear();  /* Do not propagate normal promotion errors */
            }
            return -1;  // Return error
        }
    }

    // Set new_op_dtypes to 'common' for all input operands
    for (int i = 0; i < ufunc_obj->nargs; i++) {
        PyArray_DTypeMeta *tmp = common;
        if (signature[i]) {
            tmp = signature[i];  /* never replace a fixed one. */
        }
        Py_INCREF(tmp);
        new_op_dtypes[i] = tmp;
    }

    // For output operands beyond inputs, increment reference and set new_op_dtypes
    for (int i = ufunc_obj->nin; i < ufunc_obj->nargs; i++) {
        Py_XINCREF(op_dtypes[i]);
        new_op_dtypes[i] = op_dtypes[i];
    }

    Py_DECREF(common);  // Decrease reference count for 'common'
    return 0;  // Return success
}
/*
 * 仅处理对象类型的通用函数促进器。
 * 这个函数用于设置通用函数的输入和输出数据类型，以便处理对象类型。
 */
object_only_ufunc_promoter(PyObject *ufunc,
        PyArray_DTypeMeta *NPY_UNUSED(op_dtypes[]),
        PyArray_DTypeMeta *signature[],
        PyArray_DTypeMeta *new_op_dtypes[])
{
    // 指向对象类型的数据类型元数据
    PyArray_DTypeMeta *object_DType = &PyArray_ObjectDType;

    // 遍历通用函数的参数
    for (int i = 0; i < ((PyUFuncObject *)ufunc)->nargs; i++) {
        // 如果参数签名为空，则使用对象类型作为新的操作数据类型
        if (signature[i] == NULL) {
            Py_INCREF(object_DType);  // 增加对象类型的引用计数
            new_op_dtypes[i] = object_DType;  // 设置新的操作数据类型为对象类型
        }
    }

    return 0;  // 返回成功状态
}

/*
 * 逻辑通用函数的特殊促进器。逻辑通用函数可以始终使用??->?规则并正确输出结果
 * （只要输出不是对象类型）。
 */
static int
logical_ufunc_promoter(PyObject *NPY_UNUSED(ufunc),
        PyArray_DTypeMeta *op_dtypes[], PyArray_DTypeMeta *signature[],
        PyArray_DTypeMeta *new_op_dtypes[])
{
    /*
     * 如果我们找到任何对象类型的数据类型，我们目前会强制转换为对象类型。
     * 然而，如果输出已经指定且不是对象类型，则没有必要强制转换，
     * 最好是对输入进行安全的类型转换。
     */
    int force_object = 0;

    // 遍历三个可能的参数位置
    for (int i = 0; i < 3; i++) {
        PyArray_DTypeMeta *item;
        if (signature[i] != NULL) {
            item = signature[i];
            Py_INCREF(item);
            // 如果参数签名中的数据类型为对象类型，设置强制为对象类型标志
            if (item->type_num == NPY_OBJECT) {
                force_object = 1;
            }
        }
        else {
            /* 总是覆盖为布尔类型 */
            item = &PyArray_BoolDType;
            Py_INCREF(item);
            // 如果操作数据类型不为空且为对象类型，设置强制为对象类型标志
            if (op_dtypes[i] != NULL && op_dtypes[i]->type_num == NPY_OBJECT) {
                force_object = 1;
            }
        }
        new_op_dtypes[i] = item;  // 设置新的操作数据类型
    }

    // 如果不需要强制为对象类型或者输出数据类型不是对象类型，返回成功状态
    if (!force_object || (op_dtypes[2] != NULL
                          && op_dtypes[2]->type_num != NPY_OBJECT)) {
        return 0;
    }

    /*
     * 实际上，我们仍然必须使用对象类型循环，尽可能将所有内容设置为对象类型
     * （可能不起作用，但试试看）。
     *
     * 注意：更改此处以检查 `op_dtypes[0] == NULL`，停止为 `np.logical_and.reduce(obj_arr)` 返回 `object`，
     * 这也会影响 `np.all` 和 `np.any`！
     */
    for (int i = 0; i < 3; i++) {
        if (signature[i] != NULL) {
            continue;
        }
        // 设置新的操作数据类型为对象数据类型的引用
        Py_SETREF(new_op_dtypes[i], NPY_DT_NewRef(&PyArray_ObjectDType));
    }
    return 0;  // 返回成功状态
}

/*
 * 安装逻辑通用函数的促进器
 */
NPY_NO_EXPORT int
install_logical_ufunc_promoter(PyObject *ufunc)
{
    // 如果输入对象不是通用函数类型，则抛出运行时错误
    if (PyObject_Type(ufunc) != (PyObject *)&PyUFunc_Type) {
        PyErr_SetString(PyExc_RuntimeError,
                "internal numpy array, logical ufunc was not a ufunc?!");
        return -1;  // 返回错误状态
    }
    // 创建一个包含三个 PyArrayDescr_Type 的元组
    PyObject *dtype_tuple = PyTuple_Pack(3,
            &PyArrayDescr_Type, &PyArrayDescr_Type, &PyArrayDescr_Type, NULL);
    if (dtype_tuple == NULL) {
        return -1;  // 返回错误状态
    }
    // 创建逻辑通用函数的促进器对象
    PyObject *promoter = PyCapsule_New(&logical_ufunc_promoter,
            "numpy._ufunc_promoter", NULL);

    /* 代码块结束 */
}
    // 检查 promoter 指针是否为空
    if (promoter == NULL) {
        // 如果为空，释放 dtype_tuple 对象的引用计数并返回错误码 -1
        Py_DECREF(dtype_tuple);
        return -1;
    }

    // 将 dtype_tuple 和 promoter 封装成一个元组对象 info
    PyObject *info = PyTuple_Pack(2, dtype_tuple, promoter);
    // 释放 dtype_tuple 和 promoter 对象的引用计数
    Py_DECREF(dtype_tuple);
    Py_DECREF(promoter);

    // 检查 info 是否创建成功
    if (info == NULL) {
        // 如果创建失败，返回错误码 -1
        return -1;
    }

    // 调用 PyUFunc_AddLoop 将 info 作为参数传递给 ufunc 的 AddLoop 方法
    // 并返回其返回值作为函数的返回值
    return PyUFunc_AddLoop((PyUFuncObject *)ufunc, info, 0);
/*
 * Return the PyArrayMethodObject or PyCapsule that matches a registered
 * tuple of identical dtypes. Return a borrowed ref of the first match.
 */
NPY_NO_EXPORT PyObject *
get_info_no_cast(PyUFuncObject *ufunc, PyArray_DTypeMeta *op_dtype,
                 int ndtypes)
{
    // 创建一个新的元组，用于存放指定数量的数据类型对象
    PyObject *t_dtypes = PyTuple_New(ndtypes);
    if (t_dtypes == NULL) {
        return NULL;
    }
    // 将 op_dtype 数据类型对象复制到 t_dtypes 元组中的每一个位置
    for (int i=0; i < ndtypes; i++) {
        PyTuple_SetItem(t_dtypes, i, (PyObject *)op_dtype);
    }
    // 获取 ufunc 对象的 _loops 属性，该属性应该是一个列表对象
    PyObject *loops = ufunc->_loops;
    // 获取列表 loops 的长度
    Py_ssize_t length = PyList_Size(loops);
    // 遍历 loops 列表中的每一个元素
    for (Py_ssize_t i = 0; i < length; i++) {
        // 获取列表中第 i 个元素的引用
        PyObject *item = PyList_GetItemRef(loops, i);
        // 获取 item 元素中的第一个元组，即 DType 元组
        PyObject *cur_DType_tuple = PyTuple_GetItem(item, 0);
        // 减少 item 的引用计数，这里应该是为了安全起见，避免内存泄漏
        Py_DECREF(item);
        // 比较 cur_DType_tuple 和 t_dtypes 是否相等
        int cmp = PyObject_RichCompareBool(cur_DType_tuple,
                                           t_dtypes, Py_EQ);
        // 如果比较出错，释放 t_dtypes，并返回空指针
        if (cmp < 0) {
            Py_DECREF(t_dtypes);
            return NULL;
        }
        // 如果 cur_DType_tuple 和 t_dtypes 不相等，继续下一个循环
        if (cmp == 0) {
            continue;
        }
        /* 找到匹配项 */
        // 释放 t_dtypes，并返回匹配项元组的第二个元素
        Py_DECREF(t_dtypes);
        return PyTuple_GetItem(item, 1);
    }
    // 释放 t_dtypes，并返回 None 对象
    Py_DECREF(t_dtypes);
    Py_RETURN_NONE;
}

/* UFUNC_API
 *     Register a new promoter for a ufunc.  A promoter is a function stored
 *     in a PyCapsule (see in-line comments).  It is passed the operation and
 *     requested DType signatures and can mutate it to attempt a new search
 *     for a matching loop/promoter.
 *
 * @param ufunc The ufunc object to register the promoter with.
 * @param DType_tuple A Python tuple containing DTypes or None matching the
 *        number of inputs and outputs of the ufunc.
 * @param promoter A PyCapsule with name "numpy._ufunc_promoter" containing
 *        a pointer to a `PyArrayMethod_PromoterFunction`.
 */
NPY_NO_EXPORT int
PyUFunc_AddPromoter(
        PyObject *ufunc, PyObject *DType_tuple, PyObject *promoter)
{
    // 检查 ufunc 是否为 PyUFuncObject 类型的对象
    if (!PyObject_TypeCheck(ufunc, &PyUFunc_Type)) {
        PyErr_SetString(PyExc_TypeError,
                "ufunc object passed is not a ufunc!");
        return -1;
    }
    // 检查 promoter 是否为 PyCapsule 类型的对象
    if (!PyCapsule_CheckExact(promoter)) {
        PyErr_SetString(PyExc_TypeError,
                "promoter must (currently) be a PyCapsule.");
        return -1;
    }
    // 检查 promoter 是否包含 "numpy._ufunc_promoter" 的名称
    if (PyCapsule_GetPointer(promoter, "numpy._ufunc_promoter") == NULL) {
        return -1;
    }
    // 创建一个包含 DType_tuple 和 promoter 的元组 info
    PyObject *info = PyTuple_Pack(2, DType_tuple, promoter);
    if (info == NULL) {
        return -1;
    }
    // 将 info 添加到 ufunc 对象的循环列表中，返回添加的结果
    return PyUFunc_AddLoop((PyUFuncObject *)ufunc, info, 0);
}
```