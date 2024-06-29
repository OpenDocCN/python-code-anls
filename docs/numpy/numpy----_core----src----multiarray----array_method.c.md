# `.\numpy\numpy\_core\src\multiarray\array_method.c`

```py
/*
 * This file implements an abstraction layer for "Array methods", which
 * work with a specific DType class input and provide low-level C function
 * pointers to do fast operations on the given input functions.
 * It thus adds an abstraction layer around individual ufunc loops.
 *
 * Unlike methods, a ArrayMethod can have multiple inputs and outputs.
 * This has some serious implication for garbage collection, and as far
 * as I (@seberg) understands, it is not possible to always guarantee correct
 * cyclic garbage collection of dynamically created DTypes with methods.
 * The keyword (or rather the solution) for this seems to be an "ephemeron"
 * which I believe should allow correct garbage collection but seems
 * not implemented in Python at this time.
 * The vast majority of use-cases will not require correct garbage collection.
 * Some use cases may require the user to be careful.
 *
 * Generally there are two main ways to solve this issue:
 *
 * 1. A method with a single input (or inputs of all the same DTypes) can
 *    be "owned" by that DType (it becomes unusable when the DType is deleted).
 *    This holds especially for all casts, which must have a defined output
 *    DType and must hold on to it strongly.
 * 2. A method which can infer the output DType(s) from the input types does
 *    not need to keep the output type alive. (It can use NULL for the type,
 *    or an abstract base class which is known to be persistent.)
 *    It is then sufficient for a ufunc (or other owner) to only hold a
 *    weak reference to the input DTypes.
 */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _UMATHMODULE
#define _MULTIARRAYMODULE

#include <npy_pycompat.h>
#include "arrayobject.h"
#include "array_coercion.h"
#include "array_method.h"
#include "dtypemeta.h"
#include "convert_datatype.h"
#include "common.h"
#include "numpy/ufuncobject.h"


/*
 * The default descriptor resolution function.  The logic is as follows:
 *
 * 1. The output is ensured to be canonical (currently native byte order),
 *    if it is of the correct DType.
 * 2. If any DType is was not defined, it is replaced by the common DType
 *    of all inputs. (If that common DType is parametric, this is an error.)
 *
 * We could allow setting the output descriptors specifically to simplify
 * this step.
 *
 * Note that the default version will indicate that the cast can be done
 * as using `arr.view(new_dtype)` if the default cast-safety is
 * set to "no-cast".  This default function cannot be used if a view may
 * be sufficient for casting but the cast is not always "no-cast".
 */
static NPY_CASTING
default_resolve_descriptors(
        PyArrayMethodObject *method,
        PyArray_DTypeMeta *const *dtypes,
        PyArray_Descr *const *input_descrs,
        PyArray_Descr **output_descrs,
        npy_intp *view_offset)
{
    int nin = method->nin;     // Number of input arguments for the method
    int nout = method->nout;   // Number of output arguments for the method
    // 遍历输入输出数组的所有元素
    for (int i = 0; i < nin + nout; i++) {
        // 获取当前元素对应的数据类型
        PyArray_DTypeMeta *dtype = dtypes[i];
        // 如果输入描述符不为空，将输出描述符设为规范化后的输入描述符
        if (input_descrs[i] != NULL) {
            output_descrs[i] = NPY_DT_CALL_ensure_canonical(input_descrs[i]);
        }
        // 否则，使用默认数据类型描述符
        else {
            output_descrs[i] = NPY_DT_CALL_default_descr(dtype);
        }
        // 如果输出描述符为空，跳转到失败处理标签
        if (NPY_UNLIKELY(output_descrs[i] == NULL)) {
            goto fail;
        }
    }
    /*
     * 如果方法的类型转换设置为 NPY_NO_CASTING，
     * 则假设所有的类型转换都不需要，视为可视的情况下。
     */
    if (method->casting == NPY_NO_CASTING) {
        /*
         * 根据当前定义，无类型转换应该意味着可视化。
         * 例如，对象到对象的转换就会标记为可视化。
         */
        *view_offset = 0;
    }
    // 返回当前方法的类型转换设置
    return method->casting;

  fail:
    // 失败时释放已分配的输出描述符内存
    for (int i = 0; i < nin + nout; i++) {
        Py_XDECREF(output_descrs[i]);
    }
    // 返回失败状态
    return -1;
/**
 * Check if the given strides match the element sizes of the descriptors,
 * indicating contiguous memory layout for each argument.
 *
 * @param strides An array of strides for each argument.
 * @param descriptors Array of descriptors for each argument.
 * @param nargs Number of arguments (descriptors and strides).
 * @return 1 if all strides match descriptors' element sizes, otherwise 0.
 */
static inline int
is_contiguous(
        npy_intp const *strides, PyArray_Descr *const *descriptors, int nargs)
{
    // Iterate through each argument
    for (int i = 0; i < nargs; i++) {
        // Check if the stride of the argument matches its descriptor's element size
        if (strides[i] != descriptors[i]->elsize) {
            return 0; // Not contiguous if stride doesn't match element size
        }
    }
    return 1; // All strides match descriptors' element sizes, indicating contiguous memory
}


/**
 * The default method to fetch the correct loop for a cast or ufunc
 * (at the time of writing only casts).
 * Note that the default function provided here will only indicate that a cast
 * can be done as a view (i.e., arr.view(new_dtype)) when this is trivially
 * true, i.e., for cast safety "no-cast". It will not recognize view as an
 * option for other casts (e.g., viewing '>i8' as '>i4' with an offset of 4).
 *
 * @param context The context object containing method descriptors and flags.
 * @param aligned Flag indicating if memory should be aligned.
 * @param move_references UNUSED (currently not used in the function).
 * @param strides An array of strides for each argument.
 * @param descriptors Array of descriptors for each argument.
 * @param out_loop Pointer to store the selected strided loop.
 * @param out_transferdata Unused in this function, set to NULL.
 * @param flags Pointer to store method flags.
 * @return 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
npy_default_get_strided_loop(
        PyArrayMethod_Context *context,
        int aligned, int NPY_UNUSED(move_references), const npy_intp *strides,
        PyArrayMethod_StridedLoop **out_loop, NpyAuxData **out_transferdata,
        NPY_ARRAYMETHOD_FLAGS *flags)
{
    PyArray_Descr *const *descrs = context->descriptors; // Array of descriptors
    PyArrayMethodObject *meth = context->method; // Method object from context
    *flags = meth->flags & NPY_METH_RUNTIME_FLAGS; // Set method flags

    int nargs = meth->nin + meth->nout; // Total number of arguments

    // Determine which loop to select based on alignment
    if (aligned) {
        // Use contiguous loop if available and if arguments are contiguous
        if (meth->contiguous_loop == NULL ||
                !is_contiguous(strides, descrs, nargs)) {
            *out_loop = meth->strided_loop;
            return 0;
        }
        *out_loop = meth->contiguous_loop;
    }
    else {
        // Use unaligned strided loop if available and if arguments are contiguous
        if (meth->unaligned_contiguous_loop == NULL ||
                !is_contiguous(strides, descrs, nargs)) {
            *out_loop = meth->unaligned_strided_loop;
            return 0;
        }
        *out_loop = meth->unaligned_contiguous_loop;
    }
    return 0; // Success
}


/**
 * Validate that the input specification is usable to create a new ArrayMethod.
 *
 * @param spec The specification object to validate.
 * @return 0 on success, -1 on error.
 */
static int
validate_spec(PyArrayMethod_Spec *spec)
{
    int nargs = spec->nin + spec->nout; // Total number of arguments

    // Check for invalid input specification fields/values
    if (spec->nin < 0 || spec->nout < 0 || nargs > NPY_MAXARGS) {
        PyErr_Format(PyExc_ValueError,
                "ArrayMethod inputs and outputs must be greater zero and"
                "not exceed %d. (method: %s)", NPY_MAXARGS, spec->name);
        return -1; // Return error if inputs/outputs are invalid
    }

    // Check for valid casting options
    switch (spec->casting) {
        case NPY_NO_CASTING:
        case NPY_EQUIV_CASTING:
        case NPY_SAFE_CASTING:
        case NPY_SAME_KIND_CASTING:
        case NPY_UNSAFE_CASTING:
            break; // Valid casting types
        default:
            if (spec->casting != -1) {
                PyErr_Format(PyExc_TypeError,
                        "ArrayMethod has invalid casting `%d`. (method: %s)",
                        spec->casting, spec->name);
                return -1; // Return error for invalid casting type
            }
    }

    // Input specification is valid
    return 0;
}
    for (int i = 0; i < nargs; i++) {
        /*
         * 注意，我们可以允许输出数据类型未指定
         * （数组方法必须确保支持此功能）。
         * 我们甚至可以允许某些数据类型是抽象的。
         * 目前，假设这最好在提升步骤中处理。
         * 提供所有数据类型的一个问题是明确需要保持引用。
         * 我们可能最终需要实现遍历并信任 GC 处理它。
         */

        // 检查是否存在未指定的输出数据类型
        if (spec->dtypes[i] == NULL) {
            // 报错：ArrayMethod 必须提供所有输入和输出的数据类型。
            PyErr_Format(PyExc_TypeError,
                    "ArrayMethod must provide all input and output DTypes. "
                    "(method: %s)", spec->name);
            return -1;
        }

        // 检查提供的数据类型对象是否为 DType 类型
        if (!PyObject_TypeCheck(spec->dtypes[i], &PyArrayDTypeMeta_Type)) {
            // 报错：提供的对象不是 DType 类型。
            PyErr_Format(PyExc_TypeError,
                    "ArrayMethod provided object %R is not a DType."
                    "(method: %s)", spec->dtypes[i], spec->name);
            return -1;
        }
    }
    // 所有检查通过，返回成功状态
    return 0;
/**
 * Initialize a new BoundArrayMethodObject from slots.  Slots which are
 * not provided may be filled with defaults.
 *
 * @param res The new PyBoundArrayMethodObject to be filled.
 * @param spec The specification list passed by the user.
 * @param private Private flag to limit certain slots to use in NumPy.
 * @return -1 on error 0 on success
 */
static int
fill_arraymethod_from_slots(
        PyBoundArrayMethodObject *res, PyArrayMethod_Spec *spec,
        int private)
{
    // 获取方法对象
    PyArrayMethodObject *meth = res->method;

    /* Set the defaults */
    // 设置默认的 strided loop 获取函数为 npy_default_get_strided_loop
    meth->get_strided_loop = &npy_default_get_strided_loop;
    // 设置默认的解析描述符函数为 default_resolve_descriptors
    meth->resolve_descriptors = &default_resolve_descriptors;
    // 默认没有初始值或者标识
    meth->get_reduction_initial = NULL;  /* no initial/identity by default */

    /* Fill in the slots passed by the user */
    /*
     * TODO: This is reasonable for now, but it would be nice to find a
     *       shorter solution, and add some additional error checking (e.g.
     *       the same slot used twice). Python uses an array of slot offsets.
     */
    # 遍历给定的PyType_Spec结构体的slots数组，每个元素是一个PyType_Slot结构体
    for (PyType_Slot *slot = &spec->slots[0]; slot->slot != 0; slot++) {
        # 根据slot的值进行不同的操作
        switch (slot->slot) {
            case _NPY_METH_resolve_descriptors_with_scalars:
                # 如果不是私有方法，设置运行时错误并返回-1
                if (!private) {
                    PyErr_SetString(PyExc_RuntimeError,
                            "the _NPY_METH_resolve_descriptors_with_scalars "
                            "slot private due to uncertainty about the best "
                            "signature (see gh-24915)");
                    return -1;
                }
                # 设置resolve_descriptors_with_scalars方法为slot中的pfunc函数指针
                meth->resolve_descriptors_with_scalars = slot->pfunc;
                continue;
            case NPY_METH_resolve_descriptors:
                # 设置resolve_descriptors方法为slot中的pfunc函数指针
                meth->resolve_descriptors = slot->pfunc;
                continue;
            case NPY_METH_get_loop:
                """
                 * 注意：公共API中的get_loop被认为是“不稳定的”，
                 *       我不喜欢它的签名，并且不应使用move_references参数。
                 *       （也就是说：我们不应该担心立即更改它，当然这并不会立即中断它。）
                 """
                # 设置get_strided_loop方法为slot中的pfunc函数指针
                meth->get_strided_loop = slot->pfunc;
                continue;
            # "典型"循环，由默认的`get_loop`支持
            case NPY_METH_strided_loop:
                # 设置strided_loop方法为slot中的pfunc函数指针
                meth->strided_loop = slot->pfunc;
                continue;
            case NPY_METH_contiguous_loop:
                # 设置contiguous_loop方法为slot中的pfunc函数指针
                meth->contiguous_loop = slot->pfunc;
                continue;
            case NPY_METH_unaligned_strided_loop:
                # 设置unaligned_strided_loop方法为slot中的pfunc函数指针
                meth->unaligned_strided_loop = slot->pfunc;
                continue;
            case NPY_METH_unaligned_contiguous_loop:
                # 设置unaligned_contiguous_loop方法为slot中的pfunc函数指针
                meth->unaligned_contiguous_loop = slot->pfunc;
                continue;
            case NPY_METH_get_reduction_initial:
                # 设置get_reduction_initial方法为slot中的pfunc函数指针
                meth->get_reduction_initial = slot->pfunc;
                continue;
            case NPY_METH_contiguous_indexed_loop:
                # 设置contiguous_indexed_loop方法为slot中的pfunc函数指针
                meth->contiguous_indexed_loop = slot->pfunc;
                continue;
            case _NPY_METH_static_data:
                # 设置static_data方法为slot中的pfunc函数指针
                meth->static_data = slot->pfunc;
                continue;
            default:
                break;
        }
        # 如果遇到未知的slot编号，抛出运行时错误并返回-1
        PyErr_Format(PyExc_RuntimeError,
                "invalid slot number %d to ArrayMethod: %s",
                slot->slot, spec->name);
        return -1;
    }

    # 检查slots是否有效
    // 检查是否使用默认的解析描述符函数
    if (meth->resolve_descriptors == &default_resolve_descriptors) {
        // 如果设置了casting为-1但未提供默认的`resolve_descriptors`函数，则抛出TypeError异常
        if (spec->casting == -1) {
            PyErr_Format(PyExc_TypeError,
                    "Cannot set casting to -1 (invalid) when not providing "
                    "the default `resolve_descriptors` function. "
                    "(method: %s)", spec->name);
            return -1;
        }
        // 遍历输入和输出的数据类型列表
        for (int i = 0; i < meth->nin + meth->nout; i++) {
            // 如果某个数据类型为NULL
            if (res->dtypes[i] == NULL) {
                // 如果是输入数据类型并且使用了默认的`resolve_descriptors`函数，则抛出TypeError异常
                if (i < meth->nin) {
                    PyErr_Format(PyExc_TypeError,
                            "All input DTypes must be specified when using "
                            "the default `resolve_descriptors` function. "
                            "(method: %s)", spec->name);
                    return -1;
                }
                // 如果没有输入数据并且未指定输出数据类型或使用自定义的`resolve_descriptors`函数，则抛出TypeError异常
                else if (meth->nin == 0) {
                    PyErr_Format(PyExc_TypeError,
                            "Must specify output DTypes or use custom "
                            "`resolve_descriptors` when there are no inputs. "
                            "(method: %s)", spec->name);
                    return -1;
                }
            }
            // 如果是输出数据类型且为参数化数据类型，则必须提供`resolve_descriptors`函数，否则抛出TypeError异常
            if (i >= meth->nin && NPY_DT_is_parametric(res->dtypes[i])) {
                PyErr_Format(PyExc_TypeError,
                        "must provide a `resolve_descriptors` function if any "
                        "output DType is parametric. (method: %s)",
                        spec->name);
                return -1;
            }
        }
    }
    // 如果不是使用默认的获取步进循环函数，则返回0
    if (meth->get_strided_loop != &npy_default_get_strided_loop) {
        /* Do not check the actual loop fields. */
        return 0;
    }

    /* Check whether the provided loops make sense. */

    // 如果方法标志包含NPY_METH_SUPPORTS_UNALIGNED
    if (meth->flags & NPY_METH_SUPPORTS_UNALIGNED) {
        // 如果未提供非对齐步进内部循环，则抛出TypeError异常
        if (meth->unaligned_strided_loop == NULL) {
            PyErr_Format(PyExc_TypeError,
                    "Must provide unaligned strided inner loop when using "
                    "NPY_METH_SUPPORTS_UNALIGNED flag (in method: %s)",
                    spec->name);
            return -1;
        }
    }
    // 如果方法标志不包含NPY_METH_SUPPORTS_UNALIGNED
    else {
        // 如果提供了非对齐步进内部循环，则抛出TypeError异常
        if (meth->unaligned_strided_loop != NULL) {
            PyErr_Format(PyExc_TypeError,
                    "Must not provide unaligned strided inner loop when not "
                    "using NPY_METH_SUPPORTS_UNALIGNED flag (in method: %s)",
                    spec->name);
            return -1;
        }
    }
    /* Fill in the blanks: */
    
    // 如果未提供非对齐连续循环，则使用非对齐步进内部循环
    if (meth->unaligned_contiguous_loop == NULL) {
        meth->unaligned_contiguous_loop = meth->unaligned_strided_loop;
    }
    // 如果未提供步进循环，则使用非对齐步进内部循环
    if (meth->strided_loop == NULL) {
        meth->strided_loop = meth->unaligned_strided_loop;
    }
    // 如果未提供连续循环，则使用步进循环
    if (meth->contiguous_loop == NULL) {
        meth->contiguous_loop = meth->strided_loop;
    }
    # 检查指针变量 `meth` 的 `strided_loop` 成员是否为 NULL
    if (meth->strided_loop == NULL) {
        # 如果为 NULL，则抛出带有特定格式化错误消息的异常
        PyErr_Format(PyExc_TypeError,
                "Must provide a strided inner loop function. (method: %s)",
                spec->name);
        # 返回 -1 表示函数执行失败
        return -1;
    }

    # 如果 `strided_loop` 不为 NULL，表示条件通过，返回 0 表示函数执行成功
    return 0;
/*
 * Public version of `PyArrayMethod_FromSpec_int` (see below).
 *
 * TODO: Error paths will probably need to be improved before a release into
 *       the non-experimental public API.
 */
NPY_NO_EXPORT PyObject *
PyArrayMethod_FromSpec(PyArrayMethod_Spec *spec)
{
    // 遍历输入输出类型列表，确保每个类型都是 PyArrayDTypeMeta_Type 的实例
    for (int i = 0; i < spec->nin + spec->nout; i++) {
        if (!PyObject_TypeCheck(spec->dtypes[i], &PyArrayDTypeMeta_Type)) {
            PyErr_SetString(PyExc_RuntimeError,
                    "ArrayMethod spec contained a non DType.");
            return NULL;
        }
    }
    // 调用内部函数 PyArrayMethod_FromSpec_int 处理具体逻辑，返回处理结果
    return (PyObject *)PyArrayMethod_FromSpec_int(spec, 0);
}


/**
 * Create a new ArrayMethod (internal version).
 *
 * @param spec A filled context object to pass generic information about
 *        the method (such as usually needing the API, and the DTypes).
 *        Unused fields must be NULL.
 * @param private Some slots are currently considered private, if not true,
 *        these will be rejected.
 *
 * @returns A new (bound) ArrayMethod object.
 */
NPY_NO_EXPORT PyBoundArrayMethodObject *
PyArrayMethod_FromSpec_int(PyArrayMethod_Spec *spec, int private)
{
    int nargs = spec->nin + spec->nout;

    // 如果方法名为 NULL，则设置为 "<unknown>"
    if (spec->name == NULL) {
        spec->name = "<unknown>";
    }

    // 验证 spec 结构是否有效，若无效则返回 NULL
    if (validate_spec(spec) < 0) {
        return NULL;
    }

    PyBoundArrayMethodObject *res;
    // 为返回的 PyBoundArrayMethodObject 分配内存空间
    res = PyObject_New(PyBoundArrayMethodObject, &PyBoundArrayMethod_Type);
    if (res == NULL) {
        return NULL;
    }
    res->method = NULL;

    // 为 dtypes 数组分配内存空间，并复制 spec 中的 dtypes
    res->dtypes = PyMem_Malloc(sizeof(PyArray_DTypeMeta *) * nargs);
    if (res->dtypes == NULL) {
        Py_DECREF(res);
        PyErr_NoMemory();
        return NULL;
    }
    for (int i = 0; i < nargs ; i++) {
        Py_XINCREF(spec->dtypes[i]);
        res->dtypes[i] = spec->dtypes[i];
    }

    // 为 res->method 分配内存空间，并初始化其字段
    res->method = PyObject_New(PyArrayMethodObject, &PyArrayMethod_Type);
    if (res->method == NULL) {
        Py_DECREF(res);
        PyErr_NoMemory();
        return NULL;
    }
    memset((char *)(res->method) + sizeof(PyObject), 0,
           sizeof(PyArrayMethodObject) - sizeof(PyObject));

    // 将 spec 中的信息填充到 res->method 中
    res->method->nin = spec->nin;
    res->method->nout = spec->nout;
    res->method->flags = spec->flags;
    res->method->casting = spec->casting;

    // 使用填充函数填充 res 中的 slots 信息
    if (fill_arraymethod_from_slots(res, spec, private) < 0) {
        Py_DECREF(res);
        return NULL;
    }

    // 为方法名字符串分配内存空间，并复制 spec->name
    Py_ssize_t length = strlen(spec->name);
    res->method->name = PyMem_Malloc(length + 1);
    if (res->method->name == NULL) {
        Py_DECREF(res);
        PyErr_NoMemory();
        return NULL;
    }
    strcpy(res->method->name, spec->name);

    return res;
}


static void
arraymethod_dealloc(PyObject *self)
{
    // 释放 PyArrayMethodObject 对象中的 name 字段内存
    PyArrayMethodObject *meth;
    meth = ((PyArrayMethodObject *)self);
    PyMem_Free(meth->name);
}
    # 检查包装方法是否存在，如果存在则进行清理操作（该方法在 umath 中定义）
    if (meth->wrapped_meth != NULL) {
        # 减少包装方法的引用计数，准备释放其内存
        Py_DECREF(meth->wrapped_meth);
        # 遍历包装的数据类型数组，释放每个元素的内存
        for (int i = 0; i < meth->nin + meth->nout; i++) {
            Py_XDECREF(meth->wrapped_dtypes[i]);
        }
        # 释放包装的数据类型数组的内存
        PyMem_Free(meth->wrapped_dtypes);
    }

    # 释放 self 对象所属类型的内存
    Py_TYPE(self)->tp_free(self);
/*
 * 定义了一个名为 PyArrayMethod_Type 的 PyTypeObject 结构体，表示了一个 NumPy 中的数组方法类型。
 * 这个类型结构体包含了类型的基本信息和方法，如名称、基本大小、析构函数、标志等。
 */
NPY_NO_EXPORT PyTypeObject PyArrayMethod_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "numpy._ArrayMethod",  // 类型名称为 "numpy._ArrayMethod"
    .tp_basicsize = sizeof(PyArrayMethodObject),  // 基本大小为 PyArrayMethodObject 结构体的大小
    .tp_dealloc = arraymethod_dealloc,  // 析构函数为 arraymethod_dealloc 函数
    .tp_flags = Py_TPFLAGS_DEFAULT,  // 默认的类型标志
};


/*
 * 定义了一个名为 boundarraymethod_repr 的静态函数，用于返回绑定数组方法对象的字符串表示形式。
 * 函数根据绑定方法对象的 dtypes 属性构造一个表示形式，并返回一个新的 Python 字符串对象。
 */
static PyObject *
boundarraymethod_repr(PyBoundArrayMethodObject *self)
{
    int nargs = self->method->nin + self->method->nout;
    PyObject *dtypes = PyArray_TupleFromItems(
            nargs, (PyObject **)self->dtypes, 0);
    if (dtypes == NULL) {
        return NULL;  // 如果创建元组失败，返回 NULL
    }
    PyObject *repr = PyUnicode_FromFormat(
                        "<np._BoundArrayMethod `%s` for dtypes %S>",
                        self->method->name, dtypes);
    Py_DECREF(dtypes);  // 减少 dtypes 对象的引用计数
    return repr;  // 返回表示形式的字符串对象
}


/*
 * 定义了一个名为 boundarraymethod_dealloc 的静态函数，用于释放绑定数组方法对象所占用的内存。
 * 函数释放 dtypes 数组和 method 对象，并最终释放对象自身所占用的内存。
 */
static void
boundarraymethod_dealloc(PyObject *self)
{
    PyBoundArrayMethodObject *meth;
    meth = ((PyBoundArrayMethodObject *)self);
    int nargs = meth->method->nin + meth->method->nout;

    for (int i = 0; i < nargs; i++) {
        Py_XDECREF(meth->dtypes[i]);  // 释放每个 dtypes 元素对象的引用
    }
    PyMem_Free(meth->dtypes);  // 释放 dtypes 数组的内存

    Py_XDECREF(meth->method);  // 释放 method 对象的引用

    Py_TYPE(self)->tp_free(self);  // 释放对象自身所占用的内存
}


/*
 * 定义了一个名为 boundarraymethod__resolve_descriptors 的静态函数，用于解析描述符并返回相关信息。
 * 函数验证传入的描述符元组是否符合预期格式，并根据验证结果返回相应的结果或错误。
 */
static PyObject *
boundarraymethod__resolve_descripors(
        PyBoundArrayMethodObject *self, PyObject *descr_tuple)
{
    int nin = self->method->nin;
    int nout = self->method->nout;

    PyArray_Descr *given_descrs[NPY_MAXARGS];
    PyArray_Descr *loop_descrs[NPY_MAXARGS];

    if (!PyTuple_CheckExact(descr_tuple) ||
            PyTuple_Size(descr_tuple) != nin + nout) {
        PyErr_Format(PyExc_TypeError,
                "_resolve_descriptors() takes exactly one tuple with as many "
                "elements as the method takes arguments (%d+%d).", nin, nout);
        return NULL;  // 如果描述符元组格式不符合预期，返回错误并 NULL
    }
    for (int i = 0; i < nin + nout; i++) {
        // 获取描述符元组中第 i 个元素
        PyObject *tmp = PyTuple_GetItem(descr_tuple, i);
        // 如果获取失败，返回空指针异常
        if (tmp == NULL) {
            return NULL;
        }
        // 如果获取的元素是 Py_None
        else if (tmp == Py_None) {
            // 如果 i 小于输入个数 nin，设置类型错误并返回空指针异常
            if (i < nin) {
                PyErr_SetString(PyExc_TypeError,
                        "only output dtypes may be omitted (set to None).");
                return NULL;
            }
            // 将给定描述符数组中第 i 个位置设置为 NULL
            given_descrs[i] = NULL;
        }
        // 如果获取的元素是 PyArray_Descr 类型
        else if (PyArray_DescrCheck(tmp)) {
            // 如果获取的类型不是与 self->dtypes[i] 完全相同的类型，设置类型错误并返回空指针异常
            if (Py_TYPE(tmp) != (PyTypeObject *)self->dtypes[i]) {
                PyErr_Format(PyExc_TypeError,
                        "input dtype %S was not an exact instance of the bound "
                        "DType class %S.", tmp, self->dtypes[i]);
                return NULL;
            }
            // 将给定描述符数组中第 i 个位置设置为 tmp 的 PyArray_Descr 类型
            given_descrs[i] = (PyArray_Descr *)tmp;
        }
        // 如果获取的元素类型不是预期的 PyArray_Descr 类型或 Py_None，设置类型错误并返回空指针异常
        else {
            PyErr_SetString(PyExc_TypeError,
                    "dtype tuple can only contain dtype instances or None.");
            return NULL;
        }
    }

    // 初始化视图偏移量为 NPY_MIN_INTP
    npy_intp view_offset = NPY_MIN_INTP;
    // 解析描述符并获取转换规则
    NPY_CASTING casting = self->method->resolve_descriptors(
            self->method, self->dtypes, given_descrs, loop_descrs, &view_offset);

    // 如果转换规则小于 0 并且发生异常，返回空指针异常
    if (casting < 0 && PyErr_Occurred()) {
        return NULL;
    }
    // 如果转换规则小于 0，返回具有整数、Py_None、Py_None 值的元组
    else if (casting < 0) {
        return Py_BuildValue("iOO", casting, Py_None, Py_None);
    }

    // 创建长度为 nin + nout 的元组对象 result_tuple
    PyObject *result_tuple = PyTuple_New(nin + nout);
    // 如果创建元组对象失败，返回空指针异常
    if (result_tuple == NULL) {
        return NULL;
    }
    // 将 loop_descrs 中的对象转移所有权给元组
    for (int i = 0; i < nin + nout; i++) {
        /* transfer ownership to the tuple. */
        PyTuple_SET_ITEM(result_tuple, i, (PyObject *)loop_descrs[i]);
    }

    // 初始化视图偏移量对象为 Py_None
    PyObject *view_offset_obj;
    if (view_offset == NPY_MIN_INTP) {
        Py_INCREF(Py_None);
        view_offset_obj = Py_None;
    }
    // 否则，创建一个 PyLong 对象表示视图偏移量
    else {
        view_offset_obj = PyLong_FromSsize_t(view_offset);
        // 如果创建失败，释放 result_tuple 并返回空指针异常
        if (view_offset_obj == NULL) {
            Py_DECREF(result_tuple);
            return NULL;
        }
    }

    /*
     * The casting flags should be the most generic casting level.
     * If no input is parametric, it must match exactly.
     *
     * (Note that these checks are only debugging checks.)
     */
    // 初始化 parametric 变量为 0，用于检查是否存在参数化输入
    int parametric = 0;
    // 检查 self->dtypes 中是否存在参数化输入，设置 parametric 为 1 并退出循环
    for (int i = 0; i < nin + nout; i++) {
        if (NPY_DT_is_parametric(self->dtypes[i])) {
            parametric = 1;
            break;
        }
    }
    // 如果当前对象的方法中的转换值不为 -1，则执行以下代码块
    if (self->method->casting != -1) {
        // 将当前函数中的 casting 值存入本地变量 cast
        NPY_CASTING cast = casting;
        // 如果当前对象的方法中的转换值与通过 PyArray_MinCastSafety 函数计算的值不匹配
        if (self->method->casting !=
                PyArray_MinCastSafety(cast, self->method->casting)) {
            // 抛出运行时错误，说明描述符解析中的转换级别与存储的不匹配
            PyErr_Format(PyExc_RuntimeError,
                    "resolve_descriptors cast level did not match stored one. "
                    "(set level is %d, got %d for method %s)",
                    self->method->casting, cast, self->method->name);
            // 释放引用并返回 NULL
            Py_DECREF(result_tuple);
            Py_DECREF(view_offset_obj);
            return NULL;
        }
        // 如果不是参数化情况下
        if (!parametric) {
            /*
             * 非参数化情况只有在从等效转换到无转换时才可能不匹配
             * (例如由于字节顺序的变化)。
             */
            // 如果转换级别与当前对象的方法中的转换级别不同，并且当前方法的转换级别不是 NPY_EQUIV_CASTING
            if (cast != self->method->casting &&
                    self->method->casting != NPY_EQUIV_CASTING) {
                // 抛出运行时错误，说明转换级别发生变化，虽然转换是非参数化的，唯一可能的变化只能是从等效到无转换
                PyErr_Format(PyExc_RuntimeError,
                        "resolve_descriptors cast level changed even though "
                        "the cast is non-parametric where the only possible "
                        "change should be from equivalent to no casting. "
                        "(set level is %d, got %d for method %s)",
                        self->method->casting, cast, self->method->name);
                // 释放引用并返回 NULL
                Py_DECREF(result_tuple);
                Py_DECREF(view_offset_obj);
                return NULL;
            }
        }
    }

    // 返回一个 Python 对象，格式化为 "iNN"，包括 casting、result_tuple 和 view_offset_obj
    return Py_BuildValue("iNN", casting, result_tuple, view_offset_obj);
/*
 * TODO: This function is not public API, and certain code paths will need
 *       changes and especially testing if they were to be made public.
 */
static PyObject *
boundarraymethod__simple_strided_call(
        PyBoundArrayMethodObject *self, PyObject *arr_tuple)
{
    // 定义存储输入和输出数组、描述符、输出描述符、步长、参数长度等变量
    PyArrayObject *arrays[NPY_MAXARGS];
    PyArray_Descr *descrs[NPY_MAXARGS];
    PyArray_Descr *out_descrs[NPY_MAXARGS];
    char *args[NPY_MAXARGS];
    npy_intp strides[NPY_MAXARGS];
    Py_ssize_t length = -1;
    int aligned = 1;
    int nin = self->method->nin; // 获取输入参数数量
    int nout = self->method->nout; // 获取输出参数数量

    // 检查输入的参数是否为元组，并且元组长度是否等于输入参数数量加上输出参数数量
    if (!PyTuple_CheckExact(arr_tuple) ||
            PyTuple_Size(arr_tuple) != nin + nout) {
        PyErr_Format(PyExc_TypeError,
                "_simple_strided_call() takes exactly one tuple with as many "
                "arrays as the method takes arguments (%d+%d).", nin, nout);
        return NULL;
    }

    // 遍历输入和输出数组
    for (int i = 0; i < nin + nout; i++) {
        // 从元组中获取第i个元素作为输入或输出数组
        PyObject *tmp = PyTuple_GetItem(arr_tuple, i);
        if (tmp == NULL) {
            return NULL; // 获取元素失败，返回错误
        }
        else if (!PyArray_CheckExact(tmp)) {
            PyErr_SetString(PyExc_TypeError,
                    "All inputs must be NumPy arrays.");
            return NULL; // 非NumPy数组，返回错误
        }
        // 将获取的数组强制转换为PyArrayObject类型
        arrays[i] = (PyArrayObject *)tmp;
        // 获取数组的描述符
        descrs[i] = PyArray_DESCR(arrays[i]);

        // 检查输入的描述符是否与绑定的DType类的类型匹配
        if (Py_TYPE(descrs[i]) != (PyTypeObject *)self->dtypes[i]) {
            PyErr_Format(PyExc_TypeError,
                    "input dtype %S was not an exact instance of the bound "
                    "DType class %S.", descrs[i], self->dtypes[i]);
            return NULL; // 描述符类型不匹配，返回错误
        }
        // 检查数组是否为一维数组
        if (PyArray_NDIM(arrays[i]) != 1) {
            PyErr_SetString(PyExc_ValueError,
                    "All arrays must be one dimensional.");
            return NULL; // 非一维数组，返回错误
        }
        // 获取第一个数组的长度作为参考长度
        if (i == 0) {
            length = PyArray_SIZE(arrays[i]);
        }
        else if (PyArray_SIZE(arrays[i]) != length) {
            PyErr_SetString(PyExc_ValueError,
                    "All arrays must have the same length.");
            return NULL; // 数组长度不一致，返回错误
        }
        // 如果是输出数组，检查是否可写
        if (i >= nin) {
            if (PyArray_FailUnlessWriteable(
                    arrays[i], "_simple_strided_call() output") < 0) {
                return NULL; // 输出数组不可写，返回错误
            }
        }

        // 获取数组的数据指针和步长
        args[i] = PyArray_BYTES(arrays[i]);
        strides[i] = PyArray_STRIDES(arrays[i])[0];
        // 检查数组是否是对齐的
        aligned &= PyArray_ISALIGNED(arrays[i]);
    }

    // 如果存在未对齐的数组，并且方法不支持非对齐输入，返回错误
    if (!aligned && !(self->method->flags & NPY_METH_SUPPORTS_UNALIGNED)) {
        PyErr_SetString(PyExc_ValueError,
                "method does not support unaligned input.");
        return NULL;
    }

    // 定义视图偏移和转换模式，解析描述符
    npy_intp view_offset = NPY_MIN_INTP;
    NPY_CASTING casting = self->method->resolve_descriptors(
            self->method, self->dtypes, descrs, out_descrs, &view_offset);
    // 如果 casting 小于 0，表示类型转换失败
    if (casting < 0) {
        // 保存当前的错误类型、值和回溯信息
        PyObject *err_type = NULL, *err_value = NULL, *err_traceback = NULL;
        PyErr_Fetch(&err_type, &err_value, &err_traceback);
        // 设置类型错误异常并链式传播之前保存的异常信息
        PyErr_SetString(PyExc_TypeError,
                "cannot perform method call with the given dtypes.");
        npy_PyErr_ChainExceptions(err_type, err_value, err_traceback);
        // 返回 NULL 表示异常
        return NULL;
    }

    // 标记是否有类型需要调整
    int dtypes_were_adapted = 0;
    // 遍历输入输出描述符列表，检查是否有需要调整的描述符
    for (int i = 0; i < nin + nout; i++) {
        /* NOTE: This check is probably much stricter than necessary... */
        // 使用按位或运算符检查当前索引的描述符是否需要调整
        dtypes_were_adapted |= descrs[i] != out_descrs[i];
        // 减少输出描述符的引用计数，以便内存管理
        Py_DECREF(out_descrs[i]);
    }
    // 如果有类型需要调整，则设置类型错误异常
    if (dtypes_were_adapted) {
        PyErr_SetString(PyExc_TypeError,
                "_simple_strided_call(): requires dtypes to not require a cast "
                "(must match exactly with `_resolve_descriptors()`).");
        // 返回 NULL 表示异常
        return NULL;
    }

    // 准备数组方法调用的上下文信息
    PyArrayMethod_Context context = {
            .caller = NULL,
            .method = self->method,
            .descriptors = descrs,
    };
    PyArrayMethod_StridedLoop *strided_loop = NULL;
    NpyAuxData *loop_data = NULL;
    NPY_ARRAYMETHOD_FLAGS flags = 0;

    // 获取数组方法的跨步循环函数和相关数据
    if (self->method->get_strided_loop(
            &context, aligned, 0, strides,
            &strided_loop, &loop_data, &flags) < 0) {
        // 如果获取失败，返回 NULL 表示异常
        return NULL;
    }

    /*
     * TODO: 如果有请求，添加浮点数错误检查，并在标志允许的情况下释放 GIL。
     */
    
    // 执行跨步循环函数，并获取返回值
    int res = strided_loop(&context, args, &length, strides, loop_data);
    // 如果存在循环数据，释放其占用的资源
    if (loop_data != NULL) {
        loop_data->free(loop_data);
    }
    // 如果执行结果小于 0，返回 NULL 表示异常
    if (res < 0) {
        return NULL;
    }
    // 返回 Py_None 表示成功执行，没有返回值需要传递
    Py_RETURN_NONE;
/*
 * Support for masked inner-strided loops.  Masked inner-strided loops are
 * only used in the ufunc machinery.  So this special cases them.
 * In the future it probably makes sense to create an::
 *
 *     Arraymethod->get_masked_strided_loop()
 *
 * Function which this can wrap instead.
 */

/*
 * 定义了一个结构体 _masked_stridedloop_data，用于支持带掩码的内部步进循环。
 * 带掩码的内部步进循环仅在ufunc机制中使用。因此这里对它们进行特殊处理。
 * 未来可能会创建一个函数 Arraymethod->get_masked_strided_loop() 来替代它。
 */
typedef struct {
    NpyAuxData base;
    PyArrayMethod_StridedLoop *unmasked_stridedloop;
    NpyAuxData *unmasked_auxdata;
    int nargs;
    char *dataptrs[];
} _masked_stridedloop_data;

/*
 * 释放 _masked_stridedloop_data 类型的辅助数据的函数
 */
static void
_masked_stridedloop_data_free(NpyAuxData *auxdata)
{
    _masked_stridedloop_data *data = (_masked_stridedloop_data *)auxdata;
    NPY_AUXDATA_FREE(data->unmasked_auxdata);
    PyMem_Free(data);
}

/*
 * 这个函数将一个常规的未掩码步进循环封装成带掩码的步进循环，仅对掩码为True的元素调用函数。
 *
 * TODO: Reducers（减少器）也使用此代码来实现带掩码的减少操作。
 *       在合并它们之前，reducers 对广播有一个特例：当掩码步幅为0时，代码不会像 npy_memchr 当前那样检查所有元素。
 *       如果广播掩码足够普遍，重新添加这样的优化可能是值得的。
 */
static int
generic_masked_strided_loop(PyArrayMethod_Context *context,
        char *const *data, const npy_intp *dimensions,
        const npy_intp *strides, NpyAuxData *_auxdata)
{
    _masked_stridedloop_data *auxdata = (_masked_stridedloop_data *)_auxdata;
    int nargs = auxdata->nargs;
    PyArrayMethod_StridedLoop *strided_loop = auxdata->unmasked_stridedloop;
    NpyAuxData *strided_loop_auxdata = auxdata->unmasked_auxdata;

    char **dataptrs = auxdata->dataptrs;
    memcpy(dataptrs, data, nargs * sizeof(char *));
    char *mask = data[nargs];
    npy_intp mask_stride = strides[nargs];

    npy_intp N = dimensions[0];
    /* Process the data as runs of unmasked values */
    do {
        Py_ssize_t subloopsize;

        /* Skip masked values */
        mask = npy_memchr(mask, 0, mask_stride, N, &subloopsize, 1);
        for (int i = 0; i < nargs; i++) {
            dataptrs[i] += subloopsize * strides[i];
        }
        N -= subloopsize;

        /* Process unmasked values */
        mask = npy_memchr(mask, 0, mask_stride, N, &subloopsize, 0);
        if (subloopsize > 0) {
            int res = strided_loop(context,
                    dataptrs, &subloopsize, strides, strided_loop_auxdata);
            if (res != 0) {
                return res;
            }
            for (int i = 0; i < nargs; i++) {
                dataptrs[i] += subloopsize * strides[i];
            }
            N -= subloopsize;
        }
    } while (N > 0);

    return 0;
}
/*
 * Fetches a strided-loop function that supports a boolean mask as additional
 * (last) operand to the strided-loop.  It is otherwise largely identical to
 * the `get_strided_loop` method which it wraps.
 * This is the core implementation for the ufunc `where=...` keyword argument.
 *
 * NOTE: This function does not support `move_references` or inner dimensions.
 */
NPY_NO_EXPORT int
PyArrayMethod_GetMaskedStridedLoop(
        PyArrayMethod_Context *context,
        int aligned, npy_intp *fixed_strides,
        PyArrayMethod_StridedLoop **out_loop,
        NpyAuxData **out_transferdata,
        NPY_ARRAYMETHOD_FLAGS *flags)
{
    _masked_stridedloop_data *data;
    int nargs = context->method->nin + context->method->nout;

    /* Allocate memory for _masked_stridedloop_data struct */
    data = PyMem_Malloc(sizeof(_masked_stridedloop_data) +
                        sizeof(char *) * nargs);
    if (data == NULL) {
        PyErr_NoMemory();  /* Raise memory error if allocation fails */
        return -1;  /* Return -1 to indicate failure */
    }
    data->base.free = _masked_stridedloop_data_free;  /* Set free function */
    data->base.clone = NULL;  /* Set clone function as not used */
    data->unmasked_stridedloop = NULL;  /* Initialize unmasked strided loop */

    /* Retrieve unmasked strided loop using context method */
    if (context->method->get_strided_loop(context,
            aligned, 0, fixed_strides,
            &data->unmasked_stridedloop, &data->unmasked_auxdata, flags) < 0) {
        PyMem_Free(data);  /* Free allocated memory on failure */
        return -1;  /* Return -1 to indicate failure */
    }
    *out_transferdata = (NpyAuxData *)data;  /* Set output transfer data */
    *out_loop = generic_masked_strided_loop;  /* Set output loop function */
    return 0;  /* Return 0 to indicate success */
}


/* Definition of methods associated with PyBoundArrayMethodObject */
PyMethodDef boundarraymethod_methods[] = {
    {"_resolve_descriptors", (PyCFunction)boundarraymethod__resolve_descripors,
     METH_O, "Resolve the given dtypes."},
    {"_simple_strided_call", (PyCFunction)boundarraymethod__simple_strided_call,
     METH_O, "call on 1-d inputs and pre-allocated outputs (single call)."},
    {NULL, 0, 0, NULL},  /* Sentinel indicating end of method definitions */
};


/* Function to retrieve whether the method supports unaligned inputs/outputs */
static PyObject *
boundarraymethod__supports_unaligned(PyBoundArrayMethodObject *self)
{
    return PyBool_FromLong(self->method->flags & NPY_METH_SUPPORTS_UNALIGNED);
}


/* Getter definitions for PyBoundArrayMethodObject attributes */
PyGetSetDef boundarraymethods_getters[] = {
    {"_supports_unaligned",
     (getter)boundarraymethod__supports_unaligned, NULL,
     "whether the method supports unaligned inputs/outputs.", NULL},
    {NULL, NULL, NULL, NULL, NULL},  /* Sentinel indicating end of getter definitions */
};


/* Type definition for PyBoundArrayMethod_Type */
NPY_NO_EXPORT PyTypeObject PyBoundArrayMethod_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)  /* Initialize base object */
    .tp_name = "numpy._BoundArrayMethod",  /* Type name */
    .tp_basicsize = sizeof(PyBoundArrayMethodObject),  /* Size of the object */
    .tp_dealloc = boundarraymethod_dealloc,  /* Deallocation function */
    .tp_repr = (reprfunc)boundarraymethod_repr,  /* Representation function */
    .tp_flags = Py_TPFLAGS_DEFAULT,  /* Default flags */
    .tp_methods = boundarraymethod_methods,  /* Methods associated */
    .tp_getset = boundarraymethods_getters,  /* Getters and setters */
};
```