# `.\numpy\numpy\_core\include\numpy\dtype_api.h`

```py
/*
 * The public DType API
 */

#ifndef NUMPY_CORE_INCLUDE_NUMPY___DTYPE_API_H_
#define NUMPY_CORE_INCLUDE_NUMPY___DTYPE_API_H_

// 定义 PyArrayMethodObject_tag 结构体
struct PyArrayMethodObject_tag;

/*
 * Largely opaque struct for DType classes (i.e. metaclass instances).
 * The internal definition is currently in `ndarraytypes.h` (export is a bit
 * more complex because `PyArray_Descr` is a DTypeMeta internally but not
 * externally).
 */
#if !(defined(NPY_INTERNAL_BUILD) && NPY_INTERNAL_BUILD)

#ifndef Py_LIMITED_API

// 定义 PyArray_DTypeMeta_tag 结构体
typedef struct PyArray_DTypeMeta_tag {
    // 继承自 PyHeapTypeObject
    PyHeapTypeObject super;

    /*
    * Most DTypes will have a singleton default instance, for the
    * parametric legacy DTypes (bytes, string, void, datetime) this
    * may be a pointer to the *prototype* instance?
    */
    // 单例实例指针，用于大多数 DTypes，默认实例
    PyArray_Descr *singleton;
    // 复制传统 DTypes 的类型编号，通常无效
    int type_num;

    // 标量实例的类型对象（可能为 NULL）
    // DType 实例的标量类型对象
    PyTypeObject *scalar_type;
    /*
    * DType flags to signal legacy, parametric, or
    * abstract.  But plenty of space for additional information/flags.
    */
    // DType 标志，指示传统、参数化或抽象
    npy_uint64 flags;

    /*
    * Use indirection in order to allow a fixed size for this struct.
    * A stable ABI size makes creating a static DType less painful
    * while also ensuring flexibility for all opaque API (with one
    * indirection due the pointer lookup).
    */
    // 允许固定大小结构体的间接引用
    void *dt_slots;
    // 保留字段，允许增长（当前也超出此范围）
    void *reserved[3];
} PyArray_DTypeMeta;

#else

// 限制 API 的情况下，将 PyArray_DTypeMeta 视为 PyTypeObject
typedef PyTypeObject PyArray_DTypeMeta;

#endif /* Py_LIMITED_API */

#endif  /* not internal build */

/*
 * ******************************************************
 *         ArrayMethod API (Casting and UFuncs)
 * ******************************************************
 */

// 定义枚举类型，用于描述 ArrayMethod 的属性
typedef enum {
    /* Flag for whether the GIL is required */
    NPY_METH_REQUIRES_PYAPI = 1 << 0,
    /*
     * Some functions cannot set floating point error flags, this flag
     * gives us the option (not requirement) to skip floating point error
     * setup/check. No function should set error flags and ignore them
     * since it would interfere with chaining operations (e.g. casting).
     */
    // 指示是否需要 GIL
    NPY_METH_NO_FLOATINGPOINT_ERRORS = 1 << 1,
    /* Whether the method supports unaligned access (not runtime) */
    // 方法是否支持非对齐访问
    NPY_METH_SUPPORTS_UNALIGNED = 1 << 2,
    /*
     * Used for reductions to allow reordering the operation.  At this point
     * assume that if set, it also applies to normal operations though!
     */
    // 用于允许重排序操作的标志
    NPY_METH_IS_REORDERABLE = 1 << 3,
    /*
     * Private flag for now for *logic* functions.  The logical functions
     * `logical_or` and `logical_and` can always cast the inputs to booleans
     * "safely" (because that is how the cast to bool is defined).
     * @seberg: I am not sure this is the best way to handle this, so its
     * private for now (also it is very limited anyway).
     * There is one "exception". NA aware dtypes cannot cast to bool
     * (hopefully), so the `??->?` loop should error even with this flag.
     * But a second NA fallback loop will be necessary.
     */
    _NPY_METH_FORCE_CAST_INPUTS = 1 << 17,
    
    /* All flags which can change at runtime */
    NPY_METH_RUNTIME_FLAGS = (
            NPY_METH_REQUIRES_PYAPI |
            NPY_METH_NO_FLOATINGPOINT_ERRORS),
    
    
    注释：
    
    
    # 定义一个私有标志位，目前仅用于逻辑函数。逻辑函数 `logical_or` 和 `logical_and` 可以安全地将输入强制转换为布尔值
    # （因为这是 bool 强制转换的定义方式）。
    # @seberg: 我不确定这是否是处理此问题的最佳方式，因此目前将其设置为私有（而且其功能也非常有限）。
    # 有一个“例外情况”。对 NA 意识到的数据类型不能转换为布尔值（希望如此），因此即使有此标志，`??->?` 循环也应出错。
    # 但是第二个 NA 回退循环将是必要的。
    _NPY_METH_FORCE_CAST_INPUTS = 1 << 17,
    
    # 所有可能在运行时更改的标志位
    NPY_METH_RUNTIME_FLAGS = (
            NPY_METH_REQUIRES_PYAPI |
            NPY_METH_NO_FLOATINGPOINT_ERRORS),
/*
 * 结构体定义：NPY_ARRAYMETHOD_FLAGS
 * ------------------------------
 * 描述一个数组方法的标志位集合。
 */
} NPY_ARRAYMETHOD_FLAGS;


/*
 * 结构体定义：PyArrayMethod_Context_tag
 * ------------------------------------
 * 描述数组方法的上下文信息。
 */
typedef struct PyArrayMethod_Context_tag {
    /* 调用者，通常是原始的通用函数。可能为NULL */
    PyObject *caller;
    /* 方法的"self"对象，目前是一个不透明对象 */
    struct PyArrayMethodObject_tag *method;

    /* 操作数描述符，在resolve_descriptors函数中填充 */
    PyArray_Descr *const *descriptors;
    /* 结构体可能会扩展（对DType作者无害） */
} PyArrayMethod_Context;


/*
 * 主要对象：PyArrayMethod_Spec
 * ----------------------------
 * 用于创建新数组方法的主要对象。使用Python有限API中的典型“slots”机制。
 */
typedef struct {
    const char *name;
    int nin, nout;
    NPY_CASTING casting;
    NPY_ARRAYMETHOD_FLAGS flags;
    PyArray_DTypeMeta **dtypes;
    PyType_Slot *slots;
} PyArrayMethod_Spec;


/*
 * 数组方法的槽位定义
 * -----------------
 *
 * 数组方法的创建槽位ID。一旦完全公开，ID将固定，但可以弃用和任意扩展。
 */
#define _NPY_METH_resolve_descriptors_with_scalars 1
#define NPY_METH_resolve_descriptors 2
#define NPY_METH_get_loop 3
#define NPY_METH_get_reduction_initial 4
/* 用于构造/默认get_loop的特定循环： */
#define NPY_METH_strided_loop 5
#define NPY_METH_contiguous_loop 6
#define NPY_METH_unaligned_strided_loop 7
#define NPY_METH_unaligned_contiguous_loop 8
#define NPY_METH_contiguous_indexed_loop 9
#define _NPY_METH_static_data 10


/*
 * 解析描述符函数
 * ---------------
 *
 * 必须能够处理所有输出（但不是输入）的NULL值，并填充loop_descrs。
 * 如果操作不可能无错误地执行，则返回-1；如果没有错误设置，则返回0。
 * 对于正常函数，几乎总是返回"safe"（或者"equivalent"）。
 *
 * 如果所有输出DType都是非参数化的，则resolve_descriptors函数是可选的。
 */
typedef NPY_CASTING (PyArrayMethod_ResolveDescriptors)(
        /* "method"目前是不透明的（例如在Python中包装时必需）。 */
        struct PyArrayMethodObject_tag *method,
        /* 方法创建时使用的DTypes */
        PyArray_DTypeMeta *const *dtypes,
        /* 输入描述符（实例）。输出可能为NULL。 */
        PyArray_Descr *const *given_descrs,
        /* 必须在错误时不持有引用的确切循环描述符 */
        PyArray_Descr **loop_descrs,
        npy_intp *view_offset);


/*
 * 很少需要的、稍微更强大版本的resolve_descriptors函数。
 * 详细信息请参见`PyArrayMethod_ResolveDescriptors`。
 *
 * 注意：此函数现在是私有的，因为不清楚如何以及确切传递额外信息以处理标量。
 * 参见gh-24915。
 */
/**
 * Define a typedef for a function resolving descriptors with a scalar, given a
 * PyArrayMethodObject, an array of dtype meta pointers, an array of possibly
 * NULL descriptors, an array of input scalars or NULL, an array of loop
 * descriptors, and a view offset.
 *
 * @param method The PyArrayMethodObject for which descriptors are being resolved.
 * @param dtypes An array of dtype meta pointers for the method.
 * @param given_descrs An array of possibly NULL descriptors.
 * @param input_scalars An array of input scalars or NULL.
 * @param loop_descrs An array of loop descriptors to be filled.
 * @param view_offset A pointer to the view offset value.
 * @returns NPY_CASTING indicating the casting method.
 */
typedef NPY_CASTING (PyArrayMethod_ResolveDescriptorsWithScalar)(
        struct PyArrayMethodObject_tag *method,
        PyArray_DTypeMeta *const *dtypes,
        PyArray_Descr *const *given_descrs,
        PyObject *const *input_scalars,
        PyArray_Descr **loop_descrs,
        npy_intp *view_offset);



/**
 * Define a typedef for a strided loop function, taking a PyArrayMethod_Context
 * pointer, a data pointer array, dimension sizes, stride sizes, and transfer
 * data.
 *
 * @param context The PyArrayMethod_Context containing loop information.
 * @param data An array of pointers to data arrays for the strided loop.
 * @param dimensions An array of dimension sizes.
 * @param strides An array of stride sizes.
 * @param transferdata The transfer data for the strided loop.
 * @returns An integer indicating success or failure of the strided loop.
 */
typedef int (PyArrayMethod_StridedLoop)(PyArrayMethod_Context *context,
        char *const *data, const npy_intp *dimensions, const npy_intp *strides,
        NpyAuxData *transferdata);


/**
 * Define a typedef for a function getting a loop from an ArrayMethod_Context,
 * considering alignment, move references, strides, loop function pointer,
 * transfer data, and method flags.
 *
 * @param context The PyArrayMethod_Context containing method context.
 * @param aligned Whether the loop should be aligned.
 * @param move_references Whether references should be moved.
 * @param strides An array of stride sizes.
 * @param out_loop The output pointer to the strided loop function.
 * @param out_transferdata The output pointer to the transfer data.
 * @param flags The method flags for the function.
 * @returns An integer indicating success or failure in getting the loop.
 */
typedef int (PyArrayMethod_GetLoop)(
        PyArrayMethod_Context *context,
        int aligned, int move_references,
        const npy_intp *strides,
        PyArrayMethod_StridedLoop **out_loop,
        NpyAuxData **out_transferdata,
        NPY_ARRAYMETHOD_FLAGS *flags);

/**
 * Query an ArrayMethod for the initial value for use in reduction, considering
 * context, reduction status, and initial data to be filled.
 *
 * @param context The PyArrayMethod_Context for accessing descriptors.
 * @param reduction_is_empty Whether the reduction is empty.
 * @param initial Pointer to initial data to be filled if possible.
 * @returns -1, 0, or 1 indicating error, no initial value, or successful
 *          initialization.
 */
typedef int (PyArrayMethod_GetReductionInitial)(
        PyArrayMethod_Context *context, npy_bool reduction_is_empty,
        void *initial);

/*
 * The following functions are only used by the wrapping array method defined
 * in umath/wrapping_array_method.c
 */
/*
 * The function to convert the given descriptors (passed in to
 * `resolve_descriptors`) and translates them for the wrapped loop.
 * The new descriptors MUST be viewable with the old ones, `NULL` must be
 * supported (for outputs) and should normally be forwarded.
 *
 * The function must clean up on error.
 *
 * NOTE: We currently assume that this translation gives "viewable" results.
 *       I.e. there is no additional casting related to the wrapping process.
 *       In principle that could be supported, but not sure it is useful.
 *       This currently also means that e.g. alignment must apply identically
 *       to the new dtypes.
 *
 * TODO: Due to the fact that `resolve_descriptors` is also used for `can_cast`
 *       there is no way to "pass out" the result of this function.  This means
 *       it will be called twice for every ufunc call.
 *       (I am considering including `auxdata` as an "optional" parameter to
 *       `resolve_descriptors`, so that it can be filled there if not NULL.)
 */
typedef int (PyArrayMethod_TranslateGivenDescriptors)(int nin, int nout,
        PyArray_DTypeMeta *const wrapped_dtypes[],
        PyArray_Descr *const given_descrs[], PyArray_Descr *new_descrs[]);

/**
 * The function to convert the actual loop descriptors (as returned by the
 * original `resolve_descriptors` function) to the ones the output array
 * should use.
 * This function must return "viewable" types, it must not mutate them in any
 * form that would break the inner-loop logic.  Does not need to support NULL.
 *
 * The function must clean up on error.
 *
 * @param nargs Number of arguments
 * @param new_dtypes The DTypes of the output (usually probably not needed)
 * @param given_descrs Original given_descrs to the resolver, necessary to
 *        fetch any information related to the new dtypes from the original.
 * @param original_descrs The `loop_descrs` returned by the wrapped loop.
 * @param loop_descrs The output descriptors, compatible to `original_descrs`.
 *
 * @returns 0 on success, -1 on failure.
 */
typedef int (PyArrayMethod_TranslateLoopDescriptors)(int nin, int nout,
        PyArray_DTypeMeta *const new_dtypes[], PyArray_Descr *const given_descrs[],
        PyArray_Descr *original_descrs[], PyArray_Descr *loop_descrs[]);
/*
 * A traverse loop working on a single array. This is similar to the general
 * strided-loop function. This is designed for loops that need to visit every
 * element of a single array.
 *
 * Currently this is used for array clearing, via the NPY_DT_get_clear_loop
 * API hook, and zero-filling, via the NPY_DT_get_fill_zero_loop API hook.
 * These are most useful for handling arrays storing embedded references to
 * python objects or heap-allocated data.
 *
 * The `void *traverse_context` is passed in because we may need to pass in
 * Interpreter state or similar in the future, but we don't want to pass in
 * a full context (with pointers to dtypes, method, caller which all make
 * no sense for a traverse function).
 *
 * We assume for now that this context can be just passed through in the
 * the future (for structured dtypes).
 *
 */
typedef int (PyArrayMethod_TraverseLoop)(
        void *traverse_context, const PyArray_Descr *descr, char *data,
        npy_intp size, npy_intp stride, NpyAuxData *auxdata);


/*
 * Simplified get_loop function specific to dtype traversal
 *
 * It should set the flags needed for the traversal loop and set out_loop to the
 * loop function, which must be a valid PyArrayMethod_TraverseLoop
 * pointer. Currently this is used for zero-filling and clearing arrays storing
 * embedded references.
 *
 */
typedef int (PyArrayMethod_GetTraverseLoop)(
        void *traverse_context, const PyArray_Descr *descr,
        int aligned, npy_intp fixed_stride,
        PyArrayMethod_TraverseLoop **out_loop, NpyAuxData **out_auxdata,
        NPY_ARRAYMETHOD_FLAGS *flags);


/*
 * Type of the C promoter function, which must be wrapped into a
 * PyCapsule with name "numpy._ufunc_promoter".
 *
 * Note that currently the output dtypes are always NULL unless they are
 * also part of the signature. This is an implementation detail and could
 * change in the future. However, in general promoters should not have a
 * need for output dtypes.
 * (There are potential use-cases, these are currently unsupported.)
 */
typedef int (PyArrayMethod_PromoterFunction)(PyObject *ufunc,
        PyArray_DTypeMeta *const op_dtypes[], PyArray_DTypeMeta *const signature[],
        PyArray_DTypeMeta *new_op_dtypes[]);

/*
 * ****************************
 *          DTYPE API
 * ****************************
 */

#define NPY_DT_ABSTRACT 1 << 1
// 标志位，表示数据类型是抽象的
#define NPY_DT_PARAMETRIC 1 << 2
// 标志位，表示数据类型是参数化的
#define NPY_DT_NUMERIC 1 << 3
// 标志位，表示数据类型是数值类型

/*
 * These correspond to slots in the NPY_DType_Slots struct and must
 * be in the same order as the members of that struct. If new slots
 * get added or old slots get removed NPY_NUM_DTYPE_SLOTS must also
 * be updated
 */

#define NPY_DT_discover_descr_from_pyobject 1
// 数据类型 API 中的槽位，用于从 Python 对象中发现描述符
// 此槽位被视为私有，因为其 API 尚未确定
#define _NPY_DT_is_known_scalar_type 2
// 槽位，用于确定标量类型是否已知
#define NPY_DT_default_descr 3
// 槽位，用于获取默认描述符
#define NPY_DT_common_dtype 4
// 槽位，用于获取通用数据类型
#define NPY_DT_common_instance 5
// 槽位，用于获取通用实例
#define NPY_DT_ensure_canonical 6
// 槽位，用于确保规范化
#define NPY_DT_setitem 7
// 槽位，用于设置项目
#define NPY_DT_getitem 8
// 槽位，用于获取项目
// 定义常量 NPY_DT_get_clear_loop 的值为 9
#define NPY_DT_get_clear_loop 9

// 定义常量 NPY_DT_get_fill_zero_loop 的值为 10
#define NPY_DT_get_fill_zero_loop 10

// 定义常量 NPY_DT_finalize_descr 的值为 11
#define NPY_DT_finalize_descr 11

// 这些 PyArray_ArrFunc 槽位将会被弃用并最终替换
// getitem 和 setitem 可以作为性能优化定义;
// 默认情况下，用户自定义的数据类型调用 `legacy_getitem_using_DType`
// 和 `legacy_setitem_using_DType`，分别对应获取和设置操作。此功能仅支持基本的 NumPy 数据类型。

// 用于将 dtype 的槽位与 arrfuncs 的槽位分隔开来
// 本意仅用于内部使用，但在此处定义以增加清晰度
#define _NPY_DT_ARRFUNCS_OFFSET (1 << 10)

// 禁用 Cast 操作
// #define NPY_DT_PyArray_ArrFuncs_cast 0 + _NPY_DT_ARRFUNCS_OFFSET

// 定义常量 NPY_DT_PyArray_ArrFuncs_getitem 的值为 1 + _NPY_DT_ARRFUNCS_OFFSET
#define NPY_DT_PyArray_ArrFuncs_getitem 1 + _NPY_DT_ARRFUNCS_OFFSET

// 定义常量 NPY_DT_PyArray_ArrFuncs_setitem 的值为 2 + _NPY_DT_ARRFUNCS_OFFSET
#define NPY_DT_PyArray_ArrFuncs_setitem 2 + _NPY_DT_ARRFUNCS_OFFSET

// 禁用 Copyswap 操作
// #define NPY_DT_PyArray_ArrFuncs_copyswapn 3 + _NPY_DT_ARRFUNCS_OFFSET
// #define NPY_DT_PyArray_ArrFuncs_copyswap 4 + _NPY_DT_ARRFUNCS_OFFSET

// 定义常量 NPY_DT_PyArray_ArrFuncs_compare 的值为 5 + _NPY_DT_ARRFUNCS_OFFSET
#define NPY_DT_PyArray_ArrFuncs_compare 5 + _NPY_DT_ARRFUNCS_OFFSET

// 定义常量 NPY_DT_PyArray_ArrFuncs_argmax 的值为 6 + _NPY_DT_ARRFUNCS_OFFSET
#define NPY_DT_PyArray_ArrFuncs_argmax 6 + _NPY_DT_ARRFUNCS_OFFSET

// 定义常量 NPY_DT_PyArray_ArrFuncs_dotfunc 的值为 7 + _NPY_DT_ARRFUNCS_OFFSET
#define NPY_DT_PyArray_ArrFuncs_dotfunc 7 + _NPY_DT_ARRFUNCS_OFFSET

// 定义常量 NPY_DT_PyArray_ArrFuncs_scanfunc 的值为 8 + _NPY_DT_ARRFUNCS_OFFSET
#define NPY_DT_PyArray_ArrFuncs_scanfunc 8 + _NPY_DT_ARRFUNCS_OFFSET

// 定义常量 NPY_DT_PyArray_ArrFuncs_fromstr 的值为 9 + _NPY_DT_ARRFUNCS_OFFSET
#define NPY_DT_PyArray_ArrFuncs_fromstr 9 + _NPY_DT_ARRFUNCS_OFFSET

// 定义常量 NPY_DT_PyArray_ArrFuncs_nonzero 的值为 10 + _NPY_DT_ARRFUNCS_OFFSET
#define NPY_DT_PyArray_ArrFuncs_nonzero 10 + _NPY_DT_ARRFUNCS_OFFSET

// 定义常量 NPY_DT_PyArray_ArrFuncs_fill 的值为 11 + _NPY_DT_ARRFUNCS_OFFSET
#define NPY_DT_PyArray_ArrFuncs_fill 11 + _NPY_DT_ARRFUNCS_OFFSET

// 定义常量 NPY_DT_PyArray_ArrFuncs_fillwithscalar 的值为 12 + _NPY_DT_ARRFUNCS_OFFSET
#define NPY_DT_PyArray_ArrFuncs_fillwithscalar 12 + _NPY_DT_ARRFUNCS_OFFSET

// 定义常量 NPY_DT_PyArray_ArrFuncs_sort 的值为 13 + _NPY_DT_ARRFUNCS_OFFSET
#define NPY_DT_PyArray_ArrFuncs_sort 13 + _NPY_DT_ARRFUNCS_OFFSET

// 定义常量 NPY_DT_PyArray_ArrFuncs_argsort 的值为 14 + _NPY_DT_ARRFUNCS_OFFSET
#define NPY_DT_PyArray_ArrFuncs_argsort 14 + _NPY_DT_ARRFUNCS_OFFSET

// Casting 相关的槽位被禁用。参考
// https://github.com/numpy/numpy/pull/23173#discussion_r1101098163
// #define NPY_DT_PyArray_ArrFuncs_castdict 15 + _NPY_DT_ARRFUNCS_OFFSET
// #define NPY_DT_PyArray_ArrFuncs_scalarkind 16 + _NPY_DT_ARRFUNCS_OFFSET
// #define NPY_DT_PyArray_ArrFuncs_cancastscalarkindto 17 + _NPY_DT_ARRFUNCS_OFFSET
// #define NPY_DT_PyArray_ArrFuncs_cancastto 18 + _NPY_DT_ARRFUNCS_OFFSET

// 这些在 NumPy 1.19 中已被弃用，因此在此处被禁用
// #define NPY_DT_PyArray_ArrFuncs_fastclip 19 + _NPY_DT_ARRFUNCS_OFFSET
// #define NPY_DT_PyArray_ArrFuncs_fastputmask 20 + _NPY_DT_ARRFUNCS_OFFSET
// #define NPY_DT_PyArray_ArrFuncs_fasttake 21 + _NPY_DT_ARRFUNCS_OFFSET

// 定义常量 NPY_DT_PyArray_ArrFuncs_argmin 的值为 22 + _NPY_DT_ARRFUNCS_OFFSET
#define NPY_DT_PyArray_ArrFuncs_argmin 22 + _NPY_DT_ARRFUNCS_OFFSET

// TODO: 这些槽位可能仍需要进一步思考，或者有一种“增长”的方式？
// 定义结构体 PyArrayDTypeMeta_Spec，描述了数组数据类型元信息
typedef struct {
    PyTypeObject *typeobj;    /* Python 标量的类型或者为 NULL */
    int flags;                /* 标志，包括参数化和抽象 */
    /* 用于定义空结束的转换定义。对于新创建的 DType，请使用 NULL */
    PyArrayMethod_Spec **casts;
    PyType_Slot *slots;
    /* 基类或者为 NULL（将始终是 `np.dtype` 的子类） */
    PyTypeObject *baseclass;
} PyArrayDTypeMeta_Spec;
/*
 * typedef声明定义了多个函数指针类型，这些类型用于与PyArray_DTypeMeta交互。
 * 这些函数包括从Python对象中发现描述符、确定是否为已知的标量类型、获取默认描述符、
 * 确定两个数据类型的常见数据类型、获取公共实例、确保规范化描述符、最终化描述符。
 */

/*
 * Convenience utility for getting a reference to the DType metaclass associated
 * with a dtype instance.
 */
#define NPY_DTYPE(descr) ((PyArray_DTypeMeta *)Py_TYPE(descr))

/*
 * 定义了一个静态内联函数NPY_DT_NewRef，用于增加PyArray_DTypeMeta实例的引用计数，
 * 并返回该实例的指针。
 */

/*
 * 结束了头文件的条件编译指令，防止重复包含NUMPY_CORE_INCLUDE_NUMPY___DTYPE_API_H_宏定义的头文件内容。
 */
#endif  /* NUMPY_CORE_INCLUDE_NUMPY___DTYPE_API_H_ */
```