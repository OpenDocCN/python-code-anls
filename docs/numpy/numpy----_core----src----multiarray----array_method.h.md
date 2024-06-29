# `.\numpy\numpy\_core\src\multiarray\array_method.h`

```py
#ifndef NUMPY_CORE_SRC_MULTIARRAY_ARRAY_METHOD_H_
#define NUMPY_CORE_SRC_MULTIARRAY_ARRAY_METHOD_H_

// 定义防止使用过时 API 的宏
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
// 定义 _MULTIARRAYMODULE 宏
#define _MULTIARRAYMODULE

// 包含 Python 标准头文件
#include <Python.h>
// 包含 NumPy 数组类型相关的头文件
#include <numpy/ndarraytypes.h>

#ifdef __cplusplus
extern "C" {
#endif

// 包含 NumPy 的 dtype API 头文件
#include "numpy/dtype_api.h"

/*
 * 以下是 PyArrayMethod_MINIMAL_FLAGS 宏的定义：
 * 默认最小标志位，目前仅指定了 NPY_METH_NO_FLOATINGPOINT_ERRORS
 */
#define PyArrayMethod_MINIMAL_FLAGS NPY_METH_NO_FLOATINGPOINT_ERRORS

/*
 * 定义 PyArrayMethod_COMBINED_FLAGS 宏：
 * 组合两个输入标志位，去除 PyArrayMethod_MINIMAL_FLAGS，然后合并剩余部分
 */
#define PyArrayMethod_COMBINED_FLAGS(flags1, flags2)  \
        ((NPY_ARRAYMETHOD_FLAGS)(  \
            ((flags1 | flags2) & ~PyArrayMethod_MINIMAL_FLAGS)  \
            | (flags1 & flags2)))

/*
 * 定义 PyArrayMethodObject_tag 结构体：
 * 该结构体描述了数组方法的属性和函数指针，不建议公开
 */
typedef struct PyArrayMethodObject_tag {
    PyObject_HEAD
    char *name;  // 方法名称
    int nin, nout;  // 输入和输出参数个数
    NPY_CASTING casting;  // 类型转换方式
    NPY_ARRAYMETHOD_FLAGS flags;  // 方法的标志位
    void *static_data;  // 方法可能需要的静态数据指针
    // 以下是用于解析描述符和获取循环的函数指针
    PyArrayMethod_ResolveDescriptorsWithScalar *resolve_descriptors_with_scalars;
    PyArrayMethod_ResolveDescriptors *resolve_descriptors;
    PyArrayMethod_GetLoop *get_strided_loop;
    PyArrayMethod_GetReductionInitial *get_reduction_initial;
    // 典型的循环函数指针
    PyArrayMethod_StridedLoop *strided_loop;
    PyArrayMethod_StridedLoop *contiguous_loop;
    PyArrayMethod_StridedLoop *unaligned_strided_loop;
    PyArrayMethod_StridedLoop *unaligned_contiguous_loop;
    PyArrayMethod_StridedLoop *contiguous_indexed_loop;
    // 用于包装在 umath 中定义的数组方法的结构体指针
    struct PyArrayMethodObject_tag *wrapped_meth;
    PyArray_DTypeMeta **wrapped_dtypes;  // 包装的数据类型数组
    // 用于翻译给定描述符和循环描述符的函数指针
    PyArrayMethod_TranslateGivenDescriptors *translate_given_descrs;
    PyArrayMethod_TranslateLoopDescriptors *translate_loop_descrs;
    // 保留给遗留回退数组方法使用的区块
    char legacy_initial[sizeof(npy_clongdouble)];  // 初始值存储
} PyArrayMethodObject;

#endif  // NUMPY_CORE_SRC_MULTIARRAY_ARRAY_METHOD_H_
/*
 * We will sometimes have to create a ArrayMethod and allow passing it around,
 * similar to `instance.method` returning a bound method, e.g. a function like
 * `ufunc.resolve()` can return a bound object.
 * The current main purpose of the BoundArrayMethod is that it holds on to the
 * `dtypes` (the classes), so that the `ArrayMethod` (e.g. for casts) will
 * not create references cycles.  In principle, it could hold any information
 * which is also stored on the ufunc (and thus does not need to be repeated
 * on the `ArrayMethod` itself.
 */
typedef struct {
    PyObject_HEAD
    // 指向 PyArray_DTypeMeta 指针的数组，用于存储数据类型信息
    PyArray_DTypeMeta **dtypes;
    // 指向 PyArrayMethodObject 对象的指针，表示绑定的数组方法对象
    PyArrayMethodObject *method;
} PyBoundArrayMethodObject;


// 外部声明 PyArrayMethod_Type 类型
extern NPY_NO_EXPORT PyTypeObject PyArrayMethod_Type;
// 外部声明 PyBoundArrayMethod_Type 类型
extern NPY_NO_EXPORT PyTypeObject PyBoundArrayMethod_Type;


/*
 * Used internally (initially) for real to complex loops only
 */
// NPY_NO_EXPORT 指示该函数在本文件中使用，作用是获取默认的分步循环函数
NPY_NO_EXPORT int
npy_default_get_strided_loop(
        PyArrayMethod_Context *context,
        int aligned, int NPY_UNUSED(move_references), const npy_intp *strides,
        PyArrayMethod_StridedLoop **out_loop, NpyAuxData **out_transferdata,
        NPY_ARRAYMETHOD_FLAGS *flags);


// NPY_NO_EXPORT 指示该函数在本文件中使用，作用是获取带掩码的分步循环函数
NPY_NO_EXPORT int
PyArrayMethod_GetMaskedStridedLoop(
        PyArrayMethod_Context *context,
        int aligned,
        npy_intp *fixed_strides,
        PyArrayMethod_StridedLoop **out_loop,
        NpyAuxData **out_transferdata,
        NPY_ARRAYMETHOD_FLAGS *flags);


// NPY_NO_EXPORT 指示该函数在本文件中使用，作用是根据规范创建数组方法对象
NPY_NO_EXPORT PyObject *
PyArrayMethod_FromSpec(PyArrayMethod_Spec *spec);


/*
 * TODO: This function is the internal version, and its error paths may
 *       need better tests when a public version is exposed.
 */
// NPY_NO_EXPORT 指示该函数在本文件中使用，作用是根据规范创建内部版本的绑定数组方法对象
NPY_NO_EXPORT PyBoundArrayMethodObject *
PyArrayMethod_FromSpec_int(PyArrayMethod_Spec *spec, int priv);

#ifdef __cplusplus
}
#endif

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_ARRAY_METHOD_H_ */
```