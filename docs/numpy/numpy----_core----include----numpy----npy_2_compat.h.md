# `.\numpy\numpy\_core\include\numpy\npy_2_compat.h`

```
/*
 * This header file defines relevant features which:
 * - Require runtime inspection depending on the NumPy version.
 * - May be needed when compiling with an older version of NumPy to allow
 *   a smooth transition.
 *
 * As such, it is shipped with NumPy 2.0, but designed to be vendored in full
 * or parts by downstream projects.
 *
 * It must be included after any other includes.  `import_array()` must have
 * been called in the scope or version dependency will misbehave, even when
 * only `PyUFunc_` API is used.
 *
 * If required complicated defs (with inline functions) should be written as:
 *
 *     #if NPY_FEATURE_VERSION >= NPY_2_0_API_VERSION
 *         Simple definition when NumPy 2.0 API is guaranteed.
 *     #else
 *         static inline definition of a 1.x compatibility shim
 *         #if NPY_ABI_VERSION < 0x02000000
 *            Make 1.x compatibility shim the public API (1.x only branch)
 *         #else
 *             Runtime dispatched version (1.x or 2.x)
 *         #endif
 *     #endif
 *
 * An internal build always passes NPY_FEATURE_VERSION >= NPY_2_0_API_VERSION
 */

#ifndef NUMPY_CORE_INCLUDE_NUMPY_NPY_2_COMPAT_H_
#define NUMPY_CORE_INCLUDE_NUMPY_NPY_2_COMPAT_H_

/*
 * New macros for accessing real and complex part of a complex number can be
 * found in "npy_2_complexcompat.h".
 */

/*
 * This header is meant to be included by downstream directly for 1.x compat.
 * In that case we need to ensure that users first included the full headers
 * and not just `ndarraytypes.h`.
 */

#ifndef NPY_FEATURE_VERSION
  #error "The NumPy 2 compat header requires `import_array()` for which "  \
         "the `ndarraytypes.h` header include is not sufficient.  Please "  \
         "include it after `numpy/ndarrayobject.h` or similar.\n"  \
         "To simplify inclusion, you may use `PyArray_ImportNumPy()` " \
         "which is defined in the compat header and is lightweight (can be)."
#endif

#if NPY_ABI_VERSION < 0x02000000
  /*
   * Define 2.0 feature version as it is needed below to decide whether we
   * compile for both 1.x and 2.x (defining it guarantees 1.x only).
   */
  #define NPY_2_0_API_VERSION 0x00000012
  /*
   * If we are compiling with NumPy 1.x, PyArray_RUNTIME_VERSION so we
   * pretend the `PyArray_RUNTIME_VERSION` is `NPY_FEATURE_VERSION`.
   * This allows downstream to use `PyArray_RUNTIME_VERSION` if they need to.
   */
  #define PyArray_RUNTIME_VERSION NPY_FEATURE_VERSION
  /* Compiling on NumPy 1.x where these are the same: */
  #define PyArray_DescrProto PyArray_Descr
#endif

/*
 * Define a better way to call `_import_array()` to simplify backporting as
 * we now require imports more often (necessary to make ABI flexible).
 */
#ifdef import_array1

static inline int
PyArray_ImportNumPyAPI()
{
    if (NPY_UNLIKELY(PyArray_API == NULL)) {
        import_array1(-1);
    }
    return 0;
}

#endif  /* import_array1 */

#endif  /* NUMPY_CORE_INCLUDE_NUMPY_NPY_2_COMPAT_H_ */
/*
 * NPY_DEFAULT_INT
 *
 * 默认整数类型，根据 NumPy 的版本不同有不同的定义。在运行时可以用作类型编号，
 * 例如 `PyArray_DescrFromType(NPY_DEFAULT_INT)`。
 *
 * NPY_RAVEL_AXIS
 *
 * 在 NumPy 2.0 中引入，用于指示在操作中应该展平（ravel）的轴。在 NumPy 2.0 之前，
 * 使用 NPY_MAXDIMS 来表示相同的含义。
 *
 * NPY_MAXDIMS
 *
 * 表示创建 ndarray 时允许的最大维度数。
 *
 * NPY_NTYPES_LEGACY
 *
 * NumPy 内置数据类型的数量，这是一个遗留的定义。
 */
#if NPY_FEATURE_VERSION >= NPY_2_0_API_VERSION
    #define NPY_DEFAULT_INT NPY_INTP
    #define NPY_RAVEL_AXIS NPY_MIN_INT
    #define NPY_MAXARGS 64

#elif NPY_ABI_VERSION < 0x02000000
    #define NPY_DEFAULT_INT NPY_LONG
    #define NPY_RAVEL_AXIS 32
    #define NPY_MAXARGS 32

    /* NumPy 1.x 兼容的别名 */
    #define NPY_NTYPES NPY_NTYPES_LEGACY
    #define PyArray_DescrProto PyArray_Descr
    #define _PyArray_LegacyDescr PyArray_Descr
    /* NumPy 2 定义适用于 NumPy 1.x */
    #define PyDataType_ISLEGACY(dtype) (1)
#else
    #define NPY_DEFAULT_INT  \
        (PyArray_RUNTIME_VERSION >= NPY_2_0_API_VERSION ? NPY_INTP : NPY_LONG)
    #define NPY_RAVEL_AXIS  \
        (PyArray_RUNTIME_VERSION >= NPY_2_0_API_VERSION ? -1 : 32)
    #define NPY_MAXARGS  \
        (PyArray_RUNTIME_VERSION >= NPY_2_0_API_VERSION ? 64 : 32)
#endif


/*
 * Access inline functions for descriptor fields.  Except for the first
 * few fields, these needed to be moved (elsize, alignment) for
 * additional space.  Or they are descriptor specific and are not generally
 * available anymore (metadata, c_metadata, subarray, names, fields).
 *
 * Most of these are defined via the `DESCR_ACCESSOR` macro helper.
 */
#if NPY_FEATURE_VERSION >= NPY_2_0_API_VERSION || NPY_ABI_VERSION < 0x02000000
    /* 编译环境为 1.x 或 2.x，直接访问字段是允许的 */

    static inline void
    PyDataType_SET_ELSIZE(PyArray_Descr *dtype, npy_intp size)
    {
        // 设置数据类型描述符的元素大小
        dtype->elsize = size;
    }

    static inline npy_uint64
    PyDataType_FLAGS(const PyArray_Descr *dtype)
    {
    #if NPY_FEATURE_VERSION >= NPY_2_0_API_VERSION
        // 返回数据类型描述符的标志位
        return dtype->flags;
    #else
        // 在 1.x 版本中，需要将标志位转换为无符号字符型
        return (unsigned char)dtype->flags;
    #endif
    }

    #define DESCR_ACCESSOR(FIELD, field, type, legacy_only)    \
        static inline type                                     \
        PyDataType_##FIELD(const PyArray_Descr *dtype) {       \
            if (legacy_only && !PyDataType_ISLEGACY(dtype)) {  \
                return (type)0;                                \
            }                                                  \
            // 返回数据类型描述符中指定字段的值
            return ((_PyArray_LegacyDescr *)dtype)->field;     \
        }
#else  /* 编译环境同时支持 1.x 和 2.x */

    static inline void
    PyDataType_SET_ELSIZE(PyArray_Descr *dtype, npy_intp size)
    {
        // 根据运行时版本决定设置 dtype 的元素大小
        if (PyArray_RUNTIME_VERSION >= NPY_2_0_API_VERSION) {
            ((_PyArray_DescrNumPy2 *)dtype)->elsize = size;
        }
        else {
            ((PyArray_DescrProto *)dtype)->elsize = (int)size;
        }
    }
    
    static inline npy_uint64
    PyDataType_FLAGS(const PyArray_Descr *dtype)
    {
        // 根据运行时版本返回 dtype 的 flags 属性
        if (PyArray_RUNTIME_VERSION >= NPY_2_0_API_VERSION) {
            return ((_PyArray_DescrNumPy2 *)dtype)->flags;
        }
        else {
            return (unsigned char)((PyArray_DescrProto *)dtype)->flags;
        }
    }
    
    /* Cast to LegacyDescr always fine but needed when `legacy_only` */
    #define DESCR_ACCESSOR(FIELD, field, type, legacy_only)        \
        static inline type                                         \
        PyDataType_##FIELD(const PyArray_Descr *dtype) {           \
            // 如果 legacy_only 为真且 dtype 不是遗留描述符，则返回默认值
            if (legacy_only && !PyDataType_ISLEGACY(dtype)) {      \
                return (type)0;
            }
            // 根据运行时版本返回 dtype 的指定字段值
            if (PyArray_RUNTIME_VERSION >= NPY_2_0_API_VERSION) {  \
                return ((_PyArray_LegacyDescr *)dtype)->field;     \
            }                                                      \
            else {                                                 \
                return ((PyArray_DescrProto *)dtype)->field;       \
            }                                                      \
        }
#endif

DESCR_ACCESSOR(ELSIZE, elsize, npy_intp, 0)
// 定义描述符访问器，访问元素大小的描述符

DESCR_ACCESSOR(ALIGNMENT, alignment, npy_intp, 0)
// 定义描述符访问器，访问对齐方式的描述符

DESCR_ACCESSOR(METADATA, metadata, PyObject *, 1)
// 定义描述符访问器，访问元数据的描述符，数据类型为 PyObject*

DESCR_ACCESSOR(SUBARRAY, subarray, PyArray_ArrayDescr *, 1)
// 定义描述符访问器，访问子数组的描述符，数据类型为 PyArray_ArrayDescr*

DESCR_ACCESSOR(NAMES, names, PyObject *, 1)
// 定义描述符访问器，访问字段名称的描述符，数据类型为 PyObject*

DESCR_ACCESSOR(FIELDS, fields, PyObject *, 1)
// 定义描述符访问器，访问字段的描述符，数据类型为 PyObject*

DESCR_ACCESSOR(C_METADATA, c_metadata, NpyAuxData *, 1)
// 定义描述符访问器，访问 C 语言元数据的描述符，数据类型为 NpyAuxData*

#undef DESCR_ACCESSOR

#if !(defined(NPY_INTERNAL_BUILD) && NPY_INTERNAL_BUILD)
#if NPY_FEATURE_VERSION >= NPY_2_0_API_VERSION
    static inline PyArray_ArrFuncs *
    PyDataType_GetArrFuncs(const PyArray_Descr *descr)
    {
        return _PyDataType_GetArrFuncs(descr);
    }
#elif NPY_ABI_VERSION < 0x02000000
    static inline PyArray_ArrFuncs *
    PyDataType_GetArrFuncs(const PyArray_Descr *descr)
    {
        return descr->f;
    }
#else
    static inline PyArray_ArrFuncs *
    PyDataType_GetArrFuncs(const PyArray_Descr *descr)
    {
        if (PyArray_RUNTIME_VERSION >= NPY_2_0_API_VERSION) {
            return _PyDataType_GetArrFuncs(descr);
        }
        else {
            return ((PyArray_DescrProto *)descr)->f;
        }
    }
#endif

#endif  /* not internal build */

#endif  /* NUMPY_CORE_INCLUDE_NUMPY_NPY_2_COMPAT_H_ */
```