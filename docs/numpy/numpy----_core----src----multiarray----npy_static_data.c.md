# `.\numpy\numpy\_core\src\multiarray\npy_static_data.c`

```
/* numpy static data structs and initialization */

/* Define NPY_NO_DEPRECATED_API to ensure we use the latest API version */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
/* Define _UMATHMODULE and _MULTIARRAYMODULE to include corresponding modules */
#define _UMATHMODULE
#define _MULTIARRAYMODULE

/* Ensure PY_SSIZE_T_CLEAN is defined to use the new ssize_t API */
#define PY_SSIZE_T_CLEAN
/* Include Python.h to access Python/C API */
#include <Python.h>
/* Include structmember.h for Python object structure handling */
#include <structmember.h>

/* Include numpy C API headers */
#include "numpy/ndarraytypes.h"
#include "numpy/npy_common.h"
#include "numpy/arrayobject.h"
#include "npy_import.h"
#include "npy_static_data.h"
#include "extobj.h"

/* Definition of static variables */
/* NPY_VISIBILITY_HIDDEN indicates these are not exposed externally */
NPY_VISIBILITY_HIDDEN npy_interned_str_struct npy_interned_str;
NPY_VISIBILITY_HIDDEN npy_static_pydata_struct npy_static_pydata;
NPY_VISIBILITY_HIDDEN npy_static_cdata_struct npy_static_cdata;

/* Macro to intern a string into a Python Unicode object and assign to a struct member */
#define INTERN_STRING(struct_member, string)                             \
    /* Assertion to ensure the struct member is initially NULL */        \
    assert(npy_interned_str.struct_member == NULL);                      \
    /* Intern the string literal into a Python Unicode object */         \
    npy_interned_str.struct_member = PyUnicode_InternFromString(string); \
    /* Check if the intern operation succeeded */                        \
    if (npy_interned_str.struct_member == NULL) {                        \
        return -1;                                                       \
    }                                                                    \

/* Function to intern a list of predefined strings */
NPY_NO_EXPORT int
intern_strings(void)
{
    /* Macro calls to intern each string into corresponding struct members */
    INTERN_STRING(current_allocator, "current_allocator");
    INTERN_STRING(array, "__array__");
    INTERN_STRING(array_function, "__array_function__");
    INTERN_STRING(array_struct, "__array_struct__");
    INTERN_STRING(array_priority, "__array_priority__");
    INTERN_STRING(array_interface, "__array_interface__");
    INTERN_STRING(array_ufunc, "__array_ufunc__");
    INTERN_STRING(array_wrap, "__array_wrap__");
    INTERN_STRING(array_finalize, "__array_finalize__");
    INTERN_STRING(implementation, "_implementation");
    INTERN_STRING(axis1, "axis1");
    INTERN_STRING(axis2, "axis2");
    INTERN_STRING(like, "like");
    INTERN_STRING(numpy, "numpy");
    INTERN_STRING(where, "where");
    INTERN_STRING(convert, "convert");
    INTERN_STRING(preserve, "preserve");
    INTERN_STRING(convert_if_no_array, "convert_if_no_array");
    INTERN_STRING(cpu, "cpu");
    INTERN_STRING(dtype, "dtype");
    INTERN_STRING(
            array_err_msg_substr,
            "__array__() got an unexpected keyword argument 'copy'");
    INTERN_STRING(out, "out");
    INTERN_STRING(errmode_strings[0], "ignore");
    INTERN_STRING(errmode_strings[1], "warn");
    INTERN_STRING(errmode_strings[2], "raise");
    INTERN_STRING(errmode_strings[3], "call");
    INTERN_STRING(errmode_strings[4], "print");
    INTERN_STRING(errmode_strings[5], "log");
    INTERN_STRING(__dlpack__, "__dlpack__");
    INTERN_STRING(pyvals_name, "UFUNC_PYVALS_NAME");
    /* Return success */
    return 0;
}

/* Macro to import a global object from a module and assert its existence */
#define IMPORT_GLOBAL(base_path, name, object)  \
    /* Assertion to ensure the object is initially NULL */                     \
    assert(object == NULL);                     \
    /* Attempt to import the object from the module using npy_cache_import */   \
    npy_cache_import(base_path, name, &object); \
    /* Check if the import succeeded */         \
    if (object == NULL) {                       \
        return -1;                              \
    }
/*
 * Initializes global constants.
 *
 * All global constants should live inside the npy_static_pydata
 * struct.
 *
 * Not all entries in the struct are initialized here, some are
 * initialized later but care must be taken in those cases to initialize
 * the constant in a thread-safe manner, ensuring it is initialized
 * exactly once.
 *
 * Anything initialized here is initialized during module import which
 * the python interpreter ensures is done in a single thread.
 *
 * Anything imported here should not need the C-layer at all and will be
 * imported before anything on the C-side is initialized.
 */
NPY_NO_EXPORT int
initialize_static_globals(void)
{
    /*
     * Initialize contents of npy_static_pydata struct
     *
     * This struct holds cached references to python objects
     * that we want to keep alive for the lifetime of the
     * module for performance reasons
     */

    // 导入全局变量 "math" 模块中的 "floor" 函数，并存储在 npy_static_pydata.math_floor_func 中
    IMPORT_GLOBAL("math", "floor",
                  npy_static_pydata.math_floor_func);

    // 导入全局变量 "math" 模块中的 "ceil" 函数，并存储在 npy_static_pydata.math_ceil_func 中
    IMPORT_GLOBAL("math", "ceil",
                  npy_static_pydata.math_ceil_func);

    // 导入全局变量 "math" 模块中的 "trunc" 函数，并存储在 npy_static_pydata.math_trunc_func 中
    IMPORT_GLOBAL("math", "trunc",
                  npy_static_pydata.math_trunc_func);

    // 导入全局变量 "math" 模块中的 "gcd" 函数，并存储在 npy_static_pydata.math_gcd_func 中
    IMPORT_GLOBAL("math", "gcd",
                  npy_static_pydata.math_gcd_func);

    // 导入全局变量 "numpy.exceptions" 模块中的 "AxisError" 异常，并存储在 npy_static_pydata.AxisError 中
    IMPORT_GLOBAL("numpy.exceptions", "AxisError",
                  npy_static_pydata.AxisError);

    // 导入全局变量 "numpy.exceptions" 模块中的 "ComplexWarning" 异常，并存储在 npy_static_pydata.ComplexWarning 中
    IMPORT_GLOBAL("numpy.exceptions", "ComplexWarning",
                  npy_static_pydata.ComplexWarning);

    // 导入全局变量 "numpy.exceptions" 模块中的 "DTypePromotionError" 异常，并存储在 npy_static_pydata.DTypePromotionError 中
    IMPORT_GLOBAL("numpy.exceptions", "DTypePromotionError",
                  npy_static_pydata.DTypePromotionError);

    // 导入全局变量 "numpy.exceptions" 模块中的 "TooHardError" 异常，并存储在 npy_static_pydata.TooHardError 中
    IMPORT_GLOBAL("numpy.exceptions", "TooHardError",
                  npy_static_pydata.TooHardError);

    // 导入全局变量 "numpy.exceptions" 模块中的 "VisibleDeprecationWarning" 异常，并存储在 npy_static_pydata.VisibleDeprecationWarning 中
    IMPORT_GLOBAL("numpy.exceptions", "VisibleDeprecationWarning",
                  npy_static_pydata.VisibleDeprecationWarning);

    // 导入全局变量 "numpy._globals" 模块中的 "_CopyMode" 对象，并存储在 npy_static_pydata._CopyMode 中
    IMPORT_GLOBAL("numpy._globals", "_CopyMode",
                  npy_static_pydata._CopyMode);

    // 导入全局变量 "numpy._globals" 模块中的 "_NoValue" 对象，并存储在 npy_static_pydata._NoValue 中
    IMPORT_GLOBAL("numpy._globals", "_NoValue",
                  npy_static_pydata._NoValue);

    // 导入全局变量 "numpy._core._exceptions" 模块中的 "_ArrayMemoryError" 异常，并存储在 npy_static_pydata._ArrayMemoryError 中
    IMPORT_GLOBAL("numpy._core._exceptions", "_ArrayMemoryError",
                  npy_static_pydata._ArrayMemoryError);

    // 导入全局变量 "numpy._core._exceptions" 模块中的 "_UFuncBinaryResolutionError" 异常，并存储在 npy_static_pydata._UFuncBinaryResolutionError 中
    IMPORT_GLOBAL("numpy._core._exceptions", "_UFuncBinaryResolutionError",
                  npy_static_pydata._UFuncBinaryResolutionError);

    // 导入全局变量 "numpy._core._exceptions" 模块中的 "_UFuncInputCastingError" 异常，并存储在 npy_static_pydata._UFuncInputCastingError 中
    IMPORT_GLOBAL("numpy._core._exceptions", "_UFuncInputCastingError",
                  npy_static_pydata._UFuncInputCastingError);

    // 导入全局变量 "numpy._core._exceptions" 模块中的 "_UFuncNoLoopError" 异常，并存储在 npy_static_pydata._UFuncNoLoopError 中
    IMPORT_GLOBAL("numpy._core._exceptions", "_UFuncNoLoopError",
                  npy_static_pydata._UFuncNoLoopError);

    // 导入全局变量 "numpy._core._exceptions" 模块中的 "_UFuncOutputCastingError" 异常，并存储在 npy_static_pydata._UFuncOutputCastingError 中
    IMPORT_GLOBAL("numpy._core._exceptions", "_UFuncOutputCastingError",
                  npy_static_pydata._UFuncOutputCastingError);

    // 导入全局变量 "os" 模块中的 "fspath" 函数，并存储在 npy_static_pydata.os_fspath 中
    IMPORT_GLOBAL("os", "fspath",
                  npy_static_pydata.os_fspath);

    // 导入全局变量 "os" 模块中的 "PathLike" 对象，并存储在 npy_static_pydata.os_PathLike 中
    IMPORT_GLOBAL("os", "PathLike",
                  npy_static_pydata.os_PathLike);

    // 创建 PyArray_Descr 结构，用于表示 NPY_DOUBLE 类型的数组描述符，赋值给 tmp 变量
    PyArray_Descr *tmp = PyArray_DescrFromType(NPY_DOUBLE);
    npy_static_pydata.default_truediv_type_tup =
            PyTuple_Pack(3, tmp, tmp, tmp);
    // 创建一个包含三个相同对象 tmp 的元组，并赋值给全局变量 default_truediv_type_tup
    Py_DECREF(tmp);
    // 减少 tmp 的引用计数，防止内存泄漏
    if (npy_static_pydata.default_truediv_type_tup == NULL) {
        return -1;
    }
    // 检查元组创建是否成功，若失败则返回 -1

    npy_static_pydata.kwnames_is_copy = Py_BuildValue("(s)", "copy");
    // 创建一个包含字符串 "copy" 的元组，并赋值给全局变量 kwnames_is_copy
    if (npy_static_pydata.kwnames_is_copy == NULL) {
        return -1;
    }
    // 检查元组创建是否成功，若失败则返回 -1

    npy_static_pydata.one_obj = PyLong_FromLong((long) 1);
    // 创建一个包含整数 1 的 PyLong 对象，并赋值给全局变量 one_obj
    if (npy_static_pydata.one_obj == NULL) {
        return -1;
    }
    // 检查对象创建是否成功，若失败则返回 -1

    npy_static_pydata.zero_obj = PyLong_FromLong((long) 0);
    // 创建一个包含整数 0 的 PyLong 对象，并赋值给全局变量 zero_obj
    if (npy_static_pydata.zero_obj == NULL) {
        return -1;
    }
    // 检查对象创建是否成功，若失败则返回 -1

    /*
     * Initialize contents of npy_static_cdata struct
     *
     * Note that some entries are initialized elsewhere. Care
     * must be taken to ensure all entries are initialized during
     * module initialization and immutable thereafter.
     *
     * This struct holds global static caches. These are set
     * up this way for performance reasons.
     */

    PyObject *flags = PySys_GetObject("flags");  /* borrowed object */
    // 获取名为 "flags" 的系统模块对象，并赋值给 flags（借用引用）
    if (flags == NULL) {
        PyErr_SetString(PyExc_AttributeError, "cannot get sys.flags");
        return -1;
    }
    // 检查获取是否成功，若失败则设置异常并返回 -1
    PyObject *level = PyObject_GetAttrString(flags, "optimize");
    // 获取 flags 对象的名为 "optimize" 的属性，并赋值给 level
    if (level == NULL) {
        return -1;
    }
    // 检查获取是否成功，若失败则返回 -1
    npy_static_cdata.optimize = PyLong_AsLong(level);
    // 将 level 转换为长整型并赋值给全局变量 optimize
    Py_DECREF(level);
    // 减少 level 的引用计数，防止内存泄漏

    /*
     * see unpack_bits for how this table is used.
     *
     * LUT for bigendian bitorder, littleendian is handled via
     * byteswapping in the loop.
     *
     * 256 8 byte blocks representing 8 bits expanded to 1 or 0 bytes
     */
    npy_intp j;
    // 循环变量 j 的声明
    for (j=0; j < 256; j++) {
        npy_intp k;
        // 循环变量 k 的声明
        for (k=0; k < 8; k++) {
            npy_uint8 v = (j & (1 << k)) == (1 << k);
            // 计算当前位的值（0或1），并赋值给 v
            npy_static_cdata.unpack_lookup_big[j].bytes[7 - k] = v;
            // 将 v 存储在 unpack_lookup_big[j] 的相应位置
        }
    }

    return 0;
    // 函数正常执行完毕，返回 0 表示成功
/*
 * Verifies all entries in npy_interned_str and npy_static_pydata are
 * non-NULL.
 *
 * Called at the end of initialization for _multiarray_umath. Some
 * entries are initialized outside of this file because they depend on
 * items that are initialized late in module initialization but they
 * should all be initialized by the time this function is called.
 */
NPY_NO_EXPORT int
verify_static_structs_initialized(void) {
    // verify all entries in npy_interned_str are filled in
    for (int i=0; i < (sizeof(npy_interned_str_struct)/sizeof(PyObject *)); i++) {
        // Check if the i-th entry in npy_interned_str is NULL
        if (*(((PyObject **)&npy_interned_str) + i) == NULL) {
            // Raise a SystemError if a NULL entry is found
            PyErr_Format(
                    PyExc_SystemError,
                    "NumPy internal error: NULL entry detected in "
                    "npy_interned_str at index %d", i);
            // Return -1 indicating error
            return -1;
        }
    }

    // verify all entries in npy_static_pydata are filled in
    for (int i=0; i < (sizeof(npy_static_pydata_struct)/sizeof(PyObject *)); i++) {
        // Check if the i-th entry in npy_static_pydata is NULL
        if (*(((PyObject **)&npy_static_pydata) + i) == NULL) {
            // Raise a SystemError if a NULL entry is found
            PyErr_Format(
                    PyExc_SystemError,
                    "NumPy internal error: NULL entry detected in "
                    "npy_static_pydata at index %d", i);
            // Return -1 indicating error
            return -1;
        }
    }
    // All entries are initialized correctly, return 0 indicating success
    return 0;
}
```