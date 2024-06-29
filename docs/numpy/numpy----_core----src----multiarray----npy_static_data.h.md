# `.\numpy\numpy\_core\src\multiarray\npy_static_data.h`

```
#ifndef NUMPY_CORE_SRC_MULTIARRAY_STATIC_DATA_H_
#define NUMPY_CORE_SRC_MULTIARRAY_STATIC_DATA_H_

// 定义了头文件的宏保护，防止多重包含

NPY_NO_EXPORT int
initialize_static_globals(void);

// 声明了初始化静态全局变量的函数

NPY_NO_EXPORT int
intern_strings(void);

// 声明了内部字符串初始化函数

NPY_NO_EXPORT int
verify_static_structs_initialized(void);

// 声明了验证静态结构体是否初始化完成的函数

typedef struct npy_interned_str_struct {
    // 定义了结构体 npy_interned_str_struct，用于存储内部字符串对象
    PyObject *current_allocator;  // 当前分配器
    PyObject *array;  // 数组
    PyObject *array_function;  // 数组函数
    PyObject *array_struct;  // 数组结构
    PyObject *array_priority;  // 数组优先级
    PyObject *array_interface;  // 数组接口
    PyObject *array_wrap;  // 数组包装
    PyObject *array_finalize;  // 数组终结器
    PyObject *array_ufunc;  // 数组 ufunc
    PyObject *implementation;  // 实现
    PyObject *axis1;  // 轴1
    PyObject *axis2;  // 轴2
    PyObject *like;  // 类似
    PyObject *numpy;  // NumPy
    PyObject *where;  // where 函数
    PyObject *convert;  // 转换
    PyObject *preserve;  // 保持
    PyObject *convert_if_no_array;  // 如果无数组则转换
    PyObject *cpu;  // CPU
    PyObject *dtype;  // 数据类型
    PyObject *array_err_msg_substr;  // 数组错误消息子字符串
    PyObject *out;  // 输出
    PyObject *errmode_strings[6];  // 错误模式字符串数组
    PyObject *__dlpack__;  // __dlpack__ 对象
    PyObject *pyvals_name;  // Python 值名称
} npy_interned_str_struct;

/*
 * A struct that stores static global data used throughout
 * _multiarray_umath, mostly to cache results that would be
 * prohibitively expensive to compute at runtime in a tight loop.
 *
 * All items in this struct should be initialized during module
 * initialization and thereafter should be immutable. Mutating items in
 * this struct after module initialization is likely not thread-safe.
 */

// 定义了一个结构体 npy_static_pydata_struct，用于存储在 _multiarray_umath 中使用的静态全局数据
typedef struct npy_static_pydata_struct {
    /*
     * Used in ufunc_type_resolution.c to avoid reconstructing a tuple
     * storing the default true division return types.
     */
    PyObject *default_truediv_type_tup;  // 用于存储默认真除法返回类型的元组，避免重复构建

    /*
     * Used to set up the default extobj context variable
     */
    PyObject *default_extobj_capsule;  // 用于设置默认的 extobj 上下文变量

    /*
     * The global ContextVar to store the extobject. It is exposed to Python
     * as `_extobj_contextvar`.
     */
    PyObject *npy_extobj_contextvar;  // 用于存储 extobject 的全局 ContextVar，作为 `_extobj_contextvar` 暴露给 Python 使用

    /*
     * A reference to ndarray's implementations for __array_*__ special methods
     */
    PyObject *ndarray_array_ufunc;  // ndarray 的实现，用于 __array_*__ 特殊方法

    PyObject *ndarray_array_finalize;  // ndarray 的 array_finalize 方法
    PyObject *ndarray_array_function;  // ndarray 的 array_function 方法

    /*
     * References to the '1' and '0' PyLong objects
     */
    PyObject *one_obj;  // '1' 的 PyLong 对象引用
    PyObject *zero_obj;  // '0' 的 PyLong 对象引用

    /*
     * Reference to an np.array(0, dtype=np.long) instance
     */
    PyObject *zero_pyint_like_arr;  // np.array(0, dtype=np.long) 实例的引用

    /*
     * References to items obtained via an import at module initialization
     */
    PyObject *AxisError;  // AxisError 对象引用
    PyObject *ComplexWarning;  // ComplexWarning 对象引用
    PyObject *DTypePromotionError;  // DTypePromotionError 对象引用
    PyObject *TooHardError;  // TooHardError 对象引用
    PyObject *VisibleDeprecationWarning;  // VisibleDeprecationWarning 对象引用
    PyObject *_CopyMode;  // _CopyMode 对象引用
    PyObject *_NoValue;  // _NoValue 对象引用
    PyObject *_ArrayMemoryError;  // _ArrayMemoryError 对象引用
    PyObject *_UFuncBinaryResolutionError;  // _UFuncBinaryResolutionError 对象引用
    PyObject *_UFuncInputCastingError;  // _UFuncInputCastingError 对象引用
    PyObject *_UFuncNoLoopError;  // _UFuncNoLoopError 对象引用
    PyObject *_UFuncOutputCastingError;  // _UFuncOutputCastingError 对象引用
    PyObject *math_floor_func;  // math_floor_func 函数引用
    PyObject *math_ceil_func;  // math_ceil_func 函数引用
    PyObject *math_trunc_func;  // math_trunc_func 函数引用
    PyObject *math_gcd_func;  // math_gcd_func 函数引用
    PyObject *os_PathLike;  // os_PathLike 对象引用
    PyObject *os_fspath;  // os_fspath 对象引用
    /*
     * 在 __array__ 的内部使用，避免在内联构建元组
     */
    PyObject *kwnames_is_copy;
    
    /*
     * 在 __imatmul__ 中使用，避免在内联构建元组
     */
    PyObject *axes_1d_obj_kwargs;
    PyObject *axes_2d_obj_kwargs;
    
    /*
     * 用于 CPU 特性检测和调度
     */
    PyObject *cpu_dispatch_registry;
    
    /*
     * 引用了缓存的 ArrayMethod 实现，避免重复创建它们
     */
    PyObject *VoidToGenericMethod;
    PyObject *GenericToVoidMethod;
    PyObject *ObjectToGenericMethod;
    PyObject *GenericToObjectMethod;
/*
 * 结构体定义：npy_static_pydata_struct，用于存储静态的 Python 数据结构
 */
typedef struct npy_static_pydata_struct {
    /*
     * 存储 sys.flags.optimize 的长整型值，用于在 add_docstring 实现中使用
     */
    long optimize;

    /*
     * 用于 unpack_bits 的查找表
     */
    union {
        npy_uint8  bytes[8];
        npy_uint64 uint64;
    } unpack_lookup_big[256];

    /*
     * 从类型字符中恢复整数类型编号的查找表。
     *
     * 参见 arraytypes.c.src 中的 _MAX_LETTER 和 LETTER_TO_NUM 宏。
     *
     * 最小的类型编号是 '?'，最大的受到 'z' 限制。
     *
     * 这与内置的 dtypes 同时初始化。
     */
    npy_int16 _letter_to_num['z' + 1 - '?'];
} npy_static_cdata_struct;

/*
 * NPY_VISIBILITY_HIDDEN：用于声明不对外部可见的符号
 * 声明一个对外部不可见的字符串结构 npy_interned_str
 */
NPY_VISIBILITY_HIDDEN extern npy_interned_str_struct npy_interned_str;

/*
 * NPY_VISIBILITY_HIDDEN：用于声明不对外部可见的符号
 * 声明一个对外部不可见的静态 Python 数据结构 npy_static_pydata
 */
NPY_VISIBILITY_HIDDEN extern npy_static_pydata_struct npy_static_pydata;

/*
 * NPY_VISIBILITY_HIDDEN：用于声明不对外部可见的符号
 * 声明一个对外部不可见的静态 C 数据结构 npy_static_cdata
 */
NPY_VISIBILITY_HIDDEN extern npy_static_cdata_struct npy_static_cdata;

#endif  // NUMPY_CORE_SRC_MULTIARRAY_STATIC_DATA_H_
```