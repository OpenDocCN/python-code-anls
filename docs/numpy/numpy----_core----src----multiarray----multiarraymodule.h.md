# `.\numpy\numpy\_core\src\multiarray\multiarraymodule.h`

```py
#ifndef NUMPY_CORE_SRC_MULTIARRAY_MULTIARRAYMODULE_H_
#define NUMPY_CORE_SRC_MULTIARRAY_MULTIARRAYMODULE_H_

/*
 * A struct storing thread-unsafe global state for the _multiarray_umath
 * module. We should refactor so the global state is thread-safe,
 * e.g. by adding locking.
 */
// 定义一个结构体，用于存储 _multiarray_umath 模块的线程不安全的全局状态
typedef struct npy_thread_unsafe_state_struct {
    /*
     * Cached references to objects obtained via an import. All of these are
     * can be initialized at any time by npy_cache_import.
     *
     * Currently these are not initialized in a thread-safe manner but the
     * failure mode is a reference leak for references to imported immortal
     * modules so it will never lead to a crash unless users are doing something
     * janky that we don't support like reloading.
     *
     * TODO: maybe make each entry a struct that looks like:
     *
     *      struct {
     *          atomic_int initialized;
     *          PyObject *value;
     *      }
     *
     * so the initialization is thread-safe and the only possibile lock
     * contention happens before the cache is initialized, not on every single
     * read.
     */
    // 下面的字段是通过导入获取的对象的缓存引用，这些引用可以随时由 npy_cache_import 初始化
    // 当前这些字段的初始化方式不是线程安全的，但失败模式是对导入的不可销毁模块的引用泄漏，
    // 因此不会导致崩溃，除非用户在重新加载等不支持的情况下做了一些奇怪的事情。
    // TODO: 可以考虑将每个条目改造成像这样的结构体：
    //      struct {
    //          atomic_int initialized;
    //          PyObject *value;
    //      }
    // 这样初始化将是线程安全的，唯一可能的锁争用发生在缓存初始化之前，而不是在每次读取时。
    PyObject *_add_dtype_helper;
    PyObject *_all;
    PyObject *_amax;
    PyObject *_amin;
    PyObject *_any;
    PyObject *array_function_errmsg_formatter;
    PyObject *array_ufunc_errmsg_formatter;
    PyObject *_clip;
    PyObject *_commastring;
    PyObject *_convert_to_stringdtype_kwargs;
    PyObject *_default_array_repr;
    PyObject *_default_array_str;
    PyObject *_dump;
    PyObject *_dumps;
    PyObject *_getfield_is_safe;
    PyObject *internal_gcd_func;
    PyObject *_mean;
    PyObject *NO_NEP50_WARNING;
    PyObject *npy_ctypes_check;
    PyObject *numpy_matrix;
    PyObject *_prod;
    PyObject *_promote_fields;
    PyObject *_std;
    PyObject *_sum;
    PyObject *_ufunc_doc_signature_formatter;
    PyObject *_var;
    PyObject *_view_is_safe;
    PyObject *_void_scalar_to_string;

    /*
     * Used to test the internal-only scaled float test dtype
     */
    // 用于测试仅内部使用的缩放浮点测试数据类型
    npy_bool get_sfloat_dtype_initialized;

    /*
     * controls the global madvise hugepage setting
     */
    // 控制全局 madvise 巨页设置
    int madvise_hugepage;

    /*
     * used to detect module reloading in the reload guard
     */
    // 用于在重新加载保护中检测模块重新加载
    int reload_guard_initialized;

     /*
      * global variable to determine if legacy printing is enabled,
      * accessible from C. For simplicity the mode is encoded as an
      * integer where INT_MAX means no legacy mode, and '113'/'121'
      * means 1.13/1.21 legacy mode; and 0 maps to INT_MAX. We can
      * upgrade this if we have more complex requirements in the future.
      */
    // 全局变量，用于确定是否启用传统打印，可从 C 中访问。
    // 简单起见，模式被编码为一个整数，其中 INT_MAX 表示没有传统模式，
    // '113'/'121' 表示 1.13/1.21 传统模式；而 0 映射到 INT_MAX。
    // 如果将来有更复杂的要求，可以升级这个设置。
    int legacy_print_mode;

    /*
     * Holds the user-defined setting for whether or not to warn
     * if there is no memory policy set
     */
    // 存储用户定义的设置，用于确定是否在没有设置内存策略时发出警告
    int warn_if_no_mem_policy;

} npy_thread_unsafe_state_struct;

// 声明一个外部可见性为隐藏的 npy_thread_unsafe_state_struct 类型的变量
NPY_VISIBILITY_HIDDEN extern npy_thread_unsafe_state_struct npy_thread_unsafe_state;

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_MULTIARRAYMODULE_H_ */
```