# `.\numpy\numpy\_core\src\common\npy_cpu_dispatch_distutils.h`

```
#ifndef NUMPY_CORE_SRC_COMMON_NPY_CPU_DISPATCH_DISTUTILS_H_
#define NUMPY_CORE_SRC_COMMON_NPY_CPU_DISPATCH_DISTUTILS_H_
#ifndef NUMPY_CORE_SRC_COMMON_NPY_CPU_DISPATCH_H_
    #error "Not standalone header please use 'npy_cpu_dispatch.h'"
#endif

/**
 * This header should be removed after support for distutils is removed.
 * It provides helper macros required for CPU runtime dispatching,
 * which are already defined within `meson_cpu/main_config.h.in`.
 *
 * The following macros are explained within `meson_cpu/main_config.h.in`,
 * although there are some differences in their usage:
 *
 * - Dispatched targets must be defined at the top of each dispatch-able
 *   source file within an inline or multi-line comment block.
 *   For example: //@targets baseline SSE2 AVX2 AVX512_SKX
 *
 * - The generated configuration derived from each dispatch-able source
 *   file must be guarded with `#ifndef NPY_DISABLE_OPTIMIZATION`.
 *   For example:
 *   #ifndef NPY_DISABLE_OPTIMIZATION
 *      #include "arithmetic.dispatch.h"
 *   #endif
 */

#include "npy_cpu_features.h" // NPY_CPU_HAVE
#include "numpy/utils.h" // NPY_EXPAND, NPY_CAT

#ifdef NPY__CPU_TARGET_CURRENT
    // 'NPY__CPU_TARGET_CURRENT': only defined by the dispatch-able sources
    #define NPY_CPU_DISPATCH_CURFX(NAME) NPY_CAT(NPY_CAT(NAME, _), NPY__CPU_TARGET_CURRENT)
#else
    #define NPY_CPU_DISPATCH_CURFX(NAME) NPY_EXPAND(NAME)
#endif

/**
 * Defining the default behavior for the configurable macros of dispatch-able sources,
 * 'NPY__CPU_DISPATCH_CALL(...)' and 'NPY__CPU_DISPATCH_BASELINE_CALL(...)'
 *
 * These macros are defined inside the generated config files that have been derived from
 * the configuration statements of the dispatch-able sources.
 *
 * The generated config file takes the same name of the dispatch-able source with replacing
 * the extension to '.h' instead of '.c', and it should be treated as a header template.
 */
#ifndef NPY_DISABLE_OPTIMIZATION
    #define NPY__CPU_DISPATCH_BASELINE_CALL(CB, ...) \
        &&"Expected config header of the dispatch-able source";
    #define NPY__CPU_DISPATCH_CALL(CHK, CB, ...) \
        &&"Expected config header of the dispatch-able source";
#else
    /**
     * We assume by default that all configuration statements contain 'baseline' option, however,
     * if the dispatch-able source doesn't require it, then the dispatch-able source and following macros
     * need to be guarded with '#ifndef NPY_DISABLE_OPTIMIZATION'
     */
    #define NPY__CPU_DISPATCH_BASELINE_CALL(CB, ...) \
        NPY_EXPAND(CB(__VA_ARGS__))
    #define NPY__CPU_DISPATCH_CALL(CHK, CB, ...)
#endif // !NPY_DISABLE_OPTIMIZATION

#define NPY_CPU_DISPATCH_DECLARE(...) \
    NPY__CPU_DISPATCH_CALL(NPY_CPU_DISPATCH_DECLARE_CHK_, NPY_CPU_DISPATCH_DECLARE_CB_, __VA_ARGS__) \
    NPY__CPU_DISPATCH_BASELINE_CALL(NPY_CPU_DISPATCH_DECLARE_BASE_CB_, __VA_ARGS__)

// Preprocessor callbacks
#define NPY_CPU_DISPATCH_DECLARE_CB_(DUMMY, TARGET_NAME, LEFT, ...) \
    // Placeholder macro for defining callback behavior based on dispatch targets

#endif // NUMPY_CORE_SRC_COMMON_NPY_CPU_DISPATCH_DISTUTILS_H_
    # 将宏展开为形如 LEFT_TARGET_NAME 的标识符，后接额外的参数（如果有的话）
    NPY_CAT(NPY_CAT(LEFT, _), TARGET_NAME) __VA_ARGS__;
#define NPY_CPU_DISPATCH_DECLARE_BASE_CB_(LEFT, ...) \
    LEFT __VA_ARGS__;
// 定义一个宏，展开为给定的左参数，后跟可变参数列表

// Dummy CPU runtime checking
#define NPY_CPU_DISPATCH_DECLARE_CHK_(FEATURE)
// 定义一个宏，用于虚拟的 CPU 运行时检查，该宏为空

#define NPY_CPU_DISPATCH_DECLARE_XB(...) \
    NPY__CPU_DISPATCH_CALL(NPY_CPU_DISPATCH_DECLARE_CHK_, NPY_CPU_DISPATCH_DECLARE_CB_, __VA_ARGS__)
// 定义一个宏，展开为调用 NPY__CPU_DISPATCH_CALL 宏，传入 NPY_CPU_DISPATCH_DECLARE_CHK_ 和 NPY_CPU_DISPATCH_DECLARE_CB_ 宏以及给定的可变参数列表

#define NPY_CPU_DISPATCH_CALL(...) \
    NPY__CPU_DISPATCH_CALL(NPY_CPU_HAVE, NPY_CPU_DISPATCH_CALL_CB_, __VA_ARGS__) \
    NPY__CPU_DISPATCH_BASELINE_CALL(NPY_CPU_DISPATCH_CALL_BASE_CB_, __VA_ARGS__)
// 定义一个宏，展开为调用 NPY__CPU_DISPATCH_CALL 和 NPY__CPU_DISPATCH_BASELINE_CALL 宏，传入 NPY_CPU_HAVE、NPY_CPU_DISPATCH_CALL_CB_ 和 NPY_CPU_DISPATCH_CALL_BASE_CB_ 宏以及给定的可变参数列表

// Preprocessor callbacks
#define NPY_CPU_DISPATCH_CALL_CB_(TESTED_FEATURES, TARGET_NAME, LEFT, ...) \
    (TESTED_FEATURES) ? (NPY_CAT(NPY_CAT(LEFT, _), TARGET_NAME) __VA_ARGS__) :
// 定义预处理器回调宏，根据 TESTED_FEATURES 条件展开为 LEFT_TARGET_NAME 或者空

#define NPY_CPU_DISPATCH_CALL_BASE_CB_(LEFT, ...) \
    (LEFT __VA_ARGS__)
// 定义预处理器基础回调宏，展开为 LEFT 加上给定的可变参数列表

#define NPY_CPU_DISPATCH_CALL_XB(...) \
    NPY__CPU_DISPATCH_CALL(NPY_CPU_HAVE, NPY_CPU_DISPATCH_CALL_XB_CB_, __VA_ARGS__) \
    ((void) 0 /* discarded expression value */)
// 定义一个宏，展开为调用 NPY__CPU_DISPATCH_CALL 和 ((void) 0) 的组合，传入 NPY_CPU_HAVE、NPY_CPU_DISPATCH_CALL_XB_CB_ 宏以及给定的可变参数列表，且忽略表达式的值

#define NPY_CPU_DISPATCH_CALL_XB_CB_(TESTED_FEATURES, TARGET_NAME, LEFT, ...) \
    (TESTED_FEATURES) ? (void) (NPY_CAT(NPY_CAT(LEFT, _), TARGET_NAME) __VA_ARGS__) :
// 定义预处理器回调宏，根据 TESTED_FEATURES 条件展开为 void 类型的 LEFT_TARGET_NAME 或者空

#define NPY_CPU_DISPATCH_CALL_ALL(...) \
    (NPY__CPU_DISPATCH_CALL(NPY_CPU_HAVE, NPY_CPU_DISPATCH_CALL_ALL_CB_, __VA_ARGS__) \
    NPY__CPU_DISPATCH_BASELINE_CALL(NPY_CPU_DISPATCH_CALL_ALL_BASE_CB_, __VA_ARGS__))
// 定义一个宏，展开为调用 NPY__CPU_DISPATCH_CALL 和 NPY__CPU_DISPATCH_BASELINE_CALL 宏，传入 NPY_CPU_HAVE、NPY_CPU_DISPATCH_CALL_ALL_CB_ 和 NPY_CPU_DISPATCH_CALL_ALL_BASE_CB_ 宏以及给定的可变参数列表

// Preprocessor callbacks
#define NPY_CPU_DISPATCH_CALL_ALL_CB_(TESTED_FEATURES, TARGET_NAME, LEFT, ...) \
    ((TESTED_FEATURES) ? (NPY_CAT(NPY_CAT(LEFT, _), TARGET_NAME) __VA_ARGS__) : (void) 0),
// 定义预处理器回调宏，根据 TESTED_FEATURES 条件展开为 LEFT_TARGET_NAME 或者空，并在否定情况下返回空

#define NPY_CPU_DISPATCH_CALL_ALL_BASE_CB_(LEFT, ...) \
    ( LEFT __VA_ARGS__ )
// 定义预处理器基础回调宏，展开为 LEFT 加上给定的可变参数列表

#define NPY_CPU_DISPATCH_INFO() \
    { \
        NPY__CPU_DISPATCH_CALL(NPY_CPU_HAVE, NPY_CPU_DISPATCH_INFO_HIGH_CB_, DUMMY) \
        NPY__CPU_DISPATCH_BASELINE_CALL(NPY_CPU_DISPATCH_INFO_BASE_HIGH_CB_, DUMMY) \
        "", \
        NPY__CPU_DISPATCH_CALL(NPY_CPU_HAVE, NPY_CPU_DISPATCH_INFO_CB_, DUMMY) \
        NPY__CPU_DISPATCH_BASELINE_CALL(NPY_CPU_DISPATCH_INFO_BASE_CB_, DUMMY) \
        ""\
    }
// 定义一个宏，展开为一个包含两个调用的代码块，分别调用 NPY__CPU_DISPATCH_CALL 和 NPY__CPU_DISPATCH_BASELINE_CALL 宏，传入 NPY_CPU_HAVE、NPY_CPU_DISPATCH_INFO_HIGH_CB_、NPY_CPU_DISPATCH_INFO_BASE_HIGH_CB_、NPY_CPU_DISPATCH_INFO_CB_ 和 NPY_CPU_DISPATCH_INFO_BASE_CB_ 宏以及 DUMMY 参数

#define NPY_CPU_DISPATCH_INFO_HIGH_CB_(TESTED_FEATURES, TARGET_NAME, ...) \
    (TESTED_FEATURES) ? NPY_TOSTRING(TARGET_NAME) :
// 定义预处理器高级信息回调宏，根据 TESTED_FEATURES 条件展开为 TARGET_NAME 的字符串化或空

#define NPY_CPU_DISPATCH_INFO_BASE_HIGH_CB_(...) \
    (1) ? "baseline(" NPY_WITH_CPU_BASELINE ")" :
// 定义预处理器基础高级信息回调宏，展开为字符串 "baseline(NPY_WITH_CPU_BASELINE)"

// Preprocessor callbacks
#define NPY_CPU_DISPATCH_INFO_CB_(TESTED_FEATURES, TARGET_NAME, ...) \
    NPY_TOSTRING(TARGET_NAME) " "
// 定义预处理器信息回调宏，将 TARGET_NAME 字符串化并添加空格

#define NPY_CPU_DISPATCH_INFO_BASE_CB_(...) \
    "baseline(" NPY_WITH_CPU_BASELINE ")"
// 定义预处理器基础信息回调宏，展开为字符串 "baseline(NPY_WITH_CPU_BASELINE)"

#endif  // NUMPY_CORE_SRC_COMMON_NPY_CPU_DISPATCH_DISTUTILS_H_
// 结束宏定义，用于条件编译
```