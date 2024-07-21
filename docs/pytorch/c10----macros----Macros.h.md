# `.\pytorch\c10\macros\Macros.h`

```
#ifndef C10_MACROS_MACROS_H_
#define C10_MACROS_MACROS_H_
#include <cassert>

/* Main entry for c10/macros.
 *
 * In your code, include c10/macros/Macros.h directly, instead of individual
 * files in this folder.
 */

// For build systems that do not directly depend on CMake and directly build
// from the source directory (such as Buck), one may not have a cmake_macros.h
// file at all. In this case, the build system is responsible for providing
// correct macro definitions corresponding to the cmake_macros.h.in file.
//
// In such scenarios, one should define the macro
//     C10_USING_CUSTOM_GENERATED_MACROS
// to inform this header that it does not need to include the cmake_macros.h
// file.

#ifndef C10_USING_CUSTOM_GENERATED_MACROS
#include <c10/macros/cmake_macros.h>
#endif // C10_USING_CUSTOM_GENERATED_MACROS

#include <c10/macros/Export.h>

#if defined(__clang__)
#define __ubsan_ignore_float_divide_by_zero__ \
  __attribute__((no_sanitize("float-divide-by-zero")))
#define __ubsan_ignore_undefined__ __attribute__((no_sanitize("undefined")))
#define __ubsan_ignore_signed_int_overflow__ \
  __attribute__((no_sanitize("signed-integer-overflow")))
#define __ubsan_ignore_pointer_overflow__ \
  __attribute__((no_sanitize("pointer-overflow")))
#define __ubsan_ignore_function__ __attribute__((no_sanitize("function")))
#else
#define __ubsan_ignore_float_divide_by_zero__
#define __ubsan_ignore_undefined__
#define __ubsan_ignore_signed_int_overflow__
#define __ubsan_ignore_pointer_overflow__
#define __ubsan_ignore_function__
#endif

// Detect address sanitizer as some stuff doesn't work with it
#undef C10_ASAN_ENABLED

// for clang
#if defined(__has_feature)
#if ((__has_feature(address_sanitizer)))
#define C10_ASAN_ENABLED 1
#endif
#endif

// for gcc
#if defined(__SANITIZE_ADDRESS__)
#if __SANITIZE_ADDRESS__
#if !defined(C10_ASAN_ENABLED)
#define C10_ASAN_ENABLED 1
#endif
#endif
#endif

#if !defined(C10_ASAN_ENABLED)
#define C10_ASAN_ENABLED 0
#endif

// Detect undefined-behavior sanitizer (UBSAN)
#undef C10_UBSAN_ENABLED

// for clang or gcc >= 14
// NB: gcc 14 adds support for Clang's __has_feature
//   https://gcc.gnu.org/gcc-14/changes.html
//   gcc < 14 doesn't have a macro for UBSAN
//   (e.g. __SANITIZE_UNDEFINED__ does not exist in gcc)
//   https://github.com/google/sanitizers/issues/765
#if defined(__has_feature)
#if ((__has_feature(undefined_behavior_sanitizer)))
#define C10_UBSAN_ENABLED 1
#endif
#endif

#if !defined(C10_UBSAN_ENABLED)
#define C10_UBSAN_ENABLED 0
#endif

// Disable the copy and assignment operator for a class. Note that this will
// disable the usage of the class in std containers.
#define C10_DISABLE_COPY_AND_ASSIGN(classname) \
  classname(const classname&) = delete;        \
  classname& operator=(const classname&) = delete

#define C10_CONCATENATE_IMPL(s1, s2) s1##s2
#define C10_CONCATENATE(s1, s2) C10_CONCATENATE_IMPL(s1, s2)

#define C10_MACRO_EXPAND(args) args

#define C10_STRINGIZE_IMPL(x) #x
/// 定义宏 C10_STRINGIZE(x)，将 x 转换为字符串
#define C10_STRINGIZE(x) C10_STRINGIZE_IMPL(x)

/**
 * C10_ANONYMOUS_VARIABLE(str) 引入一个新的标识符，以 str 开头并以唯一数字结尾。
 * 如果编译器支持 __COUNTER__，则使用它生成唯一标识符，否则使用 __LINE__。
 */
#ifdef __COUNTER__
#define C10_UID __COUNTER__
#define C10_ANONYMOUS_VARIABLE(str) C10_CONCATENATE(str, __COUNTER__)
#else
#define C10_UID __LINE__
#define C10_ANONYMOUS_VARIABLE(str) C10_CONCATENATE(str, __LINE__)
#endif

#ifdef __has_cpp_attribute
/// 定义宏 C10_HAS_CPP_ATTRIBUTE(x)，检查编译器是否支持指定的 C++ 属性
#define C10_HAS_CPP_ATTRIBUTE(x) __has_cpp_attribute(x)
#else
#define C10_HAS_CPP_ATTRIBUTE(x) (0)
#endif

/// C10_NODISCARD - 如果类型或返回值被丢弃，则发出警告。
//
// 在此我们应该检查 __cplusplus > 201402L，因为 [[nodiscard]] 只在 C++17 中定义。
// 但是一些我们关心的编译器没有明确声明支持 C++17（如 clang），但仍支持该属性。
// 实际上，这不仅仅是一个好主意，而是一种法律要求：clang::warn_unused_result
// 在 nvcc + clang 上无法正常工作，而这种情况的最佳解决方案是使用 [[nodiscard]]；
// 参见 https://github.com/pytorch/pytorch/issues/13118
//
// 注：未来的编辑者注意：如果您注意到编译器的行为不正常（例如，它宣称支持，但实际支持不起作用，
// 或者它发出警告），一些严格的编译器包括 MSVC，会报错：
//
//  error C2429: attribute 'nodiscard' requires compiler flag '/std:c++latest'
//
// 例子：
//  - MSVC 19.14: https://godbolt.org/z/Dzd7gn （需要 /std:c++latest）
//  - Clang 8.0.0: https://godbolt.org/z/3PYL4Z （始终宣称支持）
//  - gcc 8.3: https://godbolt.org/z/4tLMQS （始终宣称支持）
#if C10_HAS_CPP_ATTRIBUTE(nodiscard)
#define C10_NODISCARD [[nodiscard]]
// 对于 llvm.org/PR23435 的问题，因此 clang 3.6 及以下在 C 模式下给 __has_cpp_attribute
// 传递作用域属性时会发出错误，因此这里使用一个临时的兼容性宏。
#elif __cplusplus && C10_HAS_CPP_ATTRIBUTE(clang::warn_unused_result)
// TODO: 如果在 Windows 上仍然触发 https://github.com/pytorch/pytorch/issues/13118，
// 需要修复。
#define C10_NODISCARD [[clang::warn_unused_result]]
#else
#define C10_NODISCARD
#endif

// suppress an unused variable.
#if defined(_MSC_VER) && !defined(__clang__)
/// 在 MSVC 编译器中，用于抑制未使用变量的警告。
#define C10_UNUSED __pragma(warning(suppress : 4100 4101))
#else
#define C10_UNUSED __attribute__((__unused__))
#endif //_MSC_VER

#if !defined(__has_attribute)
#define __has_attribute(x) 0
#endif

// Direct port of LLVM_ATTRIBUTE_USED.
#if __has_attribute(used)
/// 定义宏 C10_USED，用于标记变量或函数，表示该变量或函数可能会被使用。
#define C10_USED __attribute__((__used__))
#else
#define C10_USED
#endif

/// 定义宏 C10_RESTRICT，表示指针具有限制性（restrictive）。
#define C10_RESTRICT __restrict

// Simply define the namespace, in case a dependent library want to refer to
// the c10 namespace but not any nontrivial files.
/// 简单地定义命名空间，以便依赖库可以引用 c10 命名空间，但不引入任何复杂文件。
namespace c10 {}
namespace c10::cuda {}
namespace c10::hip {}
namespace c10::xpu {}

// Since C10 is the core library for caffe2 (and aten), we will simply reroute
// all abstractions defined in c10 to be available in caffe2 as well.
//
// 由于 C10 是 caffe2（和 aten）的核心库，我们将所有在 c10 中定义的抽象重新路由，
// 以便它们在 caffe2 中也可用。
/// This namespace declaration ensures backward compatibility by allowing the use of symbols from the c10 namespace within caffe2.
namespace caffe2 {
using namespace c10;
}

/// This namespace declaration ensures backward compatibility by allowing the use of symbols from the c10 namespace within at.
namespace at {
using namespace c10;
}

/// This namespace declaration ensures backward compatibility by allowing the use of symbols from the c10::cuda namespace within at::cuda.
namespace at::cuda {
using namespace c10::cuda;
} // namespace at::cuda

// WARNING!!! THIS IS A GIANT HACK!!!
/// This namespace declaration ensures compatibility between c10::hip and at::cuda namespaces when including and using both c10/hip and c10/cuda simultaneously.
/// This hack is necessary because HIPIFY modifies files in place and assumes c10::hip is available from at::cuda.
namespace at::cuda {
using namespace c10::hip;
} // namespace at::cuda

/// This namespace declaration ensures the use of symbols from the c10::xpu namespace within at::xpu.
namespace at::xpu {
using namespace c10::xpu;
} // namespace at::xpu

// C10_LIKELY/C10_UNLIKELY
//
// These macros provide hinting to the compiler about the likelihood of a condition being true or false,
// improving branch prediction.
//
// The __builtin_expect function in GCC/Clang is used to indicate that a certain branch is more likely to be taken.
#if defined(__GNUC__) || defined(__ICL) || defined(__clang__)
#define C10_LIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 1))
#define C10_UNLIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 0))
#else
#define C10_LIKELY(expr) (expr)
#define C10_UNLIKELY(expr) (expr)
#endif

/// C10_NOINLINE - Functions marked with this attribute will not be inlined by the compiler.
#ifdef __GNUC__
#define C10_NOINLINE __attribute__((noinline))
#elif _MSC_VER
#define C10_NOINLINE __declspec(noinline)
#else
#define C10_NOINLINE
#endif

/// C10_ALWAYS_INLINE - Functions marked with this attribute will always be inlined by the compiler when possible.
#if defined(_MSC_VER)
#define C10_ALWAYS_INLINE __forceinline
#elif __has_attribute(always_inline) || defined(__GNUC__)
#define C10_ALWAYS_INLINE __attribute__((__always_inline__)) inline
#else
#define C10_ALWAYS_INLINE inline
#endif

/// C10_ATTR_VISIBILITY_HIDDEN - Sets the visibility attribute to hidden for functions and variables.
#if defined(_MSC_VER)
#define C10_ATTR_VISIBILITY_HIDDEN
#elif defined(__GNUC__)
#define C10_ATTR_VISIBILITY_HIDDEN __attribute__((__visibility__("hidden")))
#else
#define C10_ATTR_VISIBILITY_HIDDEN
#endif

#define C10_ERASE C10_ALWAYS_INLINE C10_ATTR_VISIBILITY_HIDDEN

#include <cstdint>

#ifdef __HIPCC__
// Unlike CUDA, HIP requires a HIP header to be included for __host__ to work properly.
// This include ensures that C10_HOST_DEVICE and related macros function correctly with HIP.
#include <hip/hip_runtime.h>
#endif

#if defined(__CUDACC__) || defined(__HIPCC__)
// Designates functions callable from both the host (CPU) and the device (GPU)
#define C10_HOST_DEVICE __host__ __device__
#define C10_DEVICE __device__
#define C10_HOST __host__
// Constants from CUDA programming guide
// The maximum number of threads per multiprocessor is 1024 for Turing GPUs.
// 定义了根据 CUDA 架构选择不同的线程限制常量，以避免警告
#if __CUDA_ARCH__ == 750
// 对于架构 7.5，每个流多处理器的最大线程数为 1024
constexpr uint32_t CUDA_MAX_THREADS_PER_SM = 1024;
#elif __CUDA_ARCH__ == 860 || __CUDA_ARCH__ == 870 || __CUDA_ARCH__ == 890
// 对于架构 8.6, 8.7, 8.9，每个流多处理器的最大线程数为 1536
constexpr uint32_t CUDA_MAX_THREADS_PER_SM = 1536;
#else
// 对于其它架构，每个流多处理器的最大线程数为 2048
constexpr uint32_t CUDA_MAX_THREADS_PER_SM = 2048;
#endif

// 所有架构当前都使用相同的每个块的最大线程数
constexpr uint32_t CUDA_MAX_THREADS_PER_BLOCK = 1024;

// CUDA_THREADS_PER_BLOCK_FALLBACK 是块大小的备用选择，256 是一个良好的默认值
constexpr uint32_t CUDA_THREADS_PER_BLOCK_FALLBACK = 256;

// 注意事项：如果你打算对启动边界的输入进行 constexpr 处理，
//         就要知道 __launch_bounds__ 能够接受 constexpr，但不能接受与模板相关的 constexpr。
//         目前我们在 Loops.cuh、Reduce.cuh 和 LossCTC.cuh 中使用依赖于模板参数的 launch_bounds。
//         因此，C10_MAX_THREADS_PER_BLOCK 和 C10_MIN_BLOCKS_PER_SM 被保留为宏。

// C10_MAX_THREADS_PER_BLOCK(val) 根据输入值选择适当的每块最大线程数
#define C10_MAX_THREADS_PER_BLOCK(val)           \
  (((val) <= CUDA_MAX_THREADS_PER_BLOCK) ? (val) \
                                         : CUDA_THREADS_PER_BLOCK_FALLBACK)

// C10_MIN_BLOCKS_PER_SM(threads_per_block, blocks_per_sm) 根据每块线程数和每个流多处理器的块数选择最小块数
#define C10_MIN_BLOCKS_PER_SM(threads_per_block, blocks_per_sm)        \
  ((((threads_per_block) * (blocks_per_sm) <= CUDA_MAX_THREADS_PER_SM) \
        ? (blocks_per_sm)                                              \
        : ((CUDA_MAX_THREADS_PER_SM + (threads_per_block)-1) /         \
           (threads_per_block))))

// C10_LAUNCH_BOUNDS 相当于 __launch_bounds__
#define C10_LAUNCH_BOUNDS_0 \
  __launch_bounds__(        \
      256, 4) // 默认的启动边界，应适合所有架构的占用和灵活性

// C10_LAUNCH_BOUNDS_1(max_threads_per_block) 基于最大每块线程数设置启动边界
#define C10_LAUNCH_BOUNDS_1(max_threads_per_block) \
  __launch_bounds__((C10_MAX_THREADS_PER_BLOCK((max_threads_per_block))))

// C10_LAUNCH_BOUNDS_2(max_threads_per_block, min_blocks_per_sm) 基于最大每块线程数和最小每多处理器块数设置启动边界
#define C10_LAUNCH_BOUNDS_2(max_threads_per_block, min_blocks_per_sm) \
  __launch_bounds__(                                                  \
      (C10_MAX_THREADS_PER_BLOCK((max_threads_per_block))),           \
      (C10_MIN_BLOCKS_PER_SM((max_threads_per_block), (min_blocks_per_sm))))

#else
// 如果未定义 CUDA，则定义为空的宏，表示不使用 CUDA
#define C10_HOST_DEVICE
#define C10_HOST
#define C10_DEVICE
#endif

#if defined(USE_ROCM)
// 如果定义了 USE_ROCM，则定义 HIP 的主机设备函数属性
#define C10_HIP_HOST_DEVICE __host__ __device__
#else
// 否则为空宏
#define C10_HIP_HOST_DEVICE
#endif
#if defined(USE_ROCM)
// 如果定义了 USE_ROCM，定义 C10_WARP_SIZE 为 warpSize（在 hip_runtime.h 中定义，通常为 64 或 32）
#define C10_WARP_SIZE warpSize
#else
// 否则，定义 C10_WARP_SIZE 为 32
#define C10_WARP_SIZE 32
#endif

#if defined(_MSC_VER) && _MSC_VER <= 1900
// 如果在 MSVC 编译器中且版本号小于等于 1900，将 __func__ 定义为 __FUNCTION__
#define __func__ __FUNCTION__
#endif

// CUDA_KERNEL_ASSERT 检查断言，即使在定义了 NDEBUG 的情况下也会执行。这对于 CUDA 代码中重要的断言在 Release 模式下仍然被执行是有用的。
#if defined(__ANDROID__) || defined(__APPLE__) || defined(__FreeBSD__)
// 这些平台不支持 assert()，因此将 CUDA_KERNEL_ASSERT 和 SYCL_KERNEL_ASSERT 定义为空
#define CUDA_KERNEL_ASSERT(cond)
#define SYCL_KERNEL_ASSERT(cond)
#elif defined(_MSC_VER)
#if defined(NDEBUG)
// 如果在 MSVC 编译器中且定义了 NDEBUG

// 声明 _wassert 函数，用于在 CUDA 设备上执行断言检查
extern "C" {
C10_IMPORT
#if defined(__SYCL_DEVICE_ONLY__)
extern SYCL_EXTERNAL void _wassert(
    const wchar_t* wexpr,
    const wchar_t* wfile,
    unsigned line);
#else
#if defined(__CUDA_ARCH__)
__host__ __device__
#endif // __CUDA_ARCH__
    void
    _wassert(wchar_t const* _Message, wchar_t const* _File, unsigned _Line);
#endif // __SYCL_DEVICE_ONLY__
}
#endif // NDEBUG

// 定义 CUDA_KERNEL_ASSERT 和 SYCL_KERNEL_ASSERT 宏，用于执行断言检查
#define CUDA_KERNEL_ASSERT(cond)                 \
  if (C10_UNLIKELY(!(cond))) {                   \
    (void)(_wassert(                             \
               _CRT_WIDE(#cond),                 \
               _CRT_WIDE(__FILE__),              \
               static_cast<unsigned>(__LINE__)), \
           0);                                   \
  }
#define SYCL_KERNEL_ASSERT(cond)                 \
  if (C10_UNLIKELY(!(cond))) {                   \
    (void)(_wassert(                             \
               _CRT_WIDE(#cond),                 \
               _CRT_WIDE(__FILE__),              \
               static_cast<unsigned>(__LINE__)), \
           0);                                   \
  }
#else // __APPLE__, _MSC_VER
#if defined(NDEBUG)
// 如果不在上述平台且定义了 NDEBUG

extern "C" {
#if defined(__SYCL_DEVICE_ONLY__)
// 对于 SYCL 设备代码，声明 __assert_fail 函数用于执行断言失败处理
extern SYCL_EXTERNAL void __assert_fail(
    const char* expr,
    const char* file,
    unsigned int line,
    const char* func);
#else // __SYCL_DEVICE_ONLY__
#if (defined(__CUDA_ARCH__) && !(defined(__clang__) && defined(__CUDA__)))
// CUDA 支持 __assert_fail 函数，用于在设备和主机代码中执行相同的断言处理
__host__ __device__
#endif

    // 此处前向声明匹配 __assert_fail 函数的声明，确保不同编译单元中使用相同的声明以避免 'ambiguous declaration' 错误
    void
    __assert_fail(
        const char* assertion,
        const char* file,
        unsigned int line,
        const char* function) noexcept __attribute__((__noreturn__));
#endif // __SYCL_DEVICE_ONLY__
}
#endif // NDEBUG

// 在 ROCm 平台下，默认禁用内核断言
#if !defined(C10_USE_ROCM_KERNEL_ASSERT) and defined(USE_ROCM)
#define CUDA_KERNEL_ASSERT(cond)
#define SYCL_KERNEL_ASSERT(cond)
#else
#define CUDA_KERNEL_ASSERT(cond)                                         \
  // CUDA环境下的断言宏定义，如果条件cond不成立，则触发断言失败，显示条件字符串、文件名、行号和函数名信息 \
  if (C10_UNLIKELY(!(cond))) {                                           \
    __assert_fail(                                                       \
        #cond, __FILE__, static_cast<unsigned int>(__LINE__), __func__); \
  }

#define SYCL_KERNEL_ASSERT(cond)                                         \
  // SYCL环境下的断言宏定义，如果条件cond不成立，则触发断言失败，显示条件字符串、文件名、行号和函数名信息 \
  if (C10_UNLIKELY(!(cond))) {                                           \
    __assert_fail(                                                       \
        #cond, __FILE__, static_cast<unsigned int>(__LINE__), __func__); \
  }
#endif //  C10_USE_ROCM_KERNEL_ASSERT and USE_ROCM

#ifdef __APPLE__
#include <TargetConditionals.h>
#endif

#if defined(__ANDROID__)
#define C10_ANDROID 1
#define C10_MOBILE 1
// 如果目标平台是Android，则定义C10_ANDROID和C10_MOBILE宏
#elif (                   \
    defined(__APPLE__) && \
    (TARGET_IPHONE_SIMULATOR || TARGET_OS_SIMULATOR || TARGET_OS_IPHONE))
#define C10_IOS 1
#define C10_MOBILE 1
// 如果目标平台是iOS，则定义C10_IOS和C10_MOBILE宏
#endif // ANDROID / IOS

#if defined(C10_MOBILE) && C10_MOBILE
// 如果编译目标是移动平台，并且C10_MOBILE宏定义了，则定义为inline，否则使用默认的C10_ALWAYS_INLINE
#define C10_ALWAYS_INLINE_UNLESS_MOBILE inline
#else
#define C10_ALWAYS_INLINE_UNLESS_MOBILE C10_ALWAYS_INLINE
#endif

#if defined(__CUDA_ARCH__)
#if defined(_MSC_VER) && defined(__CUDACC__)
// 当CUDA架构存在时，如果是Windows平台并且使用CUDA编译器，则将const定义为非constexpr \
// 用于Windows NVCC编译器不支持静态constexpr类成员的情况，使用静态const char*成员替代
#define CONSTEXPR_EXCEPT_WIN_CUDA const
#define C10_HOST_CONSTEXPR_EXCEPT_WIN_CUDA __host__

#define STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(field, val) \
  static const char* field;
#define STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cls, field, val) \
  const char* cls::field = val;
#else
#define CONSTEXPR_EXCEPT_WIN_CUDA constexpr
#define C10_HOST_CONSTEXPR_EXCEPT_WIN_CUDA __host__

#define STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(field, val) \
  // 在非Windows NVCC的情况下，使用constexpr定义静态const char*成员 \
  static constexpr const char* field = val;
#define STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cls, field, val)
#endif
#else
#if defined(_MSC_VER) && defined(__CUDACC__)
// 当非CUDA架构时，如果是Windows平台并且使用CUDA编译器，则将const定义为非constexpr \
// 用于Windows NVCC编译器不支持静态constexpr类成员的情况，使用静态const char*成员替代
#define CONSTEXPR_EXCEPT_WIN_CUDA const
// 定义 CONSTEXPR_EXCEPT_WIN_CUDA 为 const

#define C10_HOST_CONSTEXPR_EXCEPT_WIN_CUDA
// 定义 C10_HOST_CONSTEXPR_EXCEPT_WIN_CUDA 为空

#ifdef __cplusplus
// 如果是 C++ 环境

#ifdef _MSC_VER
// 如果是 Microsoft Visual C++ 编译器

#define STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(field, val) \
  static const char* field;
// 定义静态常量字符串，用于非 Windows 平台，仅声明

#define STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cls, field, val) \
  const char* cls::field = val;
// 定义静态常量字符串，用于 Windows 平台，同时赋值

#else
// 对于非 Microsoft Visual C++ 编译器

#define STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(field, val) \
  static constexpr const char* field = val;
// 定义静态 constexpr 字符串，用于非 Windows 平台，同时赋值

#define STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cls, field, val)
// 在 Windows 平台上，静态常量字符串不作为类的一部分

#endif
#else
// 如果不是 C++ 环境

#define CONSTEXPR_EXCEPT_WIN_CUDA constexpr
// 定义 CONSTEXPR_EXCEPT_WIN_CUDA 为 constexpr

#define C10_HOST_CONSTEXPR_EXCEPT_WIN_CUDA constexpr
// 定义 C10_HOST_CONSTEXPR_EXCEPT_WIN_CUDA 为 constexpr

#define STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(field, val) \
  static constexpr const char* field = val;
// 定义静态 constexpr 字符串，用于非 Windows 平台，同时赋值

#define STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cls, field, val)
// 在 Windows 平台上，静态常量字符串不作为类的一部分

#endif
#endif

#ifndef HAS_DEMANGLE
// 如果未定义 HAS_DEMANGLE

#if defined(__ANDROID__) || defined(_WIN32) || defined(__EMSCRIPTEN__)
#define HAS_DEMANGLE 0
// 如果是 Android、Windows 或者 Emscripten 平台，HAS_DEMANGLE 定义为 0
#elif defined(__APPLE__) && \
    (TARGET_IPHONE_SIMULATOR || TARGET_OS_SIMULATOR || TARGET_OS_IPHONE)
#define HAS_DEMANGLE 0
// 如果是 Apple 平台的模拟器或者 iPhone/iPad 设备，HAS_DEMANGLE 定义为 0
#else
#define HAS_DEMANGLE 1
// 其他情况下，HAS_DEMANGLE 定义为 1
#endif
#endif // HAS_DEMANGLE

#define _C10_PRAGMA__(string) _Pragma(#string)
// 将参数 string 转换为 _Pragma 指令的字符串

#define _C10_PRAGMA_(string) _C10_PRAGMA__(string)
// 调用 _C10_PRAGMA__ 宏来实现 _C10_PRAGMA_ 宏

#ifdef __clang__
// 如果是 Clang 编译器

#define C10_CLANG_DIAGNOSTIC_PUSH() _Pragma("clang diagnostic push")
// 定义推送 Clang 编译器诊断信息的指令

#define C10_CLANG_DIAGNOSTIC_POP() _Pragma("clang diagnostic pop")
// 定义弹出 Clang 编译器诊断信息的指令

#define C10_CLANG_DIAGNOSTIC_IGNORE(flag) \
  _C10_PRAGMA_(clang diagnostic ignored flag)
// 定义忽略特定警告标志的指令

#define C10_CLANG_HAS_WARNING(flag) __has_warning(flag)
// 检查是否存在特定的 Clang 警告

#else
// 对于非 Clang 编译器

#define C10_CLANG_DIAGNOSTIC_PUSH()
// 空指令，不进行任何操作

#define C10_CLANG_DIAGNOSTIC_POP()
// 空指令，不进行任何操作

#define C10_CLANG_DIAGNOSTIC_IGNORE(flag)
// 空指令，不进行任何操作

#define C10_CLANG_HAS_WARNING(flag) 0
// 默认情况下不存在 Clang 警告

#endif

#ifdef __clang__
// 如果是 Clang 编译器

#define C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED(warning)         \
  _C10_PRAGMA_(clang diagnostic push)                               \
  _C10_PRAGMA_(clang diagnostic ignored "-Wunknown-warning-option") \
  _C10_PRAGMA_(clang diagnostic ignored warning)
// 推送并忽略指定警告选项的 Clang 编译器指令序列

#define C10_DIAGNOSTIC_POP() _C10_PRAGMA_(clang diagnostic pop)
// 弹出 Clang 编译器的诊断信息指令

#elif __GNUC__
// 如果是 GCC 编译器

#define C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED(warning) \
  _C10_PRAGMA_(GCC diagnostic push)                         \
  _C10_PRAGMA_(GCC diagnostic ignored "-Wpragmas")          \
  _C10_PRAGMA_(GCC diagnostic ignored warning)
// 推送并忽略指定警告选项的 GCC 编译器指令序列

#define C10_DIAGNOSTIC_POP() _C10_PRAGMA_(GCC diagnostic pop)
// 弹出 GCC 编译器的诊断信息指令

#else
// 对于其他编译器

#define C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED(warning)
// 空指令，不进行任何操作

#define C10_DIAGNOSTIC_POP()
// 空指令，不进行任何操作

#endif

#endif // C10_MACROS_MACROS_H_
// 结束宏定义文件的条件编译指令
```