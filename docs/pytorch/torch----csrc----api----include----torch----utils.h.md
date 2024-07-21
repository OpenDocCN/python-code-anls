# `.\pytorch\torch\csrc\api\include\torch\utils.h`

```py
#pragma once
// 预处理指令，确保头文件只被编译一次

#include <ATen/Parallel.h>
// 包含 ATen 库的并行处理功能

#include <ATen/record_function.h>
// 包含 ATen 库的记录函数功能

#include <torch/csrc/api/include/torch/types.h>
// 包含 Torch 库的数据类型定义

#include <torch/csrc/autograd/grad_mode.h>
// 包含 Torch 库的自动求导模式管理功能

#include <torch/csrc/autograd/profiler.h>
// 包含 Torch 库的性能分析器功能

#include <cstdint>
// 包含标准整数类型的头文件

namespace torch {

/// A RAII, thread-local guard that disabled gradient calculation.
///
/// Disabling gradient calculation is useful for inference, when you are sure
/// that you will not call `at::Tensor::backward`. It will reduce memory
/// consumption for computations that would otherwise have `requires_grad() ==
/// true`.
///
/// In this mode, the result of every computation will have
/// `requires_grad() == false`, even when the inputs have `requires_grad() ==
/// true`.
///
/// This context manager is thread-local; it will not affect computation
/// in other threads.
///
/// Example:
/// @code
/// auto x = torch::tensor({1.}, torch::requires_grad());
/// {
///   torch::NoGradGuard no_grad;
///   auto y = x * 2;
///   std::cout << y.requires_grad() << std::endl; // prints `false`
/// }
/// {
///   auto doubler = [](torch::Tensor x) {
///     torch::NoGradGuard no_grad;
///     return x * 2;
///   };
///   auto z = doubler(x);
///   std::cout << z.requires_grad() << std::endl; // prints `false`
/// }
/// @endcode
using NoGradGuard = at::NoGradGuard;
// 使用 ATen 库定义的 NoGradGuard 类别名，用于禁用梯度计算的线程本地保护

/// A RAII, thread-local guard that sets gradient calculation to on or off.
///
/// ``AutoGradMode`` will enable or disable grads based on its argument
/// `enabled`.
///
/// This context manager is thread-local; it will not affect computation
/// in other threads.
///
/// \param enabled: Flag whether to enable grad (``true``), or disable
///              (``false``). This can be used to conditionally enable
///              gradients.
///
/// Example:
/// @code
/// auto x = torch::tensor({1.}, torch::requires_grad());
/// {
///   torch::AutoGradMode enable_grad(true);
///   auto y = x * 2;
///   std::cout << y.requires_grad() << std::endl; // prints `true`
/// }
/// {
///   torch::AutoGradMode enable_grad(false);
///   auto y = x * 2;
///   std::cout << y.requires_grad() << std::endl; // prints `false`
/// }
/// @endcode
using AutoGradMode = at::AutoGradMode;
// 使用 ATen 库定义的 AutoGradMode 类别名，用于在线程本地设置梯度计算开启或关闭

/// Sets the global random seed for all newly created CPU and CUDA tensors.
using at::manual_seed;
// 使用 ATen 库中的 manual_seed 函数，设置全局随机种子，适用于所有新创建的 CPU 和 CUDA 张量

// Called during new thread initialization
using at::init_num_threads;
// 使用 ATen 库中的 init_num_threads 函数，用于新线程初始化时调用

// Returns the number of threads used in parallel region.
using at::get_num_threads;
// 使用 ATen 库中的 get_num_threads 函数，返回并行区域中使用的线程数目

// Sets the number of threads to be used in parallel region.
using at::set_num_threads;
// 使用 ATen 库中的 set_num_threads 函数，设置并行区域中要使用的线程数目

// Returns the number of threads used for inter-op parallelism.
using at::get_num_interop_threads;
// 使用 ATen 库中的 get_num_interop_threads 函数，返回用于跨操作并行的线程数目

// Sets the number of threads to be used for inter-op parallelism.
using at::set_num_interop_threads;
// 使用 ATen 库中的 set_num_interop_threads 函数，设置用于跨操作并行的线程数目

// Returns true if both t1, t2 are undefined or both are defined and equal
inline bool equal_if_defined(Tensor t1, Tensor t2) {
  return (
      (!t1.defined() && !t2.defined()) ||
      (t1.defined() && t2.defined() && torch::equal(t1, t2)));
}
// 内联函数，如果 t1 和 t2 都未定义，或者两者均定义且相等，则返回 true

// RecordFunction API
// 引入命名空间 torch 中的 at 模块提供的一系列函数和类
using at::addGlobalCallback;                   // 使用 at 命名空间中的 addGlobalCallback 函数
using at::addThreadLocalCallback;              // 使用 at 命名空间中的 addThreadLocalCallback 函数
using at::CallbackHandle;                      // 使用 at 命名空间中的 CallbackHandle 类
using at::clearCallbacks;                      // 使用 at 命名空间中的 clearCallbacks 函数
using at::clearGlobalCallbacks;                // 使用 at 命名空间中的 clearGlobalCallbacks 函数
using at::clearThreadLocalCallbacks;           // 使用 at 命名空间中的 clearThreadLocalCallbacks 函数
using at::DisableRecordFunctionGuard;          // 使用 at 命名空间中的 DisableRecordFunctionGuard 类
using at::enableRecordFunction;                // 使用 at 命名空间中的 enableRecordFunction 函数
using at::hasCallbacks;                        // 使用 at 命名空间中的 hasCallbacks 函数
using at::hasGlobalCallbacks;                  // 使用 at 命名空间中的 hasGlobalCallbacks 函数
using at::hasThreadLocalCallbacks;             // 使用 at 命名空间中的 hasThreadLocalCallbacks 函数
using at::isRecordFunctionEnabled;             // 使用 at 命名空间中的 isRecordFunctionEnabled 函数
using at::RecordFunction;                      // 使用 at 命名空间中的 RecordFunction 类
using at::RecordFunctionCallback;              // 使用 at 命名空间中的 RecordFunctionCallback 类
using at::RecordFunctionGuard;                 // 使用 at 命名空间中的 RecordFunctionGuard 类
using at::removeCallback;                      // 使用 at 命名空间中的 removeCallback 函数

} // 命名空间 torch 的结束标志
```