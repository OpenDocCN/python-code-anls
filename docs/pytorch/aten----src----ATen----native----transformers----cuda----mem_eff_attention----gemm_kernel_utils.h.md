# `.\pytorch\aten\src\ATen\native\transformers\cuda\mem_eff_attention\gemm_kernel_utils.h`

```
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <cutlass/arch/mma.h>

////////////////////////////////////////////////////////////////////////////////
// Some helper functions
////////////////////////////////////////////////////////////////////////////////

// 根据张量的数据类型分发函数调用
#define DISPATCH_TYPES(tensor, func)                                           \
  {                                                                            \
    if (query.scalar_type() == at::ScalarType::Float) {                        \
      using scalar_t = float;                                                  \
      func();                                                                  \
    } else if (query.scalar_type() == at::ScalarType::Half) {                  \
      using scalar_t = cutlass::half_t;                                        \
      func();                                                                  \
    } else if (query.scalar_type() == at::ScalarType::BFloat16) {              \
      using scalar_t = cutlass::bfloat16_t;                                    \
      func();                                                                  \
    } else {                                                                   \
      TORCH_CHECK(false, "Only fp32, half & bf16 supported at the moment");    \
    }                                                                          \
  }

// 根据布尔值分发函数调用
#define DISPATCH_BOOL(BOOL_V, BOOL_NAME, F) \
  {                                         \
    if (BOOL_V) {                           \
      constexpr bool BOOL_NAME = true;      \
      F();                                  \
    } else {                                \
      constexpr bool BOOL_NAME = false;     \
      F();                                  \
    }                                       \
  }

// 根据计算能力分发函数调用
#define DISPATCH_ARCHTAG(CC, func)                                        \
  {                                                                       \
    if (CC >= 80) {                                                       \
      using ArchTag = cutlass::arch::Sm80;                                \
      func();                                                             \
    } else if (CC >= 75) {                                                \
      using ArchTag = cutlass::arch::Sm75;                                \
      func();                                                             \
    } else if (CC >= 70) {                                                \
      using ArchTag = cutlass::arch::Sm70;                                \
      func();                                                             \


注释：这段代码定义了几个宏和函数，用于根据不同的条件分发函数调用。其中包括根据张量的数据类型（浮点数、半精度、BF16）、布尔值和计算能力（CUDA Compute Capability）选择合适的实现。
    } else if (CC >= 50) {                                                \
      // 如果计算能力 CC 大于或等于 50，使用 Sm50 架构标签
      using ArchTag = cutlass::arch::Sm50;                                \
      // 调用 func() 函数
      func();                                                             \
    } else {                                                              \
      // 如果计算能力 CC 小于 50，则抛出错误信息
      TORCH_CHECK(                                                     \
          false,                                                          \
          "Your device is too old. We require compute capability >= 50"); \
    }                                                                     \
  }
#define CHECK_NOSPARSE_CONTIGUOUS_CUDA(TENSOR)                            \
  // 检查张量是否在 CUDA 上运行
  TORCH_CHECK(TENSOR.is_cuda(), #TENSOR " must be a CUDA tensor");     \
  // 检查张量是否为稠密张量
  TORCH_CHECK(!TENSOR.is_sparse(), #TENSOR " must be a dense tensor"); \
  // 检查张量是否是连续的
  TORCH_CHECK(TENSOR.is_contiguous());

#define CHECK_NOSPARSE_LASTCONTIGUOUS_CUDA(TENSOR)                        \
  // 检查张量是否在 CUDA 上运行
  TORCH_CHECK(TENSOR.is_cuda(), #TENSOR " must be a CUDA tensor");     \
  // 检查张量是否为稠密张量
  TORCH_CHECK(!TENSOR.is_sparse(), #TENSOR " must be a dense tensor"); \
  // 检查张量的最后一个维度是否是连续的
  TORCH_CHECK(                                                         \
      TENSOR.stride(-1) == 1, #TENSOR ": last dimension must be contiguous");

#define CHECK_ALIGNED_PTR(PTR, ALIGNMENT) \
  // 检查指针是否按指定的对齐方式对齐
  TORCH_CHECK(                         \
      uint64_t(PTR) % ALIGNMENT == 0, #PTR " is not correctly aligned")

#define ASSIGN_CHECK_OVERFLOW(A, B)                                    \
  {                                                                    \
    A = B;                                                             \
    // 检查赋值后的结果是否溢出了 A 的类型限制
    TORCH_CHECK(                                                    \
        B < std::numeric_limits<decltype(A)>::max(), #B " overflows"); \
  }

namespace gemm_kernel_utils {

template <typename integer>
constexpr CUTLASS_HOST_DEVICE integer ceil_div(integer n, integer m) {
  // 计算 n 除以 m 的上取整结果
  return (n + m - 1) / m;
}

template <typename integer>
constexpr CUTLASS_HOST_DEVICE integer align_up(integer n, integer m) {
  // 将 n 向上对齐到最接近的 m 的倍数
  return ((n + m - 1) / m) * m;
}

////////////////////////////////////////////////////////////////////////////////
// Determine the type of GEMM we do (TensorCores or not, Shapes ...)
// TODO: Maybe we could rely on Cutlass's DefaultGemm templates
////////////////////////////////////////////////////////////////////////////////

// 如果不在以下特殊情况中，则回退到 Simt (CUDA 核心上的 FMA)
template <typename ArchTag, typename scalar_t_, typename Enable = void>
struct DefaultGemmType {
  static constexpr int ThreadK = 8;
  static constexpr int WarpK = 8;
  static constexpr int kMinimumAlignment = 1;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
  using OpClass = cutlass::arch::OpClassSimt;
  using Operator = cutlass::arch::OpMultiplyAdd;
};

// 专门用于具有 f32 的 Tensor Cores 的情况
template <typename ArchTag>
struct DefaultGemmType<
    ArchTag,
    float,
    typename cutlass::platform::enable_if<
        ArchTag::kMinComputeCapability >= 80>::type> {
  static constexpr int ThreadK = 32;
  static constexpr int WarpK = 32;
  static constexpr int kMinimumAlignment = 4;
  using OpClass = cutlass::arch::OpClassTensorOp;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
  using Operator = cutlass::arch::OpMultiplyAddFastF32;
};

// 专门用于具有 f16/bf16 的 Tensor Cores 的情况 (适用于 Sm75+)
template <typename ArchTag, typename scalar_t>
struct DefaultGemmType<
    ArchTag,
    scalar_t,
    typename cutlass::platform::enable_if<
        ArchTag::kMinComputeCapability >= 75>::type> {
  static constexpr int ThreadK = 32;
  static constexpr int WarpK = 32;
  static constexpr int kMinimumAlignment = 2;
  using OpClass = cutlass::arch::OpClassTensorOp;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 8>;
  using Operator = cutlass::arch::OpMultiplyAddFastF16;
};
    // 定义一个模板特化，基于条件 ArchTag::kMinComputeCapability 大于等于 75
    // 和 scalar_t 类型的位数等于 16 的情况
    typename cutlass::platform::enable_if<
        ArchTag::kMinComputeCapability >= 75 &&
        cutlass::sizeof_bits<scalar_t>::value == 16>::type> {
      // 线程维度 ThreadK 设为 32
      static constexpr int ThreadK = 32;
      // WarpK 维度设为 32
      static constexpr int WarpK = 32;
      // 最小对齐要求设为 4
      static constexpr int kMinimumAlignment = 4;
      // 使用 OpClassTensorOp 类型的操作类
      using OpClass = cutlass::arch::OpClassTensorOp;
      // 使用 GemmShape<16, 8, 8> 类型的指令形状
      using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
      // 使用 OpMultiplyAdd 类型的操作符
      using Operator = cutlass::arch::OpMultiplyAdd;
// 结构体模板的特化，用于具有 f16 张量核心的 Volta 架构
template <>
struct DefaultGemmType<cutlass::arch::Sm70, cutlass::half_t, void> {
  // 定义线程块大小 ThreadK
  static constexpr int ThreadK = 32;
  // 定义线程束大小 WarpK
  static constexpr int WarpK = 32;
  // 定义最小对齐要求
  static constexpr int kMinimumAlignment = 2;
  // 使用张量核心操作类
  using OpClass = cutlass::arch::OpClassTensorOp;
  // 定义指令形状
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  // 操作类型为乘加操作
  using Operator = cutlass::arch::OpMultiplyAdd;
};

// 使得可以根据条件选择不同类型的函数调用
template <bool kVal, typename TA, typename TB>
struct call_conditional;

// 当 kVal 为 true 时的部分特化
template <typename TA, typename TB>
struct call_conditional<true, TA, TB> {
  // 应用函数模板，返回类型为 ta(arg) 的结果类型
  template <typename Arg>
  static CUTLASS_HOST_DEVICE auto apply(TA ta, TB tb, Arg arg)
      -> decltype(ta(arg)) {
    return ta(arg);
  }
};

// 当 kVal 为 false 时的部分特化
template <typename TA, typename TB>
struct call_conditional<false, TA, TB> {
  // 应用函数模板，返回类型为 tb(arg) 的结果类型
  template <typename Arg>
  static CUTLASS_HOST_DEVICE auto apply(TA ta, TB tb, Arg arg)
      -> decltype(tb(arg)) {
    return tb(arg);
  }
};

////////////////////////////////////////////////////////////////////////////////
// 将变量标记为 warp-uniform - 以启用一些编译器优化
// 最便捷的方法是从第 0 个线程束中广播它
////////////////////////////////////////////////////////////////////////////////

// 将值标记为 warp-uniform 的模板函数
template <typename T>
CUTLASS_DEVICE T warp_uniform(T value) {
  // 匿名结构体定义
  struct {
    union {
      T value;          // 值
      uint32_t asInt;   // 整数表示
    };
  } p;
  p.value = value;      // 将 value 赋给结构体中的 value
  // 使用 warp shuffle 指令广播第 0 个线程束中的整数值
  p.asInt = __shfl_sync(0xffffffff, (unsigned)p.asInt, 0);
  return p.value;       // 返回原始值
}

// 将指针标记为 warp-uniform 的模板函数
template <typename T>
CUTLASS_DEVICE T* warp_uniform(T* ptr) {
  // 匿名结构体定义
  struct {
    union {
      T* ptr;           // 指针
      uint32_t asInt[2]; // 整数表示数组
    };
  } p;
  p.ptr = ptr;          // 将指针赋给结构体中的 ptr
  // 分别将两个整数表示标记为 warp-uniform
  p.asInt[0] = warp_uniform(p.asInt[0]);
  p.asInt[1] = warp_uniform(p.asInt[1]);
  return p.ptr;         // 返回原始指针
}
} // namespace gemm_kernel_utils
```