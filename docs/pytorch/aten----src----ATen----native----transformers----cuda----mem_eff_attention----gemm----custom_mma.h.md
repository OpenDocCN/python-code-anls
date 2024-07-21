# `.\pytorch\aten\src\ATen\native\transformers\cuda\mem_eff_attention\gemm\custom_mma.h`

```
/*
 * 版权所有 Meta Platforms, Inc. 及其关联公司。
 * 保留所有权利。
 *
 * 本源代码使用 BSD 风格许可证授权，该许可证可在源代码根目录下的 LICENSE 文件中找到。
 */
#pragma once

#include <ATen/native/transformers/cuda/mem_eff_attention/gemm/custom_mma_multistage.h>
#include <ATen/native/transformers/cuda/mem_eff_attention/gemm/custom_mma_pipelined.h>

#include <cutlass/gemm/threadblock/mma_multistage.h>
#include <cutlass/gemm/threadblock/mma_pipelined.h>

// 定义一个模板结构体 MakeCustomMma，根据不同的 Mma 类型和 kMaxK 值来生成自定义的 Mma 类型
template <typename Mma, int kMaxK>
struct MakeCustomMma;

// 特化模板结构体 MakeCustomMma，用于多阶段 MMA 线程块
template <
    typename Shape,
    typename IteratorA,
    typename SmemIteratorA,
    cutlass::arch::CacheOperation::Kind CacheOpA,
    typename IteratorB,
    typename SmemIteratorB,
    cutlass::arch::CacheOperation::Kind CacheOpB,
    typename ElementC,
    typename LayoutC,
    typename Policy,
    int Stages,
    cutlass::gemm::SharedMemoryClearOption SharedMemoryClear,
    int kMaxK>
struct MakeCustomMma<
    cutlass::gemm::threadblock::MmaMultistage<
        Shape,
        IteratorA,
        SmemIteratorA,
        CacheOpA,
        IteratorB,
        SmemIteratorB,
        CacheOpB,
        ElementC,
        LayoutC,
        Policy,
        Stages,
        SharedMemoryClear>,
    kMaxK> {
  
  // 如果 kMaxK 不是最大整数，根据 kMaxK 和 Shape::kK 动态计算所需的阶段数，以降低阶段数量
  static int constexpr kStages =
      kMaxK == cutlass::platform::numeric_limits<int>::max()
      ? Stages
      : cutlass::const_min(
            Stages,
            (kMaxK + int(Shape::kK) - 1) / int(Shape::kK));
  
  // 定义 Mma 类型，使用自定义的多阶段 MMA 类
  using Mma = cutlass::gemm::threadblock::CustomMmaMultistage<
      Shape,
      IteratorA,
      SmemIteratorA,
      CacheOpA,
      IteratorB,
      SmemIteratorB,
      CacheOpB,
      ElementC,
      LayoutC,
      Policy,
      kStages,
      SharedMemoryClear,
      kMaxK>;
};

// 特化模板结构体 MakeCustomMma，用于流水线化 MMA 线程块
template <
    typename Shape,
    typename IteratorA,
    typename SmemIteratorA,
    typename IteratorB,
    typename SmemIteratorB,
    typename ElementC,
    typename LayoutC,
    typename Policy,
    int kMaxK>
struct MakeCustomMma<
    cutlass::gemm::threadblock::MmaPipelined<
        Shape,
        IteratorA,
        SmemIteratorA,
        IteratorB,
        SmemIteratorB,
        ElementC,
        LayoutC,
        Policy>,
    kMaxK> {
  
  // 定义 Mma 类型，使用自定义的流水线化 MMA 类
  using Mma = cutlass::gemm::threadblock::CustomMmaPipelined<
      Shape,
      IteratorA,
      SmemIteratorA,
      IteratorB,
      SmemIteratorB,
      ElementC,
      LayoutC,
      Policy>;
};
```