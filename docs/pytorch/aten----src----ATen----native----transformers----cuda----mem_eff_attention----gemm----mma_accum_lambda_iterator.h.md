# `.\pytorch\aten\src\ATen\native\transformers\cuda\mem_eff_attention\gemm\mma_accum_lambda_iterator.h`

```
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <cutlass/functional.h>
#include <cutlass/gemm/warp/mma_simt_tile_iterator.h>
#include <cutlass/gemm/warp/mma_tensor_op_tile_iterator_sm70.h>
#include <cutlass/gemm/warp/mma_tensor_op_tile_iterator_sm80.h>
#include <cutlass/matrix_shape.h>

/*
TensorCores have different accumulator layouts.
This file provides a class to easily map the accumulator
i-th element with the corresponding matrix row/col.
*/

// 定义一个结构体模板 AccumLambdaIteratorSm80，包含类型 T、accum_t 和 kWarpSize
template <typename T, typename accum_t, int kWarpSize>
struct AccumLambdaIteratorSm80 {
  // 静态断言，确保 T 的布局是行主序（RowMajor）
  static_assert(
      cutlass::platform::
          is_same<typename T::Layout, cutlass::layout::RowMajor>::value,
      "only RowMajor is supported");

  // 使用 T 定义一些类型别名
  using Policy = typename T::Policy;
  using InstructionShape = typename T::InstructionShape;
  using OpDelta = typename T::OpDelta;
  using Shape = typename T::Shape;
  
  // 定义常量表示每次访问的元素数和每个瓦片的行数
  static int const kElementsPerAccess = InstructionShape::kN / 4;
  static int const kRowsPerTile = 8;
  static int const kAccumulatorRows = InstructionShape::kM / kRowsPerTile;

  // 静态函数，计算并返回每个线程块的偏移量
  static cutlass::MatrixCoord CUTLASS_DEVICE get_lane_offset(
      int8_t lane_id,
      int8_t warp_id,
      typename T::TensorCoord const& tile_offset) {
    int quad = (lane_id >> 2);
    int lane_in_quad = (lane_id & 3);
    return cutlass::MatrixCoord(
        quad + tile_offset.row() * Shape::kRow,
        lane_in_quad * kElementsPerAccess +
            tile_offset.column() * Shape::kColumn);
  }

  // 模板函数，迭代每个行，执行操作 op，并处理起始和结束行
  template <typename FA, typename FB, typename FC>
  CUTLASS_DEVICE static void iterateRows(
      cutlass::MatrixCoord& lane_offset,
      FA beginRow,
      FB op,
      FC endRow) {
    // 循环遍历 MmaIterations 的行
    CUTLASS_PRAGMA_UNROLL
    for (int mma_m = 0; mma_m < Policy::MmaIterations::kRow; ++mma_m) {
      // 循环遍历累加器的行
      CUTLASS_PRAGMA_UNROLL
      for (int row = 0; row < kAccumulatorRows; ++row) {
        // 计算累加器的行索引
        int accum_m = mma_m * InstructionShape::kM * OpDelta::kRow +
            row * kRowsPerTile + lane_offset.row();
        beginRow(accum_m);

        // 循环遍历 MmaIterations 的列
        CUTLASS_PRAGMA_UNROLL
        for (int mma_n = 0; mma_n < Policy::MmaIterations::kColumn; ++mma_n) {
          // 计算累加器的列索引和开始位置
          int mma_accum_start = kAccumulatorRows * kElementsPerAccess *
              (mma_n * Policy::MmaIterations::kRow + mma_m);
          CUTLASS_PRAGMA_UNROLL
          for (int col = 0; col < kElementsPerAccess; ++col) {
            // 计算累加器的列索引和具体索引
            int accum_n = mma_n * InstructionShape::kN * OpDelta::kColumn +
                col + lane_offset.column();
            int idx = mma_accum_start + row * kElementsPerAccess + col;
            op(accum_m, accum_n, idx); // 执行操作 op
          }
        }

        endRow(accum_m); // 结束当前行的处理
      }
    }
  }

  // 模板函数，减少同一行的处理，根据线程 ID 和函数执行操作
  template <typename DT, typename F>
  CUTLASS_DEVICE static bool reduceSameRow(int lane_id, DT& myValue, F fn) {
    // 在每个 warp 中，4 个线程将处理同一行
    // - 即具有相同 `quad` 的线程
    auto otherV = __shfl_xor_sync(0xffffffff, myValue, 1);
    // 使用 warp 内的线程协作函数 __shfl_xor_sync()，
    // 将本线程的值与偏移为1的线程的值进行异或操作并交换结果
    myValue = fn(myValue, otherV);
    otherV = __shfl_xor_sync(0xffffffff, myValue, 2);
    // 继续使用 __shfl_xor_sync()，将本线程的值与偏移为2的线程的值进行异或操作并交换结果
    myValue = fn(myValue, otherV);
    // 计算当前线程在其所在的 quad 中的 lane ID
    int lane_in_quad = (lane_id & 3);
    // 返回是否为 quad 中的第一个 lane（即 lane_in_quad == 0）
    return lane_in_quad == 0;
};

template <typename T, typename accum_t, int kWarpSize>
struct AccumLambdaIteratorSm70 {
  // 确保模板类型 T 使用 RowMajor 布局
  static_assert(
      cutlass::platform::
          is_same<typename T::Layout, cutlass::layout::RowMajor>::value,
      "only RowMajor is supported");

  // 定义模板类型成员变量
  using Policy = typename T::Policy;
  using InstructionShape = typename T::InstructionShape;
  using OpDelta = typename T::OpDelta;
  using Shape = typename T::Shape;
  using Element = accum_t;

  // 每个部分的元素数量
  static int const kElementsPerPartial = 4;
  // 元素形状，根据 Element 类型选择不同的矩阵形状
  using EleShapePerPatial = typename cutlass::platform::conditional<
      cutlass::platform::is_same<Element, float>::value,
      cutlass::MatrixShape<2, 2>,
      cutlass::MatrixShape<1, 4>>::type;
  // 每次 MMA 操作的元素数量
  static int const kElementsPerMma = 8;
  // 累加器的部分数量
  static int const kAccumulatorPatials = 2;
  // 每个部分的四重乘积形状
  using QuadShapePerPatialMma = cutlass::MatrixShape<4, 4>;

  // 获取每个线程的矩阵偏移量
  static cutlass::MatrixCoord CUTLASS_DEVICE get_lane_offset(
      int8_t lane_id,
      int8_t warp_id,
      typename T::TensorCoord const& tile_offset) {
    int quad = (lane_id >> 2);
    int lane_in_quad = (lane_id & 3);
    int accum_m, accum_n;

    if (cutlass::platform::is_same<Element, float>::value) {
      // 计算累加器的行和列偏移量，用于 float 类型元素
      accum_m = (((quad & 0x4) >> 1) + (quad & 0x1)) * 8 + (lane_in_quad & 1);
      accum_n =
          ((quad >> 1) & 0x1) * kElementsPerPartial * kAccumulatorPatials +
          (lane_in_quad & 2);
    } else {
      // 计算累加器的行和列偏移量，用于非 float 类型元素
      accum_m = (((quad & 0x4) >> 1) + (quad & 0x1)) * 8 +
          lane_in_quad; // (quad[2],quad[0])
      accum_n = ((quad >> 1) & 0x1) * kElementsPerPartial * kAccumulatorPatials;
    }
    // 返回计算得到的矩阵坐标
    return cutlass::MatrixCoord(
        accum_m + tile_offset.row() * Shape::kRow,
        accum_n + tile_offset.column() * Shape::kColumn);
  }

  // 函数模板，用于同一行内的值归约操作
  template <typename DT, typename F>
  CUTLASS_DEVICE static bool reduceSameRow(int lane_id, DT& myValue, F fn) {
    // 静态断言，仅支持 float 类型的累加
    static_assert(
        cutlass::platform::is_same<Element, float>::value,
        "update to support non-float accum");
    // 执行同步归约操作，用于 float 类型
    auto otherV = __shfl_xor_sync(0xffffffff, myValue, 1 << 1);
    myValue = fn(myValue, otherV);
    otherV = __shfl_xor_sync(0xffffffff, myValue, 1 << 3);
    myValue = fn(myValue, otherV);
    // 返回是否属于同一行的布尔值
    return (lane_id & ((1 << 1) | (1 << 3))) == 0;
  }

  // 函数模板，用于迭代行
  template <typename FA, typename FB, typename FC>
  CUTLASS_DEVICE static void iterateRows(
      cutlass::MatrixCoord& lane_offset,
      FA beginRow,
      FB op,
      FC endRow) {
    CUTLASS_PRAGMA_UNROLL
    // 迭代每个输出矩阵的行（tile_m）
    for (int tile_m = 0; tile_m < Policy::TileIterations::kRow; ++tile_m) {
      // 使用编译器指令展开内部循环，优化性能
      CUTLASS_PRAGMA_UNROLL
      // 迭代每个 MMA 操作的行（mma_m）
      for (int mma_m = 0; mma_m < Policy::MmaIterations::kRow; ++mma_m) {
        // 使用编译器指令展开内部循环，优化性能
        CUTLASS_PRAGMA_UNROLL
        // 迭代每个元素形状的行（m）
        for (int m = 0; m < EleShapePerPatial::kRow; ++m) {
          // 计算累积索引的行偏移量
          int accum_m = tile_m * Policy::InterleavedTile::kRow +
              mma_m * QuadShapePerPatialMma::kRow + m * 2 + lane_offset.row();
          // 调用操作的起始行
          beginRow(accum_m);

          // 迭代每个输出矩阵的列（tile_n）
          CUTLASS_PRAGMA_UNROLL
          for (int tile_n = 0; tile_n < Policy::TileIterations::kColumn;
               ++tile_n) {
            // 使用编译器指令展开内部循环，优化性能
            CUTLASS_PRAGMA_UNROLL
            // 迭代每个 MMA 操作的列（mma_n）
            for (int mma_n = 0; mma_n < Policy::MmaIterations::kColumn;
                 ++mma_n) {
              // 使用编译器指令展开内部循环，优化性能
              CUTLASS_PRAGMA_UNROLL
              // 迭代每个累加器的部分（p）
              for (int p = 0; p < kAccumulatorPatials; ++p) {
                // 使用编译器指令展开内部循环，优化性能
                CUTLASS_PRAGMA_UNROLL
                // 迭代每个元素形状的列（n）
                for (int n = 0; n < EleShapePerPatial::kColumn; ++n) {
                  // 计算 MMA 累积的起始索引
                  int mma_accum_start =
                      (((tile_n * Policy::TileIterations::kRow + tile_m) *
                            Policy::MmaIterations::kColumn +
                        mma_n) *
                           Policy::MmaIterations::kRow +
                       mma_m) *
                      kElementsPerMma;
                  // 计算累积索引的列偏移量
                  int accum_n = tile_n * Policy::InterleavedTile::kColumn +
                      mma_n * QuadShapePerPatialMma::kColumn +
                      p * Policy::InterleavedTile::kColumn / 2 + n +
                      lane_offset.column();
                  // 计算在累积中的全局索引
                  int idx = mma_accum_start + p * kElementsPerPartial +
                      m * EleShapePerPatial::kColumn + n;
                  // 调用操作函数
                  op(accum_m, accum_n, idx);
                }
              }
            }
          }
          // 调用操作的结束行
          endRow(accum_m);
        }
      }
    }
};

// 定义一个模板结构体 AccumLambdaIteratorSimt，接受三个模板参数：T 类型，accum_t 类型，kWarpSize 整数
template <typename T, typename accum_t, int kWarpSize>
struct AccumLambdaIteratorSimt {
  // 使用 T 类型的别名定义
  using Policy = typename T::Policy;
  using Iterations = typename T::Iterations;
  using Element = typename T::Element;
  using Delta = typename T::Delta;
  using Shape = typename T::Shape;
  
  // 静态断言，检查 T::Layout 是否为 cutlass::layout::RowMajor，若不是则编译时报错
  static_assert(
      cutlass::platform::
          is_same<typename T::Layout, cutlass::layout::RowMajor>::value,
      "only RowMajor is supported");

  // 静态方法，用于在同一行内进行归约操作
  template <typename DT, typename F>
  CUTLASS_DEVICE static bool reduceSameRow(int lane_id, DT& myValue, F fn) {
    CUTLASS_PRAGMA_UNROLL
    // 使用循环进行归约操作，每次将 myValue 和相邻线程的值使用 fn 函数进行归约
    for (int bit = 1; bit < Policy::WarpShape::kColumn; bit *= 2) {
      auto otherV = __shfl_xor_sync(0xffffffff, myValue, bit);
      myValue = fn(myValue, otherV);
    }
    // 判断当前线程是否为该行的第一个线程，返回结果
    return (lane_id & (Policy::WarpShape::kColumn - 1)) == 0;
  }

  // 静态方法，迭代处理每一行数据
  template <typename FA, typename FB, typename FC>
  CUTLASS_DEVICE static void iterateRows(
      cutlass::MatrixCoord& lane_offset,
      FA beginRow,
      FB op,
      FC endRow) {
    CUTLASS_PRAGMA_UNROLL
    // 外层循环遍历行迭代次数
    for (int mma_m = 0; mma_m < Iterations::kRow; ++mma_m) {
      CUTLASS_PRAGMA_UNROLL
      // 内层循环遍历每个线程块的行数
      for (int m = 0; m < Policy::LaneMmaShape::kM; ++m) {
        int accum_m = mma_m * Delta::kRow + m + lane_offset.row();
        // 调用 beginRow 函数处理当前行的起始位置
        beginRow(accum_m);

        CUTLASS_PRAGMA_UNROLL
        // 遍历每一列的迭代次数
        for (int mma_n = 0; mma_n < Iterations::kColumn; ++mma_n) {
          int accum_n =
              mma_n * Policy::WarpShape::kColumn * Policy::LaneMmaShape::kN +
              lane_offset.column();
          CUTLASS_PRAGMA_UNROLL
          // 遍历线程块中每一列的元素
          for (int n = 0; n < Policy::LaneMmaShape::kN; ++n) {
            int idx = n +
                Policy::LaneMmaShape::kN *
                    (mma_n +
                     Iterations::kColumn *
                         (m + mma_m * Policy::LaneMmaShape::kM));
            // 调用 op 函数处理当前位置的元素
            op(accum_m, accum_n + n, idx);
          }
        }
        // 调用 endRow 函数处理当前行的结束位置
        endRow(accum_m);
      }
    }
  }

  // 静态方法，计算线程块的偏移量
  static cutlass::MatrixCoord CUTLASS_DEVICE get_lane_offset(
      int8_t lane_id,
      int8_t warp_id,
      typename T::TensorCoord const& tile_offset) {
    static_assert(
        cutlass::platform::is_same<
            typename Policy::LaneLayout,
            cutlass::layout::RowMajorInterleaved<1>>::value,
        "");
    // 获取线程块的排列方式
    typename Policy::LaneLayout lane_layout = Policy::get_lane_layout();

    // 计算线程的偏移量
    cutlass::MatrixCoord lane_offset = lane_layout.inverse(lane_id) *
        cutlass::MatrixCoord(Policy::LaneMmaShape::kM,
                             Policy::LaneMmaShape::kN);
    return lane_offset +
        tile_offset * cutlass::MatrixCoord(Shape::kRow, Shape::kColumn);
  }
};

// 模板特化结构体，用于默认的 MMA 累积 Lambda 迭代器
template <typename T, typename accum_t, int kWarpSize>
struct DefaultMmaAccumLambdaIterator;

// 模板特化结构体，用于 MMA Simt Tile 迭代器
template <typename S, typename P, typename accum_t, int kWarpSize>
struct DefaultMmaAccumLambdaIterator<
    cutlass::gemm::warp::MmaSimtTileIterator<
        S,
        cutlass::gemm::Operand::kC,
        accum_t,
        cutlass::layout::RowMajor,
        P,
        1,
        1>,
    accum_t,                        # 模板参数：累加器的数据类型
    kWarpSize> {                    # 模板参数：Warp中线程数量的常数表达式

  using WarpIterator = typename cutlass::gemm::warp::MmaSimtTileIterator<
      S,                            # 模板参数：数据类型S，用于矩阵乘法中的操作数
      cutlass::gemm::Operand::kC,   # 模板参数：指定操作数为C矩阵
      accum_t,                      # 模板参数：累加器的数据类型
      cutlass::layout::RowMajor,    # 模板参数：C矩阵的存储布局，这里是行主序
      P,                            # 模板参数：操作数P，通常表示矩阵的高度或深度
      1,                            # 模板参数：操作数Q，通常表示矩阵的宽度
      1>;                           # 模板参数：操作数K，通常表示矩阵的深度或高度

  using Iterator = AccumLambdaIteratorSimt<WarpIterator, accum_t, kWarpSize>;
                                    # 定义迭代器类型Iterator，使用累加器Lambda函数在Simt模式下迭代Warp中的数据
// 结构模板特化，用于定义在Volta架构上的TensorOp累加Lambda迭代器
template <typename S1, typename S2, typename accum_t, int kWarpSize>
struct DefaultMmaAccumLambdaIterator<
    cutlass::gemm::warp::MmaVoltaTensorOpAccumulatorTileIterator<
        S1,
        accum_t,
        cutlass::layout::RowMajor,
        S2,
        cutlass::MatrixShape<1, 1>>,
    accum_t,
    kWarpSize> {
  // 定义Warp迭代器类型，基于Volta架构的TensorOp累加器瓦片迭代器
  using WarpIterator =
      typename cutlass::gemm::warp::MmaVoltaTensorOpAccumulatorTileIterator<
          S1,
          accum_t,
          cutlass::layout::RowMajor,
          S2,
          cutlass::MatrixShape<1, 1>>;
  // 使用具体的迭代器类型定义为Sm70架构上的累加Lambda迭代器
  using Iterator = AccumLambdaIteratorSm70<WarpIterator, accum_t, kWarpSize>;
};

// 结构模板特化，用于定义在Sm75+架构上的TensorOp累加Lambda迭代器
template <
    typename S1,
    typename S2,
    typename S3,
    typename accum_t,
    int kWarpSize>
struct DefaultMmaAccumLambdaIterator<
    cutlass::gemm::warp::MmaTensorOpAccumulatorTileIterator<
        S1,
        accum_t,
        cutlass::layout::RowMajor,
        S2,
        S3>,
    accum_t,
    kWarpSize> {
  // 定义Warp迭代器类型，基于Sm75+架构的TensorOp累加器瓦片迭代器
  using WarpIterator =
      typename cutlass::gemm::warp::MmaTensorOpAccumulatorTileIterator<
          S1,
          accum_t,
          cutlass::layout::RowMajor,
          S2,
          S3>;
  // 使用具体的迭代器类型定义为Sm80架构上的累加Lambda迭代器
  using Iterator = AccumLambdaIteratorSm80<WarpIterator, accum_t, kWarpSize>;
};
```