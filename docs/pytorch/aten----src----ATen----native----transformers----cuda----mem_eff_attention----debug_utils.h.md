# `.\pytorch\aten\src\ATen\native\transformers\cuda\mem_eff_attention\debug_utils.h`

```
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once
#include <cfloat>
#include <cstdio>
#include <cmath>

////////////////////////////////////////////////////////////////////////////////
// Debugging functions
////////////////////////////////////////////////////////////////////////////////

// 宏定义 NANCHECK，用于检测数组 frag 中的 NaN 和无穷大
#define NANCHECK(frag)                         \
  {                                            \
    for (int _i = 0; _i < frag.size(); ++_i) { \
      assert(std::isfinite(float(frag[_i])));  \
      assert(!std::isnan(float(frag[_i])));    \
    }                                          \
  }

// 宏定义 PRINT_WARP_ID 和 PRINT_LANE_ID，用于指定打印线程的 warp 和 lane ID
#define PRINT_WARP_ID 0
#define PRINT_LANE_ID 0

// 宏定义 PRINT_B0_T0，当线程为第一个 block 的第一个线程时打印消息
#define PRINT_B0_T0(msg, ...)                                         \
  if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 &&        \
      threadIdx.x == PRINT_LANE_ID && threadIdx.y == PRINT_WARP_ID && \
      threadIdx.z == 0) {                                             \
    printf(msg "\n", ##__VA_ARGS__);                                  \
  }

// 宏定义 PRINT_T0，当线程为指定 warp 和 lane ID 的线程时打印消息
#define PRINT_T0(msg, ...)                                            \
  if (threadIdx.x == PRINT_LANE_ID && threadIdx.y == PRINT_WARP_ID && \
      threadIdx.z == 0) {                                             \
    printf(msg "\n", ##__VA_ARGS__);                                  \
  }

// 宏定义 PRINT_TX_LX，用于循环遍历每个 block 中的线程，以打印消息
#define PRINT_TX_LX(msg, ...)                                                 \
  for (int bx = 0; bx < gridDim.x; ++bx) {                                    \
    // 外层循环遍历 gridDim.y 维度
    for (int by = 0; by < gridDim.y; ++by) {                                  
      // 中层循环遍历 gridDim.z 维度
      for (int bz = 0; bz < gridDim.z; ++bz) {                                
        // 内层循环遍历 blockDim.x 维度
        for (int tx = 0; tx < blockDim.x; ++tx) {                             
          // 第四层循环遍历 blockDim.y 维度
          for (int ty = 0; ty < blockDim.y; ++ty) {                           
            // 最内层循环遍历 blockDim.z 维度
            for (int tz = 0; tz < blockDim.z; ++tz) {                         
              // 同步所有线程，等待直到所有线程都执行到这里
              __syncthreads();                                                
              // 如果当前线程的块索引和线程索引与给定的索引匹配，则执行打印操作
              if (blockIdx.x == bx && blockIdx.y == by && blockIdx.z == bz && 
                  threadIdx.x == tx && threadIdx.y == ty &&                   
                  threadIdx.z == tz) {                                        
                // 使用 printf 打印带有格式化的消息
                printf(                                                       
                    "[%d,%d,%d][%d,%d,%d]" msg "\n",                          
                    bx,                                                       
                    by,                                                       
                    bz,                                                       
                    tx,                                                       
                    ty,                                                       
                    tz,                                                       
                    ##__VA_ARGS__);                                           
              }                                                               
            }                                                                 
          }                                                                   
        }                                                                     
      }                                                                       
    }                                                                         
  }
// 如果未定义 PRINT_B0_T0 和 PRINT_TX_LX 宏，则定义 PRINT_B0_T0 宏
// 并使用给定格式化字符串和参数打印输出到标准输出
#define PRINT_B0_T0                                    \
  if constexpr (true) {                                \
    printf("TODO");                                    \
  } else

// 如果未定义 PRINT_B0_T0 和 PRINT_TX_LX 宏，则定义 PRINT_TX_LX 宏
// 并使用给定格式化字符串和参数打印输出到标准输出
#define PRINT_TX_LX                                    \
  if constexpr (true) {                                \
    printf("TODO");                                    \
  } else

// 表示一个字符串视图结构体，包含指向数据和大小的指针
struct __string_view {
  char const* data;         // 指向字符数据的指针
  std::size_t size;         // 字符数据的大小
};

// 如果编译器版本支持 C++14 或更高版本，则定义一个模板函数 __get_type_name
// 返回类型名称的字符串视图
#if __cplusplus >= 201402L
template <class T>
constexpr __string_view __get_type_name() {
  char const* p = __PRETTY_FUNCTION__;  // 获取当前函数的字符串表示
  while (*p++ != '=')                  // 跳过等号之前的字符
    ;
  for (; *p == ' '; ++p)               // 跳过空格字符
    ;
  char const* p2 = p;
  int count = 1;
  for (;; ++p2) {
    switch (*p2) {
      case '[':
        ++count;                       // 计数增加
        break;
      case ']':
        --count;                       // 计数减少
        if (!count)
          return {p, std::size_t(p2 - p)};  // 返回类型名称的字符串视图
    }
  }
  return {};
}
// 如果不支持 C++14 或更高版本，则定义一个模板函数 __get_type_name
// 返回字符串视图 "unsupported" 和大小 11
#else
template <class T>
constexpr __string_view __get_type_name() {
  return {"unsupported", 11};          // 返回固定字符串视图
}
#endif

// 定义宏 PRINT_ACCUM8_T0_L0_START，用于打印给定数组的前 8 个元素
#define PRINT_ACCUM8_T0_L0_START(name, accum, start)  \
  PRINT_B0_T0(                                        \
      "%s[%d:%d] - {%f, %f, %f, %f, %f, %f, %f, %f}", \
      name,                                           \
      int(start),                                     \
      int(start + 8),                                 \
      float(accum[start + 0]),                        \
      float(accum[start + 1]),                        \
      float(accum[start + 2]),                        \
      float(accum[start + 3]),                        \
      float(accum[start + 4]),                        \
      float(accum[start + 5]),                        \
      float(accum[start + 6]),                        \
      float(accum[start + 7]));                      // 打印数组指定范围内的元素

// 定义宏 PRINT_ACCUM8_T0_L0，使用 PRINT_ACCUM8_T0_L0_START 宏
// 打印给定数组 accum 的所有元素
#define PRINT_ACCUM8_T0_L0(name, accum) PRINT_ACCUM8_T0_L0_START(name, accum, 0)

// 定义宏 PRINT_FRAG_T0_L0，打印给定容器 frag 的内容
#define PRINT_FRAG_T0_L0(name, frag)                          \
  {                                                           \
    auto typeStr = __get_type_name<decltype(frag)>();         // 获取 frag 的类型名称
    PRINT_B0_T0("printing %s (%s)", name, typeStr.data);      // 打印容器名称及其类型
    for (int _start = 0; _start < frag.size(); _start += 8) { \
      PRINT_ACCUM8_T0_L0_START("  ", frag, _start);           // 循环打印容器内每 8 个元素
    }                                                         \
    /*__syncthreads();                                        \
    NANCHECK(frag); */                                        \
  }

// 定义宏 PRINT_ARRAY_T0_L0_INCR，打印给定数组的内容，每 incr 步长打印一次
#define PRINT_ARRAY_T0_L0_INCR(name, array, length, incr)   \
  {                                                         \
    PRINT_B0_T0("printing %s (len=%d)", name, int(length)); \
    for (int _start = 0; _start < length; _start += incr) { \
      PRINT_ACCUM8_T0_L0_START("  ", array, _start);        // 循环打印数组的每 incr 个元素
    }                                                       \
  }

// 定义宏 PRINT_ARRAY_T0_L0，使用 PRINT_ARRAY_T0_L0_INCR 宏
// 打印给定数组的内容，每 8 个元素打印一次
#define PRINT_ARRAY_T0_L0(name, array, length) \
  PRINT_ARRAY_T0_L0_INCR(name, array, length, 8)

// 打印一个 4x4 矩阵
# 定义宏 PRINT_TENSOR4x4_T0_L0_START，用于打印一个4x4张量的部分内容
#define PRINT_TENSOR4x4_T0_L0_START(name, ref, start_x, start_y)                                           \
  PRINT_B0_T0(                                                                                             \
      "%s[%d:%d, %d:%d]:\n    %f, %f, %f, %f\n    %f, %f, %f, %f\n    %f, %f, %f, %f\n    %f, %f, %f, %f", \
      name,                                                                                                \  # 打印张量的名称
      int(start_x),                                                                                        \  # 张量切片的起始 X 坐标
      int(start_x + 4),                                                                                    \  # 张量切片的结束 X 坐标
      int(start_y),                                                                                        \  # 张量切片的起始 Y 坐标
      int(start_y + 4),                                                                                    \  # 张量切片的结束 Y 坐标
      float(ref.at({start_x + 0, start_y + 0})),                                                           \  # 访问张量在指定位置的元素值
      float(ref.at({start_x + 0, start_y + 1})),                                                           \
      float(ref.at({start_x + 0, start_y + 2})),                                                           \
      float(ref.at({start_x + 0, start_y + 3})),                                                           \
      float(ref.at({start_x + 1, start_y + 0})),                                                           \
      float(ref.at({start_x + 1, start_y + 1})),                                                           \
      float(ref.at({start_x + 1, start_y + 2})),                                                           \
      float(ref.at({start_x + 1, start_y + 3})),                                                           \
      float(ref.at({start_x + 2, start_y + 0})),                                                           \
      float(ref.at({start_x + 2, start_y + 1})),                                                           \
      float(ref.at({start_x + 2, start_y + 2})),                                                           \
      float(ref.at({start_x + 2, start_y + 3})),                                                           \
      float(ref.at({start_x + 3, start_y + 0})),                                                           \
      float(ref.at({start_x + 3, start_y + 1})),                                                           \
      float(ref.at({start_x + 3, start_y + 2})),                                                           \
      float(ref.at({start_x + 3, start_y + 3})));                                                          \  # 访问张量在指定位置的元素值，并打印出来

# 定义宏 PRINT_TENSOR4x4_T0_L0，用于打印一个从左上角开始的4x4张量的内容
#define PRINT_TENSOR4x4_T0_L0(name, ref) \
  PRINT_TENSOR4x4_T0_L0_START(name, ref, 0, 0)

# 定义宏 PRINT_PROBLEM_SIZE，用于打印问题的大小，包括 m、n、k 三个维度
#define PRINT_PROBLEM_SIZE(name, ps)            \
  PRINT_B0_T0(                                  \  # 打印问题大小的格式化字符串
      "%s.problem_size: {.m=%d, .n=%d, .k=%d}", \  # 格式化输出问题名字以及各维度的大小
      name,                                     \  # 问题的名称
      int(ps.m()),                              \  # 获取问题的 m 维度大小
      int(ps.n()),                              \  # 获取问题的 n 维度大小
      int(ps.k()))                              \  # 获取问题的 k 维度大小
// 定义一个函数，用于在 CUDA warp 中打印累积结果
template <typename LambdaIterator, typename LaneOffsetT, typename AccumT>
CUTLASS_DEVICE void print_warp_accum(
    AccumT accum,                // 累积结果数组
    LaneOffsetT lane_offset,     // warp 中的 lane 偏移量
    int32_t num_rows,            // 矩阵的行数
    int32_t num_cols) {          // 矩阵的列数

  // 检查当前线程是否为主线程（第一个线程块和第一个线程）
  bool is_main = blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 &&
      threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0;

  // 遍历矩阵的每一行
  for (int row = 0; row < num_rows; ++row) {
    // 遍历矩阵的每一列
    for (int col = 0; col < num_cols; ++col) {
      // 每当列数是32的倍数时
      if (col % 32 == 0) {
        // 如果是主线程，则打印换行及矩阵位置信息
        if (is_main) {
          printf("\nmat[%3d, %3d:%3d]", row, col, col + 32);
        }
        // 同步所有线程，确保前面的打印操作完成
        __syncthreads();
      }

      // 调用 LambdaIterator 类的 iterateRows 方法
      LambdaIterator::iterateRows(
          lane_offset,
          [&](int accum_m) {},                         // 空的 lambda 表达式
          [&](int accum_m, int accum_n, int idx) {      // 处理累积结果的 lambda 表达式
            // 如果当前位置是需要打印累积结果的位置，并且是第一个线程块
            if (row == accum_m && col == accum_n &&
                (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0)) {
              printf(" %6.1f", float(accum[idx]));      // 打印浮点数累积结果
            }
          },
          [&](int accum_m) {}                          // 空的 lambda 表达式
      );

      // 同步所有线程，确保前面的打印操作完成
      __syncthreads();
    }

    // 如果是主线程，则打印换行
    if (is_main) {
      printf("\n");
    }
  }
}
```