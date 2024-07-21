# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\qnnpack\pack.h`

```py
/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <qnnpack/math.h>
#include <stdint.h>

// Legend:
//  dq: Design-time Quantization
//  rq: Run-time Quantization

// 定义静态内联函数，用于量化矩阵权重数据，支持设计时和运行时量化
static inline void pytorch_pack_q8gemm_wdq(
    size_t nc,                      // 输出通道数量
    size_t kc,                      // 输入通道数量
    uint32_t nr,                    // 输出通道块大小
    uint32_t np,                    // 内部并行度
    uint32_t kr,                    // 输入通道块大小
    uint8_t izp,                    // 输入通道零点
    uint8_t kzp,                    // 权重零点
    const uint8_t* k,               // 权重数据指针
    const int32_t* b,               // 偏置数据指针
    void* packed_w                  // 输出的打包后的权重数据
) {
  const int32_t boff = (int32_t)kc * (int32_t)izp * (int32_t)kzp; // 计算偏置量

  // 遍历输出通道块
  for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
    const size_t nr_block_size = min(nc - nr_block_start, nr); // 计算当前输出通道块的大小
    int32_t* packed_b = (int32_t*)packed_w; // 将打包后的权重数据转换为int32_t指针

    // 填充偏置数据
    for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
      *((int32_t*)packed_w) = b ? b[nr_block_start + nr_block_offset] + boff : 0.0f; // 添加偏置量到打包的权重数据中
      packed_w = (void*)((uintptr_t)packed_w + sizeof(int32_t)); // 移动到下一个位置
    }

    packed_w = (void*)((uintptr_t)packed_w + (nr - nr_block_size) * sizeof(int32_t)); // 对齐打包的权重数据

    // 填充权重数据
    for (size_t kr_block_start = 0; kr_block_start < kc; kr_block_start += kr) {
      const size_t kr_block_size = min(kc - kr_block_start, kr); // 计算当前输入通道块的大小

      for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
        int32_t ksum = 0;

        // 累加权重数据并填充到打包的权重数据中
        for (size_t kr_block_offset = 0; kr_block_offset < kr_block_size; kr_block_offset++) {
          const uint8_t kv =
              k[(nr_block_start + nr_block_offset) * kc +
                (kr_block_start + kr_block_offset)]; // 获取当前权重数据
          ksum += (int32_t)kv; // 累加权重数据
          *((uint8_t*)packed_w) = kv; // 填充权重数据到打包的权重数据中
          packed_w = (void*)((uintptr_t)packed_w + sizeof(uint8_t)); // 移动到下一个位置
        }

        packed_b[nr_block_offset] -= ksum * (int32_t)izp; // 根据输入通道零点调整偏置数据
        packed_w = (void*)((uintptr_t)packed_w + (kr - kr_block_size) * sizeof(uint8_t)); // 对齐打包的权重数据
      }

      packed_w = (void*)((uintptr_t)packed_w + ((nr - nr_block_size) & (np - 1)) * kr * sizeof(uint8_t)); // 对齐打包的权重数据
    }
  }
}

// 注意事项: 动态量化和运行时量化线性函数使用相同的打包函数。
// 这意味着动态模式会因为引入 `if(kzp!=0)` 而产生一些性能损失，不过这不应该太显著。
// 定义静态内联函数，用于量化矩阵权重数据，支持运行时量化
static inline void pytorch_pack_q8gemm_wrq(
    const size_t nc,                // 输出通道数量
    const size_t kc,                // 输入通道数量
    const uint32_t nr,              // 输出通道块大小
    const uint32_t np,              // 内部并行度
    const uint32_t kr,              // 输入通道块大小
    const uint8_t* const k,         // 权重数据指针
    const int32_t* const b,         // 偏置数据指针
    const uint8_t* const kzp,       // 权重零点数据指针
    void* const packed_w            // 输出的打包后的权重数据
) {
  union {
    void* const as_void_ptr;
    uint8_t* as_uint8_ptr;
    int32_t* as_int32_ptr;
  } packed = {packed_w}; // 联合体，用于访问打包后的权重数据的不同视图

  // 遍历输出通道块
  for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
    const size_t nr_block_size = min(nc - nr_block_start, nr); // 计算当前输出通道块的大小

    // 填充偏置数据
    for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
      // 添加偏置量到打包的权重数据中
      *((int32_t*)packed_w) = b ? b[nr_block_start + nr_block_offset] + (int32_t)kc * (int32_t)kzp[nr_block_start + nr_block_offset] : 0.0f;
      packed_w = (void*)((uintptr_t)packed_w + sizeof(int32_t)); // 移动到下一个位置
    }

    packed_w = (void*)((uintptr_t)packed_w + (nr - nr_block_size) * sizeof(int32_t)); // 对齐打包的权重数据

    // 填充权重数据
    for (size_t kr_block_start = 0; kr_block_start < kc; kr_block_start += kr) {
      const size_t kr_block_size = min(kc - kr_block_start, kr); // 计算当前输入通道块的大小

      for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
        int32_t ksum = 0;

        // 累加权重数据并填充到打包的权重数据中
        for (size_t kr_block_offset = 0; kr_block_offset < kr_block_size; kr_block_offset++) {
          const uint8_t kv =
              k[(nr_block_start + nr_block_offset) * kc +
                (kr_block_start + kr_block_offset)]; // 获取当前权重数据
          ksum += (int32_t)kv; // 累加权重数据
          *((uint8_t*)packed_w) = kv; // 填充权重数据到打包的权重数据中
          packed_w = (void*)((uintptr_t)packed_w + sizeof(uint8_t)); // 移动到下一个位置
        }

        packed_w = (void*)((uintptr_t)packed_w + (kr - kr_block_size) * sizeof(uint8_t)); // 对齐打包的权重数据
      }

      packed_w = (void*)((uintptr_t)packed_w + ((nr - nr_block_size) & (np - 1)) * kr * sizeof(uint8_t)); // 对齐打包的权重数据
    }
  }
}
    // 对于每个 nr_block_offset，从 b 数组复制数据到 packed 中
    for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size;
         nr_block_offset++) {
      // 将 b 数组中的数据（如果存在）复制到 packed.as_int32_ptr 指向的位置，否则写入零值
      *(packed.as_int32_ptr++) = b ? b[nr_block_start + nr_block_offset] : 0;
    }
    // 跳过 (nr - nr_block_size) 个位置，以便进行下一个 kr_block 的处理
    packed.as_int32_ptr += (nr - nr_block_size);
    
    // 通过 kr_block_start 对 kr_block 进行循环处理，步长为 kr
    for (size_t kr_block_start = 0; kr_block_start < kc; kr_block_start += kr) {
      const size_t kr_block_size = min(kc - kr_block_start, kr);
      
      // 对于每个 nr_block_offset，对 kr_block_offset 进行循环处理
      for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size;
           nr_block_offset++) {
        for (size_t kr_block_offset = 0; kr_block_offset < kr_block_size;
             kr_block_offset++) {
          // 获取 k 数组中的值，并将其存储到 packed.as_uint8_ptr 指向的位置
          const uint8_t kv =
              k[(nr_block_start + nr_block_offset) * kc +
                (kr_block_start + kr_block_offset)];
          *(packed.as_uint8_ptr++) = kv;
        }
        
        // 当存在 kzp 时，用 kzp 中的值填充尾部，以保证 packed 块的大小不是输入大小的倍数时也能正确处理
        if (kzp != 0) {
          for (size_t kr_block_offset = 0; kr_block_offset < (kr - kr_block_size);
               kr_block_offset++) {
            const uint8_t kv =
                kzp[(nr_block_start + nr_block_offset)];
            *(packed.as_uint8_ptr++) = kv;
          }
        } else {
          // 否则跳过 (kr - kr_block_size) 个位置
          packed.as_uint8_ptr += (kr - kr_block_size);
        }
      }
      
      // 当存在 kzp 时，填充 packed.weights 以保证输出通道不可被 nr 块参数整除时，依然能正确运行
      if (kzp != 0) {
        size_t remaining_nr_blocks = ((nr - nr_block_size) & (np - 1));
        for (size_t nr_block_offset = 0; nr_block_offset < remaining_nr_blocks;
             nr_block_offset++) {
          for (size_t kr_block_offset = 0; kr_block_offset < kr;
               kr_block_offset++) {
            const uint8_t kv =
                kzp[(nr_block_start + nr_block_size + nr_block_offset)];
            *(packed.as_uint8_ptr++) = kv;
          }
        }
      } else {
        // 否则跳过 ((nr - nr_block_size) & (np - 1)) * kr 个位置
        packed.as_uint8_ptr += ((nr - nr_block_size) & (np - 1)) * kr;
      }
    }
}

static inline void pytorch_pack_q8conv_wdq(
    size_t n,                              // 参数：输出通道的总数
    size_t ks,                             // 参数：卷积核的尺寸
    size_t kc,                             // 参数：输入通道的总数
    uint32_t nr,                            // 参数：输出通道的分块大小
    uint32_t kr,                            // 参数：卷积核的分块大小
    uint8_t izp,                            // 参数：输入零点
    uint8_t kzp,                            // 参数：卷积核零点
    const uint8_t* k,                      // 参数：卷积核数据的指针
    const int32_t* b,                      // 参数：偏置数据的指针
    void* packed_w) {                      // 参数：打包后的权重数据的指针
  const int32_t boff = (int32_t)ks * (int32_t)kc * (int32_t)izp * (int32_t)kzp;  // 计算偏置值
  for (size_t nr_block_start = 0; nr_block_start < n; nr_block_start += nr) {    // 循环：遍历输出通道
    const size_t nr_block_size = min(n - nr_block_start, nr);                    // 计算当前循环中的输出通道块大小
    int32_t* packed_b = (int32_t*)packed_w;                                      // 将打包后的权重数据转换为 int32_t 指针
    for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size;
         nr_block_offset++) {
      *((int32_t*)packed_w) = b ? b[nr_block_start + nr_block_offset] + boff : 0.0f;  // 计算并存储偏置值
      packed_w = (void*)((uintptr_t)packed_w + sizeof(int32_t));                 // 更新打包后的权重数据指针
    }
    packed_w =
        (void*)((uintptr_t)packed_w + (nr - nr_block_size) * sizeof(int32_t));    // 跳过未使用的权重数据空间
    for (size_t ki = 0; ki < ks; ki++) {                                          // 循环：遍历卷积核尺寸
      for (size_t kr_block_start = 0; kr_block_start < kc;
           kr_block_start += kr) {
        const size_t kr_block_size = min(kc - kr_block_start, kr);                // 计算当前循环中的卷积核块大小
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size;
             nr_block_offset++) {
          int32_t ksum = 0;
          for (size_t kr_block_offset = 0; kr_block_offset < kr_block_size;
               kr_block_offset++) {
            const uint8_t kv =
                k[((nr_block_start + nr_block_offset) * ks + ki) * kc +
                  (kr_block_start + kr_block_offset)];                           // 访问卷积核数据
            ksum += (int32_t)kv;                                                  // 计算卷积核数据的和
            *((uint8_t*)packed_w) = kv;                                           // 存储卷积核数据
            packed_w = (void*)((uintptr_t)packed_w + sizeof(uint8_t));           // 更新打包后的权重数据指针
          }
          packed_b[nr_block_offset] -= ksum * (int32_t)izp;                       // 更新偏置值
          packed_w =
              (void*)((uintptr_t)packed_w + (kr - kr_block_size) * sizeof(uint8_t));  // 跳过未使用的权重数据空间
        }
        packed_w =
            (void*)((uintptr_t)packed_w + (nr - nr_block_size) * kr * sizeof(uint8_t));  // 跳过未使用的权重数据空间
      }
    }
  }
}

static inline void pytorch_pack_q8conv_wrq(
    const size_t n,                        // 参数：输出通道的总数
    const size_t ks,                       // 参数：卷积核的尺寸
    const size_t kc,                       // 参数：输入通道的总数
    const uint32_t nr,                      // 参数：输出通道的分块大小
    const uint32_t kr,                      // 参数：卷积核的分块大小
    const uint8_t* const k,                // 参数：卷积核数据的指针
    const int32_t* const b,                // 参数：偏置数据的指针
    const uint8_t* const kzp,              // 参数：卷积核零点数据的指针
    void* const packed_w) {                // 参数：打包后的权重数据的指针
  union {
    void* const as_void_ptr;
    uint8_t* as_uint8_ptr;
    int32_t* as_int32_ptr;
  } packed = {packed_w};                   // 声明并初始化打包后的权重数据联合体

  for (size_t nr_block_start = 0; nr_block_start < n; nr_block_start += nr) {    // 循环：遍历输出通道
    const size_t nr_block_size = min(n - nr_block_start, nr);                    // 计算当前循环中的输出通道块大小
    for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size;
         nr_block_offset++) {
      *(packed.as_int32_ptr++) = b ? b[nr_block_start + nr_block_offset] : 0.0f;  // 存储偏置值
    }
    packed.as_int32_ptr += (nr - nr_block_size);                                 // 跳过未使用的权重数据空间
    // 外层循环遍历输入通道数 ki
    for (size_t ki = 0; ki < ks; ki++) {
      // 中层循环遍历内核的列数，以块为单位，每次处理 kr 个列
      for (size_t kr_block_start = 0; kr_block_start < kc;
           kr_block_start += kr) {
        // 计算当前内核块的实际大小，可能不足 kr 列
        const size_t kr_block_size = min(kc - kr_block_start, kr);
        // 内层循环遍历输出通道数的块偏移量
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size;
             nr_block_offset++) {
          // 最内层循环遍历当前内核块的列偏移量
          for (size_t kr_block_offset = 0; kr_block_offset < kr_block_size;
               kr_block_offset++) {
            // 计算当前位置的权重值 kv
            const uint8_t kv =
                k[((nr_block_start + nr_block_offset) * ks + ki) * kc +
                  (kr_block_start + kr_block_offset)];
            // 将权重值 kv 打包到 packed 中
            *(packed.as_uint8_ptr++) = kv;
          }
          // 权重需要与零点预先打包在一起，位于块尾部，块大小不是输入大小的倍数时
          // 例如，对于 kr=2 且 k 为 3，则第二块必须用零点填充。
          // 这是因为减去零点时，填充的值将得到零，这是我们想要的结果。
          if (kzp != 0) {
            // 如果存在零点，填充当前块剩余的 kr 列
            for (size_t kr_block_offset = 0; kr_block_offset < (kr - kr_block_size);
                 kr_block_offset++) {
              const uint8_t kv =
                  kzp[(nr_block_start + nr_block_offset)];
              *(packed.as_uint8_ptr++) = kv;
            }
          } else {
            // 否则跳过当前块剩余的 kr 列
            packed.as_uint8_ptr += (kr - kr_block_size);
          }
        }
        // 如果存在零点，填充输出通道剩余的 nr 块
        if (kzp != 0) {
          for (size_t nr_block_offset = 0; nr_block_offset < (nr - nr_block_size);
               nr_block_offset++) {
            // 填充未被处理的输出通道块
            for (size_t kr_block_offset = 0; kr_block_offset < kr;
                 kr_block_offset++) {
              const uint8_t kv =
                  kzp[(nr_block_start + nr_block_size + nr_block_offset)];
              *(packed.as_uint8_ptr++) = kv;
            }
          }
        } else {
          // 否则跳过输出通道剩余的 nr 块
          packed.as_uint8_ptr += (nr - nr_block_size) * kr;
        }
      }
    }
}

// 定义一个静态内联函数，用于压缩 Q8 反卷积权重和偏置数据
static inline void pytorch_pack_q8deconv_wdq(
    size_t n,                    // 输入数据的数量
    size_t ks,                   // 内核尺寸（kernel size）
    size_t kc,                   // 输入通道数
    uint32_t nr,                  // 行块大小
    uint32_t kr,                  // 内核块大小
    uint8_t izp,                 // 输入零点
    uint8_t kzp,                 // 内核零点
    const uint8_t* k,            // 内核数据指针
    const int32_t* b,            // 偏置数据指针
    void* packed_w) {            // 输出压缩权重数据的指针

  // 计算偏置偏移量
  const int32_t boff = (int32_t)ks * (int32_t)kc * (int32_t)izp * (int32_t)kzp;

  // 对每个行块进行循环处理
  for (size_t nr_block_start = 0; nr_block_start < n; nr_block_start += nr) {
    // 计算当前行块的实际大小
    const size_t nr_block_size = min(n - nr_block_start, nr);

    // 将偏置数据存入压缩权重中
    int32_t* packed_b = (int32_t*)packed_w;
    for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size;
         nr_block_offset++) {
      *((int32_t*)packed_w) = b ? b[nr_block_start + nr_block_offset] + boff : 0.0f;
      packed_w = (void*)((uintptr_t)packed_w + sizeof(int32_t));
    }

    // 调整指针位置，确保下一步能正确存储数据
    packed_w =
        (void*)((uintptr_t)packed_w + (nr - nr_block_size) * sizeof(int32_t));

    // 对每个内核块进行循环处理
    for (size_t ki = 0; ki < ks; ki++) {
      for (size_t kr_block_start = 0; kr_block_start < kc;
           kr_block_start += kr) {
        const size_t kr_block_size = min(kc - kr_block_start, kr);

        // 对每个行块进行循环处理
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size;
             nr_block_offset++) {
          int32_t ksum = 0;

          // 对每个内核块进行循环处理
          for (size_t kr_block_offset = 0; kr_block_offset < kr_block_size;
               kr_block_offset++) {
            const uint8_t kv =
                k[((kr_block_start + kr_block_offset) * ks + ki) * n +
                  (nr_block_start + nr_block_offset)];
            ksum += (int32_t)kv;
            *((uint8_t*)packed_w) = kv;
            packed_w = (void*)((uintptr_t)packed_w + sizeof(uint8_t));
          }

          // 调整偏置数据，确保下一步能正确存储数据
          packed_b[nr_block_offset] -= ksum * (int32_t)izp;
          packed_w =
              (void*)((uintptr_t)packed_w + (kr - kr_block_size) * sizeof(uint8_t));
        }

        // 调整指针位置，确保下一步能正确存储数据
        packed_w =
            (void*)((uintptr_t)packed_w + (nr - nr_block_size) * kr * sizeof(uint8_t));
      }
    }
  }
}

// 定义一个静态内联函数，用于压缩 Q8 反卷积权重和偏置数据，支持偏置为零情况
static inline void pytorch_pack_q8deconv_wrq(
    const size_t n,              // 输入数据的数量
    const size_t ks,             // 内核尺寸（kernel size）
    const size_t kc,             // 输入通道数
    const uint32_t nr,           // 行块大小
    const uint32_t kr,           // 内核块大小
    const uint8_t* const k,      // 内核数据指针
    const int32_t* const b,      // 偏置数据指针
    const uint8_t* const kzp,    // 内核零点数据指针
    void* const packed_w) {      // 输出压缩权重数据的指针

  // 使用联合类型以便于在不同类型间进行指针转换
  union {
    void* const as_void_ptr;     // 作为 void* 类型的指针
    uint8_t* as_uint8_ptr;       // 作为 uint8_t* 类型的指针
    int32_t* as_int32_ptr;       // 作为 int32_t* 类型的指针
  } packed = {packed_w};

  // 对每个行块进行循环处理
  for (size_t nr_block_start = 0; nr_block_start < n; nr_block_start += nr) {
    // 计算当前行块的实际大小
    const size_t nr_block_size = min(n - nr_block_start, nr);

    // 将偏置数据存入压缩权重中
    for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size;
         nr_block_offset++) {
      *(packed.as_int32_ptr++) = b ? b[nr_block_start + nr_block_offset] : 0.0f;
    }

    // 调整指针位置，确保下一步能正确存储数据
    packed.as_int32_ptr += (nr - nr_block_size);
    // 外层循环遍历输入通道的索引 ki
    for (size_t ki = 0; ki < ks; ki++) {
      // 中间循环遍历输出通道块的起始索引 kr_block_start
      for (size_t kr_block_start = 0; kr_block_start < kc;
           kr_block_start += kr) {
        // 计算当前输出通道块的大小 kr_block_size
        const size_t kr_block_size = min(kc - kr_block_start, kr);
        // 内部循环遍历输出通道块的每个元素在输出通道的偏移量 nr_block_offset
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size;
             nr_block_offset++) {
          // 最内层循环遍历当前输出通道块内部的权重元素在输入通道上的偏移量 kr_block_offset
          for (size_t kr_block_offset = 0; kr_block_offset < kr_block_size;
               kr_block_offset++) {
            // 计算权重 k 的索引并获取对应的值 kv
            const uint8_t kv =
                k[((kr_block_start + kr_block_offset) * ks + ki) * n +
                  (nr_block_start + nr_block_offset)];
            // 将权重值 kv 存入打包后的数据结构 packed 中
            *(packed.as_uint8_ptr++) = kv;
          }
          // 如果存在非零点数组 kzp，则用零点填充权重块尾部，以满足块大小要求
          if (kzp != 0) {
            // 遍历填充剩余部分的权重块
            for (size_t kr_block_offset = 0; kr_block_offset < (kr - kr_block_size);
                 kr_block_offset++) {
              const uint8_t kv =
                  kzp[(nr_block_start + nr_block_offset)];
              // 将零点值 kv 存入 packed 中
              *(packed.as_uint8_ptr++) = kv;
            }
          } else {
            // 如果不存在非零点数组 kzp，则直接移动 packed 的指针位置
            packed.as_uint8_ptr += (kr - kr_block_size);
          }
        }
        // 如果存在非零点数组 kzp，则用零点填充输出通道尾部，以满足块大小要求
        if (kzp != 0) {
          // 遍历填充剩余部分的输出通道块
          for (size_t nr_block_offset = 0; nr_block_offset < (nr - nr_block_size);
               nr_block_offset++) {
            // 遍历当前输出通道块的每个元素在输出通道上的偏移量
            for (size_t kr_block_offset = 0; kr_block_offset < kr;
                 kr_block_offset++) {
              const uint8_t kv =
                  kzp[(nr_block_start + nr_block_size + nr_block_offset)];
              // 将零点值 kv 存入 packed 中
              *(packed.as_uint8_ptr++) = kv;
            }
          }
        } else {
          // 如果不存在非零点数组 kzp，则直接移动 packed 的指针位置
          packed.as_uint8_ptr += (nr - nr_block_size) * kr;
        }
      }
    }
}

// 压缩函数 pytorch_pack_q8dw_wdq，将权重数据压缩为 int32_t 和 uint8_t 格式
static inline void pytorch_pack_q8dw_wdq(
    size_t h,                        // 高度
    size_t w,                        // 宽度
    size_t c,                        // 通道数
    size_t cr,                       // 压缩率
    uint8_t izp,                     // 输入零点
    uint8_t* kzp,                    // 压缩因子指针
    const uint8_t* k,                // 权重数据指针
    const int32_t* b,                // 偏置数据指针
    void* packed_w) {                // 压缩后的权重数据
  const int32_t boff = (int32_t)h * (int32_t)w * (int32_t)izp;  // 计算偏置值
  for (size_t cr_block_start = 0; cr_block_start < c; cr_block_start += cr) {
    const size_t cr_block_size = min(c - cr_block_start, cr);   // 当前压缩块的大小
    int32_t* packed_b = (int32_t*)packed_w;                     // 转换为 int32_t 指针
    for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size;
         cr_block_offset++) {
      *((int32_t*)packed_w) =   // 填充压缩后的偏置值
        b ?
            b[cr_block_start + cr_block_offset] +
            boff * kzp[cr_block_start + cr_block_offset] : 0.0f;
      packed_w = (void*)((uintptr_t)packed_w + sizeof(int32_t));  // 更新 packed_w 指针位置
    }
    packed_w =
        (void*)((uintptr_t)packed_w + (cr - cr_block_size) * sizeof(int32_t));  // 更新 packed_w 指针位置
    for (size_t x = 0; x < w; x++) {
      for (size_t y = 0; y < h; y++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size;
             cr_block_offset++) {
          const uint8_t kv =
              k[((cr_block_start + cr_block_offset) * h + y) * w + x];   // 获取权重数据
          packed_b[cr_block_offset] -= (int32_t)kv * (int32_t)izp;       // 更新压缩后的偏置值
          *((uint8_t*)packed_w) = kv;                                    // 填充压缩后的权重数据
          packed_w = (void*)((uintptr_t)packed_w + sizeof(uint8_t));     // 更新 packed_w 指针位置
        }
        packed_w =
            (void*)((uintptr_t)packed_w + (cr - cr_block_size) * sizeof(uint8_t));  // 更新 packed_w 指针位置
      }
    }
  }
}

// 压缩函数 pytorch_pack_q8dw_wrq，将权重数据压缩为 int32_t 和 uint8_t 格式
static inline void pytorch_pack_q8dw_wrq(
    const size_t h,                  // 高度
    const size_t w,                  // 宽度
    const size_t c,                  // 通道数
    const size_t cr,                 // 压缩率
    const uint8_t* const k,          // 权重数据指针
    const int32_t* const b,          // 偏置数据指针
    void* const packed_w) {          // 压缩后的权重数据
  union {
    void* const as_void_ptr;         // 作为 void 指针
    uint8_t* as_uint8_ptr;           // 作为 uint8_t 指针
    int32_t* as_int32_ptr;           // 作为 int32_t 指针
  } packed = {packed_w};             // 初始化 union

  for (size_t cr_block_start = 0; cr_block_start < c; cr_block_start += cr) {
    const size_t cr_block_size = min(c - cr_block_start, cr);   // 当前压缩块的大小
    for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size;
         cr_block_offset++) {
      *(packed.as_int32_ptr++) = b ? b[cr_block_start + cr_block_offset] : 0.0f;  // 填充压缩后的偏置值
    }
    packed.as_int32_ptr += (cr - cr_block_size);  // 更新 packed_w 指针位置
    for (size_t x = 0; x < w; x++) {
      for (size_t y = 0; y < h; y++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size;
             cr_block_offset++) {
          const uint8_t kv =
              k[((cr_block_start + cr_block_offset) * h + y) * w + x];   // 获取权重数据
          *(packed.as_uint8_ptr++) = kv;                                  // 填充压缩后的权重数据
        }
        packed.as_uint8_ptr += (cr - cr_block_size);  // 更新 packed_w 指针位置
      }
    }
  }
}

// 压缩函数 pytorch_pack_q8dw_3d_w_dilation，将 3D 权重数据压缩为 int32_t 和 uint8_t 格式
static inline void pytorch_pack_q8dw_3d_w_dilation(
    size_t d,                        // 深度
    size_t h,                        // 高度
    size_t w,                        // 宽度
    size_t c,                        // 通道数
    size_t cr,                       // 压缩率
    size_t z_start,                  // 起始深度
    size_t z_end,                    // 结束深度
    size_t y_start,                  // 起始高度
    size_t y_end,                    // 结束高度
    size_t x_start,                  // 起始宽度
    size_t x_end,                    // 结束宽度
    const uint8_t* k,                // 权重数据指针
    const int32_t* b,                // 偏置数据指针
    void* packed_w,                  // 压缩后的权重数据
    bool pytorch_pack_b) {           // 是否进行 PyTorch 压缩
  for (size_t cr_block_start = 0; cr_block_start < c; cr_block_start += cr) {
    // 计算当前块的大小，取最小值以防止越界
    const size_t cr_block_size = min(c - cr_block_start, cr);
    
    // 如果需要使用 PyTorch 打包，则进行以下操作
    if (pytorch_pack_b) {
      // 遍历当前块的每个元素
      for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size;
           cr_block_offset++) {
        // 将 b 数组中的值复制到 packed_w 中，如果 b 为空则复制 0.0f
        *((int32_t*)packed_w) = b ? b[cr_block_start + cr_block_offset] : 0.0f;
        // 移动 packed_w 到下一个元素位置
        packed_w = (void*)((int32_t*)packed_w + 1);
      }
      // 跳过未使用的元素，以确保 packed_w 指向下一块的起始位置
      packed_w =
          (void*)((int32_t*)packed_w + (cr - cr_block_size));
    }
    
    // 遍历 x, y, z 范围内的所有坐标
    for (size_t x = x_start; x < x_end; x++) {
      for (size_t y = y_start; y < y_end; y++) {
        for (size_t z = z_start; z < z_end; z++) {
          // 遍历当前块的每个元素
          for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size;
               cr_block_offset++) {
            // 将 k 数组中的值复制到 packed_w 中，计算索引根据当前块的偏移量和 x, y, z 坐标
            *((uint8_t*)packed_w) =
                k[(((cr_block_start + cr_block_offset) * d + z) * h + y) * w + x];
            // 移动 packed_w 到下一个元素位置
            packed_w = (void*)((uint8_t*)packed_w + 1);
          }
          // 跳过未使用的元素，以确保 packed_w 指向下一块的起始位置
          packed_w =
              (void*)((uint8_t*)packed_w + (cr - cr_block_size));
        }
      }
    }
}

static inline void pytorch_pack_q8dw_2d_w_dilation(
    size_t h,                        // 输入图像的高度
    size_t w,                        // 输入图像的宽度
    size_t c,                        // 输入通道数
    size_t cr,                       // 压缩后的输出通道数
    size_t y_start,                  // 垂直方向的起始位置
    size_t y_end,                    // 垂直方向的结束位置
    size_t x_start,                  // 水平方向的起始位置
    size_t x_end,                    // 水平方向的结束位置
    const uint8_t* k,                // 卷积核数据
    const int32_t* b,                // 偏置数据
    void* packed_w,                  // 压缩后的权重数据
    bool pytorch_pack_b) {           // 是否对偏置进行压缩
  pytorch_pack_q8dw_3d_w_dilation(
      1,                             // 深度设为1，表示二维操作
      h,
      w,
      c,
      cr,
      0,                             // 深度方向的起始位置为0
      1,                             // 深度方向的结束位置为1
      y_start,
      y_end,
      x_start,
      x_end,
      k,
      b,
      packed_w,
      pytorch_pack_b);
}

static inline void pytorch_pack_swizzle_q8gemm_bdq(
    size_t n,                        // 批处理大小
    size_t kc,                       // 卷积核通道数
    uint32_t nr,                     // 矩阵行数
    uint32_t kr,                     // 矩阵列数
    uint32_t sr,                     // 步长
    uint8_t izp,                     // 输入零点
    uint8_t kzp,                     // 卷积核零点
    const uint8_t* k,                // 卷积核数据
    const int32_t* b,                // 偏置数据
    void* packed_w) {                // 压缩后的权重数据
  const int32_t boff = (int32_t)kc * (int32_t)izp * (int32_t)kzp;
  for (size_t nr_block_start = 0; nr_block_start < n; nr_block_start += nr) {
    const size_t nr_block_size = min(n - nr_block_start, nr);
    int32_t* packed_b = (int32_t*)packed_w;
    for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size;
         nr_block_offset++) {
      *((int32_t*)packed_w) = b ? b[nr_block_start + nr_block_offset] + boff : 0.0f;  // 计算并存储压缩后的偏置值
      packed_w = (void*)((uintptr_t)packed_w + sizeof(int32_t));
    }
    packed_w =
        (void*)((uintptr_t)packed_w + (nr - nr_block_size) * sizeof(int32_t));  // 偏置数据对齐处理

    for (size_t kr_block_start = 0; kr_block_start < (kc & -sr);
         kr_block_start += kr) {
      for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size;
           nr_block_offset++) {
        for (size_t kr_block_offset = 0; kr_block_offset < kr;
             kr_block_offset++) {
          const uint8_t kv =
              k[(nr_block_start + nr_block_offset) * kc +
                (kr_block_start & -sr) +
                ((kr_block_start + nr_block_offset * kr) & (sr - 1)) +
                kr_block_offset];  // 计算卷积核值并存储
          packed_b[nr_block_offset] -= (int32_t)kv * (int32_t)izp;  // 压缩偏置值
          *((uint8_t*)packed_w) = kv;
          packed_w = (void*)((uintptr_t)packed_w + sizeof(uint8_t));
        }
      }
      packed_w =
          (void*)((uintptr_t)packed_w + (nr - nr_block_size) * kr * sizeof(uint8_t));  // 压缩后的权重数据对齐处理
    }
    // 外层循环，从 kc 的最小整数倍开始，递增 kr，直至小于 kc
    for (size_t kr_block_start = (kc & -sr); kr_block_start < kc;
         kr_block_start += kr) {
      // 计算当前 kr 块的大小，取 kc 减去 kr 块开始位置与 kr 的较小值
      const size_t kr_block_size = min(kc - kr_block_start, kr);
      // 循环处理 nr_block_size 大小的 nr 块偏移量
      for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size;
           nr_block_offset++) {
        // 循环处理当前 kr 块大小的 kr 块偏移量
        for (size_t kr_block_offset = 0; kr_block_offset < kr_block_size;
             kr_block_offset++) {
          // 计算出 k 中当前位置的值 kv
          const uint8_t kv =
              k[(nr_block_start + nr_block_offset) * kc +
                (kr_block_start + kr_block_offset)];
          // 更新 packed_b 中的值，减去 kv 乘以 izp 的结果
          packed_b[nr_block_offset] -= (int32_t)kv * (int32_t)izp;
          // 将 kv 的值写入 packed_w 指向的位置
          *((uint8_t*)packed_w) = kv;
          // 更新 packed_w 的指针，移动 sizeof(uint8_t) 字节
          packed_w = (void*)((uintptr_t)packed_w + sizeof(uint8_t));
        }
        // 调整 packed_w 的指针，向前移动 (kr - kr_block_size) 个 uint8_t 大小
        packed_w =
            (void*)((uintptr_t)packed_w + (kr - kr_block_size) * sizeof(uint8_t));
      }
      // 调整 packed_w 的指针，向前移动 (nr - nr_block_size) * kr 个 uint8_t 大小
      packed_w =
          (void*)((uintptr_t)packed_w + (nr - nr_block_size) * kr * sizeof(uint8_t));
    }
}

// 将 Q8GEMM 乘法的权重矩阵打包并重新排列，用于优化计算
static inline void pytorch_pack_swizzle_q8gemm_brq(
    const size_t n,                  // 输出通道数
    const size_t kc,                 // 输入通道数乘以 4
    const uint32_t nr,               // 寄存器数量
    const uint32_t kr,               // 重叠因子
    const uint32_t sr,               // 步长因子
    const uint8_t* const k,          // 权重矩阵
    const int32_t* const b,          // 偏置向量
    void* const packed_w) {          // 打包后的权重矩阵

  union {
    void* const as_void_ptr;         // 转换为 void 指针
    uint8_t* as_uint8_ptr;           // 转换为 uint8_t 指针
    int32_t* as_int32_ptr;           // 转换为 int32_t 指针
  } packed = {packed_w};             // 初始化 packed 结构体

  for (size_t nr_block_start = 0; nr_block_start < n; nr_block_start += nr) {
    const size_t nr_block_size = min(n - nr_block_start, nr);  // 当前迭代中处理的寄存器数量
    for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size;
         nr_block_offset++) {
      *(packed.as_int32_ptr++) = b ? b[nr_block_start + nr_block_offset] : 0.0f;  // 将偏置向量的值打包到 packed_w 中
    }

    packed.as_int32_ptr += (nr - nr_block_size);  // 跳过未使用的寄存器位置

    for (size_t kr_block_start = 0; kr_block_start < (kc & -sr);
         kr_block_start += kr) {
      for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size;
           nr_block_offset++) {
        for (size_t kr_block_offset = 0; kr_block_offset < kr;
             kr_block_offset++) {
          const uint8_t kv =
              k[(nr_block_start + nr_block_offset) * kc +
                (kr_block_start & -sr) +
                ((kr_block_start + nr_block_offset * kr) & (sr - 1)) +
                kr_block_offset];
          *(packed.as_uint8_ptr++) = kv;  // 将权重矩阵的值按顺序打包到 packed_w 中
        }
      }
      packed.as_uint8_ptr += (nr - nr_block_size) * kr;  // 跳过未使用的位置
    }

    for (size_t kr_block_start = (kc & -sr); kr_block_start < kc;
         kr_block_start += kr) {
      const size_t kr_block_size = min(kc - kr_block_start, kr);  // 当前迭代中处理的重叠因子数量
      for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size;
           nr_block_offset++) {
        for (size_t kr_block_offset = 0; kr_block_offset < kr_block_size;
             kr_block_offset++) {
          const uint8_t kv =
              k[(nr_block_start + nr_block_offset) * kc +
                (kr_block_start + kr_block_offset)];
          *(packed.as_uint8_ptr++) = kv;  // 将权重矩阵的值按顺序打包到 packed_w 中
        }
        packed.as_uint8_ptr += (kr - kr_block_size);  // 跳过未使用的位置
      }
      packed.as_uint8_ptr += (nr - nr_block_size) * kr;  // 跳过未使用的位置
    }
  }
}

// 将 HGEMM 乘法的权重矩阵打包，用于优化计算
static inline void pytorch_pack_hgemm_w(
    size_t nc,                   // 输出通道数
    size_t kc,                   // 输入通道数乘以 2
    size_t nr,                   // 寄存器数量
    size_t kr,                   // 重叠因子
    const uint16_t* k,           // 权重矩阵
    const uint16_t* b,           // 偏置向量
    uint16_t* packed_w) {        // 打包后的权重矩阵

  for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
    const size_t nr_block_size = min(nc - nr_block_start, nr);  // 当前迭代中处理的寄存器数量
    for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size;
         nr_block_offset++) {
      *packed_w++ = b ? b[nr_block_start + nr_block_offset] : 0.0f;  // 将偏置向量的值打包到 packed_w 中
    }
    packed_w += nr - nr_block_size;  // 跳过未使用的位置
    # 外层循环，以 kr 为步长遍历 kc（输出通道数）
    for (size_t kr_block_start = 0; kr_block_start < kc; kr_block_start += kr) {
      # 当前 kr 块的大小，取 kc 和 kr 块起始点与 kc 的差值的最小值
      const size_t kr_block_size = min(kc - kr_block_start, kr);
      # 遍历 nr_block_size（输入通道数）个偏移量
      for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size;
           nr_block_offset++) {
        # 遍历 kr_block_size（当前 kr 块的大小）次
        for (size_t kr_block_offset = 0; kr_block_offset < kr_block_size;
             kr_block_offset++) {
          # 将 k 数组中的值复制到 packed_w 中
          *packed_w++ =
              k[(nr_block_start + nr_block_offset) * kc +
                (kr_block_start + kr_block_offset)];
        }
        # 在完成当前 kr 块后，向前移动（kr - kr_block_size）个位置
        packed_w += kr - kr_block_size;
      }
      # 在完成所有 nr_block_size 偏移量后，向前移动（nr - nr_block_size）* kr 个位置
      packed_w += (nr - nr_block_size) * kr;
    }
  }
}

// 定义一个静态内联函数，用于打包单精度矩阵乘法的权重
static inline void pytorch_pack_sgemm_w(
    size_t nc,               // 列数
    size_t kc,               // 每块列数
    size_t nr,               // 行数
    size_t kr,               // 每块行数
    const float* k,          // 权重矩阵指针
    const float* b,          // 偏置向量指针
    float* packed_w) {       // 打包后的权重存储位置指针
  for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
    const size_t nr_block_size = min(nc - nr_block_start, nr);  // 当前块的行数
    for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size;
         nr_block_offset++) {
      *packed_w++ = b ? b[nr_block_start + nr_block_offset] : 0.0f;  // 如果有偏置，则加入偏置；否则填充为0
    }
    packed_w += nr - nr_block_size;  // 如果当前块行数小于预期行数，则填充0
    for (size_t kr_block_start = 0; kr_block_start < kc; kr_block_start += kr) {
      const size_t kr_block_size = min(kc - kr_block_start, kr);  // 当前块的列数
      for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size;
           nr_block_offset++) {
        for (size_t kr_block_offset = 0; kr_block_offset < kr_block_size;
             kr_block_offset++) {
          *packed_w++ =
              k[(nr_block_start + nr_block_offset) * kc +
                (kr_block_start + kr_block_offset)];  // 打包权重矩阵数据
        }
        packed_w += kr - kr_block_size;  // 如果当前块列数小于预期列数，则填充0
      }
      packed_w += (nr - nr_block_size) * kr;  // 填充多余的行数*列数个0
    }
  }
}

// 定义一个静态内联函数，用于打包单精度卷积的权重
static inline void pytorch_pack_sconv_w(
    size_t n,                // 卷积核数
    size_t ks,               // 核大小
    size_t kc,               // 通道数
    size_t nr,               // 行数
    size_t kr,               // 每块行数
    const float* k,          // 权重矩阵指针
    const float* b,          // 偏置向量指针
    float* packed_w) {       // 打包后的权重存储位置指针
  for (size_t nr_block_start = 0; nr_block_start < n; nr_block_start += nr) {
    const size_t nr_block_size = min(n - nr_block_start, nr);  // 当前块的行数
    for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size;
         nr_block_offset++) {
      *packed_w++ = b ? b[nr_block_start + nr_block_offset] : 0.0f;  // 如果有偏置，则加入偏置；否则填充为0
    }
    packed_w += nr - nr_block_size;  // 如果当前块行数小于预期行数，则填充0
    for (size_t ki = 0; ki < ks; ki++) {
      for (size_t kr_block_start = 0; kr_block_start < kc;
           kr_block_start += kr) {
        const size_t kr_block_size = min(kc - kr_block_start, kr);  // 当前块的列数
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size;
             nr_block_offset++) {
          for (size_t kr_block_offset = 0; kr_block_offset < kr_block_size;
               kr_block_offset++) {
            *packed_w++ =
                k[((nr_block_start + nr_block_offset) * ks + ki) * kc +
                  (kr_block_start + kr_block_offset)];  // 打包权重矩阵数据
          }
          packed_w += kr - kr_block_size;  // 如果当前块列数小于预期列数，则填充0
        }
        packed_w += (nr - nr_block_size) * kr;  // 填充多余的行数*列数个0
      }
    }
  }
}

// 如果使用 QNNPACK 运行时量化
#if PYTORCH_QNNPACK_RUNTIME_QUANTIZATION

// 定义量化 QNNPACK 单精度矩阵乘法权重打包宏
#define pytorch_pack_q8gemm_w pytorch_pack_q8gemm_wrq
// 定义量化 QNNPACK 单精度卷积权重打包宏
#define pytorch_pack_q8conv_w pytorch_pack_q8conv_wrq
// 定义量化 QNNPACK 单精度反卷积权重打包宏
#define pytorch_pack_q8deconv_w pytorch_pack_q8deconv_wrq
// 定义量化 QNNPACK 单精度深度卷积权重打包宏
#define pytorch_pack_q8dw_w pytorch_pack_q8dw_wrq
// 定义量化 QNNPACK 单精度矩阵乘法偏置数据打包宏
#define pytorch_pack_swizzle_q8gemm_b pytorch_pack_swizzle_q8gemm_brq

// 否则使用 QNNPACK 动态量化
#else

// 定义量化 QNNPACK 单精度矩阵乘法权重打包宏
#define pytorch_pack_q8gemm_w pytorch_pack_q8gemm_wdq
// 定义量化 QNNPACK 单精度卷积权重打包宏
#define pytorch_pack_q8conv_w pytorch_pack_q8conv_wdq
// 定义量化 QNNPACK 单精度反卷积权重打包宏
#define pytorch_pack_q8deconv_w pytorch_pack_q8deconv_wdq
// 定义量化 QNNPACK 单精度深度卷积权重打包宏
#define pytorch_pack_q8dw_w pytorch_pack_q8dw_wdq
// 定义量化 QNNPACK 单精度矩阵乘法偏置数据打包宏
#define pytorch_pack_swizzle_q8gemm_b pytorch_pack_swizzle_q8gemm_bdq

#endif
```