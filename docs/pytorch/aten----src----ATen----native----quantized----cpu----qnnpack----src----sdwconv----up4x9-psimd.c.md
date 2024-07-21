# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\sdwconv\up4x9-psimd.c`

```
/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <psimd.h>  // 引入 PSIMD 库，用于 SIMD 加速计算
#include <qnnpack/sdwconv.h>  // 引入 SDWConv 相关头文件

void pytorch_sdwconv_ukernel_up4x9__psimd(
    size_t channels,  // 输入通道数
    size_t output_width,  // 输出宽度
    const float** input,  // 输入数据的指针数组
    const float* weights,  // 卷积核权重数组
    float* output,  // 输出数据数组
    size_t input_stride,  // 输入步长
    size_t output_increment,  // 输出增量
    const struct pytorch_qnnp_fp32_clamping_params
        clamping_params[restrict static 1]) {  // 浮点数范围约束参数结构体数组
  const psimd_f32 vmax = psimd_splat_f32(clamping_params->max);  // 获取最大浮点数约束值
  const psimd_f32 vmin = psimd_splat_f32(clamping_params->min);  // 获取最小浮点数约束值
  do {
    const float* i0 = input[0];  // 获取输入数据指针
    const float* i1 = input[1];  // 获取输入数据指针
    const float* i2 = input[2];  // 获取输入数据指针
    const float* i3 = input[3];  // 获取输入数据指针
    const float* i4 = input[4];  // 获取输入数据指针
    const float* i5 = input[5];  // 获取输入数据指针
    const float* i6 = input[6];  // 获取输入数据指针
    const float* i7 = input[7];  // 获取输入数据指针
    const float* i8 = input[8];  // 获取输入数据指针

    input = (const float**)((uintptr_t)input + input_stride);  // 更新输入指针数组位置

    size_t c = channels;  // 初始化通道数计数器
    const float* w = weights;  // 获取权重数组指针
    for (; c >= 4; c -= 4) {  // 循环处理每四个通道
      psimd_f32 vacc = psimd_load_f32(w);  // 初始化累加寄存器，加载权重数据

      const psimd_f32 vi0 = psimd_load_f32(i0);  // 加载输入数据到寄存器
      i0 += 4;  // 更新输入数据指针
      const psimd_f32 vk0 = psimd_load_f32(w + 8);  // 加载权重数据到寄存器
      vacc += vi0 * vk0;  // 执行乘加操作

      const psimd_f32 vi1 = psimd_load_f32(i1);  // 加载输入数据到寄存器
      i1 += 4;  // 更新输入数据指针
      const psimd_f32 vk1 = psimd_load_f32(w + 12);  // 加载权重数据到寄存器
      psimd_f32 vacc2 = vi1 * vk1;  // 执行乘操作

      // 以下类似地加载输入和权重数据，执行乘加操作，并进行浮点数约束处理
      const psimd_f32 vi2 = psimd_load_f32(i2);
      i2 += 4;
      const psimd_f32 vk2 = psimd_load_f32(w + 16);
      vacc += vi2 * vk2;

      const psimd_f32 vi3 = psimd_load_f32(i3);
      i3 += 4;
      const psimd_f32 vk3 = psimd_load_f32(w + 20);
      vacc2 += vi3 * vk3;

      const psimd_f32 vi4 = psimd_load_f32(i4);
      i4 += 4;
      const psimd_f32 vk4 = psimd_load_f32(w + 24);
      vacc += vi4 * vk4;

      const psimd_f32 vi5 = psimd_load_f32(i5);
      i5 += 4;
      const psimd_f32 vk5 = psimd_load_f32(w + 28);
      vacc2 += vi5 * vk5;

      const psimd_f32 vi6 = psimd_load_f32(i6);
      i6 += 4;
      const psimd_f32 vk6 = psimd_load_f32(w + 32);
      vacc += vi6 * vk6;

      const psimd_f32 vi7 = psimd_load_f32(i7);
      i7 += 4;
      const psimd_f32 vk7 = psimd_load_f32(w + 36);
      vacc2 += vi7 * vk7;

      const psimd_f32 vi8 = psimd_load_f32(i8);
      i8 += 4;
      const psimd_f32 vk8 = psimd_load_f32(w + 40);
      vacc += vi8 * vk8;

      vacc += vacc2;  // 合并两个累加寄存器的结果

      vacc = psimd_min_f32(vacc, vmax);  // 对结果进行最小值约束
      vacc = psimd_max_f32(vacc, vmin);  // 对结果进行最大值约束

      psimd_store_f32(output, vacc);  // 将结果存储到输出数组
      w += 44;  // 更新权重指针
    }
    # 如果 c 不等于 0，则执行以下操作
    if (c != 0) {
      # 从内存中加载向量 w 到 psimd_f32 类型的变量 vacc
      psimd_f32 vacc = psimd_load_f32(w);
      # 计算 c 的字节大小并赋值给 c
      c *= sizeof(float);

      # 调整指针 i0，使其指向前一个 c 字节处，然后加载该地址处的 psimd_f32 向量 vi0
      i0 = (const float*)((uintptr_t)i0 - c);
      const psimd_f32 vi0 = psimd_load_f32(i0);
      # 加载向量 w 中偏移 8 字节处的 psimd_f32 向量 vk0
      const psimd_f32 vk0 = psimd_load_f32(w + 8);
      # 计算新的 vacc 值，vi0 乘以 vk0 然后加到 vacc 中
      vacc += vi0 * vk0;

      # 以下类似地处理 i1 到 i8，vi1 到 vi8，vk1 到 vk8，以及相应的计算

      # 将 psimd_f32 向量 vi1 乘以 psimd_f32 向量 vk1 存入 vacc2
      i1 = (const float*)((uintptr_t)i1 - c);
      const psimd_f32 vi1 = psimd_load_f32(i1);
      const psimd_f32 vk1 = psimd_load_f32(w + 12);
      psimd_f32 vacc2 = vi1 * vk1;

      # 类似地处理 i2 到 i8，vi2 到 vi8，vk2 到 vk8，以及相应的计算

      # 将 psimd_f32 向量 vi3 乘以 psimd_f32 向量 vk3 加到 vacc2 中
      i3 = (const float*)((uintptr_t)i3 - c);
      const psimd_f32 vi3 = psimd_load_f32(i3);
      const psimd_f32 vk3 = psimd_load_f32(w + 20);
      vacc2 += vi3 * vk3;

      # 类似地处理 i4 到 i8，vi4 到 vi8，vk4 到 vk8，以及相应的计算

      # 将 psimd_f32 向量 vi4 乘以 psimd_f32 向量 vk4 加到 vacc 中
      i4 = (const float*)((uintptr_t)i4 - c);
      const psimd_f32 vi4 = psimd_load_f32(i4);
      const psimd_f32 vk4 = psimd_load_f32(w + 24);
      vacc += vi4 * vk4;

      # 类似地处理 i5 到 i8，vi5 到 vi8，vk5 到 vk8，以及相应的计算

      # 将 psimd_f32 向量 vi5 乘以 psimd_f32 向量 vk5 加到 vacc2 中
      i5 = (const float*)((uintptr_t)i5 - c);
      const psimd_f32 vi5 = psimd_load_f32(i5);
      const psimd_f32 vk5 = psimd_load_f32(w + 28);
      vacc2 += vi5 * vk5;

      # 类似地处理 i6 到 i8，vi6 到 vi8，vk6 到 vk8，以及相应的计算

      # 将 psimd_f32 向量 vi6 乘以 psimd_f32 向量 vk6 加到 vacc 中
      i6 = (const float*)((uintptr_t)i6 - c);
      const psimd_f32 vi6 = psimd_load_f32(i6);
      const psimd_f32 vk6 = psimd_load_f32(w + 32);
      vacc += vi6 * vk6;

      # 类似地处理 i7 到 i8，vi7 到 vi8，vk7 到 vk8，以及相应的计算

      # 将 psimd_f32 向量 vi7 乘以 psimd_f32 向量 vk7 加到 vacc2 中
      i7 = (const float*)((uintptr_t)i7 - c);
      const psimd_f32 vi7 = psimd_load_f32(i7);
      const psimd_f32 vk7 = psimd_load_f32(w + 36);
      vacc2 += vi7 * vk7;

      # 类似地处理 i8，vi8，vk8，以及相应的计算

      # 将 psimd_f32 向量 vi8 乘以 psimd_f32 向量 vk8 加到 vacc 中
      i8 = (const float*)((uintptr_t)i8 - c);
      const psimd_f32 vi8 = psimd_load_f32(i8);
      const psimd_f32 vk8 = psimd_load_f32(w + 40);
      vacc += vi8 * vk8;

      # 将 vacc2 加到 vacc 中，得到最终的 vacc 值
      vacc += vacc2;

      # 使用 psimd_min_f32 函数将 vacc 中的值与 vmax 中的最小值进行比较并存回 vacc
      vacc = psimd_min_f32(vacc, vmax);
      # 使用 psimd_max_f32 函数将 vacc 中的值与 vmin 中的最大值进行比较并存回 vacc
      vacc = psimd_max_f32(vacc, vmin);

      # 调整指针 output，使其指向前一个 c 字节处，然后将 vacc 中的值存入该地址处
      output = (float*)((uintptr_t)output - c);
      psimd_store_f32(output, vacc);
    }

    # 调整指针 output，使其指向下一个 output_increment 字节处
    output = (float*)((uintptr_t)output + output_increment);
  } while (--output_width != 0);  # 循环直到 output_width 为 0
}



# 这行代码关闭了一个代码块。在很多编程语言中，大括号被用来标记代码块的开始和结束，这里的右大括号表示了一个代码块的结束。
```