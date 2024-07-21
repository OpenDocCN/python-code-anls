# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\requantization\fp32-scalar.c`

```py
/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <assert.h>
#include <math.h>
#include <stdint.h>

#include <fp16/bitcasts.h>
#include <qnnpack/requantization-stubs.h>

// 使用 LRINTF 方式对 FP32 数据进行量化，标量版本
void pytorch_qnnp_requantize_fp32__scalar_lrintf(
    size_t n,                            // 输入数据的数量，必须是 4 的倍数
    const int32_t* input,                // 输入数据的指针，为 int32_t 类型
    float scale,                         // 缩放因子，为 float 类型
    uint8_t zero_point,                  // 零点，为 uint8_t 类型
    uint8_t qmin,                        // 最小量化值，为 uint8_t 类型
    uint8_t qmax,                        // 最大量化值，为 uint8_t 类型
    uint8_t* output) {                   // 输出数据的指针，为 uint8_t 类型
  assert(n % 4 == 0);                   // 断言：输入数据数量必须是 4 的倍数
  assert(scale < 1.0f);                 // 断言：缩放因子必须小于 1.0
  assert(scale >= 0x1.0p-32f);          // 断言：缩放因子必须大于等于 2^-32

  const long lmin =                     // 计算最小量化值对应的 long 类型数值
      (long)((int32_t)(uint32_t)qmin - (int32_t)(uint32_t)zero_point);
  const long lmax =                     // 计算最大量化值对应的 long 类型数值
      (long)((int32_t)(uint32_t)qmax - (int32_t)(uint32_t)zero_point);
  for (; n != 0; n -= 4) {              // 循环处理每组四个输入数据
    const int32_t x = input[0];         // 获取输入数据的第一个元素
    const int32_t y = input[1];         // 获取输入数据的第二个元素
    const int32_t z = input[2];         // 获取输入数据的第三个元素
    const int32_t w = input[3];         // 获取输入数据的第四个元素
    input += 4;                         // 更新输入数据指针到下一组四个元素

    const float x_scaled = (float)x * scale;  // 将 x 缩放到 float 类型
    const float y_scaled = (float)y * scale;  // 将 y 缩放到 float 类型
    const float z_scaled = (float)z * scale;  // 将 z 缩放到 float 类型
    const float w_scaled = (float)w * scale;  // 将 w 缩放到 float 类型

    const long x_rounded = lrintf(x_scaled);  // 对 x 进行四舍五入到 long 类型
    const long y_rounded = lrintf(y_scaled);  // 对 y 进行四舍五入到 long 类型
    const long z_rounded = lrintf(z_scaled);  // 对 z 进行四舍五入到 long 类型
    const long w_rounded = lrintf(w_scaled);  // 对 w 进行四舍五入到 long 类型

    const int32_t x_clamped = (int32_t)(       // 对 x 进行量化并进行上下限截断
        x_rounded < lmin ? lmin : x_rounded > lmax ? lmax : x_rounded);
    const int32_t y_clamped = (int32_t)(       // 对 y 进行量化并进行上下限截断
        y_rounded < lmin ? lmin : y_rounded > lmax ? lmax : y_rounded);
    const int32_t z_clamped = (int32_t)(       // 对 z 进行量化并进行上下限截断
        z_rounded < lmin ? lmin : z_rounded > lmax ? lmax : z_rounded);
    const int32_t w_clamped = (int32_t)(       // 对 w 进行量化并进行上下限截断
        w_rounded < lmin ? lmin : w_rounded > lmax ? lmax : w_rounded);

    const int32_t x_biased = x_clamped + (int32_t)(uint32_t)zero_point;  // 加上零点偏置得到最终量化结果
    const int32_t y_biased = y_clamped + (int32_t)(uint32_t)zero_point;  // 加上零点偏置得到最终量化结果
    const int32_t z_biased = z_clamped + (int32_t)(uint32_t)zero_point;  // 加上零点偏置得到最终量化结果
    const int32_t w_biased = w_clamped + (int32_t)(uint32_t)zero_point;  // 加上零点偏置得到最终量化结果

    output[0] = (uint8_t)x_biased;      // 将 x_biased 转换成 uint8_t 类型存入输出
    output[1] = (uint8_t)y_biased;      // 将 y_biased 转换成 uint8_t 类型存入输出
    output[2] = (uint8_t)z_biased;      // 将 z_biased 转换成 uint8_t 类型存入输出
    output[3] = (uint8_t)w_biased;      // 将 w_biased 转换成 uint8_t 类型存入输出
    output += 4;                        // 更新输出指针到下一组四个元素
  }
}

// 使用魔术数方法对 FP32 数据进行量化，标量版本
void pytorch_qnnp_requantize_fp32__scalar_magic(
    size_t n,                            // 输入数据的数量，必须是 4 的倍数
    const int32_t* input,                // 输入数据的指针，为 int32_t 类型
    float scale,                         // 缩放因子，为 float 类型
    uint8_t zero_point,                  // 零点，为 uint8_t 类型
    uint8_t qmin,                        // 最小量化值，为 uint8_t 类型
    uint8_t qmax,                        // 最大量化值，为 uint8_t 类型
    uint8_t* output) {                   // 输出数据的指针，为 uint8_t 类型
  assert(n % 4 == 0);                   // 断言：输入数据数量必须是 4 的倍数
  assert(scale < 1.0f);                 // 断言：缩放因子必须小于 1.0
  assert(scale >= 0x1.0p-32f);          // 断言：缩放因子必须大于等于 2^-32

  const float fmin =                    // 计算最小量化值对应的 float 类型数值
      (float)((int32_t)(uint32_t)qmin - (int32_t)(uint32_t)zero_point);
  const float fmax =                    // 计算最大量化值对应的 float 类型数值
      (float)((int32_t)(uint32_t)qmax - (int32_t)(uint32_t)zero_point);
  const float fmagic = 12582912.0f;     // 预设的魔术数常量
  const int32_t imagic = INT32_C(0x4B400000) - (int32_t)(uint32_t)zero_point;  // 预设的魔术数常量
  for (; n != 0; n -= 4) {              // 循环处理每组四个输入数据
    const int32_t x = input[0];         // 获取输入数据的第一个元素
    const int32_t y = input[1];         // 获取输入数据的第二个元素
    const int32_t z = input[2];         // 获取输入数据的第三个元素
    const int32_t w = input[3];         // 获取输入数据的第四个元素
    input += 4;                         // 更新输入数据指针到下一组四个元素
    // 计算 x、y、z、w 的缩放值，将整数 x、y、z、w 乘以缩放因子转换为浮点数
    const float x_scaled = (float)x * scale;
    const float y_scaled = (float)y * scale;
    const float z_scaled = (float)z * scale;
    const float w_scaled = (float)w * scale;

    // 对缩放后的值进行截断，确保在指定的范围内
    const float x_clamped =
        x_scaled < fmin ? fmin : x_scaled > fmax ? fmax : x_scaled;
    const float y_clamped =
        y_scaled < fmin ? fmin : y_scaled > fmax ? fmax : y_scaled;
    const float z_clamped =
        z_scaled < fmin ? fmin : z_scaled > fmax ? fmax : z_scaled;
    const float w_clamped =
        w_scaled < fmin ? fmin : w_scaled > fmax ? fmax : w_scaled;

    // 将截断后的浮点数值转换为带偏置的整数
    const int32_t x_biased = (int32_t)fp32_to_bits(x_clamped + fmagic) - imagic;
    const int32_t y_biased = (int32_t)fp32_to_bits(y_clamped + fmagic) - imagic;
    const int32_t z_biased = (int32_t)fp32_to_bits(z_clamped + fmagic) - imagic;
    const int32_t w_biased = (int32_t)fp32_to_bits(w_clamped + fmagic) - imagic;

    // 将带偏置的整数转换为无符号字节，并存入输出数组
    output[0] = (uint8_t)x_biased;
    output[1] = (uint8_t)y_biased;
    output[2] = (uint8_t)z_biased;
    output[3] = (uint8_t)w_biased;
    // 更新输出数组指针，使其指向下一个输出位置
    output += 4;
}
}



# 这行代码关闭了一个函数的定义。在 Python 中，'}' 表示函数或类定义的结束。
```