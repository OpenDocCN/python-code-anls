# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\q8gemm\4x8c2-xzp-neon.c`

```
/*
 * 版权所有（c）Facebook，Inc. 及其关联公司。
 * 保留所有权利。
 *
 * 此源代码在根目录下的 LICENSE 文件中以 BSD 风格许可证授权。
 */

#include <arm_neon.h> // 包含 ARM NEON 指令集的头文件

#include <qnnpack/q8gemm.h> // 包含 QNNPACK Q8 GEMM 头文件

void pytorch_q8gemm_xzp_ukernel_4x8c2__neon(
    size_t mr, // 定义矩阵 A 的行数（寄存器数）
    size_t nr, // 定义矩阵 B 的列数（寄存器数的两倍）
    size_t k, // 定义矩阵的列数，即矩阵 A 的列数或矩阵 B 的行数
    const uint8_t* restrict a, // 输入矩阵 A 的指针（8 位无符号整数）
    size_t a_stride, // 矩阵 A 的行步长
    const int32_t* restrict a_sum, // 每行矩阵 A 的和的指针（32 位整数）
    const void* restrict w, // 输入矩阵 B 的指针（void 类型）
    uint8_t* restrict c, // 输出矩阵 C 的指针（8 位无符号整数）
    size_t c_stride, // 矩阵 C 的行步长
    const union pytorch_qnnp_q31_requantization_params
        requantization_params[restrict static 1]) // 重量化参数的联合结构体
{
  int32x4_t vacc0x0123 = vld1q_s32(w); // 从 w 中加载四个 32 位整数到寄存器
  w = (const void*)((uintptr_t)w + 16); // 更新 w 的地址，指向下一个寄存器
  int32x4_t vacc0x4567 = vld1q_s32(w); // 从更新后的 w 中加载四个 32 位整数到寄存器
  w = (const void*)((uintptr_t)w + 16); // 更新 w 的地址，指向下一个寄存器
  int32x4_t vacc1x0123 = vacc0x0123; // 复制第一个寄存器的值到第二个寄存器
  int32x4_t vacc1x4567 = vacc0x4567; // 复制第二个寄存器的值到第三个寄存器
  int32x4_t vacc2x0123 = vacc0x0123; // 复制第一个寄存器的值到第四个寄存器
  int32x4_t vacc2x4567 = vacc0x4567; // 复制第二个寄存器的值到第五个寄存器
  int32x4_t vacc3x0123 = vacc0x0123; // 复制第一个寄存器的值到第六个寄存器
  int32x4_t vacc3x4567 = vacc0x4567; // 复制第二个寄存器的值到第七个寄存器

  const uint8_t* a0 = a; // 设置第一行矩阵 A 的指针
  const uint8_t* a1 = a0; // 设置第二行矩阵 A 的指针
  const int32_t* a_sum0 = a_sum; // 设置第一行矩阵 A 和的指针
  const int32_t* a_sum1 = a_sum0; // 设置第二行矩阵 A 和的指针
  if (mr >= 2) { // 如果矩阵 A 的行数大于等于 2
    a1 += a_stride; // 更新第二行矩阵 A 的指针
    a_sum1 += 1; // 更新第二行矩阵 A 和的指针
  }
  const uint8_t* a2 = a1; // 设置第三行矩阵 A 的指针
  const int32_t* a_sum2 = a_sum1; // 设置第三行矩阵 A 和的指针
  if (mr > 2) { // 如果矩阵 A 的行数大于 2
    a2 += a_stride; // 更新第三行矩阵 A 的指针
    a_sum2 += 1; // 更新第三行矩阵 A 和的指针
  }
  const uint8_t* a3 = a2; // 设置第四行矩阵 A 的指针
  const int32_t* a_sum3 = a_sum2; // 设置第四行矩阵 A 和的指针
  if (mr == 4) { // 如果矩阵 A 的行数等于 4
    a3 += a_stride; // 更新第四行矩阵 A 的指针
    a_sum3 += 1; // 更新第四行矩阵 A 和的指针
  }

  const int32x4_t va_sum0 = vld1q_dup_s32(a_sum0); // 从 a_sum0 中加载一个 32 位整数到寄存器，并复制到向量寄存器
  const int32x4_t va_sum1 = vld1q_dup_s32(a_sum1); // 从 a_sum1 中加载一个 32 位整数到寄存器，并复制到向量寄存器
  const int32x4_t va_sum2 = vld1q_dup_s32(a_sum2); // 从 a_sum2 中加载一个 32 位整数到寄存器，并复制到向量寄存器
  const int32x4_t va_sum3 = vld1q_dup_s32(a_sum3); // 从 a_sum3 中加载一个 32 位整数到寄存器，并复制到向量寄存器
  vacc0x0123 = vaddq_s32(vacc0x0123, va_sum0); // 将 va_sum0 的值添加到寄存器 vacc0x0123
  vacc0x4567 = vaddq_s32(vacc0x4567, va_sum0); // 将 va_sum0 的值添加到寄存器 vacc0x4567
  vacc1x0123 = vaddq_s32(vacc1x0123, va_sum1); // 将 va_sum1 的值添加到寄存器 vacc1x0123
  vacc1x4567 = vaddq_s32(vacc1x4567, va_sum1); // 将 va_sum1 的值添加到寄存器 vacc1x4567
  vacc2x0123 = vaddq_s32(vacc2x0123, va_sum2); // 将 va_sum2 的值添加到寄存器 vacc2x0123
  vacc2x4567 = vaddq_s32(vacc2x4567, va_sum2); // 将 va_sum2 的值添加到寄存器 vacc2x4567
  vacc3x0123 = vaddq_s32(vacc3x0123, va_sum3); // 将 va_sum3 的值添加到寄存器 vacc3x0123
  vacc3x4567 = vaddq_s32(vacc3x4567, va_sum3); // 将 va_sum3 的值添加到寄存器 vacc3x4567

  for (; k >= 8; k -= 8) { // 迭代处理每组 8 个元素，直到 k 小于 8
    uint8x8_t va0x01234567 = vld1_u8(a0); // 从 a0 加载一个 8 位无符号整数向量到寄存器
    a0 += 8; // 更新 a0 的指针，指向下一个向量
    uint8x8_t va1x01234567 = vld1_u8(a1); // 从 a1 加载一个 8 位无符号整数向量到寄存器
    a1 += 8; // 更新 a1 的指针，指向下一个向量
    uint8x8_t va2x01234567 = vld1_u8(a2); // 从 a2 加载一个 8 位无符号整数向量到寄存器
    a2 += 8
    # 更新向量累加器 vacc2x0123 和 vacc2x4567，使用 vb01234567x01 的低位和高位分别乘以 va2x01234567 的每个元素
    vacc2x0123 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc2x0123),
        vmull_u8(va2x01234567, vget_low_u8(vb01234567x01))));
    vacc2x4567 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc2x4567),
        vmull_u8(va2x01234567, vget_high_u8(vb01234567x01))));

    # 更新向量累加器 vacc3x0123 和 vacc3x4567，使用 vb01234567x01 的低位和高位分别乘以 va3x01234567 的每个元素
    vacc3x0123 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc3x0123),
        vmull_u8(va3x01234567, vget_low_u8(vb01234567x01))));
    vacc3x4567 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc3x4567),
        vmull_u8(va3x01234567, vget_high_u8(vb01234567x01))));

    # 将向量 va0x01234567, va1x01234567, va2x01234567, va3x01234567 每个元素向左循环移动两位
    va0x01234567 = vext_u8(va0x01234567, va0x01234567, 2);
    va1x01234567 = vext_u8(va1x01234567, va1x01234567, 2);
    va2x01234567 = vext_u8(va2x01234567, va2x01234567, 2);
    va3x01234567 = vext_u8(va3x01234567, va3x01234567, 2);

    # 加载数组 w 中的下一个 16 字节到向量 vb01234567x23
    const uint8x16_t vb01234567x23 = vld1q_u8(w);
    w += 16;

    # 更新向量累加器 vacc0x0123 和 vacc0x4567，使用 vb01234567x23 的低位和高位分别乘以 va0x01234567 的每个元素
    vacc0x0123 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc0x0123),
        vmull_u8(va0x01234567, vget_low_u8(vb01234567x23))));
    vacc0x4567 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc0x4567),
        vmull_u8(va0x01234567, vget_high_u8(vb01234567x23))));

    # 更新向量累加器 vacc1x0123 和 vacc1x4567，使用 vb01234567x23 的低位和高位分别乘以 va1x01234567 的每个元素
    vacc1x0123 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc1x0123),
        vmull_u8(va1x01234567, vget_low_u8(vb01234567x23))));
    vacc1x4567 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc1x4567),
        vmull_u8(va1x01234567, vget_high_u8(vb01234567x23))));

    # 更新向量累加器 vacc2x0123 和 vacc2x4567，使用 vb01234567x23 的低位和高位分别乘以 va2x01234567 的每个元素
    vacc2x0123 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc2x0123),
        vmull_u8(va2x01234567, vget_low_u8(vb01234567x23))));
    vacc2x4567 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc2x4567),
        vmull_u8(va2x01234567, vget_high_u8(vb01234567x23))));

    # 更新向量累加器 vacc3x0123 和 vacc3x4567，使用 vb01234567x23 的低位和高位分别乘以 va3x01234567 的每个元素
    vacc3x0123 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc3x0123),
        vmull_u8(va3x01234567, vget_low_u8(vb01234567x23))));
    vacc3x4567 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc3x4567),
        vmull_u8(va3x01234567, vget_high_u8(vb01234567x23))));

    # 将向量 va0x01234567, va1x01234567, va2x01234567, va3x01234567 每个元素向左循环移动两位
    va0x01234567 = vext_u8(va0x01234567, va0x01234567, 2);
    va1x01234567 = vext_u8(va1x01234567, va1x01234567, 2);
    va2x01234567 = vext_u8(va2x01234567, va2x01234567, 2);
    va3x01234567 = vext_u8(va3x01234567, va3x01234567, 2);

    # 加载数组 w 中的下一个 16 字节到向量 vb01234567x45
    const uint8x16_t vb01234567x45 = vld1q_u8(w);
    w += 16;

    # 更新向量累加器 vacc0x0123 和 vacc0x4567，使用 vb01234567x45 的低位和高位分别乘以 va0x01234567 的每个元素
    vacc0x0123 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc0x0123),
        vmull_u8(va0x01234567, vget_low_u8(vb01234567x45))));
    vacc0x4567 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc0x4567),
        vmull_u8(va0x01234567, vget_high_u8(vb01234567x45))));

    # 更新向量累加器 vacc1x0123 和 vacc1x4567，使用 vb01234567x45 的低位和高位分别乘以 va1x01234567 的每个元素
    vacc1x0123 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc1x0123),
        vmull_u8(va1x01234567, vget_low_u8(vb01234567x45))));
    vacc1x4567 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc1x4567),
        vmull_u8(va1x01234567, vget_high_u8(vb01234567x45))));

    # 更新向量累加器 vacc2x0123 和 vacc2x4567，使用 vb01234567x45 的低位和高位分别乘以 va2x01234567 的每个元素
    vacc2x0123 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc2x0123),
        vmull_u8(va2x01234567, vget_low_u8(vb01234567x45))));
    vacc2x4567 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc2x4567),
        vmull_u8(va2x01234567, vget_high_u8(vb01234567x45))));

    # 更新向量累加器 vacc3x0123 和 vacc3x4567，使用 vb01234567x45 的低位和高位分
    // 更新累加器 vacc1x0123 和 vacc1x4567，通过向量乘法和累加将结果添加到累加器中
    vacc1x0123 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc1x0123),
        vmull_u8(va1x01234567, vget_low_u8(vb01234567x45))));
    vacc1x4567 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc1x4567),
        vmull_u8(va1x01234567, vget_high_u8(vb01234567x45))));

    // 更新累加器 vacc2x0123 和 vacc2x4567，通过向量乘法和累加将结果添加到累加器中
    vacc2x0123 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc2x0123),
        vmull_u8(va2x01234567, vget_low_u8(vb01234567x45))));
    vacc2x4567 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc2x4567),
        vmull_u8(va2x01234567, vget_high_u8(vb01234567x45))));

    // 更新累加器 vacc3x0123 和 vacc3x4567，通过向量乘法和累加将结果添加到累加器中
    vacc3x0123 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc3x0123),
        vmull_u8(va3x01234567, vget_low_u8(vb01234567x45))));
    vacc3x4567 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc3x4567),
        vmull_u8(va3x01234567, vget_high_u8(vb01234567x45))));

    /* k = 6, 7 */
    // 将每个 va0x01234567 向量向左移动两个字节位置
    va0x01234567 = vext_u8(va0x01234567, va0x01234567, 2);
    va1x01234567 = vext_u8(va1x01234567, va1x01234567, 2);
    va2x01234567 = vext_u8(va2x01234567, va2x01234567, 2);
    va3x01234567 = vext_u8(va3x01234567, va3x01234567, 2);

    // 加载新的 vb01234567x67 向量，逐步读取 w 指向的内存
    const uint8x16_t vb01234567x67 = vld1q_u8(w);
    w += 16;

    // 更新累加器 vacc0x0123 和 vacc0x4567，通过向量乘法和累加将结果添加到累加器中
    vacc0x0123 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc0x0123),
        vmull_u8(va0x01234567, vget_low_u8(vb01234567x67))));
    vacc0x4567 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc0x4567),
        vmull_u8(va0x01234567, vget_high_u8(vb01234567x67))));

    // 更新累加器 vacc1x0123 和 vacc1x4567，通过向量乘法和累加将结果添加到累加器中
    vacc1x0123 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc1x0123),
        vmull_u8(va1x01234567, vget_low_u8(vb01234567x67))));
    vacc1x4567 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc1x4567),
        vmull_u8(va1x01234567, vget_high_u8(vb01234567x67))));

    // 更新累加器 vacc2x0123 和 vacc2x4567，通过向量乘法和累加将结果添加到累加器中
    vacc2x0123 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc2x0123),
        vmull_u8(va2x01234567, vget_low_u8(vb01234567x67))));
    vacc2x4567 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc2x4567),
        vmull_u8(va2x01234567, vget_high_u8(vb01234567x67))));

    // 更新累加器 vacc3x0123 和 vacc3x4567，通过向量乘法和累加将结果添加到累加器中
    vacc3x0123 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc3x0123),
        vmull_u8(va3x01234567, vget_low_u8(vb01234567x67))));
    vacc3x4567 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc3x4567),
        vmull_u8(va3x01234567, vget_high_u8(vb01234567x67))));
  }

  /* for k < 8, reuse the packing scheme for the original xzp ukernel */
  if (k & 4) {
    /* k = 0, 1 */
    // 加载 a0 和 a1 指向的内存作为 uint16_t 数组，然后转换为 uint8x8_t 向量
    const uint8x8_t va0x01010101 = vreinterpret_u8_u16(
        vld1_dup_u16(__builtin_assume_aligned((const uint16_t*)a0, 1)));
    a0 += 2;
    const uint8x8_t va1x01010101 = vreinterpret_u8_u16(
        vld1_dup_u16(__builtin_assume_aligned((const uint16_t*)a1, 1)));
    a1 += 2;
    // 将数组 a2 的前两个元素以 uint16_t 类型读入，再转换为 uint8x8_t 类型的向量 va2x01010101
    const uint8x8_t va2x01010101 = vreinterpret_u8_u16(
        vld1_dup_u16(__builtin_assume_aligned((const uint16_t*)a2, 1)));
    // 指针 a2 向后移动两个元素的位置
    a2 += 2;
    
    // 将数组 a3 的前两个元素以 uint16_t 类型读入，再转换为 uint8x8_t 类型的向量 va3x01010101
    const uint8x8_t va3x01010101 = vreinterpret_u8_u16(
        vld1_dup_u16(__builtin_assume_aligned((const uint16_t*)a3, 1)));
    // 指针 a3 向后移动两个元素的位置
    a3 += 2;
    
    // 将指针 w 指向的 16 字节数据读取为 uint8x16_t 类型的向量 vb01234567x01
    const uint8x16_t vb01234567x01 = vld1q_u8(w);
    // 指针 w 向后移动 16 字节的位置
    w += 16;
    
    // 使用 va0x01010101 和 vb01234567x01 计算累加和并更新到 vacc0x0123 和 vacc0x4567 中
    vacc0x0123 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc0x0123),
        vmull_u8(va0x01010101, vget_low_u8(vb01234567x01))));
    vacc0x4567 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc0x4567),
        vmull_u8(va0x01010101, vget_high_u8(vb01234567x01))));
    
    // 使用 va1x01010101 和 vb01234567x01 计算累加和并更新到 vacc1x0123 和 vacc1x4567 中
    vacc1x0123 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc1x0123),
        vmull_u8(va1x01010101, vget_low_u8(vb01234567x01))));
    vacc1x4567 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc1x4567),
        vmull_u8(va1x01010101, vget_high_u8(vb01234567x01))));
    
    // 使用 va2x01010101 和 vb01234567x01 计算累加和并更新到 vacc2x0123 和 vacc2x4567 中
    vacc2x0123 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc2x0123),
        vmull_u8(va2x01010101, vget_low_u8(vb01234567x01))));
    vacc2x4567 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc2x4567),
        vmull_u8(va2x01010101, vget_high_u8(vb01234567x01))));
    
    // 使用 va3x01010101 和 vb01234567x01 计算累加和并更新到 vacc3x0123 和 vacc3x4567 中
    vacc3x0123 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc3x0123),
        vmull_u8(va3x01010101, vget_low_u8(vb01234567x01))));
    vacc3x4567 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc3x4567),
        vmull_u8(va3x01010101, vget_high_u8(vb01234567x01))));

    /* k = 2, 3 */
    // 将数组 a0 的前两个元素以 uint16_t 类型读入，再转换为 uint8x8_t 类型的向量 va0x23232323
    const uint8x8_t va0x23232323 = vreinterpret_u8_u16(
        vld1_dup_u16(__builtin_assume_aligned((const uint16_t*)a0, 1)));
    // 指针 a0 向后移动两个元素的位置
    a0 += 2;
    
    // 将数组 a1 的前两个元素以 uint16_t 类型读入，再转换为 uint8x8_t 类型的向量 va1x23232323
    const uint8x8_t va1x23232323 = vreinterpret_u8_u16(
        vld1_dup_u16(__builtin_assume_aligned((const uint16_t*)a1, 1)));
    // 指针 a1 向后移动两个元素的位置
    a1 += 2;
    
    // 将数组 a2 的前两个元素以 uint16_t 类型读入，再转换为 uint8x8_t 类型的向量 va2x23232323
    const uint8x8_t va2x23232323 = vreinterpret_u8_u16(
        vld1_dup_u16(__builtin_assume_aligned((const uint16_t*)a2, 1)));
    // 指针 a2 向后移动两个元素的位置
    a2 += 2;
    
    // 将数组 a3 的前两个元素以 uint16_t 类型读入，再转换为 uint8x8_t 类型的向量 va3x23232323
    const uint8x8_t va3x23232323 = vreinterpret_u8_u16(
        vld1_dup_u16(__builtin_assume_aligned((const uint16_t*)a3, 1)));
    // 指针 a3 向后移动两个元素的位置
    a3 += 2;
    
    // 将指针 w 指向的 16 字节数据读取为 uint8x16_t 类型的向量 vb01234567x23
    const uint8x16_t vb01234567x23 = vld1q_u8(w);
    // 指针 w 向后移动 16 字节的位置
    w += 16;
    
    // 使用 va0x23232323 和 vb01234567x23 计算累加和并更新到 vacc0x0123 和 vacc0x4567 中
    vacc0x0123 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc0x0123),
        vmull_u8(va0x23232323, vget_low_u8(vb01234567x23))));
    vacc0x4567 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc0x4567),
        vmull_u8(va0x23232323, vget_high_u8(vb01234567x23))));
    
    // 使用 va1x23232323 和 vb01234567x23 计算累加和并更新到 vacc1x0123 和 vacc1x4567 中
    vacc1x0123 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc1x0123),
        vmull_u8(va1x23232323, vget_low_u8(vb01234567x23))));
    vacc1x4567 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc1x4567),
        vmull_u8(va1x23232323, vget_high_u8(vb01234567x23))));
    vacc2x0123 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc2x0123),
        vmull_u8(va2x23232323, vget_low_u8(vb01234567x23))));
    # 将 vb01234567x23 的低 8 字节扩展为 16 位，与 va2x23232323 的每个字节进行无符号 8 位乘法，然后将结果累加到 vacc2x0123 中

    vacc2x4567 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc2x4567),
        vmull_u8(va2x23232323, vget_high_u8(vb01234567x23))));
    # 将 vb01234567x23 的高 8 字节扩展为 16 位，与 va2x23232323 的每个字节进行无符号 8 位乘法，然后将结果累加到 vacc2x4567 中

    vacc3x0123 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc3x0123),
        vmull_u8(va3x23232323, vget_low_u8(vb01234567x23))));
    # 将 vb01234567x23 的低 8 字节扩展为 16 位，与 va3x23232323 的每个字节进行无符号 8 位乘法，然后将结果累加到 vacc3x0123 中

    vacc3x4567 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc3x4567),
        vmull_u8(va3x23232323, vget_high_u8(vb01234567x23))));
    # 将 vb01234567x23 的高 8 字节扩展为 16 位，与 va3x23232323 的每个字节进行无符号 8 位乘法，然后将结果累加到 vacc3x4567 中

  }
  if (k & 2) {
    /* k = 0, 1 */
    const uint8x8_t va0x01010101 = vreinterpret_u8_u16(
        vld1_dup_u16(__builtin_assume_aligned((const uint16_t*)a0, 1)));
    # 从 a0 加载一个 16 位的无符号整数，将其转换为 8 位的无符号整数向量 va0x01010101
    a0 += 2;  // a0 指针向后移动 2 个字节

    const uint8x8_t va1x01010101 = vreinterpret_u8_u16(
        vld1_dup_u16(__builtin_assume_aligned((const uint16_t*)a1, 1)));
    # 从 a1 加载一个 16 位的无符号整数，将其转换为 8 位的无符号整数向量 va1x01010101
    a1 += 2;  // a1 指针向后移动 2 个字节

    const uint8x8_t va2x01010101 = vreinterpret_u8_u16(
        vld1_dup_u16(__builtin_assume_aligned((const uint16_t*)a2, 1)));
    # 从 a2 加载一个 16 位的无符号整数，将其转换为 8 位的无符号整数向量 va2x01010101
    a2 += 2;  // a2 指针向后移动 2 个字节

    const uint8x8_t va3x01010101 = vreinterpret_u8_u16(
        vld1_dup_u16(__builtin_assume_aligned((const uint16_t*)a3, 1)));
    # 从 a3 加载一个 16 位的无符号整数，将其转换为 8 位的无符号整数向量 va3x01010101
    a3 += 2;  // a3 指针向后移动 2 个字节

    const uint8x16_t vb01234567x01 = vld1q_u8(w);
    # 从指针 w 处加载一个 128 位的无符号整数向量 vb01234567x01
    w += 16;  // w 指针向后移动 16 个字节

    vacc0x0123 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc0x0123),
        vmull_u8(va0x01010101, vget_low_u8(vb01234567x01))));
    # 将 vb01234567x01 的低 8 字节扩展为 16 位，与 va0x01010101 的每个字节进行无符号 8 位乘法，然后将结果累加到 vacc0x0123 中

    vacc0x4567 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc0x4567),
        vmull_u8(va0x01010101, vget_high_u8(vb01234567x01))));
    # 将 vb01234567x01 的高 8 字节扩展为 16 位，与 va0x01010101 的每个字节进行无符号 8 位乘法，然后将结果累加到 vacc0x4567 中

    vacc1x0123 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc1x0123),
        vmull_u8(va1x01010101, vget_low_u8(vb01234567x01))));
    # 将 vb01234567x01 的低 8 字节扩展为 16 位，与 va1x01010101 的每个字节进行无符号 8 位乘法，然后将结果累加到 vacc1x0123 中

    vacc1x4567 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc1x4567),
        vmull_u8(va1x01010101, vget_high_u8(vb01234567x01))));
    # 将 vb01234567x01 的高 8 字节扩展为 16 位，与 va1x01010101 的每个字节进行无符号 8 位乘法，然后将结果累加到 vacc1x4567 中

    vacc2x0123 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc2x0123),
        vmull_u8(va2x01010101, vget_low_u8(vb01234567x01))));
    # 将 vb01234567x01 的低 8 字节扩展为 16 位，与 va2x01010101 的每个字节进行无符号 8 位乘法，然后将结果累加到 vacc2x0123 中

    vacc2x4567 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc2x4567),
        vmull_u8(va2x01010101, vget_high_u8(vb01234567x01))));
    # 将 vb01234567x01 的高 8 字节扩展为 16 位，与 va2x01010101 的每个字节进行无符号 8 位乘法，然后将结果累加到 vacc2x4567 中

    vacc3x0123 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc3x0123),
        vmull_u8(va3x01010101, vget_low_u8(vb01234567x01))));
    # 将 vb01234567x01 的低 8 字节扩展为 16 位，与 va3x01010101 的每个字节进行无符号 8 位乘法，然后将结果累加到 vacc3x0123 中

    vacc3x4567 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc3x4567),
        vmull_u8(va3x01010101, vget_high_u8(vb01234567x01))));
    # 将 vb01234567x01 的高 8 字节扩展为 16 位，与 va3x01010101 的每个字节进行无符
    # 使用 NEON 指令进行向量操作，将 vb01234567x0 的高位字节与 va0x00000000 的无符号扩展乘法结果累加到 vacc0x4567 中
    vacc0x4567 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc0x4567),
        vmull_u8(va0x00000000, vget_high_u8(vb01234567x0))));

    # 同上，但是将 vb01234567x0 的低位字节与 va1x00000000 的无符号扩展乘法结果累加到 vacc1x0123 中
    vacc1x0123 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc1x0123),
        vmull_u8(va1x00000000, vget_low_u8(vb01234567x0))));

    # 同上，但是将 vb01234567x0 的高位字节与 va1x00000000 的无符号扩展乘法结果累加到 vacc1x4567 中
    vacc1x4567 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc1x4567),
        vmull_u8(va1x00000000, vget_high_u8(vb01234567x0))));

    # 同上，但是将 vb01234567x0 的低位字节与 va2x00000000 的无符号扩展乘法结果累加到 vacc2x0123 中
    vacc2x0123 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc2x0123),
        vmull_u8(va2x00000000, vget_low_u8(vb01234567x0))));

    # 同上，但是将 vb01234567x0 的高位字节与 va2x00000000 的无符号扩展乘法结果累加到 vacc2x4567 中
    vacc2x4567 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc2x4567),
        vmull_u8(va2x00000000, vget_high_u8(vb01234567x0))));

    # 同上，但是将 vb01234567x0 的低位字节与 va3x00000000 的无符号扩展乘法结果累加到 vacc3x0123 中
    vacc3x0123 = vreinterpretq_s32_u32(vpadalq_u16(
        vreinterpretq_u32_s32(vacc3x0123),
        vmull_u8(va3x00000000, vget_low_u8(vb01234567x0))));
  vacc3x4567 = vreinterpretq_s32_u32(vpadalq_u16(
      vreinterpretq_u32_s32(vacc3x4567),
      vmull_u8(va3x00000000, vget_high_u8(vb01234567x0))));



  const int32x4_t vmultiplier =
      vld1q_dup_s32(&requantization_params->neon.multiplier);



  vacc0x0123 = vqrdmulhq_s32(vacc0x0123, vmultiplier);



  vacc0x4567 = vqrdmulhq_s32(vacc0x4567, vmultiplier);



  vacc1x0123 = vqrdmulhq_s32(vacc1x0123, vmultiplier);



  vacc1x4567 = vqrdmulhq_s32(vacc1x4567, vmultiplier);



  vacc2x0123 = vqrdmulhq_s32(vacc2x0123, vmultiplier);



  vacc2x4567 = vqrdmulhq_s32(vacc2x4567, vmultiplier);



  vacc3x0123 = vqrdmulhq_s32(vacc3x0123, vmultiplier);



  vacc3x4567 = vqrdmulhq_s32(vacc3x4567, vmultiplier);



  const int32x4_t vright_shift =
      vld1q_dup_s32(&requantization_params->neon.right_shift);



  const int32x4_t vzero_shift_mask =
      vreinterpretq_s32_u32(vceqq_s32(vright_shift, vmovq_n_s32(0)));



  vacc0x0123 =
      vsraq_n_s32(vacc0x0123, vbicq_s32(vacc0x0123, vzero_shift_mask), 31);



  vacc0x4567 =
      vsraq_n_s32(vacc0x4567, vbicq_s32(vacc0x4567, vzero_shift_mask), 31);



  vacc1x0123 =
      vsraq_n_s32(vacc1x0123, vbicq_s32(vacc1x0123, vzero_shift_mask), 31);



  vacc1x4567 =
      vsraq_n_s32(vacc1x4567, vbicq_s32(vacc1x4567, vzero_shift_mask), 31);



  vacc2x0123 =
      vsraq_n_s32(vacc2x0123, vbicq_s32(vacc2x0123, vzero_shift_mask), 31);



  vacc2x4567 =
      vsraq_n_s32(vacc2x4567, vbicq_s32(vacc2x4567, vzero_shift_mask), 31);



  vacc3x0123 =
      vsraq_n_s32(vacc3x0123, vbicq_s32(vacc3x0123, vzero_shift_mask), 31);



  vacc3x4567 =
      vsraq_n_s32(vacc3x4567, vbicq_s32(vacc3x4567, vzero_shift_mask), 31);



  vacc0x0123 = vrshlq_s32(vacc0x0123, vright_shift);



  vacc0x4567 = vrshlq_s32(vacc0x4567, vright_shift);



  vacc1x0123 = vrshlq_s32(vacc1x0123, vright_shift);



  vacc1x4567 = vrshlq_s32(vacc1x4567, vright_shift);



  vacc2x0123 = vrshlq_s32(vacc2x0123, vright_shift);



  vacc2x4567 = vrshlq_s32(vacc2x4567, vright_shift);



  vacc3x0123 = vrshlq_s32(vacc3x0123, vright_shift);



  vacc3x4567 = vrshlq_s32(vacc3x4567, vright_shift);



  const int16x8_t vzero_point =
      vld1q_dup_s16(&requantization_params->neon.zero_point);
#ifdef __aarch64__
  // 在 ARM64 架构下，使用 vqaddq_s16 函数对四个 int16x8_t 向量进行饱和加法
  const int16x8_t vacc0x01234567 = vqaddq_s16(
      // 对前半部分 int16x8_t 向量进行转换和饱和操作
      vqmovn_high_s32(vqmovn_s32(vacc0x0123), vacc0x4567), vzero_point);
  const int16x8_t vacc1x01234567 = vqaddq_s16(
      // 对前半部分 int16x8_t 向量进行转换和饱和操作
      vqmovn_high_s32(vqmovn_s32(vacc1x0123), vacc1x4567), vzero_point);
  const int16x8_t vacc2x01234567 = vqaddq_s16(
      // 对后半部分 int16x8_t 向量进行转换和饱和操作
      vqmovn_high_s32(vqmovn_s32(vacc2x0123), vacc2x4567), vzero_point);
  const int16x8_t vacc3x01234567 = vqaddq_s16(
      // 对后半部分 int16x8_t 向量进行转换和饱和操作
      vqmovn_high_s32(vqmovn_s32(vacc3x0123), vacc3x4567), vzero_point);

  // 将两个 vqmovun_s16 转换结果合并成一个 uint8x16_t 向量
  uint8x16_t vout0x01234567_1x01234567 =
      vqmovun_high_s16(vqmovun_s16(vacc0x01234567), vacc1x01234567);
  uint8x16_t vout2x01234567_3x01234567 =
      vqmovun_high_s16(vqmovun_s16(vacc2x01234567), vacc3x01234567);
#else
  // 在非 ARM64 架构下，使用 vcombine_s16 函数对四个 int16x8_t 向量进行转换和饱和加法
  const int16x8_t vacc0x01234567 = vqaddq_s16(
      vcombine_s16(vqmovn_s32(vacc0x0123), vqmovn_s32(vacc0x4567)),
      vzero_point);
  const int16x8_t vacc1x01234567 = vqaddq_s16(
      vcombine_s16(vqmovn_s32(vacc1x0123), vqmovn_s32(vacc1x4567)),
      vzero_point);
  const int16x8_t vacc2x01234567 = vqaddq_s16(
      vcombine_s16(vqmovn_s32(vacc2x0123), vqmovn_s32(vacc2x4567)),
      vzero_point);
  const int16x8_t vacc3x01234567 = vqaddq_s16(
      vcombine_s16(vqmovn_s32(vacc3x0123), vqmovn_s32(vacc3x4567)),
      vzero_point);

  // 将四个 vqmovun_s16 转换结果合并成一个 uint8x16_t 向量
  uint8x16_t vout0x01234567_1x01234567 =
      vcombine_u8(vqmovun_s16(vacc0x01234567), vqmovun_s16(vacc1x01234567));
  uint8x16_t vout2x01234567_3x01234567 =
      vcombine_u8(vqmovun_s16(vacc2x01234567), vqmovun_s16(vacc3x01234567));
#endif

// 加载并复制 requantization_params 结构中的最小值和最大值
const uint8x16_t vmin = vld1q_dup_u8(&requantization_params->neon.min);
const uint8x16_t vmax = vld1q_dup_u8(&requantization_params->neon.max);

// 对 vout0x01234567_1x01234567 和 vout2x01234567_3x01234567 进行饱和操作
vout0x01234567_1x01234567 = vmaxq_u8(vout0x01234567_1x01234567, vmin);
vout2x01234567_3x01234567 = vmaxq_u8(vout2x01234567_3x01234567, vmin);
vout0x01234567_1x01234567 = vminq_u8(vout0x01234567_1x01234567, vmax);
vout2x01234567_3x01234567 = vminq_u8(vout2x01234567_3x01234567, vmax);

// 根据 mr 和 nr 的值选择性地将 vout0x01234567_1x01234567 和 vout2x01234567_3x01234567 写入 c0, c1, c2, c3
uint8_t* c0 = c;
uint8_t* c1 = c0;
if (mr >= 2) {
  c1 += c_stride;
}
uint8_t* c2 = c1;
if (mr > 2) {
  c2 += c_stride;
}
uint8_t* c3 = c2;
if (mr == 4) {
  c3 += c_stride;
}
if (nr == 8) {
  vst1_u8(c0, vget_low_u8(vout0x01234567_1x01234567));
  vst1_u8(c1, vget_high_u8(vout0x01234567_1x01234567));
  vst1_u8(c2, vget_low_u8(vout2x01234567_3x01234567));
  vst1_u8(c3, vget_high_u8(vout2x01234567_3x01234567));
} else {
    // 如果剩余处理的数量大于等于4个元素
    if (nr >= 4) {
      // 将 vout0x01234567_1x01234567 的第0个元素存储到 c0 指向的地址
      vst1q_lane_u32(
          __builtin_assume_aligned(c0, 1),
          vreinterpretq_u32_u8(vout0x01234567_1x01234567),
          0);
      // c0 指针向后移动4个元素
      c0 += 4;
      // 将 vout0x01234567_1x01234567 的第2个元素存储到 c1 指向的地址
      vst1q_lane_u32(
          __builtin_assume_aligned(c1, 1),
          vreinterpretq_u32_u8(vout0x01234567_1x01234567),
          2);
      // c1 指针向后移动4个元素
      c1 += 4;
      // 将 vout2x01234567_3x01234567 的第0个元素存储到 c2 指向的地址
      vst1q_lane_u32(
          __builtin_assume_aligned(c2, 1),
          vreinterpretq_u32_u8(vout2x01234567_3x01234567),
          0);
      // c2 指针向后移动4个元素
      c2 += 4;
      // 将 vout2x01234567_3x01234567 的第2个元素存储到 c3 指向的地址
      vst1q_lane_u32(
          __builtin_assume_aligned(c3, 1),
          vreinterpretq_u32_u8(vout2x01234567_3x01234567),
          2);
      // c3 指针向后移动4个元素
      c3 += 4;
      // 将 vout0x01234567_1x01234567 向左循环移动4个元素
      vout0x01234567_1x01234567 =
          vextq_u8(vout0x01234567_1x01234567, vout0x01234567_1x01234567, 4);
      // 将 vout2x01234567_3x01234567 向左循环移动4个元素
      vout2x01234567_3x01234567 =
          vextq_u8(vout2x01234567_3x01234567, vout2x01234567_3x01234567, 4);
      // 减少剩余处理数量
      nr -= 4;
    }
    // 如果剩余处理的数量大于等于2个元素
    if (nr >= 2) {
      // 将 vout0x01234567_1x01234567 的第0个元素存储到 c0 指向的地址
      vst1q_lane_u16(
          __builtin_assume_aligned(c0, 1),
          vreinterpretq_u16_u8(vout0x01234567_1x01234567),
          0);
      // c0 指针向后移动2个元素
      c0 += 2;
      // 将 vout0x01234567_1x01234567 的第4个元素存储到 c1 指向的地址
      vst1q_lane_u16(
          __builtin_assume_aligned(c1, 1),
          vreinterpretq_u16_u8(vout0x01234567_1x01234567),
          4);
      // c1 指针向后移动2个元素
      c1 += 2;
      // 将 vout2x01234567_3x01234567 的第0个元素存储到 c2 指向的地址
      vst1q_lane_u16(
          __builtin_assume_aligned(c2, 1),
          vreinterpretq_u16_u8(vout2x01234567_3x01234567),
          0);
      // c2 指针向后移动2个元素
      c2 += 2;
      // 将 vout2x01234567_3x01234567 的第4个元素存储到 c3 指向的地址
      vst1q_lane_u16(
          __builtin_assume_aligned(c3, 1),
          vreinterpretq_u16_u8(vout2x01234567_3x01234567),
          4);
      // c3 指针向后移动2个元素
      c3 += 2;
      // 将 vout0x01234567_1x01234567 向左循环移动2个元素
      vout0x01234567_1x01234567 =
          vextq_u8(vout0x01234567_1x01234567, vout0x01234567_1x01234567, 2);
      // 将 vout2x01234567_3x01234567 向左循环移动2个元素
      vout2x01234567_3x01234567 =
          vextq_u8(vout2x01234567_3x01234567, vout2x01234567_3x01234567, 2);
      // 减少剩余处理数量
      nr -= 2;
    }
    // 如果剩余处理的数量不为0
    if (nr != 0) {
      // 将 vout0x01234567_1x01234567 的第0个元素存储到 c0 指向的地址
      vst1q_lane_u8(c0, vout0x01234567_1x01234567, 0);
      // 将 vout0x01234567_1x01234567 的第8个元素存储到 c1 指向的地址
      vst1q_lane_u8(c1, vout0x01234567_1x01234567, 8);
      // 将 vout2x01234567_3x01234567 的第0个元素存储到 c2 指向的地址
      vst1q_lane_u8(c2, vout2x01234567_3x01234567, 0);
      // 将 vout2x01234567_3x01234567 的第8个元素存储到 c3 指向的地址
      vst1q_lane_u8(c3, vout2x01234567_3x01234567, 8);
    }
}



# 这行代码表示一个代码块的结束，通常用于结束一个函数、循环、条件语句或其他代码块。
```