# `.\pytorch\aten\src\ATen\native\cpu\avx_mathfun.h`

```
#pragma once
/*
   AVX implementation of sin, cos, sincos, exp and log

   Based on "sse_mathfun.h", by Julien Pommier
   http://gruntthepeon.free.fr/ssemath/

   Copyright (C) 2012 Giovanni Garberoglio
   Interdisciplinary Laboratory for Computational Science (LISC)
   Fondazione Bruno Kessler and University of Trento
   via Sommarive, 18
   I-38123 Trento (Italy)

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.

  (this is the zlib license)
*/

#include <ATen/native/cpu/Intrinsics.h>

/* The original source of this file has been modified. */
#if defined(CPU_CAPABILITY_AVX2)

#if defined(__GNUC__)
# define ALIGN32_BEG __attribute__((aligned(32)))
#elif defined(_WIN32)
# define ALIGN32_BEG __declspec(align(32))
#endif

typedef __m256  v8sf; // vector of 8 float (avx2)
typedef __m256i v8si; // vector of 8 int   (avx2)

/* declare some AVX constants -- why can't I figure a better way to do that? */

// Define a constant vector of 8 floats with the same value
#define _PS256_CONST(Name, Val)                                            \
  static const ALIGN32_BEG float _ps256_##Name[8] = { Val, Val, Val, Val, Val, Val, Val, Val }

// Define a constant vector of 8 integers with the same value
#define _PI32_CONST256(Name, Val)                                            \
  static const ALIGN32_BEG int _pi32_256_##Name[8] = { Val, Val, Val, Val, Val, Val, Val, Val }

// Define a constant vector of 8 values of type 'Type' with the same value
#define _PS256_CONST_TYPE(Name, Type, Val)                                 \
  static const ALIGN32_BEG Type _ps256_##Name[8] = { Val, Val, Val, Val, Val, Val, Val, Val }

// Constants for floating point operations
_PS256_CONST(1  , 1.0f);               // Vector of 1.0f
_PS256_CONST(0p5, 0.5f);               // Vector of 0.5f

// Constants related to floating point representation
_PS256_CONST_TYPE(min_norm_pos, int, 0x00800000);      // Smallest non-denormalized float number
_PS256_CONST_TYPE(mant_mask, int, 0x7f800000);          // Mantissa mask
_PS256_CONST_TYPE(inv_mant_mask, int, ~0x7f800000);     // Inverted mantissa mask

// Constants related to integer and floating point bit masks
_PS256_CONST_TYPE(sign_mask, int, (int)0x80000000);     // Sign bit mask
_PS256_CONST_TYPE(inv_sign_mask, int, ~0x80000000);     // Inverted sign bit mask

// Integer constants
_PI32_CONST256(0, 0);                 // Vector of 0
_PI32_CONST256(1, 1);                 // Vector of 1
_PI32_CONST256(inv1, ~1);             // Vector of ~1 (bitwise inverse of 1)
_PI32_CONST256(2, 2);                 // Vector of 2
_PI32_CONST256(4, 4);                 // Vector of 4
_PI32_CONST256(0x7f, 0x7f);           // Vector of 0x7f

// Constants related to mathematical computations
_PS256_CONST(cephes_SQRTHF, 0.707106781186547524);    // Square root of 0.5
_PS256_CONST(cephes_log_p0, 7.0376836292E-2);         // Coefficients for log approximation
_PS256_CONST(cephes_log_p1, - 1.1514610310E-1);
_PS256_CONST(cephes_log_p2, 1.1676998740E-1);
_PS256_CONST(cephes_log_p3, - 1.2420140846E-1);
// 定义一系列常量并初始化，这些常量用于计算自然对数和指数函数
_PS256_CONST(cephes_log_p4, + 1.4249322787E-1);
_PS256_CONST(cephes_log_p5, - 1.6668057665E-1);
_PS256_CONST(cephes_log_p6, + 2.0000714765E-1);
_PS256_CONST(cephes_log_p7, - 2.4999993993E-1);
_PS256_CONST(cephes_log_p8, + 3.3333331174E-1);
_PS256_CONST(cephes_log_q1, -2.12194440e-4);
_PS256_CONST(cephes_log_q2, 0.693359375);

// 定义一些常量，用于计算 AVX2 向量化的自然对数
// 如果 x <= 0，则返回 NaN
inline v8sf log256_ps(v8sf x) {
    v8si imm0;
    v8sf one = *(v8sf*)_ps256_1;

    // 检查 x 是否小于等于零的掩码
    v8sf invalid_mask = _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_LE_OS);

    // 将 x 的值限制为大于最小正常值，去除非规格化值
    x = _mm256_max_ps(x, *(v8sf*)_ps256_min_norm_pos);

    // 计算指数部分，右移 23 位，得到指数值
    imm0 = _mm256_srli_epi32(_mm256_castps_si256(x), 23);

    // 保留小数部分
    x = _mm256_and_ps(x, *(v8sf*)_ps256_inv_mant_mask);
    x = _mm256_or_ps(x, *(v8sf*)_ps256_0p5);

    // 计算偏移后的指数值 e
    imm0 = _mm256_sub_epi32(imm0, *(v8si*)_pi32_256_0x7f);
    v8sf e = _mm256_cvtepi32_ps(imm0);
    e = _mm256_add_ps(e, one);

    // 第二部分处理:
    // 如果 x < SQRTHF，则 e -= 1，同时 x = x + x - 1.0
    // 否则 x = x - 1.0
    v8sf mask = _mm256_cmp_ps(x, *(v8sf*)_ps256_cephes_SQRTHF, _CMP_LT_OS);
    v8sf tmp = _mm256_and_ps(x, mask);
    x = _mm256_sub_ps(x, one);
    e = _mm256_sub_ps(e, _mm256_and_ps(one, mask));
    x = _mm256_add_ps(x, tmp);

    v8sf z = _mm256_mul_ps(x,x);

    // 计算多项式的系数乘积
    v8sf y = *(v8sf*)_ps256_cephes_log_p0;
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, *(v8sf*)_ps256_cephes_log_p1);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, *(v8sf*)_ps256_cephes_log_p2);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, *(v8sf*)_ps256_cephes_log_p3);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, *(v8sf*)_ps256_cephes_log_p4);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, *(v8sf*)_ps256_cephes_log_p5);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, *(v8sf*)_ps256_cephes_log_p6);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, *(v8sf*)_ps256_cephes_log_p7);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, *(v8sf*)_ps256_cephes_log_p8);
    y = _mm256_mul_ps(y, x);

    // 乘以 z
    y = _mm256_mul_ps(y, z);

    // 计算 e 乘以常量 _ps256_cephes_log_q1，加到 y 上
    tmp = _mm256_mul_ps(e, *(v8sf*)_ps256_cephes_log_q1);
    y = _mm256_add_ps(y, tmp);

    // 计算 z 乘以 0.5，从 y 中减去
    tmp = _mm256_mul_ps(z, *(v8sf*)_ps256_0p5);
    y = _mm256_sub_ps(y, tmp);

    // 计算 e 乘以常量 _ps256_cephes_log_q2，加到 x 上
    tmp = _mm256_mul_ps(e, *(v8sf*)_ps256_cephes_log_q2);
    x = _mm256_add_ps(x, y);

    // 将无效掩码应用到 x 上，负参数将返回 NAN
    x = _mm256_or_ps(x, invalid_mask);
    return x;
}

// 定义一系列常量并初始化，这些常量用于计算指数函数
_PS256_CONST(exp_hi,        88.3762626647949f);
_PS256_CONST(exp_lo,        -88.3762626647949f);
_PS256_CONST(cephes_LOG2EF, 1.44269504088896341);
_PS256_CONST(cephes_exp_C1, 0.693359375);
_PS256_CONST(cephes_exp_C2, -2.12194440e-4);
_PS256_CONST(cephes_exp_p0, 1.9875691500E-4);
_PS256_CONST(cephes_exp_p1, 1.3981999507E-3);
// 定义一系列常量，用于 AVX 向量计算，分别对应于 cephes_exp_p2 到 cephes_exp_p5
_PS256_CONST(cephes_exp_p2, 8.3334519073E-3);
_PS256_CONST(cephes_exp_p3, 4.1665795894E-2);
_PS256_CONST(cephes_exp_p4, 1.6666665459E-1);
_PS256_CONST(cephes_exp_p5, 5.0000001201E-1);

// 定义内联函数 exp256_ps，用于计算 AVX 向量 x 的指数函数
inline v8sf exp256_ps(v8sf x) {
  // tmp 用于临时存储向量运算的结果，fx 为 AVX 向量类型的浮点数
  v8sf tmp = _mm256_setzero_ps(), fx;
  // imm0 为 AVX 向量类型的整数，one 为 AVX 向量中的 1.0

  // 将 x 限制在指定的范围内
  x = _mm256_min_ps(x, *(v8sf*)_ps256_exp_hi);
  x = _mm256_max_ps(x, *(v8sf*)_ps256_exp_lo);

  /* express exp(x) as exp(g + n*log(2)) */
  // 将 exp(x) 表示为 exp(g + n*log(2)) 的形式，其中 g = x * LOG2EF + 0.5
  fx = _mm256_mul_ps(x, *(v8sf*)_ps256_cephes_LOG2EF);
  fx = _mm256_add_ps(fx, *(v8sf*)_ps256_0p5);

  /* how to perform a floorf with SSE: just below */
  // 使用 AVX 指令实现 floorf 操作的近似
  //imm0 = _mm256_cvttps_epi32(fx);
  //tmp  = _mm256_cvtepi32_ps(imm0);

  tmp = _mm256_floor_ps(fx);

  /* if greater, subtract 1 */
  // 如果 tmp 大于 fx，则减去 1
  //v8sf mask = _mm256_cmpgt_ps(tmp, fx);
  v8sf mask = _mm256_cmp_ps(tmp, fx, _CMP_GT_OS);
  mask = _mm256_and_ps(mask, one);
  fx = _mm256_sub_ps(tmp, mask);

  // 计算 exp(x) 的近似值，使用 cephes_exp_C1 和 cephes_exp_C2
  tmp = _mm256_mul_ps(fx, *(v8sf*)_ps256_cephes_exp_C1);
  v8sf z = _mm256_mul_ps(fx, *(v8sf*)_ps256_cephes_exp_C2);
  x = _mm256_sub_ps(x, tmp);
  x = _mm256_sub_ps(x, z);

  z = _mm256_mul_ps(x,x);

  // 使用 cephes_exp_p0 到 cephes_exp_p5 计算 exp(x) 的近似值
  v8sf y = *(v8sf*)_ps256_cephes_exp_p0;
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, *(v8sf*)_ps256_cephes_exp_p1);
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, *(v8sf*)_ps256_cephes_exp_p2);
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, *(v8sf*)_ps256_cephes_exp_p3);
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, *(v8sf*)_ps256_cephes_exp_p4);
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, *(v8sf*)_ps256_cephes_exp_p5);
  y = _mm256_mul_ps(y, z);
  y = _mm256_add_ps(y, x);
  y = _mm256_add_ps(y, one);

  /* build 2^n */
  // 将 fx 转换为整数，然后构建 2^n 的近似值
  imm0 = _mm256_cvttps_epi32(fx);
  // another two AVX2 instructions
  imm0 = _mm256_add_epi32(imm0, *(v8si*)_pi32_256_0x7f);
  imm0 = _mm256_slli_epi32(imm0, 23);
  v8sf pow2n = _mm256_castsi256_ps(imm0);
  y = _mm256_mul_ps(y, pow2n);
  return y;
}

// 定义一系列常量，用于 AVX 向量计算，分别对应于 minus_cephes_DP1 到 cephes_FOPI
_PS256_CONST(minus_cephes_DP1, -0.78515625);
_PS256_CONST(minus_cephes_DP2, -2.4187564849853515625e-4);
_PS256_CONST(minus_cephes_DP3, -3.77489497744594108e-8);
_PS256_CONST(sincof_p0, -1.9515295891E-4);
_PS256_CONST(sincof_p1,  8.3321608736E-3);
_PS256_CONST(sincof_p2, -1.6666654611E-1);
_PS256_CONST(coscof_p0,  2.443315711809948E-005);
_PS256_CONST(coscof_p1, -1.388731625493765E-003);
_PS256_CONST(coscof_p2,  4.166664568298827E-002);
_PS256_CONST(cephes_FOPI, 1.27323954473516); // 4 / M_PI


/* evaluation of 8 sines at onces using AVX intrinsics

   The code is the exact rewriting of the cephes sinf function.
   Precision is excellent as long as x < 8192 (I did not bother to
   take into account the special handling they have for greater values
   -- it does not return garbage for arguments over 8192, though, but
   the extra precision is missing).

   Note that it is such that sinf((float)M_PI) = 8.74e-8, which is the
   surprising but correct result.

*/
inline v8sf sin256_ps(v8sf x) { // 定义一个内联函数，计算256位 AVX 向量的 sin 函数近似值，参数 x 是输入的向量
  v8sf xmm1, xmm2 = _mm256_setzero_ps(), xmm3, sign_bit, y; // 声明 AVX 向量 xmm1, xmm2, xmm3, sign_bit, y，并初始化 xmm2 为全零向量

  sign_bit = x; // 将输入向量 x 复制给 sign_bit
  /* 取 x 的绝对值 */
  x = _mm256_and_ps(x, *(v8sf*)_ps256_inv_sign_mask); // 使用 AVX 指令计算 x 的绝对值，_ps256_inv_sign_mask 是一个掩码向量
  /* 提取符号位 (最高位) */
  sign_bit = _mm256_and_ps(sign_bit, *(v8sf*)_ps256_sign_mask); // 提取输入向量 x 的符号位，并保存到 sign_bit 中

  /* 缩放 x 乘以 4/Pi */
  y = _mm256_mul_ps(x, *(v8sf*)_ps256_cephes_FOPI); // 将 x 乘以一个常数来进行缩放，_ps256_cephes_FOPI 是预定义的常数向量

  /*
    开始一系列整数运算，这些运算处于 AVX2 的领域。
    如果没有 AVX2 支持，则使用 SSE2 指令集执行这些操作。
  */

  /* 将 y 的整数部分存储在 imm2 中 */
  imm2 = _mm256_cvttps_epi32(y); // 将 y 向量中的每个浮点数转换为整数，截断到整数，存储在 imm2 中
  /* j=(j+1) & (~1) (参见 cephes 源码) */
  imm2 = _mm256_add_epi32(imm2, *(v8si*)_pi32_256_1); // 将 imm2 中的每个整数加 1，并与一个掩码向量相与
  imm2 = _mm256_and_si256(imm2, *(v8si*)_pi32_256_inv1); // 将 imm2 中的每个整数与另一个掩码向量相与
  y = _mm256_cvtepi32_ps(imm2); // 将 imm2 中的整数转换回浮点数，存储在 y 中

  /* 获取交换符号标志 */
  imm0 = _mm256_and_si256(imm2, *(v8si*)_pi32_256_4); // 将 imm2 中的整数与一个掩码向量相与，用于获取交换符号的标志位
  imm0 = _mm256_slli_epi32(imm0, 29); // 将 imm0 中的每个整数向左逻辑移位 29 位
  /* 获取多项式选择掩码
     在 0 <= x <= Pi/4 范围内选择一个多项式
     在 Pi/4 < x <= Pi/2 范围内选择另一个多项式

     两个分支都将被计算。
  */
  imm2 = _mm256_and_si256(imm2, *(v8si*)_pi32_256_2); // 将 imm2 中的整数与一个掩码向量相与
  imm2 = _mm256_cmpeq_epi32(imm2, *(v8si*)_pi32_256_0); // 比较 imm2 中的整数是否等于另一个常数向量中的值

  v8sf swap_sign_bit = _mm256_castsi256_ps(imm0); // 将整数向量 imm0 转换为浮点数向量
  v8sf poly_mask = _mm256_castsi256_ps(imm2); // 将整数向量 imm2 转换为浮点数向量
  sign_bit = _mm256_xor_ps(sign_bit, swap_sign_bit); // 将 sign_bit 和 swap_sign_bit 中的每个元素进行异或操作

  /* 魔术操作: "扩展精度模数运算"
     x = ((x - y * DP1) - y * DP2) - y * DP3; */
  xmm1 = *(v8sf*)_ps256_minus_cephes_DP1; // 加载预定义常数 DP1
  xmm2 = *(v8sf*)_ps256_minus_cephes_DP2; // 加载预定义常数 DP2
  xmm3 = *(v8sf*)_ps256_minus_cephes_DP3; // 加载预定义常数 DP3
  xmm1 = _mm256_mul_ps(y, xmm1); // 将 y 与 xmm1 中的每个浮点数进行乘法运算
  xmm2 = _mm256_mul_ps(y, xmm2); // 将 y 与 xmm2 中的每个浮点数进行乘法运算
  xmm3 = _mm256_mul_ps(y, xmm3); // 将 y 与 xmm3 中的每个浮点数进行乘法运算
  x = _mm256_add_ps(x, xmm1); // 将 x 与 xmm1 中的每个浮点数进行加法运算
  x = _mm256_add_ps(x, xmm2); // 将 x 与 xmm2 中的每个浮点数进行加法运算
  x = _mm256_add_ps(x, xmm3); // 将 x 与 xmm3 中的每个浮点数进行加法运算

  /* 计算第一个多项式 (0 <= x <= Pi/4) */
  y = *(v8sf*)_ps256_coscof_p0; // 加载预定义的多项式系数向量 p0
  v8sf z = _mm256_mul_ps(x,x); // 将 x 与自身的每个元素进行乘法运算，结果存储在 z 中

  y = _mm256_mul_ps(y, z); // 将 y 与 z 中的每个元素进行乘法运算
  y = _mm256_add_ps(y, *(v8sf*)_ps256_coscof_p1); // 将 y 与预定义的常数向量 p1 中的每个元素进行加法运算
  y = _mm256_mul_ps(y, z); // 将 y 与 z 中的每个元素进行乘法运算
  y = _mm256_add_ps(y, *(v8sf*)_ps256_coscof_p2); // 将 y 与预定义的常数向量 p2 中的每个元素进行加法运算
  y = _mm256_mul_ps(y, z); // 将 y 与 z 中的每个元素进行乘法运算
  y = _mm256_mul_ps(y, z); // 将 y 与 z 中的每个元素再次进行乘法运算
  v8sf tmp = _mm256_mul_ps(z, *(v8sf*)_ps256_0p5); // 将 z 与预定义的常数向量 0.5 中的每个元素进行乘法运算
  y = _mm256_sub_ps(y, tmp); // 将 y 与 tmp 中的每个元素进行减法运算
  y = _mm256_add_ps(y, *(v8sf*)_ps256_1); // 将 y 与预定义的常数向量 1 中的每个元素进行加法运算

  /* 计算第二个多项式 (Pi/4 <= x <= 0) */

  v8sf y2 = *(v8sf*)_ps256_sincof_p0; // 加载预定义的多项式系数向量 p0
  y2 = _mm256_mul_ps(y2, z); // 将 y2 与 z 中的每个元素进行乘法运算
  y2 = _mm256_add_ps(y2, *(v8sf*)_ps256_sincof_p1); // 将 y2 与预定义的常数向量 p1 中的每个元素进行加法运算
  y2 = _mm256_mul_ps(y2, z); // 将 y2 与 z 中的每个元素进行乘法运算
  y2 = _mm256_add_ps(y2, *(v8sf*)_ps256_sincof_p2); // 将 y2 与预定义的常数向量 p2 中的每个元素进行加法运算
  y2 = _mm256_mul_ps(y
inline v8sf cos256_ps(v8sf x) { // 定义内联函数，计算 8 个单精度浮点数向量的余弦值
  v8sf xmm1, xmm2 = _mm256_setzero_ps(), xmm3, y; // 声明 4 个 256 位 AVX 寄存器，其中 xmm2 初始化为全零

  /* 取 x 的绝对值 */
  x = _mm256_and_ps(x, *(v8sf*)_ps256_inv_sign_mask); // 使用位与操作将 x 向量中的符号位置零

  /* 将 x 缩放为 4/Pi */
  y = _mm256_mul_ps(x, *(v8sf*)_ps256_cephes_FOPI); // 将 x 向量乘以预先计算好的 4/Pi 常数向量

  /* 将 y 的整数部分存储在 imm2 中 */
  imm2 = _mm256_cvttps_epi32(y); // 将 y 向量转换为整数向量
  /* j=(j+1) & (~1) (参见 cephes 源码) */
  imm2 = _mm256_add_epi32(imm2, *(v8si*)_pi32_256_1); // 对 imm2 向量中的整数加 1
  imm2 = _mm256_and_si256(imm2, *(v8si*)_pi32_256_inv1); // 对 imm2 向量中的整数进行与操作
  y = _mm256_cvtepi32_ps(imm2); // 将 imm2 中的整数向量转换回单精度浮点数向量
  imm2 = _mm256_sub_epi32(imm2, *(v8si*)_pi32_256_2); // 在 imm2 中的整数向量中减去常数向量中的 2

  /* 获取交换符号标志 */
  imm0 =  _mm256_andnot_si256(imm2, *(v8si*)_pi32_256_4); // 对 imm2 取反后再与另一个常数向量进行与操作
  imm0 = _mm256_slli_epi32(imm0, 29); // 将 imm0 中的整数向量逻辑左移 29 位
  /* 获取多项式选择掩码 */
  imm2 = _mm256_and_si256(imm2, *(v8si*)_pi32_256_2); // 对 imm2 向量中的整数应用与操作
  imm2 = _mm256_cmpeq_epi32(imm2, *(v8si*)_pi32_256_0); // 检查 imm2 中的整数向量是否等于另一个常数向量中的 0

  v8sf sign_bit = _mm256_castsi256_ps(imm0); // 将整数向量 imm0 转换为单精度浮点数向量
  v8sf poly_mask = _mm256_castsi256_ps(imm2); // 将整数向量 imm2 转换为单精度浮点数向量

  /* 魔法步骤: "扩展精度模算术"
     x = ((x - y * DP1) - y * DP2) - y * DP3; */
  xmm1 = *(v8sf*)_ps256_minus_cephes_DP1; // 载入预先计算好的常数向量
  xmm2 = *(v8sf*)_ps256_minus_cephes_DP2; // 载入预先计算好的常数向量
  xmm3 = *(v8sf*)_ps256_minus_cephes_DP3; // 载入预先计算好的常数向量
  xmm1 = _mm256_mul_ps(y, xmm1); // 将 y 与 xmm1 向量中的常数向量相乘
  xmm2 = _mm256_mul_ps(y, xmm2); // 将 y 与 xmm2 向量中的常数向量相乘
  xmm3 = _mm256_mul_ps(y, xmm3); // 将 y 与 xmm3 向量中的常数向量相乘
  x = _mm256_add_ps(x, xmm1); // 将 x 向量与 xmm1 向量中的结果相加
  x = _mm256_add_ps(x, xmm2); // 将 x 向量与 xmm2 向量中的结果相加
  x = _mm256_add_ps(x, xmm3); // 将 x 向量与 xmm3 向量中的结果相加

  /* 计算第一个多项式 (0 <= x <= Pi/4) */
  y = *(v8sf*)_ps256_coscof_p0; // 载入预先计算好的常数向量
  v8sf z = _mm256_mul_ps(x,x); // 计算 x 向量的平方

  y = _mm256_mul_ps(y, z); // 将 y 与 z 向量中的结果相乘
  y = _mm256_add_ps(y, *(v8sf*)_ps256_coscof_p1); // 将 y 向量与常数向量相加
  y = _mm256_mul_ps(y, z); // 将 y 与 z 向量中的结果相乘
  y = _mm256_add_ps(y, *(v8sf*)_ps256_coscof_p2); // 将 y 向量与常数向量相加
  y = _mm256_mul_ps(y, z); // 将 y 与 z 向量中的结果相乘
  y = _mm256_mul_ps(y, z); // 再次将 y 与 z 向量中的结果相乘
  v8sf tmp = _mm256_mul_ps(z, *(v8sf*)_ps256_0p5); // 将 z 向量与常数向量相乘得到临时结果
  y = _mm256_sub_ps(y, tmp); // 将 y 向量与 tmp 向量中的结果相减
  y = _mm256_add_ps(y, *(v8sf*)_ps256_1); // 将 y 向量与常数向量相加

  /* 计算第二个多项式 (Pi/4 <= x <= 0) */

  v8sf y2 = *(v8sf*)_ps256_sincof_p0; // 载入预先计算好的常数向量
  y2 = _mm256_mul_ps(y2, z); // 将 y2 与 z 向量中的结果相乘
  y2 = _mm256_add_ps(y2, *(v8sf*)_ps256_sincof_p1); // 将 y2 向量与常数向量相加
  y2 = _mm256_mul_ps(y2, z); // 将 y2 与 z 向量中的结果相乘
  y2 = _mm256_add_ps(y2, *(v8sf*)_ps256_sincof_p2); // 将 y2 向量与常数向量相加
  y2 = _mm256_mul_ps(y2, z); // 将 y2 与 z 向量中的结果相乘
  y2 = _mm256_mul_ps(y2, x); // 将 y2 与 x 向量中的结果相乘
  y2 = _mm256_add_ps(y2, x); // 将 y2 向量与 x 向量中的结果相加

  /* 从两个多项式中选择正确的结果 */
  xmm3 = poly_mask; // 将 poly_mask 向量赋值给 xmm3
  y2 = _mm256_and_ps(xmm3, y2); // 将 y2 向量与 xmm3 向量进行与操作
  y = _mm256_andnot_ps(xmm3, y); // 对 y 向量应用非与操作
  y = _mm256_add_ps(y,y2); // 将 y 向量与 y2 向量中的结果相加
  /* 更新符号 */
  y = _mm256_xor_ps(y, sign_bit); // 将 y 向量与 sign_bit 向量进行异或操作

  return y; // 返回计算出的余弦值向量
}

/* 由于 sin256_ps 和 cos256_ps 几乎相同，sincos256_ps 可以替代它们两个..
   它几乎与它们一样快，并且在计算正弦值时额外提供了余弦值 */
}

#endif // CPU_CAPABILITY_AVX2
``
```