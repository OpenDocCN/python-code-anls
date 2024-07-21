# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\qnnpack\math.h`

```py
/*
 * 版权声明
 * Facebook, Inc.及其关联公司版权所有。
 *
 * 此源代码根据根目录中的LICENSE文件中所述的BSD风格许可证许可。
 */

#pragma once

#include <stddef.h>
#ifdef _MSC_VER
#undef min
#undef max
#endif

// 返回两个大小值中较小的一个
inline static size_t min(size_t a, size_t b) {
  return a < b ? a : b;
}

// 返回两个大小值中较大的一个
inline static size_t max(size_t a, size_t b) {
  return a > b ? a : b;
}

// 如果a小于b，则返回0；否则返回a减去b的差值
inline static size_t doz(size_t a, size_t b) {
  return a < b ? 0 : a - b;
}

// 将n除以q并向上取整，返回结果
inline static size_t divide_round_up(size_t n, size_t q) {
  return n % q == 0 ? n / q : n / q + 1;
}

// 将n向上舍入到最接近的q的倍数，并返回结果
inline static size_t round_up(size_t n, size_t q) {
  return divide_round_up(n, q) * q;
}
```