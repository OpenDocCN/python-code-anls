# `.\pytorch\aten\src\ATen\cpu\vec\vec256\missing_vst1_neon.h`

```py
/* Workaround for missing vst1q_f32_x2 in gcc-8.  */

__extension__ extern __inline void
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vst1q_f32_x2 (float32_t * __a, float32x4x2_t val)
{
  // 使用内联汇编实现的函数，将两个 float32x4x2_t 类型的值存储到地址 __a 处
  asm volatile("st1 {%S1.4s - %T1.4s}, %0" : "=Q" (*__a) : "w" (val));
}
```