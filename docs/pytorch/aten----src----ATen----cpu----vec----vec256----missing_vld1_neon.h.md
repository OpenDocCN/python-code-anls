# `.\pytorch\aten\src\ATen\cpu\vec\vec256\missing_vld1_neon.h`

```
/* 对于在 gcc-7 中缺失的 vld1_*_x2 和 vst1_*_x2 内联函数的补充实现 */

__extension__ extern __inline uint8x8x2_t
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vld1_u8_x2 (const uint8_t *__a)
{
  uint8x8x2_t ret;
  // 使用内联汇编指令 ld1，加载两个 uint8_t 元素到 ret 中
  asm volatile("ld1 {%S0.8b - %T0.8b}, %1" : "=w" (ret) : "Q"(*__a));
  return ret;
}

__extension__ extern __inline int8x8x2_t
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vld1_s8_x2 (const int8_t *__a)
{
  int8x8x2_t ret;
  // 使用内联汇编指令 ld1，加载两个 int8_t 元素到 ret 中
  asm volatile("ld1 {%S0.8b - %T0.8b}, %1" : "=w" (ret) : "Q"(*__a));
  return ret;
}

__extension__ extern __inline uint16x4x2_t
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vld1_u16_x2 (const uint16_t *__a)
{
  uint16x4x2_t ret;
  // 使用内联汇编指令 ld1，加载两个 uint16_t 元素到 ret 中
  asm volatile("ld1 {%S0.4h - %T0.4h}, %1" : "=w" (ret) : "Q"(*__a));
  return ret;
}

__extension__ extern __inline int16x4x2_t
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vld1_s16_x2 (const int16_t *__a)
{
  int16x4x2_t ret;
  // 使用内联汇编指令 ld1，加载两个 int16_t 元素到 ret 中
  asm volatile("ld1 {%S0.4h - %T0.4h}, %1" : "=w" (ret) : "Q"(*__a));
  return ret;
}

__extension__ extern __inline uint32x2x2_t
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vld1_u32_x2 (const uint32_t *__a)
{
  uint32x2x2_t ret;
  // 使用内联汇编指令 ld1，加载两个 uint32_t 元素到 ret 中
  asm volatile("ld1 {%S0.2s - %T0.2s}, %1" : "=w" (ret) : "Q"(*__a));
  return ret;
}

__extension__ extern __inline int32x2x2_t
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vld1_s32_x2 (const int32_t *__a)
{
  int32x2x2_t ret;
  // 使用内联汇编指令 ld1，加载两个 int32_t 元素到 ret 中
  asm volatile("ld1 {%S0.2s - %T0.2s}, %1" : "=w" (ret) : "Q"(*__a));
  return ret;
}

__extension__ extern __inline uint64x1x2_t
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vld1_u64_x2 (const uint64_t *__a)
{
  uint64x1x2_t ret;
  // 使用内联汇编指令 ld1，加载两个 uint64_t 元素到 ret 中
  asm volatile("ld1 {%S0.1d - %T0.1d}, %1" : "=w" (ret) : "Q"(*__a));
  return ret;
}

__extension__ extern __inline int64x1x2_t
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vld1_s64_x2 (const int64_t *__a)
{
  int64x1x2_t ret;
  __builtin_aarch64_simd_oi __o;
  // 使用内联汇编指令 ld1，加载两个 int64_t 元素到 ret 中
  asm volatile("ld1 {%S0.1d - %T0.1d}, %1" : "=w" (ret) : "Q"(*__a));
  return ret;
}

__extension__ extern __inline float16x4x2_t
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vld1_f16_x2 (const float16_t *__a)
{
  float16x4x2_t ret;
  // 使用内联汇编指令 ld1，加载两个 float16_t 元素到 ret 中
  asm volatile("ld1 {%S0.4h - %T0.4h}, %1" : "=w" (ret) : "Q"(*__a));
  return ret;
}

__extension__ extern __inline float32x2x2_t
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vld1_f32_x2 (const float32_t *__a)
{
  float32x2x2_t ret;
  // 使用内联汇编指令 ld1，加载两个 float32_t 元素到 ret 中
  asm volatile("ld1 {%S0.2s - %T0.2s}, %1" : "=w" (ret) : "Q"(*__a));
  return ret;
}

__extension__ extern __inline float64x1x2_t
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vld1_f64_x2 (const float64_t *__a)
{
  float64x1x2_t ret;
  // 使用内联汇编指令 ld1，加载两个 float64_t 元素到 ret 中
  asm volatile("ld1 {%S0.1d - %T0.1d}, %1" : "=w" (ret) : "Q"(*__a));
  return ret;
}

__extension__ extern __inline poly8x8x2_t
__extension__ extern __inline poly8x8x2_t
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vld1_p8_x2 (const poly8_t *__a)
{
  poly8x8x2_t ret;
  // 使用内联汇编指令加载两个连续的poly8_t元素到poly8x8x2_t结构体变量ret中
  asm volatile("ld1 {%S0.8b - %T0.8b}, %1" : "=w" (ret) : "Q"(*__a));
  return ret;
}

__extension__ extern __inline poly16x4x2_t
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vld1_p16_x2 (const poly16_t *__a)
{
  poly16x4x2_t ret;
  // 使用内联汇编指令加载两个连续的poly16_t元素到poly16x4x2_t结构体变量ret中
  asm volatile("ld1 {%S0.4h - %T0.4h}, %1" : "=w" (ret) : "Q"(*__a));
  return ret;
}

__extension__ extern __inline poly64x1x2_t
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vld1_p64_x2 (const poly64_t *__a)
{
  poly64x1x2_t ret;
  // 使用内联汇编指令加载两个连续的poly64_t元素到poly64x1x2_t结构体变量ret中
  asm volatile("ld1 {%S0.1d - %T0.1d}, %1" : "=w" (ret) : "Q"(*__a));
  return ret;
}

__extension__ extern __inline uint8x16x2_t
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vld1q_u8_x2 (const uint8_t *__a)
{
  uint8x16x2_t ret;
  // 使用内联汇编指令加载两个连续的uint8_t元素到uint8x16x2_t结构体变量ret中
  asm volatile("ld1 {%S0.16b - %T0.16b}, %1" : "=w" (ret) : "Q"(*__a));
  return ret;
}

__extension__ extern __inline int8x16x2_t
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vld1q_s8_x2 (const int8_t *__a)
{
  int8x16x2_t ret;
  // 使用内联汇编指令加载两个连续的int8_t元素到int8x16x2_t结构体变量ret中
  asm volatile("ld1 {%S0.16b - %T0.16b}, %1" : "=w" (ret) : "Q"(*__a));
  return ret;
}

__extension__ extern __inline uint16x8x2_t
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vld1q_u16_x2 (const uint16_t *__a)
{
  uint16x8x2_t ret;
  // 使用内联汇编指令加载两个连续的uint16_t元素到uint16x8x2_t结构体变量ret中
  asm volatile("ld1 {%S0.8h - %T0.8h}, %1" : "=w" (ret) : "Q"(*__a));
  return ret;
}

__extension__ extern __inline int16x8x2_t
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vld1q_s16_x2 (const int16_t *__a)
{
  int16x8x2_t ret;
  // 使用内联汇编指令加载两个连续的int16_t元素到int16x8x2_t结构体变量ret中
  asm volatile("ld1 {%S0.8h - %T0.8h}, %1" : "=w" (ret) : "Q"(*__a));
  return ret;
}

__extension__ extern __inline uint32x4x2_t
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vld1q_u32_x2 (const uint32_t *__a)
{
  uint32x4x2_t ret;
  // 使用内联汇编指令加载两个连续的uint32_t元素到uint32x4x2_t结构体变量ret中
  asm volatile("ld1 {%S0.4s - %T0.4s}, %1" : "=w" (ret) : "Q"(*__a));
  return ret;
}

__extension__ extern __inline int32x4x2_t
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vld1q_s32_x2 (const int32_t *__a)
{
  int32x4x2_t ret;
  // 使用内联汇编指令加载两个连续的int32_t元素到int32x4x2_t结构体变量ret中
  asm volatile("ld1 {%S0.4s - %T0.4s}, %1" : "=w" (ret) : "Q"(*__a));
  return ret;
}

__extension__ extern __inline uint64x2x2_t
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vld1q_u64_x2 (const uint64_t *__a)
{
  uint64x2x2_t ret;
  // 使用内联汇编指令加载两个连续的uint64_t元素到uint64x2x2_t结构体变量ret中
  asm volatile("ld1 {%S0.2d - %T0.2d}, %1" : "=w" (ret) : "Q"(*__a));
  return ret;
}

__extension__ extern __inline int64x2x2_t
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vld1q_s64_x2 (const int64_t *__a)
{
  int64x2x2_t ret;
  // 使用内联汇编指令加载两个连续的int64_t元素到int64x2x2_t结构体变量ret中
  asm volatile("ld1 {%S0.2d - %T0.2d}, %1" : "=w" (ret) : "Q"(*__a));
  return ret;
}

__extension__ extern __inline float16x8x2_t
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vld1q_f16_x2 (const float16_t *__a)
{
  float16x8x2_t ret;
  // 使用内联汇编指令加载两个连续的float16_t元素到float16x8x2_t结构体变量ret中
  asm volatile("ld1 {%S0.8h - %T0.8h}, %1" : "=w" (ret) : "Q"(*__a));
  return ret;
}
__extension__ extern __inline float32x4x2_t
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vld1q_f32_x2 (const float32_t *__a)
{
  // 定义返回类型为 float32x4x2_t 的函数 vld1q_f32_x2，接受一个指向 float32_t 的指针参数 __a
  float32x4x2_t ret;
  // 使用内联汇编，将 __a 指向的数据加载到 ret 结构体中
  asm volatile("ld1 {%S0.4s - %T0.4s}, %1" : "=w" (ret) : "Q"(*__a));
  // 返回加载后的结果
  return ret;
}

__extension__ extern __inline float64x2x2_t
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vld1q_f64_x2 (const float64_t *__a)
{
  // 定义返回类型为 float64x2x2_t 的函数 vld1q_f64_x2，接受一个指向 float64_t 的指针参数 __a
  float64x2x2_t ret;
  // 使用内联汇编，将 __a 指向的数据加载到 ret 结构体中
  asm volatile("ld1 {%S0.2d - %T0.2d}, %1" : "=w" (ret) : "Q"(*__a));
  // 返回加载后的结果
  return ret;
}

__extension__ extern __inline poly8x16x2_t
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vld1q_p8_x2 (const poly8_t *__a)
{
  // 定义返回类型为 poly8x16x2_t 的函数 vld1q_p8_x2，接受一个指向 poly8_t 的指针参数 __a
  poly8x16x2_t ret;
  // 使用内联汇编，将 __a 指向的数据加载到 ret 结构体中
  asm volatile("ld1 {%S0.16b - %T0.16b}, %1" : "=w" (ret) : "Q"(*__a));
  // 返回加载后的结果
  return ret;
}

__extension__ extern __inline poly16x8x2_t
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vld1q_p16_x2 (const poly16_t *__a)
{
  // 定义返回类型为 poly16x8x2_t 的函数 vld1q_p16_x2，接受一个指向 poly16_t 的指针参数 __a
  poly16x8x2_t ret;
  // 使用内联汇编，将 __a 指向的数据加载到 ret 结构体中
  asm volatile("ld1 {%S0.8h - %T0.8h}, %1" : "=w" (ret) : "Q"(*__a));
  // 返回加载后的结果
  return ret;
}

__extension__ extern __inline poly64x2x2_t
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vld1q_p64_x2 (const poly64_t *__a)
{
  // 定义返回类型为 poly64x2x2_t 的函数 vld1q_p64_x2，接受一个指向 poly64_t 的指针参数 __a
  poly64x2x2_t ret;
  // 使用内联汇编，将 __a 指向的数据加载到 ret 结构体中
  asm volatile("ld1 {%S0.2d - %T0.2d}, %1" : "=w" (ret) : "Q"(*__a));
  // 返回加载后的结果
  return ret;
}

/* vst1x2 */

__extension__ extern __inline void
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vst1_s64_x2 (int64_t * __a, int64x1x2_t val)
{
  // 定义无返回值的函数 vst1_s64_x2，接受一个 int64_t 类型的指针参数 __a 和一个 int64x1x2_t 类型的结构体参数 val
  // 使用内联汇编，将 val 结构体中的数据存储到 __a 指向的位置
  asm volatile("st1 {%S1.1d - %T1.1d}, %0" : "=Q" (*__a) : "w" (val));
}

__extension__ extern __inline void
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vst1_u64_x2 (uint64_t * __a, uint64x1x2_t val)
{
  // 定义无返回值的函数 vst1_u64_x2，接受一个 uint64_t 类型的指针参数 __a 和一个 uint64x1x2_t 类型的结构体参数 val
  // 使用内联汇编，将 val 结构体中的数据存储到 __a 指向的位置
  asm volatile("st1 {%S1.1d - %T1.1d}, %0" : "=Q" (*__a) : "w" (val));
}

__extension__ extern __inline void
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vst1_f64_x2 (float64_t * __a, float64x1x2_t val)
{
  // 定义无返回值的函数 vst1_f64_x2，接受一个 float64_t 类型的指针参数 __a 和一个 float64x1x2_t 类型的结构体参数 val
  // 使用内联汇编，将 val 结构体中的数据存储到 __a 指向的位置
  asm volatile("st1 {%S1.1d - %T1.1d}, %0" : "=Q" (*__a) : "w" (val));
}

__extension__ extern __inline void
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vst1_s8_x2 (int8_t * __a, int8x8x2_t val)
{
  // 定义无返回值的函数 vst1_s8_x2，接受一个 int8_t 类型的指针参数 __a 和一个 int8x8x2_t 类型的结构体参数 val
  // 使用内联汇编，将 val 结构体中的数据存储到 __a 指向的位置
  asm volatile("st1 {%S1.8b - %T1.8b}, %0" : "=Q" (*__a) : "w" (val));
}

__extension__ extern __inline void
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vst1_p8_x2 (poly8_t * __a, poly8x8x2_t val)
{
  // 定义无返回值的函数 vst1_p8_x2，接受一个 poly8_t 类型的指针参数 __a 和一个 poly8x8x2_t 类型的结构体参数 val
  // 使用内联汇编，将 val 结构体中的数据存储到 __a 指向的位置
  asm volatile("st1 {%S1.8b - %T1.8b}, %0" : "=Q" (*__a) : "w" (val));
}

__extension__ extern __inline void
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vst1_s16_x2 (int16_t * __a, int16x4x2_t val)
{
  // 定义无返回值的函数 vst1_s16_x2，接受一个 int16_t 类型的指针参数 __a 和一个 int16x4x2_t 类型的结构体参数 val
  // 使用内联汇编，将 val 结构体中的数据存储到 __a 指向的位置
  asm volatile("st1 {%S1.4h - %T1.4h}, %0" : "=Q" (*__a) : "w" (val));
}

__extension__ extern __inline void
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vst1_p16_x2 (poly16_t * __a, poly16x4x2_t val)
{
  // 定义无返回值的函数 vst1_p16_x2，接受一个 poly16_t 类型的指针参数 __a 和一个 poly16x4x2_t 类型的结构体参数 val
  // 使用内联汇编，将 val 结构体中的数据存储到 __a 指向的位置
  asm volatile("st1 {%S1.4h -
# 将 int32x2x2_t 类型的数据存储到 int32_t 类型的指针所指向的内存位置
__extension__ extern __inline void
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vst1_s32_x2 (int32_t * __a, int32x2x2_t val)
{
  # 使用内联汇编指令将 val 中的数据存储到 *__a 所指向的内存位置
  asm volatile("st1 {%S1.2s - %T1.2s}, %0" : "=Q" (*__a) : "w" (val));
}

# 将 uint8x8x2_t 类型的数据存储到 uint8_t 类型的指针所指向的内存位置
__extension__ extern __inline void
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vst1_u8_x2 (uint8_t * __a, uint8x8x2_t val)
{
  # 使用内联汇编指令将 val 中的数据存储到 *__a 所指向的内存位置
  asm volatile("st1 {%S1.8b - %T1.8b}, %0" : "=Q" (*__a) : "w" (val));
}

# 将 uint16x4x2_t 类型的数据存储到 uint16_t 类型的指针所指向的内存位置
__extension__ extern __inline void
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vst1_u16_x2 (uint16_t * __a, uint16x4x2_t val)
{
  # 使用内联汇编指令将 val 中的数据存储到 *__a 所指向的内存位置
  asm volatile("st1 {%S1.4h - %T1.4h}, %0" : "=Q" (*__a) : "w" (val));
}

# 将 uint32x2x2_t 类型的数据存储到 uint32_t 类型的指针所指向的内存位置
__extension__ extern __inline void
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vst1_u32_x2 (uint32_t * __a, uint32x2x2_t val)
{
  # 使用内联汇编指令将 val 中的数据存储到 *__a 所指向的内存位置
  asm volatile("st1 {%S1.2s - %T1.2s}, %0" : "=Q" (*__a) : "w" (val));
}

# 将 float16x4x2_t 类型的数据存储到 float16_t 类型的指针所指向的内存位置
__extension__ extern __inline void
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vst1_f16_x2 (float16_t * __a, float16x4x2_t val)
{
  # 使用内联汇编指令将 val 中的数据存储到 *__a 所指向的内存位置
  asm volatile("st1 {%S1.4h - %T1.4h}, %0" : "=Q" (*__a) : "w" (val));
}

# 将 float32x2x2_t 类型的数据存储到 float32_t 类型的指针所指向的内存位置
__extension__ extern __inline void
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vst1_f32_x2 (float32_t * __a, float32x2x2_t val)
{
  # 使用内联汇编指令将 val 中的数据存储到 *__a 所指向的内存位置
  asm volatile("st1 {%S1.2s - %T1.2s}, %0" : "=Q" (*__a) : "w" (val));
}

# 将 poly64x1x2_t 类型的数据存储到 poly64_t 类型的指针所指向的内存位置
__extension__ extern __inline void
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vst1_p64_x2 (poly64_t * __a, poly64x1x2_t val)
{
  # 使用内联汇编指令将 val 中的数据存储到 *__a 所指向的内存位置
  asm volatile("st1 {%S1.1d - %T1.1d}, %0" : "=Q" (*__a) : "w" (val));
}

# 将 int8x16x2_t 类型的数据存储到 int8_t 类型的指针所指向的内存位置
__extension__ extern __inline void
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vst1q_s8_x2 (int8_t * __a, int8x16x2_t val)
{
  # 使用内联汇编指令将 val 中的数据存储到 *__a 所指向的内存位置
  asm volatile("st1 {%S1.16b - %T1.16b}, %0" : "=Q" (*__a) : "w" (val));
}

# 将 poly8x16x2_t 类型的数据存储到 poly8_t 类型的指针所指向的内存位置
__extension__ extern __inline void
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vst1q_p8_x2 (poly8_t * __a, poly8x16x2_t val)
{
  # 使用内联汇编指令将 val 中的数据存储到 *__a 所指向的内存位置
  asm volatile("st1 {%S1.16b - %T1.16b}, %0" : "=Q" (*__a) : "w" (val));
}

# 将 int16x8x2_t 类型的数据存储到 int16_t 类型的指针所指向的内存位置
__extension__ extern __inline void
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vst1q_s16_x2 (int16_t * __a, int16x8x2_t val)
{
  # 使用内联汇编指令将 val 中的数据存储到 *__a 所指向的内存位置
  asm volatile("st1 {%S1.8h - %T1.8h}, %0" : "=Q" (*__a) : "w" (val));
}

# 将 poly16x8x2_t 类型的数据存储到 poly16_t 类型的指针所指向的内存位置
__extension__ extern __inline void
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vst1q_p16_x2 (poly16_t * __a, poly16x8x2_t val)
{
  # 使用内联汇编指令将 val 中的数据存储到 *__a 所指向的内存位置
  asm volatile("st1 {%S1.8h - %T1.8h}, %0" : "=Q" (*__a) : "w" (val));
}

# 将 int32x4x2_t 类型的数据存储到 int32_t 类型的指针所指向的内存位置
__extension__ extern __inline void
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vst1q_s32_x2 (int32_t * __a, int32x4x2_t val)
{
  # 使用内联汇编指令将 val 中的数据存储到 *__a 所指向的内存位置
  asm volatile("st1 {%S1.4s - %T1.4s}, %0" : "=Q" (*__a) : "w" (val));
}

# 将 int64x2x2_t 类型的数据存储到 int64_t 类型的指针所指向的内存位置
__extension__ extern __inline void
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vst1q_s64_x2 (int64_t * __a, int64x2x2_t val)
{
  # 使用内联汇编指令将 val 中的数据存储到 *__a 所指向的内存位置
  asm volatile("st1 {%S1.2d - %T1.2d}, %0" : "=Q" (*__a) : "w" (val));
}

# 将 uint8x16x2_t 类型的数据存储到 uint8_t 类型的指针所指向的内存位置
__extension__ extern __inline void
__attribute__ ((__always_inline__,
{
  asm volatile("st1 {%S1.16b - %T1.16b}, %0" : "=Q" (*__a) : "w" (val));
}
# 使用内联汇编语句将val中的数据存储到__a指向的地址中，使用指定的数据类型和格式

__extension__ extern __inline void
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vst1q_u16_x2 (uint16_t * __a, uint16x8x2_t val)
{
  asm volatile("st1 {%S1.8h - %T1.8h}, %0" : "=Q" (*__a) : "w" (val));
}
# 内联函数，将val中的两个uint16x8x2_t类型的值存储到__a指向的地址中，使用指定的数据类型和格式

__extension__ extern __inline void
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vst1q_u32_x2 (uint32_t * __a, uint32x4x2_t val)
{
  asm volatile("st1 {%S1.4s - %T1.4s}, %0" : "=Q" (*__a) : "w" (val));
}
# 内联函数，将val中的两个uint32x4x2_t类型的值存储到__a指向的地址中，使用指定的数据类型和格式

__extension__ extern __inline void
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vst1q_u64_x2 (uint64_t * __a, uint64x2x2_t val)
{
  asm volatile("st1 {%S1.2d - %T1.2d}, %0" : "=Q" (*__a) : "w" (val));
}
# 内联函数，将val中的两个uint64x2x2_t类型的值存储到__a指向的地址中，使用指定的数据类型和格式

__extension__ extern __inline void
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vst1q_f16_x2 (float16_t * __a, float16x8x2_t val)
{
  asm volatile("st1 {%S1.8h - %T1.8h}, %0" : "=Q" (*__a) : "w" (val));
}
# 内联函数，将val中的两个float16x8x2_t类型的值存储到__a指向的地址中，使用指定的数据类型和格式

__extension__ extern __inline void
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vst1q_f32_x2 (float32_t * __a, float32x4x2_t val)
{
  asm volatile("st1 {%S1.4s - %T1.4s}, %0" : "=Q" (*__a) : "w" (val));
}
# 内联函数，将val中的两个float32x4x2_t类型的值存储到__a指向的地址中，使用指定的数据类型和格式

__extension__ extern __inline void
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vst1q_f64_x2 (float64_t * __a, float64x2x2_t val)
{
  asm volatile("st1 {%S1.2d - %T1.2d}, %0" : "=Q" (*__a) : "w" (val));
}
# 内联函数，将val中的两个float64x2x2_t类型的值存储到__a指向的地址中，使用指定的数据类型和格式

__extension__ extern __inline void
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vst1q_p64_x2 (poly64_t * __a, poly64x2x2_t val)
{
  asm volatile("st1 {%S1.2d - %T1.2d}, %0" : "=Q" (*__a) : "w" (val));
}
# 内联函数，将val中的两个poly64x2x2_t类型的值存储到__a指向的地址中，使用指定的数据类型和格式
```