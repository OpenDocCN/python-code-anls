# `.\pytorch\aten\src\ATen\native\cpu\AtomicAddFloat.h`

```
#ifndef ATOMIC_ADD_FLOAT
#define ATOMIC_ADD_FLOAT

#if (defined(__x86_64__) || defined(__i386__) || defined(__aarch64__))
#include <ATen/native/cpu/Intrinsics.h>  // 如果目标架构是 x86_64、i386 或 aarch64，则包含 CPU 指令集头文件
#else
#define _mm_pause()  // 如果不是上述架构，则定义 _mm_pause() 为空宏
#endif

#include <atomic>  // 包含 C++ 原子操作头文件

static inline void cpu_atomic_add_float(float* dst, float fvalue)
{
  typedef union {
    unsigned intV;
    float floatV;
  } uf32_t;  // 定义联合体 uf32_t，用于处理浮点数和整数的类型转换

  uf32_t new_value, old_value;  // 声明新旧值的 uf32_t 类型变量
  std::atomic<unsigned>* dst_intV = (std::atomic<unsigned>*)(dst);  // 将目标地址强制转换为 std::atomic<unsigned>* 类型

  old_value.floatV = *dst;  // 将目标地址处的浮点数值读入 old_value
  new_value.floatV = old_value.floatV + fvalue;  // 计算新的浮点数值

  unsigned* old_intV = (unsigned*)(&old_value.intV);  // 获取 old_value 的整数成员地址
  while (!std::atomic_compare_exchange_strong(dst_intV, old_intV, new_value.intV)) {  // 使用原子操作尝试更新目标地址处的值
#ifdef __aarch64__
    __asm__ __volatile__("yield;" : : : "memory");  // 如果是 aarch64 架构，执行 yield 指令以等待内存操作完成
#else
    _mm_pause();  // 如果不是 aarch64 架构，调用 _mm_pause() 函数来暂停执行
#endif
    old_value.floatV = *dst;  // 重新读取目标地址处的浮点数值
    new_value.floatV = old_value.floatV + fvalue;  // 计算新的浮点数值
  }
}

#endif
```