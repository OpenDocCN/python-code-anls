# `.\pytorch\c10\util\safe_numerics.h`

```
#pragma once
#include <c10/macros/Macros.h>  // 包含 c10 库的宏定义

#include <cstdint>  // 包含 C++ 标准头文件 <cstdint>

// GCC 之前版本支持 __builtin_mul_overflow，但在支持 __has_builtin 之前
#ifdef _MSC_VER  // 如果是 Microsoft Visual C++ 编译器
#define C10_HAS_BUILTIN_OVERFLOW() (0)  // 定义 C10_HAS_BUILTIN_OVERFLOW 宏为 0
#include <c10/util/llvmMathExtras.h>  // 包含 c10 库的 LLVM 数学扩展
#include <intrin.h>  // 包含 Microsoft Visual C++ 的内联汇编头文件
#else  // 对于其他编译器（假设是 GCC 或类似的）
#define C10_HAS_BUILTIN_OVERFLOW() (1)  // 定义 C10_HAS_BUILTIN_OVERFLOW 宏为 1
#endif

namespace c10 {

// 声明 C10_ALWAYS_INLINE 宏，确保函数 inline 化，提高执行效率，并定义 add_overflows 函数
C10_ALWAYS_INLINE bool add_overflows(uint64_t a, uint64_t b, uint64_t* out) {
#if C10_HAS_BUILTIN_OVERFLOW()
  return __builtin_add_overflow(a, b, out);  // 使用内建函数处理加法溢出检测
#else
  unsigned long long tmp;
#if defined(_M_IX86) || defined(_M_X64)  // 如果是 x86 或 x64 架构
  auto carry = _addcarry_u64(0, a, b, &tmp);  // 使用内联汇编执行加法，处理进位
#else
  tmp = a + b;  // 普通加法操作
  unsigned long long vector = (a & b) ^ ((a ^ b) & ~tmp);  // 计算进位向量
  auto carry = vector >> 63;  // 提取最高位作为进位标志
#endif
  *out = tmp;  // 将结果存入 out 指向的地址
  return carry;  // 返回进位标志
#endif
}

// 声明 C10_ALWAYS_INLINE 宏，定义 mul_overflows 函数用于处理乘法溢出检测（对于 uint64_t 类型）
C10_ALWAYS_INLINE bool mul_overflows(uint64_t a, uint64_t b, uint64_t* out) {
#if C10_HAS_BUILTIN_OVERFLOW()
  return __builtin_mul_overflow(a, b, out);  // 使用内建函数处理乘法溢出检测
#else
  *out = a * b;  // 普通乘法操作
  // 这个测试并不精确，但避免了整数除法操作
  return (
      (c10::llvm::countLeadingZeros(a) + c10::llvm::countLeadingZeros(b)) < 64);  // 返回是否溢出的估计值
#endif
}

// 声明 C10_ALWAYS_INLINE 宏，定义 mul_overflows 函数用于处理乘法溢出检测（对于 int64_t 类型）
C10_ALWAYS_INLINE bool mul_overflows(int64_t a, int64_t b, int64_t* out) {
#if C10_HAS_BUILTIN_OVERFLOW()
  return __builtin_mul_overflow(a, b, out);  // 使用内建函数处理乘法溢出检测
#else
  volatile int64_t tmp = a * b;  // 使用 volatile 修饰的临时变量存储乘法结果
  *out = tmp;  // 将结果存入 out 指向的地址
  if (a == 0 || b == 0) {  // 如果其中一个操作数为 0
    return false;  // 不会溢出
  }
  return !(a == tmp / b);  // 返回是否溢出的判断结果
#endif
}

// 定义模板函数 safe_multiplies_u64，处理多个 uint64_t 类型数值的安全乘法
template <typename It>
bool safe_multiplies_u64(It first, It last, uint64_t* out) {
#if C10_HAS_BUILTIN_OVERFLOW()
  uint64_t prod = 1;  // 初始化乘积为 1
  bool overflow = false;  // 溢出标志初始化为 false
  for (; first != last; ++first) {
    overflow |= c10::mul_overflows(prod, *first, &prod);  // 调用 mul_overflows 处理乘法溢出
  }
  *out = prod;  // 将最终乘积存入 out 指向的地址
  return overflow;  // 返回溢出标志
#else
  uint64_t prod = 1;  // 初始化乘积为 1
  uint64_t prod_log2 = 0;  // 初始化乘积的 log2 值为 0
  bool is_zero = false;  // 是否有操作数为 0 的标志初始化为 false
  for (; first != last; ++first) {
    auto x = static_cast<uint64_t>(*first);  // 将迭代器指向的值转换为 uint64_t 类型
    prod *= x;  // 计算乘积
    // log2(0) 不合法，需特别处理
    is_zero |= (x == 0);  // 更新是否有操作数为 0 的标志
    prod_log2 += c10::llvm::Log2_64_Ceil(x);  // 更新 log2 值
  }
  *out = prod;  // 将最终乘积存入 out 指向的地址
  // 这个测试并不精确，但避免了整数除法操作
  return !is_zero && (prod_log2 >= 64);  // 返回是否溢出的估计值
#endif
}

// 定义模板函数 safe_multiplies_u64，处理容器内多个 uint64_t 类型数值的安全乘法
template <typename Container>
bool safe_multiplies_u64(const Container& c, uint64_t* out) {
  return safe_multiplies_u64(c.begin(), c.end(), out);  // 调用范围版本的 safe_multiplies_u64 函数
}

} // namespace c10  // 结束 c10 命名空间
```