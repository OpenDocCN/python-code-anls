# `.\pytorch\aten\src\ATen\native\Pow.h`

```py
#pragma once
// 如果未定义 __CUDACC__ 或者 __HIPCC__ 宏，则定义 HOST_DEVICE 为空
#include <ATen/native/DispatchStub.h>

// 声明 c10 命名空间下的 Scalar 类
namespace c10 {
class Scalar;
}

// 声明 at 命名空间
namespace at {

// 声明 TensorIterator 和 TensorIteratorBase 结构体
struct TensorIterator;
struct TensorIteratorBase;

// 声明 native 命名空间
namespace native {

// 如果在 CUDA 或者 HIP 环境下编译，定义 HOST_DEVICE 宏为 __host__ __device__
#if defined(__CUDACC__) || defined(__HIPCC__)
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif

// 定义 powi_impl 模板函数，用于整数类型的幂运算实现
// __ubsan_ignore_signed_int_overflow__ 是编译器指令，忽略有符号整数溢出检查
template <class T,
  typename std::enable_if<std::is_integral<T>::value, T>::type* = nullptr>
inline HOST_DEVICE __ubsan_ignore_signed_int_overflow__ T powi_impl(T a, T b) {
  T result = 1;
  // 循环计算 a 的 b 次幂
  while (b) {
    if (b & 1) {
       result *= a;  // 如果 b 的最低位为 1，则累乘当前 a 到结果中
    }
    b /= 2;  // 将 b 右移一位，相当于除以 2
    a *= a;  // 计算 a 的平方
  }
  return result;  // 返回计算结果
}

// 定义 powi 模板函数，用于处理整数类型的幂运算
template <class T,
  typename std::enable_if<std::is_integral<T>::value && !std::is_signed<T>::value, T>::type* = nullptr>
inline HOST_DEVICE T powi(T a, T b) {
  return powi_impl(a, b);  // 调用 powi_impl 函数进行幂运算
}

// 定义 powi 模板函数的有符号整数版本，处理特殊情况
template <class T,
  typename std::enable_if<std::is_integral<T>::value && std::is_signed<T>::value, T>::type* = nullptr>
inline HOST_DEVICE T powi(T a, T b) {
  if ( b < 0 ) {  // 如果指数 b 为负数
      if ( a == 1 ) {  // 如果底数 a 为 1
          return 1;  // 返回 1
      } else if ( a == -1 ) {  // 如果底数 a 为 -1
          auto negative = (-b) % static_cast<T>(2);  // 计算指数 b 取绝对值后除以 2 的余数
          return negative ? -1 : 1;  // 如果余数非零返回 -1，否则返回 1
      } else {
          return 0;  // 其他情况返回 0
      }
  }
  return powi_impl(a, b);  // 否则调用 powi_impl 进行幂运算
}

// 声明 pow_tensor_tensor_fn 类型的函数指针，用于处理张量迭代器之间的幂运算
using pow_tensor_tensor_fn = void (*)(TensorIteratorBase&);
// 声明 pow_tensor_scalar_fn 类型的函数指针，用于处理张量迭代器和标量之间的幂运算
using pow_tensor_scalar_fn = void (*)(TensorIteratorBase&, const c10::Scalar&);

// 声明具体的分发函数，用于处理张量迭代器之间的幂运算
DECLARE_DISPATCH(pow_tensor_tensor_fn, pow_tensor_tensor_stub);
// 声明具体的分发函数，用于处理张量迭代器和标量之间的幂运算
DECLARE_DISPATCH(pow_tensor_scalar_fn, pow_tensor_scalar_stub);

} // namespace native

} // namespace at
```