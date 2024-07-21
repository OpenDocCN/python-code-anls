# `.\pytorch\aten\src\ATen\native\Scalar.cpp`

```
// 定义宏，仅包含方法操作符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 包含 ATen 库中的 Tensor 类和分发机制头文件
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch_v2.h>

// 根据不同的编译宏条件，选择性地包含 ATen 库的函数头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_local_scalar_dense.h>
#include <ATen/ops/_local_scalar_dense_native.h>
#include <ATen/ops/item_native.h>
#endif

// ATen 库的命名空间 at::native
namespace at::native {

// 函数 item，用于获取 Tensor 的标量值
Scalar item(const Tensor& self) {
  // 获取 Tensor 的元素数量
  auto numel = self.sym_numel();
  // 使用 TORCH_CHECK 检查元素数量是否为 1，如果不是则抛出异常
  TORCH_CHECK(numel == 1, "a Tensor with ", numel, " elements cannot be converted to Scalar");
  
  // 如果 Tensor 是稀疏的
  if (self.is_sparse()) {
    // 如果稀疏 Tensor 的非零元素数量为 0，则返回标量值 0
    if (self._nnz() == 0) return Scalar(0);
    // 如果稀疏 Tensor 已经整理(coalesced)，直接返回其本地标量稠密表示
    if (self.is_coalesced()) return at::_local_scalar_dense(self._values());
    // 否则，返回稀疏 Tensor _values() 的和的本地标量稠密表示
    return at::_local_scalar_dense(self._values().sum());
  
  // 如果 Tensor 是量化的
  } else if (self.is_quantized()) {
    // 返回 Tensor 的反量化值的标量表示
    return self.dequantize().item();
  
  // 对于其他类型的 Tensor
  } else {
    // 返回 Tensor 的本地标量稠密表示
    return _local_scalar_dense(self);
  }
}

// 定义基础类型宏 AT_SD_BASE_TYPES，扩展了所有的基本数据类型和复杂数据类型
#define AT_SD_BASE_TYPES AT_EXPAND(AT_ALL_TYPES), AT_EXPAND(AT_COMPLEX_TYPES), kComplexHalf, kHalf, kBool, kBFloat16, AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES)
// 根据不同的宏定义条件，选择性地定义 AT_SD_TYPES
#if !defined(C10_MOBILE)
#define AT_SD_TYPES AT_EXPAND(AT_SD_BASE_TYPES), AT_EXPAND(AT_FLOAT8_TYPES)
#else
#define AT_SD_TYPES AT_EXPAND(AT_SD_BASE_TYPES)
#endif

// 函数 _local_scalar_dense_cpu，用于在 CPU 上获取 Tensor 的本地标量稠密表示
Scalar _local_scalar_dense_cpu(const Tensor& self) {
  // 声明变量 r 用于存储标量结果
  Scalar r;
  // 使用 AT_DISPATCH_V2 宏根据 Tensor 的数据类型分发操作
  AT_DISPATCH_V2(
    self.scalar_type(), // Tensor 的标量类型
    "_local_scalar_dense_cpu", // 分发函数的名称字符串
    AT_WRAP([&] { // 包装 Lambda 函数，处理标量数据
      scalar_t value = *self.const_data_ptr<scalar_t>(); // 获取 Tensor 数据指针，并解引用赋值给 value
      r = Scalar(value); // 将 value 转换为 Scalar 类型，并赋值给 r
    }),
    AT_EXPAND(AT_SD_TYPES) // 展开标量类型列表
  );
  return r; // 返回标量结果
}

} // at::native
```