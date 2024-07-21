# `.\pytorch\aten\src\ATen\templates\Functions.h`

```
#pragma once

// ${generated_comment}

#ifdef TORCH_ASSERT_NO_OPERATORS
#error This change adds a dependency on native_functions.yaml,            \
  meaning the file will need to be re-compiled every time an operator     \
  is changed or added. Consider if your change would be better placed in  \
  another file, or if a more specific header might achieve the same goal. \
  See NOTE: [Tensor vs. TensorBase]
#endif

#if defined(AT_PER_OPERATOR_HEADERS) && defined(TORCH_ASSERT_ONLY_METHOD_OPERATORS)
#error This change adds a dependency on all pytorch operators, meaning the     \
  file will need to be re-compiled every time an operator is changed or added. \
  Consider including a specific operator from <ATen/ops/{my_operator}.h> and   \
  see NOTE [TORCH_ASSERT_ONLY_METHOD_OPERATORS].
#endif

// NOTE: [TORCH_ASSERT_ONLY_METHOD_OPERATORS]
//
// In ATen, certain generated headers files include the definitions of
// every single operator in PyTorch. Unfortunately this means every
// time an operator signature is updated or changed in
// native_functions.yaml, you (and every other PyTorch developer) need
// to recompile every source file that includes any of these headers.
//
// To break up these header dependencies, and improve incremental
// build times for all PyTorch developers. These headers are split
// into per-operator headers in the `ATen/ops` folder. This limits
// incremental builds to only changes to methods of `Tensor`, or files
// that use the specific operator being changed. With `at::sum` as an
// example, you should include
//
//   <ATen/ops/sum.h>               // instead of ATen/Functions.h
//   <ATen/ops/sum_native.h>        // instead of ATen/NativeFunctions.h
//   <ATen/ops/sum_ops.h>           // instead of ATen/Operators.h
//   <ATen/ops/sum_cpu_dispatch.h>  // instead of ATen/CPUFunctions.h
//
// However, even if you're careful to use this in your own code.
// `Functions.h` might be included indirectly through another header
// without you realising. To avoid this, you can add
//
//   #define TORCH_ASSERT_ONLY_METHOD_OPERATORS
//
// to the top of your source file. This way any time the non-specific
// headers are included, the compiler will error out.
//
// Also, be aware that `ops` are not available in all build
// configurations (namely fb-internal) so you must guard these
// includes with `#ifdef AT_PER_OPERATOR_HEADERS`. e.g.
//
//   #ifndef AT_PER_OPERATOR_HEADERS
//   #include <ATen/Functions.h>
//   #else
//   #include <ATen/ops/sum.h>
//   #endif

#include <ATen/Context.h>
#include <ATen/DeviceGuard.h>
#include <ATen/TensorUtils.h>
#include <ATen/TracerMode.h>
#include <ATen/core/Generator.h>
#include <ATen/core/Reduction.h>
#include <c10/core/SymInt.h>
#include <ATen/core/Tensor.h>
#include <c10/core/Scalar.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/Deprecated.h>
#include <c10/util/Optional.h>
#include <c10/util/OptionalArrayRef.h>

#include <ATen/ops/from_blob.h>
// 包含 ATen 库中的 tensor.h 文件，提供了与张量操作相关的函数和类
#include <ATen/ops/tensor.h>

// 引入命名空间 at，用于访问 ATen 库中的函数和类
namespace at {

// 声明 ATen 库中的函数

// 定义 TORCH_API 修饰符下的 inline 函数，计算张量在指定维度上的方差
TORCH_API inline Tensor var(const Tensor& self, int dim) {
  // 调用 var 函数，传入 IntArrayRef 类型的参数，转换维度参数为数组引用
  return at::var(self, IntArrayRef{dim});
}

// 定义 TORCH_API 修饰符下的 inline 函数，计算张量在指定维度上的方差和均值
TORCH_API inline std::tuple<Tensor, Tensor> var_mean(const Tensor& self, int dim) {
  // 调用 var_mean 函数，传入 IntArrayRef 类型的参数，转换维度参数为数组引用
  return at::var_mean(self, IntArrayRef{dim});
}

// 定义 TORCH_API 修饰符下的 inline 函数，计算张量在指定维度上的标准差
TORCH_API inline Tensor std(const Tensor& self, int dim) {
  // 调用 std 函数，传入 IntArrayRef 类型的参数，转换维度参数为数组引用
  return at::std(self, IntArrayRef{dim});
}

// 定义 TORCH_API 修饰符下的 inline 函数，计算张量在指定维度上的标准差和均值
TORCH_API inline std::tuple<Tensor, Tensor> std_mean(const Tensor& self, int dim) {
  // 调用 std_mean 函数，传入 IntArrayRef 类型的参数，转换维度参数为数组引用
  return at::std_mean(self, IntArrayRef{dim});
}

// 定义 inline 函数，返回张量的元素总数
inline int64_t numel(const Tensor& tensor) {
  return tensor.numel();
}

// 定义 inline 函数，返回张量在指定维度上的大小
inline int64_t size(const Tensor& tensor, int64_t dim) {
  return tensor.size(dim);
}

// 定义 inline 函数，返回张量在指定维度上的步幅
inline int64_t stride(const Tensor& tensor, int64_t dim) {
  return tensor.stride(dim);
}

// 定义 inline 函数，返回张量是否为复数类型
inline bool is_complex(const Tensor& tensor) {
  return tensor.is_complex();
}

// 定义 inline 函数，返回张量是否为浮点数类型
inline bool is_floating_point(const Tensor& tensor) {
  return tensor.is_floating_point();
}

// 定义 inline 函数，返回张量是否为有符号数类型
inline bool is_signed(const Tensor& tensor) {
  return tensor.is_signed();
}

// 定义 inline 函数，返回张量是否为推断类型
inline bool is_inference(const Tensor& tensor) {
  return tensor.is_inference();
}

// 定义 inline 函数，返回张量是否为零张量
inline bool _is_zerotensor(const Tensor& tensor) {
  return tensor._is_zerotensor();
}

// 定义 inline 函数，返回张量是否为共轭复数类型
inline bool is_conj(const Tensor& tensor) {
  return tensor.is_conj();
}

// 定义 inline 函数，返回张量的共轭张量
inline Tensor conj(const Tensor& tensor) {
  return tensor.conj();
}

// 定义 inline 函数，返回张量是否为负数
inline bool is_neg(const Tensor& tensor) {
  return tensor.is_neg();
}

}
```