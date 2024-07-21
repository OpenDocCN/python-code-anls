# `.\pytorch\aten\src\ATen\native\mkldnn\MkldnnTensorMath.cpp`

```
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Config.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/zero_native.h>
#endif

#if !AT_MKLDNN_ENABLED()

namespace at {
namespace native {

// 如果未启用 MKLDNN 支持，则定义 mkldnn_zero_ 函数
Tensor& mkldnn_zero_(Tensor& self) {
  // 使用 TORCH_CHECK 断言确保不会调用该函数
  TORCH_CHECK(false, "mkldnn_zero_: ATen not compiled with MKLDNN support");
}

} // namespace native
} // namespace at

#else // AT_MKLDNN_ENABLED

#include <ATen/native/mkldnn/MKLDNNCommon.h>

namespace at {
namespace native {

// 如果启用了 MKLDNN 支持，则定义 mkldnn_zero_ 函数
Tensor& mkldnn_zero_(Tensor& self) {
  // 使用 Vec 类型别名来表示 float 类型的 SIMD 向量操作
  using Vec = vec::Vectorized<float>;

  // 从 MKLDNN 张量中获取 ideep::tensor 引用
  ideep::tensor& x = itensor_from_mkldnn(self);

  // 获取张量元素个数
  auto n = x.get_nelems();
  // 获取数据指针并转换为 float* 类型
  auto* x_ = static_cast<float*>(x.get_data_handle());
  // 使用并行化策略执行以下操作：
  // 将 lambda 函数应用于 x_ 指针范围内的数据，实现向量化操作
  parallel_for(0, n, 2048, [x_](int64_t begin, int64_t end) {
    vec::map(
        [](Vec /* unused */) { return 0.0; }, // 对每个向量元素执行将其设为 0 的操作
        x_ + begin,  // 起始位置
        x_ + begin,  // 结束位置
        end - begin);  // 元素个数
  });

  // 返回自身的引用
  return self;
}

} // namespace native
} // namespace at

#endif // AT_MKLDNN_ENABLED
```