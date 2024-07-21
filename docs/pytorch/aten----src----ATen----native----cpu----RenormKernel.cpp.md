# `.\pytorch\aten\src\ATen\native\cpu\RenormKernel.cpp`

```
#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/Normalization.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>

#include <ATen/cpu/vec/vec.h>

#include <ATen/Dispatch.h>

namespace at::native {
namespace {

// 实现标准化尺度因子的函数，应用于给定的迭代器
void renorm_scale_factor_impl(TensorIteratorBase& iter, double maxnorm) {
  // 根据迭代器的通用数据类型选择合适的浮点类型进行分派
  AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "renorm_scale_factor_cpu", [&] {
    using vec_t = at::vec::Vectorized<scalar_t>;  // 使用矢量化类型进行计算
    const auto maxnorm_s = static_cast<scalar_t>(maxnorm);  // 将最大范数转换为对应类型的标量
    const auto maxnorm_v = vec_t(maxnorm_s);  // 将最大范数转换为矢量化类型
    const auto eps_v = vec_t(static_cast<scalar_t>(1e-7));  // 设置一个很小的值作为epsilon
    const auto one_v = vec_t(1.0);  // 创建一个值为1的矢量化常量
    // 调用CPU内核函数，对每个元素进行标准化尺度因子的计算
    cpu_kernel_vec(
      iter,
      [maxnorm_s](scalar_t norm) -> scalar_t {
        const auto eps = static_cast<scalar_t>(1e-7);  // 设置一个很小的值作为epsilon
        return (norm > maxnorm_s) ?
            maxnorm_s / (norm + eps) : static_cast<scalar_t>(1.0);  // 根据条件计算标准化因子
      },
      [maxnorm_v, eps_v, one_v](vec_t norm) -> vec_t {
        auto fct = maxnorm_v / (norm + eps_v);  // 计算矢量化的标准化因子
        return vec_t::blendv(one_v, fct, norm > maxnorm_v);  // 根据条件混合选择标准化因子
      });
  });
}

}  // namespace (anonymous)

// 注册标准化尺度因子的分派函数
REGISTER_DISPATCH(renorm_scale_factor_stub, &renorm_scale_factor_impl);

}  // namespace at::native
```