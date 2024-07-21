# `.\pytorch\aten\src\ATen\native\TensorTransformations.h`

```py
// 引入 ATen 核心张量头文件
#include <ATen/core/Tensor.h>

// 根据条件引入 ATen 函数或者 ATen 操作库中的 roll 函数
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/roll.h>
#endif

// 引入 C10 异常处理工具
#include <c10/util/Exception.h>

// 进入 ATen 的 native 命名空间
namespace at::native {

// 定义静态内联函数 roll_common，接收张量 self、移动量 shifts 和维度 dims 作为参数
static inline Tensor roll_common(const Tensor& self, IntArrayRef shifts, IntArrayRef dims) {
  // 检查 shifts 参数非空
  TORCH_CHECK(!shifts.empty(), "`shifts` required");

  // 如果 dims 为空且 shifts 的长度为1，对张量进行展平操作并进行滚动，最后恢复形状
  if (dims.empty() && shifts.size() == 1) {
    auto flattened = self.contiguous().view(self.numel());
    return roll(flattened, shifts[0], 0).view(self.sizes());
  }

  // 检查 shifts 和 dims 的长度一致
  TORCH_CHECK(
    shifts.size() == dims.size(),
    "shifts and dimensions must align. shifts: ", shifts.size(), ", dims:", dims.size()
  );

  // 断言 dims 的长度大于1
  AT_ASSERT(dims.size() > 1);

  // 获取除第一个维度外的剩余移动量和维度
  auto tail_shifts = shifts.slice(1);
  auto tail_dims = dims.slice(1);

  // 对第一个维度进行滚动操作
  auto first_dim_rolled = roll(self, shifts[0], dims[0]);

  // 继续对剩余维度进行滚动操作
  return at::roll(first_dim_rolled, tail_shifts, tail_dims);
}

}  // namespace at::native


这段代码是一个 C++ 的函数定义，位于 ATen 的 native 命名空间中。函数 `roll_common` 实现了张量的滚动操作，根据给定的移动量 `shifts` 和维度 `dims` 对张量 `self` 进行滚动操作。
```