# `.\pytorch\aten\src\ATen\native\mps\operations\Equal.cpp`

```
// 定义 TORCH_ASSERT_ONLY_METHOD_OPERATORS 宏，可能用于限制操作符方法
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 包含 ATen 库中的 Tensor 类定义
#include <ATen/core/Tensor.h>
// 包含 ATen 库中的 NamedTensorUtils 头文件
#include <ATen/NamedTensorUtils.h>

// 如果未定义 AT_PER_OPERATOR_HEADERS 宏，则包含以下头文件
#ifndef AT_PER_OPERATOR_HEADERS
// 包含 ATen 库中的 NativeFunctions 头文件
#include <ATen/NativeFunctions.h>
// 包含 ATen 库中的 MPSFunctions 头文件
#include <ATen/MPSFunctions.h>
// 否则，包含 MPS 分发相关的头文件
#else
#include <ATen/ops/eq_mps_dispatch.h>
#include <ATen/ops/equal_native.h>
#endif

// 定义命名空间 at
namespace at {
  // 定义命名空间 mps
  namespace mps {
    // 声明 TORCH_API 修饰的 eq 函数，接受两个 Tensor 参数，返回 Tensor
    TORCH_API at::Tensor eq(const at::Tensor & self, const at::Tensor & other);
  } // namespace mps
  
  // 定义命名空间 native
  namespace native {
    
    // 实现 mps_equal 函数，接受两个 Tensor 参数，返回布尔值
    bool mps_equal(const Tensor& self, const Tensor &src) {
      // 检查 self 和 src 的命名是否相等
      if (!at::namedinference::are_names_equal(
              self.unsafeGetTensorImpl(), src.unsafeGetTensorImpl())) {
        return false;
      }
      // 使用 NoNamesGuard 禁用命名
      at::NoNamesGuard guard;
      // 检查 self 和 src 的设备是否相同
      TORCH_CHECK(self.device() == src.device(), "Cannot compare two tensors on "
                  "different devices. Got: ", self.device(), " and ", src.device());
      // 检查 self 和 src 的尺寸是否相同
      if (self.sizes() != src.sizes()) {
        return false;
      }
      // 如果 self 的元素数量为 0，直接返回 true
      if (self.numel() == 0) {
        return true;
      }
      // 调用 MPS 库中的 eq 函数比较 self 和 src 是否相等，并检查所有元素是否都为 true
      return at::mps::eq(self, src).all().item().to<bool>();
    }

  } // namespace native
} // namespace at
```