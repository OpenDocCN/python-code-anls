# `.\pytorch\aten\src\ATen\native\Onehot.cpp`

```
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/arange.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/eq.h>
#include <ATen/ops/one_hot_native.h>
#include <ATen/ops/zeros.h>
#endif

namespace at::native {

// 实现将输入张量转换为 one-hot 编码的张量
Tensor one_hot(const Tensor &self, int64_t num_classes) {
    // 检查输入张量的数据类型是否为 kLong，否则抛出错误
    TORCH_CHECK(self.dtype() == kLong, "one_hot is only applicable to index tensor.");

    // 使用元位测试捕捉 Fake Tensor，同时支持 __torch_function__
    if (self.key_set().has_all(DispatchKeySet(BackendComponent::MetaBit)) ||
            self.key_set().has_all(DispatchKeySet(DispatchKey::Python))) {
        // 在 Torch 编译效果更好且支持动态形状的 functional 版本
        if (num_classes == -1) {
          num_classes = self.max().item().toLong() + 1;
        }
        // 生成 [0, num_classes) 范围的索引张量
        at::Tensor index = at::arange(num_classes, self.options());
        // 将输入张量与索引张量比较，生成 one-hot 编码的张量，类型为 kLong
        return at::eq(self.unsqueeze(-1), index).to(kLong);
    }

    auto shape = self.sizes().vec();

    // 空张量可以转换为 one-hot 表示，但无法进行形状推断
    if (self.numel() == 0) {
        if (num_classes <= 0) {
            // 如果无法从空张量推断出总类数，则抛出错误
            AT_ERROR("Can not infer total number of classes from empty tensor.");
        } else {
            // 根据指定的类数创建空张量
            shape.push_back(num_classes);
            return at::empty(shape, self.options());
        }
    }

    // 非空张量
    if (self.device().type() != at::kCUDA && self.device().type() != at::kMPS &&
        self.device().type() != at::kPrivateUse1 && self.device().type() != at::kXLA) {
      // 对于非 CUDA 设备，确保类值非负
      TORCH_CHECK(self.min().item().toLong() >= 0, "Class values must be non-negative.");
    }
    if (num_classes == -1) {
        num_classes = self.max().item().toLong() + 1;
    } else {
        if (self.device().type() != at::kCUDA && self.device().type() != at::kMPS &&
            self.device().type() != at::kPrivateUse1 && self.device().type() != at::kXLA) {
          // 对于非 CUDA 设备，确保类值小于 num_classes，避免同步
          TORCH_CHECK(num_classes > self.max().item().toLong(), "Class values must be smaller than num_classes.");
        } else {
            // 对于 CUDA 设备，确保 num_classes 至少为 1
            TORCH_CHECK(num_classes >= 1, "num_classes should be positive");
        }
    }

    // 将 num_classes 添加到形状中，创建全零张量
    shape.push_back(num_classes);
    Tensor ret = at::zeros(shape, self.options());
    // 在最后一个维度上，将输入张量的每个值转换为 one-hot 编码
    ret.scatter_(-1, self.unsqueeze(-1), 1);
    return ret;
}

} // namespace at::native
```