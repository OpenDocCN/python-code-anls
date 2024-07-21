# `.\pytorch\aten\src\ATen\native\SummaryOps.cpp`

```py
// 定义宏，仅限于操作符方法的断言
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

// 引入必要的头文件
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <c10/util/irange.h>

// 如果未定义每个操作符的头文件，则引入以下头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
// 如果定义了每个操作符的头文件，则引入以下头文件
#else
#include <ATen/ops/bincount_native.h>
#include <ATen/ops/zeros.h>
#endif

// at::native 命名空间下的函数定义开始
namespace at::native {

///////////////// bincount /////////////////

// 匿名命名空间下的模板函数定义
namespace {

template <typename input_t, typename weights_t>
Tensor _bincount_cpu_template(
    const Tensor& self,
    const Tensor& weights,
    int64_t minlength) {
  // 检查 minlength 是否为负数，若是则报错
  if (minlength < 0) {
    AT_ERROR("minlength should be >= 0");
  }
  // 如果输入张量 self 是一维且元素数为 0，则返回长度为 minlength 的零张量
  if (self.dim() == 1 && self.numel() == 0) {
    return at::zeros({minlength}, kLong);
  }
  // 如果输入张量 self 不是一维或者包含负数元素，则报错
  if (self.dim() != 1 || *self.min().data_ptr<input_t>() < 0) {
    AT_ERROR("bincount only supports 1-d non-negative integral inputs.");
  }

  // 检查是否定义了权重张量 weights，且其维度为一维且长度与输入 self 相同
  bool has_weights = weights.defined();
  if (has_weights && (weights.dim() != 1 || weights.size(0) != self.size(0))) {
    AT_ERROR("weights should be 1-d and have the same length as input");
  }

  // 定义输出张量 output
  Tensor output;
  int64_t self_size = self.size(0);
  // 计算 bins 的数量，即 self 中元素的最大值加 1
  int64_t nbins = static_cast<int64_t>(*self.max().data_ptr<input_t>()) + 1L;
  nbins = std::max(nbins, minlength); // 至少有 minlength 个 bins

  // 获取 self 的数据指针 self_p
  const input_t* self_p = self.const_data_ptr<input_t>();
  if (has_weights) {
    // 如果有权重 weights，则创建与 weights 类型相同的 output 张量
    output = at::zeros(
        {nbins},
        optTypeMetaToScalarType(weights.options().dtype_opt()),
        weights.options().layout_opt(),
        weights.options().device_opt(),
        weights.options().pinned_memory_opt());
    weights_t* output_p = output.data_ptr<weights_t>();
    const weights_t* weights_p = weights.const_data_ptr<weights_t>();
    // 遍历 self，根据 self 中的元素值更新 output 中对应位置的权重值
    for (const auto i : c10::irange(self_size)) {
      output_p[self_p[i]] += weights_p[i];
    }
  } else {
    // 如果没有权重 weights，则创建类型为 kLong 的 output 张量
    output = at::zeros({nbins}, kLong);
    int64_t* output_p = output.data_ptr<int64_t>();
    // 遍历 self，统计每个元素值在 self 中出现的次数
    for (const auto i : c10::irange(self_size)) {
      output_p[self_p[i]] += 1L;
    }
  }
  return output;
}
} // namespace

// bincount_cpu 函数定义，接收输入张量 self、可选的权重张量 weights_opt 和 minlength
Tensor
_bincount_cpu(const Tensor& self, const std::optional<Tensor>& weights_opt, int64_t minlength) {
  // 从可选的权重张量 weights_opt 中获取权重张量
  c10::MaybeOwned<Tensor> weights_maybe_owned = at::borrow_from_optional_tensor(weights_opt);
  const Tensor& weights = *weights_maybe_owned;

  // 使用 AT_DISPATCH_INTEGRAL_TYPES 宏分发到对应整数类型的具体实现
  return AT_DISPATCH_INTEGRAL_TYPES(self.scalar_type(), "bincount_cpu", [&] {
    const auto scalar = weights.scalar_type();
    // 根据 weights 的标量类型选择具体的模板函数并调用
    if (scalar == ScalarType::Undefined || scalar == ScalarType::Float)
      return _bincount_cpu_template<scalar_t, float>(self.contiguous(), weights.contiguous(), minlength);
    return _bincount_cpu_template<scalar_t, double>(
        self.contiguous(), weights.contiguous().to(kDouble), minlength);
  });
}

} // namespace at::native
```