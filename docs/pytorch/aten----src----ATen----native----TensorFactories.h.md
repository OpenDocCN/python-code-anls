# `.\pytorch\aten\src\ATen\native\TensorFactories.h`

```
#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/EmptyTensor.h>
#include <ATen/TensorIterator.h>
#include <ATen/Dispatch.h>
#include <ATen/Dispatch_v2.h>
#include <ATen/native/DispatchStub.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/scalar_tensor.h>
#endif

namespace at::native {

// 定义函数，计算下三角矩阵的大小
// 参数 row: 行数, col: 列数, offset: 偏移量
inline int64_t get_tril_size(int64_t row, int64_t col, int64_t offset) {
  // 如果行数或列数为0，则下三角矩阵大小为0
  if (row == 0 || col == 0) {
    return 0;
  }

  // 计算第一行和最后一行的元素数量
  auto m_first_row = offset > 0 ?
    std::min<int64_t>(col, 1 + offset) : // 上界为 col
    row + offset > 0; // 结果为0或1
  auto m_last_row = std::max<int64_t>(0, std::min<int64_t>(col, row + offset));

  // 计算行数
  auto n_row_all = std::max<int64_t>(0, std::min<int64_t>(row, row + offset));
  auto n_row_trapezoid = (m_last_row - m_first_row + 1);

  // 计算上梯形部分的元素数量
  auto tril_size = (m_first_row + m_last_row) * n_row_trapezoid >> 1;

  // 如果存在底部矩形部分，则计算其元素数量
  auto diff_row = n_row_all - n_row_trapezoid;
  if (diff_row > 0) {
    tril_size += diff_row * col;
  }

  return tril_size;
}

// 检查函数参数的有效性
// 参数 row: 行数, col: 列数, layout_opt: 布局选项
inline void check_args(
    int64_t row, int64_t col, std::optional<Layout> layout_opt) {
  TORCH_CHECK(row >= 0, "row must be non-negative, got", row);
  TORCH_CHECK(col >= 0, "col must be non-negative, got", col);
  if (layout_opt.has_value()) {
    TORCH_CHECK(
      *layout_opt == at::kStrided,
      "only support layout=torch.strided, got",
      *layout_opt)
  }
}

// 使用 at 命名空间中的 check_size_nonnegative
using at::check_size_nonnegative;

// 检查生成的张量中的最大整数是否受支持
// 参数 n: 最大整数, tensor: 张量对象
inline void check_supported_max_int_with_precision(int64_t n, const Tensor& tensor) {
  // 确保 n-1 在指定的张量类型中是有效的
  TORCH_CHECK(at::scalar_tensor(n>0?n-1:n, tensor.options()).defined(),
              "n is too large for result tensor type: '", tensor.toString(), "'");

  // 确保浮点数表示的精度足够
  switch (tensor.scalar_type()) {
    # 对于半精度数据类型（Half），检查 n 是否不超过 2049
    case at::ScalarType::Half:
      TORCH_CHECK(n <= (int64_t(1) << 11) + 1, "n cannot be greater than 2049 for Half type.");
      break;
    # 对于单精度数据类型（Float），检查 n 是否不超过 2^24+1
    case at::ScalarType::Float:
      TORCH_CHECK(n <= (int64_t(1) << 24) + 1, "n cannot be greater than 2^24+1 for Float type.");
      break;
    # 对于双精度数据类型（Double），虽然不太可能发生，但仍然检查 n 是否不超过 2^53+1
    case at::ScalarType::Double:  // Unlikely to happen, but doesn't hurt to check
      TORCH_CHECK(n <= (int64_t(1) << 53) + 1, "n cannot be greater than 2^53+1 for Double type.");
      break;
    # 其他情况，默认不进行额外的检查
    default:
      break;
}

// `empty*` 函数在确定性算法启用时调用，用于根据张量的类型填充 NaN（浮点数或复数类型），
// 或者填充最大值（整数类型）。
inline Tensor& fill_empty_deterministic_(Tensor& tensor) {
  // 检查张量是否为浮点数或复数类型
  if (tensor.is_floating_point() || tensor.is_complex()) {
    // 使用 AT_DISPATCH_V2 宏根据张量的标量类型进行分发
    AT_DISPATCH_V2(
      tensor.scalar_type(), "fill_empty_deterministic_", AT_WRAP([&]() {
        // 填充张量为 NaN，具体数值由 std::numeric_limits<scalar_t>::quiet_NaN() 给出
        tensor.fill_(std::numeric_limits<scalar_t>::quiet_NaN());
    }), AT_EXPAND(AT_FLOATING_TYPES), AT_EXPAND(AT_COMPLEX_TYPES), AT_EXPAND(AT_FLOAT8_TYPES), kBFloat16, kHalf);
  } else {
    // 对于整数类型的张量
    AT_DISPATCH_V2(
      tensor.scalar_type(), "fill_empty_deterministic_", AT_WRAP([&]() {
        // 填充张量为最大值，具体数值由 std::numeric_limits<scalar_t>::max() 给出
        tensor.fill_(std::numeric_limits<scalar_t>::max());
    }), kBool, AT_EXPAND(AT_INTEGRAL_TYPES_V2));
  }
  // 返回填充后的张量
  return tensor;
}

// ZeroTensorAllocator 分配器忽略任何分配请求，并始终返回 nullptr
struct ZeroTensorAllocator final : public at::Allocator {
  ZeroTensorAllocator(at::Device device) : device_(device) {};
  ~ZeroTensorAllocator() override = default;
  // 删除函数，用于释放指针，确保指针为 nullptr
  static void deleter(void* const pointer) {
    TORCH_INTERNAL_ASSERT(!pointer);
  }
  // 分配函数，总是返回 nullptr
  DataPtr allocate(const size_t /*nbytes*/) override {
    return {nullptr, nullptr, &deleter, device_};
  }
  // 原始删除函数指针，指向 deleter
  DeleterFnPtr raw_deleter() const override {
    return deleter;
  }
  // 数据拷贝函数，空实现
  void copy_data(void* dest, const void* src, std::size_t count) const final {}
  // 分配器的设备信息
  at::Device device_;
};

// 用于声明 binary_fn 类型的函数指针，指向 TensorIterator 的函数
using binary_fn = void (*)(TensorIterator&);

// 声明 complex_stub 和 polar_stub 的分发函数指针声明
DECLARE_DISPATCH(binary_fn, complex_stub);
DECLARE_DISPATCH(binary_fn, polar_stub);

} // namespace at::native
```