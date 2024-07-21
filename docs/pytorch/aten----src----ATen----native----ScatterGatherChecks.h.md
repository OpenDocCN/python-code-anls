# `.\pytorch\aten\src\ATen\native\ScatterGatherChecks.h`

```
#pragma once

#include <vector>
#include <ATen/core/Tensor.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <c10/util/irange.h>

namespace at::native {

namespace {

// 检查 index 的数据类型是否为 int64
// 如果 src 是 Tensor，则检查 self 和 src 的数据类型是否相同
static void scatter_gather_dtype_check(
  const std::string& method_name,
  const Tensor& self,
  const Tensor& index,
  const std::optional<Tensor>& src_opt = c10::nullopt
) {
  if (index.numel() != 0) {
    TORCH_CHECK(
      index.scalar_type() == at::ScalarType::Long,
      method_name, "(): Expected dtype int64 for index"
    );
  }

  if (src_opt.has_value()) {
    const auto& src = src_opt.value();
    TORCH_CHECK(
      self.scalar_type() == src.scalar_type(),
      method_name, "(): Expected self.dtype to be equal to src.dtype"
    );
  }
}

// 用于类似于 `gather` 方法的形状检查
// 注意：这里的 self 表示输入张量
// 测试：
// 1. 对于所有 d != dim，index.size(d) <= self.size(d)
// 2. index.dim() == self.dim()
static C10_UNUSED void gather_shape_check(const Tensor& self, int64_t dim,
  const Tensor& index
) {
  auto self_dims = ensure_nonempty_dim(self.dim());
  TORCH_CHECK(self_dims == ensure_nonempty_dim(index.dim()),
    "Index tensor must have the same number of dimensions as input tensor"
  );

  for (const auto i : c10::irange(self_dims)) {
    if (i != dim) {
      TORCH_CHECK(
        ensure_nonempty_size(index, i) <= ensure_nonempty_size(self, i),
        "Size does not match at dimension ", i,
        " expected index ", index.sizes(),
        " to be smaller than self ", self.sizes(),
        " apart from dimension ", dim
      );
    }
  }
}

// 用于 `scatter` 和 `scatter_add` 方法的形状检查
// 测试：
//  1. 对于所有 d != dim，index.size(d) <= self.size(d)
//  2. 如果 src 是 Tensor，对于所有 d，index.size(d) <= src.size(d)
//  3. index.dim() == self.dim() == src.dim()
static C10_UNUSED void scatter_shape_check(
  const Tensor& self, int64_t dim, const Tensor& index,
  const std::optional<Tensor>& src_opt = c10::nullopt
) {
  if (index.numel() == 0) return;
  TORCH_CHECK(
    ensure_nonempty_dim(self.dim()) == ensure_nonempty_dim(index.dim()),
    "Index tensor must have the same number of dimensions as self tensor"
  );

  bool is_wrong_shape = false;
  int64_t self_dims = ensure_nonempty_dim(self.dim());

  // 检查：对于所有 d != dim，index.size(d) <= self.size(d)
  for (const auto d : c10::irange(self_dims)) {
    int64_t index_d_size = ensure_nonempty_size(index, d);
    if (d == dim) continue;
    if (index_d_size > ensure_nonempty_size(self, d)) {
      is_wrong_shape = true;
      break;
    }
  }

  // 检查：如果 src 是 Tensor，对于所有 d，index.size(d) <= src.size(d)
  if (!is_wrong_shape && src_opt.has_value()) {
    const auto& src = src_opt.value();
    for (const auto d : c10::irange(self_dims)) {
      int64_t index_d_size = ensure_nonempty_size(index, d);
      if (index_d_size > ensure_nonempty_size(src, d)) {
        is_wrong_shape = true;
        break;
      }
    }
  }
  }
}

if (src_opt.has_value()) {
  // 获取包含在src_opt中的源张量src的常量引用
  const auto& src = src_opt.value();

  // 使用TORCH_CHECK检查条件，确保索引张量index的非空维数与src张量的维数相同
  TORCH_CHECK(
    ensure_nonempty_dim(src.dim()) == ensure_nonempty_dim(index.dim()),
    "Index tensor must have the same number of dimensions as src tensor"
  );

  // 使用TORCH_CHECK检查条件，确保索引index的形状正确，即除了维度dim外，还需小于self张量的大小，且小于src张量的大小
  TORCH_CHECK(!is_wrong_shape,
    "Expected index ", index.sizes(),
    " to be smaller than self ", self.sizes(),
    " apart from dimension ", dim,
    " and to be smaller size than src ", src.sizes()
  );
}
else {
  // 使用TORCH_CHECK检查条件，确保索引index的形状正确，即除了维度dim外，还需小于self张量的大小
  TORCH_CHECK(!is_wrong_shape,
    "Expected index ", index.sizes(),
    " to be smaller than self ", self.sizes(),
    " apart from dimension ", dim
  );
}
}

} // anonymous namespace

} // namespace at::native
```