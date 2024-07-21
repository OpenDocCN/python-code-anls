# `.\pytorch\aten\src\ATen\native\AutogradComposite.cpp`

```py
// 定义 TORCH_ASSERT_ONLY_METHOD_OPERATORS 符号
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

// 包含头文件：张量、小缓冲区和 COW（复制时写入）实现
#include <ATen/core/Tensor.h>
#include <c10/util/SmallBuffer.h>
#include <c10/core/impl/COW.h>

// 根据不同条件选择是否包含特定头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_has_same_storage_numel_native.h>
#include <ATen/ops/_make_dual_native.h>
#include <ATen/ops/_new_zeros_with_same_feature_meta_native.h>
#include <ATen/ops/_unpack_dual_native.h>
#include <ATen/ops/_lazy_clone_native.h>
#include <ATen/ops/alias.h>
#include <ATen/ops/zeros.h>
#endif

// 命名空间定义：at::native
namespace at::native {

// 函数：_make_dual
// 参数：primal 主张量，tangent 切线张量，level 等级
// 返回：一个主张量的别名（引用）
Tensor _make_dual(const Tensor& primal, const Tensor& tangent, int64_t level) {
  // 内部断言：仅在推断模式下且所有输入均为推断张量时生效
  TORCH_INTERNAL_ASSERT(
      InferenceMode::is_enabled() && primal.is_inference() && tangent.is_inference(),
      "Expected this function to only be reached in inference mode and when all the "
      "inputs are inference tensors. You should NOT call this function directly as "
      "native::_make_dual. Please use the dispatcher, i.e., at::_make_dual. Please "
      "file an issue if you come across this error otherwise.");
  // 返回主张量的别名
  return at::alias(primal);
}

// 函数：_unpack_dual
// 参数：tensor 双重张量，level 等级
// 返回：主张量和切线张量的元组
// 说明：此函数用于解包给定的双重张量，返回主张量的视图和切线张量本身
std::tuple<at::Tensor, at::Tensor> _unpack_dual(const at::Tensor& tensor, int64_t level) {
  return std::tuple<at::Tensor, at::Tensor>(tensor._fw_primal(level), tensor._fw_grad(level));
}

// 函数：_new_zeros_with_same_feature_meta
// 参数：self 自身张量，other 其它张量，self_num_batch_dims 自身批次维度数
// 返回：一个新张量，与 other 具有相同特征元数据
Tensor _new_zeros_with_same_feature_meta(
    const at::Tensor& self,
    const at::Tensor& other,
    int64_t self_num_batch_dims) {
  // 获取 other 张量的符号大小、符号步长、存储偏移和元素数量
  auto other_sizes = other.sym_sizes();
  auto other_strides = other.sym_strides();
  auto other_storage_offset = other.storage_offset();
  auto other_storage_numel = other.storage().sym_nbytes() / other.itemsize();

  if (self_num_batch_dims == 0) {
    // 创建一个新的符号整数张量，其大小为 other 的存储元素数量，使用 other 的选项
    auto new_tensor = at::zeros_symint({other_storage_numel}, other.options());
  // 返回一个新的张量，该张量采用符号整数的步幅，并使用其他尺寸和步幅作为参数
  }

  // 获取自身的符号整数尺寸
  auto self_sizes = self.sym_sizes();

  // 注意：我们不检查自身的尺寸是否与其他张量的尺寸相同，
  // 因为这个函数也用于原地视图的情况。
  // 在原地视图的情况下，我们不能依赖于自身和其他张量具有相同的大小。
  // 因此，我们将使用其他张量的大小，并简单地添加自身的批处理维度。
  // 例如：如果 self.sizes 为 [B, 2, 3]，而 other.size 为 [6]，我们将返回 [B, 6]。
  // 另请参阅测试 test_inplace_on_view_not_same_layout，以了解我们何时达到这种情况。

  constexpr int64_t kSmallBufferSizeHint = 8;

  // 创建一个具有适当容量的 SmallVector，用于存储输出的尺寸
  auto out_sizes = c10::SmallVector<c10::SymInt, kSmallBufferSizeHint>(other.dim() + self_num_batch_dims);
  // 复制自身的尺寸的批处理维度到输出尺寸中
  std::copy(self_sizes.begin(), self_sizes.begin() + self_num_batch_dims, out_sizes.begin());
  // 复制其他张量的尺寸到输出尺寸中，从批处理维度后开始复制
  std::copy(other_sizes.begin(), other_sizes.end(), out_sizes.begin() + self_num_batch_dims);

  // 使用其他张量的步幅，并根据自身的批处理维度计算步幅，以保证切片的连续性
  auto out_strides = c10::SmallVector<c10::SymInt, kSmallBufferSizeHint>(other.dim() + self_num_batch_dims);
  auto prod = other_storage_numel;

  // 从最后一个批处理维度开始向前计算步幅
  for (int64_t i = self_num_batch_dims - 1; i >= 0; --i) {
    out_strides[i] = prod;
    prod *= self_sizes[i];
  }
  // 复制其他张量的步幅到输出步幅中，从批处理维度后开始复制
  std::copy(other_strides.begin(), other_strides.end(), out_strides.begin() + self_num_batch_dims);

  // 计算存储的元素数量
  auto storage_numel = prod;

  // 继承原始张量的 TensorOptions
  auto new_tensor = at::zeros_symint({storage_numel}, other.options());
  // 返回作为符号整数步幅张量的新张量
  return new_tensor.as_strided_symint(out_sizes, out_strides, other_storage_offset);
}

# 定义一个私有函数，用于检查两个张量的存储空间是否具有相同的元素数量
bool _has_same_storage_numel(const at::Tensor& base, const at::Tensor& other) {
    # 计算基础张量和另一个张量的存储空间的总字节数，并除以元素大小，判断是否相等
    return base.storage().sym_nbytes() / base.itemsize() == other.storage().sym_nbytes() / other.itemsize();
}

# 实现一个惰性克隆函数，用于克隆给定张量
Tensor _lazy_clone(Tensor const& self) {
    # 获取当前张量的存储实现
    c10::StorageImpl* self_storage = self.storage().unsafeGetStorageImpl();
    # 使用惰性克隆技术克隆存储实现
    c10::intrusive_ptr<c10::StorageImpl> storage =
        c10::impl::cow::lazy_clone_storage(*self_storage);
    # 使用 TORCH_CHECK 确保存储实现非空
    TORCH_CHECK(storage != nullptr);
    # 创建新的张量实现对象，基于克隆后的存储实现、张量键集合和数据类型
    auto tensor = c10::make_intrusive<c10::TensorImpl>(
        c10::Storage(std::move(storage)),
        self.key_set(),
        self.dtype());
    # 设置新张量的大小、步长和存储偏移量
    tensor->set_sizes_and_strides(self.sym_sizes(),
                                  self.sym_strides(),
                                  self.sym_storage_offset());
    # 返回新创建的张量
    return Tensor(std::move(tensor));
}

} // namespace at::native
```