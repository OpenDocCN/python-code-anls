# `.\pytorch\aten\src\ATen\TensorGeometry.h`

```py
#pragma once
// 预处理指令，确保头文件只被包含一次

#include <ATen/core/TensorBase.h>
// 引入 ATen 核心库中的 TensorBase 头文件
#include <c10/core/WrapDimMinimal.h>
// 引入 c10 核心库中的 WrapDimMinimal 头文件

namespace at {
// 命名空间 at，定义了 ATen 库的命名空间

// 返回表示由 sizes 和 strides 表示的张量几何形状是否连续
// 虽然现在我们在张量中缓存 is_contiguous，但这仍然有用，因为它允许检查特定几何形状是否连续，而无需显式构造张量，例如当您想根据子几何形状是否连续选择内核策略时。
TORCH_API bool geometry_is_contiguous(IntArrayRef sizes, IntArrayRef strides);

struct TORCH_API TensorGeometry {
  // 定义了一个名为 TensorGeometry 的结构体

  TensorGeometry() = default;
  // 默认构造函数

  explicit TensorGeometry(c10::SymIntArrayRef sizes)
      : sizes_(sizes.vec()),
        strides_(sizes.size()),
        has_symbolic_sizes_strides_(
            !c10::asIntArrayRefSlowOpt(sizes).has_value()) {
    // 显式构造函数，根据 sizes 初始化 TensorGeometry
    // 将 sizes 转换为 std::vector<c10::SymInt>
    int64_t dim = static_cast<int64_t>(sizes.size());
    c10::SymInt expected_stride = 1;
    for (int64_t i = dim - 1; i >= 0; i--) {
      strides_[i] = expected_stride;
      expected_stride *= sizes_[i];
    }
    numel_ = expected_stride;
    // 计算张量的总元素数
  }

  explicit TensorGeometry(const TensorBase& t)
      : sizes_(t.sym_sizes().vec()),
        strides_(t.sym_strides().vec()),
        storage_offset_(t.sym_storage_offset()),
        numel_(t.sym_numel()),
        has_symbolic_sizes_strides_(
            t.unsafeGetTensorImpl()->has_symbolic_sizes_strides()) {
    // 显式构造函数，根据 TensorBase 对象 t 初始化 TensorGeometry
    // 初始化 sizes、strides、storage_offset、numel 和 has_symbolic_sizes_strides_
  }

  // 返回张量是否连续
  bool is_contiguous() const;

  // 返回张量的维度
  int64_t dim() const {
    return static_cast<int64_t>(sizes_.size());
  }

  // 返回指定维度的大小
  int64_t size(int64_t dim) const {
    TORCH_INTERNAL_ASSERT(!has_symbolic_sizes_strides_);
    dim = c10::maybe_wrap_dim(dim, this->dim());
    return sizes_.at(static_cast<size_t>(dim)).as_int_unchecked();
  }

  // 返回张量的大小数组
  c10::IntArrayRef sizes() const {
    TORCH_INTERNAL_ASSERT(!has_symbolic_sizes_strides_);
    return c10::asIntArrayRefUnchecked(sizes_);
  }

  // 返回指定维度的步长
  int64_t stride(int64_t dim) const {
    TORCH_INTERNAL_ASSERT(!has_symbolic_sizes_strides_);
    dim = c10::maybe_wrap_dim(dim, this->dim());
    return strides_.at(static_cast<size_t>(dim)).as_int_unchecked();
  }

  // 返回张量的步长数组
  c10::IntArrayRef strides() const {
    TORCH_INTERNAL_ASSERT(!has_symbolic_sizes_strides_);
    return c10::asIntArrayRefUnchecked(strides_);
  }

  // 返回张量的存储偏移量
  int64_t storage_offset() const {
    TORCH_INTERNAL_ASSERT(!has_symbolic_sizes_strides_);
    return storage_offset_.as_int_unchecked();
  }

  // 返回张量的元素总数
  int64_t numel() const {
    TORCH_INTERNAL_ASSERT(!has_symbolic_sizes_strides_);
    return numel_.as_int_unchecked();
  }

  // 返回指定维度的符号化大小
  c10::SymInt sym_size(int64_t dim) const {
    dim = c10::maybe_wrap_dim(dim, this->dim());
    return sizes_.at(static_cast<size_t>(dim));
  }

  // 返回符号化大小数组
  c10::SymIntArrayRef sym_sizes() const {
    return sizes_;
  }

  // 返回指定维度的符号化步长
  c10::SymInt sym_stride(int64_t dim) const {
    dim = c10::maybe_wrap_dim(dim, this->dim());
    return strides_.at(static_cast<size_t>(dim));
  }

  // 返回符号化步长数组
  c10::SymIntArrayRef sym_strides() const {
    return strides_;
  }

  // 返回符号化存储偏移量
  c10::SymInt sym_storage_offset() const {
    // 未完成的成员函数，没有实现体


**注意：**由于代码部分较长，这里只显示了前面的部分注释。
  // 返回存储偏移量
  return storage_offset_;
}
// 返回张量的元素数
c10::SymInt sym_numel() const {
  return numel_;
}

// 返回在维度 dim0 和 dim1 上转置后的张量几何信息
TensorGeometry transpose(int64_t dim0, int64_t dim1) {
  // 复制当前对象
  TensorGeometry r = *this; // copy
  // 检查 dim0 是否在有效范围内
  TORCH_CHECK(
      dim0 < dim(),
      "transpose: dim0=",
      dim0,
      " out of range (dim=",
      dim(),
      ")")
  // 检查 dim1 是否在有效范围内
  TORCH_CHECK(
      dim1 < dim(),
      "transpose: dim1=",
      dim1,
      " out of range (dim=",
      dim(),
      ")")
  // 交换维度 dim0 和 dim1 的大小和步长
  std::swap(r.sizes_[dim0], r.sizes_[dim1]);
  std::swap(r.strides_[dim0], r.strides_[dim1]);
  // 返回转置后的张量几何信息
  return r;
}

// 返回可变的张量大小向量的引用
std::vector<c10::SymInt>& mutable_sizes() {
  return sizes_;
}

// 返回可变的张量步长向量的引用
std::vector<c10::SymInt>& mutable_strides() {
  return strides_;
}

// 返回可变的存储偏移量的引用
c10::SymInt& mutable_storage_offset() {
  return storage_offset_;
}

// 重新计算张量的元素数及符号大小和步长标志
void recompute() {
  // 重新计算元素数
  c10::SymInt numel = 1;
  for (const auto& i : sizes_) {
    numel = numel * i;
  }
  // 将重新计算得到的元素数移动赋值给成员变量
  numel_ = std::move(numel);
  // 更新是否具有符号大小和步长的标志
  has_symbolic_sizes_strides_ =
      !c10::asIntArrayRefSlowOpt(sizes_).has_value();
}
};

} // namespace at
```