# `.\pytorch\c10\core\Layout.h`

```py
#pragma once

#include <c10/core/Backend.h>
#include <c10/util/Exception.h>

#include <cstdint>
#include <ostream>

namespace c10 {

// 枚举类型 Layout 表示张量的存储布局
enum class Layout : int8_t {
  Strided,      // 连续布局
  Sparse,       // 稀疏布局
  SparseCsr,    // CSR 稀疏布局
  Mkldnn,       // MKLDNN 布局
  SparseCsc,    // CSC 稀疏布局
  SparseBsr,    // BSR 稀疏布局
  SparseBsc,    // BSC 稀疏布局
  Jagged,       // Jagged 布局
  NumOptions    // 布局选项数量
};

// 各种默认的布局常量
constexpr auto kStrided = Layout::Strided;
constexpr auto kSparse = Layout::Sparse;
constexpr auto kSparseCsr = Layout::SparseCsr;
constexpr auto kMkldnn = Layout::Mkldnn;
constexpr auto kSparseCsc = Layout::SparseCsc;
constexpr auto kSparseBsr = Layout::SparseBsr;
constexpr auto kSparseBsc = Layout::SparseBsc;
constexpr auto kJagged = Layout::Jagged;

// 根据后端类型返回对应的布局类型
inline Layout layout_from_backend(Backend backend) {
  switch (backend) {
    case Backend::SparseCPU:
    case Backend::SparseCUDA:
    case Backend::SparseHIP:
    case Backend::SparseVE:
    case Backend::SparseXPU:
    case Backend::SparsePrivateUse1:
      return Layout::Sparse; // 稀疏布局
    case Backend::MkldnnCPU:
      return Layout::Mkldnn; // MKLDNN 布局
    case Backend::SparseCsrCPU:
    case Backend::SparseCsrCUDA:
    case Backend::SparseCsrHIP:
    case Backend::SparseCsrVE:
    case Backend::SparseCsrXPU:
      // 对于 SparseCsr(CPU|CUDA|HIP|VE|XPU) 后端，抛出异常，因为无法映射到唯一布局
      TORCH_CHECK(
          false,
          "Cannot map Backend SparseCsr(CPU|CUDA|HIP|VE|XPU) to a unique layout.");
    default:
      return Layout::Strided; // 默认使用连续布局
  }
}

// 自定义输出流操作符，用于将 Layout 转换为字符串输出
inline std::ostream& operator<<(std::ostream& stream, at::Layout layout) {
  switch (layout) {
    case at::kStrided:
      return stream << "Strided";    // 输出 "Strided"
    case at::kSparse:
      return stream << "Sparse";     // 输出 "Sparse"
    case at::kSparseCsr:
      return stream << "SparseCsr";  // 输出 "SparseCsr"
    case at::kSparseCsc:
      return stream << "SparseCsc";  // 输出 "SparseCsc"
    case at::kSparseBsr:
      return stream << "SparseBsr";  // 输出 "SparseBsr"
    case at::kSparseBsc:
      return stream << "SparseBsc";  // 输出 "SparseBsc"
    case at::kMkldnn:
      return stream << "Mkldnn";     // 输出 "Mkldnn"
    case at::kJagged:
      return stream << "Jagged";     // 输出 "Jagged"
    default:
      TORCH_CHECK(false, "Unknown layout"); // 未知布局类型，抛出异常
  }
}

} // namespace c10
```