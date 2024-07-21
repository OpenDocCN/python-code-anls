# `.\pytorch\aten\src\ATen\native\NonEmptyUtils.h`

```
#include <ATen/core/TensorBase.h>
#include <algorithm>
#include <vector>

namespace at::native {

// 确保维度不为空的辅助函数，返回大于等于1的维度值
inline int64_t ensure_nonempty_dim(int64_t dim) {
  return std::max<int64_t>(dim, 1);
}

// 确保给定维度不为空的辅助函数，如果张量为空则返回1，否则返回指定维度的大小
inline int64_t ensure_nonempty_size(const TensorBase &t, int64_t dim) {
  return t.dim() == 0 ? 1 : t.size(dim);
}

// 确保给定维度不为空的辅助函数，如果张量为空则返回1，否则返回指定维度的步长
inline int64_t ensure_nonempty_stride(const TensorBase &t, int64_t dim) {
  return t.dim() == 0 ? 1 : t.stride(dim);
}

// 确保向量不为空的辅助函数，如果向量为空则添加一个元素1并返回
using IdxVec = std::vector<int64_t>;
inline IdxVec ensure_nonempty_vec(IdxVec vec) {
  if (vec.empty()) {
    vec.push_back(1);
  }
  return vec;
}

}  // namespace at::native


这段代码是C++中关于张量操作的辅助函数，用于确保维度、大小和步长不为空。
```