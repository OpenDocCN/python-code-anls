# `.\pytorch\aten\src\ATen\native\IndexingUtils.cpp`

```
// 定义预处理器宏，用于仅支持方法操作符的 Torch 断言
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 引入 ATen 库中的索引工具头文件
#include <ATen/native/IndexingUtils.h>

// 进入 ATen 命名空间的 native 子命名空间
namespace at::native {

// 判断是否可以使用32位索引运算，参数包括张量对象和最大元素数
bool canUse32BitIndexMath(const TensorBase& t, int64_t max_elem) {
  // 获取张量的符号化元素数
  auto elements = t.sym_numel();
  // 如果元素数大于等于最大元素数，返回 false
  if (elements >= max_elem) {
    return false;
  }
  // 如果元素数为零，只有当最大元素数大于零时才返回 true
  if (elements == 0) {
    return max_elem > 0;
  }

  // 初始化符号整数 offset 为 0
  c10::SymInt offset = 0;
  // 计算线性索引值，从元素数减一开始
  auto linearId = elements - 1;

  // NOTE: 假设所有步长都是正数，目前为真
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  // 逆序遍历张量的维度
  for (int i = t.dim() - 1; i >= 0; --i) {
    // 计算当前维度的索引
    auto curDimIndex = linearId % t.sym_size(i);
    // 计算当前维度的偏移量
    auto curDimOffset = curDimIndex * t.sym_stride(i);
    // 累加偏移量到 offset
    offset += curDimOffset;
    // 更新线性索引值
    linearId /= t.sym_size(i);
  }

  // 如果计算得到的 offset 大于等于最大元素数，返回 false
  if (offset >= max_elem) {
    return false;
  }

  // 否则返回 true
  return true;
}

} // namespace at::native
```