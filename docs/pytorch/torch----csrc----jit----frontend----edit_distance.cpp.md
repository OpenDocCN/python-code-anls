# `.\pytorch\torch\csrc\jit\frontend\edit_distance.cpp`

```py
// 包含头文件，用于计算两个单词之间的编辑距离
#include <torch/csrc/jit/frontend/edit_distance.h>
#include <algorithm>
#include <cstring>
#include <memory>

namespace torch::jit {

// 计算两个单词之间的莱文斯坦编辑距离
// 如果编辑距离超过最大编辑距离，则返回最大编辑距离 + 1
// 参考链接: http://llvm.org/doxygen/edit__distance_8h_source.html
size_t ComputeEditDistance(
    const char* word1,
    const char* word2,
    size_t maxEditDistance) {
  // 计算两个单词的长度
  size_t m = strlen(word1);
  size_t n = strlen(word2);

  const unsigned small_buffer_size = 64;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
  unsigned small_buffer[small_buffer_size];
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
  std::unique_ptr<unsigned[]> allocated;
  unsigned* row = small_buffer;
  // 如果需要更大的空间，则动态分配内存
  if (n + 1 > small_buffer_size) {
    row = new unsigned[n + 1];
    allocated.reset(row);
  }

  // 初始化第一行
  for (unsigned i = 1; i <= n; ++i)
    row[i] = i;

  // 计算编辑距离
  for (size_t y = 1; y <= m; ++y) {
    row[0] = y;
    unsigned best_this_row = row[0];

    unsigned previous = y - 1;
    for (size_t x = 1; x <= n; ++x) {
      const auto old_row = row[x];
      row[x] = std::min(
          previous + (word1[y - 1] == word2[x - 1] ? 0u : 1u),
          std::min(row[x - 1], row[x]) + 1);
      previous = old_row;
      best_this_row = std::min(best_this_row, row[x]);
    }

    // 如果编辑距离超过最大编辑距离，则返回最大编辑距离 + 1
    if (maxEditDistance && best_this_row > maxEditDistance)
      return maxEditDistance + 1;
  }

  // NOLINTNEXTLINE(clang-analyzer-core.uninitialized.Assign)
  unsigned result = row[n];
  return result;
}

} // namespace torch::jit
```