# `.\pytorch\caffe2\utils\string_utils.h`

```py
#pragma once
// 预处理指令，确保本文件只被编译一次

#include <algorithm>
// 包含算法库，用于提供各种算法函数

#include <memory>
// 包含内存管理相关的头文件，提供智能指针等功能

#include <string>
// 包含字符串处理相关的头文件

#include <vector>
// 包含向量容器相关的头文件，提供动态数组功能

#include <c10/macros/Export.h>
// 包含导出宏的头文件，用于在不同平台上定义导出函数

namespace caffe2 {
// 定义命名空间 caffe2

TORCH_API std::vector<std::string>
split(char separator, const std::string& string, bool ignore_empty = false);
// 函数声明：按指定分隔符分割字符串并返回结果向量

TORCH_API std::string trim(const std::string& str);
// 函数声明：去除字符串两端空白字符并返回结果

TORCH_API size_t editDistance(
    const std::string& s1,
    const std::string& s2,
    size_t max_distance = 0);
// 函数声明：计算两个字符串之间的编辑距离，并返回结果

TORCH_API inline bool StartsWith(
    const std::string& str,
    const std::string& prefix) {
  // 内联函数定义：检查字符串是否以指定前缀开头，返回布尔值
  return str.length() >= prefix.length() &&
      std::mismatch(prefix.begin(), prefix.end(), str.begin()).first ==
      prefix.end();
}

TORCH_API inline bool EndsWith(
    const std::string& full,
    const std::string& ending) {
  // 内联函数定义：检查字符串是否以指定后缀结尾，返回布尔值
  if (full.length() >= ending.length()) {
    return (
        0 ==
        full.compare(full.length() - ending.length(), ending.length(), ending));
  } else {
    return false;
  }
}

TORCH_API int32_t editDistanceHelper(
    const char* s1,
    size_t s1_len,
    const char* s2,
    size_t s2_len,
    std::vector<size_t>& current,
    std::vector<size_t>& previous,
    std::vector<size_t>& previous1,
    size_t max_distance);
// 函数声明：辅助函数，计算两个 C 字符串之间的编辑距离，并返回结果

} // namespace caffe2
// 命名空间 caffe2 结束
```