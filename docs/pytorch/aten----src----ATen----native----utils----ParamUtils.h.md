# `.\pytorch\aten\src\ATen\native\utils\ParamUtils.h`

```
#pragma once
// 使用预处理指令#pragma once，确保头文件只被包含一次

#include <c10/util/ArrayRef.h>
#include <vector>

namespace at {
namespace native {

// 模板函数 _expand_param_if_needed，用于处理参数的扩展
template <typename T>
inline std::vector<T> _expand_param_if_needed(
    ArrayRef<T> list_param,     // 输入参数：ArrayRef 类型的参数列表
    const char* param_name,     // 输入参数：参数名称的 C 字符串指针
    int64_t expected_dim) {     // 输入参数：期望的维度大小

  // 如果 list_param 只有一个元素，则返回一个大小为 expected_dim，填充为 list_param[0] 的 vector
  if (list_param.size() == 1) {
    return std::vector<T>(expected_dim, list_param[0]);
  } 
  // 如果 list_param 的大小不为 1，并且不等于 expected_dim，则抛出错误
  else if ((int64_t)list_param.size() != expected_dim) {
    std::ostringstream ss;
    ss << "expected " << param_name << " to be a single integer value or a "
       << "list of " << expected_dim << " values to match the convolution "
       << "dimensions, but got " << param_name << "=" << list_param;
    AT_ERROR(ss.str());
  } 
  // 否则，返回 list_param 的 vector 表示
  else {
    return list_param.vec();
  }
}

// 函数 expand_param_if_needed，处理 IntArrayRef 类型的参数
inline std::vector<int64_t> expand_param_if_needed(
    IntArrayRef list_param,     // 输入参数：IntArrayRef 类型的参数列表
    const char* param_name,     // 输入参数：参数名称的 C 字符串指针
    int64_t expected_dim) {     // 输入参数：期望的维度大小
  return _expand_param_if_needed(list_param, param_name, expected_dim);
}

// 函数 expand_param_if_needed，处理 SymIntArrayRef 类型的参数
inline std::vector<c10::SymInt> expand_param_if_needed(
    SymIntArrayRef list_param,  // 输入参数：SymIntArrayRef 类型的参数列表
    const char* param_name,     // 输入参数：参数名称的 C 字符串指针
    int64_t expected_dim) {     // 输入参数：期望的维度大小
  return _expand_param_if_needed(list_param, param_name, expected_dim);
}

} // namespace native
} // namespace at
```