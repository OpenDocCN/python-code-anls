# `.\pytorch\torch\csrc\jit\tensorexpr\cpp_intrinsics.h`

```
#pragma once
// 定义命名空间 torch::jit::tensorexpr，用于组织代码
namespace torch {
namespace jit {
namespace tensorexpr {

// 定义 constexpr 字符串 cpp_intrinsics_definition，包含一些 C++ 内置函数的定义
constexpr auto cpp_intrinsics_definition = R"(
namespace std {

// 定义模板函数 rsqrt，接受一个浮点数类型参数 T，返回其平方根的倒数
template <typename T,
          typename std::enable_if<std::is_floating_point<T>::value, int>::type = 0>
T rsqrt(T v) {
  return 1.0f / std::sqrt(v);
}

// 定义模板函数 frac，接受一个浮点数类型参数 T，返回其小数部分
template <typename T,
          typename std::enable_if<std::is_floating_point<T>::value, int>::type = 0>
T frac(T v) {
  T intpart;
  return std::modf(v, &intpart);
}

// 定义模板函数 bitcast，实现类型间的位转换，参数 From 是源类型，To 是目标类型
template <typename From, typename To>
To bitcast(const From& v) {
  // 断言源类型和目标类型大小相同
  assert(sizeof(To) == sizeof(From));
  To res;
  // 使用 std::memcpy 进行内存拷贝，实现位转换
  std::memcpy(&res, &v, sizeof(From));
  return res;
}

} // namespace std
)";
// 命名空间结束

} // namespace tensorexpr
} // namespace jit
} // namespace torch
// 命名空间 torch::jit::tensorexpr::torch 结束
```