# `.\pytorch\aten\src\ATen\native\cpu\IsContiguous.h`

```py
#pragma once

namespace at { namespace native { inline namespace CPU_CAPABILITY {

// 定义模板结构体 IsContiguous，用于检查张量操作的连续性
// n: 函数参数数量（元数）
// stride_index: 步幅数组中用于检查的索引
// traits: 函数特性（详见 FunctionTraits.h）
// s: 标量参数的索引或者为 -1
template <int n, int stride_index, typename traits, int s=-1>
struct IsContiguous {
  // 静态函数 eval 用于评估张量操作是否连续
  static bool eval(const int64_t* strides) {
    // 使用 type 定义为 traits 的第 n-1 个参数的类型
    using type = typename traits::template arg<n - 1>::type;
    // 检查当前步幅是否与类型大小或零相匹配，并递归检查更低索引的连续性
    return strides[stride_index] == (s == n ? 0 : sizeof(type)) &&
           IsContiguous<n - 1, stride_index - 1, traits, s>::eval(strides);
  }
};

// 当存在输出时调用
template <typename traits, int s>
struct IsContiguous<0, 0, traits, s> {
  static bool eval(const int64_t* strides) {
    // 检查首个步幅是否与结果类型大小相匹配
    return strides[0] == sizeof(typename traits::result_type);
  }
};

// 当不存在输出时调用
template <typename traits, int s>
struct IsContiguous<0, -1, traits, s> {
  static bool eval(const int64_t* /*strides*/) {
    // 如果不存在输出，直接返回 true
    return true;
  }
};

// 当输出和所有输入都是连续的
template <typename traits,
    typename std::enable_if<std::is_void<typename traits::result_type>::value>::type* = nullptr>
static inline bool is_contiguous(const int64_t* strides) {
  // 调用 IsContiguous 结构体来评估连续性
  return IsContiguous<traits::arity, traits::arity - 1, traits>::eval(strides);
}

// 当输出和所有输入都是连续的（但输出不为空）
template <typename traits,
    typename std::enable_if<!std::is_void<typename traits::result_type>::value>::type* = nullptr>
static inline bool is_contiguous(const int64_t* strides) {
  // 调用 IsContiguous 结构体来评估连续性
  return IsContiguous<traits::arity, traits::arity, traits>::eval(strides);
}

// 输入参数 s 是标量（步幅为 0），输出和其他输入是连续的
// 注意：输出通常在 strides[0]，因此第一个输入对应 s=1
template <typename traits, int s,
    typename std::enable_if<std::is_void<typename traits::result_type>::value>::type* = nullptr>
static inline bool is_contiguous_scalar(const int64_t* strides) {
  // 静态断言，确保 s 大于 0 且小于等于参数数目
  static_assert(s > 0 && s <= traits::arity, "scalar argument index out of bounds");
  // 调用 IsContiguous 结构体来评估连续性
  return IsContiguous<traits::arity, traits::arity - 1, traits, s>::eval(strides);
}

// 输入参数 s 是标量（步幅为 0），输出和其他输入是连续的（但输出不为空）
template <typename traits, int s,
    typename std::enable_if<!std::is_void<typename traits::result_type>::value>::type* = nullptr>
static inline bool is_contiguous_scalar(const int64_t* strides) {
  // 静态断言，确保 s 大于 0 且小于等于参数数目
  static_assert(s > 0 && s <= traits::arity, "scalar argument index out of bounds");
  // 调用 IsContiguous 结构体来评估连续性
  return IsContiguous<traits::arity, traits::arity, traits, s>::eval(strides);
}

}}}
```