# `.\pytorch\aten\src\ATen\core\functional.h`

```py
#pragma once

#include <vector>
#include <c10/util/ArrayRef.h>

namespace c10 {

// The passed in function must take T by value (T), or by
// const reference (const T&); taking T by non-const reference
// will result in an error like:
//
//    error: no type named 'type' in 'class std::invoke_result<foobar::__lambda, T>'
//
// No explicit template parameters are required.

// 定义模板函数 fmap，用于将函数 fn 应用到 inputs 中的每个元素，并返回结果向量
template<class F, class T>
inline auto fmap(const T& inputs, const F& fn) -> std::vector<decltype(fn(*inputs.begin()))> {
  // 创建结果向量 r，并预留 inputs 大小的空间
  std::vector<decltype(fn(*inputs.begin()))> r;
  r.reserve(inputs.size());
  // 遍历 inputs 中的每个元素，将 fn 应用到元素上，并将结果存入 r 中
  for(const auto & input : inputs)
    r.push_back(fn(input));
  // 返回结果向量 r
  return r;
}

// C++ forbids taking an address of a constructor, so here's a workaround...
// 定义模板函数 fmap，用于将构造函数 R 应用到 inputs 中的每个元素，并返回结果向量
template<typename R, typename T>
inline std::vector<R> fmap(const T& inputs) {
  // 创建结果向量 r，并预留 inputs 大小的空间
  std::vector<R> r;
  r.reserve(inputs.size());
  // 遍历 inputs 中的每个元素，将构造函数 R 应用到元素上，并将结果存入 r 中
  for(auto & input : inputs)
    r.push_back(R(input));
  // 返回结果向量 r
  return r;
}

// 定义模板函数 filter，用于根据 fn 函数过滤 inputs 中的元素，并返回符合条件的结果向量
template<typename F, typename T>
inline std::vector<T> filter(at::ArrayRef<T> inputs, const F& fn) {
  // 创建结果向量 r
  std::vector<T> r;
  r.reserve(inputs.size());
  // 遍历 inputs 中的每个元素，如果 fn 返回 true，则将元素存入 r 中
  for(auto & input : inputs) {
    if (fn(input)) {
      r.push_back(input);
    }
  }
  // 返回结果向量 r
  return r;
}

// 定义模板函数 filter，将 std::vector<T> 转换为 at::ArrayRef<T> 后调用上面的 filter 函数
template<typename F, typename T>
inline std::vector<T> filter(const std::vector<T>& inputs, const F& fn) {
  return filter<F, T>(static_cast<at::ArrayRef<T>>(inputs), fn);
}

} // namespace c10
```