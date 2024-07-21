# `.\pytorch\aten\src\ATen\core\Variadic.h`

```py
#pragma once
// 防止头文件被多重包含

#include <utility>
// 包含标准库中的 utility 头文件

#include <c10/util/ArrayRef.h>
// 包含 c10 库中的 ArrayRef 头文件

#include <ATen/core/List.h>
// 包含 ATen 库中的 List 头文件

namespace at {
// 进入 at 命名空间

// 这个类允许你编写可变参数函数，可以依次调用（可能是重载的）函数处理每个参数。
// 这在自动生成的代码中最常见，因为可以方便地处理不同类型的参数。
// 如果参数是同质的，请考虑使用 std::initializer_list。
//
// 有关其使用示例，请参见 torch/csrc/utils/variadic.h。
template <typename F>
struct IterArgs {
  // 应用函数的模板方法，用于处理参数列表中的每一个参数
  template <typename... Args>
  inline F& apply() {
    return self();
  }

  // 使用完美转发，以避免对所有参数进行值复制！
  template <typename T, typename... Args>
  inline F& apply(T&& arg, Args&&... args) {
    self()(std::forward<T>(arg));
    // 如果有短路要求，则直接返回当前对象
    if (self().short_circuit()) {
      return self();
    } else {
      // 否则继续递归调用 apply 处理剩余的参数
      return apply(std::forward<Args>(args)...);
    }
  }

  // 下面是一些有用的重载函数，为容器结构提供了合理的默认处理方式，可以递归处理这些结构。
  // 如果想启用它们，请在你的结构体中添加：
  //
  //    using IterArgs<YourStructName>::operator()
  //
  // 这些重载函数默认不启用，因为你可能比单独处理这些结构体更有效率。

  // 处理 c10 库中的 IListRef<T> 类型参数
  template <typename T>
  void operator()(c10::IListRef<T> args) {
    for (const auto& arg : args) {
      self()(arg);
      if (self().short_circuit())
        return;
    }
  }

  // 处理 ATen 库中的 ArrayRef<T> 类型参数
  template <typename T>
  void operator()(at::ArrayRef<T> args) {
    for (const auto& arg : args) {
      self()(arg);
      if (self().short_circuit())
        return;
    }
  }

  // 处理 torch::List<T> 类型参数
  template <typename T>
  void operator()(const torch::List<T>& args) {
    for (const auto& arg : args) {
      self()(arg);
      if (self().short_circuit())
        return;
    }
  }

  // 处理 std::vector<T> 类型参数
  // 需要手动指定，因为 C++ 不会进行隐式转换以使模板推导成功
  template <typename T>
  void operator()(const std::vector<T>& args) {
    self()(at::ArrayRef<T>{args});
  }

  // 返回短路状态，默认为 false，不短路
  constexpr bool short_circuit() const {
    return false;
  }

 private:
  // 返回自身的引用，通过静态转型实现
  inline F& self() {
    return *static_cast<F*>(this);
  }
};

} // namespace at
// 结束 at 命名空间
```