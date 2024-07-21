# `.\pytorch\torch\csrc\autograd\edge.h`

```
#pragma once
// 声明了一个预处理指令#pragma once，确保此头文件只被编译一次

#include <cstdint>
// 包含了用于定义特定大小整数类型的头文件cstdint

#include <functional>
// 包含了函数对象和函数调用包装器的头文件functional

#include <memory>
// 包含了智能指针和相关工具的头文件memory

#include <c10/util/hash.h>
// 包含了C10库中与哈希相关的实用工具的头文件hash.h

namespace torch::autograd {

struct Node;
// 声明了一个名为Node的结构体，用于表示计算图中的节点

/// Represents a particular input of a function.
// 表示函数的特定输入的结构体Edge
struct Edge {
  Edge() noexcept : function(nullptr), input_nr(0) {}
  // 默认构造函数，初始化function为空指针，input_nr为0

  Edge(std::shared_ptr<Node> function_, uint32_t input_nr_) noexcept
      : function(std::move(function_)), input_nr(input_nr_) {}
  // 构造函数，接受一个shared_ptr<Node>和一个uint32_t作为参数

  /// Convenience method to test if an edge is valid.
  // 便利方法，用于测试Edge是否有效
  bool is_valid() const noexcept {
    return function != nullptr;
  }

  // Required for use in associative containers.
  // 必需的成员函数，用于在关联容器中使用Edge对象进行比较
  bool operator==(const Edge& other) const noexcept {
    return this->function == other.function && this->input_nr == other.input_nr;
  }

  bool operator!=(const Edge& other) const noexcept {
    return !(*this == other);
  }

  /// The function this `Edge` points to.
  // 指向的函数的shared_ptr<Node>成员变量
  std::shared_ptr<Node> function;

  /// The identifier of a particular input to the function.
  // 函数的特定输入标识符，uint32_t类型的成员变量
  uint32_t input_nr;
};
} // namespace torch::autograd

// The idiomatic way of enabling use of a custom type as the key of hash
// containers in C++11. This method removes the requirement of having to pass
// a custom hasher to std::unordered_{map, set}.
// See http://en.cppreference.com/w/cpp/utility/hash for more information.

// 声明了一个std命名空间下的模板特化，用于定义torch::autograd::Edge类型的哈希函数
namespace std {
template <>
struct hash<torch::autograd::Edge> {
  // 标准要求的类型别名
  using argument_type = torch::autograd::Edge;
  using return_type = size_t;
  
  // 哈希函数的重载运算符，接受一个Edge对象作为参数，返回其哈希值
  return_type operator()(const argument_type& edge) const noexcept {
    return c10::get_hash(edge.function, edge.input_nr);
  }
};
} // namespace std
```