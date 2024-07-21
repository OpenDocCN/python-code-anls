# `.\pytorch\torch\csrc\jit\ir\node_hashing.cpp`

```
#include <torch/csrc/jit/ir/ir.h>  // 引入 Torch JIT IR 相关头文件

#include <algorithm>  // 引入标准库中的算法模块
#include <unordered_map>  // 引入标准库中的无序映射模块

#include <ATen/core/functional.h>  // 引入 ATen 核心库中的功能模块
#include <ATen/core/symbol.h>  // 引入 ATen 核心库中的符号模块
#include <c10/util/Exception.h>  // 引入 C10 实用工具中的异常处理模块
#include <c10/util/hash.h>  // 引入 C10 实用工具中的哈希模块
#include <c10/util/irange.h>  // 引入 C10 实用工具中的范围模块
#include <torch/csrc/jit/ir/node_hashing.h>  // 引入 Torch JIT IR 中的节点哈希模块
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>  // 引入 Torch JIT 中的公共子表达式消除模块

namespace torch::jit {  // 进入 torch::jit 命名空间

namespace {  // 匿名命名空间，用于隐藏实现细节

bool tensorEqual(const at::Tensor& lhs, const at::Tensor& rhs) {
  // 检查两个 Tensor 是否相等，不考虑 mkldnn 或嵌套 Tensor
  if (lhs.is_mkldnn() || rhs.is_mkldnn()) {
    return false;
  }
  if (lhs.is_nested() || rhs.is_nested()) {
    return false;
  }
  // 如果设备不同，直接返回不相等
  if (lhs.device() != rhs.device()) {
    return false;
  }
  // 检查 Tensor 的选项是否相等，并比较内容是否相等
  return lhs.options().type_equal(rhs.options()) && lhs.equal(rhs);
}

bool typeListEqual(
    const std::vector<TypePtr>& lhs,
    const std::vector<TypePtr>& rhs) {
  // 比较两个 TypePtr 的向量是否相等
  if (lhs.size() != rhs.size())
    return false;
  // 逐一比较向量中的元素是否相等
  for (const auto i : c10::irange(lhs.size())) {
    if (*lhs[i] != *rhs[i]) {
      return false;
    }
  }
  return true;
}

template <typename attribute_type> // int64_t, bool, double
bool attributesEqual(attribute_type a1, attribute_type a2) {
  // 比较两个基本类型的属性是否相等
  return a1 == a2;
}

bool attributesEqual(const at::Tensor& a1, const at::Tensor& a2) {
  // 比较两个 Tensor 是否相等
  return tensorEqual(a1, a2);
}

bool attributesEqual(
    const std::vector<at::Tensor>& lhs,
    const std::vector<at::Tensor>& rhs) {
  // 比较两个 Tensor 向量是否相等
  if (lhs.size() != rhs.size())
    return false;
  // 使用 tensorEqual 函数比较两个向量中对应位置的 Tensor 是否相等
  return std::equal(lhs.begin(), lhs.end(), rhs.begin(), tensorEqual);
}

bool attributesEqual(at::ArrayRef<IValue> a1, at::ArrayRef<IValue> a2) {
  // 比较两个 IValue 数组是否相等
  if (a1.size() != a2.size()) {
    return false;
  }
  // 逐一比较数组中的元素是否相等
  for (const auto i : c10::irange(a1.size())) {
    if (!ivaluesEqual(a1[i], a2[i])) {
      return false;
    }
  }
  return true;
}

bool attributesEqual(const IValue& a1, const IValue& a2) {
  // 比较两个 IValue 是否相等
  return ivaluesEqual(a1, a2);
}

// this is not a general-purpose comparison of IValues, it only covers the
// ivalues that are allowed as attributes, and it does not check type
// equivalence of containers.
bool ivaluesEqual(const IValue& a1, const IValue& a2) {
  // 比较两个 IValue 是否相等，限制于允许用作属性的 IValue，不检查容器类型的类型等价性
  if (a1.tagKind() != a2.tagKind()) {
    return false;
  }
  if (a1.isInt()) {
    return a1.toInt() == a2.toInt();
  }
  if (a1.isBool()) {
    return a1.toBool() == a2.toBool();
  }
  if (a1.isDouble()) {
    return a1.toDouble() == a2.toDouble();
  }
  if (a1.isTensor()) {
    return attributesEqual(a1.toTensor(), a2.toTensor());
  }
  if (a1.isNone()) {
    return true;
  }
  if (a1.isString()) {
    return a1.toStringRef() == a2.toStringRef();
  }
  if (a1.isList()) {
    return attributesEqual(a1.toListRef(), a2.toListRef());
  }
  if (a1.isTuple()) {
    at::ArrayRef<IValue> a1_elem = a1.toTupleRef().elements();
    // 如果 a1 和 a2 是 Tensor 类型，比较它们的元素是否相等
    at::ArrayRef<IValue> a2_elem = a2.toTupleRef().elements();
    return attributesEqual(a1_elem, a2_elem);
  }
  // 如果 a1 和 a2 是 GenericDict 类型
  if (a1.isGenericDict()) {
    // 获取 a1 和 a2 的 GenericDict 对象
    auto a1_dict = a1.toGenericDict();
    auto a2_dict = a2.toGenericDict();
    // 如果两个字典的大小不相等，则返回 false
    if (a1_dict.size() != a2_dict.size()) {
      return false;
    }

    // 使用迭代器遍历字典中的元素
    auto it_a1 = a1_dict.begin();
    auto it_a2 = a2_dict.begin();

    while (it_a1 != a1_dict.end()) {
      // 获取当前迭代器指向的键值对
      const auto& e_a1 = *it_a1;
      const auto& e_a2 = *it_a2;

      // 如果键或值不相等，则返回 false
      if (!ivaluesEqual(e_a1.key(), e_a2.key()) ||
          !ivaluesEqual(e_a1.value(), e_a2.value())) {
        return false;
      }
      // 向前移动迭代器
      it_a1++;
      it_a2++;
    }
    // 如果所有键值对都相等，则返回 true
    return true;
  }
  // 如果 a1 和 a2 是 Enum 类型，直接比较它们的值是否相等
  if (a1.isEnum()) {
    return a1.toEnumHolder() == a2.toEnumHolder();
  }
  // 如果 a1 和 a2 是 Object 类型，比较它们的引用是否相等
  if (a1.isObject()) {
    return &a1.toObjectRef() == &a2.toObjectRef();
  }
  // 如果 a1 和 a2 是 Generator 类型，比较它们的值是否相等
  if (a1.isGenerator()) {
    return a1.toGenerator() == a2.toGenerator();
  }
  // 如果以上所有情况都不符合，则断言失败
  TORCH_INTERNAL_ASSERT(false);
// Check whether two nodes have the same attributes in CSE.
// This function may be too conservative for general use.
// Do NOT support g/gs attributes.
bool attributesEqualCSE(const Node* lhs, const Node* rhs) {
  AT_ASSERT(lhs != nullptr); // Assert that lhs pointer is not null
  AT_ASSERT(rhs != nullptr); // Assert that rhs pointer is not null
  
  // One has attributes, the other does not.
  if (lhs->hasAttributes() != rhs->hasAttributes())
    return false;
  
  // Neither has attributes.
  if (!lhs->hasAttributes() && !rhs->hasAttributes())
    return true;

  // Retrieve attribute names for both nodes and sort them
  auto lnames = lhs->attributeNames();
  auto rnames = rhs->attributeNames();
  std::sort(lnames.begin(), lnames.end());
  std::sort(rnames.begin(), rnames.end());
  
  // If attribute names are not identical, return false
  if (lnames != rnames)
    return false;

  // Iterate through sorted attribute names
  for (auto name : lnames) {
    // Switch based on attribute kind
    switch (lhs->kindOf(name)) {
      // Macro to compare attribute values based on their kind
      #define COMPARE_ATTRIBUTEVALUE(selector)                            \
        case AttributeKind::selector: {                                   \
          if (!attributesEqual(lhs->selector(name), rhs->selector(name))) \
            return false;                                                 \
        } break;
      
      // Compare attributes based on their kind
      COMPARE_ATTRIBUTEVALUE(f)
      COMPARE_ATTRIBUTEVALUE(c)
      COMPARE_ATTRIBUTEVALUE(fs)
      COMPARE_ATTRIBUTEVALUE(cs)
      COMPARE_ATTRIBUTEVALUE(i)
      COMPARE_ATTRIBUTEVALUE(is)
      COMPARE_ATTRIBUTEVALUE(s)
      COMPARE_ATTRIBUTEVALUE(ss)
      COMPARE_ATTRIBUTEVALUE(t)
      COMPARE_ATTRIBUTEVALUE(ts)
      COMPARE_ATTRIBUTEVALUE(ival)
      case AttributeKind::ty:
        // Compare 'ty' attribute values
        if (*lhs->ty(name) != *rhs->ty(name)) {
          return false;
        }
        break;
      case AttributeKind::tys:
        // Compare 'tys' attribute values
        if (!typeListEqual(lhs->tys(name), rhs->tys(name))) {
          return false;
        }
        break;
      case AttributeKind::g:
      case AttributeKind::gs:
        // Attributes 'g' and 'gs' are not supported, return false
        return false;
    }

    // Undefine the macro after usage
    #undef COMPARE_ATTRIBUTEVALUE
  }

  // If all comparisons passed, return true
  return true;
}

} // anonymous namespace

// Makes a hash that hashes the input Value, the output type
// as well as the node attributes
size_t HashNode::operator()(const Node* k) const {
  AT_ASSERT(k != nullptr); // Assert that k pointer is not null
  size_t constant_hash = 0;
  
  // Compute hash based on node type and attributes
  if (k->kind() == prim::Constant) {
    TypePtr type = k->output()->type();
    if (type->isSubtypeOf(*NumberType::get()) &&
        k->kindOf(attr::value) == AttributeKind::i) {
      constant_hash = std::hash<int64_t>{}(k->i(attr::value));
    } else if (
        type->isSubtypeOf(*NumberType::get()) &&
        k->kindOf(attr::value) == AttributeKind::f) {
      constant_hash = std::hash<double>{}(k->f(attr::value));
    } else if (
        type->isSubtypeOf(*NumberType::get()) &&
        k->kindOf(attr::value) == AttributeKind::c) {
      constant_hash = c10::hash<c10::complex<double>>{}(k->c(attr::value));
    } else if (type->isSubtypeOf(*BoolType::get())) {
      constant_hash = std::hash<bool>{}(k->i(attr::value));
    }
  }

  // Return computed constant hash
  return constant_hash;
}
    }
  }

这两行可能是函数的结束部分，但在当前上下文中缺少完整的函数定义，因此具体含义不明确。


  return get_hash(
      k->kind(),
      fmap(k->outputs(), [](const Value* v) { return v->type()->kind(); }),
      fmap(k->inputs(), [](const Value* v) { return v->unique(); }),
      constant_hash);

返回一个哈希值，该哈希值通过 `get_hash` 函数计算得出，包括以下参数：
- `k->kind()`：获取对象 `k` 的种类信息。
- `fmap(k->outputs(), [](const Value* v) { return v->type()->kind(); })`：对 `k` 的输出进行映射操作，返回一个列表，列表元素为每个输出值 `v` 的类型种类信息。
- `fmap(k->inputs(), [](const Value* v) { return v->unique(); })`：对 `k` 的输入进行映射操作，返回一个列表，列表元素为每个输入值 `v` 的唯一标识符。
- `constant_hash`：可能是一个常量参数，参与哈希计算的额外信息。

以上是针对给定代码片段的注释，描述了每一行代码的作用和可能的参数来源。
// 定义一个名为 EqualNode 的类，实现了 () 运算符用于比较两个节点是否相等
bool EqualNode::operator()(const Node* lhs, const Node* rhs) const {
    // 如果左右节点均为空指针，则认为它们相等
    if (lhs == nullptr && rhs == nullptr)
        return true;
    // 如果左右节点其中一个为空指针，则认为它们不相等
    if (lhs == nullptr || rhs == nullptr)
        return false;

    // 检查节点的类型是否相同
    if (lhs->kind() != rhs->kind())
        return false;

    // 检查节点的输出类型是否相同
    auto lhs_outputs = lhs->outputs();
    auto rhs_outputs = rhs->outputs();
    if (lhs_outputs.size() != rhs_outputs.size())
        return false;
    for (const auto i : c10::irange(lhs_outputs.size())) {
        const auto& lt = lhs_outputs[i]->type();
        const auto& rt = rhs_outputs[i]->type();
        // 如果输出类型不相同，则认为节点不相等
        if (!(lt == rt || *lt == *rt))
            return false;
    }

    // 检查节点的输入是否相同
    auto lhs_inputs = lhs->inputs();
    auto rhs_inputs = rhs->inputs();
    if (lhs_inputs.size() != rhs_inputs.size())
        return false;
    // 使用 std::equal 检查输入列表是否相同
    if (!std::equal(lhs_inputs.begin(), lhs_inputs.end(), rhs_inputs.begin()))
        return false;

    // 检查节点的属性是否相同
    if (!attributesEqualCSE(lhs, rhs))
        return false;

    // 检查节点的块是否相同
    if (lhs->blocks().size() != rhs->blocks().size()) {
        return false;
    }
    for (size_t i = 0; i < lhs->blocks().size(); ++i) {
        // 如果任意一个块不相等，则认为节点不相等
        if (lhs->blocks()[i] != rhs->blocks()[i]) {
            return false;
        }
    }

    // 如果所有条件都满足，则认为节点相等
    return true;
}

// 命名空间结束声明，结束了 torch::jit 命名空间的定义
} // namespace torch::jit
```