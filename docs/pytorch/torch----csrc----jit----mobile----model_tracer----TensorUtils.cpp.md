# `.\pytorch\torch\csrc\jit\mobile\model_tracer\TensorUtils.cpp`

```
namespace torch {
namespace jit {
namespace mobile {
// 定义函数，用于递归遍历给定的 IValue，对其中的 Tensor 类型执行指定操作
void for_each_tensor_in_ivalue(
    const c10::IValue& iv,  // 输入参数，表示当前处理的 IValue
    std::function<void(const ::at::Tensor&)> const& func) {  // 函数对象参数，用于处理每个 Tensor
  // 判断当前 IValue 是否为叶子节点类型，如果是则不执行任何操作，直接返回
  const bool is_leaf_type = iv.isString() || iv.isNone() || iv.isScalar() ||
      iv.isDouble() || iv.isInt() || iv.isBool() || iv.isDevice() ||
      iv.isIntList() || iv.isDoubleList() || iv.isBoolList();
  if (is_leaf_type) {
    // 叶子节点类型，无需处理，直接返回
    return;
  }

  if (iv.isTensor()) {  // 如果当前 IValue 是 Tensor 类型
    func(iv.toTensor());  // 调用传入的函数对象处理该 Tensor
  } else if (iv.isTuple()) {  // 如果当前 IValue 是 Tuple 类型
    c10::intrusive_ptr<at::ivalue::Tuple> tup_ptr = iv.toTuple();  // 获取 Tuple 智能指针
    for (const auto& e : tup_ptr->elements()) {  // 遍历 Tuple 的每个元素
      for_each_tensor_in_ivalue(e, func);  // 递归处理每个元素中的 Tensor
    }
  } else if (iv.isList()) {  // 如果当前 IValue 是 List 类型
    c10::List<c10::IValue> l = iv.toList();  // 获取 List
    for (auto&& i : l) {  // 遍历 List 中的每个元素
      c10::IValue item = i;
      for_each_tensor_in_ivalue(item, func);  // 递归处理每个元素中的 Tensor
    }
  } else if (iv.isGenericDict()) {  // 如果当前 IValue 是 GenericDict 类型
    c10::Dict<c10::IValue, c10::IValue> dict = iv.toGenericDict();  // 获取 GenericDict
    for (auto& it : dict) {  // 遍历 GenericDict 中的每对键值对
      for_each_tensor_in_ivalue(it.value(), func);  // 递归处理每个值中的 Tensor
    }
  } else {
    AT_ERROR("Unhandled type of IValue. Got ", iv.tagKind());  // 如果遇到未处理的 IValue 类型，抛出错误
  }
}
} // namespace mobile
} // namespace jit
} // namespace torch
```