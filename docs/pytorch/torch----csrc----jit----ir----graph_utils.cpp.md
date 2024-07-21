# `.\pytorch\torch\csrc\jit\ir\graph_utils.cpp`

```py
namespace torch {
namespace jit {

// 根据给定的 Tensor 对象 t 创建并返回对应的 TensorType
TypePtr getTensorType(const at::Tensor& t, bool complete) {
  auto r = TensorType::create(t);
  // 如果 complete 参数为 false，则返回仅包含维度信息的 TensorType
  if (!complete) {
    r = r->dimensionedOnly();
  }
  return r;
}

// 推断输入的类型和形状
TypePtr inferShapeAndTypeForInput(
    TypePtr input_type,
    Stack::const_iterator& s_iter,
    const Stack::const_iterator& s_iter_end,
    bool complete) {
  // 如果输入类型是 TupleType，则递归处理其中的每个子类型
  if (auto tuple_type = input_type->cast<TupleType>()) {
    std::vector<TypePtr> types;
    for (const auto& sub_type : tuple_type->containedTypes()) {
      TORCH_INTERNAL_ASSERT(s_iter != s_iter_end);
      // 递归推断子类型的形状和类型
      types.emplace_back(
          inferShapeAndTypeForInput(sub_type, s_iter, s_iter_end, complete));
    }
    return TupleType::create(types);
  } else if (auto list_type = input_type->cast<ListType>()) {
    // 如果输入类型是 ListType，则推断列表元素的类型和形状
    const TypePtr& sub_type = list_type->getElementType();
    auto elem_type =
        inferShapeAndTypeForInput(sub_type, s_iter, s_iter_end, complete);
    return ListType::create(elem_type);
  } else if (auto tensor_type = input_type->cast<TensorType>()) {
    // 如果输入类型是 TensorType，则根据堆栈中的 Tensor 推断类型和形状
    auto type = getTensorType(s_iter->toTensor(), complete);
    s_iter++;
    return type;
  } else if (auto optional_type = input_type->cast<OptionalType>()) {
    // 如果输入类型是 OptionalType，则推断其元素类型的类型和形状
    const TypePtr& sub_type = optional_type->getElementType();
    auto elem_type =
        inferShapeAndTypeForInput(sub_type, s_iter, s_iter_end, complete);
    return OptionalType::create(elem_type);
  } else {
    // 如果是原始类型，则直接前进到下一个堆栈元素并保持类型不变
    s_iter++;
    return input_type;
  }
}

// 设置图中输入张量的类型
void setInputTensorTypes(
    Graph& g,
    const Stack& stack,
    bool complete,
    const std::vector<int>& param_count_list) {
  // 获取图中的输入值
  at::ArrayRef<Value*> input_values = g.inputs();
  auto s_iter = stack.begin();
  size_t list_idx = 0;
  // 如果参数计数列表非空，则确保输入值和参数计数列表的大小匹配
  if (!param_count_list.empty()) {
    TORCH_INTERNAL_ASSERT(
        input_values.size() == param_count_list.size(),
        " input_values:",
        input_values.size(),
        " vs param_count_list:",
        param_count_list.size());
  }
  // 遍历图的输入值
  for (auto v : input_values) {
    // 对于具有命名类型的值，跳过类型设置以保持参数打包状态
    if (auto named_type = v->type()->cast<c10::NamedType>()) {
      if (auto qualname = named_type->name()) {
        if (getCustomClass(qualname->qualifiedName())) {
          // 如果参数计数列表为空，简单地前进到下一个堆栈元素
          if (param_count_list.empty()) {
            AT_ASSERT(s_iter != stack.end());
            s_iter++;
          } else {
            // 否则，根据参数计数列表前进相应数量的堆栈元素
            if (param_count_list[list_idx] > 0) {
              AT_ASSERT(s_iter != stack.end());
            }
            s_iter += param_count_list[list_idx];
          }
          list_idx++;
          continue;
        }
      }
    }
    // 对当前输入值进行类型和形状的推断
    auto type =
        inferShapeAndTypeForInput(v->type(), s_iter, stack.end(), complete);
    v->setType(type);
    list_idx++;
  }
}

} // namespace jit
} // namespace torch
```