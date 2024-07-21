# `.\pytorch\aten\src\ATen\core\function_schema.cpp`

```
// 包含 ATen 库中的 function_schema.h 头文件

#include <ATen/core/function_schema.h>

// 包含标准输入输出流、堆栈和实用工具库
#include <iostream>
#include <stack>
#include <utility>

// 定义命名空间 c10
namespace c10 {

// FunctionSchema 类的 dump 方法，用于打印对象内容到标准输出流
void FunctionSchema::dump() const {
  std::cout << *this << "\n";
}

// 根据参数类型返回正确的参数列表或返回值列表
const std::vector<Argument>& FunctionSchema::getCorrectList(SchemaArgType type) const {
  if (type == SchemaArgType::input) {
    return arguments();
  } else {
    return returns();
  }
}

// 克隆 FunctionSchema 对象，使用真实类型替换虚拟类型
FunctionSchema FunctionSchema::cloneWithRealTypes(bool with_symint) const {
  // lambda 函数，根据是否使用 SymInt 克隆参数的真实类型
  auto alwaysCloneWithRealTypes = [&](const Argument& a) {
    return a.cloneWithType(a.real_type());
  };
  auto cloneWithRealTypes = [&](const Argument& a) {
    if (with_symint) {
      return a.cloneWithType(a.real_type());
    }
    // 如果参数类型类似 SymInt，则保留虚拟类型
    // 注意：需要与 KernelFunction_impl.h 中的 unpackSymInt 保持同步
    if (
      *a.real_type() == *getTypePtr<c10::SymInt>() ||
      *a.real_type() == *getTypePtr<std::optional<c10::SymInt>>() ||
      *a.real_type() == *getTypePtr<c10::SymIntArrayRef>() ||
      *a.real_type() == *getTypePtr<at::OptionalSymIntArrayRef>()
    ) {
      // 保留虚拟类型
      return a.cloneWithType(a.type());
    } else {
      // 使用真实类型
      return a.cloneWithType(a.real_type());
    }
  };
  // 创建新的参数和返回值列表，根据上述克隆策略
  std::vector<Argument> new_arguments, new_returns;
  std::transform(arguments().begin(), arguments().end(), std::back_inserter(new_arguments), cloneWithRealTypes);
  // 所有返回值都使用真实类型进行克隆
  std::transform(returns().begin(), returns().end(), std::back_inserter(new_returns), alwaysCloneWithRealTypes);
  // 返回克隆后的 FunctionSchema 对象
  return FunctionSchema(
    name(),
    overload_name(),
    std::move(new_arguments),
    std::move(new_returns),
    is_vararg(),
    is_varret());
}

// 判断两个别名类型集合是否存在类型别名
bool FunctionSchema::canAliasTypeSetsAlias(const std::optional<AliasTypeSet> &lhs, const std::optional<AliasTypeSet> &rhs) const {
  if (!lhs || !rhs) {
    return false;
  }
  // 遍历左右两个别名类型集合，判断是否存在相同的类型指针
  for (const TypePtr& lhsType : *lhs) {
    for (const TypePtr& rhsType : *rhs) {
      if (lhsType == rhsType) {
        return true;
      }
    }
  }
  return false;
}

// 获取别名类型集合中包含的所有类型
std::optional<AliasTypeSet> FunctionSchema::getAliasTypeSetContainedTypes(const std::optional<AliasTypeSet> &aliasTypeSet) const {
  if (!aliasTypeSet) {
    return c10::nullopt;
  }
  // 使用堆栈和哈希集合存储包含的所有类型
  std::unordered_set<TypePtr> containedTypes;
  std::stack<TypePtr> typeStack;
  
  // 将第一级包含类型推入堆栈
  for (const TypePtr& type: *aliasTypeSet) {
    for (const TypePtr& containedType : type->containedTypes()){
      typeStack.push(containedType);
    }
  }

  // 处理堆栈中的更深层次包含类型
  while (!typeStack.empty()) {
    TypePtr current = typeStack.top();
    typeStack.pop();
    // 如果当前类型未被处理过，则继续处理其包含类型
    if (!containedTypes.count(current)) {
      for (const TypePtr& containedType : current->containedTypes()) {
        typeStack.push(containedType);
      }
    }
    // 将当前类型加入已包含类型集合中
    containedTypes.insert(current);
  }

  // 返回包含的所有类型的别名类型集合
  return AliasTypeSet(containedTypes.begin(), containedTypes.end());
}

} // namespace c10
// 判断给定类型是否能映射到别名类型集合，返回一个 std::optional 包含 AliasTypeSet 或者空值
std::optional<AliasTypeSet> FunctionSchema::mapTypeToAliasTypeSet(const TypePtr& type) const {
  // 根据类型的种类进行不同的处理
  switch(type->kind()) {
    // 对于 ListType、DictType、ClassType、TensorType 类型，返回包含对应 unshapedType 的 AliasTypeSet
    case TypeKind::ListType:
    case TypeKind::DictType:
    case TypeKind::ClassType:
    case TypeKind::TensorType:
      return AliasTypeSet {c10::unshapedType(type)};
    // 对于 UnionType 类型，遍历其中的每个类型并递归调用 mapTypeToAliasTypeSet，合并结果
    case TypeKind::UnionType: {
      AliasTypeSet mutable_types;
      for (const TypePtr& inner : type->expectRef<UnionType>().containedTypes()) {
        if (auto maybe_inner_types = mapTypeToAliasTypeSet(inner)) {
          mutable_types.insert(
              mutable_types.end(),
              (*maybe_inner_types).begin(),
              (*maybe_inner_types).end());
        }
      }
      // 如果合并后结果为空，则返回空值
      if (mutable_types.empty()) {
        return c10::nullopt;
      }
      return mutable_types;
    }
    // 对于 AnyType 类型，返回一个包含该类型的 AliasTypeSet
    case TypeKind::AnyType:
      return {AliasTypeSet{type}};
    // 对于 OptionalType 类型，获取其元素类型并递归调用 mapTypeToAliasTypeSet
    case TypeKind::OptionalType: {
      auto inner = type->castRaw<OptionalType>()->getElementType();
      return mapTypeToAliasTypeSet(inner);
    }
    // 对于 TupleType 类型，遍历其中的每个元素类型并递归调用 mapTypeToAliasTypeSet，合并结果
    case TypeKind::TupleType: {
      AliasTypeSet mutable_types;
      for (const TypePtr& inner : type->expectRef<TupleType>().elements()) {
        if (auto maybe_inner_types = mapTypeToAliasTypeSet(inner)) {
          mutable_types.insert(
              mutable_types.end(),
              (*maybe_inner_types).begin(),
              (*maybe_inner_types).end());
        }
      }
      // 如果合并后结果为空，则返回空值；否则，返回一个包含 TupleType 创建结果的 AliasTypeSet
      if (mutable_types.empty()) {
        return c10::nullopt;
      }
      return {AliasTypeSet{TupleType::create(std::move(mutable_types))}};
    }
    // 默认情况下，返回空值表示无法映射该类型到 AliasTypeSet
    default:
      return c10::nullopt;
  }
}

// 判断两个 SchemaArgument 是否可能具有相同的别名集合
bool FunctionSchema::may_alias(const SchemaArgument& lhs, const SchemaArgument& rhs) const {
  // 检查 lhs 的索引是否有效
  TORCH_INTERNAL_ASSERT(
      (lhs.index < getCorrectList(lhs.type).size()),
      "Invalid index for schema.");
  // 检查 rhs 的索引是否有效
  TORCH_INTERNAL_ASSERT(
      (rhs.index < getCorrectList(rhs.type).size()),
      "Invalid index for schema.");

  // 获取 lhs 和 rhs 对应索引位置的 Argument
  const Argument lhsArg = getCorrectList(lhs.type)[lhs.index];
  const Argument rhsArg = getCorrectList(rhs.type)[rhs.index];

  // 分别映射 lhsArg 和 rhsArg 的类型到 AliasTypeSet
  std::optional<AliasTypeSet> lhsTypes = mapTypeToAliasTypeSet(lhsArg.type());
  std::optional<AliasTypeSet> rhsTypes = mapTypeToAliasTypeSet(rhsArg.type());

  // 检查 lhsTypes 和 rhsTypes 是否具有相同的别名集合
  if (canAliasTypeSetsAlias(lhsTypes, rhsTypes)) {
    // 如果 lhsArg 和 rhsArg 都有 alias_info，则逐一比较它们的 afterSets
    if (lhsArg.alias_info() && rhsArg.alias_info()) {
      for (const auto& lhsSet : lhsArg.alias_info()->afterSets()) {
        for (const auto& rhsSet : rhsArg.alias_info()->afterSets()) {
          // 如果存在相同的 afterSet，则返回 true
          if (lhsSet == rhsSet) {
            return true;
          }
        }
      }
    }
  }

  // 否则返回 false
  return false;
}

// 判断两个 SchemaArgument 是否可能包含别名，考虑可能的双向性
bool FunctionSchema::may_contain_alias(const SchemaArgument& lhs, const SchemaArgument& rhs, bool bidirectional) const {
  // 判断是否存在别名
  bool may_alias_result = may_alias(lhs, rhs);
  if (may_alias_result) {
    // 返回 true，结束函数
    return true;
  }

  // 获取左操作数的正确列表中的参数
  const c10::Argument lhsArg = getCorrectList(lhs.type)[lhs.index];
  // 获取右操作数的正确列表中的参数
  const c10::Argument rhsArg = getCorrectList(rhs.type)[rhs.index];
  // 将左操作数的类型映射为别名类型集合，返回可选类型
  std::optional<AliasTypeSet> lhsTypes = mapTypeToAliasTypeSet(lhsArg.type());
  // 将右操作数的类型映射为别名类型集合，返回可选类型
  std::optional<AliasTypeSet> rhsTypes = mapTypeToAliasTypeSet(rhsArg.type());
  // 获取左操作数类型的别名类型集合的包含类型，返回可选类型
  std::optional<AliasTypeSet> lhsContainedTypes = getAliasTypeSetContainedTypes(lhsTypes);
  // 获取右操作数类型的别名类型集合的包含类型，返回可选类型
  std::optional<AliasTypeSet> rhsContainedTypes = getAliasTypeSetContainedTypes(rhsTypes);

  // 检查一侧是否为通配符，另一侧是否为相同类型的容器
  bool lhsWildcard = lhsArg.alias_info() && lhsArg.alias_info()->isWildcardAfter() && canAliasTypeSetsAlias(lhsTypes, rhsContainedTypes);
  bool rhsWildcard = rhsArg.alias_info() && rhsArg.alias_info()->isWildcardAfter() && canAliasTypeSetsAlias(rhsTypes, lhsContainedTypes);

  // 如果是双向的比较，则返回左侧通配符或右侧通配符或左右类型集合的别名是否可以别名
  if (bidirectional) {
    return lhsWildcard || rhsWildcard || canAliasTypeSetsAlias(lhsContainedTypes, rhsContainedTypes);
  } else {
    // 如果不是双向比较，则返回右侧通配符或左右类型集合的别名是否可以别名
    return rhsWildcard || canAliasTypeSetsAlias(lhsContainedTypes, rhsContainedTypes);
  }
}
} // namespace c10


// 结束 c10 命名空间的定义
}
// 关闭 c10 命名空间
} // namespace c10
```