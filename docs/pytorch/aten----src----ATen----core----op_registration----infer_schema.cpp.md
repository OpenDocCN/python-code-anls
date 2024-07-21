# `.\pytorch\aten\src\ATen\core\op_registration\infer_schema.cpp`

```
namespace c10 {

namespace detail {
namespace infer_schema {
namespace {

// 创建一个 Argument 对象的向量，根据给定的 ArgumentDef 对象集合 args
std::vector<Argument> createArgumentVector(c10::ArrayRef<ArgumentDef> args) {
  std::vector<Argument> result;
  result.reserve(args.size());
  // 遍历 args 的索引范围
  for (const auto i : c10::irange(args.size())) {
    // 每个 Argument 的名称为 "_<index>"
    result.emplace_back(
        fmt::format("_{}", i),
        (*args[i].getFakeTypeFn)(),
        (*args[i].getTypeFn)());
  }
  return result;
}
} // namespace

// 创建函数模式的工具函数，返回一个 FunctionSchema 对象
// 此函数在 .cpp 文件中实现，以减小模板大小，有利于二进制文件大小
FunctionSchema make_function_schema(
    std::string&& name,
    std::string&& overload_name,
    c10::ArrayRef<ArgumentDef> arguments,
    c10::ArrayRef<ArgumentDef> returns) {
  return FunctionSchema(
      std::move(name),
      std::move(overload_name),
      createArgumentVector(arguments),
      createArgumentVector(returns));
}

// 创建函数模式的工具函数的重载，使用默认的名称和重载名称
FunctionSchema make_function_schema(
    c10::ArrayRef<ArgumentDef> arguments,
    c10::ArrayRef<ArgumentDef> returns) {
  return make_function_schema("", "", arguments, returns);
}
} // namespace infer_schema
} // namespace detail

// 查找两个函数模式之间的差异，返回可选的错误消息
std::optional<std::string> findSchemaDifferences(
    const FunctionSchema& lhs,
    const FunctionSchema& rhs) {
  // 检查参数数量是否一致
  if (lhs.arguments().size() != rhs.arguments().size()) {
    return fmt::format(
        "The number of arguments is different. {} vs {}.",
        lhs.arguments().size(),
        rhs.arguments().size());
  }
  // 检查返回值数量是否一致
  if (lhs.returns().size() != rhs.returns().size()) {
    return fmt::format(
        "The number of returns is different. {} vs {}.",
        lhs.returns().size(),
        rhs.returns().size());
  }

  // 检查每个参数的类型是否一致
  for (const auto i : c10::irange(lhs.arguments().size())) {
    const TypePtr& leftType = lhs.arguments()[i].type();
    const TypePtr& rightType = rhs.arguments()[i].type();
    // 首先比较指针，然后再比较类型，以提高效率
    if (leftType.get() != rightType.get() && *leftType != *rightType) {
      return fmt::format(
          "Type mismatch in argument {}: {} vs {}.",
          i + 1,
          lhs.arguments()[i].type()->str(),
          rhs.arguments()[i].type()->str());
    }
  }

  // 检查每个返回值的类型是否一致
  for (const auto i : c10::irange(lhs.returns().size())) {
    const TypePtr& leftType = lhs.returns()[i].type();
    const TypePtr& rightType = rhs.returns()[i].type();
    // 首先比较指针，然后再比较类型，以提高效率
    if (leftType.get() != rightType.get() && *leftType != *rightType) {
      return fmt::format(
          "Type mismatch in return {}: {} vs {}.",
          i + 1,
          lhs.returns()[i].type()->str(),
          rhs.returns()[i].type()->str());
    }
  }

  // 没有找到差异，返回空的可选类型
  return c10::nullopt;
}

} // namespace c10
```