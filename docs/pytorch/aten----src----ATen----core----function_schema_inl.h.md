# `.\pytorch\aten\src\ATen\core\function_schema_inl.h`

```py
#pragma once
#include <ostream>
#include <sstream>

// note: windows build doesn't find symbols in operator files unless
// this is a header file

// 命名空间 c10 中的自定义 << 运算符重载，用于输出 FunctionSchema 对象到流
namespace c10 {

// 实现自定义的 << 运算符重载，用于将 FunctionSchema 对象输出到给定的输出流 out
inline std::ostream& operator<<(std::ostream& out, const FunctionSchema& schema) {
  // eventually this should look almost identical to python arg parser, but
  // it is simpler for now to work directly on this schema

  // 输出函数名称
  out << schema.name();
  // 如果存在重载名称，添加点号和重载名称
  if (!schema.overload_name().empty()) {
    out << "." << schema.overload_name();
  }
  // 输出参数列表开始的左括号
  out << "(";

  bool seen_kwarg_only = false;
  // 遍历函数参数列表
  for (const auto i : c10::irange(schema.arguments().size())) {
    if (i > 0) out << ", ";
    // 如果参数是仅关键字参数并且之前未见到关键字参数，添加星号和逗号
    if (schema.arguments()[i].kwarg_only() && !seen_kwarg_only) {
      out << "*, ";
      seen_kwarg_only = true;
    }
    // 输出参数名称
    out << schema.arguments()[i];
  }

  // 如果函数接受可变参数，添加省略号
  if(schema.is_vararg()) {
    if(!schema.arguments().empty())
      out << ", ";
    out << "...";
  }

  // 输出参数列表结束的右括号
  out << ") -> ";

  const auto& returns = schema.returns();

  /*
   * We should skip parenthesis if we return a single item and it's not varret,
   * or we return nothing but varret.
   *
   * Need special handling for schema
   *   aten::items.str(Dict(str, t) self) -> (str,t)[]
   * Even though this schema returns a single item, we need add parenthesis.
   * The is necessary so the printed schema can be parsed by the C++ SchemaParser
   * Without the extra parenthesis, the parser sees the first parenthesis in '(str,t)' and mistakenly
   * treat the return type as a tuple. An alternative is to enhance the Lexer
   * to lookahead multiple tokens to accurately decide if the return type is
   * a tuple.
   */
  
  // 判断是否需要添加额外的括号来包围返回类型
  bool need_paren = !(
    (returns.size() == 1 && !schema.is_varret()) ||
    (returns.empty() && schema.is_varret()));

  // 如果只有一个返回值且不是可变返回类型，检查返回类型字符串开头是否为左括号，若是则需要添加括号
  if (returns.size() == 1 && !schema.is_varret()) {
    std::stringstream return_ss;
    return_ss << returns.at(0);
    auto return_str = return_ss.str();

    // 如果返回类型非空且以左括号开头，需要添加额外的括号
    if (!return_str.empty() && return_str.front() == '(') {
      need_paren = true;
    }
  }

  // 如果需要添加括号，则在输出流中添加左括号
  if (need_paren) {
    out << "(";
  }
  // 遍历返回值列表，并将每个返回值输出到流中
  for (const auto i : c10::irange(returns.size())) {
    if (i > 0) {
      out << ", ";
    }
    out << returns.at(i);
  }
  // 如果函数有可变返回值，添加省略号
  if (schema.is_varret()) {
    if (!returns.empty()) {
      out << ", ";
    }
    out << "...";
  }
  // 如果需要添加括号，则在输出流中添加右括号
  if (need_paren) {
    out << ")";
  }
  // 返回输出流
  return out;
}

} // namespace c10
// 在给定的参数列表中查找第一个输出参数的索引
inline size_t findFirstOutArg(const std::vector<Argument>& args) {
  // 遍历参数列表，查找第一个输出参数的起始索引
  for (const auto out_start_idx : c10::irange(args.size())) {
    // 如果当前参数是输出参数，则返回其索引
    if (args.at(out_start_idx).is_out()) {
      return out_start_idx;
    }
  }
  // 如果没有找到输出参数，则返回参数列表的大小
  return args.size();
}

// 判断当前参数是否与旧参数兼容，用于向后兼容性检查
inline bool Argument::isBackwardCompatibleWith(
      const Argument& old,
      std::ostream* why_not) const {
    const Argument* lhs = this;
    const Argument* rhs = &old;
    // 检查参数名、数量、别名信息是否相同
    if (!(lhs->name() == rhs->name()
        && lhs->N() == rhs->N()
          && (lhs->alias_info() == rhs->alias_info()
              || (lhs->alias_info() != nullptr && rhs->alias_info() != nullptr
                  && *lhs->alias_info() == *rhs->alias_info())))) {
      return false;
    }
    // 检查是否仅限关键字参数
    if (lhs->kwarg_only() && !rhs->kwarg_only()) {
      return false;
    }
    // 检查类型是否在向后扩展方面兼容
    if (!rhs->type()->isSubtypeOfExt(*lhs->type(), why_not)) {
      return false;
    }
    // 检查默认值是否相同
    if (rhs->default_value().has_value() &&
        lhs->default_value() != rhs->default_value()) {
      return false;
    }
    // 如果所有检查通过，则说明参数向后兼容
    return true;
}

// 判断当前参数是否与旧参数兼容，用于向前兼容性检查
inline bool Argument::isForwardCompatibleWith(
    const Argument& old,
    std::ostream* why_not) const {
  const Argument* lhs = this;
  const Argument* rhs = &old;
  // 检查参数名、数量、别名信息是否相同
  if (!(lhs->name() == rhs->name()
      && lhs->N() == rhs->N()
        && (lhs->alias_info() == rhs->alias_info()
            || (lhs->alias_info() != nullptr && rhs->alias_info() != nullptr
                && *lhs->alias_info() == *rhs->alias_info())))) {
    return false;
  }
  // 检查是否仅限关键字参数
  if (lhs->kwarg_only() && !rhs->kwarg_only()) {
    return false;
  }
  // 检查类型是否在向前扩展方面兼容
  if (!lhs->type()->isSubtypeOfExt(rhs->type(), why_not)) {
    return false;
  }
  // 检查默认值是否相同
  if (rhs->default_value().has_value() &&
      lhs->default_value() != rhs->default_value()) {
    return false;
  }
  // 如果当前参数有默认值而旧参数没有，则不向前兼容
  if (lhs->default_value().has_value() && !rhs->default_value().has_value()) {
    return false;
  }
  // 如果所有检查通过，则说明参数向前兼容
  return true;
}

// 格式化类型不匹配消息，用于生成详细的错误消息
inline std::string FunctionSchema::formatTypeMismatchMsg(
    const Argument& expected,
    const std::string& actual_type,
    std::optional<size_t> position,
    std::optional<std::string> value) const {
  std::string position_str;
  // 如果位置信息存在，则格式化为字符串
  if (position) {
    position_str = c10::str("Position: ", *position, "\n");
  }
  std::string value_str;
  // 如果值信息存在，则格式化为字符串
  if (value) {
    value_str = c10::str("Value: ", *value, "\n");
  }
  // 返回格式化后的错误消息字符串，包括函数名称、类型不匹配详细信息、位置信息、值信息和声明信息
  return c10::str(
      name(),
      "() ",
      expected.formatTypeMismatchMsg(actual_type),
      position_str,
      value_str,
      "Declaration: ",
      *this);
}

// 判断当前函数架构是否与旧函数架构向后兼容
inline bool FunctionSchema::isBackwardCompatibleWith(
    const FunctionSchema& old,
    std::ostream* why_not) const {
  // 检查函数名称、重载名称、是否可变参数和返回值是否向后兼容
  if (!(name() == old.name()
        && overload_name() == old.overload_name()
        // 对于 is_vararg 和 is_varret，我们在内部操作符中保守处理，
        // 因为它们仅由内部运算符使用
        && is_vararg() == old.is_vararg()
        && is_varret() == old.is_varret()
        && returns().size() == old.returns().size()
        && arguments().size() >= old.arguments().size())) {
        // 如果有不向后兼容的条件，则返回 false
    return false;
  }
  // 如果所有检查通过，则说明函数向后兼容

  return true;
}


这些注释解释了给定代码中每个函数和方法的作用及其实现细节，包括参数的兼容性检查、类型不匹配错误消息的格式化等功能。
    // 如果输入参数为空，则返回 false
    return false;
  }
  // 遍历当前函数对象的返回值列表
  for (const auto i : c10::irange(returns().size())) {
    // 为了向后兼容性，参数类型需要协变（即更通用），返回类型需要逆变（即更具体）。
    // 检查当前函数对象的第 i 个返回值是否与旧版本的第 i 个返回值具有向后兼容性
    if (!old.returns().at(i).isBackwardCompatibleWith(
          returns().at(i),
          why_not)) {
      // 如果不兼容，则返回 false
      return false;
    }
  }

  // 需要分别测试输出参数和默认参数
  // 查找旧版本函数的第一个输出参数的索引
  size_t old_out_start_idx = findFirstOutArg(old.arguments());
  // 查找当前函数的第一个输出参数的索引
  size_t new_out_start_idx = findFirstOutArg(arguments());

  // 确保在默认参数中，它们是向后兼容的
  for (const auto i : c10::irange(old_out_start_idx)) {
    // 检查当前函数对象的第 i 个参数是否与旧版本函数对象的第 i 个参数具有向后兼容性
    if (!arguments().at(i).isBackwardCompatibleWith(
          old.arguments().at(i), why_not)) {
      // 如果不兼容，则返回 false
      return false;
    }
  }

  // 验证所有提供的新参数是否具有默认值
  for (const auto i : c10::irange(old_out_start_idx, new_out_start_idx)) {
    // 检查当前函数对象的第 i 个参数是否具有默认值
    if (!arguments().at(i).default_value()) {
      // 如果没有默认值，生成详细错误信息
      if (why_not) {
        *why_not
            << "Function schema not backward compatible since the new argument '"
            << arguments().at(i).name() << "' of type "
            << arguments().at(i).type()->str()
            << " did not provide a default value.";
      }
      // 返回 false，表明不向后兼容
      return false;
    }
  }

  // 现在比较输出参数
  for (const auto i : c10::irange(old_out_start_idx, old.arguments().size())) {
    // 检查当前函数对象的第 i + new_out_start_idx - old_out_start_idx 个参数
    // 是否与旧版本函数对象的第 i 个参数具有向后兼容性
    if (!arguments()
             .at(i - old_out_start_idx + new_out_start_idx)
             .isBackwardCompatibleWith(old.arguments().at(i), why_not)) {
      // 如果不兼容，则返回 false
      return false;
    }
  }

  // 如果所有条件都通过，则返回 true，表明函数对象是向后兼容的
  return true;
// 判断当前函数模式是否与给定的旧函数模式在各方面上兼容
inline bool FunctionSchema::isForwardCompatibleWith(
    const FunctionSchema& old,
    std::ostringstream& why_not) const {
  // 检查函数名称、重载名称、是否可变参数和是否可变返回值是否相同
  if (!(name() == old.name() &&
        overload_name() == old.overload_name()
        // 对于 is_vararg 和 is_varret，我们保守地处理，
        // 因为它们仅由内部运算符使用
        && is_vararg() == old.is_vararg() && is_varret() == old.is_varret() &&
        returns().size() == old.returns().size())) {
    return false;
  }

  // 分别测试输出参数和默认参数
  size_t old_out_start_idx = findFirstOutArg(old.arguments());
  size_t new_out_start_idx = findFirstOutArg(arguments());

  // 检查输出参数数量是否相同
  if (old.arguments().size() - old_out_start_idx !=
      arguments().size() - new_out_start_idx) {
    if (why_not) {
      why_not << "Function schema should have the "
              << "same number of out arguments";
    }
    return false;
  }

  // 确保在默认参数中，它们是向前兼容的
  for (size_t i = 0; i < std::min(old_out_start_idx, new_out_start_idx); i++) {
    if (!arguments().at(i).isForwardCompatibleWith(old.arguments().at(i))) {
      if (why_not) {
        why_not
            << "'" << arguments().at(i).name() << "'"
            << " is not forward compatible with the older version of the schema";
      }
      return false;
    }
  }

  // 验证所有提供的新参数是否有默认值
  for (size_t i = old_out_start_idx; i < new_out_start_idx; ++i) {
    if (!arguments().at(i).default_value()) {
      if (why_not) {
        why_not
            << "Function schema is not forward compatible since the new argument '"
            << arguments().at(i).name() << "' of type "
            << arguments().at(i).type()->str()
            << " did not provide a default value.";
      }
      return false;
    }

    auto default_val = arguments().at(i).default_value().value();
    // 检查默认值是否为列表或泛型字典
    if (default_val.isList() || default_val.isGenericDict()) {
      if (why_not) {
        why_not
            << "Function schema is not forward compatible since the new argument '"
            << arguments().at(i).name() << "' of type "
            << arguments().at(i).type()->str() << " has a container type "
            << "as its default value.";
      }
      return false;
    }
  }

  // 比较输出参数
  for (size_t i = old_out_start_idx; i < old.arguments().size(); i++) {
    if (!arguments()
             .at(i - old_out_start_idx + new_out_start_idx)
             .isForwardCompatibleWith(old.arguments().at(i))) {
      if (why_not) {
        why_not << "Out argument '"
                << "'" << arguments().at(i).name()
                << " is not FC with the older version of the schema";
      }
      return false;
    }
  }

  return true;
}

// 检查参数是否符合预期类型
template<typename T>
inline void FunctionSchema::checkArg(
    const IValue& value,
    const Argument& argument,
    optional<size_t> pos) const {
  // 如果值是张量并且参数类型为 TensorType::get()
  if (value.isTensor() && argument.type() == TensorType::get()) {
    // 快速路径，用于处理常见情况，直接返回，不执行后续逻辑
    return;
  }
  // 检查传入的值是否为类型 T 的子类型
  if (!value.type<T>()->isSubtypeOf(*argument.type())) {
    // 如果类型不匹配，则抛出错误信息
    TORCH_CHECK(
        false,
        // 格式化类型不匹配的错误信息，包括参数名、期望类型和错误位置
        formatTypeMismatchMsg(
            argument, value.type<T>()->repr_str(), pos));
  }
// 找到不在模式参数列表中的关键字参数，并返回错误消息
inline std::string FunctionSchema::findErrorInKwargs(const std::vector<std::string>& kwargs) const {
  // 遍历所有关键字参数，检查是否有未知的关键字（即不在模式参数列表中的）
  for (const auto& kwarg : kwargs) {
    // 如果没有找到与关键字参数名称匹配的模式参数，则返回错误信息
    if (!std::count_if(
            arguments().begin(),
            arguments().end(),
            [&kwarg](const Argument& argument) {
              return argument.name() == kwarg;
            })) {
      return c10::str(
          "Unknown keyword argument '",
          kwarg,
          "' for operator '",
          name(),
          "'. Schema: ",
          *this);
    }
  }
  // 如果有未消耗的关键字参数，但是它们都不是未知的，则第一个位置参数在关键字参数中重复出现
  for (const auto& argument : arguments()) {
    // 如果模式参数名称在关键字参数列表中出现，则说明这个参数同时作为位置参数和关键字参数出现
    if (std::find(kwargs.begin(), kwargs.end(), argument.name()) != kwargs.end()) {
      AT_ASSERT(!argument.default_value()); // 断言确保没有默认值
      return c10::str(
          "Argument '",
          argument.name(),
          "' specified both as positional and ",
          "keyword argument. Schema: ",
          *this);
    }
  }
  // 如果没有发现任何问题，则返回空字符串
  return "";
}

// 检查和规范化输入，并确保输入数量不超过模式参数的数量
template <typename T>
inline void FunctionSchema::checkAndNormalizeInputs(
    std::vector<IValue>& inputs, // 输入值向量
    const std::unordered_map<std::string, IValue>& kwargs) const { // 关键字参数映射
  // 检查是否输入的参数数量超过了模式定义的数量上限
  TORCH_CHECK(
      inputs.size() <= arguments().size(),
      "Expected at most ",
      arguments().size(),
      " argument(s) for operator '",
      name(),
      "', but received ",
      inputs.size(),
      " argument(s). Declaration: ",
      *this);

  size_t consumed_kwargs = 0; // 已消耗的关键字参数计数
  for (const auto pos : c10::irange(arguments().size())) {
    const auto& argument = arguments()[pos]; // 获取当前位置的模式参数
    if (pos < inputs.size()) {
      // 如果当前位置小于输入值数量，则检查该位置的输入值是否合法
      checkArg<T>(inputs[pos], argument, pos);
      continue;
    }
    auto it = kwargs.find(argument.name());
    if (it != kwargs.end()) {
      // 如果模式参数在关键字参数中找到对应值，则检查并将其添加到输入值中
      checkArg<T>(it->second, argument, nullopt);
      inputs.push_back(it->second);
      consumed_kwargs++; // 增加已消耗的关键字参数计数
      continue;
    }
    if (argument.default_value()) {
      // 如果模式参数有默认值，则将其添加到输入值中
      inputs.push_back(*argument.default_value());
      continue;
    }
    // 如果找不到模式参数的值，并且没有默认值，则抛出错误
    AT_ERROR(
        name(),
        "() is missing value for argument '",
        argument.name(),
        "'. Declaration: ",
        *this);
  }
  // 如果有未消耗的关键字参数，则抛出运行时错误并显示详细信息
  if (consumed_kwargs != kwargs.size()) {
    std::vector<std::string> names;
    names.reserve(kwargs.size());
    for(const auto& k : kwargs) {
      names.emplace_back(k.first);
    }
    throw std::runtime_error(findErrorInKwargs(names));
  }
}

// 克隆并重新映射类型的函数模式
inline FunctionSchema FunctionSchema::cloneWithRemappedTypes(
    const std::function<TypePtr(TypePtr)> type_map) const {
  auto update_args = [&](const std::vector<Argument>& args) {
    std::vector<Argument> new_args;
    new_args.reserve(args.size());
    for(const Argument& arg : args) {
      // 使用给定的类型映射函数克隆模式参数，并生成新的模式参数向量
      new_args.emplace_back(arg.cloneWithType(type_map(arg.type())));
    }
    // 返回包含新模式参数的函数模式对象
    # 返回处理后的新参数列表
    return new_args;
  };

  # 返回函数的模式定义，包括函数名、重载名、更新后的参数列表、更新后的返回值列表、是否可变参数、是否可变返回值
  return FunctionSchema(
      name(),
      overload_name(),
      update_args(arguments()),
      update_args(returns()),
      is_vararg(),
      is_varret());
// } 是函数结束的大括号，表示 isSubtypeOfList 函数的结束
}

// covariant subtyping of list of Arguments
// 参数的列表中的协变子类型化
inline bool isSubtypeOfList(
    ArrayRef<Argument> child,  // 子列表的参数，ArrayRef 是 STL 提供的对数组的引用包装
    ArrayRef<Argument> parent, // 父列表的参数，ArrayRef 是 STL 提供的对数组的引用包装
    std::ostream* why_not) {   // 用于输出为什么不是子类型的流

  // 如果子列表和父列表的长度不相等，则不是子类型
  if (child.size() != parent.size()) {
    return false;
  }

  // 遍历子列表和父列表的每个参数
  for (const auto i : c10::irange(child.size())) {
    const Argument& c = child[i];  // 获取子列表中的参数
    const Argument& p = parent[i]; // 获取父列表中的参数

    // 如果子列表和父列表中对应位置的参数名不相同，则不是子类型
    if (c.name() != p.name()) {
      return false;
    }

    // 如果子列表中的参数类型不是父列表中对应位置参数类型的子类型，则不是子类型
    if (!c.type()->isSubtypeOfExt(*p.type(), why_not)) {
      return false;
    }
  }

  // 如果以上条件都满足，则子列表是父列表的子类型
  return true;
}

// FunctionSchema 类的方法，判断当前函数模式是否是 rhs 函数模式的子类型
inline bool FunctionSchema::isSubtypeOf(
    const FunctionSchema& rhs,   // 右手边的函数模式
    bool as_method,              // 是否作为方法
    std::ostream* why_not) const {  // 用于输出为什么不是子类型的流

  size_t start = as_method ? 1 : 0;  // 如果作为方法，从第一个参数开始比较，否则从第0个参数开始比较

  // 函数参数在参数上是反变的，但在返回值上是协变的
  // 检查参数列表部分是否是子类型
  return isSubtypeOfList(
             ArrayRef<Argument>(rhs.arguments()).slice(start),  // 右手边函数的参数列表，从 start 开始的子列表
             ArrayRef<Argument>(arguments()).slice(start),      // 当前函数的参数列表，从 start 开始的子列表
             why_not) &&                                          // 输出为什么不是子类型的流
      isSubtypeOfList(returns(), rhs.returns(), why_not);        // 检查返回值列表是否是子类型
}

// 命名空间 c10 的结束
} // namespace c10
```