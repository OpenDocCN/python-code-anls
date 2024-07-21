# `.\pytorch\aten\src\ATen\core\function_schema.h`

```
#pragma once

#include <c10/util/StringUtil.h>
#include <c10/util/string_view.h>
#include <c10/util/irange.h>
#include <ATen/core/jit_type.h>
#include <ATen/core/symbol.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/alias_info.h>
#include <ATen/core/operator_name.h>
#include <ATen/core/dispatch/OperatorOptions.h>
#include <unordered_map>
#include <utility>

namespace c10 {

// schema as used in the compiler for resolving function calls and reporting
// errors. These objects should be constructed from C10 schema once those
// are available.

// Argument struct represents an argument to a function or method.
struct Argument;

// FunctionSchema is used to define the schema of a function.
struct FunctionSchema;

// AliasTypeSet is a vector of TypePtr used to represent a set of alias types.
using AliasTypeSet = std::vector<TypePtr>;

// Operator for comparing two Argument objects for equality.
bool operator==(const Argument& lhs, const Argument& rhs);

// Argument struct represents an argument to a function or method.
struct Argument {
  // Constructor initializing Argument object with optional parameters.
  Argument(
      std::string name = "",
      const TypePtr& type = nullptr,
      std::optional<int32_t> N = c10::nullopt,
      std::optional<IValue> default_value = c10::nullopt,
      bool kwarg_only = false,
      std::optional<AliasInfo> alias_info = c10::nullopt)
    : Argument(std::move(name), type, type, N, std::move(default_value), kwarg_only, std::move(alias_info)) {}

  // Constructor initializing Argument object with detailed type information.
  Argument(
      std::string name,
      TypePtr fake_type,
      TypePtr real_type,
      std::optional<int32_t> N = c10::nullopt,
      std::optional<IValue> default_value = c10::nullopt,
      bool kwarg_only = false,
      std::optional<AliasInfo> alias_info = c10::nullopt)
      : name_(std::move(name)),
        type_(fake_type ? std::move(fake_type) : TensorType::get()),
        real_type_(real_type ? std::move(real_type) : type_),
        N_(N),
        default_value_(std::move(default_value)),
        alias_info_(alias_info ? std::make_unique<AliasInfo>(std::move(*alias_info)) : nullptr),
        kwarg_only_(kwarg_only) {
    // Determine if the argument is an 'out' parameter based on alias information.
    bool is_alias = alias_info_ != nullptr && alias_info_->isWrite();
    is_out_ = kwarg_only_ && is_alias;
  }

  // Move constructor for Argument object.
  Argument(Argument&& rhs) noexcept = default;

  // Copy constructor for Argument object.
  Argument(const Argument& rhs)
      : name_(rhs.name_),
        type_(rhs.type_),
        real_type_(rhs.real_type_),
        N_(rhs.N_),
        default_value_(rhs.default_value_),
        alias_info_(rhs.alias_info_ ? std::make_unique<AliasInfo>(*rhs.alias_info_) : nullptr),
        kwarg_only_(rhs.kwarg_only_),
        is_out_(rhs.is_out_) {}

  // Move assignment operator for Argument object.
  Argument& operator=(Argument&& rhs) = default;

  // Copy assignment operator for Argument object.
  Argument& operator=(const Argument& rhs) {
    if (this != &rhs) {
      name_ = rhs.name_;
      type_ = rhs.type_;
      real_type_ = rhs.real_type_;
      N_ = rhs.N_;
      default_value_ = rhs.default_value_;
      alias_info_ = rhs.alias_info_ ? std::make_unique<AliasInfo>(*rhs.alias_info_) : nullptr;
      kwarg_only_ = rhs.kwarg_only_;
      is_out_ = rhs.is_out_;
    }
    return *this;
  }

  // Accessor method for retrieving the argument's name.
  const std::string& name() const {
    return name_;
  }

  // Accessor method for retrieving the argument's type.
  const TypePtr& type() const {
    return type_;
  }
  // 返回参数的类型
  return type_;
}
// 如果 type() 不为空，则返回真实类型；如果未提供真实类型，则使用 type() 的值
const TypePtr& real_type() const {
  return real_type_;
}
// 返回参数的 N 值，作为可选的整数值
std::optional<int32_t> N() const {
  return N_;
}
// 返回参数的默认值，作为可选的 IValue 对象
const std::optional<IValue>& default_value() const {
  return default_value_;
}
// 返回参数是否仅限关键字传参
bool kwarg_only() const {
  return kwarg_only_;
}

// 返回参数是否为输出参数
bool is_out() const {
  return is_out_;
}

// 返回参数的别名信息指针，可能为空
C10_NODISCARD const AliasInfo* alias_info() const {
  return alias_info_.get();
}

// 返回参数是否为推断类型（即没有显式类型注解的情况下推断为 Tensor 类型）
bool is_inferred_type() const {
  bool is_inferred_type = false;
  TORCH_INTERNAL_ASSERT(type_);
  if (auto pt = type_->cast<TensorType>()) {
    if (pt->isInferredType()) {
      is_inferred_type = true;
    }
  }
  return is_inferred_type;
}

// 格式化类型不匹配的错误消息，包括实际类型和推断类型提示（如果适用）
std::string formatTypeMismatchMsg(const std::string& actual_type) const {
  std::string inferred_type_hint;
  if (is_inferred_type()) {
    inferred_type_hint = c10::str(
        "Inferred '",
        name(),
        "' to be of type 'Tensor' ",
        "because it was not annotated with an explicit type.\n");
  }
  return c10::str(
      "Expected a value of type '",
      type()->repr_str(),
      "' for argument '",
      name(),
      "' but instead found type '",
      actual_type,
      "'.\n",
      inferred_type_hint);
}

// 使用新的类型克隆参数对象并返回
Argument cloneWithType(const TypePtr& new_type) const {
  // 返回一个 Argument 对象，其中包括名称、新类型、N 值、默认值、是否只能作为关键字参数、别名信息（如果有）
  return Argument(
      name_,
      new_type,
      N_,
      default_value_,
      kwarg_only_,
      alias_info_ ? std::optional<AliasInfo>(*alias_info_) : c10::nullopt);
}

// 检查该 Argument 是否向后兼容于旧版本。兼容条件包括：
//   1) 两个参数完全相等
//   2) 当前参数的类型应为旧参数类型的子类型
//   3) 如果旧参数有默认值，则当前参数必须提供相同的默认值
bool isBackwardCompatibleWith(
    const Argument& old,
    std::ostream* why_not=nullptr) const;

// 检查该 Argument 是否向前兼容于旧版本。兼容条件包括：
//   1) 两个参数完全相等
//   2) 当前参数的类型应为旧参数类型的子类型
//   3) 如果旧参数有默认值，则当前参数必须提供相同的默认值
bool isForwardCompatibleWith(
    const Argument& old,
    std::ostream* why_not = nullptr) const;

private:
std::string name_;        // 参数名称
TypePtr type_;             // 参数类型
TypePtr real_type_;        // 实际类型，例如 ScalarType
std::optional<int32_t> N_; // 对于列表类型，列表的静态已知长度
std::optional<IValue> default_value_; // 默认值（如果有）
std::unique_ptr<AliasInfo> alias_info_; // 别名信息的唯一指针
bool kwarg_only_;          // 是否仅限关键字参数
bool is_out_;              // 参数是否是 schema 的输出变量
};

// 重载运算符 ==，用于比较两个 Argument 对象是否相等
inline bool operator==(const Argument& lhs, const Argument& rhs) {
  return lhs.name() == rhs.name()
          && *lhs.type() == *rhs.type()
          && lhs.N() == rhs.N()
          && lhs.default_value() == rhs.default_value()
          && lhs.kwarg_only() == rhs.kwarg_only()
          && (lhs.alias_info() == rhs.alias_info()
              || (lhs.alias_info() != nullptr && rhs.alias_info() != nullptr
                   && *lhs.alias_info() == *rhs.alias_info()));
}

// 重载运算符 !=，用于比较两个 Argument 对象是否不相等
inline bool operator!=(const Argument& lhs, const Argument& rhs) {
  return !(lhs == rhs);
}

// 枚举类型 SchemaArgType，表示结构的参数类型，包括输入和输出
enum struct TORCH_API SchemaArgType { input, output };

/**
 * 结构体 SchemaArgument
 *
 * 用于表示模式的参数或返回值的结构体。
 */
struct TORCH_API SchemaArgument {
  SchemaArgType type;   // 参数类型（输入或输出）
  size_t index;         // 参数的索引
  SchemaArgument(SchemaArgType tpe, size_t idx) : type(tpe), index(idx) {}
  // 重载运算符 ==，用于比较两个 SchemaArgument 对象是否相等
  bool operator==(const SchemaArgument& rhs) const {
    return type == rhs.type && index == rhs.index;
  }
};

// 前置声明，用于声明两个 FunctionSchema 对象之间的 == 运算符重载
bool operator==(const FunctionSchema& lhs, const FunctionSchema& rhs);

// 结构体 FunctionSchema
struct TORCH_API FunctionSchema {
  // 构造函数，用于初始化 FunctionSchema 对象
  FunctionSchema(
      std::string name,
      std::string overload_name,
      std::vector<Argument> arguments,
      std::vector<Argument> returns,
      bool is_vararg = false,
      bool is_varret = false)
      : name_({std::move(name), std::move(overload_name)}),
        arguments_(std::move(arguments)),
        returns_(std::move(returns)),
        is_vararg_(is_vararg),
        is_varret_(is_varret) {
    checkSchema();  // 检查模式的合法性
  }

  // 第二个构造函数，通过符号名称初始化 FunctionSchema 对象
  FunctionSchema(
      Symbol name,
      std::string overload_name,
      std::vector<Argument> arguments,
      std::vector<Argument> returns,
      bool is_vararg = false,
      bool is_varret = false)
      : FunctionSchema(
            name.toQualString(),
            std::move(overload_name),
            std::move(arguments),
            std::move(returns),
            is_vararg,
            is_varret) {
    bool seen_default_arg = false;
    for (const auto& arg : arguments()) {
      if (arg.default_value()) {
        seen_default_arg = true;
      } else {
        // 在不打破 BC 的情况下，历史上我们序列化了带有广播列表但没有默认值的列表
        // 因此，在这里允许列表存在
        if (arg.type()->kind() == ListType::Kind) {
          continue;
        }
        // 断言：非默认位置参数跟随默认参数。参数 arg.name() 在 *this 中
        TORCH_INTERNAL_ASSERT(
            !seen_default_arg || arg.kwarg_only(),
            "Non-default positional argument follows default argument. Parameter ",
            arg.name(),
            " in ",
            *this);
      }
    }
  }

 public:
  
  // 打印函数，用于输出对象信息
  void dump() const;

  // 返回操作符名称对象的引用
  const OperatorName& operator_name() const {
    return name_;
  }
  // 返回名称的引用
  const std::string& name() const {
    return name_.name;
  }
  // 返回重载名称的引用
  const std::string& overload_name() const {
    return name_.overload_name;
  }
  // 返回参数列表的引用
  const std::vector<Argument>& arguments() const {
    return arguments_;
  }
  // 返回返回值列表的引用
  const std::vector<Argument>& returns() const {
    return returns_;
  }
  // 返回是否是可变参数的布尔值
  bool is_vararg() const {
  // 返回 is_vararg_ 变量的值，指示是否为可变参数
  }
  // 返回 is_varret_ 变量的值，指示是否为可变返回值
  bool is_varret() const {
    return is_varret_;
  }
  // 检查当前参数是否与给定参数对象存在别名关系
  bool is_aliasing(const c10::SchemaArgument &argument) const {
    TORCH_INTERNAL_ASSERT(
    // 确保 argument.index 在正确的列表范围内，防止索引无效
    argument.index < getCorrectList(argument.type).size(),
    "Invalid index for schema.");
    // 获取参数对应的别名信息
    const AliasInfo* aliasInfo = getCorrectList(argument.type)[argument.index].alias_info();
    return aliasInfo;  // 返回参数的别名信息
  }
  // 返回是否存在可变参数的标志
  bool is_mutable() const {
    return std::any_of(
        arguments_.cbegin(), arguments_.cend(), [](const Argument& arg) {
          const AliasInfo* aliasInfo = arg.alias_info();
          // 检查别名信息是否存在且具有写权限
          return aliasInfo && aliasInfo->isWrite();
        });
  }
  // 检查给定参数是否可变
  bool is_mutable(const c10::SchemaArgument &argument) const {
    TORCH_INTERNAL_ASSERT(
        argument.index < getCorrectList(argument.type).size(),
        "Invalid index for schema.");
    // 获取参数对应的别名信息
    const AliasInfo* aliasInfo = getCorrectList(argument.type)[argument.index].alias_info();
    return aliasInfo && aliasInfo->isWrite();  // 返回参数是否可写
  }
  // 检查具有指定名称的参数是否可变
  bool is_mutable(c10::string_view name) const {
    // 根据参数名称获取其索引
    std::optional<int> index = argumentIndexWithName(name);
    TORCH_INTERNAL_ASSERT(
        index != c10::nullopt, "Schema has no argument named ", name);
    // 确保参数名称存在于模式中
    ```
  // 返回一个表达是否可变的布尔值，基于参数c10::SchemaArgType::input和static_cast<size_t>(*index)
  return is_mutable({c10::SchemaArgType::input, static_cast<size_t>(*index)});
}

// 返回lhs和rhs是否可能直接别名。
// 不考虑lhs或rhs为可能包含别名元素的容器的情况。
// FunctionSchema::may_contain_alias将包括该功能。
bool may_alias(const SchemaArgument& lhs, const SchemaArgument& rhs) const;

// 返回lhs和rhs是否可能直接别名，或者lhs/rhs是否为可能包含别名元素的容器。
// bidirectional = false仅返回lhs是否可能包含rhs的别名，
// 而bidirectional = true返回双向检查。
bool may_contain_alias(const SchemaArgument& lhs, const SchemaArgument& rhs, bool bidirectional = true) const;

// 返回两个AliasTypeSets是否包含任何相似性，
// 即这两个类型集是否可以别名。
bool canAliasTypeSetsAlias(const std::optional<AliasTypeSet> &lhs, const std::optional<AliasTypeSet> &rhs) const;

// 递归查找AliasTypeSet中包含的所有类型。
std::optional<AliasTypeSet> getAliasTypeSetContainedTypes(const std::optional<AliasTypeSet> &aliasTypeSet) const;

// 类似于alias_analysis.cpp中的mapTypeToAliasTypeSet。
// 用于将类型映射到一种类型，使得所有可能别名的类型都映射到相同的类型。
// 例如，对'Optional[List[int]]'调用此方法与对'List[int]'调用此方法相同。
std::optional<AliasTypeSet> mapTypeToAliasTypeSet(const TypePtr& type) const;

// 根据SchemaArgType返回arguments()或returns()。
// output => returns(), input => arguments()
const std::vector<Argument>& getCorrectList(SchemaArgType type) const;

// 在参数中按名称查找并返回索引值，如果未找到则返回c10::nullopt。
std::optional<int> argumentIndexWithName(c10::string_view name) const {
  for (const auto i : c10::irange(arguments().size())) {
    if(name == arguments()[i].name())
      return i;
  }
  return c10::nullopt;
}

// 使用新的函数名称和重载名称创建并返回一个克隆的FunctionSchema对象。
FunctionSchema cloneWithName(std::string name, std::string overload_name) const {
  return FunctionSchema(
      std::move(name),
      std::move(overload_name),
      arguments(),
      returns(),
      is_vararg(),
      is_varret()
      );
}

// 使用新的参数列表创建并返回一个克隆的FunctionSchema对象。
FunctionSchema cloneWithArguments(std::vector<Argument> new_arguments) const {
  return FunctionSchema(
      name(),
      overload_name(),
      std::move(new_arguments),
      returns(),
      is_vararg(),
      is_varret());
}

// 使用新的返回值列表创建并返回一个克隆的FunctionSchema对象。
FunctionSchema cloneWithReturns(std::vector<Argument> new_returns) const {
  // 返回一个 FunctionSchema 对象，其中包含函数的名称、重载名称、参数列表、移动后的返回值、是否可变参数和可变返回值信息
  return FunctionSchema(
      name(),                        // 调用 name() 函数获取函数名称
      overload_name(),               // 调用 overload_name() 函数获取重载名称
      arguments(),                   // 调用 arguments() 函数获取参数列表
      std::move(new_returns),        // 移动 new_returns 对象作为返回值
      is_vararg(),                   // 调用 is_vararg() 函数获取是否为可变参数
      is_varret());                  // 调用 is_varret() 函数获取是否为可变返回值
}

// 格式化类型不匹配的错误消息
std::string formatTypeMismatchMsg(
    const Argument& expected,                 // 期望的参数类型
    const std::string& actual_type,           // 实际的参数类型
    std::optional<size_t> position = c10::nullopt,  // 参数的位置，可选，默认为空
    std::optional<std::string> value = c10::nullopt) const;  // 参数的值，可选，默认为空

// 使用给定的类型映射函数，克隆当前函数模式，但使用重新映射的类型
FunctionSchema cloneWithRemappedTypes(
    const std::function<TypePtr(TypePtr)> type_map) const;

// 克隆当前函数模式，并可以选择使用真实类型（默认为 true）
FunctionSchema cloneWithRealTypes(bool with_symint=true) const;

// 检查输入是否具有正确的类型，并添加任何缺失的默认值
template <typename T = c10::PlatformType>
void checkAndNormalizeInputs(
    std::vector<IValue>& inputs,                          // 输入参数列表
    const std::unordered_map<std::string, IValue>& kwargs =  // 可选关键字参数，默认为空的无序映射
        std::unordered_map<std::string, IValue>{}) const;

// 在关键字参数中查找错误，并返回错误信息字符串
std::string findErrorInKwargs(const std::vector<std::string>& kwargs) const;

// 检查是否有任何别名信息存在于参数或返回值中
bool hasAnyAliasInfo() const {
  for (const auto& arg : arguments_) {      // 遍历参数列表
    if (arg.alias_info() != nullptr) {      // 检查参数是否有别名信息
      return true;                          // 如果有别名信息，返回 true
    }
  }
  for (const auto& ret : returns_) {        // 遍历返回值列表
    if (ret.alias_info() != nullptr) {      // 检查返回值是否有别名信息
      return true;                          // 如果有别名信息，返回 true
    }
  }
  return false;                             // 如果参数和返回值中均无别名信息，返回 false
}

// 返回是否默认的别名分析类型
bool isDefaultAliasAnalysisKind() const {
  return !alias_kind_;                      // 返回是否未指定别名分析类型
}

// 返回当前别名分析的类型
AliasAnalysisKind aliasAnalysis() const {
  return alias_kind_.value_or(AliasAnalysisKind::CONSERVATIVE);  // 返回当前别名分析类型或默认为 CONSERVATIVE
}

// 设置别名分析的类型
void setAliasAnalysis(AliasAnalysisKind v) {
  alias_kind_ = v;                          // 设置别名分析的类型为给定值 v
}

// 返回名称空间，如果有的话
std::optional<c10::string_view> getNamespace() const {
  return name_.getNamespace();              // 返回名称的命名空间，如果存在的话
}

// 如果尚未设置命名空间，则设置名称空间，并返回是否成功设置
bool setNamespaceIfNotSet(const char* ns) {
  return name_.setNamespaceIfNotSet(ns);    // 设置名称的命名空间为给定值 ns，并返回是否成功设置
}

// 检查是否可以将当前函数模式作为 rhs 的子类型进行替换，并在需要时输出详细信息到 why_not 流
bool isSubtypeOf(const FunctionSchema& rhs, bool as_method, std::ostream* why_not=nullptr) const;
// 定义比较两个 FunctionSchema 对象是否相等的操作符重载函数
inline bool operator==(const FunctionSchema& lhs, const FunctionSchema& rhs) {
  // 检查函数名、重载名、参数、返回值、是否可变参数、是否可变返回值是否全部相等
  return lhs.name() == rhs.name()
     && lhs.overload_name() == rhs.overload_name()
     && lhs.arguments() == rhs.arguments()
     && lhs.returns() == rhs.returns()
     && lhs.is_vararg() == rhs.is_vararg()
     && lhs.is_varret() == rhs.is_varret();
}

// 定义比较两个 FunctionSchema 对象是否不相等的操作符重载函数
inline bool operator!=(const FunctionSchema& lhs, const FunctionSchema& rhs) {
  // 利用之前定义的相等操作符重载函数，返回相反的结果
  return !(lhs == rhs);
}

// 打印 Argument 对象的输出流重载函数，以与 FunctionSchema 解析器兼容的方式打印
// 完整格式：Type(alias)? name=default_value
inline std::ostream& operator<<(std::ostream& out, const Argument& arg) {

  // 调整 ? 的位置。
  // 在 schema 中，我们有 Tensor?(a!) input 和 t(a!)?。
  // 然而，t?(a!) 无法与 schema 解析器兼容。
  // 因此，我们总是使用 Type(alias)? 格式
  // real_type versus fake_type: 为了与 FunctionSchema 解析器兼容，打印具有 MemoryFormat 或 Layout 类型的参数应给出原始 schema 字符串，因此打印 real_type。
  auto type = arg.real_type();
  bool is_opt = type->kind() == OptionalType::Kind;
  auto unopt_type = is_opt ? type->castRaw<OptionalType>()->getElementType() : type;

  // 如果是 ListType 类型，打印其元素类型和大小信息
  if (unopt_type->kind() == ListType::Kind) {
    auto list = unopt_type->cast<c10::ListType>();
    out << list->getElementType()->str();
    if (arg.alias_info() && !arg.alias_info()->containedTypes().empty()){
      out << arg.alias_info()->containedTypes()[0];
    }
    std::string N = "";
    if (arg.N()) {
        N = std::to_string(*arg.N());
    }
    out << "[" << N << "]";
  } else {
    out << unopt_type->str();
  }

  // 如果有 alias_info，并且 beforeSets 不为空，打印 alias_info
  if (arg.alias_info() && !arg.alias_info()->beforeSets().empty()) {
    out << *arg.alias_info();
  }

  // 如果是可选类型，打印 '?'
  if (is_opt) {
    out << "?";
  }

  // 如果参数有名称，打印参数名称
  if (!arg.name().empty()) {
    out << " " << arg.name();
  }

  // 如果有默认值，打印 '=' 和默认值的字符串表示
  if (arg.default_value()) {
    out << "=";
    if ((type->kind() == c10::TypeKind::StringType ||
        unopt_type->kind() == c10::TypeKind::StringType) &&
        arg.default_value().value().isString()) {
      printQuotedString(out, arg.default_value().value().toStringRef());
    } else if (type->kind() == TypeKind::ListType && type->castRaw<ListType>()->getElementType()->kind() == c10::TypeKind::IntType) {
      // 如果参数类型是 Int 数组类型，按照 JIT schema 的规范输出默认值
      // 在 native_functions.yaml 中，带有单个值的 int 数组默认看起来像是
      //   int[2] stride=1
      // 而不是
      //   int[2] stride=[1, 1]
      auto default_val = arg.default_value().value().toIntList();
      // 检查默认值列表的长度是否大于1
      if (default_val.size() > 1) {
        // 检查所有的默认值是否相同
        auto all_defaults_the_same = true;
        for (const auto i : c10::irange(1, default_val.size())) {
          if (default_val[0] != default_val[i]) all_defaults_the_same = false;
        }
        // 如果所有默认值都相同，则输出第一个默认值
        if (all_defaults_the_same) {
          out << default_val[0];
        } else {
          // 否则输出完整的默认值列表
          out << arg.default_value().value();
        }
      } else {
        // 如果默认值列表长度为1或者为空，则直接输出默认值
        out << arg.default_value().value();
      }
    } else {
      // 对于非 Int 数组类型的参数，直接输出默认值
      out << arg.default_value().value();
    }
  }

  // 返回输出流对象
  return out;
} // 结束 c10 命名空间

// 重载流输出操作符，用于将 FunctionSchema 对象输出到流中
inline std::ostream& operator<<(std::ostream& out, const FunctionSchema& schema);

// 将 FunctionSchema 对象转换为字符串形式
inline std::string toString(const FunctionSchema& schema) {
  // 创建一个字符串流对象
  std::ostringstream str;
  // 将 FunctionSchema 对象写入字符串流
  str << schema;
  // 将字符串流内容转换为字符串并返回
  return str.str();
}

} // 结束 c10 命名空间

namespace std {
// 特化 std::hash 模板，用于计算 SchemaArgument 对象的哈希值
template<>
  struct hash<c10::SchemaArgument> {
    size_t operator()(const c10::SchemaArgument& arg) const
    {
      // 计算哈希值，结合参数的索引和类型的哈希值
      return c10::hash_combine(std::hash<size_t>()(arg.index), std::hash<size_t>()(static_cast<std::size_t>(arg.type)));
    }
  };

// 特化 std::hash 模板，用于计算 Argument 对象的哈希值
template<>
  struct hash<c10::Argument> {
    size_t operator()(const c10::Argument& arg) const
    {
      // 计算哈希值，结合参数名称、类型和是否为仅关键字参数的标志
      auto hash = std::hash<std::string>{}(arg.name());
      auto type_hash = std::hash<c10::TypePtr>{}(arg.type());
      auto kwarg_only_hash = std::hash<bool>{}(arg.kwarg_only());
      hash = c10::hash_combine(hash, type_hash);
      hash = c10::hash_combine(hash, kwarg_only_hash);
      // 如果存在默认值，则计算默认值的哈希值
      if (arg.default_value()) {
        auto default_value_hash = c10::hash<c10::IValue>{}(arg.default_value().value());
        hash = c10::hash_combine(hash, default_value_hash);
      }
      // 如果存在 N 值，则计算其哈希值
      if (arg.N()) {
        auto N_hash = std::hash<int64_t>{}(*arg.N());
        hash = c10::hash_combine(hash, N_hash);
      }
      // 如果存在别名信息，则计算别名信息的哈希值
      if (arg.alias_info()) {
        auto alias_info_hash = std::hash<c10::AliasInfo>{}(*arg.alias_info());
        hash = c10::hash_combine(hash, alias_info_hash);
      }
      return hash;
    }
  };

// 特化 std::hash 模板，用于计算 FunctionSchema 对象的哈希值
template<>
  struct hash<c10::FunctionSchema> {
    size_t operator()(const c10::FunctionSchema& schema) const
    {
      // 计算哈希值，结合操作符名称、参数列表、返回值列表、是否可变参数和是否可变返回值的标志
      auto hash = std::hash<c10::OperatorName>{}(schema.operator_name());
      auto args_hash = c10::hash<std::vector<c10::Argument>>{}(schema.arguments());
      auto returns_hash = c10::hash<std::vector<c10::Argument>>{}(schema.returns());
      auto is_vararg_hash = std::hash<bool>{}(schema.is_vararg());
      auto is_varret_hash = std::hash<bool>{}(schema.is_varret());
      hash = c10::hash_combine(hash, args_hash);
      hash = c10::hash_combine(hash, returns_hash);
      hash = c10::hash_combine(hash, is_vararg_hash);
      hash = c10::hash_combine(hash, is_varret_hash);
      return
```