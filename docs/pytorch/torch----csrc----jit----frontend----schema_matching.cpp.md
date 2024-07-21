# `.\pytorch\torch\csrc\jit\frontend\schema_matching.cpp`

```
// 包含 Torch JIT 前端的头文件，用于模式匹配
#include <torch/csrc/jit/frontend/schema_matching.h>

// 包含 ATen 核心的头文件，包括国际化字符串和 JIT 类型
#include <ATen/core/interned_strings.h>
#include <ATen/core/jit_type.h>

// 包含 C10 异常处理和可选类型的头文件
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>

// 包含 C10 的范围工具头文件
#include <c10/util/irange.h>

// 包含 Caffe2 序列化版本相关的头文件
#include <caffe2/serialize/versions.h>

// 包含 Torch JIT 前端的内置函数头文件
#include <torch/csrc/jit/frontend/builtin_functions.h>

// 包含 Torch JIT 前端的错误报告相关头文件
#include <torch/csrc/jit/frontend/error_report.h>

// 包含 Torch JIT 前端的函数模式解析头文件
#include <torch/csrc/jit/frontend/function_schema_parser.h>

// 包含 Torch JIT IR 相关的头文件
#include <torch/csrc/jit/ir/ir.h>

// 包含 Torch JIT 运算符升级相关的实用函数头文件
#include <torch/csrc/jit/operator_upgraders/utils.h>

// 包含 Torch JIT 运算符升级相关的版本映射头文件
#include <torch/csrc/jit/operator_upgraders/version_map.h>

// 包含 Torch JIT 运行时操作符相关的头文件
#include <torch/csrc/jit/runtime/operator.h>

namespace torch::jit {

// 定义一个内联函数，用于递归解包 Optional 类型，直到找到非 Optional 类型
static inline TypePtr unwrapOptional(TypePtr opt_type) {
  if (auto dyn = opt_type->castRaw<c10::DynamicType>()) {
    return unwrapOptional(dyn->fallback());
  }
  if (auto unwrap_list_type = opt_type->cast<OptionalType>()) {
    return unwrap_list_type->getElementType();
  }
  return opt_type;
}

// 定义一个内联函数，用于判断值是否以 Int 或 Float 类型作为 List 使用
static inline bool isIntOrFloatUsedAsList(
    const Value* value,
    const Argument& arg) {
  // 获取值的类型
  const auto& v_type = value->type();
  // 如果值的类型不是 Float 或 Int，直接返回 false
  if (v_type != FloatType::get() && v_type != IntType::get())
    return false;
  // 解包参数的类型，如果是 Optional 类型，获取其元素类型
  auto arg_type = unwrapOptional(arg.type());
  // 尝试将参数类型转换为 ListType
  auto list_type = arg_type->cast<ListType>();
  // 如果成功将参数类型转换为 ListType，并且 ListType 的元素类型与值的类型相同，并且参数的 N() 返回 true，则返回 true
  return list_type && list_type->getElementType() == v_type && arg.N();
}

/// 返回 true 如果 `type` 是一个 Tuple，其中所有元素具有相同类型，或者是 `list_type_` 的子类型。
bool convertibleToList(const TypePtr& type, const TypePtr& list_type_) {
  auto list_type = list_type_->castRaw<ListType>();
  // 如果 `list_type_` 不是 ListType 类型，则返回 false
  if (!list_type) {
    return false;
  }
  // 如果 `type` 是 `list_type_` 的子类型，则返回 true
  if (type->isSubtypeOf(*list_type_)) {
    return true;
  }
  // 如果 `type` 是 TupleType 类型
  if (auto tuple = type->castRaw<TupleType>()) {
    // 检查 Tuple 中所有元素是否都是 `list_type->getElementType()` 的子类型
    return std::all_of(
        tuple->elements().begin(),
        tuple->elements().end(),
        [&](const TypePtr& t) {
          // TODO: 如果需要，解析 VarType
          return t->isSubtypeOf(*list_type->getElementType());
        });
  }
  return false;
}

// 尝试将值转换为指定类型 `concrete_type`，并在可能的情况下进行隐式转换。如果 `return_value->isSubtypeOf(concrete_type)` 成功，则返回转换后的值。
Value* tryConvertToType(
    const SourceRange& loc,
    Graph& graph,
    const TypePtr& concrete_type,
    Value* value,
    bool allow_conversions) {
  // 将转换到 Optional[T] 视为转换到 T
  if (OptionalTypePtr op = concrete_type->cast<OptionalType>()) {
    if (value->type()->kind() != OptionalType::Kind &&
        !value->type()->isSubtypeOf(*NoneType::get())) {
      return tryConvertToType(
          loc, graph, op->getElementType(), value, allow_conversions);
    }
  }

  // 允许临时未注释的空列表文字 `[]` 匹配到任意列表类型
  if (value->node()->kind() == prim::EmptyListLiteral &&
      concrete_type->cast<ListType>()) {
    // 将 graph 中的节点插入到 graph 中
    value = graph
                .insertNode(graph.createList(
                    concrete_type->cast<ListType>()->getElementType(), {}))
                ->output();
  }

  // 如果 value 是元组类型
  if (auto value_tuple = value->type()->cast<TupleType>()) {
    // 允许将同类型的元组隐式转换为相应类型的列表
    if (convertibleToList(value->type(), unwrapOptional(concrete_type))) {
      // 创建元组解包
      auto unpacked = createTupleUnpack(value);
      // 获取列表元素类型
      auto elem_type =
          unwrapOptional(concrete_type)->expectRef<ListType>().getElementType();
      // 将解包后的元组插入到 graph 中并获取输出值
      value = graph.insertNode(graph.createList(elem_type, unpacked))->output();
    }

    // 对元组进行隐式类型转换
    if (auto concrete_tuple = concrete_type->cast<TupleType>()) {
      // 如果元组类型不是目标类型的子类型，并且元素数量相同
      if (!value_tuple->isSubtypeOf(*concrete_tuple) &&
          concrete_tuple->elements().size() == value_tuple->elements().size()) {
        // 创建元组解包
        auto unpacked = createTupleUnpack(value);
        std::vector<Value*> converted;
        // 逐个元素进行类型转换
        for (size_t i = 0; i < concrete_tuple->elements().size(); ++i) {
          converted.emplace_back(tryConvertToType(
              loc,
              graph,
              concrete_tuple->elements().at(i),
              unpacked.at(i),
              allow_conversions));
        }
        // 将转换后的元素插入到 graph 中并获取输出值
        value = graph.insertNode(graph.createTuple(converted))->output();
      }
    }
  }

  // 隐式类型转换
  if (allow_conversions) {
    // 判断 value 是否是 TensorType 或者 NumberType
    bool value_isa_tensor = value->type()->isSubtypeOf(*TensorType::get());
    bool value_equals_number = *value->type() == *NumberType::get();
    bool concrete_float = *concrete_type == *FloatType::get();
    bool concrete_complex = *concrete_type == *ComplexType::get();
    bool concrete_int = *concrete_type == *IntType::get();
    bool concrete_number = *concrete_type == *NumberType::get();
    // 如果 value 是 TensorType
    if (value_isa_tensor) {
      // 根据 concrete_type 进行隐式转换
      if (concrete_float) {
        value = graph.insert(aten::FloatImplicit, {value}, {}, loc);
      } else if (concrete_complex) {
        value = graph.insert(aten::ComplexImplicit, {value}, {}, loc);
      } else if (concrete_int) {
        value = graph.insert(aten::IntImplicit, {value}, {}, loc);
      } else if (concrete_number) {
        value = graph.insert(aten::ScalarImplicit, {value}, {}, loc);
      }
    } else if (value_equals_number) {
      // 如果 value 是 NumberType
      // 根据 concrete_type 进行隐式转换
      if (concrete_float) {
        value = graph.insert(aten::Float, {value}, {}, loc);
      } else if (concrete_complex) {
        value = graph.insert(aten::Complex, {value}, {}, loc);
      } else if (concrete_int) {
        value = graph.insert(aten::Int, {value}, {}, loc);
      }
    } else if (*value->type() == *BoolType::get()) {
      // 如果 value 是 BoolType
      // 根据 concrete_type 进行隐式转换
      if (concrete_float) {
        value = graph.insert(aten::Float, {value}, {}, loc);
      } else if (concrete_int) {
        value = graph.insert(aten::Int, {value}, {}, loc);
      } else if (concrete_number) {
        value = graph.insert(aten::Int, {value}, {}, loc);
      }
    }
    // 如果 value 的类型是 String 类型的子类型，并且 concrete_type 是 DeviceObjType 的子类型
    if (value->type()->isSubtypeOf(*StringType::get()) &&
        concrete_type->isSubtypeOf(*DeviceObjType::get())) {
      // 在计算图中插入一个设备相关操作 aten::device，操作的输入为 value，无额外参数，位置信息为 loc
      return graph.insert(aten::device, {value}, {}, loc);
    }
  }

  // 如果上述条件不满足，则直接返回 value
  return value;
// Checks if `named_value` can be used as a value for `arg`.
// If `arg` is a VarType, it will be added to the type_env through `matchTypeVariables`
// as the corresponding actual type.
// If `allow_conversions` is true, implicit conversions to the `arg` type may be performed
// through `tryConvertToType`.
static Value* tryMatchArgument(
    const Argument& arg,                  // The argument to match against
    Graph& graph,                         // The graph in which operations are performed
    const SourceRange& loc,               // Source location of the argument
    const NamedValue& named_value,        // Named value being matched
    std::ostream* failure_messages,       // Stream for error messages
    const std::function<std::ostream&()>& err,  // Function to get error stream
    bool allow_conversions,               // Flag indicating if conversions are allowed
    TypeEnv& type_env                     // Type environment for type matching
) {
    // Retrieve the value associated with the named value in the graph
    Value* value = named_value.value(graph);

    // Check if the value is an integer or float being used as a list element for fixed size arrays
    if (isIntOrFloatUsedAsList(value, arg)) {
        // Create a vector of repeated values matching the size of the fixed array
        std::vector<Value*> repeated(*arg.N(), value);
        // Insert a node into the graph representing the list of repeated values
        value = graph.insertNode(graph.createList(value->type(), repeated))->output();
    }

    // Resolve VarType variables in the argument type against the actual value type
    const MatchTypeReturn matched = matchTypeVariables(arg.type(), value->type(), type_env);
    if (!matched.success()) {
        // If matching fails, output an error message to the failure_messages stream
        if (failure_messages) {
            err() << "Could not match type " << value->type()->repr_str() << " to "
                  << arg.type()->repr_str() << " in argument '" << arg.name()
                  << "': " << matched.reason() << ".\n";
        }
        return nullptr; // Return nullptr indicating failure
    }

    // Try to evaluate type variables in the argument type using the type environment
    const auto concrete_type = tryEvalTypeVariables(arg.type(), type_env);
    if (!concrete_type) {
        // If type evaluation fails, output an error message to the failure_messages stream
        if (failure_messages) {
            err() << "Type variables in type " << arg.type()->repr_str()
                  << " could not be inferred from actual type "
                  << value->type()->repr_str();
        }
        return nullptr; // Return nullptr indicating failure
    }

    // Attempt to convert the value to the concrete type inferred
    value = tryConvertToType(loc, graph, concrete_type, value, allow_conversions);

    // Check if the value can be matched to the argument type through any implicit conversions
    std::stringstream ss;
    if (!value->type()->isSubtypeOfExt(
            *concrete_type, /*why_not=*/(failure_messages) ? &ss : nullptr)) {
    // 如果存在失败消息
    if (failure_messages) {
      // 获取错误流，并输出格式类型不匹配的消息
      auto& ostream = err()
          << arg.formatTypeMismatchMsg(value->type()->repr_str());

      // 如果值的类型可以转换为张量类型
      if (auto pt = value->type()->cast<TensorType>()) {
        // 如果类型是推断的
        if (pt->isInferredType()) {
          // 构建推断类型的提示信息字符串
          std::string inferred_type_hint;
          inferred_type_hint = c10::str(
              "Inferred the value for argument '",
              arg.name(),
              "' to be of type 'Tensor' ",
              "because it was not annotated with an explicit type.\n");
          // 将推断类型的提示信息添加到错误流
          ostream << inferred_type_hint;
        }
      }

      // 如果值的类型可以转换为列表类型
      if (auto v = value->type()->cast<ListType>()) {
        // 如果列表元素类型是张量类型的子类型
        if (v->getElementType()->isSubtypeOf(*TensorType::get())) {
          // 输出空列表默认为 List[Tensor] 的信息
          ostream << "Empty lists default to List[Tensor]. Add a variable "
                     "annotation to the assignment to create an empty list "
                     "of another type (torch.jit.annotate(List[T, []]) where T "
                     "is the type of elements in the list for Python 2)\n";
        }
      }

      // 将其余的字符串流内容添加到错误流
      ostream << ss.str();
    }

    // 返回空指针
    return nullptr;
  }
  // 返回值对象
  return value;
}

// 在 kwargs 数组中查找指定名称的输入参数的索引，返回 std::optional<size_t>
// 如果找到匹配的参数，则返回其索引；否则返回 std::nullopt
std::optional<size_t> findInputWithName(
    const std::string& name,                    // 要查找的参数名
    at::ArrayRef<NamedValue> kwargs,             // 参数列表
    bool is_aten) {                              // 是否为 aten 函数
  for (const auto i : c10::irange(kwargs.size())) {
    // 如果是 aten 函数，并且要查找的名称为 "self"，则将其重命名为 "input"
    if (is_aten && name == "self" && kwargs[i].name() == "input") {
      return i;
    }
    // 查找与指定名称匹配的参数，并返回其索引
    if (kwargs[i].name() == name) {
      return i;
    }
  }
  // 如果未找到匹配的参数，则返回 std::nullopt
  return c10::nullopt;
}

/// 尝试创建一个列表，其中每个值的类型可以匹配到 elem_type 类型的参数
/// 如果 varargs 中的某个类型与 elem_type 不匹配，则返回 nullptr。
/// 用于从 varargs 创建列表，以便使类似 torch.zeros(1, 2, 3) 的调用匹配到 aten::zeros(int[])。
static Value* tryCreateList(
    const TypePtr& elem_type,                    // 列表元素的类型
    Graph& graph,                               // 当前的图
    const SourceRange& loc,                     // 源代码位置
    at::ArrayRef<NamedValue> varargs,           // 参数列表
    std::ostream* failure_messages,             // 失败消息的输出流
    const std::function<std::ostream&()>& err,  // 出错时的处理函数
    bool convert_tensor_to_num,                 // 是否将张量转换为数字
    TypeEnv& type_env) {                        // 类型环境
  Argument elem_arg("<varargs>", elem_type);
  std::vector<Value*> list_elements;
  for (const auto& named_value : varargs) {
    // 尝试将 named_value 转换为 elem_type 类型的参数
    Value* matched_value = tryMatchArgument(
        /*arg=*/elem_arg,
        graph,
        loc,
        named_value,
        failure_messages,
        err,
        /*allow_conversions=*/convert_tensor_to_num,
        type_env);
    if (!matched_value) {
      return nullptr;
    }
    list_elements.push_back(matched_value);
  }

  // 在图中插入一个节点来创建列表，并返回其输出值
  return graph.insertNode(graph.createList(elem_type, list_elements))->output();
}

// 检查是否可以将所有剩余的非关键字参数转换为列表。
// 这使得类似于 zeros(IntArrayRef sizes) 的调用可以与 zeros(1, 2) 或 zeros(1) 匹配。
static bool varargsCanBeUsedAsList(
    const FunctionSchema& schema,   // 函数的模式
    size_t arg_index,               // 参数的索引
    const Argument& arg) {          // 参数本身
  // 参数必须是不是关键字参数列表中的最后一个
  bool is_last_argument = arg_index + 1 == schema.arguments().size() ||
      schema.arguments()[arg_index + 1].kwarg_only();

  auto arg_type = arg.type();
  if (auto dyn = arg_type->castRaw<c10::DynamicType>()) {
    arg_type = dyn->fallback();
  }

  // 形式必须是列表
  bool argument_is_list = arg_type->kind() == TypeKind::ListType;

  // 匹配类型变量列表 nyi
  bool typevar_list = argument_is_list &&
      arg_type->castRaw<ListType>()->getElementType()->cast<VarType>();

  // 它不能是广播列表，如 int[3]，否则单个 int 将是有效的输入
  bool arg_is_broadcasting_list = bool(arg.N());

  return is_last_argument && argument_is_list && !arg_is_broadcasting_list &&
      !typevar_list;
}
bool isBlockListedSchema(const FunctionSchema& schema) {
  // 检查函数是否在阻止列表中
  // 这是一个对 https://github.com/pytorch/pytorch/issues/47964 的临时解决方案。
  // 目前 JIT 不能区分 ScalarType 和 int，因此无法区分 x.view(1) 和 x.view(torch.int8)。
  // 因此我们需要在这里硬编码 aten::view.dtype 来阻止这种重载。
  // 当 JIT 完全支持 ScalarType 作为自己的类型时，应该移除此阻止列表。
  if (schema.name() == "aten::view" && schema.overload_name() == "dtype") {
    return true;
  }
  // TorchScript 不支持关键字参数，所以该操作与 aten.max.others 冲突，
  // 因为它们都有两个 Tensor 输入。我们不希望用户在 TorchScript 中使用此操作，因此跳过它。
  // 当 TorchScript 完全支持关键字参数时，应该重新评估此冲突的解决方案。
  if (schema.name() == "aten::max" && schema.overload_name() == "unary_out") {
    return true;
  }
  if (schema.name() == "aten::min" && schema.overload_name() == "unary_out") {
    return true;
  }
  return false;
}

static std::optional<MatchedSchema> tryMatchSchema(
    const FunctionSchema& schema,
    const SourceRange& loc,
    Graph& graph,
    at::ArrayRef<NamedValue> args,
    at::ArrayRef<NamedValue> kwargs,
    std::optional<NamedValue> self,
    std::ostream* failure_messages,
    bool allow_conversions) {
  // 如果函数在阻止列表中，返回空匹配模式
  if (isBlockListedSchema(schema)) {
    return c10::nullopt;
  }

  // 错误消息的输出流，用于记录匹配失败时的详细信息
  auto err = [&]() -> std::ostream& {
    *failure_messages << "\n" << schema << ":\n";
    return *failure_messages;
  };

  // 类型环境，用于跟踪变量类型
  TypeEnv type_env;

  // 位置参数的列表
  std::vector<Value*> positional_inputs;

  // 标记关键字参数是否已被使用的列表
  std::vector<bool> used_kwarg(kwargs.size(), false);

  // 检查函数是否来自 aten 命名空间
  auto schema_namespace = schema.operator_name().getNamespace();
  bool is_aten = false;
  if (schema_namespace.has_value()) {
    if (schema_namespace.value() == "aten") {
      is_aten = true;
    }
  }

  // 记录已使用的位置参数的数量
  size_t used_args = 0;

  // 遍历函数的每个参数
  for (const auto schema_i : c10::irange(schema.arguments().size())) {
    // 获取参数的定义
    const auto& arg = schema.arguments()[schema_i];

    // 实际传入的参数值
    std::optional<NamedValue> actual_named_value;

    // 如果参数名为 "self" 并且有自身参数传入，则将其视为实际传入的值
    if (arg.name() == "self" && self) {
      actual_named_value = self;
      self = c10::nullopt;
      // self 参数只能出现一次，因此标记为已使用
      // 如果再次遇到 self，则会引发错误
    } else {
      // 查找在位置参数列表中的对应参数
      // 这里可能需要进一步处理关键字参数和位置参数的情况
      // 这部分代码可能会对如何处理参数的顺序和类型有所依赖
    }
  } else if (!arg.kwarg_only() && used_args < args.size()) {
    // 如果参数不仅限于关键字参数且未使用的参数小于参数列表长度
    // 尝试将所有剩余的非关键字参数（used_args）转换为列表
    // 允许 zeros(IntArrayRef sizes) 与 zeros(1, 2) 或 zeros(1) 一起使用
    if (allow_conversions && varargsCanBeUsedAsList(schema, schema_i, arg)) {
      auto value = args[used_args].value(graph);
      const auto& actual_type = value->type();
      // 实际值不能已经是列表
      if (actual_type->kind() != TypeKind::ListType &&
          !convertibleToList(actual_type, unwrapOptional(arg.type()))) {
        auto formal_type = unwrapOptional(arg.type())
                               ->expectRef<ListType>()
                               .getElementType();
        // 尝试创建列表
        Value* list = tryCreateList(
            formal_type,
            graph,
            loc,
            at::ArrayRef<NamedValue>(args).slice(used_args),
            failure_messages,
            err,
            allow_conversions,
            type_env);
        if (!list) {
          return c10::nullopt;
        }
        used_args = args.size();  // 使用了所有参数
        positional_inputs.push_back(list);  // 将列表添加到位置参数列表
        continue;
      }
    }

    // 设置 actual_named_value 为参数值并标记该参数位置为已使用
    actual_named_value = args[used_args];
    used_args++;
  } else if (
      auto kwarg_idx = findInputWithName(arg.name(), kwargs, is_aten)) {
    // 在关键字参数中查找具有指定名称的输入索引
    const NamedValue& nv = kwargs[*kwarg_idx];
    if (used_kwarg[*kwarg_idx]) {
      if (failure_messages) {
        err() << "Argument " << nv.name()
              << " specified twice in schema, submit a bug report!\n";
      }
      return c10::nullopt;
    }
    used_kwarg[*kwarg_idx] = true;  // 标记关键字参数为已使用
    actual_named_value = nv;  // 设置 actual_named_value 为关键字参数值
  } else if (arg.default_value()) {
    // 参数具有默认值且未提供任何值，因此使用默认值
    actual_named_value = NamedValue(*arg.default_value());
  } else {
    if (failure_messages) {
      err() << "Argument " << schema.arguments()[schema_i].name()
            << " not provided.\n";
    }
    return c10::nullopt;
  }

  // 确保找到的 actual_named_value 与参数类型匹配
  Value* positional = tryMatchArgument(
      arg,
      graph,
      loc,
      *actual_named_value,
      failure_messages,
      err,
      allow_conversions,
      type_env);
  if (!positional) {
    return c10::nullopt;
  }
  positional_inputs.push_back(positional);  // 将匹配的位置参数添加到列表中
}

// 检查是否有未使用的 self 参数
if (self != c10::nullopt) {
  if (failure_messages) {
    err() << "Provided self argument not used in schema.\n";
  }
  return c10::nullopt;
}

if (schema.is_vararg()) {
  // 如果模式是可变参数的情况下
  for (; used_args < args.size(); ++used_args) {
    positional_inputs.push_back(args[used_args].value(graph));
  }
}

// 检查是否有未使用的位置参数
if (used_args < args.size()) {
  // 如果存在失败消息
  if (failure_messages) {
    // 输出错误消息：预期最多 used_args 个参数，但找到 args.size() 个位置参数。
    err() << "Expected at most " << used_args << " arguments "
          << "but found " << args.size() << " positional arguments.\n";
  }
  // 返回空的optional类型
  return c10::nullopt;
}

// 检查未使用的关键字参数
for (const auto i : c10::irange(kwargs.size())) {
  const auto& nv = kwargs[i];
  // 如果未使用关键字参数
  if (!used_kwarg[i]) {
    // 如果存在失败消息
    if (failure_messages) {
      // 如果关键字参数 nv.name() 在schema中不存在
      if (!schema.argumentIndexWithName(nv.name())) {
        // 输出错误消息：关键字参数 nv.name() 未知。
        err() << "Keyword argument " << nv.name() << " unknown.\n";
      } else {
        // 输出错误消息：关键字参数 nv.name() 指定了两次。
        err() << "Keyword argument " << nv.name() << " specified twice.\n";
      }
    }
    // 返回空的optional类型
    return c10::nullopt;
  }
}

// 获取schema的返回值
const auto& returns = schema.returns();
// 尝试解析返回值类型中的类型变量
auto return_types = fmap(returns, [&](const Argument& r) {
  TypePtr result = tryEvalTypeVariables(r.type(), type_env);
  // 断言：结果不能为空
  TORCH_INTERNAL_ASSERT(
      result, r.type()->repr_str(), " has unbound type variables.");
  return result;
});

// Codegen不支持带有未定义字段名的命名元组的返回。
// 因此，所有返回值要么都有字段名，要么都没有。
bool return_has_field_names =
    std::all_of(returns.begin(), returns.end(), [&](const Argument& r) {
      return r.name().length() > 0;
    });
c10::OptNameList return_field_names = c10::nullopt;
// 如果所有返回值都有字段名
if (return_has_field_names) {
  // 获取所有返回值的字段名列表
  return_field_names =
      fmap(returns, [&](const Argument& r) { return r.name(); });
}

// 构造用于更容易查找的schema的完整名称
auto schema_name = getFullSchemaName(schema);

// 返回MatchedSchema对象，包含移动后的位置输入、返回类型、返回字段名列表和schema名称
return MatchedSchema{
    std::move(positional_inputs),
    std::move(return_types),
    std::move(return_field_names),
    schema_name};
}

// 匹配给定函数模式的函数，返回匹配的模式对象或者抛出错误
MatchedSchema matchSchema(
    const ::c10::FunctionSchema& schema,  // 给定的函数模式
    const SourceRange& loc,               // 源代码位置范围
    Graph& graph,                         // 图对象
    at::ArrayRef<NamedValue> args,        // 参数列表
    at::ArrayRef<NamedValue> kwargs,      // 关键字参数列表
    const std::optional<NamedValue>& self // 可选的self参数
) {
  std::stringstream failure_messages;    // 用于收集错误信息的字符串流
  if (auto result = tryMatchSchema(
          schema,
          loc,
          graph,
          args,
          kwargs,
          self,
          &failure_messages,
          /*allow_conversions=*/true)) {  // 尝试匹配函数模式，允许类型转换
    return *result;  // 如果匹配成功则返回匹配结果
  }
  throw ErrorReport(loc) << failure_messages.str();  // 如果匹配失败则抛出错误报告
}

// 在字符串每行前添加前缀的函数
static std::string prefixLine(
    const std::string& str,   // 输入的字符串
    const std::string& prefix // 要添加的前缀
) {
  std::stringstream ss;    // 字符串流，用于构建结果字符串
  bool was_newline = true;  // 标记上一个字符是否为换行符
  for (auto c : str) {     // 遍历输入字符串的每个字符
    if (was_newline)
      ss << prefix;        // 如果上一个字符是换行符，则添加前缀
    ss.put(c);             // 添加当前字符到结果字符串流
    was_newline = c == '\n';  // 更新换行符标记
  }
  return ss.str();          // 返回添加前缀后的字符串
}

// 匹配多个函数模式并返回匹配结果的函数
std::pair<size_t, MatchedSchema> matchSchemas(
    const std::vector<const FunctionSchema*>& schemas,  // 函数模式列表
    const SourceRange& loc,        // 源代码位置范围
    Graph& graph,                  // 图对象
    at::ArrayRef<NamedValue> args, // 参数列表
    at::ArrayRef<NamedValue> kwargs,  // 关键字参数列表
    const std::optional<NamedValue>& self,  // 可选的self参数
    bool render_errors              // 是否渲染错误信息
) {
  TORCH_INTERNAL_ASSERT(!schemas.empty());  // 断言函数模式列表不为空
  if (schemas.size() == 1) {
    return std::make_pair(
        0, matchSchema(*schemas.at(0), loc, graph, args, kwargs, self));  // 若只有一个模式，则直接匹配并返回结果
  }
  std::stringstream failure_messages;   // 用于收集错误信息的字符串流
  for (bool allow_conversions : {false, true}) {  // 遍历是否允许类型转换的选项
    failure_messages.str("");   // 清空之前的错误信息
    for (const auto i : c10::irange(schemas.size())) {  // 遍历所有函数模式
      const auto matched_schema = tryMatchSchema(
          *schemas[i],
          loc,
          graph,
          args,
          kwargs,
          self,
          render_errors ? &failure_messages : nullptr,
          allow_conversions);   // 尝试匹配函数模式
      if (matched_schema) {
        return std::make_pair(i, *matched_schema);  // 如果匹配成功则返回匹配结果
      }
    }
  }
  // 如果不渲染错误信息，则再次调用函数以渲染错误信息并返回结果
  if (!render_errors) {
    return matchSchemas(
        schemas, loc, graph, args, kwargs, self, /*render_errors=*/true);
  }

  // 如果所有尝试均未成功匹配，则抛出错误报告
  throw ErrorReport(loc) << "Arguments for call are not valid.\n"
                         << "The following variants are available:\n"
                         << prefixLine(failure_messages.str(), "  ")
                         << "\nThe original call is";
  throw ErrorReport(loc) << failure_messages.str();  // 抛出错误报告，显示错误信息
}

// 按照 Python 规则打包函数输出结果，如果只有一个值则返回 SimpleValue，否则打包成 Tuple
static Value* packOutputs(
    Graph& g,                       // 图对象
    at::ArrayRef<Value*> values,    // 值列表
    c10::OptNameList field_names    // 字段名称列表
) {
  if (values.size() == 1) {
    return values[0];
  }
  std::shared_ptr<FunctionSchema> schema;
  TupleTypePtr named_tuple = nullptr;
  if (field_names) {
    // 如果提供了字段名，获取所有值的类型并创建命名元组类型
    auto types = fmap(values, [](Value* v) { return v->type(); });
    named_tuple =
        TupleType::createNamed(c10::nullopt, field_names.value(), types);
  }
  // 将值组成元组，并将命名元组与图中插入的节点的输出绑定
  return g.insertNode(g.createTuple(values, named_tuple))->output();


这些注释解释了给定代码的每一行的作用和意图。
}

// 给定操作符模式和符号成功匹配后，生成一个节点
// 包括适当的输入和输出。
static Value* emitBuiltinNode(
    const MatchedSchema& matched_schema,
    const SourceRange& loc,
    Graph& graph,
    Symbol name,
    std::optional<size_t> version) {
  // 在图中插入一个节点，使用匹配的模式创建操作符
  auto n = graph.insertNode(graph.create(name, matched_schema.inputs, 0))
               ->setSourceRange(loc);

  // 为每个返回类型添加输出端口
  for (auto& ret : matched_schema.return_types) {
    n->addOutput()->setType(ret);
  }

  // 如果没有提供版本号，或者操作符与版本号匹配
  if (!version.has_value() ||
      isOpSymbolCurrent(matched_schema.schema_name, version.value())) {
    // 获取操作符的实现
    n->getOperation();
  } else {
    // 设置历史模式名称，以便与服务器版本同步
    n->setHistoricSchemaName(matched_schema.schema_name);
  }

  // 打包输出，返回操作符的输出
  return packOutputs(graph, n->outputs(), matched_schema.return_field_names);
}

// 获取完整的模式名称，包括重载名称（如果存在）
std::string getFullSchemaName(const ::c10::FunctionSchema& schema) {
  if (!schema.overload_name().empty()) {
    return schema.operator_name().name + "." + schema.overload_name();
  }
  return schema.operator_name().name;
}

// 搜索与提供的符号名称和输入类型匹配的操作符
// 如果找到，则在图中生成该操作符的节点。
Value* emitBuiltinCall(
    const SourceRange& loc,
    Graph& graph,
    Symbol name,
    at::ArrayRef<NamedValue> args,
    at::ArrayRef<NamedValue> kwargs,
    const std::optional<NamedValue>& self) {
  // 获取所有与给定名称匹配的操作符变体和内建函数
  const auto& variants = getAllOperatorsFor(name);
  const auto& builtin_functions = getAllBuiltinFunctionsFor(name);

  // 首先设置图的版本号
  auto graph_version = graph.get_op_version();

  // 用于存储失败消息的流
  std::stringstream failure_messages;
  // 存储函数模式的指针
  std::vector<const FunctionSchema*> schemas;
  // 我们稍后将它们附加到 schemas，因为 parseSchema 返回右值，无法转换为 const 指针。
  std::vector<FunctionSchema> upgrader_schemas;
  schemas.reserve(variants.size());

  // 遍历所有操作符变体
  for (const std::shared_ptr<Operator>& op : variants) {
    bool found_upgrader = false;
    auto op_name = getFullSchemaName(op->schema());

    // 如果图的版本号已提供
    if (graph_version.has_value()) {
      // 查找操作符的版本映射
      auto version_entry = get_operator_version_map().find(op_name);
      if (version_entry != get_operator_version_map().end()) {
        // 查找旧模式并将其解析为函数模式
        auto old_schema_entry = findUpgrader(version_entry->second, graph_version.value());
        if (old_schema_entry.has_value()) {
          FunctionSchema old_schema = parseSchema(old_schema_entry.value().old_schema);
          upgrader_schemas.push_back(old_schema);
          found_upgrader = true;
        } else {
          // 如果没有找到有效的升级器条目，但操作符不是基于当前的更新条目
          if (!isOpCurrentBasedOnUpgraderEntries(version_entry->second, graph_version.value())) {
            TORCH_INTERNAL_ASSERT(false, "Valid upgrader must be present");
          }
        }
      }
    }
  // 如果没有找到 upgrader，将 op 的 schema 加入 schemas 列表中
  if (!found_upgrader)
    schemas.push_back(&op->schema());
}

// 可能会看到已经被废弃的历史操作
if (variants.empty()) {
  // 加载可能的历史操作的 schema
  auto oldSchemas =
      loadPossibleHistoricOps(name.toQualString(), graph_version);
  // 预留空间以容纳旧的 schema
  upgrader_schemas.reserve(oldSchemas.size());
  // 将旧的 schema 解析并添加到 upgrader_schemas 列表中
  for (const auto& old_schema_entry : oldSchemas) {
    FunctionSchema old_schema = parseSchema(old_schema_entry);
    upgrader_schemas.emplace_back(old_schema);
  }
}

// TODO (tugsuu): 确保以后优化这部分代码
for (const auto& schema : upgrader_schemas) {
  // 将 upgrader_schemas 中的 schema 加入到 schemas 列表中
  schemas.push_back(&schema);
}

// 针对每个内置函数，确保其已定义并将其 schema 加入到 schemas 列表中
for (const auto method : builtin_functions) {
  method->ensure_defined();
  schemas.push_back(&method->getSchema());
}

// 如果没有找到相同名称的操作符，打印出类似名称的操作符
if (schemas.empty()) {
  // 查找与给定名称相似的操作符符号
  const auto close_symbols = findSimilarOperators(name);
  // 创建错误报告
  auto error = ErrorReport(loc);
  const auto& user_function_name = name.toQualString();
  error << "Unknown builtin op: " << user_function_name << ".\n";
  if (close_symbols.empty()) {
    error
        << "Could not find any similar ops to " << user_function_name
        << ". This op may not exist or may not be currently supported in TorchScript.\n";
  } else {
    error << "Here are some suggestions: \n";
    for (const auto& sym : close_symbols) {
      error << "\t" << sym.toQualString() << "\n";
    }
    error << "\nThe original call is";
  }
  // 抛出错误报告
  throw error;
}

// 匹配 schemas 列表中的 schema 并返回匹配结果
auto matched = matchSchemas(schemas, loc, graph, args, kwargs, self);

// 如果匹配的索引小于 variants 大小加上 upgrader_schemas 大小，生成内置节点
if (matched.first < variants.size() + upgrader_schemas.size()) {
  return emitBuiltinNode(matched.second, loc, graph, name, graph_version);
} else {
  // 否则，内联内置调用，因为它们通常很小且不适合用于调试
  auto& fn = *builtin_functions[matched.first - variants.size()];
  return insertGraph(
             graph, *toGraphFunction(fn).graph(), matched.second.inputs)
      .at(0);
}
}

} // namespace torch::jit
```