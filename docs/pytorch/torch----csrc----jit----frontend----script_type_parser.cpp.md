# `.\pytorch\torch\csrc\jit\frontend\script_type_parser.cpp`

```py
#include <torch/csrc/jit/frontend/script_type_parser.h>  // 引入ScriptTypeParser头文件

#include <ATen/core/type_factory.h>  // 引入ATen核心类型工厂头文件
#include <torch/csrc/jit/frontend/parser.h>  // 引入前端解析器头文件
#include <torch/csrc/jit/ir/ir.h>  // 引入IR图头文件
#include <torch/custom_class.h>  // 引入自定义类头文件

namespace torch::jit {  // 进入torch::jit命名空间
namespace {  // 匿名命名空间，定义局部函数和变量

bool isTorch(const Expr& expr) {  // 定义isTorch函数，检查表达式是否为'torch'
  return expr.kind() == TK_VAR && Var(expr).name().name() == "torch";
}

std::string collectQualname(const Select& select) {  // 定义collectQualname函数，收集限定名称
  Expr base = select.value();
  if (base.kind() == TK_VAR) {  // 如果基础表达式是变量
    return Var(base).name().name() + "." + select.selector().name();
  }
  std::string basename = collectQualname(Select(base));  // 递归调用collectQualname收集基础名称
  return basename + "." + select.selector().name();
}

const std::unordered_map<std::string, c10::TypePtr>& string_to_type_lut() {  // 定义string_to_type_lut函数，返回类型查找表
  return c10::DefaultTypeFactory::basePythonTypes();  // 返回基础Python类型的查找表
}

} // namespace

TypePtr ScriptTypeParser::subscriptToType(  // ScriptTypeParser类的subscriptToType方法实现
    const std::string& typeName,  // typeName参数，表示类型名称
    const Subscript& subscript) const {  // subscript参数，表示下标操作
  if (typeName == "Tuple" || typeName == "tuple") {  // 如果类型名为Tuple或tuple
    if (subscript.subscript_exprs().size() == 1 &&  // 如果下标表达式的大小为1，并且是元组字面量
        subscript.subscript_exprs()[0].kind() == TK_TUPLE_LITERAL) {
      // 处理特殊情况：空元组字面量
      auto tup_literal = TupleLiteral(subscript.subscript_exprs()[0]);
      if (!tup_literal.inputs().empty()) {  // 如果元组字面量不为空
        throw ErrorReport(tup_literal.range())
            << "Tuple literal in Tuple type annotation must not "
            << "have any elements!";  // 抛出错误：元组字面量在元组类型注释中不应包含任何元素
      }
      return TupleType::create({});  // 创建空的元组类型
    }
    std::vector<TypePtr> subscript_expr_types;  // 子脚本表达式类型的向量
    for (auto expr : subscript.subscript_exprs()) {
      subscript_expr_types.emplace_back(parseTypeFromExprImpl(expr));  // 解析每个子脚本表达式的类型
    }
    return TupleType::create(subscript_expr_types);  // 创建元组类型
  } else if (typeName == "List" || typeName == "list") {  // 如果类型名为List或list
    if (subscript.subscript_exprs().size() != 1) {  // 如果子脚本表达式的大小不为1
      throw ErrorReport(subscript)
          << " expected exactly one element type but found "
          << subscript.subscript_exprs().size();  // 抛出错误：期望精确一个元素类型，但找到了其他数量
    }
    auto elem_type =
        parseTypeFromExprImpl(*subscript.subscript_exprs().begin());  // 解析第一个子脚本表达式的类型
    return ListType::create(elem_type);  // 创建列表类型
  } else if (typeName == "Optional") {  // 如果类型名为Optional
    if (subscript.subscript_exprs().size() != 1) {  // 如果子脚本表达式的大小不为1
      throw ErrorReport(subscript)
          << " expected exactly one element type but found "
          << subscript.subscript_exprs().size();  // 抛出错误：期望精确一个元素类型，但找到了其他数量
    }
    auto elem_type =
        parseTypeFromExprImpl(*subscript.subscript_exprs().begin());  // 解析第一个子脚本表达式的类型
    return OptionalType::create(elem_type);  // 创建可选类型
  } else if (typeName == "Union") {  // 如果类型名为Union
    std::vector<TypePtr> subscript_expr_types;  // 子脚本表达式类型的向量
    subscript_expr_types.reserve(subscript.subscript_exprs().size());
    for (auto expr : subscript.subscript_exprs()) {
      subscript_expr_types.emplace_back(parseTypeFromExprImpl(expr));  // 解析每个子脚本表达式的类型
    }
    # 如果 typeName 是 "List" 或 "torch.jit.List"
    if (typeName == "List" || typeName == "torch.jit.List") {
        # 检查 subscript 的表达式数量是否为 1
        if (subscript.subscript_exprs().size() != 1) {
            # 如果不是，则抛出错误报告，指示预期只有一个元素类型，但找到了实际的数量
            throw ErrorReport(subscript)
                << " expected exactly one element type but found "
                << subscript.subscript_exprs().size();
        }
        # 解析并获取第一个 subscript 表达式的类型
        auto elem_type =
            parseTypeFromExprImpl(*subscript.subscript_exprs().begin());
        # 返回一个 ListType 对象，使用 elem_type 创建
        return ListType::create(elem_type);
    } else if (typeName == "Tuple" || typeName == "torch.jit.Tuple") {
        # 如果 typeName 是 "Tuple" 或 "torch.jit.Tuple"
        # 检查 subscript 的表达式数量是否为 1
        if (subscript.subscript_exprs().size() != 1) {
            # 如果不是，则抛出错误报告，指示预期只有一个元素类型，但找到了实际的数量
            throw ErrorReport(subscript)
                << " expected exactly one element type but found "
                << subscript.subscript_exprs().size();
        }
        # 解析并获取第一个 subscript 表达式的类型
        auto elem_type =
            parseTypeFromExprImpl(*subscript.subscript_exprs().begin());
        # 返回一个 TupleType 对象，使用 elem_type 创建
        return TupleType::create(elem_type);
    } else if (typeName == "Future" || typeName == "torch.jit.Future") {
        # 如果 typeName 是 "Future" 或 "torch.jit.Future"
        # 检查 subscript 的表达式数量是否为 1
        if (subscript.subscript_exprs().size() != 1) {
            # 如果不是，则抛出错误报告，指示预期只有一个元素类型，但找到了实际的数量
            throw ErrorReport(subscript)
                << " expected exactly one element type but found "
                << subscript.subscript_exprs().size();
        }
        # 解析并获取第一个 subscript 表达式的类型
        auto elem_type =
            parseTypeFromExprImpl(*subscript.subscript_exprs().begin());
        # 返回一个 FutureType 对象，使用 elem_type 创建
        return FutureType::create(elem_type);
    } else if (typeName == "Await" || typeName == "torch.jit._Await") {
        # 如果 typeName 是 "Await" 或 "torch.jit._Await"
        # 检查 subscript 的表达式数量是否为 1
        if (subscript.subscript_exprs().size() != 1) {
            # 如果不是，则抛出错误报告，指示预期只有一个元素类型，但找到了实际的数量
            throw ErrorReport(subscript)
                << " expected exactly one element type but found "
                << subscript.subscript_exprs().size();
        }
        # 解析并获取第一个 subscript 表达式的类型
        auto elem_type =
            parseTypeFromExprImpl(*subscript.subscript_exprs().begin());
        # 返回一个 AwaitType 对象，使用 elem_type 创建
        return AwaitType::create(elem_type);
    } else if (typeName == "RRef") {
        # 如果 typeName 是 "RRef"
        # 检查 subscript 的表达式数量是否为 1
        if (subscript.subscript_exprs().size() != 1) {
            # 如果不是，则抛出错误报告，指示预期只有一个元素类型，但找到了实际的数量
            throw ErrorReport(subscript)
                << " expected exactly one element type but found "
                << subscript.subscript_exprs().size();
        }
        # 解析并获取第一个 subscript 表达式的类型
        auto elem_type =
            parseTypeFromExprImpl(*subscript.subscript_exprs().begin());
        # 返回一个 RRefType 对象，使用 elem_type 创建
        return RRefType::create(elem_type);
    } else if (typeName == "Dict" || typeName == "dict") {
        # 如果 typeName 是 "Dict" 或 "dict"
        # 检查 subscript 的表达式数量是否为 2
        if (subscript.subscript_exprs().size() != 2) {
            # 如果不是，则抛出错误报告，指示预期有两个元素类型，但找到了实际的数量
            throw ErrorReport(subscript)
                << " expected exactly 2 element types but found "
                << subscript.subscript_exprs().size();
        }
        # 解析第一个 subscript 表达式作为键类型
        auto key_type = parseTypeFromExprImpl(subscript.subscript_exprs()[0]);
        # 解析第二个 subscript 表达式作为值类型
        auto value_type = parseTypeFromExprImpl(subscript.subscript_exprs()[1]);
        # 返回一个 DictType 对象，使用 key_type 和 value_type 创建
        return DictType::create(key_type, value_type);
    } else {
        # 如果 typeName 无法匹配已知的类型构造器，则抛出错误报告
        throw ErrorReport(subscript.range())
            << "Unknown type constructor " << typeName;
    }
}

// 解析广播列表类型的函数，返回一个可选的类型和整数对
std::optional<std::pair<TypePtr, int32_t>> ScriptTypeParser::parseBroadcastList(
    const Expr& expr) const {
  // 如果表达式的类型是变量
  if (expr.kind() == TK_VAR) {
    auto var = Var(expr);
    auto& name = var.name().name();
    constexpr auto _size_prefix = "_size_";
    constexpr auto _size_suffix = "_t";
    constexpr auto _size_n_len = 9; // "_size_X_t" 的长度
    constexpr auto _size_prefix_len = 6; // "_size_" 的长度
    // 判断变量名是否以 "_size_" 开头，以 "_t" 结尾，并且长度为 9
    if (name.find(_size_prefix) == 0 && name.length() == _size_n_len &&
        name.find(_size_suffix) == _size_prefix_len + 1 &&
        ::isdigit(name[_size_prefix_len])) {
      int n = name[_size_prefix_len] - '0';
      // 返回一个包含 ListType::create(IntType::get()) 和 n 的类型对
      return std::pair<TypePtr, int32_t>(ListType::create(IntType::get()), n);
    }
  }

  // 如果表达式的类型不是 TK_SUBSCRIPT，则返回空
  if (expr.kind() != TK_SUBSCRIPT)
    return c10::nullopt;
  auto subscript = Subscript(expr);
  // 如果 subscript 的值类型不是 TK_VAR，则返回空
  if (subscript.value().kind() != TK_VAR)
    return c10::nullopt;
  auto var = Var(subscript.value());
  auto subscript_exprs = subscript.subscript_exprs();

  // 处理 BroadcastingList 被 Optional 类型包装的情况
  if (var.name().name() == "Optional") {
    auto broadcast_list = parseBroadcastList(subscript_exprs[0]);
    // 如果成功解析到 broadcast_list
    if (broadcast_list) {
      TypePtr opt_type = OptionalType::create(broadcast_list->first);
      // 返回一个包含 OptionalType::create(broadcast_list->first) 和 broadcast_list->second 的类型对
      return std::pair<TypePtr, int32_t>(opt_type, broadcast_list->second);
    } else {
      return c10::nullopt;
    }
  } else if (var.name().name().find("BroadcastingList") != 0) {
    return c10::nullopt;
  }

  // 如果 subscript_exprs 的大小不为 1，则抛出错误
  if (subscript_exprs.size() != 1)
    throw ErrorReport(subscript.subscript_exprs().range())
        << "BroadcastingList/Optional[BroadcastingList] "
           "must be subscripted with a type";

  auto typ = subscript_exprs[0];
  auto len = var.name().name().substr(strlen("BroadcastingList"));

  // 如果 typ 的类型不是 TK_VAR，则抛出错误
  if (typ.kind() != TK_VAR)
    throw ErrorReport(subscript.value().range())
        << "Subscripted type must be a type identifier";

  auto value_name = Var(typ).name().name();
  // 如果 value_name 不是 "float" 也不是 "int"，则抛出错误
  if (value_name != "float" && value_name != "int")
    throw ErrorReport(subscript.value().range())
        << "Broadcastable lists only supported for int or float";

  // 查找 value_name 对应的类型指针
  auto elem_ptr = string_to_type_lut().find(value_name);
  AT_ASSERT(elem_ptr != string_to_type_lut().end());
  TypePtr list_ptr = ListType::create(elem_ptr->second);

  const char* len_c = len.c_str();
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  char* end;
  size_t len_v = strtoull(len_c, &end, 10);
  // 如果 end 不等于 len_c + len.size()，则抛出错误
  if (end != len_c + len.size()) {
    throw ErrorReport(subscript.subscript_exprs().range())
        << "subscript of Broadcastable list must be a positive integer";
  }
  // 返回一个包含 list_ptr 和 len_v 的类型对
  return std::pair<TypePtr, int32_t>(list_ptr, len_v);
}

// 解析基础类型名称的函数，返回一个可选的字符串
std::optional<std::string> ScriptTypeParser::parseBaseTypeName(
    const Expr& expr) const {
  switch (expr.kind()) {
    // 对于 TK_VAR 类型的情况，创建 Var 对象并返回其名称（变量名）
    case TK_VAR: {
      return Var(expr).name().name();
    }
    // 对于 TK_NONE 类型的情况，返回字符串 "None"
    case TK_NONE: {
      return "None";
    }
    // 对于 TK_NONE_TYPE 类型的情况，返回字符串 "NoneType"
    case TK_NONE_TYPE: {
      return "NoneType";
    }
    // 对于 '.' 字符的情况，进行特殊处理
    case '.': {
      // 创建 Select 对象来处理表达式
      auto select = Select(expr);
      // 获取选择器的名称
      const std::string& name = select.selector().name();
      // 特殊情况处理：针对 torch.Tensor 及其子类
      const std::unordered_set<std::string> tensor_subtypes = {
          "Tensor",
          "LongTensor",
          "FloatTensor",
          "DoubleTensor",
          "IntTensor",
          "ShortTensor",
          "HalfTensor",
          "CharTensor",
          "ByteTensor",
          "BoolTensor"};
      // 如果是 torch.Tensor 及其子类中的一种，则返回名称
      if (isTorch(select.value()) && tensor_subtypes.count(name) == 1) {
        return name;
      } else {
        // 否则，返回完全限定的类名
        return collectQualname(select);
      }
    } break;
  }
  // 如果以上情况均不匹配，则返回空的 optional 对象
  return at::nullopt;
}

TypePtr ScriptTypeParser::parseTypeFromExpr(const Expr& expr) const {
  // 如果表达式的类型是 '|'，则转换为 PEP 604 规定的 Union 类型
  if (expr.kind() == '|') {
    auto converted = pep604union_to_union(expr);
    return parseTypeFromExpr(converted);
  }
  // 如果存在解析器，则尝试解析表达式对应的类型
  if (resolver_) {
    // 通过解析器解析表达式对应的类型名称，并返回其类型指针
    if (auto typePtr =
            resolver_->resolveType(expr.range().text().str(), expr.range())) {
      return typePtr;
    }
  }
  // 否则，调用内部实现方法进行类型解析
  return parseTypeFromExprImpl(expr);
}

TypePtr ScriptTypeParser::parseTypeFromExprImpl(const Expr& expr) const {
  // 如果表达式的类型是 '|'，则转换为 PEP 604 规定的 Union 类型
  if (expr.kind() == '|') {
    auto converted = pep604union_to_union(expr);
    return parseTypeFromExprImpl(converted);
  }
  // 如果表达式的类型是 TK_SUBSCRIPT，则解析下标表达式
  if (expr.kind() == TK_SUBSCRIPT) {
    auto subscript = Subscript(expr);
    // 解析下标表达式中基本类型的名称
    auto value_name = parseBaseTypeName(subscript.value());
    if (!value_name) {
      // 抛出错误，表明下标类型必须是类型标识符
      throw ErrorReport(subscript.value().range())
          << "Subscripted type must be a type identifier";
    }
    // 将下标表达式转换为对应的类型
    return subscriptToType(*value_name, subscript);

  } else if (expr.kind() == TK_STRINGLITERAL) {
    // 如果表达式的类型是 TK_STRINGLITERAL，则解析字符串字面量类型名称
    const auto& type_name = StringLiteral(expr).text();

    // 检查类型是否是自定义类。这通过检查 type_name 是否以 "torch.classes." 开头来判断
    if (type_name.find("torch.classes.") == 0) {
      // 获取对应的自定义类类型
      auto custom_class_type = getCustomClass("__torch__." + type_name);
      return custom_class_type;
    }

    // 对于特定的 CUDA 类型，返回相应的自定义类类型
    if (type_name.find("torch.cuda.Stream") == 0) {
      auto custom_class_type =
          getCustomClass("__torch__.torch.classes.cuda.Stream");
      return custom_class_type;
    }

    if (type_name.find("torch.cuda.Event") == 0) {
      auto custom_class_type =
          getCustomClass("__torch__.torch.classes.cuda.Event");
      return custom_class_type;
    }

    // 如果存在解析器，则尝试解析字符串字面量对应的类型
    if (resolver_) {
      if (auto typePtr = resolver_->resolveType(type_name, expr.range())) {
        return typePtr;
      }
    }

    // 抛出错误，表明未知类型名称
    throw ErrorReport(expr) << "Unknown type name '" << type_name << "'";
  } else if (auto name = parseBaseTypeName(expr)) {
    // 解析表达式对应的基本类型名称，并查找类型名称到类型指针的映射
    auto itr = string_to_type_lut().find(*name);
    if (itr != string_to_type_lut().end()) {
      return itr->second;
    }
    // 如果存在解析器，则尝试解析表达式对应的类型
    if (resolver_) {
      if (auto typePtr = resolver_->resolveType(*name, expr.range())) {
        return typePtr;
      }
    }
    // 获取自定义类类型
    if (auto custom_class_type = getCustomClass(*name)) {
      return custom_class_type;
    }

    // 抛出错误，表明未知类型名称
    throw ErrorReport(expr) << "Unknown type name '" << *name << "'";
  }
  // 抛出错误，表明不支持使用该类型表达式
  throw ErrorReport(expr.range())
      << "Expression of type " << kindToString(expr.kind())
      << " cannot be used in a type expression";
}
// 解析函数类型字符串，并返回相应的类型指针
TypePtr ScriptTypeParser::parseType(const std::string& str) {
  // 创建解析器对象，传入字符串源作为参数
  Parser p(std::make_shared<Source>(str));
  // 解析表达式并返回其类型
  return parseTypeFromExpr(p.parseExp());
}

// 对默认参数表达式进行评估，生成默认值列表
std::vector<IValue> ScriptTypeParser::evaluateDefaults(
    const SourceRange& r,
    const std::vector<Expr>& default_types,
    const std::vector<Expr>& default_exprs) {
  // 初始化默认值向量
  std::vector<IValue> default_values;
  // 如果默认表达式为空，则直接返回空的默认值向量
  if (default_exprs.empty())
    return default_values;
  
  // 创建元组类型表达式
  auto tuple_type = Subscript::create(
      r,
      Var::create(r, Ident::create(r, "Tuple")),
      List<Expr>::create(r, default_types));
  // 创建空声明
  auto blank_decl = Decl::create(
      r, List<Param>::create(r, {}), Maybe<Expr>::create(r, tuple_type));

  // 创建元组字面量表达式
  auto tuple_expr =
      TupleLiteral::create(r, List<Expr>::create(r, default_exprs));
  // 创建返回语句
  auto ret = Return::create(r, tuple_expr);
  // 创建函数定义
  auto def = Def::create(
      r,
      Ident::create(r, "defaults"),
      blank_decl,
      List<Stmt>::create(r, {ret}));

  // 创建编译单元
  CompilationUnit cu;
  // 定义函数属性和解析器
  cu.define(
      c10::nullopt,
      /*properties=*/{},
      /*propResolvers=*/{},
      {def},
      {resolver_},
      nullptr);
  // 创建堆栈
  Stack stack;
  // XXX: 这里需要关闭优化，因为否则会在 DecomposeOps 中递归初始化内容
  GraphOptimizerEnabledGuard guard(false);
  // 运行函数
  cu.get_function(def.name().name()).run(stack);
  // 返回堆栈中的元组元素向量
  return stack.at(0).toTupleRef().elements().vec();
}

// 从声明中解析参数列表
std::vector<Argument> ScriptTypeParser::parseArgsFromDecl(
    const Decl& decl,
    bool skip_self) {
  // 获取参数列表的起始和结束位置迭代器
  auto params_begin = decl.params().begin();
  auto params_end = decl.params().end();
  // 如果需要跳过 self 参数，则将起始位置迭代器向前移动
  if (skip_self) {
    ++params_begin;
  }
  // 初始化返回值向量
  std::vector<Argument> retval;

  // 初始化默认类型表达式和默认表达式向量
  std::vector<Expr> default_types;
  std::vector<Expr> default_exprs;
  // 收集任何非空默认参数
  for (auto it = params_begin; it != params_end; ++it) {
    // 获取参数对象
    auto param = *it;
    // 获取参数的默认值表达式
    auto def = param.defaultValue();
    // 如果参数有默认值定义
    if (def.present()) {
      // 如果参数没有显式类型提示
      if (!param.type().present()) {
        // 对于默认表达式，我们需要显式类型提示。
        // 如果参数没有类型，我们可以默认为 "Tensor"，
        // 就像 Python 前端的行为一样。
        // 然而这里情况稍微复杂，因为默认表达式是使用自定义构建的图进行评估的，
        // 如果类型与值不匹配，出现的错误消息会相当难以理解。
        throw ErrorReport(param.range())
            << "Keyword arguments with defaults need to be type-hinted (TorchScript C++ frontend)";
      }
      // 将参数的类型和默认表达式加入到相应的列表中
      default_types.emplace_back(param.type().get());
      default_exprs.emplace_back(def.get());
    }
  }

  // 使用默认类型和表达式进行评估
  auto default_values =
      evaluateDefaults(decl.range(), default_types, default_exprs);

  // 开始处理参数列表中的每个参数
  auto defaults_it = default_values.begin();
  for (auto it = params_begin; it != params_end; ++it) {
    auto decl_arg = *it;

    TypePtr type;
    std::optional<int32_t> N = c10::nullopt;
    // 如果参数没有类型提示，默认为 "tensor"
    if (!decl_arg.type().present()) {
      type = TensorType::getInferred();
    } else {
      // 解析参数的类型表达式，可能包含 BroadcastList
      Expr type_expr = decl_arg.type().get();
      if (auto maybe_broad_list = parseBroadcastList(type_expr)) {
        // 如果是 BroadcastList，则提取类型和 N 值
        type = maybe_broad_list->first;
        N = maybe_broad_list->second;
      } else {
        // 否则，解析参数的普通类型表达式
        type = parseTypeFromExpr(decl_arg.type().get());
      }
    }
    std::optional<IValue> default_value = c10::nullopt;
    // 如果参数有默认值
    if (decl_arg.defaultValue().present()) {
      // 获取默认值列表中的下一个值
      default_value = *defaults_it++;
    }
    // 创建 Argument 对象，并添加到返回值列表中
    auto arg = Argument(
        decl_arg.ident().name(),
        type,
        N,
        default_value,
        decl_arg.kwarg_only(),
        /*alias_info=*/c10::nullopt);
    retval.push_back(arg);
  }
  // 返回最终的参数列表
  return retval;
}

std::vector<Argument> ScriptTypeParser::parseReturnFromDecl(const Decl& decl) {
    // 如果声明中没有返回类型注解，返回空的参数列表，表示没有返回值
    // 在 emitReturn 中，如果没有提供返回类型，我们会将实际返回值视为返回语句的值
    if (!decl.return_type().present())
        return {};

    // 检查返回类型中是否包含广播列表，如果包含则抛出错误
    if (parseBroadcastList(decl.return_type().get()))
        throw ErrorReport(decl.return_type().range())
            << "Broadcastable lists cannot appear as a return type";

    TypePtr parsed_type;
    // 获取声明中的返回类型表达式
    Expr type_expr = decl.return_type().get();
    // 解析表达式获取类型信息
    parsed_type = parseTypeFromExpr(type_expr);
    // 返回包含单个返回值信息的参数列表
    return {Argument(
        "",
        parsed_type,
        /*N =*/c10::nullopt,
        /*default_value =*/c10::nullopt,
        /*kwarg_only =*/false)};
}

FunctionSchema ScriptTypeParser::parseSchemaFromDef(
    const Def& def,
    bool skip_self) {
    // 获取定义的名称
    const auto name = def.name().name();
    // 解析定义中的参数列表
    std::vector<Argument> args = parseArgsFromDecl(def.decl(), skip_self);
    // 解析定义中的返回值信息
    std::vector<Argument> returns = parseReturnFromDecl(def.decl());
    // 构造函数的模式对象并返回
    return FunctionSchema(
        name, "", std::move(args), std::move(returns), false, false);
}

c10::IValue ScriptTypeParser::parseClassConstant(const Assign& assign) {
    // 检查是否为变量赋值操作
    if (assign.lhs().kind() != TK_VAR) {
        throw ErrorReport(assign.range())
            << "Expected to a variable for class constant";
    }
    // 检查是否存在类型注解
    if (!assign.type().present()) {
        throw ErrorReport(assign.range())
            << "Expected a type to present for class constant";
    }
    // 获取最终的类型表达式
    const auto final_type = assign.type().get();
    auto expr = assign.rhs().get();
    // 检查最终的类型是否为子类型化
    if (final_type.kind() != TK_SUBSCRIPT) {
        throw ErrorReport(assign.range())
            << "Expected subscripted type for class constant";
    }
    auto subscript = Subscript(final_type);
    // 解析基本类型名称
    auto value_name = parseBaseTypeName(subscript.value());
    // 检查基本类型名称是否存在
    if (!value_name) {
        throw ErrorReport(subscript.value().range())
            << "Subscripted type must be a type identifier";
    }
    // 检查基本类型名称是否为 "Final"
    if (*value_name != "Final") {
        throw ErrorReport(subscript.range())
            << "Base type must be Final for class constant";
    }
    // 检查子脚本表达式的数量，确保只有一个
    if (subscript.subscript_exprs().size() != 1) {
        throw ErrorReport(subscript)
            << " expected exactly one element type but found "
            << subscript.subscript_exprs().size();
    }
    // 获取子脚本的类型表达式和默认值
    auto type = *subscript.subscript_exprs().begin();
    auto default_val = evaluateDefaults(expr.range(), {type}, {expr});
    // 返回计算后的默认值
    return *default_val.begin();
}

} // namespace torch::jit
```