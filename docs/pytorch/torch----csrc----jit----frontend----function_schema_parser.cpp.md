# `.\pytorch\torch\csrc\jit\frontend\function_schema_parser.cpp`

```
// 引入 Torch JIT 前端的函数模式解析头文件
#include <torch/csrc/jit/frontend/function_schema_parser.h>

// 引入 ATen 核心的 Reducion 头文件
#include <ATen/core/Reduction.h>
// 引入 ATen 核心的 JIT 类型头文件
#include <ATen/core/jit_type.h>
// 引入 ATen 核心的类型工厂头文件
#include <ATen/core/type_factory.h>
// 引入 C10 实用工具的 Optional 头文件
#include <c10/util/Optional.h>
// 引入 Torch JIT 前端的词法分析器头文件
#include <torch/csrc/jit/frontend/lexer.h>
// 引入 Torch JIT 前端的解析字符串字面量头文件
#include <torch/csrc/jit/frontend/parse_string_literal.h>
// 引入 Torch JIT 前端的模式类型解析器头文件
#include <torch/csrc/jit/frontend/schema_type_parser.h>

// 引入标准库中的功能性组件
#include <functional>
// 引入标准库中的智能指针
#include <memory>
// 引入标准库中的向量容器
#include <vector>

// 使用 ATen 命名空间中的 TypeKind
using at::TypeKind;
// 使用 C10 命名空间中的 Argument
using c10::Argument;
// 使用 C10 命名空间中的 FunctionSchema
using c10::FunctionSchema;
// 使用 C10 命名空间中的 IValue
using c10::IValue;
// 使用 C10 命名空间中的 ListType
using c10::ListType;
// 使用 C10 命名空间中的 OperatorName
using c10::OperatorName;

// 定义 Torch JIT 的命名空间
namespace torch::jit {

// 匿名命名空间用于实现 SchemaParser 结构体
namespace {
// SchemaParser 结构体定义
struct SchemaParser {
  // 构造函数，接受字符串和是否允许类型变量作为参数
  explicit SchemaParser(const std::string& str, bool allow_typevars)
      : L(std::make_shared<Source>(
            c10::string_view(str),  // 使用给定的字符串视图创建 Source 对象
            c10::nullopt,           // 无需额外信息
            0,                      // 初始行号为 0
            nullptr,                // 初始位置为空
            Source::DONT_COPY)),    // 不进行字符串复制
        type_parser(L, /*parse_complete_tensor_types*/ false, allow_typevars) {}

  // 解析声明，返回 OperatorName 或 FunctionSchema 对象
  std::variant<OperatorName, FunctionSchema> parseDeclaration() {
    OperatorName name = parseName();  // 解析操作符名称

    // 如果当前符号不是 '('，则只是操作符名称，没有参数列表
    if (L.cur().kind != '(') {
      return OperatorName(std::move(name));  // 返回仅包含操作符名称的 OperatorName 对象
    }

    std::vector<Argument> arguments;  // 存储参数的向量
    std::vector<Argument> returns;    // 存储返回值的向量
    bool kwarg_only = false;          // 标志是否仅关键字参数
    bool is_vararg = false;           // 标志是否有可变参数
    bool is_varret = false;           // 标志是否有可变返回值
    size_t idx = 0;                   // 参数索引

    // 解析参数列表，支持 ',' 分隔符
    parseList('(', ',', ')', [&] {
      if (is_vararg)
        throw ErrorReport(L.cur())
            << "... must be the last element of the argument list";  // 报告错误：可变参数必须是参数列表的最后一个元素
      if (L.nextIf('*')) {
        kwarg_only = true;  // 下一个符号是 '*'，说明是仅关键字参数
      } else if (L.nextIf(TK_DOTS)) {
        is_vararg = true;  // 下一个符号是 '...', 说明是可变参数
      } else {
        arguments.push_back(parseArgument(
            idx++, /*is_return=*/false, /*kwarg_only=*/kwarg_only));  // 解析参数并添加到参数列表中
      }
    });

    // 检查所有参数是否对于可变参数模式来说都是非默认的
    if (is_vararg) {
      for (const auto& arg : arguments) {
        if (arg.default_value().has_value()) {
          throw ErrorReport(L.cur())
              << "schemas with vararg (...) can't have default value args";  // 报告错误：可变参数模式下不能有默认值参数
        }
      }
    }

    idx = 0;  // 重置参数索引
    L.expect(TK_ARROW);  // 期望遇到箭头符号 '->'

    // 解析返回值列表
    if (L.nextIf(TK_DOTS)) {
      is_varret = true;  // 下一个符号是 '...', 说明是可变返回值
    } else if (L.cur().kind == '(') {
      parseList('(', ',', ')', [&] {
        if (is_varret) {
          throw ErrorReport(L.cur())
              << "... must be the last element of the return list";  // 报告错误：可变返回值必须是返回列表的最后一个元素
        }
        if (L.nextIf(TK_DOTS)) {
          is_varret = true;  // 下一个符号是 '...', 说明是可变返回值
        } else {
          returns.push_back(
              parseArgument(idx++, /*is_return=*/true, /*kwarg_only=*/false));  // 解析返回值并添加到返回列表中
        }
      });
    } else {
      returns.push_back(
          parseArgument(0, /*is_return=*/true, /*kwarg_only=*/false));  // 解析返回值并添加到返回列表中
    }

    // 构造并返回 FunctionSchema 对象
    return FunctionSchema(
        std::move(name.name),      // 操作符名称
        std::move(name.overload_name),  // 操作符重载名称
        std::move(arguments),     // 参数列表
        std::move(returns),       // 返回值列表
        is_vararg,                // 是否有可变参数
        is_varret);               // 是否有可变返回值
  }

  // 解析操作符名称
  c10::OperatorName parseName() {
    // 从词法分析器中获取下一个标识符作为变量名
    std::string name = L.expect(TK_IDENT).text();
    // 如果下一个字符是冒号，则表示当前变量名后面会跟着命名空间限定符
    if (L.nextIf(':')) {
      L.expect(':');
      // 将命名空间限定符连接到变量名后面
      name = name + "::" + L.expect(TK_IDENT).text();
    }
    // 初始化重载名为空字符串
    std::string overload_name = "";
    // 如果下一个字符是点号，则获取下一个标识符作为重载名
    if (L.nextIf('.')) {
      overload_name = L.expect(TK_IDENT).text();
    }
    // 检查重载名是否有效，不能为"default"且不能以双下划线开头
    bool is_a_valid_overload_name =
        !((overload_name == "default") || (overload_name.rfind("__", 0) == 0));
    // 使用 TORCH_CHECK 断言检查重载名的合法性
    TORCH_CHECK(
        is_a_valid_overload_name,
        overload_name,
        " is not a legal overload name for aten operators");
    // 返回变量名和重载名作为结果
    return {name, overload_name};
  }

  // 解析多个声明并返回其结果的向量
  std::vector<std::variant<OperatorName, FunctionSchema>> parseDeclarations() {
    std::vector<std::variant<OperatorName, FunctionSchema>> results;
    // 循环解析声明直到遇到换行符
    do {
      results.emplace_back(parseDeclaration());
    } while (L.nextIf(TK_NEWLINE));
    // 最后期望文件结束符
    L.expect(TK_EOF);
    // 返回解析结果向量
    return results;
  }

  // 解析精确一个声明并返回其结果
  std::variant<OperatorName, FunctionSchema> parseExactlyOneDeclaration() {
    auto result = parseDeclaration();
    // 如果下一个字符是换行符，则跳过它
    L.nextIf(TK_NEWLINE);
    // 最后期望文件结束符
    L.expect(TK_EOF);
    // 返回解析结果
    return result;
  }

  // 解析一个参数
  Argument parseArgument(size_t /*idx*/, bool is_return, bool kwarg_only) {
    // 解析类型信息，包括真实类型和伪类型
    auto p = type_parser.parseFakeAndRealType();
    auto fake_type = std::move(std::get<0>(p));
    auto real_type = std::move(std::get<1>(p));
    auto alias_info = std::move(std::get<2>(p));
    std::optional<int32_t> N;
    std::optional<IValue> default_value;
    std::optional<std::string> alias_set;
    std::string name;
    // 如果下一个字符是左方括号，则解析数组参数
    if (L.nextIf('[')) {
      // 获取数组的大小提示
      N = std::stoll(L.expect(TK_NUMBER).text());
      L.expect(']');
      // 解析类型别名注解
      auto container = type_parser.parseAliasAnnotation();
      // 如果之前有别名信息，则将其添加到容器中
      if (alias_info) {
        if (!container) {
          container = std::optional<at::AliasInfo>(at::AliasInfo());
          container->setIsWrite(alias_info->isWrite());
        }
        container->addContainedType(std::move(*alias_info));
      }
      alias_info = std::move(container);
      // 如果下一个字符是问号，则表示可选类型
      if (L.nextIf('?')) {
        fake_type =
            c10::TypeFactory::create<c10::OptionalType>(std::move(fake_type));
        real_type =
            c10::TypeFactory::create<c10::OptionalType>(std::move(real_type));
      }
    }
    // 如果是返回值，可能会有字段名
    if (is_return) {
      // 如果当前符号是标识符，则作为参数名
      if (L.cur().kind == TK_IDENT) {
        name = L.next().text();
      } else {
        name = "";
      }
      // 实际参数解析和类型信息构建
      // 省略了剩余的解析过程，不在当前代码段内
  } else {
    // 从词法分析器中期望获取标识符作为参数的名称
    name = L.expect(TK_IDENT).text();
    // 如果接下来是 '='，则表示有默认值需要解析
    if (L.nextIf('=')) {
      // 注意：这意味着我们也需要解析默认值的结构
      default_value =
          // 解析默认值，使用虚拟类型、类型种类、真实类型和名称参数
          parseDefaultValue(*fake_type, fake_type->kind(), *real_type, N);
    }
  }
  // 返回 Argument 对象，其中包括参数名称、虚拟类型、真实类型、参数位置、默认值、是否不是返回值且是关键字参数、别名信息
  return Argument(
      std::move(name),
      std::move(fake_type),
      std::move(real_type),
      N,
      std::move(default_value),
      !is_return && kwarg_only,
      std::move(alias_info));
}

bool isPossiblyOptionalScalarType(const c10::Type& type) {
  // 如果类型是标量类型的类型（ScalarTypeType），则可能是可选的标量类型
  if (type.kind() == at::ScalarTypeType::Kind) {
    return true;
  }
  // 如果类型是可选类型的类型（OptionalType），则检查其包含的类型
  if (type.kind() == at::OptionalType::Kind) {
    // 遍历包含的每种类型，如果其中任意一种是可能的可选标量类型，则返回 true
    for (const auto& inner : type.containedTypes()) {
      if (isPossiblyOptionalScalarType(*inner))
        return true;
    }
  }
  // 默认情况下返回 false，表示不是可能的可选标量类型
  return false;
}

IValue parseSingleConstant(
    const c10::Type& type,
    TypeKind kind,
    const c10::Type& real_type) {
  // 如果类型的种类是动态类型（DynamicType），则解析单个常量时需要根据动态类型继续解析
  if (kind == c10::TypeKind::DynamicType) {
    return parseSingleConstant(
        type, type.expectRef<c10::DynamicType>().dynamicKind(), real_type);
  }
  // 获取字符串到数据类型映射表
  const auto& str2dtype = c10::getStringToDtypeMap();
    switch (L.cur().kind) {
      case TK_TRUE:
        L.next();
        return true;  // 返回布尔值 true
      case TK_FALSE:
        L.next();
        return false;  // 返回布尔值 false
      case TK_NONE:
        L.next();
        return IValue();  // 返回默认构造的 IValue 对象
      case TK_STRINGLITERAL: {
        auto token = L.next();
        return parseStringLiteral(token.range, token.text());  // 解析字符串字面量并返回结果
      }
      case TK_IDENT: {
        auto tok = L.next();
        auto text = tok.text();
        // NB: float/complex/long are here for BC purposes. Other dtypes
        // are handled via str2dtype.
        // Please don't add more cases to this if-else block.
        // 根据标识符的文本内容返回对应的整数值，用于类型转换
        if ("float" == text) {
          return static_cast<int64_t>(at::kFloat);
        } else if ("complex" == text) {
          return static_cast<int64_t>(at::kComplexFloat);
        } else if ("long" == text) {
          return static_cast<int64_t>(at::kLong);
        } else if ("strided" == text) {
          return static_cast<int64_t>(at::kStrided);
        } else if ("Mean" == text) {
          return static_cast<int64_t>(at::Reduction::Mean);
        } else if ("contiguous_format" == text) {
          return static_cast<int64_t>(c10::MemoryFormat::Contiguous);
        } else if (
            isPossiblyOptionalScalarType(real_type) &&
            str2dtype.count(text) > 0) {
          return static_cast<int64_t>(str2dtype.at(text));
        } else {
          throw ErrorReport(L.cur().range) << "invalid numeric default value";  // 抛出异常，标识找到无效的默认数值
        }
      }
      default:
        std::string n;
        if (L.nextIf('-'))
          n = "-" + L.expect(TK_NUMBER).text();  // 处理负数情况
        else
          n = L.expect(TK_NUMBER).text();  // 获取正数或者其他数值

        if (kind == TypeKind::ComplexType || n.find('j') != std::string::npos) {
          auto imag = std::stod(n.substr(0, n.size() - 1));
          return c10::complex<double>(0, imag);  // 处理复数类型或者带有虚部的数值
        } else if (
            kind == TypeKind::FloatType || n.find('.') != std::string::npos ||
            n.find('e') != std::string::npos) {
          return std::stod(n);  // 处理浮点数类型
        } else {
          int64_t v = std::stoll(n);
          return v;  // 处理整数类型
        }
    }
  }
  IValue convertToList(
      const c10::Type& type,
      TypeKind kind,
      const SourceRange& range,
      const std::vector<IValue>& vs) {
    switch (kind) {
      case TypeKind::ComplexType:
        return fmap(vs, [](const IValue& v) { return v.toComplexDouble(); });  // 转换为复数列表
      case TypeKind::FloatType:
        return fmap(vs, [](const IValue& v) { return v.toDouble(); });  // 转换为浮点数列表
      case TypeKind::IntType:
        return fmap(vs, [](const IValue& v) { return v.toInt(); });  // 转换为整数列表
      case TypeKind::BoolType:
        return fmap(vs, [](const IValue& v) { return v.toBool(); });  // 转换为布尔值列表
      case TypeKind::DynamicType:
        return convertToList(
            type, type.expectRef<c10::DynamicType>().dynamicKind(), range, vs);  // 处理动态类型列表
      default:
        throw ErrorReport(range)
            << "lists are only supported for float, int and complex types";  // 抛出异常，表示仅支持浮点数、整数和复数类型的列表
    }
  }
  }
  IValue parseConstantList(
      const c10::Type& type,
      TypeKind kind,
      const c10::Type& real_type) {
    auto tok = L.expect('[');
    // 期望当前标记是左方括号'['，获取其位置信息
    std::vector<IValue> vs;
    if (L.cur().kind != ']') {
      // 如果当前标记不是右方括号']'，进入循环
      do {
        // 解析单个常量，并将其添加到向量vs中
        vs.push_back(parseSingleConstant(type, kind, real_type));
      } while (L.nextIf(','));
      // 如果下一个标记是逗号','，继续循环
    }
    L.expect(']');
    // 期望当前标记是右方括号']'，获取其位置信息
    return convertToList(type, kind, tok.range, vs);
    // 将解析得到的向量vs转换为列表类型IValue并返回
  }

  IValue parseTensorDefault(const SourceRange& /*range*/) {
    L.expect(TK_NONE);
    // 期望当前标记是TK_NONE，表示空标记
    return IValue();
    // 返回一个空的IValue对象
  }

  IValue parseDefaultValue(
      const c10::Type& arg_type,
      TypeKind kind,
      const c10::Type& real_type,
      std::optional<int32_t> arg_N) {
    auto range = L.cur().range;
    // 获取当前标记的位置信息
    switch (kind) {
      case TypeKind::TensorType:
      case TypeKind::GeneratorType:
      case TypeKind::QuantizerType: {
        // 对于张量、生成器、量化器类型，调用解析张量默认值函数
        return parseTensorDefault(range);
      } break;
      case TypeKind::StringType:
      case TypeKind::OptionalType:
      case TypeKind::NumberType:
      case TypeKind::IntType:
      case TypeKind::BoolType:
      case TypeKind::FloatType:
      case TypeKind::ComplexType:
        // 对于字符串、可选类型、数字、整数、布尔、浮点、复数类型，调用解析单个常量函数
        return parseSingleConstant(arg_type, kind, real_type);
        break;
      case TypeKind::DeviceObjType: {
        auto device_text =
            parseStringLiteral(range, L.expect(TK_STRINGLITERAL).text());
        // 解析字符串文字作为设备对象类型的值
        return c10::Device(device_text);
        // 返回一个表示设备的对象
        break;
      }
      case TypeKind::ListType: {
        auto elem_type = arg_type.containedType(0);
        auto real_elem_type = real_type.containedType(0);
        if (L.cur().kind == TK_IDENT) {
          // 如果当前标记是标识符，调用解析张量默认值函数
          return parseTensorDefault(range);
        } else if (arg_N && L.cur().kind != '[') {
          // 如果有参数N且当前标记不是左方括号'['，解析单个常量，并将其重复N次作为列表的值
          IValue v = parseSingleConstant(
              *elem_type, elem_type->kind(), *real_elem_type);
          std::vector<IValue> repeated(*arg_N, v);
          return convertToList(*elem_type, elem_type->kind(), range, repeated);
        } else {
          // 否则，解析常量列表
          return parseConstantList(
              *elem_type, elem_type->kind(), *real_elem_type);
        }
      } break;
      case TypeKind::DynamicType:
        // 对于动态类型，调用解析默认值函数
        return parseDefaultValue(
            arg_type,
            arg_type.expectRef<c10::DynamicType>().dynamicKind(),
            real_type,
            arg_N);
      default:
        // 对于未预期的类型，抛出错误报告
        throw ErrorReport(range) << "unexpected type, file a bug report";
    }
    return IValue(); // silence warnings
    // 返回一个空的IValue对象，用于消除警告
  }

  void parseList(
      int begin,
      int sep,
      int end,
      c10::function_ref<void()> callback) {
    auto r = L.cur().range;
    // 获取当前标记的位置信息
    if (begin != TK_NOTHING)
      L.expect(begin);
    // 如果开始标记不是TK_NOTHING，期望当前标记是begin
    if (L.cur().kind != end) {
      // 如果当前标记不是end，进入循环
      do {
        // 调用回调函数处理列表中的元素
        callback();
      } while (L.nextIf(sep));
      // 如果下一个标记是分隔符sep，继续循环
    }
    if (end != TK_NOTHING)
      // 如果结束标记不是TK_NOTHING，期望当前标记是end
      L.expect(end);
  }
  Lexer L;
  SchemaTypeParser type_parser;
  bool allow_typevars_;
} // namespace
```