# `.\pytorch\torch\csrc\jit\frontend\schema_type_parser.cpp`

```
// 引入 Torch 的 JIT 前端模块中的头文件，用于解析类型模式
#include <torch/csrc/jit/frontend/schema_type_parser.h>

// 引入 ATen 库中的各种类型定义和功能
#include <ATen/core/alias_info.h>
#include <ATen/core/jit_type.h>
#include <ATen/core/symbol.h>
#include <ATen/core/type_factory.h>
#include <torch/csrc/jit/frontend/lexer.h>
#include <torch/csrc/jit/frontend/parse_string_literal.h>
#include <torch/custom_class.h>

// 引入标准库中的 string 类型定义
#include <string>

// 使用 c10 命名空间中的各种类型别名
using c10::AliasInfo;
using c10::AwaitType;
using c10::BoolType;
using c10::CapsuleType;
using c10::ComplexType;
using c10::DeviceObjType;
using c10::DictType;
using c10::FloatType;
using c10::FutureType;
using c10::GeneratorType;
using c10::IntType;
using c10::LayoutType;
using c10::ListType;
using c10::MemoryFormatType;
using c10::NoneType;
using c10::NumberType;
using c10::QSchemeType;
using c10::QuantizerType;
using c10::RRefType;
using c10::ScalarTypeType;
using c10::StorageType;
using c10::StreamObjType;
using c10::StringType;
using c10::Symbol;
using c10::SymIntType;
using c10::TensorType;
using c10::TupleType;
using c10::UnionType;
using c10::VarType;

// Torch 的 JIT 命名空间
namespace torch::jit {

// 解析器类 SchemaTypeParser 的成员函数 parseBaseType
TypePtr SchemaTypeParser::parseBaseType() {
  // 静态映射，将字符串类型映射到对应的类型指针
  static std::unordered_map<std::string, TypePtr> type_map = {
      {"Generator", c10::TypeFactory::get<GeneratorType>()},
      {"Dimname", c10::TypeFactory::get<StringType>()},
      {"ScalarType", c10::TypeFactory::get<ScalarTypeType>()},
      {"Layout", c10::TypeFactory::get<LayoutType>()},
      {"MemoryFormat", c10::TypeFactory::get<MemoryFormatType>()},
      {"Storage", c10::TypeFactory::get<StorageType>()},
      {"QScheme", c10::TypeFactory::get<QSchemeType>()},
      {"Quantizer", c10::TypeFactory::get<QuantizerType>()},
      {"ConstQuantizerPtr",
       c10::TypeFactory::get<IntType>()}, // TODO: 此类型应该从模式解析器中移除，
                                          // 应改用自定义类机制。@jerryzh
      {"Device", c10::TypeFactory::get<DeviceObjType>()},
      {"DeviceIndex", c10::TypeFactory::get<IntType>()},
      {"Stream", c10::TypeFactory::get<StreamObjType>()},
      {"Scalar", c10::TypeFactory::get<NumberType>()},
      {"str", c10::TypeFactory::get<StringType>()},
      {"float", c10::TypeFactory::get<FloatType>()},
      {"complex", c10::TypeFactory::get<ComplexType>()},
      {"int", c10::TypeFactory::get<IntType>()},
      {"SymInt", c10::TypeFactory::get<SymIntType>()},
      {"bool", c10::TypeFactory::get<BoolType>()},
      {"None", c10::TypeFactory::get<NoneType>()},
      {"NoneType", c10::TypeFactory::get<NoneType>()},
      {"Capsule", c10::TypeFactory::get<CapsuleType>()},
      {"Any", c10::TypeFactory::get<c10::AnyType>()},
      {"AnyClassType", c10::TypeFactory::get<c10::AnyClassType>()},
      {"AnyEnumType", c10::TypeFactory::get<c10::AnyEnumType>()},
  };

  // 获取当前词法分析器中的当前令牌
  auto tok = L.cur();

  // 如果当前令牌不是 TK_NONE 也不是 TK_NONE_TYPE，则...
  if (!L.nextIf(TK_NONE) && !L.nextIf(TK_NONE_TYPE)) {
    L.expect(TK_IDENT);

要求解析器期望下一个令牌是标识符（identifier）。


  }
  std::string text = tok.text();

结束前面的代码块并将当前令牌的文本内容存储在字符串变量 `text` 中。


  auto it = type_map.find(text);

在 `type_map` 中查找键为 `text` 的条目，并将结果保存在迭代器 `it` 中。


  if (it == type_map.end()) {

如果迭代器 `it` 指向 `type_map` 的末尾（即未找到匹配的类型）。


    if (allow_typevars_ && !text.empty() && islower(text[0])) {
      // lower case identifiers that are not otherwise valid types
      // are treated as type variables
      return c10::TypeFactory::createNamed<VarType>(text);
    }

如果允许类型变量，并且 `text` 非空且首字母是小写字母，则将其视为类型变量并返回相应的类型工厂创建的类型对象。


    if (text == "double") {
      throw ErrorReport(tok.range)
          << "Use `float` instead of `double` in an operator's schema string. "
             "`float` in schema corresponds to the double type in C++";
    }

如果 `text` 是 "double"，则抛出错误报告，建议在运算符的模式字符串中使用 `float` 而不是 `double`，因为模式中的 `float` 对应于 C++ 中的 double 类型。


    if (text == "int64_t") {
      throw ErrorReport(tok.range)
          << "Use `SymInt` or `int` instead of `int64_t` in an operator's schema string. "
             "`SymInt` corresponds to c10::SymInt in C++ while `int` in schema corresponds "
             "to the int64_t type in C++.";
    }

如果 `text` 是 "int64_t"，则抛出错误报告，建议在运算符的模式字符串中使用 `SymInt` 或 `int` 而不是 `int64_t`，因为模式中的 `SymInt` 对应于 C++ 中的 `c10::SymInt` 类型，而 `int` 对应于 C++ 中的 `int64_t` 类型。


    throw ErrorReport(tok.range)
        << "unknown type specifier. Common valid schema types include "
           "Tensor, SymInt, int, float, bool, Scalar; "
           "for a full list, please see "
           "https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/README.md#func ";

如果 `text` 不是已知的类型，则抛出错误报告，指出未知的类型说明符，并提供常见有效的模式类型列表的链接。


  }
  return it->second;

如果找到了匹配的类型，则返回 `type_map` 中对应键的值，即类型信息。
// 定义 SchemaTypeParser 类中的 parseAliasAnnotation 方法，返回 std::optional<AliasInfo> 类型的结果
std::optional<AliasInfo> SchemaTypeParser::parseAliasAnnotation() {
  // 创建 AliasInfo 对象以存储别名信息
  AliasInfo alias_info;
  // 如果下一个字符是 '('，表示存在可选的 'alias set annotation'
  if (L.nextIf('(')) {
    // 解析以 '|' 分隔的列表，每个元素为 TK_IDENT 或 '*'
    parseList(TK_NOTHING, '|', TK_NOTHING, [&] {
      // 如果下一个字符是 '*'，表示通配符，添加通配符到 'before set'
      if (L.nextIf('*')) {
        alias_info.addBeforeSet(AliasInfo::wildcardSet());
        // 如果找到通配符，则忽略所有后续的注释
      } else if (!alias_info.isWildcardBefore()) {
        // 否则，将形如 "alias::identifier" 的符号添加到 'before set'
        alias_info.addBeforeSet(
            Symbol::fromQualString("alias::" + L.expect(TK_IDENT).text()));
      }
    });
    // 如果下一个字符是 '!'，表示这是一个写入操作
    if (L.nextIf('!')) {
      alias_info.setIsWrite(true);
    }
    // 如果下一个字符是 TK_ARROW，表示存在可选的 'alias set annotation'
    if (L.nextIf(TK_ARROW)) {
      // 解析以 '|' 分隔的列表，每个元素为 TK_IDENT 或 '*'
      parseList(TK_NOTHING, '|', TK_NOTHING, [&] {
        // 如果下一个字符是 '*'，表示通配符，添加通配符到 'after set'
        if (L.nextIf('*')) {
          alias_info.addAfterSet(AliasInfo::wildcardSet());
          // 如果找到通配符，则忽略所有后续的注释
        } else if (!alias_info.isWildcardAfter()) {
          // 否则，将形如 "alias::identifier" 的符号添加到 'after set'
          alias_info.addAfterSet(
              Symbol::fromQualString("alias::" + L.expect(TK_IDENT).text()));
        }
      });
    } else {
      // 如果没有遇到箭头符号 TK_ARROW，则认为 'after set' 与 'before set' 相同
      AT_ASSERT(alias_info.afterSets().empty());
      for (const auto& set : alias_info.beforeSets()) {
        alias_info.addAfterSet(set);
      }
    }
    // 解析完列表后，期望下一个字符是 ')'
    L.expect(')');
  } else if (L.nextIf('!')) {
    // 如果下一个字符是 '!'，表示这是一个写入操作，并且使用唯一标识符添加到 'before set'
    alias_info.addBeforeSet(
        Symbol::fromQualString("alias::$" + std::to_string(next_id++)));
    alias_info.setIsWrite(true);
  } else {
    // 如果不是 '(' 也不是 '!'，则返回空值，表示没有找到有效的别名注解
    return c10::nullopt;
  }

  // 返回解析得到的 alias_info 对象
  return alias_info;
}

// 定义 SchemaTypeParser 类中的 parseTensorDType 方法，返回 std::optional<at::ScalarType> 类型的结果
std::optional<at::ScalarType> SchemaTypeParser::parseTensorDType(
    const std::string& dtype) {
  // 定义宏，展开为对应的标量类型映射
#define DEFINE_SCALAR_TYPE(_1, n) {#n, at::ScalarType::n},
  // 创建静态无序映射表，将标量类型名称映射为对应的 at::ScalarType 类型
  static std::unordered_map<std::string, at::ScalarType> type_map = {
      AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(DEFINE_SCALAR_TYPE)};
  // 取消宏定义
#undef DEFINE_SCALAR_TYPE
  // 在 type_map 中查找指定的 dtype
  auto type = type_map.find(dtype);
  // 如果找到了，则返回对应的标量类型，否则返回空值
  if (type != type_map.end()) {
    return type->second;
  }
  return c10::nullopt;
}

// 定义 SchemaTypeParser 类中的 tryToParseDeviceType 方法，返回 std::optional<c10::Device> 类型的结果
std::optional<c10::Device> SchemaTypeParser::tryToParseDeviceType() {
  // 期望下一个字符是 '='
  L.expect('=');
  // 期望下一个 token 是 TK_IDENT，并获取其文本内容作为设备类型 dev
  const std::string& dev = L.expect(TK_IDENT).text();

  // 如果 dev 是 "cpu"，返回对应的 CPU 设备
  if (dev == "cpu") {
    return c10::Device(at::kCPU);
  }

  // 如果 dev 是 "cuda" 或 "hpu"，暂时不处理具体的设备索引，返回相应的设备
  if (dev == "cuda" || dev == "hpu") {
    c10::DeviceIndex device_idx = -1;
    // 这里应该还有后续的处理设备索引的代码，未完待续

    // 这里应该还有后续的处理设备索引的代码，未完待续
    // 在这种情况下，返回对应的设备类型
  }
  
  // 如果 dev 不是 "cpu"、"cuda" 或 "hpu"，则返回空值，表示无法识别的设备类型
  return c10::nullopt;
}
    // 如果当前词法单元是冒号
    if (L.cur().kind == ':') {
      // 预期下一个词法单元为冒号，表明语法结构正确
      L.expect(':');
      // 获取下一个词法单元，应为数字，将其转换为字符串形式
      const std::string& num = L.expect(TK_NUMBER).text();
      // 忽略此行的变量初始化警告，因为 num_len 将在后续的 std::stoi 中初始化
      // 用于存储转换后的数字字符串长度的变量
      std::string::size_type num_len;
      try {
        // 将 num 转换为整数类型的设备索引，同时获取转换后的数字字符串长度
        device_idx = std::stoi(num, &num_len);
      } catch (const std::invalid_argument& e) {
        // 如果转换失败，抛出异常，指示设备索引无法转换为整数
        throw ErrorReport(L.cur())
            << "Device index cannot be converted to integer";
      } catch (const std::out_of_range& e) {
        // 如果转换后的数字超出范围，抛出异常，指示设备索引值过大
        throw ErrorReport(L.cur()) << "Device index is too long";
      }
    }
    // 如果设备类型为 "cuda"
    if (dev == "cuda") {
      // 返回一个 CUDA 设备对象，使用先前解析的设备索引
      return c10::Device(at::kCUDA, device_idx);
    } else {
      // 返回一个 HPU 设备对象，使用先前解析的设备索引
      return c10::Device(at::kHPU, device_idx);
    }
  }

  // 如果无法解析设备类型，抛出异常，指示无法识别的设备类型
  throw ErrorReport(L.cur()) << "cannot parse device type '" << dev << "'\n";
}

std::optional<bool> SchemaTypeParser::tryToParseRequiresGrad() {
  // 要求下一个字符为 '='
  L.expect('=');
  // 期望下一个标记为数字，并获取其文本内容
  const std::string& num = L.expect(TK_NUMBER).text();
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  // 初始化 num_len 用于存储成功转换的字符数
  std::string::size_type num_len;

  try {
    // 尝试将 num 转换为 bool 类型
    return (bool)std::stoi(num, &num_len);
  } catch (const std::invalid_argument& e) {
    // 如果转换失败，抛出错误报告，指出无法将 requires_grad 字段转换为整数
    throw ErrorReport(L.cur())
        << "Field requires_grad cannot be converted to integer";
  } catch (const std::out_of_range& e) {
    // 如果数值超出范围，抛出错误报告，指出 requires_grad 字段过长
    throw ErrorReport(L.cur()) << "Field requires_grad is too long";
  }
}

TypePtr SchemaTypeParser::parseRefinedTensor() {
  // 解析张量的数据类型，并确保成功获取
  auto maybe_dtype = parseTensorDType(L.expect(TK_IDENT).text());
  AT_ASSERT(maybe_dtype);
  // 将数据类型转换为 ScalarType
  at::ScalarType dtype = *maybe_dtype;
  // 初始化类型指针
  TypePtr ptr;
  // 期望下一个字符为 '('
  L.expect('(');
  // 初始化张量类型指针
  TypePtr tensor_type;
  // 初始化设备选项和 requires_grad 可选值
  std::optional<c10::Device> device;
  std::optional<bool> requires_grad;
  // 解析包含不同维度、大小、梯度等选项的类型描述
  // 例如: Long(10, 8, 6, strides=[48, 6, 1], requires_grad=0, device=cuda:1)
  //      Float(10, *, 20, device=cuda:1)
  //      Float(requires_grad=1)
  std::vector<std::optional<int64_t>> dims;
  bool seen_strides = false;
  std::vector<int64_t> strides;
  parseList(TK_NOTHING, ',', ')', [&] {
    // 处理 'device' 和 'requires_grad' 等选项的额外处理
    // 如果当前词法单元是标识符且不是"SS"
    if (L.cur().kind == TK_IDENT && L.cur().text() != "SS") {
      // 获取下一个标识符作为字段名
      const std::string& field = L.expect(TK_IDENT).text();
      // 如果字段名是"device"
      if (field == "device") {
        // 尝试解析设备类型
        auto parsed_device = tryToParseDeviceType();
        // 如果解析成功
        if (parsed_device.has_value()) {
          // 如果已经指定了设备类型，则抛出错误
          if (device.has_value()) {
            throw ErrorReport(L.cur()) << "'device' is specified twice";
          }
          // 将解析到的设备类型赋值给device
          device = parsed_device;
        }
        return;
      }
      // 如果字段名是"requires_grad"
      if (field == "requires_grad") {
        // 尝试解析requires_grad值
        auto parsed_requires_grad = tryToParseRequiresGrad();
        // 如果解析成功
        if (parsed_requires_grad.has_value()) {
          // 如果已经指定了requires_grad，则抛出错误
          if (requires_grad.has_value()) {
            throw ErrorReport(L.cur()) << "'requires_grad' is specified twice";
          }
          // 将解析到的requires_grad值赋值给requires_grad
          requires_grad = parsed_requires_grad;
        }
        return;
      }
      // 如果字段名是"strides"
      if (field == "strides") {
        seen_strides = true;
        // 期望遇到'='符号
        L.expect('=');
        // 解析以逗号分隔的列表
        parseList('[', ',', ']', [&] {
          // 获取下一个数值作为字符串
          const std::string& num = L.expect(TK_NUMBER).text();
          // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
          std::string::size_type num_len;
          try {
            // 尝试将字符串转换为长整型数值
            auto stride = std::stoll(num, &num_len);
            // 将转换后的数值加入strides向量
            strides.push_back(stride);
          } catch (const std::invalid_argument& e) {
            throw ErrorReport(L.cur())
                << "The stride value cannot be converted to int";
          } catch (const std::out_of_range& e) {
            throw ErrorReport(L.cur()) << "The stride is too big";
          }
        });
        return;
      }
      // 如果字段名既不是"device"也不是"requires_grad"也不是"strides"，抛出意外的字段名错误
      throw ErrorReport(L.cur()) << "Unexpected specifier '" << field << "'";
    }
    // 如果已经指定了device或requires_grad，抛出错误
    if (device.has_value() || requires_grad.has_value()) {
      throw ErrorReport(L.cur())
          << "'device' and 'requires_grad' should come after dimensions in the type specification";
    }

    // 解析维度，支持大小和未指定大小的混合，或者只有步长的维度
    if (L.cur().kind == '*') {
      // 将一个无值的Optional加入dims
      dims.emplace_back(c10::nullopt);
      // 移动到下一个词法单元
      L.next();
      // 如果接下来是':'，则抛出错误，不支持未指定大小的维度的步长
      if (L.cur().kind == ':') {
        throw ErrorReport(L.cur()) << "Strides for unsized ranks not supported";
      }
      return;
    }
    // 如果当前词法单元是标识符且其文本为"SS"
    bool shape_symbol = false;
    if (L.cur().kind == TK_IDENT && L.cur().text() == "SS") {
      // 移动到下一个词法单元
      L.next();
      // 期望下一个字符是'('
      L.expect('(');
      // 期望下一个字符是'-'
      L.expect('-');
      shape_symbol = true;
    }
    // 获取下一个数字作为字符串
    const std::string& num = L.expect(TK_NUMBER).text();
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    std::string::size_type num_len;
    int64_t dim = 0;
    try {
      // 尝试将数字字符串转换为长整型数值
      dim = std::stoll(num, &num_len);
    } catch (const std::invalid_argument& e) {
      throw ErrorReport(L.cur()) << "The number can't be converted to int";
    } catch (const std::out_of_range& e) {
      throw ErrorReport(L.cur()) << "Number is too big";
    }
    // 如果shape_symbol为真
    if (shape_symbol) {
      // 期望下一个字符是')'
      L.expect(')');
      // 将dim设为负数
      dim = -dim;
    }
    // 将dim加入dims向量
    dims.emplace_back(dim);
  });
  // 如果seen_strides为真
  if (seen_strides) {
    // 创建一个对strides的常量引用
    at::IntArrayRef strides_ref(strides);
    // 如果 strides 的大小不等于 dims 的大小，则抛出错误
    if (strides.size() != dims.size()) {
      // 注意：混合使用未指定步幅的维度和指定步幅的维度将总是触发这个错误
      throw ErrorReport(L.cur())
          << "Strides info is specified for some but not for all dimensions";
    }
    // 使用指定的参数创建一个新的 TensorType 对象，并赋值给指针 ptr
    ptr = at::TensorType::create(
        dtype,                                        // 数据类型
        device,                                       // 设备类型
        c10::VaryingShape<int64_t>(dims),              // 维度信息
        c10::VaryingShape<int64_t>(strides),           // 步幅信息
        requires_grad);                               // 是否需要梯度
  } else {
    // 使用指定的参数创建一个新的 TensorType 对象，并赋值给指针 ptr
    ptr = at::TensorType::create(
        dtype,                                        // 数据类型
        device,                                       // 设备类型
        c10::VaryingShape<int64_t>(dims),              // 维度信息
        c10::VaryingShape<int64_t>(dims.size()),       // 所有维度的默认步幅信息
        requires_grad);                               // 是否需要梯度
  }
  // 返回创建的 TensorType 对象的指针
  return ptr;
}

std::pair<TypePtr, std::optional<AliasInfo>> SchemaTypeParser::parseType() {
  // 调用parseFakeAndRealType函数解析假和真实类型，返回结果作为pair的一部分
  auto r = parseFakeAndRealType();
  // 返回包含解析结果的pair，第一个元素是假类型(TypePtr)，第二个元素是可选的别名信息(std::optional<AliasInfo>)
  return std::make_pair(std::move(std::get<0>(r)), std::move(std::get<2>(r)));
}

std::tuple</*fake*/ TypePtr, /*real*/ TypePtr, std::optional<AliasInfo>>
SchemaTypeParser::parseFakeAndRealType() {
  TypePtr fake_value;
  TypePtr real_value;
  std::optional<AliasInfo> alias_info;
  // 如果当前token是'('，则表示为元组类型
  if (L.cur().kind == '(') {
    std::vector<TypePtr> types;
    // 调用parseList函数解析元组中的类型列表
    parseList('(', ',', ')', [&] {
      auto r = parseType();
      types.push_back(std::move(r.first));
      // 如果已有别名信息并且当前类型也有别名信息，则将其添加到已有的别名信息中
      if (alias_info && r.second) {
        alias_info->addContainedType(std::move(*r.second));
      }
    });
    // 创建一个TupleType对象，包含解析出的类型列表
    fake_value = real_value =
        c10::TypeFactory::create<TupleType>(std::move(types));
  } else if (L.cur().kind == TK_IDENT && L.cur().text() == "Future") {
    L.next(); // Future
    L.expect('(');
    // 解析Future类型中的子类型
    auto p = parseType();
    auto subtype = std::move(p.first);
    auto subalias = std::move(p.second);
    L.expect(')');
    // 创建一个FutureType对象，包含解析出的子类型
    fake_value = real_value = c10::TypeFactory::create<FutureType>(subtype);
  } else if (L.cur().kind == TK_IDENT && L.cur().text() == "Await") {
    L.next(); // Await
    L.expect('(');
    // 解析Await类型中的子类型
    auto p = parseType();
    auto subtype = std::move(p.first);
    auto subalias = std::move(p.second);
    L.expect(')');
    // 创建一个AwaitType对象，包含解析出的子类型
    fake_value = real_value = c10::TypeFactory::create<AwaitType>(subtype);
  } else if (L.cur().kind == TK_IDENT && L.cur().text() == "RRef") {
    L.next(); // RRef
    L.expect('(');
    // 解析RRef类型中的子类型
    auto p = parseType();
    auto subtype = std::move(p.first);
    auto subalias = std::move(p.second);
    L.expect(')');
    // 创建一个RRefType对象，包含解析出的子类型
    fake_value = real_value = c10::TypeFactory::create<RRefType>(subtype);
  } else if (L.cur().kind == TK_IDENT && L.cur().text() == "Tensor") {
    L.next();
    // 解析Tensor类型
    fake_value = real_value = c10::TypeFactory::get<TensorType>();
    // 解析可能存在的别名注解信息
    alias_info = parseAliasAnnotation();
  } else if (L.cur().kind == TK_IDENT && L.cur().text() == "Dict") {
    L.next();
    L.expect('(');
    // 解析字典类型的键类型和值类型
    auto key_type = parseType().first;
    L.expect(',');
    auto value_type = parseType().first;
    L.expect(')');
    // 解析可能存在的别名注解信息
    alias_info = parseAliasAnnotation();
    // 创建一个DictType对象，包含解析出的键类型和值类型
    fake_value = real_value =
        c10::TypeFactory::create<DictType>(key_type, value_type);
  } else if (L.cur().kind == TK_IDENT && L.cur().text() == "Union") {
    L.next();
    L.expect('(');
    // 解析联合类型中的多个类型
    std::vector<TypePtr> types;
    types.emplace_back(parseType().first);
    while (L.cur().kind != ')') {
      L.expect(',');
      types.emplace_back(parseType().first);
    }
    L.expect(')');
    // 解析可能存在的别名注解信息
    alias_info = parseAliasAnnotation();
    // 创建一个UnionType对象，包含解析出的多个类型
    fake_value = real_value =
        c10::TypeFactory::create<c10::UnionType>(std::move(types));
  } else if (
      complete_tensor_types && L.cur().kind == TK_IDENT &&
      parseTensorDType(L.cur().text())) {
    // 如果complete_tensor_types为真，并且当前token是合法的张量类型
    fake_value = real_value = parseRefinedTensor();
    // 解析可能存在的别名注解信息
    alias_info = parseAliasAnnotation();
  } else if (L.cur().kind == TK_IDENT && L.cur().text() == "__torch__") {
    // 如果当前token是'__torch__'
    L.next(); // 移动到下一个词法单元
    L.expect('.'); // 确保下一个词法单元是句号
    auto torch_tok = L.expect(TK_IDENT); // 期望一个标识符，表示torch命名空间
    if (torch_tok.text() != "torch") { // 检查标识符是否为"torch"
      throw ErrorReport(torch_tok.range)
          << "Expected classes namespace but got " << torch_tok.text();
    }
    L.expect('.'); // 确保下一个词法单元是句号
    auto classes_tok = L.expect(TK_IDENT); // 期望一个标识符，表示classes命名空间
    if (classes_tok.text() != "classes") { // 检查标识符是否为"classes"
      throw ErrorReport(classes_tok.range)
          << "Expected classes namespace but got " << classes_tok.text();
    }
    L.expect('.'); // 确保下一个词法单元是句号
    auto ns_tok = L.expect(TK_IDENT); // 期望一个标识符，表示命名空间
    L.expect('.'); // 确保下一个词法单元是句号
    auto class_tok = L.expect(TK_IDENT); // 期望一个标识符，表示类名
    fake_value = real_value = getCustomClass(
        std::string("__torch__.torch.classes.") + ns_tok.text() + "." +
        class_tok.text()); // 获取自定义类的值，并设置为fake_value和real_value
    if (!fake_value) { // 如果未找到自定义类
      throw ErrorReport(class_tok.range)
          << "Unknown custom class type "
          << ns_tok.text() + "." + class_tok.text()
          << ". Please ensure it is registered.";
    }
  } else {
    real_value = parseBaseType(); // 解析基本类型
    if (real_value->kind() == ScalarTypeType::Kind ||
        real_value->kind() == MemoryFormatType::Kind ||
        real_value->kind() == LayoutType::Kind ||
        real_value->kind() == SymIntType::Kind) { // 检查真实值的类型种类
      fake_value = c10::TypeFactory::get<IntType>(); // 设置fake_value为整数类型
    } else {
      fake_value = real_value; // 否则，fake_value等于real_value
    }
    alias_info = parseAliasAnnotation(); // 解析别名注解信息
  }
  while (true) { // 进入循环，直到条件为false才退出
    if (L.cur().kind == '[' && L.lookahead().kind == ']') { // 如果当前和下一个词法单元是左右方括号
      L.next(); // 移动到下一个词法单元（左方括号）
      L.next(); // 再移动到下一个词法单元（右方括号）
      fake_value = c10::TypeFactory::create<ListType>(fake_value); // 创建fake_value的列表类型
      real_value = c10::TypeFactory::create<ListType>(real_value); // 创建real_value的列表类型
      auto container = parseAliasAnnotation(); // 解析别名注解信息到container
      if (alias_info) { // 如果有别名信息
        if (!container) { // 如果容器为空
          container = std::optional<AliasInfo>(AliasInfo()); // 创建一个空的别名信息容器
          container->setIsWrite(alias_info->isWrite()); // 设置写入状态
        }
        container->addContainedType(std::move(*alias_info)); // 添加包含的类型信息到容器
      }
      alias_info = std::move(container); // 移动容器到alias_info
    } else if (L.nextIf('?')) { // 如果下一个词法单元是问号
      fake_value = c10::OptionalType::get(fake_value); // 创建fake_value的可选类型
      real_value = c10::OptionalType::get(real_value); // 创建real_value的可选类型
    } else {
      break; // 否则跳出循环
    }
  }
  return std::make_tuple(
      std::move(fake_value), std::move(real_value), std::move(alias_info)); // 返回移动后的fake_value、real_value和alias_info元组
}

void SchemaTypeParser::parseList(
    int begin,                          // 参数：列表开始标记的类型
    int sep,                            // 参数：列表项之间的分隔符类型
    int end,                            // 参数：列表结束标记的类型
    c10::function_ref<void()> callback) {  // 参数：回调函数，处理每个列表项

  auto r = L.cur().range;               // 获取当前词法分析器中当前词法单元的范围
  if (begin != TK_NOTHING)              // 如果开始标记不是空标记，则期望下一个词法单元是开始标记
    L.expect(begin);
  
  if (L.cur().kind != end) {            // 如果当前词法单元的类型不是结束标记
    do {
      callback();                       // 执行回调函数处理当前列表项
    } while (L.nextIf(sep));            // 如果下一个词法单元是分隔符，则继续处理下一个列表项
  }

  if (end != TK_NOTHING)                // 如果结束标记不是空标记，则期望下一个词法单元是结束标记
    L.expect(end);
}

} // namespace torch::jit  // 结束命名空间 torch::jit
```