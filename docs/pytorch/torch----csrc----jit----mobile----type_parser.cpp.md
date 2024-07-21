# `.\pytorch\torch\csrc\jit\mobile\type_parser.cpp`

```py
// 包含 Torch 的移动端类型解析器头文件
#include <torch/csrc/jit/mobile/type_parser.h>

// 引入标准库中的队列
#include <queue>

// 引入 ATen 库中的类型定义和工厂函数
#include <ATen/core/jit_type.h>
#include <ATen/core/type_factory.h>

// 引入 C10 库中的字符串视图
#include <c10/util/string_view.h>

// 引入 Torch 前端解析器常量
#include <torch/csrc/jit/frontend/parser_constants.h>

// 引入 Torch 自定义类相关头文件
#include <torch/custom_class.h>

// 使用 torch::jit 命名空间下的 valid_single_char_tokens 函数
using torch::jit::valid_single_char_tokens;

// C10 命名空间
namespace c10 {

// 匿名命名空间，用于私有函数和静态变量
namespace {

// Torchbind 自定义类的类型前缀，作为 torchbind 自定义类类型的标识符
static constexpr const char* kTypeTorchbindCustomClass =
    "__torch__.torch.classes";

// NamedTuple 类型的字符串表示
static constexpr const char* kTypeNamedTuple = "NamedTuple";

// 检查字符是否为特殊字符
bool isSpecialChar(char a) {
  // 遍历 valid_single_char_tokens 中定义的有效单字符令牌
  for (const char* c = valid_single_char_tokens; *c; c++) {
    if (a == *c)
      return true;
  }
  return false;
}
} // namespace

// TypeParser 类的构造函数，接受一个 Python 字符串作为参数
TypeParser::TypeParser(std::string pythonStr)
    : pythonStr_(std::move(pythonStr)), start_(0) {
  // 调用 lex() 方法对字符串进行词法分析
  lex();
}

// TypeParser 类的构造函数，接受一个 Python 字符串向量作为参数
TypeParser::TypeParser(std::vector<std::string>& pythonStrs)
    : start_(0), pythonStrs_(pythonStrs) {}

// 解析 Python 字符串列表，返回类型指针向量
std::vector<TypePtr> TypeParser::parseList() {
  // 创建类型指针向量
  std::vector<TypePtr> typePtrs;
  // 将向量大小设置为 pythonStrs_ 的大小
  typePtrs.resize(pythonStrs_.size());

  // 定义 Torchbind 自定义类类型的限定名称前缀
  static const c10::QualifiedName classPrefix = "__torch__.torch.classes";

  // 遍历 pythonStrs_ 中的每个字符串
  for (size_t i = 0; i < pythonStrs_.size(); i++) {
    // 构造 QualifiedName 对象
    c10::QualifiedName qn(pythonStrs_[i]);

    // 定义类型指针
    c10::TypePtr type_ptr;

    // 如果类型限定名以 classPrefix 开头，则表示为 Torchbind 自定义类
    if (classPrefix.isPrefixOf(qn)) {
      // 获取 Torchbind 自定义类的类型指针
      type_ptr = torch::getCustomClass(qn.qualifiedName());

      // 检查类型指针是否为空，如果为空则抛出错误
      TORCH_CHECK(
          type_ptr,
          "The implementation of class ",
          qn.qualifiedName(),
          " cannot be found.");
    } else {
      // 否则，使用当前字符串进行解析
      pythonStr_ = pythonStrs_[i];
      start_ = 0;
      lex();
      type_ptr = parse();
    }

    // 将解析得到的类型指针存入 typePtrs 中
    typePtrs[i] = type_ptr;

    // 将类型指针与其字符串表示映射存入 str_type_ptr_map_
    str_type_ptr_map_[type_ptr->repr_str()] = type_ptr;
  }

  // 返回类型指针向量
  return typePtrs;
}

// 返回当前解析器支持的非简单类型集合
const std::unordered_set<std::string>& TypeParser::getNonSimpleType() {
  // 静态非简单类型集合
  static std::unordered_set<std::string> nonSimpleTypes{
      "List", "Optional", "Dict", "Tuple"};
  return nonSimpleTypes;
}

// 返回当前解析器支持的自定义类型集合
const std::unordered_set<std::string>& TypeParser::getCustomType() {
  // 静态自定义类型集合
  static std::unordered_set<std::string> customTypes{
      kTypeTorchbindCustomClass, kTypeNamedTuple};
  return customTypes;
}
// 返回包含已解析类型的无序集合
std::unordered_set<std::string> TypeParser::getContainedTypes() {
  return contained_types_;
}

// 解析单个元素类型的模板方法
template <typename T>
TypePtr TypeParser::parseSingleElementType() {
  // 期望下一个字符为 '['
  expectChar('[');
  // 解析当前类型，并用其创建动态类型对象
  auto result = DynamicTypeFactory::create<T>(parse());
  // 期望下一个字符为 ']'
  expectChar(']');
  // 返回解析的类型指针
  return result;
}

// 解析非简单类型的方法，根据给定的token进行解析
TypePtr TypeParser::parseNonSimple(const std::string& token) {
  if (token == "List") {
    // 解析 List 类型的单个元素类型
    return parseSingleElementType<ListType>();
  } else if (token == "Optional") {
    // 解析 Optional 类型的单个元素类型
    return parseSingleElementType<OptionalType>();
  } else if (token == "Dict") {
    // 期望下一个字符为 '['
    expectChar('[');
    // 解析字典类型的键
    auto key = parse();
    // 期望下一个字符为 ','
    expectChar(',');
    // 解析字典类型的值
    auto val = parse();
    // 期望下一个字符为 ']'
    expectChar(']');
    // 创建字典类型对象并返回
    return DynamicTypeFactory::create<DictType>(std::move(key), std::move(val));
  } else if (token == "Tuple") {
    // 解析元组类型
    std::vector<TypePtr> types;
    // 期望下一个字符为 '['
    expectChar('[');
    // 循环解析元组中的每个类型
    while (cur() != "]") {
      types.emplace_back(parse());
      // 如果当前字符不是 ']', 则期望下一个字符为 ','
      if (cur() != "]") {
        expectChar(',');
      }
    }
    // 期望下一个字符为 ']'
    expect("]");
    // 创建元组类型对象并返回
    return DynamicTypeFactory::create<TupleType>(std::move(types));
  }
  // 如果无法识别的token，返回空指针
  return nullptr;
}

// 解析类型的主方法，返回解析的类型指针
TypePtr TypeParser::parse() {
  // 获取下一个token
  std::string token = next();
  // 获取基本Python类型集合
  const auto& baseTypes = DynamicTypeFactory::basePythonTypes();
  // 查找是否为简单类型
  auto simpleTypeIt = baseTypes.find(token);
  if (simpleTypeIt != baseTypes.end()) {
    // 如果是简单类型，检查后续字符是否合法，并将类型加入已包含的类型集合中
    if (cur() != "]" && cur() != "," && !cur().empty()) {
      TORCH_CHECK(
          false, "Simple type ", token, " is followed by ", "invalid chars.");
    }
    contained_types_.insert(token);
    // 返回简单类型对应的动态类型对象
    return simpleTypeIt->second;
  } else if (getNonSimpleType().find(token) != getNonSimpleType().end()) {
    // 如果是非简单类型，则解析非简单类型
    contained_types_.insert(token);
    return parseNonSimple(token);
  } else if (token == "__torch__") {
    // 如果是 __torch__ 开头的类型
    expectChar('.');
    if (cur() == "torch") {
      // 如果后续是 'torch'，则解析 Torchbind 类型
      return parseTorchbindClassType();
    } else {
      // 否则解析自定义类型
      return parseCustomType();
    }
  } else if (token == "Union") {
    // 如果是 Union 类型，返回空指针，暂不支持
    // TODO: 对于嵌入式运行时不支持 Union 类型，需要为用户脚本生成编译错误
    return nullptr;
  } else {
    // 如果是未知的类型，抛出错误
    TORCH_CHECK(
        false,
        "Type ",
        token,
        " is not supported in the parser, ",
        "or the token is in wrong format.");
  }
  // 返回空指针
  return nullptr;
}

// 命名元组（NamedTuple）的自定义类型将遵循以下结构：
// "qualified_named[
//   NamedTuple, [
//       [filed_name_1, field_type_1],
//       [filed_name_2, field_type_2]
//   ]
// ]"
// 示例 NamedTuple 类型：
// "__torch__.base_models.sparse_nn.pytorch_preproc_types.PreprocOutputType[
//     NamedTuple, [
//         [float_features, Tensor],
//         [id_list_features, List[Tensor]],
//         [label,  Tensor],
//         [weight, Tensor],
//         ]
//     ]"
TypePtr TypeParser::parseNamedTuple(const std::string& qualified_name) {
    // 初始化存储字段名和字段类型的向量
    std::vector<c10::string_view> field_names;
    std::vector<TypePtr> field_types;
    // 初始化命名空间字符串
    std::string ns;
    // 期望当前符号为逗号
    expect(",");
    // 期望当前符号为左方括号
    expect("[");
    // 当当前符号不为右方括号时，循环执行以下操作
    while (cur() != "]") {
        // 期望当前符号为左方括号
        expect("[");
        // 获取下一个字段名视图
        auto field_name = nextView();
        // 期望当前符号为逗号
        expect(",");
        // 解析字段类型并存储
        TypePtr field_type = parse();
        // 将字段名和字段类型存入相应向量
        field_names.emplace_back(field_name);
        field_types.emplace_back(field_type);
        // 期望当前符号为右方括号
        expect("]");
        // 若当前符号为逗号，则移动到下一个符号
        if (cur() == ",") {
            next();
        }
    }
    // 调用工厂方法创建命名元组类型并返回
    return DynamicTypeFactory::createNamedTuple(
        qualified_name, field_names, field_types);
}

// 自定义类型将遵循以下结构:
// "qualified_named[
//   custom_type, [
//       [filed_name_1, field_type_1],
//       [filed_name_2, field_type_2]
//   ]
// ]"
TypePtr TypeParser::parseCustomType() {
    // 获取当前符号
    c10::string_view token = cur();
    // 初始化限定名字符串
    std::string qualified_name = "__torch__.";
    // 预留空间以便添加当前符号大小
    qualified_name.reserve(qualified_name.size() + token.size());
    // 添加当前符号的字符到限定名字符串
    qualified_name.append(token.begin(), token.end());
    // 移动到下一个符号
    next();
    // 当当前符号为"."时，继续添加限定名的后续部分
    while (cur() == ".") {
        qualified_name.append(next());
        qualified_name.append(next());
    }
    // 当cur()移动到限定名后的下一个符号，如果是"[", 表示自定义类型后面跟着它的类定义。
    // 否则，它是一个裸限定名，并需要查找str_type_ptr_map_来找到typeptr。
    if (cur() == "[") {
        next();
        // 获取下一个类型名
        std::string type_name = next();
        // 目前仅支持命名元组自定义类型，如果需要支持更多类型，在这里扩展。
        if (type_name == kTypeNamedTuple) {
            // 插入已包含的类型
            contained_types_.insert(kTypeNamedTuple);
            // 解析命名元组并返回
            return parseNamedTuple(qualified_name);
        } else {
            // 抛出错误，不支持的自定义类型
            TORCH_CHECK(
                false, "Custom Type ", type_name, " is not supported in the parser.");
        }
    } else {
        // 查找限定名在映射中的类型指针
        auto find_type = str_type_ptr_map_.find(qualified_name);
        if (find_type != str_type_ptr_map_.end()) {
            // 如果找到，返回对应的类型指针
            return find_type->second;
        } else {
            // 当找不到类型定义时，可能有两个原因：
            // 1. bytecode.pkl中的类型列表顺序不正确
            // 2. bytecode.pkl类型表中不存在此自定义类型定义
            TORCH_CHECK(
                false, "Can't find definition for the type: ", qualified_name);
        }
        // 返回空指针
        return nullptr;
    }
}

TypePtr TypeParser::parseTorchbindClassType() {
    // 静态预期原子数组
    static constexpr std::array<const char*, 4> expected_atoms = {
        "torch", ".", "classes", "."};
    // 对于预期原子数组中的每个原子，期望当前符号为之
    for (const auto& atom : expected_atoms) {
        expect(atom);
    }
    // 获取下一个符号作为命名空间
    std::string ns = next();
    // 期望当前符号为'.'
    expectChar('.');
    // 获取下一个符号作为类名
    std::string classname = next();
    // 初始化自定义类名字符串
    std::string customClassName = "__torch__.torch.classes.";
    // 预留空间以便添加命名空间和类名
    customClassName.reserve(
        customClassName.size() + ns.size() + 1 + classname.size());
    // 添加命名空间到自定义类名字符串
    customClassName.append(ns);
    customClassName.push_back('.');
    // 添加类名到自定义类名字符串
    customClassName.append(classname);
    // 调用torch::getCustomClass方法获取自定义类类型并返回
    return torch::getCustomClass(customClassName);
}
// 函数定义：检查当前位置的 token 是否与给定的字符串 s 相同，若不同则抛出错误
void TypeParser::expect(const char* s) {
  // 获取当前位置的 token
  c10::string_view token = cur();
  // 使用 TORCH_CHECK 检查当前 token 是否等于 s，若不等则抛出错误，显示错误信息和当前 token
  TORCH_CHECK(
      token == s,
      "Error when parsing type ",
      pythonStr_,
      ": Expect ",
      s,
      ", but get ",
      token);
  // 前进到下一个 token
  advance();
}

// 函数定义：检查当前位置的 token 是否为指定的字符 c，若不是则抛出错误
// 特化比较函数，用于优化单字符比较性能
void TypeParser::expectChar(char c) {
  // 获取当前位置的 token
  c10::string_view token = cur();
  // 使用 TORCH_CHECK 检查当前 token 是否是长度为 1 且等于 c 的字符，若不是则抛出错误，显示错误信息和当前 token
  TORCH_CHECK(
      token.size() == 1 && token[0] == c,
      "Error when parsing type ",
      pythonStr_,
      ": Expect ",
      c,
      ", but get ",
      token);
  // 前进到下一个 token
  advance();
}

// 函数定义：词法分析，跳过空格并识别下一个 token
void TypeParser::lex() {
  // 跳过空格
  while (start_ < pythonStr_.size() && pythonStr_[start_] == ' ')
    ++start_;
  // 如果还有未处理的字符
  if (start_ < pythonStr_.size()) {
    // 如果当前字符是特殊字符
    if (isSpecialChar(pythonStr_[start_])) {
      // 下一个 token 是当前字符
      next_token_ = c10::string_view(pythonStr_.data() + start_++, 1);
    } else { // 一个单词
      size_t end = start_;
      // 找到单词的结束位置，即下一个特殊字符或空格
      for (; end < pythonStr_.size() && !isSpecialChar(pythonStr_[end]) &&
           pythonStr_[end] != ' ';
           ++end)
        ;
      // 提取单词作为下一个 token
      next_token_ = c10::string_view(pythonStr_.data() + start_, end - start_);
      // 更新 start_ 到单词结束的位置
      start_ = end;
    }
  }
}

// 函数定义：返回下一个 token 的 c10::string_view 视图
c10::string_view TypeParser::nextView() {
  // 检查下一个 token 是否为空，若为空则抛出错误
  TORCH_CHECK(
      !next_token_.empty(),
      "Empty token queue in mobile type parser.",
      "Check the format of the type string and make sure it's correct.");
  // 获取当前 token 的视图
  c10::string_view token = cur();
  // 前进到下一个 token
  advance();
  // 返回当前 token 的视图
  return token;
}

// 函数定义：返回下一个 token 的字符串表示
std::string TypeParser::next() {
  // 获取下一个 token 的视图
  auto token = nextView();
  // 转换为 std::string 并返回
  return std::string(token.begin(), token.end());
}

// 函数定义：前进到下一个 token
void TypeParser::advance() {
  // 清空下一个 token
  next_token_ = "";
  // 执行词法分析，获取下一个 token
  lex();
}

// 函数定义：返回当前 token 的 c10::string_view 视图
C10_NODISCARD c10::string_view TypeParser::cur() const {
  // 返回当前 token 的视图
  return next_token_;
}

// 函数定义：解析给定的 Python 类型字符串，返回对应的 TypePtr
TORCH_API at::TypePtr parseType(const std::string& pythonStr) {
  // 创建 TypeParser 对象，并传入 Python 类型字符串进行初始化
  at::TypeParser parser(pythonStr);
  // 调用 TypeParser 的 parse 方法进行解析，并返回解析结果
  return parser.parse();
}

// 函数定义：解析多个 Python 类型字符串，返回对应的 TypePtr 向量
TORCH_API std::vector<at::TypePtr> parseType(
    std::vector<std::string>& pythonStrs) {
  // 创建 TypeParser 对象，并传入 Python 类型字符串向量进行初始化
  at::TypeParser parser(pythonStrs);
  // 调用 TypeParser 的 parseList 方法进行解析，并返回解析结果向量
  return parser.parseList();
}
```