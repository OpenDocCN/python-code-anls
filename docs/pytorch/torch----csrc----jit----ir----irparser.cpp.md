# `.\pytorch\torch\csrc\jit\ir\irparser.cpp`

```py
#include <torch/csrc/jit/ir/irparser.h>
// 引入 Torch 的 IR 解析器头文件

#include <ATen/EmptyTensor.h>
#include <torch/csrc/jit/frontend/lexer.h>
#include <torch/csrc/jit/frontend/parse_string_literal.h>
#include <torch/csrc/jit/frontend/schema_type_parser.h>
#include <torch/csrc/jit/ir/ir.h>
// 引入其他必要的头文件，包括 ATen 库和 Torch 的前端解析器等

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_strided.h>
#endif
// 根据宏定义决定是否引入 ATen 的函数头文件或者分别引入空张量相关的头文件

#include <string>
#include <vector>
// 引入标准库的 string 和 vector 头文件

namespace torch::jit {

struct VarWithType;
struct ParsedLiteral;
// 声明 VarWithType 和 ParsedLiteral 结构体

class IRParser {
  friend void parseIR(
      const std::string& str,
      torch::jit::Graph* graph,
      std::unordered_map<std::string, Value*>& vmap,
      bool parse_tensor_constants);
  // IRParser 类，负责解析 Torch 的图形 IR

  IRParser(
      const std::string& str,
      torch::jit::Graph* graph,
      std::unordered_map<std::string, Value*>& vmap,
      bool parse_tensor_constants)
      : L(std::make_shared<Source>(str)),
        g(graph),
        vmap(vmap),
        type_parser(
            L,
            /*parse_complete_tensor_types*/ true,
            /*allow_type_vars*/ true),
        parse_tensor_constants_(parse_tensor_constants) {}
  // IRParser 构造函数，初始化成员变量和解析器

  std::string parseVar();
  VarWithType parseVarWithType(bool allow_optional = false);
  ParsedLiteral parseScalarLiteral(Node* n);
  // 成员函数声明，用于解析变量、字面量和标量字面量

  void parse();
  void parseGraphInputs();
  void parseReturnOperator();
  // 成员函数声明，用于解析图、输入和返回操作符

  void parseBlocks(Node* parentNode);
  void parseBlock(Node* parentNode);
  void parseBlockInputs(Block* b);
  void parseBlockOutputs(Block* b);
  // 成员函数声明，用于解析块和块的输入输出

  void parseOperatorsList(Block* b);
  void parseOperator(Block* b);
  void parseOperatorOutputs(std::vector<VarWithType>* outs);
  std::string parseOperatorName();
  void parseOperatorInputs(Node* n);
  void parseAttrs(Node* n);
  void parseAttr(Node* n);
  // 成员函数声明，用于解析操作符、操作符的输入输出和属性

  void parseList(
      int begin,
      int sep,
      int end,
      const std::function<void()>& callback);
  // 解析列表的成员函数声明

  void bypassTypeAnnotationList();
  // 跳过类型注解列表的成员函数声明

  Value* findValueInVMap(const std::string& name);
  // 在值映射中查找值的成员函数声明

  torch::jit::Lexer L;
  torch::jit::Graph* g = nullptr;
  std::unordered_map<std::string, Value*>& vmap;
  SchemaTypeParser type_parser;
  bool parse_tensor_constants_;
  std::vector<Node*> deferred_tensor_value_initializations_;
  std::vector<Node*> deferred_empty_container_initializations_;
  // IRParser 的成员变量声明
};

struct ParsedLiteral {
  ParsedLiteral() = default;
  // 字面量解析结构体

  AttributeKind k = AttributeKind::t;

  int64_t i = 0;
  std::string s = "";
  double f = 0.0;
  c10::complex<double> c = c10::complex<double>(0, 0);
  TypePtr ty;
  std::vector<int64_t> is;
  std::vector<std::string> ss;
  std::vector<double> fs;
  std::vector<c10::complex<double>> cs;
  std::vector<TypePtr> tys;
  // 字面量解析结构体的成员变量声明
};

struct VarWithType {
  VarWithType() = default;
  // 变量与类型结构体

  std::string name;
  TypePtr type;
  // 变量名称和类型指针的成员变量声明
};

void parseIR(
    const std::string& str,
    torch::jit::Graph* graph,
    std::unordered_map<std::string, Value*>& vmap,
    bool parse_tensor_constants) {
  torch::jit::IRParser p(str, graph, vmap, parse_tensor_constants);
  p.parse();
}
// 解析 IR 的全局函数声明及定义
    torch::jit::Graph* graph,
    bool parse_tensor_constants) {


// 接收一个 torch::jit::Graph 指针和一个布尔值作为参数
std::unordered_map<std::string, Value*> vmap;
// 创建一个无序映射，用于存储字符串到 torch::jit::Value* 的映射关系
parseIR(str, graph, vmap, parse_tensor_constants);
// 调用 parseIR 函数，解析输入的 str 字符串，填充 graph 中的节点和边，同时更新 vmap 映射
// 解析带有变量和类型的结构体，allow_optional 控制是否允许类型为 nullptr
VarWithType IRParser::parseVarWithType(bool allow_optional) {
  VarWithType r;  // 创建 VarWithType 结构体对象 r
  r.name = parseVar();  // 解析变量名并赋值给 r 的 name 成员
  if (allow_optional) {
    r.type = nullptr;  // 如果允许可选类型，则 r 的 type 成员设为 nullptr
  } else {
    r.type = TensorType::get();  // 否则，设置 r 的 type 成员为 TensorType 类型对象
  }
  if (L.nextIf(':')) {
    auto type_alias = type_parser.parseType();  // 如果下一个 token 是 ':', 则解析类型别名
    AT_ASSERTM(!type_alias.second, "Parsing IR with Alias Info not handled");
    r.type = type_alias.first;  // 将解析得到的类型赋给 r 的 type 成员
  }
  return r;  // 返回解析完成的 VarWithType 结构体对象
}

// 解析变量名，以 '%' 开头
std::string IRParser::parseVar() {
  L.expect('%');  // 确保当前 token 是 '%' 符号
  std::string name;  // 创建空字符串用于存储变量名
  bool continue_parsing;  // 控制是否继续解析
  do {
    if (L.cur().kind == TK_IDENT) {
      name += L.expect(TK_IDENT).text();  // 如果当前 token 是标识符，则将其添加到 name 中
    } else {
      name += L.expect(TK_NUMBER).text();  // 否则，将当前 token 视为数字，并将其文本添加到 name 中
    }
    continue_parsing = false;  // 默认不继续解析
    if (L.nextIf('.')) {
      continue_parsing = true;  // 如果下一个 token 是 '.', 则继续解析
      name += '.';  // 将 '.' 添加到 name 中
    } else if (L.cur().kind == TK_NUMBER && L.cur().text()[0] == '.') {
      continue_parsing = true;  // 如果当前 token 是以 '.' 开头的数字，则继续解析
    }
  } while (continue_parsing);  // 当需要继续解析时循环
  return name;  // 返回解析完成的变量名字符串
}

// 解析操作符的输出变量列表
void IRParser::parseOperatorOutputs(std::vector<VarWithType>* outs) {
  if (L.cur().kind != '%') {
    return;  // 如果当前 token 不是 '%' 符号，则直接返回
  }
  parseList(TK_NOTHING, ',', TK_NOTHING, [&] {
    outs->push_back(parseVarWithType(true));  // 解析变量和类型，并添加到输出变量列表中
  });
  L.expect('=');  // 确保接下来的 token 是 '=' 符号
}

// 解析字符串或数值字面量，并返回其值和类型
ParsedLiteral IRParser::parseScalarLiteral(Node* n) {
  auto token = L.cur();  // 获取当前 token
  std::string str;  // 用于存储解析出的字符串
  std::pair<TypePtr, std::optional<c10::AliasInfo>> type_alias;  // 存储解析出的类型及其别名信息
  ParsedLiteral r;  // 创建 ParsedLiteral 结构体对象 r
  switch (token.kind) {
    case TK_STRINGLITERAL:
      r.k = AttributeKind::s;  // 如果当前 token 是字符串字面量，则设置 r 的类型为字符串
      r.s = parseStringLiteral(token.range, token.text());  // 解析字符串字面量并赋值给 r 的 s 成员
      L.next();  // 移动到下一个 token
      return r;  // 返回解析完成的 ParsedLiteral 结构体对象
    case '-':
      str = "-";  // 如果当前 token 是 '-' 符号，则初始化 str 为 "-"
      L.next();  // 移动到下一个 token
      if (L.cur().kind != TK_NUMBER) {
        throw ErrorReport(token.range)
            << "Expected a number after '-' but got:" << token.text();  // 抛出错误，期望在 '-' 后是数字
      }
      [[fallthrough]];  // 继续执行下一个 case
    case TK_NUMBER:
      // 如果当前 token 是数字，将其文本添加到字符串 str 中
      str += L.cur().text();
      // 如果字符串中包含 'j'，则认为是复数
      if (str.find('j') != std::string::npos) {
        r.k = AttributeKind::c;
        double imag = 0.0f;
        // 尝试将去除 'j' 后的部分转换为 double 类型作为虚部
        try {
          imag = std::stod(str.substr(0, str.size() - 1));
        } catch (const std::invalid_argument& e) {
          throw ErrorReport(token.range)
              << "Number cannot be converted to double";
        } catch (const std::out_of_range& e) {
          throw ErrorReport(token.range)
              << "Number is too long to be represented in type double";
        }
        // 创建复数对象
        r.c = c10::complex<double>(0, imag);
      } else if (
          // 如果字符串中包含 '.' 或 'e'，则认为是浮点数
          str.find('.') != std::string::npos ||
          str.find('e') != std::string::npos) {
        r.k = AttributeKind::f;
        // 尝试将字符串转换为 double 类型
        try {
          r.f = std::stod(str);
        } catch (const std::invalid_argument& e) {
          throw ErrorReport(token.range)
              << "Number cannot be converted to double";
        } catch (const std::out_of_range& e) {
          throw ErrorReport(token.range)
              << "Number is too long to be represented in type double";
        }
      } else {
        // 否则认为是整数
        r.k = AttributeKind::i;
        // 尝试将字符串转换为 long long 类型
        try {
          r.i = std::stoll(str);
        } catch (const std::invalid_argument& e) {
          throw ErrorReport(token.range)
              << "Number cannot be converted to integer";
        } catch (const std::out_of_range& e) {
          throw ErrorReport(token.range) << "Number is too big";
        }
      }
      // 移动到下一个 token
      L.next();
      // 返回解析结果
      return r;

    case TK_IDENT:
      // 如果当前 token 是标识符，认为是类型字面量
      r.k = AttributeKind::ty;
      // 解析类型
      type_alias = type_parser.parseType();
      AT_ASSERTM(!type_alias.second, "Parsing IR with Alias Info not handled");
      // 设置类型信息
      r.ty = type_alias.first;
      // 返回解析结果
      return r;

    case '<': {
      // 处理 '<' 开始的情况
      L.next();
      // 预期下一个 token 是标识符
      auto text = L.expect(TK_IDENT);
      // 如果不是 "Tensor"，报错
      if (text.text() != "Tensor") {
        throw ErrorReport(token.range)
            << "Could not parse literal" << token.text();
      }
      // 如果不允许解析张量常量，报错
      if (!parse_tensor_constants_) {
        throw ErrorReport(token.range)
            << "Tensor constant encountered but `parse_tensor_constants` set to false"
            << token.text();
      }
      // 预期 '>' 结束
      L.expect('>');
      // 将当前解析的位置添加到延迟张量值初始化列表中
      deferred_tensor_value_initializations_.push_back(n);
      // 设置属性为张量类型
      r.k = AttributeKind::t;
      // 返回解析结果
      return r;
    }

    case '{': {
      // 处理 '{' 开始的情况
      L.next();
      // 如果下一个 token 是 '-'，跳过
      if (L.cur().kind == '-') {
        L.next();
      }
      // 预期下一个 token 是数字
      auto text = L.expect(TK_NUMBER);
      // 如果不允许解析张量常量，报错
      if (!parse_tensor_constants_) {
        throw ErrorReport(token.range)
            << "Single-element tensor constant encountered but "
            << "`parse_tensor_constants` is set to false " << token.text();
      }
      // 预期 '}' 结束
      L.expect('}');
      // 将当前解析的位置添加到延迟张量值初始化列表中
      deferred_tensor_value_initializations_.push_back(n);
      // 设置属性为张量类型
      r.k = AttributeKind::t;
      // 返回解析结果
      return r;
    }

    default:
      // 对于其他情况，报错无法解析字面量
      throw ErrorReport(token.range)
          << "Could not parse literal" << token.text();
/** \brief 跳过类型注解列表的解析。
 *
 * 此函数用于跳过 IR 中的类型注解列表。在解析过程中，会检查当前符号的类型，
 * 如果是 '['，则认为进入了类型注解列表，增加深度计数；如果是 ']'，则表示
 * 离开了一个类型注解列表，深度计数减少。函数执行完毕后，确保所有类型注解
 * 列表均已跳过。
 */
void IRParser::bypassTypeAnnotationList() {
  int depth = 0;  // 初始化深度计数为0
  bool bypassed_list = false;  // 是否已经跳过类型注解列表的标志
  while (depth != 0 || !bypassed_list) {  // 循环直到所有类型注解列表均已跳过
    if (L.cur().kind == '[') {  // 如果当前符号为 '['
      bypassed_list = true;  // 标记已经跳过类型注解列表
      depth++;  // 深度计数加1
    } else if (L.cur().kind == ']') {  // 如果当前符号为 ']'
      depth--;  // 深度计数减1，表示离开一个类型注解列表
    }
    L.next();  // 移动到下一个符号
  }
}

/** \brief 解析属性并将其添加到节点 N 中。
 *
 * 该函数确定属性类型（字符串、整数、浮点数、复数、字符串列表、整数列表、
 * 浮点数列表、复数列表以及张量列表（目前仅支持空列表））。属性的形式如下：
 * AttrName=AttrValue，其中 AttrValue 可以是标量文字或列表，例如：
 * size = 27
 * name = "Bob"
 * coefs = [1.2, 3.4, 0.6]
 */
void IRParser::parseAttr(Node* n) {
  std::string attrname = L.expect(TK_IDENT).text();  // 期望并获取属性名
  L.expect('=');  // 期望并跳过 '=' 符号

  if (L.cur().kind == '[') {  // 如果当前符号为 '['，表示属性值为列表
    // 处理列表类型的属性值
    AttributeKind k = AttributeKind::ts;  // 属性值类型初始化为 'ts'
    c10::List<int64_t> is;  // 整数列表
    c10::List<std::string> ss;  // 字符串列表
    c10::List<double> fs;  // 浮点数列表
    c10::List<c10::complex<double>> cs;  // 复数列表
    std::vector<TypePtr> tys;  // 类型指针列表
    int elem_num = 0;  // 元素计数器初始化为0

    // 解析列表中的每个元素
    parseList('[', ',', ']', [&] {
      ParsedLiteral r = parseScalarLiteral(n);  // 解析标量文字
      switch (r.k) {
        case AttributeKind::s:
          ss.push_back(r.s);  // 将解析得到的字符串添加到字符串列表
          AT_ASSERT(!elem_num++ || k == AttributeKind::ss);  // 断言保证类型一致性
          k = AttributeKind::ss;  // 更新属性值类型为字符串列表
          break;
        case AttributeKind::i:
          is.push_back(r.i);  // 将解析得到的整数添加到整数列表
          AT_ASSERT(!elem_num++ || k == AttributeKind::is);  // 断言保证类型一致性
          k = AttributeKind::is;  // 更新属性值类型为整数列表
          break;
        case AttributeKind::f:
          fs.push_back(r.f);  // 将解析得到的浮点数添加到浮点数列表
          AT_ASSERT(!elem_num++ || k == AttributeKind::fs);  // 断言保证类型一致性
          k = AttributeKind::fs;  // 更新属性值类型为浮点数列表
          break;
        case AttributeKind::c:
          cs.push_back(r.c);  // 将解析得到的复数添加到复数列表
          AT_ASSERT(!elem_num++ || k == AttributeKind::cs);  // 断言保证类型一致性
          k = AttributeKind::cs;  // 更新属性值类型为复数列表
          break;
        case AttributeKind::ty:
          tys.push_back(r.ty);  // 将解析得到的类型指针添加到类型指针列表
          AT_ASSERT(!elem_num++ || k == AttributeKind::tys);  // 断言保证类型一致性
          k = AttributeKind::tys;  // 更新属性值类型为类型指针列表
          break;
        default:
          throw ErrorReport(L.cur().range) << "Unexpected attr type";  // 抛出异常，表示意外的属性类型
      }
    });

    // 根据属性值类型 k，将属性值设置到节点 N 的对应属性中
    switch (k) {
      case AttributeKind::ts:
        n->ival_(Symbol::attr(attrname), IValue());  // 设置空值
        break;
      case AttributeKind::ss:
        n->ival_(Symbol::attr(attrname), IValue(ss));  // 设置字符串列表
        break;
      case AttributeKind::fs:
        n->ival_(Symbol::attr(attrname), IValue(fs));  // 设置浮点数列表
        break;
      case AttributeKind::cs:
        n->ival_(Symbol::attr(attrname), IValue(cs));  // 设置复数列表
        break;
      case AttributeKind::is:
        n->ival_(Symbol::attr(attrname), IValue(is));  // 设置整数列表
        break;
      case AttributeKind::tys:
        n->tys_(Symbol::attr(attrname), tys);  // 设置类型指针列表
        break;
      default:
        throw ErrorReport(L.cur().range) << "Unexpected attr type";  // 抛出异常，表示意外的属性类型
    }
  } else if (L.cur().text() == "annotate") {  // 如果当前符号为 "annotate"
    L.next();  // 跳过 "annotate"
    L.expect('(');  // 期望并跳过 '('
    auto type = L.cur().text();  // 获取类型名
    // 检查注解类型是否为 "List" 或 "Dict"，如果不是则抛出错误报告
    if (type != "List" && type != "Dict") {
      throw ErrorReport(L.cur().range)
          << "Unexpected annotation (only List and Dict can be parsed)";
    }
    // 移动到下一个词法单元
    L.next();
    // 忽略 IValue 常量上的注解，从 Node 输出中恢复类型
    // 注意：也可以使用 script_type_parser
    bypassTypeAnnotationList();
    // 期望遇到逗号分隔符
    L.expect(',');
    // 预期一个空定义（注意 - 这并非总是成立）
    if (type == "Dict") {
      // 预期一个字典定义
      L.expect('{');
      L.expect('}');
    } else if (type == "List") {
      // 预期一个列表定义
      L.expect('[');
      L.expect(']');
    }
    // 预期一个右括号
    L.expect(')');
    // 将当前节点 n 添加到延迟空容器初始化列表中
    deferred_empty_container_initializations_.push_back(n);
  } else {
    // 处理标量情况
    // 解析标量字面量 r
    ParsedLiteral r = parseScalarLiteral(n);
    switch (r.k) {
      case AttributeKind::s:
        // 将字符串属性 attrname 关联到节点 n 上
        n->s_(Symbol::attr(attrname), r.s);
        break;
      case AttributeKind::i:
        // 将整数属性 attrname 关联到节点 n 上
        n->i_(Symbol::attr(attrname), r.i);
        break;
      case AttributeKind::f:
        // 将浮点数属性 attrname 关联到节点 n 上
        n->f_(Symbol::attr(attrname), r.f);
        break;
      case AttributeKind::c:
        // 将字符属性 attrname 关联到节点 n 上
        n->c_(Symbol::attr(attrname), r.c);
        break;
      case AttributeKind::ty:
        // 将类型属性 attrname 关联到节点 n 上
        n->ty_(Symbol::attr(attrname), r.ty);
        break;
      case AttributeKind::t:
        // 暂时不做任何初始化，稍后使用随机数据初始化
        break;
      default:
        // 抛出错误报告，表示遇到了意外的属性类型
        throw ErrorReport(L.cur().range) << "Unexpected attr type";
    }
    // 返回，结束函数执行
    return;
  }
}

/** \brief Parse attributes for a node.
 *
 * This function parses attributes enclosed in square brackets for a given node.
 * It uses a lambda function to parse each attribute.
 */
void IRParser::parseAttrs(Node* n) {
  // Call parseList to parse attributes within square brackets
  parseList('[', ',', ']', [&] { parseAttr(n); });
}

/** \brief Parse operator inputs for a node.
 *
 * This function checks if the current token is '[', indicating attribute parsing,
 * and calls parseAttrs if true. Otherwise, it parses inputs enclosed in parentheses.
 */
void IRParser::parseOperatorInputs(Node* n) {
  if (L.cur().kind == '[') {
    parseAttrs(n);
  }
  // Parse inputs enclosed in parentheses
  parseList('(', ',', ')', [&] {
    std::string var_name = parseVar();
    n->addInput(findValueInVMap(var_name));
  });
}

/** \brief Parse blocks within a node.
 *
 * This function expects an indented block of code and parses each block using parseBlock.
 */
void IRParser::parseBlocks(Node* parentNode) {
  // Expect an indentation to begin parsing blocks
  L.expect(TK_INDENT);
  // Continue parsing blocks until a dedent token is encountered
  while (L.cur().kind != TK_DEDENT) {
    parseBlock(parentNode);
  }
  // Expect a dedent token to end block parsing
  L.expect(TK_DEDENT);
}

/** \brief Parse block inputs.
 *
 * This function parses inputs for a block enclosed in parentheses.
 * It assigns a unique name if the variable name is valid.
 */
void IRParser::parseBlockInputs(Block* b) {
  // Parse inputs enclosed in parentheses
  parseList('(', ',', ')', [&] {
    VarWithType v = parseVarWithType();
    // Assign a unique name to the variable if it's valid
    std::string uniq_name = Value::isValidName(v.name) ? v.name : "";
    vmap[v.name] = b->addInput(uniq_name);
    vmap[v.name]->setType(v.type);
  });
}

/** \brief Parse block outputs.
 *
 * This function expects an arrow token followed by a list of outputs enclosed in parentheses.
 * It parses each output and registers it with the block.
 */
void IRParser::parseBlockOutputs(Block* b) {
  // Expect an arrow token before parsing outputs
  L.expect(TK_ARROW);
  // Parse outputs enclosed in parentheses
  parseList('(', ',', ')', [&] {
    std::string var_name = parseVar();
    b->registerOutput(findValueInVMap(var_name));
  });
  // Expect a newline after parsing outputs
  L.expect(TK_NEWLINE);
  // Expect a dedent token after parsing block outputs
  L.expect(TK_DEDENT);
}

/** \brief Parse a block of statements.
 *
 * This function parses a block of statements that define a block in the graph.
 * It parses block inputs, operators list, and block outputs sequentially.
 */
void IRParser::parseBlock(Node* parentNode) {
  // Create a new block under the given parent node
  Block* b = parentNode->addBlock();
  // Expect an identifier (block name), but it's not used further
  L.expect(TK_IDENT).text(); // Block name is not used anywhere.
  // Parse inputs for the block
  parseBlockInputs(b);
  // Expect a colon token after parsing block inputs
  L.expect(':');
  // Parse operators list within the block
  parseOperatorsList(b);
  // Parse outputs for the block
  parseBlockOutputs(b);
}

/** \brief Parse a list of operators within a block.
 *
 * This function expects an indented block of operator statements and parses each operator.
 * It continues parsing until it encounters an arrow or return token.
 */
void IRParser::parseOperatorsList(Block* b) {
  // Expect an indentation to begin parsing operators
  L.expect(TK_INDENT);
  // Continue parsing operators until an arrow or return token is encountered
  while (L.cur().kind != TK_ARROW && L.cur().kind != TK_RETURN) {
    parseOperator(b);
  }
}

/** \brief Parse the name of an operator.
 *
 * This function parses and constructs the qualified name of an operator,
 * which consists of one or more identifiers separated by double colons (::).
 */
std::string IRParser::parseOperatorName() {
  // Expect an identifier (operator name) followed by '::'
  std::string name = L.expect(TK_IDENT).text();
  L.expect(':');
  L.expect(':');
  // Append the second identifier to form the qualified operator name
  name += "::" + L.expect(TK_IDENT).text();
  return name;
}

/** \brief Parse a single operator statement.
 *
 * This function parses a single operator statement in the form of:
 * <outputs> = NodeName[<attributes>](<inputs>)
 * It creates the corresponding node in the graph and registers its outputs.
 */
void IRParser::parseOperator(Block* b) {
  // Parse left-hand side (outputs)
  std::vector<VarWithType> outs;
  parseOperatorOutputs(&outs);

  // Parse the name of the operator and create the corresponding node in the graph
  auto source_range = L.cur().range;
  std::string name = parseOperatorName();
  Node* n = g->create(Symbol::fromQualString(name), {}, outs.size())
                ->setSourceRange(source_range);

  // Parse attributes and inputs for the operator node
  parseOperatorInputs(n);

  // Retrieve the function schema associated with the node, if available
  const FunctionSchema* schema = n->maybeSchema();

  // Register outputs of the operator node with their corresponding variables
  unsigned idx = 0;
  for (const VarWithType& v : outs) {
    vmap[v.name] = n->outputs()[idx];
    idx++;
  }
}
    // 检查 schema 是否存在且返回值不是变长的情况
    if (schema && !schema->is_varret()) {
      // 检查返回值列表的大小是否大于索引 idx
      TORCH_CHECK(
          schema->returns().size() > idx,
          "Operator parsing error: out of bounds access at ",
          idx,
          " to schema->returns() which size is ",
          schema->returns().size(),
          " in size");
      // 获取 schema 返回值列表中索引为 idx 的返回类型
      auto schema_return_type = schema->returns().at(idx).type();
      // 如果 v 没有指定类型，则设置其类型为 schema_return_type
      if (!v.type) {
        vmap[v.name]->setType(schema_return_type);
      } else {
        // 当前不支持对类型变量的检查
        // TODO: 支持类型变量检查？
        // 如果 schema_return_type 没有自由变量并且 v.type 是 schema_return_type 的子类型，则设置 vmap[v.name] 的类型为 v.type
        if (!schema_return_type->hasFreeVariables() &&
            !v.type->isSubtypeOf(*schema_return_type)) {
          // 抛出类型不匹配的错误信息，包括详细信息和出错的源代码位置
          throw ErrorReport(source_range)
              << "Annotated type " << v.type->repr_str()
              << " does not match schema type "
              << schema_return_type->repr_str() << " for operator " << *schema;
        }
        // 否则，设置 vmap[v.name] 的类型为 v.type
        vmap[v.name]->setType(v.type);
      }
    } else {
      // 如果 schema 不存在或返回值是变长的，则将 vmap[v.name] 的类型设置为 v.type 或默认的 TensorType::get()
      vmap[v.name]->setType(v.type ? v.type : TensorType::get());
    }
    // 索引增加
    idx++;
  }

  // 将新节点 n 插入到块 B 中
  b->appendNode(n);

  // 如果语句有嵌套的块，解析这些块
  if (L.cur().kind == TK_INDENT) {
    parseBlocks(n);
  }
  // 移动到下一个 token，如果是换行符则继续移动
  L.nextIf(TK_NEWLINE);
}

void IRParser::parseGraphInputs() {
  // 解析以 '(' 开始、',' 分隔、')' 结束的列表
  parseList('(', ',', ')', [&] {
    // 解析带类型的变量
    VarWithType v = parseVarWithType();
    // 如果变量名有效，使用它；否则使用空字符串
    std::string uniq_name = Value::isValidName(v.name) ? v.name : "";
    // 将变量名映射到图的输入节点
    vmap[v.name] = g->addInput(uniq_name);
    // 设置节点的类型
    vmap[v.name]->setType(v.type);
  });
}

/** \brief 解析 return 语句。
 *
 * 语法应该如下所示：
 *   return (x : TypeX, y : TypeY, z, ...)
 */
void IRParser::parseReturnOperator() {
  L.expect(TK_RETURN);  // 解析 'return' 关键字

  // 解析输出名称和类型列表
  parseList('(', ',', ')', [&] {
    std::string var_name = parseVar();  // 解析变量名
    // 在 vmap 中查找变量对应的值，并将其注册为图的输出
    g->registerOutput(findValueInVMap(var_name));
  });

  // 消耗结尾的 token
  if (L.cur().kind != TK_EOF) {
    L.expect(TK_NEWLINE);  // 解析换行符
    L.expect(TK_DEDENT);   // 解析缩进减少符号
  }
}

/** \brief 解析整个图。
 *
 * 语法应该如下所示：
 *   graphName (input1, input2, ... inputN):
 *     op1
 *     op2
 *     ...
 *     opN
 *     return (output1, output2, ... outputN)
 */
void IRParser::parse() {
  // 解析图定义，应该如下所示：
  // graphName (input1, input2, ... inputN):
  std::string graphName = L.expect(TK_IDENT).text();  // 解析图的名称
  parseGraphInputs();  // 解析图的输入
  L.expect(':');  // 解析冒号表示后续是操作列表

  // 解析操作列表
  parseOperatorsList(g->block());

  // 最后一个语句应该是 return，指定图的输出
  parseReturnOperator();

  // 处理延迟的张量值初始化
  for (Node* n : deferred_tensor_value_initializations_) {
    auto type = n->output()->type()->expect<TensorType>();
    auto tt = n->output()->type()->cast<TensorType>();
    TORCH_INTERNAL_ASSERT(tt, "expected tensor output ", *n);
    auto sizes = tt->sizes().concrete_sizes();
    TORCH_INTERNAL_ASSERT(sizes);
    auto strides = tt->strides().concrete_sizes();
    TORCH_INTERNAL_ASSERT(strides);
    auto device = tt->device();
    TORCH_INTERNAL_ASSERT(device);
    auto dtype = tt->scalarType();
    TORCH_INTERNAL_ASSERT(dtype);
    auto options = at::TensorOptions(*device).dtype(*dtype);
    auto t = n->t_(attr::value, at::empty_strided(*sizes, *strides, options));
    (void)t;
  }

  // 处理延迟的空容器初始化
  for (Node* n : deferred_empty_container_initializations_) {
    auto type = n->output()->type();
    IValue val;
    if (type->kind() == TypeKind::ListType) {
      val = c10::impl::GenericList(type->containedType(0));
    } else if (type->kind() == TypeKind::DictType) {
      val = c10::impl::GenericDict(
          type->containedType(0), type->containedType(1));
    }
    n->ival_(attr::value, val);
  }
}

void IRParser::parseList(
    int begin,
    int sep,
    int end,
    const std::function<void()>& callback) {
  if (begin != TK_NOTHING) {
    L.expect(begin);  // 解析起始符号
  }
  if (L.cur().kind != end) {
    do {
      callback();  // 调用回调函数处理列表项
    } while (L.nextIf(sep));  // 如果遇到分隔符，继续解析下一个列表项
  }
  if (end != TK_NOTHING) {
    L.expect(end);  // 解析结束符号
  }
}

Value* IRParser::findValueInVMap(const std::string& name) {
  // 如果 vmap 中不存在变量名，则抛出异常
  if (!vmap.count(name)) {
    // 抛出错误报告，报告当前词法单元的范围，指示找不到名称为 'name' 的变量
    throw ErrorReport(L.cur().range)
        << "Cannot find a variable with name '" << name << "'";
    // 返回变量映射（vmap）中名称为 'name' 的变量的值
    }
    return vmap.at(name);
}

} // namespace torch::jit
```