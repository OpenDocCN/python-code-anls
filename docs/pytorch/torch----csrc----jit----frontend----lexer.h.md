# `.\pytorch\torch\csrc\jit\frontend\lexer.h`

```py
#pragma once
// 预处理指令，确保头文件只被包含一次

#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <torch/csrc/Export.h>
#include <torch/csrc/jit/frontend/parser_constants.h>
#include <torch/csrc/jit/frontend/source_range.h>
#include <torch/csrc/jit/frontend/strtod.h>
#include <algorithm>
#include <clocale>
#include <cstdlib>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

C10_CLANG_DIAGNOSTIC_PUSH()
#if C10_CLANG_HAS_WARNING("-Wshorten-64-to-32")
C10_CLANG_DIAGNOSTIC_IGNORE("-Wshorten-64-to-32")
#endif

namespace torch {
namespace jit {

// 单个字符的标记只是字符本身，例如 '+'
// 多字符标记需要在此处添加条目
// 如果第三个条目不是空字符串，则在词法分析器中用于匹配此标记

// 这些标记也用作 Tree.h 中 AST 节点的类型。
// 一些标记如 TK_APPLY、TK_LIST 仅用于 AST 中，不在词法分析器中看到

enum TokenKind {
  // 使用字符本身来表示标记，因此在为多字符标记分配枚举值之前，跳过所有有效字符。
  TK_DUMMY_START = 256,
#define DEFINE_TOKEN(tok, _, _2) tok,
  TC_FORALL_TOKEN_KINDS(DEFINE_TOKEN)
#undef DEFINE_TOKEN
};

// 将标记类型转换为字符串
TORCH_API std::string kindToString(int kind);
// 将字符串转换为标记类型
TORCH_API int stringToKind(const std::string& str);

// 表示每个字符的嵌套哈希表，指示什么是有效标记
struct TokenTrie;
using TokenTrieRef = std::unique_ptr<TokenTrie>;
struct TokenTrie {
  TokenTrie() : kind(0) {}
  // 插入标记到字符哈希表中
  void insert(const char* str, int tok) {
    if (*str == '\0') {
      AT_ASSERT(kind == 0); // 断言保证 kind 为 0
      kind = tok;
      return;
    }

    for (size_t i = 0, e = child_chars.size(); i < e; ++i) {
      if (child_chars[i] == *str) {
        child_tries[i]->insert(str + 1, tok);
        return;
      }
    }

    child_chars.emplace_back(*str);
    child_tries.emplace_back(std::make_unique<TokenTrie>());
    child_tries.back()->insert(str + 1, tok);
  }
  int kind; // 0 == 无效标记

  std::vector<char> child_chars; // 子字符列表
  std::vector<TokenTrieRef> child_tries; // 子字符哈希表
};

// 所有 TC 词法分析器/解析器共享的数据，仅初始化一次
struct TORCH_API SharedParserData {
  SharedParserData() : head(new TokenTrie()) {
    std::stringstream ss;
    // 遍历有效单字符标记，插入到字符哈希表中
    for (const char* c = valid_single_char_tokens; *c; c++) {
      std::string str(1, *c);
      head->insert(str.c_str(), *c);
    }

#define ADD_CASE(tok, _, tokstring)   \
  if (*(tokstring) != '\0') {         \
    head->insert((tokstring), (tok)); \
  }
    TC_FORALL_TOKEN_KINDS(ADD_CASE) // 插入所有标记种类到字符哈希表中
#undef ADD_CASE
  }

  // 匹配标记，返回标记类型和标记位置
  bool match(
      StringCordView::Iterator pos,
      bool continuation, // 是否在不计入换行符的范围内的作用域中（例如括号内）
      bool whitespace_token, // 是否将空白字符作为标记
      int* kind,
      StringCordView::Iterator* start,
      StringCordView::Iterator* end) {
    *start = pos; // 记录匹配起始位置
    // 跳过空白字符
    // 当位置对象有下一个字符并且当前字符是空白时，进行循环
    while (pos.has_next() && isblank(*pos)) {
      ++pos;
    }

    // 特殊处理
    if (pos.has_next()) {
      // 如果当前字符是 '#' 并且不是类型注释，则跳过注释内容
      if (*pos == '#' && !isTypeComment(pos)) {
        // 跳过注释内容直到遇到换行符
        while (pos.has_next() && *pos != '\n')
          ++pos;
        // 尾递归，处理空白和更多的注释
        return match(pos, continuation, whitespace_token, kind, start, end);
      }
      // 如果当前字符是 '\'，处理转义字符
      if (*pos == '\\') {
        auto newiter = pos;
        ++newiter;
        // 如果下一个字符是换行符并且不是空白令牌，则跳过换行符
        if (newiter.has_next() && *newiter == '\n' && !whitespace_token) {
          ++newiter;
          return match(newiter, continuation, false, kind, start, end);
        }
      }
      // 如果当前字符是换行符，处理换行符情况
      if (*pos == '\n') {
        return match(++pos, continuation, !continuation, kind, start, end);
      }
    }
    // 在处理文件末尾之前处理空白字符，因为在某些情况下需要生成去除缩进的令牌
    if (whitespace_token) {
      // 如果没有下一个字符，则表示到达了文件末尾的空白
      *kind = !pos.has_next() ? TK_WHITESPACE_EOF : TK_WHITESPACE;
      *end = pos;
      return true;
    }
    // 如果没有下一个字符，则表示到达了文件末尾
    if (!pos.has_next()) {
      *kind = TK_EOF;
      *start = pos;
      *end = *start;
      return true;
    }
    // 不变式：下一个令牌不是空白或换行符
    *start = pos;
    // 检查是否是有效的数字
    size_t len;
    if (isNumber(pos.rest_line(), 0, &len)) {
      *end = *start;
      *end += len;
      *kind = TK_NUMBER;
      return true;
    }
    // 检查是否是字符串
    if (isString(pos.rest_line(), 0, &len)) {
      *kind = TK_STRINGLITERAL;
      *end = *start;
      *end += len;
      return true;
    }

    // 检查是否是标识符或者令牌
    // ident 表示到目前为止扫描的内容可能是一个标识符
    // matched 表示是否已经找到匹配的令牌或标识符
    bool matched = false;
    bool ident = true;
    TokenTrie* cur = head.get();
    // 对于每个字符直到位置对象没有下一个字符或者仍然可能是标识符或者仍然有未匹配的 Trie 节点
    for (size_t i = 0; pos.has_next() && (ident || cur != nullptr);
         ++pos, ++i) {
      ident = ident && validIdent(i, *pos);
      if (ident) {
        matched = true;
        *end = pos.next_iter();
        *kind = TK_IDENT;
      }
      // 先检查令牌，以便例如 'max' 匹配 TK_MAX 而不是标识符 'max'
      if (cur) {
        const auto begin_it = cur->child_chars.begin();
        const auto end_it = cur->child_chars.end();
        const auto ch_it = std::find(begin_it, end_it, *pos);

        cur = (ch_it == end_it) ? nullptr
                                : cur->child_tries[ch_it - begin_it].get();

        if (cur && cur->kind != 0) {
          matched = true;
          *end = pos.next_iter();
          *kind = cur->kind;
        }
      }
    }
    return matched;
  }

  // 检查是否是一元操作符
  bool isUnary(int kind, int* prec);
  // 检查是否是二元操作符
  bool isBinary(int kind, int* prec);
  // 检查是否是右结合操作符
  bool isRightAssociative(int kind) {
    switch (kind) {
      // 如果 kind 是 '?'、TK_POW、TK_IF 中的任何一个，返回 true
      case '?':
      case TK_POW:
      case TK_IF:
        return true;
      // 对于其他情况，返回 false
      default:
        return false;
    }
  }

 private:
  // 检查字符是否可以用作有效的标识符的一部分
  bool validIdent(size_t i, char n) {
    return isalpha(n) || n == '_' || (i > 0 && isdigit(n));
  }

  // 1. 跳过空白字符
  // 2. 处理注释或换行符
  //
  bool isNumber(c10::string_view str, size_t start, size_t* len) {
    char first = str[start];
    // 如果第一个字符是 '-'、'+' 或字母，则不是一个数字
    if (first == '-' || first == '+' || isalpha(first))
      return false;
    const char* startptr = str.data() + start;
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    char* endptr;
    // 使用 strtod_c 函数将字符串转换为浮点数，获取转换后的长度
    torch::jit::strtod_c(startptr, &endptr);
    *len = endptr - startptr;
    // 如果最后一个字符是 'j'，表示这是一个复数值
    if (endptr != nullptr && *endptr == 'j') {
      *len += 1;
    }
    return *len > 0;
  }

  // 检查字符串是否是有效的字符串字面值
  bool isString(c10::string_view str, size_t start, size_t* len) {
    char quote = str[start];
    if (quote != '\"' && quote != '\'')
      return false;
    int quote_len = isCharCount(quote, str, start, 3) ? 3 : 1;

    // end 现在设置为超过开头的引号
    size_t end = start + quote_len;
    while (end < str.size() && !isCharCount(quote, str, end, quote_len)) {
      if (str[end] == '\n' && quote_len != 3) {
        return false;
      }
      // 处理转义字符，跳过转义的引号、换行符和反斜杠
      if (str[end] == '\\') {
        end++;
      }
      end++;
    }
    // 设置长度为包括引号的完整字符串长度
    *len = end - start + quote_len;
    // 如果 end 没有超过字符串的最后一个字符，则匹配成功
    return end < str.size();
  }

  // 检查字符是否为空白字符，但不包括换行符 '\n'
  bool isblank(int n) {
    return isspace(n) && n != '\n';
  }

  // 检查是否是类型注释，即是否以 "# type:" 开头
  bool isTypeComment(StringCordView::Iterator str_iter) {
    c10::string_view rest_line = str_iter.rest_line();
    const std::string type_string = "# type:";
    if (rest_line.size() < type_string.length()) {
      return false;
    }
    // 检查字符串是否以 "# type:" 开头
    return rest_line.substr(0, type_string.length()) == type_string;
  }

  // 检查字符在指定的范围内出现的次数是否符合预期
  bool isCharCount(char c, c10::string_view str, size_t start, int len) {
    // 检查从 [start, start + len) 范围内字符 c 的数量
    return start + len <= str.size() &&
        std::count(str.begin() + start, str.begin() + start + len, c) == len;
  }
    // 从字符串的开头提取与给定类型字符串相同长度的子字符串
    auto match_string = rest_line.substr(0, type_string.size());
    // 返回是否提取的子字符串与给定类型字符串相同
    return match_string == type_string;
  }

  // 检查是否是类型注解的注释，忽略注释符号
  bool isTypeComment(StringCordView str, size_t pos) {
    // 定义类型注解的标识符字符串
    const std::string type_string = "# type:";
    // 如果字符串长度不足以包含标识符，返回 false
    if (str.size() < pos + type_string.length()) {
      return false;
    }
    // 从指定位置开始提取与类型注解标识符相同长度的子字符串
    auto match_string = str.substr(pos, type_string.size());
    // 返回是否提取的子字符串与类型注解标识符相同
    return match_string == type_string;
  }

  // TokenTrieRef 类型的变量 head
  TokenTrieRef head;
};

// 返回共享的解析器数据的引用
TORCH_API SharedParserData& sharedParserData();

// 表示一个词法分析器的 Token 结构
struct Token {
  int kind;                 // 表示 Token 的类型
  SourceRange range;        // 表示 Token 的源代码范围

  // 构造函数，初始化 Token 类型和源代码范围
  Token(int kind, SourceRange range) : kind(kind), range(std::move(range)) {}

  // 返回 Token 的文本表示
  std::string text() {
    return std::string(range.token_text());
  }

  // 返回 Token 类型的字符串表示
  std::string kindString() const {
    return kindToString(kind);
  }
};

// 表示词法分析器 Lexer
struct Lexer {
  explicit Lexer(std::shared_ptr<Source> source)
      : source(std::move(source)),  // 使用给定的源码初始化 Lexer 的源码对象
        pos(0),                    // 初始化位置为 0
        nesting(0),                // 初始化嵌套层级为 0
        indent_stack(),            // 初始化缩进栈
        next_tokens(),             // 初始化下一个 Token 的队列为空
        shared(sharedParserData()) {  // 初始化共享的解析器数据
    auto first_indent = lexRaw(true);  // 调用词法分析函数，获取第一个缩进
    indent_stack.push_back(first_indent.range.size());  // 将第一个缩进大小压入栈中
    lex();  // 执行词法分析
  }

  // 返回当前 Token，并移动到下一个 Token
  Token next() {
    if (next_tokens.empty())
      reportError("Lexer invariant violated: empty token queue");  // 报告错误，如果 Token 队列为空
    Token r = std::move(next_tokens.front());  // 获取队列中的第一个 Token
    next_tokens.erase(next_tokens.begin());    // 从队列中移除第一个 Token
    if (next_tokens.empty()) {
      lex();  // 如果队列为空，则重新进行词法分析
    }
    return r;
  }

  // 如果当前 Token 的类型与给定的类型匹配，则跳过当前 Token
  bool nextIf(int kind) {
    if (cur().kind != kind)
      return false;
    next();  // 移动到下一个 Token
    return true;
  }

  // 报告错误并终止程序，给定错误信息
  [[noreturn]] void reportError(const std::string& what) {
    reportError(what, cur());
  }

  // 报告错误并终止程序，给定错误信息和相关 Token
  [[noreturn]] void reportError(const std::string& what, const Token& t) {
    std::stringstream ss;
    ss << what << ":\n";
    t.range.highlight(ss);  // 在错误信息中突出显示相关 Token 的源代码范围
    throw std::runtime_error(ss.str());  // 抛出运行时异常
  }

  // 报告预期的错误信息，并终止程序
  [[noreturn]] void expected(const std::string& what, const Token& t) {
    std::stringstream ss;
    ss << "expected " << what << " but found '" << t.kindString()
       << "' here:\n";
    t.range.highlight(ss);  // 在错误信息中突出显示相关 Token 的源代码范围
    throw std::runtime_error(ss.str());  // 抛出运行时异常
  }

  // 报告预期的错误信息，并终止程序
  [[noreturn]] void expected(const std::string& what) {
    expected(what, cur());
  }

  // 检查当前 Token 的类型是否与给定类型匹配，返回当前 Token，并移动到下一个 Token
  Token expect(int kind) {
    if (cur().kind != kind) {
      expected(kindToString(kind));  // 报告预期的错误信息
    }
    return next();  // 返回当前 Token，并移动到下一个 Token
  }

  // 返回下一个 Token（向前看一个 Token，不移动当前位置）
  Token& lookahead() {
    if (next_tokens.size() < 2) {
      lex();  // 执行词法分析
    }
    return next_tokens[1];  // 返回向前看的第二个 Token
  }

  // 返回当前 Token
  Token& cur() {
    return next_tokens.front();  // 返回队列中的第一个 Token
  }

 private:
  // 执行词法分析
  void lex() {
    auto r = lexRaw();  // 调用底层的词法分析函数
    // 实现细节未提供
  }

  // 成员变量
  std::shared_ptr<Source> source;  // 源码对象的共享指针
  int pos;                         // 当前位置
  int nesting;                     // 嵌套层级
  std::vector<int> indent_stack;   // 缩进栈
  std::vector<Token> next_tokens;  // 下一个 Token 的队列
  SharedParserData& shared;        // 共享的解析器数据
};
    // 根据语法分析器中的 token 类型进行不同的处理
    switch (r.kind) {
      case '(':  // 处理左括号，增加嵌套深度
      case '[':  // 处理左方括号，增加嵌套深度
      case '{':  // 处理左大括号，增加嵌套深度
        nesting++;
        break;
      case ')':  // 处理右括号，减少嵌套深度
      case ']':  // 处理右方括号，减少嵌套深度
      case '}':  // 处理右大括号，减少嵌套深度
        nesting--;
        break;
      case TK_WHITESPACE:
      case TK_WHITESPACE_EOF: {
        const auto depth = static_cast<int64_t>(
            r.kind == TK_WHITESPACE_EOF ? indent_stack.front()
                                        : r.range.size());
        // 注释: TK_WHITESPACE_EOF 是 EOF 之前的空白，用于处理最终空白
        // 我们允许代码具有特定的初始缩进级别，也允许最终缩进为任意值，并将其设置回初始缩进级别。
        // 这使得代码可以放入代码内的字符串字面值中，而无需担心最终的空白字符。
        if (depth > indent_stack.back()) {
          indent_stack.push_back(depth);
          r.kind = TK_INDENT;
        } else if (depth == indent_stack.back()) {
          r.kind = TK_NEWLINE;
        } else {
          next_tokens.emplace_back(TK_NEWLINE, r.range);
          while (indent_stack.back() != depth) {
            indent_stack.pop_back();
            next_tokens.emplace_back(TK_DEDENT, r.range);
            if (indent_stack.empty()) {
              reportError("invalid indent level " + std::to_string(depth), r);
            }
          }
          return; // 已经将 token 排队，直接返回
        }
      } break;
      default:
        break;
    }
    // 将处理完的 token 加入到下一个 token 集合中
    next_tokens.push_back(std::move(r));
  }

  Token lexRaw(bool whitespace_token = false) {
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    int kind;
    AT_ASSERT(source);  // 确保源代码不为空
    if (current == nullptr) {
      AT_ASSERT(pos == 0);
      // 初始化当前迭代器为源代码文本的开头
      current = std::make_unique<StringCordView::Iterator>(
          source->text_str().begin());
    }

    StringCordView::Iterator start_iter = *current;
    StringCordView::Iterator end_iter = *current;
    // 如果当前迭代器无法匹配有效 token，则报告预期的有效 token
    if (!shared.match(
            *current,
            nesting > 0,
            whitespace_token,
            &kind,
            &start_iter,
            &end_iter)) {
      expected(
          "a valid token",
          Token(
              **current,
              SourceRange(source, start_iter, start_iter.pos() + 1)));
    }

    // 创建 token 对象并更新迭代器位置
    auto t = Token(kind, SourceRange(source, start_iter, end_iter.pos()));
    pos = end_iter.pos();
    *current = end_iter;
    return t;
  }

  std::shared_ptr<Source> source;  // 源代码的共享指针
  std::unique_ptr<StringCordView::Iterator> current;  // 当前字符迭代器的唯一指针
  size_t pos;  // 当前位置在源代码中的索引
  size_t nesting;  // 嵌套深度，用于跟踪 ( [ { 的嵌套级别
  std::vector<int> indent_stack;  // 代码块的缩进级别堆栈
  // 不变量: 这个堆栈应该始终至少包含一个元素
  std::vector<Token> next_tokens;  // 下一个 token 集合
  SharedParserData& shared;  // 共享的解析器数据
};
} // namespace jit
} // namespace torch
```