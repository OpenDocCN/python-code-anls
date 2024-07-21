# `.\pytorch\torch\csrc\jit\frontend\parse_string_literal.h`

```py
#pragma once
// 包含C++标准库中的可选类型
#include <c10/util/Optional.h>
// 包含Torch的错误报告相关头文件
#include <torch/csrc/jit/frontend/error_report.h>
// 包含Torch的词法分析器相关头文件
#include <torch/csrc/jit/frontend/lexer.h>

// Torch命名空间开始
namespace torch {
// JIT命名空间开始
namespace jit {

// 定义一个内联函数，用于检查字符串中指定范围内字符的数量是否符合要求
inline bool isCharCount(char c, const std::string& str, size_t start, int len) {
  // 检查从[start, start + len)范围内是否有指定字符c出现len次
  return start + len <= str.size() &&
      std::count(str.begin() + start, str.begin() + start + len, c) == len;
}

// 定义一个内联函数，用于解析八进制字符串字面量中的字符
inline std::optional<char> parseOctal(const std::string& str, size_t pos) {
  // 如果位置pos后面不足三个字符，则返回空
  if (pos + 3 >= str.size())
    return c10::nullopt;
  size_t c = 0;
  for (size_t i = 1, b = 64; i < 4; ++i, b /= 8) {
    // NOLINTNEXTLINE(bugprone-signed-char-misuse)
    int d = str[pos + i];
    // 如果字符不是八进制数字，则返回空
    if (d < '0' || d > '7')
      return c10::nullopt;
    c += b * (d - '0');
  }
  // 如果计算结果超过了256，则返回空
  if (c >= 256)
    return c10::nullopt;
  return c;
}

// 定义一个内联函数，用于解析字符串字面量
inline std::string parseStringLiteral(
    const SourceRange& range,
    const std::string& str) {
  // 确定引号的长度，可以是'''或者'"
  int quote_len = isCharCount(str[0], str, 0, 3) ? 3 : 1;
  // 获取去除引号后的字符串内容
  auto ret_str = str.substr(quote_len, str.size() - quote_len * 2);
  size_t pos = ret_str.find('\\');
  // 当找到转义字符时进行处理
  while (pos != std::string::npos) {
    // 获取转义字符的实际值
    char c = ret_str[pos + 1];
    size_t to_erase = 2;
    switch (ret_str[pos + 1]) {
      // 对不同转义字符进行处理
      case '\\':
      case '\'':
      case '\"':
      case '\n':
        break;
      case 'a':
        c = '\a';
        break;
      case 'b':
        c = '\b';
        break;
      case 'f':
        c = '\f';
        break;
      case 'n':
        c = '\n';
        break;
      case 'v':
        c = '\v';
        break;
      case 't':
        c = '\t';
        break;
      case 'x':
        throw ErrorReport(range) << "unsupported hex specifier";
      case 'u':
      case 'U':
        throw ErrorReport(range) << "unsupported unicode specifier";
      default:
        // 如果是八进制值，替换为对应字符；否则报错
        if (auto v = parseOctal(ret_str, pos)) {
          to_erase = 4;
          c = *v;
        } else {
          throw ErrorReport(range) << " ill formed octal specifier";
        }
    }
    // 替换转义字符为实际字符
    ret_str.replace(pos, to_erase, /* num copies */ 1, c);
    pos = ret_str.find('\\', pos + 1);
  }
  // 返回处理后的字符串
  return ret_str;
}

} // namespace jit
} // namespace torch
```