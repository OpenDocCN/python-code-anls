# `.\pytorch\torch\csrc\jit\mobile\type_parser.h`

```py
#pragma once
// 预处理指令，确保头文件只被包含一次

#include <ATen/core/dynamic_type.h>
#include <ATen/core/jit_type.h>
#include <unordered_set>
// 包含必要的头文件

namespace c10 {

class TORCH_API TypeParser {
 public:
  explicit TypeParser(std::string pythonStr);
  // 构造函数，接受一个 Python 字符串作为参数

  explicit TypeParser(std::vector<std::string>& pythonStrs);
  // 构造函数，接受一个 Python 字符串向量作为参数

  TypePtr parse();
  // 解析单个类型字符串并返回其类型指针

  std::vector<TypePtr> parseList();
  // 解析多个类型字符串并返回类型指针向量

  static const std::unordered_set<std::string>& getNonSimpleType();
  // 静态方法，返回非简单类型的集合

  static const std::unordered_set<std::string>& getCustomType();
  // 静态方法，返回自定义类型的集合

  std::unordered_set<std::string> getContainedTypes();
  // 返回解析过程中包含的类型集合

 private:
  TypePtr parseNamedTuple(const std::string& qualified_name);
  // 解析命名元组类型字符串并返回类型指针

  TypePtr parseCustomType();
  // 解析自定义类型字符串并返回类型指针

  TypePtr parseTorchbindClassType();
  // 解析 Torchbind 类型字符串并返回类型指针

  TypePtr parseNonSimple(const std::string& token);
  // 解析非简单类型字符串并返回类型指针

  void expect(const char* s);
  // 检查当前位置的字符串是否与参数相匹配

  void expectChar(char c);
  // 检查当前位置的字符是否与参数相匹配

  template <typename T>
  TypePtr parseSingleElementType();
  // 解析单个元素类型并返回类型指针

  void lex();
  // 对输入的字符串进行词法分析

  std::string next();
  // 返回下一个词法单元作为字符串

  c10::string_view nextView();
  // 返回下一个词法单元作为字符串视图

  void advance();
  // 推进到下一个词法单元

  C10_NODISCARD c10::string_view cur() const;
  // 返回当前词法单元的字符串视图

  std::string pythonStr_;
  // 存储传入的 Python 字符串

  size_t start_;
  // 记录当前解析的起始位置

  c10::string_view next_token_;
  // 存储下一个词法单元的字符串视图

  // 用于解析字符串列表
  std::vector<std::string> pythonStrs_;
  // 存储传入的 Python 字符串向量

  std::unordered_map<std::string, c10::TypePtr> str_type_ptr_map_;
  // 存储字符串到类型指针的映射

  std::unordered_set<std::string> contained_types_;
  // 存储解析过程中包含的类型集合
};

TORCH_API TypePtr parseType(const std::string& pythonStr);
// 解析单个类型字符串并返回其类型指针

TORCH_API std::vector<TypePtr> parseType(std::vector<std::string>& pythonStrs);
// 解析多个类型字符串并返回类型指针向量

} // namespace c10
// 命名空间 c10 的结束
```