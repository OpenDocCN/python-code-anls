# `.\pytorch\aten\src\ATen\code_template.h`

```py
#pragma once

#include <c10/util/irange.h>  // 引入 c10 库中的 irange.h 头文件

#include <sstream>  // 引入标准库中的 stringstream 头文件
#include <string>   // 引入标准库中的 string 头文件
#include <unordered_map>  // 引入标准库中的 unordered_map 头文件
#include <vector>   // 引入标准库中的 vector 头文件

namespace at::jit {

// A template environment is a mapping from template variable names, e.g.,
// identifier (corresponding to $identifier) to their expansions.
//
// This template environment supports storing strings, numbers and lists
// of strings, and can be chained together (so that lookup proceeds in
// in the top level environment, and then recurses into a parent
// environment if the key is not found.)
struct TemplateEnv {
  TemplateEnv() = default;  // 默认构造函数

  // Constructor that initializes with a parent environment.
  TemplateEnv(TemplateEnv& parent) : parent(&parent) {}

  using string_list = std::vector<std::string>;  // 定义 string_list 类型为 vector<string>

  // Add a string 'v' to the map at key 'k'.
  void s(const std::string& k, const std::string& v) {
    strings_[k] = v;    // 将字符串 v 添加到键为 k 的 strings_ 映射中
    lists_.erase(k);    // 如果存在同名键的列表，将其移除
  }

  // Add a number 'v' to the map at key 'k'
  template <typename T>
  void d(const std::string& k, const T& v) {
    strings_[k] = std::to_string(v);  // 将数值 v 转换为字符串并存储在键为 k 的 strings_ 映射中
    lists_.erase(k);    // 如果存在同名键的列表，将其移除
  }

  // Retrieve the string representation of the value stored at 'k' from the map.
  // Raises an exception if the key is not found.
  const std::string& s(const std::string& k) const {
    if (strings_.count(k) == 0) {   // 如果键 k 不存在于 strings_ 映射中
      if (parent) {   // 如果存在父环境
        return parent->s(k);    // 递归地从父环境中获取键 k 的值
      }
      notFound(k);    // 抛出逻辑错误异常，表示未找到键 k
    }
    return strings_.at(k);    // 返回键 k 对应的字符串值
  }

  // Store a list of strings 'v' in the map at 'k'.
  void v(const std::string& k, const string_list& v) {
    lists_[k] = v;    // 将字符串列表 v 存储在键为 k 的 lists_ 映射中
    strings_.erase(k);    // 如果存在同名键的字符串，将其移除
  }

  // Retrieve a list of strings stored at 'k' from the map.
  // Raises an exception if the key is not found.
  const string_list& v(const std::string& k) const {
    if (lists_.count(k) == 0) {   // 如果键 k 不存在于 lists_ 映射中
      if (parent) {   // 如果存在父环境
        return parent->v(k);    // 递归地从父环境中获取键 k 的列表值
      }
      notFound(k);    // 抛出逻辑错误异常，表示未找到键 k
    }
    return lists_.at(k);    // 返回键 k 对应的字符串列表值
  }

  // Test if a string 'k' is a string (as opposed to a list.)
  bool keyIsString(const std::string& k) const {
    if (strings_.count(k) > 0)    // 如果键 k 存在于 strings_ 映射中
      return true;    // 返回 true，表示键 k 对应的值为字符串
    if (lists_.count(k) > 0)    // 如果键 k 存在于 lists_ 映射中
      return false;   // 返回 false，表示键 k 对应的值为列表
    if (parent)    // 如果存在父环境
      return parent->keyIsString(k);    // 递归地判断父环境中键 k 的类型
    notFound(k);    // 抛出逻辑错误异常，表示未找到键 k
  }

 private:
  [[noreturn]] void notFound(const std::string& k) const {
    std::stringstream ss;
    ss << "key not found: " << k;   // 构造异常信息，指示未找到键 k
    throw std::logic_error(ss.str());   // 抛出逻辑错误异常，包含异常信息
  }

  std::unordered_map<std::string, std::string> strings_;   // 存储键为字符串，值为字符串的映射
  std::unordered_map<std::string, string_list> lists_;   // 存储键为字符串，值为字符串列表的映射
  TemplateEnv* parent{nullptr};   // 指向父模板环境的指针，默认为 nullptr
};

/*
# Match $identifier or ${identifier} and replace with the value in env.
# If this identifier is at the beginning of whitespace on a line
# and its value is a list then it is treated as
# block substitution by indenting all lines of all elements.
# If the identifier is on a line starting with non-whitespace and a list
# then it is comma separated. ${,foo} will insert a comma before the list
# if this list is not empty and ${foo,} will insert one after.
*/
struct CodeTemplate {
  /* implicit */ CodeTemplate(std::string t) : template_text(std::move(t)) {}

  // 格式化模板文本并替换其中的变量
  std::string format(const TemplateEnv& env) const {
    std::stringstream out;  // 输出流
    size_t pos = 0;  // 当前处理的位置
    size_t indent = 0;  // 当前行的缩进级别
    bool all_whitespace = true;  // 标记当前行是否全为空白

    // 遍历模板文本
    while (pos < template_text.size()) {
      char c = template_text[pos];  // 当前字符
      if (c == '$') {  // 如果是变量标记
        std::stringstream kss;  // 用于存储变量名
        bool comma_before;  // 是否有逗号在变量名之前
        bool comma_after;   // 是否有逗号在变量名之后
        size_t new_pos = parseKey(pos, kss, comma_before, comma_after);  // 解析变量名位置
        std::string k = kss.str();  // 变量名

        bool is_string = env.keyIsString(k);  // 判断变量是否为字符串类型

        // 根据上下文输出变量内容
        if (all_whitespace) {
          if (is_string)
            emitStringWithIndents(out, indent, env.s(k));  // 输出带有缩进的字符串
          else
            emitLinesIndented(out, indent, env.v(k));      // 输出带有缩进的行列表
        } else {
          if (is_string)
            out << env.s(k);  // 直接输出字符串内容
          else
            emitCommaSeparatedList(out, env.v(k), comma_before, comma_after);  // 输出逗号分隔的列表
        }

        all_whitespace = false;  // 更新行的空白状态
        pos = new_pos;  // 更新位置指针
      } else {
        out << c;  // 将非变量字符直接输出到流中
        if (!isspace(c))
          all_whitespace = false;  // 如果非空白字符，更新空白状态
        indent++;  // 更新缩进级别

        if (c == '\n') {  // 如果是换行符
          indent = 0;  // 重置缩进级别
          all_whitespace = true;  // 标记为全空白行
        }

        pos++;  // 移动位置指针
      }
    }

    return out.str();  // 返回格式化后的字符串
  }

 private:
  using string_list = std::vector<std::string>;
  
  // 获取字符串中指定位置的字符
  char charAt(size_t p) const {
    if (p >= template_text.size())
      throw std::logic_error("EOS found in key");
    return template_text[p];
  }

  // 解析变量名，并返回变量名结束后的位置
  size_t parseKey(
      size_t pos,
      std::ostream& k,
      bool& comma_before,
      bool& comma_after) const {
    comma_before = false;  // 初始化逗号标记为false
    comma_after = false;   // 初始化逗号标记为false
    pos++;  // 跳过'$'，指向变量名开头

    if (charAt(pos) == '{') {  // 如果变量名是大括号包裹的形式
      pos++;  // 跳过'{'

      if (charAt(pos) == ',') {  // 如果变量名之前有逗号
        comma_before = true;  // 设置逗号标记为true
        pos++;  // 跳过逗号
      }

      pos = parseIdent(pos, k);  // 解析变量名
      if (charAt(pos) == ',') {  // 如果变量名之后有逗号
        comma_after = true;  // 设置逗号标记为true
        pos++;  // 跳过逗号
      }

      if (charAt(pos) != '}')  // 如果没有找到闭合的'}'
        throw std::logic_error("missing terminating '}'");  // 抛出异常

      pos++;  // 跳过'}'
      return pos;  // 返回变量名解析结束后的位置
    } else {
      return parseIdent(pos, k);  // 直接解析变量名
    }
  }

  // 解析变量名，将其写入流中，并返回变量名结束后的位置
  size_t parseIdent(size_t pos, std::ostream& k) const {
    while (pos < template_text.size() &&
           (isalnum(template_text[pos]) || template_text[pos] == '_')) {
      k << template_text[pos];  // 将变量名字符写入流中
      pos++;  // 移动位置指针
    }
    return pos;  // 返回变量名结束后的位置
  }

  // 输出逗号分隔的列表
  void emitCommaSeparatedList(
      std::ostream& out,
      const string_list& strings,
      bool comma_before,
      bool comma_after) const {
    if (comma_before && !strings.empty())  // 如果列表非空且在变量名之前有逗号
      out << ", ";  // 输出逗号和空格

    for (const auto i : c10::irange(strings.size())) {  // 遍历字符串列表
      if (i > 0)
        out << ", ";  // 输出逗号和空格
      out << strings[i];  // 输出列表中的字符串元素
    }
}
    // 如果 comma_after 为真且 strings 不为空，则在输出流 out 中添加逗号和空格
    if (comma_after && !strings.empty())
      out << ", ";
  }
  // 这些缩进函数遵循的约定是，当输入字符串没有前导或尾随换行符时，它们不会产生前导或尾随换行符。
  // 在调用函数的上下文中，正确缩进是调用者的责任。
  void emitIndent(std::ostream& out, size_t indent) const {
    // 根据缩进值 indent，在输出流 out 中插入相应数量的空格
    for (C10_UNUSED const auto i : c10::irange(indent)) {
      out << " ";
    }
  }
  void emitStringWithIndents(
      std::ostream& out,
      size_t indent,
      const std::string& str) const {
    // 遍历输入字符串 str 中的每个字符
    for (auto c : str) {
      // 将字符 c 输出到输出流 out
      out << c;
      // 如果字符 c 是换行符 '\n'，则调用 emitIndent 函数以确保正确的缩进
      if (c == '\n') {
        emitIndent(out, indent);
      }
    }
  }
  void emitLinesIndented(
      std::stringstream& out,
      size_t indent,
      const string_list& strings) const {
    // 遍历字符串列表 strings 中的每个字符串
    for (const auto i : c10::irange(strings.size())) {
      // 如果不是第一行字符串，则在输出流 out 中插入正确的缩进
      if (i > 0)
        emitIndent(out, indent);
      // 将当前字符串 strings[i] 以正确的缩进输出到输出流 out
      emitStringWithIndents(out, indent, strings[i]);
      // 如果当前字符串不是最后一个字符串，则在输出流 out 中添加换行符
      if (i + 1 != strings.size())
        out << "\n";
    }
  }
  // 存储模板文本的字符串
  std::string template_text;
};

// 关闭匿名命名空间

static inline std::string format(const std::string& fmt, TemplateEnv& env) {
  // 使用给定的模板字符串 fmt 和环境变量 env 创建 CodeTemplate 对象，并格式化输出字符串
  return CodeTemplate(fmt).format(env);
}

// 结束命名空间 at::jit
} // namespace at::jit
```