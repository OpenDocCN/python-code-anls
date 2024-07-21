# `.\pytorch\c10\util\StringUtil.h`

```
#ifndef C10_UTIL_STRINGUTIL_H_
#define C10_UTIL_STRINGUTIL_H_

#include <c10/macros/Macros.h>
#include <c10/util/string_utils.h>
#include <c10/util/string_view.h>

#include <cstddef>
#include <ostream>
#include <sstream>
#include <string>

// 开始命名空间 c10
namespace c10 {

// detail 命名空间，用于实现细节
namespace detail {

// 从完整路径中获取基本文件名
C10_API std::string StripBasename(const std::string& full_path);

// 从完整路径中排除文件扩展名
C10_API std::string ExcludeFileExtension(const std::string& full_path);

// 用于编译时空字符串的结构体
struct CompileTimeEmptyString {
  // 转换操作符，返回一个静态的空字符串字面量引用
  operator const std::string&() const {
    static const std::string empty_string_literal;
    return empty_string_literal;
  }
  // 转换操作符，返回一个空字符指针
  operator const char*() const {
    return "";
  }
};

// 模板用于标准化字符串类型 T
template <typename T>
struct CanonicalizeStrTypes {
  using type = const T&;
};

// 对于字符数组类型 char[N] 的模板特化，标准化为 const char*
template <size_t N>
// NOLINTNEXTLINE(*c-arrays*)
struct CanonicalizeStrTypes<char[N]> {
  using type = const char*;
};

// 内联函数，用于向输出流写入内容
inline std::ostream& _str(std::ostream& ss) {
  return ss;
}

// 模板函数，将参数 t 写入输出流 ss
template <typename T>
inline std::ostream& _str(std::ostream& ss, const T& t) {
  // NOLINTNEXTLINE(clang-analyzer-core.CallAndMessage)
  ss << t;
  return ss;
}

// 重载 _str 函数，处理宽字符类型，强制窄化
C10_API std::ostream& _str(std::ostream& ss, const wchar_t* wCStr);
C10_API std::ostream& _str(std::ostream& ss, const wchar_t& wChar);
C10_API std::ostream& _str(std::ostream& ss, const std::wstring& wString);

// 模板特化，处理编译时空字符串
template <>
inline std::ostream& _str<CompileTimeEmptyString>(
    std::ostream& ss,
    const CompileTimeEmptyString&) {
  return ss;
}

// 变长模板函数，递归调用 _str 函数，处理多个参数
template <typename T, typename... Args>
inline std::ostream& _str(std::ostream& ss, const T& t, const Args&... args) {
  return _str(_str(ss, t), args...);
}

// _str_wrapper 结构体模板，将参数转换为单个字符串并返回
template <typename... Args>
struct _str_wrapper final {
  static std::string call(const Args&... args) {
    std::ostringstream ss;
    _str(ss, args...);
    return ss.str();
  }
};

// 对已经是字符串类型的特化处理
template <>
struct _str_wrapper<std::string> final {
  // 返回字符串的引用，避免复制字符串的二进制大小
  static const std::string& call(const std::string& str) {
    return str;
  }
};

// 对 const char* 类型的特化处理
template <>
struct _str_wrapper<const char*> final {
  static const char* call(const char* str) {
    return str;
  }
};

// 处理 c10::str() 参数列表为空的情况，返回编译时空字符串
template <>
struct _str_wrapper<> final {
  static CompileTimeEmptyString call() {
    return CompileTimeEmptyString();
  }
};

} // namespace detail

// 将一组类似字符串的参数转换为单个字符串
template <typename... Args>
inline decltype(auto) str(const Args&... args) {
  return detail::_str_wrapper<
      typename detail::CanonicalizeStrTypes<Args>::type...>::call(args...);
}

template <class Container>
/// 将容器中的元素用指定的分隔符连接成一个字符串并返回。
inline std::string Join(const std::string& delimiter, const Container& v) {
  // 创建一个字符串流
  std::stringstream s;
  // 计算容器中元素个数减一，作为循环结束条件的初始值
  int cnt = static_cast<int64_t>(v.size()) - 1;
  // 遍历容器中的元素
  for (auto i = v.begin(); i != v.end(); ++i, --cnt) {
    // 将当前元素写入字符串流
    s << (*i) << (cnt ? delimiter : "");
  }
  // 返回连接后的字符串
  return s.str();
}

// 替换字符串中所有的 "from" 子串为 "to" 字符串。
// 返回替换的次数
size_t C10_API
ReplaceAll(std::string& s, c10::string_view from, c10::string_view to);

/// 表示源代码中的位置（用于调试）。
struct C10_API SourceLocation {
  const char* function;  ///< 函数名
  const char* file;      ///< 文件名
  uint32_t line;         ///< 行号
};

/// 将源代码位置信息输出到流中。
std::ostream& operator<<(std::ostream& out, const SourceLocation& loc);

// unix isprint 的替代，但不受区域设置影响
inline bool isPrint(char s) {
  // 判断字符是否可打印（ASCII 可见字符）
  return s > 0x1f && s < 0x7f;
}

/// 将给定的字符串视图打印到输出流中，带引号，并转义特殊字符。
inline void printQuotedString(std::ostream& stmt, const string_view str) {
  stmt << "\"";
  // 遍历字符串中的每个字符
  for (auto s : str) {
    switch (s) {
      case '\\':
        stmt << "\\\\";  // 转义反斜杠
        break;
      case '\'':
        stmt << "\\'";   // 转义单引号
        break;
      case '\"':
        stmt << "\\\"";  // 转义双引号
        break;
      case '\a':
        stmt << "\\a";   // 转义响铃符
        break;
      case '\b':
        stmt << "\\b";   // 转义退格符
        break;
      case '\f':
        stmt << "\\f";   // 转义换页符
        break;
      case '\n':
        stmt << "\\n";   // 转义换行符
        break;
      case '\r':
        stmt << "\\r";   // 转义回车符
        break;
      case '\t':
        stmt << "\\t";   // 转义制表符
        break;
      case '\v':
        stmt << "\\v";   // 转义垂直制表符
        break;
      default:
        if (isPrint(s)) {
          stmt << s;    // 如果是可打印字符，则直接输出
        } else {
          // 对于不可打印字符，手动转义为八进制表示
          char buf[4] = "000";
          buf[2] += s % 8;
          s /= 8;
          buf[1] += s % 8;
          s /= 8;
          buf[0] += s;
          stmt << "\\" << buf;
        }
        break;
    }
  }
  stmt << "\"";
}

} // namespace c10

C10_CLANG_DIAGNOSTIC_POP()

#endif // C10_UTIL_STRINGUTIL_H_
```