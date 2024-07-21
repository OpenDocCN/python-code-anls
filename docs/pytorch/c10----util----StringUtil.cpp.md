# `.\pytorch\c10\util\StringUtil.cpp`

```py
// 包含头文件<c10/util/StringUtil.h>
#include <c10/util/StringUtil.h>

// 包含标准库中的<string>头文件
#include <string>

// 如果不是在Windows平台编译，包含以下头文件
#ifndef _WIN32
#include <codecvt>  // 用于字符编码转换
#include <locale>   // 用于本地化设置
#else
#include <c10/util/Unicode.h>  // Windows平台下的Unicode支持
#endif

// c10命名空间
namespace c10 {

// detail命名空间
namespace detail {

// 函数：从完整路径中提取文件名
std::string StripBasename(const std::string& full_path) {
  // 根据操作系统确定路径分隔符
#ifdef _WIN32
  const std::string separators("/\\");
#else
  const std::string separators("/");
#endif
  // 找到路径中最后一个分隔符的位置
  size_t pos = full_path.find_last_of(separators);
  // 如果找到分隔符，则返回分隔符后面的部分作为文件名，否则返回整个路径
  if (pos != std::string::npos) {
    return full_path.substr(pos + 1, std::string::npos);
  } else {
    return full_path;
  }
}

// 函数：去除文件名的扩展名
std::string ExcludeFileExtension(const std::string& file_name) {
  const char sep = '.';  // 扩展名分隔符为'.'
  // 找到最后一个'.'的位置作为扩展名的起始位置
  auto end_index = file_name.find_last_of(sep) == std::string::npos
      ? -1  // 如果找不到扩展名，则返回-1
      : file_name.find_last_of(sep);
  // 返回去除扩展名后的文件名部分
  return file_name.substr(0, end_index);
}

// 函数：将宽字符字符串转换为窄字符输出流
// 假设输入的宽字符文本为UTF-16编码
std::ostream& _strFromWide(std::ostream& ss, const std::wstring& wString);

// 如果不是在Windows平台编译
#ifndef _WIN32

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
// 函数：将宽字符字符串转换为窄字符输出流
// 使用std::codecvt进行UTF-16到UTF-8编码的转换
std::ostream& _strFromWide(std::ostream& ss, const std::wstring& wString) {
  std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;  // 创建编码转换器对象
  return _str(ss, converter.to_bytes(wString));  // 调用_str函数进行转换
}
#pragma GCC diagnostic pop

#else // #ifndef _WIN32
// 函数：将宽字符字符串转换为窄字符输出流
// 使用Windows平台的Unicode支持进行转换
std::ostream& _strFromWide(std::ostream& ss, const std::wstring& wString) {
  return _str(ss, u16u8(wString));  // 调用_str函数进行转换
}

#endif // _WIN32

// 函数：将宽字符C字符串转换为窄字符输出流
std::ostream& _str(std::ostream& ss, const wchar_t* wCStr) {
  return _strFromWide(ss, std::wstring(wCStr));  // 调用_strFromWide函数进行转换
}

// 函数：将宽字符字符转换为窄字符输出流
std::ostream& _str(std::ostream& ss, const wchar_t& wChar) {
  return _strFromWide(ss, std::wstring(1, wChar));  // 调用_strFromWide函数进行转换
}

// 函数：将宽字符字符串转换为窄字符输出流
std::ostream& _str(std::ostream& ss, const std::wstring& wString) {
  return _strFromWide(ss, wString);  // 调用_strFromWide函数进行转换
}

} // namespace detail

// 函数：重载<<操作符，用于输出SourceLocation对象信息到输出流中
std::ostream& operator<<(std::ostream& out, const SourceLocation& loc) {
  out << loc.function << " at " << loc.file << ":" << loc.line;  // 输出函数名、文件名和行号信息
  return out;  // 返回输出流
}

// 函数：替换字符串中所有的from子串为to子串
size_t ReplaceAll(std::string& s, c10::string_view from, c10::string_view to) {
  if (from.empty()) {  // 如果from为空字符串，则直接返回0
    return 0;
  }

  size_t numReplaced = 0;  // 替换计数器
  std::string::size_type last_pos = 0u;  // 上一个匹配位置
  std::string::size_type cur_pos = 0u;  // 当前匹配位置
  std::string::size_type write_pos = 0u;  // 当前写入位置
  const c10::string_view input(s);  // 输入字符串的视图

  if (from.size() >= to.size()) {
    // 如果替换字符串长度不大于原始字符串长度，可以在原地进行替换而不需要额外空间
    char* s_data = &s[0];  // 字符串的数据指针


这段代码涵盖了一些字符串处理和字符编码转换的功能，其中包括跨平台处理路径分隔符、提取文件名和去除文件扩展名等操作。
    // 在字符串 s 中查找 from 子串，并进行替换操作，替换为 to 子串
    while ((cur_pos = s.find(from.data(), last_pos, from.size())) !=
           std::string::npos) {
      ++numReplaced;
      // 如果替换位置不是起始位置，则将替换位置之前的内容复制到指定位置
      if (write_pos != last_pos) {
        std::copy(s_data + last_pos, s_data + cur_pos, s_data + write_pos);
      }
      // 更新写入位置，指向当前替换后的位置
      write_pos += cur_pos - last_pos;
      // 将 to 子串复制到指定位置
      std::copy(to.begin(), to.end(), s_data + write_pos);
      // 更新写入位置，指向插入 to 子串后的位置
      write_pos += to.size();
      // 更新搜索起始位置，从 from 子串之后的位置开始继续查找
      last_pos = cur_pos + from.size();
    }

    // 将替换完成后剩余的输入内容追加到末尾
    if (write_pos != last_pos) {
      std::copy(s_data + last_pos, s_data + input.size(), s_data + write_pos);
      // 更新写入位置，指向追加完剩余内容后的位置
      write_pos += input.size() - last_pos;
      // 调整字符串 s 的大小，截断多余的部分
      s.resize(write_pos);
    }
    // 返回替换的总次数
    return numReplaced;
  }

  // 如果未能进行原地替换，则在临时缓冲区中执行替换操作
  std::string buffer;

  // 在输入 input 中查找 from 子串，并进行替换操作
  while ((cur_pos = s.find(from.data(), last_pos, from.size())) !=
         std::string::npos) {
    ++numReplaced;
    // 将替换子串之前的内容追加到临时缓冲区中
    buffer.append(input.begin() + last_pos, input.begin() + cur_pos);
    // 将 to 子串追加到临时缓冲区中
    buffer.append(to.begin(), to.end());
    // 更新搜索起始位置，从 from 子串之后的位置开始继续查找
    last_pos = cur_pos + from.size();
  }
  // 如果没有进行任何替换操作，则直接返回 0
  if (numReplaced == 0) {
    return 0;
  }
  // 将替换完成后剩余的输入内容追加到临时缓冲区的末尾
  buffer.append(input.begin() + last_pos, input.end());
  // 将临时缓冲区的内容移动到字符串 s 中，完成替换操作
  s = std::move(buffer);
  // 返回替换的总次数
  return numReplaced;
}

// 结束 c10 命名空间的定义
} // namespace c10
```