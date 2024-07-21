# `.\pytorch\c10\util\tempfile.h`

```
#pragma once

#include <c10/macros/Export.h> // 引入 c10 库的导出宏定义

#include <optional> // 引入 optional 类型支持
#include <string> // 引入字符串支持
#include <string_view> // 引入字符串视图支持
#include <utility> // 引入实用工具支持

namespace c10 {
struct C10_API TempFile {
  TempFile(std::string_view name, int fd = -1) noexcept : fd(fd), name(name) {} // TempFile 构造函数，接受文件名和文件描述符参数
  TempFile(const TempFile&) = delete; // 删除复制构造函数
  TempFile(TempFile&& other) noexcept // 移动构造函数
      : fd(other.fd), name(std::move(other.name)) {
    other.fd = -1; // 将原对象的文件描述符置为无效
  }

  TempFile& operator=(const TempFile&) = delete; // 删除赋值运算符
  TempFile& operator=(TempFile&& other) noexcept { // 移动赋值运算符
    fd = other.fd; // 移动文件描述符
    name = std::move(other.name); // 移动文件名
    other.fd = -1; // 将原对象的文件描述符置为无效
    return *this;
  }

#if defined(_WIN32)
  bool open(); // 在 Windows 下打开临时文件的函数声明
#endif

  ~TempFile(); // 析构函数，用于释放资源

  int fd; // 文件描述符

  std::string name; // 文件名
};

struct C10_API TempDir {
  TempDir() = delete; // 删除默认构造函数
  explicit TempDir(std::string_view name) noexcept : name(name) {} // TempDir 构造函数，接受目录名参数
  TempDir(const TempDir&) = delete; // 删除复制构造函数
  TempDir(TempDir&& other) noexcept : name(std::move(other.name)) { // 移动构造函数
    other.name.clear(); // 清空原对象的目录名
  }

  TempDir& operator=(const TempDir&) = delete; // 删除赋值运算符
  TempDir& operator=(TempDir&& other) noexcept { // 移动赋值运算符
    name = std::move(other.name); // 移动目录名
    return *this;
  }

  ~TempDir(); // 析构函数，用于释放资源

  std::string name; // 目录名
};

/// Attempts to return a temporary file or returns `nullopt` if an error
/// occurred.
///
/// The file returned follows the pattern
/// `<tmp-dir>/<name-prefix><random-pattern>`, where `<tmp-dir>` is the value of
/// the `"TMPDIR"`, `"TMP"`, `"TEMP"` or
/// `"TEMPDIR"` environment variable if any is set, or otherwise `/tmp`;
/// `<name-prefix>` is the value supplied to this function, and
/// `<random-pattern>` is a random sequence of numbers.
/// On Windows, `name_prefix` is ignored and `tmpnam_s` is used,
/// and no temporary file is opened.
C10_API std::optional<TempFile> try_make_tempfile(
    std::string_view name_prefix = "torch-file-"); // 尝试创建临时文件的函数声明

/// Like `try_make_tempfile`, but throws an exception if a temporary file could
/// not be returned.
C10_API TempFile make_tempfile(std::string_view name_prefix = "torch-file-"); // 创建临时文件的函数声明，如果创建失败则抛出异常

/// Attempts to return a temporary directory or returns `nullopt` if an error
/// occurred.
///
/// The directory returned follows the pattern
/// `<tmp-dir>/<name-prefix><random-pattern>/`, where `<tmp-dir>` is the value
/// of the `"TMPDIR"`, `"TMP"`, `"TEMP"` or
/// `"TEMPDIR"` environment variable if any is set, or otherwise `/tmp`;
/// `<name-prefix>` is the value supplied to this function, and
/// `<random-pattern>` is a random sequence of numbers.
/// On Windows, `name_prefix` is ignored.
C10_API std::optional<TempDir> try_make_tempdir(
    std::string_view name_prefix = "torch-dir-"); // 尝试创建临时目录的函数声明

/// Like `try_make_tempdir`, but throws an exception if a temporary directory
/// could not be returned.
C10_API TempDir make_tempdir(std::string_view name_prefix = "torch-dir-"); // 创建临时目录的函数声明，如果创建失败则抛出异常
} // namespace c10
```