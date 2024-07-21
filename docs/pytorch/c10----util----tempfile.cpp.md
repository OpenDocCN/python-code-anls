# `.\pytorch\c10\util\tempfile.cpp`

```py
// 引入C10库的异常处理、临时文件和格式化工具
#include <c10/util/Exception.h>
#include <c10/util/tempfile.h>
#include <fmt/format.h>

// 如果不是Windows系统，引入Unix系统相关头文件
#if !defined(_WIN32)
#include <unistd.h>
#include <cerrno>
#else // defined(_WIN32)
// 如果是Windows系统，引入Windows API相关头文件
#include <Windows.h>
#include <fcntl.h>
#include <fileapi.h>
#include <io.h>
#endif // defined(_WIN32)

// 创建传递给`mkstemp`的文件名模式
#if !defined(_WIN32)
// 如果不是Windows系统，定义创建文件名的函数
static std::string make_filename(std::string_view name_prefix) {
  // 根据http://pubs.opengroup.org/onlinepubs/009695399/functions/mkstemp.html的要求，`mkstemp`函数的文件名参数需要以"XXXXXX"结尾
  constexpr const char* kRandomPattern = "XXXXXX";

  // 检查以下环境变量是否设置，使用它们的值作为临时目录，否则默认使用`/tmp`
  const char* tmp_directory = "/tmp";
  for (const char* variable : {"TMPDIR", "TMP", "TEMP", "TEMPDIR"}) {
    if (const char* path = getenv(variable)) {
      tmp_directory = path;
      break;
    }
  }
  return fmt::format("{}/{}{}", tmp_directory, name_prefix, kRandomPattern);
}
#else
// 如果是Windows系统，定义创建文件名的函数
static std::string make_filename() {
  // 使用tmpnam_s函数生成临时文件名
  char name[L_tmpnam_s]{};
  auto res = tmpnam_s(name, L_tmpnam_s);
  // 如果生成失败，记录警告并返回空字符串
  if (res != 0) {
    TORCH_WARN("Error generating temporary file");
    return "";
  }
  return name;
}
#endif // !defined(_WIN32)

// 定义C10命名空间
namespace c10 {
/// 尝试创建临时文件，如果失败则返回`nullopt`
std::optional<TempFile> try_make_tempfile(std::string_view name_prefix) {
#if defined(_WIN32)
  // 如果是Windows系统，生成临时文件名
  auto filename = make_filename();
#else
  // 如果不是Windows系统，根据前缀生成临时文件名
  auto filename = make_filename(name_prefix);
#endif
  // 如果文件名为空，返回`nullopt`
  if (filename.empty()) {
    return std::nullopt;
  }
#if defined(_WIN32)
  // 如果是Windows系统，返回临时文件对象
  return TempFile(std::move(filename));
#else
  // 如果不是Windows系统，使用mkstemp创建临时文件
  const int fd = mkstemp(filename.data());
  // 如果创建失败，返回`nullopt`
  if (fd == -1) {
    return std::nullopt;
  }
  // 返回带文件描述符的临时文件对象
  return TempFile(std::move(filename), fd);
#endif // defined(_WIN32)
}

/// 类似于`try_make_tempfile`，但如果无法创建临时文件则抛出异常
TempFile make_tempfile(std::string_view name_prefix) {
  // 尝试创建临时文件
  if (auto tempfile = try_make_tempfile(name_prefix)) {
    return std::move(*tempfile);
  }
  // 如果创建失败，抛出异常并记录错误信息
  TORCH_CHECK(false, "Error generating temporary file: ", std::strerror(errno));
}

/// 尝试创建临时目录，如果失败则返回`nullopt`
std::optional<TempDir> try_make_tempdir(std::string_view name_prefix) {
#if defined(_WIN32)
  // 如果是Windows系统，尝试多次创建临时目录
  for (int i = 0; i < 10; i++) {
    auto dirname = make_filename();
    // 如果目录名为空，返回`nullopt`
    if (dirname.empty()) {
      return std::nullopt;
    }
    // 使用CreateDirectoryA创建目录，如果成功则返回临时目录对象
    if (CreateDirectoryA(dirname.c_str(), nullptr)) {
      return TempDir(dirname);
    }
    // 如果出现错误，检查错误码，如果是成功状态，则返回`nullopt`
    if (GetLastError() == ERROR_SUCCESS) {
      return std::nullopt;
    }
  }
  // 多次尝试后仍无法创建目录，返回`nullopt`
  return std::nullopt;
#else
  // 如果不是Windows系统，根据前缀生成临时文件名
  auto filename = make_filename(name_prefix);
  // 使用mkdtemp创建临时目录，如果失败返回`nullopt`
  const char* dirname = mkdtemp(filename.data());
  if (!dirname) {
    return std::nullopt;
  }
  // 返回临时目录对象
  return TempDir(dirname);
#endif // defined(_WIN32)
}

#if defined(_WIN32)
// 如果是Windows系统，定义TempFile类的打开方法
bool TempFile::open() {
  // 如果文件描述符不为-1，表示已经打开
  if (fd != -1) {
    return false;
  }


# 返回 false，表示函数结束，未成功打开文件



  auto err = _sopen_s(
      &fd,
      name.c_str(),
      _O_CREAT | _O_TEMPORARY | _O_EXCL | _O_BINARY | _O_RDWR,
      _SH_DENYNO,
      _S_IREAD | _S_IWRITE);


# 使用 _sopen_s 函数打开文件：
#   - &fd 是用来存放文件描述符的指针
#   - name.c_str() 提供文件名的 C 字符串表示
#   - _O_CREAT 创建文件，_O_TEMPORARY 临时文件，_O_EXCL 确保文件不存在才创建，_O_BINARY 以二进制模式打开，_O_RDWR 读写模式
#   - _SH_DENYNO 允许其他进程读写文件
#   - _S_IREAD | _S_IWRITE 设置文件权限为可读可写
# 错误码 err 将记录函数执行的错误信息



  if (err != 0) {
    fd = -1;
    return false;
  }


# 如果 _sopen_s 执行出错（err != 0），则：
#   - 将 fd 设置为 -1，表示文件描述符无效
#   - 返回 false，表示函数未成功打开文件



  return true;


# 如果 _sopen_s 执行成功，返回 true，表示函数成功打开文件并且文件描述符有效
}

#endif

/// TempFile 析构函数，用于清理临时文件资源
TempFile::~TempFile() {
  // 如果文件名不为空
  if (!name.empty()) {
    // 非 Windows 系统下
#if !defined(_WIN32)
    // 如果文件描述符有效
    if (fd >= 0) {
      // 删除文件（解除文件名关联），关闭文件描述符
      unlink(name.c_str());
      close(fd);
    }
#else
    // Windows 系统下
    if (fd >= 0) {
      // 关闭文件描述符
      _close(fd);
    }
#endif
  }
}

/// TempDir 析构函数，用于清理临时目录资源
TempDir::~TempDir() {
  // 如果目录名不为空
  if (!name.empty()) {
    // 非 Windows 系统下
#if !defined(_WIN32)
    // 删除目录
    rmdir(name.c_str());
#else // defined(_WIN32)
    // Windows 系统下
    RemoveDirectoryA(name.c_str());
#endif // defined(_WIN32)
  }
}

/// 创建临时目录，类似 `try_make_tempdir`，但如果无法创建临时目录则抛出异常
TempDir make_tempdir(std::string_view name_prefix) {
  // 尝试创建临时目录
  if (auto tempdir = try_make_tempdir(name_prefix)) {
    // 若成功创建，则移动临时目录对象并返回
    return std::move(*tempdir);
  }
  // 创建临时目录失败，根据操作系统报告错误信息
#if !defined(_WIN32)
  TORCH_CHECK(
      false, "Error generating temporary directory: ", std::strerror(errno));
#else // defined(_WIN32)
  TORCH_CHECK(false, "Error generating temporary directory");
#endif // defined(_WIN32)
}

} // namespace c10
```