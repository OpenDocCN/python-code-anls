# `.\pytorch\torch\csrc\jit\codegen\fuser\cpu\temp_file.h`

```
#pragma once
// 只允许此头文件被包含一次

#include <ATen/ATen.h>
#include <ATen/Utils.h>
#include <c10/util/Exception.h>
#include <torch/csrc/Export.h>
// 引入需要的头文件

#ifdef _WIN32
#include <WinError.h>
#include <c10/util/Unicode.h>
#include <c10/util/win32-headers.h>
#include <fcntl.h>
#include <io.h>
#include <process.h>
#include <stdio.h>
#include <sys/stat.h>
#include <random>
#else
#include <unistd.h>
#endif
// 根据操作系统不同引入不同的头文件

#include <string>
#include <vector>
// 引入标准库头文件

namespace torch {
namespace jit {
namespace fuser {
namespace cpu {

#ifdef _MSC_VER
// 如果是在 Visual Studio 编译环境下

int wmkstemps(wchar_t* tmpl, int suffix_len) {
  // 创建带随机后缀的临时文件
  int len;
  wchar_t* name;
  int fd = -1;
  int save_errno = errno;

  len = wcslen(tmpl);
  // 检查模板长度和后缀长度是否合适
  if (len < 6 + suffix_len ||
      wcsncmp(&tmpl[len - 6 - suffix_len], L"XXXXXX", 6)) {
    return -1;
  }

  name = &tmpl[len - 6 - suffix_len];

  std::random_device rd;
  do {
    // 生成随机文件名后缀
    for (unsigned i = 0; i < 6; ++i) {
      name[i] = "abcdefghijklmnopqrstuvwxyz0123456789"[rd() % 36];
    }

    // 尝试打开文件，如果存在则重新生成文件名后缀
    fd = _wopen(tmpl, _O_RDWR | _O_CREAT | _O_EXCL, _S_IWRITE | _S_IREAD);
  } while (errno == EEXIST);

  if (fd >= 0) {
    errno = save_errno;
    return fd;
  } else {
    return -1;
  }
}
#endif

struct TempFile {
  AT_DISALLOW_COPY_AND_ASSIGN(TempFile);
  // 禁止拷贝和赋值操作

  TempFile(const std::string& t, int suffix) {
#ifdef _MSC_VER
    // 如果是在 Visual Studio 编译环境下
    auto wt = c10::u8u16(t);
    std::vector<wchar_t> tt(wt.c_str(), wt.c_str() + wt.size() + 1);
    int fd = wmkstemps(tt.data(), suffix);
    AT_ASSERT(fd != -1);
    file_ = _wfdopen(fd, L"r+");
    auto wname = std::wstring(tt.begin(), tt.end() - 1);
    name_ = c10::u16u8(wname);
#else
    // 在 Unix-like 系统下
    // mkstemps 会修改其第一个参数，所以我们在这里做一个副本
    std::vector<char> tt(t.c_str(), t.c_str() + t.size() + 1);
    int fd = mkstemps(tt.data(), suffix);
    AT_ASSERT(fd != -1);
    file_ = fdopen(fd, "r+");
    // -1 是因为 tt.size() 包含空字符，而 std::string 不包含
    name_ = std::string(tt.begin(), tt.end() - 1);
#endif
  }

  const std::string& name() const {
    return name_;
  }
  // 返回临时文件名

  void sync() {
    fflush(file_);
  }
  // 刷新文件流

  void write(const std::string& str) {
    size_t result = fwrite(str.c_str(), 1, str.size(), file_);
    AT_ASSERT(str.size() == result);
  }
  // 将字符串写入文件

#ifdef _MSC_VER
  void close() {
    if (file_ != nullptr) {
      fclose(file_);
    }
    file_ = nullptr;
  }
#endif
  // 在 Visual Studio 下关闭文件流

  FILE* file() {
    return file_;
  }
  // 返回文件流指针

  ~TempFile() {
#ifdef _MSC_VER
    // 在 Visual Studio 下析构函数
    if (file_ != nullptr) {
      fclose(file_);
    }
    auto wname = c10::u8u16(name_);
    // 如果文件名非空且存在，则删除临时文件
    if (!wname.empty() && _waccess(wname.c_str(), 0) != -1) {
      _wunlink(wname.c_str());
    }
#else
    // 在 Unix-like 系统下析构函数
    if (file_ != nullptr) {
      // 先删除文件以防止在 close 和 unlink 之间发生 mkstemps 竞争
      unlink(name_.c_str());
      fclose(file_);
    }
#endif
  }

 private:
  FILE* file_ = nullptr;
  std::string name_;
};

} // namespace cpu
} // namespace fuser
} // namespace jit
} // namespace torch
// 命名空间结束
```