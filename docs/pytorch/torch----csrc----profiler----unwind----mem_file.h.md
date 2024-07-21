# `.\pytorch\torch\csrc\profiler\unwind\mem_file.h`

```py
// 版权声明和许可证信息，这段代码的版权归 Meta Platforms, Inc. 及其关联公司所有
// 在根目录下的 LICENSE 文件中可以找到 BSD 风格的许可证

#pragma once

#include <elf.h>  // ELF 头文件
#include <fcntl.h>  // 文件控制操作
#include <fmt/format.h>  // 格式化输出库
#include <sys/mman.h>  // 内存映射操作
#include <sys/stat.h>  // 文件状态操作
#include <torch/csrc/profiler/unwind/lexer.h>  // 解析器头文件
#include <torch/csrc/profiler/unwind/unwind_error.h>  // 错误处理头文件
#include <unistd.h>  // POSIX 系统调用 API
#include <cerrno>  // 错误号
#include <cstdio>  // C 标准 I/O 头文件
#include <cstring>  // 字符串操作
#include <iostream>  // 标准输入输出流

namespace torch::unwind {

struct Section {
  char* data = nullptr;  // 数据指针，初始化为空
  size_t size = 0;  // 数据大小，默认为 0

  // 返回指定偏移量处的 C 字符串
  const char* string(size_t offset) {
    return lexer(offset).readCString();
  }

  // 创建一个 CheckedLexer 对象，用于解析数据流
  CheckedLexer lexer(size_t offset) {
    return CheckedLexer(data + offset, data, data + size);
  }
};

/// 将文件以只读方式映射到地址空间，并管理映射的生命周期。以下是一些使用情况：
/// 1. 在加载器中用于读取初始镜像，并在调用 dlopen 之前检查 ELF 文件的依赖项。
///
/// 2. 在 Unity 中用于加载 ELF 文件。
struct MemFile {
  explicit MemFile(const char* filename_)
      : fd_(open(filename_, O_RDONLY)),  // 打开文件，只读模式
        mem_(nullptr),  // 内存映射指针初始化为空
        n_bytes_(0),  // 文件大小初始化为 0
        name_(filename_) {  // 记录文件名
    // 检查文件是否成功打开，否则抛出错误信息
    UNWIND_CHECK(fd_ != -1, "failed to open {}: {}", filename_, strerror(errno));

    // 获取文件状态信息
    struct stat s;
    if (-1 == fstat(fd_, &s)) {
      close(fd_);  // 如果获取状态失败，关闭文件描述符
      UNWIND_CHECK(false, "failed to stat {}: {}", filename_, strerror(errno));
    }
    n_bytes_ = s.st_size;  // 记录文件大小
    // 检查文件是否为空的共享库文件
    UNWIND_CHECK(n_bytes_ > sizeof(Elf64_Ehdr), "empty shared library: {}", filename_);

    // 将文件映射到内存中，只读模式
    mem_ = (char*)mmap(nullptr, n_bytes_, PROT_READ, MAP_SHARED, fd_, 0);
    if (MAP_FAILED == mem_) {
      close(fd_);  // 映射失败时关闭文件描述符
      UNWIND_CHECK(false, "failed to mmap {}: {}", filename_, strerror(errno));
    }

    ehdr_ = (Elf64_Ehdr*)mem_;  // ELF 头部指针

    // 宏定义检查 ELF 文件的各种属性
#define ELF_CHECK(cond) UNWIND_CHECK(cond, "not an ELF file: {}", filename_)
    ELF_CHECK(ehdr_->e_ident[EI_MAG0] == ELFMAG0);
    ELF_CHECK(ehdr_->e_ident[EI_MAG1] == ELFMAG1);
    ELF_CHECK(ehdr_->e_ident[EI_MAG2] == ELFMAG2);
    ELF_CHECK(ehdr_->e_ident[EI_MAG3] == ELFMAG3);
    ELF_CHECK(ehdr_->e_ident[EI_CLASS] == ELFCLASS64);
    ELF_CHECK(ehdr_->e_ident[EI_VERSION] == EV_CURRENT);
    ELF_CHECK(ehdr_->e_version == EV_CURRENT);
    ELF_CHECK(ehdr_->e_machine == EM_X86_64);
#undef ELF_CHECK

    // 检查节头表的有效性
    UNWIND_CHECK(
        ehdr_->e_shoff + sizeof(Elf64_Shdr) * ehdr_->e_shnum <= n_bytes_,
        "invalid section header table {} {} {}",
        ehdr_->e_shoff + sizeof(Elf64_Shdr) * ehdr_->e_shnum,
        n_bytes_,
        ehdr_->e_shnum);

    shdr_ = (Elf64_Shdr*)(mem_ + ehdr_->e_shoff);  // 节头指针

    // 检查节字符串表偏移量的有效性
    UNWIND_CHECK(
        ehdr_->e_shstrndx < ehdr_->e_shnum, "invalid strtab section offset");

    auto& strtab_hdr = shdr_[ehdr_->e_shstrndx];  // 字符串表头部信息
  }

  // 禁用复制构造函数
  MemFile(const MemFile&) = delete;
  // 禁用赋值运算符
  MemFile& operator=(const MemFile&) = delete;
  // 返回内部数据的指针，转换为字符指针
  [[nodiscard]] const char* data() const {
    return (const char*)mem_;
  }

  /// 返回底层文件描述符是否有效
  int valid() {
    // 使用 fcntl 函数检查文件描述符的有效性
    return fcntl(fd_, F_GETFD) != -1 || errno != EBADF;
  }

  ~MemFile() {
    if (mem_) {
      // 如果内存指针不为空，解除映射
      munmap((void*)mem_, n_bytes_);
    }
    if (fd_) {
      // 如果文件描述符不为零，关闭文件
      close(fd_);
    }
  }

  /// 返回 `MemFile` 定义的底层文件的大小
  size_t size() {
    return n_bytes_;
  }
  [[nodiscard]] int fd() const {
    return fd_;
  }

  // 根据给定的 Elf64_Shdr 结构体，返回对应的 Section 对象
  Section getSection(const Elf64_Shdr& shdr) {
    // 检查偏移量和大小是否在有效范围内
    UNWIND_CHECK(shdr.sh_offset + shdr.sh_size <= n_bytes_, "invalid section");
    return Section{mem_ + shdr.sh_offset, shdr.sh_size};
  }

  // 根据节名称和可选标志，返回对应的 Section 对象
  Section getSection(const char* name, bool optional) {
    for (int i = 0; i < ehdr_->e_shnum; i++) {
      // 比较节名称是否匹配
      if (strcmp(strtab_.string(shdr_[i].sh_name), name) == 0) {
        return getSection(shdr_[i]);
      }
    }
    // 如果未找到节并且不是可选的，则抛出错误
    UNWIND_CHECK(optional, "{} has no section {}", name_, name);
    return Section{nullptr, 0};
  }

  // 返回字符串表 Section 对象
  Section strtab() {
    return strtab_;
  }

 private:
  // 根据偏移量加载泛型类型 T 的数据
  template <typename T>
  T* load(size_t offset) {
    // 检查偏移量是否在有效范围内
    UNWIND_CHECK(offset < n_bytes_, "out of range");
    return (T*)(mem_ + offset);
  }
  int fd_;
  char* mem_;
  size_t n_bytes_;
  std::string name_;
  Elf64_Ehdr* ehdr_;
  Elf64_Shdr* shdr_;
  Section strtab_ = {nullptr, 0};
};

} // namespace torch::unwind
```