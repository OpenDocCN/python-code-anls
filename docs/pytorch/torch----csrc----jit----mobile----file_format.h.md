# `.\pytorch\torch\csrc\jit\mobile\file_format.h`

```py
#pragma once

#include <array>  // 引入数组容器
#include <cerrno>  // 错误码定义
#include <cstddef>  // 定义零大小类型
#include <cstring>  // C字符串操作函数
#include <fstream>  // 文件输入输出流
#include <istream>  // 输入流
#include <memory>  // 内存管理

#include <c10/core/CPUAllocator.h>  // CPU 内存分配器
#include <c10/core/impl/alloc_cpu.h>  // CPU 内存分配实现
#include <caffe2/serialize/read_adapter_interface.h>  // 读适配器接口

#if defined(HAVE_MMAP)
#include <fcntl.h>  // 文件控制
#include <sys/mman.h>  // 内存映射
#include <sys/stat.h>  // 文件状态
#include <sys/types.h>  // 数据类型定义
#include <unistd.h>  // POSIX 操作
#endif

/**
 * @file
 *
 * Helpers for identifying file formats when reading serialized data.
 *
 * Note that these functions are declared inline because they will typically
 * only be called from one or two locations per binary.
 */

namespace torch {
namespace jit {

/**
 * The format of a file or data stream.
 */
enum class FileFormat {
  UnknownFileFormat = 0,  // 未知文件格式
  FlatbufferFileFormat,   // Flatbuffer 文件格式
  ZipFileFormat,          // ZIP 文件格式
};

/// The size of the buffer to pass to #getFileFormat(), in bytes.
constexpr size_t kFileFormatHeaderSize = 8;  // 文件格式头大小
constexpr size_t kMaxAlignment = 16;  // 最大对齐大小

/**
 * Returns the likely file format based on the magic header bytes in @p header,
 * which should contain the first bytes of a file or data stream.
 */
// NOLINTNEXTLINE(facebook-hte-NamespaceScopedStaticDeclaration)
static inline FileFormat getFileFormat(const char* data) {
  // The size of magic strings to look for in the buffer.
  static constexpr size_t kMagicSize = 4;  // 魔术字符串大小

  // Bytes 4..7 of a Flatbuffer-encoded file produced by
  // `flatbuffer_serializer.h`. (The first four bytes contain an offset to the
  // actual Flatbuffer data.)
  static constexpr std::array<char, kMagicSize> kFlatbufferMagicString = {
      'P', 'T', 'M', 'F'};  // Flatbuffer 文件的魔术字符串
  static constexpr size_t kFlatbufferMagicOffset = 4;  // Flatbuffer 文件魔术字符串偏移量

  // The first four bytes of a ZIP file.
  static constexpr std::array<char, kMagicSize> kZipMagicString = {
      'P', 'K', '\x03', '\x04'};  // ZIP 文件的魔术字符串

  // Note that we check for Flatbuffer magic first. Since the first four bytes
  // of flatbuffer data contain an offset to the root struct, it's theoretically
  // possible to construct a file whose offset looks like the ZIP magic. On the
  // other hand, bytes 4-7 of ZIP files are constrained to a small set of values
  // that do not typically cross into the printable ASCII range, so a ZIP file
  // should never have a header that looks like a Flatbuffer file.
  
  // 首先检查 Flatbuffer 魔术字符串。由于 Flatbuffer 数据的前四个字节包含到根结构的偏移量，
  // 理论上可以构造一个文件，其偏移量看起来像 ZIP 文件的魔术字符串。
  // 另一方面，ZIP 文件的字节 4 到 7 限制在一个小的值集合中，通常不会跨越到可打印的 ASCII 范围，
  // 因此一个 ZIP 文件不应该有一个看起来像 Flatbuffer 文件的头部。

  if (std::memcmp(
          data + kFlatbufferMagicOffset,
          kFlatbufferMagicString.data(),
          kMagicSize) == 0) {
    // Magic header for a binary file containing a Flatbuffer-serialized mobile
    // Module.
    return FileFormat::FlatbufferFileFormat;  // 返回 Flatbuffer 文件格式
  } else if (std::memcmp(data, kZipMagicString.data(), kMagicSize) == 0) {
    // Magic header for a zip file, which we use to store pickled sub-files.
    return FileFormat::ZipFileFormat;  // 返回 ZIP 文件格式
  }
  return FileFormat::UnknownFileFormat;  // 返回未知文件格式
}
/**
 * Returns the likely file format based on the magic header bytes of @p data.
 * If the stream position changes while inspecting the data, this function will
 * restore the stream position to its original offset before returning.
 */
// NOLINTNEXTLINE(facebook-hte-NamespaceScopedStaticDeclaration)
static inline FileFormat getFileFormat(std::istream& data) {
  // 初始设定文件格式为未知
  FileFormat format = FileFormat::UnknownFileFormat;
  // 记录当前流的位置
  std::streampos orig_pos = data.tellg();
  // 读取文件格式头部的固定字节数组
  std::array<char, kFileFormatHeaderSize> header;
  data.read(header.data(), header.size());
  // 如果读取操作成功
  if (data.good()) {
    // 通过读取到的头部数据获取文件格式
    format = getFileFormat(header.data());
  }
  // 恢复到初始位置
  data.seekg(orig_pos, data.beg);
  return format;
}

/**
 * Returns the likely file format based on the magic header bytes of the file
 * named @p filename.
 */
// NOLINTNEXTLINE(facebook-hte-NamespaceScopedStaticDeclaration)
static inline FileFormat getFileFormat(const std::string& filename) {
  // 使用二进制模式打开文件流
  std::ifstream data(filename, std::ifstream::binary);
  // 调用上面的函数获取文件格式
  return getFileFormat(data);
}

// NOLINTNEXTLINE(facebook-hte-NamespaceScopedStaticDeclaration)
static void file_not_found_error() {
  // 构造文件打开错误信息
  std::stringstream message;
  message << "Error while opening file: ";
  // 根据错误号码设置错误描述
  if (errno == ENOENT) {
    message << "no such file or directory" << std::endl;
  } else {
    message << "error no is: " << errno << std::endl;
  }
  // 使用 TORCH_CHECK 抛出错误并包含错误信息
  TORCH_CHECK(false, message.str());
}

// NOLINTNEXTLINE(facebook-hte-NamespaceScopedStaticDeclaration)
static inline std::tuple<std::shared_ptr<char>, size_t> get_file_content(
    const char* filename) {
#if defined(HAVE_MMAP)
  // 使用 mmap 映射文件到内存
  int fd = open(filename, O_RDONLY);
  if (fd < 0) {
    // 打开文件失败，抛出文件未找到的错误
    file_not_found_error();
  }
  struct stat statbuf {};
  // 获取文件状态信息
  fstat(fd, &statbuf);
  size_t size = statbuf.st_size;
  // 在内存中映射文件内容
  void* ptr = mmap(nullptr, statbuf.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  // 定义释放函数，并使用共享指针管理映射内存
  auto deleter = [statbuf](char* ptr) { munmap(ptr, statbuf.st_size); };
  std::shared_ptr<char> data(reinterpret_cast<char*>(ptr), deleter);
#else
  // 使用标准文件流打开文件
  FILE* f = fopen(filename, "rb");
  if (f == nullptr) {
    // 打开文件失败，抛出文件未找到的错误
    file_not_found_error();
  }
  fseek(f, 0, SEEK_END);
  size_t size = ftell(f);
  fseek(f, 0, SEEK_SET);
  // 确保缓冲区大小是对齐的倍数
  size_t buffer_size = (size / kMaxAlignment + 1) * kMaxAlignment;
  // 使用共享指针分配内存，并指定释放函数
  std::shared_ptr<char> data(
      static_cast<char*>(c10::alloc_cpu(buffer_size)), c10::free_cpu);
  fread(data.get(), size, 1, f);
  fclose(f);
#endif
  return std::make_tuple(data, size);
}

// NOLINTNEXTLINE(facebook-hte-NamespaceScopedStaticDeclaration)
static inline std::tuple<std::shared_ptr<char>, size_t> get_stream_content(
    // 获取流的大小并重置到原始位置
    // 获取当前流的位置并保存到 orig_pos
    std::streampos orig_pos = in.tellg();
    // 将流的位置设置到末尾以获取流的总大小
    in.seekg(orig_pos, std::ios::end);
    // 获取流的总大小
    const long size = in.tellg();
    // 将流的位置重置到原始位置
    in.seekg(orig_pos, in.beg);
    
    // 读取流的内容
    // NOLINT 确保缓冲区大小是对齐的倍数
    // 计算需要的缓冲区大小，确保是 kMaxAlignment 的倍数
    size_t buffer_size = (size / kMaxAlignment + 1) * kMaxAlignment;
    // 使用 c10::alloc_cpu 分配共享内存指针 data，并在数据不再需要时调用 c10::free_cpu 释放
    std::shared_ptr<char> data(
        static_cast<char*>(c10::alloc_cpu(buffer_size)), c10::free_cpu);
    // 从流中读取 size 大小的数据到 data 指向的缓冲区
    in.read(data.get(), size);
    
    // 将流的位置恢复到原始位置
    in.seekg(orig_pos, in.beg);
    // 返回包含数据和大小的元组
    return std::make_tuple(data, size);
}

// NOLINTNEXTLINE(facebook-hte-NamespaceScopedStaticDeclaration)
// 定义一个静态内联函数，从给定的ReadAdapterInterface对象中获取数据内容
static inline std::tuple<std::shared_ptr<char>, size_t> get_rai_content(
    caffe2::serialize::ReadAdapterInterface* rai) {
  // 计算缓冲区大小，确保足够容纳ReadAdapterInterface对象的大小
  size_t buffer_size = (rai->size() / kMaxAlignment + 1) * kMaxAlignment;
  // 使用c10库分配CPU内存，将其封装在shared_ptr<char>中，使用c10::free_cpu释放
  std::shared_ptr<char> data(
      static_cast<char*>(c10::alloc_cpu(buffer_size)), c10::free_cpu);
  // 从ReadAdapterInterface对象中读取数据到分配的内存中
  rai->read(
      0, data.get(), rai->size(), "Loading ReadAdapterInterface to bytes");
  // 返回包含数据指针和缓冲区大小的元组
  return std::make_tuple(data, buffer_size);
}

} // namespace jit
} // namespace torch
```