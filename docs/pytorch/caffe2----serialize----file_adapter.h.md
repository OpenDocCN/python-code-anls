# `.\pytorch\caffe2\serialize\file_adapter.h`

```
#pragma once
// 预处理指令，确保头文件只被包含一次

#include <fstream>
// 包含文件流操作相关的标准库头文件

#include <memory>
// 包含内存管理相关的标准库头文件

#include <c10/macros/Macros.h>
// 包含C10宏定义相关的头文件

#include "caffe2/serialize/istream_adapter.h"
// 包含自定义的istream适配器头文件

#include "caffe2/serialize/read_adapter_interface.h"
// 包含自定义的读取适配器接口头文件

namespace caffe2 {
namespace serialize {

class TORCH_API FileAdapter final : public ReadAdapterInterface {
// 定义FileAdapter类，继承自ReadAdapterInterface接口，并且是最终类（不能被继承）

 public:
  C10_DISABLE_COPY_AND_ASSIGN(FileAdapter);
  // 禁用复制构造函数和赋值操作符

  explicit FileAdapter(const std::string& file_name);
  // 显式声明构造函数，接受一个字符串类型的文件名作为参数

  size_t size() const override;
  // 重写父类的虚函数size()，返回文件大小，不改变对象状态

  size_t read(uint64_t pos, void* buf, size_t n, const char* what = "")
      const override;
  // 重写父类的虚函数read()，从文件指定位置读取数据到缓冲区，返回读取的字节数

  ~FileAdapter() override;
  // 虚析构函数，用于释放资源

 private:
  // An RAII Wrapper for a FILE pointer. Closes on destruction.
  // 一个RAII包装的FILE指针，析构时关闭文件

  struct RAIIFile {
    FILE* fp_;
    // FILE指针成员变量

    explicit RAIIFile(const std::string& file_name);
    // 显式声明构造函数，接受一个字符串类型的文件名作为参数

    ~RAIIFile();
    // 析构函数，用于关闭文件
  };

  RAIIFile file_;
  // RAIIFile结构体对象，用于管理文件资源

  // The size of the opened file in bytes
  uint64_t size_;
  // 文件的大小，以字节为单位
};

} // namespace serialize
} // namespace caffe2
```