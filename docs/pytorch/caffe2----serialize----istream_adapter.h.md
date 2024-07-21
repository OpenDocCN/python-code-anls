# `.\pytorch\caffe2\serialize\istream_adapter.h`

```
#pragma once
// 预处理指令：确保本头文件只被编译一次

#include <istream>
// 包含标准输入流头文件

#include "c10/macros/Macros.h"
// 包含 Caffe2 的宏定义头文件

#include "caffe2/serialize/read_adapter_interface.h"
// 包含读取适配器接口的头文件

namespace caffe2 {
namespace serialize {

// 这是由 std::istream 实现的读取器
class TORCH_API IStreamAdapter final : public ReadAdapterInterface {
  // IStreamAdapter 类的声明，继承自 ReadAdapterInterface 接口

 public:
  C10_DISABLE_COPY_AND_ASSIGN(IStreamAdapter);
  // 禁用复制和赋值操作

  explicit IStreamAdapter(std::istream* istream);
  // 显式构造函数，接受一个指向 std::istream 的指针作为参数

  size_t size() const override;
  // 重写 size 方法，返回数据流的大小

  size_t read(uint64_t pos, void* buf, size_t n, const char* what = "")
      const override;
  // 重写 read 方法，从指定位置 pos 开始读取 n 字节数据到 buf 中，可选的 what 参数用于指定读取操作的描述信息

  ~IStreamAdapter() override;
  // 析构函数声明，执行资源的清理工作

 private:
  std::istream* istream_;
  // 私有成员变量，指向 std::istream 的指针，用于操作输入流

  void validate(const char* what) const;
  // 私有方法声明，用于验证操作的有效性
};

} // namespace serialize
} // namespace caffe2
```