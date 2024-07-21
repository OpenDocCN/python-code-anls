# `.\pytorch\torch\csrc\api\include\torch\data\samplers\custom_batch_request.h`

```
#pragma once
// 预处理指令：确保头文件只被编译一次

#include <torch/csrc/Export.h>
// 包含 Torch 库的导出定义头文件

#include <cstddef>
// 包含标准库的 cstddef 头文件，定义了 size_t 等

namespace torch {
namespace data {
namespace samplers {
/// A base class for custom index types.
// 自定义索引类型的基类

struct TORCH_API CustomBatchRequest {
  CustomBatchRequest() = default;
  // 默认构造函数

  CustomBatchRequest(const CustomBatchRequest&) = default;
  // 拷贝构造函数

  CustomBatchRequest(CustomBatchRequest&&) noexcept = default;
  // 移动构造函数（无异常保证）

  virtual ~CustomBatchRequest() = default;
  // 虚析构函数

  /// The number of elements accessed by this index.
  // 返回此索引访问的元素数量的纯虚函数
  virtual size_t size() const = 0;
  // 返回类型为 size_t 的整数，表示索引访问的元素数量
};
} // namespace samplers
} // namespace data
} // namespace torch
```