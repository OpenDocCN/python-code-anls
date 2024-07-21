# `.\pytorch\test\cpp\tensorexpr\padded_buffer.cpp`

```
// 包含所需的头文件：定义了 PaddedBuffer 类的实现
#include "test/cpp/tensorexpr/padded_buffer.h"

// 包含日志记录和其它实用工具的头文件
#include <c10/util/Logging.h>
#include <c10/util/irange.h>
#include <sstream>

// 定义命名空间 torch::jit::tensorexpr，用于存放所有相关的类和函数
namespace torch {
namespace jit {
namespace tensorexpr {

// 计算索引值的函数，根据给定的索引向量计算出在 PaddedBuffer 中的总索引
int PaddedBufferBase::Index(const std::vector<int>& indices) const {
  // 断言维度数必须匹配，以保证索引有效性
  TORCH_DCHECK_EQ(dims_.size(), indices.size());
  // 计算总索引的初始值
  int total_index = 0;
  // 遍历维度数组，计算出每个维度的偏移量乘积并累加到总索引中
  for (const auto i : c10::irange(dims_.size())) {
    total_index += indices[i] * strides_[i];
  }
  // 返回计算得到的总索引
  return total_index;
}

// 构造函数：根据给定的维度和名称初始化 PaddedBufferBase 对象
PaddedBufferBase::PaddedBufferBase(
    const std::vector<int>& dims,
    // 使用 const 引用接收名称参数，避免不必要的复制
    const std::string& name)
    : dims_(dims), name_(name), strides_(dims.size()) {
  // 计算各维度的步长（strides）
  for (int i = (int)dims.size() - 1; i >= 0; --i) {
    // 如果是最后一个维度，步长为1
    if (i == (int)dims.size() - 1) {
      strides_[i] = 1;
    } else {
      // 计算当前维度的步长，即后一维度步长乘以后一维度大小
      strides_[i] = strides_[i + 1] * dims[i + 1];
    }
  }
  // 计算整个 PaddedBuffer 的总大小，即第一维度步长乘以第一维度大小
  total_size_ = strides_[0] * dims[0];
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
```