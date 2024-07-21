# `.\pytorch\torch\csrc\api\src\optim\serialize.cpp`

```
// 包含 Torch 库中的序列化相关头文件
#include <torch/optim/serialize.h>

// 包含 Torch 库中的归档操作相关头文件
#include <torch/serialize/archive.h>

// 包含 Torch 库中的基本数据类型定义
#include <torch/types.h>

// 包含标准 C++ 库头文件
#include <cstddef>
#include <cstdint>
#include <deque>
#include <string>
#include <vector>

// Torch 命名空间
namespace torch {
// Torch 优化器命名空间
namespace optim {

// 序列化函数，将 int64_t 类型的值写入归档
void serialize(
    serialize::OutputArchive& archive,    // 输出归档对象的引用
    const std::string& key,               // 键值对中的键
    const int64_t& value) {               // 要写入的 int64_t 类型值
  archive.write(key, IValue(value));      // 使用归档对象写入键值对
}

// 反序列化函数，从归档中读取 int64_t 类型的值
void serialize(
    serialize::InputArchive& archive,     // 输入归档对象的引用
    const std::string& key,               // 键值对中的键
    int64_t& value) {                     // 读取结果存储的 int64_t 引用
  IValue ivalue;                          // 定义存储值的 IValue 对象
  archive.read(key, ivalue);              // 使用归档对象读取指定键的值
  value = ivalue.toInt();                 // 将读取到的值转换为 int64_t 并存储在 value 中
}

// 序列化函数，将 std::vector<int64_t> 类型的值写入归档
void serialize(
    serialize::OutputArchive& archive,    // 输出归档对象的引用
    const std::string& key,               // 键值对中的键
    const std::vector<int64_t>& steps) {  // 要写入的 int64_t 向量
  std::vector<torch::Tensor> tensors;     // 创建存储 Tensor 的向量
  tensors.reserve(steps.size());          // 预留足够的空间以容纳所有步骤

  // 遍历每个步骤，将其转换为 Tensor 并添加到 tensors 中
  for (const auto& step : steps) {
    tensors.push_back(torch::tensor(static_cast<int64_t>(step)));
  }

  serialize(archive, key, tensors);       // 调用重载的 serialize 函数，将 tensors 写入归档
}

// 反序列化函数，从归档中读取 std::vector<int64_t> 类型的值
void serialize(
    serialize::InputArchive& archive,     // 输入归档对象的引用
    const std::string& key,               // 键值对中的键
    std::vector<int64_t>& steps) {        // 读取结果存储的 int64_t 向量引用
  steps.clear();                          // 清空步骤向量
  std::vector<torch::Tensor> tensors;     // 创建存储 Tensor 的向量

  serialize(archive, key, tensors);       // 使用归档对象读取键为 key 的数据到 tensors 中

  // 将每个 Tensor 转换为 int64_t 并添加到 steps 中
  for (const auto& step : tensors) {
    steps.push_back(step.item<int64_t>());
  }
}

} // namespace optim
} // namespace torch
```