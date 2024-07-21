# `.\pytorch\torch\csrc\distributed\c10d\Utils.cpp`

```py
#include <torch/csrc/distributed/c10d/Utils.hpp>
// 引入 Torch 库中分布式相关工具的头文件

#include <cstring>
// 引入 C 字符串操作相关的头文件

namespace c10d {
// 进入 c10d 命名空间

std::vector<at::Tensor> getTensorShapes(
    const std::vector<at::Tensor>& tensors) {
  // 定义函数 getTensorShapes，接收一个张量向量作为输入参数，返回一个张量向量
  std::vector<at::Tensor> shapeTensors;
  // 创建一个空的张量向量 shapeTensors，用于存储形状张量
  shapeTensors.reserve(tensors.size());
  // 预留空间以容纳与输入张量向量相同数量的元素

  for (const auto& tensor : tensors) {
    // 遍历输入的张量向量中的每一个张量
    // 使用 at::tensor() 复制 sizes() 下的数据，因为它可能在其他地方释放。
    at::Tensor shapesTensor =
        at::tensor(tensor.sizes(), at::TensorOptions().dtype(at::kLong));
    // 创建一个具有指定大小和数据类型（长整型）的张量 shapesTensor
    shapeTensors.emplace_back(std::move(shapesTensor));
    // 将创建的张量移动到 shapeTensors 的末尾
  }

  return shapeTensors;
  // 返回包含所有形状张量的张量向量
}

size_t getTensorsNumel(const std::vector<at::Tensor>& tensors) {
  // 定义函数 getTensorsNumel，接收一个张量向量作为输入参数，返回一个 size_t 类型的值
  size_t numel = 0;
  // 初始化 numel 为 0，用于计算所有张量的元素总数

  for (auto& tensor : tensors) {
    // 遍历输入的张量向量中的每一个张量
    numel += tensor.numel();
    // 累加每个张量的元素数量到 numel 变量
  }

  return numel;
  // 返回所有张量的总元素数量
}

} // namespace c10d
// 结束 c10d 命名空间
```