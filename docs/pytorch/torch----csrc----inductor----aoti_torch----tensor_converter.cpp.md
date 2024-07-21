# `.\pytorch\torch\csrc\inductor\aoti_torch\tensor_converter.cpp`

```py
// 引入 Torch 的头文件，用于张量转换和实用函数
#include <torch/csrc/inductor/aoti_torch/tensor_converter.h>
#include <torch/csrc/inductor/aoti_torch/utils.h>

// 定义 Torch 命名空间中的 aot_inductor 命名空间
namespace torch {
namespace aot_inductor {

// 从给定张量列表中不安全地分配新的张量句柄
std::vector<AtenTensorHandle> unsafe_alloc_new_handles_from_tensors(
    std::vector<at::Tensor>& tensors) {
  // 初始化结果向量，预留足够的空间以容纳所有张量
  std::vector<AtenTensorHandle> result;
  result.reserve(tensors.size());
  
  // 遍历每个张量
  for (auto tensor : tensors) {
    // 为每个张量分配新的内存，并将其移动到新的位置
    auto allocated = new at::Tensor(std::move(tensor));
    // 将分配的张量句柄转换为张量句柄对象，并添加到结果向量中
    result.push_back(tensor_pointer_to_tensor_handle(allocated));
  }
  
  // 返回结果向量
  return result;
}

// 通过从句柄数组中窃取张量对象来分配张量
std::vector<at::Tensor> alloc_tensors_by_stealing_from_handles(
    AtenTensorHandle* handles,
    size_t length) {
  // 使用哈希映射记录每个句柄的最后已知索引，以检测重复句柄
  std::unordered_map<AtenTensorHandle, size_t> lastKnownIdx;
  for (size_t i = 0; i < length; i++) {
    lastKnownIdx[handles[i]] = i;
  }

  // 初始化结果张量向量，预留足够的空间以容纳所有张量
  std::vector<at::Tensor> result;
  result.reserve(length);

  // 遍历每个句柄
  for (size_t i = 0; i < length; i++) {
    // 如果句柄为空指针，则在结果中添加一个空张量并继续
    if (handles[i] == nullptr) {
      result.emplace_back();
      continue;
    }

    // 将句柄转换为张量对象，并复制到结果中
    at::Tensor tensor = *tensor_handle_to_tensor_pointer(handles[i]);

    // 如果当前句柄不是最后已知的索引，则复制张量对象
    if (lastKnownIdx[handles[i]] != i) {
      result.emplace_back(tensor);
    } else {
      // 否则，移动张量对象并删除相应的张量对象
      result.emplace_back(std::move(tensor));
      aoti_torch_delete_tensor_object(handles[i]);
    }

    // 将处理过的句柄置为空指针，避免重复处理
    handles[i] = nullptr;
  }

  // 返回结果张量向量
  return result;
}

} // namespace aot_inductor
} // namespace torch
```