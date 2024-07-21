# `.\pytorch\test\cpp_extensions\cuda_dlink_extension.cpp`

```
// 包含 Torch C++ 扩展的头文件
#include <torch/extension.h>

// 声明来自 cuda_dlink_extension.cu 的函数 add_cuda
void add_cuda(const float* a, const float* b, float* output, int size);

// 定义 Torch 扩展中的函数 add
at::Tensor add(at::Tensor a, at::Tensor b) {
  // 检查 a 是否在 CUDA 设备上
  TORCH_CHECK(a.device().is_cuda(), "a is a cuda tensor");
  // 检查 b 是否在 CUDA 设备上
  TORCH_CHECK(b.device().is_cuda(), "b is a cuda tensor");
  // 检查 a 是否是浮点数类型的张量
  TORCH_CHECK(a.dtype() == at::kFloat, "a is a float tensor");
  // 检查 b 是否是浮点数类型的张量
  TORCH_CHECK(b.dtype() == at::kFloat, "b is a float tensor");
  // 检查 a 和 b 是否具有相同的尺寸
  TORCH_CHECK(a.sizes() == b.sizes(), "a and b should have same size");

  // 创建一个和 a 具有相同尺寸和数据类型的空张量作为输出
  at::Tensor output = at::empty_like(a);
  // 调用 CUDA 函数 add_cuda 执行张量 a 和 b 的元素级加法，将结果存入 output
  add_cuda(a.data_ptr<float>(), b.data_ptr<float>(), output.data_ptr<float>(), a.numel());

  // 返回结果张量 output
  return output;
}

// 使用 PYBIND11_MODULE 宏定义 Torch 扩展的入口函数，命名为 TORCH_EXTENSION_NAME
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // 将 add 函数绑定到扩展模块中，命名为 "add"，说明为 "a + b"
  m.def("add", &add, "a + b");
}
```