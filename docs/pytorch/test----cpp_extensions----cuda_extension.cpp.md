# `.\pytorch\test\cpp_extensions\cuda_extension.cpp`

```
// 包含 Torch 的 C++ 扩展头文件
#include <torch/extension.h>

// 声明来自 cuda_extension.cu 的函数。它将使用 nvcc 单独编译，并与 cuda_extension.cpp 的对象文件链接为一个共享库。
void sigmoid_add_cuda(const float* x, const float* y, float* output, int size);

// 定义 Torch 的张量函数 sigmoid_add，接受两个 CUDA 张量 x 和 y 作为输入
torch::Tensor sigmoid_add(torch::Tensor x, torch::Tensor y) {
  // 检查 x 是否为 CUDA 张量
  TORCH_CHECK(x.device().is_cuda(), "x must be a CUDA tensor");
  // 检查 y 是否为 CUDA 张量
  TORCH_CHECK(y.device().is_cuda(), "y must be a CUDA tensor");

  // 创建一个与 x 相同大小的全零张量 output
  auto output = torch::zeros_like(x);

  // 调用 sigmoid_add_cuda 函数，计算 x 和 y 的元素级 sigmoid 相加，结果存储在 output 中
  sigmoid_add_cuda(
      x.data_ptr<float>(), y.data_ptr<float>(), output.data_ptr<float>(), output.numel());

  // 返回计算结果张量 output
  return output;
}

// 使用 PyBind11 定义 Python 模块 TORCH_EXTENSION_NAME，并将 sigmoid_add 函数绑定到模块中
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sigmoid_add", &sigmoid_add, "sigmoid(x) + sigmoid(y)");
}
```