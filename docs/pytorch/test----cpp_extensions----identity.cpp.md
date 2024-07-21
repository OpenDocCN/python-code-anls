# `.\pytorch\test\cpp_extensions\identity.cpp`

```py
// 引入 PyTorch C++ 扩展的头文件
#include <torch/extension.h>
// 引入 PyTorch 核心功能的头文件
#include <torch/torch.h>

// 使用 torch::autograd 命名空间
using namespace torch::autograd;

// 定义一个继承自 Function<Identity> 的类 Identity，用于实现自定义操作
class Identity : public Function<Identity> {
 public:
  // 实现 forward 方法，接收 AutogradContext 指针和输入张量，返回输入张量本身
  static torch::Tensor forward(AutogradContext* ctx, torch::Tensor input) {
    return input;
  }

  // 实现 backward 方法，接收 AutogradContext 指针和梯度输出列表，返回梯度列表中的第一个元素
  static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs) {
    return {grad_outputs[0]};
  }
};

// 定义一个函数 identity，调用 Identity::apply 方法进行操作
torch::Tensor identity(torch::Tensor input) {
  return Identity::apply(input);
}

// 使用 PYBIND11_MODULE 宏定义 PyTorch C++ 扩展模块
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // 将 identity 函数绑定到扩展模块中，命名为 "identity"，可以在 Python 中调用
  m.def("identity", &identity, "identity");
}
```