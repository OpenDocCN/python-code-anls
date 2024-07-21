# `.\pytorch\test\cpp_extensions\jit_extension.cpp`

```
#include <torch/extension.h> // 引入 PyTorch C++ 扩展的头文件

#include "doubler.h" // 引入自定义的 doubler.h 头文件

using namespace at; // 使用 PyTorch 的命名空间

Tensor exp_add(Tensor x, Tensor y); // 声明一个函数 exp_add，用于计算两个张量的指数和

Tensor tanh_add(Tensor x, Tensor y) { // 定义函数 tanh_add，接收两个张量参数，返回它们的双曲正切之和
  return x.tanh() + y.tanh(); // 返回 x 和 y 的双曲正切的张量之和
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { // 定义 PyTorch 扩展模块的入口函数
  m.def("tanh_add", &tanh_add, "tanh(x) + tanh(y)"); // 将 tanh_add 函数绑定为模块的方法，说明文档为 "tanh(x) + tanh(y)"
  m.def("exp_add", &exp_add, "exp(x) + exp(y)"); // 将 exp_add 函数绑定为模块的方法，说明文档为 "exp(x) + exp(y)"
  py::class_<Doubler>(m, "Doubler") // 定义一个 Python 可调用的类 Doubler
    .def(py::init<int, int>()) // 定义 Doubler 类的构造函数，接收两个整数参数
    .def("forward", &Doubler::forward) // 定义 Doubler 类的 forward 方法，绑定到 C++ 中的 forward 方法
    .def("get", &Doubler::get); // 定义 Doubler 类的 get 方法，绑定到 C++ 中的 get 方法
}
```