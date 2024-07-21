# `.\pytorch\test\cpp_extensions\cpp_frontend_extension.cpp`

```
// 引入 Torch C++ 扩展库
#include <torch/extension.h>

// 引入标准库头文件
#include <cstddef>
#include <string>

// 定义网络结构 Net，继承自 Cloneable 接口
struct Net : torch::nn::Cloneable<Net> {
  // 构造函数，初始化输入和输出大小
  Net(int64_t in, int64_t out) : in_(in), out_(out) {
    // 调用 reset 方法进行初始化
    reset();
  }

  // 重写 reset 方法
  void reset() override {
    // 注册全连接层模块 fc，输入维度为 in_，输出维度为 out_
    fc = register_module("fc", torch::nn::Linear(in_, out_));
    // 注册缓冲区 buffer，初始化为 5x5 的单位矩阵
    buffer = register_buffer("buf", torch::eye(5));
  }

  // 前向传播方法，接收输入张量 x，返回经过全连接层处理后的张量
  torch::Tensor forward(torch::Tensor x) {
    return fc->forward(x);
  }

  // 设置偏置方法，接收偏置张量 bias，使用 NoGradGuard 禁止梯度计算
  void set_bias(torch::Tensor bias) {
    torch::NoGradGuard guard;
    fc->bias.set_(bias);
  }

  // 获取偏置方法，返回当前偏置张量
  torch::Tensor get_bias() const {
    return fc->bias;
  }

  // 添加新参数的方法，接收参数名称 name 和参数张量 tensor，注册为网络参数
  void add_new_parameter(const std::string& name, torch::Tensor tensor) {
    register_parameter(name, tensor);
  }

  // 添加新缓冲区的方法，接收缓冲区名称 name 和缓冲区张量 tensor，注册为网络缓冲区
  void add_new_buffer(const std::string& name, torch::Tensor tensor) {
    register_buffer(name, tensor);
  }

  // 添加新子模块的方法，接收子模块名称 name，使用当前全连接层选项注册一个新的线性子模块
  void add_new_submodule(const std::string& name) {
    register_module(name, torch::nn::Linear(fc->options));
  }

  // 输入和输出大小
  int64_t in_, out_;
  // 全连接层模块指针
  torch::nn::Linear fc{nullptr};
  // 缓冲区张量
  torch::Tensor buffer;
};

// 定义 Python 绑定模块的函数 TORCH_EXTENSION_NAME 是由 PyTorch 提供的扩展名称
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // 绑定 Net 结构体为 Python 模块 Net
  torch::python::bind_module<Net>(m, "Net")
      // 绑定构造函数
      .def(py::init<int64_t, int64_t>())
      // 绑定 set_bias 方法
      .def("set_bias", &Net::set_bias)
      // 绑定 get_bias 方法
      .def("get_bias", &Net::get_bias)
      // 绑定 add_new_parameter 方法
      .def("add_new_parameter", &Net::add_new_parameter)
      // 绑定 add_new_buffer 方法
      .def("add_new_buffer", &Net::add_new_buffer)
      // 绑定 add_new_submodule 方法
      .def("add_new_submodule", &Net::add_new_submodule);
}


这段代码定义了一个 `Net` 结构体，实现了一个简单的神经网络模型，包括全连接层和一些辅助方法，然后使用 PyTorch 的 C++ 扩展库将其绑定为 Python 模块。
```