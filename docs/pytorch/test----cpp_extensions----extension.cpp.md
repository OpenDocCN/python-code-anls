# `.\pytorch\test\cpp_extensions\extension.cpp`

```
#include <torch/extension.h>

// 包含头文件，以便使用 PyTorch 扩展所需的功能

// test include_dirs in setuptools.setup with relative path
#include <tmp.h>
// 包含一个相对路径的测试头文件 tmp.h

#include <ATen/OpMathType.h>
// 包含 ATen 库中的 OpMathType 头文件

torch::Tensor sigmoid_add(torch::Tensor x, torch::Tensor y) {
  // 定义一个函数 sigmoid_add，接受两个张量参数 x 和 y，返回它们的 sigmoid 函数结果相加后的张量
  return x.sigmoid() + y.sigmoid();
}

struct MatrixMultiplier {
  MatrixMultiplier(int A, int B) {
    // MatrixMultiplier 结构体的构造函数，初始化一个 A x B 大小的张量，数据类型为 float64，开启梯度跟踪
    tensor_ = torch::ones({A, B}, torch::dtype(torch::kFloat64).requires_grad(true));
  }
  torch::Tensor forward(torch::Tensor weights) {
    // 定义 MatrixMultiplier 结构体的 forward 方法，接受一个权重张量参数，返回该张量与内部张量的矩阵乘法结果
    return tensor_.mm(weights);
  }
  torch::Tensor get() const {
    // 定义 MatrixMultiplier 结构体的 get 方法，返回内部张量 tensor_
    return tensor_;
  }

 private:
  torch::Tensor tensor_;
  // MatrixMultiplier 结构体的私有成员变量，存储一个张量
};

bool function_taking_optional(std::optional<torch::Tensor> tensor) {
  // 定义一个函数 function_taking_optional，接受一个可选的 torch::Tensor 参数，返回是否有值的布尔值
  return tensor.has_value();
}

torch::Tensor random_tensor() {
  // 定义一个函数 random_tensor，返回一个随机生成的张量
  return torch::randn({1});
}

at::ScalarType get_math_type(at::ScalarType other) {
  // 定义一个函数 get_math_type，接受一个标量类型 other，返回其对应的 OpMathType
  return at::toOpMathType(other);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // 定义 Python 绑定模块，命名为 TORCH_EXTENSION_NAME，传入模块对象 m

  m.def("sigmoid_add", &sigmoid_add, "sigmoid(x) + sigmoid(y)");
  // 将 sigmoid_add 函数绑定到模块 m 中，命名为 "sigmoid_add"，提供说明文档 "sigmoid(x) + sigmoid(y)"

  m.def(
      "function_taking_optional",
      &function_taking_optional,
      "function_taking_optional");
  // 将 function_taking_optional 函数绑定到模块 m 中，命名为 "function_taking_optional"，提供说明文档 "function_taking_optional"

  py::class_<MatrixMultiplier>(m, "MatrixMultiplier")
      .def(py::init<int, int>())
      .def("forward", &MatrixMultiplier::forward)
      .def("get", &MatrixMultiplier::get);
  // 创建 Python 绑定的 MatrixMultiplier 类型，绑定构造函数和两个成员方法 forward 和 get

  m.def("get_complex", []() { return c10::complex<double>(1.0, 2.0); });
  // 将 lambda 函数绑定到模块 m 中，命名为 "get_complex"，返回一个复数 c10::complex<double>(1.0, 2.0)

  m.def("get_device", []() { return at::device_of(random_tensor()).value(); });
  // 将 lambda 函数绑定到模块 m 中，命名为 "get_device"，返回 random_tensor 的设备信息

  m.def("get_generator", []() { return at::detail::getDefaultCPUGenerator(); });
  // 将 lambda 函数绑定到模块 m 中，命名为 "get_generator"，返回默认的 CPU 生成器

  m.def("get_intarrayref", []() { return at::IntArrayRef({1, 2, 3}); });
  // 将 lambda 函数绑定到模块 m 中，命名为 "get_intarrayref"，返回一个整数数组引用 at::IntArrayRef({1, 2, 3})

  m.def("get_memory_format", []() { return c10::get_contiguous_memory_format(); });
  // 将 lambda 函数绑定到模块 m 中，命名为 "get_memory_format"，返回连续内存格式 c10::get_contiguous_memory_format()

  m.def("get_storage", []() { return random_tensor().storage(); });
  // 将 lambda 函数绑定到模块 m 中，命名为 "get_storage"，返回 random_tensor 的存储对象

  m.def("get_symfloat", []() { return c10::SymFloat(1.0); });
  // 将 lambda 函数绑定到模块 m 中，命名为 "get_symfloat"，返回一个符号浮点数 c10::SymFloat(1.0)

  m.def("get_symint", []() { return c10::SymInt(1); });
  // 将 lambda 函数绑定到模块 m 中，命名为 "get_symint"，返回一个符号整数 c10::SymInt(1)

  m.def("get_symintarrayref", []() { return at::SymIntArrayRef({1, 2, 3}); });
  // 将 lambda 函数绑定到模块 m 中，命名为 "get_symintarrayref"，返回一个符号整数数组引用 at::SymIntArrayRef({1, 2, 3})

  m.def("get_tensor", []() { return random_tensor(); });
  // 将 lambda 函数绑定到模块 m 中，命名为 "get_tensor"，返回一个随机生成的张量

  m.def("get_math_type", &get_math_type);
  // 将 get_math_type 函数绑定到模块 m 中，命名为 "get_math_type"
}
```