# `.\pytorch\test\cpp_extensions\dangling_impl_extension.cpp`

```py
# 包含 Torch 扩展的头文件
#include <torch/extension.h>

# 定义一个空的函数 foo
void foo() { }

# 实现 Torch 库的扩展模块 __test，针对 CPU
TORCH_LIBRARY_IMPL(__test, CPU, m) {
  # 将函数 foo 的实现注册到模块 m 的 "foo" 接口
  m.impl("foo", foo);
}

# 使用 Pybind11 定义 Torch 扩展模块的入口点，名称为 TORCH_EXTENSION_NAME
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  # 将函数 foo 绑定为模块 m 的 "bar" 函数
  m.def("bar", foo);
}
```