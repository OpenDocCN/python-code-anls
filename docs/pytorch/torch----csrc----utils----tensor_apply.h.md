# `.\pytorch\torch\csrc\utils\tensor_apply.h`

```
#pragma once


// 指令：确保此头文件只被编译一次包含，避免重复定义错误


#include <ATen/core/Tensor.h>
#include <torch/csrc/python_headers.h>


// 包含 ATen 张量核心功能和 Torch Python 头文件


namespace torch::utils {

const at::Tensor& apply_(const at::Tensor& self, PyObject* fn);


// 命名空间：torch::utils，包含以下函数声明：

// 函数 apply_
// 参数：
//   - self: 类型为 at::Tensor 的引用，表示操作的对象张量
//   - fn: 指向 PyObject 的指针，表示应用于张量的 Python 函数对象
// 返回值：
//   - const at::Tensor&: 返回一个常量引用，表示应用函数后的张量


const at::Tensor& map_(
    const at::Tensor& self,
    const at::Tensor& other_,
    PyObject* fn);


// 函数 map_
// 参数：
//   - self: 类型为 at::Tensor 的引用，表示操作的对象张量
//   - other_: 类型为 at::Tensor 的常量引用，表示第二个输入张量
//   - fn: 指向 PyObject 的指针，表示应用于张量的 Python 函数对象
// 返回值：
//   - const at::Tensor&: 返回一个常量引用，表示映射函数后的张量


const at::Tensor& map2_(
    const at::Tensor& self,
    const at::Tensor& x_,
    const at::Tensor& y_,
    PyObject* fn);


// 函数 map2_
// 参数：
//   - self: 类型为 at::Tensor 的引用，表示操作的对象张量
//   - x_: 类型为 at::Tensor 的常量引用，表示第一个输入张量
//   - y_: 类型为 at::Tensor 的常量引用，表示第二个输入张量
//   - fn: 指向 PyObject 的指针，表示应用于张量的 Python 函数对象
// 返回值：
//   - const at::Tensor&: 返回一个常量引用，表示二元映射函数后的张量


} // namespace torch::utils


// 命名空间结束：torch::utils
```