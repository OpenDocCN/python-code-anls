# `.\pytorch\torch\csrc\api\src\nn\modules\container\functional.cpp`

```py
#include <torch/nn/modules/container/functional.h>  // 引入torch库中的functional.h头文件

#include <torch/types.h>  // 引入torch库中的types.h头文件

#include <functional>  // 引入C++标准库中的functional头文件，用于std::function等功能
#include <utility>  // 引入C++标准库中的utility头文件，用于std::move等功能

namespace torch {
namespace nn {

FunctionalImpl::FunctionalImpl(Function function)
    : function_(std::move(function)) {}  // FunctionalImpl类的构造函数实现，初始化function_成员变量

void FunctionalImpl::reset() {}  // reset函数的定义，空实现，无功能

void FunctionalImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::Functional()";  // pretty_print函数的实现，向输出流中打印固定字符串
}

Tensor FunctionalImpl::forward(Tensor input) {
  return function_(std::move(input));  // forward函数的实现，调用function_成员变量，传递input参数，并返回结果Tensor
}

Tensor FunctionalImpl::operator()(Tensor input) {
  return forward(std::move(input));  // 重载函数调用操作符()，调用forward函数处理输入Tensor并返回结果Tensor
}

bool FunctionalImpl::is_serializable() const {
  return false;  // is_serializable函数的实现，始终返回false，表明对象不支持序列化
}

} // namespace nn
} // namespace torch
```