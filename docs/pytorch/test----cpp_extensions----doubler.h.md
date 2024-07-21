# `.\pytorch\test\cpp_extensions\doubler.h`

```py
#include <torch/extension.h>

// 定义一个结构体 Doubler
struct Doubler {
  // Doubler 的构造函数，接受两个整数参数 A 和 B
  Doubler(int A, int B) {
    // 初始化 tensor_，创建一个 A x B 大小的张量，元素类型为 float64，且开启梯度跟踪
    tensor_ = torch::ones({A, B}, torch::dtype(torch::kFloat64).requires_grad(true));
  }

  // 前向传播函数 forward
  torch::Tensor forward() {
    // 返回当前张量 tensor_ 的每个元素乘以 2 后的结果张量
    return tensor_ * 2;
  }

  // 获取当前张量 tensor_ 的函数 get，返回不可修改的张量
  torch::Tensor get() const {
    return tensor_;
  }

 private:
  // 私有成员变量，存储 Doubler 内部的张量
  torch::Tensor tensor_;
};
```