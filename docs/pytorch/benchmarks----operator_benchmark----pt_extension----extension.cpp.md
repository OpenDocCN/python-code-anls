# `.\pytorch\benchmarks\operator_benchmark\pt_extension\extension.cpp`

```py
// 引入 Torch 库中的扩展和脚本模块
#include <torch/extension.h>
#include <torch/script.h>

// 使用 torch 命名空间中的 List 和 Tensor 类型
using torch::List;
using torch::Tensor;

// 接受一个 Tensor 并返回它，函数 consume 的定义
Tensor consume(Tensor a) {
  return a;
}

// 接受一个 Tensor 列表并返回它，函数 consume_list 的定义
List<Tensor> consume_list(List<Tensor> a) {
  return a;
}

// 当使用 JIT 追踪具有常量循环的函数时，
// 由于死代码消除的作用，for 循环会被优化掉。
// 这对于我们的运算基准 (benchmark) 存在问题，因为我们需要在循环中运行一个操作
// 并报告执行时间。此差异通过将此 consume 操作注册为正确的别名信息（DEFAULT）来解决该问题。
TORCH_LIBRARY_FRAGMENT(operator_benchmark, m) {
  // 在 operator_benchmark 库中注册 _consume 函数，指向 consume 函数
  m.def("_consume", &consume);
  // 在 operator_benchmark 库中注册 _consume.list 函数，指向 consume_list 函数
  m.def("_consume.list", &consume_list);
}

// PYBIND11_MODULE 宏定义了 Python 扩展模块的入口点
PYBIND11_MODULE(benchmark_cpp_extension, m) {
  // 将 consume 函数暴露给 Python，命名为 "consume"
  m.def("_consume", &consume, "consume");
  // 将 consume_list 函数暴露给 Python，命名为 "consume_list"
  m.def("_consume_list", &consume_list, "consume_list");
}
```