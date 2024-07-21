# `.\pytorch\torch\csrc\jit\passes\onnx\constant_fold.h`

```
#pragma once


// 指令：#pragma once
// 作用：确保头文件只被编译一次，避免多重包含问题

#include <memory>


// 包含：memory 头文件
// 作用：定义了智能指针等内存管理工具

#include <c10/util/Optional.h>


// 包含：c10/util/Optional.h 头文件
// 作用：定义了可选类型（Optional）和相关的实用功能

#include <torch/csrc/jit/ir/ir.h>


// 包含：torch/csrc/jit/ir/ir.h 头文件
// 作用：包含了 JIT 中间表示（IR）的定义和相关功能

namespace torch {
namespace jit {


// 命名空间：torch::jit
// 作用：定义了 TorchScript 的 JIT 编译器的相关功能和类

const int ONNX_OPSET_9 = 9;
const int ONNX_OPSET_10 = 10;
const int ONNX_OPSET_11 = 11;
const int ONNX_OPSET_12 = 12;
const int ONNX_OPSET_13 = 13;
const int ONNX_OPSET_14 = 14;


// 常量定义：ONNX_OPSET_9 到 ONNX_OPSET_14
// 作用：定义了不同版本的 ONNX 运算集（Operator Set）的版本号常量

namespace onnx_constant_fold {


// 命名空间：torch::jit::onnx_constant_fold
// 作用：定义了 ONNX 运算常量折叠的相关功能和类

at::Tensor IntToTensor(int64_t value);


// 函数声明：IntToTensor
// 参数：int64_t value
// 返回类型：at::Tensor
// 作用：将整数值转换为 PyTorch 的 Tensor 对象

std::optional<at::Tensor> runTorchBackendForOnnx(
    const Node* node,
    std::vector<at::Tensor>& inputTensorValues,
    int opset_version);


// 函数声明：runTorchBackendForOnnx
// 参数：
//   - const Node* node：表示节点的指针
//   - std::vector<at::Tensor>& inputTensorValues：输入 Tensor 的引用向量
//   - int opset_version：ONNX 运算集的版本号
// 返回类型：std::optional<at::Tensor>
// 作用：运行 Torch 后端处理 ONNX 节点，返回计算结果的可选 Tensor 对象

} // namespace onnx_constant_fold


// 结束命名空间：torch::jit::onnx_constant_fold
// 作用：命名空间 onnx_constant_fold 的结束标记

void ConstantFoldONNX(
    std::shared_ptr<Graph>& g,
    std::map<std::string, IValue>& paramDict,
    int opset_version);


// 函数声明：ConstantFoldONNX
// 参数：
//   - std::shared_ptr<Graph>& g：指向图（Graph）的共享指针
//   - std::map<std::string, IValue>& paramDict：参数字典，映射参数名到 IValue
//   - int opset_version：ONNX 运算集的版本号
// 返回类型：void
// 作用：对 ONNX 表示的图进行常量折叠优化

} // namespace jit


// 结束命名空间：torch::jit
// 作用：命名空间 jit 的结束标记

} // namespace torch


// 结束命名空间：torch
// 作用：命名空间 torch 的结束标记
```