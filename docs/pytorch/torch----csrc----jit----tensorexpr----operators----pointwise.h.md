# `.\pytorch\torch\csrc\jit\tensorexpr\operators\pointwise.h`

```py
#pragma once
// 引入 Torch TensorExpr 框架的核心头文件

#include <torch/csrc/jit/tensorexpr/kernel.h>

namespace torch {
namespace jit {
namespace tensorexpr {

// 定义了一个公共的 Tensor 计算函数 computeSign，用于计算签名
TORCH_API Tensor computeSign(
    const std::vector<ArgValue>& inputs, // 输入参数列表
    const std::vector<ExprHandle>& outputShape, // 输出张量的形状描述
    std::optional<std::vector<ExprHandle>> outputStrides = c10::nullopt); // 可选参数，输出张量的步长

// 定义了一个 Tensor 计算函数 computeOneOperand，用于单操作数计算
Tensor computeOneOperand(
    const std::string& name, // 计算函数的名称
    const std::vector<ArgValue>& inputValues, // 输入参数列表
    const std::vector<ExprHandle>& outputShape, // 输出张量的形状描述
    const std::vector<ExprHandle>& outputStrides, // 输出张量的步长描述
    const std::optional<ScalarType>& outputType, // 可选参数，输出张量的数据类型
    const std::function<ExprHandle(const ExprHandle&)>& innerExpr, // 内部表达式的函数对象
    const int checkParamTypes = kAllTypes); // 检查参数类型的选项

// 定义了一个 Tensor 计算函数 computeTwoOperand，用于双操作数计算
Tensor computeTwoOperand(
    const std::string& name, // 计算函数的名称
    const std::vector<ArgValue>& inputValues, // 输入参数列表
    const std::vector<ExprHandle>& outputShape, // 输出张量的形状描述
    const std::vector<ExprHandle>& outputStrides, // 输出张量的步长描述
    const std::optional<ScalarType>& outputType, // 可选参数，输出张量的数据类型
    const std::function<ExprHandle(const ExprHandle&, const ExprHandle&)>&
        innerExpr); // 内部表达式的函数对象

// 定义了一个 Tensor 计算函数 computeTwoOperandWithAlpha，用于带 alpha 参数的双操作数计算
Tensor computeTwoOperandWithAlpha(
    const std::string& name, // 计算函数的名称
    const std::vector<ArgValue>& inputValues, // 输入参数列表
    const std::vector<ExprHandle>& outputShape, // 输出张量的形状描述
    const std::vector<ExprHandle>& outputStrides, // 输出张量的步长描述
    const std::optional<ScalarType>& outputType, // 可选参数，输出张量的数据类型
    const std::function<ExprHandle(const ExprHandle&, const ExprHandle&)>&
        innerExpr); // 内部表达式的函数对象

// 定义了一个 Tensor 计算函数 computeConditionWithTwoOperand，用于带条件的双操作数计算
Tensor computeConditionWithTwoOperand(
    const std::string& name, // 计算函数的名称
    const std::vector<ArgValue>& inputValues, // 输入参数列表
    const std::vector<ExprHandle>& outputShape, // 输出张量的形状描述
    const std::vector<ExprHandle>& outputStrides, // 输出张量的步长描述
    const std::optional<ScalarType>& outputType, // 可选参数，输出张量的数据类型
    const std::function<
        ExprHandle(const ExprHandle&, const ExprHandle&, const ExprHandle&)>&
        innerExpr); // 内部表达式的函数对象

// 定义了一个 Tensor 计算函数 computeThreeOperand，用于三操作数计算
Tensor computeThreeOperand(
    const std::string& name, // 计算函数的名称
    const std::vector<ArgValue>& inputValues, // 输入参数列表
    const std::vector<ExprHandle>& outputShape, // 输出张量的形状描述
    const std::vector<ExprHandle>& outputStrides, // 输出张量的步长描述
    const std::optional<ScalarType>& outputType, // 可选参数，输出张量的数据类型
    const std::function<
        ExprHandle(const ExprHandle&, const ExprHandle&, const ExprHandle&)>&
        innerExpr, // 内部表达式的函数对象
    bool promote_inputs = true); // 是否推广输入参数的标志

// 定义了一个 Tensor 计算函数 computeFourOperand，用于四操作数计算
Tensor computeFourOperand(
    const std::string& name, // 计算函数的名称
    const std::vector<ArgValue>& inputValues, // 输入参数列表
    const std::vector<ExprHandle>& outputShape, // 输出张量的形状描述
    const std::vector<ExprHandle>& outputStrides, // 输出张量的步长描述
    const std::optional<ScalarType>& outputType, // 可选参数，输出张量的数据类型
    const std::function<ExprHandle(
        const ExprHandle&,
        const ExprHandle&,
        const ExprHandle&,
        const ExprHandle&)>& innerExpr); // 内部表达式的函数对象

// 定义了一个 Tensor 计算函数 computeNoop，用于无操作的计算
Tensor computeNoop(
    const std::vector<ArgValue>& inputs, // 输入参数列表
    const std::vector<ExprHandle>& outputShape, // 输出张量的形状描述
    const std::vector<ExprHandle>& outputStrides, // 输出张量的步长描述
    const std::optional<ScalarType>& outputType, // 可选参数，输出张量的数据类型
    at::Device device); // 计算设备对象

// 定义了一个 Tensor 计算函数 computeScalar，用于标量计算
Tensor computeScalar(
    const std::string& name, // 计算函数的名称
    const std::vector<ArgValue>& inputValues, // 输入参数列表
    const std::vector<ExprHandle>& outputShape, // 输出张量的形状描述
    const std::vector<ExprHandle>& outputStrides, // 输出张量的步长描述
    // const std::optional<ScalarType>& outputType,
    // 定义了一个名为 outputType 的常量引用，类型为 std::optional<ScalarType>
    // std::optional 是一个 C++17 中引入的模板类，表示一个可选的值，可以包含或不包含值
    // ScalarType 是一个模板参数，表示具体的数据类型，例如 int、float 等

    // const std::function<ExprHandle(const ExprHandle&, const ExprHandle&)>&
    // 定义了一个名为 innerExpr 的常量引用，类型为 std::function<ExprHandle(const ExprHandle&, const ExprHandle&)>
    // std::function 是一个可调用对象的封装器，可以包含函数指针、函数对象、Lambda 表达式等
    // 它定义了一个接受两个 ExprHandle 参数并返回 ExprHandle 的函数签名
} // 结束 torch 命名空间
} // 结束 jit 命名空间
} // 结束 tensorexpr 命名空间
```