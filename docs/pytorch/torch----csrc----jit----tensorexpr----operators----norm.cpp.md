# `.\pytorch\torch\csrc\jit\tensorexpr\operators\norm.cpp`

```py
// 引入 Torch 的 TensorExpr 库中定义的运算符和函数
#include <torch/csrc/jit/tensorexpr/operators/misc.h>
#include <torch/csrc/jit/tensorexpr/operators/norm.h>

// Torch 的命名空间
namespace torch {
namespace jit {
namespace tensorexpr {

// 定义计算批归一化的函数
Tensor computeBatchNorm(
    const std::vector<ArgValue>& inputs,        // 输入参数列表
    const std::vector<ExprHandle>& outputShape, // 输出张量的形状描述
    const std::vector<ExprHandle>& outputStrides, // 输出张量的步幅描述
    const std::optional<ScalarType>& outputType,  // 输出张量的数据类型（可选）
    at::Device device) {                         // 设备类型

  // 是否有权重和偏置的标志，默认为 true
  bool hasWeight = true;
  bool hasBias = true;

  // 检查第二个输入参数是否为空
  if (std::holds_alternative<ArgNone>(inputs[1])) {
    hasWeight = false;  // 如果为空，则没有权重
  }

  // 检查第三个输入参数是否为空
  if (std::holds_alternative<ArgNone>(inputs[2])) {
    hasBias = false;    // 如果为空，则没有偏置
  }

  // 构建并返回计算表达式
  return Compute(
      "aten_batch_norm",                   // 计算的名称
      outputShape,                         // 输出张量的形状
      outputStrides,
      [&](const std::vector<VarHandle>& axes) { // 匿名函数，计算具体表达式
        TORCH_INTERNAL_ASSERT(axes.size() >= 2); // 断言，确保轴的数量不少于2
        // axes: N, C, H, W，轴向描述

        // 将轴向转换为表达式
        std::vector<ExprHandle> indices(axes.begin(), axes.end());
        ExprHandle c = indices[1];  // 获取通道数的表达式

        // 参数列表：
        // input, weight, bias, mean, var, training, momentum, eps,
        // cudnn_enabled
        std::vector<ExprHandle> exprInputs = {
            tensorOrConstant(inputs[0], indices),  // 输入张量或常数
            tensorOrConstant(inputs[3], {c}),      // 均值
            tensorOrConstant(inputs[4], {c}),      // 方差
            constant(inputs[7])                   // epsilon（极小值）
        };

        // 默认权重和偏置的初始表达式
        ExprHandle weight = FloatImm::make(1);
        ExprHandle bias = FloatImm::make(0);

        // 如果有权重，则获取权重的表达式，并添加到参数列表中
        if (hasWeight) {
          weight = tensorOrConstant(inputs[1], {c});
          exprInputs.push_back(weight);
        }

        // 如果有偏置，则获取偏置的表达式，并添加到参数列表中
        if (hasBias) {
          bias = tensorOrConstant(inputs[2], {c});
          exprInputs.push_back(bias);
        }

        // 推广输入表达式
        promoteInputs(exprInputs);

        // 获取输入、均值、方差和 epsilon 的表达式
        ExprHandle input = exprInputs[0];
        ExprHandle mean = exprInputs[1];
        ExprHandle var = exprInputs[2];
        ExprHandle eps = exprInputs[3];

        // 计算倒数平方根
        auto inv_var = rsqrt(var + eps);
        // 计算 alpha 和 beta
        auto alpha = inv_var * weight;
        auto beta = bias - mean * alpha;
        // 计算输出
        auto output = input * alpha + beta;
        // 降级输出到指定的输出类型
        return demoteOutput(output, outputType);
      });
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
```