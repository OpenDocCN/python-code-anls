# `.\pytorch\torch\csrc\jit\tensorexpr\operators\matmul.cpp`

```py
namespace torch {
namespace jit {
namespace tensorexpr {

// 定义计算矩阵乘法的函数，接受输入参数、输出形状、输出步长、输出类型和设备
Tensor computeMatmul(
    const std::vector<ArgValue>& inputs,              // 输入参数列表
    const std::vector<ExprHandle>& outputShape,       // 输出张量的形状
    const std::vector<ExprHandle>& outputStrides,     // 输出张量的步长
    const std::optional<ScalarType>& outputType,      // 可选的输出类型
    at::Device device) {                              // 输出张量所在设备

  Dtype dtype = kFloat;                              // 默认数据类型为浮点数
  if (outputType) {
    dtype = Dtype(*outputType);                      // 如果有指定输出类型，则使用指定的类型
  }

  BufHandle ResultBuf("matmul", outputShape, dtype);  // 创建结果缓冲区对象

  const BufHandle a = std::get<BufHandle>(inputs[0]); // 获取输入张量 a
  const BufHandle b = std::get<BufHandle>(inputs[1]); // 获取输入张量 b

  auto size_a = a.dims();                            // 获取张量 a 的维度
  auto size_b = b.dims();                            // 获取张量 b 的维度

  // 断言只支持二维矩阵乘法
  TORCH_INTERNAL_ASSERT(size_a.size() == 2 && size_b.size() == 2);

  // 计算总元素数量
  auto total_size =
      to<LongImm>(IRSimplifier::simplify(
                      cast<int64_t>(size_a[0]) * cast<int64_t>(size_a[1]) *
                      cast<int64_t>(size_b[1]))
                      .node());

  // 对于小尺寸的矩阵乘法（N*M*K < 1000），使用简单的三层循环嵌套计算
  // 这有利于消除调度开销
  if (total_size && total_size->value() < 1000) {
    return Reduce(
        "nnc_matmul",
        {size_a[0], size_b[1]},                       // 输出张量的形状
        Sum(),                                        // 求和操作
        [&](const ExprHandle& m, const ExprHandle& n, const ExprHandle& k) {
          return Load::make(a, {m, k}) * Load::make(b, {k, n});  // 加载并计算乘积
        },
        {size_a[1]});                                 // 循环变量范围
  } else {
    // 对于较大的尺寸，生成外部调用来执行矩阵乘法
    return Tensor(
        ResultBuf.node(),
        ExternalCall::make(ResultBuf, "nnc_aten_matmul", {a, b}, {})); // 调用 ATen 的矩阵乘法
  }
}

// 定义计算加权矩阵乘法的函数，接受输入参数、输出形状、输出步长、输出类型和设备
Tensor computeAddMM(
    const std::vector<ArgValue>& inputs,              // 输入参数列表
    const std::vector<ExprHandle>& outputShape,       // 输出张量的形状
    const std::vector<ExprHandle>& outputStrides,     // 输出张量的步长
    const std::optional<ScalarType>& outputType,      // 可选的输出类型
    at::Device device) {                              // 输出张量所在设备

  Dtype dtype = kFloat;                              // 默认数据类型为浮点数
  if (outputType) {
    dtype = Dtype(*outputType);                      // 如果有指定输出类型，则使用指定的类型
  }

  BufHandle ResultBuf("addmm", outputShape, dtype);   // 创建结果缓冲区对象

  // 返回结果张量，调用 ATen 的加权矩阵乘法
  return Tensor(
      ResultBuf.node(),
      ExternalCall::make(
          ResultBuf,
          "nnc_aten_addmm",
          {std::get<BufHandle>(inputs[0]),             // 输入张量 a
           std::get<BufHandle>(inputs[1]),             // 输入张量 b
           std::get<BufHandle>(inputs[2])},            // 输入张量 c
          {std::get<int64_t>(inputs[3]),               // alpha 参数
           std::get<int64_t>(inputs[4])}));            // beta 参数
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
```