# `.\pytorch\torch\csrc\jit\tensorexpr\operators\softmax.cpp`

```
// 引入 Torch 的头文件 softmax.h，其中包含了对 softmax 操作符的定义和实现
#include <torch/csrc/jit/tensorexpr/operators/softmax.h>

// Torch 的命名空间，包含了与 JIT 相关的功能
namespace torch {
namespace jit {
namespace tensorexpr {

// 使用 torch::jit::tensorexpr 命名空间的简便方式
using namespace torch::jit::tensorexpr;

// 定义一个函数 computeSoftmax，计算 softmax 或 log_softmax
Tensor computeSoftmax(
    const std::vector<ArgValue>& inputs, // 输入参数列表
    const std::vector<ExprHandle>& outputShape, // 输出张量形状的表达式列表
    const std::vector<ExprHandle>& outputStrides, // 输出张量的步长
    bool log_softmax) { // 是否进行 log_softmax 的标志

  // Softmax 的计算方法如下：
  //    softmax(vi) = exp(vi) / sum(exp(vi))
  //
  // 为了避免由于指数函数作用于大数而导致的溢出问题，我们会在计算 exp 前减去该维度上的最大值。
  //    softmax(vi) = exp(vi - max(vi)) / sum(exp(vi - max(vi)))
  //
  // 这个函数实现了 4 个嵌套循环：
  //   - 第一个循环计算 softmax 维度上的最大值。
  //   - 第二个循环计算减去 softmax 维度上的最大值后的每个元素的 exp。
  //   - 第三个循环计算 softmax 维度上的总和。
  //   - 最后一个循环计算每个元素的 softmax 值。

  // 对于 log_softmax，计算方法如下：
  //    log_softmax(vi) = log(softmax(vi))
  //                    = vi - log(sum(exp(vi)))
  //
  // 使用与上述相同的最大值技巧：
  //    log_softmax(vi) = vi - max(vi) - log(sum(exp(vi - max(vi))))
  //
  // 这个函数实现了 5 个嵌套循环：
  //   - 第一个循环计算 softmax 维度上的最大值。
  //   - 第二个循环计算减去 softmax 维度上的最大值后的每个元素的 exp。
  //   - 第三个循环计算 softmax 维度上的总和。
  //   - 第四个循环计算总和中每个元素的对数。
  //   - 最后一个循环计算每个元素的 log_softmax 值。

  // 内部断言，确保输入参数的数量为 3
  TORCH_INTERNAL_ASSERT(inputs.size() == 3);

  // 不处理维度参数（输入参数 1）为 None 的情况，因为这被认为是不推荐的用法
  TORCH_INTERNAL_ASSERT(std::get_if<int64_t>(&inputs[1]));

  // 计算输入张量的秩
  int64_t rank = valueShape(inputs[0]).size();

  // 规范化并检查 softmax 维度的索引，确保其有效
  size_t softmax_dim =
      normalizeAndCheckIndex(std::get<int64_t>(inputs[1]), rank);

  // 创建一个存储非 softmax 维度形状的表达式列表
  std::vector<ExprHandle> non_softmax_dims;
  for (size_t i = 0; i < outputShape.size(); ++i) {
    if (i != softmax_dim) {
      non_softmax_dims.push_back(outputShape[i]);
    }
  }

  // Softmax 的实现包括两个约简操作，一个用于找到最大值，另一个用于计算沿 softmax 维度的总和。
  // 这些约简操作将 softmax 维度作为最内层循环。因此，索引中的最内层索引将引用 softmax 维度。

  // 更新索引，将 softmax 维度的索引移动到适当的位置
  auto move_softmax_dim_index_to_pos = [&](const ParameterList& indices) {
    std::vector<ExprHandle> new_indices;
    for (const auto& ind : indices) {
      new_indices.push_back(ind);
    }
    for (size_t i = softmax_dim; i < indices.size() - 1; ++i) {
      new_indices[i + 1] = indices[i];
    }
    new_indices[softmax_dim] = indices[indices.size() - 1];
    return new_indices;
  };

  // 移除与 softmax 维度对应的索引。
  auto remove_softmax_dim_index = [&](const ParameterList& indices) {
    std::vector<ExprHandle> new_indices;
    for (size_t i = 0; i < indices.size(); ++i) {
      if (i != softmax_dim) {
        new_indices.push_back(indices[i]);
      }
    }
    return new_indices;
  };

  // 将参数列表中的索引转换为表达式句柄。
  auto convert_indices_to_expr_handle = [&](const ParameterList& indices) {
    std::vector<ExprHandle> new_indices(indices.size());
    for (size_t i = 0; i < indices.size(); ++i) {
      new_indices[i] = indices[i];
    }
    return new_indices;
  };

  // 从输入中获取缓冲区。
  auto inp_buf = std::get<BufHandle>(inputs[0]);

  // 确定输入的数据类型。
  auto dtype = inp_buf.dtype();
  if (auto d = std::get_if<int64_t>(&inputs[2])) {
    dtype = ToDtype(static_cast<ScalarType>(*d));
  }

  // 计算最大值。
  auto max = Reduce(
      "aten_softmax_max",
      non_softmax_dims,
      c10::nullopt,
      Maximum(dtype),
      [&](ParameterList& indices) {
        return tensorOrConstant(
            inputs[0], move_softmax_dim_index_to_pos(indices));
      },
      {outputShape[softmax_dim]});
  
  // 计算指数。
  auto e = Compute(
      "aten_softmax_exp",
      outputShape,
      c10::nullopt,
      [&](ParameterList& indices) {
        auto inp = tensorOrConstant(
            inputs[0], convert_indices_to_expr_handle(indices));
        return exp(inp - max.load(remove_softmax_dim_index(indices)));
      });
  
  // 计算求和。
  auto sum = Reduce(
      "aten_softmax_sum",
      non_softmax_dims,
      c10::nullopt,
      Sum(),
      [&](ParameterList& indices) {
        return e.load(move_softmax_dim_index_to_pos(indices));
      },
      {outputShape[softmax_dim]});
  
  // 如果不是 log_softmax，则计算 softmax。
  if (!log_softmax) {
    auto result = Compute(
        "aten_softmax", outputShape, c10::nullopt, [&](ParameterList& indices) {
          return e.load(indices) / sum.load(remove_softmax_dim_index(indices));
        });
    return Tensor(
        result.buf(),
        alloc<tensorexpr::Block>(std::vector<StmtPtr>(
            {max.stmt(), e.stmt(), sum.stmt(), result.stmt()})));
  }

  // 计算 log_sum。
  auto log_sum = Compute(
      "aten_softmax_log_sum",
      non_softmax_dims,
      c10::nullopt,
      [&](ParameterList& indices) { return log(sum.load(indices)); });
  
  // 计算 log_softmax。
  auto result = Compute(
      "aten_log_softmax",
      outputShape,
      c10::nullopt,
      [&](ParameterList& indices) {
        auto inp = tensorOrConstant(
            inputs[0], convert_indices_to_expr_handle(indices));
        auto non_softmax_indices = remove_softmax_dim_index(indices);
        return inp - max.load(non_softmax_indices) -
            log_sum.load(non_softmax_indices);
      });
  
  // 返回最终的张量结果。
  return Tensor(
      result.buf(),
      alloc<tensorexpr::Block>(std::vector<StmtPtr>(
          {max.stmt(), e.stmt(), sum.stmt(), log_sum.stmt(), result.stmt()})));
}
} // namespace tensorexpr
} // namespace jit
} // namespace torch


注释：


// 结束了命名空间 torch
} // namespace tensorexpr
// 结束了命名空间 jit
} // namespace jit
// 结束了命名空间 torch
} // namespace torch


这段代码是 C++ 中的命名空间结束声明，用来关闭之前定义的命名空间区域。在这里，依次结束了命名空间 `torch`、`jit` 和 `tensorexpr`。这种方式有助于在大型代码库中组织和隔离不同模块或组件的代码，避免命名冲突并提高代码的可维护性和可读性。
```