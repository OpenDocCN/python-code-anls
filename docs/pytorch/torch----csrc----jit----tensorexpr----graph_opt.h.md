# `.\pytorch\torch\csrc\jit\tensorexpr\graph_opt.h`

```py
#pragma once
// 剪裁给定的图形，使其只保留前 iters 个迭代。返回剪裁后的图形的共享指针。
TORCH_API std::shared_ptr<Graph> trimGraph(
    const std::shared_ptr<Graph>& graph,
    int64_t iters);

// 扫描给定图形中的所有值，并用在 SIZES 中出现的每个维度大小 Xi 替换为符号形状 Yi。返回一个符号值的向量 [Y0, Y1, .., Yn]。
//
// 例如：
// 输入：
// graph(%x : Float(10, 20, 30, 40)):
//   %y : Float(10, 20, 30, 40) = aten::relu(%x)
//   return %y
//
// 如果我们运行 makeShapesSymbolic(graph, {20, 40})，那么我们将得到：
//
// graph(%x : Float(10, SS(-3), 30, SS(-5))):
//   %y : Float(10, SS(-3), 30, SS(-5)) = aten::relu(%x)
//   return %y
//
// 并且返回 {-3, -5} 作为返回值。
TORCH_API std::vector<int64_t> makeShapesSymbolic(
    std::shared_ptr<Graph>& graph,
    const std::vector<int64_t>& sizes);

// 检查图形并报告是否可以转换为 TE IR（Tensor Expression 中间表示）。
// TODO: 对于不能转换的图形添加错误报告。
TORCH_API bool isGraphCompilable(const std::shared_ptr<Graph>& graph);

// 检查图形并（hack地）填充缺失的张量类型信息，如标量类型、设备和步幅。理想情况下，这应该由正确的数据类型/设备/形状传播通道完成，
// 但在它们准备好之前，我们可以使用这个并不总是正确的临时解决方案。
TORCH_API void fixupMissingShapeInfo(const std::shared_ptr<Graph>& graph);

} // namespace tensorexpr
} // namespace jit
} // namespace torch
```