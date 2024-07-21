# `.\pytorch\torch\csrc\jit\backends\xnnpack\xnnpack_backend_preprocess.cpp`

```
// 包含 Torch 库的头文件，用于 JIT 后端和预处理
#include <torch/csrc/jit/backends/backend.h>
#include <torch/csrc/jit/backends/backend_preprocess.h>

// 包含 TensorExpr 库的图优化功能
#include <torch/csrc/jit/tensorexpr/graph_opt.h>
// 包含 PyTorch 核心库
#include <torch/torch.h>
// 包含 XNNPACK 头文件
#include <xnnpack.h>

// 包含 ATen 核心库的 List 头文件
#include <ATen/core/List.h>
// 包含 XNNPACK 后端的图构建器
#include <torch/csrc/jit/backends/xnnpack/xnnpack_graph_builder.h>

// Torch 命名空间
namespace torch {
// JIT 命名空间
namespace jit {
// XNNPACK 后端命名空间
namespace xnnpack {
// 代理命名空间
namespace delegate {

// 函数 preprocess 对输入模块进行预处理，以准备委托给 XNNPACK 后端
c10::IValue preprocess(
    const Module& mod,
    const c10::Dict<c10::IValue, c10::IValue>& method_compile_spec,
    const BackendDebugHandleGenerator& generate_debug_handles) {
  // 克隆输入的模块，并设置为评估模式
  auto eval_mod = mod.clone();
  eval_mod.eval();
  // 冻结模块以便优化
  eval_mod = torch::jit::freeze(eval_mod);

  // 创建用于编译的字典，映射字符串到张量类型
  c10::Dict<IValue, IValue> compiled(StringType::get(), TensorType::get());

  // 输入和输出的 IValue
  c10::IValue inp;
  c10::IValue out;

  // 检查方法编译规范中是否包含 "forward" 键
  TORCH_CHECK(
      method_compile_spec.contains("forward"),
      "method_compile_spec does not contain the \"forward\" key.");
  auto innerDict = method_compile_spec.at("forward");

  // 检查 "forward" 字典中是否包含 "inputs" 和 "outputs" 键
  TORCH_CHECK(
      innerDict.isGenericDict() &&
          innerDict.toGenericDict().contains("inputs") &&
          innerDict.toGenericDict().contains("outputs"),
      "method_compile_spec does not contain a dictionary with an \"inputs\" key, under \"forward\" key.");

  // 从 "forward" 字典中获取输入和输出
  inp = innerDict.toGenericDict().at("inputs");
  out = innerDict.toGenericDict().at("outputs");

  // 检查输入是否为张量或张量列表
  TORCH_CHECK(
      inp.isTensor() || inp.isTensorList(),
      "method_compile_spec does not contain either a Tensor or TensorList, under it's \"inputs\" key.");
  // 检查输出是否为张量或张量列表
  TORCH_CHECK(
      out.isTensor() || out.isTensorList(),
      "method_compile_spec does not contain either a Tensor or TensorList, under it's \"outputs\" key.");

  // 图预处理
  const auto& forward_method = eval_mod.get_method("forward");

  // 获取 forward 方法的图表示，并复制该图
  auto graph = toGraphFunction(forward_method.function()).graph()->copy();
  // 移除未使用的 self 参数
  graph = tensorexpr::removeUnusedSelfArgument(graph);
  // 示例输入向量
  std::vector<c10::IValue> example_inputs;

  // 如果输入是张量列表
  if (inp.isTensorList()) {
    // 转换输入为张量列表
    c10::List<at::Tensor> inp_list = inp.toTensorList();
    // 检查图的输入数量是否与张量列表的大小相匹配
    TORCH_CHECK(
        graph->inputs().size() == inp_list.size(),
        "method_compile_spec inputs do not match expected number of forward inputs");

    // 预留空间并填充示例输入
    example_inputs.reserve(inp_list.size());
    for (const auto i : c10::irange(inp_list.size())) {
      example_inputs.emplace_back(inp_list[i]);
    }
  } else {
    // 检查图的输入数量是否为 1
    TORCH_CHECK(
        graph->inputs().size() == 1,
        "method_compile_spec inputs do not match expected number of forward inputs");


这样的方式能确保每行代码都被适当地注释，且注释不会超出代码块的范围。
    example_inputs.emplace_back(inp.toTensor());
  }



  // 将输入张量转换并添加到示例输入列表中
  example_inputs.emplace_back(inp.toTensor());



  // inp above has been confirmed to be either Tensor or TensorList
  // 确认上面的 inp 是 Tensor 或 TensorList
  XNNGraph graph_builder;
  // 创建 XNNGraph 对象
  graph_builder.buildXNNGraph(graph, example_inputs);
  // 使用示例输入构建 XNN 图形
  // 此时图形 graph 已经完整，为了测试预处理，在这一点上我们将进行运行时设置并使用一些默认值运行



  // grabbing the inputs from compile spec for testing
  // 从编译规范中获取输入以进行测试

  // gather sample inputs from compile spec
  // 从编译规范中收集样本输入
  std::vector<at::Tensor> inputs;
  auto input_list = inp.toList();

  for (int i = 0; i < input_list.size(); i++) {
    // 将输入列表中的每个项转换为张量并添加到输入向量中
    inputs.push_back(input_list.get(i).toTensor());
  }
  std::vector<at::Tensor> outputs;
  auto output_list = out.toList();
  std::vector<c10::IntList> output_shapes;

  // gather sample outputs from compile spec
  // 从编译规范中收集样本输出
  for (int i = 0; i < output_list.size(); i++) {
    auto sample_output = output_list.get(i).toTensor();
    // 将样本输出添加到输出向量中
    outputs.push_back(sample_output);
    // 同时收集输出形状以便传递到设备
    output_shapes.push_back(sample_output.sizes());
  }



  // sample run on sample inputs
  // 对样本输入运行图形
  graph_builder.runGraphOnInputs(inputs, outputs);
  c10::List<c10::IntList> shapes_list(output_shapes);

  // 将编译好的结果插入到 compiled 对象中
  compiled.insert("ser_model", graph_builder.serializedXNNGraph());
  compiled.insert("outputs", shapes_list);
  compiled.insert("Answer", outputs);

  // 返回编译好的结果
  return compiled;
}
// 结束当前命名空间 delegate
namespace delegate {
// 结束当前命名空间 xnnpack
namespace xnnpack {
// 结束当前命名空间 jit
namespace jit {
// 结束当前命名空间 torch
namespace torch
```