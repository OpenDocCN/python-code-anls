# `.\pytorch\test\cpp\jit\test_misc.cpp`

```
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/core/interned_strings.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type_base.h>
#include <c10/macros/Macros.h>
#include <test/cpp/jit/test_utils.h>
#include <torch/csrc/jit/passes/remove_mutation.h>
#include <torch/csrc/jit/passes/tensorexpr_fuser.h>
#include <torch/csrc/jit/tensorexpr/kernel.h>

#include <torch/csrc/autograd/engine.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/csrc/autograd/profiler.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/jit/api/function_impl.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/codegen/fuser/interface.h>
#include <torch/csrc/jit/frontend/ir_emitter.h>
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/ir/attributes.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/ir/scope.h>
#include <torch/csrc/jit/ir/type_hashing.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/bailout_graph.h>
#include <torch/csrc/jit/passes/canonicalize.h>
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/create_autodiff_subgraphs.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/graph_fuser.h>
#include <torch/csrc/jit/passes/guard_elimination.h>
#include <torch/csrc/jit/passes/inline_autodiff_subgraphs.h>
#include <torch/csrc/jit/passes/insert_guards.h>
#include <torch/csrc/jit/passes/liveness.h>
#include <torch/csrc/jit/passes/loop_unrolling.h>
#include <torch/csrc/jit/passes/lower_grad_of.h>
#include <torch/csrc/jit/passes/lower_tuples.h>
#include <torch/csrc/jit/passes/pass_manager.h>
#include <torch/csrc/jit/passes/requires_grad_analysis.h>
#include <torch/csrc/jit/passes/restore_mutation.h>
#include <torch/csrc/jit/passes/shape_analysis.h>
#include <torch/csrc/jit/passes/symbolic_shape_analysis.h>
#include <torch/csrc/jit/passes/utils/subgraph_utils.h>
#include <torch/csrc/jit/runtime/argument_spec.h>
#include <torch/csrc/jit/runtime/autodiff.h>
#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/runtime/decomposition_registry.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/csrc/jit/runtime/interpreter.h>
#include <torch/csrc/jit/runtime/jit_trace.h>
#include <torch/csrc/jit/runtime/profiling_record.h>
#include <torch/csrc/jit/runtime/symbolic_script.h>
#include <torch/csrc/jit/runtime/symbolic_shape_registry.h>
#include <torch/csrc/jit/serialization/import.h>
#include <torch/csrc/jit/testing/file_check.h>
#include <torch/jit.h>
#include <torch/script.h>

#include <onnx/onnx_pb.h>

#include <c10/util/Exception.h>
#include <c10/util/ThreadLocalDebugInfo.h>

#include <torch/csrc/jit/passes/freeze_module.h>



// 包含 Google Mock 和 Google Test 库的头文件
#include <gmock/gmock.h>
#include <gtest/gtest.h>

// 包含 PyTorch 的头文件
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/core/interned_strings.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type_base.h>
#include <c10/macros/Macros.h>
#include <test/cpp/jit/test_utils.h>
#include <torch/csrc/jit/passes/remove_mutation.h>
#include <torch/csrc/jit/passes/tensorexpr_fuser.h>
#include <torch/csrc/jit/tensorexpr/kernel.h>

// 包含 PyTorch 自动求导和 JIT 相关的头文件
#include <torch/csrc/autograd/engine.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/csrc/autograd/profiler.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/jit/api/function_impl.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/codegen/fuser/interface.h>
#include <torch/csrc/jit/frontend/ir_emitter.h>
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/ir/attributes.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/ir/scope.h>
#include <torch/csrc/jit/ir/type_hashing.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/bailout_graph.h>
#include <torch/csrc/jit/passes/canonicalize.h>
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/create_autodiff_subgraphs.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/graph_fuser.h>
#include <torch/csrc/jit/passes/guard_elimination.h>
#include <torch/csrc/jit/passes/inline_autodiff_subgraphs.h>
#include <torch/csrc/jit/passes/insert_guards.h>
#include <torch/csrc/jit/passes/liveness.h>
#include <torch/csrc/jit/passes/loop_unrolling.h>
#include <torch/csrc/jit/passes/lower_grad_of.h>
#include <torch/csrc/jit/passes/lower_tuples.h>
#include <torch/csrc/jit/passes/pass_manager.h>
#include <torch/csrc/jit/passes/requires_grad_analysis.h>
#include <torch/csrc/jit/passes/restore_mutation.h>
#include <torch/csrc/jit/passes/shape_analysis.h>
#include <torch/csrc/jit/passes/symbolic_shape_analysis.h>
#include <torch/csrc/jit/passes/utils/subgraph_utils.h>
#include <torch/csrc/jit/runtime/argument_spec.h>
#include <torch/csrc/jit/runtime/autodiff.h>
#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/runtime/decomposition_registry.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/csrc/jit/runtime/interpreter.h>
#include <torch/csrc/jit/runtime/jit_trace.h>
#include <torch/csrc/jit/runtime/profiling_record.h>
#include <torch/csrc/jit/runtime/symbolic_script.h>
#include <torch/csrc/jit/runtime/symbolic_shape_registry.h>
#include <torch/csrc/jit/serialization/import.h>
#include <torch/csrc/jit/testing/file_check.h>
#include <torch/jit.h>
#include <torch/script.h>

// 包含 ONNX 的头文件
#include <onnx/onnx_pb.h>

// 包含 C10 的异常处理和调试信息头文件
#include <c10/util/Exception.h>
#include <c10/util/ThreadLocalDebugInfo.h>

// 包含 PyTorch JIT passes 中的 freeze_module 头文件
#include <torch/csrc/jit/passes/freeze_module.h>
#include <torch/csrc/jit/passes/frozen_graph_optimizations.h>
// 包含 Torch 的头文件 frozen_graph_optimizations.h

#include <algorithm>
// 包含算法标准库头文件

#include <cstddef>
// 包含大小标准库头文件

#include <functional>
// 包含函数对象标准库头文件

#include <iostream>
// 包含输入输出流标准库头文件

#include <memory>
// 包含智能指针标准库头文件

#include <set>
// 包含集合容器标准库头文件

#include <stdexcept>
// 包含标准异常类头文件

#include <string>
// 包含字符串标准库头文件

#include <tuple>
// 包含元组标准库头文件

#include <unordered_map>
// 包含无序映射容器标准库头文件

#include <unordered_set>
// 包含无序集合容器标准库头文件

#include <utility>
// 包含实用程序标准库头文件

#include <vector>
// 包含向量容器标准库头文件

namespace torch {
namespace jit {
// Torch 和 JIT 命名空间

inline c10::AliasAnalysisKind aliasAnalysisFromSchema() {
  // 定义一个内联函数，返回 AliasAnalysisKind 枚举类型，从模式中获取
  return c10::AliasAnalysisKind::FROM_SCHEMA;
}

template <typename T>
std::ostream& operator<<(std::ostream& out, const std::vector<T>& list) {
  // 定义一个模板函数，用于向输出流中输出向量内容
  size_t i = 0;
  out << "{";
  for (auto&& e : list) {
    if (i++ > 0)
      out << ", ";
    out << e;
  }
  out << "}";
  return out;
}

TEST(InternedStringsTest, Basic) {
  // 测试用例：InternedStringsTest，基本测试

  ASSERT_EQ(prim::Param, Symbol::prim("Param"));
  // 断言：prim::Param 等于 Symbol::prim("Param")

  ASSERT_EQ(prim::Return, Symbol::prim("Return"));
  // 断言：prim::Return 等于 Symbol::prim("Return")

  ASSERT_EQ(prim::Return.toUnqualString(), std::string("Return"));
  // 断言：prim::Return 的非限定字符串等于 "Return"

  ASSERT_EQ(prim::Return.toQualString(), std::string("prim::Return"));
  // 断言：prim::Return 的限定字符串等于 "prim::Return"

  Symbol newsym = Symbol::aten("__NEW_SYMBOL");
  // 创建一个新符号 newsym，使用 Symbol::aten("__NEW_SYMBOL")

  size_t symstart = newsym;
  // 获取 newsym 的起始符号值

  ASSERT_EQ(newsym.toQualString(), std::string("aten::__NEW_SYMBOL"));
  // 断言：newsym 的限定字符串等于 "aten::__NEW_SYMBOL"

  // TODO: This test is a bit too close to the implementation details.
  ASSERT_EQ(Symbol::aten("What"), symstart + 1);
  // 断言：Symbol::aten("What") 等于 symstart + 1

  ASSERT_EQ(Symbol::aten("What2"), symstart + 2);
  // 断言：Symbol::aten("What2") 等于 symstart + 2

  ASSERT_EQ(Symbol::aten("What"), symstart + 1);
  // 断言：Symbol::aten("What") 等于 symstart + 1

  ASSERT_EQ(Symbol::aten("What2"), symstart + 2);
  // 断言：Symbol::aten("What2") 等于 symstart + 2

  ASSERT_EQ(Symbol(symstart + 2).toUnqualString(), std::string("What2"));
  // 断言：符号 symstart + 2 的非限定字符串等于 "What2"
}

TEST(FromQualStringTest, Basic) {
  // 测试用例：FromQualStringTest，基本测试

  ASSERT_EQ(Symbol::fromQualString("prim::Param"), Symbol::prim("Param"));
  // 断言：从限定字符串 "prim::Param" 创建的符号等于 Symbol::prim("Param")

  ASSERT_EQ(Symbol::fromQualString("aten::mm"), Symbol::aten("mm"));
  // 断言：从限定字符串 "aten::mm" 创建的符号等于 Symbol::aten("mm")

  ASSERT_EQ(Symbol::fromQualString("onnx::LSTM"), Symbol::onnx("LSTM"));
  // 断言：从限定字符串 "onnx::LSTM" 创建的符号等于 Symbol::onnx("LSTM")

  ASSERT_EQ(Symbol::fromQualString("attr::value"), Symbol::attr("value"));
  // 断言：从限定字符串 "attr::value" 创建的符号等于 Symbol::attr("value")

  ASSERT_EQ(Symbol::fromQualString("scope::"), Symbol::scope(""));
  // 断言：从限定字符串 "scope::" 创建的符号等于 Symbol::scope("")

  ASSERT_EQ(Symbol::fromQualString("::").toUnqualString(), std::string(""));
  // 断言：从限定字符串 "::" 创建的符号的非限定字符串为空字符串 ""

  ASSERT_EQ(
      Symbol::fromQualString("::").ns().toQualString(),
      std::string("namespaces::"));
  // 断言：从限定字符串 "::" 创建的符号的命名空间的限定字符串为 "namespaces::"

  ASSERT_EQ(
      Symbol::fromQualString("new_ns::param").toUnqualString(),
      std::string("param"));
  // 断言：从限定字符串 "new_ns::param" 创建的符号的非限定字符串为 "param"

  ASSERT_EQ(
      Symbol::fromQualString("new_ns::param").ns().toUnqualString(),
      std::string("new_ns"));
  // 断言：从限定字符串 "new_ns::param" 创建的符号的命名空间的非限定字符串为 "new_ns"

  ASSERT_EQ(
      Symbol::fromQualString("new_ns::param").ns(),
      Symbol::fromQualString("namespaces::new_ns"));
  // 断言：从限定字符串 "new_ns::param" 创建的符号的命名空间等于 Symbol::fromQualString("namespaces::new_ns")

  auto bad_inputs = {"scope", ":", ""};
  // 定义一个包含错误输入的字符串数组

  for (auto input : bad_inputs) {
    try {
      Symbol::fromQualString(input);
      ASSERT_TRUE(0);
      // 尝试从输入创建符号，如果成功则断言失败
    } catch (const std::exception& c) {
      // 捕获可能抛出的异常
    }
  }
}
TEST(THNNConvTest, Basic) {
  std::vector<int64_t> input_size = {4, 3, 15, 17}; // B x C x H x W，输入张量的形状
  std::vector<int64_t> kernel_size = {3, 5};       // 卷积核大小
  std::vector<int64_t> stride = {1, 2};            // 卷积步长
  std::vector<int64_t> padding = {2, 1};           // 卷积填充
  constexpr int out_channels = 5;                  // 输出通道数

  // make inputs，生成随机输入张量
  at::Tensor input = torch::randn(input_size);
  // 生成随机权重张量
  at::Tensor weight = torch::randn(
      {out_channels, input_size[1], kernel_size[0], kernel_size[1]});
  // 生成随机偏置张量
  at::Tensor bias = torch::randn({out_channels});

  // run forward eagerly，前向传播
  at::Tensor output = at::_slow_conv2d_forward(
      input, weight, kernel_size, bias, stride, padding);

  // make grad_outputs，生成随机梯度输出张量
  at::Tensor grad_output =
      torch::randn_like(output, at::MemoryFormat::Preserve);

  // run backward eagerly，反向传播
  at::Tensor grad_input, grad_weight, grad_bias;
  std::tie(grad_input, grad_weight, grad_bias) = at::_slow_conv2d_backward(
      grad_output,
      input,
      weight,
      kernel_size,
      stride,
      padding,
      {true, true, true});

  // make JIT graph，创建 JIT 图
  auto graph = std::make_shared<Graph>();
  auto ksz_val = graph->insertConstant(kernel_size);   // 插入卷积核大小常量
  auto kst_val = graph->insertConstant(stride);        // 插入卷积步长常量
  auto pad_val = graph->insertConstant(padding);       // 插入卷积填充常量

  auto inputg = graph->addInput("self");               // 图中添加输入节点
  auto weightg = graph->addInput("weight");            // 图中添加权重节点
  auto biasg = graph->addInput("bias");                // 图中添加偏置节点

  // 在图中插入卷积操作节点
  Value* conv = graph->insert(
      aten::_slow_conv2d_forward,
      {inputg, weightg, ksz_val, biasg, kst_val, pad_val});
  auto outputs = conv->node()->outputs();              // 获取卷积操作的输出节点
  for (auto output : outputs) {
    graph->registerOutput(output);                     // 注册输出节点
  }
  LowerAllTuples(graph);                              // 降低图中所有元组节点
  graph->lint();                                      // 检查图的一致性

  // differentiate JIT graph，对 JIT 图进行求导
  EliminateDeadCode(graph);                            // 消除死代码，优化图
  ConstantPropagation(graph);                          // 常量传播，优化图
  auto grad_spec = differentiate(graph);               // 对图进行求导
  LowerGradOf(*grad_spec.df);                          // 降低梯度函数

  // prepare JIT inputs / gradients，准备 JIT 输入和梯度
  tensor_list tensors_in;
  tensors_in.push_back(input);
  tensors_in.push_back(weight);
  tensors_in.push_back(bias);

  tensor_list tensor_grads_in;
  tensor_grads_in.push_back(grad_output);

  // Get outputs from the interpreter，从解释器中获取输出
  tensor_list tensors_out, tensor_grads_out;
  std::tie(tensors_out, tensor_grads_out) =
      runGradient(grad_spec, tensors_in, tensor_grads_in);

  // prepare expected structs，准备预期结果
  tensor_list expected_tensors_out, expected_tensor_grads_out;
  expected_tensors_out.push_back(output);
  expected_tensor_grads_out.push_back(grad_input);
  expected_tensor_grads_out.push_back(grad_weight);
  expected_tensor_grads_out.push_back(grad_bias);

  // Compare results，比较结果
  assertAllClose(tensors_out, expected_tensors_out);         // 断言输出张量近似相等
  assertAllClose(tensor_grads_out, expected_tensor_grads_out); // 断言梯度张量近似相等
}
TEST(ATenNativeBatchNormTest, Basic) {
  // 定义测试用例名称和基本测试函数

  // aten::native_batch_norm(Tensor input, Tensor weight, Tensor bias, Tensor
  // running_mean, Tensor running_var, bool training, float momentum, float eps)
  // -> (Tensor, Tensor, Tensor)
  // 定义批归一化函数的签名和返回值类型

  std::vector<int64_t> input_size = {4, 3, 15, 17}; // B x C x H x W
  // 定义输入张量的尺寸

  bool training = true;
  // 指定是否处于训练模式

  float momentum = 0.9;
  // 设置动量值

  float eps = 1e-5;
  // 设置 epsilon 值，用于数值稳定性

  // make inputs
  at::Tensor input = torch::randn(input_size);
  // 生成随机的输入张量

  at::Tensor weight = torch::randn({input_size[1]});
  // 生成随机的权重张量，维度与通道数相同

  at::Tensor bias = torch::randn({input_size[1]});
  // 生成随机的偏置张量，维度与通道数相同

  at::Tensor running_mean = torch::randn({input_size[1]});
  // 生成随机的运行时均值张量，维度与通道数相同

  at::Tensor running_var = torch::randn({input_size[1]});
  // 生成随机的运行时方差张量，维度与通道数相同

  // running_mean and running_var are changed in-place, so clone and send them
  at::Tensor running_mean_eager = running_mean.clone();
  // 深拷贝运行时均值张量以便后续使用

  at::Tensor running_var_eager = running_var.clone();
  // 深拷贝运行时方差张量以便后续使用

  at::Tensor running_mean_jit = running_mean.clone();
  // 深拷贝运行时均值张量以便 JIT 使用

  at::Tensor running_var_jit = running_var.clone();
  // 深拷贝运行时方差张量以便 JIT 使用

  // run forward eagerly
  at::Tensor output, savemean, saveinvstd;
  // 定义前向传播输出张量及保存的均值和标准差张量

  std::tie(output, savemean, saveinvstd) = at::native_batch_norm(
      input,
      weight,
      bias,
      running_mean_eager,
      running_var_eager,
      training,
      momentum,
      eps);
  // 执行前向传播计算批归一化

  // make grad_outputs
  at::Tensor grad_output =
      torch::randn_like(output, at::MemoryFormat::Preserve);
  // 生成与输出张量相同尺寸的随机梯度输出张量

  at::Tensor grad_savemean =
      torch::zeros_like(savemean, at::MemoryFormat::Preserve);
  // 生成与均值保存张量相同尺寸的零张量，用于梯度计算

  at::Tensor grad_saveinvstd =
      torch::zeros_like(saveinvstd, at::MemoryFormat::Preserve);
  // 生成与标准差保存张量相同尺寸的零张量，用于梯度计算

  // run backward eagerly
  at::Tensor grad_input, grad_weight, grad_bias;
  // 定义反向传播的输入梯度张量、权重梯度张量和偏置梯度张量

  // aten::native_batch_norm_backward(Tensor grad_out, Tensor input, Tensor
  // weight, Tensor running_mean, Tensor running_var, Tensor save_mean, Tensor
  // save_invstd, bool train, float eps, bool[3] output_mask) -> (Tensor,
  // Tensor, Tensor)
  // 定义批归一化反向传播函数的签名和返回值类型

  std::tie(grad_input, grad_weight, grad_bias) = at::native_batch_norm_backward(
      grad_output,
      input,
      weight,
      running_mean_eager,
      running_var_eager,
      savemean,
      saveinvstd,
      training,
      eps,
      {true, true, true});
  // 执行批归一化的反向传播计算

  // make JIT graph
  auto graph = std::make_shared<Graph>();
  // 创建 JIT 图

  auto training_val = graph->insertConstant(IValue(training));
  // 在 JIT 图中插入训练标志常量

  auto momentum_val = graph->insertConstant(IValue(momentum));
  // 在 JIT 图中插入动量常量

  auto eps_val = graph->insertConstant(IValue(eps));
  // 在 JIT 图中插入 epsilon 常量

  auto inputg = graph->addInput("self");
  // 在 JIT 图中添加输入张量节点

  auto weightg = graph->addInput("weight");
  // 在 JIT 图中添加权重张量节点

  auto biasg = graph->addInput("bias");
  // 在 JIT 图中添加偏置张量节点

  auto running_meang = graph->addInput("running_mean");
  // 在 JIT 图中添加运行时均值张量节点

  auto running_varg = graph->addInput("running_var");
  // 在 JIT 图中添加运行时方差张量节点

  Value* bn = graph->insert(
      aten::native_batch_norm,
      {inputg,
       weightg,
       biasg,
       running_meang,
       running_varg,
       training_val,
       momentum_val,
       eps_val});
  // 在 JIT 图中插入批归一化前向传播函数节点

  auto outputs = bn->node()->outputs();
  // 获取节点的输出

  for (auto output : outputs) {
    // 遍历节点的所有输出
  graph->registerOutput(output);
  // 将输出节点注册到计算图中

  LowerAllTuples(graph);
  // 降低计算图中所有元组的复杂度

  graph->lint();
  // 检查计算图的一致性和规范性

  // differentiate JIT graph
  // 对 JIT 图进行求导操作
  EliminateDeadCode(graph); // Tracing of some ops depends on the DCE trick
  // 消除死代码，某些操作的追踪依赖于死代码消除技巧
  ConstantPropagation(graph);
  // 常量传播优化
  auto grad_spec = differentiate(graph);
  // 对图进行求导，返回求导规范
  LowerGradOf(*grad_spec.df);
  // 降低求导的结果

  // prepare JIT inputs / gradients
  // 准备 JIT 输入和梯度
  tensor_list tensors_in;
  // 输入张量列表
  tensors_in.push_back(input);
  tensors_in.push_back(weight);
  tensors_in.push_back(bias);
  tensors_in.push_back(running_mean_jit);
  tensors_in.push_back(running_var_jit);

  tensor_list tensor_grads_in;
  // 梯度张量列表
  tensor_grads_in.push_back(grad_output);
  tensor_grads_in.push_back(grad_savemean);
  tensor_grads_in.push_back(grad_saveinvstd);

  // Get outputs from the interpreter
  // 从解释器获取输出
  tensor_list tensors_out, tensor_grads_out;
  std::tie(tensors_out, tensor_grads_out) =
      runGradient(grad_spec, tensors_in, tensor_grads_in);

  // prepare expected structs
  // 准备期望的数据结构
  tensor_list expected_tensors_out, expected_tensor_grads_out;
  expected_tensors_out.push_back(output);
  expected_tensors_out.push_back(savemean);
  expected_tensors_out.push_back(saveinvstd);
  expected_tensors_out.push_back(running_mean_eager);
  expected_tensors_out.push_back(running_var_eager);
  expected_tensor_grads_out.push_back(grad_input);
  expected_tensor_grads_out.push_back(grad_weight);
  expected_tensor_grads_out.push_back(grad_bias);

  tensors_out.push_back(running_mean_jit);
  tensors_out.push_back(running_var_jit);

  // Compare results
  // 比较结果
  assertAllClose(tensors_out, expected_tensors_out);
  assertAllClose(tensor_grads_out, expected_tensor_grads_out);
TEST(CustomFusionTest, NestedBlocks) {
#if defined(FBCODE_CAFFE2)
  return;
#endif

  // 定义一个字符串，包含表示图形的内部表示（IR）
  auto graph_string = R"IR(
  graph(%0 : Float(2, 3, 4),
        %1 : Float(2, 3, 4),
        %2 : Float(2, 3, 4)):
    %3 : int = prim::Constant[value=1]()
    %4 : Tensor = prim::If(%2)
      block0():
        %5 : Tensor = aten::mul(%0, %2)
        %6 : Tensor = aten::mul(%5, %1)
        -> (%6)
      block1():
        %7 : Tensor = aten::add(%0, %2, %3)
        %8 : Tensor = aten::add(%7, %1, %3)
        -> (%8)
    %9 : Tensor = aten::add(%4, %2, %3)
    return (%4))IR";

  // 创建一个共享指针，指向新建的图对象
  auto g = std::make_shared<Graph>();
  // 解析图形字符串，将其填充到图对象中
  torch::jit::parseIR(graph_string, g.get());

  // 将满足条件的操作节点进行融合为一个 FusionGroup 节点
  CustomFuseGraph(
      g,
      [](Node* n) { return n->kind() == aten::mul; },
      Symbol::fromQualString("prim::FusionGroup"));

  // 定义一个深度优先搜索函数，用于查找特定类型的子图块
  std::function<bool(const Block*, Symbol)> dfs = [&](const Block* b,
                                                      Symbol s) {
    for (auto node : b->nodes()) {
      // 如果找到了指定类型的节点，则返回 true
      if (node->kind() == s)
        return true;
      // 递归遍历嵌套的子图块
      for (auto nested_b : node->blocks())
        if (dfs(nested_b, s))
          return true;
    }
    return false;
  };

  // 断言是否存在 FusionGroup 类型的节点
  AT_ASSERT(dfs(g->block(), Symbol::fromQualString("prim::FusionGroup")));
}
    return stack;
  };

  // 定义 lambda 函数 L，接受一个 int64_t 类型的参数 l，返回对应的 IValue
  auto L = [](int64_t l) { return IValue(scalar_to_tensor(at::Scalar(l))); };

  // 定义 lambda 函数 V，接受一个 IValue 类型的参数 t，将其转换为 Tensor 后取出 int64_t 类型的数值
  auto V = [](IValue t) { return std::move(t).toTensor().item<int64_t>(); };

  // 定义 run_binary 函数，接受运算符名称 name、两个 int64_t 类型的参数 a 和 b
  // 调用 run 函数执行运算，并返回运算结果中第一个元素的 Tensor 的 int64_t 值
  auto run_binary = [&](const std::string& name, int64_t a, int64_t b) {
    return V(run(name, {L(a), L(b)})[0]);
  };

  // 断言以下测试结果是否相等，测试 run_binary 函数的运行结果
  ASSERT_EQ(2, run_binary("if_test", 1, 2));
  ASSERT_EQ(3, run_binary("if_test", 3, 2));
  ASSERT_EQ(2, run_binary("if_one", 2, 3));
  ASSERT_EQ(2, run_binary("if_one", 3, 2));
  ASSERT_EQ(256, run_binary("while_test", 2, 0));
// 定义了一个测试用例 ProtoTest，用于测试 Protocol Buffer 的基本功能
TEST(ProtoTest, Basic) {
  // 创建一个 ModelProto 对象
  ::ONNX_NAMESPACE::ModelProto proto;
  // 设置生产者名称为 "foo"
  proto.set_producer_name("foo");
}

// 定义了一个测试用例 SchemaParserTest，用于测试 Schema 解析器的功能
TEST(SchemaParserTest, NestedArrays) {
  // 解析包含嵌套数组的模式字符串，并进行断言检查
  auto s = parseSchema("at::what(int[][4] foo) -> ()");
  ASSERT_TRUE(s.arguments().at(0).N() == 4);
  ASSERT_TRUE(IntType::get()->isSubtypeOf(*s.arguments()
                                               .at(0)
                                               .type()
                                               ->expectRef<ListType>()
                                               .getElementType()
                                               ->expectRef<ListType>()
                                               .getElementType()));
  
  // 解析包含嵌套数组的另一个模式字符串，并进行断言检查
  auto s2 = parseSchema("at::what(int[][] foo) -> ()");
  ASSERT_TRUE(IntType::get()->isSubtypeOf(*s2.arguments()
                                               .at(0)
                                               .type()
                                               ->expectRef<ListType>()
                                               .getElementType()
                                               ->expectRef<ListType>()
                                               .getElementType()));
}

// 定义了一个测试用例 SchemaParserTest，用于测试带有 out 变量的模式解析
TEST(SchemaParserTest, OutVariant) {
  // 解析包含 out 变量的模式字符串，并进行断言检查
  auto schema_with_out = parseSchema(
      "at::foo(Tensor self, *, Tensor(a!) f, Tensor(b!) l) -> (Tensor(a!) f, Tensor(b!) l)");
  ASSERT_TRUE(schema_with_out.arguments().at(1).is_out());
  ASSERT_TRUE(schema_with_out.arguments().at(2).is_out());

  // 解析不包含 out 变量的模式字符串，并进行断言检查
  auto schema_without_out =
      parseSchema("at::foo(Tensor self, *, int scalar) -> (int)");

  for (const auto& arg : schema_without_out.arguments()) {
    ASSERT_TRUE(!arg.is_out());
  }

  // 解析包含写入操作的模式字符串，并进行断言检查
  auto schema_with_is_write = parseSchema(
      "aten::ne_.Scalar(Tensor(a!) self, Scalar other) -> (Tensor(a!))");

  for (const auto& arg : schema_with_is_write.arguments()) {
    ASSERT_TRUE(!arg.is_out());
  }
}

// 定义了一个测试用例 SchemaParserTest，用于测试带有命名返回值的模式解析
TEST(SchemaParserTest, NamedReturns) {
  // 解析包含命名返回值的模式字符串
  parseSchema("at::what(Tensor! i_will_be_written_to) -> ()");
  // 解析包含多个命名返回值的模式字符串，并进行断言检查
  auto s3 =
      parseSchema("at::what() -> (Tensor the_return, Tensor the_return2)");
  ASSERT_TRUE(s3.returns().at(0).name() == "the_return");
  ASSERT_TRUE(s3.returns().at(1).name() == "the_return2");
}

// 定义了一个测试用例 SchemaParserTest，用于测试带有 Futures 类型的模式解析
TEST(SchemaParserTest, Futures) {
  // 解析包含 Futures 类型参数的模式字符串，并进行断言检查
  auto s4 = parseSchema("at::what(Future(int) foo) -> ()");
  ASSERT_TRUE(IntType::get()->isSubtypeOf(
      *s4.arguments().at(0).type()->expectRef<FutureType>().getElementType()));
}

// 定义了一个测试用例 SchemaParserTest，用于测试带有注解别名集合的模式解析
TEST(SchemaParserTest, AnnotatedAliasSets) {
  // 解析包含带有注解别名集合的 Tensor 参数的模式字符串
  parseSchema("at::what(Tensor(a) foo) -> (Tensor(a))");
}
TEST(SchemaParserTest, TensorListAnnotatedAliasSets) {
  // 解析给定的函数签名字符串，返回一个 Schema 对象
  const auto s = parseSchema(
      "at::foo(Tensor(a!) self, Tensor(b!)[] out)"
      " -> ()");
  // 获取 self 参数的别名信息
  const AliasInfo* selfAliasInfo = s.arguments().at(0).alias_info();
  // 获取 out 参数的别名信息
  const AliasInfo* outAliasInfo = s.arguments().at(1).alias_info();
  // 断言 self 参数的 beforeSets 包含预期的符号集合
  ASSERT_TRUE(
      selfAliasInfo->beforeSets() ==
      std::unordered_set<Symbol>{Symbol::fromQualString("alias::a")});
  // 断言 self 参数是写操作
  ASSERT_TRUE(selfAliasInfo->isWrite());

  // 断言 out 参数是写操作
  ASSERT_TRUE(outAliasInfo->isWrite());
  // 断言 out 参数的 beforeSets 为空集合
  ASSERT_TRUE(outAliasInfo->beforeSets().empty());
  // 断言 out 参数的 containedTypes 大小为 1
  ASSERT_EQ(outAliasInfo->containedTypes().size(), 1);

  // 获取 containedTypes 的第一个元素
  auto containedType = outAliasInfo->containedTypes()[0];

  // 断言 containedType 是写操作
  ASSERT_TRUE(containedType.isWrite());
  // 断言 containedType 的 beforeSets 包含预期的符号集合
  ASSERT_TRUE(
      containedType.beforeSets() ==
      std::unordered_set<Symbol>{Symbol::fromQualString("alias::b")});
}

TEST(SchemaParserTest, AnnotatedAliasWithoutBeforeSet) {
  // 断言解析包含错误标识符的函数签名会抛出运行时错误
  EXPECT_THAT(
      []() { parseSchema("at::foo(Tensor(!) self) -> Tensor"); },
      ::testing::Throws<std::runtime_error>(::testing::Property(
          &std::runtime_error::what,
          ::testing::HasSubstr("expected ident but found '!' here"))));
}

TEST(SchemaParserTest, BeforeAfterSets) {
  // 解析给定的函数签名字符串，返回一个 Schema 对象
  const auto s = parseSchema(
      "at::what(Tensor(b|c)[](a!) list, Tensor(c) element)"
      " -> (Tensor(b|c)[](a!))");

  // 获取 list 参数的别名信息
  const AliasInfo* aliasInfo = s.arguments().at(0).alias_info();
  // 断言 list 参数的 beforeSets 包含预期的符号集合
  ASSERT_TRUE(
      aliasInfo->beforeSets() ==
      std::unordered_set<Symbol>{Symbol::fromQualString("alias::a")});
  // 断言 list 参数是写操作
  ASSERT_TRUE(aliasInfo->isWrite());

  // 检查 containedTypes
  ASSERT_TRUE(!aliasInfo->containedTypes().empty());
  const auto& containedAliasInfo = aliasInfo->containedTypes()[0];
  const auto expected = std::unordered_set<Symbol>{
      Symbol::fromQualString("alias::b"),
      Symbol::fromQualString("alias::c"),
  };
  // 断言 containedAliasInfo 的 beforeSets 包含预期的符号集合
  ASSERT_TRUE(containedAliasInfo.beforeSets() == expected);
  // 断言 containedAliasInfo 的 afterSets 包含预期的符号集合
  ASSERT_TRUE(containedAliasInfo.afterSets() == expected);
  // 断言 containedAliasInfo 不是写操作
  ASSERT_FALSE(containedAliasInfo.isWrite());
}
TEST(SchemaParserTest, BeforeAfterSets2) {
  // 解析给定的模式字符串，生成对应的 Schema 对象 s
  const auto s = parseSchema(
      "at::what(Tensor(b -> b|c)[](a!) list, Tensor(c) element)"
      " -> (Tensor(b|c)[](a!))");

  // 获取第一个参数的别名信息
  const AliasInfo* aliasInfo = s.arguments().at(0).alias_info();
  // 断言别名信息不为空
  ASSERT_NE(aliasInfo, nullptr);
  // 断言该参数的 beforeSets 包含符号 "alias::a"
  ASSERT_EQ(
      aliasInfo->beforeSets(),
      std::unordered_set<Symbol>{Symbol::fromQualString("alias::a")});
  // 断言该参数的 afterSets 包含符号 "alias::a"
  ASSERT_EQ(
      aliasInfo->afterSets(),
      std::unordered_set<Symbol>{Symbol::fromQualString("alias::a")});
  // 断言该参数为写操作
  ASSERT_TRUE(aliasInfo->isWrite());
  // 断言该参数的 containedTypes 集合大小为 1
  ASSERT_EQ(aliasInfo->containedTypes().size(), 1);

  // 检查 containedTypes 集合非空
  ASSERT_TRUE(!aliasInfo->containedTypes().empty());
  // 获取 containedTypes 集合中的第一个元素的别名信息
  const auto& containedAliasInfo = aliasInfo->containedTypes()[0];
  // 期望的 containedAliasInfo 的 beforeSets 集合
  const auto expectedBefore = std::unordered_set<Symbol>{
      Symbol::fromQualString("alias::b"),
  };
  // 期望的 containedAliasInfo 的 afterSets 集合
  const auto expectedAfter = std::unordered_set<Symbol>{
      Symbol::fromQualString("alias::b"), Symbol::fromQualString("alias::c")};
  // 断言 containedAliasInfo 的 beforeSets 与期望的一致
  ASSERT_TRUE(containedAliasInfo.beforeSets() == expectedBefore);
  // 断言 containedAliasInfo 的 afterSets 与期望的一致
  ASSERT_TRUE(containedAliasInfo.afterSets() == expectedAfter);
  // 断言 containedAliasInfo 不是写操作
  ASSERT_FALSE(containedAliasInfo.isWrite());
}

TEST(TopologicalIndexTest, Basic) {
  // 创建一个空的 Graph 对象
  Graph graph;
  // 创建四个具有 AutogradZero 类型的节点
  auto node1 = graph.create(prim::AutogradZero);
  auto node2 = graph.create(prim::AutogradZero);
  auto node3 = graph.create(prim::AutogradZero);
  auto node4 = graph.create(prim::AutogradZero);

  // 将 node4 添加到图中
  graph.appendNode(node4);
  // 将 node1 添加到图的开头
  graph.prependNode(node1);
  // 将 node2 插入到 node1 后面
  node2->insertAfter(node1);
  // 将 node3 插入到 node4 前面
  node3->insertBefore(node4);

  // 断言节点按照数值顺序排列
  ASSERT_TRUE(node1->isBefore(node2));
  ASSERT_TRUE(node1->isBefore(node3));
  ASSERT_TRUE(node1->isBefore(node4));
  ASSERT_TRUE(node2->isAfter(node1));
  ASSERT_TRUE(node2->isBefore(node3));
  ASSERT_TRUE(node2->isBefore(node4));
  ASSERT_FALSE(node3->isBefore(node1));
  ASSERT_FALSE(node3->isBefore(node2));
  ASSERT_FALSE(node3->isAfter(node4));

  // 构建一个块结构
  //  node3
  //   /\
  //  A  B
  //      \
  //      C
  auto block1 = node3->addBlock();
  auto A = graph.create(prim::AutogradZero);
  block1->appendNode(A);
  auto B = graph.create(prim::AutogradZero);
  block1->appendNode(B);
  auto block2 = B->addBlock();
  auto C = graph.create(prim::AutogradZero);
  block2->appendNode(C);

  // 检查不同块级别上的 isAfter
  ASSERT_TRUE(node1->isBefore(A));
  ASSERT_TRUE(A->isBefore(B));
  ASSERT_TRUE(A->isBefore(C));

  // 确保删除操作不会导致异常
  node2->destroy();
  auto node2p = graph.create(prim::AutogradZero);
  node2p->insertAfter(node1);
  ASSERT_TRUE(node1->isBefore(node2p));
  ASSERT_TRUE(node2p->isBefore(node3));
}
TEST(TopologicalIndexTest, Reindex) {
  // Induce reindexing to test that path
  // 创建一个图对象
  Graph graph;
  // 存储节点指针的映射
  std::map<size_t, Node*> nodes;

  // 创建一个起始节点
  auto anchor = graph.create(prim::AutogradZero);
  // 将起始节点添加到图中
  graph.appendNode(anchor);
  
  // 插入大量节点到相同位置将触发重新索引
  for (auto i = 0; i < 100; ++i) {
    // 创建一个新节点
    auto n = graph.create(prim::AutogradZero);
    // 在起始节点之后插入新节点
    n->insertAfter(anchor);
    // 将节点存储到映射中
    nodes[i] = n;
  }

  // 节点应该是倒序排列的
  for (auto i = 0; i < 100; ++i) {
    for (auto j = i + 1; j < 100; ++j) {
      // 断言节点 i 在节点 j 之后
      ASSERT_TRUE(nodes[i]->isAfter(nodes[j]));
    }
  }
}

at::Tensor invokeTestRecordFunction(at::Tensor& t) {
  // 记录函数调用 "test"，并传入张量 t
  RECORD_FUNCTION("test", std::vector<c10::IValue>({t}));

  // 对张量 t 进行平方操作
  auto t2 = t.pow(2);
  // 返回平方后的张量
  return t2;
}

static const auto invokeTestRecordFunction_JIT = R"JIT(
  def foo(self, t):
    t2 = t.pow(2)
    return t2

  def forward(self, t):
    return self.foo(t)
)JIT";

at::Tensor invokeTestRecordFunctionJIT(at::Tensor& t) {
  // 记录函数调用 "test"，并传入张量 t
  RECORD_FUNCTION("test", std::vector<c10::IValue>({t}));

  // 创建一个名为 RecordFunctionTestModule 的脚本模块
  auto module = std::make_shared<script::Module>(
      "RecordFunctionTestModule", std::make_shared<script::CompilationUnit>());
  // 定义模块的前向传播逻辑
  module->define(invokeTestRecordFunction_JIT);
  // 执行模块的前向传播并返回结果张量
  return module->forward({t}).toTensor();
}

using TracedTestValues =
    std::vector<std::tuple<std::string, std::vector<std::vector<int64_t>>>>;

void checkTracedInputs(const TracedTestValues& inputs) {
  // 标志变量，用于跟踪是否找到了特定的函数调用
  bool found_test = false;
  bool found_pow = false;
  bool found_mul = false;
  
  // 遍历输入的跟踪值
  for (const auto& input : inputs) {
    // 获取函数名称和尺寸信息
    const auto& fn = std::get<0>(input);
    const auto& sizes = std::get<1>(input);

    // 检查是否找到了 "test" 函数调用
    if (fn == "test") {
      found_test = true;
      // 断言尺寸信息的有效性
      TORCH_CHECK(sizes.size() == 1);
      TORCH_CHECK(sizes[0] == std::vector<int64_t>({1, 2, 3}));
    } else if (fn == "aten::pow") {
      found_pow = true;
      // 断言尺寸信息的有效性
      TORCH_CHECK(sizes.size() == 2);
      TORCH_CHECK(sizes[0] == std::vector<int64_t>({1, 2, 3}));
      TORCH_CHECK(sizes[1].empty());
    } else if (fn == "aten::mul") {
      found_mul = true;
      // 断言尺寸信息的有效性
      TORCH_CHECK(sizes.size() > 1);
      TORCH_CHECK(sizes[0] == std::vector<int64_t>({1, 2, 3}));
    }
  }
  // 断言是否找到了所有预期的函数调用
  TORCH_CHECK(found_test);
  TORCH_CHECK(found_pow);
  TORCH_CHECK(found_mul);
}

void checkTracedOutputs(const TracedTestValues& outputs) {
  // 标志变量，用于跟踪是否找到了特定的函数调用
  bool found_test = false;
  bool found_pow = false;
  bool found_mul = false;
  
  // 遍历输出的跟踪值
  for (const auto& output : outputs) {
    // 获取函数名称和尺寸信息
    const auto& fn = std::get<0>(output);
    const auto& sizes = std::get<1>(output);

    // 检查是否找到了 "test" 函数调用
    if (fn == "test") {
      found_test = true;
      // 断言尺寸信息的有效性
      TORCH_CHECK(sizes.empty());
    } else if (fn == "aten::pow") {
      found_pow = true;
      // 断言尺寸信息的有效性
      TORCH_CHECK(sizes.size() == 1);
      TORCH_CHECK(sizes[0] == std::vector<int64_t>({1, 2, 3}));
    } else if (fn == "aten::mul") {
      found_mul = true;
      // 断言尺寸信息的有效性
      TORCH_CHECK(sizes.size() == 1);
      TORCH_CHECK(sizes[0] == std::vector<int64_t>({1, 2, 3}));
    }
  }
  // 断言是否找到了所有预期的函数调用
  TORCH_CHECK(found_test);
  TORCH_CHECK(found_pow);
  TORCH_CHECK(found_mul);
}

static bool bad_scope = false;
template <RecordScope scope, size_t* cnt>
// 定义一个模板函数，接收记录作用域和计数器作为参数

std::unique_ptr<at::ObserverContext> checkScopeCallback(
    const at::RecordFunction& fn) {
  // 检查传入的记录函数是否匹配指定的作用域
  if (fn.scope() == scope) {
    // 如果匹配，增加计数器的值
    ++(*cnt);
  } else {
    // 如果不匹配，设置全局变量表示存在错误的作用域
    bad_scope = true;
  }
  // 返回空指针
  return nullptr;
}

template <RecordScope scope, size_t* cnt>
// 定义一个模板函数，接收记录作用域和计数器作为参数

void pushScopedCallback() {
  // 将 checkScopeCallback 函数作为全局回调添加到全局回调函数列表中
  at::addGlobalCallback(
      at::RecordFunctionCallback(checkScopeCallback<scope, cnt>)
          .scopes({scope}));
}

// 以下静态变量不能在函数局部声明，因为这样会阻止它们在 C++17 之前被用作模板参数。

// 定义几个全局变量用于记录函数调用的次数
static size_t fun_cnt;
static size_t ts_fun_cnt;
static size_t user_scope_cnt;

void checkScopeCallbacks() {
  // 定义静态布尔变量，用于跟踪是否找到特定作用域的记录函数
  static bool found_function_scope;
  static bool found_method_scope;
  static bool found_user_scope;
  found_function_scope = false;
  found_method_scope = false;
  found_user_scope = false;

  // 添加全局回调函数，用 lambda 表达式实现对特定作用域的记录函数的跟踪
  at::addGlobalCallback(at::RecordFunctionCallback(
      [](const at::RecordFunction& fn) -> std::unique_ptr<at::ObserverContext> {
        if (fn.scope() == at::RecordScope::FUNCTION &&
            std::string(fn.name()) == "test_function") {
          found_function_scope = true;
        }
        if (fn.scope() == at::RecordScope::TORCHSCRIPT_FUNCTION &&
            std::string(fn.name()) == "test_method") {
          found_method_scope = true;
        }
        if (fn.scope() == at::RecordScope::USER_SCOPE &&
            std::string(fn.name()) == "test_user_scope") {
          found_user_scope = true;
        }
        return nullptr;
      }));

  // 初始化全局变量，用于记录特定作用域记录函数的调用次数
  bad_scope = false;
  fun_cnt = 0;
  pushScopedCallback<at::RecordScope::FUNCTION, &fun_cnt>();
  ts_fun_cnt = 0;
  pushScopedCallback<at::RecordScope::TORCHSCRIPT_FUNCTION, &ts_fun_cnt>();
  user_scope_cnt = 0;
  pushScopedCallback<at::RecordScope::USER_SCOPE, &user_scope_cnt>();

  // 检查是否存在全局回调函数
  TORCH_CHECK(at::hasCallbacks());

  {
    RECORD_TORCHSCRIPT_FUNCTION("test_method", {});
    { RECORD_FUNCTION("test_function", {}); }
    { RECORD_USER_SCOPE("test_user_scope"); }
  }

  // 断言检查，确保没有错误的作用域记录、每个作用域记录函数调用了一次
  TORCH_CHECK(!bad_scope);
  TORCH_CHECK(fun_cnt == 1);
  TORCH_CHECK(ts_fun_cnt == 1);
  TORCH_CHECK(user_scope_cnt == 1);

  // 断言检查，确保找到了特定作用域的记录函数
  TORCH_CHECK(found_function_scope);
  TORCH_CHECK(found_method_scope);
  TORCH_CHECK(found_user_scope);
}

// 定义几个全局变量，用于存储跟踪的输入和输出值以及输入和输出的名称
static TracedTestValues traced_inputs;
static TracedTestValues traced_outputs;
static std::unordered_set<std::string> ts_input_names;
static std::unordered_set<std::string> ts_output_names;

std::unique_ptr<at::ObserverContext> tracedInputsCallback(
    const RecordFunction& fn) {
  // 如果记录函数的作用域是 FUNCTION
  if (fn.scope() == RecordScope::FUNCTION) {
    // 获取输入，并记录其尺寸信息
    auto inputs = fn.inputs();
    std::vector<std::vector<int64_t>> sizes;
    for (const auto& input : inputs) {
      if (input.isTensor()) {
        sizes.push_back(input.toTensor().sizes().vec());
      } else if (input.isScalar()) {
        // 如果输入是标量，将空的尺寸向量加入列表
        sizes.push_back(std::vector<int64_t>());
      }
    }
    # 如果函数的作用域是 `FunctionScope::TORCHSCRIPT_FUNCTION`，则执行以下操作
    if (fn.scope() == FunctionScope::TORCHSCRIPT_FUNCTION) {
        # 将函数名和 sizes 组成的元组推入 traced_inputs 容器的末尾
        traced_inputs.push_back(std::make_tuple(fn.name(), sizes));
    } else if (fn.scope() == RecordScope::TORCHSCRIPT_FUNCTION) {
        # 如果函数的作用域是 `RecordScope::TORCHSCRIPT_FUNCTION`，则将函数名插入 ts_input_names 集合中
        ts_input_names.insert(fn.name());
    }
    # 返回空指针
    return nullptr;
TEST(RecordFunctionTest, SampledCallbacks) {
  // 禁用方法调用的内联优化
  GraphOptimizerEnabledGuard opt_guard(false);

  // 设置采样回调函数
  sampled_cb_ctr = 0;
  auto setup_sampled_callback = [](double sampling_prob) {
    addGlobalCallback(RecordFunctionCallback(sampledCallback)
                          .needsInputs(false)
                          .needsOutputs(false)
                          .samplingProb(sampling_prob));
  };

  // 设置采样概率为 0.5
  setup_sampled_callback(0.5);
  auto t = torch::randn({1, 2, 3}, at::kCPU);
  invokeTestRecordFunction(t);
  // 检查是否调用了采样回调函数
  ASSERT_GT(sampled_cb_ctr, 0);

  // 测试非采样回调
  non_sampled_cb_ctr = 0;
  auto setup_non_sampled_callback = []() {
    addGlobalCallback(RecordFunctionCallback(nonSampledCallback)
                          .needsInputs(false)
                          .needsOutputs(false)
                          .samplingProb(0.0));
  };

  setup_non_sampled_callback();
  t = torch::randn({1, 2, 3}, at::kCPU);
  invokeTestRecordFunction(t);
  // 检查是否未调用非采样回调函数
  ASSERT_EQ(non_sampled_cb_ctr, 0);
}


注释：

// RecordFunctionTest 的测试用例，用于测试采样和非采样的回调函数
TEST(RecordFunctionTest, SampledCallbacks) {
  // 禁用方法调用的内联优化
  GraphOptimizerEnabledGuard opt_guard(false);

  // 设置采样回调函数计数器为 0
  sampled_cb_ctr = 0;

  // 定义设置采样回调函数的 Lambda 函数
  auto setup_sampled_callback = [](double sampling_prob) {
    // 将 sampledCallback 注册为全局回调函数
    addGlobalCallback(RecordFunctionCallback(sampledCallback)
                          .needsInputs(false)
                          .needsOutputs(false)
                          .samplingProb(sampling_prob));
  };

  // 设置采样概率为 0.5 的采样回调函数
  setup_sampled_callback(0.5);
  
  // 生成一个随机张量并调用测试记录函数
  auto t = torch::randn({1, 2, 3}, at::kCPU);
  invokeTestRecordFunction(t);
  
  // 断言采样回调函数的计数大于 0，即已被调用
  ASSERT_GT(sampled_cb_ctr, 0);

  // 测试非采样回调函数
  non_sampled_cb_ctr = 0;
  
  // 定义设置非采样回调函数的 Lambda 函数
  auto setup_non_sampled_callback = []() {
    // 将 nonSampledCallback 注册为全局回调函数，但设置采样概率为 0.0，即不会被采样
    addGlobalCallback(RecordFunctionCallback(nonSampledCallback)
                          .needsInputs(false)
                          .needsOutputs(false)
                          .samplingProb(0.0));
  };

  // 设置非采样回调函数
  setup_non_sampled_callback();
  
  // 再次生成一个随机张量并调用测试记录函数
  t = torch::randn({1, 2, 3}, at::kCPU);
  invokeTestRecordFunction(t);
  
  // 断言非采样回调函数的计数为 0，即未被调用
  ASSERT_EQ(non_sampled_cb_ctr, 0);
}
  // 调用 addGlobalCallback 函数，将 RecordFunctionCallback 对象作为全局回调添加
  return addGlobalCallback(
      RecordFunctionCallback(sampledCallback).samplingProb(sampling_prob));
};

// 添加 RecordFunctionCallback 对象作为全局回调，但不设置采样概率
addGlobalCallback(RecordFunctionCallback(nonSampledCallback));

// 设置一个采样概率为 0.5 的采样回调，并返回其句柄
auto handle = setup_sampled_callback(0.5);

// 定义并运行一个测试函数，生成一个形状为 {1, 2, 3} 的 CPU 上的随机张量 t
auto run_test_function = []() {
  auto t = torch::randn({1, 2, 3}, at::kCPU);
  // 循环调用 invokeTestRecordFunction 函数 1000 次
  for (auto k = 0; k < 1000; k++) {
    invokeTestRecordFunction(t);
  }
};

// 运行测试函数，期望非采样回调计数器为 1000
run_test_function();
TORCH_CHECK(non_sampled_cb_ctr == 1000);
// 期望采样回调计数器大于 0 且小于 1000
TORCH_CHECK(sampled_cb_ctr > 0 && sampled_cb_ctr < 1000);

// 重置采样回调计数器
sampled_cb_ctr = 0;
// 移除之前设置的回调句柄
removeCallback(handle);
// 重新设置一个采样概率为 0.0 的采样回调句柄
handle = setup_sampled_callback(0.0);
// 再次运行测试函数
run_test_function();

// 期望非采样回调计数器为 2000
TORCH_CHECK(non_sampled_cb_ctr == 2000);
// 期望采样回调计数器为 0
TORCH_CHECK(sampled_cb_ctr == 0);

// 重置采样回调计数器
sampled_cb_ctr = 0;
// 移除之前设置的回调句柄
removeCallback(handle);
// 设置一个采样概率为 1.0 的采样回调句柄
handle = setup_sampled_callback(1.0);
// 再次运行测试函数
run_test_function();

// 期望非采样回调计数器为 3000
TORCH_CHECK(non_sampled_cb_ctr == 3000);
// 期望采样回调计数器为 1000
TORCH_CHECK(sampled_cb_ctr == 1000);

// 清除所有回调
clearCallbacks();

// 测试回调函数的作用域
checkScopeCallbacks();
// 再次清除所有回调
clearCallbacks();
}

// 定义单元测试 RecordFunctionTest.RecordFunctionGuard
TEST(RecordFunctionTest, RecordFunctionGuard) {
  // 禁用方法调用的内联化
  GraphOptimizerEnabledGuard opt_guard(false);

  // 静态变量：存储函数名列表和互斥锁
  static std::vector<std::string> fn_names;
  static std::mutex guard_mtx;

  // 检查记录函数的守卫
  addGlobalCallback(RecordFunctionCallback(
      [](const RecordFunction& fn) -> std::unique_ptr<at::ObserverContext> {
        // 锁定互斥锁保护共享数据
        std::lock_guard<std::mutex> lock(guard_mtx);
        // 将函数名添加到列表中
        // NOLINTNEXTLINE(modernize-use-emplace)
        fn_names.push_back(fn.name());
        return nullptr;
      }));

  {
    // 创建 RecordFunctionGuard 对象 g1，禁用记录函数
    RecordFunctionGuard g1(false);
    {
      // 记录用户范围 "A"
      RECORD_USER_SCOPE("A");
      {
        // 创建 RecordFunctionGuard 对象 g2，启用记录函数
        RecordFunctionGuard g2(true);
        // 记录用户范围 "B"
        RECORD_USER_SCOPE("B");
        {
          // 创建 DisableRecordFunctionGuard 对象 g3
          DisableRecordFunctionGuard g3;
          // 记录用户范围 "C"
          RECORD_USER_SCOPE("C");
        }
      }
      // 记录用户范围 "D"
      { RECORD_USER_SCOPE("D"); }
    }
  }

  // 检查函数名列表长度为 1
  TORCH_CHECK(fn_names.size() == 1);
  // 检查函数名列表第一个元素为 "B"
  TORCH_CHECK(fn_names[0] == "B");

  // 清空回调函数
  clearCallbacks();
}

// 静态变量：存储回调函数的 id 列表
static std::vector<size_t> ids;

// 模板函数：添加和移除回调函数，并返回句柄
template <size_t id>
auto add_remove_test_add_cb() {
  return addGlobalCallback(RecordFunctionCallback(
      [](const RecordFunction& fn) -> std::unique_ptr<at::ObserverContext> {
        // 将回调函数的 id 添加到列表中
        ids.push_back(id);
        return nullptr;
      }));
}

// 定义单元测试 RecordFunctionTest.Callbacks
TEST(RecordFunctionTest, Callbacks) {
  // 禁用方法调用的内联化
  GraphOptimizerEnabledGuard opt_guard(false);

  // 添加回调函数 h1, h3，并存储其句柄
  auto h1 = add_remove_test_add_cb<1>();
  add_remove_test_add_cb<2>();
  auto h3 = add_remove_test_add_cb<3>();

  // 记录用户范围 "test"
  { RECORD_USER_SCOPE("test"); }

  // 检查 ids 列表长度为 3
  TORCH_CHECK(ids.size() == 3);
  // 检查 ids 列表包含 1
  TORCH_CHECK(std::find(ids.begin(), ids.end(), 1) != ids.end());
  // 检查 ids 列表包含 2
  TORCH_CHECK(std::find(ids.begin(), ids.end(), 2) != ids.end());
  // 检查 ids 列表包含 3
  TORCH_CHECK(std::find(ids.begin(), ids.end(), 3) != ids.end());

  // 清空 ids 列表
  ids.clear();
  // 移除回调函数 h1

  removeCallback(h1);

  // 记录用户范围 "test"
  { RECORD_USER_SCOPE("test"); }

  // 检查 ids 列表长度为 2
  TORCH_CHECK(ids.size() == 2);
  // 检查 ids 列表包含 2
  TORCH_CHECK(std::find(ids.begin(), ids.end(), 2) != ids.end());
  // 检查 ids 列表包含 3
  TORCH_CHECK(std::find(ids.begin(), ids.end(), 3) != ids.end());

  // 清空 ids 列表
  ids.clear();
  // 移除回调函数 h3

  removeCallback(h3);

  // 记录用户范围 "test"
  { RECORD_USER_SCOPE("test"); }

  // 检查 ids 列表长度为 1
  TORCH_CHECK(ids.size() == 1);
  // 检查 ids 列表包含 2
  TORCH_CHECK(std::find(ids.begin(), ids.end(), 2) != ids.end());

  // 清空回调函数
  clearCallbacks();

  // 线程本地 / 全局回调函数

  // 清空 ids 列表
  ids.clear();
  // 添加回调函数，并存储其 id

  add_remove_test_add_cb<1>();

  // 记录用户范围 "test"
  { RECORD_USER_SCOPE("test"); }

  // 检查 ids 列表长度为 1
  TORCH_CHECK(ids.size() == 1);
  // 检查 ids 列表第一个元素为 1
  TORCH_CHECK(ids[0] == 1);
  // 清空 ids 列表
  ids.clear();

  // 创建线程 th
  auto th = std::thread([]() {
    // 添加线程本地回调函数
    addThreadLocalCallback(RecordFunctionCallback(
        [](const RecordFunction& fn) -> std::unique_ptr<at::ObserverContext> {
          // 将回调函数的 id 添加到列表中
          ids.push_back(2);
          return nullptr;
        }));
    { RECORD_USER_SCOPE("test_thread"); }


// 记录用户作用域为 "test_thread"
{ RECORD_USER_SCOPE("test_thread"); }



  });


// 等待线程结束
  th.join();



  TORCH_CHECK(ids.size() == 2);
  TORCH_CHECK(std::find(ids.begin(), ids.end(), 1) != ids.end());
  TORCH_CHECK(std::find(ids.begin(), ids.end(), 2) != ids.end());
  ids.clear();


// 检查 ids 的大小为 2，以及是否包含值为 1 和 2 的元素，然后清空 ids
  TORCH_CHECK(ids.size() == 2);
  TORCH_CHECK(std::find(ids.begin(), ids.end(), 1) != ids.end());
  TORCH_CHECK(std::find(ids.begin(), ids.end(), 2) != ids.end());
  ids.clear();



  { RECORD_USER_SCOPE("test"); }


// 记录用户作用域为 "test"
  { RECORD_USER_SCOPE("test"); }



  TORCH_CHECK(ids.size() == 1);
  TORCH_CHECK(ids[0] == 1);
  ids.clear();


// 检查 ids 的大小为 1，并且第一个元素的值为 1，然后清空 ids
  TORCH_CHECK(ids.size() == 1);
  TORCH_CHECK(ids[0] == 1);
  ids.clear();



  clearCallbacks();


// 清空回调函数
  clearCallbacks();



  // START: thread local / global context check callbacks


// 开始线程局部和全局上下文检查回调
  // START: thread local / global context check callbacks



  struct TestContext : public ObserverContext {
    int a{0};
    std::string b;
  };
  ids.clear();


// 定义 TestContext 结构体，继承自 ObserverContext
// 初始化 ids 并清空
  struct TestContext : public ObserverContext {
    int a{0};
    std::string b;
  };
  ids.clear();



  { // START: global test


// 开始全局测试
  { // START: global test



    { RECORD_USER_SCOPE("test"); }


// 记录用户作用域为 "test"
    { RECORD_USER_SCOPE("test"); }



    TORCH_CHECK(ids.size() == 1);
    TORCH_CHECK(ids[0] == 1);
    ids.clear();


// 检查 ids 的大小为 1，并且第一个元素的值为 1，然后清空 ids
    TORCH_CHECK(ids.size() == 1);
    TORCH_CHECK(ids[0] == 1);
    ids.clear();



  } // END: global test


// 结束全局测试
  } // END: global test



  { // START: thread local test


// 开始线程局部测试
  { // START: thread local test



    auto ctx_th = std::thread([]() {
      const std::string test_str = "test thread str";


// 创建一个新线程，其中定义了一个字符串 test_str
    auto ctx_th = std::thread([]() {
      const std::string test_str = "test thread str";



      // Will call both global and thread local callbacks.
      { RECORD_USER_SCOPE("test_thread"); }


// 记录用户作用域为 "test_thread"
      { RECORD_USER_SCOPE("test_thread"); }



    });
    ctx_th.join();
    TORCH_CHECK(ids.size() == 2);
    TORCH_CHECK(std::find(ids.begin(), ids.end(), 1) != ids.end());
    TORCH_CHECK(std::find(ids.begin(), ids.end(), 2) != ids.end());
    ids.clear();
  } // END: thread local test


// 等待线程结束
    });
    ctx_th.join();
// 检查 ids 的大小为 2，以及是否包含值为 1 和 2 的元素，然后清空 ids
    TORCH_CHECK(ids.size() == 2);
    TORCH_CHECK(std::find(ids.begin(), ids.end(), 1) != ids.end());
    TORCH_CHECK(std::find(ids.begin(), ids.end(), 2) != ids.end());
    ids.clear();
  } // END: thread local test



  clearCallbacks();


// 清空回调函数
  clearCallbacks();
TEST(RecordFunctionTest, ShouldRun) {
  // disabling the inlining of method calls
  // 创建一个作用域，禁用方法调用的内联优化
  GraphOptimizerEnabledGuard opt_guard(false);

  static bool ran = false;
  // 添加全局回调函数，记录函数运行时的回调
  auto handle = addGlobalCallback(RecordFunctionCallback(
      [](const RecordFunction& fn) -> std::unique_ptr<at::ObserverContext> {
        // 设置标志表明函数已经运行过
        ran = true;
        return nullptr;
      }));

  { RECORD_USER_SCOPE("test"); }  // 记录用户自定义作用域 "test"

  // 断言第一次运行已发生
  EXPECT_TRUE(ran) << "first run didn't happen";
  ran = false;

  // 禁用回调函数
  disableCallback(handle);

  { RECORD_USER_SCOPE("test"); }  // 记录用户自定义作用域 "test"

  // 断言第二次运行未发生
  EXPECT_FALSE(ran) << "second run happened but shouldn't have";
  ran = false;

  // 重新启用回调函数
  reenableCallback(handle);

  { RECORD_USER_SCOPE("test"); }  // 记录用户自定义作用域 "test"

  // 断言重新启用后运行发生
  EXPECT_TRUE(ran) << "run after re-enable didn't happen";
  ran = false;

  clearCallbacks();  // 清除所有回调函数
}

TEST(RecordFunctionTest, Basic) {
  // disabling the inlining of method calls
  // 创建一个作用域，禁用方法调用的内联优化
  GraphOptimizerEnabledGuard opt_guard(false);

  static std::string recorded_op;
  static bool has_ids = false;

  // 在新线程中测试 TLS 回调传播
  std::thread t([]() {
    RecordFunctionGuard enable_rec_fn;  // 开启记录函数
    // 添加线程本地回调函数
    auto handle = addThreadLocalCallback(RecordFunctionCallback(
        [](const RecordFunction& fn) -> std::unique_ptr<at::ObserverContext> {
          // 记录操作的名称
          recorded_op = fn.name();
          return nullptr;
        }));
    ThreadLocalState state;
    std::thread t_child([state]() {
      ThreadLocalStateGuard g_tls(state);  // 管理线程本地状态
      RECORD_USER_SCOPE("test_in_thread");  // 记录用户自定义作用域 "test_in_thread"
    });
    t_child.join();  // 等待子线程结束
    // 断言记录的操作名称为 "test_in_thread"
    EXPECT_EQ(recorded_op, "test_in_thread");
    removeCallback(handle);  // 移除线程本地回调函数
  });
  t.join();  // 等待线程结束
  clearCallbacks();  // 清除所有回调函数

  // 测试设置标识符
  addGlobalCallback(
      RecordFunctionCallback(
          [](const RecordFunction& fn) -> std::unique_ptr<at::ObserverContext> {
            // 检查是否需要标识符
            has_ids = fn.handle() > 0;
            return nullptr;
          })
          .needsIds(true));  // 设置需要标识符

  { RECORD_USER_SCOPE("test"); }  // 记录用户自定义作用域 "test"
  
  TORCH_CHECK(has_ids);  // 使用 Torch 的断言检查是否有标识符
  clearCallbacks();  // 清除所有回调函数
  has_ids = false;

  // 添加全局回调函数，不需要标识符
  addGlobalCallback(RecordFunctionCallback(
      [](const RecordFunction& fn) -> std::unique_ptr<at::ObserverContext> {
        // 检查是否需要标识符
        has_ids = fn.handle() > 0;
        return nullptr;
      }));

  { RECORD_USER_SCOPE("test"); }  // 记录用户自定义作用域 "test"
  
  TORCH_CHECK(!has_ids);  // 使用 Torch 的断言检查是否无标识符
  clearCallbacks();  // 清除所有回调函数
}
TEST(RecordFunctionTest, OperatorNameOverload) {
  // 静态变量，存储运算符名称集合
  static std::set<std::string> operator_names;
  
  // 添加全局回调函数来记录函数调用信息
  at::addGlobalCallback(at::RecordFunctionCallback(
                            [](const at::RecordFunction& fn)
                                -> std::unique_ptr<at::ObserverContext> {
                              // 获取函数的操作符名称
                              std::optional<c10::OperatorName> op_name =
                                  fn.operator_name();
                              // 如果存在操作符名称，则插入集合中
                              if (op_name.has_value()) {
                                operator_names.insert(c10::toString(*op_name));
                              } else {
                                // 否则插入 "No Operator Name" 字符串
                                operator_names.insert("No Operator Name");
                              }
                              return nullptr;
                            })
                            .scopes({at::RecordScope::FUNCTION}));
  
  // 创建一个张量并设置梯度计算为不需求
  auto t = torch::randn({1, 2, 3}, at::kCPU);
  t.set_requires_grad(false);
  
  // 对张量进行平方操作
  auto t2 = t.pow(2);

  // 清除所有回调函数
  at::clearCallbacks();

  // 断言：期望没有 "No Operator Name" 的记录存在
  EXPECT_TRUE(operator_names.count("No Operator Name") == 0)
      << "Expected that all traced operators had an associated OperatorName object";
  
  // 断言：期望 "aten::randn" 的操作被记录了一次
  EXPECT_TRUE(operator_names.count("aten::randn") == 1)
      << "Expected aten::randn to have been called and recorded, but it was not";
  
  // 断言：期望 "aten::pow.Tensor_Scalar" 的操作被记录了一次
  EXPECT_TRUE(operator_names.count("aten::pow.Tensor_Scalar") == 1)
      << "Expected aten::pow.Tensor_Scalar to have been called and recorded, but it was not";
}

class TestThreadLocalDebugInfo : public c10::DebugInfoBase {
 public:
  // 获取模型ID
  int getModelId() const {
    return model_id_;
  }

  // 设置模型ID
  void setModelId(int model_id) {
    model_id_ = model_id;
  }

  // 虚析构函数
  virtual ~TestThreadLocalDebugInfo() override {}

 private:
  // 模型ID，默认为0
  int model_id_ = 0;
};

// 检查调试信息
void checkDebugInfo(c10::DebugInfoKind kind, int model_id) {
  // 获取指定类型的线程局部调试信息
  auto* debug_info = c10::ThreadLocalDebugInfo::get(kind);
  TORCH_CHECK(debug_info != nullptr);
  // 尝试将其转换为 TestThreadLocalDebugInfo 类型
  auto* test_debug_info = dynamic_cast<TestThreadLocalDebugInfo*>(debug_info);
  TORCH_CHECK(test_debug_info != nullptr);
  // 断言：检查模型ID是否符合预期
  TORCH_CHECK(test_debug_info->getModelId() == model_id);
}

TEST(ThreadLocalDebugInfoTest, Basic) {
  // 静态原子布尔值，标识测试是否完成
  static std::atomic<bool> done{false};

  // 断言：检查指定类型的线程局部调试信息是否为nullptr
  TORCH_CHECK(
      c10::ThreadLocalDebugInfo::get(c10::DebugInfoKind::TEST_INFO) == nullptr);
  
  // 创建一个共享的调试信息对象，并设置模型ID为42
  auto debug_info = std::make_shared<TestThreadLocalDebugInfo>();
  debug_info->setModelId(42);
  
  {
    // 创建一个调试信息卫士，保护调试信息在作用域内有效
    c10::DebugInfoGuard guard(c10::DebugInfoKind::TEST_INFO, debug_info);
    // 检查调试信息是否符合预期
    checkDebugInfo(c10::DebugInfoKind::TEST_INFO, 42);
  }

  // 检查线程局部调试信息是否在fork调用中传播
  TORCH_CHECK(
      c10::ThreadLocalDebugInfo::get(c10::DebugInfoKind::TEST_INFO) == nullptr);
  
  {
    // 再次创建一个调试信息卫士，并在新线程中运行lambda表达式
    c10::DebugInfoGuard guard(c10::DebugInfoKind::TEST_INFO, debug_info);
    at::launch([]() {
      // 在新线程中检查调试信息是否符合预期
      checkDebugInfo(c10::DebugInfoKind::TEST_INFO, 42);
      // 标记测试已完成
      done = true;
      });
  }
}
  });
  }
  while (!done) {
  }

  // 检查线程本地调试信息在反向传播过程中是否传播
  TORCH_CHECK(
      c10::ThreadLocalDebugInfo::get(c10::DebugInfoKind::TEST_INFO) == nullptr);
  done = false;
  // 添加全局回调函数，用于记录函数调用
  auto handle = addGlobalCallback(RecordFunctionCallback(
      [](const RecordFunction&) -> std::unique_ptr<at::ObserverContext> {
        // 检查调试信息种类为 TEST_INFO 是否存在并设置为 42
        checkDebugInfo(c10::DebugInfoKind::TEST_INFO, 42);
        // 设置完成标志为 true
        done = true;
        return nullptr;
      }));
  {
    // 使用调试信息守卫，设置调试信息为 debug_info
    c10::DebugInfoGuard guard(c10::DebugInfoKind::TEST_INFO, debug_info);
    // 创建一个形状为 {1, 2, 3} 的张量并分配到 CPU
    auto t = torch::randn({1, 2, 3}, at::kCPU);
    // 设置张量需要梯度计算
    t.set_requires_grad(true);
    // 对张量进行平方操作
    auto t2 = t.pow(2);
    // 对 t2 进行反向传播，保持内存格式为 Preserve
    t2.backward(torch::ones_like(t2, at::MemoryFormat::Preserve));
  }
  // 移除回调函数
  removeCallback(handle);
  // 检查 done 标志位是否为 true
  TORCH_CHECK(done);

  // 检查嵌套调试信息
  TORCH_CHECK(
      c10::ThreadLocalDebugInfo::get(c10::DebugInfoKind::TEST_INFO) == nullptr);
  {
    // 使用调试信息守卫，设置调试信息为 debug_info
    c10::DebugInfoGuard guard(c10::DebugInfoKind::TEST_INFO, debug_info);
    {
      // 检查调试信息种类为 TEST_INFO 是否存在并设置为 42
      checkDebugInfo(c10::DebugInfoKind::TEST_INFO, 42);
      {
        // 创建一个共享的 TestThreadLocalDebugInfo 对象，并设置模型 ID 为 314
        auto debug_info = std::make_shared<TestThreadLocalDebugInfo>();
        debug_info->setModelId(314);
        // 使用调试信息守卫，设置调试信息种类为 TEST_INFO_2
        c10::DebugInfoGuard guard(c10::DebugInfoKind::TEST_INFO_2, debug_info);
        {
          // 检查调试信息种类为 TEST_INFO 是否存在并设置为 42
          checkDebugInfo(c10::DebugInfoKind::TEST_INFO, 42);
          // 检查调试信息种类为 TEST_INFO_2 是否存在并设置为 314
          checkDebugInfo(c10::DebugInfoKind::TEST_INFO_2, 314);
          done = false;
          // 在新线程中执行 lambda 表达式
          at::launch([]() {
            // 检查调试信息种类为 TEST_INFO 是否存在并设置为 42
            checkDebugInfo(c10::DebugInfoKind::TEST_INFO, 42);
            // 检查调试信息种类为 TEST_INFO_2 是否存在并设置为 314
            checkDebugInfo(c10::DebugInfoKind::TEST_INFO_2, 314);
            // 设置完成标志为 true
            done = true;
          });
          while (!done) {
          }
        }
      }
    }
  }
}

# 定义一个名为 `TestSymIntArrayRef` 的测试案例
TEST(TestSymIntArrayRef, BasicConversion) {
  # 定义常量 X, Y, Z 分别为 2, 4, 5
  const size_t X = 2, Y = 4, Z = 5;
  # 创建一个包含元素 2, 4, 5 的整型向量 tgt_size_v
  std::vector<int64_t> tgt_size_v{2, 4, 5};
  # 创建一个包含 SymInt 对象的向量 tgt_size，用于测试
  std::vector<c10::SymInt> tgt_size({SymInt(X), SymInt(Y), SymInt(Z)});
  # 生成一个形状为 [1, 4, 1] 的 CPU 上的随机张量 a
  auto a = at::randn({1, 4, 1}, at::kCPU);
  # 使用 SymInt 扩展张量 a，形状为 tgt_size，得到张量 b
  auto b = a.expand_symint(tgt_size);
  # 使用普通整数值扩展张量 a，形状为 tgt_size_v，得到张量 c
  auto c = a.expand(tgt_size_v);
  # 断言张量 b 和 c 在数值上全部接近
  ASSERT_TRUE(torch::allclose(b, c));
}

# 定义一个名为 `TestSymInt` 的测试案例，用于测试 SymInt 类的功能
TEST(TestSymInt, NarrowCopyWithSymbolicInt) {
  # 定义静态常量 LENGTH 为 5
  static const size_t LENGTH = 5;
  # 生成一个形状为 [10] 的 CPU 上的随机张量 a
  auto a = at::randn({10}, at::kCPU);
  # 创建一个 SymInt 对象 si，其长度为 LENGTH
  c10::SymInt si(LENGTH);
  # 使用 SymInt 对象 si 对张量 a 进行窄复制，得到张量 b
  auto b = a.narrow_copy_symint(0, 0, si);
  # 使用普通整数值对张量 a 进行窄复制，范围为 [0, LENGTH)，得到张量 c
  auto c = a.narrow(0, 0, LENGTH);
  # 断言张量 b 和 c 在数值上全部接近
  ASSERT_TRUE(torch::allclose(b, c));
}

# 定义一个名为 `TestSymInt` 的测试案例，用于测试 SymInt 类的功能
TEST(TestSymInt, NarrowCopy) {
  # 定义静态常量 LENGTH 为 5
  static const size_t LENGTH = 5;
  # 生成一个形状为 [10] 的 CPU 上的随机张量 a
  auto a = at::randn({10}, at::kCPU);
  # 使用普通整数值对张量 a 进行窄复制，范围为 [0, LENGTH)，得到张量 b
  auto b = a.narrow_copy(0, 0, LENGTH);
  # 使用普通整数值对张量 a 进行窄复制，范围为 [0, LENGTH)，得到张量 c
  auto c = a.narrow(0, 0, LENGTH);
  # 断言张量 b 和 c 在数值上全部接近
  ASSERT_TRUE(torch::allclose(b, c));
}

# 定义一个名为 `TestSymInt` 的测试案例，用于测试 SymInt 类的加法运算
TEST(TestSymInt, AddSymbolicInt) {
  # 创建 SymInt 对象 a 和 b，分别为 5 和 3
  c10::SymInt a(5);
  c10::SymInt b(3);
  # 断言 a 加 b 的结果期望是 8
  ASSERT_TRUE((a + b).expect_int() == 8);
}

# 定义一个名为 `FallbackGraphsTest` 的测试案例，用于测试图函数和性能分析
TEST(FallbackGraphsTest, Basic) {
  # 生成形状为 [1] 的 CPU 上的随机张量 x 和 y
  auto x = at::randn({1}, at::kCPU);
  auto y = at::randn({1}, at::kCPU);
  # 创建一个堆栈 stack，包含 x 和 y 的克隆张量
  auto stack = createStack({x.clone(), y.clone()});

  # 定义一个包含 IR 字符串的图 graph_string
  auto graph_string = R"IR(
    graph(%0 : Float(1),
          %1 : Float(1)):
      %2 : Tensor = aten::mul(%0, %1)
      %3 : Tensor = aten::mul(%2, %0)
      return (%3))IR";
  # 解析 IR 字符串生成一个图 graph
  auto graph = std::make_shared<Graph>();
  torch::jit::parseIR(graph_string, graph.get());

  {
    # 创建代码对象 code，用于解释执行图 graph
    Code code(graph, "");
    # 创建解释器状态对象 interpreter，传入代码对象 code
    InterpreterState interpreter{code};
    # 运行解释器，对堆栈进行计算
    interpreter.run(stack);
  }
  
  # 弹出堆栈中的结果张量 et
  at::Tensor et;
  pop(stack, et);
  # 提取张量 et 的浮点数值 ef
  float ef = et.item<float>();

  {
    # 启用性能分析器
    EnableProfilingGuard epg;
    # 创建图函数对象 f，用于执行名为 "fallbackGraphs" 的图
    GraphFunction f("fallbackGraphs", graph, nullptr);
    # 迭代执行性能分析次数加一次
    for (size_t i = 0; i < getNumProfiledRuns() + 1; i++) {
      # 压入堆栈 x 和 y 的克隆张量
      stack.emplace_back(x.clone());
      stack.emplace_back(y.clone());
      if (i == getNumProfiledRuns()) {
        # 当达到性能分析次数时，修改一个经过分析的图
        auto opt_graph = lastExecutedOptimizedGraph();
        # 移除优化图的分析计数器
        ProfilingRecord::removeProfileCounter(opt_graph->block());
        # 使用备用图替换块，输入为优化图的输入
        replaceBlockWithFallbackGraph(opt_graph->block(), opt_graph->inputs());
        auto it = opt_graph->block()->nodes().begin();
        # 断言优化图的第一个节点为 prim::FallbackGraph 类型
        ASSERT_EQ(it->kind(), prim::FallbackGraph);
        auto fallback = *it++;
        # 断言无更多节点
        ASSERT_EQ(it, opt_graph->block()->nodes().end());
        # 断言 fallback 包含子图属性
        ASSERT_TRUE(fallback->hasAttribute(attr::Subgraph));
        # 使用 FileCheck 验证子图包含期望的运算
        testing::FileCheck()
            .check("Tensor = aten::mul")
            ->check("Tensor = aten::mul")
            ->run(*fallback->g(attr::Subgraph));
      }
      # 运行图函数 f，对堆栈进行计算
      f.run(stack);
      # 弹出堆栈中的结果张量 at
      at::Tensor at;
      pop(stack, at);
      # 提取张量 at 的浮点数值 af
      float af = at.item<float>();
      # 断言 af 等于 ef
      ASSERT_EQ(af, ef);
    }

    # 获取最后执行的优化图
    auto opt_graph = lastExecutedOptimizedGraph();
    # 使用 FileCheck 验证优化图包含期望的调用函数
    testing::FileCheck()
        .check("(Tensor) = prim::CallFunction")
        ->run(*opt_graph);
  }
}

# 待修复的测试案例，目前不运行并且有问题
// TODO this test wasn't running and is broken.
// 定义一个单元测试，用于测试 AutogradProfiler 的基本功能
TEST(AutogradProfilerTest, Basic) {
  // 定义常量：批处理大小、输入大小、序列长度
  constexpr int batch_size = 4;
  constexpr int input_size = 256;
  constexpr int seq_len = 32;

  // 计算隐藏层大小
  int hidden_size = 2 * input_size;

  // 生成随机输入张量和隐藏状态张量
  auto input = torch::randn({seq_len, batch_size, input_size}, at::kCPU);
  auto hx = torch::randn({batch_size, hidden_size}, at::kCPU);
  auto cx = torch::randn({batch_size, hidden_size}, at::kCPU);

  // 生成随机权重张量
  auto w_ih = t_def(torch::randn({4 * hidden_size, input_size}, at::kCPU));
  auto w_hh = t_def(torch::randn({4 * hidden_size, hidden_size}, at::kCPU));

  // 创建一个字符串流对象 ss
  std::stringstream ss;
  {
    // 在 ss 上记录性能信息
    RecordProfile guard(ss);
    // 执行 100 次 LSTM 操作，记录性能信息
    for (size_t i = 0; i < 100; ++i) {
      std::tie(hx, cx) = lstm(input[0], hx, cx, w_ih, w_hh);
    }
  }

  // 将性能信息转换为字符串
  std::string result = ss.str();
  size_t count = 0;
  // 统计字符串中 "tanh" 出现的次数
  for (size_t pos = 0; (pos = result.find("tanh", pos)) != std::string::npos;
       count++, pos++) {
  }
  // 断言 "tanh" 出现的次数为 200
  ASSERT_EQ(count, 200);
}

// 定义一个单元测试，用于测试 NoneSchemaMatch 的基本功能
TEST(NoneSchemaMatchTest, Basic) {
  // 注册两个操作符到 RegisterOperators 中
  RegisterOperators reg({
      Operator(
          "prim::test_none() -> int?",
          [](Stack& stack) { push(stack, IValue()); },
          aliasAnalysisFromSchema()),
      Operator(
          "prim::is_none(int? a) -> bool",
          [](Stack& stack) {
            IValue a = pop(stack);
            if (a.isNone()) {
              push(stack, true);
            } else {
              push(stack, false);
            }
          },
          aliasAnalysisFromSchema()),
  });

  // 创建一个新的计算图
  auto r = std::make_shared<Graph>();
  auto& g = *r;
  // 在计算图中插入操作符节点
  auto opt_int = g.insert(Symbol::fromQualString("prim::test_none"), {});
  auto out_bool = g.insert(Symbol::fromQualString("prim::is_none"), {opt_int});
  // 注册输出节点
  g.registerOutput(out_bool);
  // 运行常量传播优化
  ConstantPropagation(r);

  // 获取计算图中的节点列表
  auto nodes = r->block()->nodes();
  // 断言常量传播运行后节点数量为 1
  AT_ASSERT(std::distance(nodes.begin(), nodes.end()) == 1);
}

// 定义一个静态变量，用于测试 Pass 的基本功能
static int testPassValue = 0;
void fakePass(std::shared_ptr<Graph>& g) {
  // 增加测试通过值
  testPassValue++;
  return;
}

// 注册一个 Pass 到 PassManager 中
RegisterPass p(fakePass);

// 定义一个单元测试，用于测试 Pass 管理的基本功能
TEST(PassManagementTest, Basic) {
  // 创建一个新的计算图
  std::shared_ptr<Graph> graph = std::make_shared<Graph>();
  // 解析简单的 IR 字符串到计算图中
  parseIR(
      R"IR(
graph(%a):
  return (%a))IR",
      &*graph);

  // 创建输入栈
  std::vector<IValue> stack = {IValue(torch::randn({22}, at::kCPU))};
  
  // 定义一个运行函数，用于执行计算图
  auto run = [&](std::shared_ptr<Graph>& graph, std::vector<IValue> stack) {
    GraphExecutor executor(graph, "");
    executor.run(stack);
    return stack;
  };
  
  // 运行计算图
  run(graph, stack);
  
  // 如果不是简单模式，断言测试通过值非零
  if (!getExecutorMode()) {
    AT_ASSERT(testPassValue);
  }
}

// 定义一个静态函数，用于检查张量类型的形状
static void checkShape(TypePtr typ, std::vector<int64_t> expected) {
  auto ptp = typ->expect<TensorType>();
  // 断言张量形状与预期形状相等
  ASSERT_EQ(ptp->sizes().concrete_sizes().value(), expected);
}

// 定义一个函数重载，用于检查节点的张量类型形状
static void checkShape(
    Node* n,
    std::vector<int64_t> expected,
   `
    // 定义布尔变量 prev，初始值为 true
    bool prev = true) {
  // 根据条件 prev 决定选择输入节点的第一个节点还是节点本身
  auto profile = (prev) ? n->inputs().at(0)->node() : n;
  // 检查 profile 节点的输出类型是否符合预期类型 expected
  checkShape(profile->output()->type(), expected);
}

// 递归函数，用于遍历给定 block 下的所有节点，并统计符合条件的节点数量
void count_(
    Block* block,                                    // 当前处理的基本块指针
    const std::function<bool(Node* n)>& pred,        // 判断节点是否符合条件的函数对象
    size_t& count) {                                 // 符合条件的节点数量的引用
  for (Node* n : block->nodes()) {                   // 遍历当前基本块下的所有节点
    if (pred(n)) {                                   // 如果当前节点符合条件
      count++;                                       // 符合条件节点数量加一
    }

    for (Block* ib : n->blocks()) {                  // 遍历当前节点下的所有子基本块
      count_(ib, pred, count);                       // 递归调用 count_ 函数，处理子基本块
    }
  }
}

// 统计图中符合条件的节点数量
size_t countNodes(
    const std::shared_ptr<Graph>& graph,             // 图的共享指针
    const std::function<bool(Node* n)>& pred) {      // 判断节点是否符合条件的函数对象
  size_t count = 0;                                  // 初始化符合条件的节点数量
  count_(graph->block(), pred, count);               // 调用 count_ 函数，从图的根基本块开始统计
  return count;                                      // 返回符合条件的节点数量
}

// 判断节点是否总是返回 true
bool true_pred(Node* n) {
  return true;
};

// 判断节点是否为循环节点
bool is_loop(Node* n) {
  return n->kind() == prim::Loop;
};

// 测试用例：验证不使用归纳变量时的循环展开
TEST(LoopPeelerTest, NoInductionVariableUse) {
  // 不显式使用归纳变量的函数定义字符串
  static const auto str_func_def = R"JIT(
    def test_peel_n_times():
      sum = 0
      for i in range(10):
        sum += 2
      return sum
    )JIT";

  auto cu = compile(str_func_def);                   // 编译函数定义字符串
  auto& f = toGraphFunction(cu->get_function("test_peel_n_times"));  // 获取函数图
  auto stack = createStack({});                      // 创建执行栈

  // 循环展开一次
  {
    LoopsPeeler peeler(true_pred, 1);                // 创建循环展开器对象，展开一次
    auto copy = f.graph()->copy();                   // 复制函数图
    peeler.run(copy);                                // 执行循环展开
    int num_loops =
        std::count_if(copy->nodes().begin(), copy->nodes().end(), is_loop);  // 统计循环节点数量
    ASSERT_EQ(num_loops, 2);                         // 断言循环节点数量为 2
    Code code(copy, "");                             // 创建代码对象
    InterpreterState interpreter{code};              // 创建解释器状态
    interpreter.run(stack);                          // 运行解释器
    ASSERT_EQ(stack.back().toInt(), 20);             // 断言栈顶值为 20
  }

  // 测试多次循环展开
  {
    LoopsPeeler peeler(true_pred, 3);                // 创建循环展开器对象，展开三次
    auto copy = f.graph()->copy();                   // 复制函数图
    peeler.run(copy);                                // 执行循环展开
    int num_loops =
        std::count_if(copy->nodes().begin(), copy->nodes().end(), is_loop);  // 统计循环节点数量
    ASSERT_EQ(num_loops, 2);                         // 断言循环节点数量为 2
    Code code(copy, "");                             // 创建代码对象
    InterpreterState interpreter{code};              // 创建解释器状态
    interpreter.run(stack);                          // 运行解释器
    ASSERT_EQ(stack.back().toInt(), 20);             // 断言栈顶值为 20
  }
}

// 测试用例：验证使用归纳变量时的循环展开
TEST(LoopPeelerTest, YesInductionVariableUse) {
  // 显式使用归纳变量的函数定义字符串
  static const auto str_func_def = R"JIT(
    def test_peel_n_times():
      sum = 0
      for i in range(10):
        sum += i
      return sum
    )JIT";

  auto cu = compile(str_func_def);                   // 编译函数定义字符串
  auto& f = toGraphFunction(cu->get_function("test_peel_n_times"));  // 获取函数图
  auto stack = createStack({});                      // 创建执行栈

  // 循环展开一次
  {
    LoopsPeeler peeler(true_pred, 1);                // 创建循环展开器对象，展开一次
    auto copy = f.graph()->copy();                   // 复制函数图
    peeler.run(copy);                                // 执行循环展开
    int num_loops =
        std::count_if(copy->nodes().begin(), copy->nodes().end(), is_loop);  // 统计循环节点数量
    ASSERT_EQ(num_loops, 2);                         // 断言循环节点数量为 2
    Code code(copy, "");                             // 创建代码对象
    InterpreterState interpreter{code};              // 创建解释器状态
    interpreter.run(stack);                          // 运行解释器
    ASSERT_EQ(stack.back().toInt(), 45);             // 断言栈顶值为 45
  }

  // 测试多次循环展开
  {
    LoopsPeeler peeler(true_pred, 3);                // 创建循环展开器对象，展开三次
    auto copy = f.graph()->copy();                   // 复制函数图
    peeler.run(copy);                                // 执行循环展开
    int num_loops =
        std::count_if(copy->nodes().begin(), copy->nodes().end(), is_loop);  // 统计循环节点数量
    ASSERT_EQ(num_loops, 2);                         // 断言循环节点数量为 2
    Code code(copy, "");                             // 创建代码对象
    InterpreterState interpreter{code};              // 创建解释器状态
    interpreter.run(stack);                          // 运行解释器
    ASSERT_EQ(stack.back().toInt(), 45);             // 断言栈顶值为 45
  }
}
`
TEST(LoopPeelerTest, LoopWithTerminationCondition) {
  // 测试带有显式终止条件的循环
  static const auto str_func_def = R"JIT(
    def test_with_cond_times():
      sum = 0
      i = 0
      while (sum < 2):
        sum += i
        i += 1
      return sum
    )JIT";

  // 剥离循环会将终止条件改为 false，因此原始循环不会运行
  auto cu = compile(str_func_def);  // 编译函数定义字符串，生成计算单元
  auto& f = toGraphFunction(cu->get_function("test_with_cond_times"));  // 将函数转换为图形函数
  auto stack = createStack({});  // 创建一个空栈

  // 剥离 5 次迭代应该更新终止条件为 false
  {
    LoopsPeeler peeler(true_pred, 5);  // 初始化 LoopsPeeler，传入条件判断函数和迭代次数
    auto copy = f.graph()->copy();  // 复制图形函数的图形
    peeler.run(copy);  // 执行剥离操作
    int num_loops =
        std::count_if(copy->nodes().begin(), copy->nodes().end(), is_loop);  // 计算图中的循环节点数量
    ASSERT_EQ(num_loops, 2);  // 验证循环节点数量是否为 2
    Code code(copy, "");  // 创建代码对象
    InterpreterState interpreter{code};  // 初始化解释器状态
    interpreter.run(stack);  // 运行代码
    ASSERT_EQ(stack.back().toInt(), 3);  // 验证栈顶元素是否为 3
  }

  // 终止条件保持为 true
  {
    LoopsPeeler peeler(true_pred, 1);  // 初始化 LoopsPeeler，传入条件判断函数和迭代次数
    auto copy = f.graph()->copy();  // 复制图形函数的图形
    peeler.run(copy);  // 执行剥离操作
    int num_loops =
        std::count_if(copy->nodes().begin(), copy->nodes().end(), is_loop);  // 计算图中的循环节点数量
    ASSERT_EQ(num_loops, 2);  // 验证循环节点数量是否为 2
    Code code(copy, "");  // 创建代码对象
    InterpreterState interpreter{code};  // 初始化解释器状态
    interpreter.run(stack);  // 运行代码
    ASSERT_EQ(stack.back().toInt(), 3);  // 验证栈顶元素是否为 3
  }
}

// 测试简单的嵌套循环
TEST(LoopPeelerTest, SimpleNestedLoops) {
  static const auto str_func_def = R"JIT(
    def test_nested_loops():
      sum = 0
      i = 0
      for i in range(10):
        for j in range(10):
          sum += i + j
      return sum
    )JIT";

  auto cu = compile(str_func_def);  // 编译函数定义字符串，生成计算单元
  auto& f = toGraphFunction(cu->get_function("test_nested_loops"));  // 将函数转换为图形函数
  auto stack = createStack({});  // 创建一个空栈

  {
    LoopsPeeler peeler(true_pred, 1);  // 初始化 LoopsPeeler，传入条件判断函数和迭代次数
    auto copy = f.graph()->copy();  // 复制图形函数的图形
    peeler.run(copy);  // 执行剥离操作
    ASSERT_EQ(countNodes(copy, is_loop), 5);  // 验证循环节点数量是否为 5
    Code code(copy, "");  // 创建代码对象
    InterpreterState interpreter{code};  // 初始化解释器状态
    interpreter.run(stack);  // 运行代码
    ASSERT_EQ(stack.back().toInt(), 900);  // 验证栈顶元素是否为 900
  }

  {
    LoopsPeeler peeler(true_pred, 5);  // 初始化 LoopsPeeler，传入条件判断函数和迭代次数
    auto copy = f.graph()->copy();  // 复制图形函数的图形
    peeler.run(copy);  // 执行剥离操作
    ASSERT_EQ(countNodes(copy, is_loop), 5);  // 验证循环节点数量是否为 5
    Code code(copy, "");  // 创建代码对象
    InterpreterState interpreter{code};  // 初始化解释器状态
    interpreter.run(stack);  // 运行代码
    ASSERT_EQ(stack.back().toInt(), 900);  // 验证栈顶元素是否为 900
  }
}

TEST(LoopPeelerTest, SimpleNestedLoops2) {
  static const auto str_func_def = R"JIT(
    def test_nested_loops():
      sum = 0
      i = 0
      for i in range(10):
        j = 0
        while sum < 2:
          sum += i + j
          j += 1
      return sum
    )JIT";

  auto cu = compile(str_func_def);  // 编译函数定义字符串，生成计算单元
  auto& f = toGraphFunction(cu->get_function("test_nested_loops"));  // 将函数转换为图形函数
  auto stack = createStack({});  // 创建一个空栈
  {
    LoopsPeeler peeler(true_pred, 1);  // 初始化 LoopsPeeler，传入条件判断函数和迭代次数
    auto copy = f.graph()->copy();  // 复制图形函数的图形
    peeler.run(copy);  // 执行剥离操作
    ASSERT_EQ(countNodes(copy, is_loop), 5);  // 验证循环节点数量是否为 5
    Code code(copy, "");  // 创建代码对象
    InterpreterState interpreter{code};  // 初始化解释器状态
    interpreter.run(stack);  // 运行代码
    # 断言最后一个栈顶元素的整数值等于 3
    ASSERT_EQ(stack.back().toInt(), 3);
  }

  {
    # 创建一个 LoopsPeeler 对象，传入 true_pred 作为真实谓词，以及循环节点数目 5
    LoopsPeeler peeler(true_pred, 5);
    # 复制当前图形对象 f.graph()
    auto copy = f.graph()->copy();
    # 对复制的图形对象运行循环剥离操作
    peeler.run(copy);
    # 断言在剥离后的复制图形中循环节点的数量为 5
    ASSERT_EQ(countNodes(copy, is_loop), 5);
    # 使用复制的图形对象和空字符串创建一个 Code 对象
    Code code(copy, "");
    # 创建一个 InterpreterState 对象，传入前面创建的 Code 对象
    InterpreterState interpreter{code};
    # 运行解释器，将栈作为参数传入
    interpreter.run(stack);
    # 断言最后一个栈顶元素的整数值等于 3
    ASSERT_EQ(stack.back().toInt(), 3);
  }
}

TEST(JitTracing, Basic) {
  // 定义批处理大小和输入大小
  constexpr int batch_size = 4;
  constexpr int input_size = 256;

  // 计算隐藏层大小
  int hidden_size = 2 * input_size;

  // 生成随机输入数据
  auto input = at::randn({batch_size, input_size}, at::kCPU);
  auto hx = at::randn({batch_size, hidden_size}, at::kCPU);
  auto cx = at::randn({batch_size, hidden_size}, at::kCPU);

  // 随机生成权重张量
  auto w_ih = t_def(at::randn({4 * hidden_size, input_size}, at::kCPU));
  auto w_hh = t_def(at::randn({4 * hidden_size, hidden_size}, at::kCPU));

  // 构建 LSTM 图形
  auto graph = build_lstm();
  // 创建堆栈并将输入数据推入堆栈
  auto stack = createStack({input, hx, cx, w_ih, w_hh});
  // 对图形进行追踪
  auto traced = TraceGraph(graph, stack);

  // 检查追踪图的输入类型与指定的输入类型相同
  ASSERT_EQ(*traced->inputs().at(0)->type(), *TensorType::create(input));
  ASSERT_EQ(*traced->inputs().at(1)->type(), *TensorType::create(hx));
  ASSERT_EQ(*traced->inputs().at(2)->type(), *TensorType::create(cx));
  ASSERT_EQ(*traced->inputs().at(3)->type(), *TensorType::create(w_ih));
  ASSERT_EQ(*traced->inputs().at(4)->type(), *TensorType::create(w_hh));

  // 在堆栈中弹出分析输出
  Tensor prof_out;
  pop(stack, prof_out);

  {
    // 重新创建堆栈并将输入数据推入堆栈
    stack = createStack({input, hx, cx, w_ih, w_hh});
    // 使用追踪的代码构建代码对象
    Code cd(traced, "traced");
    InterpreterState is{cd};
    // 运行解释器状态
    is.run(stack);
    Tensor traced_out;
    // 在堆栈中弹出追踪输出
    pop(stack, traced_out);
    // 检查追踪输出与分析输出的全部近似性
    torch::allclose(prof_out, traced_out);
  }

  {
    // 重新创建堆栈并将输入数据推入堆栈
    stack = createStack({input, hx, cx, w_ih, w_hh});
    // 使用图形构建代码对象
    Code cd(graph, "graph");
    InterpreterState is{cd};
    // 运行解释器状态
    is.run(stack);
    Tensor scripted_out;
    // 在堆栈中弹出脚本化输出
    pop(stack, scripted_out);
    // 检查脚本化输出与分析输出的全部近似性
    torch::allclose(prof_out, scripted_out);
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(InsertAndEliminateRedundantGuardsTest, Basic) {
  // 静态定义基本示例字符串
  static const auto basic_example = R"JIT(
  def basic(x, y):
    a = x + y
    b = x * y
    c = x + 1
    d = a - c
    e = b - c
    return d + e
  )JIT";

  // 编译基本示例
  auto cu = compile(basic_example);
  auto& fun = toGraphFunction(cu->get_function("basic"));
  // 对函数图形进行 profiling 记录
  auto pr = ProfilingRecord::instrumentGraph(fun.graph());
  auto x = at::randn({2, 3}, at::kCPU);
  auto y = at::randn({2, 3}, at::kCPU);
  auto stack = createStack({x, y});
  // 引入一些 profiling 信息
  Code cd(pr->profiled_graph_, "");
  InterpreterState is{cd};
  // 运行解释器状态
  is.run(stack);
  // 复制 profiling 图形
  auto copy = pr->profiled_graph_->copy();
  // 移除 profiling 计数器
  ProfilingRecord::removeProfileCounter(copy->block());
  // 插入 guards
  InsertGuards(copy);
  // 获取节点列表
  auto nodes = copy->block()->nodes();
  // 查找第一个 guard 节点
  auto guard = std::find_if(nodes.begin(), nodes.end(), [](Node* n) {
  return n->kind() == prim::Guard;
});
ASSERT_NE(guard, nodes.end());
ASSERT_EQ(
    guard->input()->type()->expectRef<TensorType>().sizes().size(),
    c10::nullopt);
checkShape(*guard, {2, 3}, false);
auto is_guard = [](Node* n) { return n->kind() == prim::Guard; };
int num_guards = std::count_if(nodes.begin(), nodes.end(), is_guard);
ASSERT_EQ(num_guards, 12);
// 现在尽可能消除尽可能多的 guards
// 我们应该只剩下 x 和 y 定义上的两个 guards
EliminateRedundantGuards(copy);
num_guards = std::count_if(nodes.begin(), nodes.end(), is_guard);
ASSERT_EQ(num_guards, 2);
TEST(InsertBailOutsTest, Basic) {
  // 定义包含 JIT 代码的静态常量字符串
  static const auto basic_example = R"JIT(
  def basic_loop(x, y):

      a = x + 1
      b = y + 2
      c = x + y + 3

      for i in range(10):
          a = a + b
          // 不变量
          d = b * c
          //
          a = a - d

      e = a + 4
      return e
  )JIT";

  // 编译 JIT 代码得到编译单元
  auto cu = compile(basic_example);
  // 获取函数图
  auto& fun = toGraphFunction(cu->get_function("basic_loop"));
  // 对函数图进行性能分析记录
  auto pr = ProfilingRecord::instrumentGraph(fun.graph());
  // 创建输入张量 x 和 y
  auto x = at::randn({2, 3}, at::kCPU);
  auto y = at::randn({2, 3}, at::kCPU);
  // 创建输入张量的栈
  auto stack = createStack({x, y});
  // 引入一些性能分析信息
  Code cd(pr->profiled_graph_, "");
  // 创建解释器状态
  InterpreterState is{cd};
  // 运行解释器
  is.run(stack);
  // 复制性能分析后的图
  auto copy = pr->profiled_graph_->copy();
  // 移除性能分析计数器
  ProfilingRecord::removeProfileCounter(copy->block());
  // 插入守卫节点
  InsertGuards(copy);
  // 消除冗余的守卫节点
  EliminateRedundantGuards(copy);
  // 获取图中的所有节点
  auto nodes = copy->block()->nodes();
  // 定义判断节点是否为守卫节点的函数
  auto is_guard = [](Node* n) { return n->kind() == prim::Guard; };
  // 计算守卫节点的数量
  auto num_guards = std::count_if(nodes.begin(), nodes.end(), is_guard);
  // 断言守卫节点的数量为 3
  ASSERT_EQ(num_guards, 3);
  // 插入异常退出节点
  InsertBailOuts(copy);
  // 定义判断节点是否为异常退出节点的函数
  auto is_bailout = [](Node* n) { return n->kind() == prim::BailOut; };
  // 计算异常退出节点的数量
  auto num_bailouts = std::count_if(nodes.begin(), nodes.end(), is_bailout);
  // 断言守卫节点的数量与异常退出节点的数量相等
  ASSERT_EQ(num_guards, num_bailouts);
  // 创建异常退出节点的向量
  std::vector<Node*> bailouts(num_bailouts);
  // 将异常退出节点复制到向量中
  std::copy_if(nodes.begin(), nodes.end(), bailouts.begin(), is_bailout);

  // 遍历所有的异常退出节点
  for (auto blo : bailouts) {
    // 断言异常退出节点的输入的节点的类型为 BailoutTemplate
    ASSERT_EQ(blo->inputs().at(0)->node()->kind(), prim::BailoutTemplate);
  }
}
TEST(ProfilerTest, Basic) {
  // 定义批量大小和输入大小
  constexpr int batch_size = 4;
  constexpr int input_size = 256;

  // 计算隐藏层大小
  int hidden_size = 2 * input_size;

  // 创建随机输入张量和隐藏状态张量
  auto input = at::randn({batch_size, input_size}, at::kCPU);
  auto hx = at::randn({batch_size, hidden_size}, at::kCPU);
  auto cx = at::randn({batch_size, hidden_size}, at::kCPU);

  // 创建输入到隐藏状态权重和隐藏状态到隐藏状态权重张量
  auto w_ih = t_def(at::randn({4 * hidden_size, input_size}, at::kCPU));
  auto w_hh = t_def(at::randn({4 * hidden_size, hidden_size}, at::kCPU));

  // 构建 LSTM 模型
  auto g = build_lstm();

  // 创建输入栈
  auto stack = createStack({input, hx, cx, w_ih, w_hh});

  // 获取优化图的引用
  auto& opt_graph = *g.get();

  // 创建参数规范的实例
  ArgumentSpecCreator arg_spec_creator(opt_graph);
  ArgumentSpec spec =
      arg_spec_creator.create(autograd::GradMode::is_enabled(), stack);

  // 为优化图特化类型
  arg_spec_creator.specializeTypes(opt_graph, spec);

  // 对图进行性能分析
  auto pr = ProfilingRecord::instrumentGraph(g);

  // 创建代码对象
  Code cd(pr->profiled_graph_, "");

  // 创建解释器状态并运行
  InterpreterState is{cd};
  is.run(stack);

  // 检查分析的类型是否存储为属性并在转储中显示
  // 例如：Tensor = prim::profile[profiled_type=Double(4, 256, strides=[256, 1],
  // requires_grad=0, device=cpu)
  testing::FileCheck()
      .check("Tensor = prim::profile[profiled_type")
      ->check_same("256")
      ->run(*pr->profiled_graph_);

  // 查找特定类型的节点（如加法）并进行断言
  auto begin = pr->profiled_graph_->block()->nodes().begin();
  auto end = pr->profiled_graph_->block()->nodes().end();
  auto mm =
      std::find_if(begin, end, [](Node* n) { return n->kind() == aten::add; });
  ASSERT_NE(mm, end);
  std::vector<int64_t> mm_expected{4, 2048};

  // 检查节点的形状
  checkShape(mm->inputs().at(0)->node()->ty(attr::profiled_type), mm_expected);

  // 查找其他类型的节点（如乘法）并进行形状检查
  auto mul_n =
      std::find_if(begin, end, [](Node* n) { return n->kind() == aten::mul; });
  ASSERT_NE(mul_n, end);
  std::vector<int64_t> eltwise{4, 512};
  checkShape(mul_n->inputs().at(0)->node()->ty(attr::profiled_type), eltwise);

  // 查找另一种类型的节点（如双曲正切）并进行形状检查
  auto tanh_n =
      std::find_if(begin, end, [](Node* n) { return n->kind() == aten::tanh; });
  checkShape(tanh_n->inputs().at(0)->node()->ty(attr::profiled_type), eltwise);
}

TEST(ProfilerTest, OptionalProfiling) {
  // 创建图的共享指针和值映射
  auto graph = std::make_shared<Graph>();
  std::unordered_map<std::string, Value*> vmap;

  // 解析IR代码字符串
  parseIR(
      R"IR(
// 定义一个名为 graph 的函数，接受三个参数：inp（输入张量）、weight（权重张量）、bias（偏置张量）
graph(%inp : Tensor,
      %weight : Tensor,
      %bias : Tensor?):
  // 调用 PyTorch 的 linear 函数，对输入张量 inp 进行线性变换，使用 weight 作为权重，bias 作为偏置（可选）
  %1 : Tensor = aten::linear(%inp, %weight, %bias)
  // 返回线性变换的结果张量 %1
  return (%1))IR",

  // 用 graph 对象创建一个 ProfilingRecord，用于记录性能分析信息
  &*graph,
  vmap);

  // 设置性能分析计数为 2
  auto pr = ProfilingRecord::instrumentGraph(graph);
  pr->profiling_count_ = 2;

  // 创建随机张量 input、weight、bias
  auto input = torch::randn({1, 2});
  auto weight = torch::randn({2, 2});
  auto bias = torch::randn({1, 2});

  // 创建张量栈 stack，包含 input、weight、bias
  auto stack = createStack({input, weight, bias});
  // 创建代码对象 cd，使用 profiled_graph 进行初始化
  Code cd(pr->profiled_graph_, "");
  // 创建解释器状态 is，使用 cd 进行初始化
  InterpreterState is{cd};
  // 运行解释器状态 is，执行栈 stack 中的操作
  is.run(stack);

  // 运行测试，检查是否有一个 prim::profile 类型的节点
  testing::FileCheck()
      .check_count("Tensor? = prim::profile[profiled_type", 1, true)
      ->run(*pr->profiled_graph_);

  // 确保记录了张量形状
  auto begin = pr->profiled_graph_->block()->nodes().begin();
  auto end = pr->profiled_graph_->block()->nodes().end();
  // 查找第一个 aten::linear 类型的节点
  auto linear = std::find_if(
      begin, end, [](Node* n) { return n->kind() == aten::linear; });
  // 断言找到了该节点
  ASSERT_NE(linear, end);
  // 预期的偏置张量形状
  std::vector<int64_t> bias_expected_shape = {1, 2};
  // 获取 profiled_bias 节点
  auto profiled_bias = linear->namedInput("bias")->node();
  // 检查 profiled_bias 的张量类型是否符合预期形状 bias_expected_shape
  checkShape(profiled_bias->ty(attr::profiled_type), bias_expected_shape);
  // 断言 seen_none 属性为 0
  ASSERT_EQ(0, profiled_bias->i(attr::seen_none));

  // 创建一个空的 IValue 对象 none_bias
  auto none_bias = c10::IValue();

  // 清空栈 stack，并放入 input、weight、none_bias 三个张量
  stack.clear();
  stack.emplace_back(input);
  stack.emplace_back(weight);
  stack.emplace_back(none_bias);
  // 重新初始化解释器状态 is，并运行栈 stack
  is = InterpreterState{cd};
  is.run(stack);

  // 确保记录了 "None" 的信息
  begin = pr->profiled_graph_->block()->nodes().begin();
  end = pr->profiled_graph_->block()->nodes().end();
  linear = std::find_if(
      begin, end, [](Node* n) { return n->kind() == aten::linear; });
  ASSERT_NE(linear, end);
  profiled_bias = linear->namedInput("bias")->node();
  checkShape(profiled_bias->ty(attr::profiled_type), bias_expected_shape);
  // 断言 seen_none 属性为 1
  ASSERT_EQ(1, profiled_bias->i(attr::seen_none));
}

// 测试函数 CallStackTest.Basic
TEST(CallStackTest, Basic) {
  // 定义一个字符串 text，包含 Python 代码，定义了四个函数 ham、bar、baz 和 foo
  const auto text = R"(
def ham(x):
    return x/7

def bar(x):
    return x*3

def baz(x):
    return ham(x)*x

def foo(x):
    return bar(x)*baz(x)*11
  )";
  // 编译字符串 text，返回编译单元 cu
  auto cu = compile(text);
  // 获取函数 foo 的图形函数，并存储在 foo 中
  const auto& foo = toGraphFunction(cu->get_function("foo"));
  // 遍历 foo 函数的优化图中的每个节点
  for (Node* n : foo.optimized_graph()->nodes()) {
    // 迭代优化图中的每个节点
    for (Node* n : baz.optimized_graph()->nodes()) {
        // 检查节点是否为常量节点
        if (n->kind() == prim::Constant) {
            // 如果节点是常量节点，检查是否具有'value'属性且属性类型为整型
            if (!n->hasAttribute(attr::value) ||
                n->kindOf(attr::value) != AttributeKind::i) {
                // 如果不满足条件，则继续下一个节点的检查
                continue;
            }
            // 从节点中获取整型值v
            int v = n->i(attr::value);
            // 根据v的值进行不同的处理
            switch (v) {
                case 3: {
                    // 当v为3时，断言节点的调用栈(callstack)存在
                    ASSERT_TRUE(n->callstack());
                    // 获取调用栈的向量表示
                    auto callstack_vector = (*n->callstack())->vec();
                    // 断言调用栈向量大小为1
                    ASSERT_EQ(callstack_vector.size(), 1);
                    // 断言调用栈的第一个元素是函数'bar'
                    ASSERT_EQ(std::get<0>(callstack_vector[0]), &cu->get_function("bar"));
                    break;
                }
                case 7: {
                    // 当v为7时，断言节点的调用栈(callstack)存在
                    ASSERT_TRUE(n->callstack());
                    // 获取调用栈的向量表示
                    auto callstack_vector = (*n->callstack())->vec();
                    // 断言调用栈向量大小为2
                    ASSERT_EQ(callstack_vector.size(), 2);
                    // 断言调用栈的第一个元素是函数'baz'
                    ASSERT_EQ(std::get<0>(callstack_vector[0]), &cu->get_function("baz"));
                    // 断言调用栈的第二个元素是函数'ham'
                    ASSERT_EQ(std::get<0>(callstack_vector[1]), &cu->get_function("ham"));
                    break;
                }
                case 11: {
                    // 当v为11时，断言节点的调用栈(callstack)不存在
                    ASSERT_FALSE(n->callstack());
                    break;
                }
            }
        }
    }

    // 检查内联是否会破坏被调用函数节点的调用栈
    const auto& baz = toGraphFunction(cu->get_function("baz"));
    // 迭代优化图中的每个节点
    for (Node* n : baz.optimized_graph()->nodes()) {
        // 检查节点是否为常量节点
        if (n->kind() == prim::Constant) {
            // 如果节点是常量节点，检查是否具有'value'属性且属性类型为整型
            if (!n->hasAttribute(attr::value) ||
                n->kindOf(attr::value) != AttributeKind::i) {
                // 如果不满足条件，则继续下一个节点的检查
                continue;
            }
            // 从节点中获取整型值v
            int v = n->i(attr::value);
            // 断言v的值为7
            ASSERT_TRUE(v == 7);
            // 断言节点的调用栈(callstack)存在
            ASSERT_TRUE(n->callstack());
            // 获取调用栈的向量表示
            auto callstack_vector = (*n->callstack())->vec();
            // 断言调用栈向量大小为1
            ASSERT_EQ(callstack_vector.size(), 1);
            // 断言调用栈的第一个元素是函数'ham'
            ASSERT_EQ(std::get<0>(callstack_vector[0]), &cu->get_function("ham"));
        }
    }
TEST(CallStackTest, Caching) {
  const auto text = R"(

def a(x):
    print("a1")
    print("a2")
    return x

def b(x):
    print("b1")
    print("b2")
    a(x)
    return x

def c(x):
    print("c1")
    print("c2")
    b(x)
    return x
  )";
  auto cu = compile(text);
  // 获取函数 'c' 的图形化函数表示
  const auto& baz = toGraphFunction(cu->get_function("c"));
  // 用于存储调用堆栈对象的无序映射
  std::unordered_map<std::string, InlinedCallStack*> callstack_objects;
  // 遍历优化后图形的节点
  for (Node* n : baz.optimized_graph()->nodes()) {
    // 检查节点是否为常量
    if (n->kind() == prim::Constant) {
      // 检查常量节点是否包含 'value' 属性且属性类型为字符串
      if (!n->hasAttribute(attr::value) ||
          n->kindOf(attr::value) != AttributeKind::s) {
        continue;
      }
      // 获取字符串值
      std::string v = n->s(attr::value);
      // 如果节点有调用堆栈，则将调用堆栈对象存储到映射中
      if (n->callstack()) {
        callstack_objects[v] = n->callstack()->get();
      }
    }
  }
  // 断言确保常量值 "a1" 和 "a2" 在函数 'c' 中被内联，且它们的调用堆栈相同 (a->b->c)
  ASSERT_TRUE(callstack_objects.count("a1") && callstack_objects.count("a2"));
  ASSERT_TRUE(callstack_objects.at("a1") == callstack_objects.at("a2"));
}

TEST(InlinedCallStackTest, BlockAnnotation) {
  Module a("A");
  // 定义模块 'A' 的前向方法
  a.define(R"(
    def forward(self, x, y, z: int):
      if (z == 1):
        return x + y
      else:
        return x * y
  )");
  Module b("B");
  // 定义模块 'B' 的前向方法
  b.define(R"(
    def forward(self, x):
      return x + 2
  )");
  Module c("C");
  // 将模块 'A' 和 'B' 注册到模块 'C' 中
  c.register_module("A0", a);
  c.register_module("B0", b);
  // 定义模块 'C' 的前向方法
  c.define(R"(
    def forward(self, x, y, z: int):
      return self.A0.forward(x, y, z) + self.B0.forward(x)
  )");

  // 转换 'forward' 方法为图形化函数表示，并获取优化后的图形
  auto graph =
      toGraphFunction(c.get_method("forward").function()).optimized_graph();
  // 用于存储添加和乘法操作节点调用堆栈信息的流
  std::stringstream add_ss, mul_ss;
  // 遍历图形中的节点
  for (Node* n : graph->nodes()) {
    // 检查节点是否为 'prim::If' 类型
    if (n->kind() == prim::If) {
      // 遍历 'If' 节点的各个块
      for (Block* block : n->blocks()) {
        // 遍历块中的节点
        for (Node* if_node : block->nodes()) {
          // 如果节点是加法操作
          if (if_node->kind() == aten::add) {
            // 将加法操作节点的调用堆栈信息添加到 add_ss 流中
            for (const auto& e : if_node->callstack().value()->vec()) {
              add_ss << std::get<1>(e);
            }
            add_ss << if_node->sourceRange();
          }
          // 如果节点是乘法操作
          if (if_node->kind() == aten::mul) {
            // 将乘法操作节点的调用堆栈信息添加到 mul_ss 流中
            for (const auto& e : if_node->callstack().value()->vec()) {
              mul_ss << std::get<1>(e);
            }
            mul_ss << if_node->sourceRange();
          }
        }
      }
    }
  }
  }
  // 断言：确保字符串 "line 3" 在 add_ss 的输出中可以找到
  ASSERT_NE(add_ss.str().find("line 3"), std::string::npos);
  // 断言：确保字符串 "line 4" 在 add_ss 的输出中可以找到
  ASSERT_NE(add_ss.str().find("line 4"), std::string::npos);
  // 断言：确保字符串 "return self.A0.forward(x, y, z)" 在 add_ss 的输出中可以找到
  ASSERT_NE(
      add_ss.str().find("return self.A0.forward(x, y, z)"), std::string::npos);
  // 断言：确保字符串 "return x + y" 在 add_ss 的输出中可以找到
  ASSERT_NE(add_ss.str().find("return x + y"), std::string::npos);
  // 断言：确保字符串 "line 3" 在 mul_ss 的输出中可以找到
  ASSERT_NE(mul_ss.str().find("line 3"), std::string::npos);
  // 断言：确保字符串 "line 6" 在 mul_ss 的输出中可以找到
  ASSERT_NE(mul_ss.str().find("line 6"), std::string::npos);
  // 断言：确保字符串 "return self.A0.forward(x, y, z)" 在 mul_ss 的输出中可以找到
  ASSERT_NE(
      mul_ss.str().find("return self.A0.forward(x, y, z)"), std::string::npos);
  // 断言：确保字符串 "return x * y" 在 mul_ss 的输出中可以找到
  ASSERT_NE(mul_ss.str().find("return x * y"), std::string::npos);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(InlinedCallStackTest, SelfCallMethods) {
  // 创建名为 "A" 的模块并定义多个方法
  Module a("A");
  a.define(R"(
    def my_new_method(self, x):
      return x * 3
    def forward_impl_(self, x, y):
      return self.my_new_method(x) + y
    def forward(self, x, y):
      y = y + 2
      return self.forward_impl_(x, y)
  )");
  // 创建名为 "B" 的模块并定义一个方法
  Module b("B");
  b.define(R"(
    def forward(self, x):
      return x + 2
  )");
  // 创建名为 "C" 的模块并注册 "A" 和 "B" 模块
  Module c("C");
  c.register_module("A0", a);
  c.register_module("B0", b);
  c.define(R"(
    def call_b(self, x):
      return self.B0.forward(x)
    def forward(self, x, y):
      return self.A0.forward(x, y) + self.call_b(x)
  )");

  // 获取 "forward" 方法的图形表示并优化
  auto graph =
      toGraphFunction(c.get_method("forward").function()).optimized_graph();
  // 创建模块层次的哈希映射
  std::unordered_map<std::string, size_t> module_hierarchies;
  // 遍历图中的节点
  for (Node* n : graph->nodes()) {
    // 获取节点的模块层次
    auto hierarchy = torch::jit::utils::getNodesModuleHierarchy(*n);
    // 如果映射中不存在该层次，则初始化计数为 0
    if (module_hierarchies.count(hierarchy) == 0) {
      module_hierarchies[hierarchy] = 0;
    }
    // 增加该层次的计数
    module_hierarchies[hierarchy] += 1;
  }
  // 断言不同模块层次的出现次数
  ASSERT_EQ(module_hierarchies["A0(A)"], 2);
  ASSERT_EQ(module_hierarchies["A0(A).SELF(A).SELF(A)"], 2);
  ASSERT_EQ(module_hierarchies["A0(A).SELF(A)"], 1);
  ASSERT_EQ(module_hierarchies["SELF(C)"], 1);
  ASSERT_EQ(module_hierarchies["SELF(C).B0(B)"], 1);
}

// 符号测试
TEST(AutogradSymbolsTest, Basic) {
  // 创建一个符号对象
  Symbol sym = Symbol::fromQualString("aten::test_symbol");
  // 创建一个图形对象
  Graph graph;
  // 在图中创建一个节点
  auto node = graph.create(sym);
  // 检查节点是否能够运行自动微分
  TORCH_CHECK(canRunWithAutograd(node));

  // 更改符号对象并在图中创建一个新节点
  sym = Symbol::fromQualString("prim::test_symbol");
  node = graph.create(sym);
  // 再次检查节点是否能够运行自动微分
  TORCH_CHECK(canRunWithAutograd(node));

  // 更改符号对象至无法运行自动微分的类型，并创建新节点
  sym = Symbol::fromQualString("prim::FusionGroup");
  node = graph.create(sym);
  // 检查节点不能运行自动微分
  TORCH_CHECK(!canRunWithAutograd(node));

  // 更改符号对象至自定义类型，并创建新节点
  sym = Symbol::fromQualString("custom::test_symbol");
  node = graph.create(sym);
  // 检查节点不能运行自动微分
  TORCH_CHECK(!canRunWithAutograd(node));
}

// 默认参数类型提示测试
TEST(DefaultArgTypeHintingTest, Basic) {
  // 定义未提供类型提示的函数文本
  const auto text_non_hinted = R"(

def a(x, y=1):
    print("a1")
    print("a2")
    return x
  )";

  // 定义提供了类型提示的函数文本
  const auto text_hinted = R"(

def a(x, y:int=1):
    print("a1")
    print("a2")
    return x
  )";

  // 编译未提供类型提示的函数文本，预期会抛出异常
  try {
    compile(text_non_hinted);
    ASSERT_TRUE(0);
  } catch (const std::exception& c) {
  }

  // 编译提供了类型提示的函数文本
  auto cu = compile(text_hinted);
}

// FuturesTest 测试用例
TEST(FuturesTest, Basic) {
  // 创建一个带有整数类型的 Future 对象
  auto f1 = c10::make_intrusive<Future>(IntType::get());
  // 断言 Future 对象未完成和无值
  ASSERT_FALSE(f1->completed());
  ASSERT_FALSE(f1->hasValue());
  // 定义两个整型变量
  int32_t sat1 = 0;
  int32_t sat2 = 0;
  // 向 Future 对象添加回调函数，并标记其完成并赋值为 43
  f1->addCallback([&](Future& /* unused */) { ++sat1; });
  f1->markCompleted(43);
  // 断言 Future 对象已完成和有值，并无错误
  ASSERT_TRUE(f1->completed());
  ASSERT_TRUE(f1->hasValue());
  ASSERT_FALSE(f1->hasError());
  // 断言回调函数被调用次数为 1，值为 43
  ASSERT_EQ(sat1, 1);
  ASSERT_EQ(f1->constValue().toInt(), 43);
  ASSERT_EQ(f1->value().toInt(), 43);
  // 再次添加回调函数，并断言回调函数调用次数
  f1->addCallback([&](Future& /* unused */) { ++sat2; });
  ASSERT_EQ(sat1, 1);
  ASSERT_EQ(sat2, 1);
}
TEST(FutureTest, SparseTensor) {
  // 检查是否有 CUDA 可用，如果没有则跳过测试
  bool has_cuda = at::globalContext().hasCUDA();
  if (!has_cuda) {
    LOG(INFO) << "CUDA not available, skipping test";
  }
  // 循环两次，测试两种情况
  for (int i = 0; i < 2; ++i) {
    // 创建一个 Future 对象，表示未来可能返回的稀疏张量
    auto f = c10::make_intrusive<Future>(TensorType::get());
    // 设置张量选项为在 CUDA 设备上运行
    at::TensorOptions opts = at::TensorOptions().device(at::DeviceType::CUDA);
    // 创建稀疏张量，根据 i 的值选择不同的创建方式
    auto sparse_tensor = i == 0 ? at::ones(10).to_sparse()
                                : at::sparse_coo_tensor(
                                      at::arange(10).unsqueeze(0).to(at::kLong),
                                      at::ones({10, 10}),
                                      opts);
    // 标记 Future 完成，并设置其返回值为稀疏张量
    f->markCompleted(sparse_tensor);
    // 断言 Future 已完成
    ASSERT_TRUE(f->completed());
    // 断言 Future 没有错误
    ASSERT_FALSE(f->hasError());
  }
}

// Basic error cases.
TEST(FuturesTest, Error) {
  // 创建一个 Future 对象，表示未来可能返回的整数
  auto f1 = c10::make_intrusive<Future>(IntType::get());
  // 用于计数的变量
  int sat1 = 0;
  int sat2 = 0;
  // 向 Future 对象添加回调函数
  f1->addCallback([&](Future& /* unused */) { ++sat1; });
  // 设置 Future 对象的错误状态
  f1->setError(
      std::make_exception_ptr(c10::ivalue::Future::FutureError("Failed")));
  // 断言回调函数被调用了一次
  ASSERT_EQ(sat1, 1);
  // 断言 Future 已完成
  ASSERT_TRUE(f1->completed());
  // 断言 Future 出现了错误
  ASSERT_TRUE(f1->hasError());
  // 断言 Future 没有返回值
  ASSERT_FALSE(f1->hasValue());
  // 尝试获取 Future 的值，预期抛出异常
  try {
    (void)f1->value();
    ASSERT_TRUE(false); // Supposed to throw.
  } catch (const std::exception& e) {
    // 断言异常的消息与预期的一致
    ASSERT_TRUE(strcmp(e.what(), "Failed") == 0);
  }
  // 向 Future 对象添加另一个回调函数
  f1->addCallback([&](Future& /* unused */) { ++sat2; });
  // 断言第一个回调函数调用次数不变
  ASSERT_EQ(sat1, 1);
  // 断言第二个回调函数被调用了一次
  ASSERT_EQ(sat2, 1);
  // 如果有必要设置 Future 对象的错误状态
  f1->setErrorIfNeeded(
      std::make_exception_ptr(c10::ivalue::Future::FutureError("Dup")));
  // 断言尝试检索错误消息，返回的消息与预期一致
  ASSERT_TRUE(strcmp(f1->tryRetrieveErrorMessage().c_str(), "Failed") == 0);
  // 断言第一个回调函数调用次数不变
  ASSERT_EQ(sat1, 1);
  // 断言第二个回调函数调用次数不变
  ASSERT_EQ(sat2, 1);
  // 尝试获取 Future 的值，预期抛出异常
  try {
    (void)f1->constValue();
    ASSERT_TRUE(false); // Supposed to throw.
  } catch (const std::exception& e) {
    // 原始错误应已记录
    ASSERT_TRUE(std::string(e.what()).find("Failed") != std::string::npos);
  }
}

// then
TEST(FuturesTest, Then) {
  // 创建一个 Future 对象，表示未来可能返回的整数
  auto f1 = c10::make_intrusive<Future>(IntType::get());
  // 使用 f1 创建一个新的 Future 对象 f2，其返回值为 f1 的返回值加一
  auto f2 = f1->then(
      [](Future& f1) -> IValue { return f1.constValue().toInt() + 1; },
      IntType::get());
  // 使用 f2 创建一个新的 Future 对象 f3，其返回值为 f2 的返回值乘三
  auto f3 = f2->then(
      [](Future& f2) -> IValue { return f2.constValue().toInt() * 3; },
      IntType::get());
  // 用于标记测试完成的标志
  bool done = false;
  // 向 f3 添加回调函数
  f3->addCallback([&done](Future& f3) {
    // 断言 f3 的返回值等于 (42 + 1) * 3
    ASSERT_EQ(f3.constValue().toInt(), (42 + 1) * 3);
    // 标记测试完成
    done = true;
  });
  // 断言测试未完成
  ASSERT_FALSE(done);
  // 标记 f1 完成，设置其返回值为 42
  f1->markCompleted(42);
  // 断言测试已完成
  ASSERT_TRUE(done);
}

// collectAll()
TEST(FuturesTest, CollectAll) {
  // 创建三个类型为IntType的Future对象s1, s2, s3
  auto s1 = c10::make_intrusive<Future>(IntType::get());
  auto s2 = c10::make_intrusive<Future>(IntType::get());
  auto s3 = c10::make_intrusive<Future>(IntType::get());

  // 创建一个空的Future列表futures
  c10::List<intrusive_ptr<ivalue::Future>> futures(
      FutureType::create(IntType::get()));

  // 调用collectAll函数处理空列表futures
  auto c1 = collectAll(futures);

  // 断言c1已完成
  ASSERT_TRUE(c1->completed());

  // 断言c1的值为空列表
  ASSERT_EQ(c1->value().toList().size(), 0);

  // 断言c1的值的元素类型与IntType的Future类型相同
  ASSERT_TRUE(
      *(c1->value().toList().elementType()) ==
      *FutureType::create(IntType::get()));

  // 将s1添加到futures中，形成1个元素的列表
  futures.push_back(s1);

  // 再次调用collectAll函数处理带有一个元素的列表futures
  auto c2 = collectAll(futures);

  // 断言c2未完成
  ASSERT_FALSE(c2->completed());

  // 标记s1为已完成状态
  s1->markCompleted(5);

  // 断言c2已完成
  ASSERT_TRUE(c2->completed());

  // 断言c2的值为包含1个元素的列表
  ASSERT_EQ(c2->value().toList().size(), 1);

  // 断言c2的值的元素类型与IntType的Future类型相同
  ASSERT_TRUE(
      *(c2->value().toList().elementType()) ==
      *FutureType::create(IntType::get()));

  // 断言c2的第一个元素的值为5
  ASSERT_EQ(c2->value().toList().get(0).toFuture()->value().toInt(), 5);

  // 再次调用collectAll函数处理带有一个元素的列表futures
  auto c3 = collectAll(futures);

  // 断言c3已完成
  ASSERT_TRUE(c3->completed());

  // 断言c3的值为包含1个元素的列表
  ASSERT_EQ(c3->value().toList().size(), 1);

  // 断言c3的第一个元素的值为5
  ASSERT_EQ(c3->value().toList().get(0).toFuture()->value().toInt(), 5);

  // 将s2和s3添加到futures中，形成3个元素的列表
  futures.push_back(s2);
  futures.push_back(s3);

  // 再次调用collectAll函数处理带有三个元素的列表futures
  auto c4 = collectAll(futures);

  // 断言c4未完成
  ASSERT_FALSE(c4->completed());

  // 标记s3为已完成状态
  s3->markCompleted(7);

  // 断言c4未完成
  ASSERT_FALSE(c4->completed());

  // 标记s2为已完成状态
  s2->markCompleted(6);

  // 断言c4已完成
  ASSERT_TRUE(c4->completed());

  // 断言c4的值为包含3个元素的列表
  ASSERT_EQ(c4->value().toList().size(), 3);

  // 断言c4的第一个元素的值为5
  ASSERT_EQ(c4->value().toList().get(0).toFuture()->value().toInt(), 5);

  // 断言c4的第二个元素的值为6
  ASSERT_EQ(c4->value().toList().get(1).toFuture()->value().toInt(), 6);

  // 断言c4的第三个元素的值为7
  ASSERT_EQ(c4->value().toList().get(2).toFuture()->value().toInt(), 7);

  // 断言c4的值的元素类型与IntType的Future类型相同
  ASSERT_TRUE(
      *(c4->value().toList().elementType()) ==
      *FutureType::create(IntType::get()));

  // 创建一个新的Future对象s4，并将其添加到futures中
  auto s4 = c10::make_intrusive<Future>(IntType::get());
  futures.push_back(s4);

  // 再次调用collectAll函数处理带有四个元素的列表futures
  auto c5 = collectAll(futures);

  // 断言c5未完成
  ASSERT_FALSE(c5->completed());

  // 设置s4的错误状态
  s4->setError(
      std::make_exception_ptr(c10::ivalue::Future::FutureError("Failed")));

  // 断言c5已完成
  ASSERT_TRUE(c5->completed());

  // 尝试获取c5的值，断言抛出异常
  try {
    c5->value();
    ASSERT_TRUE(false); // supposed to throw
  } catch (const std::exception& e) {
    // 断言异常的消息为"Failed"
    ASSERT_EQ(std::string(e.what()), "Failed");
  }
}

// collectAny()
TEST(FuturesTest, CollectAny) {
  auto s1 = c10::make_intrusive<Future>(IntType::get());

  // Empty case
  c10::List<intrusive_ptr<ivalue::Future>> futures(
      FutureType::create(IntType::get()));
  auto c1 = collectAny(futures);
  ASSERT_TRUE(c1->completed());

  // 1 element, not yet satisfied
  futures.push_back(s1);
  auto c2 = collectAny(futures);
  ASSERT_FALSE(c2->completed());
  s1->markCompleted(5);
  ASSERT_TRUE(c2->completed());
  ASSERT_TRUE(c2->value().isInt());
  ASSERT_EQ(c2->value().toInt(), 5);

  // 1 element already satisfied.
  auto c3 = collectAny(futures);
  ASSERT_TRUE(c3->completed());
  ASSERT_TRUE(c3->value().isInt());
  ASSERT_EQ(c3->value().toInt(), 5);

  // 2 elements
  futures.clear();
  auto s2 = c10::make_intrusive<Future>(IntType::get());
  auto s3 = c10::make_intrusive<Future>(IntType::get());
  futures.push_back(s2);
  futures.push_back(s3);
  auto c4 = collectAny(futures);
  ASSERT_FALSE(c4->completed());
  s3->markCompleted(7);
  ASSERT_TRUE(c4->completed());
  ASSERT_EQ(c4->value().toInt(), 7);
  s2->markCompleted(1);
  ASSERT_EQ(c4->value().toInt(), 7);
}

TEST(TLSFutureCallbacksTest, Basic) {
  // cb that verifies the profiler is enabled
  auto profilerEnabledCb = [](Future& /* unused */) {
    ASSERT_TRUE(torch::autograd::profiler::profilerEnabled());
  };
  // test running callbacks with propagation of TLS state.
  {
    // Enable the profiler in this thread
    torch::autograd::profiler::enableProfilerLegacy(
        torch::autograd::profiler::ProfilerConfig(
            torch::autograd::profiler::ProfilerState::CPU, false, false));
    auto s1 = c10::make_intrusive<Future>(IntType::get());
    s1->addCallback(wrapPropagateTLSState(profilerEnabledCb));
    std::thread t([s1 = std::move(s1)]() { s1->markCompleted(); });
    // Since we join here, we can ensure that all callbacks corresponding to
    // markCompleted() have finished.
    t.join();
    torch::autograd::profiler::disableProfilerLegacy();
  }
  // then() with TLS State
  {
    // Enable the profiler in this thread
    torch::autograd::profiler::enableProfilerLegacy(
        torch::autograd::profiler::ProfilerConfig(
            torch::autograd::profiler::ProfilerState::CPU, false, false));
    auto s1 = c10::make_intrusive<Future>(IntType::get());
    auto s2 = s1->then(
        wrapPropagateTLSState([&profilerEnabledCb](Future& s1) {
          profilerEnabledCb(s1);
          return at::IValue(1);
        }),
        IntType::get());
    std::thread t([s1 = std::move(s1)]() { s1->markCompleted(); });
    t.join();
    s2->wait();
    torch::autograd::profiler::disableProfilerLegacy();
  }
}

TEST(ProfilerDisableInCallbackTest, Basic) {
  // cb that verifies the profiler is enabled
  auto profilerEnabledCb = []() {
    // code to be added here...
  };


注释：


// 定义测试用例 FuturesTest.CollectAny，测试 collectAny 函数的多种情况
TEST(FuturesTest, CollectAny) {
  auto s1 = c10::make_intrusive<Future>(IntType::get());

  // 创建一个空的 futures 列表
  c10::List<intrusive_ptr<ivalue::Future>> futures(
      FutureType::create(IntType::get()));
  // 对空列表进行 collectAny 操作
  auto c1 = collectAny(futures);
  ASSERT_TRUE(c1->completed());

  // 添加一个未满足的元素到列表中
  futures.push_back(s1);
  auto c2 = collectAny(futures);
  ASSERT_FALSE(c2->completed());
  // 标记第一个 future 为已完成，并验证其返回值
  s1->markCompleted(5);
  ASSERT_TRUE(c2->completed());
  ASSERT_TRUE(c2->value().isInt());
  ASSERT_EQ(c2->value().toInt(), 5);

  // 已满足的元素
  auto c3 = collectAny(futures);
  ASSERT_TRUE(c3->completed());
  ASSERT_TRUE(c3->value().isInt());
  ASSERT_EQ(c3->value().toInt(), 5);

  // 添加两个元素到列表中
  futures.clear();
  auto s2 = c10::make_intrusive<Future>(IntType::get());
  auto s3 = c10::make_intrusive<Future>(IntType::get());
  futures.push_back(s2);
  futures.push_back(s3);
  auto c4 = collectAny(futures);
  ASSERT_FALSE(c4->completed());
  // 标记第二个 future 为已完成，并验证 collectAny 返回的值
  s3->markCompleted(7);
  ASSERT_TRUE(c4->completed());
  ASSERT_EQ(c4->value().toInt(), 7);
  s2->markCompleted(1);
  ASSERT_EQ(c4->value().toInt(), 7);
}

// 定义测试用例 TLSFutureCallbacksTest.Basic，测试带有 TLS 状态传播的回调函数
TEST(TLSFutureCallbacksTest, Basic) {
  // 定义一个验证分析器是否启用的回调函数
  auto profilerEnabledCb = [](Future& /* unused */) {
    ASSERT_TRUE(torch::autograd::profiler::profilerEnabled());
  };
  // 测试在传播 TLS 状态下运行回调函数
  {
    // 在当前线程启用分析器
    torch::autograd::profiler::enableProfilerLegacy(
        torch::autograd::profiler::ProfilerConfig(
            torch::autograd::profiler::ProfilerState::CPU, false, false));
    auto s1 = c10::make_intrusive<Future>(IntType::get());
    s1->addCallback(wrapPropagateTLSState(profilerEnabledCb));
    // 创建一个新线程运行 markCompleted() 方法
    std::thread t([s1 = std::move(s1)]() { s1->markCompleted(); });
    // 等待线程执行完成，确保 markCompleted() 对应的所有回调函数都已经完成
    t.join();
    // 禁用分析器
    torch::autograd::profiler::disableProfilerLegacy();
  }
  // 使用 TLS 状态的 then() 方法测试
  {
    // 在当前线程启用分析器
    torch::autograd::profiler::enableProfilerLegacy(
        torch::autograd::profiler::ProfilerConfig(
            torch::autograd::profiler::ProfilerState::CPU, false, false));
    auto s1 = c10::make_intrusive<Future>(IntType::get());
    auto s2 = s1->then(
        wrapPropagateTLSState([&profilerEnabledCb](Future& s1) {
          profilerEnabledCb(s1);
          return at::IValue(1);
        }),
        IntType::get());
    // 创建一个新线程运行 markCompleted() 方法
    std::thread t([s1 = std::move(s1)]() { s1->markCompleted(); });
    t.join();
    // 等待 s2 future 完成
    s2->wait();
    // 禁用分析器
    torch::autograd::profiler::disableProfilerLegacy();
  }
}

// 定义测试用例 ProfilerDisableInCallbackTest.Basic，测试回调函数中分析器禁用的情况
TEST(ProfilerDisableInCallbackTest, Basic) {
  // 定义一个验证分析器是否启用的回调函数
  auto profilerEnabledCb = []() {
    // 在此处添加验证代码...
  };
    // 确保 Torch 的分析器处于启用状态
    ASSERT_TRUE(torch::autograd::profiler::profilerEnabled());
  };
  // 启用传统的分析器，并配置为仅在 CPU 上运行，不收集线程事件
  torch::autograd::profiler::enableProfilerLegacy(
      torch::autograd::profiler::ProfilerConfig(
          torch::autograd::profiler::ProfilerState::CPU, false, false));
  // 创建一个具有整数类型的 Future 对象
  auto s1 = c10::make_intrusive<Future>(IntType::get());
  // 包装函数，用于在 TLS 状态中传播启用分析器的回调
  auto verifyProfilerCb =
      wrapPropagateTLSState([&profilerEnabledCb](Future& /* unused */) {
        // 确保分析器仍在此线程中启用
        profilerEnabledCb();
        // 创建两个大小为 2x2 的全一张量
        auto t1 = torch::ones({2, 2});
        auto t2 = torch::ones({2, 2});
        // 对两个张量执行加法操作
        torch::add(t1, t2);
        // 设置选项，禁用分析器并不清理 TLS 状态，仅进行合并
        auto opts =
            torch::autograd::profiler::ProfilerDisableOptions(false, true);
        // 禁用分析器并返回线程事件列表
        auto thread_event_lists =
            // NOLINTNEXTLINE(performance-move-const-arg)
            torch::autograd::profiler::disableProfilerLegacy(std::move(opts));
        // 确保在调用 disableProfilerLegacy() 后仍会对该线程的事件进行分析，
        // 并在合并列表中获取预期的事件
        bool found_ones = false;
        bool found_add = false;
        for (const auto& li : thread_event_lists) {
          for (const auto& evt : li) {
            // 检查事件名称以确定是否找到预期的 aten::add 和 aten::ones 事件
            if (strcmp(evt.name(), "aten::add") == 0) {
              found_add = true;
            } else if (strcmp(evt.name(), "aten::ones") == 0) {
              found_ones = true;
            }
          }
          if (found_add && found_ones) {
            break;
          }
        }
        // 确保找到了 aten::ones 和 aten::add 事件
        ASSERT_TRUE(found_ones);
        ASSERT_TRUE(found_add);
      });

  // 将 verifyProfilerCb 添加为回调函数到 s1 对象中
  s1->addCallback(verifyProfilerCb);
  // 禁用分析器，但不在主线程中合并结果
  auto opts = torch::autograd::profiler::ProfilerDisableOptions(true, false);
  // NOLINTNEXTLINE(performance-move-const-arg)
  torch::autograd::profiler::disableProfilerLegacy(std::move(opts));
  // 创建一个线程 t，在其中完成 s1 对象的标记
  std::thread t([s1 = std::move(s1)]() { s1->markCompleted(at::IValue(1)); });
  // 等待线程 t 完成
  t.join();

  // 与上述测试类似，但在后续运行在主线程时验证正确性
  torch::autograd::profiler::enableProfilerLegacy(
      torch::autograd::profiler::ProfilerConfig(
          torch::autograd::profiler::ProfilerState::CPU, false, false));
  // 重新创建具有整数类型的 Future 对象
  s1 = c10::make_intrusive<Future>(IntType::get());
  // 将 verifyProfilerCb 添加为回调函数到 s1 对象中
  s1->addCallback(verifyProfilerCb);
  // 在当前线程内完成回调函数的执行
  s1->markCompleted(at::IValue(1));
  // 设置选项，禁用分析器，但不在主线程中合并结果
  opts = torch::autograd::profiler::ProfilerDisableOptions(true, false);
  // NOLINTNEXTLINE(performance-move-const-arg)
  torch::autograd::profiler::disableProfilerLegacy(std::move(opts));
TEST(RecordDebugHandles, Basic) {
  // 创建一个包含 CPU 活动类型的集合，用于配置分析器
  const std::set<torch::autograd::profiler::ActivityType> activities(
      {torch::autograd::profiler::ActivityType::CPU});
  // 准备分析器，使用 KINETO 状态，并禁用额外的配置选项
  torch::autograd::profiler::prepareProfiler(
      torch::autograd::profiler::ProfilerConfig(
          torch::autograd::profiler::ProfilerState::KINETO, false, false),
      activities);
  // 启用分析器，使用 KINETO 状态，并禁用额外的配置选项
  torch::autograd::profiler::enableProfiler(
      torch::autograd::profiler::ProfilerConfig(
          torch::autograd::profiler::ProfilerState::KINETO, false, false),
      activities);

  {
    // 记录具有调试句柄和输入的边缘范围作用域
    RECORD_EDGE_SCOPE_WITH_DEBUG_HANDLE_AND_INPUTS("my_function", 42, {});
    // 定义变量 x 和 y，并计算它们的除法
    float x{5.9999}, y{2.1212};
    float z = x / y;
    // 禁止编译器优化未使用的变量
    (void)z;
  }

  {
    // 记录具有输入的用户范围作用域
    RECORD_USER_SCOPE_WITH_INPUTS("not_my_function", {});
    // 定义变量 x 和 y，并计算它们的除法
    float x{5.9999}, y{2.1212};
    float z = x / y;
    // 禁止编译器优化未使用的变量
    (void)z;
  }

  // 禁用分析器并获取分析结果的指针
  auto profiler_results_ptr = torch::autograd::profiler::disableProfiler();
  // 获取 KINETO 事件列表的常量引用
  const auto& kineto_events = profiler_results_ptr->events();
  // 初始化计数器来记录特定事件名的事件数量
  size_t my_events{0};
  // 遍历 KINETO 事件列表
  for (const auto& e : kineto_events) {
    // 检查事件名是否为 "my_function"
    if (e.name() == "my_function") {
      // 断言调试句柄是否为 42
      ASSERT_EQ(e.debugHandle(), 42);
      // 增加符合条件的事件计数
      my_events++;
    } 
    // 检查事件名是否为 "not_my_function"
    else if (e.name() == "not_my_function") {
      // 断言调试句柄是否为 -1
      ASSERT_EQ(e.debugHandle(), -1);
      // 增加符合条件的事件计数
      my_events++;
    }
  }
  // 断言符合条件的事件数量为 2
  ASSERT_EQ(my_events, 2);
}

TEST(RecordDebugHandles, ScopedCallbacks) {
  // 启用分析器，使用 KINETO 状态，并禁用额外的配置选项
  torch::autograd::profiler::prepareProfiler(
      torch::autograd::profiler::ProfilerConfig(
          torch::autograd::profiler::ProfilerState::KINETO, false, false),
      {torch::autograd::profiler::ActivityType::CPU});
  // 启用分析器，使用 KINETO 状态，并禁用额外的配置选项
  torch::autograd::profiler::enableProfiler(
      torch::autograd::profiler::ProfilerConfig(
          torch::autograd::profiler::ProfilerState::KINETO, false, false),
      {torch::autograd::profiler::ActivityType::CPU});

  {
    // 生成大小为 128x128 的随机张量 a 和 b
    auto a = torch::rand({128, 128});
    auto b = torch::rand({128, 128});
    // 计算张量 c，它是张量 a 和 b 的元素和
    auto c = a + b;
  }

  // 禁用分析器并获取分析结果的指针
  auto profiler_results_ptr = torch::autograd::profiler::disableProfiler();
  // 断言获得的事件列表不为空
  ASSERT_TRUE(profiler_results_ptr->events().size() > 0);

  // 启用分析器，使用 KINETO 状态，并禁用额外的配置选项，同时指定 LITE_INTERPRETER 记录范围
  torch::autograd::profiler::prepareProfiler(
      torch::autograd::profiler::ProfilerConfig(
          torch::autograd::profiler::ProfilerState::KINETO, false, false),
      {torch::autograd::profiler::ActivityType::CPU});
  // 启用分析器，使用 KINETO 状态，并禁用额外的配置选项，同时指定 LITE_INTERPRETER 记录范围
  torch::autograd::profiler::enableProfiler(
      torch::autograd::profiler::ProfilerConfig(
          torch::autograd::profiler::ProfilerState::KINETO, false, false),
      {torch::autograd::profiler::ActivityType::CPU},
      {at::RecordScope::LITE_INTERPRETER});

  {
    // 生成大小为 128x128 的随机张量 a 和 b
    auto a = torch::rand({128, 128});
    auto b = torch::rand({128, 128});
  auto c = a + b;
  // 计算变量 c 的值，为变量 a 和 b 的和

profiler_results_ptr = torch::autograd::profiler::disableProfiler();
// 禁用 PyTorch 自动求导的分析器，返回分析结果指针

ASSERT_TRUE(profiler_results_ptr->events().size() == 0);
// 断言分析结果中的事件数量为 0

torch::autograd::profiler::prepareProfiler(
    torch::autograd::profiler::ProfilerConfig(
        torch::autograd::profiler::ProfilerState::KINETO, false, false),
    {torch::autograd::profiler::ActivityType::CPU});
// 准备使用 Kineto 后端的 PyTorch 分析器，配置为不包括 CUDA 事件，不启用内存分析

torch::autograd::profiler::enableProfiler(
    torch::autograd::profiler::ProfilerConfig(
        torch::autograd::profiler::ProfilerState::KINETO, false, false),
    {torch::autograd::profiler::ActivityType::CPU},
    {at::RecordScope::LITE_INTERPRETER});
// 启用 PyTorch 分析器，使用 Kineto 后端，配置为不包括 CUDA 事件，不启用内存分析，记录轻量级解释器的范围

{
  RECORD_EDGE_SCOPE_WITH_DEBUG_HANDLE_AND_INPUTS("my_function", 42, {});
  // 在分析器中记录带有调试句柄和空输入的边缘范围，命名为 "my_function"，调试句柄为 42
  auto a = torch::rand({128, 128});
  // 创建一个大小为 128x128 的随机张量 a
  auto b = torch::rand({128, 128});
  // 创建一个大小为 128x128 的随机张量 b
  auto c = a + b;
  // 计算变量 c 的值，为变量 a 和 b 的和
}

{
  RECORD_USER_SCOPE_WITH_INPUTS("not_my_function", {});
  // 在分析器中记录带有空输入的用户范围，命名为 "not_my_function"
  auto a = torch::rand({128, 128});
  // 创建一个大小为 128x128 的随机张量 a
  auto b = torch::rand({128, 128});
  // 创建一个大小为 128x128 的随机张量 b
  auto c = a + b;
  // 计算变量 c 的值，为变量 a 和 b 的和
}

profiler_results_ptr = torch::autograd::profiler::disableProfiler();
// 再次禁用 PyTorch 自动求导的分析器，返回分析结果指针

const auto& kineto_events = profiler_results_ptr->events();
// 获取分析结果中的事件列表引用

for (const auto& e : kineto_events) {
  // 遍历分析结果中的每一个事件
  if (e.name() == "my_function") {
    // 如果事件的名称为 "my_function"
    ASSERT_EQ(e.debugHandle(), 42);
    // 使用断言检查事件的调试句柄是否为 42
  }
}

ASSERT_TRUE(profiler_results_ptr->events().size() == 1);
// 断言分析结果中的事件数量为 1
}

// 定义名为 IValueKWargsTest 的测试用例
TEST(IValueKWargsTest, Basic) {
  // 定义一个包含 Python 代码的字符串 text，其中包含一个带有默认参数的函数定义
  const auto text = R"(
    def foo(a : int, b : int, c : int = 4):
      return a + 2*b + 3*c
  )";
  // 编译给定的 Python 代码，返回一个编译单元 cu
  auto cu = compile(text);
  // 从编译单元 cu 中获取名为 "foo" 的函数，并传入参数 {1} 和关键字参数 {"b": 3} 运行该函数
  auto result = cu->get_function("foo")({1}, {{"b", 3}});
  // 断言运行结果 result 的整数值为 19
  ASSERT_EQ(result.toInt(), 19);
}

// 定义名为 TestConstant 的测试用例，测试张量的梯度
TEST(TestConstant, TensorGrad) {
  // 创建一个新的计算图 graph
  auto graph = std::make_shared<Graph>();
  // 使用 torch::randn 创建一个形状为 {3, 5} 的张量 ten，并设置其需要梯度
  IValue ten = torch::randn({3, 5}).requires_grad_(true);
  // 尝试将张量 ten 插入到计算图 graph 中的常量中
  auto con = tryInsertConstant(*graph, ten);
  // 断言 con 为空值
  ASSERT_TRUE(con == c10::nullopt);
}

// 定义名为 TestMutation 的测试用例，测试移除张量的变异操作
TEST(TestMutation, Basic) {
  // 创建一个新的计算图 graph
  auto graph = std::make_shared<Graph>();
  // 创建一个空的无序映射 vmap
  std::unordered_map<std::string, Value*> vmap;
  // 解析给定的 IR 字符串，并将结果填充到计算图 graph 中
  parseIR(
      R"IR(
graph(%x.1 : Tensor):
  %2 : int = prim::Constant[value=1]()
  %9 : int = prim::Constant[value=4]()
  %x.3 : Tensor = aten::add(%x.1, %2, %2)
  %7 : Tensor = aten::add_(%x.3, %2, %2)
  %y.1 : Tensor = aten::add(%x.3, %9, %2)
  return (%y.1))IR",
      &*graph,
      vmap);
  // 移除计算图 graph 中满足特定条件的张量变异操作
  RemoveTensorMutation(graph, [](Node*) { return false; });
  // 运行 FileCheck，检查计算图中是否存在 "aten::add_" 操作
  testing::FileCheck().check("aten::add_")->run(*graph);
  // 再次移除计算图 graph 中满足特定条件的张量变异操作
  RemoveTensorMutation(graph, [](Node*) { return true; });
  // 运行 FileCheck，检查计算图中是否不存在 "aten::add_" 操作
  testing::FileCheck().check_not("aten::add_")->run(*graph);
}

// 定义名为 TestInplaceToFunctionalActivation 的测试用例，测试原地操作转换为函数式激活
TEST(TestInplaceToFunctionalActivation, Basic) {
  // 创建一个新的计算图 graph
  auto graph = std::make_shared<Graph>();
  // 创建一个空的无序映射 vmap
  std::unordered_map<std::string, Value*> vmap;
  // 解析给定的 IR 字符串，并将结果填充到计算图 graph 中
  parseIR(
      R"IR(
graph(%x.1 : Tensor):
  %2 : int = prim::Constant[value=1]()
  %x.3 : Tensor = aten::add(%x.1, %2, %2)
  %y : Tensor = aten::relu_(%x.3)
  return (%y))IR",
      &*graph,
      vmap);
  // 将计算图 graph 中的原地激活操作转换为函数式激活
  InplaceToFunctionalActivation(graph);
  // 运行 FileCheck，检查计算图中是否存在 "aten::relu" 操作
  testing::FileCheck().check("aten::relu")->run(*graph);
  // 运行 FileCheck，检查计算图中是否不存在 "aten::relu_" 操作
  testing::FileCheck().check_not("aten::relu_")->run(*graph);
}

// 定义名为 TestRegisterShapeOp 的测试用例，测试注册形状操作
TEST(TestRegisterShapeOp, Basic) {
  // 创建一个新的计算图 graph
  auto graph = std::make_shared<Graph>();
  // 创建一个空的无序映射 vmap
  std::unordered_map<std::string, Value*> vmap;
  // 解析给定的 IR 字符串，并将结果填充到计算图 graph 中
  parseIR(
      R"IR(
graph():
  %2 : int = prim::Constant[value=5]()
  %3: int[] = prim::ListConstruct(%2, %2)
  return (%3))IR",
      &*graph,
      vmap);

  // 创建一个新的计算图 g2
  auto g2 = std::make_shared<Graph>();
  // 解析给定的 IR 字符串，并将结果填充到计算图 g2 中
  parseIR(
      R"IR(
graph():
  %2 : Tensor = prim::MakeTestTensor()
  return (%2))IR",
      &*g2,
      vmap);

  // 获取 g2 中第一个节点的函数模式 schema
  const FunctionSchema& schema = g2->nodes().begin()->schema();
  // 为给定的 schema 注册形状计算图 graph
  torch::jit::RegisterShapeComputeGraphForSchema(schema, graph);
  // 在 g2 上传播形状
  PropagateShapesOnGraph(g2);
  // 运行 FileCheck，检查计算图 g2 中是否包含 "5, 5" 形状信息
  testing::FileCheck().check("5, 5")->run(*g2);
}

// 定义名为 TestFunctionalToInplaceActivation 的测试用例，测试函数式激活转换为原地操作
TEST(TestFunctionalToInplaceActivation, Basic) {
  // 创建一个新的计算图 graph
  auto graph = std::make_shared<Graph>();
  // 创建一个空的无序映射 vmap
  std::unordered_map<std::string, Value*> vmap;
  // 解析给定的 IR 字符串，并将结果填充到计算图 graph 中
  parseIR(
      R"IR(
graph(%x.1 : Tensor):
  %2 : int = prim::Constant[value=1]()
  %x.3 : Tensor = aten::add(%x.1, %2, %2)
  %y : Tensor = aten::relu(%x.3)
  return (%y))IR",
      &*graph,
      vmap);
  // 将计算图 graph 中的函数式激活操作转换为原地操作
  FunctionalToInplaceActivation(graph);
  // 运行 FileCheck，检查计算图中是否存在 "aten::relu_" 操作
  testing::FileCheck().check("aten::relu_")->run(*graph);
  // 运行 FileCheck，检查计算图中是否不存在 "aten::relu(" 操作
  testing::FileCheck().check_not("aten::relu(")->run(*graph);
}

// 定义名为 TestFunctionExecutor 的测试用例，测试简单的函数执行器
TEST(TestFunctionExecutor, SimpleExecutorTest) {
  // 创建一个新的计算图 graph
  auto graph = std::make_shared<Graph>();
  // 解析给定的 IR 字符串，并将结果填充到计算图 graph 中
  parseIR(
      R"IR(
// 定义名为 graph 的函数，参数为一个 Tensor 对象 %x.1
graph(%x.1 : Tensor):
  // 创建一个整数常量 %2，值为 1
  %2 : int = prim::Constant[value=1]()
  // 计算 %x.1 加 %2 的结果，并将结果存储在 %x.3 中
  %x.3 : Tensor = aten::add(%x.1, %2, %2)
  // 对 %x.3 应用 ReLU 激活函数，将结果存储在 %y 中
  %y : Tensor = aten::relu(%x.3)
  // 返回 %y
  return (%y))IR",
      &*graph);
  {
    // 创建一个名为 func 的唯一指针，调用 GraphFunction 构造函数，初始化为 PROFILING 模式
    auto func = std::make_unique<GraphFunction>(
        "name", graph, [](GraphFunction&) {}, ExecutorExecutionMode::PROFILING);
    // 创建一个形状为 {2, 2, 2}，类型为 float 的随机张量 a
    auto a = at::rand({2, 2, 2}, TensorOptions(kCPU).dtype(at::kFloat));
    // 创建一个栈 stack，将张量 a 压入栈中
    Stack stack = {a};
    // 运行 func 执行函数
    func->run(stack);
    // 获取最近执行的优化图 g
    auto g = lastExecutedOptimizedGraph();
    // 运行测试，验证优化图 g 的内容
    testing::FileCheck()
        .check("prim::profile")
        ->check("aten::add")
        ->check("aten::relu")
        ->run(*g);
  }
  {
    // 创建一个名为 func 的唯一指针，调用 GraphFunction 构造函数，初始化为 SIMPLE 模式
    auto func = std::make_unique<GraphFunction>(
        "name", graph, [](GraphFunction&) {}, ExecutorExecutionMode::SIMPLE);
    // 创建一个形状为 {2, 2, 2}，类型为 float 的随机张量 a
    auto a = at::rand({2, 2, 2}, TensorOptions(kCPU).dtype(at::kFloat));
    // 创建一个栈 stack，将张量 a 压入栈中
    Stack stack = {a};
    // 运行 func 执行函数
    func->run(stack);
    // 获取 func 的调试状态下的图形 g
    auto g = func->getDebugState().graph;
    // 运行测试，验证图形 g 的内容
    testing::FileCheck()
        .check_not("prim::profile")
        ->check("aten::add")
        ->check("aten::relu")
        ->run(*g);
  }
}

// 测试函数 TestFunctionExecutor 的 RunDecompositionTest 测试用例
TEST(TestFunctionExecutor, RunDecompositionTest) {
  // 获取名称为 "aten::var(Tensor self, bool unbiased=True) -> Tensor" 的分解执行器 func
  static auto* func = torch::jit::GetDecompositionExecutor(
      "aten::var(Tensor self, bool unbiased=True) -> Tensor");
  // 对于布尔值 unbiased 取值为 true 和 false 的情况，进行循环
  for (bool unbiased : {true, false}) {
    // 创建一个形状为 {4, 4} 的随机输入张量 input
    auto input = at::rand({4, 4});
    // 创建一个栈 stack，将输入张量 input 和 unbiased 布尔值压入栈中
    Stack stack = {input, unbiased};
    // 运行 func 执行函数
    func->run(stack);
    // 弹出栈顶的张量，赋值给变量 out
    at::Tensor out = pop(stack).toTensor();
    // 断言：验证 out 和 input.var(unbiased) 是否在数值上近似相等
    ASSERT_TRUE(at::allclose(out, input.var(unbiased)));
  }
}

// 测试函数 TestShapeGraphLinting 的 Basic 测试用例
TEST(TestShapeGraphLinting, Basic) {
  // 获取已注册的所有形状计算模式的架构 schemas
  auto schemas = RegisteredShapeComputeSchemas();
  // 对于每个架构 schema 进行循环处理
  for (const auto& schema : schemas) {
    // 如果架构的名称是 "aten::arange"，则跳过当前循环
    if (schema->name() == "aten::arange") {
      continue;
    }
    // 创建一个形状计算图 g，根据当前架构 schema
    auto g = shapeComputeGraphForSchema(*schema);
    // 内部断言：验证 g 不为空
    TORCH_INTERNAL_ASSERT(g);
    // 对架构 schema 和其形状计算图 g 进行形状计算图检验
    LintShapeComputeGraph(schema, *g);
  }
}

// 类 Composed，继承自测试框架的 Test 类
class Composed : public ::testing::Test {
 public:
  // 设置测试环境：禁用 LLVM 在 CPU 上的使用
  void SetUp() override {
    torch::jit::tensorexpr::getTEMustUseLLVMOnCPU() = false;
  }
};

// Composed 测试类的测试用例 ComposedOp
TEST_F(Composed, ComposedOp) {
  // 定义一个结构体 WithCPUFuser
  struct WithCPUFuser {
    // 构造函数：根据传入的参数 val，设置和恢复 CPU 融合器的状态
    WithCPUFuser(bool val = true) : cpuFuserEnabled(canFuseOnCPU()) {
      overrideCanFuseOnCPU(val);
    }

    // 析构函数：恢复 CPU 融合器的状态为构造时的状态
    ~WithCPUFuser() {
      overrideCanFuseOnCPU(cpuFuserEnabled);
    }

    // 记录 CPU 融合器的当前状态
    bool cpuFuserEnabled;
  };
}
#ifdef TORCH_ENABLE_LLVM
  // 定义图形描述的字符串，包含两个输入张量的乘法操作，并返回两个输出张量
  const auto graph_string = R"IR(
      graph(%0 : Float(5, 3, strides=[3, 1], device=cpu),
            %1 : Float(5, 3, strides=[1, 5], device=cpu)):
        %2 : Float(5, 3, strides=[3, 1], device=cpu) = aten::mul(%0, %1)
        %3 : Float(5, 3, strides=[3, 1], device=cpu) = aten::mul(%0, %2)
        %4 : Float(5, 3, strides=[3, 1], device=cpu) = aten::mul(%0, %3)
        return (%3, %4))IR";
  
  // 创建共享指针指向一个新的图形对象
  auto graph = std::make_shared<Graph>();
  // 解析图形描述字符串并填充到图形对象中
  parseIR(graph_string, &*graph);

  // 创建随机张量 a 和 b，指定设备为 CPU
  auto a = at::rand({2, 2, 2}, TensorOptions(kCPU).dtype(at::kFloat));
  auto b = at::rand({2, 2, 2}, TensorOptions(kCPU).dtype(at::kFloat))
               .transpose(0, 1);
  
  // 计算参考结果 ref1 和 ref2
  auto ref1 = a * (a * b);
  auto ref2 = a * ref1;

  // 创建一个使用 CPU Fuser 的上下文
  WithCPUFuser g(true);
  
  // 获取当前是否必须在 CPU 上使用 LLVM 的标志，并将其设置为 false
  bool fusable_on_device = torch::jit::tensorexpr::getTEMustUseLLVMOnCPU();
  torch::jit::tensorexpr::getTEMustUseLLVMOnCPU() = false;

  // 对图形进行张量表达式融合，设置最小组大小为 2，添加复合操作并启用动态形状融合
  FuseTensorExprs(
      graph,
      /*min_group_size*/ 2,
      /*add_composed_op*/ true,
      /*fuse_to_dynamic_shapes*/ true);
  
  // 用图形对象和空字符串创建代码对象
  Code code(graph, "");
  
  // 创建解释器状态对象并运行，使用随机张量 a 和 b 作为输入栈
  InterpreterState interpreter{code};
  std::vector<IValue> stack = {a, b};
  interpreter.run(stack);

  // 弹出栈顶的张量作为输出张量 out2 和 out1
  at::Tensor out2 = pop(stack).toTensor();
  at::Tensor out1 = pop(stack).toTensor();
  
  // 使用 ASSERT_TRUE 断言 ref1 与 out1 的所有元素近似相等
  ASSERT_TRUE(at::allclose(ref1, out1));
  
  // 使用 ASSERT_TRUE 断言 ref2 与 out2 的所有元素近似相等
  ASSERT_TRUE(at::allclose(ref2, out2));

  // 创建全为 1 的输入张量 inp_1 和 inp_2，设备为 CPU
  auto inp_1 = at::ones({4, 4}, TensorOptions(kCPU).dtype(at::kFloat));
  auto inp_2 = at::ones({4, 4}, TensorOptions(kCPU).dtype(at::kFloat));
  
  // 重置输入栈为 inp_1、inp_2、a 和 b
  stack = {inp_1, inp_2, a, b};
  
  // 创建另一个解释器状态对象并运行，使用相同的代码对象
  InterpreterState interpreter2{code};
  interpreter2.run(stack);
  
  // 弹出栈顶的张量作为输出张量 out2 和 out1
  out2 = pop(stack).toTensor();
  out1 = pop(stack).toTensor();
  
  // 使用 ASSERT_TRUE 断言 ref1 与 out1 的所有元素近似相等
  ASSERT_TRUE(at::allclose(ref1, out1));
  
  // 使用 ASSERT_TRUE 断言 ref2 与 out2 的所有元素近似相等
  ASSERT_TRUE(at::allclose(ref2, out2));
  
  // 使用 ASSERT_TRUE 断言 inp_1 与 ref2 的所有元素近似相等
  ASSERT_TRUE(at::allclose(inp_1, ref2));
  
  // 使用 ASSERT_TRUE 断言 inp_2 与 ref1 的所有元素近似相等
  ASSERT_TRUE(at::allclose(inp_2, ref1));
  
  // 恢复原来的设备必须使用 LLVM 的标志
  torch::jit::tensorexpr::getTEMustUseLLVMOnCPU() = fusable_on_device;
#endif
}
    def graph():
        # 创建一个空值（NoneType）的常量节点
        %none: NoneType = prim::Constant()
        # 创建一个整数常量节点，值为3，表示张量的维度
        %dim: int = prim::Constant[value=3]()
        # 创建一个整数数组常量节点，表示张量的形状为[3, 3]
        %shape: int[] = prim::ListConstruct(%dim, %dim)
        # 创建一个张量常量节点，表示元素全为1的3x3张量
        %weight: Tensor = aten::ones(%shape, %none, %none, %none, %none)
        # 创建一个浮点数常量节点，值为1.0，表示量化的缩放因子
        %scale: float = prim::Constant[value=1.]()
        # 创建一个整数常量节点，值为0，表示量化的零点
        %zero_point: int = prim::Constant[value=0]()
        # 创建一个整数常量节点，值为12，表示数据类型（dtype）为int8
        %dtype: int = prim::Constant[value=12]()
        # 使用给定的量化参数对权重张量进行量化
        %weight_q: Tensor = aten::quantize_per_tensor(%weight, %scale, %zero_point, %dtype)
        # 使用量化后的权重张量和空值创建量化线性层的打包参数
        %params: __torch__.torch.classes.quantized.LinearPackedParamsBase = quantized::linear_prepack(%weight_q, %none)
        # 返回量化线性层的打包参数
        return (%params)
#endif



#endif

这行代码用于结束一个条件编译指令块，通常与 `#ifdef` 或 `#if` 配合使用，表示条件编译的结束点。


}



}

这是一个 C++ 的代码块结束标记，用于结束一个函数、循环、或者其他代码块。


} // namespace jit
} // namespace torch



} // namespace jit
} // namespace torch

这两行代码用于结束命名空间 `jit` 和 `torch`，用来确保代码中定义的类、函数、变量等不会污染全局命名空间，增加了代码的模块化和可维护性。
```