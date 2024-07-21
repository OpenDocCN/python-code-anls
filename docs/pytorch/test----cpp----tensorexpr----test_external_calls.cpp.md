# `.\pytorch\test\cpp\tensorexpr\test_external_calls.cpp`

```py
#include <gtest/gtest.h>

#include <test/cpp/tensorexpr/test_base.h>

#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>
#include <torch/csrc/jit/passes/tensorexpr_fuser.h>
#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/tensorexpr/kernel.h>

#include <test/cpp/tensorexpr/test_utils.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <torch/csrc/jit/runtime/symbolic_shape_registry.h>
#include <torch/csrc/jit/tensorexpr/eval.h>
#include <torch/csrc/jit/tensorexpr/external_functions_registry.h>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_printer.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/llvm_codegen.h>
#include <torch/csrc/jit/tensorexpr/loopnest.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>

#include <torch/csrc/jit/testing/file_check.h>
#include <torch/jit.h>

#include <ATen/NativeFunctions.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/native/xnnpack/OpContext.h>

namespace torch {
namespace jit {
using namespace torch::jit::tensorexpr;

// 定义测试用例 ExternalCall.Conv1d_float，用于测试 Conv1d 的正确性
TEST(ExternalCall, Conv1d_float) {
  // 定义输入缓存 Input，大小为 {1, 100, 115}，数据类型为 kFloat
  BufHandle Input("Input", {1, 100, 115}, kFloat);
  // 定义权重缓存 Weight，大小为 {100, 1, 7}，数据类型为 kFloat
  BufHandle Weight("Weight", {100, 1, 7}, kFloat);
  // 定义偏置缓存 Bias，大小为 {100}，数据类型为 kFloat
  BufHandle Bias("Bias", {100}, kFloat);
  // 定义结果缓存 ResultBuf，大小为 {1, 100, 115}，数据类型为 kFloat
  BufHandle ResultBuf("Result", {1, 100, 115}, kFloat);
  // 定义卷积操作的步长、填充、膨胀、分组数
  int64_t stride = 1;
  int64_t pad = 3;
  int64_t dilation = 1;
  int64_t groups = 100;

  // 创建结果 Tensor 对象，通过 ExternalCall 调用 nnc_aten_conv1d 函数
  Tensor Result = Tensor(
      ResultBuf.node(),
      ExternalCall::make(
          ResultBuf,
          "nnc_aten_conv1d",
          {Input, Weight, Bias},
          {stride, pad, dilation, groups}));

  // 创建循环嵌套对象 l，准备进行代码生成前的准备工作
  LoopNest l({Result});
  l.prepareForCodegen();
  // 简化生成的循环嵌套结构
  l.simplify();

  // 创建 TensorOptions 用于指定张量的属性，这里指定为 kFloat 类型，CPU 设备，不需要梯度计算
  auto options = at::TensorOptions()
                     .dtype(at::kFloat)
                     .layout(at::kStrided)
                     .device(at::kCPU)
                     .requires_grad(false);
  // 创建输入张量 input，尺寸为 {1, 100, 115}，每个元素值为 5.0
  at::Tensor input = at::ones({1, 100, 115}, options) * 5.f;
  // 创建权重张量 weight，尺寸为 {100, 1, 7}，每个元素值为 6.0
  at::Tensor weight = at::ones({100, 1, 7}, options) * 6.f;
  // 创建偏置张量 bias，尺寸为 {100}，每个元素值为 11.0
  at::Tensor bias = at::ones({100}, options) * 11.f;
  // 计算基准结果张量 ref，使用 PyTorch 的 conv1d 函数计算
  at::Tensor ref =
      at::conv1d(input, weight, bias, {stride}, {pad}, {dilation}, groups);

  // 定义 nnc_result 用于存储 LLVM 生成的结果
  at::Tensor nnc_result;
  // 创建输入数据缓存 input_buf，weight_buf，bias_buf，result_buf
  std::vector<float> input_buf(1 * 100 * 115, 5.f);
  std::vector<float> weight_buf(100 * 1 * 7, 6.f);
  std::vector<float> bias_buf(100, 11.f);
  std::vector<float> result_buf(1 * 100 * 115, -1.f);

  // 如果编译开启了 LLVM 支持
#ifdef TORCH_ENABLE_LLVM
  // 创建 LLVMCodeGen 对象，用于生成 LLVM 代码
  LLVMCodeGen llvm_codegen(l.root_stmt(), {Input, Weight, Bias, Result});
  // 调用 LLVMCodeGen 对象的 call 函数，生成结果数据到 result_buf 中
  llvm_codegen.call({input_buf, weight_buf, bias_buf, result_buf});
  // 将 result_buf 转换为张量 nnc_result
  nnc_result = at::from_blob(result_buf.data(), {1, 100, 115}, options);
  // 使用 ASSERT_TRUE 判断 nnc_result 是否与 ref 张量相似
  ASSERT_TRUE(at::allclose(nnc_result, ref));
#endif

  // 创建 SimpleIREvaluator 对象，用于执行简化后的 IR 表达式
  SimpleIREvaluator ir_eval(l.root_stmt(), {Input, Weight, Bias, Result});
  // 调用 ir_eval 对象的 call 函数，生成结果数据到 result_buf 中
  ir_eval.call({input_buf, weight_buf, bias_buf, result_buf});
  // 将 result_buf 转换为张量 nnc_result
  nnc_result = at::from_blob(result_buf.data(), {1, 100, 115}, options);
  // 使用 ASSERT_TRUE 判断 nnc_result 是否与 ref 张量相似
  ASSERT_TRUE(at::allclose(nnc_result, ref));
}

} // namespace jit
} // namespace torch
TEST(ExternalCall, Conv1d_int) {
  // 定义输入缓冲区及其形状，数据类型为整数
  BufHandle Input("Input", {1, 100, 115}, kInt);
  // 定义权重缓冲区及其形状，数据类型为整数
  BufHandle Weight("Weight", {100, 1, 7}, kInt);
  // 定义偏置缓冲区及其形状，数据类型为整数
  BufHandle Bias("Bias", {100}, kInt);
  // 定义结果缓冲区及其形状，数据类型为整数
  BufHandle ResultBuf("Result", {1, 100, 115}, kInt);
  // 设置卷积的步长、填充、扩张和组数
  int64_t stride = 1;
  int64_t pad = 3;
  int64_t dilation = 1;
  int64_t groups = 100;

  // 创建一个张量对象，表示卷积的输出结果
  Tensor Result = Tensor(
      ResultBuf.node(),
      ExternalCall::make(
          ResultBuf,
          "nnc_aten_conv1d",
          {Input, Weight, Bias},  // 输入的缓冲区及其形状
          {stride, pad, dilation, groups}));  // 卷积参数

  // 创建一个循环嵌套对象，将用于生成代码
  LoopNest l({Result});
  // 准备循环嵌套以便于代码生成
  l.prepareForCodegen();
  // 简化循环嵌套结构
  l.simplify();

  // 设置张量的选项：整数类型、跨步布局、CPU设备、不需要梯度
  auto options = at::TensorOptions()
                     .dtype(at::kInt)
                     .layout(at::kStrided)
                     .device(at::kCPU)
                     .requires_grad(false);
  // 创建输入张量，填充值为5
  at::Tensor input = at::ones({1, 100, 115}, options) * 5;
  // 创建权重张量，填充值为6
  at::Tensor weight = at::ones({100, 1, 7}, options) * 6;
  // 创建偏置张量，填充值为11
  at::Tensor bias = at::ones({100}, options) * 11;
  // 创建参考结果张量，使用 PyTorch 的 conv1d 函数计算
  at::Tensor ref =
      at::conv1d(input, weight, bias, {stride}, {pad}, {dilation}, groups);

  // 定义用于存储 NNCompiler 生成结果的张量
  at::Tensor nnc_result;
  // 创建输入缓冲区，并用值5填充
  std::vector<int32_t> input_buf(1 * 100 * 115, 5);
  // 创建权重缓冲区，并用值6填充
  std::vector<int32_t> weight_buf(100 * 1 * 7, 6);
  // 创建偏置缓冲区，并用值11填充
  std::vector<int32_t> bias_buf(100, 11);
  // 创建结果缓冲区，初始值为-1
  std::vector<int32_t> result_buf(1 * 100 * 115, -1);

#ifdef TORCH_ENABLE_LLVM
  // 如果启用 LLVM 支持，则创建 LLVMCodeGen 对象
  LLVMCodeGen llvm_codegen(l.root_stmt(), {Input, Weight, Bias, Result});
  // 调用 LLVMCodeGen 对象生成代码，传入输入、权重、偏置和结果缓冲区
  llvm_codegen.call({input_buf, weight_buf, bias_buf, result_buf});
  // 将生成的结果转换为 PyTorch 张量
  nnc_result = at::from_blob(result_buf.data(), {1, 100, 115}, options);
  // 使用断言确保 NNCompiler 生成的结果与参考结果一致
  ASSERT_TRUE(at::allclose(nnc_result, ref));
#endif

  // 创建 SimpleIREvaluator 对象，用于评估 IR 代码
  SimpleIREvaluator ir_eval(l.root_stmt(), {Input, Weight, Bias, Result});
  // 调用 SimpleIREvaluator 对象评估 IR 代码，传入输入、权重、偏置和结果缓冲区
  ir_eval.call({input_buf, weight_buf, bias_buf, result_buf});
  // 将评估的结果转换为 PyTorch 张量
  nnc_result = at::from_blob(result_buf.data(), {1, 100, 115}, options);
  // 使用断言确保评估的结果与参考结果一致
  ASSERT_TRUE(at::allclose(nnc_result, ref));
}

TEST(ExternalCall, Conv1d_nobias_noargs) {
  // 定义输入缓冲区及其形状，数据类型为浮点数
  BufHandle Input("Input", {1, 1, 115}, kFloat);
  // 定义权重缓冲区及其形状，数据类型为浮点数
  BufHandle Weight("Weight", {10, 1, 7}, kFloat);
  // 定义结果缓冲区及其形状，数据类型为浮点数
  BufHandle ResultBuf("Result", {1, 10, 109}, kFloat);

  // 创建一个张量对象，表示卷积的输出结果，没有偏置和额外参数
  Tensor Result = Tensor(
      ResultBuf.node(),
      ExternalCall::make(ResultBuf, "nnc_aten_conv1d", {Input, Weight}, {}));

  // 创建一个循环嵌套对象，将用于生成代码
  LoopNest l({Result});
  // 准备循环嵌套以便于代码生成
  l.prepareForCodegen();
  // 简化循环嵌套结构
  l.simplify();

  // 设置张量的选项：浮点数类型、跨步布局、CPU设备、不需要梯度
  auto options = at::TensorOptions()
                     .dtype(at::kFloat)
                     .layout(at::kStrided)
                     .device(at::kCPU)
                     .requires_grad(false);
  // 创建输入张量，填充值为5.0
  at::Tensor input = at::ones({1, 1, 115}, options) * 5.f;
  // 创建权重张量，填充值为6.0
  at::Tensor weight = at::ones({10, 1, 7}, options) * 6.f;
  // 创建参考结果张量，使用 PyTorch 的 conv1d 函数计算
  at::Tensor ref = at::conv1d(input, weight);

  // 定义用于存储 NNCompiler 生成结果的张量
  at::Tensor nnc_result;
  // 创建输入缓冲区，并用值5.0填充
  std::vector<float> input_buf(1 * 1 * 115, 5.f);
  // 创建权重缓冲区，并用值6.0填充
  std::vector<float> weight_buf(10 * 1 * 7, 6.f);
  // 创建结果缓冲区，初始值为-1.0
  std::vector<float> result_buf(1 * 10 * 109, -1.f);
#ifdef TORCH_ENABLE_LLVM
  // 如果定义了 TORCH_ENABLE_LLVM，使用 LLVMCodeGen 来生成 LLVM IR，并执行外部调用
  LLVMCodeGen llvm_codegen(l.root_stmt(), {Input, Weight, Result});

  // 调用 LLVMCodeGen 对象来生成 LLVM IR，并传入输入、权重和结果缓冲区
  llvm_codegen.call({input_buf, weight_buf, result_buf});

  // 从结果缓冲区创建张量，形状为 {1, 10, 109}，使用给定的选项
  nnc_result = at::from_blob(result_buf.data(), {1, 10, 109}, options);

  // 断言新生成的张量与参考结果 ref 在数值上相近
  ASSERT_TRUE(at::allclose(nnc_result, ref));
#endif

// 使用 SimpleIREvaluator 来执行简化后的 IR，并进行外部调用
SimpleIREvaluator ir_eval(l.root_stmt(), {Input, Weight, Result});

// 调用 SimpleIREvaluator 对象来执行简化后的 IR，并传入输入、权重和结果缓冲区
ir_eval.call({input_buf, weight_buf, result_buf});

// 从结果缓冲区创建张量，形状为 {1, 10, 109}，使用给定的选项
nnc_result = at::from_blob(result_buf.data(), {1, 10, 109}, options);

// 断言新生成的张量与参考结果 ref 在数值上相近
ASSERT_TRUE(at::allclose(nnc_result, ref));
}
TEST(ExternalCall, Conv2d_int) {
  // 定义一个名为 Conv2d_int 的测试用例，测试整数类型的卷积操作

  BufHandle Input("Input", {1, 3, 224, 224}, kInt);
  // 创建一个名为 Input 的缓冲区句柄，表示输入张量，维度为 [1, 3, 224, 224]，数据类型为整数

  BufHandle Weight("Weight", {16, 3, 3, 3}, kInt);
  // 创建一个名为 Weight 的缓冲区句柄，表示卷积核张量，维度为 [16, 3, 3, 3]，数据类型为整数

  BufHandle Bias("Bias", {16}, kInt);
  // 创建一个名为 Bias 的缓冲区句柄，表示偏置张量，维度为 [16]，数据类型为整数

  BufHandle ResultBuf("Result", {1, 16, 112, 112}, kInt);
  // 创建一个名为 Result 的缓冲区句柄，表示卷积结果张量，维度为 [1, 16, 112, 112]，数据类型为整数

  int64_t stride = 2;
  // 设置卷积的步长为 2
  int64_t pad = 1;
  // 设置卷积的填充大小为 1
  int64_t dilation = 1;
  // 设置卷积的扩展大小为 1
  int64_t groups = 1;
  // 设置卷积的分组大小为 1

  Tensor Result = Tensor(
      ResultBuf.node(),
      ExternalCall::make(
          ResultBuf,
          "nnc_aten_conv2d",
          {Input, Weight, Bias},
          {stride, stride, pad, pad, dilation, dilation, groups}));
  // 创建一个名为 Result 的张量对象，表示卷积操作的结果，通过 ExternalCall::make 调用 nnc_aten_conv2d 函数，
  // 并传入 Input, Weight, Bias 作为参数，同时传入步长、填充、扩展和分组信息

  LoopNest l({Result});
  // 创建一个循环嵌套对象 l，其中包含 Result 张量

  l.prepareForCodegen();
  // 准备循环嵌套对象 l 进行代码生成

  l.simplify();
  // 简化循环嵌套对象 l

  auto options = at::TensorOptions()
                     .dtype(at::kInt)
                     .layout(at::kStrided)
                     .device(at::kCPU)
                     .requires_grad(false);
  // 创建张量选项 options，指定数据类型为整数，布局为 Strided，设备为 CPU，不需要梯度

  at::Tensor input = at::ones({1, 3, 224, 224}, options) * 5;
  // 创建一个大小为 [1, 3, 224, 224] 的张量 input，元素为 5

  at::Tensor weight = at::ones({16, 3, 3, 3}, options) * 6;
  // 创建一个大小为 [16, 3, 3, 3] 的张量 weight，元素为 6

  at::Tensor bias = at::ones({16}, options) * 11;
  // 创建一个大小为 [16] 的张量 bias，元素为 11

  at::Tensor ref = at::conv2d(
      input,
      weight,
      bias,
      {stride, stride},
      {pad, pad},
      {dilation, dilation},
      groups);
  // 创建一个参考结果张量 ref，通过调用 PyTorch 的 conv2d 函数进行卷积计算

  at::Tensor nnc_result;
  // 创建一个变量 nnc_result，用于存储 NNCompiler 计算的结果张量

  std::vector<int32_t> input_buf(1 * 3 * 224 * 224, 5);
  // 创建一个大小为 1 * 3 * 224 * 224 的整数向量 input_buf，元素初始化为 5

  std::vector<int32_t> weight_buf(16 * 3 * 3 * 3, 6);
  // 创建一个大小为 16 * 3 * 3 * 3 的整数向量 weight_buf，元素初始化为 6

  std::vector<int32_t> bias_buf(16, 11);
  // 创建一个大小为 16 的整数向量 bias_buf，元素初始化为 11

  std::vector<int32_t> result_buf(1 * 16 * 112 * 112, -1);
  // 创建一个大小为 1 * 16 * 112 * 112 的整数向量 result_buf，元素初始化为 -1

#ifdef TORCH_ENABLE_LLVM
  LLVMCodeGen llvm_codegen(l.root_stmt(), {Input, Weight, Bias, Result});
  // 如果启用 LLVM，则创建 LLVMCodeGen 对象，用于生成 LLVM IR 代码，传入 Input, Weight, Bias, Result 句柄

  llvm_codegen.call({input_buf, weight_buf, bias_buf, result_buf});
  // 调用 LLVMCodeGen 对象的 call 方法，传入输入、权重、偏置和结果缓冲区

  nnc_result = at::from_blob(result_buf.data(), {1, 16, 112, 112}, options);
  // 将 LLVM 计算的结果存储到 nnc_result 中，通过 from_blob 方法将 result_buf 转换为张量

  ASSERT_TRUE(at::allclose(nnc_result, ref));
  // 使用 PyTorch 的 allclose 函数检查 nnc_result 是否与 ref 张量接近
#endif

  SimpleIREvaluator ir_eval(l.root_stmt(), {Input, Weight, Bias, Result});
  // 创建 SimpleIREvaluator 对象，用于执行简化后的 IR，传入 Input, Weight, Bias, Result 句柄

  ir_eval.call({input_buf, weight_buf, bias_buf, result_buf});
  // 调用 SimpleIREvaluator 对象的 call 方法，传入输入、权重、偏置和结果缓冲区

  nnc_result = at::from_blob(result_buf.data(), {1, 16, 112, 112}, options);
  // 将简化后 IR 的结果存储到 nnc_result 中，通过 from_blob 方法将 result_buf 转换为张量

  ASSERT_TRUE(at::allclose(nnc_result, ref));
  // 使用 PyTorch 的 allclose 函数检查 nnc_result 是否与 ref 张量接近
}

TEST(ExternalCall, Conv2d_nobias_noargs) {
  BufHandle Input("Input", {1, 16, 112, 112}, kFloat);
  // 创建一个名为 Input 的缓冲区句柄，表示输入张量，维度为 [1, 16, 112, 112]，数据类型为浮点数

  BufHandle Weight("Weight", {16, 16, 1, 1}, kFloat);
  // 创建一个名为 Weight 的缓冲区句柄，表示卷积核张量，维度为 [16, 16, 1, 1]，数据类型为浮点数

  BufHandle ResultBuf("Result", {1, 16, 112, 112}, kFloat);
  // 创建一个名为 Result 的缓冲区句柄，表示卷积结果张量，维度为 [1, 16, 112, 112]，数据类型为浮点数

  Tensor Result = Tensor(
      ResultBuf.node(),
      ExternalCall::make(ResultBuf, "nnc_aten_conv2d", {Input, Weight}, {}));
  // 创建一个名为 Result 的张量对象，表示卷积操作的结果，通过 ExternalCall::make 调用 nnc_aten_conv2d 函数，
  // 并传入 Input, Weight 作为参数，不传递额外的参数

  LoopNest l({Result});
  // 创建一个循环嵌套对象 l，其中包含 Result 张量

  l.prepareForCodegen();
  // 准备循环嵌套对象 l 进行代码生成

  l.simplify();
  // 简化循环嵌套对象 l

  auto options = at::TensorOptions()
                     .dtype(at::kFloat)
                     .layout(at::kStrided)
                     .device(at::kCPU)
                     .requires_grad(false);
  // 创建张量选项 options，指定数据类型为浮点数，布局为 Strided，设备为 CPU
#ifdef TORCH_ENABLE_LLVM
  // 如果编译器支持 LLVM，则使用 LLVMCodeGen 进行代码生成，传入需要处理的语句和缓冲区列表
  LLVMCodeGen llvm_codegen(l.root_stmt(), {Input, Mat1, Mat2, Result});

  // 调用 LLVMCodeGen 对象的 call 方法，传入输入、权重和结果缓冲区的数据
  llvm_codegen.call({input_buf, mat1_buf, mat2_buf, result_buf});
  // 从结果缓冲区创建 Tensor 对象，尺寸为 {100, 16, 112, 112}，使用指定的选项
  nnc_result = at::from_blob(result_buf.data(), {100, 300}, options);
  // 断言 nnc_result 与参考结果 ref 的各元素是否接近
  ASSERT_TRUE(at::allclose(nnc_result, ref));
#endif

// 使用 SimpleIREvaluator 进行简单的 IR 评估，传入需要处理的语句和缓冲区列表
SimpleIREvaluator ir_eval(l.root_stmt(), {Input, Mat1, Mat2, Result});

// 调用 SimpleIREvaluator 对象的 call 方法，传入输入、权重和结果缓冲区的数据
ir_eval.call({input_buf, mat1_buf, mat2_buf, result_buf});
// 从结果缓冲区创建 Tensor 对象，尺寸为 {100, 16, 112, 112}，使用指定的选项
nnc_result = at::from_blob(result_buf.data(), {100, 300}, options);
// 断言 nnc_result 与参考结果 ref 的各元素是否接近
ASSERT_TRUE(at::allclose(nnc_result, ref));
}
TEST(ExternalCall, MaxReduction) {
  // 定义输入缓冲区，形状为 {1, 115, 152}，数据类型为 float
  BufHandle Input("Input", {1, 115, 152}, kFloat);
  // 定义结果缓冲区，形状为 {1, 152}，数据类型为 float
  BufHandle ResultBuf("Result", {1, 152}, kFloat);
  // 指定进行最大值约简的维度为 1
  int64_t dim = 1;
  // 指定是否保持维度的标志为 false
  bool keep_dim = false;

  // 创建结果张量，使用 ExternalCall 调用名为 "nnc_aten_max_red" 的外部函数
  Tensor Result = Tensor(
      ResultBuf.node(),
      ExternalCall::make(
          ResultBuf, "nnc_aten_max_red", {Input}, {dim, (int64_t)keep_dim}));
  
  // 创建循环嵌套对象，包含结果张量
  LoopNest l({Result});
  // 准备进行代码生成前的准备工作
  l.prepareForCodegen();
  // 简化循环嵌套结构
  l.simplify();

  // 设置张量选项，指定为 float 类型，布局为 kStrided，设备为 CPU，不需要梯度
  auto options = at::TensorOptions()
                     .dtype(at::kFloat)
                     .layout(at::kStrided)
                     .device(at::kCPU)
                     .requires_grad(false);

  // 创建输入张量，形状为 {1, 115, 152}，并填充值为 5.0
  at::Tensor input = at::ones({1, 115, 152}, options) * 5.f;
  // 计算参考结果，对输入张量进行最大值约简，指定维度为 1，保持维度为 false
  at::Tensor ref = std::get<0>(at::max(input, dim, keep_dim));

  // 声明用于存储 NNCompiler 结果的张量
  at::Tensor nnc_result;
  // 创建输入缓冲区，存储大小为 1 * 115 * 152 的 float 数据，初始化为 5.0
  std::vector<float> input_buf(1 * 115 * 152, 5.f);
  // 创建结果缓冲区，存储大小为 1 * 152 的 float 数据，初始化为 -1.0
  std::vector<float> result_buf(1 * 152, -1.f);

#ifdef TORCH_ENABLE_LLVM
  // 如果启用了 LLVM 支持，则创建 LLVMCodeGen 对象，传入输入和结果缓冲区
  LLVMCodeGen llvm_codegen(l.root_stmt(), {Input, Result});
  // 调用 LLVMCodeGen 的 call 方法，传入输入和结果缓冲区
  llvm_codegen.call({input_buf, result_buf});
  // 将结果从缓冲区创建为张量，形状为 {1, 152}，数据类型为 float
  nnc_result = at::from_blob(result_buf.data(), {1, 152}, options);
  // 使用 ASSERT_TRUE 检查 NNCompiler 生成的结果与参考结果是否接近
  ASSERT_TRUE(at::allclose(nnc_result, ref));
#endif

  // 创建 SimpleIREvaluator 对象，传入输入和结果缓冲区
  SimpleIREvaluator ir_eval(l.root_stmt(), {Input, Result});
  // 调用 SimpleIREvaluator 的 call 方法，传入输入和结果缓冲区
  ir_eval.call({input_buf, result_buf});
  // 将结果从缓冲区创建为张量，形状为 {1, 152}，数据类型为 float
  nnc_result = at::from_blob(result_buf.data(), {1, 152}, options);
  // 使用 ASSERT_TRUE 检查 IR 生成器生成的结果与参考结果是否接近
  ASSERT_TRUE(at::allclose(nnc_result, ref));
}
#ifdef USE_XNNPACK
// 定义测试函数，验证使用 XNNPACK 预打包线性运算的功能
TEST(ExternalCall, Prepacked_Linear_float) {
  using namespace at::native::xnnpack;

  // 定义输入和输出缓冲区对象
  BufHandle Input("Input", {100, 200}, kFloat);
  BufHandle ResultBuf("Result", {100, 300}, kFloat);

  // 计算参考结果，使用 at::linear 函数
  auto options = at::TensorOptions()
                     .dtype(at::kFloat)
                     .layout(at::kStrided)
                     .device(at::kCPU)
                     .requires_grad(false);
  // 创建输入张量
  at::Tensor input =
      at::linspace(-10.0, 10.0, 100 * 200, options).resize_({100, 200});
  // 创建权重张量
  at::Tensor weight =
      at::linspace(-10.0, 10.0, 300 * 200, options).resize_({300, 200});
  // 创建偏置张量
  at::Tensor bias = at::linspace(-10.0, 10.0, 300, options);
  // 计算参考结果
  at::Tensor ref = at::linear(input, weight, bias);

  // 创建预打包的 XNNPACK 上下文对象
  auto linear_clamp_prepack_op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("prepacked::linear_clamp_prepack", "")
          .typed<c10::intrusive_ptr<LinearOpContext>(
              at::Tensor,
              std::optional<at::Tensor>,
              const std::optional<at::Scalar>&,
              const std::optional<at::Scalar>&)>();
  auto prepacked = linear_clamp_prepack_op.call(
      weight, bias, std::optional<at::Scalar>(), std::optional<at::Scalar>());

  // 定义虚拟的预打包对象
  BufHandle DummyPrepacked("DummyPrepacked", {1}, kFloat);
  // 创建结果张量
  Tensor Result = Tensor(
      ResultBuf.node(),
      ExternalCall::make(
          ResultBuf,
          "nnc_prepacked_linear_clamp_run",
          {Input, DummyPrepacked},
          {}));
  // 创建循环嵌套对象
  LoopNest l({Result});
  l.prepareForCodegen();
  l.simplify();

  at::Tensor nnc_result;
  // 创建输入和结果的缓冲区
  std::vector<float> input_buf(
      input.data_ptr<float>(), input.data_ptr<float>() + 100 * 200);
  std::vector<float> result_buf(100 * 300, -1.f);

#ifdef TORCH_ENABLE_LLVM
  // 使用 LLVM 进行代码生成
  LLVMCodeGen llvm_codegen(l.root_stmt(), {Input, DummyPrepacked, Result});

  // 调用 LLVM 生成的代码进行计算
  llvm_codegen.call({input_buf, prepacked.get(), result_buf});
  // 从结果缓冲区创建张量对象
  nnc_result = at::from_blob(result_buf.data(), {100, 300}, options);
  // 验证生成的结果与参考结果的接近程度
  ASSERT_TRUE(at::allclose(nnc_result, ref));
#endif

  // 使用简单的 IR 评估器进行评估
  SimpleIREvaluator ir_eval(l.root_stmt(), {Input, DummyPrepacked, Result});

  // 调用简单的 IR 评估器计算结果
  ir_eval.call({input_buf, prepacked.get(), result_buf});
  // 从结果缓冲区创建张量对象
  nnc_result = at::from_blob(result_buf.data(), {100, 300}, options);
  // 验证评估器计算的结果与参考结果的接近程度
  ASSERT_TRUE(at::allclose(nnc_result, ref));
}
#endif
TEST(ExternalCall, Prepacked_Conv2d_float) {
  using namespace at::native::xnnpack;

  // 定义输入和输出缓冲区
  BufHandle Input("Input", {1, 3, 224, 224}, kFloat);
  BufHandle ResultBuf("Result", {1, 16, 112, 112}, kFloat);

  // 定义卷积操作的参数
  int64_t stride = 2;
  int64_t pad = 1;
  int64_t dilation = 1;
  int64_t groups = 1;

  // 使用 at::conv2d 计算参考结果
  auto options = at::TensorOptions()
                     .dtype(at::kFloat)
                     .layout(at::kStrided)
                     .device(at::kCPU)
                     .requires_grad(false);
  at::Tensor input = at::linspace(-10.0, 10.0, 1 * 3 * 224 * 224, options)
                         .resize_({1, 3, 224, 224});
  at::Tensor weight =
      at::linspace(-10.0, 10.0, 16 * 3 * 3 * 3, options).resize_({16, 3, 3, 3});
  at::Tensor bias = at::linspace(-10.0, 10.0, 16, options);
  at::Tensor ref = at::conv2d(
      input,
      weight,
      bias,
      {stride, stride},
      {pad, pad},
      {dilation, dilation},
      groups);

  // 创建预打包的 xnnpack 上下文对象
  auto conv2d_clamp_prepack_op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("prepacked::conv2d_clamp_prepack", "")
          .typed<c10::intrusive_ptr<Conv2dOpContext>(
              at::Tensor,
              std::optional<at::Tensor>,
              std::vector<int64_t>,
              std::vector<int64_t>,
              std::vector<int64_t>,
              int64_t,
              const std::optional<at::Scalar>&,
              const std::optional<at::Scalar>&)>();
  auto prepacked = conv2d_clamp_prepack_op.call(
      weight,
      bias,
      {stride, stride},
      {pad, pad},
      {dilation, dilation},
      groups,
      std::optional<at::Scalar>(),
      std::optional<at::Scalar>());

  // 定义一个用于存储预打包结果的虚拟缓冲区
  BufHandle DummyPrepacked("DummyPrepacked", {1}, kFloat);

  // 创建 Tensor 对象，该对象表示外部调用的运行
  Tensor Result = Tensor(
      ResultBuf.node(),
      ExternalCall::make(
          ResultBuf,
          "nnc_prepacked_conv2d_clamp_run",
          {Input, DummyPrepacked},
          {}));

  // 准备循环嵌套结构以进行代码生成
  LoopNest l({Result});
  l.prepareForCodegen();
  l.simplify();

  // 创建 LLVM 代码生成器对象，并进行调用
  at::Tensor nnc_result;
  std::vector<float> input_buf(
      input.data_ptr<float>(), input.data_ptr<float>() + 1 * 3 * 224 * 224);
  std::vector<float> result_buf(1 * 16 * 112 * 112, -1.f);

#ifdef TORCH_ENABLE_LLVM
  LLVMCodeGen llvm_codegen(l.root_stmt(), {Input, DummyPrepacked, Result});

  // 调用 LLVM 代码生成器，生成结果
  llvm_codegen.call({input_buf, prepacked.get(), result_buf});
  nnc_result = at::from_blob(result_buf.data(), {1, 16, 112, 112}, options);
  ASSERT_TRUE(at::allclose(nnc_result, ref, 1e-03, 1e-03));
#endif

  // 创建简单的 IR 评估器对象，并进行调用
  SimpleIREvaluator ir_eval(l.root_stmt(), {Input, DummyPrepacked, Result});

  // 调用 IR 评估器，生成结果
  ir_eval.call({input_buf, prepacked.get(), result_buf});
  nnc_result = at::from_blob(result_buf.data(), {1, 16, 112, 112}, options);
  ASSERT_TRUE(at::allclose(nnc_result, ref, 1e-03, 1e-03));
}

#endif // USE_XNNPACK
TEST(ExternalCall, BinaryFloat) {
  // 定义类型别名，表示一个接受两个张量并返回张量的函数
  using TensorFunc = std::function<at::Tensor(at::Tensor, at::Tensor)>;
  // 定义测试用例的元组类型，包含输入张量形状、输出张量形状、测试函数、外部调用名称
  using Test = std::tuple<
      std::vector<int64_t>,
      std::vector<int64_t>,
      std::vector<int64_t>,
      TensorFunc,
      std::string>;
  // 初始化测试用例容器
  std::vector<Test> tests = {};
  // 添加第一个测试用例到容器
  tests.push_back(
      Test{{100, 200}, {200, 300}, {100, 300}, at::matmul, "nnc_aten_matmul"});
  // 添加第二个测试用例到容器
  tests.push_back(Test{{100, 300}, {300}, {100}, at::mv, "nnc_aten_mv"});
  // 添加第三个测试用例到容器
  tests.push_back(
      Test{{100, 200}, {200, 300}, {100, 300}, at::mm, "nnc_aten_mm"});
  // 遍历所有测试用例
  for (auto curTest : tests) {
    // 分解当前测试用例的元组成员
    std::vector<int64_t> aShape, bShape, resShape;
    TensorFunc torchFunc;
    std::string externCallName;
    std::tie(aShape, bShape, resShape, torchFunc, externCallName) = curTest;
    // 将整数向量转换为 ExprHandle 向量的函数
    auto toExprHandleVec = [](std::vector<int64_t> v) {
      auto intV = std::vector<int>(v.begin(), v.end());
      return std::vector<ExprHandle>(intV.begin(), intV.end());
    };
    // 创建张量缓存对象 A、B、Result
    BufHandle A("A", toExprHandleVec(aShape), kFloat);
    BufHandle B("B", toExprHandleVec(bShape), kFloat);
    BufHandle ResultBuf("Result", toExprHandleVec(resShape), kFloat);

    // 创建 Tensor 对象，表示对外部函数的调用
    Tensor Result = Tensor(
        ResultBuf.node(),
        ExternalCall::make(ResultBuf, externCallName, {A, B}, {}));
    // 创建循环嵌套对象 l，并进行代码生成前的准备
    LoopNest l({Result});
    l.prepareForCodegen();
    l.simplify();

    // 创建 TensorOptions 用于创建 Torch 张量
    auto options = at::TensorOptions()
                       .dtype(at::kFloat)
                       .layout(at::kStrided)
                       .device(at::kCPU)
                       .requires_grad(false);
    // 创建张量 a、b、ref，分别表示初始化为 5.0 的张量 a 和 b，以及通过 torchFunc 计算的参考结果张量
    at::Tensor a = at::ones(c10::IntArrayRef(aShape), options) * 5.f;
    at::Tensor b = at::ones(c10::IntArrayRef(bShape), options) * 6.f;
    at::Tensor ref = torchFunc(a, b);

    // 计算整数向量的乘积的 lambda 函数
    auto prod = [](std::vector<int64_t> v) {
      // NOLINTNEXTLINE(modernize-use-transparent-functors)
      return std::accumulate(v.begin(), v.end(), 1, std::multiplies<int64_t>());
    };

    // 创建用于存储计算结果的张量 nnc_result，以及用于存储缓冲区数据的向量 a_buf、b_buf、result_buf
    at::Tensor nnc_result;
    std::vector<float> a_buf(prod(aShape), 5.f);
    std::vector<float> b_buf(prod(bShape), 6.f);
    std::vector<float> result_buf(prod(resShape), -1.f);

    // 条件编译，当开启 LLVM 支持时执行以下代码块
#ifdef TORCH_ENABLE_LLVM
    // 使用 LLVMCodeGen 对象 llvm_codegen 进行 LLVM 代码生成
    LLVMCodeGen llvm_codegen(l.root_stmt(), {A, B, Result});

    // 调用 LLVMCodeGen 对象的 call 方法进行计算，填充 result_buf
    llvm_codegen.call({a_buf, b_buf, result_buf});
    // 从 result_buf 中创建 nnc_result 张量，与 ref 进行比较
    nnc_result =
        at::from_blob(result_buf.data(), c10::IntArrayRef(resShape), options);
    // 使用 ASSERT_TRUE 断言检查 nnc_result 是否与 ref 在允许误差范围内相等
    ASSERT_TRUE(at::allclose(nnc_result, ref));
#endif

    // 创建 SimpleIREvaluator 对象 ir_eval，并使用其 call 方法执行简单的 IR 评估
    SimpleIREvaluator ir_eval(l.root_stmt(), {A, B, Result});
    // 使用 ir_eval 对象执行计算，填充 result_buf
    ir_eval.call({a_buf, b_buf, result_buf});
    // 从 result_buf 中创建 nnc_result 张量，与 ref 进行比较
    nnc_result =
        at::from_blob(result_buf.data(), c10::IntArrayRef(resShape), options);
    // 使用 ASSERT_TRUE 断言检查 nnc_result 是否与 ref 在允许误差范围内相等
    ASSERT_TRUE(at::allclose(nnc_result, ref));
  }
}

TEST(ExternalCall, UnaryFloat) {
  // 定义类型别名，表示一个接受一个张量并返回张量的函数
  using TensorFunc = std::function<at::Tensor(at::Tensor)>;
  // 定义将整数向量转换为 ExprHandle 向量的 lambda 函数
  auto toExprHandleVec = [](std::vector<int64_t> v) {
    auto intV = std::vector<int>(v.begin(), v.end());
    // 返回一个包含 intV 所有元素的 std::vector<ExprHandle> 对象
    return std::vector<ExprHandle>(intV.begin(), intV.end());
  };

  // 定义一个包含测试用例的数据结构 Test
  using Test = std::tuple<
      std::vector<int64_t>,    // 输入张量形状 aShape
      std::vector<int64_t>,    // 期望输出张量形状 resShape
      TensorFunc,              // Torch 函数指针类型
      std::string,             // 外部调用名称
      std::vector<ExprHandle>>; // 外部调用参数表达式列表

  // 初始化测试用例向量 tests
  std::vector<Test> tests = {};

  // 添加第一个测试用例到 tests 中
  tests.push_back(Test{
                       {1, 64, 8, 9},                   // aShape
                       {1, 64, 5, 7},                   // resShape
                       [](at::Tensor x) {              // Torch 函数定义
                         return at::adaptive_avg_pool2d(x, {5, 7});
                       },
                       "nnc_aten_adaptive_avg_pool2d", // 外部调用名称
                       toExprHandleVec({5, 7})         // 外部调用参数表达式列表
                   });

  // 添加第二个测试用例到 tests 中
  tests.push_back(Test{
                       {100, 200},                      // aShape
                       {100},                           // resShape
                       [](at::Tensor x) {              // Torch 函数定义
                         return at::mean(x, {1});
                       },
                       "nnc_aten_mean",                 // 外部调用名称
                       toExprHandleVec({1, /*keepdim=*/0}) // 外部调用参数表达式列表
                   });

  // 遍历所有测试用例
  for (auto curTest : tests) {
    // 初始化当前测试用例的变量
    std::vector<int64_t> aShape, resShape;
    TensorFunc torchFunc;
    std::string externCallName;
    std::vector<ExprHandle> externCallArgs;

    // 解包当前测试用例的元组到变量中
    std::tie(aShape, resShape, torchFunc, externCallName, externCallArgs) =
        curTest;

    // 创建输入缓冲区对象 A，用于表达式处理
    BufHandle A("A", toExprHandleVec(aShape), kFloat);

    // 创建结果缓冲区对象 ResultBuf，用于表达式处理
    BufHandle ResultBuf("Result", toExprHandleVec(resShape), kFloat);

    // 创建 Tensor 对象 Result，使用外部调用创建结果缓冲区
    Tensor Result = Tensor(
        ResultBuf.node(),
        ExternalCall::make(ResultBuf, externCallName, {A}, externCallArgs));

    // 创建循环嵌套对象 l，包含 Result 作为循环体
    LoopNest l({Result});

    // 准备循环嵌套对象 l 进行代码生成前的准备工作
    l.prepareForCodegen();

    // 简化循环嵌套对象 l
    l.simplify();

    // 创建 TensorOptions 对象 options，指定张量的属性
    auto options = at::TensorOptions()
                       .dtype(at::kFloat)
                       .layout(at::kStrided)
                       .device(at::kCPU)
                       .requires_grad(false);

    // 创建张量 a，使用全为 1 的数据乘以 5，并设置其选项
    at::Tensor a = at::ones(c10::IntArrayRef(aShape), options) * 5.f;

    // 创建张量 ref，通过 torchFunc 计算张量 a 的结果
    at::Tensor ref = torchFunc(a);

    // 创建 prod 函数，用于计算 vector<int64_t> v 所有元素的乘积
    auto prod = [](std::vector<int64_t> v) {
      // NOLINTNEXTLINE(modernize-use-transparent-functors)
      return std::accumulate(v.begin(), v.end(), 1, std::multiplies<int64_t>());
    };

    // 初始化 nnc_result 为张量类型
    at::Tensor nnc_result;

    // 初始化 a_buf，包含 prod(aShape) 个元素，每个元素的值为 5.0
    std::vector<float> a_buf(prod(aShape), 5.f);

    // 初始化 result_buf，包含 prod(resShape) 个元素，每个元素的值为 -1.0
    std::vector<float> result_buf(prod(resShape), -1.f);
#ifdef TORCH_ENABLE_LLVM
    // 如果 TORCH_ENABLE_LLVM 宏被定义，则进行以下代码块
    LLVMCodeGen llvm_codegen(l.root_stmt(), {A, Result});
    
    // 调用 LLVMCodeGen 对象的 call 方法，传入 a_buf 和 result_buf 参数
    llvm_codegen.call({a_buf, result_buf});
    
    // 将 result_buf 中的数据转换为 Tensor 对象 nnc_result，使用给定的形状和选项
    nnc_result =
        at::from_blob(result_buf.data(), c10::IntArrayRef(resShape), options);
    
    // 断言 nnc_result 和 ref 的值在数值上全部相近
    ASSERT_TRUE(at::allclose(nnc_result, ref));
#endif

// 创建 SimpleIREvaluator 对象，用于执行简单的 LLVM IR 代码
SimpleIREvaluator ir_eval(l.root_stmt(), {A, Result});

// 调用 ir_eval 对象的 call 方法，传入 a_buf 和 result_buf 参数
ir_eval.call({a_buf, result_buf});

// 将 result_buf 中的数据转换为 Tensor 对象 nnc_result，使用给定的形状和选项
nnc_result =
    at::from_blob(result_buf.data(), c10::IntArrayRef(resShape), options);

// 断言 nnc_result 和 ref 的值在数值上全部相近
ASSERT_TRUE(at::allclose(nnc_result, ref));
TEST(ExternalCall, ComputeInterop) {
  // This test verifies that Tensors using external calls can be used by and can
  // use Tensors built with Compute API.

  // 定义用于存储卷积和矩阵乘法结果的缓冲区
  BufHandle ConvResultBuf("ConvResult", {1, 16, 32, 32}, kFloat);
  BufHandle MatmulResultBuf("MatmulResult", {1, 16, 32, 32}, kFloat);

  // 创建一个常数值为5.0的输入 Tensor
  Tensor Input = Compute(
      "Input",
      {1, 16, 32, 32},
      [&](const VarHandle& n,
          const VarHandle& c,
          const VarHandle& h,
          const VarHandle& w) { return FloatImm::make(5.0f); });
  
  // 创建一个常数值为6.0的权重 Tensor
  Tensor Weight = Compute(
      "Weight",
      {16, 16, 1, 1},
      [&](const VarHandle& n,
          const VarHandle& c,
          const VarHandle& h,
          const VarHandle& w) { return FloatImm::make(6.0f); });

  // 使用外部调用创建卷积结果的 Tensor
  Tensor ConvResult = Tensor(
      ConvResultBuf.node(),
      ExternalCall::make(
          ConvResultBuf,
          "nnc_aten_conv2d",
          {BufHandle(Input.buf()), BufHandle(Weight.buf())},
          {}));

  // 使用外部调用创建矩阵乘法结果的 Tensor
  Tensor MatmulResult = Tensor(
      MatmulResultBuf.node(),
      ExternalCall::make(
          MatmulResultBuf,
          "nnc_aten_matmul",
          {BufHandle(ConvResult.buf()), BufHandle(ConvResult.buf())},
          {}));

  // 创建结果 Tensor，通过加载卷积和矩阵乘法结果进行求和得到
  Tensor Result = Compute(
      "Result",
      {1, 16, 32, 32},
      [&](const VarHandle& n,
          const VarHandle& c,
          const VarHandle& h,
          const VarHandle& w) {
        return ConvResult.load(n, c, h, w) + MatmulResult.load(n, c, h, w);
      });

  // 创建循环嵌套对象，并将所有 Tensor 添加到其中
  LoopNest l({Input, Weight, ConvResult, MatmulResult, Result});

  // 启用中间缓冲区的内联，用于测试，但不会进行实际内联，因为所有的 Buf 都在外部调用中定义或使用
  l.inlineIntermediateBufs(true);

  // 准备进行代码生成的前处理步骤
  l.prepareForCodegen();
  
  // 简化循环嵌套结构
  l.simplify();

  // 设置张量的选项，用于创建 PyTorch 张量
  auto options = at::TensorOptions()
                     .dtype(at::kFloat)
                     .layout(at::kStrided)
                     .device(at::kCPU)
                     .requires_grad(false);

  // 创建输入和权重的 PyTorch 张量，并执行卷积和矩阵乘法
  at::Tensor input = at::ones({1, 16, 32, 32}, options) * 5.f;
  at::Tensor weight = at::ones({16, 16, 1, 1}, options) * 6.f;
  at::Tensor t = at::conv2d(input, weight);
  at::Tensor t2 = at::matmul(t, t);

  // 计算参考结果，即卷积结果和矩阵乘法结果的和
  at::Tensor ref = t + t2;

  // 创建用于存储 NNCompiler 结果的张量
  at::Tensor nnc_result;

  // 创建用于存储输入、权重、结果的缓冲区，并调用 LLVM 代码生成器
  std::vector<float> input_buf(1 * 16 * 32 * 32, 5.f);
  std::vector<float> weight_buf(16 * 16 * 1 * 1, 6.f);
  std::vector<float> conv_result_buf(1 * 16 * 32 * 32, -1.f);
  std::vector<float> matmul_result_buf(1 * 16 * 32 * 32, -1.f);
  std::vector<float> result_buf(1 * 16 * 32 * 32, -1.f);

#ifdef TORCH_ENABLE_LLVM
  // 使用 LLVM 代码生成器，传入循环嵌套的根语句和相关的 Buf 对象
  LLVMCodeGen llvm_codegen(
      l.root_stmt(), {Input, Weight, ConvResult, MatmulResult, Result});

  // 调用 LLVM 代码生成器生成代码，填充结果缓冲区
  llvm_codegen.call(
      {input_buf, weight_buf, conv_result_buf, matmul_result_buf, result_buf});

  // 从结果缓冲区中创建 PyTorch 张量
  nnc_result = at::from_blob(result_buf.data(), {1, 16, 32, 32}, options);

  // 断言 NNCompiler 生成的结果与参考结果一致
  ASSERT_TRUE(at::allclose(nnc_result, ref));
#endif

SimpleIREvaluator ir_eval(
    l.root_stmt(), {Input, Weight, ConvResult, MatmulResult, Result});
// 创建一个简单的 IR 评估器对象，用于评估根语句和相关输入 Tensor

ir_eval.call(
    {input_buf, weight_buf, conv_result_buf, matmul_result_buf, result_buf});
// 调用 IR 评估器，传递输入缓冲区和输出缓冲区作为参数

nnc_result = at::from_blob(result_buf.data(), {1, 16, 32, 32}, options);
// 从给定的数据缓冲区创建一个 Tensor，指定形状为 {1, 16, 32, 32}，使用指定的选项

ASSERT_TRUE(at::allclose(nnc_result, ref));
// 断言：验证 nnc_result 和参考值 ref 是否在数值上全部接近

}

TEST(ExternalCall, Inlining) {
  // This test verifies that Tensors using external calls can be used by and
  // can use Tensors built with Compute API.

  BufHandle MatmulResultBuf("MatmulResult", {8, 8}, kFloat);

  Tensor A = Compute("A", {8, 8}, [&](const VarHandle& i, const VarHandle& j) {
    return FloatImm::make(5.0f);
  });
  // 创建一个 Tensor A，形状为 {8, 8}，元素值全为 5.0

  Tensor B = Compute("B", {8, 8}, [&](const VarHandle& i, const VarHandle& j) {
    return FloatImm::make(4.0f);
  });
  // 创建一个 Tensor B，形状为 {8, 8}，元素值全为 4.0

  Tensor MatmulResult = Tensor(
      MatmulResultBuf.node(),
      ExternalCall::make(
          MatmulResultBuf,
          "nnc_aten_matmul",
          {BufHandle(A.buf()), BufHandle(B.buf())},
          {}));
  // 创建一个外部调用的 Tensor MatmulResult，使用 nnc_aten_matmul 函数，传递 A 和 B 的缓冲区作为参数

  Tensor Result =
      Compute("Result", {8, 8}, [&](const VarHandle& i, const VarHandle& j) {
        return MatmulResult.load(i, j) + FloatImm::make(3.0f);
      });
  // 创建一个计算 Tensor Result，定义为 MatmulResult 的加载值加上 3.0

  StmtPtr root_stmt = alloc<torch::jit::tensorexpr::Block>(std::vector<StmtPtr>(
      {A.stmt(), B.stmt(), MatmulResult.stmt(), Result.stmt()}));
  // 创建一个根语句块，包含 A、B、MatmulResult 和 Result 的语句

  LoopNest l(root_stmt, {Result.buf()});
  // 创建一个 LoopNest 对象，以 Result 的缓冲区为参数

  // Inlining should not inline anything here since all Bufs are either
  // defined or used in ExternalCalls
  l.inlineIntermediateBufs(false);
  // 设置不进行中间缓冲区的内联优化

  l.prepareForCodegen();
  // 为代码生成做准备

  l.simplify();
  // 简化循环嵌套结构

  auto options = at::TensorOptions()
                     .dtype(at::kFloat)
                     .layout(at::kStrided)
                     .device(at::kCPU)
                     .requires_grad(false);
  // 设置 Tensor 的选项：数据类型为浮点型，布局为 Strided，设备为 CPU，不需要梯度

  at::Tensor a = at::ones({8, 8}, options) * 5.f;
  // 创建一个形状为 {8, 8} 的全 1.0 的 Tensor，并乘以 5.0 得到 Tensor a

  at::Tensor b = at::ones({8, 8}, options) * 4.f;
  // 创建一个形状为 {8, 8} 的全 1.0 的 Tensor，并乘以 4.0 得到 Tensor b

  at::Tensor t = at::matmul(a, b);
  // 计算 a 和 b 的矩阵乘法得到 Tensor t

  at::Tensor ref = t + 3.f;
  // 计算 t 加上 3.0 得到参考 Tensor ref

  at::Tensor nnc_result;
  // 声明一个用于存放计算结果的 Tensor 变量

#ifdef TORCH_ENABLE_LLVM
  LLVMCodeGen llvm_codegen(l.root_stmt(), {Result});
  // 使用 LLVM 进行代码生成，传入根语句和结果 Tensor

  llvm_codegen.call({result_buf});
  // 调用 LLVM 代码生成器，传递结果缓冲区作为参数

  nnc_result = at::from_blob(result_buf.data(), {8, 8}, options);
  // 从给定的数据缓冲区创建一个 Tensor，形状为 {8, 8}，使用指定的选项

  ASSERT_TRUE(at::allclose(nnc_result, ref));
  // 断言：验证 nnc_result 和参考值 ref 是否在数值上全部接近
#endif

  SimpleIREvaluator ir_eval(l.root_stmt(), {Result});
  // 创建一个简单的 IR 评估器对象，用于评估根语句和相关结果 Tensor

  ir_eval.call({result_buf});
  // 调用 IR 评估器，传递结果缓冲区作为参数

  nnc_result = at::from_blob(result_buf.data(), {8, 8}, options);
  // 从给定的数据缓冲区创建一个 Tensor，形状为 {8, 8}，使用指定的选项

  ASSERT_TRUE(at::allclose(nnc_result, ref));
  // 断言：验证 nnc_result 和参考值 ref 是否在数值上全部接近
}
TEST(ExternalCall, JitCustomFusionOp) {
  // 定义自定义操作的模式字符串
  const char* custom_op_schema_literal =
      "nnc_custom::add_mul(Tensor a, Tensor b, Tensor c) -> Tensor";
  // 定义外部函数的名称字符串
  const char* external_func_name = "nnc_add_mul";

  // 创建下降函数 lambda 表达式，用于处理自定义融合操作
  auto add_mul_lowering_func =
      [external_func_name](
          const std::vector<torch::jit::tensorexpr::ArgValue>& inputs,
          const std::vector<torch::jit::tensorexpr::ExprHandle>& output_shape,
          const std::vector<torch::jit::tensorexpr::ExprHandle>& output_strides,
          const std::optional<torch::jit::tensorexpr::ScalarType>& output_type,
          at::Device device) {
        // 确定输出的数据类型
        auto output_dtype = Dtype(*output_type);
        // 创建结果缓冲区
        torch::jit::tensorexpr::BufHandle result_buf(
            "nnc_add_mul_res_buf", output_shape, output_dtype);
        // 获取输入缓冲区
        const torch::jit::tensorexpr::BufHandle& a =
            std::get<torch::jit::tensorexpr::BufHandle>(inputs[0]);
        const torch::jit::tensorexpr::BufHandle& b =
            std::get<torch::jit::tensorexpr::BufHandle>(inputs[1]);
        const torch::jit::tensorexpr::BufHandle& c =
            std::get<torch::jit::tensorexpr::BufHandle>(inputs[2]);
        // 创建外部调用语句
        torch::jit::tensorexpr::StmtPtr s =
            torch::jit::tensorexpr::ExternalCall::make(
                result_buf, external_func_name, {a, b, c}, {});
        // 返回张量对象
        return Tensor(result_buf.node(), s);
      };

  // 定义外部函数的 lambda 表达式
  auto add_mul_external_func = [](int64_t bufs_num,
                                  void** buf_data,
                                  int64_t* buf_ranks,
                                  int64_t* buf_dims,
                                  int64_t* buf_strides,
                                  int8_t* buf_dtypes,
                                  int64_t args_num,
                                  int64_t* extra_args) {};

  // 注册自定义操作到 Torch 脚本运算符
  torch::jit::RegisterOperators reg({Operator(
      custom_op_schema_literal,
      [](const Node* node) -> Operation {
        return [](Stack& _stack) {
          auto a = std::move(peek(_stack, 0, 3)).toTensor();
          auto b = std::move(peek(_stack, 1, 3)).toTensor();
          auto c = std::move(peek(_stack, 2, 3)).toTensor();
          drop(_stack, 3);
          auto result = (a + b) * c;
          pack(_stack, std::move(result));
          return 0;
        };
      },
      c10::AliasAnalysisKind::FROM_SCHEMA)});

  // 获取自定义操作集合的引用，并插入新定义的自定义操作模式字符串
  auto& custom_operator_set = torch::jit::tensorexpr::getCustomOperatorSet();
  custom_operator_set.insert({custom_op_schema_literal});

  // 获取下降函数注册表的引用，并插入解析后的自定义操作模式
  auto& te_lowering_registry = torch::jit::tensorexpr::getNNCLoweringRegistry();
  te_lowering_registry.insert(
      parseSchema(custom_op_schema_literal), add_mul_lowering_func);

  // 获取 NNCFunc 注册表的引用，并将外部函数插入到注册表中
  auto& te_nnc_func_registry = torch::jit::tensorexpr::getNNCFunctionRegistry();
  te_nnc_func_registry[external_func_name] = add_mul_external_func;

  // 定义 IR 图字符串
  std::string graph_string = R"IR(
    graph(%a : Float(10, 20, strides=[20, 1], device=cpu),
          %b : Float(10, 20, strides=[20, 1], device=cpu),
          %c : Float(10, 20, strides=[20, 1], device=cpu)):
      %res : Float(10, 20, strides=[20, 1], device=cpu) = nnc_custom::add_mul(%a, %b, %c)
      return (%res))IR";



  auto graph = std::make_shared<Graph>();
  // 解析传入的 IR 字符串并构建成图对象
  torch::jit::parseIR(graph_string, graph.get());

  // 定义一个 Python 字符串，用于描述一个 Python 函数
  std::string shape_compute_python_string = R"PY(
  def computOutput(a: List[int], b: List[int], c: List[int]):
    expandedSizes: List[int] = []
    dimsA = len(a)
    dimsB = len(b)
    dimsC = len(c)
    ndim = max(dimsA, dimsB, dimsC)
    for i in range(ndim):
        offset = ndim - 1 - i
        dimA = dimsA - 1 - offset
        dimB = dimsB - 1 - offset
        dimC = dimsC - 1 - offset
        sizeA = a[dimA] if (dimA >= 0) else 1
        sizeB = b[dimB] if (dimB >= 0) else 1
        sizeC = a[dimC] if (dimC >= 0) else 1

        // 检查张量大小是否匹配，如果不匹配则抛出断言错误
        if sizeA != sizeB and sizeB != sizeC and sizeA != 1 and sizeB != 1 and sizeC != 1:
            raise AssertionError(
                "The size of tensor a {} must match the size of tensor b ("
                "{} and c {}) at non-singleton dimension {}".format(sizeA, sizeB, sizeC, i)
            )

        // 将最大的维度大小加入扩展大小列表
        expandedSizes.append(max(sizeA, sizeB, sizeC))

    return expandedSizes
  )PY";
  
  // 编译并获取函数指针，此处使用 torch::jit::compile 方法来执行 Python 代码
  auto cu_ptr = torch::jit::compile(shape_compute_python_string);
  
  // 将编译后的函数指针转换为 GraphFunction 类型
  torch::jit::GraphFunction* gf =
      (torch::jit::GraphFunction*)&cu_ptr->get_function("computOutput");
  
  // 断言函数指针有效
  ASSERT_TRUE(gf);
#ifdef TORCH_ENABLE_LLVM
  // 如果编译时启用了LLVM支持，则执行以下代码块

  // 复制当前图形对象，以便进行静态图案例处理
  auto static_graph_case = graph->copy();
  
  // 对静态图案例应用张量表达式融合优化
  FuseTensorExprs(static_graph_case, 1);
  
  // 运行FileCheck测试工具，验证特定图形操作是否存在
  torch::jit::testing::FileCheck()
      .check("prim::TensorExprGroup_")
      ->check("nnc_custom::add_mul")
      ->run(*static_graph_case);

  // 复制当前图形对象，以便进行动态图案例处理
  auto dynamic_graph_case = graph->copy();
  
  // 获取自定义操作的运算符对象，并确保其存在
  auto custom_op = torch::jit::getOperatorForLiteral(custom_op_schema_literal);
  ASSERT_TRUE(custom_op);
  
  // 注册自定义操作的形状计算图
  torch::jit::RegisterShapeComputeGraphForSchema(
      custom_op->schema(), gf->graph());
  
  // 对动态图案例应用张量表达式融合优化，关闭内联优化，启用内存优化
  FuseTensorExprs(dynamic_graph_case, 1, false, true);
  
  // 运行FileCheck测试工具，验证特定图形操作是否存在
  torch::jit::testing::FileCheck()
      .check("prim::TensorExprGroup_")
      ->check("nnc_custom::add_mul")
      ->run(*dynamic_graph_case);
#else
  // 如果未启用LLVM支持，则执行以下代码块

  // 运行FileCheck测试工具，验证特定图形操作是否存在
  torch::jit::testing::FileCheck().check("nnc_custom::add_mul")->run(*graph);
#endif
}

} // namespace jit
} // namespace torch
```