# `.\pytorch\test\mobile\nnc\test_nnc_backend.cpp`

```
// 包含 ATen 库中的函数定义头文件
#include <ATen/Functions.h>
// 包含 Google Test 框架的头文件
#include <gtest/gtest.h>
// 包含 Torch JIT 后端接口的头文件
#include <torch/csrc/jit/backends/backend.h>
// 包含 Torch JIT 后端详细信息的头文件
#include <torch/csrc/jit/backends/backend_detail.h>
// 包含 Torch JIT 后处理功能的头文件
#include <torch/csrc/jit/backends/backend_preprocess.h>
// 包含 Torch JIT 前端解析器的头文件
#include <torch/csrc/jit/frontend/resolver.h>
// 包含 Torch 移动端模型导入功能的头文件
#include <torch/csrc/jit/mobile/import.h>
// 包含 Torch 移动端模型模块定义的头文件
#include <torch/csrc/jit/mobile/module.h>
// 包含 Torch 移动端模型 NNCompiler 上下文的头文件
#include <torch/csrc/jit/mobile/nnc/context.h>
// 包含 Torch 移动端模型 NNCompiler 注册表的头文件
#include <torch/csrc/jit/mobile/nnc/registry.h>
// 包含 Torch JIT 模块冻结功能的头文件
#include <torch/csrc/jit/passes/freeze_module.h>
// 包含 Torch 自定义类的头文件
#include <torch/custom_class.h>
// 包含 Torch 脚本解释器的头文件
#include <torch/script.h>

// 命名空间声明：torch::jit::mobile::nnc
namespace torch {
namespace jit {
namespace mobile {
namespace nnc {

// 匿名命名空间，用于内部函数和数据封装
namespace {

// 创建编译规格的函数，接受多个字符串参数
c10::Dict<c10::IValue, c10::IValue> create_compile_spec(
    const std::string& method_name,
    const std::string& model_name,
    const std::string& input_shapes,
    const std::string& input_types,
    const std::string& memory_formats,
    const std::string& dynamic_sizes) {
  // 创建方法规格字典，键和值的类型分别为字符串和任意类型
  c10::Dict<c10::IValue, c10::IValue> method_spec(
      c10::StringType::get(), c10::AnyType::get());

  // 向方法规格字典中插入不同的键值对
  method_spec.insert("sizes", input_shapes);
  method_spec.insert("types", input_types);
  method_spec.insert("model_name", model_name);
  method_spec.insert("model_version", "v1");
  method_spec.insert("asmfile", "fake_nnc_model.s");
  method_spec.insert("arch", "x86-64");
  method_spec.insert("memory_formats", memory_formats);
  method_spec.insert("dynamic_sizes", dynamic_sizes);

  // 创建编译规格字典，键和值的类型同样为字符串和任意类型
  c10::Dict<c10::IValue, c10::IValue> compile_spec(
      c10::StringType::get(), c10::AnyType::get());
  // 向编译规格字典中插入方法名称和方法规格字典
  compile_spec.insert(method_name, method_spec);
  return compile_spec;
}

} // namespace

// C 语言风格的 extern 块，包含 NNCompiler 生成的测试内核函数的声明
extern "C" {

// 添加内核函数的实现，接受 void 指针数组参数并返回整数
int add_kernel(void** args) {
  // 从内存中创建 Float 类型的 Tensor 对象 input
  at::Tensor input = at::from_blob(args[0], {4, 4}, at::kFloat);
  // 从内存中创建 Float 类型的 Tensor 对象 out
  at::Tensor out = at::from_blob(args[1], {4, 4}, at::kFloat);
  // 从内存中创建 Float 类型的 Tensor 对象 param
  at::Tensor param = at::from_blob(args[2], {1}, at::kFloat);
  // 将 out 的值设为 input 和 param 相加后的结果
  out.copy_(at::add(input, param));
  // 返回整数 0 表示函数执行成功
  return 0;
}

} // extern "C"

// 注册 NNCompiler 生成的内核函数，方法名以字符串形式指定
REGISTER_NNC_KERNEL(
    "_add_kernel_nnc_fake_model:v1:forward:VERTOKEN",
    add_kernel)

// Google Test 单元测试，测试 AOT 编译后执行功能
TEST(DISABLED_NNCBackendTest, AOTCompileThenExecute) {
  // 创建名为 m 的 Torch JIT 模块
  torch::jit::Module m("m");
  // 创建全为 1 的 Float 类型的 Tensor param，并注册为模块参数
  auto param = torch::ones({1});
  m.register_parameter("param", param, false);
  // 定义模块的方法实现，使用原始字符串字面量定义
  m.define(R"(
    // 定义一个类方法 `forward`，接受一个参数 `input`，返回输入参数加上类成员变量 `param` 的结果
    def forward(self, input):
        return input + self.param
  )");

  // 运行 TorchScript 模块以获取参考结果。
  std::vector<IValue> inputs;
  // 创建一个包含值为 2.0 的 4x4 张量，并将其作为输入参数加入到 `inputs` 向量中
  inputs.emplace_back(2.0 * torch::ones({4, 4}));
  // 调用模型的 `forward` 方法得到参考结果
  auto reference = m.forward(inputs);

  // 使用 NNC 编译模型。
  // 创建编译规范，指定要编译的函数名称、模块名称、输入参数维度、数据类型等信息
  auto compile_spec = create_compile_spec(
      "forward", "_add_kernel_nnc_fake_model", "4,4", "float", "", "");
  // 创建一个键值对字典类型，键为字符串类型，值为任意类型
  auto any_dict_ty =
      c10::DictType::create(c10::StringType::get(), c10::AnyType::get());
  // 克隆并冻结模型 `m`，然后使用 NNC 进行代码生成，得到编译后的模块 `compiled_module`
  auto frozen_m = torch::jit::freeze_module(m.clone());
  auto compiled_module = torch::jit::detail::codegen_backend_module(
      "nnc", frozen_m, compile_spec, any_dict_ty);

  // 将编译后的模型保存。
  // 创建一个字符串流 `ss`，将编译后的模块保存到其中
  std::stringstream ss;
  compiled_module._save_for_mobile(ss);

  // 加载并运行保存的模型。
  // 从字符串流 `ss` 中加载模型 `loaded_module`
  auto loaded_module = _load_for_mobile(ss);
  // 使用 `inputs` 作为输入参数运行加载后的模型，并将结果保存在 `result` 中
  auto result = loaded_module.forward(inputs);
  // 断言 `result` 是一个所有元素为 3.0 的 4x4 张量
  EXPECT_TRUE(result.toTensor().equal(3.0 * torch::ones({4, 4})));
  // 断言 `result` 与参考结果 `reference` 相等
  EXPECT_TRUE(result.toTensor().equal(reference.toTensor()));
  // 删除保存的模型文件 "fake_nnc_model.s"，并断言删除成功
  EXPECT_EQ(remove("fake_nnc_model.s"), 0);
}

} // namespace nnc
} // namespace mobile
} // namespace jit
} // namespace torch
```