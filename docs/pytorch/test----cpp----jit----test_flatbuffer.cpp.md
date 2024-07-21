# `.\pytorch\test\cpp\jit\test_flatbuffer.cpp`

```
#include <test/cpp/jit/test_utils.h>  // 引入测试工具头文件

#include <gtest/gtest.h>  // 引入 Google 测试框架头文件

#include <c10/core/TensorOptions.h>  // 引入 TensorOptions 相关头文件
#include <torch/csrc/autograd/generated/variable_factories.h>  // 引入自动生成的变量工厂头文件
#include <torch/csrc/jit/api/module.h>  // 引入 Torch JIT 模块 API 头文件
#include <torch/csrc/jit/frontend/resolver.h>  // 引入 Torch JIT 前端解析器头文件
#include <torch/csrc/jit/mobile/compatibility/backport.h>  // 引入 Torch 移动兼容性回溯头文件
#include <torch/csrc/jit/mobile/compatibility/backport_manager.h>  // 引入 Torch 移动兼容性回溯管理器头文件
#include <torch/csrc/jit/mobile/compatibility/model_compatibility.h>  // 引入 Torch 移动模型兼容性头文件
#include <torch/csrc/jit/mobile/compatibility/runtime_compatibility.h>  // 引入 Torch 移动运行时兼容性头文件
#include <torch/csrc/jit/mobile/flatbuffer_loader.h>  // 引入 Torch 移动 flatbuffer 加载器头文件
#include <torch/csrc/jit/mobile/import.h>  // 引入 Torch 移动导入头文件
#include <torch/csrc/jit/mobile/interpreter.h>  // 引入 Torch 移动解释器头文件
#include <torch/csrc/jit/mobile/module.h>  // 引入 Torch 移动模块头文件
#include <torch/csrc/jit/mobile/parse_bytecode.h>  // 引入 Torch 移动解析字节码头文件
#include <torch/csrc/jit/mobile/parse_operators.h>  // 引入 Torch 移动解析操作符头文件
#include <torch/csrc/jit/serialization/export.h>  // 引入 Torch JIT 序列化导出头文件
#include <torch/csrc/jit/serialization/export_bytecode.h>  // 引入 Torch JIT 字节码导出头文件
#include <torch/csrc/jit/serialization/flatbuffer_serializer.h>  // 引入 Torch JIT flatbuffer 序列化器头文件
#include <torch/csrc/jit/serialization/flatbuffer_serializer_jit.h>  // 引入 Torch JIT JIT flatbuffer 序列化器头文件
#include <torch/csrc/jit/serialization/import.h>  // 引入 Torch JIT 序列化导入头文件
#include <torch/custom_class.h>  // 引入 Torch 自定义类头文件
#include <torch/torch.h>  // 引入 Torch 主头文件

#include <caffe2/serialize/versions.h>  // 引入 Caffe2 序列化版本头文件
#include <torch/csrc/jit/serialization/import_export_functions.h>  // 引入 Torch JIT 导入导出函数头文件
#include <unordered_set>  // 引入无序集合头文件

#if defined(FB_XPLAT_BUILD) || defined(FBCODE_CAFFE2)
#include <torch/csrc/jit/serialization/mobile_bytecode_generated_fbsource.h> // NOLINT
namespace flatbuffers = flatbuffers_fbsource;
#define FLATBUFFERS_MAX_ALIGNMENT FLATBUFFERS_FBSOURCE_MAX_ALIGNMENT
#else
#include <torch/csrc/jit/serialization/mobile_bytecode_generated.h> // NOLINT
#endif
// 在 torch::jit 命名空间下进行测试
namespace torch {
namespace jit {

namespace {
// 解析移动模块的私有函数
mobile::Module parse_mobile_module(
    void* data,
    size_t size,
    bool should_copy_tensor_memory = false) {
  return parse_and_initialize_mobile_module(
      static_cast<char*>(data),
      size,
      /*device=*/c10::nullopt,
      /*extra_files=*/nullptr,
      should_copy_tensor_memory);
}
} // namespace

// FlatbufferTest 测试套件，测试加载不正常模块情况
TEST(FlatbufferTest, LoadMalformedModule) {
  // 手动创建包含 Flatbuffer 头部的数据流
  std::stringstream bad_data;
  bad_data << "PK\x03\x04PTMF\x00\x00"
           << "*}NV\xb3\xfa\xdf\x00pa";

  // 从数据流加载模块应该抛出异常
  // 在 parse_and_initialize_mobile_module_for_jit 进行异常检查
  ASSERT_THROWS_WITH_MESSAGE(
      torch::jit::load(bad_data), "Malformed Flatbuffer module");

  // 在 parse_and_initialize_mobile_module 进行异常检查
  ASSERT_THROWS_WITH_MESSAGE(
      parse_mobile_module(bad_data.str().data(), bad_data.str().size()),
      "Malformed Flatbuffer module");
}

// FlatbufferTest 测试套件，测试最近邻上采样
TEST(FlatbufferTest, UpsampleNearest2d) {
  // 创建名为 m 的模块
  Module m("m");
  // 定义模块中的脚本代码
  m.define(R"(
    # 定义一个方法 `forward`，接受一个张量输入 `input` 和一个缩放比例 `scale`
    def forward(self, input: Tensor, scale:float):
      # 使用最近邻插值方式对输入张量进行二维上采样，将尺寸缩放到指定大小
      return torch.upsample_nearest2d(input, [1, 1], float(scale), float(scale))
  )");

  # 创建一个空的输入向量，用于存放模型输入
  std::vector<IValue> inputs;
  # 向输入向量中添加一个形状为 [1, 3, 128, 128] 的随机张量
  inputs.emplace_back(torch::rand({1, 3, 128, 128}));
  # 向输入向量中添加一个标量值 2.0
  inputs.emplace_back(at::Scalar(2.0));
  # 使用模型 `m` 对输入进行前向推理，得到输出 `ref`
  auto ref = m.forward(inputs);

  # 设置编译选项
  CompilationOptions options;
  # 将 JIT 模块 `m` 转换为移动端模块 `bc`
  mobile::Module bc = jitModuleToMobile(m, options);
  # 定义一个变量 `res` 来存放模型 `bc` 对输入 `inputs` 的前向输出
  IValue res;
  res = bc.forward(inputs);

  # 将 `res` 转换为张量 `resd`
  auto resd = res.toTensor();
  # 将 `ref` 转换为张量 `refd`
  auto refd = ref.toTensor();
  # 使用断言验证 `resd` 和 `refd` 的内容是否完全相等
  ASSERT_TRUE(resd.equal(refd));

  # 将移动端模块 `bc` 保存为字节流 `buff`
  auto buff = save_mobile_module_to_bytes(bc);
  # 从字节流 `buff` 中解析出移动端模块 `bc2`
  mobile::Module bc2 = parse_mobile_module(buff->data(), buff->size());
  # 对模型 `bc2` 进行前向推理，得到输出 `res2`
  auto res2 = bc2.forward(inputs);
  # 将 `res2` 转换为张量 `resd2`
  auto resd2 = res2.toTensor();
  # 使用断言验证 `resd2` 和 `refd` 的内容是否完全相等
  ASSERT_TRUE(resd2.equal(refd));
}

TEST(FlatbufferTest, UpsampleNearest2dWithCopyTensorMemory) {
  // 创建一个名为 "m" 的模块对象
  Module m("m");
  // 定义模块的前向传播函数，使用 torch.upsample_nearest2d 函数进行最近邻上采样
  m.define(R"(
    def forward(self, input: Tensor, scale:float):
      return torch.upsample_nearest2d(input, [1, 1], float(scale), float(scale))
  )");

  // 准备输入数据列表
  std::vector<IValue> inputs;
  // 在输入数据列表中添加随机生成的大小为 [1, 3, 128, 128] 的张量
  inputs.emplace_back(torch::rand({1, 3, 128, 128}));
  // 在输入数据列表中添加一个标量值为 2.0 的张量
  inputs.emplace_back(at::Scalar(2.0));
  // 使用定义好的模块对象执行前向传播，得到参考结果
  auto ref = m.forward(inputs);

  // 配置编译选项
  CompilationOptions options;
  // 将 JIT 编译的模块转换为移动端模块
  mobile::Module bc = jitModuleToMobile(m, options);
  // 定义结果存储变量
  IValue res;
  // 使用移动端模块执行前向传播
  res = bc.forward(inputs);

  // 将结果转换为张量
  auto resd = res.toTensor();
  auto refd = ref.toTensor();
  // 断言结果张量是否相等
  ASSERT_TRUE(resd.equal(refd));

  // 将移动端模块保存为字节流
  auto buff = save_mobile_module_to_bytes(bc);
  // 解析字节流，恢复为移动端模块对象
  mobile::Module bc2 = parse_mobile_module(buff->data(), buff->size(), true);

  // 使用恢复的移动端模块对象执行前向传播
  auto res2 = bc2.forward(inputs);
  auto resd2 = res2.toTensor();
  // 再次断言结果张量是否相等
  ASSERT_TRUE(resd2.equal(refd));
}

TEST(FlatbufferTest, CheckAttrAccess) {
  // 创建一个名为 "m" 的模块对象
  Module m("m");
  // 注册一个名为 "mobile_optimized" 的布尔类型属性，初始值为 true
  m.register_attribute("mobile_optimized", BoolType::get(), true);

  // 配置编译选项
  CompilationOptions options;
  // 将 JIT 编译的模块转换为移动端模块
  mobile::Module bc = jitModuleToMobile(m, options);
  // 从移动端模块中获取名为 "mobile_optimized" 的属性值，期望为 true
  bool mobile_optimized = bc.attr("mobile_optimized", false).toBool();

  // 断言获取的属性值为 true
  AT_ASSERT(mobile_optimized);
  // 修改模块属性 "mobile_optimized" 的值为 false
  m.setattr("mobile_optimized", false);
  // 再次将 JIT 编译的模块转换为移动端模块
  bc = jitModuleToMobile(m, options);
  // 重新获取名为 "mobile_optimized" 的属性值，期望为 false
  mobile_optimized = bc.attr("mobile_optimized", false).toBool();

  // 断言获取的属性值为 false
  AT_ASSERT(!mobile_optimized);

  // 将移动端模块保存为字节流
  auto buff = save_mobile_module_to_bytes(bc);
  // 解析字节流，恢复为移动端模块对象
  mobile::Module bc2 = parse_mobile_module(buff->data(), buff->size());
  // 从恢复的移动端模块对象中获取名为 "mobile_optimized" 的属性值，期望为 false
  auto mobile_optimized2 = bc2.attr("mobile_optimized", false).toBool();
  // 再次断言获取的属性值为 false
  AT_ASSERT(!mobile_optimized2);
}

TEST(FlatbufferTest, MethodInvocation) { // NOLINT (use =delete in gtest)
  // 定义测试程序列表
  const std::vector<std::string> test_programs{
      // 测试调用带有默认参数的方法
      R"(
      def test_func(self, x, b : int = 4):
        return self.foo + x + b
      )",
      // 内部方法调用带有默认参数（会被内联）
      R"(
      def add_with_default_arg(self, x, b : int = 4):
        return self.foo + x + b
      def test_func(self, x):
        return self.add_with_default_arg(x)  # 调用带有默认参数的方法
      )",
      // 简单方法调用
      R"(
      def test_func(self, x):
        b = 4
        return self.foo + x + b
      )",
  };
  // 遍历测试程序列表
  for (const auto& test_program : test_programs) {
    // 创建一个名为 "m" 的模块对象
    Module m("m");
    // 注册一个名为 "foo" 的参数，值为大小为 1 的张量
    m.register_parameter("foo", torch::ones({}), false);
    // 定义模块的方法
    m.define(test_program);

    const int fortyTwo = 42; // (keep linter happy)
    // 准备输入数据
    auto minput = fortyTwo * torch::ones({});
    // 在模块上运行名为 "test_func" 的方法，得到参考结果
    auto ref = m.run_method("test_func", minput);

    // 配置编译选项
    CompilationOptions options;
    // 将 JIT 编译的模块转换为移动端模块
    mobile::Module bc = jitModuleToMobile(m, options);
    // 获取名为 "test_func" 的方法
    const auto& test_func = bc.get_method("test_func");
    // 定义结果存储变量
    IValue res;
    // 多次执行移动端模块的方法
    for (int i = 0; i < 3; ++i) {
      res = test_func({minput});
    }

    // 将结果转换为 float 类型的张量项
    auto resd = res.toTensor().item<float>();
    auto refd = ref.toTensor().item<float>();
    // 断言结果张量项是否相等
    AT_ASSERT(resd == refd);

    // 将移动端模块保存为字节流
    auto buff = save_mobile_module_to_bytes(bc);
    ```
   `
    // 使用给定的缓冲区数据和大小解析为移动模块对象 bc2
    mobile::Module bc2 = parse_mobile_module(buff->data(), buff->size());

    // 获取移动模块 bc2 中的名为 "test_func" 的方法
    const auto& test_func2 = bc2.get_method("test_func");

    // 定义一个 IValue 对象 res2 用于存储方法调用的结果
    IValue res2;

    // 多次调用 test_func2 方法，每次传入参数 minput，共计3次
    for (int i = 0; i < 3; ++i) {
      res2 = test_func2({minput});
    }

    // 将 res2 转换为 Tensor，并获取其作为浮点数的值
    auto resd2 = res2.toTensor().item<float>();

    // 断言 resd2 的值等于预期的 refd 值
    AT_ASSERT(resd2 == refd);
}

// 如果未定义 FB_XPLAT_BUILD 宏，则执行以下测试
TEST(FlatbufferTest, FlatbufferBackPortTest) {
  // 创建一个名为 "m" 的 Module 对象
  Module m("m");
  // 定义一个 Torch 脚本，包含一个 forward 方法，使用 torch.upsample_nearest2d 函数
  m.define(R"(
    def forward(self, input: Tensor, scale:float):
      return torch.upsample_nearest2d(input, [1, 1], float(scale), float(scale))
  )");
  // 创建一个 stringstream 对象 ss
  std::stringstream ss;
  // 将模块 m 保存为移动端可用的格式到 ss 中，不包括额外文件，且不使用 flatbuffer
  m._save_for_mobile(ss, {}, false, true);

  // 创建一个空的 stringstream 对象 oss
  std::stringstream oss;
  // 调用 _backport_for_mobile 函数，将 ss 中的内容回溯为旧版本，版本号为 5
  bool backPortSuccess = _backport_for_mobile(ss, oss, 5);
  // 断言回溯操作成功
  ASSERT_TRUE(backPortSuccess);
}
#endif // !defined(FB_XPLAT_BUILD)

// 测试包含额外文件的移动端模块保存与加载
TEST(FlatbufferTest, ExtraFiles) {
  // 定义一个 Torch 脚本
  const auto script = R"JIT(
    def forward(self):
        x = torch.rand(5, 5)
        x = x.mm(x)
        return x
  )JIT";

  // 创建一个名为 module 的共享指针，包含编译单元的 Module 对象
  auto module =
      std::make_shared<Module>("Module", std::make_shared<CompilationUnit>());
  // 在 module 中定义脚本
  module->define(script);
  // 创建一个 ostringstream 对象 oss
  std::ostringstream oss;
  // 创建一个 unordered_map，表示额外文件
  std::unordered_map<std::string, std::string> extra_files;
  extra_files["metadata.json"] = "abc";
  extra_files["mobile_info.json"] = "{\"key\": 23}";

  // 创建一个空的 unordered_map，用于加载额外文件
  std::unordered_map<std::string, std::string> loaded_extra_files;
  // 创建一个 stringstream 对象 ss
  std::stringstream ss;
  // 将 module 保存为移动端可用的格式到 ss 中，包括额外文件，使用 flatbuffer
  module->_save_for_mobile(ss, extra_files, true, /*use_flatbuffer=*/true);

  // 初始化 loaded_extra_files 中的 "metadata.json" 为空字符串
  loaded_extra_files["metadata.json"] = "";
  // 使用 _load_for_mobile 函数加载移动端模块，并将加载的额外文件保存到 loaded_extra_files 中
  auto mobile_module = _load_for_mobile(ss, c10::nullopt, loaded_extra_files);

  // 断言加载后的 "metadata.json" 文件内容正确
  ASSERT_EQ(loaded_extra_files["metadata.json"], "abc");
  // 断言加载后的 "mobile_info.json" 文件内容正确
  ASSERT_EQ(loaded_extra_files["mobile_info.json"], "{\"key\": 23}");

  // 使用相同的流 ss 再次加载模块
  auto mobile_module2 = _load_for_mobile(ss, c10::nullopt, loaded_extra_files);

  // 断言再次加载后的 "metadata.json" 文件内容正确
  ASSERT_EQ(loaded_extra_files["metadata.json"], "abc");
  // 断言再次加载后的 "mobile_info.json" 文件内容正确
  ASSERT_EQ(loaded_extra_files["mobile_info.json"], "{\"key\": 23}");

  // 测试 flatbuffer 在不需要显式键条目映射的情况下是否有效
  std::unordered_map<std::string, std::string>
      loaded_extra_files_without_explicit_entries;
  // 使用 PARSE_ALL_EXTRA_FILE_MAPS 选项加载移动端模块
  auto mobile_module3 = _load_for_mobile(
      ss,
      c10::nullopt,
      loaded_extra_files_without_explicit_entries,
      MobileModuleLoadOptions::PARSE_ALL_EXTRA_FILE_MAPS);

  // 断言加载后的 "metadata.json" 文件内容正确
  ASSERT_EQ(
      loaded_extra_files_without_explicit_entries["metadata.json"], "abc");
  // 断言加载后的 "mobile_info.json" 文件内容正确
  ASSERT_EQ(
      loaded_extra_files_without_explicit_entries["mobile_info.json"],
      "{\"key\": 23}");
}

// Conv 测试
TEST(FlatbufferTest, Conv) {
  // 获取环境变量 "PYTORCH_TEST_WITH_TSAN"
  auto s = std::getenv("PYTORCH_TEST_WITH_TSAN");
  // 如果 s 存在且等于 "1"，则直接返回
  if (s && strcmp(s, "1") == 0)
    return;

  // 创建一个 torch::jit::IValue 类型的输入向量 inputs
  std::vector<torch::jit::IValue> inputs;

  // 创建一个名为 "m" 的 Module 对象
  Module m("m");
  // 注册名为 "weight" 和 "bias" 的参数
  m.register_parameter("weight", torch::ones({20, 1, 5, 5}), false);
  m.register_parameter("bias", torch::ones({20}), false);
  // 定义一个 Torch 脚本，包含一个 forward 方法，使用 torch._convolution 函数
  m.define(R"(
    def forward(self, input):
      return torch._convolution(input, self.weight, self.bias, [1, 1], [0, 0], [1, 1], False, [0, 0], 1, False, False, True, True)
  )");

  // 向 inputs 添加一个 Tensor
  inputs.push_back(torch::ones({1, 1, 28, 28}));

  // 调用 Module 的 forward 方法，计算输出
  auto outputref = m.forward(inputs).toTensor();

  // 编译选项
  CompilationOptions options;
  // 将 m 转换为移动端模块 bc
  mobile::Module bc = jitModuleToMobile(m, options);
  // IValue 结果变量 res
  IValue res;
  // 循环执行 3 次
  for (int i = 0; i < 3; ++i) {
  res = bc.get_method("forward")(inputs);

# 调用变量 bc 的 get_method 方法，并传入 inputs 参数，执行其 forward 方法，将结果赋给 res。


  auto output = res.toTensor();

# 将变量 res 转换为张量（tensor）类型，并将结果赋给 output。


  AT_ASSERT(outputref.dim() == output.dim());

# 使用 AT_ASSERT 宏断言 outputref 的维度与 output 的维度相同。


  AT_ASSERT(
      outputref[0][0][0][0].item<int>() == output[0][0][0][0].item<int>());

# 使用 AT_ASSERT 宏断言 outputref 和 output 在第一个元素位置上的整数值相等。


  auto buff = save_mobile_module_to_bytes(bc);

# 调用 save_mobile_module_to_bytes 函数，将变量 bc 保存为字节流，并将结果赋给 buff。


  mobile::Module bc2 = parse_mobile_module(buff->data(), buff->size());

# 使用 parse_mobile_module 函数解析 buff 中的数据，创建 mobile::Module 类型的 bc2 对象。


  for (int i = 0; i < 3; ++i) {
    res = bc2.get_method("forward")(inputs);
  }

# 循环三次，每次调用 bc2 对象的 get_method 方法，并传入 inputs 参数，执行其 forward 方法，将结果赋给 res。


  output = res.toTensor();

# 将变量 res 转换为张量（tensor）类型，并将结果赋给 output。


  AT_ASSERT(outputref.dim() == output.dim());

# 使用 AT_ASSERT 宏断言 outputref 的维度与 output 的维度相同。


  AT_ASSERT(
      outputref[0][0][0][0].item<int>() == output[0][0][0][0].item<int>());

# 使用 AT_ASSERT 宏断言 outputref 和 output 在第一个元素位置上的整数值相等。
TEST(FlatbufferTest, ConvWithCopyTensorMemory) {
  // 获取环境变量 "PYTORCH_TEST_WITH_TSAN" 的值
  auto s = std::getenv("PYTORCH_TEST_WITH_TSAN");
  // 如果环境变量存在并且其值为 "1"，则直接返回，不执行后续测试
  if (s && strcmp(s, "1") == 0)
    return;

  // 创建一个空的输入向量列表
  std::vector<torch::jit::IValue> inputs;

  // 创建名为 "m" 的 TorchScript 模块
  Module m("m");
  // 注册模块参数 "weight" 和 "bias"，分别初始化为全1的张量
  m.register_parameter("weight", torch::ones({20, 1, 5, 5}), false);
  m.register_parameter("bias", torch::ones({20}), false);
  // 定义模块的 forward 方法，执行卷积操作
  m.define(R"(
    def forward(self, input):
      return torch._convolution(input, self.weight, self.bias, [1, 1], [0, 0], [1, 1], False, [0, 0], 1, False, False, True, True)
  )");

  // 将输入张量（全1，形状为 [1, 1, 28, 28]）添加到输入向量列表中
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers,modernize-use-emplace)
  inputs.push_back(torch::ones({1, 1, 28, 28}));

  // 调用模块的 forward 方法，得到输出张量 outputref
  auto outputref = m.forward(inputs).toTensor();

  // 创建编译选项对象
  CompilationOptions options;
  // 将 TorchScript 模块转换为移动端模块 bc
  mobile::Module bc = jitModuleToMobile(m, options);
  // 创建空的 IValue 对象 res
  IValue res;
  // 多次执行模块的 forward 方法
  for (int i = 0; i < 3; ++i) {
    res = bc.get_method("forward")(inputs);
  }
  // 将结果转换为张量 output
  auto output = res.toTensor();
  // 断言输出张量的维度与参考输出张量 outputref 的维度相同
  AT_ASSERT(outputref.dim() == output.dim());
  // 断言输出张量的特定元素值相同
  AT_ASSERT(
      outputref[0][0][0][0].item<int>() == output[0][0][0][0].item<int>());

  // 将移动端模块 bc 序列化为字节流
  auto buff = save_mobile_module_to_bytes(bc);
  // 从字节流中解析出移动端模块 bc2
  mobile::Module bc2 = parse_mobile_module(buff->data(), buff->size(), true);

  // 再次执行 bc2 模块的 forward 方法
  for (int i = 0; i < 3; ++i) {
    res = bc2.get_method("forward")(inputs);
  }
  // 将结果转换为张量 output
  output = res.toTensor();
  // 断言输出张量的维度与参考输出张量 outputref 的维度相同
  AT_ASSERT(outputref.dim() == output.dim());
  // 断言输出张量的特定元素值相同
  AT_ASSERT(
      outputref[0][0][0][0].item<int>() == output[0][0][0][0].item<int>());
}

TEST(FlatbufferTest, Inline) {
  // 创建名为 "m" 的 TorchScript 模块
  Module m("m");
  // 定义模块的多个函数
  m.define(R"JIT(
  def foo1(self, x):
      return x + 1

  def foo2(self, x):
      return self.foo1(x) + 2

  def foo3(self, x):
      return self.foo2(x) + 3
  )JIT");
  // 创建编译选项对象
  CompilationOptions options;
  // 将 TorchScript 模块转换为移动端模块 bc
  mobile::Module bc = jitModuleToMobile(m, options);
  // 创建输入向量列表，包含一个全1的标量张量
  std::vector<torch::jit::IValue> inputs({torch::ones({})});
  // 调用模块的 foo3 方法，得到输出结果 output
  auto output = bc.get_method("foo3")(inputs);
  // 断言输出结果的标量值为 7.0
  AT_ASSERT(output.toTensor().item<float>() == 7.0);

  // 将移动端模块 bc 序列化为字节流
  auto buff = save_mobile_module_to_bytes(bc);
  // 从字节流中解析出移动端模块 bc2
  mobile::Module bc2 = parse_mobile_module(buff->data(), buff->size());
  // 创建输入向量列表，包含一个全1的标量张量
  std::vector<torch::jit::IValue> inputs2({torch::ones({})});
  // 再次调用 bc2 模块的 foo3 方法，得到输出结果 output
  output = bc2.get_method("foo3")(inputs2);
  // 断言输出结果的标量值为 7.0
  AT_ASSERT(output.toTensor().item<float>() == 7.0);
}

TEST(FlatbufferTest, InlineWithCopyTensorMemory) {
  // 创建名为 "m" 的 TorchScript 模块
  Module m("m");
  // 定义模块的多个函数
  m.define(R"JIT(
  def foo1(self, x):
      return x + 1

  def foo2(self, x):
      return self.foo1(x) + 2

  def foo3(self, x):
      return self.foo2(x) + 3
  )JIT");
  // 创建编译选项对象
  CompilationOptions options;
  // 将 TorchScript 模块转换为移动端模块 bc
  mobile::Module bc = jitModuleToMobile(m, options);
  // 创建输入向量列表，包含一个全1的标量张量
  std::vector<torch::jit::IValue> inputs({torch::ones({})});
  // 调用模块的 foo3 方法，得到输出结果 output
  auto output = bc.get_method("foo3")(inputs);
  // 断言输出结果的标量值为 7.0
  AT_ASSERT(output.toTensor().item<float>() == 7.0);

  // 将移动端模块 bc 序列化为字节流
  auto buff = save_mobile_module_to_bytes(bc);
  // 从字节流中解析出移动端模块 bc2
  mobile::Module bc2 = parse_mobile_module(buff->data(), buff->size(), true);
  // 创建输入向量列表，包含一个全1的标量张量
  std::vector<torch::jit::IValue> inputs2({torch::ones({})});
  // 再次调用 bc2 模块的 foo3 方法，得到输出结果 output
  output = bc2.get_method("foo3")(inputs2);
  // 断言输出结果的标量值为 7.0
  AT_ASSERT(output.toTensor().item<float>() == 7.0);
}
TEST(FlatbufferTest, Tuple) {
  // 创建一个名为 "m" 的模块对象
  Module m("m");
  // 在模块中定义一个 JIT 编译的函数 foo 和 forward
  m.define(R"JIT(
  def foo(self, x):
      return (1, 2, x + 3)

  def forward(self, x):
      tuple = self.foo(x)
      return tuple
  )JIT");
  // 设置编译选项
  CompilationOptions options;
  // 将 JIT 编译后的模块转换为移动端模块
  mobile::Module bc = jitModuleToMobile(m, options);
  // 准备输入数据
  std::vector<torch::jit::IValue> inputs({torch::ones({})});
  // 调用 forward 方法并获取输出结果
  auto output = bc.get_method("forward")(inputs);
  // 断言输出的元组中第二个元素是否为 2
  AT_ASSERT(output.toTupleRef().elements()[1].toInt() == 2);

  // 将移动端模块保存为字节流
  auto buff = save_mobile_module_to_bytes(bc);
  // 解析字节流为移动端模块
  mobile::Module bc2 = parse_mobile_module(buff->data(), buff->size());
  // 再次调用 forward 方法并获取输出结果
  output = bc2.get_method("forward")(inputs);
  // 再次断言输出的元组中第二个元素是否为 2
  AT_ASSERT(output.toTuple()->elements()[1].toInt() == 2);
}

TEST(FlatbufferTest, Dict) {
  // 创建一个名为 "m" 的模块对象
  Module m("m");
  // 在模块中定义一个 JIT 编译的函数 foo 和 forward
  m.define(R"JIT(
  def foo(self, x):
      return {"result": x + 1}

  def forward(self, x):
      d = self.foo(x)
      return d
  )JIT");
  // 设置编译选项
  CompilationOptions options;
  // 将 JIT 编译后的模块转换为移动端模块
  mobile::Module bc = jitModuleToMobile(m, options);
  // 准备输入数据
  std::vector<torch::jit::IValue> inputs({torch::ones({})});
  // 调用 forward 方法并获取输出结果
  auto output = bc.get_method("forward")(inputs);
  // 断言输出的字典中 "result" 键对应的值是否为 2
  AT_ASSERT(output.toGenericDict().at("result").toTensor().item().toInt() == 2);

  // 将移动端模块保存为字节流
  auto buff = save_mobile_module_to_bytes(bc);
  // 解析字节流为移动端模块
  mobile::Module bc2 = parse_mobile_module(buff->data(), buff->size());
  // 再次调用 forward 方法并获取输出结果
  output = bc2.get_method("forward")(inputs);
  // 再次断言输出的字典中 "result" 键对应的值是否为 2
  AT_ASSERT(output.toGenericDict().at("result").toTensor().item().toInt() == 2);
}

TEST(FlatbufferTest, Prim) {
  // 创建一个名为 "m" 的模块对象
  Module m("m");
  // 在模块中定义一个 JIT 编译的函数 forward
  m.define(R"JIT(
        def forward(self, x):
            return int(x)
  )JIT");

  // 准备输入数据
  std::vector<IValue> inputs;
  auto minput = 3.5 * torch::ones({});
  inputs.emplace_back(minput);
  // 在模块中运行 forward 方法并获取参考结果
  auto ref = m.run_method("forward", minput);

  // 设置编译选项
  CompilationOptions options;
  // 将 JIT 编译后的模块转换为移动端模块
  mobile::Module bc = jitModuleToMobile(m, options);
  IValue res;
  // 多次使用相同输入调用移动端模块的 forward 方法
  for (int i = 0; i < 3; ++i) {
    // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
    auto bcinputs = inputs;
    res = bc.get_method("forward")(bcinputs);
  }

  // 将结果转换为整数并进行断言比较
  auto resi = res.toInt();
  auto refi = ref.toInt();
  AT_ASSERT(resi == refi);

  // 将移动端模块保存为字节流
  auto buff = save_mobile_module_to_bytes(bc);
  // 解析字节流为移动端模块
  mobile::Module bc2 = parse_mobile_module(buff->data(), buff->size());
  // 再次使用相同输入调用移动端模块的 forward 方法
  for (int i = 0; i < 3; ++i) {
    // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
    auto bcinputs = inputs;
    res = bc2.get_method("forward")(bcinputs);
  }
  // 将结果转换为整数并进行断言比较
  auto resi2 = res.toInt();
  AT_ASSERT(resi2 == refi);
}

TEST(FlatbufferTest, PrimScalar) {
  // 创建一个名为 "m" 的模块对象
  Module m("m");
  // 在模块中定义一个 JIT 编译的函数 forward
  m.define(R"JIT(
        def forward(self, x):
            return int(x.item())
  )JIT");

  // 准备输入数据
  std::vector<IValue> inputs;
  auto minput = 3.5 * torch::ones({});
  inputs.emplace_back(minput);
  // 在模块中运行 forward 方法并获取参考结果
  auto ref = m.run_method("forward", minput);

  // 设置编译选项
  CompilationOptions options;
  // 将 JIT 编译后的模块转换为移动端模块
  mobile::Module bc = jitModuleToMobile(m, options);
  IValue res;
  // 多次使用相同输入调用移动端模块的 forward 方法
  for (int i = 0; i < 3; ++i) {
    // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
    auto bcinputs = inputs;
    res = bc.get_method("forward")(bcinputs);
  }

  // 将结果转换为整数并进行断言比较
  auto resi = res.toInt();
  auto refi = ref.toInt();
  AT_ASSERT(resi == refi);

  // 将移动端模块保存为字节流
  auto buff = save_mobile_module_to_bytes(bc);
  // 解析字节流为移动端模块
  mobile::Module bc2 = parse_mobile_module(buff->data(), buff->size());
  // 再次使用相同输入调用移动端模块的 forward 方法
  for (int i = 0; i < 3; ++i) {
    // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
    auto bcinputs = inputs;
    res = bc2.get_method("forward")(bcinputs);
  }
  // 将结果转换为整数并进行断言比较
  auto resi2 = res.toInt();
  AT_ASSERT(resi2 == refi);
}
  // 调用bc对象的"forward"方法，并传入bcinputs参数，得到计算结果
  res = bc.get_method("forward")(bcinputs);

auto resi = res.toInt();
auto refi = ref.toInt();
// 断言计算结果resi与参考值refi相等
AT_ASSERT(resi == refi);

// 将mobile模块保存为字节流
auto buff = save_mobile_module_to_bytes(bc);
// 从字节流解析mobile模块，得到新的Module对象bc2
mobile::Module bc2 = parse_mobile_module(buff->data(), buff->size());
for (int i = 0; i < 3; ++i) {
  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
  // 复制inputs到bcinputs，准备进行下一次前向计算
  auto bcinputs = inputs;
  // 调用bc2对象的"forward"方法，并传入bcinputs参数，更新计算结果res
  res = bc2.get_method("forward")(bcinputs);
}

auto resi2 = res.toInt();
// 再次断言新的计算结果resi2与原始参考值refi相等
AT_ASSERT(resi2 == refi);
}

TEST(FlatbufferTest, WrongMethodName) {
  // 创建一个名为 "m" 的模块对象
  Module m("m");
  // 向模块中注册一个名为 "foo" 的参数，初始化为全1的张量，不需要梯度
  m.register_parameter("foo", torch::ones({}), false);
  // 定义一个方法 "add"，接受参数 x，计算并返回 self.foo + x + 4
  m.define(R"(
    def add(self, x):
      b = 4
      return self.foo + x + b
  )");
  // 配置编译选项
  CompilationOptions options;
  // 将 JIT 编译后的模块转换为移动端模块
  mobile::Module bc = jitModuleToMobile(m, options);
  // 准备输入数据
  std::vector<IValue> inputs;
  auto minput = 5 * torch::ones({});
  inputs.emplace_back(minput);
  // 断言在调用不存在的方法 "forward" 时抛出异常，并带有指定错误消息
  ASSERT_THROWS_WITH_MESSAGE(
      bc.get_method("forward")(inputs), "is not defined");

  // 将编译后的移动端模块序列化为字节流
  auto buff = save_mobile_module_to_bytes(bc);
  // 解析序列化后的字节流为移动端模块对象
  mobile::Module bc2 = parse_mobile_module(buff->data(), buff->size());
  // 再次断言在调用不存在的方法 "forward" 时抛出异常，并带有指定错误消息
  ASSERT_THROWS_WITH_MESSAGE(
      bc2.get_method("forward")(inputs), "is not defined");
}

TEST(FlatbufferTest, SetState) {
  // 创建一个名为 "m" 的模块对象
  Module m("m");
  // 向模块中注册一个名为 "foo" 的参数，初始化为全1的张量，不需要梯度
  m.register_parameter("foo", torch::ones({}), false);
  // 定义方法 "__getstate__" 返回 self.foo，方法 "__setstate__" 设置 self.foo
  // 定义方法 "forward"，接受参数 x，计算并返回 self.foo + x + 4
  m.define(R"(
    def __getstate__(self):
      return self.foo
    def __setstate__(self, a):
      self.foo = a
    def forward(self, x):
      b = 4
      return self.foo + x + b
  )");

  // 准备输入数据
  std::vector<IValue> inputs;
  auto minput = 5 * torch::ones({});
  inputs.emplace_back(minput);

  // 将模块保存到 stringstream 中
  std::stringstream ms;
  m.save(ms);
  // 从 stringstream 中加载模块
  auto loaded_m = load(ms);
  // 运行加载后的模块的 "forward" 方法，并保存结果为 ref
  auto ref = loaded_m.run_method("forward", minput);

  // 配置编译选项
  CompilationOptions options;
  // 将 JIT 编译后的模块转换为移动端模块
  mobile::Module bc = jitModuleToMobile(m, options);
  IValue res;
  for (int i = 0; i < 3; ++i) {
    // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
    auto bcinputs = inputs;
    // 多次调用移动端模块的 "forward" 方法，并保存结果为 res
    res = bc.get_method("forward")(bcinputs);
  }

  // 将 res 转换为 float 类型，并断言其值与 ref 相等
  auto resd = res.toTensor().item<float>();
  auto refd = ref.toTensor().item<float>();
  AT_ASSERT(resd == refd);

  // 将编译后的移动端模块序列化为字节流
  auto buff = save_mobile_module_to_bytes(bc);
  // 解析序列化后的字节流为移动端模块对象
  mobile::Module bc2 = parse_mobile_module(buff->data(), buff->size());
  for (int i = 0; i < 3; ++i) {
    // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
    auto bcinputs = inputs;
    // 多次调用重新解析后的移动端模块的 "forward" 方法，并保存结果为 res
    res = bc2.get_method("forward")(bcinputs);
  }

  // 将 res 转换为 float 类型，并断言其值与 ref 相等
  auto resd2 = res.toTensor().item<float>();
  AT_ASSERT(resd2 == refd);
}

class TorchBindFlatbufferTestStruct : public torch::jit::CustomClassHolder {
 public:
  // 自定义类，用于处理 Tensor 类型数据的简单描述
  std::string get(at::Tensor t) {
    std::stringstream ss;
    ss << "Hello! Your tensor has ";
    ss << t.numel();
    ss << " elements!";
    return ss.str();
  }
};

namespace {
struct ClassNamespaceValue : public SugaredValue {
  explicit ClassNamespaceValue(c10::QualifiedName name)
      : basename_(std::move(name)) {}

  // 获取命名空间中的属性值，支持自定义类
  std::shared_ptr<SugaredValue> attr(
      const SourceRange& loc,
      GraphFunction& m,
      const std::string& name) override {
    const auto fullName = c10::QualifiedName(basename_, name);

    // 检查属性是否是自定义类，如果是，返回该类的值
    if (auto custom_class = getCustomClass(fullName.qualifiedName())) {
      return std::make_shared<ClassValue>(custom_class);
    }

    // 如果不是自定义类，假设其是另一个命名空间
    // NOLINTNEXTLINE(performance-move-const-arg)
    return std::make_shared<ClassNamespaceValue>(std::move(fullName));
  }

  // 返回该值的类型信息
  std::string kind() const override {
    return "Class Namespace";
  }


    // 返回一个字符串 "Class Namespace"
    return "Class Namespace";
  }



 private:
  c10::QualifiedName basename_;


 private:
  // 声明一个私有成员变量，类型为 c10::QualifiedName，用于存储基本名称
  c10::QualifiedName basename_;
};

// 定义一个结构体 TestModuleResolver，继承自 Resolver
struct TestModuleResolver : public Resolver {
  // 解析值的方法，根据名称返回相应的 SugaredValue
  std::shared_ptr<SugaredValue> resolveValue(
      const std::string& name,
      GraphFunction& m,
      const SourceRange& loc) override {
    // 如果名称为 "torch"，返回一个内置模块 "aten" 的 SugaredValue
    if (name == "torch") {
      return std::make_shared<BuiltinModule>("aten");
    }
    // 如果名称为 "__torch__"，返回一个 ClassNamespaceValue，表示一个类命名空间
    else if (name == "__torch__") {
      return std::make_shared<ClassNamespaceValue>(c10::QualifiedName(name));
    }

    // 其他情况返回空指针
    return nullptr;
  }

  // 解析类型的方法，根据名称返回相应的 TypePtr
  TypePtr resolveType(const std::string& name, const SourceRange& loc)
      override {
    // 在这里简单返回空指针，表示没有特定类型的解析实现
    return nullptr;
  }
};
} // namespace

// 测试用例，测试 Flatbuffer 类型的自定义类
TEST(FlatbufferTest, BuiltinClass) {
  // 创建一个名为 "m" 的脚本模块
  script::Module m("m");

  // 获取自定义类 "__torch__.torch.classes._TorchScriptTesting._FlatbufferTest"
  auto cls = getCustomClass(
      "__torch__.torch.classes._TorchScriptTesting._FlatbufferTest");
  // 断言自定义类对象非空
  TORCH_INTERNAL_ASSERT(cls);

  // 创建一个 CustomClassHolder 对象
  c10::intrusive_ptr<torch::CustomClassHolder> obj_holder;

  // 在模块中注册名为 "my_obj" 的属性，类型为 cls，值为 obj_holder 的 Capsule
  m.register_attribute("my_obj", cls, IValue::make_capsule(obj_holder));

  // 在模块中注册一个参数 "foo"，值为 torch.ones({})，不是可训练的参数
  m.register_parameter("foo", torch::ones({}), false);

  // 定义脚本字符串，包含 __getstate__、__setstate__ 和 forward 方法的实现
  m.define(
      R"(
    def __getstate__(self):
      return 1
    def __setstate__(self, a):
      self.my_obj = __torch__.torch.classes._TorchScriptTesting._FlatbufferTest()

    def forward(self, x) -> str:
      return self.my_obj.get(x)
  )",
      // 使用 TestModuleResolver 解析器来解析这些方法
      std::make_shared<TestModuleResolver>());

  // 配置编译选项
  CompilationOptions options;

  // 将 JIT 编译的模块转换为移动端模块
  mobile::Module bc = jitModuleToMobile(m, options);

  // 将移动端模块保存为字节流
  auto buff = save_mobile_module_to_bytes(bc);

  // 解析保存的移动端模块字节流为移动端模块 bc2
  mobile::Module bc2 = parse_mobile_module(buff->data(), buff->size());

  // 期望的字符串结果
  std::string expected = "Hello! Your tensor has 12 elements!";

  // 调用 bc2 中的 forward 方法，传入参数 torch::zeros({3, 4})
  auto res =
      bc2.get_method("forward")(std::vector<IValue>{torch::zeros({3, 4})});

  // 获取返回值的字符串表示，并断言与期望的结果相同
  const auto& str2 = res.toStringRef();
  AT_ASSERT(str2 == expected);
}

// 测试用例，测试 Flatbuffer 类型的内置函数
TEST(FlatbufferTest, BuiltinFunction) {
  // 创建一个名为 "m" 的脚本模块
  script::Module m("m");

  // 创建一个自定义类对象，类型为 TorchBindFlatbufferTestStruct
  auto custom_class_obj = make_custom_class<TorchBindFlatbufferTestStruct>();

  // 在模块中注册名为 "my_obj" 的属性，类型为 custom_class_obj 的类型，值为 custom_class_obj
  m.register_attribute("my_obj", custom_class_obj.type(), custom_class_obj);

  // 定义脚本字符串，包含 forward 方法的实现
  m.define(R"(
    def forward(self, x) -> str:
      return self.my_obj.get(x)
  )");

  // 配置编译选项
  CompilationOptions options;

  // 将 JIT 编译的模块转换为移动端模块
  mobile::Module bc = jitModuleToMobile(m, options);

  // 调用 bc 中的 forward 方法，传入参数 torch::zeros({3, 4})
  auto res =
      bc.get_method("forward")(std::vector<IValue>{torch::zeros({3, 4})});

  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
  auto str = res.toStringRef();

  // 期望的字符串结果
  std::string expected = "Hello! Your tensor has 12 elements!";
  
  // 断言返回的字符串与期望结果相同
  AT_ASSERT(str == expected);

  // 将移动端模块保存为字节流
  auto buff = save_mobile_module_to_bytes(bc);

  // 解析保存的移动端模块字节流为移动端模块 bc2
  mobile::Module bc2 = parse_mobile_module(buff->data(), buff->size());

  // 再次调用 bc2 中的 forward 方法，传入参数 torch::zeros({3, 4})
  res = bc2.get_method("forward")(std::vector<IValue>{torch::zeros({3, 4})});

  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
  str = res.toStringRef();

  // 断言返回的字符串与期望结果相同
  AT_ASSERT(str == expected);
}

// 测试用例，测试评估功能
TEST(FlatbufferTest, Eval) {
  // 创建一个空的 torch::jit::IValue 向量 inputs
  std::vector<torch::jit::IValue> inputs;

  // 创建一个名为 "m" 的脚本模块
  Module m("m");

  // 定义脚本字符串，包含 __init__ 方法的实现
  m.define(R"(
    def __init__(self, x):
      self.training = True
    // 定义一个方法 `forward`，接收一个输入 `input`
    def forward(self, input):
      // 使用 PyTorch 的 dropout 函数对输入进行处理，保持所有元素（概率为 1.0），根据当前模型训练状态进行处理
      return torch.dropout(input, 1.0, self.training)
  )");

  // 将一个全为 1 的张量作为输入添加到 `inputs` 向量中
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers,modernize-use-emplace)
  inputs.push_back(torch::ones({1, 1, 28, 28}));

  // 将模型设置为评估模式
  m.eval();

  // 调用模型的 `forward` 方法，传入 `inputs` 向量作为参数，并将结果转换为张量
  auto outputref = m.forward(inputs).toTensor();

  // 将模型设置为训练模式，以确保移动设备上的评估操作能正确切换到评估模式
  m.train();

  // 设置编译选项
  CompilationOptions options;

  // 将 JIT 编译后的模型转换为移动端可执行的模块 `bc`
  mobile::Module bc = jitModuleToMobile(m, options);

  // 将 `bc` 设置为评估模式
  bc.eval();

  // 定义一个 `IValue` 类型的变量 `res`
  IValue res;

  // 多次调用 `bc` 模块的 `forward` 方法，传入 `inputs` 向量作为参数
  for (int i = 0; i < 3; ++i) {
    res = bc.get_method("forward")(inputs);
  }

  // 将 `res` 转换为张量 `output`
  auto output = res.toTensor();

  // 断言 `outputref` 和 `output` 的维度相同
  AT_ASSERT(outputref.dim() == output.dim());

  // 断言 `outputref` 和 `output` 在索引 `(0, 0, 0, 0)` 处的元素值相同
  AT_ASSERT(
      outputref[0][0][0][0].item<int>() == output[0][0][0][0].item<int>());

  // 将 `bc` 模块保存为字节流 `buff`
  auto buff = save_mobile_module_to_bytes(bc);

  // 解析字节流 `buff`，并将其转换为移动端模块 `bc2`
  mobile::Module bc2 = parse_mobile_module(buff->data(), buff->size());

  // 将 `bc2` 设置为评估模式
  bc2.eval();

  // 再次多次调用 `bc2` 模块的 `forward` 方法，传入 `inputs` 向量作为参数
  for (int i = 0; i < 3; ++i) {
    res = bc2.get_method("forward")(inputs);
  }

  // 将 `res` 转换为张量 `output`
  output = res.toTensor();

  // 断言 `outputref` 和 `output` 的维度相同
  AT_ASSERT(outputref.dim() == output.dim());

  // 断言 `outputref` 和 `output` 在索引 `(0, 0, 0, 0)` 处的元素值相同
  AT_ASSERT(
      outputref[0][0][0][0].item<int>() == output[0][0][0][0].item<int>());
TEST(FlatbufferTest, DuplicateSetState) {
  // 创建名为"M"的模块对象，注册一个名为"foo"的参数，初始值为全1的张量，不可训练
  Module m("M");
  // 定义模块"M"的方法和计算图
  m.define(R"(
    // 定义序列化方法__getstate__，返回self.foo + self.foo的值
    def __getstate__(self):
      return self.foo + self.foo
    // 定义反序列化方法__setstate__，将传入值a赋给self.foo
    def __setstate__(self, a):
      self.foo = a
    // 定义前向传播方法forward，接受输入x，计算并返回self.foo + x + 4的结果
    def forward(self, x):
      b = 4
      return self.foo + x + b
  )");

  // 创建名为"B"的模块对象
  Module b("B");
  // 向模块"B"注册两个子模块"M0"和"M1"，它们都使用模块"M"作为基础模块
  b.register_module("M0", m);
  b.register_module("M1", m);
  // 定义模块"B"的方法和计算图
  b.define(R"(
    # 定义一个方法 `forward`，接受输入 `x`，调用 `M0` 和 `M1` 的 `forward` 方法并返回它们的求和结果
    def forward(self, x):
      return self.M0.forward(x) + self.M1.forward(x)
  )");

  # 创建一个编译选项对象
  CompilationOptions options;
  # 将 JIT 编译后的模块转换为移动端模块 `bc`
  mobile::Module bc = jitModuleToMobile(m, options);
  # 获取转换后模块 `bc` 的方法集合
  const auto methods = bc.get_methods();
  # 期望的方法数量
  const size_t expected_n = 3;
  # 断言方法集合的大小与期望的数量相等
  ASSERT_EQ(methods.size(), expected_n);

  # 将移动端模块 `bc` 序列化为字节流
  auto buff = save_mobile_module_to_bytes(bc);
  # 解析字节流并重建为移动端模块 `bc2`
  mobile::Module bc2 = parse_mobile_module(buff->data(), buff->size());
  # 获取重建后模块 `bc2` 的方法集合
  const auto methods2 = bc.get_methods();
  # 断言重建后方法集合的大小与期望的数量相等
  ASSERT_EQ(methods2.size(), expected_n);
namespace {
void testLiteModuleCompareResultTensors(
    Module& m,
    const std::vector<torch::jit::IValue>& inputs,
    const std::string& method_name = "forward") {
  // 调用给定模块的指定方法，使用输入数据获取输出张量作为参考结果
  auto outputref = m.get_method(method_name)(inputs).toTensor();

  // 设置编译选项
  CompilationOptions options;
  // 将 JIT 模块转换为移动端模块
  mobile::Module bc = jitModuleToMobile(m, options);
  // 定义结果 IValue 对象
  IValue res;
  // 进行三次前向推理
  for (int i = 0; i < 3; ++i) {
  // 使用反射从`bc`对象中获取指定方法(`method_name`)的函数指针，并执行该方法，传入`inputs`作为参数，将结果赋给`res`
  res = bc.get_method(method_name)(inputs);
}
// 将`res`转换为张量(`Tensor`)类型，存储在`output`变量中
auto output = res.toTensor();
// 断言检查输出张量`outputref`的维度与`output`的维度是否相等
AT_ASSERT(outputref.dim() == output.dim());
// 断言检查输出张量`output`与`outputref`是否完全相等

auto buff = save_mobile_module_to_bytes(bc);
// 将移动端模块(`bc`)保存为字节流(`buff`)
mobile::Module bc2 = parse_mobile_module(buff->data(), buff->size());
// 解析字节流(`buff`)并创建一个新的移动端模块(`bc2`)

for (int i = 0; i < 3; ++i) {
  // 使用反射从`bc2`对象中获取指定方法(`method_name`)的函数指针，并执行该方法，传入`inputs`作为参数，将结果赋给`res`
  res = bc2.get_method(method_name)(inputs);
}
// 将`res`转换为张量(`Tensor`)类型，存储在`output`变量中
output = res.toTensor();
// 断言检查输出张量`outputref`的维度与`output`的维度是否相等
AT_ASSERT(outputref.dim() == output.dim());
// 断言检查输出张量`output`与`outputref`是否完全相等
}

// 定义静态函数 testDefaultArgsPinv，接受一个整数参数 num_args
static void testDefaultArgsPinv(int num_args) {
  // 创建名为 m 的模块对象
  Module m("m");
  // 根据 num_args 的不同值定义不同的前向传播函数
  if (num_args == 1) {
    // 使用字符串字面量定义只有一个参数的前向传播函数
    m.define(R"(
      def forward(self, input):
        return torch.linalg_pinv(input)
    )");
  } else if (num_args == 2) {
    // 使用字符串字面量定义两个参数的前向传播函数
    m.define(R"(
      def forward(self, input):
        return torch.linalg_pinv(input, 1e-5)
    )");
  } else if (num_args == 3) {
    // 使用字符串字面量定义三个参数的前向传播函数
    m.define(R"(
      def forward(self, input):
        return torch.linalg_pinv(input, 1e-5, True)
    )");
  }

  // 创建存储输入数据的 IValue 向量
  std::vector<torch::jit::IValue> inputs;
  const int N = 28;
  // 创建从 1 到 N*N 的步长为 1 的张量 input
  auto input = torch::range(1, N * N, 1);
  // 设定 input 的第一个元素为 1，以确保生成一个更稳定的矩阵
  input[0] = 1; // a more stable matrix
  // 将 input 调整为 N x N 的形状
  input = input.view({N, N});
  // 将调整后的 input 添加到输入向量 inputs 中
  inputs.emplace_back(input);
  // 调用 testLiteModuleCompareResultTensors 函数，比较模块 m 的输出结果和输入向量 inputs
  testLiteModuleCompareResultTensors(m, inputs);
}
} // namespace

#if !defined FB_XPLAT_BUILD
// 定义 FlatbufferTest 测试集中的 DefaultArgsPinv 测试
TEST(FlatbufferTest, DefaultArgsPinv) {
  // 测试不同数量的指定参数情况
  // 未指定的参数采用默认值
  for (int num_args = 1; num_args <= 3; ++num_args) {
    // 定义一个函数或方法，名称为 testDefaultArgsPinv，接受一个参数 num_args
    testDefaultArgsPinv(num_args);
    
    // bytecode（字节码）带有一个指定参数：
    // （6，
    //     ('__torch__.m.forward',
    //         (('instructions',
    //             (('STOREN', 1, 2),        // 存储操作，将寄存器1中的值存储到寄存器2
    //              ('DROPR', 1, 0),        // 丢弃操作，丢弃寄存器1中的值
    //              ('MOVE', 2, 0),         // 移动操作，将寄存器0中的值移动到寄存器2
    //              ('OP', 0, 0),           // 操作指令，对寄存器0执行操作
    //              ('RET', 0, 0))),        // 返回指令，返回寄存器0的值
    //         ('operators', (('aten::linalg_pinv', '', 1),)),   // 运算符，指定使用的运算符为 aten::linalg_pinv，参数数量为1
    //         ('constants', (False, 1e-15)),  // 常量表，包含两个常量值（False, 1e-15），默认常量未使用
    //         ('types', ()),           // 类型信息，空元组表示没有类型信息
    //         ('register_size', 2)),   // 寄存器大小，设置为2个寄存器
    //     (('arguments',
    //         ((('name', 'self'), ('type', '__torch__.m'), ('default_value', None)),   // 参数信息，self 参数类型为 '__torch__.m'，默认值为 None
    //          (('name', 'input'), ('type', 'Tensor'), ('default_value', None)))),    // 参数信息，input 参数类型为 'Tensor'，默认值为 None
    //         ('returns',
    //             ((('name', ''), ('type', 'Tensor'), ('default_value', None)),))))   // 返回值信息，返回类型为 'Tensor'，默认值为 None
    
    // bytecode（字节码）带有两个指定参数：
    // （6，
    //     ('__torch__.m.forward',
    //         (('instructions',
    //             (('STOREN', 1, 2),        // 存储操作，将寄存器1中的值存储到寄存器2
    //              ('DROPR', 1, 0),        // 丢弃操作，丢弃寄存器1中的值
    //              ('MOVE', 2, 0),         // 移动操作，将寄存器0中的值移动到寄存器2
    //              ('LOADC', 1, 0),        // 加载常量操作，将常量表中索引为1的值加载到寄存器0
    //              ('OP', 0, 0),           // 操作指令，对寄存器0执行操作
    //              ('RET', 0, 0))),        // 返回指令，返回寄存器0的值
    //         ('operators', (('aten::linalg_pinv', '', 2),)),   // 运算符，指定使用的运算符为 aten::linalg_pinv，参数数量为2
    //         ('constants', (False, 1e-05)),  // 常量表，包含两个常量值（False, 1e-05），更新后的常量表
    //         ('types', ()),           // 类型信息，空元组表示没有类型信息
    //         ('register_size', 2)),   // 寄存器大小，设置为2个寄存器
    //     (('arguments',
    //         ((('name', 'self'), ('type', '__torch__.m'), ('default_value', None)),   // 参数信息，self 参数类型为 '__torch__.m'，默认值为 None
    //          (('name', 'input'), ('type', 'Tensor'), ('default_value', None)))),    // 参数信息，input 参数类型为 'Tensor'，默认值为 None
    //         ('returns',
    //             ((('name', ''), ('type', 'Tensor'), ('default_value', None)),))))   // 返回值信息，返回类型为 'Tensor'，默认值为 None
    
    // bytecode（字节码）带有三个指定参数：
    // （6，
    //     ('__torch__.m.forward',
    //         (('instructions',
    //             (('STOREN', 1, 2),        // 存储操作，将寄存器1中的值存储到寄存器2
    //              ('DROPR', 1, 0),        // 丢弃操作，丢弃寄存器1中的值
    //              ('MOVE', 2, 0),         // 移动操作，将寄存器0中的值移动到寄存器2
    //              ('LOADC', 1, 0),        // 加载常量操作，将常量表中索引为1的值加载到寄存器0
    //              ('LOADC', 0, 0),        // 加载常量操作，将常量表中索引为0的值加载到寄存器0
    //              ('OP', 0, 0),           // 操作指令，对寄存器0执行操作
    //              ('RET', 0, 0))),        // 返回指令，返回寄存器0的值
    //         ('operators', (('aten::linalg_pinv', '', 3),)),   // 运算符，指定使用的运算符为 aten::linalg_pinv，参数数量为3
    //         ('constants', (True, 1e-05)),  // 常量表，包含两个常量值（True, 1e-05）
    //         ('types', ()),           // 类型信息，空元组表示没有类型信息
    //         ('register_size', 2)),   // 寄存器大小，设置为2个寄存器
    //     (('arguments',
    //         ((('name', 'self'), ('type', '__torch__.m'), ('default_value', None)),   // 参数信息，self 参数类型为 '__torch__.m'，默认值为 None
    //          (('name', 'input'), ('type', 'Tensor'), ('default_value', None)))),    // 参数信息，input 参数类型为 'Tensor'，默认值为 None
    //         ('returns',
    //             ((('name', ''), ('type', 'Tensor'), ('default_value', None)),))))   // 返回值信息，返回类型为 'Tensor'，默认值为 None
#endif // !defined(FB_XPLAT_BUILD)

namespace {
// 定义静态变量 reg，用于在匿名命名空间内注册模块
static auto reg =


这段代码主要包含了条件编译结束符和一个匿名命名空间内的静态变量定义。条件编译 `#endif` 用于结束之前的条件编译指令 `#if !defined(FB_XPLAT_BUILD)`。匿名命名空间中的静态变量 `reg` 被用来注册模块，它的具体内容在下文未提供，但通常在类似的测试框架中，这样的注册操作用于自动化测试的初始化或其他类似的任务。
    // 定义一个名为 _TorchScriptTesting 的 TorchScript 类，绑定到 _FlatbufferTest
    torch::class_<TorchBindFlatbufferTestStruct>(
        "_TorchScriptTesting",
        "_FlatbufferTest")
        // 定义默认构造函数
        .def(torch::init<>())
        // 绑定名为 "get" 的成员函数，实现为 TorchBindFlatbufferTestStruct 类的 get 方法
        .def("get", &TorchBindFlatbufferTestStruct::get)
        // 定义 pickle 操作，序列化和反序列化该类的对象
        .def_pickle(
            // __getattr__ 函数，返回整数值 0
            [](const c10::intrusive_ptr<TorchBindFlatbufferTestStruct>& self)
                -> int64_t { return 0; },
            // __setattr__ 函数，接收整数状态值并返回一个新的 TorchBindFlatbufferTestStruct 对象
            [](int64_t state) {
              return c10::make_intrusive<TorchBindFlatbufferTestStruct>();
            });
} // namespace

TEST(FlatbufferTest, OperatorCacheDifferentiatesDefaultArgs) {
  // 创建三个方法：
  //
  // 1. forward() 返回一个数据类型为 torch.int64 的张量 (4)
  // 2. forward2() 返回一个数据类型为 torch.float32 的张量 (6)
  // 3. forward3() 返回一个数据类型为 torch.float32 的张量，但其数据类型由输入张量的数据类型推断得出
  //
  // 如果缓存正常工作，则完整的 JIT 模块和轻量级模块的结果将相同。否则，如果我们没有正确地忽略具有不同参数数量的运算符的缓存条目，结果将不同。
  Module m("m");
  // 定义 forward() 方法
  m.define(R"(
    def forward(self):
      ret1 = torch.new_empty(torch.zeros(10), [10], dtype=4)
      return ret1.fill_(25)
  )");
  // 定义 forward2() 方法
  m.define(R"(
    def forward2(self):
      ret1 = torch.new_empty(torch.zeros(10), [10], dtype=6)
      return ret1.fill_(32.0)
  )");
  // 定义 forward3() 方法
  m.define(R"(
    def forward3(self):
      ret1 = torch.new_empty(torch.zeros(10), [10])
      return ret1.fill_(12.0)
  )");

  std::vector<torch::jit::IValue> inputs;
  // 测试比较 JIT 模块和轻量级模块的张量结果
  testLiteModuleCompareResultTensors(m, inputs, "forward");
  testLiteModuleCompareResultTensors(m, inputs, "forward2");
  testLiteModuleCompareResultTensors(m, inputs, "forward3");
}

TEST(FlatbufferTest, OperatorSize1) {
  Module m("m");
  // 定义带有输入张量和缩放比例参数的 forward() 方法
  m.define(R"(
    def forward(self, input: Tensor, scale:float):
      return torch.upsample_nearest2d(input, [1, 1], float(scale), float(scale))
  )");

  CompilationOptions options;
  // 将 JIT 模块转换为移动端模块
  mobile::Module bc = jitModuleToMobile(m, options);
  const auto& func = bc.get_method("forward").function();
  // 断言操作输入大小与操作数大小相等
  ASSERT_EQ(
      func.get_code().operator_input_sizes_.size(),
      func.get_code().operators_.size());

  auto buff = save_mobile_module_to_bytes(bc);
  // 解析移动端模块的字节流
  mobile::Module bc2 = parse_mobile_module(buff->data(), buff->size());
  const auto& func2 = bc.get_method("forward").function();
  // 再次断言操作输入大小与操作数大小相等
  ASSERT_EQ(
      func2.get_code().operator_input_sizes_.size(),
      func2.get_code().operators_.size());
}

TEST(FlatbufferTest, BoolAndDoubleList) {
  Module m("m");
  c10::List<bool> boollist;
  boollist.push_back(false);
  IValue boollist_ival = boollist;
  IValue doublelist = std::vector<double>{2.0};
  // 注册布尔列表和双精度浮点数列表属性
  m.register_attribute("bool_list", boollist_ival.type(), boollist_ival);
  m.register_attribute("double_list", doublelist.type(), doublelist);

  CompilationOptions options;
  // 将 JIT 模块转换为移动端模块
  mobile::Module bc = jitModuleToMobile(m, options);
  auto buff = save_mobile_module_to_bytes(bc);
  // 解析移动端模块的字节流
  mobile::Module bc2 = parse_mobile_module(buff->data(), buff->size());

  // 如果读取的变量类型不正确，转换将引发异常
  // 读取布尔列表变量并验证
  auto boolval = bc2.attr("bool_list", {}).toBoolList().get(0);
  // 读取双精度浮点数列表变量并验证
  auto doubleval = bc2.attr("double_list", {}).toDoubleList().get(0);

  ASSERT_EQ(boolval, false);
  ASSERT_EQ(doubleval, 2.0);
}
TEST(FlatbufferTest, OperatorTest2) { // NOLINT (use =delete in gtest)
  const std::vector<std::string> test_programs{
      // test invoking a method with default parameter
      R"(
      def test_func(self, x, b : int = 4):
        return self.foo + x + b
      )",
      // inner method call with default parameter (gets inlined)
      R"(
      def add_with_default_arg(self, x, b : int = 4):
        return self.foo + x + b
      def test_func(self, x):
        return self.add_with_default_arg(x)  # invoke method w/ default arg
      )",
      // simple method call
      R"(
      def test_func(self, x):
        b = 4
        return self.foo + x + b
      )",
  };
  // 循环测试不同的程序片段
  for (const auto& test_program : test_programs) {
    // 创建名为 "m" 的模块对象
    Module m("m");
    // 注册名为 "foo" 的参数，使用 torch::ones({}) 初始化，不可变
    m.register_parameter("foo", torch::ones({}), false);
    // 定义当前迭代的测试程序片段
    m.define(test_program);

    // 设置编译选项
    CompilationOptions options;
    // 将 JIT 模块转换为移动端模块
    mobile::Module bc = jitModuleToMobile(m, options);
    // 获取名为 "test_func" 的方法对象
    const auto& func = bc.get_method("test_func").function();
    // 断言输入运算符大小等于运算符列表大小
    ASSERT_EQ(
        func.get_code().operator_input_sizes_.size(),
        func.get_code().operators_.size());

    // 保存移动端模块为字节流
    auto buff = save_mobile_module_to_bytes(bc);
    // 从字节流解析出移动端模块对象
    mobile::Module bc2 = parse_mobile_module(buff->data(), buff->size());
    // 再次获取名为 "test_func" 的方法对象
    const auto& func2 = bc.get_method("test_func").function();
    // 断言输入运算符大小等于运算符列表大小
    ASSERT_EQ(
        func2.get_code().operator_input_sizes_.size(),
        func2.get_code().operators_.size());
  }
}

Module jitModuleFromBuffer(void* data, size_t size) {
  // 复制数据以便使用现有 API，该 API 接受所有权。由于 `data` 参数可能指向缓冲区中间，因此不能直接安全地接管它的所有权。
  // @nolint CLANGTIDY cppcoreguidelines-no-malloc
  std::shared_ptr<char> copy(static_cast<char*>(malloc(size)), free);
  // 将数据复制到新分配的内存块中
  memcpy(copy.get(), data, size);

  // 初始化额外文件映射
  ExtraFilesMap extra_files;
  // 解析并初始化 JIT 模块
  return parse_and_initialize_jit_module(std::move(copy), size, extra_files);
}

TEST(TestSourceFlatbuffer, UpsampleNearest2d) {
  // 创建名为 "m" 的模块对象
  Module m("m");
  // 定义模块的前向传播方法
  m.define(R"(
    def forward(self, input: Tensor, scale:float):
      return torch.upsample_nearest2d(input, [1, 1], float(scale), float(scale))
  )");

  // 准备输入值列表
  std::vector<IValue> inputs;
  // 添加随机生成的张量作为输入
  inputs.emplace_back(torch::rand({1, 3, 128, 128}));
  // 添加标量值 2.0 作为输入
  inputs.emplace_back(at::Scalar(2.0));
  // 获取前向传播的参考输出
  auto ref = m.forward(inputs);

  // 创建流对象
  std::stringstream ss;
  // 将模块保存为移动端格式到流中，不使用压缩
  m._save_for_mobile(ss, {}, false, /*use_fatbuffer=*/true);
  // 从流中加载移动端模块
  auto mm = _load_for_mobile(ss);
  // 从流中加载模块
  auto m2 = load(ss);

  // 分别使用不同的加载模块对象执行前向传播
  auto res = m2.forward(inputs);
  auto resm = mm.forward(inputs);

  // 将结果转换为张量
  auto resd = res.toTensor();
  auto refd = ref.toTensor();
  auto resmd = resm.toTensor();
  // 断言两次加载模块的输出与参考输出相等
  ASSERT_TRUE(resd.equal(refd));
  ASSERT_TRUE(resmd.equal(refd));
}
TEST(TestSourceFlatbuffer, CheckAttrAccess) {
  // 创建一个名为 m 的模块对象
  Module m("m");
  // 向模块注册一个名为 "mobile_optimized" 的布尔类型属性，初始值为 true
  m.register_attribute("mobile_optimized", BoolType::get(), true);
  // 将模块 m 保存为字节流
  auto data = save_jit_module_to_bytes(m);
  // 从字节流中恢复模块对象 m2
  Module m2 = jitModuleFromBuffer(data->data(), data->size());
  // 获取 m2 中名为 "mobile_optimized" 的属性值，并转换为布尔类型
  bool mobile_optimized = m2.attr("mobile_optimized", false).toBool();
  // 断言 mobile_optimized 的值为 true
  AT_ASSERT(mobile_optimized);
  // 解析字节流中的 mobile 模块对象 m3
  mobile::Module m3 = parse_mobile_module(data->data(), data->size());
  // 获取 m3 中名为 "mobile_optimized" 的属性值，并转换为布尔类型
  mobile_optimized = m3.attr("mobile_optimized", false).toBool();
  // 断言 mobile_optimized 的值为 true
  AT_ASSERT(mobile_optimized);
}

TEST(TestSourceFlatbuffer,
     MethodInvocation) { // NOLINT (use =delete in gtest)
  const std::vector<std::string> test_programs{
      // 测试调用一个带有默认参数的方法
      R"(
      def test_func(self, x, b : int = 4):
        return self.foo + x + b
      )",
      // 内部调用带有默认参数的方法（将被内联）
      R"(
      def add_with_default_arg(self, x, b : int = 4):
        return self.foo + x + b
      def test_func(self, x):
        return self.add_with_default_arg(x)  # 调用带有默认参数的方法
      )",
      // 简单方法调用
      R"(
      def test_func(self, x):
        b = 4
        return self.foo + x + b
      )",
  };
  for (const auto& test_program : test_programs) {
    // 创建名为 m 的模块对象
    Module m("m");
    // 向模块注册名为 "foo" 的参数，初始值为 torch::ones({})，非参数化
    m.register_parameter("foo", torch::ones({}), false);
    // 定义模块 m 中的测试程序
    m.define(test_program);

    const int fortyTwo = 42; // (keep linter happy)
    // 创建 minput，其值为 fortyTwo 乘以 torch::ones({})
    auto minput = fortyTwo * torch::ones({});
    // 运行名为 "test_func" 的方法，返回 ref
    auto ref = m.run_method("test_func", minput);

    // 将模块 m 保存为字节流
    auto data = save_jit_module_to_bytes(m);
    // 从字节流中恢复模块对象 m2
    Module m2 = jitModuleFromBuffer(data->data(), data->size());
    // 获取 m2 中名为 "test_func" 的方法
    const auto& test_func = m2.get_method("test_func");
    IValue res;
    // 多次调用 test_func 方法
    for (int i = 0; i < 3; ++i) {
      res = test_func({minput});
    }
    // 将结果 res 转换为 float，并赋给 resd 和 refd
    auto resd = res.toTensor().item<float>();
    auto refd = ref.toTensor().item<float>();
    // 断言 resd 和 refd 的值相等
    AT_ASSERT(resd == refd);

    // 解析字节流中的 mobile 模块对象 m3
    mobile::Module m3 = parse_mobile_module(data->data(), data->size());
    // 获取 m3 中名为 "test_func" 的方法
    const auto& test_func3 = m3.get_method("test_func");
    // 多次调用 test_func3 方法
    for (int i = 0; i < 3; ++i) {
      res = test_func3({minput});
    }
    // 将结果 res 转换为 float，并赋给 resd 和 refd
    resd = res.toTensor().item<float>();
    refd = ref.toTensor().item<float>();
    // 断言 resd 和 refd 的值相等
    AT_ASSERT(resd == refd);
  }
}

#if !defined FB_XPLAT_BUILD
// 以下测试仅在 fbcode 中运行
TEST(FlatbufferUpgraderTest, DivTensorV2) {
  // 获取当前文件路径
  std::string filePath(__FILE__);
  // 构造测试模型文件的路径
  auto test_model_file = filePath.substr(0, filePath.find_last_of("/\\") + 1);
  test_model_file.append("upgrader_models/test_versioned_div_tensor_v2.ptl.ff");
  /*
  (('__torch__.MyModule.forward',
    // 定义一个包含指令序列的嵌套元组，描述了模型的计算图
    (('instructions',
      (('STOREN', 1, 3),        // 将计算结果存储到寄存器索引为1的位置，共3个寄存器
       ('DROPR', 1, 0),         // 丢弃寄存器索引为1的值
       ('LOAD', 2, 0),          // 载入寄存器索引为2的值
       ('LOAD', 3, 0),          // 载入寄存器索引为3的值
       ('OP', 0, 0),            // 执行操作（操作码为0）
       ('LOAD', 2, 0),          // 载入寄存器索引为2的值
       ('LOAD', 3, 0),          // 载入寄存器索引为3的值
       ('OP', 1, 0),            // 执行操作（操作码为1）
       ('MOVE', 2, 0),          // 将寄存器索引为2的值移动到索引为0的位置
       ('MOVE', 3, 0),          // 将寄存器索引为3的值移动到索引为0的位置
       ('OP', 2, 0),            // 执行操作（操作码为2）
       ('TUPLE_CONSTRUCT', 3, 0),  // 构造一个包含3个元素的元组
       ('RET', 0, 0))),          // 返回寄存器索引为0的值
     ('operators',
      (('aten::div', 'Tensor'),  // 使用 aten::div 操作处理 Tensor
       ('aten::div', 'Tensor'),  // 使用 aten::div 操作处理 Tensor
       ('aten::div', 'Tensor'))),  // 使用 aten::div 操作处理 Tensor
     ('constants', ()),          // 没有常量
     ('types', ()),              // 没有类型信息
     ('register_size', 3))),)    // 寄存器大小为3

  */
  // 从文件中加载移动模块
  mobile::Module m_module = load_mobile_module_from_file(test_model_file);
  // 获取名为 "forward" 的方法的代码对象，进而获取指令列表
  auto intrsuction_list =
      m_module.get_method("forward").function().get_code().instructions_;
  uint64_t number_of_call_instruction = 0;
  // 遍历指令列表，统计 OpCode 为 CALL 的指令数量
  for (auto& instruction : intrsuction_list) {
    number_of_call_instruction += (instruction.op == OpCode::CALL);
  }
  // 断言 CALL 指令的数量为3
  ASSERT_EQ(number_of_call_instruction, 3);

  // 准备输入数据，创建包含两个 Tensor 的 IValue 向量
  std::vector<IValue> inputs = {
      IValue(6 * torch::ones({1})), IValue(3 * torch::ones({1}))};
  // 执行模型的前向传播，获取实际输出
  auto actual_output = m_module.forward(inputs);
  // 创建期望输出的 Tensor（值为2.0）
  auto expect_output = 2.0 * torch::ones({1});
  // 将实际输出转换为元组，获取其元素列表
  auto actual_output_list = actual_output.toTuple()->elements();
  // 断言实际输出列表中第一个元素的 Tensor 与期望输出相等
  ASSERT_TRUE(actual_output_list[0].toTensor().equal(expect_output));
TEST(FlatbufferUpgraderTest, DivTensorOutV2) {
  // 获取当前文件的路径
  std::string filePath(__FILE__);
  // 构建测试模型文件的完整路径
  auto test_model_file = filePath.substr(0, filePath.find_last_of("/\\") + 1);
  test_model_file.append(
      "upgrader_models/test_versioned_div_tensor_out_v2.ptl.ff");
  /*
  (('__torch__.MyModule.forward',
    (('instructions',
      (('STOREN', 1, 4),
       ('DROPR', 1, 0),
       ('MOVE', 2, 0),
       ('MOVE', 3, 0),
       ('MOVE', 4, 0),
       ('OP', 0, 0),
       ('RET', 0, 0))),
     ('operators', (('aten::div', 'out'),)),
     ('constants', ()),
     ('types', ()),
     ('register_size', 4))),)
  */
  // 从文件加载移动端模块
  mobile::Module m_module = load_mobile_module_from_file(test_model_file);

  // 获取前向方法的指令列表
  auto instruction_list =
      m_module.get_method("forward").function().get_code().instructions_;
  // 统计 CALL 操作码的出现次数
  uint64_t number_of_call_instruction = 0;
  for (auto& instruction : instruction_list) {
    number_of_call_instruction += (instruction.op == OpCode::CALL);
  }
  // 断言仅有一个操作使用了升级器
  ASSERT_EQ(number_of_call_instruction, 1);

  // 设置输入值
  std::vector<IValue> inputs{
      IValue(6 * torch::ones({1})),
      IValue(3 * torch::ones({1})),
      IValue(torch::empty({1}))};
  // 调用模块的前向方法
  m_module.forward(inputs);
  // 预期的输出为 2.0 乘以全1张量
  auto expect_output = 2.0 * torch::ones({1});
  // 获取实际输出
  auto actual_output = inputs[2].toTensor();
  // 断言输出张量等于预期输出
  // 输出参数将被输出值覆盖
  ASSERT_TRUE(actual_output.equal(expect_output));
}

TEST(FlatbufferUpgraderTest, DivTensorInplaceV2) {
  // 获取当前文件的路径
  std::string filePath(__FILE__);
  // 构建测试模型文件的完整路径
  auto test_model_file = filePath.substr(0, filePath.find_last_of("/\\") + 1);
  test_model_file.append(
      "upgrader_models/test_versioned_div_tensor_inplace_v2.ptl.ff");
  /*
  (('__torch__.MyModule.forward',
    (('instructions',
      (('STOREN', 1, 3),
       ('DROPR', 1, 0),
       ('MOVE', 2, 0),
       ('MOVE', 3, 0),
       ('OP', 0, 0),
       ('RET', 0, 0))),
     ('operators', (('aten::div_', 'Tensor'),)),
     ('constants', ()),
     ('types', ()),
     ('register_size', 3))),)
  */
  // 从文件加载移动端模块
  mobile::Module m_module = load_mobile_module_from_file(test_model_file);

  // 获取前向方法的指令列表
  auto instruction_list =
      m_module.get_method("forward").function().get_code().instructions_;
  // 统计 CALL 操作码的出现次数
  uint64_t number_of_call_instruction = 0;
  for (auto& instruction : instruction_list) {
    number_of_call_instruction += (instruction.op == OpCode::CALL);
  }
  // 断言仅有一个操作使用了升级器
  ASSERT_EQ(number_of_call_instruction, 1);

  // 设置输入值
  std::vector<IValue> inputs{
      IValue(6 * torch::ones({1})), IValue(3 * torch::ones({1}))};
  // 调用模块的前向方法
  m_module.forward(inputs);
  // 预期的输出为 2.0 乘以全1张量
  auto expect_output = 2.0 * torch::ones({1});
  // 获取实际输出
  auto actual_output = inputs[0].toTensor();
  // 断言输出张量等于预期输出
  // 输出参数将被输出值覆盖
  ASSERT_TRUE(actual_output.equal(expect_output));
}
TEST(FlatbufferUpgraderTest, DivScalarFloatV2) {
  // 获取当前文件路径
  std::string filePath(__FILE__);
  // 构建测试模型文件的完整路径
  auto test_model_file = filePath.substr(0, filePath.find_last_of("/\\") + 1);
  test_model_file.append(
      "upgrader_models/test_versioned_div_scalar_float_v2.ptl.ff");
  /*
  示例 FlatBuffer 的模型定义：
  ((__torch__.MyModuleFloat.forward',
    (('instructions',
    (('STOREN', 1, 3),
    ('DROPR', 1, 0),
    ('MOVE', 2, 0),
    ('MOVE', 3, 0),
    ('OP', 0, 0),
    ('RET', 0, 0))),
    ('operators', (('aten::div', 'Scalar'),)),
    ('constants', ()),
    ('types', ()),
    ('register_size', 3))),)
  */

  // 从文件加载移动端模型
  mobile::Module m_module = load_mobile_module_from_file(test_model_file);

  // 获取模型中 forward 方法的指令列表
  auto instruction_list =
      m_module.get_method("forward").function().get_code().instructions_;
  uint64_t number_of_call_instructions = 0;
  // 统计 CALL 操作码的数量
  for (auto& instruction : instruction_list) {
    number_of_call_instructions += (instruction.op == OpCode::CALL);
  }
  // 断言仅有一个操作会使用升级器
  ASSERT_EQ(number_of_call_instructions, 1);

  // 准备输入数据
  std::vector<IValue> inputs{IValue(6 * torch::ones({1})), IValue(3.0)};
  // 执行模型的 forward 方法
  auto output = m_module.forward(inputs);
  auto expected_output = 2.0 * torch::ones({1});
  auto actual_output = output.toTensor();

  // 断言模型输出与期望输出相等
  ASSERT_TRUE(actual_output.equal(expected_output));
}

TEST(FlatbufferUpgraderTest, DivScalarReciprocalFloatV2) {
  // 获取当前文件路径
  std::string filePath(__FILE__);
  // 构建测试模型文件的完整路径
  auto test_model_file = filePath.substr(0, filePath.find_last_of("/\\") + 1);
  test_model_file.append(
      "upgrader_models/test_versioned_div_scalar_reciprocal_float_v2.ptl.ff");
  /*
  示例 FlatBuffer 的模型定义：
  ((__torch__.MyModuleFloat.forward',
    (('instructions',
      (('STOREN', 1, 3),
      ('DROPR', 1, 0),
      ('MOVE', 2, 0),
      ('OP', 0, 0),
      ('MOVE', 3, 0),
      ('OP', 1, 0),
      ('RET', 0, 0))),
    ('operators', (('aten::reciprocal', ''), ('aten::mul', 'Scalar'))),
    ('constants', ()),
    ('types', ()),
    ('register_size', 3))),)
  */
  // 从文件加载移动端模型
  mobile::Module m_module = load_mobile_module_from_file(test_model_file);

  // 获取模型中 forward 方法的指令列表
  auto instruction_list =
      m_module.get_method("forward").function().get_code().instructions_;
  uint64_t number_of_call_instructions = 0;
  // 统计 CALL 操作码的数量
  for (auto& instruction : instruction_list) {
    number_of_call_instructions += (instruction.op == OpCode::CALL);
  }
  // 断言没有操作会使用升级器
  ASSERT_EQ(number_of_call_instructions, 0);

  // 准备输入数据
  std::vector<IValue> inputs{IValue(6 * torch::ones({1})), IValue(3.0)};
  // 执行模型的 forward 方法
  auto output = m_module.forward(inputs);
  auto expected_output = 0.5 * torch::ones({1});
  auto actual_output = output.toTensor();

  // 断言模型输出与期望输出相等
  ASSERT_TRUE(actual_output.equal(expected_output));
}
TEST(FlatbufferUpgraderTest, DivScalarReciprocalIntV2) {
  // 获取当前测试文件的路径
  std::string filePath(__FILE__);
  // 提取出测试模型文件的完整路径
  auto test_model_file = filePath.substr(0, filePath.find_last_of("/\\") + 1);
  test_model_file.append(
      "upgrader_models/test_versioned_div_scalar_reciprocal_int_v2.ptl.ff");
  /*
  (('__torch__.MyModuleInt.forward',
  (('instructions',
    (('STOREN', 1, 3),
     ('DROPR', 1, 0),
     ('MOVE', 2, 0),
     ('OP', 0, 0),
     ('MOVE', 3, 0),
     ('OP', 1, 0),
     ('RET', 0, 0))),
   ('operators', (('aten::reciprocal', ''), ('aten::mul', 'Scalar'))),
   ('constants', ()),
   ('types', ()),
   ('register_size', 3))),)
  */
  // 从文件加载移动模块
  mobile::Module m_module = load_mobile_module_from_file(test_model_file);

  // 获取模块的前向方法的指令列表
  auto intrsuction_list =
      m_module.get_method("forward").function().get_code().instructions_;
  // 统计调用指令的数量
  uint64_t number_of_call_instruction = 0;
  for (auto& instruction : intrsuction_list) {
    number_of_call_instruction += (instruction.op == OpCode::CALL);
  }
  // 断言没有调用指令使用升级器
  ASSERT_EQ(number_of_call_instruction, 0);

  // 准备输入向量
  std::vector<IValue> inputs{IValue(6 * torch::ones({1})), IValue(3.0)};
  // 执行模块的前向传播
  auto output = m_module.forward(inputs);
  // 准备期望输出
  auto expect_output = 0.5 * torch::ones({1});
  auto actual_output = output.toTensor();

  // 断言实际输出与期望输出相等
  // 输出参数将会被输出重写为实际输出
  ASSERT_TRUE(actual_output.equal(expect_output));
}

TEST(FlatbufferUpgraderTest, DivScalarScalarV2) {
  std::string filePath(__FILE__);
  auto test_model_file = filePath.substr(0, filePath.find_last_of("/\\") + 1);
  test_model_file.append(
      "upgrader_models/test_versioned_div_scalar_scalar_v2.ptl.ff");
  /*
  (('__torch__.MyModule.forward',
    (('instructions',
      (('STOREN', 1, 5),
      ('DROPR', 1, 0),
      ('LOAD', 2, 0),
      ('LOAD', 3, 0),
      ('OP', 0, 0),
      ('MOVE', 2, 0),
      ('LOAD', 4, 0),
      ('OP', 1, 0),
      ('LOAD', 3, 0),
      ('MOVE', 4, 0),
      ('OP', 2, 0),
      ('MOVE', 3, 0),
      ('MOVE', 5, 0),
      ('OP', 3, 0),
      ('TUPLE_CONSTRUCT', 4, 0),
      ('RET', 0, 0))),
    ('operators',
      (('aten::div', ''),
      ('aten::div', 'float'),
      ('aten::div', ''),
      ('aten::div', 'int'))),
    ('constants', ()),
    ('types', ()),
    ('register_size', 5))),)
  */
  // 从文件加载移动模块
  mobile::Module m_module = load_mobile_module_from_file(test_model_file);
  // 获取模块的前向方法的指令列表
  auto intrsuction_list =
      m_module.get_method("forward").function().get_code().instructions_;
  // 统计调用指令的数量
  uint64_t number_of_call_instruction = 0;
  for (auto& instruction : intrsuction_list) {
    // 如果指令是调用操作码，则增加计数
    number_of_call_instruction += (instruction.op == OpCode::CALL);
  }
    // 增加调用指令的计数，如果当前指令是 CALL 操作码，则计数加一
    number_of_call_instruction += (instruction.op == OpCode::CALL);
  }
  // 断言：没有操作符会使用 upgrader
  ASSERT_EQ(number_of_call_instruction, 0);

  // 创建一个包含四个 IValue 元素的输入向量，分别为 20.0、10、2.0、5
  std::vector<IValue> inputs{IValue(20.0), IValue(10), IValue(2.0), IValue(5)};
  // 对模块 m_module 执行前向推理，使用上述输入向量作为输入，获得输出结果
  auto output = m_module.forward(inputs);
  // 从输出中获取元组的引用，并将其元素保存到 output_list 中
  auto output_list = output.toTupleRef().elements();
  // 期望的输出结果是包含四个 IValue 元素的向量，依次为 2.0、10.0、5.0、2.0
  auto expect_output = std::vector<IValue>(
      {IValue(2.0), IValue(10.0), IValue(5.0), IValue(2.0)});
  // 遍历期望输出和实际输出列表，逐个断言它们相等
  for (size_t i = 0; i < expect_output.size(); i++) {
    ASSERT_EQ(output_list[i], expect_output[i]);
  }
// 定义一个名为 DivScalarIntV2 的测试用例
TEST(FlatbufferUpgraderTest, DivScalarIntV2) {
  // 获取当前文件的路径
  std::string filePath(__FILE__);
  // 将测试模型文件的路径添加到文件路径后面
  auto test_model_file = filePath.substr(0, filePath.find_last_of("/\\") + 1);
  test_model_file.append(
      "upgrader_models/test_versioned_div_scalar_int_v2.ptl.ff");
  /*
  下面是模型的序列化表示，包含模型的前向方法及其指令、操作符、常量等信息
  (('__torch__.MyModuleInt.forward',
    (('instructions',
      (('STOREN', 1, 3),
      ('DROPR', 1, 0),
      ('MOVE', 2, 0),
      ('MOVE', 3, 0),
      ('OP', 0, 0),
      ('RET', 0, 0))),
    ('operators', (('aten::div', 'Scalar'),)),
    ('constants', ()),
    ('types', ()),
    ('register_size', 3))),)
  */
  
  // 从文件中加载移动模块
  mobile::Module m_module = load_mobile_module_from_file(test_model_file);

  // 获取模块前向方法的指令列表
  auto intrsuction_list =
      m_module.get_method("forward").function().get_code().instructions_;
  // 统计调用指令中操作码为 CALL 的数量
  uint64_t number_of_call_instruction = 0;
  for (auto& instruction : intrsuction_list) {
    number_of_call_instruction += (instruction.op == OpCode::CALL);
  }
  // 断言只有一个操作码为 CALL 的指令
  // 这表明仅有一个操作符将使用升级器
  ASSERT_EQ(number_of_call_instruction, 1);

  // 准备输入参数列表
  std::vector<IValue> inputs{IValue(6 * torch::ones({1})), IValue(3)};
  // 调用模块的前向方法进行推理
  auto output = m_module.forward(inputs);
  // 期望的输出结果
  auto expect_output = 2.0 * torch::ones({1});
  // 实际的输出结果
  auto actual_output = output.toTensor();

  // 断言实际输出结果与期望输出结果相等
  ASSERT_TRUE(actual_output.equal(expect_output));
}



// 定义一个名为 DivScalarInplaceFloatV2 的测试用例
TEST(FlatbufferUpgraderTest, DivScalarInplaceFloatV2) {
  // 获取当前文件的路径
  std::string filePath(__FILE__);
  // 将测试模型文件的路径添加到文件路径后面
  auto test_model_file = filePath.substr(0, filePath.find_last_of("/\\") + 1);
  test_model_file.append(
      "upgrader_models/test_versioned_div_scalar_inplace_float_v2.ptl.ff");
  /*
  下面是模型的序列化表示，包含模型的前向方法及其指令、操作符、常量等信息
  (('__torch__.MyModuleFloat.forward',
    (('instructions',
      (('STOREN', 1, 3),
      ('DROPR', 1, 0),
      ('MOVE', 2, 0),
      ('MOVE', 3, 0),
      ('OP', 0, 0),
      ('RET', 0, 0))),
    ('operators', (('aten::div_', 'Scalar'),)),
    ('constants', ()),
    ('types', ()),
    ('register_size', 3))),)
  */

  // 从文件中加载移动模块
  mobile::Module m_module = load_mobile_module_from_file(test_model_file);

  // 获取模块前向方法的指令列表
  auto intrsuction_list =
      m_module.get_method("forward").function().get_code().instructions_;
  // 统计调用指令中操作码为 CALL 的数量
  uint64_t number_of_call_instruction = 0;
  for (auto& instruction : intrsuction_list) {
    number_of_call_instruction += (instruction.op == OpCode::CALL);
  }
  // 断言只有一个操作码为 CALL 的指令
  // 这表明仅有一个操作符将使用升级器
  ASSERT_EQ(number_of_call_instruction, 1);

  // 准备输入参数列表
  std::vector<IValue> inputs{IValue(6 * torch::ones({1})), IValue(3.0)};
  // 调用模块的前向方法进行推理
  auto output = m_module.forward(inputs);
  // 期望的输出结果
  auto expect_output = 2.0 * torch::ones({1});
  // 实际的输出结果
  auto actual_output = output.toTensor();

  // 断言实际输出结果与期望输出结果相等
  ASSERT_TRUE(actual_output.equal(expect_output));
}



// 定义一个名为 DivScalarInplaceIntV2 的测试用例
TEST(FlatbufferUpgraderTest, DivScalarInplaceIntV2) {
  // 获取当前文件的路径
  std::string filePath(__FILE__);
  // 将测试模型文件的路径添加到文件路径后面
  auto test_model_file = filePath.substr(0, filePath.find_last_of("/\\") + 1);
  test_model_file.append(
      "upgrader_models/test_versioned_div_scalar_inplace_int_v2.ptl.ff");
  /*
  下面是模型的序列化表示，包含模型的前向方法及其指令、操作符、常量等信息
  (('__torch__.MyModuleInt.forward',
    (('instructions',
      (('STOREN', 1, 3),
      ('DROPR', 1, 0),
      ('MOVE', 2, 0),
      ('MOVE', 3, 0),
      ('OP', 0, 0),
      ('RET', 0, 0))),
    ('operators', (('aten::div_', 'Scalar'),)),
    ('constants', ()),
    ('types', ()),
    ('register_size', 3))),)
  */

  // 从文件中加载移动模块
  mobile::Module m_module = load_mobile_module_from_file(test_model_file);

  // 获取模块前向方法的指令列表
  auto intrsuction_list =
      m_module.get_method("forward").function().get_code().instructions_;
  // 统计调用指令中操作码为 CALL 的数量
  uint64_t number_of_call_instruction = 0;
  for (auto& instruction : intrsuction_list) {
    number_of_call_instruction += (instruction.op == OpCode::CALL);
  }
  // 断言只有一个操作码为 CALL 的指令
  // 这表明仅有一个操作符将使用升级器
  ASSERT_EQ(number_of_call_instruction, 1);

  // 准备输入参数列表
  std::vector<IValue> inputs{IValue(6 * torch::ones({1})), IValue(3)};
  // 调用模块的前向方法进行推理
  auto output = m_module.forward(inputs);
  // 期望的输出结果
  auto expect_output = 2.0 * torch::ones({1});
  // 实际的输出结果
  auto actual_output = output.toTensor();

  // 断言实际输出结果与期望输出结果相等
  ASSERT_TRUE(actual_output.equal(expect_output));
}
  /*
    定义一个复杂的数据结构，描述了模型的指令序列、操作符、常量、类型和寄存器大小
  */
  mobile::Module m_module = load_mobile_module_from_file(test_model_file);

  // 获取模型中 "forward" 方法的指令列表
  auto intrsuction_list =
      m_module.get_method("forward").function().get_code().instructions_;
  
  // 统计指令列表中 CALL 操作码的出现次数
  uint64_t number_of_call_instruction = 0;
  for (auto& instruction : intrsuction_list) {
    number_of_call_instruction += (instruction.op == OpCode::CALL);
  }
  
  // 断言模型中 CALL 操作码的出现次数为 1
  ASSERT_EQ(number_of_call_instruction, 1);

  // 准备模型输入，一个是大小为 [1] 的全 6 张量，另一个是标量 3
  std::vector<IValue> inputs{IValue(6 * torch::ones({1})), IValue(3)};
  
  // 执行模型的 forward 方法，获取输出
  auto output = m_module.forward(inputs);
  
  // 期望的输出是一个大小为 [1] 的全 2 张量
  auto expect_output = 2.0 * torch::ones({1});
  
  // 实际的输出张量
  auto actual_output = output.toTensor();
  
  // 断言实际输出与期望输出张量相等
  ASSERT_TRUE(actual_output.equal(expect_output));
} // namespace jit
} // namespace torch
namespace torch {
namespace jit {



/**
 * An Allocator that can only deallocate (using delete []), counting
 * the number of times that it has been asked to deallocate.
 */
class TestAllocator : public flatbuffers::Allocator {
 public:
  /**
   * *deallocate_call_count will be incremented whenever deallocate() is called.
   */
  explicit TestAllocator(int* deallocate_call_count)
      : deallocate_call_count_(deallocate_call_count) {}

  // 释放内存块并增加对释放次数的计数
  void deallocate(uint8_t* p, size_t /*size*/) override {
    *deallocate_call_count_ += 1;
    delete[] p;
  }

  // 不应该调用 allocate()，如果调用则抛出错误
  uint8_t* allocate(size_t) override {
    TORCH_CHECK(false, "allocate() should not be called");
  }

  // 不应该调用 reallocate_downward()，如果调用则抛出错误
  uint8_t* reallocate_downward(uint8_t*, size_t, size_t, size_t, size_t)
      override {
    TORCH_CHECK(false, "reallocate_downward() should not be called");
  }

 private:
  int* deallocate_call_count_;
};



/// Provides access to DetachedBuffer::destroy().
struct DetachedBufferTestingFriend {
  /**
   * Returns a UniqueDetachedBuffer that wraps the provided DetachedBuffer.
   * A copy of similar code in flatbuffer_serializer.cpp.
   */
  static DetachedBuffer::UniqueDetachedBuffer make_unique_detached_buffer(
      DetachedBuffer* buf) {
    return DetachedBuffer::UniqueDetachedBuffer(buf, DetachedBuffer::destroy);
  }
};
TEST(FlatbufferTest, DetachedBufferSmoke) {
    // 在这个测试函数中，测试 flatbuffers::DetachedBuffer 的生命周期和行为

    // 自定义分配器 TestAllocator，用于监视 flatbuffers::DetachedBuffer 的生命周期
    int deallocate_call_count = 0;
    TestAllocator alloc(&deallocate_call_count);

    // 分配一个大小为 4 的 uint8_t 数组作为数据
    constexpr size_t data_size = 4;
    uint8_t* data = new uint8_t[data_size];

    // 创建一个 flatbuffers::DetachedBuffer 对象，但不拥有分配器
    flatbuffers::DetachedBuffer fb_buf_local(
        &alloc, /*own_allocator=*/false, data, data_size, data, data_size);
    EXPECT_EQ(fb_buf_local.data(), data);
    EXPECT_EQ(fb_buf_local.size(), data_size);

    // 模仿 save_mobile_module_to_bytes 函数内部的代码，将所有权转移给堆对象
    auto fb_buf_ptr = new flatbuffers::DetachedBuffer(std::move(fb_buf_local));
    // 这时数据还未被删除
    EXPECT_EQ(deallocate_call_count, 0);
    // 新对象指向同样的数据
    EXPECT_EQ(fb_buf_ptr->data(), data);
    EXPECT_EQ(fb_buf_ptr->size(), data_size);
    // 旧对象不再指向任何内容
    // @lint-ignore CLANGTIDY bugprone-use-after-move
    EXPECT_EQ(fb_buf_local.data(), nullptr);
    // @lint-ignore CLANGTIDY bugprone-use-after-move
    EXPECT_EQ(fb_buf_local.size(), 0);

    // 创建顶层的 torch::jit::DetachedBuffer
    auto wrapped_buf =
        new DetachedBuffer(fb_buf_ptr->data(), fb_buf_ptr->size(), fb_buf_ptr);
    EXPECT_EQ(wrapped_buf->data(), data);
    EXPECT_EQ(wrapped_buf->size(), data_size);

    // 拥有 torch::jit::DetachedBuffer 及其内容的 unique_ptr
    {
        DetachedBuffer::UniqueDetachedBuffer unique_buf =
            DetachedBufferTestingFriend::make_unique_detached_buffer(wrapped_buf);
        EXPECT_EQ(unique_buf->data(), data);
        EXPECT_EQ(unique_buf->size(), data_size);

        // 数据此时还未被删除
        EXPECT_EQ(deallocate_call_count, 0);
    }

    // 现在 unique_ptr 超出范围，数据应该已被删除
    EXPECT_EQ(deallocate_call_count, 1);
}

TEST(FlatbufferTest, DetachedBufferNullOwner) {
    // 测试具有空内部所有者的 torch::jit::DetachedBuffer

    // 创建一个包含 4 个字节的 vector
    std::vector<uint8_t> data(4);
    auto wrapped_buf = new DetachedBuffer(data.data(), data.size());

    // 拥有 torch::jit::DetachedBuffer 及其内容的 unique_ptr
    {
        DetachedBuffer::UniqueDetachedBuffer unique_buf =
            DetachedBufferTestingFriend::make_unique_detached_buffer(wrapped_buf);
        EXPECT_EQ(unique_buf->data(), data.data());
        EXPECT_EQ(unique_buf->size(), data.size());
    }

    // 当 UniqueDetachedBuffer 超出范围时，DetachedBuffer 应该已被销毁。
    // 如果没有崩溃或 ASAN（地址无关存取检查器）警告，应该是正常的。
}
```