# `.\pytorch\test\cpp\jit\test_save_load.cpp`

```
#include <gtest/gtest.h> // 引入 Google Test 框架的头文件

#include <test/cpp/jit/test_utils.h> // 引入测试实用工具的头文件
#include <cstdlib> // 引入 C 标准库中的通用函数
#include <iostream> // 引入输入输出流库
#include <sstream> // 引入字符串流库

#include <caffe2/serialize/inline_container.h> // 引入 Caffe2 序列化相关的头文件
#include <torch/csrc/jit/mobile/module.h> // 引入 Torch 移动端模块相关的头文件
#include <torch/csrc/jit/runtime/calculate_necessary_args.h> // 引入 Torch JIT 运行时的参数计算头文件
#include <torch/csrc/jit/serialization/export.h> // 引入 Torch JIT 序列化导出头文件
#include <torch/csrc/jit/serialization/export_bytecode.h> // 引入 Torch JIT 字节码导出头文件
#include <torch/csrc/jit/serialization/import.h> // 引入 Torch JIT 序列化导入头文件
#include <torch/csrc/jit/serialization/import_source.h> // 引入 Torch JIT 源码导入头文件
#include <torch/script.h> // 引入 Torch 脚本模块
#include <torch/torch.h> // 引入 Torch 核心库

#include "caffe2/serialize/istream_adapter.h" // 引入 Caffe2 输入流适配器头文件

namespace torch {
namespace jit {

namespace {

Module roundtripThroughMobile(const Module& m) {
  ExtraFilesMap files; // 定义一个额外文件的映射
  std::vector<IValue> constants; // 定义常量向量
  jitModuleToPythonCodeAndConstants(m, &files, &constants); // 将 JIT 模块转换为 Python 代码和常量
  CompilationOptions options; // 定义编译选项
  mobile::Module mobilem = jitModuleToMobile(m, options); // 将 JIT 模块转换为移动端模块
  return jitModuleFromSourceAndConstants(
      mobilem._ivalue(), files, constants, 8); // 从源码和常量中重新构建 JIT 模块
}

template <class Functor>
inline void expectThrowsEq(Functor&& functor, const char* expectedMessage) {
  try {
    std::forward<Functor>(functor)(); // 调用传入的函数对象
  } catch (const Error& e) {
    EXPECT_STREQ(e.what_without_backtrace(), expectedMessage); // 断言捕获的错误消息与预期的消息相同
    return;
  }
  ADD_FAILURE() << "Expected to throw exception with message \""
                << expectedMessage << "\" but didn't throw"; // 添加测试失败消息，指示未抛出预期异常
}

} // namespace

TEST(SerializationTest, ExtraFilesHookPreference) {
  // 测试显式写入的额外文件优先于钩子函数写入的额外文件
  const auto script = R"JIT(
    def forward(self):
        x = torch.rand(5, 5)
        x = x.mm(x)
        return x
  )JIT";

  auto module =
      std::make_shared<Module>("Module", std::make_shared<CompilationUnit>()); // 创建共享指针的模块对象
  module->define(script); // 定义模块的脚本内容
  std::ostringstream oss; // 创建输出字符串流对象
  std::unordered_map<std::string, std::string> extra_files; // 创建无序映射，存储额外文件名和内容
  extra_files["metadata.json"] = "abc"; // 设置额外文件的键值对
  SetExportModuleExtraFilesHook([](const Module&) -> ExtraFilesMap { // 设置导出模块额外文件钩子函数
    return {{"metadata.json", "def"}}; // 返回固定的额外文件内容
  });
  module->save(oss, extra_files); // 保存模块及额外文件到输出字符串流
  SetExportModuleExtraFilesHook(nullptr); // 清除导出模块额外文件钩子函数

  std::istringstream iss(oss.str()); // 创建输入字符串流对象并初始化
  caffe2::serialize::IStreamAdapter adapter{&iss}; // 创建输入流适配器对象
  std::unordered_map<std::string, std::string> loaded_extra_files; // 创建无序映射，存储加载的额外文件名和内容
  loaded_extra_files["metadata.json"] = ""; // 初始化加载的额外文件内容为空字符串
  auto loaded_module = torch::jit::load(iss, torch::kCPU, loaded_extra_files); // 加载模块及其额外文件
  ASSERT_EQ(loaded_extra_files["metadata.json"], "abc"); // 断言加载的额外文件内容与预期相同
}

TEST(SerializationTest, ExtraFileHooksNoSecret) {
  // 没有秘密
  std::stringstream ss; // 创建字符串流对象
  {
    Module m("__torch__.m"); // 创建名为 "__torch__.m" 的模块对象
    ExtraFilesMap extra; // 创建额外文件映射
    extra["metadata.json"] = "abc"; // 设置额外文件的键值对
    m.save(ss, extra); // 将模块及其额外文件保存到字符串流
  }
  ss.seekg(0); // 将字符串流的读取位置设置到流的起始位置
  {
    ExtraFilesMap extra; // 创建额外文件映射
    extra["metadata.json"] = ""; // 初始化额外文件内容为空字符串
    extra["secret.json"] = ""; // 设置额外文件的另一个键值对
    jit::load(ss, c10::nullopt, extra); // 加载模块及其额外文件
    ASSERT_EQ(extra["metadata.json"], "abc"); // 断言加载的额外文件内容与预期相同
    ASSERT_EQ(extra["secret.json"], ""); // 断言加载的额外文件内容与预期相同
  }
}
// 定义名为 SerializationTest 的测试套件，包含 ExtraFileHooksWithSecret 单元测试
TEST(SerializationTest, ExtraFileHooksWithSecret) {
  // 创建一个字符串流对象 ss
  std::stringstream ss;
  {
    // 设置导出模块的额外文件钩子，返回一个固定的映射 {"secret.json": "topsecret"}
    SetExportModuleExtraFilesHook([](const Module&) -> ExtraFilesMap {
      return {{"secret.json", "topsecret"}};
    });
    // 创建一个名为 m 的模块对象 "__torch__.m"
    Module m("__torch__.m");
    // 创建一个额外文件的映射 extra，包含 {"metadata.json": "abc"}
    ExtraFilesMap extra;
    extra["metadata.json"] = "abc";
    // 将模块 m 的内容保存到字符串流 ss 中
    m.save(ss, extra);
    // 取消导出模块的额外文件钩子
    SetExportModuleExtraFilesHook(nullptr);
  }
  // 将字符串流 ss 的读指针定位到开头
  ss.seekg(0);
  {
    // 创建一个额外文件的映射 extra
    ExtraFilesMap extra;
    // 将额外文件映射 extra 初始化为空的 "metadata.json" 和 "secret.json" 键
    extra["metadata.json"] = "";
    extra["secret.json"] = "";
    // 从字符串流 ss 中加载数据，并附加额外文件映射 extra
    jit::load(ss, c10::nullopt, extra);
    // 断言加载后的 "metadata.json" 等于 "abc"
    ASSERT_EQ(extra["metadata.json"], "abc");
    // 断言加载后的 "secret.json" 等于 "topsecret"
    ASSERT_EQ(extra["secret.json"], "topsecret");
  }
}

// 定义名为 SerializationTest 的测试套件，包含 TypeTags 单元测试
TEST(SerializationTest, TypeTags) {
  // 创建一个 c10::List 对象 list，包含两个子列表
  auto list = c10::List<c10::List<int64_t>>();
  list.push_back(c10::List<int64_t>({1, 2, 3}));
  list.push_back(c10::List<int64_t>({4, 5, 6}));
  // 创建一个 c10::Dict 对象 dict，包含一个名为 "Hello" 的张量
  auto dict = c10::Dict<std::string, at::Tensor>();
  dict.insert("Hello", torch::ones({2, 2}));
  // 创建一个 c10::List 对象 dict_list，包含五个 c10::Dict 对象
  auto dict_list = c10::List<c10::Dict<std::string, at::Tensor>>();
  for (size_t i = 0; i < 5; i++) {
    auto another_dict = c10::Dict<std::string, at::Tensor>();
    another_dict.insert("Hello" + std::to_string(i), torch::ones({2, 2}));
    dict_list.push_back(another_dict);
  }
  // 创建一个 std::tuple 对象 tuple，包含一个整数和一个字符串
  auto tuple = std::tuple<int, std::string>(2, "hi");
  // 定义一个名为 TestItem 的结构体，包含值和预期类型
  struct TestItem {
    IValue value;
    TypePtr expected_type;
  };
  // 创建一个 TestItem 对象的向量 items，包含多个值和它们的预期类型
  std::vector<TestItem> items = {
      {list, ListType::create(ListType::create(IntType::get()))},
      {2, IntType::get()},
      {dict, DictType::create(StringType::get(), TensorType::get())},
      {dict_list,
       ListType::create(
           DictType::create(StringType::get(), TensorType::get()))},
      {tuple, TupleType::create({IntType::get(), StringType::get()})}};
  // 遍历 items 中的每个 TestItem 对象
  // NOLINTNEXTLINE(performance-for-range-copy)
  for (auto item : items) {
    // 使用 torch::pickle_save 将 item 的值序列化为字节流 bytes
    auto bytes = torch::pickle_save(item.value);
    // 使用 torch::pickle_load 将字节流 bytes 反序列化为 loaded
    auto loaded = torch::pickle_load(bytes);
    // 断言 loaded 的类型是 item 的预期类型的子类型
    ASSERT_TRUE(loaded.type()->isSubtypeOf(*item.expected_type));
    // 断言 item 的预期类型是 loaded 的类型的子类型
    ASSERT_TRUE(item.expected_type->isSubtypeOf(*loaded.type()));
  }
}
TEST(SerializationTest, TestJitStream_CUDA) {
  // 创建一个空的 Torch 模型
  torch::jit::Module model;
  // 创建一个空的输入向量
  std::vector<torch::jit::IValue> inputs;
  // 从文件中反序列化 ScriptModule，使用 torch::jit::load() 函数
  // 加载预先生成的测试脚本模型，参考 tests_setup.py 中的 TorchSaveJitStream_CUDA
  model = torch::jit::load("saved_stream_model.pt");

  // 运行模型推理过程
  auto output = model.forward(inputs);
  // 获取输出元组中的元素列表
  const auto& list_of_elements = output.toTupleRef().elements();
  // 获取第一个元素，表示流是否设置为真
  auto is_stream_s = list_of_elements[0].toBool();

  // 获取输入张量 a, b 和输出张量 c
  // a, b 是两个输入张量
  auto a = list_of_elements[1].toTensor();
  auto b = list_of_elements[2].toTensor();
  // c 是操作 torch.cat(a, b) 生成的输出张量
  auto c = list_of_elements[3].toTensor();

  // 使用 at::cat() 函数对 a 和 b 进行拼接，得到 op 张量
  auto op = at::cat({a, b}, 0);

  // 断言流是否设置为真
  ASSERT_TRUE(is_stream_s);
  // 断言 op 和 c 的大小在 GPU 和 CPU 上是否相同
  ASSERT_EQ(op.sizes(), c.sizes());
  // 断言 op 和 c 两个张量是否完全相等
  ASSERT_TRUE(op.equal(c));
}

TEST(TestSourceRoundTrip, UpsampleNearest2d) {
  // 创建名为 m 的 Torch 模块
  Module m("m");
  // 定义模块的 forward 方法，使用 torch.upsample_nearest2d 函数
  m.define(R"(
    def forward(self, input: Tensor, scale:float):
      return torch.upsample_nearest2d(input, [1, 1], float(scale), float(scale))
  )");

  // 创建输入向量
  std::vector<IValue> inputs;
  // 向输入向量添加随机生成的张量和标量 2.0
  inputs.emplace_back(torch::rand({1, 3, 128, 128}));
  inputs.emplace_back(at::Scalar(2.0));
  // 执行模块的 forward 方法，获得 ref 结果
  auto ref = m.forward(inputs);

  // 通过 roundtripThroughMobile 函数将模块 m 序列化再反序列化为 m2
  Module m2 = roundtripThroughMobile(m);
  // 使用 m2 执行 forward 方法，获得 res 结果
  auto res = m2.forward(inputs);

  // 将 res 转换为张量 resd 和 ref 转换为张量 refd
  auto resd = res.toTensor();
  auto refd = ref.toTensor();
  // 断言 resd 和 refd 两个张量是否完全相等
  ASSERT_TRUE(resd.equal(refd));
}

TEST(TestSourceRoundTrip, CheckAttrAccess) {
  // 创建名为 m 的 Torch 模块
  Module m("m");
  // 注册名为 mobile_optimized 的布尔类型属性
  m.register_attribute("mobile_optimized", BoolType::get(), true);
  // 通过 roundtripThroughMobile 函数将模块 m 序列化再反序列化为 m2
  Module m2 = roundtripThroughMobile(m);
  // 获取 m2 中名为 mobile_optimized 的属性，并转为布尔类型
  bool mobile_optimized = m2.attr("mobile_optimized", false).toBool();
  // 断言 mobile_optimized 属性是否为真
  AT_ASSERT(mobile_optimized);
}

TEST(TestSourceRoundTrip,
     MethodInvocation) { // NOLINT (use =delete in gtest)
  // 定义测试程序列表，每个程序用于测试不同的方法调用情况
  const std::vector<std::string> test_programs{
      // 测试调用具有默认参数的方法
      R"(
      def test_func(self, x, b : int = 4):
        return self.foo + x + b
      )",
      // 内部方法调用带有默认参数（被内联）
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
    // 创建名为 m 的 Torch 模块
    Module m("m");
    // 注册名为 foo 的张量参数，初始化为全 1
    m.register_parameter("foo", torch::ones({}), false);
    // 定义测试程序
    m.define(test_program);

    const int fortyTwo = 42; // (keep linter happy)
    // 生成一个输入张量 minput，维度为 1，元素值为 42
    auto minput = fortyTwo * torch::ones({});
    // 调用对象 m 的 "test_func" 方法，并传入参数 minput，返回结果存储在 ref 中
    auto ref = m.run_method("test_func", minput);

    // 将对象 m 通过 roundtripThroughMobile 转换为 Module 类型的对象 m2
    Module m2 = roundtripThroughMobile(m);

    // 从对象 m2 中获取名为 "test_func" 的方法，并存储在 test_func 中
    const auto& test_func = m2.get_method("test_func");

    // 定义变量 res 用于存储方法调用的结果
    IValue res;

    // 循环调用 test_func 方法三次，每次传入参数 minput，并将结果存储在 res 中
    for (int i = 0; i < 3; ++i) {
      res = test_func({minput});
    }

    // 将 res 转换为 Tensor 类型，提取其单个浮点数值，并存储在 resd 中
    auto resd = res.toTensor().item<float>();

    // 将 ref 转换为 Tensor 类型，提取其单个浮点数值，并存储在 refd 中
    auto refd = ref.toTensor().item<float>();

    // 断言 resd 与 refd 的值相等，否则抛出异常
    AT_ASSERT(resd == refd);
TEST(TestSaveLoad, LoadWithoutDebugInfo) { // NOLINT (use =delete in gtest)
  // 创建一个名为 "m" 的模块对象
  Module m("m");
  // 向模块注册一个名为 "foo" 的参数，其值为 torch::ones({})
  m.register_parameter("foo", torch::ones({}), false);
  // 定义模块的第一个函数 test_func，接受参数 self 和 x
  m.define(
      R"(
    def test_func(self, x):
      b = 4
      return self.foo + x + b
    )");
  // 定义模块的第二个函数 exception，没有参数
  m.define(
      R"(
    def exception(self):
      assert False, "message"
    )");
  // 创建一个字符串流对象 ss，用于保存模块 m 的状态
  std::stringstream ss;
  // 将模块 m 的状态保存到字符串流 ss 中
  m.save(ss);
  // 将流 ss 的读取位置移动到起始位置
  ss.seekg(0);
  // 创建一个 PyTorchStreamReader 对象 reader，用于读取流 ss 中的数据
  caffe2::serialize::PyTorchStreamReader reader(&ss);
  // 设置 reader 应该加载调试符号信息
  reader.setShouldLoadDebugSymbol(true);
  // 断言 reader 是否包含名为 "code/__torch__.py.debug_pkl" 的记录
  EXPECT_TRUE(reader.hasRecord("code/__torch__.py.debug_pkl"));
  // 设置 reader 不加载调试符号信息
  reader.setShouldLoadDebugSymbol(false);
  // 断言 reader 是否不包含名为 "code/__torch__.py.debug_pkl" 的记录
  EXPECT_FALSE(reader.hasRecord("code/__torch__.py.debug_pkl"));
  // 将流 ss 的读取位置移动到起始位置
  ss.seekg(0);
  // 从流 ss 中加载模块数据到新的模块对象 m2
  Module m2 = torch::jit::load(ss);
  // 定义一个字符串 error_msg，用于描述预期的错误消息格式
  std::string error_msg = R"(
    def exception(self):
      assert False, "message"
      ~~~~~~~~~~~~~~~~~~~~~~~ <--- HERE)";
  // 断言调用 m2 的 run_method("exception") 方法会抛出带有指定错误消息的异常
  ASSERT_THROWS_WITH_MESSAGE(m2.run_method("exception"), error_msg);

  // 将流 ss 的读取位置移动到起始位置
  ss.seekg(0);
  // 如果没有调试跟踪信息，则错误消息指向 torchscript 生成的源代码而不是原始的 Python 源代码
  std::string error2 = R"(
    # 定义一个异常处理方法，属于 __torch__.m 类的成员方法，无返回值
    def exception(self: __torch__.m) -> NoneType:
      # 创建一个未初始化的 NoneType 对象 _0
      _0 = uninitialized(NoneType)
      # 抛出一个异常，错误信息为 "AssertionError: message"
      ops.prim.RaiseException("AssertionError: message")
      # 返回未初始化的对象 _0
      return _0
  )";
  # 从字符串流 ss 中加载 Torch 脚本模块 m3，不使用任何额外参数，不是迭代模式
  Module m3 = torch::jit::load(ss, c10::nullopt, false);
  # 断言调用 m3 对象的 run_method("exception") 方法会抛出 error2 异常
  ASSERT_THROWS_WITH_MESSAGE(m3.run_method("exception"), error2);
}

TEST(SerializationTest, TestPickleAppend) {
  // 创建一个包含特定字节序列的 vector 对象，用于模拟序列化数据
  auto data = std::vector<char>({'\x80', char(2), ']', 'K', char(2), 'a', '.'});

  // 使用 Torch 的 unpickle 函数解析数据，返回一个 torch::IValue 对象
  torch::IValue actual = torch::jit::unpickle(data.data(), data.size());

  // 创建预期的 torch::IValue 对象，这里是一个包含整数 2 的 GenericList
  torch::IValue expected = c10::impl::GenericList(at::AnyType::get());
  expected.toList().push_back(2);

  // 断言实际解析得到的对象与预期对象相等
  ASSERT_EQ(expected, actual);
}

} // namespace jit
} // namespace torch
```