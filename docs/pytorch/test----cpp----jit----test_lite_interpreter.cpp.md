# `.\pytorch\test\cpp\jit\test_lite_interpreter.cpp`

```
#include <test/cpp/jit/test_utils.h>

#include <c10/core/TensorOptions.h>
#include <gtest/gtest.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/frontend/resolver.h>
#include <torch/csrc/jit/mobile/compatibility/backport.h>
#include <torch/csrc/jit/mobile/compatibility/backport_manager.h>
#include <torch/csrc/jit/mobile/compatibility/model_compatibility.h>
#include <torch/csrc/jit/mobile/compatibility/runtime_compatibility.h>
#include <torch/csrc/jit/mobile/import.h>
#include <torch/csrc/jit/mobile/interpreter.h>
#include <torch/csrc/jit/mobile/module.h>
#include <torch/csrc/jit/mobile/parse_bytecode.h>
#include <torch/csrc/jit/mobile/parse_operators.h>
#include <torch/csrc/jit/mobile/upgrader_mobile.h>
#include <torch/csrc/jit/serialization/export.h>
#include <torch/csrc/jit/serialization/import.h>
#include <torch/custom_class.h>
#include <torch/torch.h>

#include <torch/csrc/jit/serialization/import_export_functions.h>
#include <unordered_set>

// Tests go in torch::jit
namespace torch {
namespace jit {

// 测试用例：LiteInterpreterTest::UpsampleNearest2d
TEST(LiteInterpreterTest, UpsampleNearest2d) {
  // 创建名为 "m" 的模块对象
  Module m("m");
  // 定义模块的前向传播方法，使用 torch.upsample_nearest2d 进行最近邻上采样
  m.define(R"(
    def forward(self, input: Tensor, scale:float):
      return torch.upsample_nearest2d(input, [1, 1], float(scale), float(scale))
  )");

  // 准备输入参数列表
  std::vector<IValue> inputs;
  inputs.emplace_back(torch::rand({1, 3, 128, 128}));
  inputs.emplace_back(at::Scalar(2.0));
  // 执行模块的前向传播，并保存结果作为参考结果
  auto ref = m.forward(inputs);

  // 将模块保存为移动端模型到字符串流
  std::stringstream ss;
  m._save_for_mobile(ss);
  // 从字符串流中加载移动端模型
  mobile::Module bc = _load_for_mobile(ss);
  // 执行加载后的移动端模型的前向传播
  IValue res;
  res = bc.forward(inputs);

  // 将结果转换为张量
  auto resd = res.toTensor();
  auto refd = ref.toTensor();
  // 断言结果张量相等
  ASSERT_TRUE(resd.equal(refd));
}

// 测试用例：LiteInterpreterTest::CheckAttrAccess
TEST(LiteInterpreterTest, CheckAttrAccess) {
  // 创建名为 "m" 的模块对象
  Module m("m");
  // 注册名为 "mobile_optimized" 的布尔类型属性，并设置默认值为 true
  m.register_attribute("mobile_optimized", BoolType::get(), true);

  // 将模块保存为移动端模型到字符串流
  std::stringstream ss;
  m._save_for_mobile(ss);
  // 从字符串流中加载移动端模型
  mobile::Module bc = _load_for_mobile(ss);
  // 从加载后的模型中获取名为 "mobile_optimized" 的属性，并转换为布尔值
  bool mobile_optimized = bc.attr("mobile_optimized", false).toBool();

  // 断言属性值为 true
  AT_ASSERT(mobile_optimized);
  // 设置名为 "mobile_optimized" 的属性为 false
  m.setattr("mobile_optimized", false);
  // 清空字符串流
  ss = std::stringstream();
  // 将修改后的模型再次保存为移动端模型到字符串流
  m._save_for_mobile(ss);
  // 重新加载模型
  bc = _load_for_mobile(ss);
  // 再次获取 "mobile_optimized" 属性，并转换为布尔值
  mobile_optimized = bc.attr("mobile_optimized", false).toBool();

  // 断言属性值为 false
  AT_ASSERT(!mobile_optimized);
}

// 测试用例：LiteInterpreterTest::MethodInvocation
TEST(LiteInterpreterTest, MethodInvocation) { // NOLINT (use =delete in gtest)
  // 测试程序列表
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
    // 创建名为 m 的模块对象
    Module m("m");
    // 向模块 m 注册一个名为 "foo" 的参数，初始值为 torch 中的 1，不要求需要梯度
    m.register_parameter("foo", torch::ones({}), false);
    // 定义一个名为 test_program 的程序，并绑定到模块 m 上
    m.define(test_program);

    // 定义一个整型常量 fortyTwo 并赋值为 42，用于代码风格检查的目的
    const int fortyTwo = 42; // (keep linter happy)
    // 创建一个自动类型推断的变量 minput，其值为 fortyTwo 乘以一个维度为空的全 1 张量
    auto minput = fortyTwo * torch::ones({});
    // 调用模块 m 的 test_func 方法，并传入 minput 作为参数，将结果赋给 ref 变量
    auto ref = m.run_method("test_func", minput);

    // 创建一个字符串流对象 ss
    std::stringstream ss;
    // 将模块 m 保存为移动端格式到字符串流 ss 中
    m._save_for_mobile(ss);
    // 从字符串流 ss 中加载移动端模块，赋给 bc 变量
    mobile::Module bc = _load_for_mobile(ss);
    // 从 bc 中获取名为 "test_func" 的方法，并赋给 test_func 变量
    const auto& test_func = bc.get_method("test_func");
    // 声明一个 IValue 类型的变量 res
    IValue res;
    // 循环执行 3 次，调用 test_func 方法，传入 minput 作为参数，并将结果赋给 res 变量
    for (int i = 0; i < 3; ++i) {
      res = test_func({minput});
    }

    // 将 res 转换为 float 类型的张量，并提取其单个值，赋给 resd 变量
    auto resd = res.toTensor().item<float>();
    // 将 ref 转换为 float 类型的张量，并提取其单个值，赋给 refd 变量
    auto refd = ref.toTensor().item<float>();
    // 断言 resd 和 refd 的值相等
    AT_ASSERT(resd == refd);
  }
TEST(LiteInterpreterTest, Conv) {
  auto s = std::getenv("PYTORCH_TEST_WITH_TSAN");
  // 检查环境变量PYTORCH_TEST_WITH_TSAN是否存在且其值为"1"，如果是则直接返回，不进行测试
  if (s && strcmp(s, "1") == 0)
    return;

  std::vector<torch::jit::IValue> inputs;

  Module m("m");
  // 注册模型参数weight和bias，都为torch::Tensor类型，初始化为全1
  m.register_parameter("weight", torch::ones({20, 1, 5, 5}), false);
  m.register_parameter("bias", torch::ones({20}), false);
  // 定义模型的forward方法，进行卷积运算
  m.define(R"(
    def forward(self, input):
      return torch._convolution(input, self.weight, self.bias, [1, 1], [0, 0], [1, 1], False, [0, 0], 1, False, False, True, True)
  )");

  // 将输入的torch::Tensor对象加入inputs向量中
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers,modernize-use-emplace)
  inputs.push_back(torch::ones({1, 1, 28, 28}));

  // 执行模型的forward方法，并获取输出的torch::Tensor对象
  auto outputref = m.forward(inputs).toTensor();

  std::stringstream ss;
  // 将模型保存为移动端可用的形式并写入stringstream ss中
  m._save_for_mobile(ss);
  // 从stringstream ss中加载为移动端模型
  mobile::Module bc = _load_for_mobile(ss);
  IValue res;
  // 对加载的移动端模型进行三次forward调用
  for (int i = 0; i < 3; ++i) {
    res = bc.get_method("forward")(inputs);
  }
  // 获取forward方法的输出结果，并转换为torch::Tensor对象
  auto output = res.toTensor();
  // 断言两个torch::Tensor对象的维度相同
  AT_ASSERT(outputref.dim() == output.dim());
  // 断言两个torch::Tensor对象第一个元素的数值相同
  AT_ASSERT(
      outputref[0][0][0][0].item<int>() == output[0][0][0][0].item<int>());
}

TEST(LiteInterpreterTest, Inline) {
  Module m("m");
  // 定义包含多个嵌套函数的模型，名称为m
  m.define(R"JIT(
  def foo1(self, x):
      return x + 1

  def foo2(self, x):
      return self.foo1(x) + 2

  def foo3(self, x):
      return self.foo2(x) + 3
  )JIT");
  std::stringstream ss;
  // 将模型保存为移动端可用的形式并写入stringstream ss中
  m._save_for_mobile(ss);
  // 从stringstream ss中加载为移动端模型
  mobile::Module bc = _load_for_mobile(ss);
  // 构造输入向量，包含一个torch::Tensor对象
  std::vector<torch::jit::IValue> inputs({torch::ones({})});
  // 调用移动端模型的方法foo3，并获取输出结果
  auto output = bc.get_method("foo3")(inputs);
  // 断言移动端模型的输出结果为7.0
  AT_ASSERT(output.toTensor().item<float>() == 7.0);
}

TEST(LiteInterpreterTest, Tuple) {
  Module m("m");
  // 定义包含元组返回的模型函数
  m.define(R"JIT(
  def foo(self, x):
      return (1, 2, x + 3)

  def forward(self, x):
      tuple = self.foo(x)
      return tuple
  )JIT");
  std::stringstream ss;
  // 将模型保存为移动端可用的形式并写入stringstream ss中
  m._save_for_mobile(ss);
  // 从stringstream ss中加载为移动端模型
  mobile::Module bc = _load_for_mobile(ss);
  // 构造输入向量，包含一个torch::Tensor对象
  std::vector<torch::jit::IValue> inputs({torch::ones({})});
  // 调用移动端模型的forward方法，并获取输出结果
  auto output = bc.get_method("forward")(inputs);
  // 断言移动端模型输出元组的第二个元素为2
  AT_ASSERT(output.toTupleRef().elements()[1].toInt() == 2);
}

TEST(LiteInterpreterTest, AtenFormat) {
  Module m("m");
  // 定义包含字符串格式化操作的模型方法
  m.define(R"""(
  def forward(self, fmt:str="first {} {}", num:str="abc"):
    x = 2
    x = x * x
    return fmt.format(num, x)
  )""");
  std::stringstream ss;
  // 将模型保存为移动端可用的形式并写入stringstream ss中
  m._save_for_mobile(ss);
  // 从stringstream ss中加载为移动端模型
  mobile::Module bc = _load_for_mobile(ss);
  std::vector<torch::jit::IValue> inputs;
  // 调用移动端模型的forward方法，并获取输出结果
  auto output_bc = bc.get_method("forward")(inputs);
  auto output_m = m.get_method("forward")(inputs);
  // 断言移动端模型和原始模型的输出字符串相同
  // std::cout << output_m.toStringRef() << "\n"
  //           << output_bc.toStringRef() << std::endl;
  AT_ASSERT(output_m.toStringRef() == output_bc.toStringRef());
}

TEST(LiteInterpreterTest, PrimDevice) {
  Module m("m");
  // 定义包含torch.Tensor作为输入参数的模型方法
  m.define(R"""(
  def forward(self, x:torch.Tensor):
    x = 2
    x = x * x
    return x
  )""");
    return x.device
  )""");


    // 返回张量 x 的设备信息
    // 此处是一个字符串插值终结符，用于结束一个多行字符串常量的定义



  std::stringstream ss;


  // 创建一个字符串流对象 ss，用于存储字符串数据



  m._save_for_mobile(ss);


  // 将模型 m 保存为移动端可用的格式，并写入字符串流 ss 中



  mobile::Module bc = _load_for_mobile(ss);


  // 从字符串流 ss 中加载移动端模型，返回一个移动端模块 bc



  std::vector<torch::jit::IValue> inputs;


  // 创建一个存储 TorchScript 的 IValue 类型的向量 inputs，用于存放模型输入



  auto minput = 3.5 * torch::ones({});
  inputs.emplace_back(minput);


  // 创建一个张量 minput，其值为 3.5，形状为空
  // 将 minput 添加到 inputs 向量中，作为模型的输入之一



  auto output_bc = bc.get_method("forward")(inputs);


  // 调用移动端模块 bc 的 forward 方法，传入 inputs 作为参数，得到输出 output_bc



  auto output_m = m.get_method("forward")(inputs);


  // 调用原始模型 m 的 forward 方法，传入 inputs 作为参数，得到输出 output_m



  AT_ASSERT(output_bc.toDevice().str() == output_m.toDevice().str());


  // 使用 AT_ASSERT 断言，比较 output_bc 和 output_m 的设备信息字符串表示是否相等
TEST(LiteInterpreterTest, Dict) {
  // 创建名为 "m" 的模块对象
  Module m("m");
  // 定义模块的前向计算逻辑，使用 JIT 脚本字符串定义
  m.define(R"JIT(
  def foo(self, x):
      return {"result": x + 1}

  def forward(self, x):
      d = self.foo(x)
      return d
  )JIT");
  // 创建一个字符串流对象
  std::stringstream ss;
  // 将模块保存为移动端可用的格式，并将其序列化到字符串流中
  m._save_for_mobile(ss);
  // 从字符串流中加载移动端模块
  mobile::Module bc = _load_for_mobile(ss);
  // 创建输入向量，包含一个形状为 () 的张量
  std::vector<torch::jit::IValue> inputs({torch::ones({})});
  // 调用加载后模块的 forward 方法，传入输入向量，并获取输出
  auto output = bc.get_method("forward")(inputs);
  // 使用断言验证输出的 'result' 键对应的张量值为 2
  AT_ASSERT(output.toGenericDict().at("result").toTensor().item().toInt() == 2);
}

TEST(LiteInterpreterTest, List) {
  // 创建名为 "m" 的模块对象
  Module m("m");
  // 定义模块的前向计算逻辑，使用 JIT 脚本字符串定义
  m.define(R"JIT(
  def foo(self, x):
      return [x + 2]

  def forward(self, x):
      d = self.foo(x)
      return d
  )JIT");
  // 创建一个字符串流对象
  std::stringstream ss;
  // 将模块保存为移动端可用的格式，并将其序列化到字符串流中
  m._save_for_mobile(ss);
  // 从字符串流中加载移动端模块
  mobile::Module bc = _load_for_mobile(ss);
  // 创建输入向量，包含一个形状为 () 的张量
  std::vector<torch::jit::IValue> inputs({torch::ones({})});
  // 调用加载后模块的 forward 方法，传入输入向量，并获取输出
  auto output = bc.get_method("forward")(inputs);
  // 调用原始模块的 forward 方法，传入输入向量，并获取输出
  auto server_output = m.forward(inputs);
  // 使用断言验证输出的第一个元素对应的张量值为 3
  EXPECT_EQ(output.toList().get(0).toTensor().item().toInt(), 3);
  // 使用断言验证加载后模块的输出与原始模块的输出相等
  EXPECT_EQ(output, server_output);
}

TEST(LiteInterpreterTest, PrimOverload) {
  /*
  // 暂时禁用
  script::Module m("m");
  m.define(R"JIT(
  def forward(self, x):
      result = [1, 2]
      result.append(3)
      return result
  )JIT");
  std::stringstream ss;
  m._save_for_mobile(ss);
  mobile::Module bc = _load_for_mobile(ss);
  std::vector<torch::jit::IValue> inputs({torch::ones({})});
  auto output = bc.get_method("forward")(inputs);
  AT_ASSERT(output.toIntList()[2] == 3);
  */
}

TEST(LiteInterpreterTest, Prim) {
  // 创建名为 "m" 的模块对象
  Module m("m");
  // 定义模块的前向计算逻辑，使用 JIT 脚本字符串定义
  m.define(R"JIT(
        def forward(self, x):
            return int(x)
  )JIT");

  // 创建输入向量
  std::vector<IValue> inputs;
  auto minput = 3.5 * torch::ones({});
  inputs.emplace_back(minput);
  // 运行模块的 forward 方法，传入输入向量，获取输出
  auto ref = m.run_method("forward", minput);

  // 创建一个字符串流对象
  std::stringstream ss;
  // 将模块保存为移动端可用的格式，并将其序列化到字符串流中
  m._save_for_mobile(ss);
  // 从字符串流中加载移动端模块
  mobile::Module bc = _load_for_mobile(ss);
  IValue res;
  // 循环执行加载后模块的 forward 方法，传入输入向量，获取输出
  for (int i = 0; i < 3; ++i) {
    // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
    auto bcinputs = inputs;
    res = bc.get_method("forward")(bcinputs);
  }

  // 将结果转换为整数并进行断言验证
  auto resi = res.toInt();
  auto refi = ref.toInt();
  AT_ASSERT(resi == refi);
}

TEST(LiteInterpreterTest, PrimScalar) {
  // 创建名为 "m" 的模块对象
  Module m("m");
  // 定义模块的前向计算逻辑，使用 JIT 脚本字符串定义
  m.define(R"JIT(
        def forward(self, x):
            return int(x.item())
  )JIT");

  // 创建输入向量
  std::vector<IValue> inputs;
  auto minput = 3.5 * torch::ones({});
  inputs.emplace_back(minput);
  // 运行模块的 forward 方法，传入输入向量，获取输出
  auto ref = m.run_method("forward", minput);

  // 创建一个字符串流对象
  std::stringstream ss;
  // 将模块保存为移动端可用的格式，并将其序列化到字符串流中
  m._save_for_mobile(ss);
  // 从字符串流中加载移动端模块
  mobile::Module bc = _load_for_mobile(ss);
  IValue res;
  // 循环执行加载后模块的 forward 方法，传入输入向量，获取输出
  for (int i = 0; i < 3; ++i) {
    // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
    auto bcinputs = inputs;
    res = bc.get_method("forward")(bcinputs);
  }

  // 将结果转换为整数并进行断言验证
  auto resi = res.toInt();
  auto refi = ref.toInt();
  AT_ASSERT(resi == refi);
}

TEST(LiteInterpreterTest, LoadOrigJit) {
  // 创建名为 "m" 的模块对象
  Module m("m");
  // 注册名为 "foo" 的参数，其值为形状为 () 的张量
  m.register_parameter("foo", torch::ones({}), false);
  // 定义模块的前向计算逻辑，使用 JIT 脚本字符串定义
  m.define(R"(
        def forward(self, x):
            result = [1, 2]
            result.append(3)
            return result
  )");

  std::stringstream ss;
  // 将模块保存为移动端可用的格式，并将其序列化到字符串流中
  m._save_for_mobile(ss);
  // 从字符串流中加载移动端模块
  mobile::Module bc = _load_for_mobile(ss);
  // 创建输入向量，包含一个形状为 () 的张量
  std::vector<torch::jit::IValue> inputs({torch::ones({})});
  // 调用加载后模块的 forward 方法，传入输入向量，并获取输出
  auto output = bc.get_method("forward")(inputs);
  // 使用断言验证输出的第三个元素为 3
  AT_ASSERT(output.toIntList()[2] == 3);
}
    # 定义一个名为 forward 的方法，接收参数 x
    def forward(self, x):
      # 设置变量 b 并赋值为 4
      b = 4
      # 返回 self 对象的 foo 属性值加上参数 x 和变量 b 的和
      return self.foo + x + b
  )");
  # 创建一个 std::stringstream 对象 ss
  std::stringstream ss;
  # 将模型 m 序列化保存到 stringstream 对象 ss 中
  m.save(ss);
  # 断言调用 _load_for_mobile(ss) 函数时抛出特定异常并带有特定错误信息 "file not found"
  ASSERT_THROWS_WITH_MESSAGE(_load_for_mobile(ss), "file not found");
}

// 定义一个单元测试用例，测试错误的方法名
TEST(LiteInterpreterTest, WrongMethodName) {
  // 创建一个名为 "m" 的模块对象
  Module m("m");
  // 向模块注册一个名为 "foo" 的参数，初始值为全1张量，不要求梯度
  m.register_parameter("foo", torch::ones({}), false);
  // 定义模块方法 "add"
  m.define(R"(
    def add(self, x):
      b = 4
      return self.foo + x + b
  )");
  // 创建一个字符串流对象用于序列化模块
  std::stringstream ss;
  // 将模块序列化为移动端格式并写入字符串流
  m._save_for_mobile(ss);
  // 从字符串流中加载序列化后的模块
  mobile::Module bc = _load_for_mobile(ss);
  // 创建输入向量列表
  std::vector<IValue> inputs;
  auto minput = 5 * torch::ones({});
  inputs.emplace_back(minput);
  // 断言调用模块方法 "forward" 抛出异常，并包含特定错误信息
  ASSERT_THROWS_WITH_MESSAGE(
      bc.get_method("forward")(inputs), "is not defined");
}

// 定义一个单元测试用例，测试设置模块状态
TEST(LiteInterpreterTest, SetState) {
  // 创建一个名为 "m" 的模块对象
  Module m("m");
  // 向模块注册一个名为 "foo" 的参数，初始值为全1张量，不要求梯度
  m.register_parameter("foo", torch::ones({}), false);
  // 定义模块方法 "__getstate__" 和 "__setstate__" 以及 "forward"
  m.define(R"(
    def __getstate__(self):
      return self.foo + self.foo
    def __setstate__(self, a):
      self.foo = a
    def forward(self, x):
      b = 4
      return self.foo + x + b
  )");

  // 创建输入向量列表
  std::vector<IValue> inputs;
  auto minput = 5 * torch::ones({});
  inputs.emplace_back(minput);

  // 创建字符串流对象用于普通模块的序列化
  std::stringstream ms;
  // 将模块序列化并写入字符串流
  m.save(ms);
  // 从字符串流中加载序列化后的模块
  auto loaded_m = load(ms);
  // 运行加载后的模块的 "forward" 方法
  auto ref = loaded_m.run_method("forward", minput);

  // 创建字符串流对象用于移动端模块的序列化
  std::stringstream ss;
  // 将模块序列化为移动端格式并写入字符串流
  m._save_for_mobile(ss);
  // 从字符串流中加载序列化后的移动端模块
  mobile::Module bc = _load_for_mobile(ss);

  // 定义一个 IValue 类型变量 res
  IValue res;
  // 多次执行移动端模块的 "forward" 方法
  for (int i = 0; i < 3; ++i) {
    // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
    auto bcinputs = inputs;
    res = bc.get_method("forward")(bcinputs);
  }

  // 将 res 转换为 float 类型并获取其值
  auto resd = res.toTensor().item<float>();
  // 将参考值 ref 转换为 float 类型并获取其值
  auto refd = ref.toTensor().item<float>();
  // 断言 resd 与 refd 相等
  AT_ASSERT(resd == refd);
}

// 定义一个结构体，继承自 torch::jit::CustomClassHolder，用于处理自定义类
class TorchBindLiteInterpreterTestStruct
    : public torch::jit::CustomClassHolder {
 public:
  // 返回一个字符串，描述输入张量的信息
  std::string get(at::Tensor t) {
    std::stringstream ss;
    ss << "Hello! Your tensor has ";
    ss << t.numel();
    ss << " elements!";
    return ss.str();
  }
};

// 匿名命名空间，定义一个结构体实现 SugaredValue 接口
namespace {
struct ClassNamespaceValue : public SugaredValue {
  // 构造函数，接受一个限定名称作为参数
  explicit ClassNamespaceValue(c10::QualifiedName name)
      : basename_(std::move(name)) {}

  // 返回指定属性的 SugaredValue 对象
  std::shared_ptr<SugaredValue> attr(
      const SourceRange& loc,
      GraphFunction& m,
      const std::string& name) override {
    // 构建完整的限定名称
    const auto fullName = c10::QualifiedName(basename_, name);

    // 检查是否为自定义类
    if (auto custom_class = getCustomClass(fullName.qualifiedName())) {
      // 返回自定义类的 SugaredValue 对象
      return std::make_shared<ClassValue>(custom_class);
    }

    // 如果不是自定义类，假设为另一个命名空间
    // NOLINTNEXTLINE(performance-move-const-arg)
    return std::make_shared<ClassNamespaceValue>(std::move(fullName));
  }

  // 返回对象的种类描述字符串
  std::string kind() const override {
    return "Class Namespace";
  }

 private:
  // 基础名称
  c10::QualifiedName basename_;
};

// 定义一个结构体，实现 Resolver 接口
struct TestModuleResolver : public Resolver {
  // 解析指定名称对应的 SugaredValue 对象
  std::shared_ptr<SugaredValue> resolveValue(
      const std::string& name,
      GraphFunction& m,
      const SourceRange& loc) override {
    // 如果名称为 "torch"
    if (name == "torch") {
      // 返回内置模块 "aten" 的 SugaredValue 对象
      return std::make_shared<BuiltinModule>("aten");
    } else if (name == "__torch__") {
      // 返回类命名空间 "__torch__" 的 SugaredValue 对象
      return std::make_shared<ClassNamespaceValue>(c10::QualifiedName(name));
    }
    // 如果未匹配到特定名称，返回空指针
      return nullptr;
  }
};
    // 返回空指针，表示没有找到符合条件的结果
    return nullptr;
  }

  // 重写接口方法，根据给定名称和源代码范围解析类型，返回空指针表示未能解析出类型
  TypePtr resolveType(const std::string& name, const SourceRange& loc)
      override {
    // 返回空指针，表示没有找到符合条件的类型
    return nullptr;
  }
TEST(LiteInterpreterTest, BuiltinClass) {
  // 创建一个名为 "m" 的脚本模块对象
  script::Module m("m");

  // 获取自定义类 "__torch__.torch.classes._TorchScriptTesting._LiteInterpreterTest"
  auto cls = getCustomClass(
      "__torch__.torch.classes._TorchScriptTesting._LiteInterpreterTest");
  // 内部断言，确保类对象非空
  TORCH_INTERNAL_ASSERT(cls);

  // 创建一个指向 torch::CustomClassHolder 的智能指针对象
  c10::intrusive_ptr<torch::CustomClassHolder> obj_holder;

  // 在模块 m 中注册名为 "my_obj" 的属性，其类型为 cls，初始值为 obj_holder 的胶囊化值
  m.register_attribute("my_obj", cls, IValue::make_capsule(obj_holder));

  // 在模块 m 中注册名为 "foo" 的参数，值为 torch.ones({})，不是可训练的
  m.register_parameter("foo", torch::ones({}), false);

  // 在模块 m 中定义以下 Torch 脚本字符串代码块
  m.define(
      R"(
    def __getstate__(self):
      return 1
    def __setstate__(self, a):
      self.my_obj = __torch__.torch.classes._TorchScriptTesting._LiteInterpreterTest()

    def forward(self, x) -> str:
      return self.my_obj.get(x)
  )",
      // 使用 TestModuleResolver 共享解析器
      std::make_shared<TestModuleResolver>());

  // 创建一个字符串流对象 ss
  std::stringstream ss;
  // 将模块 m 保存为移动端模块到字符串流 ss 中
  m._save_for_mobile(ss);
  // 从字符串流 ss 中加载移动端模块并赋值给 mobile::Module 对象 bc
  mobile::Module bc = _load_for_mobile(ss);

  // 调用 bc 模块中的 forward 方法，传入参数 torch::zeros({3, 4})，并获取返回值
  auto res =
      bc.get_method("forward")(std::vector<IValue>{torch::zeros({3, 4})});
  // 将返回值转换为字符串引用
  const auto& str = res.toStringRef();
  // 预期的字符串结果
  std::string expected = "Hello! Your tensor has 12 elements!";
  // 断言返回的字符串与预期结果相等
  AT_ASSERT(str == expected);
}

TEST(LiteInterpreterTest, BuiltinFunction) {
  // 创建一个名为 "m" 的脚本模块对象
  script::Module m("m");

  // 创建自定义类对象 TorchBindLiteInterpreterTestStruct
  auto custom_class_obj =
      make_custom_class<TorchBindLiteInterpreterTestStruct>();

  // 在模块 m 中注册名为 "my_obj" 的属性，类型为 custom_class_obj 的类型，初始值为 custom_class_obj
  m.register_attribute("my_obj", custom_class_obj.type(), custom_class_obj);

  // 在模块 m 中定义以下 Torch 脚本字符串代码块
  m.define(R"(
    def forward(self, x) -> str:
      return self.my_obj.get(x)
  )");

  // 创建一个字符串流对象 ss
  std::stringstream ss;
  // 将模块 m 保存为移动端模块到字符串流 ss 中
  m._save_for_mobile(ss);
  // 从字符串流 ss 中加载移动端模块并赋值给 mobile::Module 对象 bc
  mobile::Module bc = _load_for_mobile(ss);

  // 调用 bc 模块中的 forward 方法，传入参数 torch::zeros({3, 4})，并获取返回值
  auto res =
      bc.get_method("forward")(std::vector<IValue>{torch::zeros({3, 4})});
  // 将返回值转换为字符串引用
  auto str = res.toStringRef();
  // 预期的字符串结果
  std::string expected = "Hello! Your tensor has 12 elements!";
  // 断言返回的字符串与预期结果相等
  AT_ASSERT(str == expected);
}

#if !defined FB_XPLAT_BUILD
TEST(LiteInterpreterTest, GetRuntimeByteCodeVersion) {
  // 获取运行时字节码版本
  auto runtime_bytecode_version = _get_runtime_bytecode_version();
  // 断言运行时字节码版本与 caffe2::serialize::kMaxSupportedBytecodeVersion 相等
  AT_ASSERT(
      runtime_bytecode_version ==
      caffe2::serialize::kMaxSupportedBytecodeVersion);
}

TEST(LiteInterpreterTest, GetRuntimeOperatorsVersion) {
  // 获取运行时操作符版本
  auto runtime_operators_version = _get_runtime_operators_min_max_versions();
  // 断言运行时操作符版本的第一个和第二个元素分别与 caffe2::serialize::kMinSupportedFileFormatVersion
  // 和 caffe2::serialize::kMaxSupportedFileFormatVersion 相等
  AT_ASSERT(
      runtime_operators_version.first ==
          caffe2::serialize::kMinSupportedFileFormatVersion &&
      runtime_operators_version.second ==
          caffe2::serialize::kMaxSupportedFileFormatVersion);
}
TEST(LiteInterpreterTest, GetByteCodeVersion) {
  // 获取当前文件的路径
  std::string filePath(__FILE__);
  // 构造测试模型文件路径
  auto test_model_file_v4 =
      filePath.substr(0, filePath.find_last_of("/\\") + 1);
  test_model_file_v4.append("script_module_v4.ptl");

  // 获取模型字节码版本
  auto version_v4 = _get_model_bytecode_version(test_model_file_v4);
  // 断言模型字节码版本为4
  AT_ASSERT(version_v4 == 4);
}

#endif // !defined(FB_XPLAT_BUILD)

TEST(LiteInterpreterTest, GetContainTypes) {
  // 创建一个名为m的模块并定义其前向方法
  Module m("m");
  m.define(R"(
    def forward(self):
      return 3
  )");

  // 创建一个字符串流对象
  std::stringstream ss;
  // 将模块序列化为移动端模型保存到字符串流中
  m._save_for_mobile(ss, {}, true);

  // 分析字符串流中包含的移动端模型的类型
  _get_mobile_model_contained_types(ss);
}

namespace {

void compareModelOutput(
    c10::ArrayRef<IValue> actual_result_list,
    const std::vector<IValue>& expect_result_list) {
  // 断言实际结果列表与期望结果列表的长度相等
  AT_ASSERT(actual_result_list.size() == expect_result_list.size());
  // 断言第一个张量相等
  AT_ASSERT(
      actual_result_list[0].toTensor().equal(expect_result_list[0].toTensor()));
  // 断言第二个张量的维度相等
  AT_ASSERT(
      actual_result_list[1].toTensor().dim() ==
      expect_result_list[1].toTensor().dim());
  // 断言第三个张量相等
  AT_ASSERT(
      actual_result_list[2].toTensor().equal(expect_result_list[2].toTensor()));
  // 断言第四个张量相等
  AT_ASSERT(
      actual_result_list[3].toTensor().equal(expect_result_list[3].toTensor()));
  // 断言第五个字符串相等
  ASSERT_EQ(
      actual_result_list[4].toStringRef(), expect_result_list[4].toStringRef());
  // 断言第六个布尔值相等
  ASSERT_EQ(actual_result_list[5].toBool(), expect_result_list[5].toBool());
  // 断言第七个布尔值相等
  ASSERT_EQ(actual_result_list[6].toBool(), expect_result_list[6].toBool());
  // 断言第八个布尔值相等
  ASSERT_EQ(actual_result_list[7].toBool(), expect_result_list[7].toBool());
  // 断言第九个张量相等
  AT_ASSERT(
      actual_result_list[8].toTensor().equal(expect_result_list[8].toTensor()));
  // 断言第十个字符串相等
  ASSERT_EQ(
      actual_result_list[9].toStringRef(), expect_result_list[9].toStringRef());
  // 断言第十一个整数相等
  ASSERT_EQ(actual_result_list[10].toInt(), expect_result_list[10].toInt());
  // 断言第十二个布尔值相等
  ASSERT_EQ(actual_result_list[11].toBool(), expect_result_list[11].toBool());
}

void runAndCheckTorchScriptModel(
    std::stringstream& input_model_stream,
    const std::vector<IValue>& input_data,
    const std::vector<IValue>& expect_result_list,
    const uint64_t expect_version) {
  // 获取输入模型流的字节码版本
  auto actual_version = _get_model_bytecode_version(input_model_stream);
  // 断言实际的字节码版本与期望的版本相等
  AT_ASSERT(actual_version == expect_version);

  // 加载并运行后移植的模型，然后与期望结果比较
  Module m_mobile = load(input_model_stream);

  // 获取实际的模型输出结果
  auto actual_result = m_mobile.forward(input_data);
  const auto& actual_result_list = actual_result.toTupleRef().elements();
  // 比较模型输出结果
  compareModelOutput(actual_result_list, expect_result_list);
}

void runAndCheckBytecodeModel(
    std::stringstream& input_model_stream,
    const std::vector<IValue>& input_data,
    const std::vector<IValue>& expect_result_list,
    // 检查实际模型字节码版本是否与期望版本相符
    const uint64_t expect_version) {
  // 调用函数获取输入模型流的实际版本号
  auto actual_version = _get_model_bytecode_version(input_model_stream);
  // 使用断言确保实际版本号与期望版本号相同
  AT_ASSERT(actual_version == expect_version);

  // 加载并运行后移模型，然后将结果与期望结果进行比较
  // 从输入模型流中加载模型并存储在 Module 对象 m_mobile 中
  Module m_mobile = load(input_model_stream);

  // 对输入数据进行前向传播，获取实际的模型输出结果
  auto actual_result = m_mobile.forward(input_data);
  // 将实际结果转换为元组引用，并获取其元素列表
  const auto& actual_result_list = actual_result.toTupleRef().elements();

  // 比较实际模型输出结果与期望的结果列表
  compareModelOutput(actual_result_list, expect_result_list);
// backportAllVersionCheck 函数定义，用于检查和执行模型的版本回溯操作
void backportAllVersionCheck(
    // test_model_file_stream 是一个字符串流，用于读取测试模型文件的内容
    std::stringstream& test_model_file_stream,
    // input_data 是一个存储输入数据的 IValue 向量
    std::vector<IValue>& input_data,
    // expect_result_list 是一个存储期望输出结果的 IValue 向量
    std::vector<IValue>& expect_result_list,
    // expect_from_version 是预期的模型字节码版本号
    const uint64_t expect_from_version) {
  // 获取模型的实际字节码版本号
  auto from_version = _get_model_bytecode_version(test_model_file_stream);
  // 断言实际字节码版本号与预期版本号相等
  EXPECT_EQ(from_version, expect_from_version);
  // 断言实际字节码版本号大于 0
  AT_ASSERT(from_version > 0);

  // 将 script_module_v5.ptl 回溯至较旧版本
  constexpr int64_t minimum_to_version = 4;
  auto current_to_version = from_version - 1;

  // 验证所有候选的 to_version 是否按预期工作。所有大于 minimum_to_version 的回溯版本应该成功。
  while (current_to_version >= minimum_to_version) {
    // 在 while 循环中声明 std::stringstream oss，因为 oss.clear() 只会清除流的错误状态标志，而不会重置流内容，
    // 这可能会导致问题。声明新的 stringstream 更为清晰和安全。
    std::stringstream oss;
    // 执行模型的版本回溯操作，并将结果存储在 oss 中
    bool backPortSuccess =
        _backport_for_mobile(test_model_file_stream, oss, current_to_version);
    // 断言回溯操作成功
    AT_ASSERT(backPortSuccess);

    // 检查回溯后的模型版本号
    auto backport_version = _get_model_bytecode_version(oss);
    // 再次确认回溯后的模型版本号与当前回溯的版本号相等
    AT_ASSERT(backport_version == current_to_version);

    // 加载并运行回溯后的模型，并与期望结果进行比较
    runAndCheckBytecodeModel(
        oss, input_data, expect_result_list, current_to_version);
    // 重置 oss 流的读指针到开头位置
    oss.seekg(0, oss.beg);
    // 运行并检查 TorchScript 模型的结果
    runAndCheckTorchScriptModel(
        oss, input_data, expect_result_list, current_to_version);

    // 将当前回溯版本号减一，以便进行下一次回溯
    current_to_version--;
  }
  // 尝试回溯至最低版本的前一个版本（minimum_to_version - 1），应该会失败
  std::stringstream oss;
  bool backPortSuccess =
      _backport_for_mobile(test_model_file_stream, oss, minimum_to_version - 1);
  // 断言回溯操作失败
  AT_ASSERT(!backPortSuccess);
}
// 命名空间结束
} // namespace

// 如果不是 FB_XPLAT_BUILD 构建，执行 BackPortByteCodeModelAllVersions 测试
#if !defined FB_XPLAT_BUILD
TEST(LiteInterpreterTest, BackPortByteCodeModelAllVersions) {
  // 创建一个名为 "m" 的 TorchScript 模块
  torch::jit::Module module("m");
  // 注册模块的参数 "weight"，初始化为全 1 的张量
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  module.register_parameter("weight", torch::ones({20, 1, 5, 5}), false);
  // 注册模块的参数 "bias"，初始化为全 1 的张量
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  module.register_parameter("bias", torch::ones({20}), false);
  // 定义模块的 TorchScript 函数 fn，接受一个默认值为 1.0 的浮点数参数 x，返回 x
  module.define(R"(
    def fn(self, x:float=1.0):
      return x
    # 定义一个类方法 `forward`，用于模型的前向传播
    def forward(self, input):
      # 创建一个 2x2 的全零张量 x1
      x1 = torch.zeros(2, 2)
      # 使用与给定张量相同形状的空张量来初始化 x2
      x2 = torch.empty_like(torch.empty(2, 2))
      # 调用底层的卷积函数 `_convolution` 执行卷积操作，生成 x3
      x3 = torch._convolution(input, self.weight, self.bias, [1, 1], [0, 0], [1, 1], False, [0, 0], 1, False, False, True, True)
      # 使用 torch.ones 创建一个值为 2 的张量 x
      x = 2 * torch.ones(1)
      # 使用 torch.ones 创建一个值为 1 的张量 h，然后将其加到 x 上，并将结果存储在 x 中
      h = torch.ones(1)
      torch.add(x, h, out=x)
      # 创建一个包含值为 1 的张量，然后将其转移到 CPU 设备，并获取其设备类型
      device = torch.ones(1, 1).cpu().device.type
      # 检查 x1 张量是否在 GPU 上
      is_cuda = x1.is_cuda
      # 布尔变量 bool_val 被设置为 True
      bool_val = True
      # 检查空列表是否为 None
      check_is = [] is None
      # 检查非空列表是否不为 None
      check_is_not = [1] is not None
      # 对 bool_val 取反
      check_not = not bool_val
      # 调用 self.fn() 获取一个数值，并将其转换为张量
      num_to_tensor = torch.tensor([self.fn()])
      # 创建一个包含键值对 "a": "abc" 的字典 d，然后获取 "a" 对应的值
      d = {"a": "abc"}
      check_dict_index = d["a"]
      # 获取 x1 的维度数
      check_dim = x1.dim()
      # 返回多个变量作为元组的形式
      return (
        x1, x2, x3, x, device, is_cuda, check_is,
        check_is_not, num_to_tensor, check_dict_index,
        check_dim, check_not
        )
      );

  # 调用 freeze 函数冻结模型，并将结果保存到 module_freeze 中
  torch::jit::Module module_freeze = freeze(module);

  # 创建一个字符串流 input_model_stream，用于存储模型序列化后的数据
  std::stringstream input_model_stream;
  # 调用 _save_for_mobile 方法将冻结的模型保存到 input_model_stream 中，使用 FlatBuffer 格式
  module_freeze._save_for_mobile(
      input_model_stream,
      /*extra_files=*/{},
      /*save_mobile_debug_info=*/false,
      /*use_flatbuffer=*/true);
  # 创建一个输入数据向量 input_data，包含一个 1x1x28x28 的张量
  std::vector<IValue> input_data =
      std::vector<IValue>({torch::ones({1, 1, 28, 28})});
  # 创建一个期望结果列表 expect_result_list，包含四个张量的初始化结果
  std::vector<IValue> expect_result_list;
  expect_result_list.emplace_back(at::ones({2, 2}, ScalarType::Float) * 0);
  expect_result_list.emplace_back(at::ones({2, 2}, ScalarType::Float));
  expect_result_list.emplace_back(
      at::ones({1, 20, 24, 24}, ScalarType::Float) * 26);
  expect_result_list.emplace_back(3 * at::ones({1}));
  # 向期望结果列表中添加额外的数据项，包括字符串、布尔值和张量
  expect_result_list.emplace_back(c10::IValue("cpu"));
  expect_result_list.emplace_back(c10::IValue(false));
  expect_result_list.emplace_back(c10::IValue(false));
  expect_result_list.emplace_back(c10::IValue(true));
  expect_result_list.emplace_back(c10::IValue(at::ones({1})));
  expect_result_list.emplace_back(c10::IValue("abc"));
  expect_result_list.emplace_back(c10::IValue(2));
  expect_result_list.emplace_back(c10::IValue(false));

  # 调用 backportAllVersionCheck 函数，将模型流、输入数据和期望结果列表作为参数传递，并指定 flatbuffer 版本号为 9
  backportAllVersionCheck(
      input_model_stream,
      input_data,
      expect_result_list,
      9); // flatbuffer starts at 9
TEST(LiteInterpreterTest, GetRuntimeOpsAndInfo) {
  // 调用函数获取运行时操作和信息
  auto runtime_ops = _get_runtime_ops_and_info();
  // 对最小操作数量的粗略估计；用于验证 API 返回的操作数量足够大
  AT_ASSERT(runtime_ops.size() > 2900);
}

TEST(LiteInterpreterTest, isCompatibleSuccess) {
  // 测试平凡的成功情况
  // 获取运行时兼容性信息
  auto runtime_info = RuntimeCompatibilityInfo::get();
  // 创建模型操作的无序映射
  std::unordered_map<std::string, OperatorInfo> model_ops;
  model_ops["aten::add.Scalar"] = OperatorInfo{2};

  // 创建类型集合
  std::unordered_set<std::string> types = {"List", "int", "NamedTuple"};
  // 创建模型兼容性信息
  auto model_info = ModelCompatibilityInfo{
      caffe2::serialize::kMaxSupportedBytecodeVersion,
      model_ops,
      types,
      _get_runtime_bytecode_min_max_versions().first};

  // 断言：检查模型和运行时信息的兼容性状态是否为 OK
  AT_ASSERT(
      is_compatible(runtime_info, model_info).status ==
      ModelCompatibilityStatus::OK);
}
TEST(LiteInterpreterTest, isCompatibleFail) {
  // 定义模型操作符的哈希表，用于存储操作符信息
  std::unordered_map<std::string, OperatorInfo> model_ops;
  // 向模型操作符哈希表中添加一个名为 "aten::add.Scalar" 的操作符信息
  model_ops["aten::add.Scalar"] = OperatorInfo{2};
  // 创建模型兼容性信息对象，包括最大支持的字节码版本和模型操作符哈希表
  auto model_info = ModelCompatibilityInfo{
      caffe2::serialize::kMaxSupportedBytecodeVersion, model_ops};
  // 定义运行时操作符的哈希表，用于存储运行时操作符信息
  std::unordered_map<std::string, OperatorInfo> runtime_ops;
  // 向运行时操作符哈希表中添加一个名为 "aten::add.Int" 的操作符信息
  runtime_ops["aten::add.Int"] = OperatorInfo{2};
  // 创建运行时兼容性信息对象，包括最小和最大支持的字节码版本范围、运行时操作符哈希表和支持的移动设备类型
  auto runtime_info = RuntimeCompatibilityInfo{
      std::pair<uint64_t, uint64_t>(
          caffe2::serialize::kMinSupportedBytecodeVersion,
          caffe2::serialize::kMaxSupportedBytecodeVersion),
      runtime_ops,
      _get_mobile_supported_types()};

  // 调用 is_compatible 函数，判断模型和运行时环境的兼容性，期望返回错误状态
  auto result = is_compatible(runtime_info, model_info);
  // 使用 AT_ASSERT 确保结果状态为 ModelCompatibilityStatus::ERROR
  AT_ASSERT(result.status = ModelCompatibilityStatus::ERROR);
  // 使用 AT_ASSERT 确保错误列表中包含特定错误信息
  AT_ASSERT(
      result.errors[0] ==
      "Operator 'aten::add.Scalar' missing from runtime (not found)");

  // 向运行时操作符哈希表中添加一个名为 "aten::add.Scalar" 的操作符信息
  runtime_ops["aten::add.Scalar"] = OperatorInfo{2};
  // 更新运行时兼容性信息对象，将模型字节码版本设置为超出最大支持版本
  runtime_info = RuntimeCompatibilityInfo{
      std::pair<uint64_t, uint64_t>(
          caffe2::serialize::kMinSupportedBytecodeVersion,
          caffe2::serialize::kMaxSupportedBytecodeVersion),
      runtime_ops,
      _get_mobile_supported_types()};
  model_info.bytecode_version =
      caffe2::serialize::kMaxSupportedBytecodeVersion + 1;

  // 再次调用 is_compatible 函数，期望返回错误状态
  result = is_compatible(runtime_info, model_info);
  // 使用 AT_ASSERT 确保结果状态为 ModelCompatibilityStatus::ERROR
  AT_ASSERT(result.status = ModelCompatibilityStatus::ERROR);

  // 向运行时操作符哈希表中添加一个名为 "aten::add.Scalar" 的操作符信息
  runtime_ops["aten::add.Scalar"] = OperatorInfo{2};
  // 更新运行时兼容性信息对象，将模型字节码版本设置为小于最小支持版本
  runtime_info = RuntimeCompatibilityInfo{
      std::pair<uint64_t, uint64_t>(
          caffe2::serialize::kMinSupportedBytecodeVersion,
          caffe2::serialize::kMaxSupportedBytecodeVersion),
      runtime_ops,
      _get_mobile_supported_types()};
  model_info.bytecode_version =
      caffe2::serialize::kMinSupportedBytecodeVersion - 1;

  // 再次调用 is_compatible 函数，期望返回错误状态
  result = is_compatible(runtime_info, model_info);
  // 使用 AT_ASSERT 确保结果状态为 ModelCompatibilityStatus::ERROR
  AT_ASSERT(result.status = ModelCompatibilityStatus::ERROR);

  // 创建运行时兼容性信息对象，默认包含当前运行时支持的信息
  runtime_info = RuntimeCompatibilityInfo::get();
  // 定义一组类型集合，用于模型兼容性检查
  std::unordered_set<std::string> types = {"List", "int", "Sequence"};

  // 创建模型兼容性信息对象，包括最大支持的字节码版本、模型操作符哈希表、类型集合和运行时最小字节码版本
  model_info = ModelCompatibilityInfo{
      caffe2::serialize::kMaxSupportedBytecodeVersion,
      model_ops,
      types,
      _get_runtime_bytecode_min_max_versions().first};

  // 使用 AT_ASSERT 确保 is_compatible 函数返回错误状态
  AT_ASSERT(
      is_compatible(runtime_info, model_info).status ==
      ModelCompatibilityStatus::ERROR);

  // 创建运行时兼容性信息对象，默认包含当前运行时支持的信息
  runtime_info = RuntimeCompatibilityInfo::get();
  // 创建模型兼容性信息对象，包括最大支持的字节码版本和模型操作符哈希表
  model_info = ModelCompatibilityInfo{
      caffe2::serialize::kMaxSupportedBytecodeVersion, model_ops, {}, 0};

  // 使用 AT_ASSERT 确保 is_compatible 函数返回错误状态
  AT_ASSERT(
      is_compatible(runtime_info, model_info).status ==
      ModelCompatibilityStatus::ERROR);
}
// 定义名为 "Eval" 的测试用例，用于测试轻量级解释器的评估功能
TEST(LiteInterpreterTest, Eval) {
  // 创建一个空的输入向量
  std::vector<torch::jit::IValue> inputs;

  // 创建名为 "m" 的模块对象，并定义其初始化方法和前向传播方法
  Module m("m");
  m.define(R"(
    def __init__(self, x):
      self.training = True

    def forward(self, input):
      return torch.dropout(input, 1.0, self.training)
  )");

  // 向输入向量中添加一个形状为 [1, 1, 28, 28] 的全1张量
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers,modernize-use-emplace)
  inputs.push_back(torch::ones({1, 1, 28, 28}));

  // 将模块设为评估模式
  m.eval();

  // 调用模块的前向传播方法并获取输出张量
  auto outputref = m.forward(inputs).toTensor();

  // 将模块设为训练模式，确保移动端的 eval() 调用正确地切换回评估模式
  m.train();

  // 创建一个字符串流对象，用于保存模块状态
  std::stringstream ss;
  m._save_for_mobile(ss);

  // 从字符串流中加载移动端模块，并将其设为评估模式
  mobile::Module bc = _load_for_mobile(ss);
  bc.eval();

  // 定义一个 IValue 变量 res，并多次调用移动端模块的前向传播方法
  IValue res;
  for (int i = 0; i < 3; ++i) {
    res = bc.get_method("forward")(inputs);
  }

  // 获取最终输出的张量
  auto output = res.toTensor();

  // 断言输出张量的维度与参考输出张量的维度相同
  AT_ASSERT(outputref.dim() == output.dim());

  // 断言输出张量的特定元素值与参考输出张量相同
  AT_ASSERT(
      outputref[0][0][0][0].item<int>() == output[0][0][0][0].item<int>());
}

// 定义名为 "FindWrongMethodName" 的测试用例，用于测试模块查找不存在方法名的行为
TEST(LiteInterpreterTest, FindWrongMethodName) {
  // 创建名为 "m" 的模块对象，并注册一个名为 "foo" 的参数张量
  Module m("m");
  m.register_parameter("foo", torch::ones({}), false);

  // 定义模块的方法，该方法尝试访问不存在的方法 "forward"
  m.define(R"(
    def add(self, x):
      b = 4
      return self.foo + x + b
  )");

  // 创建一个字符串流对象，用于保存模块状态
  std::stringstream ss;
  m._save_for_mobile(ss);

  // 从字符串流中加载移动端模块
  mobile::Module bc = _load_for_mobile(ss);

  // 断言模块在移动端无法找到方法名为 "forward"
  ASSERT_TRUE(bc.find_method("forward") == c10::nullopt);
}

// 定义名为 "FindAndRunMethod" 的测试用例，测试模块查找和运行方法的行为
TEST(LiteInterpreterTest, FindAndRunMethod) {
  // 创建名为 "m" 的模块对象，并注册一个名为 "foo" 的参数张量
  Module m("m");
  m.register_parameter("foo", torch::ones({}), false);

  // 定义模块的方法 "add_it"，该方法执行简单的加法运算
  m.define(R"(
    def add_it(self, x):
      b = 4
      return self.foo + x + b
  )");

  // 创建一个输入向量并添加一个标量张量作为输入
  std::vector<IValue> inputs;
  auto minput = 5 * torch::ones({});
  inputs.emplace_back(minput);

  // 调用模块的方法 "add_it" 并获取其返回值作为参考值
  auto ref = m.get_method("add_it")(inputs);

  // 创建一个字符串流对象，用于保存模块状态
  std::stringstream ss;
  m._save_for_mobile(ss);

  // 从字符串流中加载移动端模块，并定义一个 IValue 变量 res
  mobile::Module bc = _load_for_mobile(ss);
  IValue res;

  // 多次在移动端模块上查找和运行方法 "add_it"
  for (int i = 0; i < 3; ++i) {
    auto bcinputs = inputs;
    auto method = bc.find_method("add_it");
    AT_ASSERT(method != c10::nullopt);
    res = (*method)(std::move(bcinputs));
  }

  // 获取最终输出的浮点数值，并与参考值进行断言比较
  auto resd = res.toTensor().item<float>();
  auto refd = ref.toTensor().item<float>();
  AT_ASSERT(resd == refd);
}

// 定义名为 "RunMethodVariadic" 的测试用例，测试模块调用可变参数方法的行为
TEST(LiteInterpreterTest, RunMethodVariadic) {
  // 创建名为 "m" 的模块对象，并注册一个名为 "foo" 的参数张量
  Module m("m");
  m.register_parameter("foo", torch::ones({}), false);

  // 定义模块的方法 "add_three"，该方法执行三个参数的加法运算
  m.define(R"(
    def add_three(self, x, y):
      return self.foo + x + y
  )");

  // 创建两个输入张量作为方法的参数
  std::vector<IValue> inputs;
  auto inputx = 5 * torch::ones({});
  auto inputy = 4 * torch::ones({});

  // 调用模块的方法 "add_three" 并获取其返回值作为参考值
  auto ref = m.run_method("add_three", inputx, inputy);

  // 创建一个字符串流对象，用于保存模块状态
  std::stringstream ss;
  m._save_for_mobile(ss);

  // 从字符串流中加载移动端模块，并调用方法 "add_three" 并获取返回值
  mobile::Module bc = _load_for_mobile(ss);
  IValue res = bc.run_method("add_three", inputx, inputy);

  // 获取最终输出的浮点数值，并与参考值进行断言比较
  auto resd = res.toTensor().item<float>();
  auto refd = ref.toTensor().item<float>();
  AT_ASSERT(resd == refd);
}

// 定义名为 "DuplicateSetState" 的测试用例，测试模块序列化和反序列化的行为
TEST(LiteInterpreterTest, DuplicateSetState) {
  // 创建名为 "M" 的模块对象，并注册一个名为 "foo" 的参数张量
  Module m("M");
  m.register_parameter("foo", torch::ones({}), false);

  // 定义模块的 "__getstate__" 和 "__setstate__" 方法，实现参数的序列化和反序列化
  m.define(R"(
    def __getstate__(self):
      return self.foo + self.foo
    def __setstate__(self, a):
      self.foo = a
  )");
}


这段代码是一系列用于测试 TorchScript 模块在移动端运行时行为的单元测试。每个测试用例中都定义了模块、方法以及对应的序列化与反序列化操作，用于验证模块在不同环境下的正确性。
    # 定义一个新的模块对象 `m`，名称为 "A"
    Module m("A");
    # 在模块 `m` 中注册两个子模块，名称分别为 "M0" 和 "M1"，这两个子模块都指向模块 `m`
    m.register_module("M0", m);
    m.register_module("M1", m);
    # 定义模块 `m` 的前向传播函数，接收输入 `x`，调用子模块 "M0" 和 "M1" 的前向传播函数，并返回它们的结果的和
    m.define(R"(
      def forward(self, x):
        return self.M0.forward(x) + self.M1.forward(x)
    )");

    # 创建一个字符串流对象 `ss`
    std::stringstream ss;
    # 将模块 `m` 保存为移动端模型格式，并将结果存入字符串流 `ss`
    m._save_for_mobile(ss);
    # 从字符串流 `ss` 加载一个移动端模型，存入 `bc`
    mobile::Module bc = _load_for_mobile(ss);
    # 获取加载后模型 `bc` 中的所有方法
    const auto methods = bc.get_methods();
    # 预期 `bc` 模型中的方法数量为 3
    const size_t expected_n = 3;
    # 断言加载后模型 `bc` 的方法数量与预期数量相等
    ASSERT_EQ(methods.size(), expected_n);
}

TEST(LiteInterpreterTest, ExtraFiles) {
  const auto script = R"JIT(
    def forward(self):
        x = torch.rand(5, 5)                         // 创建一个 5x5 的随机张量 x
        x = x.mm(x)                                  // 计算 x 与其自身的矩阵乘法
        return x                                      // 返回乘法结果 x
  )JIT";

  auto module =
      std::make_shared<Module>("Module", std::make_shared<CompilationUnit>());  // 创建名为 "Module" 的模块
  module->define(script);                                                     // 将脚本定义到模块中
  std::ostringstream oss;                                                    // 创建一个字符串输出流 oss
  std::unordered_map<std::string, std::string> extra_files;                 // 创建一个无序字符串映射 extra_files
  extra_files["metadata.json"] = "abc";                                    // 设置 extra_files 中的 "metadata.json" 为 "abc"
  extra_files["mobile_info.json"] = "{\"key\": 23}";                      // 设置 extra_files 中的 "mobile_info.json" 为 "{\"key\": 23}"
  module->_save_for_mobile(oss, extra_files);                            // 将模块保存为移动端格式到字符串流 oss 中

  std::istringstream iss(oss.str());                                   // 创建一个字符串输入流 iss，使用 oss 的内容
  std::unordered_map<std::string, std::string> loaded_extra_files;    // 创建一个加载后的额外文件映射 loaded_extra_files
  loaded_extra_files["metadata.json"] = "";                           // 初始化 loaded_extra_files 中的 "metadata.json" 为空字符串
  torch::jit::_load_for_mobile(iss, torch::kCPU, loaded_extra_files);  // 加载移动端模块，其中包括 loaded_extra_files

  ASSERT_EQ(loaded_extra_files["metadata.json"], "abc");             // 断言检查加载后的 "metadata.json" 是否等于 "abc"

  loaded_extra_files.clear();                                       // 清空 loaded_extra_files 映射
  std::vector<std::string> all_files =                             // 创建字符串向量 all_files，
      caffe2::serialize::PyTorchStreamReader(&iss).getAllRecords();  // 使用 PyTorchStreamReader 获取所有记录

  for (auto& file_name : all_files) {                             // 遍历 all_files 中的每个文件名
    if (file_name.find("extra/") == 0) {                         // 如果文件名以 "extra/" 开头
      loaded_extra_files[file_name.substr(6)] = "";             // 在 loaded_extra_files 中去掉前缀 "extra/" 后，设置为空字符串
    }
  }
  iss.seekg(0, iss.beg);                                       // 将 iss 流的位置设置回开头
  torch::jit::_load_for_mobile(iss, torch::kCPU, loaded_extra_files);  // 再次加载移动端模块，包括 loaded_extra_files

  ASSERT_EQ(loaded_extra_files["metadata.json"], "abc");         // 断言检查重新加载后的 "metadata.json" 是否等于 "abc"
  ASSERT_EQ(loaded_extra_files["mobile_info.json"], "{\"key\": 23}");  // 断言检查重新加载后的 "mobile_info.json" 是否等于 "{\"key\": 23}"

  std::unordered_map<std::string, std::string>                    // 创建未显式映射的加载后额外文件映射
      loaded_extra_files_without_explicit_mapping;
  iss.seekg(0, iss.beg);                                         // 将 iss 流的位置设置回开头
  torch::jit::_load_for_mobile(                                 // 加载移动端模块，包括未显式映射的 loaded_extra_files_without_explicit_mapping，
      iss,
      torch::kCPU,
      loaded_extra_files_without_explicit_mapping,
      MobileModuleLoadOptions::PARSE_ALL_EXTRA_FILE_MAPS);
  ASSERT_EQ(                                                     // 断言检查加载后的 "metadata.json" 是否等于 "abc"
      loaded_extra_files_without_explicit_mapping["metadata.json"], "abc");
  ASSERT_EQ(                                                     // 断言检查加载后的 "mobile_info.json" 是否等于 "{\"key\": 23}"
      loaded_extra_files_without_explicit_mapping["mobile_info.json"],
      "{\"key\": 23}");
}

TEST(LiteInterpreterTest, OpNameExportFetchRootOperators) {
  torch::jit::Module m("m");                                      // 创建名为 "m" 的 Torch 模块 m
  m.register_parameter("weight", torch::ones({20, 1, 5, 5}), false);  // 注册名为 "weight" 的参数
  m.register_parameter("bias", torch::ones({20}), false);             // 注册名为 "bias" 的参数
  m.define(R"(                                                      // 定义模块的前向计算方法
    def forward(self, input):
      x1 = torch.zeros(2, 2)                                       // 创建一个 2x2 的零张量 x1
      x2 = torch.empty_like(torch.empty(2, 2))                      // 创建一个与给定张量形状相同的空张量 x2
      x3 = torch._convolution(input, self.weight, self.bias, [1, 1], [0, 0], [1, 1], False, [0, 0], 1, False, False, True, True)  // 使用输入、权重和偏置进行卷积计算，得到 x3
      return (x1, x2, x3)                                           // 返回 x1, x2, x3 的元组
  )");
  m.eval();                                                         // 设置模块为评估模式

  std::stringstream ss;                                            // 创建字符串流 ss
  m._save_for_mobile(ss);                                         // 将模块保存为移动端格式到字符串流 ss 中

  torch::jit::mobile::Module ptl_model = torch::jit::_load_for_mobile(ss);  // 加载移动端模块为 ptl_model
  std::set<std::string> operator_names =                               // 创建操作符名称集合 operator_names
      torch::jit::mobile::_export_operator_list(ptl_model);           // 导出 ptl_model 中的操作符列表
  std::set<std::string> expected_operator_names = {                   // 创建预期的操作符名称集合 expected_operator_names
      "aten::_convolution",                                          // 添加操作符 "_convolution"
      "aten::empty.memory_format",                                   // 添加操作符 "empty.memory_format"
      "aten::empty_like",                                            // 添加操作符 "empty_like"
      "aten::zeros",                                                 // 添加操作符 "zeros"
  };
  EXPECT_EQ(operator_names, expected_operator_names)                   // 检查 operator_names 是否与 expected_operator_names 相等
      << "Expected the root operator lists to be the same";            // 如果不相等，输出预期操作符列表应该相同的信息
}

TEST(LiteInterpreterTest, DefaultArgsConv) {
  auto s = std::getenv("PYTORCH_TEST_WITH_TSAN");                     // 获取环境变量 "PYTORCH_TEST_WITH_TSAN" 的值并赋给 s
  if (s && strcmp(s, "1") == 0)                                      // 如果 s 不为空且等于 "1"
    // 空语句，直接返回，表示函数结束
    return;

  // 创建一个存储输入的向量
  std::vector<torch::jit::IValue> inputs;

  // 创建一个名为 m 的神经网络模块
  Module m("m");
  // 注册名为 "weight" 和 "bias" 的参数，并初始化为全1张量
  m.register_parameter("weight", torch::ones({20, 1, 5, 5}), false);
  m.register_parameter("bias", torch::ones({20}), false);
  // 定义模块的前向传播方法，使用 torch.conv2d 进行卷积操作
  m.define(R"(
    def forward(self, input):
      return torch.conv2d(input, self.weight, self.bias, [1, 1], [0, 0], [1, 1], 1)
  )");

  // 将输入张量（1, 1, 28, 28）加入输入向量
  inputs.push_back(torch::ones({1, 1, 28, 28}));

  // 调用模块的 forward 方法进行前向传播，获得输出张量
  auto outputref = m.forward(inputs).toTensor();

  // 创建一个字符串流 ss，用于序列化模块
  std::stringstream ss;
  m._save_for_mobile(ss);
  // 从字符串流中加载移动端的模块 bc
  mobile::Module bc = _load_for_mobile(ss);
  IValue res;
  // 使用加载的移动端模块 bc 进行前向传播，获取结果 res
  for (int i = 0; i < 1; ++i) {
    res = bc.get_method("forward")(inputs);
  }
  // 将结果张量转换为 output
  auto output = res.toTensor();
  // 断言输出张量的维度与参考输出的维度相同
  AT_ASSERT(outputref.dim() == output.dim());
  // 断言输出张量与参考输出张量相等
  AT_ASSERT(output.equal(outputref));
}

TEST(RunTimeTest, ParseBytecode) {
  // 定义一个测试用例：ParseBytecode，用于测试字节码解析的功能

  // 通过简单示例展示一个独立于 PyTorch TorchScript 序列化（反序列化器等）和运算库的简单字节码。
  // 其中包含基本的控制流（if、else）和基本的数据组织（列表构造）。
  // 原始的 PyTorch 程序为：
  //
  //  class Module(torch.nn.Module):
  //  
  //    def __init__(self):
  //      super().__init__()
  //    
  //    def forward(self, x: int, h: int, xfirst: bool):
  //      if xfirst:
  //        return [x, h]
  //      else:
  //        return [h, x]

  // 1. 准备字节码。实际上可以从自定义的反序列化器中获取。
  std::vector<IValue> instructions{
      to_tuple({"STOREN", 1, 4}),
      to_tuple({"DROPR", 1, 0}),
      to_tuple({"MOVE", 4, 0}),
      to_tuple({"JF", 5, 0}),
      to_tuple({"LOAD", 2, 0}),
      to_tuple({"LOAD", 3, 0}),
      to_tuple({"LIST_CONSTRUCT", 0, 2}),
      to_tuple({"JMP", 4, 0}),
      to_tuple({"LOAD", 3, 0}),
      to_tuple({"LOAD", 2, 0}),
      to_tuple({"LIST_CONSTRUCT", 1, 2}),
      to_tuple({"STORE", 5, 0}),
      to_tuple({"DROPR", 3, 0}),
      to_tuple({"DROPR", 2, 0}),
      to_tuple({"MOVE", 5, 0}),
      to_tuple({"RET", 0, 0}),
  };
  std::vector<IValue> operators; // 本例中为空
  std::vector<IValue> constants; // 本例中为空

  std::vector<IValue> types{"List[int]", "List[int]"};
  // 2. 解析函数
  std::string function_name("test_function");
  // 创建一个移动函数对象的唯一指针，其名称为 test_function
  auto function = std::unique_ptr<mobile::Function>(
      new mobile::Function(c10::QualifiedName(function_name)));
  // 调用 parseInstructions 函数解析指令
  c10::ivalue::TupleElements debug_handles_m_tuple;
  parseInstructions(
      function_name,
      std::move(*c10::ivalue::Tuple::create(instructions)).elements(),
      debug_handles_m_tuple,
      function.get());
  // 调用 parseTypes 函数解析类型
  parseTypes(c10::ivalue::Tuple::create(types)->elements(), function.get());
  // 设置寄存器大小
  const size_t rsize = 5;
  parseRegisterSize(rsize, function.get());

  // 3. 准备输入并运行函数
  // 注意第一个输入保留给 Module 对象。
  // 由于这是一个函数测试，不需要 Module 对象，这里添加一个虚拟的 IValue (0)。
  std::vector<IValue> inputs{0, 1, 2, true};
  // 运行函数
  function->run(inputs);
  // 获取输出并进行断言
  auto output = inputs[0].toList();
  ASSERT_EQ(output[0], 1);
  ASSERT_EQ(output[1], 2);

  // 准备第二组输入并运行函数
  std::vector<IValue> inputs1{0, 1, 2, false};
  // 再次运行函数
  function->run(inputs1);
  // 获取输出并进行断言
  auto output1 = inputs1[0].toList();
  ASSERT_EQ(output1[0], 2);
  ASSERT_EQ(output1[1], 1);
}
TEST(RunTimeTest, ParseOperator) {
  // 定义一个测试用例，测试运行时操作符解析
  // 这个示例展示了一个简单的字节码，可以独立于 PyTorch TorchScript 序列化（反序列化器等）和操作符库使用
  // 它包含一个操作符，我们应该能够注册它。原始的 PyTorch 程序如下：

  // class Add(torch.nn.Module):
  //     def __init__(self):
  //         super().__init__()

  //     def forward(self, a, b):
  //         return a + b

  // 1. 准备字节码。实际上，它可以来自自定义的反序列化器。
  std::vector<IValue> instructions{
      to_tuple({"STOREN", 1, 3}),  // 存储操作
      to_tuple({"DROPR", 1, 0}),   // 删除寄存器
      to_tuple({"MOVE", 2, 0}),    // 移动操作
      to_tuple({"MOVE", 3, 0}),    // 移动操作
      to_tuple({"OP", 0, 0}),      // 执行操作
      to_tuple({"RET", 0, 0}),     // 返回操作
  };
  std::vector<IValue> operators{
      to_tuple({"aten::add", "Tensor", 2}),  // 注册一个操作符
  };
  std::vector<IValue> constants{
      to_tuple({1}),  // 常量值
  };
  // 2. 解析函数
  std::string function_name("test_function");
  auto function = std::unique_ptr<mobile::Function>(
      new mobile::Function(c10::QualifiedName(function_name)));
  c10::ivalue::TupleElements debug_handles_m_tuple;
  parseInstructions(
      function_name,
      std::move(*c10::ivalue::Tuple::create(instructions)).elements(),
      debug_handles_m_tuple,
      function.get());
  parseOperators(
      std::move(*c10::ivalue::Tuple::create(operators)).elements(),
      1,
      function.get());
  const size_t rsize = 5;
  parseRegisterSize(rsize, function.get());

  // 3. 准备输入并运行函数
  // 注意第一个输入保留用于模块对象。
  // 由于这是一个函数测试，不需要模块对象，因此在这里添加一个虚拟的 IValue（0）。
  std::vector<IValue> inputs{0, at::tensor(1), at::tensor(2)};
  function->run(inputs);
  auto output = inputs[0];
  ASSERT_EQ(output, at::tensor(3));  // 断言输出结果与预期的张量相等
}

namespace {
void testLiteModuleCompareResultTensors(
    Module& m,
    const std::vector<torch::jit::IValue>& inputs,
    const std::string& method_name = "forward") {
  auto outputref = m.get_method(method_name)(inputs).toTensor();

  std::stringstream ss;
  m._save_for_mobile(ss);
  mobile::Module bc = _load_for_mobile(ss);
  IValue res;
  for (int i = 0; i < 3; ++i) {
    res = bc.get_method(method_name)(inputs);
  }
  auto output = res.toTensor();
  AT_ASSERT(outputref.dim() == output.dim());  // 断言输出张量维度相同
  AT_ASSERT(output.equal(outputref));  // 断言输出张量内容相等
}

void testDefaultArgsPinv(int num_args) {
  Module m("m");
  if (num_args == 1) {
    m.define(R"(
      def forward(self, input):
        return torch.linalg_pinv(input)
    )");
  } else if (num_args == 2) {
    m.define(R"(
      def forward(self, input):
        return torch.linalg_pinv(input, 1e-5)
    )");
  } else if (num_args == 3) {
    m.define(R"(
      def forward(self, input):
        return torch.linalg_pinv(input, 1e-5, True)
    )");
  }
}
    )");



  // 向标准输出流中写入一条多行字符串，这里是一个 C++ 语言代码片段
  // 用于将一个字符串形式的 C++ 代码输出到控制台
  std::vector<torch::jit::IValue> inputs;
  // 定义常量 N 并初始化为 28
  const int N = 28;
  // 创建一个包含从 1 到 N*N 的整数序列的张量
  auto input = torch::range(1, N * N, 1);
  // 将第一个元素设置为 1，以获得一个更稳定的矩阵
  input[0] = 1;
  // 重新调整张量的形状为 N x N
  input = input.view({N, N});
  // 将输入张量添加到输入向量中
  inputs.push_back(input);
  // 调用 testLiteModuleCompareResultTensors 函数，传入模型 m 和输入向量 inputs
  testLiteModuleCompareResultTensors(m, inputs);



  // 这段代码片段的作用是准备输入数据并使用模型进行测试比较
#if !defined FB_XPLAT_BUILD
// 如果未定义 FB_XPLAT_BUILD 宏，则编译以下代码块

TEST(LiteInterpreterTest, DefaultArgsPinv) {
    // 定义一个测试用例，名称为 LiteInterpreterTest.DefaultArgsPinv
    // 测试不同数量的指定参数情况
    // 未指定的参数将采用默认值
    for (int num_args = 1; num_args <= 3; ++num_args) {
        // 循环测试参数个数从 1 到 3
    // 调用一个函数 `testDefaultArgsPinv` 并传入参数 `num_args`
    testDefaultArgsPinv(num_args);
    
    
    
    // 包含一个字节码，具有一个指定的参数：
    // (6,
    //     ('__torch__.m.forward',
    //         (('instructions',
    //             (('STOREN', 1, 2),
    //                 ('DROPR', 1, 0),
    //                 ('MOVE', 2, 0),
    //                 ('OP', 0, 0),
    //                 ('RET', 0, 0))),
    //             ('operators', (('aten::linalg_pinv', '', 1),)),
    //             ('constants', (False, 1e-15)), // 默认常量未使用
    //             ('types', ()),
    //             ('register_size', 2)),
    //         (('arguments',
    //             ((('name', 'self'), ('type', '__torch__.m'), ('default_value',
    //             None)),
    //                 (('name', 'input'), ('type', 'Tensor'), ('default_value',
    //                 None)))),
    //             ('returns',
    //                 ((('name', ''), ('type', 'Tensor'), ('default_value',
    //                 None)),)))))
    
    
    
    // 包含一个字节码，具有两个指定的参数：
    // (6,
    //     ('__torch__.m.forward',
    //         (('instructions',
    //             (('STOREN', 1, 2),
    //                 ('DROPR', 1, 0),
    //                 ('MOVE', 2, 0),
    //                 ('LOADC', 1, 0), // 添加了用于指定参数的LOADC
    //                 ('OP', 0, 0),
    //                 ('RET', 0, 0))),
    //             ('operators', (('aten::linalg_pinv', '', 2),)),
    //             ('constants', (False, 1e-05)), // 更新的常量表
    //             ('types', ()),
    //             ('register_size', 2)),
    //         (('arguments',
    //             ((('name', 'self'), ('type', '__torch__.m'), ('default_value',
    //             None)),
    //                 (('name', 'input'), ('type', 'Tensor'), ('default_value',
    //                 None)))),
    //             ('returns',
    //                 ((('name', ''), ('type', 'Tensor'), ('default_value',
    //                 None)),)))))
    
    
    
    // 包含一个字节码，具有三个指定的参数：
    // (6,
    //     ('__torch__.m.forward',
    //         (('instructions',
    //             (('STOREN', 1, 2),
    //                 ('DROPR', 1, 0),
    //                 ('MOVE', 2, 0),
    //                 ('LOADC', 1, 0),
    //                 ('LOADC', 0, 0),
    //                 ('OP', 0, 0),
    //                 ('RET', 0, 0))),
    //             ('operators', (('aten::linalg_pinv', '', 3),)),
    //             ('constants', (True, 1e-05)),
    //             ('types', ()),
    //             ('register_size', 2)),
    //         (('arguments',
    //             ((('name', 'self'), ('type', '__torch__.m'), ('default_value',
    //             None)),
    //                 (('name', 'input'), ('type', 'Tensor'), ('default_value',
    //                 None)))),
    //             ('returns',
    //                 ((('name', ''), ('type', 'Tensor'), ('default_value',
    //                 None)),)))))
TEST(LiteInterpreterTest, DefaultArgsTensorinvSpecifyDefault) {
  // 测试默认参数的情况下，使用 torch.linalg_tensorinv 函数
  // 第二个参数指定了，但其值与默认值相同，因此被视为“未指定”，因为该值可以从 schema 获取
  Module m("m"); // 创建一个名为 "m" 的模块对象
  m.define(R"(
    def forward(self, input):
      return torch.linalg_tensorinv(input, 2)
  )"); // 定义模块的前向传播方法，使用 torch.linalg_tensorinv 函数进行张量求逆操作
  torch::jit::MobileCode code(m.get_method("forward").graph(), "forward"); // 创建移动端代码对象
  auto arg_nums = code.op_to_num_specified_args(); // 获取每个操作符的指定参数数量
  ASSERT_EQ(arg_nums.size(), 1); // 断言操作符数量为1
  ASSERT_EQ(arg_nums["aten::linalg_tensorinv"], 1); // 断言 torch.linalg_tensorinv 操作符的指定参数数量为1
  std::vector<torch::jit::IValue> inputs; // 创建输入值向量
  const int N = 4; // 设置张量维度大小
  auto input = torch::rand({N, N, N, N}); // 生成一个随机张量作为输入
  inputs.push_back(input); // 将输入张量加入输入值向量
  testLiteModuleCompareResultTensors(m, inputs); // 测试模块前向传播结果与预期张量是否一致
}

void testDefaultArgsPinvWithOutArg(int num_args) {
  Module m("m"); // 创建一个名为 "m" 的模块对象
  if (num_args == 1) {
    m.define(R"(
      def forward(self, input):
        return torch.linalg_pinv(input, out=input)
    )"); // 当参数数量为1时，定义模块的前向传播方法，使用 torch.linalg_pinv 函数进行广义逆操作，指定输出为输入
  } else if (num_args == 2) {
    m.define(R"(
      def forward(self, input):
        return torch.linalg_pinv(input, 1e-5, out=input)
    )"); // 当参数数量为2时，定义模块的前向传播方法，使用 torch.linalg_pinv 函数进行广义逆操作，设置额外参数和输出为输入
  } else if (num_args == 3) {
    m.define(R"(
      def forward(self, input):
        return torch.linalg_pinv(input, 1e-5, True, out=input)
    )"); // 当参数数量为3时，定义模块的前向传播方法，使用 torch.linalg_pinv 函数进行广义逆操作，设置额外参数、布尔标志和输出为输入
  }

  const int N = 28; // 设置张量维度大小
  auto input = torch::range(1, N * N, 1); // 生成一个张量范围从1到N*N的序列
  input[0] = 10000; // 修改第一个元素的值，使得矩阵更为稳定
  input = input.view({N, N}); // 将张量视图变换为 N x N 的形状
  auto ref = m.run_method("forward", input); // 运行模块的前向传播方法，并获取结果
  TORCH_CHECK(!input.equal(torch::range(1, N * N, 1))); // 断言输入张量与指定范围的张量不相等
  TORCH_CHECK(input.equal(ref.toTensor())); // 断言输入张量与前向传播结果张量相等
}

TEST(LiteInterpreterTest, DefaultArgsPinvWithOutArg) {
  // 测试不同数量的指定参数和输出参数的 torch.linalg_pinv 函数
  // 未指定的参数采用默认值
  for (int num_args = 1; num_args <= 3; ++num_args) {
    testDefaultArgsPinvWithOutArg(num_args); // 调用测试函数，测试不同参数数量的 torch.linalg_pinv 函数
  }
}

TEST(LiteInterpreterTest, DefaultArgsWithOutArg) {
  Module m("m"); // 创建一个名为 "m" 的模块对象
  m.define(R"(
    def forward(self, x, h):
      torch.add(x, h, out=x)
  )"); // 定义模块的前向传播方法，使用 torch.add 函数进行张量相加，并将输出结果保存到 x 中

  std::vector<IValue> inputs; // 创建输入值向量
  auto input_x = 2 * torch::ones({}); // 生成一个全为2的张量
  auto input_h = torch::ones({}); // 生成一个全为1的张量
  auto ref = m.run_method("forward", input_x, input_h); // 运行模块的前向传播方法，并获取结果

  std::stringstream ss; // 创建字符串流对象

  m._save_for_mobile(ss, {}, true); // 将模块保存为移动端格式到字符串流中
  mobile::Module bc = _load_for_mobile(ss); // 从字符串流中加载移动端模块
  bc.run_method("forward", input_x, input_h); // 运行加载的移动端模块的前向传播方法
  AT_ASSERT(input_x.equal(4 * torch::ones({}))); // 断言输入张量 x 的值是否等于4

  auto ops = _get_model_ops_and_info(ss); // 获取模型操作和信息
  auto op = ops.find("aten::add.out"); // 查找特定操作符 "aten::add.out"
  TORCH_CHECK(
      op != ops.end() && op->second.num_schema_args.has_value() &&
      op->second.num_schema_args.value() == 3); // 断言操作符存在且其 schema 参数数量为3
}

TEST(LiteInterpreterTest, TestExceptionStackWithTwoLevelModuleHierarchy) {
  Module a("A"); // 创建一个名为 "A" 的模块对象
  a.define(R"(
    def bar(self, x, y):
      return x + y
  )"); // 定义模块的 bar 方法，返回 x 和 y 的和
  Module b("B"); // 创建一个名为 "B" 的模块对象
  b.register_module("A0", a); // 将模块 a 注册为模块 b 的子模块 "A0"
  b.define(R"(
    def foo(self, x, y):
      return self.A0.bar(x, y) + 2
  )"); // 定义模块的 foo 方法，调用子模块 A0 的 bar 方法，并加上2
  Module c("C"); // 创建一个名为 "C" 的模块对象
  c.register_module("B0", b); // 将模块 b 注册为模块 c 的子模块 "B0"
  c.define(R"( // 定义模块的 c 方法
    # 定义一个方法 `forward`，接受两个参数 x 和 y，返回调用 `B0` 对象的 `foo` 方法计算结果再加 3 的值
    def forward(self, x, y):
      return self.B0.foo(x, y) + 3
  )");

  # 创建一个空的 `std::vector`，用于存储 `IValue` 类型的对象
  std::vector<IValue> inputs;
  
  # 在 `inputs` 向量中添加一个形状为 [2, 4] 的随机张量
  inputs.emplace_back(torch::rand({2, 4}));
  
  # 在 `inputs` 向量中添加一个形状为 [13, 9] 的随机张量
  inputs.emplace_back(torch::rand({13, 9}));

  # 创建一个 `std::stringstream` 对象 `ss`，用于字符串流操作
  std::stringstream ss;
  
  # 将模块 `c` 保存为移动端模型到 `ss` 字符串流中，并传递空的额外文件映射和 true 参数
  c._save_for_mobile(ss, ExtraFilesMap(), true);
  
  # 从 `ss` 字符串流中加载移动端模型，存储在 `lite_m` 变量中
  auto lite_m = _load_for_mobile(ss);
  
  # 创建一个字符串 `error_pattern`，包含一段模块层次结构的错误信息
  std::string error_pattern = R"(
  Module hierarchy:top(C)::<unknown>.B0(B)::foo.A0(A)::bar.aten::add
// 定义 TorchScript 模块，并添加 forward 方法，返回一个整型张量，dtype=torch.int64 (4)
m.define(R"(
  def forward(self):
    ret1 = torch.new_empty(torch.zeros(10), [10], dtype=4)
    return ret1.fill_(25)
)");

// 定义 TorchScript 模块，并添加 forward2 方法，返回一个浮点型张量，dtype=torch.float32 (6)
m.define(R"(
  def forward2(self):
    ret1 = torch.new_empty(torch.zeros(10), [10], dtype=6)
    return ret1.fill_(32.0)
)");

// 定义 TorchScript 模块，并添加 forward3 方法，返回一个浮点型张量，其dtype由输入张量的dtype推断而来
m.define(R"(
  def forward3(self):
    ret1 = torch.new_empty(torch.zeros(10), [10])
    return ret1.fill_(12.0)
)");

// 测试用例，比较 Lite 模块和完全 JIT 编译模块的结果张量是否相同
std::vector<torch::jit::IValue> inputs;
testLiteModuleCompareResultTensors(m, inputs, "forward");
testLiteModuleCompareResultTensors(m, inputs, "forward2");
testLiteModuleCompareResultTensors(m, inputs, "forward3");
TEST(RunTimeTest, RuntimeCall) {
  // 定义指令序列 instructionsCall 用于模拟函数 call 的指令流
  std::vector<IValue> instructionsCall{
      to_tuple({"STORE", 1, 0}),   // 存储指令，将第1个元素存入位置0
      to_tuple({"LOAD", 1, 0}),    // 载入指令，加载位置0的值到第1个元素
      to_tuple({"MOVE", 1, 0}),    // 移动指令，将位置0的值移动到第1个元素
      to_tuple({"LOADC", 0, 0}),   // 载入常量指令，加载常量0到位置0
      to_tuple({"OP", 0, 0}),      // 运算指令，对位置0的值进行操作
      to_tuple({"RET", 0, 0}),     // 返回指令，返回位置0的值
  };
  // 定义指令序列 instructionsFoo 用于模拟函数 foo 的指令流
  std::vector<IValue> instructionsFoo{
      to_tuple({"STORE", 1, 0}),   // 存储指令，将第1个元素存入位置0
      to_tuple({"LOAD", 1, 0}),    // 载入指令，加载位置0的值到第1个元素
      to_tuple({"LOAD", 1, 0}),    // 载入指令，加载位置0的值到第1个元素
      to_tuple({"MOVE", 1, 0}),    // 移动指令，将位置0的值移动到第1个元素
      to_tuple({"CALL", 0, 0}),    // 调用指令，调用位置0的函数
      to_tuple({"LOADC", 0, 0}),   // 载入常量指令，加载常量0到位置0
      to_tuple({"OP", 0, 0}),      // 运算指令，对位置0的值进行操作
      to_tuple({"CALL", 0, 0}),    // 调用指令，再次调用位置0的函数
      to_tuple({"LOADC", 0, 0}),   // 载入常量指令，加载常量0到位置0
      to_tuple({"OP", 0, 0}),      // 运算指令，对位置0的值进行操作
      to_tuple({"RET", 0, 0}),     // 返回指令，返回位置0的值
  };
  // 定义操作符序列 operatorsFoo 用于函数 foo 中的操作符
  std::vector<IValue> operatorsFoo{
      to_tuple({"aten::add", "Tensor", 3}),   // 添加操作符，作用于类型为 Tensor 的对象，有3个参数
  };
  // 定义常量序列 constantsFoo 用于函数 foo 中的常量
  std::vector<IValue> constantsFoo{
      1,   // 常量值为1
  };
  // 定义操作符序列 operatorsCall 用于函数 call 中的操作符
  std::vector<IValue> operatorsCall{
      to_tuple({"aten::add", "Tensor", 3}),   // 添加操作符，作用于类型为 Tensor 的对象，有3个参数
  };
  // 定义常量序列 constantsCall 用于函数 call 中的常量
  std::vector<IValue> constantsCall{
      1,   // 常量值为1
  };

  // 创建名为 foo 的移动函数对象
  auto foo = std::make_unique<mobile::Function>(c10::QualifiedName("foo"));
  c10::ivalue::TupleElements debug_handles_m_tuple;
  // 解析函数 foo 的指令、操作符、常量及寄存器大小
  parseInstructions(
      "foo",
      std::move(*c10::ivalue::Tuple::create(instructionsFoo)).elements(),
      debug_handles_m_tuple,
      foo.get());
  parseOperators(
      std::move(*c10::ivalue::Tuple::create(operatorsFoo)).elements(),
      1,
      foo.get());
  parseConstants(
      std::move(*c10::ivalue::Tuple::create(constantsFoo)).elements(),
      foo.get());
  const size_t rsize = 5;
  parseRegisterSize(rsize, foo.get());

  // 创建名为 call 的移动函数对象
  auto call = std::make_unique<mobile::Function>(c10::QualifiedName("call"));
  // 解析函数 call 的指令、操作符、常量及寄存器大小
  parseInstructions(
      "call",
      std::move(*c10::ivalue::Tuple::create(instructionsCall)).elements(),
      debug_handles_m_tuple,
      call.get());
  parseOperators(
      std::move(*c10::ivalue::Tuple::create(operatorsCall)).elements(),
      1,
      call.get());
  parseConstants(
      std::move(*c10::ivalue::Tuple::create(constantsCall)).elements(),
      call.get());
  parseRegisterSize(rsize, call.get());

  // 在 foo 函数中附加 call 函数
  foo->append_function(*call);

  // 定义输入向量 inputs
  std::vector<IValue> inputs{at::tensor(1)};
  // 运行 foo 函数，传入输入向量 inputs
  foo->run(inputs);
  // 获取运行结果 output，并验证其等于预期值
  auto output = inputs[0];
  ASSERT_EQ(output, at::tensor(7));
}

TEST(LiteInterpreterTest, OperatorSize1) {
  // 创建名为 m 的模块
  Module m("m");
  // 定义模块 m 的前向方法，包括输入为 Tensor 类型的 input 和 float 类型的 scale
  m.define(R"(
    def forward(self, input: Tensor, scale:float):
      return torch.upsample_nearest2d(input, [1, 1], float(scale), float(scale))
  )");

  // 创建字符串流 ss
  std::stringstream ss;
  // 将模块 m 保存为移动模块格式到 ss 中
  m._save_for_mobile(ss);
  // 从字符串流 ss 中加载移动模块为 bc
  mobile::Module bc = _load_for_mobile(ss);
  // 获取模块 bc 的 forward 方法的函数对象
  const auto& func = bc.get_method("forward").function();
  // 断言操作符输入大小与操作符数量相等
  ASSERT_EQ(
      func.get_code().operator_input_sizes_.size(),
      func.get_code().operators_.size());
}
TEST(LiteInterpreterTest, OperatorTest2) { // NOLINT (use =delete in gtest)
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
  // 遍历测试程序列表
  for (const auto& test_program : test_programs) {
    // 创建名为 "m" 的模块对象
    Module m("m");
    // 注册名为 "foo" 的参数，初始值为 torch::ones({})，不可变
    m.register_parameter("foo", torch::ones({}), false);
    // 定义当前测试程序
    m.define(test_program);

    // 创建一个字符串流对象
    std::stringstream ss;
    // 将模块保存为移动端格式到字符串流中
    m._save_for_mobile(ss);
    // 从字符串流中加载移动端模块
    mobile::Module bc = _load_for_mobile(ss);
    // 获取名为 "test_func" 的方法函数
    const auto& func = bc.get_method("test_func").function();
    // 断言操作数输入大小与操作数数量相等
    ASSERT_EQ(
        func.get_code().operator_input_sizes_.size(),
        func.get_code().operators_.size());
  }
}

#if !defined FB_XPLAT_BUILD
// The following test run in fbcode only
TEST(LiteInterpreterUpgraderTest, DivTensorV2) {
  // 获取当前文件路径
  std::string filePath(__FILE__);
  // 拼接测试模型文件路径
  auto test_model_file = filePath.substr(0, filePath.find_last_of("/\\") + 1);
  test_model_file.append("upgrader_models/test_versioned_div_tensor_v2.ptl");
  /*
  (('__torch__.MyModule.forward',
    (('instructions',
      (('STOREN', 1, 3),
       ('DROPR', 1, 0),
       ('LOAD', 2, 0),
       ('LOAD', 3, 0),
       ('OP', 0, 0),
       ('LOAD', 2, 0),
       ('LOAD', 3, 0),
       ('OP', 1, 0),
       ('MOVE', 2, 0),
       ('MOVE', 3, 0),
       ('OP', 2, 0),
       ('TUPLE_CONSTRUCT', 3, 0),
       ('RET', 0, 0))),
     ('operators',
      (('aten::div', 'Tensor'),
       ('aten::div', 'Tensor'),
       ('aten::div', 'Tensor'))),
     ('constants', ()),
     ('types', ()),
     ('register_size', 3))),)

  */
  // 加载移动端模块
  mobile::Module m_module = _load_for_mobile(test_model_file);
  // 获取名为 "forward" 的方法的指令列表
  auto instruction_list =
      m_module.get_method("forward").function().get_code().instructions_;
  // 统计 CALL 操作码的数量
  uint64_t number_of_call_instruction = 0;
  for (auto& instruction : instruction_list) {
    number_of_call_instruction += (instruction.op == OpCode::CALL);
  }
  // 断言 CALL 操作码的数量为 3
  ASSERT_EQ(number_of_call_instruction, 3);

  // 定义输入数据列表
  std::vector<IValue> inputs = {
      IValue(6 * torch::ones({1})), IValue(3 * torch::ones({1}))};
  // 调用模块的 forward 方法，获取实际输出
  auto actual_output = m_module.forward(inputs);
  // 期望的输出结果
  auto expect_output = 2.0 * torch::ones({1});
  // 获取实际输出的元组元素列表
  auto actual_output_list = actual_output.toTuple()->elements();
  // 断言实际输出与期望输出相等
  ASSERT_TRUE(actual_output_list[0].toTensor().equal(expect_output));
}
TEST(LiteInterpreterUpgraderTest, DivTensorOutV2) {
  // 获取当前文件路径
  std::string filePath(__FILE__);
  // 构造测试模型文件路径
  auto test_model_file = filePath.substr(0, filePath.find_last_of("/\\") + 1);
  test_model_file.append(
      "upgrader_models/test_versioned_div_tensor_out_v2.ptl");
  /*
  定义模型前向方法的序列化指令和相关信息
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
  // 加载移动端模型
  mobile::Module m_module = _load_for_mobile(test_model_file);

  // 获取前向方法的指令列表
  auto instruction_list =
      m_module.get_method("forward").function().get_code().instructions_;
  // 统计调用指令的数量
  uint64_t number_of_call_instruction = 0;
  for (auto& instruction : instruction_list) {
    number_of_call_instruction += (instruction.op == OpCode::CALL);
  }
  // 断言：期望只有一个调用指令
  ASSERT_EQ(number_of_call_instruction, 1);

  // 设置输入向量
  std::vector<IValue> inputs{
      IValue(6 * torch::ones({1})),
      IValue(3 * torch::ones({1})),
      IValue(torch::empty({1}))};
  // 执行前向方法
  m_module.forward(inputs);
  // 期望的输出
  auto expect_output = 2.0 * torch::ones({1});
  // 实际输出
  auto actual_output = inputs[2].toTensor();
  // 断言：输出张量应该等于期望的输出
  ASSERT_TRUE(actual_output.equal(expect_output));
}

TEST(LiteInterpreterUpgraderTest, DivTensorInplaceV2) {
  // 获取当前文件路径
  std::string filePath(__FILE__);
  // 构造测试模型文件路径
  auto test_model_file = filePath.substr(0, filePath.find_last_of("/\\") + 1);
  test_model_file.append(
      "upgrader_models/test_versioned_div_tensor_inplace_v2.ptl");
  /*
  定义模型前向方法的序列化指令和相关信息
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
  // 加载移动端模型
  mobile::Module m_module = _load_for_mobile(test_model_file);

  // 获取前向方法的指令列表
  auto instruction_list =
      m_module.get_method("forward").function().get_code().instructions_;
  // 统计调用指令的数量
  uint64_t number_of_call_instruction = 0;
  for (auto& instruction : instruction_list) {
    number_of_call_instruction += (instruction.op == OpCode::CALL);
  }
  // 断言：期望只有一个调用指令
  ASSERT_EQ(number_of_call_instruction, 1);

  // 设置输入向量
  std::vector<IValue> inputs{
      IValue(6 * torch::ones({1})), IValue(3 * torch::ones({1}))};
  // 执行前向方法
  m_module.forward(inputs);
  // 期望的输出
  auto expect_output = 2.0 * torch::ones({1});
  // 实际输出
  auto actual_output = inputs[0].toTensor();
  // 断言：输出张量应该等于期望的输出
  ASSERT_TRUE(actual_output.equal(expect_output));
}
TEST(LiteInterpreterUpgraderTest, DivScalarFloatV2) {
  // 获取当前文件路径
  std::string filePath(__FILE__);
  // 构建测试模型文件路径
  auto test_model_file = filePath.substr(0, filePath.find_last_of("/\\") + 1);
  test_model_file.append(
      "upgrader_models/test_versioned_div_scalar_float_v2.ptl");
  /*
  (('__torch__.MyModuleFloat.forward',
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
  
  // 使用_load_for_mobile函数加载模型文件
  mobile::Module m_module = _load_for_mobile(test_model_file);

  // 获取模型中forward方法的指令列表
  auto intrsuction_list =
      m_module.get_method("forward").function().get_code().instructions_;
  // 统计CALL指令的数量
  uint64_t number_of_call_instruction = 0;
  for (auto& instruction : intrsuction_list) {
    number_of_call_instruction += (instruction.op == OpCode::CALL);
  }
  // 断言CALL指令的数量为1
  // 表示有一个操作符使用了升级器
  ASSERT_EQ(number_of_call_instruction, 1);

  // 构建输入值列表
  std::vector<IValue> inputs{IValue(6 * torch::ones({1})), IValue(3.0)};
  // 调用模型的forward方法获取输出
  auto output = m_module.forward(inputs);
  // 期望的输出
  auto expect_output = 2.0 * torch::ones({1});
  auto actual_output = output.toTensor();

  // 断言实际输出与期望输出相等
  ASSERT_TRUE(actual_output.equal(expect_output));
}

TEST(LiteInterpreterUpgraderTest, DivScalarReciprocalFloatV2) {
  // 获取当前文件路径
  std::string filePath(__FILE__);
  // 构建测试模型文件路径
  auto test_model_file = filePath.substr(0, filePath.find_last_of("/\\") + 1);
  test_model_file.append(
      "upgrader_models/test_versioned_div_scalar_reciprocal_float_v2.ptl");
  /*
  (('__torch__.MyModuleFloat.forward',
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
  
  // 使用_load_for_mobile函数加载模型文件
  mobile::Module m_module = _load_for_mobile(test_model_file);

  // 获取模型中forward方法的指令列表
  auto intrsuction_list =
      m_module.get_method("forward").function().get_code().instructions_;
  // 统计CALL指令的数量
  uint64_t number_of_call_instruction = 0;
  for (auto& instruction : intrsuction_list) {
    number_of_call_instruction += (instruction.op == OpCode::CALL);
  }
  // 断言CALL指令的数量为0
  // 表示没有操作符使用了升级器
  ASSERT_EQ(number_of_call_instruction, 0);

  // 构建输入值列表
  std::vector<IValue> inputs{IValue(6 * torch::ones({1})), IValue(3.0)};
  // 调用模型的forward方法获取输出
  auto output = m_module.forward(inputs);
  // 期望的输出
  auto expect_output = 0.5 * torch::ones({1});
  auto actual_output = output.toTensor();

  // 输出期望输出和实际输出，用于调试
  std::cout << "expect output: " << expect_output;
  std::cout << "actual output: " << actual_output;
  // 断言实际输出与期望输出相等
  ASSERT_TRUE(actual_output.equal(expect_output));
}
TEST(LiteInterpreterUpgraderTest, DivScalarReciprocalIntV2) {
  // 获取当前文件的路径
  std::string filePath(__FILE__);
  // 构建测试模型文件的完整路径
  auto test_model_file = filePath.substr(0, filePath.find_last_of("/\\") + 1);
  test_model_file.append(
      "upgrader_models/test_versioned_div_scalar_reciprocal_int_v2.ptl");
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
  // 加载移动模块文件
  mobile::Module m_module = _load_for_mobile(test_model_file);

  // 获取 forward 方法的指令列表
  auto intrsuction_list =
      m_module.get_method("forward").function().get_code().instructions_;
  // 统计 CALL 操作码的数量
  uint64_t number_of_call_instruction = 0;
  for (auto& instruction : intrsuction_list) {
    number_of_call_instruction += (instruction.op == OpCode::CALL);
  }
  // 断言没有 CALL 操作码被使用
  ASSERT_EQ(number_of_call_instruction, 0);

  // 准备输入向量
  std::vector<IValue> inputs{IValue(6 * torch::ones({1})), IValue(3.0)};
  // 执行模型的 forward 方法
  auto output = m_module.forward(inputs);
  // 准备期望的输出
  auto expect_output = 0.5 * torch::ones({1});
  // 获取实际输出
  auto actual_output = output.toTensor();

  // 断言实际输出等于期望输出
  ASSERT_TRUE(actual_output.equal(expect_output));
}

TEST(LiteInterpreterUpgraderTest, DivScalarScalarV2) {
  // 获取当前文件的路径
  std::string filePath(__FILE__);
  // 构建测试模型文件的完整路径
  auto test_model_file = filePath.substr(0, filePath.find_last_of("/\\") + 1);
  test_model_file.append(
      "upgrader_models/test_versioned_div_scalar_scalar_v2.ptl");
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
  // 加载移动模块文件
  mobile::Module m_module = _load_for_mobile(test_model_file);
  // 获取 forward 方法的指令列表
  auto intrsuction_list =
      m_module.get_method("forward").function().get_code().instructions_;
  // 统计 CALL 操作码的数量
  uint64_t number_of_call_instruction = 0;
  for (auto& instruction : intrsuction_list) {
    // 计算 CALL 操作码的数量
    number_of_call_instruction += (instruction.op == OpCode::CALL);
  }
  // 增加调用指令的数量，如果当前指令是 CALL 操作码
  number_of_call_instruction += (instruction.op == OpCode::CALL);
}
// 确保没有操作符使用 upgrader
ASSERT_EQ(number_of_call_instruction, 0);

// 创建包含四个 IValue 元素的输入向量
std::vector<IValue> inputs{IValue(20.0), IValue(10), IValue(2.0), IValue(5)};
// 使用模块 m_module 进行前向推断，得到输出
auto output = m_module.forward(inputs);
// 将输出转换为元组引用，并获取其元素列表
auto output_list = output.toTupleRef().elements();
// 定义期望的输出向量
auto expect_output = std::vector<IValue>(
    {IValue(2.0), IValue(10.0), IValue(5.0), IValue(2.0)});
// 检查每个输出元素是否与期望输出相等
for (size_t i = 0; i < expect_output.size(); i++) {
  ASSERT_EQ(output_list[i], expect_output[i]);
}
// 定义一个测试用例，测试LiteInterpreterUpgraderTest类中的DivScalarIntV2方法
TEST(LiteInterpreterUpgraderTest, DivScalarIntV2) {
  // 获取当前文件的路径
  std::string filePath(__FILE__);
  // 从文件路径中提取目录部分，用于拼接模型文件路径
  auto test_model_file = filePath.substr(0, filePath.find_last_of("/\\") + 1);
  // 拼接具体的模型文件路径
  test_model_file.append("upgrader_models/test_versioned_div_scalar_int_v2.ptl");

  /*
  以下是模型文件中的序列化信息，包含模块名称、前向指令、操作符、常量、类型和寄存器大小信息
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

  // 加载模型文件为移动端模块
  mobile::Module m_module = _load_for_mobile(test_model_file);

  // 获取前向方法的指令列表
  auto instruction_list = m_module.get_method("forward").function().get_code().instructions_;
  uint64_t number_of_call_instruction = 0;
  // 统计调用指令的数量
  for (auto& instruction : instruction_list) {
    number_of_call_instruction += (instruction.op == OpCode::CALL);
  }
  // 断言只有一个操作符使用了升级器
  ASSERT_EQ(number_of_call_instruction, 1);

  // 准备输入数据
  std::vector<IValue> inputs{IValue(6 * torch::ones({1})), IValue(3)};
  // 执行前向传播
  auto output = m_module.forward(inputs);
  auto expect_output = 2.0 * torch::ones({1});
  auto actual_output = output.toTensor();

  // 断言输出是否与预期一致
  ASSERT_TRUE(actual_output.equal(expect_output));
}



// 定义一个测试用例，测试LiteInterpreterUpgraderTest类中的DivScalarInplaceFloatV2方法
TEST(LiteInterpreterUpgraderTest, DivScalarInplaceFloatV2) {
  // 获取当前文件的路径
  std::string filePath(__FILE__);
  // 从文件路径中提取目录部分，用于拼接模型文件路径
  auto test_model_file = filePath.substr(0, filePath.find_last_of("/\\") + 1);
  // 拼接具体的模型文件路径
  test_model_file.append("upgrader_models/test_versioned_div_scalar_inplace_float_v2.ptl");

  /*
  以下是模型文件中的序列化信息，包含模块名称、前向指令、操作符、常量、类型和寄存器大小信息
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

  // 加载模型文件为移动端模块
  mobile::Module m_module = _load_for_mobile(test_model_file);

  // 获取前向方法的指令列表
  auto instruction_list = m_module.get_method("forward").function().get_code().instructions_;
  uint64_t number_of_call_instruction = 0;
  // 统计调用指令的数量
  for (auto& instruction : instruction_list) {
    number_of_call_instruction += (instruction.op == OpCode::CALL);
  }
  // 断言只有一个操作符使用了升级器
  ASSERT_EQ(number_of_call_instruction, 1);

  // 准备输入数据
  std::vector<IValue> inputs{IValue(6 * torch::ones({1})), IValue(3.0)};
  // 执行前向传播
  auto output = m_module.forward(inputs);
  auto expect_output = 2.0 * torch::ones({1});
  auto actual_output = output.toTensor();

  // 断言输出是否与预期一致
  ASSERT_TRUE(actual_output.equal(expect_output));
}



// 定义一个测试用例，测试LiteInterpreterUpgraderTest类中的DivScalarInplaceIntV2方法
TEST(LiteInterpreterUpgraderTest, DivScalarInplaceIntV2) {
  // 获取当前文件的路径
  std::string filePath(__FILE__);
  // 从文件路径中提取目录部分，用于拼接模型文件路径
  auto test_model_file = filePath.substr(0, filePath.find_last_of("/\\") + 1);
  // 拼接具体的模型文件路径
  test_model_file.append("upgrader_models/test_versioned_div_scalar_inplace_int_v2.ptl");

  /*
  以下是模型文件中的序列化信息，包含模块名称、前向指令、操作符、常量、类型和寄存器大小信息
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

  // 加载模型文件为移动端模块
  mobile::Module m_module = _load_for_mobile(test_model_file);

  // 获取前向方法的指令列表
  auto instruction_list = m_module.get_method("forward").function().get_code().instructions_;
  uint64_t number_of_call_instruction = 0;
  // 统计调用指令的数量
  for (auto& instruction : instruction_list) {
    number_of_call_instruction += (instruction.op == OpCode::CALL);
  }
  // 断言只有一个操作符使用了升级器
  ASSERT_EQ(number_of_call_instruction, 1);

  // 准备输入数据
  std::vector<IValue> inputs{IValue(6 * torch::ones({1})), IValue(3)};
  // 执行前向传播
  auto output = m_module.forward(inputs);
  auto expect_output = 2.0 * torch::ones({1});
  auto actual_output = output.toTensor();

  // 断言输出是否与预期一致
  ASSERT_TRUE(actual_output.equal(expect_output));
}
    /*
      获取用于描述模型的 mobile::Module 对象，通过加载移动端模型文件实现
    */
    mobile::Module m_module = _load_for_mobile(test_model_file);
    
    /*
      获取模型中 "forward" 方法的指令列表，并保存在 intrsuction_list 中
    */
    auto intrsuction_list =
        m_module.get_method("forward").function().get_code().instructions_;
    
    /*
      统计指令列表中 CALL 操作码的出现次数，记录在 number_of_call_instruction 中
    */
    uint64_t number_of_call_instruction = 0;
    for (auto& instruction : intrsuction_list) {
      number_of_call_instruction += (instruction.op == OpCode::CALL);
    }
    
    // 断言：期望 CALL 操作码出现一次
    ASSERT_EQ(number_of_call_instruction, 1);
    
    /*
      准备模型前向推断所需的输入数据，以 IValue 类型存储在 inputs 向量中
    */
    std::vector<IValue> inputs{IValue(6 * torch::ones({1})), IValue(3)};
    auto output = m_module.forward(inputs);  // 调用模型的 forward 方法进行推断
    auto expect_output = 2.0 * torch::ones({1});  // 期望的推断输出结果
    auto actual_output = output.toTensor();  // 将模型输出转换为 Tensor 类型
    
    // 断言：实际输出 actual_output 应与期望输出 expect_output 相等
    ASSERT_TRUE(actual_output.equal(expect_output));
    
    /*
      断言：输出参数 out 将会被推断结果覆盖
    */
// 结束宏定义区域
}

// 测试 LiteInterpreterUpgraderTest 的 Upgrader 函数
TEST(LiteInterpreterUpgraderTest, Upgrader) {
  // 创建升级函数向量
  std::vector<mobile::Function> upgrader_functions;

  // 遍历升级字节码列表
  for (auto& byteCodeFunctionWithOperator : getUpgraderBytecodeList()) {
    // 初始化操作符
    byteCodeFunctionWithOperator.function.initialize_operators(true);
    // 断言操作符数量和操作符名称数量相等
    ASSERT_EQ(
        byteCodeFunctionWithOperator.function.get_code().operators_.size(),
        byteCodeFunctionWithOperator.function.get_code().op_names_.size());
    // 如果操作符为空
    if (byteCodeFunctionWithOperator.function.get_code().operators_.empty()) {
      // 添加操作符
      for (const auto& op : byteCodeFunctionWithOperator.operators) {
        byteCodeFunctionWithOperator.function.append_operator(
            op.name, op.overload_name, op.num_specified_args);
      }
    }
    // 将函数添加到升级函数向量中
    upgrader_functions.push_back(byteCodeFunctionWithOperator.function);
  }

  // 断言升级字节码列表和升级函数向量的大小相等
  ASSERT_EQ(getUpgraderBytecodeList().size(), upgrader_functions.size());
}

// 枚举元组类型
void enumerateTupleType(
    size_t depth,
    std::vector<TypePtr>& current,
    const std::vector<TypePtr>& candidates,
    std::vector<TypePtr>& out) {
  // 静态字段名向量
  static std::vector<std::string> fieldNames;
  // 如果深度大于字段名向量大小
  if (depth > fieldNames.size()) {
    // 扩展字段名向量
    fieldNames.reserve(depth);
    for (size_t i = fieldNames.size(); i < depth; i++) {
      fieldNames.push_back("field" + std::to_string(i));
    }
  }
  // 如果深度为 0
  if (depth == 0) {
    // 创建无名称元组类型并添加到输出向量
    out.push_back(TupleType::create(current));
    // 缩减字段名向量
    while (fieldNames.size() > current.size()) {
      fieldNames.pop_back();
    }
    // 创建命名元组类型并添加到输出向量
    out.push_back(TupleType::createNamed("NamedTuple", fieldNames, current));
    return;
  }
  // 遍历候选类型
  for (const auto& type : candidates) {
    // 如果包含任意类型则跳过
    if (containsAnyType(type)) {
      continue;
    }
    // 添加当前类型到当前向量，递归调用枚举元组类型
    current.push_back(type);
    enumerateTupleType(depth - 1, current, candidates, out);
    current.pop_back();
  }
}

// LiteInterpreterDynamicTypeTestFixture 类，继承自测试框架
class LiteInterpreterDynamicTypeTestFixture
    : public ::testing::TestWithParam<size_t> {
 protected:
  void SetUp() override {
    // 创建编译单元
    cu = std::make_shared<CompilationUnit>();
    // 关键类型向量
    std::vector<TypePtr> keyTypes = {
        AnyType::get(),
        IntType::get(),
        BoolType::get(),
        FloatType::get(),
        ComplexType::get(),
        StringType::get(),
        TensorType::get(),
        DeviceObjType::get(),
    };
    // 类型向量
    types = {
        NoneType::get(),
        NumberType::get(),
        ClassType::create("__torch__.TestClass1", cu),
        ClassType::create("__torch__.TestClass2", cu),
        AnyListType::get(),
        AnyTupleType::get(),
        StreamObjType::get(),
        CapsuleType::get(),
        GeneratorType::get(),
        StorageType::get(),
        VarType::create("t"),
        VarType::create("v"),
        AnyClassType::get()};
    // 将关键类型添加到类型向量
    std::copy(keyTypes.begin(), keyTypes.end(), back_inserter(types));
    // 定义一个 lambda 函数 expandTypes，用于根据给定的元组大小扩展类型
    auto expandTypes = [&](size_t tupleSize) {
      // 声明一个空的 TypePtr 向量 nested，用于存储扩展后的类型
      std::vector<TypePtr> nested;
      // 遍历已有的 types 向量
      for (const auto& type : types) {
        // 如果当前类型不是 AnyType，则创建一个 ListType 的类型并添加到 nested 中
        if (!(type == AnyType::get())) {
          nested.emplace_back(ListType::create(type));
          // 如果当前类型不是 NoneType 或者不是 OptionalType 的类型，则创建一个 OptionalType 的类型并添加到 nested 中
          if (!(type == NoneType::get() ||
                type->kind() == OptionalType::Kind)) {
            nested.emplace_back(OptionalType::create(type));
          }
        }
        // 遍历 keyTypes 向量，为每一个 keyType 创建一个 DictType 的类型，并将其添加到 nested 中
        for (const auto& keyType : keyTypes) {
          nested.emplace_back(DictType::create(keyType, type));
        }
      }
      // 声明一个临时的 TypePtr 向量 tmp，调用 enumerateTupleType 函数处理 tupleSize、types 和 nested 向量
      std::vector<TypePtr> tmp;
      enumerateTupleType(tupleSize, tmp, types, nested);
      // 将 nested 向量中的元素移动到 types 向量的末尾
      std::move(
          std::begin(nested), std::end(nested), std::back_inserter(types));
    };
    // 调用 expandTypes 函数，扩展类型，tupleSize 参数为 1
    expandTypes(1);
    // 再次调用 expandTypes 函数，扩展类型，tupleSize 参数再次为 1
    expandTypes(1);
  }
  // 声明一个指向 CompilationUnit 的 shared_ptr，命名为 cu
  std::shared_ptr<CompilationUnit> cu;
  // 声明一个 TypePtr 向量 types，用于存储类型信息
  std::vector<TypePtr> types;

 public:
  // 声明一个静态 constexpr 大小为 10 的常量 kNumSplits，表示分割数量
  static constexpr size_t kNumSplits = 10;
};

/**
 * Enumerate all possible JIT types appearing in mobile runtime, and test
 * whether subtyping relation is preserved after one of the JIT types is
 * converted to DynamicType.
 *
 * We firstly enumerate all "base" types in a vector, and implement
 * expandTypes() to enumerate container types one "level" up for a given set
 * of types. We call expandTypes() twice to test types nested less or equal
 * to two levels. e.g. List[Optional[Tensor]], Optional[Dict[Int, Bool]], etc.
 */
TEST_P(LiteInterpreterDynamicTypeTestFixture, Conformance) {
  // Determine the number of types to process per test instance
  size_t num = types.size() / LiteInterpreterDynamicTypeTestFixture::kNumSplits;
  // Calculate the starting index for this test instance
  size_t begin = num * GetParam();
  // Calculate the ending index for this test instance, ensuring it does not exceed the total number of types
  size_t end = std::min(types.size(), begin + num);
  // Iterate over all types in the 'types' vector
  for (const auto& a : types) {
    // Create a DynamicType object from type 'a'
    auto da = DynamicType::create(*a);
    // Iterate over a subset of types determined by 'begin' and 'end'
    for (size_t i = begin; i < end; i++) {
      const auto& b = types[i];
      // Check if 'a' is a subtype of 'b' and compare with the DynamicType version
      bool result = a->isSubtypeOf(*b);
      EXPECT_EQ(result, da->isSubtypeOf(*b));
      // Check if 'b' is a subtype of 'a' and compare with the DynamicType version
      result = b->isSubtypeOf(*a);
      EXPECT_EQ(result, b->isSubtypeOf(*da));
    }
  }
}

// Instantiate the test suite for different ranges of test parameters
INSTANTIATE_TEST_SUITE_P(
    PyTorch,
    LiteInterpreterDynamicTypeTestFixture,
    ::testing::Range(
        static_cast<size_t>(0),
        LiteInterpreterDynamicTypeTestFixture::kNumSplits));

// End of namespace declarations for 'jit' and 'torch'
} // namespace jit
} // namespace torch
```