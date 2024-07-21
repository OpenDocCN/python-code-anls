# `.\pytorch\test\cpp\jit\test_lite_interpreter_direct.cpp`

```
// 包含测试所需的头文件
#include <test/cpp/jit/test_utils.h>

// 包含 Google Test 框架的头文件
#include <gtest/gtest.h>

// 包含 PyTorch 的核心 TensorOptions 定义
#include <c10/core/TensorOptions.h>

// 包含 PyTorch 自动求导生成的变量工厂函数
#include <torch/csrc/autograd/generated/variable_factories.h>

// 包含 PyTorch JIT 的模块定义
#include <torch/csrc/jit/api/module.h>

// 包含 PyTorch JIT 前端解析器相关的头文件
#include <torch/csrc/jit/frontend/resolver.h>

// 包含 PyTorch 移动端兼容性后向兼容处理函数
#include <torch/csrc/jit/mobile/compatibility/backport.h>

// 包含 PyTorch 移动端兼容性后向兼容管理器
#include <torch/csrc/jit/mobile/compatibility/backport_manager.h>

// 包含 PyTorch 移动端兼容性模型兼容性处理
#include <torch/csrc/jit/mobile/compatibility/model_compatibility.h>

// 包含 PyTorch 移动端兼容性运行时兼容性处理
#include <torch/csrc/jit/mobile/compatibility/runtime_compatibility.h>

// 包含 PyTorch 移动端导入相关函数
#include <torch/csrc/jit/mobile/import.h>

// 包含 PyTorch 移动端解释器
#include <torch/csrc/jit/mobile/interpreter.h>

// 包含 PyTorch 移动端模块定义
#include <torch/csrc/jit/mobile/module.h>

// 包含 PyTorch 移动端解析字节码相关函数
#include <torch/csrc/jit/mobile/parse_bytecode.h>

// 包含 PyTorch 移动端解析操作符相关函数
#include <torch/csrc/jit/mobile/parse_operators.h>

// 包含 PyTorch JIT 序列化导出功能
#include <torch/csrc/jit/serialization/export.h>

// 包含 PyTorch JIT 序列化导出字节码功能
#include <torch/csrc/jit/serialization/export_bytecode.h>

// 包含 PyTorch JIT 序列化导入功能
#include <torch/csrc/jit/serialization/import.h>

// 包含自定义类的头文件
#include <torch/custom_class.h>

// 包含 PyTorch 的主头文件
#include <torch/torch.h>

// 包含无序集合的标准头文件
#include <unordered_set>

// 将测试代码放入 torch::jit 命名空间
namespace torch {
namespace jit {

// 测试用例，测试 torch::upsample_nearest2d 函数
TEST(LiteInterpreterDirectTest, UpsampleNearest2d) {
  // 创建名为 m 的模块
  Module m("m");

  // 定义 m 模块的前向传播函数，使用 torch.upsample_nearest2d 函数
  m.define(R"(
    def forward(self, input: Tensor, scale:float):
      return torch.upsample_nearest2d(input, [1, 1], float(scale), float(scale))
  )");

  // 准备输入数据
  std::vector<IValue> inputs;
  inputs.emplace_back(torch::rand({1, 3, 128, 128}));
  inputs.emplace_back(at::Scalar(2.0));

  // 对模块进行前向传播，获取参考结果
  auto ref = m.forward(inputs);

  // 设置编译选项
  CompilationOptions options;

  // 将 JIT 模块转换为移动端模块
  mobile::Module bc = jitModuleToMobile(m, options);

  // 进行移动端模块的前向传播
  IValue res;
  res = bc.forward(inputs);

  // 将结果转换为 Tensor
  auto resd = res.toTensor();
  auto refd = ref.toTensor();

  // 断言结果一致
  ASSERT_TRUE(resd.equal(refd));
}

// 测试用例，测试模块属性访问
TEST(LiteInterpreterDirectTest, CheckAttrAccess) {
  // 创建名为 m 的模块
  Module m("m");

  // 注册名为 "mobile_optimized" 的布尔类型属性
  m.register_attribute("mobile_optimized", BoolType::get(), true);

  // 设置编译选项
  CompilationOptions options;

  // 将 JIT 模块转换为移动端模块
  mobile::Module bc = jitModuleToMobile(m, options);

  // 获取移动端模块中的 "mobile_optimized" 属性值
  bool mobile_optimized = bc.attr("mobile_optimized", false).toBool();

  // 断言属性值为真
  AT_ASSERT(mobile_optimized);

  // 修改模块的 "mobile_optimized" 属性值为假
  m.setattr("mobile_optimized", false);

  // 再次将 JIT 模块转换为移动端模块
  bc = jitModuleToMobile(m, options);

  // 获取修改后的 "mobile_optimized" 属性值
  mobile_optimized = bc.attr("mobile_optimized", false).toBool();

  // 断言属性值为假
  AT_ASSERT(!mobile_optimized);
}

// 测试用例，测试方法调用
TEST(
    LiteInterpreterDirectTest,
    MethodInvocation) { // NOLINT (use =delete in gtest)
  const std::vector<std::string> test_programs{
      // 测试调用带有默认参数的方法
      R"(
      def test_func(self, x, b : int = 4):
        return self.foo + x + b
      )",
      // 内部调用带有默认参数的方法（被内联）
      R"(
      def add_with_default_arg(self, x, b : int = 4):
        return self.foo + x + b
      def test_func(self, x):
        return self.add_with_default_arg(x)  # 调用带有默认参数的方法
      )",
      // 简单的方法调用
      R"(
      def test_func(self, x):
        b = 4
        return self.foo + x + b
      )",
  };

  // 循环测试不同的测试程序
  for (const auto& test_program : test_programs) {
    // 创建名为 m 的模块
    Module m("m");

    // 注册名为 "foo" 的参数，使用 torch::ones 初始化
    m.register_parameter("foo", torch::ones({}), false);
    // 定义模型并加载测试程序
    m.define(test_program);

    // 声明一个常量 fortyTwo 并初始化为 42，用于使代码检查工具保持满意状态
    const int fortyTwo = 42; // (keep linter happy)

    // 创建一个张量 minput，其元素值均为 42
    auto minput = fortyTwo * torch::ones({});

    // 调用模型的方法 "test_func" 并传入 minput，获取返回结果 ref
    auto ref = m.run_method("test_func", minput);

    // 配置编译选项
    CompilationOptions options;

    // 将 JIT 模块转换为移动端模块
    mobile::Module bc = jitModuleToMobile(m, options);

    // 获取移动端模块中的方法 "test_func"
    const auto& test_func = bc.get_method("test_func");

    // 输出调试信息 "hello"
    std::cerr << "hello " << std::endl;

    // 定义 IValue 类型的变量 res
    IValue res;

    // 多次调用 test_func 方法，每次传入 minput，并将结果赋给 res
    for (int i = 0; i < 3; ++i) {
      res = test_func({minput});
    }

    // 输出调试信息 "hello 3"
    std::cerr << "hello 3" << std::endl;

    // 将 res 转换为 float 类型的张量，并提取其单个元素值
    auto resd = res.toTensor().item<float>();

    // 将 ref 转换为 float 类型的张量，并提取其单个元素值
    auto refd = ref.toTensor().item<float>();

    // 断言 resd 和 refd 的值相等
    AT_ASSERT(resd == refd);
}
}

// 定义一个测试用例，测试 LiteInterpreterDirectTest 类的 Conv 方法
TEST(LiteInterpreterDirectTest, Conv) {
  // 获取环境变量 PYTORCH_TEST_WITH_TSAN 的值
  auto s = std::getenv("PYTORCH_TEST_WITH_TSAN");
  // 如果环境变量存在且其值为 "1"，则直接返回，不进行测试
  if (s && strcmp(s, "1") == 0)
    return;

  // 创建输入数据向量
  std::vector<torch::jit::IValue> inputs;

  // 创建名为 "m" 的 Module 对象
  Module m("m");
  // 注册模型参数 "weight" 和 "bias" 到 Module 对象 m 中
  m.register_parameter("weight", torch::ones({20, 1, 5, 5}), false);
  m.register_parameter("bias", torch::ones({20}), false);
  // 定义 Module 对象 m 的前向传播函数，实现卷积操作
  m.define(R"(
    def forward(self, input):
      return torch._convolution(input, self.weight, self.bias, [1, 1], [0, 0], [1, 1], False, [0, 0], 1, False, False, True, True)
  )");

  // 向输入数据向量中添加数据，此处添加大小为 [1, 1, 28, 28] 的张量
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers,modernize-use-emplace)
  inputs.push_back(torch::ones({1, 1, 28, 28}));

  // 调用 Module 对象 m 的前向传播方法 forward，并获取输出张量 outputref
  auto outputref = m.forward(inputs).toTensor();

  // 创建编译选项对象
  CompilationOptions options;
  // 将 Module 对象 m 转换为移动端的模块 bc
  mobile::Module bc = jitModuleToMobile(m, options);
  // 创建 IValue 对象 res
  IValue res;
  // 多次调用移动端模块 bc 的 forward 方法，并将结果赋给 res
  for (int i = 0; i < 3; ++i) {
    res = bc.get_method("forward")(inputs);
  }
  // 将 res 转换为张量 output
  auto output = res.toTensor();
  // 断言 outputref 的维度与 output 的维度相同
  AT_ASSERT(outputref.dim() == output.dim());
  // 断言 outputref 和 output 的第一个元素的值相同
  AT_ASSERT(
      outputref[0][0][0][0].item<int>() == output[0][0][0][0].item<int>());
}

// 定义一个测试用例，测试 LiteInterpreterDirectTest 类的 Inline 方法
TEST(LiteInterpreterDirectTest, Inline) {
  // 创建名为 "m" 的 Module 对象
  Module m("m");
  // 定义 Module 对象 m 的多个函数 foo1、foo2、foo3
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
  // 将 Module 对象 m 转换为移动端的模块 bc
  mobile::Module bc = jitModuleToMobile(m, options);
  // 创建输入数据向量，添加大小为 [1] 的张量
  std::vector<torch::jit::IValue> inputs({torch::ones({})});
  // 调用移动端模块 bc 的方法 foo3，并获取输出
  auto output = bc.get_method("foo3")(inputs);
  // 断言输出张量的值为 7.0
  AT_ASSERT(output.toTensor().item<float>() == 7.0);
}

// 定义一个测试用例，测试 LiteInterpreterDirectTest 类的 Tuple 方法
TEST(LiteInterpreterDirectTest, Tuple) {
  // 创建名为 "m" 的 Module 对象
  Module m("m");
  // 定义 Module 对象 m 的函数 foo 和 forward
  m.define(R"JIT(
  def foo(self, x):
      return (1, 2, x + 3)

  def forward(self, x):
      tuple = self.foo(x)
      return tuple
  )JIT");
  // 创建编译选项对象
  CompilationOptions options;
  // 将 Module 对象 m 转换为移动端的模块 bc
  mobile::Module bc = jitModuleToMobile(m, options);
  // 创建输入数据向量，添加大小为 [1] 的张量
  std::vector<torch::jit::IValue> inputs({torch::ones({})});
  // 调用移动端模块 bc 的方法 forward，并获取输出
  auto output = bc.get_method("forward")(inputs);
  // 断言输出元组中第二个元素的值为 2
  AT_ASSERT(output.toTupleRef().elements()[1].toInt() == 2);
}

// 定义一个测试用例，测试 LiteInterpreterDirectTest 类的 Dict 方法
TEST(LiteInterpreterDirectTest, Dict) {
  // 创建名为 "m" 的 Module 对象
  Module m("m");
  // 定义 Module 对象 m 的函数 foo 和 forward
  m.define(R"JIT(
  def foo(self, x):
      return {"result": x + 1}

  def forward(self, x):
      d = self.foo(x)
      return d
  )JIT");
  // 创建编译选项对象
  CompilationOptions options;
  // 将 Module 对象 m 转换为移动端的模块 bc
  mobile::Module bc = jitModuleToMobile(m, options);
  // 创建输入数据向量，添加大小为 [1] 的张量
  std::vector<torch::jit::IValue> inputs({torch::ones({})});
  // 调用移动端模块 bc 的方法 forward，并获取输出
  auto output = bc.get_method("forward")(inputs);
  // 断言输出字典中键 "result" 对应的张量的整数值为 2
  AT_ASSERT(output.toGenericDict().at("result").toTensor().item().toInt() == 2);
}

// 定义一个测试用例，测试 LiteInterpreterDirectTest 类的 Prim 方法
TEST(LiteInterpreterDirectTest, Prim) {
  // 创建名为 "m" 的 Module 对象
  Module m("m");
  // 定义 Module 对象 m 的函数 forward，将输入值转换为整数
  m.define(R"JIT(
        def forward(self, x):
            return int(x)
  )JIT");

  // 创建输入数据向量，添加一个大小为 [] 的张量，值为 3.5
  std::vector<IValue> inputs;
  auto minput = 3.5 * torch::ones({});
  inputs.emplace_back(minput);
  // 运行 Module 对象 m 的方法 forward，获取参考值 ref
  auto ref = m.run_method("forward", minput);

  // 创建编译选项对象
  CompilationOptions options;
  // 将 Module 对象 m 转换为移动端的模块 bc
  mobile::Module bc = jitModuleToMobile(m, options);

  // 创建 IValue 对象 res
  IValue res;
  // 多次调用移动端模块 bc 的 forward 方法，并将结果赋给 res
  for (int i = 0; i < 3; ++i) {
    // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
    auto bcinputs = inputs;
    # 使用bc对象调用其方法"forward"，并传入bcinputs作为参数，得到结果存储在res中
    res = bc.get_method("forward")(bcinputs);
  }

  # 将res转换为整数类型并存储在resi中
  auto resi = res.toInt();
  # 将ref转换为整数类型并存储在refi中
  auto refi = ref.toInt();
  # 断言resi与refi相等，如果不相等则抛出异常
  AT_ASSERT(resi == refi);
}

// 定义测试用例 LiteInterpreterDirectTest.PrimScalar，验证主要标量运算功能
TEST(LiteInterpreterDirectTest, PrimScalar) {
  // 创建名为 "m" 的模块对象
  Module m("m");
  // 定义模块的前向方法，将输入 x 转换为整数并返回
  m.define(R"JIT(
        def forward(self, x):
            return int(x.item())
  )JIT");

  // 准备输入数据
  std::vector<IValue> inputs;
  auto minput = 3.5 * torch::ones({});
  inputs.emplace_back(minput);
  // 在模块 m 上运行 forward 方法，保存结果为 ref
  auto ref = m.run_method("forward", minput);

  // 配置编译选项
  CompilationOptions options;
  // 将 JIT 模块 m 转换为移动端模块 bc
  mobile::Module bc = jitModuleToMobile(m, options);
  IValue res;
  // 循环运行 forward 方法 3 次
  for (int i = 0; i < 3; ++i) {
    // 复制 inputs 到 bcinputs
    auto bcinputs = inputs;
    // 在移动端模块 bc 上运行 forward 方法，保存结果为 res
    res = bc.get_method("forward")(bcinputs);
  }

  // 将 res 和 ref 转换为整数并断言它们相等
  auto resi = res.toInt();
  auto refi = ref.toInt();
  AT_ASSERT(resi == refi);
}

// 定义测试用例 LiteInterpreterDirectTest.WrongMethodName，验证错误方法名的处理
TEST(LiteInterpreterDirectTest, WrongMethodName) {
  // 创建名为 "m" 的模块对象
  Module m("m");
  // 向模块注册参数 "foo"，值为 torch.ones({})，不需要梯度
  m.register_parameter("foo", torch::ones({}), false);
  // 定义模块的方法 add，实现对参数 foo 和输入 x 的加法操作
  m.define(R"(
    def add(self, x):
      b = 4
      return self.foo + x + b
  )");
  // 配置编译选项
  CompilationOptions options;
  // 将 JIT 模块 m 转换为移动端模块 bc
  mobile::Module bc = jitModuleToMobile(m, options);
  std::vector<IValue> inputs;
  auto minput = 5 * torch::ones({});
  inputs.emplace_back(minput);
  // 断言在移动端模块 bc 上运行 forward 方法时抛出异常，异常信息包含 "is not defined"
  ASSERT_THROWS_WITH_MESSAGE(
      bc.get_method("forward")(inputs), "is not defined");
}

// 定义测试用例 LiteInterpreterDirectTest.SetState，验证状态设置和保存功能
TEST(LiteInterpreterDirectTest, SetState) {
  // 创建名为 "m" 的模块对象
  Module m("m");
  // 向模块注册参数 "foo"，值为 torch.ones({})，不需要梯度
  m.register_parameter("foo", torch::ones({}), false);
  // 定义模块的特殊方法 __getstate__ 和 __setstate__，用于状态保存和加载
  // 同时定义模块的前向方法，实现对参数 foo 和输入 x 的加法操作
  m.define(R"(
    def __getstate__(self):
      return self.foo
    def __setstate__(self, a):
      self.foo = a
    def forward(self, x):
      b = 4
      return self.foo + x + b
  )");

  std::vector<IValue> inputs;
  auto minput = 5 * torch::ones({});
  inputs.emplace_back(minput);

  // 将模块 m 保存到 stringstream ms 中
  std::stringstream ms;
  m.save(ms);
  // 从 stringstream ms 中加载模块，并保存为 loaded_m
  auto loaded_m = load(ms);
  // 在 loaded_m 上运行 forward 方法，保存结果为 ref
  auto ref = loaded_m.run_method("forward", minput);

  // 配置编译选项
  CompilationOptions options;
  // 将 JIT 模块 m 转换为移动端模块 bc
  mobile::Module bc = jitModuleToMobile(m, options);
  IValue res;
  // 循环运行 forward 方法 3 次
  for (int i = 0; i < 3; ++i) {
    // 复制 inputs 到 bcinputs
    auto bcinputs = inputs;
    // 在移动端模块 bc 上运行 forward 方法，保存结果为 res
    res = bc.get_method("forward")(bcinputs);
  }

  // 将 res 和 ref 转换为 float 后，断言它们相等
  auto resd = res.toTensor().item<float>();
  auto refd = ref.toTensor().item<float>();
  AT_ASSERT(resd == refd);
}

// 定义 TorchBindLiteInterpreterDirectTestStruct 类，用于测试自定义类的持有和输出
class TorchBindLiteInterpreterDirectTestStruct
    : public torch::jit::CustomClassHolder {
 public:
  // 实现自定义方法 get，返回包含张量元素个数信息的字符串
  std::string get(at::Tensor t) {
    std::stringstream ss;
    ss << "Hello! Your tensor has ";
    ss << t.numel();
    ss << " elements!";
    return ss.str();
  }
};

// 定义 ClassNamespaceValue 结构体，实现 SugaredValue 接口用于处理命名空间和自定义类
namespace {
struct ClassNamespaceValue : public SugaredValue {
  explicit ClassNamespaceValue(c10::QualifiedName name)
      : basename_(std::move(name)) {}

  // 实现 attr 方法，处理命名空间中的属性访问和自定义类
  std::shared_ptr<SugaredValue> attr(
      const SourceRange&,
      GraphFunction&,
      const std::string& name) override {
    const auto fullName = c10::QualifiedName(basename_, name);

    // 检查是否为自定义类
    if (auto custom_class = getCustomClass(fullName.qualifiedName())) {
      return std::make_shared<ClassValue>(custom_class);
    }

    // 如果不是自定义类，假设它是另一个命名空间
    // NOLINTNEXTLINE(performance-move-const-arg)
    // 返回一个指向 ClassNamespaceValue 类的 shared_ptr，该对象以 fullName 为参数构造
    return std::make_shared<ClassNamespaceValue>(fullName);
  }

  // 返回字符串 "Class Namespace"，表示该对象的种类
  std::string kind() const override {
    return "Class Namespace";
  }

 private:
  // 私有成员变量，用于存储类的基本名称
  c10::QualifiedName basename_;
};

// TestModuleResolver 是 Resolver 的子类，用于解析名称和类型
struct TestModuleResolver : public Resolver {
  // 解析值的函数，根据名称返回相应的 SugaredValue
  std::shared_ptr<SugaredValue> resolveValue(
      const std::string& name,        // 解析的名称
      GraphFunction&,                 // 图函数对象
      const SourceRange&) override {  // 源范围对象
    if (name == "torch") {            // 如果名称是 "torch"
      return std::make_shared<BuiltinModule>("aten");  // 返回内置模块 "aten"
    } else if (name == "__torch__") {  // 如果名称是 "__torch__"
      return std::make_shared<ClassNamespaceValue>(c10::QualifiedName(name));  // 返回类命名空间值
    }

    return nullptr;  // 否则返回空指针
  }

  // 解析类型的函数，根据名称返回类型指针
  TypePtr resolveType(const std::string&, const SourceRange&) override {
    return nullptr;  // 返回空指针
  }
};  // 结构体结束

} // namespace  // 命名空间结束

// LiteInterpreterDirectTest 测试案例，测试内置函数
TEST(LiteInterpreterDirectTest, BuiltinFunction) {
  script::Module m("m");  // 创建脚本模块 m
  auto custom_class_obj = make_custom_class<TorchBindLiteInterpreterDirectTestStruct>();  // 创建自定义类对象
  m.register_attribute("my_obj", custom_class_obj.type(), custom_class_obj);  // 注册属性 my_obj 到模块 m

  // 定义 forward 方法，返回字符串类型
  m.define(R"(
    def forward(self, x) -> str:
      return self.my_obj.get(x)
  )");

  CompilationOptions options;  // 编译选项对象
  mobile::Module bc = jitModuleToMobile(m, options);  // 将 JIT 模块转换为移动模块
  auto res = bc.get_method("forward")(std::vector<IValue>{torch::zeros({3, 4})});  // 调用 forward 方法并传入参数
  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
  auto str = res.toStringRef();  // 将结果转换为字符串引用
  std::string expected = "Hello! Your tensor has 12 elements!";  // 预期的字符串结果
  AT_ASSERT(str == expected);  // 断言结果与预期相同
}

#if !defined FB_XPLAT_BUILD
// 测试获取运行时字节码版本
TEST(LiteInterpreterDirectTest, GetRuntimeByteCodeVersion) {
  auto runtime_bytecode_version = _get_runtime_bytecode_version();  // 获取运行时字节码版本
  AT_ASSERT(
      runtime_bytecode_version ==
      caffe2::serialize::kMaxSupportedBytecodeVersion);  // 断言版本与最大支持版本相同
}

// 测试获取运行时操作符版本
TEST(LiteInterpreterDirectTest, GetRuntimeOperatorsVersion) {
  auto runtime_operators_version = _get_runtime_operators_min_max_versions();  // 获取运行时操作符版本范围
  AT_ASSERT(
      runtime_operators_version.first ==
          caffe2::serialize::kMinSupportedFileFormatVersion &&  // 断言最小支持文件格式版本
      runtime_operators_version.second ==
          caffe2::serialize::kMaxSupportedFileFormatVersion);  // 断言最大支持文件格式版本
}

/**
 * 由于 BUCK 要求我们将 script_module_v4.ptl 文件作为资源依赖项传递给
 * 构建规则的 FB 内部 xplat 构建，因此我们需要通过 C++ 资源 API 访问它，
 * 而不是直接从磁盘读取（这是开源构建/运行所做的）。
 */
// 测试获取字节码版本
TEST(LiteInterpreterDirectTest, GetByteCodeVersion) {
  std::string filePath(__FILE__);  // 获取当前文件路径
  auto test_model_file_v4 =
      filePath.substr(0, filePath.find_last_of("/\\") + 1);  // 获取 script_module_v4.ptl 文件路径
  test_model_file_v4.append("script_module_v4.ptl");  // 拼接文件名

  auto version_v4 = _get_model_bytecode_version(test_model_file_v4);  // 获取模型字节码版本
  AT_ASSERT(version_v4 == 4);  // 断言版本为 4
}

#endif // !defined(FB_XPLAT_BUILD)

// 测试获取运行时操作和信息
TEST(LiteInterpreterDirectTest, GetRuntimeOpsAndInfo) {
  auto runtime_ops = _get_runtime_ops_and_info();  // 获取运行时操作和信息
  // 估算最小操作数的数量，用于验证 API 返回一个合理大的数目
  AT_ASSERT(runtime_ops.size() > 2900);  // 断言操作数数量大于 2900
}

// Eval 测试案例
TEST(LiteInterpreterDirectTest, Eval) {
  std::vector<torch::jit::IValue> inputs;  // 输入向量

  Module m("m");  // 创建脚本模块 m
  // 定义脚本
  m.define(R"(
    # 初始化方法，设置对象的训练状态为 True
    def __init__(self, x):
      self.training = True

    # 前向传播方法，应用输入数据的随机丢弃（dropout），丢弃率为 1.0（即不丢弃），根据训练状态决定是否执行丢弃操作
    def forward(self, input):
      return torch.dropout(input, 1.0, self.training)
  )");

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers,modernize-use-emplace)
  // 向输入列表中添加一个张量，大小为 [1, 1, 28, 28]
  inputs.push_back(torch::ones({1, 1, 28, 28}));
  // 将模型切换为评估模式
  m.eval();
  // 执行模型的前向传播，并将输出转换为张量
  auto outputref = m.forward(inputs).toTensor();

  // 将模型设置为训练模式，以确保移动端的 eval() 正确切换回评估模式
  m.train();
  // 配置编译选项
  CompilationOptions options;
  // 将 JIT 模块转换为移动端模块
  mobile::Module bc = jitModuleToMobile(m, options);
  // 将移动端模块设置为评估模式
  bc.eval();
  // 定义结果的 IValue 对象
  IValue res;
  // 多次调用模型的前向方法，共计 3 次
  for (int i = 0; i < 3; ++i) {
    res = bc.get_method("forward")(inputs);
  }
  // 将结果转换为张量
  auto output = res.toTensor();
  // 断言输出张量的维度与参考输出的维度相同
  AT_ASSERT(outputref.dim() == output.dim());
  // 断言输出张量的第一个元素是否相同
  AT_ASSERT(
      outputref[0][0][0][0].item<int>() == output[0][0][0][0].item<int>());
}

TEST(LiteInterpreterDirectTest, FindWrongMethodName) {
  // 创建名为 "m" 的模块对象
  Module m("m");
  // 向模块注册名为 "foo" 的参数，形状为标量（scalar）的张量，不需要梯度
  m.register_parameter("foo", torch::ones({}), false);
  // 定义一个方法 "add"，接受参数 self 和 x
  m.define(R"(
    def add(self, x):
      b = 4
      return self.foo + x + b
  )");
  // 创建编译选项对象
  CompilationOptions options;
  // 将 JIT 编译的模块转换为移动端模块
  mobile::Module bc = jitModuleToMobile(m, options);
  // 断言查找名为 "forward" 的方法返回空
  ASSERT_TRUE(bc.find_method("forward") == c10::nullopt);
}

TEST(LiteInterpreterDirectTest, FindAndRunMethod) {
  // 创建名为 "m" 的模块对象
  Module m("m");
  // 向模块注册名为 "foo" 的参数，形状为标量的张量，不需要梯度
  m.register_parameter("foo", torch::ones({}), false);
  // 定义一个方法 "add_it"，接受参数 self 和 x
  m.define(R"(
    def add_it(self, x):
      b = 4
      return self.foo + x + b
  )");

  // 创建输入向量并初始化
  std::vector<IValue> inputs;
  auto minput = 5 * torch::ones({});
  inputs.emplace_back(minput);
  // 获取方法 "add_it" 并调用
  auto ref = m.get_method("add_it")(inputs);

  // 创建编译选项对象
  CompilationOptions options;
  // 将 JIT 编译的模块转换为移动端模块
  mobile::Module bc = jitModuleToMobile(m, options);
  // 定义变量 res，用于存储方法运行结果
  IValue res;
  // 循环执行方法 "add_it" 三次
  for (int i = 0; i < 3; ++i) {
    auto bcinputs = inputs;
    // 查找名为 "add_it" 的方法，并断言不为空
    auto method = bc.find_method("add_it");
    AT_ASSERT(method != c10::nullopt);
    // 调用找到的方法并存储结果到 res
    res = (*method)(std::move(bcinputs));
  }

  // 将 res 和 ref 转换为 float 类型的标量，并进行断言比较
  auto resd = res.toTensor().item<float>();
  auto refd = ref.toTensor().item<float>();
  AT_ASSERT(resd == refd);
}

TEST(LiteInterpreterDirectTest, RunMethodVariadic) {
  // 创建名为 "m" 的模块对象
  Module m("m");
  // 向模块注册名为 "foo" 的参数，形状为标量的张量，不需要梯度
  m.register_parameter("foo", torch::ones({}), false);
  // 定义一个方法 "add_three"，接受参数 self、x 和 y
  m.define(R"(
    def add_three(self, x, y):
      return self.foo + x + y
  )");

  // 创建输入向量并初始化
  std::vector<IValue> inputs;
  auto inputx = 5 * torch::ones({});
  auto inputy = 4 * torch::ones({});
  // 使用方法 "run_method" 执行方法 "add_three" 并存储结果到 ref
  auto ref = m.run_method("add_three", inputx, inputy);

  // 创建编译选项对象
  CompilationOptions options;
  // 将 JIT 编译的模块转换为移动端模块
  mobile::Module bc = jitModuleToMobile(m, options);
  // 使用移动端模块执行方法 "add_three" 并存储结果到 res
  IValue res = bc.run_method("add_three", inputx, inputy);

  // 将 res 和 ref 转换为 float 类型的标量，并进行断言比较
  auto resd = res.toTensor().item<float>();
  auto refd = ref.toTensor().item<float>();
  AT_ASSERT(resd == refd);
}

TEST(LiteInterpreterDirectTest, DuplicateSetState) {
  // 创建名为 "M" 的模块对象
  Module m("M");
  // 向模块注册名为 "foo" 的参数，形状为标量的张量，不需要梯度
  m.register_parameter("foo", torch::ones({}), false);
  // 定义方法 "__getstate__" 返回两次 self.foo 的和
  // 定义方法 "__setstate__" 将参数 a 赋值给 self.foo
  // 定义方法 "forward"，接受参数 self 和 x
  m.define(R"(
    def __getstate__(self):
      return self.foo + self.foo
    def __setstate__(self, a):
      self.foo = a
    def forward(self, x):
      b = 4
      return self.foo + x + b
  )");

  // 创建名为 "B" 的模块对象
  Module b("B");
  // 注册两个名为 "M0" 和 "M1" 的模块实例到模块 "B"
  b.register_module("M0", m);
  b.register_module("M1", m);
  // 定义方法 "forward"，接受参数 self 和 x
  b.define(R"(
    def forward(self, x):
      return self.M0.forward(x) + self.M1.forward(x)
  )");

  // 创建编译选项对象
  CompilationOptions options;
  // 将 JIT 编译的模块转换为移动端模块
  mobile::Module bc = jitModuleToMobile(m, options);
  // 获取所有方法并断言其数量为 3
  const auto methods = bc.get_methods();
  const size_t expected_n = 3;
  ASSERT_EQ(methods.size(), expected_n);
}

TEST(LiteInterpreterDirectTest, OpNameExportFetchRootOperators) {
  // 创建名为 "m" 的模块对象
  torch::jit::Module m("m");
  // 向模块注册名为 "weight" 和 "bias" 的参数张量，具有指定的形状和初始值，不需要梯度
  m.register_parameter("weight", torch::ones({20, 1, 5, 5}), false);
  m.register_parameter("bias", torch::ones({20}), false);
  // 定义模块方法
  m.define(R"(
    # 定义一个名为 forward 的方法，用于执行神经网络的前向传播
    def forward(self, input):
      # 创建一个大小为 2x2 的零张量 x1
      x1 = torch.zeros(2, 2)
      # 使用与给定张量 input 相同的大小和数据类型创建一个空张量 x2
      x2 = torch.empty_like(torch.empty(2, 2))
      # 调用底层的卷积函数 _convolution，对输入 input 进行卷积操作，使用模型的权重和偏置
      x3 = torch._convolution(input, self.weight, self.bias, [1, 1], [0, 0], [1, 1], False, [0, 0], 1, False, False, True, True)
      # 返回三个张量 x1, x2, x3
      return (x1, x2, x3)
  )");
  # 将模型设置为评估模式
  m.eval();

  # 定义编译选项
  CompilationOptions options;
  # 将 JIT 模块转换为移动端模型
  mobile::Module ptl_model = jitModuleToMobile(m, options);
  # 获取导出的操作符名称列表
  std::set<std::string> operator_names =
      torch::jit::mobile::_export_operator_list(ptl_model);
  # 定义预期的操作符名称集合
  std::set<std::string> expected_operator_names = {
      "aten::_convolution",
      "aten::empty.memory_format",
      "aten::empty_like",
      "aten::zeros",
  };
  # 断言实际导出的操作符名称与预期的操作符名称集合相等
  EXPECT_EQ(operator_names, expected_operator_names)
      << "Expected the root operator lists to be the same";
// 命名空间开始
namespace {

// 测试函数：比较轻量级模块的结果张量
void testLiteModuleCompareResultTensors(
    Module& m, // 模块引用，用于执行前向传播
    const std::vector<torch::jit::IValue>& inputs, // 输入数据的向量
    const std::string& method_name = "forward") { // 方法名，默认为"forward"

  // 获取参考输出张量
  auto outputref = m.get_method(method_name)(inputs).toTensor();

  // 配置编译选项
  CompilationOptions options;
  // 将 JIT 模块转换为移动模块
  mobile::Module bc = jitModuleToMobile(m, options);
  IValue res;
  // 执行前向传播方法多次，获取结果
  for (int i = 0; i < 3; ++i) {
    res = bc.get_method(method_name)(inputs);
  }
  // 转换结果为张量
  auto output = res.toTensor();
  // 断言：输出张量的维度与参考输出张量的维度相同
  AT_ASSERT(outputref.dim() == output.dim());
  // 断言：输出张量与参考输出张量相等
  AT_ASSERT(output.equal(outputref));
}

// 函数：测试带默认参数的 torch.linalg_pinv 函数
void testDefaultArgsPinv2(int num_args) {
  Module m("m");
  // 根据参数个数定义前向传播函数不同版本
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

  // 准备输入数据向量
  std::vector<torch::jit::IValue> inputs;
  const int N = 28;
  auto input = torch::range(1, N * N, 1);
  input[0] = 1; // 一个更稳定的矩阵
  input = input.view({N, N});
  inputs.emplace_back(input);

  // 测试函数：比较轻量级模块的结果张量
  testLiteModuleCompareResultTensors(m, inputs);
}

} // 命名空间结束
  testDefaultArgsPinv2(num_args);


# 调用函数 testDefaultArgsPinv2，并传入参数 num_args
testDefaultArgsPinv2(num_args);



  //  bytecode with one specified argument:
  //  (6,
  //      ('__torch__.m.forward',
  //          (('instructions',
  //              (('STOREN', 1, 2),
  //                  ('DROPR', 1, 0),
  //                  ('MOVE', 2, 0),
  //                  ('OP', 0, 0),
  //                  ('RET', 0, 0))),
  //              ('operators', (('aten::linalg_pinv', '', 1),)),
  //              ('constants', (False, 1e-15)), # default constants are not
  //              used
  //              ('types', ()),
  //              ('register_size', 2)),
  //          (('arguments',
  //              ((('name', 'self'), ('type', '__torch__.m'), ('default_value',
  //              None)),
  //                  (('name', 'input'), ('type', 'Tensor'), ('default_value',
  //                  None)))),
  //              ('returns',
  //                  ((('name', ''), ('type', 'Tensor'), ('default_value',
  //                  None)),)))))


# 字节码示例，使用了一个指定的参数:
# (6,
#     ('__torch__.m.forward',
#         (('instructions',
#             (('STOREN', 1, 2),
#                 ('DROPR', 1, 0),
#                 ('MOVE', 2, 0),
#                 ('OP', 0, 0),
#                 ('RET', 0, 0))),
#             ('operators', (('aten::linalg_pinv', '', 1),)),
#             ('constants', (False, 1e-15)), # 默认常量未使用
#             ('types', ()),
#             ('register_size', 2)),
#         (('arguments',
#             ((('name', 'self'), ('type', '__torch__.m'), ('default_value',
#             None)),
#                 (('name', 'input'), ('type', 'Tensor'), ('default_value',
#                 None)))),
#             ('returns',
#                 ((('name', ''), ('type', 'Tensor'), ('default_value',
#                 None)),)))))



  //  bytecode with 2 specified argument:
  //  (6,
  //      ('__torch__.m.forward',
  //          (('instructions',
  //              (('STOREN', 1, 2),
  //                  ('DROPR', 1, 0),
  //                  ('MOVE', 2, 0),
  //                  ('LOADC', 1, 0), # added LOADC for specified argument
  //                  ('OP', 0, 0),
  //                  ('RET', 0, 0))),
  //              ('operators', (('aten::linalg_pinv', '', 2),)),
  //              ('constants', (False, 1e-05)), # updated constant table
  //              ('types', ()),
  //              ('register_size', 2)),
  //          (('arguments',
  //              ((('name', 'self'), ('type', '__torch__.m'), ('default_value',
  //              None)),
  //                  (('name', 'input'), ('type', 'Tensor'), ('default_value',
  //                  None)))),
  //              ('returns',
  //                  ((('name', ''), ('type', 'Tensor'), ('default_value',
  //                  None)),)))))


# 字节码示例，使用了两个指定的参数:
# (6,
#     ('__torch__.m.forward',
#         (('instructions',
#             (('STOREN', 1, 2),
#                 ('DROPR', 1, 0),
#                 ('MOVE', 2, 0),
#                 ('LOADC', 1, 0), # 添加了为指定参数而新增的 LOADC
#                 ('OP', 0, 0),
#                 ('RET', 0, 0))),
#             ('operators', (('aten::linalg_pinv', '', 2),)),
#             ('constants', (False, 1e-05)), # 更新的常量表
#             ('types', ()),
#             ('register_size', 2)),
#         (('arguments',
#             ((('name', 'self'), ('type', '__torch__.m'), ('default_value',
#             None)),
#                 (('name', 'input'), ('type', 'Tensor'), ('default_value',
#                 None)))),
#             ('returns',
#                 ((('name', ''), ('type', 'Tensor'), ('default_value',
#                 None)),)))))



  //  bytecode with 3 specified arguments:
  //  (6,
  //      ('__torch__.m.forward',
  //          (('instructions',
  //              (('STOREN', 1, 2),
  //                  ('DROPR', 1, 0),
  //                  ('MOVE', 2, 0),
  //                  ('LOADC', 1, 0),
  //                  ('LOADC', 0, 0),
  //                  ('OP', 0, 0),
  //                  ('RET', 0, 0))),
  //              ('operators', (('aten::linalg_pinv', '', 3),)),
  //              ('constants', (True, 1e-05)),
  //              ('types', ()),
  //              ('register_size', 2)),
  //          (('arguments',
  //              ((('name', 'self'), ('type', '__torch__.m'), ('default_value',
  //              None)),
  //                  (('name', 'input'), ('type', 'Tensor'), ('default_value',
  //                  None)))),
  //              ('returns',
  //                  ((('name', ''), ('type', 'Tensor'), ('default_value',
  //                  None)),)))))


# 字节码示例，使用了三个指定的参数:
# (6,
#     ('__torch__.m.forward',
#         (('instructions',
#             (('STOREN', 1, 2),
#                 ('DROPR', 1, 0),
#                 ('MOVE', 2, 0),
#                 ('LOADC', 1, 0),
#                 ('LOADC', 0, 0),
#                 ('OP', 0, 0),
#                 ('RET', 0, 0))),
#             ('operators', (('aten::linalg_pinv', '', 3),)),
#             ('constants', (True, 1e-05)),
#             ('types', ()),
#             ('register_size', 2)),
#         (('arguments',
#             ((('name', 'self'), ('type', '__torch__.m'), ('default_value',
#             None)),
#                 (('name', 'input'), ('type', 'Tensor'), ('default_value',
#                 None)))),
#             ('returns',
#                 ((('name', ''), ('type', 'Tensor'), ('default_value',
#                 None)),)))))
TEST(LiteInterpreterDirectTest, DefaultArgsTensorinvSpecifyDefault) {
  // 测试默认参数和指定参数 out 的情况下的 torch.linalg_tensorinv 函数
  // 定义一个模块 m
  Module m("m");
  // 在模块 m 中定义 forward 方法，使用 torch.linalg_tensorinv 函数计算
  m.define(R"(
    def forward(self, input):
      return torch.linalg_tensorinv(input, 2)
  )");
  // 将 forward 方法编译成移动端代码
  torch::jit::MobileCode code(m.get_method("forward").graph(), "forward");
  // 获取指定操作符的指定参数数量
  auto arg_nums = code.op_to_num_specified_args();
  // 断言指定操作符的指定参数数量为 1
  ASSERT_EQ(arg_nums.size(), 1);
  // 断言 aten::linalg_tensorinv 操作符的指定参数数量为 1
  ASSERT_EQ(arg_nums["aten::linalg_tensorinv"], 1);
  // 创建输入向量 inputs
  std::vector<torch::jit::IValue> inputs;
  // 定义 N 的值为 4
  const int N = 4;
  // 生成一个随机的 N x N x N x N 的张量 input
  auto input = torch::rand({N, N, N, N});
  // 将 input 加入 inputs 中
  inputs.emplace_back(input);
  // 测试 lite 模块的结果与原始模块的结果是否相同
  testLiteModuleCompareResultTensors(m, inputs);
}

void testDefaultArgsPinvWithOutArg2(int num_args) {
  // 测试 torch.linalg_pinv 函数，默认参数和 out 参数的情况
  // 定义一个模块 m
  Module m("m");
  // 根据传入的 num_args 值选择不同的定义方式
  if (num_args == 1) {
    // 定义 forward 方法，使用 torch.linalg_pinv 函数，out 参数为 input 自身
    m.define(R"(
      def forward(self, input):
        return torch.linalg_pinv(input, out=input)
    )");
  } else if (num_args == 2) {
    // 定义 forward 方法，使用 torch.linalg_pinv 函数，指定第二个参数为 1e-5，out 参数为 input 自身
    m.define(R"(
      def forward(self, input):
        return torch.linalg_pinv(input, 1e-5, out=input)
    )");
  } else if (num_args == 3) {
    // 定义 forward 方法，使用 torch.linalg_pinv 函数，指定第二个参数为 1e-5，第三个参数为 True，out 参数为 input 自身
    m.define(R"(
      def forward(self, input):
        return torch.linalg_pinv(input, 1e-5, True, out=input)
    )");
  }

  // 定义 N 的值为 28
  const int N = 28;
  // 生成一个从 1 到 N*N 的序列，构成一个稳定的 N x N 的张量 input
  auto input = torch::range(1, N * N, 1);
  // 将第一个元素设为 10000，使得 input 变为更稳定的矩阵
  input[0] = 10000;
  // 将 input 重新视图为 N x N 的形状
  input = input.view({N, N});
  // 运行模块的 forward 方法，获取结果 ref
  auto ref = m.run_method("forward", input);
  // 检查 input 是否与原始的 1 到 N*N 的序列不相等
  TORCH_CHECK(!input.equal(torch::range(1, N * N, 1)));
  // 检查 input 是否与 ref 相等
  TORCH_CHECK(input.equal(ref.toTensor()));
}

TEST(LiteInterpreterDirectTest, DefaultArgsPinvWithOutArg) {
  // 测试 torch.linalg_pinv 函数，带有 out 参数的情况
  // 不同数量的指定参数 + out 参数的测试
  // 参数未指定时取默认值
  for (int num_args = 1; num_args <= 3; ++num_args) {
    // 调用 testDefaultArgsPinvWithOutArg2 函数，传入 num_args 作为参数
    testDefaultArgsPinvWithOutArg2(num_args);
  }
}

TEST(LiteInterpreterDirectTest, DefaultArgsWithOutArg) {
  // 测试带有 out 参数的 torch.add 函数
  Module m("m");
  // 定义 forward 方法，使用 torch.add 函数，将结果存入 x 中
  m.define(R"(
    def forward(self, x, h):
      torch.add(x, h, out=x)
  )");

  // 创建输入向量 inputs
  std::vector<IValue> inputs;
  // 初始化 input_x 为 2 的全 1 张量
  auto input_x = 2 * torch::ones({});
  // 初始化 input_h 为全 1 张量
  auto input_h = torch::ones({});
  // 运行模块的 forward 方法，获取结果 ref
  auto ref = m.run_method("forward", input_x, input_h);

  // 编译选项
  CompilationOptions options;
  // 将模块 m 转换为移动端模块 bc
  mobile::Module bc = jitModuleToMobile(m, options);
  // 运行移动端模块的 forward 方法，传入 input_x 和 input_h
  bc.run_method("forward", input_x, input_h);
  // 断言 input_x 是否等于 4 的全 1 张量
  AT_ASSERT(input_x.equal(4 * torch::ones({})));
}

TEST(LiteInterpreterDirectTest, TestExceptionStackWithTwoLevelModuleHierarchy) {
  // 测试两层模块层次结构下的异常堆栈
  // 定义模块 A
  Module a("A");
  // 定义模块 A 的方法 bar
  a.define(R"(
    def bar(self, x, y):
      return x + y
  )");
  // 定义模块 B
  Module b("B");
  // 将模块 A 注册到模块 B 中
  b.register_module("A0", a);
  // 定义模块 B 的方法 foo
  b.define(R"(
    def foo(self, x, y):
      return self.A0.bar(x, y) + 2
  )");
  // 定义模块 C
  Module c("C");
  // 将模块 B 注册到模块 C 中
  c.register_module("B0", b);
  // 定义模块 C 的方法
    # 定义一个类方法 `forward`，接收两个参数 `x` 和 `y`
    def forward(self, x, y):
        # 调用 self.B0 对象的 foo 方法，传入参数 x 和 y，然后将结果加上 3 并返回
        return self.B0.foo(x, y) + 3
  )");

  # 创建一个空的 `std::vector<IValue>` 类型的变量 inputs
  std::vector<IValue> inputs;

  # 向 inputs 中添加一个大小为 [2, 4] 的随机张量
  inputs.emplace_back(torch::rand({2, 4}));

  # 向 inputs 中添加一个大小为 [13, 9] 的随机张量
  inputs.emplace_back(torch::rand({13, 9}));

  # 创建 CompilationOptions 对象 options
  CompilationOptions options;

  # 将 JIT 模块 `c` 转换为移动端模型 `lite_m`，使用指定的编译选项 options
  auto lite_m = jitModuleToMobile(c, options);

  # 定义一个包含错误信息的字符串 error_pattern，使用原始字符串字面量 R"(...) 进行定义
  std::string error_pattern = R"(
  Module hierarchy:top(C)::<unknown>.B0(B)::foo.A0(A)::bar.aten::add
// TorchScript 的追踪信息，显示了发生错误的位置及其上下文
Traceback of TorchScript (most recent call last):
  File "<string>", line 3, in <unknown>
  
  // 定义一个名为 `forward` 的方法，接受两个参数 `x` 和 `y`，并返回调用 `B0` 对象的 `foo` 方法返回值加上 3
    def forward(self, x, y):
      return self.B0.foo(x, y) + 3
             ~~~~~~~~~~~ <--- HERE

  File "<string>", line 3, in foo

  // 定义一个名为 `foo` 的方法，接受两个参数 `x` 和 `y`，并返回调用 `A0` 对象的 `bar` 方法返回值加上 2
    def foo(self, x, y):
      return self.A0.bar(x, y) + 2
             ~~~~~~~~~~~ <--- HERE

  File "<string>", line 3, in bar

  // 定义一个名为 `bar` 的方法，接受两个参数 `x` 和 `y`，并返回它们的和
    def bar(self, x, y):
      return x + y
             ~~~~~ <--- HERE
  )";
  
  // 使用 ASSERT_THROWS_WITH_MESSAGE 检查 lite_m.forward(inputs) 的执行结果是否抛出了特定的错误模式
  ASSERT_THROWS_WITH_MESSAGE(lite_m.forward(inputs), error_pattern);
}
#endif // !defined(FB_XPLAT_BUILD)

// 匿名命名空间，定义了一个名为 `reg` 的静态变量，注册了 TorchScript 类 `_TorchScriptTesting` 的方法和 pickle 支持
namespace {
static auto reg =
    torch::class_<TorchBindLiteInterpreterDirectTestStruct>(
        "_TorchScriptTesting",
        "_LiteInterpreterDirectTest")
        .def(torch::init<>())
        .def("get", &TorchBindLiteInterpreterDirectTestStruct::get)
        .def_pickle(
            // __getattr__
            [](const c10::intrusive_ptr<
                TorchBindLiteInterpreterDirectTestStruct>&) -> int64_t {
              return 0;
            },
            // __setattr__
            [](int64_t) {
              return c10::make_intrusive<
                  TorchBindLiteInterpreterDirectTestStruct>();
            });

} // namespace

// 定义了一个测试案例，验证运算符缓存在不同默认参数下的行为
TEST(LiteInterpreterDirectTest, OperatorCacheDifferentiatesDefaultArgs) {
  // 创建一个名为 `m` 的 TorchScript 模块，包含三个方法 `forward()`、`forward2()`、`forward3()` 分别返回不同的张量类型和值
  Module m("m");
  m.define(R"(
    def forward(self):
      ret1 = torch.new_empty(torch.zeros(10), [10], dtype=4)
      return ret1.fill_(25)
  )");
  m.define(R"(
    def forward2(self):
      ret1 = torch.new_empty(torch.zeros(10), [10], dtype=6)
      return ret1.fill_(32.0)
  )");
  m.define(R"(
    def forward3(self):
      ret1 = torch.new_empty(torch.zeros(10), [10])
      return ret1.fill_(12.0)
  )");

  // 准备输入值为空的测试用例，依次对三个方法进行比较 lite module 结果的测试
  std::vector<torch::jit::IValue> inputs;
  testLiteModuleCompareResultTensors(m, inputs, "forward");
  testLiteModuleCompareResultTensors(m, inputs, "forward2");
  testLiteModuleCompareResultTensors(m, inputs, "forward3");
}

} // namespace jit
} // namespace torch
```