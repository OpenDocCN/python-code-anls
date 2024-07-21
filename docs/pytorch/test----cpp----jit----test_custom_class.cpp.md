# `.\pytorch\test\cpp\jit\test_custom_class.cpp`

```
#include <gtest/gtest.h>  // 包含 Google Test 框架的头文件

#include <test/cpp/jit/test_custom_class_registrations.h>  // 包含自定义类注册测试的头文件
#include <torch/csrc/jit/passes/freeze_module.h>  // 包含冻结模块的头文件
#include <torch/custom_class.h>  // 包含自定义类相关的头文件
#include <torch/script.h>  // 包含 Torch 脚本相关的头文件

#include <iostream>  // 标准输入输出流
#include <string>  // 标准字符串库
#include <vector>  // 标准向量库

namespace torch {
namespace jit {

TEST(CustomClassTest, TorchbindIValueAPI) {  // 定义测试用例 CustomClassTest.TorchbindIValueAPI

  script::Module m("m");  // 创建名为 "m" 的 Torch 脚本模块

  // 测试 make_custom_class API
  auto custom_class_obj = make_custom_class<MyStackClass<std::string>>(
      std::vector<std::string>{"foo", "bar"});  // 创建自定义类对象

  // 定义 Torch 脚本模块方法 forward，接受 __torch__.torch.classes._TorchScriptTesting._StackString 类型参数
  m.define(R"(
    def forward(self, s : __torch__.torch.classes._TorchScriptTesting._StackString):
      return s.pop(), s
  )");

  // 定义测试函数 test_with_obj，接受 IValue 类型参数 obj 和预期字符串 expected
  auto test_with_obj = [&m](IValue obj, std::string expected) {
    auto res = m.run_method("forward", obj);  // 运行模块方法 forward
    auto tup = res.toTuple();  // 将结果转换为元组类型
    AT_ASSERT(tup->elements().size() == 2);  // 断言元组包含两个元素
    auto str = tup->elements()[0].toStringRef();  // 获取元组第一个元素的字符串表示
    auto other_obj =
        tup->elements()[1].toCustomClass<MyStackClass<std::string>>();  // 获取元组第二个元素作为自定义类对象
    AT_ASSERT(str == expected);  // 断言字符串与预期相等
    auto ref_obj = obj.toCustomClass<MyStackClass<std::string>>();  // 获取原始对象的自定义类表示
    AT_ASSERT(other_obj.get() == ref_obj.get());  // 断言其他对象与原始对象相同
  };

  test_with_obj(custom_class_obj, "bar");  // 使用自定义类对象进行测试

  // 测试 IValue() API
  auto my_new_stack = c10::make_intrusive<MyStackClass<std::string>>(
      std::vector<std::string>{"baz", "boo"});  // 创建新的自定义类对象
  auto new_stack_ivalue = c10::IValue(my_new_stack);  // 将自定义类对象包装为 IValue

  test_with_obj(new_stack_ivalue, "boo");  // 使用新的自定义类对象进行测试
}

TEST(CustomClassTest, ScalarTypeClass) {  // 定义测试用例 CustomClassTest.ScalarTypeClass

  script::Module m("m");  // 创建名为 "m" 的 Torch 脚本模块

  // 测试 make_custom_class API
  auto cc = make_custom_class<ScalarTypeClass>(at::kFloat);  // 创建标量类型自定义类对象
  m.register_attribute("s", cc.type(), cc, false);  // 注册模块属性 "s"

  std::ostringstream oss;
  m.save(oss);  // 将模块保存到输出流
  std::istringstream iss(oss.str());  // 从输出流创建输入流
  caffe2::serialize::IStreamAdapter adapter{&iss};  // 创建流适配器
  auto loaded_module = torch::jit::load(iss, torch::kCPU);  // 加载模块到 CPU
}

class TorchBindTestClass : public torch::jit::CustomClassHolder {
 public:
  std::string get() {  // 定义方法 get 返回字符串
    return "Hello, I am your test custom class";
  }
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
constexpr char class_doc_string[] = R"(
  I am docstring for TorchBindTestClass
  Args:
      What is an argument? Oh never mind, I don't take any.

  Return:
      How would I know? I am just a holder of some meaningless test methods.
  )";

// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
constexpr char method_doc_string[] =
    "I am docstring for TorchBindTestClass get_with_docstring method";

namespace {
static auto reg =
    torch::class_<TorchBindTestClass>(
        "_TorchBindTest",
        "_TorchBindTestClass",
        class_doc_string)
        .def("get", &TorchBindTestClass::get)  // 定义方法 "get"
        .def("get_with_docstring", &TorchBindTestClass::get, method_doc_string);  // 定义方法 "get_with_docstring" 带文档字符串

} // namespace

// Tests DocString is properly propagated when defining CustomClasses.
TEST(CustomClassTest, TestDocString) {
  // 获取自定义类的类型对象
  auto class_type = getCustomClass(
      "__torch__.torch.classes._TorchBindTest._TorchBindTestClass");
  // 断言类类型对象非空
  AT_ASSERT(class_type);
  // 断言类的文档字符串与预期相符
  AT_ASSERT(class_type->doc_string() == class_doc_string);

  // 断言类方法 "get" 的文档字符串为空
  AT_ASSERT(class_type->getMethod("get").doc_string().empty());
  // 断言类方法 "get_with_docstring" 的文档字符串与预期相符
  AT_ASSERT(
      class_type->getMethod("get_with_docstring").doc_string() ==
      method_doc_string);
}

TEST(CustomClassTest, Serialization) {
  script::Module m("m");

  // 测试 make_custom_class API
  auto custom_class_obj = make_custom_class<MyStackClass<std::string>>(
      std::vector<std::string>{"foo", "bar"});
  // 注册属性 "s" 到模块，使用自定义类对象
  m.register_attribute(
      "s",
      custom_class_obj.type(),
      custom_class_obj,
      // NOLINTNEXTLINE(bugprone-argument-comment)
      /*is_parameter=*/false);
  // 定义模块的 forward 方法
  m.define(R"(
    def forward(self):
      return self.s.return_a_tuple()
  )");

  // 定义测试函数 test_with_obj
  auto test_with_obj = [](script::Module& mod) {
    // 运行模块的 forward 方法，获取结果
    auto res = mod.run_method("forward");
    // 将结果转换为元组
    auto tup = res.toTuple();
    // 断言元组的元素数量为 2
    AT_ASSERT(tup->elements().size() == 2);
    // 获取元组的第二个元素，并断言其为整数 123
    auto i = tup->elements()[1].toInt();
    AT_ASSERT(i == 123);
  };

  // 冻结模块 m 并测试
  auto frozen_m = torch::jit::freeze_module(m.clone());

  // 分别使用未冻结和冻结的模块进行测试
  test_with_obj(m);
  test_with_obj(frozen_m);

  // 将模块 m 序列化到字符串流 oss
  std::ostringstream oss;
  m.save(oss);
  // 从字符串流 iss 中加载模块
  std::istringstream iss(oss.str());
  caffe2::serialize::IStreamAdapter adapter{&iss};
  auto loaded_module = torch::jit::load(iss, torch::kCPU);

  // 将冻结的模块序列化到字符串流 oss_frozen
  std::ostringstream oss_frozen;
  frozen_m.save(oss_frozen);
  // 从字符串流 iss_frozen 中加载冻结的模块
  std::istringstream iss_frozen(oss_frozen.str());
  caffe2::serialize::IStreamAdapter adapter_frozen{&iss_frozen};
  auto loaded_frozen_module = torch::jit::load(iss_frozen, torch::kCPU);
}

} // namespace jit
} // namespace torch
```