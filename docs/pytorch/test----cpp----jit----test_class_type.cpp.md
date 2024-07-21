# `.\pytorch\test\cpp\jit\test_class_type.cpp`

```py
#include <gtest/gtest.h> // 引入 Google Test 框架的头文件

#include <test/cpp/jit/test_utils.h> // 引入测试工具函数的头文件
#include <torch/csrc/jit/testing/file_check.h> // 引入文件检查函数的头文件
#include <torch/torch.h> // 引入 PyTorch 的头文件

namespace torch {
namespace jit {

TEST(ClassTypeTest, AddRemoveAttr) { // 定义 ClassType 类的单元测试 AddRemoveAttr
  auto cu = std::make_shared<CompilationUnit>(); // 创建共享的编译单元指针 cu
  auto cls = ClassType::create("foo.bar", cu, true); // 创建名为 "foo.bar" 的 ClassType 对象 cls，指定为可变对象
  cls->addAttribute("attr1", TensorType::get(), true); // 向 cls 添加名为 "attr1" 的属性，类型为 TensorType，指定为可变属性
  cls->addAttribute("attr2", TensorType::get()); // 向 cls 添加名为 "attr2" 的属性，类型为 TensorType
  cls->addAttribute("attr3", TensorType::get()); // 向 cls 添加名为 "attr3" 的属性，类型为 TensorType
  ASSERT_TRUE(cls->hasAttribute("attr1")); // 断言 cls 是否包含属性 "attr1"
  ASSERT_TRUE(cls->hasAttribute("attr2")); // 断言 cls 是否包含属性 "attr2"
  ASSERT_TRUE(cls->hasAttribute("attr3")); // 断言 cls 是否包含属性 "attr3"

  // removing attribute attr2
  cls->unsafeRemoveAttribute("attr2"); // 移除属性 "attr2"
  ASSERT_TRUE(cls->hasAttribute("attr1")); // 断言 cls 是否包含属性 "attr1"
  ASSERT_FALSE(cls->hasAttribute("attr2")); // 断言 cls 是否不包含属性 "attr2"
  ASSERT_TRUE(cls->hasAttribute("attr3")); // 断言 cls 是否包含属性 "attr3"

  // removing parameter attr1
  cls->unsafeRemoveAttribute("attr1"); // 移除属性 "attr1"
  ASSERT_FALSE(cls->hasAttribute("attr1")); // 断言 cls 是否不包含属性 "attr1"
  ASSERT_FALSE(cls->hasAttribute("attr2")); // 断言 cls 是否不包含属性 "attr2"
  ASSERT_TRUE(cls->hasAttribute("attr3")); // 断言 cls 是否包含属性 "attr3"

  // check that we can still add a non-parameter attr1 with
  // different type
  cls->addAttribute("attr1", IntType::get()); // 向 cls 添加名为 "attr1" 的属性，类型为 IntType
}

TEST(ClassTypeTest, AddRemoveConstant) { // 定义 ClassType 类的单元测试 AddRemoveConstant
  auto cu = std::make_shared<CompilationUnit>(); // 创建共享的编译单元指针 cu
  auto cls = ClassType::create("foo.bar", cu); // 创建名为 "foo.bar" 的 ClassType 对象 cls
  cls->addConstant("const1", IValue(1)); // 向 cls 添加名为 "const1" 的常量，值为整数 1
  cls->addConstant("const2", IValue(2)); // 向 cls 添加名为 "const2" 的常量，值为整数 2
  cls->addConstant("const3", IValue(3)); // 向 cls 添加名为 "const3" 的常量，值为整数 3
  ASSERT_EQ(cls->numConstants(), 3); // 断言 cls 的常量数量为 3
  ASSERT_TRUE(cls->hasConstant("const1")); // 断言 cls 是否包含常量 "const1"
  ASSERT_TRUE(cls->hasConstant("const2")); // 断言 cls 是否包含常量 "const2"
  ASSERT_TRUE(cls->hasConstant("const3")); // 断言 cls 是否包含常量 "const3"
  ASSERT_FALSE(cls->hasConstant("const4")); // 断言 cls 是否不包含常量 "const4"

  ASSERT_EQ(cls->getConstant("const1").toInt(), 1); // 断言 cls 的常量 "const1" 的整数值为 1
  ASSERT_EQ(cls->getConstant("const2").toInt(), 2); // 断言 cls 的常量 "const2" 的整数值为 2
  ASSERT_EQ(cls->getConstant("const3").toInt(), 3); // 断言 cls 的常量 "const3" 的整数值为 3

  cls->unsafeRemoveConstant("const2"); // 移除常量 "const2"
  ASSERT_TRUE(cls->hasConstant("const1")); // 断言 cls 是否包含常量 "const1"
  ASSERT_FALSE(cls->hasConstant("const2")); // 断言 cls 是否不包含常量 "const2"
  ASSERT_TRUE(cls->hasConstant("const3")); // 断言 cls 是否包含常量 "const3"
}

TEST(ClassTypeTest, IdenticalTypesDifferentCus) { // 定义 ClassType 类的单元测试 IdenticalTypesDifferentCus
  auto cu1 = std::make_shared<CompilationUnit>(); // 创建共享的编译单元指针 cu1
  auto cu2 = std::make_shared<CompilationUnit>(); // 创建共享的编译单元指针 cu2

  // Create two identically named ClassTypes and put them
  // in separate compilation units.
  auto cls1 = ClassType::create("foo", cu1); // 在 cu1 中创建名为 "foo" 的 ClassType 对象 cls1
  auto cls2 = ClassType::create("foo", cu2); // 在 cu2 中创建名为 "foo" 的 ClassType 对象 cls2

  // Create a function that accepts "foo" (cls1) as input.
  Argument arg("arg", cls1); // 创建接受 cls1 类型参数的 Argument 对象 arg
  Argument ret("ret", IntType::get()); // 创建返回类型为整数的 Argument 对象 ret

  FunctionSchema schema("fn", "", {arg}, {ret}); // 创建函数模式 schema，接受一个参数和一个返回值

  jit::BuiltinOpFunction method(
      "method",
      std::move(schema),
      [](jit::Stack& stack) mutable -> void {
        pop(stack); // 弹出栈顶元素
        push(stack, 0); // 将整数 0 推入栈中
      },
      ""); // 创建名为 "method" 的内置操作函数对象

  // Create an object of type cls2.
  Object obj(cu2, cls2); // 使用 cu2 和 cls2 创建 Object 对象 obj

  // Call method with the above object; this should
  // throw an error because the types have identical
  // names but are in different compilation units.
  Stack stack; // 创建堆栈对象 stack
  push(stack, obj._ivalue()); // 将 obj 的 IValue 推入堆栈中
  try {
    method(stack, {}); // 调用 method 函数，期望抛出异常
  } catch (const std::exception& e) {
    // 检查异常中是否包含编译单元的地址，以及类类型的名称。
    testing::FileCheck()
        .check("foo (of Python compilation unit at: 0x")
        ->check_same(")")
        ->check("foo (of Python compilation unit at: 0x")
        ->check_same(")")
        ->run(e.what());
    // 返回，结束函数执行
    return;
    }
    
    // 这段代码不应该执行到。
    // 断言确保条件为真，如果条件为假，将触发断言失败。
    ASSERT_TRUE(false);
}

} // namespace jit
} // namespace torch
```