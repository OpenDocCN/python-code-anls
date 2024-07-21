# `.\pytorch\c10\test\util\registry_test.cpp`

```py
#include <gtest/gtest.h> // 包含 Google Test 框架的头文件
#include <iostream> // 包含标准输入输出流的头文件
#include <memory> // 包含智能指针相关的头文件

#include <c10/util/Registry.h> // 包含 Caffe2/C10 中的注册表头文件

// Note: we use a different namespace to test if the macros defined in
// Registry.h actually works with a different namespace from c10.
namespace c10_test {

class Foo { // 定义类 Foo
 public:
  explicit Foo(int x) { // Foo 类的构造函数，接受一个整数参数 x
    // LOG(INFO) << "Foo " << x;
  }
  virtual ~Foo() = default; // 虚析构函数，默认实现
};

C10_DECLARE_REGISTRY(FooRegistry, Foo, int); // 声明一个名为 FooRegistry 的注册表，注册 Foo 类型，带一个整数参数

C10_DEFINE_REGISTRY(FooRegistry, Foo, int); // 定义 FooRegistry 注册表，注册 Foo 类型，带一个整数参数

#define REGISTER_FOO(clsname) C10_REGISTER_CLASS(FooRegistry, clsname, clsname) // 宏定义，用于注册 FooRegistry 中的类

class Bar : public Foo { // 类 Bar 继承自 Foo
 public:
  explicit Bar(int x) : Foo(x) { // Bar 类的构造函数，调用 Foo 的构造函数
    // LOG(INFO) << "Bar " << x;
  }
};
REGISTER_FOO(Bar); // 将 Bar 类注册到 FooRegistry 中

class AnotherBar : public Foo { // 类 AnotherBar 继承自 Foo
 public:
  explicit AnotherBar(int x) : Foo(x) { // AnotherBar 类的构造函数，调用 Foo 的构造函数
    // LOG(INFO) << "AnotherBar " << x;
  }
};
REGISTER_FOO(AnotherBar); // 将 AnotherBar 类注册到 FooRegistry 中

TEST(RegistryTest, CanRunCreator) { // Google Test 单元测试，测试注册表能否正确创建对象
  std::unique_ptr<Foo> bar(FooRegistry()->Create("Bar", 1)); // 通过 FooRegistry 创建 Bar 类型对象
  EXPECT_TRUE(bar != nullptr) << "Cannot create bar."; // 断言：bar 不为空，否则输出错误信息
  std::unique_ptr<Foo> another_bar(FooRegistry()->Create("AnotherBar", 1)); // 通过 FooRegistry 创建 AnotherBar 类型对象
  EXPECT_TRUE(another_bar != nullptr); // 断言：another_bar 不为空
}

TEST(RegistryTest, ReturnNullOnNonExistingCreator) { // Google Test 单元测试，测试在不存在的创建器时返回空指针
  EXPECT_EQ(FooRegistry()->Create("Non-existing bar", 1), nullptr); // 断言：当创建不存在的对象时返回空指针
}

// C10_REGISTER_CLASS_WITH_PRIORITY defines static variable
void RegisterFooDefault() { // 注册默认优先级的 Foo 类
  C10_REGISTER_CLASS_WITH_PRIORITY(
      FooRegistry, FooWithPriority, c10::REGISTRY_DEFAULT, Foo); // 使用默认优先级注册 Foo 类
}

void RegisterFooDefaultAgain() { // 再次注册默认优先级的 Foo 类
  C10_REGISTER_CLASS_WITH_PRIORITY(
      FooRegistry, FooWithPriority, c10::REGISTRY_DEFAULT, Foo); // 使用默认优先级注册 Foo 类
}

void RegisterFooBarFallback() { // 注册回退优先级的 Bar 类
  C10_REGISTER_CLASS_WITH_PRIORITY(
      FooRegistry, FooWithPriority, c10::REGISTRY_FALLBACK, Bar); // 使用回退优先级注册 Bar 类
}

void RegisterFooBarPreferred() { // 注册首选优先级的 Bar 类
  C10_REGISTER_CLASS_WITH_PRIORITY(
      FooRegistry, FooWithPriority, c10::REGISTRY_PREFERRED, Bar); // 使用首选优先级注册 Bar 类
}

TEST(RegistryTest, RegistryPriorities) { // Google Test 单元测试，测试注册表的优先级
  FooRegistry()->SetTerminate(false); // 设置 FooRegistry 不终止

  RegisterFooDefault(); // 注册默认优先级的 Foo 类

  // throws because Foo is already registered with default priority
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  EXPECT_THROW(RegisterFooDefaultAgain(), std::runtime_error); // 断言：因为 Foo 已经使用默认优先级注册，所以再次注册会抛出异常

#ifdef __GXX_RTTI
  // not going to register Bar because Foo is registered with Default priority
  RegisterFooBarFallback(); // 尝试注册回退优先级的 Bar 类
  std::unique_ptr<Foo> bar1(FooRegistry()->Create("FooWithPriority", 1)); // 通过 FooRegistry 创建 FooWithPriority 类型对象
  EXPECT_EQ(dynamic_cast<Bar*>(bar1.get()), nullptr); // 断言：bar1 不是 Bar 类型的对象

  // will register Bar because of higher priority
  RegisterFooBarPreferred(); // 注册首选优先级的 Bar 类
  std::unique_ptr<Foo> bar2(FooRegistry()->Create("FooWithPriority", 1)); // 通过 FooRegistry 创建 FooWithPriority 类型对象
  EXPECT_NE(dynamic_cast<Bar*>(bar2.get()), nullptr); // 断言：bar2 是 Bar 类型的对象
#endif
}

} // namespace c10_test
```