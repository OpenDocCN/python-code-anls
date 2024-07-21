# `.\pytorch\c10\test\util\ThreadLocal_test.cpp`

```
#include <c10/util/ThreadLocal.h>  // 引入c10库中的ThreadLocal工具
#include <gtest/gtest.h>           // 引入gtest测试框架

#include <atomic>   // 原子操作相关头文件
#include <thread>   // 线程相关头文件

namespace {  // 匿名命名空间，限定符内的定义只在当前文件内有效

TEST(ThreadLocal, TestNoOpScopeWithOneVar) {  // ThreadLocal测试用例：无操作范围内定义一个变量
  C10_DEFINE_TLS_static(std::string, str);  // 定义一个静态的ThreadLocal变量str，类型为std::string
}

TEST(ThreadLocalTest, TestNoOpScopeWithTwoVars) {  // ThreadLocalTest测试用例：无操作范围内定义两个变量
  C10_DEFINE_TLS_static(std::string, str);   // 定义第一个ThreadLocal变量str，类型为std::string
  C10_DEFINE_TLS_static(std::string, str2);  // 定义第二个ThreadLocal变量str2，类型为std::string
}

TEST(ThreadLocalTest, TestScopeWithOneVar) {  // ThreadLocalTest测试用例：有操作范围内定义一个变量
  C10_DEFINE_TLS_static(std::string, str);  // 定义ThreadLocal变量str，类型为std::string
  EXPECT_EQ(*str, std::string());  // 断言str指向的值为空字符串
  EXPECT_EQ(*str, "");             // 断言str指向的值为空字符串

  *str = "abc";  // 将str指向的值设为"abc"
  EXPECT_EQ(*str, "abc");         // 断言str指向的值为"abc"
  EXPECT_EQ(str->length(), 3);    // 断言str指向的字符串长度为3
  EXPECT_EQ(str.get(), "abc");    // 断言str指向的值为"abc"
}

TEST(ThreadLocalTest, TestScopeWithTwoVars) {  // ThreadLocalTest测试用例：有操作范围内定义两个变量
  C10_DEFINE_TLS_static(std::string, str);   // 定义第一个ThreadLocal变量str，类型为std::string
  EXPECT_EQ(*str, "");                      // 断言str指向的值为空字符串

  C10_DEFINE_TLS_static(std::string, str2);  // 定义第二个ThreadLocal变量str2，类型为std::string

  *str = "abc";  // 将str指向的值设为"abc"
  EXPECT_EQ(*str, "abc");         // 断言str指向的值为"abc"
  EXPECT_EQ(*str2, "");           // 断言str2指向的值为空字符串

  *str2 = *str;  // 将str2指向的值设为str指向的值
  EXPECT_EQ(*str, "abc");         // 断言str指向的值为"abc"
  EXPECT_EQ(*str2, "abc");        // 断言str2指向的值为"abc"

  str->clear();  // 清空str指向的值
  EXPECT_EQ(*str, "");            // 断言str指向的值为空字符串
  EXPECT_EQ(*str2, "abc");        // 断言str2指向的值为"abc"
}

TEST(ThreadLocalTest, TestInnerScopeWithTwoVars) {  // ThreadLocalTest测试用例：内部范围定义两个变量
  C10_DEFINE_TLS_static(std::string, str);  // 定义ThreadLocal变量str，类型为std::string
  *str = "abc";                            // 将str指向的值设为"abc"

  {
    C10_DEFINE_TLS_static(std::string, str2);  // 在内部作用域定义ThreadLocal变量str2，类型为std::string
    EXPECT_EQ(*str2, "");                     // 断言str2指向的值为空字符串

    *str2 = *str;       // 将str2指向的值设为str指向的值
    EXPECT_EQ(*str, "abc");   // 断言str指向的值为"abc"
    EXPECT_EQ(*str2, "abc");  // 断言str2指向的值为"abc"

    str->clear();        // 清空str指向的值
    EXPECT_EQ(*str2, "abc");  // 断言str2指向的值为"abc"
  }

  EXPECT_EQ(*str, "");   // 断言str指向的值为空字符串
}

struct Foo {  // 定义结构体Foo
  C10_DECLARE_TLS_class_static(Foo, std::string, str_);  // 在Foo结构体内声明一个ThreadLocal静态成员str_
};

C10_DEFINE_TLS_class_static(Foo, std::string, str_);  // 定义Foo结构体内ThreadLocal静态成员str_

TEST(ThreadLocalTest, TestClassScope) {  // ThreadLocalTest测试用例：类作用域
  EXPECT_EQ(*Foo::str_, "");   // 断言Foo结构体的str_指向的值为空字符串

  *Foo::str_ = "abc";          // 将Foo结构体的str_指向的值设为"abc"
  EXPECT_EQ(*Foo::str_, "abc");  // 断言Foo结构体的str_指向的值为"abc"
  EXPECT_EQ(Foo::str_->length(), 3);  // 断言Foo结构体的str_指向的字符串长度为3
  EXPECT_EQ(Foo::str_.get(), "abc");  // 断言Foo结构体的str_指向的值为"abc"
}

C10_DEFINE_TLS_static(std::string, global_);  // 定义全局ThreadLocal静态变量global_

C10_DEFINE_TLS_static(std::string, global2_);  // 定义全局ThreadLocal静态变量global2_
TEST(ThreadLocalTest, TestTwoGlobalScopeVars) {  // ThreadLocalTest测试用例：两个全局作用域变量
  EXPECT_EQ(*global_, "");   // 断言global_指向的值为空字符串
  EXPECT_EQ(*global2_, "");  // 断言global2_指向的值为空字符串

  *global_ = "abc";          // 将global_指向的值设为"abc"
  EXPECT_EQ(global_->length(), 3);  // 断言global_指向的字符串长度为3
  EXPECT_EQ(*global_, "abc");  // 断言global_指向的值为"abc"
  EXPECT_EQ(*global2_, "");   // 断言global2_指向的值为空字符串

  *global2_ = *global_;      // 将global2_指向的值设为global_指向的值
  EXPECT_EQ(*global_, "abc");   // 断言global_指向的值为"abc"
  EXPECT_EQ(*global2_, "abc");  // 断言global2_指向的值为"abc"

  global_->clear();          // 清空global_指向的值
  EXPECT_EQ(*global_, "");   // 断言global_指向的值为空字符串
  EXPECT_EQ(*global2_, "abc");  // 断言global2_指向的值为"abc"
  EXPECT_EQ(global2_.get(), "abc");  // 断言global2_指向的值为"abc"
}

C10_DEFINE_TLS_static(std::string, global3_);  // 定义全局ThreadLocal静态变量global3_
TEST(ThreadLocalTest, TestGlobalWithLocalScopeVars) {  // ThreadLocalTest测试用例：全局作用域和局部作用域变量
  *global3_ = "abc";        // 将global3_指向的值设为"abc"

  C10_DEFINE_TLS_static(std::string, str);  // 在局部作用域定义ThreadLocal变量str，类型为std::string

  std::swap(*global3_, *str);  // 交换global3_和str指向的值
  EXPECT_EQ(*str, "abc");     // 断言str指向的值为"abc"
  EXPECT_EQ(*global3_, "");   // 断言global3_指向的值为空字符串
}

TEST(ThreadLocalTest, TestThreadWithLocalScopeVar) {  // ThreadLocalTest测试用例：线程与局部作用域变量
  C10_DEFINE_TLS_static(std::string, str);  // 定义局部ThreadLocal静态变量str，类型为std::string
  *str = "abc";               // 将str指向的值设为"abc"

  std::atomic_bool b(false);  // 定义原子布尔变量b，初始
    // 使用 EXPECT_EQ 断言检查 global4_ 指针所指的内容是否等于 "def"
    EXPECT_EQ(*global4_, "def");
  });
  // 等待子线程 t 的结束
  t.join();

  // 使用 EXPECT_TRUE 断言检查 b 的值是否为真
  EXPECT_TRUE(b);
  // 使用 EXPECT_EQ 断言检查 global4_ 指针所指的内容是否等于 "abc"
  EXPECT_EQ(*global4_, "abc");
}

TEST(ThreadLocalTest, TestObjectsAreReleased) {
  // 静态原子整型变量，用于记录构造函数调用次数
  static std::atomic<int> ctors{0};
  // 静态原子整型变量，用于记录析构函数调用次数
  static std::atomic<int> dtors{0};
  
  // 定义结构体 A
  struct A {
    // 构造函数，初始化 i 并增加构造函数调用计数
    A() : i() {
      ++ctors;
    }

    // 析构函数，增加析构函数调用计数
    ~A() {
      ++dtors;
    }

    // 禁用拷贝构造函数
    A(const A&) = delete;
    // 禁用赋值运算符
    A& operator=(const A&) = delete;

    // 整型成员变量 i
    int i;
  };

  // 定义静态线程局部存储，存储类型为 A，命名为 a
  C10_DEFINE_TLS_static(A, a);

  // 原子布尔变量，初始化为 false
  std::atomic_bool b(false);
  
  // 创建新线程
  std::thread t([&b]() {
    // 检查线程局部存储 a 的成员变量 i 的初始值是否为 0
    EXPECT_EQ(a->i, 0);
    // 修改线程局部存储 a 的成员变量 i 的值为 1
    a->i = 1;
    // 再次检查线程局部存储 a 的成员变量 i 的值是否为 1
    EXPECT_EQ(a->i, 1);
    // 将原子变量 b 置为 true
    b = true;
  });

  // 等待线程 t 执行完毕
  t.join();

  // 断言原子变量 b 的值为 true
  EXPECT_TRUE(b);

  // 断言构造函数调用次数为 1
  EXPECT_EQ(ctors, 1);
  // 断言析构函数调用次数为 1
  EXPECT_EQ(dtors, 1);
}

TEST(ThreadLocalTest, TestObjectsAreReleasedByNonstaticThreadLocal) {
  // 静态原子整型变量，用于记录构造函数调用次数
  static std::atomic<int> ctors(0);
  // 静态原子整型变量，用于记录析构函数调用次数
  static std::atomic<int> dtors(0);

  // 定义结构体 A
  struct A {
    // 构造函数，初始化 i 并增加构造函数调用计数
    A() : i() {
      ++ctors;
    }

    // 析构函数，增加析构函数调用计数
    ~A() {
      ++dtors;
    }

    // 禁用拷贝构造函数
    A(const A&) = delete;
    // 禁用赋值运算符
    A& operator=(const A&) = delete;

    // 整型成员变量 i
    int i;
  };

  // 原子布尔变量，初始化为 false
  std::atomic_bool b(false);
  
  // 创建新线程
  std::thread t([&b]() {
    // 检查线程局部存储 a 的成员变量 i 的初始值是否为 0
    EXPECT_EQ(a->i, 0);
    // 修改线程局部存储 a 的成员变量 i 的值为 1
    a->i = 1;
    // 再次检查线程局部存储 a 的成员变量 i 的值是否为 1
    EXPECT_EQ(a->i, 1);
    // 将原子变量 b 置为 true
    b = true;
  });

  // 等待线程 t 执行完毕
  t.join();

  // 断言原子变量 b 的值为 true
  EXPECT_TRUE(b);

  // 断言构造函数调用次数为 1
  EXPECT_EQ(ctors, 1);
  // 断言析构函数调用次数为 1
  EXPECT_EQ(dtors, 1);
}

} // namespace
```