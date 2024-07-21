# `.\pytorch\c10\test\util\intrusive_ptr_test.cpp`

```py
#include <c10/util/intrusive_ptr.h>  // 引入 c10 库中的 intrusive_ptr 头文件

#include <gtest/gtest.h>  // 引入 Google 测试框架的头文件
#include <map>  // 引入 C++ 标准库中的 map 头文件
#include <set>  // 引入 C++ 标准库中的 set 头文件
#include <unordered_map>  // 引入 C++ 标准库中的 unordered_map 头文件
#include <unordered_set>  // 引入 C++ 标准库中的 unordered_set 头文件

using c10::intrusive_ptr;  // 使用 c10 命名空间中的 intrusive_ptr
using c10::intrusive_ptr_target;  // 使用 c10 命名空间中的 intrusive_ptr_target
using c10::make_intrusive;  // 使用 c10 命名空间中的 make_intrusive
using c10::weak_intrusive_ptr;  // 使用 c10 命名空间中的 weak_intrusive_ptr

#ifndef _MSC_VER
#pragma GCC diagnostic ignored "-Wpragmas"  // 忽略 GCC 编译器的特定警告：未知的 pragma
#pragma GCC diagnostic ignored "-Wunknown-warning-option"  // 忽略 GCC 编译器的特定警告：未知的警告选项
#pragma GCC diagnostic ignored "-Wself-move"  // 忽略 GCC 编译器的特定警告：自我移动
#pragma GCC diagnostic ignored "-Wfree-nonheap-object"  // 忽略 GCC 编译器的特定警告：释放非堆对象
#endif

#ifdef __clang__
#pragma clang diagnostic ignored "-Wself-assign-overloaded"  // 忽略 Clang 编译器的特定警告：自我赋值重载
#endif

// NOLINTBEGIN(clang-analyzer-cplusplus*)
namespace {
class SomeClass0Parameters : public intrusive_ptr_target {};  // 定义继承自 intrusive_ptr_target 的 SomeClass0Parameters 类
class SomeClass1Parameter : public intrusive_ptr_target {  // 定义继承自 intrusive_ptr_target 的 SomeClass1Parameter 类
 public:
  SomeClass1Parameter(int param_) : param(param_) {}  // SomeClass1Parameter 类的构造函数，接受一个整型参数 param
  int param;  // 整型成员变量 param
};
class SomeClass2Parameters : public intrusive_ptr_target {  // 定义继承自 intrusive_ptr_target 的 SomeClass2Parameters 类
 public:
  SomeClass2Parameters(int param1_, int param2_)  // SomeClass2Parameters 类的构造函数，接受两个整型参数 param1_ 和 param2_
      : param1(param1_), param2(param2_) {}
  int param1;  // 整型成员变量 param1
  int param2;  // 整型成员变量 param2
};
using SomeClass = SomeClass0Parameters;  // 使用 SomeClass0Parameters 作为 SomeClass 的别名
struct SomeBaseClass : public intrusive_ptr_target {  // 定义继承自 intrusive_ptr_target 的 SomeBaseClass 结构体
  SomeBaseClass(int v_) : v(v_) {}  // SomeBaseClass 结构体的构造函数，接受一个整型参数 v_
  int v;  // 整型成员变量 v
};
struct SomeChildClass : SomeBaseClass {  // 定义继承自 SomeBaseClass 的 SomeChildClass 结构体
  SomeChildClass(int v) : SomeBaseClass(v) {}  // SomeChildClass 结构体的构造函数，继承自父类的构造函数
};

class DestructableMock : public intrusive_ptr_target {  // 定义继承自 intrusive_ptr_target 的 DestructableMock 类
 public:
  DestructableMock(bool* resourcesReleased, bool* wasDestructed)  // DestructableMock 类的构造函数，接受两个 bool 类型指针参数
      : resourcesReleased_(resourcesReleased), wasDestructed_(wasDestructed) {}

  ~DestructableMock() override {  // 虚析构函数，用于资源释放
    *resourcesReleased_ = true;  // 标记资源已释放
    *wasDestructed_ = true;  // 标记已析构
  }

  void release_resources() override {  // 虚函数，用于释放资源
    *resourcesReleased_ = true;  // 标记资源已释放
  }

 private:
  bool* resourcesReleased_;  // 指向资源释放标志的指针
  bool* wasDestructed_;  // 指向析构标志的指针
};

class ChildDestructableMock final : public DestructableMock {  // 定义继承自 DestructableMock 的 ChildDestructableMock 类
 public:
  ChildDestructableMock(bool* resourcesReleased, bool* wasDestructed)
      : DestructableMock(resourcesReleased, wasDestructed) {}  // ChildDestructableMock 类的构造函数，调用父类构造函数
};

class NullType1 final {
  static SomeClass singleton_;  // 静态成员变量 singleton_

 public:
  static constexpr SomeClass* singleton() {  // 返回静态成员 singleton_ 的指针
    return &singleton_;
  }
};
SomeClass NullType1::singleton_;  // 初始化 NullType1 类的静态成员变量 singleton_

class NullType2 final {
  static SomeClass singleton_;  // 静态成员变量 singleton_

 public:
  static constexpr SomeClass* singleton() {  // 返回静态成员 singleton_ 的指针
    return &singleton_;
  }
};
SomeClass NullType2::singleton_;  // 初始化 NullType2 类的静态成员变量 singleton_

static_assert(NullType1::singleton() != NullType2::singleton());  // 静态断言，确保 NullType1 和 NullType2 的 singleton() 返回不同的指针类型
} // namespace

static_assert(
    std::is_same_v<SomeClass, intrusive_ptr<SomeClass>::element_type>,  // 静态断言，检查 intrusive_ptr<SomeClass> 类型的 element_type 是否正确
    "intrusive_ptr<T>::element_type is wrong");

TEST(MakeIntrusiveTest, ClassWith0Parameters) {
  intrusive_ptr<SomeClass0Parameters> var =
      make_intrusive<SomeClass0Parameters>();  // 创建一个没有参数的 SomeClass0Parameters 类型的 intrusive_ptr 对象
  // 检查 var 的类型是否正确
  EXPECT_EQ(var.get(), dynamic_cast<SomeClass0Parameters*>(var.get()));
}

TEST(MakeIntrusiveTest, ClassWith1Parameter) {
  intrusive_ptr<SomeClass1Parameter> var =
      make_intrusive<SomeClass1Parameter>(5);  // 创建一个带有一个整型参数的 SomeClass1Parameter 类型的 intrusive_ptr 对象
  EXPECT_EQ(5, var->param);  // 检查 var 的 param 成员是否为 5
}
TEST(MakeIntrusiveTest, ClassWith2Parameters) {
  // 创建一个具有两个参数的 SomeClass2Parameters 的 intrusive_ptr
  intrusive_ptr<SomeClass2Parameters> var =
      make_intrusive<SomeClass2Parameters>(7, 2);
  // 验证第一个参数是否为7
  EXPECT_EQ(7, var->param1);
  // 验证第二个参数是否为2
  EXPECT_EQ(2, var->param2);
}

TEST(MakeIntrusiveTest, TypeIsAutoDeductible) {
  // 使用 auto 推导类型，创建没有参数的 SomeClass0Parameters 的 intrusive_ptr
  auto var2 = make_intrusive<SomeClass0Parameters>();
  // 使用 auto 推导类型，创建带有一个参数的 SomeClass1Parameter 的 intrusive_ptr
  auto var3 = make_intrusive<SomeClass1Parameter>(2);
  // 使用 auto 推导类型，创建带有两个参数的 SomeClass2Parameters 的 intrusive_ptr
  auto var4 = make_intrusive<SomeClass2Parameters>(2, 3);
}

TEST(MakeIntrusiveTest, CanAssignToBaseClassPtr) {
  // 将一个指向 SomeChildClass 实例的 intrusive_ptr 赋给 SomeBaseClass 类型的变量
  intrusive_ptr<SomeBaseClass> var = make_intrusive<SomeChildClass>(3);
  // 验证变量的 v 成员是否为3
  EXPECT_EQ(3, var->v);
}

TEST(IntrusivePtrTargetTest, whenAllocatedOnStack_thenDoesntCrash) {
  // 在栈上分配 SomeClass 的实例
  SomeClass myClass;
}

TEST(IntrusivePtrTest, givenValidPtr_whenCallingGet_thenReturnsObject) {
  // 创建一个带有一个参数的 SomeClass1Parameter 的 intrusive_ptr
  intrusive_ptr<SomeClass1Parameter> obj =
      make_intrusive<SomeClass1Parameter>(5);
  // 验证对象的 param 成员是否为5
  EXPECT_EQ(5, obj.get()->param);
}

TEST(IntrusivePtrTest, givenValidPtr_whenCallingConstGet_thenReturnsObject) {
  // 创建一个带有一个参数的 SomeClass1Parameter 的 const intrusive_ptr
  const intrusive_ptr<SomeClass1Parameter> obj =
      make_intrusive<SomeClass1Parameter>(5);
  // 验证对象的 param 成员是否为5
  EXPECT_EQ(5, obj.get()->param);
}

TEST(IntrusivePtrTest, givenInvalidPtr_whenCallingGet_thenReturnsNullptr) {
  // 创建一个空的 intrusive_ptr<SomeClass1Parameter>
  intrusive_ptr<SomeClass1Parameter> obj;
  // 验证调用 get() 后是否返回 nullptr
  EXPECT_EQ(nullptr, obj.get());
}

TEST(IntrusivePtrTest, givenNullptr_whenCallingGet_thenReturnsNullptr) {
  // 创建一个指向 nullptr 的 intrusive_ptr<SomeClass1Parameter>
  intrusive_ptr<SomeClass1Parameter> obj(nullptr);
  // 验证调用 get() 后是否返回 nullptr
  EXPECT_EQ(nullptr, obj.get());
}

TEST(IntrusivePtrTest, givenValidPtr_whenDereferencing_thenReturnsObject) {
  // 创建一个带有一个参数的 SomeClass1Parameter 的 intrusive_ptr
  intrusive_ptr<SomeClass1Parameter> obj =
      make_intrusive<SomeClass1Parameter>(5);
  // 验证通过 * 操作符解引用后，对象的 param 成员是否为5
  EXPECT_EQ(5, (*obj).param);
}

TEST(IntrusivePtrTest, givenValidPtr_whenConstDereferencing_thenReturnsObject) {
  // 创建一个带有一个参数的 SomeClass1Parameter 的 const intrusive_ptr
  const intrusive_ptr<SomeClass1Parameter> obj =
      make_intrusive<SomeClass1Parameter>(5);
  // 验证通过 * 操作符解引用后，对象的 param 成员是否为5
  EXPECT_EQ(5, (*obj).param);
}

TEST(IntrusivePtrTest, givenValidPtr_whenArrowDereferencing_thenReturnsObject) {
  // 创建一个带有一个参数的 SomeClass1Parameter 的 intrusive_ptr
  intrusive_ptr<SomeClass1Parameter> obj =
      make_intrusive<SomeClass1Parameter>(3);
  // 验证通过 -> 操作符解引用后，对象的 param 成员是否为3
  EXPECT_EQ(3, obj->param);
}

TEST(
    IntrusivePtrTest,
    givenValidPtr_whenConstArrowDereferencing_thenReturnsObject) {
  // 创建一个带有一个参数的 SomeClass1Parameter 的 const intrusive_ptr
  const intrusive_ptr<SomeClass1Parameter> obj =
      make_intrusive<SomeClass1Parameter>(3);
  // 验证通过 -> 操作符解引用后，对象的 param 成员是否为3
  EXPECT_EQ(3, obj->param);
}

TEST(IntrusivePtrTest, givenValidPtr_whenMoveAssigning_thenPointsToSameObject) {
  // 创建两个 SomeClass 的 intrusive_ptr 实例
  intrusive_ptr<SomeClass> obj1 = make_intrusive<SomeClass>();
  intrusive_ptr<SomeClass> obj2 = make_intrusive<SomeClass>();
  // 获取 obj1 的原始指针
  SomeClass* obj1ptr = obj1.get();
  // 使用 std::move 进行移动赋值操作
  obj2 = std::move(obj1);
  // 验证移动后 obj2 是否指向与 obj1ptr 相同的对象
  EXPECT_EQ(obj1ptr, obj2.get());
}

TEST(IntrusivePtrTest, givenValidPtr_whenMoveAssigning_thenOldInstanceInvalid) {
  // 创建两个 SomeClass 的 intrusive_ptr 实例
  intrusive_ptr<SomeClass> obj1 = make_intrusive<SomeClass>();
  intrusive_ptr<SomeClass> obj2 = make_intrusive<SomeClass>();
  // 使用 std::move 进行移动赋值操作
  obj2 = std::move(obj1);
  // 验证移动后 obj1 是否已经无效
  // NOLINTNEXTLINE(clang-analyzer-cplusplus.Move,bugprone-use-after-move)
  EXPECT_FALSE(obj1.defined());
}

TEST(
    IntrusivePtrTest,
    givenValidPtr_whenMoveAssigningToSelf_thenPointsToSameObject) {
  // 创建一个intrusive_ptr指向SomeClass的实例
  intrusive_ptr<SomeClass> obj1 = make_intrusive<SomeClass>();
  // 获取obj1的裸指针
  SomeClass* obj1ptr = obj1.get();
  // 将obj1移动给自身（实际上不会移动，但语法上是有效的）
  obj1 = std::move(obj1);
  // 由于移动赋值不会改变裸指针的值，所以这里期望obj1和obj1ptr指向相同的对象
  // NOLINTNEXTLINE(bugprone-use-after-move)：此处禁止linter提示使用移动后对象的bug
  EXPECT_EQ(obj1ptr, obj1.get());
}

TEST(IntrusivePtrTest, givenValidPtr_whenMoveAssigningToSelf_thenStaysValid) {
  // 创建一个intrusive_ptr对象obj1，指向SomeClass类型的实例
  intrusive_ptr<SomeClass> obj1 = make_intrusive<SomeClass>();
  // 将obj1移动赋值给自身，这应该保持obj1有效
  obj1 = std::move(obj1);
  // NOLINTNEXTLINE(bugprone-use-after-move)
  // 验证obj1仍然有效
  EXPECT_TRUE(obj1.defined());
}

TEST(
    IntrusivePtrTest,
    givenInvalidPtr_whenMoveAssigningToSelf_thenStaysInvalid) {
  // 创建一个空的intrusive_ptr对象obj1
  intrusive_ptr<SomeClass> obj1;
  // 将obj1移动赋值给自身，这应该保持obj1无效
  obj1 = std::move(obj1);
  // NOLINTNEXTLINE(bugprone-use-after-move)
  // 验证obj1仍然无效
  EXPECT_FALSE(obj1.defined());
}

TEST(
    IntrusivePtrTest,
    givenInvalidPtr_whenMoveAssigning_thenNewInstanceIsValid) {
  // 创建一个intrusive_ptr对象obj1，指向SomeClass类型的实例
  intrusive_ptr<SomeClass> obj1 = make_intrusive<SomeClass>();
  // 创建一个空的intrusive_ptr对象obj2
  intrusive_ptr<SomeClass> obj2;
  // 将obj1移动赋值给obj2，此时obj2应该有效
  obj2 = std::move(obj1);
  EXPECT_TRUE(obj2.defined());
}

TEST(
    IntrusivePtrTest,
    givenInvalidPtr_whenMoveAssigning_thenPointsToSameObject) {
  // 创建一个intrusive_ptr对象obj1，指向SomeClass类型的实例
  intrusive_ptr<SomeClass> obj1 = make_intrusive<SomeClass>();
  // 创建一个空的intrusive_ptr对象obj2
  intrusive_ptr<SomeClass> obj2;
  // 获取obj1当前指向的对象指针
  SomeClass* obj1ptr = obj1.get();
  // 将obj1移动赋值给obj2
  obj2 = std::move(obj1);
  // 验证obj2现在指向的对象指针与obj1之前相同
  EXPECT_EQ(obj1ptr, obj2.get());
}

TEST(
    IntrusivePtrTest,
    givenValidPtr_whenMoveAssigningFromInvalidPtr_thenNewInstanceIsInvalid) {
  // 创建一个空的intrusive_ptr对象obj1
  intrusive_ptr<SomeClass> obj1;
  // 创建一个intrusive_ptr对象obj2，指向SomeClass类型的实例
  intrusive_ptr<SomeClass> obj2 = make_intrusive<SomeClass>();
  // 验证obj2当前有效
  EXPECT_TRUE(obj2.defined());
  // 将obj1移动赋值给obj2，此时obj2应该无效
  obj2 = std::move(obj1);
  EXPECT_FALSE(obj2.defined());
}

TEST(
    IntrusivePtrTest,
    givenValidPtr_whenMoveAssigningToBaseClass_thenPointsToSameObject) {
  // 创建一个intrusive_ptr对象obj1，指向SomeChildClass类型的实例
  intrusive_ptr<SomeChildClass> obj1 = make_intrusive<SomeChildClass>(1);
  // 创建一个intrusive_ptr对象obj2，指向SomeBaseClass类型的实例
  intrusive_ptr<SomeBaseClass> obj2 = make_intrusive<SomeBaseClass>(2);
  // 获取obj1当前指向的SomeBaseClass对象指针
  SomeBaseClass* obj1ptr = obj1.get();
  // 将obj1移动赋值给obj2
  obj2 = std::move(obj1);
  // 验证obj2现在指向的对象指针与obj1之前相同
  EXPECT_EQ(obj1ptr, obj2.get());
  // 验证obj2现在指向的对象的v属性为1
  EXPECT_EQ(1, obj2->v);
}

TEST(
    IntrusivePtrTest,
    givenValidPtr_whenMoveAssigningToBaseClass_thenOldInstanceInvalid) {
  // 创建一个intrusive_ptr对象obj1，指向SomeChildClass类型的实例
  intrusive_ptr<SomeChildClass> obj1 = make_intrusive<SomeChildClass>(1);
  // 创建一个intrusive_ptr对象obj2，指向SomeBaseClass类型的实例
  intrusive_ptr<SomeBaseClass> obj2 = make_intrusive<SomeBaseClass>(2);
  // 将obj1移动赋值给obj2
  obj2 = std::move(obj1);
  // NOLINTNEXTLINE(bugprone-use-after-move)
  // 验证obj1现在应该无效
  EXPECT_FALSE(obj1.defined());
}

TEST(
    IntrusivePtrTest,
    givenInvalidPtr_whenMoveAssigningToBaseClass_thenNewInstanceIsValid) {
  // 创建一个intrusive_ptr对象obj1，指向SomeChildClass类型的实例
  intrusive_ptr<SomeChildClass> obj1 = make_intrusive<SomeChildClass>(5);
  // 创建一个空的intrusive_ptr对象obj2
  intrusive_ptr<SomeBaseClass> obj2;
  // 将obj1移动赋值给obj2，此时obj2应该有效
  obj2 = std::move(obj1);
  EXPECT_TRUE(obj2.defined());
}

TEST(
    IntrusivePtrTest,
    givenInvalidPtr_whenMoveAssigningToBaseClass_thenPointsToSameObject) {
  // 创建一个intrusive_ptr对象obj1，指向SomeChildClass类型的实例
  intrusive_ptr<SomeChildClass> obj1 = make_intrusive<SomeChildClass>(5);
  // 创建一个空的intrusive_ptr对象obj2
  intrusive_ptr<SomeBaseClass> obj2;
  // 获取obj1当前指向的SomeBaseClass对象指针
  SomeBaseClass* obj1ptr = obj1.get();
  // 将obj1移动赋值给obj2
  obj2 = std::move(obj1);
  // 验证obj2现在指向的对象指针与obj1之前相同
  EXPECT_EQ(obj1ptr, obj2.get());
  // 验证obj2现在指向的对象的v属性为5
  EXPECT_EQ(5, obj2->v);
}

TEST(
    IntrusivePtrTest,
    givenInvalidPtr_whenMoveAssigningInvalidPtrToBaseClass_thenNewInstanceIsValid) {
  // 创建一个空的intrusive_ptr对象obj1
  intrusive_ptr<SomeChildClass> obj1;
  // 创建一个intrusive_ptr对象obj2，指向SomeBaseClass类型的实例
  intrusive_ptr<SomeBaseClass> obj2 = make_intrusive<SomeBaseClass>(2);
  // 验证obj2当前有效
  EXPECT_TRUE(obj2.defined());
  // 将obj1移动赋值给obj2，此时obj2应该无效
  obj2 = std::move(obj1);
  EXPECT_FALSE(obj2.defined());
}

TEST(
    IntrusivePtrTest,
    // 定义一个测试用例，验证在将空指针赋值给不同类型的空指针后，它们确实指向不同的单例对象
    givenNullPtr_whenMoveAssigningToDifferentNullptr_thenHasNewNullptr) {
      // 创建两个空指针 intrusive_ptr，分别使用不同的 NullType
      intrusive_ptr<SomeClass, NullType1> obj1;
      intrusive_ptr<SomeClass, NullType2> obj2;
      // 将 obj1 使用 std::move 赋值给 obj2，此时 obj1 将变为空指针
      obj2 = std::move(obj1);
      // 验证两种 NullType 的单例对象不相同
      EXPECT_NE(NullType1::singleton(), NullType2::singleton());
      // NOLINTNEXTLINE(bugprone-use-after-move)
      // 验证 obj1 现在应该指向 NullType1 的单例对象
      EXPECT_EQ(NullType1::singleton(), obj1.get());
      // 验证 obj2 现在应该指向 NullType2 的单例对象
      EXPECT_EQ(NullType2::singleton(), obj2.get());
      // 验证 obj1 现在应该为未定义状态（即为空指针）
      EXPECT_FALSE(obj1.defined());
      // 验证 obj2 现在应该为未定义状态（即为空指针）
      EXPECT_FALSE(obj2.defined());
    }
TEST(IntrusivePtrTest, givenValidPtr_whenCopyAssigning_thenPointsToSameObject) {
  // 创建两个指向 SomeClass 对象的智能指针，obj1 和 obj2
  intrusive_ptr<SomeClass> obj1 = make_intrusive<SomeClass>();
  intrusive_ptr<SomeClass> obj2 = make_intrusive<SomeClass>();
  // 获取 obj1 的原始指针
  SomeClass* obj1ptr = obj1.get();
  // 将 obj1 的值赋给 obj2，此时它们应该指向相同的对象
  obj2 = obj1;
  // 断言 obj2 的原始指针与 obj1 的相同
  EXPECT_EQ(obj1ptr, obj2.get());
}

TEST(IntrusivePtrTest, givenValidPtr_whenCopyAssigning_thenOldInstanceValid) {
  // 创建两个指向 SomeClass 对象的智能指针，obj1 和 obj2
  intrusive_ptr<SomeClass> obj1 = make_intrusive<SomeClass>();
  intrusive_ptr<SomeClass> obj2 = make_intrusive<SomeClass>();
  // 将 obj1 的值赋给 obj2，此时 obj1 应该保持有效
  obj2 = obj1;
  // 断言 obj1 仍然是有效的
  EXPECT_TRUE(obj1.defined());
}

TEST(
    IntrusivePtrTest,
    givenValidPtr_whenCopyAssigningToSelf_thenPointsToSameObject) {
  // 创建一个指向 SomeClass 对象的智能指针 obj1
  intrusive_ptr<SomeClass> obj1 = make_intrusive<SomeClass>();
  // 获取 obj1 的原始指针
  SomeClass* obj1ptr = obj1.get();
  // 将 obj1 的值赋给它自己，应该依然指向相同的对象
  // NOLINTNEXTLINE(clang-diagnostic-self-assign-overloaded)
  obj1 = obj1;
  // 断言 obj1 的原始指针与之前获取的相同
  EXPECT_EQ(obj1ptr, obj1.get());
}

TEST(IntrusivePtrTest, givenValidPtr_whenCopyAssigningToSelf_thenStaysValid) {
  // 创建一个指向 SomeClass 对象的智能指针 obj1
  intrusive_ptr<SomeClass> obj1 = make_intrusive<SomeClass>();
  // 将 obj1 的值赋给它自己，应该依然保持有效
  // NOLINTNEXTLINE(clang-diagnostic-self-assign-overloaded)
  obj1 = obj1;
  // 断言 obj1 仍然是有效的
  EXPECT_TRUE(obj1.defined());
}

TEST(
    IntrusivePtrTest,
    givenInvalidPtr_whenCopyAssigningToSelf_thenStaysInvalid) {
  // 创建一个空的智能指针 obj1
  intrusive_ptr<SomeClass> obj1;
  // 将 obj1 的值赋给它自己，应该保持无效状态
  // NOLINTNEXTLINE(clang-diagnostic-self-assign-overloaded)
  obj1 = obj1;
  // 断言 obj1 是无效的
  EXPECT_FALSE(obj1.defined());
}

TEST(
    IntrusivePtrTest,
    givenInvalidPtr_whenCopyAssigning_thenNewInstanceIsValid) {
  // 创建一个指向 SomeClass 对象的智能指针 obj1
  intrusive_ptr<SomeClass> obj1 = make_intrusive<SomeClass>();
  // 创建一个空的智能指针 obj2，将 obj1 的值赋给 obj2
  intrusive_ptr<SomeClass> obj2;
  obj2 = obj1;
  // 断言 obj2 现在是有效的
  EXPECT_TRUE(obj2.defined());
}

TEST(
    IntrusivePtrTest,
    givenValidPtr_whenCopyAssigningToBaseClass_thenPointsToSameObject) {
  // 创建一个指向 SomeChildClass 对象的智能指针 child
  intrusive_ptr<SomeChildClass> child = make_intrusive<SomeChildClass>(3);
  // 创建一个指向 SomeBaseClass 对象的智能指针 base
  intrusive_ptr<SomeBaseClass> base = make_intrusive<SomeBaseClass>(10);
  // 将 child 的值赋给 base，因为 SomeChildClass 是 SomeBaseClass 的子类，
  // 所以 base 应该指向 child 所指向的对象
  base = child;
  // 断言 base 所指向的对象的某个属性值为 3
  EXPECT_EQ(3, base->v);
}

TEST(
    IntrusivePtrTest,
    givenValidPtr_whenCopyAssigningToBaseClass_thenOldInstanceInvalid) {
  // 创建一个指向 SomeChildClass 对象的智能指针 obj1
  intrusive_ptr<SomeChildClass> obj1 = make_intrusive<SomeChildClass>(3);
  // 创建一个指向 SomeBaseClass 对象的智能指针 obj2
  intrusive_ptr<SomeBaseClass> obj2 = make_intrusive<SomeBaseClass>(10);
  // 将 obj1 的值赋给 obj2，obj1 应该保持有效
  obj2 = obj1;
  // 断言 obj1 仍然是有效的
  EXPECT_TRUE(obj1.defined());
}

TEST(
    IntrusivePtrTest,
    givenInvalidPtr_whenCopyAssigningToBaseClass_thenNewInstanceIsValid) {
  // 创建一个指向 SomeChildClass 对象的智能指针 obj1
  intrusive_ptr<SomeChildClass> obj1 = make_intrusive<SomeChildClass>(5);
  // 创建一个空的指向 SomeBaseClass 对象的智能指针 obj2
  intrusive_ptr<SomeBaseClass> obj2;
  // 将 obj1 的值赋给 obj2
  obj2 = obj1;
  // 断言 obj2 现在是有效的
  EXPECT_TRUE(obj2.defined());
}

TEST(
    IntrusivePtrTest,
    givenInvalidPtr_whenCopyAssigningToBaseClass_thenPointsToSameObject) {
  // 创建一个指向 SomeChildClass 对象的智能指针 obj1
  intrusive_ptr<SomeChildClass> obj1 = make_intrusive<SomeChildClass>(5);
  // 创建一个空的指向 SomeBaseClass 对象的智能指针 obj2
  intrusive_ptr<SomeBaseClass> obj2;
  // 获取 obj1 的原始指针
  SomeBaseClass* obj1ptr = obj1.get();
  // 将 obj1 的值赋给 obj2，obj2 应该指向与 obj1 相同的对象
  obj2 = obj1;
  // 断言 obj2 的原始指针与 obj1 的相同
  EXPECT_EQ(obj1ptr, obj2.get());
  // 断言 obj2 所指向的对象的某个属性值为 5
  EXPECT_EQ(5, obj2->v);
}
    givenPtr_whenCopyAssigningInvalidPtrToBaseClass_thenNewInstanceIsInvalid) {
        // 创建一个名为 givenPtr_whenCopyAssigningInvalidPtrToBaseClass_thenNewInstanceIsInvalid 的测试用例
        intrusive_ptr<SomeChildClass> obj1;
        // 声明一个空的 intrusive_ptr 对象 obj1，指向 SomeChildClass 类型的对象
        intrusive_ptr<SomeBaseClass> obj2 = make_intrusive<SomeBaseClass>(2);
        // 声明一个 intrusive_ptr 对象 obj2，指向通过 make_intrusive 创建的 SomeBaseClass 类型对象，参数为 2
        EXPECT_TRUE(obj2.defined());
        // 断言 obj2 是否已定义（即指向了有效对象）
        obj2 = obj1;
        // 将 obj1 赋值给 obj2
        EXPECT_FALSE(obj2.defined());
        // 断言 obj2 是否未定义（此时应为无效指针）
TEST(
    IntrusivePtrTest,
    givenNullPtr_whenCopyAssigningToDifferentNullptr_thenHasNewNullptr) {
  // 创建两个空的intrusive_ptr，使用不同的空类型
  intrusive_ptr<SomeClass, NullType1> obj1;
  intrusive_ptr<SomeClass, NullType2> obj2;
  // 将obj1赋值给obj2
  obj2 = obj1;
  // 检查两个空类型不相同
  EXPECT_NE(NullType1::singleton(), NullType2::singleton());
  // 检查obj1指向NullType1的空指针
  EXPECT_EQ(NullType1::singleton(), obj1.get());
  // 检查obj2指向NullType2的空指针
  EXPECT_EQ(NullType2::singleton(), obj2.get());
  // 检查obj1未定义
  EXPECT_FALSE(obj1.defined());
  // 检查obj2未定义
  EXPECT_FALSE(obj2.defined());
}

TEST(IntrusivePtrTest, givenPtr_whenMoveConstructing_thenPointsToSameObject) {
  // 创建intrusive_ptr对象obj1，指向新建的SomeClass实例
  intrusive_ptr<SomeClass> obj1 = make_intrusive<SomeClass>();
  // 获取obj1的原始指针
  SomeClass* obj1ptr = obj1.get();
  // 使用std::move将obj1移动给obj2
  intrusive_ptr<SomeClass> obj2 = std::move(obj1);
  // 检查obj2指向的对象与obj1ptr相同
  EXPECT_EQ(obj1ptr, obj2.get());
}

TEST(IntrusivePtrTest, givenPtr_whenMoveConstructing_thenOldInstanceInvalid) {
  // 创建intrusive_ptr对象obj1，指向新建的SomeClass实例
  intrusive_ptr<SomeClass> obj1 = make_intrusive<SomeClass>();
  // 使用std::move将obj1移动给obj2
  intrusive_ptr<SomeClass> obj2 = std::move(obj1);
  // NOLINTNEXTLINE(clang-analyzer-cplusplus.Move,bugprone-use-after-move)
  // 检查obj1未定义（在移动后使用）
  EXPECT_FALSE(obj1.defined());
}

TEST(IntrusivePtrTest, givenPtr_whenMoveConstructing_thenNewInstanceValid) {
  // 创建intrusive_ptr对象obj1，指向新建的SomeClass实例
  intrusive_ptr<SomeClass> obj1 = make_intrusive<SomeClass>();
  // 使用std::move将obj1移动给obj2
  intrusive_ptr<SomeClass> obj2 = std::move(obj1);
  // 检查obj2定义
  EXPECT_TRUE(obj2.defined());
}

TEST(
    IntrusivePtrTest,
    givenPtr_whenMoveConstructingFromInvalidPtr_thenNewInstanceInvalid) {
  // 创建一个空的intrusive_ptr对象obj1
  intrusive_ptr<SomeClass> obj1;
  // 使用std::move将obj1移动给obj2
  intrusive_ptr<SomeClass> obj2 = std::move(obj1);
  // 检查obj2未定义
  EXPECT_FALSE(obj2.defined());
}

TEST(
    IntrusivePtrTest,
    givenPtr_whenMoveConstructingToBaseClass_thenPointsToSameObject) {
  // 创建intrusive_ptr对象child，指向使用参数3创建的SomeChildClass实例
  intrusive_ptr<SomeChildClass> child = make_intrusive<SomeChildClass>(3);
  // 获取child的原始指针
  SomeBaseClass* objptr = child.get();
  // 使用std::move将child移动给base
  intrusive_ptr<SomeBaseClass> base = std::move(child);
  // 检查base指向的对象的v属性为3
  EXPECT_EQ(3, base->v);
  // 检查base指向的对象与objptr相同
  EXPECT_EQ(objptr, base.get());
}

TEST(
    IntrusivePtrTest,
    givenPtr_whenMoveConstructingToBaseClass_thenOldInstanceInvalid) {
  // 创建intrusive_ptr对象child，指向使用参数3创建的SomeChildClass实例
  intrusive_ptr<SomeChildClass> child = make_intrusive<SomeChildClass>(3);
  // 使用std::move将child移动给base
  intrusive_ptr<SomeBaseClass> base = std::move(child);
  // NOLINTNEXTLINE(bugprone-use-after-move)
  // 检查child未定义（在移动后使用）
  EXPECT_FALSE(child.defined());
}

TEST(
    IntrusivePtrTest,
    givenPtr_whenMoveConstructingToBaseClass_thenNewInstanceValid) {
  // 创建intrusive_ptr对象obj1，指向使用参数2创建的SomeChildClass实例
  intrusive_ptr<SomeChildClass> obj1 = make_intrusive<SomeChildClass>(2);
  // 使用std::move将obj1移动给obj2
  intrusive_ptr<SomeBaseClass> obj2 = std::move(obj1);
  // 检查obj2定义
  EXPECT_TRUE(obj2.defined());
}

TEST(
    IntrusivePtrTest,
    givenPtr_whenMoveConstructingToBaseClassFromInvalidPtr_thenNewInstanceInvalid) {
  // 创建一个空的intrusive_ptr对象obj1
  intrusive_ptr<SomeChildClass> obj1;
  // 使用std::move将obj1移动给obj2
  intrusive_ptr<SomeBaseClass> obj2 = std::move(obj1);
  // 检查obj2未定义
  EXPECT_FALSE(obj2.defined());
}
    // 给定空指针时，将其移动构造到不同的空指针对象
    givenNullPtr_whenMoveConstructingToDifferentNullptr_thenHasNewNullptr) {
      // 创建一个空的intrusive_ptr对象obj1，使用NullType1作为空指针类型
      intrusive_ptr<SomeClass, NullType1> obj1;
      // 使用std::move将obj1移动构造到obj2，使用NullType2作为新的空指针类型
      intrusive_ptr<SomeClass, NullType2> obj2 = std::move(obj1);
      // 检查NullType1的单例与NullType2的单例不相等
      EXPECT_NE(NullType1::singleton(), NullType2::singleton());
      // NOLINTNEXTLINE(bugprone-use-after-move)
      // 检查obj1是否被std::move后设置为NullType1的单例空指针
      EXPECT_EQ(NullType1::singleton(), obj1.get());
      // 检查obj2是否设置为NullType2的单例空指针
      EXPECT_EQ(NullType2::singleton(), obj2.get());
      // 检查obj1是否已定义（应该是未定义的状态）
      EXPECT_FALSE(obj1.defined());
      // 检查obj2是否已定义（应该是未定义的状态）
      EXPECT_FALSE(obj2.defined());
    }
TEST(IntrusivePtrTest, givenPtr_whenCopyConstructing_thenPointsToSameObject) {
  // 创建一个指向 SomeClass 对象的 intrusive_ptr
  intrusive_ptr<SomeClass> obj1 = make_intrusive<SomeClass>();
  // 获取 obj1 指针
  SomeClass* obj1ptr = obj1.get();
  // 使用 obj1 进行拷贝构造创建 obj2
  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
  intrusive_ptr<SomeClass> obj2 = obj1;
  // 断言 obj2 与 obj1 指向相同对象
  EXPECT_EQ(obj1ptr, obj2.get());
  // 断言 obj1 是有效的（defined）
  EXPECT_TRUE(obj1.defined());
}

TEST(IntrusivePtrTest, givenPtr_whenCopyConstructing_thenOldInstanceValid) {
  // 创建一个指向 SomeClass 对象的 intrusive_ptr
  intrusive_ptr<SomeClass> obj1 = make_intrusive<SomeClass>();
  // 使用 obj1 进行拷贝构造创建 obj2
  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
  intrusive_ptr<SomeClass> obj2 = obj1;
  // 断言 obj1 是有效的（defined）
  EXPECT_TRUE(obj1.defined());
}

TEST(IntrusivePtrTest, givenPtr_whenCopyConstructing_thenNewInstanceValid) {
  // 创建一个指向 SomeClass 对象的 intrusive_ptr
  intrusive_ptr<SomeClass> obj1 = make_intrusive<SomeClass>();
  // 使用 obj1 进行拷贝构造创建 obj2
  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
  intrusive_ptr<SomeClass> obj2 = obj1;
  // 断言 obj2 是有效的（defined）
  EXPECT_TRUE(obj2.defined());
}

TEST(
    IntrusivePtrTest,
    givenPtr_whenCopyConstructingFromInvalidPtr_thenNewInstanceInvalid) {
  // 创建一个空的 intrusive_ptr<SomeClass>
  intrusive_ptr<SomeClass> obj1;
  // 使用 obj1 进行拷贝构造创建 obj2
  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
  intrusive_ptr<SomeClass> obj2 = obj1;
  // 断言 obj2 是无效的（未定义）
  EXPECT_FALSE(obj2.defined());
}

TEST(
    IntrusivePtrTest,
    givenPtr_whenCopyConstructingToBaseClass_thenPointsToSameObject) {
  // 创建一个指向 SomeChildClass 对象的 intrusive_ptr
  intrusive_ptr<SomeChildClass> child = make_intrusive<SomeChildClass>(3);
  // 获取 child 指针
  SomeBaseClass* objptr = child.get();
  // 将 child 拷贝构造为指向 SomeBaseClass 的 intrusive_ptr
  intrusive_ptr<SomeBaseClass> base = child;
  // 断言 base 指向的对象的 v 值为 3
  EXPECT_EQ(3, base->v);
  // 断言 base 与 child 指向相同对象
  EXPECT_EQ(objptr, base.get());
}

TEST(
    IntrusivePtrTest,
    givenPtr_whenCopyConstructingToBaseClass_thenOldInstanceInvalid) {
  // 创建一个指向 SomeChildClass 对象的 intrusive_ptr
  intrusive_ptr<SomeChildClass> child = make_intrusive<SomeChildClass>(3);
  // 将 child 拷贝构造为指向 SomeBaseClass 的 intrusive_ptr
  intrusive_ptr<SomeBaseClass> base = child;
  // 断言 child 是有效的（defined）
  EXPECT_TRUE(child.defined());
}

TEST(
    IntrusivePtrTest,
    givenPtr_whenCopyConstructingToBaseClass_thenNewInstanceInvalid) {
  // 创建一个指向 SomeChildClass 对象的 intrusive_ptr
  intrusive_ptr<SomeChildClass> child = make_intrusive<SomeChildClass>(3);
  // 将 child 拷贝构造为指向 SomeBaseClass 的 intrusive_ptr
  intrusive_ptr<SomeBaseClass> base = child;
  // 断言 base 是有效的（defined）
  EXPECT_TRUE(base.defined());
}

TEST(
    IntrusivePtrTest,
    givenPtr_whenCopyConstructingToBaseClassFromInvalidPtr_thenNewInstanceInvalid) {
  // 创建一个空的 intrusive_ptr<SomeChildClass>
  intrusive_ptr<SomeChildClass> obj1;
  // 将 obj1 拷贝构造为指向 SomeBaseClass 的 intrusive_ptr
  intrusive_ptr<SomeBaseClass> obj2 = obj1;
  // 断言 obj2 是无效的（未定义）
  EXPECT_FALSE(obj2.defined());
}

TEST(
    IntrusivePtrTest,
    givenNullPtr_whenCopyConstructingToDifferentNullptr_thenHasNewNullptr) {
  // 创建一个空的 intrusive_ptr<SomeClass, NullType1>
  intrusive_ptr<SomeClass, NullType1> obj1;
  // 将 obj1 拷贝构造为 intrusive_ptr<SomeClass, NullType2>
  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
  intrusive_ptr<SomeClass, NullType2> obj2 = obj1;
  // 断言 NullType1 的 singleton 与 obj1.get() 不相等
  EXPECT_NE(NullType1::singleton(), obj1.get());
  // 断言 NullType1 的 singleton 与 obj2.get() 相等
  EXPECT_EQ(NullType1::singleton(), obj1.get());
  // 断言 NullType2 的 singleton 与 obj2.get() 相等
  EXPECT_EQ(NullType2::singleton(), obj2.get());
  // 断言 obj1 是无效的（未定义）
  EXPECT_FALSE(obj1.defined());
  // 断言 obj2 是无效的（未定义）
  EXPECT_FALSE(obj2.defined());
}

TEST(IntrusivePtrTest, SwapFunction) {
  // 创建两个指向 SomeClass 对象的 intrusive_ptr
  intrusive_ptr<SomeClass> obj1 = make_intrusive<SomeClass>();
  intrusive_ptr<SomeClass> obj2 = make_intrusive<SomeClass>();
  // 获取 obj1 和 obj2 的指针
  SomeClass* obj1ptr = obj1.get();
  SomeClass* obj2ptr = obj2.get();
  // 交换 obj1 和 obj2 的内容
  swap(obj1, obj2);
  // 断言交换后 obj1 指向的对象与原来的 obj2 相同
  EXPECT_EQ(obj2ptr, obj1.get());
  // 断言交换后 obj2 指向的对象与原来的 obj1 相同
  EXPECT_EQ(obj1ptr, obj2.get());
}
TEST(IntrusivePtrTest, SwapMethod) {
  // 创建两个指向 SomeClass 对象的智能指针 obj1 和 obj2
  intrusive_ptr<SomeClass> obj1 = make_intrusive<SomeClass>();
  intrusive_ptr<SomeClass> obj2 = make_intrusive<SomeClass>();
  // 获取 obj1 和 obj2 的原始指针
  SomeClass* obj1ptr = obj1.get();
  SomeClass* obj2ptr = obj2.get();
  // 交换 obj1 和 obj2 的指针
  obj1.swap(obj2);
  // 检查交换后 obj1 和 obj2 的指针是否正确
  EXPECT_EQ(obj2ptr, obj1.get());
  EXPECT_EQ(obj1ptr, obj2.get());
}

TEST(IntrusivePtrTest, SwapFunctionFromInvalid) {
  // 创建一个空的智能指针 obj1 和一个指向 SomeClass 对象的智能指针 obj2
  intrusive_ptr<SomeClass> obj1;
  intrusive_ptr<SomeClass> obj2 = make_intrusive<SomeClass>();
  // 获取 obj2 的原始指针
  SomeClass* obj2ptr = obj2.get();
  // 使用 swap 函数交换 obj1 和 obj2 的指针
  swap(obj1, obj2);
  // 检查交换后 obj1 和 obj2 的指针是否正确，以及 obj1 是否已定义，obj2 是否未定义
  EXPECT_EQ(obj2ptr, obj1.get());
  EXPECT_TRUE(obj1.defined());
  EXPECT_FALSE(obj2.defined());
}

TEST(IntrusivePtrTest, SwapMethodFromInvalid) {
  // 创建一个空的智能指针 obj1 和一个指向 SomeClass 对象的智能指针 obj2
  intrusive_ptr<SomeClass> obj1;
  intrusive_ptr<SomeClass> obj2 = make_intrusive<SomeClass>();
  // 获取 obj2 的原始指针
  SomeClass* obj2ptr = obj2.get();
  // 使用 obj1 的 swap 方法与 obj2 交换指针
  obj1.swap(obj2);
  // 检查交换后 obj1 和 obj2 的指针是否正确，以及 obj1 是否已定义，obj2 是否未定义
  EXPECT_EQ(obj2ptr, obj1.get());
  EXPECT_TRUE(obj1.defined());
  EXPECT_FALSE(obj2.defined());
}

TEST(IntrusivePtrTest, SwapFunctionWithInvalid) {
  // 创建一个指向 SomeClass 对象的智能指针 obj1 和一个空的智能指针 obj2
  intrusive_ptr<SomeClass> obj1 = make_intrusive<SomeClass>();
  intrusive_ptr<SomeClass> obj2;
  // 获取 obj1 的原始指针
  SomeClass* obj1ptr = obj1.get();
  // 使用 swap 函数交换 obj1 和 obj2 的指针
  swap(obj1, obj2);
  // 检查交换后 obj1 和 obj2 的指针是否正确，以及 obj1 是否未定义，obj2 是否已定义
  EXPECT_FALSE(obj1.defined());
  EXPECT_TRUE(obj2.defined());
  EXPECT_EQ(obj1ptr, obj2.get());
}

TEST(IntrusivePtrTest, SwapMethodWithInvalid) {
  // 创建一个指向 SomeClass 对象的智能指针 obj1 和一个空的智能指针 obj2
  intrusive_ptr<SomeClass> obj1 = make_intrusive<SomeClass>();
  intrusive_ptr<SomeClass> obj2;
  // 获取 obj1 的原始指针
  SomeClass* obj1ptr = obj1.get();
  // 使用 obj1 的 swap 方法与 obj2 交换指针
  obj1.swap(obj2);
  // 检查交换后 obj1 和 obj2 的指针是否正确，以及 obj1 是否未定义，obj2 是否已定义
  EXPECT_FALSE(obj1.defined());
  EXPECT_TRUE(obj2.defined());
  EXPECT_EQ(obj1ptr, obj2.get());
}

TEST(IntrusivePtrTest, SwapFunctionInvalidWithInvalid) {
  // 创建两个空的智能指针 obj1 和 obj2
  intrusive_ptr<SomeClass> obj1;
  intrusive_ptr<SomeClass> obj2;
  // 使用 swap 函数交换 obj1 和 obj2 的指针
  swap(obj1, obj2);
  // 检查交换后 obj1 和 obj2 的指针是否正确，均为未定义
  EXPECT_FALSE(obj1.defined());
  EXPECT_FALSE(obj2.defined());
}

TEST(IntrusivePtrTest, SwapMethodInvalidWithInvalid) {
  // 创建两个空的智能指针 obj1 和 obj2
  intrusive_ptr<SomeClass> obj1;
  intrusive_ptr<SomeClass> obj2;
  // 使用 obj1 的 swap 方法与 obj2 交换指针
  obj1.swap(obj2);
  // 检查交换后 obj1 和 obj2 的指针是否正确，均为未定义
  EXPECT_FALSE(obj1.defined());
  EXPECT_FALSE(obj2.defined());
}

TEST(IntrusivePtrTest, CanBePutInContainer) {
  // 创建一个存储 intrusive_ptr<SomeClass1Parameter> 的向量 vec
  std::vector<intrusive_ptr<SomeClass1Parameter>> vec;
  // 将一个带参数的 SomeClass1Parameter 对象的智能指针插入到 vec 中
  vec.push_back(make_intrusive<SomeClass1Parameter>(5));
  // 检查插入的对象参数是否正确
  EXPECT_EQ(5, vec[0]->param);
}

TEST(IntrusivePtrTest, CanBePutInSet) {
  // 创建一个存储 intrusive_ptr<SomeClass1Parameter> 的集合 set
  std::set<intrusive_ptr<SomeClass1Parameter>> set;
  // 将一个带参数的 SomeClass1Parameter 对象的智能指针插入到 set 中
  set.insert(make_intrusive<SomeClass1Parameter>(5));
  // 检查插入的对象参数是否正确
  EXPECT_EQ(5, (*set.begin())->param);
}

TEST(IntrusivePtrTest, CanBePutInUnorderedSet) {
  // 创建一个存储 intrusive_ptr<SomeClass1Parameter> 的无序集合 set
  std::unordered_set<intrusive_ptr<SomeClass1Parameter>> set;
  // 将一个带参数的 SomeClass1Parameter 对象的智能指针插入到 set 中
  set.insert(make_intrusive<SomeClass1Parameter>(5));
  // 检查插入的对象参数是否正确
  EXPECT_EQ(5, (*set.begin())->param);
}

TEST(IntrusivePtrTest, CanBePutInMap) {
  // 创建一个存储 intrusive_ptr<SomeClass1Parameter> 对象对的映射 map
  std::map<
      intrusive_ptr<SomeClass1Parameter>,
      intrusive_ptr<SomeClass1Parameter>>
      map;
  // 将一对带参数的 SomeClass1Parameter 对象的智能指针插入到 map 中
  map.insert(std::make_pair(
      make_intrusive<SomeClass1Parameter>(5),
      make_intrusive<SomeClass1Parameter>(3)));
  // 检查插入的对象参数是否正确
  EXPECT_EQ(5, map.begin()->first->param);
  EXPECT_EQ(3, map.begin()->second->param);
}
// 测试用例：IntrusivePtrTest，测试插入到无序映射中的可侵入指针对象
TEST(IntrusivePtrTest, CanBePutInUnorderedMap) {
  // 声明一个无序映射，键和值都是 intrusive_ptr<SomeClass1Parameter> 类型的对象
  std::unordered_map<
      intrusive_ptr<SomeClass1Parameter>,
      intrusive_ptr<SomeClass1Parameter>>
      map;
  // 向映射中插入一对键值对，使用 make_intrusive 创建 intrusive_ptr 对象
  map.insert(std::make_pair(
      make_intrusive<SomeClass1Parameter>(3),
      make_intrusive<SomeClass1Parameter>(5)));
  // 验证插入后第一个键的参数值是否为 3
  EXPECT_EQ(3, map.begin()->first->param);
  // 验证插入后第一个值的参数值是否为 5
  EXPECT_EQ(5, map.begin()->second->param);
}

// 测试用例：IntrusivePtrTest，测试复制构造后的相等性
TEST(IntrusivePtrTest, Equality_AfterCopyConstructor) {
  // 创建 intrusive_ptr<SomeClass> 对象 var1
  intrusive_ptr<SomeClass> var1 = make_intrusive<SomeClass>();
  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)：禁止性能不必要的复制初始化警告
  // 创建 intrusive_ptr<SomeClass> 对象 var2，并使用 var1 进行复制构造
  intrusive_ptr<SomeClass> var2 = var1;
  // 验证 var1 和 var2 是否相等
  EXPECT_TRUE(var1 == var2);
  // 验证 var1 和 var2 是否不不相等
  EXPECT_FALSE(var1 != var2);
}

// 测试用例：IntrusivePtrTest，测试复制赋值后的相等性
TEST(IntrusivePtrTest, Equality_AfterCopyAssignment) {
  // 创建 intrusive_ptr<SomeClass> 对象 var1
  intrusive_ptr<SomeClass> var1 = make_intrusive<SomeClass>();
  // 创建 intrusive_ptr<SomeClass> 对象 var2
  intrusive_ptr<SomeClass> var2 = make_intrusive<SomeClass>();
  // 使用 var1 进行复制赋值给 var2
  var2 = var1;
  // 验证 var1 和 var2 是否相等
  EXPECT_TRUE(var1 == var2);
  // 验证 var1 和 var2 是否不不相等
  EXPECT_FALSE(var1 != var2);
}

// 测试用例：IntrusivePtrTest，测试空指针相等性
TEST(IntrusivePtrTest, Equality_Nullptr) {
  // 声明两个空的 intrusive_ptr<SomeClass> 对象 var1 和 var2
  intrusive_ptr<SomeClass> var1;
  intrusive_ptr<SomeClass> var2;
  // 验证 var1 和 var2 是否相等
  EXPECT_TRUE(var1 == var2);
  // 验证 var1 和 var2 是否不不相等
  EXPECT_FALSE(var1 != var2);
}

// 测试用例：IntrusivePtrTest，测试不相等性
TEST(IntrusivePtrTest, Inequality) {
  // 创建两个不同的 intrusive_ptr<SomeClass> 对象 var1 和 var2
  intrusive_ptr<SomeClass> var1 = make_intrusive<SomeClass>();
  intrusive_ptr<SomeClass> var2 = make_intrusive<SomeClass>();
  // 验证 var1 和 var2 是否不相等
  EXPECT_TRUE(var1 != var2);
  // 验证 var1 和 var2 是否相等
  EXPECT_FALSE(var1 == var2);
}

// 测试用例：IntrusivePtrTest，测试左边为 nullptr 的不相等性
TEST(IntrusivePtrTest, Inequality_NullptrLeft) {
  // 声明一个空的 intrusive_ptr<SomeClass> 对象 var1 和一个非空的 var2
  intrusive_ptr<SomeClass> var1;
  intrusive_ptr<SomeClass> var2 = make_intrusive<SomeClass>();
  // 验证 var1 和 var2 是否不相等
  EXPECT_TRUE(var1 != var2);
  // 验证 var1 和 var2 是否相等
  EXPECT_FALSE(var1 == var2);
}

// 测试用例：IntrusivePtrTest，测试右边为 nullptr 的不相等性
TEST(IntrusivePtrTest, Inequality_NullptrRight) {
  // 声明一个非空的 intrusive_ptr<SomeClass> 对象 var1 和一个空的 var2
  intrusive_ptr<SomeClass> var1 = make_intrusive<SomeClass>();
  intrusive_ptr<SomeClass> var2;
  // 验证 var1 和 var2 是否不相等
  EXPECT_TRUE(var1 != var2);
  // 验证 var1 和 var2 是否相等
  EXPECT_FALSE(var1 == var2);
}

// 测试用例：IntrusivePtrTest，测试哈希值不同
TEST(IntrusivePtrTest, HashIsDifferent) {
  // 创建两个不同的 intrusive_ptr<SomeClass> 对象 var1 和 var2
  intrusive_ptr<SomeClass> var1 = make_intrusive<SomeClass>();
  intrusive_ptr<SomeClass> var2 = make_intrusive<SomeClass>();
  // 验证 var1 和 var2 的哈希值是否不相等
  EXPECT_NE(
      std::hash<intrusive_ptr<SomeClass>>()(var1),
      std::hash<intrusive_ptr<SomeClass>>()(var2));
}

// 测试用例：IntrusivePtrTest，测试有效和无效对象的哈希值不同
TEST(IntrusivePtrTest, HashIsDifferent_ValidAndInvalid) {
  // 声明一个空的 intrusive_ptr<SomeClass> 对象 var1 和一个非空的 var2
  intrusive_ptr<SomeClass> var1;
  intrusive_ptr<SomeClass> var2 = make_intrusive<SomeClass>();
  // 验证 var1 和 var2 的哈希值是否不相等
  EXPECT_NE(
      std::hash<intrusive_ptr<SomeClass>>()(var1),
      std::hash<intrusive_ptr<SomeClass>>()(var2));
}

// 测试用例：IntrusivePtrTest，测试复制构造后的哈希值相同
TEST(IntrusivePtrTest, HashIsSame_AfterCopyConstructor) {
  // 创建 intrusive_ptr<SomeClass> 对象 var1
  intrusive_ptr<SomeClass> var1 = make_intrusive<SomeClass>();
  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)：禁止性能不必要的复制初始化警告
  // 创建 intrusive_ptr<SomeClass> 对象 var2，并使用 var1 进行复制构造
  intrusive_ptr<SomeClass> var2 = var1;
  // 验证 var1 和 var2 的哈希值是否相等
  EXPECT_EQ(
      std::hash<intrusive_ptr<SomeClass>>()(var1),
      std::hash<intrusive_ptr<SomeClass>>()(var2));
}

// 测试用例：IntrusivePtrTest，测试复制赋值后的哈希值相同
TEST(IntrusivePtrTest, HashIsSame_AfterCopyAssignment) {
  // 创建 intrusive_ptr<SomeClass> 对象 var1
  intrusive_ptr<SomeClass> var1 = make_intrusive<SomeClass>();
  // 创建 intrusive_ptr<SomeClass> 对象 var2
  intrusive_ptr<SomeClass> var2 = make_intrusive<SomeClass>();
  // 使用 var1 进行复制赋值给 var2
  var2 = var1;
  // 验证 var1 和 var2 的哈希值是否相等
  EXPECT_EQ(
      std::hash<intrusive_ptr<SomeClass>>()(var1),
      std::hash<intrusive_ptr<SomeClass>>()(var2));
}
TEST(IntrusivePtrTest, HashIsSame_BothNullptr) {
  // 创建两个空的intrusive_ptr对象
  intrusive_ptr<SomeClass> var1;
  intrusive_ptr<SomeClass> var2;
  // 断言它们的哈希值相同
  EXPECT_EQ(
      std::hash<intrusive_ptr<SomeClass>>()(var1),
      std::hash<intrusive_ptr<SomeClass>>()(var2));
}

TEST(IntrusivePtrTest, OneIsLess) {
  // 创建两个不同的intrusive_ptr对象
  intrusive_ptr<SomeClass> var1 = make_intrusive<SomeClass>();
  intrusive_ptr<SomeClass> var2 = make_intrusive<SomeClass>();
  // 断言var1小于var2，同时var2不小于var1（使用std::less进行比较）
  EXPECT_TRUE(
      // NOLINTNEXTLINE(modernize-use-transparent-functors)
      std::less<intrusive_ptr<SomeClass>>()(var1, var2) !=
      // NOLINTNEXTLINE(modernize-use-transparent-functors)
      std::less<intrusive_ptr<SomeClass>>()(var2, var1));
}

TEST(IntrusivePtrTest, NullptrIsLess1) {
  // 创建一个空的intrusive_ptr对象和一个非空的intrusive_ptr对象
  intrusive_ptr<SomeClass> var1;
  intrusive_ptr<SomeClass> var2 = make_intrusive<SomeClass>();
  // 断言空指针小于非空指针（使用std::less进行比较）
  // NOLINTNEXTLINE(modernize-use-transparent-functors)
  EXPECT_TRUE(std::less<intrusive_ptr<SomeClass>>()(var1, var2));
}

TEST(IntrusivePtrTest, NullptrIsLess2) {
  // 创建一个非空的intrusive_ptr对象和一个空的intrusive_ptr对象
  intrusive_ptr<SomeClass> var1 = make_intrusive<SomeClass>();
  intrusive_ptr<SomeClass> var2;
  // 断言空指针不小于非空指针（使用std::less进行比较）
  // NOLINTNEXTLINE(modernize-use-transparent-functors)
  EXPECT_FALSE(std::less<intrusive_ptr<SomeClass>>()(var1, var2));
}

TEST(IntrusivePtrTest, NullptrIsNotLessThanNullptr) {
  // 创建两个空的intrusive_ptr对象
  intrusive_ptr<SomeClass> var1;
  intrusive_ptr<SomeClass> var2;
  // 断言空指针不小于空指针（使用std::less进行比较）
  // NOLINTNEXTLINE(modernize-use-transparent-functors)
  EXPECT_FALSE(std::less<intrusive_ptr<SomeClass>>()(var1, var2));
}

TEST(IntrusivePtrTest, givenPtr_whenCallingReset_thenIsInvalid) {
  // 创建一个intrusive_ptr对象，并初始化为指向SomeClass对象
  auto obj = make_intrusive<SomeClass>();
  // 断言对象是有效的
  EXPECT_TRUE(obj.defined());
  // 调用reset方法，重置对象
  obj.reset();
  // 断言对象现在无效
  EXPECT_FALSE(obj.defined());
}

TEST(IntrusivePtrTest, givenPtr_whenCallingReset_thenHoldsNullptr) {
  // 创建一个intrusive_ptr对象，并初始化为指向SomeClass对象
  auto obj = make_intrusive<SomeClass>();
  // 断言对象指针不为nullptr
  EXPECT_NE(nullptr, obj.get());
  // 调用reset方法，重置对象
  obj.reset();
  // 断言对象指针现在为nullptr
  EXPECT_EQ(nullptr, obj.get());
}

TEST(IntrusivePtrTest, givenPtr_whenDestructed_thenDestructsObject) {
  // 声明两个标志位
  bool resourcesReleased = false;
  bool wasDestructed = false;
  {
    // 创建一个intrusive_ptr对象，指向DestructableMock对象
    auto obj =
        make_intrusive<DestructableMock>(&resourcesReleased, &wasDestructed);
    // 断言资源未被释放，对象未被析构
    EXPECT_FALSE(resourcesReleased);
    EXPECT_FALSE(wasDestructed);
  }
  // 断言资源已被释放，对象已被析构
  EXPECT_TRUE(resourcesReleased);
  EXPECT_TRUE(wasDestructed);
}

TEST(
    IntrusivePtrTest,
    givenPtr_whenMoveConstructed_thenDestructsObjectAfterSecondDestructed) {
  // 声明两个标志位
  bool resourcesReleased = false;
  bool wasDestructed = false;
  // 创建一个intrusive_ptr对象，指向DestructableMock对象
  auto obj =
      make_intrusive<DestructableMock>(&resourcesReleased, &wasDestructed);
  {
    // 使用move构造函数创建另一个intrusive_ptr对象
    auto obj2 = std::move(obj);
    // 断言资源未被释放，对象未被析构
    EXPECT_FALSE(resourcesReleased);
    EXPECT_FALSE(wasDestructed);
  }
  // 断言资源已被释放，对象已被析构
  EXPECT_TRUE(resourcesReleased);
  EXPECT_TRUE(wasDestructed);
}

TEST(
    IntrusivePtrTest,
    givenPtr_whenMoveConstructedToBaseClass_thenDestructsObjectAfterSecondDestructed) {
  // 声明两个标志位
  bool resourcesReleased = false;
  bool wasDestructed = false;
  // 创建一个intrusive_ptr对象，指向ChildDestructableMock对象（继承自DestructableMock）
  auto obj =
      make_intrusive<ChildDestructableMock>(&resourcesReleased, &wasDestructed);
  {
    // 使用move构造函数创建一个指向基类DestructableMock的intrusive_ptr对象
    intrusive_ptr<DestructableMock> obj2 = std::move(obj);
    // 断言资源未被释放
    EXPECT_FALSE(resourcesReleased);
    # 检查变量 `wasDestructed` 的值是否为假
    EXPECT_FALSE(wasDestructed);
    # 断言所有资源已被释放
    EXPECT_TRUE(resourcesReleased);
    # 断言变量 `wasDestructed` 的值为真
    EXPECT_TRUE(wasDestructed);
}

TEST(IntrusivePtrTest, givenPtr_whenMoveAssigned_thenDestructsOldObject) {
  bool dummy = false;  // 创建一个布尔变量 dummy，并初始化为 false
  bool resourcesReleased = false;  // 创建一个布尔变量 resourcesReleased，并初始化为 false
  bool wasDestructed = false;  // 创建一个布尔变量 wasDestructed，并初始化为 false
  auto obj = make_intrusive<DestructableMock>(&dummy, &dummy);  // 使用 make_intrusive 创建一个 DestructableMock 对象，并传入两个指向 dummy 的指针
  {
    auto obj2 =  // 创建一个名为 obj2 的 auto 变量
        make_intrusive<DestructableMock>(&resourcesReleased, &wasDestructed);  // 使用 make_intrusive 创建另一个 DestructableMock 对象，并传入 resourcesReleased 和 wasDestructed 的地址
    EXPECT_FALSE(resourcesReleased);  // 断言 resourcesReleased 为 false
    EXPECT_FALSE(wasDestructed);  // 断言 wasDestructed 为 false
    obj2 = std::move(obj);  // 将 obj 移动给 obj2
    EXPECT_TRUE(resourcesReleased);  // 断言 resourcesReleased 为 true
    EXPECT_TRUE(wasDestructed);  // 断言 wasDestructed 为 true
  }
}

TEST(
    IntrusivePtrTest,
    givenPtr_whenMoveAssignedToBaseClass_thenDestructsOldObject) {
  bool dummy = false;  // 创建一个布尔变量 dummy，并初始化为 false
  bool resourcesReleased = false;  // 创建一个布尔变量 resourcesReleased，并初始化为 false
  bool wasDestructed = false;  // 创建一个布尔变量 wasDestructed，并初始化为 false
  auto obj = make_intrusive<ChildDestructableMock>(&dummy, &dummy);  // 使用 make_intrusive 创建一个 ChildDestructableMock 对象，并传入两个指向 dummy 的指针
  {
    auto obj2 =  // 创建一个名为 obj2 的 auto 变量
        make_intrusive<DestructableMock>(&resourcesReleased, &wasDestructed);  // 使用 make_intrusive 创建一个 DestructableMock 对象，并传入 resourcesReleased 和 wasDestructed 的地址
    EXPECT_FALSE(resourcesReleased);  // 断言 resourcesReleased 为 false
    EXPECT_FALSE(wasDestructed);  // 断言 wasDestructed 为 false
    obj2 = std::move(obj);  // 将 obj 移动给 obj2
    EXPECT_TRUE(resourcesReleased);  // 断言 resourcesReleased 为 true
    EXPECT_TRUE(wasDestructed);  // 断言 wasDestructed 为 true
  }
}

TEST(
    IntrusivePtrTest,
    givenPtrWithCopy_whenMoveAssigned_thenDestructsOldObjectAfterCopyIsDestructed) {
  bool dummy = false;  // 创建一个布尔变量 dummy，并初始化为 false
  bool resourcesReleased = false;  // 创建一个布尔变量 resourcesReleased，并初始化为 false
  bool wasDestructed = false;  // 创建一个布尔变量 wasDestructed，并初始化为 false
  auto obj = make_intrusive<DestructableMock>(&dummy, &dummy);  // 使用 make_intrusive 创建一个 DestructableMock 对象，并传入两个指向 dummy 的指针
  {
    auto obj2 =  // 创建一个名为 obj2 的 auto 变量
        make_intrusive<DestructableMock>(&resourcesReleased, &wasDestructed);  // 使用 make_intrusive 创建一个 DestructableMock 对象，并传入 resourcesReleased 和 wasDestructed 的地址
    {
      auto copy = obj2;  // 创建一个名为 copy 的 auto 变量，并将 obj2 复制给它
      EXPECT_FALSE(resourcesReleased);  // 断言 resourcesReleased 为 false
      EXPECT_FALSE(wasDestructed);  // 断言 wasDestructed 为 false
      obj2 = std::move(obj);  // 将 obj 移动给 obj2
      EXPECT_FALSE(resourcesReleased);  // 断言 resourcesReleased 为 false
      EXPECT_FALSE(wasDestructed);  // 断言 wasDestructed 为 false
    }
    EXPECT_TRUE(resourcesReleased);  // 断言 resourcesReleased 为 true
    EXPECT_TRUE(wasDestructed);  // 断言 wasDestructed 为 true
  }
}

TEST(
    IntrusivePtrTest,
    givenPtrWithBaseClassCopy_whenMoveAssigned_thenDestructsOldObjectAfterCopyIsDestructed) {
  bool dummy = false;  // 创建一个布尔变量 dummy，并初始化为 false
  bool resourcesReleased = false;  // 创建一个布尔变量 resourcesReleased，并初始化为 false
  bool wasDestructed = false;  // 创建一个布尔变量 wasDestructed，并初始化为 false
  auto obj = make_intrusive<ChildDestructableMock>(&dummy, &dummy);  // 使用 make_intrusive 创建一个 ChildDestructableMock 对象，并传入两个指向 dummy 的指针
  {
    auto obj2 = make_intrusive<ChildDestructableMock>(  // 创建一个名为 obj2 的 auto 变量，并使用 make_intrusive 创建一个 ChildDestructableMock 对象
        &resourcesReleased, &wasDestructed);
    {
      intrusive_ptr<DestructableMock> copy = obj2;  // 创建一个 intrusive_ptr<DestructableMock> 类型的 copy，并将 obj2 复制给它
      EXPECT_FALSE(resourcesReleased);  // 断言 resourcesReleased 为 false
      EXPECT_FALSE(wasDestructed);  // 断言 wasDestructed 为 false
      obj2 = std::move(obj);  // 将 obj 移动给 obj2
      EXPECT_FALSE(resourcesReleased);  // 断言 resourcesReleased 为 false
      EXPECT_FALSE(wasDestructed);  // 断言 wasDestructed 为 false
    }
    EXPECT_TRUE(resourcesReleased);  // 断言 resourcesReleased 为 true
    EXPECT_TRUE(wasDestructed);  // 断言 wasDestructed 为 true
  }
}

TEST(
    IntrusivePtrTest,
    givenPtrWithCopy_whenMoveAssignedToBaseClass_thenDestructsOldObjectAfterCopyIsDestructed) {
  bool dummy = false;  // 创建一个布尔变量 dummy，并初始化为 false
  bool resourcesReleased = false;  // 创建一个布尔变量 resourcesReleased，并初始化为 false
  bool wasDestructed = false;  // 创建一个布尔变量 wasDestructed，并初始化为 false
  auto obj = make_intrusive<ChildDestructableMock>(&dummy, &dummy);  // 使用 make_intrusive 创建一个 ChildDestructableMock 对象，并传入两个指向 dummy 的指针
  {
    auto obj2 =  // 创建一个名为 obj2 的 auto 变量
        make_intrusive<DestructableMock>(&resourcesReleased, &wasDestructed);  // 使用 make_intrusive 创建一个 DestructableMock 对象，并传入 resourcesReleased 和 wasDestructed 的地址
    {
      intrusive_ptr<DestructableMock> copy = obj2;  // 创建一个 intrusive_ptr<DestructableMock> 类型的 copy，并将 obj2 复制给它
      EXPECT_FALSE(resourcesReleased);  // 断言 resourcesReleased 为 false
      EXPECT_FALSE(wasDestructed);  // 断言 wasDestructed 为 false
      obj2 = std::move(obj);  // 将 obj 移动给 obj2
      EXPECT_FALSE(resourcesReleased);  // 断言 resourcesReleased 为 false
      EXPECT_FALSE(wasDestructed);  // 断言 wasDestructed 为 false
    }
    # 断言资源已释放为真
    EXPECT_TRUE(resourcesReleased);
    # 断言已析构为真
    EXPECT_TRUE(wasDestructed);
TEST(
    IntrusivePtrTest,
    givenPtr_whenMoveAssigned_thenDestructsObjectAfterSecondDestructed) {
  // 设置一个虚拟的布尔变量
  bool dummy = false;
  // 检查资源是否已释放
  bool resourcesReleased = false;
  // 检查对象是否已析构
  bool wasDestructed = false;
  // 创建一个 IntrusivePtr 智能指针，指向一个 DestructableMock 对象
  auto obj =
      make_intrusive<DestructableMock>(&resourcesReleased, &wasDestructed);
  {
    // 创建另一个 IntrusivePtr 智能指针，指向另一个 DestructableMock 对象
    auto obj2 = make_intrusive<DestructableMock>(&dummy, &dummy);
    // 将 obj2 指向的对象移动给 obj
    obj2 = std::move(obj);
    // 断言：资源未被释放
    EXPECT_FALSE(resourcesReleased);
    // 断言：对象未被析构
    EXPECT_FALSE(wasDestructed);
  }
  // 断言：资源已被释放
  EXPECT_TRUE(resourcesReleased);
  // 断言：对象已被析构
  EXPECT_TRUE(wasDestructed);
}

TEST(
    IntrusivePtrTest,
    givenPtr_whenMoveAssignedToBaseClass_thenDestructsObjectAfterSecondDestructed) {
  // 设置一个虚拟的布尔变量
  bool dummy = false;
  // 检查资源是否已释放
  bool resourcesReleased = false;
  // 检查对象是否已析构
  bool wasDestructed = false;
  // 创建一个 IntrusivePtr 智能指针，指向一个 ChildDestructableMock 对象
  auto obj =
      make_intrusive<ChildDestructableMock>(&resourcesReleased, &wasDestructed);
  {
    // 创建另一个 IntrusivePtr 智能指针，指向另一个 DestructableMock 对象
    auto obj2 = make_intrusive<DestructableMock>(&dummy, &dummy);
    // 将 obj2 指向的对象移动给 obj
    obj2 = std::move(obj);
    // 断言：资源未被释放
    EXPECT_FALSE(resourcesReleased);
    // 断言：对象未被析构
    EXPECT_FALSE(wasDestructed);
  }
  // 断言：资源已被释放
  EXPECT_TRUE(resourcesReleased);
  // 断言：对象已被析构
  EXPECT_TRUE(wasDestructed);
}

TEST(
    IntrusivePtrTest,
    givenPtr_whenCopyConstructedAndDestructed_thenDestructsObjectAfterLastDestruction) {
  // 检查资源是否已释放
  bool resourcesReleased = false;
  // 检查对象是否已析构
  bool wasDestructed = false;
  {
    // 创建一个 IntrusivePtr 智能指针，指向一个 DestructableMock 对象
    auto obj =
        make_intrusive<DestructableMock>(&resourcesReleased, &wasDestructed);
    {
      // 创建一个新的 IntrusivePtr 智能指针，复制 obj 的内容
      intrusive_ptr<DestructableMock> copy = obj;
      // 断言：资源未被释放
      EXPECT_FALSE(resourcesReleased);
      // 断言：对象未被析构
      EXPECT_FALSE(wasDestructed);
    }
    // 断言：资源未被释放
    EXPECT_FALSE(resourcesReleased);
    // 断言：对象未被析构
    EXPECT_FALSE(wasDestructed);
  }
  // 断言：资源已被释放
  EXPECT_TRUE(resourcesReleased);
  // 断言：对象已被析构
  EXPECT_TRUE(wasDestructed);
}

TEST(
    IntrusivePtrTest,
    givenPtr_whenCopyConstructedToBaseClassAndDestructed_thenDestructsObjectAfterLastDestruction) {
  // 检查资源是否已释放
  bool resourcesReleased = false;
  // 检查对象是否已析构
  bool wasDestructed = false;
  {
    // 创建一个 IntrusivePtr 智能指针，指向一个 ChildDestructableMock 对象
    auto obj = make_intrusive<ChildDestructableMock>(
        &resourcesReleased, &wasDestructed);
    {
      // 创建一个新的 IntrusivePtr 智能指针，复制 obj 的内容
      intrusive_ptr<DestructableMock> copy = obj;
      // 断言：资源未被释放
      EXPECT_FALSE(resourcesReleased);
      // 断言：对象未被析构
      EXPECT_FALSE(wasDestructed);
    }
    // 断言：资源未被释放
    EXPECT_FALSE(resourcesReleased);
    // 断言：对象未被析构
    EXPECT_FALSE(wasDestructed);
  }
  // 断言：资源已被释放
  EXPECT_TRUE(resourcesReleased);
  // 断言：对象已被析构
  EXPECT_TRUE(wasDestructed);
}

TEST(
    IntrusivePtrTest,
    givenPtr_whenCopyConstructedAndOriginalDestructed_thenDestructsObjectAfterLastDestruction) {
  // 检查资源是否已释放
  bool resourcesReleased = false;
  // 检查对象是否已析构
  bool wasDestructed = false;
  {
    // 创建一个 IntrusivePtr 智能指针，指向一个 DestructableMock 对象
    auto obj =
        make_intrusive<DestructableMock>(&resourcesReleased, &wasDestructed);
    // 复制 obj 的内容到另一个 IntrusivePtr 智能指针
    intrusive_ptr<DestructableMock> copy = obj;
    // 重置 obj，使其不再指向对象
    obj.reset();
    // 断言：资源未被释放
    EXPECT_FALSE(resourcesReleased);
    // 断言：对象未被析构
    EXPECT_FALSE(wasDestructed);
  }
  // 断言：资源已被释放
  EXPECT_TRUE(resourcesReleased);
  // 断言：对象已被析构
  EXPECT_TRUE(wasDestructed);
}

TEST(
    IntrusivePtrTest,
    givenPtr_whenCopyConstructedToBaseClassAndOriginalDestructed_thenDestructsObjectAfterLastDestruction) {
  // 检查资源是否已释放
  bool resourcesReleased = false;
  // 检查对象是否已析构
  bool wasDestructed = false;
  {
    # 使用 make_intrusive 创建一个指向 ChildDestructableMock 对象的智能指针 obj，
    # 并通过引用传递 resourcesReleased 和 wasDestructed 的变量地址进行初始化
    auto obj = make_intrusive<ChildDestructableMock>(
        &resourcesReleased, &wasDestructed);

    # 创建一个指向 DestructableMock 对象的 intrusive_ptr 智能指针 copy，
    # 并将其指向与 obj 相同的对象，增加其引用计数
    intrusive_ptr<DestructableMock> copy = obj;

    # 释放 obj 指向的对象，预期 resourcesReleased 和 wasDestructed 均为 false
    obj.reset();
    EXPECT_FALSE(resourcesReleased);
    EXPECT_FALSE(wasDestructed);
  }

  # 验证资源已被释放，即 resourcesReleased 为 true
  EXPECT_TRUE(resourcesReleased);

  # 验证对象已被析构，即 wasDestructed 为 true
  EXPECT_TRUE(wasDestructed);
TEST(
    IntrusivePtrTest,
    givenPtr_whenCopyAssigned_thenDestructsOldObject) {
  bool dummy = false;  // 创建一个布尔变量dummy，并初始化为false
  bool resourcesReleased = false;  // 创建一个布尔变量resourcesReleased，并初始化为false
  bool wasDestructed = false;  // 创建一个布尔变量wasDestructed，并初始化为false
  auto obj = make_intrusive<DestructableMock>(&dummy, &dummy);  // 创建一个DestructableMock类型的智能指针obj，并传入dummy的地址作为参数
  {
    auto obj2 =
        make_intrusive<DestructableMock>(&resourcesReleased, &wasDestructed);  // 创建另一个DestructableMock类型的智能指针obj2，并传入resourcesReleased和wasDestructed的地址作为参数
    EXPECT_FALSE(resourcesReleased);  // 断言resourcesReleased为false
    EXPECT_FALSE(wasDestructed);  // 断言wasDestructed为false
    obj2 = obj;  // 将obj赋值给obj2，即将obj2指向obj指向的对象
  }
  EXPECT_FALSE(resourcesReleased);  // 断言resourcesReleased为false
  EXPECT_FALSE(wasDestructed);  // 断言wasDestructed为false
}
    # 断言资源已释放
    EXPECT_TRUE(resourcesReleased);
    # 断言对象已析构
    EXPECT_TRUE(wasDestructed);
TEST(
    IntrusivePtrTest,
    givenPtr_whenCopyAssignedToBaseClass_thenDestructsOldObject) {
  // 设置虚拟变量
  bool dummy = false;
  bool resourcesReleased = false;
  bool wasDestructed = false;
  // 创建指向 ChildDestructableMock 的智能指针 obj
  auto obj = make_intrusive<ChildDestructableMock>(&dummy, &dummy);
  {
    // 创建指向 DestructableMock 的智能指针 obj2
    auto obj2 =
        make_intrusive<DestructableMock>(&resourcesReleased, &wasDestructed);
    // 验证资源未释放和未析构
    EXPECT_FALSE(resourcesReleased);
    EXPECT_FALSE(wasDestructed);
    // 将 obj 赋值给 obj2，导致 obj2 引用的对象被析构
    obj2 = obj;
    // 验证资源已释放和已析构
    EXPECT_TRUE(resourcesReleased);
    EXPECT_TRUE(wasDestructed);
  }
}

TEST(
    IntrusivePtrTest,
    givenPtrWithCopy_whenCopyAssigned_thenDestructsOldObjectAfterCopyIsDestructed) {
  // 设置虚拟变量
  bool dummy = false;
  bool resourcesReleased = false;
  bool wasDestructed = false;
  // 创建指向 DestructableMock 的智能指针 obj
  auto obj = make_intrusive<DestructableMock>(&dummy, &dummy);
  {
    // 创建指向 DestructableMock 的智能指针 obj2
    auto obj2 =
        make_intrusive<DestructableMock>(&resourcesReleased, &wasDestructed);
    {
      // 创建 obj2 的拷贝
      auto copy = obj2;
      // 验证资源未释放和未析构
      EXPECT_FALSE(resourcesReleased);
      EXPECT_FALSE(wasDestructed);
      // 将 obj 赋值给 obj2，但不影响 obj2 的拷贝对象
      obj2 = obj;
      // 验证资源未释放和未析构
      EXPECT_FALSE(resourcesReleased);
      EXPECT_FALSE(wasDestructed);
    }
    // 验证资源已释放和已析构
    EXPECT_TRUE(resourcesReleased);
    EXPECT_TRUE(wasDestructed);
  }
}

TEST(
    IntrusivePtrTest,
    givenPtrWithBaseClassCopy_whenCopyAssigned_thenDestructsOldObjectAfterCopyIsDestructed) {
  // 设置虚拟变量
  bool dummy = false;
  bool resourcesReleased = false;
  bool wasDestructed = false;
  // 创建指向 ChildDestructableMock 的智能指针 obj
  auto obj = make_intrusive<ChildDestructableMock>(&dummy, &dummy);
  {
    // 创建指向 ChildDestructableMock 的智能指针 obj2
    auto obj2 = make_intrusive<ChildDestructableMock>(
        &resourcesReleased, &wasDestructed);
    {
      // 创建 obj2 的基类 DestructableMock 的智能指针拷贝
      intrusive_ptr<DestructableMock> copy = obj2;
      // 验证资源未释放和未析构
      EXPECT_FALSE(resourcesReleased);
      EXPECT_FALSE(wasDestructed);
      // 将 obj 赋值给 obj2，但不影响 obj2 的基类拷贝对象
      obj2 = obj;
      // 验证资源未释放和未析构
      EXPECT_FALSE(resourcesReleased);
      EXPECT_FALSE(wasDestructed);
    }
    // 验证资源已释放和已析构
    EXPECT_TRUE(resourcesReleased);
    EXPECT_TRUE(wasDestructed);
  }
}

TEST(
    IntrusivePtrTest,
    givenPtrWithCopy_whenCopyAssignedToBaseClass_thenDestructsOldObjectAfterCopyIsDestructed) {
  // 设置虚拟变量
  bool dummy = false;
  bool resourcesReleased = false;
  bool wasDestructed = false;
  // 创建指向 ChildDestructableMock 的智能指针 obj
  auto obj = make_intrusive<ChildDestructableMock>(&dummy, &dummy);
  {
    // 创建指向 DestructableMock 的智能指针 obj2
    auto obj2 =
        make_intrusive<DestructableMock>(&resourcesReleased, &wasDestructed);
    {
      // 创建 obj2 的基类 DestructableMock 的智能指针拷贝
      intrusive_ptr<DestructableMock> copy = obj2;
      // 验证资源未释放和未析构
      EXPECT_FALSE(resourcesReleased);
      EXPECT_FALSE(wasDestructed);
      // 将 obj 赋值给 obj2，但不影响 obj2 的基类拷贝对象
      obj2 = obj;
      // 验证资源未释放和未析构
      EXPECT_FALSE(resourcesReleased);
      EXPECT_FALSE(wasDestructed);
    }
    // 验证资源已释放和已析构
    EXPECT_TRUE(resourcesReleased);
    EXPECT_TRUE(wasDestructed);
  }
}

TEST(IntrusivePtrTest, givenPtr_whenCallingReset_thenDestructs) {
  // 设置虚拟变量
  bool resourcesReleased = false;
  bool wasDestructed = false;
  // 创建指向 DestructableMock 的智能指针 obj
  auto obj =
      make_intrusive<DestructableMock>(&resourcesReleased, &wasDestructed);
  // 验证资源未释放和未析构
  EXPECT_FALSE(resourcesReleased);
  EXPECT_FALSE(wasDestructed);
  // 调用 reset 方法，释放资源和析构对象
  obj.reset();
  // 验证资源已释放和已析构
  EXPECT_TRUE(resourcesReleased);
  EXPECT_TRUE(wasDestructed);
}
    givenPtrWithCopy_whenCallingReset_thenDestructsAfterCopyDestructed) {
  // 初始化两个布尔变量，用于跟踪资源释放和对象析构的状态
  bool resourcesReleased = false;
  bool wasDestructed = false;
  // 创建一个共享指针对象，指向一个 DestructableMock 对象，并传入状态跟踪的指针
  auto obj =
      make_intrusive<DestructableMock>(&resourcesReleased, &wasDestructed);
  {
    // 创建 obj 的一个拷贝 copy
    auto copy = obj;
    // 重置 obj，预期不会释放资源或析构
    obj.reset();
    // 验证 obj 的资源未被释放和未析构
    EXPECT_FALSE(resourcesReleased);
    EXPECT_FALSE(wasDestructed);
    // 重置 copy，预期会释放资源和析构
    copy.reset();
    // 验证 copy 的资源已被释放和已析构
    EXPECT_TRUE(resourcesReleased);
    EXPECT_TRUE(wasDestructed);
  }
}

TEST(
    IntrusivePtrTest,
    givenPtrWithCopy_whenCallingResetOnCopy_thenDestructsAfterOriginalDestructed) {
  bool resourcesReleased = false;  // 标志资源是否被释放
  bool wasDestructed = false;       // 标志对象是否被析构

  auto obj = make_intrusive<DestructableMock>(&resourcesReleased, &wasDestructed);  // 创建一个具有析构功能的对象指针

  {
    auto copy = obj;  // 创建原始对象的拷贝
    copy.reset();     // 重置拷贝对象
    EXPECT_FALSE(resourcesReleased);  // 验证资源未被释放
    EXPECT_FALSE(wasDestructed);      // 验证对象未被析构
    obj.reset();       // 重置原始对象
    EXPECT_TRUE(resourcesReleased);   // 验证资源已被释放
    EXPECT_TRUE(wasDestructed);       // 验证对象已被析构
  }
}

TEST(
    IntrusivePtrTest,
    givenPtrWithMoved_whenCallingReset_thenDestructsAfterMovedDestructed) {
  bool resourcesReleased = false;  // 标志资源是否被释放
  bool wasDestructed = false;       // 标志对象是否被析构

  auto obj = make_intrusive<DestructableMock>(&resourcesReleased, &wasDestructed);  // 创建一个具有析构功能的对象指针

  {
    auto moved = std::move(obj);  // 将对象移动到新的指针
    // NOLINTNEXTLINE(bugprone-use-after-move)
    obj.reset();                  // 重置移动后的原始指针
    EXPECT_FALSE(resourcesReleased);  // 验证资源未被释放
    EXPECT_FALSE(wasDestructed);      // 验证对象未被析构
    moved.reset();                // 重置移动后的对象指针
    EXPECT_TRUE(resourcesReleased);   // 验证资源已被释放
    EXPECT_TRUE(wasDestructed);       // 验证对象已被析构
  }
}

TEST(
    IntrusivePtrTest,
    givenPtrWithMoved_whenCallingResetOnMoved_thenDestructsImmediately) {
  bool resourcesReleased = false;  // 标志资源是否被释放
  bool wasDestructed = false;       // 标志对象是否被析构

  auto obj = make_intrusive<DestructableMock>(&resourcesReleased, &wasDestructed);  // 创建一个具有析构功能的对象指针

  {
    auto moved = std::move(obj);  // 将对象移动到新的指针
    moved.reset();                // 重置移动后的对象指针
    EXPECT_TRUE(resourcesReleased);   // 验证资源已被释放
    EXPECT_TRUE(wasDestructed);       // 验证对象已被析构
  }
}

TEST(IntrusivePtrTest, AllowsMoveConstructingToConst) {
  intrusive_ptr<SomeClass> a = make_intrusive<SomeClass>();  // 创建普通对象指针
  intrusive_ptr<const SomeClass> b = std::move(a);           // 将普通对象指针移动到常量对象指针
}

TEST(IntrusivePtrTest, AllowsCopyConstructingToConst) {
  intrusive_ptr<SomeClass> a = make_intrusive<SomeClass>();  // 创建普通对象指针
  intrusive_ptr<const SomeClass> b = a;                      // 将普通对象指针拷贝到常量对象指针
}

TEST(IntrusivePtrTest, AllowsMoveAssigningToConst) {
  intrusive_ptr<SomeClass> a = make_intrusive<SomeClass>();       // 创建普通对象指针
  intrusive_ptr<const SomeClass> b = make_intrusive<SomeClass>(); // 创建常量对象指针
  b = std::move(a);                                              // 将普通对象指针移动到常量对象指针
}

TEST(IntrusivePtrTest, AllowsCopyAssigningToConst) {
  intrusive_ptr<SomeClass> a = make_intrusive<SomeClass>();        // 创建普通对象指针
  intrusive_ptr<const SomeClass> b = make_intrusive<const SomeClass>();  // 创建常量对象指针
  b = a;                                                           // 将普通对象指针拷贝到常量对象指针
}

TEST(IntrusivePtrTest, givenNewPtr_thenHasUseCount1) {
  intrusive_ptr<SomeClass> obj = make_intrusive<SomeClass>();  // 创建新的普通对象指针
  EXPECT_EQ(1, obj.use_count());                               // 验证对象指针的引用计数为1
}

TEST(IntrusivePtrTest, givenNewPtr_thenIsUnique) {
  intrusive_ptr<SomeClass> obj = make_intrusive<SomeClass>();  // 创建新的普通对象指针
  EXPECT_TRUE(obj.unique());                                   // 验证对象指针是唯一的
}

TEST(IntrusivePtrTest, givenEmptyPtr_thenHasUseCount0) {
  intrusive_ptr<SomeClass> obj;              // 创建空的对象指针
  EXPECT_EQ(0, obj.use_count());             // 验证对象指针的引用计数为0
}

TEST(IntrusivePtrTest, givenEmptyPtr_thenIsNotUnique) {
  intrusive_ptr<SomeClass> obj;              // 创建空的对象指针
  EXPECT_FALSE(obj.unique());                // 验证对象指针不是唯一的
}

TEST(IntrusivePtrTest, givenResetPtr_thenHasUseCount0) {
  intrusive_ptr<SomeClass> obj = make_intrusive<SomeClass>();  // 创建新的普通对象指针
  obj.reset();                                                // 重置对象指针
  EXPECT_EQ(0, obj.use_count());                               // 验证对象指针的引用计数为0
}

TEST(IntrusivePtrTest, givenResetPtr_thenIsNotUnique) {
  intrusive_ptr<SomeClass> obj = make_intrusive<SomeClass>();  // 创建新的普通对象指针
  obj.reset();                                                // 重置对象指针
  EXPECT_FALSE(obj.unique());                                 // 验证对象指针不是唯一的
}
TEST(IntrusivePtrTest, givenMoveConstructedPtr_thenHasUseCount1) {
  // 创建一个智能指针 obj，并初始化为 SomeClass 的实例
  intrusive_ptr<SomeClass> obj = make_intrusive<SomeClass>();
  // 使用 move 操作将 obj 转移给 obj2
  intrusive_ptr<SomeClass> obj2 = std::move(obj);
  // 断言 obj2 的引用计数为 1
  EXPECT_EQ(1, obj2.use_count());
}

TEST(IntrusivePtrTest, givenMoveConstructedPtr_thenIsUnique) {
  // 创建一个智能指针 obj，并初始化为 SomeClass 的实例
  intrusive_ptr<SomeClass> obj = make_intrusive<SomeClass>();
  // 使用 move 操作将 obj 转移给 obj2
  intrusive_ptr<SomeClass> obj2 = std::move(obj);
  // 断言 obj2 是唯一引用（引用计数为 1）
  EXPECT_TRUE(obj2.unique());
}

TEST(IntrusivePtrTest, givenMoveConstructedPtr_thenOldHasUseCount0) {
  // 创建一个智能指针 obj，并初始化为 SomeClass 的实例
  intrusive_ptr<SomeClass> obj = make_intrusive<SomeClass>();
  // 使用 move 操作将 obj 转移给 obj2
  intrusive_ptr<SomeClass> obj2 = std::move(obj);
  // 断言原来的 obj 引用计数为 0
  // NOLINTNEXTLINE(clang-analyzer-cplusplus.Move,bugprone-use-after-move)
  EXPECT_EQ(0, obj.use_count());
}

TEST(IntrusivePtrTest, givenMoveConstructedPtr_thenOldIsNotUnique) {
  // 创建一个智能指针 obj，并初始化为 SomeClass 的实例
  intrusive_ptr<SomeClass> obj = make_intrusive<SomeClass>();
  // 使用 move 操作将 obj 转移给 obj2
  intrusive_ptr<SomeClass> obj2 = std::move(obj);
  // 断言原来的 obj 不是唯一引用（引用计数不为 1）
  // NOLINTNEXTLINE(clang-analyzer-cplusplus.Move,bugprone-use-after-move)
  EXPECT_FALSE(obj.unique());
}

TEST(IntrusivePtrTest, givenMoveAssignedPtr_thenHasUseCount1) {
  // 创建一个智能指针 obj，并初始化为 SomeClass 的实例
  intrusive_ptr<SomeClass> obj = make_intrusive<SomeClass>();
  // 创建另一个智能指针 obj2，并初始化为 SomeClass 的实例
  intrusive_ptr<SomeClass> obj2 = make_intrusive<SomeClass>();
  // 使用 move 操作将 obj 赋值给 obj2
  obj2 = std::move(obj);
  // 断言 obj2 的引用计数为 1
  EXPECT_EQ(1, obj2.use_count());
}

TEST(IntrusivePtrTest, givenMoveAssignedPtr_thenIsUnique) {
  // 创建一个智能指针 obj，并初始化为 SomeClass 的实例
  intrusive_ptr<SomeClass> obj = make_intrusive<SomeClass>();
  // 创建另一个智能指针 obj2，并初始化为 SomeClass 的实例
  intrusive_ptr<SomeClass> obj2 = make_intrusive<SomeClass>();
  // 使用 move 操作将 obj 赋值给 obj2
  obj2 = std::move(obj);
  // 断言 obj2 是唯一引用（引用计数为 1）
  EXPECT_TRUE(obj2.unique());
}

TEST(IntrusivePtrTest, givenMoveAssignedPtr_thenOldHasUseCount0) {
  // 创建一个智能指针 obj，并初始化为 SomeClass 的实例
  intrusive_ptr<SomeClass> obj = make_intrusive<SomeClass>();
  // 创建另一个智能指针 obj2，并初始化为 SomeClass 的实例
  intrusive_ptr<SomeClass> obj2 = make_intrusive<SomeClass>();
  // 使用 move 操作将 obj 赋值给 obj2
  obj2 = std::move(obj);
  // 断言原来的 obj 引用计数为 0
  // NOLINTNEXTLINE(clang-analyzer-cplusplus.Move,bugprone-use-after-move)
  EXPECT_EQ(0, obj.use_count());
}

TEST(IntrusivePtrTest, givenMoveAssignedPtr_thenOldIsNotUnique) {
  // 创建一个智能指针 obj，并初始化为 SomeClass 的实例
  intrusive_ptr<SomeClass> obj = make_intrusive<SomeClass>();
  // 创建另一个智能指针 obj2，并初始化为 SomeClass 的实例
  intrusive_ptr<SomeClass> obj2 = make_intrusive<SomeClass>();
  // 使用 move 操作将 obj 赋值给 obj2
  obj2 = std::move(obj);
  // 断言原来的 obj 不是唯一引用（引用计数不为 1）
  // NOLINTNEXTLINE(clang-analyzer-cplusplus.Move,bugprone-use-after-move)
  EXPECT_FALSE(obj.unique());
}

TEST(IntrusivePtrTest, givenCopyConstructedPtr_thenHasUseCount2) {
  // 创建一个智能指针 obj，并初始化为 SomeClass 的实例
  intrusive_ptr<SomeClass> obj = make_intrusive<SomeClass>();
  // 使用拷贝构造将 obj 赋值给 obj2
  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
  intrusive_ptr<SomeClass> obj2 = obj;
  // 断言 obj2 的引用计数为 2
  EXPECT_EQ(2, obj2.use_count());
}

TEST(IntrusivePtrTest, givenCopyConstructedPtr_thenIsNotUnique) {
  // 创建一个智能指针 obj，并初始化为 SomeClass 的实例
  intrusive_ptr<SomeClass> obj = make_intrusive<SomeClass>();
  // 使用拷贝构造将 obj 赋值给 obj2
  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
  intrusive_ptr<SomeClass> obj2 = obj;
  // 断言 obj2 不是唯一引用（引用计数大于 1）
  EXPECT_FALSE(obj2.unique());
}

TEST(IntrusivePtrTest, givenCopyConstructedPtr_thenOldHasUseCount2) {
  // 创建一个智能指针 obj，并初始化为 SomeClass 的实例
  intrusive_ptr<SomeClass> obj = make_intrusive<SomeClass>();
  // 使用拷贝构造将 obj 赋值给 obj2
  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
  intrusive_ptr<SomeClass> obj2 = obj;
  // 断言原来的 obj 的引用计数为 2
  EXPECT_EQ(2, obj.use_count());
}
TEST(IntrusivePtrTest, givenCopyConstructedPtr_thenOldIsNotUnique) {
  // 创建一个 intrusive_ptr 智能指针，指向一个新创建的 SomeClass 实例
  intrusive_ptr<SomeClass> obj = make_intrusive<SomeClass>();
  // 使用拷贝构造函数创建另一个 intrusive_ptr，指向相同的 SomeClass 实例
  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
  intrusive_ptr<SomeClass> obj2 = obj;
  // 预期 obj 不是唯一的所有者（use_count > 1）
  EXPECT_FALSE(obj.unique());
}

TEST(
    IntrusivePtrTest,
    givenCopyConstructedPtr_whenDestructingCopy_thenHasUseCount1) {
  // 创建一个 intrusive_ptr 智能指针，指向一个新创建的 SomeClass 实例
  intrusive_ptr<SomeClass> obj = make_intrusive<SomeClass>();
  {
    // 在作用域内创建另一个 intrusive_ptr，使用拷贝构造函数指向同一 SomeClass 实例
    // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
    intrusive_ptr<SomeClass> obj2 = obj;
    // 预期 obj 的 use_count 为 2
    EXPECT_EQ(2, obj.use_count());
  }
  // 在作用域外，预期 obj 的 use_count 为 1
  EXPECT_EQ(1, obj.use_count());
}

TEST(
    IntrusivePtrTest,
    givenCopyConstructedPtr_whenDestructingCopy_thenIsUnique) {
  // 创建一个 intrusive_ptr 智能指针，指向一个新创建的 SomeClass 实例
  intrusive_ptr<SomeClass> obj = make_intrusive<SomeClass>();
  {
    // 在作用域内创建另一个 intrusive_ptr，使用拷贝构造函数指向同一 SomeClass 实例
    // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
    intrusive_ptr<SomeClass> obj2 = obj;
    // 预期 obj 不是唯一的所有者（use_count > 1）
    EXPECT_FALSE(obj.unique());
  }
  // 在作用域外，预期 obj 是唯一的所有者（use_count == 1）
  EXPECT_TRUE(obj.unique());
}

TEST(
    IntrusivePtrTest,
    givenCopyConstructedPtr_whenReassigningCopy_thenHasUseCount1) {
  // 创建一个 intrusive_ptr 智能指针，指向一个新创建的 SomeClass 实例
  intrusive_ptr<SomeClass> obj = make_intrusive<SomeClass>();
  // 使用拷贝构造函数创建另一个 intrusive_ptr，指向相同的 SomeClass 实例
  intrusive_ptr<SomeClass> obj2 = obj;
  // 预期 obj 的 use_count 为 2
  EXPECT_EQ(2, obj.use_count());
  // 重新赋值 obj2，使其指向一个新创建的 SomeClass 实例
  obj2 = make_intrusive<SomeClass>();
  // 在重新赋值后，预期 obj 的 use_count 为 1
  EXPECT_EQ(1, obj.use_count());
  // 预期 obj2 的 use_count 为 1
  EXPECT_EQ(1, obj2.use_count());
}

TEST(
    IntrusivePtrTest,
    givenCopyConstructedPtr_whenReassigningCopy_thenIsUnique) {
  // 创建一个 intrusive_ptr 智能指针，指向一个新创建的 SomeClass 实例
  intrusive_ptr<SomeClass> obj = make_intrusive<SomeClass>();
  // 使用拷贝构造函数创建另一个 intrusive_ptr，指向相同的 SomeClass 实例
  intrusive_ptr<SomeClass> obj2 = obj;
  // 预期 obj 不是唯一的所有者（use_count > 1）
  EXPECT_FALSE(obj.unique());
  // 重新赋值 obj2，使其指向一个新创建的 SomeClass 实例
  obj2 = make_intrusive<SomeClass>();
  // 在重新赋值后，预期 obj 是唯一的所有者（use_count == 1）
  EXPECT_TRUE(obj.unique());
  // 预期 obj2 是唯一的所有者（use_count == 1）
  EXPECT_TRUE(obj2.unique());
}

TEST(IntrusivePtrTest, givenCopyAssignedPtr_thenHasUseCount2) {
  // 创建一个 intrusive_ptr 智能指针，指向一个新创建的 SomeClass 实例
  intrusive_ptr<SomeClass> obj = make_intrusive<SomeClass>();
  // 创建另一个 intrusive_ptr，指向另一个新创建的 SomeClass 实例
  intrusive_ptr<SomeClass> obj2 = make_intrusive<SomeClass>();
  // 使用拷贝赋值操作符将 obj2 指向 obj 所指向的实例
  obj2 = obj;
  // 预期 obj 和 obj2 的 use_count 都为 2
  EXPECT_EQ(2, obj.use_count());
  EXPECT_EQ(2, obj2.use_count());
}

TEST(IntrusivePtrTest, givenCopyAssignedPtr_thenIsNotUnique) {
  // 创建一个 intrusive_ptr 智能指针，指向一个新创建的 SomeClass 实例
  intrusive_ptr<SomeClass> obj = make_intrusive<SomeClass>();
  // 创建另一个 intrusive_ptr，指向另一个新创建的 SomeClass 实例
  intrusive_ptr<SomeClass> obj2 = make_intrusive<SomeClass>();
  // 使用拷贝赋值操作符将 obj2 指向 obj 所指向的实例
  obj2 = obj;
  // 预期 obj 和 obj2 都不是唯一的所有者（use_count > 1）
  EXPECT_FALSE(obj.unique());
  EXPECT_FALSE(obj2.unique());
}

TEST(
    IntrusivePtrTest,
    givenCopyAssignedPtr_whenDestructingCopy_thenHasUseCount1) {
  // 创建一个 intrusive_ptr 智能指针，指向一个新创建的 SomeClass 实例
  intrusive_ptr<SomeClass> obj = make_intrusive<SomeClass>();
  {
    // 在作用域内创建另一个 intrusive_ptr，指向另一个新创建的 SomeClass 实例
    intrusive_ptr<SomeClass> obj2 = make_intrusive<SomeClass>();
    // 使用拷贝赋值操作符将 obj2 指向 obj 所指向的实例
    obj2 = obj;
    // 预期 obj 的 use_count 为 2
    EXPECT_EQ(2, obj.use_count());
  }
  // 在作用域外，预期 obj 的 use_count 为 1
  EXPECT_EQ(1, obj.use_count());
}

TEST(IntrusivePtrTest, givenCopyAssignedPtr_whenDestructingCopy_thenIsUnique) {
  // 创建一个 intrusive_ptr 智能指针，指向一个新创建的 SomeClass 实例
  intrusive_ptr<SomeClass> obj = make_intrusive<SomeClass>();
  {
    // 在作用域内创建另一个 intrusive_ptr，指向另一个新创建的 SomeClass 实例
    intrusive_ptr<SomeClass> obj2 = make_intrusive<SomeClass>();
    // 使用拷贝赋值操作符将 obj2 指向 obj 所指向的实例
    obj2 = obj;
    // 预期 obj 不是唯一的所有者（use_count > 1）
    EXPECT_FALSE(obj.unique());
  }
  // 在作用域外，预期 obj 是唯一的所有者（use_count == 1）
  EXPECT_TRUE(obj.unique());
}
    // 创建一个指向 SomeClass 的智能指针 obj，并进行复制构造，使其引用计数为 1
    intrusive_ptr<SomeClass> obj = make_intrusive<SomeClass>();
    
    // 创建另一个指向 SomeClass 的智能指针 obj2，并进行复制构造，使其引用计数为 1
    intrusive_ptr<SomeClass> obj2 = make_intrusive<SomeClass>();
    
    // 将 obj 的指针赋给 obj2，增加 obj 的引用计数为 2
    obj2 = obj;
    
    // 验证 obj 的引用计数为 2
    EXPECT_EQ(2, obj.use_count());
    
    // 重新分配 obj2 指向一个新创建的 SomeClass 对象，减少 obj 的引用计数为 1
    obj2 = make_intrusive<SomeClass>();
    
    // 验证 obj 的引用计数为 1
    EXPECT_EQ(1, obj.use_count());
    
    // 验证 obj2 的引用计数为 1
    EXPECT_EQ(1, obj2.use_count());
}

TEST(IntrusivePtrTest, givenCopyAssignedPtr_whenReassigningCopy_thenIsUnique) {
  // 创建一个 intrusive_ptr 智能指针对象，并初始化为 SomeClass 类的实例
  intrusive_ptr<SomeClass> obj = make_intrusive<SomeClass>();
  // 创建另一个 intrusive_ptr 智能指针对象，并初始化为 SomeClass 类的实例
  intrusive_ptr<SomeClass> obj2 = make_intrusive<SomeClass>();
  // 将 obj2 指向 obj 指向的对象，此时两个指针共享同一对象
  obj2 = obj;
  // 验证 obj 不是唯一的所有权者
  EXPECT_FALSE(obj.unique());
  // 将 obj2 重新指向新创建的 SomeClass 对象，验证 obj 变为唯一的所有权者
  obj2 = make_intrusive<SomeClass>();
  EXPECT_TRUE(obj.unique());
  EXPECT_TRUE(obj2.unique());
}

TEST(IntrusivePtrTest, givenPtr_whenReleasedAndReclaimed_thenDoesntCrash) {
  // 创建一个 intrusive_ptr 智能指针对象，并初始化为 SomeClass 类的实例
  intrusive_ptr<SomeClass> obj = make_intrusive<SomeClass>();
  // 释放 obj 指向的对象，并返回原始指针
  SomeClass* ptr = obj.release();
  // 验证 obj 不再定义指向对象
  EXPECT_FALSE(obj.defined());
  // 使用原始指针创建一个新的 intrusive_ptr 智能指针对象
  intrusive_ptr<SomeClass> reclaimed = intrusive_ptr<SomeClass>::reclaim(ptr);
}

TEST(
    IntrusivePtrTest,
    givenPtr_whenReleasedAndReclaimed_thenIsDestructedAtEnd) {
  // 初始化资源释放和析构标志为 false
  bool resourcesReleased = false;
  bool wasDestructed = false;
  {
    // 创建外部 intrusive_ptr 智能指针对象
    intrusive_ptr<DestructableMock> outer;
    {
      // 创建内部 intrusive_ptr 智能指针对象，并初始化为 DestructableMock 类的实例
      intrusive_ptr<DestructableMock> inner =
          make_intrusive<DestructableMock>(&resourcesReleased, &wasDestructed);
      // 释放 inner 指向的对象，并返回原始指针
      DestructableMock* ptr = inner.release();
      // 验证资源未释放和对象未析构
      EXPECT_FALSE(resourcesReleased);
      EXPECT_FALSE(wasDestructed);
      // 使用原始指针创建一个新的 intrusive_ptr 智能指针对象，并赋给 outer
      outer = intrusive_ptr<DestructableMock>::reclaim(ptr);
    }
    // inner 被析构
    EXPECT_FALSE(resourcesReleased);
    EXPECT_FALSE(wasDestructed);
  }
  // outer 被析构
  EXPECT_TRUE(resourcesReleased);
  EXPECT_TRUE(wasDestructed);
}

/*
TEST(IntrusivePtrTest, givenStackObject_whenReclaimed_thenCrashes) {
  // 在析构时可能引起奇怪的 bug，最好在创建时就提前崩溃
  SomeClass obj;
  intrusive_ptr<SomeClass> ptr;
#ifdef NDEBUG
  EXPECT_NO_THROW(ptr = intrusive_ptr<SomeClass>::reclaim(&obj));
#else
  EXPECT_ANY_THROW(ptr = intrusive_ptr<SomeClass>::reclaim(&obj));
#endif
}*/

TEST(IntrusivePtrTest, givenPtr_whenNonOwningReclaimed_thenDoesntCrash) {
  // 创建一个 intrusive_ptr 智能指针对象，并初始化为 SomeClass 类的实例
  intrusive_ptr<SomeClass> obj = make_intrusive<SomeClass>();
  // 获取 obj 指向对象的原始指针
  SomeClass* raw_ptr = obj.get();
  // 验证 obj 定义指向对象
  EXPECT_TRUE(obj.defined());
  // 使用原始指针创建一个新的 intrusive_ptr 智能指针对象
  intrusive_ptr<SomeClass> reclaimed =
      intrusive_ptr<SomeClass>::unsafe_reclaim_from_nonowning(raw_ptr);
  // 验证 reclaimed 定义指向对象，并且与 obj 指向同一对象
  EXPECT_TRUE(reclaimed.defined());
  EXPECT_EQ(reclaimed.get(), obj.get());
}

TEST(IntrusivePtrTest, givenPtr_whenNonOwningReclaimed_thenIsDestructedAtEnd) {
  // 初始化资源释放和析构标志为 false
  bool resourcesReleased = false;
  bool wasDestructed = false;
  {
    // 创建外部 intrusive_ptr 智能指针对象
    intrusive_ptr<DestructableMock> outer;
    {
      // 创建内部 intrusive_ptr 智能指针对象，并初始化为 DestructableMock 类的实例
      intrusive_ptr<DestructableMock> inner =
          make_intrusive<DestructableMock>(&resourcesReleased, &wasDestructed);
      // 获取 inner 指向对象的原始指针
      DestructableMock* raw_ptr = inner.get();
      // 使用原始指针创建一个新的 intrusive_ptr 智能指针对象，并赋给 outer
      outer = intrusive_ptr<DestructableMock>::unsafe_reclaim_from_nonowning(
          raw_ptr);
    }
    // inner 被析构
    EXPECT_FALSE(resourcesReleased);
    EXPECT_FALSE(wasDestructed);
  }
  // outer 被析构
  EXPECT_TRUE(resourcesReleased);
  EXPECT_TRUE(wasDestructed);
}

namespace {
template <class T>
struct IntrusiveAndWeak final {
  // 构造函数，接受一个 intrusive_ptr 智能指针对象，并初始化 ptr 和 weak 成员变量
  IntrusiveAndWeak(intrusive_ptr<T> ptr_) : ptr(std::move(ptr_)), weak(ptr) {}

  intrusive_ptr<T> ptr; // 指向 T 类型对象的智能指针
  weak_intrusive_ptr<T> weak; // 指向 T 类型对象的弱引用智能指针
};
template <class T, class... Args>
// 创建一个弱引用和强引用的组合对象，返回这个对象
IntrusiveAndWeak<T> make_weak_intrusive(Args&&... args) {
  // 调用make_intrusive<T>创建一个强引用对象，并用它初始化IntrusiveAndWeak<T>对象
  return IntrusiveAndWeak<T>(make_intrusive<T>(std::forward<Args>(args)...));
}

template <class T, class... Args>
// 创建一个仅包含弱引用的对象
weak_intrusive_ptr<T> make_weak_only(Args&&... args) {
  // 创建一个强引用对象intrusive
  auto intrusive = make_intrusive<T>(std::forward<Args>(args)...);
  // 用intrusive初始化一个weak_intrusive_ptr<T>对象并返回
  return weak_intrusive_ptr<T>(intrusive);
}

template <
    class T,
    class NullType = c10::detail::intrusive_target_default_null_type<T>>
// 创建一个无效的弱引用对象
weak_intrusive_ptr<T, NullType> make_invalid_weak() {
  // 返回一个带有NullType类型参数的无效的weak_intrusive_ptr<T, NullType>对象
  return weak_intrusive_ptr<T, NullType>(intrusive_ptr<T, NullType>());
}

// 定义一个结构体WeakReferenceToSelf，继承自intrusive_ptr_target
struct WeakReferenceToSelf : public intrusive_ptr_target {
  // 覆盖父类的release_resources方法
  void release_resources() override {
    // 重置ptr成员变量，释放资源
    ptr.reset();
  }
  // 初始化一个weak_intrusive_ptr<intrusive_ptr_target>类型的ptr成员变量
  weak_intrusive_ptr<intrusive_ptr_target> ptr =
      weak_intrusive_ptr<intrusive_ptr_target>(
          make_intrusive<intrusive_ptr_target>());
};
} // namespace

// 静态断言，验证SomeClass与weak_intrusive_ptr<SomeClass>::element_type是否相同
static_assert(
    std::is_same_v<SomeClass, weak_intrusive_ptr<SomeClass>::element_type>,
    "weak_intrusive_ptr<T>::element_type is wrong");

// 单元测试WeakIntrusivePtrTest中的givenPtr_whenCreatingAndDestructing_thenDoesntCrash
TEST(
    WeakIntrusivePtrTest,
    givenPtr_whenCreatingAndDestructing_thenDoesntCrash) {
  // 创建一个IntrusiveAndWeak<SomeClass>对象var，并用make_weak_intrusive<SomeClass>()初始化它
  IntrusiveAndWeak<SomeClass> var = make_weak_intrusive<SomeClass>();
}

// 单元测试WeakIntrusivePtrTest中的givenPtr_whenLocking_thenReturnsCorrectObject
TEST(WeakIntrusivePtrTest, givenPtr_whenLocking_thenReturnsCorrectObject) {
  // 创建一个IntrusiveAndWeak<SomeClass>对象var，并用make_weak_intrusive<SomeClass>()初始化它
  IntrusiveAndWeak<SomeClass> var = make_weak_intrusive<SomeClass>();
  // 将var的弱引用对象var.weak锁定到intrusive_ptr<SomeClass>对象locked
  intrusive_ptr<SomeClass> locked = var.weak.lock();
  // 断言锁定的对象与var.ptr的get()方法返回的对象相等
  EXPECT_EQ(var.ptr.get(), locked.get());
}

// 单元测试WeakIntrusivePtrTest中的expiredPtr_whenLocking_thenReturnsNullType
TEST(WeakIntrusivePtrTest, expiredPtr_whenLocking_thenReturnsNullType) {
  // 创建一个IntrusiveAndWeak<SomeClass>对象var，并用make_weak_intrusive<SomeClass>()初始化它
  IntrusiveAndWeak<SomeClass> var = make_weak_intrusive<SomeClass>();
  // 重置var.ptr，以测试弱引用指针是否仍然有效
  var.ptr.reset();
  // 断言var.weak.expired()返回true
  EXPECT_TRUE(var.weak.expired());
  // 将var.weak锁定到intrusive_ptr<SomeClass>对象locked
  intrusive_ptr<SomeClass> locked = var.weak.lock();
  // 断言locked.defined()返回false
  EXPECT_FALSE(locked.defined());
}

// 单元测试WeakIntrusivePtrTest中的weakNullPtr_locking
TEST(WeakIntrusivePtrTest, weakNullPtr_locking) {
  // 创建一个无效的弱引用指针weak_ptr，类型为SomeClass
  auto weak_ptr = make_invalid_weak<SomeClass>();
  // 将weak_ptr锁定到intrusive_ptr<SomeClass>对象locked
  intrusive_ptr<SomeClass> locked = weak_ptr.lock();
  // 断言locked.defined()返回false
  EXPECT_FALSE(locked.defined());
}

// 单元测试WeakIntrusivePtrTest中的givenValidPtr_whenMoveAssigning_thenPointsToSameObject
TEST(
    WeakIntrusivePtrTest,
    givenValidPtr_whenMoveAssigning_thenPointsToSameObject) {
  // 创建两个IntrusiveAndWeak<SomeClass>对象obj1和obj2，并分别用make_weak_intrusive<SomeClass>()初始化它们
  IntrusiveAndWeak<SomeClass> obj1 = make_weak_intrusive<SomeClass>();
  IntrusiveAndWeak<SomeClass> obj2 = make_weak_intrusive<SomeClass>();
  // 获取obj1的弱引用对象的指针
  SomeClass* obj1ptr = obj1.weak.lock().get();
  // 将obj1.weak移动到obj2.weak
  obj2.weak = std::move(obj1.weak);
  // 断言obj2.weak锁定的对象与obj1ptr相等
  EXPECT_EQ(obj1ptr, obj2.weak.lock().get());
}

// 单元测试WeakIntrusivePtrTest中的givenValidPtr_whenMoveAssigning_thenOldInstanceInvalid
TEST(
    WeakIntrusivePtrTest,
    givenValidPtr_whenMoveAssigning_thenOldInstanceInvalid) {
  // 创建两个IntrusiveAndWeak<SomeClass>对象obj1和obj2，并分别用make_weak_intrusive<SomeClass>()初始化它们
  IntrusiveAndWeak<SomeClass> obj1 = make_weak_intrusive<SomeClass>();
  IntrusiveAndWeak<SomeClass> obj2 = make_weak_intrusive<SomeClass>();
  // 将obj1.weak移动到obj2.weak
  obj2.weak = std::move(obj1.weak);
  // 断言obj1.weak.expired()返回true
  EXPECT_TRUE(obj1.weak.expired());
}

// 单元测试WeakIntrusivePtrTest中的vector_insert_weak_intrusive
TEST(WeakIntrusivePtrTest, vector_insert_weak_intrusive) {
  // 创建一个空的std::vector<weak_intrusive_ptr<SomeClass>>对象priorWorks
  std::vector<weak_intrusive_ptr<SomeClass>> priorWorks;
  // 创建一个包含一个intrusive_ptr<SomeClass>对象的std::vector对象wips
  std::vector<intrusive_ptr<SomeClass>> wips;
  wips.push_back(make_intrusive<SomeClass>());
  // 将wips中的元素插入到priorWorks的末尾
  priorWorks.insert(priorWorks.end(), wips.begin(), wips.end());
  // 断言priorWorks的大小为1
  EXPECT_EQ(priorWorks.size(), 1);
}
    WeakIntrusivePtrTest,
    // 定义测试用例名称 WeakIntrusivePtrTest，这是一个测试弱指针的单元测试
    givenInvalidPtr_whenMoveAssigning_thenNewInstanceIsValid) {
    // 当给定无效指针并移动赋值后，新实例应有效

  IntrusiveAndWeak<SomeClass> obj1 = make_weak_intrusive<SomeClass>();
  // 创建一个带有强引用和弱引用的对象 obj1，通过 make_weak_intrusive 创建弱指针

  weak_intrusive_ptr<SomeClass> obj2 = make_invalid_weak<SomeClass>();
  // 创建一个无效的弱指针 obj2，通过 make_invalid_weak 创建

  obj1.weak.lock().get();
  // 锁定 obj1 的弱引用并获取其内部指针

  obj2 = std::move(obj1.weak);
  // 将 obj1 的弱引用对象移动给 obj2，此处完成了指针的所有权转移

  EXPECT_FALSE(obj2.expired());
  // 断言 obj2 没有过期，即它指向的对象仍然有效
TEST(
    WeakIntrusivePtrTest,
    givenValidPtr_whenMoveAssigningToSelf_thenPointsToSameObject) {
  // 创建一个包含有效弱指针的对象
  IntrusiveAndWeak<SomeClass> obj1 = make_weak_intrusive<SomeClass>();
  // 获取 obj1 的弱指针指向的对象的原始指针
  SomeClass* obj1ptr = obj1.weak.lock().get();
  // 将 obj1 的弱指针移动给自身，预期指向的对象仍然是相同的
  obj1.weak = std::move(obj1.weak);
  // 断言移动后指向的对象依然是同一个对象
  EXPECT_EQ(obj1ptr, obj1.weak.lock().get());
}

TEST(
    WeakIntrusivePtrTest,
    givenValidPtr_whenMoveAssigningToSelf_thenStaysValid) {
  // 创建一个包含有效弱指针的对象
  IntrusiveAndWeak<SomeClass> obj1 = make_weak_intrusive<SomeClass>();
  // 将 obj1 的弱指针移动给自身，预期弱指针仍然有效
  obj1.weak = std::move(obj1.weak);
  // 断言弱指针没有变成无效
  EXPECT_FALSE(obj1.weak.expired());
}

TEST(
    WeakIntrusivePtrTest,
    givenInvalidPtr_whenMoveAssigning_thenPointsToSameObject) {
  // 创建一个包含有效弱指针的对象
  IntrusiveAndWeak<SomeClass> obj1 = make_weak_intrusive<SomeClass>();
  // 创建一个无效的弱指针对象
  weak_intrusive_ptr<SomeClass> obj2 = make_invalid_weak<SomeClass>();
  // 获取 obj1 的弱指针指向的对象的原始指针
  SomeClass* obj1ptr = obj1.weak.lock().get();
  // 将 obj1 的弱指针移动给 obj2，预期指向的对象仍然是相同的
  obj2 = std::move(obj1.weak);
  // 断言移动后 obj2 指向的对象与 obj1 指向的对象相同
  EXPECT_EQ(obj1ptr, obj2.lock().get());
}

TEST(
    WeakIntrusivePtrTest,
    givenInvalidPtr_whenMoveAssigningToSelf_thenStaysInvalid) {
  // 创建一个无效的弱指针对象
  weak_intrusive_ptr<SomeClass> obj1 = make_invalid_weak<SomeClass>();
  // 将 obj1 移动给自身，预期弱指针变为无效
  obj1 = std::move(obj1);
  // 断言弱指针已经变为无效
  EXPECT_TRUE(obj1.expired());
}

TEST(
    WeakIntrusivePtrTest,
    givenWeakOnlyPtr_whenMoveAssigning_thenNewInstanceIsValid) {
  // 创建一个包含有效弱指针的对象
  IntrusiveAndWeak<SomeClass> obj1 = make_weak_intrusive<SomeClass>();
  // 创建一个仅有弱指针的对象
  weak_intrusive_ptr<SomeClass> obj2 = make_weak_only<SomeClass>();
  // 获取 obj1 的弱指针指向的对象的原始指针（未使用结果）
  obj1.weak.lock().get();
  // 将 obj1 的弱指针移动给 obj2，预期 obj2 成为有效的弱指针
  obj2 = std::move(obj1.weak);
  // 断言 obj2 成为有效的弱指针
  EXPECT_FALSE(obj2.expired());
}

TEST(
    WeakIntrusivePtrTest,
    givenWeakOnlyPtr_whenMoveAssigning_thenPointsToSameObject) {
  // 创建一个包含有效弱指针的对象
  IntrusiveAndWeak<SomeClass> obj1 = make_weak_intrusive<SomeClass>();
  // 创建一个仅有弱指针的对象
  weak_intrusive_ptr<SomeClass> obj2 = make_weak_only<SomeClass>();
  // 获取 obj1 的弱指针指向的对象的原始指针
  SomeClass* obj1ptr = obj1.weak.lock().get();
  // 将 obj1 的弱指针移动给 obj2，预期 obj2 指向的对象与 obj1 指向的对象相同
  obj2 = std::move(obj1.weak);
  // 断言 obj2 指向的对象与 obj1 指向的对象相同
  EXPECT_EQ(obj1ptr, obj2.lock().get());
}

TEST(
    WeakIntrusivePtrTest,
    givenWeakOnlyPtr_whenMoveAssigningToSelf_thenStaysInvalid) {
  // 创建一个仅有弱指针的对象
  weak_intrusive_ptr<SomeClass> obj1 = make_weak_only<SomeClass>();
  // 获取 obj1 的弱指针指向的对象的原始指针（未使用结果）
  obj1.lock().get();
  // 将 obj1 移动给自身，预期弱指针变为无效
  obj1 = std::move(obj1);
  // 断言弱指针已经变为无效
  EXPECT_TRUE(obj1.expired());
}

TEST(
    WeakIntrusivePtrTest,
    givenWeakOnlyPtr_whenMoveAssigningToSelf_thenPointsToSameObject) {
  // 创建一个仅有弱指针的对象
  weak_intrusive_ptr<SomeClass> obj1 = make_weak_only<SomeClass>();
  // 获取 obj1 的弱指针指向的对象的原始指针
  SomeClass* obj1ptr = obj1.lock().get();
  // 将 obj1 移动给自身，预期 obj1 指向的对象与原始指针相同
  obj1 = std::move(obj1);
  // 断言 obj1 指向的对象与原始指针相同
  EXPECT_EQ(obj1ptr, obj1.lock().get());
}

TEST(
    WeakIntrusivePtrTest,
    givenValidPtr_whenMoveAssigningFromInvalidPtr_thenNewInstanceIsInvalid) {
  // 创建一个无效的弱指针对象
  weak_intrusive_ptr<SomeClass> obj1 = make_invalid_weak<SomeClass>();
  // 创建一个包含有效弱指针的对象
  IntrusiveAndWeak<SomeClass> obj2 = make_weak_intrusive<SomeClass>();
  // 断言 obj2 的弱指针仍然有效
  EXPECT_FALSE(obj2.weak.expired());
  // 将 obj1 的弱指针移动给 obj2，预期 obj2 的弱指针变为无效
  obj2.weak = std::move(obj1);
  // 断言 obj2 的弱指针变为无效
  EXPECT_TRUE(obj2.weak.expired());
}
    // 给定一个有效的弱指针时，从弱指针赋值给新实例后，该新实例应该无效
    givenValidPtr_whenMoveAssigningFromWeakOnlyPtr_thenNewInstanceIsInvalid) {
      // 创建一个仅包含弱指针的对象，并初始化为SomeClass类型的弱指针
      weak_intrusive_ptr<SomeClass> obj1 = make_weak_only<SomeClass>();
      // 创建一个同时包含强指针和弱指针的对象，并初始化为SomeClass类型的强弱指针对
      IntrusiveAndWeak<SomeClass> obj2 = make_weak_intrusive<SomeClass>();
      // 断言obj2的弱指针没有过期（即仍然有效）
      EXPECT_FALSE(obj2.weak.expired());
      // 使用std::move将obj1的值移动给obj2的弱指针
      obj2.weak = std::move(obj1);
      // 断言obj2的弱指针已经过期（即无效）
      EXPECT_TRUE(obj2.weak.expired());
}

TEST(
    WeakIntrusivePtrTest,
    givenValidPtr_whenMoveAssigningToBaseClass_thenPointsToSameObject) {
  // 创建一个带有弱引用的指向 SomeChildClass 类型对象的实例 obj1
  IntrusiveAndWeak<SomeChildClass> obj1 =
      make_weak_intrusive<SomeChildClass>(1);
  // 创建一个带有弱引用的指向 SomeBaseClass 类型对象的实例 obj2
  IntrusiveAndWeak<SomeBaseClass> obj2 = make_weak_intrusive<SomeBaseClass>(2);
  // 获取 obj1 弱引用指向的对象的原始指针
  SomeBaseClass* obj1ptr = obj1.weak.lock().get();
  // 将 obj1 的弱引用移动给 obj2
  obj2.weak = std::move(obj1.weak);
  // 断言移动后 obj2 弱引用指向的对象的原始指针与 obj1ptr 相同
  EXPECT_EQ(obj1ptr, obj2.weak.lock().get());
  // 断言 obj2 弱引用指向的对象的 v 属性为 1
  EXPECT_EQ(1, obj2.weak.lock()->v);
}

TEST(
    WeakIntrusivePtrTest,
    givenValidPtr_whenMoveAssigningToBaseClass_thenOldInstanceInvalid) {
  // 创建一个带有弱引用的指向 SomeChildClass 类型对象的实例 obj1
  IntrusiveAndWeak<SomeChildClass> obj1 =
      make_weak_intrusive<SomeChildClass>(1);
  // 创建一个带有弱引用的指向 SomeBaseClass 类型对象的实例 obj2
  IntrusiveAndWeak<SomeBaseClass> obj2 = make_weak_intrusive<SomeBaseClass>(2);
  // 将 obj1 的弱引用移动给 obj2
  obj2.weak = std::move(obj1.weak);
  // 断言移动后 obj1 的弱引用已经失效
  EXPECT_TRUE(obj1.weak.expired());
}

TEST(
    WeakIntrusivePtrTest,
    givenInvalidPtr_whenMoveAssigningToBaseClass_thenNewInstanceIsValid) {
  // 创建一个带有弱引用的指向 SomeChildClass 类型对象的实例 obj1
  IntrusiveAndWeak<SomeChildClass> obj1 =
      make_weak_intrusive<SomeChildClass>(5);
  // 创建一个无效的弱引用指针，指向 SomeBaseClass 类型对象的实例 obj2
  weak_intrusive_ptr<SomeBaseClass> obj2 = make_invalid_weak<SomeBaseClass>();
  // 获取 obj1 弱引用指向的对象的原始指针
  obj1.weak.lock().get();
  // 将 obj1 的弱引用移动给 obj2
  obj2 = std::move(obj1.weak);
  // 断言移动后 obj2 弱引用指向的对象不再是无效的
  EXPECT_FALSE(obj2.expired());
}

TEST(
    WeakIntrusivePtrTest,
    givenInvalidPtr_whenMoveAssigningToBaseClass_thenPointsToSameObject) {
  // 创建一个带有弱引用的指向 SomeChildClass 类型对象的实例 obj1
  IntrusiveAndWeak<SomeChildClass> obj1 =
      make_weak_intrusive<SomeChildClass>(5);
  // 创建一个无效的弱引用指针，指向 SomeBaseClass 类型对象的实例 obj2
  weak_intrusive_ptr<SomeBaseClass> obj2 = make_invalid_weak<SomeBaseClass>();
  // 获取 obj1 弱引用指向的对象的原始指针
  SomeBaseClass* obj1ptr = obj1.weak.lock().get();
  // 将 obj1 的弱引用移动给 obj2
  obj2 = std::move(obj1.weak);
  // 断言移动后 obj2 弱引用指向的对象的原始指针与 obj1ptr 相同
  EXPECT_EQ(obj1ptr, obj2.lock().get());
  // 断言 obj2 弱引用指向的对象的 v 属性为 5
  EXPECT_EQ(5, obj2.lock()->v);
}

TEST(
    WeakIntrusivePtrTest,
    givenInvalidPtr_whenMoveAssigningInvalidPtrToBaseClass_thenNewInstanceIsValid) {
  // 创建一个无效的弱引用指针，指向 SomeChildClass 类型对象的实例 obj1
  weak_intrusive_ptr<SomeChildClass> obj1 = make_invalid_weak<SomeChildClass>();
  // 创建一个带有弱引用的指向 SomeBaseClass 类型对象的实例 obj2
  IntrusiveAndWeak<SomeBaseClass> obj2 = make_weak_intrusive<SomeBaseClass>(2);
  // 断言 obj2 的弱引用指向的对象不是无效的
  EXPECT_FALSE(obj2.weak.expired());
  // 将 obj1 的弱引用移动给 obj2
  obj2.weak = std::move(obj1);
  // 断言移动后 obj2 的弱引用已经失效
  EXPECT_TRUE(obj2.weak.expired());
}

TEST(
    WeakIntrusivePtrTest,
    givenWeakOnlyPtr_whenMoveAssigningToBaseClass_thenNewInstanceIsValid) {
  // 创建一个带有弱引用的指向 SomeChildClass 类型对象的实例 obj1
  IntrusiveAndWeak<SomeChildClass> obj1 =
      make_weak_intrusive<SomeChildClass>(5);
  // 创建一个仅带有弱引用的指向 SomeBaseClass 类型对象的实例 obj2
  weak_intrusive_ptr<SomeBaseClass> obj2 = make_weak_only<SomeBaseClass>(2);
  // 获取 obj1 弱引用指向的对象的原始指针
  obj1.weak.lock().get();
  // 将 obj1 的弱引用移动给 obj2
  obj2 = std::move(obj1.weak);
  // 断言移动后 obj2 弱引用指向的对象不是无效的
  EXPECT_FALSE(obj2.expired());
}

TEST(
    WeakIntrusivePtrTest,
    givenWeakOnlyPtr_whenMoveAssigningToBaseClass_thenPointsToSameObject) {
  // 创建一个带有弱引用的指向 SomeChildClass 类型对象的实例 obj1
  IntrusiveAndWeak<SomeChildClass> obj1 =
      make_weak_intrusive<SomeChildClass>(5);
  // 创建一个仅带有弱引用的指向 SomeBaseClass 类型对象的实例 obj2
  weak_intrusive_ptr<SomeBaseClass> obj2 = make_weak_only<SomeBaseClass>(2);
  // 获取 obj1 弱引用指向的对象的原始指针
  SomeBaseClass* obj1ptr = obj1.weak.lock().get();
  // 将 obj1 的弱引用移动给 obj2
  obj2 = std::move(obj1.weak);
  // 断言移动后 obj2 弱引用指向的对象的原始指针与 obj1ptr 相同
  EXPECT_EQ(obj1ptr, obj2.lock().get());
  // 断言 obj2 弱引用指向的对象的 v 属性为 5
  EXPECT_EQ(5, obj2.lock()->v);
}

TEST(
    WeakIntrusivePtrTest,
    givenWeakOnlyPtr_whenMoveAssigningInvalidPtrToBaseClass_thenNewInstanceIsValid) {
  // 创建一个指向 SomeChildClass 对象的弱指针 obj1，使用 make_weak_only 函数
  weak_intrusive_ptr<SomeChildClass> obj1 = make_weak_only<SomeChildClass>(5);
  // 创建一个包含弱指针的结构体 IntrusiveAndWeak，指向 SomeBaseClass 对象，使用 make_weak_intrusive 函数
  IntrusiveAndWeak<SomeBaseClass> obj2 = make_weak_intrusive<SomeBaseClass>(2);
  // 断言 obj2 的弱指针没有过期
  EXPECT_FALSE(obj2.weak.expired());
  // 将 obj1 的内容移动到 obj2 的弱指针 weak 中
  obj2.weak = std::move(obj1);
  // 断言 obj2 的弱指针已经过期
  EXPECT_TRUE(obj2.weak.expired());
}

TEST(
    WeakIntrusivePtrTest,
    givenNullPtr_whenMoveAssigningToDifferentNullptr_thenHasNewNullptr) {
  // 创建一个无效的弱指针对象 obj1，使用 NullType1 作为空指针类型
  weak_intrusive_ptr<SomeClass, NullType1> obj1 =
      make_invalid_weak<SomeClass, NullType1>();
  // 创建一个无效的弱指针对象 obj2，使用 NullType2 作为空指针类型
  weak_intrusive_ptr<SomeClass, NullType2> obj2 =
      make_invalid_weak<SomeClass, NullType2>();
  // 将 obj1 移动给 obj2
  obj2 = std::move(obj1);
  // 断言 NullType1 的单例不等于 NullType2 的单例
  EXPECT_NE(NullType1::singleton(), NullType2::singleton());
  // 断言 obj1 已经失效
  // NOLINTNEXTLINE(bugprone-use-after-move)
  EXPECT_TRUE(obj1.expired());
  // 断言 obj2 已经失效
  EXPECT_TRUE(obj2.expired());
}

TEST(
    WeakIntrusivePtrTest,
    givenValidPtr_whenCopyAssigning_thenPointsToSameObject) {
  // 创建一个强指针和弱指针对象 obj1，并将其设为有效
  IntrusiveAndWeak<SomeClass> obj1 = make_weak_intrusive<SomeClass>();
  // 创建一个强指针和弱指针对象 obj2，并将其设为有效
  IntrusiveAndWeak<SomeClass> obj2 = make_weak_intrusive<SomeClass>();
  // 获取 obj1 的弱指针指向的对象的指针
  SomeClass* obj1ptr = obj1.weak.lock().get();
  // 将 obj1 的弱指针赋值给 obj2 的弱指针
  obj2.weak = obj1.weak;
  // 断言 obj1 和 obj2 的弱指针指向同一个对象
  EXPECT_EQ(obj1ptr, obj2.weak.lock().get());
}

TEST(
    WeakIntrusivePtrTest,
    givenValidPtr_whenCopyAssigning_thenOldInstanceValid) {
  // 创建一个强指针和弱指针对象 obj1，并将其设为有效
  IntrusiveAndWeak<SomeClass> obj1 = make_weak_intrusive<SomeClass>();
  // 创建一个强指针和弱指针对象 obj2，并将其设为有效
  IntrusiveAndWeak<SomeClass> obj2 = make_weak_intrusive<SomeClass>();
  // 将 obj1 的弱指针赋值给 obj2 的弱指针
  obj2.weak = obj1.weak;
  // 断言 obj1 的弱指针仍然有效
  EXPECT_FALSE(obj1.weak.expired());
}

TEST(
    WeakIntrusivePtrTest,
    givenValidPtr_whenCopyAssigningToSelf_thenPointsToSameObject) {
  // 创建一个强指针和弱指针对象 obj1，并将其设为有效
  IntrusiveAndWeak<SomeClass> obj1 = make_weak_intrusive<SomeClass>();
  // 获取 obj1 的弱指针指向的对象的指针
  SomeClass* obj1ptr = obj1.weak.lock().get();
  // 将 obj1 的弱指针赋值给自身的弱指针
  obj1.weak = obj1.weak;
  // 断言 obj1 的弱指针指向同一个对象
  EXPECT_EQ(obj1ptr, obj1.weak.lock().get());
}

TEST(
    WeakIntrusivePtrTest,
    givenValidPtr_whenCopyAssigningToSelf_thenStaysValid) {
  // 创建一个强指针和弱指针对象 obj1，并将其设为有效
  IntrusiveAndWeak<SomeClass> obj1 = make_weak_intrusive<SomeClass>();
  // 将 obj1 的弱指针赋值给自身的弱指针
  obj1.weak = obj1.weak;
  // 断言 obj1 的弱指针仍然有效
  EXPECT_FALSE(obj1.weak.expired());
}

TEST(
    WeakIntrusivePtrTest,
    givenInvalidPtr_whenCopyAssigning_thenNewInstanceIsValid) {
  // 创建一个强指针和弱指针对象 obj1，并将其设为有效
  IntrusiveAndWeak<SomeClass> obj1 = make_weak_intrusive<SomeClass>();
  // 创建一个无效的弱指针对象 obj2
  weak_intrusive_ptr<SomeClass> obj2 = make_invalid_weak<SomeClass>();
  // 获取 obj1 的弱指针指向的对象的指针
  obj1.weak.lock().get();
  // 将 obj1 的弱指针赋值给 obj2
  obj2 = obj1.weak;
  // 断言 obj2 的弱指针仍然有效
  EXPECT_FALSE(obj2.expired());
}

TEST(
    WeakIntrusivePtrTest,
    givenInvalidPtr_whenCopyAssigningToSelf_thenStaysInvalid) {
  // 创建一个无效的弱指针对象 obj1
  weak_intrusive_ptr<SomeClass> obj1 = make_invalid_weak<SomeClass>();
  // NOLINTNEXTLINE(clang-diagnostic-self-assign-overloaded)
  // 将 obj1 赋值给自身
  obj1 = obj1;
  // 断言 obj1 已经失效
  EXPECT_TRUE(obj1.expired());
}

TEST(
    WeakIntrusivePtrTest,
    givenWeakOnlyPtr_whenCopyAssigning_thenNewInstanceIsValid) {
  // 创建一个强指针和弱指针对象 obj1，并将其设为有效
  IntrusiveAndWeak<SomeClass> obj1 = make_weak_intrusive<SomeClass>();
  // 创建一个只有弱指针的对象 obj2
  weak_intrusive_ptr<SomeClass> obj2 = make_weak_only<SomeClass>();
  // 获取 obj1 的弱指针指向的对象的指针
  obj1.weak.lock().get();
  // 将 obj1 的弱指针赋值给 obj2
  obj2 = obj1.weak;
  // 断言 obj2 的弱指针仍然有效
  EXPECT_FALSE(obj2.expired());
}

TEST(
    WeakIntrusivePtrTest,
    givenWeakOnlyPtr_whenCopyAssigning_thenPointsToSameObject) {
  // 创建一个强指针和弱指针对象 obj1，并将其设为有效
  IntrusiveAndWeak<SomeClass> obj1 = make_weak_intrusive<SomeClass>();
  // 创建一个只有弱指针的对象 obj2
  weak_intrusive_ptr<SomeClass> obj2 = make_weak_only<SomeClass>();
  // 获取 obj1 的弱指针指向的对象的指针
  SomeClass* obj1ptr = obj1.weak.lock().get();
  // 将 obj1 的弱指针赋值给 obj2
  obj2 = obj1.weak;
  // 断言 obj1 和 obj2 的弱指针指向同一个对象
  EXPECT_EQ(obj1ptr, obj2.lock().get());
}

TEST(
    WeakIntrusivePtrTest,
    givenWeakOnlyPtr_whenCopyAssigningToSelf_thenStaysInvalid) {
  // 创建一个弱指针，指向通过 make_weak_only 创建的 SomeClass 实例
  weak_intrusive_ptr<SomeClass> obj1 = make_weak_only<SomeClass>();
  // 获取 obj1 的锁定的 shared_ptr，然后调用 get() 方法
  obj1.lock().get();
  // 忽略 Clang 自诊断中的自赋值警告
  // 将 obj1 赋值给自身，但实际上不会改变 obj1 的状态
  obj1 = obj1;
  // 验证 obj1 是否已经过期（即是否指向的对象已被销毁）
  EXPECT_TRUE(obj1.expired());
TEST(
    WeakIntrusivePtrTest,
    givenWeakOnlyPtr_whenCopyAssigningToSelf_thenPointsToSameObject) {
  // 创建一个弱指针，指向一个新的 SomeClass 对象
  weak_intrusive_ptr<SomeClass> obj1 = make_weak_only<SomeClass>();
  // 获取 obj1 指向的对象的裸指针
  SomeClass* obj1ptr = obj1.lock().get();
  // 自我赋值，虽然是无意义的操作，但不会改变指针指向的对象
  obj1 = obj1;
  // 断言 obj1 指针指向的对象依然是 obj1ptr 指向的对象
  EXPECT_EQ(obj1ptr, obj1.lock().get());
}

TEST(
    WeakIntrusivePtrTest,
    givenValidPtr_whenCopyAssigningToBaseClass_thenPointsToSameObject) {
  // 创建一个 IntrusiveAndWeak 包含一个 SomeChildClass 的强弱指针
  IntrusiveAndWeak<SomeChildClass> child =
      make_weak_intrusive<SomeChildClass>(3);
  // 创建一个 IntrusiveAndWeak 包含一个 SomeBaseClass 的强弱指针
  IntrusiveAndWeak<SomeBaseClass> base = make_weak_intrusive<SomeBaseClass>(10);
  // 将 base 的弱指针指向 child 的弱指针
  base.weak = child.weak;
  // 断言 base.weak 指向的对象的某个值等于 3
  EXPECT_EQ(3, base.weak.lock()->v);
}

TEST(
    WeakIntrusivePtrTest,
    givenValidPtr_whenCopyAssigningToBaseClass_thenOldInstanceInvalid) {
  // 创建一个 IntrusiveAndWeak 包含一个 SomeChildClass 的强弱指针
  IntrusiveAndWeak<SomeChildClass> obj1 =
      make_weak_intrusive<SomeChildClass>(3);
  // 创建一个 IntrusiveAndWeak 包含一个 SomeBaseClass 的强弱指针
  IntrusiveAndWeak<SomeBaseClass> obj2 = make_weak_intrusive<SomeBaseClass>(10);
  // 将 obj2 的弱指针指向 obj1 的弱指针
  obj2.weak = obj1.weak;
  // 断言 obj1 的弱指针并未过期
  EXPECT_FALSE(obj1.weak.expired());
}

TEST(
    WeakIntrusivePtrTest,
    givenInvalidPtr_whenCopyAssigningToBaseClass_thenNewInstanceIsValid) {
  // 创建一个 IntrusiveAndWeak 包含一个 SomeChildClass 的强弱指针
  IntrusiveAndWeak<SomeChildClass> obj1 =
      make_weak_intrusive<SomeChildClass>(5);
  // 创建一个无效的弱指针，指向 SomeBaseClass
  weak_intrusive_ptr<SomeBaseClass> obj2 = make_invalid_weak<SomeBaseClass>();
  // 获取 obj1 的弱指针的裸指针，虽然不使用它
  obj1.weak.lock().get();
  // 将 obj2 的值赋为 obj1 的弱指针
  obj2 = obj1.weak;
  // 断言 obj2 指针并未过期
  EXPECT_FALSE(obj2.expired());
}

TEST(
    WeakIntrusivePtrTest,
    givenInvalidPtr_whenCopyAssigningToBaseClass_thenPointsToSameObject) {
  // 创建一个 IntrusiveAndWeak 包含一个 SomeChildClass 的强弱指针
  IntrusiveAndWeak<SomeChildClass> obj1 =
      make_weak_intrusive<SomeChildClass>(5);
  // 创建一个无效的弱指针，指向 SomeBaseClass
  weak_intrusive_ptr<SomeBaseClass> obj2 = make_invalid_weak<SomeBaseClass>();
  // 获取 obj1 的弱指针的裸指针
  SomeBaseClass* obj1ptr = obj1.weak.lock().get();
  // 将 obj2 的值赋为 obj1 的弱指针
  obj2 = obj1.weak;
  // 断言 obj2 指向的对象与 obj1 指向的对象相同
  EXPECT_EQ(obj1ptr, obj2.lock().get());
  // 断言 obj2 指向的对象的某个值等于 5
  EXPECT_EQ(5, obj2.lock()->v);
}

TEST(
    WeakIntrusivePtrTest,
    givenPtr_whenCopyAssigningInvalidPtrToBaseClass_thenNewInstanceIsInvalid) {
  // 创建一个无效的弱指针，指向 SomeChildClass
  weak_intrusive_ptr<SomeChildClass> obj1 = make_invalid_weak<SomeChildClass>();
  // 创建一个 IntrusiveAndWeak 包含一个 SomeBaseClass 的强弱指针
  IntrusiveAndWeak<SomeBaseClass> obj2 = make_weak_intrusive<SomeBaseClass>(2);
  // 断言 obj2 的弱指针并未过期
  EXPECT_FALSE(obj2.weak.expired());
  // 将 obj2 的弱指针指向 obj1
  obj2.weak = obj1;
  // 断言 obj2 的弱指针已过期
  EXPECT_TRUE(obj2.weak.expired());
}

TEST(
    WeakIntrusivePtrTest,
    givenWeakOnlyPtr_whenCopyAssigningToBaseClass_thenNewInstanceIsValid) {
  // 创建一个 IntrusiveAndWeak 包含一个 SomeChildClass 的强弱指针
  IntrusiveAndWeak<SomeChildClass> obj1 =
      make_weak_intrusive<SomeChildClass>(5);
  // 创建一个仅弱指针，指向一个新的 SomeBaseClass 对象
  weak_intrusive_ptr<SomeBaseClass> obj2 = make_weak_only<SomeBaseClass>(2);
  // 获取 obj1 的弱指针的裸指针，虽然不使用它
  obj1.weak.lock().get();
  // 将 obj2 的值赋为 obj1 的弱指针
  obj2 = obj1.weak;
  // 断言 obj2 指针并未过期
  EXPECT_FALSE(obj2.expired());
}

TEST(
    WeakIntrusivePtrTest,
    givenWeakOnlyPtr_whenCopyAssigningToBaseClass_thenPointsToSameObject) {
  // 创建一个 IntrusiveAndWeak 包含一个 SomeChildClass 的强弱指针
  IntrusiveAndWeak<SomeChildClass> obj1 =
      make_weak_intrusive<SomeChildClass>(5);
  // 创建一个仅弱指针，指向一个新的 SomeBaseClass 对象
  weak_intrusive_ptr<SomeBaseClass> obj2 = make_weak_only<SomeBaseClass>(2);
  // 获取 obj1 的弱指针的裸指针
  SomeBaseClass* obj1ptr = obj1.weak.lock().get();
  // 将 obj2 的值赋为 obj1 的弱指针
  obj2 = obj1.weak;
  // 断言 obj2 指向的对象与 obj1 指向的对象相同
  EXPECT_EQ(obj1ptr, obj2.lock().get());
  // 断言 obj2 指向的对象的某个值等于 5
  EXPECT_EQ(5, obj2.lock()->v);
}
    givenPtr_whenCopyAssigningWeakOnlyPtrToBaseClass_thenNewInstanceIsValid) {
  # 创建一个名为 obj1 的弱指针，指向 SomeChildClass 类型的对象，对象内部值为 2
  weak_intrusive_ptr<SomeChildClass> obj1 = make_weak_only<SomeChildClass>(2);
  # 创建一个名为 obj2 的 IntrusiveAndWeak 对象，包含一个弱指针和一个强指针，指向 SomeBaseClass 类型的对象，对象内部值为 2
  IntrusiveAndWeak<SomeBaseClass> obj2 = make_weak_intrusive<SomeBaseClass>(2);
  # 断言 obj2 的弱指针未过期（即指向的对象仍然存在）
  EXPECT_FALSE(obj2.weak.expired());
  # 将 obj1 的弱指针赋值给 obj2 的弱指针
  obj2.weak = obj1;
  # 断言 obj2 的弱指针已经过期（因为它现在指向了 obj1 的对象，而不再是原来的对象）
  EXPECT_TRUE(obj2.weak.expired());
}

TEST(
    WeakIntrusivePtrTest,
    givenNullPtr_whenCopyAssigningToDifferentNullptr_thenHasNewNullptr) {
  // 创建一个空的弱指针对象，指向某个特定类型的无效实例
  weak_intrusive_ptr<SomeClass, NullType1> obj1 =
      make_invalid_weak<SomeClass, NullType1>();
  // 创建另一个空的弱指针对象，指向另一个特定类型的无效实例
  weak_intrusive_ptr<SomeClass, NullType2> obj2 =
      make_invalid_weak<SomeClass, NullType2>();
  // 将第一个弱指针对象指向的实例复制给第二个弱指针对象
  obj2 = obj1;
  // 检查两种空指针类型的单例不相等
  EXPECT_NE(NullType1::singleton(), NullType2::singleton());
  // 检查第一个弱指针对象是否已经过期
  EXPECT_TRUE(obj1.expired());
  // 检查第二个弱指针对象是否已经过期
  EXPECT_TRUE(obj2.expired());
}

TEST(
    WeakIntrusivePtrTest,
    givenPtr_whenMoveConstructing_thenPointsToSameObject) {
  // 创建一个包含强引用和弱引用的对象，并使用弱指针包装其弱引用
  IntrusiveAndWeak<SomeClass> obj1 = make_weak_intrusive<SomeClass>();
  // 获取第一个对象的弱指针指向的原始指针
  SomeClass* obj1ptr = obj1.weak.lock().get();
  // 使用 std::move 将第一个对象的弱指针移动给第二个弱指针对象
  weak_intrusive_ptr<SomeClass> obj2 = std::move(obj1.weak);
  // 检查移动后第二个弱指针对象是否指向与第一个对象相同的实例
  EXPECT_EQ(obj1ptr, obj2.lock().get());
}

TEST(
    WeakIntrusivePtrTest,
    givenPtr_whenMoveConstructing_thenOldInstanceInvalid) {
  // 创建一个包含强引用和弱引用的对象，并使用弱指针包装其弱引用
  IntrusiveAndWeak<SomeClass> obj1 = make_weak_intrusive<SomeClass>();
  // 使用 std::move 将第一个对象的弱指针移动给第二个弱指针对象
  weak_intrusive_ptr<SomeClass> obj2 = std::move(obj1.weak);
  // 检查移动后第一个对象的弱引用是否已经过期
  EXPECT_TRUE(obj1.weak.expired());
}

TEST(WeakIntrusivePtrTest, givenPtr_whenMoveConstructing_thenNewInstanceValid) {
  // 创建一个包含强引用和弱引用的对象，并使用弱指针包装其弱引用
  IntrusiveAndWeak<SomeClass> obj1 = make_weak_intrusive<SomeClass>();
  // 使用 std::move 将第一个对象的弱指针移动给第二个弱指针对象
  weak_intrusive_ptr<SomeClass> obj2 = std::move(obj1.weak);
  // 检查移动后第二个弱指针对象是否有效（未过期）
  EXPECT_FALSE(obj2.expired());
}

TEST(
    WeakIntrusivePtrTest,
    givenPtr_whenMoveConstructingFromInvalidPtr_thenNewInstanceInvalid) {
  // 创建一个指向无效实例的弱指针对象
  weak_intrusive_ptr<SomeClass> obj1 = make_invalid_weak<SomeClass>();
  // 使用 std::move 将第一个对象的弱指针移动给第二个弱指针对象
  weak_intrusive_ptr<SomeClass> obj2 = std::move(obj1);
  // 检查移动后第二个弱指针对象是否已经过期
  EXPECT_TRUE(obj2.expired());
}

TEST(
    WeakIntrusivePtrTest,
    givenPtr_whenMoveConstructingFromWeakOnlyPtr_thenNewInstanceInvalid) {
  // 创建一个只包含弱引用的弱指针对象
  weak_intrusive_ptr<SomeClass> obj1 = make_weak_only<SomeClass>();
  // 使用 std::move 将第一个对象的弱指针移动给第二个弱指针对象
  weak_intrusive_ptr<SomeClass> obj2 = std::move(obj1);
  // 检查移动后第二个弱指针对象是否已经过期
  EXPECT_TRUE(obj2.expired());
}

TEST(
    WeakIntrusivePtrTest,
    givenPtr_whenMoveConstructingToBaseClass_thenPointsToSameObject) {
  // 创建一个包含强引用和弱引用的派生类对象，并使用弱指针包装其弱引用
  IntrusiveAndWeak<SomeChildClass> child =
      make_weak_intrusive<SomeChildClass>(3);
  // 获取子类对象的弱指针指向的原始指针
  SomeBaseClass* objptr = child.weak.lock().get();
  // 使用 std::move 将子类对象的弱指针移动给基类的弱指针对象
  weak_intrusive_ptr<SomeBaseClass> base = std::move(child.weak);
  // 检查移动后基类的弱指针对象是否指向与子类对象相同的实例
  EXPECT_EQ(3, base.lock()->v);
  EXPECT_EQ(objptr, base.lock().get());
}

TEST(
    WeakIntrusivePtrTest,
    givenPtr_whenMoveConstructingToBaseClass_thenOldInstanceInvalid) {
  // 创建一个包含强引用和弱引用的派生类对象，并使用弱指针包装其弱引用
  IntrusiveAndWeak<SomeChildClass> child =
      make_weak_intrusive<SomeChildClass>(3);
  // 使用 std::move 将子类对象的弱指针移动给基类的弱指针对象
  weak_intrusive_ptr<SomeBaseClass> base = std::move(child.weak);
  // 检查移动后子类对象的弱引用是否已经过期
  EXPECT_TRUE(child.weak.expired());
}

TEST(
    WeakIntrusivePtrTest,
    givenPtr_whenMoveConstructingToBaseClass_thenNewInstanceValid) {
  // 创建一个包含强引用和弱引用的派生类对象，并使用弱指针包装其弱引用
  IntrusiveAndWeak<SomeChildClass> obj1 =
      make_weak_intrusive<SomeChildClass>(2);
  // 使用 std::move 将派生类对象的弱指针移动给基类的弱指针对象
  weak_intrusive_ptr<SomeBaseClass> obj2 = std::move(obj1.weak);
  // 检查移动后基类的弱指针对象是否有效（未过期）
  EXPECT_FALSE(obj2.expired());
}

TEST(
    WeakIntrusivePtrTest,
    givenPtr_whenMoveConstructingToBaseClassFromInvalidPtr_thenNewInstanceInvalid) {
  # 创建一个无效状态的 SomeChildClass 对象的弱指针
  weak_intrusive_ptr<SomeChildClass> obj1 = make_invalid_weak<SomeChildClass>();
  # 将 obj1 移动构造到 SomeBaseClass 类型的弱指针 obj2 中
  weak_intrusive_ptr<SomeBaseClass> obj2 = std::move(obj1);
  # 断言 obj2 弱指针已经过期（即指向的对象无效）
  EXPECT_TRUE(obj2.expired());
}

TEST(
    WeakIntrusivePtrTest,
    givenPtr_whenMoveConstructingToBaseClassFromWeakOnlyPtr_thenNewInstanceInvalid) {
  // 创建一个指向 SomeChildClass 对象的弱指针
  weak_intrusive_ptr<SomeChildClass> obj1 = make_weak_only<SomeChildClass>(2);
  // 将 obj1 移动到指向 SomeBaseClass 的弱指针 obj2
  weak_intrusive_ptr<SomeBaseClass> obj2 = std::move(obj1);
  // 验证 obj2 是否已经失效（指向对象的引用已经释放）
  EXPECT_TRUE(obj2.expired());
}

TEST(
    WeakIntrusivePtrTest,
    givenNullPtr_whenMoveConstructingToDifferentNullptr_thenHasNewNullptr) {
  // 创建一个无效的弱指针，指向 SomeClass 和 NullType1
  weak_intrusive_ptr<SomeClass, NullType1> obj1 =
      make_invalid_weak<SomeClass, NullType1>();
  // 将 obj1 移动到指向 SomeClass 和 NullType2 的弱指针 obj2
  weak_intrusive_ptr<SomeClass, NullType2> obj2 = std::move(obj1);
  // 验证 NullType1 和 NullType2 的单例不相等
  EXPECT_NE(NullType1::singleton(), NullType2::singleton());
  // NOLINTNEXTLINE(bugprone-use-after-move)
  // 验证 obj1 是否已经失效（移动后使用了 obj1）
  EXPECT_TRUE(obj1.expired());
  // 验证 obj2 是否已经失效
  EXPECT_TRUE(obj2.expired());
}

TEST(
    WeakIntrusivePtrTest,
    givenPtr_whenCopyConstructing_thenPointsToSameObject) {
  // 创建一个包含弱指针的对象
  IntrusiveAndWeak<SomeClass> obj1 = make_weak_intrusive<SomeClass>();
  // 获取 obj1 中的弱指针指向的对象的原始指针
  SomeClass* obj1ptr = obj1.weak.lock().get();
  // 使用 obj1 的弱指针创建一个新的弱指针 obj2
  weak_intrusive_ptr<SomeClass> obj2 = obj1.weak;
  // 验证 obj2 的锁定指针与 obj1ptr 相同
  EXPECT_EQ(obj1ptr, obj2.lock().get());
  // 验证 obj1 的弱指针是否有效
  EXPECT_FALSE(obj1.weak.expired());
}

TEST(WeakIntrusivePtrTest, givenPtr_whenCopyConstructing_thenOldInstanceValid) {
  // 创建一个包含弱指针的对象
  IntrusiveAndWeak<SomeClass> obj1 = make_weak_intrusive<SomeClass>();
  // 使用 obj1 的弱指针创建一个新的弱指针 obj2
  weak_intrusive_ptr<SomeClass> obj2 = obj1.weak;
  // 验证 obj1 的弱指针是否有效
  EXPECT_FALSE(obj1.weak.expired());
}

TEST(WeakIntrusivePtrTest, givenPtr_whenCopyConstructing_thenNewInstanceValid) {
  // 创建一个包含弱指针的对象
  IntrusiveAndWeak<SomeClass> obj1 = make_weak_intrusive<SomeClass>();
  // 使用 obj1 的弱指针创建一个新的弱指针 obj2
  weak_intrusive_ptr<SomeClass> obj2 = obj1.weak;
  // 验证 obj2 的弱指针是否有效
  EXPECT_FALSE(obj2.expired());
}

TEST(
    WeakIntrusivePtrTest,
    givenPtr_whenCopyConstructingFromInvalidPtr_thenNewInstanceInvalid) {
  // 创建一个无效的弱指针
  weak_intrusive_ptr<SomeClass> obj1 = make_invalid_weak<SomeClass>();
  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
  // 使用 obj1 的弱指针创建一个新的弱指针 obj2
  weak_intrusive_ptr<SomeClass> obj2 = obj1;
  // 验证 obj2 是否已经失效
  EXPECT_TRUE(obj2.expired());
}

TEST(
    WeakIntrusivePtrTest,
    givenPtr_whenCopyConstructingFromWeakOnlyPtr_thenNewInstanceInvalid) {
  // 创建一个只能弱指针
  weak_intrusive_ptr<SomeClass> obj1 = make_weak_only<SomeClass>();
  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
  // 使用 obj1 的弱指针创建一个新的弱指针 obj2
  weak_intrusive_ptr<SomeClass> obj2 = obj1;
  // 验证 obj2 是否已经失效
  EXPECT_TRUE(obj2.expired());
}

TEST(
    WeakIntrusivePtrTest,
    givenPtr_whenCopyConstructingToBaseClass_thenPointsToSameObject) {
  // 创建一个包含弱指针的对象，指向 SomeChildClass
  IntrusiveAndWeak<SomeChildClass> child =
      make_weak_intrusive<SomeChildClass>(3);
  // 获取 child 中的弱指针指向的对象的原始指针
  SomeBaseClass* objptr = child.weak.lock().get();
  // 使用 child 的弱指针创建一个新的弱指针 base，指向 SomeBaseClass
  weak_intrusive_ptr<SomeBaseClass> base = child.weak;
  // 验证 base 指向的对象的 v 值为 3
  EXPECT_EQ(3, base.lock()->v);
  // 验证 base 的锁定指针与 objptr 相同
  EXPECT_EQ(objptr, base.lock().get());
}

TEST(
    WeakIntrusivePtrTest,
    givenPtr_whenCopyConstructingToBaseClass_thenOldInstanceInvalid) {
  // 创建一个包含弱指针的对象，指向 SomeChildClass
  IntrusiveAndWeak<SomeChildClass> child =
      make_weak_intrusive<SomeChildClass>(3);
  // 使用 child 的弱指针创建一个新的弱指针 base，指向 SomeBaseClass
  weak_intrusive_ptr<SomeBaseClass> base = child.weak;
  // 验证 child 的弱指针是否有效
  EXPECT_FALSE(child.weak.expired());
}

TEST(
    WeakIntrusivePtrTest,
    givenPtr_whenCopyConstructingToBaseClass_thenNewInstanceInvalid) {

该行代码不完整，需要上下文来理解。应当注意，代码片段似乎有语法错误或者不完整。
TEST(
    WeakIntrusivePtrTest,
    givenPtr_whenCopyConstructingToBaseClassFromInvalidPtr_thenNewInstanceInvalid) {
  // 创建一个无效的弱引用指针，指向 SomeChildClass
  weak_intrusive_ptr<SomeChildClass> obj1 = make_invalid_weak<SomeChildClass>();
  // 将 obj1 拷贝构造到基类 SomeBaseClass 的弱引用指针 obj2
  weak_intrusive_ptr<SomeBaseClass> obj2 = obj1;
  // 断言 obj2 已经过期（失效）
  EXPECT_TRUE(obj2.expired());
}

TEST(
    WeakIntrusivePtrTest,
    givenPtr_whenCopyConstructingToBaseClassFromWeakOnlyPtr_thenNewInstanceInvalid) {
  // 创建一个只包含弱引用的指针，指向 SomeChildClass
  weak_intrusive_ptr<SomeChildClass> obj1 = make_weak_only<SomeChildClass>(2);
  // 将 obj1 拷贝构造到基类 SomeBaseClass 的弱引用指针 obj2
  weak_intrusive_ptr<SomeBaseClass> obj2 = obj1;
  // 断言 obj2 已经过期（失效）
  EXPECT_TRUE(obj2.expired());
}

TEST(
    WeakIntrusivePtrTest,
    givenNullPtr_whenCopyConstructingToDifferentNullptr_thenHasNewNullptr) {
  // 创建一个包含空指针的弱引用指针，指向 SomeClass 和 NullType1
  weak_intrusive_ptr<SomeClass, NullType1> obj1 =
      make_invalid_weak<SomeClass, NullType1>();
  // 将 obj1 拷贝构造到不同空指针类型 NullType2 的弱引用指针 obj2
  weak_intrusive_ptr<SomeClass, NullType2> obj2 = obj1;
  // 断言 NullType1 和 NullType2 的单例不相等
  EXPECT_NE(NullType1::singleton(), NullType2::singleton());
  // 断言 obj1 和 obj2 已经过期（失效）
  EXPECT_TRUE(obj1.expired());
  EXPECT_TRUE(obj2.expired());
}

TEST(WeakIntrusivePtrTest, SwapFunction) {
  // 创建两个弱侵入式指针和弱引用指针的组合对象，指向 SomeClass
  IntrusiveAndWeak<SomeClass> obj1 = make_weak_intrusive<SomeClass>();
  IntrusiveAndWeak<SomeClass> obj2 = make_weak_intrusive<SomeClass>();
  // 获取 obj1 和 obj2 的弱引用指针的原始指针
  SomeClass* obj1ptr = obj1.weak.lock().get();
  SomeClass* obj2ptr = obj2.weak.lock().get();
  // 交换 obj1 和 obj2 的弱引用指针
  swap(obj1.weak, obj2.weak);
  // 断言交换后的结果正确
  EXPECT_EQ(obj2ptr, obj1.weak.lock().get());
  EXPECT_EQ(obj1ptr, obj2.weak.lock().get());
}

TEST(WeakIntrusivePtrTest, SwapMethod) {
  // 创建两个弱侵入式指针和弱引用指针的组合对象，指向 SomeClass
  IntrusiveAndWeak<SomeClass> obj1 = make_weak_intrusive<SomeClass>();
  IntrusiveAndWeak<SomeClass> obj2 = make_weak_intrusive<SomeClass>();
  // 获取 obj1 和 obj2 的弱引用指针的原始指针
  SomeClass* obj1ptr = obj1.weak.lock().get();
  SomeClass* obj2ptr = obj2.weak.lock().get();
  // 使用 swap 方法交换 obj1 和 obj2 的弱引用指针
  obj1.weak.swap(obj2.weak);
  // 断言交换后的结果正确
  EXPECT_EQ(obj2ptr, obj1.weak.lock().get());
  EXPECT_EQ(obj1ptr, obj2.weak.lock().get());
}

TEST(WeakIntrusivePtrTest, SwapFunctionFromInvalid) {
  // 创建一个无效的弱引用指针，指向 SomeClass
  weak_intrusive_ptr<SomeClass> obj1 = make_invalid_weak<SomeClass>();
  // 创建一个弱侵入式指针和弱引用指针的组合对象，指向 SomeClass
  IntrusiveAndWeak<SomeClass> obj2 = make_weak_intrusive<SomeClass>();
  // 获取 obj2 的弱引用指针的原始指针
  SomeClass* obj2ptr = obj2.weak.lock().get();
  // 使用 swap 函数交换 obj1 和 obj2 的指针
  swap(obj1, obj2.weak);
  // 断言交换后的结果正确
  EXPECT_EQ(obj2ptr, obj1.lock().get());
  EXPECT_FALSE(obj1.expired());
  EXPECT_TRUE(obj2.weak.expired());
}

TEST(WeakIntrusivePtrTest, SwapMethodFromInvalid) {
  // 创建一个无效的弱引用指针，指向 SomeClass
  weak_intrusive_ptr<SomeClass> obj1 = make_invalid_weak<SomeClass>();
  // 创建一个弱侵入式指针和弱引用指针的组合对象，指向 SomeClass
  IntrusiveAndWeak<SomeClass> obj2 = make_weak_intrusive<SomeClass>();
  // 获取 obj2 的弱引用指针的原始指针
  SomeClass* obj2ptr = obj2.weak.lock().get();
  // 使用 swap 方法交换 obj1 和 obj2 的指针
  obj1.swap(obj2.weak);
  // 断言交换后的结果正确
  EXPECT_EQ(obj2ptr, obj1.lock().get());
  EXPECT_FALSE(obj1.expired());
  EXPECT_TRUE(obj2.weak.expired());
}

TEST(WeakIntrusivePtrTest, SwapFunctionWithInvalid) {
  // 创建一个弱侵入式指针和弱引用指针的组合对象，指向 SomeClass
  IntrusiveAndWeak<SomeClass> obj1 = make_weak_intrusive<SomeClass>();
  // 创建一个无效的弱引用指针，指向 SomeClass
  weak_intrusive_ptr<SomeClass> obj2 = make_invalid_weak<SomeClass>();
  // 获取 obj1 的弱引用指针的原始指针
  SomeClass* obj1ptr = obj1.weak.lock().get();
  // 使用 swap 函数交换 obj1 和 obj2 的指针
  swap(obj1.weak, obj2);
  // 断言交换后的结果正确
  EXPECT_TRUE(obj1.weak.expired());
  EXPECT_FALSE(obj2.expired());
  EXPECT_EQ(obj1ptr, obj2.lock().get());
}
TEST(WeakIntrusivePtrTest, SwapMethodWithInvalid) {
  // 创建一个带有弱引用的对象，使用make_weak_intrusive函数
  IntrusiveAndWeak<SomeClass> obj1 = make_weak_intrusive<SomeClass>();
  // 创建一个无效的弱引用对象，使用make_invalid_weak函数
  weak_intrusive_ptr<SomeClass> obj2 = make_invalid_weak<SomeClass>();
  // 获取obj1的弱引用的原始指针
  SomeClass* obj1ptr = obj1.weak.lock().get();
  // 交换obj1和obj2的弱引用
  obj1.weak.swap(obj2);
  // 检查obj1的弱引用是否已过期
  EXPECT_TRUE(obj1.weak.expired());
  // 检查obj2的弱引用是否未过期
  EXPECT_FALSE(obj2.expired());
  // 检查obj1ptr和obj2的弱引用指向的对象是否相同
  EXPECT_EQ(obj1ptr, obj2.lock().get());
}

TEST(WeakIntrusivePtrTest, SwapFunctionInvalidWithInvalid) {
  // 创建一个无效的弱引用对象，使用make_invalid_weak函数
  weak_intrusive_ptr<SomeClass> obj1 = make_invalid_weak<SomeClass>();
  // 创建另一个无效的弱引用对象，使用make_invalid_weak函数
  weak_intrusive_ptr<SomeClass> obj2 = make_invalid_weak<SomeClass>();
  // 交换obj1和obj2的值
  swap(obj1, obj2);
  // 检查obj1的弱引用是否已过期
  EXPECT_TRUE(obj1.expired());
  // 检查obj2的弱引用是否已过期
  EXPECT_TRUE(obj2.expired());
}

TEST(WeakIntrusivePtrTest, SwapMethodInvalidWithInvalid) {
  // 创建一个无效的弱引用对象，使用make_invalid_weak函数
  weak_intrusive_ptr<SomeClass> obj1 = make_invalid_weak<SomeClass>();
  // 创建另一个无效的弱引用对象，使用make_invalid_weak函数
  weak_intrusive_ptr<SomeClass> obj2 = make_invalid_weak<SomeClass>();
  // 交换obj1和obj2的值
  obj1.swap(obj2);
  // 检查obj1的弱引用是否已过期
  EXPECT_TRUE(obj1.expired());
  // 检查obj2的弱引用是否已过期
  EXPECT_TRUE(obj2.expired());
}

TEST(WeakIntrusivePtrTest, SwapFunctionFromWeakOnlyPtr) {
  // 创建一个仅含弱引用的对象，使用make_weak_only函数
  weak_intrusive_ptr<SomeClass> obj1 = make_weak_only<SomeClass>();
  // 创建一个带有强引用和弱引用的对象，使用make_weak_intrusive函数
  IntrusiveAndWeak<SomeClass> obj2 = make_weak_intrusive<SomeClass>();
  // 获取obj2的弱引用的原始指针
  SomeClass* obj2ptr = obj2.weak.lock().get();
  // 交换obj1和obj2的值
  swap(obj1, obj2.weak);
  // 检查obj2ptr和obj1的弱引用指向的对象是否相同
  EXPECT_EQ(obj2ptr, obj1.lock().get());
  // 检查obj1的弱引用是否未过期
  EXPECT_FALSE(obj1.expired());
  // 检查obj2的弱引用是否已过期
  EXPECT_TRUE(obj2.weak.expired());
}

TEST(WeakIntrusivePtrTest, SwapMethodFromWeakOnlyPtr) {
  // 创建一个仅含弱引用的对象，使用make_weak_only函数
  weak_intrusive_ptr<SomeClass> obj1 = make_weak_only<SomeClass>();
  // 创建一个带有强引用和弱引用的对象，使用make_weak_intrusive函数
  IntrusiveAndWeak<SomeClass> obj2 = make_weak_intrusive<SomeClass>();
  // 获取obj2的弱引用的原始指针
  SomeClass* obj2ptr = obj2.weak.lock().get();
  // 交换obj1和obj2的弱引用
  obj1.swap(obj2.weak);
  // 检查obj2ptr和obj1的弱引用指向的对象是否相同
  EXPECT_EQ(obj2ptr, obj1.lock().get());
  // 检查obj1的弱引用是否未过期
  EXPECT_FALSE(obj1.expired());
  // 检查obj2的弱引用是否已过期
  EXPECT_TRUE(obj2.weak.expired());
}

TEST(WeakIntrusivePtrTest, SwapFunctionWithWeakOnlyPtr) {
  // 创建一个带有强引用和弱引用的对象，使用make_weak_intrusive函数
  IntrusiveAndWeak<SomeClass> obj1 = make_weak_intrusive<SomeClass>();
  // 创建一个仅含弱引用的对象，使用make_weak_only函数
  weak_intrusive_ptr<SomeClass> obj2 = make_weak_only<SomeClass>();
  // 获取obj1的弱引用的原始指针
  SomeClass* obj1ptr = obj1.weak.lock().get();
  // 交换obj1的弱引用和obj2的值
  swap(obj1.weak, obj2);
  // 检查obj1的弱引用是否已过期
  EXPECT_TRUE(obj1.weak.expired());
  // 检查obj2的弱引用是否未过期
  EXPECT_FALSE(obj2.expired());
  // 检查obj1ptr和obj2的弱引用指向的对象是否相同
  EXPECT_EQ(obj1ptr, obj2.lock().get());
}

TEST(WeakIntrusivePtrTest, SwapMethodWithWeakOnlyPtr) {
  // 创建一个带有强引用和弱引用的对象，使用make_weak_intrusive函数
  IntrusiveAndWeak<SomeClass> obj1 = make_weak_intrusive<SomeClass>();
  // 创建一个仅含弱引用的对象，使用make_weak_only函数
  weak_intrusive_ptr<SomeClass> obj2 = make_weak_only<SomeClass>();
  // 获取obj1的弱引用的原始指针
  SomeClass* obj1ptr = obj1.weak.lock().get();
  // 交换obj1的弱引用和obj2的值
  obj1.weak.swap(obj2);
  // 检查obj1的弱引用是否已过期
  EXPECT_TRUE(obj1.weak.expired());
  // 检查obj2的弱引用是否未过期
  EXPECT_FALSE(obj2.expired());
  // 检查obj1ptr和obj2的弱引用指向的对象是否相同
  EXPECT_EQ(obj1ptr, obj2.lock().get());
}

TEST(WeakIntrusivePtrTest, SwapFunctionWeakOnlyPtrWithWeakOnlyPtr) {
  // 创建一个仅含弱引用的对象，使用make_weak_only函数
  weak_intrusive_ptr<SomeClass> obj1 = make_weak_only<SomeClass>();
  // 创建另一个仅含弱引用的对象，使用make_weak_only函数
  weak_intrusive_ptr<SomeClass> obj2 = make_weak_only<SomeClass>();
  // 交换obj1和obj2的值
  swap(obj1, obj2);
  // 检查obj1的弱引用是否已过期
  EXPECT_TRUE(obj1.expired());
  // 检查obj2的弱引用是否已过期
  EXPECT_TRUE(obj2.expired());
}
TEST(WeakIntrusivePtrTest, SwapMethodWeakOnlyPtrWithWeakOnlyPtr) {
  // 创建两个弱引用指针，指向新创建的 SomeClass 对象
  weak_intrusive_ptr<SomeClass> obj1 = make_weak_only<SomeClass>();
  weak_intrusive_ptr<SomeClass> obj2 = make_weak_only<SomeClass>();
  // 交换两个弱引用指针所指向的对象
  obj1.swap(obj2);
  // 检查交换后的两个指针是否已经过期
  EXPECT_TRUE(obj1.expired());
  EXPECT_TRUE(obj2.expired());
}

TEST(WeakIntrusivePtrTest, CanBePutInContainer) {
  // 创建一个 vector，存储 weak_intrusive_ptr 指向的对象
  std::vector<weak_intrusive_ptr<SomeClass1Parameter>> vec;
  // 创建一个 IntrusiveAndWeak 对象，其中包含一个弱引用指针
  IntrusiveAndWeak<SomeClass1Parameter> obj =
      make_weak_intrusive<SomeClass1Parameter>(5);
  // 将弱引用指针添加到 vector 中
  vec.push_back(obj.weak);
  // 验证 vector 中的对象参数是否为 5
  EXPECT_EQ(5, vec[0].lock()->param);
}

TEST(WeakIntrusivePtrTest, CanBePutInSet) {
  // 创建一个 set，存储 weak_intrusive_ptr 指向的对象
  std::set<weak_intrusive_ptr<SomeClass1Parameter>> set;
  // 创建一个 IntrusiveAndWeak 对象，其中包含一个弱引用指针
  IntrusiveAndWeak<SomeClass1Parameter> obj =
      make_weak_intrusive<SomeClass1Parameter>(5);
  // 将弱引用指针插入到 set 中
  set.insert(obj.weak);
  // 验证 set 中的第一个对象的参数是否为 5
  EXPECT_EQ(5, set.begin()->lock()->param);
}

TEST(WeakIntrusivePtrTest, CanBePutInUnorderedSet) {
  // 创建一个 unordered_set，存储 weak_intrusive_ptr 指向的对象
  std::unordered_set<weak_intrusive_ptr<SomeClass1Parameter>> set;
  // 创建一个 IntrusiveAndWeak 对象，其中包含一个弱引用指针
  IntrusiveAndWeak<SomeClass1Parameter> obj =
      make_weak_intrusive<SomeClass1Parameter>(5);
  // 将弱引用指针插入到 unordered_set 中
  set.insert(obj.weak);
  // 验证 unordered_set 中的第一个对象的参数是否为 5
  EXPECT_EQ(5, set.begin()->lock()->param);
}

TEST(WeakIntrusivePtrTest, CanBePutInMap) {
  // 创建一个 map，键和值都是 weak_intrusive_ptr 指向的对象
  std::map<
      weak_intrusive_ptr<SomeClass1Parameter>,
      weak_intrusive_ptr<SomeClass1Parameter>>
      map;
  // 创建两个 IntrusiveAndWeak 对象，分别包含弱引用指针，参数为 5 和 3
  IntrusiveAndWeak<SomeClass1Parameter> obj1 =
      make_weak_intrusive<SomeClass1Parameter>(5);
  IntrusiveAndWeak<SomeClass1Parameter> obj2 =
      make_weak_intrusive<SomeClass1Parameter>(3);
  // 将 obj1 的弱引用指针作为键，obj2 的弱引用指针作为值插入到 map 中
  map.insert(std::make_pair(obj1.weak, obj2.weak));
  // 验证 map 中第一对键值的参数分别为 5 和 3
  EXPECT_EQ(5, map.begin()->first.lock()->param);
  EXPECT_EQ(3, map.begin()->second.lock()->param);
}

TEST(WeakIntrusivePtrTest, CanBePutInUnorderedMap) {
  // 创建一个 unordered_map，键和值都是 weak_intrusive_ptr 指向的对象
  std::unordered_map<
      weak_intrusive_ptr<SomeClass1Parameter>,
      weak_intrusive_ptr<SomeClass1Parameter>>
      map;
  // 创建两个 IntrusiveAndWeak 对象，分别包含弱引用指针，参数为 5 和 3
  IntrusiveAndWeak<SomeClass1Parameter> obj1 =
      make_weak_intrusive<SomeClass1Parameter>(5);
  IntrusiveAndWeak<SomeClass1Parameter> obj2 =
      make_weak_intrusive<SomeClass1Parameter>(3);
  // 将 obj1 的弱引用指针作为键，obj2 的弱引用指针作为值插入到 unordered_map 中
  map.insert(std::make_pair(obj1.weak, obj2.weak));
  // 验证 unordered_map 中第一对键值的参数分别为 5 和 3
  EXPECT_EQ(5, map.begin()->first.lock()->param);
  EXPECT_EQ(3, map.begin()->second.lock()->param);
}

TEST(WeakIntrusivePtrTest, Equality_AfterCopyConstructor) {
  // 创建一个 IntrusiveAndWeak 对象，并使用 make_weak_intrusive 初始化它
  IntrusiveAndWeak<SomeClass> var1 = make_weak_intrusive<SomeClass>();
  // 从 var1 中获取其弱引用指针
  weak_intrusive_ptr<SomeClass> var2 = var1.weak;
  // 验证 var1 和 var2 的弱引用指针是否相等
  EXPECT_TRUE(var1.weak == var2);
  EXPECT_FALSE(var1.weak != var2);
}

TEST(WeakIntrusivePtrTest, Equality_AfterCopyAssignment) {
  // 创建两个 IntrusiveAndWeak 对象，并使用 make_weak_intrusive 初始化它们
  IntrusiveAndWeak<SomeClass> var1 = make_weak_intrusive<SomeClass>();
  IntrusiveAndWeak<SomeClass> var2 = make_weak_intrusive<SomeClass>();
  // 将 var1 的弱引用指针赋值给 var2
  var2.weak = var1.weak;
  // 验证 var1 和 var2 的弱引用指针是否相等
  EXPECT_TRUE(var1.weak == var2.weak);
  EXPECT_FALSE(var1.weak != var2.weak);
}
TEST(WeakIntrusivePtrTest, Equality_AfterCopyAssignment_WeakOnly) {
  // 创建一个弱引用指针 var1，指向一个 SomeClass 类的对象，通过 make_weak_only 函数创建
  weak_intrusive_ptr<SomeClass> var1 = make_weak_only<SomeClass>();
  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
  // 创建一个弱引用指针 var2，并将其复制为 var1
  weak_intrusive_ptr<SomeClass> var2 = var1;
  // 断言 var1 和 var2 指向的对象相等
  EXPECT_TRUE(var1 == var2);
  // 断言 var1 和 var2 指向的对象不不相等
  EXPECT_FALSE(var1 != var2);
}

TEST(WeakIntrusivePtrTest, Equality_Invalid) {
  // 创建一个无效的弱引用指针 var1，指向一个 SomeClass 类的对象，通过 make_invalid_weak 函数创建
  weak_intrusive_ptr<SomeClass> var1 = make_invalid_weak<SomeClass>();
  // 创建一个无效的弱引用指针 var2，指向一个 SomeClass 类的对象，通过 make_invalid_weak 函数创建
  weak_intrusive_ptr<SomeClass> var2 = make_invalid_weak<SomeClass>();
  // 断言 var1 和 var2 指向的对象相等
  EXPECT_TRUE(var1 == var2);
  // 断言 var1 和 var2 指向的对象不不相等
  EXPECT_FALSE(var1 != var2);
}

TEST(WeakIntrusivePtrTest, Inequality) {
  // 创建一个强引用和弱引用的结构体 var1，包含一个指向 SomeClass 类对象的弱引用，通过 make_intrusive 函数创建
  IntrusiveAndWeak<SomeClass> var1 = make_intrusive<SomeClass>();
  // 创建一个强引用和弱引用的结构体 var2，包含一个指向 SomeClass 类对象的弱引用，通过 make_intrusive 函数创建
  IntrusiveAndWeak<SomeClass> var2 = make_intrusive<SomeClass>();
  // 断言 var1 的弱引用和 var2 的弱引用指向的对象不相等
  EXPECT_TRUE(var1.weak != var2.weak);
  // 断言 var1 的弱引用和 var2 的弱引用指向的对象相等
  EXPECT_FALSE(var1.weak == var2.weak);
}

TEST(WeakIntrusivePtrTest, Inequality_InvalidLeft) {
  // 创建一个无效的弱引用指针 var1，指向一个 SomeClass 类的对象，通过 make_invalid_weak 函数创建
  weak_intrusive_ptr<SomeClass> var1 = make_invalid_weak<SomeClass>();
  // 创建一个强引用和弱引用的结构体 var2，包含一个指向 SomeClass 类对象的强引用和弱引用，通过 make_intrusive 函数创建
  IntrusiveAndWeak<SomeClass> var2 = make_intrusive<SomeClass>();
  // 断言 var1 和 var2 的弱引用指向的对象不相等
  EXPECT_TRUE(var1 != var2.weak);
  // 断言 var1 和 var2 的弱引用指向的对象相等
  EXPECT_FALSE(var1 == var2.weak);
}

TEST(WeakIntrusivePtrTest, Inequality_InvalidRight) {
  // 创建一个强引用和弱引用的结构体 var1，包含一个指向 SomeClass 类对象的强引用和弱引用，通过 make_intrusive 函数创建
  IntrusiveAndWeak<SomeClass> var1 = make_intrusive<SomeClass>();
  // 创建一个无效的弱引用指针 var2，指向一个 SomeClass 类的对象，通过 make_invalid_weak 函数创建
  weak_intrusive_ptr<SomeClass> var2 = make_invalid_weak<SomeClass>();
  // 断言 var1 的弱引用和 var2 的弱引用指向的对象不相等
  EXPECT_TRUE(var1.weak != var2);
  // 断言 var1 的弱引用和 var2 的弱引用指向的对象相等
  EXPECT_FALSE(var1.weak == var2);
}

TEST(WeakIntrusivePtrTest, Inequality_WeakOnly) {
  // 创建一个只有弱引用的指针 var1，指向一个 SomeClass 类的对象，通过 make_weak_only 函数创建
  weak_intrusive_ptr<SomeClass> var1 = make_weak_only<SomeClass>();
  // 创建一个只有弱引用的指针 var2，指向一个 SomeClass 类的对象，通过 make_weak_only 函数创建
  weak_intrusive_ptr<SomeClass> var2 = make_weak_only<SomeClass>();
  // 断言 var1 和 var2 指向的对象不相等
  EXPECT_TRUE(var1 != var2);
  // 断言 var1 和 var2 指向的对象相等
  EXPECT_FALSE(var1 == var2);
}

TEST(WeakIntrusivePtrTest, HashIsDifferent) {
  // 创建一个强引用和弱引用的结构体 var1，包含一个指向 SomeClass 类对象的弱引用，通过 make_weak_intrusive 函数创建
  IntrusiveAndWeak<SomeClass> var1 = make_weak_intrusive<SomeClass>();
  // 创建一个强引用和弱引用的结构体 var2，包含一个指向 SomeClass 类对象的弱引用，通过 make_weak_intrusive 函数创建
  IntrusiveAndWeak<SomeClass> var2 = make_weak_intrusive<SomeClass>();
  // 断言 var1 的弱引用和 var2 的弱引用的哈希值不相等
  EXPECT_NE(
      std::hash<weak_intrusive_ptr<SomeClass>>()(var1.weak),
      std::hash<weak_intrusive_ptr<SomeClass>>()(var2.weak));
}

TEST(WeakIntrusivePtrTest, HashIsDifferent_ValidAndInvalid) {
  // 创建一个无效的弱引用指针 var1，指向一个 SomeClass 类的对象，通过 make_invalid_weak 函数创建
  weak_intrusive_ptr<SomeClass> var1 = make_invalid_weak<SomeClass>();
  // 创建一个强引用和弱引用的结构体 var2，包含一个指向 SomeClass 类对象的弱引用，通过 make_weak_intrusive 函数创建
  IntrusiveAndWeak<SomeClass> var2 = make_weak_intrusive<SomeClass>();
  // 断言 var1 的哈希值和 var2 的弱引用的哈希值不相等
  EXPECT_NE(
      std::hash<weak_intrusive_ptr<SomeClass>>()(var1),
      std::hash<weak_intrusive_ptr<SomeClass>>()(var2.weak));
}

TEST(WeakIntrusivePtrTest, HashIsDifferent_ValidAndWeakOnly) {
  // 创建一个只有弱引用的指针 var1，指向一个 SomeClass 类的对象，通过 make_weak_only 函数创建
  weak_intrusive_ptr<SomeClass> var1 = make_weak_only<SomeClass>();
  // 创建一个强引用和弱引用的结构体 var2，包含一个指向 SomeClass 类对象的弱引用，通过 make_weak_intrusive 函数创建
  IntrusiveAndWeak<SomeClass> var2 = make_weak_intrusive<SomeClass>();
  // 断言 var1 的哈希值和 var2 的弱引用的哈希值不相等
  EXPECT_NE(
      std::hash<weak_intrusive_ptr<SomeClass>>()(var1),
      std::hash<weak_intrusive_ptr<SomeClass>>()(var2.weak));
}

TEST(WeakIntrusivePtrTest, HashIsDifferent_WeakOnlyAndWeakOnly) {
  // 创建一个只有弱引用的指针 var1，指向一个 SomeClass 类的对象，通过 make_weak_only 函数创建
  weak_intrusive_ptr<SomeClass> var1 = make_weak_only<SomeClass>();
  // 创建一个只有弱引用的指针 var2，指向一个 SomeClass 类的对象，通过 make_weak_only 函数创建
  weak_intrusive_ptr<SomeClass> var2 = make_weak_only<SomeClass>();
  // 断言 var1 的哈希值和 var2 的哈希值不相等
  EXPECT_NE(
      std::hash<weak_intrusive_ptr<SomeClass>>()(var1),
      std::hash<weak_intrusive_ptr<SomeClass>>()(var2));
}
TEST(WeakIntrusivePtrTest, HashIsSame_AfterCopyConstructor) {
  // 创建一个带有弱引用的 SomeClass 实例，并将其赋给 var1
  IntrusiveAndWeak<SomeClass> var1 = make_weak_intrusive<SomeClass>();
  // 从 var1 中获取其弱引用，赋给 var2
  weak_intrusive_ptr<SomeClass> var2 = var1.weak;
  // 断言：使用 std::hash 计算 var1.weak 和 var2 的哈希值，期望它们相等
  EXPECT_EQ(
      std::hash<weak_intrusive_ptr<SomeClass>>()(var1.weak),
      std::hash<weak_intrusive_ptr<SomeClass>>()(var2));
}

TEST(WeakIntrusivePtrTest, HashIsSame_AfterCopyConstructor_WeakOnly) {
  // 创建一个仅包含弱引用的 SomeClass 实例，并将其赋给 var1
  weak_intrusive_ptr<SomeClass> var1 = make_weak_only<SomeClass>();
  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
  // 将 var1 赋给 var2
  weak_intrusive_ptr<SomeClass> var2 = var1;
  // 断言：使用 std::hash 计算 var1 和 var2 的哈希值，期望它们相等
  EXPECT_EQ(
      std::hash<weak_intrusive_ptr<SomeClass>>()(var1),
      std::hash<weak_intrusive_ptr<SomeClass>>()(var2));
}

TEST(WeakIntrusivePtrTest, HashIsSame_AfterCopyAssignment) {
  // 创建两个带有弱引用的 SomeClass 实例，并分别赋给 var1 和 var2
  IntrusiveAndWeak<SomeClass> var1 = make_weak_intrusive<SomeClass>();
  IntrusiveAndWeak<SomeClass> var2 = make_weak_intrusive<SomeClass>();
  // 将 var1 的弱引用赋给 var2 的弱引用
  var2.weak = var1.weak;
  // 断言：使用 std::hash 计算 var1.weak 和 var2.weak 的哈希值，期望它们相等
  EXPECT_EQ(
      std::hash<weak_intrusive_ptr<SomeClass>>()(var1.weak),
      std::hash<weak_intrusive_ptr<SomeClass>>()(var2.weak));
}

TEST(WeakIntrusivePtrTest, HashIsSame_AfterCopyAssignment_WeakOnly) {
  // 创建一个仅包含弱引用的 SomeClass 实例，并将其赋给 var1
  weak_intrusive_ptr<SomeClass> var1 = make_weak_only<SomeClass>();
  // 创建一个无效的弱引用，并将其赋给 var2
  weak_intrusive_ptr<SomeClass> var2 = make_invalid_weak<SomeClass>();
  // 将 var1 赋给 var2
  var2 = var1;
  // 断言：使用 std::hash 计算 var1 和 var2 的哈希值，期望它们相等
  EXPECT_EQ(
      std::hash<weak_intrusive_ptr<SomeClass>>()(var1),
      std::hash<weak_intrusive_ptr<SomeClass>>()(var2));
}

TEST(WeakIntrusivePtrTest, HashIsSame_BothInvalid) {
  // 创建两个无效的弱引用实例，并分别赋给 var1 和 var2
  weak_intrusive_ptr<SomeClass> var1 = make_invalid_weak<SomeClass>();
  weak_intrusive_ptr<SomeClass> var2 = make_invalid_weak<SomeClass>();
  // 断言：使用 std::hash 计算 var1 和 var2 的哈希值，期望它们相等
  EXPECT_EQ(
      std::hash<weak_intrusive_ptr<SomeClass>>()(var1),
      std::hash<weak_intrusive_ptr<SomeClass>>()(var2));
}

TEST(WeakIntrusivePtrTest, OneIsLess) {
  // 创建两个带有弱引用的 SomeClass 实例，并分别赋给 var1 和 var2
  IntrusiveAndWeak<SomeClass> var1 = make_weak_intrusive<SomeClass>();
  IntrusiveAndWeak<SomeClass> var2 = make_weak_intrusive<SomeClass>();
  // 断言：使用 std::less 进行比较 var1.weak 和 var2.weak，期望它们一方小于另一方
  EXPECT_TRUE(
      // NOLINTNEXTLINE(modernize-use-transparent-functors)
      std::less<weak_intrusive_ptr<SomeClass>>()(var1.weak, var2.weak) !=
      // NOLINTNEXTLINE(modernize-use-transparent-functors)
      std::less<weak_intrusive_ptr<SomeClass>>()(var2.weak, var1.weak));
}

TEST(WeakIntrusivePtrTest, InvalidIsLess1) {
  // 创建一个无效的弱引用实例，并将其赋给 var1
  weak_intrusive_ptr<SomeClass> var1 = make_invalid_weak<SomeClass>();
  // 创建一个带有弱引用的 SomeClass 实例，并将其赋给 var2
  IntrusiveAndWeak<SomeClass> var2 = make_weak_intrusive<SomeClass>();
  // 断言：使用 std::less 进行比较 var1 和 var2.weak，期望 var1 小于 var2.weak
  // NOLINTNEXTLINE(modernize-use-transparent-functors)
  EXPECT_TRUE(std::less<weak_intrusive_ptr<SomeClass>>()(var1, var2.weak));
}

TEST(WeakIntrusivePtrTest, InvalidIsLess2) {
  // 创建一个带有弱引用的 SomeClass 实例，并将其赋给 var1
  IntrusiveAndWeak<SomeClass> var1 = make_weak_intrusive<SomeClass>();
  // 创建一个无效的弱引用实例，并将其赋给 var2
  weak_intrusive_ptr<SomeClass> var2 = make_invalid_weak<SomeClass>();
  // 断言：使用 std::less 进行比较 var1.weak 和 var2，期望 var1.weak 不小于 var2
  // NOLINTNEXTLINE(modernize-use-transparent-functors)
  EXPECT_FALSE(std::less<weak_intrusive_ptr<SomeClass>>()(var1.weak, var2));
}
TEST(WeakIntrusivePtrTest, InvalidIsNotLessThanInvalid) {
  // 创建两个无效的弱指针 var1 和 var2
  weak_intrusive_ptr<SomeClass> var1 = make_invalid_weak<SomeClass>();
  weak_intrusive_ptr<SomeClass> var2 = make_invalid_weak<SomeClass>();
  // 使用 std::less 运算符验证 var1 是否不小于 var2
  // NOLINTNEXTLINE(modernize-use-transparent-functors)
  EXPECT_FALSE(std::less<weak_intrusive_ptr<SomeClass>>()(var1, var2));
}

TEST(WeakIntrusivePtrTest, givenPtr_whenCallingResetOnWeakPtr_thenIsInvalid) {
  // 创建一个包含强指针和弱指针的对象 obj
  IntrusiveAndWeak<SomeClass> obj = make_weak_intrusive<SomeClass>();
  // 验证弱指针 obj.weak 没有过期
  EXPECT_FALSE(obj.weak.expired());
  // 重置弱指针 obj.weak
  obj.weak.reset();
  // 验证弱指针 obj.weak 已过期
  EXPECT_TRUE(obj.weak.expired());
}

TEST(WeakIntrusivePtrTest, givenPtr_whenCallingResetOnStrongPtr_thenIsInvalid) {
  // 创建一个包含强指针和弱指针的对象 obj
  IntrusiveAndWeak<SomeClass> obj = make_weak_intrusive<SomeClass>();
  // 验证弱指针 obj.weak 没有过期
  EXPECT_FALSE(obj.weak.expired());
  // 重置强指针 obj.ptr
  obj.ptr.reset();
  // 验证弱指针 obj.weak 已过期
  EXPECT_TRUE(obj.weak.expired());
}

TEST(WeakIntrusivePtrTest, AllowsMoveConstructingToConst) {
  // 创建一个包含强指针和弱指针的对象 a
  IntrusiveAndWeak<SomeClass> a = make_weak_intrusive<SomeClass>();
  // 将 a.weak 移动构造到常量弱指针 b
  weak_intrusive_ptr<const SomeClass> b = std::move(a.weak);
}

TEST(WeakIntrusivePtrTest, AllowsCopyConstructingToConst) {
  // 创建一个包含强指针和弱指针的对象 a
  IntrusiveAndWeak<SomeClass> a = make_weak_intrusive<SomeClass>();
  // 将 a.weak 拷贝构造到常量弱指针 b
  weak_intrusive_ptr<const SomeClass> b = a.weak;
}

TEST(WeakIntrusivePtrTest, AllowsMoveAssigningToConst) {
  // 创建一个包含强指针和弱指针的对象 a
  IntrusiveAndWeak<SomeClass> a = make_weak_intrusive<SomeClass>();
  // 创建一个包含常量强指针和弱指针的对象 b
  IntrusiveAndWeak<const SomeClass> b = make_weak_intrusive<const SomeClass>();
  // 将 a.weak 移动赋值到 b.weak
  b.weak = std::move(a.weak);
}

TEST(WeakIntrusivePtrTest, AllowsCopyAssigningToConst) {
  // 创建一个包含强指针和弱指针的对象 a
  IntrusiveAndWeak<SomeClass> a = make_weak_intrusive<SomeClass>();
  // 创建一个包含常量强指针和弱指针的对象 b
  IntrusiveAndWeak<const SomeClass> b = make_weak_intrusive<const SomeClass>();
  // 将 a.weak 拷贝赋值到 b.weak
  b.weak = a.weak;
}

TEST(WeakIntrusivePtrTest, givenNewPtr_thenHasUseCount1) {
  // 创建一个包含强指针和弱指针的对象 obj
  IntrusiveAndWeak<SomeClass> obj = make_weak_intrusive<SomeClass>();
  // 验证弱指针 obj.weak 的引用计数为 1
  EXPECT_EQ(1, obj.weak.use_count());
}

TEST(WeakIntrusivePtrTest, givenNewPtr_thenIsNotExpired) {
  // 创建一个包含强指针和弱指针的对象 obj
  IntrusiveAndWeak<SomeClass> obj = make_weak_intrusive<SomeClass>();
  // 验证弱指针 obj.weak 没有过期
  EXPECT_FALSE(obj.weak.expired());
}

TEST(WeakIntrusivePtrTest, givenInvalidPtr_thenHasUseCount0) {
  // 创建一个无效的弱指针 obj
  weak_intrusive_ptr<SomeClass> obj = make_invalid_weak<SomeClass>();
  // 验证无效弱指针 obj 的引用计数为 0
  EXPECT_EQ(0, obj.use_count());
}

TEST(WeakIntrusivePtrTest, givenInvalidPtr_thenIsExpired) {
  // 创建一个无效的弱指针 obj
  weak_intrusive_ptr<SomeClass> obj = make_invalid_weak<SomeClass>();
  // 验证无效弱指针 obj 已过期
  EXPECT_TRUE(obj.expired());
}

TEST(WeakIntrusivePtrTest, givenWeakOnlyPtr_thenHasUseCount0) {
  // 创建一个仅包含弱指针的对象 obj
  weak_intrusive_ptr<SomeClass> obj = make_weak_only<SomeClass>();
  // 验证仅包含弱指针 obj 的引用计数为 0
  EXPECT_EQ(0, obj.use_count());
}

TEST(WeakIntrusivePtrTest, givenWeakOnlyPtr_thenIsExpired) {
  // 创建一个仅包含弱指针的对象 obj
  weak_intrusive_ptr<SomeClass> obj = make_weak_only<SomeClass>();
  // 验证仅包含弱指针 obj 已过期
  EXPECT_TRUE(obj.expired());
}

TEST(WeakIntrusivePtrTest, givenPtr_whenCallingWeakReset_thenHasUseCount0) {
  // 创建一个包含强指针和弱指针的对象 obj
  IntrusiveAndWeak<SomeClass> obj = make_weak_intrusive<SomeClass>();
  // 调用弱指针 obj.weak 的重置方法
  obj.weak.reset();
  // 验证重置后弱指针 obj.weak 的引用计数为 0
  EXPECT_EQ(0, obj.weak.use_count());
}
# 测试用例：给定弱指针，调用 weak_reset 后应该已经过期
TEST(WeakIntrusivePtrTest, givenPtr_whenCallingWeakReset_thenIsExpired) {
  # 创建一个弱指针和强指针对象，使用 make_weak_intrusive 创建
  IntrusiveAndWeak<SomeClass> obj = make_weak_intrusive<SomeClass>();
  # 调用弱指针的 reset 方法
  obj.weak.reset();
  # 验证弱指针是否已经过期
  EXPECT_TRUE(obj.weak.expired());
}

# 测试用例：给定强指针，调用 strong_reset 后使用计数为 0
TEST(WeakIntrusivePtrTest, givenPtr_whenCallingStrongReset_thenHasUseCount0) {
  # 创建一个弱指针和强指针对象，使用 make_weak_intrusive 创建
  IntrusiveAndWeak<SomeClass> obj = make_weak_intrusive<SomeClass>();
  # 调用强指针的 reset 方法
  obj.ptr.reset();
  # 验证弱指针的使用计数是否为 0
  EXPECT_EQ(0, obj.weak.use_count());
}

# 测试用例：给定强指针，调用 strong_reset 后应该已经过期
TEST(WeakIntrusivePtrTest, givenPtr_whenCallingStrongReset_thenIsExpired) {
  # 创建一个弱指针和强指针对象，使用 make_weak_intrusive 创建
  IntrusiveAndWeak<SomeClass> obj = make_weak_intrusive<SomeClass>();
  # 调用强指针的 reset 方法
  obj.ptr.reset();
  # 验证强指针是否已经过期
  EXPECT_TRUE(obj.weak.expired());
}

# 测试用例：给定移动构造的指针，使用计数应为 1
TEST(WeakIntrusivePtrTest, givenMoveConstructedPtr_thenHasUseCount1) {
  # 创建一个弱指针和强指针对象，使用 make_weak_intrusive 创建
  IntrusiveAndWeak<SomeClass> obj = make_weak_intrusive<SomeClass>();
  # 将原对象的弱指针对象移动到新对象
  weak_intrusive_ptr<SomeClass> obj2 = std::move(obj.weak);
  # 验证新对象的弱指针使用计数是否为 1
  EXPECT_EQ(1, obj2.use_count());
}

# 测试用例：给定移动构造的指针，应该没有过期
TEST(WeakIntrusivePtrTest, givenMoveConstructedPtr_thenIsNotExpired) {
  # 创建一个弱指针和强指针对象，使用 make_weak_intrusive 创建
  IntrusiveAndWeak<SomeClass> obj = make_weak_intrusive<SomeClass>();
  # 将原对象的弱指针对象移动到新对象
  weak_intrusive_ptr<SomeClass> obj2 = std::move(obj.weak);
  # 验证新对象的弱指针是否没有过期
  EXPECT_FALSE(obj2.expired());
}

# 测试用例：给定移动构造的指针，原对象的使用计数应为 0
TEST(WeakIntrusivePtrTest, givenMoveConstructedPtr_thenOldHasUseCount0) {
  # 创建一个弱指针和强指针对象，使用 make_weak_intrusive 创建
  IntrusiveAndWeak<SomeClass> obj = make_weak_intrusive<SomeClass>();
  # 将原对象的弱指针对象移动到新对象
  weak_intrusive_ptr<SomeClass> obj2 = std::move(obj.weak);
  # 验证原对象的弱指针使用计数是否为 0
  EXPECT_EQ(0, obj.weak.use_count());
}

# 测试用例：给定移动构造的指针，原对象的弱指针应该已经过期
TEST(WeakIntrusivePtrTest, givenMoveConstructedPtr_thenOldIsExpired) {
  # 创建一个弱指针和强指针对象，使用 make_weak_intrusive 创建
  IntrusiveAndWeak<SomeClass> obj = make_weak_intrusive<SomeClass>();
  # 将原对象的弱指针对象移动到新对象
  weak_intrusive_ptr<SomeClass> obj2 = std::move(obj.weak);
  # 验证原对象的弱指针是否已经过期
  EXPECT_TRUE(obj.weak.expired());
}

# 测试用例：给定移动赋值的指针，使用计数应为 1
TEST(WeakIntrusivePtrTest, givenMoveAssignedPtr_thenHasUseCount1) {
  # 创建两个不同的弱指针和强指针对象，使用 make_weak_intrusive 创建
  IntrusiveAndWeak<SomeClass> obj = make_weak_intrusive<SomeClass>();
  IntrusiveAndWeak<SomeClass> obj2 = make_weak_intrusive<SomeClass>();
  # 将原对象的弱指针对象移动赋值到新对象的弱指针中
  obj2.weak = std::move(obj.weak);
  # 验证新对象的弱指针使用计数是否为 1
  EXPECT_EQ(1, obj2.weak.use_count());
}

# 测试用例：给定移动赋值的指针，应该没有过期
TEST(WeakIntrusivePtrTest, givenMoveAssignedPtr_thenIsNotExpired) {
  # 创建两个不同的弱指针和强指针对象，使用 make_weak_intrusive 创建
  IntrusiveAndWeak<SomeClass> obj = make_weak_intrusive<SomeClass>();
  IntrusiveAndWeak<SomeClass> obj2 = make_weak_intrusive<SomeClass>();
  # 将原对象的弱指针对象移动赋值到新对象的弱指针中
  obj2.weak = std::move(obj.weak);
  # 验证新对象的弱指针是否没有过期
  EXPECT_FALSE(obj2.weak.expired());
}

# 测试用例：给定移动赋值的指针，原对象的使用计数应为 0
TEST(WeakIntrusivePtrTest, givenMoveAssignedPtr_thenOldHasUseCount0) {
  # 创建两个不同的弱指针和强指针对象，使用 make_weak_intrusive 创建
  IntrusiveAndWeak<SomeClass> obj = make_weak_intrusive<SomeClass>();
  IntrusiveAndWeak<SomeClass> obj2 = make_weak_intrusive<SomeClass>();
  # 将原对象的弱指针对象移动赋值到新对象的弱指针中
  obj2.weak = std::move(obj.weak);
  # 验证原对象的弱指针使用计数是否为 0
  EXPECT_EQ(0, obj.weak.use_count());
}

# 测试用例：给定移动赋值的指针，原对象的弱指针应该已经过期
TEST(WeakIntrusivePtrTest, givenMoveAssignedPtr_thenOldIsExpired) {
  # 创建两个不同的弱指针和强指针对象，使用 make_weak_intrusive 创建
  IntrusiveAndWeak<SomeClass> obj = make_weak_intrusive<SomeClass>();
  IntrusiveAndWeak<SomeClass> obj2 = make_weak_intrusive<SomeClass>();
  # 将原对象的弱指针对象移动赋值到新对象的弱指针中
  obj2.weak = std::move(obj.weak);
  # 验证原对象的弱指针是否已经过期
  EXPECT_TRUE(obj.weak.expired());
}

# 测试用例：给定复制构造的指针，使用计数应为 1
TEST(WeakIntrusivePtrTest, givenCopyConstructedPtr_thenHasUseCount1) {
  # 创建一个弱指针和强指针对象，使用 make_weak_intrusive 创建
  IntrusiveAndWeak<SomeClass> obj = make_weak_intrusive<SomeClass>();
  # 使用复制构造函数将原对象的弱指针对象复制到新对象的弱指针中
  weak_intrusive_ptr<SomeClass> obj2 = obj.weak;
  # 验证新对象的弱指针使用计数是否为 1
  EXPECT_EQ(1, obj2.use_count());
}
TEST(WeakIntrusivePtrTest, givenCopyConstructedPtr_thenIsNotExpired) {
  // 创建一个强指针和弱指针的包装对象，并调用工厂函数生成弱指针
  IntrusiveAndWeak<SomeClass> obj = make_weak_intrusive<SomeClass>();
  // 从包装对象中获取弱指针
  weak_intrusive_ptr<SomeClass> obj2 = obj.weak;
  // 断言弱指针未过期
  EXPECT_FALSE(obj2.expired());
}

TEST(WeakIntrusivePtrTest, givenCopyConstructedPtr_thenOldHasUseCount1) {
  // 创建一个强指针和弱指针的包装对象，并调用工厂函数生成弱指针
  IntrusiveAndWeak<SomeClass> obj = make_weak_intrusive<SomeClass>();
  // 从包装对象中获取弱指针
  weak_intrusive_ptr<SomeClass> obj2 = obj.weak;
  // 断言弱指针的引用计数为1
  EXPECT_EQ(1, obj.weak.use_count());
}

TEST(WeakIntrusivePtrTest, givenCopyConstructedPtr_thenOldIsNotExpired) {
  // 创建一个强指针和弱指针的包装对象，并调用工厂函数生成弱指针
  IntrusiveAndWeak<SomeClass> obj = make_weak_intrusive<SomeClass>();
  // 从包装对象中获取弱指针
  weak_intrusive_ptr<SomeClass> obj2 = obj.weak;
  // 断言弱指针未过期
  EXPECT_FALSE(obj.weak.expired());
}

TEST(
    WeakIntrusivePtrTest,
    givenPtr_whenLastStrongPointerResets_thenReleasesResources) {
  // 初始化资源释放和析构标志
  bool resourcesReleased = false;
  bool wasDestructed = false;
  // 创建具有弱指针的可析构模拟对象，并传入资源释放和析构标志的地址
  auto obj =
      make_weak_intrusive<DestructableMock>(&resourcesReleased, &wasDestructed);
  // 断言资源未释放和对象未析构
  EXPECT_FALSE(resourcesReleased);
  EXPECT_FALSE(wasDestructed);
  // 重置强指针，预期释放资源，但不析构对象
  obj.ptr.reset();
  EXPECT_TRUE(resourcesReleased);
  EXPECT_FALSE(wasDestructed);
  // 重置弱指针，预期释放资源和析构对象
  obj.weak.reset();
  EXPECT_TRUE(resourcesReleased);
  EXPECT_TRUE(wasDestructed);
}

TEST(
    WeakIntrusivePtrTest,
    givenPtr_whenDestructedButStillHasStrongPointers_thenDoesntReleaseResources) {
  // 初始化资源释放和析构标志
  bool resourcesReleased = false;
  bool wasDestructed = false;
  // 创建具有弱指针的可析构模拟对象，并传入资源释放和析构标志的地址
  auto obj =
      make_weak_intrusive<DestructableMock>(&resourcesReleased, &wasDestructed);
  // 断言资源未释放和对象未析构
  EXPECT_FALSE(resourcesReleased);
  EXPECT_FALSE(wasDestructed);
  // 重置弱指针，预期不释放资源和不析构对象
  obj.weak.reset();
  EXPECT_FALSE(resourcesReleased);
  EXPECT_FALSE(wasDestructed);
  // 重置强指针，预期释放资源和析构对象
  obj.ptr.reset();
  EXPECT_TRUE(resourcesReleased);
  EXPECT_TRUE(wasDestructed);
}

TEST(WeakIntrusivePtrTest, givenPtr_whenDestructed_thenDestructsObject) {
  // 初始化资源释放和析构标志
  bool resourcesReleased = false;
  bool wasDestructed = false;
  {
    // 创建只有弱指针的可析构模拟对象，并传入资源释放和析构标志的地址
    auto obj =
        make_weak_only<DestructableMock>(&resourcesReleased, &wasDestructed);
    // 断言资源已释放和对象已析构
    EXPECT_TRUE(resourcesReleased);
    EXPECT_FALSE(wasDestructed);
  }
  // 断言资源已释放和对象已析构
  EXPECT_TRUE(resourcesReleased);
  EXPECT_TRUE(wasDestructed);
}

TEST(
    WeakIntrusivePtrTest,
    givenPtr_whenMoveConstructed_thenDestructsObjectAfterSecondDestructed) {
  // 初始化资源释放和析构标志
  bool resourcesReleased = false;
  bool wasDestructed = false;
  // 创建只有弱指针的可析构模拟对象，并传入资源释放和析构标志的地址
  auto obj =
      make_weak_only<DestructableMock>(&resourcesReleased, &wasDestructed);
  {
    // 将对象移动到另一个对象中
    auto obj2 = std::move(obj);
    // 断言资源已释放和对象未析构
    EXPECT_TRUE(resourcesReleased);
    EXPECT_FALSE(wasDestructed);
  }
  // 断言资源已释放和对象已析构
  EXPECT_TRUE(resourcesReleased);
  EXPECT_TRUE(wasDestructed);
}

TEST(
    WeakIntrusivePtrTest,
    givenPtr_whenMoveConstructedToBaseClass_thenDestructsObjectAfterSecondDestructed) {
  // 初始化资源释放和析构标志
  bool resourcesReleased = false;
  bool wasDestructed = false;
  // 创建只有弱指针的子类可析构模拟对象，并传入资源释放和析构标志的地址
  auto obj =
      make_weak_only<ChildDestructableMock>(&resourcesReleased, &wasDestructed);
  {
    // 将对象移动到基类弱指针中
    weak_intrusive_ptr<DestructableMock> obj2 = std::move(obj);
    // 断言资源已释放和对象未析构
    EXPECT_TRUE(resourcesReleased);
    EXPECT_FALSE(wasDestructed);
  }
  // 断言资源已释放和对象已析构
  EXPECT_TRUE(resourcesReleased);
  EXPECT_TRUE(wasDestructed);
}
TEST(WeakIntrusivePtrTest, givenPtr_whenMoveAssigned_thenDestructsOldObject) {
  bool dummy = false;  // 创建一个布尔变量dummy，并初始化为false
  bool resourcesReleased = false;  // 创建一个布尔变量resourcesReleased，并初始化为false
  bool wasDestructed = false;  // 创建一个布尔变量wasDestructed，并初始化为false
  auto obj = make_weak_only<DestructableMock>(&dummy, &dummy);  // 创建一个弱引用指针obj，指向一个DestructableMock对象，该对象使用dummy变量的地址作为参数
  {
    auto obj2 =  // 创建一个新的弱引用指针obj2
        make_weak_only<DestructableMock>(&resourcesReleased, &wasDestructed);  // 以resourcesReleased和wasDestructed的地址创建一个新的DestructableMock对象，并将其指针赋给obj2
    EXPECT_TRUE(resourcesReleased);  // 断言resourcesReleased为true
    EXPECT_FALSE(wasDestructed);  // 断言wasDestructed为false
    obj2 = std::move(obj);  // 将obj移动给obj2，此操作将销毁旧的obj2指向的对象
    EXPECT_TRUE(resourcesReleased);  // 断言resourcesReleased为true
    EXPECT_TRUE(wasDestructed);  // 断言wasDestructed为true，表示旧的obj2指向的对象已被销毁
  }
}

TEST(
    WeakIntrusivePtrTest,
    givenPtr_whenMoveAssignedToBaseClass_thenDestructsOldObject) {
  bool dummy = false;  // 创建一个布尔变量dummy，并初始化为false
  bool resourcesReleased = false;  // 创建一个布尔变量resourcesReleased，并初始化为false
  bool wasDestructed = false;  // 创建一个布尔变量wasDestructed，并初始化为false
  auto obj = make_weak_only<ChildDestructableMock>(&dummy, &dummy);  // 创建一个弱引用指针obj，指向一个ChildDestructableMock对象，该对象使用dummy变量的地址作为参数
  {
    auto obj2 =  // 创建一个新的弱引用指针obj2
        make_weak_only<DestructableMock>(&resourcesReleased, &wasDestructed);  // 以resourcesReleased和wasDestructed的地址创建一个新的DestructableMock对象，并将其指针赋给obj2
    EXPECT_TRUE(resourcesReleased);  // 断言resourcesReleased为true
    EXPECT_FALSE(wasDestructed);  // 断言wasDestructed为false
    obj2 = std::move(obj);  // 将obj移动给obj2，此操作将销毁旧的obj2指向的对象
    EXPECT_TRUE(resourcesReleased);  // 断言resourcesReleased为true
    EXPECT_TRUE(wasDestructed);  // 断言wasDestructed为true，表示旧的obj2指向的对象已被销毁
  }
}

TEST(
    WeakIntrusivePtrTest,
    givenPtrWithCopy_whenMoveAssigned_thenDestructsOldObjectAfterCopyIsDestructed) {
  bool dummy = false;  // 创建一个布尔变量dummy，并初始化为false
  bool resourcesReleased = false;  // 创建一个布尔变量resourcesReleased，并初始化为false
  bool wasDestructed = false;  // 创建一个布尔变量wasDestructed，并初始化为false
  auto obj = make_weak_only<DestructableMock>(&dummy, &dummy);  // 创建一个弱引用指针obj，指向一个DestructableMock对象，该对象使用dummy变量的地址作为参数
  {
    auto obj2 =  // 创建一个新的弱引用指针obj2
        make_weak_only<DestructableMock>(&resourcesReleased, &wasDestructed);  // 以resourcesReleased和wasDestructed的地址创建一个新的DestructableMock对象，并将其指针赋给obj2
    {
      auto copy = obj2;  // 创建一个新的弱引用指针copy，并将其指向obj2指向的对象，执行复制构造
      EXPECT_TRUE(resourcesReleased);  // 断言resourcesReleased为true
      EXPECT_FALSE(wasDestructed);  // 断言wasDestructed为false
      obj2 = std::move(obj);  // 将obj移动给obj2，此操作将销毁旧的obj2指向的对象
      EXPECT_TRUE(resourcesReleased);  // 断言resourcesReleased为true
      EXPECT_FALSE(wasDestructed);  // 断言wasDestructed为false，因为copy对象仍存在
    }
    EXPECT_TRUE(resourcesReleased);  // 断言resourcesReleased为true
    EXPECT_TRUE(wasDestructed);  // 断言wasDestructed为true，表示旧的obj2指向的对象已被销毁
  }
}

TEST(
    WeakIntrusivePtrTest,
    givenPtrWithBaseClassCopy_whenMoveAssigned_thenDestructsOldObjectAfterCopyIsDestructed) {
  bool dummy = false;  // 创建一个布尔变量dummy，并初始化为false
  bool resourcesReleased = false;  // 创建一个布尔变量resourcesReleased，并初始化为false
  bool wasDestructed = false;  // 创建一个布尔变量wasDestructed，并初始化为false
  auto obj = make_weak_only<ChildDestructableMock>(&dummy, &dummy);  // 创建一个弱引用指针obj，指向一个ChildDestructableMock对象，该对象使用dummy变量的地址作为参数
  {
    auto obj2 =  // 创建一个新的弱引用指针obj2
        make_weak_only<ChildDestructableMock>(&resourcesReleased, &wasDestructed);  // 以resourcesReleased和wasDestructed的地址创建一个新的ChildDestructableMock对象，并将其指针赋给obj2
    {
      weak_intrusive_ptr<DestructableMock> copy = obj2;  // 创建一个新的弱引用指针copy，指向obj2指向的对象，类型为DestructableMock
      EXPECT_TRUE(resourcesReleased);  // 断言resourcesReleased为true
      EXPECT_FALSE(wasDestructed);  // 断言wasDestructed为false
      obj2 = std::move(obj);  // 将obj移动给obj2，此操作将销毁旧的obj2指向的对象
      EXPECT_TRUE(resourcesReleased);  // 断言resourcesReleased为true
      EXPECT_FALSE(wasDestructed);  // 断言wasDestructed为false，因为copy对象仍存在
    }
    EXPECT_TRUE(resourcesReleased);  // 断言resourcesReleased为true
    EXPECT_TRUE(wasDestructed);  // 断言wasDestructed为true，表示旧的obj2指向的对象已被销毁
  }
}

TEST(
    WeakIntrusivePtrTest,
    givenPtrWithCopy_whenMoveAssignedToBaseClass_thenDestructsOldObjectAfterCopyIsDestructed) {
  bool dummy = false;  // 创建一个布尔变量dummy，并初始化为false
  bool resourcesReleased = false;  // 创建一个布尔变量resourcesReleased，并初始化为false
  bool wasDestructed = false;  // 创建一个布尔变量wasDestructed，并初始化为false
  auto obj = make_weak_only<ChildDestructableMock>(&dummy, &dummy);  // 创建一个弱引用指针obj，指向一个ChildDestructableMock对象，该对象使用dummy变量的地址作为参数
  {
    auto obj2 =  // 创建一个新的弱引用指针obj2
        make_weak_only<DestructableMock>(&resourcesReleased, &wasDestructed);  // 以resourcesReleased和wasDestructed的地址创建一个新的DestructableMock对象，并将其指针赋给obj2
    {
      weak_intrusive_ptr<DestructableMock> copy = obj2;  // 创建一个新的弱引用指针copy，指向obj2指向的对象，类型为DestructableMock
      EXPECT_TRUE(resourcesReleased);  // 断言resourcesReleased为true
      EXPECT_FALSE(wasDestructed);  // 断言wasDestructed为false
      obj2 = std::move(obj);  // 将obj移动给obj2，此操作将销毁旧的obj2指向的对象
      EXPECT_TRUE(resourcesReleased);  // 断言resourcesReleased为true
      EXPECT_FALSE(wasDestructed);  // 断言wasDestructed为false，因为copy对象仍存在
    }
    EXPECT_TRUE(resourcesReleased);  // 断言resourcesReleased为true
    EXPECT_TRUE(wasDestructed);  // 断言wasDestructed为true，表示旧的obj
    # 断言资源已释放
    EXPECT_TRUE(resourcesReleased);
    # 断言对象已析构
    EXPECT_TRUE(wasDestructed);
}

TEST(
    WeakIntrusivePtrTest,
    givenPtr_whenMoveAssigned_thenDestructsObjectAfterSecondDestructed) {
  bool dummy = false;  // 定义一个布尔变量 dummy，初始为 false
  bool resourcesReleased = false;  // 定义一个布尔变量 resourcesReleased，初始为 false
  bool wasDestructed = false;  // 定义一个布尔变量 wasDestructed，初始为 false
  auto obj =  // 创建一个弱引用指针对象 obj，指向 DestructableMock 类，传入 resourcesReleased 和 wasDestructed 的地址
      make_weak_only<DestructableMock>(&resourcesReleased, &wasDestructed);
  {
    auto obj2 =  // 创建另一个弱引用指针对象 obj2，指向 DestructableMock 类，传入 dummy 和 dummy 的地址
        make_weak_only<DestructableMock>(&dummy, &dummy);
    obj2 = std::move(obj);  // 将 obj 移动给 obj2
    EXPECT_TRUE(resourcesReleased);  // 断言：resourcesReleased 应为 true
    EXPECT_FALSE(wasDestructed);  // 断言：wasDestructed 应为 false
  }
  EXPECT_TRUE(resourcesReleased);  // 断言：resourcesReleased 应为 true
  EXPECT_TRUE(wasDestructed);  // 断言：wasDestructed 应为 true
}

TEST(
    WeakIntrusivePtrTest,
    givenPtr_whenMoveAssignedToBaseClass_thenDestructsObjectAfterSecondDestructed) {
  bool dummy = false;  // 定义一个布尔变量 dummy，初始为 false
  bool resourcesReleased = false;  // 定义一个布尔变量 resourcesReleased，初始为 false
  bool wasDestructed = false;  // 定义一个布尔变量 wasDestructed，初始为 false
  auto obj =  // 创建一个弱引用指针对象 obj，指向 ChildDestructableMock 类，传入 resourcesReleased 和 wasDestructed 的地址
      make_weak_only<ChildDestructableMock>(&resourcesReleased, &wasDestructed);
  {
    auto obj2 =  // 创建另一个弱引用指针对象 obj2，指向 DestructableMock 类，传入 dummy 和 dummy 的地址
        make_weak_only<DestructableMock>(&dummy, &dummy);
    obj2 = std::move(obj);  // 将 obj 移动给 obj2
    EXPECT_TRUE(resourcesReleased);  // 断言：resourcesReleased 应为 true
    EXPECT_FALSE(wasDestructed);  // 断言：wasDestructed 应为 false
  }
  EXPECT_TRUE(resourcesReleased);  // 断言：resourcesReleased 应为 true
  EXPECT_TRUE(wasDestructed);  // 断言：wasDestructed 应为 true
}

TEST(
    WeakIntrusivePtrTest,
    givenPtr_whenCopyConstructedAndDestructed_thenDestructsObjectAfterLastDestruction) {
  bool resourcesReleased = false;  // 定义一个布尔变量 resourcesReleased，初始为 false
  bool wasDestructed = false;  // 定义一个布尔变量 wasDestructed，初始为 false
  {
    auto obj =  // 创建一个弱引用指针对象 obj，指向 DestructableMock 类，传入 resourcesReleased 和 wasDestructed 的地址
        make_weak_only<DestructableMock>(&resourcesReleased, &wasDestructed);
    {
      // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
      weak_intrusive_ptr<DestructableMock> copy = obj;  // 通过拷贝构造函数创建 obj 的副本 copy
      EXPECT_TRUE(resourcesReleased);  // 断言：resourcesReleased 应为 true
      EXPECT_FALSE(wasDestructed);  // 断言：wasDestructed 应为 false
    }
    EXPECT_TRUE(resourcesReleased);  // 断言：resourcesReleased 应为 true
    EXPECT_FALSE(wasDestructed);  // 断言：wasDestructed 应为 false
  }
  EXPECT_TRUE(resourcesReleased);  // 断言：resourcesReleased 应为 true
  EXPECT_TRUE(wasDestructed);  // 断言：wasDestructed 应为 true
}

TEST(
    WeakIntrusivePtrTest,
    givenPtr_whenCopyConstructedToBaseClassAndDestructed_thenDestructsObjectAfterLastDestruction) {
  bool resourcesReleased = false;  // 定义一个布尔变量 resourcesReleased，初始为 false
  bool wasDestructed = false;  // 定义一个布尔变量 wasDestructed，初始为 false
  {
    auto obj =  // 创建一个弱引用指针对象 obj，指向 ChildDestructableMock 类，传入 resourcesReleased 和 wasDestructed 的地址
        make_weak_only<ChildDestructableMock>(&resourcesReleased, &wasDestructed);
    {
      weak_intrusive_ptr<DestructableMock> copy = obj;  // 通过拷贝构造函数创建 obj 的副本 copy
      EXPECT_TRUE(resourcesReleased);  // 断言：resourcesReleased 应为 true
      EXPECT_FALSE(wasDestructed);  // 断言：wasDestructed 应为 false
    }
    EXPECT_TRUE(resourcesReleased);  // 断言：resourcesReleased 应为 true
    EXPECT_FALSE(wasDestructed);  // 断言：wasDestructed 应为 false
  }
  EXPECT_TRUE(resourcesReleased);  // 断言：resourcesReleased 应为 true
  EXPECT_TRUE(wasDestructed);  // 断言：wasDestructed 应为 true
}

TEST(
    WeakIntrusivePtrTest,
    givenPtr_whenCopyConstructedAndOriginalDestructed_thenDestructsObjectAfterLastDestruction) {
  bool resourcesReleased = false;  // 定义一个布尔变量 resourcesReleased，初始为 false
  bool wasDestructed = false;  // 定义一个布尔变量 wasDestructed，初始为 false
  {
    auto obj =  // 创建一个弱引用指针对象 obj，指向 DestructableMock 类，传入 resourcesReleased 和 wasDestructed 的地址
        make_weak_only<DestructableMock>(&resourcesReleased, &wasDestructed);
    weak_intrusive_ptr<DestructableMock> copy = obj;  // 通过拷贝构造函数创建 obj 的副本 copy
    obj.reset();  // 重置 obj，触发资源释放
    EXPECT_TRUE(resourcesReleased);  // 断言：resourcesReleased 应为 true
    EXPECT_FALSE(wasDestructed);  // 断言：wasDestructed 应为 false
  }
  EXPECT_TRUE(resourcesReleased);  // 断言：resourcesReleased 应为 true
  EXPECT_TRUE(wasDestructed);  // 断言：wasDestructed 应为 true
}

TEST(
    WeakIntrusivePtrTest,
    givenPtr_whenCopyConstructedToBaseClassAndOriginalDestructed_thenDestructsObjectAfterLastDestruction) {
  bool resourcesReleased = false;  // 定义一个布尔变量 resourcesReleased，初始为 false
  bool wasDestructed = false;  // 定义一个布尔变量 wasDestructed，初始为 false
  {
    // 创建一个弱引用指针 `obj`，指向 `ChildDestructableMock` 类型的对象，
    // 并使用 `make_weak_only` 函数确保只有弱引用，传入资源释放和析构标志的指针
    auto obj = make_weak_only<ChildDestructableMock>(&resourcesReleased, &wasDestructed);
    
    // 创建另一个弱引用指针 `copy`，指向 `DestructableMock` 类型的对象，从 `obj` 复制而来
    weak_intrusive_ptr<DestructableMock> copy = obj;
    
    // 释放 `obj` 指向的对象，此时 `copy` 仍指向同一对象
    obj.reset();
    
    // 断言资源已释放，即 `resourcesReleased` 应为真
    EXPECT_TRUE(resourcesReleased);
    
    // 断言对象未被析构，即 `wasDestructed` 应为假
    EXPECT_FALSE(wasDestructed);
  }
  
  // 在代码块结束后再次断言资源已释放，此时 `copy` 指向的对象应已析构
  EXPECT_TRUE(resourcesReleased);
  
  // 断言对象已被析构，即 `wasDestructed` 应为真
  EXPECT_TRUE(wasDestructed);
}

TEST(
    WeakIntrusivePtrTest,
    givenPtr_whenCopyAssignedAndDestructed_thenDestructsObjectAfterLastDestruction) {
  bool resourcesReleased = false;
  bool wasDestructed = false;
  bool dummy = false;
  {
    // 创建一个使用自定义析构函数的弱引用指针对象
    auto obj =
        make_weak_only<DestructableMock>(&resourcesReleased, &wasDestructed);
    {
      // 创建一个新的弱引用指针对象，使用另一组参数
      weak_intrusive_ptr<DestructableMock> copy =
          make_weak_only<DestructableMock>(&dummy, &dummy);
      // 将第一个对象的引用赋值给第二个对象
      copy = obj;
      // 检查资源是否已被释放
      EXPECT_TRUE(resourcesReleased);
      // 检查对象是否已析构
      EXPECT_FALSE(wasDestructed);
    }
    // 再次检查资源是否已被释放
    EXPECT_TRUE(resourcesReleased);
    // 再次检查对象是否已析构
    EXPECT_FALSE(wasDestructed);
  }
  // 最终检查资源是否已被释放
  EXPECT_TRUE(resourcesReleased);
  // 最终检查对象是否已析构
  EXPECT_TRUE(wasDestructed);
}

TEST(
    WeakIntrusivePtrTest,
    givenPtr_whenCopyAssignedToBaseClassAndDestructed_thenDestructsObjectAfterLastDestruction) {
  bool resourcesReleased = false;
  bool wasDestructed = false;
  bool dummy = false;
  {
    // 创建一个使用自定义析构函数的子类对象的弱引用指针对象
    auto obj = make_weak_only<ChildDestructableMock>(
        &resourcesReleased, &wasDestructed);
    {
      // 创建一个新的弱引用指针对象，使用另一组参数
      weak_intrusive_ptr<DestructableMock> copy =
          make_weak_only<DestructableMock>(&dummy, &dummy);
      // 将第一个对象的引用赋值给第二个对象
      copy = obj;
      // 检查资源是否已被释放
      EXPECT_TRUE(resourcesReleased);
      // 检查对象是否已析构
      EXPECT_FALSE(wasDestructed);
    }
    // 再次检查资源是否已被释放
    EXPECT_TRUE(resourcesReleased);
    // 再次检查对象是否已析构
    EXPECT_FALSE(wasDestructed);
  }
  // 最终检查资源是否已被释放
  EXPECT_TRUE(resourcesReleased);
  // 最终检查对象是否已析构
  EXPECT_TRUE(wasDestructed);
}

TEST(
    WeakIntrusivePtrTest,
    givenPtr_whenCopyAssignedAndOriginalDestructed_thenDestructsObjectAfterLastDestruction) {
  bool resourcesReleased = false;
  bool wasDestructed = false;
  bool dummy = false;
  {
    // 创建一个新的弱引用指针对象，使用另一组参数
    auto copy = make_weak_only<DestructableMock>(&dummy, &dummy);
    {
      // 创建一个使用自定义析构函数的弱引用指针对象
      auto obj =
          make_weak_only<DestructableMock>(&resourcesReleased, &wasDestructed);
      // 将第一个对象的引用赋值给第二个对象
      copy = obj;
      // 检查资源是否已被释放
      EXPECT_TRUE(resourcesReleased);
      // 检查对象是否已析构
      EXPECT_FALSE(wasDestructed);
    }
    // 再次检查资源是否已被释放
    EXPECT_TRUE(resourcesReleased);
    // 再次检查对象是否已析构
    EXPECT_FALSE(wasDestructed);
  }
  // 最终检查资源是否已被释放
  EXPECT_TRUE(resourcesReleased);
  // 最终检查对象是否已析构
  EXPECT_TRUE(wasDestructed);
}

TEST(
    WeakIntrusivePtrTest,
    givenPtr_whenCopyAssignedToBaseClassAndOriginalDestructed_thenDestructsObjectAfterLastDestruction) {
  bool wasDestructed = false;
  bool resourcesReleased = false;
  bool dummy = false;
  {
    // 创建一个新的弱引用指针对象，使用另一组参数
    auto copy = make_weak_only<DestructableMock>(&dummy, &dummy);
    {
      // 创建一个使用自定义析构函数的子类对象的弱引用指针对象
      auto obj = make_weak_only<ChildDestructableMock>(
          &resourcesReleased, &wasDestructed);
      // 将第一个对象的引用赋值给第二个对象
      copy = obj;
      // 检查资源是否已被释放
      EXPECT_TRUE(resourcesReleased);
      // 检查对象是否已析构
      EXPECT_FALSE(wasDestructed);
    }
    // 再次检查资源是否已被释放
    EXPECT_TRUE(resourcesReleased);
    // 再次检查对象是否已析构
    EXPECT_FALSE(wasDestructed);
  }
  // 最终检查资源是否已被释放
  EXPECT_TRUE(resourcesReleased);
  // 最终检查对象是否已析构
  EXPECT_TRUE(wasDestructed);
}

TEST(WeakIntrusivePtrTest, givenPtr_whenCopyAssigned_thenDestructsOldObject) {
  bool dummy = false;
  bool resourcesReleased = false;
  bool wasDestructed = false;
  // 创建一个使用自定义析构函数的弱引用指针对象
  auto obj = make_weak_only<DestructableMock>(&dummy, &dummy);
  {
    // 创建一个使用自定义析构函数的弱引用指针对象
    auto obj2 =
        make_weak_only<DestructableMock>(&resourcesReleased, &wasDestructed);
    // 将第一个对象的引用赋值给第二个对象
    obj2 = obj;
    // 检查资源是否已被释放
    EXPECT_TRUE(resourcesReleased);
    // 检查对象是否已析构
    EXPECT_FALSE(wasDestructed);
    // 断言检查：验证资源是否已释放
    EXPECT_TRUE(resourcesReleased);
    // 断言检查：验证对象是否已析构
    EXPECT_TRUE(wasDestructed);
}

TEST(
    WeakIntrusivePtrTest,
    givenPtr_whenCopyAssignedToBaseClass_thenDestructsOldObject) {
  bool dummy = false;
  bool resourcesReleased = false;
  bool wasDestructed = false;
  auto obj = make_weak_only<ChildDestructableMock>(&dummy, &dummy);
  {
    auto obj2 =
        make_weak_only<DestructableMock>(&resourcesReleased, &wasDestructed);
    EXPECT_TRUE(resourcesReleased);
    EXPECT_FALSE(wasDestructed);
    obj2 = obj;  // 将 obj 赋值给 obj2，期望旧对象被销毁
    EXPECT_TRUE(resourcesReleased);
    EXPECT_TRUE(wasDestructed);  // 确认旧对象已被销毁
  }
}

TEST(
    WeakIntrusivePtrTest,
    givenPtrWithCopy_whenCopyAssigned_thenDestructsOldObjectAfterCopyIsDestructed) {
  bool dummy = false;
  bool resourcesReleased = false;
  bool wasDestructed = false;
  auto obj = make_weak_only<DestructableMock>(&dummy, &dummy);
  {
    auto obj2 =
        make_weak_only<DestructableMock>(&resourcesReleased, &wasDestructed);
    {
      auto copy = obj2;  // 创建 obj2 的一个副本 copy
      EXPECT_TRUE(resourcesReleased);
      EXPECT_FALSE(wasDestructed);
      obj2 = obj;  // 将 obj 赋值给 obj2
      EXPECT_TRUE(resourcesReleased);
      EXPECT_FALSE(wasDestructed);  // 确认 obj2 的旧对象未被销毁
    }
    EXPECT_TRUE(resourcesReleased);
    EXPECT_TRUE(wasDestructed);  // 确认 obj2 的旧对象已被销毁
  }
}

TEST(
    WeakIntrusivePtrTest,
    givenPtrWithBaseClassCopy_whenCopyAssigned_thenDestructsOldObjectAfterCopyIsDestructed) {
  bool dummy = false;
  bool resourcesReleased = false;
  bool wasDestructed = false;
  auto obj = make_weak_only<ChildDestructableMock>(&dummy, &dummy);
  {
    auto obj2 = make_weak_only<ChildDestructableMock>(
        &resourcesReleased, &wasDestructed);
    {
      weak_intrusive_ptr<DestructableMock> copy = obj2;  // 将 obj2 转换为 DestructableMock 类型的弱引用指针 copy
      EXPECT_TRUE(resourcesReleased);
      EXPECT_FALSE(wasDestructed);
      obj2 = obj;  // 将 obj 赋值给 obj2
      EXPECT_TRUE(resourcesReleased);
      EXPECT_FALSE(wasDestructed);  // 确认 obj2 的旧对象未被销毁
    }
    EXPECT_TRUE(resourcesReleased);
    EXPECT_TRUE(wasDestructed);  // 确认 obj2 的旧对象已被销毁
  }
}

TEST(
    WeakIntrusivePtrTest,
    givenPtrWithCopy_whenCopyAssignedToBaseClass_thenDestructsOldObjectAfterCopyIsDestructed) {
  bool dummy = false;
  bool resourcesReleased = false;
  bool wasDestructed = false;
  auto obj = make_weak_only<ChildDestructableMock>(&dummy, &dummy);
  {
    auto obj2 =
        make_weak_only<DestructableMock>(&resourcesReleased, &wasDestructed);
    {
      weak_intrusive_ptr<DestructableMock> copy = obj2;  // 将 obj2 转换为 DestructableMock 类型的弱引用指针 copy
      EXPECT_TRUE(resourcesReleased);
      EXPECT_FALSE(wasDestructed);
      obj2 = obj;  // 将 obj 赋值给 obj2
      EXPECT_TRUE(resourcesReleased);
      EXPECT_FALSE(wasDestructed);  // 确认 obj2 的旧对象未被销毁
    }
    EXPECT_TRUE(resourcesReleased);
    EXPECT_TRUE(wasDestructed);  // 确认 obj2 的旧对象已被销毁
  }
}

TEST(WeakIntrusivePtrTest, givenPtr_whenCallingReset_thenDestructs) {
  bool resourcesReleased = false;
  bool wasDestructed = false;
  auto obj =
      make_weak_only<DestructableMock>(&resourcesReleased, &wasDestructed);
  EXPECT_TRUE(resourcesReleased);
  EXPECT_FALSE(wasDestructed);
  obj.reset();  // 调用 reset() 方法，期望对象被销毁
  EXPECT_TRUE(resourcesReleased);
  EXPECT_TRUE(wasDestructed);  // 确认对象已被销毁
}

TEST(
    WeakIntrusivePtrTest,
  givenPtrWithCopy_whenCallingReset_thenDestructsAfterCopyDestructed) {
  // 初始化两个布尔变量，用于跟踪资源释放和对象析构状态
  bool resourcesReleased = false;
  bool wasDestructed = false;
  // 使用 make_weak_only 创建一个 DestructableMock 类型的智能指针 obj
  auto obj =
      make_weak_only<DestructableMock>(&resourcesReleased, &wasDestructed);
  {
    // 创建 obj 的一个副本 copy
    auto copy = obj;
    // 重置 obj 指针，此时预期资源已释放，但对象尚未析构
    obj.reset();
    // 断言资源已释放为真
    EXPECT_TRUE(resourcesReleased);
    // 断言对象未析构为假
    EXPECT_FALSE(wasDestructed);
    // 重置 copy 指针，预期资源已释放且对象已析构
    copy.reset();
    // 断言资源已释放为真
    EXPECT_TRUE(resourcesReleased);
    // 断言对象已析构为真
    EXPECT_TRUE(wasDestructed);
  }
}

TEST(
    WeakIntrusivePtrTest,
    givenPtrWithCopy_whenCallingResetOnCopy_thenDestructsAfterOriginalDestructed) {
  bool resourcesReleased = false;  // 标记资源是否已释放
  bool wasDestructed = false;       // 标记对象是否已析构
  auto obj =
      make_weak_only<DestructableMock>(&resourcesReleased, &wasDestructed);  // 创建一个弱引用对象，传入资源释放标记和析构标记
  {
    auto copy = obj;  // 复制obj，增加引用计数
    copy.reset();     // 重置copy，减少引用计数
    EXPECT_TRUE(resourcesReleased);  // 检查资源是否已释放
    EXPECT_FALSE(wasDestructed);      // 检查对象是否未被析构
    obj.reset();                      // 重置obj，减少引用计数
    EXPECT_TRUE(resourcesReleased);   // 检查资源是否已释放
    EXPECT_TRUE(wasDestructed);       // 检查对象是否已被析构
  }
}

TEST(
    WeakIntrusivePtrTest,
    givenPtrWithMoved_whenCallingReset_thenDestructsAfterMovedDestructed) {
  bool resourcesReleased = false;  // 标记资源是否已释放
  bool wasDestructed = false;       // 标记对象是否已析构
  auto obj =
      make_weak_only<DestructableMock>(&resourcesReleased, &wasDestructed);  // 创建一个弱引用对象，传入资源释放标记和析构标记
  {
    auto moved = std::move(obj);  // 移动obj，转移所有权到moved
    // NOLINTNEXTLINE(bugprone-use-after-move)
    obj.reset();                  // 重置obj，减少引用计数
    EXPECT_TRUE(resourcesReleased);  // 检查资源是否已释放
    EXPECT_FALSE(wasDestructed);      // 检查对象是否未被析构
    moved.reset();                    // 重置moved，减少引用计数
    EXPECT_TRUE(resourcesReleased);   // 检查资源是否已释放
    EXPECT_TRUE(wasDestructed);       // 检查对象是否已被析构
  }
}

TEST(
    WeakIntrusivePtrTest,
    givenPtrWithMoved_whenCallingResetOnMoved_thenDestructsImmediately) {
  bool resourcesReleased = false;  // 标记资源是否已释放
  bool wasDestructed = false;       // 标记对象是否已析构
  auto obj =
      make_weak_only<DestructableMock>(&resourcesReleased, &wasDestructed);  // 创建一个弱引用对象，传入资源释放标记和析构标记
  {
    auto moved = std::move(obj);  // 移动obj，转移所有权到moved
    moved.reset();                // 重置moved，减少引用计数
    EXPECT_TRUE(resourcesReleased);  // 检查资源是否已释放
    EXPECT_TRUE(wasDestructed);       // 检查对象是否已被析构
  }
}

TEST(WeakIntrusivePtrTest, givenPtr_whenReleasedAndReclaimed_thenDoesntCrash) {
  IntrusiveAndWeak<SomeClass> obj = make_weak_intrusive<SomeClass>();  // 创建一个强引用和弱引用对象
  SomeClass* ptr = obj.weak.release();  // 释放弱引用，返回指针
  weak_intrusive_ptr<SomeClass> reclaimed =
      weak_intrusive_ptr<SomeClass>::reclaim(ptr);  // 通过指针重新获取弱引用对象
}

TEST(
    WeakIntrusivePtrTest,
    givenWeakOnlyPtr_whenReleasedAndReclaimed_thenDoesntCrash) {
  weak_intrusive_ptr<SomeClass> obj = make_weak_only<SomeClass>();  // 创建一个仅弱引用对象
  SomeClass* ptr = obj.release();  // 释放对象，返回指针
  weak_intrusive_ptr<SomeClass> reclaimed =
      weak_intrusive_ptr<SomeClass>::reclaim(ptr);  // 通过指针重新获取弱引用对象
}

TEST(
    WeakIntrusivePtrTest,
    givenPtr_whenReleasedAndReclaimed_thenIsDestructedAtEnd) {
  bool resourcesReleased = false;  // 标记资源是否已释放
  bool wasDestructed = false;       // 标记对象是否已析构
  bool dummy = false;
  {
    IntrusiveAndWeak<DestructableMock> outer =
        make_weak_intrusive<DestructableMock>(&dummy, &dummy);  // 创建一个强引用和弱引用对象，传入dummy参数
    {
      IntrusiveAndWeak<DestructableMock> inner =
          make_weak_intrusive<DestructableMock>(
              &resourcesReleased, &wasDestructed);  // 创建一个强引用和弱引用对象，传入资源释放标记和析构标记
      EXPECT_FALSE(resourcesReleased);  // 检查资源是否未释放
      EXPECT_FALSE(wasDestructed);       // 检查对象是否未析构
      DestructableMock* ptr = inner.weak.release();  // 释放内部对象的弱引用，返回指针
      EXPECT_FALSE(resourcesReleased);  // 检查资源是否未释放
      EXPECT_FALSE(wasDestructed);       // 检查对象是否未析构
      outer.ptr = inner.ptr;  // 将外部对象的指针设置为内部对象的指针
      outer.weak = weak_intrusive_ptr<DestructableMock>::reclaim(ptr);  // 通过指针重新获取弱引用对象
    }
    // inner is destructed
    EXPECT_FALSE(resourcesReleased);  // 检查资源是否未释放
    EXPECT_FALSE(wasDestructed);       // 检查对象是否未析构
    outer.weak.reset();                // 重置外部对象的弱引用，减少引用计数
    EXPECT_FALSE(resourcesReleased);  // 检查资源是否未释放
    // 检查 wasDestructed 是否为假，断言失败如果 wasDestructed 不为假
    EXPECT_FALSE(wasDestructed);
  }
  // 外部对象已析构
  // 断言成功如果 resourcesReleased 为真，表示资源已释放
  EXPECT_TRUE(resourcesReleased);
  // 断言成功如果 wasDestructed 为真，表示对象已析构
  EXPECT_TRUE(wasDestructed);
}

TEST(
    WeakIntrusivePtrTest,
    givenWeakOnlyPtr_whenReleasedAndReclaimed_thenIsDestructedAtEnd) {
  // 初始化标志变量
  bool resourcesReleased = false;
  bool wasDestructed = false;
  {
    // 创建外部弱指针，指向一个无效对象
    weak_intrusive_ptr<DestructableMock> outer =
        make_invalid_weak<DestructableMock>();
    {
      // 创建内部仅弱指针，管理可析构对象，监控资源释放和析构调用
      weak_intrusive_ptr<DestructableMock> inner =
          make_weak_only<DestructableMock>(&resourcesReleased, &wasDestructed);
      // 检查资源是否已释放，预期为 true
      EXPECT_TRUE(resourcesReleased);
      // 检查对象是否已析构，预期为 false
      EXPECT_FALSE(wasDestructed);
      // 释放内部指针并获取指向的对象
      DestructableMock* ptr = inner.release();
      // 检查资源是否已释放，预期为 true
      EXPECT_TRUE(resourcesReleased);
      // 检查对象是否已析构，预期为 false
      EXPECT_FALSE(wasDestructed);
      // 重新声明外部弱指针来回收已释放的内部对象
      outer = weak_intrusive_ptr<DestructableMock>::reclaim(ptr);
    }
    // 内部对象已析构
    EXPECT_TRUE(resourcesReleased);
    // 检查对象是否已析构，预期为 false
    EXPECT_FALSE(wasDestructed);
  }
  // 外部对象已析构
  EXPECT_TRUE(resourcesReleased);
  // 检查对象是否已析构，预期为 true
  EXPECT_TRUE(wasDestructed);
}

TEST(WeakIntrusivePtrTest, givenStackObject_whenReclaimed_thenCrashes) {
  // 在对象析构时会导致非常奇怪的bug
  // 最好在创建时尽早崩溃
  SomeClass obj;
  // 创建一个无效的弱指针指向 SomeClass 对象
  weak_intrusive_ptr<SomeClass> ptr = make_invalid_weak<SomeClass>();
#ifdef NDEBUG
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  // 在 release 模式下，预期不会抛出异常
  EXPECT_NO_THROW(ptr = weak_intrusive_ptr<SomeClass>::reclaim(&obj));
#else
  // 在 debug 模式下，预期会抛出异常
  EXPECT_ANY_THROW(ptr = weak_intrusive_ptr<SomeClass>::reclaim(&obj));
#endif
}

TEST(
    WeakIntrusivePtrTest,
    givenObjectWithWeakReferenceToSelf_whenDestroyed_thenDoesNotCrash) {
  // 创建自引用对象的强指针
  auto p = make_intrusive<WeakReferenceToSelf>();
  // 将自引用对象赋给其内部指针
  p->ptr = weak_intrusive_ptr<intrusive_ptr_target>(
      intrusive_ptr<intrusive_ptr_target>(p));
}
// NOLINTEND(clang-analyzer-cplusplus*)
```