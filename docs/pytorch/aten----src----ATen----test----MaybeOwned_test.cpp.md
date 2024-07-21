# `.\pytorch\aten\src\ATen\test\MaybeOwned_test.cpp`

```
#include <gtest/gtest.h> // 包含 Google Test 框架的头文件

#include <ATen/Functions.h> // 包含 ATen 库的函数定义
#include <ATen/NativeFunctions.h> // 包含 ATen 库的原生函数定义
#include <ATen/Tensor.h> // 包含 ATen 库的 Tensor 类定义
#include <ATen/core/ivalue.h> // 包含 ATen 库的 IValue 类定义
#include <c10/util/intrusive_ptr.h> // 包含 c10 库的 intrusive_ptr 类定义
#include <c10/util/MaybeOwned.h> // 包含 c10 库的 MaybeOwned 类定义

#include <memory> // 包含 C++ 标准库的内存管理相关头文件
#include <string> // 包含 C++ 标准库的字符串处理相关头文件

namespace { // 匿名命名空间，用于限定符号的作用域

using at::Tensor; // 使用 ATen 库中的 Tensor 类
using c10::IValue; // 使用 c10 库中的 IValue 类

struct MyString : public c10::intrusive_ptr_target, public std::string { // 定义 MyString 结构体，继承自 std::string 和 c10::intrusive_ptr_target
  using std::string::string; // 继承 std::string 的构造函数
};

template <typename T>
class MaybeOwnedTest : public ::testing::Test { // 定义 MaybeOwnedTest 模板类，继承自 Google Test 的 Test 类
 public:
  T borrowFrom; // borrowFrom 成员，用于借用对象
  T ownCopy; // ownCopy 成员，用于拥有对象的拷贝
  T ownCopy2; // ownCopy2 成员，用于拥有对象的另一个拷贝
  c10::MaybeOwned<T> borrowed; // borrowed 成员，用于可能借用对象
  c10::MaybeOwned<T> owned; // owned 成员，用于拥有对象
  c10::MaybeOwned<T> owned2; // owned2 成员，用于拥有对象的另一个实例

 protected:
  void SetUp() override; // 设置测试前的准备工作，下面有具体的实现
  void TearDown() override { // 清理测试后的工作，释放资源
    // Release everything to try to trigger ASAN violations in the
    // test that broke things.
    borrowFrom = T(); // 将 borrowFrom 清空
    ownCopy = T(); // 将 ownCopy 清空
    ownCopy2 = T(); // 将 ownCopy2 清空

    borrowed = c10::MaybeOwned<T>(); // 将 borrowed 重置为空
    owned = c10::MaybeOwned<T>(); // 将 owned 重置为空
    owned2 = c10::MaybeOwned<T>(); // 将 owned2 重置为空
  }

};


//////////////////// Helpers that differ per tested type. ////////////////////

template <typename T>
T getSampleValue(); // 获取样本值的模板函数声明

template <typename T>
T getSampleValue2(); // 获取另一种样本值的模板函数声明

template <typename T>
void assertBorrow(const c10::MaybeOwned<T>&, const T&); // 断言借用对象的模板函数声明

template <typename T>
void assertOwn(const c10::MaybeOwned<T>&, const T&, size_t useCount = 2); // 断言拥有对象的模板函数声明

////////////////// Helper implementations for intrusive_ptr. //////////////////
template<>
c10::intrusive_ptr<MyString> getSampleValue() { // intrusive_ptr<MyString> 类型的样本值获取实现
  return c10::make_intrusive<MyString>("hello"); // 使用 make_intrusive 创建 MyString 类型的 intrusive_ptr
}

template<>
c10::intrusive_ptr<MyString> getSampleValue2() { // intrusive_ptr<MyString> 类型的另一种样本值获取实现
  return c10::make_intrusive<MyString>("goodbye"); // 使用 make_intrusive 创建 MyString 类型的 intrusive_ptr
}

bool are_equal(const c10::intrusive_ptr<MyString>& lhs, const c10::intrusive_ptr<MyString>& rhs) { // 判断两个 intrusive_ptr<MyString> 是否相等的函数实现
  if (!lhs || !rhs) { // 如果任意一个为空指针
    return !lhs && !rhs; // 则判断两者是否都为空指针
  }
  return *lhs == *rhs; // 否则比较它们指向的值是否相等
}

template <>
void assertBorrow( // 断言借用 intrusive_ptr<MyString> 对象的具体实现
    const c10::MaybeOwned<c10::intrusive_ptr<MyString>>& mo,
    const c10::intrusive_ptr<MyString>& borrowedFrom) {
  EXPECT_EQ(*mo, borrowedFrom); // 使用 Google Test 的断言判断 mo 是否与 borrowedFrom 相等
  EXPECT_EQ(mo->get(), borrowedFrom.get()); // 使用 Google Test 的断言判断 mo 所指向对象的指针是否与 borrowedFrom 相等
  EXPECT_EQ(borrowedFrom.use_count(), 1); // 使用 Google Test 的断言判断 borrowedFrom 的引用计数是否为 1
}

template <>
void assertOwn( // 断言拥有 intrusive_ptr<MyString> 对象的具体实现
    const c10::MaybeOwned<c10::intrusive_ptr<MyString>>& mo,
    const c10::intrusive_ptr<MyString>& original,
    size_t useCount) {
  EXPECT_EQ(*mo, original); // 使用 Google Test 的断言判断 mo 是否与 original 相等
  EXPECT_EQ(mo->get(), original.get()); // 使用 Google Test 的断言判断 mo 所指向对象的指针是否与 original 相等
  EXPECT_NE(&*mo, &original); // 使用 Google Test 的断言判断 mo 和 original 的地址不相等
  EXPECT_EQ(original.use_count(), useCount); // 使用 Google Test 的断言判断 original 的引用计数是否等于 useCount
}

//////////////////// Helper implementations for Tensor. ////////////////////

template<>
Tensor getSampleValue() { // Tensor 类型的样本值获取实现
  return at::zeros({2, 2}).to(at::kCPU); // 返回一个 2x2 的零矩阵在 CPU 上的 Tensor
}

template<>
Tensor getSampleValue2() { // Tensor 类型的另一种样本值获取实现
  return at::native::ones({2, 2}).to(at::kCPU); // 返回一个 2x2 的全一矩阵在 CPU 上的 Tensor
}

bool are_equal(const Tensor& lhs, const Tensor& rhs) { // 判断两个 Tensor 是否相等的函数实现
  if (!lhs.defined() || !rhs.defined()) { // 如果任意一个 Tensor 未定义
    return !lhs.defined() && !rhs.defined(); // 则判断两者是否都未定义
  }
  return at::native::cpu_equal(lhs, rhs); // 否则比较两个 Tensor 是否在 CPU 上相等
}

template <>
void assertBorrow( // 断言借用 Tensor 对象的具体实现
    const c10::MaybeOwned<Tensor>& mo,
    // 断言指针 `mo` 指向的对象与 `borrowedFrom` 指向的对象相同
    EXPECT_TRUE(mo->is_same(borrowedFrom));
    // 断言 `borrowedFrom` 指向的对象引用计数为 1
    EXPECT_EQ(borrowedFrom.use_count(), 1);
}

// 定义模板特化函数，用于断言 MaybeOwned<Tensor> 对象是否持有指定的 Tensor 对象，并验证引用计数
template <>
void assertOwn(
    const c10::MaybeOwned<Tensor>& mo,       // 可能拥有的 Tensor 对象
    const Tensor& original,                  // 原始的 Tensor 对象
    size_t useCount) {                       // 期望的引用计数
  EXPECT_TRUE(mo->is_same(original));        // 断言 MaybeOwned 对象中的 Tensor 与原始 Tensor 相同
  EXPECT_EQ(original.use_count(), useCount); // 断言原始 Tensor 的引用计数符合预期
}

//////////////////// IValue 的辅助实现 ////////////////////

// IValue 类型的模板特化函数，返回一个样例值
template<>
IValue getSampleValue() {
  return IValue(getSampleValue<Tensor>());   // 返回一个 Tensor 类型的 IValue 对象
}

// IValue 类型的模板特化函数，返回另一个样例值
template<>
IValue getSampleValue2() {
  return IValue("hello");                   // 返回一个包含字符串 "hello" 的 IValue 对象
}

// 比较两个 IValue 对象是否相等
bool are_equal(const IValue& lhs, const IValue& rhs) {
  if (lhs.isTensor() != rhs.isTensor()) {    // 如果左右两个 IValue 对象的类型不同，则返回不相等
    return false;
  }
  if (lhs.isTensor() && rhs.isTensor()) {    // 如果两个对象都是 Tensor 类型，则比较它们的内容是否相同
    return lhs.toTensor().equal(rhs.toTensor());
  }
  return lhs == rhs;                         // 否则直接比较它们的值是否相等
}

// 定义模板特化函数，用于断言 MaybeOwned<IValue> 对象是否持有指定的 IValue 对象，并验证引用计数
template <>
void assertBorrow(
    const c10::MaybeOwned<IValue>& mo,       // 可能借用的 IValue 对象
    const IValue& borrowedFrom) {            // 借用的原始 IValue 对象
  if (!borrowedFrom.isPtrType()) {           // 如果原始对象不是指针类型
    EXPECT_EQ(*mo, borrowedFrom);            // 断言 MaybeOwned 对象持有的值与原始值相等
  } else {                                   // 如果原始对象是指针类型
    EXPECT_EQ(mo->internalToPointer(), borrowedFrom.internalToPointer());  // 比较内部指针
    EXPECT_EQ(borrowedFrom.use_count(), 1);  // 断言原始对象的引用计数为 1
  }
}

// 定义模板特化函数，用于断言 MaybeOwned<IValue> 对象是否持有指定的 IValue 对象，并验证引用计数
template <>
void assertOwn(
    const c10::MaybeOwned<IValue>& mo,       // 可能拥有的 IValue 对象
    const IValue& original,                  // 原始的 IValue 对象
    size_t useCount) {                       // 期望的引用计数
  if (!original.isPtrType()) {               // 如果原始对象不是指针类型
    EXPECT_EQ(*mo, original);                // 断言 MaybeOwned 对象持有的值与原始值相等
  } else {                                   // 如果原始对象是指针类型
    EXPECT_EQ(mo->internalToPointer(), original.internalToPointer());  // 比较内部指针
    EXPECT_EQ(original.use_count(), useCount);  // 断言原始对象的引用计数符合预期
  }
}

// MaybeOwnedTest 类的 SetUp 方法
template <typename T>
void MaybeOwnedTest<T>::SetUp() {
  borrowFrom = getSampleValue<T>();          // 获取 T 类型的借用对象
  ownCopy = getSampleValue<T>();             // 获取 T 类型的拥有对象
  ownCopy2 = getSampleValue<T>();            // 获取另一个 T 类型的拥有对象
  borrowed = c10::MaybeOwned<T>::borrowed(borrowFrom);  // 创建借用的 MaybeOwned<T> 对象
  owned = c10::MaybeOwned<T>::owned(std::in_place, ownCopy);  // 创建拥有的 MaybeOwned<T> 对象
  owned2 = c10::MaybeOwned<T>::owned(T(ownCopy2));  // 创建另一个拥有的 MaybeOwned<T> 对象
}

// 定义类型参数为 MaybeOwnedTypes 的类型测试套件
using MaybeOwnedTypes = ::testing::Types<
  c10::intrusive_ptr<MyString>,  // MyString 类型的 intrusive_ptr
  at::Tensor,                    // Tensor 类型
  c10::IValue                    // IValue 类型
>;

// MaybeOwnedTest 类模板化测试套件
TYPED_TEST_SUITE(MaybeOwnedTest, MaybeOwnedTypes);

// MaybeOwnedTest 类的 SimpleDereferencingString 测试案例
TYPED_TEST(MaybeOwnedTest, SimpleDereferencingString) {
  assertBorrow(this->borrowed, this->borrowFrom);  // 断言借用操作的正确性
  assertOwn(this->owned, this->ownCopy);           // 断言拥有操作的正确性
  assertOwn(this->owned2, this->ownCopy2);         // 断言另一个拥有操作的正确性
}

// MaybeOwnedTest 类的 DefaultCtor 测试案例
TYPED_TEST(MaybeOwnedTest, DefaultCtor) {
  c10::MaybeOwned<TypeParam> borrowed, owned;  // 创建两个 MaybeOwned<TypeParam> 对象
  // 不要留下引用计数混乱的版本。
  this->borrowed = c10::MaybeOwned<TypeParam>();  // 清理 Fixture 中的借用版本
  this->owned = c10::MaybeOwned<TypeParam>();     // 清理 Fixture 中的拥有版本
  borrowed = c10::MaybeOwned<TypeParam>::borrowed(this->borrowFrom);  // 创建新的借用对象
  owned = c10::MaybeOwned<TypeParam>::owned(std::in_place, this->ownCopy);  // 创建新的拥有对象

  assertBorrow(borrowed, this->borrowFrom);  // 断言新借用对象的正确性
  assertOwn(owned, this->ownCopy);           // 断言新拥有对象的正确性
}

// MaybeOwnedTest 类的 CopyConstructor 测试案例
TYPED_TEST(MaybeOwnedTest, CopyConstructor) {
  auto copiedBorrowed(this->borrowed);     // 复制借用对象
  auto copiedOwned(this->owned);           // 复制拥有对象
  auto copiedOwned2(this->owned2);         // 复制另一个拥有对象

  assertBorrow(this->borrowed, this->borrowFrom);  // 断言原始借用对象的正确性
  assertBorrow(copiedBorrowed, this->borrowFrom);  // 断言复制借用对象的正确性

  assertOwn(this->owned, this->ownCopy, 3);    // 断言原始拥有对象的正确性
  assertOwn(copiedOwned, this->ownCopy, 3);    // 断言复制拥有对象的正确性
  assertOwn(this->owned2, this->ownCopy2, 3);   // 断言原始另一个拥有对象的正确性
  assertOwn(copiedOwned2, this->ownCopy2, 3);   // 断言复制另一个拥有对象的正确性
}
# 测试类型为 MaybeOwnedTest，测试移动解引用操作
TYPED_TEST(MaybeOwnedTest, MoveDereferencing) {
  # 使用 getSampleValue2<TypeParam>() 创建一个新的 owned 对象
  this->owned = c10::MaybeOwned<TypeParam>::owned(std::in_place, getSampleValue2<TypeParam>());

  # 断言移动后的 borrowed 指向的值与 getSampleValue<TypeParam>() 相等
  EXPECT_TRUE(are_equal(*std::move(this->borrowed), getSampleValue<TypeParam>()));
  # 断言移动后的 owned 指向的值与 getSampleValue2<TypeParam>() 相等
  EXPECT_TRUE(are_equal(*std::move(this->owned), getSampleValue2<TypeParam>()));

  # 断言 borrowed 未受影响
  assertBorrow(this->borrowed, this->borrowFrom);

  # 断言 owned 是一个空的 c10::intrusive_ptr 或空的 Tensor
  EXPECT_TRUE(are_equal(*this->owned, TypeParam()));
}

# 测试类型为 MaybeOwnedTest，测试移动构造函数
TYPED_TEST(MaybeOwnedTest, MoveConstructor) {
  # 移动 this->borrowed 到 movedBorrowed
  auto movedBorrowed(std::move(this->borrowed));
  # 移动 this->owned 到 movedOwned
  auto movedOwned(std::move(this->owned));
  # 移动 this->owned2 到 movedOwned2

  # 断言移动后的 borrowed 与原始的 borrowFrom 相等
  assertBorrow(movedBorrowed, this->borrowFrom);
  # 断言移动后的 owned 与原始的 ownCopy 相等
  assertOwn(movedOwned, this->ownCopy);
  # 断言移动后的 owned2 与原始的 ownCopy2 相等
  assertOwn(movedOwned2, this->ownCopy2);
}

# 测试类型为 MaybeOwnedTest，测试拷贝赋值到 owned
TYPED_TEST(MaybeOwnedTest, CopyAssignmentIntoOwned) {
  # 使用 std::in_place 创建 copiedBorrowed
  auto copiedBorrowed = c10::MaybeOwned<TypeParam>::owned(std::in_place);
  # 使用 std::in_place 创建 copiedOwned
  auto copiedOwned = c10::MaybeOwned<TypeParam>::owned(std::in_place);
  # 使用 std::in_place 创建 copiedOwned2
  auto copiedOwned2 = c10::MaybeOwned<TypeParam>::owned(std::in_place);

  # 将 this->borrowed 的值赋给 copiedBorrowed
  copiedBorrowed = this->borrowed;
  # 将 this->owned 的值赋给 copiedOwned
  copiedOwned = this->owned;
  # 将 this->owned2 的值赋给 copiedOwned2

  # 断言 borrowed 与原始的 borrowFrom 相等
  assertBorrow(this->borrowed, this->borrowFrom);
  # 断言 copiedBorrowed 与原始的 borrowFrom 相等
  assertBorrow(copiedBorrowed, this->borrowFrom);
  # 断言 owned 与原始的 ownCopy 相等，并且引用计数为 3
  assertOwn(this->owned, this->ownCopy, 3);
  # 断言 copiedOwned 与原始的 ownCopy 相等，并且引用计数为 3
  assertOwn(copiedOwned, this->ownCopy, 3);
  # 断言 owned2 与原始的 ownCopy2 相等，并且引用计数为 3
  assertOwn(this->owned2, this->ownCopy2, 3);
  # 断言 copiedOwned2 与原始的 ownCopy2 相等，并且引用计数为 3
  assertOwn(copiedOwned2, this->ownCopy2, 3);
}

# 测试类型为 MaybeOwnedTest，测试拷贝赋值到 borrowed
TYPED_TEST(MaybeOwnedTest, CopyAssignmentIntoBorrowed) {
  # 获取 getSampleValue2<TypeParam>() 的值作为 otherBorrowFrom
  auto otherBorrowFrom = getSampleValue2<TypeParam>();
  # 获取 getSampleValue2<TypeParam>() 的值作为 otherOwnCopy
  auto otherOwnCopy = getSampleValue2<TypeParam>();
  # 使用 otherBorrowFrom 创建 copiedBorrowed
  auto copiedBorrowed = c10::MaybeOwned<TypeParam>::borrowed(otherBorrowFrom);
  # 使用 otherOwnCopy 创建 copiedOwned
  auto copiedOwned = c10::MaybeOwned<TypeParam>::borrowed(otherOwnCopy);
  # 使用 otherOwnCopy 创建 copiedOwned2
  auto copiedOwned2 = c10::MaybeOwned<TypeParam>::borrowed(otherOwnCopy);

  # 将 this->borrowed 的值赋给 copiedBorrowed
  copiedBorrowed = this->borrowed;
  # 将 this->owned 的值赋给 copiedOwned
  copiedOwned = this->owned;
  # 将 this->owned2 的值赋给 copiedOwned2

  # 断言 borrowed 与原始的 borrowFrom 相等
  assertBorrow(this->borrowed, this->borrowFrom);
  # 断言 copiedBorrowed 与原始的 borrowFrom 相等
  assertBorrow(copiedBorrowed, this->borrowFrom);

  # 断言 owned 与原始的 ownCopy 相等，并且引用计数为 3
  assertOwn(this->owned, this->ownCopy, 3);
  # 断言 owned2 与原始的 ownCopy2 相等，并且引用计数为 3
  assertOwn(this->owned2, this->ownCopy2, 3);
  # 断言 copiedOwned 与原始的 ownCopy 相等，并且引用计数为 3
  assertOwn(copiedOwned, this->ownCopy, 3);
  # 断言 copiedOwned2 与原始的 ownCopy2 相等，并且引用计数为 3
  assertOwn(copiedOwned2, this->ownCopy2, 3);
}

# 测试类型为 MaybeOwnedTest，测试移动赋值到 owned
TYPED_TEST(MaybeOwnedTest, MoveAssignmentIntoOwned) {

  # 使用 std::in_place 创建 movedBorrowed
  auto movedBorrowed = c10::MaybeOwned<TypeParam>::owned(std::in_place);
  # 使用 std::in_place 创建 movedOwned
  auto movedOwned = c10::MaybeOwned<TypeParam>::owned(std::in_place);
  # 使用 std::in_place 创建 movedOwned2
  auto movedOwned2 = c10::MaybeOwned<TypeParam>::owned(std::in_place);

  # 将 this->borrowed 的值移动赋给 movedBorrowed
  movedBorrowed = std::move(this->borrowed);
  # 将 this->owned 的值移动赋给 movedOwned
  movedOwned = std::move(this->owned);
  # 将 this->owned2 的值移动赋给 movedOwned2

  # 断言 movedBorrowed 与原始的 borrowFrom 相等
  assertBorrow(movedBorrowed, this->borrowFrom);
  # 断言 movedOwned 与原始的 ownCopy 相等
  assertOwn(movedOwned, this->ownCopy);
  # 断言 movedOwned2 与原始的 ownCopy2 相等
  assertOwn(movedOwned2, this->ownCopy2);
}
# 定义一个测试函数，测试 MaybeOwned 类型的 Move 赋值操作到 borrowed 和 owned 对象的行为
TYPED_TEST(MaybeOwnedTest, MoveAssignmentIntoBorrowed) {
  # 获取一个 TypeParam 类型的样本值 y
  auto y = getSampleValue2<TypeParam>();
  # 使用 y 创建一个 MaybeOwned 对象 movedBorrowed，以 borrowed 方式借用 y
  auto movedBorrowed = c10::MaybeOwned<TypeParam>::borrowed(y);
  # 使用 y 创建一个 MaybeOwned 对象 movedOwned，以 borrowed 方式借用 y
  auto movedOwned = c10::MaybeOwned<TypeParam>::borrowed(y);
  # 使用 y 创建一个 MaybeOwned 对象 movedOwned2，以 borrowed 方式借用 y
  auto movedOwned2 = c10::MaybeOwned<TypeParam>::borrowed(y);

  # 将 this->borrowed 的所有权移动到 movedBorrowed
  movedBorrowed = std::move(this->borrowed);
  # 将 this->owned 的所有权移动到 movedOwned
  movedOwned = std::move(this->owned);
  # 将 this->owned2 的所有权移动到 movedOwned2
  movedOwned2 = std::move(this->owned2);

  # 断言 movedBorrowed 中的借用对象是否与 this->borrowFrom 相同
  assertBorrow(movedBorrowed, this->borrowFrom);
  # 断言 movedOwned 中的拥有对象是否与 this->ownCopy 相同
  assertOwn(movedOwned, this->ownCopy);
  # 断言 movedOwned2 中的拥有对象是否与 this->ownCopy2 相同
  assertOwn(movedOwned2, this->ownCopy2);
}

# 定义一个测试函数，测试 MaybeOwned 类型的自我赋值行为
TYPED_TEST(MaybeOwnedTest, SelfAssignment) {
  # 将 this->borrowed 赋值给自身
  this->borrowed = this->borrowed;
  # 将 this->owned 赋值给自身
  this->owned = this->owned;
  # 将 this->owned2 赋值给自身
  this->owned2 = this->owned2;

  # 断言 this->borrowed 中的借用对象是否与 this->borrowFrom 相同
  assertBorrow(this->borrowed, this->borrowFrom);
  # 断言 this->owned 中的拥有对象是否与 this->ownCopy 相同
  assertOwn(this->owned, this->ownCopy);
  # 断言 this->owned2 中的拥有对象是否与 this->ownCopy2 相同
  assertOwn(this->owned2, this->ownCopy2);
}

} // namespace
```