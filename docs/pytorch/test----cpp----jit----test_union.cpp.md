# `.\pytorch\test\cpp\jit\test_union.cpp`

```py
#include <gtest/gtest.h>  // 引入 Google Test 框架的头文件

#include <ATen/core/jit_type.h>  // 引入 PyTorch 的 JIT 类型定义头文件
#include <test/cpp/jit/test_utils.h>  // 引入用于测试的实用函数头文件
#include <torch/csrc/jit/ir/ir.h>  // 引入 PyTorch JIT 的 IR 相关头文件

namespace torch {
namespace jit {

class UnionTypeTest : public ::testing::Test {  // 定义测试类 UnionTypeTest，继承自 Google Test 的 Test 类
 public:
  const TypePtr none = NoneType::get();  // 声明一个 None 类型的 TypePtr 变量

  const TypePtr l1 = ListType::ofStrings();  // 声明一个 List[str] 类型的 TypePtr 变量

  const TypePtr opt1 = OptionalType::create(IntType::get());  // 声明一个 Optional[int] 类型的 TypePtr 变量

  const TypePtr opt2 = OptionalType::create(FloatType::get());  // 声明一个 Optional[float] 类型的 TypePtr 变量

  const TypePtr opt3 = OptionalType::create(ListType::ofStrings());  // 声明一个 Optional[List[str]] 类型的 TypePtr 变量

  const TypePtr tup1 = TupleType::create({OptionalType::create(IntType::get()), IntType::get()});
  // 声明一个 Tuple[Optional[int], int] 类型的 TypePtr 变量

  const TypePtr tup2 = TupleType::create({IntType::get(), IntType::get()});
  // 声明一个 Tuple[int, int] 类型的 TypePtr 变量

  bool hasType(UnionTypePtr u, TypePtr t) {  // 定义一个成员函数 hasType，用于判断 UnionTypePtr 中是否包含某种类型
    auto res = std::find(u->getTypes().begin(), u->getTypes().end(), t);  // 在 UnionTypePtr 中查找指定类型
    return res != u->getTypes().end();  // 返回是否找到指定类型的结果
  }
};

TEST_F(UnionTypeTest, UnionOperatorEquals) {
  const UnionTypePtr u1 = UnionType::create({l1, tup2, StringType::get()});
  // 创建一个 UnionTypePtr，包含 List[str]、Tuple[int, int] 和 StringType 类型

  const TypePtr l1_ = ListType::ofStrings();  // 获取 List[str] 类型的 TypePtr 变量
  const TypePtr tup2_ = TupleType::create({IntType::get(), IntType::get()});
  // 获取 Tuple[int, int] 类型的 TypePtr 变量
  const UnionTypePtr u2 = UnionType::create({l1_, tup2_, StringType::get()});
  // 创建另一个 UnionTypePtr，包含 List[str]、Tuple[int, int] 和 StringType 类型

  ASSERT_TRUE(*u1 == *u2);  // 断言两个 UnionTypePtr 是否相等
}

TEST_F(UnionTypeTest, UnionCreate_OptionalT1AndOptionalT2) {
  const UnionTypePtr u = UnionType::create({opt1, opt2});
  // 创建一个 UnionTypePtr，包含 Optional[int]、Optional[float] 和 None 类型

  ASSERT_EQ(u->getTypes().size(), 3);  // 断言 UnionTypePtr 中包含的类型数量为 3
  ASSERT_TRUE(UnionTypeTest::hasType(u, IntType::get()));  // 断言 UnionTypePtr 中包含 IntType 类型
  ASSERT_TRUE(UnionTypeTest::hasType(u, FloatType::get()));  // 断言 UnionTypePtr 中包含 FloatType 类型
  ASSERT_TRUE(UnionTypeTest::hasType(u, NoneType::get()));  // 断言 UnionTypePtr 中包含 NoneType 类型
}

TEST_F(UnionTypeTest, UnionCreate_OptionalTAndT) {
  const UnionTypePtr u = UnionType::create({opt1, IntType::get()});
  // 创建一个 UnionTypePtr，包含 Optional[int] 和 IntType 类型

  ASSERT_EQ(u->getTypes().size(), 2);  // 断言 UnionTypePtr 中包含的类型数量为 2
  ASSERT_TRUE(UnionTypeTest::hasType(u, IntType::get()));  // 断言 UnionTypePtr 中包含 IntType 类型
  ASSERT_TRUE(UnionTypeTest::hasType(u, NoneType::get()));  // 断言 UnionTypePtr 中包含 NoneType 类型
}

TEST_F(UnionTypeTest, UnionCreate_TupleWithSubtypingRelationship) {
  const UnionTypePtr u = UnionType::create({StringType::get(), tup1, tup2});
  // 创建一个 UnionTypePtr，包含 StringType、Tuple[Optional[int], int] 和 Tuple[int, int] 类型

  ASSERT_EQ(u->getTypes().size(), 2);  // 断言 UnionTypePtr 中包含的类型数量为 2
  ASSERT_TRUE(UnionTypeTest::hasType(u, StringType::get()));  // 断言 UnionTypePtr 中包含 StringType 类型
  ASSERT_TRUE(UnionTypeTest::hasType(u, tup1));  // 断言 UnionTypePtr 中包含 tup1 类型
}

TEST_F(UnionTypeTest, UnionCreate_ContainerTAndT) {
  const UnionTypePtr u = UnionType::create({l1, StringType::get()});
  // 创建一个 UnionTypePtr，包含 List[str] 和 StringType 类型

  ASSERT_EQ(u->getTypes().size(), 2);  // 断言 UnionTypePtr 中包含的类型数量为 2
  ASSERT_TRUE(UnionTypeTest::hasType(u, StringType::get()));  // 断言 UnionTypePtr 中包含 StringType 类型
  ASSERT_TRUE(UnionTypeTest::hasType(u, ListType::ofStrings()));  // 断言 UnionTypePtr 中包含 List[str] 类型
}


这段代码是一个 C++ 的单元测试代码，用于测试 PyTorch JIT 中的 UnionType 类的功能。代码中通过 Google Test 框架编写了多个测试用例，测试 UnionType 类的创建、比较以及包含关系。
TEST_F(UnionTypeTest, UnionCreate_OptionalContainerTAndContainerTAndT) {
  // 创建一个联合类型对象，表示 Union[List[str], None, str]
  const UnionTypePtr u = UnionType::create({l1, opt3, StringType::get()});

  // 确保联合类型包含三种类型
  ASSERT_EQ(u->getTypes().size(), 3);
  // 确保联合类型包含 StringType 类型
  ASSERT_TRUE(UnionTypeTest::hasType(u, StringType::get()));
  // 确保联合类型包含 ListType::ofStrings() 类型
  ASSERT_TRUE(UnionTypeTest::hasType(u, ListType::ofStrings()));
}

TEST_F(UnionTypeTest, Subtyping_NumberType) {
  // 创建一个联合类型对象，表示 Union[int, float, Complex]
  const UnionTypePtr union1 =
      UnionType::create({IntType::get(), FloatType::get(), ComplexType::get()});

  // 创建一个联合类型对象，表示 Union[int, float, Complex, None]
  const UnionTypePtr union2 = UnionType::create(
      {IntType::get(), FloatType::get(), ComplexType::get(), NoneType::get()});

  // 获取 NumberType 类型对象
  const NumberTypePtr num = NumberType::get();

  // 测试 NumberType 是否是 union1 的子类型
  ASSERT_TRUE(num->isSubtypeOf(*union1));
  // 测试 union1 是否是 NumberType 的子类型
  ASSERT_TRUE(union1->isSubtypeOf(*num));
  // 测试 NumberType 和 union1 是否相等
  ASSERT_TRUE(*num == *union1);

  // 测试 NumberType 是否是 union2 的子类型
  ASSERT_TRUE(num->isSubtypeOf(*union2));
  // 测试 union2 是否是 NumberType 的子类型
  ASSERT_FALSE(union2->isSubtypeOf(*num));
  // 测试 NumberType 和 union2 是否相等
  ASSERT_FALSE(*num == *union2);
}

TEST_F(UnionTypeTest, Subtyping_OptionalType) {
  // 创建一个联合类型对象，表示 Union[int, None]
  const UnionTypePtr union1 =
      UnionType::create({IntType::get(), NoneType::get()});

  // 创建一个联合类型对象，表示 Union[int, str, None]
  const UnionTypePtr union2 =
      UnionType::create({IntType::get(), StringType::get(), NoneType::get()});

  // 创建一个联合类型对象，表示 Union[int, str, List[str]]
  const UnionTypePtr union3 = UnionType::create(
      {IntType::get(), StringType::get(), ListType::ofStrings()});

  // 获取 NoneType 类型对象
  const TypePtr none = NoneType::get();
  // 获取 OptionalType<int> 类型对象
  const TypePtr opt1 = OptionalType::create(IntType::get());

  // 测试 NoneType 是否是 OptionalType<int> 的子类型
  ASSERT_TRUE(none->isSubtypeOf(opt1));
  // 测试 NoneType 是否是 union1 的子类型
  ASSERT_TRUE(none->isSubtypeOf(union1));
  // 测试 NoneType 是否是 union2 的子类型
  ASSERT_TRUE(none->isSubtypeOf(union2));
  // 测试 NoneType 是否是 union3 的子类型（应该为假）
  ASSERT_FALSE(none->isSubtypeOf(union3));

  // 测试 OptionalType<int> 是否是 NoneType 的子类型（应该为假）
  ASSERT_FALSE(opt1->isSubtypeOf(none));
  // 测试 OptionalType<int> 是否是 union1 的子类型
  ASSERT_TRUE(opt1->isSubtypeOf(union1));
  // 测试 OptionalType<int> 是否是 union2 的子类型
  ASSERT_TRUE(opt1->isSubtypeOf(union2));
  // 测试 OptionalType<int> 是否是 union3 的子类型（应该为假）
  ASSERT_FALSE(opt1->isSubtypeOf(union3));

  // 测试 union1 是否是 NoneType 的子类型（应该为假）
  ASSERT_FALSE(union1->isSubtypeOf(none));
  // 测试 union1 是否是 OptionalType<int> 的子类型
  ASSERT_TRUE(union1->isSubtypeOf(opt1));
  // 测试 union1 是否是 union2 的子类型
  ASSERT_TRUE(union1->isSubtypeOf(union2));
  // 测试 union1 是否是 union3 的子类型（应该为假）
  ASSERT_FALSE(union1->isSubtypeOf(union3));

  // 测试 union2 是否是 union1 的子类型（应该为假）
  ASSERT_FALSE(union2->isSubtypeOf(union1));
}
```