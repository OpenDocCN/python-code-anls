# `.\pytorch\c10\test\util\typeid_test.cpp`

```
#include <c10/util/typeid.h>  // 包含 C10 库的 typeid.h 文件，用于类型相关操作
#include <gtest/gtest.h>      // 包含 Google Test 框架的头文件，用于单元测试

using std::string;            // 使用 std 命名空间中的 string 类

namespace caffe2 {
namespace {

class TypeMetaTestFoo {};     // 定义一个空的测试类 TypeMetaTestFoo
class TypeMetaTestBar {};     // 定义一个空的测试类 TypeMetaTestBar
} // namespace

CAFFE_KNOWN_TYPE_NOEXPORT(TypeMetaTestFoo);  // 声明 TypeMetaTestFoo 类型为已知类型
CAFFE_KNOWN_TYPE_NOEXPORT(TypeMetaTestBar);  // 声明 TypeMetaTestBar 类型为已知类型

namespace {

TEST(TypeMetaTest, TypeMetaStatic) {
  EXPECT_EQ(TypeMeta::ItemSize<int>(), sizeof(int));  // 断言 int 类型的大小等于 sizeof(int)
  EXPECT_EQ(TypeMeta::ItemSize<float>(), sizeof(float));  // 断言 float 类型的大小等于 sizeof(float)
  EXPECT_EQ(TypeMeta::ItemSize<TypeMetaTestFoo>(), sizeof(TypeMetaTestFoo));  // 断言 TypeMetaTestFoo 类型的大小等于 sizeof(TypeMetaTestFoo)
  EXPECT_EQ(TypeMeta::ItemSize<TypeMetaTestBar>(), sizeof(TypeMetaTestBar));  // 断言 TypeMetaTestBar 类型的大小等于 sizeof(TypeMetaTestBar)
  EXPECT_NE(TypeMeta::Id<int>(), TypeMeta::Id<float>());  // 断言 int 类型和 float 类型的 ID 不相等
  EXPECT_NE(TypeMeta::Id<int>(), TypeMeta::Id<TypeMetaTestFoo>());  // 断言 int 类型和 TypeMetaTestFoo 类型的 ID 不相等
  EXPECT_NE(TypeMeta::Id<TypeMetaTestFoo>(), TypeMeta::Id<TypeMetaTestBar>());  // 断言 TypeMetaTestFoo 类型和 TypeMetaTestBar 类型的 ID 不相等
  EXPECT_EQ(TypeMeta::Id<int>(), TypeMeta::Id<int>());  // 断言 int 类型的 ID 等于 int 类型的 ID
  EXPECT_EQ(TypeMeta::Id<TypeMetaTestFoo>(), TypeMeta::Id<TypeMetaTestFoo>());  // 断言 TypeMetaTestFoo 类型的 ID 等于 TypeMetaTestFoo 类型的 ID
}

TEST(TypeMetaTest, Names) {
  TypeMeta null_meta;  // 创建一个空的 TypeMeta 对象
  EXPECT_EQ("nullptr (uninitialized)", null_meta.name());  // 断言空的 TypeMeta 对象的名称为 "nullptr (uninitialized)"
  TypeMeta int_meta = TypeMeta::Make<int>();  // 创建一个 int 类型的 TypeMeta 对象
  EXPECT_EQ("int", int_meta.name());  // 断言 int 类型的 TypeMeta 对象的名称为 "int"
  TypeMeta string_meta = TypeMeta::Make<string>();  // 创建一个 string 类型的 TypeMeta 对象
  EXPECT_TRUE(c10::string_view::npos != string_meta.name().find("string"));  // 断言 string 类型的 TypeMeta 对象的名称中包含 "string"
}

TEST(TypeMetaTest, TypeMeta) {
  TypeMeta int_meta = TypeMeta::Make<int>();  // 创建一个 int 类型的 TypeMeta 对象
  TypeMeta float_meta = TypeMeta::Make<float>();  // 创建一个 float 类型的 TypeMeta 对象
  TypeMeta foo_meta = TypeMeta::Make<TypeMetaTestFoo>();  // 创建一个 TypeMetaTestFoo 类型的 TypeMeta 对象
  TypeMeta bar_meta = TypeMeta::Make<TypeMetaTestBar>();  // 创建一个 TypeMetaTestBar 类型的 TypeMeta 对象

  TypeMeta another_int_meta = TypeMeta::Make<int>();  // 创建另一个 int 类型的 TypeMeta 对象
  TypeMeta another_foo_meta = TypeMeta::Make<TypeMetaTestFoo>();  // 创建另一个 TypeMetaTestFoo 类型的 TypeMeta 对象

  EXPECT_EQ(int_meta, another_int_meta);  // 断言两个 int 类型的 TypeMeta 对象相等
  EXPECT_EQ(foo_meta, another_foo_meta);  // 断言两个 TypeMetaTestFoo 类型的 TypeMeta 对象相等
  EXPECT_NE(int_meta, float_meta);  // 断言 int 类型的 TypeMeta 对象不等于 float 类型的 TypeMeta 对象
  EXPECT_NE(int_meta, foo_meta);  // 断言 int 类型的 TypeMeta 对象不等于 TypeMetaTestFoo 类型的 TypeMeta 对象
  EXPECT_NE(foo_meta, bar_meta);  // 断言 TypeMetaTestFoo 类型的 TypeMeta 对象不等于 TypeMetaTestBar 类型的 TypeMeta 对象
  EXPECT_TRUE(int_meta.Match<int>());  // 断言 int 类型的 TypeMeta 对象匹配 int 类型
  EXPECT_TRUE(foo_meta.Match<TypeMetaTestFoo>());  // 断言 TypeMetaTestFoo 类型的 TypeMeta 对象匹配 TypeMetaTestFoo 类型
  EXPECT_FALSE(int_meta.Match<float>());  // 断言 int 类型的 TypeMeta 对象不匹配 float 类型
  EXPECT_FALSE(int_meta.Match<TypeMetaTestFoo>());  // 断言 int 类型的 TypeMeta 对象不匹配 TypeMetaTestFoo 类型
  EXPECT_FALSE(foo_meta.Match<int>());  // 断言 TypeMetaTestFoo 类型的 TypeMeta 对象不匹配 int 类型
  EXPECT_FALSE(foo_meta.Match<TypeMetaTestBar>());  // 断言 TypeMetaTestFoo 类型的 TypeMeta 对象不匹配 TypeMetaTestBar 类型
  EXPECT_EQ(int_meta.id(), TypeMeta::Id<int>());  // 断言 int 类型的 TypeMeta 对象的 ID 等于 int 类型的 ID
  EXPECT_EQ(float_meta.id(), TypeMeta::Id<float>());  // 断言 float 类型的 TypeMeta 对象的 ID 等于 float 类型的 ID
  EXPECT_EQ(foo_meta.id(), TypeMeta::Id<TypeMetaTestFoo>());  // 断言 TypeMetaTestFoo 类型的 TypeMeta 对象的 ID 等于 TypeMetaTestFoo 类型的 ID
  EXPECT_EQ(bar_meta.id(), TypeMeta::Id<TypeMetaTestBar>());  // 断言 TypeMetaTestBar 类型的 TypeMeta 对象的 ID 等于 TypeMetaTestBar 类型的 ID
  EXPECT_EQ(int_meta.itemsize(), TypeMeta::ItemSize<int>());  // 断言 int 类型的 TypeMeta 对象的大小等于 int 类型的大小
  EXPECT_EQ(float_meta.itemsize(), TypeMeta::ItemSize<float>());  // 断言 float 类型的 TypeMeta 对象的大小等于 float 类型的大小
  EXPECT_EQ(foo_meta.itemsize(), TypeMeta::ItemSize<TypeMetaTestFoo>());  // 断言 TypeMetaTestFoo 类型的 TypeMeta 对象的大小等于 TypeMetaTestFoo 类型的大小
  EXPECT_EQ(bar_meta.itemsize(), TypeMeta::ItemSize<TypeMetaTestBar>());  // 断言 TypeMetaTestBar 类型的 TypeMeta 对象的大小等于 TypeMetaTestBar 类型的大小
  EXPECT_EQ(int_meta.name(), "int");  // 断言 int 类型的 TypeMeta 对象的名称为 "int"
  EXPECT_EQ(float_meta.name(), "float");  // 断言 float 类型的 TypeMeta 对象的名称为 "float"
  EXPECT_NE(foo_meta.name().find("TypeMetaTestFoo"), c10::string_view::npos);  // 断言 TypeMetaTestFoo 类型的 TypeMeta 对象的名称中包含 "TypeMetaTestFoo"
  EXPECT_NE(bar_meta.name().find("TypeMetaTestBar"), c10::string_view::npos);  // 断言 TypeMetaTestBar 类型的 TypeMeta 对象的名称中包含 "TypeMetaTestBar"
}
// 定义一个允许赋值的类 ClassAllowAssignment
class ClassAllowAssignment {
 public:
  // 默认构造函数，初始化 x 为 42
  ClassAllowAssignment() : x(42) {}
  // 允许使用默认的拷贝构造函数
  ClassAllowAssignment(const ClassAllowAssignment& src) = default;
  // 允许使用默认的赋值运算符重载
  ClassAllowAssignment& operator=(const ClassAllowAssignment& src) = default;
  // 整数类型成员变量 x
  int x;
};

// 定义一个禁止赋值的类 ClassNoAssignment
class ClassNoAssignment {
 public:
  // 默认构造函数，初始化 x 为 42
  ClassNoAssignment() : x(42) {}
  // 删除拷贝构造函数，禁止对象的拷贝
  ClassNoAssignment(const ClassNoAssignment& src) = delete;
  // 删除赋值运算符重载，禁止对象的赋值
  ClassNoAssignment& operator=(const ClassNoAssignment& src) = delete;
  // 整数类型成员变量 x
  int x;
};
} // namespace

// 使用 CAFFE_KNOWN_TYPE_NOEXPORT 宏注册 ClassAllowAssignment 类型
CAFFE_KNOWN_TYPE_NOEXPORT(ClassAllowAssignment);
// 使用 CAFFE_KNOWN_TYPE_NOEXPORT 宏注册 ClassNoAssignment 类型
CAFFE_KNOWN_TYPE_NOEXPORT(ClassNoAssignment);

namespace {

// 定义 TypeMetaTest 测试套件中的 CtorDtorAndCopy 测试用例
TEST(TypeMetaTest, CtorDtorAndCopy) {
  // 创建基本类型 TypeMeta 对象，指定为 int 类型
  TypeMeta fundamental_meta = TypeMeta::Make<int>();
  // 验证 fundamental_meta 的 placementNew 方法为空
  EXPECT_EQ(fundamental_meta.placementNew(), nullptr);
  // 验证 fundamental_meta 的 placementDelete 方法为空
  EXPECT_EQ(fundamental_meta.placementDelete(), nullptr);
  // 验证 fundamental_meta 的 copy 方法为空
  EXPECT_EQ(fundamental_meta.copy(), nullptr);

  // 创建 TypeMeta 对象，指定为 ClassAllowAssignment 类型
  TypeMeta meta_a = TypeMeta::Make<ClassAllowAssignment>();
  // 验证 meta_a 的 placementNew 方法不为空
  EXPECT_TRUE(meta_a.placementNew() != nullptr);
  // 验证 meta_a 的 placementDelete 方法不为空
  EXPECT_TRUE(meta_a.placementDelete() != nullptr);
  // 验证 meta_a 的 copy 方法不为空
  EXPECT_TRUE(meta_a.copy() != nullptr);
  
  // 创建 ClassAllowAssignment 对象 src，并将其 x 成员变量设置为 10
  ClassAllowAssignment src;
  src.x = 10;
  // 创建 ClassAllowAssignment 对象 dst
  ClassAllowAssignment dst;
  // 验证 dst 的 x 成员变量初始值为 42
  EXPECT_EQ(dst.x, 42);
  // 调用 meta_a 的 copy 方法，将 src 对象复制到 dst 对象
  meta_a.copy()(&src, &dst, 1);
  // 验证复制后 dst 的 x 成员变量值为 10
  EXPECT_EQ(dst.x, 10);

  // 创建 TypeMeta 对象，指定为 ClassNoAssignment 类型
  TypeMeta meta_b = TypeMeta::Make<ClassNoAssignment>();

  // 验证 meta_b 的 placementNew 方法不为空
  EXPECT_TRUE(meta_b.placementNew() != nullptr);
  // 验证 meta_b 的 placementDelete 方法不为空
  EXPECT_TRUE(meta_b.placementDelete() != nullptr);

#ifndef __clang__
  // 对于非 Clang 编译器，验证 meta_b 的 copy 方法指向 _CopyNotAllowed<ClassNoAssignment> 函数
  EXPECT_EQ(meta_b.copy(), &(detail::_CopyNotAllowed<ClassNoAssignment>));
#endif
}

// 定义 TypeMetaTest 测试套件中的 Float16IsNotUint16 测试用例
TEST(TypeMetaTest, Float16IsNotUint16) {
  // 验证 uint16_t 和 at::Half 类型的 TypeMeta 的 ID 不相等
  EXPECT_NE(TypeMeta::Id<uint16_t>(), TypeMeta::Id<at::Half>());
}

} // namespace
} // namespace caffe2
```