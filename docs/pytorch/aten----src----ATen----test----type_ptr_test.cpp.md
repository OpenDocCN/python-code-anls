# `.\pytorch\aten\src\ATen\test\type_ptr_test.cpp`

```
#include <gtest/gtest.h>
#include <ATen/core/type_ptr.h>
#include <ATen/core/jit_type.h>

using c10::SingletonOrSharedTypePtr;  // 使用 c10 命名空间中的 SingletonOrSharedTypePtr 类

namespace {

TEST(SingletonOrSharedTypePtr, Empty) {  // 定义名为 SingletonOrSharedTypePtr 的测试套件，测试空情况
  SingletonOrSharedTypePtr<int> empty;  // 创建一个空的 SingletonOrSharedTypePtr 对象，存储 int 类型
  EXPECT_TRUE(!empty);  // 断言 empty 为真
  EXPECT_EQ(nullptr, empty.get());  // 断言 empty 的指针为空
  EXPECT_EQ(empty, nullptr);  // 断言 empty 等于 nullptr
  std::shared_ptr<int> emptyShared;  // 创建一个空的 std::shared_ptr<int> 对象
  EXPECT_EQ(emptyShared, empty);  // 断言两个空指针对象相等
}

TEST(SingletonOrSharedTypePtr, NonEmpty) {  // 测试非空情况
  auto shared = std::make_shared<int>(42);  // 创建一个 shared_ptr<int> 对象，指向值为 42 的整数
  SingletonOrSharedTypePtr<int> p(shared);  // 创建一个 SingletonOrSharedTypePtr 对象，指向上述 shared_ptr<int>
  EXPECT_EQ(42, *shared);  // 断言 shared 指向的值为 42
  EXPECT_TRUE(shared);  // 断言 shared 为真
  EXPECT_EQ(42, *p);  // 断言 SingletonOrSharedTypePtr 对象 p 指向的值为 42
  EXPECT_TRUE(p);  // 断言 p 为真
  EXPECT_NE(nullptr, p.get());  // 断言 p 的指针非空
  EXPECT_NE(p, nullptr);  // 断言 p 不等于 nullptr
  EXPECT_EQ(shared, p);  // 断言 p 指向与 shared 相同的对象
  EXPECT_EQ(shared.get(), p.get());  // 断言 p 的原始指针与 shared 的原始指针相同
}

TEST(SingletonOrSharedTypePtr, Comparison) {  // 测试比较操作
  SingletonOrSharedTypePtr<int> empty;  // 创建一个空的 SingletonOrSharedTypePtr 对象
  auto shared = std::make_shared<int>(42);  // 创建一个 shared_ptr<int> 对象，指向值为 42 的整数
  SingletonOrSharedTypePtr<int> p(shared);  // 创建一个 SingletonOrSharedTypePtr 对象，指向上述 shared_ptr<int>
  auto shared2 = std::make_shared<int>(3);  // 创建另一个 shared_ptr<int> 对象，指向值为 3 的整数
  SingletonOrSharedTypePtr<int> p2(shared2);  // 创建一个 SingletonOrSharedTypePtr 对象，指向上述 shared_ptr<int>

  EXPECT_NE(empty, p);  // 断言空对象和非空对象 p 不相等
  EXPECT_NE(p, p2);  // 断言 p 和 p2 不相等
}

TEST(SingletonOrSharedTypePtr, SingletonComparison) {  // 测试单例对象的比较
  EXPECT_NE(c10::StringType::get(), c10::NoneType::get());  // 断言字符串类型和空类型不相等
  EXPECT_NE(c10::StringType::get(), c10::DeviceObjType::get());  // 断言字符串类型和设备对象类型不相等
  EXPECT_NE(c10::NoneType::get(), c10::DeviceObjType::get());  // 断言空类型和设备对象类型不相等

  c10::TypePtr type = c10::NoneType::get();  // 创建一个 c10::TypePtr 对象，指向空类型
  EXPECT_NE(type, c10::StringType::get());  // 断言 type 和字符串类型不相等
  EXPECT_NE(type, c10::DeviceObjType::get());  // 断言 type 和设备对象类型不相等
}

} // namespace
```