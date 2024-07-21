# `.\pytorch\c10\test\core\impl\InlineDeviceGuard_test.cpp`

```
#include <gtest/gtest.h>  // 引入 Google Test 框架的头文件
#include <initializer_list>  // 引入初始化列表的头文件

#include <c10/core/impl/FakeGuardImpl.h>  // 引入 FakeGuardImpl 实现的头文件
#include <c10/core/impl/InlineDeviceGuard.h>  // 引入 InlineDeviceGuard 实现的头文件

using namespace c10;  // 使用 c10 命名空间
using namespace c10::impl;  // 使用 c10::impl 命名空间

constexpr auto TestDeviceType = DeviceType::CUDA;  // 定义 TestDeviceType 为 CUDA 设备类型
using TestGuardImpl = FakeGuardImpl<TestDeviceType>;  // 使用 FakeGuardImpl 模板生成 TestGuardImpl 类型

static Device dev(DeviceIndex index) {  // 静态函数 dev，返回指定设备类型的 Device 对象
  return Device(TestDeviceType, index);
}

// -- InlineDeviceGuard -------------------------------------------------------

using TestGuard = InlineDeviceGuard<TestGuardImpl>;  // 使用 InlineDeviceGuard 模板生成 TestGuard 类型

TEST(InlineDeviceGuard, Constructor) {  // 定义测试用例 InlineDeviceGuard.Constructor
  for (DeviceIndex i : std::initializer_list<DeviceIndex>{-1, 0, 1}) {  // 遍历初始化列表中的设备索引
    DeviceIndex init_i = 0;  // 初始化设备索引为 0
    TestGuardImpl::setDeviceIndex(init_i);  // 设置 TestGuardImpl 的设备索引为 init_i
    auto test_body = [&](TestGuard& g) -> void {  // 定义测试体的 Lambda 函数
      ASSERT_EQ(g.original_device(), dev(init_i));  // 断言 g 的原始设备与初始化设备一致
      ASSERT_EQ(g.current_device(), dev(i == -1 ? init_i : i));  // 断言 g 的当前设备与循环中的设备索引一致
      ASSERT_EQ(TestGuardImpl::getDeviceIndex(), i == -1 ? init_i : i);  // 断言 TestGuardImpl 的设备索引正确设置
      // 测试未加括号的设备索引写入
      TestGuardImpl::setDeviceIndex(4);
    };
    {
      // Index 构造函数
      TestGuard g(i);  // 创建 TestGuard 对象 g
      test_body(g);  // 调用测试体函数
    }
    ASSERT_EQ(TestGuardImpl::getDeviceIndex(), init_i);  // 断言测试完成后设备索引被重置为 init_i
    {
      // Device 构造函数
      TestGuard g(dev(i));  // 用设备对象创建 TestGuard 对象 g
      test_body(g);  // 调用测试体函数
    }
    ASSERT_EQ(TestGuardImpl::getDeviceIndex(), init_i);  // 断言测试完成后设备索引被重置为 init_i
    /*
    {
      // Optional 构造函数
      TestGuard g(make_optional(dev(i)));  // 使用可选设备对象创建 TestGuard 对象 g
      test_body(g);  // 调用测试体函数
    }
    ASSERT_EQ(TestGuardImpl::getDeviceIndex(), init_i);  // 断言测试完成后设备索引被重置为 init_i
    */
  }
}

TEST(InlineDeviceGuard, ConstructorError) {  // 定义测试用例 InlineDeviceGuard.ConstructorError
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  EXPECT_ANY_THROW(InlineDeviceGuard<FakeGuardImpl<DeviceType::CUDA>> g(
      Device(DeviceType::HIP, 1)));  // 断言使用不支持的设备类型抛出异常
}

TEST(InlineDeviceGuard, SetDevice) {  // 定义测试用例 InlineDeviceGuard.SetDevice
  DeviceIndex init_i = 0;  // 初始化设备索引为 0
  TestGuardImpl::setDeviceIndex(init_i);  // 设置 TestGuardImpl 的设备索引为 init_i
  DeviceIndex i = 1;  // 设定设备索引为 1
  TestGuard g(i);  // 创建 TestGuard 对象 g
  DeviceIndex i2 = 2;  // 设定第二个设备索引为 2
  g.set_device(dev(i2));  // 设置 g 的设备为设备索引 i2
  ASSERT_EQ(g.original_device(), dev(init_i));  // 断言 g 的原始设备与初始化设备一致
  ASSERT_EQ(g.current_device(), dev(i2));  // 断言 g 的当前设备与设备索引 i2 一致
  ASSERT_EQ(TestGuardImpl::getDeviceIndex(), i2);  // 断言 TestGuardImpl 的设备索引设置为 i2
  g.set_device(dev(i2));  // 再次设置 g 的设备为设备索引 i2
  ASSERT_EQ(g.original_device(), dev(init_i));  // 断言 g 的原始设备与初始化设备一致
  ASSERT_EQ(g.current_device(), dev(i2));  // 断言 g 的当前设备与设备索引 i2 一致
  ASSERT_EQ(TestGuardImpl::getDeviceIndex(), i2);  // 断言 TestGuardImpl 的设备索引设置为 i2
}

TEST(InlineDeviceGuard, ResetDevice) {  // 定义测试用例 InlineDeviceGuard.ResetDevice
  DeviceIndex init_i = 0;  // 初始化设备索引为 0
  TestGuardImpl::setDeviceIndex(init_i);  // 设置 TestGuardImpl 的设备索引为 init_i
  DeviceIndex i = 1;  // 设定设备索引为 1
  TestGuard g(i);  // 创建 TestGuard 对象 g
  DeviceIndex i2 = 2;  // 设定第二个设备索引为 2
  g.reset_device(dev(i2));  // 重置 g 的设备为设备索引 i2
  ASSERT_EQ(g.original_device(), dev(init_i));  // 断言 g 的原始设备与初始化设备一致
  ASSERT_EQ(g.current_device(), dev(i2));  // 断言 g 的当前设备与设备索引 i2 一致
  ASSERT_EQ(TestGuardImpl::getDeviceIndex(), i2);  // 断言 TestGuardImpl 的设备索引设置为 i2
  g.reset_device(dev(i2));  // 再次重置 g 的设备为设备索引 i2
  ASSERT_EQ(g.original_device(), dev(init_i));  // 断言 g 的原始设备与初始化设备一致
  ASSERT_EQ(g.current_device(), dev(i2));  // 断言 g 的当前设备与设备索引 i2 一致
  ASSERT_EQ(TestGuardImpl::getDeviceIndex(), i2);  // 断言 TestGuardImpl 的设备索引设置为 i2
}
// 定义一个测试用例 InlineDeviceGuard.SetIndex
TEST(InlineDeviceGuard, SetIndex) {
  // 初始化设备索引为 0
  DeviceIndex init_i = 0;
  // 设置测试宏的设备索引为 init_i
  TestGuardImpl::setDeviceIndex(init_i);
  // 创建设备索引为 1 的 TestGuard 对象
  DeviceIndex i = 1;
  TestGuard g(i);
  // 设置 g 的索引为 2
  DeviceIndex i2 = 2;
  g.set_index(i2);
  // 断言初始设备与 g 的原始设备索引相等
  ASSERT_EQ(g.original_device(), dev(init_i));
  // 断言当前设备与 g 的当前设备索引相等
  ASSERT_EQ(g.current_device(), dev(i2));
  // 断言 TestGuardImpl 中的设备索引为 i2
  ASSERT_EQ(TestGuardImpl::getDeviceIndex(), i2);
  // 再次设置 g 的索引为 i2
  g.set_index(i2);
  // 断言初始设备与 g 的原始设备索引相等
  ASSERT_EQ(g.original_device(), dev(init_i));
  // 断言当前设备与 g 的当前设备索引相等
  ASSERT_EQ(g.current_device(), dev(i2));
  // 断言 TestGuardImpl 中的设备索引为 i2
  ASSERT_EQ(TestGuardImpl::getDeviceIndex(), i2);
}

// -- InlineOptionalDeviceGuard
// --------------------------------------------------

// 使用 MaybeTestGuard 作为 InlineOptionalDeviceGuard 的别名
using MaybeTestGuard = InlineOptionalDeviceGuard<TestGuardImpl>;

// 定义一个测试用例 InlineOptionalDeviceGuard.Constructor
TEST(InlineOptionalDeviceGuard, Constructor) {
  // 对于每个设备索引 i 的初始化列表中的值，执行以下测试
  for (DeviceIndex i : std::initializer_list<DeviceIndex>{-1, 0, 1}) {
    // 初始化设备索引为 0
    DeviceIndex init_i = 0;
    // 设置测试宏的设备索引为 init_i
    TestGuardImpl::setDeviceIndex(init_i);
    
    // 定义一个 lambda 函数 test_body，用于测试 MaybeTestGuard 对象 g
    auto test_body = [&](MaybeTestGuard& g) -> void {
      // 断言 g 的原始设备与 init_i 相等
      ASSERT_EQ(g.original_device(), dev(init_i));
      // 断言 g 的当前设备与 i 相等（若 i 为 -1，则当前设备与 init_i 相等）
      ASSERT_EQ(g.current_device(), dev(i == -1 ? init_i : i));
      // 断言 TestGuardImpl 中的设备索引为 i（若 i 为 -1，则设备索引为 init_i）
      ASSERT_EQ(TestGuardImpl::getDeviceIndex(), i == -1 ? init_i : i);
      // 测试未加括号的设备索引写入
      TestGuardImpl::setDeviceIndex(4);
    };

    {
      // 使用设备索引构造 MaybeTestGuard 对象 g
      MaybeTestGuard g(i);
      test_body(g);
    }
    
    // 断言测试宏的设备索引为 init_i
    ASSERT_EQ(TestGuardImpl::getDeviceIndex(), init_i);

    {
      // 使用设备构造函数构造 MaybeTestGuard 对象 g
      MaybeTestGuard g(dev(i));
      test_body(g);
    }
    
    // 断言测试宏的设备索引为 init_i
    ASSERT_EQ(TestGuardImpl::getDeviceIndex(), init_i);

    {
      // 使用可选构造函数构造 MaybeTestGuard 对象 g
      MaybeTestGuard g(make_optional(dev(i)));
      test_body(g);
    }
    
    // 断言测试宏的设备索引为 init_i
    ASSERT_EQ(TestGuardImpl::getDeviceIndex(), init_i);
  }
}

// 定义一个测试用例 InlineOptionalDeviceGuard.NullaryConstructor
TEST(InlineOptionalDeviceGuard, NullaryConstructor) {
  // 初始化设备索引为 0
  DeviceIndex init_i = 0;
  // 设置测试宏的设备索引为 init_i
  TestGuardImpl::setDeviceIndex(init_i);
  
  // 定义一个 lambda 函数 test_body，用于测试 MaybeTestGuard 对象 g
  auto test_body = [&](MaybeTestGuard& g) -> void {
    // 断言 g 的原始设备为 nullopt
    ASSERT_EQ(g.original_device(), nullopt);
    // 断言 g 的当前设备为 nullopt
    ASSERT_EQ(g.current_device(), nullopt);
    // 断言 TestGuardImpl 中的设备索引为 init_i
    ASSERT_EQ(TestGuardImpl::getDeviceIndex(), init_i);
  };

  {
    // 使用默认构造函数构造 MaybeTestGuard 对象 g
    MaybeTestGuard g;
    test_body(g);
  }
  
  {
    // 如果要使 nullopt 直接工作，请定义一个 nullopt_t 重载。
    // 但实际上不清楚为什么会需要这样做。
    // 定义一个 optional<Device> 对象 dev_opt，其值为 nullopt
    optional<Device> dev_opt = nullopt;
    // 使用 dev_opt 构造 MaybeTestGuard 对象 g
    MaybeTestGuard g(dev_opt);
    test_body(g);
  }
}

// 定义一个测试用例 InlineOptionalDeviceGuard.SetDevice
TEST(InlineOptionalDeviceGuard, SetDevice) {
  // 初始化设备索引为 0
  DeviceIndex init_i = 0;
  // 设置测试宏的设备索引为 init_i
  TestGuardImpl::setDeviceIndex(init_i);
  // 默认构造一个 MaybeTestGuard 对象 g
  MaybeTestGuard g;
  // 设置设备索引为 1 的设备到 g
  DeviceIndex i = 1;
  g.set_device(dev(i));
  // 断言 g 的原始设备为包含 init_i 的 optional<Device>
  ASSERT_EQ(g.original_device(), make_optional(dev(init_i)));
  // 断言 g 的当前设备为包含 i 的 optional<Device>
  ASSERT_EQ(g.current_device(), make_optional(dev(i)));
  // 断言 TestGuardImpl 中的设备索引为 i
  ASSERT_EQ(TestGuardImpl::getDeviceIndex(), i);
  // 再次设置设备索引为 i 的设备到 g
  g.set_device(dev(i));
  // 断言 g 的原始设备为包含 init_i 的 optional<Device>
  ASSERT_EQ(g.original_device(), make_optional(dev(init_i)));
  // 断言 g 的当前设备为包含 i 的 optional<Device>
  ASSERT_EQ(g.current_device(), make_optional(dev(i)));
  // 断言 TestGuardImpl 中的设备索引为 i
  ASSERT_EQ(TestGuardImpl::getDeviceIndex(), i);
}
# 定义一个名为 TEST 的测试函数，用于测试 InlineOptionalDeviceGuard 类的 SetIndex 方法
TEST(InlineOptionalDeviceGuard, SetIndex) {
  # 初始化设备索引为 0
  DeviceIndex init_i = 0;
  # 调用 TestGuardImpl 的 setDeviceIndex 方法，设置设备索引为 init_i
  TestGuardImpl::setDeviceIndex(init_i);
  # 设置设备索引为 1
  DeviceIndex i = 1;
  # 创建 MaybeTestGuard 对象 g
  MaybeTestGuard g;
  # 调用 g 的 set_index 方法，设置其内部设备索引为 i
  g.set_index(i);
  # 断言 g 的初始设备索引与 init_i 相等，并封装成 optional
  ASSERT_EQ(g.original_device(), make_optional(dev(init_i)));
  # 断言 g 的当前设备索引与 i 相等，并封装成 optional
  ASSERT_EQ(g.current_device(), make_optional(dev(i)));
  # 断言 TestGuardImpl 的当前设备索引与 i 相等
  ASSERT_EQ(TestGuardImpl::getDeviceIndex(), i);
  # 再次调用 g 的 set_index 方法，设置其内部设备索引为 i
  g.set_index(i);
  # 断言 g 的初始设备索引与 init_i 相等，并封装成 optional
  ASSERT_EQ(g.original_device(), make_optional(dev(init_i)));
  # 断言 g 的当前设备索引与 i 相等，并封装成 optional
  ASSERT_EQ(g.current_device(), make_optional(dev(i)));
  # 断言 TestGuardImpl 的当前设备索引与 i 相等
  ASSERT_EQ(TestGuardImpl::getDeviceIndex(), i);
}
```