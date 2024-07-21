# `.\pytorch\c10\test\core\impl\InlineStreamGuard_test.cpp`

```py
// 引入 Google Test 框架的头文件
#include <gtest/gtest.h>

// 引入 C10 库中的相关头文件
#include <c10/core/impl/FakeGuardImpl.h>
#include <c10/core/impl/InlineStreamGuard.h>

// 使用 c10 和 c10::impl 命名空间
using namespace c10;
using namespace c10::impl;

// 定义一个常量表达式，表示测试设备类型为 CUDA
constexpr auto TestDeviceType = DeviceType::CUDA;
// 使用 FakeGuardImpl 模板生成 TestGuardImpl 类型的别名
using TestGuardImpl = FakeGuardImpl<TestDeviceType>;

// 定义静态函数 dev，返回指定设备索引的 Device 对象
static Device dev(DeviceIndex index) {
  return Device(TestDeviceType, index);
}

// 定义静态函数 stream，返回指定设备索引和流ID的 Stream 对象
static Stream stream(DeviceIndex index, StreamId sid) {
  return Stream(Stream::UNSAFE, dev(index), sid);
}

// -- InlineStreamGuard -------------------------------------------------------

// 使用 InlineStreamGuard 模板生成 TestGuard 类型的别名
using TestGuard = InlineStreamGuard<TestGuardImpl>;

// 测试用例：测试 InlineStreamGuard 的构造函数
TEST(InlineStreamGuard, Constructor) {
  // 设置 FakeGuardImpl 的设备索引为 0，重置流信息
  TestGuardImpl::setDeviceIndex(0);
  TestGuardImpl::resetStreams();
  {
    // 创建 TestGuard 对象 g，其构造函数设置了指定设备和流
    TestGuard g(stream(1, 2));
    // 断言当前 FakeGuardImpl 的设备索引和流ID是否符合预期
    ASSERT_EQ(TestGuardImpl::getDeviceIndex(), 1);
    ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 2);
    ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 0);
    // 断言 TestGuard 对象 g 的原始流和当前流与预期相符
    ASSERT_EQ(g.original_stream(), stream(0, 0));
    ASSERT_EQ(g.current_stream(), stream(1, 2));
    // 断言 TestGuard 对象 g 的原始设备和当前设备与预期相符
    ASSERT_EQ(g.original_device(), dev(0));
    ASSERT_EQ(g.current_device(), dev(1));
  }
  // 断言退出 TestGuard 对象作用域后 FakeGuardImpl 的设备索引是否重置为 0
  ASSERT_EQ(TestGuardImpl::getDeviceIndex(), 0);
  ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 0);
  ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 0);
}

// 测试用例：测试 InlineStreamGuard 的 reset_stream 方法（相同设备）
TEST(InlineStreamGuard, ResetStreamSameSameDevice) {
  TestGuardImpl::setDeviceIndex(0);
  TestGuardImpl::resetStreams();
  {
    TestGuard g(stream(0, 2));
    // 调用 TestGuard 对象 g 的 reset_stream 方法，重设流信息
    g.reset_stream(stream(0, 3));
    ASSERT_EQ(TestGuardImpl::getDeviceIndex(), 0);
    ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 3);
    ASSERT_EQ(g.original_stream(), stream(0, 0));
    ASSERT_EQ(g.current_stream(), stream(0, 3));
    ASSERT_EQ(g.original_device(), dev(0));
    ASSERT_EQ(g.current_device(), dev(0));
  }
  ASSERT_EQ(TestGuardImpl::getDeviceIndex(), 0);
  ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 0);
}

// 测试用例：测试 InlineStreamGuard 的 reset_stream 方法（不同流、同一设备）
TEST(InlineStreamGuard, ResetStreamDifferentSameDevice) {
  TestGuardImpl::setDeviceIndex(0);
  TestGuardImpl::resetStreams();
  {
    TestGuard g(stream(1, 2));
    // 调用 TestGuard 对象 g 的 reset_stream 方法，重设流信息
    g.reset_stream(stream(1, 3));
    ASSERT_EQ(TestGuardImpl::getDeviceIndex(), 1);
    ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 3);
    ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 0);
    ASSERT_EQ(g.original_stream(), stream(0, 0));
    ASSERT_EQ(g.current_stream(), stream(1, 3));
    ASSERT_EQ(g.original_device(), dev(0));
    ASSERT_EQ(g.current_device(), dev(1));
  }
  ASSERT_EQ(TestGuardImpl::getDeviceIndex(), 0);
  ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 0);
  ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 0);
}

// 测试用例：测试 InlineStreamGuard 的 reset_stream 方法（不同设备）
TEST(InlineStreamGuard, ResetStreamDifferentDevice) {
  TestGuardImpl::setDeviceIndex(0);
  TestGuardImpl::resetStreams();
  {
    TestGuard g(stream(1, 2));
    // 调用 TestGuard 对象 g 的 reset_stream 方法，重设流信息
    g.reset_stream(stream(2, 3));
    ASSERT_EQ(TestGuardImpl::getDeviceIndex(), 2);
    ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(2), 3);
    ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 0);
    ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 0);
    // 断言 TestGuard 对象 g 的原始流和当前流与预期相符
    ASSERT_EQ(g.original_stream(), stream(0, 0));
    ASSERT_EQ(g.current_stream(), stream(2, 3));
    // 断言 TestGuard 对象 g 的原始设备和当前设备与预期相符
    ASSERT_EQ(g.original_device(), dev(0));
    ASSERT_EQ(g.current_device(), dev(2));
  }
  ASSERT_EQ(TestGuardImpl::getDeviceIndex(), 0);
  ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 0);
  ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 0);
}
    # 断言：验证对象 g 的 original_stream() 返回值是否等于 stream(0, 0)
    ASSERT_EQ(g.original_stream(), stream(0, 0));
    # 断言：验证对象 g 的 current_stream() 返回值是否等于 stream(2, 3)
    ASSERT_EQ(g.current_stream(), stream(2, 3));
    # 断言：验证对象 g 的 original_device() 返回值是否等于 dev(0)
    ASSERT_EQ(g.original_device(), dev(0));
    # 断言：验证对象 g 的 current_device() 返回值是否等于 dev(2)
    ASSERT_EQ(g.current_device(), dev(2));
  }
  # 断言：验证 TestGuardImpl 类的静态方法 getDeviceIndex() 返回值是否等于 0
  ASSERT_EQ(TestGuardImpl::getDeviceIndex(), 0);
  # 断言：验证 TestGuardImpl 类的静态方法 getCurrentStreamIdFor(2) 返回值是否等于 0
  ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(2), 0);
  # 断言：验证 TestGuardImpl 类的静态方法 getCurrentStreamIdFor(1) 返回值是否等于 0
  ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 0);
  # 断言：验证 TestGuardImpl 类的静态方法 getCurrentStreamIdFor(0) 返回值是否等于 0
  ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 0);
// 定义使用 InlineOptionalStreamGuard 别名的 OptionalTestGuard 类型别名
using OptionalTestGuard = InlineOptionalStreamGuard<TestGuardImpl>;

// 测试用例：构造函数测试
TEST(InlineOptionalStreamGuard, Constructor) {
  // 设置设备索引为 0，重置流
  TestGuardImpl::setDeviceIndex(0);
  TestGuardImpl::resetStreams();

  {
    // 在作用域内创建 OptionalTestGuard 对象 g，传入 stream(1, 2)
    OptionalTestGuard g(stream(1, 2));
    // 断言设备索引为 1，当前流 ID 为 2
    ASSERT_EQ(TestGuardImpl::getDeviceIndex(), 1);
    ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 2);
    // 断言设备索引为 0，当前流 ID 为 0（默认）
    ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 0);
    // 断言 g 的原始流为 make_optional(stream(0, 0))
    ASSERT_EQ(g.original_stream(), make_optional(stream(0, 0)));
    // 断言 g 的当前流为 make_optional(stream(1, 2))
    ASSERT_EQ(g.current_stream(), make_optional(stream(1, 2)));
  }

  // 断言设备索引为 0，当前流 ID 为 0
  ASSERT_EQ(TestGuardImpl::getDeviceIndex(), 0);
  ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 0);
  ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 0);

  {
    // 在作用域内创建 OptionalTestGuard 对象 g，传入 make_optional(stream(1, 2))
    OptionalTestGuard g(make_optional(stream(1, 2)));
    // 断言设备索引为 1，当前流 ID 为 2
    ASSERT_EQ(TestGuardImpl::getDeviceIndex(), 1);
    ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 2);
    // 断言设备索引为 0，当前流 ID 为 0（默认）
    ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 0);
    // 断言 g 的原始流为 make_optional(stream(0, 0))
    ASSERT_EQ(g.original_stream(), make_optional(stream(0, 0)));
    // 断言 g 的当前流为 make_optional(stream(1, 2))
    ASSERT_EQ(g.current_stream(), make_optional(stream(1, 2)));
  }

  // 断言设备索引为 0，当前流 ID 为 0
  ASSERT_EQ(TestGuardImpl::getDeviceIndex(), 0);
  ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 0);
  ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 0);

  {
    // 在作用域内创建未指定流的 OptionalTestGuard 对象 g
    OptionalTestGuard g;
    // 断言设备索引为 0
    ASSERT_EQ(TestGuardImpl::getDeviceIndex(), 0);
    // 断言各设备的当前流 ID 为 0（默认）
    ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 0);
    ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 0);
  }

  // 断言设备索引为 0，当前流 ID 为 0
  ASSERT_EQ(TestGuardImpl::getDeviceIndex(), 0);
  ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 0);
  ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 0);
}

// 测试用例：重置相同设备的流
TEST(InlineOptionalStreamGuard, ResetStreamSameDevice) {
  // 设置设备索引为 0，重置流
  TestGuardImpl::setDeviceIndex(0);
  TestGuardImpl::resetStreams();

  {
    // 在作用域内创建 OptionalTestGuard 对象 g
    OptionalTestGuard g;
    // 重置流为 stream(1, 3)
    g.reset_stream(stream(1, 3));
    // 断言设备索引为 1，当前流 ID 为 3
    ASSERT_EQ(TestGuardImpl::getDeviceIndex(), 1);
    ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 3);
    // 断言设备索引为 0，当前流 ID 为 0
    ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 0);
    // 断言 g 的原始流为 make_optional(stream(0, 0))
    ASSERT_EQ(g.original_stream(), make_optional(stream(0, 0)));
    // 断言 g 的当前流为 make_optional(stream(1, 3))
    ASSERT_EQ(g.current_stream(), make_optional(stream(1, 3)));
  }

  // 断言设备索引为 0，当前流 ID 为 0
  ASSERT_EQ(TestGuardImpl::getDeviceIndex(), 0);
  ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 0);
  ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 0);
}

// 测试用例：重置不同设备的流
TEST(InlineOptionalStreamGuard, ResetStreamDifferentDevice) {
  // 设置设备索引为 0，重置流
  TestGuardImpl::setDeviceIndex(0);
  TestGuardImpl::resetStreams();

  {
    // 在作用域内创建 OptionalTestGuard 对象 g
    OptionalTestGuard g;
    // 重置流为 stream(2, 3)
    g.reset_stream(stream(2, 3));
    // 断言设备索引为 2，当前流 ID 为 3
    ASSERT_EQ(TestGuardImpl::getDeviceIndex(), 2);
    ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(2), 3);
    // 断言设备索引为 1，当前流 ID 为 0
    ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 0);
    // 断言设备索引为 0，当前流 ID 为 0
    ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 0);
    // 断言 g 的原始流为 make_optional(stream(0, 0))
    ASSERT_EQ(g.original_stream(), make_optional(stream(0, 0)));
  }
}
    ASSERT_EQ(g.current_stream(), make_optional(stream(2, 3)));

验证当前的流是否与指定的可选流相等，确保返回的流对象包含 (2, 3)。


  }

这里结束了之前的代码块，可能是一个函数或循环的结尾。


  ASSERT_EQ(TestGuardImpl::getDeviceIndex(), 0);

验证当前设备的索引是否为0。


  ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(2), 0);

验证对于设备2，当前流的ID是否为0。


  ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 0);

验证对于设备1，当前流的ID是否为0。


  ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 0);

验证对于设备0，当前流的ID是否为0。
// -- InlineMultiStreamGuard
// -------------------------------------------------------

// 使用别名定义 MultiTestGuard 为 InlineMultiStreamGuard<TestGuardImpl> 的简化形式
using MultiTestGuard = InlineMultiStreamGuard<TestGuardImpl>;

// 定义测试用例 TEST(InlineMultiStreamGuard, Constructor)
TEST(InlineMultiStreamGuard, Constructor) {
  // 重置 TestGuardImpl 的流信息
  TestGuardImpl::resetStreams();
  
  // 第一个测试环境
  {
    // 创建空的流向量
    std::vector<Stream> streams;
    // 创建 MultiTestGuard 对象 g
    MultiTestGuard g(streams);
    // 断言第一个流的当前流 ID 为 0
    ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 0);
    // 断言第二个流的当前流 ID 为 0
    ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 0);
  }
  // 再次断言第一个流的当前流 ID 为 0
  ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 0);
  // 再次断言第二个流的当前流 ID 为 0
  ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 0);
  
  // 第二个测试环境
  {
    // 创建包含单个流的流向量
    std::vector<Stream> streams = {stream(0, 2)};
    // 创建 MultiTestGuard 对象 g
    MultiTestGuard g(streams);
    // 断言第一个流的当前流 ID 为 2
    ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 2);
    // 断言第二个流的当前流 ID 为 0
    ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 0);
  }
  // 再次断言第一个流的当前流 ID 为 0
  ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 0);
  // 再次断言第二个流的当前流 ID 为 0
  ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 0);
  
  // 第三个测试环境
  {
    // 创建包含单个流的流向量
    std::vector<Stream> streams = {stream(1, 3)};
    // 创建 MultiTestGuard 对象 g
    MultiTestGuard g(streams);
    // 断言第一个流的当前流 ID 为 0
    ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 0);
    // 断言第二个流的当前流 ID 为 3
    ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 3);
  }
  // 再次断言第一个流的当前流 ID 为 0
  ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 0);
  // 再次断言第二个流的当前流 ID 为 0
  ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 0);
  
  // 第四个测试环境
  {
    // 创建包含两个流的流向量
    std::vector<Stream> streams = {stream(0, 2), stream(1, 3)};
    // 创建 MultiTestGuard 对象 g
    MultiTestGuard g(streams);
    // 断言第一个流的当前流 ID 为 2
    ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 2);
    // 断言第二个流的当前流 ID 为 3
    ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 3);
  }
  // 再次断言第一个流的当前流 ID 为 0
  ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 0);
  // 再次断言第二个流的当前流 ID 为 0
  ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 0);
}
```