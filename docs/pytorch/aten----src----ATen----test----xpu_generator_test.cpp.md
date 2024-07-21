# `.\pytorch\aten\src\ATen\test\xpu_generator_test.cpp`

```
// 包含 Google Test 的头文件，用于测试框架
#include <gtest/gtest.h>

// 包含 ATen 库的主头文件，提供张量运算支持
#include <ATen/ATen.h>
// 包含 ATen XPU 上下文相关的头文件，用于支持特定设备（如 GPU）上的张量操作
#include <ATen/xpu/XPUContext.h>
// 包含 ATen XPU 生成器实现的头文件，用于生成随机数
#include <ATen/xpu/XPUGeneratorImpl.h>
// 包含 ATen 的 Philox 随机数引擎的头文件，用于生成随机数种子
#include <ATen/core/PhiloxRNGEngine.h>

// 包含标准 C 库的断言头文件，用于运行时检查
#include <assert.h>
// 包含 C++ 标准库的线程支持头文件，用于多线程测试
#include <thread>

// 定义 XPUGeneratorTest 测试套件，用于测试 XPU 生成器的功能
TEST(XpuGeneratorTest, testGeneratorDynamicCast) {
  // 如果当前设备不支持 XPU（如 GPU），则直接返回
  if (!at::xpu::is_available()) {
    return;
  }
  // 创建一个 XPUGeneratorImpl 类型的生成器
  auto foo = at::xpu::detail::createXPUGenerator();
  // 尝试将 foo 转换为 XPUGeneratorImpl 类型
  auto result = foo.get<at::XPUGeneratorImpl>();
  // 检查 result 的类型是否与 XPUGeneratorImpl* 的类型哈希码相等
  EXPECT_EQ(typeid(at::XPUGeneratorImpl*).hash_code(), typeid(result).hash_code());
}

// 定义测试默认生成器的测试用例
TEST(XpuGeneratorTest, testDefaultGenerator) {
  // 如果当前设备不支持 XPU（如 GPU），则直接返回
  if (!at::xpu::is_available()) {
    return;
  }
  // 获取默认的 XPUGeneratorImpl 类型的生成器 foo 和 bar
  auto foo = at::xpu::detail::getDefaultXPUGenerator();
  auto bar = at::xpu::detail::getDefaultXPUGenerator();
  // 检查 foo 和 bar 是否相等
  EXPECT_EQ(foo, bar);

  // 将 foo 的偏移量左移一位，并设置为 offset
  auto offset = foo.get_offset() << 1;
  foo.set_offset(offset);
  // 检查 foo 的偏移量是否与 offset 相等
  EXPECT_EQ(foo.get_offset(), offset);

  // 如果系统有多个 XPU 设备，则进行额外的测试
  if (c10::xpu::device_count() >= 2) {
    // 获取第一个设备的默认生成器 foo 和 bar
    foo = at::xpu::detail::getDefaultXPUGenerator(0);
    bar = at::xpu::detail::getDefaultXPUGenerator(0);
    // 检查 foo 和 bar 是否相等
    EXPECT_EQ(foo, bar);

    // 获取第一个和第二个设备的默认生成器 foo 和 bar
    foo = at::xpu::detail::getDefaultXPUGenerator(0);
    bar = at::xpu::detail::getDefaultXPUGenerator(1);
    // 检查 foo 和 bar 是否不相等
    EXPECT_NE(foo, bar);
  }
}

// 定义生成器克隆的测试用例
TEST(XpuGeneratorTest, testCloning) {
  // 如果当前设备不支持 XPU（如 GPU），则直接返回
  if (!at::xpu::is_available()) {
    return;
  }
  // 创建一个 XPUGeneratorImpl 类型的生成器 gen1，并设置当前种子为 123
  auto gen1 = at::xpu::detail::createXPUGenerator();
  gen1.set_current_seed(123); // 修改 gen1 的状态
  // 检查 gen1 是否为 XPUGeneratorImpl 类型，并获取其指针 xpu_gen1
  auto xpu_gen1 = at::check_generator<at::XPUGeneratorImpl>(gen1);
  // 设置每个线程的 Philox 偏移量为 4
  xpu_gen1->set_philox_offset_per_thread(4);
  // 克隆 gen1 生成一个新的生成器 gen2
  auto gen2 = at::xpu::detail::createXPUGenerator();
  gen2 = gen1.clone();
  // 检查 gen1 和 gen2 的当前种子是否相等
  EXPECT_EQ(gen1.current_seed(), gen2.current_seed());
  // 检查 gen1 和 gen2 的 Philox 偏移量是否相等
  EXPECT_EQ(
    xpu_gen1->philox_offset_per_thread(),
    at::check_generator<at::XPUGeneratorImpl>(gen2)->philox_offset_per_thread()
  );
}

// 定义多线程获取和设置当前种子的测试用例
void thread_func_get_set_current_seed(at::Generator generator) {
  // 获取生成器的互斥锁，并锁定当前线程
  std::lock_guard<std::mutex> lock(generator.mutex());
  // 获取当前种子，并增加 1
  auto current_seed = generator.current_seed();
  current_seed++;
  // 设置当前种子为增加后的值
  generator.set_current_seed(current_seed);
}

TEST(XpuGeneratorTest, testMultithreadingGetSetCurrentSeed) {
  // 如果当前设备不支持 XPU（如 GPU），则直接返回
  if (!at::xpu::is_available()) {
    return;
  }
  // 获取默认的 XPUGeneratorImpl 类型的生成器 gen1
  auto gen1 = at::xpu::detail::getDefaultXPUGenerator();
  // 获取初始种子值
  auto initial_seed = gen1.current_seed();
  // 创建三个线程分别执行 thread_func_get_set_current_seed 函数
  std::thread t0{thread_func_get_set_current_seed, gen1};
  std::thread t1{thread_func_get_set_current_seed, gen1};
  std::thread t2{thread_func_get_set_current_seed, gen1};
  // 等待三个线程执行完毕
  t0.join();
  t1.join();
  t2.join();
  // 检查当前种子是否增加了 3
  EXPECT_EQ(gen1.current_seed(), initial_seed+3);
}
```