# `.\pytorch\aten\src\ATen\test\cpu_generator_test.cpp`

```
// 引入 Google 测试框架的头文件
#include <gtest/gtest.h>

// 引入 ATen 库的头文件
#include <ATen/ATen.h>
#include <ATen/Utils.h>
#include <ATen/CPUGeneratorImpl.h>
#include <ATen/core/PhiloxRNGEngine.h>
#include <c10/util/irange.h>

// 引入 C++ 标准库的头文件
#include <thread>
#include <limits>
#include <random>

// 使用 ATen 命名空间
using namespace at;

// 定义测试套件 CPUGeneratorImpl 下的测试用例 TestGeneratorDynamicCast
TEST(CPUGeneratorImpl, TestGeneratorDynamicCast) {
  // 测试描述：检查在 CPU 下的动态类型转换
  auto foo = at::detail::createCPUGenerator();
  auto result = check_generator<CPUGeneratorImpl>(foo);
  // 断言实际类型与预期类型的哈希码相同
  ASSERT_EQ(typeid(CPUGeneratorImpl*).hash_code(), typeid(result).hash_code());
}

// 定义测试套件 CPUGeneratorImpl 下的测试用例 TestDefaultGenerator
TEST(CPUGeneratorImpl, TestDefaultGenerator) {
  // 测试描述：
  // 检查默认生成器仅创建一次
  // 在所有调用中生成器的地址应该相同
  auto foo = at::detail::getDefaultCPUGenerator();
  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
  auto bar = at::detail::getDefaultCPUGenerator();
  ASSERT_EQ(foo, bar);
}

// 定义测试套件 CPUGeneratorImpl 下的测试用例 TestCloning
TEST(CPUGeneratorImpl, TestCloning) {
  // 测试描述：
  // 检查新生成器的克隆操作
  // 注意，不允许将其他生成器状态克隆到默认生成器中
  auto gen1 = at::detail::createCPUGenerator();
  auto cpu_gen1 = check_generator<CPUGeneratorImpl>(gen1);
  // NOLINTNEXTLINE(clang-analyzer-security.insecureAPI.rand)
  cpu_gen1->random(); // 推进 gen1 的状态
  // NOLINTNEXTLINE(clang-analyzer-security.insecureAPI.rand)
  cpu_gen1->random();
  auto gen2 = at::detail::createCPUGenerator();
  gen2 = gen1.clone();
  auto cpu_gen2 = check_generator<CPUGeneratorImpl>(gen2);
  // NOLINTNEXTLINE(clang-analyzer-security.insecureAPI.rand)
  ASSERT_EQ(cpu_gen1->random(), cpu_gen2->random());
}

// 定义一个线程函数 thread_func_get_engine_op，用于测试套件 CPUGeneratorImpl 下的多线程测试用例 TestMultithreadingGetEngineOperator
void thread_func_get_engine_op(CPUGeneratorImpl* generator) {
  std::lock_guard<std::mutex> lock(generator->mutex_);
  // NOLINTNEXTLINE(clang-analyzer-security.insecureAPI.rand)
  generator->random();
}

// 定义测试套件 CPUGeneratorImpl 下的多线程测试用例 TestMultithreadingGetEngineOperator
TEST(CPUGeneratorImpl, TestMultithreadingGetEngineOperator) {
  // 测试描述：
  // 检查 CPUGeneratorImpl 的可重入性，当多个线程请求随机样本时，引擎状态不会损坏
  // 参见 Note [Acquire lock when using random generators]
  auto gen1 = at::detail::createCPUGenerator();
  auto cpu_gen1 = check_generator<CPUGeneratorImpl>(gen1);
  auto gen2 = at::detail::createCPUGenerator();
  {
    std::lock_guard<std::mutex> lock(gen1.mutex());
    gen2 = gen1.clone(); // 捕获默认生成器当前状态，将其赋给 gen2
  }
  std::thread t0{thread_func_get_engine_op, cpu_gen1};  // 创建线程 t0，执行 thread_func_get_engine_op 函数，传入 cpu_gen1 参数
  std::thread t1{thread_func_get_engine_op, cpu_gen1};  // 创建线程 t1，执行 thread_func_get_engine_op 函数，传入 cpu_gen1 参数
  std::thread t2{thread_func_get_engine_op, cpu_gen1};  // 创建线程 t2，执行 thread_func_get_engine_op 函数，传入 cpu_gen1 参数
  t0.join();  // 等待线程 t0 执行完成
  t1.join();  // 等待线程 t1 执行完成
  t2.join();  // 等待线程 t2 执行完成
  std::lock_guard<std::mutex> lock(gen2.mutex());  // 使用 gen2 的互斥锁创建一个锁保护范围
  auto cpu_gen2 = check_generator<CPUGeneratorImpl>(gen2);  // 检查 gen2 是否为 CPUGeneratorImpl 类型的生成器，并赋给 cpu_gen2
  // NOLINTNEXTLINE(clang-analyzer-security.insecureAPI.rand)
  cpu_gen2->random();  // 调用 cpu_gen2 的 random() 方法生成随机数
  // NOLINTNEXTLINE(clang-analyzer-security.insecureAPI.rand)
  cpu_gen2->random();  // 再次调用 cpu_gen2 的 random() 方法生成随机数
  // NOLINTNEXTLINE(clang-analyzer-security.insecureAPI.rand)
  cpu_gen2->random();  // 第三次调用 cpu_gen2 的 random() 方法生成随机数
  // NOLINTNEXTLINE(clang-analyzer-security.insecureAPI.rand)
  ASSERT_EQ(cpu_gen1->random(), cpu_gen2->random());  // 断言：cpu_gen1 和 cpu_gen2 生成的随机数相等
}

TEST(CPUGeneratorImpl, TestGetSetCurrentSeed) {
  // Test Description:
  // Test current seed getter and setter
  // See Note [Acquire lock when using random generators]

  // 获取默认的 CPU 生成器对象
  auto foo = at::detail::getDefaultCPUGenerator();
  // 使用互斥锁保护，确保线程安全地设置当前种子值
  std::lock_guard<std::mutex> lock(foo.mutex());
  // 设置当前种子值为 123
  foo.set_current_seed(123);
  // 获取当前种子值
  auto current_seed = foo.current_seed();
  // 断言当前种子值为 123
  ASSERT_EQ(current_seed, 123);
}

void thread_func_get_set_current_seed(Generator generator) {
  // 使用互斥锁保护，确保线程安全地获取和设置种子值
  std::lock_guard<std::mutex> lock(generator.mutex());
  // 获取当前种子值
  auto current_seed = generator.current_seed();
  // 将当前种子值加一
  current_seed++;
  // 设置种子值为增加后的值
  generator.set_current_seed(current_seed);
}

TEST(CPUGeneratorImpl, TestMultithreadingGetSetCurrentSeed) {
  // Test Description:
  // Test current seed getter and setter are thread safe
  // See Note [Acquire lock when using random generators]

  // 获取默认的 CPU 生成器对象
  auto gen1 = at::detail::getDefaultCPUGenerator();
  // 获取初始的种子值
  auto initial_seed = gen1.current_seed();
  // 创建三个线程，每个线程调用 thread_func_get_set_current_seed 函数
  std::thread t0{thread_func_get_set_current_seed, gen1};
  std::thread t1{thread_func_get_set_current_seed, gen1};
  std::thread t2{thread_func_get_set_current_seed, gen1};
  // 等待所有线程执行完成
  t0.join();
  t1.join();
  t2.join();
  // 断言当前种子值为初始种子值加三
  ASSERT_EQ(gen1.current_seed(), initial_seed+3);
}

TEST(CPUGeneratorImpl, TestRNGForking) {
  // Test Description:
  // Test that state of a generator can be frozen and
  // restored
  // See Note [Acquire lock when using random generators]

  // 获取默认的 CPU 生成器对象
  auto default_gen = at::detail::getDefaultCPUGenerator();
  // 创建一个新的 CPU 生成器对象
  auto current_gen = at::detail::createCPUGenerator();
  {
    // 使用互斥锁保护，默认生成器的状态
    std::lock_guard<std::mutex> lock(default_gen.mutex());
    // 克隆当前默认生成器的状态到新生成器
    current_gen = default_gen.clone(); // capture the current state of default generator
  }
  // 生成目标值
  auto target_value = at::randn({1000});
  // 改变主生成器的内部状态
  auto x = at::randn({100000});
  // 使用克隆的生成器生成分支值
  auto forked_value = at::randn({1000}, current_gen);
  // 断言目标值的总和与分支值的总和相等
  ASSERT_EQ(target_value.sum().item<double>(), forked_value.sum().item<double>());
}

/**
 * Philox CPU Engine Tests
 */

TEST(CPUGeneratorImpl, TestPhiloxEngineReproducibility) {
  // Test Description:
  //   Tests if same inputs give same results.
  //   launch on same thread index and create two engines.
  //   Given same seed, idx and offset, assert that the engines
  //   should be aligned and have the same sequence.

  // 创建两个相同参数的 Philox4_32 引擎对象
  at::Philox4_32 engine1(0, 0, 4);
  at::Philox4_32 engine2(0, 0, 4);
  // 断言两个引擎的下一个随机数相等
  ASSERT_EQ(engine1(), engine2());
}

TEST(CPUGeneratorImpl, TestPhiloxEngineOffset1) {
  // Test Description:
  //   Tests offsetting in same thread index.
  //   make one engine skip the first 8 values and
  //   make another engine increment to until the
  //   first 8 values. Assert that the first call
  //   of engine2 and the 9th call of engine1 are equal.

  // 创建两个不同偏移的 Philox4_32 引擎对象
  at::Philox4_32 engine1(123, 1, 0);
  // 注意：偏移量是4的倍数，因此要跳过8个值，偏移量为2
  at::Philox4_32 engine2(123, 1, 2);
  for (C10_UNUSED const auto i : c10::irange(8)) {
    // 注意：而不是连续调用 engine() 8 次
    // 通过调用 incr() 函数两次，我们可以达到相同的功能。
    engine1();
  }
  // 断言 engine1() 的返回值与 engine2() 的返回值相等
  ASSERT_EQ(engine1(), engine2());
}

TEST(CPUGeneratorImpl, TestPhiloxEngineOffset2) {
  // Test Description:
  //   测试生成器在第2^190个值末尾的边界情况。
  //   在相同线程索引上启动并创建两个引擎。
  //   使 engine1 在线程 0 上跳至第2^64个128位，增量值为 std::numeric_limits<uint64_t>::max()
  //   使 engine2 在第2^64个线程上跳至第2^64个128位，增量值为 std::numeric_limits<uint64_t>::max()
  unsigned long long increment_val = std::numeric_limits<uint64_t>::max();
  // 创建 engine1，线程索引为 0，初始值为 123，增量为 increment_val
  at::Philox4_32 engine1(123, 0, increment_val);
  // 创建 engine2，线程索引为 increment_val，初始值为 123，增量为 increment_val
  at::Philox4_32 engine2(123, increment_val, increment_val);

  // engine2 增加 increment_val 的步数
  engine2.incr_n(increment_val);
  // engine2 再增加一步
  engine2.incr();
  // 断言 engine1() 和 engine2() 的结果相等
  ASSERT_EQ(engine1(), engine2());
}

TEST(CPUGeneratorImpl, TestPhiloxEngineOffset3) {
  // Test Description:
  //   测试线程索引之间的边界情况。
  //   在相同线程索引上启动并创建两个引擎。
  //   使 engine1 在线程 0 上跳至第2^64个128位，增量值为 std::numeric_limits<uint64_t>::max()
  //   使 engine2 在线程 1 上从第0个128位开始
  unsigned long long increment_val = std::numeric_limits<uint64_t>::max();
  // 创建 engine1，线程索引为 0，初始值为 123，增量为 increment_val
  at::Philox4_32 engine1(123, 0, increment_val);
  // 创建 engine2，线程索引为 1，初始值为 123，增量为 0
  at::Philox4_32 engine2(123, 1, 0);
  // engine1 增加一步
  engine1.incr();
  // 断言 engine1() 和 engine2() 的结果相等
  ASSERT_EQ(engine1(), engine2());
}

TEST(CPUGeneratorImpl, TestPhiloxEngineIndex) {
  // Test Description:
  //   测试线程索引是否正常工作。
  //   创建两个引擎，线程索引不同但增量相同。
  at::Philox4_32 engine1(123456, 0, 4);
  at::Philox4_32 engine2(123456, 1, 4);
  // 断言 engine1() 和 engine2() 的结果不相等
  ASSERT_NE(engine1(), engine2());
}

/**
 * MT19937 CPU Engine Tests
 */

TEST(CPUGeneratorImpl, TestMT19937EngineReproducibility) {
  // Test Description:
  //   测试相同输入是否给出与 std::mt19937 相同的结果。

  // 使用零种子进行测试
  at::mt19937 engine1(0);
  std::mt19937 engine2(0);
  // 循环比较 10000 次 engine1() 和 engine2() 的结果是否相等
  for (C10_UNUSED const auto i : c10::irange(10000)) {
    ASSERT_EQ(engine1(), engine2());
  }

  // 使用大种子进行测试
  engine1 = at::mt19937(2147483647);
  engine2 = std::mt19937(2147483647);
  // 循环比较 10000 次 engine1() 和 engine2() 的结果是否相等
  for (C10_UNUSED const auto i : c10::irange(10000)) {
    ASSERT_EQ(engine1(), engine2());
  }

  // 使用随机种子进行测试
  std::random_device rd;
  auto seed = rd();
  engine1 = at::mt19937(seed);
  engine2 = std::mt19937(seed);
  // 循环比较 10000 次 engine1() 和 engine2() 的结果是否相等
  for (C10_UNUSED const auto i : c10::irange(10000)) {
    ASSERT_EQ(engine1(), engine2());
  }
}

TEST(CPUGeneratorImpl, TestPhiloxEngineReproducibilityRandN) {
  // 创建两个 Philox4_32 引擎，使用相同的种子和增量
  at::Philox4_32 engine1(0, 0, 4);
  at::Philox4_32 engine2(0, 0, 4);
  // 断言 engine1.randn(1) 和 engine2.randn(1) 的结果相等
  ASSERT_EQ(engine1.randn(1), engine2.randn(1));
}

TEST(CPUGeneratorImpl, TestPhiloxEngineSeedRandN) {
  // 创建两个 Philox4_32 引擎，分别使用种子 0 和 123456
  at::Philox4_32 engine1(0);
  at::Philox4_32 engine2(123456);
  // 断言 engine1.randn(1) 和 engine2.randn(1) 的结果不相等
  ASSERT_NE(engine1.randn(1), engine2.randn(1));
}
# 定义一个测试案例，用于测试 CPUGeneratorImpl 类中的 Philox 算法的确定性特性
TEST(CPUGeneratorImpl, TestPhiloxDeterministic) {
  # 创建一个 Philox4_32 类型的随机数生成器 engine1，种子参数为 (0, 0, 4)
  at::Philox4_32 engine1(0, 0, 4);
  # 断言生成器 engine1 的第一个随机数结果为 4013802324，用于验证确定性
  ASSERT_EQ(engine1(), 4013802324);  // Determinism!
  # 断言生成器 engine1 的第二个随机数结果为 2979262830，用于验证确定性
  ASSERT_EQ(engine1(), 2979262830);  // Determinism!

  # 创建一个新的 Philox4_32 类型的随机数生成器 engine2，种子参数为 (10, 0, 1)
  at::Philox4_32 engine2(10, 0, 1);
  # 断言生成器 engine2 的第一个随机数结果为 2007330488，用于验证确定性
  ASSERT_EQ(engine2(), 2007330488);  // Determinism!
  # 断言生成器 engine2 的第二个随机数结果为 2354548925，用于验证确定性
  ASSERT_EQ(engine2(), 2354548925);  // Determinism!
}
```