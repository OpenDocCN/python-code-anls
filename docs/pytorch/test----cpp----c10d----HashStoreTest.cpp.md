# `.\pytorch\test\cpp\c10d\HashStoreTest.cpp`

```
#include <c10/util/irange.h> // 引入用于迭代的辅助头文件
#include "StoreTestCommon.hpp" // 引入自定义的存储测试公共头文件

#include <unistd.h> // POSIX 标准的头文件

#include <iostream> // 输入输出流
#include <thread> // 多线程支持

#include <torch/csrc/distributed/c10d/HashStore.hpp> // 引入分布式哈希存储的头文件
#include <torch/csrc/distributed/c10d/PrefixStore.hpp> // 引入带前缀的存储头文件

constexpr int64_t kShortStoreTimeoutMillis = 100; // 定义短超时时间为100毫秒

void testGetSet(std::string prefix = "") {
  // Basic set/get
  {
    auto hashStore = c10::make_intrusive<c10d::HashStore>(); // 创建哈希存储对象
    c10d::PrefixStore store(prefix, hashStore); // 创建带前缀的存储对象
    c10d::test::set(store, "key0", "value0"); // 设置键值对 "key0" -> "value0"
    c10d::test::set(store, "key1", "value1"); // 设置键值对 "key1" -> "value1"
    c10d::test::set(store, "key2", "value2"); // 设置键值对 "key2" -> "value2"
    c10d::test::check(store, "key0", "value0"); // 检查键 "key0" 对应的值是否为 "value0"
    c10d::test::check(store, "key1", "value1"); // 检查键 "key1" 对应的值是否为 "value1"
    c10d::test::check(store, "key2", "value2"); // 检查键 "key2" 对应的值是否为 "value2"

    // Check compareSet, does not check return value
    c10d::test::compareSet(store, "key0", "wrongExpectedValue", "newValue"); // 比较设置 "key0" 的值，预期值不匹配，不检查返回值
    c10d::test::check(store, "key0", "value0"); // 检查键 "key0" 对应的值是否为 "value0"
    c10d::test::compareSet(store, "key0", "value0", "newValue"); // 比较设置 "key0" 的值，预期值匹配，设置为 "newValue"
    c10d::test::check(store, "key0", "newValue"); // 检查键 "key0" 对应的值是否为 "newValue"

    auto numKeys = store.getNumKeys(); // 获取存储中键的数量
    EXPECT_EQ(numKeys, 3); // 断言键的数量为3
    auto delSuccess = store.deleteKey("key0"); // 删除键为 "key0" 的条目，并返回是否成功
    EXPECT_TRUE(delSuccess); // 断言删除成功
    numKeys = store.getNumKeys(); // 重新获取存储中键的数量
    EXPECT_EQ(numKeys, 2); // 断言键的数量为2
    auto delFailure = store.deleteKey("badKeyName"); // 尝试删除一个不存在的键，并返回是否成功
    EXPECT_FALSE(delFailure); // 断言删除失败
    auto timeout = std::chrono::milliseconds(kShortStoreTimeoutMillis); // 设置超时时间
    store.setTimeout(timeout); // 设置存储的超时时间
    EXPECT_THROW(store.get("key0"), c10::DistStoreError); // 断言获取 "key0" 的值会抛出 DistStoreError 异常
  }

  // get() waits up to timeout_.
  {
    auto hashStore = c10::make_intrusive<c10d::HashStore>(); // 创建哈希存储对象
    c10d::PrefixStore store(prefix, hashStore); // 创建带前缀的存储对象
    std::thread th([&]() { c10d::test::set(store, "key0", "value0"); }); // 在新线程中设置键值对 "key0" -> "value0"
    c10d::test::check(store, "key0", "value0"); // 检查键 "key0" 对应的值是否为 "value0"
    th.join(); // 等待线程执行完毕
  }
}

void stressTestStore(std::string prefix = "") {
  // Hammer on HashStore::add
  const auto numThreads = 4; // 线程数量
  const auto numIterations = 100; // 每个线程迭代次数

  std::vector<std::thread> threads; // 线程容器
  c10d::test::Semaphore sem1, sem2; // 信号量
  auto hashStore = c10::make_intrusive<c10d::HashStore>(); // 创建哈希存储对象
  c10d::PrefixStore store(prefix, hashStore); // 创建带前缀的存储对象

  for (C10_UNUSED const auto i : c10::irange(numThreads)) { // 迭代生成线程
    threads.emplace_back(std::thread([&] {
      sem1.post(); // 信号量增加
      sem2.wait(); // 等待信号量
      for (C10_UNUSED const auto j : c10::irange(numIterations)) {
        store.add("counter", 1); // 在存储中添加键 "counter" 的值
      }
    }));
  }

  sem1.wait(numThreads); // 等待所有线程完成
  sem2.post(numThreads); // 发送信号量通知所有线程继续

  for (auto& thread : threads) {
    thread.join(); // 等待所有线程执行完毕
  }
  std::string expected = std::to_string(numThreads * numIterations); // 期望的最终值
  c10d::test::check(store, "counter", expected); // 检查键 "counter" 的最终值是否符合预期
}

TEST(HashStoreTest, testGetAndSet) {
  testGetSet(); // 测试基本的获取和设置操作
}

TEST(HashStoreTest, testGetAndSetWithPrefix) {
  testGetSet("testPrefix"); // 测试带前缀的获取和设置操作
}

TEST(HashStoreTest, testStressStore) {
  stressTestStore(); // 测试存储的压力性能
}

TEST(HashStoreTest, testStressStoreWithPrefix) {
  stressTestStore("testPrefix"); // 测试带前缀的存储的压力性能
}
```