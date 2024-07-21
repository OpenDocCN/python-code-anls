# `.\pytorch\test\cpp\c10d\TCPStoreTest.cpp`

```py
// 包含所需的头文件：c10/util/irange.h 和 StoreTestCommon.hpp
#include <c10/util/irange.h>
#include "StoreTestCommon.hpp"

// 包含标准库头文件
#include <cstdlib>
#include <future>
#include <iostream>
#include <string>
#include <system_error>
#include <thread>

// 包含 Google 测试框架的头文件
#include <gtest/gtest.h>

// 包含 Torch 分布式库的头文件
#include <torch/csrc/distributed/c10d/PrefixStore.hpp>
#include <torch/csrc/distributed/c10d/TCPStore.hpp>

// 定义常量
constexpr int64_t kShortStoreTimeoutMillis = 100;
constexpr int defaultTimeout = 20;

// 创建 TCP 服务器的函数
c10::intrusive_ptr<c10d::TCPStore> _createServer(
    bool useLibUV,
    int numWorkers = 1,
    int timeout = defaultTimeout) {
  return c10::make_intrusive<c10d::TCPStore>(
      "127.0.0.1",
      c10d::TCPStoreOptions{
          /* port */ 0,
          /* isServer */ true,
          numWorkers,
          /* waitWorkers */ false,
          /* timeout */ std::chrono::seconds(timeout),
          /* multiTenant */ false,
          /* masterListenFd */ c10::nullopt,
          /* useLibUV*/ useLibUV});
}

// 辅助测试函数，创建服务器并进行测试
void testHelper(bool useLibUV, const std::string& prefix = "") {
  constexpr auto numThreads = 16;
  constexpr auto numWorkers = numThreads + 1;

  // 创建 TCP 服务器
  auto serverTCPStore = _createServer(useLibUV, numWorkers);

  // 创建带有前缀的 PrefixStore 对象
  auto serverStore =
      c10::make_intrusive<c10d::PrefixStore>(prefix, serverTCPStore);

  // 启动服务器线程
  auto serverThread = std::thread([&serverStore, &serverTCPStore] {
    // 等待所有工作线程加入
    serverTCPStore->waitForWorkers();

    // 在服务器存储上进行基本的设置和获取操作
    c10d::test::set(*serverStore, "key0", "value0");
    c10d::test::set(*serverStore, "key1", "value1");
    c10d::test::set(*serverStore, "key2", "value2");
    c10d::test::check(*serverStore, "key0", "value0");
    c10d::test::check(*serverStore, "key1", "value1");
    c10d::test::check(*serverStore, "key2", "value2");

    // 在服务器存储上增加一个计数器
    serverStore->add("counter", 1);
    auto numKeys = serverStore->getNumKeys();
    // 预期有 5 个键，因为上面添加了 3 个键，'counter' 是由辅助线程添加的，还有协调工作线程的初始化键。
    EXPECT_EQ(numKeys, 5);

    // 检查 compareSet 函数，不检查返回值
    c10d::test::compareSet(
        *serverStore, "key0", "wrongExpectedValue", "newValue");
    c10d::test::check(*serverStore, "key0", "value0");
    c10d::test::compareSet(*serverStore, "key0", "value0", "newValue");
    c10d::test::check(*serverStore, "key0", "newValue");

    // 尝试删除一个键，并验证删除成功
    auto delSuccess = serverStore->deleteKey("key0");
    EXPECT_TRUE(delSuccess);

    // 尝试删除一个不存在的键，并验证删除失败
    auto delFailure = serverStore->deleteKey("badKeyName");
    EXPECT_FALSE(delFailure);

    // 获取当前键的数量，并验证为 4
    numKeys = serverStore->getNumKeys();
    EXPECT_EQ(numKeys, 4);

    // 设置短超时时间
    auto timeout = std::chrono::milliseconds(kShortStoreTimeoutMillis);
    serverStore->setTimeout(timeout);
  // 对 serverStore->get("key0") 的调用预期会抛出 c10::Error 异常，这里使用 EXPECT_THROW 进行断言

  // 创建线程容器用于并发测试
  std::vector<std::thread> threads;

  // 定义迭代次数
  constexpr auto numIterations = 1000;

  // 定义信号量用于线程同步
  c10d::test::Semaphore sem1, sem2;

  // 设置 TCPStore 的选项
  c10d::TCPStoreOptions opts{};
  opts.port = serverTCPStore->getPort();
  opts.numWorkers = numWorkers;

  // 创建多个客户端 TCPStore 和 PrefixStore 实例
  std::vector<c10::intrusive_ptr<c10d::TCPStore>> clientTCPStores;
  std::vector<c10::intrusive_ptr<c10d::PrefixStore>> clientStores;
  for (const auto i : c10::irange(numThreads)) {
    clientTCPStores.push_back(
        c10::make_intrusive<c10d::TCPStore>("127.0.0.1", opts));
    clientStores.push_back(
        c10::make_intrusive<c10d::PrefixStore>(prefix, clientTCPStores[i]));
  }

  // 预期的计数器结果，作为字符串保存
  std::string expectedCounterRes =
      std::to_string(numThreads * numIterations + 1);

  // 创建多个线程进行并发操作
  for (const auto i : c10::irange(numThreads)) {
    threads.emplace_back(
        std::thread([=, &sem1, &sem2, &clientStores, &expectedCounterRes] {
          // 每个线程循环 numIterations 次，在 clientStore 上增加计数器值
          for (C10_UNUSED const auto j : c10::irange(numIterations)) {
            clientStores[i]->add("counter", 1);
          }

          // 每个线程设置和获取其客户端存储上的键值对
          std::string key = "thread_" + std::to_string(i);
          for (const auto j : c10::irange(numIterations)) {
            std::string val = "thread_val_" + std::to_string(j);
            c10d::test::set(*clientStores[i], key, val);
            c10d::test::check(*clientStores[i], key, val);
          }

          // 发送信号量1，等待信号量2的响应
          sem1.post();
          sem2.wait();

          // 检查计数器结果
          c10d::test::check(*clientStores[i], "counter", expectedCounterRes);

          // 检查其他线程写入的数据
          for (const auto j : c10::irange(numThreads)) {
            if (j == i) {
              continue;
            }
            std::string key = "thread_" + std::to_string(i);
            std::string val = "thread_val_" + std::to_string(numIterations - 1);
            c10d::test::check(*clientStores[i], key, val);
          }
        }));
  }

  // 等待所有线程的信号量1
  sem1.wait(numThreads);

  // 发送信号量2给所有线程
  sem2.post(numThreads);

  // 等待所有线程的结束
  for (auto& thread : threads) {
    thread.join();
  }

  // 等待服务器线程的结束
  serverThread.join();

  // 清空存储以测试客户端断开连接不会关闭服务器存储
  clientStores.clear();
  clientTCPStores.clear();

  // 检查计数器是否有预期值
  c10d::test::check(*serverStore, "counter", expectedCounterRes);

  // 检查主线程中每个线程写入的数据
  for (const auto i : c10::irange(numThreads)) {
    std::string key = "thread_" + std::to_string(i);
    std::string val = "thread_val_" + std::to_string(numIterations - 1);
    c10d::test::check(*serverStore, key, val);
  }
}

// 测试用例：TCPStoreTest.testHelper 的单元测试
TEST(TCPStoreTest, testHelper) {
  // 调用 testHelper 函数进行测试，传入参数 false
  testHelper(false);
}

// 测试用例：TCPStoreTest.testHelperUV 的单元测试
TEST(TCPStoreTest, testHelperUV) {
  // 调用 testHelper 函数进行测试，传入参数 true
  testHelper(true);
}

// 测试用例：TCPStoreTest.testHelperPrefix 的单元测试
TEST(TCPStoreTest, testHelperPrefix) {
  // 调用 testHelper 函数进行测试，传入参数 false 和自定义前缀 "testPrefix"
  testHelper(false, "testPrefix");
}

// 测试用例：TCPStoreTest.testHelperPrefixUV 的单元测试
TEST(TCPStoreTest, testHelperPrefixUV) {
  // 调用 testHelper 函数进行测试，传入参数 true 和自定义前缀 "testPrefix"
  testHelper(true, "testPrefix");
}

// 测试用例：TCPStoreTest.testCleanShutdown 的单元测试
TEST(TCPStoreTest, testCleanShutdown) {
  int numWorkers = 2;

  // 创建一个服务器端的 TCPStore 对象，设置参数并进行测试
  auto serverTCPStore = std::make_unique<c10d::TCPStore>(
      "127.0.0.1",
      0,
      numWorkers,
      true,  // isServer 设置为 true
      std::chrono::seconds(defaultTimeout),
      /* wait */ false);

  // 在 serverTCPStore 上设置一个键值对 "key" -> "val"
  c10d::test::set(*serverTCPStore, "key", "val");

  // 创建一个客户端的 TCPStore 对象，设置参数并进行测试
  auto clientTCPStore = c10::make_intrusive<c10d::TCPStore>(
      "127.0.0.1",
      c10d::TCPStoreOptions{
          /* port */ serverTCPStore->getPort(),
          /* isServer */ false,
          numWorkers,
          /* waitWorkers */ false,
          /* timeout */ std::chrono::seconds(defaultTimeout)});

  // 在 clientTCPStore 上获取键 "key" 的值
  clientTCPStore->get("key");

  // 创建一个客户端线程，用于测试在客户端请求期间服务器端关闭的情况
  auto clientThread = std::thread([&clientTCPStore] {
    // 期望在客户端获取无效键 "invalid_key" 时抛出 c10::DistNetworkError 异常
    EXPECT_THROW(clientTCPStore->get("invalid_key"), c10::DistNetworkError);
  });

  // 在客户端请求期间开始关闭服务器端
  serverTCPStore = nullptr;

  // 等待客户端线程执行结束
  clientThread.join();
}

// 测试用例：TCPStoreTest.testLibUVPartialRead 的单元测试
TEST(TCPStoreTest, testLibUVPartialRead) {
  int numWorkers = 2; // 线程 0 创建服务器和客户端

  // 设置服务器端的 TCPStoreOptions 参数，包括使用 LibUV
  c10d::TCPStoreOptions server_opts{
      0,
      true, // is master 设置为 true
      numWorkers,
      false, // 不等待，否则客户端线程不会启动
      std::chrono::seconds(defaultTimeout)};
  server_opts.useLibUV = true;

  // 创建服务器端的 TCPStore 对象并进行测试
  auto serverTCPStore =
      std::make_unique<c10d::TCPStore>("127.0.0.1", server_opts);

  // 设置客户端的 TCPStoreOptions 参数，包括使用 LibUV
  c10d::TCPStoreOptions client_opts{
      serverTCPStore->getPort(),
      false, // is master 设置为 false
      numWorkers,
      false, // 等待 workers 设置为 false
      std::chrono::seconds(defaultTimeout)};
  client_opts.useLibUV = true;

  // 创建客户端的 TCPStore 对象并进行测试
  auto clientTCPStore =
      c10::make_intrusive<c10d::TCPStore>("127.0.0.1", client_opts);

  // 创建客户端线程，进行部分读取的 LibUV 测试
  auto clientThread = std::thread([&clientTCPStore] {
    std::string keyPrefix(
        "/default_pg/0//b7dc24de75e482ba2ceb9f9ee20732c25c0166d8//cuda//");
    std::string value("v");
    std::vector<uint8_t> valueBuf(value.begin(), value.end());

    // 将 store->set(key, valueBuf) 拆分为两个请求
    for (int i = 0; i < 10; ++i) {
      std::string key = keyPrefix + std::to_string(i);
      clientTCPStore->_splitSet(key, valueBuf);

      // 在服务器端检查结果
      c10d::test::check(*clientTCPStore, key, "v");
    }
  });

  // 等待客户端线程执行结束
  clientThread.join();
}
void testMultiTenantStores(bool libUV) {
  // 创建 TCPStoreOptions 对象，并设置其选项
  c10d::TCPStoreOptions opts{};
  opts.isServer = true;   // 设置为服务器模式
  opts.multiTenant = true;  // 启用多租户模式
  opts.useLibUV = libUV;   // 根据传入参数决定是否使用 LibUV

  // 在相同的端口上构建两个服务器存储实例
  auto store1 = c10::make_intrusive<c10d::TCPStore>("localhost", opts);
  auto store2 = c10::make_intrusive<c10d::TCPStore>("localhost", opts);

  // 断言两个存储实例共享同一个服务器
  c10d::test::set(*store1, "key0", "value0");  // 设置键值对到 store1
  c10d::test::check(*store2, "key0", "value0");  // 检查 store2 是否能读取相同的键值对

  // 释放第二个存储实例，并断言服务器仍然存活
  store2.reset();

  c10d::test::set(*store1, "key0", "value0");  // 再次设置键值对到 store1
  c10d::test::check(*store1, "key0", "value0");  // 检查 store1 是否能读取相同的键值对
}

TEST(TCPStoreTest, testMultiTenantStores) {
  testMultiTenantStores(false);  // 测试不使用 LibUV 的情况
}

TEST(TCPStoreTest, testMultiTenantStoresUV) {
  testMultiTenantStores(true);  // 测试使用 LibUV 的情况
}
```