# `.\pytorch\test\cpp\c10d\FileStoreTest.cpp`

```
// 包含C++标准库头文件<c10/util/irange.h>
#include <c10/util/irange.h>
// 包含自定义的测试公共头文件"StoreTestCommon.hpp"
#include "StoreTestCommon.hpp"

// 如果不是Windows平台，包含<unistd.h>头文件
#ifndef _WIN32
#include <unistd.h>
#endif

// 包含标准输入输出流头文件<iostream>
#include <iostream>
// 包含线程库头文件<thread>
#include <thread>

// 包含Google测试框架的头文件<gtest/gtest.h>
#include <gtest/gtest.h>

// 包含分布式库中的文件存储类头文件<FileStore.hpp>
#include <torch/csrc/distributed/c10d/FileStore.hpp>
// 包含分布式库中的前缀存储类头文件<PrefixStore.hpp>
#include <torch/csrc/distributed/c10d/PrefixStore.hpp>

// 如果是Windows平台，定义tmppath函数，返回自动生成的临时文件路径
#ifdef _WIN32
std::string tmppath() {
  return c10d::test::autoGenerateTmpFilePath();
}
// 如果不是Windows平台，定义tmppath函数，根据环境变量TMPDIR或默认/tmp目录生成临时文件路径
#else
std::string tmppath() {
  const char* tmpdir = getenv("TMPDIR");
  if (tmpdir == nullptr) {
    tmpdir = "/tmp";
  }

  // 创建临时文件名模板
  std::vector<char> tmp(256);
  auto len = snprintf(tmp.data(), tmp.size(), "%s/testXXXXXX", tmpdir);
  tmp.resize(len);

  // 创建临时文件
  auto fd = mkstemp(&tmp[0]);
  if (fd == -1) {
    throw std::system_error(errno, std::system_category());
  }
  close(fd);

  // 返回临时文件路径
  return std::string(tmp.data(), tmp.size());
}
#endif

// 定义测试函数testGetSet，测试给定路径和前缀的存储操作
void testGetSet(std::string path, std::string prefix = "") {
  // 在FileStore上进行基本的Set/Get操作
  {
    // 创建FileStore对象
    auto fileStore = c10::make_intrusive<c10d::FileStore>(path, 2);
    // 创建带前缀的PrefixStore对象
    c10d::PrefixStore store(prefix, fileStore);
    
    // 设置键值对 "key0"->"value0", "key1"->"value1", "key2"->"value2"
    c10d::test::set(store, "key0", "value0");
    c10d::test::set(store, "key1", "value1");
    c10d::test::set(store, "key2", "value2");
    
    // 检查键 "key0", "key1", "key2" 的值分别为 "value0", "value1", "value2"
    c10d::test::check(store, "key0", "value0");
    c10d::test::check(store, "key1", "value1");
    c10d::test::check(store, "key2", "value2");
    
    // 获取FileStore中存储的键值对数量
    auto numKeys = fileStore->getNumKeys();
    // 断言FileStore中键值对的数量为4
    EXPECT_EQ(numKeys, 4);

    // 使用compareSet方法，不检查返回值，修改 "key0" 的值
    c10d::test::compareSet(store, "key0", "wrongExpectedValue", "newValue");
    // 检查 "key0" 的值未发生变化
    c10d::test::check(store, "key0", "value0");
    // 再次使用compareSet方法，将 "key0" 的值修改为 "newValue"
    c10d::test::compareSet(store, "key0", "value0", "newValue");
    // 检查 "key0" 的值是否为 "newValue"
    c10d::test::check(store, "key0", "newValue");

    // 删除键 "key1"
    c10d::test::deleteKey(store, "key1");
    // 获取FileStore中当前的键值对数量
    numKeys = fileStore->getNumKeys();
    // 断言FileStore中键值对的数量为3
    EXPECT_EQ(numKeys, 3);
    // 检查 "key0" 的值是否为 "newValue"
    c10d::test::check(store, "key0", "newValue");
    // 检查 "key2" 的值是否为 "value2"
    c10d::test::check(store, "key2", "value2");

    // 设置键 "-key0"->"value-", 在PrefixStore中的键名前加了 "-"
    c10d::test::set(store, "-key0", "value-");
    // 检查 "key0" 的值是否为 "newValue"
    c10d::test::check(store, "key0", "newValue");
    // 检查 "-key0" 的值是否为 "value-"
    c10d::test::check(store, "-key0", "value-");
    // 获取FileStore中当前的键值对数量
    numKeys = fileStore->getNumKeys();
    // 断言FileStore中键值对的数量为4
    EXPECT_EQ(numKeys, 4);
    // 删除键 "-key0"
    c10d::test::deleteKey(store, "-key0");
    // 获取FileStore中当前的键值对数量
    numKeys = fileStore->getNumKeys();
    // 断言FileStore中键值对的数量为3
    EXPECT_EQ(numKeys, 3);
    // 检查 "key0" 的值是否为 "newValue"
    c10d::test::check(store, "key0", "newValue");
    // 检查 "key2" 的值是否为 "value2"
    c10d::test::check(store, "key2", "value2");
  }

  // 在新实例上执行get操作
  {
    // 创建FileStore对象
    auto fileStore = c10::make_intrusive<c10d::FileStore>(path, 2);
    // 创建带前缀的PrefixStore对象
    c10d::PrefixStore store(prefix, fileStore);
    // 检查 "key0" 的值是否为 "newValue"
    c10d::test::check(store, "key0", "newValue");
    // 获取FileStore中当前的键值对数量
    auto numKeys = fileStore->getNumKeys();
    // 断言FileStore中键值对的数量为4，因为仍然使用与上述存储相同的底层文件
    EXPECT_EQ(numKeys, 4);
  }
}
// 在指定路径上进行 FileStore 的压力测试，可选择添加前缀
void stressTestStore(std::string path, std::string prefix = "") {
  // 设定线程数量和迭代次数
  const auto numThreads = 4;
  const auto numIterations = 100;

  // 创建线程容器和信号量
  std::vector<std::thread> threads;
  c10d::test::Semaphore sem1, sem2;

  // 对于每个线程，启动一个新线程执行以下操作
  for (C10_UNUSED const auto i : c10::irange(numThreads)) {
    threads.emplace_back(std::thread([&] {
      // 在指定路径上创建 FileStore 对象
      auto fileStore =
          c10::make_intrusive<c10d::FileStore>(path, numThreads + 1);
      // 如果提供了前缀，则创建 PrefixStore 对象
      c10d::PrefixStore store(prefix, fileStore);
      // 释放 sem1 信号，表示线程已准备好
      sem1.post();
      // 等待 sem2 信号，同步开始迭代操作
      sem2.wait();
      // 对计数器 "counter" 进行 numIterations 次加法操作
      for (C10_UNUSED const auto j : c10::irange(numIterations)) {
        store.add("counter", 1);
      }
    }));
  }

  // 等待所有线程都准备好
  sem1.wait(numThreads);
  // 发送 sem2 信号，开始所有线程的迭代操作
  sem2.post(numThreads);
  // 等待所有线程完成
  for (auto& thread : threads) {
    thread.join();
  }

  // 检查计数器是否达到预期值
  {
    // 再次创建 FileStore 对象并创建 PrefixStore 对象
    auto fileStore = c10::make_intrusive<c10d::FileStore>(path, numThreads + 1);
    c10d::PrefixStore store(prefix, fileStore);
    // 期望的值为所有线程迭代次数的总和
    std::string expected = std::to_string(numThreads * numIterations);
    // 使用 check 函数检查计数器的值是否符合预期
    c10d::test::check(store, "counter", expected);
  }
}

class FileStoreTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // 设置临时文件路径
    path_ = tmppath();
  }

  void TearDown() override {
    // 删除临时文件
    unlink(path_.c_str());
  }

  std::string path_;
};

// 测试不带前缀的 FileStore 的 get 和 set 操作
TEST_F(FileStoreTest, testGetAndSet) {
  testGetSet(path_);
}

// 测试带有前缀的 FileStore 的 get 和 set 操作
TEST_F(FileStoreTest, testGetAndSetWithPrefix) {
  testGetSet(path_, "testPrefix");
}

// 测试不带前缀的 FileStore 的压力测试
TEST_F(FileStoreTest, testStressStore) {
  stressTestStore(path_);
}

// 测试带有前缀的 FileStore 的压力测试
TEST_F(FileStoreTest, testStressStoreWithPrefix) {
  stressTestStore(path_, "testPrefix");
}
```