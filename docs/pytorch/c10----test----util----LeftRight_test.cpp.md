# `.\pytorch\c10\test\util\LeftRight_test.cpp`

```
TEST(LeftRightTest, writesCanBeConcurrentWithReads_writeThenRead) {
  LeftRight<int> obj;  // 创建一个 LeftRight 对象，用于管理一个整数类型的数据

  std::atomic<bool> writer_running{false};  // 创建一个原子布尔变量，用于标记写操作是否正在运行
  std::atomic<bool> reader_running{false};  // 创建一个原子布尔变量，用于标记读操作是否正在运行

  std::thread writer([&]() {
    obj.read([&](const int&) {  // 在 LeftRight 对象上执行读操作
      writer_running = true;    // 设置写操作正在运行的标志
      while (!reader_running.load()) {
        // 等待读操作开始
      }
    });
  });

  std::thread reader([&]() {
    // 先运行写操作，再运行读操作
    while (!writer_running.load()) {
      // 等待写操作开始
    }

    obj.write([&](int&) { writer_running = true; });  // 在 LeftRight 对象上执行写操作
  });

  // 等待线程执行完成，确保不会出现死锁情况
  reader.join();
  writer.join();
}
    // 调用对象的 read 方法，传入 lambda 表达式作为参数，当读取到 int 类型数据时设置 reader_running 为 true
    obj.read([&](const int&) { reader_running = true; });
  });

  // 线程只有在两者都进入 read 函数后才会完成。
  // 如果 LeftRight 不允许并发，这可能导致死锁。
  writer.join();
  reader.join();
TEST(LeftRightTest, writesCannotBeConcurrentWithWrites) {
  // 创建 LeftRight 对象，存储 int 类型
  LeftRight<int> obj;
  // 定义两个原子布尔变量，用于控制第一个写线程的状态
  std::atomic<bool> first_writer_started{false};
  std::atomic<bool> first_writer_finished{false};

  // 启动第一个写线程
  std::thread writer1([&]() {
    // 在 LeftRight 对象上执行写操作
    obj.write([&](int&) {
      // 标记第一个写操作已开始
      first_writer_started = true;
      // 线程休眠 50 毫秒，模拟写操作的耗时
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      // 标记第一个写操作已完成
      first_writer_finished = true;
    });
  });

  // 启动第二个写线程
  std::thread writer2([&]() {
    // 等待第一个写线程开始
    while (!first_writer_started.load()) {
    }

    // 在 LeftRight 对象上执行写操作
    obj.write([&](int&) {
      // 检查第一个写操作是否已经完成
      EXPECT_TRUE(first_writer_finished.load());
    });
  });

  // 等待两个写线程执行结束
  writer1.join();
  writer2.join();
}

namespace {
class MyException : public std::exception {};
} // namespace

TEST(LeftRightTest, whenReadThrowsException_thenThrowsThrough) {
  // 创建 LeftRight 对象，存储 int 类型
  LeftRight<int> obj;

  // 断言读操作抛出 MyException 异常
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  EXPECT_THROW(obj.read([](const int&) { throw MyException(); }), MyException);
}

TEST(LeftRightTest, whenWriteThrowsException_thenThrowsThrough) {
  // 创建 LeftRight 对象，存储 int 类型
  LeftRight<int> obj;

  // 断言写操作抛出 MyException 异常
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  EXPECT_THROW(obj.write([](int&) { throw MyException(); }), MyException);
}

TEST(
    LeftRightTest,
    givenInt_whenWriteThrowsExceptionOnFirstCall_thenResetsToOldState) {
  // 创建 LeftRight 对象，存储 int 类型
  LeftRight<int> obj;

  // 对对象执行第一次写操作
  obj.write([](int& obj) { obj = 5; });

  // 断言写操作抛出 MyException 异常，同时检查对象恢复到旧状态
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  EXPECT_THROW(
      obj.write([](int& obj) {
        obj = 6;
        throw MyException();
      }),
      MyException);

  // 检查读取对象返回旧值
  int read = obj.read([](const int& obj) { return obj; });
  EXPECT_EQ(5, read);

  // 检查后台副本也包含相同的旧值
  obj.write([](int&) {}); // 这将切换到后台副本
  read = obj.read([](const int& obj) { return obj; });
  EXPECT_EQ(5, read);
}

// 注意：每次写操作在前台和后台副本上都执行两次。我们需要测试抛出异常的情况是否能正确处理。
TEST(
    LeftRightTest,
    givenInt_whenWriteThrowsExceptionOnSecondCall_thenKeepsNewState) {
  // 创建 LeftRight 对象，存储 int 类型
  LeftRight<int> obj;

  // 对对象执行第一次写操作，设置值为 5
  obj.write([](int& obj) { obj = 5; });
  bool write_called = false;

  // 断言写操作抛出 MyException 异常，同时检查对象保持新状态
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  EXPECT_THROW(
      obj.write([&](int& obj) {
        obj = 6;
        if (write_called) {
          // 这是写回调函数第二次执行
          throw MyException();
        } else {
          write_called = true;
        }
      }),
      MyException);

  // 检查读取对象返回新值
  int read = obj.read([](const int& obj) { return obj; });
  EXPECT_EQ(6, read);

  // 检查后台副本也包含相同的新值
  obj.write([](int&) {}); // 这将切换到后台副本
  read = obj.read([](const int& obj) { return obj; });
  EXPECT_EQ(6, read);
}
TEST(LeftRightTest, givenVector_whenWriteThrowsException_thenResetsToOldState) {
  // 创建 LeftRight 对象，用于存储 vector<int> 类型数据
  LeftRight<vector<int>> obj;

  // 调用 write 方法，向 obj 写入数据
  obj.write([](vector<int>& obj) { obj.push_back(5); });

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  // 期望抛出 MyException 异常，写入操作将添加一个新元素并抛出异常
  EXPECT_THROW(
      obj.write([](vector<int>& obj) {
        obj.push_back(6);
        throw MyException();
      }),
      MyException);

  // 检查读取数据是否返回旧值
  vector<int> read = obj.read([](const vector<int>& obj) { return obj; });
  EXPECT_EQ((vector<int>{5}), read);

  // 检查后台副本中的更改是否也存在
  obj.write([](vector<int>&) {}); // 这会切换到后台副本
  read = obj.read([](const vector<int>& obj) { return obj; });
  EXPECT_EQ((vector<int>{5}), read);
}
```