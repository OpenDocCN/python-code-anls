# `.\pytorch\aten\src\ATen\test\test_parallel.cpp`

```
#include <gtest/gtest.h> // 引入 Google Test 框架的头文件

#include <ATen/ATen.h> // 引入 PyTorch ATen 库的头文件
#include <ATen/DLConvertor.h>
#include <ATen/Parallel.h>
#include <ATen/ParallelFuture.h>

#include <iostream> // 标准输入输出流
// NOLINTNEXTLINE(modernize-deprecated-headers)
#include <string.h> // C 字符串操作函数头文件
#include <sstream> // 字符串流
#if AT_MKL_ENABLED()
#include <mkl.h> // 如果 MKL 可用，则引入 MKL 头文件
#include <thread> // C++ 线程库
#endif

struct NumThreadsGuard {
  int old_num_threads_;
  NumThreadsGuard(int nthreads) { // 构造函数，保存当前线程数并设置新的线程数
    old_num_threads_ = at::get_num_threads();
    at::set_num_threads(nthreads);
  }

  ~NumThreadsGuard() { // 析构函数，恢复原有的线程数
    at::set_num_threads(old_num_threads_);
  }
};

using namespace at; // 使用 PyTorch ATen 命名空间

TEST(TestParallel, TestParallel) { // 并行测试用例
  manual_seed(123); // 设置随机数种子
  NumThreadsGuard guard(1); // 设置线程数为1

  Tensor a = rand({1, 3}); // 创建大小为1x3的随机张量
  a[0][0] = 1; // 修改张量元素
  a[0][1] = 0;
  a[0][2] = 0;
  Tensor as = rand({3}); // 创建大小为3的随机张量
  as[0] = 1; // 修改张量元素
  as[1] = 0;
  as[2] = 0;
  ASSERT_TRUE(a.sum(0).equal(as)); // 断言：张量按维度0求和结果与预期的张量相等
}

TEST(TestParallel, NestedParallel) { // 嵌套并行测试用例
  Tensor a = ones({1024, 1024}); // 创建大小为1024x1024的全1张量
  auto expected = a.sum(); // 计算张量的总和
  // 检查在并行块中调用 sum() 是否得到相同结果
  at::parallel_for(0, 10, 1, [&](int64_t begin, int64_t end) {
    if (begin == 0) {
      ASSERT_TRUE(a.sum().equal(expected)); // 断言：在并行块内计算的和与预期相等
    }
  });
}

#ifdef TH_BLAS_MKL
TEST(TestParallel, LocalMKLThreadNumber) { // 本地 MKL 线程数测试用例
  auto master_thread_num = mkl_get_max_threads(); // 获取主线程数
  auto f = [](int nthreads){
    set_num_threads(nthreads); // 设置线程数
  };
  std::thread t(f, 1); // 创建新线程，设置线程数为1
  t.join(); // 等待线程结束
  ASSERT_EQ(master_thread_num, mkl_get_max_threads()); // 断言：线程数未改变
}
#endif

TEST(TestParallel, NestedParallelThreadId) { // 嵌套并行线程 ID 测试用例
  // 检查嵌套并行块内的线程 ID 是否正确
  at::parallel_for(0, 10, 1, [&](int64_t begin, int64_t end) {
    at::parallel_for(0, 10, 1, [&](int64_t begin, int64_t end) {
      // 嵌套并行区域在单一线程上执行
      ASSERT_EQ(begin, 0); // 断言：开始索引为0
      ASSERT_EQ(end, 10); // 断言：结束索引为10

      // 线程 ID 反映内部并行区域
      ASSERT_EQ(at::get_thread_num(), 0); // 断言：线程 ID 为0
    });
  });

  // 检查并行约减中线程 ID + 1 应为1
  at::parallel_for(0, 10, 1, [&](int64_t begin, int64_t end) {
    auto num_threads =
      at::parallel_reduce(0, 10, 1, 0, [&](int64_t begin, int64_t end, int ident) {
        // 线程 ID + 1 应为1
        return at::get_thread_num() + 1;
      }, std::plus<>{}); // 累加操作
    ASSERT_EQ(num_threads, 1); // 断言：线程数为1
  });
}

TEST(TestParallel, Exceptions) { // 异常测试用例
  // 并行情况
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  ASSERT_THROW(
    at::parallel_for(0, 10, 1, [&](int64_t begin, int64_t end) {
      throw std::runtime_error("exception"); // 抛出运行时异常
    }),
    std::runtime_error); // 断言：预期抛出运行时异常

  // 非并行情况
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  ASSERT_THROW(
    at::parallel_for(0, 1, 1000, [&](int64_t begin, int64_t end) {
      throw std::runtime_error("exception"); // 抛出运行时异常
    }),
    std::runtime_error); // 断言：预期抛出运行时异常
}

TEST(TestParallel, IntraOpLaunchFuture) { // IntraOp 启动未来测试用例
  int v1 = 0;
  int v2 = 0;

  auto fut1 = at::intraop_launch_future([&v1](){ // 启动 IntraOp 未来任务
    v1 = 1; // 修改变量
  });

  auto fut2 = at::intraop_launch_future([&v2](){
    // 未完成的部分，继续下一行代码的补充
    v2 = 2;
  });



    # 设置变量 v2 的值为 2
    v2 = 2;
  });



  fut1->wait();
  fut2->wait();



  # 等待 fut1 和 fut2 所代表的两个异步操作完成
  fut1->wait();
  fut2->wait();



  ASSERT_TRUE(v1 == 1 && v2 == 2);



  # 断言确保 v1 的值为 1 并且 v2 的值为 2
  ASSERT_TRUE(v1 == 1 && v2 == 2);
}



# 这行代码是一个代码块的结束标记，用于结束一个代码段或函数的定义。
```