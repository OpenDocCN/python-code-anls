# `.\pytorch\aten\src\ATen\test\test_thread_pool_guard.cpp`

```
#include <gtest/gtest.h>  // 包含 Google Test 框架的头文件

#include <caffe2/utils/threadpool/thread_pool_guard.h>  // 包含线程池的守卫类头文件
#include <caffe2/utils/threadpool/pthreadpool-cpp.h>    // 包含 pthreadpool 的 C++ 接口头文件

TEST(TestThreadPoolGuard, TestThreadPoolGuard) {
  auto threadpool_ptr = caffe2::pthreadpool_();  // 调用 pthreadpool_() 函数获取线程池指针

  ASSERT_NE(threadpool_ptr, nullptr);  // 断言线程池指针不为空
  {
    caffe2::_NoPThreadPoolGuard g1;  // 创建一个 _NoPThreadPoolGuard 对象 g1

    auto threadpool_ptr1 = caffe2::pthreadpool_();  // 再次获取线程池指针
    ASSERT_EQ(threadpool_ptr1, nullptr);  // 断言此时线程池指针为空

    {
      caffe2::_NoPThreadPoolGuard g2;  // 创建另一个 _NoPThreadPoolGuard 对象 g2

      auto threadpool_ptr2 = caffe2::pthreadpool_();  // 再次获取线程池指针
      ASSERT_EQ(threadpool_ptr2, nullptr);  // 断言此时线程池指针为空
    }

    // Guard should restore prev value (nullptr)
    auto threadpool_ptr3 = caffe2::pthreadpool_();  // 再次获取线程池指针
    ASSERT_EQ(threadpool_ptr3, nullptr);  // 断言此时线程池指针为空
  }

  // Guard should restore prev value (pthreadpool_)
  auto threadpool_ptr4 = caffe2::pthreadpool_();  // 再次获取线程池指针
  ASSERT_NE(threadpool_ptr4, nullptr);  // 断言线程池指针不为空
  ASSERT_EQ(threadpool_ptr4, threadpool_ptr);  // 断言新获取的线程池指针与之前获取的指针相同
}

TEST(TestThreadPoolGuard, TestRunWithGuard) {
  const std::vector<int64_t> array = {1, 2, 3};  // 创建一个包含整数的向量

  auto pool = caffe2::pthreadpool();  // 获取 pthreadpool 对象
  int64_t inner = 0;
  {
    // Run on same thread
    caffe2::_NoPThreadPoolGuard g1;  // 创建一个 _NoPThreadPoolGuard 对象 g1，用于禁用线程池

    auto fn = [&array, &inner](const size_t task_id) {  // 定义一个 lambda 函数 fn，用于计算 inner 的值
      inner += array[task_id];  // 将 array 中指定索引位置的值加到 inner 上
    };
    pool->run(fn, 3);  // 在线程池中执行 fn 函数，任务数为 3

    // confirm the guard is on
    auto threadpool_ptr = caffe2::pthreadpool_();  // 获取线程池指针
    ASSERT_EQ(threadpool_ptr, nullptr);  // 断言此时线程池指针为空
  }

  ASSERT_EQ(inner, 6);  // 断言 inner 的最终值为 6
}
```