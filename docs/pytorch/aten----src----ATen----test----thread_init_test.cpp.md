# `.\pytorch\aten\src\ATen\test\thread_init_test.cpp`

```py
// 包含 ATen 库中的头文件
#include <ATen/ATen.h>
// 包含 ATen 库中的并行处理相关头文件
#include <ATen/Parallel.h>
// 包含 C10 库中的 irange 实用工具头文件
#include <c10/util/irange.h>
// 包含测试基类的头文件
#include <test/cpp/tensorexpr/test_base.h>
// 包含 C++ 标准线程库头文件
#include <thread>

// 这个函数用于测试多线程环境下的并行性
// 检查线程是否能看到设置的全局线程数，以及多线程调用第一个并行结构时调度器是否会抛出异常
void test(int given_num_threads) {
  // 创建一个大小为 1000*1000 的全为 1.0 的张量
  auto t = at::ones({1000 * 1000}, at::CPU(at::kFloat));
  // 断言给定的线程数非负
  ASSERT_TRUE(given_num_threads >= 0);
  // 断言当前的线程数与给定的线程数相等
  ASSERT_EQ(at::get_num_threads(), given_num_threads);
  // 对张量 t 进行求和操作
  auto t_sum = t.sum();
  // 循环执行 1000 次
  for (C10_UNUSED const auto i : c10::irange(1000)) {
    // 将 t 的求和结果与 t_sum 相加，累加求和结果
    t_sum = t_sum + t.sum();
  }
}

// 主函数入口
int main() {
  // 初始化 ATen 库的线程数
  at::init_num_threads();

  // 设置 ATen 库的线程数为 4
  at::set_num_threads(4);
  // 调用 test 函数，测试给定 4 个线程数的情况
  test(4);

  // 创建一个新线程 t1，执行以下代码
  std::thread t1([](){
    // 在新线程中重新初始化 ATen 库的线程数
    at::init_num_threads();
    // 在新线程中调用 test 函数，测试给定 4 个线程数的情况
    test(4);
  });
  // 等待新线程 t1 执行完毕
  t1.join();

  // 如果不是使用原生并行，执行以下代码块
  #if !AT_PARALLEL_NATIVE
  // 设置 ATen 库的线程数为 5
  at::set_num_threads(5);
  // 断言当前的线程数是否为 5
  ASSERT_TRUE(at::get_num_threads() == 5);
  #endif

  // 设置 ATen 库的互操作线程数为 5
  at::set_num_interop_threads(5);
  // 断言当前的互操作线程数是否为 5
  ASSERT_EQ(at::get_num_interop_threads(), 5);
  // 预期抛出异常：设置互操作线程数为 6 时
  ASSERT_ANY_THROW(at::set_num_interop_threads(6));

  // 返回主函数执行成功
  return 0;
}
```