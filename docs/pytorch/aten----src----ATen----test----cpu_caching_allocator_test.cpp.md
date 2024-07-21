# `.\pytorch\aten\src\ATen\test\cpu_caching_allocator_test.cpp`

```
#include <gtest/gtest.h>  // 引入 Google Test 框架的头文件

#include <ATen/cpu/vec/vec.h>  // 引入 ATen 的 CPU 向量化相关头文件
#include <ATen/ATen.h>  // 引入 ATen 库的主头文件

#include <c10/mobile/CPUCachingAllocator.h>  // 引入 C10 库的移动端 CPU 缓存分配器头文件

// 定义测试用例 CPUCachingAllocatorTest.check_alloc_free
TEST(CPUCachingAllocatorTest, check_alloc_free) {
  c10::CPUCachingAllocator caching_allocator;  // 创建 CPUCachingAllocator 对象
  c10::WithCPUCachingAllocatorGuard caching_allocator_guard(
      &caching_allocator);  // 创建 CPUCachingAllocatorGuard 对象，使用缓存分配器

  at::Tensor a = at::rand({23, 23});  // 创建一个大小为 23x23 的随机张量 a
  float* data_ptr = a.data_ptr<float>();  // 获取张量 a 的数据指针
  a.reset();  // 重置张量 a

  a = at::rand({23, 23});  // 重新创建一个大小为 23x23 的随机张量 a
  // 断言前后两次获取的数据指针是相同的
  ASSERT_TRUE(data_ptr == a.data_ptr<float>());
}

// 定义测试用例 CPUCachingAllocatorTest.check_alloc_outside_free_inside
TEST(CPUCachingAllocatorTest, check_alloc_outside_free_inside) {
  c10::CPUCachingAllocator caching_allocator;  // 创建 CPUCachingAllocator 对象
  at::Tensor a = at::rand({23, 23});  // 创建一个大小为 23x23 的随机张量 a

  {
    c10::WithCPUCachingAllocatorGuard caching_allocator_guard(
        &caching_allocator);  // 创建 CPUCachingAllocatorGuard 对象，使用缓存分配器

    // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
    float* data_ptr = a.data_ptr<float>();  // 获取张量 a 的数据指针
    a.reset();  // 重置张量 a

    a = at::rand({23, 23});  // 重新创建一个大小为 23x23 的随机张量 a
  }
}

// 定义测试用例 CPUCachingAllocatorTest.check_alloc_inside_free_outside
TEST(CPUCachingAllocatorTest, check_alloc_inside_free_outside) {
  c10::CPUCachingAllocator caching_allocator;  // 创建 CPUCachingAllocator 对象
  at::Tensor a;  // 声明一个张量 a

  {
    c10::WithCPUCachingAllocatorGuard caching_allocator_guard(
        &caching_allocator);  // 创建 CPUCachingAllocatorGuard 对象，使用缓存分配器

    a = at::rand({23, 23});  // 创建一个大小为 23x23 的随机张量 a
  }

  a.reset();  // 重置张量 a
}

// 主函数，程序入口
int main(int argc, char* argv[]) {
  // 只有在 C10_MOBILE 宏定义被定义时才执行以下代码块
#ifdef C10_MOBILE
  ::testing::InitGoogleTest(&argc, argv);  // 初始化 Google Test 框架
  at::manual_seed(42);  // 设置随机数种子为 42
  return RUN_ALL_TESTS();  // 运行所有测试用例
#endif /* C10_Mobile */
}
```