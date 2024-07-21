# `.\pytorch\aten\src\ATen\test\cuda_allocator_test.cpp`

```
# 包含 Google Test 框架的头文件
#include <gtest/gtest.h>

# 包含 PyTorch 的 ATen 库的头文件
#include <ATen/ATen.h>

# 包含 PyTorch 的 CUDA 缓存分配器的头文件
#include <c10/cuda/CUDACachingAllocator.h>

# 包含 ATen 库中用于测试的分配器克隆测试的头文件
#include <ATen/test/allocator_clone_test.h>

# 定义名为 AllocatorTestCUDA 的测试套件，测试 CUDA 环境下的分配器克隆
TEST(AllocatorTestCUDA, test_clone) {
    # 调用测试函数 test_allocator_clone，传入 CUDA 缓存分配器对象
    test_allocator_clone(c10::cuda::CUDACachingAllocator::get());
}
```