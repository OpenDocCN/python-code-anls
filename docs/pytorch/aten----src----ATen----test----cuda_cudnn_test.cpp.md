# `.\pytorch\aten\src\ATen\test\cuda_cudnn_test.cpp`

```
# 包含 Google Test 框架头文件
#include <gtest/gtest.h>

# 包含 PyTorch ATen 库的主头文件
#include <ATen/ATen.h>
# 包含 PyTorch ATen CUDA 上下文管理的头文件
#include <ATen/cuda/CUDAContext.h>
# 包含 PyTorch ATen cuDNN 描述符的头文件
#include <ATen/cudnn/Descriptors.h>
# 包含 PyTorch ATen cuDNN 句柄的头文件
#include <ATen/cudnn/Handle.h>

# 使用 PyTorch ATen 命名空间
using namespace at;
# 使用 PyTorch ATen native 命名空间
using namespace at::native;

# 定义 CUDNNTest 类的测试用例 CUDNNTestCUDA
TEST(CUDNNTest, CUDNNTestCUDA) {
    # 如果 CUDA 不可用，则返回
    if (!at::cuda::is_available()) return;
    
    # 设置随机种子为 123
    manual_seed(123);
}
```