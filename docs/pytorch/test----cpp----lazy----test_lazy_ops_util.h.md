# `.\pytorch\test\cpp\lazy\test_lazy_ops_util.h`

```
#pragma once
// 预处理指令，确保此头文件只被包含一次

#include <gtest/gtest.h>
// 包含 Google Test 框架的头文件

#include <torch/csrc/lazy/backend/backend_device.h>
#include <torch/csrc/lazy/core/debug_util.h>
#include <torch/csrc/lazy/core/ir.h>
#include <torch/csrc/lazy/core/tensor.h>
// 包含 Torch Lazy 模块的核心头文件

#include <torch/torch.h>
// 包含 PyTorch 的主要头文件

#include <cmath>
// 包含数学函数的头文件

#include <functional>
// 包含函数对象的头文件

#include <string>
// 包含字符串操作的头文件

#include <unordered_set>
// 包含无序集合的头文件

namespace torch {
namespace lazy {

const std::unordered_set<std::string>* GetIgnoredCounters();
// 声明一个函数，返回指向无序集合字符串的常量指针

// 将 at::Tensor(device=torch::kLazy) 转换为 at::Tensor(device=torch::kCPU)
// 如果输入张量已经是 CPU 张量，则直接返回它。因为 EqualValues 和 AllClose 要求两边都是 CPU 张量。
at::Tensor ToCpuTensor(const at::Tensor& tensor);

// 辅助函数，将张量复制到指定设备上。
torch::Tensor CopyToDevice(
    const torch::Tensor& tensor,
    const torch::Device& device);

// 检查两个张量是否相等
bool EqualValues(at::Tensor tensor1, at::Tensor tensor2);

// 在不检查元素类型的情况下检查两个张量是否相等
bool EqualValuesNoElementTypeCheck(at::Tensor tensor1, at::Tensor tensor2);

// 检查两个张量的值是否接近
bool CloseValues(
    at::Tensor tensor1,
    at::Tensor tensor2,
    double rtol = 1e-5,
    double atol = 1e-8);

// 内联函数，要求两个张量的值在指定的相对和绝对容差范围内接近
static inline void AllClose(
    at::Tensor tensor,
    at::Tensor xla_tensor,
    double rtol = 1e-5,
    double atol = 1e-8) {
  EXPECT_TRUE(CloseValues(tensor, xla_tensor, rtol, atol));
}

// 内联函数，要求两个张量的值在指定的相对和绝对容差范围内接近
static inline void AllClose(
    at::Tensor tensor,
    torch::lazy::LazyTensor& xla_tensor,
    double rtol = 1e-5,
    double atol = 1e-8) {
  EXPECT_TRUE(
      CloseValues(tensor, xla_tensor.ToTensor(/*detached=*/false), rtol, atol));
}

// 内联函数，要求两个张量的值完全相等
static inline void AllEqual(at::Tensor tensor, at::Tensor xla_tensor) {
  EXPECT_TRUE(EqualValues(tensor, xla_tensor));
}

// 对每一个设备执行指定的函数
void ForEachDevice(const std::function<void(const torch::Device&)>& devfn);

// 获取张量的文本表示形式的计算图
std::string GetTensorTextGraph(at::Tensor tensor);

// 获取张量的 DOT 图形式表示的计算图
std::string GetTensorDotGraph(at::Tensor tensor);

// 获取张量的 HLO 图形式表示的计算图
std::string GetTensorHloGraph(at::Tensor tensor);

// 测试反向传播
void TestBackward(
    const std::vector<torch::Tensor>& inputs,
    const torch::Device& device,
    const std::function<torch::Tensor(const std::vector<torch::Tensor>&)>&
        testfn,
    double rtol = 1e-5,
    double atol = 1e-8,
    int derivative_level = 1);

} // namespace lazy
} // namespace torch


这段代码是一个 C++ 头文件，定义了一些函数和常量，用于处理和测试 PyTorch 中 Lazy 模块的张量操作和计算图功能。
```