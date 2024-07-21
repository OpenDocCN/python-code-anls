# `.\pytorch\aten\src\ATen\native\vulkan\ops\Stack.cpp`

```
// 引入 Vulkan 操作的共享头文件
#include <ATen/native/vulkan/ops/Common.h>

// 如果没有定义 AT_PER_OPERATOR_HEADERS，则包含标准 ATen 功能头文件，否则包含特定操作的头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/cat.h>
#include <ATen/ops/unsqueeze.h>
#endif

// 引入 C++ 标准库中的范围工具
#include <c10/util/irange.h>

// 引入 Torch 库的库头文件
#include <torch/library.h>

// 声明 at 命名空间
namespace at {
  // 声明 native 命名空间
  namespace native {
    // 声明 Vulkan 命名空间
    namespace vulkan {
      // 声明 ops 命名空间
      namespace ops {
        // 匿名命名空间用于定义私有符号
        namespace {

          // 使用 api::utils 命名空间
          using namespace api::utils;

          // 定义 Vulkan 下的 stack 函数，接受一个 tensor 列表和一个维度参数 dim
          Tensor stack(const at::TensorList tensors, const int64_t dim) {
            // 检查输入 tensor 列表不能为空
            TORCH_CHECK(!tensors.empty(), "Vulkan stack expects at least one tensor");

            // 取第一个 tensor 作为参考 tensor
            const at::Tensor& tensor = tensors[0];

            // 检查参考 tensor 的维度不超过 3
            TORCH_CHECK(
                tensor.dim() <= 3,
                "Vulkan stack only supports up to 3d tensors as input!");

            // 检查 dim 参数在有效范围内
            TORCH_CHECK(
                dim >= -tensor.dim() - 1 && dim <= tensor.dim(),
                "Vulkan stack dimension out of range expected to be in range of [",
                -tensor.dim() - 1,
                ",",
                tensor.dim(),
                "], but got ",
                dim);

            // 检查所有输入 tensor 的大小与参考 tensor 相匹配
            for (const auto& t : tensors) {
              for (const auto d : c10::irange(t.dim())) {
                TORCH_CHECK(
                    t.size(d) == tensor.size(d),
                    "Vulkan stack inputs must have matching sizes, received ",
                    t.size(d),
                    tensor.size(d));
              }
            }

            // 创建一个存放 unsqueeze 后的 tensor 的向量
            std::vector<Tensor> unsqueezed_outputs;
            // 对输入 tensor 列表中的每个 tensor 进行 unsqueeze 操作
            for (const auto& t : tensors) {
              unsqueezed_outputs.push_back(at::unsqueeze(t, dim));
            }

            // 将 unsqueeze 后的 tensor 合并成一个新的 tensor 列表
            const at::TensorList tensorList = unsqueezed_outputs;
            // 在指定维度 dim 上拼接（cat）tensor 列表中的 tensor，并返回结果
            return at::cat(tensorList, dim);
          }

          // 如果定义了 USE_VULKAN_API，则注册 Vulkan 下的 stack 实现
          #ifdef USE_VULKAN_API
          TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
            m.impl(TORCH_SELECTIVE_NAME("aten::stack"), TORCH_FN(stack));
          }
          #endif /* USE_VULKAN_API */

        } // namespace
      } // namespace ops
    } // namespace vulkan
  } // namespace native
} // namespace at
```