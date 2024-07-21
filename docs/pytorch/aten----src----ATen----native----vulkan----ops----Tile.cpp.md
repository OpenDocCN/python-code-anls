# `.\pytorch\aten\src\ATen\native\vulkan\ops\Tile.cpp`

```
// 包含 Vulkan 相关的通用头文件
#include <ATen/native/vulkan/ops/Common.h>

// 根据是否定义了 AT_PER_OPERATOR_HEADERS 决定包含的头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/repeat.h>
#endif

// 包含 Vulkan 相关的工具函数头文件
#include <ATen/native/vulkan/ops/Utils.h>

// 包含 Torch 库的头文件
#include <torch/library.h>

// 定义 at 命名空间
namespace at {
  // 定义 native 命名空间
  namespace native {
    // 定义 vulkan 命名空间
    namespace vulkan {
      // 定义 ops 命名空间
      namespace ops {
        // 匿名命名空间，用于内部函数和变量的封装
        namespace {

          // 使用 api::utils 命名空间
          using namespace api::utils;

          // 定义 tile 函数，接受 Tensor 和重复次数 repeats
          Tensor tile(const Tensor& self, const IntArrayRef repeats) {
            // 计算 self 的维度与 repeats 的差异
            const int64_t size_diff = self.dim() - static_cast<int64_t>(repeats.size());
            // 如果 size_diff 大于 0，说明 self 的维度比 repeats 的维度大
            if (size_diff > 0) {
              // 创建一个新的重复次数向量 new_repeats，前面补充 1 以匹配 self 的维度
              std::vector<int64_t> new_repeats(size_diff, 1);
              // 将原始 repeats 的元素添加到 new_repeats 中
              for (const auto i : c10::irange(repeats.size())) {
                new_repeats.emplace_back(repeats[i]);
              }
              // 使用新的重复次数向量进行 repeat 操作
              return self.repeat(IntArrayRef(new_repeats));
            }
            // 如果 size_diff <= 0，直接使用原始的 repeats 进行 repeat 操作
            return self.repeat(repeats);
          }

          // 当定义了 USE_VULKAN_API 时，注册 Vulkan 版本的 aten::tile 函数实现
          #ifdef USE_VULKAN_API
          TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
            m.impl(TORCH_SELECTIVE_NAME("aten::tile"), TORCH_FN(tile));
          }
          #endif /* USE_VULKAN_API */

        } // namespace
      } // namespace ops
    } // namespace vulkan
  } // namespace native
} // namespace at
```