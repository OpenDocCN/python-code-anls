# `.\pytorch\aten\src\ATen\native\vulkan\ops\Shape.cpp`

```
// 引入 ATen Vulkan 相关的头文件
#include <ATen/InferSize.h>
#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <torch/library.h>

// 定义 ATen 命名空间
namespace at {
  // 定义 native 命名空间
  namespace native {
    // 定义 Vulkan 命名空间
    namespace vulkan {
      // 定义 ops 命名空间

      // 实现 Tensor 视图操作的内部函数
      Tensor view_internal(const Tensor& self_arg, const IntArrayRef shape) {
        // 获取 Vulkan API 的上下文
        api::Context* const context = api::context();

        // 将输入的 Tensor 转换为 Vulkan Tensor 或者创建一个 Vulkan Tensor
        Tensor self = self_arg.is_vulkan() ? self_arg : self_arg.vulkan();
        vTensor& v_self = convert(self);

        // 推断输出的 Tensor 大小
        at::DimVector inferred_size = at::infer_size_dv(shape, self.numel());
        IntArrayRef output_size(inferred_size);

        // 创建 Vulkan Tensor 作为输出
        vTensor v_output{
            context,
            output_size.vec(),
            v_self.dtype(),
        };

        // 如果输入 Tensor 是量化的，则设置输出 Tensor 也是量化的
        if (v_self.is_quantized()) {
          v_output.set_is_quantized();
          v_output.set_scale(v_self.get_scale());
          v_output.set_zero_point(v_self.get_zero_point());
        }

        // 在 Vulkan API 上下文中创建一个存储缓冲区
        api::StorageBuffer buffer(context, api::kFloat, v_self.gpu_numel(), true);

        // 将 Vulkan Tensor 打包到存储缓冲区中
        utils::pack_vtensor_to_staging(v_self, buffer.buffer());

        // 创建一个管线屏障以确保正确的内存访问顺序
        api::PipelineBarrier pipeline_barrier{};
        add_buffer_barrier(
            pipeline_barrier,
            buffer.buffer(),
            // 前一个访问阶段
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE,
            // 下一个访问阶段
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::READ);

        // 将存储缓冲区中的数据打包到 Vulkan Tensor 中
        utils::pack_buffer_to_vtensor(buffer.buffer(), v_output, pipeline_barrier);

        // 将 Vulkan Tensor 转换为 ATen Tensor 并返回
        return convert(v_output);
      }

      // 对外公开的 Tensor 视图操作函数，调用内部实现
      inline Tensor view(const Tensor& self_arg, IntArrayRef shape) {
        return view_internal(self_arg, shape);
      }

      // 实现用于重塑的别名函数，调用视图操作函数
      Tensor _reshape_alias(
          const Tensor& self_arg,
          const IntArrayRef shape,
          const IntArrayRef strides) {
        return view_internal(self_arg, shape);
      }

#ifdef USE_VULKAN_API

      // 使用 Vulkan API 的 ATen 库的实现注册
      TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
        m.impl(TORCH_SELECTIVE_NAME("aten::view"), TORCH_FN(view));
        m.impl(
            TORCH_SELECTIVE_NAME("aten::_reshape_alias"), TORCH_FN(_reshape_alias));
      }

#endif /* USE_VULKAN_API */

    } // namespace ops
  } // namespace vulkan
} // namespace native
} // namespace at
```