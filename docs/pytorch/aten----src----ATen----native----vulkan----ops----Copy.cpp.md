# `.\pytorch\aten\src\ATen\native\vulkan\ops\Copy.cpp`

```py
// 包含 ATen 库的头文件
#include <ATen/ATen.h>
// 包含 Vulkan 操作的复制相关头文件
#include <ATen/native/vulkan/ops/Copy.h>
#include <ATen/native/vulkan/ops/Utils.h>
// 包含 Vulkan 的上下文管理头文件
#include <ATen/vulkan/Context.h>

// 命名空间定义：ATen -> native -> vulkan -> ops
namespace at {
namespace native {
namespace vulkan {
namespace ops {

// 
// 内存拷贝的实用函数
//

// 将源 Tensor 的数据复制到映射中
void memcpy_to_mapping(const Tensor& src, api::MemoryMap& dst_mapping) {
  // 根据数据类型选择相应的具体实现函数
  if (src.dtype() == at::kFloat) {
    memcpy_to_mapping_impl<float>(src, dst_mapping);
  } else if (src.dtype() == at::kHalf) {
    memcpy_to_mapping_impl<c10::Half>(src, dst_mapping);
  } else if (src.dtype() == c10::kQUInt8) {
    memcpy_to_mapping_impl<c10::quint8>(src, dst_mapping);
  } else if (src.dtype() == c10::kQInt8) {
    memcpy_to_mapping_impl<c10::qint8>(src, dst_mapping);
  } else if (src.dtype() == c10::kQInt32) {
    memcpy_to_mapping_impl<c10::qint32>(src, dst_mapping);
  } else if (src.dtype() == c10::kBool) {
    memcpy_to_mapping_uint8(src, dst_mapping);
  } else {
    // 如果数据类型无效则抛出错误
    TORCH_CHECK(
        false,
        "Invalid Data Type: expected c10::kQInt32, c10::kQInt8, c10::kQUInt8,",
        " c10::kBool, at::kHalf, or at::Float but got ",
        src.dtype());
  }
}

// 将映射中的数据复制到目标 Tensor 中
void memcpy_from_mapping(api::MemoryMap& src_mapping, Tensor& dst) {
  // 根据目标 Tensor 的数据类型选择相应的具体实现函数
  if (dst.dtype() == at::kFloat) {
    memcpy_from_mapping_impl<float>(src_mapping, dst);
  } else if (dst.dtype() == at::kHalf) {
    memcpy_from_mapping_impl<c10::Half>(src_mapping, dst);
  } else if (dst.dtype() == c10::kQUInt8) {
    memcpy_from_mapping_impl<c10::quint8>(src_mapping, dst);
  } else if (dst.dtype() == c10::kQInt8) {
    memcpy_from_mapping_impl<c10::qint8>(src_mapping, dst);
  } else if (dst.dtype() == c10::kQInt32) {
    memcpy_from_mapping_impl<c10::qint32>(src_mapping, dst);
  } else if (dst.dtype() == c10::kBool) {
    memcpy_from_mapping_bool(src_mapping, dst);
  } else {
    // 如果数据类型无效则抛出错误
    TORCH_CHECK(
        false,
        "Invalid Data Type: expected c10::kQInt32, c10::kQInt8, c10::kQUInt8,",
        " c10::kBool, at::kHalf or at::Float but got ",
        dst.dtype());
  }
}

//
// CPU <-> GPU 的复制实现 (这些函数使用传输命令)
//

// 将 CPU 上的 Tensor 数据复制到 Vulkan 上的 vTensor 中
void transfer_cpu_to_vulkan(const Tensor& src, vTensor& v_dst) {
  // 获取 Vulkan 的上下文
  api::Context* const context = api::context();

  // 将源 Tensor 转换为与 vTensor 纹理格式对应的数据类型，以保证复制时的字节对齐性
  Tensor src_nc4hw =
      utils::nchw_to_nc4hw(src).to(convert_dtype(v_dst.texture_dtype()));

  // 创建一个存储缓冲区，用于临时存储数据
  api::StorageBuffer staging(context, v_dst.texture_dtype(), v_dst.gpu_numel());

  // 将数据复制到临时缓冲区中
  {
    // 创建内存映射，以写入方式访问
    api::MemoryMap mapping(staging.buffer(), api::MemoryAccessType::WRITE);
    mapping.invalidate();

    // 调用 memcpy_to_mapping 函数，将数据复制到映射中
    memcpy_to_mapping(src_nc4hw, mapping);
  }

  // 创建管线障碍，用于确保缓冲区的正确使用
  api::PipelineBarrier pipeline_barrier{};
  // 调用 copy_buffer_to_vtensor 函数，将数据从缓冲区复制到 vTensor 中
  utils::copy_buffer_to_vtensor(staging.buffer(), v_dst, pipeline_barrier);
}
void transfer_vulkan_to_cpu(vTensor& v_src, Tensor& dst) {
  api::Context* const context = api::context();

  // 创建临时张量以接收复制的 NC4HW 数据
  at::Tensor dst_tmp = utils::create_staging_tensor(v_src);

  // 创建用于存储的缓冲区，使用 Vulkan API 的数据类型和大小
  api::StorageBuffer staging(context, v_src.texture_dtype(), v_src.gpu_numel());

  // 获取 Vulkan 环境的同步对象
  api::VulkanFence fence = context->fences().get_fence();

  {
    // 获取 Vulkan 上下文的互斥锁，确保在与 GPU 同步时不允许其他线程录制调度
    std::unique_lock<std::mutex> context_lock(context->dispatch_lock());

    // 准备管线屏障对象
    api::PipelineBarrier pipeline_barrier{};
    // 将 v_src 数据复制到缓冲区
    utils::copy_vtensor_to_buffer(
        v_src, staging.buffer(), pipeline_barrier, fence.get_submit_handle());

    // 等待 GPU 操作完成
    fence.wait();

    // 刷新 Vulkan 上下文
    context->flush();
    // 在退出作用域时释放 cmd_mutex_
  }

  // 将数据从缓冲区复制回 CPU 张量
  {
    // 创建内存映射对象，以便访问缓冲区数据
    api::MemoryMap mapping(staging.buffer(), api::MemoryAccessType::READ);
    // 使映射无效，以便可以安全地读取数据
    mapping.invalidate();

    // 从映射中复制数据到 dst_tmp 张量
    memcpy_from_mapping(mapping, dst_tmp);
  }

  // 归还 Vulkan 环境的同步对象
  context->fences().return_fence(fence);

  // 将 dst_tmp 转换为 NCHW 格式并赋值给 dst 张量
  dst = utils::nc4hw_to_nchw(dst_tmp, v_src.sizes())
            .to(convert_dtype(v_src.dtype()));
}

void transfer_vulkan_to_vulkan(vTensor& src, vTensor& dst) {
  api::Context* const context = api::context();

  // 准备管线屏障对象
  api::PipelineBarrier pipeline_barrier{};

  // 在 Vulkan 上下文中提交图像间的复制操作
  context->submit_copy<api::VulkanImage, api::VulkanImage>(
      // 管线屏障
      pipeline_barrier,
      // 源图像
      src.image(pipeline_barrier, api::PipelineStage::TRANSFER),
      // 目标图像
      dst.image(
          pipeline_barrier,
          api::PipelineStage::TRANSFER,
          api::MemoryAccessType::WRITE),
      // 复制详情
      src.extents(),
      {0u, 0u, 0u},
      {0u, 0u, 0u},
      // 同步句柄
      VK_NULL_HANDLE);
}

//
// CPU <-> GPU copy implementations (these functions use compute shaders)
//

void pack_cpu_to_vulkan(const Tensor& src, vTensor& dst) {
  api::Context* const context = api::context();

  // 确保 src 张量以其内存格式是连续的
  Tensor src_contig = src.contiguous(src.suggest_memory_format());

  // 注意：在下面的存储缓冲区中强制使用 float 数据类型。
  // 原因是 nchw_to_image 和 image_to_nchw 着色器要求输入为 float 类型。
  // GLSL/Vulkan 不原生支持 16 位算术类型，因此目前为计算着色器创建的存储缓冲区必须定义 float 作为基础数据类型。
  api::StorageBuffer staging(context, api::kFloat, dst.gpu_numel());
  {
    // 创建内存映射对象，以便写入缓冲区数据
    api::MemoryMap mapping(staging.buffer(), api::MemoryAccessType::WRITE);

    // 如果 src 的数据类型是 at::kHalf，则首先将其转换为 32 位浮点数。
    // 这是因为 nchw_to_image 着色器使用的是 float 类型
    if (src.dtype() == at::kHalf) {
      // 实施转换
      convert_half_to_float(mapping, src_contig);
    } else {
      // 直接复制数据到映射
      memcpy_to_mapping(mapping, src_contig);
    }
  }
}
    // 如果源张量的数据类型是半精度（at::kHalf），则将其转换为单精度（at::kFloat）
    if (src.dtype() == at::kHalf) {
      // 将转换后的连续内存块（src_contig.to(at::kFloat)）复制到映射（mapping）中
      memcpy_to_mapping(src_contig.to(at::kFloat), mapping);
    } else {
      // 否则直接将连续内存块（src_contig）复制到映射（mapping）中
      memcpy_to_mapping(src_contig, mapping);
    }
  }
  // 调用工具函数将缓冲区（staging.buffer()）打包到目标张量（dst）中
  utils::pack_staging_to_vtensor(staging.buffer(), dst);
}

void pack_vulkan_to_cpu(vTensor& src, Tensor& dst) {
  // 检查源张量是否为量化张量，如果是，抛出错误信息，暂不支持 Vulkan 量化张量到 CPU 的复制
  TORCH_CHECK(
      !src.is_quantized(),
      "Copy of vulkan quantized tensors to cpu is currently disabled!");
  // 获取当前 API 上下文
  api::Context* const context = api::context();

  // 根据源张量的元素数量，在 Vulkan 端创建一个存储缓冲区
  // 注意：下面使用 at::kFloat 的原因，请参考 pack_cpu_to_vulkan 中的注释。
  api::StorageBuffer staging(context, api::kFloat, src.gpu_numel());

  // 获取当前上下文的 Vulkan 栅栏对象
  api::VulkanFence fence = context->fences().get_fence();

  {
    // 引用 submit_compute_job 中的注释。在与 GPU 同步时，上下文在调用 vkQueueSubmit 和刷新上下文之间，
    // 不允许其他线程记录调度。因此，调用线程必须手动管理 cmd_mutex_。
    std::unique_lock<std::mutex> context_lock(context->dispatch_lock());

    // 尝试将 Vulkan 张量数据打包到临时缓冲区 staging 中
    bool submitted_to_gpu = utils::pack_vtensor_to_staging(
        src, staging.buffer(), fence.get_submit_handle());

    // 只有在确实向 GPU 提交了工作时，才等待栅栏。否则会导致无限期挂起。
    if (submitted_to_gpu) {
      fence.wait();
    }

    // 刷新上下文状态
    context->flush();
    // 离开作用域时，cmd_mutex_ 将被释放。
  }

  // 将缓冲区中的数据复制回 CPU 张量
  {
    // 创建缓冲区的内存映射
    api::MemoryMap mapping(staging.buffer(), api::MemoryAccessType::READ);
    mapping.invalidate();

    // 如果目标张量 dst 的数据类型是 at::kHalf，先将数据复制到其浮点数版本，类似 pack_cpu_to_vulkan() 的处理方式。
    if (dst.dtype() == at::kHalf) {
      Tensor dst_float = dst.to(at::kFloat);
      memcpy_from_mapping(mapping, dst_float);
      dst = dst_float.to(at::kHalf);
    } else {
      memcpy_from_mapping(mapping, dst);
    }
  }

  // 返回栅栏对象到上下文的池中
  context->fences().return_fence(fence);
}

//
// Copy op implementations
//

Tensor& copy_(Tensor& dst, const Tensor& src) {
  // 检查张量大小是否相等
  TORCH_CHECK(
      dst.sizes() == src.sizes(), "Vulkan copy_: Tensor sizes are mismatched!");

  // X -> Vulkan
  if (at::kVulkan == dst.device().type()) {
    vTensor& v_self = convert(dst);

    // Vulkan -> Vulkan
    if (at::kVulkan == src.device().type()) {
      vTensor& v_src = convert(src);
      // 在 Vulkan 之间进行数据传输
      transfer_vulkan_to_vulkan(v_src, v_self);
    }
    // CPU -> Vulkan
    else {
      pack_cpu_to_vulkan(src, v_self);
    }
  }
  // Vulkan -> X
  else if (at::kVulkan == src.device().type()) {
    vTensor& v_src = convert(src);

    // Vulkan -> CPU
    if (dst.device().is_cpu()) {
      // 将 Vulkan 张量数据复制到 CPU 张量
      pack_vulkan_to_cpu(v_src, dst);
    } else {
      TORCH_CHECK(false, "Unsupported!");
    }
  } else {
    // 不应该走到这里，如果进入此处分支，说明源或目标张量应为 Vulkan 张量。
    TORCH_INTERNAL_ASSERT(
        false,
        "Invalid code path taken! Either the source or the destination tensor "
        "was expected to be Vulkan a tensor!  Incorrect dispatch?");
  }

  return dst;
}
// 将输入的 CPU 张量转换为 Vulkan 张量
vTensor to_vulkan(at::Tensor& src, const api::StorageType storage_type) {
  // 检查输入张量是否在 CPU 上
  TORCH_CHECK(
      src.device().type() == at::kCPU,
      "Vulkan to_vulkan(): input tensor must be a CPU tensor!")

  // 创建 Vulkan 张量对象 v_ret
  vTensor v_ret{
      api::context(),                                // 使用 Vulkan API 的上下文
      src.sizes().vec(),                             // 使用输入张量的尺寸
      convert_dtype(src.scalar_type()),              // 转换输入张量的数据类型到 Vulkan 数据类型
      storage_type,                                  // 指定存储类型
      get_gpu_memory_layout(storage_type, src.suggest_memory_format()), // 获取 GPU 内存布局
  };

  // 将 CPU 张量 src 打包到 Vulkan 张量 v_ret 中
  ops::pack_cpu_to_vulkan(src, v_ret);

  // 返回创建的 Vulkan 张量对象 v_ret
  return v_ret;
}

// 从 Vulkan 张量 v_src 转换为 CPU 张量
at::Tensor from_vulkan(vTensor& v_src) {
  at::TensorOptions opt(at::kCPU);  // 创建 CPU 张量选项
  opt = opt.dtype(convert_dtype(v_src.dtype()));  // 将 Vulkan 数据类型转换为 CPU 数据类型

  c10::MemoryFormat v_src_memory_format;  // 声明 Vulkan 张量的内存格式

  // 根据 Vulkan 张量的 GPU 内存布局选择对应的 CPU 内存格式
  switch (v_src.gpu_memory_layout()) {
    case api::GPUMemoryLayout::TENSOR_WIDTH_PACKED:
      v_src_memory_format = c10::MemoryFormat::Contiguous;
      break;
    case api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED:
      v_src_memory_format = c10::MemoryFormat::ChannelsLast;
      break;
    default:
      // 如果没有对应的内存格式，抛出异常
      TORCH_CHECK(false, "No corresponding memory format");
  }

  // 根据 Vulkan 张量 v_src 的大小创建一个空的 CPU 张量 ret，并设置其内存格式
  at::Tensor ret = at::empty(v_src.sizes(), opt).to(v_src_memory_format);

  // 将 Vulkan 张量 v_src 转换为 CPU 张量 ret
  ops::pack_vulkan_to_cpu(v_src, ret);

  // 返回转换后的 CPU 张量 ret
  return ret;
}

//
// VulkanImpl
//

// Vulkan 实现接口的具体实现类
struct VulkanImpl final : public at::vulkan::VulkanImplInterface {
  // 检查 Vulkan 是否可用
  bool is_vulkan_available() const override {
    return api::available();
  }

  // 实现 Vulkan 下的张量复制操作
  Tensor& vulkan_copy_(Tensor& self, const Tensor& src) const override {
    return vulkan::ops::copy_(self, src);
  }
};
static at::vulkan::VulkanImplRegistrar g_vulkan_impl(new VulkanImpl());

// namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
```