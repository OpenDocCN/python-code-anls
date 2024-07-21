# `.\pytorch\aten\src\ATen\native\Copy.cpp`

```
// 定义宏，仅在编译 Torch 时启用特定方法操作符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 包含 ATen 库的 Copy.h 文件
#include <ATen/native/Copy.h>
#include <ATen/native/Copy.h>  // 重复包含同一文件（通常应避免重复包含）

// 包含 ATen 核心 Tensor 类
#include <ATen/core/Tensor.h>
// 包含 ATen 的分发机制头文件
#include <ATen/Dispatch.h>
#include <ATen/Dispatch_v2.h>
// 包含 ATen 的扩展工具函数
#include <ATen/ExpandUtils.h>
// 包含 ATen 的功能性张量封装
#include <ATen/FunctionalTensorWrapper.h>
// 包含 ATen 的张量迭代器
#include <ATen/TensorIterator.h>
// 包含 ATen 的量化模块下的复制函数
#include <ATen/native/quantized/Copy.h>
// 包含 ATen 的 MPS 模块下的复制函数
#include <ATen/native/mps/Copy.h>
// 包含 ATen 的 Vulkan 模块下的复制函数
#include <ATen/native/vulkan/ops/Copy.h>
// 包含 ATen 的张量形状操作函数
#include <ATen/native/TensorShape.h>
// 包含 ATen 的量化器模块
#include <ATen/quantized/Quantizer.h>
// 包含 ATen 的 Vulkan 上下文
#include <ATen/vulkan/Context.h>
// 包含 ATen 的 Metal 上下文
#include <ATen/metal/Context.h>
// 包含 ATen 的命名张量工具函数
#include <ATen/NamedTensorUtils.h>
// 包含 ATen 的并行计算支持
#include <ATen/Parallel.h>
// 包含 C10 的工具函数，用于范围迭代
#include <c10/util/irange.h>

// 如果未定义 AT_PER_OPERATOR_HEADERS 宏，则包含通用的 ATen 函数头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
// 否则，包含特定的 ATen 操作函数头文件
#else
#include <ATen/ops/_copy_from.h>
#include <ATen/ops/_propagate_xla_data.h>
#include <ATen/ops/_propagate_xla_data_native.h>
#include <ATen/ops/copy.h>
#include <ATen/ops/copy_native.h>
#include <ATen/ops/_foreach_copy.h>
#include <ATen/ops/_foreach_copy_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_strided.h>
#include <ATen/ops/expand_copy.h>
#endif

// 如果定义了 USE_FBGEMM 宏，则包含 FBGEMM 库的相关头文件
#ifdef USE_FBGEMM
#include <fbgemm/Fbgemm.h>
#include <fbgemm/FbgemmConvert.h>
#endif

// 使用匿名命名空间，以隐藏函数和变量的全局可见性
namespace {

// 使用 ATen 命名空间
using namespace at;

// 检查是否可以进行同类型张量的转置复制
bool copy_transpose_valid(const Tensor& self, const Tensor& src) {
  // 最小尺寸常量设定为 60 * 60
  const int MIN_SZ = 60 * 60;
  // 检查目标张量是否是连续的、源张量元素数量不为零、源张量维度为 2、特定步长条件等是否满足
  return self.is_contiguous() && src.numel() != 0 && src.dim() == 2 &&
      src.stride(0) == 1 && src.stride(1) == src.size(0) &&
      self.scalar_type() == src.scalar_type() &&
      !isBitsType(self.scalar_type()) &&
      self.sizes().equals(src.sizes()) &&
      self.is_neg() == src.is_neg() &&
      self.is_conj() == src.is_conj() &&
      self.numel() >= MIN_SZ;
}

// 定义一个特殊情况的复制函数，其中目标张量是连续的且源张量是转置的矩阵
// 这可以推广到大多数复制操作，但实现上更为复杂
void copy_same_type_transpose_(Tensor& self, const Tensor& src) {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  int64_t BLOCK_SZ;
  // 如果目标张量的标量类型是 kByte
  if (self.scalar_type() == kByte) {
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
    BLOCK_SZ = 120;  // 设置块尺寸为 120
  } else {
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
    BLOCK_SZ = 160;  // 否则设置块尺寸为 160
  }
}
    // 定义块大小为60
    BLOCK_SZ = 60;
    
    // 创建大小为BLOCK_SZ x BLOCK_SZ的张量buf，使用与self相同的选项
    Tensor buf = empty({BLOCK_SZ, BLOCK_SZ}, self.options());
    
    // 在调试模式下，断言self的大小与src的大小相等
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(self.sizes().equals(src.sizes()));
    
    // 使用self的标量类型进行调度和复制操作
    _AT_DISPATCH_CP_TYPES(self.scalar_type(), "copy_", [&] {
        // sp指向src的常量数据指针
        const scalar_t* sp = src.const_data_ptr<scalar_t>();
        // rp指向self的数据指针
        scalar_t* rp = self.data_ptr<scalar_t>();
        // bp指向buf的数据指针
        scalar_t* bp = buf.data_ptr<scalar_t>();
    
        // 获取src的行数NR和列数NC
        int64_t NR = src.size(0);
        int64_t NC = src.size(1);
    
        // 循环处理数据块，每次处理BLOCK_SZ x BLOCK_SZ的区域
        for (int64_t R = 0; R < NR; R += BLOCK_SZ) {
            for (int64_t C = 0; C < NC; C += BLOCK_SZ) {
                // spo指向当前块在src中的起始位置
                const scalar_t* spo = sp + R + C * NR;
                // rpo指向当前块在self中的起始位置
                scalar_t* rpo = rp + C + R * NC;
    
                // 计算当前块的实际大小
                int nr = std::min(NR - R, BLOCK_SZ);
                int nc = std::min(NC - C, BLOCK_SZ);
    
                // 1. 从src复制列到buf
                for (const auto c : c10::irange(nc)) {
                    memcpy(bp + c * BLOCK_SZ, spo + c * NR, nr * sizeof(scalar_t));
                }
    
                // 2. 原地转置buf
                int rc_max = std::max(nr, nc);
                int rc_min = std::min(nr, nc);
                for (const auto r : c10::irange(rc_max)) {
                    int end = std::min(r, rc_min);
                    for (const auto c : c10::irange(end)) {
                        // 交换buf中的元素以实现转置
                        scalar_t tmp = bp[r + BLOCK_SZ * c];
                        bp[r + BLOCK_SZ * c] = bp[r * BLOCK_SZ + c];
                        bp[r * BLOCK_SZ + c] = tmp;
                    }
                }
    
                // 3. 从buf复制行到self
                for (const auto r : c10::irange(nr)) {
                    memcpy(rpo + r * NC, bp + r * BLOCK_SZ, nc * sizeof(scalar_t));
                }
            }
        }
    });
} // 结束 at::native 命名空间的定义

// 检查给定设备是否被此复制实现直接支持，其他设备类型（如 XLA）可能通过覆盖 copy_ 和 _copy_from 函数来支持。
bool is_supported_device(Device device) {
  // 获取设备的类型
  DeviceType device_type = device.type();
  // 检查设备类型是否为以下几种之一：CPU、CUDA、HIP、Vulkan、Metal、MPS
  return device_type == kCPU || device_type == kCUDA || device_type == kHIP || device_type == kVulkan || device_type == kMetal || device_type == kMPS;
}

} // 结束匿名命名空间

namespace at::native {

// 实现 Tensor 的复制操作，将 src 复制到 self 中，支持非阻塞操作
static Tensor & copy_impl(Tensor & self, const Tensor & src, bool non_blocking) {
  // TODO: this should be handled during dispatch, but that's missing...
  // 检查 self 张量是否已定义
  TORCH_CHECK(self.defined(), "self is undefined");
  // 检查 src 张量是否已定义
  TORCH_CHECK(src.defined(), "src is undefined");

  // FBGeMM 内核仅支持以下情况：
  // 1. 源张量和目标张量的内存格式是连续的。
  // 2. 源张量和目标张量的设备都是 CPU。
  // 3. FP32->FP16 或 FP16->FP32 之间的 dtype 转换。
  // 此检查确保了 self 和 src 的尺寸相同，因为此代码路径不支持广播。
  // 这也防止了在复制过程中的越界访问内存，参见 fbgemm::Float16ToFloat_ref。
  // https://github.com/pytorch/pytorch/issues/88543
  #ifdef USE_FBGEMM
    if (((self.dtype() == at::kFloat && src.dtype() == at::kHalf) ||
         (self.dtype() == at::kHalf && src.dtype() == at::kFloat)) &&
        (self.device().is_cpu() && src.device().is_cpu()) &&
        ((self.is_contiguous() && src.is_contiguous()) ||
         (self.is_non_overlapping_and_dense() && self.strides() == src.strides())) &&
        (self.sizes() == src.sizes())) {
      if (src.dtype() == at::kFloat && self.dtype() == at::kHalf) {
        auto* output_ptr =
            reinterpret_cast<fbgemm::float16*>(self.data_ptr<at::Half>());
        // 如果 numel 小于 GRAIN_SIZE，使用单线程方式进行转换
        if (self.numel() < at::internal::GRAIN_SIZE) {
          fbgemm::FloatToFloat16_simd(src.const_data_ptr<float>(), output_ptr, self.numel());
        } else {
          // 使用并行方式进行大规模数据转换
          at::parallel_for(
              0,
              self.numel(),
              at::internal::GRAIN_SIZE,
              [&](int64_t begin, int64_t end) {
                fbgemm::FloatToFloat16_simd(
                    src.const_data_ptr<float>() + begin,
                    output_ptr + begin,
                    end - begin);
              });
        }
      } else {
        auto in_data = reinterpret_cast<const fbgemm::float16*>(
            src.const_data_ptr<at::Half>());
        auto* output_ptr = self.data_ptr<float>();
        // 如果 numel 小于 GRAIN_SIZE，使用单线程方式进行转换
        if (self.numel() < at::internal::GRAIN_SIZE) {
          fbgemm::Float16ToFloat_simd(in_data, output_ptr, self.numel());
        } else {
          // 使用并行方式进行大规模数据转换
          at::parallel_for(
              0,
              self.numel(),
              at::internal::GRAIN_SIZE,
              [&](int64_t begin, int64_t end) {
                fbgemm::Float16ToFloat_simd(
                    in_data + begin, output_ptr + begin, end - begin);
              });
        }
      }
      // 返回处理后的 self 张量
      return self;
  }
  #endif

  // 如果源张量与目标张量是同一个张量对象，则直接返回自身
  if (self.is_same(src)) {
    return self;
  }

  // 如果目标张量是元数据张量，直接返回自身，不进行拷贝操作
  if (self.is_meta()) {
    // 推断出目标张量的形状并进行验证，确保形状匹配
    auto shape = infer_size_symdimvector(self.sym_sizes(), src.sym_sizes());
    TORCH_CHECK(
        self.sym_sizes().equals(shape),
        "output with shape ",
        self.sym_sizes(),
        " doesn't match the broadcast shape ",
        shape);
    return self;
  }

  // 如果源张量是元数据张量，抛出错误，因为无法从元数据张量复制数据
  if (src.is_meta()) {
    TORCH_CHECK_NOT_IMPLEMENTED(false, "Cannot copy out of meta tensor; no data!")
  }

  // 当源张量或目标张量的设备不在本函数支持的设备列表中时，重新分派拷贝操作到支持的设备上
  if (!is_supported_device(src.device()) || !is_supported_device(self.device())) {
    at::_copy_from(src, self, non_blocking);
    return self;
  }

  // 如果目标张量是量化的而源张量不是量化的，调用量化拷贝函数
  if (self.is_quantized() && !src.is_quantized()) {
    return quantized_copy_from_float_(self, src);
  }

  // 如果目标张量和源张量都是量化的，验证它们的量化方案和标量类型是否一致，并设置量化器
  if (self.is_quantized() && src.is_quantized()) {
    TORCH_CHECK(self.qscheme() == src.qscheme(),
                "Quantized Copy only works with same qscheme");
    TORCH_CHECK(self.scalar_type() == src.scalar_type());
    set_quantizer_(self, src.quantizer());
  }

  // 如果目标张量不是量化的而源张量是量化的，抛出错误，因为不能从量化张量复制到非量化张量
  if (!self.is_quantized() && src.is_quantized()) {
    TORCH_CHECK(false, "Copying from quantized Tensor to non-quantized Tensor is not allowed, please use dequantize to get a float Tensor from a quantized Tensor");
  }

  // 如果目标张量或源张量的设备类型为 Vulkan
  if (self.device().type() == at::kVulkan || src.device().type() == at::kVulkan) {
  #ifdef USE_VULKAN_API
    // 使用 Vulkan API 进行张量拷贝操作
    return vulkan::ops::copy_(self, src);
  #else
    // 使用 ATen 的 Vulkan API 进行张量拷贝操作
    return at::vulkan::vulkan_copy_(self, src);
  #endif
  }

  // 如果目标张量或源张量的设备类型为 Metal，使用 Metal API 进行张量拷贝操作
  if (self.device().type() == at::kMetal || src.device().type() == at::kMetal) {
    return at::metal::metal_copy_(self, src);
  }

  // 如果目标张量和源张量是同一数据的不同视图，则直接返回自身
  const bool is_same_data = (
      self.is_alias_of(src) &&
      self.storage_offset() == src.storage_offset() &&
      self.strides().equals(src.strides()) &&
      self.sizes().equals(src.sizes()) &&
      self.scalar_type() == src.scalar_type() &&
      self.is_conj() == src.is_conj() &&
      self.is_neg() == src.is_neg()
    );
  if (is_same_data) {
    return self;
  }

  // 使用 TensorIteratorConfig 配置张量迭代器，用于复制操作
  auto iter = TensorIteratorConfig()
    .add_output(self)
    .add_const_input(src)
    .resize_outputs(false)
    .check_all_same_dtype(false)
    .check_all_same_device(false)
    .build();

  // 如果迭代器的元素数量为 0，则直接返回自身
  if (iter.numel() == 0) {
    return self;
  }

  // 确定张量迭代器的设备类型
  DeviceType device_type = iter.device_type(0);
  if (iter.device_type(1) == kCUDA) {
    device_type = kCUDA;
  } else if (iter.device_type(1) == kHIP) {
    device_type = kHIP;
  } else if (iter.device_type(1) == kMPS) {
    // 如果设备类型为 MPS，则进行相应处理
    // (此处缺少后续的代码，建议根据上下文填补)
  }
    device_type = kMPS;
  }

  // 如果需要的话，我们也可以启用这段代码来处理量化张量
  // 检查设备类型是否为 CPU，并且源张量与目标张量可以进行有效的复制和转置，并且目标张量未量化
  if (device_type == kCPU && copy_transpose_valid(self, src) && !self.is_quantized()) {
    // 调用函数来执行相同数据类型的复制和转置操作
    copy_same_type_transpose_(self, src);
    // 返回已经处理好的目标张量
    return self;
  }
#ifdef USE_MPS
  // 如果编译时定义了 USE_MPS
  // 检查张量 self 或者 src 是否位于 MPS 设备上，若是则调用 MPS 版本的复制函数
  if (self.device().type() == at::kMPS || src.device().type() == at::kMPS) {
    return at::native::mps::mps_copy_(self, src, non_blocking);
  }
#endif

// 如果 self 不是复数且不是布尔类型，并且 src 是复数类型，发出警告
if (!(self.is_complex() || self.dtype() == at::kBool) && src.is_complex()) {
  TORCH_WARN_ONCE("Casting complex values to real discards the imaginary part");
}

// 调用复制的 stub 函数，根据设备类型和迭代器执行复制操作
copy_stub(device_type, iter, non_blocking);

// 返回 self 张量本身
return self;
}

Tensor copy_meta(const Tensor& self, const Tensor& src, bool non_blocking) {
  // 必须直接使用 self()，以便可以正确地分派，如果 self 是子类
  auto r = clone_preserve_strides(self);
  // 调用 self 的 copy_ 函数来复制 src 到 r，支持非阻塞操作
  r.copy_(src, non_blocking);
  // 返回复制后的张量 r
  return r;
}

Tensor copy(const Tensor& self, const Tensor& src, bool non_blocking) {
  at::Tensor r;

  // 获取 self 的存储引用
  auto self_storage = self.unsafeGetTensorImpl()->unsafe_storage().unsafeGetStorageImpl();

  // 如果 self 没有实际存储，不能真正克隆它
  // 相反，生成一个具有正确大小/步幅的空张量，假定 copy_() 将完全用 src 的数据覆盖所有数据
  if (self_storage->nbytes() == 0) {
    // 使用 self 的大小和步幅创建一个空张量
    r = at::empty_strided(self.sizes(), self.strides(), self.options());
  } else {
    // 克隆 self，保留步幅
    r = clone_preserve_strides(self);
  }

  // 将 src 的数据复制到 r，支持非阻塞操作
  r.copy_(src, non_blocking);

  // 返回复制后的张量 r
  return r;
}

::std::vector<at::Tensor> _foreach_copy(at::TensorList self, at::TensorList src, bool non_blocking) {
  std::vector<at::Tensor> outs;
  outs.reserve(self.size());

  // 这是一个非常慢的实现，但需要直接调用上面的 copy() 核心处理当 self 有零存储时的情况
  // 这个核心实际上不应该运行，除非使用 compile(backend="aot_eager") 进行调试
  for (const auto i : c10::irange(src.size())) {
    auto curr_src = src[i];
    auto curr_self = self[i];
    // 调用 copy 函数来复制 curr_src 到 curr_self，支持非阻塞操作
    outs.push_back(at::copy(curr_self, curr_src, non_blocking));
  }

  // 返回复制后的输出向量
  return outs;
}

Tensor& copy_(Tensor& self, const Tensor& src, bool non_blocking) {
  // 计算可能的输出名称
  auto maybe_outnames = namedinference::compute_broadcast_outnames(self, src);

  {
    // 临时禁用命名保护
    NoNamesGuard guard;

    // 如果 self 是零张量，则抛出错误，零张量是不可变的
    if (self._is_zerotensor()) {
      TORCH_CHECK(false, "ZeroTensors are immutable. Please materialize the tensor using `.clone()`, if you want a mutable zero tensor.");
    }

    // 如果 src 是零张量，则将 self 设置为零张量并返回
    if (src._is_zerotensor()) {
      return self.zero_();
    }

    // 执行实际的复制操作
    copy_impl(self, src, non_blocking);
  }

  // 如果输出名称不为空，则传播名称
  namedinference::propagate_names_if_nonempty(self, maybe_outnames);

  // 返回 self 本身
  return self;
}
// 复制函数，处理重叠索引，但不关心哪个写入者胜出。虽然有些粗糙，但能运行。
// 仅在 CUDA_tensor_apply2 中用于处理写入重叠情况。
// FIXME: 实际上，在 Torch 中应该将重叠写入视为非法或错误。
void copy_ignoring_overlaps(const TensorBase &dst, const TensorBase &src) {
  // 创建一个张量迭代器配置对象，用于配置复制操作的参数
  auto iter = TensorIteratorConfig()
      // 添加输出张量
      .add_output(dst)
      // 添加输入张量（作为常量输入）
      .add_const_input(src)
      // 禁止调整输出张量大小
      .resize_outputs(false)
      // 不检查内存重叠
      .set_check_mem_overlap(false)
      // 检查所有张量是否具有相同的数据类型
      .check_all_same_dtype(true)
      // 检查所有张量是否位于相同的设备上
      .check_all_same_device(true)
      // 构建张量迭代器
      .build();
  // 调用复制函数的分发函数，根据设备类型选择合适的复制实现
  copy_stub(iter.device_type(), iter, /*non_blocking=*/false);
}

// 在内部使用的函数，用于传播 XLA 数据
void _propagate_xla_data(const Tensor& input, const Tensor& output) {
  // 断言输入张量位于 XLA 设备上，否则抛出错误信息
  TORCH_INTERNAL_ASSERT(input.device().type() == kXLA, "This op should only be called by XLA")
}

// 定义复制函数的分发函数
DEFINE_DISPATCH(copy_stub);

} // namespace at::native
```