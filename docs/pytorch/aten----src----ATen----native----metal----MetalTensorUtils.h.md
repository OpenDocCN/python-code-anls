# `.\pytorch\aten\src\ATen\native\metal\MetalTensorUtils.h`

```py
// 包含 ATen 库中相关头文件，用于张量操作
#include <ATen/Tensor.h>
#include <ATen/native/metal/MetalContext.h>
#include <ATen/native/metal/MetalCommandBuffer.h>
#include <ATen/native/metal/MetalTensorImpl.h>
#include <ATen/native/metal/MetalTensorImplStorage.h>

// 根据 ARM NEON 的定义，选择合适的 fp16_t 类型
#if (defined(__ARM_NEON__) || defined(__ARM_NEON))
typedef float16_t fp16_t;
#else
typedef uint16_t fp16_t;
#endif

// 定义 at::native::metal 命名空间
namespace at::native::metal {

// 获取张量的批次大小
uint32_t batchSize(const Tensor& tensor);

// 获取张量的通道数大小
uint32_t channelsSize(const Tensor& tensor);

// 获取张量的高度大小
uint32_t heightSize(const Tensor& tensor);

// 获取张量的宽度大小
uint32_t widthSize(const Tensor& tensor);

// 计算张量的步长（strides），基于连续内存格式
static inline std::vector<int64_t> computeStrides(
    const std::vector<int64_t>& sizes) {
  const auto dim = sizes.size();
  std::vector<int64_t> strides(dim, 0);
  if (dim > 0) {
    const auto last_idx = dim - 1;
    strides[last_idx] = 1;
    for (int64_t i = last_idx - 1; i >= 0; --i) {
      // 根据当前维度的尺寸和后续维度的步长计算当前维度的步长
      strides[i] = strides[i + 1] * std::max<int64_t>(sizes[i + 1], 1);
    }
  }
  return strides;
}

// 获取 MetalTensorImplStorage 对象，用于 Metal 张量的实现存储
static inline MetalTensorImplStorage& getTensorImplStorage(
    const at::Tensor& tensor) {
  using MetalTensorImpl = at::MetalTensorImpl<MetalTensorImplStorage>;
  // 断言张量确实是 Metal 张量
  TORCH_CHECK(tensor.is_metal());
  // 强制转换为 MetalTensorImpl 类型，获取不安全的张量实现指针
  MetalTensorImpl* impl =
      static_cast<MetalTensorImpl*>(tensor.unsafeGetTensorImpl());
  return impl->unsafe_opaque_handle(); // 返回 MetalTensorImplStorage 对象
}

// 创建 Metal 张量的封装函数
static inline at::Tensor makeTensor(
    MetalTensorImplStorage&& mt,
    const TensorOptions& options) {
  using MetalTensorImpl = at::MetalTensorImpl<MetalTensorImplStorage>;
  auto sizes = mt.sizes(); // 从 MetalTensorImplStorage 中获取尺寸
  auto strides = mt.strides(); // 从 MetalTensorImplStorage 中获取步长
  // 使用 MetalTensorImpl 创建张量，并指定属性
  return detail::make_tensor<MetalTensorImpl>(
      DispatchKeySet(DispatchKey::Metal),
      options.dtype(),
      at::Device(at::kMetal),
      std::move(mt),
      std::vector<int64_t>(sizes.begin(), sizes.end()),
      std::vector<int64_t>(strides.begin(), strides.end()));
}

// 获取 Metal 张量的命令缓冲区
static inline MetalCommandBuffer* getCommandBuffer(
    const Tensor& tensor) {
  // 断言张量确实是 Metal 张量
  TORCH_CHECK(tensor.is_metal());
  // 获取张量的实现存储
  auto implStorage = getTensorImplStorage(tensor);
  // 获取 Metal 张量实现存储中的命令缓冲区
  MetalCommandBuffer* cmdBuffer = implStorage.texture()->commandBuffer();
  // 如果命令缓冲区不存在或无效，则使用当前的 Metal 命令缓冲区
  if (!cmdBuffer || !cmdBuffer.valid) {
    cmdBuffer = [MetalCommandBuffer currentBuffer];
  }
  return cmdBuffer; // 返回命令缓冲区指针
}

} // namespace at::native::metal
```