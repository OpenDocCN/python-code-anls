# `.\pytorch\aten\src\ATen\native\metal\mpscnn\MPSImageUtils.h`

```
#import <ATen/Tensor.h>
#import <ATen/native/metal/MetalCommandBuffer.h>
#import <ATen/native/metal/MetalTensorImpl.h>
#import <ATen/native/metal/MetalTensorUtils.h>

#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

namespace at {
namespace native {
namespace metal {

// 创建一个静态的 MPSImage 对象，根据给定的尺寸数组
MPSImage* createStaticImage(IntArrayRef sizes);

// 根据给定的数据源和尺寸数组创建静态的 MPSImage 对象
MPSImage* createStaticImage(const float* src, const IntArrayRef sizes);

// 根据临时图像和 Metal 命令缓冲区创建静态的 MPSImage 对象，可以选择是否等待完成
MPSImage* createStaticImage(
    MPSTemporaryImage* image,
    MetalCommandBuffer* buffer,
    bool waitUntilCompleted);

// 根据 Metal 命令缓冲区和尺寸数组创建临时的 MPSTemporaryImage 对象
MPSTemporaryImage* createTemporaryImage(
    MetalCommandBuffer* buffer,
    const IntArrayRef sizes);

// 根据 Metal 命令缓冲区、尺寸数组和数据源创建临时的 MPSTemporaryImage 对象
MPSTemporaryImage* createTemporaryImage(
    MetalCommandBuffer* buffer,
    const IntArrayRef sizes,
    const float* src);

// 根据 Metal 命令缓冲区和现有的 MPSImage 对象创建临时的 MPSTemporaryImage 对象
MPSTemporaryImage* createTemporaryImage(
    MetalCommandBuffer* buffer,
    MPSImage* image);

// 将 MPSImage 对象的数据复制到浮点数缓冲区中
void copyImageToFloatBuffer(float* dst, MPSImage* image);

// 将 MPSImage 对象的数据复制到 Metal 缓冲区中
void copyImageToMetalBuffer(
    MetalCommandBuffer* buffer,
    id<MTLBuffer> dst,
    MPSImage* image);

// 根据给定的 Tensor 对象创建相应的 MPSImage 对象
static inline MPSImage* imageFromTensor(const Tensor& tensor) {
  TORCH_CHECK(tensor.is_metal());
  using MetalTensorImplStorage = at::native::metal::MetalTensorImplStorage;
  using MetalTensorImpl = at::MetalTensorImpl<MetalTensorImplStorage>;
  MetalTensorImpl* impl = (MetalTensorImpl*)tensor.unsafeGetTensorImpl();
  MetalTensorImplStorage& implStorage = impl->unsafe_opaque_handle();
  return implStorage.texture()->image();
}

/*
MPSImage 携带一个 IntList 形状，其与其转换自的 CPU 张量的形状相同。
1) 1D 张量（W,）始终存储为 MPSImage(N=1, C=1, H=1, W=W)。
2) 2D 张量（H, W）始终存储为 MPSImage(N=1, C=1, H=H, W=W)。
3) 3D 张量（C, H, W）始终存储为 MPSImage(N=1, C=C, H=H, W=W)。
4) 4D 张量（N, C, H, W）始终存储为 MPSImage(N=N, C=C, H=H, W=W)。
5) 5D 张量（T, N, C, H, W）始终存储为 MPSImage(N=T*N, C=C, H=H, W=W)。
6) ...
 */
static inline std::vector<int64_t> computeImageSize(IntArrayRef sizes) {
  // 初始化图像尺寸数组，将长度设为4，初始值均为1
  std::vector<int64_t> imageSize(4, 1);
  // 从最后一个尺寸开始遍历，index 表示当前要填充的 imageSize 的索引
  int64_t index = 3;
  // 初始化 batch 为1，用于存储除了前三个维度外的所有维度的乘积
  int64_t batch = 1;
  for (int64_t i = sizes.size() - 1; i >= 0; i--) {
    if (index != 0) {
      // 将 sizes[i] 的值存入 imageSize 的当前索引位置，然后将 index 减一
      imageSize[index] = sizes[i];
      index--;
      continue;
    }
    // 对于更高维度的张量，将其余的维度乘积存入 imageSize[0]
    batch *= sizes[i];
  }
  // 将计算得到的 batch 存入 imageSize[0]，表示最终的图像数量
  imageSize[0] = batch;
  return imageSize;
}

} // namespace metal
} // namespace native
} // namespace at
```