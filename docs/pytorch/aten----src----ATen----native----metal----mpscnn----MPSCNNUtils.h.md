# `.\pytorch\aten\src\ATen\native\metal\mpscnn\MPSCNNUtils.h`

```
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <string>

// 定义一个实用的宏，用于在 Metal API 函数产生 NSError 时抛出异常。异常包含了从 NSError 提取的有用信息。
#define METAL_THROW_IF_ERROR(error, preamble)                                    \
  do {                                                                           \
    // 如果 error 存在
    if C10_LIKELY(error) {                                                       \
      // 抛出 C10 异常，包含函数名、文件名和行号，并附带错误相关信息
      throw c10::Error(                                                          \
          {__func__, __FILE__, static_cast<uint32_t>(__LINE__)},                 \
          c10::str(                                                              \
              preamble,                                                          \
              " Error details: ",                                                \
              " Localized_description: ", error.localizedDescription.UTF8String, \
              " Domain: ", error.domain.UTF8String,                              \
              " Code: ", error.code,                                             \
              " User Info: ", error.userInfo.description.UTF8String));           \
    }                                                                            \
  } while (false)

namespace at::native::metal::mpscnn {

// 定义结构体 LaunchParams，用于描述 Metal 计算内核启动时的参数
struct LaunchParams {
  MTLSize threadsPerThreadgroup;     // 每个线程组中的线程数
  MTLSize threadgroupsPerGrid;       // 网格中线程组的数量
  MTLSize threadsPerGrid;            // 网格中的线程数量（iOS 11.0）
};

// 声明一个 Metal Performance Shaders 中的函数，用于获取空间点操作内核启动时的 LaunchParams
API_AVAILABLE(ios(11.0), macos(10.13))
LaunchParams spatialPointwiseKernelLaunchParams(
    id<MTLComputePipelineState> pipeline,   // Metal 计算管线状态对象
    MPSImage* im);                         // MPSImage 对象

// 声明一个 Metal Performance Shaders 中的函数，用于获取空间点操作内核启动时的 LaunchParams
API_AVAILABLE(ios(11.0), macos(10.13))
LaunchParams spatialPointwiseKernelLaunchParams(
    id<MTLComputePipelineState> pipeline,   // Metal 计算管线状态对象
    NSUInteger numberOfImages,              // 图像数量
    NSUInteger featureChannels,             // 特征通道数
    NSUInteger height,                      // 图像高度
    NSUInteger width);                      // 图像宽度

// 定义一个静态内联函数，返回适合用于给定 MPSImage 的内核名称
static inline std::string kernelFor(
    MPSImage* image,                        // MPSImage 对象
    const std::string& arrayKernel,         // 数组内核名称
    const std::string& nonArrayKernel) {    // 非数组内核名称
  // 如果图像的特征通道大于4或图像数量大于1，返回数组内核名称，否则返回非数组内核名称
  if (image.featureChannels > 4 || image.numberOfImages > 1) {
    return arrayKernel;
  }
  return nonArrayKernel;
}

// 定义一个静态内联函数，计算 MPS 对齐偏移量
static inline int computeMPSAlignOffset(int kernel, int pad) {
  // 为了设置偏移量，我们可以匹配原始实现中查看的顶部左侧像素（在输入图像中，使用负值来表示填充）。
  // 对于 3x3s1p1，我们查看原始实现中的 (-1, -1) 像素。对于 3x3s1p0，我们查看 (0, 0) 像素。
  // 对于 3x3s1p2，查看 (-2, -2)。MPSCNN 总是查看 (-floor(kernel_size - 1 / 2), -floor(kernel_size - 1 / 2))。
  // 因此，我们只需要进行匹配即可。

  // 对于 3x3s1p1，偏移应为 (0, 0)
  // 对于 3x3s1p0，偏移应为 (1, 1)
  // 对于 3x3s1p2，偏移应为 (-1, -1)
  const int mps_offset = kernel / 2;
  const int pt_offset = pad;
  return mps_offset - pt_offset;
}

} // namespace at::native::metal::mpscnn
```