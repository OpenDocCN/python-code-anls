# `.\pytorch\aten\src\ATen\native\metal\mpscnn\MPSImageWrapper.h`

```
#ifndef MPSImageWrapper_h
#define MPSImageWrapper_h

#import <ATen/native/metal/MetalCommandBuffer.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <c10/util/ArrayRef.h>

namespace at {
namespace native {
namespace metal {

// 定义 MPSImageWrapper 类，提供对 Metal Performance Shaders 图像的封装
class API_AVAILABLE(ios(11.0), macos(10.13)) MPSImageWrapper {
 public:
  // 构造函数，根据给定的尺寸数组初始化 MPSImageWrapper
  MPSImageWrapper(IntArrayRef sizes);
  // 析构函数，释放资源
  ~MPSImageWrapper();
  // 从主机内存复制数据到 MPS 图像
  void copyDataFromHost(const float* inputData);
  // 从 MPS 图像复制数据到主机内存
  void copyDataToHost(float* hostData);
  // 分配存储空间，根据给定的尺寸数组
  void allocateStorage(IntArrayRef sizes);
  // 分配临时存储空间，根据给定的尺寸数组和 Metal 命令缓冲区
  void allocateTemporaryStorage(
      IntArrayRef sizes,
      MetalCommandBuffer* commandBuffer);
  // 设置当前使用的 Metal 命令缓冲区
  void setCommandBuffer(MetalCommandBuffer* buffer);
  // 获取当前使用的 Metal 命令缓冲区
  MetalCommandBuffer* commandBuffer() const;
  // 设置 MPS 图像对象
  void setImage(MPSImage* image);
  // 获取当前 MPS 图像对象
  MPSImage* image() const;
  // 获取当前 Metal 缓冲区对象
  id<MTLBuffer> buffer() const;
  // 同步操作，确保所有待处理的命令已执行完成
  void synchronize();
  // 准备 MPS 图像对象，使其可用于计算
  void prepare();
  // 释放 MPS 图像对象及其相关资源
  void release();

 private:
  std::vector<int64_t> _imageSizes;  // 存储图像尺寸的数组
  MPSImage* _image = nil;            // MPS 图像对象
  id<MTLBuffer> _buffer = nil;       // Metal 缓冲区对象
  MetalCommandBuffer* _commandBuffer = nil;  // 当前使用的 Metal 命令缓冲区
  id<PTMetalCommandBuffer> _delegate = nil;  // Metal 命令缓冲区的代理对象
};

} // namespace metal
} // namespace native
} // namespace at

#endif /* MPSImageWrapper_h */
```