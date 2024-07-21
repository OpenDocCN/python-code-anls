# `.\pytorch\aten\src\ATen\mps\MPSStream.h`

```py
/*
   著作权声明，指明该文件版权归属于 Apple 公司，表示版权所有年份为 2022 年
*/
#pragma once

#include <cstdint>                   // 包含标准整数类型定义
#include <utility>                   // 包含标准实用工具

#include <c10/core/DeviceGuard.h>    // 包含 C10 核心设备保护相关头文件
#include <c10/util/Exception.h>      // 包含 C10 异常处理相关头文件
#include <c10/core/Stream.h>         // 包含 C10 流处理相关头文件
#include <ATen/mps/MPSDevice.h>      // 包含 ATen MPS 设备相关头文件

#ifdef __OBJC__
#include <Foundation/Foundation.h>                       // 包含 Objective-C Foundation 框架头文件
#include <Metal/Metal.h>                                 // 包含 Metal 图形处理框架头文件
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>  // 包含 Metal Performance Shaders 头文件
#include <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>  // 包含 Metal Performance Shaders Graph 头文件
typedef id<MTLCommandQueue> MTLCommandQueue_t;            // 定义 Metal 命令队列类型为 MTLCommandQueue_t
typedef id<MTLCommandBuffer> MTLCommandBuffer_t;          // 定义 Metal 命令缓冲区类型为 MTLCommandBuffer_t
typedef id<MTLComputeCommandEncoder> MTLComputeCommandEncoder_t;  // 定义 Metal 计算命令编码器类型为 MTLComputeCommandEncoder_t
typedef id<MTLSharedEvent> MTLSharedEvent_t;              // 定义 Metal 共享事件类型为 MTLSharedEvent_t
typedef id<MTLDevice> MTLDevice_t;                        // 定义 Metal 设备类型为 MTLDevice_t
#else
typedef void* MTLCommandQueue_t;                          // 定义非 Objective-C 模式下的 Metal 命令队列类型为 MTLCommandQueue_t
typedef void* MTLCommandQueue;                            // 同上，修正了拼写错误
typedef void* MTLCommandBuffer_t;                         // 定义非 Objective-C 模式下的 Metal 命令缓冲区类型为 MTLCommandBuffer_t
typedef void* MTLCommandBuffer;                           // 同上，修正了拼写错误
typedef void* MTLComputeCommandEncoder_t;                 // 定义非 Objective-C 模式下的 Metal 计算命令编码器类型为 MTLComputeCommandEncoder_t
typedef void* MTLSharedEvent_t;                           // 定义非 Objective-C 模式下的 Metal 共享事件类型为 MTLSharedEvent_t
typedef void* dispatch_queue_t;                           // 定义非 Objective-C 模式下的调度队列类型为 dispatch_queue_t
typedef void* MTLDevice_t;                                // 定义非 Objective-C 模式下的 Metal 设备类型为 MTLDevice_t
#define nil NULL;                                         // 定义 nil 宏为 NULL
#endif


namespace at::mps {

//-----------------------------------------------------------------
//  MPSStream
//-----------------------------------------------------------------

enum class SyncType {
  NONE,               // 同步类型：无，不提交命令缓冲区
  COMMIT,             // 同步类型：提交并刷新命令缓冲区
  COMMIT_AND_WAIT,    // 同步类型：提交并等待命令缓冲区执行完成
  COMMIT_AND_CONTINUE,// 同步类型：提交并继续使用新的底层命令缓冲区
  COMMIT_ADAPTIVE,    // 同步类型：自适应提交，根据可用内存自动决定提交方式
};

class TORCH_API MPSStream
{
/// This enum allows construction of an MPSStream with unchecked parameters.
public:
  enum Unchecked { UNCHECKED };

  /// Construct a MPSStream from a Stream.  This construction is checked,
  /// and will raise an error if the Stream is not, in fact, a MPS stream.
  explicit MPSStream(Stream stream);

  /// Destructor for MPSStream.
  ~MPSStream();

  /// Returns the Metal command queue associated with this MPSStream.
  MTLCommandQueue_t commandQueue() const { return _commandQueue; };

  /// Returns the dispatch queue associated with this MPSStream.
  dispatch_queue_t queue() const { return _serialQueue; }

  /// Returns a command buffer for executing MPS operations.
  MPSCommandBuffer* commandBuffer();

  /// Returns a Metal compute command encoder for MPS operations.
  MTLComputeCommandEncoder_t commandEncoder();

  /// Ends kernel coalescing for the current MPSStream.
  void endKernelCoalescing();

  /// Synchronizes execution of MPS operations.
  void synchronize(SyncType syncType);

  /// Fills a Metal buffer with a specified value.
  void fill(id<MTLBuffer> buffer, uint8_t value, size_t length, size_t offset, SyncType syncType = SyncType::NONE);

  /// Copies data from one Metal buffer to another.
  void copy(id<MTLBuffer> srcBuffer, id<MTLBuffer> dstBuffer,
            size_t length, size_t srcOffset, size_t dstOffset,
            uint64_t profileId, SyncType syncType = SyncType::NONE);

  /// Copies data from one Metal buffer to another and synchronizes the operation.
  void copy_and_sync(id<MTLBuffer> srcBuffer, id<MTLBuffer> dstBuffer,
                     size_t length, size_t srcOffset, size_t dstOffset,
                     bool non_blocking, uint64_t profileId);

  /// Executes an MPS graph with provided feeds and collects results.
  void executeMPSGraph(MPSGraph* mpsGraph, NSDictionary* feeds, NSDictionary* results, SyncType syncType = SyncType::NONE);

  /// Adds a completion handler block to the command buffer.
  void addCompletedHandler(MTLCommandBufferHandler block);

  /// Retrieves the device index associated with this MPSStream.
  c10::DeviceIndex device_index() const { return _stream.device_index(); }

  /// Returns the Metal command queue associated with this MPSStream.
  MTLCommandQueue_t stream() const { return _commandQueue; };

  /// Returns the Metal device associated with this MPSStream.
  MTLDevice_t device() const { return [_commandQueue device];}

  /// Converts the MPSStream back to its underlying Stream object.
  /// This method is explicit.
  Stream unwrap() const { return _stream; }

private:
  Stream _stream;  ///< The underlying Stream object.
  MTLCommandQueue_t _commandQueue = nil;  ///< The Metal command queue.
  MPSCommandBuffer* _commandBuffer = nil;  ///< Current MPS command buffer.
  MPSCommandBuffer* _prevCommandBuffer = nil;  ///< Previous MPS command buffer.
  MTLComputeCommandEncoder_t _commandEncoder = nil;  ///< Metal compute command encoder.
  MPSGraphExecutionDescriptor *_executionDescriptor = nil;  ///< MPS graph execution descriptor.
  MPSGraphCompilationDescriptor *_compilationDescriptor = nil;  ///< MPS graph compilation descriptor.
  dispatch_queue_t _serialQueue = nullptr;  ///< Dispatch queue for serialization.
  bool _enableCommitAndContinue = true;  ///< Flag indicating CommitAndContinue is enabled by default.

  /// Internal method to commit operations.
  /// Use synchronize() to access any of these commit functions outside MPSStream.
  void commit();

  /// Internal method to commit operations and wait for completion.
  /// Use synchronize() to access any of these commit functions outside MPSStream.
  void commitAndWait();

  /// Internal method to commit operations and continue.
  /// Use synchronize() to access any of these commit functions outside MPSStream.
  void commitAndContinue();

  /// Internal method to flush operations.
  /// Use synchronize() to access any of these commit functions outside MPSStream.
  void flush();
};

/**
 * Get the current MPS stream.
 */
TORCH_API MPSStream* getCurrentMPSStream();

/**
 * Get the default MPS stream.
 */
TORCH_API MPSStream* getDefaultMPSStream();

//-----------------------------------------------------------------
//  MPSStreamImpl
//-----------------------------------------------------------------

/// Implementation details for MPSStream.
class TORCH_API MPSStreamImpl
{
 public:
  /**
   * Gets the single instance of the MPSStream.
   */
  static MPSStream* getInstance();

 private:
  static MPSStream* _stream;  ///< Singleton instance of MPSStream.

  /// Private constructor to prevent external instantiation.
  MPSStreamImpl();
};

} // namespace at::mps
```