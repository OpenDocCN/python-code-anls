# `.\pytorch\aten\src\ATen\native\metal\MetalContext.h`

```py
```objective-c
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <string>

API_AVAILABLE(ios(11.0), macos(10.13))
@interface MetalContext : NSObject
@property(nonatomic, strong, readonly) id<MTLDevice> device;  // Metal 设备的属性，用于管理和执行 GPU 上的计算任务
@property(nonatomic, strong, readonly) id<MTLCommandQueue> commandQueue;  // Metal 命令队列，用于提交 GPU 命令
@property(nonatomic, strong, readonly) id<MTLLibrary> library;  // Metal 库，包含编译好的着色器函数和其他 GPU 计算任务所需的函数

+ (instancetype)sharedInstance;  // 获取 MetalContext 的共享实例
- (BOOL)available;  // 检查 Metal 环境是否可用
- (id<MTLComputePipelineState>)pipelineState:(const std::string&)kernel;  // 创建并返回一个未特化的 Metal 计算管线状态对象
- (id<MTLComputePipelineState>)specializedPipelineState:(const std::string&)kernel
                                              Constants:(NSArray<NSNumber*>*)constants;  // 创建并返回一个已特化的 Metal 计算管线状态对象，可传入常量参数
- (id<MTLBuffer>)emptyMTLBuffer:(int64_t) size;  // 创建并返回一个空的 Metal 缓冲区对象，用于存储 GPU 计算结果或输入数据

@end
```