# `.\pytorch\aten\src\ATen\native\metal\MetalCommandBuffer.h`

```py
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

# 导入 MetalPerformanceShaders 框架，这是一个使用 Metal 进行图像处理和机器学习加速的框架。


@protocol PTMetalCommandBuffer<NSObject>
@optional
- (void)beginSynchronization;
- (void)endSynchronization:(NSError*)error;
@end

# 定义一个名为 PTMetalCommandBuffer 的协议，继承自 NSObject。该协议声明了两个可选方法：
# - beginSynchronization: 开始同步方法，用于开始同步操作。
# - endSynchronization: 结束同步方法，接受一个 NSError 对象作为参数，用于结束同步操作并可能返回错误信息。


@interface MetalCommandBuffer : NSObject
@property(nonatomic, strong, readonly) id<MTLCommandBuffer> buffer;
@property(nonatomic, assign, readonly) BOOL valid;

+ (MetalCommandBuffer*)newBuffer;
+ (MetalCommandBuffer*)currentBuffer;
- (void)addSubscriber:(id<PTMetalCommandBuffer>)subscriber;
- (void)removeSubscriber:(id<PTMetalCommandBuffer>)subscriber;
- (void)commit;
- (void)add:(MPSTemporaryImage*)image;
- (void)remove:(MPSTemporaryImage*)image;

@end

# MetalCommandBuffer 类的声明，继承自 NSObject 类。
# - buffer 属性：只读，存储一个遵循 MTLCommandBuffer 协议的对象。
# - valid 属性：只读，表示当前缓冲区的有效性。

# 提供的类方法：
# - newBuffer: 创建并返回一个 MetalCommandBuffer 对象。
# - currentBuffer: 返回当前的 MetalCommandBuffer 对象。

# 实例方法：
# - addSubscriber: 添加一个遵循 PTMetalCommandBuffer 协议的订阅者对象。
# - removeSubscriber: 移除一个订阅者对象。
# - commit: 提交当前的缓冲区操作。
# - add: 向缓冲区中添加一个 MPSTemporaryImage 对象。
# - remove: 从缓冲区中移除一个 MPSTemporaryImage 对象。
```