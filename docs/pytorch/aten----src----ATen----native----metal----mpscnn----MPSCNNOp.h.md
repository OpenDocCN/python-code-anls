# `.\pytorch\aten\src\ATen\native\metal\mpscnn\MPSCNNOp.h`

```py
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

@protocol MPSCNNOp<NSObject>
// 定义了 MPSCNNOp 协议，用于描述 MPS 卷积神经网络操作的接口

@property(nonatomic, strong) MPSCNNKernel* kernel;
// 定义了一个属性 kernel，类型为 MPSCNNKernel 对象，用于存储 MPS 卷积核对象

- (void)encode:(id<MTLCommandBuffer>)cb
   sourceImage:(MPSImage*)src
destinationImage:(MPSImage*)dst;
// 定义了一个方法 encode:sourceImage:destinationImage:，用于将输入图像 src 进行处理，并将结果输出到图像 dst
@end

@protocol MPSCNNShaderOp<NSObject>
// 定义了 MPSCNNShaderOp 协议，用于描述 MPS 卷积神经网络着色器操作的接口

+ (id<MPSCNNShaderOp>)newWithTextures:(NSArray<MPSImage*>*)textures
                                 Args:(NSArray<NSNumber*>*)args;
// 定义了一个静态方法 newWithTextures:Args:，用于创建一个实现了 MPSCNNShaderOp 协议的对象，并传入纹理和参数数组

- (void)encode:(id<MTLCommandBuffer>)cb;
// 定义了一个方法 encode:，用于将操作编码到 Metal 命令缓冲区中
@end
```