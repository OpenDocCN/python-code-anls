# `.\pytorch\aten\src\ATen\native\metal\mpscnn\MPSCNNConvOp.h`

```
#import <ATen/native/metal/MetalConvParams.h>
#import <ATen/native/metal/MetalNeuronType.h>
#import <ATen/native/metal/mpscnn/MPSCNNOp.h>
#import <Foundation/Foundation.h>

// 声明 MPSCNNConvDataSource 类，继承自 NSObject，并实现 MPSCNNConvolutionDataSource 协议
API_AVAILABLE(ios(11.0), macos(10.13))
@interface MPSCNNConvDataSource : NSObject<MPSCNNConvolutionDataSource>

// 指向权重数据的指针
@property(nonatomic, assign) void* weights;
// 指向偏置数据的指针
@property(nonatomic, assign) float* bias;

// 初始化方法，接受权重指针、偏置指针和 MPSCNNConvolutionDescriptor 对象作为参数
- (id)initWithWeights:(void*)weights
                 Bias:(float*)bias
                 Desc:(MPSCNNConvolutionDescriptor*)desc;

@end

// 使用 at::native::metal 命名空间
using namespace at::native::metal;

// 声明 MPSCNNConvOp 类，继承自 NSObject，并实现 MPSCNNOp 协议
API_AVAILABLE(ios(11.0), macos(10.13))
@interface MPSCNNConvOp : NSObject<MPSCNNOp>

// 静态方法，用于创建一个 MPSCNNConvOp 对象，接受 Conv2DParams 结构体、权重指针、偏置指针和 NeuronType 枚举作为参数
+ (MPSCNNConvOp*)conv2d:(const Conv2DParams&)params
                weights:(float*)w
                   bias:(float*)b
           neuronFilter:(NeuronType)t;

@end
```