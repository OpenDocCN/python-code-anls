# `.\pytorch\aten\src\ATen\native\metal\mpscnn\MPSCNNNeuronOp.h`

```
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

# 导入 MetalPerformanceShaders 框架，这个框架包含了 Metal Performance Shaders 的相关功能和类。


@interface MPSCNNNeuronOp : NSObject

# 定义一个 Objective-C 类 MPSCNNNeuronOp，继承自 NSObject。


+ (MPSCNNNeuronHardSigmoid*)hardSigmoid API_AVAILABLE(ios(11.0), macos(10.13));

# 声明类方法 hardSigmoid，返回一个 MPSCNNNeuronHardSigmoid 对象。这个方法在 iOS 11.0 及 macOS 10.13 及以上版本可用。


+ (MPSCNNNeuronReLU*)relu;

# 声明类方法 relu，返回一个 MPSCNNNeuronReLU 对象。


+ (MPSCNNNeuronSigmoid*)sigmoid;

# 声明类方法 sigmoid，返回一个 MPSCNNNeuronSigmoid 对象。


+ (MPSCNNNeuronTanH*)tanh;

# 声明类方法 tanh，返回一个 MPSCNNNeuronTanH 对象。


@end

# 结束 MPSCNNNeuronOp 类的声明。


API_AVAILABLE(ios(11.3), macos(10.13), macCatalyst(13.0))
@interface MPSCNNNeuronOpDescriptor : NSObject

# 定义一个 Objective-C 类 MPSCNNNeuronOpDescriptor，继承自 NSObject，并且这个类在 iOS 11.3 及 macOS 10.13 及 macOS Catalyst 13.0 及以上版本可用。


+ (MPSNNNeuronDescriptor*)hardSigmoidDescriptor;

# 声明类方法 hardSigmoidDescriptor，返回一个 MPSNNNeuronDescriptor 对象。


+ (MPSNNNeuronDescriptor*)reluDescriptor;

# 声明类方法 reluDescriptor，返回一个 MPSNNNeuronDescriptor 对象。


+ (MPSNNNeuronDescriptor*)sigmoidDescriptor;

# 声明类方法 sigmoidDescriptor，返回一个 MPSNNNeuronDescriptor 对象。


+ (MPSNNNeuronDescriptor*)tanhDescriptor;

# 声明类方法 tanhDescriptor，返回一个 MPSNNNeuronDescriptor 对象。


@end

# 结束 MPSCNNNeuronOpDescriptor 类的声明。
```