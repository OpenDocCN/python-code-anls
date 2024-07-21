# `.\pytorch\aten\src\ATen\native\metal\mpscnn\MPSCNNFullyConnectedOp.h`

```py
// 导入 MetalConvParams.h 头文件，这是 MetalConvParams 类的声明
#import <ATen/native/metal/MetalConvParams.h>
// 导入 MetalNeuronType.h 头文件，这是 MetalNeuronType 枚举类型的声明
#import <ATen/native/metal/MetalNeuronType.h>
// 导入 MPSCNNConvOp.h 头文件，这是 MPSCNNConvOp 类的声明
#import <ATen/native/metal/mpscnn/MPSCNNConvOp.h>
// 导入 Foundation.h 头文件，这是 Foundation 框架的基础头文件
#import <Foundation/Foundation.h>

// 声明使用 at::native::metal 命名空间
using namespace at::native::metal;

// 声明 MPSCNNFullyConnectedOp 类，继承自 NSObject 并遵循 MPSCNNOp 协议
// 在 iOS 11.0 及 macOS 10.13 及以上版本可用
API_AVAILABLE(ios(11.0), macos(10.13))
@interface MPSCNNFullyConnectedOp : NSObject<MPSCNNOp>

// 类方法声明，用于创建 MPSCNNFullyConnectedOp 实例
// 参数包括 Conv2DParams 结构体引用 params，float 类型指针 weights 和 bias，NeuronType 类型 neuronFilter
+ (MPSCNNFullyConnectedOp*)linear:(const Conv2DParams&)params
                          weights:(float*)w
                             bias:(float*)b
                     neuronFilter:(NeuronType)t;

@end
```