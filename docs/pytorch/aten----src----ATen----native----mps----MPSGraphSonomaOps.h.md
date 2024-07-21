# `.\pytorch\aten\src\ATen\native\mps\MPSGraphSonomaOps.h`

```py
#pragma once
// 预处理指令，确保头文件只被包含一次

#include <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
// 引入 MetalPerformanceShadersGraph 框架头文件

#if !defined(__MAC_14_0) && \
    (!defined(MAC_OS_X_VERSION_14_0) || (MAC_OS_X_VERSION_MIN_REQUIRED < MAC_OS_X_VERSION_14_0))
// 如果不是 macOS 14.0 或更高版本，则执行以下内容

typedef NS_ENUM(NSUInteger, MPSGraphFFTScalingMode)
{
    MPSGraphFFTScalingModeNone          = 0L,
    MPSGraphFFTScalingModeSize          = 1L,
    MPSGraphFFTScalingModeUnitary       = 2L,
};
// 定义 MPSGraphFFTScalingMode 枚举类型，表示 FFT 的缩放模式

@interface FakeMPSGraphFFTDescriptor : NSObject<NSCopying>
// 定义 FakeMPSGraphFFTDescriptor 类，继承自 NSObject，实现 NSCopying 协议
@property (readwrite, nonatomic) BOOL inverse;
// 属性：表示是否执行逆 FFT
@property (readwrite, nonatomic) MPSGraphFFTScalingMode scalingMode;
// 属性：表示 FFT 的缩放模式
@property (readwrite, nonatomic) BOOL roundToOddHermitean;
// 属性：表示是否对奇厄米特变换进行舍入

+(nullable instancetype) descriptor;
// 类方法：返回一个可空的 FakeMPSGraphFFTDescriptor 实例

@end

@compatibility_alias MPSGraphFFTDescriptor FakeMPSGraphFFTDescriptor;
// 兼容别名定义：MPSGraphFFTDescriptor 别名为 FakeMPSGraphFFTDescriptor

@interface MPSGraph (SonomaOps)
// MPSGraph 的 SonomaOps 扩展

-(MPSGraphTensor * _Nonnull) conjugateWithTensor:(MPSGraphTensor * _Nonnull) tensor
                                            name:(NSString * _Nullable) name;
// 方法：对输入张量进行共轭操作，返回结果张量

-(MPSGraphTensor * _Nonnull) realPartOfTensor:(MPSGraphTensor * _Nonnull) tensor
                                         name:(NSString * _Nullable) name;
// 方法：获取输入张量的实部，返回结果张量

-(MPSGraphTensor * _Nonnull) fastFourierTransformWithTensor:(MPSGraphTensor * _Nonnull) tensor
                                                       axes:(NSArray<NSNumber *> * _Nonnull) axes
                                                 descriptor:(MPSGraphFFTDescriptor * _Nonnull) descriptor
                                                       name:(NSString * _Nullable) name;
// 方法：执行快速傅里叶变换（FFT），返回结果张量

-(MPSGraphTensor * _Nonnull) realToHermiteanFFTWithTensor:(MPSGraphTensor * _Nonnull) tensor
                                                     axes:(NSArray<NSNumber *> * _Nonnull) axes
                                               descriptor:(MPSGraphFFTDescriptor * _Nonnull) descriptor
                                                     name:(NSString * _Nullable) name;
// 方法：将实数张量转换为厄米特变换，返回结果张量

-(MPSGraphTensor * _Nonnull) HermiteanToRealFFTWithTensor:(MPSGraphTensor * _Nonnull) tensor
                                                     axes:(NSArray<NSNumber *> * _Nonnull) axes
                                               descriptor:(MPSGraphFFTDescriptor * _Nonnull) descriptor
                                                     name:(NSString * _Nullable) name;
// 方法：将厄米特变换转换为实数张量，返回结果张量

@end

// 定义 BFloat16 类型的枚举，用于 macOS 13
#define MPSDataTypeBFloat16 ((MPSDataType) (MPSDataTypeAlternateEncodingBit | MPSDataTypeFloat16))

// 定义 Metal 的语言版本为 3.1
#define MTLLanguageVersion3_1 ((MTLLanguageVersion) ((3 << 16) + 1))
#endif
// 结束条件：如果不是 macOS 14.0 或更高版本，则结束预处理指令块
```