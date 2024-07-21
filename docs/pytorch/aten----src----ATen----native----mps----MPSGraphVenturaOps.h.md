# `.\pytorch\aten\src\ATen\native\mps\MPSGraphVenturaOps.h`

```py
#pragma once
#include <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

// TODO: Remove me when moved to MacOS 13
// 如果不是在 MacOS 13 或者未定义 MAC_OS_X_VERSION_13_2 或者 MAC_OS_X_VERSION_MIN_REQUIRED 小于 MAC_OS_X_VERSION_13_2，执行下面的代码块
#if !defined(__MAC_13_2) && \
    (!defined(MAC_OS_X_VERSION_13_2) || (MAC_OS_X_VERSION_MIN_REQUIRED < MAC_OS_X_VERSION_13_2))

// 定义一个虚拟的 MPSGraphConvolution3DOpDescriptor 类，实现 NSCopying 协议
@interface FakeMPSGraphConvolution3DOpDescriptor : NSObject<NSCopying>

// 定义 strideInX、strideInY、strideInZ 等属性，用于描述 3D 卷积操作的参数
@property (readwrite, nonatomic) NSUInteger strideInX;
@property (readwrite, nonatomic) NSUInteger strideInY;
@property (readwrite, nonatomic) NSUInteger strideInZ;
@property (readwrite, nonatomic) NSUInteger dilationRateInX;
@property (readwrite, nonatomic) NSUInteger dilationRateInY;
@property (readwrite, nonatomic) NSUInteger dilationRateInZ;

@property (readwrite, nonatomic) NSUInteger paddingLeft;
@property (readwrite, nonatomic) NSUInteger paddingRight;
@property (readwrite, nonatomic) NSUInteger paddingTop;
@property (readwrite, nonatomic) NSUInteger paddingBottom;
@property (readwrite, nonatomic) NSUInteger paddingFront;
@property (readwrite, nonatomic) NSUInteger paddingBack;

@property (readwrite, nonatomic) MPSGraphPaddingStyle paddingStyle; // 描述填充样式的枚举值
@property (readwrite, nonatomic) MPSGraphTensorNamedDataLayout dataLayout; // 描述数据布局的枚举值
@property (readwrite, nonatomic) MPSGraphTensorNamedDataLayout weightsLayout; // 描述权重布局的枚举值

@property (readwrite, nonatomic) NSUInteger groups; // 分组卷积时的分组数

@end

// 为了兼容性，将 FakeMPSGraphConvolution3DOpDescriptor 别名为 MPSGraphConvolution3DOpDescriptor
@compatibility_alias MPSGraphConvolution3DOpDescriptor FakeMPSGraphConvolution3DOpDescriptor;

#endif

// 定义 MPSGraph 类的类别，命名为 VenturaOps
@interface MPSGraph (VenturaOps)

// 如果不是在 MacOS 13.0 或者未定义 MAC_OS_X_VERSION_13_0 或者 MAC_OS_X_VERSION_MIN_REQUIRED 小于 MAC_OS_X_VERSION_13_0，执行下面的代码块
#if !defined(__MAC_13_0) && \
    (!defined(MAC_OS_X_VERSION_13_0) || (MAC_OS_X_VERSION_MIN_REQUIRED < MAC_OS_X_VERSION_13_0))

// 定义 MPSGraphResizeNearestRoundingMode 枚举类型，用于描述最近邻插值的舍入模式
typedef NS_ENUM(NSUInteger, MPSGraphResizeNearestRoundingMode)
{
    MPSGraphResizeNearestRoundingModeRoundPreferCeil   =  0L,
    MPSGraphResizeNearestRoundingModeRoundPreferFloor  =  1L,
    MPSGraphResizeNearestRoundingModeCeil              =  2L,
    MPSGraphResizeNearestRoundingModeFloor             =  3L,
    MPSGraphResizeNearestRoundingModeRoundToEven       =  4L,
    MPSGraphResizeNearestRoundingModeRoundToOdd        =  5L,
};

// 定义复杂的枚举值，用于 MacOS 12 的数据类型
#define MPSDataTypeComplexBit 0x01000000
#define MPSDataTypeComplexFloat32 ((MPSDataType) (MPSDataTypeFloatBit | MPSDataTypeComplexBit | 64))
#define MPSDataTypeComplexFloat16 ((MPSDataType) (MPSDataTypeFloatBit | MPSDataTypeComplexBit | 32))
#endif

// 定义 convolution3DWithSourceTensor 方法，用于进行 3D 卷积操作
- (MPSGraphTensor * _Nonnull) convolution3DWithSourceTensor:(MPSGraphTensor * _Nonnull) source
                                            weightsTensor:(MPSGraphTensor * _Nonnull) weights
                                               descriptor:(MPSGraphConvolution3DOpDescriptor * _Nonnull) descriptor
                                                     name:(NSString * _Nullable) name;
# 计算 3D 卷积数据梯度
- (MPSGraphTensor * _Nonnull) convolution3DDataGradientWithIncomingGradientTensor:(MPSGraphTensor * _Nonnull) incomingGradient
                                                                  weightsTensor:(MPSGraphTensor * _Nonnull) weights
                                                                    outputShape:(MPSShape * _Nonnull) outputShape
                                                   forwardConvolutionDescriptor:(MPSGraphConvolution3DOpDescriptor * _Nonnull) forwardConvolutionDescriptor
                                                                           name:(NSString * _Nullable) name;

# 计算 3D 卷积权重梯度
- (MPSGraphTensor * _Nonnull) convolution3DWeightsGradientWithIncomingGradientTensor:(MPSGraphTensor * _Nonnull) incomingGradient
                                                                      sourceTensor:(MPSGraphTensor * _Nonnull) source
                                                                       outputShape:(MPSShape * _Nonnull) outputShape
                                                      forwardConvolutionDescriptor:(MPSGraphConvolution3DOpDescriptor * _Nonnull) forwardConvolutionDescriptor
                                                                              name:(NSString * _Nullable) name;

# 计算张量在指定轴上的累积和
- (MPSGraphTensor * _Nonnull)cumulativeSumWithTensor:(MPSGraphTensor * _Nonnull)tensor
                                                axis:(NSInteger)axis
                                                name:(NSString * _Nullable)name;

# 对张量在指定轴上进行排序
- (MPSGraphTensor * _Nonnull)sortWithTensor:(MPSGraphTensor * _Nonnull)tensor
                                       axis:(NSInteger)axis
                                       name:(NSString * _Nullable)name;

# 对张量在指定轴上进行排序，支持降序排序
- (MPSGraphTensor * _Nonnull) sortWithTensor:(MPSGraphTensor * _Nonnull) tensor
                               axis:(NSInteger) axis
                         descending:(BOOL) descending
                               name:(NSString * _Nullable) name;

# 对张量在指定轴上进行排序，轴使用张量表示，支持降序排序
- (MPSGraphTensor * _Nonnull) sortWithTensor:(MPSGraphTensor * _Nonnull) tensor
                         axisTensor:(MPSGraphTensor * _Nonnull) axisTensor
                         descending:(BOOL) descending
                               name:(NSString * _Nullable) name;

# 对张量在指定轴上进行排序，轴使用张量表示
- (MPSGraphTensor * _Nonnull) sortWithTensor:(MPSGraphTensor * _Nonnull) tensor
                         axisTensor:(MPSGraphTensor * _Nonnull) axisTensor
                               name:(NSString * _Nullable) name;

# 对张量在指定轴上进行排序并返回排序的索引
- (MPSGraphTensor * _Nonnull)argSortWithTensor:(MPSGraphTensor * _Nonnull)tensor
                                          axis:(NSInteger)axis
                                          name:(NSString * _Nullable)name;

# 对张量在指定轴上进行排序并返回排序的索引，支持降序排序
- (MPSGraphTensor * _Nonnull) argSortWithTensor:(MPSGraphTensor * _Nonnull) tensor
                                  axis:(NSInteger) axis
                            descending:(BOOL) descending
                                  name:(NSString * _Nullable) name;
# 返回一个根据张量排序后的索引的张量，可指定排序轴和降序排列方式
- (MPSGraphTensor * _Nonnull) argSortWithTensor:(MPSGraphTensor * _Nonnull) tensor
                           axisTensor:(MPSGraphTensor * _Nonnull) axisTensor
                           descending:(BOOL) descending
                                 name:(NSString * _Nullable) name;

# 返回一个根据张量排序后的索引的张量，可指定排序轴，按升序排列
- (MPSGraphTensor * _Nonnull) argSortWithTensor:(MPSGraphTensor * _Nonnull) tensor
                           axisTensor:(MPSGraphTensor * _Nonnull) axisTensor
                                 name:(NSString * _Nullable) name;

# 返回输入张量的逆张量，对每个元素求倒数
- (MPSGraphTensor * _Nonnull)inverseOfTensor:(MPSGraphTensor * _Nonnull) inputTensor
                                        name:(NSString * _Nullable)name;

# 返回通过最近邻插值调整大小后的张量，可以指定大小、舍入方式、结果中心化、角落对齐、布局和名称
- (MPSGraphTensor * _Nonnull) resizeNearestWithTensor:(MPSGraphTensor * _Nonnull) imagesTensor
                                           sizeTensor:(MPSGraphTensor * _Nonnull) size
                                  nearestRoundingMode:(MPSGraphResizeNearestRoundingMode) nearestRoundingMode
                                         centerResult:(BOOL) centerResult
                                         alignCorners:(BOOL) alignCorners
                                               layout:(MPSGraphTensorNamedDataLayout) layout
                                                 name:(NSString * _Nullable) name;

# 返回通过最近邻插值调整大小后的张量，可以指定大小、比例偏移、舍入方式、布局和名称
- (MPSGraphTensor * _Nonnull) resizeNearestWithTensor:(MPSGraphTensor * _Nonnull) imagesTensor
                                           sizeTensor:(MPSGraphTensor * _Nonnull) size
                                    scaleOffsetTensor:(MPSGraphTensor * _Nonnull) scaleOffset
                                  nearestRoundingMode:(MPSGraphResizeNearestRoundingMode) nearestRoundingMode
                                               layout:(MPSGraphTensorNamedDataLayout) layout
                                                 name:(NSString * _Nullable) name;

# 返回通过双线性插值调整大小后的张量，可以指定大小、结果中心化、角落对齐、布局和名称
- (MPSGraphTensor * _Nonnull) resizeBilinearWithTensor:(MPSGraphTensor * _Nonnull) imagesTensor
                                            sizeTensor:(MPSGraphTensor * _Nonnull) size
                                          centerResult:(BOOL) centerResult
                                          alignCorners:(BOOL) alignCorners
                                                layout:(MPSGraphTensorNamedDataLayout) layout
                                                  name:(NSString * _Nullable) name;

# 返回通过双线性插值调整大小后的张量，可以指定大小、比例偏移、布局和名称
- (MPSGraphTensor * _Nonnull) resizeBilinearWithTensor:(MPSGraphTensor * _Nonnull) imagesTensor
                                            sizeTensor:(MPSGraphTensor * _Nonnull) size
                                     scaleOffsetTensor:(MPSGraphTensor * _Nonnull) scaleOffset
                                                layout:(MPSGraphTensorNamedDataLayout) layout
                                                  name:(NSString * _Nullable) name;
# 使用最近邻插值方式调整张量大小，并计算梯度
- (MPSGraphTensor * _Nonnull) resizeNearestWithGradientTensor:(MPSGraphTensor * _Nonnull) gradient
                                                        input:(MPSGraphTensor * _Nonnull) input
                                          nearestRoundingMode:(MPSGraphResizeNearestRoundingMode) nearestRoundingMode
                                                 centerResult:(BOOL) centerResult
                                                 alignCorners:(BOOL) alignCorners
                                                       layout:(MPSGraphTensorNamedDataLayout) layout
                                                         name:(NSString * _Nullable) name;



# 使用最近邻插值方式调整张量大小，并考虑缩放偏移张量，同时计算梯度
- (MPSGraphTensor * _Nonnull) resizeNearestWithGradientTensor:(MPSGraphTensor * _Nonnull) gradient
                                                        input:(MPSGraphTensor * _Nonnull) input
                                            scaleOffsetTensor:(MPSGraphTensor * _Nonnull) scaleOffset
                                          nearestRoundingMode:(MPSGraphResizeNearestRoundingMode) nearestRoundingMode
                                                       layout:(MPSGraphTensorNamedDataLayout) layout
                                                         name:(NSString * _Nullable) name;



# 使用双线性插值方式调整张量大小，并计算梯度
- (MPSGraphTensor * _Nonnull) resizeBilinearWithGradientTensor:(MPSGraphTensor * _Nonnull) gradient
                                                         input:(MPSGraphTensor * _Nonnull) input
                                                  centerResult:(BOOL) centerResult
                                                  alignCorners:(BOOL) alignCorners
                                                        layout:(MPSGraphTensorNamedDataLayout) layout
                                                          name:(NSString * _Nullable) name;



# 使用双线性插值方式调整张量大小，并考虑缩放偏移张量
- (MPSGraphTensor * _Nonnull) resizeBilinearWithGradientTensor:(MPSGraphTensor * _Nonnull) gradient
                                                         input:(MPSGraphTensor * _Nonnull) input
                                             scaleOffsetTensor:(MPSGraphTensor * _Nonnull) scaleOffset
                                                        layout:(MPSGraphTensorNamedDataLayout) layout
                                                          name:(NSString * _Nullable) name;
# 样例类声明开始
- (MPSGraphTensor * _Nonnull) sampleGridWithSourceTensor:(MPSGraphTensor * _Nonnull) source
                                        coordinateTensor:(MPSGraphTensor * _Nonnull) coordinates
                                                  layout:(MPSGraphTensorNamedDataLayout) layout
                                    normalizeCoordinates:(BOOL) normalizeCoordinates
                                     relativeCoordinates:(BOOL) relativeCoordinates
                                            alignCorners:(BOOL) alignCorners
                                             paddingMode:(MPSGraphPaddingMode) paddingMode
                                            samplingMode:(MPSGraphResizeMode) samplingMode
                                           constantValue:(double) constantValue
                                                    name:(NSString * _Nullable) name;
# 样例方法声明：使用源张量和坐标张量进行网格采样，返回一个 MPSGraphTensor 对象
- (MPSGraphTensor * _Nonnull) sampleGridWithSourceTensor:(MPSGraphTensor * _Nonnull) source
                                        coordinateTensor:(MPSGraphTensor * _Nonnull) coordinates
                                                  layout:(MPSGraphTensorNamedDataLayout) layout
                                    normalizeCoordinates:(BOOL) normalizeCoordinates
                                     relativeCoordinates:(BOOL) relativeCoordinates
                                            alignCorners:(BOOL) alignCorners
                                             paddingMode:(MPSGraphPaddingMode) paddingMode
                                     nearestRoundingMode:(MPSGraphResizeNearestRoundingMode) nearestRoundingMode
                                           constantValue:(double) constantValue
                                                    name:(NSString * _Nullable) name;
# 样例方法声明：使用源张量进行截断操作，返回一个 MPSGraphTensor 对象
- (MPSGraphTensor * _Nonnull) truncateWithTensor:(MPSGraphTensor * _Nonnull) tensor
                                            name:(NSString * _Nullable) name;

@end
# 样例类声明结束
```