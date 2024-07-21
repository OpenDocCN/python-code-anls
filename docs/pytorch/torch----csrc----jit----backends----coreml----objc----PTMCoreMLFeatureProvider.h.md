# `.\pytorch\torch\csrc\jit\backends\coreml\objc\PTMCoreMLFeatureProvider.h`

```
#import <ATen/ATen.h>
#import <CoreML/CoreML.h>

NS_ASSUME_NONNULL_BEGIN

@interface PTMCoreMLFeatureProvider : NSObject<MLFeatureProvider>
// 定义 PTMCoreMLFeatureProvider 类，实现 MLFeatureProvider 协议

- (instancetype)initWithFeatureNames:(NSSet<NSString*>*)featureNames;
// 初始化方法，接收一个特征名称的集合作为参数

- (void)clearInputTensors;
// 清空输入张量的方法，无返回值

- (void)setInputTensor:(const at::Tensor&)tensor forFeatureName:(NSString*)name;
// 设置特定特征名称的输入张量的方法，接收一个 ATen 张量类型的常量引用和一个特征名称字符串作为参数

@end

NS_ASSUME_NONNULL_END
// 结束 NS_ASSUME_NONNULL_BEGIN 块，标记为可空指针结束
```