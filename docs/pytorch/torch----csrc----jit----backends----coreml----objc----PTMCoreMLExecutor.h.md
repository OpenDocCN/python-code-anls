# `.\pytorch\torch\csrc\jit\backends\coreml\objc\PTMCoreMLExecutor.h`

```
#import <torch/csrc/jit/backends/coreml/objc/PTMCoreMLFeatureProvider.h>

# 导入 PTMCoreMLFeatureProvider.h 文件，这是 Torch 框架在 CoreML 后端的一个特定功能提供者的 Objective-C 头文件。


#import <CoreML/CoreML.h>

# 导入 CoreML 框架的头文件，这是苹果的机器学习框架，用于在 iOS 和 macOS 上部署机器学习模型。


NS_ASSUME_NONNULL_BEGIN

# 开始一个 Objective-C 的命名空间，定义在此命名空间中的对象引用默认为非空。


@interface PTMCoreMLExecutor : NSObject

# 定义一个名为 PTMCoreMLExecutor 的 Objective-C 类，继承自 NSObject。


@property(atomic, strong) MLModel* model;

# 声明 PTMCoreMLExecutor 类的属性 model，使用 atomic 和 strong 修饰符，用于存储 MLModel 对象，保证在多线程环境下的原子性操作。


- (instancetype)initWithFeatureNames:(NSArray<NSString*>*)featureNames;

# 定义 PTMCoreMLExecutor 类的初始化方法 initWithFeatureNames:，接受一个 NSArray<NSString*> 类型的 featureNames 参数作为输入。


- (void)setInputs:(c10::impl::GenericList)inputs;

# 定义 PTMCoreMLExecutor 类的方法 setInputs:，接受一个 c10::impl::GenericList 类型的 inputs 参数作为输入，用于设置输入数据。


- (id<MLFeatureProvider>)forward:(NSError**)error;

# 定义 PTMCoreMLExecutor 类的方法 forward:，接受一个 NSError** 类型的 error 参数作为输入，并返回一个实现了 MLFeatureProvider 协议的对象。


@end

# 结束 PTMCoreMLExecutor 类的声明。


NS_ASSUME_NONNULL_END

# 结束之前声明的 Objective-C 命名空间。
```