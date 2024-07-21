# `.\pytorch\torch\csrc\jit\backends\coreml\objc\PTMCoreMLCompiler.h`

```
#import <CoreML/CoreML.h>


// 导入 CoreML 框架，使得 CoreML 的类和方法可用



#include <string>


// 引入 C++ 标准库中的 string 头文件，以支持 C++ 中的字符串操作



NS_ASSUME_NONNULL_BEGIN


// 进入非空指针假设的命名空间开始



@interface PTMCoreMLCompiler : NSObject


// 定义 PTMCoreMLCompiler 类，继承自 NSObject



+ (void)setCacheDirectory:(const std::string&)dir;


// 设置缓存目录的静态方法声明，参数为 C++ 标准库的 string 引用



+ (NSString*)cacheDirectory;


// 获取缓存目录的静态方法声明，返回 NSString 类型对象



+ (BOOL)compileModel:(const std::string&)modelSpecs modelID:(const std::string&)modelID;


// 编译模型的静态方法声明，接受模型规格和模型ID作为参数，返回布尔值



+ (nullable MLModel*)loadModel:(const std::string)modelID
                       backend:(const std::string)backend
             allowLowPrecision:(BOOL)allowLowPrecision
                         error:(NSError**)error;


// 载入模型的静态方法声明，接受模型ID、后端、是否允许低精度和错误信息作为参数，返回 MLModel 对象或者空值



@end


// 结束 PTMCoreMLCompiler 类的声明



NS_ASSUME_NONNULL_END


// 结束非空指针假设的命名空间
```