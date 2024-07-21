# `.\pytorch\aten\src\ATen\native\metal\mpscnn\tests\MetalOpTestRunner.h`

```
// 引入 Foundation 框架，包含 Objective-C 所需的基础功能
#import <Foundation/Foundation.h>
// 引入 C++ 的无序映射容器头文件
#include <unordered_map>

// MetalOpTestRunner 类的接口声明开始
@interface MetalOpTestRunner : NSObject

// 定义 testBlock 为返回 BOOL 的块对象类型
typedef BOOL(^testBlock)(void);

// 类方法，返回 MetalOpTestRunner 的单例实例
+ (instancetype)sharedInstance;

// 实例方法，返回一个字典，字典的键是 NSString 类型，值是 testBlock 块对象
- (NSDictionary<NSString *, testBlock> *)tests;

// MetalOpTestRunner 类的接口声明结束
@end
```