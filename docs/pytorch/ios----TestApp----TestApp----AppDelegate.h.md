# `.\pytorch\ios\TestApp\TestApp\AppDelegate.h`

```py
# 导入UIKit框架中的UIKit头文件，这是iOS开发的基本框架
#import <UIKit/UIKit.h>

# 定义一个AppDelegate类，继承自UIResponder类，并遵循UIApplicationDelegate协议
@interface AppDelegate : UIResponder <UIApplicationDelegate>

# 定义一个强引用的UIWindow属性，用于管理应用程序的窗口
@property(strong, nonatomic) UIWindow *window;

# 结束接口定义
@end
```