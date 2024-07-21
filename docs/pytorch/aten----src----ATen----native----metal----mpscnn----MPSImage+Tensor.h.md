# `.\pytorch\aten\src\ATen\native\metal\mpscnn\MPSImage+Tensor.h`

```
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>


// 导入 MetalPerformanceShaders 框架中的头文件，用于使用 Metal 性能框架中的图像处理功能



#include <vector>


// 导入 C++ 标准库中的 vector 头文件，以便在 Objective-C++ 中使用 std::vector 容器



@interface MPSImage (Tensor)


// 定义一个类别 (Category) MPSImage，扩展现有的 MPSImage 类，添加名为 Tensor 的方法



- (std::vector<int64_t>)sizes;


// 声明一个实例方法 sizes，返回类型为 std::vector<int64_t>，用于获取图像的尺寸信息



- (int64_t)readCount;


// 声明一个实例方法 readCount，返回类型为 int64_t，用于获取图像的读取次数



- (BOOL)isTemporaryImage;


// 声明一个实例方法 isTemporaryImage，返回类型为 BOOL，用于判断图像是否为临时图像



- (void)markRead;


// 声明一个实例方法 markRead，返回类型为 void，用于标记图像为已读



- (void)recycle;


// 声明一个实例方法 recycle，返回类型为 void，用于回收图像资源



@end


// 结束类别的声明
```