# `.\pytorch\aten\src\ATen\native\metal\MetalConvParams.h`

```py
#ifndef MetalConvParams_h
#define MetalConvParams_h

#include <c10/util/ArrayRef.h>

namespace at::native::metal {

// 定义了用于 Metal 设备上卷积操作的参数结构体 Conv2DParams
struct Conv2DParams final {
  Conv2DParams() {}  // 默认构造函数
  
  // 带参数的构造函数，用于初始化卷积参数
  Conv2DParams(
      c10::IntArrayRef inputSizes,   // 输入尺寸数组引用
      c10::IntArrayRef weightSizes,  // 权重尺寸数组引用
      c10::IntArrayRef padding,      // 填充数组引用
      c10::IntArrayRef stride,       // 步长数组引用
      c10::IntArrayRef dilation,     // 膨胀数组引用
      int64_t groups);               // 组数

  // 返回输出尺寸的数组
  std::vector<int64_t> output_sizes() const {
    return {N, OC, OH, OW};  // 返回包含批大小、输出通道数、输出高度和宽度的数组
  }

  // 判断是否为深度卷积
  bool isDepthwise() const {
    // 目前仅支持通道乘数为 1，即输入通道数等于输出通道数
    return G > 1 && IC == 1 && OC == G && OC == C;
  }

  int64_t N;   // 批大小
  int64_t C;   // 输入通道数
  int64_t H;   // 输入高度
  int64_t W;   // 输入宽度
  int64_t OC;  // 输出通道数
  int64_t IC;  // 输入通道数
  int64_t KH;  // 卷积核高度
  int64_t KW;  // 卷积核宽度
  int64_t SY;  // 步长 y 方向（高度）
  int64_t SX;  // 步长 x 方向（宽度）
  int64_t PY;  // 填充 y 方向（高度）
  int64_t PX;  // 填充 x 方向（宽度）
  int64_t DY;  // 膨胀 y 方向（高度）
  int64_t DX;  // 膨胀 x 方向（宽度）
  int64_t G;   // 组数
  int64_t OW;  // 输出宽度
  int64_t OH;  // 输出高度
};

} // namespace at::native::metal

#endif /* MetalConvParams_h */
```