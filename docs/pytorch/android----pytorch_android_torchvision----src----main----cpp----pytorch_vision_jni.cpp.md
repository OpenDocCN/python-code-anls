# `.\pytorch\android\pytorch_android_torchvision\src\main\cpp\pytorch_vision_jni.cpp`

```
// 引入必要的头文件
#include <cassert>
#include <cmath>
#include <vector>

// 引入 JNI 头文件
#include "jni.h"

// 定义命名空间 pytorch_vision_jni
namespace pytorch_vision_jni {

// JNI 方法：对 YUV420 图像进行中心裁剪并转换为浮点缓冲区
static void imageYUV420CenterCropToFloatBuffer(
    JNIEnv* jniEnv,                        // JNI 环境指针
    jclass,                                // 调用该方法的类
    jobject yBuffer,                       // Y 数据缓冲区对象
    jint yRowStride,                       // Y 数据行跨度
    jint yPixelStride,                      // Y 数据像素跨度
    jobject uBuffer,                       // U 数据缓冲区对象
    jobject vBuffer,                       // V 数据缓冲区对象
    jint uRowStride,                       // U 数据行跨度
    jint uvPixelStride,                     // UV 数据像素跨度
    jint imageWidth,                       // 图像宽度
    jint imageHeight,                      // 图像高度
    jint rotateCWDegrees,                  // 顺时针旋转角度
    jint tensorWidth,                      // 张量宽度
    jint tensorHeight,                     // 张量高度
    jfloatArray jnormMeanRGB,              // RGB 均值数组对象
    jfloatArray jnormStdRGB,               // RGB 标准差数组对象
    jobject outBuffer,                     // 输出浮点缓冲区对象
    jint outOffset,                        // 输出缓冲区偏移量
    jint memoryFormatCode) {               // 内存格式代码

  // 定义内存格式常量
  constexpr static int32_t kMemoryFormatContiguous = 1;
  constexpr static int32_t kMemoryFormatChannelsLast = 2;

  // 获取输出缓冲区的地址并转换为 float 指针
  float* outData = (float*)jniEnv->GetDirectBufferAddress(outBuffer);

  // 从 jnormMeanRGB 和 jnormStdRGB 数组中获取 RGB 均值和标准差
  jfloat normMeanRGB[3];
  jfloat normStdRGB[3];
  jniEnv->GetFloatArrayRegion(jnormMeanRGB, 0, 3, normMeanRGB);
  jniEnv->GetFloatArrayRegion(jnormStdRGB, 0, 3, normStdRGB);

  // 初始化宽度和高度调整后的值
  int widthAfterRtn = imageWidth;
  int heightAfterRtn = imageHeight;

  // 检查是否为奇数角度旋转
  bool oddRotation = rotateCWDegrees == 90 || rotateCWDegrees == 270;
  if (oddRotation) {
    widthAfterRtn = imageHeight;
    heightAfterRtn = imageWidth;
  }

  // 计算旋转后的裁剪宽度和高度
  int cropWidthAfterRtn = widthAfterRtn;
  int cropHeightAfterRtn = heightAfterRtn;

  // 根据张量的宽高比例调整裁剪后的宽度和高度
  if (tensorWidth * heightAfterRtn <= tensorHeight * widthAfterRtn) {
    cropWidthAfterRtn = tensorWidth * heightAfterRtn / tensorHeight;
  } else {
    cropHeightAfterRtn = tensorHeight * widthAfterRtn / tensorWidth;
  }

  // 计算旋转前的裁剪宽度和高度
  int cropWidthBeforeRtn = cropWidthAfterRtn;
  int cropHeightBeforeRtn = cropHeightAfterRtn;
  if (oddRotation) {
    cropWidthBeforeRtn = cropHeightAfterRtn;
    cropHeightBeforeRtn = cropWidthAfterRtn;
  }

  // 计算裁剪的偏移量
  const int offsetX = (imageWidth - cropWidthBeforeRtn) / 2.f;
  const int offsetY = (imageHeight - cropHeightBeforeRtn) / 2.f;

  // 获取 YUV 数据的直接缓冲区地址
  const uint8_t* yData = (uint8_t*)jniEnv->GetDirectBufferAddress(yBuffer);
  const uint8_t* uData = (uint8_t*)jniEnv->GetDirectBufferAddress(uBuffer);
  const uint8_t* vData = (uint8_t*)jniEnv->GetDirectBufferAddress(vBuffer);

  // 计算缩放比例和 UV 行跨度
  float scale = cropWidthAfterRtn / tensorWidth;
  int uvRowStride = uRowStride;
  int cropXMult = 1;
  int cropYMult = 1;
  int cropXAdd = offsetX;
  int cropYAdd = offsetY;

  // 根据旋转角度调整裁剪的倍数和偏移量
  if (rotateCWDegrees == 90) {
    cropYMult = -1;
    cropYAdd = offsetY + (cropHeightBeforeRtn - 1);
  } else if (rotateCWDegrees == 180) {
    cropXMult = -1;
    cropXAdd = offsetX + (cropWidthBeforeRtn - 1);
    cropYMult = -1;
    cropYAdd = offsetY + (cropHeightBeforeRtn - 1);
  } else if (rotateCWDegrees == 270) {
    cropXMult = -1;
    cropXAdd = offsetX + (cropWidthBeforeRtn - 1);
  }
    // 计算裁剪后的 x 起始位置
    cropXAdd = offsetX + (cropWidthBeforeRtn - 1);
  }

  // 计算归一化均值和标准差的倍数值
  float normMeanRm255 = 255 * normMeanRGB[0];
  float normMeanGm255 = 255 * normMeanRGB[1];
  float normMeanBm255 = 255 * normMeanRGB[2];
  float normStdRm255 = 255 * normStdRGB[0];
  float normStdGm255 = 255 * normStdRGB[1];
  float normStdBm255 = 255 * normStdRGB[2];

  // 初始化变量用于存储像素索引和颜色分量
  int xBeforeRtn, yBeforeRtn;
  int yi, yIdx, uvIdx, ui, vi, a0, ri, gi, bi;
  int channelSize = tensorWidth * tensorHeight;
  // 为了避免在循环中使用分支，部分代码有所重复
  if (memoryFormatCode == kMemoryFormatContiguous) {
    // 计算输出数据的偏移量
    int wr = outOffset;
    int wg = wr + channelSize;
    int wb = wg + channelSize;
    // 循环遍历图像的每个像素
    for (int y = 0; y < tensorHeight; y++) {
      for (int x = 0; x < tensorWidth; x++) {
        // 计算经过裁剪和缩放后的像素位置
        xBeforeRtn = cropXAdd + cropXMult * (int)(x * scale);
        yBeforeRtn = cropYAdd + cropYMult * (int)(y * scale);
        // 计算 YUV 和 RGB 数据的索引
        yIdx = yBeforeRtn * yRowStride + xBeforeRtn * yPixelStride;
        uvIdx =
            (yBeforeRtn >> 1) * uvRowStride + (xBeforeRtn >> 1) * uvPixelStride;
        // 从数据数组中获取 YUV 值
        ui = uData[uvIdx];
        vi = vData[uvIdx];
        yi = yData[yIdx];
        // 对 YUV 进行归一化处理
        yi = (yi - 16) < 0 ? 0 : (yi - 16);
        ui -= 128;
        vi -= 128;
        // 计算 RGB 值
        a0 = 1192 * yi;
        ri = (a0 + 1634 * vi) >> 10;
        gi = (a0 - 833 * vi - 400 * ui) >> 10;
        bi = (a0 + 2066 * ui) >> 10;
        // 将计算得到的 RGB 值进行范围限制
        ri = ri > 255 ? 255 : ri < 0 ? 0 : ri;
        gi = gi > 255 ? 255 : gi < 0 ? 0 : gi;
        bi = bi > 255 ? 255 : bi < 0 ? 0 : bi;
        // 归一化并存储结果到输出数据数组
        outData[wr++] = (ri - normMeanRm255) / normStdRm255;
        outData[wg++] = (gi - normMeanGm255) / normStdGm255;
        outData[wb++] = (bi - normMeanBm255) / normStdBm255;
      }
    }
  } else if (memoryFormatCode == kMemoryFormatChannelsLast) {
    // 计算输出数据的偏移量
    int wc = outOffset;
    // 循环遍历图像的每个像素
    for (int y = 0; y < tensorHeight; y++) {
      for (int x = 0; x < tensorWidth; x++) {
        // 计算经过裁剪和缩放后的像素位置
        xBeforeRtn = cropXAdd + cropXMult * (int)(x * scale);
        yBeforeRtn = cropYAdd + cropYMult * (int)(y * scale);
        // 计算 YUV 和 RGB 数据的索引
        yIdx = yBeforeRtn * yRowStride + xBeforeRtn * yPixelStride;
        uvIdx =
            (yBeforeRtn >> 1) * uvRowStride + (xBeforeRtn >> 1) * uvPixelStride;
        // 从数据数组中获取 YUV 值
        ui = uData[uvIdx];
        vi = vData[uvIdx];
        yi = yData[yIdx];
        // 对 YUV 进行归一化处理
        yi = (yi - 16) < 0 ? 0 : (yi - 16);
        ui -= 128;
        vi -= 128;
        // 计算 RGB 值
        a0 = 1192 * yi;
        ri = (a0 + 1634 * vi) >> 10;
        gi = (a0 - 833 * vi - 400 * ui) >> 10;
        bi = (a0 + 2066 * ui) >> 10;
        // 将计算得到的 RGB 值进行范围限制
        ri = ri > 255 ? 255 : ri < 0 ? 0 : ri;
        gi = gi > 255 ? 255 : gi < 0 ? 0 : gi;
        bi = bi > 255 ? 255 : bi < 0 ? 0 : bi;
        // 归一化并存储结果到输出数据数组
        outData[wc++] = (ri - normMeanRm255) / normStdRm255;
        outData[wc++] = (gi - normMeanGm255) / normStdGm255;
        outData[wc++] = (bi - normMeanBm255) / normStdBm255;
      }
    }
  } else {
    // 若内存格式码不匹配，则抛出异常
    jclass Exception = jniEnv->FindClass("java/lang/IllegalArgumentException");
    jniEnv->ThrowNew(Exception, "Illegal memory format code");
  }
} // namespace pytorch_vision_jni
```