# `.\pytorch\aten\src\ATen\native\Unfold3d.h`

```py
#pragma once
#include <c10/core/ScalarType.h>

namespace at::native {

// 声明一个函数 Unfold3dCopyCPU，用于在 CPU 上执行 3D 数据的展开复制操作
void Unfold3dCopyCPU(
    ScalarType dtype,                   // 数据类型
    const void *src,                    // 源数据指针
    int64_t C,                          // 输入通道数
    int64_t X_D, int64_t X_H, int64_t X_W,   // 输入数据的深度、高度、宽度
    int64_t Y_D, int64_t Y_H, int64_t Y_W,   // 输出数据的深度、高度、宽度
    int64_t kernel_d,                   // 卷积核的深度
    int64_t kernel_h,                   // 卷积核的高度
    int64_t kernel_w,                   // 卷积核的宽度
    int64_t stride_d,                   // 步长的深度方向
    int64_t stride_h,                   // 步长的高度方向
    int64_t stride_w,                   // 步长的宽度方向
    int64_t pad_d,                      // 填充的深度
    int64_t pad_h,                      // 填充的高度
    int64_t pad_w,                      // 填充的宽度
    void* dst                           // 目标数据指针
);

// 声明一个函数 Unfold3dAccCPU，用于在 CPU 上执行 3D 数据的展开累加操作
void Unfold3dAccCPU(
    ScalarType dtype,                   // 数据类型
    const void *src,                    // 源数据指针
    int64_t C,                          // 输入通道数
    int64_t X_D, int64_t X_H, int64_t X_W,   // 输入数据的深度、高度、宽度
    int64_t Y_D, int64_t Y_H, int64_t Y_W,   // 输出数据的深度、高度、宽度
    int64_t kernel_d,                   // 卷积核的深度
    int64_t kernel_h,                   // 卷积核的高度
    int64_t kernel_w,                   // 卷积核的宽度
    int64_t stride_d,                   // 步长的深度方向
    int64_t stride_h,                   // 步长的高度方向
    int64_t stride_w,                   // 步长的宽度方向
    int64_t pad_d,                      // 填充的深度
    int64_t pad_h,                      // 填充的高度
    int64_t pad_w,                      // 填充的宽度
    void *dst                           // 目标数据指针
);

} // namespace at::native
```