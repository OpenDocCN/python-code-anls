# `.\pytorch\aten\src\ATen\native\cpu\Unfold2d.cpp`

```py
// 定义宏以禁用 Torch 操作符断言
#define TORCH_ASSERT_NO_OPERATORS

// 包含 ATen 库的调度和并行处理头文件
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>

// 包含 CPU 矢量化操作相关头文件
#include <ATen/cpu/vec/vec.h>

// 包含 Unfold2d 相关头文件
#include <ATen/native/Unfold2d.h>

// 包含 CPU 循环相关头文件
#include <ATen/native/cpu/Loops.h>

// 包含 C++ 10 实用工具库中的范围处理头文件
#include <c10/util/irange.h>

// 包含 CPU 实用工具函数头文件
#include <ATen/native/cpu/utils.h>

// 包含标准数学库头文件
#include <cmath>

// ATen 命名空间下的 native 命名空间
namespace at::native {

// 匿名命名空间开始，定义内部静态函数
namespace {

// 模板函数：向量化加法，用于将两个数组相加
template <typename scalar_t>
static inline void cadd(
    scalar_t* z,                   // 输出数组指针
    const scalar_t* x,             // 第一个输入数组指针
    const scalar_t* y,             // 第二个输入数组指针
    int64_t n) {                   // 数组长度
  using Vec = vec::Vectorized<scalar_t>;

  // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
  // 创建包含 z、x、y 指针的字符数组，用于向量化处理
  char* ptrs[] = {
      reinterpret_cast<char*>(z),                           // 输出数组
      reinterpret_cast<char*>(const_cast<scalar_t*>(x)),    // 第一个输入数组
      reinterpret_cast<char*>(const_cast<scalar_t*>(y))     // 第二个输入数组
  };

  // 调用向量化循环执行加法操作，处理长度为 n 的数组
  vectorized_loop(
      ptrs,
      n,
      -1,
      [](scalar_t x, scalar_t y) -> scalar_t { return x + y; },   // 标量加法操作
      [](Vec x, Vec y) -> Vec { return x + y; }                   // 向量化加法操作
  );
}

// 函数：二维展开累加，用于处理卷积操作中的展开数据
template <typename scalar_t>
static void unfolded2d_acc(
    scalar_t* finput_data,          // 展开后数据的输出数组指针
    scalar_t* input_data,           // 输入数据数组指针
    int64_t kH,                     // 卷积核的高度
    int64_t kW,                     // 卷积核的宽度
    int64_t dH,                     // 高度方向的步长
    int64_t dW,                     // 宽度方向的步长
    int64_t padH,                   // 高度方向的填充
    int64_t padW,                   // 宽度方向的填充
    int64_t n_input_plane,          // 输入平面数量
    int64_t input_height,           // 输入数据的高度
    int64_t input_width,            // 输入数据的宽度
    int64_t output_height,          // 输出数据的高度
    int64_t output_width) {         // 输出数据的宽度

  // 使用 ATen 的并行处理，按照输入平面的数量并行执行以下操作
  at::parallel_for(0, n_input_plane, 0, [&](int64_t start, int64_t end) {
    // 对于给定的 nip 范围内的每个值，执行循环迭代
    for (const auto nip : c10::irange(start, end)) {
      // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
      // 定义变量 kw, kh, y, x，并初始化为0
      int64_t kw, kh, y, x;
      // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
      // 定义变量 ix, iy，并初始化为0
      int64_t ix, iy;
      // 遍历卷积核的高度 kH 和宽度 kW
      for (kh = 0; kh < kH; kh++) {
        for (kw = 0; kw < kW; kw++) {
          // 计算源数据（src）和目标数据（dst）的指针位置
          scalar_t* src = finput_data +
              nip * ((size_t)kH * kW * output_height * output_width) +
              kh * ((size_t)kW * output_height * output_width) +
              kw * ((size_t)output_height * output_width);
          scalar_t* dst =
              input_data + nip * ((size_t)input_height * input_width);
          // 如果存在填充(pad)，初始化填充左边界和右边界
          if (padW > 0 || padH > 0) {
            // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
            // 初始化变量 lpad 和 rpad
            int64_t lpad, rpad;
            // 遍历输出图像的高度 output_height
            for (y = 0; y < output_height; y++) {
              // 计算输入图像在 y 方向上的索引 iy
              iy = (int64_t)y * dH - padH + kh;
              // 如果 iy 超出输入图像高度范围，跳过当前迭代
              if (iy < 0 || iy >= input_height) {
              } else {
                // 如果滑动窗口的宽度 dW 等于 1
                if (dW == 1) {
                  // 计算输入图像在 x 方向上的索引 ix，并初始化 lpad 和 rpad
                  ix = 0 - padW + kw;
                  lpad = std::max<int64_t>(0, padW - kw);
                  rpad = std::max<int64_t>(0, padW - (kW - kw - 1));
                  // 计算目标数据片段的指针位置 dst_slice 和执行加法操作
                  scalar_t* dst_slice =
                      dst + (size_t)iy * input_width + ix + lpad;
                  cadd(
                      dst_slice,
                      dst_slice,
                      src + (size_t)y * output_width + lpad,
                      output_width - lpad - rpad);
                } else {
                  // 遍历输出图像的宽度 output_width
                  for (x = 0; x < output_width; x++) {
                    // 计算输入图像在 x 方向上的索引 ix
                    ix = (int64_t)x * dW - padW + kw;
                    // 如果 ix 超出输入图像宽度范围，跳过当前迭代
                    if (ix < 0 || ix >= input_width) {
                    } else {
                      // 计算目标数据片段的指针位置 dst_slice 和执行加法操作
                      scalar_t* dst_slice = dst + (size_t)iy * input_width + ix;
                      *dst_slice = *dst_slice + src[(size_t)y * output_width + x];
                    }
                  }
                }
              }
            }
          } else {
            // 如果不存在填充(pad)，遍历输出图像的高度 output_height
            for (y = 0; y < output_height; y++) {
              // 计算输入图像在 y 方向上的索引 iy
              iy = (int64_t)y * dH + kh;
              // 计算输入图像在 x 方向上的索引 ix
              ix = 0 + kw;
              // 如果滑动窗口的宽度 dW 等于 1
              if (dW == 1) {
                // 计算目标数据片段的指针位置 dst_slice 和执行加法操作
                scalar_t* dst_slice = dst + (size_t)iy * input_width + ix;
                cadd(
                    dst_slice,
                    dst_slice,
                    src + (size_t)y * output_width,
                    output_width);
              } else {
                // 遍历输出图像的宽度 output_width
                for (x = 0; x < output_width; x++) {
                  // 计算目标数据片段的指针位置 dst_slice 和执行加法操作
                  scalar_t* dst_slice =
                      dst + (size_t)iy * input_width + ix + x * dW;
                  *dst_slice = *dst_slice + src[(size_t)y * output_width + x];
                }
              }
            }
          }
        }
      }
    }
// 定义了一个静态函数，用于将展开的二维数据累加到输入数据中，假设数据按通道为最后维度排列
template <typename scalar_t>
static void unfolded2d_acc_channels_last(
    scalar_t* finput_data,            // 展开后的输入数据指针
    scalar_t* input_data,             // 输入数据指针
    int64_t kH,                       // 卷积核的高度
    int64_t kW,                       // 卷积核的宽度
    int64_t dH,                       // 高度方向的步长
    int64_t dW,                       // 宽度方向的步长
    int64_t padH,                     // 高度方向的填充
    int64_t padW,                     // 宽度方向的填充
    int64_t n_input_plane,            // 输入通道数
    int64_t input_height,             // 输入数据的高度
    int64_t input_width,              // 输入数据的宽度
    int64_t output_height,            // 输出数据的高度
    int64_t output_width) {           // 输出数据的宽度

  // 遍历每一个输出像素点
  for (int64_t y = 0; y < output_height; y++) {
    for (int64_t x = 0; x < output_width; x++) {
      // 计算当前输出像素点对应的展开数据的起始地址
      scalar_t* src = finput_data + y * output_width * kH * kW * n_input_plane + x * kH * kW * n_input_plane;
      // 输入数据的起始地址
      scalar_t* dst = input_data;

      // 如果存在填充，则需要考虑填充的影响
      if (padW > 0 || padH > 0) {
        // 遍历卷积核的每个位置
        for (int64_t kh = 0; kh < kH; kh++) {
          for (int64_t kw = 0; kw < kW; kw++) {
            // 计算输入数据中的索引位置
            int64_t iy = y * dH - padH + kh;
            int64_t ix = x * dW - padW + kw;
            // 检查索引是否在有效范围内
            if (iy < 0 || iy >= input_height || ix < 0 || ix >= input_width) {
              // 如果索引超出范围，不执行累加操作
            } else {
              // 计算在输入数据中的具体位置
              scalar_t* dst_slice = dst + iy * input_width * n_input_plane + ix * n_input_plane;
              scalar_t* src_slice = src + kh * kW * n_input_plane + kw * n_input_plane;
              // 执行累加操作，将展开的数据加到输入数据中
              cadd(dst_slice,
                   dst_slice,
                   src_slice,
                   n_input_plane);
            }
          }
        }
      } else {
        // 无填充的情况下，直接进行累加操作
        for (int64_t kh = 0; kh < kH; kh++) {
          for (int64_t kw = 0; kw < kW; kw++) {
            // 计算在输入数据中的具体位置
            int64_t iy = y * dH + kh;
            int64_t ix = x * dW + kw;
            scalar_t* dst_slice = dst + iy * input_width * n_input_plane + ix * n_input_plane;
            scalar_t* src_slice = src + kh * kW * n_input_plane + kw * n_input_plane;
            // 执行累加操作，将展开的数据加到输入数据中
            cadd(dst_slice,
                 dst_slice,
                 src_slice,
                 n_input_plane);
          }
        }
      }
    }
  }
}

/* note: due to write issues, this one cannot be parallelized as well as
 * unfolded2d_copy */
// 此函数由于写入问题，不能像 unfolded2d_copy 一样进行并行化
void unfolded2d_acc_kernel(
    ScalarType dtype,                  // 数据类型
    void *finput_data,                 // 展开后的输入数据指针
    void *input_data,                  // 输入数据指针
    int64_t kH,                        // 卷积核的高度
    int64_t kW,                        // 卷积核的宽度
    int64_t dH,                        // 高度方向的步长
    int64_t dW,                        // 宽度方向的步长
    int64_t padH,                      // 高度方向的填充
    int64_t padW,                      // 宽度方向的填充
    int64_t n_input_plane,             // 输入通道数
    int64_t input_height,              // 输入数据的高度
    int64_t input_width,               // 输入数据的宽度
    int64_t output_height,             // 输出数据的高度
    int64_t output_width,              // 输出数据的宽度
    bool is_channels_last) {           // 是否按通道为最后维度排列

  // 此函数假设 output_height*dH 不会导致 int64_t 溢出
  // output_width*dW 也不会导致 int64_t 溢出

  // 如果按通道为最后维度排列
  if (is_channels_last) {
    // 根据数据类型分发函数，调用 unfolded2d_acc_channels_last 处理函数
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::BFloat16, at::ScalarType::Half, dtype, "unfolded2d_acc_channels_last", [&] {
      unfolded2d_acc_channels_last(
          static_cast<scalar_t*>(finput_data),
          static_cast<scalar_t*>(input_data),
          kH, kW,
          dH, dW,
          padH, padW,
          n_input_plane,
          input_height,
          input_width,
          output_height,
          output_width);
     });
  } else {
    // 根据 AT_DISPATCH_FLOATING_TYPES_AND2 宏展开并调用 unfolded2d_acc 函数，处理二维张量展开操作
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::BFloat16, at::ScalarType::Half, dtype, "unfolded2d_acc", [&] {
      // 转换并传递 finput_data 和 input_data 的指针给 unfolded2d_acc 函数
      unfolded2d_acc(
          static_cast<scalar_t*>(finput_data),  // finput_data 的类型转换为 scalar_t* 并传递
          static_cast<scalar_t*>(input_data),   // input_data 的类型转换为 scalar_t* 并传递
          kH, kW,                               // kH 和 kW 分别表示 kernel 的高度和宽度
          dH, dW,                               // dH 和 dW 分别表示步长（stride）的高度和宽度
          padH, padW,                           // padH 和 padW 分别表示填充（padding）的高度和宽度
          n_input_plane,                        // 输入张量的通道数
          input_height, input_width,            // 输入张量的高度和宽度
          output_height, output_width);         // 输出张量的高度和宽度
      });
  }
// 并行化地处理 channels last 格式的未展开输入数据到目标数组的复制操作
template <typename scalar_t>
static void unfolded2d_copy_channels_last(
    const scalar_t* input_data,  // 输入数据的指针，按照 scalar_t 类型解释
    scalar_t* finput_data,       // 目标数组的指针，按照 scalar_t 类型解释，用于存储未展开的输入数据
    int64_t kH,                  // 卷积核的高度
    int64_t kW,                  // 卷积核的宽度
    int64_t dH,                  // 高度方向的步幅
    int64_t dW,                  // 宽度方向的步幅
    int64_t padH,                // 高度方向的填充大小
    int64_t padW,                // 宽度方向的填充大小
    int64_t n_input_plane,       // 输入通道数
    int64_t input_height,        // 输入数据的高度
    int64_t input_width,         // 输入数据的宽度
    int64_t output_height,       // 输出数据的高度
    int64_t output_width) {      // 输出数据的宽度

  // 使用 ATen 提供的并行化函数对每个输出索引进行处理
  at::parallel_for(0, output_height * output_width, 0, [&](int64_t start, int64_t end) {
    int64_t y = 0;               // 初始化输出数据的高度索引
    int64_t x = 0;               // 初始化输出数据的宽度索引
    data_index_init(start, y, output_height, x, output_width);  // 根据 start 初始化 y 和 x

    // 遍历每个输出位置的数据处理
    for (const auto k C10_UNUSED: c10::irange(start, end)) {
      scalar_t* dst = finput_data + y * output_width * kH * kW * n_input_plane + x * kH * kW * n_input_plane;
      // 计算目标数组中当前位置的起始地址

      const scalar_t* src = input_data;  // 源数据的指针，按照 scalar_t 类型解释

      // 如果存在填充，则需要特殊处理
      if (padW > 0 || padH > 0) {
        // 处理有填充的情况
        for (int64_t kh = 0; kh < kH; kh++) {
          for (int64_t kw = 0; kw < kW; kw++) {
            int64_t iy = y * dH - padH + kh;  // 计算输入数据的高度索引
            int64_t ix = x * dW - padW + kw;  // 计算输入数据的宽度索引
            if (iy < 0 || iy >= input_height || ix < 0 || ix >= input_width) {
              // 如果索引超出输入数据范围，则填充零
              memset(dst + kh * kW * n_input_plane + kw * n_input_plane,
                     0,
                     sizeof(scalar_t) * n_input_plane);
            } else {
              // 否则从输入数据拷贝对应位置的数据到目标数组
              memcpy(dst + kh * kW * n_input_plane + kw * n_input_plane,
                     src + iy * input_width * n_input_plane + ix * n_input_plane,
                     sizeof(scalar_t) * n_input_plane);
            }
          }
        }
      } else {
        // 处理无填充的情况
        for (int64_t kh = 0; kh < kH; kh++) {
          for (int64_t kw = 0; kw < kW; kw++) {
            int64_t iy = y * dH + kh;  // 计算输入数据的高度索引
            int64_t ix = x * dW + kw;  // 计算输入数据的宽度索引
            // 从输入数据拷贝对应位置的数据到目标数组
            memcpy(dst + kh * kW * n_input_plane + kw * n_input_plane,
                   src + iy * input_width * n_input_plane + ix * n_input_plane,
                   sizeof(scalar_t) * n_input_plane);
          }
        }
      }
      // 移动到下一个输出位置的索引
      data_index_step(y, output_height, x, output_width);
    }
  });
}
    # 如果 unfold_type 是 "channels_last"，则执行以下代码块
    AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::BFloat16, at::ScalarType::Half, dtype, "unfolded2d_copy_channels_last", [&] {
      # 调用 unfolded2d_copy_channels_last 函数，复制输入数据到 finput_data
      unfolded2d_copy_channels_last(
          static_cast<const scalar_t*>(input_data),  # 输入数据的强制类型转换为 scalar_t 指针
          static_cast<scalar_t*>(finput_data),       # finput_data 的强制类型转换为 scalar_t 指针
          kH, kW,                                     # kernel 的高度和宽度
          dH, dW,                                     # 水平和垂直的步长
          padH, padW,                                 # 上下和左右的填充数
          n_input_plane,                              # 输入平面数
          input_height, input_width,                  # 输入数据的高度和宽度
          output_height, output_width);               # 输出数据的高度和宽度
    });
  } else {
    # 如果 unfold_type 不是 "channels_last"，则执行以下代码块
    AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::BFloat16, at::ScalarType::Half, dtype, "unfolded2d_copy", [&] {
      # 调用 unfolded2d_copy 函数，复制输入数据到 finput_data
      unfolded2d_copy(
          static_cast<const scalar_t*>(input_data),   # 输入数据的强制类型转换为 scalar_t 指针
          static_cast<scalar_t*>(finput_data),        # finput_data 的强制类型转换为 scalar_t 指针
          kH, kW,                                     # kernel 的高度和宽度
          dH, dW,                                     # 水平和垂直的步长
          padH, padW,                                 # 上下和左右的填充数
          n_input_plane,                              # 输入平面数
          input_height, input_width,                  # 输入数据的高度和宽度
          output_height, output_width);               # 输出数据的高度和宽度
    });
  }
}

} // namespace


注释：

// 结束当前的命名空间定义，匹配之前的namespace开头



REGISTER_DISPATCH(unfolded2d_copy_stub, &unfolded2d_copy_kernel);


注释：

// 注册 unfolded2d_copy_stub 对应的函数指针 unfolded2d_copy_kernel



REGISTER_DISPATCH(unfolded2d_acc_stub, &unfolded2d_acc_kernel);


注释：

// 注册 unfolded2d_acc_stub 对应的函数指针 unfolded2d_acc_kernel



} // namespace at::native


注释：

// 结束当前的命名空间定义，命名空间为 at::native
```