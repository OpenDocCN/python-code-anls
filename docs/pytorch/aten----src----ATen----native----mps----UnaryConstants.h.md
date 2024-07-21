# `.\pytorch\aten\src\ATen\native\mps\UnaryConstants.h`

```py
#pragma once

const char* UNARY_KERNEL_TEMPLATE = R"METAL(
#include <metal_stdlib>
using namespace metal;

constant float a[4] = {{0.886226899, -1.645349621, 0.914624893, -0.140543331}};
constant float b[4] = {{-2.118377725, 1.442710462, -0.329097515, 0.012229801}};
constant float c[4] = {{-1.970840454, -1.624906493, 3.429567803, 1.641345311}};
constant float d[2] = {{3.543889200, 1.637067800}};

// 定义计算 erfinv 函数的 Metal 内核
kernel void erfinv_kernel( device {0} *output [[buffer(0)]],
                           device {1} *input [[buffer(1)]],
                           uint index [[thread_position_in_grid]]) {

  float y = input[index];
  float x, z, num, dem; /*working variables */
  /* coefficients in rational expansion */

  float y_abs = abs(y);
  // 处理输入 y 的绝对值大于等于 1.0 的情况
  if (y_abs >= 1.0f) {
    // 如果 y 的绝对值大于 1.0，则输出结果为 NaN 或符号与 y 相同的无穷大
    output[index] = {0}( y_abs > 1.0f ? NAN : copysign(INFINITY, y));
    return;
  }
  // 处理输入 y 的绝对值小于等于 0.7 的情况
  if (y_abs <= 0.7f) {
    z = y * y;
    // 计算分子
    num = ((a[3] * z + a[2]) * z + a[1])*z + a[0];
    // 计算分母
    dem = (((b[3] * z + b[2]) * z + b[1]) * z +b[0]) * z + 1.0f;
    // 计算结果 x
    x = y * num / dem;
  } else {
    z = sqrt(-1.0f*log((1.0-y_abs)/2.0));
    // 计算分子
    num = ((c[3] * z + c[2]) * z + c[1]) * z + c[0];
    // 计算分母
    dem = (d[1] * z + d[0]) * z + 1.0f;
    // 计算结果 x
    x = copysign(num, y) / dem;
  }

  // 将计算结果 x 存入输出数组
  output[index] = {0}(x);
}

// 定义计算 exp 函数的 Metal 内核
kernel void exp_kernel( device {0} *output [[buffer(0)]],
                        device {1} *input [[ buffer(1)]],
                        uint index [[thread_position_in_grid]]) {
  // 计算输入指数函数的值，并将结果存入输出数组
  output[index] = {0}(precise::exp(input[index]));
}

// 定义计算复数 exp 函数的 Metal 内核
kernel void exp_complex_kernel( device {0}2 *output [[buffer(0)]],
                                device {0}2 *input [[ buffer(1)]],
                                uint index [[thread_position_in_grid]]) {
  // 计算复数指数函数的实部和虚部，并将结果存入输出数组
  output[index].x = {0}(precise::exp(input[index].x)*precise::cos(input[index].y));
  output[index].y = {0}(precise::exp(input[index].x)*precise::sin(input[index].y));
}

// 定义计算双曲正切函数的 Metal 内核
kernel void tanh_kernel( device {0} *output [[buffer(0)]],
                        device {1} *input [[ buffer(1)]],
                        uint index [[thread_position_in_grid]]) {
  // 计算输入双曲正切函数的值，并将结果存入输出数组
  output[index] = {0}(precise::tanh(input[index]));
}

// 如果 Metal 版本大于等于 3.1，定义 bfloat2 类型的点乘函数 dot
#if __METAL_VERSION__ >= 310
bfloat dot(bfloat2 a, bfloat2 b) {
  return a.x * b.x + a.y * b.y;
}
#endif

// 定义复数除法模板函数
template<typename T>
T complex_div(T a, T b) {
  auto denom = dot(b, b);
  // 返回复数除法结果
  return T(dot(a, b), a.y * b.x - a.x * b.y)/denom;
}

// 定义计算复数双曲正切函数的 Metal 内核
kernel void tanh_complex_kernel( device {0}2 *output [[buffer(0)]],
                                 device {0}2 *input [[ buffer(1)]],
                                 uint index [[thread_position_in_grid]]) {
  // 计算复数双曲正切函数的值，并将结果存入输出数组
  // tanh(x+iy)=(tanh(x)+itan(y))/(1+itahnh(x)*tan(y));
  auto tanh_x = {0}(precise::tanh(input[index].x));
  auto tan_y = {0}(precise::tan(input[index].y));
  output[index] = complex_div({0}2(tanh_x, tan_y), {0}2({0}(1), tanh_x * tan_y));
}
)METAL";
```