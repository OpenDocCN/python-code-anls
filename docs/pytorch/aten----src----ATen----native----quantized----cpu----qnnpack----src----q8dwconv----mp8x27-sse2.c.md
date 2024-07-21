# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\q8dwconv\mp8x27-sse2.c`

```py
/*
 * 包含必要的头文件
 */
#include <immintrin.h>

#include <math.h>
#include <qnnpack/q8dwconv.h>

/*
 * 定义一个函数 pytorch_q8dwconv_ukernel_mp8x27__sse2，用于实现量化的深度可分离卷积操作。
 * 函数参数说明：
 * - channels: 输入通道数
 * - output_height: 输出图像的高度
 * - output_width: 输出图像的宽度
 * - input: 输入数据的指针数组
 * - weights: 卷积核权重
 * - outacc32: 输出的累加器，用于累积结果
 * - output: 输出数据的指针
 * - input_row_stride: 输入数据行的跨度
 * - input_col_stride: 输入数据列的跨度
 * - output_increment: 输出数据的增量
 * - quantization_params: 量化参数结构体
 */
void pytorch_q8dwconv_ukernel_mp8x27__sse2(
    size_t channels,
    size_t output_height,
    size_t output_width,
    const uint8_t** input,
    const void* weights,
    int32_t* outacc32,
    uint8_t* output,
    size_t input_row_stride,
    size_t input_col_stride,
    size_t output_increment,
    const union pytorch_qnnp_conv_quantization_params
        quantization_params[RESTRICT_STATIC 1]) {
  
  /*
   * 从量化参数结构体中提取相关参数
   */
  const int16_t input_zero_point =
      quantization_params->sse2.input_zero_point[0];
  const uint8_t* kernel_zero_points =
      quantization_params->sse2.kernel_zero_points;
  const float* requantization_scales =
      quantization_params->sse2.requantization_scales;
  const int16_t output_zero_point =
      quantization_params->sse2.output_zero_point[0];
  const uint8_t output_min = quantization_params->sse2.output_min[0];
  const uint8_t output_max = quantization_params->sse2.output_max[0];

  /*
   * 定义一个联合类型，用于在 uint8_t* 和 int32_t* 之间进行转换
   */
  union {
    const uint8_t* as_uint8_ptr;
    const int32_t* as_int32_ptr;
  } weights_ptr = {weights};

  /*
   * 定义常量
   */
  const size_t cr_block = 8;
  const size_t kernel_depth = 3;
  const size_t kernel_height = 3;
  const size_t kernel_width = 3;

  /*
   * 计算分组数
   */
  const size_t num_groups = ((channels - 1) / cr_block) + 1;

  /*
   * 计算 yz 块大小和偏置大小
   */
  const size_t yz_block = kernel_depth * kernel_height;
  const size_t yz_bias_size = (cr_block * sizeof(int32_t));
  const size_t yz_weight_size = yz_block * cr_block;

  /*
   * 外层循环遍历输出图像的每一行
   */
  for (size_t output_y = 0; output_y < output_height; output_y++) {
    /*
     * 指向输入数据的行起始位置
     */
    const uint8_t** input_row_start = input;

    /*
     * 内层循环遍历输出图像的每一列
     */
    for (size_t output_x = 0; output_x < output_width; output_x++) {
      /*
       * 遍历每个通道
       */
      for (size_t c = 0; c < channels; c++) {
        /*
         * 初始化累加器，从权重中获取偏置值
         */
        int32_t accumulator =
            (weights_ptr.as_int32_ptr +
             ((c / cr_block) * (yz_bias_size + yz_weight_size) /
              sizeof(int32_t)))[c % cr_block];

        /*
         * 内层循环遍历卷积核的每一个像素
         */
        for (int x = 0; x < kernel_width; x++) {
          for (int y = 0; y < kernel_height; y++) {
            for (int z = 0; z < kernel_depth; z++) {
              /*
               * 从输入数据中获取像素值，并进行类型转换
               */
              int32_t input_val =
                  (int32_t)(input
                                [z + kernel_depth * y +
                                 kernel_depth * kernel_height * x][c]);
            }
          }
        }
      }
      /*
       * 更新输入数据的指针，指向下一列数据
       */
      input = (const uint8_t**)((uint8_t*)input_row_start + input_row_stride);
    }
  }
}
```