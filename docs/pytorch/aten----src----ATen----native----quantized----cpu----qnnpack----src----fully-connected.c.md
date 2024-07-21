# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\fully-connected.c`

```
/*
 * 本函数用于创建一个基于QNNPACK的量化全连接神经网络运算符。
 * 其中，输入参数如下：
 * - input_channels: 输入通道数
 * - output_channels: 输出通道数
 * - input_zero_point: 输入数据的零点
 * - kernel_zero_points: 每个卷积核的零点数组
 * - kernel: 卷积核的权重数组
 * - bias: 偏置数组
 * - output_zero_point: 输出数据的零点
 * - output_min: 输出数据的最小值
 * - output_max: 输出数据的最大值
 * - flags: 标志位
 * - requantization_scales: 重新量化的尺度因子数组
 * - fully_connected_out: 输出的全连接运算符指针
 */
enum pytorch_qnnp_status pytorch_qnnp_create_fully_connected_nc_q8(
    size_t input_channels,
    size_t output_channels,
    uint8_t input_zero_point,
    const uint8_t* kernel_zero_points,
    const uint8_t* kernel,
    const int32_t* bias,
    uint8_t output_zero_point,
    uint8_t output_min,
    uint8_t output_max,
    uint32_t flags,
    const float* requantization_scales,
    pytorch_qnnp_operator_t* fully_connected_out) {
  
  // 初始化本地变量
  pytorch_qnnp_operator_t fully_connected = NULL;
  enum pytorch_qnnp_status status = pytorch_qnnp_status_uninitialized;

  // 检查QNNPACK是否已经初始化
  if (!pytorch_qnnp_params.initialized) {
    pytorch_qnnp_log_error(
        "pytorch_qnnp_create_fully_connected_nc_q8 failed because QNNPACK is not properly initialized");
    goto error;
  }

  // 检查每个重新量化的尺度因子是否有效
  status = pytorch_qnnp_status_unsupported_parameter;
  for (int i = 0; i < output_channels; ++i) {
    if (requantization_scales[i] <= 0.0f ||
        !isnormal(requantization_scales[i])) {
      pytorch_qnnp_log_error(
          "failed to create fully connected operator with %.7g requantization scale: scale must be finite and positive",
          requantization_scales[i]);
      goto error;
    }
  }

  // 分配内存以存储全连接运算符的结构体
  status = pytorch_qnnp_status_out_of_memory;
  fully_connected = calloc(1, sizeof(struct pytorch_qnnp_operator));
  if (fully_connected == NULL) {
    pytorch_qnnp_log_error(
        "failed to allocate %zu bytes for pytorch_qnnp_operator structure",
        sizeof(struct pytorch_qnnp_operator));
    goto error;
  }

  // 根据QNNPACK的参数，计算所需的填充和步长
  const uint32_t nr = pytorch_qnnp_params.q8conv.nr;
  const uint32_t kr = pytorch_qnnp_params.q8conv.kr;
  const uint32_t n_stride = (output_channels + (nr - 1)) & -nr;
  const uint32_t k_stride = (input_channels + (kr - 1)) & -kr;

  // 分配内存以存储打包后的权重数据
  fully_connected->packed_weights =
      malloc(n_stride * (k_stride * sizeof(uint8_t) + sizeof(int32_t)));
  if (fully_connected->packed_weights == NULL) {
    pytorch_qnnp_log_error(
        "failed to allocate %zu bytes for packed weights",
        n_stride * (k_stride * sizeof(uint8_t) + sizeof(int32_t)));
    goto error;
  }

  // 使用第一个卷积核的零点初始化打包权重数据
  memset(
      fully_connected->packed_weights,
      kernel_zero_points[0],
      n_stride * (k_stride * sizeof(uint8_t) + sizeof(int32_t)));

  // 调用QNNPACK的函数，打包Q8 GEMM的权重
  pytorch_pack_q8gemm_w(
      output_channels,
      input_channels,
      nr,
      nr,
      kr,
#if !PYTORCH_QNNPACK_RUNTIME_QUANTIZATION
      input_zero_point,
      kernel_zero_points[0],
#endif
      kernel,
      bias,
#if PYTORCH_QNNPACK_RUNTIME_QUANTIZATION
      kernel_zero_points,
#endif
      fully_connected->packed_weights);



  fully_connected->groups = 1;
  // 设置全连接层的分组数为1，即无分组
  fully_connected->group_input_channels = input_channels;
  // 设置全连接层的分组输入通道数为输入通道数
  fully_connected->group_output_channels = output_channels;
  // 设置全连接层的分组输出通道数为输出通道数

  fully_connected->kernel_zero_point = kernel_zero_points[0];
  // 设置全连接层的内核零点为第一个内核零点值

  fully_connected->conv_quantization_params =
      pytorch_qnnp_compute_conv_quantization_params(
          input_zero_point,
          kernel_zero_points,
          requantization_scales,
          output_zero_point,
          output_min,
          output_max);
  // 计算全连接层的量化参数，包括输入零点、内核零点、重新量化比例、输出零点、输出范围的参数

  fully_connected->ukernel_type = pytorch_qnnp_ukernel_type_gemm;
  // 设置全连接层的微内核类型为GEMM（通用矩阵乘法）

  fully_connected->format = pytorch_qnnp_format_quint8;
  // 设置全连接层的数据格式为8位无符号整数（quint8）

  *fully_connected_out = fully_connected;
  // 将配置好的全连接层操作符指针赋值给输出指针

  return pytorch_qnnp_status_success;
  // 返回成功状态
error:
  pytorch_qnnp_delete_operator(fully_connected);
  // 在错误发生时，释放已分配的全连接层操作符内存
  return status;
}

enum pytorch_qnnp_status pytorch_qnnp_setup_fully_connected_nc_q8(
    pytorch_qnnp_operator_t fully_connected,
    size_t batch_size,
    const uint8_t* input,
    size_t input_stride,
    uint8_t* output,
    size_t output_stride) {
  if (!pytorch_qnnp_params.initialized) {
    pytorch_qnnp_log_error(
        "pytorch_qnnp_setup_fully_connected_nc_q8 failed because QNNPACK is not properly initialized");
    return pytorch_qnnp_status_uninitialized;
    // 如果 QNNPACK 没有正确初始化，则返回未初始化状态
  }

  if (batch_size == 0) {
    fully_connected->batch_size = 0;
    // 如果批大小为0，则设置全连接层的批大小为0
    return pytorch_qnnp_status_success;
    // 返回成功状态
  }

  fully_connected->batch_size = 1;
  // 设置全连接层的批大小为1
  fully_connected->input_height = batch_size;
  // 设置全连接层的输入高度为批大小
  fully_connected->input_width = 1;
  // 设置全连接层的输入宽度为1
  fully_connected->input = input;
  // 设置全连接层的输入指针
  fully_connected->input_pixel_stride = input_stride;
  // 设置全连接层的输入像素跨度

  fully_connected->output_height = batch_size;
  // 设置全连接层的输出高度为批大小
  fully_connected->output_width = 1;
  // 设置全连接层的输出宽度为1
  fully_connected->output = output;
  // 设置全连接层的输出指针
  fully_connected->output_pixel_stride = output_stride;
  // 设置全连接层的输出像素跨度

  return pytorch_qnnp_status_success;
  // 返回成功状态
}
```