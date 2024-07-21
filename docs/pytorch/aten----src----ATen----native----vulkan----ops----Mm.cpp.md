# `.\pytorch\aten\src\ATen\native\vulkan\ops\Mm.cpp`

```
// 引入 Vulkan 相关头文件
#include <ATen/native/vulkan/ops/Mm.h>
#include <ATen/native/vulkan/ops/Utils.h>

// 引入 ATen 相关头文件
#include <ATen/Context.h>
#include <ATen/Functions.h>
#include <ATen/native/vulkan/api/Tensor.h>
#include <ATen/native/vulkan/api/Types.h>
#include <ATen/native/vulkan/impl/Packing.h>
#include <c10/util/irange.h>

// 定义命名空间
namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

// 使用 Vulkan 相关命名空间和操作命名空间
using namespace api::utils;
using namespace at::native::vulkan::ops;

// 函数：使用宽度打包对输入进行打包
vTensor pack_inputs_using_width_packing(const Tensor& input_arg) {
  // 内部断言：输入不能是量化的张量
  TORCH_INTERNAL_ASSERT(
      !input_arg.is_quantized(),
      "Vulkan Linear not usable! "
      "Reason: Input packing only supports non-quantized tensors.");
  // 内部断言：输入张量的维度必须是2或3
  TORCH_INTERNAL_ASSERT(
      input_arg.dim() == 2 || input_arg.dim() == 3,
      "Vulkan Linear not usable! "
      "Reason: Input packing only supports 2D or 3D tensors.");

  // 如果输入张量在 CPU 上，则将其转换到 Vulkan 设备上
  Tensor input = input_arg;
  if (input.is_cpu()) {
    input = input.vulkan();
  }

  // 检查输入张量必须在 Vulkan 设备上
  TORCH_CHECK(input.is_vulkan(), "Input must be on Vulkan device!");

  // 将 Vulkan 张量进行类型转换
  vTensor v_input = convert(input);
  // 如果输入张量的 GPU 内存布局是 TENSOR_CHANNELS_PACKED，则转换为 TENSOR_WIDTH_PACKED
  if (v_input.gpu_memory_layout() ==
      api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED) {
    v_input = packing::convert_image_channels_packed_to_width_packed(v_input);
  }

  // 检查转换后的 GPU 内存布局必须是 TENSOR_WIDTH_PACKED
  TORCH_CHECK(
      v_input.gpu_memory_layout() == api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
      "After packing, the v_input must be in TENSOR_WIDTH_PACKED format");

  // 返回转换后的 Vulkan 张量
  return v_input;
}

// 函数：使用高度打包对权重进行打包
vTensor pack_weights_using_height_packing(const Tensor& weight_arg) {
  // 内部断言：权重不能是量化的张量
  TORCH_INTERNAL_ASSERT(
      !weight_arg.is_quantized(),
      "Vulkan Linear not usable! "
      "Reason: Weight packing only supports non-quantized tensors.");
  // 内部断言：权重张量的维度必须是2或3
  TORCH_INTERNAL_ASSERT(
      weight_arg.dim() == 2 || weight_arg.dim() == 3,
      "Vulkan Linear not usable! "
      "Reason: Weight packing only supports 2D or 3D tensors.");

  // 将输入的权重张量赋值给局部变量 weight
  Tensor weight = weight_arg;

  // 如果权重张量在 CPU 上，则将其转换到 Vulkan 设备上
  if (weight.is_cpu()) {
    weight = weight.vulkan();
  }

  // 检查权重张量必须在 Vulkan 设备上
  TORCH_CHECK(weight.is_vulkan(), "Weight must be on Vulkan device!");

  // 将 Vulkan 张量进行类型转换
  vTensor v_weight = convert(weight);
  // 如果权重张量的 GPU 内存布局是 TENSOR_CHANNELS_PACKED，则转换为 TENSOR_HEIGHT_PACKED
  if (v_weight.gpu_memory_layout() ==
      api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED) {
    v_weight =
        packing::convert_image_channels_packed_to_height_packed(v_weight);
  }

  // 检查转换后的 GPU 内存布局必须是 TENSOR_HEIGHT_PACKED
  TORCH_CHECK(
      v_weight.gpu_memory_layout() ==
          api::GPUMemoryLayout::TENSOR_HEIGHT_PACKED,
      "After packing, the v_weight must be in TENSOR_HEIGHT_PACKED format");

  // 返回转换后的 Vulkan 张量
  return v_weight;
}

// 函数：根据参数进行权重打包，支持是否使用批处理
vTensor pack_weights(const Tensor& weight_arg, const bool use_batch = false) {
  // 如果权重张量没有量化，则使用高度打包进行打包
  if (!weight_arg.is_quantized()) {
    return pack_weights_using_height_packing(weight_arg);
  }

  // 内部断言：仅支持量化的权重逻辑
  TORCH_CHECK(
      weight_arg.is_quantized(), "Only quantized weights logic after here");

  // 以下逻辑是量化的或者批处理的逻辑

  // 获取当前的 Vulkan API 上下文
  api::Context* const context = api::context();

  // 获取连续的权重张量
  const Tensor weight = weight_arg.contiguous();
  // 获取权重张量的尺寸信息
  const IntArrayRef w_sizes = weight.sizes();

  // 如果使用批处理
  if (use_batch) {
    // 检查权重尺寸是否为3，否则抛出错误信息，指出 Vulkan 线性不可用的原因
    TORCH_CHECK(
        w_sizes.size() == 3,
        "Vulkan Linear not usable! "
        "Reason: Unable to perform weight packing with batch; the input tensor of a batch of matrices should contain 3 dimensions: batch, height, width.");
  }
  /* Source */
  // 源张量的批量大小、宽度和高度尺寸
  int64_t src_kb_sz = 0;
  int64_t src_kw_sz = 0;
  int64_t src_kh_sz = 0;
  /* Destination */
  // 目标张量的批量大小、宽度和高度尺寸
  int64_t dst_kb_sz = 0;
  int64_t dst_kw_sz = 0;
  int64_t dst_kh_sz = 0;
  // 目标张量的虚拟尺寸向量
  std::vector<int64_t> dst_vtensor_sizes;
  /* Source */
  // 如果使用批量处理，则源张量的批量大小为 w_sizes 中指定的索引位置；否则为默认值1
  src_kb_sz = use_batch ? w_sizes[Layout::BatchMatrices::batch] : 1;
  src_kw_sz = use_batch ? w_sizes[Layout::BatchMatrices::width]
                        : w_sizes[Layout::Parameter::width];
  src_kh_sz = use_batch ? w_sizes[Layout::BatchMatrices::height]
                        : w_sizes[Layout::Parameter::height];

  /* Destination */
  // 目标张量的批量大小与源张量相同
  dst_kb_sz = src_kb_sz;
  // 目标张量的宽度尺寸为源张量宽度尺寸向上取整除以2
  dst_kw_sz = div_up(src_kw_sz, INT64_C(2));
  // 目标张量的高度尺寸为源张量高度尺寸向上取整除以2
  dst_kh_sz = div_up(src_kh_sz, INT64_C(2));
  // 设置目标张量的虚拟尺寸向量
  dst_vtensor_sizes = {
      dst_kb_sz,
      4,
      dst_kh_sz,
      dst_kw_sz,
  };

  // 创建一个 Vulkan 张量对象 v_weight，使用指定的上下文、尺寸和数据类型
  vTensor v_weight{
      context, dst_vtensor_sizes, convert_dtype(weight_arg.scalar_type())};

  // 设置 v_weight 为量化张量
  v_weight.set_is_quantized();
  // 设置 v_weight 的量化比例
  v_weight.set_scale(weight_arg.q_scale());
  // 设置 v_weight 的量化零点
  v_weight.set_zero_point(weight_arg.q_zero_point());

  // 调用函数 stage_pack_weights 对权重进行打包，并返回打包后的 v_weight 张量
  stage_pack_weights<int8_t>(
      context,
      v_weight,
      weight,
      src_kb_sz,
      src_kh_sz,
      src_kw_sz,
      dst_kh_sz,
      dst_kw_sz);
  // 返回打包后的 v_weight 张量对象
  return v_weight;
}

// pack_biases 函数的定义，用于将偏置参数打包成 vTensor 对象
vTensor pack_biases(
    const Tensor& weight_arg,  // 输入参数：权重张量
    const std::optional<Tensor>& bias_arg,  // 输入参数：可选的偏置张量
    const bool use_batch = false) {  // 输入参数：是否使用批处理，默认为 false

  // 如果提供了偏置参数
  if (bias_arg) {
    Tensor bias = *bias_arg;  // 解引用获取偏置张量
    // 如果偏置张量在 CPU 上，则将其转换到 Vulkan 上
    if (bias.is_cpu()) {
      bias = bias.vulkan();
    }
    return convert(bias);  // 将偏置张量转换为 vTensor 并返回
  } else {
    // 如果没有提供偏置参数，则创建一个全零张量作为偏置
    return convert(at::zeros({}, at::device(at::kVulkan).dtype(at::kFloat)));
  }
}

// 旧版本的 pack_biases 函数，用于处理量化问题，在未来将被移除
vTensor pack_biases_quantized_weights(
    const Tensor& weight_arg,  // 输入参数：权重张量
    const std::optional<Tensor>& bias_arg,  // 输入参数：可选的偏置张量
    const bool use_batch = false) {  // 输入参数：是否使用批处理，默认为 false

  // 检查权重张量是否已经量化，否则抛出错误信息
  TORCH_CHECK(
      weight_arg.is_quantized(),
      "pack_biases_quantized to be used only when using quantized linear ops");

  // 如果提供了偏置参数，并且该偏置张量在 Vulkan 上
  if (bias_arg && bias_arg->is_vulkan()) {
    return convert(*bias_arg);  // 将偏置张量转换为 vTensor 并返回
  }

  api::Context* const context = api::context();  // 获取当前 API 上下文对象

  // 如果提供了偏置参数
  if (bias_arg) {
    const Tensor bias = bias_arg->contiguous();  // 获取连续的偏置张量
    const IntArrayRef b_sizes = bias.sizes();  // 获取偏置张量的尺寸信息
    const float* const src_bias_ptr = bias.const_data_ptr<float>();  // 获取偏置张量的数据指针

    /* Source */
    int64_t src_kb_sz = 0;
    int64_t src_kw_sz = 0;
    int64_t src_kh_sz = 0;

    // 根据是否使用批处理及偏置张量的维度情况，确定源张量的尺寸
    if (use_batch) {
      if (bias.sizes().size() == 3) {
        src_kb_sz = b_sizes[Layout::BatchMatrices::batch];
        src_kw_sz = b_sizes[Layout::BatchMatrices::width];
        src_kh_sz = b_sizes[Layout::BatchMatrices::height];
      } else if (bias.sizes().size() == 2) {
        // 跳过批处理维度进行广播；索引 -1
        src_kb_sz = 1;
        src_kw_sz = b_sizes[Layout::BatchMatrices::height];
        src_kh_sz = b_sizes[Layout::BatchMatrices::batch];
      } else {
        // 跳过批处理和高度维度进行广播；索引 -2
        src_kb_sz = 1;
        src_kw_sz = b_sizes[Layout::BatchMatrices::batch];
        src_kh_sz = 1;
      }
    } else {
      src_kb_sz = 1;
      if (bias.sizes().size() == 2) {
        src_kw_sz = b_sizes[Layout::Parameter::width];
        src_kh_sz = b_sizes[Layout::Parameter::height];
      } else {
        src_kw_sz = b_sizes[Layout::Parameter::height];
        src_kh_sz = 1;
      }
    }
    const int64_t src_matrix_sz = src_kw_sz * src_kh_sz;

    /* Destination */
    // 计算目标张量的尺寸
    const int64_t dst_kw_sz = div_up(src_kw_sz, INT64_C(2));
    const int64_t dst_kh_sz = div_up(src_kh_sz, INT64_C(2));
    const int64_t dst_plane_sz = dst_kw_sz * dst_kh_sz;
    const int64_t dst_matrix_sz = dst_plane_sz * 4;

    // 创建一个 vTensor 对象作为 Vulkan 张量
    vTensor v_bias{
        context,
        {
            src_kb_sz,
            4,
            dst_kh_sz,
            dst_kw_sz,
        },
        convert_dtype(bias_arg->scalar_type()),  // 使用偏置张量的数据类型进行转换
    };

    // 创建一个用于临时存储的存储缓冲区
    api::StorageBuffer staging(
        context, api::ScalarType::Float, v_bias.gpu_numel());
    {
      // 创建一个内存映射对象，用于对可写的存储进行映射
      api::MemoryMap mapping(staging.buffer(), api::MemoryAccessType::WRITE);

      // 获取指向浮点数类型数据的指针
      float* dst_bias_ptr = mapping.template data<float>();

      // 将目标内存区域初始化为零，大小为目标张量的字节数
      memset(dst_bias_ptr, 0, v_bias.nbytes());

      // 遍历输入张量的维度
      for (const auto src_b : c10::irange(src_kb_sz)) {
        for (const auto src_h : c10::irange(src_kh_sz == 1 ? 2 : src_kh_sz)) {
          for (const auto src_w :
               c10::irange((use_batch && src_kw_sz == 1) ? 2 : src_kw_sz)) {
            // 计算目标平面和索引
            int64_t dst_plane = 2 * (src_h % 2) + (src_w % 2);
            int64_t dst_index = (src_h / 2) * dst_kw_sz + (src_w / 2);
            // 复制源数据到目标位置，每次复制一个浮点数大小的数据
            memcpy(
                dst_bias_ptr + src_b * dst_matrix_sz +
                    dst_plane * dst_plane_sz + dst_index,
                src_bias_ptr + src_b * src_matrix_sz +
                    (src_kh_sz == 1 ? 0 : src_h * src_kw_sz) +
                    ((use_batch && src_kw_sz == 1) ? 0 : src_w),
                sizeof(float));
          }
        }
      }
    }
    // 将缓冲区内容打包到张量中
    utils::pack_staging_to_vtensor(staging.buffer(), v_bias);

    // 返回填充后的目标张量
    return v_bias;
  } else {
    // 创建一个形状为{1}的目标张量，使用指定的数据类型
    vTensor v_bias{
        api::context(),
        {1},
        convert_dtype(weight_arg.scalar_type()),
    };

    // 创建一个存储缓冲区，使用指定的上下文和数据类型，并计算缓冲区元素的数量
    api::StorageBuffer staging(
        context, api::ScalarType::Float, v_bias.gpu_numel());

    {
      // 创建一个内存映射对象，用于对可写的存储进行映射
      api::MemoryMap mapping(staging.buffer(), api::MemoryAccessType::WRITE);

      // 获取指向浮点数类型数据的指针
      float* data_ptr = mapping.template data<float>();

      // 将数据指针指向的内存区域初始化为零，大小为目标张量的字节数
      memset(
          data_ptr,
          // 对于0，2的补码整数和IEEE-754浮点数具有相同的位表示，因此可以使用memset，它只接受uint8_t参数。
          0,
          v_bias.nbytes());
    }
    // 将缓冲区内容打包到张量中
    utils::pack_staging_to_vtensor(staging.buffer(), v_bias);

    // 返回填充后的目标张量
    return v_bias;
  }
}

// 检查带有批处理的权重和可选偏置是否可用
bool available_check_with_batch(
    const Tensor& weight,  // 输入的权重张量
    const std::optional<Tensor>& bias) {  // 可选的偏置张量

  // 检查权重是否可用的条件
  const bool weight_available = (3 == weight.ndimension()) &&  // 权重张量必须是三维的
      (weight.size(Layout::BatchMatrices::batch) > 0) &&  // 批次维度大小必须大于0
      (weight.size(Layout::BatchMatrices::height) > 0) &&  // 高度维度大小必须大于0
      (weight.size(Layout::BatchMatrices::width) > 0) &&  // 宽度维度大小必须大于0
      ((weight.device().is_cpu()) ||  // 设备类型可以是 CPU 或 Vulkan
       (c10::DeviceType::Vulkan == weight.device().type())) &&
      (kFloat == weight.scalar_type()) && !weight.requires_grad();  // 标量类型必须是浮点数且不需要梯度

  if (!weight_available) {
    return false;  // 如果权重不可用，则返回 false
  }

  if (!bias || !bias->defined()) {
    // 如果没有偏置或者偏置未定义，则不需要检查偏置，直接返回 true
    return true;
  }

  bool bias_available = true;

  // 检查偏置是否可用的条件
  bias_available &= (bias->ndimension() > 0);  // 偏置张量的维度数必须大于0
  bias_available &=
      ((bias->device().is_cpu()) ||  // 偏置张量的设备类型可以是 CPU 或 Vulkan
       (c10::DeviceType::Vulkan == bias->device().type()));
  bias_available &= (kFloat == bias->scalar_type());  // 偏置张量的标量类型必须是浮点数

  // 根据偏置张量的维度数进行不同的检查
  if (bias->ndimension() == 3) {
    // 对于三维的偏置张量，检查批次和宽度维度的一致性
    bias_available &=
        (bias->size(Layout::BatchMatrices::width) ==
             weight.size(Layout::BatchMatrices::width) ||
         bias->size(Layout::BatchMatrices::width) == 1);
    bias_available &=
        (bias->size(Layout::BatchMatrices::batch) ==
             weight.size(Layout::BatchMatrices::batch) ||
         bias->size(Layout::BatchMatrices::batch) == 1);
  } else if (bias->ndimension() == 2) {
    // 对于二维的偏置张量，跳过批次维度进行广播，索引为 -1
    bias_available &=
        (bias->size(Layout::BatchMatrices::height) ==
             weight.size(Layout::BatchMatrices::width) ||
         bias->size(Layout::BatchMatrices::height) == 1);
  } else {
    // 对于其他维度的偏置张量，跳过批次和高度维度进行广播，索引为 -2
    bias_available &=
        (bias->size(Layout::BatchMatrices::batch) ==
             weight.size(Layout::BatchMatrices::width) ||
         bias->size(Layout::BatchMatrices::batch) == 1);
  }

  bias_available &= !bias->requires_grad();  // 偏置张量不需要梯度

  return bias_available;  // 返回偏置是否可用的结果
}

// 检查单个权重和可选偏置是否可用
bool available(
    const Tensor& weight,  // 输入的权重张量
    const std::optional<Tensor>& bias,  // 可选的偏置张量
    const bool use_batch = false) {  // 是否使用批处理，默认为 false

  if (!api::available()) {
    return false;  // 如果 API 不可用，则返回 false
  }

  if (use_batch) {
    return available_check_with_batch(weight, bias);  // 如果使用批处理，则调用带批处理的检查函数
  }

  // 检查权重是否可用的条件（不使用批处理）
  const bool weight_available = (2 == weight.ndimension()) &&  // 权重张量必须是二维的
      (weight.size(Layout::Parameter::height) > 0) &&  // 高度维度大小必须大于0
      (weight.size(Layout::Parameter::width) > 0) &&  // 宽度维度大小必须大于0
      ((weight.device().is_cpu()) ||  // 设备类型可以是 CPU 或 Vulkan
       (c10::DeviceType::Vulkan == weight.device().type())) &&
      (kFloat == weight.scalar_type() || kQInt8 == weight.scalar_type()) &&  // 标量类型可以是浮点数或QInt8
      !weight.requires_grad();  // 不需要梯度

  if (!weight_available) {
    // 如果权重不可用，则返回 false
    return false;
  }
    // 返回 false，表示条件不满足
    return false;
  }

  // 检查是否存在偏置项，并且满足一定的条件
  const bool bias_available =
      // 如果偏置项存在并且已定义
      ((bias && bias.has_value() && bias->defined())
           ? (
              // 检查偏置项的维度大于0，且设备为CPU或Vulkan类型，数据类型为float
              (bias->ndimension() > 0) &&
              ((bias->device().is_cpu()) ||
               (c10::DeviceType::Vulkan == bias->device().type())) &&
              (kFloat == bias->scalar_type()) &&
              // 如果偏置项的维度大于1，则要求其宽度与权重的宽度相等
              ((bias->ndimension() > 1)
                   ? (bias->size(Layout::Parameter::width) ==
                      weight.size(Layout::Parameter::width))
                   : true) &&
              // 偏置项不需要梯度计算
              !bias->requires_grad())
           // 如果偏置项不存在或者未定义，则默认偏置可用
           : true);
  
  // 返回偏置是否可用的结果
  return bias_available;
}

// 检查是否可以使用批处理模式进行计算
bool usable_check_with_batch(
    const Tensor& input,
    const IntArrayRef unpacked_weight_sizes) {
  // 检查输入张量是否是三维的，设备类型是否为Vulkan，标量类型是否为float，
  // 并且满足特定尺寸要求，不需要梯度计算，返回真值
  return (3 == input.ndimension()) &&
      (c10::DeviceType::Vulkan == input.device().type()) &&
      (kFloat == input.scalar_type()) &&
      (input.size(Layout::BatchMatrices::width) ==
       unpacked_weight_sizes[Layout::BatchMatrices::height]) &&
      (input.size(Layout::BatchMatrices::batch) ==
       unpacked_weight_sizes[Layout::BatchMatrices::batch]) &&
      !input.requires_grad() && true;
}

// 检查是否可以使用线性操作
bool usable(
    const Tensor& input,
    const IntArrayRef unpacked_weight_sizes,
    const bool use_batch = false) {
  if (use_batch) {
    // 如果指定使用批处理模式，则调用 usable_check_with_batch 函数进行检查
    return usable_check_with_batch(input, unpacked_weight_sizes);
  }
  // 将输入张量转换为 v_input
  const auto v_input = convert(input);
  // 检查输入张量是否是二维的，设备类型是否为Vulkan，标量类型为float或者是量化类型，
  // 并且满足特定尺寸要求，不需要梯度计算，返回真值
  return (2 == input.ndimension()) &&
      (c10::DeviceType::Vulkan == input.device().type()) &&
      ((kFloat == input.scalar_type()) ||
       (v_input.is_quantized() &&
        (kQUInt8 == input.scalar_type() || kQInt8 == input.scalar_type()))) &&
      (input.size(Layout::Parameter::width) ==
       unpacked_weight_sizes[Layout::Parameter::height]) &&
      !input.requires_grad() && true;
}

// 将输入张量重新形状为二维张量
static Tensor reshape_to_2d(const Tensor& input_arg) {
  // 检查输入张量的维度是否大于等于1
  TORCH_CHECK(
      input_arg.dim() >= 1,
      "Vulkan Linear op only supports input tensor with dim >= 1");

  if (input_arg.dim() == 1) {
    // 如果输入张量是一维的，则在第0维度上增加一个维度，并返回
    return input_arg.unsqueeze(0);
  }
  // 获取输入张量的尺寸信息
  const IntArrayRef input_sizes = input_arg.sizes();
  // 计算除最后一个维度外所有维度的乘积，作为新的第一维度
  const auto d =
      c10::multiply_integers(input_sizes.cbegin(), input_sizes.end() - 1);
  // 将输入张量重新形状为 {d, 最后一个维度大小} 的二维张量，并返回
  return input_arg.reshape({d, input_arg.size(-1)});
}

Tensor run_quantized_addmm_context(
    const Tensor& input_arg,
    const float alpha,
    const float beta,
    const c10::intrusive_ptr<LinearPackedContext>& linear_context,
    double output_scale,
  // 获取当前的运行环境上下文
  api::Context* const context = api::context();

  // 如果输入张量是二维的则使用它，否则对输入进行二维重塑
  const Tensor input_arg_2d =
      input_arg.dim() == 2 ? input_arg : reshape_to_2d(input_arg);

  // 如果二维化后的输入张量支持 Vulkan 加速，则使用它；否则尝试使用 Vulkan 加速
  const Tensor input =
      input_arg_2d.is_vulkan() ? input_arg_2d : input_arg_2d.vulkan();

  // 将输入张量转换为 Vulkan 张量类型
  const vTensor& v_input = convert(input);

  // 将线性上下文中打包的权重转换为 Vulkan 张量
  const vTensor& packed_v_weight = convert(
      linear_context->get_val(LinearPackedContext::Packed::Weight).toTensor());

  // 将线性上下文中打包的偏置转换为 Vulkan 张量
  const vTensor& packed_v_bias = convert(
      linear_context->get_val(LinearPackedContext::Packed::Bias).toTensor());

  // 获取未打包的权重的尺寸信息
  const std::vector<int64_t> unpacked_weight_sizes =
      linear_context->get_val(LinearPackedContext::Packed::WeightSizes)
          .toIntVector();

  // 检查是否定义了偏置
  const bool bias_defined =
      linear_context->get_val(LinearPackedContext::Packed::BiasDefined)
          .toBool();

  // 检查 Vulkan 线性运算是否可用，包括输入张量的有效性和与提供的权重和偏置张量的组合是否被 Vulkan 实现支持
  TORCH_CHECK(
      usable(input, unpacked_weight_sizes),
      "Vulkan Linear not usable! "
      "Reason: The provided input tensor is either invalid on its own, or its "
      "combination with the provided weight and bias tensors are unsupported by "
      "Vulkan impl.");

  // 检查是否输入张量和打包的权重张量都是量化的，以确保量化版本的运行
  TORCH_CHECK(
      (packed_v_weight.is_quantized() && v_input.is_quantized()),
      "run_quantized_addmm_context called for quantized version with unquantized input");

  // 创建输出 Vulkan 张量，设置其尺寸和数据类型
  vTensor v_output{
      context,
      {
          input_arg_2d.sizes()[Layout::Parameter::height],
          unpacked_weight_sizes[Layout::Parameter::width],
      },
      v_input.dtype(),
  };

  // 标记输出张量为量化
  v_output.set_is_quantized();

  // 设置输出张量的比例因子和零点偏移量
  v_output.set_scale(output_scale);
  v_output.set_zero_point(output_zero_point);

  // 如果定义了偏置，则进一步处理偏置相关的参数
  if (bias_defined) {
    // 初始化 API 的均匀参数缓冲区
    api::UniformParamsBuffer params;
    // 初始化计算着色器信息，根据输入张量的数据类型选择相应的量化加法乘法运算内核
    api::ShaderInfo compute_shader;
    compute_shader = (kQInt8 == input_arg.scalar_type())
        ? VK_KERNEL(quantized_addmm_qint8)
        : VK_KERNEL(quantized_addmm_quint8);
    // 定义用于 GPU 块中的结构体，存储运算所需的参数
    const struct {
      uvec3 size;
      int32_t K;
      uvec3 um1_size;
      int32_t K1;
      uvec3 um2_size;
      int32_t K2;
      uvec3 ut_size;
      int32_t K3;
      vec2 multiplier;
      vec2 input_scales;
      float out_scale;
      float _1;
      ivec2 input_zero_points;
      int32_t out_zero_point;
      int32_t _2;
    } block{
        v_output.extents(),
        safe_downcast<int32_t>(
            div_up(v_input.sizes()[Layout::Parameter::width], INT64_C(2))),
        v_input.extents(),
        0u,
        packed_v_weight.extents(),
        0u,
        packed_v_bias.extents(),
        0u,
        {
            alpha,
            beta,
        },
        {
            safe_downcast<float>(v_input.get_scale()),
            safe_downcast<float>(packed_v_weight.get_scale()),
        },
        safe_downcast<float>(output_scale),
        0.0f,
        {
            safe_downcast<int32_t>(v_input.get_zero_point()),
            safe_downcast<int32_t>(packed_v_weight.get_zero_point()),
        },
        safe_downcast<int32_t>(output_zero_point),
        0u,
    };
    // 将块数据复制到 API 的均匀参数缓冲区中
    params = api::UniformParamsBuffer(context, block);
    // 创建一个空的管线屏障对象，用于后续的提交计算作业时使用
    api::PipelineBarrier pipeline_barrier{};
    
    // 提交计算作业到上下文对象，包括以下参数：
    context->submit_compute_job(
        // 计算着色器描述符
        compute_shader,
        
        // 管线屏障对象，用于同步计算作业的执行流程
        pipeline_barrier,
        
        // 全局工作组大小
        {
            // 计算并向下转型 v_output 的宽度除以 2
            safe_downcast<uint32_t>(div_up(v_output.sizes()[Layout::Parameter::width], INT64_C(2))),
            // 计算并向下转型 v_output 的高度除以 2
            safe_downcast<uint32_t>(div_up(v_output.sizes()[Layout::Parameter::height], INT64_C(2))),
            1,
        },
        
        // 局部工作组大小
        {8, 8, 1},
        
        // 围栏句柄，此处设置为 VK_NULL_HANDLE 表示无围栏
        VK_NULL_HANDLE,
        
        // 计算着色器的参数，包括图像输出
        v_output.image(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        
        // 计算着色器的参数，包括图像输入
        v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        
        // 计算着色器的参数，包括打包的权重图像
        packed_v_weight.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        
        // 计算着色器的参数，包括打包的偏置图像
        packed_v_bias.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        
        // 参数缓冲区，用于传递参数到计算着色器
        params.buffer());

  } else { // 如果没有偏置
    // 创建统一参数缓冲区对象
    api::UniformParamsBuffer params;
    
    // 创建计算着色器信息对象
    api::ShaderInfo compute_shader;
    
    // 定义不带偏置的数据块
    const struct {
      uvec3 size;
      int32_t K;
      uvec3 um1_size;
      int32_t K1;
      uvec3 um2_size;
      int32_t K2;
      vec2 input_scales;
      float out_scale;
      float _1;
      ivec2 input_zero_points;
      int32_t out_zero_point;
      int32_t _2;
    } block_no_bias{
        v_output.extents(),
        // 计算并向下转型 v_input 的宽度除以 2
        safe_downcast<int32_t>(div_up(v_input.sizes()[Layout::Parameter::width], INT64_C(2))),
        v_input.extents(),
        0u,
        packed_v_weight.extents(),
        0u,
        {
            // 向下转型并安全获取 v_input 的比例因子
            safe_downcast<float>(v_input.get_scale()),
            // 向下转型并安全获取 packed_v_weight 的比例因子
            safe_downcast<float>(packed_v_weight.get_scale()),
        },
        // 向下转型并安全获取输出比例因子
        safe_downcast<float>(output_scale),
        0.0f,
        {
            // 向下转型并安全获取 v_input 的零点偏移量
            safe_downcast<int32_t>(v_input.get_zero_point()),
            // 向下转型并安全获取 packed_v_weight 的零点偏移量
            safe_downcast<int32_t>(packed_v_weight.get_zero_point()),
        },
        // 向下转型并安全获取输出的零点偏移量
        safe_downcast<int32_t>(output_zero_point),
        0u,
    };
    
    // 使用块数据初始化统一参数缓冲区
    params = api::UniformParamsBuffer(context, block_no_bias);
    
    // 根据输入参数的标量类型选择不同的计算着色器
    compute_shader = (kQInt8 == input_arg.scalar_type())
        ? VK_KERNEL(quantized_mm_qint8)
        : VK_KERNEL(quantized_mm_quint8);
    
    // 创建一个空的管线屏障对象，用于后续的提交计算作业时使用
    api::PipelineBarrier pipeline_barrier{};
    context->submit_compute_job(
        // 提交计算作业到上下文对象中
        // 使用计算着色器描述符
        compute_shader,
        // 应用管线障碍
        pipeline_barrier,
        // 全局工作组大小
        {
            safe_downcast<uint32_t>(
                div_up(v_output.sizes()[Layout::Parameter::width], INT64_C(2))),
            safe_downcast<uint32_t>(div_up(
                v_output.sizes()[Layout::Parameter::height], INT64_C(2))),
            1,
        },
        // 本地工作组大小
        {8, 8, 1},
        // 等待句柄
        VK_NULL_HANDLE,
        // 计算着色器参数
        v_output.image(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        packed_v_weight.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        // 参数缓冲区
        params.buffer());
  }
  // 转换输出张量为普通张量
  Tensor output = convert(v_output);
  // 如果输入参数的维度为2，则直接返回输出
  if (input_arg.dim() == 2) {
    return output;
  } else {
    // 构造输出形状的向量
    std::vector<int64_t> shape;
    for (const auto i : c10::irange(input_arg.dim() - 1)) {
      shape.emplace_back(input_arg.size(i));
    }
    shape.emplace_back(output.size(-1));
    // 根据构造的形状调整输出张量的形状并返回
    return output.reshape(shape);
  }
// 定义一个函数，用于执行加权矩阵乘法和加法操作，支持不同的输入参数和量化选项
Tensor run_addmm_context(
    const Tensor& input_arg,  // 输入张量
    const float alpha,        // 矩阵运算中的标量参数
    const float beta,         // 矩阵运算中的标量参数
    const c10::intrusive_ptr<LinearPackedContext>& linear_context,  // 包含线性层上下文信息的指针
    bool quantized,           // 是否使用量化计算
    double output_scale,      // 输出张量的缩放因子
    int64_t output_zero_point // 输出张量的零点
) {
  if (quantized) {
    // 如果需要量化计算，则调用量化版本的加权矩阵乘法和加法操作
    return run_quantized_addmm_context(
        input_arg,
        alpha,
        beta,
        linear_context,
        output_scale,
        output_zero_point);
  }

  // 获取当前的运行上下文
  api::Context* const context = api::context();

  // 将输入张量展平为二维张量（如果不是二维的话）
  const Tensor input_arg_2d =
      input_arg.dim() == 2 ? input_arg : reshape_to_2d(input_arg);
  
  // 将展平后的二维张量转换为 Vulkan 张量
  const Tensor input =
      input_arg_2d.is_vulkan() ? input_arg_2d : input_arg_2d.vulkan();
  
  // 使用宽度打包方式对输入进行打包，返回 Vulkan 张量
  const vTensor& v_input = pack_inputs_using_width_packing(input);

  // 从线性上下文中获取打包后的权重张量和偏置张量
  const vTensor& packed_v_weight = convert(
      linear_context->get_val(LinearPackedContext::Packed::Weight).toTensor());
  const vTensor& packed_v_bias = convert(
      linear_context->get_val(LinearPackedContext::Packed::Bias).toTensor());
  
  // 获取未打包的权重张量的大小信息
  const std::vector<int64_t> unpacked_weight_sizes =
      linear_context->get_val(LinearPackedContext::Packed::WeightSizes)
          .toIntVector();

  // 检查 Vulkan 线性运算是否可用，即输入张量与权重、偏置张量的组合是否支持 Vulkan 实现
  TORCH_CHECK(
      usable(input, unpacked_weight_sizes),
      "Vulkan Linear not usable! "
      "Reason: The provided input tensor is either invalid on its own, or its "
      "combination with the provided weight and bias tensors are unsupported by "
      "Vulkan impl.");

  // 检查 Vulkan 张量的内存布局是否为 TENSOR_WIDTH_PACKED
  TORCH_CHECK(
      v_input.gpu_memory_layout() == api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
      "run_addmm_context must have width packed input");

  // 检查打包后的权重张量的内存布局是否为 TENSOR_HEIGHT_PACKED
  TORCH_CHECK(
      packed_v_weight.gpu_memory_layout() ==
          api::GPUMemoryLayout::TENSOR_HEIGHT_PACKED,
      "run_addmm_context must have height packed weight");

  // 创建 Vulkan 输出张量，设置其上下文、形状和数据类型
  vTensor v_output{
      context,
      {
          input_arg_2d.sizes()[Layout::Parameter::height],
          unpacked_weight_sizes[Layout::Parameter::width],
      },
      v_input.dtype(),
  };

  // Vulkan 接口参数和计算着色器信息的定义
  api::UniformParamsBuffer params;
  api::ShaderInfo compute_shader;
  
  // 计算步长，即二维输入的宽度维度除以 4 后的结果
  int step_size = div_up(v_input.sizes()[Layout::Parameter::width], INT64_C(4));

  // 定义一个结构体，用于存储着色器的范围信息
  const struct {
    uvec3 shader_extents;
    // 继续添加结构体的成员变量和字段
    // 定义并初始化一个无偏移的结构体，包含一个名为 mm_step_size 的 uint32_t 变量
    struct {
        uint32_t mm_step_size;
    } block_no_bias {
        v_output.extents(), // 使用 v_output 的维度信息来初始化结构体
        safe_downcast<uint32_t>(step_size), // 使用 step_size 的值来初始化 mm_step_size
    };

    // 使用 block_no_bias 结构体创建 UniformParamsBuffer 对象并赋值给 params
    params = api::UniformParamsBuffer(context, block_no_bias);

    // 使用 VK_KERNEL 宏创建 compute_shader，表示一个 Vulkan 的计算着色器
    compute_shader = VK_KERNEL(mm);

    // 定义一个空的 PipelineBarrier 对象
    api::PipelineBarrier pipeline_barrier {};

    // 提交计算作业到 context，执行 compute_shader 计算着色器
    context->submit_compute_job(
        // 计算着色器描述符
        compute_shader,
        // 管线障碍物
        pipeline_barrier,
        // 全局工作组大小，根据 v_output 的尺寸计算而来
        {
            safe_downcast<uint32_t>(
                div_up(v_output.sizes()[Layout::Parameter::width], INT64_C(4))),
            safe_downcast<uint32_t>(
                div_up(v_output.sizes()[Layout::Parameter::height], INT64_C(4))),
            1,
        },
        // 局部工作组大小为 {8, 8, 1}
        {8, 8, 1},
        // 使用 VK_NULL_HANDLE 表示没有指定 fence handle
        VK_NULL_HANDLE,
        // shader 参数
        v_output.image(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        packed_v_weight.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        // params buffer
        params.buffer()
    );

    // 将 v_output 转换为 Tensor 类型的 output
    Tensor output = convert(v_output);

    // 执行 addmm 操作，对 output 执行 alpha 倍并加上 bias
    output = output.mul(alpha).add(convert(packed_v_bias).mul(beta));

    // 如果 input_arg 的维度为 2，则直接返回 output
    if (input_arg.dim() == 2) {
        return output;
    } else {
        // 否则，根据 input_arg 的维度动态构造一个新的 shape 向量，并对 output 进行 reshape 操作后返回
        std::vector<int64_t> shape;
        for (const auto i : c10::irange(input_arg.dim() - 1)) {
            shape.emplace_back(input_arg.size(i));
        }
        shape.emplace_back(output.size(-1));
        return output.reshape(shape);
    }
}

Tensor run_baddbmm_context(
    const Tensor& input_arg,
    const float alpha,
    const float beta,
    const c10::intrusive_ptr<LinearPackedContext>& linear_context) {
  // 获取当前 API 的上下文对象
  api::Context* const context = api::context();

  // 检查输入张量的维度是否为 3
  TORCH_CHECK(
      input_arg.dim() == 3,
      "Vulkan Linear not usable! "
      "Reason: The input has the wrong dimension; the tensor of a batch of matrices should contain 3 dimensions: batch, height, width.");

  // 如果输入张量是 Vulkan 张量，则直接使用；否则将其转换为 Vulkan 张量
  const Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
  // 使用宽度打包方法对输入张量进行打包
  vTensor packed_v_input = pack_inputs_using_width_packing(input);

  // 获取线性上下文中打包的权重和偏置张量
  const vTensor& packed_v_weight = convert(
      linear_context->get_val(LinearPackedContext::Packed::Weight).toTensor());
  const vTensor& packed_v_bias = convert(
      linear_context->get_val(LinearPackedContext::Packed::Bias).toTensor());
  // 获取打包的权重张量的未打包大小
  const std::vector<int64_t> unpacked_weight_sizes =
      linear_context->get_val(LinearPackedContext::Packed::WeightSizes)
          .toIntVector();

  // 检查输入张量和权重大小是否兼容，确保 Vulkan 实现可以使用
  TORCH_CHECK(
      usable(input, unpacked_weight_sizes, true /*use batch*/),
      "Vulkan Linear not usable! "
      "Reason: The provided input tensor is either invalid on its own, or its "
      "combination with the provided weight and bias tensors are unsupported by "
      "Vulkan impl.");

  // 检查打包的输入张量是否使用了正确的 GPU 内存布局
  TORCH_CHECK(
      packed_v_input.gpu_memory_layout() ==
          api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
      "run_addmm_context called for non-quantized version with unpacked weight");

  // 检查打包的权重张量是否使用了正确的 GPU 内存布局
  TORCH_CHECK(
      packed_v_weight.gpu_memory_layout() ==
          api::GPUMemoryLayout::TENSOR_HEIGHT_PACKED,
      "run_addmm_context called for non-quantized version with unpacked weight");

  // 在着色器中，每个批次在单独的调用中计算，结果存储在 texel 的 .x 位置。
  // 默认情况下张量是通道打包的，着色器实际上会产生三个全零层的问题。
  // 我们通过创建一个 4 倍于批次大小的 vTensor 来解决这个问题。
  // 计算结束时，我们使用步长为 4 的“切片”来恢复原始形状。

  // 获取输入张量的批次大小
  int64_t input_batch = packed_v_input.sizes()[Layout::BatchMatrices::batch];

  // 计算步长，即输入张量的宽度维度除以 4
  int64_t input_width = packed_v_input.sizes()[Layout::BatchMatrices::width];
  int64_t mm_step_size = div_up(input_width, INT64_C(4));

  // 创建输出 vTensor 对象，尺寸为 (批次大小 * 4, 高度, 权重矩阵的 w 维度)
  vTensor v_output{
      context,
      {
          input_batch * 4,
          packed_v_input.sizes()[Layout::BatchMatrices::height],
          unpacked_weight_sizes.back(), // "w" dimension in weight matrix
      },
      packed_v_input.dtype(),
  };

  // 定义结构体 shader_extents
  const struct {
    uvec3 shader_extents;
    // 定义一个无偏置的结构体 block_no_bias，其中包含一个称为 mm_step_size 的 uint32_t 类型的成员变量
    uint32_t mm_step_size;
  } block_no_bias{
      // 使用 v_output 的维度创建 extents，同时将 mm_step_size 转换为 uint32_t 类型
      v_output.extents(),
      safe_downcast<uint32_t>(mm_step_size),
  };

  // 使用 block_no_bias 结构体创建 UniformParamsBuffer 对象 params，用于传递给 GPU 计算任务
  api::UniformParamsBuffer params(context, block_no_bias);

  // 创建一个空的 PipelineBarrier 对象 pipeline_barrier，用于控制计算任务的执行流程

  // 在 Vulkan 上下文中提交计算任务，执行矩阵乘法计算
  context->submit_compute_job(
      // 使用名称为 mm 的着色器描述符执行计算任务
      VK_KERNEL(mm),
      // 插入一个空的管线障碍以确保计算顺序的正确性
      pipeline_barrier,
      // 定义全局工作组大小，根据 v_output 的维度进行计算
      {
          safe_downcast<uint32_t>(div_up(
              v_output.sizes()[Layout::BatchMatrices::width], INT64_C(4))),
          safe_downcast<uint32_t>(div_up(
              v_output.sizes()[Layout::BatchMatrices::height], INT64_C(4))),
          safe_downcast<uint32_t>(
              v_output.sizes()[Layout::BatchMatrices::batch]),
      },
      // 定义局部工作组大小为 {8, 8, 1}
      {8, 8, 1},
      // 使用 VK_NULL_HANDLE 作为屏障的句柄，表示没有特定的屏障需求
      VK_NULL_HANDLE,
      // 设置着色器参数，包括输出图像 v_output 和输入图像 packed_v_input、packed_v_weight，
      // 以及参数缓冲区 params.buffer()
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      packed_v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      packed_v_weight.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // 将 params 缓冲区作为着色器参数传递
      params.buffer());

  // 在执行完乘法计算后，按照批次维度切片以获取通道打包的布局
  auto mm_output_unpacked = convert(v_output);
  // 定义步长为 4
  int step = 4;
  // 对 mm_output_unpacked 执行切片操作，以获取通道打包的输出数据
  auto mm_output = mm_output_unpacked.slice(
      Layout::BatchMatrices::batch, 0, input_batch * step, step);

  // 返回经过 alpha 缩放并加上偏置 packed_v_bias 乘以 beta 的 mm_output
  return mm_output.mul(alpha).add(convert(packed_v_bias).mul(beta));
}

// 定义了一个函数 addmm，实现矩阵相加乘运算，返回计算结果张量
Tensor addmm(
    const Tensor& bias,
    const Tensor& input,
    const Tensor& weight,
    const Scalar& beta,
    const Scalar& alpha) {
  // 调用 run_addmm_context 函数执行矩阵相加乘计算，返回结果张量
  return run_addmm_context(
      input,
      alpha.to<float>(),  // 将 alpha 标量转换为 float 类型
      beta.to<float>(),   // 将 beta 标量转换为 float 类型
      c10::make_intrusive<LinearPackedContext>(  // 创建 LinearPackedContext 对象的智能指针
          LinearPackedContext(weight, bias)),   // 使用 weight 和 bias 创建 LinearPackedContext
      false,  // 布尔参数，表示不使用 batch 处理
      0,      // 整数参数，暂时未用
      0);     // 整数参数，暂时未用
}

// 定义了一个函数 mm，实现矩阵相乘运算，返回计算结果张量
Tensor mm(const Tensor& mat1_arg, const Tensor& mat2_arg) {
  // 调用 run_addmm_context 函数执行矩阵相乘计算，返回结果张量
  return run_addmm_context(
      mat1_arg,
      1.0f,   // alpha 参数为 1.0
      1.0f,   // beta 参数为 1.0
      c10::make_intrusive<LinearPackedContext>(  // 创建 LinearPackedContext 对象的智能指针
          LinearPackedContext(mat2_arg, std::optional<Tensor>())),  // 使用 mat2_arg 创建 LinearPackedContext
      false,  // 布尔参数，表示不使用 batch 处理
      0,      // 整数参数，暂时未用
      0);     // 整数参数，暂时未用
}

// 定义了一个函数 bmm，实现 batch 矩阵相乘运算，返回计算结果张量
Tensor bmm(const Tensor& mat1_arg, const Tensor& mat2_arg) {
  // 调用 run_baddbmm_context 函数执行 batch 矩阵相乘计算，返回结果张量
  return run_baddbmm_context(
      mat1_arg,
      1.0f,   // alpha 参数为 1.0
      1.0f,   // beta 参数为 1.0
      c10::make_intrusive<LinearPackedContext>(  // 创建 LinearPackedContext 对象的智能指针
          LinearPackedContext(mat2_arg, std::optional<Tensor>(), true /*use batch*/)));  // 使用 mat2_arg 创建 LinearPackedContext，并指定使用 batch 处理
}

// 定义了一个函数 baddbmm，实现 batch 矩阵相加乘运算，返回计算结果张量
Tensor baddbmm(
    const Tensor& bias,
    const Tensor& input,
    const Tensor& weight,
    const Scalar& beta,
    const Scalar& alpha) {
  // 调用 run_baddbmm_context 函数执行 batch 矩阵相加乘计算，返回结果张量
  return run_baddbmm_context(
      input,
      alpha.to<float>(),  // 将 alpha 标量转换为 float 类型
      beta.to<float>(),   // 将 beta 标量转换为 float 类型
      c10::make_intrusive<LinearPackedContext>(  // 创建 LinearPackedContext 对象的智能指针
          LinearPackedContext(weight, bias, true /*use batch*/)));  // 使用 weight 和 bias 创建 LinearPackedContext，并指定使用 batch 处理
}

#ifdef USE_VULKAN_API

// 使用 Vulkan API 实现的 TORCH_LIBRARY_IMPL 宏，注册了 Vulkan 实现的矩阵运算函数
TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::addmm"), TORCH_FN(addmm));  // 注册 addmm 函数的 Vulkan 实现
  m.impl(TORCH_SELECTIVE_NAME("aten::mm"), TORCH_FN(mm));        // 注册 mm 函数的 Vulkan 实现
  m.impl(TORCH_SELECTIVE_NAME("aten::bmm"), TORCH_FN(bmm));      // 注册 bmm 函数的 Vulkan 实现
  m.impl(TORCH_SELECTIVE_NAME("aten::baddbmm"), TORCH_FN(baddbmm));  // 注册 baddbmm 函数的 Vulkan 实现
}

#endif /* USE_VULKAN_API */

} // namespace

// LinearPackedContext 类的构造函数，用于创建线性层参数打包的上下文
LinearPackedContext::LinearPackedContext(
    const Tensor& weight,
    const std::optional<Tensor>& bias,
    const bool use_batch)
    : unpacked_{c10::AnyType::get()} {
  // 检查 weight、bias 和 use_batch 参数是否有效，否则抛出异常
  TORCH_CHECK(
      available(weight, bias, use_batch),
      "Vulkan Linear not available! "
      "Reason: The provided (weight, bias) parameters are either invalid "
      "individually or their combination is not supported by Vulkan Impl.");

  // 初始化 packed_ 数组，用于存储打包后的参数
  packed_.reserve(Packed::NumArgs);
  // 将权重 weight 打包并添加到 packed_ 数组中
  packed_.emplace_back(convert(pack_weights(weight, use_batch)));
  // 根据权重是否量化，打包并添加偏置到 packed_ 数组中
  const auto& packed_biases = weight.is_quantized()
      ? pack_biases_quantized_weights(weight, bias, use_batch)
      : pack_biases(weight, bias, use_batch);
  packed_.emplace_back(convert(packed_biases));
  // 添加权重的尺寸信息到 packed_ 数组中
  packed_.emplace_back(weight.sizes());
  // 若存在偏置，则将偏置的定义状态添加到 packed_ 数组中
  packed_.emplace_back(bias && bias->defined());

  // 如果全局上下文不释放权重，则初始化 unpacked_ 数组，用于存储未打包的参数
  if (!at::globalContext().releaseWeightsWhenPrepacking()) {
    unpacked_.reserve(Unpacked::NumArgs);
    unpacked_.emplace_back(weight);
    unpacked_.emplace_back(bias);
  }
}

// 静态方法，将 unpacked 参数列表打包成 LinearPackedContext 对象并返回
LinearPackedContext LinearPackedContext::pack(c10::impl::GenericList unpacked) {
  return LinearPackedContext(
      unpacked.get(Unpacked::Weight).toTensor(),
      get_optional_tensor(unpacked, Unpacked::Bias));
}

// 创建线性层上下文的函数，返回 LinearPackedContext 的智能指针
c10::intrusive_ptr<LinearPackedContext> create_linear_context(
    Tensor&& weight,
  // 使用 std::optional 包装的 Tensor 类型的权重和偏置参数，构造 LinearPackedContext 对象
  return c10::make_intrusive<LinearPackedContext>(
      // 使用权重和偏置参数初始化 LinearPackedContext 对象
      LinearPackedContext(weight, bias));
}

// 结束 ops 命名空间
namespace ops {
// 结束 vulkan 命名空间
namespace vulkan {
// 结束 native 命名空间
namespace native {
// 结束 at 命名空间
namespace at {

// 运行线性加法矩阵乘法运算的上下文，返回结果张量
Tensor run_linear_context(
    const Tensor& input,
    const c10::intrusive_ptr<LinearPackedContext>& linear_context) {
  // 调用通用的加法矩阵乘法运算上下文函数，参数为输入张量、权重因子、偏置因子、线性上下文、是否量化、输出缩放因子、输出零点
  return run_addmm_context(input, 1.0f, 1.0f, linear_context, false, 0, 0);
}

// 运行量化线性加法矩阵乘法运算的上下文，返回结果张量
Tensor run_qlinear_context(
    const Tensor& input_arg,
    double output_scale,
    int64_t output_zero_point,
    const c10::intrusive_ptr<LinearPackedContext>& linear_context) {
  // 调用通用的加法矩阵乘法运算上下文函数，参数为输入张量、权重因子、偏置因子、线性上下文、是否量化、输出缩放因子、输出零点
  return run_addmm_context(
      input_arg,
      1.0f,
      1.0f,
      linear_context,
      true,
      output_scale,
      output_zero_point);
}

} // namespace at
} // namespace native
} // namespace vulkan
} // namespace ops
```