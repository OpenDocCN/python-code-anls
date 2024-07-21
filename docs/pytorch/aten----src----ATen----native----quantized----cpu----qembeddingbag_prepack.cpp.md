# `.\pytorch\aten\src\ATen\native\quantized\cpu\qembeddingbag_prepack.cpp`

```py
// 定义宏以仅支持方法操作符断言
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 引入嵌入包预打包头文件
#include <ATen/native/quantized/cpu/qembeddingbag_prepack.h>

// 引入并行处理头文件
#include <ATen/Parallel.h>
// 引入通用实用功能头文件
#include <ATen/Utils.h>
// 引入张量核心功能头文件
#include <ATen/core/Tensor.h>
// 引入自定义类头文件
#include <ATen/core/custom_class.h>
// 引入量化嵌入包参数头文件
#include <ATen/native/quantized/cpu/EmbeddingPackedParams.h>
// 引入量化嵌入包工具头文件
#include <ATen/native/quantized/cpu/fbgemm_utils.h>
// 引入标量类型头文件
#include <c10/core/ScalarType.h>
// 引入Torch库头文件
#include <torch/library.h>

// 如果未包含每个操作符的头文件，则引入函数操作头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
// 否则，引入优化选择参数头文件、空头文件和本地调整大小头文件
#else
#include <ATen/ops/choose_qparams_optimized.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/resize_native.h>
#endif

// 引入C10实用工具范围头文件
#include <c10/util/irange.h>

// 引入实用工具头文件
#include <utility>

// 注册嵌入参数函数声明
int register_embedding_params();

/*
 * 嵌入包装预打包函数。
 * 此函数期望每行均为量化权重张量，具有浮点比例和零点值。
 * 零点设置为(-Xmin/scale)
 * 为了预打包权重，我们为每行存储比例和偏差（其中偏差为Xmin），
 * 以及量化权重。
 */
// 返回包装的嵌入包装参数的智能指针
c10::intrusive_ptr<EmbeddingPackedParamsBase> PackedEmbeddingBagWeight::prepack(
    at::Tensor qweight) {
  // 预期权重张量的秩为2
  static constexpr int64_t version = 1;
  TORCH_CHECK(
      qweight.dim() == 2,
      "quantized::embedding_bag_prepack weight tensor rank should be 2");
  TORCH_CHECK(
      qweight.scalar_type() == c10::kQUInt8 ||
          qweight.scalar_type() == c10::kQUInt4x2,
      "qembedding_bag_prepack currently only supports quint8 and quint4x2 weights");

  // 获得连续存储的权重张量
  at::Tensor weight_contig =
      qweight.contiguous(qweight.suggest_memory_format());

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  // 定义比特宽度和比例偏差字节数
  int bit_width, scale_bias_bytes;
  // 获取权重数据指针
  uint8_t* weight_data = static_cast<uint8_t*>(weight_contig.data_ptr());
  // 根据张量类型设置比特宽度和比例偏差字节数
  if (qweight.scalar_type() == c10::kQUInt8) {
    bit_width = 8;
    scale_bias_bytes =
        sizeof(float) * 2; // 每行额外8字节以存储FP比例和偏差。
  } else {
    bit_width = 4;
    scale_bias_bytes = sizeof(at::Half) *
        2; // 每行额外4字节以存储at::Half比例和偏差。
  }
  // 每字节的元素数
  const auto num_elem_per_byte = 8 / bit_width;

  // 权重张量的嵌入行数和列数
  int64_t embedding_rows = qweight.size(0);
  int64_t embedding_cols = qweight.size(1);
  // 获取量化方案类型
  const auto qtype = qweight.qscheme();
  TORCH_CHECK(
      qtype == c10::kPerChannelAffineFloatQParams,
      "Expect embedding_bag weights to be quantized using kPerChannelAffineFloatQParams");
  // 存储权重偏差的向量
  std::vector<float> weight_bias(embedding_rows);

  // 获取每通道比例和零点
  at::Tensor channel_scales = qweight.q_per_channel_scales();
  at::Tensor channel_zero_points = qweight.q_per_channel_zero_points();
  // 获取权重比例和零点的向量
  std::vector<float> weight_scales(
      channel_scales.data_ptr<float>(),
      channel_scales.data_ptr<float>() + embedding_rows);
  std::vector<float> weight_zero_points(
      channel_zero_points.data_ptr<float>(),
      channel_zero_points.data_ptr<float>() + embedding_rows);

  // 对于每个嵌入行的循环
  for (const auto i : c10::irange(embedding_rows)) {
    // 计算每个权重偏置的值，使用权重零点和权重比例乘积再乘以 -1
    weight_bias[i] = weight_zero_points[i] * weight_scales[i] * -1;
  }

  // 计算输出张量的形状，包括嵌入行数和用于存储每行比例和偏置的额外字节
  std::vector<int64_t> output_shape = {
      embedding_rows,
      static_cast<std::int64_t>(
          (embedding_cols + num_elem_per_byte - 1) / num_elem_per_byte +
          scale_bias_bytes)}; // extra bytes to store scale and bias per row.
  size_t output_columns = output_shape[1];

  // 分配输出打包后的权重张量
  at::Tensor output = at::empty(
      output_shape,
      weight_contig.options().dtype(at::kByte),
      weight_contig.suggest_memory_format());
  auto* output_data = output.data_ptr<uint8_t>();

  if (bit_width == 8) {
    // 如果比特宽度为 8，使用并行处理循环，为每行打包权重数据
    at::parallel_for(
        0, embedding_rows, 1, [&](int64_t start_idx, int64_t end_idx) {
          for (const auto row : c10::irange(start_idx, end_idx)) {
            const uint8_t* input_row = weight_data + row * embedding_cols;
            std::uint8_t* output_row = output_data + row * output_columns;
            auto output_row_scale_bias = output_row + embedding_cols;
            // 避免使用 float* 来避免未对齐地址访问，将比例和偏置拷贝到输出行
            std::memcpy(
                output_row_scale_bias, &(weight_scales[row]), sizeof(float));
            std::memcpy(
                output_row_scale_bias + sizeof(float),
                &(weight_bias[row]),
                sizeof(float));
            // 将输入行的数据复制到输出行
            for (const auto col : c10::irange(embedding_cols)) {
              output_row[col] = input_row[col];
            }
          }
        });
  } else {
    // 如果比特宽度不为 8，重新计算每行的嵌入列数以便打包到字节中
    embedding_cols =
        (embedding_cols + num_elem_per_byte - 1) / num_elem_per_byte;
    // 使用并行处理循环，为每行打包权重数据
    at::parallel_for(
        0, embedding_rows, 1, [&](int64_t start_idx, int64_t end_idx) {
          for (const auto row : c10::irange(start_idx, end_idx)) {
            const uint8_t* input_row = weight_data + row * embedding_cols;
            std::uint8_t* output_row = output_data + row * output_columns;
            auto output_row_scale_bias = output_row + embedding_cols;
            at::Half weight_scale = weight_scales[row];
            at::Half weight_bias_half = weight_bias[row];
            // 避免使用 at::Half* 来避免未对齐地址访问，将比例和偏置拷贝到输出行
            std::memcpy(output_row_scale_bias, &weight_scale, sizeof(at::Half));
            std::memcpy(
                output_row_scale_bias + sizeof(at::Half),
                &weight_bias_half,
                sizeof(at::Half));
            // 将输入行的数据复制到输出行
            for (const auto col : c10::irange(embedding_cols)) {
              // 权重值已经打包，这里只需将其存储到输出张量中
              output_row[col] = input_row[col];
            }
          }
        });
  }

  // 创建打包后的嵌入包权重对象，并返回指针
  auto packed_ptr = c10::make_intrusive<PackedEmbeddingBagWeight>(
      output,
      std::move(weight_scales),
      std::move(weight_zero_points),
      bit_width,
      qtype,
      version);

  return packed_ptr;
// Note - This is a temporary pack function for embedding bag which quantizes
// and packs the float weight tensor. In the next step it will be replaced by a
// quantize and pack function once we support FP scale and FP zero_point
//
// Python example examining a packed 8bit zero_point and scale:
//
// >> x = torch.from_numpy(np.array([[[10, 20], [30, 40]],[[50, 60], [70, 80]]],
// dtype=np.float32))
// >> x_packed = torch.ops.quantized.embedding_bag_byte_prepack(x);
//
// # Pull out and examine packed scales, zero_points and values
// >> zero_points = x_packed[:,:,-4:].numpy();
// >> scales = x_packed[:,:,-8:-4].numpy();
// >> values = x_packed[:,:,:-8].numpy();
//
// >> zero_points
// array([[[  0,   0,  32,  65],
//        [  0,   0, 240,  65]],
//
//       [[  0,   0,  72,  66],
//        [  0,   0, 140,  66]]], dtype=uint8)
//
// >> scales
// array([[[161, 160,  32,  61],
//        [161, 160,  32,  61]],
//
//       [[161, 160,  32,  61],
//        [161, 160,  32,  61]]], dtype=uint8)
// >> values
// array([[[  0, 255],
//        [  0, 255]],
//
//       [[  0, 255],
//        [  0, 255]]], dtype=uint8)
//
// # Convert 4 byte packed scales and zero_points to float
// # and apply against values in order to recover unquantized values.
// def bytes2float(arr);
//    packed_hex = bytearray(arr);
//    return struct.unpack('f', packed_hex);
//
// >> float_zero_points = np.apply_along_axis(bytes2float, 2, zero_points);
// >> float_zero_points
// array([[[10.],
//         [30.]],
//
//        [[50.],
//         [70.]]])
// >> float_scales = np.apply_along_axis(bytes2float, 2, scales);
// >> float_scales
// array([[[0.03921569],
//        [0.03921569]],
//
//       [[0.03921569],
//        [0.03921569]]])
// >> values *  float_scales + float_zero_points
// array([[[10.        , 20.00000035],
//         [30.        , 40.00000035]],
//
//        [[50.        , 60.00000035],
//         [70.        , 80.00000035]]])
// 更新输出张量 `output` 的数据以进行量化处理，基于输入张量 `weight` 的内容
Tensor& qembeddingbag_byte_prepack_out(Tensor& output, const Tensor& weight) {
  // 检查输入张量 `weight` 的数据类型是否为 float32 或 float16
  TORCH_CHECK(
      weight.scalar_type() == at::ScalarType::Float ||
          weight.scalar_type() == at::ScalarType::Half,
      "'embedding_bag_byte_prepack' only support float32 or float16.");

  // 获取输入张量 `weight` 的尺寸信息
  const auto weight_sizes = weight.sizes();
  // 确定张量中列的维度
  const auto cols_dim = weight_sizes.size() - 1;
  // 计算嵌入行数
  const int64_t embedding_rows = c10::size_to_dim_(cols_dim, weight_sizes);
  // 获取每行的列数（嵌入维度）
  const int32_t embedding_cols = weight_sizes[cols_dim];
  // 每列增加 8 个字节以存储每行的 FP32 缩放因子和零点
  const int32_t output_columns = embedding_cols + 2 * sizeof(float);
  // 期望输入张量 `weight` 是连续存储的
  const auto weight_contig =
      weight.expect_contiguous(weight.suggest_memory_format());

  // 调整输出张量的维度，以适应新的列数（包括额外的 FP32 信息）
  std::vector<int64_t> output_shape = weight_sizes.vec();
  output_shape[cols_dim] = output_columns;
  at::native::resize_(output, output_shape, c10::nullopt);
  // 获取输出张量的数据指针
  auto* output_data = output.data_ptr<uint8_t>();

  // 如果使用了 FBGEMM 库
#ifdef USE_FBGEMM
  // 如果输入张量 `weight` 的数据类型是 Half
  if (weight_contig->scalar_type() == at::ScalarType::Half) {
    // 获取输入张量 `weight` 的数据指针，并转换为 float16 类型
    const auto weight_data =
        static_cast<fbgemm::float16*>(weight_contig->data_ptr());
    // 使用并行处理方式，将 float16 转换为融合 8 位行压缩的浮点数
    at::parallel_for(
        0, embedding_rows, 1, [&](int64_t start_idx, int64_t end_idx) {
          fbgemm::FloatOrHalfToFused8BitRowwiseQuantizedSBFloat<
              fbgemm::float16>(
              weight_data + start_idx * embedding_cols,
              end_idx - start_idx,
              embedding_cols,
              output_data + start_idx * output_columns);
        });
  } else { // 如果输入张量 `weight` 的数据类型是 Float
    // 获取输入张量 `weight` 的数据指针，并转换为 float 类型
    const auto weight_data = weight_contig->data_ptr<float>();
    // 使用并行处理方式，将 float 转换为融合 8 位行压缩的浮点数
    at::parallel_for(
        0, embedding_rows, 1, [&](int64_t start_idx, int64_t end_idx) {
          fbgemm::FloatOrHalfToFused8BitRowwiseQuantizedSBFloat<float>(
              weight_data + start_idx * embedding_cols,
              end_idx - start_idx,
              embedding_cols,
              output_data + start_idx * output_columns);
        });
  }
}
#else
  // 如果权重张量不是连续的，将其转换为连续张量并赋值给 float_weight
  const Tensor& float_weight =
      weight_contig->scalar_type() == at::ScalarType::Half
      ? weight_contig->to(at::ScalarType::Float)
      : *weight_contig;
  // 获取权重数据指针
  const auto weight_data = float_weight.data_ptr<float>();
  // 定义一个极小值常量 kEpsilon
  constexpr float kEpsilon = 1e-8f;
  // 遍历嵌入行数
  for (auto row : c10::irange(embedding_rows)) {
    // 获取输入行的起始指针
    const float* input_row = weight_data + row * embedding_cols;
    // 获取输出行的起始指针
    std::uint8_t* output_row = output_data + row * output_columns;
    // 获取输出行中用于存储缩放和零点的指针
    float* output_row_scale_zp =
        reinterpret_cast<float*>(output_row + embedding_cols);

    // 计算输入行的最小元素
    float minimum_element =
        *std::min_element(input_row, input_row + embedding_cols);
    // 计算输入行的最大元素
    float maximum_element =
        *std::max_element(input_row, input_row + embedding_cols);
    // 计算输入行的值范围
    float range = maximum_element - minimum_element;

    // 计算并存储缩放因子到输出行的第一个位置
    output_row_scale_zp[0] = range / 255.0f;
    // 存储输入行的最小元素到输出行的第二个位置
    output_row_scale_zp[1] = minimum_element;
    // 计算反向缩放因子
    const auto inverse_scale = 255.0f / (range + kEpsilon);
    // 遍历嵌入列数
    for (auto col : c10::irange(embedding_cols)) {
      // 将输入行的每个元素按照反向缩放和四舍五入后存储到输出行中
      output_row[col] =
          lrintf((input_row[col] - minimum_element) * inverse_scale);
    } // embedding_cols
  } // embedding_rows
#endif // USE_FBGEMM

// 返回输出张量
return output;
}

// 根据权重张量预打包字节数据
Tensor qembeddingbag_byte_prepack(const Tensor& weight) {
  // 确保权重张量是连续的
  const auto weight_contig =
      weight.expect_contiguous(weight.suggest_memory_format());
  // 创建一个空的 CPU 张量作为输出
  Tensor output = at::detail::empty_cpu(
      {0},
      at::kByte,
      weight_contig->layout(),
      weight_contig->device(),
      c10::nullopt,
      c10::nullopt);
  // 调用 qembeddingbag_byte_prepack_out 函数进行打包
  qembeddingbag_byte_prepack_out(output, weight);
  // 返回输出张量
  return output;
}

// 根据权重张量预打包字节数据的元数据
Tensor qembeddingbag_byte_prepack_meta(const Tensor& weight) {
  // 确保权重张量是连续的
  const auto weight_contig =
      weight.expect_contiguous(weight.suggest_memory_format());
  // 检查权重张量的标量类型是否为 float32 或 float16
  TORCH_CHECK(
      weight.scalar_type() == at::ScalarType::Float ||
          weight.scalar_type() == at::ScalarType::Half,
      "'embedding_bag_byte_prepack' only support float32 or float16.");
  // 获取权重张量的尺寸
  const auto weight_sizes = weight.sizes();
  // 获取张量的列数维度
  const auto cols_dim = weight_sizes.size() - 1;
  // 获取嵌入列数
  const int32_t embedding_cols = weight_sizes[cols_dim];
  // 每列增加 8 个字节用于存储每行的 FP32 缩放因子和零点
  const int32_t output_columns = embedding_cols + 2 * sizeof(float);

  // 调整输出张量的维度以适应 FP32 缩放因子和零点
  std::vector<int64_t> output_shape = weight_sizes.vec();
  output_shape[cols_dim] = output_columns;
  at::SymDimVector output_shape_vec(output_shape);

  // 创建一个空的符号整数张量作为输出
  return at::empty_symint(
      output_shape_vec,
      weight.options().dtype(weight.scalar_type()),
      weight.suggest_memory_format());
}

namespace {

// TODO: 扩展支持到 N-D 批量嵌入，类似于 qembeddingbag_byte_prepack
// 辅助函数，根据权重张量预打包 N 位数据
Tensor _qembeddingbag_nbit_prepack_helper(
    const Tensor& weight,
    int bit_width,
    const bool optimized_qparams,
    const int64_t nbins,
    // 检查权重张量的数据类型是否为Float或Half，否则抛出错误提示信息
    TORCH_CHECK(
        weight.scalar_type() == at::ScalarType::Float ||
            weight.scalar_type() == at::ScalarType::Half,
        "'qembeddingbag_nbit_prepack' only support float32 or float16.");
    
    // 获取权重张量的行数和列数
    int64_t embedding_rows = weight.size(0);
    int64_t embedding_cols = weight.size(1);
    
    // 将权重张量转换为内存连续的形式，可能会按建议的内存格式进行优化
    Tensor weight_contig = weight.contiguous(weight.suggest_memory_format());
    
    // 检查bit_width参数是否为2或4，否则抛出错误提示信息
    TORCH_CHECK(
        bit_width == 4 || bit_width == 2,
        "bit_width must be either 2 or 4 to use 'qembeddingbag_nbit_prepack'."
        "For 8bit, consider using 'embedding_bag_byte_prepack'.");
    
    // 计算每字节中包含的元素数量，以字节宽度计算
    int NUM_ELEM_PER_BYTE = 8 / bit_width;
    // 检查权重张量的最后一个维度的大小是否是NUM_ELEM_PER_BYTE的整数倍，否则抛出错误提示信息
    TORCH_CHECK(
        weight_contig.size(weight.dim() - 1) % NUM_ELEM_PER_BYTE == 0,
        "qembeddingbag_",
        std::to_string(bit_width),
        "bit_prepack only works for the number of columns a multiple of ",
        std::to_string(NUM_ELEM_PER_BYTE));
    
    // "融合"表示法将比例和偏置与按行量化的数据一起存储在一个张量中
    // 由于比例和偏置用16位浮点数表示，我们将使用每行的最后4字节存储（2字节比例 + 2字节偏置）
    // | ... 量化数据 ... | 比例 | 偏置 |
    // |   列数     |  2字节   |  2字节  |
    std::vector<int64_t> output_shape = {
        embedding_rows,
        static_cast<std::int64_t>(
            (embedding_cols + NUM_ELEM_PER_BYTE - 1) / NUM_ELEM_PER_BYTE +
            2 * sizeof(at::Half))};
    // 创建一个与output_shape相匹配的空张量，数据类型为kByte
    auto output = at::empty(
        output_shape,
        weight_contig.options().dtype(at::kByte),
        weight_contig.suggest_memory_format());
    // 获取输出张量的数据指针，类型为uint8_t
    auto* output_data = output.data_ptr<uint8_t>();
#ifdef USE_FBGEMM
  // 如果未优化的量化参数不存在
  if (!optimized_qparams) {
    // 如果权重张量的数据类型是半精度浮点数
    if (weight_contig.scalar_type() == at::ScalarType::Half) {
      // 将权重数据转换为半精度浮点数类型
      const auto weight_data =
          static_cast<fbgemm::float16*>(weight_contig.data_ptr());
      // 使用并行方式处理每个嵌入行的数据
      at::parallel_for(
          0, embedding_rows, 1, [&](int64_t start_idx, int64_t end_idx) {
            // 将半精度浮点数转换为N比特行逐行量化的半精度浮点数
            fbgemm::FloatOrHalfToFusedNBitRowwiseQuantizedSBHalf<
                fbgemm::float16>(
                bit_width,
                weight_data + start_idx * embedding_cols,
                end_idx - start_idx,
                embedding_cols,
                output_data + start_idx * output_shape[1]);
          });
    } else {
      // 否则，权重数据为单精度浮点数
      const auto weight_data = weight_contig.data_ptr<float>();
      // 使用并行方式处理每个嵌入行的数据
      at::parallel_for(
          0, embedding_rows, 1, [&](int64_t start_idx, int64_t end_idx) {
            // 将单精度浮点数转换为N比特行逐行量化的半精度浮点数
            fbgemm::FloatOrHalfToFusedNBitRowwiseQuantizedSBHalf<float>(
                bit_width,
                weight_data + start_idx * embedding_cols,
                end_idx - start_idx,
                embedding_cols,
                output_data + start_idx * output_shape[1]);
          });
    }
  } else {
#endif // USE_FBGEMM
    // 计算输出张量的最后一个维度大小
    const auto output_columns = output.size(output.dim() - 1);
    // 将权重张量转换为单精度浮点数类型（如果原始数据类型为半精度浮点数）
    const auto float_weight =
        weight_contig.scalar_type() == at::ScalarType::Half
        ? weight_contig.to(at::ScalarType::Float)
        : std::move(weight_contig);
    // 获取转换后的权重数据的指针
    const auto weight_data = float_weight.data_ptr<float>();
    // 遍历每一行的嵌入数据
    for (const auto row : c10::irange(embedding_rows)) {
      // 获取当前行的输入数据起始地址
      const float* input_row = weight_data + row * embedding_cols;
      // 获取当前行的输出数据起始地址
      std::uint8_t* output_row = output_data + row * output_columns;

      // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
      // 初始化 Xmin 和 Xmax 变量
      float Xmin, Xmax;
      // 如果启用了优化的量化参数计算
      if (optimized_qparams) {
        // 调用优化的量化参数选择函数
        auto [xmax_tensor, xmin_tensor] = at::choose_qparams_optimized(
            float_weight[row], embedding_cols, nbins, ratio, bit_width);
        // 检查返回的 xmax 和 xmin 张量大小是否为 1
        TORCH_CHECK(
            xmax_tensor.numel() == 1 && xmin_tensor.numel() == 1,
            "Expected choose_qparams_optimized to return min/max tensors of size 1");
        // 获取具体的最大和最小值
        Xmax = xmax_tensor.item<float>();
        Xmin = xmin_tensor.item<float>();
      } else {
        // 否则，计算当前行输入数据的最大和最小值
        Xmin = *std::min_element(input_row, input_row + embedding_cols);
        Xmax = *std::max_element(input_row, input_row + embedding_cols);
      }
      // 将 Xmin 转换为半精度浮点数
      Xmin = static_cast<at::Half>(Xmin);
      // 计算数值范围
      float range = Xmax - Xmin;
      // 计算量化比例因子 scale，确保不出现除以零的情况
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      at::Half scale = range == 0 ? 1.0f : range / ((1 << bit_width) - 1);
      float inverse_scale = scale == 0 ? 1.0f : 1.0f / scale;
      // 处理特殊情况，当 scale 为零或 inverse_scale 为无穷大时
      if (scale == 0 || std::isinf(inverse_scale)) {
        // 如果 Xmax == Xmin，则任何 scale 值都可以，设置为 1.0f
        scale = 1.0f;
        inverse_scale = 1.0f;
      }
      // 更新每行输出的 scale 和 zero_point
      at::Half* output_row_scale_zp = reinterpret_cast<at::Half*>(
          output_row +
          (embedding_cols + NUM_ELEM_PER_BYTE - 1) / NUM_ELEM_PER_BYTE);

      output_row_scale_zp[0] = scale;
      output_row_scale_zp[1] = Xmin;

      // 打包权重值
      for (const auto col : c10::irange(embedding_cols)) {
        float X = input_row[col];
        // 计算量化后的值并限制在合理范围内
        std::uint8_t quantized = std::max(
            0,
            std::min<int>(
                lrintf((X - Xmin) * inverse_scale), (1 << bit_width) - 1));
        // 将两个 4 位值打包到一个字节中，第一个值占据低 4 位，第二个值占据高 4 位
        if (col % NUM_ELEM_PER_BYTE == 0) {
          output_row[col / NUM_ELEM_PER_BYTE] = quantized;
        } else {
          output_row[col / NUM_ELEM_PER_BYTE] |=
              (quantized << ((col % NUM_ELEM_PER_BYTE) * bit_width));
        }
      } // embedding_cols
    } // embedding_rows
#ifdef USE_FBGEMM
  }
#endif // USE_FBGEMM



// 如果定义了 USE_FBGEMM 宏，则关闭相应的命名空间
#endif // USE_FBGEMM

// 返回一个张量
return output;
}

// 对输入矩阵进行4位行级量化，确定每行的范围（最大-最小值）和偏差（最小值），
// 然后将每个元素缩放为0到15之间的2位数字。
// 为了后续反量化值，存储了比例（范围 / 15）和零点偏移量，
// 具体来说，每行首先有量化值，然后是2字节的fp16比例和2字节的零点偏移。
Tensor qembeddingbag_4bit_prepack(
    const Tensor& weight,
    const bool optimized_qparams,
    const int64_t nbins,
    const double ratio) {
  return _qembeddingbag_nbit_prepack_helper(
      weight, 4 /*bit_width*/, optimized_qparams, nbins, ratio);
}

// 对输入矩阵进行2位行级量化，确定每行的范围（最大-最小值）和偏差（最小值），
// 然后将每个元素缩放为0到3之间的2位数字。
// 为了后续反量化值，存储了比例（范围 / 3）和零点偏移量，
// 具体来说，每行首先有量化值，然后是2字节的fp16比例和2字节的零点偏移。
// TODO() - 添加2位嵌入查找操作符。
Tensor qembeddingbag_2bit_prepack(
    const Tensor& weight,
    const bool optimized_qparams,
    const int64_t nbins,
    const double ratio) {
  return _qembeddingbag_nbit_prepack_helper(
      weight, 2 /*bit_width*/, optimized_qparams, nbins, ratio);
}

// 定义 QEmbeddingPackWeights 类
class QEmbeddingPackWeights final {
 public:
  // 静态方法，接收权重张量并返回封装后的嵌入包参数基类指针
  static c10::intrusive_ptr<EmbeddingPackedParamsBase> run(at::Tensor weight) {
    return PackedEmbeddingBagWeight::prepack(std::move(weight));
  }
};

// 在 quantized 库的 CPU 实现中注册预打包函数
TORCH_LIBRARY_IMPL(quantized, CPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("quantized::embedding_bag_byte_prepack"),
      TORCH_FN(qembeddingbag_byte_prepack));
  m.impl(
      TORCH_SELECTIVE_NAME("quantized::embedding_bag_4bit_prepack"),
      TORCH_FN(qembeddingbag_4bit_prepack));
  m.impl(
      TORCH_SELECTIVE_NAME("quantized::embedding_bag_2bit_prepack"),
      TORCH_FN(qembeddingbag_2bit_prepack));
}

// 在 quantized 库的 QuantizedCPU 实现中注册预打包函数
TORCH_LIBRARY_IMPL(quantized, QuantizedCPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("quantized::embedding_bag_prepack"),
      TORCH_FN(QEmbeddingPackWeights::run));
}

// 在 quantized 库的 Meta 实现中注册预打包函数
TORCH_LIBRARY_IMPL(quantized, Meta, m) {
  m.impl(
      "quantized::embedding_bag_byte_prepack", qembeddingbag_byte_prepack_meta);
}

} // namespace
} // namespace native
} // namespace at
```