# `.\pytorch\aten\src\ATen\native\quantized\cpu\qembeddingbag_unpack.cpp`

```
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Parallel.h>
#include <ATen/native/quantized/cpu/EmbeddingPackedParams.h>
#include <ATen/native/quantized/cpu/fbgemm_utils.h>
#include <ATen/native/quantized/cpu/qembeddingbag.h>
#include <c10/util/irange.h>
#include <torch/library.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_empty_per_channel_affine_quantized.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/from_blob.h>
#include <ATen/ops/resize_native.h>
#endif

// 注册嵌入参数函数的声明
int register_embedding_params();

// 解压缩嵌入包装参数的函数定义
at::Tensor PackedEmbeddingBagWeight::unpack() {
  // 将 packed_w 赋值给 packed_weight
  auto packed_weight = packed_w;
  // 声明一个名为 weight_origin 的张量
  at::Tensor weight_origin;

  // 如果量化位数为 8 或 4
  if (bit_rate_ == 8 || bit_rate_ == 4) {
    // 获取输入的行数和列数
    const auto input_rows = packed_weight.size(0);
    const auto input_columns = packed_weight.size(1);
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    int scale_bias_bytes;
    // 每字节的元素数量
    const auto num_elem_per_byte = 8 / bit_rate_;
    // 如果量化位数为 8
    if (bit_rate_ == 8) {
      // 最后两个值用于存储每行的 FP32 scale 和 zero_point 值
      scale_bias_bytes = 8;
    } else {
      // 如果量化位数为 4
      scale_bias_bytes = 4;
    }

    // 获取 packed_weight 的常量数据指针
    const auto* input = packed_weight.const_data_ptr<uint8_t>();
    // 计算输出形状，考虑最后 n 字节用于 scale/bias，其余条目根据位宽打包
    std::vector<int64_t> output_shape = {
        input_rows,
        static_cast<std::int64_t>(input_columns - scale_bias_bytes) *
            num_elem_per_byte};

    // 从数组 w_scale 创建张量 scales，数据类型为 kFloat，存储于 CPU 上
    auto scales = at::from_blob(
        w_scale.data(), w_scale.size(), device(c10::kCPU).dtype(c10::kFloat));
    // 从数组 w_zp 创建张量 zero_points，数据类型为 kFloat，存储于 CPU 上
    auto zero_points = at::from_blob(
        w_zp.data(), w_zp.size(), device(c10::kCPU).dtype(c10::kFloat));

    // 获取输出的列数
    auto output_columns = output_shape[1];
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    uint8_t* output_data;

    // 根据位宽分配输出权重张量
    if (bit_rate_ == 8) {
      // 使用 per-channel affine quantized 空张量创建权重张量，数据类型为 kQUInt8
      weight_origin = at::_empty_per_channel_affine_quantized(
          output_shape,
          scales.toType(c10::kFloat),
          zero_points.toType(c10::kFloat),
          0, // 输出通道轴为 0
          device(c10::kCPU).dtype(c10::kQUInt8));
      // 获取权重张量的数据指针
      output_data = static_cast<uint8_t*>(weight_origin.data_ptr());
    } else {
      // 使用 per-channel affine quantized 空张量创建权重张量，数据类型为 kQUInt4x2
      weight_origin = at::_empty_per_channel_affine_quantized(
          output_shape,
          scales.toType(c10::kFloat),
          zero_points.toType(c10::kFloat),
          0, // 输出通道轴为 0
          device(c10::kCPU).dtype(c10::kQUInt4x2));
      // 获取权重张量的数据指针
      output_data = static_cast<uint8_t*>(weight_origin.data_ptr());
    }

    // 从 packed_weight 复制数据到输出
    // 并行处理每行数据，拷贝压缩字节数据到输出张量中，适用于子字节张量的情况，因为子字节量化张量预期以压缩格式存储数据。

    // 使用并行处理，遍历输入张量的行数范围
    at::parallel_for(0, input_rows, 1, [&](int32_t start_idx, int32_t end_idx) {
      for (const auto row : c10::irange(start_idx, end_idx)) {
        // 指向当前输入行的指针
        const std::uint8_t* input_row = input + row * input_columns;
        // 指向当前输出行的指针，将数据写入到输出数据中，考虑每字节元素的数量
        uint8_t* output_row =
            output_data + row * output_columns / num_elem_per_byte;

        // 针对输出列进行迭代处理，将输入行的数据复制到输出行
        for (const auto col : c10::irange(output_columns / num_elem_per_byte)) {
          output_row[col] = input_row[col];
        }
      }
    });

    // 如果程序执行到此处，表示量化类型不支持，抛出内部断言错误并返回原始权重数据
    return weight_origin;
  }
  
  // 如果程序执行到此处，表示量化类型不支持，抛出内部断言错误并返回原始权重数据
  TORCH_INTERNAL_ASSERT(
      false,
      "We currently only support 8-bit and 4-bit quantization of embedding_bag.");
  return weight_origin;
}

namespace at {
namespace native {

// 解包量化的嵌入权重数据，并将结果写入输出张量
Tensor& qembeddingbag_byte_unpack_out(Tensor& output, const Tensor& packed_weight) {
  // N 维嵌入袋的最后一个维度是量化通道。例如，对于二维嵌入袋，维度可能是 [行, 列]；对于批量的嵌入袋，维度可能是 [批次, 行, 列]。
  // Python 批量嵌入的示例：
  // weights = torch.from_numpy((np.random.random_sample((
  //          2, 10, 3)).squeeze() + 1).astype(np.float32))
  // assert(weights.size() == torch.Size([2, 10, 3]))
  // # 注意：由于 fp32 的 zero_point 和 scale，每行多了 8 个字节（列）
  // packed_weights = torch.ops.quantized.embedding_bag_byte_prepack(weights)
  // assert(packed_weights.size() == torch.Size([2, 10, 11]))
  // unpacked_weights = torch.ops.quantized.embedding_bag_byte_unpack(packed_weights)
  // assert(unpacked_weights.size() == torch.Size([2, 10, 3]))

  // 获取输入张量 packed_weight 的尺寸
  const auto packed_weight_sizes = packed_weight.sizes();
  // 获取最后一个维度的索引
  const auto col_dim = packed_weight_sizes.size() - 1;
  // 获取输入行数
  const int64_t input_rows = c10::size_to_dim_(col_dim, packed_weight_sizes);
  // 获取输入列数
  const int32_t input_columns = packed_weight_sizes[col_dim];
  // 输出列数为输入列数减去两个 float 类型的尺寸（scale 和 zero_point）
  const int32_t output_columns = input_columns - 2 * sizeof(float);
  // 获取 packed_weight 的数据指针
  const auto* input_data = packed_weight.const_data_ptr<uint8_t>();

  // 根据 packed_weight 的尺寸调整输出张量 output 的形状
  std::vector<int64_t> output_shape = packed_weight_sizes.vec();
  output_shape[col_dim] = output_columns;
  at::native::resize_(output, output_shape);
  // 获取 output 张量的连续版本
  auto output_contig = output.expect_contiguous();
  // 获取输出数据的 float 类型指针
  float* output_data = output_contig->data_ptr<float>();

#ifdef USE_FBGEMM
  // 使用 FBGEMM 库进行并行计算
  at::parallel_for(0, input_rows, 1, [&](int64_t start_idx, int64_t end_idx) {
    fbgemm::Fused8BitRowwiseQuantizedSBFloatToFloatOrHalf<float>(
        input_data + start_idx * input_columns,
        end_idx - start_idx,
        input_columns,
        output_data + start_idx * output_columns);
  });
#else
  // 如果未定义 USE_FBGEMM，使用普通的循环计算
  for (auto row : c10::irange(input_rows)) {
    const std::uint8_t* input_row = input_data + row * input_columns;
    const float* input_row_scale_zp =
        reinterpret_cast<const float*>(input_row + output_columns);
    float* output_row = output_data + row * output_columns;

    for (auto col : c10::irange(output_columns)) {
      // 计算解包后的权重值
      output_row[col] =
          input_row[col] * input_row_scale_zp[0] + input_row_scale_zp[1];
    } // output_columns
  } // input_rows
#endif // USE_FBGEMM
  // 返回输出张量
  return output;
}

namespace {
// 调用 qembeddingbag_byte_unpack_out 函数进行解包，并返回输出张量
Tensor qembeddingbag_byte_unpack(const Tensor& packed_weight) {
  at::Tensor output = at::empty(
      {},
      packed_weight.options().dtype(kFloat),
      packed_weight.suggest_memory_format());
  qembeddingbag_byte_unpack_out(output, packed_weight);
  return output;
}
// 对输入的 packed_weight 进行解压，返回解压后的元数据 Tensor
Tensor qembeddingbag_byte_unpack_meta(const Tensor& packed_weight) {
  // 获取 packed_weight 的符号化尺寸
  const auto packed_weight_sizes = packed_weight.sym_sizes();
  // 获取列维度数，通常是最后一个维度
  const auto col_dim = packed_weight_sizes.size() - 1;
  // 获取输入的列数
  const auto input_columns = packed_weight_sizes[col_dim];
  
  // 每行最后两个值用于存储每行的 FP32 scale 和 zero_point 值
  const auto output_columns = input_columns - 2 * sizeof(float);

  // 复制输出形状，并调整最后一个维度的大小为 output_columns
  auto output_shape = packed_weight_sizes.vec();
  output_shape[col_dim] = output_columns;

  // 创建一个与输出形状匹配的空 SymIntTensor，数据类型为 kFloat
  at::SymDimVector output_shape_vec(output_shape);
  return at::empty_symint(output_shape_vec, packed_weight.options().dtype(kFloat), packed_weight.suggest_memory_format());
}

// 辅助函数，用于解压带有 N 比特量化的权重 packed_weight
Tensor _qembeddingbag_nbit_unpack_helper(
    const Tensor& packed_weight,
    int BIT_RATE) {
  // 获取输入的行数和列数
  const auto input_rows = packed_weight.size(0);
  const auto input_columns = packed_weight.size(1);
  // 获取输入数据的指针，类型为 uint8_t
  const auto* input_data = packed_weight.const_data_ptr<uint8_t>();
  // 每字节的元素个数，根据 BIT_RATE 计算得出
  int NUM_ELEM_PER_BYTE = 8 / BIT_RATE;

  // 每行最后四个字节包含两个 fp16 的 scale 和 zero_point
  // 剩余的 input_columns 是原始行中值的数量
  std::vector<int64_t> output_dimensions = {
      input_rows,
      static_cast<std::int64_t>(input_columns - 2 * sizeof(at::Half)) *
          NUM_ELEM_PER_BYTE};

  // 创建一个空的输出 Tensor，形状为 output_dimensions，数据类型为 kFloat
  auto output = at::empty(
      output_dimensions,
      packed_weight.options().dtype(kFloat),
      packed_weight.suggest_memory_format());
  // 获取输出数据的指针，类型为 float
  float* output_data = output.data_ptr<float>();

  // 根据是否使用 FBGEMM 进行不同的解压方式
#ifdef USE_FBGEMM
  // 使用 FBGEMM 进行并行解压
  at::parallel_for(0, input_rows, 1, [&](int64_t start_idx, int64_t end_idx) {
    fbgemm::FusedNBitRowwiseQuantizedSBHalfToFloatOrHalf<float>(
        BIT_RATE,
        input_data + start_idx * input_columns,
        end_idx - start_idx,
        input_columns,
        output_data + start_idx * output_dimensions[1]);
  });
#else
  // 如果未使用 FBGEMM，则使用普通的解压方式
  auto output_columns = output_dimensions[1];
  for (auto row : c10::irange(input_rows)) {
    // 获取当前行的输出指针和输入指针
    float* output_row = output_data + row * output_columns;
    const std::uint8_t* input_row = input_data + row * input_columns;
    // 获取当前行的 scale 和 zero_point
    const at::Half* input_row_scale_zp = reinterpret_cast<const at::Half*>(
        input_row +
        (output_columns + NUM_ELEM_PER_BYTE - 1) / NUM_ELEM_PER_BYTE);
    float scale = input_row_scale_zp[0];
    float zero_point = input_row_scale_zp[1];

    // 遍历当前行的输出列
    for (const auto col : c10::irange(output_columns)) {
      // 获取当前列的量化值
      std::uint8_t quantized = input_row[col / NUM_ELEM_PER_BYTE];
      quantized >>= (col % NUM_ELEM_PER_BYTE) * BIT_RATE;
      quantized &= (1 << BIT_RATE) - 1;
      // 计算并存储解压后的值
      output_row[col] = scale * quantized + zero_point;
    } // output_columns
  } // input_rows
#endif // USE_FBGEMM

  return output;
}

// 用于反量化 qembeddingbag_4bit_prepack 操作的结果
// 输入预期首先有量化值，然后是两个字节的 fp16 scale 和 zero_offset
// 输出是仅包含值的矩阵，但是已解压
// 解压通过将每个值乘以其
// 使用 4 位量化率对 packed_weight 进行解压缩操作，返回解压后的 Tensor
Tensor qembeddingbag_4bit_unpack(const Tensor& packed_weight) {
  return _qembeddingbag_nbit_unpack_helper(packed_weight, 4 /*BIT_RATE*/);
}

// 使用 2 位量化率对 packed_weight 进行解压缩操作，返回解压后的 Tensor
// 输入的 packed_weight 首先包含量化值，然后是两个字节的 fp16 缩放因子和偏移量
// 输出是一个仅包含解压后值的矩阵
// 解压操作通过将每个值乘以其行的缩放因子和偏移量来执行
// 解压后的值因此可能不完全等于原始的未量化浮点值
Tensor qembeddingbag_2bit_unpack(const Tensor& packed_weight) {
  return _qembeddingbag_nbit_unpack_helper(packed_weight, 2 /*BIT_RATE*/);
}

// QEmbeddingUnpackWeights 类的静态方法，用于解压 EmbeddingPackedParamsBase 类型的 packed_weight
// 调用 packed_weight 对象的 unpack 方法进行解压操作，并返回解压后的 Tensor
class QEmbeddingUnpackWeights final {
 public:
  static at::Tensor run(
      const c10::intrusive_ptr<EmbeddingPackedParamsBase>& packed_weight) {
    return packed_weight->unpack();
  }
};

// 注册 quantized 库的 CPU 实现，注册各种量化嵌入操作的解压函数
TORCH_LIBRARY_IMPL(quantized, CPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("quantized::embedding_bag_byte_unpack"),
      qembeddingbag_byte_unpack);
  m.impl(
      TORCH_SELECTIVE_NAME("quantized::embedding_bag_4bit_unpack"),
      qembeddingbag_4bit_unpack);
  m.impl(
      TORCH_SELECTIVE_NAME("quantized::embedding_bag_2bit_unpack"),
      qembeddingbag_2bit_unpack);
}

// 注册 quantized 库的 CatchAll 实现，注册量化嵌入包的解压函数
// 使用 TorchBind 自定义类 QEmbeddingUnpackWeights::run 扩展支持 4 位量化张量
TORCH_LIBRARY_IMPL(quantized, CatchAll, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("quantized::embedding_bag_unpack"),
      TORCH_FN(QEmbeddingUnpackWeights::run));
}

// 注册 quantized 库的 Meta 实现，注册字节量化嵌入包的解压函数
m.impl(
    "quantized::embedding_bag_byte_unpack",
    qembeddingbag_byte_unpack_meta);
```