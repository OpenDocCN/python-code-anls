# `.\pytorch\aten\src\ATen\native\ao_sparse\quantized\cpu\qlinear_serialize.cpp`

```
/**
 * Includes the ATen library for PyTorch tensor operations.
 */
#include <ATen/ATen.h>

#ifdef USE_FBGEMM
/**
 * Includes FBGEMM utilities specific to sparse quantized operations.
 */
#include <ATen/native/ao_sparse/quantized/cpu/fbgemm_utils.h>
#endif

#ifdef USE_PYTORCH_QNNPACK
/**
 * Includes QNNPACK utilities specific to sparse quantized operations.
 */
#include <ATen/native/ao_sparse/quantized/cpu/qnnpack_utils.h>
#endif

/**
 * Defines the namespace ao::sparse for encapsulating sparse tensor operations.
 */
namespace ao {
namespace sparse {

namespace {
/**
 * Function template to wrap a vector-like structure into a PyTorch Tensor.
 * Copies data into the Tensor's own data pointer.
 * The template types allow compatibility with any vector-like structure having .data() and .size().
 * @tparam UNDERLYING_DTYPE The underlying data type of the vector elements.
 * @tparam T The type of the vector-like data structure.
 * @param vec The vector-like data structure to wrap.
 * @param dtype Scalar type of the resulting Tensor.
 * @return A PyTorch Tensor containing the data from the input vector.
 */
template <typename UNDERLYING_DTYPE, typename T>
at::Tensor wrap_vector(T& vec, c10::ScalarType dtype) {
  // Creates an empty Tensor on the CPU with the specified dtype and size matching the vector.
  at::Tensor t = at::empty(
      {static_cast<long>(vec.size())}, at::device(c10::kCPU).dtype(dtype));
  // Copies data from the vector to the Tensor's mutable data pointer.
  std::copy(
      vec.data(), vec.data() + vec.size(), t.mutable_data_ptr<UNDERLYING_DTYPE>());
  return t;
}

#ifdef USE_FBGEMM
/**
 * Packs a sparse matrix in Block Compressed Sparse Row (BCSR) format.
 * Adapted from FBGEMM's BCSRMatrix::pack method with modifications.
 * Does not include zero points, tiling, or row offset determination.
 * @param src Pointer to the input matrix data.
 * @param R Number of rows in the matrix.
 * @param C Number of columns in the matrix.
 * @param RB Number of rows per block.
 * @param CB Number of columns per block.
 * @param zero_points Pointer to zero points array (optional).
 * @param qscheme_per_tensor Flag indicating if quantization scheme is per tensor.
 * @return BCSR structure containing packed matrix data.
 */
ao::sparse::BCSR pack_bcsr(
    const int8_t* src,
    const int64_t R,
    const int64_t C,
    const int64_t RB,
    const int64_t CB,
    const int8_t* zero_points,
    const bool qscheme_per_tensor) {
  const size_t ld = C;
  std::vector<int32_t> rowBPtr;
  std::vector<int32_t> colBIdx;
  std::vector<int8_t> values;
  rowBPtr.push_back(0);
  int64_t nnzb = 0;
  int64_t rowBlocks = (R + RB - 1) / RB;
  for (int64_t i = 0; i < rowBlocks; ++i) {
    int64_t curCols = C;
    int64_t curColBlocks = (curCols + CB - 1) / CB;
    for (int64_t j = 0; j < curColBlocks; ++j) {
      // Check if the entire block is zero.
      bool isCurrentBlockNonZero = false;
      for (int64_t ib = 0; ib < RB; ++ib) {
        if (isCurrentBlockNonZero || (i * RB + ib) >= R) {
          break;
        }
        const int64_t curr_row = i * RB + ib;
        const int8_t curr_row_zero_point =
            qscheme_per_tensor ? zero_points[0] : zero_points[curr_row];
        for (int64_t jb = 0; jb < CB; ++jb) {
          if ((j * CB + jb) >= C) {
            continue;
          } else {
            if (src[curr_row * ld + j * CB + jb] != curr_row_zero_point) {
              isCurrentBlockNonZero = true;
              break;
            }
          }
        }
      }
      if (isCurrentBlockNonZero) {
        for (int64_t ib = 0; ib < RB; ++ib) {
          for (int64_t jb = 0; jb < CB; ++jb) {
            if ((i * RB + ib) >= R || (j * CB + jb) >= C) {
              values.push_back(0);
            } else {
              int8_t val = src[(i * RB + ib) * ld + j * CB + jb];
              values.push_back(val);
            }
          }
        }
        colBIdx.push_back(static_cast<int32_t>(j));
        nnzb++;
      }
    }
    # 将整数 nnzb 强制类型转换为 int32_t 类型，并将其添加到 rowBPtr 后面
    rowBPtr.push_back(static_cast<int32_t>(nnzb));
    # 返回一个 ao::sparse::BCSR 对象，该对象使用移动语义初始化，包括 values、rowBPtr 和 colBIdx
    return ao::sparse::BCSR(
        std::move(values), std::move(rowBPtr), std::move(colBIdx));
#ifdef USE_FBGEMM
} // 结束条件编译指令 USE_FBGEMM

} // 结束命名空间

#ifdef USE_FBGEMM

BCSRSerializationType PackedLinearWeight::serialize() {
  // 获取权重、行索引和列索引的非平铺形式；
  // 解压缩平铺的BCSR格式，然后以非平铺形式打包
  std::vector<int8_t> dense_weight_values = std::vector<int8_t>(w->R * w->C);
  w->unpack(dense_weight_values.data());

  const bool qscheme_per_tensor = (q_scheme == c10::kPerTensorAffine);
  at::Tensor zero_points = wrap_vector<int8_t>(w_zp, c10::kChar);

  ao::sparse::BCSR untiled_bcsr = pack_bcsr(
      dense_weight_values.data(),
      w->R,
      w->C,
      w->RB,
      w->CB,
      zero_points.data_ptr<int8_t>(),
      qscheme_per_tensor);

  std::vector<int8_t>& packed_weight_values = std::get<0>(untiled_bcsr);
  // 对每个权重值加上 128。这种序列化格式最适合于QNNPack以减少内存占用

  at::Tensor weight_values = at::empty(
      {static_cast<long>(packed_weight_values.size())},
      at::device(c10::kCPU).dtype(c10::kByte));
  std::transform(
      packed_weight_values.begin(),
      packed_weight_values.end(),
      weight_values.mutable_data_ptr<uint8_t>(),
      [](int8_t v) {
        return static_cast<uint8_t>(static_cast<int16_t>(v) + 128);
      });

  return BCSRSerializationType(
      SPARSE_LINEAR_PACKED_PARAM_SERIALIZATION_VERSION,
      bias_,
      out_features_block_size_,
      in_features_block_size_,
      wrap_vector<float>(w_scale, c10::kFloat),
      // 从int32_t缩减到int8_t；这是安全的，因为qint8零点被限制在int_8的范围内
      std::move(zero_points),
      qscheme_per_tensor,
      wrap_vector<int>(
          std::get<1>(untiled_bcsr), c10::kInt), // 行块索引
      wrap_vector<int>(
          std::get<2>(untiled_bcsr), c10::kInt), // 列块索引
      std::move(weight_values),
      w->R,
      w->C);
}

#endif // 结束条件编译指令 USE_FBGEMM

#ifdef USE_PYTORCH_QNNPACK

BCSRSerializationType PackedLinearWeightQnnp::serialize() {
  at::Tensor w_scales_compact;
  at::Tensor w_zero_points_compact;
  const float* w_scales_data_ptr = w_scales_.const_data_ptr<float>();
  std::function<int8_t(uint8_t)> subtract_128 = [](uint8_t v) {
    return static_cast<int8_t>(static_cast<int16_t>(v) - 128);
  };

  if (q_scheme_ == at::kPerTensorAffine) {
    w_scales_compact = at::empty({1}, at::device(c10::kCPU).dtype(c10::kFloat));
    w_zero_points_compact =
        at::empty({1}, at::device(c10::kCPU).dtype(c10::kChar));

    w_scales_compact.mutable_data_ptr<float>()[0] = w_scales_data_ptr[0];
    w_zero_points_compact.mutable_data_ptr<int8_t>()[0] =
        static_cast<int8_t>(static_cast<int16_t>(w_zero_points_[0]) - 128);
  } else if (q_scheme_ == at::kPerChannelAffine) {
    w_scales_compact =
        at::empty({output_channels_}, at::device(c10::kCPU).dtype(c10::kFloat));
    w_zero_points_compact =
        at::empty({output_channels_}, at::device(c10::kCPU).dtype(c10::kChar));
    std::copy(
        w_scales_data_ptr,
        w_scales_data_ptr +
            output_channels_, // 不考虑填充，只复制到输出通道数目的数据
        w_scales_compact.mutable_data_ptr<float>());

    // 从每个零点中减去 128，以撤销预打包时的加法操作
    std::transform(
        w_zero_points_.begin(),
        w_zero_points_.begin() +
            output_channels_, // 不考虑填充，只转换输出通道数目的数据
        w_zero_points_compact.mutable_data_ptr<int8_t>(),
        std::move(subtract_128));
  } else {
    TORCH_CHECK(false, "Unsupported quantization scheme."); // 抛出错误，不支持的量化方案
  }

  at::Tensor wrapped_row_values;
  at::Tensor wrapped_col_indices;

  const uint32_t max_index = bcsr_matrix_->max_index();

  if (max_index <= std::numeric_limits<uint8_t>::max()) {
    // 将数据类型从 uint8_t 转换为 int8_t
    wrapped_row_values = QNNPACK_BCSRMATRIX_DISPATCH_INDICES_DTYPE(
        bcsr_matrix_,
        { return wrap_vector<int8_t>(typed_bcsr->row_values, c10::kChar); });
    wrapped_col_indices = QNNPACK_BCSRMATRIX_DISPATCH_INDICES_DTYPE(
        bcsr_matrix_,
        { return wrap_vector<int8_t>(typed_bcsr->col_indices, c10::kChar); });
  } else if (max_index <= std::numeric_limits<uint16_t>::max()) {
    // 将数据类型从 uint16_t 转换为 int16_t
    wrapped_row_values = QNNPACK_BCSRMATRIX_DISPATCH_INDICES_DTYPE(
        bcsr_matrix_,
        { return wrap_vector<int16_t>(typed_bcsr->row_values, c10::kShort); });
    wrapped_col_indices = QNNPACK_BCSRMATRIX_DISPATCH_INDICES_DTYPE(
        bcsr_matrix_,
        { return wrap_vector<int16_t>(typed_bcsr->col_indices, c10::kShort); });
  } else {
    // 将数据类型从 uint32_t 转换为 int32_t
    wrapped_row_values = QNNPACK_BCSRMATRIX_DISPATCH_INDICES_DTYPE(
        bcsr_matrix_,
        { return wrap_vector<int>(typed_bcsr->row_values, c10::kInt); });
    wrapped_col_indices = QNNPACK_BCSRMATRIX_DISPATCH_INDICES_DTYPE(
        bcsr_matrix_,
        { return wrap_vector<int>(typed_bcsr->col_indices, c10::kInt); });
  }

  // 返回稀疏线性紧凑化的序列化类型对象
  return BCSRSerializationType(
      SPARSE_LINEAR_PACKED_PARAM_SERIALIZATION_VERSION,
      orig_bias_,
      out_features_block_size_,
      in_features_block_size_,
      std::move(w_scales_compact),
      std::move(w_zero_points_compact),
      (q_scheme_ == c10::kPerTensorAffine),
      wrapped_row_values,
      wrapped_col_indices,
      wrap_vector<uint8_t>(bcsr_matrix_->values, c10::kByte),
      output_channels_,
      input_channels_);
}

#endif // USE_PYTORCH_QNNPACK


注释：


// 关闭 namespace ao 和 namespace sparse 嵌套命名空间的定义
}

// 结束条件编译指令，检查是否定义了 USE_PYTORCH_QNNPACK 宏，用于条件编译
#endif // USE_PYTORCH_QNNPACK


这段代码是 C++ 中的命名空间闭合和条件编译的结构。在这里：

- `}` 表示结束了 `namespace sparse` 命名空间的定义。
- `#endif // USE_PYTORCH_QNNPACK` 是一个预处理器指令，用于结束 `#ifdef` 或 `#if` 条件编译块，检查是否定义了 `USE_PYTORCH_QNNPACK` 宏。

这种结构通常用于根据预定义的宏条件编译不同的代码块，以便在不同的编译配置下执行不同的代码路径。
```