# `.\pytorch\aten\src\ATen\native\quantized\cpu\qembeddingbag.cpp`

```py
// 微调实现，当 FBGEMM 不可用时备用方案
template <
    typename IndexType,  // 索引类型
    typename OffsetType, // 偏移类型
    int BIT_RATE,        // 比特率
    int NUM_ELEM_PER_BYTE> // 每字节元素数量
// 嵌入查找的后备实现，用于量化嵌入查询
at::Tensor& embedding_lookup_fallback_impl(
    const at::Tensor& weight,                      // 权重张量
    const at::Tensor& indices,                     // 索引张量
    const at::Tensor& offsets,                     // 偏移量张量
    const std::optional<at::Tensor>& per_sample_weights_, // 每样本权重（可选）
    const std::optional<at::Tensor>& compressed_indices_mapping, // 压缩索引映射（可选）
    at::Tensor& output,                            // 输出张量
    const int64_t block_size,                      // 块大小
    const int64_t output_size,                     // 输出大小
    bool include_last_offset,                      // 是否包括最后一个偏移
    bool pruned) {                                 // 是否剪枝
  auto* output_data = output.data_ptr<float>();     // 输出数据指针
  const auto weight_data = weight.data_ptr<uint8_t>(); // 权重数据指针
  const auto indices_data = indices.data_ptr<IndexType>(); // 索引数据指针
  int32_t* compressed_indices_mapping_data = nullptr; // 压缩索引映射数据指针
  const auto weight_sizes = weight.sizes();         // 权重张量大小
  const int64_t N = weight_sizes[0];                // 权重张量的第一维大小
  const int64_t weight_size = weight_sizes[1];      // 权重张量的第二维大小
  const int index_size = indices.numel();           // 索引张量的元素数量

  auto accessor = offsets.accessor<OffsetType, 1>(); // 偏移量访问器
  std::vector<OffsetType> lengths_data;             // 长度数据向量

  int64_t lower = accessor[0];                      // 初始化下限为第一个偏移
  for (const auto i : c10::irange(1, offsets.numel())) {
    lengths_data.push_back(accessor[i] - lower);    // 计算每段长度
    lower = accessor[i];                            // 更新下限
  }
  if (!include_last_offset) {
    lengths_data.push_back(indices.numel() - lower); // 如果不包括最后一个偏移，则加入最后一段长度
  }

  int64_t current = 0;                              // 当前索引位置初始化为0
  float* per_sample_weights_data;                    // 每样本权重数据指针
  if (per_sample_weights_.has_value()) {
    per_sample_weights_data = per_sample_weights_.value().data_ptr<float>(); // 获取每样本权重数据指针
  }
  for (const auto m : c10::irange(output_size)) {    // 遍历输出大小
    memset(output_data, 0, block_size * sizeof(float)); // 初始化输出数据为0
    TORCH_CHECK(
        current + lengths_data[m] <= index_size,     // 检查长度数据是否小于等于索引张量大小
        "Expect the lengths data to be less than indices size");
    // 遍历当前长度数据所指定的次数
    for (int i = 0; i < lengths_data[m]; ++i, ++current) {
      // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
      // 定义64位整数变量idx
      int64_t idx;
      // 如果未经修剪
      if (!pruned) {
        // 从indices_data中获取当前索引值
        idx = indices_data[current];
        // 检查索引是否在有效范围内
        TORCH_CHECK((idx >= 0 && idx < N), "Invalid indices data");
      } else {
        // 从indices_data获取未压缩的索引值
        int64_t uncompressed_idx = indices_data[current];
        // 获取压缩索引映射的大小
        int compressed_index_size = compressed_indices_mapping.value().numel();
        // 获取压缩索引映射的数据指针
        compressed_indices_mapping_data =
            compressed_indices_mapping.value().data_ptr<int32_t>();
        // 检查未压缩索引是否在有效范围内
        TORCH_CHECK(
            uncompressed_idx >= 0 && uncompressed_idx < compressed_index_size,
            "Invalid indices data for Sparse Op.")
        // 从压缩索引映射中获取对应的索引值
        idx = compressed_indices_mapping_data[uncompressed_idx];
        // 如果索引值为-1，则跳过当前循环
        if (idx == -1) {
          continue;
        }
      }

      // 默认权重值为1.0
      float weight_val = 1.0f;
      // 如果存在每个样本的权重值
      if (per_sample_weights_.has_value()) {
        // 获取当前权重值
        weight_val = per_sample_weights_data[current];
      }
      // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
      // 定义浮点数变量scale和bias
      float scale, bias;
      // 如果BIT_RATE为8
      if (BIT_RATE == 8) {
        // 获取权重数据指针的指定位置，以获取scale和bias
        const uint8_t* scale_bias =
            weight_data + (idx + 1) * weight_size - 2 * sizeof(float);
        uint32_t scale_val_int32 = 0;
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
        // 如果系统的字节顺序为小端（低位字节存储在低地址），将 scale_bias 转换为一个 32 位整数
        scale_val_int32 = scale_val_int32 |
          (scale_bias[0]) |
          (scale_bias[1] << 8) |
          (scale_bias[2] << 16) |
          (scale_bias[3] << 24);
#elif __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
        // 如果系统的字节顺序为大端（高位字节存储在低地址），将 scale_bias 转换为一个 32 位整数
        scale_val_int32 = scale_val_int32 |
          (scale_bias[3]) |
          (scale_bias[2] << 8) |
          (scale_bias[1] << 16) |
          (scale_bias[0] << 24);
#else
#error Unexpected or undefined __BYTE_ORDER__
#endif

        // 将 32 位整数的值解释为 float 类型，得到 scale_val
        float scale_val = (reinterpret_cast<float*>(&scale_val_int32))[0];

        uint32_t bias_val_int32 = 0;
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
        // 如果系统的字节顺序为小端，将 scale_bias 转换为一个 32 位整数
        bias_val_int32 = bias_val_int32 |
          (scale_bias[4]) |
          (scale_bias[5] << 8) |
          (scale_bias[6] << 16) |
          (scale_bias[7] << 24);
#elif __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
        // 如果系统的字节顺序为大端，将 scale_bias 转换为一个 32 位整数
        bias_val_int32 = bias_val_int32 |
          (scale_bias[7]) |
          (scale_bias[6] << 8) |
          (scale_bias[5] << 16) |
          (scale_bias[4] << 24);
#else
#error Unexpected or undefined __BYTE_ORDER__
#endif

        // 将 32 位整数的值解释为 float 类型，得到 bias_val
        float bias_val = (reinterpret_cast<float*>(&bias_val_int32))[0];

        // 计算权重值乘以 scale_val 和 bias_val，并赋给 scale 和 bias
        scale = weight_val * scale_val;
        bias = weight_val * bias_val;
      } else {
        // 如果权重数据是半精度浮点数
        const uint8_t* scale_bias =
            weight_data + (idx + 1) * weight_size - 2 * sizeof(at::Half);
        uint16_t scale_val_int16 = 0;
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
        // 如果系统的字节顺序为小端，将 scale_bias 转换为一个 16 位整数
        scale_val_int16 = scale_val_int16 |
          (scale_bias[0]) |
          (scale_bias[1] << 8);
#elif __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
        // 如果系统的字节顺序为大端，将 scale_bias 转换为一个 16 位整数
        scale_val_int16 = scale_val_int16 |
          (scale_bias[1]) |
          (scale_bias[0] << 8);
#else
#error Unexpected or undefined __BYTE_ORDER__
#endif

        // 将 16 位整数的值解释为半精度浮点数，得到 scale_val
        at::Half scale_val = (reinterpret_cast<at::Half*>(&scale_val_int16))[0];

        uint16_t bias_val_int16 = 0;
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
        // 如果系统的字节顺序为小端，将 scale_bias 转换为一个 16 位整数
        bias_val_int16 = bias_val_int16 |
          (scale_bias[2]) |
          (scale_bias[3] << 8);
#elif __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
        // 如果系统的字节顺序为大端，将 scale_bias 转换为一个 16 位整数
        bias_val_int16 = bias_val_int16 |
          (scale_bias[3]) |
          (scale_bias[2] << 8);
#else
#error Unexpected or undefined __BYTE_ORDER__
#endif

        // 将 16 位整数的值解释为半精度浮点数，得到 bias_val
        at::Half bias_val = (reinterpret_cast<at::Half*>(&bias_val_int16))[0];

        // 计算权重值乘以 scale_val 和 bias_val，并赋给 scale 和 bias
        scale = weight_val * scale_val;
        bias = weight_val * bias_val;
      }

      for (const auto j : c10::irange(block_size)) {
        // 从权重数据中获取量化的数据
        uint8_t quantized =
            weight_data[idx * weight_size + j / NUM_ELEM_PER_BYTE];
        quantized >>= (j % NUM_ELEM_PER_BYTE) * BIT_RATE;
        quantized &= (1 << BIT_RATE) - 1;

        // 使用 FMA（fused multiply-add）计算输出数据中的结果
        output_data[j] = fma(scale, quantized, output_data[j] + bias);
      }
    } // for each i
    output_data += block_size;
  } // for each m
  return output;
}

namespace {
template <typename IndexType, typename OffsetType>
void fbgemm_spmdm_report_error_(
    int64_t output_size,
    int index_size,
    int64_t N,
    // 遍历每个输出项，使用其偏移量和索引检查边界
    for (const auto m : c10::irange(output_size)) {
        // 遍历当前输出项的索引偏移范围
        for (OffsetType i = offsets[m]; i < offsets[m + 1]; ++i) {
            // 确保索引 i 在有效范围内
            TORCH_CHECK(i < index_size);
            // 获取当前索引值
            IndexType idx = indices[i];
            // 确保索引值在合法范围内
            TORCH_CHECK(
                0 <= idx && idx < N,
                "Index ",
                i,
                " is out of bounds: ",
                idx,
                ", range 0 to ",
                N);
        }
    }
    
    // 检查最后一个偏移量是否与索引总数匹配，以验证输入的正确性
    TORCH_CHECK(
        offsets[output_size] == index_size,
        "Your input seems to be incorrect: the last offset value should be "
        "the size of the indices tensor, but it appears not.");
} // namespace
    // 生成 fbgemm 内核函数，用于稀疏矩阵乘法操作
    auto kernel = fbgemm::GenerateEmbeddingSpMDMNBit<IndexType, OffsetType>(
        /*bit rate=*/bit_width,                            // 指定比特率
        /*block size=*/block_size,                         // 指定块大小
        /*has weights=*/per_sample_weights_.has_value(),   // 是否有每样本权重
        /*normalize_by_lengths=*/false,                    // 是否按长度归一化
        /*prefetch distance=*/prefetch_distance,           // 预取距离
        /*is_weight_positional=*/false,                    // 是否是权重位置相关的
        /*use_offsets=*/true);                             // 是否使用偏移量

    // 调用生成的内核函数进行计算
    bool success = kernel(
        /*output_size=*/output_size,                       // 输出尺寸
        /*index_size=*/index_size,                         // 索引尺寸
        /*data_size=*/N,                                  // 数据大小
        /*input=*/weight_data,                            // 输入权重数据
        /*indices=*/indices_data,                         // 输入索引数据
        /*offsets=*/offsets_data,                         // 输入偏移量数据
        /*weights=*/per_sample_weights_.has_value()       // 可选的每样本权重数据
            ? per_sample_weights_.value().data_ptr<float>()
            : nullptr,
        /*output=*/output_data);                          // 输出数据

    // 如果计算失败，则报告错误
    if (!success) {
      fbgemm_spmdm_report_error_(
          output_size, index_size, N, offsets_data, indices_data);
    }
  } else {
    // 生成稀疏行存储格式的 fbgemm 内核函数
    auto kernel =
        fbgemm::GenerateEmbeddingSpMDMNBitRowWiseSparse<IndexType, OffsetType>(
            /*bit rate=*/bit_width,                        // 指定比特率
            /*block_size=*/block_size,                     // 指定块大小
            /*has weights=*/per_sample_weights_.has_value(), // 是否有每样本权重
            /*normalize_by_lengths=*/false,                // 是否按长度归一化
            /*prefetch distance=*/prefetch_distance,       // 预取距离
            /*is_weight_positional*/ false,                // 是否是权重位置相关的
            /*use_offsets*/ true);                         // 是否使用偏移量

    // 调用生成的内核函数进行计算
    bool success = kernel(
        /*output_size=*/output_size,                       // 输出尺寸
        /*index_size=*/index_size,                         // 索引尺寸
        /*data_size=*/compressed_index_size,               // 压缩索引大小
        /*input=*/weight_data,                            // 输入权重数据
        /*indices=*/indices_data,                         // 输入索引数据
        /*offsets=*/offsets_data,                         // 输入偏移量数据
        /*weights=*/per_sample_weights_.has_value()       // 可选的每样本权重数据
            ? per_sample_weights_.value().data_ptr<float>()
            : nullptr,
        /*output=*/output_data,                           // 输出数据
        /*compressed_indices_table=*/compressed_indices_mapping_data); // 压缩索引映射表

    // 如果计算失败，则报告错误
    if (!success) {
      fbgemm_spmdm_report_error_(
          output_size,
          index_size,
          compressed_index_size,
          offsets_data,
          indices_data);
    }
  }
  // 返回计算结果输出
  return output;
#else
  // 如果 bit_width 等于 4，则调用对应的 embedding_lookup_fallback_impl 函数
  if (bit_width == 4) {
    return embedding_lookup_fallback_impl<IndexType, OffsetType, 4, 2>(
      weight,
      indices,
      offsets,
      per_sample_weights_,
      compressed_indices_mapping,
      output,
      D,
      output_size,
      include_last_offset,
      (pruned_weights && !fallback_to_no_sparse));
  }
  // 否则，如果 bit_width 等于 2，则调用对应的 embedding_lookup_fallback_impl 函数
  // 这里假设 bit_width 只可能是 2 或 4，因为没有其他情况
  return embedding_lookup_fallback_impl<IndexType, OffsetType, 2, 4>(
    weight,
    indices,
    offsets,
    per_sample_weights_,
    compressed_indices_mapping,
    output,
    D,
    output_size,
    include_last_offset,
    (pruned_weights && !fallback_to_no_sparse));
#endif
}

template <typename IndexType, typename OffsetType>
// 在 weight 的类型检查通过后，执行 byte 类型的 embedding_bag 操作
at::Tensor& embedding_bag_byte_impl(
    at::Tensor& output,
    const at::Tensor& weight,
    const at::Tensor& indices,
    const at::Tensor& offsets,
    bool pruned_weights,
    const std::optional<at::Tensor>& per_sample_weights_,
    const std::optional<at::Tensor>& compressed_indices_mapping,
    bool include_last_offset,
    bool is_embedding_op) {
  // 检查 weight 张量的数据类型是否为 Byte 类型
  TORCH_CHECK(weight.scalar_type() == at::kByte);
  // 检查 weight 张量的维度是否为 2
  TORCH_CHECK(weight.dim() == 2);
  // 检查 offsets 张量的维度是否为 1
  TORCH_CHECK(offsets.dim() == 1);
  // 获取 offsets 张量的数据指针
  auto offsets_data = offsets.data_ptr<OffsetType>();

  // 获取压缩索引，用于稀疏权重处理
  int32_t* compressed_indices_mapping_data = nullptr;
  int compressed_index_size = 0;
  bool fallback_to_no_sparse = false;
  if (pruned_weights) {
    // 获取压缩索引映射的大小
    compressed_index_size = compressed_indices_mapping.value().numel();
    // 获取压缩索引映射的数据指针
    compressed_indices_mapping_data =
        compressed_indices_mapping.value().data_ptr<int32_t>();

    // 如果压缩索引映射为 [0]，表示应该回退到非稀疏嵌入查找内核
    if ((compressed_index_size == 1 &&
         compressed_indices_mapping_data[0] == 0)) {
      fallback_to_no_sparse = true;
    }
  }

  // 获取 weight 张量的大小信息
  const auto weight_sizes = weight.sizes();
  // 计算 D 的值，等于 weight 的第二维大小减去 8，以考虑到 scale 和 bias
  const int64_t D = weight_sizes[1] - 8; // NB: -8 to account for scale and bias
  // 计算 M 的值，即 offsets 张量的第一维大小
  const int64_t M = offsets.sizes()[0];

  // 初始化 output_size 为 M - 1
  int64_t output_size = M - 1;
  // 定义用于存储包含最后一个偏移量的 offsets 的向量
  std::vector<OffsetType> offsets_include_last_val;

  // 如果不包含最后一个偏移量
  if (!include_last_offset) {
    // 更新 output_size 为 M
    output_size = M;
    // 调整 offsets_include_last_val 的大小为 M+1
    offsets_include_last_val.resize(M + 1);
    // 避免在 offsets 张量为空时引发的 ASAN 违规（空指针作为参数 2）
    if (M > 0) {
      // 将 offsets_data 复制到 offsets_include_last_val 中
      std::memcpy(
          offsets_include_last_val.data(),
          offsets_data,
          sizeof(OffsetType) * M);
    }
    // 设置 offsets_data 指向 offsets_include_last_val 的数据
    offsets_include_last_val[M] = indices.numel();
    offsets_data = offsets_include_last_val.data();
  }
  {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
    // 定义一个形状为 [3] 的数组 shape_arr
    std::array<int64_t, 3> shape_arr;
    c10::IntArrayRef shape;
    // 如果 indices 张量的维度为 2 并且是嵌入操作
    if (indices.dim() == 2 && is_embedding_op) {
      // 获取 indices 张量的大小信息
      const auto indices_sizes = indices.sizes();
      // 设置 shape_arr 的第一个维度为 indices 的第一个维度大小
      shape_arr[0] = indices_sizes[0];
      // 设置 shape_arr 的第二个维度为 indices 的第二个维度大小
      shape_arr[1] = indices_sizes[1];
      // 设置 shape_arr 的第三个维度为 D
      shape_arr[2] = D;
      // 将 shape_arr 赋值给 shape
      shape = shape_arr;
    } else {
      // 如果不是第一次调整大小，则使用默认的输出大小和维度 D
      shape_arr[0] = output_size;
      shape_arr[1] = D;
      // 创建一个大小为 2 的 IntArrayRef 对象，指向 shape_arr 数组
      shape = c10::IntArrayRef(&shape_arr[0], 2);
    }
    // 调用 ATen 库中的原生 resize_ 函数，重新设置 output 的形状为 shape 所指定的大小
    at::native::resize_(output, shape, c10::nullopt);
#ifdef USE_FBGEMM
  // 获取权重大小的第一个维度 N
  const int64_t N = weight_sizes[0];
  // 获取权重数据的指针
  const auto weight_data = weight.data_ptr<uint8_t>();
  // 获取索引数据的指针
  const auto indices_data = indices.data_ptr<IndexType>();
  // 获取输出数据的指针
  auto* output_data = output.data_ptr<float>();
  // 获取索引数据的元素个数
  const int index_size = indices.numel();

  // 如果未剪枝权重或者需要回退到非稀疏模式
  if (!pruned_weights || fallback_to_no_sparse) {
    // 创建一个稠密数据的 SpMDM 内核
    auto kernel_i8 =
        fbgemm::GenerateEmbeddingSpMDM<uint8_t, IndexType, OffsetType, /*OutType=*/float, /*TRHEAD_LOCAL=*/true>(
            /*block_size=*/D,
            /*has_weight=*/per_sample_weights_.has_value(),
            /*normalize_by_lengths=*/false,
            /*prefetch=*/16, // NOLINT(cppcoreguidelines-avoid-magic-numbers)
            /*is_weight_positional=*/false,
            /*use_offsets=*/true);

    // 并行处理每个输出数据的索引范围
    at::parallel_for(
        0, output_size, 1, [&](int64_t start_idx, int64_t end_idx) {
          // 调用稠密数据的 SpMDM 内核
          bool success = kernel_i8(
              /*output_size=*/end_idx - start_idx,
              /*index_size=*/offsets_data[end_idx] - offsets_data[start_idx],
              /*data_size=*/N,
              /*input=*/weight_data,
              /*indices=*/indices_data + offsets_data[start_idx],
              /*offsets_or_lengths=*/offsets_data + start_idx,
              /*weights=*/
              per_sample_weights_
                  ? per_sample_weights_.value().const_data_ptr<float>() +
                      offsets_data[start_idx]
                  : nullptr,
              /*out=*/output_data + start_idx * D);

          // 如果内核执行失败，则报告错误
          if (!success) {
            fbgemm_spmdm_report_error_(
                end_idx - start_idx,
                offsets_data[end_idx] - offsets_data[start_idx],
                N,
                offsets_data + start_idx,
                indices_data + offsets_data[start_idx]);
          }
        });
  } else {
    // 如果存在剪枝权重
    // 创建一个稀疏数据的 SpMDMRowWise 内核
    auto kernel_i8_sparse = fbgemm::
        GenerateEmbeddingSpMDMRowWiseSparse<uint8_t, IndexType, OffsetType>(
            /*block_size=*/D,
            /*has_weight=*/per_sample_weights_.has_value(),
            /*normalize_by_lengths=*/false,
            /*prefetch=*/16, // NOLINT(cppcoreguidelines-avoid-magic-numbers)
            /*is_weight_positional=*/false,
            /*use_offsets=*/true);

    // 调用稀疏数据的 SpMDMRowWise 内核
    auto success = kernel_i8_sparse(
        /*output_size=*/output_size,
        /*index_size=*/index_size,
        /*data_size=*/compressed_index_size,
        /*input=*/weight_data,
        /*indices=*/indices_data,
        /*offsets=*/offsets_data,
        /*weights=*/
        per_sample_weights_.has_value()
            ? per_sample_weights_.value().data_ptr<float>()
            : nullptr,
        /*output=*/output_data,
        /*compressed_indices_table=*/compressed_indices_mapping_data);
    
    // 如果内核执行失败，则报告错误
    if (!success) {
      fbgemm_spmdm_report_error_(
          output_size,
          index_size,
          compressed_index_size,
          offsets_data,
          indices_data);
    }
  }
  // 返回计算结果的输出数据
  return output;
#endif
#else
  return embedding_lookup_fallback_impl<IndexType, OffsetType, 8, 1>(
      weight,
      indices,
      offsets,
      per_sample_weights_,
      compressed_indices_mapping,
      output,
      D,
      output_size,
      include_last_offset,
      (pruned_weights && !fallback_to_no_sparse));
#endif
}


// 如果不满足上述两个条件，则返回使用备用函数进行查找的结果
#else
  return embedding_lookup_fallback_impl<IndexType, OffsetType, 8, 1>(
      weight,  // 权重张量
      indices,  // 索引张量
      offsets,  // 偏移张量
      per_sample_weights_,  // 每样本权重（可选）
      compressed_indices_mapping,  // 压缩索引映射（可选）
      output,  // 输出张量
      D,  // 维度 D
      output_size,  // 输出大小
      include_last_offset,  // 是否包含最后一个偏移量
      (pruned_weights && !fallback_to_no_sparse));  // 是否剪枝权重且不回退到非稀疏模式
#endif
}

at::Tensor& embedding_bag_byte_helper(
    at::Tensor& output,
    const at::Tensor& weight,
    const at::Tensor& indices,
    const std::optional<at::Tensor>& offsets_in,
    bool pruned_weights,
    const std::optional<at::Tensor>& per_sample_weights_,
    const std::optional<at::Tensor>& compressed_indices_mapping,
    bool include_last_offset,
    bool is_embedding_op) {
  c10::MaybeOwned<at::Tensor> offsets;
  TORCH_CHECK(
      indices.dim() == 1 || indices.dim() == 2,
      "qembedding/qembedding_bag operator supports 1 or 2d indices, got ",
      indices.dim());
  // For embedding_bag operator with 2D indices, we set the offsets explicitly
  // here.
  if (indices.dim() == 2 && !is_embedding_op) {
    TORCH_CHECK(
        !offsets_in.has_value(),
        "embedding_bag_byte operator: input is 2D, then offsets has to be None, as input is treated is a mini-batch of fixed length sequences.");

    // 创建一个从 0 到 indices.numel() 的整数张量，步长为 indices.sizes()[1]
    offsets = c10::MaybeOwned<at::Tensor>::owned(at::arange(0, indices.numel(), indices.sizes()[1], indices.scalar_type()));
  } else {
    TORCH_CHECK(
        offsets_in.has_value(),
        "embedding_bag_byte expects offsets to be set for 1D indices.");
    // 如果输入的 offsets_in 不为空，则使用它的引用
    offsets = c10::MaybeOwned<at::Tensor>::borrowed(offsets_in.value());
  }

  TORCH_CHECK(
      indices.scalar_type() == at::kInt || indices.scalar_type() == at::kLong,
      "Expect 32 or 64 bit indices, but found ",
      indices.scalar_type(),
      " instead.");
  TORCH_CHECK(
      offsets->scalar_type() == at::kInt || offsets->scalar_type() == at::kLong,
      "Expect 32 or 64 bit offsets, but found ",
      offsets->scalar_type(),
      " instead.");
  TORCH_CHECK(
      weight.is_contiguous() && indices.is_contiguous() &&
          offsets->is_contiguous(),
      "Expect weight, indices, and offsets to be contiguous.");

  // 使用帮助函数支持不同类型组合，避免额外的性能开销
  if (indices.scalar_type() == at::kInt && offsets->scalar_type() == at::kInt) {
    return embedding_bag_byte_impl<int, int>(
        output,
        weight,
        indices,
        *offsets,
        pruned_weights,
        per_sample_weights_,
        compressed_indices_mapping,
        include_last_offset,
        is_embedding_op);
  } else if (
      indices.scalar_type() == at::kInt && offsets->scalar_type() == at::kLong) {
    # 如果 indices 和 offsets 的数据类型均为 int，则调用模板函数 embedding_bag_byte_impl<int, int64_t>
    if (
        indices.scalar_type() == at::kInt and offsets->scalar_type() == at::kInt
    ) {
        # 调用 embedding_bag_byte_impl 模板函数，使用 int 类型的 indices 和 int64_t 类型的 offsets
        return embedding_bag_byte_impl<int, int64_t>(
            output,
            weight,
            indices,
            *offsets,
            pruned_weights,
            per_sample_weights_,
            compressed_indices_mapping,
            include_last_offset,
            is_embedding_op);
    } 
    
    # 如果 indices 的数据类型为 int64_t，offsets 的数据类型为 int，则调用模板函数 embedding_bag_byte_impl<int64_t, int>
    else if (
        indices.scalar_type() == at::kLong and offsets->scalar_type() == at::kInt
    ) {
        # 调用 embedding_bag_byte_impl 模板函数，使用 int64_t 类型的 indices 和 int 类型的 offsets
        return embedding_bag_byte_impl<int64_t, int>(
            output,
            weight,
            indices,
            *offsets,
            pruned_weights,
            per_sample_weights_,
            compressed_indices_mapping,
            include_last_offset,
            is_embedding_op);
    }
    
    # 如果以上两个条件都不满足，则默认情况下调用 embedding_bag_byte_impl<int64_t, int64_t>
    # 这是基于上面的 TORCH_CHECK 默认情况的返回
    # 使用 int64_t 类型的 indices 和 int64_t 类型的 offsets
    return embedding_bag_byte_impl<int64_t, int64_t>(
        output,
        weight,
        indices,
        *offsets,
        pruned_weights,
        per_sample_weights_,
        compressed_indices_mapping,
        include_last_offset,
        is_embedding_op);
  }
  
  // 辅助函数，用于处理嵌入袋操作中的 n-bit 嵌入计算，支持不同类型的组合
  at::Tensor& _embedding_bag_nbit_helper(
      at::Tensor& output,
      const at::Tensor& weight,
      const int bit_width,
      const at::Tensor& indices,
      const std::optional<at::Tensor>& offsets_in,
      bool pruned_weights,
      const std::optional<at::Tensor>& per_sample_weights_,
      const std::optional<at::Tensor>& compressed_indices_mapping,
      bool include_last_offset,
      bool is_embedding_op) {
    // offsets 变量的类型可能拥有或借用一个张量
    c10::MaybeOwned<at::Tensor> offsets;
    
    // 检查 bit_width 是否为有效值，仅支持 2 或 4
    TORCH_CHECK(
        bit_width == 4 || bit_width == 2,
        "qembedding/qembedding_bag operator supports bit_width 2 or 4, got ",
        bit_width);
    
    // 检查 indices 张量的维度是否为 1 或 2
    TORCH_CHECK(
        indices.dim() == 1 || indices.dim() == 2,
        "qembedding/qembedding_bag operator supports 1 or 2d indices, got ",
        indices.dim());

    // 对于具有 2D indices 的 embedding_bag 操作，需要在此显式设置 offsets
    if (indices.dim() == 2 && !is_embedding_op) {
      // 检查 offsets_in 是否为空值，对于 2D 输入，offsets 必须为 None
      TORCH_CHECK(
          !offsets_in.has_value(),
          "embedding_bag_4bit/embedding_bag_2bit operator: input is 2D, then offsets has to be None, as input is treated is a mini-batch of fixed length sequences.");

      // 创建一个范围内的张量，用于表示每个序列的起始索引
      offsets = c10::MaybeOwned<at::Tensor>::owned(at::arange(
          0, indices.numel(), indices.sizes()[1], indices.scalar_type()));
    } else {
      // 对于 1D indices，使用传入的 offsets_in 张量
      TORCH_CHECK(
          offsets_in.has_value(),
          "embedding_bag_4bit/embedding_bag_2bit operator expects offsets to be set for 1D indices.");
      offsets = c10::MaybeOwned<at::Tensor>::borrowed(offsets_in.value());
    }

    // 检查 indices 张量的数据类型是否为 int 或 long
    TORCH_CHECK(
        indices.scalar_type() == at::kInt || indices.scalar_type() == at::kLong,
        "Expect 32 or 64 bit indices, but found ",
        indices.scalar_type(),
        " instead.");
    
    // 检查 offsets 张量的数据类型是否为 int 或 long
    TORCH_CHECK(
        offsets->scalar_type() == at::kInt || offsets->scalar_type() == at::kLong,
        "Expect 32 or 64 bit offsets, but found ",
        offsets->scalar_type(),
        " instead.");
    
    // 检查 weight, indices 和 offsets 张量是否是连续的
    TORCH_CHECK(
        weight.is_contiguous() && indices.is_contiguous() &&
            offsets->is_contiguous(),
        "Expect weight, indices, and offsets to be contiguous.");

    // 使用辅助函数支持不同类型组合，避免需要类型转换的性能开销
    if (indices.scalar_type() == at::kInt && offsets->scalar_type() == at::kInt) {
      // 调用具体的 n-bit 嵌入实现函数，使用 int 类型的 indices 和 offsets
      return embedding_bag_nbit_impl<int, int>(
          output,
          weight,
          bit_width,
          indices,
          *offsets,
          pruned_weights,
          per_sample_weights_,
          compressed_indices_mapping,
          include_last_offset,
          is_embedding_op);
    } else if (
        indices.scalar_type() == at::kInt && offsets->scalar_type() == at::kLong) {
    // 如果 indices 的数据类型是 int，并且 offsets 的数据类型是 int，则调用模板函数 embedding_bag_nbit_impl<int, int64_t>
    return embedding_bag_nbit_impl<int, int64_t>(
        output,                      // 输出张量
        weight,                      // 权重张量
        bit_width,                   // 嵌入的比特宽度
        indices,                     // 索引张量
        *offsets,                    // 偏移量张量的解引用
        pruned_weights,              // 精简后的权重张量
        per_sample_weights_,         // 每个样本的权重张量
        compressed_indices_mapping,  // 压缩索引映射
        include_last_offset,         // 是否包含最后一个偏移量
        is_embedding_op);            // 是否为嵌入操作的标志
    } else if (
        // 如果 indices 的数据类型是 long，并且 offsets 的数据类型是 int，则调用模板函数 embedding_bag_nbit_impl<int64_t, int>
        indices.scalar_type() == at::kLong && offsets->scalar_type() == at::kInt) {
      return embedding_bag_nbit_impl<int64_t, int>(
          output,                      // 输出张量
          weight,                      // 权重张量
          bit_width,                   // 嵌入的比特宽度
          indices,                     // 索引张量
          *offsets,                    // 偏移量张量的解引用
          pruned_weights,              // 精简后的权重张量
          per_sample_weights_,         // 每个样本的权重张量
          compressed_indices_mapping,  // 压缩索引映射
          include_last_offset,         // 是否包含最后一个偏移量
          is_embedding_op);            // 是否为嵌入操作的标志
    }
    // 默认情况下，调用模板函数 embedding_bag_nbit_impl<int64_t, int64_t>
    return embedding_bag_nbit_impl<int64_t, int64_t>(
        output,                      // 输出张量
        weight,                      // 权重张量
        bit_width,                   // 嵌入的比特宽度
        indices,                     // 索引张量
        *offsets,                    // 偏移量张量的解引用
        pruned_weights,              // 精简后的权重张量
        per_sample_weights_,         // 每个样本的权重张量
        compressed_indices_mapping,  // 压缩索引映射
        include_last_offset,         // 是否包含最后一个偏移量
        is_embedding_op);            // 是否为嵌入操作的标志
}

} // namespace

// 定义了 PackedEmbeddingBagWeight 类的成员函数，用于计算 byte 类型的嵌入
at::Tensor PackedEmbeddingBagWeight::embeddingbag_byte(
    const at::Tensor& indices, // 输入的索引张量
    const std::optional<at::Tensor>& offsets_in, // 可选的偏移张量
    bool pruned_weights, // 是否使用了稀疏权重
    const std::optional<at::Tensor>& per_sample_weights_, // 可选的每样本权重张量
    const std::optional<at::Tensor>& compressed_indices_mapping, // 可选的压缩索引映射张量
    bool include_last_offset, // 是否包含最后一个偏移量
    bool is_embedding_op) { // 是否为嵌入操作
  auto output = at::empty({0}, packed_w.options().dtype(at::kFloat)); // 创建一个空张量作为输出
  return embedding_bag_byte_helper(
      output,
      packed_w,
      indices,
      offsets_in,
      pruned_weights,
      per_sample_weights_,
      compressed_indices_mapping,
      include_last_offset,
      is_embedding_op); // 调用辅助函数进行 byte 类型的嵌入计算
}

// 定义了 PackedEmbeddingBagWeight 类的成员函数，用于计算 4-bit 类型的嵌入
at::Tensor PackedEmbeddingBagWeight::embeddingbag_4bit(
    const at::Tensor& indices, // 输入的索引张量
    const std::optional<at::Tensor>& offsets_in, // 可选的偏移张量
    bool pruned_weights, // 是否使用了稀疏权重
    const std::optional<at::Tensor>& per_sample_weights_, // 可选的每样本权重张量
    const std::optional<at::Tensor>& compressed_indices_mapping, // 可选的压缩索引映射张量
    bool include_last_offset, // 是否包含最后一个偏移量
    bool is_embedding_op) { // 是否为嵌入操作
  if (per_sample_weights_.has_value()) {
    // 检查每样本权重张量的数据类型是否为 float32 或 float16
    TORCH_CHECK(
        (per_sample_weights_.value().scalar_type() == at::kFloat ||
         per_sample_weights_.value().scalar_type() == at::kHalf),
        "Expect fp32 or fp16 weights, but found",
        per_sample_weights_.value().scalar_type(),
        " instead")
  }

  auto output = at::empty({0}, packed_w.options().dtype(at::kFloat)); // 创建一个空张量作为输出
  return _embedding_bag_nbit_helper(
    output,
    packed_w,
    4, // 使用 4-bit 编码
    indices,
    offsets_in,
    pruned_weights,
    per_sample_weights_.has_value()
        ? per_sample_weights_.value().to(at::kFloat) // 将每样本权重张量转换为 float 类型
        : per_sample_weights_,
    compressed_indices_mapping,
    include_last_offset,
    is_embedding_op); // 调用辅助函数进行 4-bit 类型的嵌入计算
}

namespace at {
namespace native {

// 在输出张量中进行 byte 类型的行偏移嵌入计算
Tensor& embedding_bag_byte_rowwise_offsets_out(
    Tensor& output,
    const Tensor& weight,
    const Tensor& indices,
    const std::optional<Tensor>& offsets_in,
    const bool /* scale_grad_by_freq */,
    const int64_t /* mode */,
    bool pruned_weights,
    const std::optional<Tensor>& per_sample_weights_,
    const std::optional<Tensor>& compressed_indices_mapping,
    bool include_last_offset) {
  return embedding_bag_byte_helper(
      output,
      weight,
      indices,
      offsets_in,
      pruned_weights,
      per_sample_weights_,
      compressed_indices_mapping,
      include_last_offset,
      false /* is_embedding_op */); // 调用辅助函数进行 byte 类型的行偏移嵌入计算
}

// 在输出张量中进行 4-bit 类型的行偏移嵌入计算
Tensor& embedding_bag_4bit_rowwise_offsets_out(
    Tensor& output,
    const Tensor& weight,
    const Tensor& indices,
    const std::optional<Tensor>& offsets_in,
    const bool /* scale_grad_by_freq */,
    const int64_t /* mode */,
    bool pruned_weights,
    const std::optional<Tensor>& per_sample_weights_,
    const std::optional<Tensor>& compressed_indices_mapping,
    bool include_last_offset) {

  if (per_sample_weights_.has_value()) {
    // 检查每样本权重张量的数据类型是否为 float32 或 float16
    ```
    // 使用 TORCH_CHECK 宏来检查条件，确保 per_sample_weights_ 的值是 float32 或者 float16 类型
    TORCH_CHECK(
        (per_sample_weights_.value().scalar_type() == at::kFloat ||
         per_sample_weights_.value().scalar_type() == at::kHalf),
        "Expect fp32 or fp16 weights, but found",
        per_sample_weights_.value().scalar_type(),
        " instead")
  }
  // 返回调用 _embedding_bag_nbit_helper 函数的结果，传入多个参数：
  // output: 输出张量
  // weight: 权重张量
  // 4: nbit 参数
  // indices: 索引张量
  // offsets_in: 偏移量张量
  // pruned_weights: 稀疏权重张量
  // per_sample_weights_.has_value() ? per_sample_weights_.value().to(at::kFloat) : per_sample_weights_:
  // 如果 per_sample_weights_ 有值，则将其转换为 float 类型，否则保持原值
  // compressed_indices_mapping: 压缩索引映射
  // include_last_offset: 是否包含最后一个偏移量
  // false: 不进行平均
  return _embedding_bag_nbit_helper(
      output,
      weight,
      4,
      indices,
      offsets_in,
      pruned_weights,
      per_sample_weights_.has_value()
          ? per_sample_weights_.value().to(at::kFloat)
          : per_sample_weights_,
      compressed_indices_mapping,
      include_last_offset,
      false);
}

// 定义一个静态函数，用于计算二进制压缩的行偏移量嵌入
static Tensor& embedding_bag_2bit_rowwise_offsets_out(
    Tensor& output, // 输出张量的引用
    const Tensor& weight, // 权重张量
    const Tensor& indices, // 索引张量
    const std::optional<Tensor>& offsets_in, // 可选的偏移量张量
    const bool /* scale_grad_by_freq */, // 是否按频率缩放梯度，未使用
    const int64_t /* mode */, // 模式，未使用
    bool pruned_weights, // 是否使用了修剪后的权重
    const std::optional<Tensor>& per_sample_weights_, // 可选的每样本权重张量
    const std::optional<Tensor>& compressed_indices_mapping, // 可选的压缩索引映射
    bool include_last_offset) { // 是否包含最后一个偏移量

  // 如果存在每样本权重张量，则进行类型检查，应为float32或float16
  if (per_sample_weights_.has_value()) {
    TORCH_CHECK(
        (per_sample_weights_.value().scalar_type() == at::kFloat ||
         per_sample_weights_.value().scalar_type() == at::kHalf),
        "Expect fp32 or fp16 weights, but found",
        per_sample_weights_.value().scalar_type(),
        " instead")
  }
  // 调用内部的二进制压缩嵌入助手函数，并返回其结果
  return _embedding_bag_nbit_helper(
      output,
      weight,
      2, // 使用2比特行偏移量压缩
      indices,
      offsets_in,
      pruned_weights,
      per_sample_weights_.has_value()
          ? per_sample_weights_.value().to(at::kFloat)
          : per_sample_weights_, // 如果存在每样本权重，将其转换为float类型
      compressed_indices_mapping,
      include_last_offset,
      false); // 不进行缩放梯度按频率操作
}

// 定义一个命名空间
namespace {

// 内联函数，创建一个与给定张量具有相同属性的空张量
inline at::Tensor create_empty_from(
    const at::Tensor& t, // 给定的张量
    c10::ScalarType dtype) { // 数据类型
  return at::detail::empty_cpu(
      {0}, dtype, t.layout(), t.device(), c10::nullopt, c10::nullopt); // 创建一个空张量
}

// 计算字节级别的行偏移量嵌入
Tensor embedding_bag_byte_rowwise_offsets(
    const Tensor& weight, // 权重张量
    const Tensor& indices, // 索引张量
    const std::optional<Tensor>& offsets_in, // 可选的偏移量张量
    const bool /* scale_grad_by_freq */, // 是否按频率缩放梯度，未使用
    const int64_t /* mode */, // 模式，未使用
    bool pruned_weights, // 是否使用了修剪后的权重
    const std::optional<Tensor>& per_sample_weights_, // 可选的每样本权重张量
    const std::optional<Tensor>& compressed_indices_mapping, // 可选的压缩索引映射
    bool include_last_offset) { // 是否包含最后一个偏移量
  auto output = create_empty_from(weight, at::kFloat); // 创建一个空的float类型输出张量
  embedding_bag_byte_rowwise_offsets_out(
      output,
      weight,
      indices,
      offsets_in,
      false /*unused scale_grad_by_freq*/, // 不使用的缩放梯度按频率参数
      0 /*unused mode*/, // 不使用的模式参数
      pruned_weights,
      per_sample_weights_,
      compressed_indices_mapping,
      include_last_offset); // 调用字节级行偏移量嵌入的输出函数
  return output; // 返回结果张量
}

// 计算四比特级别的行偏移量嵌入
Tensor embedding_bag_4bit_rowwise_offsets(
    const Tensor& weight, // 权重张量
    const Tensor& indices, // 索引张量
    const std::optional<Tensor>& offsets_in, // 可选的偏移量张量
    const bool /* scale_grad_by_freq */, // 是否按频率缩放梯度，未使用
    const int64_t /* mode */, // 模式，未使用
    bool pruned_weights, // 是否使用了修剪后的权重
    const std::optional<Tensor>& per_sample_weights_, // 可选的每样本权重张量
    const std::optional<Tensor>& compressed_indices_mapping, // 可选的压缩索引映射
    bool include_last_offset) { // 是否包含最后一个偏移量
  auto output = create_empty_from(weight, at::kFloat); // 创建一个空的float类型输出张量
  embedding_bag_4bit_rowwise_offsets_out(
    output,
    weight,
    indices,
    offsets_in,
    false, // unused scale_grad_by_freq，不使用的缩放梯度按频率参数
    0, // unused mode，不使用的模式参数
    pruned_weights,
    per_sample_weights_,
    compressed_indices_mapping,
    include_last_offset); // 调用四比特级行偏移量嵌入的输出函数
  return output; // 返回结果张量
}

// 计算二比特级别的行偏移量嵌入
Tensor embedding_bag_2bit_rowwise_offsets(
    const Tensor& weight, // 权重张量
    const Tensor& indices, // 索引张量
    const std::optional<Tensor>& offsets_in, // 可选的偏移量张量
    const bool /* scale_grad_by_freq */, // 是否按频率缩放梯度，未使用
    const int64_t /* mode */, // 模式，未使用
    bool pruned_weights,
    const std::optional<Tensor>& per_sample_weights_, // 可选的每样本权重张量
    const std::optional<Tensor>& compressed_indices_mapping, // 可选的压缩索引映射
    bool include_last_offset) { // 是否包含最后一个偏移量
  // 创建一个空的float类型输出张量
  auto output = create_empty_from(weight, at::kFloat);
  // 调用二比特级行偏移量嵌入的输出函数
  return embedding_bag_2bit_rowwise_offsets_out(
      output,
      weight,
      indices,
      offsets_in,
      false, // unused scale_grad_by_freq，不使用的缩放梯度按频率参数
      0, // unused mode，不使用的模式参数
      pruned_weights,
      per_sample_weights_,
      compressed_indices_mapping,
      include_last_offset);
}
    const std::optional<Tensor>& per_sample_weights_,  

# 接收可选的张量作为每个样本的权重

    const std::optional<Tensor>& compressed_indices_mapping,

# 接收可选的张量作为压缩索引映射

    bool include_last_offset) {

# 布尔值参数，指定是否包含最后一个偏移量

  auto output = create_empty_from(weight, at::kFloat);

# 创建一个与给定权重张量相同形状的空张量，数据类型为 kFloat

  embedding_bag_2bit_rowwise_offsets_out(
    output,
    weight,
    indices,
    offsets_in,
    false, // unused scale_grad_by_freq

# 调用嵌入袋操作函数，传递输出张量、权重张量、索引张量、偏移量张量，以及不使用的参数 scale_grad_by_freq

    0, // unused mode

# 指定嵌入袋操作的模式，此处未使用，设为 0

    pruned_weights,

# 传递经过修剪的权重张量

    per_sample_weights_,

# 传递每个样本的权重张量或空值

    compressed_indices_mapping,

# 传递压缩索引映射张量或空值

    include_last_offset);

# 传递是否包含最后一个偏移量的布尔值

  return output;

# 返回生成的输出张量
  // 根据给定参数和条件执行量化的嵌入袋操作，返回嵌入结果张量
  if (bit_rate == 8) {
    // 如果量化比特率为8，调用8位字节嵌入袋操作
    return packed_weight->embeddingbag_byte(
        indices,
        offsets,
        pruned_weights,
        per_sample_weights_,
        compressed_indices_mapping,
        include_last_offset,
        false /* is_embedding_op */);
  } else if (bit_rate == 4) {
    // 如果量化比特率为4，调用4位字节嵌入袋操作
    return packed_weight->embeddingbag_4bit(
        indices,
        offsets,
        pruned_weights,
        per_sample_weights_,
        compressed_indices_mapping,
        include_last_offset,
        false);
  } else {
    // 如果量化比特率既不是8也不是4，则抛出内部断言错误
    TORCH_INTERNAL_ASSERT(
        false,
        "Currently only support 8-bit embedding_bag quantization");
  }
}
    # 计算索引张量的元素数量
    const auto offsets_size = indices.numel();
    # 创建一个张量 `offsets`，包含从0到offsets_size-1的整数，使用与indices相同的数据类型
    at::Tensor offsets = at::arange(0, offsets_size, indices.scalar_type());
    # 初始化输出张量
    at::Tensor output;
    # 如果比特率为8
    if (bit_rate == 8) {
      # 调用packed_weight的embeddingbag_byte方法，返回嵌入数据
      return packed_weight->embeddingbag_byte(
          indices,
          offsets,
          pruned_weights,
          c10::nullopt,
          c10::nullopt,
          false /* include_last_offset */,
          true /* is_embedding_op */);
    } else if (bit_rate == 4) {
      # 如果比特率为4
      # 调用packed_weight的embeddingbag_4bit方法，返回4位量化的嵌入数据
      return packed_weight->embeddingbag_4bit(
          indices,
          offsets,
          pruned_weights,
          c10::nullopt,
          c10::nullopt,
          false,
          true);
    } else {
      # 如果比特率既不是8也不是4，抛出错误，因为当前只支持8位嵌入量化
      TORCH_INTERNAL_ASSERT(
          false,
          "Currently only support 8-bit embedding quantization");
    }
    # 返回输出张量
    return output;
  }
};

TORCH_LIBRARY_IMPL(quantized, CPU, m) {
  // 定义 Torch 库的 quantized 实现，使用 CPU 平台，传入 m 参数
  // 注册 quantized::embedding_bag_byte 函数，使用 QEmbeddingBag<8>::run 实现
  m.impl(
      TORCH_SELECTIVE_NAME("quantized::embedding_bag_byte"),
      TORCH_FN(QEmbeddingBag<8>::run));
  // 注册 quantized::embedding_bag_4bit 函数，使用 QEmbeddingBag<4>::run 实现
  m.impl(
      TORCH_SELECTIVE_NAME("quantized::embedding_bag_4bit"),
      TORCH_FN(QEmbeddingBag<4>::run));
  // 注册 quantized::embedding_byte 函数，使用 QEmbedding<8>::run 实现
  m.impl(
      TORCH_SELECTIVE_NAME("quantized::embedding_byte"),
      TORCH_FN(QEmbedding<8>::run));
  // 注册 quantized::embedding_4bit 函数，使用 QEmbedding<4>::run 实现
  m.impl(
      TORCH_SELECTIVE_NAME("quantized::embedding_4bit"),
      TORCH_FN(QEmbedding<4>::run));

  // 注册函数，处理 at::Tensor 类型的压缩权重
  m.impl(
      TORCH_SELECTIVE_NAME("quantized::embedding_bag_byte_rowwise_offsets"),
      embedding_bag_byte_rowwise_offsets);
  m.impl(
      TORCH_SELECTIVE_NAME("quantized::embedding_bag_4bit_rowwise_offsets"),
      embedding_bag_4bit_rowwise_offsets);
  m.impl(
      TORCH_SELECTIVE_NAME("quantized::embedding_bag_2bit_rowwise_offsets"),
      embedding_bag_2bit_rowwise_offsets);
}

TORCH_LIBRARY_IMPL(quantized, Meta, m) {
  // 定义 Torch 库的 quantized Meta 实现，传入 m 参数
  // 注册 quantized::embedding_bag_byte_rowwise_offsets 函数，使用 embedding_bag_byte_rowwise_offsets_meta 实现
  m.impl(
      "quantized::embedding_bag_byte_rowwise_offsets",
      embedding_bag_byte_rowwise_offsets_meta);
}

} // namespace
} // namespace native
} // namespace at
```