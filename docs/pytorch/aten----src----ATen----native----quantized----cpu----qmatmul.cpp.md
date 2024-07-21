# `.\pytorch\aten\src\ATen\native\quantized\cpu\qmatmul.cpp`

```py
// 匿名命名空间用于封装局部函数或变量，限制它们的作用域在当前编译单元内
namespace {

// 检查输入张量的数据类型和量化方案是否符合要求
inline void check_inputs(const Tensor& qa, const Tensor& qb) {
  // 检查张量 qa 和 qb 是否为 QInt8 或 QUInt8 数据类型
  TORCH_CHECK(
      qa.scalar_type() == c10::kQInt8 || qa.scalar_type() == c10::kQUInt8,
      "MatMul operands should use QInt8 or QUInt8 data types.");
  // 检查 qa 和 qb 是否具有相同的数据类型
  TORCH_CHECK(
      qa.scalar_type() == qb.scalar_type(),
      "MatMul operands should have same data type.");
  // 检查 qa 是否使用 kPerTensorAffine 或 kPerTensorSymmetric 量化方案
  TORCH_CHECK(
      qa.qscheme() == kPerTensorAffine || qa.qscheme() == kPerTensorSymmetric,
      "Only per-tensor quantization is supported in Matmul.");
  // 检查 qa 和 qb 是否具有相同的量化方案
  TORCH_CHECK(
      qa.qscheme() == qb.qscheme(),
      "Both inputs to Matmul must have the same quantization scheme.");
}

#ifdef USE_RUY_QMATMUL

// 执行量化矩阵乘法运算，并返回结果张量
Tensor qmatmul(
    const Tensor& qa,
    const Tensor& qb,
    const double output_scale,
    const int64_t output_zero_point) {
  
  // 调用 check_inputs 函数检查输入张量的合法性
  check_inputs(qa, qb);

  // 获取输入张量的维度信息
  const int64_t num_dims = qa.dim();
  const int64_t b_num_dims = qb.dim();

  // 检查输入张量的维度是否一致
  TORCH_CHECK(
      num_dims == b_num_dims,
      "MatMul operands should have the same dimensionality. (", num_dims,
      " and ", b_num_dims, " provided)");
  
  // 检查输入张量的维度是否符合要求
  TORCH_CHECK(
      num_dims >= 2,
      "Quantized Matmul currently only supports operands which are at least 2-dimensional. (",
      num_dims, " provided)");

  // 获取输入张量的尺寸信息
  const int64_t m = qa.size(num_dims - 2);
  const int64_t k = qa.size(num_dims - 1);
  const int64_t b_k = qb.size(num_dims - 2);
  const int64_t n = qb.size(num_dims - 1);

  // 检查输入张量在矩阵乘法维度上的尺寸是否匹配
  TORCH_CHECK(
      b_k == k,
      "For Quantized Matmul, the size of tensor a (", k,
      ") at dimension ", num_dims - 1, " must match the size of tensor b (",
      b_k, ") at dimension ", num_dims - 2, ".");

  // 初始化输出张量的大小向量和乘积计数器
  std::vector<int64_t> out_size_vec(num_dims);
  size_t num_matmuls = 1;
  for (int64_t i = 0; i < num_dims - 2; i++) {
    // 获取当前维度的尺寸
    const int64_t dim = qa.size(i);
    const int64_t qb_dim = qb.size(i);

    // 检查当前维度的尺寸是否一致
    TORCH_CHECK(
        dim == qb_dim,
        "For Quantized Matmul, the size of tensor a (", dim,
        ") must match the size of tensor b (", qb_dim,
        ") at dimension ", i);

    // 更新输出大小向量和乘积计数器
    out_size_vec[i] = dim;
    num_matmuls *= dim;
  }
  // 设置输出张量的最后两个维度的尺寸
  out_size_vec[num_dims - 2] = m;
  out_size_vec[num_dims - 1] = n;

  // 创建一个空的仿射量化张量作为输出
  Tensor out = at::_empty_affine_quantized(
      IntArrayRef(out_size_vec),
      at::device(kCPU)
          .dtype(qa.scalar_type())
          .memory_format(qa.suggest_memory_format()),
      output_scale,
      output_zero_point,
      c10::nullopt);

  // 获取连续化后的输入张量
  const Tensor& qa_contig = qa.contiguous();
  const Tensor& qb_contig = qb.contiguous();

  // 使用宏展开来处理输入张量的数据类型，以及执行矩阵乘法的具体计算
  AT_DISPATCH_QINT_BYTE_TYPES(qa.scalar_type(), "qmatmul", [&] {
    using underlying_t = typename scalar_t::underlying;

    // 从连续化后的输入张量中获取数据指针
    const underlying_t* qa_data = reinterpret_cast<const underlying_t*>(
        qa_contig.data_ptr<scalar_t>());
    // 将 qb_contig 的数据指针转换为 underlying_t 类型的常量指针
    const underlying_t* qb_data = reinterpret_cast<const underlying_t*>(
        qb_contig.data_ptr<scalar_t>());
    // 将 out 的数据指针转换为 underlying_t 类型的指针
    underlying_t* out_data =
        reinterpret_cast<underlying_t*>(out.data_ptr<scalar_t>());

    // 计算各矩阵的步长
    const size_t qa_stride = m * k;
    const size_t qb_stride = k * n;
    const size_t out_stride = m * n;

    // 并行执行矩阵乘法的 Lambda 函数 matmuls
    auto matmuls = [&](int64_t begin, int64_t end) {

      // 创建 QA 矩阵并初始化
      ruy::Matrix<underlying_t> qa_matrix;
      ruy::MakeSimpleLayout(
          m, k, ruy::Order::kRowMajor, qa_matrix.mutable_layout());
      qa_matrix.set_zero_point(qa.q_zero_point());

      // 创建 QB 矩阵并初始化
      ruy::Matrix<underlying_t> qb_matrix;
      ruy::MakeSimpleLayout(
          k, n, ruy::Order::kRowMajor, qb_matrix.mutable_layout());
      qb_matrix.set_zero_point(qb.q_zero_point());

      // 创建输出矩阵并初始化
      ruy::Matrix<underlying_t> out_matrix;
      ruy::MakeSimpleLayout(
          m, n, ruy::Order::kRowMajor, out_matrix.mutable_layout());
      out_matrix.set_zero_point(output_zero_point);

      // 计算重新量化比例的倒数
      const double requantization_scale_inv =
          (qa.q_scale() * qb.q_scale()) / output_scale;

      // 设置矩阵乘法参数
      ruy::MulParams<int32_t, underlying_t> mul_params;

      int multiplier_fixedpoint;
      int multiplier_exponent;
      // 计算乘法器的定点数和指数
      ruy_utils::quantize_multiplier(requantization_scale_inv,
                                     &multiplier_fixedpoint,
                                     &multiplier_exponent);
      mul_params.set_multiplier_fixedpoint(multiplier_fixedpoint);
      mul_params.set_multiplier_exponent(multiplier_exponent);

      // 定义 QA、QB 和输出子张量的指针
      const underlying_t* qa_subtensor = qa_data + begin * qa_stride;
      const underlying_t* qb_subtensor = qb_data + begin * qb_stride;
      underlying_t* out_subtensor = out_data + begin * out_stride;

      // 执行矩阵乘法循环
      for (int64_t i = begin; i < end; i++) {
        // 设置当前迭代的子张量数据
        qa_matrix.set_data(qa_subtensor);
        qb_matrix.set_data(qb_subtensor);
        out_matrix.set_data(out_subtensor);
        // 调用 Ruy 库的矩阵乘法函数 Mul
        ruy::Mul(qa_matrix,
                 qb_matrix,
                 mul_params,
                 ruy_utils::get_ruy_context(),
                 &out_matrix);

        // 更新下一个迭代的子张量指针
        qa_subtensor += qa_stride;
        qb_subtensor += qb_stride;
        out_subtensor += out_stride;
      }
    };

    // 使用 ATen 的并行方法 parallel_for 调度 matmuls Lambda 函数的执行
    at::parallel_for(0, num_matmuls, 1, matmuls);

    // 返回存储计算结果的输出张量 out
    return out;
}

#else // ifdef USE_RUY_QMATMUL


Tensor qmatmul(
    const Tensor& qa,
    const Tensor& qb,
    const double output_scale,
    const int64_t output_zero_point) {


  check_inputs(qa, qb);


  Tensor ra = at::dequantize(qa);
  Tensor rb = at::dequantize(qb);


  Tensor rc = at::matmul(ra, rb);


  return at::quantize_per_tensor(
      rc, output_scale, output_zero_point, qa.scalar_type());
}


#endif // ifdef USE_RUY_QMATMUL


TORCH_LIBRARY_IMPL(quantized, QuantizedCPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("quantized::matmul"), TORCH_FN(qmatmul));
}


} // namespace

} // namespace native
} // namespace at
```