# `.\pytorch\aten\src\ATen\native\CPUBlas.cpp`

```py
void normalize_last_dims(
    // 标准化矩阵乘法的最后维度，根据转置类型和维度大小调整矩阵的维度
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    int64_t *lda, int64_t *ldb, int64_t *ldc) {
  
  // 如果输出矩阵的列数为1，调整输出矩阵的行列数为m*n
  if (n == 1) {
    *ldc = m;
  }

  // 如果输入矩阵A需要转置，且输入矩阵的行数为1，调整lda为k
  if(transa != TransposeType::NoTranspose) {
    if (m == 1) {
      *lda = k;
    }
  } else if(k == 1) {
    *lda = m;
  }

  // 如果输入矩阵B需要转置，且输入矩阵的列数为1，调整ldb为k
  if(transb != TransposeType::NoTranspose) {
    if (k == 1) {
      *ldb = n;
    }
  } else if (n == 1) {
    *ldb = k;
  }
}
    // 检查矩阵乘法参数的有效性，确保能够进行矩阵乘法操作
    int64_t lda, int64_t ldb, int64_t ldc) {
      // 判断是否需要对矩阵 A 进行转置
      const bool transa_ = transa != TransposeType::NoTranspose;
      // 判断是否需要对矩阵 B 进行转置
      const bool transb_ = transb != TransposeType::NoTranspose;
      // 检查矩阵维度和步长是否在合理范围内，并且大于等于1
      return (
          (m <= INT_MAX) && (n <= INT_MAX) && (k <= INT_MAX) &&
          (lda <= INT_MAX) && (ldb <= INT_MAX) && (ldc <= INT_MAX) &&
          // 检查 lda 是否满足矩阵 A 的转置要求
          (lda >= std::max(int64_t{1}, (transa_ ? k : m))) &&
          // 检查 ldb 是否满足矩阵 B 的转置要求
          (ldb >= std::max(int64_t{1}, (transb_ ? n : k))) &&
          // 检查 ldc 是否满足矩阵 C 的行数要求
          (ldc >= std::max(int64_t{1}, m)));
#ifdef USE_FBGEMM
// 如果定义了 USE_FBGEMM 宏，则转换 TransposeType 到 fbgemm 的 matrix_op_t 类型
fbgemm::matrix_op_t to_fbgemm(TransposeType trans) {
  switch (trans) {
    case TransposeType::Transpose: return fbgemm::matrix_op_t::Transpose;
    case TransposeType::NoTranspose: return fbgemm::matrix_op_t::NoTranspose;
    case TransposeType::ConjTranspose: TORCH_INTERNAL_ASSERT(false, "ConjTranspose type is not supported in fbgemm");
  }
  TORCH_INTERNAL_ASSERT(false, "Invalid transpose type");
}
#endif  // USE_FBGEMM

#if (AT_BUILD_WITH_BLAS() && C10_IOS)
// 如果在构建时包括了 BLAS 并且是 C10_IOS 环境，则转换 TransposeType 到 CBLAS_TRANSPOSE 类型
CBLAS_TRANSPOSE to_apple_accelerate_transpose(TransposeType trans) {
  switch (trans) {
    case TransposeType::Transpose: return CblasTrans;
    case TransposeType::NoTranspose: return CblasNoTrans;
    case TransposeType::ConjTranspose: return CblasConjTrans;
  }
  TORCH_INTERNAL_ASSERT(false, "Invalid transpose type");
}
#endif

}  // namespace (anonymous)

// 定义 gemm_stub 的分发函数
DEFINE_DISPATCH(gemm_stub);

// 实现 gemm 函数，用于执行矩阵乘法
void gemm(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    const double alpha,
    const double *a, int64_t lda,
    const double *b, int64_t ldb,
    const double beta,
    double *c, int64_t ldc) {
  // 标准化最后维度
  internal::normalize_last_dims(transa, transb, m, n, k, &lda, &ldb, &ldc);
  
  // 如果使用 BLAS 加速 gemm
#if AT_BUILD_WITH_BLAS()
  if (use_blas_gemm(transa, transb, m, n, k, lda, ldb, ldc)) {
    int m_ = m, n_ = n, k_ = k, lda_ = lda, ldb_ = ldb, ldc_ = ldc;
    double alpha_ = alpha, beta_ = beta;
    
    // 如果是在 C10_IOS 平台上
    #if C10_IOS
    CBLAS_TRANSPOSE transa_ = to_apple_accelerate_transpose(transa);
    CBLAS_TRANSPOSE transb_ = to_apple_accelerate_transpose(transb);
    
    // 调用 Apple Accelerate 的双精度矩阵乘法函数
    cblas_dgemm(CblasColMajor,
      transa_, transb_,
      m_, n_, k_,
      alpha_,
      a, lda_,
      b, ldb_,
      beta_,
      c, ldc_);
    #else
    char transa_ = to_blas(transa), transb_ = to_blas(transb);
    
    // 调用 BLAS 的双精度矩阵乘法函数
    dgemm_(
        &transa_, &transb_,
        &m_, &n_, &k_,
        &alpha_,
        a, &lda_,
        b, &ldb_,
        &beta_,
        c, &ldc_);
    #endif
    
    return;
  }
#endif

  // 如果不使用 BLAS 加速，则调用 gemm_stub 进行处理
  gemm_stub(
      at::kCPU, at::kDouble,
      transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

// 实现 gemm 函数，用于执行矩阵乘法（单精度版本）
void gemm(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    const float alpha,
    const float *a, int64_t lda,
    const float *b, int64_t ldb,
    const float beta,
    float *c, int64_t ldc) {
  // 标准化最后维度
  internal::normalize_last_dims(transa, transb, m, n, k, &lda, &ldb, &ldc);
  
  // 如果启用了 MKL-DNN 加速 gemm
#if AT_MKLDNN_ENABLED()
   if (mkldnn_bf32_gemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)) {
     return;
   }
#endif

  // 如果使用 BLAS 加速 gemm
#if AT_BUILD_WITH_BLAS()
  if (use_blas_gemm(transa, transb, m, n, k, lda, ldb, ldc)) {
    int m_ = m, n_ = n, k_ = k, lda_ = lda, ldb_ = ldb, ldc_ = ldc;
    float alpha_ = alpha, beta_ = beta;
    
    // 如果是在 C10_IOS 平台上
    #if C10_IOS
    CBLAS_TRANSPOSE transa_ = to_apple_accelerate_transpose(transa);
    CBLAS_TRANSPOSE transb_ = to_apple_accelerate_transpose(transb);
    
    // 调用 Apple Accelerate 的单精度矩阵乘法函数
    cblas_sgemm(CblasColMajor,
      transa_, transb_,
      m_, n_, k_,
      alpha_,
      a, lda_,
      b, ldb_,
      beta_,
      c, ldc_);
    #else
    
    // 调用 BLAS 的单精度矩阵乘法函数
    char transa_ = to_blas(transa), transb_ = to_blas(transb);
    sgemm_(
        &transa_, &transb_,
        &m_, &n_, &k_,
        &alpha_,
        a, &lda_,
        b, &ldb_,
        &beta_,
        c, &ldc_);
    #endif
    
    return;
  }
#endif

  // 如果没有使用 BLAS 或 MKL-DNN 加速，则调用 gemm_stub 进行处理
  gemm_stub(
      at::kCPU, at::kFloat,
      transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
    cblas_sgemm(CblasColMajor,
      transa_, transb_,
      m_, n_, k_,
      alpha_,
      a, lda_,
      b, ldb_,
      beta_,
      c, ldc_);


    // 调用 cblas_sgemm 函数，执行矩阵乘法操作
    // 使用列主序存储方式
    // transa_ 表示是否对 A 矩阵进行转置操作
    // transb_ 表示是否对 B 矩阵进行转置操作
    // m_ 表示 A 矩阵的行数或者 C 矩阵的行数（根据 CblasColMajor 决定）
    // n_ 表示 B 矩阵的列数或者 C 矩阵的列数（根据 CblasColMajor 决定）
    // k_ 表示 A 矩阵的列数或者 B 矩阵的行数（根据 CblasColMajor 决定）
    // alpha_ 表示乘法的缩放因子
    // a 是指向 A 矩阵的指针
    // lda_ 是 A 矩阵的 leading dimension（领先维度）
    // b 是指向 B 矩阵的指针
    // ldb_ 是 B 矩阵的 leading dimension
    // beta_ 表示 C 矩阵的缩放因子
    // c 是指向 C 矩阵的指针
    // ldc_ 是 C 矩阵的 leading dimension



    #else
    char transa_ = to_blas(transa), transb_ = to_blas(transb);
    sgemm_(
        &transa_, &transb_,
        &m_, &n_, &k_,
        &alpha_,
        a, &lda_,
        b, &ldb_,
        &beta_,
        c, &ldc_);
    #endif


    // 在不使用 cblas 库的情况下，调用 sgemm_ 函数执行矩阵乘法操作
    // 使用自定义的转置标志（通过 to_blas 函数转换）
    // transa_ 表示是否对 A 矩阵进行转置操作
    // transb_ 表示是否对 B 矩阵进行转置操作
    // m_ 表示 A 矩阵的行数
    // n_ 表示 B 矩阵的列数
    // k_ 表示 A 矩阵的列数或者 B 矩阵的行数（根据转置标志决定）
    // alpha_ 表示乘法的缩放因子
    // a 是指向 A 矩阵的指针
    // lda_ 是 A 矩阵的 leading dimension
    // b 是指向 B 矩阵的指针
    // ldb_ 是 B 矩阵的 leading dimension
    // beta_ 表示 C 矩阵的缩放因子
    // c 是指向 C 矩阵的指针
    // ldc_ 是 C 矩阵的 leading dimension



    return;
  }


    // 函数结束，无返回值
    return;
  }
void gemm(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    const c10::complex<double> alpha,
    const c10::complex<double> *a, int64_t lda,
    const c10::complex<double> *b, int64_t ldb,
    const c10::complex<double> beta,
    c10::complex<double> *c, int64_t ldc) {
  // 根据需要标准化最后的维度和步长
  internal::normalize_last_dims(transa, transb, m, n, k, &lda, &ldb, &ldc);
  // 如果使用 BLAS 加速库进行矩阵乘法计算
#if AT_BUILD_WITH_BLAS()
  if (use_blas_gemm(transa, transb, m, n, k, lda, ldb, ldc)) {
    // 复制参数以调用底层 BLAS 函数
    int m_ = m, n_ = n, k_ = k, lda_ = lda, ldb_ = ldb, ldc_ = ldc;
    c10::complex<double> alpha_ = alpha, beta_ = beta;
    // 根据平台选择相应的 BLAS 函数调用方式
    #if C10_IOS
    CBLAS_TRANSPOSE transa_ = to_apple_accelerate_transpose(transa);
    CBLAS_TRANSPOSE transb_ = to_apple_accelerate_transpose(transb);
    // 调用苹果加速框架中的复数双精度矩阵乘法函数
    cblas_zgemm(CblasColMajor,
      transa_, transb_,
      m_, n_, k_,
      &alpha_,
      a, lda_,
      b, ldb_,
      &beta_,
      c, ldc_);
    #else
    char transa_ = to_blas(transa), transb_ = to_blas(transb);
    // 调用 BLAS 中的复数双精度矩阵乘法函数
    zgemm_(
        &transa_, &transb_,
        &m_, &n_, &k_,
        &alpha_,
        a, &lda_,
        b, &ldb_,
        &beta_,
        c, &ldc_);
    #endif
    return;
  }
#endif
  // 如果未使用 BLAS 加速，则调用 CPU 上的通用复数双精度矩阵乘法实现
  gemm_stub(
      at::kCPU, at::kComplexDouble,
      transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}



void gemm(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    const c10::complex<float> alpha,
    const c10::complex<float> *a, int64_t lda,
    const c10::complex<float> *b, int64_t ldb,
    const c10::complex<float> beta,
    c10::complex<float> *c, int64_t ldc) {
  // 根据需要标准化最后的维度和步长
  internal::normalize_last_dims(transa, transb, m, n, k, &lda, &ldb, &ldc);
  // 如果使用 BLAS 加速库进行矩阵乘法计算
#if AT_BUILD_WITH_BLAS()
  if (use_blas_gemm(transa, transb, m, n, k, lda, ldb, ldc)) {
    // 复制参数以调用底层 BLAS 函数
    int m_ = m, n_ = n, k_ = k, lda_ = lda, ldb_ = ldb, ldc_ = ldc;
    c10::complex<float> alpha_ = alpha, beta_ = beta;
    // 根据平台选择相应的 BLAS 函数调用方式
    #if C10_IOS
    CBLAS_TRANSPOSE transa_ = to_apple_accelerate_transpose(transa);
    CBLAS_TRANSPOSE transb_ = to_apple_accelerate_transpose(transb);
    // 调用苹果加速框架中的复数单精度矩阵乘法函数
    cblas_cgemm(CblasColMajor,
      transa_, transb_,
      m_, n_, k_,
      &alpha_,
      a, lda_,
      b, ldb_,
      &beta_,
      c, ldc_);
    #else
    char transa_ = to_blas(transa), transb_ = to_blas(transb);
    // 调用 BLAS 中的复数单精度矩阵乘法函数
    cgemm_(
        &transa_, &transb_,
        &m_, &n_, &k_,
        &alpha_,
        a, &lda_,
        b, &ldb_,
        &beta_,
        c, &ldc_);
    #endif
    return;
  }
#endif
  // 如果未使用 BLAS 加速，则调用 CPU 上的通用复数单精度矩阵乘法实现
  gemm_stub(
      at::kCPU, at::kComplexFloat,
      transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}



void gemm(
   TransposeType transa, TransposeType transb,
   int64_t m, int64_t n, int64_t k,
   const float alpha,
   const at::BFloat16 *a, int64_t lda,
   const at::BFloat16 *b, int64_t ldb,
   const float beta,
   at::BFloat16 *c, int64_t ldc) {
   // 根据需要标准化最后的维度和步长
   internal::normalize_last_dims(transa, transb, m, n, k, &lda, &ldb, &ldc);
#if AT_BUILD_WITH_BLAS() && defined(BLAS_HAS_SBGEMM)
   // 检查是否支持使用 BLAS 库的 SBGEMM 函数
   if (use_blas_gemm(transa, transb, m, n, k, lda, ldb, ldc)) {
      // 如果支持，调用 SBGEMM 函数进行矩阵乘法计算
      int m_ = m, n_ = n, k_ = k, lda_ = lda, ldb_ = ldb, ldc_ = ldc;
      char transa_ = to_blas(transa), transb_ = to_blas(transb);
      float alpha_ = alpha, beta_ = beta;
      // 将输出矩阵 c 转换为 float 类型向量，因为 OpenBLAS 的 SBGEMM 函数要求输入为 float
      std::vector<float> float_v(c, c + n * ldc);
      // 调用 SBGEMM 函数进行计算，结果存储在 float_v 中
      sbgemm_(&transa_, &transb_,
              &m_, &n_, &k_,
              &alpha_,
              a, &lda_,
              b, &ldb_,
              &beta_,
              float_v.data(), &ldc_);
      // 将 float_v 中的数据转换回 BFloat16 类型，存储到 c 中
      for (auto cv: float_v) {
        *(c++) = c10::convert<at::BFloat16>(cv);
      }
      // 函数执行完毕，返回
      return;
   }
#endif
#if AT_MKLDNN_ENABLED()
   // 如果使用了 MKLDNN 库，尝试调用 MKLDNN 的 BFloat16 GEMM 函数
   if (mkldnn_bf16_gemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)) {
     // 如果调用成功，直接返回
     return;
   }
#endif
   // 否则调用 gemm_stub 函数，使用 CPU 上的 gemm 函数进行计算
   gemm_stub(
      at::kCPU, at::kBFloat16,
      transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void gemm(
   TransposeType transa, TransposeType transb,
   int64_t m, int64_t n, int64_t k,
   const float alpha,
   const at::Half *a, int64_t lda,
   const at::Half *b, int64_t ldb,
   const float beta,
   at::Half *c, int64_t ldc) {
   // 规范化矩阵维度和步长
   internal::normalize_last_dims(transa, transb, m, n, k, &lda, &ldb, &ldc);
#if AT_MKLDNN_ENABLED()
   // 如果使用了 MKLDNN 库，尝试调用 MKLDNN 的 FP16 GEMM 函数
   if (mkldnn_fp16_gemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)) {
     // 如果调用成功，直接返回
     return;
   }
#endif
   // 否则调用 gemm_stub 函数，使用 CPU 上的 gemm 函数进行计算
   gemm_stub(
      at::kCPU, at::kHalf,
      transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void gemm(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    const float alpha,
    const at::BFloat16 *a, int64_t lda,
    const at::BFloat16 *b, int64_t ldb,
    const float beta,
    float *c, int64_t ldc) {
  // 规范化矩阵维度和步长
  internal::normalize_last_dims(transa, transb, m, n, k, &lda, &ldb, &ldc);
#if AT_BUILD_WITH_BLAS() && defined(BLAS_HAS_SBGEMM)
   // 如果支持使用 BLAS 库的 SBGEMM 函数，调用该函数进行计算
   if (use_blas_gemm(transa, transb, m, n, k, lda, ldb, ldc)) {
      int m_ = m, n_ = n, k_ = k, lda_ = lda, ldb_ = ldb, ldc_ = ldc;
      char transa_ = to_blas(transa), transb_ = to_blas(transb);
      float alpha_ = alpha, beta_ = beta;
      // 调用 SBGEMM 函数进行计算
      sbgemm_(&transa_, &transb_,
              &m_, &n_, &k_,
              &alpha_,
              a, &lda_,
              b, &ldb_,
              &beta_,
              c, &ldc_);
      // 函数执行完毕，返回
      return;
   }
#endif
#ifdef MKL_HAS_SBGEMM
  // 如果支持使用 MKL 库的 SBGEMM 函数，调用该函数进行计算
  if (use_blas_gemm(transa, transb, m, n, k, lda, ldb, ldc)) {
    int m_ = m, n_ = n, k_ = k, lda_ = lda, ldb_ = ldb, ldc_ = ldc;
    // 调用 MKL 库的 BFloat16 GEMM 函数进行计算
    mkl_gemm_bf16bf16f32(transa, transb, m_, n_, k_, alpha, a, lda_, b, ldb_, beta, c, ldc_);
    // 函数执行完毕，返回
    return;
  }
#endif
#endif
// 如果定义了 MKL_HAS_SHGEMM 宏，则尝试使用 MKL 提供的高效 gemm 实现
if (use_blas_gemm(transa, transb, m, n, k, lda, ldb, ldc)) {
    // 在使用 MKL 的情况下，调用 MKL 提供的混合精度 gemm 函数 mkl_gemm_f16f16f32
    int m_ = m, n_ = n, k_ = k, lda_ = lda, ldb_ = ldb, ldc_ = ldc;
    mkl_gemm_f16f16f32(transa, transb, m_, n_, k_, alpha, a, lda_, b, ldb_, beta, c, ldc_);
    return;
}
// 如果未定义 MKL_HAS_SHGEMM 宏，则使用 fallback 路径：
// 首先以 beta = 0 计算 gemm，然后在完整精度下添加 c
int64_t c_size = n * m;
// 创建一个存储 Half 类型数据的向量 float16_c，并初始化为零
std::vector<at::Half> float16_c(c_size, 0.f);
// 调用 gemm_stub 函数进行 gemm 计算，使用 Half 类型数据，并以 beta = 0 初始化结果存储在 float16_c 中
gemm_stub(
    at::kCPU, at::kHalf,
    transa, transb, m, n, k, alpha, a, lda, b, ldb, 0.f, float16_c.data(), m);
// 遍历结果矩阵 C
for (const auto j : c10::irange(n)) {
    for (const auto i : c10::irange(m)) {
        auto offset = j * ldc + i;
        // 如果 beta = 0，则不会从 C 中传播 NaN
        if (beta == 0.f) {
            // 将 Half 类型的数据转换为 float，并存储在 C 中
            c[offset] = c10::convert<float>(float16_c[j * m + i]);
        } else {
            // 否则，按照 beta * C + float16_c 的值更新 C 的元素
            c[offset] = beta * c[offset] + c10::convert<float>(float16_c[j * m + i]);
        }
    }
}



void gemm(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    const float alpha,
    const at::Half *a, int64_t lda,
    const at::Half *b, int64_t ldb,
    const float beta,
    float *c, int64_t ldc) {
// 规范化 gemm 函数的最后几个维度参数
internal::normalize_last_dims(transa, transb, m, n, k, &lda, &ldb, &ldc);
#ifdef MKL_HAS_SHGEMM
// 如果定义了 MKL_HAS_SHGEMM 宏，则调用封装了 MKL 混合精度 gemm 的 gemm 函数
if (use_blas_gemm(transa, transb, m, n, k, lda, ldb, ldc)) {
    int m_ = m, n_ = n, k_ = k, lda_ = lda, ldb_ = ldb, ldc_ = ldc;
    mkl_gemm_f16f16f32(transa, transb, m_, n_, k_, alpha, a, lda_, b, ldb_, beta, c, ldc_);
    return;
}
#endif
// 如果未定义 MKL_HAS_SHGEMM 宏，则使用 fallback 路径：
// 首先以 beta = 0 计算 gemm，然后在完整精度下添加 c
int64_t c_size = n * m;
// 创建一个存储 BFloat16 类型数据的向量 bfloat_c，并初始化为零
std::vector<at::BFloat16> bfloat_c(c_size, 0.f);
// 调用 gemm_stub 函数进行 gemm 计算，使用 BFloat16 类型数据，并以 beta = 0 初始化结果存储在 bfloat_c 中
gemm_stub(
    at::kCPU, at::kBFloat16,
    transa, transb, m, n, k, alpha, a, lda, b, ldb, 0.f, bfloat_c.data(), m);
// 遍历结果矩阵 C
for (const auto j : c10::irange(n)) {
    for (const auto i : c10::irange(m)) {
        auto offset = j * ldc + i;
        // 如果 beta = 0，则不会从 C 中传播 NaN
        if (beta == 0.f) {
            // 将 BFloat16 类型的数据转换为 float，并存储在 C 中
            c[offset] = c10::convert<float>(bfloat_c[j * m + i]);
        } else {
            // 否则，按照 beta * C + bfloat_c 的值更新 C 的元素
            c[offset] = beta * c[offset] + c10::convert<float>(bfloat_c[j * m + i]);
        }
    }
}



void gemm(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    const int64_t alpha,
    const int64_t *a, int64_t lda,
    const int64_t *b, int64_t ldb,
    const int64_t beta,
    int64_t *c, int64_t ldc) {
// 规范化 gemm 函数的最后几个维度参数
internal::normalize_last_dims(transa, transb, m, n, k, &lda, &ldb, &ldc);
#ifdef USE_FBGEMM
// 如果定义了 USE_FBGEMM 宏，且 alpha = 1 且 beta = 0 或 beta = 1，则使用下述算法：
// 通过转置 A 和 B，计算 C^T (n x m) = B^T (n x k) * A^T (k x m)
if (alpha == 1 && (beta == 0 || beta == 1)) {
    // 在此函数中，我们假设 A 和 B 是按照列主序列（column-major ordering）存储的，
    // 但是 C 的存储顺序按照 FORTRAN 传统是行主序列（row-major ordering）。
    // 为了在传递给 FBGEMM 时视 C^T 为行主序列，我们使用转置的方法。
    // 这种方式下，我们把 C^T 视为行主序列，传递给 FBGEMM。
    // 使用fbgemm库中的cblas_gemm_i64_i64acc函数进行整型矩阵乘法运算
    // 将transb参数转换为fbgemm库的格式
    // 将transa参数转换为fbgemm库的格式
    // 设置矩阵运算的维度参数：n为结果矩阵的行数，m为结果矩阵的列数，k为乘积运算的中间维度
    // b: 输入矩阵B的数据指针
    // ldb: 矩阵B的leading dimension（领先维度，即B的列数）
    // a: 输入矩阵A的数据指针
    // lda: 矩阵A的leading dimension（领先维度，即A的列数）
    // beta == 1: 是否对结果矩阵C进行beta倍的累加操作
    // c: 输出结果矩阵C的数据指针
    // ldc: 矩阵C的leading dimension（领先维度，即C的列数）
    fbgemm::cblas_gemm_i64_i64acc(
        to_fbgemm(transb),
        to_fbgemm(transa),
        n,
        m,
        k,
        b,
        ldb,
        a,
        lda,
        beta == 1,
        c,
        ldc);
    // 函数执行完毕，返回调用处
    return;
}
#endif

// 调用 gemm_stub 函数进行矩阵乘法运算，使用 CPU 上的长整型数据类型
gemm_stub(
    kCPU, kLong,
    transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

template <typename scalar_t>
static void gemm_batched_mkl_impl(
      TransposeType transa, TransposeType transb,
      int64_t batch_size, int64_t m, int64_t n, int64_t k,
      scalar_t alpha,
      const scalar_t **a, int64_t lda,
      const scalar_t **b, int64_t ldb,
      scalar_t beta,
      scalar_t **c, int64_t ldc) {
  // 遍历批处理中的每个子批次
  for (int64_t i = 0; i < batch_size;) {
    // 计算当前子批次的大小，不超过 INT_MAX
    int sub_batch = std::min(batch_size - i, int64_t{INT_MAX});
    // 调用 MKL 库进行批量矩阵乘法运算
    mkl_gemm_batched(transa, transb, sub_batch, m, n, k, alpha,
                     &a[i], lda, &b[i], ldb, beta, &c[i], ldc);
    // 更新子批次计数
    i += sub_batch;
  }
}

template <typename scalar_t>
using is_blas_library_type = std::integral_constant<bool,
    std::is_same<scalar_t, double>::value ||
    std::is_same<scalar_t, float>::value ||
    std::is_same<scalar_t, c10::complex<double>>::value ||
    std::is_same<scalar_t, c10::complex<float>>::value>;

template <typename scalar_t>
void gemm_batched_generic(
    TransposeType transa, TransposeType transb,
    int64_t batch_size, int64_t m, int64_t n, int64_t k,
    scalar_t alpha,
    const scalar_t **a, int64_t lda,
    const scalar_t **b, int64_t ldb,
    scalar_t beta,
    scalar_t **c, int64_t ldc) {
  // 遍历批处理中的每个批次
  for (const auto batch : c10::irange(batch_size)) {
    // 调用通用的矩阵乘法函数 gemm
    gemm(transa, transb, m, n, k, alpha, a[batch], lda, b[batch], ldb, beta, c[batch], ldc);
  }
}

template <typename scalar_t>
void gemm_batched(
    TransposeType transa, TransposeType transb,
    int64_t batch_size, int64_t m, int64_t n, int64_t k,
    scalar_t alpha,
    const scalar_t **a, int64_t lda,
    const scalar_t **b, int64_t ldb,
    scalar_t beta,
    scalar_t **c, int64_t ldc) {
  // 如果批处理大小为 1，直接调用 gemm 函数进行矩阵乘法
  if (batch_size == 1) {
    return gemm(transa, transb, m, n, k, alpha, a[0], lda, b[0], ldb, beta, c[0], ldc);
  }

  // 判断是否可以使用 BLAS 库进行优化
  if constexpr (AT_MKL_ENABLED() && is_blas_library_type<scalar_t>::value) {
    // 规范化最后维度的参数
    internal::normalize_last_dims(transa, transb, m, n, k, &lda, &ldb, &ldc);
    // 判断是否使用 BLAS 库的 gemm 函数
    if (use_blas_gemm(transa, transb, m, n, k, lda, ldb, ldc)) {
      // 使用 MKL 库进行批量矩阵乘法运算
      gemm_batched_mkl_impl(
          transa, transb, batch_size, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    } else {
      // 否则调用通用的批量矩阵乘法函数
      gemm_batched_generic(
          transa, transb, batch_size, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
  } else {
    // 如果不能使用 BLAS 库，调用通用的批量矩阵乘法函数
    gemm_batched_generic(
        transa, transb, batch_size, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  }
}

template <typename scalar_t>
void gemm_batched_with_stride_generic(
    TransposeType transa, TransposeType transb,
    int64_t batch_size, int64_t m, int64_t n, int64_t k,
    scalar_t alpha,
    const scalar_t *a, int64_t lda, int64_t batch_stride_a,
    const scalar_t *b, int64_t ldb, int64_t batch_stride_b,
    scalar_t beta,
    scalar_t *c, int64_t ldc, int64_t batch_stride_c) {
  // 遍历批处理中的每个批次
  for (const auto batch : c10::irange(batch_size)) {
    // 计算当前批次对应的起始地址
    const auto a_batch = a + batch_stride_a * batch;
    # 计算当前批次的矩阵 B 的起始地址
    const auto b_batch = b + batch_stride_b * batch;
    # 计算当前批次的矩阵 C 的起始地址
    const auto c_batch = c + batch_stride_c * batch;
    # 执行通用矩阵乘法运算（GEMM）
    gemm(transa, transb, m, n, k, alpha, a_batch, lda, b_batch, ldb, beta, c_batch, ldc);
// 定义了一个模板函数 gemm_batched_with_stride，用于批量处理矩阵乘法，支持不同数据类型的标量
template <typename scalar_t>
void gemm_batched_with_stride(
    TransposeType transa, TransposeType transb,  // 矩阵 A 和 B 是否需要转置
    int64_t batch_size, int64_t m, int64_t n, int64_t k,  // 批次大小以及矩阵 A、B 和 C 的维度
    scalar_t alpha,  // 乘法的比例因子
    const scalar_t *a, int64_t lda, int64_t batch_stride_a,  // 矩阵 A 数据指针、每批次 A 数据的跨度
    const scalar_t *b, int64_t ldb, int64_t batch_stride_b,  // 矩阵 B 数据指针、每批次 B 数据的跨度
    scalar_t beta,  // C 矩阵的比例因子
    scalar_t *c, int64_t ldc, int64_t batch_stride_c) {  // 矩阵 C 数据指针、每批次 C 数据的跨度

  // 如果 batch_size 等于 1，则直接调用 gemm 函数处理单个矩阵乘法并返回
  if (batch_size == 1) {
    return gemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  }

  // 如果支持 MKL 并且标量类型是 BLAS 库支持的类型
  if constexpr (AT_MKL_ENABLED() && is_blas_library_type<scalar_t>::value) {
    // 标准化最后的维度，确保和 BLAS 接口匹配
    internal::normalize_last_dims(transa, transb, m, n, k, &lda, &ldb, &ldc);

    // 判断是否可以使用 BLAS 提供的 GEMM 函数
    if (use_blas_gemm(transa, transb, m, n, k, lda, ldb, ldc)) {
      // 使用 SmallBuffer 存储每个批次的 A、B、C 数据指针
      c10::SmallBuffer<const scalar_t*, 16> a_ptrs(batch_size);
      c10::SmallBuffer<const scalar_t*, 16> b_ptrs(batch_size);
      c10::SmallBuffer<scalar_t*, 16> c_ptrs(batch_size);

      // 遍历每个批次，设置相应的 A、B、C 数据指针
      for (const auto batch : c10::irange(batch_size)) {
        a_ptrs[batch] = a + batch_stride_a * batch;
        b_ptrs[batch] = b + batch_stride_b * batch;
        c_ptrs[batch] = c + batch_stride_c * batch;
      }

      // 调用 MKL 实现的批量矩阵乘法函数
      gemm_batched_mkl_impl(
          transa, transb, batch_size, m, n, k, alpha, a_ptrs.data(), lda,
          b_ptrs.data(), ldb, beta, c_ptrs.data(), ldc);
    } else {
      // 如果无法使用 BLAS 提供的 GEMM 函数，则调用通用的批量矩阵乘法实现
      gemm_batched_with_stride_generic(
          transa, transb, batch_size, m, n, k, alpha, a, lda, batch_stride_a,
          b, ldb, batch_stride_b, beta, c, ldc, batch_stride_c);
    }
  } else {
    // 如果不支持 MKL 或者标量类型不是 BLAS 库支持的类型，则调用通用的批量矩阵乘法实现
    gemm_batched_with_stride_generic(transa, transb, batch_size, m, n, k, alpha,
                                     a, lda, batch_stride_a, b, ldb, batch_stride_b,
                                     beta, c, ldc, batch_stride_c);
  }
}

// 实例化模板函数 gemm_batched_with_stride，支持所有标量类型的批量矩阵乘法
#define INSTANTIATE_BATCHED_GEMM(scalar_t, DType)               \
  template void gemm_batched_with_stride(                       \
      TransposeType transa, TransposeType transb,               \
      int64_t batch_size, int64_t m, int64_t n, int64_t k,      \
      scalar_t alpha,                                           \
      const scalar_t *a, int64_t lda, int64_t batch_stride_a,   \
      const scalar_t *b, int64_t ldb, int64_t batch_stride_b,   \
      scalar_t beta,                                            \
      scalar_t *c, int64_t ldc, int64_t batch_stride_c);

// 对所有支持的标量类型实例化 gemm_batched_with_stride 函数
AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_EXCEPT_COMPLEX_HALF_F8NZ(INSTANTIATE_BATCHED_GEMM)

// 定义分发的 axpy_stub 函数
DEFINE_DISPATCH(axpy_stub);
void axpy(int64_t n, double a, const double *x, int64_t incx, double *y, int64_t incy) {
  // 如果 n 等于 1，则将步长 incx 和 incy 设置为 1
  if(n == 1)
  {
    incx = 1;
    incy = 1;
  }
  // 如果使用 BLAS 并且 n、incx、incy 均不超过 INT_MAX
  #if AT_BUILD_WITH_BLAS()
  if( (n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX) )
  {
    // 将 n、incx、incy 转换为 int 类型
    int i_n = (int)n;
    int i_incx = (int)incx;
    int i_incy = (int)incy;
    // 根据 C10_IOS 宏选择调用 cblas_daxpy 或 daxpy_
    #if C10_IOS
    cblas_daxpy(i_n, a, x, i_incx, y, i_incy);
    #else
    daxpy_(&i_n, &a, x, &i_incx, y, &i_incy);
    #endif
    // 函数执行完毕，返回
    return;
  }
  #endif
  // 调用 axpy_stub 函数处理，传递所需参数
  axpy_stub(
      kCPU, at::kDouble,
      n, a, x, incx, y, incy);
}

void axpy(int64_t n, float a, const float *x, int64_t incx, float *y, int64_t incy) {
  // 如果 n 等于 1，则将步长 incx 和 incy 设置为 1
  if(n == 1)
  {
    incx = 1;
    incy = 1;
  }
  // 如果使用 BLAS 并且 n、incx、incy 均不超过 INT_MAX
  #if AT_BUILD_WITH_BLAS()
  if( (n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX) )
  {
    // 将 n、incx、incy 转换为 int 类型
    int i_n = (int)n;
    int i_incx = (int)incx;
    int i_incy = (int)incy;
    // 根据 C10_IOS 宏选择调用 cblas_saxpy 或 saxpy_
    #if C10_IOS
    cblas_saxpy(i_n, a, x, i_incx, y, i_incy);
    #else
    saxpy_(&i_n, &a, x, &i_incx, y, &i_incy);
    #endif
    // 函数执行完毕，返回
    return;
  }
  #endif
  // 调用 axpy_stub 函数处理，传递所需参数
  axpy_stub(
      kCPU, at::kFloat,
      n, a, x, incx, y, incy);
}

void axpy(int64_t n, c10::complex<double> a, const c10::complex<double> *x, int64_t incx, c10::complex<double> *y, int64_t incy) {
  // 如果 n 等于 1，则将步长 incx 和 incy 设置为 1
  if(n == 1)
  {
    incx = 1;
    incy = 1;
  }
  // 如果使用 BLAS 并且 n、incx、incy 均不超过 INT_MAX
  #if AT_BUILD_WITH_BLAS()
  if( (n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX) )
  {
    // 将 n、incx、incy 转换为 int 类型
    int i_n = (int)n;
    int i_incx = (int)incx;
    int i_incy = (int)incy;
    // 根据 C10_IOS 宏选择调用 cblas_zaxpy 或 zaxpy_
    #if C10_IOS
    cblas_zaxpy(i_n, &a, x, i_incx, y, i_incy);
    #else
    zaxpy_(&i_n, &a, x, &i_incx, y, &i_incy);
    #endif
    // 函数执行完毕，返回
    return;
  }
  #endif
  // 调用 axpy_stub 函数处理，传递所需参数
  axpy_stub(
      kCPU, at::kComplexDouble,
      n, a, x, incx, y, incy);
}

void axpy(int64_t n, c10::complex<float> a, const c10::complex<float> *x, int64_t incx, c10::complex<float> *y, int64_t incy) {
  // 如果 n 等于 1，则将步长 incx 和 incy 设置为 1
  if(n == 1)
  {
    incx = 1;
    incy = 1;
  }
  // 如果使用 BLAS 并且 n、incx、incy 均不超过 INT_MAX
  #if AT_BUILD_WITH_BLAS()
  if( (n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX) )
  {
    // 将 n、incx、incy 转换为 int 类型
    int i_n = (int)n;
    int i_incx = (int)incx;
    int i_incy = (int)incy;
    // 根据 C10_IOS 宏选择调用 cblas_caxpy 或 caxpy_
    #if C10_IOS
    cblas_caxpy(i_n, &a, x, i_incx, y, i_incy);
    #else
    caxpy_(&i_n, &a, x, &i_incx, y, &i_incy);
    #endif
    // 函数执行完毕，返回
    return;
  }
  #endif
  // 调用 axpy_stub 函数处理，传递所需参数
  axpy_stub(
      kCPU, at::kComplexFloat,
      n, a, x, incx, y, incy);
}

DEFINE_DISPATCH(copy_stub);

void copy(int64_t n, const double *x, int64_t incx, double *y, int64_t incy) {
  // 如果 n 等于 1，则将步长 incx 和 incy 设置为 1
  if(n == 1)
  {
    incx = 1;
    incy = 1;
  }
  // 如果使用 BLAS 并且 n、incx、incy 均不超过 INT_MAX
  #if AT_BUILD_WITH_BLAS()
  if( (n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX) ) {
    // 将 n、incx、incy 转换为 int 类型
    int i_n = (int)n;
    int i_incx = (int)incx;
    int i_incy = (int)incy;
    // 根据 C10_IOS 宏选择调用 cblas_dcopy 或 dcopy_
    #if C10_IOS
    cblas_dcopy(i_n, x, i_incx, y, i_incy);
    #else
    dcopy_(&i_n, x, &i_incx, y, &i_incy);
    #endif
    // 函数执行完毕，返回
    return;
  }
  #endif
  // 调用 copy_stub 函数处理，传递所需参数
  copy_stub(
      kCPU, at::kDouble,
      n, x, incx, y, incy);
}

void copy(int64_t n, const float *x, int64_t incx, float *y, int64_t incy) {
  // 如果 n 等于 1，则将步长 incx 设置为 1
  if(n == 1)
  {
    incx = 1;
    incy = 1; // Note: there is a missing increment statement for `incy` in the original code
  }
  // 如果使用 BLAS 并且 n、incx、incy 均不超过 INT_MAX
  #if AT_BUILD_WITH_BLAS()
  if( (n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX) ) {
    // 将 n、incx、incy 转换为 int 类型
    int i_n = (int)n;
    int i_incx = (int)incx;
    int i_incy = (int)incy;
    // 根据 C10_IOS 宏选择调用 cblas_scopy 或 scopy_
    #if C10_IOS
    cblas_scopy(i_n, x, i_incx, y, i_incy);
    #else
    scopy_(&i_n, x, &i_incx
    incy = 1;
  }
  #if AT_BUILD_WITH_BLAS()
  #if AT_BUILD_WITH_BLAS()  // 检查是否构建了 BLAS 支持
  if( (n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX) ) {
    // 将输入参数转换为整数，以便与 BLAS 接口兼容
    int i_n = (int)n;
    int i_incx = (int)incx;
    int i_incy = (int)incy;
    #if C10_IOS
    cblas_scopy(i_n, x, i_incx, y, i_incy);  // 调用 BLAS 的单精度浮点数复制函数
    #else
    scopy_(&i_n, x, &i_incx, y, &i_incy);    // 使用 Fortran 接口的复制函数
    #endif
    return;  // 返回，函数执行完毕
  }
  #endif
  copy_stub(
      kCPU, at::kFloat,
      n, x, incx, y, incy);  // 调用替代的复制函数 stub
}

// 定义一个函数 copy，用于复制复数数组 x 到数组 y，支持不同步长的情况
void copy(int64_t n, const c10::complex<double> *x, int64_t incx, c10::complex<double> *y, int64_t incy) {
  // 如果复制的元素个数 n 为 1，则设置步长为 1
  if(n == 1)
  {
    incx = 1;
    incy = 1;
  }
  // 如果编译时使用了 BLAS 库
  #if AT_BUILD_WITH_BLAS()
  // 确保参数 n, incx, incy 都在合理范围内
  if( (n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX) ) {
    // 将参数转换为整型
    int i_n = (int)n;
    int i_incx = (int)incx;
    int i_incy = (int)incy;
    // 根据平台选择调用不同的 BLAS 函数处理复制操作
    #if C10_IOS
    cblas_zcopy(i_n, x, i_incx, y, i_incy);
    #else
    zcopy_(&i_n, x, &i_incx, y, &i_incy);
    #endif
    // 函数执行完毕，直接返回
    return;
  }
  #endif
  // 若未使用 BLAS 或参数超出整型范围，则调用 copy_stub 处理复制操作
  copy_stub(
      kCPU, at::kComplexDouble,
      n, x, incx, y, incy);
}

// 定义一个函数 copy，用于复制浮点复数数组 x 到数组 y，支持不同步长的情况
void copy(int64_t n, const c10::complex<float> *x, int64_t incx, c10::complex<float> *y, int64_t incy){
  // 如果复制的元素个数 n 为 1，则设置步长为 1
  if(n == 1)
  {
    incx = 1;
    incy = 1;
  }
  // 如果编译时使用了 BLAS 库
  #if AT_BUILD_WITH_BLAS()
  // 确保参数 n, incx, incy 都在合理范围内
  if( (n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX) ) {
    // 将参数转换为整型
    int i_n = (int)n;
    int i_incx = (int)incx;
    int i_incy = (int)incy;
    // 根据平台选择调用不同的 BLAS 函数处理复制操作
    #if C10_IOS
    cblas_ccopy(i_n, x, i_incx, y, i_incy);
    #else
    ccopy_(&i_n, x, &i_incx, y, &i_incy);
    #endif
    // 函数执行完毕，直接返回
    return;
  }
  #endif
  // 若未使用 BLAS 或参数超出整型范围，则调用 copy_stub 处理复制操作
  copy_stub(
      kCPU, at::kComplexFloat,
      n, x, incx, y, incy);
}

}  // namespace at::native::cpublas


这段代码定义了两个函数 `copy`，用于复制复数数组（分别是双精度和单精度复数）。这两个函数根据参数是否为 1 来调整步长，并根据编译时是否启用 BLAS 库选择不同的复制方式。
```