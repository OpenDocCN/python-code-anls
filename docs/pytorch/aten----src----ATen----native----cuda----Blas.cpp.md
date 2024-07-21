# `.\pytorch\aten\src\ATen\native\cuda\Blas.cpp`

```
// 包含 C++ 标准库头文件和 PyTorch 相关头文件
#include <cstdint>
#include <c10/util/Exception.h>
#include <c10/core/Scalar.h>
#include <c10/core/ScalarType.h>
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/core/NamedTensor.h>
#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <ATen/OpMathType.h>
#include <ATen/TensorUtils.h>
#include <ATen/cuda/CUDABlas.h>
#include <ATen/cuda/tunable/Tunable.h>
#include <ATen/cuda/tunable/TunableGemm.h>
#include <ATen/native/Resize.h>
#include <c10/util/MaybeOwned.h>
#include <ATen/native/cuda/RowwiseScaledMM.h>

// 如果没有定义 AT_PER_OPERATOR_HEADERS，则包含所有操作相关头文件，否则包含指定的操作头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_addmm_activation_native.h>
#include <ATen/ops/_efficientzerotensor.h>
#include <ATen/ops/_scaled_mm_native.h>
#include <ATen/ops/_unsafe_view_native.h>
#include <ATen/ops/abs.h>
#include <ATen/ops/addmm_native.h>
#include <ATen/ops/addmv_native.h>
#include <ATen/ops/baddbmm_native.h>
#include <ATen/ops/bmm_native.h>
#include <ATen/ops/copy_native.h>
#include <ATen/ops/dot_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/gelu.h>
#include <ATen/ops/max.h>
#include <ATen/ops/mm_native.h>
#include <ATen/ops/mul.h>
#include <ATen/ops/relu.h>
#include <ATen/ops/ones.h>
#include <ATen/ops/scalar_tensor_native.h>
#include <ATen/ops/vdot_native.h>
#endif

namespace at::native {

namespace {

// 解析是否需要共轭变换，并返回对应的 Tensor
c10::MaybeOwned<Tensor> inline resolve_conj_if_indicated(const Tensor& tensor, bool resolve_conj) {
  // 如果需要共轭变换并且当前张量是共轭的，则解析并返回共轭解析后的张量
  if (resolve_conj && tensor.is_conj()) {
    return c10::MaybeOwned<Tensor>::owned(tensor.resolve_conj());
  } else {
    // 否则直接返回当前张量的 borrow 引用
    return c10::MaybeOwned<Tensor>::borrowed(tensor);
  }
}

// 准备张量以便于传递给 CUBlas 进行操作，并返回对应的 Tensor
c10::MaybeOwned<Tensor> inline prepare_matrix_for_cublas(const Tensor& tensor, bool& transpose_tensor, bool transpose_result) {
  // 如果张量是非重叠且密集的情况（常见情况）
  if (tensor.is_non_overlapping_and_dense()) {
    // 如果张量是连续的，则不需要转置
    transpose_tensor = tensor.is_contiguous();
    // 根据是否需要转置结果，解析是否需要共轭并返回对应的 Tensor
    return resolve_conj_if_indicated(tensor, transpose_result ? transpose_tensor : !transpose_tensor);
  }

  // 获取张量的步长和大小
  IntArrayRef tensor_strides = tensor.strides();
  IntArrayRef tensor_sizes = tensor.sizes();

  // 如果张量的步长和大小满足特定条件，选择是否需要转置并返回对应的 Tensor
  if ((tensor_strides[0] == 1) && (tensor_strides[1] >= std::max<int64_t>(1, tensor_sizes[0]))) {
    transpose_tensor = false;
    return resolve_conj_if_indicated(tensor, !transpose_result);
  } else if ((tensor_strides[1] == 1) && (tensor_strides[0] >= std::max<int64_t>(1, tensor_sizes[1]))) {
    transpose_tensor = true;
    return resolve_conj_if_indicated(tensor, transpose_result);
  } else {
    // 如果以上条件都不满足，则默认需要转置并返回一个连续内存格式的张量的拷贝
    transpose_tensor = true;
    return c10::MaybeOwned<Tensor>::owned(tensor.clone(at::MemoryFormat::Contiguous));
  }
}
c10::MaybeOwned<Tensor> inline prepare_matrix_for_cublas(const Tensor& tensor, bool& transpose_tensor) {
  // 检查是否为非重叠且稠密的张量，是常见情况
  if (tensor.is_non_overlapping_and_dense()) {
      // 若张量是连续的，则不需转置
      transpose_tensor = tensor.is_contiguous();
      // 返回解析后的张量，考虑是否需要共轭
      return resolve_conj_if_indicated(tensor, true);
  }

  // 获取张量的步长和尺寸信息
  IntArrayRef tensor_strides = tensor.strides();
  IntArrayRef tensor_sizes = tensor.sizes();

  // 检查是否满足特定条件来确定是否需要转置
  if ((tensor_strides[0] == 1) && (tensor_strides[1] >= std::max<int64_t>(1, tensor_sizes[0]))) {
    // 不需要转置
    transpose_tensor = false;
    // 返回解析后的张量，考虑是否需要共轭
    return resolve_conj_if_indicated(tensor, true);
  } else if ((tensor_strides[1] == 1) && (tensor_strides[0] >= std::max<int64_t>(1, tensor_sizes[1]))) {
    // 需要转置
    transpose_tensor = true;
    // 返回解析后的张量，考虑是否需要共轭
    return resolve_conj_if_indicated(tensor, true);
  } else {
    // 默认情况下需要转置
    transpose_tensor = true;
    // 返回一个新的张量副本，保证内存布局是连续的
    return c10::MaybeOwned<Tensor>::owned(tensor.clone(at::MemoryFormat::Contiguous));
  }
}

struct cublasCommonArgs {
  // 构造函数，初始化 Cublas 操作的参数
  cublasCommonArgs(const Tensor& mat1, const Tensor& mat2, Tensor& c) {
    bool transpose_result, transpose_mat1, transpose_mat2;
    // 准备结果矩阵
    result = prepare_matrix_for_cublas(c, transpose_result);
    // 准备矩阵 A
    mata = prepare_matrix_for_cublas(transpose_result ? mat2 : mat1, transpose_mat1, transpose_result);
    // 准备矩阵 B
    matb = prepare_matrix_for_cublas(transpose_result ? mat1 : mat2, transpose_mat2, transpose_result);

    // 根据是否转置结果，调整矩阵的尺寸信息
    auto mat1_sizes = mat1.sizes();
    auto mat2_sizes = mat2.sizes();
    if (transpose_result) {
      transpose_mat1 = !transpose_mat1;
      transpose_mat2 = !transpose_mat2;
      mat1_sizes = mata->sizes();
      mat2_sizes = matb->sizes();
    }

    // 设置矩阵操作的维度和尺寸信息
    m = mat1_sizes[transpose_result ? 1 : 0];
    k = mat1_sizes[transpose_result ? 0 : 1];
    n = mat2_sizes[transpose_result ? 0 : 1];
    lda = mata->stride((transpose_mat1 == transpose_result) ? 1 : 0);
    ldb = matb->stride((transpose_mat2 == transpose_result) ? 1 : 0);
    result_ld = result->stride(transpose_result ? 0 : 1);
    // 设置矩阵 A 和 B 的转置标志
    transa = transpose_mat1 ?  mata->is_conj() ? 'c' : 't' : 'n';
    transb = transpose_mat2 ?  matb->is_conj() ? 'c' : 't' : 'n';
  }
  char transa, transb; // 矩阵 A 和 B 的转置标志
  int64_t m, n, k; // 矩阵的维度信息
  int64_t lda, ldb, result_ld; // 矩阵操作时的步长信息
  c10::MaybeOwned<Tensor> mata, matb, result; // 矩阵 A、B 和结果矩阵的可能拥有的张量
};

namespace {
// 准备用于 Cublas 的批量矩阵操作的函数
c10::MaybeOwned<Tensor> prepare_batch_matrix_for_cublas(const Tensor& tensor, bool& transpose_tensor, int64_t& ld_tensor, bool transpose_result, int64_t m, int64_t n) {
  // 获取张量的步长信息
  IntArrayRef tensor_strides = tensor.strides();
  c10::MaybeOwned<Tensor> tensor_;

  // 根据转置结果确定快速维度和主导维度
  int fast_dim = transpose_result ? 2 : 1;
  int leading_dim = transpose_result ? 1 : 2;

  // 检查是否符合特定条件来决定是否需要转置
  if (tensor_strides[fast_dim] == 1 &&
    (tensor_strides[leading_dim] >= std::max<int64_t>(1, m))) {
    // 不需要转置
    transpose_tensor = false;
    // 解析张量，考虑是否需要共轭
    tensor_ = resolve_conj_if_indicated(tensor, true);
    // 设置主导维度的步长信息
    ld_tensor = tensor_->strides()[leading_dim];
  } else if ((tensor_strides[leading_dim] == 1) &&
    (tensor_strides[fast_dim] >= std::max<int64_t>(1, n))) {
    // 需要转置
    transpose_tensor = true;
    // 解析张量，考虑是否需要共轭
    tensor_ = resolve_conj_if_indicated(tensor, false);
    // 设置快速维度的步长信息
    ld_tensor = tensor_->strides()[fast_dim];
  } else {
    // 默认情况下需要转置
    // 将 transpose_result 赋值给 transpose_tensor
    transpose_tensor = !transpose_result;
    // gemm 调用要求 leading dimension 和 stride 参数不能为零
    bool is_stride_non_zero = tensor.strides()[1] != 0 && tensor.strides()[2] != 0;
    // 检查 tensor 是否是连续的，并且其 stride 参数非零
    if (tensor.is_contiguous() && is_stride_non_zero) {
      // 如果满足条件，则根据 transpose_result 解析 tensor 是否需要共轭，并赋值给 tensor_
      tensor_ = resolve_conj_if_indicated(tensor, transpose_result);
    } else {
      // 如果不满足条件，则克隆 tensor 并确保其内存格式为连续的，然后赋值给 tensor_
      tensor_ = c10::MaybeOwned<Tensor>::owned(tensor.clone(at::MemoryFormat::Contiguous));
    }
    // 获取 tensor_ 的 leading dimension，赋值给 ld_tensor
    ld_tensor = tensor_->strides()[1];
  }

  // 返回处理后的 tensor_
  return tensor_;
// 命名空间内部结束

namespace {

// 激活函数枚举，包括无激活、RELU 和 GELU
enum class Activation {
  None,
  RELU,
  GELU,
};

// 将 Activation 枚举转换为 cuda::blas::GEMMAndBiasActivationEpilogue 类型的参数
cuda::blas::GEMMAndBiasActivationEpilogue activation_to_gemm_and_blas_arg(Activation a) {
  switch (a) {
    case Activation::None:
      return cuda::blas::GEMMAndBiasActivationEpilogue::None;
    case Activation::RELU:
      return cuda::blas::GEMMAndBiasActivationEpilogue::RELU;
    case Activation::GELU:
      return cuda::blas::GEMMAndBiasActivationEpilogue::GELU;
    default:
      TORCH_CHECK(false);  // 如果出现未知的 Activation 类型，则抛出错误
      return cuda::blas::GEMMAndBiasActivationEpilogue::None;
  }
}

// 获取环境变量 DISABLE_ADDMM_CUDA_LT 的值，用于确定是否禁用 addmm CUDA LT
static bool getDisableAddmmCudaLt() {
    static const char* env_value = std::getenv("DISABLE_ADDMM_CUDA_LT");
#ifdef USE_ROCM
    // 对于 ROCm 构建，如果启用了可调整操作，则优先级高于 hipBLASLT (启发式)
    // 当前的可调整操作不是 hipblaslt 路径 (gemm_and_bias)
    auto tuning_ctx = at::cuda::tunable::getTuningContext();
    if (tuning_ctx->IsTunableOpEnabled()) {
      return true;
    }
    // 允许使用 CUDA 和 HIP 的环境变量名称
    // 对于 ROCm 构建，默认情况下禁用
    if (env_value == nullptr) {
        env_value = std::getenv("DISABLE_ADDMM_HIP_LT");
    }
    // 如果环境变量值为 "0"，则返回 false，否则返回 true
    if (env_value != nullptr && strcmp(env_value, "0") == 0) {
      return false;
    }
    return true;
#else
    // 如果环境变量值为 "1"，则返回 true，否则返回 false
    if (env_value != nullptr && strcmp(env_value, "1") == 0) {
      return true;
    }
    return false;
#endif
}

#ifdef USE_ROCM
// 检查当前的 HIP 设备架构是否支持 hipBLASLt
static bool isSupportedHipLtROCmArch(int index) {
    hipDeviceProp_t* prop = at::cuda::getDeviceProperties(index);
    std::string device_arch = prop->gcnArchName;
    // 支持的 ROCm 架构列表
    static const std::vector<std::string> archs = {"gfx90a", "gfx940", "gfx941", "gfx942"};
    // 遍历架构列表，检查当前设备架构是否匹配
    for (std::string arch : archs) {
        size_t substring = device_arch.find(arch);
        if (substring != std::string::npos) {
            return true;  // 如果找到匹配的架构，则返回 true
        }
    }
    // 如果不支持当前架构，则抛出错误信息
    TORCH_CHECK(false, "Attempting to use hipBLASLt on a unsupported architecture!");
    return false;
}
#endif

// 在 CUDA 实现中执行 addmm 操作，并支持激活函数
Tensor& addmm_out_cuda_impl(Tensor& result, const Tensor& self, const Tensor& mat1, const Tensor& mat2, const Scalar& beta, const Scalar& alpha, Activation activation=Activation::None) {
  // 确保 mat1 和 mat2 是二维张量
  TORCH_CHECK(mat1.dim() == 2 && mat2.dim() == 2, "tensors must be 2-D");
  // 确保 mat1 和 mat2 具有相同的数据类型
  TORCH_CHECK(
    mat1.dtype() == mat2.dtype(),
    "expected mat1 and mat2 to have the same dtype, but got: ", mat1.dtype(), " != ", mat2.dtype()
  )

  // 检查所有张量是否在同一 GPU 上
  TensorArg targs[]{{result, "out", 0}, {self, "self", 1}, {mat1, "mat1", 2}, {mat2, "mat2", 3}};
  checkAllSameGPU(__func__, targs);

  // 获取 mat1 和 mat2 的尺寸
  IntArrayRef mat1_sizes = mat1.sizes();
  IntArrayRef mat2_sizes = mat2.sizes();
  IntArrayRef self__sizes;
  bool useLtInterface = false;

  // 获取是否禁用了 addmm CUDA LT 的状态
  static bool disable_addmm_cuda_lt = getDisableAddmmCudaLt();

  // 获取 self 张量的标量类型
  at::ScalarType scalar_type = self.scalar_type();

  // 如果 result 不是 self，则需要进行额外的操作
  c10::MaybeOwned<Tensor> self_;
  if (&result != &self) {
#if (defined(CUDA_VERSION) && (CUDA_VERSION >= 11040)) || defined(USE_ROCM)
    // 如果 mat2 只有一行或一列，cublasLtMatmulAlgoGetHeuristic 会报 CUBLAS_STATUS_INVALID_VALUE 错误
    // self.dim() == 1 && result.dim() == 2 && self.sizes()[0] == mat2_sizes[1] 是为了在 self 是偏置时使用 lt 接口
    // 对于 CUDA 11.4，启用 cublasLtMatmul
    // 最后两个条件是为了在从大张量切片出来时，跳过 16b transA 和非 trans-B 的 leading dim >> rows 的情况
    // 参见 fbcode/caffe2/test/test_linalg.py:test_corner_cases_of_cublasltmatmul
    if (!disable_addmm_cuda_lt) {
      // 检查是否满足使用 lt 接口的条件
      useLtInterface = beta.toComplexDouble() == 1.0 && self.dim() == 1 &&
          result.dim() == 2 && self.sizes()[0] == mat2_sizes[1] &&
          self.is_contiguous() && result.is_contiguous() &&
#ifdef USE_ROCM
          isSupportedHipLtROCmArch(self.device().index()) &&
          (scalar_type == at::ScalarType::Float ||
           scalar_type == at::ScalarType::Half ||
           scalar_type == at::ScalarType::BFloat16) &&
#else
          (scalar_type == at::ScalarType::Double ||
           scalar_type == at::ScalarType::Float ||
           scalar_type == at::ScalarType::Half ||
           scalar_type == at::ScalarType::BFloat16) &&
#endif
#if (defined(CUDA_VERSION) && CUDA_VERSION >= 12010 && !defined(USE_ROCM))
          mat2_sizes[0] > 1 && mat2_sizes[1] > 1;
#else
          mat2_sizes[0] > 1 && mat2_sizes[1] > 1 &&
          mat2_sizes[0] < 65535 * 32 && mat2_sizes[1] < 65535 * 32 &&
          mat1_sizes[0] < 65535 * 32 && mat1_sizes[1] < 65535 * 32 &&
          // 避免 leading dim >> rows 的 bug
          ((mat1.strides()[0] == 1 && mat1.strides()[1] == mat1_sizes[0]) ||
           (mat1.strides()[1] == 1 && mat1.strides()[0] == mat1_sizes[1]) ||
           (scalar_type != at::ScalarType::Half &&
            scalar_type != at::ScalarType::BFloat16)) &&
          ((mat2.strides()[0] == 1 && mat2.strides()[1] == mat2_sizes[0]) ||
           (mat2.strides()[1] == 1 && mat2.strides()[0] == mat2_sizes[1]) ||
           (scalar_type != at::ScalarType::Half &&
            scalar_type != at::ScalarType::BFloat16));
#endif
    }
#endif
    // 如果不使用 lt 接口，则扩展 self 的大小为 {mat1_sizes[0], mat2_sizes[1]}，用于 addmm 操作
    if (!useLtInterface) {
      self_ = expand_size(self, {mat1_sizes[0], mat2_sizes[1]}, "addmm");
    }
    // 获取扩展后 self_ 的大小
    self__sizes = self_->sizes();
  } else {
#if defined(USE_ROCM)
    // 对于 ROCm 平台，如果满足条件，使用 lt 接口
    useLtInterface = !disable_addmm_cuda_lt &&
        result.dim() == 2 && result.is_contiguous() &&
        isSupportedHipLtROCmArch(self.device().index()) &&
        (scalar_type == at::ScalarType::Float ||
          scalar_type == at::ScalarType::Half ||
          scalar_type == at::ScalarType::BFloat16);
#endif
    // 将 self 转换为 MaybeOwned<Tensor> 类型，并获取其大小
    self_ = c10::MaybeOwned<Tensor>::borrowed(self);
    self__sizes = self_->sizes();
    // 检查 result 的维度是否为 2
    TORCH_CHECK(result.dim() == 2, "tensors must be 2-D");
    // 检查 self_ 的第 0 维是否与 mat1 的第 0 维匹配
    TORCH_CHECK(self__sizes[0] == mat1_sizes[0], "self_ dim 0 must match mat1 dim 0");
    # 检查张量 self__sizes 的第二个维度是否与 mat2_sizes 的第二个维度相匹配，若不匹配则抛出错误信息
    TORCH_CHECK(self__sizes[1] == mat2_sizes[1], "self_ dim 1 must match mat2 dim 1");
  }

  # 如果 result 引用不是指向 self 的引用
  if (&result != &self) {
    # 调整 result 的大小为 mat1_sizes[0] 行，mat2_sizes[1] 列
    at::native::resize_output(result, {mat1_sizes[0], mat2_sizes[1]});
    # 如果 beta 的复双精度值不为 0，且不使用 Lt 接口
    if (beta.toComplexDouble() != 0.0 && !useLtInterface) {
      # 将 self_ 的数据复制到 result 中
      at::native::copy_(result, *self_);
    }
  }

  # 获取 result 的大小信息
  IntArrayRef result_sizes = result.sizes();
  # 如果 result 的第一维或第二维为 0，则直接返回 result
  if ((result_sizes[0] == 0) || (result_sizes[1] == 0)) {
    return result;
  }

  # 创建 cublasCommonArgs 对象 args，用于管理 mat1、mat2 和 result
  cublasCommonArgs args(mat1, mat2, result);

  # 如果 mat1 的元素个数为 0
  if (mat1.numel() == 0) {
    # 根据 beta 的值进行不同处理：
    # 当 beta==0 时，self 中的值应该被忽略，不应传播 nans 和 infs
    if (beta.toComplexDouble() == 0.) {
      # 将 result 的值全部置为 0
      return result.zero_();
    }
    # 否则，执行以下操作：
    # 扩展 self 的大小以匹配 result，然后使用 beta 创建一个标量张量并执行乘法操作
    return at::mul_out(
        result,
        self.expand(result.sizes()),
        at::native::scalar_tensor(
            beta,
            self.scalar_type(),
            c10::nullopt /* layout */,
            at::kCPU,
            c10::nullopt /* pin_memory */));
  }

  # 在调试模式下，断言 args.result 不是共轭的
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!args.result->is_conj());

  # 如果使用 Lt 接口
  if (useLtInterface) {
#if defined(USE_ROCM)
    // 如果定义了 USE_ROCM，则使用 ROCm 平台特定的代码路径
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        scalar_type,
        "addmm_cuda_lt",
        [&] {
          // 调用 ROCm 平台的 gemm_and_bias 函数进行矩阵乘法和偏置计算
          at::cuda::blas::gemm_and_bias<scalar_t>(
              args.transa == 't',
              args.transb == 't',
              args.m,
              args.n,
              args.k,
              alpha.to<at::opmath_type<scalar_t>>(),
              args.mata->const_data_ptr<scalar_t>(),
              args.lda,
              args.matb->const_data_ptr<scalar_t>(),
              args.ldb,
              // ROCm 上 mm 情况需要使用 hipblasLt 路径，将偏置指针设为 null 以避免精度问题
              (&result != &self) ? self.const_data_ptr<scalar_t>() : nullptr,
              args.result->data_ptr<scalar_t>(),
              args.result_ld,
              activation_to_gemm_and_blas_arg(activation)
          );
        });
#else
    // 如果未定义 USE_ROCM，则使用通用的 CUDA 平台路径
    auto activation_epilogue = activation_to_gemm_and_blas_arg(activation);
#if (defined(CUDA_VERSION) && (CUDA_VERSION < 11080))
    // 在 CUDA 版本小于 11.8 时，GELU 激活函数的后处理不受支持且无法编译
    // 在 CUDA 11.4 中观察到 GELU 后处理存在精度问题；为 CUDA 版本 < 11.8 禁用 GELU 后处理路径
    if (activation == Activation::GELU)
      activation_epilogue = cuda::blas::GEMMAndBiasActivationEpilogue::None;
#endif

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        scalar_type,
        "addmm_cuda_lt",
        [&] {
          // 调用 CUDA 平台的 gemm_and_bias 函数进行矩阵乘法和偏置计算
          at::cuda::blas::gemm_and_bias<scalar_t>(
              args.transa == 't',
              args.transb == 't',
              args.m,
              args.n,
              args.k,
              alpha.to<at::opmath_type<scalar_t>>(),
              args.mata->const_data_ptr<scalar_t>(),
              args.lda,
              args.matb->const_data_ptr<scalar_t>(),
              args.ldb,
              // 使用 self 的数据作为偏置
              self.const_data_ptr<scalar_t>(),
              args.result->data_ptr<scalar_t>(),
              args.result_ld,
              activation_epilogue
          );
        });
#endif
  } else
  {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
        at::ScalarType::Half,                  # 指定要处理的标量类型，包括Half和BFloat16
        at::ScalarType::BFloat16,
        scalar_type,
        "addmm_cuda",                         # 指定CUDA函数名称为addmm_cuda
        [&] {                                  # Lambda函数开始，捕获外部所有变量
          using opmath_t = at::opmath_type<scalar_t>;  # 定义opmath_t为标量类型scalar_t对应的操作数类型
          opmath_t alpha_val = alpha.to<opmath_t>();   # 将alpha转换为opmath_t类型并赋值给alpha_val
          opmath_t beta_val = beta.to<opmath_t>();     # 将beta转换为opmath_t类型并赋值给beta_val
          const scalar_t* mat1_ptr = args.mata->const_data_ptr<scalar_t>();  # 获取args.mata指向的常量数据指针
          const scalar_t* mat2_ptr = args.matb->const_data_ptr<scalar_t>();  # 获取args.matb指向的常量数据指针
          scalar_t* result_ptr = args.result->mutable_data_ptr<scalar_t>();   # 获取args.result指向的可变数据指针
          at::cuda::blas::gemm<scalar_t>(         # 调用CUDA的GEMM函数，进行矩阵乘法运算
              args.transa,                       # 指定是否对第一个矩阵进行转置操作
              args.transb,                       # 指定是否对第二个矩阵进行转置操作
              args.m, args.n, args.k,            # 矩阵的维度信息
              alpha_val,                         # 乘法运算中的alpha值
              mat1_ptr, args.lda,                # 第一个矩阵数据及其leading dimension
              mat2_ptr, args.ldb,                # 第二个矩阵数据及其leading dimension
              beta_val,                          # 乘法运算中的beta值
              result_ptr, args.result_ld);       # 结果矩阵数据及其leading dimension
        });                                    # Lambda函数结束

    switch (activation) {                       # 根据激活函数类型进行分支处理
      case Activation::RELU:                    # 如果是RELU激活函数
        at::relu_(const_cast<Tensor&>(*args.result));  # 对args.result进行in-place RELU操作
        break;                                 # 跳出switch语句
      case Activation::GELU:                    # 如果是GELU激活函数
        at::gelu_(const_cast<Tensor&>(*args.result), "tanh");  # 对args.result进行in-place GELU操作，指定内部tanh函数
        break;                                 # 跳出switch语句
      default: break;                          # 默认情况不做任何操作
    }
  }
// 如果 CUDA 版本不支持或者未定义 USE_ROCM，且使用了 Lt 接口且激活函数是 GELU
#if !(defined(CUDA_VERSION) && CUDA_VERSION >= 11080) && !defined(USE_ROCM)
  if (useLtInterface && activation == Activation::GELU) {
    // 对于无法使用上述 GELU 后处理的情况，这里手动执行 GELU 的后处理
    at::gelu_(const_cast<Tensor&>(*args.result), "tanh");
  }
#endif

  // 如果 result 不与 args.result 相同，将 args.result 的内容复制到 result 中
  if (!result.is_same(*args.result)) {
    result.copy_(*args.result);
  }
  // 返回 result
  return result;
}

// CUDA 实现的 baddbmm_out 函数，用于计算 result = beta * result + alpha * (self @ batch1 @ batch2)
const Tensor& baddbmm_out_cuda_impl(const Tensor& result, const Tensor& self, const Tensor& batch1, const Tensor& batch2, const Scalar& beta, const Scalar& alpha) {
  // 处理一些特殊情况，避免 BLAS 函数可能不喜欢的情况
  if (result.numel() == 0) {
    // 如果 result 是空的，直接返回 result
    return result;
  } else if (batch1.size(2) == 0) {
    // 如果 batch1 的第三维大小为0
    if (beta.to<c10::complex<double>>() == 0.0) {
      // 如果 beta 是复数且为0，将 result 全部置为0并返回
      return result.zero_();
    } else {
      // 否则将 result 中的每个元素乘以 beta，并返回
      return result.mul_(beta);
    }
  }

  // 初始化变量
  bool transpose_result = false;
  c10::MaybeOwned<Tensor> result_;
  IntArrayRef result_strides = result.strides();
  IntArrayRef result_sizes = result.sizes();

  // 根据 result 的 strides 和 sizes，决定是否需要转置 result
  if ((result_strides[1] == 1) &&
      ((result_sizes[2] == 1) || (result_strides[2] >= std::max<int64_t>(1, result_sizes[1])))) {
    // 需要转置 result
    result_ = resolve_conj_if_indicated(result, true);
  } else if ((result_strides[2] == 1) &&
    (result_sizes[1] == 1 || (result_strides[1] >= std::max<int64_t>(1, result_sizes[2])))) {
    // 需要转置 result
    transpose_result = true;
    result_ = resolve_conj_if_indicated(result, true);
  } else {
    // 否则，先转置 result，再克隆一个连续内存格式的 tensor
    result_ = c10::MaybeOwned<Tensor>::owned(result.transpose(1, 2).clone(at::MemoryFormat::Contiguous).transpose(1, 2));
  }

  // 确定 leading dimension 的值
  int leading_dim = transpose_result ? 1 : 2;

  // 获取矩阵的尺寸
  int64_t m = result_sizes[transpose_result ? 2 : 1];
  int64_t n = result_sizes[leading_dim];
  int64_t k = (transpose_result ? batch2 : batch1).sizes()[leading_dim];

  int64_t lda, ldb, ldc;
  bool transpose_batch1, transpose_batch2;

  // 准备 batch1 的矩阵，以便在 cublas 中使用
  auto batch1_ = prepare_batch_matrix_for_cublas(transpose_result ? batch2 : batch1, transpose_batch1, lda, transpose_result, m, k);
  // 准备 batch2 的矩阵，以便在 cublas 中使用
  auto batch2_ = prepare_batch_matrix_for_cublas(transpose_result ? batch1 : batch2, transpose_batch2, ldb, transpose_result, k, n);

  // 设置 ldc 为 result_ 的 leading dimension
  ldc = result_->strides()[leading_dim];
  int64_t num_batches = result_->sizes()[0];

  // 内部断言，确保 result_ 不是共轭的
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!result_->is_conj());

  // 根据 result_ 的数据类型调度不同类型的计算函数
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, self.scalar_type(), "baddbmm_cuda", [&] {
    using opmath_t = at::opmath_type<scalar_t>;
    opmath_t alpha_val = alpha.to<opmath_t>();
    opmath_t beta_val = beta.to<opmath_t>();
    const scalar_t* batch1_ptr = batch1_->const_data_ptr<scalar_t>();
    const scalar_t* batch2_ptr = batch2_->const_data_ptr<scalar_t>();
    scalar_t* result_ptr = result_->mutable_data_ptr<scalar_t>();
    // 根据是否需要转置 batch1，选择合适的矩阵乘法类型
    const auto transa = transpose_batch1 ? batch1_->is_conj() ? 'c' : 't' : 'n';
    // 确定是否需要对 batch2 进行转置，并根据转置情况选择 'c' 或 't'，否则选择 'n'
    const auto transb = transpose_batch2 ? batch2_->is_conj() ? 'c' : 't' : 'n';
    // 如果只有一个批次，调用普通的 gemm 函数而不是 bgemm
    if (num_batches == 1) {
      // 调用 cuBLAS 中的 gemm 函数进行矩阵乘法运算
      at::cuda::blas::gemm<scalar_t>(
          transa, transb,  // 矩阵 A 和 B 是否需要转置
          m, n, k,        // 矩阵尺寸
          alpha_val,      // 乘法的 alpha 参数
          batch1_ptr, lda,  // 矩阵 A 的指针和 leading dimension
          batch2_ptr, ldb,  // 矩阵 B 的指针和 leading dimension
          beta_val,       // 乘法的 beta 参数
          result_ptr, ldc);  // 结果矩阵的指针和 leading dimension
    } else {
      // 调用 cuBLAS 中的 bgemm 函数进行批量矩阵乘法运算
      at::cuda::blas::bgemm<scalar_t>(
        transa, transb,  // 矩阵 A 和 B 是否需要转置
        m, n, k,        // 矩阵尺寸
        alpha_val,      // 乘法的 alpha 参数
        batch1_ptr, lda, batch1_->strides()[0],  // 矩阵 A 的指针、leading dimension 和 stride
        batch2_ptr, ldb, batch2_->strides()[0],  // 矩阵 B 的指针、leading dimension 和 stride
        beta_val,       // 乘法的 beta 参数
        result_ptr, ldc, result_->strides()[0],  // 结果矩阵的指针、leading dimension 和 stride
        num_batches    // 批次数量
      );
   }
  });
  // 如果 result 不是指向 result_，则将 result_ 的内容复制给 result
  if (!result.is_same(*result_)) {
    result.copy_(*result_);
  }
  // 返回计算结果 result
  return result;
// 匿名命名空间，用于定义内部函数 dot_check
namespace {

// 检查两个向量 self 和 other 是否为 1 维
inline void dot_check(const Tensor& self, const Tensor& other) {
  // 检查 self 和 other 的维度是否为 1 维
  TORCH_CHECK(
      self.dim() == 1 && other.dim() == 1,
      "1D tensors expected, but got ",
      self.dim(),
      "D and ",
      other.dim(),
      "D tensors");
  // 检查 self 和 other 的数据类型是否相同
  TORCH_CHECK(
      self.scalar_type() == other.scalar_type(),
      "dot : expected both vectors to have same dtype, but found ",
      self.scalar_type(),
      " and ",
      other.scalar_type());
  // 检查 self 和 other 的元素数量是否相同
  TORCH_CHECK(
      self.numel() == other.numel(),
      "inconsistent tensor size, expected tensor [",
      self.numel(),
      "] and src [",
      other.numel(),
      "] to have the same number of elements, but got ",
      self.numel(),
      " and ",
      other.numel(),
      " elements respectively");
  // 检查 self 和 other 的元素数量、步长是否在 INT_MAX 范围内
  TORCH_CHECK(
      (self.numel() <= INT_MAX) && (self.stride(0) <= INT_MAX) &&
          (other.stride(0) <= INT_MAX),
      "dot only supports n, incx, incy with the bound [val] <= %d",
      INT_MAX);
}

} // 匿名命名空间结束

// 使用 CUDA 实现的 dot 函数，计算两个张量的点积
Tensor dot_cuda(const Tensor& self, const Tensor& other) {
  // 如果 self 是复数张量
  if (self.is_complex()) {
    // 如果 self 是共轭张量
    if (self.is_conj()) {
      // 如果 other 也是共轭张量，返回它们的点积的共轭
      if (other.is_conj()) {
        return (dot_cuda(self.conj(), other.conj())).conj();
       } else {
         // 如果 other 不是共轭张量，返回它们的共轭点积
         return vdot_cuda(self.conj(), other);
       }
    } else if (other.is_conj()) {
      // 如果 self 不是共轭张量但 other 是共轭张量，返回 other 和 self 的共轭点积
      return vdot_cuda(other.conj(), self);
    }
  }

  // 进入无名称保护块，确保执行期间不生成命名张量
  at::NoNamesGuard guard;
  // 调用 dot_check 函数检查 self 和 other 张量的一致性
  dot_check(self, other);

  // 获取 self 的元素数量并转换为整数
  const int n = static_cast<int>(self.numel());
  // 获取 self 的步长并转换为整数
  int incx = static_cast<int>(self.stride(0));
  // 获取 other 的步长并转换为整数
  int incy = static_cast<int>(other.stride(0));
  // 如果元素数量为 1，设置步长为 1
  if (n == 1) {
    incx = 1;
    incy = 1;
  }

  // 如果 self 或 other 是零张量，则返回一个有效的零张量
  if (self._is_zerotensor() || other._is_zerotensor()) {
    return at::_efficientzerotensor({}, self.options());
  }
// 返回 AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2 的结果，该宏允许在浮点和复数类型之间进行调度，
// 包括半精度浮点型和 BF16 浮点型。在这里，它被用于确定当前张量的标量类型并在 GPU 上执行 dot 操作。
return AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      ScalarType::Half, ScalarType::BFloat16,
      self.scalar_type(), "dot",
      [&] {
        // 创建一个空张量 result，使用与 self 相同的选项进行分配。
        Tensor result = at::empty({}, self.options());

        // 获取当前 CUDA blas 句柄
        auto handle = at::cuda::getCurrentCUDABlasHandle();
        // 设置指针模式为设备模式，以便在 CUDA BLAS 调用中使用设备指针
        at::cuda::blas::PointerModeGuard pointerModeGuard(handle, CUBLAS_POINTER_MODE_DEVICE);
        // 调用 CUDA BLAS 库中的 dot 函数，计算 self 和 other 张量的点积，结果存储在 result 中
        at::cuda::blas::dot<scalar_t>(
            handle,
            n,
            self.const_data_ptr<scalar_t>(),
            incx,
            other.const_data_ptr<scalar_t>(),
            incy,
            result.mutable_data_ptr<scalar_t>());

        // 返回计算结果张量 result
        return result;
      });
}

// 在 CUDA 上执行复数类型的向量内积操作 vdot
Tensor vdot_cuda(const Tensor& self, const Tensor& other) {
  // 如果 self 不是复数类型，则调用 dot_cuda 函数计算向量内积并返回结果
  if (!self.is_complex()) {
    return dot_cuda(self, other);
  }

  // 如果 self 是共轭的，则处理共轭情况
  if (self.is_conj()) {
    if (other.is_conj()) {
      // 如果 other 也是共轭的，则递归调用 vdot_cuda 函数，传入共轭的张量，并返回结果
      return vdot_cuda(other.conj(), self.conj());
    } else {
      // 如果 other 不是共轭的，则调用 dot_cuda 计算 self.conj() 和 other 的内积并返回
      return dot_cuda(self.conj(), other);
    }
  } else if (other.is_conj()) {
    // 如果 other 是共轭的，则调用 dot_cuda 计算 self 和 other.conj() 的内积，并返回结果的共轭
    return (dot_cuda(self, other.conj())).conj();
  }

  // 禁用命名检查，进行 dot 操作的参数检查
  at::NoNamesGuard guard;
  dot_check(self, other);

  // 如果 self 或 other 是零张量，则返回与 self 相同类型和选项的零张量
  if (self._is_zerotensor() || other._is_zerotensor()) {
    return at::_efficientzerotensor({}, self.options());
  }

  // 获取 self 张量的元素数量 n，以及两个张量的步长 incx 和 incy
  const int n = static_cast<int>(self.numel());
  int incx = static_cast<int>(self.stride(0));
  int incy = static_cast<int>(other.stride(0));
  // 如果 n 为 1，则将步长设置为 1
  if (n == 1) {
    incx = 1;
    incy = 1;
  }

  // 使用 AT_DISPATCH_COMPLEX_TYPES 宏，根据 self 的标量类型执行 vdot 操作
  return AT_DISPATCH_COMPLEX_TYPES(self.scalar_type(), "vdot", [&] {
    // 创建一个空张量 result，使用与 self 相同的选项进行分配
    Tensor result = at::empty({}, self.options());

    // 获取当前 CUDA blas 句柄
    auto handle = at::cuda::getCurrentCUDABlasHandle();
    // 设置指针模式为设备模式，以便在 CUDA BLAS 调用中使用设备指针
    at::cuda::blas::PointerModeGuard pointerModeGuard(
        handle, CUBLAS_POINTER_MODE_DEVICE);
    // 调用 CUDA BLAS 库中的 vdot 函数，计算复数类型的向量内积，结果存储在 result 中
    at::cuda::blas::vdot<scalar_t>(
        handle,
        n,
        self.const_data_ptr<scalar_t>(),
        incx,
        other.const_data_ptr<scalar_t>(),
        incy,
        result.mutable_data_ptr<scalar_t>());

    // 返回计算结果张量 result
    return result;
  });
}

// 在 CUDA 上实现的 addmv_out 操作的具体实现函数
TORCH_IMPL_FUNC(addmv_out_cuda)(const Tensor &self, const Tensor &mat, const Tensor &vec, const Scalar& beta_, const Scalar& alpha_, const Tensor& result) {
  // 使用 expand_size 函数，将 self 张量扩展到与 mat 的第一个维度相同的形状，并返回
  c10::MaybeOwned<Tensor> self_ = expand_size(self, {mat.size(0)});
  // 将 beta_ 转换为复数双精度数值
  auto betaval = beta_.toComplexDouble();
  // 如果 mat 张量元素数量为 0，则进行空矩阵的特殊处理
  if (mat.numel() == 0) {
    // 当 beta==0 时，result 的值应被忽略。NaN 和 Inf 不应传播
    if (betaval == 0.0) {
      // 如果 beta 为 0，则将 result 张量的所有元素置零
      result.zero_();
    } else {
      // 否则，使用 scalar_tensor 函数创建一个标量张量，与 self 具有相同的标量类型和值 beta_
      at::mul_out(
          const_cast<Tensor&>(result),
          self,
          at::native::scalar_tensor(
              beta_, self.scalar_type(), c10::nullopt /* layout */, at::kCPU, c10::nullopt /* pin_memory */));
    }
  } else {
    // 如果 result 与 self_ 不是同一张量，并且 betaval 不为 0，则进行复制操作
    if (!result.is_same(*self_) && betaval != 0.0) { // 如果 beta 为 0，后续会将 result 内容置零
      at::native::copy_(const_cast<Tensor&>(result), *self_);
    }
    // 如果结果张量非空
    if (result.numel() != 0) {
      // 获取结果张量的第一个维度的步长
      auto r_stride = result.stride(0);
      // 获取向量张量的第一个维度的步长
      auto vec_stride = vec.stride(0);

      // 检查向量张量是否连续，并相应更新其步长
      const auto vec_contiguous = vec_stride == 0 ? vec.contiguous() : vec;
      // 如果向量长度为1，即使是连续的，步长可能为0
      vec_stride = std::max<int64_t>(vec_contiguous.stride(0), 1LL);

      // 根据张量的数据类型调度相应的操作，使用 lambda 表达式进行 CUDA 计算
      AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, mat.scalar_type(), "addmv_impl_cuda", [&] {
        // 获取 beta 和 alpha 的标量值
        auto beta = beta_.to<scalar_t>();
        auto alpha = alpha_.to<scalar_t>();

        // 如果矩阵的第一个维度步长为1且第二个维度步长大于等于矩阵的第一个维度长度
        if (mat.stride(0) == 1 && mat.stride(1) >= std::max<int64_t>(1, mat.size(0))) {
          // 使用 CUDA 中的 gemv 函数进行矩阵-向量乘法
          at::cuda::blas::gemv<scalar_t>('n',
            mat.size(0), mat.size(1), alpha, mat.const_data_ptr<scalar_t>(), mat.stride(1), vec_contiguous.const_data_ptr<scalar_t>(),
            vec_stride, beta, result.mutable_data_ptr<scalar_t>(), r_stride);
        }
        // 如果矩阵的第二个维度步长为1且第一个维度步长大于等于矩阵的第二个维度长度
        else if (mat.stride(1) == 1 && mat.stride(0) >= std::max<int64_t>(1, mat.size(1))) {
          // 使用 CUDA 中的 gemv 函数进行矩阵转置-向量乘法
          at::cuda::blas::gemv<scalar_t>('t',
            mat.size(1), mat.size(0), alpha, mat.const_data_ptr<scalar_t>(), mat.stride(0),
            vec_contiguous.const_data_ptr<scalar_t>(), vec_stride, beta, result.mutable_data_ptr<scalar_t>(), r_stride);
        }
        // 否则，需要对矩阵进行连续化处理，然后进行矩阵转置-向量乘法
        else {
          Tensor cmat = mat.contiguous();
          at::cuda::blas::gemv<scalar_t>('t',
              mat.size(1), mat.size(0), alpha, cmat.const_data_ptr<scalar_t>(), cmat.stride(0),
              vec_contiguous.const_data_ptr<scalar_t>(), vec_stride, beta, result.mutable_data_ptr<scalar_t>(), r_stride);
        }
      });
    }
  }
}

Tensor& _int_mm_out_cuda(const Tensor& self, const Tensor& mat2, Tensor& result) {
  // 检查输入张量 self 和 mat2 的维度是否为 2
  TORCH_CHECK(self.dim() == 2, "Expected self to be of dimension 2 but got ", self.dim());
  TORCH_CHECK(mat2.dim() == 2, "Expected mat2 to be of dimension 2 but got ", mat2.dim());
  // 检查 self 的第一个维度是否大于 16
  TORCH_CHECK(self.size(0) > 16, "self.size(0) needs to be greater than 16, but got ", self.size(0));
  // 检查 self 的第二个维度是否大于 0 并且是 8 的倍数
  TORCH_CHECK(self.size(1) > 0 && self.size(1) % 8 == 0, "self.size(1) needs to be greater than 0 and a multiple of 8, but got ", self.size(1));
  // 检查 self 的第二个维度是否与 mat2 的第一个维度相同
  TORCH_CHECK(self.size(1) == mat2.size(0), "self.size(1) needs to match mat2.size(0) but got ", self.size(1), " and ", mat2.size(0));
  // 检查 mat2 的第二个维度是否大于 0 并且是 8 的倍数
  TORCH_CHECK(mat2.size(1) > 0 && mat2.size(1) % 8 == 0, "mat2.size(1) needs to be greater than 0 and a multiple of 8, but got ", mat2.size(1));

  // 检查 result 张量的数据类型是否为 kInt 类型
  TORCH_CHECK(result.dtype() == at::kInt, "Expected result dtype to be of type kInt but got ", result.dtype());
  // 检查 result 张量的第一个维度是否与 self 的第一个维度相同
  TORCH_CHECK(result.size(0) == self.size(0), "Expected result.size(0) to be ", self.size(0), " but got ", result.size(0));
  // 检查 result 张量的第二个维度是否与 mat2 的第二个维度相同
  TORCH_CHECK(result.size(1) == mat2.size(1), "Expected result.size(1) to be ", mat2.size(1), " but got ", result.size(1));

  // 检查 result 张量是否为二维张量
  TORCH_CHECK(result.dim() == 2, "Expected result to be of dimension 2 but got ", result.dim());

  // 检查 result 张量是否是连续的
  TORCH_CHECK(result.is_contiguous(), "Expected result to be contiguous.");

  // 检查是否支持当前平台的 CUDA 版本，如果支持则执行以下操作
#if (defined(CUDA_VERSION) && (CUDA_VERSION >= 11070)) || defined(USE_ROCM)
  // 创建 cublasCommonArgs 对象
  cublasCommonArgs args(self, mat2, result);

  // 调用 CUDA 的 int8_gemm 函数进行矩阵乘法计算
  at::cuda::blas::int8_gemm(
      args.transa == 't',
      args.transb == 't',
      args.m,
      args.n,
      args.k,
      args.mata->data_ptr<int8_t>(),
      args.lda,
      args.matb->data_ptr<int8_t>(),
      args.ldb,
      args.result->data_ptr<int32_t>(),
      args.result_ld);

  // 如果 result 不是同一个对象则复制计算结果到 result 中
  if (!result.is_same(*args.result)) {
    result.copy_(*args.result);
  }
#else
  // 如果不支持当前平台的 CUDA 版本则抛出错误
#if !defined(USE_ROCM) && defined(CUDA_VERSION)
  TORCH_CHECK(false, "_int_mm_out_cuda not compiled for CUDA ", CUDA_VERSION);
#else
  TORCH_CHECK(false, "_int_mm_out_cuda not compiled for this platform.");
#endif
#endif

  // 返回 result 张量
  return result;
}

Tensor _int_mm_cuda(const Tensor& self, const Tensor& mat2) {
  // 创建一个新的张量 result，形状为 (self.size(0), mat2.size(1))，数据类型为 kInt
  Tensor result = at::empty({self.size(0), mat2.size(1)}, self.options().dtype(at::kInt));
  // 调用 _int_mm_out_cuda 函数计算矩阵乘法，将结果存储到 result 中
  return _int_mm_out_cuda(self, mat2, result);
}

static bool _scaled_mm_allowed_device() {
  // 获取当前 CUDA 设备的属性
  auto dprops = at::cuda::getCurrentDeviceProperties();
#ifdef USE_ROCM
  // 获取设备架构的名称
  std::string device_arch = dprops->gcnArchName;
  // 支持的设备架构列表
  static const std::vector<std::string> archs = {"gfx940", "gfx941", "gfx942"};
  // 遍历支持的设备架构列表，如果当前设备架构存在于列表中则返回 true
  for (std::string arch : archs) {
    size_t substring = device_arch.find(arch);
    if (substring != std::string::npos) {
      return true;
    }
  }
  // 否则返回 false
  return false;
#else
  // 检查设备的 CUDA 主版本号是否大于等于 9，或者主版本号为 8 且次版本号为 9
  return dprops->major >= 9 || (dprops->major == 8 && dprops->minor == 9);
#endif
}

namespace{

// 枚举类型，表示缩放类型，包括 TensorWise（张量级别）、RowWise（行级别）、Error（错误）
enum class ScalingType {
  TensorWise,
  RowWise,
  Error
};
/*
 * Scaling Type Determination:
 * ---------------------------
 * Conditions and corresponding Scaling Types:
 *
 * - If scale_a.numel() == 1 && scale_b.numel() == 1:
 *   - Returns TensorWise.
 *
 * - Else if scale_a.dim() == 1 && scale_a.size(0) == dim_m && scale_b.size(0) == dim_n:
 *   - Returns RowWise.
 *
 * - Otherwise:
 *   - Returns Error.
 */

// Validates the scale tensors to scaled_mm
// And returns the type of scaling/which kernel to use
ScalingType get_scaling_type(
    const at::Tensor& scale_a,            // Input tensor for scaling factor A
    const at::Tensor& scale_b,            // Input tensor for scaling factor B
    int64_t dim_m,                        // Dimension size of matrix A
    int64_t dim_n) {                      // Dimension size of matrix B
  // Both Per-Tensor and Row-wise scaling expect fp32 tensors
  TORCH_CHECK(
      scale_a.scalar_type() == kFloat && scale_b.scalar_type() == kFloat,
      "Both scale_a and scale_b must be float (fp32) tensors.");

  // Check the singular scale case for per-tensor scaling
  if (scale_a.numel() == 1 && scale_b.numel() == 1) {
    return ScalingType::TensorWise;
  } else if (scale_a.dim() == 1 && scale_a.size(0) == dim_m) {
    // Check the per-row scaling case
#if !defined(USE_ROCM) && !defined(_MSC_VER) || \
    (defined(USE_ROCM) && ROCM_VERSION >= 60000)
    TORCH_CHECK(
        scale_a.dim() == 1 && scale_b.dim() == 1,
        "Both scale_a and scale_b must be 1-dimensional tensors");
    TORCH_CHECK(
        scale_b.size(0) == dim_n,
        "For row-wise scaling, scale_b must have size ",
        dim_n,
        " but got ",
        scale_b.size(0),
        ".");
    TORCH_CHECK(
        scale_a.is_contiguous() && scale_b.is_contiguous(),
        "Both scale_a and scale_b must be contiguous.");
    return ScalingType::RowWise;
#else
    TORCH_CHECK(false, "Per-row scaling is not supported for this platform!");
    return ScalingType::Error;
#endif // !defined(USE_ROCM) && !defined(_MSC_VER) || (defined(USE_ROCM) &&
       // ROCM_VERSION >= 60000)
  } else {
    // Prettier Error Case messaging
    TORCH_CHECK(
        false,
        "For row-wise scaling, scale_a must be size ",
        dim_m,
        " but got ",
        scale_a.numel(),
        " and scale_b must be size ",
        dim_n,
        " but got ",
        scale_b.numel(),
        ".");
    // Unreachable
    return ScalingType::RowWise;
  }
  return ScalingType::Error;
}

} // namespace

// Computes matrix multiply + bias while applying scaling to input and output matrices and computes amax
// Scales are only applicable when matrices are of Float8 type and assumbed to be equal to 1.0 by default.
// If output matrix type is 16 or 32-bit type, neither scale_result is applied nor amax is computed.
// Known limitations:
//  - Only works if mat1 is row-major and mat2 is column-major
//  - Only works if matrices sizes are divisible by 32
//  - If 1-dimensional tensors are used then scale_a should be size = mat1.size(0)
//    and scale_b should have size = to mat2.size(1)
//  Arguments:
//    - `mat1`: the first operand of the matrix multiply, can be type `torch.float8_e4m3fn` or `torch.float8_e5m2`
# `mat1`: 第一个矩阵乘法操作数，类型可以是 `torch.float8_e4m3fn` 或 `torch.float8_e5m2`
# `mat2`: 第二个矩阵乘法操作数，类型可以是 `torch.float8_e4m3fn` 或 `torch.float8_e5m2`
# `bias`: 偏置项，类型可以是 `torch.float16` 或 `torch.bfloat16`
# `out_dtype`: 输出数据类型，可以是 float8 或更高精度的浮点数类型
# `scale_a`: 标量或者一维张量，存储 `mat1` 的倒数缩放比，仅在 `mat1` 是 float8 类型时需要
# `scale_b`: 标量或者一维张量，存储 `mat2` 的倒数缩放比，仅在 `mat2` 是 float8 类型时需要
# `scale_result`: 标量张量，存储输出的缩放比，仅在输出是 float8 类型时使用
# `use_fast_accum`: 如果为真，启用快速的 float8 累加计算
# `out`: 输出张量的引用
Tensor&
// 定义 CUDA 函数 _scaled_mm_out_cuda，用于执行矩阵乘法操作，支持缩放、偏置和结果类型设置
_scaled_mm_out_cuda(const Tensor& mat1,  // 输入矩阵 mat1
          const Tensor& mat2,            // 输入矩阵 mat2
          const Tensor& scale_a,         // 缩放因子 scale_a
          const Tensor& scale_b,         // 缩放因子 scale_b
          const std::optional<at::Tensor>& bias,  // 可选的偏置向量
          const std::optional<at::Tensor>& scale_result,  // 可选的缩放结果
          std::optional<c10::ScalarType> out_dtype,       // 可选的输出数据类型
          bool use_fast_accum,            // 是否使用快速累加
          Tensor& out) {                 // 输出张量 out
  // 检查是否允许在当前设备上执行 _scaled_mm 操作
  bool allowed_device = _scaled_mm_allowed_device();
  TORCH_CHECK(allowed_device, "torch._scaled_mm is only supported on CUDA devices with compute capability >= 9.0 or 8.9, or ROCm MI300+");

  // 检查输入矩阵 mat1 和 mat2 的维度
  TORCH_CHECK(mat1.dim() == 2, "mat1 must be a matrix");
  TORCH_CHECK(mat2.dim() == 2, "mat2 must be a matrix");

  // 检查 mat1 和 mat2 的形状是否可以相乘
  TORCH_CHECK(
      mat1.sizes()[1] == mat2.sizes()[0], "mat1 and mat2 shapes cannot be multiplied (",
      mat1.sizes()[0], "x", mat1.sizes()[1], " and ", mat2.sizes()[0], "x", mat2.sizes()[1], ")");

  // 根据输入确定使用的缩放类型
  ScalingType scaling_choice = get_scaling_type(scale_a, scale_b, mat1.size(0), mat2.size(1));
  TORCH_INTERNAL_ASSERT(scaling_choice != ScalingType::Error, "Scaling type not supported");

  // 检查 scale_result 是否为 null 或者是包含单个浮点数的张量
  TORCH_CHECK(!scale_result || (scale_result->numel() == 1 && scale_result->scalar_type() == kFloat),
       "scale_result must be a float scalar");

  // 如果存在偏置，则检查其长度是否与 mat2 的列数相等
  TORCH_CHECK(!bias || bias->numel() == mat2.sizes()[1], "Bias must be size ", mat2.sizes()[1],
       " but got ", bias->numel());

  // 检查 mat1 的第二维度是否可以被 16 整除
  TORCH_CHECK(
      mat1.sizes()[1] % 16 == 0,
      "Expected trailing dimension of mat1 to be divisible by 16 ",
      "but got mat1 shape: (",
      mat1.sizes()[0],
      "x",
      mat1.sizes()[1],
      ".");

  // 检查 mat2 的形状是否每个维度都可以被 16 整除
  TORCH_CHECK(mat2.sizes()[0] % 16 == 0 && mat2.sizes()[1] % 16 == 0, "mat2 shape (", mat2.sizes()[0], "x",
       mat2.sizes()[1], " must be divisible by 16");

  // 检查输出数据类型是否与指定的 out_dtype 匹配
  TORCH_CHECK(!out_dtype || *out_dtype == out.scalar_type(), "out_dtype must match output matrix type");

  // 检查 mat1 和 mat2 的数据类型是否为 Float8
  TORCH_CHECK(isFloat8Type(mat1.scalar_type()), "Expected mat1 to be Float8 matrix got ", mat1.scalar_type());
  TORCH_CHECK(isFloat8Type(mat2.scalar_type()), "Expected mat2 to be Float8 matrix got ", mat2.scalar_type());

  // CUDA-12.1 中 CuBLASLt 对 Float8_e5m2 类型的矩阵乘法有限制
  TORCH_CHECK(mat1.scalar_type() != ScalarType::Float8_e5m2 || mat2.scalar_type() != ScalarType::Float8_e5m2,
        "Multiplication of two Float8_e5m2 matrices is not supported");

  // 如果存在偏置，进一步检查输出类型是否支持与偏置相关的计算
  if (bias) {
    TORCH_CHECK(out.scalar_type() != kFloat, "Bias is not supported when out_dtype is set to Float32");
    TORCH_CHECK(bias->scalar_type() == ScalarType::BFloat16 || bias->scalar_type() == ScalarType::Half,
         "Bias must be either Half or BFloat16, but got ", bias->scalar_type());
    TORCH_CHECK((out.scalar_type() != kFloat && out.scalar_type() != ScalarType::BFloat16) ||
          bias->scalar_type() == ScalarType::BFloat16,
          "Bias must be BFloat16 to compute ", out.scalar_type(), " output, but got ", bias->scalar_type());
  // 检查输出张量的数据类型不是半精度（Half）或者偏置项的数据类型是半精度
  // 如果输出数据类型是半精度，则偏置项必须也是半精度，否则抛出异常
  TORCH_CHECK(out.scalar_type() != ScalarType::Half || bias->scalar_type() == ScalarType::Half,
        "Bias must be Float16 to compute ", out.scalar_type(), " output, but got ", bias->scalar_type());
}
{
  // 获取可选的偏置项（如果有的话）和尺度结果（如果有的话）
  auto bias_ = bias.value_or(Tensor());
  auto scale_result_ = scale_result.value_or(Tensor());

  // 创建张量参数数组，用于后续的GPU一致性检查
  TensorArg targs[]{{out, "out", 0}, {mat1, "mat1", 1}, {mat2, "mat2", 2},
                    {bias_, "bias", 3}, {scale_a, "scale_a", 4}, {scale_b, "scale_b", 5},
                    {scale_result_, "scale_result", 6}};
  // 检查所有张量是否在相同的GPU上
  checkAllSameGPU(__func__, targs);
}
// 验证检查通过后，调整输出张量的尺寸以匹配实际尺寸
IntArrayRef mat1_sizes = mat1.sizes();
IntArrayRef mat2_sizes = mat2.sizes();
at::native::resize_output(out, {mat1_sizes[0], mat2_sizes[1]});

// 进行行级别的缩放
if (scaling_choice == ScalingType::RowWise) {
  // 检查输出张量的数据类型是否为BF16，仅支持BF16高精度输出类型用于行级缩放
  TORCH_CHECK(out.dtype() == kBFloat16, "Only bf16 high precsion output types are supported for row-wise scaling.");
  // 使用CUDA实现的函数进行行级别的缩放计算
  at::cuda::detail::f8f8bf16_rowwise(
      mat1,
      mat2,
      scale_a,
      scale_b,
      bias,
      use_fast_accum,
      out);
  return out;
}

// 创建 cuBLASLt 所需的通用参数对象
cublasCommonArgs args(mat1, mat2, out);
const auto out_dtype_ = args.result->scalar_type();
// 检查是否仅支持行主序和列主序矩阵的乘法操作
TORCH_CHECK(args.transa == 't' && args.transb == 'n', "Only multiplication of row-major and column-major matrices is supported by cuBLASLt");

// 某些缩放的 GEMM 需要一个 amax 来填充，这里创建一个空的张量用于存储 amax
Tensor amax = at::empty({0}, mat1.options().dtype(ScalarType::Float));
#ifdef USE_ROCM
  // 获取 ROCm 下的调优上下文
  auto tuning_ctx = at::cuda::tunable::getTuningContext();
  // 如果可调优操作已启用
  if (tuning_ctx->IsTunableOpEnabled()) {
    // 定义可调优分发宏，根据不同的数据类型组合生成不同的调优操作
#define TUNABLE_DISPATCH(BLASOP_A, BLASOP_B)                            \
        // 如果 mat1 和 mat2 的数据类型都是 Float8_e4m3fnuz
        if (mat1.scalar_type() == ScalarType::Float8_e4m3fnuz) {        \
          if (mat2.scalar_type() == ScalarType::Float8_e4m3fnuz) {      \
            // 创建静态的 ScaledGemmTunableOp 对象，处理指定的数据类型组合
            static at::cuda::tunable::ScaledGemmTunableOp<              \
                at::Float8_e4m3fnuz, at::Float8_e4m3fnuz, scalar_t,     \
                BLASOP_A, BLASOP_B> scaledgemm{};                       \
            // 调用 scaledgemm 对象处理参数
            scaledgemm(&params);                                        \
          }                                                             \
          // 如果 mat1 的数据类型是 Float8_e4m3fnuz，而 mat2 的数据类型是 Float8_e5m2fnuz
          else if (mat2.scalar_type() == ScalarType::Float8_e5m2fnuz) { \
            // 创建静态的 ScaledGemmTunableOp 对象，处理指定的数据类型组合
            static at::cuda::tunable::ScaledGemmTunableOp<              \
                at::Float8_e4m3fnuz, at::Float8_e5m2fnuz, scalar_t,     \
                BLASOP_A, BLASOP_B> scaledgemm{};                       \
            // 调用 scaledgemm 对象处理参数
            scaledgemm(&params);                                        \
          }                                                             \
        }                                                               \
        // 如果 mat1 的数据类型是 Float8_e5m2fnuz
        else if (mat1.scalar_type() == ScalarType::Float8_e5m2fnuz) {   \
          // 如果 mat2 的数据类型是 Float8_e4m3fnuz
          if (mat2.scalar_type() == ScalarType::Float8_e4m3fnuz) {      \
            // 创建静态的 ScaledGemmTunableOp 对象，处理指定的数据类型组合
            static at::cuda::tunable::ScaledGemmTunableOp<              \
                at::Float8_e5m2fnuz, at::Float8_e4m3fnuz, scalar_t,     \
                BLASOP_A, BLASOP_B> scaledgemm{};                       \
            // 调用 scaledgemm 对象处理参数
            scaledgemm(&params);                                        \
          }                                                             \
          // 如果 mat2 的数据类型是 Float8_e5m2fnuz
          else if (mat2.scalar_type() == ScalarType::Float8_e5m2fnuz) { \
            // 创建静态的 ScaledGemmTunableOp 对象，处理指定的数据类型组合
            static at::cuda::tunable::ScaledGemmTunableOp<              \
                at::Float8_e5m2fnuz, at::Float8_e5m2fnuz, scalar_t,     \
                BLASOP_A, BLASOP_B> scaledgemm{};                       \
            // 调用 scaledgemm 对象处理参数
            scaledgemm(&params);                                        \
          }                                                             \
        }
  }
#endif
    AT_DISPATCH_V2(out_dtype_, "_tunable_scaled_gemm", AT_WRAP([&] {
      // 检查矩阵操作是否需要转置
      bool transa_ = ((args.transa != 'n') && (args.transa != 'N'));
      bool transb_ = ((args.transb != 'n') && (args.transb != 'N'));
      // 创建用于 GPU 可调谐的矩阵乘法参数对象
      at::cuda::tunable::ScaledGemmParams<scalar_t> params;
      // 设置矩阵乘法的参数
      params.transa = args.transa;
      params.transb = args.transb;
      params.m = args.m;
      params.n = args.n;
      params.k = args.k;
      // 设置第一个矩阵 A 的数据指针及其缩放因子
      params.a = args.mata->data_ptr();
      params.a_scale_ptr = scale_a.data_ptr();
      params.lda = args.lda;
      params.a_dtype = args.mata->scalar_type();
      // 设置第二个矩阵 B 的数据指针及其缩放因子
      params.b = args.matb->data_ptr();
      params.b_scale_ptr = scale_b.data_ptr();
      params.ldb = args.ldb;
      params.b_dtype = args.matb->scalar_type();
      // 设置偏置项的数据指针及其数据类型
      params.bias_ptr = bias ? bias->data_ptr(): nullptr;
      params.bias_dtype = bias ? bias->scalar_type() : isFloat8Type(out_dtype_) ? at::ScalarType::Half : out_dtype_;
      // 设置结果矩阵 C 的数据指针及其缩放因子
      params.c = args.result->data_ptr();
      params.c_scale_ptr = scale_result ? scale_result->data_ptr() : nullptr;
      params.ldc = args.result_ld;
      params.c_dtype = out_dtype_;
      // 设置用于搜集最大值的指针
      params.amax_ptr = amax.data_ptr();
      params.use_fast_accum = use_fast_accum;
      // 根据矩阵是否需要转置，选择不同的 CUDA 可调谐分派
      if (transa_ && transb_) {
        TUNABLE_DISPATCH(at::cuda::tunable::BlasOp::T, at::cuda::tunable::BlasOp::T)
      }
      else if (transa_ && !transb_) {
        TUNABLE_DISPATCH(at::cuda::tunable::BlasOp::T, at::cuda::tunable::BlasOp::N)
      }
      else if (!transa_ && transb_) {
        TUNABLE_DISPATCH(at::cuda::tunable::BlasOp::N, at::cuda::tunable::BlasOp::T)
      }
      else if (!transa_ && !transb_) {
        TUNABLE_DISPATCH(at::cuda::tunable::BlasOp::N, at::cuda::tunable::BlasOp::N)
      }
      else {
        // 如果出现了无法到达的情况，抛出错误信息
        TORCH_CHECK(false, "unreachable");
      }
    }),
    // 以下是展开的参数列表，用于传递给 AT_DISPATCH_V2 宏
    kHalf, kBFloat16, kFloat8_e4m3fnuz, kFloat8_e5m2fnuz, AT_EXPAND(AT_FLOATING_TYPES));
// 如果未定义 TUNABLE_DISPATCH，则进入该分支
  }
  else
#endif
  {
#if defined(USE_ROCM) && ROCM_VERSION >= 60200
  // 在 ROCm 版本大于等于 60200 时，hipBlasLT 要求 scaleD 设置为某个值以便使用 AMAX
    auto dummy_options = TensorOptions().dtype(kFloat).device(kCUDA);
    auto dummy_scale = at::ones(1, dummy_options);
#endif
    // 执行 CUDA 的 scaled_gemm 操作，进行矩阵乘法计算
    at::cuda::blas::scaled_gemm(
        args.transa,                             // 矩阵 A 是否需要转置
        args.transb,                             // 矩阵 B 是否需要转置
        args.m,                                  // 矩阵 A 的行数
        args.n,                                  // 矩阵 B 的列数
        args.k,                                  // 矩阵 A 的列数（同时也是矩阵 B 的行数）
        args.mata->data_ptr(),                   // 矩阵 A 的数据指针
        scale_a.data_ptr(),                      // 矩阵 A 的缩放因子数据指针
        args.lda,                                // 矩阵 A 的领先维度
        args.mata->scalar_type(),                // 矩阵 A 的数据类型
        args.matb->data_ptr(),                   // 矩阵 B 的数据指针
        scale_b.data_ptr(),                      // 矩阵 B 的缩放因子数据指针
        args.ldb,                                // 矩阵 B 的领先维度
        args.matb->scalar_type(),                // 矩阵 B 的数据类型
        bias ? bias->data_ptr() : nullptr,       // 偏置项数据指针（如果存在）
        bias ? bias->scalar_type() :             // 偏置项数据类型（如果存在）
          isFloat8Type(out_dtype_) ? at::ScalarType::Half : out_dtype_,
        args.result->data_ptr(),                 // 输出结果数据指针
#if defined(USE_ROCM) && ROCM_VERSION >= 60200
        scale_result ? scale_result->data_ptr() : dummy_scale.data_ptr(), // 输出结果的缩放因子数据指针或虚拟缩放因子数据指针
#else
        scale_result ? scale_result->data_ptr() : nullptr, // 输出结果的缩放因子数据指针（如果存在）
#endif
        args.result_ld,                          // 输出结果的领先维度
        out_dtype_,                              // 输出结果的数据类型
        amax.data_ptr(),                         // 输出结果的 AMAX 数据指针
        use_fast_accum);                         // 是否使用快速累积模式
  }

  return out;
}

Tensor
_scaled_mm_cuda(const Tensor& mat_a, const Tensor& mat_b,
          const Tensor& scale_a,
          const Tensor& scale_b,
          const std::optional<at::Tensor>& bias,
          const std::optional<at::Tensor>& scale_result,
          std::optional<c10::ScalarType> out_dtype,
          bool use_fast_accum) {
  const auto out_dtype_ = out_dtype.value_or(mat_a.scalar_type());
  // 创建一个与输入张量类型相同的空张量 out
  Tensor out = at::empty({0}, mat_a.options().dtype(out_dtype_));
  // 调用 _scaled_mm_out_cuda 函数计算并存储结果到 out 中
  return _scaled_mm_out_cuda(mat_a, mat_b, scale_a, scale_b, bias, scale_result, out_dtype, use_fast_accum, out);
}

} // namespace at::native
```