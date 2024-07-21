# `.\pytorch\aten\src\ATen\Context.cpp`

```py
#include <ATen/Config.h>

#include <ATen/Context.h>

#include <c10/core/CPUAllocator.h>

#include <algorithm>
#include <cctype>
#include <string>
#include <stdexcept>

#include <ATen/cpu/FlushDenormal.h>

#ifdef USE_FBGEMM
#include <fbgemm/Fbgemm.h>
#endif // USE_FBGEMM
#if defined(__aarch64__) && !defined(C10_MOBILE)
#include <cpuinfo.h>
#endif

namespace at {

// 默认构造函数，使用默认参数
Context::Context() = default;

// 返回全局上下文对象的引用
// 如果在静态对象的析构函数中调用globalContext()可能会有问题
Context& globalContext() {
  static Context globalContext_;
  return globalContext_;
}

// 返回用户是否启用了 CuDNN 的状态
bool Context::userEnabledCuDNN() const {
  return enabled_cudnn;
}

// 设置用户是否启用 CuDNN 的状态
void Context::setUserEnabledCuDNN(bool e) {
  enabled_cudnn = e;
}

// 返回用户是否启用了 MKL-DNN 的状态
bool Context::userEnabledMkldnn() const {
  return enabled_mkldnn;
}

// 设置用户是否启用 MKL-DNN 的状态
void Context::setUserEnabledMkldnn(bool e) {
  enabled_mkldnn = e;
}

// 返回 CuDNN 是否被设置为确定性模式
bool Context::deterministicCuDNN() const {
  return deterministic_cudnn;
}

// 设置 CuDNN 是否使用确定性模式
void Context::setDeterministicCuDNN(bool b) {
  deterministic_cudnn = b;
}

// 返回 MKL-DNN 是否被设置为确定性模式
bool Context::deterministicMkldnn() const {
  return deterministic_mkldnn;
}

// 设置 MKL-DNN 是否使用确定性模式
void Context::setDeterministicMkldnn(bool b) {
  deterministic_mkldnn = b;
}

// 返回全局算法是否使用确定性的状态
bool Context::deterministicAlgorithms() const {
  return _deterministic_algorithms;
}

// 返回是否仅仅警告而不中断运行的确定性算法状态
bool Context::deterministicAlgorithmsWarnOnly() const {
  return _deterministic_algorithms_warn_only;
}

// 设置全局算法是否使用确定性的状态及警告选项
void Context::setDeterministicAlgorithms(bool b, bool warn_only=false) {
  _deterministic_algorithms = b;
  _deterministic_algorithms_warn_only = warn_only;
}

// 返回是否使用确定性填充未初始化内存的状态
bool Context::deterministicFillUninitializedMemory() const {
  return _deterministic_fill_uninitialized_memory;
}

// 设置是否使用确定性填充未初始化内存的状态
void Context::setDeterministicFillUninitializedMemory(bool b) {
  _deterministic_fill_uninitialized_memory = b;
}

// 如果全局算法设置为确定性，但调用了非确定性方法，产生警告或错误信息
void Context::alertNotDeterministic(c10::string_view const& caller) {
  if (globalContext().deterministicAlgorithms()) {
    if (globalContext().deterministicAlgorithmsWarnOnly()) {
      TORCH_WARN(
        caller, " does not have a deterministic implementation, but you set "
        "'torch.use_deterministic_algorithms(True, warn_only=True)'. "
        "You can file an issue at https://github.com/pytorch/pytorch/issues "
        "to help us prioritize adding deterministic support for this operation.");
    } else {
      TORCH_CHECK(false,
        caller, " does not have a deterministic implementation, but you set "
        "'torch.use_deterministic_algorithms(True)'. You can turn off "
        "determinism just for this operation, or you can use the "
        "'warn_only=True' option, if that's acceptable for your application. "
        "You can also file an issue at https://github.com/pytorch/pytorch/issues "
        "to help us prioritize adding deterministic support for this operation.");
    }
  }
}

} // namespace at


这段代码定义了 `at` 命名空间下的 `Context` 类及其成员函数，实现了一些与全局运行环境及算法确定性相关的功能。
// 返回当前对象中的 NNPACK 是否启用的状态
bool Context::userEnabledNNPACK() const {
  return enabled_nnpack;
}

// 设置当前对象中的 NNPACK 是否启用的状态
void Context::setUserEnabledNNPACK(bool e) {
  enabled_nnpack = e;
}

// 返回当前对象中的是否允许 TF32 CuDNN 的状态
bool Context::allowTF32CuDNN() const {
  return allow_tf32_cudnn;
}

// 设置当前对象中是否允许 TF32 CuDNN 的状态
void Context::setAllowTF32CuDNN(bool b) {
  allow_tf32_cudnn = b;
}

// 返回当前对象中的 FlashSDP 是否启用的状态
bool Context::userEnabledFlashSDP() const {
  return enabled_flashSDP;
}

// 设置当前对象中的 FlashSDP 是否启用的状态
void Context::setSDPUseFlash(bool e) {
  enabled_flashSDP = e;
}

// 返回当前对象中的 MemEfficientSDP 是否启用的状态
bool Context::userEnabledMemEfficientSDP() const {
  return enabled_mem_efficientSDP;
}

// 设置当前对象中的 MemEfficientSDP 是否启用的状态
void Context::setSDPUseMemEfficient(bool e) {
  enabled_mem_efficientSDP = e;
}

// 返回当前对象中的 MathSDP 是否启用的状态
bool Context::userEnabledMathSDP() const {
  return enabled_mathSDP;
}

// 设置当前对象中的 MathSDP 是否启用的状态
void Context::setSDPUseMath(bool e) {
  enabled_mathSDP = e;
}

// 返回当前对象中的 CuDNN SDP 是否启用的状态
bool Context::userEnabledCuDNNSDP() const {
  return enabled_cudnnSDP;
}

// 设置当前对象中的 CuDNN SDP 是否启用的状态
void Context::setSDPUseCuDNN(bool e) {
  enabled_cudnnSDP = e;
}

// 设置当前对象中的是否允许覆盖 SDP 的状态
void Context::setSDPUseOverrideable(bool e) {
  enabled_overrideable = e;
}

// 返回当前对象中的是否允许覆盖 SDP 的状态
bool Context::userEnabledOverrideableSDP() const {
  return enabled_overrideable;
}

// 声明 CuBLAS 的工作空间配置变量名
static const char cublas_config_var_name[] = "CUBLAS_WORKSPACE_CONFIG";
// 定义 CuBLAS 的确定性配置选项
static const char* const cublas_deterministic_configs[] = { ":4096:8", ":16:8" };

// 检查当前 CuBLAS 的配置是否为确定性的
bool Context::checkCuBLASConfigDeterministic() {
  bool cublas_config_deterministic = true;
  // 如果使用 CUDA 10.2 或更高版本，需要确保 CuBLAS 的工作空间配置为确定性设置
  if (hasCUDART() && (versionCUDART() >= 10020)) {
    char* workspace_config = std::getenv(cublas_config_var_name);
    cublas_config_deterministic = (workspace_config != nullptr) && (
      (strcmp(workspace_config, cublas_deterministic_configs[0]) == 0)
      || (strcmp(workspace_config, cublas_deterministic_configs[1]) == 0)
    );
  }
  return cublas_config_deterministic;
}

// 当 CuBLAS 配置不是确定性时发出警告
void Context::alertCuBLASConfigNotDeterministic() const {
  // 静态变量，记录当前 CuBLAS 配置是否为确定性的
  static bool cublas_config_deterministic = checkCuBLASConfigDeterministic();
  // 如果允许使用确定性算法或当前 CuBLAS 配置为确定性，则直接返回
  if (C10_LIKELY(!deterministicAlgorithms() || cublas_config_deterministic)) {
    return;
  }

  // 构造警告或错误信息，提示用户关于 CuBLAS 配置的不确定性
  auto msg = c10::str(
    "Deterministic behavior was enabled with either `torch.use_deterministic_algorithms(True)` or ",
    "`at::Context::setDeterministicAlgorithms(true)`, but this operation is not deterministic because ",
    "it uses CuBLAS and you have CUDA >= 10.2. To enable deterministic behavior in this ",
    "case, you must set an environment variable before running your PyTorch application: ",
    cublas_config_var_name, "=", cublas_deterministic_configs[0], " or ",
    cublas_config_var_name, "=", cublas_deterministic_configs[1], ". For more information, go to ",
    "https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility"
  );

  // 如果仅仅警告，则发出警告
  if (deterministicAlgorithmsWarnOnly()) {
    TORCH_WARN(msg);
  } else {  // 否则抛出异常
    TORCH_CHECK(false, msg);
  }
}
// 返回 benchmark_cudnn 成员变量的当前值，用于确定是否开启 CuDNN 的基准测试
bool Context::benchmarkCuDNN() const {
  return benchmark_cudnn;
}

// 设置 benchmark_cudnn 成员变量的值，用于开启或关闭 CuDNN 的基准测试
void Context::setBenchmarkCuDNN(bool b) {
  benchmark_cudnn = b;
}

// 返回 benchmark_limit_cudnn 成员变量的当前值，用于确定 CuDNN 基准测试的限制
int Context::benchmarkLimitCuDNN() const {
  return benchmark_limit_cudnn;
}

// 设置 benchmark_limit_cudnn 成员变量的值，用于设定 CuDNN 基准测试的限制
void Context::setBenchmarkLimitCuDNN(int b) {
  benchmark_limit_cudnn = b;
}

// 返回 float32_matmul_precision 成员变量的当前值，确定是否允许 TF32 的 CuBLAS 加速
bool Context::allowTF32CuBLAS() const {
  return float32_matmul_precision != at::Float32MatmulPrecision::HIGHEST;
}

// 设置 float32_matmul_precision 成员变量的值，用于允许或禁止 TF32 的 CuBLAS 加速
void Context::setAllowTF32CuBLAS(bool b) {
  float32_matmul_precision = b ? at::Float32MatmulPrecision::HIGH : at::Float32MatmulPrecision::HIGHEST;
}

// 返回 float32_matmul_precision 成员变量的当前值，表示当前的 TF32 精度设置
Float32MatmulPrecision Context::float32MatmulPrecision() const {
  return float32_matmul_precision;
}

// 设置 float32_matmul_precision 成员变量的值，根据传入的枚举类型设定 TF32 精度
void Context::setFloat32MatmulPrecision(Float32MatmulPrecision p) {
  float32_matmul_precision = p;
}

// 根据传入的字符串设定 float32_matmul_precision 成员变量的值，考虑不同的精度选项
void Context::setFloat32MatmulPrecision(const std::string &s) {
  auto match = [this](const std::string & s_) {
    // TODO: consider if CuDNN field needs to also be set for potential future CuDNN ops like multi-headed attention
    if (s_ == "highest") {
      float32_matmul_precision = at::Float32MatmulPrecision::HIGHEST;
      return true;
    } else if (s_ == "high") {
      float32_matmul_precision = at::Float32MatmulPrecision::HIGH;
      return true;
    } else if (s_ == "medium") {
      float32_matmul_precision = at::Float32MatmulPrecision::MEDIUM;
      return true;
    }
    return false;
  };
  if (match(s)) { return; }
  std::string sl;
  std::transform(s.begin(), s.end(), sl.begin(),
                 [](unsigned char c) -> unsigned char { return std::tolower(c); });
  if (match(sl)) { return; }
  // 若传入的字符串无法匹配有效的精度选项，则发出警告
  TORCH_WARN(s, " is not one of 'highest', 'high', or 'medium'; the current"
    "setFloat32MatmulPrecision call has no effect.");
}

// 返回 linalg_preferred_backend 成员变量的当前值，指示优选的线性代数库后端
at::LinalgBackend Context::linalgPreferredBackend() const {
  return linalg_preferred_backend;
}

// 设置 linalg_preferred_backend 成员变量的值，用于指定优选的线性代数库后端
void Context::setLinalgPreferredBackend(at::LinalgBackend b) {
  linalg_preferred_backend = b;
  // 对特定后端设置做验证，确保编译环境支持相应的库
  TORCH_CHECK((b != at::LinalgBackend::Cusolver) || hasCuSOLVER(),
      "Cannot set preferred backend to cuSOLVER if PyTorch has not been compiled with cuSOLVER.");
  TORCH_CHECK((b != at::LinalgBackend::Magma) || hasMAGMA(),
      "Cannot set preferred backend to MAGMA if PyTorch has not been compiled with MAGMA.");
  // 若非默认后端，发出一次性警告，提示实验性功能
  if (b != at::LinalgBackend::Default) {
    TORCH_WARN_ONCE(
      "torch.backends.cuda.preferred_linalg_library is an experimental feature. "
      "If you see any error or unexpected behavior when this flag is set "
      "please file an issue on GitHub."
    );
  }
}

// 返回 blas_preferred_backend 成员变量的当前值，指示优选的 BLAS 库后端
at::BlasBackend Context::blasPreferredBackend() const {
  return blas_preferred_backend;
}

// 设置 blas_preferred_backend 成员变量的值，用于指定优选的 BLAS 库后端
void Context::setBlasPreferredBackend(at::BlasBackend b) {
#ifdef _MSC_VER
  // 若在 Windows 上，发出一次性警告，指出实验性功能不支持 Windows
  TORCH_WARN_ONCE(
    "torch.backends.cuda.preferred_blas_library is an experimental feature. "
    "It is not supported on Windows."
  );
#endif
#else
  // 如果设置的 BLAS 后端不是 cuBLAS，发出一次性警告
  TORCH_CHECK((b != at::BlasBackend::Cublaslt) || hasCuBLASLt(),
      "Cannot set preferred backend to cuBLASLt if PyTorch has not been compiled with cuBLASLt.");
  // 设置的 BLAS 后端不是 cuBLAS 时，输出实验性特性的警告信息
  if (b != at::BlasBackend::Cublas) {
    TORCH_WARN_ONCE(
      "torch.backends.cuda.preferred_blas_library is an experimental feature. "
      "If you see any error or unexpected behavior when this flag is set "
      "please file an issue on GitHub."
    );
  }
  // 设置优选的 BLAS 后端为给定的后端 b
  blas_preferred_backend = b;
#endif
}

// 返回是否允许在 cuBLAS 中进行 FP16 缩减操作
bool Context::allowFP16ReductionCuBLAS() const {
  return allow_fp16_reduction_cublas;
}

// 设置是否允许在 cuBLAS 中进行 FP16 缩减操作
void Context::setAllowFP16ReductionCuBLAS(bool b) {
  allow_fp16_reduction_cublas = b;
}

// 返回是否允许在 cuBLAS 中进行 BF16 缩减操作
bool Context::allowBF16ReductionCuBLAS() const {
  return allow_bf16_reduction_cublas;
}

// 设置是否允许在 cuBLAS 中进行 BF16 缩减操作
void Context::setAllowBF16ReductionCuBLAS(bool b) {
  allow_bf16_reduction_cublas = b;
}

// 返回是否支持 MKL
bool Context::hasMKL() {
#if AT_MKL_ENABLED()
  return true;
#else
  return false;
#endif
}

// 返回是否支持 MKLDNN
bool Context::hasMKLDNN() {
#if AT_MKLDNN_ENABLED()
  return true;
#else
  return false;
#endif
}

// 返回是否支持 OpenMP
bool Context::hasOpenMP() {
#ifdef _OPENMP
  return true;
#else
  return false;
#endif
}

// 返回是否支持 LAPACK
bool Context::hasLAPACK() {
#if AT_BUILD_WITH_LAPACK()
  return true;
#else
  return false;
#endif
}

// 返回量化引擎的当前设置
at::QEngine Context::qEngine() const {
  static auto _quantized_engine = []() {
    at::QEngine qengine = at::kNoQEngine;
#if defined(C10_MOBILE) && defined(USE_PYTORCH_QNNPACK)
    qengine = at::kQNNPACK;
#endif

#if AT_MKLDNN_ENABLED()
    qengine = at::kONEDNN;
#endif

#ifdef USE_FBGEMM
    if (fbgemm::fbgemmSupportedCPU()) {
      /* X86 is enabled if and only if fbgemm is available.
       * It combines goodness of fbgemm and onednn by dispatching.
       * If onednn not available, always dispatch to fbgemm.
       * Make it default qengine for X86 CPU platforms.
      */
      qengine = at::kX86;
    }
#endif
    return qengine;
  }();
  return quantized_engine.value_or(_quantized_engine);
}

// 设置量化引擎为给定的引擎 e
void Context::setQEngine(at::QEngine e) {
  const auto& qengines = supportedQEngines();
  // 如果设置的引擎 e 在支持的引擎列表中，则进行设置
  if (std::find(qengines.begin(), qengines.end(), e) != qengines.end()) {
    quantized_engine = e;
    return;
  }
  // 否则，抛出错误，显示不支持的量化引擎
  TORCH_CHECK(false, "quantized engine ", toString(e), " is not supported");
}

// 返回支持的量化引擎列表
const std::vector<at::QEngine>& Context::supportedQEngines() {
  static auto supported_qengines = []() {
    std::vector<at::QEngine> engines = {};
    // 引擎按优先级顺序列出，后面的引擎优先
    // 默认情况下，如果在服务器端运行，我们首选 FBGEMM
    // 在服务器端，QNNPACK 存在问题，因此默认情况下禁用它。
#ifdef C10_MOBILE
    engines.push_back(at::kNoQEngine);
#ifdef USE_PYTORCH_QNNPACK
    engines.push_back(at::kQNNPACK);
#endif
#else  // C10_MOBILE
#ifdef USE_PYTORCH_QNNPACK
    engines.push_back(at::kQNNPACK);
#endif
    engines.push_back(at::kNoQEngine);
#endif // C10_MOBILE

#if AT_MKLDNN_ENABLED()
    engines.push_back(at::kONEDNN);
#endif

#ifdef USE_FBGEMM
    engines.push_back(at::kX86);
#endif
    return engines;
  }();
  return supported_qengines;
}
    # 检查当前系统是否支持 FBGEMM 加速库
    if (fbgemm::fbgemmSupportedCPU()) {
      # 如果系统支持 FBGEMM，则添加 X86 引擎到引擎列表
      engines.push_back(at::kX86);
      # 由于只有在 FBGEMM 可用时才可用，因此添加 FBGEMM 引擎到引擎列表
      engines.push_back(at::kFBGEMM);
    }
#endif
  // 返回静态局部变量engines，这是一个支持的查询引擎列表
  return engines;
}();
// 返回支持的量化引擎列表
return supported_qengines;
}

// 检查是否支持XNNPACK加速库
bool Context::isXNNPACKAvailable() {
#ifdef USE_XNNPACK
  // 如果定义了USE_XNNPACK，则返回true，表示XNNPACK可用
  return true;
#else
  // 否则返回false，表示XNNPACK不可用
  return false;
#endif
}

// 设置检查稀疏张量不变量的开关
void Context::setCheckSparseTensorInvariants(bool e) {
  enable_sparse_tensor_invariant_checks = e;
}

// 检查稀疏张量不变量是否开启
bool Context::checkSparseTensorInvariants() const {
  // 返回稀疏张量不变量检查的状态
  return enable_sparse_tensor_invariant_checks;
}

// 返回在预打包时是否释放原始权重的设置
bool Context::releaseWeightsWhenPrepacking() const {
  return release_original_weights;
}

// 设置在预打包时是否释放原始权重
void Context::setReleaseWeightsWhenPrepacking(bool e) {
  release_original_weights = e;
}

// 设置是否刷新非规格化数的开关
bool Context::setFlushDenormal(bool on) {
  // 调用ATen库的函数设置是否刷新非规格化数，返回设置结果
  return at::cpu::set_flush_denormal(on);
}

// 获取CPU的分配器
Allocator* getCPUAllocator() {
  return c10::GetCPUAllocator();
}

// 线程本地变量，控制是否覆盖allow_tf32标志
// override_allow_tf32_flag = true 表示强制禁用tf32
// override_allow_tf32_flag = false 表示遵循原始的allow_tf32标志
thread_local bool override_allow_tf32_flag = false;

// NoTF32Guard类的构造函数
NoTF32Guard::NoTF32Guard() {
  // 如果未覆盖allow_tf32标志，则进行覆盖，并标记为已改变
  if (!override_allow_tf32_flag) {
    changed = true;
    override_allow_tf32_flag = true;
  }
}

// NoTF32Guard类的析构函数
NoTF32Guard::~NoTF32Guard() {
  // 如果标记为已改变，则恢复allow_tf32标志
  if (changed) {
    override_allow_tf32_flag = false;
  }
}

// 查询是否应禁用tf32
bool NoTF32Guard::should_disable_tf32() {
  // 返回当前是否覆盖了allow_tf32标志
  return override_allow_tf32_flag;
}

// 线程本地变量，标记是否处于反向传播过程中
thread_local bool rocm_is_backward_pass;

// ROCmBackwardPassGuard类的构造函数
ROCmBackwardPassGuard::ROCmBackwardPassGuard() {
  // 设置标志表示处于ROCm的反向传播过程中
  rocm_is_backward_pass = true;
}

// ROCmBackwardPassGuard类的析构函数
ROCmBackwardPassGuard::~ROCmBackwardPassGuard() {
  // 恢复标志表示不再处于ROCm的反向传播过程中
  rocm_is_backward_pass = false;
}

// 查询是否处于ROCm的反向传播过程中
bool ROCmBackwardPassGuard::is_backward_pass() {
  return rocm_is_backward_pass;
}

// 获取是否启用Vmap的回退警告
bool Context::areVmapFallbackWarningsEnabled() const {
  return display_vmap_fallback_warnings_;
}

// 设置是否显示Vmap的回退警告
void Context::setDisplayVmapFallbackWarnings(bool enabled) {
  display_vmap_fallback_warnings_ = enabled;
}

// 设置默认的移动端CPU分配器
void Context::setDefaultMobileCPUAllocator() {
  // 断言前一个分配器指针为空，表示不在另一个非默认CPU分配器的作用域内
  TORCH_CHECK(prev_allocator_ptr_ == nullptr,
      "Already within the scope of another non-default cpu allocator."
      "Cannot set another allocator.");
  // 将当前的CPU分配器作为前一个分配器指针，并设置默认的移动端CPU分配器
  prev_allocator_ptr_ = c10::GetCPUAllocator();
  c10::SetCPUAllocator(c10::GetDefaultMobileCPUAllocator(), /*priority*/ 100);
}

// 取消设置默认的移动端CPU分配器
void Context::unsetDefaultMobileCPUAllocator() {
  // 断言前一个分配器指针不为空，表示之前已经调用了setDefaultMobileCPUAllocator
  TORCH_CHECK(prev_allocator_ptr_ != nullptr,
      "setDefaultMobileCPUAllocator must have been called "
      "before unsetDefaultMobileCPUAllocator.");
  // 恢复前一个CPU分配器，并设置其优先级为高
  c10::SetCPUAllocator(prev_allocator_ptr_ , /*priority*/ 100);
  prev_allocator_ptr_ = nullptr;
}

// 返回是否允许在CPU上进行FP16的归约操作
bool Context::allowFP16ReductionCPU() const {
  return allow_fp16_reduction_cpu;
}
void Context::setAllowFP16ReductionCPU(bool b) {
  // 如果允许 FP16 降级且当前未允许 FP16 降级
  if ( b && !allow_fp16_reduction_cpu) {
    // 检查 CPU 是否支持 FP16 运算
    // 对于 ARM 64 位架构且非移动平台
#if defined(__aarch64__) && !defined(C10_MOBILE)
    // 如果未能成功初始化 CPU 信息或者 CPU 不支持 ARM 的 FP16 算术运算
    if (!cpuinfo_initialize() || !cpuinfo_has_arm_fp16_arith())
#else
    // 其他情况默认为支持
    if (true)
#endif
      // 抛出运行时错误，指示 CPU 不支持 FP16 算术运算
      throw std::runtime_error("Float16 arithmetic is not supported by the CPU!");
  }
  // 更新允许 FP16 降级的状态
  allow_fp16_reduction_cpu = b;
}
} // namespace at
```