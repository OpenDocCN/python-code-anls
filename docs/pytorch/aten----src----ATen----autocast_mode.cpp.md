# `.\pytorch\aten\src\ATen\autocast_mode.cpp`

```
#include <ATen/autocast_mode.h>  // 包含 ATen 自动转换模式的头文件

#include <mutex>  // 包含互斥锁相关的头文件
#include <ATen/CachedTensorUtils.h>  // 包含 ATen 缓存张量工具的头文件
#include <c10/util/flat_hash_map.h>  // 包含 C10 的 flat_hash_map 实用工具的头文件

namespace at::autocast {

bool is_autocast_enabled(at::DeviceType device_type) {
  at::DispatchKey dispatch_key = get_autocast_dispatch_key_from_device_type(device_type);  // 获取自动转换分发键
  return !c10::impl::tls_is_dispatch_key_excluded(dispatch_key);  // 检查分发键是否排除
}

void set_autocast_enabled(at::DeviceType device_type, bool enabled) {
  at::DispatchKey dispatch_key = get_autocast_dispatch_key_from_device_type(device_type);  // 获取自动转换分发键
  c10::impl::tls_set_dispatch_key_excluded(dispatch_key, !enabled);  // 设置是否排除分发键
}

namespace {
// 模仿 Apex 并缓存一些类型转换，以优化参数重用。
// 我们的启发是缓存 fp32 模型权重的 lower_precision_fp 转换（参见下面的 cached_cast）。

// 经与 @ezyang 讨论后，缓存使用以下结构：
// 键是 fp32 源张量的 TensorImpl*，它是一个在浅拷贝中不变的张量 uuid 代理。
// 值是一个元组，其中第一个元素是指向源张量 TensorImpl 的弱引用，第二个元素是转换后的张量。

// 弱引用保持源张量的 TensorImpl 不被删除。我们需要这样做是因为我们将源 TensorImpl* 用作键。
// 如果它被删除，可能会分配另一个随机的 Tensor，其 TensorImpl* 恰好具有相同的值。
// 这个 TensorImpl* 然后会错误地在缓存中命中：这是一个罕见的、间歇性的、不可预测的 bug。

// 我们不使用 weak_intrusive_ptr 作为键，因为它更难直接与传入的 TensorImpl* 进行比较。
using weakref_type = c10::weak_intrusive_ptr<TensorImpl, UndefinedTensorImpl>;  // 定义弱引用类型
using val_type = std::tuple<weakref_type, Tensor>;  // 定义值类型为元组，包含弱引用和张量

static ska::flat_hash_map<TensorImpl*, val_type>& get_cached_casts() {
  static ska::flat_hash_map<TensorImpl*, val_type> cached_casts;  // 静态变量，用于存储类型转换的缓存
  return cached_casts;
}
std::mutex cached_casts_mutex;  // 互斥锁，保护 cached_casts 的访问

// nesting 追踪 Python 端上下文管理器的嵌套深度。
// 当自动转换上下文管理器退出到超出任何自动转换实例的嵌套级别时
// （这应该发生在每次前向传递结束时），它调用 clear_cache() 以确保缓存的张量不会泄漏到自动转换区域之外。
thread_local int nesting = 0;  // 定义线程局部变量，跟踪嵌套深度

// 此数组的顺序必须与 c10/core/DeviceType.h 中 DeviceType 的定义顺序完全匹配。
thread_local std::array<at::ScalarType, at::COMPILE_TIME_MAX_DEVICE_TYPES>  // 定义线程局部变量，存储标量类型数组
    # 定义自动转换的数据类型字典，将不同设备对应的数据类型映射到标准的数据类型
    autocast_dtype = {
        at::kBFloat16,             // CPU
        at::kHalf,                 // CUDA.
        at::ScalarType::Undefined, // Reserved for explicit MKLDNN
        at::ScalarType::Undefined, // OpenGL
        at::ScalarType::Undefined, // OpenCL
        at::ScalarType::Undefined, // IDEEP.
        at::kHalf,                 // AMD HIP
        at::ScalarType::Undefined, // FPGA
        at::ScalarType::Undefined, // ONNX Runtime / Microsoft
        at::kBFloat16,             // XLA / TPU
        at::ScalarType::Undefined, // Vulkan
        at::ScalarType::Undefined, // Metal
        at::kHalf,                 // XPU
        at::ScalarType::Undefined, // MPS
        at::ScalarType::Undefined, // Meta (tensors with no data)
        at::kBFloat16,             // HPU / HABANA
        at::ScalarType::Undefined, // SX-Aurora / NEC
        at::ScalarType::Undefined, // Lazy Tensors
        at::kHalf,                 // Graphcore IPU
        at::ScalarType::Undefined, // Meta training and inference devices
        at::kHalf                  // PrivateUse1 device
};

// 匿名命名空间结束

// 清空缓存函数，使用互斥锁确保线程安全
void clear_cache() {
  const std::lock_guard<std::mutex> lock(cached_casts_mutex);
  // 清空缓存中的类型转换结果
  get_cached_casts().clear();
}

// 增加嵌套层级计数器，并返回增加后的值
int increment_nesting() {
  return ++nesting;
}

// 减少嵌套层级计数器，并返回减少后的值
int decrement_nesting() {
  return --nesting;
}

// 获取自动转换数据类型
at::ScalarType get_autocast_dtype(at::DeviceType device_type) {
  return autocast_dtype[static_cast<int>(device_type)];
}

// 设置自动转换数据类型
void set_autocast_dtype(at::DeviceType device_type, at::ScalarType dtype) {
  autocast_dtype[static_cast<int>(device_type)] = dtype;
}

// 返回是否启用自动转换缓存
bool is_autocast_cache_enabled() {
  return cache_enabled;
}

// 设置是否启用自动转换缓存
void set_autocast_cache_enabled(bool enabled) {
  cache_enabled = enabled;
}

// 重载函数以处理 Tensor 类型参数的自动转换
// TODO (possible optimization):
// 将 cast_cache 移动到一个头文件中，并将 cached_casts 声明为头文件中的 extern thread_local 变量。
Tensor cached_cast(at::ScalarType to_type, const Tensor& arg, DeviceType device_type) {
  // 如果参数符合自动转换条件且需要类型转换，则尝试使用缓存
  if (is_eligible(arg, device_type) && (arg.scalar_type() != to_type)) {
    // 启发式方法：对于 fp32 模型权重（叶子节点），缓存较低精度的 fp 转换
    bool can_try_cache = (to_type == get_lower_precision_fp_from_device_type(device_type) &&
                         arg.scalar_type() == at::kFloat && arg.requires_grad() &&
                         arg.is_leaf() && !arg.is_view() && cache_enabled &&
                         !at::caching::is_cached_tensor(arg));

    if (can_try_cache) {
      const std::lock_guard<std::mutex> lock(cached_casts_mutex);
      auto it = get_cached_casts().find(arg.unsafeGetTensorImpl());
      if (it != get_cached_casts().end()) {
        // 如果找到缓存的转换结果，则直接返回缓存的结果
        return std::get<1>(it->second);
      } else {
        // 否则，进行类型转换并将结果缓存起来
        auto casted_arg = arg.to(to_type);
        get_cached_casts().emplace(arg.unsafeGetTensorImpl(), val_type{weakref_type(arg.getIntrusivePtr()), casted_arg});
        return casted_arg;
      }
    } else {
      // 如果不满足缓存条件，则直接进行类型转换
      return arg.to(to_type);
    }
  } else {
    // 如果不需要类型转换，则直接返回原始的 Tensor
    return arg;
  }
}

/*******************************
Banned functions
*******************************/

// 禁止使用的二进制交叉熵函数
static Tensor binary_cross_entropy_banned(const Tensor &, const Tensor &, const std::optional<Tensor>&, int64_t) {
  AT_ERROR("torch.nn.functional.binary_cross_entropy and torch.nn.BCELoss are unsafe to autocast.\n"
           "Many models use a sigmoid layer right before the binary cross entropy layer.\n"
           "In this case, combine the two layers using torch.nn.functional.binary_cross_entropy_with_logits\n"
           "or torch.nn.BCEWithLogitsLoss.  binary_cross_entropy_with_logits and BCEWithLogits are\n"
           "safe to autocast.");
}

namespace {

/*****************************************
Explicit registration for out-of-place ops
*****************************************/

// 实现 out-of-place 操作的显式注册
TORCH_LIBRARY_IMPL(_, Autocast, m) {
  m.fallback(torch::CppFunction::makeFallthrough());
}
TORCH_LIBRARY_IMPL(aten, Autocast, m) {
  // 注册 Autocast 库的实现，对于 aten 命名空间的操作
#define _KERNEL_CUDA_LOW_PRECISION_FP(...) \
  KERNEL_CUDA(__VA_ARGS__, lower_precision_fp)

  // 针对所有的 lower_precision_fp 操作，生成 CUDA 核函数
  AT_FORALL_LOWER_PRECISION_FP(_KERNEL_CUDA_LOW_PRECISION_FP)
  // 为 cudnn_convolution 操作生成 CUDA 核函数，采用 lower_precision_fp 策略
  KERNEL_CUDA(cudnn_convolution, lower_precision_fp)
  // 为 cudnn_convolution_transpose 操作生成 CUDA 核函数，采用 lower_precision_fp 策略
  KERNEL_CUDA(cudnn_convolution_transpose, lower_precision_fp)

  // fp32 策略
#define _KERNEL_CUDA_FP32(...) KERNEL_CUDA(__VA_ARGS__, fp32)

  // 为所有的 fp32 操作生成 CUDA 核函数
  AT_FORALL_FP32(_KERNEL_CUDA_FP32)

  // fp32_set_opt_dtype 策略
#define _KERNEL_CUDA_FP32_SET_OPT_DTYPE(...) \
  KERNEL_CUDA(__VA_ARGS__, fp32_set_opt_dtype)

  // 为所有的 fp32_set_opt_dtype 操作生成 CUDA 核函数
  AT_FORALL_FP32_SET_OPT_DTYPE(_KERNEL_CUDA_FP32_SET_OPT_DTYPE)
  // 下面的代码已被注释掉，因为这些操作接受显式（非可选）的数据类型，即使在自动类型转换时也不应该进行更改。
  // KERNEL_CUDA(norm, ScalarOpt_dtype, fp32_set_opt_dtype)
  // KERNEL_CUDA(norm, ScalarOpt_dim_dtype, fp32_set_opt_dtype)
  // KERNEL_CUDA(norm, names_ScalarOpt_dim_dtype, fp32_set_opt_dtype)

  // fp32_append_dtype 策略
  // fp32_append_dtype 包装器覆盖了隐式提升行为。norm 操作不隐式提升，但在添加新操作时要注意这个策略。
  AT_FORALL_DIFFERENT_REDISPATCH_SIGNATURE(
      KERNEL_DIFFERENT_REDISPATCH_SIGNATURE_CUDA)

  // promote 策略
#define _KERNEL_CUDA_PROMOTE(...) KERNEL_CUDA(__VA_ARGS__, promote)

  // 为所有的 promote 操作生成 CUDA 核函数
  AT_FORALL_PROMOTE(_KERNEL_CUDA_PROMOTE)

  // 注册 binary_cross_entropy 操作的 Autocast 实现，使用自定义的 banned 函数
  m.impl(TORCH_SELECTIVE_NAME("aten::binary_cross_entropy"),
         TORCH_FN((&at::autocast::binary_cross_entropy_banned)));
}

TORCH_LIBRARY_IMPL(_, AutocastCPU, m) {
  // AutocastCPU 库的实现，使用 C++ 函数的默认实现
  m.fallback(torch::CppFunction::makeFallthrough());
}

}

TORCH_LIBRARY_IMPL(_, AutocastXPU, m) {
  // AutocastXPU 库的实现，使用 C++ 函数的默认实现
  m.fallback(torch::CppFunction::makeFallthrough());
}

TORCH_LIBRARY_IMPL(aten, AutocastXPU, m) {
  // 注册 AutocastXPU 库的实现，对于 aten 命名空间的操作
#define _KERNEL_XPU_LOW_PRECISION_FP(...) \
  KERNEL_XPU(__VA_ARGS__, lower_precision_fp)

  // 针对所有的 lower_precision_fp 操作，生成 XPU 核函数
  AT_FORALL_LOWER_PRECISION_FP(_KERNEL_XPU_LOW_PRECISION_FP)

  // fp32 策略
#define _KERNEL_XPU_FP32(...) KERNEL_XPU(__VA_ARGS__, fp32)

  // 为所有的 fp32 操作生成 XPU 核函数
  AT_FORALL_FP32(_KERNEL_XPU_FP32)

  // fp32_set_opt_dtype 策略
#define _KERNEL_XPU_FP32_SET_OPT_DTYPE(...) \
  KERNEL_XPU(__VA_ARGS__, fp32_set_opt_dtype)

  // 为所有的 fp32_set_opt_dtype 操作生成 XPU 核函数
  AT_FORALL_FP32_SET_OPT_DTYPE(_KERNEL_XPU_FP32_SET_OPT_DTYPE)

  // fp32_append_dtype 策略
  // fp32_append_dtype 包装器覆盖了隐式提升行为。norm 操作不隐式提升，但在添加新操作时要注意这个策略。
  AT_FORALL_DIFFERENT_REDISPATCH_SIGNATURE(
      KERNEL_DIFFERENT_REDISPATCH_SIGNATURE_XPU)

  // promote 策略
#define _KERNEL_XPU_PROMOTE(...) KERNEL_XPU(__VA_ARGS__, promote)

  // 为所有的 promote 操作生成 XPU 核函数
  AT_FORALL_PROMOTE(_KERNEL_XPU_PROMOTE)

  // 注册 binary_cross_entropy 操作的 AutocastXPU 实现，使用自定义的 banned 函数
  m.impl(TORCH_SELECTIVE_NAME("aten::binary_cross_entropy"),
         TORCH_FN((&at::autocast::binary_cross_entropy_banned)));
}

} // namespace
} // namespace at::autocast
```