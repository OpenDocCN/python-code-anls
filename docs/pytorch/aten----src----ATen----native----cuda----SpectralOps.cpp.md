# `.\pytorch\aten\src\ATen\native\cuda\SpectralOps.cpp`

```py
// 定义 TORCH_ASSERT_ONLY_METHOD_OPERATORS 宏，用于指定仅包含方法操作符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

// 包含 ATen 库的头文件，用于张量操作和 CUDA 相关功能
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/Config.h>
#include <ATen/Dispatch.h>
#include <ATen/ScalarOps.h>
#include <ATen/TensorIterator.h>
#include <ATen/detail/CUDAHooksInterface.h>
#include <ATen/native/Resize.h>
#include <ATen/native/SpectralOpsUtils.h>
#include <ATen/native/cuda/CuFFTUtils.h>
#include <ATen/native/cuda/CuFFTPlanCache.h>
#include <ATen/cuda/nvrtc_stub/ATenNVRTC.h>
#include <c10/util/irange.h>

// 如果未定义 AT_PER_OPERATOR_HEADERS 宏，则包含下列头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
// 如果定义了 AT_PER_OPERATOR_HEADERS 宏，则包含下列头文件
#else
#include <ATen/ops/_fft_c2c_native.h>
#include <ATen/ops/_fft_c2r_native.h>
#include <ATen/ops/_fft_r2c_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/mul.h>
#endif

// 包含 CUDA FFT 库的头文件
#include <cufft.h>
#include <cufftXt.h>

// 包含数学函数库的头文件
#include <cmath>

// 使用 at::native 命名空间
namespace at::native {

// 使用 at::native::detail 命名空间中的内容
using namespace at::native::detail;

// 执行预先计划好的 cuFFT 变换
static void exec_cufft_plan(
    const CuFFTConfig &config, void* in_data, void* out_data, bool forward) {
  // 获取 cuFFT 计划对象的引用
  auto& plan = config.plan();
  // 执行 cuFFT 变换，根据 forward 参数选择正向或逆向变换
  CUFFT_CHECK(cufftXtExec(plan, in_data, out_data,
                          forward ? CUFFT_FORWARD : CUFFT_INVERSE));
}

// NOTE [ cuFFT Embedded Strides ]
//
// cuFFT 支持一部分任意步幅的张量，通过其“高级数据布局”选项
// (http://docs.nvidia.com/cuda/cufft/index.html#advanced-data-layout) 支持。
// 具体来说，这些张量可以被视为从一个更大的连续张量切片得到的子张量。
// 对于这样的输入张量，设其包围张量的大小为 inembed，对于三维情况，我们有：
//
//     input[x, y, z] = input[((x * inembed[1] + y) * inembed[2] + z)]
//
// 上述是简化的公式，忽略了批处理维度。事实上，包围张量的最后一个维度不必是连续的，
// 即它可以大于 1。然后可以使用 istride 设置包围张量的基础步幅。然后我们有：
//
//     input[x, y, z] = input[((x * inembed[1] + y) * inembed[2] + z) * istride]
//
// 例如，考虑以下情况：
//
//     enclosing = torch.zeros(6, 8, 10)  // 连续的
//     input = enclosing[:4, 2:6, 6:]
//     input.size()                       // [ 4,  4,  4]
//     input.stride()                     // [80, 10,  1]
//     // inembed = [6, 8, 10]
//     input[2, 1, 3] = input[((2 * 8) + 1) * 10 + 3]   // 使用上述公式
//                    = input[173]
//                    = input[2 * 80 + 1 * 10 + 1 * 3]  // 直接使用步幅
//
// 通常，嵌入步幅可以计算为
//
//     embed[i] = stride[i - 1] / stride[i].
//
// 注意，embed[0] 的值不用于计算索引，也不重要。
//
// 与高级数据布局相反，简单布局意味着 *embed 具有单位步幅。特别地，单位步幅指的是
// 输入和输出张量是连续的，并且最内部的信号的步幅。
// dimension being unit (1) w.r.t. the corresponding data type.

// The cuFFT plan cache
// unique_ptr for nullability and to avoid reference invalidation on vector resize
// 静态变量，存储指向 CuFFTParamsLRUCache 的 unique_ptr，用于缓存 cuFFT 计划
static std::vector<std::unique_ptr<CuFFTParamsLRUCache>> plan_caches;
// 互斥锁，保护对 plan_caches 的并发访问
static std::mutex plan_caches_mutex;

// 内联函数，获取给定设备索引的 cuFFT 计划缓存对象的引用
static inline
CuFFTParamsLRUCache &cufft_get_plan_cache(DeviceIndex device_index) {
  // 使用互斥锁保护对 plan_caches 的并发访问
  std::lock_guard<std::mutex> guard(plan_caches_mutex);

  // 断言设备索引为非负数
  AT_ASSERT(device_index >= 0);

  // 如果设备索引超出当前 plan_caches 的大小，则进行扩展
  if (device_index >= static_cast<int64_t>(plan_caches.size())) {
    plan_caches.resize(device_index + 1);
  }

  // 如果当前设备索引位置的缓存为空，则创建一个新的 CuFFTParamsLRUCache 对象
  if (!plan_caches[device_index]) {
    plan_caches[device_index] = std::make_unique<CuFFTParamsLRUCache>();
  }

  // 返回当前设备索引位置的 cuFFT 计划缓存对象的引用
  return *plan_caches[device_index];
}

namespace detail {

// 获取给定设备索引的 cuFFT 计划缓存对象的最大大小
int64_t cufft_get_plan_cache_max_size_impl(DeviceIndex device_index) {
  // 检查设备索引的有效性，确保在合法范围内
  TORCH_CHECK(0 <= device_index && device_index < at::detail::getCUDAHooks().getNumGPUs(),
    "cufft_get_plan_cache_max_size: expected 0 <= device_index < ",
    at::detail::getCUDAHooks().getNumGPUs(), "], but got device_index=",
    device_index);
  // 调用 cufft_get_plan_cache 函数获取计划缓存对象，并返回其最大大小
  return cufft_get_plan_cache(device_index).max_size();
}

// 设置给定设备索引的 cuFFT 计划缓存对象的最大大小
void cufft_set_plan_cache_max_size_impl(DeviceIndex device_index, int64_t max_size) {
  // 检查设备索引的有效性，确保在合法范围内
  TORCH_CHECK(0 <= device_index && device_index < at::detail::getCUDAHooks().getNumGPUs(),
    "cufft_set_plan_cache_max_size: expected 0 <= device_index < ",
    at::detail::getCUDAHooks().getNumGPUs(), "], but got device_index=",
    device_index);
  // 调用 cufft_get_plan_cache 函数设置计划缓存对象的最大大小
  return cufft_get_plan_cache(device_index).resize(max_size);
}

// 获取给定设备索引的 cuFFT 计划缓存对象的当前大小
int64_t cufft_get_plan_cache_size_impl(DeviceIndex device_index) {
  // 检查设备索引的有效性，确保在合法范围内
  TORCH_CHECK(0 <= device_index && device_index < at::detail::getCUDAHooks().getNumGPUs(),
    "cufft_get_plan_cache_size: expected 0 <= device_index < ",
    at::detail::getCUDAHooks().getNumGPUs(), "], but got device_index=",
    device_index);
  // 调用 cufft_get_plan_cache 函数获取计划缓存对象的当前大小
  return cufft_get_plan_cache(device_index).size();
}

// 清空给定设备索引的 cuFFT 计划缓存对象
void cufft_clear_plan_cache_impl(DeviceIndex device_index) {
  // 检查设备索引的有效性，确保在合法范围内
  TORCH_CHECK(0 <= device_index && device_index < at::detail::getCUDAHooks().getNumGPUs(),
    "cufft_clear_plan_cache: expected 0 <= device_index < ",
    at::detail::getCUDAHooks().getNumGPUs(), "], but got device_index=",
    device_index);
  // 调用 cufft_get_plan_cache 函数清空计划缓存对象
  return cufft_get_plan_cache(device_index).clear();
}

} // namespace at::native::detail

namespace {
// 定义常量，指定 cuFFT 支持的最大维度为 3
constexpr int64_t cufft_max_ndim = 3;

// 判断给定整数 n 是否包含较大的素数因子，影响 cuFFT 的性能
// 参考文档：https://docs.nvidia.com/cuda/cufft/index.html#accuracy-and-performance
bool has_large_prime_factor(int64_t n) {
  // 定义首个较大素数为 11，以及一组小于该值的特定素数
  constexpr int64_t first_large_prime = 11;
  const std::array<int64_t, 4> prime_radices{{2, 3, 5, 7}};
  // 遍历小于首个较大素数的特定素数
  for (auto prime : prime_radices) {
    // 如果 n 小于首个较大素数，则返回 false
    if (n < first_large_prime) {
        return false;
    }
    // 当 n 能被当前素数整除时，不断除以该素数
    while (n % prime == 0) {
      n /= prime;
    }
  }
  // 返回 n 是否不等于 1，用于判断 n 是否包含其他较大素数因子
  return n != 1;
}

// 执行通用的 FFT 操作（可以是 c2c、单边 r2c 或单边 c2r）
// 获取输入张量的维度数
const auto ndim = self.dim();
// 获取信号维度的数量
const int64_t signal_ndim = dim.size();
// 计算批次维度数
const auto batch_dims = ndim - signal_ndim;

// 对维度进行重排，使得批次维度首先出现，并按照步幅顺序排列
// 这样可以在将其折叠到单个批次维度时最大化数据局部性
DimVector dim_permute(ndim);
std::iota(dim_permute.begin(), dim_permute.end(), int64_t{0});

// 标记哪些维度需要变换
c10::SmallVector<bool, kDimVectorStaticSize> is_transformed_dim(ndim);
for (const auto& d : dim) {
  is_transformed_dim[d] = true;
}

// 将不需要变换的维度移动到批次维度的末尾
auto batch_end = std::partition(dim_permute.begin(), dim_permute.end(),
                                [&](int64_t d) {return !is_transformed_dim[d]; });
// 根据输入张量的步幅对维度进行排序
auto self_strides = self.strides();
std::sort(dim_permute.begin(), batch_end,
          [&](int64_t a, int64_t b) { return self_strides[a] > self_strides[b]; });
// 将dim中的维度复制到batch_end之后
std::copy(dim.cbegin(), dim.cend(), batch_end);
// 对输入张量进行维度重排
auto input = self.permute(dim_permute);

// 将批次维度折叠到单个维度
DimVector batched_sizes(signal_ndim + 1);
batched_sizes[0] = -1;
std::copy(input.sizes().cbegin() + batch_dims, input.sizes().cend(), batched_sizes.begin() + 1);
input = input.reshape(batched_sizes);

// 获取批次大小
const auto batch_size = input.sizes()[0];
// 设置信号大小
DimVector signal_size(signal_ndim + 1);
signal_size[0] = batch_size;
for (const auto i : c10::irange(signal_ndim)) {
  auto in_size = input.sizes()[i + 1];
  auto out_size = out_sizes[dim[i]];
  // 确保输入和输出大小匹配或符合FFT的特定条件
  signal_size[i + 1] = std::max(in_size, out_size);
  TORCH_INTERNAL_ASSERT(in_size == signal_size[i + 1] ||
                        in_size == (signal_size[i + 1] / 2) + 1);
  TORCH_INTERNAL_ASSERT(out_size == signal_size[i + 1] ||
                        out_size == (signal_size[i + 1] / 2) + 1);
}

// 更新批次大小到正确的大小
batched_sizes[0] = batch_size;
DimVector batched_out_sizes(batched_sizes.begin(), batched_sizes.end());
for (const auto i : c10::irange(dim.size())) {
  batched_out_sizes[i + 1] = out_sizes[dim[i]];
}
// 调整输出张量的大小
out.resize_(batched_out_sizes, MemoryFormat::Contiguous);

// 创建变换计划（可以从缓存获取或本地创建）
const auto value_type = c10::toRealValueType(input.scalar_type());
auto fft_type = GetCuFFTTransformType(input.is_complex(), out.is_complex());
CuFFTParams Params(input.strides(), out.strides(), signal_size, fft_type, value_type);
CuFFTParamsLRUCache& plan_cache = cufft_get_plan_cache(input.device().index());
std::unique_lock<std::mutex> guard(plan_cache.mutex, std::defer_lock);
std::optional<CuFFTConfig> uncached_plan;
const CuFFTConfig * config = nullptr;

// 为 gh-63152 和 gh-58724 进行的临时解决方案
// CUDA 11.1（cufft 10.3）中的 Bluestein 计划无法被重用
// 当尺寸具有大素数因子时才使用 Bluestein 算法
// 只有具有小素数因子的尺寸才能被缓存
bool use_caching = true;
#ifdef CUFFT_VERSION
  // 检查 CUFFT 版本号，确保在指定范围内
  if (10300 <= CUFFT_VERSION && CUFFT_VERSION < 10400) {
    // 只为具有较小素数因子的变换缓存计划
    use_caching = std::none_of(
        signal_size.begin() + 1, signal_size.end(), [](int64_t dim_size) {
      return has_large_prime_factor(dim_size);
    });
  }
#endif

  // 如果允许缓存且计划缓存大小大于0，则进入以下代码块
  if (use_caching && plan_cache.max_size() > 0) {
    guard.lock();
    // 再次检查计划缓存大小是否大于0（在获取锁之后进行检查）
    if (plan_cache.max_size() > 0) {  // check again after acquiring the lock
      // 查找并获取缓存中的计划配置
      config = &plan_cache.lookup(Params);
    }
  }

  // 如果配置为空指针，则执行以下操作
  if (config == nullptr) {
    // 在未缓存的计划中添加参数，并获取对应的配置
    uncached_plan.emplace(Params);
    config = &uncached_plan.value();
  }

  // 获取计划对象的引用
  auto & plan = config->plan();

  // 如果需要克隆输入，则按连续内存格式克隆输入数据
  if (config->should_clone_input()) {
    input = input.clone(MemoryFormat::Contiguous);
  }

  // 准备 CUFFT 执行
  CUFFT_CHECK(cufftSetStream(plan, at::cuda::getCurrentCUDAStream()));
  // 创建工作空间，并设置给 CUFFT 执行计划使用
  auto workspace = at::empty({ config->workspace_size() }, at::device(at::kCUDA).dtype(at::kByte));
  CUFFT_CHECK(cufftSetWorkArea(plan, workspace.mutable_data_ptr()));

  // 执行变换计划
#if !defined(USE_ROCM)
  // 获取当前 CUDA 上下文
  CUcontext pctx = nullptr;
  at::globalContext().getNVRTC().cuCtxGetCurrent(&pctx);
  // 如果没有当前上下文，则尝试设置主要上下文作为当前上下文的一个解决方法
  if (C10_UNLIKELY(!pctx)) {
    TORCH_WARN_ONCE("Attempting to run cuFFT, but there was no current CUDA context! Attempting to set the primary context...");
    at::globalContext().getNVRTC().cuDevicePrimaryCtxRetain(&pctx, 0);
    at::globalContext().getNVRTC().cuCtxSetCurrent(pctx);
  }
#endif /* !defined(USE_ROCM) */

  // 执行 CUFFT 变换计划
  exec_cufft_plan(*config, const_cast<void*>(input.const_data_ptr()), out.data_ptr(), forward);

  // 将输出张量按原始批次形状进行就地重塑，并反转维度置换
  DimVector out_strides(ndim);
  int64_t batch_numel = 1;
  for (int64_t i = batch_dims - 1; i >= 0; --i) {
    out_strides[dim_permute[i]] = batch_numel * out.strides()[0];
    batch_numel *= out_sizes[dim_permute[i]];
  }
  for (const auto i : c10::irange(batch_dims, ndim)) {
    out_strides[dim_permute[i]] = out.strides()[1 + (i - batch_dims)];
  }
  // 返回按指定大小和步幅进行重塑后的输出张量
  return out.as_strided_(out_sizes, out_strides, out.storage_offset());
}

// 计算归一化常数并就地应用于 self
// sizes 是双侧张量的大小，dims 是所有转换维度
double _fft_normalization_scale(int64_t normalization, IntArrayRef sizes, IntArrayRef dims) {
  // 将归一化模式转换为枚举类型
  auto norm = static_cast<fft_norm_mode>(normalization);
  // 如果归一化模式为 none，则返回1.0
  if (norm == fft_norm_mode::none) {
    return 1.0;
  }

  // 计算信号的总元素数量
  int64_t signal_numel = 1;
  for (auto dim : dims) {
    signal_numel *= sizes[dim];
  }

  // 计算归一化系数的分母部分
  const double scale_denom = (norm == fft_norm_mode::by_root_n) ?
    std::sqrt(signal_numel) : static_cast<double>(signal_numel);
  
  // 返回归一化比例系数
  return 1.0 / scale_denom;
}
// 应用规范化到傅立叶变换输出的张量
const Tensor& _fft_apply_normalization(const Tensor& self, int64_t normalization, IntArrayRef sizes, IntArrayRef dims) {
  // 计算规范化的比例因子
  auto scale = _fft_normalization_scale(normalization, sizes, dims);
  // 如果比例因子为1，则直接返回输入张量
  return (scale == 1.0) ? self : self.mul_(scale);
}

// 将规范化应用到输出张量的函数版本
Tensor& _fft_apply_normalization_out(Tensor& out, const Tensor& self, int64_t normalization, IntArrayRef sizes, IntArrayRef dims) {
  // 计算规范化的比例因子
  auto scale = _fft_normalization_scale(normalization, sizes, dims);
  // 在输出张量上执行乘法操作，结果存储在out中
  return at::mul_out(out, self, c10::scalar_to_tensor(scale));
}

}  // namespace (anonymous)

// 根据cuFFT是否支持指定的维度，决定是否使用优化路径进行FFT
bool use_optimized_cufft_path(IntArrayRef dim) {
  // 如果维度数超过cuFFT支持的最大维度数，或者维度以(0, 1)开头，则不使用优化路径
  if (dim.size() > cufft_max_ndim || (
    dim.size() >= 2 && dim[0] == 0 && dim[1] == 1
  )) {
    return false;
  } else {
    return true;
  }
}

// n维实到复FFT
Tensor _fft_r2c_cufft(const Tensor& self, IntArrayRef dim, int64_t normalization, bool onesided) {
  // 断言输入张量为浮点型
  TORCH_CHECK(self.is_floating_point());
  auto input_sizes = self.sizes();
  DimVector onesided_sizes(input_sizes.begin(), input_sizes.end());
  auto last_dim = dim.back();
  auto last_dim_halfsize = (input_sizes[last_dim]) / 2 + 1;
  onesided_sizes[last_dim] = last_dim_halfsize;
  IntArrayRef out_sizes = onesided ? onesided_sizes : input_sizes;

  // 创建输出张量，类型为输入张量的复数类型
  const auto out_options = self.options().dtype(c10::toComplexType(self.scalar_type()));
  auto output = at::empty(out_sizes, out_options);

  // 检查实数输入是否符合CuFFT的对齐要求，需要按照复数格式对齐
  const auto complex_size = 2 * self.element_size();
  const bool complex_aligned = (
      reinterpret_cast<std::uintptr_t>(self.const_data_ptr()) % complex_size == 0);
  auto working_tensor = self;
  // 如果输入不符合对齐要求，则进行处理使其对齐
  if (!complex_aligned) {
    working_tensor = self.movedim(last_dim, -1)
                         .clone(MemoryFormat::Contiguous)
                         .movedim(-1, last_dim);
  }

  // 根据是否使用优化路径，选择不同的FFT执行方式
  if (use_optimized_cufft_path(dim)) {
    // 在优化路径下执行FFT变换
    _exec_fft(output, working_tensor, out_sizes, dim, /*forward=*/true);
  } else {
    // 首先在最后一个维度上执行实到复变换
    {
      auto target_sizes = dim.size() == 1 ? out_sizes : onesided_sizes;
      _exec_fft(output, working_tensor, target_sizes, last_dim, /*forward=*/true);
      // 如果维度数大于1，则需要为下一步的复到复变换创建新的工作张量
      if (dim.size() > 1) {
        working_tensor = at::empty(out_sizes, out_options);
      }
    }

    // 然后执行任何剩余的复到复变换
    DimVector sorted_dims(dim.begin(), dim.end() - 1);
    // 当 sorted_dims 非空时执行循环
    while (!sorted_dims.empty()) {
      // 交换 output 和 working_tensor，以便重新使用 working_tensor
      std::swap(output, working_tensor);

      // 每次重新排序 dimensions，因为 _exec_fft 会重新调整 output 的步长
      auto strides = working_tensor.strides();
      // 根据步长 strides 对 sorted_dims 进行排序
      std::sort(sorted_dims.begin(), sorted_dims.end(),
                [&](int64_t a, int64_t b) { return strides[a] > strides[b]; });

      // 确定最大处理维度数，不超过 cufft_max_ndim 和 sorted_dims 的长度
      const auto max_dims = std::min(static_cast<size_t>(cufft_max_ndim), sorted_dims.size());
      // 获取 sorted_dims 中最后的 max_dims 个维度
      auto last_dims = IntArrayRef(sorted_dims).slice(sorted_dims.size() - max_dims, max_dims);

      // 执行 FFT，输出为 onesided 数据
      _exec_fft(output, working_tensor, onesided_sizes, last_dims, /*forward=*/true);
      // 减少 sorted_dims 的长度
      sorted_dims.resize(sorted_dims.size() - max_dims);
    }
  }

  // 只需对 onesided 切片进行归一化，因为另一半数据会被覆盖
  auto out_slice = output.slice(last_dim, 0, last_dim_halfsize);
  // 应用归一化操作到 FFT 输出的切片
  _fft_apply_normalization(out_slice, normalization, input_sizes, dim);

  // 如果不是 onesided FFT
  if (!onesided) {
    // 如果 output 在 last_dim 维度上的大小不等于 out_sizes 中的对应大小
    if (output.sizes()[last_dim] != out_sizes[last_dim]) {
      // 调整 working_tensor 的大小并复制 output 的切片到 working_tensor
      working_tensor.resize_(out_sizes, MemoryFormat::Contiguous);
      working_tensor.slice(last_dim, 0, last_dim_halfsize).copy_(output);
      // 将 working_tensor 移动到 output
      output = std::move(working_tensor);
    }
    // 填充 output 的共轭对称部分
    at::native::_fft_fill_with_conjugate_symmetry_(output, dim);
  }
  // 返回 FFT 的结果 output
  return output;
}

Tensor& _fft_r2c_cufft_out(const Tensor& self, IntArrayRef dim,
                           int64_t normalization, bool onesided, Tensor& out) {
    // 调用 _fft_r2c_cufft 函数执行实数到复数的 FFT 转换，结果存储在 result 中
    auto result = _fft_r2c_cufft(self, dim, static_cast<int64_t>(fft_norm_mode::none), /*onesided=*/true);

    // 如果是单边谱
    if (onesided) {
        // 对输出 out 进行归一化处理，并返回
        return _fft_apply_normalization_out(out, result, normalization, self.sizes(), dim);
    }

    // 调整输出 out 的大小与输入 self 相同
    resize_output(out, self.sizes());

    // 获取最后一个维度的大小
    auto last_dim = dim.back();
    auto last_dim_halfsize = result.sizes()[last_dim];

    // 在 out 上切片得到与 result 最后一个维度半大小相匹配的部分
    auto out_slice = out.slice(last_dim, 0, last_dim_halfsize);

    // 对切片部分 out_slice 进行归一化处理
    _fft_apply_normalization_out(out_slice, result, normalization, self.sizes(), dim);

    // 在 out 上填充共轭对称性
    at::native::_fft_fill_with_conjugate_symmetry_(out, dim);

    // 返回结果 out
    return out;
}

// n-dimensional complex to real IFFT
Tensor _fft_c2r_cufft(const Tensor& self, IntArrayRef dim, int64_t normalization, int64_t lastdim) {
    // 检查输入张量 self 是否为复数类型
    TORCH_CHECK(self.is_complex());

    // 获取输入张量 self 的大小
    auto in_sizes = self.sizes();

    // 根据输入大小创建输出大小的向量，将 dim 最后一个维度的大小替换为 lastdim
    DimVector out_sizes(in_sizes.begin(), in_sizes.end());
    out_sizes[dim.back()] = lastdim;

    // 根据输出大小创建空张量 output，数据类型与 self 相同的实数类型
    auto output = at::empty(out_sizes, self.options().dtype(c10::toRealValueType(self.scalar_type())));

    // 如果使用优化的 cuFFT 路径
    if (use_optimized_cufft_path(dim)) {
        Tensor temp;
        // 复数到实数的 FFT 可能会覆盖输入缓冲区，因此必须始终克隆（gh-34551）
        temp = self.clone(MemoryFormat::Contiguous);
        // 执行 FFT 操作
        _exec_fft(output, temp, out_sizes, dim, /*forward=*/false);
    } else {
        // 首先完成任何 C2C 变换
        Tensor temp;
        if (dim.size() > 1) {
            // 执行复数到复数的 FFT 变换
            temp = _fft_c2c_cufft(
                self, dim.slice(0, dim.size() - 1),
                static_cast<int64_t>(fft_norm_mode::none), /*forward=*/false);
        } else {
            // 复数到实数的 FFT 可能会覆盖输入缓冲区，因此必须始终克隆（gh-34551）
            temp = self.clone(MemoryFormat::Contiguous);
        }

        // 最后，执行一维的 C2R 变换
        // TODO: 可以在同一个 cuFFT 操作中转换最多 2 个其他维度
        _exec_fft(output, temp, out_sizes, dim.back(), /*forward=*/false);
    }

    // 对输出张量 output 进行归一化处理，并返回结果
    return _fft_apply_normalization(output, normalization, out_sizes, dim);
}

Tensor& _fft_c2r_cufft_out(const Tensor& self, IntArrayRef dim,
                           int64_t normalization, int64_t lastdim, Tensor& out) {
    // 调用 _fft_c2r_cufft 函数执行复数到实数的 FFT 转换，结果存储在 result 中
    auto result = _fft_c2r_cufft(self, dim, static_cast<int64_t>(fft_norm_mode::none), lastdim);

    // 对输出 out 进行归一化处理，并返回
    return _fft_apply_normalization_out(out, result, normalization, result.sizes(), dim);
}

// n-dimensional complex to complex FFT/IFFT
Tensor _fft_c2c_cufft(const Tensor& self, IntArrayRef dim, int64_t normalization, bool forward) {
    // 检查输入张量 self 是否为复数类型
    TORCH_CHECK(self.is_complex());

    // 如果维度为空，则直接克隆返回
    if (dim.empty()) {
        return self.clone();
    }

    // 获取输入张量 self 的大小
    auto out_sizes = self.sizes();

    // 根据输入大小创建与 self 相同的空输出张量 output
    auto output = at::empty(out_sizes, self.options());

    // 执行任意数量的复数到复数的 FFT 变换
    DimVector sorted_dims(dim.begin(), dim.end());
    auto working_tensor = self;
    while (true) {
        // 每次都重新排序维度，因为 _exec_fft 会重新分配输出的步长
        auto strides = working_tensor.strides();
    // 使用 lambda 表达式对 sorted_dims 中的维度按照 strides 数组中的值进行降序排序
    std::sort(sorted_dims.begin(), sorted_dims.end(),
              [&](int64_t a, int64_t b) { return strides[a] > strides[b]; });

    // 确定要处理的维度数，取 sorted_dims 和 cufft_max_ndim 中较小的值
    const auto max_dims = std::min(static_cast<size_t>(cufft_max_ndim), sorted_dims.size());
    // 从 sorted_dims 中选择最大的 max_dims 个维度作为 first_dims
    auto first_dims = IntArrayRef(sorted_dims).slice(sorted_dims.size() - max_dims, max_dims);

    // 调用 _exec_fft 函数执行快速傅里叶变换操作
    _exec_fft(output, working_tensor, out_sizes, first_dims, forward);
    
    // 调整 sorted_dims 的大小，移除已经处理过的维度
    sorted_dims.resize(sorted_dims.size() - max_dims);

    // 如果 sorted_dims 已经为空，跳出循环
    if (sorted_dims.empty()) {
      break;
    }

    // 如果 working_tensor 与 self 相同，将 output 移动到 working_tensor 中，并创建一个新的 output 张量
    if (working_tensor.is_same(self)) {
      working_tensor = std::move(output);
      output = at::empty(out_sizes, self.options());
    } else {
      // 否则交换 output 和 working_tensor
      std::swap(output, working_tensor);
    }
  }

  // 返回进行正规化后的 FFT 结果
  return _fft_apply_normalization(output, normalization, out_sizes, dim);
}

// 结束当前命名空间 at::native

Tensor& _fft_c2c_cufft_out(const Tensor& self, IntArrayRef dim,
                           int64_t normalization, bool forward, Tensor& out) {
    // 调用 _fft_c2c_cufft 函数执行 FFT 变换，并返回结果
    auto result = _fft_c2c_cufft(self, dim, static_cast<int64_t>(fft_norm_mode::none), forward);
    // 将 FFT 结果应用标准化，并将结果存入输出张量 out 中
    return _fft_apply_normalization_out(out, result, normalization, result.sizes(), dim);
}

// 结束当前命名空间 at::native
```