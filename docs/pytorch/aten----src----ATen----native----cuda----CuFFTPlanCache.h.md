# `.\pytorch\aten\src\ATen\native\cuda\CuFFTPlanCache.h`

```py
// 包含 ATen 库的必要头文件
#include <ATen/Config.h>
#include <ATen/core/DimVector.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/CuFFTUtils.h>
#include <ATen/native/utils/ParamsHash.h>
#include <c10/util/accumulate.h>
#include <c10/util/irange.h>

// 包含 CUDA FFT 库的头文件
#include <cufft.h>
#include <cufftXt.h>

// 包含标准库头文件
#include <limits>
#include <list>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

// 命名空间声明：at::native::detail
namespace at { namespace native { namespace detail {

// 枚举类型，表示 FFT 的类型
enum class CuFFTTransformType : int8_t {
  C2C,  // 复数到复数
  R2C,  // 实数到复数
  C2R,  // 复数到实数
};

// 用于计算参数哈希值的结构体，作为计划缓存的键
struct CuFFTParams
{
  int64_t signal_ndim_; // 信号维度数，介于 1 到 max_rank 之间，即 1 <= signal_ndim <= 3
  int64_t sizes_[max_rank + 1]; // 包括额外的批次维度
  int64_t input_strides_[max_rank + 1]; // 输入数据步长数组
  int64_t output_strides_[max_rank + 1]; // 输出数据步长数组
  CuFFTTransformType fft_type_; // FFT 类型
  ScalarType value_type_; // 数据类型

  CuFFTParams() = default;

  // 构造函数，初始化 CuFFTParams 对象
  CuFFTParams(IntArrayRef in_strides, IntArrayRef out_strides,
      IntArrayRef signal_sizes, CuFFTTransformType fft_type, ScalarType value_type) {
    // 用于哈希计算的填充位必须为零
    memset(this, 0, sizeof(*this));
    signal_ndim_ = signal_sizes.size() - 1;
    fft_type_ = fft_type;
    value_type_ = value_type;

    // 断言确保输入输出步长和信号尺寸数组的大小一致，以及信号维度在有效范围内
    TORCH_INTERNAL_ASSERT(in_strides.size() == signal_sizes.size());
    TORCH_INTERNAL_ASSERT(out_strides.size() == signal_sizes.size());
    TORCH_INTERNAL_ASSERT(1 <= signal_ndim_ && signal_ndim_ <= max_rank);

    // 复制信号尺寸、输入输出步长到对应数组中
    std::copy(signal_sizes.cbegin(), signal_sizes.cend(), sizes_);
    std::copy(in_strides.cbegin(), in_strides.cend(), input_strides_);
    std::copy(out_strides.cbegin(), out_strides.cend(), output_strides_);
  }
};

// 静态断言检查 CuFFTParams 是否为平凡类型
static_assert(std::is_trivial<CuFFTParams>::value, "");

// 根据变量表示的复杂输入情况返回 true
inline bool cufft_complex_input(CuFFTTransformType type) {
  switch (type) {
    case CuFFTTransformType::C2C:
    case CuFFTTransformType::C2R:
      return true;

    case CuFFTTransformType::R2C:
      return false;
  }
  // 断言，不应该到达这里
  TORCH_INTERNAL_ASSERT(false);
}

// 根据变量表示的复杂输出情况返回 true
inline bool cufft_complex_output(CuFFTTransformType type) {
  switch (type) {
    case CuFFTTransformType::C2C:
    case CuFFTTransformType::R2C:
      return true;

    case CuFFTTransformType::C2R:
      return false;
  }
  // 断言，不应该到达这里
  TORCH_INTERNAL_ASSERT(false);
}

// 根据输入和输出是否复数创建 FFT 变换类型的枚举值
inline CuFFTTransformType GetCuFFTTransformType(bool complex_input, bool complex_output) {
  if (complex_input && complex_output) {
    return CuFFTTransformType::C2C;
  } else if (complex_input && !complex_output) {
    return CuFFTTransformType::C2R;
  } else if (!complex_input && complex_output) {
    return CuFFTTransformType::R2C;
  }


    // 返回 CuFFTTransformType::R2C，表示实数到复数的 FFT 变换类型
    // 这里的代码将当前函数返回值设为 CuFFTTransformType::R2C
  TORCH_INTERNAL_ASSERT(false, "Real to real FFTs are not supported");


    // 如果执行到这里，表示不支持实数到实数的 FFT 变换
    // 这个断言用于确保不会执行到这里，如果执行到这里，会输出错误信息："Real to real FFTs are not supported"
// 定义 CuFFTHandle 类，封装 cuFFT 的句柄管理
class CuFFTHandle {
  ::cufftHandle handle_; // cuFFT 句柄对象

public:
  // 构造函数，创建 cuFFT 句柄
  CuFFTHandle() {
    CUFFT_CHECK(cufftCreate(&handle_));
  }

  // 返回 cuFFT 句柄的引用
  ::cufftHandle & get() { return handle_; }

  // 返回 const 类型的 cuFFT 句柄引用
  const ::cufftHandle & get() const { return handle_; }

  // 析构函数，销毁 cuFFT 句柄
  ~CuFFTHandle() {
    // 为了避免 rocFFT 中的双重释放句柄问题，这里不使用 cufftDestroy()
#if !defined(USE_ROCM)
    cufftDestroy(handle_);
#endif
  }
};

// 检查一个整数是否是 2 的幂次方
__forceinline__
static bool is_pow_of_two(int64_t x) {
  return (x & (x - 1)) == 0;
}

// 定义 cuFFT 中使用的数据类型
using cufft_size_type = long long int;

// 定义 CuFFTDimVector 类型，用于表示 cuFFT 中的维度向量
using CuFFTDimVector = c10::SmallVector<cufft_size_type, at::kDimVectorStaticSize>;

// 结构体，表示 CuFFT 中用于计划变换的数据布局
// 参见 NOTE [ cuFFT Embedded Strides ].
struct CuFFTDataLayout {
  CuFFTDimVector embed; // 嵌入的维度向量
  cufft_size_type stride, dist; // 步长和距离
  bool must_clone, simple; // 是否需要克隆数据和是否简单布局
};

// 返回一个连续信号的 cuFFT 嵌入表示
// 如果输入需要克隆，则返回简单布局并设置 must_clone 标志
// 参见 NOTE [ cuFFT Embedded Strides ].
inline CuFFTDataLayout cufft_simple_embed(IntArrayRef sizes, bool onesided) {
  CuFFTDataLayout layout; // 创建一个 CuFFTDataLayout 结构体对象
  layout.simple = true; // 设置布局为简单布局
  layout.must_clone = false; // 设置不需要克隆数据
  layout.embed.assign(sizes.cbegin() + 1, sizes.cend()); // 分配嵌入的维度向量
  if (onesided) {
    layout.embed.back() = sizes.back() / 2 + 1; // 如果是单边数据，调整最后一个维度
  }
  layout.stride = 1; // 设置步长为 1
  layout.dist = 1; // 设置距离为 1
  for (const auto& len : layout.embed) {
    layout.dist *= len; // 计算距离，即各维度大小的乘积
  }
  return layout; // 返回布局对象
}

// 将步长转换为 cuFFT 的嵌入表示
// 如果步长不能被嵌入，则返回简单布局并设置 must_clone 标志
// 参见 NOTE [ cuFFT Embedded Strides ].
inline CuFFTDataLayout as_cufft_embed(IntArrayRef strides, IntArrayRef sizes, bool onesided) {
  const auto signal_ndim = strides.size() - 1; // 信号维度数
  CuFFTDataLayout layout; // 创建一个 CuFFTDataLayout 结构体对象
  auto last_stride = strides[signal_ndim]; // 最后一个维度的步长
  layout.must_clone = (last_stride <= 0); // 设置是否需要克隆数据的标志

  const auto last_dim_size = onesided ?
      sizes[signal_ndim] / 2 + 1 : sizes[signal_ndim]; // 计算最后一个维度的大小
  const auto signal_numel = c10::multiply_integers(sizes.slice(1, sizes.size() - 2)) * last_dim_size; // 计算信号的元素数

  // 零步长是不允许的，即使批量大小为一。如果出现这种情况，设置一个虚拟值。
  if (sizes[0] == 1) {
    layout.dist = signal_numel; // 如果大小为一，设置距离为信号元素数
  } else if (strides[0] == 0) {
    layout.must_clone = true; // 如果步长为零，设置必须克隆数据的标志
  } else {
    layout.dist = strides[0]; // 否则，设置距离为第一个维度的步长
  }

  // 计算嵌入的形状，或者设置必须克隆标志如果步长无法嵌入
  layout.embed.resize(signal_ndim); // 调整嵌入的维度向量大小
  for (auto i = signal_ndim - 1; !layout.must_clone && i > 0; i--) {
    auto stride = strides[i]; // 获取当前维度的步长
    if (sizes[i] == 1) {
      layout.embed[i] = 1; // 如果维度大小为一，设置嵌入为一
    } else if (stride > 0 && stride % last_stride == 0) {
      layout.embed[i] = stride / last_stride; // 如果可以嵌入，计算嵌入的大小
      last_stride = stride;
    } else {
      layout.must_clone = true; // 否则，设置必须克隆数据的标志
    }
  }

  if (layout.must_clone) {
    // 如果输入需要克隆，假设它将是连续的
    layout = cufft_simple_embed(sizes, onesided); // 返回简单的布局对象
    layout.must_clone = true; // 设置必须克隆数据的标志
  } else {
    // 将第一个嵌入尺寸设置为 sizes 数组中的第二个元素
    layout.embed[0] = sizes[1];
    // 设置步长为 strides 数组中与信号维度相对应的值
    layout.stride = strides[signal_ndim];
    // 确定 layout 是否表示简单的嵌入（连续数据）
    layout.simple = [&] {
      // 遍历从 1 到信号维度减 1 的范围
      for (const auto i : c10::irange(1, signal_ndim - 1)) {
        // 如果当前嵌入尺寸不等于对应的 sizes 中的尺寸
        if (layout.embed[i] != sizes[i + 1]) {
          // 返回 false，表示不是简单嵌入
          return false;
        }
      }
      // 检查步长为 1、距离为信号元素数、最后一个嵌入尺寸等于最后一个维度的尺寸
      return (layout.stride == 1 && layout.dist == signal_numel &&
          layout.embed.back() == last_dim_size);
    }();
  }
  // 返回最终确定的 layout 结构
  return layout;
}

// 这个类包含执行 cuFFT 计划所需的所有信息：
//   1. 计划本身
//   2. 执行计划前是否克隆输入
//   3. 需要的工作空间大小
//
// 这个类将是计划缓存中的值。
// 它通过 unique_ptr 拥有原始计划。
class CuFFTConfig {
public:

  // 只实现移动语义足够了。虽然我们已经为计划使用了 unique_ptr，
  // 但仍然删除复制构造函数和赋值操作符，以免意外复制并降低性能。
  CuFFTConfig(const CuFFTConfig&) = delete;
  CuFFTConfig& operator=(CuFFTConfig const&) = delete;

  // 使用 CuFFTParams 初始化对象
  explicit CuFFTConfig(const CuFFTParams& params):
      CuFFTConfig(
          IntArrayRef(params.input_strides_, params.signal_ndim_ + 1),
          IntArrayRef(params.output_strides_, params.signal_ndim_ + 1),
          IntArrayRef(params.sizes_, params.signal_ndim_ + 1),
          params.fft_type_,
          params.value_type_) {}

  // 对于复杂类型，步长以 2 * 元素大小(dtype) 为单位
  // sizes 包含了整个信号的大小，包括批次大小和总是双侧的
  CuFFTConfig(IntArrayRef in_strides, IntArrayRef out_strides,
      IntArrayRef sizes, CuFFTTransformType fft_type, ScalarType dtype):
        fft_type_(fft_type), value_type_(dtype) {

    // 信号大小（不包括批次维度）
    CuFFTDimVector signal_sizes(sizes.begin() + 1, sizes.end());

    // 输入批次大小
    const int64_t batch = sizes[0];
    const int64_t signal_ndim = sizes.size() - 1;

    // 因为 cuFFT 有限的非单位步长支持和各种约束，我们使用一个标志来跟踪
    // 在这个函数中是否需要克隆输入
#if defined(USE_ROCM)
    // 克隆输入以避免 hipfft 破坏输入并导致测试失败
    clone_input = true;
#else
    clone_input = false;
#endif

    // 对于 half 类型，基于实部到复数和复数到实数变换的基本步长不受支持。
    // 由于我们的输出总是连续的，只需要检查实数到复数的情况。
    if (dtype == ScalarType::Half) {
      // half 类型的 cuFFT 需要至少 SM_53 的计算能力
      auto dev_prop = at::cuda::getCurrentDeviceProperties();
      TORCH_CHECK(dev_prop->major >= 5 && !(dev_prop->major == 5 && dev_prop->minor < 3),
               "cuFFT 不支持计算能力小于 SM_53 的 half 类型信号，但包含输入 half 张量的设备仅支持 SM_", dev_prop->major, dev_prop->minor);
      for (const auto i : c10::irange(signal_ndim)) {
        TORCH_CHECK(is_pow_of_two(sizes[i + 1]),
            "cuFFT 仅在计算半精度时支持大小为二的幂的维度，但收到了信号大小", sizes.slice(1));
      }
      clone_input |= in_strides.back() != 1;
    }

    // CuFFTDataLayout in_layout;
    // 根据 clone_input 的布尔值选择不同的输入布局计算方法
    if (clone_input) {
      // 如果 clone_input 为真，则使用简单嵌入函数计算输入布局
      in_layout = cufft_simple_embed(sizes, fft_type == CuFFTTransformType::C2R);
    } else {
      // 如果 clone_input 为假，则使用 as_cufft_embed 函数计算输入布局
      in_layout = as_cufft_embed(in_strides, sizes, fft_type == CuFFTTransformType::C2R);
    }
    // 使用 as_cufft_embed 函数计算输出布局
    auto out_layout = as_cufft_embed(out_strides, sizes, fft_type == CuFFTTransformType::R2C);
    // 确保输出布局不需要克隆，否则抛出错误信息
    TORCH_INTERNAL_ASSERT(!out_layout.must_clone, "Out strides cannot be represented as CuFFT embedding");
    // 更新 clone_input 标志以反映输入布局是否需要克隆
    clone_input |= in_layout.must_clone;

    // 检查是否可以利用简单数据布局优化
    //
    // 参见 native/cuda/SpectralOps.cu 中的 NOTE [ cuFFT Embedded Strides ]。

    // 根据数据类型选择 cuFFT 所需的输入、输出和执行类型
    const bool simple_layout = in_layout.simple && out_layout.simple;
    cudaDataType itype, otype, exec_type;
    const auto complex_input = cufft_complex_input(fft_type);
    const auto complex_output = cufft_complex_output(fft_type);
    if (dtype == ScalarType::Float) {
      itype = complex_input ? CUDA_C_32F : CUDA_R_32F;
      otype = complex_output ? CUDA_C_32F : CUDA_R_32F;
      exec_type = CUDA_C_32F;
    } else if (dtype == ScalarType::Double) {
      itype = complex_input ? CUDA_C_64F : CUDA_R_64F;
      otype = complex_output ? CUDA_C_64F : CUDA_R_64F;
      exec_type = CUDA_C_64F;
    } else if (dtype == ScalarType::Half) {
      itype = complex_input ? CUDA_C_16F : CUDA_R_16F;
      otype = complex_output ? CUDA_C_16F : CUDA_R_16F;
      exec_type = CUDA_C_16F;
    } else {
      // 如果数据类型不支持，抛出错误信息
      TORCH_CHECK(false, "cuFFT doesn't support tensor of type: ", dtype);
    }

    // 禁用 cuFFT 的自动工作区分配，以便使用 THC 分配器
    CUFFT_CHECK(cufftSetAutoAllocation(plan(), /* autoAllocate */ 0));

    size_t ws_size_t;

    // 创建 cuFFT 计划
    if (simple_layout) {
      // 如果是简单数据布局，通过设置 inembed 和 onembed 为 nullptr，告知 cuFFT 使用单位步幅
      // 在这种情况下，cuFFT 忽略 istride、ostride、idist 和 odist，假定它们都等于 1。
      //
      // 参见 native/cuda/SpectralOps.cu 中的 NOTE [ cuFFT Embedded Strides ]。
      CUFFT_CHECK(cufftXtMakePlanMany(plan(), signal_ndim, signal_sizes.data(),
        /* inembed */ nullptr, /* base_istride */ 1, /* idist */ 1, itype,
        /* onembed */ nullptr, /* base_ostride */ 1, /* odist */ 1, otype,
        batch, &ws_size_t, exec_type));
    } else {
      // 如果不是简单数据布局，则使用具体的嵌入和步幅信息创建 cuFFT 计划
      CUFFT_CHECK(cufftXtMakePlanMany(plan(), signal_ndim, signal_sizes.data(),
            in_layout.embed.data(), in_layout.stride, in_layout.dist, itype,
            out_layout.embed.data(), out_layout.stride, out_layout.dist, otype,
            batch, &ws_size_t, exec_type));
    }
    // 将工作区大小转换为 int64_t 类型并存储在 ws_size 中
    ws_size = static_cast<int64_t>(ws_size_t);
  }

  // 返回 cuFFT 计划句柄
  const cufftHandle &plan() const { return plan_ptr.get(); }

  // 返回 cuFFT 变换类型
  CuFFTTransformType transform_type() const { return fft_type_; }

  // 返回数据类型
  ScalarType data_type() const { return value_type_; }

  // 返回是否需要克隆输入
  bool should_clone_input() const { return clone_input; }

  // 返回工作区大小
  int64_t workspace_size() const { return ws_size; }
private:
  // CuFFT 计划句柄指针
  CuFFTHandle plan_ptr;
  // 是否克隆输入数据
  bool clone_input;
  // 工作空间大小
  int64_t ws_size;
  // FFT 类型
  CuFFTTransformType fft_type_;
  // 数值类型
  ScalarType value_type_;
};

#if defined(USE_ROCM)
  // 注意：对于 CUDA 版本 < 10，由于 bug，最大计划数必须为 1023
  constexpr int64_t CUFFT_MAX_PLAN_NUM = 1023;
  // CUFFT 默认缓存大小设置为最大计划数
  constexpr int64_t CUFFT_DEFAULT_CACHE_SIZE = CUFFT_MAX_PLAN_NUM;
#else
  // CUDA 版本 > 10 的默认最大缓存大小选择为 4096，这个数字限制了默认情况下计划缓存的大小
  constexpr int64_t CUFFT_MAX_PLAN_NUM = std::numeric_limits<int64_t>::max();
  // CUFFT 默认缓存大小，可以通过 cufft_set_plan_cache_max_size 进行配置
  constexpr int64_t CUFFT_DEFAULT_CACHE_SIZE = 4096;
#endif
// 确保 CUFFT_MAX_PLAN_NUM 在 size_t 范围内
static_assert(0 <= CUFFT_MAX_PLAN_NUM && CUFFT_MAX_PLAN_NUM <= std::numeric_limits<int64_t>::max(),
              "CUFFT_MAX_PLAN_NUM not in size_t range");
// 确保 CUFFT_DEFAULT_CACHE_SIZE 在 [0, CUFFT_MAX_PLAN_NUM] 范围内
static_assert(CUFFT_DEFAULT_CACHE_SIZE >= 0 && CUFFT_DEFAULT_CACHE_SIZE <= CUFFT_MAX_PLAN_NUM,
              "CUFFT_DEFAULT_CACHE_SIZE not in [0, CUFFT_MAX_PLAN_NUM] range");

// 此缓存假设从键到值的映射不会更改。
// 这 **不是** 线程安全的。在使用此缓存和 try_emplace_value 返回值时，请使用互斥锁。
// 使用此缓存的契约是，只有在 max_size 大于零时才能使用 try_emplace_value。
class CuFFTParamsLRUCache {
public:
  // 定义键值对类型
  using kv_t = typename std::pair<CuFFTParams, CuFFTConfig>;
  // 定义映射类型，使用 CuFFTParams 的引用作为键
  using map_t = typename std::unordered_map<std::reference_wrapper<CuFFTParams>,
                                            typename std::list<kv_t>::iterator,
                                            ParamsHash<CuFFTParams>,
                                            ParamsEqual<CuFFTParams>>;
  // 定义映射迭代器类型
  using map_kkv_iter_t = typename map_t::iterator;

  // 默认构造函数，使用默认缓存大小
  CuFFTParamsLRUCache() : CuFFTParamsLRUCache(CUFFT_DEFAULT_CACHE_SIZE) {}

  // 构造函数，指定最大缓存大小
  CuFFTParamsLRUCache(int64_t max_size) {
    _set_max_size(max_size);
  }

  // 移动构造函数
  CuFFTParamsLRUCache(CuFFTParamsLRUCache&& other) noexcept :
    _usage_list(std::move(other._usage_list)),
    _cache_map(std::move(other._cache_map)),
    _max_size(other._max_size) {}

  // 移动赋值运算符
  CuFFTParamsLRUCache& operator=(CuFFTParamsLRUCache&& other) noexcept {
    _usage_list = std::move(other._usage_list);
    _cache_map = std::move(other._cache_map);
    _max_size = other._max_size;
    return *this;
  }

  // 查找键对应的配置项，如果不存在则插入新配置并返回
  // 返回常量引用，因为一旦创建，CuFFTConfig 就不应该被篡改
  const CuFFTConfig &lookup(CuFFTParams params) {
    // 断言确保最大大小大于零
    AT_ASSERT(_max_size > 0);

    // 在缓存中查找键
    map_kkv_iter_t map_it = _cache_map.find(params);
    // 命中，将使用情况移到列表前部并返回配置项
    if (map_it != _cache_map.end()) {
      _usage_list.splice(_usage_list.begin(), _usage_list, map_it->second);
      return map_it->second->second;
    }

    // 未命中，处理缓存删除逻辑（未完整）
    // 如果缓存使用列表的大小大于等于最大允许大小
    if (_usage_list.size() >= _max_size) {
      // 找到最后一个元素的迭代器
      auto last = _usage_list.end();
      last--;
      // 从缓存映射中删除最后一个元素对应的条目
      _cache_map.erase(last->first);
      // 弹出使用列表的最后一个元素
      _usage_list.pop_back();
    }

    // 在使用列表的最前面构造新的计划，然后插入到缓存映射中
    _usage_list.emplace_front(std::piecewise_construct,
                       std::forward_as_tuple(params),
                       std::forward_as_tuple(params));
    // 获取插入的元素的迭代器
    auto kv_it = _usage_list.begin();
    // 向缓存映射中插入新的条目，使用参数作为键和值
    _cache_map.emplace(std::piecewise_construct,
                std::forward_as_tuple(kv_it->first),
                std::forward_as_tuple(kv_it));
    // 返回插入元素的值部分
    return kv_it->second;
  }

  // 清空缓存映射和使用列表
  void clear() {
    _cache_map.clear();
    _usage_list.clear();
  }

  // 调整缓存的最大大小
  void resize(int64_t new_size) {
    // 设置新的最大大小
    _set_max_size(new_size);
    // 获取当前使用列表的大小
    auto cur_size = _usage_list.size();
    // 如果当前大小超过最大大小
    if (cur_size > _max_size) {
      // 找到需要删除的元素的迭代器位置
      auto delete_it = _usage_list.end();
      for (size_t i = 0; i < cur_size - _max_size; i++) {
        delete_it--;
        // 从缓存映射中删除对应于删除元素的条目
        _cache_map.erase(delete_it->first);
      }
      // 从使用列表中删除超出最大大小的元素
      _usage_list.erase(delete_it, _usage_list.end());
    }
  }

  // 返回缓存映射的大小
  size_t size() const { return _cache_map.size(); }

  // 返回缓存的最大允许大小
  size_t max_size() const noexcept { return _max_size; }

  // 互斥锁，用于线程安全操作
  std::mutex mutex;
private:
  // 只设置大小并进行值检查，不重新调整数据结构。
  void _set_max_size(int64_t new_size) {
    // 在这里检查 0 <= new_size <= CUFFT_MAX_PLAN_NUM。由于
    // CUFFT_MAX_PLAN_NUM 的类型是 size_t，因此我们需要先进行非负性检查。
    TORCH_CHECK(new_size >= 0,
             "cuFFT plan cache size must be non-negative, but got ", new_size);
    TORCH_CHECK(new_size <= CUFFT_MAX_PLAN_NUM,
             "cuFFT plan cache size can not be larger than ", CUFFT_MAX_PLAN_NUM, ", but got ", new_size);
    // 将 new_size 转换为 size_t 类型，然后赋值给 _max_size
    _max_size = static_cast<size_t>(new_size);
  }

  // 使用链表存储使用情况
  std::list<kv_t> _usage_list;
  // 使用 map 存储缓存数据
  map_t _cache_map;
  // 存储最大缓存大小
  size_t _max_size;
};

// 由于 ATen 分为 CPU 构建和 CUDA 构建，我们需要一种方式仅在加载 CUDA 时调用这些函数。
// 我们使用 CUDA 钩子来实现这一目的（位于 cuda/detail/CUDAHooks.cpp），并从实际的本地函数
// 对应项（位于 native/SpectralOps.cpp）调用 hooked 函数，即
// _cufft_get_plan_cache_max_size、_cufft_set_plan_cache_max_size、
// _cufft_get_plan_cache_size 和 _cufft_clear_plan_cache。
int64_t cufft_get_plan_cache_max_size_impl(DeviceIndex device_index);
void cufft_set_plan_cache_max_size_impl(DeviceIndex device_index, int64_t max_size);
int64_t cufft_get_plan_cache_size_impl(DeviceIndex device_index);
void cufft_clear_plan_cache_impl(DeviceIndex device_index);

}}} // namespace at::native::detail
```