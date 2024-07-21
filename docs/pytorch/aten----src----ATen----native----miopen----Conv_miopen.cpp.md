# `.\pytorch\aten\src\ATen\native\miopen\Conv_miopen.cpp`

```
// 定义 TORCH_ASSERT_ONLY_METHOD_OPERATORS 宏，用于指定仅包含方法操作符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 引入 ATen 核心张量库的头文件
#include <ATen/core/Tensor.h>
// 引入 ATen 配置文件
#include <ATen/Config.h>
// 引入 ATen 中的卷积工具函数
#include <ATen/native/ConvUtils.h>

// 如果未定义 AT_PER_OPERATOR_HEADERS 宏，则引入一组操作函数的头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
// 否则，引入另一组操作函数的头文件
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/empty_native.h>
#include <ATen/ops/miopen_convolution_add_relu_native.h>
#include <ATen/ops/miopen_convolution_native.h>
#include <ATen/ops/miopen_convolution_relu_native.h>
#include <ATen/ops/miopen_convolution_transpose_native.h>
#include <ATen/ops/miopen_depthwise_convolution_native.h>
#include <ATen/ops/squeeze.h>
#include <ATen/ops/sum.h>
#include <ATen/ops/zeros.h>
#endif

// 在未启用 AT_ROCM_ENABLED 的条件下
#if !AT_ROCM_ENABLED()

// 进入 at 命名空间下的 native 命名空间
namespace at { namespace native {

// 定义 miopen_convolution 函数，用于 MIOpen 卷积操作
at::Tensor miopen_convolution(
    const Tensor& input, const Tensor& weight, const std::optional<Tensor>& bias_opt /* optional */,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation,
    int64_t groups, bool benchmark, bool deterministic) {
  // 抛出错误信息，因为当前 ATen 未编译支持 MIOpen
  AT_ERROR("miopen_convolution: ATen not compiled with MIOpen support");
}

// 定义 miopen_convolution_backward_input 函数，用于 MIOpen 卷积反向输入操作
at::Tensor miopen_convolution_backward_input(
    IntArrayRef input_size, const at::Tensor& grad_output, const at::Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic) {
  // 抛出错误信息，因为当前 ATen 未编译支持 MIOpen
  AT_ERROR("miopen_convolution_backward_input: ATen not compiled with MIOpen support");
}

// 定义 miopen_convolution_backward_weight 函数，用于 MIOpen 卷积反向权重操作
at::Tensor miopen_convolution_backward_weight(
    IntArrayRef weight_size, const at::Tensor& grad_output, const at::Tensor& input,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic) {
  // 抛出错误信息，因为当前 ATen 未编译支持 MIOpen
  AT_ERROR("miopen_convolution_backward_weight: ATen not compiled with MIOpen support");
}

// 定义 miopen_convolution_backward_bias 函数，用于 MIOpen 卷积反向偏置操作
at::Tensor miopen_convolution_backward_bias(
    const at::Tensor& grad_output) {
  // 抛出错误信息，因为当前 ATen 未编译支持 MIOpen
  AT_ERROR("miopen_convolution_backward_bias: ATen not compiled with MIOpen support");
}

// 定义 miopen_convolution_backward 函数，用于 MIOpen 卷积反向操作
std::tuple<at::Tensor,at::Tensor,at::Tensor> miopen_convolution_backward(
    const at::Tensor& input, const at::Tensor& grad_output, const at::Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic, std::array<bool,3> output_mask) {
  // 抛出错误信息，因为当前 ATen 未编译支持 MIOpen
  AT_ERROR("miopen_convolution_backward: ATen not compiled with MIOpen support");
}

// 定义 miopen_convolution_transpose 函数，用于 MIOpen 转置卷积操作
at::Tensor miopen_convolution_transpose(
    const Tensor& input, const Tensor& weight, const std::optional<Tensor>& bias_opt /* optional */,
    IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation,
    int64_t groups, bool benchmark, bool deterministic) {
  // 抛出错误信息，因为当前 ATen 未编译支持 MIOpen
  AT_ERROR("miopen_convolution_transpose: ATen not compiled with MIOpen support");
}

// 定义 miopen_convolution_transpose_backward_input 函数，用于 MIOpen 转置卷积反向输入操作
    // 定义 miopen_convolution_transpose_backward 函数，该函数用于执行反卷积操作的反向传播
    const at::Tensor& grad_output, const at::Tensor& weight,
    // grad_output：反向传播过程中的梯度输出张量
    // weight：卷积核的权重张量
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation,
    // padding、stride、dilation：反卷积操作的填充、步幅和膨胀参数
    int64_t groups, bool benchmark, bool deterministic) {
    // groups：分组卷积的组数
    // benchmark、deterministic：控制运行效率和确定性的布尔参数

  // 抛出错误，指示 ATen 没有使用 MIOpen 支持编译
  AT_ERROR("miopen_convolution_transpose_backward: ATen not compiled with MIOpen support");
}

// 下面是一系列的函数声明，用于在未编译 MIOpen 支持时抛出错误
at::Tensor miopen_convolution_transpose_backward_weight(
    IntArrayRef weight_size, const at::Tensor& grad_output, const at::Tensor& input,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic) {
  AT_ERROR("miopen_convolution_transpose_backward_weight: ATen not compiled with MIOpen support");
}

// miopen_convolution_transpose_backward 函数声明
std::tuple<at::Tensor,at::Tensor,at::Tensor> miopen_convolution_transpose_backward(
    const at::Tensor& input, const at::Tensor& grad_output, const at::Tensor& weight,
    IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic, std::array<bool,3> output_mask) {
  AT_ERROR("miopen_convolution_transpose_backward: ATen not compiled with MIOpen support");
}

// miopen_depthwise_convolution 函数声明
at::Tensor miopen_depthwise_convolution(
    const Tensor& input, const Tensor& weight, const std::optional<Tensor>& bias_opt /* optional */,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation,
    int64_t groups, bool benchmark, bool deterministic) {
  AT_ERROR("miopen_depthwise_convolution: ATen not compiled with MIOpen support");
}

// miopen_depthwise_convolution_backward_input 函数声明
at::Tensor miopen_depthwise_convolution_backward_input(
    IntArrayRef input_size, const at::Tensor& grad_output, const at::Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic) {
  AT_ERROR("miopen_depthwise_convolution_backward_input: ATen not compiled with MIOpen support");
}

// miopen_depthwise_convolution_backward_weight 函数声明
at::Tensor miopen_depthwise_convolution_backward_weight(
    IntArrayRef weight_size, const at::Tensor& grad_output, const at::Tensor& input,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic) {
  AT_ERROR("miopen_depthwise_convolution_backward_weight: ATen not compiled with MIOpen support");
}

// miopen_depthwise_convolution_backward 函数声明
std::tuple<at::Tensor,at::Tensor,at::Tensor> miopen_depthwise_convolution_backward(
    const at::Tensor& input, const at::Tensor& grad_output, const at::Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic, std::array<bool,3> output_mask) {
  AT_ERROR("miopen_depthwise_convolution_backward: ATen not compiled with MIOpen support");
}

// miopen_convolution_add_relu 函数声明
at::Tensor miopen_convolution_add_relu(
    const at::Tensor& input, const at::Tensor& weight, const at::Tensor& z,
    const std::optional<Scalar>& alpha, const std::optional<Tensor>& bias, IntArrayRef stride,
    IntArrayRef padding, IntArrayRef dilation, int64_t groups) {
  AT_ERROR("miopen_convolution_add_relu: ATen not compiled with MIOpen support");
}

// miopen_convolution_relu 函数声明
at::Tensor miopen_convolution_relu(
    const at::Tensor& input, const at::Tensor& weight, const std::optional<Tensor>& bias,
    IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, int64_t groups) {
  AT_ERROR("miopen_convolution_relu: ATen not compiled with MIOpen support");
}
#else  // AT_ROCM_ENABLED
// 如果未启用 ROCm，则包含以下头文件

#include <ATen/miopen/miopen-wrapper.h>
#include <ATen/miopen/Descriptors.h>
#include <ATen/miopen/Types.h>
#include <ATen/miopen/Utils.h>
#include <ATen/hip/EmptyTensor.h>

#include <ATen/TensorUtils.h>
#include <ATen/native/ConvUtils.h>
#include <c10/util/irange.h>

#include <c10/hip/HIPCachingAllocator.h>

#include <functional>
#include <iterator>
#include <sstream>
#include <algorithm>
#include <memory>
#include <mutex>
#include <stdint.h>
#include <unordered_map>

// 定义最大的解决方案数量
#define AT_MIOPEN_MAX_SOLUTIONS 10

namespace at { namespace native {

// 缩小张量的分组，返回指定分组的张量
Tensor narrowGroup(const Tensor& t, int dim, int group_idx, int64_t groups) {
  auto group_size = t.size(dim) / groups;
  return t.narrow(dim, group_idx * group_size, group_size);
}

// 这个 POD 结构体用于计算参数的哈希值
struct ConvolutionParams
{
  miopenHandle_t handle; // miopen 句柄
  miopenDataType_t dataType; // 数据类型
  int input_size[2 + max_dim]; // 输入尺寸
  int input_stride[2 + max_dim]; // 输入步幅
  int weight_size[2 + max_dim]; // 权重尺寸
  int padding[max_dim]; // 填充
  int stride[max_dim]; // 步幅
  int dilation[max_dim]; // 膨胀率
  int64_t groups; // 分组数量
  bool deterministic; // 是否确定性计算
  int device_id; // 设备 ID，用于区分多个 GPU 的 miopen 句柄
  // 注意：transposed 故意省略：转置只是正向和反向的交换，因此可以重用基准条目
};

// 确保 ConvolutionParams 是 POD 类型，因为在计算哈希时会将其内存内容视为 char*
static_assert(std::is_standard_layout<ConvolutionParams>::value, "ConvolutionParams not POD");

// 设置卷积参数
void setConvolutionParams(
    ConvolutionParams* params, miopenHandle_t handle,
    const at::Tensor& input, const at::Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation,
    int64_t groups, bool deterministic) {

  miopenDataType_t dataType = getMiopenDataType(input); // 获取输入张量的 miopen 数据类型
  memset(params, 0, sizeof(ConvolutionParams)); // 将参数结构体的内存清零
  params->dataType = dataType; // 设置数据类型
  params->handle = handle; // 设置句柄
  // 确保权重和输入张量的维度相同
  for (int i = 0; i != input.dim(); ++i) {
    params->input_size[i] = (int) input.size(i); // 设置输入尺寸
    params->input_stride[i] = (int) input.stride(i); // 设置输入步幅
    params->weight_size[i] = (int) weight.size(i); // 设置权重尺寸
  }
  // 确保填充、步幅和膨胀率的维度相同
  for (size_t i = 0; i != padding.size(); ++i) {
    params->padding[i] = padding[i]; // 设置填充
    params->stride[i] = stride[i]; // 设置步幅
    params->dilation[i] = dilation[i]; // 设置膨胀率
  }
  params->groups = groups; // 设置分组数量
  params->deterministic = deterministic; // 设置是否确定性计算
  int device_id;
  HIP_CHECK(hipGetDevice(&device_id)); // 获取当前设备 ID
  params->device_id = device_id; // 设置设备 ID
}

// 用于方便传递描述符和数据指针的结构体
// 定义结构体 `ConvolutionArgs`，包含 miopen 句柄、卷积参数、输入输出张量描述符、滤波器描述符、输入、输出、权重张量和卷积描述符
struct ConvolutionArgs {
  miopenHandle_t handle;        // miopen 句柄
  ConvolutionParams params;     // 卷积参数
  TensorDescriptor idesc, odesc; // 输入和输出张量描述符
  FilterDescriptor wdesc;       // 滤波器描述符
  const Tensor& input, output, weight; // 输入、输出、权重张量的引用
  ConvolutionDescriptor cdesc;  // 卷积描述符

  // 构造函数，初始化输入、输出、权重张量引用
  ConvolutionArgs(const Tensor& input, const Tensor& output, const Tensor& weight) : input(input), output(output), weight(weight) {
  }
};

// ---------------------------------------------------------------------
//
// Benchmarking
//
// ---------------------------------------------------------------------

// 用于 ConvolutionParams 的哈希计算结构 `ParamsHash`
struct ParamsHash {
  // 计算 ConvolutionParams 的哈希值
  std::size_t operator()(const ConvolutionParams& params) const {
    auto ptr = reinterpret_cast<const uint8_t*>(&params);
    uint32_t value = 0x811C9DC5;
    for (const auto i : c10::irange((int)sizeof(ConvolutionParams))) {
      value ^= ptr[i];
      value *= 0x01000193;
    }
    return (size_t)value;
  }
};

// 用于比较两个 ConvolutionParams 是否相等的结构 `ParamsEqual`
struct ParamsEqual {
  // 比较两个 ConvolutionParams 是否相等
  bool operator()(const ConvolutionParams& a, const ConvolutionParams& b) const {
    auto ptr1 = reinterpret_cast<const uint8_t*>(&a);
    auto ptr2 = reinterpret_cast<const uint8_t*>(&b);
    return memcmp(ptr1, ptr2, sizeof(ConvolutionParams)) == 0;
  }
};

// 泛型类 `BenchmarkCache`，用于存储不同算法的性能测试结果
template <typename T>
struct BenchmarkCache {
  std::mutex mutex; // 互斥锁，保护并发访问
  std::unordered_map<ConvolutionParams, T, ParamsHash, ParamsEqual> map; // 使用哈希表存储 ConvolutionParams 到 T 类型的映射

  // 查找给定 ConvolutionParams 的性能测试结果
  bool find(const ConvolutionParams& params, T* results) {
    std::lock_guard<std::mutex> guard(mutex);
    auto it = map.find(params);
    if (it == map.end()) {
      return false;
    }
    *results = it->second;
    return true;
  }

  // 插入 ConvolutionParams 和其对应的性能测试结果到缓存中
  void insert(const ConvolutionParams& params, const T& results) {
    std::lock_guard<std::mutex> guard(mutex);
    map[params] = results;
  }
};

// 前向卷积算法的性能测试缓存
BenchmarkCache<miopenConvFwdAlgorithm_t> fwd_algos;
// 反向数据卷积算法的性能测试缓存
BenchmarkCache<miopenConvBwdDataAlgorithm_t> bwd_data_algos;
// 反向权重卷积算法的性能测试缓存
BenchmarkCache<miopenConvBwdWeightsAlgorithm_t> bwd_filter_algos;

// 前向卷积工作空间大小的性能测试缓存
BenchmarkCache<size_t> fwd_wssizes;
// 反向数据卷积工作空间大小的性能测试缓存
BenchmarkCache<size_t> bwd_data_wssizes;
// 反向权重卷积工作空间大小的性能测试缓存
BenchmarkCache<size_t> bwd_filter_wssizes;

// 定义 Workspace 结构，用于管理 GPU 工作空间
struct Workspace {
  size_t size;   // 工作空间大小
  void* data;    // 工作空间数据的指针

  // 分配指定大小的 GPU 工作空间
  Workspace(size_t size) : size(size), data(NULL) {
    data = c10::hip::HIPCachingAllocator::raw_alloc(size);
  }

  // 禁用拷贝构造函数
  Workspace(const Workspace&) = delete;

  // 默认移动构造函数
  Workspace(Workspace&&) = default;

  // 默认移动赋值运算符
  Workspace& operator=(Workspace&&) = default;

  // 析构函数，释放 GPU 工作空间
  ~Workspace() {
    if (data) {
      c10::hip::HIPCachingAllocator::raw_delete(data);
    }
  }
};

// 模板结构 `algorithm_search`，用于搜索卷积算法
template<typename algo_t>
struct algorithm_search {
};

// 获取前向或反向卷积的工作空间大小
size_t getWorkspaceSize(
    const ConvolutionArgs& args, const miopenConvFwdAlgorithm_t)
{
    size_t sz = 0;
    // 调用 miopen 函数获取前向卷积所需的工作空间大小
    miopenConvolutionForwardGetWorkSpaceSize(
        args.handle,
        args.wdesc.desc(),
        args.idesc.desc(),
        args.cdesc.desc(),
        args.odesc.desc(),
        &sz);
    return sz;
}

// 获取前向或反向卷积的工作空间大小
size_t getWorkspaceSize(
    const ConvolutionArgs& args, const miopenConvBwdDataAlgorithm_t)
{
    size_t sz = 0;
    // 调用 miopen 函数获取反向数据卷积所需的工作空间大小
    miopenConvolutionBackwardDataGetWorkSpaceSize(
        args.handle,
        args.wdesc.desc(),
        args.odesc.desc(),
        args.cdesc.desc(),
        args.idesc.desc(),
        &sz);
    return sz;
}
    // 调用 miopenConvolutionBackwardDataGetWorkSpaceSize 函数，计算反向卷积操作所需的工作空间大小
    miopenConvolutionBackwardDataGetWorkSpaceSize(
        args.handle,             // Miopen 库的句柄，用于管理运行时状态
        args.odesc.desc(),       // 输出描述符的描述符，描述输出张量的特征
        args.wdesc.desc(),       // 权重描述符的描述符，描述卷积核的特征
        args.cdesc.desc(),       // 卷积描述符的描述符，描述卷积操作的一般属性
        args.idesc.desc(),       // 输入描述符的描述符，描述输入张量的特征
        &sz                      // 存储计算得到的工作空间大小的变量的指针
    );
    // 返回计算得到的工作空间大小
    return sz;
}
// 定义函数结束

size_t getWorkspaceSize(
    const ConvolutionArgs& args, const miopenConvBwdWeightsAlgorithm_t)
{
    // 初始化工作空间大小为0
    size_t sz = 0;
    // 调用miopenConvolutionBackwardWeightsGetWorkSpaceSize函数获取工作空间大小
    miopenConvolutionBackwardWeightsGetWorkSpaceSize(
        args.handle,
        args.odesc.desc(),
        args.idesc.desc(),
        args.cdesc.desc(),
        args.wdesc.desc(),
        &sz);
    // 返回获取的工作空间大小
    return sz;
}

template<typename perf_t>
perf_t getBestAlgorithm(perf_t *perfResults, bool deterministic, int n_algo) {
  // 直接返回性能结果数组的第一个元素
  return perfResults[0];
}

template<>
struct algorithm_search<miopenConvFwdAlgorithm_t> {
  using perf_t = miopenConvAlgoPerf_t;
  using algo_t = miopenConvFwdAlgorithm_t;

  // 默认算法设定为miopenConvolutionFwdAlgoGEMM
  static constexpr auto DEFAULT_ALGO = miopenConvolutionFwdAlgoGEMM;

  // 返回前向算法的性能缓存
  static BenchmarkCache<algo_t>& cache() { return fwd_algos; }

  // 返回前向算法的工作空间大小缓存
  static BenchmarkCache<size_t>& wsscache() { return fwd_wssizes; }

  // 查找最佳前向算法
  static perf_t findAlgorithm(const ConvolutionArgs& args) {
    int perf_count;
    perf_t perf_results;
    // 获取最大工作空间大小
    size_t max_ws_size = getWorkspaceSize(args, DEFAULT_ALGO);
    // 分配工作空间
    Workspace ws(max_ws_size);
    // 调用miopenFindConvolutionForwardAlgorithm函数找到最佳前向算法
    MIOPEN_CHECK(miopenFindConvolutionForwardAlgorithm(
        args.handle,
        args.idesc.desc(), args.input.const_data_ptr(),
        args.wdesc.desc(), args.weight.const_data_ptr(),
        args.cdesc.desc(),
        args.odesc.desc(), args.output.data_ptr(),
        1,        // 只返回最快的算法
        &perf_count,
        &perf_results,
        ws.data,
        ws.size,
        false));
    // 返回找到的最佳性能结果
    return perf_results;
  }

  // 获取最佳前向算法的解决方案
  static miopenConvSolution_t getSolution(const ConvolutionArgs& args, bool force_default) {
    size_t max_solution_count;
    size_t solution_count;
    miopenConvSolution_t solutions[AT_MIOPEN_MAX_SOLUTIONS];
    // 获取前向算法的解决方案数量
    MIOPEN_CHECK(miopenConvolutionForwardGetSolutionCount(
        args.handle,
        args.wdesc.desc(),
        args.idesc.desc(),
        args.cdesc.desc(),
        args.odesc.desc(),
        &max_solution_count));
    // 如果解决方案数量超过最大允许值，抛出错误
    if (max_solution_count > AT_MIOPEN_MAX_SOLUTIONS) {
        AT_ERROR("miopenConvFwdAlgorithm_t getSolution max_solution_count > AT_MIOPEN_MAX_SOLUTIONS");
    }
    // 获取前向算法的解决方案
    MIOPEN_CHECK(miopenConvolutionForwardGetSolution(
        args.handle,
        args.wdesc.desc(),
        args.idesc.desc(),
        args.cdesc.desc(),
        args.odesc.desc(),
        max_solution_count,
        &solution_count,
        solutions));

    // 如果强制使用默认算法
    if (force_default) {
        // 查找默认算法
        for (size_t i=0; i<solution_count; ++i) {
            if (solutions[i].algorithm == (miopenConvAlgorithm_t)DEFAULT_ALGO) {
                return solutions[i];
            }
        }
        // 如果未找到默认算法，选择第一个不需要工作空间的算法
        for (size_t i=0; i<solution_count; ++i) {
            if (solutions[i].workspace_size == 0) {
                return solutions[i];
            }
        }
        // 如果仍未找到，返回第一个解决方案（没有进一步检查）
    }

    // 返回第一个解决方案（通常是最快的）
    return solutions[0];
  }
};

template<>
# 定义模板结构体，用于搜索 miopenConvBwdDataAlgorithm_t 类型的算法
struct algorithm_search<miopenConvBwdDataAlgorithm_t> {
  # 定义别名，perf_t 为 miopenConvAlgoPerf_t 类型，algo_t 为 miopenConvBwdDataAlgorithm_t 类型
  using perf_t = miopenConvAlgoPerf_t;
  using algo_t = miopenConvBwdDataAlgorithm_t;

  # 默认的算法设定为 miopenConvolutionBwdDataAlgoGEMM
  static constexpr auto DEFAULT_ALGO = miopenConvolutionBwdDataAlgoGEMM;
  # 返回反向数据传播算法的性能缓存
  static BenchmarkCache<algo_t>& cache() { return bwd_data_algos; }
  # 返回反向数据传播算法的工作空间大小缓存
  static BenchmarkCache<size_t>& wsscache() { return bwd_data_wssizes; }

  # 寻找最优算法并返回其性能结果
  static perf_t findAlgorithm(const ConvolutionArgs& args) {
    # 性能计数器
    int perf_count;
    # 性能结果
    perf_t perf_results;
    # 获取工作空间的最大尺寸
    size_t max_ws_size = getWorkspaceSize(args, DEFAULT_ALGO);
    # 创建相应大小的工作空间
    Workspace ws(max_ws_size);
    # 调用 miopenFindConvolutionBackwardDataAlgorithm 函数寻找最优的反向数据传播算法
    MIOPEN_CHECK(miopenFindConvolutionBackwardDataAlgorithm(
        args.handle,
        args.odesc.desc(), args.output.const_data_ptr(),
        args.wdesc.desc(), args.weight.const_data_ptr(),
        args.cdesc.desc(),
        args.idesc.desc(), args.input.data_ptr(),
        1,      // just return the fastest
        &perf_count,
        &perf_results,
        ws.data,
        ws.size,
        false));
    # 返回性能最优的算法结果
    return perf_results;
  }

  # 获取解决方案，可选择是否强制使用默认算法
  static miopenConvSolution_t getSolution(const ConvolutionArgs& args, bool force_default) {
    # 最大解决方案计数
    size_t max_solution_count;
    # 实际解决方案计数
    size_t solution_count;
    # 解决方案数组
    miopenConvSolution_t solutions[AT_MIOPEN_MAX_SOLUTIONS];
    # 获取反向数据传播解决方案的数量
    MIOPEN_CHECK(miopenConvolutionBackwardDataGetSolutionCount(
        args.handle,
        args.odesc.desc(),
        args.wdesc.desc(),
        args.cdesc.desc(),
        args.idesc.desc(),
        &max_solution_count));
    # 如果最大解决方案计数超过预设的上限，抛出错误
    if (max_solution_count > AT_MIOPEN_MAX_SOLUTIONS) {
        AT_ERROR("miopenConvBwdDataAlgorithm_t getSolution max_solution_count > AT_MIOPEN_MAX_SOLUTIONS");
    }
    # 获取反向数据传播的解决方案
    MIOPEN_CHECK(miopenConvolutionBackwardDataGetSolution(
        args.handle,
        args.odesc.desc(),
        args.wdesc.desc(),
        args.cdesc.desc(),
        args.idesc.desc(),
        max_solution_count,
        &solution_count,
        solutions));

    # 如果强制使用默认算法
    if (force_default) {
        # 寻找默认算法
        for (size_t i=0; i<solution_count; ++i) {
            if (solutions[i].algorithm == (miopenConvAlgorithm_t)DEFAULT_ALGO) {
                return solutions[i];
            }
        }
        # 默认算法未找到，选择第一个无需工作空间的算法
        for (size_t i=0; i<solution_count; ++i) {
            if (solutions[i].workspace_size == 0) {
                return solutions[i];
            }
        }
        # 若仍未找到合适算法，则选择第一个算法作为备选
    }

    # 返回第一个解决方案
    return solutions[0];
  }
};
    // 获取工作空间的最大尺寸
    size_t max_ws_size = getWorkspaceSize(args, DEFAULT_ALGO);
    // 根据最大尺寸创建工作空间
    Workspace ws(max_ws_size);
    // 查询反向权重卷积算法的性能，并返回最快的算法性能结果
    MIOPEN_CHECK(miopenFindConvolutionBackwardWeightsAlgorithm(
        args.handle,
        args.odesc.desc(), args.output.const_data_ptr(),
        args.idesc.desc(), args.input.const_data_ptr(),
        args.cdesc.desc(),
        args.wdesc.desc(), args.weight.data_ptr(),
        1,      // 只返回最快的算法
        &perf_count,
        &perf_results,
        ws.data,
        ws.size,
        false));
    // 返回性能结果
    return perf_results;
  }

  // 获取卷积的解决方案
  static miopenConvSolution_t getSolution(const ConvolutionArgs& args, bool force_default) {
    size_t max_solution_count;
    size_t solution_count;
    miopenConvSolution_t solutions[AT_MIOPEN_MAX_SOLUTIONS];
    // 获取可用的卷积反向权重算法的数量
    MIOPEN_CHECK(miopenConvolutionBackwardWeightsGetSolutionCount(
        args.handle,
        args.odesc.desc(),
        args.idesc.desc(),
        args.cdesc.desc(),
        args.wdesc.desc(),
        &max_solution_count));
    // 如果可用算法数量超过预定义的最大数量，抛出错误
    if (max_solution_count > AT_MIOPEN_MAX_SOLUTIONS) {
        AT_ERROR("miopenConvBwdWeightsAlgorithm_t getSolution max_solution_count > AT_MIOPEN_MAX_SOLUTIONS");
    }
    // 获取卷积反向权重的解决方案
    MIOPEN_CHECK(miopenConvolutionBackwardWeightsGetSolution(
        args.handle,
        args.odesc.desc(),
        args.idesc.desc(),
        args.cdesc.desc(),
        args.wdesc.desc(),
        max_solution_count,
        &solution_count,
        solutions));

    if (force_default) {
        // 如果强制使用默认算法，查找默认算法
        for (size_t i=0; i<solution_count; ++i) {
            if (solutions[i].algorithm == (miopenConvAlgorithm_t)DEFAULT_ALGO) {
                return solutions[i];
            }
        }
        // 如果未找到默认算法，选择第一个不需要工作空间的算法
        for (size_t i=0; i<solution_count; ++i) {
            if (solutions[i].workspace_size == 0) {
                return solutions[i];
            }
        }
        // 如果仍未找到适合的算法，返回第一个解决方案
        // 这种情况下希望选择最佳算法
    }

    // 返回第一个解决方案
    return solutions[0];
  }
};

// 模板函数，根据给定参数查找算法并缓存结果
template<typename algo_t>
void findAlgorithm(const ConvolutionArgs& args, bool benchmark, algo_t* algo) {
  // 使用算法搜索类来进行算法查找
  using search = algorithm_search<algo_t>;
  // 获取算法缓存和工作区缓存的引用
  auto& cache = search::cache();
  auto& wsscache = search::wsscache();

  // 如果缓存中已存在相同参数和算法的结果，则直接返回
  if (cache.find(args.params, algo)) {
    return;
  }

  // 如果参数要求确定性并且不是基准测试，则使用默认算法
  if (args.params.deterministic && !benchmark) {
    *algo = search::DEFAULT_ALGO;
  }

  // 再次检查缓存，因为其他线程可能已经对算法进行了基准测试
  if (cache.find(args.params, algo)) {
    return;
  }

  // 执行算法查找，获取性能结果，并将结果转换为指定的算法类型
  auto perfResults = search::findAlgorithm(args);
  *algo = reinterpret_cast<algo_t&>(perfResults);

  // 将结果插入缓存中
  cache.insert(args.params, *algo);
  // 将工作区大小插入工作区缓存中
  wsscache.insert(args.params, perfResults.memory);

  // 如果需要清空 CuDNN 的 benchmark 缓存，则执行清空操作
  if (at::native::_cudnn_get_conv_benchmark_empty_cache()) {
      c10::hip::HIPCachingAllocator::emptyCache();
  }
}

// 模板函数，选择适合的算法并返回工作区对象
template<typename algo_t>
Workspace chooseAlgorithm(
    const ConvolutionArgs& args,
    bool benchmark,
    algo_t* algo)
{
  // 调用 findAlgorithm 函数查找并设置算法
  findAlgorithm(args, benchmark, algo);

  // 使用算法搜索类来进行算法搜索
  using search = algorithm_search<algo_t>;
  size_t workspace_size;
  // 从工作区缓存中查找特定参数的工作区大小
  search::wsscache().find(args.params, &workspace_size);
  try {
    // 尝试创建指定大小的工作区对象并返回
    return Workspace(workspace_size);
  } catch (const std::exception& e) {
    hipGetLastError(); // 清除内存不足的错误

    // 切换到默认算法，并记录到缓存中，以防止进一步的内存不足错误
    *algo = search::DEFAULT_ALGO;
    workspace_size = getWorkspaceSize(args, *algo);
    search::cache().insert(args.params, *algo);
    search::wsscache().insert(args.params, workspace_size);
    return Workspace(workspace_size);
  }
}

// 模板函数，选择解决方案并返回工作区对象
template<typename algo_t>
Workspace chooseSolution(const ConvolutionArgs& args, uint64_t* solution_id)
{
  // 使用算法搜索类来获取解决方案
  using search = algorithm_search<algo_t>;
  miopenConvSolution_t solution = search::getSolution(args, false);
  try {
    // 尝试获取解决方案的标识和工作区大小，并返回工作区对象
    *solution_id = solution.solution_id;
    return Workspace(solution.workspace_size);
  } catch (const std::exception& e) {
    hipGetLastError(); // 清除内存不足的错误

    // 切换到默认算法，并获取其解决方案，记录标识和工作区大小，并返回工作区对象
    solution = search::getSolution(args, true);
    *solution_id = solution.solution_id;
    return Workspace(solution.workspace_size);
  }
}

// ---------------------------------------------------------------------
//
// Bias addition
//
// ---------------------------------------------------------------------

// 原位操作，用于在卷积输出上添加偏置
void miopen_convolution_add_bias_(CheckedFrom c, const TensorArg& output, const TensorArg& bias)
{
  // 检查输出和偏置张量的数据类型是否相同
  checkAllSameType(c, {output, bias});
  // 检查输出和偏置张量是否在同一 GPU 上
  checkAllSameGPU(c, {output, bias});
  // 检查偏置张量的尺寸是否与输出的通道维度尺寸匹配
  checkSize(c, bias, { output->size(output_channels_dim) });

  // 定义张量描述符 bdesc 和 odesc
  TensorDescriptor bdesc, odesc;

  // 推荐适合输出张量的内存格式
  auto memory_format = output->suggest_memory_format();

  // 创建形状为 output.dim() 的形状向量，所有元素初始化为 1
  std::vector<int64_t> shape( output->dim(), 1);
  // 将输出张量的特定维度（output_channels_dim）设置为 -1
  shape[output_channels_dim] = -1;
  // 重新整形偏置张量，并确保其连续性，使用推荐的内存格式
  at::Tensor bias_contig =  bias->reshape(shape).contiguous(memory_format);
  // 确保 NC11 的步幅遵循特定的公式
  bias_contig.resize_(bias_contig.sizes(), memory_format );

  // TODO: 由于 MIOpen 不支持 NHWC 格式的偏置，暂时使用此解决方法
  // 详见问题 #64426
  // 将偏置张量加到输出张量上
  output->add_( bias_contig );

  /* MIOpen 不支持 NHWC 格式的偏置；等支持添加后激活此部分。
  bdesc.set( bias_contig );
  odesc.set(*output);

  auto handle = getMiopenHandle();
  auto dataType = getMiopenDataType(*bias);
  Constant one(dataType, 1);
  Constant zero(dataType, 0);

  MIOPEN_CHECK(miopenConvolutionForwardBias(handle, &one, bdesc.desc(), bias->const_data_ptr(),
                                     &zero, odesc.desc(), output->data_ptr()));
  */
}

// 在 src/Aten/native/cudnn/Conv.cpp 中查看关于卷积设计的注意事项 [ Convolution design ]


// ---------------------------------------------------------------------
//
// 卷积前向 / 转置卷积反向
//
// ---------------------------------------------------------------------

// raw API 直接调用 MIOpen。

// 这种方式不应该通过 ATen 直接暴露出来的原因有几个：
//
//    - 它将输出作为参数（应该由函数计算！）
//    - 它不进行输入检查
//    - 它不调整输出大小（假设输出已经正确大小）

void raw_miopen_convolution_forward_out(
    const Tensor& output, const Tensor& input, const Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic) {

# 定义函数开始，接受两个布尔类型参数：benchmark用于选择是否使用基准测试，deterministic用于指定是否确定性运行。

  auto dataType = getMiopenDataType(input);

  # 获取输入数据的Miopen数据类型。
  miopenConvolutionMode_t c_mode = miopenConvolution;

  # 定义Miopen卷积模式为miopenConvolution。

  ConvolutionArgs args{ input, output, weight };

  # 创建ConvolutionArgs对象args，用于存储卷积运算的输入、输出和权重数据。
  args.handle = getMiopenHandle();

  # 获取Miopen的句柄，并将其赋值给args的handle成员变量，用于后续Miopen库函数的调用。

  setConvolutionParams(&args.params, args.handle, input, weight, padding, stride, dilation, groups, deterministic);

  # 设置卷积参数，包括填充、步幅、膨胀率、分组卷积等参数，将结果存储在args.params中。

  args.idesc.set(input);

  # 使用输入数据input设置args的输入描述符idesc。
  args.wdesc.set(weight, input.suggest_memory_format(), 0);

  # 使用权重数据weight、输入数据的内存格式和偏移量0设置args的权重描述符wdesc。

  args.odesc.set(output);

  # 使用输出数据output设置args的输出描述符odesc。
  args.cdesc.set(dataType, c_mode, input.dim() - 2, args.params.padding, args.params.stride, args.params.dilation, args.params.groups, deterministic);

  # 使用数据类型dataType、卷积模式c_mode、输入维度减2（为何减2见具体背景）以及args.params中的卷积相关参数设置args的卷积描述符cdesc。

  if (benchmark) {

  # 如果benchmark为true，则执行基准测试分支。
      miopenConvFwdAlgorithm_t fwdAlg;

      # 定义Miopen前向卷积算法变量fwdAlg。
      Workspace workspace = chooseAlgorithm(args, benchmark, &fwdAlg);

      # 选择合适的卷积算法，将结果存储在workspace中，并更新fwdAlg的值。

      Constant one(dataType, 1);
      Constant zero(dataType, 0);

      # 定义常量one和zero，类型为dataType，值分别为1和0。

      MIOPEN_CHECK(miopenConvolutionForward(
        args.handle,
        &one, args.idesc.desc(), input.const_data_ptr(),
        args.wdesc.desc(), weight.const_data_ptr(),
        args.cdesc.desc(), fwdAlg, &zero,
        args.odesc.desc(), output.data_ptr(), workspace.data, workspace.size));

      # 调用Miopen的卷积前向传播函数miopenConvolutionForward，进行卷积运算。

  } else {

  # 如果benchmark为false，则执行非基准测试分支。
      uint64_t solution_id;
      Workspace workspace = chooseSolution<miopenConvFwdAlgorithm_t>(args, &solution_id);

      # 定义uint64_t类型的solution_id，并选择适当的卷积解决方案，将结果存储在workspace中，并更新solution_id的值。

      MIOPEN_CHECK(miopenConvolutionForwardImmediate(
        args.handle,
        args.wdesc.desc(), weight.const_data_ptr(),
        args.idesc.desc(), input.const_data_ptr(),
        args.cdesc.desc(),
        args.odesc.desc(), output.data_ptr(), workspace.data, workspace.size, solution_id));

      # 调用Miopen的即时卷积前向传播函数miopenConvolutionForwardImmediate，进行卷积运算。
  }
Tensor miopen_convolution_forward(
    CheckedFrom c,  // 检查来源，用于调试和错误处理
    const TensorArg& input, const TensorArg& weight,  // 输入和权重张量的参数信息
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,  // 填充、步长、扩张和组信息
    bool benchmark, bool deterministic)  // 是否使用基准模式和确定性模式的标志位
{
  checkAllSameType(c, {input, weight});  // 检查输入和权重张量的类型是否相同
  checkAllSameGPU(c, {input, weight});  // 检查输入和权重张量是否在同一GPU上

  auto memory_format = at::MemoryFormat::Contiguous;  // 内存格式，默认为连续格式
  if (miopen_conv_use_channels_last(*input, *weight)) {  // 如果使用通道最后的内存格式
    memory_format = (weight->ndimension() == 5) ? /*at::MemoryFormat::ChannelsLast3d*/at::MemoryFormat::Contiguous : at::MemoryFormat::ChannelsLast;  // 根据权重张量的维度选择内存格式
  }

  Tensor output_t = at::detail::empty_cuda(
      conv_output_size(input->sizes(), weight->sizes(),
                       padding, stride, dilation),
      input->options().memory_format(memory_format));  // 创建一个空的CUDA张量作为输出，指定内存格式

  if (output_t.numel() == 0) {  // 如果输出张量的元素数为0，则直接返回空张量
    return output_t;
  }

  // 避免在反向传播时对"output"的歧义
  TensorArg output{ output_t, "result", 0 };  // 输出张量参数对象

  // 对卷积操作的形状进行检查
  convolution_shape_check(c, input, weight, output, padding, stride, dilation, groups);

  // 详见 issue #4500
  Tensor weight_contig = weight->contiguous(memory_format);  // 获取连续的权重张量
  // 确保NC11步幅遵循特定的公式
  weight_contig.resize_(weight_contig.sizes(), memory_format);  // 使用指定的内存格式重新调整权重张量
  Tensor input_contig = input->contiguous(memory_format);  // 获取连续的输入张量
  input_contig.resize_(input_contig.sizes(), memory_format);  // 使用指定的内存格式重新调整输入张量

  // 执行原始的miopen卷积前向计算，将结果存储在输出张量中
  raw_miopen_convolution_forward_out(
      *output, input_contig, weight_contig,
      padding, stride, dilation, groups, benchmark, deterministic);

  return *output;  // 返回计算后的输出张量
}

Tensor miopen_convolution(
    const Tensor& input_t, const Tensor& weight_t, const std::optional<Tensor>& bias_t_opt,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation,
    int64_t groups, bool benchmark, bool deterministic)
{
  // 查看[注意：为可选张量删除的包装层]
  c10::MaybeOwned<Tensor> bias_t_maybe_owned = at::borrow_from_optional_tensor(bias_t_opt);  // 从可选张量中借用偏置张量
  const Tensor& bias_t = *bias_t_maybe_owned;  // 获取实际的偏置张量

  TensorArg input  { input_t,  "input",  1 },  // 输入张量参数对象
            weight { weight_t, "weight", 2 },  // 权重张量参数对象
            bias   { bias_t,   "bias",   3 };  // 偏置张量参数对象
  CheckedFrom c = "miopen_convolution";  // 检查来源为"miopen_convolution"
  auto output_t = miopen_convolution_forward(
    c, input, weight, padding, stride, dilation, groups, benchmark, deterministic);  // 调用miopen卷积前向计算

  if (bias->defined()) {  // 如果存在定义的偏置张量
    miopen_convolution_add_bias_(c, { output_t, "result", 0 }, bias);  // 对输出张量添加偏置
  }
  return output_t;  // 返回卷积操作的输出张量
}

// 深度卷积
void raw_miopen_depthwise_convolution_forward_out(
    const Tensor& output, const Tensor& input, const Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic) {

# 接受两个布尔类型的参数：benchmark（是否进行基准测试）、deterministic（是否确定性计算）

  auto dataType = getMiopenDataType(input);

  # 根据输入张量获取其对应的 Miopen 数据类型
  ```

  miopenConvolutionMode_t c_mode = miopenDepthwise;

  # 设置 Miopen 的卷积模式为深度卷积模式（depthwise）
  ```

  ConvolutionArgs args{ input, output, weight };
  # 创建 ConvolutionArgs 对象 args，包含输入、输出、权重张量

  args.handle = getMiopenHandle();
  # 获取 Miopen 句柄并设置为 args 的处理句柄

  setConvolutionParams(&args.params, args.handle, input, weight, padding, stride, dilation, groups, deterministic);
  # 根据参数设置卷积的相关参数，填充、步幅、扩张率、分组、确定性等，并更新 args.params

  args.idesc.set(input);
  # 将输入张量的描述符设置为 args 的输入描述符

  args.wdesc.set(weight, input.suggest_memory_format(), 0);
  # 根据权重张量、建议的内存格式和索引，设置 args 的权重描述符

  args.odesc.set(output);
  # 将输出张量的描述符设置为 args 的输出描述符

  args.cdesc.set(dataType, c_mode, input.dim() - 2, args.params.padding, args.params.stride, args.params.dilation, args.params.groups, deterministic);
  # 根据数据类型、卷积模式、张量维度减去2、填充、步幅、扩张率、分组、确定性等设置卷积描述符 args.cdesc

  if (benchmark) {
      miopenConvFwdAlgorithm_t fwdAlg;
      # 定义 Miopen 的前向卷积算法变量 fwdAlg

      Workspace workspace = chooseAlgorithm(args, benchmark, &fwdAlg);
      # 通过 chooseAlgorithm 函数选择适合的算法并获取工作空间

      Constant one(dataType, 1);
      # 创建常数张量 one，数据类型为 dataType，值为 1

      Constant zero(dataType, 0);
      # 创建常数张量 zero，数据类型为 dataType，值为 0

      MIOPEN_CHECK(miopenConvolutionForward(
        args.handle,
        &one, args.idesc.desc(), input.const_data_ptr(),
        args.wdesc.desc(), weight.const_data_ptr(),
        args.cdesc.desc(), fwdAlg, &zero,
        args.odesc.desc(), output.data_ptr(), workspace.data, workspace.size));
      # 执行 Miopen 的前向卷积操作，传入相关参数和数据
  }
  else {
      uint64_t solution_id;
      # 定义解决方案 ID 变量，数据类型为 uint64_t

      Workspace workspace = chooseSolution<miopenConvFwdAlgorithm_t>(args, &solution_id);
      # 通过 chooseSolution 函数选择适合的解决方案并获取工作空间，同时获取解决方案 ID

      MIOPEN_CHECK(miopenConvolutionForwardImmediate(
        args.handle,
        args.wdesc.desc(), weight.const_data_ptr(),
        args.idesc.desc(), input.const_data_ptr(),
        args.cdesc.desc(),
        args.odesc.desc(), output.data_ptr(), workspace.data, workspace.size, solution_id));
      # 执行 Miopen 的立即前向卷积操作，传入相关参数和数据，以及选择的解决方案 ID
  }
}

Tensor miopen_depthwise_convolution_forward(
    CheckedFrom c,
    const TensorArg& input, const TensorArg& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic)
{
  // 检查输入和权重张量的数据类型是否一致
  checkAllSameType(c, {input, weight});
  // 检查输入和权重张量是否在同一GPU上
  checkAllSameGPU(c, {input, weight});

  // 设置默认的内存格式为连续存储
  auto memory_format = at::MemoryFormat::Contiguous;
  // 如果使用了 channels last 的优化，则根据权重张量的维度选择合适的内存格式
  if (miopen_conv_use_channels_last(*input, *weight)) {
    memory_format = (weight->ndimension() == 5) ? /*at::MemoryFormat::ChannelsLast3d*/at::MemoryFormat::Contiguous : at::MemoryFormat::ChannelsLast;
  }

  // 创建一个 CUDA 空张量作为输出，根据给定的内存格式
  Tensor output_t = at::detail::empty_cuda(
      conv_output_size(input->sizes(), weight->sizes(),
                       padding, stride, dilation),
      input->options().memory_format(memory_format));

  // 将输出张量封装成 TensorArg
  TensorArg output{ output_t, "result", 0 };
  // 检查卷积的形状和尺寸
  convolution_shape_check(c, input, weight, output, padding, stride, dilation, groups);

  // See #4500
  // 确保权重张量是连续的，按照指定的内存格式
  Tensor weight_contig = weight->contiguous(memory_format);
  // 确保输入张量是连续的，按照指定的内存格式
  Tensor input_contig = input->contiguous(memory_format);

  // 执行原始的深度可分离卷积正向传播操作
  raw_miopen_depthwise_convolution_forward_out(
      *output, input_contig, weight_contig,
      padding, stride, dilation, groups, benchmark, deterministic);

  // 返回输出张量
  return *output;
}

Tensor miopen_depthwise_convolution(
    const Tensor& input_t, const Tensor& weight_t, const std::optional<Tensor>& bias_t_opt,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation,
    int64_t groups, bool benchmark, bool deterministic)
{
  // See [Note: hacky wrapper removal for optional tensor]
  // 将可选的偏置张量转换为 MaybeOwned 类型
  c10::MaybeOwned<Tensor> bias_t_maybe_owned = at::borrow_from_optional_tensor(bias_t_opt);
  // 获取偏置张量的引用
  const Tensor& bias_t = *bias_t_maybe_owned;

  // 定义输入、权重和偏置张量的 TensorArg
  TensorArg input  { input_t,  "input",  1 },
            weight { weight_t, "weight", 2 },
            bias   { bias_t,   "bias",   3 };
  // 定义 CheckedFrom 对象
  CheckedFrom c = "miopen_depthwise_convolution";
  // 调用深度可分离卷积的前向传播函数，获取输出张量
  auto output_t = miopen_depthwise_convolution_forward(
    c, input, weight, padding, stride, dilation, groups, benchmark, deterministic);
  
  // 如果定义了偏置张量，则对输出张量执行 miopen 卷积添加偏置操作
  if (bias->defined()) {
    miopen_convolution_add_bias_(c, { output_t, "result", 0 }, bias);
  }
  // 返回最终的输出张量
  return output_t;
}

// ---------------------------------------------------------------------
//
// Convolution backward (bias)
//
// ---------------------------------------------------------------------

Tensor miopen_convolution_backward_bias(
    const Tensor& grad_output_t)
{
  // 定义梯度输出的张量参数
  TensorArg grad_output{ grad_output_t, "grad_output", 1 };

  // TODO: MIOpen 不支持 NHWC 格式的偏置
  // 详见问题报告 #64426
  // 创建丢弃维度列表
  std::vector<int64_t> discard_dims;
  for( int i = 0; i < grad_output_t.dim(); i++ ) {
      // 如果维度不是输出通道维度，则添加到丢弃维度列表
      if(i != output_channels_dim ) {
          discard_dims.push_back(i);
      }
  }

  // 对梯度输出张量进行求和并挤压成一维张量
  Tensor outputBias = at::squeeze( at::sum(grad_output_t, discard_dims, true) );
  if( outputBias.dim() == 0 ) {
      // 如果结果是零维张量，则添加一个维度使其成为一维张量
      return outputBias.unsqueeze(0);
  }
  else {
      return outputBias;
  }

/* MIOpen 不支持 NHWC 格式的偏置。一旦支持，取消注释以下代码块。
  auto grad_bias_t = at::empty( { grad_output->size(output_channels_dim) }, grad_output->options());

  TensorArg grad_bias{ grad_bias_t, "result", 0 };

  // 扩展梯度偏置张量的维度描述符
  TensorDescriptor bdesc{grad_bias->expand({1, grad_bias->size(0)}),
                         static_cast<size_t>(grad_output->dim())};
  TensorDescriptor odesc{*grad_output};

  // 获取 MIOpen 句柄并数据类型
  auto handle = getMiopenHandle();
  auto dataType = getMiopenDataType(*grad_bias);
  Constant one(dataType, 1);
  Constant zero(dataType, 0);

  // 执行 MIOpen 反向卷积偏置操作
  MIOPEN_CHECK(miopenConvolutionBackwardBias(handle, &one, odesc.desc(), grad_output->data_ptr(),
                                                   &zero, bdesc.desc(), grad_bias->data_ptr()));
  return *grad_bias;
*/
}

// ---------------------------------------------------------------------
//
// 卷积反向传播（权重）
//
// ---------------------------------------------------------------------

void raw_miopen_convolution_backward_weight_out(
    const Tensor& grad_weight, const Tensor& grad_output, const Tensor& input,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    // 声明一个布尔类型的 benchmark 和 deterministic 参数
    bool benchmark, bool deterministic) {

  // 获取输入数据的 Miopen 数据类型
  auto dataType = getMiopenDataType(input);
  // 设置卷积模式为 miopenConvolution
  miopenConvolutionMode_t c_mode = miopenConvolution;

  // 创建 ConvolutionArgs 对象 args，包含输入、梯度输出和梯度权重
  ConvolutionArgs args{ input, grad_output, grad_weight };
  // 获取 Miopen 句柄
  args.handle = getMiopenHandle();
  // 设置卷积参数到 args.params 结构体
  setConvolutionParams(&args.params, args.handle, input, grad_weight, padding, stride, dilation, groups, deterministic);
  // 设置输入描述器
  args.idesc.set(input);
  // 设置权重描述器
  args.wdesc.set(grad_weight, input.suggest_memory_format(), 0);
  // 设置梯度输出描述器
  args.odesc.set(grad_output);
  // 设置卷积描述器，包括数据类型、卷积模式、维度、填充、步幅、膨胀、组数和确定性标志
  args.cdesc.set(dataType, c_mode, input.dim() - 2, args.params.padding, args.params.stride, args.params.dilation, args.params.groups, deterministic);

  // 如果 benchmark 参数为 true
  if (benchmark) {
      // 定义用于反向权重梯度计算的算法类型变量
      miopenConvBwdWeightsAlgorithm_t bwdFilterAlg;
      // 选择算法并分配工作空间
      Workspace workspace = chooseAlgorithm(args, benchmark, &bwdFilterAlg);

      // 创建常量 one 和 zero，类型为 dataType
      Constant one(dataType, 1);
      Constant zero(dataType, 0);

      // 调用 Miopen API 计算卷积反向权重
      MIOPEN_CHECK(miopenConvolutionBackwardWeights(
          args.handle,
          &one, args.odesc.desc(), grad_output.const_data_ptr(),
          args.idesc.desc(), input.const_data_ptr(),
          args.cdesc.desc(), bwdFilterAlg, &zero,
          args.wdesc.desc(), grad_weight.data_ptr(), workspace.data, workspace.size));
  }
  // 如果 benchmark 参数为 false
  else {
      // 定义用于立即执行反向权重梯度计算的解决方案 ID
      uint64_t solution_id;
      // 选择解决方案并分配工作空间
      Workspace workspace = chooseSolution<miopenConvBwdWeightsAlgorithm_t>(args, &solution_id);

      // 调用 Miopen API 立即执行卷积反向权重计算
      MIOPEN_CHECK(miopenConvolutionBackwardWeightsImmediate(
          args.handle,
          args.odesc.desc(), grad_output.const_data_ptr(),
          args.idesc.desc(), input.const_data_ptr(),
          args.cdesc.desc(),
          args.wdesc.desc(), grad_weight.data_ptr(), workspace.data, workspace.size, solution_id));
  }
//Depthwise backward weights.
// 深度卷积反向传播权重的实现

void raw_miopen_depthwise_convolution_backward_weight_out(
    // 定义输入参数：梯度权重、梯度输出、输入数据
    const Tensor& grad_weight, const Tensor& grad_output, const Tensor& input,
    // 定义填充、步长、扩展、分组数等参数
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    // 是否使用基准模式和确定性模式的标志
    bool benchmark, bool deterministic) {

  // 获取输入数据的数据类型
  auto dataType = getMiopenDataType(input);
  // 设置 miopen 深度卷积模式
  miopenConvolutionMode_t c_mode = miopenDepthwise;

  // 创建卷积参数对象
  ConvolutionArgs args{ input, grad_output, grad_weight };
  // 获取 miopen 句柄
  args.handle = getMiopenHandle();
  // 设置卷积参数
  setConvolutionParams(&args.params, args.handle, input, grad_weight, padding, stride, dilation, groups, deterministic);
  // 设置输入描述符
  args.idesc.set(input);
  // 设置权重描述符
  args.wdesc.set(grad_weight, input.suggest_memory_format(), 0);
  // 设置输出梯度描述符
  args.odesc.set(grad_output);
  // 设置卷积描述符
  args.cdesc.set(dataType, c_mode, input.dim() - 2, args.params.padding, args.params.stride, args.params.dilation, args.params.groups, deterministic);

  // 如果使用基准模式
  if (benchmark) {
      // 选择卷积反向权重算法和工作空间
      miopenConvBwdWeightsAlgorithm_t bwdFilterAlg;
      Workspace workspace = chooseAlgorithm(args, benchmark, &bwdFilterAlg);

      // 创建常数对象（数据类型的常数1和0）
      Constant one(dataType, 1);
      Constant zero(dataType, 0);

      // 调用 miopen 函数执行卷积反向权重计算
      MIOPEN_CHECK(miopenConvolutionBackwardWeights(
          args.handle,
          &one, args.odesc.desc(), grad_output.const_data_ptr(),
          args.idesc.desc(), input.const_data_ptr(),
          args.cdesc.desc(), bwdFilterAlg, &zero,
          args.wdesc.desc(), grad_weight.data_ptr(), workspace.data, workspace.size));
  }
  // 如果不使用基准模式
  else {
      // 选择最优解决方案和相应的工作空间
      uint64_t solution_id;
      Workspace workspace = chooseSolution<miopenConvBwdWeightsAlgorithm_t>(args, &solution_id);

      // 调用 miopen 函数立即执行卷积反向权重计算
      MIOPEN_CHECK(miopenConvolutionBackwardWeightsImmediate(
          args.handle,
          args.odesc.desc(), grad_output.const_data_ptr(),
          args.idesc.desc(), input.const_data_ptr(),
          args.cdesc.desc(),
          args.wdesc.desc(), grad_weight.data_ptr(), workspace.data, workspace.size, solution_id));
  }
}

// miopen 深度卷积反向传播权重的入口函数
Tensor miopen_depthwise_convolution_backward_weight(
    CheckedFrom c,
    // 权重大小、梯度输出、输入数据、填充、步长、扩展、分组数、基准模式、确定性模式等参数
    IntArrayRef weight_size, const TensorArg& grad_output, const TensorArg& input,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic)
{

  // 检查所有输入张量是否为同一类型
  checkAllSameType(c, {grad_output, input});
  // 检查所有输入张量是否在同一个 GPU 上
  checkAllSameGPU(c, {grad_output, input});

  // 默认内存格式为连续
  auto memory_format = at::MemoryFormat::Contiguous;
  // 如果使用通道最后的存储格式
  if (miopen_conv_use_channels_last(*input, *grad_output)) {
    memory_format = (input->ndimension() == 5) ? /*at::MemoryFormat::ChannelsLast3d*/at::MemoryFormat::Contiguous : at::MemoryFormat::ChannelsLast;
  }



  // 如果输入张量是五维的，选择内存格式为 Contiguous；否则选择 ChannelsLast 格式
  Tensor grad_output_contig_t = grad_output->contiguous(memory_format);
  // 确保 NC11 步幅遵循公式
  grad_output_contig_t.resize_(grad_output_contig_t.sizes(), memory_format);
  TensorArg grad_output_contig{ grad_output_contig_t, "grad_output", 1 };

  // 使输入张量采用指定的内存格式进行连续化处理
  Tensor input_contig_t = input->contiguous(memory_format);
  input_contig_t.resize_(input_contig_t.sizes(), memory_format);
  TensorArg input_contig{ input_contig_t, "input", 2 };

  // 创建一个与 grad_output_contig 具有相同大小和内存格式的空张量
  auto grad_weight_t = at::empty(weight_size, grad_output_contig->options(), memory_format);

  // 对卷积的输入、输出、权重进行形状检查
  convolution_shape_check(c, input, grad_weight, grad_output_contig, padding, stride, dilation, groups);

  // 调用深度卷积反向传播权重的原始 miopen 函数
  raw_miopen_depthwise_convolution_backward_weight_out(
      *grad_weight, *grad_output_contig, *input_contig,
      padding, stride, dilation, groups, benchmark, deterministic);

  // 返回计算出的梯度权重张量
  return grad_weight_t;
}

Tensor miopen_depthwise_convolution_backward_weight(
    IntArrayRef weight_size,
    const Tensor& grad_output_t,
    const Tensor& input_t,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic)
{
  // 定义 grad_output 和 input 的 TensorArg 对象，用于函数参数验证
  TensorArg grad_output{ grad_output_t, "grad_output", 1 },
            input{ input_t, "input", 2 };
  // 调用 miopen_depthwise_convolution_backward_weight 函数，返回计算后的 Tensor 结果
  return miopen_depthwise_convolution_backward_weight(
      "miopen_depthwise_convolution_backward_weight",
      weight_size, grad_output, input,
      padding, stride, dilation, groups, benchmark, deterministic);
}

Tensor miopen_convolution_backward_weight(
    CheckedFrom c,
    IntArrayRef weight_size, const TensorArg& grad_output, const TensorArg& input,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic)
{
  // 检查 grad_output 和 input 是否具有相同的数据类型
  checkAllSameType(c, {grad_output, input});
  // 检查 grad_output 和 input 是否在同一个 GPU 上
  checkAllSameGPU(c, {grad_output, input});

  // 初始化内存格式为连续存储
  auto memory_format = at::MemoryFormat::Contiguous;
  // 如果使用 miopen_conv_use_channels_last 函数确定可以使用 Channels Last 内存格式
  if (miopen_conv_use_channels_last(*input, *grad_output)) {
    // 根据输入和梯度输出的维度情况选择内存格式
    memory_format = (input->ndimension() == 5) ? /*at::MemoryFormat::ChannelsLast3d*/at::MemoryFormat::Contiguous : at::MemoryFormat::ChannelsLast;
  }

  // 对 grad_output 进行连续性处理，并设置指定的内存格式
  Tensor grad_output_contig_t = grad_output->contiguous(memory_format);
  // 确保 NC11 步幅遵循指定的公式
  grad_output_contig_t.resize_(grad_output_contig_t.sizes(), memory_format);
  TensorArg grad_output_contig{ grad_output_contig_t, "grad_output", 1 };

  // 对 input 进行连续性处理，并设置指定的内存格式
  Tensor input_contig_t = input->contiguous(memory_format);
  input_contig_t.resize_(input_contig_t.sizes(), memory_format);
  TensorArg input_contig{ input_contig_t, "input", 2};

  // 创建一个与 grad_output_contig 具有相同大小和选项的空 Tensor 作为 grad_weight
  auto grad_weight_t = at::empty(weight_size, grad_output_contig->options(), memory_format);

  // 进行卷积形状的检查，确保参数匹配
  convolution_shape_check(c, input, grad_weight, grad_output_contig, padding, stride, dilation, groups);

  // 调用原始的 miopen_convolution_backward_weight_out 函数计算反向传播的权重梯度
  raw_miopen_convolution_backward_weight_out(
      *grad_weight, *grad_output_contig, *input_contig,
      padding, stride, dilation, groups, benchmark, deterministic);

  // 返回计算后的 grad_weight Tensor
  return grad_weight_t;
}

Tensor miopen_convolution_backward_weight(
    IntArrayRef weight_size,
    const Tensor& grad_output_t,
    const Tensor& input_t,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic)
{
  // 定义 grad_output 和 input 的 TensorArg 对象，用于函数参数验证
  TensorArg grad_output{ grad_output_t, "grad_output", 1 },
            input{ input_t, "input", 2 };
  // 调用 miopen_convolution_backward_weight 函数，返回计算后的 Tensor 结果
  return miopen_convolution_backward_weight(
      "miopen_convolution_backward_weight",
      weight_size, grad_output, input,
      padding, stride, dilation, groups, benchmark, deterministic);
}

Tensor miopen_convolution_transpose_backward_input(
    const Tensor& grad_output_t, const Tensor& weight_t,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation,


注释：以上是对 MiOpen 深度卷积和普通卷积的反向传播函数的代码注释，包括参数验证、内存格式选择、卷积形状检查和计算反向传播梯度等步骤的说明。
    int64_t groups, bool benchmark, bool deterministic)


// 声明函数的参数列表，其中包括一个名为 groups 的 int64_t 类型变量，
// 以及两个 bool 类型变量 benchmark 和 deterministic。
{
  // 定义两个 TensorArg 对象，分别表示梯度输出和权重
  TensorArg grad_output { grad_output_t,  "grad_output", 1 },
            weight      { weight_t, "weight", 2 };
  // 调用 miopen_convolution_forward 函数执行反卷积操作，返回反卷积输入的张量
  return miopen_convolution_forward(
    "miopen_convolution_transpose_backward_input",  // 反卷积操作类型
    grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic);
}

Tensor miopen_convolution_transpose_backward_weight(
    IntArrayRef weight_size,
    const Tensor& grad_output_t,
    const Tensor& input_t,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic)
{
  // 定义两个 TensorArg 对象，分别表示梯度输出和输入
  TensorArg grad_output{ grad_output_t, "grad_output", 1 },
            input{ input_t, "input", 2 };
  // 调用 miopen_convolution_backward_weight 函数执行反卷积权重更新操作，返回更新后的权重张量
  return miopen_convolution_backward_weight(
      "miopen_convolution_backward_weight",  // 反卷积权重更新操作类型
      weight_size, input, grad_output,
      padding, stride, dilation, groups, benchmark, deterministic);
}

std::tuple<at::Tensor,at::Tensor,at::Tensor> miopen_convolution_transpose_backward(
    const at::Tensor& input, const at::Tensor& grad_output_t, const at::Tensor& weight,
    IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic, std::array<bool,3> output_mask) {

  // 对梯度输出张量进行连续化处理
  Tensor grad_output = grad_output_t.contiguous();

  Tensor grad_input, grad_weight, grad_bias;
  // 如果 output_mask[0] 为真，则调用 miopen_convolution_transpose_backward_input 函数计算反卷积输入梯度
  if (output_mask[0]) {
    grad_input = miopen_convolution_transpose_backward_input(grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic);
  }
  // 如果 output_mask[1] 为真，则调用 miopen_convolution_transpose_backward_weight 函数计算反卷积权重梯度
  if (output_mask[1]) {
    grad_weight = miopen_convolution_transpose_backward_weight(weight.sizes(), grad_output, input, padding, stride, dilation, groups, benchmark, deterministic);
  }
  // 如果 output_mask[2] 为真，则调用 miopen_convolution_backward_bias 函数计算反卷积偏置梯度
  if (output_mask[2]) {
    grad_bias = miopen_convolution_backward_bias(grad_output);
  }

  // 返回包含梯度输入、梯度权重和梯度偏置的元组
  return std::tuple<Tensor,Tensor,Tensor>{grad_input, grad_weight, grad_bias};
}

// ---------------------------------------------------------------------
//
// Convolution backward / Transposed convolution forward
//
// ---------------------------------------------------------------------

void raw_miopen_convolution_backward_input_out(
    const at::Tensor& grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic) {
  // 此函数实现原始的 miopen_convolution_backward_input_out 操作，用于反卷积输入梯度计算
}
    bool benchmark, bool deterministic) {

# 定义函数参数，benchmark表示是否启用基准测试，deterministic表示是否确定性计算。

  auto dataType = getMiopenDataType(grad_output);

# 调用函数获取grad_output的数据类型，存储在变量dataType中。

  miopenConvolutionMode_t c_mode = miopenConvolution;

# 定义miopenConvolutionMode_t类型的变量c_mode，并赋值为miopenConvolution，表示使用卷积运算。

  ConvolutionArgs args{ grad_input, grad_output, weight };

# 创建ConvolutionArgs对象args，包含grad_input、grad_output和weight。

  args.handle = getMiopenHandle();

# 调用函数获取miopen的处理句柄，并将其赋值给args对象的handle属性。

  setConvolutionParams(&args.params, args.handle, grad_input, weight, padding, stride, dilation, groups, deterministic);

# 调用函数设置卷积参数，将结果存储在args对象的params属性中。

  args.idesc.set(grad_input);

# 设置args对象的idesc属性，使用grad_input初始化输入描述符。

  args.wdesc.set(weight, grad_output.suggest_memory_format(), 0);

# 设置args对象的wdesc属性，使用weight和grad_output的推荐内存格式初始化权重描述符。

  args.odesc.set(grad_output);

# 设置args对象的odesc属性，使用grad_output初始化输出描述符。

  args.cdesc.set(dataType, c_mode, grad_output.dim() - 2, args.params.padding, args.params.stride, args.params.dilation, args.params.groups, deterministic);

# 设置args对象的cdesc属性，使用dataType、c_mode以及其他参数初始化卷积描述符。

  if (benchmark) {

# 如果benchmark为true，则执行以下代码块。

      miopenConvBwdDataAlgorithm_t bwdDataAlg;

# 声明miopenConvBwdDataAlgorithm_t类型的变量bwdDataAlg，用于存储反向数据卷积的算法选择结果。

      Workspace workspace = chooseAlgorithm(args, benchmark, &bwdDataAlg);

# 调用函数选择反向数据卷积的算法，并将结果存储在workspace对象中。

      Constant one(dataType, 1);
      Constant zero(dataType, 0);

# 创建Constant对象one和zero，分别初始化为dataType类型的1和0。

      MIOPEN_CHECK(miopenConvolutionBackwardData(
          args.handle,
          &one, args.odesc.desc(), grad_output.const_data_ptr(),
          args.wdesc.desc(), weight.const_data_ptr(),
          args.cdesc.desc(), bwdDataAlg, &zero,
          args.idesc.desc(), grad_input.mutable_data_ptr(), workspace.data, workspace.size));

# 调用miopen库的反向数据卷积函数，使用选定的算法和参数进行计算。

  }
  else {

# 如果benchmark为false，则执行以下代码块。

      uint64_t solution_id;
      Workspace workspace = chooseSolution<miopenConvBwdDataAlgorithm_t>(args, &solution_id);

# 声明uint64_t类型的变量solution_id，并调用函数选择反向数据卷积的解决方案，并将解决方案ID存储在solution_id中。

      MIOPEN_CHECK(miopenConvolutionBackwardDataImmediate(
          args.handle,
          args.odesc.desc(), grad_output.const_data_ptr(),
          args.wdesc.desc(), weight.const_data_ptr(),
          args.cdesc.desc(),
          args.idesc.desc(), grad_input.mutable_data_ptr(), workspace.data, workspace.size, solution_id));

# 调用miopen库的即时反向数据卷积函数，使用选定的解决方案ID和参数进行计算。
}

// 在 src/Aten/native/cudnn/Conv.cpp 中查看有关“反向 vs 转置卷积”的说明

// 计算 miopen 卷积的反向输入，返回梯度输入张量
Tensor miopen_convolution_backward_input(
    CheckedFrom c,
    IntArrayRef input_size, const TensorArg& grad_output, const TensorArg& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic)
{
  // 检查梯度输出和权重张量的数据类型是否相同
  checkAllSameType(c, {grad_output, weight});
  // 检查梯度输出和权重张量是否在相同的 GPU 上
  checkAllSameGPU(c, {grad_output, weight});

  // 决定内存格式，默认为连续内存
  auto memory_format = at::MemoryFormat::Contiguous;
  // 如果使用 channels last 的 miopen 卷积
  if (miopen_conv_use_channels_last(*grad_output, *weight)) {
    // 根据权重张量的维度确定内存格式为 ChannelsLast3d 或 Contiguous
    memory_format = (weight->ndimension() == 5) ? /*at::MemoryFormat::ChannelsLast3d*/at::MemoryFormat::Contiguous : at::MemoryFormat::ChannelsLast;
  }

  // 创建空的 CUDA 张量 grad_input_t 作为梯度输入张量
  Tensor grad_input_t = at::detail::empty_cuda(
      input_size, grad_output->options().memory_format(memory_format));

  // 避免在转置卷积时使用 "grad_input"
  TensorArg grad_input{ grad_input_t, "result", 0 };
  // 检查卷积形状是否合法
  convolution_shape_check(c, grad_input, weight, grad_output, padding, stride, dilation, groups);

  // 参考 GitHub issue #4500
  // 确保权重张量是连续的，并使用指定的内存格式
  Tensor weight_contig = weight->contiguous(memory_format);
  // 调整权重张量的大小，使用指定的内存格式
  weight_contig.resize_(weight_contig.sizes(), memory_format);

  // 确保梯度输出张量是连续的，并使用指定的内存格式
  Tensor grad_output_contig = grad_output->contiguous(memory_format);
  // 调整梯度输出张量的大小，使用指定的内存格式
  grad_output_contig.resize_(grad_output_contig.sizes(), memory_format);

  // 执行 miopen 卷积的反向输入计算
  raw_miopen_convolution_backward_input_out(
      *grad_input, grad_output_contig, weight_contig,
      padding, stride, dilation, groups, benchmark, deterministic);

  // 返回计算得到的梯度输入张量
  return *grad_input;
}

// 计算 miopen 卷积的转置前向，返回计算结果张量
Tensor miopen_convolution_transpose_forward(
    CheckedFrom c,
    const TensorArg& grad_output, const TensorArg& weight,
    IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic)
{
  // 计算输入大小
  auto input_size = conv_input_size(grad_output->sizes(), weight->sizes(),
                                    padding, output_padding, stride, dilation, groups);
  // 调用 miopen 卷积的反向输入函数，返回计算结果张量
  return miopen_convolution_backward_input(c, input_size, grad_output, weight,
                                    padding, stride, dilation, groups, benchmark, deterministic);
}

// 计算 miopen 卷积的反向输入，返回计算结果张量
Tensor miopen_convolution_backward_input(
    IntArrayRef input_size, const Tensor& grad_output_t, const Tensor& weight_t,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic)
{
  // 定义梯度输出张量和权重张量的 TensorArg
  TensorArg grad_output{ grad_output_t, "grad_output", 1 },
            weight{ weight_t, "weight", 2 };
  // 调用 miopen 卷积的反向输入函数，返回计算结果张量
  return miopen_convolution_backward_input(
      "miopen_convolution_backward_input",
      input_size, grad_output, weight,
      padding, stride, dilation, groups, benchmark, deterministic);
}

// Depthwise 卷积的反向数据计算函数
void raw_miopen_depthwise_convolution_backward_input_out(
    const at::Tensor& grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& weight,
    // 定义函数参数：输入张量的填充，步幅，扩展，分组，是否启用基准模式，是否确定性
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic) {

  // 获取输入张量的数据类型
  auto dataType = getMiopenDataType(grad_output);
  // 设置深度卷积模式
  miopenConvolutionMode_t c_mode = miopenDepthwise;

  // 创建包含输入、输出梯度和权重的卷积参数对象
  ConvolutionArgs args{ grad_input, grad_output, weight };
  // 获取 MIOpen 的处理句柄
  args.handle = getMiopenHandle();
  // 设置卷积参数
  setConvolutionParams(&args.params, args.handle, grad_input, weight, padding, stride, dilation, groups, deterministic);
  // 设置输入描述符
  args.idesc.set(grad_input);
  // 设置权重描述符和输出梯度的内存格式
  args.wdesc.set(weight, grad_output.suggest_memory_format(), 0);
  // 设置输出梯度描述符
  args.odesc.set(grad_output);
  // 设置卷积描述符，包括数据类型、卷积模式、维度、填充、步幅、扩展、分组和确定性
  args.cdesc.set(dataType, c_mode, grad_output.dim() - 2, args.params.padding, args.params.stride, args.params.dilation, args.params.groups, deterministic);

  // 如果启用基准模式
  if (benchmark) {
      // 选择用于反向数据卷积的算法并分配工作空间
      miopenConvBwdDataAlgorithm_t bwdDataAlg;
      Workspace workspace = chooseAlgorithm(args, benchmark, &bwdDataAlg);

      // 创建常量值对象，用于传递给 MIOpen 函数
      Constant one(dataType, 1);
      Constant zero(dataType, 0);

      // 执行反向数据卷积操作，更新输入梯度
      MIOPEN_CHECK(miopenConvolutionBackwardData(
          args.handle,
          &one, args.odesc.desc(), grad_output.const_data_ptr(),
          args.wdesc.desc(), weight.const_data_ptr(),
          args.cdesc.desc(), bwdDataAlg, &zero,
          args.idesc.desc(), grad_input.mutable_data_ptr(), workspace.data, workspace.size));
  }
  // 如果不启用基准模式
  else {
      // 选择最佳解决方案并分配工作空间
      uint64_t solution_id;
      Workspace workspace = chooseSolution<miopenConvBwdDataAlgorithm_t>(args, &solution_id);

      // 执行立即模式下的反向数据卷积操作，更新输入梯度
      MIOPEN_CHECK(miopenConvolutionBackwardDataImmediate(
          args.handle,
          args.odesc.desc(), grad_output.const_data_ptr(),
          args.wdesc.desc(), weight.const_data_ptr(),
          args.cdesc.desc(),
          args.idesc.desc(), grad_input.mutable_data_ptr(), workspace.data, workspace.size, solution_id));
  }
// MIOPEN 深度卷积反向输入的函数，计算输入的梯度
Tensor miopen_depthwise_convolution_backward_input(
    CheckedFrom c, // 检查点来源
    IntArrayRef input_size, // 输入尺寸数组的引用
    const TensorArg& grad_output, // 梯度输出张量参数
    const TensorArg& weight, // 权重张量参数
    IntArrayRef padding, // 填充数组的引用
    IntArrayRef stride, // 步幅数组的引用
    IntArrayRef dilation, // 膨胀数组的引用
    int64_t groups, // 分组数
    bool benchmark, // 是否使用基准模式
    bool deterministic) // 是否确定性模式
{
  checkAllSameType(c, {grad_output, weight}); // 检查 grad_output 和 weight 张量类型是否相同
  checkAllSameGPU(c, {grad_output, weight}); // 检查 grad_output 和 weight 是否在同一 GPU 上

  auto memory_format = at::MemoryFormat::Contiguous; // 内存格式，默认为连续
  if (miopen_conv_use_channels_last(*grad_output, *weight)) { // 如果使用通道最后的内存格式
    memory_format = (weight->ndimension() == 5) ? at::MemoryFormat::Contiguous : at::MemoryFormat::ChannelsLast;
    // 如果权重张量的维度为5，则使用 ChannelsLast3d，否则使用 ChannelsLast
  }

  // 创建一个空的 CUDA 张量 grad_input_t，根据内存格式选择存储方式
  Tensor grad_input_t = at::detail::empty_cuda(
      input_size, grad_output->options().memory_format(memory_format));

  TensorArg grad_input{ grad_input_t, "result", 0 }; // 定义 grad_input 张量参数
  // 检查卷积形状是否合法
  convolution_shape_check(c, grad_input, weight, grad_output, padding, stride, dilation, groups);

  // 修正权重张量为连续内存格式，并调整为当前内存格式
  Tensor weight_contig = weight->contiguous(memory_format);
  weight_contig.resize_(weight_contig.sizes(), memory_format);

  // 修正梯度输出张量为连续内存格式，并调整为当前内存格式
  Tensor grad_output_contig = grad_output->contiguous(memory_format);
  grad_output_contig.resize_(grad_output_contig.sizes(), memory_format);

  // 调用原始的 MIOPEN 深度卷积反向输入函数，计算结果写入 grad_input
  raw_miopen_depthwise_convolution_backward_input_out(
      *grad_input, grad_output_contig, weight_contig,
      padding, stride, dilation, groups, benchmark, deterministic);

  return *grad_input; // 返回计算得到的 grad_input 张量
}

// 包装函数，接受标准张量而不是 TensorArg，并调用 miopen_depthwise_convolution_backward_input
Tensor miopen_depthwise_convolution_backward_input(
    IntArrayRef input_size, // 输入尺寸数组的引用
    const Tensor& grad_output_t, // 梯度输出张量
    const Tensor& weight_t, // 权重张量
    IntArrayRef padding, // 填充数组的引用
    IntArrayRef stride, // 步幅数组的引用
    IntArrayRef dilation, // 膨胀数组的引用
    int64_t groups, // 分组数
    bool benchmark, // 是否使用基准模式
    bool deterministic) // 是否确定性模式
{
  TensorArg grad_output{ grad_output_t, "grad_output", 1 }, // 定义梯度输出张量参数
            weight{ weight_t, "weight", 2 }; // 定义权重张量参数
  // 调用 miopen_depthwise_convolution_backward_input 函数，传递标准张量参数
  return miopen_depthwise_convolution_backward_input(
      "miopen_depthwise_convolution_backward_input", // 函数名称字符串
      input_size, grad_output, weight, // 其他参数
      padding, stride, dilation, groups, benchmark, deterministic);
}

// MIOPEN 卷积反向函数，返回输入、权重和偏置的梯度
std::tuple<at::Tensor,at::Tensor,at::Tensor> miopen_convolution_backward(
    const at::Tensor& input, // 输入张量
    const at::Tensor& grad_output_t, // 梯度输出张量
    const at::Tensor& weight, // 权重张量
    IntArrayRef padding, // 填充数组的引用
    IntArrayRef stride, // 步幅数组的引用
    IntArrayRef dilation, // 膨胀数组的引用
    int64_t groups, // 分组数
    bool benchmark, // 是否使用基准模式
    bool deterministic, // 是否确定性模式
    std::array<bool,3> output_mask) // 输出掩码数组
{
  Tensor grad_output = grad_output_t.contiguous(input.suggest_memory_format()); // 获得连续内存格式的梯度输出张量

  Tensor grad_input, grad_weight, grad_bias; // 定义梯度输入、权重和偏置张量

  // 如果需要计算梯度输入
  if (output_mask[0]) {
    // 调用 miopen_convolution_backward_input 函数计算梯度输入
    grad_input = miopen_convolution_backward_input(input.sizes(), grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic);
  }

  // 如果需要计算梯度权重
  if (output_mask[1]) {
    // 调用 miopen_convolution_backward_weight 函数计算梯度权重
    grad_weight = miopen_convolution_backward_weight(weight.sizes(), grad_output, input, padding, stride, dilation, groups, benchmark, deterministic);
  }

  // 如果需要计算梯度偏置
  if (output_mask[2]) {
    // ...
    // 继续进行计算梯度偏置的操作，这部分未完整给出，需要根据实际情况补充完整
  // 调用 miopen_convolution_backward_bias 函数计算反向传播的偏置梯度
  grad_bias = miopen_convolution_backward_bias(grad_output);
}

// 返回三个张量的元组：grad_input, grad_weight, grad_bias
return std::tuple<Tensor,Tensor,Tensor>{grad_input, grad_weight, grad_bias};
}

// 定义 MIOpen 深度可分离卷积的反向传播函数，返回输入梯度、权重梯度和偏置梯度的元组
std::tuple<at::Tensor,at::Tensor,at::Tensor> miopen_depthwise_convolution_backward(
    const at::Tensor& input, const at::Tensor& grad_output_t, const at::Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic, std::array<bool,3> output_mask) {

  // 将梯度输出张量转为连续存储
  Tensor grad_output = grad_output_t.contiguous();

  // 定义梯度输入、梯度权重和梯度偏置张量
  Tensor grad_input, grad_weight, grad_bias;
  if (output_mask[0]) {
    // 如果需要输出输入梯度，则调用 miopen_depthwise_convolution_backward_input 函数
    grad_input = miopen_depthwise_convolution_backward_input(input.sizes(), grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic);
  }
  if (output_mask[1]) {
    // 如果需要输出权重梯度，则调用 miopen_depthwise_convolution_backward_weight 函数
    grad_weight = miopen_depthwise_convolution_backward_weight(weight.sizes(), grad_output, input, padding, stride, dilation, groups, benchmark, deterministic);
  }
  if (output_mask[2]) {
    // 如果需要输出偏置梯度，则调用 miopen_convolution_backward_bias 函数
    grad_bias = miopen_convolution_backward_bias(grad_output);
  }

  // 返回输入梯度、权重梯度和偏置梯度的元组
  return std::tuple<Tensor,Tensor,Tensor>{grad_input, grad_weight, grad_bias};
}

// 执行 MIOpen 转置卷积操作
Tensor miopen_convolution_transpose(
    const Tensor& input_t, const Tensor& weight_t, const std::optional<Tensor>& bias_t_opt,
    IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation,
    int64_t groups, bool benchmark, bool deterministic)
{
  // 处理可选的偏置张量
  c10::MaybeOwned<Tensor> bias_t_maybe_owned = at::borrow_from_optional_tensor(bias_t_opt);
  const Tensor& bias_t = *bias_t_maybe_owned;

  // 定义输入张量、权重张量和偏置张量的参数信息
  TensorArg input  { input_t,  "input",  1 },
            weight { weight_t, "weight", 2 },
            bias   { bias_t,   "bias",   3 };
  CheckedFrom c = "miopen_convolution_transpose";
  
  // 调用 miopen_convolution_transpose_forward 函数进行转置卷积前向传播
  auto output_t = miopen_convolution_transpose_forward(
    c, input, weight, padding, output_padding, stride, dilation, groups, benchmark, deterministic);
  
  // 如果存在偏置张量，则调用 miopen_convolution_add_bias_ 函数添加偏置
  if (bias->defined()) {
    miopen_convolution_add_bias_(c, { output_t, "result", 0 }, bias);
  }
  
  // 返回转置卷积的输出张量
  return output_t;
}

// 执行 MIOpen 融合卷积激活函数前向传播
void raw_miopen_convolution_relu_out(
    const Tensor& output,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    // 定义函数，执行融合卷积操作
    void executeFusionConvolution(const Tensor& input, Tensor& output, const Tensor& weight, const Tensor& bias,
                                  int padding, int stride, int dilation, int groups,
                                  bool deterministic) {
    
      // 获取输入数据的 MIOPEN 数据类型
      auto dataType = getMiopenDataType(input);
      // 设置 MIOPEN 卷积模式为标准卷积
      miopenConvolutionMode_t c_mode = miopenConvolution;
    
      // 创建 ConvolutionArgs 结构体，封装输入、输出和权重张量
      ConvolutionArgs args{ input, output, weight };
      // 获取 MIOPEN 句柄
      args.handle = getMiopenHandle();
      // 设置卷积操作的参数，包括输入、权重、填充、步幅、扩展、组数等
      setConvolutionParams(&args.params, args.handle, input, weight, padding, stride, dilation, groups, deterministic);
      // 设置输入描述符
      args.idesc.set(input);
      // 设置权重描述符
      args.wdesc.set(weight, input.suggest_memory_format(), 0);
      // 设置输出描述符
      args.odesc.set(output);
      // 设置卷积描述符，包括数据类型、卷积模式、维度等
      args.cdesc.set(dataType, c_mode, input.dim() - 2, args.params.padding, args.params.stride, args.params.dilation, args.params.groups, deterministic);
    
      // 创建 bias 的描述符
      TensorDescriptor bdesc;
      bdesc.set(bias.expand({1, bias.size(0)}), output.dim());
    
      // 创建融合计划
      miopenFusionPlanDescriptor_t fusePlanDesc;
      miopenFusionOpDescriptor_t convoOp;
      miopenFusionOpDescriptor_t biasOp;
      miopenFusionOpDescriptor_t activOp;
      // 创建融合计划描述符，并指定垂直融合方式和输入描述符
      MIOPEN_CHECK(miopenCreateFusionPlan(&fusePlanDesc, miopenVerticalFusion, args.idesc.desc()));
      // 添加卷积操作到融合计划
      MIOPEN_CHECK(miopenCreateOpConvForward(fusePlanDesc, &convoOp, args.cdesc.desc(), args.wdesc.desc()));
      // 添加偏置操作到融合计划
      MIOPEN_CHECK(miopenCreateOpBiasForward(fusePlanDesc, &biasOp, bdesc.desc()));
      // 添加激活函数操作到融合计划，使用 ReLU 激活函数
      MIOPEN_CHECK(miopenCreateOpActivationForward(fusePlanDesc, &activOp, miopenActivationRELU));
    
      // 编译融合计划
      MIOPEN_CHECK(miopenCompileFusionPlan(args.handle, fusePlanDesc));
    
      // 设置操作参数
      float alpha = static_cast<float>(1);
      float beta = static_cast<float>(0);
      float activ_alpha = static_cast<float>(0);
      float activ_beta = static_cast<float>(0);
      float activ_gamma = static_cast<float>(0);
      // 创建融合操作符参数
      miopenOperatorArgs_t fusionArgs;
      MIOPEN_CHECK(miopenCreateOperatorArgs(&fusionArgs));
      // 设置卷积操作的参数
      MIOPEN_CHECK(miopenSetOpArgsConvForward(fusionArgs, convoOp, &alpha, &beta, weight.const_data_ptr()));
      // 设置偏置操作的参数
      MIOPEN_CHECK(miopenSetOpArgsBiasForward(fusionArgs, biasOp, &alpha, &beta, bias.const_data_ptr()));
      // 设置激活函数操作的参数
      MIOPEN_CHECK(miopenSetOpArgsActivForward(fusionArgs, activOp, &alpha, &beta, activ_alpha, activ_beta, activ_gamma));
    
      // 执行融合计划
      miopenExecuteFusionPlan(args.handle, fusePlanDesc, args.idesc.desc(), input.const_data_ptr(), args.odesc.desc(), output.data_ptr(), fusionArgs);
    
      // 清理资源，销毁融合计划
      miopenDestroyFusionPlan(fusePlanDesc);
    }
}

// 根据当前张量和指定的内存格式返回张量本身或者一个新的张量
static at::Tensor self_or_new_memory_format(at::Tensor& self, at::MemoryFormat memory_format) {
  // 如果张量在指定的内存格式下是连续的，则直接返回该张量
  if (self.is_contiguous(memory_format)) {
    return self;
  }
  // 否则返回一个与输入张量相同形状的空张量，使用指定的内存格式
  return at::empty_like(self, self.options(), memory_format);
}

// 执行包含添加和ReLU的MIOpen卷积操作
Tensor miopen_convolution_add_relu(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& z,
    const std::optional<Scalar>& alpha,
    const std::optional<Tensor>& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups) {

  // MIOpen不支持融合添加操作，下面的cuDNN函数包含 alpha2 * z 这一步骤
  // y = act ( alpha1 * conv(x) + alpha2 * z + bias )

  // 推荐的内存格式由输入张量决定
  auto memory_format = input.suggest_memory_format();

  // 获取全局上下文
  auto& ctx = at::globalContext();
  // 获取CuDNN性能基准标志
  bool benchmark = ctx.benchmarkCuDNN();

  // 定义输入张量和权重张量的参数
  TensorArg input_arg  { input,  "input",  1 },
            weight_arg { weight, "weight", 2 };

  // 调用MIOpen卷积前向操作
  auto output = miopen_convolution_forward(
      "miopen_convolution_add_relu",
      input_arg,
      weight_arg,
      padding,
      stride,
      dilation,
      groups,
      benchmark,
      false // deterministic
  );

  // 将输出张量转换为指定的内存格式
  auto contig_output = self_or_new_memory_format(output, memory_format);

  // 如果输出张量与转换后的张量不同，复制数据到转换后的张量
  if (!output.is_same(contig_output)) {
    contig_output.copy_(output);
  }

  // 计算 alpha * z + bias，并添加到输出张量上
  auto _alpha = alpha.has_value() ? alpha.value().to<float>() : 1.0;
  auto _bias = bias.has_value()
          ? bias.value()
          : at::zeros(
                {contig_output.size(1)},
                optTypeMetaToScalarType(contig_output.options().dtype_opt()),
                contig_output.options().layout_opt(),
                contig_output.options().device_opt(),
                contig_output.options().pinned_memory_opt());

  // 执行 alpha * z + bias 操作，并添加到输出张量上
  at::Tensor alpha_mul_z_add_bias = at::native::reshape_bias(input.dim(), _bias).add(z, _alpha);
  contig_output.add_(alpha_mul_z_add_bias);
  // 对输出张量执行ReLU操作
  contig_output.relu_();

  // 返回处理后的输出张量
  return contig_output;
}

// 执行包含ReLU的MIOpen卷积操作
Tensor miopen_convolution_relu(
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups) {

  // 推荐的内存格式由输入张量决定
  auto memory_format = input.suggest_memory_format();

  // 获取全局上下文
  auto& ctx = at::globalContext();
  // 获取CuDNN性能基准标志
  bool benchmark = ctx.benchmarkCuDNN();

  // 当前MIOpen仅支持Contiguous内存格式、fp32数据类型和4维张量
  if (input.suggest_memory_format() == at::MemoryFormat::Contiguous
          && input.scalar_type() == at::kFloat
          && input.ndimension() == 4) {

    // 调用empty_cuda函数创建空的输出张量，使用推荐的内存格式
    Tensor output_t = at::detail::empty_cuda(
        conv_output_size(
            input.sizes(), weight.sizes(), padding, stride, dilation),
        input.options().memory_format(input.suggest_memory_format()));
    // 如果输出张量的元素数为0，直接返回空张量
    if (output_t.numel() == 0) {
      return output_t;
    }
    // 检查是否存在偏置项，如果存在则使用其值，否则创建一个与输出张量维度相同的零张量作为偏置
    auto _bias = bias.has_value()
            ? bias.value()
            : at::zeros(
                  {output_t.size(1)},  // 创建与输出张量第一维大小相同的零张量
                  optTypeMetaToScalarType(output_t.options().dtype_opt()),  // 获取输出张量数据类型
                  output_t.options().layout_opt(),  // 获取输出张量布局选项
                  output_t.options().device_opt(),  // 获取输出张量设备选项
                  output_t.options().pinned_memory_opt());  // 获取输出张量固定内存选项

    // 调用原生 miopen_convolution_relu_out 函数进行卷积操作并加上 ReLU 激活
    raw_miopen_convolution_relu_out(
        output_t,  // 输出张量
        input,     // 输入张量
        weight,    // 权重张量
        _bias,     // 偏置张量
        stride,    // 步长
        padding,   // 填充
        dilation,  // 膨胀
        groups,    // 分组
        benchmark, // 是否使用基准模式
        false      // 是否确定性计算
    );

    // 返回经过卷积和 ReLU 激活后的输出张量
    return output_t;
  }
  else {
    // 如果不支持 miopen_convolution_relu，退回到 miopen_convolution_forward

    // 定义输入张量和权重张量的参数
    TensorArg input_arg  { input,  "input",  1 },
              weight_arg { weight, "weight", 2 };

    // 调用 miopen_convolution_forward 执行卷积操作，包含 ReLU 激活
    auto output = miopen_convolution_forward(
        "miopen_convolution_relu",  // 使用 miopen_convolution_relu 算法
        input_arg,                  // 输入张量参数
        weight_arg,                 // 权重张量参数
        padding,                    // 填充
        stride,                     // 步长
        dilation,                   // 膨胀
        groups,                     // 分组
        benchmark,                  // 是否使用基准模式
        false                       // 是否确定性计算
    );

    // 根据指定的内存格式获取连续的输出张量
    auto contig_output = self_or_new_memory_format(output, memory_format);

    // 如果输出张量与连续化后的张量不同，将输出复制到连续化后的张量
    if (!output.is_same(contig_output)) {
      contig_output.copy_(output);
    }

    // 检查是否存在偏置项，如果存在则使用其值，否则创建一个与连续化输出张量第一维大小相同的零张量作为偏置
    auto _bias = bias.has_value()
            ? bias.value()
            : at::zeros(
                  {contig_output.size(1)},  // 创建与连续化输出张量第一维大小相同的零张量
                  optTypeMetaToScalarType(contig_output.options().dtype_opt()),  // 获取连续化输出张量数据类型
                  contig_output.options().layout_opt(),  // 获取连续化输出张量布局选项
                  contig_output.options().device_opt(),  // 获取连续化输出张量设备选项
                  contig_output.options().pinned_memory_opt());  // 获取连续化输出张量固定内存选项

    // 重塑偏置张量以匹配输入张量的维度，并添加到连续化输出张量上
    at::Tensor reshaped_bias = at::native::reshape_bias(input.dim(), _bias);
    contig_output.add_(reshaped_bias);  // 添加偏置
    contig_output.relu_();              // 对连续化输出张量应用 ReLU 激活

    // 返回经过卷积和 ReLU 激活后的连续化输出张量
    return contig_output;
  }
}

REGISTER_CUDA_DISPATCH(miopen_convolution_backward_stub, &miopen_convolution_backward);
// 注册 CUDA 分发函数，将 miopen_convolution_backward_stub 映射到 miopen_convolution_backward

REGISTER_CUDA_DISPATCH(miopen_convolution_transpose_backward_stub, &miopen_convolution_transpose_backward);
// 注册 CUDA 分发函数，将 miopen_convolution_transpose_backward_stub 映射到 miopen_convolution_transpose_backward

REGISTER_CUDA_DISPATCH(miopen_depthwise_convolution_backward_stub, &miopen_depthwise_convolution_backward);
// 注册 CUDA 分发函数，将 miopen_depthwise_convolution_backward_stub 映射到 miopen_depthwise_convolution_backward

}}  // namespace
// 结束匿名命名空间

#endif
// 结束条件编译指令
```