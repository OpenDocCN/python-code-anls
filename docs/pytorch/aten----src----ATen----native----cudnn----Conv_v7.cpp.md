# `.\pytorch\aten\src\ATen\native\cudnn\Conv_v7.cpp`

```py
// 定义 TORCH_ASSERT_ONLY_METHOD_OPERATORS 宏，用于仅包含方法操作符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 包含 AT_CUDNN_ENABLED 宏定义的 CUDAConfig.h 文件，用于检查 cuDNN 是否启用
#include <ATen/cuda/CUDAConfig.h>

#if AT_CUDNN_ENABLED()

// 包含 Tensor.h 头文件，定义了 Tensor 类和相关操作
#include <ATen/core/Tensor.h>

// 根据 AT_PER_OPERATOR_HEADERS 宏的定义，选择性地包含不同的 ATen 头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/zeros.h>
#endif

// 包含全局配置文件 Config.h
#include <ATen/Config.h>

// 包含 CUDA 异常处理的头文件
#include <ATen/cuda/Exceptions.h>

// 包含 cudnn 相关的共享函数和数据结构的头文件
#include <ATen/native/cudnn/ConvShared.h>

// 包含 CUDA 图形工具的头文件
#include <ATen/cuda/CUDAGraphsUtils.cuh>

// 包含一些常量定义的头文件
#include <limits>
#include <vector>

// 包含 cudnn 的数据类型定义
#include <ATen/cudnn/Types.h>

// 包含 cudnn 的实用工具函数
#include <ATen/cudnn/Utils.h>

// 包含参数哈希相关的实用函数
#include <ATen/native/utils/ParamsHash.h>

// 包含 TensorUtils.h 头文件，定义了一些与 Tensor 相关的实用函数和类
#include <ATen/TensorUtils.h>

// 包含 C10 库中关于整数范围的工具函数
#include <c10/util/irange.h>

// 包含标准整数类型的头文件
#include <stdint.h>

// 包含算法相关的头文件
#include <algorithm>

// 包含函数对象相关的头文件
#include <functional>

// 包含迭代器相关的头文件
#include <iterator>

// 包含内存管理相关的头文件
#include <memory>

// 包含互斥锁和相关的头文件
#include <mutex>

// 包含字符串流相关的头文件
#include <sstream>

// 包含无序映射相关的头文件
#include <unordered_map>

// 注意 [cudnnFind 和 cudnnGet 的行为]
// 默认情况下，在 ConvolutionDescriptor 中，我们执行以下操作：
//
//     AT_CUDNN_CHECK(cudnnSetConvolutionMathType(mut_desc(),
//     CUDNN_DEFAULT_MATH));
//     if(dataType == CUDNN_DATA_HALF)
//       AT_CUDNN_CHECK(cudnnSetConvolutionMathType(mut_desc(),
//       CUDNN_TENSOR_OP_MATH));
//
// 更新：AT_CUDNN_CHECK 已更新为 AT_CUDNN_CHECK_WITH_SHAPES，当抛出 cuDNN 异常时，
// 它会自动打印张量形状和卷积参数。
//
// 在调用 cudnnGet/cudnnFind 之前调用 cudnnSetConvolutionMathType，它通知
// cudnnGet/cudnnFind 要迭代/考虑张量核心和非张量核心算法。如果在调用 cudnnGet/cudnnFind
// 之前不调用 cudnnSetConvolutionMathType，则 cudnnGet/cudnnFind 可能不会选择张量核心
// 算法。
//
// 现在，cudnnGet/cudnnFind 运行后，使用所有初始知识得出最佳的算法+数学类型组合。然后，
// 用户有责任更新卷积描述符的数学类型，并使用最佳算法和更新后的描述符调用后续 cudnn 调用。
// 如果我们不更新描述符而只是运行最佳算法，那么在底层，cudnn 将使用较慢的内核运行，因为它看到
// 最快的算法组合与子优化的数学类型。

// 定义一个后缀操作符 ""_TiB，将其用作字节大小的 TiB 转换
constexpr size_t operator"" _TiB(unsigned long long n) {
  return size_t(n) * 1024 * 1024 * 1024 * 1024;
}

// 命名空间 at 下的 native 命名空间，用于包含与本地实现相关的函数和结构
namespace at {
namespace native {

// 方便起见，定义了一个结构 ConvolutionArgs，用于传递描述符和数据指针
struct ConvolutionArgs {
  cudnnHandle_t handle; // cuDNN 句柄
  ConvolutionParams params; // 卷积参数
  TensorDescriptor idesc, odesc; // 输入和输出张量描述符
  FilterDescriptor wdesc; // 滤波器描述符
  const Tensor &input, output, weight; // 输入、输出和权重张量
  ConvolutionDescriptor cdesc; // 卷积描述符

  // 构造函数，初始化 ConvolutionArgs 结构
  ConvolutionArgs(
      const Tensor& input,
      const Tensor& output,
      const Tensor& weight)
      : input(input), output(output), weight(weight) {}
};
// 重载流插入运算符，将 ConvolutionArgs 对象输出到输出流 out
std::ostream& operator<<(std::ostream& out, const ConvolutionArgs& args) {
  // 输出参数的可重现性表示，已经包含了换行符
  out << repro_from_args(args.params)
      // 输出参数本身，已经包含了换行符
      << args.params
      // 输出输入描述，已经包含了换行符
      << "input: " << args.idesc
      // 输出输出描述，已经包含了换行符
      << "output: " << args.odesc
      // 输出权重描述，已经包含了换行符
      << "weight: " << args.wdesc
      // 输出指针地址的标题换行符
      << "Pointer addresses: "
      // 输出输入数据指针地址及换行符
      << "\n"
      << "    input: " << args.input.const_data_ptr() << "\n"
      // 输出输出数据指针地址及换行符
      << "    output: " << args.output.const_data_ptr() << "\n"
      // 输出权重数据指针地址及换行符
      << "    weight: " << args.weight.const_data_ptr() << "\n";

  return out; // 返回输出流对象
}

// ---------------------------------------------------------------------
//
// Benchmarking
//
// ---------------------------------------------------------------------

// TODO: Use something less heavy duty than a big honking mutex
// 定义用于缓存性能数据的模板结构体 BenchmarkCache
template <typename T>
struct BenchmarkCache {
  std::mutex mutex; // 互斥量，保护缓存访问的线程安全
  // 使用无序映射存储卷积参数到性能结果的映射表
  std::unordered_map<
      ConvolutionParams,
      T,
      ParamsHash<ConvolutionParams>,
      ParamsEqual<ConvolutionParams>>
      map;

  // 查找给定参数的性能结果，使用互斥量保护
  bool find(const ConvolutionParams& params, T* results) {
    std::lock_guard<std::mutex> guard(mutex); // 自动加锁
    auto it = map.find(params); // 在映射表中查找参数
    if (it == map.end()) { // 如果未找到
      return false; // 返回失败
    }
    *results = it->second; // 否则返回结果
    return true;
  }

  // 插入给定参数和性能结果到映射表中，使用互斥量保护
  void insert(const ConvolutionParams& params, const T& results) {
    std::lock_guard<std::mutex> guard(mutex); // 自动加锁
    map[params] = results; // 插入映射
  }
};

// 全局实例化三个不同算法的 BenchmarkCache 对象，用于前向、反向数据和反向过滤卷积算法
BenchmarkCache<cudnnConvolutionFwdAlgoPerf_t> fwd_algos;
BenchmarkCache<cudnnConvolutionBwdDataAlgoPerf_t> bwd_data_algos;
BenchmarkCache<cudnnConvolutionBwdFilterAlgoPerf_t> bwd_filter_algos;

// TODO: Stop manually allocating CUDA memory; allocate an ATen byte
// tensor instead.
// 定义工作空间类，用于管理 CUDA 内存
struct Workspace {
  size_t size; // 空间大小
  void* data; // 数据指针

  // 构造函数，分配指定大小的 CUDA 内存
  Workspace(size_t size) : size(size), data(NULL) {
    // 当 cuDNN 返回的工作空间大小超过 2^63 时，手动引发内存不足错误
    TORCH_CHECK_WITH(
        OutOfMemoryError, size < 1_TiB, "Not enough memory for workspace!");
    data = c10::cuda::CUDACachingAllocator::raw_alloc(size); // 分配内存
  }

  Workspace(const Workspace&) = delete; // 禁用拷贝构造函数
  Workspace(Workspace&&) = default; // 默认移动构造函数
  Workspace& operator=(Workspace&&) = default; // 默认移动赋值运算符

  // 析构函数，释放 CUDA 内存
  ~Workspace() {
    if (data) { // 如果数据指针不为空
      c10::cuda::CUDACachingAllocator::raw_delete(data); // 删除内存
    }
  }
};

// 空模板结构，用于算法搜索
template <typename perf_t>
struct algorithm_search {};

// 获取前向卷积算法所需的工作空间大小
cudnnStatus_t getWorkspaceSize(
    const ConvolutionArgs& args,
    cudnnConvolutionFwdAlgo_t algo,
    size_t* sz) {
  return cudnnGetConvolutionForwardWorkspaceSize(
      args.handle,
      args.idesc.desc(),
      args.wdesc.desc(),
      args.cdesc.desc(),
      args.odesc.desc(),
      algo,
      sz);
}
// 获取反向数据卷积算法所需的工作空间大小
cudnnStatus_t getWorkspaceSize(
    const ConvolutionArgs& args,
    cudnnConvolutionBwdDataAlgo_t algo,
    # 调用 cuDNN 函数以获取卷积反向数据操作所需的工作空间大小
    return cudnnGetConvolutionBackwardDataWorkspaceSize(
        args.handle,    # cuDNN 上下文句柄
        args.wdesc.desc(),  # 权重描述符
        args.odesc.desc(),  # 输出梯度描述符
        args.cdesc.desc(),  # 卷积描述符
        args.idesc.desc(),  # 输入描述符
        algo,   # 使用的卷积算法
        sz)     # 返回的工作空间大小存储在这个指针变量中
}

cudnnStatus_t getWorkspaceSize(
    const ConvolutionArgs& args,  // 输入参数：卷积计算的参数对象
    cudnnConvolutionBwdFilterAlgo_t algo,  // 输入参数：反向卷积过滤器算法类型
    size_t* sz) {  // 输出参数：用于存储工作空间大小的指针
  return cudnnGetConvolutionBackwardFilterWorkspaceSize(
      args.handle,  // 使用给定参数对象中的句柄
      args.idesc.desc(),  // 输入描述符
      args.odesc.desc(),  // 输出描述符
      args.cdesc.desc(),  // 卷积描述符
      args.wdesc.desc(),  // 权重描述符
      algo,  // 指定的反向卷积过滤器算法
      sz);  // 返回计算得到的工作空间大小
}

template <typename algo_t>
size_t getMaxWorkspaceSize(
    const ConvolutionArgs& args,  // 输入参数：卷积计算的参数对象
    const algo_t* algo,  // 输入参数：算法类型
    int n_algo) {  // 输入参数：算法数量
  size_t max_ws_size = 0;  // 最大工作空间大小初始化为0
  size_t max_block_size = 0;  // 最大块大小初始化为0

  const auto device = c10::cuda::current_device();  // 获取当前CUDA设备
  // 对于本地分配器，获取最大未使用块的大小。
  // 对于cudaMallocAsync，请参阅c10/cuda/CUDAMallocAsync.cpp:cacheInfo获取详细信息。
  c10::cuda::CUDACachingAllocator::cacheInfo(device, &max_block_size);

  for (const auto i : c10::irange(n_algo)) {  // 遍历算法数量范围
    cudnnStatus_t err;  // cuDNN操作的状态
    size_t sz;  // 工作空间大小
    err = getWorkspaceSize(args, algo[i], &sz);  // 获取给定算法的工作空间大小
    if (CUDNN_STATUS_SUCCESS != err || sz == 0 || sz < max_ws_size ||
        sz > max_block_size)  // 如果操作失败或者工作空间大小为0或者超出允许范围，则继续下一个算法
      continue;
    max_ws_size = sz;  // 更新最大工作空间大小
  }
  return max_ws_size;  // 返回最大工作空间大小
}

template <typename perf_t>
std::vector<perf_t> getValidAlgorithms(
    perf_t* perfResults,  // 输入参数：性能结果数组
    const ConvolutionArgs& args,  // 输入参数：卷积计算的参数对象
    int n_algo) {  // 输入参数：算法数量
  std::vector<perf_t> result;  // 存储有效算法的结果向量
  result.reserve(n_algo);  // 预留空间以存储算法结果
  for (const auto i : c10::irange(n_algo)) {  // 遍历算法数量范围
    perf_t perf = perfResults[i];  // 获取当前算法的性能结果

    // TODO: Shouldn't all returned results be successful?
    // Double check documentation for cudnnFindConvolutionForwardAlgorithmEx
    // TODO：所有返回的结果都应该是成功的吗？查看cudnnFindConvolutionForwardAlgorithmEx的文档以确认。
    if (perf.status == CUDNN_STATUS_SUCCESS) {  // 如果性能结果为成功
      if (!args.params.deterministic ||  // 如果不是确定性计算或者
          perf.determinism == CUDNN_DETERMINISTIC) {  // 性能结果表明是确定性计算
        result.push_back(perf);  // 将符合条件的性能结果添加到结果向量中
      }
    }
  }
  TORCH_CHECK(
      result.size() > 0, "no valid convolution algorithms available in CuDNN");  // 检查是否存在有效的卷积算法，否则抛出异常
  return result;  // 返回有效的算法结果向量
}

template <>
struct algorithm_search<cudnnConvolutionFwdAlgoPerf_t> {  // 模板特化：cuDNN正向卷积算法性能结构体
  using perf_t = cudnnConvolutionFwdAlgoPerf_t;  // 性能结果类型定义
  using algo_t = cudnnConvolutionFwdAlgo_t;  // 算法类型定义

  static constexpr auto DEFAULT_ALGO =  // 默认算法为隐式预编译GEMM
      CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
  static BenchmarkCache<perf_t>& cache() {  // 返回性能结果的缓存
    return fwd_algos;
  }

  static std::vector<perf_t> findAlgorithms(
      const ConvolutionArgs& args,  // 输入参数：卷积计算的参数对象
      bool benchmark) {  // 输入参数：是否进行基准测试
    static const algo_t algos[] = {  // 静态数组定义支持的算法
        CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
        CUDNN_CONVOLUTION_FWD_ALGO_FFT,
        CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING,
        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
        CUDNN_CONVOLUTION_FWD_ALGO_DIRECT,
        CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,
        CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED,
    };
    static constexpr int num_algos = CUDNN_CONVOLUTION_FWD_ALGO_COUNT;  // 算法数量
    static_assert(
        sizeof(algos) / sizeof(algos[0]) == num_algos,
        "Missing cuDNN convolution forward algorithms");  // 静态断言确保算法数量正确定义

    int perf_count;  // 性能计数
    std::unique_ptr<perf_t[]> perf_results(new perf_t[num_algos]);  // 分配性能结果数组的内存空间
    // 如果不是基准测试模式，则执行以下代码块
    if (!benchmark) {
      // 调用 cudnnGetConvolutionForwardAlgorithm_v7 函数，获取卷积前向算法的性能信息
      AT_CUDNN_CHECK_WITH_SHAPES(
          cudnnGetConvolutionForwardAlgorithm_v7(
              args.handle,
              args.idesc.desc(),
              args.wdesc.desc(),
              args.cdesc.desc(),
              args.odesc.desc(),
              num_algos,
              &perf_count,
              perf_results.get()),
          args);
    } else {
      // 如果是基准测试模式，则执行以下代码块
      // 获取最大的工作空间大小
      size_t max_ws_size = getMaxWorkspaceSize(args, algos, num_algos);
      // 创建工作空间对象，大小为 max_ws_size
      Workspace ws(max_ws_size);
      // 检查是否在进行 cudnn 基准测试，并记录错误
      at::cuda::errorIfCapturingCudnnBenchmark("cudnnFind");
      // 调用 cudnnFindConvolutionForwardAlgorithmEx 函数，找到卷积前向算法的性能信息
      AT_CUDNN_CHECK_WITH_SHAPES(
          cudnnFindConvolutionForwardAlgorithmEx(
              args.handle,
              args.idesc.desc(),
              args.input.const_data_ptr(),
              args.wdesc.desc(),
              args.weight.const_data_ptr(),
              args.cdesc.desc(),
              args.odesc.desc(),
              args.output.data_ptr(),
              num_algos,
              &perf_count,
              perf_results.get(),
              ws.data,
              ws.size),
          args);

      // 清空 CUDA 缓存分配器中的缓存块，因为上述基准测试使用了大量内存，例如几个 GB
      c10::cuda::CUDACachingAllocator::emptyCache();
    }
    // 返回有效算法的性能结果
    return getValidAlgorithms<perf_t>(perf_results.get(), args, perf_count);
  }

  // 静态函数：获取卷积前向操作所需的工作空间大小
  static void getWorkspaceSize(
      const ConvolutionArgs& args,
      algo_t algo,
      size_t* workspaceSize) {
    // 调用 cudnnGetConvolutionForwardWorkspaceSize 函数，获取卷积前向操作的工作空间大小
    AT_CUDNN_CHECK_WITH_SHAPES(
        cudnnGetConvolutionForwardWorkspaceSize(
            args.handle,
            args.idesc.desc(),
            args.wdesc.desc(),
            args.cdesc.desc(),
            args.odesc.desc(),
            algo,
            workspaceSize),
        args);
  }
};

// 特化模板结构算法搜索<cudnnConvolutionBwdDataAlgoPerf_t>
template <>
struct algorithm_search<cudnnConvolutionBwdDataAlgoPerf_t> {
  // 使用perf_t定义cudnnConvolutionBwdDataAlgoPerf_t类型
  using perf_t = cudnnConvolutionBwdDataAlgoPerf_t;
  // 使用algo_t定义cudnnConvolutionBwdDataAlgo_t类型
  using algo_t = cudnnConvolutionBwdDataAlgo_t;

  // 默认算法设置为CUDNN_CONVOLUTION_BWD_DATA_ALGO_1
  static constexpr auto DEFAULT_ALGO = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;

  // 返回后向数据算法性能的缓存BenchmarkCache<perf_t>
  static BenchmarkCache<perf_t>& cache() {
    return bwd_data_algos;
  }

  // 查找满足条件的算法列表
  static std::vector<perf_t> findAlgorithms(
      const ConvolutionArgs& args,
      bool benchmark) {
    // 静态定义算法数组
    static const algo_t algos[] = {
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_1,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED};
    // 静态定义算法数量
    static constexpr int num_algos = CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT;
    // 检查算法数组长度与算法数量的一致性
    static_assert(
        sizeof(algos) / sizeof(algos[0]) == num_algos,
        "Missing cuDNN convolution backward data algorithms.");
    
    // 性能计数器
    int perf_count;
    // 使用std::unique_ptr分配性能结果内存
    std::unique_ptr<perf_t[]> perf_results(new perf_t[num_algos]);

    // 根据benchmark标志选择不同的算法查找方式
    if (!benchmark) {
      // 调用cuDNN函数获取后向数据算法性能
      AT_CUDNN_CHECK_WITH_SHAPES(
          cudnnGetConvolutionBackwardDataAlgorithm_v7(
              args.handle,
              args.wdesc.desc(),
              args.odesc.desc(),
              args.cdesc.desc(),
              args.idesc.desc(),
              num_algos,
              &perf_count,
              perf_results.get()),
          args);
    } else {
      // 获取最大工作空间大小
      size_t max_ws_size = getMaxWorkspaceSize(args, algos, num_algos);
      Workspace ws(max_ws_size);
      // 检查是否捕获cuDNN基准测试
      at::cuda::errorIfCapturingCudnnBenchmark("cudnnFind");
      // 调用cuDNN函数查找后向数据算法性能扩展版本
      AT_CUDNN_CHECK_WITH_SHAPES(
          cudnnFindConvolutionBackwardDataAlgorithmEx(
              args.handle,
              args.wdesc.desc(),
              args.weight.const_data_ptr(),
              args.odesc.desc(),
              args.output.const_data_ptr(),
              args.cdesc.desc(),
              args.idesc.desc(),
              args.input.data_ptr(),
              num_algos,
              &perf_count,
              perf_results.get(),
              ws.data,
              ws.size),
          args);

      // 清空缓存分配器中的缓存块，因为上述基准测试使用了大量内存，例如几GB
      c10::cuda::CUDACachingAllocator::emptyCache();
    }

    // 返回有效的算法性能结果
    return getValidAlgorithms<perf_t>(perf_results.get(), args, perf_count);
  }

  // 获取指定算法的工作空间大小
  static void getWorkspaceSize(
      const ConvolutionArgs& args,
      cudnnConvolutionBwdDataAlgo_t algo,
      size_t* workspaceSize) {
    // 调用cuDNN函数获取后向数据算法的工作空间大小
    AT_CUDNN_CHECK_WITH_SHAPES(
        cudnnGetConvolutionBackwardDataWorkspaceSize(
            args.handle,
            args.wdesc.desc(),
            args.odesc.desc(),
            args.cdesc.desc(),
            args.idesc.desc(),
            algo,
            workspaceSize),
        args);
  }
};

// 特化模板结构
template <>
// 定义模板结构体 `algorithm_search`，用于查找 cudnnConvolutionBwdFilterAlgoPerf_t 类型的算法
struct algorithm_search<cudnnConvolutionBwdFilterAlgoPerf_t> {
  // 定义类型别名 perf_t 为 cudnnConvolutionBwdFilterAlgoPerf_t
  using perf_t = cudnnConvolutionBwdFilterAlgoPerf_t;
  // 定义类型别名 algo_t 为 cudnnConvolutionBwdFilterAlgo_t
  using algo_t = cudnnConvolutionBwdFilterAlgo_t;

  // 默认算法常量，默认为 CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1
  static constexpr auto DEFAULT_ALGO = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;

  // 返回后端过滤算法的基准缓存 BenchmarkCache<perf_t> 的引用
  static BenchmarkCache<perf_t>& cache() {
    return bwd_filter_algos;
  }

  // 查找并返回满足条件的算法性能 perf_t 的向量
  static std::vector<perf_t> findAlgorithms(
      const ConvolutionArgs& args,
      bool benchmark) {
    // 静态定义的算法数组，列出了几种后端过滤算法
    static const algo_t algos[] = {
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1,
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT,
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3,
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED,
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING,
    };
    // 注意: 减 1 是因为 ALGO_WINOGRAD 没有实现
    static constexpr int num_algos =
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT - 1;
    // 静态断言，检查算法数组的大小是否等于 num_algos
    static_assert(
        sizeof(algos) / sizeof(algos[0]) == num_algos,
        "Missing cuDNN convolution backward filter algorithms.");
    
    // 分配存储性能结果的内存
    std::unique_ptr<perf_t[]> perf_results(new perf_t[num_algos]);
    int perf_count;
    
    // 如果不是基准测试，则执行算法查找
    if (!benchmark) {
      // 使用 cudnnGetConvolutionBackwardFilterAlgorithm_v7 获取算法性能
      AT_CUDNN_CHECK_WITH_SHAPES(
          cudnnGetConvolutionBackwardFilterAlgorithm_v7(
              args.handle,
              args.idesc.desc(),
              args.odesc.desc(),
              args.cdesc.desc(),
              args.wdesc.desc(),
              num_algos,
              &perf_count,
              perf_results.get()),
          args);
    } else {
      // 如果是基准测试，则计算最大工作空间大小并创建工作空间
      size_t max_ws_size = getMaxWorkspaceSize(args, algos, num_algos);
      Workspace ws(max_ws_size);
      // 在进行 cudnnFindConvolutionBackwardFilterAlgorithmEx 之前，检查是否捕获 cuDNN 基准
      at::cuda::errorIfCapturingCudnnBenchmark("cudnnFind");
      // 使用 cudnnFindConvolutionBackwardFilterAlgorithmEx 查找算法性能
      AT_CUDNN_CHECK_WITH_SHAPES(
          cudnnFindConvolutionBackwardFilterAlgorithmEx(
              args.handle,
              args.idesc.desc(),
              args.input.const_data_ptr(),
              args.odesc.desc(),
              args.output.const_data_ptr(),
              args.cdesc.desc(),
              args.wdesc.desc(),
              args.weight.data_ptr(),
              num_algos,
              &perf_count,
              perf_results.get(),
              ws.data,
              ws.size),
          args);

      // 清空 CUDA 缓存分配器中的缓存块，因为上述基准测试使用了大量内存，例如几 GB
      c10::cuda::CUDACachingAllocator::emptyCache();
    }
    // 返回有效算法性能的向量
    return getValidAlgorithms<perf_t>(perf_results.get(), args, perf_count);
  }

  // 获取指定算法的工作空间大小
  static void getWorkspaceSize(
      const ConvolutionArgs& args,
      algo_t algo,
      size_t* workspaceSize) {
    // 使用 cudnnGetConvolutionBackwardFilterWorkspaceSize 获取后端过滤算法的工作空间大小
    AT_CUDNN_CHECK_WITH_SHAPES(
        cudnnGetConvolutionBackwardFilterWorkspaceSize(
            args.handle,
            args.idesc.desc(),
            args.odesc.desc(),
            args.cdesc.desc(),
            args.wdesc.desc(),
            algo,
            workspaceSize),
        args);
  }
};
class AlgoIterator {
  using search = algorithm_search<perf_t>;  // 使用别名定义搜索算法
  const ConvolutionArgs& args;  // 引用常量参数对象
  bool benchmark;  // 布尔变量，用于指示是否进行基准测试

 public:
  AlgoIterator(const ConvolutionArgs& args, bool benchmark)
      : args(args), benchmark(benchmark) {}  // 构造函数，初始化参数和标志

  static std::vector<perf_t> onlyDefaultAlgorithm(const ConvolutionArgs& args) {
    std::vector<perf_t> perfResults(1);  // 创建性能结果向量，大小为1
    perfResults[0].algo = search::DEFAULT_ALGO;  // 设置默认算法
    if (args.params.dataType == CUDNN_DATA_HALF) {
      perfResults[0].mathType = CUDNN_TENSOR_OP_MATH;  // 如果数据类型为半精度，使用张量操作数数学类型
    } else {
      perfResults[0].mathType = CUDNN_DEFAULT_MATH;  // 默认使用默认数学类型
      if (args.params.dataType == CUDNN_DATA_FLOAT && !args.params.allow_tf32) {
        perfResults[0].mathType = CUDNN_FMA_MATH;  // 如果是单精度并且不允许TF32，则使用FMA数学类型
      }
    }
    search::getWorkspaceSize(
        args, perfResults[0].algo, &(perfResults[0].memory));  // 获取工作空间大小
    return perfResults;  // 返回性能结果向量
  }

  void try_all(std::function<void(const perf_t& perf)> f) {
    bool only_use_default = args.params.deterministic && !benchmark;  // 是否仅使用默认算法

    auto& cache = search::cache();  // 获取搜索结果缓存
    perf_t algoPerf;
    if (!only_use_default && cache.find(args.params, &algoPerf)) {  // 如果不仅使用默认算法且缓存中存在结果
      try {
        f(algoPerf);  // 调用函数处理性能结果
        return;
      } catch (c10::OutOfMemoryError& e) {
        cudaGetLastError(); // 清除CUDA错误
      }
    }

    auto perfResults = only_use_default
        ? onlyDefaultAlgorithm(args)  // 如果仅使用默认算法，则获取默认算法的性能结果
        : search::findAlgorithms(args, benchmark);  // 否则搜索所有算法

    for (auto& algoPerf : perfResults) {
      try {
        f(algoPerf);  // 调用函数处理性能结果
        cache.insert(args.params, algoPerf);  // 将结果插入缓存
        return;
      } catch (c10::OutOfMemoryError& e) {
        cudaGetLastError(); // 清除CUDA错误
      } catch (c10::CuDNNError& e) {
        cudaGetLastError(); // 清除CUDA错误
      }
    }
    TORCH_CHECK(
        false, "Unable to find a valid cuDNN algorithm to run convolution");  // 如果未找到有效的cuDNN算法，则抛出错误
  }
};

inline Tensor allocate_workspace(size_t size, const Tensor& other) {
  // 有时cuDNN返回的工作空间大小大于2^63，这可能导致64位索引错误而不是OOM错误。在这种情况下，我们手动报告OOM。
  TORCH_CHECK_WITH(
      OutOfMemoryError, size < 1_TiB, "Not enough memory for workspace!");  // 检查工作空间大小是否小于1 TiB

  return at::empty({static_cast<int64_t>(size)}, other.options().dtype(kByte));  // 创建空的Tensor作为工作空间
}

// NOTE [ raw_cudnn_convolution_forward_out ]
//
//    - raw_cudnn_convolution_forward_out (Tensor)
//      处理过大以至于无法使用32位索引的张量的函数。
//      它将张量分割并调度到 `raw_cudnn_convolution_forward_out_32bit`。

//    - raw_cudnn_convolution_forward_out_32bit (Tensor)
//      低级函数，调用CuDNN，并接受一个输出张量（因此是 _out）。

// ---------------------------------------------------------------------
//
// 分割为32位
//
// ---------------------------------------------------------------------

template <typename func_t>
static inline void split_batch_dim_to_32bit_out(
    const at::Tensor& output,   // 参考输出张量，作为函数的输出结果
    const at::Tensor& input,    // 输入张量，函数的输入数据
    const at::Tensor& weight,   // 权重张量，函数的卷积核参数
    IntArrayRef padding,        // 填充参数，卷积操作中使用的填充大小数组
    IntArrayRef stride,         // 步幅参数，卷积操作中使用的步幅大小数组
    IntArrayRef dilation,       // 膨胀参数，卷积操作中使用的膨胀大小数组
    int64_t groups,             // 分组参数，卷积操作中的分组卷积数
    bool benchmark,             // 是否启用基准测试模式的布尔值
    bool deterministic,         // 是否使用确定性算法的布尔值
    bool allow_tf32,            // 是否允许使用 TF32 模式的布尔值
    int64_t max_worksize,       // 最大工作大小参数，用于限制每个计算分片的最大尺寸
    func_t func_32bit) {        // 函数指针，指向处理 32 位计算的函数

  constexpr int64_t int_max = std::numeric_limits<int>::max();  // 声明整数最大值为 int 类型的最大值
  const int64_t ni = input.numel();   // 输入张量元素总数
  const int64_t no = output.numel();  // 输出张量元素总数

  // 假设张量的形状为 (N, C, D1, D2, ...)，如果 N * C * D1 * D2 * ... <= int_max，则不需要分割
  if (ni <= int_max && no <= int_max) {
    // 调用 32 位处理函数，直接处理整个输入和输出张量
    func_32bit(
        output,
        input,
        weight,
        padding,
        stride,
        dilation,
        groups,
        benchmark,
        deterministic,
        allow_tf32);
    return;
  }

  // 否则，如果 C * D1 * D2 * ... <= int_max，则只需在 N 维度上分割
  //
  // 这里使用简单的启发式方法确定每个分片的大小
  // 不会尽量占满 2^31 地址空间，因为这个数字非常大，很可能会导致内存溢出
  int64_t n = output.size(0);  // 输出张量的第一维大小，即 N 的值
  int64_t max_inner_size = std::max<int64_t>(ni, no) / n;  // 计算每个分片的最大大小
  int64_t split_size = std::max<int64_t>(max_worksize / max_inner_size, 1L);  // 计算分片的大小
  int64_t num_splits = (n + split_size - 1) / split_size;  // 计算总共需要多少个分片

  // 如果每个分片的大小乘以最大内部大小小于 int_max，则进行分割处理
  if (split_size * max_inner_size < int_max) {
    for (const auto i : c10::irange(num_splits)) {
      int64_t start = split_size * i;  // 计算当前分片的起始索引
      int64_t split_size_ = std::min<int64_t>(split_size, n - start);  // 计算当前分片的实际大小
      Tensor input_ = input.narrow(0, start, split_size_);  // 根据当前分片大小裁剪输入张量
      Tensor output_ = output.narrow(0, start, split_size_);  // 根据当前分片大小裁剪输出张量
      func_32bit(
          output_,
          input_,
          weight,
          padding,
          stride,
          dilation,
          groups,
          benchmark,
          deterministic,
          allow_tf32);
    }
    return;
  }

  // 如果程序执行到这里，说明即使分割 N 维度也不够，这时问题开始变得复杂
  // 针对卷积操作，需要考虑以下问题：
  // - 张量的内存布局是 NCHW 还是 NHWC？
  // - 如果卷积是从 NCHW 到 NC'H'W'，那么我们应该：
  //   - 只分割 NC 吗？
  //   - 只分割 N'C' 吗？
  //   - 同时分割两者吗？
  // - 如果卷积是 NHWC，则需要在 H 维度上分割，需要特别注意边界条件的处理。
  // - 如果决定进行这些分割，内存是否连续？是否需要复制内存？考虑到问题的复杂性，
  // 最好不要在这种情况下使用 cuDNN。
  TORCH_INTERNAL_ASSERT(false, "This case should not be dispatched to cuDNN.");
}
}

// 定义宏：根据数据类型确定精度的断言
#define ASSERT_CORRECT_PRECISION(math_type)                     \
  if (args.params.dataType == CUDNN_DATA_FLOAT) {               \  // 如果数据类型为单精度浮点型
    TORCH_INTERNAL_ASSERT(                                      \  // 内部断言，确保以下条件为真
        args.params.allow_tf32 || math_type == CUDNN_FMA_MATH); \  // 如果允许 TF32 或者数学类型为 FMA 数学
  }

// ---------------------------------------------------------------------
//
// Convolution forward / Transposed convolution backward
//
// ---------------------------------------------------------------------

// 函数：使用 32 位输出进行 CUDNN 卷积前向 / 转置卷积反向计算
void raw_cudnn_convolution_forward_out_32bit(
    const Tensor& output,        // 输出张量
    const Tensor& input,         // 输入张量
    const Tensor& weight,        // 权重张量
    IntArrayRef padding,         // 填充数组引用
    IntArrayRef stride,          // 步幅数组引用
    IntArrayRef dilation,        // 膨胀数组引用
    int64_t groups,              // 组数
    bool benchmark,              // 是否基准测试模式
    bool deterministic,          // 是否确定性模式


这样，你可以为给定的 C++ 代码段生成适当的注释，说明每行代码的作用和意图。
    bool allow_tf32) {
// 定义函数开始，参数列表包括一个布尔类型参数 allow_tf32

  auto dataType = getCudnnDataType(input);
  // 获取输入张量的数据类型

  ConvolutionArgs args{input, output, weight};
  // 创建 ConvolutionArgs 结构体实例，并初始化输入、输出和权重张量

  args.handle = getCudnnHandle();
  // 获取当前 cuDNN 句柄，用于后续 cuDNN 函数调用

  at::MemoryFormat memory_format =
      cudnn_conv_suggest_memory_format(input, weight);
  // 推荐合适的内存格式以供卷积操作使用

  setConvolutionParams(
      &args.params,
      input,
      weight,
      padding,
      stride,
      dilation,
      groups,
      deterministic,
      allow_tf32,
      memory_format);
  // 设置卷积参数，包括填充、步幅、膨胀率、分组、确定性和是否允许 tf32

  args.idesc.set(input, memory_format);
  // 根据输入张量和推荐的内存格式设置输入描述符

  args.wdesc.set(weight, memory_format, 0);
  // 根据权重张量、推荐的内存格式和偏移设置权重描述符

  args.odesc.set(output, memory_format);
  // 根据输出张量和推荐的内存格式设置输出描述符

  args.cdesc.set(
      dataType,
      input.dim() - 2,
      args.params.padding,
      args.params.stride,
      args.params.dilation,
      args.params.groups,
      args.params.allow_tf32);
  // 根据数据类型、输入维度、填充、步幅、膨胀率、分组和是否允许 tf32 设置卷积描述符

  // TODO: 当我们支持传统的分组卷积时，我们将为每个卷积重新初始化工作空间。
  // 这样做有些浪费；我们更希望重用工作空间。
  // 另一方面，传统的分组卷积支持已经相当慢了，所以这可能并不重要。
  // （这也适用于 raw_cudnn_convolution_backward_input。）
  // 处理注释，这里是对传统分组卷积支持的一些考虑。

  AlgoIterator<cudnnConvolutionFwdAlgoPerf_t>(args, benchmark)
      .try_all([&](const cudnnConvolutionFwdAlgoPerf_t& fwdAlgPerf) {
        // 使用 AlgoIterator 尝试所有前向卷积算法，传入的参数是 args 和 benchmark

        Tensor workspace = allocate_workspace(fwdAlgPerf.memory, input);
        // 分配工作空间，大小为 fwdAlgPerf.memory

        // 更新 convDesc mathType，因为 cudnn 7.4+ 现在需要算法和 mathType
        // 以确定是否使用张量核心内核，参见 Note [behavior of cudnnFind and cudnnGet]
        ASSERT_CORRECT_PRECISION(fwdAlgPerf.mathType);
        // 确保精度正确

        AT_CUDNN_CHECK_WITH_SHAPES(
            cudnnSetConvolutionMathType(
                args.cdesc.mut_desc(), fwdAlgPerf.mathType),
            args);
        // 设置卷积的数学类型

        Constant one(dataType, 1);
        Constant zero(dataType, 0);
        // 创建常量张量 one 和 zero，数据类型为 dataType，值分别为 1 和 0

        AT_CUDNN_CHECK_WITH_SHAPES(
            cudnnConvolutionForward(
                args.handle,
                &one,
                args.idesc.desc(),
                input.const_data_ptr(),
                args.wdesc.desc(),
                weight.const_data_ptr(),
                args.cdesc.desc(),
                fwdAlgPerf.algo,
                workspace.data_ptr(),
                fwdAlgPerf.memory,
                &zero,
                args.odesc.desc(),
                output.data_ptr()),
            args,
            "Forward algorithm: ",
            static_cast<int>(fwdAlgPerf.algo),
            "\n");
        // 调用 cuDNN 的前向卷积函数 cudnnConvolutionForward 进行卷积计算
      });
  // 结束 AlgoIterator 的循环
}

void raw_cudnn_convolution_forward_out_v7(
    const Tensor& output,                              // 输出张量，存储前向卷积的结果
    const Tensor& input,                               // 输入张量，作为卷积的输入
    const Tensor& weight,                              // 权重张量，卷积核的权重
    IntArrayRef padding,                               // 填充数组，指定卷积的填充大小
    IntArrayRef stride,                                // 步幅数组，指定卷积操作的步幅
    IntArrayRef dilation,                              // 膨胀数组，指定卷积核的膨胀率
    int64_t groups,                                    // 分组数，用于分组卷积
    bool benchmark,                                     // 布尔值，指示是否使用基准模式
    bool deterministic,                                // 布尔值，指示是否使用确定性算法
    bool allow_tf32) {                                 // 布尔值，指示是否允许使用 TF32 混合精度
  split_batch_dim_to_32bit_out(
      output,                                          // 输出张量，分割批次维度到 32 位输出
      input,                                           // 输入张量，卷积操作的输入
      weight,                                          // 权重张量，卷积核的权重
      padding,                                         // 填充数组，指定卷积的填充大小
      stride,                                          // 步幅数组，指定卷积操作的步幅
      dilation,                                        // 膨胀数组，指定卷积核的膨胀率
      groups,                                          // 分组数，用于分组卷积
      benchmark,                                       // 布尔值，指示是否使用基准模式
      deterministic,                                   // 布尔值，指示是否使用确定性算法
      allow_tf32,                                      // 布尔值，指示是否允许使用 TF32 混合精度
      1024 * 1024 * 256,                               // 整数，指定最大输入大小的限制
      raw_cudnn_convolution_forward_out_32bit);        // 回调函数，执行 32 位输出的原始 CUDNN 前向卷积
}

// ---------------------------------------------------------------------
//
// Convolution backward / Transposed convolution forward
//
// ---------------------------------------------------------------------

void raw_cudnn_convolution_backward_input_out_32bit(
    const at::Tensor& grad_input,                      // 梯度输入张量，反向传播时的输入梯度
    const at::Tensor& grad_output,                     // 梯度输出张量，反向传播时的输出梯度
    const at::Tensor& weight,                          // 权重张量，卷积核的权重
    IntArrayRef padding,                               // 填充数组，指定卷积的填充大小
    IntArrayRef stride,                                // 步幅数组，指定卷积操作的步幅
    IntArrayRef dilation,                              // 膨胀数组，指定卷积核的膨胀率
    int64_t groups,                                    // 分组数，用于分组卷积
    bool benchmark,                                    // 布尔值，指示是否使用基准模式
    bool deterministic,                                // 布尔值，指示是否使用确定性算法


注释：以上是对C++代码中函数和参数的逐行注释，解释了每个参数的用途和功能。
    bool allow_tf32) {
  // 获取梯度输出的 cuDNN 数据类型
  auto dataType = getCudnnDataType(grad_output);

  // 构造卷积参数对象 args，并设置相关属性
  ConvolutionArgs args{grad_input, grad_output, weight};
  // 获取 cuDNN 句柄
  args.handle = getCudnnHandle();
  // 推荐内存格式给 cudnn_conv_suggest_memory_format 函数，并返回推荐的内存格式
  at::MemoryFormat memory_format =
      cudnn_conv_suggest_memory_format(grad_input, weight);
  // 设置卷积参数
  setConvolutionParams(
      &args.params,
      grad_input,
      weight,
      padding,
      stride,
      dilation,
      groups,
      deterministic,
      allow_tf32,
      memory_format);

  // 设置输入描述符的形状和内存格式
  args.idesc.set(grad_input, memory_format);
  // 设置权重描述符的形状和内存格式，第三个参数为零
  args.wdesc.set(weight, memory_format, 0);
  // 设置输出梯度描述符的形状和内存格式
  args.odesc.set(grad_output, memory_format);
  // 设置卷积描述符的相关属性
  args.cdesc.set(
      dataType,
      grad_output.dim() - 2,
      args.params.padding,
      args.params.stride,
      args.params.dilation,
      args.params.groups,
      args.params.allow_tf32);

  // 创建算法迭代器 AlgoIterator，用于尝试所有的反向数据卷积算法
  AlgoIterator<cudnnConvolutionBwdDataAlgoPerf_t>(args, benchmark)
      .try_all([&](const cudnnConvolutionBwdDataAlgoPerf_t& bwdDataAlgPerf) {
        // 分配工作空间，用于当前算法 bwdDataAlgPerf
        Tensor workspace =
            allocate_workspace(bwdDataAlgPerf.memory, grad_output);

        // 更新卷积描述符的数学类型，自 cuDNN 7.4+ 开始需要同时指定算法和数学类型来确定是否使用张量核心
        ASSERT_CORRECT_PRECISION(bwdDataAlgPerf.mathType);
        // 设置卷积数学类型
        AT_CUDNN_CHECK_WITH_SHAPES(
            cudnnSetConvolutionMathType(
                args.cdesc.mut_desc(), bwdDataAlgPerf.mathType),
            args);

        // 创建常量张量 one 和 zero，类型为 dataType
        Constant one(dataType, 1);
        Constant zero(dataType, 0);

        // 执行反向数据卷积操作
        AT_CUDNN_CHECK_WITH_SHAPES(
            cudnnConvolutionBackwardData(
                args.handle,
                &one,
                args.wdesc.desc(),
                weight.const_data_ptr(),
                args.odesc.desc(),
                grad_output.const_data_ptr(),
                args.cdesc.desc(),
                bwdDataAlgPerf.algo,
                workspace.data_ptr(),
                bwdDataAlgPerf.memory,
                &zero,
                args.idesc.desc(),
                grad_input.mutable_data_ptr()),
            args,
            "Additional pointer addresses: \n",
            "    grad_output: ",
            grad_output.const_data_ptr(),
            "\n",
            "    grad_input: ",
            grad_input.mutable_data_ptr(),
            "\n",
            "Backward data algorithm: ",
            static_cast<int>(bwdDataAlgPerf.algo),
            "\n");
      });
}

void raw_cudnn_convolution_backward_input_out_v7(
    const at::Tensor& grad_input,             // 梯度输入张量
    const at::Tensor& grad_output,            // 梯度输出张量
    const at::Tensor& weight,                 // 权重张量
    IntArrayRef padding,                      // 填充数组
    IntArrayRef stride,                       // 步幅数组
    IntArrayRef dilation,                     // 膨胀数组
    int64_t groups,                           // 分组数
    bool benchmark,                           // 是否基准测试
    bool deterministic,                       // 是否确定性操作
    bool allow_tf32) {                        // 是否允许使用 TF32 格式

  split_batch_dim_to_32bit_out(              // 调用函数，将批次维度划分到 32 位输出
      grad_input,                           // 梯度输入张量
      grad_output,                          // 梯度输出张量
      weight,                               // 权重张量
      padding,                              // 填充数组
      stride,                               // 步幅数组
      dilation,                             // 膨胀数组
      groups,                               // 分组数
      benchmark,                            // 是否基准测试
      deterministic,                        // 是否确定性操作
      allow_tf32,                           // 是否允许使用 TF32 格式
      1024 * 1024 * 128,                    // 批次维度的大小，以字节为单位
      raw_cudnn_convolution_backward_input_out_32bit);  // 使用的函数指针

}

// ---------------------------------------------------------------------
//
// Convolution backward (weight)
//
// ---------------------------------------------------------------------

void raw_cudnn_convolution_backward_weight_out_32bit(
    const Tensor& grad_weight,               // 梯度权重张量
    const Tensor& grad_output,               // 梯度输出张量
    const Tensor& input,                     // 输入张量
    IntArrayRef padding,                     // 填充数组
    IntArrayRef stride,                      // 步幅数组
    IntArrayRef dilation,                    // 膨胀数组
    int64_t groups,                          // 分组数
    bool benchmark,                          // 是否基准测试
    bool deterministic,                      // 是否确定性操作


这段代码包含了两个函数的声明和定义，分别用于反向卷积操作中的输入和权重的计算。注释解释了每个函数的参数和函数调用的含义。
    bool allow_tf32) {
```  
# 参数列表，这是一个函数定义，接受多个参数，其中包括一个布尔类型参数 `allow_tf32`。

  auto dataType = getCudnnDataType(input);
```py  
# 调用函数 `getCudnnDataType` 来获取输入张量的数据类型，并将结果保存在 `dataType` 变量中。

  ConvolutionArgs args{input, grad_output, grad_weight};
  args.handle = getCudnnHandle();
```  
# 创建 `ConvolutionArgs` 对象 `args`，并初始化其成员变量，包括输入张量 `input`、梯度输出 `grad_output` 和梯度权重 `grad_weight`。然后设置 `args` 的 `handle` 属性为 CUDNN 的句柄。

  at::MemoryFormat memory_format =
      cudnn_conv_suggest_memory_format(input, grad_weight);
```py  
# 调用 `cudnn_conv_suggest_memory_format` 函数，根据输入张量和梯度权重推荐内存格式，并将推荐的格式保存在 `memory_format` 变量中。

  setConvolutionParams(
      &args.params,
      input,
      grad_weight,
      padding,
      stride,
      dilation,
      groups,
      deterministic,
      allow_tf32,
      memory_format);
```  
# 调用 `setConvolutionParams` 函数，设置卷积参数 `args.params`，包括输入张量 `input`、梯度权重 `grad_weight`、填充 `padding`、步长 `stride`、扩展 `dilation`、分组 `groups`、确定性标志 `deterministic`、TF32 允许标志 `allow_tf32` 和推荐的内存格式 `memory_format`。

  args.idesc.set(input, memory_format);
```py  
# 调用 `args.idesc.set` 方法，设置输入描述符 `args.idesc`，使用输入张量 `input` 和内存格式 `memory_format`。

  args.wdesc.set(grad_weight, memory_format, 0);
```  
# 调用 `args.wdesc.set` 方法，设置权重描述符 `args.wdesc`，使用梯度权重 `grad_weight`、内存格式 `memory_format` 和附加参数 `0`。

  args.odesc.set(grad_output, memory_format);
```py  
# 调用 `args.odesc.set` 方法，设置输出梯度描述符 `args.odesc`，使用梯度输出 `grad_output` 和内存格式 `memory_format`。

  args.cdesc.set(
      dataType,
      input.dim() - 2,
      args.params.padding,
      args.params.stride,
      args.params.dilation,
      args.params.groups,
      args.params.allow_tf32);
```  
# 调用 `args.cdesc.set` 方法，设置卷积描述符 `args.cdesc`，使用数据类型 `dataType`、输入张量的维度减去2、参数对象中的填充、步长、扩展、分组和 TF32 允许标志。

  AlgoIterator<cudnnConvolutionBwdFilterAlgoPerf_t>(args, benchmark)
      .try_all(
          [&](const cudnnConvolutionBwdFilterAlgoPerf_t& bwdFilterAlgPerf) {
```py  
# 创建 `AlgoIterator` 对象，迭代尝试所有的反向卷积滤波算法性能对象，每个对象保存在 `bwdFilterAlgPerf` 中。

            Tensor workspace =
                allocate_workspace(bwdFilterAlgPerf.memory, input);
```  
# 调用 `allocate_workspace` 函数，为当前反向卷积滤波算法分配工作空间 `workspace`，内存大小由 `bwdFilterAlgPerf.memory` 决定。

            // update convDesc mathType since cudnn 7.4+ now requires both algo
            // + mathType to figure out whether to use Tensor core kernels or
            // not See Note [behavior of cudnnFind and cudnnGet]
            ASSERT_CORRECT_PRECISION(bwdFilterAlgPerf.mathType);
```py  
# 断言当前反向卷积滤波算法的数学类型 `bwdFilterAlgPerf.mathType` 的正确性。

            AT_CUDNN_CHECK_WITH_SHAPES(
                cudnnSetConvolutionMathType(
                    args.cdesc.mut_desc(), bwdFilterAlgPerf.mathType),
                args);
```  
# 调用 `cudnnSetConvolutionMathType` 函数，设置卷积数学类型，参数为 `args.cdesc.mut_desc()` 和 `bwdFilterAlgPerf.mathType`。

            Constant one(dataType, 1);
            Constant zero(dataType, 0);
```py  
# 创建常量对象 `one` 和 `zero`，它们的数据类型为 `dataType`，分别赋值为 `1` 和 `0`。

            AT_CUDNN_CHECK_WITH_SHAPES(
                cudnnConvolutionBackwardFilter(
                    args.handle,
                    &one,
                    args.idesc.desc(),
                    input.const_data_ptr(),
                    args.odesc.desc(),
                    grad_output.const_data_ptr(),
                    args.cdesc.desc(),
                    bwdFilterAlgPerf.algo,
                    workspace.data_ptr(),
                    bwdFilterAlgPerf.memory,
                    &zero,
                    args.wdesc.desc(),
                    grad_weight.data_ptr()),
                args,
                "Additional pointer addresses: \n",
                "    grad_output: ",
                grad_output.const_data_ptr(),
                "\n",
                "    grad_weight: ",
                grad_weight.data_ptr(),
                "\n",
                "Backward filter algorithm: ",
                static_cast<int>(bwdFilterAlgPerf.algo),
                "\n");
          });
```  
# 使用 `AlgoIterator` 对象的 `try_all` 方法，尝试所有的反向卷积滤波算法性能对象，并对每个对象执行以下操作：调用 `cudnnConvolutionBackwardFilter` 函数执行反向卷积滤波操作，使用设置的参数、工作空间和常量 `one` 和 `zero`。
}

void raw_cudnn_convolution_backward_weight_out_v7(
    const Tensor& grad_weight,  // 梯度权重张量
    const Tensor& grad_output,  // 梯度输出张量
    const Tensor& input,        // 输入张量
    IntArrayRef padding,         // 填充数组
    IntArrayRef stride,          // 步长数组
    IntArrayRef dilation,        // 膨胀数组
    int64_t groups,              // 组数
    bool benchmark,              // 是否使用基准模式
    bool deterministic,          // 是否确定性操作
    bool allow_tf32) {           // 是否允许 TF32 操作
  constexpr int64_t int_max = std::numeric_limits<int>::max();
  const int64_t ni = input.numel();        // 输入张量的元素个数
  const int64_t no = grad_output.numel();  // 梯度输出张量的元素个数
  // 假设张量的形状为 (N, C, D1, D2, ...)
  // 如果 N * C * D1 * D2 * ... <= int_max，则无需分割
  if (ni <= int_max && no <= int_max) {
    raw_cudnn_convolution_backward_weight_out_32bit(
        grad_weight,
        grad_output,
        input,
        padding,
        stride,
        dilation,
        groups,
        benchmark,
        deterministic,
        allow_tf32);
    return;
  }
  // 否则，如果 C * D1 * D2 * ... <= int_max，则只需在 N 维度上分割
  //
  // 这里使用简单的启发式方法确定每个分割的大小
  // 我们不会占用 2^31 的地址空间，因为这个数值非常大，很可能导致内存不足。
  int64_t n = grad_output.size(0);  // 获取张量在第 0 维度上的大小 N
  int64_t max_inner_size = std::max<int64_t>(ni, no) / n;  // 计算每个分割的最大内部大小
  int64_t split_size =
      std::max<int64_t>(1024 * 1024 * 512 / max_inner_size, 1L);  // 计算分割大小
  int64_t num_splits = (n + split_size - 1) / split_size;  // 计算分割的数量
  if (split_size * max_inner_size < int_max) {
    const auto kAccType = (grad_weight.scalar_type() == kHalf ||
                           grad_weight.scalar_type() == kBFloat16)
        ? kFloat
        : grad_weight.scalar_type();  // 根据梯度权重的数据类型确定累加器的数据类型
    Tensor grad_weight_accumulator =
        at::zeros(grad_weight.sizes(), grad_weight.options().dtype(kAccType));  // 创建累加器张量
    for (const auto i : c10::irange(num_splits)) {
      int64_t start = split_size * i;  // 计算当前分割的起始索引
      int64_t split_size_ = std::min<int64_t>(split_size, n - start);  // 计算当前分割的大小
      Tensor input_ = input.narrow(0, start, split_size_);  // 根据分割切片输入张量
      Tensor grad_output_ = grad_output.narrow(0, start, split_size_);  // 根据分割切片梯度输出张量
      Tensor grad_weight_ = at::empty_like(grad_weight);  // 创建当前分割的梯度权重张量
      raw_cudnn_convolution_backward_weight_out_32bit(
          grad_weight_,
          grad_output_,
          input_,
          padding,
          stride,
          dilation,
          groups,
          benchmark,
          deterministic,
          allow_tf32);  // 调用 32 位版本的反向权重卷积函数
      grad_weight_accumulator.add_(grad_weight_);  // 将当前分割的梯度权重累加到累加器中
    }
    grad_weight.copy_(grad_weight_accumulator);  // 将累加器中的梯度权重复制到原始梯度权重张量中
    return;
  }
  // 如果控制流达到这里，这意味着即使将 N 进行均分也不够，
  // 然后情况开始变得复杂：例如，对于 conv2d，需要考虑以下问题。
  // - 内存布局是 NCHW 还是 NHWC？
  // - 如果 conv 是 NCHW -> NC'H'W'，那么我们应该
  //   - 只分割 NC 吗？
  //   - 只分割 N'C' 吗？
  //   - 全部分割？
  // - 如果 conv 是 NHWC，那么我们需要跨 H 进行分割，需要非常小心处理边界条件，
  //   确保边界被正确处理。
  // - 如果决定进行这些分割，内存是否是连续的？是否需要复制内存？考虑到这个问题的复杂性，
  //   最好不要在这种情况下使用 cuDNN。
  TORCH_INTERNAL_ASSERT(false, "This case should not be dispatched to cuDNN.");
void raw_cudnn_convolution_add_relu_out_v7(
    const Tensor& output,                  // 输出张量
    const Tensor& input,                   // 输入张量
    const Tensor& weight,                  // 权重张量
    const Tensor& z,                       // z 张量
    float alpha,                           // 缩放因子 alpha
    const Tensor& bias,                    // 偏置张量
    IntArrayRef stride,                    // 步长数组
    IntArrayRef padding,                   // 填充数组
    IntArrayRef dilation,                  // 扩展数组
    int64_t groups,                        // 分组数
    bool benchmark,                        // 是否基准测试
    bool deterministic,                    // 是否确定性操作
    bool allow_tf32) {                     // 是否允许 TF32
  auto dataType = getCudnnDataType(input); // 获取 CUDNN 数据类型
  ConvolutionArgs args{input, output, weight}; // 创建卷积参数对象
  args.handle = getCudnnHandle();         // 获取 CUDNN 句柄
  at::MemoryFormat memory_format =        // 推荐的内存格式
      cudnn_conv_suggest_memory_format(input, weight);
  setConvolutionParams(                   // 设置卷积参数
      &args.params,
      input,
      weight,
      padding,
      stride,
      dilation,
      groups,
      deterministic,
      allow_tf32,
      memory_format);
  args.idesc.set(input, memory_format);   // 设置输入描述符
  args.wdesc.set(weight, memory_format, 0); // 设置权重描述符
  args.odesc.set(output, memory_format);  // 设置输出描述符
  args.cdesc.set(                         // 设置卷积描述符
      dataType,
      input.dim() - 2,
      args.params.padding,
      args.params.stride,
      args.params.dilation,
      args.params.groups,
      args.params.allow_tf32);

  TensorDescriptor zdesc;                 // 创建 z 张量描述符
  zdesc.set(z, memory_format);

  TensorDescriptor bdesc;                 // 创建偏置张量描述符
  bdesc.set(bias.expand({1, bias.size(0)}), memory_format, output.dim());

  ActivationDescriptor adesc;             // 创建激活函数描述符
  adesc.set(CUDNN_ACTIVATION_RELU);

  AlgoIterator<cudnnConvolutionFwdAlgoPerf_t>(args, benchmark)
      .try_all([&](const cudnnConvolutionFwdAlgoPerf_t& fwdAlgPerf) {
        Tensor workspace = allocate_workspace(fwdAlgPerf.memory, input);

        // 更新卷积描述符的数学类型，自 cudnn 7.4+ 开始需要算法和数学类型一起确定是否使用张量核心
        ASSERT_CORRECT_PRECISION(fwdAlgPerf.mathType);
        AT_CUDNN_CHECK_WITH_SHAPES(
            cudnnSetConvolutionMathType(
                args.cdesc.mut_desc(), fwdAlgPerf.mathType),
            args);

        Constant one(dataType, 1);         // 常数值为 1
        Constant alpha_(dataType, alpha);  // 缩放因子 alpha

        AT_CUDNN_CHECK_WITH_SHAPES(
            cudnnConvolutionBiasActivationForward(
                args.handle,
                &one,
                args.idesc.desc(),
                input.const_data_ptr(),
                args.wdesc.desc(),
                weight.const_data_ptr(),
                args.cdesc.desc(),
                fwdAlgPerf.algo,
                workspace.data_ptr(),
                fwdAlgPerf.memory,
                &alpha_,
                zdesc.desc(),
                z.const_data_ptr(),
                bdesc.desc(),
                bias.const_data_ptr(),
                adesc.desc(),
                args.odesc.desc(),
                output.data_ptr()),
            args,
            "zdesc: ",                   // z 描述符
            zdesc,
            "bdesc: ",                   // 偏置描述符
            bdesc,
            "cudnnConvolutionBiasActivationForward: ", // CUDNN 卷积偏置激活前向操作
            static_cast<int>(fwdAlgPerf.algo),       // 使用的算法
            "\n");
      });
}
    // cuDNN Conv-Bias-Activation:
    // y = act ( alpha1 * conv(x) + alpha2 * z + bias )
    // 在 PyTorch 函数 `raw_cudnn_convolution_add_relu_out` 中，alpha1 是 1，
    // alpha2 是参数 `float alpha`
    
    // 调用 cuDNN 库执行卷积操作，将结果写入 output 张量
    raw_cudnn_convolution_forward_out(
        output,
        input,
        weight,
        padding,
        stride,
        dilation,
        groups,
        benchmark,
        deterministic,
        allow_tf32);
    
    // 根据输入张量的维度重新调整偏置张量的形状，并对 z 乘以 alpha 后加上偏置
    at::Tensor alpha_mul_z_add_bias =
        at::native::reshape_bias(input.dim(), bias).add(z, alpha);
    
    // 将 alpha_mul_z_add_bias 张量加到输出张量上
    output.add_(alpha_mul_z_add_bias);
    
    // 对输出张量应用 ReLU 激活函数
    output.relu_();
}

} // namespace native
} // namespace at

#endif
```