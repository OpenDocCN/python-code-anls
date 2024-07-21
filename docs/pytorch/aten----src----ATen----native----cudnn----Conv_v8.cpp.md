# `.\pytorch\aten\src\ATen\native\cudnn\Conv_v8.cpp`

```py
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 定义宏，仅包含操作符的断言

#include <ATen/cuda/CUDAConfig.h> // 包含 AT_CUDNN_ENABLED 的定义

#if AT_CUDNN_ENABLED()
// 如果启用了 CuDNN

#include <ATen/cudnn/cudnn-wrapper.h>
// 包含 CuDNN 的头文件

#include <c10/macros/Macros.h>
// 包含 C10 的宏定义

C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wsuggest-override")
// 忽略警告并推送当前诊断状态

#include <cudnn_frontend.h>
// 包含 cudnn_frontend 头文件

C10_DIAGNOSTIC_POP()
// 恢复之前的诊断状态并弹出

#include <ATen/TensorUtils.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/cudnn/Handle.h>
#include <ATen/native/ConvUtils.h>
#include <ATen/native/cudnn/ConvShared.h>
#include <ATen/native/utils/ParamsHash.h>
#include <cudnn_frontend_find_plan.h>
#include <cudnn_frontend_get_plan.h>
// 包含其他必要的 ATen 和 CuDNN 头文件

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/env.h>
// 包含其他必要的 C10 头文件

#include <list>
#include <unordered_map>
// 包含标准库头文件

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#endif
// 根据条件选择包含不同的 ATen 头文件

#ifdef __linux__
#include <dlfcn.h>
#endif
// 如果是 Linux 系统，包含动态链接库的头文件

namespace at {
namespace native {

namespace {

// TODO: remove duplicate code in Conv_v7.cpp
// TODO：在 Conv_v7.cpp 中移除重复的代码

constexpr int64_t operator"" _TiB(unsigned long long n) {
  return size_t(n) << 40;
}
// 定义一个字面量运算符，将 TiB 转换为字节大小的整数常量

uint8_t getAlignment(const Tensor& t) {
  // 获取张量的对齐方式，单位是字节
  uint8_t alignment = 1;
  uintptr_t address = reinterpret_cast<uintptr_t>(t.const_data_ptr());
  for (; alignment < 32; alignment *= 2) {
    if (address % (alignment * 2)) {
      return alignment;
    }
  }
  return alignment;
}
// 返回张量的对齐方式，通过地址计算得出

cudnn_frontend::Tensor getTensorDescriptorWithTypeVirtual(
    const Tensor& t,
    const int64_t id,
    const uint8_t alignment,
    const cudnnDataType_t dataType,
    const at::MemoryFormat memory_format,
    const bool _virtual) {
// 获取带有虚拟类型的张量描述符

#if defined(__linux__) && !defined(FBCODE_CAFFE2) && CUDNN_MAJOR == 8 && \
    CUDNN_MINOR > 5
// 如果是 Linux 系统且不是 FBCODE_CAFFE2，并且 CuDNN 版本主要号为 8 且次要号大于 5

  // Workaround for cudnn error handling deficiency, that results in a crash on
  // Ubuntu-22+ if `libnvrtc.so` is not found on the system, which strictly
  // speaking is not necessary for usecases below See
  // https://github.com/pytorch/pytorch/issues/97041
  // 对 CuDNN 错误处理不足的问题进行处理，如果系统上找不到 `libnvrtc.so`，在 Ubuntu-22+ 上会导致崩溃，
  // 严格来说在以下情况下不是必需的用例。参见上述链接

  static C10_UNUSED auto cudnn_cnn_infer_handler = [] {
    void* handle = dlopen("libcudnn_cnn_infer.so.8", RTLD_LAZY);
    char* err = dlerror();
    if (!handle) {
      TORCH_WARN(
          "Attempt to open cnn_infer failed: handle=", handle, " error: ", err);
    } else if (err) {
      TORCH_WARN("Applied workaround for CuDNN issue, install nvrtc.so");
    }
    return handle;
  }();
  // 静态变量 cudnn_cnn_infer_handler，用于处理 CuDNN 的问题，尝试打开 cnn_infer 失败时发出警告，处理错误信息
#endif
  // 获取张量的尺寸信息
  auto sizes = t.sizes();
  // 获取张量的步幅信息
  auto strides = t.strides();
  // 检查内存格式是否为 ChannelsLast 或 ChannelsLast3d
  bool channels_last = memory_format == at::MemoryFormat::ChannelsLast ||
      memory_format == at::MemoryFormat::ChannelsLast3d;

  // 复制步幅信息到新的向量中
  std::vector<int64_t> strides_copy(std::begin(strides), std::end(strides));
  // 调整维度为1的步幅
  fixSizeOneDimStride<int64_t>(
      sizes.size(), &sizes[0], (int64_t*)&strides_copy[0], channels_last);
  // 使用 CUDNN 前端 API 创建张量描述符
  auto r = cudnn_frontend::TensorBuilder()
               .setDim(sizes.size(), sizes.data())
               .setStrides(strides_copy.size(), strides_copy.data())
               .setId(id)
               .setAlignment(alignment)
               .setDataType(dataType)
               .setVirtual(_virtual)
               .build();
  // 返回创建的张量描述符
  return r;
}

// 获取张量描述符
cudnn_frontend::Tensor getTensorDescriptor(
    const Tensor& t,
    const int64_t id,
    const uint8_t alignment,
    const at::MemoryFormat memory_format) {
  // 调用具体的获取带有类型虚拟标志的张量描述符函数
  return getTensorDescriptorWithTypeVirtual(
      t, id, alignment, getCudnnDataType(t), memory_format, false);
}

// 获取卷积描述符
cudnn_frontend::ConvDesc_v8 getConvDescriptor(
    cudnnDataType_t dataType,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    const at::ScalarType scalar_type) {
  // 确定卷积的维度
  uint64_t convDim = stride.size();
  // 如果标量类型是 kBFloat16 或 kHalf，则数据类型设置为 CUDNN_DATA_FLOAT
  if (scalar_type == kBFloat16 || scalar_type == kHalf) {
    dataType = CUDNN_DATA_FLOAT;
  }
  // 使用 CUDNN 前端 API 创建卷积描述符
  return cudnn_frontend::ConvDescBuilder()
      .setDataType(dataType)
      .setMathMode(CUDNN_CROSS_CORRELATION)
      .setNDims(convDim)
      .setStrides(convDim, stride.data())
      .setPrePadding(convDim, padding.data())
      .setPostPadding(convDim, padding.data())
      .setDilation(convDim, dilation.data())
      .build();
}

// 过滤引擎配置
void filterEngineConfigs(
    cudnn_frontend::EngineConfigList& from,
    cudnn_frontend::EngineConfigList& to,
    bool deterministic,
    bool allow_tf32,
    c10::ScalarType scalar_type) {
  // 定义过滤器函数
  auto filter = [=](cudnnBackendDescriptor_t c) {
    if (deterministic) {
      // 如果是确定性操作，检查是否有非确定性的数值注释
      if (cudnn_frontend::hasNumericalNote<
              CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC>(c)) {
        return true;
      }
    }
    // 检查是否有输入下转换的数值注释
    if (cudnn_frontend::hasNumericalNote<
            CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS>(c)) {
      return true;
    }
    // 如果标量类型是 kFloat，检查是否不允许 TF32 并且有张量核心数值注释
    if (scalar_type == kFloat) {
      // TODO: 在什么条件下这样做是合适的？
      if (!allow_tf32 &&
          cudnn_frontend::hasNumericalNote<CUDNN_NUMERICAL_NOTE_TENSOR_CORE>(
              c)) {
        return true;
      }
    }
    // 默认不返回
    return false;
  };
  // 使用 CUDNN 前端 API 进行过滤
  cudnn_frontend::filter(from, to, filter);
}

// 定义缓存键结构
struct CacheKey {
  ConvolutionParams params;  // 卷积参数
  cudnnBackendDescriptorType_t operation;  // 操作类型
  uint8_t x_alignment;  // X 对齐
  uint8_t w_alignment;  // W 对齐
  uint8_t y_alignment;  // Y 对齐
};
// 定义一个结构体 CacheKeyFused，用于缓存融合卷积操作的参数
struct CacheKeyFused {
  ConvolutionParams params; // 存储卷积参数
  // 下面几行是对齐要求，用于内存对齐优化
  uint8_t x_alignment; // 输入张量 x 的对齐要求
  uint8_t w_alignment; // 权重张量 w 的对齐要求
  uint8_t y_alignment; // 输出张量 y 的对齐要求
  uint8_t z_alignment; // 可选输入张量 z 的对齐要求
  uint8_t b_alignment; // 可选偏置张量 b 的对齐要求
  // TODO: 这里是否将 alpha 作为键值存储是有意义的？但 alpha 是图级别的参数...
  float alpha; // 融合操作中的 alpha 参数
};

// CacheKeyWrapper 结构体，继承自 ParamsWrapper<CacheKey>，用于封装 CacheKey 的参数
struct CacheKeyWrapper : ParamsWrapper<CacheKey> {
  // 构造函数，初始化 CacheKeyWrapper 对象
  CacheKeyWrapper(
      const cudnnBackendDescriptorType_t operation, // cuDNN 后端描述符类型
      const Tensor& y, // 输出张量 y
      const Tensor& x, // 输入张量 x
      const Tensor& w, // 权重张量 w
      const IntArrayRef padding, // 填充参数
      const IntArrayRef stride, // 步幅参数
      const IntArrayRef dilation, // 膨胀参数
      int64_t groups, // 分组参数
      bool deterministic, // 是否确定性操作
      bool allow_tf32) { // 是否允许 TF32 操作
    at::MemoryFormat memory_format = cudnn_conv_suggest_memory_format(x, w);
    // 设置卷积参数到 pod.params
    setConvolutionParams(
        &(this->pod.params),
        x,
        w,
        padding,
        stride,
        dilation,
        groups,
        deterministic,
        allow_tf32,
        memory_format);
    this->pod.operation = operation; // 设置操作类型
    this->pod.x_alignment = getAlignment(x); // 设置输入张量 x 的对齐要求
    this->pod.y_alignment = getAlignment(y); // 设置输出张量 y 的对齐要求
    this->pod.w_alignment = getAlignment(w); // 设置权重张量 w 的对齐要求
  }
};

// CacheKeyFusedWrapper 结构体，继承自 ParamsWrapper<CacheKeyFused>，用于封装 CacheKeyFused 的参数
struct CacheKeyFusedWrapper : ParamsWrapper<CacheKeyFused> {
  // 构造函数，初始化 CacheKeyFusedWrapper 对象
  CacheKeyFusedWrapper(
      const Tensor& y, // 输出张量 y
      const Tensor& x, // 输入张量 x
      const Tensor& w, // 权重张量 w
      const Tensor& z, // 可选输入张量 z
      const Tensor& b, // 可选偏置张量 b
      const float alpha, // 融合操作中的 alpha 参数
      const IntArrayRef padding, // 填充参数
      const IntArrayRef stride, // 步幅参数
      const IntArrayRef dilation, // 膨胀参数
      int64_t groups, // 分组参数
      bool deterministic, // 是否确定性操作
      bool allow_tf32) { // 是否允许 TF32 操作
    at::MemoryFormat memory_format = cudnn_conv_suggest_memory_format(x, w);
    // 设置卷积参数到 pod.params
    setConvolutionParams(
        &(this->pod).params,
        x,
        w,
        padding,
        stride,
        dilation,
        groups,
        deterministic,
        allow_tf32,
        memory_format);
    this->pod.x_alignment = getAlignment(x); // 设置输入张量 x 的对齐要求
    this->pod.y_alignment = getAlignment(y); // 设置输出张量 y 的对齐要求
    this->pod.w_alignment = getAlignment(w); // 设置权重张量 w 的对齐要求
    this->pod.z_alignment = getAlignment(z); // 设置可选输入张量 z 的对齐要求
    this->pod.b_alignment = getAlignment(b); // 设置可选偏置张量 b 的对齐要求
    this->pod.alpha = alpha; // 设置融合操作中的 alpha 参数
  }
};

// 静态函数 getLRUCacheLimit()，用于获取 LRUCache 的限制大小
static int getLRUCacheLimit() {
  constexpr int DEFAULT_LIMIT =
      10000; // 大约对应于 2 GiB，假设每个 ExecutionPlan 大约 200 KiB
  // 0 用于指示无限制
  // 负值用于指示不缓存
  static int limit = [&] {
    const char* val = getenv("TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT");
    if (!val) {
      return DEFAULT_LIMIT; // 如果环境变量未设置，默认使用 DEFAULT_LIMIT
    }
    try {
      return std::stoi(val); // 尝试解析环境变量为整数
    } catch (std::invalid_argument const& e) {
      TORCH_WARN(
          "invalid TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT,",
          " using default LRU cache limit of ",
          DEFAULT_LIMIT,
          " entries.");
    } catch (std::out_of_range const& e) {
      TORCH_WARN(
          "invalid TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT,",
          " using default LRU cache limit of ",
          DEFAULT_LIMIT,
          " entries.");
    }
    return DEFAULT_LIMIT; // 解析失败时，默认使用 DEFAULT_LIMIT
  }();
    // 返回默认限制值，这是一个立即调用的匿名函数
    return DEFAULT_LIMIT;
  }();
  // 返回限制值
  return limit;
}

// 结构模板 BenchmarkCache，用于缓存 cudnn_frontend::ExecutionPlan 对象
template <typename T, typename KeyType>
struct BenchmarkCache {
  // 引擎缓存的访问顺序链表
  std::list<KeyType> engine_cache_order;
  // 引擎缓存的哈希映射，存储 ExecutionPlan 对象及其在访问顺序链表中的迭代器
  std::unordered_map<
      KeyType,
      std::pair<
          cudnn_frontend::ExecutionPlan,
          typename std::list<KeyType>::iterator>,
      ParamsWrapperHash<KeyType>>
      engine_cache;

  // 不使用互斥锁，因为缓存现在是每个线程局部的（针对 v8），如果知道不会被其他线程无效化，也可以返回 Execution Plan 的指针
  cudnn_frontend::ExecutionPlan* find(const KeyType& key) {
    const int lru_cache_limit = getLRUCacheLimit();
    if (lru_cache_limit < 0) {
      return nullptr;
    }
    auto it = engine_cache.find(key);
    if (it == engine_cache.end()) {
      return nullptr;
    }
    if (lru_cache_limit) {
      // 更新最近访问的条目，将其移到访问顺序链表的开头
      engine_cache_order.splice(
          engine_cache_order.begin(), engine_cache_order, it->second.second);
    }
    return &(it->second.first);
  }

  // 更新缓存中的条目
  void update(const KeyType& key, T& results) {
    int lru_cache_limit = getLRUCacheLimit();
    if (lru_cache_limit < 0) {
      return;
    } else if (lru_cache_limit) {
      auto it = engine_cache.find(key);
      if (it == engine_cache.end()) {
        // 如果缓存中不存在该条目
        if ((long)engine_cache.size() >= lru_cache_limit) {
          // 如果缓存超过了限制，需要进行淘汰
          auto erase_count = engine_cache.erase(engine_cache_order.back());
          TORCH_INTERNAL_ASSERT(
              erase_count == 1,
              "CUDNN V8 LRU Cache Corrupted (eviction key not in map). Please report a bug to PyTorch.");
          engine_cache_order.pop_back();
        }
        // 在访问顺序链表的开头添加新条目
        engine_cache_order.emplace_front(key);
        // 在哈希映射中插入新条目
        engine_cache.emplace(
            key, std::make_pair(results, engine_cache_order.begin()));
      } else {
        // 如果缓存中已存在该条目，更新其结果
        it->second.first = results;
        // 更新最近访问的条目，将其移到访问顺序链表的开头
        engine_cache_order.splice(
            engine_cache_order.begin(), engine_cache_order, it->second.second);
      }
    } else {
      // 如果 lru_cache_limit 为 0，直接删除缓存中的条目
      engine_cache.erase(key);
      // 在哈希映射中插入新条目，使用虚拟的迭代器表示
      engine_cache.emplace(
          key,
          std::make_pair(results, engine_cache_order.end())); // dummy iterator
    }
  }
};

// @eqy: 使用线程局部缓存，因为 cuDNN Execution Plans 不能保证在所有引擎中线程安全，参见 https://docs.nvidia.com/deeplearning/cudnn/release-notes/index.html
thread_local BenchmarkCache<cudnn_frontend::ExecutionPlan, CacheKeyWrapper>
    benchmark_cache;
thread_local BenchmarkCache<cudnn_frontend::ExecutionPlan, CacheKeyFusedWrapper>
    benchmark_cache_fused;

} // namespace
    // 在函数中使用设备保护器，确保操作在正确的 CUDA 设备上进行
    c10::DeviceGuard g(x.options().device());
    // 获取计划中的工作空间大小
    auto workspace_size = plan.getWorkspaceSize();
    // 分配 CUDA 缓存分配器中的工作空间内存
    auto workspace_ptr =
        c10::cuda::CUDACachingAllocator::get()->allocate(workspace_size);
    // 数据指针数组，用于保存输入和输出数据的指针
    void* data_ptrs[3];
    
    // 根据不同的操作类型设置数据指针
    if (operation == CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR) {
      // 前向卷积操作的数据指针设置
      data_ptrs[0] = const_cast<void*>(x.const_data_ptr());
      data_ptrs[1] = y.data_ptr();
      data_ptrs[2] = const_cast<void*>(w.const_data_ptr());
    } else if (
        operation ==
        CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR) {
      // 反向数据卷积操作的数据指针设置
      data_ptrs[0] = x.data_ptr();
      data_ptrs[1] = const_cast<void*>(y.const_data_ptr());
      data_ptrs[2] = const_cast<void*>(w.const_data_ptr());
    } else if (
        operation ==
        CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR) {
      // 反向滤波器卷积操作的数据指针设置
      data_ptrs[0] = const_cast<void*>(x.const_data_ptr());
      data_ptrs[1] = const_cast<void*>(y.const_data_ptr());
      data_ptrs[2] = w.data_ptr();
    } else {
      // 默认操作的数据指针设置
      data_ptrs[0] = x.data_ptr();
      data_ptrs[1] = y.data_ptr();
      data_ptrs[2] = w.data_ptr();
    }
    
    // 设置操作的唯一标识符数组
    int64_t uids[] = {'x', 'y', 'w'};
    // 构建 CUDNN 前端的变量包，包括工作空间指针、数据指针和唯一标识符
    auto variantPack =
        cudnn_frontend::VariantPackBuilder()
            .setWorkspacePointer(workspace_size ? workspace_ptr.get() : nullptr)
            .setDataPointers(3, data_ptrs)
            .setUids(3, uids)
            .build();
    // 执行 CUDNN 后端操作，并检查执行结果
    AT_CUDNN_CHECK(cudnnBackendExecute(
        handle, plan.get_raw_desc(), variantPack.get_raw_desc()));
}

// 执行融合的卷积计划
void run_conv_plan_fused(
    cudnnHandle_t handle,                    // cuDNN句柄
    const Tensor& x,                         // 输入张量 x
    const Tensor& y,                         // 输入张量 y
    const Tensor& w,                         // 权重张量 w
    const Tensor& z,                         // 输入张量 z
    const Tensor& b,                         // 偏置张量 b
    const cudnn_frontend::ExecutionPlan& plan // cuDNN前端执行计划
) {
  c10::DeviceGuard g(x.options().device());  // 设置设备类型与张量 x 相同
  auto workspace_size = plan.getWorkspaceSize();  // 获取工作空间大小
  auto workspace_ptr =
      c10::cuda::CUDACachingAllocator::get()->allocate(workspace_size);  // 分配 CUDA 工作空间
  void* data_ptrs[] = {
      x.data_ptr(), y.data_ptr(), w.data_ptr(), z.data_ptr(), b.data_ptr()};  // 数据指针数组
  int64_t uids[] = {'x', 'y', 'w', 'z', 'b'};  // 数据唯一标识符数组
  auto variantPack =
      cudnn_frontend::VariantPackBuilder()     // 构建变体包
          .setWorkspacePointer(workspace_size ? workspace_ptr.get() : nullptr)  // 设置工作空间指针
          .setDataPointers(5, data_ptrs)       // 设置数据指针
          .setUids(5, uids)                    // 设置唯一标识符
          .build();                            // 构建变体包对象
  AT_CUDNN_CHECK(cudnnBackendExecute(
      handle, plan.get_raw_desc(), variantPack.get_raw_desc()));  // 执行 cuDNN 后端计划
}

// 构建操作图
auto build_opgraph(
    const cudnnHandle_t handle,                // cuDNN句柄
    const cudnnBackendDescriptorType_t desc,   // cuDNN后端描述类型
    const Tensor& x,                          // 输入张量 x
    const Tensor& y,                          // 输入张量 y
    const Tensor& w,                          // 权重张量 w
    const CacheKeyWrapper& key,               // 缓存键包装器
    const IntArrayRef padding,                // 填充数组引用
    const IntArrayRef stride,                 // 步长数组引用
    const IntArrayRef dilation                // 膨胀数组引用
) {
  auto op = cudnn_frontend::OperationBuilder(desc)  // 构建操作
                .setxDesc(getTensorDescriptor(
                    x, 'x', key.pod.x_alignment, key.pod.params.memory_format))  // 设置输入 x 描述符
                .setyDesc(getTensorDescriptor(
                    y, 'y', key.pod.y_alignment, key.pod.params.memory_format))  // 设置输入 y 描述符
                .setwDesc(getTensorDescriptor(
                    w, 'w', key.pod.w_alignment, key.pod.params.memory_format))  // 设置权重 w 描述符
                .setcDesc(getConvDescriptor(
                    key.pod.params.dataType,
                    padding,
                    stride,
                    dilation,
                    x.scalar_type()))  // 设置卷积描述符
                .build();  // 构建操作对象
  std::array<cudnn_frontend::Operation const*, 1> ops = {&op};  // 操作数组
  auto opGraph = cudnn_frontend::OperationGraphBuilder()  // 构建操作图
                     .setHandle(handle)                   // 设置句柄
                     .setOperationGraph(ops.size(), ops.data())  // 设置操作图
                     .build();                           // 构建操作图对象
  return opGraph;  // 返回操作图对象
}

// 构建融合操作图
auto build_opgraph_fused(
    const cudnnHandle_t handle,                // cuDNN句柄
    const Tensor& x,                          // 输入张量 x
    const Tensor& y,                          // 输入张量 y
    const Tensor& w,                          // 权重张量 w
    const Tensor& z,                          // 输入张量 z
    const Tensor& b,                          // 偏置张量 b
    const float alpha,                        // 浮点数 alpha
    const CacheKeyFusedWrapper& key,          // 融合键包装器
    const IntArrayRef padding,                // 填充数组引用
    const IntArrayRef stride                  // 步长数组引用
) {
    // 待补充
}

// 获取生成器源信息
auto get_generator_sources(
    const cudnnBackendDescriptorType_t& desc,  // cuDNN后端描述类型引用
    const Tensor& x,                          // 输入张量 x
    const bool deterministic,                 // 布尔值：确定性
    const bool allow_tf32,                    // 布尔值：允许 TF32
    const cudnnBackendHeurMode_t heur_mode,   // cuDNN后端启发模式
    const bool heuristic,                     // 布尔值：启发式
    const bool fallback                       // 布尔值：后备
) {
  // 基于启发式的引擎配置生成器方法
  const auto heurgen_method =
      [&x, deterministic, allow_tf32, heur_mode](  // 引擎配置生成器方法
          cudnn_frontend::OperationGraph& opGraph)  // 操作图引用
      -> cudnn_frontend::EngineConfigList {        // 返回值：引擎配置列表
    // 使用 cudnn_frontend 的 EngineHeuristicsBuilder 创建启发式引擎配置
    auto heuristics = cudnn_frontend::EngineHeuristicsBuilder()
                          .setOperationGraph(opGraph)  // 设置操作图
                          .setHeurMode(heur_mode)      // 设置启发模式
                          .build();                    // 构建引擎配置

    // 获取引擎配置列表中的最后一个配置
    auto& engine_configs =
        heuristics.getEngineConfig(heuristics.getEngineConfigCount());

    // 创建一个空的引擎配置列表
    cudnn_frontend::EngineConfigList filtered_configs;

    // 根据条件过滤引擎配置列表
    filterEngineConfigs(
        engine_configs,
        filtered_configs,
        deterministic,
        allow_tf32,
        x.scalar_type());

    // 返回过滤后的引擎配置列表
    return filtered_configs;
  };

  // 基于回退列表生成引擎配置的方法
  const auto fallback_method = [&desc, &x, deterministic, allow_tf32](
                                   cudnn_frontend::OperationGraph& opGraph)
      -> cudnn_frontend::EngineConfigList {

    // 使用 cudnn_frontend 的 EngineFallbackListBuilder 创建回退引擎列表
    auto fallback = cudnn_frontend::EngineFallbackListBuilder()
                        .setOperationGraph(opGraph)  // 设置操作图
                        .setOperation(desc)          // 设置操作描述
                        .build();                    // 构建回退列表

    // 获取回退列表中的引擎配置列表
    auto& fallback_list = fallback.getFallbackList();

    // 创建一个空的引擎配置列表
    cudnn_frontend::EngineConfigList filtered_configs;

    // 根据条件过滤引擎配置列表
    filterEngineConfigs(
        fallback_list,
        filtered_configs,
        deterministic,
        allow_tf32,
        x.scalar_type());

    // 返回过滤后的引擎配置列表
    return filtered_configs;
  };

  // 如果同时启用启发式和回退
  if (heuristic && fallback) {
    // 创建包含启发式方法和回退方法的源向量
    std::vector<cudnn_frontend::GeneratorSource> sources = {
        heurgen_method, fallback_method};
    return sources;  // 返回源向量
  }
  // 如果只启用启发式
  else if (heuristic) {
    // 创建包含仅启发式方法的源向量
    std::vector<cudnn_frontend::GeneratorSource> sources = {heurgen_method};
    return sources;  // 返回源向量
  }
  // 如果只启用回退
  else {
    // 创建包含仅回退方法的源向量
    std::vector<cudnn_frontend::GeneratorSource> sources = {fallback_method};
    return sources;  // 返回源向量
  }
}

# 定义一个函数，返回当前 CUDA 设备上可用的工作空间大小（以字节为单位）
int64_t get_available_workspace() {
  # 默认使用第一个 CUDA 设备
  c10::DeviceIndex device = 0;
  # 获取当前 CUDA 设备的编号
  C10_CUDA_CHECK(c10::cuda::GetDevice(&device));
  # 查询当前 CUDA 设备的最大可用内存块大小
  size_t max_block_size = 0;
  c10::cuda::CUDACachingAllocator::cacheInfo(device, &max_block_size);
  # 将最大内存块大小转换为 int64_t 类型后返回
  return static_cast<int64_t>(max_block_size);
}

# 定义一个静态的 JSON 对象，用于处理错误修正信息
static nlohmann::json errata_json_handle;

# 计划错误修正的异常处理函数，根据执行计划标签检查错误修正情况
bool plan_errata_exception(
    const cudnnHandle_t handle,
    const std::string& executionPlanTag) {
  # 尝试从配置中加载错误修正信息到静态 JSON 对象中
  static bool has_json =
      cudnn_frontend::load_from_config(errata_json_handle, "");
  # 若加载成功则进行错误修正检查，否则返回 false
  if (!has_json) {
    return false;
  } else {
    return cudnn_frontend::check_errata(
        errata_json_handle, executionPlanTag, handle, []() { return true; });
  }
}

# 生成和筛选计划的函数，选择适合的计划进行执行
void generate_and_filter_plans(
    const cudnnHandle_t handle,
    cudnn_frontend::OperationGraph& opGraph,
    cudnn_frontend::EngineConfigGenerator& generator,
    const Tensor& x,
    cudnn_frontend::executionPlans_t& valid_plans,
    at::DataPtr& workspace_ptr) {
  # 定义初始的谓词函数，用于判断计划是否存在错误修正
  auto initial_predicate_function =
      [&](cudnn_frontend::ExecutionPlan const& plan) -> bool {
    return plan_errata_exception(handle, plan.getTag());
  };
  # 获取符合条件的所有计划
  auto plans =
      generator.cudnnGetPlan(handle, opGraph, initial_predicate_function);
  # 获取当前 CUDA 设备上的最大工作空间大小
  int64_t max_block_size = get_available_workspace();
  int64_t max_workspace_size = 0;
  # 遍历所有计划，筛选出工作空间小于等于最大块大小的有效计划
  std::for_each(
      plans.begin(), plans.end(), [&](cudnn_frontend::ExecutionPlan& plan) {
        int64_t curr_workspace_size = plan.getWorkspaceSize();
        if (curr_workspace_size <= max_block_size) {
          if (curr_workspace_size > max_workspace_size) {
            max_workspace_size = plan.getWorkspaceSize();
          }
          valid_plans.emplace_back(std::move(plan));
        }
      });
  # 检查最大工作空间大小是否小于 1 TiB，否则抛出内存不足的异常
  TORCH_CHECK_WITH(
      OutOfMemoryError,
      max_workspace_size < 1_TiB,
      "Not enough memory for workspace!");
  bool remove_invalid = false;
  # 尝试为工作空间分配内存，若分配失败则减小工作空间大小并重试
  while (max_workspace_size) {
    try {
      workspace_ptr =
          c10::cuda::CUDACachingAllocator::get()->allocate(max_workspace_size);
      break;
    } catch (c10::OutOfMemoryError& e) {
      max_workspace_size /= 2;
      (void)cudaGetLastError(); // 清除 CUDA 错误状态
      remove_invalid = true;
    }
  }
  # 若有计划因为工作空间过大无法分配而被移除，则重新筛选有效计划
  if (remove_invalid) {
    cudnn_frontend::executionPlans_t new_valid_plans;
    for (auto& plan : valid_plans) {
      if (plan.getWorkspaceSize() <= max_workspace_size) {
        new_valid_plans.emplace_back(std::move(plan));
      }
    }
    valid_plans = std::move(new_valid_plans);
  }
}

# 从查询结果中获取计划的函数，用于计划查找和选择
auto get_plans_from_find(
    const cudnnHandle_t handle,
    const cudnnBackendDescriptorType_t desc,
    const Tensor& x,
    const Tensor& y,
    const Tensor& w,
    const CacheKeyWrapper& key,
    const IntArrayRef padding,
    const IntArrayRef stride,
    const IntArrayRef dilation,
    const bool deterministic,
    // 定义函数，生成和返回按照优化计划排序后的 cudnn 前端执行计划列表
    const bool allow_tf32) {
      // 构建操作图
      auto opGraph =
          build_opgraph(handle, desc, x, y, w, key, padding, stride, dilation);
      // 获取数据指针数组
      void* data_ptrs[] = {x.data_ptr(), y.data_ptr(), w.data_ptr()};
      // 定义唯一标识数组
      int64_t uids[] = {'x', 'y', 'w'};
      // 如果要运行所有算法，则不必担心获取最佳算法顺序
      // 获取生成器的源数据集合
      auto sources = get_generator_sources(
          desc, x, deterministic, allow_tf32, CUDNN_HEUR_MODE_INSTANT, true, true);
      // 创建引擎配置生成器对象
      cudnn_frontend::EngineConfigGenerator generator(
          sources.size(), sources.data());
      // 定义有效执行计划列表
      cudnn_frontend::executionPlans_t valid_plans;
      // 设备保护区域，设置当前张量的设备
      c10::DeviceGuard g(x.options().device());
      // 分配工作空间指针
      at::DataPtr workspace_ptr;
      // 生成并筛选执行计划
      generate_and_filter_plans(
          handle, opGraph, generator, x, valid_plans, workspace_ptr);
      // 构建变体包
      auto variantPack =
          cudnn_frontend::VariantPackBuilder()
              .setDataPointers(3, data_ptrs)
              .setUids(3, uids)
              .setWorkspacePointer(workspace_ptr ? workspace_ptr.get() : nullptr)
              .build();
    
      // 获取 cuDNN 全局上下文中的基准限制
      auto benchmark_limit = at::globalContext().benchmarkLimitCuDNN();
      benchmark_limit = benchmark_limit ? benchmark_limit : 10000;
      // 使用时间排序的计划生成器，生成按照一次抽样策略排序的计划
      auto plans = cudnn_frontend::time_sorted_plan<
          cudnn_frontend::CudnnFindSamplingTechnique::CUDNN_FIND_SAMPLE_ONCE>(
          handle, std::move(valid_plans), variantPack, benchmark_limit);
    
      // 创建排序后的执行计划列表
      cudnn_frontend::executionPlans_t sorted_plans;
      // 将计划逐个移动到排序计划列表中
      for (auto& plan : plans) {
        sorted_plans.emplace_back(std::move(plan));
      }
      // 返回排序后的执行计划列表
      return sorted_plans;
    }
auto get_plans_from_find_fused(
    const cudnnHandle_t handle,                           # 函数声明：获取融合操作的执行计划
    const Tensor& x,                                     # 输入张量 x
    const Tensor& y,                                     # 输入张量 y
    const Tensor& w,                                     # 权重张量 w
    const Tensor& z,                                     # 张量 z
    const Tensor& b,                                     # 偏置张量 b
    const float alpha,                                   # 浮点数 alpha
    const CacheKeyFusedWrapper& key,                     # 融合操作的缓存键
    const IntArrayRef padding,                           # 填充数组
    const IntArrayRef stride,                            # 步幅数组
    const IntArrayRef dilation,                          # 膨胀数组
    const bool deterministic,                            # 是否确定性操作
    const bool allow_tf32) {                             # 是否允许 TF32 精度
  auto opGraph = build_opgraph_fused(                    # 构建融合操作的操作图
      handle, x, y, w, z, b, alpha, key, padding, stride, dilation);
  void* data_ptrs[] = {                                  # 数据指针数组
      x.data_ptr(), y.data_ptr(), w.data_ptr(), z.data_ptr(), b.data_ptr()};
  int64_t uids[] = {'x', 'y', 'w', 'z', 'b'};            # 唯一标识数组

  auto sources = get_generator_sources(                  # 获取生成器的数据源
      CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR,  # CUDNN 后端操作描述符
      x,                                                 # 输入张量 x
      deterministic,                                     # 是否确定性
      allow_tf32,                                        # 是否允许 TF32
      CUDNN_HEUR_MODE_INSTANT,                           # 启发式模式：瞬时模式
      true,                                              # 是否使用瞬时模式
      true);                                             # 是否使用后端操作描述符
  cudnn_frontend::EngineConfigGenerator generator(       # 创建引擎配置生成器
      sources.size(), sources.data());
  cudnn_frontend::executionPlans_t valid_plans;          # 有效的执行计划集合
  c10::DeviceGuard g(x.options().device());              # 设备保护：使用 x 的设备
  at::DataPtr workspace_ptr;                             # 工作空间指针
  generate_and_filter_plans(                             # 生成和过滤计划
      handle, opGraph, generator, x, valid_plans, workspace_ptr);
  auto variantPack =                                     # 变体包
      cudnn_frontend::VariantPackBuilder()
          .setDataPointers(5, data_ptrs)                  # 设置数据指针
          .setUids(5, uids)                              # 设置唯一标识
          .setWorkspacePointer(workspace_ptr ? workspace_ptr.get() : nullptr)  # 设置工作空间指针
          .build();                                      # 构建变体包

  auto benchmark_limit = at::globalContext().benchmarkLimitCuDNN();  # 基准测试限制
  benchmark_limit = benchmark_limit ? benchmark_limit : 10000;  # 如果基准测试限制不存在，则设为 10000
  auto plans = cudnn_frontend::time_sorted_plan<         # 按时间排序的计划
      cudnn_frontend::CudnnFindSamplingTechnique::CUDNN_FIND_SAMPLE_ONCE>(
      handle, std::move(valid_plans), variantPack, benchmark_limit);  # 获取按时间排序的计划

  cudnn_frontend::executionPlans_t sorted_plans;         # 排序后的执行计划集合
  for (auto& plan : plans) {                             # 遍历计划集合
    sorted_plans.emplace_back(std::move(plan));          # 将计划移动到排序后的计划集合中
  }
  return sorted_plans;                                   # 返回排序后的执行计划集合
}

// We only get configs from this stage to avoid building unnecessary plans that
// are never executed
auto get_configs_from_heuristics(                        # 函数声明：从启发式中获取配置
    const cudnnHandle_t handle,                           # cuDNN 句柄
    const cudnnBackendDescriptorType_t desc,              # cuDNN 后端描述符类型
    std::string& opgraph_tag,                             # 操作图标签
    const Tensor& x,                                     # 输入张量 x
    const Tensor& y,                                     # 输入张量 y
    const Tensor& w,                                     # 权重张量 w
    const CacheKeyWrapper& key,                           # 缓存键
    const IntArrayRef padding,                            # 填充数组
    const IntArrayRef stride,                             # 步幅数组
    const IntArrayRef dilation,                           # 膨胀数组
    const bool deterministic,                             # 是否确定性操作
    const bool allow_tf32,                                # 是否允许 TF32 精度
    const bool fallback) {                                # 是否回退
  auto opGraph =                                          # 构建操作图
      build_opgraph(handle, desc, x, y, w, key, padding, stride, dilation);
  opgraph_tag = opGraph.getTag();                         # 获取操作图标签
  auto heuristic_mode = at::native::cudnnv8_use_heur_mode_b()  # 启发式模式
      ? CUDNN_HEUR_MODE_B                                 # 使用模式 B
      : CUDNN_HEUR_MODE_INSTANT;                          # 否则使用瞬时模式
  auto sources = get_generator_sources(                   # 获取生成器的数据源
      desc, x, deterministic, allow_tf32, heuristic_mode, !fallback, fallback);

  cudnn_frontend::EngineConfigGenerator generator(        # 创建引擎配置生成器
      sources.size(), sources.data());
  auto configs = generator.generate_engine_config(opGraph);  # 生成引擎配置
  return configs;                                         # 返回生成的引擎配置
}

auto get_configs_from_heuristics_fused(
    const cudnnHandle_t handle,


注释：
    // 构建融合操作图，整合输入张量 x, y, w, z, b, alpha, key, padding, stride, dilation 到 opGraph
    auto opGraph = build_opgraph_fused(
        handle, x, y, w, z, b, alpha, key, padding, stride, dilation);

    // 获取操作图的标签并赋值给 opgraph_tag
    opgraph_tag = opGraph.getTag();

    // 根据 cudnnv8_use_heur_mode_b() 的返回值确定启用的推断模式
    auto heuristic_mode = at::native::cudnnv8_use_heur_mode_b()
        ? CUDNN_HEUR_MODE_B  // 如果启用了启发式模式 B，则使用 CUDNN_HEUR_MODE_B
        : CUDNN_HEUR_MODE_INSTANT;  // 否则使用即时模式 CUDNN_HEUR_MODE_INSTANT

    // 获取用于生成引擎配置的生成器源
    auto sources = get_generator_sources(
        CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR,  // 前向卷积操作的描述符
        x,  // 输入张量 x
        deterministic,  // 是否确定性
        allow_tf32,  // 是否允许 TF32
        heuristic_mode,  // 启发式模式
        !fallback,  // 是否不进行回退
        fallback);  // 是否回退

    // 使用生成器源创建 cudnn_frontend::EngineConfigGenerator 对象
    cudnn_frontend::EngineConfigGenerator generator(
        sources.size(), sources.data());

    // 生成引擎配置并赋值给 configs
    auto configs = generator.generate_engine_config(opGraph);

    // 返回生成的引擎配置
    return configs;
}

void try_plans(
    cudnn_frontend::executionPlans_t& plans,  // 传入的执行计划列表的引用
    const CacheKeyWrapper& key,  // 缓存键的引用，用于标识计划
    const cudnnHandle_t handle,  // cuDNN 句柄，用于 cuDNN 操作
    const Tensor& x,  // 输入张量 x
    const Tensor& y,  // 输入张量 y
    const Tensor& w,  // 权重张量 w
    const cudnnBackendDescriptorType_t operation) {  // cuDNN 后端描述类型
  for (auto& plan : plans) {  // 遍历传入的执行计划列表
    try {
      run_conv_plan(handle, x, y, w, plan, operation);  // 执行卷积计划
      benchmark_cache.update(key, plan);  // 更新性能基准缓存，记录计划结果
      return;  // 成功执行后直接返回
    } catch (cudnn_frontend::cudnnException& e) {  // 捕获 cuDNN 前端异常
    } catch (CuDNNError& e) {  // 捕获 cuDNN 错误
    } catch (c10::OutOfMemoryError& e) {  // 捕获内存不足错误
      (void)cudaGetLastError(); // 清除 CUDA 错误状态
    }
  }
  TORCH_CHECK(
      false, "FIND was unable to find an engine to execute this computation");  // 若所有计划都执行失败，抛出 Torch 错误
}

void try_plans_fused(
    cudnn_frontend::executionPlans_t& plans,  // 传入的执行计划列表的引用
    const CacheKeyFusedWrapper& key,  // 融合计划缓存键的引用，用于标识计划
    const cudnnHandle_t handle,  // cuDNN 句柄，用于 cuDNN 操作
    const Tensor& x,  // 输入张量 x
    const Tensor& y,  // 输入张量 y
    const Tensor& w,  // 权重张量 w
    const Tensor& z,  // 输入张量 z
    const Tensor& b) {  // 输入张量 b
  for (auto& plan : plans) {  // 遍历传入的执行计划列表
    try {
      run_conv_plan_fused(handle, x, y, w, z, b, plan);  // 执行融合卷积计划
      benchmark_cache_fused.update(key, plan);  // 更新融合计划的性能基准缓存，记录计划结果
      return;  // 成功执行后直接返回
    } catch (cudnn_frontend::cudnnException& e) {  // 捕获 cuDNN 前端异常
    } catch (CuDNNError& e) {  // 捕获 cuDNN 错误
    } catch (c10::OutOfMemoryError& e) {  // 捕获内存不足错误
      (void)cudaGetLastError(); // 清除 CUDA 错误状态
    }
  }
  TORCH_CHECK(
      false, "FIND was unable to find an engine to execute this computation");  // 若所有计划都执行失败，抛出 Torch 错误
}

bool try_configs(
    cudnn_frontend::EngineConfigList& configs,  // 引擎配置列表的引用
    const std::string& opgraph_tag,  // 操作图标识
    const CacheKeyWrapper& key,  // 缓存键的引用，用于标识配置
    const cudnnHandle_t handle,  // cuDNN 句柄，用于 cuDNN 操作
    const Tensor& x,  // 输入张量 x
    const Tensor& y,  // 输入张量 y
    const Tensor& w,  // 权重张量 w
    const cudnnBackendDescriptorType_t operation) {  // cuDNN 后端描述类型
  for (auto& config : configs) {  // 遍历引擎配置列表
    try {
      auto plan = cudnn_frontend::ExecutionPlanBuilder()
                      .setHandle(handle)
                      .setEngineConfig(config, opgraph_tag)
                      .build();  // 根据配置和操作图标识构建执行计划
      if (plan_errata_exception(handle, plan.getTag())) {  // 检查执行计划异常情况
        continue;  // 若有异常，跳过该计划
      }
      run_conv_plan(handle, x, y, w, plan, operation);  // 执行卷积计划
      benchmark_cache.update(key, plan);  // 更新性能基准缓存，记录计划结果
      return true;  // 成功执行后返回 true
    } catch (cudnn_frontend::cudnnException& e) {  // 捕获 cuDNN 前端异常
    } catch (CuDNNError& e) {  // 捕获 cuDNN 错误
    } catch (c10::OutOfMemoryError& e) {  // 捕获内存不足错误
      (void)cudaGetLastError(); // 清除 CUDA 错误状态
    }
  }
  return false;  // 所有配置均执行失败，返回 false
}

bool try_configs_fused(
    cudnn_frontend::EngineConfigList& configs,  // 引擎配置列表的引用
    const std::string& opgraph_tag,  // 操作图标识
    const CacheKeyFusedWrapper& key,  // 融合计划缓存键的引用，用于标识配置
    const cudnnHandle_t handle,  // cuDNN 句柄，用于 cuDNN 操作
    const Tensor& x,  // 输入张量 x
    const Tensor& y,  // 输入张量 y
    const Tensor& w,  // 权重张量 w
    const Tensor& z,  // 输入张量 z
    const Tensor& b) {  // 输入张量 b
  for (auto& config : configs) {  // 遍历引擎配置列表
    try {
      auto plan = cudnn_frontend::ExecutionPlanBuilder()
                      .setHandle(handle)
                      .setEngineConfig(config, opgraph_tag)
                      .build();  // 根据配置和操作图标识构建执行计划
      if (plan_errata_exception(handle, plan.getTag())) {  // 检查执行计划异常情况
        continue;  // 若有异常，跳过该计划
      }
      run_conv_plan_fused(handle, x, y, w, z, b, plan);  // 执行融合卷积计划
      benchmark_cache_fused.update(key, plan);  // 更新融合计划的性能基准缓存，记录计划结果
      return true;  // 成功执行后返回 true
    } catch (cudnn_frontend::cudnnException& e) {
        // 捕获 cudnn_frontend 库抛出的 cudnnException 异常
    } catch (CuDNNError& e) {
        // 捕获 CuDNNError 异常
    } catch (c10::OutOfMemoryError& e) {
        // 捕获 c10 库抛出的 OutOfMemoryError 异常
        (void)cudaGetLastError(); // 清除 CUDA 错误状态
    }
  }
  return false;
}

// 运行单个卷积操作的函数，使用 cuDNN 后端描述符进行配置
void run_single_conv(
    const cudnnBackendDescriptorType_t operation,  // 操作类型，用于 cuDNN 后端描述符
    const Tensor& x,  // 输入张量 x
    const Tensor& y,  // 输出张量 y
    const Tensor& w,  // 权重张量 w
    const IntArrayRef padding,  // 填充数组
    const IntArrayRef stride,   // 步幅数组
    const IntArrayRef dilation,  // 膨胀数组
    const int64_t groups,  // 组数
    const bool benchmark,  // 是否进行基准测试
    const bool deterministic,  // 是否确定性运行
    const bool allow_tf32) {  // 是否允许 TF32 加速

  cudnnHandle_t handle = getCudnnHandle();  // 获取 cuDNN 句柄
  CacheKeyWrapper key(
      operation,
      y,
      x,
      w,
      padding,
      stride,
      dilation,
      groups,
      deterministic,
      allow_tf32);  // 创建缓存键对象

  // TODO: 如果缓存更新，这段代码是否线程安全？指针是否过期？
  auto search = benchmark_cache.find(key);  // 在缓存中查找键为 key 的计划
  if (search) {  // 如果找到缓存项
    try {
      run_conv_plan(handle, x, y, w, *search, operation);  // 使用缓存的计划执行卷积操作
      return;
    } catch (c10::OutOfMemoryError& e) {
      (void)cudaGetLastError(); // 清除 CUDA 错误
    }
  }

  if (!benchmark) {  // 如果不进行基准测试
    std::string opgraph_tag;  // 需要用于 errata 过滤器的额外数据
    // 根据启发式算法获取引擎配置列表
    cudnn_frontend::EngineConfigList configs = get_configs_from_heuristics(
        handle,
        operation,
        opgraph_tag,
        x,
        y,
        w,
        key,
        padding,
        stride,
        dilation,
        deterministic,
        allow_tf32,
        false);
    if (try_configs(configs, opgraph_tag, key, handle, x, y, w, operation)) {
      return;
    }

    // 如果启发式配置失败，则使用回退配置
    configs = get_configs_from_heuristics(
        handle,
        operation,
        opgraph_tag,
        x,
        y,
        w,
        key,
        padding,
        stride,
        dilation,
        deterministic,
        allow_tf32,
        true);
    if (try_configs(configs, opgraph_tag, key, handle, x, y, w, operation)) {
      return;
    }

    // 如果仍然失败，抛出错误信息
    TORCH_CHECK(
        false, "GET was unable to find an engine to execute this computation");
  } else {  // 如果进行基准测试
    // 从查找中获取执行计划
    cudnn_frontend::executionPlans_t plans = get_plans_from_find(
        handle,
        operation,
        x,
        y,
        w,
        key,
        padding,
        stride,
        dilation,
        deterministic,
        allow_tf32);

    // 复制 v7 版本的行为：在基准测试期间清除缓存块，以减少内存消耗
    if (at::native::_cudnn_get_conv_benchmark_empty_cache()) {
      c10::cuda::CUDACachingAllocator::emptyCache();
    }

    // 尝试执行计划
    try_plans(plans, key, handle, x, y, w, operation);
  }
}
    // 获取当前 CUDNN 句柄以便操作
    cudnnHandle_t handle = getCudnnHandle();
    
    // 创建缓存键对象，用于查找或存储卷积操作的优化计划
    CacheKeyFusedWrapper key(
        y,
        x,
        w,
        z,
        b,
        alpha,
        padding,
        stride,
        dilation,
        groups,
        deterministic,
        allow_tf32);
    
    // 在优化计划缓存中查找是否存在与给定键对应的优化计划
    auto search = benchmark_cache_fused.find(key);
    if (search != benchmark_cache_fused.end()) {
        try {
            // 如果找到了对应的优化计划，则直接运行卷积操作
            run_conv_plan_fused(handle, x, y, w, z, b, *search);
            return;
        } catch (c10::OutOfMemoryError& e) {
            // 捕获内存不足的异常，清除 CUDA 错误状态
            (void)cudaGetLastError(); // clear CUDA error
        }
    }
    
    // 如果未找到优化计划，并且不是在基准测试模式下
    if (!benchmark) {
        std::string opgraph_tag; // 需要用于错误过滤的额外数据
        // 从启发式算法配置中获取引擎配置列表
        cudnn_frontend::EngineConfigList configs =
            get_configs_from_heuristics_fused(
                handle,
                opgraph_tag,
                x,
                y,
                w,
                z,
                b,
                alpha,
                key,
                padding,
                stride,
                dilation,
                deterministic,
                allow_tf32,
                false);
    
        // 尝试使用获取的配置执行卷积操作
        if (try_configs_fused(configs, opgraph_tag, key, handle, x, y, w, z, b)) {
            return;
        }
    
        // 如果启发式配置未成功，尝试使用回退配置
        configs = get_configs_from_heuristics_fused(
            handle,
            opgraph_tag,
            x,
            y,
            w,
            z,
            b,
            alpha,
            key,
            padding,
            stride,
            dilation,
            deterministic,
            allow_tf32,
            true);
    
        // 再次尝试使用回退配置执行卷积操作
        if (try_configs_fused(configs, opgraph_tag, key, handle, x, y, w, z, b)) {
            return;
        }
    
        // 如果以上尝试都失败，则抛出错误
        TORCH_CHECK(
            false, "GET was unable to find an engine to execute this computation");
    } else {
        // 如果是基准测试模式，从查找操作中获取执行计划列表
        cudnn_frontend::executionPlans_t plans = get_plans_from_find_fused(
            handle,
            x,
            y,
            w,
            z,
            b,
            alpha,
            key,
            padding,
            stride,
            dilation,
            deterministic,
            allow_tf32);
    
        // 尝试执行获取的执行计划列表中的计划
        try_plans_fused(plans, key, handle, x, y, w, z, b);
    }
void raw_cudnn_convolution_forward_out(
    // 输出张量，用于存放卷积操作的结果
    const Tensor& output,
    // 输入张量，作为卷积操作的输入数据
    const Tensor& input,
    // 卷积核张量，包含卷积操作的权重参数
    const Tensor& weight,
    // 填充大小的整数数组，用于指定卷积操作的填充方式
    const IntArrayRef padding,
    // 步幅大小的整数数组，决定卷积核在输入上的滑动步长
    const IntArrayRef stride,
    // 膨胀系数大小的整数数组，指定卷积核内部元素之间的间距
    const IntArrayRef dilation,
    // 分组数，用于分组卷积操作
    const int64_t groups,
    // 是否开启基准模式的布尔值，影响卷积运算的性能评估
    const bool benchmark,
    // 是否开启确定性模式的布尔值，影响卷积运算的重复性
    const bool deterministic,
    // 是否允许 TF32 模式的布尔值，控制是否使用 TF32 数据格式
    const bool allow_tf32) {
  // 如果输出张量的元素数量为零，则直接返回，不进行卷积操作
  if (output.numel() == 0) {
    return;
  }
  // 检查是否启用 CUDNN V8 版本调试模式
  if (at::native::cudnnv8_enabled_check_debug()) {
    // 调用单次卷积运算函数，使用 CUDNN 后端实现版本 8
    run_single_conv(
        CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR,
        input,
        output,
        weight,
        padding,
        stride,
        dilation,
        groups,
        benchmark,
        deterministic,
        allow_tf32);
  } else {
    // 否则，调用 CUDNN V7 版本的前向卷积操作函数
    raw_cudnn_convolution_forward_out_v7(
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
  }
}

void raw_cudnn_convolution_backward_input_out(
    // 梯度输入张量，用于反向传播计算输入的梯度
    const at::Tensor& grad_input,
    // 梯度输出张量，用于反向传播计算输出的梯度
    const at::Tensor& grad_output,
    // 卷积核张量，包含卷积操作的权重参数
    const at::Tensor& weight,
    // 填充大小的整数数组，用于指定卷积操作的填充方式
    const IntArrayRef padding,
    // 步幅大小的整数数组，决定卷积核在输入上的滑动步长
    const IntArrayRef stride,
    // 膨胀系数大小的整数数组，指定卷积核内部元素之间的间距
    const IntArrayRef dilation,
    // 分组数，用于分组卷积操作
    const int64_t groups,
    // 是否开启基准模式的布尔值，影响卷积运算的性能评估
    const bool benchmark,
    // 是否开启确定性模式的布尔值，影响卷积运算的重复性
    const bool deterministic,
    // 是否允许 TF32 模式的布尔值，控制是否使用 TF32 数据格式
    const bool allow_tf32) {
  // 如果梯度输入张量的元素数量为零，则直接返回，不进行反向传播计算
  if (grad_input.numel() == 0) {
    return;
  }
  // 检查是否启用 CUDNN V8 版本调试模式
  if (at::native::cudnnv8_enabled_check_debug()) {
    // 调用单次卷积运算函数，使用 CUDNN 后端实现版本 8
    run_single_conv(
        CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR,
        grad_input,
        grad_output,
        weight,
        padding,
        stride,
        dilation,
        groups,
        benchmark,
        deterministic,
        allow_tf32);
  } else {
    // 否则，调用 CUDNN V7 版本的反向传播输入梯度计算函数
    raw_cudnn_convolution_backward_input_out_v7(
        grad_input,
        grad_output,
        weight,
        padding,
        stride,
        dilation,
        groups,
        benchmark,
        deterministic,
        allow_tf32);
  }
}

void raw_cudnn_convolution_backward_weight_out(
    // 梯度权重张量，用于反向传播计算权重的梯度
    const Tensor& grad_weight,
    // 梯度输出张量，用于反向传播计算输出的梯度
    const Tensor& grad_output,
    // 输入张量，作为卷积操作的输入数据
    const Tensor& input,
    // 填充大小的整数数组，用于指定卷积操作的填充方式
    const IntArrayRef padding,
    // 步幅大小的整数数组，决定卷积核在输入上的滑动步长
    const IntArrayRef stride,
    // 膨胀系数大小的整数数组，指定卷积核内部元素之间的间距
    const IntArrayRef dilation,
    // 分组数，用于分组卷积操作
    const int64_t groups,
    // 是否开启基准模式的布尔值，影响卷积运算的性能评估
    const bool benchmark,
    // 是否开启确定性模式的布尔值，影响卷积运算的重复性
    const bool deterministic,
    // 是否允许 TF32 模式的布尔值，控制是否使用 TF32 数据格式
    const bool allow_tf32) {
  // 如果梯度权重张量的元素数量为零，则直接返回，不进行反向传播计算
  if (grad_weight.numel() == 0) {
    return;
  }
  // 检查是否启用 CUDNN V8 版本调试模式
  if (at::native::cudnnv8_enabled_check_debug()) {
    // 调用单次卷积运算函数，使用 CUDNN 后端实现版本 8
    run_single_conv(
        CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR,
        input,
        grad_output,
        grad_weight,
        padding,
        stride,
        dilation,
        groups,
        benchmark,
        deterministic,
        allow_tf32);
  } else {
    // 否则，调用 CUDNN V7 版本的反向传播权重梯度计算函数
    raw_cudnn_convolution_backward_weight_out_v7(
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
  }
}

void raw_cudnn_convolution_add_relu_out(
    // 输出张量，用于存放卷积加ReLU操作的结果
    const Tensor& output,
    // 输入张量，作为卷积操作的输入数据
    const Tensor& input,
    // 卷积核张量，包含卷积操作的权重参数
    const Tensor& weight,
    # 如果输出张量为空（元素数量为0），则直接返回，不进行任何操作
    if (output.numel() == 0) {
        return;
    }
    
    # 检查是否启用了 cudnnv8 并进行调试检查
    if (at::native::cudnnv8_enabled_check_debug()) {
        # 根据输入张量的维度情况选择合适的偏置张量视图
        auto bias_ = input.ndimension() == 4
            ? bias.view({1, bias.numel(), 1, 1})
            : bias.view({1, bias.numel(), 1, 1, 1});
        # 调用融合的卷积运算，使用了 cuDNN 库的 v8 版本
        run_fused_conv(
            input,
            output,
            weight,
            z,
            bias_,
            alpha,
            stride,
            padding,
            dilation,
            groups,
            benchmark,
            deterministic,
            allow_tf32);
    } else {
        # 调用使用 cuDNN 库 v7 的原始卷积加ReLU操作
        raw_cudnn_convolution_add_relu_out_v7(
            output,
            input,
            weight,
            z,
            alpha,
            bias,
            stride,
            padding,
            dilation,
            groups,
            benchmark,
            deterministic,
            allow_tf32);
    }
} // namespace native
} // namespace at

#endif // AT_CUDNN_ENABLED


注释：

// 结束 namespace at
} // namespace native

// 结束 namespace at

// 结束 #ifdef 指令，检查 AT_CUDNN_ENABLED 是否被定义
#endif // AT_CUDNN_ENABLED
```