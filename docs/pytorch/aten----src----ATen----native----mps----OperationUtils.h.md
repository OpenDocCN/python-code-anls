# `.\pytorch\aten\src\ATen\native\mps\OperationUtils.h`

```
//  Copyright © 2022 Apple Inc.
// 版权声明，代码版权归Apple Inc.所有

#pragma once
// 预处理指令，确保头文件只被编译一次

#include <initializer_list>
// 包含初始化列表头文件

#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 定义宏，限制仅使用操作符方法

#include <ATen/Tensor.h>
#include <ATen/Utils.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/native/mps/TensorFactory.h>
#include <c10/util/Optional.h>
#include <c10/core/ScalarType.h>
#include <torch/library.h>
#include <exception>
#include <unordered_map>
// 包含各种ATen和torch库的头文件

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/zeros_like.h>
#endif
// 条件编译指令，根据AT_PER_OPERATOR_HEADERS的定义包含不同的ATen函数头文件

#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
// 包含MetalPerformanceShaders头文件

// Fwd declarations
namespace at {
  struct TensorIteratorBase;
}
// 前向声明at命名空间下的TensorIteratorBase结构体

using namespace at::mps;
// 使用at::mps命名空间

namespace at::native::mps {

void dispatch_sync_with_rethrow(dispatch_queue_t queue, void (^block)());
// 声明dispatch_sync_with_rethrow函数原型，用于同步调度任务并重新抛出异常

struct MPSScalar {
  id<MTLBuffer> getMTLBuffer() const { return __builtin_bit_cast(id<MTLBuffer>, buffer.get()); }
  // 获取MTLBuffer对象的方法，使用内置函数__builtin_bit_cast将buffer转换为MTLBuffer类型

  size_t size = 0;
  ScalarType type = ScalarType::Undefined;
  c10::DataPtr buffer; // 存储MTLBuffer（当MPSScalar实例超出作用域时释放buffer）
  union {
    float f; // MPS不支持'double'
    at::Half h;
    int64_t i;
    bool b;
    c10::complex<float> cf;
    c10::complex<at::Half> ch;
    at::BFloat16 bf16;
  } value {};
  // 联合体value，存储MPSScalar的值，支持不同的数据类型
};

void runMPSGraph(MPSStream* mpsStream,
    MPSGraph* mpsGraph,
    NSDictionary* feeds,
    NSDictionary* results);
// 声明runMPSGraph函数原型，用于运行MPS图，接受MPSStream、MPSGraph、输入feeds和输出results

MPSDataType getMPSDataType(ScalarType scalar_type);
// 声明getMPSDataType函数原型，根据标量类型获取MPS数据类型

static inline MPSDataType getMPSDataType(const Tensor& t) {
  return getMPSDataType(t.scalar_type());
}
// 内联函数，根据Tensor对象获取MPS数据类型

MPSDataType getMPSScalarType(ScalarType scalar_type);
// 声明getMPSScalarType函数原型，根据标量类型获取MPSScalar类型

static inline MPSDataType getMPSScalarType(const Tensor& t) {
  return getMPSScalarType(t.scalar_type());
}
// 内联函数，根据Tensor对象获取MPSScalar类型

MPSScalar   getMPSScalar(const Scalar& scalar, ScalarType type);
// 声明getMPSScalar函数原型，根据标量和类型获取MPSScalar对象

std::string getMPSTypeString(ScalarType scalar_type, bool short_name = false);
// 声明getMPSTypeString函数原型，根据标量类型获取MPS类型字符串

static inline std::string getMPSTypeString(const Tensor& t, bool short_name = false) {
  return getMPSTypeString(t.scalar_type(), short_name);
}
// 内联函数，根据Tensor对象获取MPS类型字符串

std::string scalarToMetalTypeString(const c10::ScalarType& scalar_type);
// 声明scalarToMetalTypeString函数原型，将c10::ScalarType转换为Metal类型字符串

static inline std::string scalarToMetalTypeString(const Tensor& t) {
  return scalarToMetalTypeString(t.scalar_type());
}
// 内联函数，根据Tensor对象将c10::ScalarType转换为Metal类型字符串

NSArray<NSNumber*>* getTensorAxes(const Tensor& t);
// 声明getTensorAxes函数原型，获取Tensor对象的轴数组

NSArray<NSNumber*>* getTensorAxes(const IntArrayRef& sizes, at::OptionalIntArrayRef dim);
// 声明getTensorAxes函数原型，根据大小数组和可选维度获取轴数组

std::string getMPSShapeString(MPSShape* shape);
// 声明getMPSShapeString函数原型，获取MPSShape对象的形状字符串

std::string getTensorsStringKey(const TensorList& tensors, bool short_dtype = true, bool exclude_shape = false);
// 声明getTensorsStringKey函数原型，获取Tensor列表的字符串键

std::string getArrayRefString(const IntArrayRef s);
// 声明getArrayRefString函数原型，获取IntArrayRef对象的字符串表示

// 使用has_storage()函数判断返回的Tensor是否实际上是视图
Tensor gatherViewTensor(const at::Tensor& src, at::Tensor& dst);

// 将Tensor视图收集到输出Tensor中
Tensor& scatterViewTensor(const at::Tensor& src, at::Tensor& output);

// 检查是否可以对Tensor视图进行切片
bool canSliceViewTensor(const Tensor& src, MPSShape *mpsShape);
// 获取用于视图的 MPSGraphTensorData 数据，从给定的 Tensor 对象中，同时指定 MPS 格式和数据类型
MPSGraphTensorData* getMPSGraphTensorDataForView(const Tensor& src, MPSShape *mpsShape, const MPSDataType mpsDataType);

// 将输入的 MPSGraphTensor 对象转换为 IHF（Int8、Half、Float16）类型
MPSGraphTensor* castToIHFTypes(MPSGraph* mpsGraph, MPSGraphTensor* inputTensor, const Tensor& input, bool includesInt64 = false);

// 将输入的 MPSGraphTensor 对象从 IHF（Int8、Half、Float16）类型转换回原始类型
MPSGraphTensor* castFromIHFTypes(MPSGraph* mpsGraph, MPSGraphTensor* inputTensor, const Tensor& input, bool includesInt64 = false);

// 根据 Tensor 对象和内存格式获取对应的 MPSShape 对象
MPSShape* getMPSShape(const Tensor& t, c10::MemoryFormat memory_format = MemoryFormat::Contiguous);

// 根据给定的尺寸数组和内存格式获取对应的 MPSShape 对象
MPSShape* getMPSShape(IntArrayRef sizes, c10::MemoryFormat memory_format = MemoryFormat::Contiguous);

// 获取 Tensor 对象的 Metal 缓冲区存储，返回对应的 Metal 缓冲区对象
static inline id<MTLBuffer> getMTLBufferStorage(const at::Tensor& tensor) {
  return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
}

// 占位符类，用于表示 MPS 图中的占位符张量
class Placeholder {
 public:
  // 默认构造函数，初始化占位符为 nullptr 和默认张量对象
  Placeholder() : _placeholder(nullptr), _value(nullptr), _tensor(Tensor()) {}

  // 使用给定的 MPSGraphTensor 对象初始化占位符，同时设定相关属性
  Placeholder(MPSGraphTensor* mpsGraphTensor) : _placeholder(mpsGraphTensor), _value(nullptr), _tensor(Tensor()) {}

  // 使用给定的 MPSGraphTensor 对象、Tensor 对象、MPSShape 等属性初始化占位符
  Placeholder(MPSGraphTensor* mpsGraphTensor, const Tensor& self, MPSShape *mpsShape = nullptr,
              bool gatherTensorData = true, MPSDataType dataType = MPSDataTypeInvalid);

  // 返回当前占位符的 MPSGraphTensor 对象
  MPSGraphTensor* getMPSGraphTensor() {
    return _placeholder;
  }

  // 返回当前占位符的 MPSGraphTensorData 数据对象
  MPSGraphTensorData* getMPSGraphTensorData() {
    return _value;
  }

  // 判断当前占位符是否为中间结果，即 _value 是否为 nullptr
  bool isIntermediate() {
    return _value == nullptr;
  }

 private:
  MPSGraphTensor* _placeholder;  // MPS 图中的占位符张量
  MPSGraphTensorData* _value;    // MPS 图中占位符的数据
  Tensor _tensor;                // 关联的 Tensor 对象
};

// 调整 Tensor 对象的大小
void resize_tensor(Tensor* output);

// 将标量值包装为 MPSGraphTensor 对象，返回对应的 MPSGraphTensor 对象
Tensor wrapped_scalar_tensor_mps(const Scalar& scalar, const Device device);

// 截断 MPSGraphTensor 对象，返回截断后的 MPSGraphTensor 对象
MPSGraphTensor* trunc_tensor(MPSGraph* mpsGraph, MPSGraphTensor* inputTensor);

// 将 NHWC 格式的 MPSGraphTensor 对象转换为 NCHW 格式
MPSGraphTensor* convertNHWCtoNCHW(MPSGraph *mpsGraph, MPSGraphTensor* tensor);

// 将 MPSGraphTensor 对象转换为指定类型的 MPSGraphTensor 对象
MPSGraphTensor* castMPSTensor(MPSGraph *mpsGraph, MPSGraphTensor* tensor, ScalarType toType);

// 将 MPSGraphTensor 对象转换为指定数据类型的 MPSGraphTensor 对象
MPSGraphTensor* castMPSTensor(MPSGraph *mpsGraph, MPSGraphTensor* tensor, MPSDataType toType);

// 获取 MPS 图中给定 Tensor 对象的数据
MPSGraphTensorData *getMPSGraphTensorData(MPSGraph* mpsGraph, MPSStream* mpsStream, const Tensor& tensor);

// 从标量值创建 MPSGraphTensorData 对象
MPSGraphTensorData* getMPSGraphTensorFromScalar(MPSStream* mpsStream, MPSScalar& scalar);

// 创建一个 MPS 图对象
MPSGraph* make_mps_graph();

// 打印 Tensor 对象的多维数组信息
void printTensorNDArray(const Tensor& t);

// 根据 Tensor 对象创建 MPSNDArray 对象
MPSNDArray* ndArrayFromTensor(const Tensor& tensor, MPSShape *shape, MPSDataType mpsType);

// 创建无尺寸限制的 MPSGraphTensor 对象
MPSGraphTensor* mpsGraphUnrankedPlaceHolder(MPSGraph *mpsGraph, MPSDataType dataType);

// 创建带有尺寸限制的 MPSGraphTensor 对象
MPSGraphTensor* mpsGraphRankedPlaceHolder(MPSGraph *mpsGraph, MPSDataType dataType, MPSShape* mpsShape);

// 创建带有尺寸限制的 MPSGraphTensor 对象，从给定的 Tensor 对象推断形状
MPSGraphTensor* mpsGraphRankedPlaceHolder(MPSGraph *mpsGraph, const Tensor& tensor);

// 创建标量值的 MPSGraphTensor 对象
MPSGraphTensor* mpsGraphScalarPlaceHolder(MPSGraph *mpsGraph, MPSDataType dataType);

// 创建标量值的 MPSGraphTensor 对象，从给定的 Scalar 对象
MPSGraphTensor* mpsGraphScalarPlaceHolder(MPSGraph *mpsGraph, const Scalar& scalar);

// 获取内存格式的字符串表示
string get_mem_format_string(c10::MemoryFormat memory_format);

// 使用 uint64_t 类型作为 MPSCacheKey，用于缓存图及其输入/输出
// 可用于存储任何 NSObject 对象
using MPSCacheKey = uint64_t;

// 表示缓存图及其输入/输出的结构体，可用于派生以存储任何 NSObject 对象
struct MPSCachedGraph
  // 构造函数，初始化成员变量 _object 为传入对象的引用
  MPSCachedGraph(NSObject *object) : _object([object retain]) {}

  // 虚析构函数，释放 _object 引用并将其置为空指针
  virtual ~MPSCachedGraph() {
    [_object release];
    _object = nullptr;
  }

  // 模板函数，用于将当前对象转换为指定类型 T 的指针
  template<typename T>
  inline T* as() {
    return static_cast<T*>(this);
  }

  // 返回 _object 强制转换为 MPSGraph* 类型的指针
  MPSGraph *graph() const { return (MPSGraph *)_object; }

  // 返回 _object 的指针，即 NSObject*
  NSObject *object() const { return _object; }

private:
  // 成员变量，指向 NSObject 的指针，默认为 nullptr
  NSObject *_object = nullptr;
};

// 继承自 MPSCachedGraph 的结构体
struct MPSUnaryCachedGraph : public MPSCachedGraph
{
  // 构造函数，调用基类构造函数初始化
  MPSUnaryCachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}

  // 输入张量
  MPSGraphTensor *inputTensor_ = nil;

  // 输出张量
  MPSGraphTensor *outputTensor_ = nil;
};

// 继承自 MPSCachedGraph 的结构体
struct MPSUnaryGradCachedGraph : public MPSCachedGraph
{
  // 构造函数，调用基类构造函数初始化
  MPSUnaryGradCachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}

  // 梯度输出张量
  MPSGraphTensor *gradOutputTensor_ = nil;

  // 输入张量
  MPSGraphTensor *inputTensor_ = nil;

  // 输出张量，用于反向传播时作为前向传播的输入
  MPSGraphTensor *outputTensor_ = nil;

  // 梯度输入张量
  MPSGraphTensor *gradInputTensor_ = nil;
};

// 继承自 MPSCachedGraph 的结构体
struct MPSBinaryCachedGraph : public MPSCachedGraph
{
  // 构造函数，调用基类构造函数初始化
  MPSBinaryCachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}

  // 输入张量
  MPSGraphTensor *inputTensor_ = nil;

  // 另一个张量
  MPSGraphTensor *otherTensor_ = nil;

  // 输出张量
  MPSGraphTensor *outputTensor_ = nil;
};

// 继承自 MPSCachedGraph 的结构体
struct MPSBinaryGradCachedGraph : public MPSCachedGraph
{
  // 构造函数，调用基类构造函数初始化
  MPSBinaryGradCachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}

  // 梯度输出张量
  MPSGraphTensor *gradOutputTensor_ = nil;

  // 输入张量
  MPSGraphTensor *inputTensor_ = nil;

  // 另一个张量
  MPSGraphTensor *otherTensor_ = nil;

  // 梯度输入张量
  MPSGraphTensor *gradInputTensor_ = nil;
};

// MPSGraphCache 的声明
// TODO: Improve the overall design of MPSGraphCache.
// https://github.com/pytorch/pytorch/issues/77176
// 缓存持有各种键映射到图形的条目
struct MPSGraphCache
{
  // 定义 CreateCachedGraphBlock 类型为返回 MPSCachedGraph* 的块
  typedef MPSCachedGraph * (^CreateCachedGraphBlock)();

  // 缓存条目结构体
  struct CacheEntry {
    // 构造函数，初始化缓存图和键
    CacheEntry(const std::string& key, MPSCachedGraph *cachedGraph) : cachedGraph_(cachedGraph), key_(key) {}
    // 缓存的图形指针
    MPSCachedGraph* cachedGraph_ = nullptr;
    // 缓存的键
    std::string key_;
  };

public:
  // 获取单例实例的静态方法
  static MPSGraphCache* getInstance() {
    // 如果 _instance_cache 为空，则创建新的实例
    if(_instance_cache == nullptr) {
      _instance_cache = new MPSGraphCache();
    }
    return _instance_cache;
  }

  // 析构函数，释放资源，包括 serialQueue_ 和缓存中的所有图形对象
  ~MPSGraphCache() {
    dispatch_release(serialQueue_);

    // 遍历缓存，删除每个缓存图形对象
    for (const auto& i : cache_) {
      delete i.second.cachedGraph_;
    }
  }

  // 禁止复制构造函数和赋值运算符
  MPSGraphCache(const MPSGraphCache&) = delete;
  void operator=(const MPSGraphCache&) = delete;

  // 创建缓存图形对象的方法
  MPSCachedGraph* CreateCachedGraph(const std::string& key, CreateCachedGraphBlock createCacheBlock) {

    __block MPSCachedGraph* cachedGraph = nil;

    // 计算 key 的哈希值
    MPSCacheKey hash = std::hash<std::string>{}(key);
    dispatch_sync_with_rethrow(serialQueue_, ^() {
      // 验证缓存条目是否已存在
      if (cache_.count(hash) != 0) {
        // 如果缓存中已存在对应哈希值的条目，则获取该条目的引用
        auto& entry = cache_.at(hash);
        // 断言检查，用于调试，确保键值与条目的键值相符
        TORCH_INTERNAL_ASSERT_DEBUG_ONLY(key == entry.key_, "Key collision in the MPS cached graph!\n");
        // 如果符合条件，将缓存的图形赋值给cachedGraph
        cachedGraph = entry.cachedGraph_;
      } else {
        // 如果缓存中不存在对应哈希值的条目，则创建一个新的缓存块
        cachedGraph = createCacheBlock();
        // 创建CacheEntry对象，并将其插入缓存map中
        CacheEntry entry(key, cachedGraph);
        cache_.emplace(hash, entry);
        // 对新创建的缓存进行性能分析
        profileCachedGraph(entry);
      }
    });
    // 返回查找到的或者新创建的缓存图形
    return cachedGraph;
  }

  template<typename T>
  inline T* CreateCachedGraphAs(const std::string& key, CreateCachedGraphBlock createCacheBlock) {
    // 使用CreateCachedGraph函数创建缓存图形，并将其强制类型转换为指定类型T返回
    return static_cast<T *>(CreateCachedGraph(key, createCacheBlock));
  }

  MPSCachedGraph* LookUp(const std::string& key) const {

    __block MPSCachedGraph* cachedGraph = nullptr;

    // 计算输入键的哈希值
    MPSCacheKey hash = std::hash<std::string>{}(key);

    dispatch_sync(serialQueue_, ^() {
      // 在串行队列中查找缓存条目
      if (cache_.count(hash) != 0) {
        // 如果缓存中已存在对应哈希值的条目，则获取该条目的引用
        auto& entry = cache_.at(hash);
        // 断言检查，用于调试，确保键值与条目的键值相符
        TORCH_INTERNAL_ASSERT_DEBUG_ONLY(key == entry.key_, "Key collision in the MPS cached graph!\n");
        // 将缓存的图形赋值给cachedGraph
        cachedGraph = entry.cachedGraph_;
        // 对缓存的图形进行性能分析
        profileCachedGraph(entry);
      }
    });
    // 返回查找到的缓存图形，如果未找到则为nullptr
    return cachedGraph;
  }

  template<typename T>
  inline T* LookUpAs(const std::string& key) const {
    // 使用LookUp函数查找缓存图形，并将其强制类型转换为指定类型T返回
    return static_cast<T *>(LookUp(key));
  }

 private:
  MPSGraphCache() {
    // 创建一个串行调度队列，用于操作缓存
    serialQueue_ = dispatch_queue_create("cache queue", DISPATCH_QUEUE_SERIAL);
  }
  // 通过CacheEntry对象进行缓存图形的性能分析，该函数在OperationUtils.mm中定义，以避免在头文件中包含MPSProfiler.h
  void profileCachedGraph(const CacheEntry& cacheEntry) const;

  static MPSGraphCache* _instance_cache;
  // 使用无序map来存储缓存数据，键为MPSCacheKey类型，值为CacheEntry对象
  std::unordered_map<MPSCacheKey, CacheEntry> cache_;
  // 串行调度队列，用于控制对缓存的访问
  dispatch_queue_t serialQueue_ = nullptr;
};

/**
 * Common template for creating a graph with a specified cache if missing.
 * @tparam T Type of graph object to create.
 * @param key Key to look up or create the graph in the cache.
 * @param instantiate Function to instantiate the graph object.
 * @return Pointer to the cached or newly created graph object.
 */
template<typename T>
inline T* LookUpOrCreateCachedGraph(const std::string& key, std::function<void(MPSGraph*, T*)> instantiate) {
  // Retrieve singleton instance of MPSGraphCache
  auto cache_ = MPSGraphCache::getInstance();
  
  // Look up the graph object in the cache and return if found
  if (auto rc  = cache_->LookUpAs<T>(key)) {
    return rc;
  }
  
  // Create and cache a new graph object if not found in the cache
  return cache_->CreateCachedGraphAs<T>(key, ^mps::MPSCachedGraph*() {
    T* newCachedGraph = nil;
    @autoreleasepool {
      // Initialize a new MPSGraph object
      auto mpsGraph = mps::make_mps_graph();
      // Instantiate the new graph object of type T
      newCachedGraph = new T(mpsGraph);
      // Call the provided instantiation function
      instantiate(mpsGraph, newCachedGraph);
    }
    return newCachedGraph;
  });
}

/**
 * Log1p function for MPSGraphTensor.
 */
MPSGraphTensor* log1p(MPSGraph* mpsGraph, MPSGraphTensor* inputTensor);

/**
 * Macro to check if int64 operations are supported on the current platform.
 * @param input_tensor The tensor to check.
 * @param mac_os_13_3_plus Boolean indicating macOS version support.
 * @param op_name Name of the operation being checked.
 */
#define MPS_CHECK_INT64_OP_SUPPORTED(input_tensor, mac_os_13_3_plus, op_name)                                           \
  if (!mac_os_13_3_plus && input_tensor.scalar_type() == kLong) {                                                       \
     TORCH_WARN_ONCE("MPS: no support for int64 for ", op_name,                                                         \
     ", downcasting to a smaller data type (int32/float32). Native support for int64 has been added in macOS 13.3.");   \
  }

/**
 * Computes the distance from the lowest to the highest element offset in the given tensor.
 * @param t The input tensor.
 * @return Size representing the distance in storage.
 */
size_t compute_storage_numel_distance(const at::Tensor& t);

/**
 * Checks whether the tensor is mapped to a contiguous area in the storage.
 * @param t The input tensor to check.
 * @return True if the tensor is dense in storage, false otherwise.
 */
inline bool is_dense_in_storage(const at::Tensor& t) {
  return compute_storage_numel_distance(t) == static_cast<size_t>(t.numel());
}

/**
 * Class representing a Metal shader library.
 */
class MetalShaderLibrary {
public:
  /**
   * Constructor for MetalShaderLibrary with source and optional parameters.
   * @param src Source code of the shader.
   */
  MetalShaderLibrary(const std::string& src): shaderSource(src), nparams(0), compile_options(nullptr){}
  
  /**
   * Constructor for MetalShaderLibrary with source and number of parameters.
   * @param src Source code of the shader.
   * @param nparams_ Number of parameters for the shader.
   */
  MetalShaderLibrary(const std::string& src, unsigned nparams_): shaderSource(src), nparams(nparams_), compile_options(nullptr){}
  
  /**
   * Constructor for MetalShaderLibrary with source, number of parameters, and compile options.
   * @param src Source code of the shader.
   * @param nparams_ Number of parameters for the shader.
   * @param compile_options_ Compilation options for the shader.
   */
  MetalShaderLibrary(const std::string& src, unsigned nparams_, MTLCompileOptions* compile_options_): shaderSource(src), nparams(nparams_), compile_options(compile_options_) {}
  
  /**
   * Deleted copy constructor to prevent copying.
   */
  MetalShaderLibrary(const MetalShaderLibrary&) = delete;
  
  /**
   * Retrieves the Metal compute pipeline state for a given function name.
   * @param fname Name of the function.
   * @return Metal compute pipeline state object.
   */
  inline id<MTLComputePipelineState> getPipelineStateForFunc(const std::string& fname) {
    return getLibraryPipelineState(getLibrary(), fname).first;
  }
  
  /**
   * Retrieves the Metal compute pipeline state for a given function name and parameters.
   * @param fname Name of the function.
   * @param params List of parameters for the function.
   * @return Metal compute pipeline state object.
   */
  id<MTLComputePipelineState> getPipelineStateForFunc(const std::string& fname, const std::initializer_list<std::string>& params) {
    return getLibraryPipelineState(getLibrary(params), fname).first;
  }
  
  /**
   * Retrieves the Metal function for a given function name.
   * @param fname Name of the function.
   * @return Metal function object.
   */
  inline id<MTLFunction> getMTLFunction(const std::string& fname) {
    return getLibraryPipelineState(getLibrary(), fname).second;
  }
  
  /**
   * Retrieves the Metal function for a given function name and parameters.
   * @param fname Name of the function.
   * @param params List of parameters for the function.
   * @return Metal function object.
   */
  id<MTLFunction> getMTLFunction(const std::string& fname, const std::initializer_list<std::string>& params) {
    return getLibraryPipelineState(getLibrary(params), fname).second;
  }
// 获取给定 Metal 库中指定函数名的计算管线状态和函数对象的 pair
private:
  std::pair<id<MTLComputePipelineState>, id<MTLFunction>> getLibraryPipelineState(id<MTLLibrary> lib, const std::string& fname);

  // 获取默认 Metal 库
  id<MTLLibrary> getLibrary();

  // 根据参数列表获取 Metal 库
  id<MTLLibrary> getLibrary(const std::initializer_list<std::string>& params);

  // 编译给定源代码字符串生成 Metal 库
  id<MTLLibrary> compileLibrary(const std::string& src);

  // Metal 着色器源代码字符串
  std::string shaderSource;

  // 参数个数
  unsigned nparams;

  // Metal 编译选项
  MTLCompileOptions* compile_options;

  // 当前 Metal 库对象，默认为空
  id<MTLLibrary> library = nil;

  // 字符串到 Metal 库对象的映射
  std::unordered_map<std::string, id<MTLLibrary>> libMap;

  // 字符串到 Metal 计算管线状态和函数对象 pair 的映射
  std::unordered_map<std::string, std::pair<id<MTLComputePipelineState>, id<MTLFunction>>> cplMap;
};

// 设置 Metal 计算命令编码器的缓冲区
static inline void mtl_setBuffer(id<MTLComputeCommandEncoder> encoder, const Tensor& t, unsigned idx) {
  [encoder setBuffer:getMTLBufferStorage(t)
              offset:t.storage_offset() * t.element_size()
             atIndex:idx];
}

// 在 Metal 计算命令编码器上调度一维作业
static inline void mtl_dispatch1DJob(id<MTLComputeCommandEncoder> encoder,
                                     id<MTLComputePipelineState> cplState,
                                     uint32_t length) {
  // 获取计算管线状态允许的最大线程数
  const uint32_t maxThreadsPerGroup = [cplState maxTotalThreadsPerThreadgroup];
  auto size = MTLSizeMake(length, 1, 1);
  auto threadGroupSize = MTLSizeMake(std::min(maxThreadsPerGroup, length), 1, 1);
  // 调度线程执行作业
  [encoder dispatchThreads:size threadsPerThreadgroup:threadGroupSize];
}

// 生成 Metal 缓冲区数据偏移
id<MTLBuffer> generateKernelDataOffsets(id<MTLComputeCommandEncoder> commandEncoder, const TensorIteratorBase& iter, bool use_64bit_index = false);

// 从单个 Placeholder 生成对应的 NSDictionary
inline NSDictionary* dictionaryFromPlaceholders(Placeholder& p1) {
  return @{ p1.getMPSGraphTensor(): p1.getMPSGraphTensorData() };
}

// 从两个 Placeholder 生成对应的 NSDictionary
inline NSDictionary* dictionaryFromPlaceholders(Placeholder& p1, Placeholder& p2) {
  return @{
          p1.getMPSGraphTensor(): p1.getMPSGraphTensorData(),
          p2.getMPSGraphTensor(): p2.getMPSGraphTensorData(),
   };
}

// 从三个 Placeholder 生成对应的 NSDictionary
inline NSDictionary* dictionaryFromPlaceholders(Placeholder& p1, Placeholder& p2, Placeholder& p3) {
  return @{
          p1.getMPSGraphTensor(): p1.getMPSGraphTensorData(),
          p2.getMPSGraphTensor(): p2.getMPSGraphTensorData(),
          p3.getMPSGraphTensor(): p3.getMPSGraphTensorData(),
   };
}

// 从四个 Placeholder 生成对应的 NSDictionary
inline NSDictionary* dictionaryFromPlaceholders(Placeholder& p1, Placeholder& p2, Placeholder& p3, Placeholder& p4) {
  return @{
          p1.getMPSGraphTensor(): p1.getMPSGraphTensorData(),
          p2.getMPSGraphTensor(): p2.getMPSGraphTensorData(),
          p3.getMPSGraphTensor(): p3.getMPSGraphTensorData(),
          p4.getMPSGraphTensor(): p4.getMPSGraphTensorData(),
   };
}

// 运行 MPS 图计算，并将结果存储在指定的 Placeholder 中
inline void runMPSGraph(MPSStream* stream, MPSGraph* graph, NSDictionary* feeds, Placeholder& result) {
  runMPSGraph(stream, graph, feeds, dictionaryFromPlaceholders(result));
}

// 检查当前系统是否支持复数类型计算
inline bool supportsComplex() {
  return is_macos_13_or_newer(MacOSVersion::MACOS_VER_14_0_PLUS);
}

// 提示 MPS 尚不支持双精度类型，但从 MacOS 14 开始支持 bfloat16 类型
// （此注释位于代码块外，仅为说明性质，并不包括在输出中）
// 检查给定的数据类型是否为支持的浮点类型（kFloat, kHalf, kBFloat16）
inline bool supportedFloatingType(ScalarType dtype) {
  return dtype == kFloat || dtype == kHalf || dtype == kBFloat16;
}

// 检查给定张量的数据类型是否为支持的浮点类型
inline bool supportedFloatingType(const Tensor& t) {
  return supportedFloatingType(t.scalar_type());
}

// 检查给定的数据类型是否为支持的浮点或复数类型，如果是复数类型，还要检查是否支持复数操作
inline bool supportedFloatingOrComplexType(ScalarType dtype) {
  // 如果是复数浮点类型或半精度复数类型，检查是否支持复数操作
  if (dtype == kComplexFloat || dtype == kComplexHalf) {
    return supportsComplex();
  }
  // 否则，检查是否为支持的浮点类型
  return supportedFloatingType(dtype);
}

// 检查给定张量的数据类型是否为支持的浮点或复数类型
inline bool supportedFloatingOrComplexType(const Tensor& t) {
  return supportedFloatingOrComplexType(t.scalar_type());
}

// 检查张量是否需要进行 gather 操作（非连续存储或有偏移）
inline bool needsGather(const Tensor& t) {
  return !t.is_contiguous() || t.storage_offset();
}

} // namespace at::native::mps
```