# `.\pytorch\torch\csrc\lazy\core\lazy_graph_executor.h`

```
#pragma once

#include <c10/util/ArrayRef.h>
#include <torch/csrc/lazy/backend/lowering_context.h>
#include <torch/csrc/lazy/core/cache.h>
#include <torch/csrc/lazy/core/ir_util.h>
#include <torch/csrc/lazy/core/multi_wait.h>
#include <torch/csrc/lazy/core/tensor.h>
#include <torch/csrc/lazy/core/util.h>

namespace torch {
namespace lazy {

class TORCH_API LazyGraphExecutor {
 public:
  // 定义用于存储设备数据信息的结构体，继承自 BackendData::Info
  struct DeviceDataInfo : public BackendData::Info {
    DeviceDataInfo(int64_t tensor_id, bool read_only)
        : tensor_id(tensor_id), read_only(read_only) {}

    int64_t tensor_id = 0;
    ComputationPtr computation;  // 指向计算对象的指针
  };

  using ComputationCache = Cache<hash_t, CachedComputation, HashReducer>;

  // 获取计算缓存对象的指针
  ComputationCache* GetComputationCache();

  // 根据懒惰张量的集合计算图的哈希值
  hash_t GetGraphHash(const std::vector<LazyTensorPtr>& tensors);

 protected:
  // TODO(alanwaketan): Revisit if all of them need to be accessible to
  // derived classes.

  // 定义用于同步张量配置的结构体
  struct SyncTensorsConfig {
    bool force_ltc_data = true;   // 是否强制将数据写入目标张量
    bool sync_ltc_data = true;    // 设置数据时，是否重置张量状态的其他属性
  };

  // 定义同步张量集合的结构体
  struct SyncTensorCollection {
    SyncTensorCollection() : hash(0) {}

    SyncTensorsConfig config;     // 同步张量的配置信息
    std::vector<size_t> indices;  // 张量索引集合
    hash_t hash;                  // 哈希值
    std::vector<ExceptionCleanup> unlocker;  // 异常处理对象集合
    BackendDevice device;         // 后端设备对象
  };

  // 定义后序数据的结构体
  struct PostOrderData {
    std::vector<const Node*> post_order;     // 后序节点集合
    Util::EmissionMap emission_map;          // 发射映射对象
    std::vector<BackendDataPtr> parameters_data;  // 参数数据集合
    std::vector<size_t> parameter_sequence;  // 参数序列
  };

  // 锁定机制:
  // 我们对张量执行两种操作，同步和异步。ApplyPendingGraph() 是同步操作，因为我们需要立即获取设备数据结果。
  // 在同步操作开始之前，需要等待待处理的异步操作完成。同步操作不持有设备锁，因为它们严格按照 PyTorch 的执行顺序进行。
  // SyncTensorsGraph() 是异步的，在安排完异步操作后立即返回。在执行过程中，异步操作将持有所有参与设备的锁定（在大多数情况下，只会有一个设备）。
  // 因为异步操作捕获设备锁，因此同一时间只能执行一个设备上的异步操作。发送数据到设备的张量操作在执行时不需要持有任何设备锁。
  // 只有使用设备数据的操作（计算和从服务器传输）需要等待异步操作完成（屏障）。

  class DeviceLocker {
   public:
    // 构造函数，接受一个 BackendDevice 对象并移动到成员变量 device_ 中
    explicit DeviceLocker(BackendDevice device) : device_(std::move(device)) {}

    // 返回当前设备的常量引用
    const BackendDevice& device() const {
      return device_;
    }

    // 锁定设备的方法声明
    void Lock();
    
    // 解锁设备的方法声明，接受异常指针作为参数
    void Unlock(std::exception_ptr exptr);
    
    // 执行同步屏障的方法声明
    void Barrier();

   private:
    // 检查重置异常的私有方法声明
    void CheckResetException();

    // 存储设备对象的成员变量
    BackendDevice device_;
    
    // 用于互斥访问的互斥量
    std::mutex mutex_;
    
    // 条件变量，用于在锁定时等待和唤醒
    std::condition_variable cv_;
    
    // 表示设备当前是否被锁定的布尔标志
    bool locked_ = false;
    
    // 异常指针，用于存储可能在解锁时抛出的异常
    std::exception_ptr exptr_;
  };

  // 设备锁管理类
  class DeviceLockerArena {
   public:
    // 获取单例实例的静态方法声明
    static DeviceLockerArena* Get();

    // 根据设备获取设备锁的方法声明，返回一个共享指针
    std::shared_ptr<DeviceLocker> GetLocker(const BackendDevice& device);

    // 对一组设备进行锁定，返回异常清理器的向量
    std::vector<ExceptionCleanup> LockDevices(
        const std::set<BackendDevice>& devices);

   private:
    // 对单个设备进行锁定的私有方法声明，返回异常清理器
    ExceptionCleanup LockDevice(const BackendDevice& device);

    // 用于互斥访问的互斥量
    std::mutex mutex_;
    
    // 映射，将设备与设备锁的共享指针关联起来
    std::map<BackendDevice, std::shared_ptr<DeviceLocker>> lockers_;
  };

  // 数据缓存管理类
  class DataCacheArena {
   public:
    // 获取单例实例的静态方法声明
    static DataCacheArena* Get();

    // 根据张量和设备获取后端数据指针的方法声明
    BackendDataPtr GetDeviceData(
        const at::Tensor& tensor,
        const BackendDevice& device);

    // 根据标量值、标量类型和设备获取后端数据指针的方法声明
    BackendDataPtr GetDeviceData(
        const at::Scalar& value,
        at::ScalarType scalar_type,
        const BackendDevice& device);

   private:
    // 用于哈希张量的哈希器结构声明
    struct TensorHasher {
      size_t operator()(const at::Tensor& tensor) const;
    };
    
    // 用于比较张量的比较器结构声明
    struct TensorComparer {
      bool operator()(const at::Tensor& tensor1, const at::Tensor& tensor2)
          const;
    };

    // 构造函数，接受最大缓存大小作为参数
    explicit DataCacheArena(size_t max_cache_size);

    // 获取数据缓存的方法声明，根据设备返回指向缓存的指针
    DataCache* GetDataCache(const BackendDevice& device);

    // 最大缓存大小的成员变量
    size_t max_cache_size_ = 0;
    
    // 用于互斥访问的互斥量
    std::mutex mutex_;
    
    // 映射，将设备与数据缓存的唯一指针关联起来
    std::map<BackendDevice, std::unique_ptr<DataCache>> device_caches_;
  };

  // 设备上下文管理类
  class DeviceContextArena {
   protected:
    // 设备上下文结构声明
    struct DeviceContext {
      std::mutex lock;  // 用于互斥访问的互斥量
      std::map<int64_t, std::weak_ptr<LazyTensor::Data>> tensors_data;  // 弱指针映射，用于存储懒惰张量的数据
      uint64_t seed = 101;  // 种子值
      uint64_t running_seed = 101;  // 运行时种子值
      Value seed_ir_value;  // IR 值
    };

   public:
    // 获取单例实例的静态方法声明
    static DeviceContextArena* Get();
    
    // 虚析构函数声明
    virtual ~DeviceContextArena() = default;

    // 注册张量的方法声明，接受张量数据的共享指针作为参数
    void RegisterTensor(std::shared_ptr<LazyTensor::Data> data);
    
    // 注销张量的方法声明，接受张量数据的指针作为参数
    void UnregisterTensor(LazyTensor::Data* data);

    // 获取活跃张量的方法声明，根据设备返回懒惰张量指针的向量
    std::vector<LazyTensorPtr> GetLiveTensors(const BackendDevice* device);

    // 获取随机数生成器种子的方法声明，根据设备返回值对象的 IR
    virtual Value GetRngSeed(const BackendDevice& device);
    // 注释结束
    // 获取正在运行的种子值，根据给定的设备
    uint64_t GetRunningSeed(const BackendDevice& device);
    
    // 设置随机数种子，根据给定的设备和种子值
    void SetRngSeed(const BackendDevice& device, uint64_t seed);
    
    // 标记当前步骤在给定的设备上进行
    void MarkStep(const BackendDevice& device);
    
    // 返回所有活跃设备的向量
    std::vector<BackendDevice> GetActiveDevices();
    
    protected:
    // 获取特定设备上的设备上下文
    DeviceContext* GetDeviceContext(const BackendDevice& device);
    
    // 遍历所有设备上下文，应用给定函数
    void ForAllDeviceContexts(
        const std::function<void(DeviceContext*)>& fn,
        const BackendDevice* device);
    
    // 通过重写允许派生类使用其自己的类型转换，将标量值转换为 IR 中的值
    virtual Value IrValueFromScalar(
        const at::Scalar& value,
        at::ScalarType scalar_type,
        const BackendDevice& device);
    
    private:
    // 返回所有设备上下文的向量
    std::vector<DeviceContext*> GetAllDeviceContexts();
    
    // 用于同步的互斥量
    std::mutex lock_;
    
    // 映射每个后端设备到其设备上下文的映射表
    std::map<BackendDevice, DeviceContext*> device_contexts_;
    };
    
    // 异步结构体，用于异步操作的管理
    struct Async {
    // 构造函数，初始化异步操作所需的参数
    Async(
        SyncTensorCollection* coll,
        std::vector<BackendDataPtr> parameters_data,
        std::vector<BackendDataPtr> tensors_data,
        ComputationCache::TypePtr cached_computation);
    
    // 默认虚析构函数
    virtual ~Async() = default;
    
    // 等待异步操作完成
    void Wait();
    
    // 多等待对象，用于等待多个异步操作完成
    MultiWait mwait;
    
    // 索引向量，记录异步操作的索引
    std::vector<size_t> indices;
    
    // 异常清理对象，用于异步操作异常时的清理
    std::vector<ExceptionCleanup> unlocker;
    
    // 参数数据指针向量，存储异步操作涉及的参数数据
    std::vector<BackendDataPtr> parameters_data;
    
    // 后端设备对象，指示异步操作所在的设备
    BackendDevice device;
    
    // 缓存的计算对象指针，用于异步操作的缓存计算结果
    ComputationCache::TypePtr cached_computation;
    
    // 张量数据指针向量，存储异步操作涉及的张量数据
    std::vector<BackendDataPtr> tensors_data;
    };
    
    // 重置修剪计数器的方法，声明为常量方法
    void ResetTrimCounter() const;
    
    // 等待 SyncTensorCollection 的设备屏障，并获取锁
    virtual void TensorCollectionBarrier(SyncTensorCollection* coll);
    
    // 可重写的方法，用于插入自定义的分析器
    virtual PostOrderData RunPostOrder(
        const std::vector<Value>& ir_values,
        SyncTensorCollection* coll);
    
    private:
    // 编译结果结构体，记录编译的结果信息
    struct CompilationResult {
    // 后端设备对象，指示编译操作所在的设备
    BackendDevice device;
    
    // 已发出的节点数量
    size_t emitted_nodes = 0;
    
    // 计算对象指针，存储编译操作的计算结果
    ComputationPtr computation;
    // 存储后端数据的指针向量
    std::vector<BackendDataPtr> parameters_data;
  };

  // 判断是否应该同步张量
  virtual bool ShouldSyncTensor(const LazyTensorPtr& tensor) const;

  // 收集需要同步的张量信息
  SyncTensorCollection CollectSyncTensors(
      const std::vector<LazyTensorPtr>& tensors,
      const SyncTensorsConfig& config);

  // 收集根张量
  std::vector<Value> CollectRoots(
      const std::vector<LazyTensorPtr>& tensors,
      c10::ArrayRef<size_t> indices);

  // 设置张量数据
  std::vector<BackendDataPtr> SetTensorData(
      std::vector<LazyTensorPtr>* tensors,
      const SyncTensorsConfig& config,
      c10::ArrayRef<size_t> indices,
      const std::vector<torch::lazy::BackendDataPtr>& tensor_data_vec);

  // 提取IR并准备张量数据
  void ExtractIRAndPrepareTensorData(
      std::vector<LazyTensorPtr>* tensors,
      const SyncTensorsConfig& config,
      c10::ArrayRef<size_t> indices,
      std::vector<Value>& ir_values,
      std::vector<BackendDataPtr>& tensor_data_vec);

  // 尝试运行缓存同步操作
  std::shared_ptr<Async> TryRunCachedSync(
      std::vector<LazyTensorPtr>* tensors,
      SyncTensorCollection* coll,
      PostOrderData* po_data,
      const std::vector<BackendDataPtr>& tensor_data_vec);

  // 编译操作
  CompilationResult Compile(
      const std::vector<LazyTensorPtr>& tensors,
      c10::ArrayRef<std::string> devices,
      const SyncTensorCollection& coll,
      PostOrderData* po_data,
      const std::vector<Value>& ir_values);

  // 查找缓存编译的类型指针
  ComputationCache::TypePtr LookupCachedCompile(const hash_t& hash);

  // 内部同步张量图的实现
  std::shared_ptr<Async> SyncTensorsGraphInternal(
      std::vector<LazyTensorPtr>* tensors,
      c10::ArrayRef<std::string> devices,
      const SyncTensorsConfig& config);

  // 调度同步张量图的执行，后台异步操作将持有设备锁
  std::shared_ptr<Async> ScheduleSyncTensorsGraph(
      SyncTensorCollection* coll,
      std::vector<BackendDataPtr> parameters_data,
      std::vector<BackendDataPtr> tensors_data,
      ComputationCache::TypePtr cached_computation);

  // 调度同步张量图的执行，后台异步操作将持有设备锁
  std::shared_ptr<Async> ScheduleSyncTensorsGraph(
      std::vector<LazyTensorPtr>* tensors,
      SyncTensorCollection* coll,
      std::vector<BackendDataPtr> parameters_data,
      ComputationCache::TypePtr cached_computation,
      const std::vector<BackendDataPtr>& tensor_data_vec);

  // 获取融合后的张量
  std::vector<at::Tensor> GetTensorsFused(std::vector<LazyTensorPtr>* tensors);

  // 获取张量数据
  std::vector<at::Tensor> FetchTensors(
      std::vector<LazyTensorPtr>* tensors,
      c10::ArrayRef<BackendDataPtr> tensors_data,
      const std::vector<size_t>* indices);

  // 收集所有输入张量的设备数据，在异步操作后
  std::vector<BackendDataPtr> GatherTensorsData(
      const std::vector<LazyTensorPtr>& tensors,
      c10::ArrayRef<size_t> indices,
      c10::ArrayRef<BackendDataPtr> tensors_data);
};

} // namespace lazy
} // namespace torch
```