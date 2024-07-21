# `.\pytorch\torch\csrc\lazy\core\lazy_graph_executor.cpp`

```
#include <torch/csrc/lazy/core/lazy_graph_executor.h>

#include <ATen/ScalarOps.h>
#include <c10/util/Logging.h>
#include <c10/util/irange.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/lazy/core/config.h>
#include <torch/csrc/lazy/core/internal_ops/ltc_ops.h>
#include <torch/csrc/lazy/core/ir_dump_util.h>
#include <torch/csrc/lazy/core/ir_util.h>
#include <torch/csrc/lazy/core/tensor_util.h>
#include <torch/csrc/lazy/core/unique.h>

#include <torch/csrc/lazy/core/debug_util.h>
#include <torch/csrc/lazy/core/ir_builder.h>
#include <torch/csrc/lazy/core/metrics.h>
#include <torch/csrc/lazy/core/ops/arithmetic_ir_ops.h>
#include <torch/csrc/lazy/core/thread_pool.h>

#include <ATen/ScalarOps.h>

namespace torch {
namespace lazy {
namespace {

// Thread local data structure for managing per-thread state
struct TlsData {
  void Reset() {
    trim_counter = 0;
  }

  size_t trim_counter = 0;
};

// Thread local storage for TlsData
thread_local TlsData g_tls_data;

// Function to compare two tensors for equality in scalar type, sizes, and data
bool TensorCompare(const at::Tensor& t1, const at::Tensor& t2) {
  if (t1.scalar_type() != t2.scalar_type() || t1.sizes() != t2.sizes()) {
    return false;
  }
  // PyTorch currently has an issue comparing tensors with NaN values,
  // so we perform a memory compare until this is fixed.
  at::Tensor contiguous_t1 = t1.contiguous();
  at::Tensor contiguous_t2 = t2.contiguous();
  return std::memcmp(
             contiguous_t1.data_ptr(),
             contiguous_t2.data_ptr(),
             contiguous_t1.numel() * contiguous_t1.itemsize()) == 0;
}

// Function to check if any tensor in a list has an associated IR (Intermediate Representation)
bool TensorsHaveIR(const std::vector<LazyTensorPtr>& tensors) {
  for (const auto& tensor : tensors) {
    if (tensor->CurrentDataHandle() || tensor->CurrentIrValue()) {
      return true;
    }
  }
  return false;
}

// Atomic pointer to the LazyGraphExecutor instance
std::atomic<LazyGraphExecutor*> lazy_graph_executor_registry;

} // namespace

// Get the singleton instance of DeviceContextArena for lazy graph execution
auto LazyGraphExecutor::DeviceContextArena::Get()
    -> LazyGraphExecutor::DeviceContextArena* {
  static DeviceContextArena* arena = new DeviceContextArena();
  return arena;
}

// Register tensor data into the device context arena
void LazyGraphExecutor::DeviceContextArena::RegisterTensor(
    std::shared_ptr<LazyTensor::Data> data) {
  DeviceContext* devctx = GetDeviceContext(data->device);
  std::lock_guard<std::mutex> lock(devctx->lock);
  devctx->tensors_data.emplace(data->unique_id, data);
}

// Unregister tensor data from the device context arena
void LazyGraphExecutor::DeviceContextArena::UnregisterTensor(
    LazyTensor::Data* data) {
  DeviceContext* devctx = GetDeviceContext(data->device);
  std::lock_guard<std::mutex> lock(devctx->lock);
  devctx->tensors_data.erase(data->unique_id);
}

// Get live tensors associated with a specific backend device
std::vector<LazyTensorPtr> LazyGraphExecutor::DeviceContextArena::
    GetLiveTensors(const BackendDevice* device) {
  std::vector<LazyTensorPtr> tensors;
  auto fn = [&](DeviceContext* devctx) {
    std::lock_guard<std::mutex> lock(devctx->lock);



// Continued from the previous annotation

    // Retrieve all live tensors associated with the given device
    for (const auto& kv : devctx->tensors_data) {
      if (auto ptr = kv.second.lock()) {
        tensors.push_back(ptr);
      }
    }
  };

  // Execute the lambda function fn on the device context of the given device
  ExecOnDeviceContext(device, fn);
  
  // Return the vector of live tensors
  return tensors;
}

} // namespace lazy
} // namespace torch
    for (auto& uid_wptr : devctx->tensors_data) {
      // 遍历 devctx 指针指向的 tensors_data 的每对键值对，uid_wptr 是每个键值对的引用
      std::shared_ptr<LazyTensor::Data> data = uid_wptr.second.lock();
      // 使用 uid_wptr 的第二个元素（即数据指针）创建一个 shared_ptr，data 指向这个数据
      if (data != nullptr) {
        // 如果 data 不为空
        tensors.push_back(LazyTensor::Create(std::move(data)));
        // 调用 LazyTensor 的静态方法 Create，使用 data 移动语义创建 LazyTensor 对象，并将其添加到 tensors 容器中
      }
    }
  };
  // 对于每个设备上下文调用 fn 函数，将其包含的 tensors_data 转换为 LazyTensor 对象，返回结果放入 tensors
  ForAllDeviceContexts(fn, device);
  // 调用 ForAllDeviceContexts 函数处理所有设备上下文，执行之前定义的 lambda 表达式 fn
  return tensors;
  // 返回包含 LazyTensor 对象的容器 tensors
}

// 获取设备上下文的随机种子值
Value LazyGraphExecutor::DeviceContextArena::GetRngSeed(
    const BackendDevice& device) {
  // 静态常量，用于定义种子的数据类型和增量
  static const at::ScalarType kSeedType = at::ScalarType::Long;
  static const uint64_t kSeedMul = 214013;
  static const uint64_t kSeedAdd = 2531011;
  
  // 获取设备的上下文对象
  DeviceContext* devctx = GetDeviceContext(device);
  
  // 使用互斥锁保护，确保线程安全
  std::lock_guard<std::mutex> lock(devctx->lock);
  
  // 如果种子的 IR 值尚未计算，则进行计算
  if (!devctx->seed_ir_value) {
    devctx->seed_ir_value =
        IrValueFromScalar(MakeIntScalar(devctx->seed), kSeedType, device);
  }
  
  // 更新运行中的种子值，以保持其为标量，避免执行图形操作。
  devctx->running_seed = kSeedAdd + kSeedMul * devctx->running_seed;
  
  // 通过组合根种子创建新的种子，以避免创建过多的计算参数可能导致设备容量溢出。
  Value k = MakeScalar(MakeIntScalar(kSeedMul), kSeedType);
  Value b = MakeScalar(MakeIntScalar(kSeedAdd), kSeedType);
  devctx->seed_ir_value = b + k * devctx->seed_ir_value;
  
  // 返回计算得到的 IR 值作为种子
  return devctx->seed_ir_value;
}

// 获取设备上下文的当前运行种子值
uint64_t LazyGraphExecutor::DeviceContextArena::GetRunningSeed(
    const BackendDevice& device) {
  // 获取设备的上下文对象
  DeviceContext* devctx = GetDeviceContext(device);
  
  // 使用互斥锁保护，确保线程安全
  std::lock_guard<std::mutex> lock(devctx->lock);
  
  // 返回当前运行的种子值
  return devctx->running_seed;
}

// 设置设备上下文的种子值
void LazyGraphExecutor::DeviceContextArena::SetRngSeed(
    const BackendDevice& device,
    uint64_t seed) {
  // 获取设备的上下文对象
  DeviceContext* devctx = GetDeviceContext(device);
  
  // 使用互斥锁保护，确保线程安全
  std::lock_guard<std::mutex> lock(devctx->lock);
  
  // 设置种子值并更新运行中的种子值
  devctx->seed = seed;
  devctx->running_seed = devctx->seed;
  devctx->seed_ir_value = Value();  // 清空 IR 值
}

// 标记设备上下文的一步操作
void LazyGraphExecutor::DeviceContextArena::MarkStep(
    const BackendDevice& device) {
  // 获取设备的上下文对象
  DeviceContext* devctx = GetDeviceContext(device);
  
  // 使用互斥锁保护，确保线程安全
  std::lock_guard<std::mutex> lock(devctx->lock);
  
  // 标记一步操作并更新运行中的种子值
  devctx->seed = 1012031 + devctx->seed * 7012063;
  devctx->running_seed = devctx->seed;
  devctx->seed_ir_value = Value();  // 清空 IR 值
}

// 获取所有活跃设备的列表
std::vector<BackendDevice> LazyGraphExecutor::DeviceContextArena::
    GetActiveDevices() {
  // 存储活跃设备的向量
  std::vector<BackendDevice> active_devices;
  
  // 使用互斥锁保护，确保线程安全
  std::lock_guard<std::mutex> lock(lock_);
  
  // 预留足够的空间以避免动态扩展
  active_devices.reserve(device_contexts_.size());
  
  // 将每个设备上下文对应的设备添加到活跃设备列表中
  for (auto& device_contexts : device_contexts_) {
    active_devices.push_back(device_contexts.first);
  }
  
  // 返回活跃设备列表
  return active_devices;
}

// 获取所有设备上下文的列表
auto LazyGraphExecutor::DeviceContextArena::GetAllDeviceContexts()
    -> std::vector<DeviceContext*> {
  // 存储所有设备上下文的指针向量
  std::vector<DeviceContext*> all_device_contexts;
  
  // 使用互斥锁保护，确保线程安全
  std::lock_guard<std::mutex> lock(lock_);
  
  // 预留足够的空间以避免动态扩展
  all_device_contexts.reserve(device_contexts_.size());
  
  // 将每个设备上下文的指针添加到所有设备上下文列表中
  for (auto& device_contexts : device_contexts_) {
    all_device_contexts.push_back(device_contexts.second);
  }
  
  // 返回所有设备上下文列表
  return all_device_contexts;
}

// 针对所有设备上下文执行指定函数
void LazyGraphExecutor::DeviceContextArena::ForAllDeviceContexts(
    const std::function<void(DeviceContext*)>& fn,
    const BackendDevice* device) {
  // 如果设备为空，则对所有设备上下文执行函数
  if (device == nullptr) {
    for (auto devctx : GetAllDeviceContexts()) {
      fn(devctx);
    }
  } else {  // 否则只对指定设备上下文执行函数
    fn(GetDeviceContext(*device));
  }
}
// 获取特定设备的设备上下文对象，如果不存在则创建新的，并返回该设备上下文指针
auto LazyGraphExecutor::DeviceContextArena::GetDeviceContext(
    const BackendDevice& device) -> DeviceContext* {
  // 使用互斥锁保护临界区
  std::lock_guard<std::mutex> lock(lock_);
  // 在设备上下文映射中查找给定设备
  auto it = device_contexts_.find(device);
  // 如果设备上下文映射中不存在该设备，则创建新的设备上下文并插入映射中
  if (it == device_contexts_.end()) {
    it = device_contexts_.emplace(device, new DeviceContext()).first;
  }
  // 返回找到或新创建的设备上下文指针
  return it->second;
}

// 从标量值创建对应的 IR 值
Value LazyGraphExecutor::DeviceContextArena::IrValueFromScalar(
    const at::Scalar& value,
    at::ScalarType scalar_type,
    const BackendDevice& device) {
  // 根据标量值和标量类型创建张量
  at::Tensor tensor = at::scalar_tensor(value, at::TensorOptions(scalar_type));
  // 将张量转换为特定设备的后端数据指针
  BackendDataPtr device_data = TensorToDataHandle(tensor, device);
  // 将后端数据指针包装成设备数据对象，并返回
  return MakeDeviceData(std::move(device_data));
}

// 加锁当前设备
void LazyGraphExecutor::DeviceLocker::Lock() {
  // 使用互斥锁进行唤醒等待，直到不再处于锁定状态
  std::unique_lock<std::mutex> lock(mutex_);
  cv_.wait(lock, [this] { return !locked_; });
  // 检查是否有异常需要重新抛出
  CheckResetException();
  // 将锁定状态置为 true
  locked_ = true;
}

// 解锁当前设备，可选参数异常指针
void LazyGraphExecutor::DeviceLocker::Unlock(std::exception_ptr exptr) {
  // 使用互斥锁进行锁定
  std::lock_guard<std::mutex> lock(mutex_);
  // 将锁定状态置为 false
  locked_ = false;
  // 设置异常指针
  exptr_ = std::move(exptr);
  // 通知所有等待线程解锁
  cv_.notify_all();
}

// 设备屏障，等待设备解锁并通知所有等待线程
void LazyGraphExecutor::DeviceLocker::Barrier() {
  // 使用互斥锁进行唤醒等待，直到不再处于锁定状态
  std::unique_lock<std::mutex> lock(mutex_);
  cv_.wait(lock, [this] { return !locked_; });
  // 通知所有等待线程解锁
  cv_.notify_all();
  // 检查是否有异常需要重新抛出
  CheckResetException();
}

// 检查并重新抛出存储的异常指针
void LazyGraphExecutor::DeviceLocker::CheckResetException() {
  // 获取并移动异常指针
  std::exception_ptr exptr = std::move(exptr_);
  // 清空异常指针
  exptr_ = nullptr;
  // 如果异常指针不为空，则重新抛出异常
  if (exptr != nullptr) {
    std::rethrow_exception(exptr);
  }
}

// 获取设备锁定器的静态实例
auto LazyGraphExecutor::DeviceLockerArena::Get() -> DeviceLockerArena* {
  // 静态变量保证单例模式，返回设备锁定器竞技场的唯一实例
  static DeviceLockerArena* arena = new DeviceLockerArena();
  return arena;
}

// 获取特定设备的设备锁定器，如果不存在则创建新的，并返回其共享指针
auto LazyGraphExecutor::DeviceLockerArena::GetLocker(
    const BackendDevice& device) -> std::shared_ptr<DeviceLocker> {
  // 使用互斥锁保护临界区
  std::lock_guard<std::mutex> lock(mutex_);
  // 在设备锁定器映射中查找给定设备
  auto it = lockers_.find(device);
  // 如果设备锁定器映射中不存在该设备，则创建新的设备锁定器并插入映射中
  if (it == lockers_.end()) {
    it = lockers_.emplace(device, std::make_shared<DeviceLocker>(device)).first;
  }
  // 返回找到或新创建的设备锁定器的共享指针
  return it->second;
}

// 对特定设备执行屏障操作
void LazyGraphExecutor::DeviceLockerArena::DeviceBarrier(
    const BackendDevice& device) {
  // 获取特定设备的设备锁定器，并执行屏障操作
  auto locker = DeviceLockerArena::Get()->GetLocker(device);
  locker->Barrier();
}

// 锁定一组设备，并返回异常清理器的向量
std::vector<ExceptionCleanup> LazyGraphExecutor::DeviceLockerArena::LockDevices(
    const std::set<BackendDevice>& devices) {
  // 创建异常清理器向量，预留足够空间以容纳所有设备
  std::vector<ExceptionCleanup> unlocker;
  unlocker.reserve(devices.size());
  // 遍历设备集合，对每个设备执行设备锁定操作，并将异常清理器添加到向量中
  for (auto& device : devices) {
    unlocker.emplace_back(LockDevice(device));
  }
  // 返回包含所有异常清理器的向量
  return unlocker;
}

// 锁定特定设备，并返回相应的异常清理器
ExceptionCleanup LazyGraphExecutor::DeviceLockerArena::LockDevice(
    const BackendDevice& device) {
  // 记录设备屏障等待消息
  VLOG(4) << "Waiting on device barrier for device " << device << " ...";
  std::shared_ptr<DeviceLocker> locker;
  {
    // 测量设备锁定等待时间
    TORCH_LAZY_TIMED("DeviceLockWait");
    // 获取特定设备的设备锁定器
    locker = DeviceLockerArena::Get()->GetLocker(device);
    // 执行设备屏障操作
    locker->Barrier();
  }
  // 返回包含设备锁定器和设备的异常清理器

  return ExceptionCleanup([locker]() {
    // 解锁设备
    locker->Unlock(std::current_exception());
  });
}
    // 获得对锁对象的独占访问，阻塞直到获得锁
    locker->Lock();
  }
  // 记录日志：等待设备障碍结束，对应设备为 device
  VLOG(4) << "Waiting on device barrier for device " << device << " done!";
  // 返回一个异常清理函数，确保在作用域结束时释放锁
  return torch::lazy::ExceptionCleanup(
      [locker = std::move(locker)](
          torch::lazy::ExceptionCleanup::StatusType status) {
        // 在异常清理函数中调用 Unlock() 来释放锁，传递当前的状态
        locker->Unlock(std::move(status));
      });
}

// 获取数据缓存区域的单例对象，根据标志设定最大缓存大小
auto LazyGraphExecutor::DataCacheArena::Get() -> DataCacheArena* {
  static DataCacheArena* arena =
      new DataCacheArena(FLAGS_torch_lazy_device_data_cache_size);
  return arena;
}

// 数据缓存区域的构造函数，设定最大缓存大小
LazyGraphExecutor::DataCacheArena::DataCacheArena(size_t max_cache_size)
    : max_cache_size_(max_cache_size) {}

// 获取特定设备上张量的后端数据指针
BackendDataPtr LazyGraphExecutor::DataCacheArena::GetDeviceData(
    const at::Tensor& tensor,
    const BackendDevice& device) {
  // 获取设备对应的数据缓存
  DataCacheArena::DataCache* cache = Get()->GetDataCache(device);
  ;
  // 尝试从缓存中获取张量对应的后端数据指针
  BackendDataPtr device_data = cache->Get(tensor);
  // 如果未命中缓存
  if (device_data == nullptr) {
    // 复制张量
    at::Tensor tensor_copy = CopyTensor(tensor);
    // 将复制的张量转换为数据句柄
    device_data = TensorToDataHandle(tensor_copy, device);
    // 将复制的张量及其数据句柄添加到缓存中
    cache->Add(std::move(tensor_copy), device_data);
    // 增加惰性执行计数器，记录缓存未命中事件
    TORCH_LAZY_COUNTER("DeviceDataCacheMiss", 1);
  }
  return device_data;
}

// 获取特定数值在设备上的后端数据指针
BackendDataPtr LazyGraphExecutor::DataCacheArena::GetDeviceData(
    const at::Scalar& value,
    at::ScalarType scalar_type,
    const BackendDevice& device) {
  // 当 at::scalar_tensor 不支持 bfloat16 时的临时解决方案
  at::Tensor t = at::scalar_tensor(
      value,
      at::TensorOptions(
          scalar_type == at::ScalarType::BFloat16 ? at::ScalarType::Float
                                                  : scalar_type));
  // 如果是 bfloat16 类型，则进行额外的类型转换
  if (scalar_type == at::ScalarType::BFloat16) {
    t = t.to(scalar_type);
  }
  // 获取该数值对应的设备数据
  return GetDeviceData(t, device);
}

// 张量的哈希计算器，用于在哈希映射中使用
size_t LazyGraphExecutor::DataCacheArena::TensorHasher::operator()(
    const at::Tensor& tensor) const {
  // 使用自定义的哈希函数组合张量的类型和哈希值
  return HashReduce(
      HashCombine(GetEnumValue(tensor.scalar_type()), TensorHash(tensor)));
}

// 张量的比较器，用于在哈希映射中使用
bool LazyGraphExecutor::DataCacheArena::TensorComparer::operator()(
    const at::Tensor& tensor1,
    const at::Tensor& tensor2) const {
  // 比较两个张量是否相等
  return TensorCompare(tensor1, tensor2);
}

// 获取特定设备上的数据缓存，支持多线程安全
auto LazyGraphExecutor::DataCacheArena::GetDataCache(
    const BackendDevice& device) -> DataCache* {
  // 使用互斥锁保护并发访问
  std::lock_guard<std::mutex> lock(mutex_);
  // 如果启用了设备数据缓存
  if (FLAGS_torch_lazy_enable_device_data_cache) {
    // 查找当前设备的缓存，如果不存在则创建新的缓存
    auto it = device_caches_.find(device);
    if (it == device_caches_.end()) {
      it = device_caches_
               .emplace(device, std::make_unique<DataCache>(max_cache_size_))
               .first;
    }
    return it->second.get();
  } else {
    // 如果禁用了缓存，则始终返回一个空的缓存
    static DataCache s_empty_cache(0);
    return &s_empty_cache;
  }
}

// 注册惰性图执行器
void LazyGraphExecutor::Register(LazyGraphExecutor* executor) {
  lazy_graph_executor_registry.store(executor);
}

// 获取惰性图执行器的单例对象
LazyGraphExecutor* LazyGraphExecutor::Get() {
  auto* executor = lazy_graph_executor_registry.load();
  // 检查惰性图执行器是否已注册
  TORCH_CHECK(executor, "Lazy graph executor not registered.");
  return executor;
}

// 注册惰性张量的数据到设备上下文中
void LazyGraphExecutor::RegisterTensor(std::shared_ptr<LazyTensor::Data> data) {
  DeviceContextArena::Get()->RegisterTensor(data);
  // 增加惰性计数器，记录创建惰性张量的事件
  TORCH_LAZY_COUNTER("CreateLtcTensor", 1);
}
void LazyGraphExecutor::UnregisterTensor(LazyTensor::Data* data) {
  // 调用 DeviceContextArena 的方法，注销指定的 LazyTensor 数据
  DeviceContextArena::Get()->UnregisterTensor(data);
  // 增加 TORCH_LAZY_COUNTER 计数器 "DestroyLtcTensor" 的计数值
  TORCH_LAZY_COUNTER("DestroyLtcTensor", 1);
}

Value LazyGraphExecutor::GetRngSeed(const BackendDevice& device) {
  // 调用 DeviceContextArena 的方法，获取指定设备的随机数种子
  return DeviceContextArena::Get()->GetRngSeed(device);
}

uint64_t LazyGraphExecutor::GetRunningSeed(const BackendDevice& device) {
  // 调用 DeviceContextArena 的方法，获取指定设备的运行时种子
  return DeviceContextArena::Get()->GetRunningSeed(device);
}

void LazyGraphExecutor::SetRngSeed(const BackendDevice& device, uint64_t seed) {
  // 调用 DeviceContextArena 的方法，设置指定设备的随机数种子
  DeviceContextArena::Get()->SetRngSeed(device, seed);
}

void LazyGraphExecutor::DeviceBarrier(const BackendDevice& device) {
  // 调用 DeviceLockerArena 的方法，设定指定设备的设备屏障
  DeviceLockerArena::Get()->DeviceBarrier(device);
}

BackendDataPtr LazyGraphExecutor::GetDeviceData(
    const at::Tensor& tensor,
    const BackendDevice& device) {
  // 调用 DataCacheArena 的方法，获取特定张量在指定设备上的后端数据指针
  return DataCacheArena::Get()->GetDeviceData(tensor, device);
}

BackendDataPtr LazyGraphExecutor::GetDeviceData(
    const at::Scalar& value,
    at::ScalarType scalar_type,
    const BackendDevice& device) {
  // 调用 DataCacheArena 的方法，获取特定标量在指定设备上的后端数据指针
  return DataCacheArena::Get()->GetDeviceData(value, scalar_type, device);
}

std::vector<LazyTensorPtr> LazyGraphExecutor::GetLiveTensors(
    const BackendDevice* device) {
  // 调用 DeviceContextArena 的方法，获取指定设备上的所有活跃 LazyTensor 指针
  return DeviceContextArena::Get()->GetLiveTensors(device);
}

void LazyGraphExecutor::SyncLiveTensorsGraph(
    const BackendDevice* device,
    c10::ArrayRef<std::string> devices,
    bool wait) {
  // 获取指定设备上的所有活跃 LazyTensor 指针
  auto tensors = GetLiveTensors(device);
  // 记录日志，显示活跃张量数量及设备列表
  VLOG(4) << tensors.size() << " live tensors: devices=("
          << c10::Join(", ", devices) << ")";
  // 同步张量图，包括 LTC 数据，等待完成
  SyncTensorsGraph(&tensors, devices, wait, /*sync_ltc_data=*/true);
}

void LazyGraphExecutor::SyncTensorsGraph(
    std::vector<LazyTensorPtr>* tensors,
    c10::ArrayRef<std::string> devices,
    bool wait,
    bool sync_ltc_data) {
  // 记录日志，显示尝试同步张量的数量
  VLOG(4) << "Trying to sync the value of " << tensors->size() << " tensor(s)";
  // 配置同步张量的选项
  SyncTensorsConfig config;
  config.sync_ltc_data = sync_ltc_data;

  // 异步执行张量图的同步操作
  auto async = SyncTensorsGraphInternal(tensors, devices, config);
  // 如果开启线程池，并且需要等待，并且异步操作不为空，则等待其完成
  if (FLAGS_torch_lazy_use_thread_pool && wait && async != nullptr) {
    async->mwait.Wait();
  }
}

void LazyGraphExecutor::MarkStep(const BackendDevice& device) {
  // 增加 TORCH_LAZY_COUNTER 计数器 "MarkStep" 的计数值
  TORCH_LAZY_COUNTER("MarkStep", 1);
  // 调用 DeviceContextArena 的方法，标记指定设备的步骤
  DeviceContextArena::Get()->MarkStep(device);
  // 重置作用域推送器的作用域
  ScopePusher::ResetScopes();
  // 重置修剪计数器
  ResetTrimCounter();
  // 将 TrieCache 的当前指针移回其根部
  TrieCache::Get()->ResetCurrent();
}

void LazyGraphExecutor::WaitDeviceOps(c10::ArrayRef<BackendDevice> devices) {
  // 创建一个等待设备集合
  std::set<BackendDevice> wait_devices;
  // 如果设备列表非空，则依次插入到等待设备集合中
  if (!devices.empty()) {
    for (auto& device : devices) {
      wait_devices.insert(device);
    }
  } else {
    // 否则，遍历所有活跃设备，并插入到等待设备集合中
    for (auto& device_str : DeviceContextArena::Get()->GetActiveDevices()) {
      // TODO: Remove the last use of Device(const std::string& device_spec).
      wait_devices.insert(BackendDevice(device_str));
    }
  }
    }
  }
  // The LockDevices() API returns a vector of
  // ExceptionCleanup object, which is going to be freed
  // immediately, turning this operation into a lock barrier.
  // NOLINTNEXTLINE
  // 获取 DeviceLockerArena 的单例，并调用其 LockDevices() 方法对设备进行锁定
  DeviceLockerArena::Get()->LockDevices(wait_devices);
}

// 返回给定 LazyTensorPtr 向量中张量的值
std::vector<at::Tensor> LazyGraphExecutor::GetTensors(
    std::vector<LazyTensorPtr>* tensors) {
  // 记录尝试获取张量值的日志信息
  VLOG(4) << "Trying to get the value of " << tensors->size() << " tensor(s)";
  // 调用 GetTensorsFused 函数获取张量的值并返回
  return GetTensorsFused(tensors);
}

// 重置 trim 计数器
void LazyGraphExecutor::ResetTrimCounter() const {
  // 调用 g_tls_data 的 Reset 方法
  g_tls_data.Reset();
}

// 增加 trim 计数器并返回新值
size_t LazyGraphExecutor::IncTrimCounter() const {
  // 递增并返回 g_tls_data 的 trim_counter 成员变量
  return ++g_tls_data.trim_counter;
}

// 将给定张量的当前 IR 值转储为后端计算字符串
std::string LazyGraphExecutor::DumpBackendComputation(
    const std::vector<LazyTensorPtr>& tensors) {
  // 创建存储 IR 值的向量
  std::vector<Value> ir_values;
  // 遍历每个 LazyTensorPtr 并获取其当前 IR 值
  for (auto& tensor : tensors) {
    Value ir_value = tensor->CurrentIrValue();
    // 如果 IR 值有效，则加入到 ir_values 中
    if (ir_value) {
      ir_values.push_back(std::move(ir_value));
    }
  }
  // 如果 ir_values 非空，则将其转储为后端计算字符串；否则返回空字符串
  return !ir_values.empty() ? DumpUtil::ToBackend(ir_values, BackendDevice())
                            : std::string();
}

// 获取标量值的设备数据 IR 值
Value LazyGraphExecutor::GetDeviceDataIrValue(
    const at::Scalar& value,
    c10::ScalarType type,
    const BackendDevice& device) {
  // 获取与给定值、类型、设备对应的设备数据指针
  BackendDataPtr data = GetDeviceData(value, type, device);
  // 设置数据的信息为只读状态的设备数据信息
  data->SetInfo(std::make_shared<DeviceDataInfo>(
      /*tensor_id=*/-1, /*read_only=*/true));
  // 生成并返回包含给定数据的设备数据 IR 值
  return MakeDeviceData(std::move(data));
}

// 从代码生成器获取标量值的 IR 值
Value LazyGraphExecutor::GetIrValueForScalarFromCodegen(
    const at::Scalar& value,
    const BackendDevice& device) {
  // 如果是特殊标量值，则直接创建标量 IR 值返回
  if (IsSpecialScalar(value)) {
    return MakeScalar(value, value.type());
  }
  // 否则，获取与给定值、类型、设备对应的设备数据，并生成包含其 IR 值的设备数据返回
  auto data = GetDeviceData(value, value.type(), device);
  data->SetInfo(
      std::make_shared<DeviceDataInfo>(/*tensor_id=*/-1, /*read_only=*/true));
  return MakeDeviceData(std::move(data));
}

// 获取标量值的 IR 值
Value LazyGraphExecutor::GetIrValueForScalar(
    const at::Scalar& value,
    c10::ScalarType type,
    const BackendDevice& device) {
  // 如果是特殊标量值，则直接创建标量 IR 值返回
  if (IsSpecialScalar(value)) {
    return MakeScalar(value, type);
  }
  // 否则，调用 GetDeviceDataIrValue 获取设备数据的 IR 值并返回
  return GetDeviceDataIrValue(value, type, device);
}

// 获取标量值的 IR 值
Value LazyGraphExecutor::GetIrValueForScalar(
    const at::Scalar& value,
    const BackendDevice& device) {
  // 调用具有默认类型参数的 GetIrValueForScalar 函数获取标量值的 IR 值并返回
  return GetIrValueForScalar(value, value.type(), device);
}

// 获取扩展标量值的 IR 值
Value LazyGraphExecutor::GetIrValueForExpandedScalar(
    const at::Scalar& value,
    const Shape& shape,
    const BackendDevice& device) {
  // 获取形状的维度和类型
  c10::ArrayRef<int64_t> dimensions = shape.sizes();
  auto type = shape.scalar_type();
  // 获取标量值的 IR 值
  Value ir_value = GetIrValueForScalar(value, type, device);
  // 如果维度非空，则调用 MakeExpand 函数扩展 IR 值
  if (!dimensions.empty()) {
    ir_value = MakeExpand(
        ir_value,
        dimensions.vec(),
        /*is_scalar_expand=*/true);
  }
  // 返回 IR 值
  return ir_value;
}

// 异步构造函数，初始化对象成员变量
LazyGraphExecutor::Async::Async(
    SyncTensorCollection* coll,
    std::vector<BackendDataPtr> parameters_data,
    std::vector<BackendDataPtr> tensors_data,
    ComputationCache::TypePtr cached_computation)
    : mwait(1),
      indices(std::move(coll->indices)),
      unlocker(std::move(coll->unlocker)),
      parameters_data(std::move(parameters_data)),
      device(coll->device),
      cached_computation(std::move(cached_computation)),
      tensors_data(std::move(tensors_data)) {}
// 等待所有异步操作完成
void LazyGraphExecutor::Async::Wait() {
  mwait.Wait();
  // 等待 MultiWait::Wait() 完成后，访问其他 Async 成员是安全的

  ExceptionCleanup::StatusType status;
  for (auto& cleanup : unlocker) {
    const ExceptionCleanup::StatusType& cleanup_status = cleanup.GetStatus();
    // 检查每个 cleanup 对象的状态
    if (cleanup_status != nullptr) {
      // 如果当前 status 为空，则将 cleanup 的状态赋给 status
      if (status == nullptr) {
        status = cleanup_status;
      }
      // 清空 cleanup 的状态，避免在下一次设备锁操作中继续传播
      cleanup.SetStatus(nullptr);
    }
  }
  // 如果有异常状态，则重新抛出该异常
  if (status != nullptr) {
    std::rethrow_exception(status);
  }
}

// 判断是否需要同步张量
bool LazyGraphExecutor::ShouldSyncTensor(const LazyTensorPtr& tensor) const {
  return tensor->GetIrValue()->op() != ltc_not_supported;
}

// 收集需要同步的张量集合
LazyGraphExecutor::SyncTensorCollection LazyGraphExecutor::CollectSyncTensors(
    const std::vector<LazyTensorPtr>& tensors,
    const SyncTensorsConfig& config) {
  Unique<BackendDevice> unique_device;
  // 将张量的设备类型设置为唯一的 BackendDevice
  for (const auto& tensor : tensors) {
    unique_device.set(tensor->GetDevice());
  }
  SyncTensorCollection coll;
  if (!unique_device) {
    return coll;
  }
  if (!config.force_ltc_data && !TensorsHaveIR(tensors)) {
    return coll;
  }

  std::vector<at::Tensor> at_tensors;
  std::vector<BackendDevice> devices;
  std::vector<size_t> at_tensor_index;
  std::unordered_set<int64_t> tensor_ids;
  // force_ltc_data 控制别名编译，因此强制打开或关闭 force_ltc_data 不应该哈希相同的图
  coll.hash = MHash(config.force_ltc_data);
  coll.config = config;
  coll.device = *unique_device;
  coll.indices.reserve(tensors.size());

  for (const auto i : c10::irange(tensors.size())) {
    // 如果张量 ID 是唯一的并且当前数据处理句柄为空
    if (tensor_ids.insert(tensors[i]->GetUniqueId()).second &&
        tensors[i]->CurrentDataHandle() == nullptr) {
      Value ir_value = tensors[i]->CurrentIrValue();
      if (ir_value) {
        if (ShouldSyncTensor(tensors[i])) {
          TORCH_LAZY_COUNTER("SyncedTensorsWithIR", 1);
          // 只添加需要同步的张量
          coll.hash = HashCombine(coll.hash, ir_value.hash());
          coll.indices.push_back(i);
        }
      } else if (config.force_ltc_data) {
        // 张量只有 at::Tensor 数据，需要将其排队进行设备上传
        std::optional<at::Tensor> tensor_data = tensors[i]->CurrentTensorData();
        TORCH_CHECK(tensor_data);
        at_tensors.push_back(*tensor_data);
        devices.push_back(tensors[i]->GetDevice());
        at_tensor_index.push_back(i);
      }
    }
  }
  if (!at_tensors.empty()) {
    TORCH_LAZY_COUNTER("SyncTensorsToData", at_tensors.size());
    // 创建张量数据的 BackendDataPtr
    std::vector<BackendDataPtr> handles =
        CreateTensorsData(at_tensors, devices);
    // 遍历 handles 容器中的元素，handles 是一个存储了设备上张量数据句柄的容器
    for (const auto i : c10::irange(handles.size())) {
      // 如果程序执行到这里，表示张量的 IR 值不存在。
      // 同时，我们已经将 at::Tensor 数据上传到设备，但是这些数据仍然有效，
      // 所以我们保留在延迟张量上（这样后续的 ToTensor() 操作不需要从设备重新获取数据）。
      tensors[at_tensor_index[i]]->data()->handle = std::move(handles[i]);
    }
  }
  // 记录张量图的哈希值以及所在设备，输出日志
  VLOG(4) << "Tensors graph hash " << HashToString(coll.hash) << " on device "
          << coll.device;
  // 返回整理好的数据集对象 coll
  return coll;
}

std::vector<Value> LazyGraphExecutor::CollectRoots(
    const std::vector<LazyTensorPtr>& tensors,
    c10::ArrayRef<size_t> indices) {
  // 创建一个空的值向量 roots，预留足够空间以容纳 indices.size() 个元素
  std::vector<Value> roots;
  roots.reserve(indices.size());
  // 遍历 indices 中的每个索引，将对应 LazyTensorPtr 的当前 IR 值添加到 roots 中
  for (auto index : indices) {
    roots.push_back(tensors.at(index)->CurrentIrValue());
  }
  return roots;  // 返回包含根节点 IR 值的向量
}

void LazyGraphExecutor::ExtractIRAndPrepareTensorData(
    std::vector<LazyTensorPtr>* tensors,
    const SyncTensorsConfig& config,
    c10::ArrayRef<size_t> indices,
    std::vector<Value>& ir_values,
    std::vector<BackendDataPtr>& tensor_data_vec) {
  // 预留空间以容纳 indices.size() 个元素的 ir_values 和 tensor_data_vec
  ir_values.reserve(indices.size());
  tensor_data_vec.reserve(indices.size());
  // 遍历 indices 中的每个索引
  for (auto index : indices) {
    LazyTensorPtr& tensor = (*tensors)[index];
    // 获取当前 LazyTensor 的 IR 值，并将其添加到 ir_values 中
    Value ir_value = tensor->CurrentIrValue();
    ir_values.push_back(ir_value);
    // 获取 tensor 的设备信息
    const BackendDevice& tensor_device = tensor->GetDevice();
    // 创建数据占位符，根据 tensor_device 和 tensor 的形状，存储在 tensor_data_vec 中
    BackendDataPtr handle = getBackend()->CreateDataPlaceholder(
        tensor_device, std::move(tensor->shape()));
    tensor_data_vec.push_back(handle);
    // 如果 tensor 当前的数据句柄为 nullptr 且配置要求同步 LTC 数据，则将 IR 值重置为空值
    if (tensor->CurrentDataHandle() == nullptr && config.sync_ltc_data) {
      tensor->AssignIrValue(Value());
    }
  }
}

std::vector<torch::lazy::BackendDataPtr> LazyGraphExecutor::SetTensorData(
    std::vector<LazyTensorPtr>* tensors,
    const SyncTensorsConfig& config,
    c10::ArrayRef<size_t> indices,
    const std::vector<BackendDataPtr>& tensor_data_vec) {
  // 创建一个空的 tensors_data 向量，预留足够空间以容纳 indices.size() 个元素
  std::vector<BackendDataPtr> tensors_data;
  tensors_data.reserve(indices.size());
  // 遍历 indices 中的每个索引
  for (const auto i : c10::irange(indices.size())) {
    auto index = indices[i];
    LazyTensorPtr& tensor = (*tensors)[index];
    // 如果 config.force_ltc_data 标志为真，说明是为了截断 IR 图并在选定的张量上实现设备数据同步操作
    // 异步操作完成前，如果张量没有设备数据，需要安装一个占位符
    BackendDataPtr handle = tensor->CurrentDataHandle();
    if (handle == nullptr && config.force_ltc_data) {
      // 使用预先准备的 tensor_data_vec 中的 handle
      handle = tensor_data_vec[i];
      // 注意：这里不使用 SetHandleData 方法，因为该方法会重置 ir_value
      // 在 ExtractIRAndPrepareTensorData 中已经执行了重置，以重叠前一次执行
      tensor->data()->handle = handle;
      tensor->data()->tensor_data = c10::nullopt;
    }
    // 将处理后的 handle 放入 tensors_data 中
    tensors_data.emplace_back(std::move(handle));
  }
  return tensors_data;  // 返回包含 tensor 数据句柄的向量
}

LazyGraphExecutor::PostOrderData LazyGraphExecutor::RunPostOrder(
    const std::vector<Value>& ir_values,
    ...
    SyncTensorCollection* coll) {
  // 创建一个空的节点指针向量 roots，用于存储 ir_values 中每个 IR 值的节点指针
  std::vector<const Node*> roots;
  // 预留空间以容纳 ir_values 的大小
  roots.reserve(ir_values.size());
  // 遍历 ir_values 中的每个 ir_value
  for (const auto& ir_value : ir_values) {
    // 将 ir_value 的节点指针加入 roots
    roots.push_back(ir_value.node.get());
  }
  // 创建一个 PostOrderData 结构 po_data，用于存储后序遍历的数据
  PostOrderData po_data;
  // 计算节点的后序遍历顺序，存储在 po_data 的 post_order 中
  po_data.post_order = Util::ComputePostOrder(roots, &po_data.emission_map);
  // 创建一个空的映射 data_handles，用于存储 BackendData 的 handle 和对应的索引
  std::unordered_map<BackendData::Handle, size_t> data_handles;
  // 遍历后序遍历顺序中的每个节点 node
  for (auto node : po_data.post_order) {
    // 从后端获取与节点关联的计算数据 backend_data
    const auto backend_data = getBackend()->GetComputationDataFromNode(node);
    // 如果 backend_data 存在
    if (backend_data) {
      /* 可接受的竞态条件：HasValue 可能返回 false。这是可以接受的，
       * 因为条件屏障是性能优化。 */
      // 如果 backend_data 没有值
      if (!backend_data->HasValue()) {
        // 在 TensorCollectionBarrier 中同步 Tensor 集合 coll
        TensorCollectionBarrier(coll);
      }
      // 获取 backend_data 的 handle
      BackendData::Handle handle = backend_data->GetHandle();
      // 查找 data_handles 中是否已存在 handle 对应的索引
      auto it = data_handles.find(handle);
      // 如果找到了
      if (it != data_handles.end()) {
        // 将已存在的索引加入到 po_data 的参数序列 parameter_sequence 中
        po_data.parameter_sequence.push_back(it->second);
      } else {
        // 否则将新的索引加入到 parameter_sequence 中
        po_data.parameter_sequence.push_back(po_data.parameters_data.size());
        // 将 handle 及其对应的数据 backend_data 加入 parameters_data
        data_handles[handle] = po_data.parameters_data.size();
        po_data.parameters_data.push_back(backend_data);
      }
    }
  }
  // 返回后序遍历数据 po_data
  return po_data;
  // 尝试从缓存中查找已编译的计算图
std::shared_ptr<LazyGraphExecutor::Async> LazyGraphExecutor::TryRunCachedSync(
    std::vector<LazyTensorPtr>* tensors,                        // 输入参数：懒惰张量指针的向量
    SyncTensorCollection* coll,                                 // 输入参数：同步张量集合对象指针
    PostOrderData* po_data,                                     // 输入输出参数：后序数据对象指针
    const std::vector<BackendDataPtr>& tensor_data_vec) {       // 输入参数：后端数据指针的向量引用
  ComputationCache::TypePtr cached_computation =                 // 查找缓存中已编译的计算图
      LookupCachedCompile(coll->hash);
  if (cached_computation == nullptr) {                           // 如果未找到缓存的计算图
    return nullptr;                                             // 返回空指针
  }
  if (GRAPH_DUMP_ENABLED) {                                      // 如果启用了图形转储
    auto* comp = cached_computation->computation.get();          // 获取缓存计算图的指针
    LOG(ERROR) << "Run a cached graph: " << comp->to_string() << std::endl;  // 记录日志：运行一个缓存的计算图
  }
  TORCH_LAZY_VALUE_METRIC("TensorsGraphSize", po_data->post_order.size());  // 记录懒惰计算图的张量大小度量
  VLOG(5) << "TensorsGraphSize=" << po_data->post_order.size();  // 输出日志：张量图大小

  return ScheduleSyncTensorsGraph(                               // 调度同步张量图
      tensors,                                                   // 懒惰张量指针的向量
      coll,                                                      // 同步张量集合对象指针
      std::move(po_data->parameters_data),                        // 移动后的后序数据参数
      std::move(cached_computation),                             // 移动后的缓存计算图
      tensor_data_vec);                                          // 后端数据指针的向量
}

LazyGraphExecutor::CompilationResult LazyGraphExecutor::Compile(
    const std::vector<LazyTensorPtr>& tensors,                   // 输入参数：懒惰张量指针的向量
    c10::ArrayRef<std::string> devices,                          // 输入参数：设备数组引用
    const SyncTensorCollection& coll,                            // 输入参数：同步张量集合对象的常量引用
    PostOrderData* po_data,                                     // 输入输出参数：后序数据对象指针
    const std::vector<Value>& ir_values) {                       // 输入参数：值对象的向量引用
  auto lowering_ctx = LoweringContext::Create(                   // 创建降低上下文对象
      "SyncTensorsGraph",                                       // 名称："SyncTensorsGraph"
      coll.device,                                              // 设备名称
      po_data->post_order,                                      // 后序数据的顺序
      std::move(po_data->emission_map));                        // 移动后的发射映射

  for (const auto& ir_value : ir_values) {                       // 遍历值对象的向量
    lowering_ctx->AddResult(ir_value);                          // 添加结果到降低上下文对象中
  }

  ComputationPtr computation = lowering_ctx->Build();            // 构建计算对象
  // 如果 force_ltc_data 为 true，则表示我们进行了正确的同步并处于标记步骤中。
  // 如果调用了 GetTensors，则 force_ltc_data 将为 false，表示我们过早地评估了某些值。
  computation->in_mark_step = coll.config.force_ltc_data;        // 设置标记步骤的标志

  VLOG(3) << "Compiling IR graph hash " << HashToString(coll.hash)
          << " on device " << coll.device << " ...";             // 输出日志：编译 IR 图的哈希值和设备信息
  std::vector<ComputationPtr> computations =                     // 编译计算对象的向量
      getBackend()->Compile({computation});                     // 调用后端进行编译

  VLOG(3) << "Compiling IR graph hash " << HashToString(coll.hash)
          << " on device " << coll.device << " done!";           // 输出日志：编译 IR 图完成

  if (computation) {                                            // 如果计算对象不为空
    TORCH_CHECK(                                                // 断言：计算参数的大小与后序数据的参数数据大小相等
        computation->parameters_size() ==
        static_cast<int>(po_data->parameters_data.size()));
  }

  return {                                                      // 返回编译结果对象
      /*device=*/coll.device,                                   // 设备名称
      /*emitted_nodes=*/lowering_ctx->GetEmittedNodeCount(),    // 发出的节点数量
      /*computation=*/std::move(computations.front()),          // 移动后的计算对象
      /*parameters_data=*/std::move(po_data->parameters_data)}; // 移动后的参数数据
}

LazyGraphExecutor::ComputationCache* LazyGraphExecutor::GetComputationCache() {
  static ComputationCache* cache =                              // 静态缓存对象
      new ComputationCache(FLAGS_torch_lazy_compilation_cache_size);  // 使用标志设置大小创建计算缓存对象
  return cache;                                                 // 返回缓存对象指针
}

LazyGraphExecutor::ComputationCache::TypePtr LazyGraphExecutor::
    LookupCachedCompile(const hash_t& hash) {                    // 查找缓存中的已编译计算对象
  ComputationCache::TypePtr cached_computation =                 // 获取计算对象的类型指针
      GetComputationCache()->Get(hash);                         // 从计算缓存中获取指定哈希的计算对象
  if (cached_computation == nullptr) {                           // 如果未找到缓存的计算对象
    TORCH_LAZY_COUNTER("UncachedCompile", 1);                    // 记录未缓存的计算次数
    // 记录日志：未缓存的计算
    // 返回空指针，表示函数结束且没有返回值
    return nullptr;
  }
  // 增加名为"CachedCompile"的懒惰计数器的计数，增加1
  TORCH_LAZY_COUNTER("CachedCompile", 1);
  // 返回缓存的计算结果
  return cached_computation;
}

#if defined(_MSC_VER)
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif

// 定义 LazyGraphExecutor 类的 SyncTensorsGraphInternal 方法，返回一个 Async 共享指针
std::shared_ptr<LazyGraphExecutor::Async> LazyGraphExecutor::
    SyncTensorsGraphInternal(
        std::vector<LazyTensorPtr>* tensors,
        c10::ArrayRef<std::string> devices,
        const SyncTensorsConfig& config) {
  // 收集同步张量
  SyncTensorCollection coll = CollectSyncTensors(*tensors, config);
  // 如果没有需要同步的张量，则等待之前的执行完成并返回空指针
  if (coll.indices.empty()) {
    /* Enure previous execution is complete before exiting this
     * function */
    // 等待张量集合的栅栏，确保之前的执行已完成
    TensorCollectionBarrier(&coll);
    return nullptr;
  }
  // 保存张量图信息以便调试
  DebugUtil::SaveTensorsGraphInfo(
      "ScheduleSyncTensorsGraph", *tensors, &coll.indices);
  // 初始化 IR 值和张量数据向量
  std::vector<Value> ir_values;
  std::vector<BackendDataPtr> tensor_data_vec;
  // 提取 IR 并准备张量数据
  ExtractIRAndPrepareTensorData(
      tensors, coll.config, coll.indices, ir_values, tensor_data_vec);
  // 运行后序处理并获取后序数据
  PostOrderData po_data = RunPostOrder(ir_values, &coll);
  // 计算哈希值
  coll.hash = HashCombine(coll.hash, Hash(po_data.parameter_sequence));
  // 记录哈希值的日志信息
  VLOG(4) << "Parameter sequence graph hash " << HashToString(coll.hash);
  // 尝试运行缓存的同步操作
  std::shared_ptr<Async> async =
      TryRunCachedSync(tensors, &coll, &po_data, tensor_data_vec);
  // 如果成功则直接返回 async
  if (async != nullptr) {
    return async;
  }

  // 编译张量图
  CompilationResult compile_result =
      Compile(*tensors, devices, coll, &po_data, ir_values);
  // 如果启用了图形转储，则记录计算哈希和图形信息
  if (GRAPH_DUMP_ENABLED) {
    auto* comp = compile_result.computation.get();
    LOG(ERROR) << "Add a cached computation with hash " << coll.hash
               << std::endl;
    LOG(ERROR) << "Add a graph to cache: " << comp->to_string() << std::endl;
  }

  // 记录张量图大小的延迟值度量
  TORCH_LAZY_VALUE_METRIC("TensorsGraphSize", compile_result.emitted_nodes);
  // 记录张量图大小的详细日志信息
  VLOG(5) << "TensorsGraphSize=" << compile_result.emitted_nodes;

  // 创建缓存的计算对象并添加到计算缓存中
  auto cached_computation = std::make_shared<CachedComputation>(
      std::move(compile_result.computation));
  GetComputationCache()->Add(coll.hash, cached_computation);

  // 调度同步张量图的执行，并返回异步对象
  return ScheduleSyncTensorsGraph(
      tensors,
      &coll,
      std::move(compile_result.parameters_data),
      std::move(cached_computation),
      tensor_data_vec);
}

// 定义 LazyGraphExecutor 类的 ScheduleSyncTensorsGraph 方法，返回一个 Async 共享指针
std::shared_ptr<LazyGraphExecutor::Async> LazyGraphExecutor::
    ScheduleSyncTensorsGraph(
        SyncTensorCollection* coll,
        std::vector<BackendDataPtr> parameters_data,
        std::vector<BackendDataPtr> tensors_data,
        ComputationCache::TypePtr cached_computation) {
  // 张量集合的栅栏，确保同步执行
  TensorCollectionBarrier(coll);
  // 创建异步对象，并初始化参数、张量数据及缓存的计算对象
  std::shared_ptr<Async> async = std::make_shared<Async>(
      coll,
      std::move(parameters_data),
      std::move(tensors_data),
      std::move(cached_computation));

  // 定义同步函数，使用异步对象和哈希值
  auto syncfn = [async, hash = coll->hash]() {
    // 尝试执行异步任务的 IR 图哈希计算，将结果赋值给 async->tensors_data
    try {
      // 输出执行 IR 图哈希计算的日志信息，包括哈希值和设备信息
      VLOG(3) << "Executing IR graph hash " << HashToString(hash)
              << " on device " << async->device << " ...";
      // 调用后端接口执行计算，返回结果
      auto results = getBackend()->ExecuteComputation(
          async->cached_computation->computation,
          async->parameters_data,
          async->device);
      // 输出 IR 图哈希计算完成的日志信息，包括哈希值和设备信息
      VLOG(3) << "Executing IR graph hash " << HashToString(hash)
              << " on device " << async->device << " done!";
    
      // 检查异步任务中的输出张量数量是否与结果数量一致
      TORCH_CHECK(
          async->tensors_data.size() == results.size(),
          "Expected number of outputs does not match TorchScript Stack size: ",
          async->tensors_data.size(),
          " != ",
          results.size());
    
      // 遍历结果数组，将计算结果赋值给对应的 async->tensors_data 中的张量指针
      for (const auto i : c10::irange(results.size())) {
        if (async->tensors_data[i] != nullptr) {
          async->tensors_data[i]->Assign(*results[i]);
        } else {
          async->tensors_data[i] = std::move(results[i]);
        }
      }
    } catch (...) {
      // 捕获任何异常，处理异步任务中可能发生的异常情况
      // 在异步任务中发生异常的情况下，通过设置 unlocker 的状态来表明异常情况
      // 这样可以在下次尝试获取设备锁时暴露异常
      for (auto& unlocker : async->unlocker) {
        unlocker.SetStatus(std::current_exception());
      }
      // 重新抛出异常，以便上层调用能够捕获并处理
      throw;
    }
    
    // 根据 FLAGS_torch_lazy_use_thread_pool 标志选择是使用线程池调度异步任务还是直接执行同步函数
    if (FLAGS_torch_lazy_use_thread_pool) {
      // 将异步任务的完成回调函数通过 ScheduleIoClosure 调度到线程池中执行
      ScheduleIoClosure(async->mwait.Completer(std::move(syncfn)));
    } else {
      // 直接调用同步函数 syncfn 执行任务
      syncfn();
    }
    
    // 返回异步任务对象 async
    return async;
}

std::shared_ptr<LazyGraphExecutor::Async> LazyGraphExecutor::
    ScheduleSyncTensorsGraph(
        std::vector<LazyTensorPtr>* tensors,  // 指向存储 LazyTensor 指针的向量的指针
        SyncTensorCollection* coll,  // 指向 SyncTensorCollection 对象的指针
        std::vector<BackendDataPtr> parameters_data,  // 后端数据的向量
        ComputationCache::TypePtr cached_computation,  // 缓存的计算对象指针
        const std::vector<BackendDataPtr>& tensor_data_vec) {  // 常量引用的后端数据的向量
  auto tensors_data =
      SetTensorData(tensors, coll->config, coll->indices, tensor_data_vec);  // 调用 SetTensorData 函数设置张量数据
  return ScheduleSyncTensorsGraph(
      coll,
      std::move(parameters_data),
      std::move(tensors_data),
      std::move(cached_computation));  // 调度同步张量图的执行，并返回异步执行对象的指针
}

std::vector<at::Tensor> LazyGraphExecutor::GetTensorsFused(
    std::vector<LazyTensorPtr>* tensors) {  // 指向存储 LazyTensor 指针的向量的指针
  SyncTensorsConfig config;  // 同步张量的配置对象
  config.force_ltc_data = false;  // 设置强制 LTC 数据为 false
  auto async = SyncTensorsGraphInternal(tensors, {}, config);  // 调用 SyncTensorsGraphInternal 函数同步张量图，并返回异步执行对象
  if (FLAGS_torch_lazy_use_thread_pool && async != nullptr) {  // 如果使用线程池且异步对象非空
    async->mwait.Wait();  // 等待异步操作完成
  }
  std::vector<BackendDataPtr> tensors_data = GatherTensorsData(
      *tensors,
      async != nullptr ? async->indices : c10::ArrayRef<size_t>(),
      async != nullptr ? async->tensors_data : c10::ArrayRef<BackendDataPtr>());  // 收集张量的数据
  return FetchTensors(
      tensors, tensors_data, async != nullptr ? &async->indices : nullptr);  // 获取张量数据并返回
}

// This gets tensors from the backend
// for TS backend, we'd ideally just cut through these layers and
// not need to copy the tensor, just move it

// for XLA backend, a copy is going to have to happen,

// could we replace the 'Data' object with an at::Tensor, which is 'undefined'
// unless a backend attaches a buffer to it?  That way we can have a
// 'PopulateTensor' method on backend, which can either attach an existing
// tensor buffer to the wrapper, or copy data?
std::vector<at::Tensor> LazyGraphExecutor::FetchTensors(
    std::vector<LazyTensorPtr>* tensors,  // 指向存储 LazyTensor 指针的向量的指针
    c10::ArrayRef<BackendDataPtr> tensors_data,  // 后端数据的常量引用
    const std::vector<size_t>* indices) {  // 索引的指针
  std::vector<at::Tensor> results;  // 存储返回的 at::Tensor 结果
  size_t literals_index = 0;  // 字面量索引
  size_t sync_index = 0;  // 同步索引
  results.reserve(tensors->size());  // 预留空间以存储结果

  for (const auto i : c10::irange(tensors->size())) {  // 迭代处理每个 LazyTensor
    if (indices != nullptr && sync_index < indices->size() &&
        i == (*indices)[sync_index]) {  // 如果索引非空且索引有效
      results.push_back(getBackend()->MakeTensorFromComputationData(
          tensors_data[literals_index], (*tensors)[i]->dtype()));  // 从计算数据创建张量并添加到结果中
      ++literals_index;  // 增加字面量索引
      ++sync_index;  // 增加同步索引
    } else {
      std::optional<at::Tensor> tensor_data =
          (*tensors)[i]->CurrentTensorData();  // 获取当前张量数据
      if (tensor_data) {  // 如果有张量数据
        results.push_back(*tensor_data);  // 添加张量到结果中
      } else {
        TORCH_CHECK(literals_index < tensors_data.size());  // 检查字面量索引小于后端数据的大小
        results.push_back(getBackend()->MakeTensorFromComputationData(
            tensors_data[literals_index], (*tensors)[i]->dtype()));  // 从计算数据创建张量并添加到结果中
        ++literals_index;  // 增加字面量索引
      }
    }
  }
  return results;  // 返回结果向量
}

std::vector<BackendDataPtr> LazyGraphExecutor::GatherTensorsData(
    const std::vector<LazyTensorPtr>& tensors,  // 存储 LazyTensor 指针的向量的常量引用
    c10::ArrayRef<size_t> indices,  // 索引的常量引用
    // 定义一个函数，接受一个张量的列表和对应的张量数据引用，返回处理后的张量数据列表
    c10::ArrayRef<BackendDataPtr> tensors_data) {
  // 创建一个空的结果张量数据列表
  std::vector<BackendDataPtr> result_tensors_data;
  // 创建一个映射，用于存储张量唯一标识符到结果张量数据列表索引的映射关系
  std::unordered_map<int64_t, size_t> uid_index_map;
  // 初始化索引变量，用于跟踪当前处理的索引
  size_t indices_index = 0;
  // 遍历张量列表中的每一个张量
  for (const auto i : c10::irange(tensors.size())) {
    // 获取当前张量的唯一标识符
    int64_t tensor_id = tensors[i]->GetUniqueId();
    // 在映射中查找当前张量的唯一标识符
    auto it = uid_index_map.find(tensor_id);
    // 如果找到了，则当前张量是之前处理过的重复张量，从结果张量数据列表中复制数据
    if (it != uid_index_map.end()) {
      // 将之前处理过的张量数据放入结果张量数据列表
      result_tensors_data.push_back(result_tensors_data[it->second]);
    } else if (indices_index < indices.size() && i == indices[indices_index]) {
      // 如果当前索引是需要同步 IR 节点的索引，则使用异步对象中的数据
      // 将当前张量的唯一标识符与结果张量数据列表的索引关联存储
      uid_index_map.emplace(tensor_id, result_tensors_data.size());
      // 将异步数据放入结果张量数据列表
      result_tensors_data.push_back(tensors_data[indices_index]);
      // 更新索引，移动到下一个需要同步 IR 节点的索引
      ++indices_index;
    } else if (!tensors[i]->CurrentTensorData()) {
      // 如果当前张量没有当前的张量数据，则获取其当前数据处理句柄
      BackendDataPtr handle = tensors[i]->CurrentDataHandle();
      // 检查处理句柄不能为空
      TORCH_CHECK(handle != nullptr);
      // 将处理句柄放入结果张量数据列表
      result_tensors_data.push_back(std::move(handle));
    }
  }
  // 返回处理后的结果张量数据列表
  return result_tensors_data;
}
}

// 在 LazyGraphExecutor 类中定义的 TensorCollectionBarrier 方法
void LazyGraphExecutor::TensorCollectionBarrier(SyncTensorCollection* coll) {
  // 如果传入的 SyncTensorCollection 指针不为空
  if (coll) {
    // 静态常量字符串，用于标识未分配设备的临时解决方案
    static const std::string invalid_device(
        "Unknown0"); /* Temp solution to idetify unassigned devices */
    // 如果 coll 的设备名称为无效设备或者 unlocker 非空，则直接返回
    if (coll->device.toString() == invalid_device || !coll->unlocker.empty()) {
      return;
    }
    // 记录日志，等待设备屏障，显示相关设备信息
    VLOG(4) << "Waiting on device barrier for device " << coll->device
            << " ...";
    {
      // 使用 TORCH_LAZY_TIMED 宏记录时间，等待设备锁
      TORCH_LAZY_TIMED("DeviceLockWait");
      // 调用 DeviceLockerArena 的 LockDevices 方法锁定指定设备，更新 coll 的 unlocker
      coll->unlocker = DeviceLockerArena::Get()->LockDevices({coll->device});
    }
    // 记录日志，设备屏障完成
    VLOG(4) << "Waiting on device barrier for device " << coll->device
            << " done!";
  }
}

// 在 LazyGraphExecutor 类中定义的 GetGraphHash 方法
hash_t LazyGraphExecutor::GetGraphHash(
    const std::vector<LazyTensorPtr>& tensors) {
  // 创建 SyncTensorsConfig 对象，并设置 sync_ltc_data 为 false
  SyncTensorsConfig config;
  config.sync_ltc_data = false;

  // 收集与给定懒惰张量相关的同步张量信息
  auto coll = CollectSyncTensors(tensors, config);
  
  // 创建用于存储 IR 值的向量
  std::vector<Value> ir_values;
  // 遍历 coll.indices 中的索引，获取每个张量的当前 IR 值并存储到 ir_values 中
  for (auto index : coll.indices) {
    Value ir_value = tensors[index]->CurrentIrValue();
    ir_values.push_back(ir_value);
  }
  
  // 运行后序遍历操作，生成参数序列的数据
  auto po_data = RunPostOrder(ir_values, &coll);
  
  // 更新 coll 的 hash 值，结合后序遍历生成的参数序列的哈希值
  coll.hash = HashCombine(coll.hash, Hash(po_data.parameter_sequence));
  
  // 返回最终计算得到的哈希值
  return coll.hash;
}

// 命名空间 lazy 内的结束标记
} // namespace lazy

// 命名空间 torch 内的结束标记
} // namespace torch
```