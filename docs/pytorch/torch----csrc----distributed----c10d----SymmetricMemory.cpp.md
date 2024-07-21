# `.\pytorch\torch\csrc\distributed\c10d\SymmetricMemory.cpp`

```
// 包含 SymmetricMemory 头文件中的 SymmetricMemory 类型定义
#include <torch/csrc/distributed/c10d/SymmetricMemory.hpp>

// 使用匿名命名空间，以限制符号的作用范围
namespace {

// 引入 c10d::symmetric_memory 命名空间中的符号
using namespace c10d::symmetric_memory;

// 全局变量，标识是否正在进行最终化
static bool is_finalizing_ = false;

// 分配器映射类，用于管理不同设备类型的内存分配器
class AllocatorMap {
 public:
  // 获取 AllocatorMap 的单例实例
  static AllocatorMap& get() {
    static AllocatorMap instance;
    return instance;
  }

  // 注册特定设备类型的内存分配器
  void register_allocator(
      c10::DeviceType device_type,
      c10::intrusive_ptr<SymmetricMemoryAllocator> allocator) {
    map_[device_type] = std::move(allocator);
  }

  // 获取特定设备类型的内存分配器
  c10::intrusive_ptr<SymmetricMemoryAllocator> get_allocator(
      c10::DeviceType device_type) {
    auto it = map_.find(device_type);
    // 检查设备类型是否在映射中，否则抛出异常
    TORCH_CHECK(
        it != map_.end(),
        "SymmetricMemory does not support device type ",
        device_type);
    return it->second;
  }

  // 析构函数，在销毁时设置最终化标志
  ~AllocatorMap() {
    is_finalizing_ = true;
  }

 private:
  AllocatorMap() = default;  // 默认构造函数私有化，保证单例模式
  AllocatorMap(const AllocatorMap&) = delete;  // 禁用拷贝构造函数
  AllocatorMap& operator=(const AllocatorMap&) = delete;  // 禁用赋值操作符

  // 映射表，将设备类型映射到对应的内存分配器
  std::unordered_map<
      c10::DeviceType,
      c10::intrusive_ptr<SymmetricMemoryAllocator>>
      map_;
};

// 全局变量，用于存储群组信息的映射表
static std::unordered_map<std::string, GroupInfo> group_info_map{};

// 全局变量，将分配 ID 映射到设备指针的映射表
static std::unordered_map<uint64_t, void*> alloc_id_to_dev_ptr{};

// 全局变量，将分配 ID 映射到 StorageImpl 的弱引用的映射表
static std::unordered_map<uint64_t, c10::weak_intrusive_ptr<c10::StorageImpl>>
    alloc_id_to_storage{};

// 创建空的分布式张量，支持零拷贝的持久性分配
static at::Tensor empty_strided_p2p_persistent(
    c10::IntArrayRef size,
    c10::IntArrayRef stride,
    c10::ScalarType dtype,
    c10::Device device,
    const std::string& group_name,
    uint64_t alloc_id) {
  // 如果先前具有相同 alloc_id 的分配仍处于活动状态，则使分配失败
  auto storage = alloc_id_to_storage.find(alloc_id);
  if (storage != alloc_id_to_storage.end() && storage->second.use_count() > 0) {
    TORCH_CHECK(
        false,
        "SymmetricMemory::empty_strided_p2p_persistent: ",
        "can not allocate with alloc_id == ",
        alloc_id,
        " because a previous allocation with the same alloc_id "
        "is still active.");
  }

  // 计算张量的元素数
  const size_t numel =
      std::accumulate(size.begin(), size.end(), 1, std::multiplies<int>());
  // 计算每个元素的字节大小
  const size_t element_size = c10::elementSize(dtype);
  // 计算总分配大小
  const size_t alloc_size = numel * element_size;

  // 获取对应设备类型的内存分配器
  auto allocator = get_allocator(device.type());
  void* dev_ptr = nullptr;
  // 如果已经存在相同 alloc_id 的设备指针，则检查分配大小是否一致
  if (alloc_id_to_dev_ptr.find(alloc_id) != alloc_id_to_dev_ptr.end()) {
    dev_ptr = alloc_id_to_dev_ptr[alloc_id];
    TORCH_CHECK(
        alloc_size == allocator->get_alloc_size(dev_ptr),
        "SymmetricMemory::empty_strided_p2p_persistent: ",
        "requested allocation size (",
        alloc_size,
        ") is different from the size of a previous allocation ",
        "with the same alloc_id ",
        allocator->get_alloc_size(dev_ptr));
  } else {
    // 否则，使用分配器进行内存分配
    dev_ptr = allocator->alloc(alloc_size, device.index(), group_name);
    alloc_id_to_dev_ptr[alloc_id] = dev_ptr;  // 将设备指针与 alloc_id 关联存储
  }
    # 将分配ID映射到设备指针
    alloc_id_to_dev_ptr[alloc_id] = dev_ptr;
  }

  # 创建张量的选项，指定数据类型和设备
  auto options = at::TensorOptions().dtype(dtype).device(device);
  # 从给定的原始数据(dev_ptr)创建张量
  auto allocated = at::from_blob(dev_ptr, size, stride, options);

  // 跟踪分配的活跃性
  # 从分配ID到存储对象的映射中删除旧的分配
  alloc_id_to_storage.erase(alloc_id);
  # 向分配ID到存储对象的映射中添加新的分配
  alloc_id_to_storage.emplace(
      alloc_id, allocated.storage().getWeakStorageImpl());
  # 返回新分配的张量
  return allocated;
} // 结束 symmetric_memory 命名空间

} // 结束 c10d 命名空间

namespace c10d {
namespace symmetric_memory {

// 返回当前是否正在进行终止操作的状态
bool is_finalizing() {
  return is_finalizing_;
}

// 注册给定设备类型的对称内存分配器
void register_allocator(
    c10::DeviceType device_type,
    c10::intrusive_ptr<SymmetricMemoryAllocator> allocator) {
  return AllocatorMap::get().register_allocator(
      device_type, std::move(allocator));
}

// 获取给定设备类型的对称内存分配器
c10::intrusive_ptr<SymmetricMemoryAllocator> get_allocator(
    c10::DeviceType device_type) {
  return AllocatorMap::get().get_allocator(device_type);
}

// 设置组信息，包括组名、排名、世界大小和存储对象
void set_group_info(
    const std::string& group_name,
    int rank,
    int world_size,
    c10::intrusive_ptr<Store> store) {
  // 检查组信息映射中是否已经存在同名组信息
  TORCH_CHECK(group_info_map.find(group_name) == group_info_map.end());
  GroupInfo group_info;
  group_info.rank = rank;
  group_info.world_size = world_size;
  group_info.store = std::move(store);
  group_info_map.emplace(group_name, std::move(group_info));
}

// 获取指定组名的组信息
const GroupInfo& get_group_info(const std::string& group_name) {
  // 检查组信息映射中是否存在指定组名的组信息
  TORCH_CHECK(
      group_info_map.find(group_name) != group_info_map.end(),
      "get_group_info: no group info associated with the group name ",
      group_name);
  return group_info_map[group_name];
}

// 创建一个空的分布式张量，使用指定的大小、步幅、数据类型和设备，可选择使用分配ID
at::Tensor empty_strided_p2p(
    c10::IntArrayRef size,
    c10::IntArrayRef stride,
    c10::ScalarType dtype,
    c10::Device device,
    const std::string& group_name,
    std::optional<uint64_t> alloc_id) {
  // 如果分配ID有值，则使用持久化方式创建空的分布式张量
  if (alloc_id.has_value()) {
    return empty_strided_p2p_persistent(
        size, stride, dtype, device, group_name, *alloc_id);
  }
  // 计算张量的元素数量
  const size_t numel =
      std::accumulate(size.begin(), size.end(), 1, std::multiplies<int>());
  // 计算每个元素的大小
  const size_t element_size = c10::elementSize(dtype);
  // 计算需要分配的总内存大小
  const size_t alloc_size = numel * element_size;

  // 获取指定设备类型的内存分配器
  auto allocator = get_allocator(device.type());
  // 使用分配器在指定设备上分配内存
  void* dev_ptr = allocator->alloc(alloc_size, device.index(), group_name);

  // 设置张量的选项，包括数据类型和设备
  auto options = at::TensorOptions().dtype(dtype).device(device);
  // 创建张量，从分配的内存中构造，并指定释放函数
  return at::from_blob(
      dev_ptr,
      size,
      stride,
      [allocator = std::move(allocator)](void* ptr) { allocator->free(ptr); },
      options);
}

// 将张量通过分布式内存进行汇合
TORCH_API c10::intrusive_ptr<SymmetricMemory> rendezvous(
    const at::Tensor& tensor) {
  // 获取张量所在设备的内存分配器
  auto allocator = get_allocator(tensor.device().type());
  // 使用分配器进行汇合操作，返回对称内存对象
  return allocator->rendezvous(tensor.data_ptr());
}

// 获取张量所在的对称内存对象
c10::intrusive_ptr<SymmetricMemory> get_symmetric_memory(
    const at::Tensor& tensor) {
  // 获取张量所在设备的内存分配器
  auto allocator = get_allocator(tensor.device().type());
  // 检查指定张量的数据是否已经汇合完成
  TORCH_CHECK(
      allocator->is_rendezvous_completed(tensor.data_ptr()),
      "SymmetricMemory: must invoke rendezvous on a tensor ",
      "before calling get_symmetric_memory on it");
  // 返回张量的对称内存对象
  return allocator->rendezvous(tensor.data_ptr());
}

} // 结束 symmetric_memory 命名空间
} // 结束 c10d 命名空间
```