# `.\pytorch\torch\csrc\CudaIPCTypes.cpp`

```py
// 引入必要的头文件
#include <ATen/MapAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/csrc/CudaIPCTypes.h>
#include <atomic>
#include <map>
#include <mutex>
#include <string>

// 定义命名空间 torch
namespace torch {

// 匿名命名空间内定义的静态函数，用于警告生产者进程在释放所有共享 CUDA 张量之前终止
namespace {

void warnProducerTerminatedBeforeSharedTensorsReleased() {
  // 静态变量，用于确保仅发出一次警告
  static bool warned = false;
  // 如果尚未发出警告，则记录警告并打印相关信息
  if (!warned) {
    LOG(WARNING)
        << "Producer process has been terminated before all shared CUDA tensors released. See Note [Sharing CUDA tensors]";
    warned = true;
  }
}

} // namespace

// CUDA IPC 全局实体结构体定义
struct CudaIPCGlobalEntities {
  // alive 用于跟踪对象生命周期，避免在对象销毁后访问而导致段错误
  static bool alive;

  // 互斥量，用于保护引用计数器和同步事件的操作
  std::mutex ref_counters_mutex_;

  // 记录已使用的同步事件数量的原子计数器
  std::atomic<int64_t> sync_events_used_{0};

  // 映射，将文件名映射到共享指针的引用计数器文件
  std::map<std::string, std::shared_ptr<CudaIPCRefCountersFile>> ref_counters_files_;

  // 下一个可用的引用计数器文件的共享指针
  std::shared_ptr<CudaIPCRefCountersFile> next_available_ref_counters_file_;

  // CUDA IPC 发送数据 limbo 对象
  CudaIPCSentDataLimbo CudaIPCSentDataLimbo_;

  // 构造函数，初始化对象时将 alive 设置为 true
  CudaIPCGlobalEntities() {
    alive = true;
  }

  // 析构函数，在对象销毁时执行清理操作
  ~CudaIPCGlobalEntities() {
    // 执行 CUDA IPC 发送数据 limbo 对象的收集操作
    CudaIPCSentDataLimbo_.collect();
    // 安全清理当前文件
    safe_clean_current_file();
    // 如果存在下一个可用的引用计数器文件，则发出警告
    if (next_available_ref_counters_file_) {
      warnProducerTerminatedBeforeSharedTensorsReleased();
    }
    // 将 alive 标记为 false，表明对象已销毁
    alive = false;
  }

  // 安全清理当前文件的方法
  void safe_clean_current_file() {
    // 使用互斥量保护操作
    std::lock_guard<std::mutex> lock(ref_counters_mutex_);
    // 如果下一个可用的引用计数器文件存在且没有被使用，则从映射中移除
    if (next_available_ref_counters_file_ &&
        next_available_ref_counters_file_->offsets_in_use() == 0) {
      ref_counters_files_.erase(next_available_ref_counters_file_->handle());
      next_available_ref_counters_file_.reset();
    }
  }
};

// 初始化静态变量 alive 为 false
bool CudaIPCGlobalEntities::alive = false;

// 创建全局单例对象 cuda_ipc_global_entities，类型为 CudaIPCGlobalEntities
CudaIPCGlobalEntities cuda_ipc_global_entities;

// CUDA IPC 发送数据 limbo 的析构函数实现
CudaIPCSentDataLimbo::~CudaIPCSentDataLimbo() {
  // 执行收集操作以释放内存
  collect();
  // 如果 limbo 中仍有数据块，则发出警告
  if (size() > 0) {
    warnProducerTerminatedBeforeSharedTensorsReleased();
  }
}

// CUDA IPC 发送数据 limbo 的收集方法实现
bool CudaIPCSentDataLimbo::collect() {
  // 标志变量，指示是否释放了内存
  bool freed_memory = false;
  // 存储需要重置的块的唯一指针
  std::vector<std::unique_ptr<CudaIPCSentData>> reset_blocks;
  { // 进入临界区域以修改共享块
    std::lock_guard<std::mutex> lock(limbo_mutex_);
    // 存储保留的块的列表
    std::vector<std::unique_ptr<CudaIPCSentData>> kept_blocks;
    // 遍历 limbo 中的共享块
    for (auto& sd : shared_blocks_) {
      // 如果块的计数值大于 0，则将其保留
      if (sd->counter_value() > 0) {
        kept_blocks.push_back(std::move(sd));
      } else {
        // 否则释放内存并将块添加到重置列表中
        freed_memory = true;
        reset_blocks.push_back(std::move(sd));
      }
    }
    // 更新共享块列表为保留的块
    shared_blocks_ = std::move(kept_blocks);
  }
  // 在临界区域外重置块，以避免死锁
  for (auto& sd : reset_blocks) {
    sd.reset();
  }
  // 返回是否释放了内存的标志
  return freed_memory;
}

// 向 limbo 中添加 CUDA IPC 发送数据块的方法
void CudaIPCSentDataLimbo::add(std::unique_ptr<CudaIPCSentData> shared_block) {
  // 使用互斥量保护操作
  std::lock_guard<std::mutex> lock(limbo_mutex_);
  // 静态变量，用于确保仅在 limbo 中存在多个块时发出一次警告
  static bool warned = false;
  // 如果 limbo 中的共享块数超过阈值且尚未发出警告，则发出警告
  if (shared_blocks_.size() > CUDA_IPC_WARN_AFTER_X_BLOCKS_IN_LIMBO &&
      !warned) {


这样，每一行代码都被详细注释，解释了其作用和相关的注意事项。
    LOG(WARNING)
        << "Producer process tried to deallocate over "
        << CUDA_IPC_WARN_AFTER_X_BLOCKS_IN_LIMBO
        << " memory blocks referred by consumer processes. Deallocation might be significantly slowed down. "
        << "We assume it will never going to be the case, but if it is, please file but to https://github.com/pytorch/pytorch";
    // 记录警告日志，提示生产者进程试图释放超过 CUDA_IPC_WARN_AFTER_X_BLOCKS_IN_LIMBO 所定义的内存块数。
    // 如果确实发生这种情况，可能会显著减慢释放速度。
    // 代码假设这种情况永远不会发生，但如果确实发生，请提交 bug 到 https://github.com/pytorch/pytorch。
    warned = true;
  }
  shared_blocks_.push_back(std::move(shared_block));
    // 将 shared_block 移动到 shared_blocks_ 向量中的末尾
// CudaIPCSentDataLimbo 类的成员函数，返回共享块的数量
uint64_t CudaIPCSentDataLimbo::size() {
  // 使用互斥锁保护临界区域，确保线程安全访问
  std::lock_guard<std::mutex> lock(limbo_mutex_);
  // 返回共享块的数量
  return shared_blocks_.size();
}

// CudaIPCSentDataDelete 函数，用于删除 CUDA IPC 发送的数据
void CudaIPCSentDataDelete(void* ptr) {
  // 将 void 指针转换为 CudaIPCSentData 类型的唯一指针
  std::unique_ptr<CudaIPCSentData> sent_data(
      static_cast<CudaIPCSentData*>(ptr));
  // 如果 CUDA IPC 全局实体不存活，直接返回
  if (!CudaIPCGlobalEntities::alive) {
    return;
  }
  // 如果发送数据的引用计数大于 0，则将其添加到 CudaIPCSentDataLimbo 中
  if (sent_data->counter_value() > 0) {
    cuda_ipc_global_entities.CudaIPCSentDataLimbo_.add(std::move(sent_data));
  }
  // 执行清理操作，收集不再使用的数据
  cuda_ipc_global_entities.CudaIPCSentDataLimbo_.collect();
}

// ReturnRefCounter 函数，用于返回引用计数器
void ReturnRefCounter(const std::string& handle, uint64_t offset /* unused */) {
  // 如果 CUDA IPC 全局实体不存活，直接返回
  if (!CudaIPCGlobalEntities::alive) {
    return;
  }
  // 使用互斥锁保护引用计数器的临界区域
  std::lock_guard<std::mutex> lock(
      cuda_ipc_global_entities.ref_counters_mutex_);
  // 获取引用计数器文件映射
  auto& map = cuda_ipc_global_entities.ref_counters_files_;
  // 查找指定 handle 的引用计数器
  auto it = map.find(handle);
  // 如果找到了该 handle 的引用计数器
  if (it != map.end()) {
    // 将 offset 返回给引用计数器
    it->second->return_offset(offset);
    // 如果该引用计数器不再使用且没有偏移量，则从映射中删除
    if (it->second->offsets_in_use() == 0 && !it->second->have_offsets()) {
      map.erase(handle);
    }
  }
}
    # 检查是否需要同步事件
    if (event_sync_required_) {
      # 设置CUDA设备为当前设备
      at::cuda::CUDAGuard device_guard(device_.index());
      # 销毁CUDA事件对象
      C10_CUDA_CHECK(cudaEventDestroy(event_));
      # 如果CUDA IPC全局实体未存活，直接返回
      if (!CudaIPCGlobalEntities::alive) {
        return;
      }
      # 减少CUDA IPC全局实体使用的同步事件计数
      cuda_ipc_global_entities.sync_events_used_--;
    }
  } catch (...) { /* No throw */
  }
#endif

// 结束一个条件编译指令块，用于关闭之前的一个条件编译指令块


}

uint64_t CudaIPCSentData::counter_value() {
  return *counter_ptr_;
}

// 返回 CudaIPCSentData 类的 counter_ptr_ 指向的值，该值类型为 uint64_t


at::DataPtr GetNewRefCountedSentData(void* data, at::Device device) {
  {
    std::lock_guard<std::mutex> lock(
        cuda_ipc_global_entities.ref_counters_mutex_);
    // 使用互斥锁保护全局变量 cuda_ipc_global_entities 的访问，确保线程安全性
    if (!cuda_ipc_global_entities.next_available_ref_counters_file_) {
      // 如果未找到下一个可用的引用计数器文件，则执行以下操作：
      std::string ref_counter_handle = at::NewProcessWideShmHandle();
      // 生成一个新的进程范围共享内存句柄

      int flags =
          at::ALLOCATOR_MAPPED_SHAREDMEM | at::ALLOCATOR_MAPPED_EXCLUSIVE;
      // 指定共享内存分配器的标志

      at::DataPtr sptr = at::RefcountedMapAllocator::makeDataPtr(
          ref_counter_handle.c_str(),
          flags,
          sizeof(int64_t) * CUDA_IPC_REF_COUNTER_FILE_SIZE,
          nullptr);
      // 使用共享内存分配器创建数据指针 sptr，指向新的引用计数器文件

      auto rc = std::make_shared<CudaIPCRefCountersFile>(
          ref_counter_handle, CUDA_IPC_REF_COUNTER_FILE_SIZE, std::move(sptr));
      // 创建共享指针 rc，指向新的 CudaIPCRefCountersFile 对象，表示引用计数器文件

      cuda_ipc_global_entities.ref_counters_files_[ref_counter_handle] = rc;
      // 将新创建的引用计数器文件对象 rc 放入全局变量中管理

      cuda_ipc_global_entities.next_available_ref_counters_file_ = rc;
      // 更新全局变量，指定下一个可用的引用计数器文件为 rc
    }
  }
  cuda_ipc_global_entities.next_available_ref_counters_file_->set_counter(1);
  // 设置当前使用的引用计数器文件的计数器为 1

  auto sent_data = new CudaIPCSentData(
      cuda_ipc_global_entities.next_available_ref_counters_file_->handle(),
      cuda_ipc_global_entities.next_available_ref_counters_file_->get_offset(),
      cuda_ipc_global_entities.next_available_ref_counters_file_->counter_ptr(),
      device);
  // 创建一个新的 CudaIPCSentData 对象，使用当前引用计数器文件的信息

  cuda_ipc_global_entities.next_available_ref_counters_file_->rotate_offset();
  // 旋转当前引用计数器文件的偏移量

  if (!cuda_ipc_global_entities.next_available_ref_counters_file_
           ->have_offsets()) {
    cuda_ipc_global_entities.next_available_ref_counters_file_.reset();
    // 如果当前引用计数器文件没有偏移量，则重置该文件对象
  }

  return at::DataPtr(data, sent_data, CudaIPCSentDataDelete, device);
  // 返回一个 at::DataPtr 对象，表示新创建的数据指针，用于管理 sent_data
}

bool CudaIPCCollect() {
  if (!CudaIPCGlobalEntities::alive) {
    return true;
  }
  // 如果 CUDA IPC 全局实体不再活跃，则返回 true，表示可以收集内存

  bool freed_memory = cuda_ipc_global_entities.CudaIPCSentDataLimbo_.collect();
  // 尝试收集 CUDA IPC 发送数据的临时存储区中的内存

  if (cuda_ipc_global_entities.CudaIPCSentDataLimbo_.size() == 0) {
    cuda_ipc_global_entities.safe_clean_current_file();
    // 如果临时存储区中没有数据了，则安全地清理当前文件
  }

  return freed_memory;
  // 返回是否成功释放了内存
}

} // namespace torch

namespace c10 {
namespace {
REGISTER_FREE_MEMORY_CALLBACK("cuda_ipc_collect", CudaIPCCollectCallback);
}
} // namespace c10
```