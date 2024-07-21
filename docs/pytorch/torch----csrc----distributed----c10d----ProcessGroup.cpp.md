# `.\pytorch\torch\csrc\distributed\c10d\ProcessGroup.cpp`

```
#include <ATen/ThreadLocalState.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

#include <c10/util/Logging.h>
#include <fmt/format.h>
#include <string_view>

#include <torch/csrc/distributed/c10d/PrefixStore.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupGloo.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupMPI.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupUCC.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupWrapper.hpp>

// 命名空间 c10d 下的定义
namespace c10d {

// 静态函数，将字符串视图转换为对应的 ProcessGroup::BackendType 枚举类型
static ProcessGroup::BackendType strToBackendType(std::string_view backend) {
  // 根据输入的字符串判断后端类型并返回对应的枚举类型
  if (backend == "undefined") {
    return ProcessGroup::BackendType::UNDEFINED;
  } else if (backend == "gloo") {
    return ProcessGroup::BackendType::GLOO;
  } else if (backend == "nccl") {
    return ProcessGroup::BackendType::NCCL;
  } else if (backend == "ucc") {
    return ProcessGroup::BackendType::UCC;
  } else if (backend == "mpi") {
    return ProcessGroup::BackendType::MPI;
  } else {
    return ProcessGroup::BackendType::CUSTOM;
  }
}

// 将 OpType 转换为对应的字符串表示
std::string opTypeToString(OpType opType) {
  switch (opType) {
    case OpType::BROADCAST:
      return "BROADCAST";
    case OpType::ALLREDUCE:
      return "ALLREDUCE";
    case OpType::ALLREDUCE_COALESCED:
      return "ALLREDUCE_COALESCED";
    case OpType::REDUCE:
      return "REDUCE";
    case OpType::ALLGATHER:
      return "ALLGATHER";
    case OpType::_ALLGATHER_BASE:
      return "_ALLGATHER_BASE";
    case OpType::ALLGATHER_COALESCED:
      return "ALLGATHER_COALESCED";
    case OpType::GATHER:
      return "GATHER";
    case OpType::SCATTER:
      return "SCATTER";
    case OpType::REDUCE_SCATTER:
      return "REDUCE_SCATTER";
    case OpType::ALLTOALL_BASE:
      return "ALLTOALL_BASE";
    case OpType::ALLTOALL:
      return "ALLTOALL";
    case OpType::SEND:
      return "SEND";
    case OpType::RECV:
      return "RECV";
    case OpType::RECVANYSOURCE:
      return "RECVANYSOURCE";
    case OpType::BARRIER:
      return "BARRIER";
    case OpType::UNKNOWN:
      return "UNKNOWN";
    case OpType::_REDUCE_SCATTER_BASE:
      return "_REDUCE_SCATTER_BASE";
    case OpType::COALESCED:
      return "COALESCED";
    case OpType::_ALLREDUCE_SPARSE:
      return "_ALLREDUCE_SPARSE";
    default:
      // 如果出现未知的 OpType，抛出内部断言错误
      TORCH_INTERNAL_ASSERT(false, "Unknown op type!");
  }
  // 默认返回 "UNKNOWN"，虽然在未知类型时会抛出异常
  return "UNKNOWN";
}

// 判断给定的 OpType 是否为点对点操作
bool isP2POp(OpType opType, bool batchP2P /*= false*/) {
  // 如果是批量 P2P 操作，则直接返回 false
  if (batchP2P)
    return false;
  // 否则判断 opType 是否为 SEND、RECV 或 RECVANYSOURCE 中的一种
  return opType == OpType::SEND || opType == OpType::RECV ||
      opType == OpType::RECVANYSOURCE;
}

// 获取与设备类型对应的后端对象
c10::intrusive_ptr<Backend> ProcessGroup::getBackend(
    c10::DeviceType deviceType) {
  // 如果 deviceTypeToBackend_ 中存在对应的后端，则返回它
  if (deviceTypeToBackend_.find(deviceType) != deviceTypeToBackend_.end()) {
  // 返回 deviceType 对应的后端对象
  return deviceTypeToBackend_.at(deviceType);
}

// 获取与设备关联的后端类型
ProcessGroup::BackendType backendType{ProcessGroup::BackendType::UNDEFINED};
try {
  // 尝试获取设备类型对应的后端类型
  backendType = deviceTypeToBackendType_.at(deviceType);
} catch (const std::out_of_range& e) {
  // 如果找不到对应的后端类型，抛出异常并显示错误消息
  TORCH_CHECK(
      false, "No backend type associated with device type ", deviceType);
}

// 检查是否已经初始化了该后端类型
if (backendTypeToBackend_.find(backendType) != backendTypeToBackend_.end()) {
  // 如果已经初始化，获取对应的后端对象并将其与设备类型关联起来
  auto backend = backendTypeToBackend_.at(backendType);
  deviceTypeToBackend_[deviceType] = backend;
  return backend;
}

// 如果无法获取或创建指定后端类型的后端对象，抛出错误消息
TORCH_CHECK(
    false,
    "Could not retrieve or create the backend ",
    backendType,
    " for device type ",
    deviceType);
}

// ProcessGroup 类的构造函数，接受 store 对象、rank、size 和 options 参数
ProcessGroup::ProcessGroup(
    const c10::intrusive_ptr<::c10d::Store>& store,
    int rank,
    int size,
    c10::intrusive_ptr<Options> options)
    : store_(store),
      rank_(rank),
      size_(size),
      options_(std::move(options)),
      backendType_(strToBackendType(options_->backend)), // 使用 options 中的 backend 字段转换为对应的 BackendType
      dist_debug_level_(debug_level()) { // 初始化 dist_debug_level_，调用 debug_level() 函数获取值
  C10_LOG_API_USAGE_ONCE("c10d.process_group"); // 记录一次 API 使用情况
}

// ProcessGroup 类的另一个构造函数，接受 rank 和 size 参数
ProcessGroup::ProcessGroup(int rank, int size)
    : rank_(rank), size_(size), backendType_(BackendType::UNDEFINED) {} // 初始化 rank、size 和 backendType_

// ProcessGroup 类的析构函数，默认实现
ProcessGroup::~ProcessGroup() = default;

// ProcessGroup 类的初始化方法
void ProcessGroup::init() {
  C10_LOG_API_USAGE_ONCE(
      fmt::format("c10d.process_group_{}", getBackendName())); // 记录一次 API 使用情况，使用 getBackendName() 函数获取后端名称
}

// 获取 ProcessGroup 的组名的常量引用
const std::string& ProcessGroup::getGroupName() const {
  TORCH_CHECK(!deviceTypeToBackend_.empty(), "ProcessGroup name not set"); // 检查 deviceTypeToBackend_ 是否为空，如果为空则抛出异常
  return deviceTypeToBackend_.begin()->second->getGroupName(); // 返回第一个设备类型的 backend 的组名
}

// 设置 ProcessGroup 的组名
void ProcessGroup::setGroupName(const std::string& name) {
  for (auto& kv : deviceTypeToBackend_) {
    kv.second->setGroupName(name); // 设置所有设备类型的 backend 的组名
  }
}

// 获取 ProcessGroup 的组描述的常量引用
const std::string& ProcessGroup::getGroupDesc() const {
  return pg_desc_; // 返回组描述 pg_desc_
}

// 设置 ProcessGroup 的组描述
void ProcessGroup::setGroupDesc(const std::string& name) {
  pg_desc_ = name; // 设置组描述 pg_desc_
  // 设置所有设备类型的 backend 的组描述
  for (auto& kv : deviceTypeToBackend_) {
    kv.second->setGroupDesc(name);
  }
}

// 启用所有设备类型的 backend 的集体操作计时
void ProcessGroup::enableCollectivesTiming() {
  for (auto& kv : deviceTypeToBackend_) {
    kv.second->enableCollectivesTiming(); // 启用集体操作计时
  }
}

// 释放 ProcessGroup 的资源
void ProcessGroup::release_resources() {
  store_.reset(); // 重置 store_
  deviceTypeToBackend_.clear(); // 清空 deviceTypeToBackend_ 映射
  backendTypeToBackend_.clear(); // 清空 backendTypeToBackend_ 映射
}

} // namespace c10d // 结束 c10d 命名空间
```