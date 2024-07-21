# `.\pytorch\torch\csrc\distributed\c10d\intra_node_comm.cpp`

```
/**
 * 包含所需的头文件
 */
#include <torch/csrc/distributed/c10d/intra_node_comm.hpp>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Logging.h>
#include <torch/csrc/distributed/c10d/Utils.hpp>

#include <iostream>
#include <utility>

#include <fcntl.h>
#include <pthread.h>
#include <semaphore.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#if !defined(USE_ROCM) && defined(PYTORCH_C10_DRIVER_API_SUPPORTED)
#include <c10/cuda/driver_api.h>
#include <nvml.h>
#endif

#include <cuda_runtime.h>

/**
 * 定义命名空间 c10d::intra_node_comm
 */
namespace c10d::intra_node_comm {

/**
 * 静态变量，用于控制是否启用节点内通信
 */
static std::vector<std::string> ENABLE_INTRA_NODE_COMM = {
    "ENABLE_INTRA_NODE_COMM"};

/**
 * 静态变量，用于测试目的，强制 detectedTopology() 返回 Topology::FULLY_CONNECTED，
 * 即使没有 NVLink 连接。
 */
static std::vector<std::string> TEST_INTRA_NODE_COMM = {"TEST_INTRA_NODE_COMM"};

////////////////////////////////////////////////////////////////////////////////
// CUDA Functions
////////////////////////////////////////////////////////////////////////////////

/**
 * 检查是否支持节点内通信的 CUDA 函数
 */
bool isIntraNodeCommSupported();

/**
 * 获取混合立方体网格的可选值，基于给定的 NvlMesh
 */
std::optional<HybridCubeMesh> getHybridCubeMesh(NvlMesh nvlMesh);

/**
 * 初始化点对点状态，返回 void 指针
 */
void* initP2pState();

/**
 * 初始化拓扑信息，基于给定的拓扑结构、NvlMesh 和排名
 */
void* initTopoInfo(Topology topology, NvlMesh nvlMesh, size_t rank);

////////////////////////////////////////////////////////////////////////////////
// Topology Detection
////////////////////////////////////////////////////////////////////////////////

/**
 * 重载运算符 << ，用于将 NvlMesh 输出到流中
 */
static std::ostream& operator<<(std::ostream& os, const NvlMesh& nvlMesh) {
  std::ostringstream oss;
  for (size_t i = 0; i < kMaxDevices; ++i) {
    for (size_t j = 0; j < kMaxDevices; ++j) {
      oss << nvlMesh[i][j] << " ";
    }
    oss << '\n';
  }
  os << oss.str();
  return os;
}

/**
 * 检查两个 NvlMesh 是否相同
 */
static bool isSame(NvlMesh lhs, NvlMesh rhs) {
  for (size_t i = 0; i < kMaxDevices; ++i) {
    for (size_t j = 0; j < kMaxDevices; ++j) {
      if (lhs[i][j] != rhs[i][j]) {
        return false;
      }
    }
  }
  return true;
}

/**
 * 查询设备间的 nvlink 连接，返回 NvlMesh
 */
static NvlMesh getNvlMesh(const std::vector<std::string>& rankToBusId) {
#if !defined(USE_ROCM) && defined(PYTORCH_C10_DRIVER_API_SUPPORTED)
  using namespace c10::cuda;

  NvlMesh nvlMesh = {};
  auto driverApi = DriverAPI::get();
  if (driverApi == nullptr) {
    return nvlMesh;
  }

  const auto worldSize = rankToBusId.size();
  std::vector<nvmlDevice_t> devices(worldSize, nullptr);
  std::unordered_map<std::string, size_t> busIdToRank;
  std::vector<size_t> switchLinkCount(worldSize, 0);

  for (size_t r = 0; r < worldSize; ++r) {
    busIdToRank.emplace(rankToBusId[r], r);
    TORCH_CHECK(
        driverApi->nvmlDeviceGetHandleByPciBusId_v2_(
            rankToBusId[r].c_str(), &devices[r]) == NVML_SUCCESS);
  }

  // TODO: find a better way to determine this
  constexpr size_t kMaxNvLinks = 20;

  // For each device, loop over devices connected to it via NVLink
  for (size_t idx = 0; idx < worldSize; ++idx) {
    // 遍历每个设备的 NVLink 连接，最多不超过 kMaxNvLinks 个
    for (size_t link = 0; link < kMaxNvLinks; ++link) {
      nvmlReturn_t ret;
      nvmlIntNvLinkDeviceType_t deviceType;
      // 获取指定设备和链接上远程设备的 NVLink 类型
      ret = driverApi->nvmlDeviceGetNvLinkRemoteDeviceType_(
          devices[idx], link, &deviceType);
      if (ret != NVML_SUCCESS) {
        // 如果 API 调用失败，表示没有更多的 NVLink 连接可用
        // 这是一种正常情况，因为无法可靠获取可以传递给 API 的最大链接值
        // 因此，我们简单地递增链接值，直到 API 失败或达到预定义的最大值
        break;
      }
      // 远程设备是 GPU
      if (deviceType == NVML_NVLINK_DEVICE_TYPE_GPU) {
        nvmlPciInfo_t pciInfo;
        // 获取远程 GPU 设备的 PCI 信息
        ret = driverApi->nvmlDeviceGetNvLinkRemotePciInfo_v2_(
            devices[idx], link, &pciInfo);
        if (ret != NVML_SUCCESS) {
          // 出现意外错误，返回空的 NvlMesh
          return {};
        }
        auto it = busIdToRank.find(pciInfo.busId);
        // 如果远程设备的 busId 在 busIdToRank 中能找到对应的索引
        if (it != busIdToRank.end()) {
          // 如果当前设备索引不等于 busId 对应的索引
          if (idx != it->second) {
            // 在 nvlMesh 中增加连接计数
            nvlMesh[idx][it->second] += 1;
          }
        }
        // 远程设备是 NVSwitch
      } else if (deviceType == NVML_NVLINK_DEVICE_TYPE_SWITCH) {
        // 增加与 NVSwitch 相关的链接计数
        switchLinkCount[idx] += 1;
      }
    }
  }
  // 处理 NVSwitch 连接。简化起见，假设所有的 NVSwitch 互相连接。
  for (size_t i = 0; i < worldSize; ++i) {
    for (size_t j = 0; j < worldSize; ++j) {
      // 跳过自身连接
      if (i == j) {
        continue;
      }
      // 更新 nvlMesh[i][j]，增加最小的 NVSwitch 连接计数
      nvlMesh[i][j] += std::min(switchLinkCount[i], switchLinkCount[j]);
    }
  }
  // 返回计算后的 nvlMesh
  return nvlMesh;
#else
  // 如果上述条件都不满足，则返回一个空的字典
  return {};
#endif
}

/**
 * Determine if the devices form a hybrid cube mesh
 * topology given a NvlMesh.
 */
static bool isHybridCubeMesh(const NvlMesh nvlMesh) {
  // 初始化每个设备的邻居数量为0
  std::array<size_t, kMaxDevices> numNeighbors = {};
  // 遍历设备列表
  for (size_t i = 0; i < kMaxDevices; ++i) {
    // 遍历每个设备的连接情况
    for (size_t j = 0; j < kMaxDevices; ++j) {
      // 如果设备 i 和设备 j 之间有连接
      if (nvlMesh[i][j] > 0) {
        // 增加设备 i 的邻居数量
        numNeighbors[i] += 1;
      }
    }
  }
  // 检查每个设备的邻居数量是否为4
  for (size_t i = 0; i < kMaxDevices; ++i) {
    // TODO: this is insufficent and needs revisit
    // 如果任何一个设备的邻居数量不为4，则返回 false
    if (numNeighbors[i] != 4) {
      return false;
    }
  }
  // 如果所有设备的邻居数量都为4，则返回 true
  return true;
}

/**
 * Detect topology given a NvlMesh.
 */
static Topology detectTopology(const NvlMesh nvlMesh, size_t worldSize) {
  // 如果启用了测试节点内通信选项，则返回 FULLY_CONNECTED 拓扑
  if (getCvarBool(TEST_INTRA_NODE_COMM, false)) {
    return Topology::FULLY_CONNECTED;
  }
  // 初始化一个变量，用于检查是否所有设备都完全连接
  bool fullyConnected = true;
  // 检查每对设备之间的连接情况
  for (size_t i = 0; i < worldSize - 1; ++i) {
    for (size_t j = i + 1; j < worldSize; ++j) {
      // 如果设备 i 和设备 j 之间没有双向连接，则标记为未完全连接
      if (nvlMesh[i][j] == 0 || nvlMesh[j][i] == 0) {
        fullyConnected = false;
      }
    }
  }
  // 如果所有设备都完全连接，则返回 FULLY_CONNECTED 拓扑
  if (fullyConnected) {
    LOG(INFO) << "IntraNodeComm: Topology::FULLY_CONNECTED";
    return Topology::FULLY_CONNECTED;
  }
  // 如果设备数量等于 kMaxDevices，并且形成了混合立方体网格，则返回 HYBRID_CUBE_MESH 拓扑
  if (worldSize == kMaxDevices && getHybridCubeMesh(nvlMesh) != std::nullopt) {
    LOG(INFO) << "IntraNodeComm: Topology::HYBRID_CUBE_MESH";
    return Topology::HYBRID_CUBE_MESH;
  }
  // 默认情况下，返回未知的拓扑类型
  LOG(INFO) << "IntraNodeComm: Topology::UNKNOWN";
  return Topology::UNKNOWN;
};

////////////////////////////////////////////////////////////////////////////////
// Rendezvous and Initialization
////////////////////////////////////////////////////////////////////////////////

IntraNodeComm::IntraNodeComm(
    c10::intrusive_ptr<c10d::Store> store,
    size_t rank,
    size_t worldSize,
    std::optional<size_t> bufferSize)
    : store_(std::move(store)),
      rank_(rank),
      worldSize_(worldSize),
      bufferSize_(bufferSize.has_value() ? *bufferSize : kDefaultBufferSize),
      barrierReady_(at::cuda::CUDAEvent()) {}

IntraNodeComm::~IntraNodeComm() {
  // 如果未初始化，则直接返回，不进行清理工作
  if (!isInitialized_) {
    return;
  }
  // 获取 CUDA 设备的内存分配器，并释放对称内存指针
  auto allocator = get_allocator(c10::DeviceType::CUDA);
  allocator->free(symmetricMemoryPtr_);
}

bool IntraNodeComm::isEnabled() {
  // 检查是否启用了节点内通信
  return getCvarBool(ENABLE_INTRA_NODE_COMM, false);
}

/**
 * Use c10d::Store to perform allgather on a trivially copyable type.
 */
template <typename T>
std::vector<T> storeAllGather(
    const c10::intrusive_ptr<c10d::Store>& store,
    const std::string& prefix,
    size_t rank,
    size_t worldSize,
    T val) {
  // 断言类型 T 是可以平凡复制的
  static_assert(std::is_trivially_copyable_v<T>);

  // 创建用于存储所有 peer 的 key 的数组
  std::vector<std::string> peerKeys;
  for (size_t r = 0; r < worldSize; ++r) {
    // 每个 peer 的 key 是 prefix 加上其索引
    std::ostringstream oss;
    oss << prefix << "-" << r;
    peerKeys.push_back(oss.str());
  }

  {
    // 将 val 转换为字节数组，存储到 store 中
    std::vector<uint8_t> payload(
        reinterpret_cast<uint8_t*>(&val),
        reinterpret_cast<uint8_t*>(&val) + sizeof(T));
    store->set(peerKeys[rank], payload);
  }

  // 创建用于存储所有 peer 值的数组
  std::vector<T> peerVals;
  for (size_t r = 0; r < worldSize; ++r) {
    // 如果当前迭代器r等于rank
    if (r == rank) {
      // 将val添加到peerVals的末尾
      peerVals.push_back(val);
      // 跳过当前循环，继续下一次迭代
      continue;
    }
    // 等待存储实例store完成对peerKeys[r]的操作
    store->wait({peerKeys[r]});
    // 获取存储实例store中peerKeys[r]对应的数据
    auto payload = store->get(peerKeys[r]);
    // 检查payload的大小是否等于类型T的大小
    TORCH_CHECK(payload.size() == sizeof(T));
    // 声明并初始化类型T的变量peerVal
    T peerVal{};
    // 将payload的数据拷贝到peerVal中，拷贝的长度为sizeof(T)
    std::memcpy(&peerVal, payload.data(), sizeof(T));
    // 将peerVal添加到peerVals的末尾
    peerVals.push_back(peerVal);
  }
  // 返回peerVals向量，其中包含了所有peerVal的值
  return peerVals;
bool IntraNodeComm::rendezvous() {
  // 如果已经初始化过，则直接返回 true
  if (isInitialized_) {
    return true;
  }

  // 如果不支持节点内通信或者节点数量小于 2 或者超过最大设备数，则返回 false
#if !defined(USE_ROCM) && defined(PYTORCH_C10_DRIVER_API_SUPPORTED)
  if (!isIntraNodeCommSupported() || worldSize_ < 2 ||
      worldSize_ > kMaxDevices) {
    return false;
  }

  // 获取当前 CUDA 设备索引并设置 CUDA guard
  auto deviceIdx = at::cuda::current_device();
  c10::cuda::CUDAGuard guard(deviceIdx);

  // 第一个握手：交换主机名和设备总线 ID
  struct DevInfo {
    char hostname[HOST_NAME_MAX + 1]; // 主机名缓冲区
    char busId[80]; // 设备总线 ID 缓冲区
  };

  DevInfo devInfo{};
  gethostname(devInfo.hostname, sizeof(devInfo.hostname)); // 获取本地主机名
  cudaDeviceProp prop{};
  AT_CUDA_CHECK(cudaGetDeviceProperties(&prop, deviceIdx)); // 获取当前设备的属性
  // 格式化设备总线 ID
  snprintf(
      devInfo.busId,
      sizeof(devInfo.busId),
      NVML_DEVICE_PCI_BUS_ID_FMT,
      prop.pciDomainID,
      prop.pciBusID,
      prop.pciDeviceID);

  // 使用 storeAllGather 交换握手信息
  auto peerDevInfos =
      storeAllGather(store_, "handshake-0", rank_, worldSize_, devInfo);

  // 检查所有参与者是否在同一主机上，并收集设备总线 ID
  std::vector<std::string> rankToBusId;
  for (const auto& info : peerDevInfos) {
    if (strcmp(info.hostname, peerDevInfos.front().hostname) != 0) {
      LOG(WARNING) << "Aborting IntraNodeComm::rendezvous because some "
                      "participants are not on the same host ("
                   << info.hostname << ", " << devInfo.hostname << ")";
      return false;
    }
    rankToBusId.emplace_back(info.busId);
  }

  // 验证设备总线 ID 的唯一性
  {
    std::unordered_set uniqueBusIds(rankToBusId.begin(), rankToBusId.end());
    TORCH_CHECK(
        uniqueBusIds.size() == worldSize_,
        "IntraNodeComm::rendezvous: detected overlapping devices across ranks. "
        "Please properly set device via torch.cuda.set_device() before "
        "initiating rendezvous.");
  }

  // 查询 nvlink 连接
  auto nvlMesh = getNvlMesh(rankToBusId);

  // 检测拓扑结构
  Topology topology = detectTopology(nvlMesh, worldSize_);

  // 设置组信息
  set_group_info("IntraNodeComm", rank_, worldSize_, store_);

  // 获取 CUDA 分配器并分配对称内存
  auto allocator = get_allocator(c10::DeviceType::CUDA);
  symmetricMemoryPtr_ =
      allocator->alloc(bufferSize_, deviceIdx, "IntraNodeComm");
  symmetricMemory_ = allocator->rendezvous(symmetricMemoryPtr_);

  // 检查对称内存的信号填充大小
  TORCH_CHECK(symmetricMemory_->get_signal_pad_size() >= kP2pStateSize);

  // 初始化拓扑信息
  void* topoInfo = initTopoInfo(topology, nvlMesh, rank_);

  // 设置初始化标志和相关成员变量
  isInitialized_ = true;
  topology_ = topology;
  p2pStatesDev_ = symmetricMemory_->get_signal_pad_ptrs_dev();
  buffersDev_ = symmetricMemory_->get_buffer_ptrs_dev();
  topoInfo_ = topoInfo;

  // 初始化完成，返回 true
  return true;
#endif

  // 如果未满足预处理条件，则返回 false
  return false;
}
```