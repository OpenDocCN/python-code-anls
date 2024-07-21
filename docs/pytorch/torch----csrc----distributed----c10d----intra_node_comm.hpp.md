# `.\pytorch\torch\csrc\distributed\c10d\intra_node_comm.hpp`

```py
#pragma once
// 预处理指令，确保此头文件只被编译一次

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/SymmetricMemory.hpp>
#include <torch/csrc/distributed/c10d/Work.hpp>

namespace c10d::intra_node_comm {

using namespace c10d::symmetric_memory;
// 命名空间声明，使用 c10d::symmetric_memory 命名空间中的符号

constexpr size_t kMaxDevices = 8;
// 声明一个常量，表示最大设备数为 8
constexpr size_t kDefaultBufferSize = 10ull * 1024 * 1024;
// 声明一个常量，表示默认缓冲区大小为 10MB
constexpr size_t kP2pStateSize = 2048;
// 声明一个常量，表示点对点状态大小为 2048

using NvlMesh = std::array<std::array<size_t, kMaxDevices>, kMaxDevices>;
// 定义 NvlMesh 类型为一个二维数组，大小为 kMaxDevices × kMaxDevices
using HybridCubeMesh = std::array<std::array<int, 4>, kMaxDevices>;
// 定义 HybridCubeMesh 类型为一个包含 kMaxDevices 个数组的二维数组，每个数组大小为 4

enum class Topology : uint8_t {
  UNKNOWN = 0,
  FULLY_CONNECTED = 1,
  HYBRID_CUBE_MESH = 2
};
// 枚举类型 Topology，表示通信拓扑结构，包括 UNKNOWN（未知）、FULLY_CONNECTED（全连接）、HYBRID_CUBE_MESH（混合立方体网格）

enum class AllReduceAlgo : uint8_t {
  NONE = 0,
  ONE_SHOT = 1,
  TWO_SHOT = 2,
  HCM = 3
};
// 枚举类型 AllReduceAlgo，表示全局归约算法的选择，包括 NONE、ONE_SHOT、TWO_SHOT、HCM

// NOTE: this class will be be removed soon in favor of SymmetricMemory
// 此类将很快被移除，建议使用 SymmetricMemory 替代
class TORCH_API IntraNodeComm : public c10::intrusive_ptr_target {
 public:
  // 公共构造函数，接受一个存储对象、当前进程的排名、世界大小和可选的缓冲区大小作为参数
  IntraNodeComm(
      c10::intrusive_ptr<c10d::Store> store,
      size_t rank,
      size_t worldSize,
      std::optional<size_t> bufferSize = c10::nullopt);

  // 析构函数，清理资源
  ~IntraNodeComm() override;

  // 静态函数，检查通信是否启用
  static bool isEnabled();

  /**
   * Performs rendezvous.
   * If rendezvous fails, the IntraNodeComm object will be in an invalid
   * state and it is the caller's responsibility to dispose it.
   */
  // 执行会合过程，如果会合失败，对象将处于无效状态，需要调用者负责释放它
  bool rendezvous();

  // 返回通信拓扑结构
  Topology getTopology() {
    return topology_;
  }

  // 返回缓冲区大小
  size_t getBufferSize() {
    return bufferSize_;
  }

  /**
   * Selects a AllReduceAlgo that we think will outperform nccl.
   * Returns AllReduceAlgo::NONE if we don't think we can outperform nccl.
   */
  // 选择一个我们认为会优于 nccl 的 AllReduce 算法，如果无法优于 nccl，则返回 AllReduceAlgo::NONE
  AllReduceAlgo selectAllReduceAlgo(const at::Tensor& input);

  // 执行全局归约操作
  at::Tensor allReduce(const at::Tensor& input, AllReduceAlgo algo);

  /**
   * Perform a barrier among the specified ranks.
   */
  // 执行指定排名之间的屏障操作
  void barrier(std::optional<std::vector<int64_t>> ranks = c10::nullopt);

  // 获取缓冲区
  at::Tensor getBuffer(
      size_t rank,
      const std::vector<int64_t>& sizes,
      c10::ScalarType dtype,
      int64_t storageOffset);

 private:
  // 执行一次性全局归约操作
  at::Tensor oneShotAllReduce(
      const at::Tensor& input,
      at::cuda::CUDAStream& stream);

  // 执行两次式全局归约操作
  at::Tensor twoShotAllReduce(
      const at::Tensor& input,
      at::cuda::CUDAStream& stream);

  // 执行混合立方体网格全局归约操作
  at::Tensor hybridCubeMeshAllReduce(
      const at::Tensor& input,
      at::cuda::CUDAStream& stream);

  // 存储对象的指针
  c10::intrusive_ptr<Store> store_;

  // 当前进程的排名
  size_t rank_;

  // 总进程数
  size_t worldSize_;

  // 缓冲区大小
  size_t bufferSize_;

  // CUDA 事件，用于屏障同步
  at::cuda::CUDAEvent barrierReady_;

  /**
   * Members initialized after rendezvous
   */

  // 标记是否初始化完成
  bool isInitialized_ = false;

  // 通信拓扑结构
  Topology topology_ = Topology::UNKNOWN;

  // 对称内存指针
  void* symmetricMemoryPtr_ = nullptr;

  // 对称内存对象
  c10::intrusive_ptr<SymmetricMemory> symmetricMemory_ = nullptr;

  // 点对点状态设备指针
  void* p2pStatesDev_{};

  // 缓冲区设备指针
  void* buffersDev_{};

  // 拓扑信息指针
  void* topoInfo_{};
};
/**
 * NOTE [IntraNodeComm Stream Semantics]
 *
 * ProcessGroupNCCL launches kernels differently from the conventional PyTorch
 * CUDA semantics: it always launches collective kernels onto a dedicated
 * communication stream. Therefore, it needs to:
 *
 * - Synchronize the calling stream and the comm stream.
 * - Ensure the memory safety of the operands (via record_stream or stashing).
 * - Synchronize the waiting stream with the comm stream.
 *
 * Unconditionally performing these tasks makes sense when we expect most of the
 * communication to benefit from compute/comm overlap. However, IntraNodeComm
 * primarily aims to optimize small, latency-sensitive, blocking communication,
 * in which the overhead incurred by the above steps can be quite pronounced.
 *
 * Thus, IntraNodeComm follows the conventional PyTorch CUDA semantics and
 * launches kernels onto the stream specified by the user. Although the user
 * can perform necessary synchronization via wait_stream, to provide a UX
 * consistent to that of ProcessGroupNCCL, the necessary stream
 * synchronization can also be performed via IntraNodeWork::wait().
 */
class IntraNodeCommWork : public c10d::Work {
 public:
  /**
   * Constructor for IntraNodeCommWork.
   * Initializes the CUDA event to record the completion of work.
   */
  IntraNodeCommWork() : c10d::Work() {
    event_.record();  // Record the completion of work using a CUDA event.
  }

  /**
   * Wait for the completion of the IntraNodeCommWork.
   * Blocks the calling thread until the CUDA event is completed on the current CUDA stream.
   * @param timeout Optional timeout value in milliseconds (default: kNoTimeout).
   * @return Always returns true indicating successful completion.
   */
  bool wait(std::chrono::milliseconds timeout = kNoTimeout) override {
    event_.block(at::cuda::getCurrentCUDAStream());  // Block until the CUDA event completes on the current stream.
    return true;  // Return true indicating successful completion.
  }

 private:
  at::cuda::CUDAEvent event_;  // CUDA event used for synchronization.
};

/**
 * Retrieves the usage counter for IntraNodeComm.
 * @return An integer representing the usage counter.
 */
TORCH_API int64_t getIntraNodeCommUsageCounter();

} // namespace c10d::intra_node_comm
```