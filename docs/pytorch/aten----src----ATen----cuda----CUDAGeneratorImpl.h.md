# `.\pytorch\aten\src\ATen\cuda\CUDAGeneratorImpl.h`

```
#pragma once

# 使用 `#pragma once` 指令，确保头文件只被编译一次，避免重复包含导致的编译错误和效率问题


#include <ATen/Context.h>
#include <ATen/core/Generator.h>
#include <ATen/core/TensorBase.h>
#include <ATen/cuda/PhiloxCudaState.h>

# 包含了几个 ATen 库的头文件，用于提供张量操作和 CUDA 加速计算的基础设施


#include <atomic>
#include <limits>
#include <memory>
#include <unordered_set>

# 包含了 C++ 标准库中的原子操作、数值极限、内存管理和无序集合等组件的头文件，用于实现并发安全、内存管理和数据结构支持


namespace at {

# 进入名为 `at` 的命名空间，用于将以下代码放置在 `at` 命名空间中


namespace cuda {
struct CUDAGraph;
}

# 在 `at` 命名空间内部声明了一个名为 `cuda` 的命名空间，并定义了一个名为 `CUDAGraph` 的结构体
/**
 * Note [CUDA Graph-safe RNG states]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *
 * Strategy:
 * ~~~~~~~~~
 * (It helps to look at
 * cuda/detail/PhiloxCudaStateRaw.cuh and
 * cuda/detail/UnpackRaw.cuh
 * while you read this.)
 *
 * A CUDA graph containing multiple RNG ops behaves like a
 * single giant kernel from the perspective of ops external
 * to the graph.  During graph capture, logic in CUDAGeneratorImpl
 * records the total of all offset increments that occur in the
 * graphed region, and records the final total as the offset for
 * the entire graph.
 *
 * When the graph reruns, the logic that reruns it
 * increments this device's CUDA generator's offset
 * by that total.
 *
 * Meanwhile, within the graph, at capture time, instead of
 * populating PhiloxCudaStates with the uint64_t offset pulled
 * directly from the global state, PhiloxCudaState uses a pointer
 * to a one-element stream-local int64_t device tensor
 * holding an initial offset value, and a uint64_t holding an
 * intra-graph offset. (The intra-graph offset starts from zero
 * when capture begins.)  In each consumer kernel,
 * at::cuda::philox::unpack computes the offset to use for this kernel
 * as intra-graph offset + *initial offset.
 *
 * When the graph reruns, the logic that reruns it first
 * fills the initial offset tensor with this device's
 * CUDA generator's current offset.
 *
 * The control flow above ensures graphed execution is bitwise
 * identical to eager execution as long as RNG ops are enqueued
 * from a single thread, even if RNG ops and graphs containing
 * RNG ops are enqueued and run simultaneously on multiple streams.
 *
 * Usage:
 * ~~~~~~
 * PhiloxCudaState in this file, and unpack() in
 * cuda/CUDAGraphsUtils.cuh allow non-divergent use of
 * CUDAGeneratorImpl whether graph capture is underway or not.
 *
 * Each PhiloxCudaState instance should be used for one and only one
 * consumer kernel.
 *
 * Example (see e.g. native/cuda/Dropout.cu):
 *
 * #include <ATen/cuda/CUDAGeneratorImpl.h>
 * #include <ATen/cuda/CUDAGraphsUtils.cuh>
 *
 * __global__ void kernel(..., PhiloxCudaState philox_args) {
 *   auto seeds = at::cuda::philox::unpack(philox_args);
 *   IndexType idx = blockIdx.x * blockDim.x + threadIdx.x;
 *   curandStatePhilox4_32_10_t state;
 *   curand_init(std::get<0>(seeds), // seed
 *               idx,                // per-thread subsequence
 *               std::get<1>(seeds), // offset in subsequence
 *               &state);
 *   ...
 * }
 *
 * host_caller(...) {
 *   PhiloxCudaState rng_engine_inputs;
 *   {
 *     // See Note [Acquire lock when using random generators]
 *     std::lock_guard<std::mutex> lock(gen->mutex_);
 *
 *     // gen could be HostState or DevState here! No divergent code needed!
 *     rng_engine_inputs = gen->philox_cuda_state(offset_increment);
 *   }
 *   kernel<<<...>>>(..., rng_engine_inputs);
 * }
 */
// 定义了一个结构体 CUDAGeneratorState，它继承自 c10::intrusive_ptr_target
struct CUDAGeneratorState : public c10::intrusive_ptr_target {
  uint64_t seed_; // 64位整数，用于生成随机数的种子
  uint64_t philox_offset_per_thread_; // 64位整数，每个线程的 Philox 算法偏移量
  uint32_t offset_intragraph_; // 32位整数，图内偏移量
  bool capturing_{}; // 布尔值，指示是否处于捕获状态
  std::unordered_set<cuda::CUDAGraph*> registered_graphs_; // 无序集合，存储注册的 CUDA 图对象
  at::TensorBase seed_extragraph_{}; // ATen 张量基类，用于外部图的种子
  at::TensorBase offset_extragraph_{}; // ATen 张量基类，用于外部图的偏移量

  // 构造函数，初始化成员变量
  CUDAGeneratorState(
      uint64_t seed = default_rng_seed_val,
      uint64_t philox_offset_per_thread = 0,
      uint32_t offset_intragraph = 0)
      : seed_(seed),
        philox_offset_per_thread_(philox_offset_per_thread),
        offset_intragraph_(offset_intragraph) {}

  // 增加种子的函数声明
  void increase(uint64_t increment);

  // 注册 CUDA 图对象的函数声明
  void register_graph(cuda::CUDAGraph* graph);
  // 注销 CUDA 图对象的函数声明
  void unregister_graph(cuda::CUDAGraph* graph);

  // 开始捕获状态的函数声明
  void capture_prologue();
  // 结束捕获状态的函数声明，并返回整个图的增量
  uint64_t capture_epilogue();
  // 重新播放状态的函数声明，使用整个图的增量
  void replay_prologue(uint64_t wholegraph_increment);
  // 克隆当前对象的函数声明
  c10::intrusive_ptr<CUDAGeneratorState> clone();
};

// 定义了一个结构体 CUDAGeneratorImpl，它继承自 c10::GeneratorImpl
struct TORCH_CUDA_CPP_API CUDAGeneratorImpl : public c10::GeneratorImpl {
  // 构造函数声明
  CUDAGeneratorImpl(DeviceIndex device_index = -1);
  CUDAGeneratorImpl(
      DeviceIndex device_index,
      c10::intrusive_ptr<CUDAGeneratorState> state_);
  ~CUDAGeneratorImpl() override = default;

  // 克隆当前对象的函数声明
  std::shared_ptr<CUDAGeneratorImpl> clone() const;
  // 设置当前种子的函数声明
  void set_current_seed(uint64_t seed) override;
  // 设置偏移量的函数声明
  void set_offset(uint64_t offset) override;
  // 获取偏移量的函数声明
  uint64_t get_offset() const override;
  // 获取当前种子的函数声明
  uint64_t current_seed() const override;
  // 获取种子的函数声明
  uint64_t seed() override;
  // 设置状态的函数声明，使用新状态作为参数
  void set_state(const c10::TensorImpl& new_state) override;
  // 获取状态的函数声明
  c10::intrusive_ptr<c10::TensorImpl> get_state() const override;
  // 设置图安全状态的函数声明，使用新状态作为参数
  void graphsafe_set_state(
      const c10::intrusive_ptr<GeneratorImpl>& state) override;
  // 获取图安全状态的函数声明
  c10::intrusive_ptr<c10::GeneratorImpl> graphsafe_get_state() const override;

  // 设置每线程的 Philox 算法偏移量的函数声明
  void set_philox_offset_per_thread(uint64_t offset);
  // 获取每线程的 Philox 算法偏移量的函数声明
  uint64_t philox_offset_per_thread() const;

  // 注册 CUDA 图对象的函数声明
  void register_graph(cuda::CUDAGraph* graph);
  // 注销 CUDA 图对象的函数声明
  void unregister_graph(cuda::CUDAGraph* graph);

  // 使用指定增量生成 PhiloxCudaState，并增加当前状态
  PhiloxCudaState philox_cuda_state(uint64_t increment);

  // 重置 RNN 状态的函数声明
  bool reset_rnn_state() {
    return !no_reset_rnn_state_.test_and_set();
  }

  // 临时支持使用 philox_engine_inputs 的调用位置
  // 允许逐步重构调用位置以使用 philox_cuda_state
  std::pair<uint64_t, uint64_t> philox_engine_inputs(uint64_t increment);

  // 静态方法，返回设备类型为 CUDA
  static c10::DeviceType device_type();

 private:
  // 克隆当前对象的私有实现函数声明
  CUDAGeneratorImpl* clone_impl() const override;

  c10::intrusive_ptr<CUDAGeneratorState> state_; // 指向 CUDAGeneratorState 的智能指针
  std::atomic_flag no_reset_rnn_state_; // 原子标志，用于 RNN 状态重置
};

// CUDA 的详细命名空间
namespace cuda::detail {

// 返回默认的 CUDA 生成器
TORCH_CUDA_CPP_API const Generator& getDefaultCUDAGenerator(
    DeviceIndex device_index = -1);
// 创建指定设备索引的 CUDA 生成器
TORCH_CUDA_CPP_API Generator createCUDAGenerator(DeviceIndex device_index = -1);

} // namespace cuda::detail
// namespace at 的结束
} // namespace at
```