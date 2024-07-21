# `.\pytorch\aten\src\ATen\cuda\CUDAGeneratorImpl.cpp`

```py
// 包含 ATen 库中所需的头文件
#include <ATen/Functions.h>
#include <ATen/Tensor.h>
#include <ATen/Utils.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/CUDAGraph.h>
#include <ATen/cuda/CUDAGraphsUtils.cuh>
#include <c10/core/StreamGuard.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/util/CallOnce.h>
#include <deque> // 包含标准库 deque 头文件

// 定义在 at 命名空间下的 cuda::detail 命名空间
namespace at {
namespace cuda::detail {

namespace {

// 保证 cudaGetDeviceCount 只被调用一次的标志
static c10::once_flag num_gpu_init_flag;

// 系统中 GPU 的总数
static int64_t num_gpus;

// 保证 default_gens_cuda 仅被初始化一次的标志队列
static std::deque<c10::once_flag> cuda_gens_init_flag;

// 默认的全局 CUDA 生成器，每个 GPU 一个
static std::vector<Generator> default_gens_cuda;

/*
 * 初始化与 CUDA 生成器相关的全局变量
 * 警告：此函数只能调用一次！
 */
static void initCUDAGenVector() {
  // 获取系统中 CUDA 设备的数量
  num_gpus = c10::cuda::device_count();
  // 调整 cuda_gens_init_flag 的大小以匹配 GPU 数量
  cuda_gens_init_flag.resize(num_gpus);
  // 调整 default_gens_cuda 的大小以匹配 GPU 数量
  default_gens_cuda.resize(num_gpus);
}

} // 匿名命名空间结束

/**
 * PyTorch 维护一组默认生成器，仅在初始化时被初始化一次。
 * 这些默认生成器的目的是在用户未显式指定任何生成器时，
 * 维护伪随机数生成的全局运行状态。
 * getDefaultCUDAGenerator 获取特定 CUDA 设备的默认生成器。
 */
const Generator& getDefaultCUDAGenerator(DeviceIndex device_index) {
  // 确保 initCUDAGenVector 函数只被调用一次
  c10::call_once(num_gpu_init_flag, initCUDAGenVector);
  // 获取当前设备索引
  DeviceIndex idx = device_index;
  // 如果设备索引为 -1，则使用当前 CUDA 设备索引
  if (idx == -1) {
    idx = c10::cuda::current_device();
  } else {
    // 否则，检查设备索引的有效性
    TORCH_CHECK(idx >= 0 && idx < num_gpus);
  }
  // 保证每个设备只初始化一次生成器
  c10::call_once(cuda_gens_init_flag[idx], [&] {
    // 创建一个新的 CUDA 生成器实例
    default_gens_cuda[idx] = make_generator<CUDAGeneratorImpl>(idx);
    // 设置生成器的种子
    default_gens_cuda[idx].seed();
  });
  // 返回特定设备的默认 CUDA 生成器
  return default_gens_cuda[idx];
}

/**
 * 创建一个 CUDAGeneratorImpl 的实例并返回共享指针
 */
Generator createCUDAGenerator(DeviceIndex device_index) {
  // 确保 initCUDAGenVector 函数只被调用一次
  c10::call_once(num_gpu_init_flag, initCUDAGenVector);
  // 获取当前设备索引
  DeviceIndex idx = device_index;
  // 如果设备索引为 -1，则使用当前 CUDA 设备索引
  if (idx == -1) {
    idx = c10::cuda::current_device();
  }
  // 检查设备索引的有效性
  TORCH_CHECK(idx >= 0 && idx < num_gpus, "The device_index is invalid.");
  // 创建一个新的 CUDA 生成器实例
  auto gen = make_generator<CUDAGeneratorImpl>(idx);
  auto cuda_gen = check_generator<CUDAGeneratorImpl>(gen);
  // 设置生成器的当前种子值
  cuda_gen->set_current_seed(default_rng_seed_val);
  // 设置每个线程的 Philox 偏移量
  cuda_gen->set_philox_offset_per_thread(0);
  // 返回生成器实例
  return gen;
}

} // namespace cuda::detail

/**
 * 创建此 CUDA 生成器状态的克隆。
 */
c10::intrusive_ptr<CUDAGeneratorState> CUDAGeneratorState::clone() {
  // 返回一个克隆的 CUDA 生成器状态对象
  return make_intrusive<CUDAGeneratorState>(
      seed_, philox_offset_per_thread_, offset_intragraph_);
}

/**
 * 基于指定的增量增加内部偏移量的函数。
 */
/**
 * Increases the offset by a specified amount, ensuring alignment and checking
 * constraints based on whether CUDA graph capturing is active or not.
 */
void CUDAGeneratorState::increase(uint64_t increment) {
  // Rounds increment up to the nearest multiple of 4 to meet alignment
  // requirements.
  // see Note [Why enforce RNG offset % 4 == 0?]
  increment = ((increment + 3) / 4) * 4;

  // Handling different behaviors based on whether capturing is active.
  if (at::cuda::currentStreamCaptureStatus() != at::cuda::CaptureStatus::None) {
    // Ensures that the state is actually capturing.
    TORCH_CHECK(
        capturing_,
        "Attempt to increase offset for a CUDA generator not in capture mode.");
    // Ensures the offset is a multiple of 4
    // see Note [Why enforce RNG offset % 4 == 0?]
    TORCH_INTERNAL_ASSERT(
        offset_intragraph_ % 4 == 0, "RNG offset must be a multiple of 4.");
    // Ensures the increment does not cause overflow.
    TORCH_INTERNAL_ASSERT(
        offset_intragraph_ <= std::numeric_limits<uint32_t>::max() - increment,
        "Increment causes overflow in the offset value.");
    offset_intragraph_ += increment;
  } else {
    // Checks that the increment is expected outside graph capturing.
    TORCH_CHECK(
        !capturing_,
        "Offset increment outside graph capture encountered unexpectedly.");
    // Ensures the offset is a multiple of 4
    // see Note [Why enforce RNG offset % 4 == 0?]
    TORCH_INTERNAL_ASSERT(
        philox_offset_per_thread_ % 4 == 0,
        "RNG offset must be a multiple of 4.");
    philox_offset_per_thread_ += increment;
  }
}

/**
 * Registers this state to a CUDA graph to manage within the graph.
 */
void CUDAGeneratorState::register_graph(cuda::CUDAGraph* graph) {
  // Ensures that the RNG state is not currently being captured.
  at::cuda::assertNotCapturing(
      "Cannot register the state during capturing stage.");

  // If this is the first graph to be registered, allocate memory for the seed
  // and offset on the GPU.
  if (registered_graphs_.empty()) {
    auto options = at::TensorOptions().device(at::kCUDA).dtype(at::kLong);
    seed_extragraph_ = at::empty({1}, options);
    offset_extragraph_ = at::empty({1}, options);
  }

  // Insert the graph into the set of registered graphs if it's not already
  // registered.
  if (registered_graphs_.find(graph) == registered_graphs_.end()) {
    registered_graphs_.insert(graph);
  }
}

/**
 * Unregisters a CUDA graph from the RNG state.
 */
void CUDAGeneratorState::unregister_graph(cuda::CUDAGraph* graph) {
  // Verify the graph was previously registered.
  TORCH_CHECK(
      registered_graphs_.find(graph) != registered_graphs_.end(),
      "The graph should be registered to the state");

  // Remove the graph from the set of registered graphs.
  registered_graphs_.erase(graph);

  // If no more graphs are registered, deallocate the GPU memory for the seed
  // and offset.
  if (registered_graphs_.empty()) {
    seed_extragraph_.reset();
    offset_extragraph_.reset();
  }
}
/**
 * Note [Explicit Registration of Generators to the CUDA Graph]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *
 * Ideally, it would be more user-friendly if the state could be exchanged and generators
 * could be registered with the CUDA graph implicitly. However, resetting GPU tensors during
 * the capture stage causes these reset operations to be recorded within the CUDA graph.
 * This behavior is undesirable because we do not want these tensors to be reset during
 * the replay stage of the graph.
 *
 * As of now, there is no available method to perform a CUDA operation during the graph's
 * recording phase without having that operation be included in the CUDA graph.
 * This limitation necessitates explicit user action to register generators with the graph.
 * By requiring users to manually register their generators, we can ensure that state resets
 * (capture_prologue) only occur before the graph capture begins, thus avoiding unintended
 * resets during the replay of the graph. See https://github.com/pytorch/pytorch/pull/114068.
 */

/**
 * Performs the prologue steps for capturing a CUDA graph state.
 * This method is intended to reset graph-related state variables before capturing begins.
 */
void CUDAGeneratorState::capture_prologue() {
  capturing_ = true;  // 设置捕获标志为true，表明正在进行捕获阶段
  offset_intragraph_ = 0;  // 将图内偏移量设置为0，准备开始捕获
  seed_extragraph_.fill_(int64_t(seed_));  // 使用当前种子填充图外种子状态张量
  offset_extragraph_.fill_(int64_t(0));  // 将图外偏移量设置为0
}

/**
 * Ends the capturing phase and resets related variables, returning the whole
 * graph increment.
 */
uint64_t CUDAGeneratorState::capture_epilogue() {
  capturing_ = false;  // 结束捕获阶段，将捕获标志设为false
  return offset_intragraph_;  // 返回图内偏移量，表示整个图的增量
}

/**
 * Prepares the state for replay by setting initial state tensors and applying
 * total increment.
 */
void CUDAGeneratorState::replay_prologue(uint64_t wholegraph_increment) {
  // Ensures the generator is not in capturing mode.
  at::cuda::assertNotCapturing(
      "Cannot prepare for replay during capturing stage.");  // 确保生成器不处于捕获模式
  seed_extragraph_.fill_(int64_t(seed_));  // 使用当前种子填充图外种子状态张量
  offset_extragraph_.fill_(int64_t(philox_offset_per_thread_));  // 使用每个线程的Philox偏移量填充图外偏移量张量
  // Applies the total increment achieved during previous captures to update the
  // offset.
  increase(wholegraph_increment);  // 将之前捕获阶段获得的总增量应用于更新偏移量
}

/**
 * Note [Why enforce RNG offset % 4 == 0?]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * Curand philox does allow offsets that aren't a multiple of 4.
 * But jit kernels don't use curand, they use a custom "Philox" class (see
 * torch/csrc/jit/tensorexpr/cuda_random.h or
 * torch/csrc/jit/codegen/cuda/runtime/random_numbers.cu).
 * The "Philox" constructor computes offset/4 (a uint64_t division) to locate its
 * internal start in its virtual bitstream viewed as 128-bit chunks, then, when called
 * in a thread, returns one 32-bit chunk at a time from that start in the bitstream.
 * In other words, if the incoming offset is not a multiple of 4, each thread
 * might repeat some previously-generated 32-bit values in the bitstream. See
 * https://github.com/pytorch/pytorch/pull/50169.
 */
/**
 * CUDAGeneratorImpl 类的实现
 */
CUDAGeneratorImpl::CUDAGeneratorImpl(DeviceIndex device_index)
  : c10::GeneratorImpl{Device(DeviceType::CUDA, device_index),
          DispatchKeySet(c10::DispatchKey::CUDA)} {
  // 在构造函数中设置设备类型和分发键集合
  at::cuda::assertNotCapturing("Cannot construct a new CUDAGeneratorImpl");
  // 创建 CUDAGeneratorState 对象并将其赋给 state_
  state_ = make_intrusive<CUDAGeneratorState>();
  // 清空 no_reset_rnn_state_ 容器
  no_reset_rnn_state_.clear();
}

CUDAGeneratorImpl::CUDAGeneratorImpl(
    DeviceIndex device_index,
    c10::intrusive_ptr<CUDAGeneratorState> state)
    : c10::
          GeneratorImpl{Device(DeviceType::CUDA, device_index), DispatchKeySet(c10::DispatchKey::CUDA)},
      state_(std::move(state)) {
  // 设置设备类型和分发键集合，并初始化 state_ 为给定的状态
  no_reset_rnn_state_.clear();
}

/**
 * 设置当前的种子，用于 curandStatePhilox4_32_10
 * 将 philox_offset_per_thread_ 重置为 0
 *
 * See Note [Acquire lock when using random generators]
 */
void CUDAGeneratorImpl::set_current_seed(uint64_t seed) {
  // 检查是否在捕获状态，如果是则抛出异常
  at::cuda::assertNotCapturing(
      "Cannot call CUDAGeneratorImpl::set_current_seed");
  // 设置状态的种子值为给定的种子
  state_->seed_ = seed;
  // 重置每个线程的 philox_offset_per_thread_ 为 0
  state_->philox_offset_per_thread_ = 0;
  // 清空 no_reset_rnn_state_ 容器
  no_reset_rnn_state_.clear();
}

/**
 * 设置用于 curandStatePhilox4_32_10 的偏移量
 *
 * See Note [Acquire lock when using random generators]
 */
void CUDAGeneratorImpl::set_offset(uint64_t offset) {
  // 检查是否在捕获状态，如果是则抛出异常
  at::cuda::assertNotCapturing("Cannot call CUDAGeneratorImpl::set_offset");
  // 调用 set_philox_offset_per_thread() 设置偏移量，并检查是否为 4 的倍数
  set_philox_offset_per_thread(offset);
  // 清空 no_reset_rnn_state_ 容器
  no_reset_rnn_state_.clear();
}

/**
 * 获取 CUDAGeneratorImpl 的当前偏移量
 */
uint64_t CUDAGeneratorImpl::get_offset() const {
  // 在捕获状态下禁止调用 get_offset()
  at::cuda::assertNotCapturing("Cannot call CUDAGeneratorImpl::get_offset");
  // 返回当前线程的 philox_offset_per_thread_
  return state_->philox_offset_per_thread_;
}

/**
 * 获取 CUDAGeneratorImpl 的当前种子
 */
uint64_t CUDAGeneratorImpl::current_seed() const {
  // 在捕获状态下禁止调用 current_seed()
  at::cuda::assertNotCapturing("Cannot call CUDAGeneratorImpl::current_seed");
  // 返回当前状态的种子值
  return state_->seed_;
}

/**
 * 从 /dev/urandom 或时间获取一个非确定性随机数，
 * 使用该随机数设置 CPUGeneratorImpl 的种子并返回该随机数
 *
 * FIXME: 如果 getNonDeterministicRandom 的算法对 CPU 和 CUDA 统一，可以将此函数移动到 Generator.cpp
 */
uint64_t CUDAGeneratorImpl::seed() {
  // 在捕获状态下禁止调用 seed()
  at::cuda::assertNotCapturing("Cannot call CUDAGeneratorImpl::seed");
  // 获取一个非确定性的随机数
  auto random = c10::detail::getNonDeterministicRandom(true);
  // 使用随机数设置当前的种子
  this->set_current_seed(random);
  // 返回生成的随机数
  return random;
}

/**
 * 获取 CUDAGeneratorImpl 的当前内部状态，返回一个 CPU 字节张量
 */
// 返回当前 CUDA 生成器状态的 RNG 状态。该状态包括种子和用于 Philox 的偏移量。
c10::intrusive_ptr<c10::TensorImpl> CUDAGeneratorImpl::get_state() const {
  // 定义 RNG 状态的大小，包括种子和 Philox 使用的偏移量
  static const size_t seed_size = sizeof(uint64_t);
  static const size_t offset_size = sizeof(int64_t);
  static const size_t total_size = seed_size + offset_size;

  // 创建一个空的 CPU 字节张量来存储状态
  auto state_tensor = at::detail::empty_cpu({(int64_t)total_size}, ScalarType::Byte, c10::nullopt, c10::nullopt, c10::nullopt, c10::nullopt);
  // 获取指向 RNG 状态数据的指针
  auto rng_state = state_tensor.data_ptr<uint8_t>();
  // 获取当前的种子值
  auto current_seed = this->current_seed();
  // 获取当前的 Philox 偏移量，并转换为 int64_t 类型
  auto offset = static_cast<int64_t>(this->philox_offset_per_thread());
  // 将当前种子值和偏移量复制到 RNG 状态的内存中
  memcpy(rng_state, &current_seed, seed_size);
  memcpy(rng_state + seed_size, &offset, offset_size);

  // 返回状态张量的侵入式指针
  return state_tensor.getIntrusivePtr();
}

/**
 * 设置 CUDAGeneratorImpl 的内部状态。新的内部状态必须是步进的 CPU 字节张量，并且具有适当的大小。
 * 参见 CUDAGeneratorImpl::state 的注释，了解内部状态的布局和大小信息。
 */
void CUDAGeneratorImpl::set_state(const c10::TensorImpl& new_state) {
  // 检查是否正在捕获 CUDA 设备
  at::cuda::assertNotCapturing(
      "Please ensure to utilize the CUDAGeneratorImpl::set_state_index method during capturing.");
  // 定义 RNG 状态的大小，包括种子和 Philox 使用的偏移量
  static const size_t seed_size = sizeof(uint64_t);
  static const size_t offset_size = sizeof(int64_t);
  static const size_t total_size = seed_size + offset_size;

  // 检查新状态张量是否有效
  detail::check_rng_state(new_state);

  // 检查是否有 Philox 种子
  bool no_philox_seed = false;
  auto new_state_size = new_state.numel();
  if (new_state_size == total_size - offset_size) {
    no_philox_seed = true;
  } else {
    // 检查新状态张量的大小是否正确
    TORCH_CHECK(new_state_size == total_size, "RNG state is wrong size");
  }

  // 从新状态张量中复制种子值和 Philox 偏移量
  uint64_t input_seed = 0;
  auto new_rng_state = new_state.data_dtype_initialized<uint8_t>();
  memcpy(&input_seed, new_rng_state, seed_size);
  this->set_current_seed(input_seed);
  int64_t philox_offset = 0;
  if (!no_philox_seed) {
    memcpy(&philox_offset, new_rng_state + seed_size, offset_size);
  }
  this->set_philox_offset_per_thread(static_cast<uint64_t>(philox_offset));
}

/**
 * 将生成器的当前状态设置为另一个注册状态的状态。
 * 允许在不同的生成器状态之间切换。
 */
void CUDAGeneratorImpl::graphsafe_set_state(
    const c10::intrusive_ptr<GeneratorImpl>& gen) {
  // 将传入的 GeneratorImpl 转换为 CUDAGeneratorImpl
  c10::intrusive_ptr<CUDAGeneratorImpl> cuda_gen =
      dynamic_intrusive_pointer_cast<CUDAGeneratorImpl>(gen);
  // 检查是否成功转换为 CUDA 生成器
  TORCH_CHECK(cuda_gen, "Expected a CUDA Generator");
  // 将状态设置为另一个 CUDA 生成器的状态
  state_ = cuda_gen->state_;
}

/**
 * 获取指向当前 state_ 的 GeneratorImpl
 */
c10::intrusive_ptr<c10::GeneratorImpl> CUDAGeneratorImpl::graphsafe_get_state()
    const {
  // 创建一个新的 CUDAGeneratorImpl 对象，指向当前的 state_
  auto gen = make_intrusive<CUDAGeneratorImpl>(device().index(), state_);
  // 返回该对象的侵入式指针
  return gen;
}

/**
 * 设置 philox_offset_per_thread_，用于 curandStatePhilox4_32_10 使用
 *
 * 参见注释 [Acquire lock when using random generators]
 */
/**
 * Sets the philox_offset_per_thread_ for the current CUDAGeneratorImpl state.
 * Ensures the offset is a multiple of 4, following Note [Why enforce RNG offset % 4 == 0?]
 * 
 * @param offset The offset value to set, must be a multiple of 4.
 */
void CUDAGeneratorImpl::set_philox_offset_per_thread(uint64_t offset) {
  TORCH_CHECK(offset % 4 == 0, "offset must be a multiple of 4");
  state_->philox_offset_per_thread_ = offset;
}

/**
 * Retrieves the current philox_offset_per_thread_ value from the CUDAGeneratorImpl state.
 * 
 * @return The current philox_offset_per_thread_ value.
 */
uint64_t CUDAGeneratorImpl::philox_offset_per_thread() const {
  return state_->philox_offset_per_thread_;
}

/**
 * Registers the RNG state of this CUDAGeneratorImpl instance to a CUDA graph for management.
 * 
 * @param graph Pointer to the CUDA graph to register with.
 */
void CUDAGeneratorImpl::register_graph(cuda::CUDAGraph* graph) {
  graph->register_generator_state(state_);
  state_->register_graph(graph);
}

/**
 * Unregisters the RNG state of this CUDAGeneratorImpl instance from a CUDA graph.
 * 
 * @param graph Pointer to the CUDA graph to unregister from.
 */
void CUDAGeneratorImpl::unregister_graph(cuda::CUDAGraph* graph) {
  state_->unregister_graph(graph);
}

/**
 * Generates a PhiloxCudaState object suitable for use in CUDA kernels,
 * ensuring safety and non-divergent behavior in CUDA graphs.
 * 
 * @param increment The amount by which to increment the offset for future use.
 * @return A PhiloxCudaState object initialized with appropriate state values.
 * 
 * See Note [CUDA Graph-safe RNG states] for more details.
 */
PhiloxCudaState CUDAGeneratorImpl::philox_cuda_state(uint64_t increment) {
  if (at::cuda::currentStreamCaptureStatus() != at::cuda::CaptureStatus::None) {
    uint32_t offset = state_->offset_intragraph_;
    state_->increase(increment);
    return PhiloxCudaState(
        state_->seed_extragraph_.data_ptr<int64_t>(),
        state_->offset_extragraph_.data_ptr<int64_t>(),
        offset);
  } else {
    uint64_t offset = state_->philox_offset_per_thread_;
    state_->increase(increment);
    return PhiloxCudaState(state_->seed_, offset);
  }
}

/**
 * Provides backward compatibility for call sites using philox_engine_inputs,
 * encouraging transition to philox_cuda_state for CUDA graph safety.
 * 
 * @param increment The increment amount to apply to the offset.
 * @return A pair of seed and offset values for engine initialization.
 */
std::pair<uint64_t, uint64_t> CUDAGeneratorImpl::philox_engine_inputs(
    uint64_t increment) {
  at::cuda::assertNotCapturing(
      "Refactor this op to use CUDAGeneratorImpl::philox_cuda_state. Cannot call CUDAGeneratorImpl::philox_engine_inputs");
  uint64_t offset = state_->philox_offset_per_thread_;
  state_->increase(increment);
  return std::make_pair(state_->seed_, offset);
}
DeviceType CUDAGeneratorImpl::device_type() {
  return DeviceType::CUDA;
}


// 返回此 CUDA 生成器的设备类型为 CUDA
DeviceType CUDAGeneratorImpl::device_type() {
  return DeviceType::CUDA;
}



/**
 * Public clone method implementation
 *
 * See Note [Acquire lock when using random generators]
 */
std::shared_ptr<CUDAGeneratorImpl> CUDAGeneratorImpl::clone() const {
  return std::shared_ptr<CUDAGeneratorImpl>(this->clone_impl());
}


/**
 * 实现公共的克隆方法
 *
 * 参见注释 [使用随机生成器时获取锁] 
 */
std::shared_ptr<CUDAGeneratorImpl> CUDAGeneratorImpl::clone() const {
  // 调用私有的克隆方法，返回一个新的共享指针实例
  return std::shared_ptr<CUDAGeneratorImpl>(this->clone_impl());
}



/**
 * Private clone method implementation
 *
 * See Note [Acquire lock when using random generators]
 */
CUDAGeneratorImpl* CUDAGeneratorImpl::clone_impl() const {
  // 断言当前不处于捕获状态，否则抛出异常
  at::cuda::assertNotCapturing("Cannot call CUDAGeneratorImpl::clone_impl");
  // 创建一个新的 CUDAGeneratorImpl 实例，使用当前设备索引和克隆的状态
  auto gen = new CUDAGeneratorImpl(this->device().index(), state_->clone());
  return gen;
}


/**
 * 实现私有的克隆方法
 *
 * 参见注释 [使用随机生成器时获取锁]
 */
CUDAGeneratorImpl* CUDAGeneratorImpl::clone_impl() const {
  // 断言当前不处于捕获状态，否则抛出异常
  at::cuda::assertNotCapturing("Cannot call CUDAGeneratorImpl::clone_impl");
  // 创建一个新的 CUDAGeneratorImpl 实例，使用当前设备索引和克隆的状态
  auto gen = new CUDAGeneratorImpl(this->device().index(), state_->clone());
  return gen;
}



} // namespace at


} // namespace at
```