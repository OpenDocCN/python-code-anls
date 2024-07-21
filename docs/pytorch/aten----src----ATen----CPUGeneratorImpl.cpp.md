# `.\pytorch\aten\src\ATen\CPUGeneratorImpl.cpp`

```
/**
 * Including necessary headers for ATen and standard libraries
 */
#include <ATen/CPUGeneratorImpl.h>
#include <ATen/Utils.h>
#include <ATen/core/MT19937RNGEngine.h>
#include <c10/util/MathConstants.h>
#include <algorithm>

namespace at {

namespace detail {

/**
 * CPUGeneratorImplStateLegacy is a POD class needed for memcpys
 * in torch.get_rng_state() and torch.set_rng_state().
 * It is a legacy class and even though it is replaced with
 * at::CPUGeneratorImpl, we need this class and some of its fields
 * to support backward compatibility on loading checkpoints.
 */
struct CPUGeneratorImplStateLegacy {
  /* The initial seed. */
  uint64_t the_initial_seed;
  int left;  /* = 1; */
  int seeded; /* = 0; */
  uint64_t next;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
  uint64_t state[at::MERSENNE_STATE_N]; /* the array for the state vector  */

  /********************************/

  /* For normal distribution */
  double normal_x;
  double normal_y;
  double normal_rho;
  int normal_is_valid; /* = 0; */
};

/**
 * CPUGeneratorImplState is a POD class containing
 * new data introduced in at::CPUGeneratorImpl and the legacy state. It is used
 * as a helper for torch.get_rng_state() and torch.set_rng_state()
 * functions.
 */
struct CPUGeneratorImplState {
  CPUGeneratorImplStateLegacy legacy_pod;
  float next_float_normal_sample;
  bool is_next_float_normal_sample_valid;
};

/**
 * PyTorch maintains a collection of default generators that get
 * initialized once. The purpose of these default generators is to
 * maintain a global running state of the pseudo random number generation,
 * when a user does not explicitly mention any generator.
 * getDefaultCPUGenerator gets the default generator for a particular
 * device.
 */
const Generator& getDefaultCPUGenerator() {
  static auto default_gen_cpu = createCPUGenerator(c10::detail::getNonDeterministicRandom());
  return default_gen_cpu;
}

/**
 * Utility to create a CPUGeneratorImpl. Returns a shared_ptr
 */
Generator createCPUGenerator(uint64_t seed_val) {
  return make_generator<CPUGeneratorImpl>(seed_val);
}

/**
 * Helper function to concatenate two 32 bit unsigned int
 * and return them as a 64 bit unsigned int
 */
inline uint64_t make64BitsFrom32Bits(uint32_t hi, uint32_t lo) {
  return (static_cast<uint64_t>(hi) << 32) | lo;
}

} // namespace detail

/**
 * CPUGeneratorImpl class implementation
 */
CPUGeneratorImpl::CPUGeneratorImpl(uint64_t seed_in)
  : c10::GeneratorImpl{Device(DeviceType::CPU), DispatchKeySet(c10::DispatchKey::CPU)},
    engine_{seed_in},
    next_float_normal_sample_{std::optional<float>()},
    next_double_normal_sample_{std::optional<double>()} { }

/**
 * Manually seeds the engine with the seed input
 * See Note [Acquire lock when using random generators]
 */
void CPUGeneratorImpl::set_current_seed(uint64_t seed) {
  next_float_normal_sample_.reset();
  next_double_normal_sample_.reset();
  engine_ = mt19937(seed);
}
/**
 * Sets the offset of RNG state.
 * See Note [Acquire lock when using random generators]
 */
void CPUGeneratorImpl::set_offset(uint64_t offset) {
  // 在使用随机生成器时，参见注意事项 [Acquire lock when using random generators]，设置 RNG 状态的偏移量
  TORCH_CHECK(false, "CPU Generator does not use offset");
  // CPU 生成器不使用偏移量，因此抛出错误
}

/**
 * Gets the current offset of CPUGeneratorImpl.
 */
uint64_t CPUGeneratorImpl::get_offset() const {
  // 返回当前 CPUGeneratorImpl 的偏移量
  TORCH_CHECK(false, "CPU Generator does not use offset");
  // CPU 生成器不使用偏移量，因此抛出错误
}

/**
 * Gets the current seed of CPUGeneratorImpl.
 */
uint64_t CPUGeneratorImpl::current_seed() const {
  // 返回当前 CPUGeneratorImpl 的种子值
  return engine_.seed();
  // 返回随机数引擎 engine_ 的种子值
}

/**
 * Gets a nondeterministic random number from /dev/urandom or time,
 * seeds the CPUGeneratorImpl with it and then returns that number.
 *
 * FIXME: You can move this function to Generator.cpp if the algorithm
 * in getNonDeterministicRandom is unified for both CPU and CUDA
 */
uint64_t CPUGeneratorImpl::seed() {
  // 获取一个来自 /dev/urandom 或时间的非确定性随机数，
  // 使用它来种子化 CPUGeneratorImpl，然后返回该随机数
  auto random = c10::detail::getNonDeterministicRandom();
  this->set_current_seed(random);
  // 设置当前的种子值为 random
  return random;
  // 返回获取的非确定性随机数
}

/**
 * Sets the internal state of CPUGeneratorImpl. The new internal state
 * must be a strided CPU byte tensor and of the same size as either
 * CPUGeneratorImplStateLegacy (for legacy CPU generator state) or
 * CPUGeneratorImplState (for new state).
 *
 * FIXME: Remove support of the legacy state in the future?
 */
void CPUGeneratorImpl::set_state(const c10::TensorImpl& new_state) {
  using detail::CPUGeneratorImplState;
  using detail::CPUGeneratorImplStateLegacy;

  static_assert(std::is_standard_layout_v<CPUGeneratorImplStateLegacy>, "CPUGeneratorImplStateLegacy is not a PODType");
  static_assert(std::is_standard_layout_v<CPUGeneratorImplState>, "CPUGeneratorImplState is not a PODType");

  static const size_t size_legacy = sizeof(CPUGeneratorImplStateLegacy);
  static const size_t size_current = sizeof(CPUGeneratorImplState);
  static_assert(size_legacy != size_current, "CPUGeneratorImplStateLegacy and CPUGeneratorImplState can't be of the same size");

  detail::check_rng_state(new_state);

  at::mt19937 engine;
  auto float_normal_sample = std::optional<float>();
  auto double_normal_sample = std::optional<double>();

  // Construct the state of at::CPUGeneratorImpl based on input byte tensor size.
  CPUGeneratorImplStateLegacy* legacy_pod{nullptr};
  auto new_state_size = new_state.numel();
  if (new_state_size == size_legacy) {
    legacy_pod = (CPUGeneratorImplStateLegacy*)new_state.data();
    // Note that in CPUGeneratorImplStateLegacy, we didn't have float version
    // of normal sample and hence we leave the std::optional<float> as is

    // Update next_double_normal_sample.
    // Note that CPUGeneratorImplStateLegacy stores two uniform values (normal_x, normal_y)
    // and a rho value (normal_rho). These three values were redundant and in the new
    // DistributionsHelper.h, we store the actual extra normal sample, rather than three
    // intermediate values.
    // 根据输入的字节张量大小构造 at::CPUGeneratorImpl 的状态
    // 如果 legacy_pod 的 normal_is_valid 标志为真，则执行以下操作
    if (legacy_pod->normal_is_valid) {
      // 从 legacy_pod 中读取 normal_rho 到 r
      auto r = legacy_pod->normal_rho;
      // 计算 theta，使用 legacy_pod 中的 normal_x
      auto theta = 2.0 * c10::pi<double> * legacy_pod->normal_x;
      // 当处于缓存模式时，返回正弦版本的正态分布样本
      double_normal_sample = std::optional<double>(r * ::sin(theta));
    }
  } else if (new_state_size == size_current) {
    // 将 new_state 转换为 CPUGeneratorImplState 指针
    auto rng_state = (CPUGeneratorImplState*)new_state.data();
    // 设置 legacy_pod 指针指向 rng_state 的 legacy_pod 成员
    legacy_pod = &rng_state->legacy_pod;
    // 更新 next_float_normal_sample
    if (rng_state->is_next_float_normal_sample_valid) {
      float_normal_sample = std::optional<float>(rng_state->next_float_normal_sample);
    }

    // 更新 next_double_normal_sample
    // 注意，在 getRNGState 中，我们现在在 normal_y 中返回实际的正态分布样本，
    // 并在 normal_is_valid 中返回其有效性。冗余的 normal_x 和 normal_rho 被压缩为 0.0。
    if (legacy_pod->normal_is_valid) {
      double_normal_sample = std::optional<double>(legacy_pod->normal_y);
    }
  } else {
    // 报错，预期输入的 RNG 状态大小应为 size_legacy 或 size_current，但发现 new_state_size 为其他值
    AT_ERROR("Expected either a CPUGeneratorImplStateLegacy of size ", size_legacy,
             " or a CPUGeneratorImplState of size ", size_current,
             " but found the input RNG state size to be ", new_state_size);
  }

  // 构造 engine_
  // 注意，CPUGeneratorImplStateLegacy 存储了一个 64 位无符号整数的状态数组，
  // 而我们重新定义的 mt19937 使用了一个 32 位无符号整数的状态数组。因此，我们进行了 std::copy 操作。
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  at::mt19937_data_pod rng_data;
  // 将 legacy_pod 的 state 数组复制到 rng_data 的 state_ 数组中
  std::copy(std::begin(legacy_pod->state), std::end(legacy_pod->state), rng_data.state_.begin());
  // 设置 rng_data 的种子为 legacy_pod 的 the_initial_seed
  rng_data.seed_ = legacy_pod->the_initial_seed;
  // 设置 rng_data 的 left_ 为 legacy_pod 的 left
  rng_data.left_ = legacy_pod->left;
  // 设置 rng_data 的 seeded_ 为 legacy_pod 的 seeded
  rng_data.seeded_ = legacy_pod->seeded;
  // 将 legacy_pod 的 next 转换为 uint32_t 后赋给 rng_data 的 next_
  rng_data.next_ = static_cast<uint32_t>(legacy_pod->next);
  // 使用 rng_data 设置 engine 的数据
  engine.set_data(rng_data);
  // 检查 engine 的有效性，若无效则报错
  TORCH_CHECK(engine.is_valid(), "Invalid mt19937 state");
  // 将 engine 赋给 this 指针所指向的对象的 engine_ 成员
  this->engine_ = engine;
  // 将 float_normal_sample 赋给 this 指针所指向的对象的 next_float_normal_sample_ 成员
  this->next_float_normal_sample_ = float_normal_sample;
  // 将 double_normal_sample 赋给 this 指针所指向的对象的 next_double_normal_sample_ 成员
  this->next_double_normal_sample_ = double_normal_sample;
}

/**
 * 获取 CPUGeneratorImpl 的当前内部状态。内部状态以 CPU 字节张量的形式返回。
 */
c10::intrusive_ptr<c10::TensorImpl> CPUGeneratorImpl::get_state() const {
  using detail::CPUGeneratorImplState;

  static const size_t size = sizeof(CPUGeneratorImplState);
  // 确保 CPUGeneratorImplState 是一个标准布局类型
  static_assert(std::is_standard_layout_v<CPUGeneratorImplState>, "CPUGeneratorImplState is not a PODType");

  // 创建一个大小为 size 的 CPU 字节张量
  auto state_tensor = at::detail::empty_cpu({(int64_t)size}, ScalarType::Byte, c10::nullopt, c10::nullopt, c10::nullopt, c10::nullopt);
  auto rng_state = state_tensor.data_ptr();

  // 累积生成器数据以复制到字节张量中
  auto accum_state = std::make_unique<CPUGeneratorImplState>();
  auto rng_data = this->engine_.data();
  accum_state->legacy_pod.the_initial_seed = rng_data.seed_;
  accum_state->legacy_pod.left = rng_data.left_;
  accum_state->legacy_pod.seeded = rng_data.seeded_;
  accum_state->legacy_pod.next = rng_data.next_;
  std::copy(rng_data.state_.begin(), rng_data.state_.end(), std::begin(accum_state->legacy_pod.state));
  accum_state->legacy_pod.normal_x = 0.0; // 不再使用，仅为示例
  accum_state->legacy_pod.normal_rho = 0.0; // 不再使用，仅为示例
  accum_state->legacy_pod.normal_is_valid = false;
  accum_state->legacy_pod.normal_y = 0.0;
  accum_state->next_float_normal_sample = 0.0f;
  accum_state->is_next_float_normal_sample_valid = false;
  if (this->next_double_normal_sample_) {
    accum_state->legacy_pod.normal_is_valid = true;
    accum_state->legacy_pod.normal_y = *(this->next_double_normal_sample_);
  }
  if (this->next_float_normal_sample_) {
    accum_state->is_next_float_normal_sample_valid = true;
    accum_state->next_float_normal_sample = *(this->next_float_normal_sample_);
  }

  // 将累积的状态数据复制到 rng_state 中
  memcpy(rng_state, accum_state.get(), size);
  // 返回状态张量的侵入式指针
  return state_tensor.getIntrusivePtr();
}

/**
 * 获取 CPUGeneratorImpl 的设备类型。
 * 用于运行时类型检查。
 */
DeviceType CPUGeneratorImpl::device_type() {
  return DeviceType::CPU;
}

/**
 * 从引擎中获取一个随机的 32 位无符号整数
 *
 * 参见注释 [使用随机生成器时获取锁]
 */
uint32_t CPUGeneratorImpl::random() {
  return engine_();
}

/**
 * 从引擎中获取一个随机的 64 位无符号整数
 *
 * 参见注释 [使用随机生成器时获取锁]
 */
uint64_t CPUGeneratorImpl::random64() {
  uint32_t random1 = engine_();
  uint32_t random2 = engine_();
  return detail::make64BitsFrom32Bits(random1, random2);
}

/**
 * 获取缓存的浮点数正态分布随机数
 */
std::optional<float> CPUGeneratorImpl::next_float_normal_sample() {
  return next_float_normal_sample_;
}

/**
 * 获取缓存的双精度浮点数正态分布随机数
 */
std::optional<double> CPUGeneratorImpl::next_double_normal_sample() {
  return next_double_normal_sample_;
}

/**
 * 缓存浮点数正态分布随机数
 *
 * 参见注释 [使用随机生成器时获取锁]
 */
/**
 * Set the next normal random sample in single precision (float)
 *
 * @param randn Optional value representing the next normal random sample
 */
void CPUGeneratorImpl::set_next_float_normal_sample(std::optional<float> randn) {
  next_float_normal_sample_ = randn;
}

/**
 * Cache normal random sample in double precision
 *
 * See Note [Acquire lock when using random generators]
 */
void CPUGeneratorImpl::set_next_double_normal_sample(std::optional<double> randn) {
  next_double_normal_sample_ = randn;
}

/**
 * Retrieve the random number generation engine used by CPUGeneratorImpl
 */
at::mt19937 CPUGeneratorImpl::engine() {
  return engine_;
}

/**
 * Set the random number generation engine for CPUGeneratorImpl
 *
 * @param engine An instance of mt19937 engine to set as the generator
 *
 * See Note [Acquire lock when using random generators]
 */
void CPUGeneratorImpl::set_engine(at::mt19937 engine) {
  engine_ = engine;
}

/**
 * Public method to create a shared pointer clone of CPUGeneratorImpl instance
 *
 * @return Shared pointer to a newly created CPUGeneratorImpl instance
 *
 * See Note [Acquire lock when using random generators]
 */
std::shared_ptr<CPUGeneratorImpl> CPUGeneratorImpl::clone() const {
  return std::shared_ptr<CPUGeneratorImpl>(this->clone_impl());
}

/**
 * Private method to clone CPUGeneratorImpl instance
 *
 * @return Pointer to a newly created CPUGeneratorImpl instance
 *
 * See Note [Acquire lock when using random generators]
 */
CPUGeneratorImpl* CPUGeneratorImpl::clone_impl() const {
  // Create a new instance of CPUGeneratorImpl
  auto gen = new CPUGeneratorImpl();
  // Copy the random number generation engine to the new instance
  gen->set_engine(engine_);
  // Copy the next float normal sample to the new instance
  gen->set_next_float_normal_sample(next_float_normal_sample_);
  // Copy the next double normal sample to the new instance
  gen->set_next_double_normal_sample(next_double_normal_sample_);
  // Return the pointer to the newly created instance
  return gen;
}

} // namespace at
```