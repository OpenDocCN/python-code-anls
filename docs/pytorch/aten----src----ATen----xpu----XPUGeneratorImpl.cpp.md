# `.\pytorch\aten\src\ATen\xpu\XPUGeneratorImpl.cpp`

```py
/*
 * This namespace contains implementation details for XPU generators within ATen.
 */
#include <ATen/Utils.h>
#include <ATen/xpu/XPUGeneratorImpl.h>
#include <c10/core/StreamGuard.h>
#include <c10/util/CallOnce.h>
#include <c10/xpu/XPUFunctions.h>

namespace at {
namespace xpu::detail {
namespace {

/*
 * Currently, there is one generator pool containing XPU generator per device.
 * Each generator is lazily initialized the first time generator is
 * requested for a device.
 */
c10::once_flag init_flag; // A flag to ensure initialization occurs only once across threads
DeviceIndex num_gpus = -1; // Number of XPU devices available
std::deque<c10::once_flag> xpu_gens_init_flag; // Flags for each XPU device's generator initialization
std::vector<Generator> default_gens_xpu; // Vector holding default generators for each XPU device

void initXPUGenVector() {
  num_gpus = device_count(); // Determine the number of available XPU devices
  xpu_gens_init_flag.resize(num_gpus); // Resize to match the number of XPU devices
  default_gens_xpu.resize(num_gpus); // Resize to store default generators for each XPU device
}

inline void check_device(DeviceIndex device) {
  TORCH_CHECK(
      device >= 0 && device < num_gpus,
      "device is out of range, device is ",
      static_cast<int16_t>(device),
      ", total number of device is ",
      static_cast<int16_t>(num_gpus),
      ".");
}

} // anonymous namespace

// Get the default generator with a random seed for a specific xpu device.
const Generator& getDefaultXPUGenerator(DeviceIndex device) {
  c10::call_once(init_flag, initXPUGenVector); // Ensure initialization of generator vector
  if (device == -1) {
    device = c10::xpu::current_device(); // Get current XPU device if none specified
  }
  check_device(device); // Validate the specified XPU device
  c10::call_once(xpu_gens_init_flag[device], [&]() {
    default_gens_xpu[device] = make_generator<XPUGeneratorImpl>(device); // Create generator for the device
    default_gens_xpu[device].seed(); // Seed the generator
  });
  return default_gens_xpu[device]; // Return the default generator for the device
}

// Create a generator with a fixed seed for a specific xpu device.
Generator createXPUGenerator(DeviceIndex device) {
  c10::call_once(init_flag, initXPUGenVector); // Ensure initialization of generator vector
  if (device == -1) {
    device = c10::xpu::current_device(); // Get current XPU device if none specified
  }
  check_device(device); // Validate the specified XPU device
  auto gen = make_generator<XPUGeneratorImpl>(device); // Create generator for the device
  auto xpu_gen = check_generator<XPUGeneratorImpl>(gen); // Validate and cast to XPU generator type
  xpu_gen->set_current_seed(default_rng_seed_val); // Set the seed for the generator
  xpu_gen->set_philox_offset_per_thread(0); // Set the offset for the generator
  return gen; // Return the created generator
}

} // namespace xpu::detail

// Constructor for XPUGeneratorImpl, initializing with a device index.
XPUGeneratorImpl::XPUGeneratorImpl(DeviceIndex device_index)
    : GeneratorImpl{
          Device(DeviceType::XPU, device_index), // Initialize base class with XPU device type and index
          DispatchKeySet(c10::DispatchKey::XPU)} {} // Set dispatch key to XPU for this generator

void XPUGeneratorImpl::set_current_seed(uint64_t seed) {
  seed_ = seed; // Set the current seed value for the generator
  set_philox_offset_per_thread(0); // Reset the offset for the generator
}

void XPUGeneratorImpl::set_offset(uint64_t offset) {
  set_philox_offset_per_thread(offset); // Set the offset for the generator
}

uint64_t XPUGeneratorImpl::get_offset() const {
  return philox_offset_per_thread_; // Get the current offset value for the generator
}

uint64_t XPUGeneratorImpl::current_seed() const {
  return seed_; // Get the current seed value for the generator
}

uint64_t XPUGeneratorImpl::seed() {
  auto random = c10::detail::getNonDeterministicRandom(true); // Get a nondeterministic random number
  this->set_current_seed(random); // Set the current seed value for the generator
  return random; // Return the generated random number
}
// 返回 XPUGeneratorImpl 对象的 RNG 状态，包括种子和用于 Philox 的偏移量
c10::intrusive_ptr<c10::TensorImpl> XPUGeneratorImpl::get_state() const {
    // RNG 状态包括种子和用于 Philox 的偏移量，每个占用64位（8字节）
    static const size_t seed_size = sizeof(uint64_t);
    static const size_t offset_size = sizeof(uint64_t);
    static const size_t total_size = seed_size + offset_size;

    // 创建一个 CPU 字节张量来保存内部状态
    auto state_tensor = at::detail::empty_cpu(
        {static_cast<int64_t>(total_size)},
        ScalarType::Byte,
        c10::nullopt,
        c10::nullopt,
        c10::nullopt,
        c10::nullopt);
    auto rng_state = state_tensor.data_ptr<uint8_t>();  // 获取张量的数据指针
    auto current_seed = this->current_seed();  // 获取当前种子
    auto offset = this->philox_offset_per_thread();  // 获取当前 Philox 偏移量
    memcpy(rng_state, &current_seed, seed_size);  // 将当前种子复制到 RNG 状态的开头
    memcpy(rng_state + seed_size, &offset, offset_size);  // 将 Philox 偏移量复制到 RNG 状态的末尾

    return state_tensor.getIntrusivePtr();  // 返回内部状态张量的指针
}

// 设置 XPUGeneratorImpl 对象的 RNG 状态
void XPUGeneratorImpl::set_state(const c10::TensorImpl& new_state) {
    static const size_t seed_size = sizeof(uint64_t);
    static const size_t offset_size = sizeof(uint64_t);
    static const size_t total_size = seed_size + offset_size;

    at::detail::check_rng_state(new_state);  // 检查新状态的有效性
    auto new_state_size = new_state.numel();  // 获取新状态张量的元素数
    TORCH_CHECK(new_state_size == total_size, "RNG state is wrong size");  // 检查新状态的大小是否正确

    uint64_t input_seed;
    auto new_rng_state = new_state.data_ptr<uint8_t>();  // 获取新状态张量的数据指针
    memcpy(&input_seed, new_rng_state, seed_size);  // 从新状态中复制种子值
    this->set_current_seed(input_seed);  // 设置当前种子值
    uint64_t philox_offset;
    memcpy(&philox_offset, new_rng_state + seed_size, offset_size);  // 从新状态中复制 Philox 偏移量
    this->set_philox_offset_per_thread(philox_offset);  // 设置 Philox 偏移量
}

// 设置 XPUGeneratorImpl 对象的 Philox 偏移量，要求偏移量必须是4的倍数
void XPUGeneratorImpl::set_philox_offset_per_thread(uint64_t offset) {
    TORCH_CHECK(offset % 4 == 0, "offset must be a multiple of 4");  // 检查偏移量是否是4的倍数
    philox_offset_per_thread_ = offset;  // 设置 Philox 偏移量
}

// 返回 XPUGeneratorImpl 对象当前的 Philox 偏移量
uint64_t XPUGeneratorImpl::philox_offset_per_thread() const {
    return philox_offset_per_thread_;  // 返回当前的 Philox 偏移量
}

// 计算用于 Philox 引擎的输入参数对，并返回种子和偏移量
std::pair<uint64_t, uint64_t> XPUGeneratorImpl::philox_engine_inputs(
    uint64_t increment) {
    increment = ((increment + 3) / 4) * 4;  // 将增量调整为4的倍数
    TORCH_INTERNAL_ASSERT(this->philox_offset_per_thread_ % 4 == 0);  // 内部断言，确保当前 Philox 偏移量是4的倍数
    uint64_t offset = this->philox_offset_per_thread_;  // 获取当前 Philox 偏移量
    this->philox_offset_per_thread_ += increment;  // 更新 Philox 偏移量
    return std::make_pair(this->seed_, offset);  // 返回种子和偏移量的 pair
}

// 返回 XPUGeneratorImpl 对象的设备类型
DeviceType XPUGeneratorImpl::device_type() {
    return DeviceType::XPU;  // 返回设备类型为 XPU
}

// 克隆 XPUGeneratorImpl 对象，并返回共享指针
std::shared_ptr<XPUGeneratorImpl> XPUGeneratorImpl::clone() const {
    return std::shared_ptr<XPUGeneratorImpl>(this->clone_impl());  // 返回克隆的 XPUGeneratorImpl 对象的共享指针
}

// 克隆 XPUGeneratorImpl 对象的实现，并返回指针
XPUGeneratorImpl* XPUGeneratorImpl::clone_impl() const {
    auto gen = new XPUGeneratorImpl(this->device().index());  // 创建新的 XPUGeneratorImpl 对象
    gen->set_current_seed(this->seed_);  // 设置新对象的种子值
    gen->set_philox_offset_per_thread(this->philox_offset_per_thread_);  // 设置新对象的 Philox 偏移量
    return gen;  // 返回新对象的指针
}
```