# `.\pytorch\test\cpp_extensions\mtia_extension.cpp`

```
// 包含 ATen 库中的 MTIAHooksInterface 头文件
#include <ATen/detail/MTIAHooksInterface.h>
// 包含 c10 核心库中的 Device 相关头文件
#include <c10/core/Device.h>
// 包含 c10 核心库中的 Stream 相关头文件
#include <c10/core/Stream.h>
// 包含 c10 核心库中的 DeviceGuardImplInterface 接口头文件
#include <c10/core/impl/DeviceGuardImplInterface.h>
// 包含 c10 工具库中的 Logging 头文件
#include <c10/util/Logging.h>
// 包含 Torch 库中的 device_lazy_init 头文件
#include <torch/csrc/utils/device_lazy_init.h>
// 包含 C++ 标准库中的线程头文件
#include <thread>

// 定义命名空间 torch::mtia
namespace torch::mtia {

// 定义 MTIA 设备类型为常量表达式 c10::DeviceType::MTIA
constexpr c10::DeviceType kMTIADeviceType = c10::DeviceType::MTIA;
// 定义 MTIA 设备数量为常量表达式 2
constexpr c10::DeviceIndex kMTIADeviceCount = 2;

// 声明当前线程的 MTIA 设备索引为静态线程局部变量，默认为 0
static thread_local c10::DeviceIndex current_device = 0;

// 声明当前线程的 MTIA 设备流数组为静态线程局部变量，包含两个预定义的流
static thread_local std::array<c10::Stream, kMTIADeviceCount> current_streams =
    {c10::Stream::unpack3(0, 0, c10::DeviceType::MTIA),
     c10::Stream::unpack3(0, 1, c10::DeviceType::MTIA)};

// 声明流 ID 生成器为静态变量，初始值为 1
static int64_t stream_id_gen = 1;
// 声明事件 ID 生成器为静态变量，初始值为 1
static int64_t event_id_gen = 1;

// 声明默认流数组为静态常量表达式，包含两个预定义的流
static std::array<c10::Stream, kMTIADeviceCount> default_streams = {
    c10::Stream::unpack3(0, 0, c10::DeviceType::MTIA),
    c10::Stream::unpack3(0, 1, c10::DeviceType::MTIA)};

// 定义 MTIAGuardImpl 结构体，实现 c10::impl::DeviceGuardImplInterface 接口
struct MTIAGuardImpl final : public c10::impl::DeviceGuardImplInterface {
  MTIAGuardImpl() = default;
  // 显式构造函数，验证设备类型是否为 MTIA
  explicit MTIAGuardImpl(c10::DeviceType t) {
    TORCH_INTERNAL_ASSERT(t == kMTIADeviceType);
  }

  // 获取设备类型为 MTIA
  c10::DeviceType type() const override {
    return kMTIADeviceType;
  }

  // 交换设备并返回旧设备
  c10::Device exchangeDevice(c10::Device d) const override {
    c10::Device old_device = getDevice();
    if (old_device.index() != d.index()) {
      setDevice(d);
    }
    return old_device;
  }

  // 获取当前设备
  c10::Device getDevice() const override {
    return c10::Device(kMTIADeviceType, current_device);
  }

  // 设置当前设备
  void setDevice(c10::Device d) const override {
    c10::Device current_device = getDevice();
    if (current_device.index() != d.index()) {
      current_device = d;
    }
  }

  // 不安全地设置设备（未检查）
  void uncheckedSetDevice(c10::Device d) const noexcept override {
    (void)d;
  }

  // 获取设备对应的流
  c10::Stream getStream(c10::Device d) const noexcept override {
    return current_streams[d.index()];
  }

  // 获取新的设备流
  c10::Stream getNewStream(c10::Device d, int priority = 0) const override {
    (void)priority;
    return c10::Stream::unpack3(stream_id_gen++, d.index(), d.type());
  }

  // 获取默认设备流
  c10::Stream getDefaultStream(c10::Device d) const override {
    return default_streams[d.index()];
  }

  // 从全局池中获取设备流（带有优先级）
  c10::Stream getStreamFromGlobalPool(
      c10::Device d,
      bool isHighPriority = false) const override {
    return c10::Stream::unpack3(stream_id_gen++, d.index(), d.type());
  }

  // 交换流并返回旧流（注意：不修改当前设备）
  c10::Stream exchangeStream(c10::Stream s) const noexcept override {
    c10::Stream old_stream = getStream(s.device());
    return old_stream;
  }

  // 获取设备数量
  c10::DeviceIndex deviceCount() const noexcept override {
    return kMTIADeviceCount;
  }

  // 销毁事件（无操作）
  void destroyEvent(void* event, const c10::DeviceIndex device_index)
      const noexcept override {
    (void)device_index;
  }

  // 记录事件（无操作）
  void record(
      void** event,
      const c10::Stream& stream,
      const c10::DeviceIndex device_index,
      const c10::EventFlag flag) const override {
    // 检查事件的设备索引是否匹配流的设备索引或者设备索引为 -1（表示不关心设备），否则抛出错误
    TORCH_CHECK(
        device_index == -1 || device_index == stream.device_index(),
        "Event device index ",
        device_index,
        " does not match recording stream's device index ",
        stream.device_index(),
        ".");

    // 保存当前的设备，以便后续恢复
    const auto orig_device = getDevice();

    // 将当前设备切换为流的设备
    setDevice(stream.device());

    // 如果事件为空指针，则分配一个新的事件 ID
    if (*event == nullptr) {
      *event = reinterpret_cast<void*>(event_id_gen++);
    }
    
    // 恢复原始设备
    setDevice(orig_device);
  }

  // 阻塞函数，无操作，因为该函数是一个空实现
  void block(void* event, const c10::Stream& stream) const override {
    (void)event;  // 防止未使用的参数警告
    (void)stream; // 防止未使用的参数警告
  }

  // 可以从任何设备调用的事件查询函数，始终返回 true
  bool queryEvent(void* event) const override {
    (void)event;  // 防止未使用的参数警告
    return true;
  }

  // 流查询函数，始终返回 true
  bool queryStream(const c10::Stream& stream) const override {
    (void)stream; // 防止未使用的参数警告
    return true;
  }

  // 同步流操作的空实现，不做任何操作
  void synchronizeStream(const c10::Stream& stream) const override {
    (void)stream; // 防止未使用的参数警告
  }

  // 在流上记录数据指针的空实现，不做任何操作
  void recordDataPtrOnStream(
      const c10::DataPtr& data_ptr,
      const c10::Stream& stream) const override {
    (void)data_ptr; // 防止未使用的参数警告
    (void)stream;   // 防止未使用的参数警告
  }

  // 计算两个事件之间的时间差，假设时间差为 1 微秒（虚拟实现）
  double elapsedTime(void* event1, void* event2, const c10::DeviceIndex device_index) const override {
    (void)event1;      // 防止未使用的参数警告
    (void)event2;      // 防止未使用的参数警告
    (void)device_index;// 防止未使用的参数警告
    uint64_t elapsed_time = 1e6;
    return (double)(elapsed_time / 1e6);
  }

  // 同步事件的空实现，不做任何操作
  void synchronizeEvent(void* event) const override {
    (void)event;  // 防止未使用的参数警告
  }
};

// 结构体 MTIAHooks 继承自 at::MTIAHooksInterface 接口
struct MTIAHooks : public at::MTIAHooksInterface {
  // 构造函数，接受 at::MTIAHooksArgs 参数
  explicit MTIAHooks(at::MTIAHooksArgs) {}

  // 初始化 MTIA
  void initMTIA() const override {}

  // 检查是否支持 MTIA
  bool hasMTIA() const override {
    return true;
  }

  // 返回设备数量
  c10::DeviceIndex deviceCount() const override {
    // 惰性初始化设备
    torch::utils::device_lazy_init(at::kMTIA);
    return c10::DeviceIndex(2);
  }

  // 设备同步函数
  void deviceSynchronize(c10::DeviceIndex device_index) const override {
    // 惰性初始化设备
    torch::utils::device_lazy_init(at::kMTIA);
    // 忽略参数 device_index
    (void)device_index;
  }

  // 返回配置信息
  std::string showConfig() const override {
    return "None config";
  }

  // 交换设备函数
  c10::DeviceIndex exchangeDevice(c10::DeviceIndex device) const override {
    // 惰性初始化设备
    torch::utils::device_lazy_init(at::kMTIA);
    // 保存原始设备
    auto orig_device = current_device;
    // 如果当前设备与要交换的设备不同，则更新当前设备
    if (current_device != device) {
      current_device = device;
    }
    return orig_device;
  }

  // 可能的设备交换函数
  c10::DeviceIndex maybeExchangeDevice(c10::DeviceIndex device) const override {
    // 惰性初始化设备
    torch::utils::device_lazy_init(at::kMTIA);
    // 保存原始设备
    auto orig_device = current_device;
    // 如果当前设备与要交换的设备不同，则更新当前设备
    if (current_device != device) {
      current_device = device;
    }
    return orig_device;
  }

  // 获取默认流函数
  c10::Stream getDefaultStream(c10::DeviceIndex device) const override {
    // 惰性初始化设备
    torch::utils::device_lazy_init(at::kMTIA);
    // 返回默认流
    return default_streams[device];
  }

  // 获取当前流函数
  c10::Stream getCurrentStream(c10::DeviceIndex device) const override {
    // 惰性初始化设备
    torch::utils::device_lazy_init(at::kMTIA);
    // 返回当前流
    return current_streams[device];
  }

  // 设置当前流函数
  void setCurrentStream(const c10::Stream& stream) const override {
    // 惰性初始化设备
    torch::utils::device_lazy_init(at::kMTIA);
    // 根据流的设备索引设置当前流
    current_streams[stream.device_index()] = stream;
  }

  // 获取当前设备函数
  c10::DeviceIndex getCurrentDevice() const override {
    // 惰性初始化设备
    torch::utils::device_lazy_init(at::kMTIA);
    // 返回当前设备
    return current_device;
  }

  // 设置当前设备函数
  void setCurrentDevice(c10::DeviceIndex device) const override {
    // 惰性初始化设备
    torch::utils::device_lazy_init(at::kMTIA);
    // 如果当前设备与要设置的设备不同，则更新当前设备
    if (current_device != device) {
      current_device = device;
    }
  }
};

// 使用 at::MTIAHooksRegistry 注册 MTIAHooks
using at::MTIAHooksRegistry;
using at::RegistererMTIAHooksRegistry;

REGISTER_MTIA_HOOKS(MTIAHooks);

// 使用 C10_REGISTER_GUARD_IMPL 宏注册 MTIA 的实现
C10_REGISTER_GUARD_IMPL(MTIA, MTIAGuardImpl);

// 命名空间结束声明
} // namespace torch::mtia
```