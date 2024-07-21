# `.\pytorch\c10\xpu\XPUFunctions.cpp`

```py
/*
 * Note [Device Management]
 *
 * An Intel GPU device qualifies as a type of SYCL device. This classification
 * allows for the runtime querying of Intel GPU device information through the
 * SYCL runtime library.
 *
 * Device status is managed through a SYCL device pool, with SYCL devices
 * determined at runtime. There's currently a SYCL device pool that is lazily
 * created and only initialized once, ensuring thread-local safety. Each device
 * within the device pool shares the same default context.
 */
namespace c10::xpu {
namespace {

/*
 * Note [Device Management]
 *
 * An Intel GPU device qualifies as a type of SYCL device. This classification
 * allows for the runtime querying of Intel GPU device information through the
 * SYCL runtime library.
 *
 * Device status is managed through a SYCL device pool, with SYCL devices
 * determined at runtime. There's currently a SYCL device pool that is lazily
 * created and only initialized once, ensuring thread-local safety. Each device
 * within the device pool shares the same default context.
 */
c10::once_flag init_flag; // 用于确保 initGlobalDevicePoolState 函数只被调用一次的标志

thread_local DeviceIndex curDeviceIndex = 0; // 当前线程局部存储的设备索引

struct DevicePool {
  std::vector<std::unique_ptr<sycl::device>> devices; // 存储 SYCL 设备的唯一指针向量
  std::unique_ptr<sycl::context> context; // 每个设备池共享的上下文对象的唯一指针
} gDevicePool; // 全局设备池对象

void enumDevices(std::vector<std::unique_ptr<sycl::device>>& devices) {
  auto platform_list = sycl::platform::get_platforms();
  // 从特定平台枚举出的 GPU 设备
  for (const auto& platform : platform_list) {
    if (platform.get_backend() != sycl::backend::ext_oneapi_level_zero) {
      continue;
    }
    auto device_list = platform.get_devices();
    for (const auto& device : device_list) {
      if (device.is_gpu()) {
        devices.push_back(std::make_unique<sycl::device>(device));
      }
    }
  }
}

inline void initGlobalDevicePoolState() {
  // 枚举所有 GPU 设备并记录它们
  enumDevices(gDevicePool.devices);
  if (gDevicePool.devices.empty()) {
    TORCH_WARN("XPU device count is zero!"); // 如果设备数为零，则发出警告
    return;
  }

#ifdef _WIN32
  // 在 Windows 上默认上下文特性默认禁用
  std::vector<sycl::device> deviceList;
  for (auto it = gDevicePool.devices.begin(); it != gDevicePool.devices.end();
       ++it) {
    deviceList.push_back(*(*it));
  }
  gDevicePool.context = std::make_unique<sycl::context>(deviceList); // 创建设备池的上下文对象
#else
  // 默认上下文用于每个 Intel GPU 设备，允许从任何 GPU 设备检索上下文
  gDevicePool.context = std::make_unique<sycl::context>(
      gDevicePool.devices[0]->get_platform().ext_oneapi_get_default_context());
#endif
}

inline void initDevicePoolCallOnce() {
  c10::call_once(init_flag, initGlobalDevicePoolState); // 使用 c10::call_once 确保 initGlobalDevicePoolState 只被调用一次
}

void initDeviceProperties(DeviceProp* device_prop, int device) {
  using namespace sycl::info;
  using namespace sycl::ext;
  // 获取与设备索引关联的原始 SYCL 设备
  auto& raw_device = *gDevicePool.devices[device];

  // 初始化与特定设备关联的设备属性
#define ASSIGN_DEVICE_PROP(property) \
  device_prop->property = raw_device.get_info<device::property>(); // 获取设备的特定属性信息
#define ASSIGN_EXT_DEVICE_PROP(property, default_value)                      \
  // 如果原始设备中存在指定的扩展属性，将其值赋给设备属性结构体中的对应属性；否则使用默认值
  device_prop->property = raw_device.has(sycl::aspect::ext_intel_##property) \
      ? raw_device.get_info<intel::info::device::property>()                 \
      : default_value;

#define ASSIGN_DEVICE_ASPECT(member) \
  // 检查原始设备是否具有指定的设备方面，将结果存储在设备属性结构体中
  device_prop->has_##member = raw_device.has(sycl::aspect::member);

// 使用宏批量为设备属性赋值
AT_FORALL_XPU_DEVICE_PROPERTIES(ASSIGN_DEVICE_PROP);

// 获取设备的平台名称，并赋值给设备属性结构体的平台名称字段
device_prop->platform_name =
    raw_device.get_info<device::platform>().get_info<platform::name>();

// 使用宏批量为扩展设备属性赋值
AT_FORALL_XPU_EXT_DEVICE_PROPERTIES(ASSIGN_EXT_DEVICE_PROP);

// 使用宏批量为设备方面赋值
AT_FORALL_XPU_DEVICE_ASPECT(ASSIGN_DEVICE_ASPECT);

// 函数返回，没有返回值
return;
}

inline void check_device(DeviceIndex device) {
  // TODO: 直接使用 c10::Device::MAX_NUM_DEVICES。DeviceIndex 是 int8_t 类型的值，
  // PyTorch 可识别的最大 GPU 数量为 64。因此，我们需要检查是否发生了溢出。
  // 当 DeviceIndex 更改为 int16_t，并提供 c10::Device::MAX_NUM_DEVICES 时，
  // 应直接使用它来检查是否检测到了过多的 XPU 设备。
  TORCH_CHECK(
      gDevicePool.devices.size() <= std::numeric_limits<DeviceIndex>::max(),
      "Too many XPU devices, DeviceIndex overflowed");
  auto total = static_cast<DeviceIndex>(gDevicePool.devices.size());
  TORCH_CHECK(
      device >= 0 && device < total,
      "device is out of range, device is ",
      device,
      ", total number of device is ",
      total,
      ".");
}

} // 匿名命名空间结束

// 获取原始设备对象的引用，并确保设备池初始化后进行调用
sycl::device& get_raw_device(DeviceIndex device) {
  initDevicePoolCallOnce();
  // 检查给定设备索引的有效性
  check_device(device);
  return *gDevicePool.devices[device];
}

// 获取设备上下文对象的引用，并确保设备池初始化后进行调用
sycl::context& get_device_context() {
  initDevicePoolCallOnce();
  // 检查设备池的上下文对象是否已初始化，若未初始化则抛出错误信息
  TORCH_CHECK(
      gDevicePool.context,
      "Device pool initialization failed, you might not have an XPU device.")
  return *gDevicePool.context;
}

// 获取设备的属性，并存储在指定的设备属性结构体中
void get_device_properties(DeviceProp* device_prop, DeviceIndex device) {
  initDevicePoolCallOnce();
  // 检查设备属性结构体的有效性
  TORCH_CHECK(device_prop, "device_prop is an invalid pointer.");
  // 检查给定设备索引的有效性
  check_device(device);
  // 初始化设备属性
  initDeviceProperties(device_prop, device);
}

// 从指针获取设备索引
DeviceIndex get_device_idx_from_pointer(void* ptr) {
  initDevicePoolCallOnce();
  // 检查指针的有效性
  TORCH_CHECK(ptr, "ptr is an invalid pointer.");
  // 获取指针所指向对象的类型，并检查是否为设备类型指针
  auto type = sycl::get_pointer_type(ptr, get_device_context());
  TORCH_CHECK(
      type == sycl::usm::alloc::device, "ptr is not a device type pointer.");

  // 获取指针所在设备的原始设备对象
  sycl::device raw_device = sycl::get_pointer_device(ptr, get_device_context());
  // 匹配设备池中的设备，查找指定设备的索引
  auto match_device = [raw_device](const auto& device) -> bool {
    return raw_device == *device;
  };
  auto it = std::find_if(
      gDevicePool.devices.begin(), gDevicePool.devices.end(), match_device);
  // 检查是否能在设备池中找到匹配的设备，否则抛出错误信息
  TORCH_CHECK(
      it != gDevicePool.devices.end(),
      "Can't find the pointer from XPU devices.");
  // 返回找到的设备索引
  return static_cast<DeviceIndex>(
      std::distance(gDevicePool.devices.begin(), it));
}
// 初始化设备池（如果尚未初始化），确保调用一次
DeviceIndex device_count() {
  initDevicePoolCallOnce();
  // 返回当前设备池中设备的数量，转换为 DeviceIndex 类型并返回
  return static_cast<DeviceIndex>(gDevicePool.devices.size());
}

// 确保设备数量非零，并返回设备数量
DeviceIndex device_count_ensure_non_zero() {
  auto count = device_count();
  // 如果设备数量为零，则抛出错误信息，指示没有可用的 XPU 设备
  TORCH_CHECK(count, "No XPU devices are available.");
  return count;
}

// 返回当前设备的索引
DeviceIndex current_device() {
  initDevicePoolCallOnce();
  // 返回当前设备的索引 curDeviceIndex
  return curDeviceIndex;
}

// 设置当前设备的索引为指定的设备
void set_device(DeviceIndex device) {
  initDevicePoolCallOnce();
  // 检查设备是否有效
  check_device(device);
  // 将当前设备索引 curDeviceIndex 设置为指定的设备索引 device
  curDeviceIndex = device;
}

// 切换当前设备索引到指定的设备，并返回之前的设备索引
c10::DeviceIndex exchange_device(c10::DeviceIndex to_device) {
  // 获取当前设备索引
  auto cur_device = current_device();
  // 如果目标设备索引和当前设备索引相同，直接返回当前设备索引
  if (to_device == cur_device) {
    return cur_device;
  }
  // 将当前设备索引切换为目标设备索引 to_device
  set_device(to_device);
  // 返回切换前的当前设备索引
  return cur_device;
}

// 可能切换当前设备索引到指定的设备，与 exchange_device 功能相同
c10::DeviceIndex maybe_exchange_device(c10::DeviceIndex to_device) {
  return exchange_device(to_device);
}

} // namespace c10::xpu
```