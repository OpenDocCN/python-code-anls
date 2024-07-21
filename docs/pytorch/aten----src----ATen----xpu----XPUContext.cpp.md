# `.\pytorch\aten\src\ATen\xpu\XPUContext.cpp`

```
namespace at::xpu {
namespace {

/*
 * 目前，有一个设备属性池，包含每个计算设备的信息和能力。
 *
 * 当首次请求设备的属性时，设备属性会延迟初始化。
 */
DeviceIndex num_gpus = -1; // GPU 设备数量，默认为 -1
c10::once_flag init_flag; // 用于确保初始化函数仅执行一次的标志
std::deque<c10::once_flag> device_prop_flags; // 每个设备属性初始化状态的队列
std::vector<DeviceProp> device_properties; // 存储设备属性信息的向量

std::deque<c10::once_flag> device_global_idx_flags; // 全局索引初始化状态的队列
std::vector<int32_t> device_global_idxs; // 存储设备全局索引的向量

// 初始化 XPU 上下文的向量信息
void initXPUContextVectors() {
  num_gpus = c10::xpu::device_count(); // 获取当前系统中的 GPU 数量
  device_prop_flags.resize(num_gpus); // 调整设备属性初始化状态队列的大小
  device_properties.resize(num_gpus); // 调整设备属性信息向量的大小
  device_global_idx_flags.resize(num_gpus); // 调整设备全局索引初始化状态队列的大小
  device_global_idxs.resize(num_gpus); // 调整设备全局索引向量的大小
}

// 初始化特定设备的属性信息
void initDeviceProperty(DeviceIndex device) {
  c10::xpu::get_device_properties(&device_properties[device], device); // 获取指定设备的属性信息
}

// 初始化特定设备的全局索引
void initDeviceGlobalIdx(DeviceIndex device) {
  sycl::device& raw_device = c10::xpu::get_raw_device(device); // 获取指定设备的 SYCL 设备
  // 获取与 SYCL 平台关联的所有 SYCL 设备
  auto devices = sycl::device::get_devices();
  auto match_device = [raw_device](const auto& dev) -> bool {
    return raw_device == dev;
  };
  // 查找匹配的 SYCL 设备
  auto it = std::find_if(devices.begin(), devices.end(), match_device);
  // 检查是否找到了对应的 SYCL 设备
  TORCH_CHECK(
      it != devices.end(), "Can't find the global index of XPU device.");
  // 计算设备在设备列表中的索引并存储
  device_global_idxs[device] =
      static_cast<int32_t>(std::distance(devices.begin(), it));
}

// 内联函数：检查设备索引的有效性
inline void check_device(DeviceIndex device) {
  TORCH_CHECK(
      device >= 0 && device < num_gpus,
      "device is out of range, device is ",
      static_cast<int>(device),
      ", total number of device is ",
      static_cast<int>(num_gpus),
      ".");
}

} // anonymous namespace

// 返回当前设备的设备属性
DeviceProp* getCurrentDeviceProperties() {
  auto device = c10::xpu::current_device(); // 获取当前设备索引
  return getDeviceProperties(device); // 返回当前设备的设备属性
}

// 返回指定设备的设备属性
DeviceProp* getDeviceProperties(DeviceIndex device) {
  c10::call_once(init_flag, initXPUContextVectors); // 确保初始化 XPU 上下文向量信息
  if (device == -1)
    device = c10::xpu::current_device(); // 如果设备索引为 -1，则获取当前设备索引
  check_device(device); // 检查设备索引的有效性
  c10::call_once(device_prop_flags[device], initDeviceProperty, device); // 确保特定设备的属性已初始化
  return &device_properties[device]; // 返回特定设备的设备属性指针
}

// 返回 XPU 设备在 SYCL 设备列表中的全局索引
int32_t getGlobalIdxFromDevice(DeviceIndex device) {
  c10::call_once(init_flag, initXPUContextVectors); // 确保初始化 XPU 上下文向量信息
  check_device(device); // 检查设备索引的有效性
  c10::call_once(device_global_idx_flags[device], initDeviceGlobalIdx, device); // 确保特定设备的全局索引已初始化
  return device_global_idxs[device]; // 返回特定设备的全局索引
}

} // namespace at::xpu
```