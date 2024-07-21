# `.\pytorch\aten\src\ATen\xpu\detail\XPUHooks.cpp`

```py
namespace at::xpu::detail {

// 初始化 XPU 相关设置，记录 API 使用情况
void XPUHooks::initXPU() const {
  C10_LOG_API_USAGE_ONCE("aten.init.xpu");
  // 获取当前系统上的 XPU 设备数量，并确保至少有一个设备
  const auto device_count = c10::xpu::device_count_ensure_non_zero();
  // 初始化 XPUCachingAllocator，为每个设备分配缓存
  c10::xpu::XPUCachingAllocator::init(device_count);
}

// 返回是否支持 XPU
bool XPUHooks::hasXPU() const {
  return true;
}

// 返回 XPU 的配置信息
std::string XPUHooks::showConfig() const {
  return "XPU backend";
}

// 根据设备获取其全局索引
int32_t XPUHooks::getGlobalIdxFromDevice(const at::Device& device) const {
  // 检查设备是否为 XPU 类型
  TORCH_CHECK(device.is_xpu(), "Only the XPU device type is expected.");
#ifdef _WIN32
  // Windows 系统不支持默认上下文，因此无法获取设备的全局索引
  TORCH_CHECK(
      false,
      "Default context is not supported on XPU on Windows. So we can NOT find its global index of the ATen device.");
#else
  // 返回设备对应的全局索引
  return at::xpu::getGlobalIdxFromDevice(device.index());
#endif
}

// 获取 XPU 的随机数生成器
Generator XPUHooks::getXPUGenerator(DeviceIndex device_index) const {
  return make_generator<at::XPUGeneratorImpl>(device_index);
}

// 获取默认的 XPU 随机数生成器
const Generator& XPUHooks::getDefaultXPUGenerator(
    DeviceIndex device_index) const {
  return at::xpu::detail::getDefaultXPUGenerator(device_index);
}

// 根据指针获取其所属的设备
Device XPUHooks::getDeviceFromPtr(void* data) const {
#ifdef _WIN32
  // Windows 系统不支持默认上下文，因此无法获取指针对应的设备
  TORCH_CHECK(
      false,
      "Default context is not supported on XPU on Windows. So we can NOT find the ATen device of a pointer.");
#else
  // 返回指针所在的设备
  return at::xpu::getDeviceFromPtr(data);
#endif
}

// 获取系统上的 XPU 数量
c10::DeviceIndex XPUHooks::getNumGPUs() const {
  return at::xpu::device_count();
}

// 获取当前设备的索引
DeviceIndex XPUHooks::current_device() const {
  return c10::xpu::current_device();
}

// 同步指定设备的流
void XPUHooks::deviceSynchronize(DeviceIndex device_index) const {
  // 只同步我们已经预留的 SYCL 队列，请参阅 [Synchronize Streams on Device] 注释
  c10::xpu::syncStreamsOnDevice(device_index);
}

// 获取固定内存分配器
Allocator* XPUHooks::getPinnedMemoryAllocator() const {
  return at::xpu::getPinnedMemoryAllocator();
}

// 检查指针是否是固定内存指针
bool XPUHooks::isPinnedPtr(const void* data) const {
  // 如果 XPU 不可用，则返回 false
  if (!at::xpu::is_available()) {
    return false;
  }

  // 检查指针是否是主机固定内存分配器的类型
  return sycl::usm::alloc::host ==
      sycl::get_pointer_type(data, c10::xpu::get_device_context());
}

REGISTER_XPU_HOOKS(XPUHooks);

} // namespace at::xpu::detail
```