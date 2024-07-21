# `.\pytorch\aten\src\ATen\cuda\CUDAContext.cpp`

```py
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/util/CallOnce.h>

#include <ATen/cuda/CUDAConfig.h>
#include <deque>
#include <vector>

namespace at::cuda {

namespace {

DeviceIndex num_gpus = -1;  // GPU 设备数量，初始值为 -1
c10::once_flag init_flag;  // 用于标记初始化操作的标志
std::deque<c10::once_flag> device_flags;  // 存储每个设备初始化状态的队列
std::vector<cudaDeviceProp> device_properties;  // 存储每个设备属性信息的向量

void initCUDAContextVectors() {
  num_gpus = c10::cuda::device_count();  // 获取当前系统中的 GPU 数量
  device_flags.resize(num_gpus);  // 调整设备标志队列的大小以匹配 GPU 数量
  device_properties.resize(num_gpus);  // 调整设备属性向量的大小以匹配 GPU 数量
}

void initDeviceProperty(DeviceIndex device_index) {
  cudaDeviceProp device_prop{};
  AT_CUDA_CHECK(cudaGetDeviceProperties(&device_prop, device_index));  // 获取指定设备的属性信息
  device_properties[device_index] = device_prop;  // 将设备属性信息存储到对应位置
}

} // anonymous namespace

// We need this function to force the linking against torch_cuda(_cpp) on Windows.
// If you need to modify this function, please specify a new function and apply
// the changes according to https://github.com/pytorch/pytorch/pull/34288.
// Related issue: https://github.com/pytorch/pytorch/issues/31611.
/* Device info */
int warp_size() {
  return getCurrentDeviceProperties()->warpSize;  // 返回当前设备的 warp 大小
}

cudaDeviceProp* getCurrentDeviceProperties() {
  auto device = c10::cuda::current_device();  // 获取当前 CUDA 设备索引
  return getDeviceProperties(device);  // 获取当前设备的属性信息指针
}

cudaDeviceProp* getDeviceProperties(c10::DeviceIndex device) {
  c10::call_once(init_flag, initCUDAContextVectors);  // 使用 call_once 确保 CUDA 上下文初始化仅执行一次
  if (device == -1) device = c10::cuda::current_device();  // 如果未指定设备，则使用当前设备索引
  AT_ASSERT(device >= 0 && device < num_gpus, "device=", device, ", num_gpus=", num_gpus);  // 断言设备索引在有效范围内
  c10::call_once(device_flags[device], initDeviceProperty, device);  // 使用 call_once 确保设备属性初始化仅执行一次
  return &device_properties[device];  // 返回指定设备的属性信息指针
}

bool canDeviceAccessPeer(c10::DeviceIndex device, c10::DeviceIndex peer_device) {
  c10::call_once(init_flag, initCUDAContextVectors);  // 使用 call_once 确保 CUDA 上下文初始化仅执行一次
  if (device == -1) device = c10::cuda::current_device();  // 如果未指定设备，则使用当前设备索引
  AT_ASSERT(device >= 0 && device < num_gpus, "device=", device, ", num_gpus=", num_gpus);  // 断言设备索引在有效范围内
  AT_ASSERT(peer_device >= 0 && peer_device < num_gpus, "peer_device=", peer_device, ", num_gpus=", num_gpus);  // 断言对等设备索引在有效范围内
  int can_access = 0;
  AT_CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, device, peer_device));  // 检查设备之间是否可以互相访问
  return can_access != 0;  // 返回设备之间是否可以互相访问的结果
}

Allocator* getCUDADeviceAllocator() {
  return c10::cuda::CUDACachingAllocator::get();  // 获取 CUDA 设备分配器的实例
}

} // namespace at::cuda
```