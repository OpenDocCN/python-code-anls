# `.\pytorch\aten\src\ATen\cuda\PeerToPeerAccess.cpp`

```py
// 引入ATen库中与CUDA对等访问有关的头文件
#include <ATen/cuda/PeerToPeerAccess.h>

// 引入ATen库中CUDA上下文管理相关的头文件
#include <ATen/cuda/CUDAContext.h>

// 引入C10库中CUDA缓存分配器相关的头文件
#include <c10/cuda/CUDACachingAllocator.h>

// 引入C10库中CUDA设备管理相关的头文件
#include <c10/cuda/CUDAGuard.h>

// 引入C10库中异常处理相关的头文件
#include <c10/util/Exception.h>

// 引入C10库中整数范围迭代器相关的头文件
#include <c10/util/irange.h>

// 引入vector标准库
#include <vector>

// 定义在at::cuda命名空间下
namespace at::cuda {

// 静态变量，用于记录每对设备之间是否允许P2P复制的信息
static std::vector<int8_t> p2pAccessEnabled_;

// 静态变量，记录CUDA设备的数量，默认为-1
static int64_t num_devices_ = -1;

// 定义在at::cuda命名空间下的detail命名空间
namespace detail {

// 初始化P2P访问缓存，设置设备数目为num_devices
void init_p2p_access_cache(int64_t num_devices) {
  // p2pAccessEnabled记录设备对之间是否允许P2P复制的状态，-1表示未知
  p2pAccessEnabled_.clear();  // 清空现有的P2P访问状态信息
  p2pAccessEnabled_.resize(num_devices * num_devices, -1);  // 根据设备数目重新分配空间
  num_devices_ = num_devices;  // 记录设备总数

  // 对角线上的设备对允许P2P复制，即设备自身可以访问自身
  for (const auto i : c10::irange(num_devices)) {
    p2pAccessEnabled_[i * num_devices + i] = 1;
  }
}

}  // namespace detail

// 获取设备dev到dev_to_access之间的P2P访问是否允许
bool get_p2p_access(int dev, int dev_to_access) {
  // 惰性初始化CUDA上下文
  at::globalContext().lazyInitCUDA();

  // 检查设备dev是否在有效范围内
  TORCH_CHECK(dev >= 0 || dev < num_devices_,
              dev, " is not a device");

  // 检查设备dev_to_access是否在有效范围内
  TORCH_CHECK(dev_to_access >= 0 || dev_to_access < num_devices_,
              dev_to_access, " is not a device");

  // 内部断言，确保P2P访问缓存已经初始化
  TORCH_INTERNAL_ASSERT(num_devices_ >= 0, "p2p access cache not initialized");

  // 获取dev到dev_to_access的P2P访问状态缓存
  auto &cache = p2pAccessEnabled_[dev * num_devices_ + dev_to_access];

  // 如果缓存值已知，则直接返回
  if (cache != -1) {
    return cache;
  }

  // 否则，查询CUDA API获取设备之间的P2P访问能力
  int result = 0;
  C10_CUDA_CHECK(cudaDeviceCanAccessPeer(&result, dev, dev_to_access));
  cache = result ? 1 : 0;

  // 如果P2P访问允许，则启用对等访问
  if (cache) {
    CUDACachingAllocator::enablePeerAccess(dev, dev_to_access);
  }

  // 返回P2P访问是否允许的结果
  return cache;
}

}  // namespace at::cuda::detail


这段代码是一个CUDA库中关于设备对等访问的实现。它包括初始化设备对之间P2P（Peer-to-Peer）访问的缓存，以及检索特定设备对之间P2P访问状态的函数。
```