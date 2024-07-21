# `.\pytorch\torch\csrc\distributed\c10d\CudaDMAConnectivity.cpp`

```
// 如果未定义 USE_ROCM 并且定义了 PYTORCH_C10_DRIVER_API_SUPPORTED
#if !defined(USE_ROCM) && defined(PYTORCH_C10_DRIVER_API_SUPPORTED)

// 包含 DMAConnectivity.hpp 头文件，用于 DMA 连通性检测
#include <torch/csrc/distributed/c10d/DMAConnectivity.hpp>

// 包含 CUDA 异常和驱动 API 头文件
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/driver_api.h>

// 包含 CUDA 运行时和 NVML 头文件
#include <cuda_runtime.h>
#include <nvml.h>

// 命名空间开始
namespace {

// 定义最大的 NVLink 数量为 64
constexpr int max_nvlinks = 64;

// 函数：获取指定设备索引的总线 ID
std::string get_bus_id(int device_idx) {
  char bus_id[80];
  cudaDeviceProp prop{};
  // 获取设备属性，填充 prop 结构
  C10_CUDA_CHECK(cudaGetDeviceProperties(&prop, device_idx));
  // 格式化总线 ID 字符串
  snprintf(
      bus_id,
      sizeof(bus_id),
      NVML_DEVICE_PCI_BUS_ID_FMT,
      prop.pciDomainID,
      prop.pciBusID,
      prop.pciDeviceID);
  return std::string(bus_id);
}

// 结构体：NVLinkDetector，实现 DMAConnectivityDetector 接口
struct C10_EXPORT NVLinkDetector : public c10d::DMAConnectivityDetector {
  c10::intrusive_ptr<c10d::DMAConnectivity> detect() override {
    int num_devices;
    // 获取 CUDA 可见设备的数量
    C10_CUDA_CHECK(cudaGetDeviceCount(&num_devices));

    // 创建 num_devices x num_devices 的连接矩阵
    std::vector<std::vector<int>> matrix;
    matrix.reserve(num_devices);
    for (int i = 0; i < num_devices; ++i) {
      matrix.emplace_back(num_devices, 0);
    }

    // 获取所有可见设备的总线 ID
    std::unordered_map<std::string, int> bus_id_to_device_idx;
    std::vector<std::string> bus_ids;
    bus_ids.reserve(num_devices);
    for (int i = 0; i < num_devices; ++i) {
      auto bus_id = get_bus_id(i);
      bus_id_to_device_idx.emplace(bus_id, i);
      bus_ids.push_back(std::move(bus_id));
    }

    // 使用 driver_api 获取所有 bus_ids 对应的 nvml 设备
    auto driver_api = c10::cuda::DriverAPI::get();
    std::vector<nvmlDevice_t> nvml_devices(num_devices, nullptr);
    for (int i = 0; i < num_devices; ++i) {
      TORCH_CHECK_EQ(
          driver_api->nvmlDeviceGetHandleByPciBusId_v2_(
              bus_ids[i].c_str(), &nvml_devices[i]),
          NVML_SUCCESS);
    }

    // 创建包含每个设备交换链接数量的向量
    std::vector<int> switch_link_count(num_devices, 0);


这段代码主要涉及 CUDA 设备管理和 NVLink 连接检测的功能实现，通过注释逐行解释其功能和作用。
    // 遍历每个设备
    for (int i = 0; i < num_devices; ++i) {
      // 遍历当前设备的每个 NVLink
      for (int link = 0; link < max_nvlinks; ++link) {
        nvmlReturn_t ret;
        nvmlIntNvLinkDeviceType_t deviceType;
        // 获取当前设备上指定 NVLink 的远程设备类型
        ret = driver_api->nvmlDeviceGetNvLinkRemoteDeviceType_(
            nvml_devices[i], link, &deviceType);
        // 如果获取失败，说明已经遍历完当前设备上的所有 NVLinks
        if (ret != NVML_SUCCESS) {
          // 我们已经遍历完连接到该设备的所有 NVLink，这个错误是无害的。
          // API 没有提供可靠的方法来获取可以传递给 API 的最大链接值。
          // 因此，我们简单地增加链接值，直到 API 失败或达到预定义的最大值。
          break;
        }
        // 如果远程设备是 GPU
        if (deviceType == NVML_NVLINK_DEVICE_TYPE_GPU) {
          nvmlPciInfo_t pciInfo;
          // 获取远程 GPU 设备的 PCI 信息
          TORCH_CHECK_EQ(
              driver_api->nvmlDeviceGetNvLinkRemotePciInfo_v2_(
                  nvml_devices[i], link, &pciInfo),
              NVML_SUCCESS);
          // 查找远程设备的索引
          auto it = bus_id_to_device_idx.find(pciInfo.busId);
          // 如果找到了远程设备的索引
          if (it != bus_id_to_device_idx.end()) {
            // 如果当前设备和远程设备不是同一个设备
            if (i != it->second) {
              // 增加连接矩阵中的连接计数
              matrix[i][it->second] += 1;
            }
          }
        // 如果远程设备是 NVSwitch
        } else if (deviceType == NVML_NVLINK_DEVICE_TYPE_SWITCH) {
          // 增加 NVSwitch 的连接计数
          switch_link_count[i] += 1;
        }
      }
    }

    // 处理 NVSwitch 的连接
    // 简化起见，假设所有 NVSwitch 都相互连接
    for (int i = 0; i < num_devices; ++i) {
      for (int j = 0; j < num_devices; ++j) {
        // 如果是同一个设备，跳过
        if (i == j) {
          continue;
        }
        // 更新连接矩阵，增加两个设备之间的连接数，取最小的 NVSwitch 连接数
        matrix[i][j] += std::min(switch_link_count[i], switch_link_count[j]);
      }
    }

    // 返回一个包含连接信息的 DMAConnectivity 对象
    return c10::make_intrusive<c10d::DMAConnectivity>(
        c10::DeviceType::CUDA, "nvlink", std::move(matrix));
  }
};

struct RegisterDetector {
  RegisterDetector() {
    // 在构造函数中注册 DMA 连通性检测器，针对 CUDA 设备类型，使用 "nvlink"，并使用 NVLinkDetector 类创建一个对象
    register_dma_connectivity_detector(
        c10::DeviceType::CUDA, "nvlink", c10::make_intrusive<NVLinkDetector>());
  }
};

// 创建静态 RegisterDetector 对象，用于在程序启动时自动注册 DMA 连通性检测器
static RegisterDetector register_detector_;

} // namespace
#endif


注释中描述了每行代码的作用，包括注册 DMA 连通性检测器的详细过程和静态对象的定义。
```