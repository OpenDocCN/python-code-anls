# `.\pytorch\torch\csrc\distributed\c10d\DMAConnectivity.hpp`

```py
#pragma once
// 预处理指令：确保此头文件只被包含一次

#include <optional>
// 包含标准库头文件 <optional>，提供可选值的支持

#include <ATen/ATen.h>
// 包含 ATen 库的头文件 <ATen/ATen.h>

namespace c10d {

struct TORCH_API DMAConnectivity : c10::intrusive_ptr_target {
  // DMAConnectivity 结构体定义，继承自 c10::intrusive_ptr_target

  c10::DeviceType device_type;
  // 设备类型，使用 c10::DeviceType 枚举

  std::string connection_type;
  // 连接类型，使用 std::string 存储连接类型的名称

  // 这是一个 NxN 矩阵，表示 N 个设备之间的连接性，
  // 其中每个元素 matrix[i][j] 表示设备 i 和设备 j 之间的连接情况。
  // 值为 0 表示设备 i 和 j 之间没有连接。
  // 非零值的具体含义取决于连接类型（例如，对于 NVLink，它表示 NVLink 的数量）。
  std::vector<std::vector<int>> matrix;

  explicit DMAConnectivity(
      c10::DeviceType device_type,
      std::string connection_type,
      std::vector<std::vector<int>> matrix);
  // DMAConnectivity 结构体的构造函数声明
};

struct DMAConnectivityDetector : c10::intrusive_ptr_target {
  // DMAConnectivityDetector 结构体定义，继承自 c10::intrusive_ptr_target

  virtual c10::intrusive_ptr<DMAConnectivity> detect() = 0;
  // 纯虚函数 detect()，返回 c10::intrusive_ptr<DMAConnectivity>，
  // 用于检测并获取设备之间的连接性信息

  virtual ~DMAConnectivityDetector() {}
  // 虚析构函数，用于多态对象的安全销毁
};

C10_EXPORT void register_dma_connectivity_detector(
    c10::DeviceType device_type,
    const std::string& connection_type,
    c10::intrusive_ptr<DMAConnectivityDetector> detector);
// 注册 DMA 连接性检测器的函数声明

TORCH_API c10::intrusive_ptr<DMAConnectivity> detect_dma_connectivity(
    c10::DeviceType device_type,
    const std::string& connection_type);
// 检测并获取 DMA 连接性信息的函数声明

} // namespace c10d
// c10d 命名空间的结束
```