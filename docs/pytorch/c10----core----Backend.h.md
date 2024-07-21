# `.\pytorch\c10\core\Backend.h`

```
#pragma once

#include <c10/core/DeviceType.h>
#include <c10/core/DispatchKey.h>
#include <c10/core/DispatchKeySet.h>
#include <c10/util/Exception.h>

#include <stdexcept>

namespace c10 {

/**
 * This legacy enum class defines the set of backends supported by old school,
 * code generated Type-based ATen.  A "backend" in this sense roughly
 * corresponds to the cartesian product of (device type, layout), but restricted
 * only to combinations which we actually have kernels for.  Backend does NOT
 * include dtype.
 *
 * The reason we are sunsetting this enum class is because it doesn't allow for
 * open registration; e.g., if you want to add SparseXLA, you'd have to
 * edit this enum; you wouldn't be able to do it out of tree.  DispatchKey is
 * the replacement for Backend which supports open registration.
 *
 * NB: The concept of 'Backend' here disagrees with the notion of backend
 * exposed to users in torch.backends.  Backend here is something like "CPU"
 * or "SparseCUDA"; backend in torch.backends is something like "MKL" or
 * "CUDNN".
 */
enum class Backend {
  CPU,                      // CPU device backend
  CUDA,                     // CUDA device backend
  HIP,                      // HIP (AMD GPU) device backend
  VE,                       // NEC Vector Engine device backend
  FPGA,                     // FPGA device backend
  IPU,                      // Intel Processor Graphics (IPU) device backend
  XPU,                      // Generic accelerator device backend
  SparseCPU,                // Sparse CPU tensor backend
  SparseCUDA,               // Sparse CUDA tensor backend
  SparseCsrCPU,             // Sparse CSR CPU tensor backend
  SparseCsrCUDA,            // Sparse CSR CUDA tensor backend
  SparseHIP,                // Sparse HIP tensor backend
  SparseVE,                 // Sparse VE tensor backend
  SparseXPU,                // Sparse XPU tensor backend
  SparsePrivateUse1,        // Sparse private use 1 tensor backend
  SparseCsrHIP,             // Sparse CSR HIP tensor backend
  SparseCsrVE,              // Sparse CSR VE tensor backend
  SparseCsrXPU,             // Sparse CSR XPU tensor backend
  SparseCsrPrivateUse1,     // Sparse CSR private use 1 tensor backend
  MAIA,                     // MAIA accelerator backend
  XLA,                      // XLA (TensorFlow XLA) device backend
  Vulkan,                   // Vulkan device backend
  Metal,                    // Metal device backend
  Meta,                     // Meta backend
  QuantizedCPU,             // Quantized CPU tensor backend
  QuantizedCUDA,            // Quantized CUDA tensor backend
  QuantizedXPU,             // Quantized XPU tensor backend
  QuantizedPrivateUse1,     // Quantized private use 1 tensor backend
  Undefined,                // Undefined backend
  MkldnnCPU,                // MKL-DNN (Math Kernel Library for Deep Neural Networks) CPU backend
  MPS,                      // MPS (Multi-Processing Service) device backend
  HPU,                      // HPU (High-Performance Unit) device backend
  Lazy,                     // Lazy tensor backend
  MTIA,                     // MTIA (Mobile Tensor Integration Accelerator) backend
  PrivateUse1,              // Private use 1 backend
  NumOptions                // Number of backend options
};

/**
 * Convert a DispatchKey to a corresponding Backend enumeration.
 *
 * @param t DispatchKey to convert
 * @return Corresponding Backend enumeration
 */
inline Backend dispatchKeyToBackend(DispatchKey t) {
  if (t == DispatchKey::CPU || t == DispatchKey::AutogradCPU) {
    return Backend::CPU;
  } else if (t == DispatchKey::CUDA || t == DispatchKey::AutogradCUDA) {
    return Backend::CUDA;
  } else if (t == DispatchKey::HIP) {
    return Backend::HIP;
  } else if (t == DispatchKey::VE) {
    return Backend::VE;
  } else if (t == DispatchKey::FPGA) {
    return Backend::FPGA;
  } else if (t == DispatchKey::MAIA) {
    return Backend::MAIA;
  } else if (t == DispatchKey::XLA || t == DispatchKey::AutogradXLA) {
    return Backend::XLA;
  } else if (t == DispatchKey::Lazy || t == DispatchKey::AutogradLazy) {
    return Backend::Lazy;
  } else if (t == DispatchKey::MPS || t == DispatchKey::AutogradMPS) {
    return Backend::MPS;
  } else if (t == DispatchKey::Vulkan) {
    return Backend::Vulkan;
  } else if (t == DispatchKey::Metal) {
    return Backend::Metal;
  } else if (t == DispatchKey::Meta) {
    return Backend::Meta;
  } else if (t == DispatchKey::SparseCPU) {
    return Backend::SparseCPU;
  } else if (t == DispatchKey::SparseCUDA) {
    return Backend::SparseCUDA;
  } else if (t == DispatchKey::SparseHIP) {
    return Backend::SparseHIP;
  } else if (t == DispatchKey::SparseVE) {
    return Backend::SparseVE;
  } else if (t == DispatchKey::SparsePrivateUse1) {
    return Backend::SparsePrivateUse1;
  } else if (t == DispatchKey::SparseCsrCPU) {
    # 如果 tensor 类型为 SparseCsrCPU，则返回对应的 Backend 值
    return Backend::SparseCsrCPU;
  } else if (t == DispatchKey::SparseCsrCUDA) {
    # 如果 tensor 类型为 SparseCsrCUDA，则返回对应的 Backend 值
    return Backend::SparseCsrCUDA;
  } else if (t == DispatchKey::SparseCsrHIP) {
    # 如果 tensor 类型为 SparseCsrHIP，则返回对应的 Backend 值
    return Backend::SparseCsrHIP;
  } else if (t == DispatchKey::SparseCsrVE) {
    # 如果 tensor 类型为 SparseCsrVE，则返回对应的 Backend 值
    return Backend::SparseCsrVE;
  } else if (t == DispatchKey::SparseCsrPrivateUse1) {
    # 如果 tensor 类型为 SparseCsrPrivateUse1，则返回对应的 Backend 值
    return Backend::SparseCsrPrivateUse1;
  } else if (t == DispatchKey::MkldnnCPU) {
    # 如果 tensor 类型为 MkldnnCPU，则返回对应的 Backend 值
    return Backend::MkldnnCPU;
  } else if (t == DispatchKey::QuantizedCPU) {
    # 如果 tensor 类型为 QuantizedCPU，则返回对应的 Backend 值
    return Backend::QuantizedCPU;
  } else if (t == DispatchKey::QuantizedCUDA) {
    # 如果 tensor 类型为 QuantizedCUDA，则返回对应的 Backend 值
    return Backend::QuantizedCUDA;
  } else if (t == DispatchKey::IPU || t == DispatchKey::AutogradIPU) {
    # 如果 tensor 类型为 IPU 或 AutogradIPU，则返回对应的 Backend 值
    return Backend::IPU;
  } else if (t == DispatchKey::XPU || t == DispatchKey::AutogradXPU) {
    # 如果 tensor 类型为 XPU 或 AutogradXPU，则返回对应的 Backend 值
    return Backend::XPU;
  } else if (t == DispatchKey::SparseXPU) {
    # 如果 tensor 类型为 SparseXPU，则返回对应的 Backend 值
    return Backend::SparseXPU;
  } else if (t == DispatchKey::SparseCsrXPU) {
    # 如果 tensor 类型为 SparseCsrXPU，则返回对应的 Backend 值
    return Backend::SparseCsrXPU;
  } else if (t == DispatchKey::QuantizedXPU) {
    # 如果 tensor 类型为 QuantizedXPU，则返回对应的 Backend 值
    return Backend::QuantizedXPU;
  } else if (t == DispatchKey::QuantizedPrivateUse1) {
    # 如果 tensor 类型为 QuantizedPrivateUse1，则返回对应的 Backend 值
    return Backend::QuantizedPrivateUse1;
  } else if (t == DispatchKey::HPU || t == DispatchKey::AutogradHPU) {
    # 如果 tensor 类型为 HPU 或 AutogradHPU，则返回对应的 Backend 值
    return Backend::HPU;
  } else if (t == DispatchKey::MTIA || t == DispatchKey::AutogradMTIA) {
    # 如果 tensor 类型为 MTIA 或 AutogradMTIA，则返回对应的 Backend 值
    return Backend::MTIA;
  } else if (
      t == DispatchKey::PrivateUse1 || t == DispatchKey::AutogradPrivateUse1) {
    # 如果 tensor 类型为 PrivateUse1 或 AutogradPrivateUse1，则返回对应的 Backend 值
    return Backend::PrivateUse1;
  } else if (t == DispatchKey::Undefined) {
    # 如果 tensor 类型为 Undefined，则返回对应的 Backend 值
    return Backend::Undefined;
  } else {
    # 如果遇到未知的 tensor 类型 ID，则抛出错误并显示错误信息
    TORCH_CHECK(false, "Unrecognized tensor type ID: ", t);
  }
// 根据给定的后端枚举类型转换为调度键类型，用于指定任务的执行环境
inline DispatchKey backendToDispatchKey(Backend b) {
  // 根据不同的后端类型进行分支选择
  switch (b) {
    // 如果是 CPU 后端，则返回对应的调度键类型 CPU
    case Backend::CPU:
      return DispatchKey::CPU;
    // 如果是 CUDA 后端，则返回对应的调度键类型 CUDA
    case Backend::CUDA:
      return DispatchKey::CUDA;
    // 如果是 HIP 后端，则返回对应的调度键类型 HIP
    case Backend::HIP:
      return DispatchKey::HIP;
    // 如果是 VE 后端，则返回对应的调度键类型 VE
    case Backend::VE:
      return DispatchKey::VE;
    // 如果是 FPGA 后端，则返回对应的调度键类型 FPGA
    case Backend::FPGA:
      return DispatchKey::FPGA;
    // 如果是 MAIA 后端，则返回对应的调度键类型 MAIA
    case Backend::MAIA:
      return DispatchKey::MAIA;
    // 如果是 XLA 后端，则返回对应的调度键类型 XLA
    case Backend::XLA:
      return DispatchKey::XLA;
    // 如果是 Lazy 后端，则返回对应的调度键类型 Lazy
    case Backend::Lazy:
      return DispatchKey::Lazy;
    // 如果是 IPU 后端，则返回对应的调度键类型 IPU
    case Backend::IPU:
      return DispatchKey::IPU;
    // 如果是 XPU 后端，则返回对应的调度键类型 XPU
    case Backend::XPU:
      return DispatchKey::XPU;
    // 如果是 SparseXPU 后端，则返回对应的调度键类型 SparseXPU
    case Backend::SparseXPU:
      return DispatchKey::SparseXPU;
    // 如果是 SparseCsrXPU 后端，则返回对应的调度键类型 SparseCsrXPU
    case Backend::SparseCsrXPU:
      return DispatchKey::SparseCsrXPU;
    // 如果是 SparseCPU 后端，则返回对应的调度键类型 SparseCPU
    case Backend::SparseCPU:
      return DispatchKey::SparseCPU;
    // 如果是 SparseCUDA 后端，则返回对应的调度键类型 SparseCUDA
    case Backend::SparseCUDA:
      return DispatchKey::SparseCUDA;
    // 如果是 SparseHIP 后端，则返回对应的调度键类型 SparseHIP
    case Backend::SparseHIP:
      return DispatchKey::SparseHIP;
    // 如果是 SparseVE 后端，则返回对应的调度键类型 SparseVE
    case Backend::SparseVE:
      return DispatchKey::SparseVE;
    // 如果是 SparsePrivateUse1 后端，则返回对应的调度键类型 SparsePrivateUse1
    case Backend::SparsePrivateUse1:
      return DispatchKey::SparsePrivateUse1;
    // 如果是 SparseCsrCPU 后端，则返回对应的调度键类型 SparseCsrCPU
    case Backend::SparseCsrCPU:
      return DispatchKey::SparseCsrCPU;
    // 如果是 SparseCsrCUDA 后端，则返回对应的调度键类型 SparseCsrCUDA
    case Backend::SparseCsrCUDA:
      return DispatchKey::SparseCsrCUDA;
    // 如果是 SparseCsrHIP 后端，则返回对应的调度键类型 SparseCsrHIP
    case Backend::SparseCsrHIP:
      return DispatchKey::SparseCsrHIP;
    // 如果是 SparseCsrVE 后端，则返回对应的调度键类型 SparseCsrVE
    case Backend::SparseCsrVE:
      return DispatchKey::SparseCsrVE;
    // 如果是 SparseCsrPrivateUse1 后端，则返回对应的调度键类型 SparseCsrPrivateUse1
    case Backend::SparseCsrPrivateUse1:
      return DispatchKey::SparseCsrPrivateUse1;
    // 如果是 MkldnnCPU 后端，则返回对应的调度键类型 MkldnnCPU
    case Backend::MkldnnCPU:
      return DispatchKey::MkldnnCPU;
    // 如果是 Vulkan 后端，则返回对应的调度键类型 Vulkan
    case Backend::Vulkan:
      return DispatchKey::Vulkan;
    // 如果是 Metal 后端，则返回对应的调度键类型 Metal
    case Backend::Metal:
      return DispatchKey::Metal;
    // 如果是 Meta 后端，则返回对应的调度键类型 Meta
    case Backend::Meta:
      return DispatchKey::Meta;
    // 如果是 QuantizedCPU 后端，则返回对应的调度键类型 QuantizedCPU
    case Backend::QuantizedCPU:
      return DispatchKey::QuantizedCPU;
    // 如果是 QuantizedCUDA 后端，则返回对应的调度键类型 QuantizedCUDA
    case Backend::QuantizedCUDA:
      return DispatchKey::QuantizedCUDA;
    // 如果是 QuantizedPrivateUse1 后端，则返回对应的调度键类型 QuantizedPrivateUse1
    case Backend::QuantizedPrivateUse1:
      return DispatchKey::QuantizedPrivateUse1;
    // 如果是 Undefined 后端，则返回对应的调度键类型 Undefined
    case Backend::Undefined:
      return DispatchKey::Undefined;
    // 如果是 MPS 后端，则返回对应的调度键类型 MPS
    case Backend::MPS:
      return DispatchKey::MPS;
    // 如果是 HPU 后端，则返回对应的调度键类型 HPU
    case Backend::HPU:
      return DispatchKey::HPU;
    // 如果是 MTIA 后端，则返回对应的调度键类型 MTIA
    case Backend::MTIA:
      return DispatchKey::MTIA;
    // 如果是 PrivateUse1 后端，则返回对应的调度键类型 PrivateUse1
    case Backend::PrivateUse1:
      return DispatchKey::PrivateUse1;
    // 如果遇到未知的后端类型，则抛出运行时错误
    default:
      throw std::runtime_error("Unknown backend");
  }
}

// 根据给定的后端枚举类型转换为设备类型，用于指定资源的使用环境
inline DeviceType backendToDeviceType(Backend b) {
  // 根据不同的后端类型进行分支选择
  switch (b) {
    // 如果是 CPU 相关后端，则返回对应的设备类型 CPU
    case Backend::CPU:
    case Backend::MkldnnCPU:
    case Backend::SparseCPU:
    case Backend::SparseCsrCPU:
    case Backend::QuantizedCPU:
      return DeviceType::CPU;
    // 如果是 CUDA 相关后端，则返回对应的设备类型 CUDA
    case Backend::CUDA:
    case Backend::SparseCUDA:
    case Backend::QuantizedCUDA:
    case Backend::SparseCsrCUDA:
      return DeviceType::CUDA;
    // 如果是 HIP 后端，则返回对应的设备类型 HIP
    case Backend::HIP:
      return DeviceType::HIP;
    // 如果是 VE 后端，则返回对应的设备类型 VE
    case Backend::VE:
      return DeviceType::VE;
    // 如果是 FPGA 后端，则返回对应的设备类型 FPGA
    case Backend::FPGA:
      return DeviceType::FPGA;
    // 如果是 MAIA 后端，则返回对应的设备类型 MAIA
    case Backend::MAIA:
      return DeviceType::MAIA;
    // 如果是 XLA 后端，则返回对应的设备类型 XLA
    case Backend::XLA:
      return DeviceType::XLA;
    // 对于未处理的后端类型，抛出运行时错误
    default:
      throw std::runtime_error("Unknown backend");
  }
}
    # 根据不同的后端类型返回相应的设备类型
    case Backend::Lazy:
      return DeviceType::Lazy;
    # 如果后端类型是 SparseHIP，则返回 HIP 设备类型
    case Backend::SparseHIP:
      return DeviceType::HIP;
    # 如果后端类型是 SparseVE，则返回 VE 设备类型
    case Backend::SparseVE:
      return DeviceType::VE;
    # 如果后端类型是 SparseCsrHIP，则返回 HIP 设备类型
    case Backend::SparseCsrHIP:
      return DeviceType::HIP;
    # 如果后端类型是 SparseCsrVE，则返回 VE 设备类型
    case Backend::SparseCsrVE:
      return DeviceType::VE;
    # 如果后端类型是 IPU，则返回 IPU 设备类型
    case Backend::IPU:
      return DeviceType::IPU;
    # 如果后端类型是 XPU、SparseXPU、SparseCsrXPU 或 QuantizedXPU，则返回 XPU 设备类型
    case Backend::XPU:
    case Backend::SparseXPU:
    case Backend::SparseCsrXPU:
    case Backend::QuantizedXPU:
      return DeviceType::XPU;
    # 如果后端类型是 Vulkan，则返回 Vulkan 设备类型
    case Backend::Vulkan:
      return DeviceType::Vulkan;
    # 如果后端类型是 Metal，则返回 Metal 设备类型
    case Backend::Metal:
      return DeviceType::Metal;
    # 如果后端类型是 Meta，则返回 Meta 设备类型
    case Backend::Meta:
      return DeviceType::Meta;
    # 如果后端类型是 MPS，则返回 MPS 设备类型
    case Backend::MPS:
      return DeviceType::MPS;
    # 如果后端类型是 HPU，则返回 HPU 设备类型
    case Backend::HPU:
      return DeviceType::HPU;
    # 如果后端类型是 MTIA，则返回 MTIA 设备类型
    case Backend::MTIA:
      return DeviceType::MTIA;
    # 如果后端类型是 PrivateUse1、SparsePrivateUse1、SparseCsrPrivateUse1 或 QuantizedPrivateUse1，则返回 PrivateUse1 设备类型
    case Backend::PrivateUse1:
    case Backend::SparsePrivateUse1:
    case Backend::SparseCsrPrivateUse1:
    case Backend::QuantizedPrivateUse1:
      return DeviceType::PrivateUse1;
    # 如果后端类型是 Undefined，则抛出错误，提示未定义的后端不是有效的设备类型
    case Backend::Undefined:
      TORCH_CHECK(false, "Undefined backend is not a valid device type");
    # 默认情况下，如果后端类型未知，则抛出错误，提示未知的后端类型
    default:
      TORCH_CHECK(false, "Unknown backend");
}

// 将枚举类型 Backend 转换为对应的字符串表示
inline const char* toString(Backend b) {
  switch (b) {
    case Backend::CPU:
      return "CPU";
    case Backend::CUDA:
      return "CUDA";
    case Backend::HIP:
      return "HIP";
    case Backend::VE:
      return "VE";
    case Backend::FPGA:
      return "FPGA";
    case Backend::XPU:
      return "XPU";
    case Backend::IPU:
      return "IPU";
    case Backend::MAIA:
      return "MAIA";
    case Backend::XLA:
      return "XLA";
    case Backend::Lazy:
      return "Lazy";
    case Backend::MPS:
      return "MPS";
    case Backend::SparseCPU:
      return "SparseCPU";
    case Backend::SparseCUDA:
      return "SparseCUDA";
    case Backend::SparseHIP:
      return "SparseHIP";
    case Backend::SparseVE:
      return "SparseVE";
    case Backend::SparseXPU:
      return "SparseXPU";
    case Backend::SparsePrivateUse1:
      return "SparsePrivateUse1";
    case Backend::SparseCsrCPU:
      return "SparseCsrCPU";
    case Backend::SparseCsrCUDA:
      return "SparseCsrCUDA";
    case Backend::SparseCsrHIP:
      return "SparseCsrHIP";
    case Backend::SparseCsrVE:
      return "SparseCsrVE";
    case Backend::SparseCsrXPU:
      return "SparseCsrXPU";
    case Backend::SparseCsrPrivateUse1:
      return "SparseCsrPrivateUse1";
    case Backend::MkldnnCPU:
      return "MkldnnCPU";
    case Backend::Vulkan:
      return "Vulkan";
    case Backend::Metal:
      return "Metal";
    case Backend::Meta:
      return "Meta";
    case Backend::QuantizedCPU:
      return "QuantizedCPU";
    case Backend::QuantizedCUDA:
      return "QuantizedCUDA";
    case Backend::QuantizedXPU:
      return "QuantizedXPU";
    case Backend::QuantizedPrivateUse1:
      return "QuantizedPrivateUse1";
    case Backend::HPU:
      return "HPU";
    case Backend::MTIA:
      return "MTIA";
    case Backend::PrivateUse1:
      return "PrivateUseOne";
    default:
      return "UNKNOWN_BACKEND";
  }
}

// 检查给定的 Backend 是否为稀疏类型
inline bool isSparse(Backend b) {
  switch (b) {
    case Backend::SparseXPU:
    case Backend::SparseCPU:
    case Backend::SparseCUDA:
    case Backend::SparseHIP:
    case Backend::SparseVE:
    case Backend::SparsePrivateUse1:
      return true;
    default:
      return false;
  }
}

// 检查给定的 Backend 是否为稀疏 CSR 类型
inline bool isSparseCsr(Backend b) {
  switch (b) {
    case Backend::SparseCsrXPU:
    case Backend::SparseCsrCPU:
    case Backend::SparseCsrCUDA:
    case Backend::SparseCsrHIP:
    case Backend::SparseCsrVE:
    case Backend::SparseCsrPrivateUse1:
      return true;
    default:
      return false;
  }
}

} // namespace c10
```