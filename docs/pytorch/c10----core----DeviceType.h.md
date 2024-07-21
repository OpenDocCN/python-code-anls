# `.\pytorch\c10\core\DeviceType.h`

```
#pragma once

// 这段代码与 caffe2/proto/caffe2.proto 直接同步，但不需要将 Protobuf 头文件引入 ATen/core，
// 这样避免了需要对构建系统进行大量修改。
// 如果修改此处，请确保与 caffe2.proto 文件保持同步。

#include <c10/macros/Export.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <ostream>
#include <string>

namespace c10 {

// 定义了所有具有 BackendComponent 并因此参与每个后端功能调度键的设备类型。
// 大多数后端都包括在内，但不包括 PrivateUse2 和 PrivateUse3。
#define C10_FORALL_BACKEND_DEVICE_TYPES(_, extra) \
  _(CPU, extra)                                   \
  _(CUDA, extra)                                  \
  _(HIP, extra)                                   \
  _(XLA, extra)                                   \
  _(MPS, extra)                                   \
  _(IPU, extra)                                   \
  _(XPU, extra)                                   \
  _(HPU, extra)                                   \
  _(VE, extra)                                    \
  _(Lazy, extra)                                  \
  _(Meta, extra)                                  \
  _(MTIA, extra)                                  \
  _(PrivateUse1, extra)

// 设备类型的枚举类，使用 int8_t 类型
enum class DeviceType : int8_t {
  CPU = 0,          // CPU 设备类型
  CUDA = 1,         // CUDA 设备类型
  MKLDNN = 2,       // 专用于显式 MKLDNN
  OPENGL = 3,       // OpenGL 设备类型
  OPENCL = 4,       // OpenCL 设备类型
  IDEEP = 5,        // IDEEP 设备类型
  HIP = 6,          // AMD HIP 设备类型
  FPGA = 7,         // FPGA 设备类型
  MAIA = 8,         // ONNX Runtime / Microsoft 设备类型
  XLA = 9,          // XLA / TPU 设备类型
  Vulkan = 10,      // Vulkan 设备类型
  Metal = 11,       // Metal 设备类型
  XPU = 12,         // XPU 设备类型
  MPS = 13,         // MPS 设备类型
  Meta = 14,        // Meta 设备类型（无实际数据的张量）
  HPU = 15,         // HPU / HABANA 设备类型
  VE = 16,          // SX-Aurora / NEC 设备类型
  Lazy = 17,        // 懒惰张量设备类型
  IPU = 18,         // Graphcore IPU 设备类型
  MTIA = 19,        // 元训练和推理设备类型
  PrivateUse1 = 20, // PrivateUse1 设备类型
  // 注意：如果添加更多设备类型：
  // - 需要在 DeviceType.cpp 中修改 DeviceTypeName 和 isValidDeviceType 的实现
  // - 修改下方的设备类型数量
  COMPILE_TIME_MAX_DEVICE_TYPES = 21,
};

// 定义了常量对应于各个设备类型
constexpr DeviceType kCPU = DeviceType::CPU;
constexpr DeviceType kCUDA = DeviceType::CUDA;
constexpr DeviceType kHIP = DeviceType::HIP;
constexpr DeviceType kFPGA = DeviceType::FPGA;
constexpr DeviceType kMAIA = DeviceType::MAIA;
constexpr DeviceType kXLA = DeviceType::XLA;
constexpr DeviceType kMPS = DeviceType::MPS;
constexpr DeviceType kMeta = DeviceType::Meta;
constexpr DeviceType kVulkan = DeviceType::Vulkan;
constexpr DeviceType kMetal = DeviceType::Metal;
constexpr DeviceType kXPU = DeviceType::XPU;
constexpr DeviceType kHPU = DeviceType::HPU;
constexpr DeviceType kVE = DeviceType::VE;
constexpr DeviceType kLazy = DeviceType::Lazy;
constexpr DeviceType kIPU = DeviceType::IPU;
constexpr DeviceType kMTIA = DeviceType::MTIA;
constexpr DeviceType kPrivateUse1 = DeviceType::PrivateUse1;

// 定义了显式的整数常量
# 定义一个 constexpr 常量，表示编译时设备类型的最大数量
constexpr int COMPILE_TIME_MAX_DEVICE_TYPES =
    static_cast<int>(DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES);

# 使用 static_assert 进行断言检查，确保 COMPILE_TIME_MAX_DEVICE_TYPES 不大于 21
static_assert(
    COMPILE_TIME_MAX_DEVICE_TYPES <= 21,
    "Hey!  You seem to be adding a lot of new DeviceTypes.  The intent was "
    "for this constant to reflect the actual number of DeviceTypes we support "
    "in PyTorch; it's important that this number is not too large as we "
    "use this to allocate stack arrays in some places in our code.  If you "
    "are indeed just adding the 20th device type, feel free to change "
    "the check to 32; but if you are adding some sort of extensible device "
    "types registration, please be aware that you are affecting code that "
    "this number is small.  Try auditing uses of this constant.");

# 声明一系列 C10_API 的函数和操作符，用于设备类型的处理和注册
C10_API std::string DeviceTypeName(DeviceType d, bool lower_case = false);

C10_API bool isValidDeviceType(DeviceType d);

C10_API std::ostream& operator<<(std::ostream& stream, DeviceType type);

C10_API void register_privateuse1_backend(const std::string& backend_name);
C10_API std::string get_privateuse1_backend(bool lower_case = true);

C10_API bool is_privateuse1_backend_registered();

# 声明命名空间 c10 中的特化 hash 结构，用于 DeviceType 的哈希函数
} // namespace c10
namespace std {
template <>
struct hash<c10::DeviceType> {
  std::size_t operator()(c10::DeviceType k) const {
    return std::hash<int>()(static_cast<int>(k));
  }
};
} // namespace std

# 声明命名空间 torch 中使用 c10::DeviceType，NOLINTNEXTLINE(misc-unused-using-decls) 避免 lint 提示未使用声明
namespace torch {
// NOLINTNEXTLINE(misc-unused-using-decls)
using c10::DeviceType;
} // namespace torch
```