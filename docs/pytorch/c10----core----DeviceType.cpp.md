# `.\pytorch\c10\core\DeviceType.cpp`

```
// 包含必要的头文件
#include <c10/core/DeviceType.h>
#include <c10/util/Exception.h>
#include <atomic> // 原子操作的支持
#include <mutex> // 互斥锁的支持

// 命名空间开始
namespace c10 {

// 根据设备类型返回设备名称的字符串表示，可以选择是否小写
std::string DeviceTypeName(DeviceType d, bool lower_case) {
  // 根据不同的设备类型返回相应的字符串表示
  switch (d) {
    case DeviceType::CPU:
      return lower_case ? "cpu" : "CPU";
    case DeviceType::CUDA:
      return lower_case ? "cuda" : "CUDA";
    case DeviceType::OPENGL:
      return lower_case ? "opengl" : "OPENGL";
    case DeviceType::OPENCL:
      return lower_case ? "opencl" : "OPENCL";
    case DeviceType::MKLDNN:
      return lower_case ? "mkldnn" : "MKLDNN";
    case DeviceType::IDEEP:
      return lower_case ? "ideep" : "IDEEP";
    case DeviceType::HIP:
      return lower_case ? "hip" : "HIP";
    case DeviceType::VE:
      return lower_case ? "ve" : "VE";
    case DeviceType::FPGA:
      return lower_case ? "fpga" : "FPGA";
    case DeviceType::MAIA:
      return lower_case ? "maia" : "MAIA";
    case DeviceType::XLA:
      return lower_case ? "xla" : "XLA";
    case DeviceType::Lazy:
      return lower_case ? "lazy" : "LAZY";
    case DeviceType::MPS:
      return lower_case ? "mps" : "MPS";
    case DeviceType::Vulkan:
      return lower_case ? "vulkan" : "VULKAN";
    case DeviceType::Metal:
      return lower_case ? "metal" : "METAL";
    case DeviceType::XPU:
      return lower_case ? "xpu" : "XPU";
    case DeviceType::Meta:
      return lower_case ? "meta" : "META";
    case DeviceType::HPU:
      return lower_case ? "hpu" : "HPU";
    case DeviceType::IPU:
      return lower_case ? "ipu" : "IPU";
    case DeviceType::MTIA:
      return lower_case ? "mtia" : "MTIA";
    case DeviceType::PrivateUse1:
      // 如果是私有使用的设备类型，调用函数获取其后端名称的字符串表示
      return get_privateuse1_backend(/*lower_case=*/lower_case);
    default:
      // 报错处理：未知的设备类型，输出错误信息
      TORCH_CHECK(
          false,
          "Unknown device: ",
          static_cast<int16_t>(d),
          ". If you have recently updated the caffe2.proto file to add a new "
          "device type, did you forget to update the DeviceTypeName() "
          "function to reflect such recent changes?");
      // 下面的代码不会运行，但是需要防止一些编译器警告
      return "";
  }
}

// 检查设备类型是否有效
bool isValidDeviceType(DeviceType d) {
  switch (d) {
    // 列出所有有效的设备类型
    case DeviceType::CPU:
    case DeviceType::CUDA:
    case DeviceType::OPENGL:
    case DeviceType::OPENCL:
    case DeviceType::MKLDNN:
    case DeviceType::IDEEP:
    // 省略了其他已知的有效设备类型以及私有使用的设备类型
    // 对给定的 DeviceType 进行检查，判断是否为支持的硬件类型之一
    case DeviceType::HIP:
    case DeviceType::VE:
    case DeviceType::FPGA:
    case DeviceType::MAIA:
    case DeviceType::XLA:
    case DeviceType::Lazy:
    case DeviceType::MPS:
    case DeviceType::Vulkan:
    case DeviceType::Metal:
    case DeviceType::XPU:
    case DeviceType::Meta:
    case DeviceType::HPU:
    case DeviceType::IPU:
    case DeviceType::MTIA:
    case DeviceType::PrivateUse1:
      // 如果是上述列出的任一硬件类型，则返回 true
      return true;
    // 如果不是上述列出的硬件类型，则返回 false
    default:
      return false;
  }
}

// 重载输出流操作符，用于打印 DeviceType 类型的名称
std::ostream& operator<<(std::ostream& stream, DeviceType type) {
  // 调用 DeviceTypeName 函数获取 DeviceType 对应的名称，并选择是否小写输出
  stream << DeviceTypeName(type, /* lower case */ true);
  return stream;
}

// 在这里同时使用互斥锁和原子变量的原因是：
// (1) 互斥锁在写入时是必需的：
//     我们需要先检查值并可能出现错误，然后再设置值（确保没有其他线程在中途插入）。
//     这个操作可以较慢，因为它仅在导入时发生一次。
// (2) 原子变量在读取时是必需的：
//     每当用户打印 privateuse1 设备名称时，他们需要读取这个变量。
//     虽然可能性不大，但如果另一个线程正试图设置此变量，同时另一个线程正在打印设备名称，我们会出现数据竞争。
//     我们可以重用相同的互斥锁，但读取原子变量会更快。
static std::atomic<bool> privateuse1_backend_name_set;
static std::string privateuse1_backend_name;
static std::mutex privateuse1_lock;

// 获取 privateuse1 后端的名称
std::string get_privateuse1_backend(bool lower_case) {
  // 应用与 Python 解释器标签上相同的原子读取内存排序逻辑
  auto name_registered =
      privateuse1_backend_name_set.load(std::memory_order_acquire);
  // 如果标志已设置，则保证 privateuse1_backend_name 已设置，且不会再写入。
  auto backend_name =
      name_registered ? privateuse1_backend_name : "privateuseone";
  return backend_name;
}

// 注册 privateuse1 后端的名称
void register_privateuse1_backend(const std::string& backend_name) {
  std::lock_guard<std::mutex> guard(privateuse1_lock);
  // 检查是否已设置 privateuse1_backend_name，如果已设置，则后端名称必须与当前后端名称相同。
  TORCH_CHECK(
      !privateuse1_backend_name_set.load() ||
          privateuse1_backend_name == backend_name,
      "torch.register_privateuse1_backend() has already been set! Current backend: ",
      privateuse1_backend_name);

  privateuse1_backend_name = backend_name;
  // 不变条件：一旦设置了此标志，privateuse1_backend_name 将永远不会再被写入。
  privateuse1_backend_name_set.store(true, std::memory_order_relaxed);
}

// 检查是否已注册 privateuse1 后端
bool is_privateuse1_backend_registered() {
  return privateuse1_backend_name_set.load(std::memory_order_acquire);
}

} // namespace c10
```