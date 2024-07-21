# `.\pytorch\torch\csrc\distributed\c10d\GlooDeviceFactory.cpp`

```py
// 包含 GlooDeviceFactory.hpp 头文件，其中包含了 Gloo 设备工厂的定义和实现
#include <torch/csrc/distributed/c10d/GlooDeviceFactory.hpp>

// 如果定义了 USE_C10D_GLOO 宏，则继续编译以下内容
#ifdef USE_C10D_GLOO

// 包含标准库头文件
#include <cstdlib>

// 包含 C10 库中的异常处理头文件
#include <c10/util/Exception.h>

// 如果定义了 GLOO_HAVE_TRANSPORT_TCP 宏，则包含 TCP 传输设备的头文件
#if GLOO_HAVE_TRANSPORT_TCP
#include <gloo/transport/tcp/device.h>
#endif

// 如果定义了 GLOO_HAVE_TRANSPORT_TCP_TLS 宏，则包含 TCP TLS 传输设备的头文件
#if GLOO_HAVE_TRANSPORT_TCP_TLS
#include <gloo/transport/tcp/tls/device.h>
#endif

// 如果定义了 GLOO_HAVE_TRANSPORT_UV 宏，则包含 UV 传输设备的头文件
#if GLOO_HAVE_TRANSPORT_UV
#include <gloo/transport/uv/device.h>
#endif

// 在 Linux 上，检查是否存在 TCP 传输设备，如果不存在则报错
#ifdef __linux__
#if !GLOO_HAVE_TRANSPORT_TCP
#error "Expected the tcp transport to be available on Linux."
#endif
#endif

// 在 macOS 上，检查是否存在 UV 传输设备，如果不存在则报错
#ifdef __APPLE__
#if !GLOO_HAVE_TRANSPORT_UV
#error "Expected the uv transport to be available on macOS."
#endif
#endif

// 命名空间 c10d，定义 GlooDeviceRegistry 共享注册表，用于存储 Gloo 设备对象的注册信息
namespace c10d {

// 定义一个共享注册表 GlooDeviceRegistry，注册的对象类型为 ::gloo::transport::Device
C10_DEFINE_SHARED_REGISTRY_WITHOUT_WARNING(
    GlooDeviceRegistry,
    ::gloo::transport::Device,
    const std::string& /* interface */,
    const std::string& /* hostname */);

// 如果定义了 GLOO_HAVE_TRANSPORT_TCP 宏，则定义一个静态函数 makeTCPDevice，
// 创建并返回一个 TCP 传输设备的共享指针
#if GLOO_HAVE_TRANSPORT_TCP
static std::shared_ptr<::gloo::transport::Device> makeTCPDevice(
    const std::string& interfaceName,
    const std::string& hostname) {
  // 检查接口名和主机名不能同时为空，否则抛出异常
  TORCH_CHECK(
      !interfaceName.empty() || !hostname.empty(),
      "GlooDeviceFactory::makeTCPDevice(): interface or hostname "
      "can't be empty");

  // 创建 TCP 传输设备的属性对象 attr
  ::gloo::transport::tcp::attr attr;
  if (!interfaceName.empty()) {
    attr.iface = interfaceName;
  } else {
    attr.hostname = hostname;
  }
  // 调用 Gloo 库中的函数创建 TCP 传输设备并返回
  return ::gloo::transport::tcp::CreateDevice(attr);
}

// 将 makeTCPDevice 函数注册到 GlooDeviceRegistry 共享注册表中，
// 使用键值 'LINUX' 和 'TCP' 进行注册，供后续灵活选择使用
C10_REGISTER_CREATOR(GlooDeviceRegistry, LINUX, makeTCPDevice);
C10_REGISTER_CREATOR(GlooDeviceRegistry, TCP, makeTCPDevice);
#endif

// 如果定义了 GLOO_HAVE_TRANSPORT_TCP_TLS 宏，则定义一个静态函数 makeTCPTLSDevice，
// 创建并返回一个 TCP TLS 传输设备的共享指针
#if GLOO_HAVE_TRANSPORT_TCP_TLS
static std::string cstr_to_std_string(const char* chars) {
  return std::string(chars != nullptr ? chars : "");
}

static std::shared_ptr<::gloo::transport::Device> makeTCPTLSDevice(
    const std::string& interface,
    const std::string& hostname) {
  // 检查接口名和主机名不能同时为空，否则抛出异常
  TORCH_CHECK(
      !interface.empty() || !hostname.empty(),
      "GlooDeviceFactory::makeTCPTLSDevice(): interface or hostname "
      "can't be empty");

  // 创建 TCP TLS 传输设备的属性对象 attr
  ::gloo::transport::tcp::attr attr;
  if (!interface.empty()) {
    attr.iface = interface;
  } else {
    attr.hostname = hostname;
  }
  // 从环境变量中获取 TLS 相关参数，创建 TCP TLS 传输设备并返回
  const auto pkey =
      cstr_to_std_string(std::getenv("GLOO_DEVICE_TRANSPORT_TCP_TLS_PKEY"));
  const auto cert =
      cstr_to_std_string(std::getenv("GLOO_DEVICE_TRANSPORT_TCP_TLS_CERT"));
  const auto caFile =
      cstr_to_std_string(std::getenv("GLOO_DEVICE_TRANSPORT_TCP_TLS_CA_FILE"));
  const auto caPath =
      cstr_to_std_string(std::getenv("GLOO_DEVICE_TRANSPORT_TCP_TLS_CA_PATH"));
  return ::gloo::transport::tcp::tls::CreateDevice(
      attr, pkey, cert, caFile, caPath);
}

// 将 makeTCPTLSDevice 函数注册到 GlooDeviceRegistry 共享注册表中，
// 使用键值 'TCP_TLS' 进行注册，供后续灵活选择使用
C10_REGISTER_CREATOR(GlooDeviceRegistry, TCP_TLS, makeTCPTLSDevice);
#endif

// 如果定义了 GLOO_HAVE_TRANSPORT_UV 宏，则继续处理 UV 传输设备的注册和实现
#if GLOO_HAVE_TRANSPORT_UV
// 创建一个 UV 设备的工厂方法，根据接口名和主机名创建对应的设备
static std::shared_ptr<::gloo::transport::Device> makeUVDevice(
    const std::string& interfaceName,
    const std::string& hostname) {
  // 检查接口名和主机名至少有一个非空，否则抛出错误信息
  TORCH_CHECK(
      !interfaceName.empty() || !hostname.empty(),
      "GlooDeviceFactory::makeUVDevice(): interface or hostname "
      "can't be empty");

  // 创建 uv::attr 结构体
  ::gloo::transport::uv::attr attr;
  // 如果接口名非空，设置为属性的接口名
  if (!interfaceName.empty()) {
    attr.iface = interfaceName;
  } else {
    // 否则设置为属性的主机名
    attr.hostname = hostname;
  }
  // 使用属性创建 UV 设备并返回
  return ::gloo::transport::uv::CreateDevice(attr);
}

// Registry priority is per key identifier. We register UV to `APPLE` for
// the flexibility of other application to override by priority. Register
// UV to `UV` for env "GLOO_DEVICE_TRANSPORT" override.
// 注册 UV 设备到不同的注册表键，例如 APPLE，WIN32，UV，以支持优先级覆盖
C10_REGISTER_CREATOR(GlooDeviceRegistry, APPLE, makeUVDevice);
C10_REGISTER_CREATOR(GlooDeviceRegistry, WIN32, makeUVDevice);
C10_REGISTER_CREATOR(GlooDeviceRegistry, UV, makeUVDevice);

#endif

namespace {
// 创建 Gloo 设备的工厂方法，根据接口名和主机名创建对应的设备
std::shared_ptr<::gloo::transport::Device> makeGlooDevice(
    const std::string& interfaceName,
    const std::string& hostName) {
  // 获取环境变量 "GLOO_DEVICE_TRANSPORT" 的值作为传输名称
  static auto transportName = getenv("GLOO_DEVICE_TRANSPORT");
  // 如果有传输名称，使用注册表创建对应的设备并返回
  if (transportName) {
    return GlooDeviceRegistry()->Create(transportName, interfaceName, hostName);
  }

#ifdef __linux__
  // 如果是 Linux 系统，使用 LINUX 键注册表创建设备并返回
  return GlooDeviceRegistry()->Create("LINUX", interfaceName, hostName);
#endif

#ifdef __APPLE__
  // 如果是苹果系统，使用 APPLE 键注册表创建设备并返回
  return GlooDeviceRegistry()->Create("APPLE", interfaceName, hostName);
#endif

#ifdef _WIN32
  // 如果是 Windows 系统，使用 WIN32 键注册表创建设备并返回
  return GlooDeviceRegistry()->Create("WIN32", interfaceName, hostName);
#endif

  // 如果没有匹配的系统和环境变量，返回空指针
  return nullptr;
}
} // anonymous namespace

// 根据接口名创建 Gloo 设备的方法
std::shared_ptr<::gloo::transport::Device> GlooDeviceFactory::
    makeDeviceForInterface(const std::string& interfaceName) {
  // 使用 makeGlooDevice 方法创建设备，并检查设备是否有效
  auto device = makeGlooDevice(interfaceName, "");
  if (!device) {
    // 如果设备无效，抛出错误信息
    TORCH_CHECK(false, "makeDeviceForInterface(): unsupported gloo device");
  }
  // 返回创建的设备
  return device;
}

// 根据主机名创建 Gloo 设备的方法
std::shared_ptr<::gloo::transport::Device> GlooDeviceFactory::
    makeDeviceForHostname(const std::string& hostname) {
  // 使用 makeGlooDevice 方法创建设备，并检查设备是否有效
  auto device = makeGlooDevice("", hostname);
  if (!device) {
    // 如果设备无效，抛出错误信息
    TORCH_CHECK(false, "makeDeviceForHostname(): unsupported gloo device");
  }
  // 返回创建的设备
  return device;
}

} // namespace c10d

#endif // USE_C10D_GLOO
```