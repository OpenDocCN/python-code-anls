# `.\pytorch\c10\cuda\CUDADeviceAssertionHost.cpp`

```py
/// 引入必要的CUDA头文件
#include <c10/cuda/CUDADeviceAssertionHost.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/util/Backtrace.h>
#include <c10/util/Exception.h>
#include <c10/util/irange.h>
#include <cuda_runtime.h>

#include <memory>
#include <string>

#ifdef TORCH_USE_CUDA_DSA
#include <chrono>
#include <thread>
#endif

/// 定义CUDA错误检查宏，不包含DSA
#define C10_CUDA_CHECK_WO_DSA(EXPR)                                 \
  do {                                                              \
    const cudaError_t __err = EXPR;                                 \
    c10::cuda::c10_cuda_check_implementation(                       \
        static_cast<int32_t>(__err),                                \
        __FILE__,                                                   \
        __func__, /* Line number data type not well-defined between \
                      compilers, so we perform an explicit cast */  \
        static_cast<uint32_t>(__LINE__),                            \
        false);                                                     \
  } while (0)

namespace c10::cuda {

namespace {

#ifdef TORCH_USE_CUDA_DSA
/// 获取当前设备ID的函数实现
/// 需要自定义此函数以防止CUDAKernelLaunchRegistry的无限初始化循环
int dsa_get_device_id() {
  c10::DeviceIndex device = -1;
  C10_CUDA_CHECK_WO_DSA(c10::cuda::GetDevice(&device));
  return device;
}

/// 获取指定设备的计算能力
/// 注意：这里假设所有支持设备侧断言的CUDA GPU都具有相同的计算能力
/// 这是合理的，因为最新的不支持UVM的CUDA GPU是2014年发布的K80
/// 与更新的GPU混合使用的可能性很小，因此不太需要进行防御性检查
int dsa_get_device_compute_capability(const int device_num) {
  int compute_capability = -1;
  C10_CUDA_CHECK_WO_DSA(cudaDeviceGetAttribute(
      &compute_capability, cudaDevAttrComputeCapabilityMajor, device_num));
  return compute_capability;
}
#endif

/// 获取CUDA设备数量的函数实现
/// 需要自定义此函数以防止CUDAKernelLaunchRegistry的无限初始化循环
int dsa_get_device_count() {
  int device_count = -1;
  C10_CUDA_CHECK_WO_DSA(c10::cuda::GetDeviceCount(&device_count));
  return device_count;
}

/// 检查所有设备是否支持托管内存的函数
bool dsa_check_if_all_devices_support_managed_memory() {
  // 这个函数最适合CUDA架构为Pascal或更新版本的GPU
  // 参考：https://developer.nvidia.com/blog/unified-memory-cuda-beginners/
#ifdef TORCH_USE_CUDA_DSA
  // 遍历所有CUDA设备，检查其计算能力
  for (const auto i : c10::irange(dsa_get_device_count())) {
    if (dsa_get_device_compute_capability(i) < 6) {
      return false;
    }
  }
  return true;
#else
  return false;
#endif
}

} // namespace
} // namespace c10::cuda


这段代码是一些CUDA设备管理的辅助功能和宏定义，主要用于检查CUDA设备的状态和功能支持情况。
// 检查环境变量是否设置为特定值
bool env_flag_set(const char* env_var_name) {
  // 获取环境变量的字符串值
  const char* const env_string = std::getenv(env_var_name);
  // 如果环境变量为空，返回 false；否则比较环境变量值和字符串 "0" 是否相同
  return (env_string == nullptr) ? false : std::strcmp(env_string, "0");
}

/// UVM/managed 内存指针的删除器
void uvm_deleter(DeviceAssertionsData* uvm_assertions_ptr) {
  // 在析构函数中忽略错误
  if (uvm_assertions_ptr) {
    // 释放 CUDA 分配的内存
    C10_CUDA_IGNORE_ERROR(cudaFree(uvm_assertions_ptr));
  }
}

} // namespace

/// 检查内核是否正确运行，通过检查消息缓冲区。BLOCKING.
std::string c10_retrieve_device_side_assertion_info() {
#ifdef TORCH_USE_CUDA_DSA
  // 获取 CUDA 内核启动注册表的单例引用
  const auto& launch_registry = CUDAKernelLaunchRegistry::get_singleton_ref();
  // 如果运行时未启用设备端断言追踪，返回相应的错误信息
  if (!launch_registry.enabled_at_runtime) {
    return "Device-side assertion tracking was not enabled by user.";
  } else if (!launch_registry.do_all_devices_support_managed_memory) {
    // 如果不是所有设备都支持管理内存，则返回相应的错误信息
    return "Device-side assertions disabled because not all devices support managed memory.";
  }

  // 使用一个简单的暂停来等待 GPU 写入错误信息到内存
  // 这段时间不应影响性能，因为通常是在出现问题时才会执行此函数
  std::this_thread::sleep_for(std::chrono::seconds(1));

  // 获取 CUDA 内核启动注册表的快照数据
  const auto launch_data = launch_registry.snapshot();
  const auto& assertion_data = launch_data.first;
  const auto& launch_infos = launch_data.second;

  // 创建一个字符串流，用于构建返回信息
  std::stringstream oss;
  oss << "Looking for device-side assertion failure information...\n";

  // 遍历每个可能由进程管理的设备
  for (const auto device_num : c10::irange(assertion_data.size())) {
    // 获取特定设备的断言数据
    const auto& assertion_data_for_device = assertion_data.at(device_num);

    // 检查是否有断言失败
    const auto failures_found = std::min(
        assertion_data_for_device.assertion_count,
        C10_CUDA_DSA_ASSERTION_COUNT);
    if (failures_found == 0) {
      continue;
    }

    // 如果有断言失败，记录相关信息到输出流中
    oss << failures_found
        << " CUDA device-side assertion failures were found on GPU #"
        << device_num << "!" << std::endl;
    if (assertion_data_for_device.assertion_count >
        C10_CUDA_DSA_ASSERTION_COUNT) {
      oss << "But at least " << assertion_data_for_device.assertion_count
          << " assertion failures occurred on the device" << std::endl;
      oss << "Adjust `C10_CUDA_DSA_ASSERTION_COUNT` if you need more assertion failure info"
          << std::endl;
    }
    // 对于每一个发现的失败，遍历处理
    for (const auto i : c10::irange(failures_found)) {
      // 获取当前断言失败的详细数据
      const auto& self = assertion_data_for_device.assertions[i];
      // 确定使用的启动信息，以处理可能的循环情况
      const auto& launch_info = launch_infos[self.caller % launch_infos.size()];
      // 将断言失败的索引输出到输出流
      oss << "Assertion failure " << i << std::endl;
      // 输出 GPU 断言失败消息
      oss << "  GPU assertion failure message = " << self.assertion_msg
          << std::endl;
      // 输出包含断言的文件和行号
      oss << "  File containing assertion = " << self.filename << ":"
          << self.line_number << std::endl;
      // 输出包含断言的设备函数名
      oss << "  Device function containing assertion = " << self.function_name
          << std::endl;
      // 输出导致断言失败的线程 ID
      oss << "  Thread ID that failed assertion = [" << self.thread_id[0] << ","
          << self.thread_id[1] << "," << self.thread_id[2] << "]" << std::endl;
      // 输出导致断言失败的块 ID
      oss << "  Block ID that failed assertion = [" << self.block_id[0] << ","
          << self.block_id[1] << "," << self.block_id[2] << "]" << std::endl;
      // 如果生成号与调用者匹配，则输出关于内核启动的信息
      if (launch_info.generation_number == self.caller) {
        // 输出包含内核启动的文件和行号
        oss << "  File containing kernel launch = "
            << launch_info.launch_filename << ":" << launch_info.launch_linenum
            << std::endl;
        // 输出包含内核启动的函数名
        oss << "  Function containing kernel launch = "
            << launch_info.launch_function << std::endl;
        // 输出导致失败的内核名称
        oss << "  Name of kernel launched that led to failure = "
            << launch_info.kernel_name << std::endl;
        // 输出启动内核的设备
        oss << "  Device that launched kernel = " << launch_info.device
            << std::endl;
        // 输出内核启动的流
        oss << "  Stream kernel was launched on = " << launch_info.stream
            << std::endl;
        // 输出内核启动位置的回溯信息
        oss << "  Backtrace of kernel launch site = ";
        // 如果允许收集启动堆栈跟踪，则输出启动堆栈跟踪信息；否则输出禁用消息
        if (launch_registry.gather_launch_stacktrace) {
          oss << "Launch stacktracing disabled." << std::endl;
        } else {
          oss << "\n" << launch_info.launch_stacktrace << std::endl;
        }
      } else {
        // 输出 CPU 启动位置的不可用信息，指出循环队列已回绕
        oss << "  CPU launch site info: Unavailable, the circular queue wrapped around. Increase `CUDAKernelLaunchRegistry::max_size`."
            << std::endl;
      }
    }
  }
  // 返回输出流中的所有内容作为字符串
  return oss.str();
// CUDAKernelLaunchRegistry 类的默认构造函数，初始化成员变量
CUDAKernelLaunchRegistry::CUDAKernelLaunchRegistry()
    : do_all_devices_support_managed_memory(
          dsa_check_if_all_devices_support_managed_memory()), // 检查所有设备是否支持托管内存
      gather_launch_stacktrace(check_env_for_enable_launch_stacktracing()), // 检查环境变量以启用启动堆栈跟踪
      enabled_at_runtime(check_env_for_dsa_enabled()) { // 检查环境变量以确定在运行时是否启用 DSA
  // 根据设备数量为 uvm_assertions 动态分配内存，并初始化为 nullptr
  for (C10_UNUSED const auto _ : c10::irange(dsa_get_device_count())) {
    uvm_assertions.emplace_back(nullptr, uvm_deleter);
  }

  // 调整 kernel_launches 的大小为 max_kernel_launches
  kernel_launches.resize(max_kernel_launches);
}

// 检查是否设置了启用启动堆栈跟踪的环境变量
bool CUDAKernelLaunchRegistry::check_env_for_enable_launch_stacktracing()
    const {
  return env_flag_set("PYTORCH_CUDA_DSA_STACKTRACING");
}

// 检查是否设置了启用 DSA 的环境变量
bool CUDAKernelLaunchRegistry::check_env_for_dsa_enabled() const {
  return env_flag_set("PYTORCH_USE_CUDA_DSA");
}

// 向注册表中插入一个 CUDA 内核启动的记录
uint32_t CUDAKernelLaunchRegistry::insert(
    const char* launch_filename,
    const char* launch_function,
    const uint32_t launch_linenum,
    const char* kernel_name,
    const int32_t stream_id) {
#ifdef TORCH_USE_CUDA_DSA
  if (!enabled_at_runtime) { // 如果在运行时未启用 DSA，则返回 0
    return 0;
  }

  // 收集启动堆栈跟踪信息（如果已启用）
  const auto backtrace = gather_launch_stacktrace ? c10::get_backtrace() : "";

  // 使用互斥锁保护临界区
  const std::lock_guard<std::mutex> lock(read_write_mutex);

  // 计算当前生成号并增加
  const auto my_gen_number = generation_number++;

  // 将 kernel_launches 数组中的一个位置更新为新的内核启动信息
  kernel_launches[my_gen_number % max_kernel_launches] = {
      launch_filename,
      launch_function,
      launch_linenum,
      backtrace,
      kernel_name,
      dsa_get_device_id(),
      stream_id,
      my_gen_number};
  
  // 返回生成号作为记录的唯一标识符
  return my_gen_number;
#else
  return 0; // 如果未编译使用 TORCH_USE_CUDA_DSA，则返回 0
#endif
}

// 返回当前注册表的快照，包括设备断言数据和内核启动信息
std::pair<std::vector<DeviceAssertionsData>, std::vector<CUDAKernelLaunchInfo>>
CUDAKernelLaunchRegistry::snapshot() const {
  // 锁定互斥锁，保证在获取快照期间数据的一致性
  const std::lock_guard<std::mutex> lock(read_write_mutex);

  // 构造设备断言数据的快照
  std::vector<DeviceAssertionsData> device_assertions_data;
  for (const auto& x : uvm_assertions) {
    if (x) {
      device_assertions_data.push_back(*x);
    } else {
      device_assertions_data.emplace_back();
    }
  }

  // 返回设备断言数据和内核启动信息的快照
  return std::make_pair(device_assertions_data, kernel_launches);
}

// 返回当前设备的 UVM 断言数据的指针
DeviceAssertionsData* CUDAKernelLaunchRegistry::
    get_uvm_assertions_ptr_for_current_device() {
#ifdef TORCH_USE_CUDA_DSA
  if (!enabled_at_runtime) { // 如果未在运行时启用 DSA，则返回 nullptr
    return nullptr;
  }

  // 获取当前设备的编号
  const auto device_num = dsa_get_device_id();

  // 如果已经为该 GPU 设置了托管内存，返回对托管内存的指针（快速返回路径）
  if (uvm_assertions.at(device_num)) {
      //```
  // 如果已经为该 GPU 设置了托管内存，返回对托管内存的指针（快速返回路径）
  return uvm_assertions.at(device_num).get();
#else
  return nullptr; // 如果未编译使用 TORCH_USE_CUDA_DSA，则返回 nullptr
#endif
}
    return uvm_assertions.at(device_num).get();
  }

  // 在此处需要加锁，以避免在创建新设备断言缓冲区时出现竞争条件
  const std::lock_guard<std::mutex> lock(gpu_alloc_mutex);

  // 如果已经为此 GPU 设置了托管内存，返回指向托管内存的指针。通过加锁确保设备内存只分配一次
  if (uvm_assertions.at(device_num)) {
    return uvm_assertions.at(device_num).get();
  }

  // 否则，设置 GPU 以便能够使用设备端的断言系统
  DeviceAssertionsData* uvm_assertions_ptr = nullptr;

  C10_CUDA_CHECK_WO_DSA(
      cudaMallocManaged(&uvm_assertions_ptr, sizeof(DeviceAssertionsData)));

  C10_CUDA_CHECK_WO_DSA(cudaMemAdvise(
      uvm_assertions_ptr,
      sizeof(DeviceAssertionsData),
      cudaMemAdviseSetPreferredLocation,
      cudaCpuDeviceId));

  // GPU 将在 CPU 内存中建立数据的直接映射，不会生成页面错误
  C10_CUDA_CHECK_WO_DSA(cudaMemAdvise(
      uvm_assertions_ptr,
      sizeof(DeviceAssertionsData),
      cudaMemAdviseSetAccessedBy,
      cudaCpuDeviceId));

  // 从 CPU 初始化内存；否则，可能需要按需创建页面。我们认为 UVM 文档表明，首次访问可能不会遵循首选位置，如果为真，这将是不好的，因为我们希望该内存驻留在主机上以便在断言后访问。在 CPU 上初始化有助于确保内存将驻留在那里。
  *uvm_assertions_ptr = DeviceAssertionsData();

  // 现在将 `uvm_assertions_ptr` 的所有权和生命周期管理传递给 uvm_assertions 的 unique_ptr 向量
  uvm_assertions.at(device_num).reset(uvm_assertions_ptr);

  return uvm_assertions_ptr;
#else
  // 如果没有满足前面条件的情况，返回空指针
  return nullptr;
#endif
}

// 返回静态的 CUDA 核心启动注册表的引用
CUDAKernelLaunchRegistry& CUDAKernelLaunchRegistry::get_singleton_ref() {
  // 静态变量，只会在第一次调用时初始化，保证唯一实例
  static CUDAKernelLaunchRegistry launch_registry;
  return launch_registry;
}

// 检查是否存在失败的 CUDA 核心启动
bool CUDAKernelLaunchRegistry::has_failed() const {
  // 遍历所有的 UVM 断言
  for (const auto& x : uvm_assertions) {
    // 如果某个断言存在且其断言计数大于 0，则认为有失败的情况
    if (x && x->assertion_count > 0) {
      return true;
    }
  }
  // 如果所有断言都未失败，返回 false
  return false;
}

// 结束命名空间 c10::cuda
} // namespace c10::cuda
```