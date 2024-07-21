# `.\pytorch\c10\xpu\XPUStream.cpp`

```py
// 引入头文件：c10/util/CallOnce.h、c10/util/irange.h、c10/xpu/XPUException.h、c10/xpu/XPUStream.h
#include <c10/util/CallOnce.h>
#include <c10/util/irange.h>
#include <c10/xpu/XPUException.h>
#include <c10/xpu/XPUStream.h>

// 引入标准库头文件
#include <atomic>
#include <deque>
#include <mutex>
#include <vector>

// 定义命名空间 c10::xpu，这里使用了匿名命名空间
namespace c10::xpu {
namespace {

// 全局流状态和常量定义
c10::once_flag init_flag;  // 初始化标志，确保初始化操作只执行一次
DeviceIndex num_gpus = -1; // GPU 设备数量，初始值设为 -1
constexpr int kStreamsPerPoolBits = 5;  // 每个池中流的数量（位数）
constexpr int kStreamsPerPool = 1 << kStreamsPerPoolBits;  // 每个池中流的数量（总数）
constexpr int kStreamTypeBits = 3;  // 流类型的位数

// 当第一次请求设备的队列时，SYCL 队列池会延迟初始化。设备标志用于跟踪每个设备的初始化。
// 当请求队列时，按照循环方式从池中返回下一个队列，见 Note [Stream Management]。
std::deque<c10::once_flag> device_flags;  // 设备初始化标志队列
std::vector<std::array<
    std::array<std::unique_ptr<sycl::queue>, kStreamsPerPool>,
    max_compile_time_stream_priorities>>
    streams;  // 存储 SYCL 队列的向量，使用多维数组组织
std::deque<
    std::array<std::atomic<uint32_t>, max_compile_time_stream_priorities>>
    priority_counters;  // 优先级计数器队列

thread_local std::unique_ptr<StreamId[]> current_streams = nullptr;  // 线程本地存储的当前流数组

// Note [StreamId assignment]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~
// 如何分配流 ID？
//
// -- 57 位 --  -- 5 位 -----  -- 3 位 --
//     零         StreamIdIndex   StreamIdType
//
// StreamIdType:
//  000 = 普通优先级队列
//  001 = 高优先级队列
//
// StreamId 是 64 位的，因此我们可以依靠常规的提升规则。
// 我们依赖于 StreamIdIndex 和 StreamIdType 非负的特性。

using StreamIdIndex = uint8_t;  // 流 ID 索引类型定义
enum class StreamIdType : uint8_t {  // 流 ID 类型枚举类
  // 数字越高，优先级越高。
  NORMAL = 0x0,
  HIGH = 0X1,
};

// 重载输出流操作符，用于打印 StreamIdType 类型
inline std::ostream& operator<<(std::ostream& stream, StreamIdType q) {
  switch (q) {
    case StreamIdType::NORMAL:
      return stream << "NORMAL";
    case StreamIdType::HIGH:
      return stream << "HIGH";
    default:
      break;
  }
  return stream << static_cast<int16_t>(q);
}

// 获取流 ID 的类型（高优先级或普通优先级）
inline StreamIdType streamIdType(StreamId s) {
  int mask_for_type = (1 << kStreamTypeBits) - 1;
  auto st = static_cast<StreamIdType>(s & mask_for_type);
  TORCH_CHECK(
      st == StreamIdType::NORMAL || st == StreamIdType::HIGH,
      "invalid StreamId: ",
      s);
  return st;
}

// 获取流 ID 的索引
inline StreamIdIndex streamIdIndex(StreamId s) {
  return static_cast<StreamIdIndex>(
      (s >> kStreamTypeBits) & ((1 << kStreamsPerPoolBits) - 1));
}

// 创建流 ID
inline StreamId makeStreamId(StreamIdType st, StreamIdIndex si) {
  return (static_cast<StreamId>(si) << kStreamTypeBits) |
      static_cast<StreamId>(st);
}

// 初始化全局流状态
void initGlobalStreamState() {
  num_gpus = c10::xpu::device_count();  // 获取 GPU 设备数量
  device_flags.resize(num_gpus);  // 调整设备标志队列大小
  streams.resize(num_gpus);  // 调整存储 SYCL 队列的向量大小
  priority_counters.resize(num_gpus);  // 调整优先级计数器队列大小
}

// 为指定设备创建预留的 SYCL 队列池。应该只调用一次。
// 初始化设备流状态的函数，为指定设备创建多个队列，并设置优先级和属性
void initDeviceStreamState(DeviceIndex device) {
  // 使用命名空间sycl::ext::oneapi::property，以便使用属性
  using namespace sycl::ext::oneapi::property;
  // 需要与StreamIdType对齐。
  // 定义不同优先级的队列属性列表
  const std::vector<sycl::property_list> properties = {
      {sycl::property::queue::in_order(), queue::priority_normal()},
      {sycl::property::queue::in_order(), queue::priority_high()}};
  
  // 对于每个编译时流优先级中的每个索引，初始化流对象
  for (const auto p : c10::irange(max_compile_time_stream_priorities)) {
    for (const auto i : c10::irange(kStreamsPerPool)) {
      auto& stream = streams[device][p][i];
      // 使用给定的属性创建SYCL队列对象
      stream = std::make_unique<sycl::queue>(sycl::queue(
          c10::xpu::get_device_context(),
          c10::xpu::get_raw_device(device),
          c10::xpu::asyncHandler,
          properties[p]));
      
      // 获取GPU跟踪器的实例，如果存在，则追踪GPU流的创建
      const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
      if (C10_UNLIKELY(interp)) {
        (*interp)->trace_gpu_stream_creation(
            c10::kXPU, reinterpret_cast<uintptr_t>(stream.get()));
      }
    }
    // 为每个优先级计数器初始化为0
    priority_counters[device][p] = 0;
  }
}

// 初始化XPU流的函数，仅执行一次全局状态的初始化
void initXPUStreamsOnce() {
  // 使用C++11标准库的call_once确保全局流状态只被初始化一次
  c10::call_once(init_flag, initGlobalStreamState);

  // 如果当前流已经初始化，则直接返回
  if (current_streams) {
    return;
  }

  // 初始化当前流（线程局部变量）为“正常优先级”队列池中最后一个队列
  // 注意：队列池尚未初始化，具体初始化将在initDeviceStreamState函数中完成
  current_streams = std::make_unique<StreamId[]>(num_gpus);
  for (const auto i : c10::irange(num_gpus)) {
    // 将当前流分配给池中最后一个流，这在某些场景下有利，
    // 特别是当用户将工作负载初始化为使用当前流（最后一个流）进行计算，
    // 并利用池中的流（第一个流）进行通信时，允许不同流在计算和通信中重叠。
    current_streams[i] =
        makeStreamId(StreamIdType::NORMAL, kStreamsPerPool - 1);
  }
}

// 创建设备流的函数，确保仅执行一次设备队列池的初始化
inline void initDeviceStreamOnce(DeviceIndex device) {
  // 使用C++11标准库的call_once确保设备流只被初始化一次
  c10::call_once(device_flags[device], initDeviceStreamState, device);
}

// 检查设备索引是否在有效范围内
inline void check_device(DeviceIndex device) {
  // 使用TORCH_CHECK确保设备索引在有效范围内
  TORCH_CHECK(
      device >= 0 && device < num_gpus,
      "device is out of range, device is ",
      static_cast<int16_t>(device),
      ", total number of device is ",
      static_cast<int16_t>(num_gpus),
      ".");
}

// 获取原子计数器的索引
uint32_t get_idx(std::atomic<uint32_t>& counter) {
  auto raw_idx = counter++;
  return raw_idx % kStreamsPerPool;
}

// 根据设备索引和流ID创建XPU流对象
XPUStream XPUStreamForId(DeviceIndex device_index, StreamId stream_id) {
  return XPUStream(
      XPUStream::UNCHECKED,
      Stream(
          Stream::UNSAFE,
          c10::Device(DeviceType::XPU, device_index),
          stream_id));
}
// 返回当前流的优先级，优先级与流的类型相关，优先级数值是其类型的负数形式。
int XPUStream::priority() const {
  // 获取流的唯一标识符
  StreamId stream_id = stream_.id();
  // 根据流的唯一标识符确定其类型
  StreamIdType st = streamIdType(stream_id);
  // 返回该流类型的负数形式作为优先级
  return -static_cast<int>(st);
}

// 查看注释 [StreamId assignment]
sycl::queue& XPUStream::queue() const {
  // 获取流所在设备的索引
  DeviceIndex device_index = stream_.device_index();
  // 获取流的唯一标识符
  StreamId stream_id = stream_.id();
  // 根据流的唯一标识符确定其类型
  StreamIdType st = streamIdType(stream_id);
  // 获取流的索引
  StreamIdIndex si = streamIdIndex(stream_id);

  // 根据流的类型进行不同的处理
  switch (st) {
    case StreamIdType::NORMAL:
    case StreamIdType::HIGH:
      // 返回指向对应设备、类型和索引的流队列的引用
      return *streams[device_index][static_cast<uint8_t>(st)][si];
    default:
      // 如果流类型未知，抛出错误消息
      TORCH_CHECK(
          false,
          "Unrecognized stream ",
          stream_,
          " (I didn't recognize the stream type, ",
          st,
          ").",
          " Did you manufacture the StreamId yourself?  Don't do that;");
  }
}

// 从请求的池中返回一个流
// 注意: 如果需要，流池将在首次调用此函数时进行初始化
XPUStream getStreamFromPool(const int priority, DeviceIndex device) {
  // 初始化流池（仅初始化一次）
  initXPUStreamsOnce();
  // 如果设备为-1，则使用当前设备
  if (device == -1) {
    device = c10::xpu::current_device();
  }
  // 检查设备的有效性
  check_device(device);
  // 检查优先级是否合法
  TORCH_CHECK(
      priority <= 0,
      "Expected XPU stream priority to be less than or equal to 0, got ",
      priority);
  // 初始化设备流池（仅初始化一次）
  initDeviceStreamOnce(device);
  // 计算优先级索引
  auto priority_idx =
      std::min(-priority, max_compile_time_stream_priorities - 1);
  // 获取优先级计数器的索引
  const auto idx = get_idx(priority_counters[device][priority_idx]);
  // 将优先级索引转换为流的类型
  auto id_type = static_cast<StreamIdType>(priority_idx);
  // 根据类型和索引创建流对象并返回
  return XPUStreamForId(device, makeStreamId(id_type, idx));
}

// 从池中获取流对象，如果isHighPriority为true，则获取优先级最高的流
XPUStream getStreamFromPool(const bool isHighPriority, DeviceIndex device) {
  // 初始化流池（仅初始化一次）
  initXPUStreamsOnce();
  // 如果isHighPriority为true，则设定最高优先级的流
  int priority = isHighPriority ? -max_compile_time_stream_priorities + 1 : 0;
  // 调用获取流的函数
  return getStreamFromPool(priority, device);
}

// 返回当前设备的当前流对象
// 注意: 如果需要，流池将在首次调用此函数时进行初始化
XPUStream getCurrentXPUStream(DeviceIndex device) {
  // 初始化流池（仅初始化一次）
  initXPUStreamsOnce();
  // 如果设备为-1，则使用当前设备
  if (device == -1) {
    device = c10::xpu::current_device();
  }
  // 检查设备的有效性
  check_device(device);
  // 初始化设备流池（仅初始化一次）
  initDeviceStreamOnce(device);
  // 返回当前设备的当前流对象
  return XPUStreamForId(device, current_streams[device]);
}

// 设定当前设备的当前流对象
// 注意: 如果需要，流池将在首次调用此函数时进行初始化
void setCurrentXPUStream(XPUStream stream) {
  // 初始化流池（仅初始化一次）
  initXPUStreamsOnce();
  // 设定当前设备的当前流对象
  current_streams[stream.device_index()] = stream.id();
}

// 重载运算符 << ，将流对象的包装写入流中
std::ostream& operator<<(std::ostream& stream, const XPUStream& s) {
  return stream << s.unwrap();
}
/*
 * Note [Synchronize Streams on Device]
 *
 * There are two stream pools per device to manage our reserved SYCL queues.
 * When syncStreamsOnDevice is called, all reserved SYCL queues in the pools of
 * the specified device will be blocked, and wait for their synchronizations. We
 * realize the semantics via a loop through the stream pools of the specified
 * device and make each command queue synchronization sequentially.
 *
 * There is a semantic gap with device synchronization because only the SYCL
 * queues we have reserved (in our pools) will be synchronized, rather than
 * synchronizing all SYCL queues on the specified device.
 */

// Note: The stream pools will be initialized if needed, at the first invocation
// to this function.
void syncStreamsOnDevice(DeviceIndex device) {
  // 初始化流池，如果需要的话，将会在首次调用此函数时进行
  initXPUStreamsOnce();

  // 如果 device 为 -1，则使用当前的 XPU 设备
  if (device == -1) {
    device = c10::xpu::current_device();
  }
  
  // 检查设备的有效性
  check_device(device);
  
  // 初始化指定设备的流池（仅进行一次）
  initDeviceStreamOnce(device);

  // 对于每个设备，我们有 max_compile_time_stream_priorities 个优先级的流池
  for (const auto p : c10::irange(max_compile_time_stream_priorities)) {
    // 每个优先级有 kStreamsPerPool（32）个保留队列
    for (const auto i : c10::irange(kStreamsPerPool)) {
      // 等待每个队列的同步
      streams[device][p][i]->wait();
    }
  }

  // 获取 GPU 追踪的解释器实例，如果存在，则调用追踪 GPU 设备同步函数
  const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
  if (C10_UNLIKELY(interp)) {
    (*interp)->trace_gpu_device_synchronization(c10::kXPU);
  }
}
```