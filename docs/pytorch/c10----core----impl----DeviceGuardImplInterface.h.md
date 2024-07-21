# `.\pytorch\c10\core\impl\DeviceGuardImplInterface.h`

```py
#pragma once

#include <c10/core/Device.h> // 包含设备相关的头文件
#include <c10/core/DeviceType.h> // 包含设备类型相关的头文件
#include <c10/core/Stream.h> // 包含流相关的头文件
#include <c10/util/Exception.h> // 包含异常处理相关的头文件

// Just for C10_ANONYMOUS_VARIABLE
#include <c10/util/Registry.h> // 包含注册相关的头文件

#include <atomic> // 包含原子操作相关的头文件

namespace c10 {

// Forward declaration
class DataPtr; // 前向声明类 DataPtr

/**
 * Note [Flags defining the behavior of events]
 *
 * PYTORCH_DEFAULT and BACKEND_DEFAULT are valid for all backends. The
 * BACKEND_DEFAULT is what a particular backend would select if no
 * flags were given. PYTORCH_DEFAULT is the PyTorch's framework default
 * choice for events on that backend, which may not be the same.
 *
 * The mapping of PYTORCH_DEFAULT and BACKEND_DEFAULT is done by each
 * backend implementation.
 */
enum class EventFlag {
  // Disable timing
  PYTORCH_DEFAULT, // PyTorch 框架默认事件标志，用于所有后端
  // Enable timing
  BACKEND_DEFAULT, // 后端默认事件标志，当没有其他标志时使用
  // FOR TESTING ONLY
  INVALID // 无效标志，仅用于测试目的
};

namespace impl {

/**
 * DeviceGuardImplInterface represents the virtual interface which provides
 * functionality to provide an RAII class for device and stream switching,
 * via DeviceGuard.  Every distinct device type, e.g., CUDA and HIP, is
 * expected to implement and register an implementation of this interface.
 * All classes which inherit from DeviceGuardImplInterface should be declared
 * 'final'.
 *
 * This class exists because we provide a unified interface for performing
 * device guards via DeviceGuard, but we cannot assume that we have actually
 * compiled against the, e.g., CUDA library, which actually implements
 * this guard functionality.  In this case, a dynamic dispatch is required
 * to cross the library boundary.
 *
 * If possible, you should directly use implementations of this interface;
 * those uses will be devirtualized.
 */
/**
 * Interface for managing device-specific operations.
 * Defines methods for manipulating devices and streams.
 */
struct C10_API DeviceGuardImplInterface {
  /**
   * Default constructor.
   */
  DeviceGuardImplInterface() = default;

  /**
   * Copy constructor.
   */
  DeviceGuardImplInterface(const DeviceGuardImplInterface&) = default;

  /**
   * Copy assignment operator.
   */
  DeviceGuardImplInterface& operator=(const DeviceGuardImplInterface&) =
      default;

  /**
   * Move constructor.
   */
  DeviceGuardImplInterface(DeviceGuardImplInterface&&) noexcept = default;

  /**
   * Move assignment operator.
   */
  DeviceGuardImplInterface& operator=(DeviceGuardImplInterface&&) noexcept =
      default;

  /**
   * Pure virtual method to return the type of device managed by this guard implementation.
   */
  virtual DeviceType type() const = 0;

  /**
   * Pure virtual method to exchange the current device with a new one.
   * Returns the previous device.
   */
  virtual Device exchangeDevice(Device) const = 0;

  /**
   * Pure virtual method to retrieve the current device.
   */
  virtual Device getDevice() const = 0;

  /**
   * Pure virtual method to set the current device to a new one.
   */
  virtual void setDevice(Device) const = 0;

  /**
   * Pure virtual method to set the current device to a new one without error checking.
   * Can be called safely from a destructor.
   */
  virtual void uncheckedSetDevice(Device) const noexcept = 0;

  /**
   * Pure virtual method to get the current stream for a given device.
   */
  virtual Stream getStream(Device) const noexcept = 0;

  /**
   * Virtual method to get the default stream for a given device.
   * Throws an error since the backend doesn't support acquiring a default stream.
   */
  virtual Stream getDefaultStream(Device) const {
    TORCH_CHECK(false, "Backend doesn't support acquiring a default stream.")
  }

  /**
   * Virtual method to get a stream from the global pool for a given device.
   * Throws an error since the backend doesn't support acquiring a stream from the pool.
   * @param isHighPriority Unused parameter (suppresses unused variable warning).
   */
  virtual Stream getStreamFromGlobalPool(Device, bool isHighPriority = false)
      const {
    (void)isHighPriority;
    TORCH_CHECK(false, "Backend doesn't support acquiring a stream from pool.")
  }

  /**
   * Virtual method to return a new stream for a given device and priority.
   * Throws an error since this method should be implemented by the specific backend.
   * @param priority Unused parameter.
   */
  virtual Stream getNewStream(Device, int priority = 0) const {
    (void)priority;
    // Specific backends should implement this method
    // for creating a new stream with appropriate handling.
    TORCH_CHECK(false, "Backend-specific implementation required for creating a new stream.")
  }
};
  // 使用 TORCH_CHECK 断言检查条件 false，如果条件为真，则输出错误消息并终止程序
  TORCH_CHECK(false, "Backend doesn't support create a new Stream.")
}

/**
 * 设置一个流作为其设备的线程本地当前流。
 * 返回该设备的先前流。不需要设置当前设备以匹配此流的设备。
 */
virtual Stream exchangeStream(Stream) const noexcept = 0;

/**
 * 销毁给定的事件。
 */
virtual void destroyEvent(void* /*event*/, const DeviceIndex /*device_index*/)
    const noexcept {}

/**
 * 增加事件的版本号，并将带有此版本号的作业排队到流的工作队列中。
 * 当流处理该作业时，通知所有等待或被此版本事件阻塞的流继续，并将该版本标记为已记录。
 */
virtual void record(
    void** /*event*/,
    const Stream& /*stream*/,
    const DeviceIndex /*device_index*/,
    const c10::EventFlag /*flag*/) const {
  // 使用 TORCH_CHECK 断言检查条件 false，如果条件为真，则输出错误消息并终止程序
  TORCH_CHECK(false, "Backend doesn't support events.");
}

/**
 * 如果事件尚未计划记录，则不执行任何操作。
 * 如果之前已将事件排队记录，则在流的工作队列中插入一个等待此版本事件的命令。
 * 当流达到此命令时，它将停止处理其他命令，直到该事件的该版本被标记为已记录。
 */
virtual void block(void* /*event*/, const Stream& /*stream*/) const {
  // 使用 TORCH_CHECK 断言检查条件 false，如果条件为真，则输出错误消息并终止程序
  TORCH_CHECK(false, "Backend doesn't support events.");
}

/**
 * 如果（且仅如果）：
 * （1）事件从未计划记录过；
 * （2）当前版本已标记为已记录。
 * 则返回 true，否则返回 false。
 */
virtual bool queryEvent(void* /*event*/) const {
  // 使用 TORCH_CHECK 断言检查条件 false，如果条件为真，则输出错误消息并终止程序
  TORCH_CHECK(false, "Backend doesn't support events.");
}

/**
 * 获取设备的数量。警告：此方法必须不引发异常。
 * 如果存在问题，例如驱动程序错误，则应报告可用设备数为零。
 */
virtual DeviceIndex deviceCount() const noexcept = 0;

/**
 * 返回 true 如果流上先前排队的所有异步执行的工作已在设备上完成运行。
 */
virtual bool queryStream(const Stream& /*stream*/) const {
  // 使用 TORCH_CHECK 断言检查条件 false，如果条件为真，则输出错误消息并终止程序
  TORCH_CHECK(false, "Backend doesn't support querying streams.");
}

/**
 * 等待（通过阻塞调用线程）直到流上先前排队的所有工作在设备上完成运行。
 */
virtual void synchronizeStream(const Stream& /*stream*/) const {
  // 使用 TORCH_CHECK 断言检查条件 false，如果条件为真，则输出错误消息并终止程序
  TORCH_CHECK(false, "Backend doesn't support synchronizing streams.");
}

/**
 * 等待（通过阻塞调用线程）直到事件上先前记录的所有工作在设备上完成运行。
 */
virtual void synchronizeEvent(void* /*event*/) const {
    # 使用 TORCH_CHECK 断言，如果条件为 false，则输出指定的错误消息
    TORCH_CHECK(false, "Backend doesn't support synchronizing events.");
  }

  /**
   * Ensure the caching allocator (if any) is aware that the given DataPtr is
   * being used on the given stream, and that it should thus avoid recycling the
   * DataPtr until all work on that stream is done.
   */
  # 在给定流上记录 DataPtr 的使用，以通知缓存分配器避免在流上的工作完成之前回收 DataPtr
  virtual void recordDataPtrOnStream(const c10::DataPtr&, const Stream&) const {
  }

  /**
   * Fetch the elapsed time between two recorded events.
   */
  # 获取两个记录事件之间的经过时间
  virtual double elapsedTime(
      void* /*event1*/,
      void* /*event2*/,
      const DeviceIndex /*device_index*/) const {
    # 使用 TORCH_CHECK 断言，如果条件为 false，则输出指定的错误消息
    TORCH_CHECK(false, "Backend doesn't support elapsedTime.");
  }

  /**
   * Intended use of this class is to leak the DeviceGuardImpl at program end.
   * So you better not call the destructor, buster!
   */
  # 此类的预期用途是在程序结束时泄露 DeviceGuardImpl。
  # 因此，最好不要调用析构函数，小子！
  virtual ~DeviceGuardImplInterface() = default;
// };

// 一个无操作的设备保护实现，不执行任何有趣的操作。对于实际上没有设备索引概念的设备非常有用，主要的例子是 CPU 和 Meta。
template <DeviceType D>
struct NoOpDeviceGuardImpl final : public DeviceGuardImplInterface {
  NoOpDeviceGuardImpl() = default;
  
  // 返回设备类型
  DeviceType type() const override {
    return D;
  }
  
  // 交换设备并返回一个无操作设备
  Device exchangeDevice(Device) const override {
    return Device(D, -1); // no-op
  }
  
  // 返回一个无操作设备
  Device getDevice() const override {
    return Device(D, -1);
  }
  
  // 不执行任何操作
  void setDevice(Device) const override {
    // no-op
  }
  
  // 不执行任何操作
  void uncheckedSetDevice(Device) const noexcept override {
    // no-op
  }
  
  // 返回一个默认流的无操作流
  Stream getStream(Device) const noexcept override {
    // no-op
    return Stream(Stream::DEFAULT, Device(D, -1));
  }
  
  // 返回一个默认流的无操作流
  Stream getNewStream(Device, int priority = 0) const override {
    // no-op
    (void)priority; // 防止未使用的变量警告
    return Stream(Stream::DEFAULT, Device(D, -1));
  }
  
  // 不设置当前设备，返回一个默认流的无操作流
  Stream exchangeStream(Stream) const noexcept override {
    // no-op
    return Stream(Stream::DEFAULT, Device(D, -1));
  }
  
  // 返回设备数量为1
  DeviceIndex deviceCount() const noexcept override {
    return 1;
  }
  
  // Event 相关函数，抛出错误，指示后端不支持事件
  void record(
      void** /*event*/,
      const Stream& /*stream*/,
      const DeviceIndex /*device_index*/,
      const EventFlag /*flag*/) const override {
    TORCH_CHECK(false, D, " backend doesn't support events.");
  }
  
  // Event 相关函数，抛出错误，指示后端不支持事件
  void block(void* /*event*/, const Stream& /*stream*/) const override {
    TORCH_CHECK(false, D, " backend doesn't support events.")
  }
  
  // Event 相关函数，抛出错误，指示后端不支持事件
  bool queryEvent(void* /*event*/) const override {
    TORCH_CHECK(false, D, " backend doesn't support events.")
  }
  
  // Event 相关函数，不执行任何操作
  void destroyEvent(void* /*event*/, const DeviceIndex /*device_index*/) const noexcept override {
  }

  // Stream 相关函数，总是返回 true
  bool queryStream(const Stream& /*stream*/) const override {
    return true;
  }
  
  // Stream 相关函数，不等待任何内容
  void synchronizeStream(const Stream& /*stream*/) const override {
    // Don't wait for anything.
  }
};

// 注册表是非拥有的。每个存储的指针都是 std::atomic，因此在所有注册表调用的交错中，该结构是无竞争的。
// 这在 X86 上的读取上并不会带来额外开销。（一个未同步的实现可能也可以，但我不想证明我们在某些注册发生时从 device_guard_impl_registry 读取时永不读取。颤抖。）
//
// 我希望这个注册表在程序销毁时也有效（例如，如果某人在析构函数中使用 DeviceGuard 来清理 CUDA API 中的一些内容）。由于没有直接访问用于强制初始化顺序的底层拥有对象（不像 Meyer 单例中那样），这意味着在将对象放入注册表时必须 *泄漏* 对象。这通过删除 DeviceGuardImplInterface 的析构函数来实现。
// NOLINTNEXTLINE(*c-arrays*)
extern C10_API std::atomic<const DeviceGuardImplInterface*>
    // 使用 static_cast 将 DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES 转换为 size_t 类型，
    // 并作为 device_guard_impl_registry 的索引，获取对应的元素。
    device_guard_impl_registry[static_cast<size_t>(
        DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES)];
// 定义设备保护实现注册器类，用于注册不同设备类型的设备保护实现对象
class C10_API DeviceGuardImplRegistrar {
 public:
  // 构造函数，注册特定设备类型的设备保护实现对象
  DeviceGuardImplRegistrar(DeviceType, const DeviceGuardImplInterface*);
};

// 宏定义，用于注册设备保护实现对象到设备保护实现注册表
#define C10_REGISTER_GUARD_IMPL(DevType, DeviceGuardImpl)              \
  static ::c10::impl::DeviceGuardImplRegistrar C10_ANONYMOUS_VARIABLE( \
      g_##DeviceType)(::c10::DeviceType::DevType, new DeviceGuardImpl());

// 获取特定设备类型的设备保护实现对象的函数
inline const DeviceGuardImplInterface* getDeviceGuardImpl(DeviceType type) {
  // 检查设备类型的大小是否为1字节
  static_assert(sizeof(DeviceType) == 1, "DeviceType is not 8-bit");
  // 使用掩码操作获取设备类型对应的设备保护实现对象指针
  auto p = device_guard_impl_registry[static_cast<size_t>(type) & 0xFF].load();

  // 如果指针为空，抛出错误信息，指明 PyTorch 未链接支持特定设备类型的设备保护
  TORCH_CHECK(p, "PyTorch is not linked with support for ", type, " devices");
  return p;
}

// 检查是否存在特定设备类型的设备保护实现对象的函数
inline bool hasDeviceGuardImpl(DeviceType type) {
  return device_guard_impl_registry[static_cast<size_t>(type)].load();
}
```