# `.\pytorch\c10\core\impl\InlineEvent.h`

```py
#pragma once
// 只允许本头文件被包含一次

#include <c10/core/DeviceType.h>
// 引入设备类型定义

#include <c10/core/Stream.h>
// 引入流定义

#include <c10/core/impl/DeviceGuardImplInterface.h>
// 引入设备保护接口实现

#include <c10/util/Exception.h>
// 引入异常处理工具

namespace c10::impl {

template <typename T>
// 使用模板定义 InlineEvent 结构

struct InlineEvent final {
  // 禁止默认构造函数
  InlineEvent() = delete;

  // 定义构造函数，初始化设备类型和事件标志
  InlineEvent(
      const DeviceType _device_type,
      const EventFlag _flag = EventFlag::PYTORCH_DEFAULT)
      : backend_{_device_type}, device_type_{_device_type}, flag_{_flag} {}

  // 复制构造函数和赋值运算符（禁用）
  InlineEvent(const InlineEvent&) = delete;
  InlineEvent& operator=(const InlineEvent&) = delete;

  // 移动构造函数和移动赋值运算符
  InlineEvent(InlineEvent&& other) noexcept
      : event_(other.event_),
        backend_(std::move(other.backend_)),
        device_type_(other.device_type_),
        device_index_(other.device_index_),
        flag_(other.flag_),
        was_marked_for_recording_(other.was_marked_for_recording_) {
    other.event_ = nullptr;
  }
  InlineEvent& operator=(InlineEvent&& other) noexcept {
    // 交换资源所有权
    swap(other);
    return *this;
  }

  // 交换函数，用于移动赋值操作
  void swap(InlineEvent& other) noexcept {
    std::swap(event_, other.event_);
    std::swap(backend_, other.backend_);
    std::swap(device_type_, other.device_type_);
    std::swap(device_index_, other.device_index_);
    std::swap(flag_, other.flag_);
    std::swap(was_marked_for_recording_, other.was_marked_for_recording_);
  }

  // 析构函数，释放事件资源
  ~InlineEvent() noexcept {
    if (event_)
      backend_.destroyEvent(event_, device_index_);
  }

  // 返回事件关联的设备类型
  DeviceType device_type() const noexcept {
    return device_type_;
  }

  // 返回事件关联的设备索引
  DeviceIndex device_index() const noexcept {
    return device_index_;
  }

  // 返回事件标志
  EventFlag flag() const noexcept {
    return flag_;
  }

  // 返回事件是否已经标记为记录
  bool was_marked_for_recording() const noexcept {
    return was_marked_for_recording_;
  }

  // 标记一次事件记录
  void recordOnce(const Stream& stream) {
    if (!was_marked_for_recording_)
      record(stream);
  }

  // 记录事件
  void record(const Stream& stream) {
    // 检查流的设备类型与事件关联的设备类型是否匹配
    TORCH_CHECK(
        stream.device_type() == device_type_,
        "Event device type ",
        DeviceTypeName(device_type_),
        " does not match recording stream's device type ",
        DeviceTypeName(stream.device_type()),
        ".");

    // 记录事件到流
    backend_.record(&event_, stream, device_index_, flag_);
    was_marked_for_recording_ = true;
    device_index_ = stream.device_index();
  }

  // 阻塞等待事件完成
  void block(const Stream& stream) const {
    if (!was_marked_for_recording_)
      return;

    // 检查流的设备类型与事件关联的设备类型是否匹配
    TORCH_CHECK(
        stream.device_type() == device_type_,
        "Event device type ",
        DeviceTypeName(device_type_),
        " does not match blocking stream's device type ",
        DeviceTypeName(stream.device_type()),
        ".");

    // 阻塞等待事件完成
    backend_.block(event_, stream);
  }

  // 查询事件是否完成
  bool query() const {
    if (!was_marked_for_recording_)
      return true;
    return backend_.queryEvent(event_);
  }

  // 返回事件标识符
  void* eventId() const {
    return event_;
  }

  // 计算事件间的时间差
  double elapsedTime(const InlineEvent& other) const {
    # 检查另一个事件对象是否被标记为录制，如果没有，抛出错误信息
    TORCH_CHECK(
        other.was_marked_for_recording(),
        "other was not marked for recording.");
    # 检查当前事件对象是否被标记为录制，如果没有，抛出错误信息
    TORCH_CHECK(
        was_marked_for_recording(), "self was not marked for recording.");
    # 检查另一个事件对象的设备类型是否与当前事件对象的设备类型相匹配，如果不匹配，抛出错误信息
    TORCH_CHECK(
        other.device_type() == device_type_,
        "Event device type ",
        DeviceTypeName(device_type_),
        " does not match other's device type ",
        DeviceTypeName(other.device_type()),
        ".");
    # 返回使用特定后端计算两个事件之间的时间差
    return backend_.elapsedTime(event_, other.event_, device_index_);
  }

  # 同步当前事件对象关联的事件
  void synchronize() const {
    # 如果当前事件对象没有被标记为录制，则直接返回
    if (!was_marked_for_recording_)
      return;
    # 使用后端方法同步当前事件对象关联的事件
    backend_.synchronizeEvent(event_);
  }

 private:
  # 事件对象指针，默认为 nullptr
  void* event_ = nullptr;
  # 后端处理对象，泛型类型 T
  T backend_;
  # 事件对象的设备类型
  DeviceType device_type_;
  # 事件对象的设备索引，默认为 -1
  DeviceIndex device_index_ = -1;
  # 事件标志，默认为 PYTORCH_DEFAULT
  EventFlag flag_ = EventFlag::PYTORCH_DEFAULT;
  # 当前事件对象是否被标记为录制，默认为 false
  bool was_marked_for_recording_ = false;
};

} // namespace c10::impl
```