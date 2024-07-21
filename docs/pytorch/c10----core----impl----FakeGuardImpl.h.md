# `.\pytorch\c10\core\impl\FakeGuardImpl.h`

```
#pragma once

#include <c10/core/impl/DeviceGuardImplInterface.h>

#include <array>

namespace c10::impl {

// FakeGuardImpl is hardcoded to have eight devices.  Not for
// any good reason, just to simplify code.
constexpr DeviceIndex kFakeGuardImplMaxDevices = 8;

/**
 * A fake implementation of DeviceGuardImplInterface suitable for testing.
 * The current device is modeled as a mutable field in the guard implementation
 * class.  See DeviceGuard_test.cpp for an example use.
 */
template <DeviceType T>
struct FakeGuardImpl final : public DeviceGuardImplInterface {
  static constexpr DeviceType static_type = T;
  
  // Constructor accepting DeviceType parameter, not used at runtime
  FakeGuardImpl(DeviceType) {}

  // Default constructor
  FakeGuardImpl() = default;

  // Returns the static device type associated with this instance
  DeviceType type() const override {
    return T;
  }

  // Exchange the current device with a new one, returning the old device
  Device exchangeDevice(Device d) const override {
    AT_ASSERT(d.type() == type());  // Assertion: Ensure device types match
    AT_ASSERT(d.index() < kFakeGuardImplMaxDevices);  // Assertion: Ensure device index is within bounds
    Device old_device = getDevice();
    if (old_device.index() != d.index()) {
      current_device_ = d.index();
    }
    return old_device;
  }

  // Retrieve the current device
  Device getDevice() const override {
    return Device(type(), current_device_);
  }

  // Set the current device to a new device
  void setDevice(Device d) const override {
    AT_ASSERT(d.type() == type());  // Assertion: Ensure device types match
    AT_ASSERT(d.index() >= 0);      // Assertion: Ensure device index is non-negative
    AT_ASSERT(d.index() < kFakeGuardImplMaxDevices);  // Assertion: Ensure device index is within bounds
    current_device_ = d.index();
  }

  // Set the current device without assertions (unchecked)
  void uncheckedSetDevice(Device d) const noexcept override {
    current_device_ = d.index();
  }

  // Get the stream associated with a device (not used safely)
  Stream getStream(Device d) const noexcept override {
    return Stream(Stream::UNSAFE, d, current_streams_[d.index()]);
  }

  // Exchange the current stream with a new stream, returning the old stream
  Stream exchangeStream(Stream s) const noexcept override {
    auto old_id = current_streams_[s.device_index()];
    current_streams_[s.device_index()] = s.id();
    return Stream(Stream::UNSAFE, s.device(), old_id);
  }

  // Return the number of devices this guard can manage
  DeviceIndex deviceCount() const noexcept override {
    return kFakeGuardImplMaxDevices;
  }

  // Event-related functions (no actual implementation details provided)

  // Record an event associated with a stream and device index
  void record(
      void** event,
      const Stream& stream,
      const DeviceIndex device_index,
      const EventFlag flag) const override {}

  // Block until an event associated with a stream completes
  void block(void* event, const Stream& stream) const override {}

  // Query whether an event has completed
  bool queryEvent(void* event) const override {
    return true;
  }

  // Destroy an event associated with a device index
  void destroyEvent(void* event, const DeviceIndex device_index) const noexcept override {}

  // Convenience methods for testing

  // Get the current device index (static method)
  static DeviceIndex getDeviceIndex() {
    return current_device_;
  }

  // Set the current device index (static method)
  static void setDeviceIndex(DeviceIndex i) {
    AT_ASSERT(i >= 0);  // Assertion: Ensure device index is non-negative
    AT_ASSERT(i < kFakeGuardImplMaxDevices);  // Assertion: Ensure device index is within bounds
    current_device_ = i;
  }

  // Get the current stream ID associated with a device index (static method)
  static StreamId getCurrentStreamIdFor(DeviceIndex i) {
    return current_streams_.at(i);
  }

  // Reset all current streams to zero (static method)
  static void resetStreams() {
    current_streams_.fill(0);
  }

 private:
  thread_local static DeviceIndex current_device_;  // Thread-local variable for current device index
  thread_local static std::array<StreamId, kFakeGuardImplMaxDevices> current_streams_;  // Thread-local array of stream IDs
};

template <DeviceType T>
thread_local DeviceIndex FakeGuardImpl<T>::current_device_ = 0;  // Initialize current_device_ to 0

template <DeviceType T>
thread_local std::array<StreamId, FakeGuardImpl<T>::kFakeGuardImplMaxDevices>
    FakeGuardImpl<T>::current_streams_;  // Initialize current_streams_ with zeros

} // namespace c10::impl
# 定义一个静态的线程局部存储数组，用于保存当前流的 ID，数组长度为 kFakeGuardImplMaxDevices
thread_local std::array<StreamId, kFakeGuardImplMaxDevices>
    FakeGuardImpl<T>::current_streams_ = {0, 0, 0, 0, 0, 0, 0, 0};
# 结束命名空间 c10::impl
} // namespace c10::impl
```