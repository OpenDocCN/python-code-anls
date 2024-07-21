# `.\pytorch\torch\csrc\lazy\backend\backend_data.h`

```
#pragma once

#include <torch/csrc/lazy/backend/backend_device.h>
#include <torch/csrc/lazy/core/shape.h>
#include <cstring>

namespace torch {
namespace lazy {

/**
 * Represents backend data used by Lazy execution framework.
 * Provides interfaces for accessing device information and shape.
 */
class TORCH_API BackendData {
 public:
  /**
   * Abstract base class for additional information associated
   * with BackendData objects.
   */
  struct Info {
    virtual ~Info() = default;  ///< Used by Lazy Graph Executor to tag info on BackendData objs
  };

  /**
   * Handle type representing data stored on a backend device.
   */
  using Handle = int64_t;

  /**
   * Constructor initializing BackendData with specified device and shape.
   */
  BackendData(BackendDevice device, Shape shape)
      : device_(std::move(device)), shape_(std::move(shape)) {}

  /**
   * Virtual destructor for BackendData.
   */
  virtual ~BackendData() = default;

  /**
   * Returns the backend device associated with this BackendData.
   */
  const BackendDevice& device() const {
    return device_;
  }

  /**
   * Returns the shape of the data stored by this BackendData.
   */
  const Shape& shape() const {
    return shape_;
  }

  /**
   * Returns the additional information associated with this BackendData.
   */
  Info* info() const {
    return info_.get();
  }

  /**
   * Sets the additional information associated with this BackendData.
   * Returns the previous info object.
   */
  std::shared_ptr<Info> SetInfo(std::shared_ptr<Info> info) {
    std::swap(info, info_);
    return info;
  }

  /**
   * Pure virtual function to retrieve the handle for the data stored by this BackendData.
   */
  virtual Handle GetHandle() = 0;

  /**
   * Pure virtual function to assign data from another BackendData object.
   */
  virtual void Assign(const BackendData& data) = 0;

  /**
   * Pure virtual function to check if this BackendData object has valid data.
   */
  virtual bool HasValue() const = 0;

 private:
  BackendDevice device_;             ///< Backend device associated with this BackendData
  Shape shape_;                      ///< Shape of the data stored by this BackendData
  std::shared_ptr<Info> info_;       ///< Additional information associated with this BackendData
};

/**
 * Shared pointer type for BackendData objects.
 */
using BackendDataPtr = std::shared_ptr<BackendData>;

} // namespace lazy
} // namespace torch
```