# `.\pytorch\c10\util\Synchronized.h`

```
#pragma`
/**
 * @file
 * 
 * This header defines a **Synchronized** class template for safe multi-threaded data access.
 * It is inspired by folly::Synchronized<T> and provides basic functionality similar to it.
 * 
 * @namespace c10
 */

#pragma once

#include <mutex> // Include the mutex header for synchronization

namespace c10 {

/**
 * @brief A simple synchronization class for error-free multi-threaded data access.
 *        Inspired by folly/docs/Synchronized.md.
 * 
 * @tparam T Type of the data to be synchronized.
 */
template <typename T>
class Synchronized final {
  mutable std::mutex mutex_; // Mutex for thread synchronization
  T data_; // The synchronized data

public:
  /**
   * @brief Default constructor.
   */
  Synchronized() = default;

  /**
   * @brief Constructor initializing with data.
   * 
   * @param data The initial data to be synchronized.
   */
  Synchronized(T const& data) : data_(data) {}

  /**
   * @brief Constructor initializing with moveable data.
   * 
   * @param data The initial data to be synchronized (move semantics).
   */
  Synchronized(T&& data) : data_(std::move(data)) {}

  // Disable copy and move operations to prevent misuse of the mutex
  Synchronized(Synchronized const&) = delete;
  Synchronized(Synchronized&&) = delete;
  Synchronized operator=(Synchronized const&) = delete;
  Synchronized operator=(Synchronized&&) = delete;

  /**
   * @brief Execute a callback function with exclusive lock on the synchronized data.
   * 
   * @tparam CB Type of the callback function.
   * @param cb Callback function that accepts T either by copy or reference.
   * @return Return type of the callback function.
   */
  template <typename CB>
  auto withLock(CB&& cb) {
    std::lock_guard<std::mutex> guard(this->mutex_); // Lock the mutex
    return std::forward<CB>(cb)(this->data_); // Execute the callback with synchronized data
  }

  /**
   * @brief Execute a callback function with shared lock on the synchronized data.
   *        (Const version for read-only operations)
   * 
   * @tparam CB Type of the callback function.
   * @param cb Callback function that accepts T by const reference.
   * @return Return type of the callback function.
   */
  template <typename CB>
  auto withLock(CB&& cb) const {
    std::lock_guard<std::mutex> guard(this->mutex_); // Lock the mutex
    return std::forward<CB>(cb)(this->data_); // Execute the callback with synchronized data
  }
};

} // end namespace c10
```