# `.\pytorch\c10\util\Lazy.h`

```
#pragma once

#include <atomic>
#include <utility>

namespace c10 {

/**
 * 线程安全的乐观懒加载值：在并发首次访问时，工厂可能会被多个线程调用，但只有一个结果会被存储，
 * 并且其引用会返回给所有的调用者。
 *
 * 值是堆分配的；这优化了实际上值从未被计算的情况。
 */
template <class T>
class OptimisticLazy {
 public:
  OptimisticLazy() = default;
  
  // 拷贝构造函数：从另一个OptimisticLazy对象拷贝值
  OptimisticLazy(const OptimisticLazy& other) {
    if (T* value = other.value_.load(std::memory_order_acquire)) {
      value_ = new T(*value);
    }
  }
  
  // 移动构造函数：从另一个OptimisticLazy对象移动值（无锁操作）
  OptimisticLazy(OptimisticLazy&& other) noexcept
      : value_(other.value_.exchange(nullptr, std::memory_order_acq_rel)) {}
  
  // 析构函数：释放内存
  ~OptimisticLazy() {
    reset();
  }

  // 确保方法：确保值的存在或计算
  template <class Factory>
  T& ensure(Factory&& factory) {
    // 如果值已存在，直接返回
    if (T* value = value_.load(std::memory_order_acquire)) {
      return *value;
    }
    // 否则，使用工厂函数计算值并存储
    T* value = new T(factory());
    T* old = nullptr;
    // 使用CAS操作（比较并交换），确保只有一个线程可以设置值
    if (!value_.compare_exchange_strong(
            old, value, std::memory_order_release, std::memory_order_acquire)) {
      delete value;
      value = old;
    }
    return *value;
  }

  // 下面的方法不是线程安全的：不应与任何其他方法并发调用。

  // 拷贝赋值运算符：从另一个OptimisticLazy对象拷贝值
  OptimisticLazy& operator=(const OptimisticLazy& other) {
    *this = OptimisticLazy{other};
    return *this;
  }

  // 移动赋值运算符：从另一个OptimisticLazy对象移动值（无锁操作）
  OptimisticLazy& operator=(OptimisticLazy&& other) noexcept {
    if (this != &other) {
      reset();
      value_.store(
          other.value_.exchange(nullptr, std::memory_order_acquire),
          std::memory_order_release);
    }
    return *this;
  }

  // 重置方法：重置值并释放内存
  void reset() {
    if (T* old = value_.load(std::memory_order_relaxed)) {
      value_.store(nullptr, std::memory_order_relaxed);
      delete old;
    }
  }

 private:
  std::atomic<T*> value_{nullptr}; // 原子指针，用于存储值
};

/**
 * 延迟访问计算值的接口。
 */
template <class T>
class LazyValue {
 public:
  virtual ~LazyValue() = default;

  // 获取值的虚函数，必须被子类实现
  virtual const T& get() const = 0;
};

/**
 * 方便的线程安全LazyValue实现，使用乐观懒加载策略。
 */
template <class T>
class OptimisticLazyValue : public LazyValue<T> {
 public:
  const T& get() const override {
    return value_.ensure([this] { return compute(); });
  }

 private:
  // 计算方法，由子类实现
  virtual T compute() const = 0;

  mutable OptimisticLazy<T> value_; // 懒加载值
};

/**
 * 为不需要懒加载的情况提供的线程安全LazyValue实现。
 */
template <class T>
class PrecomputedLazyValue : public LazyValue<T> {
 public:
  // 构造函数：使用预先计算的值初始化
  PrecomputedLazyValue(T value) : value_(std::move(value)) {}

  // 获取值的方法：直接返回预先计算的值
  const T& get() const override {
    return value_;
  }

 private:
  T value_; // 预先计算的值
};

} // namespace c10
```