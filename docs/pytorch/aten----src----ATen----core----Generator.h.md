# `.\pytorch\aten\src\ATen\core\Generator.h`

```
/**
 * Note [Generator]
 * ~~~~~~~~~~~~~~~~
 * A Pseudo Random Number Generator (PRNG) is an engine that uses an algorithm to
 * generate a seemingly random sequence of numbers, that may be later be used in creating
 * a random distribution. Such an engine almost always maintains a state and requires a
 * seed to start off the creation of random numbers. Often times, users have
 * found it beneficial to be able to explicitly create, retain, and destroy
 * PRNG states and also be able to have control over the seed value.
 *
 * A Generator in ATen gives users the ability to read, write and modify a PRNG engine.
 * For instance, it does so by letting users seed a PRNG engine, fork the state of the
 * engine, etc.
 *
 * By default, there is one generator per device, and a device's generator is
 * lazily created. A user can use the torch.Generator() api to create their own generator.
 */

/**
 * Note [Acquire lock when using random generators]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * Generator and its derived classes are NOT thread-safe. Please note that most of the
 * places where we have inserted locking for generators are historically based, and we
 * haven't actually checked that everything is truly thread safe (and it probably isn't).
 * Please use the public mutex_ when using any methods from these classes, except for the
 * read-only methods. You can learn about the usage by looking into the unittests
 * (aten/src/ATen/cpu_generator_test.cpp) and other places where we have used lock_guard.
 *
 * TODO: Look into changing the threading semantics of Generators in ATen (e.g., making
 * them non-thread safe and instead making the generator state splittable, to accommodate
 * forks into other threads).
 */

#pragma once

#include <cstdint>
#include <deque>
#include <mutex>
#include <utility>

#include <c10/util/Exception.h>
#include <c10/util/intrusive_ptr.h>
#include <c10/core/Device.h>
#include <c10/core/DispatchKeySet.h>

// For the record I don't think this is a correct pimpl idiom.
// Including Impl header in interface header defeats the purpose
// because you can't change Impl private members without forcing
// everything that included the interface to rebuild.
// Impl should be forward-declared in the interface header instead.
#include <c10/core/GeneratorImpl.h>

namespace at {

/**
 * @brief Represents a generator for generating pseudo-random numbers.
 * 
 * This class provides functionalities to interact with a PRNG engine,
 * such as seeding, forking, and accessing the underlying implementation.
 */
class Generator {
  public:
    Generator() = default;  // Default constructor

    /**
     * @brief Constructs a Generator with a given GeneratorImpl instance.
     * 
     * @param gen_impl A smart pointer to a GeneratorImpl instance.
     * @throw std::runtime_error if gen_impl is nullptr.
     */
    explicit Generator(c10::intrusive_ptr<c10::GeneratorImpl> gen_impl)
     : impl_(std::move(gen_impl)) {
        if (impl_.get() == nullptr) {
            throw std::runtime_error("GeneratorImpl with nullptr is not supported");
        }
    }

    /**
     * @brief Equality operator for comparing two Generator instances.
     * 
     * @param rhs Another Generator instance to compare.
     * @return true if both Generator instances are equal, false otherwise.
     */
    bool operator==(const Generator& rhs) const {
        return this->impl_ == rhs.impl_;
    }

    /**
     * @brief Inequality operator for comparing two Generator instances.
     * 
     * @param rhs Another Generator instance to compare.
     * @return true if both Generator instances are not equal, false otherwise.
     */
    bool operator!=(const Generator& rhs) const {
        return !((*this) == rhs);
    }

    /**
     * @brief Checks if the Generator instance is defined (non-null).
     * 
     * @return true if the Generator instance is defined, false otherwise.
     */
    bool defined() const {
        return static_cast<bool>(impl_);
    }

    /**
     * @brief Retrieves the raw pointer to the GeneratorImpl.
     * 
     * @return A raw pointer to the GeneratorImpl.
     */
    c10::GeneratorImpl* unsafeGetGeneratorImpl() const {
        return impl_.get();
    }

    /**
     * @brief Releases ownership of the GeneratorImpl.
     * 
     * @return A raw pointer to the released GeneratorImpl.
     */
    c10::GeneratorImpl* unsafeReleaseGeneratorImpl() {
        return impl_.release();
    }

  private:
    c10::intrusive_ptr<c10::GeneratorImpl> impl_;  ///< Pointer to the GeneratorImpl instance.
};

} // namespace at
    // 调用 impl_ 对象的 release() 方法并返回结果
    return impl_.release();
  }

  // 返回 impl_ 对象的常量引用作为 intrusive_ptr<c10::GeneratorImpl> 类型
  const c10::intrusive_ptr<c10::GeneratorImpl>& getIntrusivePtr() const {
    return impl_;
  }

  // 设置生成器的当前种子值为给定的 seed 参数
  void set_current_seed(uint64_t seed) { impl_->set_current_seed(seed); }

  // 设置生成器状态的偏移量为指定的 offset 参数。目前仅支持基于 Philox 的生成器，例如 CUDA 和 MPS。
  void set_offset(uint64_t offset) { impl_->set_offset(offset); }

  // 返回生成器状态的偏移量。目前仅支持基于 Philox 的生成器，例如 CUDA 和 MPS。
  uint64_t get_offset() const { return impl_->get_offset(); }

  // 返回当前生成器的种子值
  uint64_t current_seed() const { return impl_->current_seed(); }

  // 返回生成器的种子值
  uint64_t seed() { return impl_->seed(); }

  // 设置状态的实现方法，不内联以防止 `ATen/core/Generator.h` 和 `ATen/core/Tensor.h` 之间的循环引用
  void set_state(const at::Tensor& new_state);

  // 返回当前状态的 Tensor 对象
  at::Tensor get_state() const;

  // 设置状态的图安全版本
  void graphsafe_set_state(const Generator& new_state);

  // 返回图安全状态
  Generator graphsafe_get_state() const;

  // 返回 impl_ 对象的互斥锁引用
  std::mutex& mutex() {
    return impl_->mutex_;
  }

  // 返回 impl_ 对象的调度键集合
  DispatchKeySet key_set() const {
    return impl_->key_set();
  }

  // 返回 impl_ 对象的设备
  Device device() const { return impl_->device(); }

  // 设置 Python 对象的指针
  inline void set_pyobj(PyObject* pyobj) const noexcept {
    impl_->set_pyobj(pyobj);
  }

  // 返回 Python 对象的指针
  inline PyObject* pyobj() const noexcept {
    return impl_->pyobj();
  }

  // 返回 T 类型的指针，类型转换为 T*
  template<typename T>
  T* get() const { return static_cast<T*>(impl_.get()); }

  // 克隆生成器对象，返回生成器的副本
  Generator clone() const {
    return Generator(impl_->clone());
  }
};

/**
 * Creates a generator object of a specific implementation using variadic arguments.
 * This function wraps the creation of the generator with the provided arguments.
 */
template<class Impl, class... Args>
Generator make_generator(Args&&... args) {
  // Create an instance of Impl using c10::make_intrusive and forward the arguments
  return Generator(c10::make_intrusive<Impl>(std::forward<Args>(args)...));
}

/**
 * Utility function to check and static cast an optional Generator pointer to a specific type.
 * Ensures the generator is defined and matches the expected device type.
 * Throws errors if expectations are not met.
 */
template <typename T>
inline T * check_generator(std::optional<Generator> gen) {
  // Check if generator is provided
  TORCH_CHECK(gen.has_value(), "Expected Generator but received nullopt");
  // Ensure the generator is defined
  TORCH_CHECK(gen->defined(), "Generator with undefined implementation is not allowed");
  // Check if the generator's device type matches the expected type T::device_type()
  TORCH_CHECK(T::device_type() == gen->device().type(), "Expected a '", T::device_type(), "' device type for generator but found '", gen->device().type(), "'");
  // Return the generator casted to type T
  return gen->get<T>();
}

/**
 * Utility function used in tensor implementations to get the appropriate generator.
 * If gen is provided and defined, it checks and casts gen to type T.
 * Otherwise, it falls back to casting default_gen to type T.
 */
template <typename T>
inline T* get_generator_or_default(const std::optional<Generator>& gen, const Generator& default_gen) {
  // If gen is provided and defined, use it; otherwise, fallback to default_gen
  return gen.has_value() && gen->defined() ? check_generator<T>(gen) : check_generator<T>(default_gen);
}

namespace detail {

/**
 * Helper function for validating the new random generator state tensor.
 * Checks that:
 * - The tensor layout is strided
 * - The device type is CPU
 * - The data type is Byte
 * - The tensor data is contiguous
 */
inline void check_rng_state(const c10::TensorImpl& new_state) {
  // Check tensor properties: layout, device type, and data type
  TORCH_CHECK_TYPE(
    new_state.layout() == kStrided && new_state.device().type() == kCPU && new_state.dtype() == kByte,
    "RNG state must be a torch.ByteTensor"
  );
  // Check if tensor data is contiguous
  TORCH_CHECK(new_state.is_contiguous(), "RNG state must be contiguous");
}

} // namespace detail

} // namespace at
```