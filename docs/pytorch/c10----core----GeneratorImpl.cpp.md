# `.\pytorch\c10\core\GeneratorImpl.cpp`

```py
/**
 * Header files inclusion for C++ GeneratorImpl class.
 */
#include <c10/core/GeneratorImpl.h>
#include <random>

/**
 * Conditional inclusion for SGX environment.
 */
#if defined(__SGX_ENABLED__)
#include <sgx_trts.h>
#endif

/**
 * Non-Windows specific header inclusion for file operations.
 */
#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#else
#include <chrono>
#endif

/**
 * Namespace declaration for C10 library.
 */
namespace c10 {

/**
 * Implementation of the GeneratorImpl class constructor.
 * Initializes the device and key set for the generator.
 */
GeneratorImpl::GeneratorImpl(Device device_in, DispatchKeySet key_set)
    : device_{device_in}, key_set_(key_set) {}

/**
 * Clone method for GeneratorImpl.
 * Creates a deep copy of the generator instance.
 */
c10::intrusive_ptr<GeneratorImpl> GeneratorImpl::clone() const {
  auto res = this->clone_impl();
  c10::raw::intrusive_ptr::incref(res);
  c10::raw::weak_intrusive_ptr::incref(res);
  return c10::intrusive_ptr<GeneratorImpl>::reclaim(res);
}

/**
 * Method not implemented: graphsafe_set_state.
 * Raises an error as this operation is not supported.
 */
void GeneratorImpl::graphsafe_set_state(
    const c10::intrusive_ptr<c10::GeneratorImpl>& state) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "graphsafe_set_state is not supported in this Generator");
}

/**
 * Method not implemented: graphsafe_get_state.
 * Raises an error as this operation is not supported.
 */
c10::intrusive_ptr<c10::GeneratorImpl> GeneratorImpl::graphsafe_get_state()
    const {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "graphsafe_get_state is not supported in this Generator");
}

/**
 * Retrieves the device associated with the generator.
 * Returns the device enum value.
 */
Device GeneratorImpl::device() const {
  return device_;
}

/**
 * Namespace for internal details of C10 library.
 */
namespace detail {

/**
 * Reads a random 64-bit unsigned integer from /dev/urandom.
 * This method is used for non-Windows platforms.
 * Throws an error if /dev/urandom cannot be opened or read.
 */
#if !defined(_WIN32)
static uint64_t readURandomLong() {
  int randDev = open("/dev/urandom", O_RDONLY);
  TORCH_CHECK(randDev >= 0, "Unable to open /dev/urandom");
  uint64_t randValue{};
  ssize_t readBytes = read(randDev, &randValue, sizeof(randValue));
  close(randDev);
  TORCH_CHECK(
      readBytes >= (ssize_t)sizeof(randValue),
      "Unable to read from /dev/urandom");
  return randValue;
}
#endif // _WIN32

/**
 * Generates a non-deterministic random number.
 * Depending on the platform and device, retrieves randomness
 * from /dev/urandom, current time, or uses platform-specific APIs.
 * Adjusts behavior for CUDA and Intel SGX environments.
 * TODO: Review and potentially improve the method for better consistency
 *       across different platforms and devices.
 */
uint64_t getNonDeterministicRandom(bool is_cuda) {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  uint64_t s;
  if (!is_cuda) {
#ifdef _WIN32
    // Use current high resolution time for Windows platform
    s = (uint64_t)std::chrono::high_resolution_clock::now()
            .time_since_epoch()
            .count();
#elif defined(__SGX_ENABLED__)
    // Use SGX-specific random number generation for Intel SGX platform
    // as /dev/urandom usage is prohibited
    // (SGX-specific implementation detail)
    s = sgx_read_rand(reinterpret_cast<unsigned char*>(&s), sizeof(s));
#else
    // For non-Windows non-SGX platforms, use /dev/urandom
    s = readURandomLong();
#endif
    return s;
}

} // namespace detail
} // namespace c10
    # 使用 TORCH_CHECK 宏来检查随机数生成是否成功，并在失败时输出错误消息
    TORCH_CHECK(
        sgx_read_rand(reinterpret_cast<uint8_t*>(&s), sizeof(s)) == SGX_SUCCESS,
        "Could not generate random number with sgx_read_rand.");
#else
    s = readURandomLong();
#endif
```  

// 如果条件不满足（即不处于某种特定环境），调用 readURandomLong() 函数生成一个随机长整型数赋值给 s



  } else {
    std::random_device rd;
    // limit to 53 bits to ensure unique representation in double
    // 使用 std::random_device 创建一个随机数生成设备对象 rd
    // 限制生成的随机数在 53 位内，以确保在转换为双精度浮点数时有唯一的表示
    s = ((((uint64_t)rd()) << 32) + rd()) & 0x1FFFFFFFFFFFFF;
    // 将两个随机数组合成一个 64 位整数，并按位与操作确保其范围在有效的 53 位内
  }



  return s;
}
```py  

// 返回生成的随机数 s



} // namespace detail
} // namespace c10
```  

// 结束命名空间 detail
// 结束命名空间 c10
```