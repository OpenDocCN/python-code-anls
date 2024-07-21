# `.\pytorch\c10\cuda\CUDADeviceAssertion.h`

```
#pragma once

#include <c10/cuda/CUDAException.h>
#include <c10/macros/Macros.h>

namespace c10::cuda {

#ifdef TORCH_USE_CUDA_DSA
// 在设备上复制字符串从 `src` 到 `dst`
static __device__ void dstrcpy(char* dst, const char* src) {
  int i = 0;
  // 从源字符串复制到目标字符串，确保不超过 `C10_CUDA_DSA_MAX_STR_LEN-1` 的长度限制
  while (*src != '\0' && i++ < C10_CUDA_DSA_MAX_STR_LEN - 1) {
    *dst++ = *src++;
  }
  *dst = '\0';
}

// 向设备端添加新的断言失败信息
static __device__ void dsa_add_new_assertion_failure(
    DeviceAssertionsData* assertions_data,
    const char* assertion_msg,
    const char* filename,
    const char* function_name,
    const int line_number,
    const uint32_t caller,
    const dim3 block_id,
    const dim3 thread_id) {
  // 如果 `assertions_data` 是空指针，表示在运行时设备端断言检查被禁用了
  // 如果在编译时被禁用，这个函数将不会被调用
  if (!assertions_data) {
    return;
  }

  // 原子操作增加断言计数，以便其他线程可以同时失败
  // 增加计数意味着 CPU 可以在我们将失败信息写入缓冲区之前就观察到失败，并开始响应
  const auto nid = atomicAdd(&(assertions_data->assertion_count), 1);

  // 如果超过了断言缓冲区的容量限制，则静默地忽略任何其他的断言失败
  if (nid >= C10_CUDA_DSA_ASSERTION_COUNT) {
    // 此时已经耗尽断言缓冲区空间
    // 如果大量线程同时执行此操作，打印相关消息可能会导致垃圾信息
    return;
  }

  // 将断言失败的信息写入内存
  // 注意，这仅在 `assertion_count` 增加后才写入，这样就表明发生了问题
  auto& self = assertions_data->assertions[nid];
  dstrcpy(self.assertion_msg, assertion_msg);
  dstrcpy(self.filename, filename);
  dstrcpy(self.function_name, function_name);
  self.line_number = line_number;
  self.caller = caller;
  self.block_id[0] = block_id.x;
  self.block_id[1] = block_id.y;
  self.block_id[2] = block_id.z;
  self.thread_id[0] = thread_id.x;
  self.thread_id[1] = thread_id.y;
  self.thread_id[2] = thread_id.z;
}

// 模拟一个内核断言。断言不会停止内核的执行进程，
// 因此如果有断言失败，应该假设内核生成的所有内容都是无效的。
// 注意：这假设 `assertions_data` 和 `assertion_caller_id` 是内核的参数，因此是可访问的。
#define CUDA_KERNEL_ASSERT2(condition)                                   \
  do {                                                                   \
    `
        # 如果条件不满足，则执行以下操作
        if (C10_UNLIKELY(!(condition))) {                                    \
          /* 如果条件不满足，调用 c10::cuda::dsa_add_new_assertion_failure 函数， */
          /* 将断言失败信息记录到 assertions_data 中 */
          c10::cuda::dsa_add_new_assertion_failure(                          \
              assertions_data,                                               \
              C10_STRINGIZE(condition),                                      \
              __FILE__,                                                      \
              __FUNCTION__,                                                  \
              __LINE__,                                                      \
              assertion_caller_id,                                           \
              blockIdx,                                                      \
              threadIdx);                                                    \
          /* 在核函数失败后，提前退出核函数；否则继续执行，依赖主机检查 UVM 并确定是否出现问题 */
          return;                                                            \
        }                                                                    \
      } while (false)
#else
// 如果不是 CUDA 编译环境，定义一个宏 CUDA_KERNEL_ASSERT2(condition) 为 assert(condition)
#define CUDA_KERNEL_ASSERT2(condition) assert(condition)
#endif

} // namespace c10::cuda
```