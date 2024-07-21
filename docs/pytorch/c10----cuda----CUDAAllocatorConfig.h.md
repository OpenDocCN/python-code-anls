# `.\pytorch\c10\cuda\CUDAAllocatorConfig.h`

```py
#pragma once
// 预处理指令：保证头文件只被包含一次

#include <c10/cuda/CUDAMacros.h>
#include <c10/util/Exception.h>

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>

namespace c10::cuda::CUDACachingAllocator {

// CUDAAllocatorConfig 类，用于解析环境配置
class C10_CUDA_API CUDAAllocatorConfig {
 public:
  // 返回最大分割大小
  static size_t max_split_size() {
    return instance().m_max_split_size;
  }

  // 返回垃圾回收阈值
  static double garbage_collection_threshold() {
    return instance().m_garbage_collection_threshold;
  }

  // 返回是否支持可扩展段
  static bool expandable_segments() {
#ifndef PYTORCH_C10_DRIVER_API_SUPPORTED
    if (instance().m_expandable_segments) {
      TORCH_WARN_ONCE("expandable_segments not supported on this platform")
    }
    return false;
#else
    return instance().m_expandable_segments;
#endif
  }

  // 返回是否在 cudamalloc 后释放锁
  static bool release_lock_on_cudamalloc() {
    return instance().m_release_lock_on_cudamalloc;
  }

  /** Pinned memory allocator settings */

  // 返回是否使用 CUDA 主机注册固定内存
  static bool pinned_use_cuda_host_register() {
    return instance().m_pinned_use_cuda_host_register;
  }

  // 返回固定内存注册的线程数
  static size_t pinned_num_register_threads() {
    return instance().m_pinned_num_register_threads;
  }

  // 返回固定内存最大注册线程数
  static size_t pinned_max_register_threads() {
    // 基于基准测试结果，使用 8 个线程可以获得更好的分配性能。然而在未来的系统中，
    // 可能需要更多的线程，这里限制为 128 个线程。
    return 128;
  }

  // 将分配大小向上舍入到最接近的 2 的幂划分
  // 更多描述见 roundup_power2_next_division 函数下面
  // 例如，如果我们希望在 2 的幂之间有 4 个划分，可以使用环境变量
  // PYTORCH_CUDA_ALLOC_CONF=roundup_power2_divisions:4
  static size_t roundup_power2_divisions(size_t size);

  // 返回向上舍入到最接近的 2 的幂划分的列表
  static std::vector<size_t> roundup_power2_divisions() {
    return instance().m_roundup_power2_divisions;
  }

  // 返回上一次分配器设置
  static std::string last_allocator_settings() {
    std::lock_guard<std::mutex> lock(
        instance().m_last_allocator_settings_mutex);
    return instance().m_last_allocator_settings;
  }

  // 返回单例实例
  static CUDAAllocatorConfig& instance() {
    static CUDAAllocatorConfig* s_instance = ([]() {
      auto inst = new CUDAAllocatorConfig();
      const char* env = getenv("PYTORCH_CUDA_ALLOC_CONF");
      inst->parseArgs(env);
      return inst;
    })();
    return *s_instance;
  }
    return *s_instance;
  }

  // 解析环境变量中的参数
  void parseArgs(const char* env);

 private:
  // 私有构造函数，用于单例模式
  CUDAAllocatorConfig();

  // 从环境变量字符串中解析参数并存储到config向量中
  static void lexArgs(const char* env, std::vector<std::string>& config);
  
  // 从config向量中消耗指定的令牌
  static void consumeToken(
      const std::vector<std::string>& config,
      size_t i,
      const char c);

  // 解析最大分割大小参数
  size_t parseMaxSplitSize(const std::vector<std::string>& config, size_t i);

  // 解析垃圾回收阈值参数
  size_t parseGarbageCollectionThreshold(
      const std::vector<std::string>& config,
      size_t i);

  // 解析向上舍入为2的幂的分割数参数
  size_t parseRoundUpPower2Divisions(
      const std::vector<std::string>& config,
      size_t i);

  // 解析分配器配置参数，包括是否使用cudaMallocAsync
  size_t parseAllocatorConfig(
      const std::vector<std::string>& config,
      size_t i,
      bool& used_cudaMallocAsync);

  // 解析固定内存使用CUDA主机注册参数
  size_t parsePinnedUseCudaHostRegister(
      const std::vector<std::string>& config,
      size_t i);

  // 解析固定内存数注册线程数参数
  size_t parsePinnedNumRegisterThreads(
      const std::vector<std::string>& config,
      size_t i);

  // 原子变量，存储最大分割大小
  std::atomic<size_t> m_max_split_size;

  // 存储向上舍入为2的幂的分割数的向量
  std::vector<size_t> m_roundup_power2_divisions;

  // 原子变量，存储垃圾回收阈值
  std::atomic<double> m_garbage_collection_threshold;

  // 原子变量，存储固定内存数注册线程数
  std::atomic<size_t> m_pinned_num_register_threads;

  // 原子布尔变量，指示是否可以扩展段
  std::atomic<bool> m_expandable_segments;

  // 原子布尔变量，指示是否在cudaMalloc时释放锁
  std::atomic<bool> m_release_lock_on_cudamalloc;

  // 原子布尔变量，指示是否使用CUDA主机注册固定内存
  std::atomic<bool> m_pinned_use_cuda_host_register;

  // 存储最后一次分配器设置的字符串
  std::string m_last_allocator_settings;

  // 用于保护最后一次分配器设置的互斥锁
  std::mutex m_last_allocator_settings_mutex;
};

// 结束了一个名为 `CUDACachingAllocator` 的命名空间，该命名空间位于 `c10::cuda` 命名空间下
// 这段代码可能是在声明或定义命名空间结束的位置

// 通用的缓存分配器工具函数
// 设置分配器的配置参数，根据环境变量的值来进行设置
C10_CUDA_API void setAllocatorSettings(const std::string& env);

} // namespace c10::cuda::CUDACachingAllocator


这段代码主要包括两个部分的注释：

1. 第一部分注释解释了 `};`，表明它结束了一个名为 `CUDACachingAllocator` 的命名空间，该命名空间是 `c10::cuda` 命名空间的一部分。

2. 第二部分注释解释了 `setAllocatorSettings` 函数的作用，说明它是一个通用的缓存分配器工具函数，用于根据环境变量设置分配器的配置参数。
```