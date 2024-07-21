# `.\pytorch\aten\src\ATen\ParallelCommon.cpp`

```py
namespace at {

namespace {

// 获取环境变量的值，如果未设置则返回默认值
const char* get_env_var(
    const char* var_name, const char* def_value = nullptr) {
  const char* value = std::getenv(var_name);
  return value ? value : def_value;
}

// 获取环境变量中指定变量的线程数，如果无法转换或值无效则返回默认值
size_t get_env_num_threads(const char* var_name, size_t def_value = 0) {
  try {
    if (auto* value = std::getenv(var_name)) {
      int nthreads = std::stoi(value);
      TORCH_CHECK(nthreads > 0);  // 确保线程数大于0
      return nthreads;
    }
  } catch (const std::exception& e) {
    std::ostringstream oss;
    oss << "Invalid " << var_name << " variable value, " << e.what();
    TORCH_WARN(oss.str());  // 记录警告信息
  }
  return def_value;  // 返回默认线程数
}

} // namespace

// 返回关于并行信息的字符串
std::string get_parallel_info() {
  std::ostringstream ss;

  ss << "ATen/Parallel:\n\tat::get_num_threads() : "
     << at::get_num_threads() << '\n';  // 获取 ATen 线程数
  ss << "\tat::get_num_interop_threads() : "
     << at::get_num_interop_threads() << '\n';  // 获取 ATen 互操作线程数

  ss << at::get_openmp_version() << '\n';  // 获取 OpenMP 版本信息
#ifdef _OPENMP
  ss << "\tomp_get_max_threads() : " << omp_get_max_threads() << '\n';  // 获取 OpenMP 最大线程数
#endif

  ss << at::get_mkl_version() << '\n';  // 获取 MKL 版本信息
#if AT_MKL_ENABLED()
  ss << "\tmkl_get_max_threads() : " << mkl_get_max_threads() << '\n';  // 获取 MKL 最大线程数
#endif

  ss << at::get_mkldnn_version() << '\n';  // 获取 MKLDNN 版本信息

  ss << "std::thread::hardware_concurrency() : "
     << std::thread::hardware_concurrency() << '\n';  // 获取硬件并发支持的线程数

  ss << "Environment variables:" << '\n';
  ss << "\tOMP_NUM_THREADS : "
     << get_env_var("OMP_NUM_THREADS", "[not set]") << '\n';  // 获取环境变量 OMP_NUM_THREADS
  ss << "\tMKL_NUM_THREADS : "
     << get_env_var("MKL_NUM_THREADS", "[not set]") << '\n';  // 获取环境变量 MKL_NUM_THREADS

  ss << "ATen parallel backend: ";
  #if AT_PARALLEL_OPENMP
  ss << "OpenMP";  // ATen 使用 OpenMP 并行后端
  #elif AT_PARALLEL_NATIVE
  ss << "native thread pool";  // ATen 使用本地线程池并行后端
  #endif
  #ifdef C10_MOBILE
  ss << " [mobile]";  // 如果是 C10_MOBILE，则标记为移动设备
  #endif
  ss << '\n';

  #if AT_EXPERIMENTAL_SINGLE_THREAD_POOL
  ss << "Experimental: single thread pool" << std::endl;  // 使用实验性单线程池
  #endif

  return ss.str();  // 返回并行信息字符串
}

// 返回默认的内部操作线程数
int intraop_default_num_threads() {
#ifdef C10_MOBILE
  // 对于移动设备，内部操作线程池大小应由移动设备的 CPU 信息决定
  // 如果需要，在移动设备上调用这个 API，我们应该与 caffe2/utils/threadpool 中的逻辑连接
  TORCH_CHECK(false, "Undefined intraop_default_num_threads on mobile.");  // 在移动设备上调用该函数时抛出错误
#else
  size_t nthreads = get_env_num_threads("OMP_NUM_THREADS", 0);  // 获取环境变量 OMP_NUM_THREADS 的值
  nthreads = get_env_num_threads("MKL_NUM_THREADS", nthreads);  // 获取环境变量 MKL_NUM_THREADS 的值，若未设置则使用上一个获取的值
  if (nthreads == 0) {
#if defined(FBCODE_CAFFE2) && defined(__aarch64__)
    nthreads = 1;  // 在特定条件下，设置线程数为 1
#else
#if defined(__aarch64__) && defined(__APPLE__)
    // 在 Apple Silicon 上，有高效和性能核心
    // 默认情况下，限制并行算法到性能核心
    int32_t num_cores = -1;
    size_t num_cores_len = sizeof(num_cores);
    // 获取 Apple Silicon 设备的核心数信息
    sysctlbyname("hw.activecpu", &num_cores, &num_cores_len, NULL, 0);
    nthreads = static_cast<size_t>(num_cores);  // 将核心数转换为 size_t 类型
#endif
#endif
  }
#endif

  return nthreads;  // 返回内部操作线程数
}
    // 调用系统函数 sysctlbyname 获取物理 CPU 核心数，并将结果存储在 num_cores 变量中
    if (sysctlbyname("hw.perflevel0.physicalcpu", &num_cores, &num_cores_len, nullptr, 0) == 0) {
        // 如果获取的物理 CPU 核心数大于 1
        if (num_cores > 1) {
            // 将核心数赋值给线程数变量 nthreads
            nthreads = num_cores;
            // 返回获取的核心数
            return num_cores;
        }
    }
#endif
    // 结束一个条件编译块，对应于之前的 #ifdef 或 #if 指令
    nthreads = TaskThreadPoolBase::defaultNumThreads();
#endif
  }
  // 返回线程数的整数值
  return static_cast<int>(nthreads);
#endif /* !defined(C10_MOBILE) */
}

} // namespace at
```