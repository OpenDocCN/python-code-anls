# `.\pytorch\c10\core\impl\alloc_cpu.cpp`

```py
// 包含C10核心实现的CPU分配器头文件
#include <c10/core/impl/alloc_cpu.h>

// 包含C10核心的对齐定义
#include <c10/core/alignment.h>
// 包含C10实用工具的标志
#include <c10/util/Flags.h>
// 包含C10实用工具的日志记录
#include <c10/util/Logging.h>
// 包含C10实用工具的范围迭代
#include <c10/util/irange.h>
// 包含C10实用工具的NUMA支持
#include <c10/util/numa.h>

#ifdef USE_MIMALLOC
// 如果定义了USE_MIMALLOC，则包含mimalloc的头文件
#include <mimalloc.h>
#endif

#ifdef __linux__
// 如果在Linux系统下编译，则包含内存映射相关的头文件
#include <sys/mman.h>
#include <unistd.h>
#endif

// TODO: 将标志重命名为C10

// 定义一个标志，确定在CPU分配内存时是否进行内存清零操作
C10_DEFINE_bool(
    caffe2_cpu_allocator_do_zero_fill,
    false,
    "If set, do memory zerofilling when allocating on CPU");

// 定义一个标志，确定在CPU分配内存时是否使用随机数据填充
C10_DEFINE_bool(
    caffe2_cpu_allocator_do_junk_fill,
    false,
    "If set, fill memory with deterministic junk when allocating on CPU");

namespace c10 {

namespace {

// 使用特定的垃圾模式填充数据内存区域，长度为num字节
// 垃圾值被选择为当作浮点数解释时为NaN，或者非常大的整数
void memset_junk(void* data, size_t num) {
  // 当被解释为浮点数或非常大的整数时，这个垃圾模式是NaN
  static constexpr int32_t kJunkPattern = 0x7fedbeef;
  static constexpr int64_t kJunkPattern64 =
      static_cast<int64_t>(kJunkPattern) << 32 | kJunkPattern;
  auto int64_count = num / sizeof(kJunkPattern64);
  auto remaining_bytes = num % sizeof(kJunkPattern64);
  int64_t* data_i64 = reinterpret_cast<int64_t*>(data);
  // 使用循环填充数据内存区域
  for (const auto i : c10::irange(int64_count)) {
    data_i64[i] = kJunkPattern64;
  }
  // 如果还有剩余的字节没有填充，使用memcpy填充
  if (remaining_bytes > 0) {
    memcpy(data_i64 + int64_count, &kJunkPattern64, remaining_bytes);
  }
}

#if defined(__linux__) && !defined(__ANDROID__)
// 检查THP（透明大页）分配是否已启用
static inline bool is_thp_alloc_enabled() {
  static bool value = [&] {
    const char* ptr = std::getenv("THP_MEM_ALLOC_ENABLE");
    return ptr != nullptr ? std::atoi(ptr) : 0;
  }();
  return value;
}

// 计算内存对齐方式，根据是否启用THP（透明大页）来选择对齐方式
inline size_t c10_compute_alignment(size_t nbytes) {
  // 获取系统页大小
  static const auto pagesize = sysconf(_SC_PAGESIZE);
  // 如果内核未提供页大小，则默认为4K
  const size_t thp_alignment = (pagesize < 0 ? gPagesize : pagesize);
  return (is_thp_alloc_enabled() ? thp_alignment : gAlignment);
}

// 检查是否应用THP（透明大页）分配策略，适用于较大的内存缓冲区
inline bool is_thp_alloc(size_t nbytes) {
  return (is_thp_alloc_enabled() && (nbytes >= gAlloc_threshold_thp));
}
#elif !defined(__ANDROID__) && !defined(_MSC_VER)
// 对于非Linux系统，使用默认的对齐方式
constexpr size_t c10_compute_alignment(C10_UNUSED size_t nbytes) {
  return gAlignment;
}

// 对于非Linux系统，不启用THP（透明大页）分配策略
constexpr bool is_thp_alloc(C10_UNUSED size_t nbytes) {
  return false;
}
#endif
} // namespace

// CPU分配器的实现函数，分配大小为nbytes的内存空间
void* alloc_cpu(size_t nbytes) {
  if (nbytes == 0) {
    return nullptr;
  }
  // 如果传入的字节数为负数，抛出异常
  CAFFE_ENFORCE(
      ((ptrdiff_t)nbytes) >= 0,
      "alloc_cpu() seems to have been called with negative number: ",
      nbytes);

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  void* data;
#ifdef __ANDROID__
  // 在 Android 平台下，使用 memalign 分配内存，按 gAlignment 对齐，分配 nbytes 字节大小的内存空间
  data = memalign(gAlignment, nbytes);
  // 确保内存分配成功，否则输出错误信息
  CAFFE_ENFORCE(
      data,
      "DefaultCPUAllocator: not enough memory: you tried to allocate ",
      nbytes,
      " bytes.");
#elif defined(_MSC_VER)
#ifdef USE_MIMALLOC
  // 在 Windows 平台下，如果启用了 Mimalloc，则使用 mi_malloc_aligned 分配对齐的内存空间
  data = mi_malloc_aligned(nbytes, gAlignment);
#else
  // 在 Windows 平台下，使用 _aligned_malloc 分配对齐的内存空间
  data = _aligned_malloc(nbytes, gAlignment);
#endif
  // 确保内存分配成功，否则输出错误信息
  CAFFE_ENFORCE(
      data,
      "DefaultCPUAllocator: not enough memory: you tried to allocate ",
      nbytes,
      " bytes.");
#else
  // 在其他 POSIX 兼容系统下，使用 posix_memalign 分配内存，根据 nbytes 和对齐要求
  int err = posix_memalign(&data, c10_compute_alignment(nbytes), nbytes);
  // 确保内存分配成功，否则输出错误信息，包括错误码和错误描述
  CAFFE_ENFORCE(
      err == 0,
      "DefaultCPUAllocator: can't allocate memory: you tried to allocate ",
      nbytes,
      " bytes. Error code ",
      err,
      " (",
      strerror(err),
      ")");
  // 如果分配的内存是大页内存，则尝试使用 MADV_HUGEPAGE 提升性能（仅限 Linux 平台）
  if (is_thp_alloc(nbytes)) {
#ifdef __linux__
    // MADV_HUGEPAGE 仅适用于 Linux 平台
    int ret = madvise(data, nbytes, MADV_HUGEPAGE);
    // 如果设置失败，输出警告信息
    if (ret != 0) {
      TORCH_WARN_ONCE("thp madvise for HUGEPAGE failed with ", strerror(errno));
    }
#endif
  }
#endif

  // 将分配的内存数据移动到当前线程的 NUMA 节点
  NUMAMove(data, nbytes, GetCurrentNUMANode());
  // 检查是否同时请求了 zero-fill 和 junk-fill，这两者不能同时进行
  CHECK(
      !FLAGS_caffe2_cpu_allocator_do_zero_fill ||
      !FLAGS_caffe2_cpu_allocator_do_junk_fill)
      << "Cannot request both zero-fill and junk-fill at the same time";
  // 如果设置了 zero-fill，则将内存清零
  if (FLAGS_caffe2_cpu_allocator_do_zero_fill) {
    memset(data, 0, nbytes);
  } else if (FLAGS_caffe2_cpu_allocator_do_junk_fill) {
    // 如果设置了 junk-fill，则使用特定的方式填充内存
    memset_junk(data, nbytes);
  }

  // 返回分配的内存指针
  return data;
}

void free_cpu(void* data) {
#ifdef _MSC_VER
#ifdef USE_MIMALLOC
  // 在 Windows 平台下，使用 Mimalloc 释放内存
  mi_free(data);
#else
  // 在 Windows 平台下，使用 _aligned_free 释放内存
  _aligned_free(data);
#endif
#else
  // 在其他平台下，使用标准的 free 函数释放内存
  // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
  free(data);
#endif
}

} // namespace c10
```