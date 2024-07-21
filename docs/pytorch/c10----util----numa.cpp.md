# `.\pytorch\c10\util\numa.cpp`

```
// 包含 C10 库中的 Exception 头文件
#include <c10/util/Exception.h>
// 包含 C10 库中的 NUMA 头文件
#include <c10/util/numa.h>

// 定义一个名为 caffe2_cpu_numa_enabled 的布尔型全局变量，默认为 false，用于指示是否启用 NUMA
C10_DEFINE_bool(caffe2_cpu_numa_enabled, false, "Use NUMA whenever possible.");

// 在 Linux 平台、启用 NUMA、非移动设备情况下进行条件编译
#if defined(__linux__) && defined(C10_USE_NUMA) && !defined(C10_MOBILE)
// 包含 NUMA 相关头文件
#include <numa.h>
#include <numaif.h>
#include <unistd.h>
// 定义宏 C10_ENABLE_NUMA
#define C10_ENABLE_NUMA
#endif

// 命名空间 c10
namespace c10 {

// 如果定义了宏 C10_ENABLE_NUMA，则定义以下函数
#ifdef C10_ENABLE_NUMA
// 返回 NUMA 是否已启用的布尔函数
bool IsNUMAEnabled() {
  return FLAGS_caffe2_cpu_numa_enabled && numa_available() >= 0;
}

// 绑定当前进程或线程到指定的 NUMA 节点
void NUMABind(int numa_node_id) {
  // 如果传入的 NUMA 节点 ID 小于 0，则直接返回
  if (numa_node_id < 0) {
    return;
  }
  // 如果 NUMA 没有启用，则直接返回
  if (!IsNUMAEnabled()) {
    return;
  }

  // 检查传入的 NUMA 节点 ID 是否有效
  TORCH_CHECK(
      numa_node_id <= numa_max_node(),
      "NUMA node id ",
      numa_node_id,
      " is unavailable");

  // 分配一个 NUMA 节点掩码并设置指定节点
  auto bm = numa_allocate_nodemask();
  numa_bitmask_setbit(bm, numa_node_id);
  numa_bind(bm);
  numa_bitmask_free(bm);
}

// 获取给定指针所在的 NUMA 节点 ID
int GetNUMANode(const void* ptr) {
  // 如果 NUMA 没有启用，则返回 -1
  if (!IsNUMAEnabled()) {
    return -1;
  }
  // 断言指针不为空
  AT_ASSERT(ptr);

  int numa_node = -1;
  // 获取指定内存位置的 NUMA 节点 ID
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  TORCH_CHECK(
      get_mempolicy(
          &numa_node,
          nullptr,
          0,
          const_cast<void*>(ptr),
          MPOL_F_NODE | MPOL_F_ADDR) == 0,
      "Unable to get memory policy, errno:",
      errno);
  return numa_node;
}

// 获取系统中配置的 NUMA 节点数量
int GetNumNUMANodes() {
  // 如果 NUMA 没有启用，则返回 -1
  if (!IsNUMAEnabled()) {
    return -1;
  }

  return numa_num_configured_nodes();
}

// 将指定大小的内存移动到指定的 NUMA 节点
void NUMAMove(void* ptr, size_t size, int numa_node_id) {
  // 如果传入的 NUMA 节点 ID 小于 0，则直接返回
  if (numa_node_id < 0) {
    return;
  }
  // 如果 NUMA 没有启用，则直接返回
  if (!IsNUMAEnabled()) {
    return;
  }
  // 断言指针不为空
  AT_ASSERT(ptr);

  // 计算页面起始地址，并计算偏移量
  uintptr_t page_start_ptr =
      ((reinterpret_cast<uintptr_t>(ptr)) & ~(getpagesize() - 1));
  // NOLINTNEXTLINE(*-conversions)
  ptrdiff_t offset = reinterpret_cast<uintptr_t>(ptr) - page_start_ptr;
  // 避免额外的动态分配和 NUMA API 调用
  AT_ASSERT(
      numa_node_id >= 0 &&
      static_cast<unsigned>(numa_node_id) < sizeof(unsigned long) * 8);
  unsigned long mask = 1UL << numa_node_id;
  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  TORCH_CHECK(
      mbind(
          reinterpret_cast<void*>(page_start_ptr),
          size + offset,
          MPOL_BIND,
          &mask,
          sizeof(mask) * 8,
          MPOL_MF_MOVE | MPOL_MF_STRICT) == 0,
      "Could not move memory to a NUMA node");
}

// 获取当前线程或进程所在的 NUMA 节点 ID
int GetCurrentNUMANode() {
  // 如果 NUMA 没有启用，则返回 -1
  if (!IsNUMAEnabled()) {
    return -1;
  }

  // 获取当前 CPU 所在的 NUMA 节点 ID
  auto n = numa_node_of_cpu(sched_getcpu());
  return n;
}

// 如果未定义宏 C10_ENABLE_NUMA，则定义以下函数
#else // C10_ENABLE_NUMA

// 返回 NUMA 是否已启用的布尔函数，始终返回 false
bool IsNUMAEnabled() {
  return false;
}

// 空函数，不执行任何操作
void NUMABind(int numa_node_id) {}

// 返回指定指针所在的 NUMA 节点 ID，始终返回 -1
int GetNUMANode(const void* ptr) {
  return -1;
}

// 返回配置的 NUMA 节点数量，始终返回 -1
int GetNumNUMANodes() {
  return -1;
}

// 空函数，不执行任何操作
void NUMAMove(void* ptr, size_t size, int numa_node_id) {}

// 返回当前线程或进程所在的 NUMA 节点 ID，始终返回 -1
int GetCurrentNUMANode() {
  return -1;
}

#endif // C10_NUMA_ENABLED

} // namespace c10
```