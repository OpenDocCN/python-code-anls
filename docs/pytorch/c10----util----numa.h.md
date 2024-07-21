# `.\pytorch\c10\util\numa.h`

```
#pragma once

# 预处理指令，确保当前头文件只被编译一次


#include <c10/macros/Export.h>
#include <c10/util/Flags.h>
#include <cstddef>

# 包含其他头文件，用于导出符号、处理标志以及定义标准库的大小类型


C10_DECLARE_bool(caffe2_cpu_numa_enabled);

# 定义宏，用于声明一个外部变量 `caffe2_cpu_numa_enabled`


namespace c10 {

# 进入命名空间 `c10`


/**
 * Check whether NUMA is enabled
 */
C10_API bool IsNUMAEnabled();

# 声明函数 `IsNUMAEnabled()`，用于检查是否启用了NUMA（Non-Uniform Memory Access，非统一内存访问）


/**
 * Bind to a given NUMA node
 */
C10_API void NUMABind(int numa_node_id);

# 声明函数 `NUMABind(int numa_node_id)`，用于将当前线程绑定到指定的NUMA节点


/**
 * Get the NUMA id for a given pointer `ptr`
 */
C10_API int GetNUMANode(const void* ptr);

# 声明函数 `GetNUMANode(const void* ptr)`，用于获取指定指针 `ptr` 所在的NUMA节点ID


/**
 * Get number of NUMA nodes
 */
C10_API int GetNumNUMANodes();

# 声明函数 `GetNumNUMANodes()`，用于获取系统中的NUMA节点数目


/**
 * Move the memory pointed to by `ptr` of a given size to another NUMA node
 */
C10_API void NUMAMove(void* ptr, size_t size, int numa_node_id);

# 声明函数 `NUMAMove(void* ptr, size_t size, int numa_node_id)`，用于将指定大小的内存移动到另一个指定的NUMA节点


/**
 * Get the current NUMA node id
 */
C10_API int GetCurrentNUMANode();

# 声明函数 `GetCurrentNUMANode()`，用于获取当前线程所在的NUMA节点ID


} // namespace c10

# 结束命名空间 `c10`
```