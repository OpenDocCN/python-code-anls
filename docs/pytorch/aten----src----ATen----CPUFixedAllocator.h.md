# `.\pytorch\aten\src\ATen\CPUFixedAllocator.h`

```
// 预处理指令，确保头文件只被包含一次
#pragma once

// 包含 C10 核心的 Allocator 头文件
#include <c10/core/Allocator.h>
// 包含 C10 异常处理的头文件
#include <c10/util/Exception.h>

// 该文件创建了一个虚拟的分配器，如果实际被使用，会抛出异常。

// state 是传递给分配器的状态，是当 ATen 释放 blob 时调用的 std::function<void(void*)>

// 命名空间 at 开始
namespace at {

// 静态函数 cpu_fixed_malloc，未指定返回类型，默认 int
static cpu_fixed_malloc(void*, ptrdiff_t) {
  // 抛出错误，尝试调整一个外部 blob 的张量视图大小
  AT_ERROR("attempting to resize a tensor view of an external blob");
}

// 静态函数 cpu_fixed_realloc，未指定返回类型，默认 int
static cpu_fixed_realloc(void*, void*, ptrdiff_t) {
  // 抛出错误，尝试调整一个外部 blob 的张量视图大小
  AT_ERROR("attempting to resize a tensor view of an external blob");
}

// 静态函数 cpu_fixed_free，接受 state 和 allocation 参数
static cpu_fixed_free(void* state, void* allocation) {
  // 将 state 转换为 std::function<void(void*)> 指针
  auto on_release = static_cast<std::function<void(void*)>*>(state);
  // 调用 on_release 函数指针，释放 allocation
  (*on_release)(allocation);
  // 删除 on_release 指针
  delete on_release;
}

// CPU_fixed_allocator 是 Allocator 结构体实例
static Allocator CPU_fixed_allocator = {
    // 分配器的 malloc 函数指针设置为 cpu_fixed_malloc
    cpu_fixed_malloc,
    // 分配器的 realloc 函数指针设置为 cpu_fixed_realloc
    cpu_fixed_realloc,
    // 分配器的 free 函数指针设置为 cpu_fixed_free
    cpu_fixed_free};

// 命名空间 at 结束
} // namespace at
```