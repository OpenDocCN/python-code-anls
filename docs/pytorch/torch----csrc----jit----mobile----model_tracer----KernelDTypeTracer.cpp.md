# `.\pytorch\torch\csrc\jit\mobile\model_tracer\KernelDTypeTracer.cpp`

```py
// 包含头文件：用于 Torch 的移动端 JIT 编译器模块中的核心功能
#include <torch/csrc/jit/mobile/model_tracer/KernelDTypeTracer.h>

// 包含 C++ 标准库中的数据结构和工具
#include <map>
#include <mutex>
#include <set>
#include <string>

// 命名空间：定义了 Torch 的 JIT 编译器模块
namespace torch {
namespace jit {
namespace mobile {

// 构造函数：初始化 KernelDTypeTracer 对象
KernelDTypeTracer::KernelDTypeTracer() {
  // 定义记录回调函数，用于捕获 RecordFunction 事件
  auto recorder_cb =
      [](const at::RecordFunction& fn) -> std::unique_ptr<at::ObserverContext> {
    // 获取 RecordFunction 的名称
    std::string name = fn.name();
    // 查找名称中第一个 '$' 符号的位置
    size_t dollar_pos = name.find_first_of('$');
    // 提取 kernel_tag，即 '$' 符号之前的部分
    std::string kernel_tag = name.substr(0, dollar_pos);
    // 提取 dtype，即 '$' 符号之后的部分
    std::string dtype = name.substr(dollar_pos + 1);

    // 使用互斥锁访问被调用的 kernel_tags 映射
    getCalledKernelTags().withLock([&](kernel_tags_type& kernel_tags) {
      // 将 dtype 插入到 kernel_tag 的集合中
      kernel_tags[kernel_tag].insert(dtype);
    });
    // 返回空指针，表示不需要执行任何特定的 ObserverContext 操作
    return nullptr;
  };

  // 将全局回调函数添加到 Torch 中，监听 KERNEL_FUNCTION_DTYPE 范围内的事件
  handle_ = at::addGlobalCallback(
      at::RecordFunctionCallback(recorder_cb)
          .scopes({at::RecordScope::KERNEL_FUNCTION_DTYPE}));
}

// 获取被调用的 kernel_tags 映射的方法
c10::Synchronized<KernelDTypeTracer::kernel_tags_type>& KernelDTypeTracer::
    getCalledKernelTags() {
  // 定义并返回静态的 synchronized 对象 called_kernel_tags
  static c10::Synchronized<kernel_tags_type> called_kernel_tags;
  return called_kernel_tags;
}

} // namespace mobile
} // namespace jit
} // namespace torch
```