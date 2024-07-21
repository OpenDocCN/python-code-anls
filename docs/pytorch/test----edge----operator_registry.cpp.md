# `.\pytorch\test\edge\operator_registry.cpp`

```
// 获取全局唯一的内核注册表对象，确保每个程序中仅有一个实例
namespace torch {
namespace executor {

KernelRegistry& getKernelRegistry() {
  // 静态变量保证 kernel_registry 只被初始化一次
  static KernelRegistry kernel_registry;
  // 返回内核注册表的引用
  return kernel_registry;
}

// 注册一组内核函数到内核注册表中
bool register_kernels(const ArrayRef<Kernel>& kernels) {
  // 调用内核注册表对象的方法注册内核函数
  return getKernelRegistry().register_kernels(kernels);
}

// 将一组内核函数注册到内核注册表中
bool KernelRegistry::register_kernels(const ArrayRef<Kernel>& kernels) {
  // 遍历传入的内核函数数组
  for (const auto& kernel : kernels) {
    // 将每个内核函数名和对应的函数指针存储到内部的映射中
    this->kernels_map_[kernel.name_] = kernel.kernel_;
  }
  // 操作成功返回 true
  return true;
}

// 检查是否存在特定名称的内核函数
bool hasKernelFn(const char* name) {
  // 调用内核注册表对象的方法检查是否存在指定名称的内核函数
  return getKernelRegistry().hasKernelFn(name);
}

// 在内核注册表中查找并返回特定名称的内核函数
KernelFunction& getKernelFn(const char* name) {
  // 调用内核注册表对象的方法获取特定名称的内核函数
  return getKernelRegistry().getKernelFn(name);
}

// 查找并返回内核注册表中特定名称的内核函数对象
KernelFunction& KernelRegistry::getKernelFn(const char* name) {
  // 在内部映射中查找指定名称的内核函数
  auto kernel = this->kernels_map_.find(name);
  // 如果找不到，则抛出异常，提示内核未找到
  TORCH_CHECK_MSG(kernel != this->kernels_map_.end(), "Kernel not found!");
  // 返回找到的内核函数对象的引用
  return kernel->second;
}

} // namespace executor
} // namespace torch
```