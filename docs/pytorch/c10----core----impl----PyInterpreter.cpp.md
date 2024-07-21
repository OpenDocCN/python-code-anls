# `.\pytorch\c10\core\impl\PyInterpreter.cpp`

```
// 声明命名空间 c10::impl，实现 NoopPyInterpreterVTable 结构体，继承自 PyInterpreterVTable
namespace c10::impl {

// NoopPyInterpreterVTable 结构体的成员函数 name，返回未加载的解释器名称字符串
struct NoopPyInterpreterVTable final : public PyInterpreterVTable {
  std::string name() const override {
    return "<unloaded interpreter>";
  }

  // 虚函数重写：decref，用于释放 PyObject，但在此处什么也不做
  void decref(PyObject* pyobj, bool has_pyobj_slot) const override {
  } // do nothing

  // 定义宏 PANIC(m)，用于在运行时检查中断，提示尝试调用的操作名称和具体信息
#define PANIC(m)              \
  TORCH_INTERNAL_ASSERT(      \
      0,                      \
      "attempted to call " #m \
      " on a Tensor with nontrivial PyObject after corresponding interpreter died")

  // 虚函数重写：detach，当解释器失效时调用 PANIC 宏
  c10::intrusive_ptr<TensorImpl> detach(const TensorImpl* self) const override {
    PANIC(detach);
  }

  // 虚函数重写：dispatch，当解释器失效时调用 PANIC 宏
  void dispatch(const c10::OperatorHandle& op, torch::jit::Stack* stack)
      const override {
    PANIC(dispatch);
  }

  // 虚函数重写：reportErrorCallback，当解释器失效时调用 PANIC 宏
  void reportErrorCallback(PyObject* callback, DispatchKey key) const override {
    PANIC(reportErrorCallback);
  }

  // 虚函数重写：python_op_registration_trampoline，当解释器失效时调用 PANIC 宏
  void python_op_registration_trampoline(
      const c10::OperatorHandle& op,
      c10::DispatchKey,
      c10::DispatchKeySet keyset,
      torch::jit::Stack* stack,
      bool with_keyset) const override {
    PANIC(python_op_registration_trampoline);
  }

  // 虚函数重写：throw_abstract_impl_not_imported_error，当解释器失效时调用 PANIC 宏
  void throw_abstract_impl_not_imported_error(
      std::string opname,
      const char* pymodule,
      const char* context) const override {
    PANIC(throw_abstract_impl_not_imported_error);
  }

  // 虚函数重写：python_dispatcher，当解释器失效时调用 PANIC 宏
  void python_dispatcher(
      const c10::OperatorHandle& op,
      c10::DispatchKeySet,
      torch::jit::Stack* stack) const override {
    PANIC(python_dispatcher);
  }

  // 虚函数重写：is_contiguous，当解释器失效时调用 PANIC 宏
  bool is_contiguous(const TensorImpl* self, at::MemoryFormat) const override {
    PANIC(is_contiguous);
  }

  // 虚函数重写：is_strides_like，当解释器失效时调用 PANIC 宏
  bool is_strides_like(const TensorImpl* self, at::MemoryFormat)
      const override {
    PANIC(is_strides_like);
  }

  // 虚函数重写：is_non_overlapping_and_dense，当解释器失效时调用 PANIC 宏
  bool is_non_overlapping_and_dense(const TensorImpl* self) const override {
    PANIC(is_non_overlapping_and_dense);
  }

  // 虚函数重写：device，当解释器失效时调用 PANIC 宏
  c10::Device device(const TensorImpl* self) const override {
    PANIC(device);
  }

  // 虚函数重写：dim，当解释器失效时调用 PANIC 宏
  int64_t dim(const TensorImpl* self) const override {
    PANIC(dim);
  }

  // 虚函数重写：strides，当解释器失效时调用 PANIC 宏
  c10::IntArrayRef strides(const TensorImpl* self) const override {
    PANIC(strides);
  }

  // 虚函数重写：sizes，当解释器失效时调用 PANIC 宏
  c10::IntArrayRef sizes(const TensorImpl* self) const override {
    PANIC(sizes);
  }

  // 虚函数重写：sym_sizes，当解释器失效时调用 PANIC 宏
  c10::SymIntArrayRef sym_sizes(const TensorImpl* self) const override {
    PANIC(sym_sizes);
  }

  // 虚函数重写：layout，当解释器失效时调用 PANIC 宏
  c10::Layout layout(const TensorImpl* self) const override {
    PANIC(layout);
  }

  // 虚函数重写：numel，当解释器失效时调用 PANIC 宏
  int64_t numel(const TensorImpl* self) const override {
    PANIC(numel);
  }

  // 虚函数重写：sym_numel，当解释器失效时调用 PANIC 宏
  c10::SymInt sym_numel(const TensorImpl* self) const override {
    PANIC(sym_numel);
  }

  // 虚函数重写：sym_strides，当解释器失效时调用 PANIC 宏
  c10::SymIntArrayRef sym_strides(const TensorImpl* self) const override {
    PANIC(sym_strides);
  }

  // 虚函数重写：sym_storage_offset，当解释器失效时调用 PANIC 宏
  c10::SymInt sym_storage_offset(const TensorImpl* self) const override {

    PANIC(sym_storage_offset);
  }
};


这段代码定义了一个名为 `NoopPyInterpreterVTable` 的结构体，它继承自 `PyInterpreterVTable` 并实现了多个虚函数。这些函数的主要作用是在解释器失效时（使用 `PANIC` 宏），向运行时系统报告错误或中断。
  // 如果发生 PANIC，记录存储偏移值并中断程序执行
  PANIC(sym_storage_offset);
}

// 忽略 GPU 事件的创建，不执行任何操作
void trace_gpu_event_creation(c10::DeviceType device_type, uintptr_t event)
    const override {}

// 忽略 GPU 事件的删除，不执行任何操作
void trace_gpu_event_deletion(c10::DeviceType device_type, uintptr_t event)
    const override {}

// 跟踪 GPU 事件的记录
void trace_gpu_event_record(
    c10::DeviceType device_type,
    uintptr_t event,
    uintptr_t stream) const override {}

// 跟踪 GPU 事件的等待
void trace_gpu_event_wait(
    c10::DeviceType device_type,
    uintptr_t event,
    uintptr_t stream) const override {}

// 跟踪 GPU 内存的分配
void trace_gpu_memory_allocation(c10::DeviceType device_type, uintptr_t ptr)
    const override {}

// 跟踪 GPU 内存的释放
void trace_gpu_memory_deallocation(c10::DeviceType device_type, uintptr_t ptr)
    const override {}

// 跟踪 GPU 流的创建
void trace_gpu_stream_creation(c10::DeviceType device_type, uintptr_t stream)
    const override {}

// 跟踪 GPU 设备的同步
void trace_gpu_device_synchronization(
    c10::DeviceType device_type) const override {}

// 跟踪 GPU 流的同步
void trace_gpu_stream_synchronization(
    c10::DeviceType device_type,
    uintptr_t stream) const override {}

// 跟踪 GPU 事件的同步
void trace_gpu_event_synchronization(
    c10::DeviceType device_type,
    uintptr_t event) const override {}

// 重设张量的后向钩子，如果执行此操作会 PANIC
void reset_backward_hooks(const TensorImpl* self) const override {
  PANIC(reset_backward_hooks);
};
};

// `};` 结束了匿名命名空间，用于限定内部定义的函数和变量的作用域。
// 在全局作用域中构造这个变量，而不是在 `disarm` 函数内部构造它。
// 这样做可以增加 `noop_vtable` 的生存期，使其比引用它的任何对象都更长。

// 如果 `noop_vtable` 在其他对象之前离开作用域，那些对象将持有一个悬空的引用。
static NoopPyInterpreterVTable noop_vtable;

void PyInterpreter::disarm() noexcept {
  // 将 `vtable_` 设置为指向 `noop_vtable`，实现 "解除武装" 的操作。
  vtable_ = &noop_vtable;
}

} // namespace c10::impl
```