# `.\pytorch\c10\core\impl\PyObjectSlot.cpp`

```py
// 在 c10::impl 命名空间中定义 PyObjectSlot 类

PyObjectSlot::PyObjectSlot() : pyobj_interpreter_(nullptr), pyobj_(nullptr) {}
// PyObjectSlot 类的默认构造函数，初始化 pyobj_interpreter_ 和 pyobj_ 为 nullptr

PyObjectSlot::~PyObjectSlot() {
  maybe_destroy_pyobj();
}
// PyObjectSlot 类的析构函数，调用 maybe_destroy_pyobj() 方法

void PyObjectSlot::maybe_destroy_pyobj() {
  if (owns_pyobj()) {
    TORCH_INTERNAL_ASSERT(pyobj_interpreter_ != nullptr);
    TORCH_INTERNAL_ASSERT(pyobj_ != nullptr);
    (*pyobj_interpreter_.load(std::memory_order_acquire))
        ->decref(_unchecked_untagged_pyobj(), /*has_pyobj_slot*/ true);
    // 通过 pyobj_interpreter_ 引用的 PyInterpreter 对象递减 pyobj_ 的引用计数
    // 注意：此析构函数只有在没有对此 C++ 对象的引用（显然）和没有对 PyObject 的引用时才能进入
    // 因此可以安全地在此处清空 pyobj_，因为不可能再次使用它（除非存在弱引用竞争）
    pyobj_ = nullptr; // 为安全起见清空 pyobj_
  }
}

PyInterpreter* PyObjectSlot::pyobj_interpreter() {
  return pyobj_interpreter_.load(std::memory_order_acquire);
}
// 返回当前 PyObjectSlot 实例的 pyobj_interpreter_ 成员变量

PyObject* PyObjectSlot::_unchecked_untagged_pyobj() const {
  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  return reinterpret_cast<PyObject*>(
      reinterpret_cast<uintptr_t>(pyobj_) & ~0x1ULL);
}
// 返回去掉 pyobj_ 最低位标志位的指针，这里假设 pyobj_ 存储了指针和一些标志位信息

void PyObjectSlot::unchecked_clear_pyobj(PyInterpreter* interpreter) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(interpreter == pyobj_interpreter_.load());
  pyobj_ = nullptr;
}
// 在满足调试断言条件下，清空 pyobj_ 成员变量

PyInterpreter& PyObjectSlot::load_pyobj_interpreter() const {
  auto interpreter = pyobj_interpreter_.load(std::memory_order_acquire);
  if (interpreter) {
    return *interpreter;
  }
  TORCH_CHECK(
      false,
      "cannot access PyObject for Tensor on interpreter ",
      (*pyobj_interpreter_.load())->name());
}
// 加载并返回 pyobj_interpreter_ 成员变量指向的 PyInterpreter 对象引用

bool PyObjectSlot::check_interpreter(PyInterpreter* interpreter) {
  return interpreter == pyobj_interpreter();
}
// 检查给定的 PyInterpreter 指针是否与当前 PyObjectSlot 实例的 pyobj_interpreter_ 相匹配

bool PyObjectSlot::has_pyobj_nonhermetic() {
  return check_pyobj(pyobj_interpreter(), /*ignore_hermetic_tls=*/true)
      .has_value();
}
// 检查当前 PyObjectSlot 实例是否有非遗传的 PyObject

bool PyObjectSlot::owns_pyobj() {
  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  return reinterpret_cast<uintptr_t>(pyobj_) & 1;
}
// 检查 pyobj_ 的最低位标志位，判断当前 PyObjectSlot 实例是否拥有该 PyObject

void PyObjectSlot::set_owns_pyobj(bool b) {
  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  pyobj_ = reinterpret_cast<PyObject*>(
      reinterpret_cast<uintptr_t>(_unchecked_untagged_pyobj()) | b);
}
// 设置 pyobj_ 的最低位标志位，以指定当前 PyObjectSlot 实例是否拥有该 PyObject
```