# `.\pytorch\c10\core\impl\PyObjectSlot.h`

```py
#pragma once

#include <c10/core/impl/HermeticPyObjectTLS.h>
#include <c10/core/impl/PyInterpreter.h>
#include <c10/util/Optional.h>
#include <c10/util/python_stub.h>

#include <atomic>

namespace c10::impl {

struct C10_API PyObjectSlot {
 public:
  // 构造函数，初始化 PyObjectSlot 对象
  PyObjectSlot();

  // 析构函数，释放 PyObjectSlot 对象
  ~PyObjectSlot();

  // 可能销毁 pyobj，根据需要执行清理操作
  void maybe_destroy_pyobj();

  // 初始化 pyobj，关联 TensorImpl 和指定的 PyObject，并根据状态标记解释器
  //
  // 注意：这个函数位于头文件中，以便可以在编译时优化掉对状态的开关
  //
  // 注意：这个函数可能会抛出异常。在必要时确保清理 PyObject！
  void init_pyobj(
      PyInterpreter* self_interpreter,
      PyObject* pyobj,
      PyInterpreterStatus status) {
    impl::PyInterpreter* expected = nullptr;
    switch (status) {
      case impl::PyInterpreterStatus::DEFINITELY_UNINITIALIZED:
        // 调用者保证没有多线程访问；如果没有数据竞争，可以使用 relaxed store
        pyobj_interpreter_.store(self_interpreter, std::memory_order_relaxed);
        break;
      case impl::PyInterpreterStatus::TAGGED_BY_US:
        // 不需要标记，标记已经是正确的了
        break;
      case impl::PyInterpreterStatus::MAYBE_UNINITIALIZED:
        // 尝试使用指定的解释器标记这个 TensorImpl
        if (pyobj_interpreter_.compare_exchange_strong(
                expected, self_interpreter, std::memory_order_acq_rel)) {
          break;
        }
        // 检查实际上是否已经被我们标记了！这种情况不可能由竞争引起，
        // 但可能由于某些情况下保守地将张量标记为 MAYBE_UNINITIALIZED
        // （因为他们没有预先检查标记），当实际上它已经被解释器拥有时。
        if (expected == self_interpreter) {
          break;
        }
        // 没有赢得竞争，继续执行。我们保证不会与自己竞争，因为使用相同解释器 ID 的 init_pyobj 调用必须由 GIL 串行化
        [[fallthrough]];
      case impl::PyInterpreterStatus::TAGGED_BY_OTHER:
        TORCH_CHECK(
            false,
            "cannot allocate PyObject for Tensor on interpreter ",
            self_interpreter,
            " that has already been used by another torch deploy interpreter ",
            pyobj_interpreter_.load());
    }

    // 我们是唯一一个可以到达这一点的线程。由于 GIL 保护访问，不可能与另一个零解释器发生冲突。
    // 注意：owns_pyobj 标记最初为 false
    // （此处应该继续注释，但根据指示不应超出代码块范围）
  }
};
} // namespace c10::impl


这段代码是一个 C++ 的类实现，用于管理与 PyObject 相关的数据结构。类中定义了构造函数、析构函数和初始化函数，用于管理和标记与 TensorImpl 相关的 PyObject。
    pyobj_ = pyobj;
  }

  // 查询 PyObject 解释器。如果没有解释器可能返回 null。这里存在竞争条件！
  PyInterpreter* pyobj_interpreter();

  // 返回未经检查的未打标签的 PyObject
  PyObject* _unchecked_untagged_pyobj() const;

  // 检查解释器标签。如果标记为当前解释器，则返回非空（可能为 null）的 PyObject。
  // 如果未打标签，则返回 nullopt。如果明确无效，则抛出错误。
  //
  // 如果 `ignore_hermetic_tls` 为 false，并且从遗传上下文调用此函数
  // （即 `HermeticPyObjectTLS::get_state()` 为 true），则返回 nullopt。
  // 如果 `ignore_hermetic_tls` 为 true，则忽略遗传上下文，允许在遗传上下文中检查非遗传 PyObject 的解释器标签。
  // 这是必要的，因为有些情况下，非遗传 PyObject 的析构函数会在遗传上下文中调用，因此必须正确处理为非遗传 PyObject。
  //
  // 注意：此函数位于头文件中，以避免实际创建 std::optional
  std::optional<PyObject*> check_pyobj(
      PyInterpreter* self_interpreter,
      bool ignore_hermetic_tls = false) const {
    // 注意 [Python 解释器标签的内存顺序]
    impl::PyInterpreter* interpreter =
        pyobj_interpreter_.load(std::memory_order_acquire);
    if (interpreter == nullptr) {
      // 注意：这里永远不会返回 DEFINITELY_UNINITIALIZED，因为可能有其他线程在此处竞争初始化。
      // 只有在刚刚分配并且尚未逃逸到其他线程时，我们才能确定张量是明确未初始化的
      return c10::nullopt;
    } else if (interpreter == self_interpreter) {
      // 注意：pyobj_ 仍然可能为 null！
      if (!ignore_hermetic_tls && c10::impl::HermeticPyObjectTLS::get_state()) {
        return c10::nullopt;
      } else {
        return c10::make_optional(_unchecked_untagged_pyobj());
      }
    } else {
      TORCH_CHECK(
          false,
          "cannot access PyObject for Tensor on interpreter ",
          (*self_interpreter)->name(),
          " that has already been used by another torch deploy interpreter ",
          (*pyobj_interpreter_.load())->name());
`
};  // 结束命名空间 c10::impl 的作用域
}   // 结束命名空间的作用域
```