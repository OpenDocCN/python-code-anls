# `.\pytorch\c10\core\impl\PyInterpreter.h`

```
#pragma once

#include <c10/core/Device.h>
#include <c10/core/DispatchKeySet.h>
#include <c10/core/Layout.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/SymIntArrayRef.h>
#include <c10/macros/Export.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/intrusive_ptr.h>
#include <c10/util/python_stub.h>
#include <string>
#include <vector>

// Forward declarations

namespace c10 {
struct IValue;
class OperatorHandle;
struct TensorImpl;
} // namespace c10

namespace torch::jit {
using Stack = std::vector<c10::IValue>;
}

// Actual implementation

namespace c10::impl {

// Forward declaration of PyInterpreter struct
struct C10_API PyInterpreter;

// Note [Python interpreter tag]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Traditionally, PyTorch is layered such that our Python library
// (libtorch_python) references our pure C++ library (libtorch) as the
// natural order of things.  However, sometimes this natural order is
// subverted: C++ objects refer to Python objects (for example, we
// store a PyObject* pointer on TensorImpl so that converting from a
// C++ Tensor to a Python Tensor is just a memory dereference).
//
// These unusual orderings must be treated with care.  To start, you need to
// virtualize the destructor so that the PyObject can be decref'ed on
// destruction (because the C++ object itself doesn't know anything about
// Python--remember, layering!).  This process itself is fraught, since
// acquiring the GIL could lead to deadlocks if someone is blocking on you
// while holding the GIL.  Furthermore, if the C++ objects outlive the
// interpreter (which can happen if you stash them in a static global
// variable defined in libtorch), you may attempt to decref the object when
// the Python interpreter has already been shutdown.
//
// BUT WAIT, IT GETS WORSE.  With torchdeploy, there may be multiple Python
// interpreters in a single process. If a C++ object is accessible from
// multiple interpreters, we must take care not to accidentally pass a
// PyObject from one interpreter with another interpreter.
//
// To prevent these mixups, we introduce a PyInterpreter "tag" (object with
// a vtable), which specifies a specific Python interpreter.
//
//  - Any given object can be associated with AT MOST one Python interpreter.
//    We represent the interpreter tag as a memory address to an instance of
//    a virtual class that is allocated once per interpreter (this is so that
//    we can request the interpreter to perform operations for us, if
//    necessary).
//
//  - It can be recorded with a PyObject (PyInterpreterObject) so that
//    we know what interpreter the object is associated with, and we can
//    raise an error if you try to use the PyObject from the wrong
//    interpreter context.
//
//  - It contains a vtable that can be used to perform various Python
//    operations from ordinary C++ code that ordinarily wouldn't be accessible
//    from libtorch.
//
// A simple use case is when a C++ object must be associated with a PyObject.
struct C10_API PyInterpreter;
} // namespace c10::impl
// However, for TensorImpl, we lazily allocate a PyObject the first time the
// object passes into Python.  The invariants for this situation are more
// subtle:

//  - A given TensorImpl's interpreter tag can only go from uninitialized to
//    tagged; once tagged, this is a quiescent state (once tagged to an
//    interpreter, ALWAYS tagged to that interpreter)

//  - A thread may mutate the PyObject field of a TensorImpl if and only if it
//    holds the GIL for the interpreter tagged on the TensorImpl.  (If the
//    TensorImpl is not tagged, it must first atomically claim its tag before it
//    can validly write)

// WARNING: This class has to be written very carefully, because it may be
// possible for a Tensor to have a reference an interpreter corresponding to
// a shared library that has ALREADY BEEN UNLOADED.  This makes blindly calling
// virtual methods very dangerous, because the vtable may be garbage at that
// point (on a good day, you might get "pure virtual method called").

// The idea to solve this problem is we always leak PyInterpreters (so they
// always stay live even after dlclose), and make sure we can disarm their
// virtual methods by indirecting through a separate PyInterpreterVTable
// object.  This can be replaced with a no-op vtable from libc10.so, which
// is guaranteed to stick around until the bitter end.

// NB: The downside with representing PyInterpreter tags as full objects is that
// it takes an extra word on TensorImpl.  If tags were instead just integer
// indices, on 64-bit architectures we could pack the tag and PyObject together
// into a single atomic word.  On 32-bit architectures we could simply say that
// only one Python interpreter is supported (erroring if a nontrivial
// interpreter tag is attempted to be set).

// The difficulty with this scheme is we need to maintain an out-of-line table
// to get at the PyInterpreters so that we can do virtual method calls on them,
// and registration/deregistration to this table must be done in a thread safe
// manner.  This can be easily done if the number of possible PyInterpreters is
// small enough (e.g., 8-bit integer) by simply preallocating an array of
// sufficient size to hold all possible interpreters.  Surely 128 threads is
// more than enough for anyone!

// I didn't decide to do this technique at the moment, because the extra word
// added by the PyInterpreter tag takes us to 24 words, which means that we
// still fit inside three eight word cache lines.  If you need to penny pinch
// another word consider doing this!
    // 返回 vtable_ 指针的值，结束函数执行
    return vtable_;
  }

  // 使此 PyInterpreter 失效，使其所有方法变为空操作。
  // 目前 vtable 指针不是原子的，这意味着在执行 disarm() 函数时，
  // 如果有活动的析构函数同时运行，会导致不安全，并触发 TSAN（线程错误分析工具）。
  // 希望这种情况实际上永远不会发生；张量的销毁应该在 dlclose 发生时静止，
  // 任何长期存在的张量，它们的析构函数在这里被失效只有在进程关闭时才开始销毁过程
  // （在 dlclose 之后很长时间）。
  void disarm() noexcept;
// PyInterpreterStatus 描述了其解释器状态相对于当前持有 GIL 的线程的状态。
enum class PyInterpreterStatus {
  // 我们刚刚分配了张量，它还没有逃逸到其他线程，
  // 我们确信它绝对没有被标记为关联到某个解释器。
  DEFINITELY_UNINITIALIZED,

  // 我们查询了解释器字段，看起来它是未初始化的。但是
  // 另一个线程可能已经与我们竞争，标记了其他解释器 id。
  // 因此，我们必须进行 CEX（？？？）以确保我们可以
  // 实际抓住它。
  MAYBE_UNINITIALIZED,

  // 我们查询了解释器字段，并且它被标记为属于我们。
  // 这意味着我们拥有独占的写访问权限（因为我们持有这个解释器的 GIL）。
  TAGGED_BY_US,

  // 别人标记了这个。我们不能从 Python 使用这个 TensorImpl。
  TAGGED_BY_OTHER,
};
} // namespace c10::impl
```