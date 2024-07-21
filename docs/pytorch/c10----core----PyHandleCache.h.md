# `.\pytorch\c10\core\PyHandleCache.h`

```py
#pragma once

#include <c10/core/impl/PyInterpreter.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <c10/util/python_stub.h>

#include <atomic>

namespace c10 {

// A PyHandleCache represents a cached pointer from a C++ object to
// a Python object that represents that object analogously in Python.
// Upon a cache hit, the relevant object can be retrieved after a test
// and then a memory load.  Two conditions must hold to be able to use this
// class:
//
//  - This must truly be a cache; e.g., the caller must be able to produce
//    the object some other way if the cache hit misses.
//
//  - This must truly be a handle; e.g., the Python object referenced by
//    this class must have static lifetime.  This means we don't have to
//    maintain strong ownership or deallocate the object when the C++ object
//    dies.  Static lifetime is a good idea in conjunction with the cache,
//    since if you are producing a fresh object on miss you won't be
//    maintaining object identity.  If you need bidirectional ownership,
//    you will want to factor out the pattern in TensorImpl with
//    resurrection.
//
// This cache is expected to not improve perf under torchdeploy, as one
// interpreter will fill up the cache, and all the interpreters will be
// unable to use the slot.  A potential improvement is to have multiple
// slots (one per interpreter), which will work in deployment scenarios
// where there a stable, fixed number of interpreters.  You can also store
// the relevant state in the Python library, rather than in the non-Python
// library (although in many cases, this is not convenient, as there may
// not be a way to conveniently index based on the object.)
class PyHandleCache {
 public:
  PyHandleCache() : pyinterpreter_(nullptr) {}

  // Attempt to fetch the pointer from the cache, if the PyInterpreter
  // matches.  If it doesn't exist, or the cache entry is not valid,
  // use slow_accessor to get the real pointer value and return that
  // (possibly writing it to the cache, if the cache entry is
  // available.)
  template <typename F>
  PyObject* ptr_or(impl::PyInterpreter* self_interpreter, F slow_accessor)
      const {
    // Note [Memory ordering on Python interpreter tag]
    // 使用 acquire 内存顺序加载当前保存的 Python 解释器指针
    impl::PyInterpreter* interpreter =
        pyinterpreter_.load(std::memory_order_acquire);
    if (C10_LIKELY(interpreter == self_interpreter)) {
      // 如果当前保存的 Python 解释器指针与传入的 self_interpreter 相同，
      // 直接返回缓存的 Python 对象指针 data_
      return data_;
    } else if (interpreter == nullptr) {
      // 如果当前缓存中没有有效的解释器指针，则通过 slow_accessor 获取实际的指针值
      auto* r = slow_accessor();
      impl::PyInterpreter* expected = nullptr;
      // 尝试使用 acq_rel 内存顺序和 self_interpreter 比较交换当前的解释器指针
      if (pyinterpreter_.compare_exchange_strong(
              expected, self_interpreter, std::memory_order_acq_rel)) {
        // 如果交换成功，更新缓存的数据指针为 r
        data_ = r;
      }
      // 这种情况不应发生，因为应该在 GIL 保护下进行
      TORCH_INTERNAL_ASSERT(expected != self_interpreter);
      return r;
    } else {
      // 如果当前缓存中保存的解释器指针与 self_interpreter 不匹配，则通过 slow_accessor 获取实际的指针值
      return slow_accessor();
    }
  }

 private:
  std::atomic<impl::PyInterpreter*> pyinterpreter_;  // 原子操作类型的 Python 解释器指针
  mutable PyObject* data_;  // 可变的 Python 对象指针，用于缓存
};

} // namespace c10
    }
  }



// 结束了一个私有部分的类定义，这里是类的结尾



 private:
  mutable std::atomic<impl::PyInterpreter*> pyinterpreter_;
  mutable PyObject* data_{nullptr};



// 私有成员部分开始，以下是类的私有成员变量的定义
// 使用 mutable 修饰的原子类型成员变量，存储指向 impl::PyInterpreter 类型对象的指针
mutable std::atomic<impl::PyInterpreter*> pyinterpreter_;

// 使用 mutable 修饰的指针成员变量，初始化为 nullptr
mutable PyObject* data_{nullptr};
};

} // namespace c10
```