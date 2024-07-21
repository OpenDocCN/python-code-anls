# `.\pytorch\c10\core\impl\HermeticPyObjectTLS.h`

```py
#pragma once
// 预处理指令，确保头文件只包含一次

#include <c10/macros/Export.h>
#include <atomic>

namespace c10::impl {

// 命名空间 c10::impl 内声明 HermeticPyObjectTLS 结构体

// This TLS controls whether or not we permanently associate PyObject
// with Tensor the first time it is allocated.  When hermetic PyObject
// TLS is enabled (state is true), we DO NOT save PyObjects to Tensor,
// meaning you get a distinct PyObject whenever you execute the code in
// question.
struct C10_API HermeticPyObjectTLS {
  // 静态成员函数声明，用于设置状态
  static void set_state(bool state);
  // 静态成员函数声明，用于获取状态
  static bool get_state() {
    // Hypothetical fastpath if torchdeploy/multipy isn't used.  Per
    // https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2020/p2055r0.pdf
    // this qualifies relaxed access because it is a single-location data
    // structure (only the boolean here).
    //
    // Forgetting about data races for a moment, is there a logical race?
    //
    //  - Boolean only ever transitions from false to true.  So the
    //    critical situation is when one interpreter is already running
    //    when a second interpreter switches haveState from false to true.
    //
    //  - The first interpreter is indifferent whether or not it sees
    //    hasState true/false; obviously false works (this is what the
    //    interpreter was previously using; more directly, the interpreter
    //    calls into itself as the handler, so being hermetic is not
    //    required), and true simply means serviced python operator calls will
    //    be hermetic; in these cases it is expected to be functionally
    //    equivalent.
    //
    //  - The second interpreter MUST see hasState true (as its requests will
    //    be forwarded to the first interpreter), but it is assumed that there
    //    is a synchronization between the interpreter initialization, and
    //    when we actually perform operations, so it is guaranteed to see
    //    hasState true.
    //
    // QED.
    //
    // This fastpath is currently disabled so that we can more easily test that
    // hermetic mode works correctly even on stock build of PyTorch.
    // 如果没有使用 torchdeploy/multipy 的话，这是一个假设的快速路径。
    // 根据 https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2020/p2055r0.pdf
    // 这符合放松访问的条件，因为这是一个单位置数据结构（这里只有一个布尔值）。

    if (false && !haveState_.load(std::memory_order_relaxed))
      return false;
    // 返回当前线程局部存储的状态
    return get_tls_state();
  }
  // 从 multipy/torchdeploy 的顶层调用此函数进行状态初始化
  static void init_state();

 private:
  // 这个标志只会在 torchdeploy/multipy 初始化时从 false 切换到 true，之后不会再改变。
  static std::atomic<bool> haveState_;
  // 获取线程局部存储的状态
  static bool get_tls_state();
};

} // namespace c10::impl
// 命名空间 c10::impl 结束
```