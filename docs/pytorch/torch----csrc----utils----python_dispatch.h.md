# `.\pytorch\torch\csrc\utils\python_dispatch.h`

```py
#include <pybind11/pybind11.h>
#include <torch/csrc/utils/pybind.h>

namespace torch::impl::dispatch {

// 声明函数 initDispatchBindings，用于初始化分发相关的绑定
void initDispatchBindings(PyObject* module);

// 声明函数 python_op_registration_trampoline_impl，实现 Python 操作的注册转发
// 参数：
//   - op: 操作的句柄，表示要注册的具体操作
//   - key: 分发键，指定操作的分发方式
//   - keyset: 分发键集合，表示操作可能使用的分发键集合
//   - stack: Torch JIT 的堆栈，用于操作调用
//   - with_keyset: 布尔值，指示是否使用分发键集合
void python_op_registration_trampoline_impl(
    const c10::OperatorHandle& op,
    c10::DispatchKey key,
    c10::DispatchKeySet keyset,
    torch::jit::Stack* stack,
    bool with_keyset);

} // namespace torch::impl::dispatch
```