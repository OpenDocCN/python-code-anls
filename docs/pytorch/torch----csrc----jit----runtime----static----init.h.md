# `.\pytorch\torch\csrc\jit\runtime\static\init.h`

```py
#include <torch/csrc/jit/python/pybind_utils.h>

// 包含了 Torch 的 C++ 库中用于 Python 绑定的头文件 `pybind_utils.h`


namespace torch::jit {

// 进入 torch::jit 命名空间，用于包含 Torch 的 JIT（即时编译）模块相关功能


void initStaticModuleBindings(PyObject* module);

// 声明了一个函数 `initStaticModuleBindings`，该函数用于初始化静态模块的绑定，接受一个 `PyObject*` 参数作为模块对象。


} // namespace torch::jit

// 退出 torch::jit 命名空间，结束对 JIT 模块的命名空间定义。
```