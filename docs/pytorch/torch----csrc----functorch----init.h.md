# `.\pytorch\torch\csrc\functorch\init.h`

```py
#include <Python.h>


// 包含 Python.h 头文件，这是 CPython 的 C/C++ 接口头文件

namespace torch::functorch::impl {

void initFuncTorchBindings(PyObject* module);

}
```