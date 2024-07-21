# `.\pytorch\torch\csrc\stub.c`

```
// 包含 Python.h 头文件，用于在 C/C++ 中调用 Python API
#include <Python.h>

// 声明 initModule 函数的外部链接，表示该函数在其他地方定义
extern PyObject* initModule(void);

// 如果不是在 Windows 平台下，则进行以下定义
#ifndef _WIN32

// 如果是在 C++ 环境下编译，则声明为 C 风格函数
#ifdef __cplusplus
extern "C"
#endif

// 使用 visibility("default") 属性，表示 PyInit__C 函数的默认可见性为公共
__attribute__((visibility("default"))) PyObject* PyInit__C(void);
#endif

// PyMODINIT_FUNC 是一个宏，展开后为 void，并表示函数为 Python 模块的初始化函数
PyMODINIT_FUNC PyInit__C(void)
{
    // 调用外部定义的 initModule 函数来初始化模块
    return initModule();
}
```