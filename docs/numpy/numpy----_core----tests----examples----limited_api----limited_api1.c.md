# `.\numpy\numpy\_core\tests\examples\limited_api\limited_api1.c`

```py
#define Py_LIMITED_API 0x03060000

# 定义了 Py_LIMITED_API 宏，指定了 Python 的限制 API 版本为 0x03060000


#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/ufuncobject.h>

# 包含了必要的头文件：Python.h 是 Python C API 的主头文件，arrayobject.h 和 ufuncobject.h 是 NumPy 数组对象和通用函数对象的头文件


static PyModuleDef moduledef = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "limited_api1"
};

# 定义了名为 moduledef 的静态变量，类型为 PyModuleDef，初始化了其中的 m_base 和 m_name 字段。m_base 使用了 PyModuleDef_HEAD_INIT 宏进行初始化，m_name 设置为字符串 "limited_api1"


PyMODINIT_FUNC PyInit_limited_api1(void)
{
    import_array();
    import_umath();
    return PyModule_Create(&moduledef);
}

# PyMODINIT_FUNC 表明这是一个 Python 模块初始化函数，名称为 PyInit_limited_api1。函数内部首先调用 import_array() 和 import_umath() 函数来导入 NumPy 的数组和通用数学函数模块。然后使用 PyModule_Create(&moduledef) 来创建一个 Python 模块对象，返回给 Python 解释器。
```