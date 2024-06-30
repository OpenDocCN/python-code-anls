# `D:\src\scipysrc\scipy\scipy\_lib\_fpumode.c`

```
/*
   Python.h 包含了Python C API的定义，使得C代码能够与Python解释器进行交互
   stdio.h 包含了标准输入输出的定义，提供了对输入输出的支持

   对于 MSC 编译器，设置浮点数精度为精确模式，并允许对浮点数环境的访问
*/
#include <Python.h>
#include <stdio.h>

/*
   定义一个静态字符串，描述了 get_fpu_mode 函数的文档字符串
   get_fpu_mode()
   获取当前的FPU控制字，以平台相关的格式返回
   如果在当前平台上未实现，则返回 None
*/
static char get_fpu_mode_doc[] = (
    "get_fpu_mode()\n"
    "\n"
    "Get the current FPU control word, in a platform-dependent format.\n"
    "Returns None if not implemented on current platform.");

/*
   获取当前的FPU控制字的函数定义
   参数 self 和 args 用于与Python解释器进行交互
*/
static PyObject *
get_fpu_mode(PyObject *self, PyObject *args)
{
    // 解析传入的参数，这里不需要任何参数
    if (!PyArg_ParseTuple(args, "")) {
        return NULL;
    }

    // 如果使用 MSC 编译器
#if defined(_MSC_VER)
    {
        // 声明并初始化一个无符号整数 result，用于存储控制字
        unsigned int result = 0;
        // 调用 _controlfp 函数获取当前FPU控制字
        result = _controlfp(0, 0);
        // 将获取的结果转换为 Python 的长整型对象并返回
        return PyLong_FromLongLong(result);
    }
// 如果使用 GCC 编译器，并且在 x86_64 或 i386 架构上
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
    {
        // 声明并初始化一个无符号短整数 cw，用于存储控制字
        unsigned short cw = 0;
        // 使用内联汇编指令 fstcw 将当前FPU控制字存入 cw 中
        __asm__("fstcw %w0" : "=m" (cw));
        // 将获取的结果转换为 Python 的长整型对象并返回
        return PyLong_FromLongLong(cw);
    }
// 如果不是以上平台，则返回 Python 的 None 对象
#else
    Py_RETURN_NONE;
#endif
}

/*
   定义一个包含函数方法信息的数组 methods
   包括一个名为 get_fpu_mode 的方法，其对应的C函数是 get_fpu_mode，接受任意数量的参数
   方法的文档字符串是 get_fpu_mode_doc
   最后一个元素是空指针，用于表示方法列表的结尾
*/
static struct PyMethodDef methods[] = {
    {"get_fpu_mode", get_fpu_mode, METH_VARARGS, get_fpu_mode_doc},
    {NULL, NULL, 0, NULL}  // 结尾标记，表示方法列表结束
};

/*
   定义一个 PyModuleDef 结构体 moduledef
   设置模块的头部信息，名称为 _fpumode，其他字段为 NULL 或 -1
   methods 指向之前定义的方法列表
   其他字段暂时不需要赋值
*/
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_fpumode",
    NULL,
    -1,
    methods,
    NULL,
    NULL,
    NULL,
    NULL
};

/*
   PyInit__fpumode 函数定义
   当 Python 解释器初始化该模块时，调用此函数来创建并返回一个新的 Python 模块对象
*/
PyMODINIT_FUNC
PyInit__fpumode(void)
{
    // 创建并返回一个新的 Python 模块对象，其中包含之前定义的 moduledef 结构
    return PyModule_Create(&moduledef);
}
```