# `.\pytorch\tools\autograd\templates\python_return_types.cpp`

```
#include <Python.h>

#include <vector>
#include <map>
#include <string>

#include "torch/csrc/autograd/generated/python_return_types.h"
#include "torch/csrc/utils/structseq.h"
#include "torch/csrc/Exceptions.h"

// 声明命名空间 torch::autograd::generated
namespace torch { namespace autograd { namespace generated {

// 插入生成的 Python 返回类型定义
${py_return_types}

}}} // closing namespaces

// 声明命名空间 torch::autograd
namespace torch::autograd {

// 静态函数：向模块中添加返回类型定义
static void addReturnType(
    PyObject* module,         // Python 模块对象
    const char* name,         // 返回类型的名称
    PyTypeObject* type) {     // 返回类型的 PyTypeObject
  // 在用户删除或覆盖它的极少情况下，持有 TypeObject 的引用
  Py_INCREF(type);
  // 将类型对象添加到模块中
  if (PyModule_AddObject(
          module,
          name,
          (PyObject*)type) != 0) {
    // 添加失败时释放类型对象并抛出 Python 异常
    Py_DECREF(type);
    throw python_error();
  }
}

// 初始化返回类型模块
void initReturnTypes(PyObject* module) {
  // 定义返回类型模块的结构
  static struct PyModuleDef def = {
      PyModuleDef_HEAD_INIT, "torch._C._return_types", nullptr, -1, {}};
  // 创建返回类型模块对象
  PyObject* return_types_module = PyModule_Create(&def);
  if (!return_types_module) {
    // 创建失败时抛出 Python 异常
    throw python_error();
  }

  // 注册生成的 Python 返回类型
  ${py_return_types_registrations}

  // 添加返回类型模块到主模块中，并在成功时窃取引用
  if (PyModule_AddObject(module, "_return_types", return_types_module) != 0) {
    Py_DECREF(return_types_module);
    // 添加失败时抛出 Python 异常
    throw python_error();
  }
}

} // namespace torch::autograd
```