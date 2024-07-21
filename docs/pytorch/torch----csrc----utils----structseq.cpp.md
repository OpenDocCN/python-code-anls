# `.\pytorch\torch\csrc\utils\structseq.cpp`

```py
/*
 * 该文件的版权归 Python 软件基金会所有
 *
 * 此文件是从 CPython 源代码复制并进行了修改：
 * https://github.com/python/cpython/blob/master/Objects/structseq.c
 * https://github.com/python/cpython/blob/2.7/Objects/structseq.c
 *
 * 此文件的目的是覆盖 structseq 的 repr 默认行为，
 * 以提供更好的打印输出，用于来自运算符的 structseq 对象，例如 torch.return_types.*
 *
 * 有关 CPython 版权的更多信息，请参见：
 * https://github.com/python/cpython#copyright-and-license-information
 */

#include <torch/csrc/utils/six.h>        // 导入六个模块的头文件，用于 Python 2/3 兼容性
#include <torch/csrc/utils/structseq.h>  // 导入结构序列的工具函数
#include <sstream>                       // 导入字符串流库，用于构建字符串

#include <structmember.h>  // 导入结构成员定义的头文件，定义了结构体的成员信息

namespace torch::utils {  // 定义 torch::utils 命名空间

// 注意：自 Python 3.8 开始，PyStructSequence 的内置 repr 方法已更新，
// 因此可能不再需要此函数。
PyObject* returned_structseq_repr(PyStructSequence* obj) {
  PyTypeObject* typ = Py_TYPE(obj);  // 获取对象的类型对象指针
  THPObjectPtr tup = six::maybeAsTuple(obj);  // 将对象转换为元组表示
  if (tup == nullptr) {
    return nullptr;  // 如果转换失败，返回空指针
  }

  std::stringstream ss;  // 创建一个字符串流对象 ss
  ss << typ->tp_name << "(\n";  // 将类型名称添加到字符串流中

  Py_ssize_t num_elements = Py_SIZE(obj);  // 获取结构序列对象的元素数量

  for (Py_ssize_t i = 0; i < num_elements; i++) {  // 遍历结构序列对象的每个元素
    const char* cname = typ->tp_members[i].name;  // 获取成员的名称
    if (cname == nullptr) {  // 如果成员名称为空指针
      PyErr_Format(
          PyExc_SystemError,
          "In structseq_repr(), member %zd name is nullptr"
          " for type %.500s",
          i,
          typ->tp_name);
      return nullptr;  // 报告系统错误并返回空指针
    }

    PyObject* val = PyTuple_GetItem(tup.get(), i);  // 获取元组中的值对象
    if (val == nullptr) {
      return nullptr;  // 如果获取值失败，返回空指针
    }

    auto repr = THPObjectPtr(PyObject_Repr(val));  // 获取值对象的 repr 表示
    if (repr == nullptr) {
      return nullptr;  // 如果获取 repr 表示失败，返回空指针
    }

    const char* crepr = PyUnicode_AsUTF8(repr);  // 将 repr 对象转换为 UTF-8 字符串
    if (crepr == nullptr) {
      return nullptr;  // 如果转换失败，返回空指针
    }

    ss << cname << '=' << crepr;  // 将成员名称和其 repr 表示添加到字符串流中
    if (i < num_elements - 1) {
      ss << ",\n";  // 如果不是最后一个成员，添加逗号和换行符
    }
  }
  ss << ")";  // 添加结构序列对象的结束符号

  return PyUnicode_FromString(ss.str().c_str());  // 返回构建的字符串流表示的 Python Unicode 对象
}

}  // namespace torch::utils
```