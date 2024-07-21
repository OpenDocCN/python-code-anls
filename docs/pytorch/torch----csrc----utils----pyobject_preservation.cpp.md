# `.\pytorch\torch\csrc\utils\pyobject_preservation.cpp`

```
#include <torch/csrc/utils/pyobject_preservation.h>
// 引入 torch 的 PyObject 保留工具头文件

#include <structmember.h>
// 引入结构成员操作的头文件

void clear_slots(PyTypeObject* type, PyObject* self) {
// 定义函数 clear_slots，接受类型对象指针和自身对象作为参数

  Py_ssize_t n = Py_SIZE(type);
  // 获取类型对象的大小作为循环计数器的上限

  PyMemberDef* mp = type->tp_members;
  // 获取类型对象的成员定义数组的指针

  for (Py_ssize_t i = 0; i < n; i++, mp++) {
  // 循环遍历类型对象的成员定义数组

    if (mp->type == T_OBJECT_EX && !(mp->flags & READONLY)) {
    // 如果成员类型是 T_OBJECT_EX 并且没有标记为只读

      char* addr = (char*)self + mp->offset;
      // 计算成员在对象中的地址

      PyObject* obj = *(PyObject**)addr;
      // 从对象中获取成员的指针

      if (obj != nullptr) {
      // 如果成员对象指针不为空

        *(PyObject**)addr = nullptr;
        // 将对象成员指针置空

        Py_DECREF(obj);
        // 递减对象的引用计数
      }
    }
  }
}
// 结束函数 clear_slots 的定义
```