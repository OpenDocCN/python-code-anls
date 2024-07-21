# `.\pytorch\torch\csrc\copy_utils.h`

```
#pragma once

#include <torch/csrc/Types.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils.h>
#include <functional>
#include <vector>

typedef std::function<void(PyObject*, PyObject*, bool)> THPCopyFunction;
// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
struct THPCopyInfo {
  PyTypeObject* srcType; // 源张量/存储器的 Python 类型
  THPCopyFunction copy; // 复制函数
  bool non_blocking; // 如果复制实现了"非阻塞"复制，则为true
  bool broadcast; // 如果复制实现了广播复制，则为true
};
typedef std::vector<THPCopyInfo> THPCopyList;

inline bool tryTHPCopy(
    const THPCopyList& v,
    PyObject* dst,
    PyObject* src,
    bool non_blocking,
    bool broadcast) {
  for (auto& i : v) {
    // 检查是否是非阻塞模式并且源对象类型是当前循环项定义的类型
    if (i.non_blocking == non_blocking &&
        PyType_IsSubtype(Py_TYPE(src), i.srcType)) {
      // 执行复制操作
      (i.copy)(dst, src, broadcast);
      return true;
    }
  }
  return false;
}

inline bool THPCopy(
    const THPCopyList& v,
    PyObject* dst,
    PyObject* src,
    bool non_blocking,
    bool broadcast) {
  // NOLINTNEXTLINE(bugprone-branch-clone)
  // 尝试进行复制，如果成功则返回true
  if (tryTHPCopy(v, dst, src, non_blocking, broadcast)) {
    return true;
  } else if (non_blocking && tryTHPCopy(v, dst, src, false, broadcast)) {
    // 如果是非阻塞复制且尝试普通复制也成功，则返回true
    return true;
  }
  // 如果以上条件都不满足，则设置错误消息并返回false
  THPUtils_setError(
      "copy from %s to %s isn't implemented",
      THPUtils_typename(src),
      THPUtils_typename(dst));
  return false;
}
```