# `.\pytorch\torch\csrc\lazy\python\python_util.cpp`

```
// 包含 Torch 的 Python 工具头文件
#include <torch/csrc/lazy/python/python_util.h>

// 包含 Python 标准库头文件
#include <Python.h>

// 包含 Python frame 相关的头文件
#include <frameobject.h>

// 包含 Pybind11 库的头文件
#include <pybind11/pybind11.h>

// 包含 Torch 核心调试工具头文件
#include <torch/csrc/lazy/core/debug_util.h>

// 包含 Torch 的 Python 绑定工具头文件
#include <torch/csrc/utils/pybind.h>

// 包含 Torch 的 Python 兼容性工具头文件
#include <torch/csrc/utils/python_compat.h>

// 包含 Torch 的 Python 字符串处理工具头文件
#include <torch/csrc/utils/python_strings.h>

// Torch 的命名空间开始
namespace torch {
namespace lazy {

// 获取当前 Python 帧的顶部位置信息
std::optional<SourceLocation> GetPythonFrameTop() {
  // 如果 Python 尚未初始化，返回空
  if (!Py_IsInitialized()) {
    return c10::nullopt;
  }
  // 获取全局解释器锁
  pybind11::gil_scoped_acquire gil;
  // 获取当前帧对象
  PyFrameObject* frame = PyEval_GetFrame();
  // 如果帧对象为空，返回空
  if (frame == nullptr) {
    return c10::nullopt;
  }
  // 构造源代码位置信息对象
  SourceLocation loc;
  // 获取当前帧对象的代码对象
  auto code = THPCodeObjectPtr(PyFrame_GetCode(frame));
  // 获取当前帧对象的行号
  loc.line = PyFrame_GetLineNumber(frame);
  // 获取当前帧对象的文件名
  loc.file = THPUtils_unpackString(code->co_filename);
  // 获取当前帧对象的函数名
  loc.function = THPUtils_unpackString(code->co_name);
  // 返回构造的位置信息对象
  return loc;
}

// 获取当前所有 Python 帧的位置信息列表
std::vector<SourceLocation> GetPythonFrames() {
  // 初始化帧位置信息列表
  std::vector<SourceLocation> frames;
  // 如果 Python 已经初始化
  if (Py_IsInitialized()) {
    // 获取全局解释器锁
    pybind11::gil_scoped_acquire gil;
    // 获取当前帧对象
    PyFrameObject* frame = PyEval_GetFrame();
    // 如果当前帧对象非空，增加引用计数
    if (frame != nullptr) {
      Py_INCREF(frame);
    }
    // 循环获取所有帧对象的位置信息
    while (frame != nullptr) {
      // 构造源代码位置信息对象
      SourceLocation loc;
      // 获取当前帧对象的代码对象
      auto code = THPCodeObjectPtr(PyFrame_GetCode(frame));
      // 获取当前帧对象的行号
      loc.line = PyFrame_GetLineNumber(frame);
      // 获取当前帧对象的文件名
      loc.file = THPUtils_unpackString(code->co_filename);
      // 获取当前帧对象的函数名
      loc.function = THPUtils_unpackString(code->co_name);
      // 将当前帧的位置信息添加到列表中
      frames.push_back(std::move(loc));
      // 获取下一个后续帧对象
      auto new_frame = PyFrame_GetBack(frame);
      // 释放当前帧对象的引用计数
      Py_DECREF(frame);
      // 将下一个帧对象设为当前帧对象
      frame = new_frame;
    }
  }
  // 返回所有帧的位置信息列表
  return frames;
}

} // namespace lazy
} // namespace torch
```