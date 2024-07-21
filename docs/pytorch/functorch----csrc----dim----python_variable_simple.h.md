# `.\pytorch\functorch\csrc\dim\python_variable_simple.h`

```
// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once
// note: pytorch's python variable simple includes pybind which conflicts with minpybind
// so this file just reproduces the minimial API needed to extract Tensors from python objects.

#include <torch/csrc/python_headers.h>
#include <ATen/core/Tensor.h>
#include <torch/csrc/Export.h>

// Python object that backs torch.autograd.Variable
// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
struct THPVariable {
  PyObject_HEAD;
  // Payload
  c10::MaybeOwned<at::Tensor> cdata;  // 存储一个可能拥有的 ATen Tensor 对象
  // Hooks to be run on backwards pass (corresponds to Python attr
  // '_backwards_hooks', set by 'register_hook')
  PyObject* backward_hooks = nullptr;  // 反向传播过程中运行的钩子
};

TORCH_PYTHON_API extern PyObject *THPVariableClass;  // THPVariable 的 Python 类对象
TORCH_PYTHON_API extern PyObject *ParameterClass;    // Parameter 的 Python 类对象

// 将 ATen TensorBase 对象包装成 THPVariable 对象并返回对应的 Python 对象
TORCH_PYTHON_API PyObject * THPVariable_Wrap(at::TensorBase var);

// 检查给定的 Python 对象是否是 THPVariable 类的实例
inline bool THPVariable_Check(PyObject *obj)
{
  if (!THPVariableClass)
      return false;

  const auto result = PyObject_IsInstance(obj, THPVariableClass);
  AT_ASSERT(result != -1);
  return result;
}

// 从 THPVariable 对象中解包出 ATen Tensor 引用并返回
inline const at::Tensor& THPVariable_Unpack(THPVariable* var) {
  return *var->cdata;
}

// 从 Python 对象中解包出 ATen Tensor 引用并返回
inline const at::Tensor& THPVariable_Unpack(PyObject* obj) {
  return THPVariable_Unpack(reinterpret_cast<THPVariable*>(obj));
}

// 获取 Python 解释器的实例对象
TORCH_PYTHON_API c10::impl::PyInterpreter* getPyInterpreter();
```