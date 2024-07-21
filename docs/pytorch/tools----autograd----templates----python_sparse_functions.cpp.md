# `.\pytorch\tools\autograd\templates\python_sparse_functions.cpp`

```py
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 定义宏，用于指定仅为方法运算符进行断言

// 包含 Torch 库的头文件
#include "torch/csrc/Device.h"
#include "torch/csrc/DynamicTypes.h"
#include "torch/csrc/Exceptions.h"
#include "torch/csrc/autograd/python_sparse_functions.h"
#include "torch/csrc/autograd/python_variable.h"
#include "torch/csrc/autograd/utils/wrap_outputs.h"
#include "torch/csrc/autograd/utils/python_arg_parsing.h"
#include "torch/csrc/utils/pycfunction_helpers.h"
#include "torch/csrc/utils/python_arg_parser.h"
#include "torch/csrc/utils/structseq.h"

// 如果未定义每个操作符的头文件，则包含 ATen 库的 Functions.h
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
// 否则，包含由变量 $ops_headers 指定的头文件
#else
$ops_headers
#endif

// 使用 ATen 的命名空间
using at::Tensor;
using at::Scalar;
using at::ScalarType;
using at::MemoryFormat;
using at::Generator;
using at::IntArrayRef;
using at::TensorList;

// 使用 Torch 自动求导工具的 utils 命名空间
using namespace torch::autograd::utils;

// Torch 自动求导命名空间
namespace torch::autograd {

// 自动生成的前向声明从这里开始

${py_forwards}

// Python 方法定义数组，存储稀疏函数的定义
static PyMethodDef sparse_functions[] = {
  ${py_method_defs}
  {NULL}  // 结束标记
};

// THPSparseVariableFunctionsModule 的声明
static PyObject* THPSparseVariableFunctionsModule = NULL;

// 初始化稀疏函数模块
void initSparseFunctions(PyObject* module) {
  // 定义 PyModuleDef 结构体
  static struct PyModuleDef def = {
     PyModuleDef_HEAD_INIT,  // 使用默认的模块定义头
     "torch._C._sparse",     // 模块名
     NULL,                   // 模块文档
     -1,                     // 状态 -1 表示可选
     sparse_functions        // Python 方法定义数组
  };
  // 创建稀疏函数模块
  PyObject* sparse = PyModule_Create(&def);
  THPSparseVariableFunctionsModule = sparse;
  // 如果创建失败，抛出 Python 异常
  if (!sparse) {
    throw python_error();
  }
  // 将稀疏函数模块添加到指定的 Python 模块中
  // PyModule_AddObject 会窃取对 sparse 的引用
  if (PyModule_AddObject(module, "_sparse", sparse) != 0) {
    throw python_error();
  }
}

// 自动生成的方法从这里开始

${py_methods}

} // namespace torch::autograd
```