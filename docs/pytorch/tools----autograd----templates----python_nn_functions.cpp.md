# `.\pytorch\tools\autograd\templates\python_nn_functions.cpp`

```py
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 定义宏，用于只包含方法操作符

// 引入所需的头文件
#include "torch/csrc/Device.h"
#include "torch/csrc/DynamicTypes.h"
#include "torch/csrc/Exceptions.h"
#include "torch/csrc/autograd/python_nn_functions.h"
#include "torch/csrc/autograd/generated/python_return_types.h"
#include "torch/csrc/autograd/python_variable.h"
#include "torch/csrc/autograd/utils/wrap_outputs.h"
#include "torch/csrc/autograd/utils/python_arg_parsing.h"
#include "torch/csrc/utils/pycfunction_helpers.h"
#include "torch/csrc/utils/python_arg_parser.h"
#include "torch/csrc/utils/structseq.h"
#include "torch/csrc/utils/tensor_memoryformats.h"

// 如果未定义每个操作符的头文件，则引入 ATen 库的全局函数
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
$ops_headers
#endif

// 使用命名空间 torch::autograd
using namespace torch::autograd;

// 定义静态全局变量 THPNNVariableFunctionsModule
static PyObject* THPNNVariableFunctionsModule = NULL;

// 定义函数 _parse_to，解析并处理 to 方法的参数
static PyObject * THPVariable__parse_to(PyObject* module, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  // 定义静态 PythonArgParser 对象，指定支持的参数格式
  static PythonArgParser parser({
    "to(Device device=None, ScalarType dtype=None, bool non_blocking=False, bool copy=False, *, MemoryFormat? memory_format=None)",
    "to(ScalarType dtype, bool non_blocking=False, bool copy=False, *, MemoryFormat? memory_format=None)",
    "to(Tensor tensor, bool non_blocking=False, bool copy=False, *, MemoryFormat? memory_format=None)",
  });
  // 解析参数
  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  // 如果解析结果含有 torch function，调用 torch function 处理
  if (r.has_torch_function()) {
    return handle_torch_function(r, args, kwargs, THPNNVariableFunctionsModule, "torch.nn", "_parse_to");
  }
  // 解析转换操作
  auto parsed = parse_to_conversion(r, /*allow_copy*/ false); // we don't want copy for nn.Module.to
  auto& device = std::get<0>(parsed);
  auto& scalarType = std::get<1>(parsed);
  auto non_blocking = std::get<2>(parsed);
  auto opt_memory_format = std::get<4>(parsed);
  // 创建一个 Python 元组对象
  auto tuple = THPObjectPtr{PyTuple_New(4)};
  if (!tuple) throw python_error();
  // 根据解析结果设置元组中的各个元素
  if (device) {
    PyTuple_SET_ITEM(tuple.get(), 0, THPDevice_New(*device));
  } else {
    Py_INCREF(Py_None);
    PyTuple_SET_ITEM(tuple.get(), 0, Py_None);
  }
  if (scalarType) {
    PyTuple_SET_ITEM(tuple.get(), 1, Py_NewRef(torch::getTHPDtype(*scalarType)));
  } else {
    Py_INCREF(Py_None);
    PyTuple_SET_ITEM(tuple.get(), 1, Py_None);
  }
  PyTuple_SET_ITEM(tuple.get(), 2, torch::autograd::utils::wrap(non_blocking));
  if (opt_memory_format.has_value()) {
    PyTuple_SET_ITEM(tuple.get(), 3, Py_NewRef(torch::utils::getTHPMemoryFormat(opt_memory_format.value())));
  } else {
    Py_INCREF(Py_None);
    PyTuple_SET_ITEM(tuple.get(), 3, Py_None);
  }
  // 返回元组对象
  return tuple.release();
  END_HANDLE_TH_ERRORS
}

// generated forward declarations start here

// PyMethodDef 数组，定义了模块中可调用的函数及其对应的 C 函数指针
static PyMethodDef nn_functions[] = {
  {"_parse_to", castPyCFunctionWithKeywords(THPVariable__parse_to),
   METH_VARARGS | METH_KEYWORDS, nullptr},
  ${py_forwards}
  {nullptr}
};
    // 定义一个包含 METH_VARARGS 和 METH_KEYWORDS 标志的方法描述结构体
    METH_VARARGS | METH_KEYWORDS, nullptr},
    // 插入由Python方法定义组成的代码
    ${py_method_defs}
    // 结束方法定义列表，用NULL结尾
    {NULL}
};

void initNNFunctions(PyObject* module) {
  // 定义静态的 PyModuleDef 结构体，用于定义模块
  static struct PyModuleDef def = {
     PyModuleDef_HEAD_INIT,    // 使用默认的头初始化
     "torch._C._nn",           // 模块名为 torch._C._nn
     NULL,                     // 模块文档字符串为空
     -1,                       // 模块状态不为子模块
     nn_functions              // 指向 nn_functions 的指针
  };
  // 创建 Python 模块对象 nn，并将其赋给 THPNNVariableFunctionsModule
  PyObject* nn = PyModule_Create(&def);
  THPNNVariableFunctionsModule = nn;
  // 如果创建失败，抛出 python_error 异常
  if (!nn) {
    throw python_error();
  }
  // 将模块 nn 添加到给定模块中，并偷取其引用
  if (PyModule_AddObject(module, "_nn", nn) != 0) {
    throw python_error();
  }
}

// generated methods start here

${py_methods}

} // namespace torch::autograd
```