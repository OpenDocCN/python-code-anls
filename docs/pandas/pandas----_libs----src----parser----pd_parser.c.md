# `D:\src\scipysrc\pandas\pandas\_libs\src\parser\pd_parser.c`

```
/*
Copyright (c) 2023, PyData Development Team
All rights reserved.

Distributed under the terms of the BSD Simplified License.
*/
#define _PANDAS_PARSER_IMPL

#include "pandas/parser/pd_parser.h"
#include "pandas/parser/io.h"
#include "pandas/portable.h"

// 定义一个静态函数 to_double，用于将字符串转换为双精度浮点数
static int to_double(char *item, double *p_value, char sci, char decimal,
                     int *maybe_int) {
  char *p_end = NULL;
  int error = 0;

  // 调用 precise_xstrtod 函数进行精确的字符串转双精度浮点数操作
  *p_value =
      precise_xstrtod(item, &p_end, decimal, sci, '\0', 1, &error, maybe_int);

  return (error == 0) && (!*p_end);
}

// 定义一个静态函数 floatify，用于将 Python 对象转换为双精度浮点数
static int floatify(PyObject *str, double *result, int *maybe_int) {
  char *data;
  PyObject *tmp = NULL;
  const char sci = 'E';
  const char dec = '.';

  // 检查传入的 Python 对象类型，并获取对应的字符串数据
  if (PyBytes_Check(str)) {
    data = PyBytes_AS_STRING(str);
  } else if (PyUnicode_Check(str)) {
    tmp = PyUnicode_AsUTF8String(str);
    if (tmp == NULL) {
      return -1;
    }
    data = PyBytes_AS_STRING(tmp);
  } else {
    // 如果传入对象类型无效，抛出 TypeError 异常
    PyErr_SetString(PyExc_TypeError, "Invalid object type");
    return -1;
  }

  // 调用 to_double 函数将字符串数据转换为双精度浮点数，并处理特殊情况
  const int status = to_double(data, result, sci, dec, maybe_int);

  if (!status) {
    // 处理特殊的浮点数情况，如无穷大和负无穷大
    if (strlen(data) == 3) {
      if (0 == strcasecmp(data, "inf")) {
        *result = HUGE_VAL;
        *maybe_int = 0;
      } else {
        goto parsingerror;
      }
    } else if (strlen(data) == 4) {
      if (0 == strcasecmp(data, "-inf")) {
        *result = -HUGE_VAL;
        *maybe_int = 0;
      } else if (0 == strcasecmp(data, "+inf")) {
        *result = HUGE_VAL;
        *maybe_int = 0;
      } else {
        goto parsingerror;
      }
    } else if (strlen(data) == 8) {
      if (0 == strcasecmp(data, "infinity")) {
        *result = HUGE_VAL;
        *maybe_int = 0;
      } else {
        goto parsingerror;
      }
    } else if (strlen(data) == 9) {
      if (0 == strcasecmp(data, "-infinity")) {
        *result = -HUGE_VAL;
        *maybe_int = 0;
      } else if (0 == strcasecmp(data, "+infinity")) {
        *result = HUGE_VAL;
        *maybe_int = 0;
      } else {
        goto parsingerror;
      }
    } else {
      goto parsingerror;
    }
  }

  Py_XDECREF(tmp);
  return 0;

parsingerror:
  // 若解析失败，抛出 ValueError 异常
  PyErr_Format(PyExc_ValueError, "Unable to parse string \"%s\"", data);
  Py_XDECREF(tmp);
  return -1;
}

// 定义一个静态函数 pandas_parser_destructor，用于销毁 PandasParser 对象
static void pandas_parser_destructor(PyObject *op) {
  void *ptr = PyCapsule_GetPointer(op, PandasParser_CAPSULE_NAME);
  PyMem_Free(ptr);
}

// 定义一个静态函数 pandas_parser_exec，用于执行 PandasParser 的初始化工作
static int pandas_parser_exec(PyObject *Py_UNUSED(module)) {
  // 分配 PandasParser_CAPI 结构体的内存空间
  PandasParser_CAPI *capi = PyMem_Malloc(sizeof(PandasParser_CAPI));
  if (capi == NULL) {
    // 如果内存分配失败，抛出内存错误异常
    PyErr_NoMemory();
    return -1;
  }

  // 将自定义的双精度转换函数注册到 C-API 结构体中
  capi->to_double = to_double;
  // 将浮点数化函数注册到 C-API 结构体中
  capi->floatify = floatify;
  // 将创建新的数据源函数注册到 C-API 结构体中
  capi->new_rd_source = new_rd_source;
  // 将删除数据源函数注册到 C-API 结构体中
  capi->del_rd_source = del_rd_source;
  // 将缓冲读取字节函数注册到 C-API 结构体中
  capi->buffer_rd_bytes = buffer_rd_bytes;
  // 将无符号整数状态初始化函数注册到 C-API 结构体中
  capi->uint_state_init = uint_state_init;
  // 将无符号整数冲突处理函数注册到 C-API 结构体中
  capi->uint64_conflict = uint64_conflict;
  // 将列迭代设置函数注册到 C-API 结构体中
  capi->coliter_setup = coliter_setup;
  // 将解析器新建函数注册到 C-API 结构体中
  capi->parser_new = parser_new;
  // 将解析器初始化函数注册到 C-API 结构体中
  capi->parser_init = parser_init;
  // 将解析器释放函数注册到 C-API 结构体中
  capi->parser_free = parser_free;
  // 将解析器删除函数注册到 C-API 结构体中
  capi->parser_del = parser_del;
  // 将解析器添加跳过行函数注册到 C-API 结构体中
  capi->parser_add_skiprow = parser_add_skiprow;
  // 将解析器设置跳过前 n 行函数注册到 C-API 结构体中
  capi->parser_set_skipfirstnrows = parser_set_skipfirstnrows;
  // 将解析器设置默认选项函数注册到 C-API 结构体中
  capi->parser_set_default_options = parser_set_default_options;
  // 将解析器消耗行数据函数注册到 C-API 结构体中
  capi->parser_consume_rows = parser_consume_rows;
  // 将解析器修剪缓冲区函数注册到 C-API 结构体中
  capi->parser_trim_buffers = parser_trim_buffers;
  // 将所有行标记化函数注册到 C-API 结构体中
  capi->tokenize_all_rows = tokenize_all_rows;
  // 将 n 行标记化函数注册到 C-API 结构体中
  capi->tokenize_nrows = tokenize_nrows;
  // 将字符串转换为 int64 函数注册到 C-API 结构体中
  capi->str_to_int64 = str_to_int64;
  // 将字符串转换为 uint64 函数注册到 C-API 结构体中
  capi->str_to_uint64 = str_to_uint64;
  // 将字符串转换为 double 函数注册到 C-API 结构体中
  capi->xstrtod = xstrtod;
  // 将精确字符串转换为 double 函数注册到 C-API 结构体中
  capi->precise_xstrtod = precise_xstrtod;
  // 将双精度数的往返转换函数注册到 C-API 结构体中
  capi->round_trip = round_trip;
  // 将布尔值转换函数注册到 C-API 结构体中
  capi->to_boolean;

  // 创建一个 Python capsule 对象，将 C-API 结构体封装其中
  PyObject *capsule =
      PyCapsule_New(capi, PandasParser_CAPSULE_NAME, pandas_parser_destructor);
  // 如果创建失败，释放 C-API 结构体内存并返回错误标志
  if (capsule == NULL) {
    PyMem_Free(capi);
    return -1;
  }

  // 导入 pandas 模块
  PyObject *pandas = PyImport_ImportModule("pandas");
  // 如果导入失败，设置错误信息并释放 capsule 对象和 C-API 结构体内存，返回错误标志
  if (!pandas) {
    PyErr_SetString(PyExc_ImportError,
                    "pd_parser.c could not import module pandas");
    Py_DECREF(capsule);
    return -1;
  }

  // 将 capsule 对象添加为 pandas 模块的 "_pandas_parser_CAPI" 属性
  // 如果添加失败，释放 capsule 对象并返回错误标志
  if (PyModule_AddObject(pandas, "_pandas_parser_CAPI", capsule) < 0) {
    Py_DECREF(capsule);
    return -1;
  }

  // 成功注册 C-API，返回成功标志
  return 0;
}

static PyModuleDef_Slot pandas_parser_slots[] = {
    {Py_mod_exec, pandas_parser_exec}, {0, NULL}};



# 定义静态变量 pandas_parser_slots，用于存储 PyModuleDef_Slot 结构的数组
static PyModuleDef_Slot pandas_parser_slots[] = {
    # 第一个元素，指定模块执行的方式为 Py_mod_exec，执行函数为 pandas_parser_exec
    {Py_mod_exec, pandas_parser_exec},
    # 第二个元素，终止标志，为 NULL
    {0, NULL}};



static struct PyModuleDef pandas_parsermodule = {
    PyModuleDef_HEAD_INIT,
    .m_name = "pandas._libs.pandas_parser",
    .m_doc = "Internal module with parser support for other extensions",
    .m_size = 0,
    .m_methods = NULL,
    .m_slots = pandas_parser_slots};



# 定义静态变量 pandas_parsermodule，用于描述 Python 模块的结构信息
static struct PyModuleDef pandas_parsermodule = {
    # 初始化 Python 模块定义的头部信息
    PyModuleDef_HEAD_INIT,
    # 模块名称，指定为 "pandas._libs.pandas_parser"
    .m_name = "pandas._libs.pandas_parser",
    # 模块的文档字符串，描述为 "Internal module with parser support for other extensions"
    .m_doc = "Internal module with parser support for other extensions",
    # 模块的大小，设置为 0
    .m_size = 0,
    # 模块的方法，此处为 NULL，表示没有额外的方法
    .m_methods = NULL,
    # 模块的插槽，使用之前定义的 pandas_parser_slots 数组
    .m_slots = pandas_parser_slots};



PyMODINIT_FUNC PyInit_pandas_parser(void) {
  return PyModuleDef_Init(&pandas_parsermodule);
}



# 定义 Python 初始化函数 PyInit_pandas_parser
PyMODINIT_FUNC PyInit_pandas_parser(void) {
    # 调用 PyModuleDef_Init 函数初始化并返回 pandas_parsermodule 模块
    return PyModuleDef_Init(&pandas_parsermodule);
}
```