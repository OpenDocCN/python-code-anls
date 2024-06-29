# `D:\src\scipysrc\pandas\pandas\_libs\src\vendored\ujson\python\ujson.c`

```
/*
Copyright (c) 2011-2013, ESN Social Software AB and Jonas Tarnstrom
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
* Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.
* Neither the name of the ESN Social Software AB nor the
names of its contributors may be used to endorse or promote products
derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL ESN SOCIAL SOFTWARE AB OR JONAS TARNSTROM BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF
THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


Portions of code from MODP_ASCII - Ascii transformations (upper/lower, etc)
https://github.com/client9/stringencoders
Copyright (c) 2007  Nick Galbreath -- nickg [at] modp [dot] com. All rights
reserved.

Numeric decoder derived from TCL library
https://www.opensource.apple.com/source/tcl/tcl-14/tcl/license.terms
* Copyright (c) 1988-1993 The Regents of the University of California.
* Copyright (c) 1994 Sun Microsystems, Inc.
*/

// Licence at LICENSES/ULTRAJSON_LICENSE

// 包含 Python 标准库的头文件
#define PY_SSIZE_T_CLEAN
#include <Python.h>
// 定义一个名为 UJSON_NUMPY 的符号，用于避免数组对象命名冲突
#define PY_ARRAY_UNIQUE_SYMBOL UJSON_NUMPY
#include "numpy/arrayobject.h"

// 声明函数 objToJSON 和 initObjToJSON
PyObject *objToJSON(PyObject *self, PyObject *args, PyObject *kwargs);
void *initObjToJSON(void);

// 声明函数 JSONToObj
PyObject *JSONToObj(PyObject *self, PyObject *args, PyObject *kwargs);

// 定义 ENCODER_HELP_TEXT 常量，提供 JSON 编码函数的帮助信息
#define ENCODER_HELP_TEXT                                                      \
  "Use ensure_ascii=false to output UTF-8. Pass in double_precision to "       \
  "alter the maximum digit precision of doubles. Set "                         \
  "encode_html_chars=True to encode < > & as unicode escape sequences."

// 定义 PyMethodDef 结构体数组 ujsonMethods，包含 ujson_dumps 方法的描述
static PyMethodDef ujsonMethods[] = {
    {"ujson_dumps", (PyCFunction)(void (*)(void))objToJSON,
     METH_VARARGS | METH_KEYWORDS,
     "Converts arbitrary object recursively into JSON. " ENCODER_HELP_TEXT},
    {"ujson_loads", (PyCFunction)(void (*)(void))JSONToObj,
     METH_VARARGS | METH_KEYWORDS,
     "Converts JSON as string to dict object structure. Use precise_float=True "
     "to use high precision float decoder."},
    {NULL, NULL, 0, NULL} /* Sentinel */



    # 第一行：定义一个包含函数名和函数指针的静态结构体，用于映射Python函数和C函数
    {"ujson_loads", (PyCFunction)(void (*)(void))JSONToObj,
     METH_VARARGS | METH_KEYWORDS,
     "Converts JSON as string to dict object structure. Use precise_float=True "
     "to use high precision float decoder."},
    # 第二行：定义一个空结构体，用作结构体数组的终止标志
    {NULL, NULL, 0, NULL} /* Sentinel */
};

// 定义模块状态结构体，包含了 pandas._libs.json 模块中的不同类型的 PyObject 指针
typedef struct {
  PyObject *type_decimal;
  PyObject *type_dataframe;
  PyObject *type_series;
  PyObject *type_index;
  PyObject *type_nat;
  PyObject *type_na;
} modulestate;

// 宏定义，用于获取模块状态结构体的指针
#define modulestate(o) ((modulestate *)PyModule_GetState(o))

// 声明模块遍历、清除和释放函数
static int module_traverse(PyObject *m, visitproc visit, void *arg);
static int module_clear(PyObject *m);
static void module_free(void *module);

// 初始化模块定义结构体
static struct PyModuleDef moduledef = {
  .m_base = PyModuleDef_HEAD_INIT,  // 使用 PyModuleDef_HEAD_INIT 宏初始化基本信息
  .m_name = "pandas._libs.json",    // 模块名称为 pandas._libs.json
  .m_methods = ujsonMethods,        // 使用 ujsonMethods 中定义的方法
  .m_size = sizeof(modulestate),    // 模块状态结构体的大小
  .m_traverse = module_traverse,    // 模块遍历函数
  .m_clear = module_clear,          // 模块清除函数
  .m_free = module_free             // 模块释放函数
};

#ifndef PYPY_VERSION
// 在 objToJSON.c 中使用的函数，用于判断对象是否是 decimal 类型
int object_is_decimal_type(PyObject *obj) {
  PyObject *module = PyState_FindModule(&moduledef); // 查找模块 pandas._libs.json
  if (module == NULL)
    return 0;
  modulestate *state = modulestate(module); // 获取模块状态结构体指针
  if (state == NULL)
    return 0;
  PyObject *type_decimal = state->type_decimal; // 获取 decimal 类型的 PyObject 指针
  if (type_decimal == NULL) {
    PyErr_Clear(); // 清除错误状态
    return 0;
  }
  int result = PyObject_IsInstance(obj, type_decimal); // 判断对象是否是 decimal 类型
  if (result == -1) {
    PyErr_Clear(); // 清除错误状态
    return 0;
  }
  return result;
}

// 判断对象是否是 dataframe 类型的函数
int object_is_dataframe_type(PyObject *obj) {
  PyObject *module = PyState_FindModule(&moduledef); // 查找模块 pandas._libs.json
  if (module == NULL)
    return 0;
  modulestate *state = modulestate(module); // 获取模块状态结构体指针
  if (state == NULL)
    return 0;
  PyObject *type_dataframe = state->type_dataframe; // 获取 dataframe 类型的 PyObject 指针
  if (type_dataframe == NULL) {
    PyErr_Clear(); // 清除错误状态
    return 0;
  }
  int result = PyObject_IsInstance(obj, type_dataframe); // 判断对象是否是 dataframe 类型
  if (result == -1) {
    PyErr_Clear(); // 清除错误状态
    return 0;
  }
  return result;
}

// 判断对象是否是 series 类型的函数
int object_is_series_type(PyObject *obj) {
  PyObject *module = PyState_FindModule(&moduledef); // 查找模块 pandas._libs.json
  if (module == NULL)
    return 0;
  modulestate *state = modulestate(module); // 获取模块状态结构体指针
  if (state == NULL)
    return 0;
  PyObject *type_series = state->type_series; // 获取 series 类型的 PyObject 指针
  if (type_series == NULL) {
    PyErr_Clear(); // 清除错误状态
    return 0;
  }
  int result = PyObject_IsInstance(obj, type_series); // 判断对象是否是 series 类型
  if (result == -1) {
    PyErr_Clear(); // 清除错误状态
    return 0;
  }
  return result;
}

// 判断对象是否是 index 类型的函数
int object_is_index_type(PyObject *obj) {
  PyObject *module = PyState_FindModule(&moduledef); // 查找模块 pandas._libs.json
  if (module == NULL)
    return 0;
  modulestate *state = modulestate(module); // 获取模块状态结构体指针
  if (state == NULL)
    return 0;
  PyObject *type_index = state->type_index; // 获取 index 类型的 PyObject 指针
  if (type_index == NULL) {
    PyErr_Clear(); // 清除错误状态
    return 0;
  }
  int result = PyObject_IsInstance(obj, type_index); // 判断对象是否是 index 类型
  if (result == -1) {
    PyErr_Clear(); // 清除错误状态
    return 0;
  }
  return result;
}

// 判断对象是否是 nat 类型的函数
int object_is_nat_type(PyObject *obj) {
  PyObject *module = PyState_FindModule(&moduledef); // 查找模块 pandas._libs.json
  if (module == NULL)
    return 0;
  modulestate *state = modulestate(module); // 获取模块状态结构体指针
  if (state == NULL)
    return 0;
  PyObject *type_nat = state->type_nat; // 获取 nat 类型的 PyObject 指针
  if (type_nat == NULL) {
    PyErr_Clear(); // 清除错误状态
    return 0;
  }
  int result = PyObject_IsInstance(obj, type_nat); // 判断对象是否是 nat 类型
  if (result == -1) {
    PyErr_Clear(); // 清除错误状态
    return 0;
  }
  return result;
}
    // 返回整数 0，表示没有找到合适的类型
    return 0;
  // 获取指向状态对象中的类型属性的指针
  PyObject *type_nat = state->type_nat;
  // 如果类型属性为空，则清除错误并返回 0
  if (type_nat == NULL) {
    PyErr_Clear();
    return 0;
  }
  // 检查给定的对象是否是指定类型的实例
  int result = PyObject_IsInstance(obj, type_nat);
  // 如果检查失败（返回值为 -1），则清除错误并返回 0
  if (result == -1) {
    PyErr_Clear();
    return 0;
  }
  // 返回检查结果，表示对象是否是指定类型的实例
  return result;
#else
/* Used in objToJSON.c */
int object_is_decimal_type(PyObject *obj) {
    // 导入名为 "decimal" 的 Python 模块
    PyObject *module = PyImport_ImportModule("decimal");
    if (module == NULL) {
        // 如果导入失败，则清除错误并返回 0
        PyErr_Clear();
        return 0;
    }
    // 从模块中获取名为 "Decimal" 的对象
    PyObject *type_decimal = PyObject_GetAttrString(module, "Decimal");
    if (type_decimal == NULL) {
        // 如果获取失败，则释放模块对象并清除错误，然后返回 0
        Py_DECREF(module);
        PyErr_Clear();
        return 0;
    }
    // 检查给定对象是否是 "Decimal" 类型的实例
    int result = PyObject_IsInstance(obj, type_decimal);
    if (result == -1) {
        // 如果检查过程中出错，则释放模块和类型对象，并清除错误，然后返回 0
        Py_DECREF(module);
        Py_DECREF(type_decimal);
        PyErr_Clear();
        return 0;
    }
    // 返回检查结果
    return result;
}

// 检查给定对象是否是 "DataFrame" 类型的实例
int object_is_dataframe_type(PyObject *obj) {
    // 导入名为 "pandas" 的 Python 模块
    PyObject *module = PyImport_ImportModule("pandas");
    if (module == NULL) {
        // 如果导入失败，则清除错误并返回 0
        PyErr_Clear();
        return 0;
    }
    // 从模块中获取名为 "DataFrame" 的对象
    PyObject *type_dataframe = PyObject_GetAttrString(module, "DataFrame");
    if (type_dataframe == NULL) {
        // 如果获取失败，则释放模块对象并清除错误，然后返回 0
        Py_DECREF(module);
        PyErr_Clear();
        return 0;
    }
    // 检查给定对象是否是 "DataFrame" 类型的实例
    int result = PyObject_IsInstance(obj, type_dataframe);
    if (result == -1) {
        // 如果检查过程中出错，则释放模块和类型对象，并清除错误，然后返回 0
        Py_DECREF(module);
        Py_DECREF(type_dataframe);
        PyErr_Clear();
        return 0;
    }
    // 返回检查结果
    return result;
}

// 检查给定对象是否是 "Series" 类型的实例
int object_is_series_type(PyObject *obj) {
    // 导入名为 "pandas" 的 Python 模块
    PyObject *module = PyImport_ImportModule("pandas");
    if (module == NULL) {
        // 如果导入失败，则清除错误并返回 0
        PyErr_Clear();
        return 0;
    }
    // 从模块中获取名为 "Series" 的对象
    PyObject *type_series = PyObject_GetAttrString(module, "Series");
    if (type_series == NULL) {
        // 如果获取失败，则释放模块对象并清除错误，然后返回 0
        Py_DECREF(module);
        PyErr_Clear();
        return 0;
    }
    // 检查给定对象是否是 "Series" 类型的实例
    int result = PyObject_IsInstance(obj, type_series);
    if (result == -1) {
        // 如果检查过程中出错，则释放模块和类型对象，并清除错误，然后返回 0
        Py_DECREF(module);
        Py_DECREF(type_series);
        PyErr_Clear();
        return 0;
    }
    // 返回检查结果
    return result;
}

// 检查给定对象是否是 "Index" 类型的实例
int object_is_index_type(PyObject *obj) {
    // 导入名为 "pandas" 的 Python 模块
    PyObject *module = PyImport_ImportModule("pandas");
    if (module == NULL) {
        // 如果导入失败，则清除错误并返回 0
        PyErr_Clear();
        return 0;
    }
    // 从模块中获取名为 "Index" 的对象
    PyObject *type_index = PyObject_GetAttrString(module, "Index");
    if (type_index == NULL) {
        // 如果获取失败，则释放模块对象并清除错误，然后返回 0
        Py_DECREF(module);
        PyErr_Clear();
        return 0;
    }
    // 检查给定对象是否是 "Index" 类型的实例
    int result = PyObject_IsInstance(obj, type_index);
    if (result == -1) {
        // 如果检查过程中出错，则释放模块和类型对象，并清除错误，然后返回 0
        Py_DECREF(module);
        Py_DECREF(type_index);
        PyErr_Clear();
        return 0;
    }
    // 返回检查结果
    return result;
}

// 检查给定对象是否是 "NaTType" 类型的实例
int object_is_nat_type(PyObject *obj) {
    // 导入名为 "pandas._libs.tslibs.nattype" 的 Python 模块
    PyObject *module = PyImport_ImportModule("pandas._libs.tslibs.nattype");
    if (module == NULL) {
        // 如果导入失败，则清除错误并返回 0
        PyErr_Clear();
        return 0;
    }
    // 从模块中获取名为 "NaTType" 的对象
    PyObject *type_nat = PyObject_GetAttrString(module, "NaTType");
    if (type_nat == NULL) {
        // 如果获取失败，则释放模块对象并清除错误，然后返回 0
        Py_DECREF(module);
        PyErr_Clear();
        return 0;
    }
    // 检查给定对象是否是 "NaTType" 类型的实例
    int result = PyObject_IsInstance(obj, type_nat);
    if (result == -1) {
        // 如果检查过程中出错，则释放模块和类型对象，并清除错误，然后返回 0
        Py_DECREF(module);
        Py_DECREF(type_nat);
        PyErr_Clear();
        return 0;
    }
    // 返回检查结果
    return result;
}
#endif
// 检查给定的 Python 对象是否为 pandas 库中的 NA 类型
int object_is_na_type(PyObject *obj) {
    // 导入 pandas._libs.missing 模块
    PyObject *module = PyImport_ImportModule("pandas._libs.missing");
    if (module == NULL) {
        PyErr_Clear();  // 清除异常状态
        return 0;  // 返回 0 表示导入失败
    }

    // 获取 NAType 对象
    PyObject *type_na = PyObject_GetAttrString(module, "NAType");
    if (type_na == NULL) {
        Py_DECREF(module);  // 释放模块对象的引用计数
        PyErr_Clear();  // 清除异常状态
        return 0;  // 返回 0 表示获取 NAType 失败
    }

    // 检查给定对象是否是 NAType 类的实例
    int result = PyObject_IsInstance(obj, type_na);
    if (result == -1) {
        Py_DECREF(module);  // 释放模块对象的引用计数
        Py_DECREF(type_na);  // 释放 NAType 对象的引用计数
        PyErr_Clear();  // 清除异常状态
        return 0;  // 返回 0 表示检查失败
    }

    return result;  // 返回检查结果（0 或 1）
}

#endif

// 遍历模块中的对象，确保递增引用它们
static int module_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(modulestate(m)->type_decimal);
    Py_VISIT(modulestate(m)->type_dataframe);
    Py_VISIT(modulestate(m)->type_series);
    Py_VISIT(modulestate(m)->type_index);
    Py_VISIT(modulestate(m)->type_nat);
    Py_VISIT(modulestate(m)->type_na);
    return 0;  // 返回 0 表示遍历完成
}

// 清理模块中的对象，释放它们的引用
static int module_clear(PyObject *m) {
    Py_CLEAR(modulestate(m)->type_decimal);
    Py_CLEAR(modulestate(m)->type_dataframe);
    Py_CLEAR(modulestate(m)->type_series);
    Py_CLEAR(modulestate(m)->type_index);
    Py_CLEAR(modulestate(m)->type_nat);
    Py_CLEAR(modulestate(m)->type_na);
    return 0;  // 返回 0 表示清理完成
}

// 释放模块对象，包括清理其中的对象引用
static void module_free(void *module) {
    module_clear((PyObject *)module);  // 调用清理函数释放模块对象
}

// Python 模块初始化函数
PyMODINIT_FUNC PyInit_json(void) {
    import_array() PyObject *module;

#ifndef PYPY_VERSION
    // 不支持 PyPy 版本的情况下直接返回已存在的模块
    if ((module = PyState_FindModule(&moduledef)) != NULL) {
        Py_INCREF(module);  // 增加模块对象的引用计数
        return module;  // 返回现有模块对象
    }
#endif

    // 创建新的 Python 模块对象
    module = PyModule_Create(&moduledef);
    if (module == NULL) {
        return NULL;  // 创建失败，返回空指针
    }

#ifndef PYPY_VERSION
    // 导入 decimal 模块并获取 Decimal 类型对象
    PyObject *mod_decimal = PyImport_ImportModule("decimal");
    if (mod_decimal) {
        PyObject *type_decimal = PyObject_GetAttrString(mod_decimal, "Decimal");
        assert(type_decimal != NULL);  // 断言确保获取成功
        modulestate(module)->type_decimal = type_decimal;  // 设置模块中的 Decimal 类型对象
        Py_DECREF(mod_decimal);  // 释放 decimal 模块对象的引用计数
    }

    // 导入 pandas 模块并获取 DataFrame、Series、Index 类型对象
    PyObject *mod_pandas = PyImport_ImportModule("pandas");
    if (mod_pandas) {
        PyObject *type_dataframe = PyObject_GetAttrString(mod_pandas, "DataFrame");
        assert(type_dataframe != NULL);  // 断言确保获取成功
        modulestate(module)->type_dataframe = type_dataframe;  // 设置模块中的 DataFrame 类型对象

        PyObject *type_series = PyObject_GetAttrString(mod_pandas, "Series");
        assert(type_series != NULL);  // 断言确保获取成功
        modulestate(module)->type_series = type_series;  // 设置模块中的 Series 类型对象

        PyObject *type_index = PyObject_GetAttrString(mod_pandas, "Index");
        assert(type_index != NULL);  // 断言确保获取成功
        modulestate(module)->type_index = type_index;  // 设置模块中的 Index 类型对象

        Py_DECREF(mod_pandas);  // 释放 pandas 模块对象的引用计数
    }

    // 导入 pandas._libs.tslibs.nattype 模块并获取 NaTType 类型对象
    PyObject *mod_nattype = PyImport_ImportModule("pandas._libs.tslibs.nattype");
    if (mod_nattype) {
        PyObject *type_nat = PyObject_GetAttrString(mod_nattype, "NaTType");
        assert(type_nat != NULL);  // 断言确保获取成功
        modulestate(module)->type_nat = type_nat;  // 设置模块中的 NaTType 类型对象

        Py_DECREF(mod_nattype);  // 释放 nattype 模块对象的引用计数
    }

    // 导入 pandas._libs.missing 模块并获取 NAType 类型对象
    PyObject *mod_natype = PyImport_ImportModule("pandas._libs.missing");
    if (mod_natype) {
        PyObject *type_na = PyObject_GetAttrString(mod_natype, "NAType");
        assert(type_na != NULL);  // 断言确保获取成功
        modulestate(module)->type_na = type_na;  // 设置模块中的 NAType 类型对象

        Py_DECREF(mod_natype);  // 释放 missing 模块对象的引用计数
    } else {
    PyErr_Clear();
  }


注释：


    # 清除当前的 Python 异常状态
    PyErr_Clear();
  }


这段代码片段看起来是在某种错误处理或异常处理的上下文中。`PyErr_Clear()` 是一个 Python C API 函数，用于清除当前的 Python 异常状态，以便在处理完异常后恢复正常的程序流程。
#endif

  /* 暂时不作为第三方库提供
     创建一个名为 JSONDecodeError 的新异常，基于 PyExc_ValueError
     JSONDecodeError = PyErr_NewException("ujson.JSONDecodeError",
                                         PyExc_ValueError, NULL);
     增加对 JSONDecodeError 的引用计数
     Py_XINCREF(JSONDecodeError);
     如果无法将 JSONDecodeError 添加到模块中，则清理所有相关资源并返回空指针
     if (PyModule_AddObject(module, "JSONDecodeError", JSONDecodeError) < 0)
     {
       减少对 JSONDecodeError 的引用计数
       Py_XDECREF(JSONDecodeError);
       清空 JSONDecodeError 对象
       Py_CLEAR(JSONDecodeError);
       释放模块对象
       Py_DECREF(module);
       返回空指针
       return NULL;
     }
  */

  // 返回已初始化的模块对象
  return module;
}
```