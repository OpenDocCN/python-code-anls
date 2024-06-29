# `.\numpy\numpy\_core\code_generators\generate_ufunc_api.py`

```
# 导入标准库模块 os，用于操作操作系统相关功能
import os
# 导入 argparse 库，用于命令行参数解析
import argparse

# 导入 genapi 模块
import genapi
# 从 genapi 模块中导入 TypeApi 和 FunctionApi 类
from genapi import TypeApi, FunctionApi
# 导入 numpy_api 模块
import numpy_api

# 定义 C 语言头文件模板字符串
h_template = r"""
#ifdef _UMATHMODULE

extern NPY_NO_EXPORT PyTypeObject PyUFunc_Type;

%s

#else

#if defined(PY_UFUNC_UNIQUE_SYMBOL)
#define PyUFunc_API PY_UFUNC_UNIQUE_SYMBOL
#endif

/* By default do not export API in an .so (was never the case on windows) */
#ifndef NPY_API_SYMBOL_ATTRIBUTE
    #define NPY_API_SYMBOL_ATTRIBUTE NPY_VISIBILITY_HIDDEN
#endif

#if defined(NO_IMPORT) || defined(NO_IMPORT_UFUNC)
extern NPY_API_SYMBOL_ATTRIBUTE void **PyUFunc_API;
#else
#if defined(PY_UFUNC_UNIQUE_SYMBOL)
NPY_API_SYMBOL_ATTRIBUTE void **PyUFunc_API;
#else
static void **PyUFunc_API=NULL;
#endif
#endif

%s

static inline int
_import_umath(void)
{
  PyObject *numpy = PyImport_ImportModule("numpy._core._multiarray_umath");
  if (numpy == NULL && PyErr_ExceptionMatches(PyExc_ModuleNotFoundError)) {
    PyErr_Clear();
    numpy = PyImport_ImportModule("numpy._core._multiarray_umath");
    if (numpy == NULL && PyErr_ExceptionMatches(PyExc_ModuleNotFoundError)) {
      PyErr_Clear();
      numpy = PyImport_ImportModule("numpy.core._multiarray_umath");
    }
  }

  if (numpy == NULL) {
      PyErr_SetString(PyExc_ImportError,
                      "_multiarray_umath failed to import");
      return -1;
  }

  PyObject *c_api = PyObject_GetAttrString(numpy, "_UFUNC_API");
  Py_DECREF(numpy);
  if (c_api == NULL) {
      PyErr_SetString(PyExc_AttributeError, "_UFUNC_API not found");
      return -1;
  }

  if (!PyCapsule_CheckExact(c_api)) {
      PyErr_SetString(PyExc_RuntimeError, "_UFUNC_API is not PyCapsule object");
      Py_DECREF(c_api);
      return -1;
  }
  PyUFunc_API = (void **)PyCapsule_GetPointer(c_api, NULL);
  Py_DECREF(c_api);
  if (PyUFunc_API == NULL) {
      PyErr_SetString(PyExc_RuntimeError, "_UFUNC_API is NULL pointer");
      return -1;
  }
  return 0;
}

# 定义宏，用于导入 umath 模块，并处理异常
#define import_umath() \
    do {\
        UFUNC_NOFPE\
        if (_import_umath() < 0) {\
            PyErr_Print();\
            PyErr_SetString(PyExc_ImportError,\
                    "numpy._core.umath failed to import");\
            return NULL;\
        }\
    } while(0)

# 定义宏，用于导入 umath 模块，并处理异常，返回指定返回值
#define import_umath1(ret) \
    do {\
        UFUNC_NOFPE\
        if (_import_umath() < 0) {\
            PyErr_Print();\
            PyErr_SetString(PyExc_ImportError,\
                    "numpy._core.umath failed to import");\
            return ret;\
        }\
    } while(0)

# 定义宏，用于导入 umath 模块，并处理异常，返回指定返回值和自定义错误消息
#define import_umath2(ret, msg) \
    do {\
        UFUNC_NOFPE\
        if (_import_umath() < 0) {\
            PyErr_Print();\
            PyErr_SetString(PyExc_ImportError, msg);\
            return ret;\
        }\
    } while(0)

# 定义宏，用于导入 ufunc 模块，并处理异常
#define import_ufunc() \
    do {\
        UFUNC_NOFPE\
        if (_import_umath() < 0) {\
            PyErr_Print();\
            PyErr_SetString(PyExc_ImportError,\
                    "numpy._core.umath failed to import");\
        }\
    } while(0)


static inline int
PyUFunc_ImportUFuncAPI()
{
    # 如果 PyUFunc_API 指针为 NULL（不太可能的情况），则执行以下操作
    if (NPY_UNLIKELY(PyUFunc_API == NULL)) {
        # 调用 import_umath1 函数，参数为 -1，用于初始化 umath1 模块
        import_umath1(-1);
    }
    # 返回整数值 0，表示函数执行成功
    return 0;
# Python 脚本的结尾，标识代码块的结束
}

# C 代码的预处理指令，标识代码块的结束
#endif

# C 代码模板字符串，包含 C 语言的注释
c_template = r"""
/* These pointers will be stored in the C-object for use in other
    extension modules
*/

# 定义生成 API 的函数，接收输出目录和是否强制覆盖的标志
def generate_api(output_dir, force=False):
    # 基础文件名
    basename = 'ufunc_api'

    # 构造头文件和 C 文件的完整路径
    h_file = os.path.join(output_dir, '__%s.h' % basename)
    c_file = os.path.join(output_dir, '__%s.c' % basename)
    targets = (h_file, c_file)

    # 源文件列表
    sources = ['ufunc_api_order.txt']
    
    # 调用具体的 API 生成函数
    do_generate_api(targets, sources)
    
    # 返回生成的文件路径
    return targets

# 执行 API 生成的具体功能函数
def do_generate_api(targets, sources):
    # 解析目标文件路径
    header_file = targets[0]
    c_file = targets[1]

    # 合并并生成 UFunc API 的索引
    ufunc_api_index = genapi.merge_api_dicts((
            numpy_api.ufunc_funcs_api,
            numpy_api.ufunc_types_api))
    genapi.check_api_dict(ufunc_api_index)

    # 获取 UFunc API 函数列表
    ufunc_api_list = genapi.get_api_functions('UFUNC_API', numpy_api.ufunc_funcs_api)

    # 创建字典，名称映射到 FunctionApi 实例
    ufunc_api_dict = {}
    api_name = 'PyUFunc_API'
    for f in ufunc_api_list:
        name = f.name
        index = ufunc_api_index[name][0]
        annotations = ufunc_api_index[name][1:]
        ufunc_api_dict[name] = FunctionApi(f.name, index, annotations,
                                           f.return_type, f.args, api_name)

    # 处理 UFunc API 类型
    for name, val in numpy_api.ufunc_types_api.items():
        index = val[0]
        ufunc_api_dict[name] = TypeApi(name, index, 'PyTypeObject', api_name)

    # 设置对象 API
    module_list = []
    extension_list = []
    init_list = []

    # 遍历排序后的 API 索引，生成模块列表、扩展列表和初始化列表
    for name, index in genapi.order_dict(ufunc_api_index):
        api_item = ufunc_api_dict[name]

        # 对于 NumPy 2.0 中可能存在的 API 空洞，填充 NULL
        while len(init_list) < api_item.index:
            init_list.append("        NULL")

        extension_list.append(api_item.define_from_array_api_string())
        init_list.append(api_item.array_api_define())
        module_list.append(api_item.internal_define())

    # 写入头文件
    s = h_template % ('\n'.join(module_list), '\n'.join(extension_list))
    genapi.write_file(header_file, s)

    # 写入 C 代码文件
    s = c_template % ',\n'.join(init_list)
    genapi.write_file(c_file, s)

    # 返回生成的文件路径
    return targets

# 主函数，用于命令行参数解析和调用 API 生成函数
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--outdir",
        type=str,
        help="Path to the output directory"
    )
    args = parser.parse_args()

    # 绝对路径化输出目录
    outdir_abs = os.path.join(os.getcwd(), args.outdir)

    # 调用生成 API 函数
    generate_api(outdir_abs)

# 程序入口，判断是否为主程序并执行主函数
if __name__ == "__main__":
    main()
```