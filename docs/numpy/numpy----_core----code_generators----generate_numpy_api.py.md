# `.\numpy\numpy\_core\code_generators\generate_numpy_api.py`

```
#!/usr/bin/env python3
# 导入标准库模块
import os
import argparse

# 导入自定义模块和部分子模块
import genapi
from genapi import \
        TypeApi, GlobalVarApi, FunctionApi, BoolValuesApi

# 导入numpy_api模块
import numpy_api

# use annotated api when running under cpychecker
h_template = r"""
#if defined(_MULTIARRAYMODULE) || defined(WITH_CPYCHECKER_STEALS_REFERENCE_TO_ARG_ATTRIBUTE)

typedef struct {
        PyObject_HEAD
        npy_bool obval;
} PyBoolScalarObject;

extern NPY_NO_EXPORT PyTypeObject PyArrayNeighborhoodIter_Type;
extern NPY_NO_EXPORT PyBoolScalarObject _PyArrayScalar_BoolValues[2];

%s

#else

#if defined(PY_ARRAY_UNIQUE_SYMBOL)
    #define PyArray_API PY_ARRAY_UNIQUE_SYMBOL
    #define _NPY_VERSION_CONCAT_HELPER2(x, y) x ## y
    #define _NPY_VERSION_CONCAT_HELPER(arg) \
        _NPY_VERSION_CONCAT_HELPER2(arg, PyArray_RUNTIME_VERSION)
    #define PyArray_RUNTIME_VERSION \
        _NPY_VERSION_CONCAT_HELPER(PY_ARRAY_UNIQUE_SYMBOL)
#endif

/* By default do not export API in an .so (was never the case on windows) */
#ifndef NPY_API_SYMBOL_ATTRIBUTE
    #define NPY_API_SYMBOL_ATTRIBUTE NPY_VISIBILITY_HIDDEN
#endif

#if defined(NO_IMPORT) || defined(NO_IMPORT_ARRAY)
extern NPY_API_SYMBOL_ATTRIBUTE void **PyArray_API;
extern NPY_API_SYMBOL_ATTRIBUTE int PyArray_RUNTIME_VERSION;
#else
#if defined(PY_ARRAY_UNIQUE_SYMBOL)
NPY_API_SYMBOL_ATTRIBUTE void **PyArray_API;
NPY_API_SYMBOL_ATTRIBUTE int PyArray_RUNTIME_VERSION;
#else
static void **PyArray_API = NULL;
static int PyArray_RUNTIME_VERSION = 0;
#endif
#endif

%s

/*
 * The DType classes are inconvenient for the Python generation so exposed
 * manually in the header below  (may be moved).
 */
#include "numpy/_public_dtype_api_table.h"

#if !defined(NO_IMPORT_ARRAY) && !defined(NO_IMPORT)
static int
_import_array(void)
{
  int st;
  PyObject *numpy = PyImport_ImportModule("numpy._core._multiarray_umath");
  if (numpy == NULL && PyErr_ExceptionMatches(PyExc_ModuleNotFoundError)) {
    PyErr_Clear();
    numpy = PyImport_ImportModule("numpy.core._multiarray_umath");
  }

  if (numpy == NULL) {
      return -1;
  }

  PyObject *c_api = PyObject_GetAttrString(numpy, "_ARRAY_API");
  Py_DECREF(numpy);
  if (c_api == NULL) {
      return -1;
  }

  if (!PyCapsule_CheckExact(c_api)) {
      PyErr_SetString(PyExc_RuntimeError, "_ARRAY_API is not PyCapsule object");
      Py_DECREF(c_api);
      return -1;
  }
  PyArray_API = (void **)PyCapsule_GetPointer(c_api, NULL);
  Py_DECREF(c_api);
  if (PyArray_API == NULL) {
      PyErr_SetString(PyExc_RuntimeError, "_ARRAY_API is NULL pointer");
      return -1;
  }

  /*
   * On exceedingly few platforms these sizes may not match, in which case
   * We do not support older NumPy versions at all.
   */
  if (sizeof(Py_ssize_t) != sizeof(Py_intptr_t) &&
        PyArray_RUNTIME_VERSION < NPY_2_0_API_VERSION) {
  /*
   * 报告运行时错误，指出模块是针对 NumPy 2.0 编译的，但在 NumPy 1.x 上运行。
   * 在 `sizeof(size_t) != sizeof(inptr_t)` 的特定平台上，不支持此功能。
   */
  PyErr_Format(PyExc_RuntimeError,
        "module compiled against NumPy 2.0 but running on NumPy 1.x. "
        "Unfortunately, this is not supported on niche platforms where "
        "`sizeof(size_t) != sizeof(inptr_t)`.");
  }
  /*
   * 执行 NumPy C API 版本的运行时检查。目前，NumPy 2.0 在实际上是 ABI 向后兼容的
   * （在公开的特征子集中）。
   */
  if (NPY_VERSION < PyArray_GetNDArrayCVersion()) {
      PyErr_Format(PyExc_RuntimeError, "module compiled against "\
             "ABI version 0x%%x but this version of numpy is 0x%%x", \
             (int) NPY_VERSION, (int) PyArray_GetNDArrayCVersion());
      return -1;
  }
  PyArray_RUNTIME_VERSION = (int)PyArray_GetNDArrayCFeatureVersion();
  /*
   * 执行 NumPy 运行时版本特性的检查，确保模块编译时与运行时的 C API 版本匹配。
   */
  if (NPY_FEATURE_VERSION > PyArray_RUNTIME_VERSION) {
      PyErr_Format(PyExc_RuntimeError,
             "module was compiled against NumPy C-API version 0x%%x "
             "(NumPy " NPY_FEATURE_VERSION_STRING ") "
             "but the running NumPy has C-API version 0x%%x. "
             "Check the section C-API incompatibility at the "
             "Troubleshooting ImportError section at "
             "https://numpy.org/devdocs/user/troubleshooting-importerror.html"
             "#c-api-incompatibility "
             "for indications on how to solve this problem.",
             (int)NPY_FEATURE_VERSION, PyArray_RUNTIME_VERSION);
      return -1;
  }

  /*
   * 执行运行时检查，确保模块的字节顺序与头文件（npy_endian.h）中设置的顺序一致，
   * 作为一种安全保障。
   */
  st = PyArray_GetEndianness();
  if (st == NPY_CPU_UNKNOWN_ENDIAN) {
      PyErr_SetString(PyExc_RuntimeError,
                      "FATAL: module compiled as unknown endian");
      return -1;
  }
#if NPY_BYTE_ORDER == NPY_BIG_ENDIAN
  // 如果模块编译为大端序，但在运行时检测到不同的字节序，抛出错误并返回-1
  if (st != NPY_CPU_BIG) {
      PyErr_SetString(PyExc_RuntimeError,
                      "FATAL: module compiled as big endian, but "
                      "detected different endianness at runtime");
      return -1;
  }
#elif NPY_BYTE_ORDER == NPY_LITTLE_ENDIAN
  // 如果模块编译为小端序，但在运行时检测到不同的字节序，抛出错误并返回-1
  if (st != NPY_CPU_LITTLE) {
      PyErr_SetString(PyExc_RuntimeError,
                      "FATAL: module compiled as little endian, but "
                      "detected different endianness at runtime");
      return -1;
  }
#endif

// 如果检测字节序没有问题，返回0表示成功
return 0;
}

#define import_array() { \
  // 调用 _import_array() 导入 numpy 扩展模块，若失败则打印错误信息并返回 NULL
  if (_import_array() < 0) { \
    PyErr_Print(); \
    PyErr_SetString( \
        PyExc_ImportError, \
        "numpy._core.multiarray failed to import" \
    ); \
    return NULL; \
  } \
}

#define import_array1(ret) { \
  // 调用 _import_array() 导入 numpy 扩展模块，若失败则打印错误信息并返回指定的 ret 值
  if (_import_array() < 0) { \
    PyErr_Print(); \
    PyErr_SetString( \
        PyExc_ImportError, \
        "numpy._core.multiarray failed to import" \
    ); \
    return ret; \
  } \
}

#define import_array2(msg, ret) { \
  // 调用 _import_array() 导入 numpy 扩展模块，若失败则打印自定义错误信息 msg 并返回指定的 ret 值
  if (_import_array() < 0) { \
    PyErr_Print(); \
    PyErr_SetString(PyExc_ImportError, msg); \
    return ret; \
  } \
}

#endif

#endif
    # 遍历 scalar_bool_values 字典中的每对键值对
    for name, val in scalar_bool_values.items():
        # 获取索引值
        index = val[0]
        # 将 BoolValuesApi 实例添加到 multiarray_api_dict 字典中
        multiarray_api_dict[name] = BoolValuesApi(name, index, api_name)

    # 遍历 types_api 字典中的每对键值对
    for name, val in types_api.items():
        # 获取索引值
        index = val[0]
        # 如果 val 的长度为 1，internal_type 被设为 None；否则，使用 val 的第二个元素作为 internal_type
        internal_type = None if len(val) == 1 else val[1]
        # 将 TypeApi 实例添加到 multiarray_api_dict 字典中
        multiarray_api_dict[name] = TypeApi(
            name, index, 'PyTypeObject', api_name, internal_type)

    # 检查 multiarray_api_dict 和 multiarray_api_index 字典的长度是否相等，若不等则抛出异常
    if len(multiarray_api_dict) != len(multiarray_api_index):
        # 获取两个字典的键集合，并计算其差异
        keys_dict = set(multiarray_api_dict.keys())
        keys_index = set(multiarray_api_index.keys())
        raise AssertionError(
            "Multiarray API size mismatch - "
            "index has extra keys {}, dict has extra keys {}"
            .format(keys_index - keys_dict, keys_dict - keys_index)
        )

    # 初始化一个空列表 extension_list
    extension_list = []
    # 遍历 multiarray_api_index 字典中的每对键值对，按照顺序生成 API 的定义字符串并添加到 extension_list 中
    for name, index in genapi.order_dict(multiarray_api_index):
        api_item = multiarray_api_dict[name]
        # 在 NumPy 2.0 中，API 可能存在空洞（后续可能填充），在这种情况下，添加 `NULL` 来填充
        while len(init_list) < api_item.index:
            init_list.append("        NULL")

        # 生成并添加 api_item 的数组 API 字符串定义到 extension_list 中
        extension_list.append(api_item.define_from_array_api_string())
        # 添加 api_item 的数组 API 定义到 init_list 中
        init_list.append(api_item.array_api_define())
        # 添加 api_item 的内部定义到 module_list 中
        module_list.append(api_item.internal_define())

    # 在 init_list 中添加足够数量的 `NULL` 来填充到 unused_index_max
    while len(init_list) <= unused_index_max:
        init_list.append("        NULL")

    # 根据 header 模板 h_template，生成包含 module_list 和 extension_list 的字符串 s
    s = h_template % ('\n'.join(module_list), '\n'.join(extension_list))
    # 将生成的字符串 s 写入 header_file 文件中
    genapi.write_file(header_file, s)

    # 根据 c-code 模板 c_template，生成以逗号分隔的 init_list 字符串 s
    s = c_template % ',\n'.join(init_list)
    # 将生成的字符串 s 写入 c_file 文件中
    genapi.write_file(c_file, s)

    # 返回 targets 变量
    return targets
# 主程序入口函数
def main():
    # 创建命令行参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加命令行参数选项 "-o" 或 "--outdir"，指定输出目录路径，类型为字符串
    parser.add_argument(
        "-o",
        "--outdir",
        type=str,
        help="Path to the output directory"
    )
    # 添加命令行参数选项 "-i" 或 "--ignore"，指定一个被忽略的输入，类型为字符串
    parser.add_argument(
        "-i",
        "--ignore",
        type=str,
        help="An ignored input - may be useful to add a "
             "dependency between custom targets"
    )
    # 解析命令行参数，将结果存储在args变量中
    args = parser.parse_args()

    # 获取当前工作目录，并与输出目录参数拼接，生成输出目录的绝对路径
    outdir_abs = os.path.join(os.getcwd(), args.outdir)

    # 调用函数生成 API，将输出目录的绝对路径作为参数传递
    generate_api(outdir_abs)


if __name__ == "__main__":
    # 如果当前脚本被直接运行，则调用主函数main()
    main()
```