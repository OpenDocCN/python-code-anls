# `D:\src\scipysrc\pandas\pandas\_libs\src\datetime\pd_datetime.c`

```
/*
 *
 * This section includes necessary headers and defines for integrating
 * Pandas datetime functionalities with NumPy and Python datetime objects.
 */

#define _PANDAS_DATETIME_IMPL

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "datetime.h"
/* Need to import_array for np_datetime.c (for NumPy 1.x support only) */
#define PY_ARRAY_UNIQUE_SYMBOL PANDAS_DATETIME_NUMPY
#include "numpy/ndarrayobject.h"
#include "pandas/datetime/pd_datetime.h"
#include "pandas/portable.h"

/*
 * pandas_datetime_destructor: Frees memory associated with Pandas datetime objects
 *                            encapsulated in PyCapsules.
 */
static void pandas_datetime_destructor(PyObject *op) {
  void *ptr = PyCapsule_GetPointer(op, PandasDateTime_CAPSULE_NAME);
  PyMem_Free(ptr);
}

/*
 *
 * convert_pydatetime_to_datetimestruct: Converts Python datetime.datetime or datetime.date
 *                                       object into a NumPy npy_datetimestruct.
 *                                       Handles time zone conversion using tzinfo (if present).
 *
 * Arguments:
 *    dtobj: PyObject* - Python datetime object to convert
 *    out: npy_datetimestruct* - Output structure where the converted datetime will be stored
 *
 * Returns:
 *    -1 on error, 0 on success, 1 if obj lacks necessary date or datetime attributes.
 */
static int convert_pydatetime_to_datetimestruct(PyObject *dtobj,
                                                npy_datetimestruct *out) {
  // Assumes that obj is a valid datetime object
  PyObject *tmp;
  PyObject *obj = (PyObject *)dtobj;

  /* Initialize the output to all zeros */
  memset(out, 0, sizeof(npy_datetimestruct));
  out->month = 1;
  out->day = 1;

  out->year = PyLong_AsLong(PyObject_GetAttrString(obj, "year"));
  out->month = PyLong_AsLong(PyObject_GetAttrString(obj, "month"));
  out->day = PyLong_AsLong(PyObject_GetAttrString(obj, "day"));

  // Check if time attributes are present; if not, treat as date only
  if (!PyObject_HasAttrString(obj, "hour") ||
      !PyObject_HasAttrString(obj, "minute") ||
      !PyObject_HasAttrString(obj, "second") ||
      !PyObject_HasAttrString(obj, "microsecond")) {
    return 0;
  }

  out->hour = PyLong_AsLong(PyObject_GetAttrString(obj, "hour"));
  out->min = PyLong_AsLong(PyObject_GetAttrString(obj, "minute"));
  out->sec = PyLong_AsLong(PyObject_GetAttrString(obj, "second"));
  out->us = PyLong_AsLong(PyObject_GetAttrString(obj, "microsecond"));

  // Apply time zone offset if datetime object is timezone-aware
  if (PyObject_HasAttrString(obj, "tzinfo")) {
    PyObject *offset = extract_utc_offset(obj);
    /* Apply the time zone offset if datetime obj is tz-aware */
    // (Offset application code would go here if implemented)
  }
    // 检查偏移量是否不为空
    if (offset != NULL) {
      // 如果偏移量是 Py_None（Python 中的空值对象），则直接返回 0
      if (offset == Py_None) {
        Py_DECREF(offset);
        return 0;
      }
      // 定义临时变量和整型变量来存储秒和分钟偏移量
      PyObject *tmp;
      int seconds_offset, minutes_offset;
      /*
       * timedelta 对象应该有一个名为 "total_seconds" 的方法，
       * 该方法返回我们需要的秒数值。
       */
      // 调用 offset 对象的 "total_seconds" 方法，返回一个 PyObject 对象
      tmp = PyObject_CallMethod(offset, "total_seconds", "");
      // 减少 offset 的引用计数
      Py_DECREF(offset);
      // 如果调用方法失败，返回 -1 表示错误
      if (tmp == NULL) {
        return -1;
      }
      // 将返回的 PyObject 对象转换为长整型对象
      PyObject *tmp_int = PyNumber_Long(tmp);
      Py_DECREF(tmp);
      // 如果转换失败，返回 -1 表示错误
      if (tmp_int == NULL) {
        return -1;
      }
      // 将长整型对象转换为 C 的 int 类型，获取秒数偏移量
      seconds_offset = PyLong_AsLong(tmp_int);
      // 检查是否转换失败或者发生了异常
      if (seconds_offset == -1 && PyErr_Occurred()) {
        Py_DECREF(tmp_int);
        return -1;
      }
      // 减少长整型对象的引用计数
      Py_DECREF(tmp_int);

      /* 将秒数偏移量转换为分钟偏移量，并应用到输出的日期时间结构 */
      minutes_offset = seconds_offset / 60;

      // 将负的分钟偏移量加到输出的日期时间结构中
      add_minutes_to_datetimestruct(out, -minutes_offset);
    }
  }

  // 返回成功状态码 0
  return 0;
// 将 Python 对象表示的日期/时间转换为 ISO 格式，精度为指定的 `base`，例如 base="s" 对应 2020-01-03T00:00:00Z，
// base="ns" 对应 2020-01-01T00:00:00.000000000Z。`len` 被改变以保存返回字符串的长度
static char *PyDateTimeToIso(PyObject *obj, NPY_DATETIMEUNIT base,
                             size_t *len) {
  npy_datetimestruct dts;
  int ret;

  // 将 Python datetime 对象转换为 numpy 的 datetimestruct 结构
  ret = convert_pydatetime_to_datetimestruct(obj, &dts);
  if (ret != 0) {
    // 如果转换失败并且没有设置异常，则设置一个值错误异常
    if (!PyErr_Occurred()) {
      PyErr_SetString(PyExc_ValueError,
                      "Could not convert PyDateTime to numpy datetime");
    }
    return NULL;
  }

  // 计算 ISO 8601 格式字符串的长度，并分配内存空间
  *len = (size_t)get_datetime_iso_8601_strlen(0, base);
  char *result = PyObject_Malloc(*len);

  // 检查 PyDateTime 是否具有时区信息，若没有则不转换为 UTC
  int is_tz_aware = 0;
  if (PyObject_HasAttrString(obj, "tzinfo")) {
    // 提取 UTC 偏移量信息
    PyObject *offset = extract_utc_offset(obj);
    if (offset == NULL) {
      PyObject_Free(result);
      return NULL;
    }
    // 判断是否是时区感知的 datetime
    is_tz_aware = offset != Py_None;
    Py_DECREF(offset);
  }

  // 将 datetimestruct 结构转换为 ISO 8601 格式的 datetime 字符串
  ret = make_iso_8601_datetime(&dts, result, *len, is_tz_aware, base);

  if (ret != 0) {
    // 如果转换失败，则设置一个值错误异常并释放内存
    PyErr_SetString(PyExc_ValueError,
                    "Could not convert datetime value to string");
    PyObject_Free(result);
    return NULL;
  }

  // 注意，get_datetime_iso_8601_strlen 只提供 ISO 字符串转换的通用长度，不是实际使用的大小
  *len = strlen(result);
  return result;
}

// 将 Python 的日期/时间对象转换为 Unix epoch 时间，精度为指定的 `base`
static npy_datetime PyDateTimeToEpoch(PyObject *dt, NPY_DATETIMEUNIT base) {
  npy_datetimestruct dts;
  int ret;

  // 将 Python datetime 对象转换为 numpy 的 datetimestruct 结构
  ret = convert_pydatetime_to_datetimestruct(dt, &dts);
  if (ret != 0) {
    // 如果转换失败并且没有设置异常，则设置一个值错误异常
    if (!PyErr_Occurred()) {
      PyErr_SetString(PyExc_ValueError,
                      "Could not convert PyDateTime to numpy datetime");

      return -1;
    }
  }

  // 将 datetimestruct 结构转换为以纳秒为单位的 numpy datetime，然后缩放为指定的 `base`
  int64_t npy_dt = npy_datetimestruct_to_datetime(NPY_FR_ns, &dts);
  if (scaleNanosecToUnit(&npy_dt, base) == -1) {
    // 如果缩放操作失败，则设置一个值错误异常
    PyErr_Format(PyExc_ValueError,
                 "Call to scaleNanosecToUnit with value %" NPY_DATETIME_FMT
                 " and base %d failed",
                 npy_dt, base);

    return -1;
  }
  return npy_dt;
}
  return -1;

# 返回错误码 -1，表示函数执行失败


  }
  capi->npy_datetimestruct_to_datetime = npy_datetimestruct_to_datetime;
  capi->scaleNanosecToUnit = scaleNanosecToUnit;
  capi->int64ToIso = int64ToIso;
  capi->PyDateTimeToIso = PyDateTimeToIso;
  capi->PyDateTimeToEpoch = PyDateTimeToEpoch;
  capi->int64ToIsoDuration = int64ToIsoDuration;
  capi->pandas_datetime_to_datetimestruct = pandas_datetime_to_datetimestruct;
  capi->pandas_timedelta_to_timedeltastruct =
      pandas_timedelta_to_timedeltastruct;
  capi->convert_pydatetime_to_datetimestruct =
      convert_pydatetime_to_datetimestruct;
  capi->cmp_npy_datetimestruct = cmp_npy_datetimestruct;
  capi->get_datetime_metadata_from_dtype = get_datetime_metadata_from_dtype;
  capi->parse_iso_8601_datetime = parse_iso_8601_datetime;
  capi->get_datetime_iso_8601_strlen = get_datetime_iso_8601_strlen;
  capi->make_iso_8601_datetime = make_iso_8601_datetime;
  capi->make_iso_8601_timedelta = make_iso_8601_timedelta;

# 将各个函数指针赋值给 C-API 结构体中对应的函数指针成员


  PyObject *capsule = PyCapsule_New(capi, PandasDateTime_CAPSULE_NAME,
                                    pandas_datetime_destructor);
  if (capsule == NULL) {
    PyMem_Free(capi);
    return -1;
  }

# 创建一个 Python Capsule 封装 C-API 结构体，并指定析构函数为 pandas_datetime_destructor；如果创建失败，则释放内存并返回 -1


  // Monkeypatch the top level pandas module to have an attribute for the
  // C-API. This is required because Python capsules do not support setting
  // this attribute on anything but the top level package. Ideally not
  // done when cpython gh-6898 gets implemented
  PyObject *pandas = PyImport_ImportModule("pandas");
  if (!pandas) {
    PyErr_SetString(PyExc_ImportError,
                    "pd_datetime.c could not import module pandas");
    Py_DECREF(capsule);
    return -1;
  }

# 导入 pandas 模块，并检查导入是否成功；如果失败，则设置异常并清理之前创建的 capsule，并返回 -1


  if (PyModule_AddObject(pandas, "_pandas_datetime_CAPI", capsule) < 0) {
    Py_DECREF(capsule);
    return -1;
  }

# 将创建的 capsule 对象作为 "_pandas_datetime_CAPI" 属性添加到 pandas 模块中；如果添加失败，则清理 capsule 并返回 -1


  return 0;

# 函数执行成功，返回 0 表示成功添加 C-API 到 pandas 模块
# 定义 pandas_datetime_slots 数组，包含 PyModuleDef_Slot 结构
static PyModuleDef_Slot pandas_datetime_slots[] = {
    # 指定模块执行时调用的函数和标志
    {Py_mod_exec, pandas_datetime_exec}, {0, NULL}};

# 定义 pandas_datetimemodule 结构体，表示 Python 模块
static struct PyModuleDef pandas_datetimemodule = {
    # 使用 PyModuleDef_HEAD_INIT 初始化模块定义
    PyModuleDef_HEAD_INIT,
    # 模块的名称字符串
    .m_name = "pandas._libs.pandas_datetime",

    # 模块的文档字符串，描述模块用途
    .m_doc = "Internal module with datetime support for other extensions",
    # 模块的大小，一般为 0
    .m_size = 0,
    # 模块中定义的方法列表，此处为 NULL
    .m_methods = NULL,
    # 模块中定义的插槽列表，包含模块的执行和其他功能的定义
    .m_slots = pandas_datetime_slots};

# Python 初始化函数，命名为 PyInit_pandas_datetime
PyMODINIT_FUNC PyInit_pandas_datetime(void) {
  # 导入 PyDateTimeAPI 符号
  PyDateTime_IMPORT;
  # 导入 NumPy C API
  import_array();
  # 返回模块定义的初始化结果
  return PyModuleDef_Init(&pandas_datetimemodule);
}
```