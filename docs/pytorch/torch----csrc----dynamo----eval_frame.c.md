# `.\pytorch\torch\csrc\dynamo\eval_frame.c`

```
#define PY_SSIZE_T_CLEAN
// 包含 Torch Dynamo 的缓存条目头文件
#include <torch/csrc/dynamo/cache_entry.h>
// 包含 Torch Dynamo 的 C++ 包装头文件
#include <torch/csrc/dynamo/cpp_shim.h>
// 包含 Torch Dynamo 的 CPython 定义头文件
#include <torch/csrc/dynamo/cpython_defs.h>
// 包含 Torch Dynamo 的调试宏定义头文件
#include <torch/csrc/dynamo/debug_macros.h>
// 包含 Torch Dynamo 的额外状态头文件
#include <torch/csrc/dynamo/extra_state.h>
// 包含 Python 兼容性工具函数头文件
#include <torch/csrc/utils/python_compat.h>
// 包含 Python 字节码操作头文件
#include <opcode.h>
// 包含标准布尔类型头文件
#include <stdbool.h>

// 错误钩子保护对象，默认为空
PyObject* guard_error_hook = NULL;
// 缓存查找分析器字符串常量
const char* cache_lookup_profiler_str = "TorchDynamo Cache Lookup";

// 全局变量，表示活动的 Dynamo 线程数，默认为 0
static int active_dynamo_threads = 0;

// 线程特定存储键值，用于评估帧回调函数
static Py_tss_t eval_frame_callback_key = Py_tss_NEEDS_INIT;

// 内联函数，获取评估帧回调函数对象
inline static PyObject* eval_frame_callback_get(void) {
  void* result = PyThread_tss_get(&eval_frame_callback_key);
  // 如果结果为空，则返回 Python 的 None 对象
  if (unlikely(result == NULL)) {
    return (PyObject*)Py_None;
  } else {
    // 否则返回结果对象转换为 PyObject*
    return (PyObject*)result;
  }
}

// 内联函数，设置评估帧回调函数对象
inline static void eval_frame_callback_set(PyObject* obj) {
  PyThread_tss_set(&eval_frame_callback_key, obj);
}

// 对于 Python 3.14 版本及以下的条件编译处理
#if !(IS_PYTHON_3_14_PLUS)

// 在混合核心和非核心构建时，CPython 的问题修复说明
// 在 3.12 版本及以上，需要取消定义 _PyGC_FINALIZED
// 参考 https://github.com/python/cpython/issues/105268
#if IS_PYTHON_3_12_PLUS
#undef _PyGC_FINALIZED
#endif

// 参考 https://bugs.python.org/issue35886
#if PY_VERSION_HEX >= 0x03080000
// 定义 Py_BUILD_CORE，包含内部的 pycore_pystate.h 头文件
#define Py_BUILD_CORE
#include <internal/pycore_pystate.h>

// 从 3.11 版本开始引入的头文件
#if IS_PYTHON_3_11_PLUS
#include <internal/pycore_frame.h>
#endif

// 取消定义 Py_BUILD_CORE
#undef Py_BUILD_CORE
#endif // PY_VERSION_HEX >= 0x03080000

// 所有的评估 API 在 3.11 版本上都有改变，因此需要根据情况选择使用哪一个
// 参考 https://docs.python.org/3/c-api/init.html#c._PyFrameEvalFunction
#if IS_PYTHON_3_11_PLUS
// 定义 THP_EVAL_API_FRAME_OBJECT 为 _PyInterpreterFrame 类型
#define THP_EVAL_API_FRAME_OBJECT _PyInterpreterFrame

// 定义 THPPyInterpreterFrame 结构体，包含一个指向 _PyInterpreterFrame 的指针
typedef struct THPPyInterpreterFrame {
  PyObject_HEAD
  _PyInterpreterFrame* frame; // 引用的指针
} THPPyInterpreterFrame;

// 创建 THPPyInterpreterFrame 结构体的新对象，传入 _PyInterpreterFrame 指针
THPPyInterpreterFrame* THPPyInterpreterFrame_New(_PyInterpreterFrame* frame);

// 定义获取 _PyInterpreterFrame 结构体中各属性的宏函数
#define DECLARE_PYOBJ_ATTR(name) \
static PyObject* THPPyInterpreterFrame_##name(THPPyInterpreterFrame* self, PyObject* _noargs) { \
  PyObject* res = (PyObject*)self->frame->name; \
  Py_XINCREF(res); \
  return res; \
}

// 对于 3.12 版本及以上，定义获取 f_funcobj 属性的宏函数
#if IS_PYTHON_3_12_PLUS
DECLARE_PYOBJ_ATTR(f_funcobj)
#else
// 对于 3.12 版本以下，定义获取 f_func 属性的宏函数
DECLARE_PYOBJ_ATTR(f_func)
#endif

// 定义获取 f_globals 属性的宏函数
DECLARE_PYOBJ_ATTR(f_globals)
// 定义获取 f_builtins 属性的宏函数
DECLARE_PYOBJ_ATTR(f_builtins)
// 定义获取 f_locals 属性的宏函数
DECLARE_PYOBJ_ATTR(f_locals)

// 对于 3.13 版本及以上，定义获取 f_executable 属性的宏函数
#if IS_PYTHON_3_13_PLUS
DECLARE_PYOBJ_ATTR(f_executable)
#else
// 对于 3.13 版本以下，定义获取 f_code 属性的宏函数
DECLARE_PYOBJ_ATTR(f_code)
#endif

// 定义获取 frame_obj 属性的宏函数
DECLARE_PYOBJ_ATTR(frame_obj)

#undef DECLARE_PYOBJ_ATTR

// 定义获取上一个帧的宏函数，创建新的 THPPyInterpreterFrame 结构体对象
static THPPyInterpreterFrame* THPPyInterpreterFrame_previous(THPPyInterpreterFrame* self, PyObject* _noargs) {
  THPPyInterpreterFrame* res = THPPyInterpreterFrame_New(self->frame->previous);
  return res;
}
#endif // IS_PYTHON_3_11_PLUS
#endif // !(IS_PYTHON_3_14_PLUS)
// 返回当前帧的指令索引（即最后执行的指令的位置）
static PyObject* THPPyInterpreterFrame_f_lasti(THPPyInterpreterFrame* self, PyObject* _noargs) {
  return PyLong_FromLong(_PyInterpreterFrame_LASTI(self->frame));
}

// 返回当前帧的行号
static PyObject* THPPyInterpreterFrame_f_lineno(THPPyInterpreterFrame* self, PyObject* _noargs) {
  if (!self->frame->frame_obj) {
    // 如果帧对象不存在，返回该帧所属代码对象的第一行行号
    return PyLong_FromLong(F_CODE(self->frame)->co_firstlineno);
  }
  // 否则，获取帧对象的当前行号
  int lineno = PyFrame_GetLineNumber(self->frame->frame_obj);
  if (lineno < 0) {
    // 如果行号小于0，表示未找到有效的行号，返回None
    Py_RETURN_NONE;
  }
  // 返回当前行号
  return PyLong_FromLong(lineno);
}

// 返回当前帧的上一帧对象
static PyObject* THPPyInterpreterFrame_f_back(THPPyInterpreterFrame* self, PyObject* _noargs) {
  if (!self->frame->frame_obj) {
    // 如果帧对象不存在，返回None
    Py_RETURN_NONE;
  }
  // 否则，返回帧对象的上一帧对象
  return (PyObject*)PyFrame_GetBack(self->frame->frame_obj);
}

// 定义 THPPyInterpreterFrame 的属性数组
// NOLINTNEXTLINE 指示代码检查工具跳过下一行的特定检查项
static struct PyGetSetDef THPPyInterpreterFrame_properties[] = {
#if IS_PYTHON_3_12_PLUS
    {"f_func", (getter)THPPyInterpreterFrame_f_funcobj, NULL, NULL, NULL},  // Python 3.12 及以上版本使用 f_funcobj
#else
    {"f_func", (getter)THPPyInterpreterFrame_f_func, NULL, NULL, NULL},     // Python 3.12 以下版本使用 f_func
#endif
    {"f_globals", (getter)THPPyInterpreterFrame_f_globals, NULL, NULL, NULL},       // 获取全局变量字典
    {"f_builtins", (getter)THPPyInterpreterFrame_f_builtins, NULL, NULL, NULL},     // 获取内建变量字典
    {"f_locals", (getter)THPPyInterpreterFrame_f_locals, NULL, NULL, NULL},         // 获取局部变量字典
#if IS_PYTHON_3_13_PLUS
    {"f_code", (getter)THPPyInterpreterFrame_f_executable, NULL, NULL, NULL},       // Python 3.13 及以上版本使用 f_executable
#else
    {"f_code", (getter)THPPyInterpreterFrame_f_code, NULL, NULL, NULL},             // Python 3.13 以下版本使用 f_code
#endif
    {"frame_obj", (getter)THPPyInterpreterFrame_frame_obj, NULL, NULL, NULL},        // 获取帧对象
    {"previous", (getter)THPPyInterpreterFrame_previous, NULL, NULL, NULL},          // 获取前一个帧对象
    {"f_lasti", (getter)THPPyInterpreterFrame_f_lasti, NULL, NULL, NULL},            // 获取最后的指令索引
    {"f_lineno", (getter)THPPyInterpreterFrame_f_lineno, NULL, NULL, NULL},          // 获取当前行号
    {"f_back", (getter)THPPyInterpreterFrame_f_back, NULL, NULL, NULL},              // 获取上一帧对象
    {NULL}};  // 结束属性定义

// 定义 THPPyInterpreterFrameType 类型对象
static PyTypeObject THPPyInterpreterFrameType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "torch._C.dynamo.eval_frame._PyInterpreterFrame",     // 类型对象名称
    .tp_basicsize = sizeof(THPPyInterpreterFrame),                   // 类型对象基本大小
    .tp_flags = Py_TPFLAGS_DEFAULT,                                  // 默认标志
    .tp_getset = THPPyInterpreterFrame_properties,                   // 属性数组
};

// 创建并返回一个新的 THPPyInterpreterFrame 对象
THPPyInterpreterFrame* THPPyInterpreterFrame_New(_PyInterpreterFrame* frame) {
  PyTypeObject* type = (PyTypeObject*)&THPPyInterpreterFrameType;
  THPPyInterpreterFrame* self = (THPPyInterpreterFrame*)type->tp_alloc(type, 0);
  if (!self)
    return NULL;
  self->frame = frame;
  return self;
}

#else
#define THP_EVAL_API_FRAME_OBJECT PyFrameObject

// 定义 THP_EVAL_API_FRAME_OBJECT 为 PyFrameObject 类型

static int
// 将快速帧转换为局部变量，并在出错时提供错误信息
THP_PyFrame_FastToLocalsWithError(THP_EVAL_API_FRAME_OBJECT *frame, int *free_vars_copied) {
  return PyFrame_FastToLocalsWithError(frame);
}
#endif

static PyObject* _custom_eval_frame_shim(
    PyThreadState* tstate,
    THP_EVAL_API_FRAME_OBJECT* frame,
    int throw_flag);
static PyObject* _custom_eval_frame(
    PyThreadState* tstate,
    THP_EVAL_API_FRAME_OBJECT* frame,
    int throw_flag,
    PyObject* callback,
    int* should_clear_frame);


注释：


    // 声明一个名为 should_clear_frame 的指针变量，指向 int 类型的数据
static PyObject *(*previous_eval_frame)(PyThreadState *tstate,
                                        THP_EVAL_API_FRAME_OBJECT* frame, int throw_flag) = NULL;
// 声明一个函数指针，用于保存之前的评估帧函数指针，参数为线程状态、评估帧对象和抛出标志

#if PY_VERSION_HEX >= 0x03090000
// 如果 Python 版本大于等于 3.9
static PyObject* custom_eval_frame_shim(
    PyThreadState* tstate,
    THP_EVAL_API_FRAME_OBJECT* frame,
    int throw_flag) {
  return _custom_eval_frame_shim(tstate, frame, throw_flag);
}
#else
// 如果 Python 版本小于 3.9
static PyObject* custom_eval_frame_shim(THP_EVAL_API_FRAME_OBJECT* frame, int throw_flag) {
  PyThreadState* tstate = PyThreadState_GET();
  return _custom_eval_frame_shim(tstate, frame, throw_flag);
}
#endif

inline static PyObject* eval_frame_default(
    PyThreadState* tstate,
    THP_EVAL_API_FRAME_OBJECT* frame,
    int throw_flag) {
#if PY_VERSION_HEX >= 0x03090000
  if (tstate == NULL) {
    tstate = PyThreadState_GET();
  }
  // 检查是否存在之前的评估帧函数，如果存在则调用它，否则调用默认的评估帧函数
  if (previous_eval_frame) {
    return previous_eval_frame(tstate, frame, throw_flag);
  }
  else {
    return _PyEval_EvalFrameDefault(tstate, frame, throw_flag);
  }
#else
  // 调用默认的评估帧函数
  return _PyEval_EvalFrameDefault(frame, throw_flag);
#endif
}

inline static void enable_eval_frame_shim(PyThreadState* tstate) {
#if PY_VERSION_HEX >= 0x03090000
  // 如果当前解释器状态中的评估帧函数不是自定义的评估帧函数，则进行替换
  if (_PyInterpreterState_GetEvalFrameFunc(tstate->interp) !=
      &custom_eval_frame_shim) {
    DEBUG_CHECK(previous_eval_frame == NULL);
    // 保存当前的评估帧函数指针并设置新的评估帧函数
    previous_eval_frame = _PyInterpreterState_GetEvalFrameFunc(tstate->interp);
    _PyInterpreterState_SetEvalFrameFunc(tstate->interp,
                                         &custom_eval_frame_shim);
  }
#else
  // 如果当前解释器状态中的评估帧函数不是自定义的评估帧函数，则进行替换
  if (tstate->interp->eval_frame != &custom_eval_frame_shim) {
    // 第一次调用时设置自定义评估帧函数
    tstate->interp->eval_frame = &custom_eval_frame_shim;
  }
#endif
}

inline static void enable_eval_frame_default(PyThreadState* tstate) {
#if PY_VERSION_HEX >= 0x03090000
  // 如果当前解释器状态中的评估帧函数不是之前保存的评估帧函数，则进行替换
  if (_PyInterpreterState_GetEvalFrameFunc(tstate->interp) !=
      previous_eval_frame) {
    DEBUG_CHECK(previous_eval_frame != NULL);
    // 恢复之前保存的评估帧函数
    _PyInterpreterState_SetEvalFrameFunc(tstate->interp,
                                         previous_eval_frame);
    previous_eval_frame = NULL;
  }
#else
  // 如果当前解释器状态中的评估帧函数不是默认的评估帧函数，则进行替换
  if (tstate->interp->eval_frame != &_PyEval_EvalFrameDefault) {
    // 第一次调用时设置默认评估帧函数
    tstate->interp->eval_frame = &_PyEval_EvalFrameDefault;
  }
#endif
}

inline static const char* get_frame_name(THP_EVAL_API_FRAME_OBJECT* frame) {
  // 返回当前帧的 C 字符串名称
  DEBUG_CHECK(PyUnicode_Check(F_CODE(frame)->co_name));
  return PyUnicode_AsUTF8(F_CODE(frame)->co_name);
}

static inline PyObject* call_callback(
    PyObject* callable,
    THP_EVAL_API_FRAME_OBJECT* _frame,
    CacheEntry* cache_entry,
    FrameState* frame_state) {
// 记得更新 torch/_dynamo/types.py 中 DynamoCallbackFn.__call__ 的类型签名，如果此函数发生变化

#if IS_PYTHON_3_11_PLUS
  // 创建一个新的 Python 解释器帧对象
  THPPyInterpreterFrame* frame = THPPyInterpreterFrame_New(_frame);
  if (frame == NULL) {
    return NULL;
  }
#else
  // 在 Python 3.11 之前，直接使用给定的评估帧对象作为帧
  PyObject* frame = Py_NewRef(_frame);
#endif
#endif

// 将 CacheEntry 结构体转换为 Python 对象
PyObject* cache_entry_pyobj = CacheEntry_to_obj(cache_entry);
// 调用 Python 可调用对象，传递三个参数：frame、cache_entry_pyobj 和 frame_state
PyObject* res = PyObject_CallFunction(
  callable,
  "OOO",
  frame,
  cache_entry_pyobj,
  frame_state);
// 减少 frame 对象的引用计数
Py_DECREF(frame);
// 减少 cache_entry_pyobj 对象的引用计数
Py_DECREF(cache_entry_pyobj);
// 返回 PyObject 调用的结果
return res;
}

// 如果 Python 版本是 3.12 及以上，则清除旧帧
static inline void clear_old_frame_if_python_312_plus(
  PyThreadState* tstate,
  THP_EVAL_API_FRAME_OBJECT* frame) {
#if IS_PYTHON_3_12_PLUS

// 清除 THP_EVAL_API_FRAME_OBJECT 对象的内容
THP_PyFrame_Clear(frame);
// 从线程状态中移除指定帧
THP_PyThreadState_PopFrame(tstate, frame);

#endif
}

// 实现自定义代码评估的内联函数
inline static PyObject* eval_custom_code_impl(
    PyThreadState* tstate,
    THP_EVAL_API_FRAME_OBJECT* frame,
    PyCodeObject* code,
    int throw_flag,
    int free_vars_copied) {

  // 断言 tstate 不为空
  DEBUG_NULL_CHECK(tstate);
  // 断言 frame 不为空
  DEBUG_NULL_CHECK(frame);
  // 断言 code 不为空
  DEBUG_NULL_CHECK(code);

#if IS_PYTHON_3_11_PLUS

  // 生成类似于 CPython ceval.c 中的 Python 函数对象和 _PyInterpreterFrame 的方法
#if IS_PYTHON_3_12_PLUS
  // 获取旧函数对象和代码的帧大小
  PyFunctionObject* old_func = (PyFunctionObject*) frame->f_funcobj;
  size_t size = code->co_framesize;
#else
  // 获取旧函数对象和代码的局部变量数、堆栈大小和特殊帧大小
  PyFunctionObject* old_func = frame->f_func;
  size_t size = code->co_nlocalsplus + code->co_stacksize + FRAME_SPECIALS_SIZE;
#endif

  // 使用新的代码对象复制函数对象
  PyFunctionObject* func = _PyFunction_CopyWithNewCode(old_func, code);
  if (func == NULL) {
    return NULL;
  }

  // 通过调用 THP_PyThreadState_BumpFramePointerSlow 在线程状态中创建新帧
  THP_EVAL_API_FRAME_OBJECT* shadow = THP_PyThreadState_BumpFramePointerSlow(tstate, size);
  if (shadow == NULL) {
    Py_DECREF(func);
    return NULL;
  }

  // 增加函数对象的引用计数
  Py_INCREF(func);
  // 初始化新帧，消耗 func 的引用
#if IS_PYTHON_3_12_PLUS
  _PyFrame_Initialize(shadow, func, NULL, code, 0);
#else
  _PyFrame_InitializeSpecials(shadow, func, NULL, code->co_nlocalsplus);
#endif

  // 获取旧帧的局部变量和新帧的局部变量
  PyObject** fastlocals_old = frame->localsplus;
  PyObject** fastlocals_new = shadow->localsplus;
  // 获取旧帧和新帧的局部变量数
  Py_ssize_t n_old = F_CODE(frame)->co_nlocalsplus;
  Py_ssize_t n_new = code->co_nlocalsplus;

  // 对于 3.11+ 版本，如果 free_vars_copied 为真，则无需运行第一个 COPY_FREE_VARS
  if (free_vars_copied && _Py_OPCODE(_PyCode_CODE(F_CODE(shadow))[0]) == COPY_FREE_VARS) {
    // 设置 PREV_INSTR(shadow) 为 _PyCode_CODE(F_CODE(shadow)) 的值
    PREV_INSTR(shadow) = _PyCode_CODE(F_CODE(shadow));
  }

#else

  // 如果 Python 版本不是 3.11+，则通过 PyFrame_New 创建新的帧对象
  THP_EVAL_API_FRAME_OBJECT* shadow = PyFrame_New(tstate, code, frame->f_globals, NULL);
  if (shadow == NULL) {
    return NULL;
  }

  // 获取旧帧和新帧的局部变量
  PyObject** fastlocals_old = frame->f_localsplus;
  PyObject** fastlocals_new = shadow->f_localsplus;
  // 获取旧帧和新帧的局部变量数
  Py_ssize_t n_old = F_CODE(frame)->co_nlocals + PyCode_GetNFreevars(F_CODE(frame)) + PyCode_GetNCellvars(F_CODE(frame));
  Py_ssize_t n_new = code->co_nlocals + PyCode_GetNFreevars(code) + PyCode_GetNCellvars(code);

  // 递增 fastlocals_old[i] 的引用计数
  Py_XINCREF(fastlocals_old[i]);

#endif
    // 将旧的 fastlocals 复制到新的 fastlocals
    fastlocals_new[i] = fastlocals_old[i];
  }

  // 复制自由变量
  Py_ssize_t nfrees_old = PyCode_GetNFreevars(F_CODE(frame));

  for (Py_ssize_t i = 0; i < nfrees_old; i++) {
    // 增加对旧 fastlocals 的引用计数
    Py_XINCREF(fastlocals_old[n_old - 1 - i]);
    // 将旧的 fastlocals 中的自由变量复制到新的 fastlocals
    fastlocals_new[n_new - 1 - i] = fastlocals_old[n_old - 1 - i];
  }

  // 复制闭包变量，从高索引到低索引，直到遇到不是闭包变量的变量为止。
  for (Py_ssize_t i = n_old - nfrees_old - 1, j = n_new - nfrees_old - 1; i >= total_argcount_old; i--, j--) {

    // 条件判断，用于判断一个变量是否不是闭包变量
    // 在 Python 3.11 及更高版本中，使用 `co_localspluskinds` 中的位标志直接判断变量是否是闭包变量。
    // 在 Python 3.10 及更低版本中，基本上是检查变量是否是新的局部变量（由于上述布局，第一个不是闭包变量的变量就是第一个新的局部变量）。在 `flocalsplus` 中，新的局部变量对应的槽位是 NULL。
#if IS_PYTHON_3_11_PLUS
    // 如果不是快速局部变量（非 CO_FAST_CELL），跳出循环
    if(!(_PyLocals_GetKind(F_CODE(frame)->co_localspluskinds, i) & CO_FAST_CELL))
    {
      break;
    }
#else
    // 如果旧的快速局部变量为 NULL，跳出循环
    if(fastlocals_old[i] == NULL)
    {
      break;
    }
#endif

    // 增加快速局部变量的引用计数
    Py_XINCREF(fastlocals_old[i]);
    // 将旧的快速局部变量赋值给新的快速局部变量数组
    fastlocals_new[j] = fastlocals_old[i];
  }

  // 注意：如果想在 3.12+ 版本中评估原始帧而不是影子帧，
  // 需要在调用 eval_frame_default 之前清除原始帧的影子，并注释掉
  // 原始帧上的 clear_old_frame_if_python_312_plus 调用。

  // 调用默认的帧评估函数，返回结果
  PyObject* result = eval_frame_default(tstate, shadow, throw_flag);

#if IS_PYTHON_3_12_PLUS

  // 在 3.12+ 中，调用者负责清除帧
  Py_DECREF(func);

#elif IS_PYTHON_3_11_PLUS

  // 在 3.11 中，影子帧的 is_entry 设置为 true，因此不会调用 _PyEvalFrameClearAndPop，
  // 因此我们手动清除并弹出影子帧。
  THP_PyFrame_Clear(shadow);
  THP_PyThreadState_PopFrame(tstate, shadow);
  Py_DECREF(func);

#else

  // 释放影子帧对象的引用计数
  Py_DECREF(shadow);

#endif

  // 返回评估结果
  return result;
}

// 这个包装函数添加了一个分析器事件
inline static PyObject* eval_custom_code(
    PyThreadState* tstate,
    THP_EVAL_API_FRAME_OBJECT* frame,
    PyCodeObject* code,
    int throw_flag,
    int free_vars_copied) {
  // 进入记录函数调用状态
  _PytorchRecordFunctionState* rf = _pytorch_record_function_enter("Torch-Compiled Region");
  // 调用实现函数来评估自定义代码
  PyObject* result = eval_custom_code_impl(
    tstate,
    frame,
    code,
    throw_flag,
    free_vars_copied
  );
  // 退出记录函数调用状态
  _pytorch_record_function_exit(rf);
  // 返回评估结果
  return result;
}

// 这个函数将逻辑嵌入到三种状态之一。后续可能可以重构为一个函数：
//  - None: 禁用 TorchDynamo
//  - False: 仅运行模式（重用现有编译）
//  - Python callable(): 启用 TorchDynamo
static PyObject* _custom_eval_frame_shim(
    PyThreadState* tstate,
    THP_EVAL_API_FRAME_OBJECT* frame,
    int throw_flag) {
  // 获取帧评估回调函数
  PyObject* callback = eval_frame_callback_get();

  // 如果回调函数为 Py_None，返回默认的帧评估结果
  if (callback == Py_None) {
    return eval_frame_default(tstate, frame, throw_flag);
  }

  // 初始化是否清除帧的标志
  int should_clear_frame = 0;
  // 调用自定义帧评估函数，并获取评估结果
  PyObject* result = _custom_eval_frame(tstate, frame, throw_flag, callback, &should_clear_frame);
  // 如果需要清除帧，则调用清除函数
  if (should_clear_frame) {
    clear_old_frame_if_python_312_plus(tstate, frame);
  }
  // 返回评估结果
  return result;
}

// 注意：在 3.12+ 中，帧评估函数（被调用者）负责清除/弹出帧，
// 这意味着除非我们默认评估原始帧，否则我们需要清除它 - 通过 clear_old_frame_if_python_312_plus。
// should_clear_frame 标志用于指示评估帧的调用者是否应该清除帧。
static PyObject* _custom_eval_frame(
    PyThreadState* tstate,
    THP_EVAL_API_FRAME_OBJECT* frame,
    int throw_flag,
    PyObject* callback,
    int* should_clear_frame) {
#if IS_PYTHON_3_11_PLUS
  // 如果是 Python 3.11 及以上版本，调试追踪信息包括函数名、文件名、起始行号和最后执行指令位置
  DEBUG_TRACE(
      "begin %s %s %i %i",
      get_frame_name(frame),
      PyUnicode_AsUTF8(F_CODE(frame)->co_filename),
      F_CODE(frame)->co_firstlineno,
      _PyInterpreterFrame_LASTI(frame));
#else
  // 如果是早于 Python 3.11 版本，调试追踪信息包括函数名、文件名、当前行号、最后执行指令位置和块索引
  DEBUG_TRACE(
      "begin %s %s %i %i %i",
      get_frame_name(frame),
      PyUnicode_AsUTF8(F_CODE(frame)->co_filename),
      frame->f_lineno,
      frame->f_lasti,
      frame->f_iblock);
#endif

// 如果 throw_flag 为真，则表示正在展开生成器，此时需要继续执行帧评估以传播异常。
if (throw_flag) {
    // 当 throw_flag 为真时，帧评估应继续展开以传播异常。 Dynamo 实际上不知道如何处理这种情况，
    // 也不希望这么做，因为不太可能捕获任何代码（你将立即退出帧，可能会沿途运行一些展开逻辑）。
    // 因此，在这种情况下，我们只运行默认处理程序。
    //
    // 注意：此补丁的早期版本返回 NULL。这是错误的，因为返回 NULL 与展开异常是不同的。
    // 特别是，如果只返回 NULL，将不会执行诸如上下文管理器 __exit__ 等操作。
    //
    // 注意：实际上，当 throw_flag == TRUE 时，你可能仍希望调用 Dynamo 回调，以便 Dynamo
    // 有机会执行任何堆栈展开代码。但这实际上并不有用，因为 (1) Dynamo 实际上不知道如何进行
    // 堆栈展开，所以它将立即跳过帧；(2) 即使知道，只有在展开代码中存在张量代码时才会有利可图。看起来是不太可能的。
    DEBUG_TRACE("throw %s", get_frame_name(frame));
    return eval_frame_default(tstate, frame, throw_flag);
  }

  // 获取帧的额外状态
  ExtraState* extra = get_extra_state(F_CODE(frame));

  // 如果 extra 为 SKIP_CODE，或者 callback 是 Py_False 且 extra 是 NULL，则跳过该帧
  if (extra == SKIP_CODE || (callback == Py_False && extra == NULL)) {
    DEBUG_TRACE("skip %s", get_frame_name(frame));
    return eval_frame_default(tstate, frame, throw_flag);
  }

  // 如果 extra 是 NULL，则初始化并设置帧的额外状态
  if (extra == NULL) {
    extra = init_and_set_extra_state(F_CODE(frame));
  }

  // 尝试将帧的自由变量快速复制到局部变量中，如果失败则返回 NULL
  int free_vars_copied = 0;
  if (THP_PyFrame_FastToLocalsWithError(frame, &free_vars_copied) < 0) {
    DEBUG_TRACE("error %s", get_frame_name(frame));
    *should_clear_frame = 1;
    return NULL;
  }

  // 获取回调函数对应的后端对象
  PyObject* backend = get_backend(callback);

  // 如果 callback 是 Py_False，表示“仅运行”模式，会检查缓存但不会编译
  if (callback == Py_False) {
    DEBUG_TRACE("In run only mode %s", get_frame_name(frame));
    // 进入记录函数调用状态
    _PytorchRecordFunctionState* rf = _pytorch_record_function_enter(cache_lookup_profiler_str);
    // 查找缓存中是否存在可能的已编译代码
    PyObject* maybe_cached_code = lookup(extra, frame->f_locals, backend);
    // 退出记录函数调用状态
    _pytorch_record_function_exit(rf);

    // 如果没有找到已缓存的代码，标记评估失败并返回 NULL
    if (maybe_cached_code == NULL) {
      *should_clear_frame = 1;
      return NULL;
    }

      // guard eval failed, keep propagating
      *should_clear_frame = 1;
      return NULL;
    }


这段代码还在处理 callback 为 Py_False 的情况，如果没有找到已缓存的代码，则标记评估失败并返回 NULL。
    } else if (maybe_cached_code == Py_None) {
      // 如果缓存中的代码为 Py_None，则表示缓存未命中
      DEBUG_TRACE("cache miss %s", get_frame_name(frame));
      // 调试信息：记录缓存未命中的情况及帧的名称
      return eval_frame_default(tstate, frame, throw_flag);
      // 调用默认的帧评估函数处理该帧，并返回结果
    }
    PyCodeObject* cached_code = (PyCodeObject*)maybe_cached_code;
    // 将缓存的代码对象类型转换为 PyCodeObject*
    // 使用了缓存的版本
    DEBUG_TRACE("cache hit %s", get_frame_name(frame));
    // 调试信息：记录缓存命中的情况及帧的名称
    *should_clear_frame = 1;
    // 设置清除帧标志位为真
    return eval_custom_code(tstate, frame, cached_code, throw_flag, free_vars_copied);
    // 调用自定义的代码评估函数处理该帧，并返回结果
  }
  DEBUG_CHECK(PyDict_CheckExact(frame->f_locals));
  // 调试检查：验证帧的局部变量字典确实是 PyDictObject 类型
  DEBUG_CHECK(PyDict_CheckExact(frame->f_globals));
  // 调试检查：验证帧的全局变量字典确实是 PyDictObject 类型
  DEBUG_CHECK(PyDict_CheckExact(frame->f_builtins));
  // 调试检查：验证帧的内置变量字典确实是 PyDictObject 类型

  // We don't run the current custom_eval_frame behavior for guards.
  // 因此，我们暂时将回调设置为 Py_None，以便在桥接中驱动正确的行为
  eval_frame_callback_set(Py_None);

  _PytorchRecordFunctionState* rf = _pytorch_record_function_enter(cache_lookup_profiler_str);
  // 进入 PyTorch 记录函数的状态，用于缓存查找的性能分析字符串
  PyObject* maybe_cached_code = lookup(extra, frame->f_locals, backend);
  // 在额外信息中查找缓存的代码对象，使用帧的局部变量和后端作为参数
  _pytorch_record_function_exit(rf);
  // 退出 PyTorch 记录的函数状态

  if (maybe_cached_code == NULL) {
    // 如果查找结果为 NULL，表示出现了 Python 错误
    *should_clear_frame = 1;
    // 设置清除帧标志位为真
    return NULL;
    // 返回空指针，表示异常情况
  } else if (maybe_cached_code != Py_None) {
    PyCodeObject* cached_code = (PyCodeObject*)maybe_cached_code;
    // 将查找到的缓存代码对象转换为 PyCodeObject*
    // 使用了缓存的版本
    DEBUG_TRACE("cache hit %s", get_frame_name(frame));
    // 调试信息：记录缓存命中的情况及帧的名称
    // 重新启用自定义行为
    eval_frame_callback_set(callback);
    *should_clear_frame = 1;
    // 设置清除帧标志位为真
    return eval_custom_code(tstate, frame, cached_code, throw_flag, free_vars_copied);
    // 调用自定义的代码评估函数处理该帧，并返回结果
  }
  // 缓存未命中
  CacheEntry* cache_entry = extract_cache_entry(extra);
  // 从额外信息中提取缓存条目
  FrameState* frame_state = extract_frame_state(extra);
  // 从额外信息中提取帧状态
  PyObject* result =
      call_callback(callback, frame, cache_entry, frame_state);
  // 调用回调函数处理帧、缓存条目和帧状态
  if (result == NULL) {
    // 如果回调返回结果为空，表示内部异常
    *should_clear_frame = 1;
    // 设置清除帧标志位为真
    return NULL;
    // 返回空指针，终止执行
  } else if (result != Py_None) {
    DEBUG_TRACE("create cache %s", get_frame_name(frame));
    // 调试信息：记录创建缓存的情况及帧的名称

    // 直接访问 extra->cache_entry 是安全的，因为 extra 在这里不会为 NULL
    CacheEntry* new_cache_entry = create_cache_entry(extra, result, backend);
    // 创建新的缓存条目，并将结果绑定到 extra 上
    Py_DECREF(result);
    // 释放结果对象的引用计数

    // 更新 extra 对象上的现有缓存条目
    // extra 现在是 CacheEntry 对象的所有者
    // 这将更改缓存条目的指针，即使 extra 的 scratch 空间中，我们只是改变了 cache_entry 指针
    // 作为结果，extra 现在是 CacheEntry 对象的所有者
    // 调用 eval_frame_callback_set(callback) 设置帧的回调函数
    eval_frame_callback_set(callback);
    // 将 should_clear_frame 指针指向的整数值设置为 1，表示需要清除帧
    *should_clear_frame = 1;
    // 返回通过 eval_custom_code 执行后的结果
    return eval_custom_code(tstate, frame, CacheEntry_get_code(new_cache_entry), throw_flag, free_vars_copied);
  } else {
    // 打印调试信息，记录创建跳过的帧名称
    DEBUG_TRACE("create skip %s", get_frame_name(frame));
    // 减少 result 对象的引用计数
    Py_DECREF(result);
    // 设置帧的额外状态为 SKIP_CODE
    set_extra_state(F_CODE(frame), SKIP_CODE);
    // 调用 eval_frame_callback_set(callback) 设置帧的回调函数
    eval_frame_callback_set(callback);
    // 返回通过 eval_frame_default 执行后的结果
    return eval_frame_default(tstate, frame, throw_flag);
  }
}

#else // IS_PYTHON_3_14_PLUS

// Fake definitions for everything we removed

typedef struct THPPyInterpreterFrame {
  PyObject_HEAD
  _PyInterpreterFrame* frame; // Borrowed reference
} THPPyInterpreterFrame;

inline static void enable_eval_frame_shim(PyThreadState* tstate) {}
inline static void enable_eval_frame_default(PyThreadState* tstate) {}

static struct PyGetSetDef THPPyInterpreterFrame_properties[] = {NULL};

static PyTypeObject THPPyInterpreterFrameType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "torch._C.dynamo.eval_frame._PyInterpreterFrame",
    .tp_basicsize = sizeof(THPPyInterpreterFrame),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_getset = THPPyInterpreterFrame_properties,
};

#endif // CPython 3.14



static PyObject* increment_working_threads(PyThreadState* tstate) {
  // 增加活跃的 Dynamo 线程计数
  active_dynamo_threads = active_dynamo_threads + 1;
  // 如果活跃的 Dynamo 线程数大于 0，则启用 eval frame 的 shim
  if (active_dynamo_threads > 0) {
    enable_eval_frame_shim(tstate);
  }
  // 返回 None 对象
  Py_RETURN_NONE;
}

static PyObject* decrement_working_threads(PyThreadState* tstate) {
  // 如果活跃的 Dynamo 线程数大于 0
  if (active_dynamo_threads > 0) {
    // 减少活跃的 Dynamo 线程数
    active_dynamo_threads = active_dynamo_threads - 1;
    // 如果活跃的 Dynamo 线程数减少到 0，则恢复 eval frame 的默认行为
    if (active_dynamo_threads == 0) {
      enable_eval_frame_default(tstate);
    }
  }
  // 返回 None 对象
  Py_RETURN_NONE;
}

static PyObject* set_eval_frame(PyObject* new_callback, PyThreadState* tstate) {
  // 更改 eval frame 的回调函数并返回旧的回调函数
  //  - None: 禁用 TorchDynamo
  //  - False: 仅运行模式（重用现有编译）
  //  - Python callable(): 启用 TorchDynamo
  PyObject* old_callback = eval_frame_callback_get();

  // 调用者拥有对旧回调函数的引用
  Py_INCREF(old_callback);

  // 如果旧回调函数不是 None 且新回调函数是 None，则减少活跃的 Dynamo 线程数
  if (old_callback != Py_None && new_callback == Py_None) {
    decrement_working_threads(tstate);
  } 
  // 如果旧回调函数是 None 且新回调函数不是 None，则增加活跃的 Dynamo 线程数
  else if (old_callback == Py_None && new_callback != Py_None) {
    increment_working_threads(tstate);
  }

  // 增加新回调函数的引用计数，以便在设置后保留它
  Py_INCREF(new_callback);
  // 减少旧回调函数的引用计数，因为它不再需要
  Py_DECREF(old_callback);

  // 设置线程局部的回调函数。这将驱动我们的 shim 的行为，如果它被安装的话。
  eval_frame_callback_set(new_callback);

  // 返回旧的回调函数
  return old_callback;
}

static PyObject* set_eval_frame_py(PyObject* dummy, PyObject* callback) {
  // 如果 callback 不是 None、不是 False，并且不是可调用对象，则抛出 TypeError 异常
  if (callback != Py_None && callback != Py_False &&
      !PyCallable_Check(callback)) {
    DEBUG_TRACE0("arg error");
    PyErr_SetString(PyExc_TypeError, "expected a callable");
    return NULL;
  }
  // 调试信息：打印是否启用了 Python，以及是否是运行模式
  DEBUG_TRACE(
      "python enabled=%d and is run_only=%d",
      callback != Py_None,
      callback == Py_False);
  // 设置 eval frame 的回调函数
  return set_eval_frame(callback, PyThreadState_GET());
}

static PyObject* reset_code(PyObject* dummy, PyObject* code) {
  // 如果 code 不是代码对象，则抛出 TypeError 异常
  if (!PyCode_Check(code)) {
    DEBUG_TRACE0("arg error");
    PyErr_SetString(PyExc_TypeError, "expected a code object");
    return NULL;
  }

  // set_extra_state 在额外的临时空间上销毁现有对象
  set_extra_state((PyCodeObject*)code, NULL);
  // 返回 None 对象
  Py_RETURN_NONE;
}
// 定义一个名为 unsupported 的静态函数，接收两个 PyObject 类型的参数
static PyObject* unsupported(PyObject* dummy, PyObject* args) {
    // 用于测试的虚拟 C 函数
    PyObject* obj1 = NULL;
    PyObject* obj2 = NULL;
    // 解析传入的参数元组，期望两个 PyObject 对象
    if (!PyArg_ParseTuple(args, "OO", &obj1, &obj2)) {
        return NULL;
    }
    // 增加 obj2 的引用计数
    Py_INCREF(obj2);
    // 返回 obj2 对象
    return obj2;
}

// 定义一个名为 skip_code 的静态函数，接收一个 PyObject 类型的参数
static PyObject* skip_code(PyObject* dummy, PyObject* obj) {
    // 检查 obj 是否为 PyCode 对象
    if (!PyCode_Check(obj)) {
        // 设置类型错误异常并返回 NULL
        PyErr_SetString(PyExc_TypeError, "expected a code object");
        return NULL;
    }
    // 调用 set_extra_state 函数，设置额外状态为 SKIP_CODE
    set_extra_state((PyCodeObject*)obj, SKIP_CODE);
    // 返回 None 对象
    Py_RETURN_NONE;
}

// 定义一个名为 set_guard_error_hook 的静态函数，接收一个 PyObject 类型的参数
static PyObject* set_guard_error_hook(PyObject* dummy, PyObject* obj) {
    // 如果 obj 是 Py_None，则将 obj 设置为 NULL
    if (obj == Py_None) {
        obj = NULL;
    }
    // 将 guard_error_hook 的值设置为 obj 的新引用
    Py_XSETREF(guard_error_hook, Py_XNewRef(obj));
    // 返回 None 对象
    Py_RETURN_NONE;
}

// 定义一个 PyMethodDef 数组 _methods，包含了模块中所有的方法定义
static PyMethodDef _methods[] = {
    {"set_eval_frame", set_eval_frame_py, METH_O, NULL},
    {"reset_code", reset_code, METH_O, NULL},
    {"unsupported", unsupported, METH_VARARGS, NULL},
    {"skip_code", skip_code, METH_O, NULL},
    {"set_guard_error_hook", set_guard_error_hook, METH_O, NULL},
    {NULL, NULL, 0, NULL}};

// 定义一个 PyModuleDef 结构体 _module，用于描述模块的基本信息和方法列表
static struct PyModuleDef _module = {
    PyModuleDef_HEAD_INIT,  // 模块定义的头部初始化
    "torch._C._dynamo.eval_frame",  // 模块名称
    "Module containing hooks to override eval_frame",  // 模块的文档字符串
    -1,  // 模块状态
    _methods  // 模块的方法列表
};

// 如果定义了 IS_PYTHON_3_12_PLUS 宏，则定义 _PyEval_RequestCodeExtraIndex 宏为 PyUnstable_Eval_RequestCodeExtraIndex
#if IS_PYTHON_3_12_PLUS
#define _PyEval_RequestCodeExtraIndex PyUnstable_Eval_RequestCodeExtraIndex
#endif

// 定义一个名为 torch_c_dynamo_eval_frame_init 的函数，用于初始化 eval_frame 相关功能
PyObject* torch_c_dynamo_eval_frame_init(void) {
    // 使用 destroy_extra_state 函数注册额外状态索引
    extra_index = _PyEval_RequestCodeExtraIndex(destroy_extra_state);
    // 如果注册失败，则设置运行时错误并返回 NULL
    if (extra_index < 0) {
        PyErr_SetString(PyExc_RuntimeError,
                        "dynamo: unable to register extra index");
        return NULL;
    }

    // 创建 eval_frame_callback_key 的线程特定存储
    int result = PyThread_tss_create(&eval_frame_callback_key);
    CHECK(result == 0);

    // 增加 Py_None 的引用计数，并将其设置为 eval_frame 回调的默认值
    Py_INCREF(Py_None);
    eval_frame_callback_set(Py_None);

    // 创建并初始化名为 _module 的 Python 模块
    PyObject* module = PyModule_Create(&_module);
    // 如果创建失败，则返回 NULL
    if (module == NULL) {
        return NULL;
    }

    // 如果定义了 IS_PYTHON_3_11_PLUS 宏，则准备 THPPyInterpreterFrameType 类型对象并添加到模块中
#if IS_PYTHON_3_11_PLUS
    if (PyType_Ready(&THPPyInterpreterFrameType) < 0) {
        return NULL;
    }
    Py_INCREF(&THPPyInterpreterFrameType);
    if (PyModule_AddObject(module, "_PyInterpreterFrame", (PyObject*)&THPPyInterpreterFrameType) != 0) {
        return NULL;
    }
#endif

    // 返回创建的模块对象
    return module;
}
```