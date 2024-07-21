# `.\pytorch\torch\csrc\dynamo\cpython_defs.c`

```
// 包含头文件 torch/csrc/dynamo/cpython_defs.h
#include <torch/csrc/dynamo/cpython_defs.h>

// 如果是在 Windows 平台下，定义 unlikely 宏为其本身
#ifdef _WIN32
#define unlikely(x) (x)
// 否则，在非 Windows 平台下，使用 __builtin_expect 进行条件预测
#else
#define unlikely(x) __builtin_expect((x), 0)
#endif

// 定义宏 CHECK，用于检查条件 cond 是否成立，如果不成立则输出调试信息并中止程序
#define CHECK(cond)                                                     \
  if (unlikely(!(cond))) {                                              \
    fprintf(stderr, "DEBUG CHECK FAILED: %s:%d\n", __FILE__, __LINE__); \
    abort();                                                            \
  } else {                                                              \
  }

// 如果 Python 版本是 3.11 及以上，则执行下面的代码
#if IS_PYTHON_3_11_PLUS

// 解决在混合核心和非核心构建时的问题
// 在 3.12 版本以上，取消定义 _PyGC_FINALIZED
#if IS_PYTHON_3_12_PLUS
#undef _PyGC_FINALIZED
#endif

// 定义宏 Py_BUILD_CORE，并包含 pycore_pystate.h 头文件
#define Py_BUILD_CORE
#include <internal/pycore_pystate.h>

// 定义宏 NEED_OPCODE_TABLES，以获取 _PyOpcode_Deopt 和 _PyOpcode_Caches
#define NEED_OPCODE_TABLES

// 如果 Python 版本是 3.13 及以上，则包含 cpython/code.h 头文件
#if IS_PYTHON_3_13_PLUS
#include <cpython/code.h>
// 定义宏 NEED_OPCODE_METADATA，以获取 pycore_opcode_metadata.h 中的信息，并取消定义 NEED_OPCODE_METADATA
#define NEED_OPCODE_METADATA
#include <internal/pycore_opcode_metadata.h>
#undef NEED_OPCODE_METADATA
// 否则，包含 pycore_opcode.h 头文件
#else
#include <internal/pycore_opcode.h>
#endif

// 取消定义 NEED_OPCODE_TABLES 宏
#undef NEED_OPCODE_TABLES

// 包含 pycore_frame.h 头文件
#include <internal/pycore_frame.h>

// 取消定义 Py_BUILD_CORE 宏
#undef Py_BUILD_CORE

// 为了简化 CPython 端 ABI 变化的影响，这里进行版本检查，如果是 3.14 及以上版本则报错
#if IS_PYTHON_3_14_PLUS
#error "Please ensure that the functions below still match the CPython implementation for 3.14"
#endif

// 定义静态函数 THP_PyFrame_OpAlreadyRan，用于检查指定帧中的操作码是否已运行过
// 参数为 _PyInterpreterFrame 类型的 frame、操作码 opcode 和操作参数 oparg
static int
THP_PyFrame_OpAlreadyRan(_PyInterpreterFrame *frame, int opcode, int oparg)
{
    // 仅当 opcode 是非快速形式时才有效：检查 _PyOpcode_Deopt 中的 opcode 是否等于 opcode
    CHECK(_PyOpcode_Deopt[opcode] == opcode);
    int check_oparg = 0;
    // 遍历帧中的指令，直到 PREV_INSTR(frame) 为止
    for (_Py_CODEUNIT *instruction = _PyCode_CODE(F_CODE(frame));
         instruction < PREV_INSTR(frame) ; instruction++)
    {
        int check_opcode = _PyOpcode_Deopt[_Py_OPCODE(*instruction)];
        check_oparg |= _Py_OPARG(*instruction);
        // 如果找到匹配的 opcode 和 oparg，则返回 1
        if (check_opcode == opcode && check_oparg == oparg) {
            return 1;
        }
        // 如果是 EXTENDED_ARG 操作码，则左移 8 位
        if (check_opcode == EXTENDED_ARG) {
            check_oparg <<= 8;
        }
        else {
            check_oparg = 0;
        }
        // 根据指令的缓存信息跳过相应的指令
        instruction += _PyOpcode_Caches[check_opcode];
    }
    // 如果未找到匹配的操作码和操作参数，则返回 0
    return 0;
}

// 如果是 Python 3.12 及以上版本，则定义函数 frame_init_get_vars
// 用于初始化帧的自由变量，并标记 free_vars_copied
static void
frame_init_get_vars(_PyInterpreterFrame *frame, int *free_vars_copied)
{
    // COPY_FREE_VARS 没有快速形式，因此不需要使用 _PyOpcode_Deopt
    // 获取帧的代码对象 co
    PyCodeObject *co = F_CODE(frame);
    // 获取帧的最后指令索引 lasti
    int lasti = _PyInterpreterFrame_LASTI(frame);
    // 检查是否需要初始化自由变量
    if (!(lasti < 0 && _PyCode_CODE(co)->op.code == COPY_FREE_VARS
          && PyFunction_Check(frame->f_funcobj)))
    {
        /* Free vars are initialized */
        // 如果不需要初始化自由变量，则直接返回
        return;
    }

    /* Free vars have not been initialized -- Do that */
    // 获取闭包对象
    PyObject *closure = ((PyFunctionObject *)frame->f_funcobj)->func_closure;
    // 根据 Python 版本选择合适的偏移量函数
    #if IS_PYTHON_3_13_PLUS
    int offset = PyUnstable_Code_GetFirstFree(co);
    #else
    int offset = PyCode_GetFirstFree(co);
    #endif
    // 遍历自由变量列表，初始化每个自由变量
    for (int i = 0; i < co->co_nfreevars; ++i) {
        // 获取闭包中的每个变量对象
        PyObject *o = PyTuple_GET_ITEM(closure, i);
        // 将变量对象添加到帧对象的局部变量中
        frame->localsplus[offset + i] = Py_NewRef(o);
    }
    // 更新上一个指令为当前代码对象的首指令
    PREV_INSTR(frame) = _PyCode_CODE(F_CODE(frame));

    // 标记自由变量已经复制完成
    *free_vars_copied = 1;
// https://github.com/python/cpython/blob/0325a8a8cdba6c091bcbbb3c995f3bf1d1217012/Objects/frameobject.c#L1213
static PyObject *
THP_PyFrame_GetLocals(_PyInterpreterFrame *frame, int include_hidden, int *free_vars_copied)
{
    /* Merge fast locals into f->f_locals */
    PyObject *locals = frame->f_locals;
    // 如果 f_locals 为空，则创建一个新的空字典作为 f_locals
    if (locals == NULL) {
        locals = frame->f_locals = PyDict_New();
        // 如果创建失败，则返回 NULL
        if (locals == NULL) {
            return NULL;
        }
    }
    PyObject *hidden = NULL;

    /* If include_hidden, "hidden" fast locals (from inlined comprehensions in
       module/class scopes) will be included in the returned dict, but not in
       frame->f_locals; the returned dict will be a modified copy. Non-hidden
       locals will still be updated in frame->f_locals. */
    // 如果 include_hidden 为真，创建一个新的字典 hidden 来存储隐藏的快速局部变量
    if (include_hidden) {
        hidden = PyDict_New();
        // 如果创建失败，则返回 NULL
        if (hidden == NULL) {
            return NULL;
        }
    }

    // 初始化获取变量
    frame_init_get_vars(frame, free_vars_copied);

    // 获取当前帧的代码对象
    PyCodeObject *co = F_CODE(frame);
    // 遍历函数对象的局部变量空间
    for (int i = 0; i < co->co_nlocalsplus; i++) {
        PyObject *value;  // 借用引用，指向局部变量的值
        // 获取当前帧的变量值，并检查是否成功获取
        if (!frame_get_var(frame, co, i, &value)) {
            continue;  // 如果获取失败，则继续下一次循环
        }

        // 获取局部变量名对象
        PyObject *name = PyTuple_GET_ITEM(co->co_localsplusnames, i);
        // 获取局部变量的种类（是否隐藏）
        _PyLocals_Kind kind = _PyLocals_GetKind(co->co_localspluskinds, i);
        
        // 如果局部变量被标记为隐藏，并且允许包含隐藏变量，并且变量值不为空
        if (kind & CO_FAST_HIDDEN) {
            if (include_hidden && value != NULL) {
                // 将隐藏的局部变量及其值添加到隐藏字典中
                if (PyObject_SetItem(hidden, name, value) != 0) {
                    goto error;  // 如果设置失败，跳转到错误处理
                }
            }
            continue;  // 继续下一次循环处理下一个变量
        }

        // 如果变量值为空
        if (value == NULL) {
            // 尝试从局部变量字典中删除该变量名
            if (PyObject_DelItem(locals, name) != 0) {
                // 如果删除失败，检查是否是因为 Key Error 异常
                if (PyErr_ExceptionMatches(PyExc_KeyError)) {
                    PyErr_Clear();  // 清除异常
                }
                else {
                    goto error;  // 其他异常则跳转到错误处理
                }
            }
        }
        else {
            // 否则，将变量名及其值设置到局部变量字典中
            if (PyObject_SetItem(locals, name, value) != 0) {
                goto error;  // 如果设置失败，跳转到错误处理
            }
        }
    }

    // 如果允许包含隐藏变量，并且隐藏字典不为空
    if (include_hidden && PyDict_Size(hidden)) {
        // 创建一个新的局部变量字典
        PyObject *innerlocals = PyDict_New();
        if (innerlocals == NULL) {
            goto error;  // 如果创建失败，跳转到错误处理
        }
        // 将当前局部变量字典和隐藏字典合并到新的局部变量字典中
        if (PyDict_Merge(innerlocals, locals, 1) != 0) {
            Py_DECREF(innerlocals);
            goto error;  // 如果合并失败，跳转到错误处理
        }
        if (PyDict_Merge(innerlocals, hidden, 1) != 0) {
            Py_DECREF(innerlocals);
            goto error;  // 如果合并失败，跳转到错误处理
        }
        locals = innerlocals;  // 更新局部变量字典为新的合并后的字典
    }
    else {
        Py_INCREF(locals);  // 否则，增加局部变量字典的引用计数
    }
    Py_CLEAR(hidden);  // 清除隐藏字典的引用

    return locals;  // 返回最终的局部变量字典

  error:
    Py_XDECREF(hidden);  // 清除隐藏字典的引用，使用 Py_XDECREF 以防止空指针错误
    return NULL;  // 返回 NULL 表示发生错误
// https://github.com/python/cpython/blob/0325a8a8cdba6c091bcbbb3c995f3bf1d1217012/Objects/frameobject.c#L1301
// 实现了将快速局部变量合并到局部变量中的函数，带有错误处理
int
THP_PyFrame_FastToLocalsWithError(_PyInterpreterFrame *frame, int *free_vars_copied)
{
    // 获取当前帧的局部变量字典
    PyObject *locals = THP_PyFrame_GetLocals(frame, 0, free_vars_copied);
    // 如果获取失败，则返回错误码
    if (locals == NULL) {
        return -1;
    }
    // 减少局部变量字典的引用计数，避免内存泄漏
    Py_DECREF(locals);
    // 返回操作成功的状态码
    return 0;
}


#else

// https://github.com/python/cpython/blob/a7715ccfba5b86ab09f86ec56ac3755c93b46b48/Objects/frameobject.c#L1182
// 当存在 free_vars_copied 参数时，用于通知调用者 COPY_FREE_VARS 代码路径已经发生
int
THP_PyFrame_FastToLocalsWithError(_PyInterpreterFrame *frame, int *free_vars_copied) {
    /* Merge fast locals into f->f_locals */
    // 定义局部变量
    PyObject *locals = NULL;
    PyObject **fast = NULL;
    PyCodeObject *co = NULL;
    // 获取当前帧的局部变量字典
    locals = frame->f_locals;
    // 如果局部变量字典为空，则创建一个新的空字典
    if (locals == NULL) {
        locals = frame->f_locals = PyDict_New();
        // 如果创建失败，则返回错误码
        if (locals == NULL)
            return -1;
    }
    // 获取当前帧的代码对象
    co = F_CODE(frame);
    // 获取当前帧的快速局部变量数组
    fast = _PyFrame_GetLocalsArray(frame);
    // 如果最后一次执行的指令索引小于零，并且第一个指令的操作码是 COPY_FREE_VARS
    if (lasti < 0 && _Py_OPCODE(_PyCode_CODE(co)[0]) == COPY_FREE_VARS) {
        /* Free vars have not been initialized -- Do that */
        // 重新获取代码对象
        PyCodeObject *co = F_CODE(frame);
        // 获取当前函数对象的闭包
        PyObject *closure = frame->f_func->func_closure;
        // 计算偏移量，以确定从闭包中复制变量的位置
        int offset = co->co_nlocals + co->co_nplaincellvars;
        // 遍历自由变量，并将其复制到局部变量数组中
        for (int i = 0; i < co->co_nfreevars; ++i) {
            PyObject *o = PyTuple_GET_ITEM(closure, i);
            Py_INCREF(o);
            frame->localsplus[offset + i] = o;
        }
        // 指定先前指令为当前代码对象的第一个指令
        PREV_INSTR(frame) = _PyCode_CODE(F_CODE(frame));

        // 设置 free_vars_copied 标志为真
        *free_vars_copied = 1;
    }
    // 遍历本地变量及其属性的数组，执行以下操作直到数组末尾
    for (int i = 0; i < co->co_nlocalsplus; i++) {
        // 获取本地变量的类型
        _PyLocals_Kind kind = _PyLocals_GetKind(co->co_localspluskinds, i);

        /* 如果命名空间未经优化，则可能出现以下情况：
           1. 不包含自由变量，因为使用了 import * 或者是顶层命名空间。
           2. 是一个类的命名空间。
           我们不希望将自由变量意外复制到类使用的本地变量字典中。 */
        // 如果变量包含 CO_FAST_FREE 标志并且命名空间未优化，则跳过当前循环
        if (kind & CO_FAST_FREE && !(co->co_flags & CO_OPTIMIZED)) {
            continue;
        }

        // 获取本地变量名对象
        PyObject *name = PyTuple_GET_ITEM(co->co_localsplusnames, i);
        // 获取本地变量的值
        PyObject *value = fast[i];
        // 如果帧栈顶部不为空，则执行以下操作
        if (frame->stacktop) {
            // 如果变量包含 CO_FAST_FREE 标志
            if (kind & CO_FAST_FREE) {
                // 如果值非空且为 PyCell 对象，则获取其真实值
                // 此处假设 value 非空，且是 PyCell 对象
                CHECK(value != NULL && PyCell_Check(value));
                value = PyCell_GET(value);
            }
            // 如果变量包含 CO_FAST_CELL 标志
            else if (kind & CO_FAST_CELL) {
                // 注意在 MAKE_CELL 执行之前不会发生 *_DEREF 操作
                if (value != NULL) {
                    // 如果值是 PyCell 对象，并且 MAKE_CELL 操作已经执行
                    if (PyCell_Check(value) &&
                            THP_PyFrame_OpAlreadyRan(frame, MAKE_CELL, i)) {
                        // 假设 MAKE_CELL 操作已经执行，获取 PyCell 对象的值
                        value = PyCell_GET(value);
                    }
                    // 否则，假设它是一个参数（kind & CO_FAST_LOCAL），
                    // 其初始值在帧创建时设置...
                    // 或者它是通过调用 PyFrame_LocalsToFast() 设置的初始值...
                    // （不太可能）...或者它是通过早期调用 PyFrame_LocalsToFast() 设置的初始值。
                }
            }
        }
        // 如果帧栈顶部为空
        else {
            // 检查值必须为空
            CHECK(value == NULL);
        }
        // 如果值为空
        if (value == NULL) {
            // 从本地变量字典中删除指定名称的项目
            if (PyObject_DelItem(locals, name) != 0) {
                // 如果发生 KeyError 异常，清除异常状态
                if (PyErr_ExceptionMatches(PyExc_KeyError)) {
                    PyErr_Clear();
                }
                else {
                    // 返回 -1 表示操作失败
                    return -1;
                }
            }
        }
        // 如果值非空
        else {
            // 将指定名称和值设置到本地变量字典中
            if (PyObject_SetItem(locals, name, value) != 0) {
                // 返回 -1 表示操作失败
                return -1;
            }
        }
    }
    // 执行成功，返回 0
    return 0;
#endif
// 结束条件编译指令，用于条件编译的结束

// e.g. COPY_FIELD(op, o, globals) becomes
// PY_XINCREF((o)->func_globals);
// (op)->func_globals = (o)->func_globals;
// 宏定义，用于复制结构体字段的宏，增加引用计数并复制字段值

// Not actually copied from CPython, but loosely based on
// https://github.com/python/cpython/blob/e715da6db1d1d70cd779dc48e1ba8110c51cc1bf/Objects/funcobject.c
// Makes a new PyFunctionObject copy of `o`, but with the code object fields
// determined from `code`.
// Ensure that all fields defined in the PyFunctionObject struct in
// https://github.com/python/cpython/blob/e715da6db1d1d70cd779dc48e1ba8110c51cc1bf/Include/cpython/funcobject.h
// are accounted for.
// 根据给定的 code 对象创建一个新的 PyFunctionObject 副本，确保复制所有必要的字段

PyFunctionObject *
_PyFunction_CopyWithNewCode(PyFunctionObject *o, PyCodeObject* code)
{
  // 分配新的 PyFunctionObject 对象，使用 PyObject_GC_New 分配内存
  PyFunctionObject *op = PyObject_GC_New(PyFunctionObject, &PyFunction_Type);
  if (op == NULL) {
    return NULL;
  }
  // 增加对 code 对象的引用计数
  Py_XINCREF(code);
  // 将 func_code 字段设置为给定的 code 对象
  op->func_code = (PyObject *) code;
  // 增加对 code 对象的 co_name 字段的引用计数，并将 func_name 设置为 code 的 co_name
  Py_XINCREF(code->co_name);
  op->func_name = code->co_name;
  // 增加对 code 对象的 co_qualname 字段的引用计数，并将 func_qualname 设置为 code 的 co_qualname
  Py_XINCREF(code->co_qualname);
  op->func_qualname = code->co_qualname;
  // 复制其他字段，使用宏定义 COPY_FIELD 复制字段值
  COPY_FIELD(op, o, globals);
  COPY_FIELD(op, o, builtins);
  COPY_FIELD(op, o, defaults);
  COPY_FIELD(op, o, kwdefaults);
  COPY_FIELD(op, o, closure);
  COPY_FIELD(op, o, doc);
  COPY_FIELD(op, o, dict);
  // 初始化 func_weakreflist 为 NULL
  op->func_weakreflist = NULL;
  COPY_FIELD(op, o, module);
  COPY_FIELD(op, o, annotations);
  // 如果是 Python 3.12 及以上版本，则复制 typeparams 字段
  #if IS_PYTHON_3_12_PLUS
  COPY_FIELD(op, o, typeparams);
  #endif
  // 复制 vectorcall 和 func_version 字段
  op->vectorcall = o->vectorcall;
  op->func_version = o->func_version;
  // 跟踪新分配的对象以进行垃圾回收
  PyObject_GC_Track(op);
  // 返回新创建的 PyFunctionObject 对象
  return op;
}

// From https://github.com/python/cpython/blob/e715da6db1d1d70cd779dc48e1ba8110c51cc1bf/Objects/frameobject.c#L1020
// 创建一个新的 PyFrameObject 对象，但不进行垃圾回收跟踪
PyFrameObject*
THP_PyFrame_New_NoTrack(const PyCodeObject *code)
{
    // DYNAMO: commented out
    // CALL_STAT_INC(frame_objects_created);
    // 计算所需的 slots 数量
    int slots = code->co_nlocalsplus + code->co_stacksize;
    // 分配新的 PyFrameObject 对象，使用 PyObject_GC_NewVar 分配内存
    PyFrameObject *f = PyObject_GC_NewVar(PyFrameObject, &PyFrame_Type, slots);
    if (f == NULL) {
        return NULL;
    }
    // 初始化 f_back 和 f_trace 为 NULL，设置跟踪行和操作码的标志
    f->f_back = NULL;
    f->f_trace = NULL;
    f->f_trace_lines = 1;
    f->f_trace_opcodes = 0;
    // 根据 Python 版本设置额外的局部变量或快速局部变量标志
#if IS_PYTHON_3_13_PLUS
    f->f_extra_locals = NULL;
#else
    f->f_fast_as_locals = 0;
#endif
    // 初始化 f_lineno 为 0
    f->f_lineno = 0;
    // 返回新创建的 PyFrameObject 对象
    return f;
}

// From https://github.com/python/cpython/blob/e715da6db1d1d70cd779dc48e1ba8110c51cc1bf/Python/frame.c#L27
// 创建并设置 PyFrameObject 对象，并根据给定的 _PyInterpreterFrame 对象进行初始化
PyFrameObject *
THP_PyFrame_MakeAndSetFrameObject(_PyInterpreterFrame *frame)
{
    // 检查 frame_obj 是否为 NULL
    CHECK(frame->frame_obj == NULL);
    PyObject *error_type = NULL, *error_value = NULL, *error_traceback = NULL;
    // 保存当前的错误状态
    PyErr_Fetch(&error_type, &error_value, &error_traceback);

    // 使用 F_CODE 宏创建一个新的 PyFrameObject 对象
    PyFrameObject *f = THP_PyFrame_New_NoTrack(F_CODE(frame));
    if (f == NULL) {
        // 如果创建失败，恢复之前保存的错误状态并返回 NULL
        Py_XDECREF(error_type);
        Py_XDECREF(error_value);
        Py_XDECREF(error_traceback);
        return NULL;
    }
    // 恢复之前保存的错误状态
    PyErr_Restore(error_type, error_value, error_traceback);
    // 检查当前帧对象是否已经存在
    if (frame->frame_obj) {
        // GH-97002: 我们是如何陷入这种可怕的情况的？最可能的是，分配 f 触发了 GC 收集，
        // 它运行了一些代码，该代码*也*创建了相同的帧...而我们正在创建这个帧的过程中！
        // 详见 test_frame.py 中的 test_sneaky_frame_object 的具体示例。
        //
        // 不管怎样，只需丢弃 f 并使用已经向用户代码暴露的那个帧，因为它已经存在。
        // 实际上，要做到这一点有点棘手，因为我们不再由一个真正的 _PyInterpreterFrame 支持。
        // 只需假装我们有一个拥有的、已清除的帧，这样 frame_dealloc 不会使情况变得更糟：
        f->f_frame = (_PyInterpreterFrame *)f->_f_frame_data;
        f->f_frame->owner = FRAME_CLEARED;
        f->f_frame->frame_obj = f;
        // 减少 f 的引用计数，因为我们不再需要它
        Py_DECREF(f);
        // 返回当前帧对象的指针
        return frame->frame_obj;
    }
    // 检查帧对象的所有权状态，确保不处于被帧对象拥有或已清除的状态
    CHECK(frame->owner != FRAME_OWNED_BY_FRAME_OBJECT);
    CHECK(frame->owner != FRAME_CLEARED);
    // 将当前帧对象设置为 f 所代表的帧
    f->f_frame = frame;
    // 设置帧对象的 frame_obj 属性为 f
    frame->frame_obj = f;
    // 返回 f
    return f;
// 从给定的 _PyInterpreterFrame 指针获取对应的 PyFrameObject 对象
static inline PyFrameObject *
THP_PyFrame_GetFrameObject(_PyInterpreterFrame *frame)
{
    // 检查帧对象不是不完整的
    CHECK(!_PyFrame_IsIncomplete(frame));
    // 获取帧对象
    PyFrameObject *res =  frame->frame_obj;
    if (res != NULL) {
        return res;
    }
    // 如果帧对象为空，则创建并设置帧对象
    return THP_PyFrame_MakeAndSetFrameObject(frame);
}

// 接受一个 PyFrameObject 对象和 _PyInterpreterFrame 对象，将所有权转移给 PyFrameObject
static void
THP_take_ownership(PyFrameObject *f, _PyInterpreterFrame *frame)
{
    // 检查帧对象的所有权不属于帧对象自身
    CHECK(frame->owner != FRAME_OWNED_BY_FRAME_OBJECT);
    // 检查帧对象的所有权不是已清除状态
    CHECK(frame->owner != FRAME_CLEARED);
    // 计算需要拷贝的内存大小
    Py_ssize_t size = ((char*)&frame->localsplus[frame->stacktop]) - (char *)frame;
    // 将 _PyInterpreterFrame 数据拷贝到 PyFrameObject 中
    memcpy((_PyInterpreterFrame *)f->_f_frame_data, frame, size);
    // 更新 PyFrameObject 的帧对象引用
    frame = (_PyInterpreterFrame *)f->_f_frame_data;
    f->f_frame = frame;
    // 将 _PyInterpreterFrame 的所有权设置为帧对象所拥有
    frame->owner = FRAME_OWNED_BY_FRAME_OBJECT;
    // 如果帧对象是不完整的，则假定第一个 RESUME 已执行
    if (_PyFrame_IsIncomplete(frame)) {
        PyCodeObject *code = F_CODE(frame);
        PREV_INSTR(frame) = _PyCode_CODE(code) + code->_co_firsttraceable;
    }
    // 再次检查帧对象不是不完整的
    CHECK(!_PyFrame_IsIncomplete(frame));
    // 确保 PyFrameObject 的 f_back 指针为空
    CHECK(f->f_back == NULL);
    // 查找前一个 _PyInterpreterFrame，确保不是不完整的
    _PyInterpreterFrame *prev = frame->previous;
    while (prev && _PyFrame_IsIncomplete(prev)) {
        prev = prev->previous;
    }
    // 如果存在前一个 _PyInterpreterFrame，则将其链接到 PyFrameObject 的 f_back 中
    if (prev) {
        PyFrameObject *back = THP_PyFrame_GetFrameObject(prev);
        if (back == NULL) {
            // 内存错误处理
            CHECK(PyErr_ExceptionMatches(PyExc_MemoryError));
            PyErr_Clear();
        }
        else {
            f->f_back = (PyFrameObject *)Py_NewRef(back);
        }
        frame->previous = NULL;
    }
    // 使用公共的 GC 函数来跟踪 PyFrameObject，而不是内部函数
    if (!PyObject_GC_IsTracked((PyObject *) f)) {
        PyObject_GC_Track((PyObject *) f);
    }
}

// 清除给定的 _PyInterpreterFrame 帧对象
void
THP_PyFrame_Clear(_PyInterpreterFrame *frame)
{
    /* It is the responsibility of the owning generator/coroutine
     * to have cleared the enclosing generator, if any. */
    // 检查帧对象的所有权不属于生成器或协程，或者其关联的生成器帧状态已经清除
    CHECK(frame->owner != FRAME_OWNED_BY_GENERATOR ||
        _PyFrame_GetGenerator(frame)->gi_frame_state == FRAME_CLEARED);
    // GH-99729: 清除此帧对象可能会暴露堆栈（通过 finalizers）。确保此帧已被解链并不再可见
#if IS_PYTHON_3_13_PLUS
    // 检查当前线程状态中的当前帧不是给定的帧对象
    CHECK(_PyThreadState_GET()->current_frame != frame);
#else
    // 检查当前线程状态中的 cframe 的当前帧不是给定的帧对象
    CHECK(_PyThreadState_GET()->cframe->current_frame != frame);
#endif
    // 如果帧对象存在，则进行处理
    if (frame->frame_obj) {
        // 将帧对象赋值给本地变量 f
        PyFrameObject *f = frame->frame_obj;
        // 将 frame->frame_obj 置为 NULL，表示不再持有引用
        frame->frame_obj = NULL;
        // 如果 f 的引用计数大于 1，则交给 THP_take_ownership 处理引用计数
        if (Py_REFCNT(f) > 1) {
            THP_take_ownership(f, frame);
            // 减少 f 的引用计数，但不销毁对象
            Py_DECREF(f);
            // 直接返回，不执行后续清理操作
            return;
        }
        // 减少 f 的引用计数，准备销毁对象
        Py_DECREF(f);
    }
    // 检查帧的栈顶是否大于等于 0
    CHECK(frame->stacktop >= 0);
    // 逐个释放帧对象的 localsplus 数组中的引用
    for (int i = 0; i < frame->stacktop; i++) {
        Py_XDECREF(frame->localsplus[i]);
    }
    // 释放帧对象的 frame_obj 引用
    Py_XDECREF(frame->frame_obj);
    // 释放帧对象的 f_locals 引用
    Py_XDECREF(frame->f_locals);
    // 根据 Python 版本选择释放 frame->f_funcobj 或 frame->f_func 的引用
    // 对于 Python 3.12 及以上版本，释放 frame->f_funcobj 的引用
    // 否则，释放 frame->f_func 的引用
    #if IS_PYTHON_3_12_PLUS
    Py_DECREF(frame->f_funcobj);
    #else
    Py_DECREF(frame->f_func);
    #endif
    // 释放帧对象关联的代码对象的引用
    Py_DECREF(F_CODE(frame));
}

// 此处是一个函数的结束标志，对应于开头的函数定义

// https://github.com/python/cpython/blob/fad48ea1816be3125ea51edcdfe2f999d6ade796/Objects/obmalloc.c#L635
// 使用 PyObject_GetArenaAllocator 获取内存分配器，然后使用其分配函数分配指定大小的内存空间
void *
THP_PyObject_VirtualAlloc(size_t size)
{
    PyObjectArenaAllocator arena;
    PyObject_GetArenaAllocator(&arena);
    return arena.alloc(arena.ctx, size);
}

// https://github.com/python/cpython/blob/fad48ea1816be3125ea51edcdfe2f999d6ade796/Objects/obmalloc.c#L641
// 使用 PyObject_GetArenaAllocator 获取内存分配器，然后使用其释放函数释放指定内存空间
void
THP_PyObject_VirtualFree(void *obj, size_t size)
{
    PyObjectArenaAllocator arena;
    PyObject_GetArenaAllocator(&arena);
    return arena.free(arena.ctx, obj, size);
}

// https://github.com/python/cpython/blob/051b8a2589ff28f0194c3701b21f729444691752/Python/pystate.c#L728
// 分配一个新的数据堆栈块，并初始化其属性
static _PyStackChunk*
allocate_chunk(int size_in_bytes, _PyStackChunk* previous)
{
    CHECK(size_in_bytes % sizeof(PyObject **) == 0);
    _PyStackChunk *res = THP_PyObject_VirtualAlloc(size_in_bytes);
    if (res == NULL) {
        return NULL;
    }
    res->previous = previous;
    res->size = size_in_bytes;
    res->top = 0;
    return res;
}

#define DATA_STACK_CHUNK_SIZE (16*1024)
#define MINIMUM_OVERHEAD 1000

// https://github.com/python/cpython/blob/051b8a2589ff28f0194c3701b21f729444691752/Python/pystate.c#L2182
// 将一个数据堆栈块推入线程状态的数据堆栈中
static PyObject **
push_chunk(PyThreadState *tstate, int size)
{
    int allocate_size = DATA_STACK_CHUNK_SIZE;
    while (allocate_size < (int)sizeof(PyObject*)*(size + MINIMUM_OVERHEAD)) {
        allocate_size *= 2;
    }
    _PyStackChunk *new = allocate_chunk(allocate_size, tstate->datastack_chunk);
    if (new == NULL) {
        return NULL;
    }
    if (tstate->datastack_chunk) {
        tstate->datastack_chunk->top = tstate->datastack_top -
                                       &tstate->datastack_chunk->data[0];
    }
    tstate->datastack_chunk = new;
    tstate->datastack_limit = (PyObject **)(((char *)new) + allocate_size);
    // 当 new 是“根”块时（即 new->previous == NULL），可以通过“跳过”第一个元素来防止 _PyThreadState_PopFrame 以后释放它：
    PyObject **res = &new->data[new->previous == NULL];
    tstate->datastack_top = res + size;
    return res;
}

// https://github.com/python/cpython/blob/051b8a2589ff28f0194c3701b21f729444691752/Include/internal/pycore_frame.h#L199
// 检查线程状态的数据堆栈是否有足够的空间来分配给定大小的数据
static inline bool
THP_PyThreadState_HasStackSpace(PyThreadState *tstate, size_t size)
{
    CHECK(
        (tstate->datastack_top == NULL && tstate->datastack_limit == NULL)
        ||
        (tstate->datastack_top != NULL && tstate->datastack_limit != NULL)
    );
    return tstate->datastack_top != NULL &&
        size < (size_t)(tstate->datastack_limit - tstate->datastack_top);
}

// https://github.com/python/cpython/blob/051b8a2589ff28f0194c3701b21f729444691752/Python/pystate.c#L2207
// 在慢速路径下调整解释器帧的指针位置
_PyInterpreterFrame *
THP_PyThreadState_BumpFramePointerSlow(PyThreadState *tstate, size_t size)
{
    # 检查线程状态中是否有足够的堆栈空间来分配指定大小的数据
    if (THP_PyThreadState_HasStackSpace(tstate, size)) {
        # 将当前数据栈顶部的位置保存到 res 中
        _PyInterpreterFrame *res = (_PyInterpreterFrame *)tstate->datastack_top;
        # 增加数据栈顶部的位置，以分配给定大小的数据空间
        tstate->datastack_top += size;
        # 返回分配的数据空间的起始位置
        return res;
    }
    # 如果请求的大小超过 INT_MAX 的一半，表示内存分配失败
    if (size > INT_MAX/2) {
        # 抛出内存不足的异常
        PyErr_NoMemory();
        return NULL;
    }
    # 使用 push_chunk 函数分配指定大小的数据空间，并返回起始位置
    return (_PyInterpreterFrame *)push_chunk(tstate, (int)size);
// 定义一个名为 THP_PyThreadState_PopFrame 的函数，接受 PyThreadState 结构和 _PyInterpreterFrame 结构作为参数
void
THP_PyThreadState_PopFrame(PyThreadState *tstate, _PyInterpreterFrame * frame)
{
    // 检查 tstate 的 datastack_chunk 是否有效
    CHECK(tstate->datastack_chunk);
    
    // 将 frame 强制类型转换为 PyObject** 类型的 base 指针
    PyObject **base = (PyObject **)frame;
    
    // 如果 base 指针指向当前数据栈块的起始位置
    if (base == &tstate->datastack_chunk->data[0]) {
        // 获取当前数据栈块和前一个数据栈块的指针
        _PyStackChunk *chunk = tstate->datastack_chunk;
        _PyStackChunk *previous = chunk->previous;
        
        // 确保 push_chunk 函数保证根数据块永不弹出
        CHECK(previous);
        
        // 调整数据栈顶和当前数据栈块指针到前一个数据栈块
        tstate->datastack_top = &previous->data[previous->top];
        tstate->datastack_chunk = previous;
        
        // 释放当前数据栈块的内存并设置数据栈的限制
        THP_PyObject_VirtualFree(chunk, chunk->size);
        tstate->datastack_limit = (PyObject **)(((char *)previous) + previous->size);
    }
    else {
        // 否则，将数据栈顶指针设置为 base 指针
        CHECK(tstate->datastack_top);
        CHECK(tstate->datastack_top >= base);
        tstate->datastack_top = base;
    }
}

// 如果定义了 IS_PYTHON_3_11_PLUS 宏，则定义 THP_PyOpcode_Caches 和 THP_PyOpcode_Caches_size
#if IS_PYTHON_3_11_PLUS

// THP_PyOpcode_Caches 指向 _PyOpcode_Caches 的首地址
const uint8_t* THP_PyOpcode_Caches = _PyOpcode_Caches;

// THP_PyOpcode_Caches_size 等于 _PyOpcode_Caches 数组的大小除以 uint8_t 的大小
const int THP_PyOpcode_Caches_size = sizeof(_PyOpcode_Caches) / sizeof(uint8_t);

// 否则，如果未定义 IS_PYTHON_3_11_PLUS 宏，则定义 THP_PyOpcode_Caches 和 THP_PyOpcode_Caches_size
#else

// THP_PyOpcode_Caches 指向 NULL
const uint8_t* THP_PyOpcode_Caches = NULL;

// THP_PyOpcode_Caches_size 等于 0
const int THP_PyOpcode_Caches_size = 0;

#endif
```