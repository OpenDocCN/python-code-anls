# `.\pytorch\torch\csrc\dynamo\extra_state.h`

```
#pragma once
// 只包含一次这个文件

#include <Python.h>
// 包含 Python.h 头文件，用于与 Python 解释器交互

#ifdef __cplusplus
// 如果是 C++ 环境，则进行以下定义和声明

#include <torch/csrc/dynamo/utils.h>
// 包含 torch/csrc/dynamo/utils.h 头文件

#include <torch/csrc/utils/pybind.h>
// 包含 torch/csrc/utils/pybind.h 头文件

#include <list>
// 包含标准库中的 list 头文件

namespace py = pybind11;
// 声明命名空间别名 py 为 pybind11

extern "C" {
// 声明以下内容是按照 C 语言风格进行编译和链接的
#endif

// Flag to just run a frame normally
#define SKIP_CODE ((void*)0x1)
// 定义 SKIP_CODE 宏，表示只是正常运行一个帧

// Points to the extra scratch space on the code object
extern Py_ssize_t extra_index;
// 声明 extra_index 变量，指向代码对象的额外临时空间

// function to call when cache lookup errors
extern PyObject* guard_error_hook;
// 声明 guard_error_hook 变量，用于缓存查找错误时调用的函数

typedef PyObject FrameState;
// 定义 FrameState 类型为 PyObject

typedef struct CacheEntry CacheEntry;
// 声明 CacheEntry 结构体类型

// ExtraState encasulates CacheEntry and FrameState. ExtraState is the highest
// level of abstraction of what is stored on the extra code object. Previously,
// we saved different parts on different extra indexes.  We prefer this way
// because of cleaner abstraction and faster SetExtra access.

#ifdef __cplusplus
// 如果是 C++ 环境，则进行以下定义和声明

typedef struct VISIBILITY_HIDDEN ExtraState {
  // List of cache entries for compiled code objects
  std::list<CacheEntry> cache_entry_list;
  // 用于编译代码对象的缓存条目列表

  // Frame state to detect dynamic shape dims
  py::dict frame_state;
  // 用于检测动态形状维度的帧状态

  CacheEntry* get_first_entry();
  // 声明 get_first_entry 函数，返回 CacheEntry 指针

  void move_to_front(CacheEntry* cache_entry);
  // 声明 move_to_front 函数，接受 CacheEntry 指针作为参数

  void invalidate(CacheEntry* cache_entry);
  // 声明 invalidate 函数，接受 CacheEntry 指针作为参数

} ExtraState;
// 定义 ExtraState 结构体类型

#else

typedef struct ExtraState ExtraState;
// 如果不是 C++ 环境，仅声明 ExtraState 结构体类型

#endif

// Helper to extra the cache_entry from the extra state.
// Ownership contract
// args
//  - extra_state: Borrowed
// return
//  - CacheEntry: Borrowed.
CacheEntry* extract_cache_entry(ExtraState* extra_state);
// 声明 extract_cache_entry 函数，从 extra_state 中提取 cache_entry

// Returns either the previously stored frame state or an empty dict.
// Ownership contract
// args
//  - extra_state: Borrowed
// return
//  - extra_state->frame_state: Borrowed.
FrameState* extract_frame_state(ExtraState* extra_state);
// 声明 extract_frame_state 函数，返回先前存储的帧状态或空字典

// Ownership contract
// args
//  - code: Borrowed
// return
//  - extra_state: Borrowed.
ExtraState* get_extra_state(PyCodeObject* code);
// 声明 get_extra_state 函数，返回与给定代码对象关联的 ExtraState 结构体指针

// This is passed as freefunc to _PyEval_RequestCodeExtraIndex. This acts as a
// deleter for the object on extra scratch space. This function is called
// internally in _PyCode_SetExtra and also during the code deallocation.

// Destroys the extra state by deleting cache_entry, frame state and finally
// freeing the constructed extra state.

// Developer note - You should not call this function directly. This is called
// directly inside set_extra_state. If you are in a situation trying to call
// this function, consider if set_extra_state should be called.
void destroy_extra_state(void* obj);
// 声明 destroy_extra_state 函数，作为 _PyEval_RequestCodeExtraIndex 的 freefunc 参数

// Clears the existing object sitting on the extra scratch spance and sets it
// up with the new state. Note that _PyCode_SetExtra calls the
// destroy_extra_state deleter internally, and therefore we don't call it
// explicity here.

// Ownership contract
// args
//  - extra_state: Stolen
// return
//  - there is no return, but the extra_state is stolen, so it becomes
//  set_extra_state responsibility to clean it up. It will be deleted during
//  the reset_code/skip, when the set_extra_state is called with
//  NULL/SKIP_CODE.
// 设置额外状态到代码对象中。
// 如果已经存在额外状态，则先释放旧的额外状态（也是新的额外状态），然后在临时空间上写入无效内容。
void set_extra_state(PyCodeObject* code, ExtraState* extra_state);

// 创建一个新的额外状态并将其放置在代码对象的额外临时空间中。
// 所有权约定
// 参数
//  - code: 借用的
// 返回值:
//  - extra_state: 新的引用
// 这些引用之后会被进一步传递给set_extra_state，后者成为这些引用的最终所有者。
ExtraState* init_and_set_extra_state(PyCodeObject* code);

// 查找由extra_state持有的缓存。
// 所有权约定
// 参数
//  - extra_state: 借用的
//  - f_locals: 借用的
// 返回值:
//  - Py_None 或 PyCodeObject: 借用的引用
PyObject* lookup(
    ExtraState* extra_state,
    PyObject* f_locals,
    const PyObject* backend);

// 在extra_state中创建一个新的缓存条目，保存guarded_code。
// 所有权约定
// 参数
//  - extra_state: 借用的
//  - guarded_code: 借用的
//  - callback: 借用的
// 返回值:
//  - cache_entry: 借用的引用
CacheEntry* create_cache_entry(
    ExtraState* extra_state,
    PyObject* guarded_code,
    PyObject* callback);

// 从回调中提取后端函数。
PyObject* get_backend(PyObject* callback);

#ifdef __cplusplus

} // extern "C"

// 返回与code_obj对应的CacheEntry列表。
// 警告: 返回的引用的生命周期由C++控制。
py::list _debug_get_cache_entry_list(const py::handle& code_obj);

#endif
```