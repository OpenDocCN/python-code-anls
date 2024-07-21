# `.\pytorch\torch\csrc\dynamo\extra_state.cpp`

```py
#include <torch/csrc/dynamo/extra_state.h>

#include <torch/csrc/dynamo/cache_entry.h>
#include <torch/csrc/dynamo/cpython_defs.h>
#include <torch/csrc/dynamo/debug_macros.h>
#include <torch/csrc/dynamo/guards.h>
#include <torch/csrc/utils/python_compat.h>

#if IS_PYTHON_3_12_PLUS
#define _PyCode_GetExtra PyUnstable_Code_GetExtra
#define _PyCode_SetExtra PyUnstable_Code_SetExtra
#endif

// 全局变量，用于索引额外状态
Py_ssize_t extra_index = -1;

// 返回第一个缓存条目的指针，如果列表为空则返回空指针
CacheEntry* ExtraState::get_first_entry() {
  if (this->cache_entry_list.empty()) {
    return nullptr;
  }
  return &this->cache_entry_list.front();
}

// 将指定的缓存条目移动到列表的最前面
void ExtraState::move_to_front(CacheEntry* cache_entry) {
  CHECK(cache_entry->_owner == this);  // 检查缓存条目的所有者是否正确
  CHECK(!this->cache_entry_list.empty());  // 检查列表是否为空
  CHECK(cache_entry == &*cache_entry->_owner_loc);  // 检查缓存条目位置的有效性
  this->cache_entry_list.splice(
      this->cache_entry_list.begin(),  // 将条目插入到列表开头
      this->cache_entry_list,
      cache_entry->_owner_loc);
}

// 从列表中使指定的缓存条目失效并删除
void ExtraState::invalidate(CacheEntry* cache_entry) {
  CHECK(cache_entry->_owner == this);  // 检查缓存条目的所有者是否正确
  CHECK(!this->cache_entry_list.empty());  // 检查列表是否为空
  CHECK(cache_entry == &*cache_entry->_owner_loc);  // 检查缓存条目位置的有效性
  this->cache_entry_list.erase(cache_entry->_owner_loc);  // 从列表中删除指定位置的条目
}

// 从额外状态中提取第一个缓存条目，如果额外状态为空或为跳过代码，则返回空指针
CacheEntry* extract_cache_entry(ExtraState* extra_state) {
  if (extra_state == nullptr || extra_state == SKIP_CODE) {
    return nullptr;
  }
  return extra_state->get_first_entry();
}

// 从额外状态中提取帧状态，如果额外状态为空或为跳过代码，则返回空指针
FrameState* extract_frame_state(ExtraState* extra_state) {
  if (extra_state == nullptr || extra_state == SKIP_CODE) {
    return nullptr;
  }
  return (FrameState*)extra_state->frame_state.ptr();
}

// 获取代码对象的额外状态
ExtraState* get_extra_state(PyCodeObject* code) {
  ExtraState* extra = nullptr;
  _PyCode_GetExtra((PyObject*)code, extra_index, (void**)&extra);  // 调用Python C API获取额外状态
  return extra;
}

// 销毁额外状态对象
void destroy_extra_state(void* obj) {
  ExtraState* extra = (ExtraState*)obj;
  if (extra != nullptr && extra != SKIP_CODE) {
    delete extra;
  }
}

// 设置代码对象的额外状态
void set_extra_state(PyCodeObject* code, ExtraState* extra_state) {
  ExtraState* old_extra_state = get_extra_state(code);
  CHECK(
      old_extra_state == nullptr || old_extra_state == SKIP_CODE ||
      old_extra_state != extra_state);  // 检查旧的额外状态是否为空或为跳过代码，且与新状态不同
  _PyCode_SetExtra((PyObject*)code, extra_index, extra_state);  // 调用Python C API设置额外状态
}

// 初始化并设置代码对象的额外状态
ExtraState* init_and_set_extra_state(PyCodeObject* code) {
  // 不变性 - 额外状态在此之前不应该已经设置，因此应该为空指针
  CHECK(get_extra_state(code) == nullptr);
  ExtraState* extra_state = new ExtraState();  // 创建新的额外状态对象
  NULL_CHECK(extra_state);  // 检查创建的对象是否为空
  set_extra_state(code, extra_state);  // 设置代码对象的额外状态
  return extra_state;
}

// 查找符合条件的缓存条目
PyObject* lookup(
    ExtraState* extra_state,
    PyObject* f_locals,
    const PyObject* backend) {
  size_t index = 0;
  CacheEntry* found = nullptr;
  py::handle locals(f_locals);  // 将Python对象转换为C++对象
  for (CacheEntry& cache_entry : extra_state->cache_entry_list) {
    // 检查后端。Py_False表示仅运行模式。
    bool valid = backend == Py_False || cache_entry.backend == backend;
    // 如果 valid 为真，则进入条件判断块
    if (valid) {
      // 尝试执行以下代码块，捕获可能抛出的异常
      try {
        // 如果 cache_entry 的 root_mgr 不为空指针
        if (cache_entry.root_mgr != nullptr) {
          // 运行 root_guard_manager 函数，传入 root_mgr 和 f_locals
          valid = torch::dynamo::run_root_guard_manager(
              cache_entry.root_mgr, f_locals);
        } else {
          // 否则，调用 cache_entry 的 check_fn 函数，并转换结果为布尔值
          valid = cache_entry.check_fn(locals).cast<bool>();
        }
      } catch (py::error_already_set& e) {
        // 如果捕获到 py::error_already_set 异常，则执行以下代码块
        if (guard_error_hook) {
          // 创建 guard_error_hook_handle 对象，传入 guard_error_hook
          py::handle guard_error_hook_handle(guard_error_hook);
          // 调用 guard_error_hook_handle 对象，传入多个参数
          guard_error_hook_handle(
              cache_entry.check_fn,
              cache_entry.code,
              locals,
              index,
              index == extra_state->cache_entry_list.size() - 1);
        }
        // 由于该函数从 C 调用，无法重新抛出异常，因此恢复异常状态
        e.restore();
        // 返回空指针
        return nullptr;
      }
    }
    // 如果 valid 为真，则执行以下代码块
    if (valid) {
      // 将 found 指向 cache_entry
      found = &cache_entry;
      // 跳出循环
      break;
    }
    // index 自增
    ++index;
  }
  // 如果 found 不为空指针，则执行以下代码块
  if (found) {
    // 将 found 移动到 extra_state 的前面位置
    extra_state->move_to_front(found);
    // 返回 found 的 code 指针
    return found->code.ptr();
  }
  // 返回 py::none() 的空指针
  return py::none().ptr();
}

// 创建缓存条目函数，返回指向 CacheEntry 的指针
CacheEntry* create_cache_entry(
    ExtraState* extra_state,         // 额外状态指针
    PyObject* guarded_code,          // 受保护的代码对象指针
    PyObject* backend) {             // 后端对象指针
  // 将新的缓存条目插入到额外状态的缓存条目列表的最前面
  extra_state->cache_entry_list.emplace_front(guarded_code, backend);
  // 获取新插入的缓存条目的迭代器
  auto new_iter = extra_state->cache_entry_list.begin();
  // 将条目的拥有者指向额外状态
  new_iter->_owner = extra_state;
  // 将条目的拥有者位置指向新的迭代器
  new_iter->_owner_loc = new_iter;
  // 设置 check_fn 的引用，注意：生命周期由 C++ 控制！
  py::handle check_fn = py::handle(guarded_code).attr("check_fn");
  // 将 cache_entry 属性设置为新的缓存条目的引用
  check_fn.attr("cache_entry") =
      py::cast(*new_iter, py::return_value_policy::reference);
  // 将 extra_state 属性设置为额外状态的引用
  check_fn.attr("extra_state") =
      py::cast(extra_state, py::return_value_policy::reference);
  // 返回新插入缓存条目的指针
  return &*new_iter;
}

// 调试用函数：获取与给定代码对象关联的缓存条目列表
py::list _debug_get_cache_entry_list(const py::handle& code_obj) {
  // 检查给定对象是否为 CodeType 的实例，否则抛出类型错误
  if (!py::isinstance(code_obj, py::module::import("types").attr("CodeType"))) {
    throw py::type_error("expected a code object!");
  }
  // 将 Python 的代码对象转换为 PyCodeObject 指针
  PyCodeObject* code = (PyCodeObject*)code_obj.ptr();
  // 获取与代码对象关联的额外状态
  ExtraState* extra = get_extra_state(code);
  // 准备返回结果列表
  py::list result;
  // 如果额外状态存在且不是 SKIP_CODE
  if (extra && extra != SKIP_CODE) {
    // 遍历额外状态的缓存条目列表，将每个条目转换为 Python 对象并添加到结果列表中
    for (CacheEntry& e : extra->cache_entry_list) {
      result.append(py::cast(e, py::return_value_policy::reference));
    }
  }
  // 返回最终的结果列表
  return result;
}
```