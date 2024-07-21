# `.\pytorch\torch\csrc\dynamo\python_compiled_autograd.cpp`

```py
// 包含 Torch 的头文件：实现了编译后自动求导的相关功能
#include <torch/csrc/dynamo/python_compiled_autograd.h>

// 包含 Torch 的自动求导引擎相关头文件
#include <torch/csrc/autograd/engine.h>

// 包含 Torch 的自动求导功能：梯度累积
#include <torch/csrc/autograd/functions/accumulate_grad.h>

// 包含 Torch 的动态编译自动求导功能
#include <torch/csrc/dynamo/compiled_autograd.h>

// 包含 Torch 的 JIT Python 绑定工具函数
#include <torch/csrc/jit/python/pybind_utils.h>

// 包含 Torch 的 Python 头文件
#include <torch/csrc/python_headers.h>

// 包含 Torch 的 Python C API 兼容性工具
#include <torch/csrc/utils/pythoncapi_compat.h>

// 包含标准输入输出流
#include <iostream>

// 包含字符串流，用于字符串处理
#include <sstream>

// 包含字符串处理函数
#include <string>

// 包含向量容器
#include <vector>

/*
[注意：编译后自动求导]

编译后自动求导通过将自动求导图转换为可以 torch.compiled 编译的 FX 图来替换标准的自动求导引擎。
它使用影子图缓存这个转换过程。我们通过同时遍历两个图并为每个原始节点计算一个 CacheKey，
来将新图与影子图进行比较。两个不同的图可能在影子图中有一个共享的公共前缀，
但在第一个不同点处分歧。存储在自动求导图节点中的张量、SavedVariables 和 SymInt 被提升为图的输入。
所有其他属性（整数、浮点数、类型等）都使用 CacheKey 进行特化，
如果某些属性不同，则会落在影子图中的不同缓存节点上。

为了与（数百种）不同的 autograd::Node 类型进行交互，我们使用访问者模式递归地遍历每个 Node 结构。

- 第一次遍历（compiled_args/collect）提取图的所有输入并为我们构建一个 CacheKey。在缓存命中时，我们在此停止并且这是唯一的传递。

- 在缓存未命中时，第二次遍历开始提取 FX 图，使用 apply_with_saved 方法，该方法使用另一种访问者模式。
  before() 访问者替换所有张量、SavedVariables 和 SymInt 为假/符号版本以允许跟踪。然后我们运行标准的 apply() 方法，并在 after() 中恢复原状。

当我们看到张量钩子时，我们直接在输出图中记录它们，而不跟踪它们。我们这样做是为了避免在跟踪时执行不安全的代码。

注意：
  - 我们要求钩子不改变张量的形状。
  - 我们要求非钩子的自动求导节点可跟踪。
*/

// 命名空间 torch::dynamo::autograd 中使用 SymInt
namespace torch::dynamo::autograd {
using c10::SymInt;

// 将整数列表包装为 Python 元组对象
static PyObject* wrap_int_list(const std::vector<int64_t>& inputs) {
  PyObject* pyinput = PyTuple_New(static_cast<Py_ssize_t>(inputs.size()));
  for (const auto i : c10::irange(inputs.size())) {
    PyTuple_SET_ITEM(pyinput, i, PyLong_FromSsize_t(inputs[i]));
  }
  return pyinput;
}

// 转换钩子列表为 Python 元组对象
static PyObject* convert_hook_list(std::vector<c10::SafePyObject>& inputs) {
  // 就地操作，消耗输入的钩子
  PyObject* pyinput = PyTuple_New(static_cast<Py_ssize_t>(inputs.size()));
  for (const auto i : c10::irange(inputs.size())) {
    PyTuple_SET_ITEM(pyinput, i, inputs[i].release());
  }
  return pyinput;
}

// 检查 Python 结果对象是否为空
static PyObject* check(PyObject* pyresult) {
  if (C10_UNLIKELY(pyresult == nullptr)) {
    // 参见 https://github.com/pytorch/pytorch/pull/34845
    python_error err;
    err.persist();
    // 调用错误对象的 persist() 方法，用于持久化错误信息或状态

    // NOLINTNEXTLINE(misc-throw-by-value-catch-by-reference)
    // 禁止 Lint 工具检查下一行（misc-throw-by-value-catch-by-reference 规则），因为它涉及通过值抛出异常，而捕获时通过引用捕获异常

    throw err;
    // 抛出捕获到的错误对象 err
  }
  // 返回 Python 结果
  return pyresult;
}

// 静态函数：检查结果是否为 false，若是，则调用 check 函数
static void check(bool result) {
  if (C10_UNLIKELY(!result))
    check(nullptr);
}

// Python 详细日志记录开关的快照
static PyObject* python_verbose_logger = nullptr;

// VerboseLogger 结构体定义
struct VerboseLogger {
  // 可选地创建 VerboseLogger 对象
  static std::optional<VerboseLogger> maybe_create() {
    // 如果 python_verbose_logger 为空，则返回空值
    if (python_verbose_logger == nullptr) {
      return std::nullopt;
    }
    // 否则返回一个新创建的 VerboseLogger 对象
    return VerboseLogger();
  }

  // 记录详细日志的函数，接受消息字符串参数
  void verbose_log_fn(std::string_view msg) const {
    // 断言 python_verbose_logger 不为空
    TORCH_CHECK(python_verbose_logger != nullptr);
    // 调用 Python 函数 python_verbose_logger 记录消息
    check(PyObject_CallFunction(python_verbose_logger, "s", msg.data()));
  }

  // 记录节点检查日志，接受节点、输入尺寸、缓存键集合、键和节点索引参数
  void log_node_check(
      const Node& fn,
      size_t size_inputs_num,
      std::unordered_set<CacheKey> cached_keys,
      const CacheKey& key,
      size_t node_idx) {
    // 构建节点名称字符串
    std::string node_name =
        fn.name() + " (NodeCall " + std::to_string(node_idx) + ")";
    
    // 将节点名称与输入尺寸映射存储
    cumulative_sizes_per_node[size_inputs_num] = node_name;

    // 如果尚未记录节点缓存未命中且缓存键集合中不包含当前键
    if (!logged_node_miss && cached_keys.find(key) == cached_keys.end()) {
      // 记录节点缓存未命中情况
      _log_node_miss(typeid(fn), cached_keys, key, node_name);
      logged_node_miss = true;
    }
  }

  // 记录节点缓存未命中的私有方法，接受节点类型信息、缓存键集合、键和节点名称参数
  void _log_node_miss(
      const std::type_info& node_type,
      std::unordered_set<CacheKey> cached_keys,
      const CacheKey& key,
      const std::string& node_name) const {
    // 创建输出字符串流
    std::ostringstream oss;
    // 拼接缓存未命中的详细信息
    oss << "Cache miss due to new autograd node: " << node_name
        << " with key size " << std::to_string(key.key_size)
        << ", previous key sizes=[";

    // 遍历缓存键集合，拼接相应节点类型的键尺寸
    for (auto it = cached_keys.begin(); it != cached_keys.end(); it++) {
      if (it->node_type != node_type) {
        continue;
      }
      oss << it->key_size;
      if (std::next(it) != cached_keys.end()) {
        oss << ",";
      }
    }
    oss << "]";
    // 调用 verbose_log_fn 记录详细日志
    verbose_log_fn(oss.str());
  }

  // 记录动态形状检查的方法，接受尺寸索引参数
  void log_dynamic_shapes_check(size_t size_idx) const {
    // 如果累计节点尺寸映射为空，直接返回
    if (cumulative_sizes_per_node.empty()) {
      return;
    }

    // 在累计节点尺寸映射中查找给定尺寸索引的位置迭代器
    auto it = cumulative_sizes_per_node.lower_bound(size_idx);
    // 断言找到有效迭代器
    TORCH_CHECK(it != cumulative_sizes_per_node.end());
    // 获取起始尺寸索引
    size_t start_idx =
        it == cumulative_sizes_per_node.begin() ? 0 : std::prev(it)->first;
    // 记录由于形状变化导致的缓存未命中情况
    verbose_log_fn(
        "Cache miss due to changed shapes: marking size idx " +
        std::to_string(size_idx - start_idx) + " of " + it->second +
        " as dynamic");
  }

  // 跟踪尺寸索引与节点的映射关系
  std::map<size_t, std::string> cumulative_sizes_per_node;
  // 记录节点缓存未命中的标志，仅记录一次
  bool logged_node_miss = false;
};

// 缓存节点结构体定义
struct CacheNode {
  // 阴影图中的节点，通过下一个边进行遍历直至图的末端
  static CacheNode* root() {
    // 静态方法，返回根节点指针
    static CacheNode _root;
    return &_root;
  }

  // 查找给定缓存键的节点，可选创建节点
  CacheNode* lookup(const CacheKey& key, bool create = true) {
    // 在 next 映射中查找给定键的节点
    auto it = next.find(key);
    ```
    if (it == next.end()) {
      // 如果迭代器 it 指向 next 的末尾，表示未找到对应条目
      if (!create)
        // 如果不允许创建新条目，则返回空指针
        return nullptr;
      // 在临时内存中复制调用者的键
      CacheKeyBuffer buffer(key.key, key.key_size);
      // 创建带有内存存储的新键
      CacheKey key_with_storage(key.node_type, buffer.get(), key.key_size);
      // 将新键和新创建的缓存节点插入到 next 中
      it = next.emplace(key_with_storage, std::make_unique<CacheNode>()).first;
      // 将 buffer 移动到 key_storage 中保留
      key_storage.emplace_back(std::move(buffer));
    }
    // 返回迭代器指向的节点的指针
    return it->second.get();
  }

  void clear() {
    // 清空 next 中的所有条目
    next.clear();
    // 清空 key_storage 中的所有条目
    key_storage.clear();
    // 清空 expected_sizes 中的所有条目
    expected_sizes.clear();
    // 将 runtime_wrapper 置空
    runtime_wrapper = nullptr;
    // 将 compiled_fn 置空
    compiled_fn = nullptr;
  }

  bool is_empty() const {
    // 检查 next 是否为空并且 compiled_fn 是否为空
    return next.empty() && !compiled_fn;
  }

  CacheNode() : runtime_wrapper(nullptr), compiled_fn(nullptr) {}
  ~CacheNode() {
    // 如果 Python 解释器未初始化，释放 runtime_wrapper 和 compiled_fn
    if (!Py_IsInitialized()) {
      // 在关闭时泄漏资源
      runtime_wrapper.release();
      compiled_fn.release();
    }
  }
  CacheNode(CacheNode&&) = delete;
  CacheNode(const CacheNode&) = delete;
  CacheNode& operator=(const CacheNode&) = delete;
  CacheNode& operator=(CacheNode&&) = delete;

  bool check_dynamic_sizes(
      AutogradCompilerCall& call,
      const std::optional<VerboseLogger>& vlogger) {
    /*
    我们开始假设所有内容都是静态的，然后在发现更改时标记为动态。该函数：
      1) 检查是否命中缓存
      2) 更新 expected_sizes 以跟踪动态大小
      3) 通过筛选 call.all_size_inputs 来填充 call.dyn_size_inputs
    */
    // 缓存是否命中的标志，初始化为 compiled_fn 不为空
    bool cache_hit = compiled_fn.get() != nullptr;
    // 获取输入的长度
    auto len = call.all_size_inputs.size();
    // 获取指向输入数据的指针
    const SizeInput* data = call.all_size_inputs.data();
    // 如果 expected_sizes 为空，则初始化并预留空间
    if (expected_sizes.empty()) {
      expected_sizes.reserve(len);
      // 遍历输入数据，将其添加到 expected_sizes 中
      for (const auto i : c10::irange(len)) {
        expected_sizes.emplace_back(data[i]);
      }
    }

    // 断言 expected_sizes 和 call.all_size_inputs 的大小相同
    TORCH_INTERNAL_ASSERT(expected_sizes.size() == call.all_size_inputs.size());
    // 遍历所有输入数据
    for (const auto i : c10::irange(len)) {
      auto& expected = expected_sizes[i];
      // 记录之前是否为动态大小
      bool was_dynamic = expected.dyn_type == SizeInput::DYNAMIC;
      // 检查值是否发生变化
      bool changed_value = expected.value != data[i].value;
      if (changed_value) {
        // 如果值发生变化且之前不是动态大小，则表示缓存未命中
        if (!was_dynamic) {
          cache_hit = false;
          // 如果提供了 vlogger，则记录动态形状检查
          if (vlogger.has_value()) {
            vlogger->log_dynamic_shapes_check(i);
          }
        }
        // 更新 expected_sizes 中的值为动态大小
        expected = SizeInput(SizeInput::DYNAMIC, data[i].value);
      }

      // 如果值发生变化或之前为动态大小，则将其添加到 dyn_size_inputs 中
      if (changed_value || was_dynamic) {
        if (call.dyn_size_inputs.empty()) {
          call.dyn_size_inputs.reserve(len);
        }
        call.dyn_size_inputs.emplace_back(data[i].value);
      }
    }

    // 如果缓存未命中，将 runtime_wrapper 和 compiled_fn 置空
    if (!cache_hit) {
      // 因为静态大小输入不匹配而未命中缓存；使用变化的大小输入强制重新编译
      runtime_wrapper = nullptr;
      compiled_fn = nullptr;
    }
    return cache_hit;
  }

  PyObject* wrap_dynamic_inputs() const {
    // 计算动态输入的数量
    size_t dynamic_count = 0;
    // 索引计数器初始化为 0
    size_t idx = 0;
    // 遍历 expected_sizes 向量中的每个元素
    for (const auto& i : expected_sizes) {
      // 如果元素的 dyn_type 属性为 SizeInput::DYNAMIC，则增加 dynamic_count 计数器
      if (i.dyn_type == SizeInput::DYNAMIC) {
        ++dynamic_count;
      }
    }
    // 创建一个 PyTuple 对象，其长度为 dynamic_count
    PyObject* pyinput = PyTuple_New(static_cast<Py_ssize_t>(dynamic_count));
    // 再次遍历 expected_sizes 向量中的每个元素
    for (const auto& i : expected_sizes) {
      // 如果元素的 dyn_type 属性为 SizeInput::DYNAMIC
      if (i.dyn_type == SizeInput::DYNAMIC) {
        // 将 i.value 转换为 PyLong 对象，并设置为 PyTuple 中的第 idx 个元素
        PyTuple_SET_ITEM(pyinput, idx++, PyLong_FromSsize_t(i.value));
      }
    }
    // 断言 idx 的值与 dynamic_count 相等，确保所有动态元素都已处理
    TORCH_INTERNAL_ASSERT(idx == dynamic_count);
    // 返回创建的 PyTuple 对象，其中包含了动态尺寸的数据
    return pyinput;
  }

  // 解封装动态输入数据
  std::vector<std::optional<SymInt>> unwrap_dynamic_inputs(
      PyObject* pyresult) const {
    // 断言 pyresult 是一个 PyList 对象
    TORCH_INTERNAL_ASSERT(PyList_CheckExact(pyresult));
    size_t idx = 0;
    // 获取 PyList 对象的长度
    size_t result_len = PyList_GET_SIZE(pyresult);
    // 创建一个用于存储结果的 vector，预留空间大小为 expected_sizes 向量的大小
    std::vector<std::optional<SymInt>> result;
    result.reserve(expected_sizes.size());
    // 遍历 expected_sizes 向量中的每个元素
    for (const auto& i : expected_sizes) {
      // 如果元素的 dyn_type 属性为 SizeInput::DYNAMIC
      if (i.dyn_type == SizeInput::DYNAMIC) {
        // 断言 idx 小于 result_len，确保索引在有效范围内
        TORCH_INTERNAL_ASSERT(idx < result_len);
        // 将 PyList 中第 idx 个元素转换为 c10::SymInt 类型，并添加到 result 向量中
        result.emplace_back(
            py::cast<c10::SymInt>(PyList_GET_ITEM(pyresult, idx++)));
      } else {
        // 如果不是动态尺寸，则在 result 向量中添加一个空的 std::optional<SymInt> 对象
        result.emplace_back();
      }
    }
    // 断言 idx 等于 result_len，确保所有动态元素都已处理，并且 result 的大小与 expected_sizes 向量的大小相等
    TORCH_INTERNAL_ASSERT(
        idx == result_len && result.size() == expected_sizes.size());
    // 返回包含动态输入数据的结果向量
    return result;
  }

  // 缓存下一步操作的数据结构
  std::unordered_map<CacheKey, std::unique_ptr<CacheNode>> next;
  // 存储缓存键的缓冲区
  std::vector<CacheKeyBuffer> key_storage;
  // 期望的输入尺寸信息
  std::vector<SizeInput> expected_sizes;

  // 运行时包装器对象指针
  THPObjectPtr runtime_wrapper;
  // 编译后的函数对象指针
  THPObjectPtr compiled_fn;
};

// 用于存储每个节点的输入缓冲区的映射关系，继承自无序映射 std::unordered_map<Node*, InputBuffer>
struct InputBuffers : public std::unordered_map<Node*, InputBuffer> {
  // 查找指定节点的输入缓冲区，如果不存在则创建一个新的缓冲区
  InputBuffer& lookup(Node* function) {
    auto it = emplace(function, InputBuffer(function->num_inputs())).first;
    return it->second;
  }
};

// 静态变量，指向自动求导编译器对象的 Python 对象
static PyObject* the_autograd_compiler = nullptr;

// 设置自动求导编译器的 Python 函数接口
static PyObject* set_autograd_compiler(PyObject* dummy, PyObject* args);

// 清空缓存节点的缓存内容的 Python 函数接口
static PyObject* clear_cache(PyObject* dummy, PyObject* args) {
  HANDLE_TH_ERRORS; // 异常处理块开始
  CacheNode::root()->clear(); // 调用缓存根节点的清空方法
  Py_RETURN_NONE; // 返回 Python 的 None 对象
  END_HANDLE_TH_ERRORS; // 异常处理块结束
}

// 检查缓存节点是否为空的 Python 函数接口
static PyObject* is_cache_empty(PyObject* dummy, PyObject* args) {
  HANDLE_TH_ERRORS; // 异常处理块开始
  // 如果缓存根节点为空，则返回 Python 的 True 对象；否则返回 False 对象
  if (CacheNode::root()->is_empty()) {
    Py_RETURN_TRUE;
  }
  Py_RETURN_FALSE;
  END_HANDLE_TH_ERRORS; // 异常处理块结束
}

// 设置详细日志记录器的 Python 函数接口
static PyObject* set_verbose_logger(PyObject* dummy, PyObject* args) {
  HANDLE_TH_ERRORS; // 异常处理块开始
  PyObject* logger = nullptr; // Python 日志记录器对象
  // 解析 Python 函数参数，如果解析失败则返回 False 对象
  if (!PyArg_ParseTuple(args, "O", &logger)) {
    Py_RETURN_FALSE;
  }

  // 如果 logger 是 None，则将全局变量 python_verbose_logger 置为空
  if (logger == Py_None) {
    python_verbose_logger = nullptr;
  } else {
    python_verbose_logger = logger; // 否则设置为传入的 logger 对象
  }
  Py_RETURN_TRUE; // 返回 Python 的 True 对象表示成功
  END_HANDLE_TH_ERRORS; // 异常处理块结束
}

// Python 方法定义数组，包含各个函数接口的映射关系
// NOLINTNEXTLINE(*array*) 是指示代码检查工具忽略对数组的某些规则检查
static PyMethodDef _methods[] = {
    {"set_autograd_compiler", set_autograd_compiler, METH_VARARGS, nullptr}, // 设置自动求导编译器函数接口
    {"clear_cache", clear_cache, METH_NOARGS, nullptr}, // 清空缓存函数接口
    {"is_cache_empty", is_cache_empty, METH_NOARGS, nullptr}, // 检查缓存是否为空函数接口
    {"set_verbose_logger", set_verbose_logger, METH_VARARGS, nullptr}, // 设置详细日志记录器函数接口
    {nullptr, nullptr, 0, nullptr} // 结束符号，表示数组结束
};

// Python 模块定义结构体，描述了模块的基本信息和方法映射
static struct PyModuleDef _module = {
    PyModuleDef_HEAD_INIT, // 初始化 Python 模块定义结构体
    "torch._C._dynamo.autograd_compiler", // 模块名
    "Hooks for compiling autograd", // 模块文档字符串
    -1, // 模块状态，-1 表示不可子解析
    _methods // 模块包含的方法映射数组
};

// 开始捕获调用时的跟踪状态
static TraceState call_begin_capture(
    PyObject* self,
    CacheNode& cache,
    AutogradCompilerCall& compiler_call,
    size_t num_outputs) {
  static PyObject* method_name = PyUnicode_InternFromString("begin_capture"); // 方法名
  THPObjectPtr pyinput(THPVariable_WrapList(compiler_call.tensor_args.inputs)); // 将输入封装为 Python 对象
  THPObjectPtr pysizeinput(cache.wrap_dynamic_inputs()); // 包装动态输入为 Python 对象
  THPObjectPtr pyresult(check(PyObject_CallMethodObjArgs(
      self, method_name, pyinput.get(), pysizeinput.get(), nullptr))); // 调用 Python 方法获取结果

  PyObject *fake_inputs{nullptr}, *fake_sizes{nullptr};
  // 解析 Python 方法返回的结果元组
  check(PyArg_ParseTuple(pyresult.get(), "OO", &fake_inputs, &fake_sizes));

  // 解封装假输入为变量列表
  variable_list proxy_inputs = THPVariable_UnpackList(fake_inputs);
  // 断言代理输入的数量与实际输入的数量相等
  TORCH_INTERNAL_ASSERT(
      proxy_inputs.size() == compiler_call.tensor_args.inputs.size());
  // 遍历代理输入列表，为每个输入设置代理张量
  for (const auto i : c10::irange(proxy_inputs.size())) {
    TensorArg& arg =
        compiler_call.tensor_args.lookup(compiler_call.tensor_args.inputs[i]);
    arg.proxy_tensor = proxy_inputs[i];
  }

  // 解封装动态大小的输入并返回跟踪状态对象
  return TraceState(cache.unwrap_dynamic_inputs(fake_sizes), num_outputs);
}

// 结束捕获调用时的跟踪状态，并返回 Python 对象结果
static PyObject* call_end_capture(PyObject* self, const variable_list& inputs) {
  static PyObject* method_name = PyUnicode_InternFromString("end_capture"); // 方法名
  THPObjectPtr pyinput(THPVariable_WrapList(inputs)); // 将输入封装为 Python 对象
  return check(PyObject_CallMethodOneArg(self, method_name, pyinput.get())); // 调用 Python 方法并返回结果
}
// 定义一个结构体，继承自THPObjectPtr，用于自动关闭持有的Python对象
struct ClosingTHPObjectPtr : public THPObjectPtr {
  ClosingTHPObjectPtr(PyObject* o) : THPObjectPtr(o) {}  // 初始化基类THPObjectPtr
  ~ClosingTHPObjectPtr() {
    if (PyErr_Occurred()) {
      // 如果出现异常，不进行关闭操作，直接返回
      return;
    }
    // 静态变量，表示方法名"close"
    static PyObject* method_name = PyUnicode_InternFromString("close");
    // 调用get()方法获取对象，并尝试调用其close()方法
    if (PyObject_CallMethodNoArgs(get(), method_name) == nullptr) {
      PyErr_WriteUnraisable(get());  // 如果调用失败，记录异常信息
      PyErr_Clear();  // 清除异常状态
    }
  }
};

// 仅在持有全局解释器锁(GIL)的情况下调用此函数
CacheNode* _compiled_autograd_impl(
    const std::shared_ptr<Node>& graph_root,  // 图的根节点
    GraphTask& graph_task,  // 图任务对象的引用
    bool accumulate_grad,  // 是否累积梯度
    const edge_list& output_edges,  // 输出边的列表
    THPObjectPtr* graph_arg_inputs,  // 图参数输入的Python对象指针
    THPObjectPtr* graph_arg_sizes,  // 图参数大小的Python对象指针
    THPObjectPtr* graph_arg_hooks) {  // 图参数钩子的Python对象指针

  std::unordered_map<Node*, int>& dependencies = graph_task.dependencies_;  // 节点依赖关系的映射
  std::vector<std::shared_ptr<Node>> worklist{graph_root};  // 工作列表初始化为图的根节点
  AutogradCompilerCall compiler_call;  // 自动求导编译器调用对象

  for (const auto i : c10::irange(output_edges.size())) {
    compiler_call.node_calls
        .lookup(output_edges[i].function)
        .mark_output(output_edges[i].input_nr, i);  // 标记输出边对应的节点和端口号
  }
  const bool check_exec_info = !graph_task.exec_info_.empty();  // 检查执行信息是否为空
  CacheNode* cache = CacheNode::root();  // 缓存节点初始化为根节点
  std::vector<NodeCall*> calls;  // 节点调用列表
  calls.reserve(
      check_exec_info ? graph_task.exec_info_.size() : dependencies.size() + 1);  // 预留空间以容纳节点调用信息

  int i = 0;  // 计数器初始化
  std::optional<VerboseLogger> vlogger = VerboseLogger::maybe_create();  // 可选的详细日志记录器
  while (!worklist.empty()) {  // 在工作列表非空的情况下循环
    std::shared_ptr<Node> fn = std::move(worklist.back());  // 获取工作列表的最后一个节点
    worklist.pop_back();  // 弹出工作列表的最后一个节点
    NodeCall& call = compiler_call.node_calls.lookup(fn);  // 查找节点对应的调用信息
    calls.emplace_back(&call);  // 将调用信息添加到节点调用列表中

    { // 更新缓存并将参数收集到`compiler_call`中
      CompiledNodeArgs node_args(compiler_call, call);  // 编译节点参数对象
      node_args.collect(call);  // 收集节点的参数
      if (node_args.cond(call.needed)) {  // 如果满足条件需要节点编译参数
        fn->compiled_args(node_args);  // 设置节点的编译参数
        node_args.collect(call.node->next_edges());  // 收集节点的下一个边信息
      }
      CacheKey key = node_args.key();  // 获取节点参数的键值
      if (vlogger.has_value()) {  // 如果详细日志记录器存在
        std::unordered_set<CacheKey> cached_keys;  // 缓存键的无序集合
        for (const auto& [k, _] : cache->next) {  // 遍历缓存的下一个节点
          cached_keys.emplace(k);  // 添加到缓存键的集合中
        }
        vlogger->log_node_check(
            *fn,
            compiler_call.all_size_inputs.size(),  // 编译器调用的所有大小输入
            std::move(cached_keys),  // 移动缓存键的集合
            key,  // 键值
            i);  // 计数器
      }
      cache = cache->lookup(key);  // 查找缓存中的键
    }
    // 遍历函数fn的下一条边集合
    for (const auto& edge : fn->next_edges()) {
      // 如果边不是有效的，则跳过本次循环
      if (!edge.is_valid()) {
        continue;
      }
      // 如果需要检查执行信息
      if (check_exec_info) {
        // 在图任务的执行信息中查找与当前边对应的函数指针
        auto it = graph_task.exec_info_.find(edge.function.get());
        // 如果找不到对应的执行信息或者不需要执行，则跳过本次循环
        if (it == graph_task.exec_info_.end() || !it->second.should_execute()) {
          continue;
        }
        // 如果不需要当前执行信息，则将编译器调用中对应节点的调用标记设置为false
        if (!it->second.needed_) {
          compiler_call.node_calls.lookup(edge.function).needed = false;
        }
      }
      // 在依赖关系中查找当前边对应的函数指针
      auto it = dependencies.find(edge.function.get());
      // 断言确保能够找到该依赖关系
      TORCH_INTERNAL_ASSERT(it != dependencies.end());
      // 减少当前函数的依赖计数，如果计数减为0，则从依赖关系中移除并将该函数加入工作列表
      if (--it->second == 0) {
        dependencies.erase(it);
        worklist.emplace_back(edge.function);
      }
    }
    // 增加循环变量i的值
    i++;
  }

  // TODO(jansel): some dynamic sizes seem to be ints not symints
  // 检查缓存中的动态尺寸是否与预期一致，如果不一致则执行以下操作
  if (!cache->check_dynamic_sizes(compiler_call, vlogger)) {
    // 缓存未命中，需要捕获FX图
    ClosingTHPObjectPtr py_compiler(
        check(PyObject_CallNoArgs((the_autograd_compiler))));

    // 开始捕获过程，并返回捕获的状态
    TraceState state = call_begin_capture(
        py_compiler, *cache, compiler_call, output_edges.size());
    InputBuffers input_buffers;

    // 执行捕获结束操作，并返回捕获结果
    PyObject* res = check(call_end_capture(py_compiler, state.outputs));
    TORCH_CHECK(PyTuple_Check(res), "Expected end_capture to return tuple");
    TORCH_CHECK(
        PyTuple_Size(res) == 2,
        "Expected end_capture to return tuple of size 2");
    // 将运行时包装器赋值给缓存
    cache->runtime_wrapper = Py_NewRef(PyTuple_GetItem(res, 0));
    TORCH_CHECK(
        PyCallable_Check(cache->runtime_wrapper),
        "Expected end_capture to return runtime_wrapper");
    // 将编译后的函数赋值给缓存
    cache->compiled_fn = Py_NewRef(PyTuple_GetItem(res, 1));
    TORCH_CHECK(
        PyCallable_Check(cache->compiled_fn),
        "Expected end_capture to return compiled_fn");
    // 断言调试状态
    state.debug_asserts();
  } // End cache miss region

  // TODO(jansel): clear grads we will overwrite below
  // 如果图任务不需要保留图结构，则清除将要覆盖的梯度
  if (!graph_task.keep_graph_) {
    for (auto& call : calls) {
      call->node->release_variables();
    }
  }

  // 将编译器调用的张量参数输入包装成THPVariable列表返回
  *graph_arg_inputs = THPVariable_WrapList(compiler_call.tensor_args.inputs);
  // 将编译器调用的动态尺寸输入包装成整数列表返回
  *graph_arg_sizes = wrap_int_list(compiler_call.dyn_size_inputs);
  // 将编译器调用的钩子列表转换为graph_arg_hooks返回
  *graph_arg_hooks = convert_hook_list(compiler_call.hooks);
  // 返回缓存对象指针
  return cache;
}

// 定义编译自动微分函数，接受图的根节点、图任务、是否累积梯度、输出边列表作为参数
variable_list compiled_autograd(
    const std::shared_ptr<Node>& graph_root,
    GraphTask& graph_task,
    bool accumulate_grad,
    const edge_list& output_edges) {
  // 检查Torch调度模式TLS的栈长度是否为0，如果不是则抛出异常
  TORCH_CHECK(
      c10::impl::TorchDispatchModeTLS::stack_len() == 0,
      "TorchDispatchMode not yet implemented for compiled autograd")
  // 静态互斥锁，用于保护共享资源
  static std::mutex lock;
  // 锁定互斥锁
  std::lock_guard<std::mutex> lock_guard(lock);
  // 获取全局解释器锁，确保线程安全
  pybind11::gil_scoped_acquire gil;
  // 线程本地状态的管理器
  at::ThreadLocalStateGuard tls_guard(graph_task.thread_locals_);

  // 定义输入、大小和钩子对象指针
  THPObjectPtr inputs;
  THPObjectPtr sizes;
  THPObjectPtr hooks;
  // 调用编译自动微分实现函数，返回缓存节点指针
  CacheNode* cache = _compiled_autograd_impl(
      graph_root,
      graph_task,
      accumulate_grad,
      output_edges,
      &inputs,
      &sizes,
      &hooks);

  // 调用缓存的运行时包装对象和编译函数执行，返回Python对象指针
  THPObjectPtr pyresult(check(PyObject_CallFunctionObjArgs(
      cache->runtime_wrapper.get(),
      cache->compiled_fn.get(),
      inputs.get(),
      sizes.get(),
      hooks.get(),
      NULL)));
  // 解包Python对象为变量列表
  variable_list outputs = THPVariable_UnpackList(pyresult);
  // 内部断言确保输出大小与输出边列表大小一致
  TORCH_INTERNAL_ASSERT(outputs.size() == output_edges.size());
  // 返回变量列表作为输出
  return outputs;
}

// 设置自动微分编译器对象的Python函数接口
static PyObject* set_autograd_compiler(PyObject* dummy, PyObject* args) {
  // 处理Python函数参数解析错误
  HANDLE_TH_ERRORS;
  // 初始化Python对象指针为空
  PyObject* obj = nullptr;
  // 解析Python元组参数，期望一个对象参数
  if (!PyArg_ParseTuple(args, "O", &obj)) {
    return nullptr;
  }

  // 保存之前的自动微分编译器对象
  PyObject* prior = the_autograd_compiler;
  // 如果传入的对象为None，则禁用编译器
  if (obj == Py_None) { // disable
    the_autograd_compiler = nullptr; // 由于有`prior`在，不需要减少引用计数
    // 设置引擎的编译自动微分对象为null
    Engine::set_compiled_autograd(nullptr);
  } else { // enable
    // 增加Python对象的引用计数
    Py_INCREF(obj);
    // 设置全局自动微分编译器对象为传入的对象
    the_autograd_compiler = obj;
    // 设置引擎的编译自动微分对象为compiled_autograd函数
    Engine::set_compiled_autograd(&compiled_autograd);
  }

  // 如果之前的编译器对象为null，则返回Python的None对象
  if (prior == nullptr) {
    Py_RETURN_NONE;
  } else {
    // 否则返回之前的编译器对象
    return prior;
  }
  // 结束处理Torch错误
  END_HANDLE_TH_ERRORS;
}

// 初始化Torch动态编译自动微分模块的Python接口函数
PyObject* torch_c_dynamo_compiled_autograd_init() {
  return PyModule_Create(&_module);
}

// 结束torch::dynamo::autograd命名空间
} // namespace torch::dynamo::autograd
```