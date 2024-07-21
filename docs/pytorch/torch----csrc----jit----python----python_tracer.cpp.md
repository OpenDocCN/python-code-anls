# `.\pytorch\torch\csrc\jit\python\python_tracer.cpp`

```
// 引入 Torch 的 Python 头文件，定义了与 Python 交互的接口
#include <torch/csrc/python_headers.h>

// 引入 Torch JIT 前端跟踪器相关的头文件
#include <torch/csrc/jit/frontend/tracer.h>
// 引入 Torch JIT 无用代码消除相关的头文件
#include <torch/csrc/jit/passes/dead_code_elimination.h>
// 引入 Torch JIT 内联器相关的头文件
#include <torch/csrc/jit/passes/inliner.h>
// 引入 Torch JIT 降低元组相关的头文件
#include <torch/csrc/jit/passes/lower_tuples.h>
// 引入 Torch JIT Python 绑定相关的头文件
#include <torch/csrc/jit/python/pybind.h>
// 引入 Torch JIT Python 跟踪器相关的头文件
#include <torch/csrc/jit/python/python_tracer.h>
// 引入 Torch JIT 序列化导出相关的头文件
#include <torch/csrc/jit/serialization/export.h>
// 引入 Torch 工具类 Python 字符串处理相关的头文件
#include <torch/csrc/utils/python_strings.h>

// 引入 C10 库异常处理相关的头文件
#include <c10/util/Exception.h>
// 引入 C10 库范围迭代器相关的头文件
#include <c10/util/irange.h>

// 引入标准字符串流库
#include <sstream>

// 使用 Torch 自动求导命名空间
using namespace torch::autograd;
// 使用 Torch JIT 命名空间
using namespace torch::jit;
// 使用 Torch JIT 跟踪器命名空间
using namespace torch::jit::tracer;

// Torch JIT 跟踪器命名空间实现

// 自定义 Python 调用堆栈获取函数，根据当前 Python 解释器帧来获取调用堆栈
std::vector<StackEntry> _pythonCallstack() {
  pybind11::gil_scoped_acquire gil; // 获取全局解释器锁
  PyFrameObject* frame = PyEval_GetFrame(); // 获取当前 Python 帧对象
  Py_XINCREF(frame); // 增加帧对象的引用计数
  std::vector<StackEntry> entries; // 存储调用堆栈条目的向量

  while (nullptr != frame) { // 循环遍历帧对象直到为空
    auto code = THPCodeObjectPtr(PyFrame_GetCode(frame)); // 获取帧对象的代码对象
    size_t line = PyCode_Addr2Line(code.get(), PyFrame_GetLasti(frame)); // 将地址转换为行号
    std::string filename = THPUtils_unpackString(code->co_filename); // 解压缩文件名字符串
    std::string funcname = THPUtils_unpackString(code->co_name); // 解压缩函数名字符串
    auto source = std::make_shared<Source>(funcname, filename, line); // 创建共享的源对象
    entries.emplace_back(
        StackEntry{funcname, SourceRange(source, 0, funcname.size())}); // 添加堆栈条目
    auto new_frame = PyFrame_GetBack(frame); // 获取前一个帧对象
    Py_DECREF(frame); // 减少帧对象的引用计数
    frame = new_frame; // 更新帧对象为新帧对象
  }
  return entries; // 返回调用堆栈条目向量
}

// 获取 Python 解释器源代码范围的函数
SourceRange getPythonInterpreterSourceRange() {
  auto cs = pythonCallstack(); // 调用 Python 调用堆栈获取函数获取堆栈条目向量
  std::optional<std::string> source_filename; // 可选的源文件名字符串
  size_t source_line = 0; // 源代码行号初始化为0
  std::stringstream stack_trace; // 字符串流用于构建堆栈跟踪信息
  for (const auto& entry : cs) { // 遍历堆栈条目向量
    auto& range = entry.range; // 获取条目的范围
    if (range.source()) { // 如果范围存在
      auto& src = range.source(); // 获取源对象
      if (src && src->filename()) { // 如果源对象和文件名存在
        auto line =
            src->starting_line_no() + src->lineno_for_offset(range.start()); // 计算行号
        stack_trace << *(src->filename()) << "(" << line
                    << "): " << entry.filename << "\n"; // 构建堆栈跟踪信息字符串
        if (!source_filename) { // 如果源文件名不存在
          source_filename = *(src->filename()); // 设置源文件名
          source_line = line; // 设置源行号
        }
      }
    }
  }

  auto stack_trace_text = stack_trace.str(); // 获取堆栈跟踪信息的字符串
  auto source =
      std::make_shared<Source>(stack_trace_text, source_filename, source_line); // 创建共享的源对象
  return SourceRange(source, 0, stack_trace_text.size()); // 返回源范围对象
}

// 创建使用字典跟踪的图形和堆栈的函数
std::pair<std::shared_ptr<Graph>, Stack> createGraphByTracingWithDict(
    const py::function& func,
    const py::dict& inputs_dict,
    Stack trace_inputs,
    const py::function& var_name_lookup_fn,
    bool strict,
    bool force_outplace,
    Module* self,
    const std::vector<std::string>& argument_names) {
  C10_LOG_API_USAGE_ONCE("torch.tracer"); // 记录 Torch 跟踪器 API 使用

  // 查找函数适配器使用变量名查找函数
  auto lookup_fn_adapter =
      [var_name_lookup_fn](const Variable& var) -> std::string {
    pybind11::gil_scoped_acquire ag; // 获取全局解释器锁
  // 将变量 var 传递给 var_name_lookup_fn 函数，并将结果转换为 std::string 后返回
  return py::cast<std::string>(var_name_lookup_fn(var));
};

// 解析参数 argument_names 在 Python 中的内容，并且其顺序与 forward() 方法中参数的声明顺序相同。
// 这些名称将作为调试名称添加到图形中，并且顺序应与我们通过 Python 字典生成的可追踪堆栈一致。
std::vector<std::string> compact_argument_names;
Stack compact_trace_inputs;
for (const auto& argument_name : argument_names) {
  // 如果 inputs_dict 包含 argument_name，则将其添加到 compact_argument_names 中
  if (inputs_dict.contains(argument_name)) {
    compact_argument_names.push_back(argument_name);
  }
}

// 遍历 compact_argument_names 中的每个紧凑参数名称
for (const auto& compact_argument_name : compact_argument_names) {
  // 遍历 inputs_dict 的每个键值对
  for (auto it = inputs_dict.begin(); it != inputs_dict.end(); it++) {
    // 如果键的字符串表示与 compact_argument_name 相等，则将其转换为 IValue，并添加到 compact_trace_inputs 中
    if (py::cast<std::string>(it->first) == compact_argument_name) {
      compact_trace_inputs.push_back(
          toIValue(it->second, tryToInferType(it->second).type()));
    }
  }
}

// 使用 tracer::trace 方法对紧凑的追踪输入进行跟踪
auto outs = tracer::trace(
    std::move(compact_trace_inputs),
    [&](Stack inputs) -> Stack {
      // 将 inputs_dict 不变地传递给 forward 方法
      auto out = func(**inputs_dict);
      // 如果 out 为 Py_None，则抛出错误，指出追踪的函数未返回任何值
      if (out.ptr() == Py_None) {
        AT_ERROR(
            "The traced function didn't return any values! Side-effects are not "
            "captured in traces, so it would be a no-op.");
      }
      // 返回包含 toTypeInferredIValue(out) 的单元素列表
      return {toTypeInferredIValue(out)};
    },
    lookup_fn_adapter,
    strict,
    force_outplace,
    self,
    compact_argument_names);

// 返回包含 outs 的第一个和第二个元素的 pair，分别为其图形和结果
return std::make_pair(std::get<0>(outs)->graph, std::get<1>(outs));
}

// 创建图形跟踪的函数，根据给定的Python函数和参数进行跟踪
std::pair<std::shared_ptr<Graph>, Stack> createGraphByTracing(
    const py::function& func,                   // 要跟踪的Python函数
    Stack trace_inputs,                         // 跟踪输入的栈
    const py::function& var_name_lookup_fn,     // 变量名称查找函数
    bool strict,                                // 是否严格模式
    bool force_outplace,                        // 是否强制非原地操作
    Module* self,                               // 自身模块指针
    const std::vector<std::string>& argument_names) {  // 参数名称列表

  // 记录API使用情况
  C10_LOG_API_USAGE_ONCE("torch.tracer");

  // 适配变量名称查找函数的闭包
  auto lookup_fn_adapter =
      [var_name_lookup_fn](const Variable& var) -> std::string {
    pybind11::gil_scoped_acquire ag;
    return py::cast<std::string>(var_name_lookup_fn(var));
  };

  // 调用跟踪器的trace函数进行跟踪
  auto outs = tracer::trace(
      std::move(trace_inputs),
      [&func](Stack inputs) -> Stack {          // 跟踪的回调函数
        size_t num_func_inputs = inputs.size();
        py::tuple py_inputs(num_func_inputs);
        for (const auto i : c10::irange(num_func_inputs)) {
          py_inputs[i] = py::cast(inputs[i]);
        }
        auto out = func(*py_inputs);            // 调用Python函数
        if (out.ptr() == Py_None) {
          AT_ERROR(
              "The traced function didn't return any values! Side-effects are not "
              "captured in traces, so it would be a no-op.");
        }
        return {toTypeInferredIValue(out)};     // 转换输出值
      },
      lookup_fn_adapter,
      strict,
      force_outplace,
      self,
      argument_names);

  // 返回图形和跟踪结果栈的pair
  return std::make_pair(std::get<0>(outs)->graph, std::get<1>(outs));
}

// 在Python跟踪之前记录节点信息
Node* preRecordPythonTrace(
    THPObjectPtr pyobj,                         // Python对象指针
    const std::string& arg_types,               // 参数类型
    at::ArrayRef<Variable> inputs,              // 输入变量的引用数组
    pyobj_list scalar_args) {                   // 标量参数列表

  THPObjectPtr apply(PyObject_GetAttrString(pyobj.get(), "apply"));  // 获取apply方法
  if (!apply) {
    throw python_error();                      // 如果获取失败，抛出异常
  }

  auto& graph = getTracingState()->graph;      // 获取当前跟踪状态的图

  // 创建Python操作节点
  Node* n = graph->createPythonOp(
      std::move(apply), arg_types, std::move(scalar_args));
  recordSourceLocation(n);                      // 记录节点的源位置信息

  // 将输入变量的值追加到节点输入中
  for (const Variable& input : inputs) {
    n->addInput(getValueTrace(input));          // 获取值追踪并添加到节点输入
  }

  graph->insertNode(n);                         // 插入节点到图中

  return n;                                     // 返回创建的节点指针
}

// 记录Python节点的源位置信息
void pythonRecordSourceLocation(Node* n) {
  n->setSourceRange(getPythonInterpreterSourceRange());  // 设置节点的源代码范围
}

// 发出Python警告
void pythonWarn(const std::string& reason) {
  pybind11::gil_scoped_acquire gil;             // 获取全局解释器锁
  auto warn_class = py::module::import("torch.jit").attr("TracerWarning");  // 导入警告类
  PyErr_WarnEx(warn_class.ptr(), reason.c_str(), 1);  // 发出警告
}
void initPythonTracerBindings(PyObject* module) {
  // 设置 Python 调用栈和记录源位置的函数
  setPythonCallstack(_pythonCallstack);
  setRecordSourceLocation(pythonRecordSourceLocation);

  // 将 Python 模块转换为 py::module 对象
  auto m = py::handle(module).cast<py::module>();
  // 定义 TracingState 类，并添加相关方法
  py::class_<TracingState, std::shared_ptr<TracingState>>(
      m, "TracingState", py::dynamic_attr())
      // 定义 __repr__ 方法，返回 TracingState 对象的字符串表示
      .def(
          "__repr__",
          [](const TracingState& s) {
            std::ostringstream ss;
            ss << "<TracingState " << (const void*)&s << ">";
            return ss.str();
          })
      // 定义 __str__ 方法，返回 TracingState 对象的图形表示
      .def(
          "__str__",
          [](const TracingState& s) -> std::string {
            std::ostringstream ss;
            ss << *s.graph;
            return ss.str();
          })
      // 定义 push_scope 方法，将作用域名称推入图中
      .def(
          "push_scope",
          [](TracingState& s, const std::string& scope_name) {
            s.graph->push_scope(scope_name);
          })
      // 定义 pop_scope 方法，从图中弹出作用域名称
      .def("pop_scope", [](TracingState& s) { s.graph->pop_scope(); })
      // 定义 current_scope 方法，返回当前作用域名称
      .def(
          "current_scope",
          [](TracingState& s) {
            return s.graph->current_scope()->name().toUnqualString();
          })
      // 定义 set_graph 方法，设置 TracingState 对象的图
      .def(
          "set_graph",
          [](TracingState& s, std::shared_ptr<Graph> g) {
            s.graph = std::move(g);
          })
      // 定义 graph 方法，返回 TracingState 对象的图
      .def("graph", [](TracingState& s) { return s.graph; });

  // 定义 _tracer_warn_use_python 方法，设置警告函数
  m.def("_tracer_warn_use_python", []() { tracer::setWarn(pythonWarn); });
  // 定义 _create_graph_by_tracing 方法，通过追踪创建图
  m.def(
      "_create_graph_by_tracing",
      createGraphByTracing,
      py::arg("func"),
      py::arg("inputs"),
      py::arg("var_name_lookup_fn"),
      py::arg("strict"),
      py::arg("force_outplace"),
      py::arg("self") = nullptr,
      py::arg("argument_names") = std::vector<std::string>());
  // 定义 _get_tracing_state 方法，获取追踪状态
  m.def("_get_tracing_state", []() { return getTracingState(); });
  // 定义 _set_tracing_state 方法，设置追踪状态
  m.def("_set_tracing_state", [](std::shared_ptr<TracingState> state) {
    return setTracingState(std::move(state));
  });
  // 定义 _get_value_trace 方法，获取值的追踪
  m.def("_get_value_trace", [](const Variable& var) {
    return getValueTrace(var);
  });
  // 定义 _set_value_trace 方法，设置值的追踪
  m.def("_set_value_trace", [](const Variable& var, Value* value) {
    return setValueTrace(var, value);
  });
  // 定义 _tracer_set_get_unique_name_fn 方法，设置获取唯一名称的函数
  m.def("_tracer_set_get_unique_name_fn", [](const py::function& func) {
    const auto& tracing_state = getTracingState();
    AT_ASSERT(tracing_state);
    // 设置变量名称查找函数
    tracing_state->lookup_var_name_fn =
        [func](const Variable& var) -> std::string {
      pybind11::gil_scoped_acquire ag;
      return py::cast<std::string>(func(var));
    };
  });
  // 定义 _tracer_set_force_outplace 方法，设置是否强制输出位置
  m.def("_tracer_set_force_outplace", [](bool force_outplace) {
    const auto& tracing_state = getTracingState();
    AT_ASSERT(tracing_state);
    // 设置是否强制输出位置
    tracing_state->force_outplace = force_outplace;
  });
}

} // namespace torch::jit::tracer
```