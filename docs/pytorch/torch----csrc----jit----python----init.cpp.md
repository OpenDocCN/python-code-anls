# `.\pytorch\torch\csrc\jit\python\init.cpp`

```py
#include <pybind11/pytypes.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/python_arg_parser.h>
#include <torch/csrc/utils/schema_info.h>

#include <ATen/core/operator_name.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/backends/backend_init.h>
#include <torch/csrc/jit/codegen/cuda/interface.h>
// #include <torch/csrc/jit/codegen/cuda/python_frontend/python_bindings.h>
#include <torch/csrc/jit/codegen/fuser/interface.h>
#include <torch/csrc/jit/codegen/fuser/kernel_cache.h>
#if (!defined(FBCODE_CAFFE2) && defined(BUILD_ONEDNN_GRAPH))
#include <torch/csrc/jit/codegen/onednn/interface.h>
#endif
#include <c10/core/SymNodeImpl.h>
#include <torch/csrc/jit/frontend/ir_emitter.h>
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/autocast.h>
#include <torch/csrc/jit/passes/batch_mm.h>
#include <torch/csrc/jit/passes/canonicalize.h>
#include <torch/csrc/jit/passes/canonicalize_graph_fuser_ops.h>
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/create_autodiff_subgraphs.h>
#include <torch/csrc/jit/passes/create_functional_graphs.h>
#include <torch/csrc/jit/passes/dbr_quantization/remove_redundant_aliases.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/decompose_ops.h>
#include <torch/csrc/jit/passes/device_type_analysis.h>
#include <torch/csrc/jit/passes/dtype_analysis.h>
#include <torch/csrc/jit/passes/erase_number_types.h>
#include <torch/csrc/jit/passes/fold_conv_bn.h>
#include <torch/csrc/jit/passes/freeze_module.h>
#include <torch/csrc/jit/passes/frozen_concat_linear.h>
#include <torch/csrc/jit/passes/frozen_conv_add_relu_fusion.h>
#include <torch/csrc/jit/passes/frozen_conv_folding.h>
#include <torch/csrc/jit/passes/frozen_graph_optimizations.h>
#include <torch/csrc/jit/passes/frozen_linear_folding.h>
#include <torch/csrc/jit/passes/frozen_linear_transpose.h>
#include <torch/csrc/jit/passes/frozen_ops_to_mkldnn.h>
#include <torch/csrc/jit/passes/fuse_linear.h>
#include <torch/csrc/jit/passes/fuse_relu.h>
#include <torch/csrc/jit/passes/graph_fuser.h>
#include <torch/csrc/jit/passes/inline_fork_wait.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/passes/integer_value_refinement.h>
#include <torch/csrc/jit/passes/loop_unrolling.h>
#include <torch/csrc/jit/passes/lower_graph.h>
#include <torch/csrc/jit/passes/lower_tuples.h>
#include <torch/csrc/jit/passes/metal_rewrite.h>
#include <torch/csrc/jit/passes/mobile_optimizer_type.h>
#include <torch/csrc/jit/passes/normalize_ops.h>
#include <torch/csrc/jit/passes/peephole.h>
#include <torch/csrc/jit/passes/peephole_list_idioms.h>
#include <torch/csrc/jit/passes/quantization/dedup_module_uses.h>



// 包含必要的头文件，用于 PyTorch 的 C++ 绑定和 JIT 优化 passes
#include <pybind11/pytypes.h>                              // PyBind11 Python 类型支持
#include <torch/csrc/utils/pybind.h>                       // PyTorch PyBind 实用函数
#include <torch/csrc/utils/python_arg_parser.h>            // PyTorch Python 参数解析
#include <torch/csrc/utils/schema_info.h>                  // PyTorch 模型架构信息

#include <ATen/core/operator_name.h>                       // ATen 运算符命名
#include <torch/csrc/jit/api/module.h>                    // PyTorch JIT 模块接口
#include <torch/csrc/jit/backends/backend_init.h>         // PyTorch JIT 后端初始化
#include <torch/csrc/jit/codegen/cuda/interface.h>        // PyTorch CUDA 代码生成接口
// #include <torch/csrc/jit/codegen/cuda/python_frontend/python_bindings.h>  // CUDA Python 前端绑定
#include <torch/csrc/jit/codegen/fuser/interface.h>       // PyTorch JIT 融合器接口
#include <torch/csrc/jit/codegen/fuser/kernel_cache.h>    // PyTorch JIT 融合器内核缓存
#if (!defined(FBCODE_CAFFE2) && defined(BUILD_ONEDNN_GRAPH))
#include <torch/csrc/jit/codegen/onednn/interface.h>      // PyTorch JIT OneDNN 接口
#endif
#include <c10/core/SymNodeImpl.h>                         // C10 符号节点实现
#include <torch/csrc/jit/frontend/ir_emitter.h>           // PyTorch JIT IR 发射器
#include <torch/csrc/jit/frontend/tracer.h>               // PyTorch JIT 追踪器
#include <torch/csrc/jit/ir/irparser.h>                   // PyTorch JIT IR 解析器
#include <torch/csrc/jit/jit_log.h>                       // PyTorch JIT 日志
#include <torch/csrc/jit/passes/autocast.h>               // PyTorch JIT 自动类型转换
#include <torch/csrc/jit/passes/batch_mm.h>               // PyTorch JIT 批量矩阵乘
#include <torch/csrc/jit/passes/canonicalize.h>           // PyTorch JIT 规范化
#include <torch/csrc/jit/passes/canonicalize_graph_fuser_ops.h>  // PyTorch JIT 融合图操作规范化
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>  // PyTorch JIT 公共子表达式消除
#include <torch/csrc/jit/passes/constant_pooling.h>      // PyTorch JIT 常量池化
#include <torch/csrc/jit/passes/constant_propagation.h>  // PyTorch JIT 常量传播
#include <torch/csrc/jit/passes/create_autodiff_subgraphs.h>  // PyTorch JIT 创建自动微分子图
#include <torch/csrc/jit/passes/create_functional_graphs.h>  // PyTorch JIT 创建功能图
#include <torch/csrc/jit/passes/dbr_quantization/remove_redundant_aliases.h>  // PyTorch JIT 去冗余别名
#include <torch/csrc/jit/passes/dead_code_elimination.h>  // PyTorch JIT 死代码消除
#include <torch/csrc/jit/passes/decompose_ops.h>          // PyTorch JIT 分解操作
#include <torch/csrc/jit/passes/device_type_analysis.h>  // PyTorch JIT 设备类型分析
#include <torch/csrc/jit/passes/dtype_analysis.h>         // PyTorch JIT 数据类型分析
#include <torch/csrc/jit/passes/erase_number_types.h>     // PyTorch JIT 擦除数值类型
#include <torch/csrc/jit/passes/fold_conv_bn.h>           // PyTorch JIT 折叠 Conv + BN
#include <torch/csrc/jit/passes/freeze_module.h>          // PyTorch JIT 冻结模块
#include <torch/csrc/jit/passes/frozen_concat_linear.h>   // PyTorch JIT 冻结 Concat + Linear
#include <torch/csrc/jit/passes/frozen_conv_add_relu_fusion.h>  // PyTorch JIT 冻结 Conv + Add + ReLU 融合
#include <torch/csrc/jit/passes/frozen_conv_folding.h>   // PyTorch JIT 冻结 Conv 折叠
#include <torch/csrc/jit/passes/frozen_graph_optimizations.h>  // PyTorch JIT 冻结图优化
#include <torch/csrc/jit/passes/frozen_linear_folding.h>  // PyTorch JIT 冻结 Linear 折叠
#include <torch/csrc/jit/passes/frozen_linear_transpose.h>  // PyTorch JIT 冻结 Linear 转置
#include <torch/csrc/jit/passes/frozen_ops_to_mkldnn.h>   // PyTorch JIT 冻结操作到 MKLDNN
#include <torch/csrc/jit/passes/fuse_linear.h>            // PyT
// 包含量化相关的头文件
#include <torch/csrc/jit/passes/quantization/finalize.h>
#include <torch/csrc/jit/passes/quantization/fusion_passes.h>
#include <torch/csrc/jit/passes/quantization/insert_observers.h>
#include <torch/csrc/jit/passes/quantization/insert_quant_dequant.h>
#include <torch/csrc/jit/passes/quantization/quantization_type.h>

// 包含用于优化和改进模型的 JIT passes 头文件
#include <torch/csrc/jit/passes/refine_tuple_types.h>
#include <torch/csrc/jit/passes/remove_dropout.h>
#include <torch/csrc/jit/passes/remove_expands.h>
#include <torch/csrc/jit/passes/remove_inplace_ops.h>
#include <torch/csrc/jit/passes/remove_mutation.h>
#include <torch/csrc/jit/passes/replacement_of_old_operators.h>
#include <torch/csrc/jit/passes/restore_mutation.h>
#include <torch/csrc/jit/passes/shape_analysis.h>
#include <torch/csrc/jit/passes/specialize_autogradzero.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>
#include <torch/csrc/jit/passes/symbolic_shape_analysis.h>
#include <torch/csrc/jit/passes/tensorexpr_fuser.h>
#include <torch/csrc/jit/passes/utils/check_alias_annotation.h>
#include <torch/csrc/jit/passes/vulkan_rewrite.h>
#include <torch/csrc/jit/passes/xnnpack_rewrite.h>

// 包含用于 Python 绑定的头文件
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/jit/python/python_arg_flatten.h>
#include <torch/csrc/jit/python/python_custom_class.h>
#include <torch/csrc/jit/python/python_ir.h>
#include <torch/csrc/jit/python/python_tracer.h>
#include <torch/csrc/jit/python/python_tree_views.h>
#include <torch/csrc/jit/python/script_init.h>
#include <torch/csrc/jit/python/utf8_decoding_ignore.h>

// 包含 JIT 运行时相关的头文件
#include <torch/csrc/jit/runtime/argument_spec.h>
#include <torch/csrc/jit/runtime/autodiff.h>
#include <torch/csrc/jit/runtime/decomposition_registry.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/csrc/jit/runtime/jit_exception.h>
#include <torch/csrc/jit/runtime/jit_trace.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <torch/csrc/jit/runtime/print_handler.h>
#include <torch/csrc/jit/runtime/static/init.h>
#include <torch/csrc/jit/runtime/symbolic_shape_registry.h>

// 包含 JIT 序列化相关的头文件
#include <torch/csrc/jit/serialization/export.h>
#include <torch/csrc/jit/serialization/import.h>

// 包含 TensorExpr 相关的头文件
#include <torch/csrc/jit/tensorexpr/kernel.h>
#include <torch/csrc/jit/tensorexpr/tensorexpr_init.h>

// 包含用于调试和错误处理的工具头文件
#include <torch/csrc/utils/cpp_stacktraces.h>

// 包含 C10 和 Caffe2 相关的头文件
#include <c10/macros/Export.h>
#include <c10/util/irange.h>
#include <c10/util/signal_handler.h>
#include <caffe2/serialize/inline_container.h>

// 包含 Pybind11 相关的头文件
#include <pybind11/cast.h>
#include <pybind11/functional.h>
#include <pybind11/iostream.h>
#include <pybind11/operators.h>

// 包含 JIT 运行时性能分析的头文件
#include <torch/csrc/jit/runtime/profiling_graph_executor_impl.h>

// 包含标准库的头文件
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

// 使用 torch::jit 命名空间
namespace torch::jit {

// 使用 C10 和 Caffe2 中的别名信息
using c10::AliasInfo;
using c10::Argument;
using c10::FunctionSchema;
using c10::SchemaArgType;
using c10::SchemaArgument;
using c10::SymNode;
using caffe2::serialize::PyTorchStreamReader;
using caffe2::serialize::PyTorchStreamWriter;

// 继续添加其他自定义的命名空间或类型别名的声明...
// 引入 torch::utils::SchemaInfo 命名空间中的类
using torch::utils::SchemaInfo;

// 匿名命名空间，限定了内部作用域
namespace {

// 使用 autograd::variable_list 别名
using autograd::variable_list;

// 加载 Python 类的函数，返回布尔值表示是否成功加载
bool loadPythonClasses() {
  // 尝试导入 "torch.jit" 模块
  // 留下这段代码的原因是因为将来可能会有用
  // PyObject *jit_module = PyImport_ImportModule("torch.jit");
  // THPUtils_assert(jit_module, "class loader couldn't access "
  // "torch.jit module");
  // PyObject *jit_dict = PyModule_GetDict(jit_module);

  return true;
}

// 判断操作符是否允许将数字作为张量处理
static bool opAllowsNumbersAsTensors(c10::Symbol symbol) {
  return symbol.is_prims() || symbol.is_nvprims() ||
         (symbol.is_aten() &&
          torch::should_allow_numbers_as_tensors(symbol.toUnqualString()));
}

// 将 Python 对象转换为类型推断的 IValue，返回 std::optional 包装的结果
std::optional<IValue> toTypeInferredIValueOptional(py::handle input) {
  // 需要在此处捕获错误，因为 toTypeInferredIValue 在各种对象类型上会报错，但我们希望它能处理所有类型
  try {
    return toTypeInferredIValue(input);
  } catch (const c10::Error& e) {
    return c10::nullopt;
  }
}
} // 匿名命名空间结束

// 如果未定义 USE_ROCM，声明一个函数 runJITCPPTests
#if !defined(USE_ROCM)
TORCH_API void runJITCPPTests();
#endif

// 初始化 JIT 绑定
void initJITBindings(PyObject* module) {
  // 将 PyObject* 转换为 py::module 对象
  auto m = py::handle(module).cast<py::module>();
  // 定义子模块 "_jit"
  auto jit = m.def_submodule("_jit");

  // 这是一个静态对象，必须泄漏 Python 对象
  // 在此处使用 "release()" 以保留对象的引用计数，防止它被 CPython 释放
  static py::handle exc =
      py::exception<JITException>(m, "JITException").release();

  // 注册异常转换器
  py::register_exception_translator([](std::exception_ptr p) {
    try {
      if (p) {
        std::rethrow_exception(p);
      }
    } catch (const JITException& e) {
      // 特殊处理 JITException，设置其 Python 类名和消息
      py::gil_scoped_acquire acquire;
      const auto& className = e.getPythonClassName();
      const auto& originalMsg = e.getOriginalMsg();
      JITException::setCaughtOriginalMsg(originalMsg.value_or(""));
      JITException::setCaughtPythonClassName(className.value_or(""));
      // 如果我们仍然有 py::exception<JITException> 对象，我们可以直接调用它
      // 但是我们必须获取一个 handle 来泄漏它，而我找不到从 handle 重新创建它的方法
      // 因此手动设置异常
      PyErr_SetString(exc.ptr(), e.what());
      
      // 如果正在构建测试且未定义 USE_ROCM，定义两个测试函数
#if defined(BUILDING_TESTS) && !defined(USE_ROCM)
      jit.def(
          "_jit_run_cpp_tests",
          []() {
            // 必须在此方法中释放 GIL，因为如果我们在这些测试中初始化 autograd 引擎，
            // 新生成的工作线程将尝试初始化它们的 PyThreadState*，它们需要 GIL
            pybind11::gil_scoped_release _no_gil;
            return runJITCPPTests();
          });
      jit.def("_jit_has_cpp_tests", []() { return true; });
      jit.def("_has_tensorexpr_cpp_tests", []() { return true; });
#endif
    }
  });
}
#else
      .def("_jit_run_cpp_tests", []() { throw std::exception(); })
      .def("_jit_has_cpp_tests", []() { return false; })
      .def("_run_tensorexpr_cpp_tests", []() { throw std::exception(); })
      .def("_has_tensorexpr_cpp_tests", []() { return false; })
#endif
      .def(
          "_jit_flatten",
          [](py::handle& obj) {
            // 调用 Python 函数 python::flatten 对象进行展平操作
            auto res = python::flatten(obj);
            // 返回展平后的结果，包括变量列表和描述符
            return std::make_pair(res.vars, res.desc);
          })
      .def(
          "_jit_unflatten",
          [](const autograd::variable_list& vars, python::IODescriptor& desc) {
            // 调用 Python 函数 python::unflatten 进行反展平操作
            return py::reinterpret_steal<py::object>(
                python::unflatten(vars, desc));
          })
      .def("_jit_pass_canonicalize_graph_fuser_ops", CanonicalizeOps)
      .def("_jit_pass_decompose_ops", DecomposeOps)
      .def("_jit_pass_specialize_autogradzero", specializeAutogradZero)
      .def("_jit_override_can_fuse_on_cpu", &overrideCanFuseOnCPU)
      .def("_jit_override_can_fuse_on_gpu", &overrideCanFuseOnGPU)
      .def("_jit_can_fuse_on_cpu", &canFuseOnCPU)
      .def("_jit_can_fuse_on_gpu", &canFuseOnGPU)
      .def("_jit_can_fuse_on_cpu_legacy", &canFuseOnCPULegacy)
      .def("_jit_override_can_fuse_on_cpu_legacy", &overrideCanFuseOnCPULegacy)
      .def(
          "_jit_differentiate",
          [](Graph& g) {
            // 对输入的图形进行差分操作
            // python 绑定在语义上略有不同，它会复制输入的图形并在副本上操作
            // jit::differentiate 会直接在输入的图形上进行修改
            auto g_clone = g.copy();  // 复制输入的图形
            return differentiate(g_clone);  // 执行差分操作并返回结果
          })
      .def(
          "_jit_check_alias_annotation",
          [](const std::shared_ptr<Graph>& g,
             const py::tuple& args,
             const std::string& unqualified_op_name) {
            // 将 Python 元组 args 转换为可追踪的堆栈
            auto stack = toTraceableStack(args);
            // 检查别名注解，传入图形、堆栈和操作名
            checkAliasAnnotation(g, std::move(stack), unqualified_op_name);
          })
#if (!defined(FBCODE_CAFFE2) && defined(BUILD_ONEDNN_GRAPH))
      .def("_jit_set_llga_enabled", &RegisterLlgaFuseGraph::setEnabled)
      .def("_jit_llga_enabled", &RegisterLlgaFuseGraph::isEnabled)
#else
      .def("_jit_set_llga_enabled", [](bool flag) { return false; })
      .def("_jit_llga_enabled", []() { return false; })
#ifdef TORCH_ENABLE_LLVM
            return true;
#else
            return false;
#define SYMNODE_UNARY(n) .def(#n, [](c10::SymNode a) { return a->n(); })
#define SYMNODE_BINARY(n) \
  .def(#n, [](c10::SymNode a, c10::SymNode b) { return a->n(b); })
#endif
    BufferAdapter(const py::object& buffer) : buffer_(buffer) {
      // 使用给定的 Python 对象 `buffer` 初始化 BufferAdapter 类
      // 调用 `tell` 方法获取当前文件指针位置
      auto current = buffer.attr("tell")();
      // 将当前文件指针位置转换为 size_t 类型，并保存为起始偏移量
      start_offset_ = py::cast<size_t>(current);
      // 将文件指针移动到文件末尾以获取文件大小
      buffer.attr("seek")(current, py::module::import("os").attr("SEEK_END"));
      // 计算文件大小并保存为 size_t 类型
      size_ = py::cast<size_t>(buffer.attr("tell")()) - start_offset_;
      // 将文件指针移动回起始位置
      buffer.attr("seek")(current);

      // 如果对象支持直接读入到缓冲区，则设置为 true，否则为 false
      use_readinto_ = py::hasattr(buffer, "readinto");
    }

    size_t size() const override {
      // 返回缓冲区的大小
      return size_;
    }

    THPObjectPtr getMemview(void* buf, size_t n) const {
      // 创建一个内存视图对象，并返回其指针
      THPObjectPtr memview(PyMemoryView_FromMemory(
          reinterpret_cast<char*>(buf), n, PyBUF_WRITE));
      if (!memview) {
        // 如果创建内存视图失败，则抛出 python_error 异常
        throw python_error();
      }
      return memview;
    }

    size_t read(uint64_t pos, void* buf, size_t n, const char* what)
        const override {
      // 计算绝对位置
      Py_ssize_t absolute_pos = start_offset_ + pos;
      // 将文件指针移动到绝对位置
      buffer_.attr("seek")(absolute_pos);

      if (use_readinto_) {
        // 如果支持直接读入到缓冲区，则调用 readinto 方法
        auto memview = getMemview(buf, n);
        auto res =
            PyObject_CallMethod(buffer_.ptr(), "readinto", "O", memview.get());
        if (res) {
          // 将返回的结果转换为 int64_t 类型
          int64_t i = static_cast<int64_t>(PyLong_AsLongLong(res));
          Py_DECREF(res);
          if (i > 0) {
            // 如果成功读取，则返回读取的字节数
            return i;
          }
        }
      }

      // 否则，从缓冲区读取字节到 buf 中
      std::string bytes = py::cast<std::string>(buffer_.attr("read")(n));
      std::copy(
          bytes.data(),
          bytes.data() + bytes.size(),
          reinterpret_cast<char*>(buf));
      // 返回实际读取的字节数
      return bytes.size();
    }

    py::object buffer_;  // Python 对象，用于存储缓冲区
    size_t size_;        // 缓冲区的大小
    size_t start_offset_;  // 缓冲区的起始偏移量
    std::optional<IValue> self_value = toTypeInferredIValueOptional(self);
    std::optional<IValue> other_value = toTypeInferredIValueOptional(other);

    // 只有在确信 self 和 other 都有值时才返回 true
    if (!self_value || !other_value) {
      return false;
    }
    // 调用 self_value 的 overlaps 方法来检查 self 和 other 是否重叠
    return self_value->overlaps(*other_value);
  });
  // 定义名为 "_awaitable" 的函数
  m.def("_awaitable", [](const py::args& args, const py::kwargs& kwargs) {
    // 确保参数 args 至少包含一个元素
    AT_ASSERT(args.size() >= 1);
    // 创建一个不包含第一个参数的新 tuple
    py::tuple args_tup(args.size() - 1);
    // 将 args 中除第一个元素外的所有元素放入 args_tup 中
    for (const auto i : c10::irange(1, args.size())) {
      args_tup[i - 1] = args[i];
    }
    // 返回一个 PythonAwaitWrapper 对象，包含第一个参数作为函数和剩余参数作为参数 tuple
    return std::make_shared<PythonAwaitWrapper>(
        py::cast<py::function>(args[0]), std::move(args_tup));
  });
  // 定义名为 "_awaitable_nowait" 的函数
  m.def("_awaitable_nowait", [](py::handle input) {
    // 返回一个 PythonAwaitWrapper 对象，包含输入的 handle 作为参数
    return std::make_shared<PythonAwaitWrapper>(std::move(input));
  });
  // 定义名为 "_awaitable_wait" 的函数
  m.def(
      "_awaitable_wait", [](const std::shared_ptr<PythonAwaitWrapper>& py_aw) {
        // 断言 py_aw 不为 nullptr
        TORCH_CHECK(py_aw, "Await can't be None");
        // 调用 PythonAwaitWrapper 对象的 wait 方法并返回结果
        return py_aw->wait();
      });
  // 定义名为 "fork" 的函数
  m.def("fork", [](const py::args& args, const py::kwargs& kwargs) {
    // 确保 args 不为空
    AT_ASSERT(!args.empty());

    // 将 args 中的第一个元素转换为 py::function
    py::function f = py::cast<py::function>(args[0]);
    // 创建一个不包含第一个参数的新 tuple
    py::tuple args_tup(args.size() - 1);

    // 将 args 中除第一个元素外的所有元素放入 args_tup 中
    for (const auto i : c10::irange(1, args.size())) {
      args_tup[i - 1] = args[i];
    }

    // 如果正在进行跟踪
    if (jit::tracer::isTracing()) {
      auto graph = jit::tracer::getTracingState()->graph;
      auto fork_node = graph->insertNode(graph->create(prim::TracedFork, 1));
      auto body_block = fork_node->addBlock();

      Value* node_output = nullptr;
      py::object py_func_output;
      // 在 fork 操作的子块中插入新的跟踪操作
      WithInsertPoint guard(body_block);
      IValue output_ivalue;
      {
        tracer::WithNestedTracingFrame env_guard;

        // 运行用户提供的函数
        py_func_output = f(*args_tup, **kwargs);

        // 将用户提供函数的输出转换为 IValue。此 IValue 的类型信息用于记录正确的类型到跟踪中
        output_ivalue = toTypeInferredIValue(py_func_output);
        Value* out_val = jit::tracer::getValueTrace(output_ivalue);
        body_block->registerOutput(out_val);
        node_output =
            fork_node->output()->setType(FutureType::create(out_val->type()));
      }

      auto retval =
          c10::make_intrusive<c10::ivalue::Future>(output_ivalue.type());

      // 在跟踪器中记录 IValue
      jit::tracer::setValueTrace(retval, node_output);

      // 将 ivalue 输出放入 Future 中
      retval->markCompleted(output_ivalue);

      return std::make_shared<PythonFutureWrapper>(retval);
    } else {
      // 否则，直接返回用户函数的结果
      auto result = toTypeInferredIValue(f(*args_tup, **kwargs));
      auto retval = c10::make_intrusive<c10::ivalue::Future>(result.type());
      retval->markCompleted(std::move(result));
      return std::make_shared<PythonFutureWrapper>(retval);
  });



  // 定义一个 Python 模块方法 "wait"，接受一个 PythonFutureWrapper 的共享指针作为参数
  m.def("wait", [](const std::shared_ptr<PythonFutureWrapper>& fut) {
    // 检查 fut 是否为空，如果为空则抛出错误信息
    TORCH_CHECK(fut, "Future can't be None");
    // 调用 PythonFutureWrapper 对象的 wait() 方法等待其完成，并返回结果
    return fut->wait();
  });



  // 定义一个名为 "_collect_all" 的 Python 模块方法
  m.def(
      "_collect_all",
      [](const std::vector<std::shared_ptr<jit::PythonFutureWrapper>>& futures)
          -> std::shared_ptr<jit::PythonFutureWrapper> {
        // 确定元素类型指针 typePtr，如果 futures 为空或第一个元素为 nullptr，则使用 AnyType::get()
        auto typePtr = futures.empty() || futures[0] == nullptr
            ? AnyType::get()
            : futures[0]->fut->elementType();
        
        // 创建一个 c10::List，用于存放 c10::ivalue::Future 对象
        c10::List<c10::intrusive_ptr<c10::ivalue::Future>> asList(
            c10::FutureType::create(typePtr));
        
        // 预留足够的空间以容纳 futures 的元素数量
        asList.reserve(futures.size());
        
        // 遍历 futures 向 asList 中添加元素
        for (const auto& f : futures) {
          // 检查每个 fut 是否为空，如果为空则抛出错误信息
          TORCH_CHECK(f, "Future can't be None");
          // 将 PythonFutureWrapper 对象中的 c10::ivalue::Future 添加到 asList 中
          asList.push_back(f->fut);
        }
        
        // 返回一个新创建的 jit::PythonFutureWrapper 对象，其中包含 collectAll(asList) 的结果
        return std::make_shared<jit::PythonFutureWrapper>(
            c10::collectAll(asList),
            /* unwrap_func */ [futures](const py::object& /*unused*/) {
              // 在返回的 Future 上调用 wait() 方法，确保所有原始 futures 都完成
              // 如果任何原始 futures 抛出异常，则在此处捕获并抛出错误信息
              for (auto& fut : futures) {
                fut->wait();
              }
            });
      },
      // 使用 py::gil_scoped_release 作为 Python GIL 的释放策略
      py::call_guard<py::gil_scoped_release>());



  // 定义一个名为 "_jit_assert_is_instance" 的 Python 模块方法，接受一个 py::object 对象和一个 TypePtr 类型的参数
  m.def("_jit_assert_is_instance", [](py::object obj, const TypePtr& type) {
    // 调用 toIValue 函数将 obj 转换为对应的 IValue，使用给定的类型指针进行类型检查
    toIValue(std::move(obj), type);
  });
#if defined(C10_SUPPORTS_FATAL_SIGNAL_HANDLERS)
  #ifdef C10_SUPPORTS_FATAL_SIGNAL_HANDLERS 宏定义的条件下，才编译以下代码段
  m.def("_set_print_stack_traces_on_fatal_signal", [](bool print) {
    // 定义 Python 绑定函数 "_set_print_stack_traces_on_fatal_signal"，接收布尔参数 print
    c10::FatalSignalHandler::getInstance().setPrintStackTracesOnFatalSignal(
        print);
    // 调用 C++ 的 FatalSignalHandler 单例，设置是否在致命信号时打印堆栈跟踪信息
  });
#endif // defined(C10_SUPPORTS_SIGNAL_HANDLER)
  // 结束条件编译段

// 调用初始化函数，设置自定义 Python 类的绑定
initPythonCustomClassBindings(module);
// 调用初始化函数，设置 Python IR 的绑定
initPythonIRBindings(module);
// 调用初始化函数，设置 Python 追踪器的绑定
tracer::initPythonTracerBindings(module);
// 调用初始化函数，设置树视图的绑定
initTreeViewBindings(module);
// 调用初始化函数，设置 JIT 脚本的绑定
initJitScriptBindings(module);
// 调用初始化函数，设置 JIT 后端的绑定
initJitBackendBindings(module);
// 调用初始化函数，设置静态模块的绑定
initStaticModuleBindings(module);
// 调用初始化函数，设置 Tensor 表达式的绑定
initTensorExprBindings(module);
// 调用初始化函数，设置 NVFuser Python 的绑定
// initNvFuserPythonBindings(module);

// 设置打印处理函数，以 lambda 形式定义，接收字符串参数 str
setPrintHandler([](const std::string& str) {
  // 获取全局解释器锁 GIL
  py::gil_scoped_acquire acquire;
  try {
    // 尝试导入 Python 的 sys 模块，并获取其 stdout 属性
    auto _stdout = py::module::import("sys").attr("stdout");
    // 调用 stdout 的 write 方法，写入参数中的字符串
    _stdout.attr("write")(str);
  } catch (py::error_already_set& e) {
    // 捕获 Python 异常，转换为 C++ 异常并抛出
    throw std::runtime_error(e.what());
  }
});

// 在退出时注册函数，重置打印处理函数为默认处理函数
auto atexit = py::module_::import("atexit");
atexit.attr("register")(
    py::cpp_function([]() { setPrintHandler(getDefaultPrintHandler()); }));
}

} // namespace torch::jit
// 命名空间结束：torch::jit
```