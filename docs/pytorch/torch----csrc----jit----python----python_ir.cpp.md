# `.\pytorch\torch\csrc\jit\python\python_ir.cpp`

```
`
// 引入 Torch JIT 库中的 Python IR 头文件
#include <torch/csrc/jit/python/python_ir.h>

// 引入 ATen 库中的 JIT 类型定义
#include <ATen/core/jit_type.h>

// 引入 pybind11 库的头文件
#include <pybind11/pybind11.h>

// 引入 Torch 的设备和数据类型定义
#include <torch/csrc/Device.h>
#include <torch/csrc/Dtype.h>

// 引入 Torch Python API 头文件
#include <torch/csrc/api/include/torch/python.h>

// 引入 Torch JIT 中的别名分析和 IR 相关头文件
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/canonicalize.h>
#include <torch/csrc/jit/passes/onnx/helper.h>
#include <torch/csrc/jit/passes/shape_analysis.h>
#include <torch/csrc/jit/passes/symbolic_shape_analysis.h>

// 引入 Torch JIT Python 绑定相关头文件
#include <torch/csrc/jit/python/pybind.h>
#include <torch/csrc/jit/python/python_tracer.h>

// 引入 Torch JIT 运行时参数和导出相关头文件
#include <torch/csrc/jit/runtime/argument_spec.h>
#include <torch/csrc/jit/serialization/export.h>
#include <torch/csrc/jit/serialization/python_print.h>

// 引入 Torch 的 Python 头文件
#include <torch/csrc/python_headers.h>

// 引入 Torch 实用工具的 Python 绑定相关头文件
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/python_strings.h>

// 引入 C++ 标准库中的输入输出流和字符串流
#include <iostream>
#include <sstream>

// 引入 STL 中的实用工具
#include <utility>

// Torch JIT 命名空间
namespace torch::jit {

// 控制是否默认打印图源范围的全局变量
bool global_print_source_ranges = true;

// 定义 ConcretePythonOp 的种类为 prim::PythonOp
Symbol ConcretePythonOp::Kind = prim::PythonOp;

// 获取 Python 对象的名称
std::string getPythonName(const PyObject* obj_) {
  pybind11::gil_scoped_acquire gil;
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  PyObject* obj = const_cast<PyObject*>(obj_);
  auto v = py::getattr(obj, "__name__", py::str("<python_value>"));
  // 如果是 autograd.Function，则恢复类的名称
  return py::str(v);
}

// 打印 PyObject 对象到输出流
std::ostream& printPyObject(std::ostream& out, const THPObjectPtr& obj) {
  pybind11::gil_scoped_acquire gil;
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  auto pyobj = py::handle(const_cast<PyObject*>(obj.get()));
  
  // 如果是元组，特殊处理以避免在 Python 2 和 Python 3 中的输出差异问题
  if (py::isinstance<py::tuple>(pyobj)) {
    auto pytuple = pyobj.cast<py::tuple>();
    out << "(";
    size_t i = 0;
    for (const autoe {
    // 直接打印其他类型的 Python 对象
    return out << THPUtils_unpackString(py::str(pyobj).ptr());
  }
}

// 在 JIT 图中查找节点
Node* findNode(
    c10::ArrayRef<torch::jit::Block*> blocks,
    Symbol kind,
    ```
    # 递归查找函数，根据指定的节点类型 `kind` 在 `blocks` 中查找节点
    bool recurse = true) {
    # 遍历所有块 `blocks`
    for (Block* block : blocks) {
        # 遍历当前块 `block` 中的所有节点 `n`
        for (Node* n : block->nodes()) {
            # 如果当前节点 `n` 的类型为指定类型 `kind`
            if (n->kind() == kind) {
                # 返回找到的节点 `n`
                return n;
            }
            # 如果允许递归搜索
            if (recurse) {
                # 递归调用 `findNode` 函数，查找当前节点 `n` 的子块中是否存在类型为 `kind` 的节点
                auto node = findNode(n->blocks(), kind, recurse);
                # 如果找到符合条件的节点 `node`
                if (node != nullptr) {
                    # 返回找到的节点 `node`
                    return node;
                }
            }
        }
    }
    # 如果在所有块和节点中都未找到符合条件的节点，则返回空指针
    return nullptr;
}

// 在给定的块中查找指定种类的节点，可以选择递归查找
Node* findNode(Block* block, Symbol kind, bool recurse = true) {
  // 将当前块作为初始块放入向量中
  std::vector<Block*> blocks = {block};
  // 调用重载函数 findNode，传入块向量、节点种类和递归标志
  return findNode(blocks, kind, recurse);
}

// 获取具体的 Python 操作的名称
std::string ConcretePythonOp::name() const {
  // 获取全局解释器锁，确保线程安全
  pybind11::gil_scoped_acquire gil;
  // 如果存在自动求导函数，则返回其 Python 名称；否则返回当前 Python 对象的名称
  if (auto autograd = autogradFunction()) {
    return getPythonName(autograd->get());
  } else {
    return getPythonName(pyobj.get());
  }
}

// 从另一个节点复制信息到当前节点
void ConcretePythonOp::cloneFrom(Node* other_) {
  // 调用父类 Node 的 cloneFrom 方法，忽略某些 LINT 错误
  // NOLINTNEXTLINE(bugprone-parent-virtual-call)
  Node::cloneFrom(other_);
  auto other = other_->cast<ConcretePythonOp>();
  // 复制从其他节点获取的信息到当前节点
  this->cconv = other->cconv;
  Py_INCREF(other->pyobj.get());
  this->pyobj = THPObjectPtr(other->pyobj.get());
  for (auto& sa : other->scalar_args) {
    Py_INCREF(sa.get());
    this->scalar_args.emplace_back(sa.get());
  }
}

// 恢复自动求导函数的实例，如果该 Python 操作的函数最初是 SomeFunction.apply
// 用于 ONNX 中的符号发现
std::optional<THPObjectPtr> ConcretePythonOp::autogradFunction() const {
  // 获取全局解释器锁，确保线程安全
  pybind11::gil_scoped_acquire gil;
  // 获取当前 Python 对象的 __self__ 属性
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  py::handle obj = const_cast<PyObject*>(pyobj.get());

  auto r = py::getattr(obj, "__self__", py::none());
  if (r.is_none())
    return c10::nullopt;

  // 获取 __self__.apply 属性
  auto apply = py::getattr(r, "apply", py::none());
  if (apply.is_none())
    return c10::nullopt;

  // 检查 apply 是否与当前对象不同
  auto c = PyObject_RichCompareBool(apply.ptr(), obj.ptr(), Py_NE);
  if (PyErr_Occurred())
    throw py::error_already_set();
  if (c)
    return c10::nullopt;

  // 返回函数实例的引用
  return THPObjectPtr(r.release().ptr());
}

// 将标量参数写入输出流
void ConcretePythonOp::writeScalars(std::ostream& out) const {
  out << "(";
  int i = 0;
  for (auto& scalar : scalar_args) {
    if (i++ > 0)
      out << ", ";
    printPyObject(out, scalar);
  }
  out << ")";
}

// 对 Python 代码进行 lint 检查
void ConcretePythonOp::lint_python() const {
  size_t n_scalars = 0, n_tensors = 0;
  // 遍历 cconv 字符串，计算标量参数和张量的数量
  for (auto c : cconv) {
    if (c == 'c') {
      n_scalars++;
    } else if (c == 'd') {
      n_tensors++;
    } else {
      // 断言，如果遇到未知字符则终止程序
      AT_ASSERT(0);
    }
    // 断言当前 Python 对象存在
    AT_ASSERT(static_cast<bool>(pyobj));
  }
  // 检查标量参数和输入张量的数量是否匹配
  AT_ASSERT(n_scalars == scalar_args.size());
  AT_ASSERT(n_tensors == inputs().size());
}

// 在图中创建 Python 操作节点
Node* Graph::createPythonOp(
    THPObjectPtr&& pyobj,
    const std::string& cconv,
    pyobj_list&& scalar_args) {
  // 创建 ConcretePythonOp 对象，并初始化
  ConcretePythonOp* op = new ConcretePythonOp(this);
  return op->init(std::move(pyobj), cconv, std::move(scalar_args));
}
void initPythonIRBindings(PyObject* module_) {
  auto m = py::handle(module_).cast<py::module>();

  // 创建一个 Python 类型 'AliasDb'，绑定 C++ 类 'AliasDb'，支持以下方法：
  // - dump: 将 AliasDb 转储为字符串
  // - to_graphviz_str: 生成 AliasDb 的 Graphviz 字符串表示
  // - may_contain_alias: 检查两个值是否可能存在别名关系
  // - has_writers: 检查给定值是否有写操作
  // - __str__: 获取 AliasDb 的字符串表示
  // - move_after_topologically_valid: 在拓扑上合法移动节点到另一个节点之后
  // - move_before_topologically_valid: 在拓扑上合法移动节点到另一个节点之前
  py::class_<AliasDb, std::shared_ptr<AliasDb>>(m, "AliasDb")
      .def("dump", &AliasDb::dump)
      .def("to_graphviz_str", &AliasDb::toGraphviz)
      .def(
          "may_contain_alias",
          [&](AliasDb& db, Value* v1, Value* v2) {
            return db.mayContainAlias(v1, v2);
          })
      .def(
          "has_writers",
          [&](AliasDb& db, Value* v1) { return db.hasWriters(v1); })
      .def("__str__", &AliasDb::toString)
      .def(
          "move_after_topologically_valid",
          [](AliasDb& db, Node* n, Node* movePoint) {
            return db.moveAfterTopologicallyValid(n, movePoint);
          })
      .def(
          "move_before_topologically_valid",
          [](AliasDb& db, Node* n, Node* movePoint) {
            return db.moveBeforeTopologicallyValid(n, movePoint);
          });

  // 取消先前定义的 'GS' 宏定义

  // 定义一个宏 'VS(name)'，用于简化 'Value' 类的方法绑定
#define VS(name) def(#name, &Value ::name)
  py::class_<Value, unwrapping_shared_ptr<Value>>(m, "Value")
      // 定义 '__repr__' 方法，返回 Value 的调试名称和其所属节点的字符串表示
      .def(
          "__repr__",
          [](Value& n) {
            std::stringstream ss;
            ss << n.debugName() << " defined in (" << *n.node() << ")";
            return ss.str();
          })
      // 绑定 Value 的方法 'type' 和 'setType'
      .VS(type)
      .VS(setType)
      // 定义 'inferTypeFrom' 方法的重载，推断类型从 'at::Tensor' 或 'c10::ivalue::Object'
      .def(
          "inferTypeFrom",
          py::overload_cast<const at::Tensor&>(&Value::inferTypeFrom))
      .def(
          "inferTypeFrom",
          py::overload_cast<const c10::intrusive_ptr<c10::ivalue::Object>&>(
              &Value::inferTypeFrom))
      // 跳过 'owningGraph' 方法，因为它返回一个存储图对象的原始指针，
      // 这可能导致双重释放内存
      // 绑定 Value 的方法 'unique', 'debugName', 'setDebugName', 'offset', 'uses',
      // 'replaceAllUsesWith', 'replaceAllUsesAfterNodeWith', 'node'
      .VS(unique)
      .VS(debugName)
      .VS(setDebugName)
      .VS(offset)
      .VS(uses)
      .VS(replaceAllUsesWith)
      .VS(replaceAllUsesAfterNodeWith)
      .def("node", [](Value& v) { return v.node(); })
      // 定义 'setTypeAs' 方法，将节点类型设置为另一个节点的类型
      .def(
          "setTypeAs",
          [](Value* node, Value* other) {
            node->setType(other->type());
            return node;
          })
      .VS(copyMetadata)
      .VS(isCompleteTensor)
      .VS(requires_grad)
      // 定义 'requiresGrad' 方法，检查值是否需要梯度计算
      .def(
          "requiresGrad",
          [](Value& n) {
            return n.type()->expectRef<TensorType>().requiresGrad();
          })
      // 定义 'toIValue' 方法，将值转换为 'IValue'
      .def("toIValue", [](Value& n) { return toIValue(&n); })
      // 绑定 Value 的方法 'type'
      .def("type", [](Value& v) { return v.type(); });
}
#undef VS
// 取消定义宏 VS

py::class_<Block, unwrapping_shared_ptr<Block>>(m, "Block")
    // 定义 Python 绑定类 Block
    .def(
        "nodes",
        [](Block& b) {
          return py::make_iterator(b.nodes().begin(), b.nodes().end());
        })
    // 定义 nodes 方法，返回 Block 中节点的迭代器

    .def(
        "findNode",
        [](Block& b, const std::string& kind, bool recurse) {
          return findNode(&b, Symbol::fromQualString(kind), recurse);
        },
        "Find Node",
        py::arg("kind"),
        py::arg("recurse") = true)
    // 定义 findNode 方法，查找指定类型的节点

    .def(
        "findAllNodes",
        [](Block& b, const std::string& kind, bool recurse) {
          return findAllNodes(b, Symbol::fromQualString(kind), recurse);
        },
        "Find all nodes",
        py::arg("kind"),
        py::arg("recurse") = true)
    // 定义 findAllNodes 方法，查找所有指定类型的节点

    .def(
        "inputs",
        [](Block& b) {
          return py::make_iterator(b.inputs().begin(), b.inputs().end());
        })
    // 定义 inputs 方法，返回 Block 中输入节点的迭代器

    .def(
        "outputs",
        [](Block& b) {
          return py::make_iterator(b.outputs().begin(), b.outputs().end());
        })
    // 定义 outputs 方法，返回 Block 中输出节点的迭代器

    .def("returnNode", [](Block& b) { return b.return_node(); })
    // 定义 returnNode 方法，返回 Block 的返回节点

    .def("paramNode", [](Block& b) { return b.param_node(); })
    // 定义 paramNode 方法，返回 Block 的参数节点

    .def("owningNode", [](Block& b) { return b.owningNode(); })
    // 定义 owningNode 方法，返回拥有该 Block 的节点

    .def(
        "addNode",
        [](Block& b, const char* str, const std::vector<Value*>& inputs) {
          return addNodeToBlock(&b, Symbol::fromQualString(str), inputs);
        })
    // 定义 addNode 方法，向 Block 中添加指定类型的节点

    .def("addInputToBlock", [](Block& b) { return addInputToBlock(&b); })
    // 定义 addInputToBlock 方法，向 Block 中添加输入节点

    .def("registerOutput", [](Block& b, Value* value) {
      return b.registerOutput(value);
    });
    // 定义 registerOutput 方法，注册输出节点

#define AS(name) def(#name, &Node::name)
    // 定义宏 AS，用于简化方法的绑定

    // methods from Attributes
    .AS(copyAttributes)
    .AS(hasAttributes)
#undef AS
#define AS(name) def(#name, &Node::name##S)
    // 定义宏 AS，用于简化带有后缀 S 的方法的绑定

    // The default method names take Symbol, but the string conversion for
    // Symbol you to qualify with attr::. This is not very user friendly
    // for attributes, so expose the string variants instead.

    .AS(hasAttribute)
    .AS(kindOf)
    .AS(removeAttribute)
    .AS(attributeNames)
#undef AS

#define CREATE_ACCESSOR(Kind, method)                                       \
  def(#method "_", [](Node& n, const char* name, Kind##Attr::ValueType v) { \
    return n.method##_(Symbol::attr(name), std::move(v));                   \
  }).def(#method, [](Node& n, const char* name) {                           \
    return n.method(Symbol::attr(name));                                    \
  })
    // 定义宏 CREATE_ACCESSOR，用于创建方法的访问器

    .CREATE_ACCESSOR(Float, f)
    .CREATE_ACCESSOR(Floats, fs)
    .CREATE_ACCESSOR(Complex, c)
    .CREATE_ACCESSOR(String, s)
    .CREATE_ACCESSOR(Strings, ss)
    .CREATE_ACCESSOR(Int, i)
    .CREATE_ACCESSOR(Ints, is)
    .CREATE_ACCESSOR(Graph, g)
    .CREATE_ACCESSOR(Graphs, gs)
    .CREATE_ACCESSOR(IValue, ival);
}
} // namespace torch::jit
// 宏定义块结束和命名空间声明
```