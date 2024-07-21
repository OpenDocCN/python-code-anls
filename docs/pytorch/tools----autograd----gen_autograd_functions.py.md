# `.\pytorch\tools\autograd\gen_autograd_functions.py`

```
# 生成 ATen 操作的 C++ 自动求导函数的代码
#
# 这段代码生成两个文件：
#  Functions.h/cpp：autograd::Node 的子类
#  python_functions.h/cpp：上述类的 Python 绑定
#

from __future__ import annotations

from typing import Sequence

# 导入所需的模块和类
from torchgen.api.autograd import (
    Derivative,                          # 导入 Derivative 类，用于导数计算
    DifferentiabilityInfo,               # 导入 DifferentiabilityInfo 类，用于不同iabilityInfo
    SavedAttribute,                      # 导入 SavedAttribute 类，用于保存属性
    uses_retain_variables,               # 导入 uses_retain_variables 函数，处理是否保留中间变量
    uses_single_grad,                    # 导入 uses_single_grad 函数，处理是否仅有单一的梯度
)
from torchgen.api.types import (
    ArrayRefCType,                       # 导入 ArrayRefCType 类，处理数组引用类型
    BaseCppType,                         # 导入 BaseCppType 类，处理基本 C++ 类型
    BaseCType,                           # 导入 BaseCType 类，处理基本类型
    Binding,                             # 导入 Binding 类，处理绑定
    boolT,                               # 导入 boolT 类，处理布尔类型
    doubleT,                             # 导入 doubleT 类，处理双精度浮点数类型
    intArrayRefT,                        # 导入 intArrayRefT 类，处理整数数组引用类型
    iTensorListRefT,                     # 导入 iTensorListRefT 类，处理张量列表引用类型
    ListCType,                           # 导入 ListCType 类，处理列表类型
    longT,                               # 导入 longT 类，处理长整型
    MutRefCType,                         # 导入 MutRefCType 类，处理可变引用类型
    OptionalCType,                       # 导入 OptionalCType 类，处理可选类型
    optionalIntArrayRefT,                # 导入 optionalIntArrayRefT 类，处理可选整数数组引用类型
    optionalSymIntArrayRefT,             # 导入 optionalSymIntArrayRefT 类，处理可选对称整数数组引用类型
    scalarT,                             # 导入 scalarT 类，处理标量类型
    stringT,                             # 导入 stringT 类，处理字符串类型
    symIntArrayRefT,                     # 导入 symIntArrayRefT 类，处理对称整数数组引用类型
    SymIntT,                             # 导入 SymIntT 类，处理对称整数类型
    TENSOR_LIST_LIKE_CTYPES,             # 导入 TENSOR_LIST_LIKE_CTYPES 类，处理张量列表样式的 C 类型
    tensorListT,                         # 导入 tensorListT 类，处理张量列表类型
    tensorT,                             # 导入 tensorT 类，处理张量类型
    VectorCType,                         # 导入 VectorCType 类，处理向量类型
)
from torchgen.code_template import CodeTemplate  # 导入 CodeTemplate 类，处理代码模板
from torchgen.model import Argument, FunctionSchema  # 导入 Argument 和 FunctionSchema 类，处理参数和函数架构
from torchgen.utils import FileManager  # 导入 FileManager 类，处理文件管理

from .gen_inplace_or_view_type import VIEW_FUNCTIONS  # 从本地模块导入 VIEW_FUNCTIONS 常量

# 定义 FUNCTION_DECLARATION 模板，生成 C++ 代码的声明部分
FUNCTION_DECLARATION = CodeTemplate(
    """\
#ifdef _WIN32
struct ${op} : public ${superclass} {
  TORCH_API ${op}() = default;
#else
struct TORCH_API ${op} : public ${superclass} {
#endif
  using ${superclass}::${superclass};  // 使用基类的构造函数
  variable_list apply(variable_list&& grads) override;  // 实现基类方法 apply，接收变量列表并返回变量列表
  std::string name() const override { return "${op}"; }  // 返回操作的名称
  void release_variables() override {  // 重写释放变量的方法
    ${thread_lock}  // 线程锁定操作
    ${release_variables}  // 实际释放变量的代码
  }
  ${will_release_variables}  // 将要释放变量的标志
  void compiled_args(CompiledNodeArgs& args) override;  // 编译参数列表的方法
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;  // 应用带有保存变量的方法
  ${saved_variables}  // 保存变量的定义
  ${saved_list_sizes}  // 保存变量列表大小的定义
};
"""
)

# 定义 WILL_RELEASE_VARIABLES 模板，生成将要释放变量的代码
WILL_RELEASE_VARIABLES = CodeTemplate(
    """\
bool retain_variables = true;  // 默认保留中间变量
void will_release_variables() override {  // 将要释放变量的方法
  retain_variables = false;  // 标志不再保留中间变量
}
"""
)

# 定义 FUNCTION_DEFINITION 模板，生成 C++ 代码的实现部分
FUNCTION_DEFINITION = CodeTemplate(
    """\
variable_list ${op}::apply(variable_list&& grads) {  // 实现 apply 方法，接收变量列表并返回变量列表
  ${thread_lock}  // 线程锁定操作
  ${asserts}  // 断言检查
  IndexRangeGenerator gen;  // 创建索引范围生成器
  ${compute_index_ranges}  // 计算索引范围的代码
  variable_list grad_inputs(gen.size());  // 根据索引范围生成变量列表
  ${body}  // 主体代码
  return grad_inputs;  // 返回生成的梯度输入列表
}
void ${op}::compiled_args(CompiledNodeArgs& args) {  // 实现编译参数的方法
    ${compiled_args}  // 编译参数的具体实现
}
variable_list ${op}::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {  // 实现带有保存变量的应用方法
    ${apply_with_saved_before}  // 应用前的准备工作
    variable_list result = apply(variable_list(grads));  // 调用 apply 方法
    ${apply_with_saved_after}  // 应用后的处理
    return result;  // 返回结果
}
"""
)

# 定义 GRAD_INPUT_MASK 模板，生成梯度输入掩码的代码
GRAD_INPUT_MASK = CodeTemplate(
    """\
  auto grad_input_mask = std::array<bool, ${n}>{
    ${masks}  // 控制变量的布尔数组
  };\
"""
)

# 定义 DERIVATIVE_SINGLE 模板，生成单个导数计算的代码
DERIVATIVE_SINGLE = CodeTemplate(
    """\
if (task_should_compute_output({ ${name}_ix })) {  // 检查是否需要计算输出
  auto grad_result = ${derivative};  // 计算导数
  copy_range(grad_inputs, ${name}_ix, grad_result);  // 复制计算结果到梯度输入列表中的指定位置
}
"""
)

# 注释：
// 定义 DERIVATIVE_SINGLE_FOREACH 代码模板，用于处理单一张量情况
DERIVATIVE_SINGLE_FOREACH = CodeTemplate(
    """\
if (task_should_compute_output({ ${name}_ix })) {
  // 存储梯度结果的向量
  std::vector<Tensor> grad_result;
  grad_result.reserve(grads.size());
  // 遍历梯度列表
  for (const auto & i : c10::irange(grads.size())) {
    // 如果梯度已定义，则使用指定的导数；否则插入空张量
    if (grads[i].defined()) {
      grad_result.emplace_back(${derivative});
    } else {
      grad_result.emplace_back(Tensor());
    }
  }
  // 将梯度结果复制到 grad_inputs 的指定位置
  copy_range(grad_inputs, ${name}_ix, grad_result);
}
"""
)

// 定义 DERIVATIVE_MULTI_COPY_RANGE 代码模板，用于处理多张量复制范围
DERIVATIVE_MULTI_COPY_RANGE = CodeTemplate(
    """\
  if (task_should_compute_output({ ${name}_ix })) {
    // 复制范围的梯度结果到 grad_inputs 的指定位置
    copy_range(grad_inputs, ${name}_ix, std::get<${i}>(grad_result));
  }
"""
)

// 定义 DERIVATIVE_MULTI 代码模板，用于处理多种情况的导数计算
DERIVATIVE_MULTI = CodeTemplate(
    """\
if (task_should_compute_output({ ${idx_ranges} })) {
  ${grad_input_mask}
  // 计算导数的结果
  auto grad_result = ${derivative};
  // 复制各个范围的梯度结果到 grad_inputs
  ${copy_ranges}
}
"""
)

// 定义 PY_FUNCTION_DEFINITION 代码模板，用于生成 Python 绑定
PY_FUNCTION_DEFINITION = CodeTemplate(
    """\
static PyTypeObject ${op}Class;
addClass<${op}>(module, ${op}Class, "${op}", ${op}_properties);
"""
)

// 定义 PY_FUNCTION_PROPS_AND_GETTERS 代码模板，用于生成 Python 属性和 getter 方法
PY_FUNCTION_PROPS_AND_GETTERS = CodeTemplate(
    """\
${all_getter_definitions}

static struct PyGetSetDef ${op}_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  ${all_getsetdef_structs}
  {nullptr} /* sentinel */
};

"""
)

// 定义 PY_GETSETDEF_STRUCT 代码模板，用于生成 Python 的 getset 结构
PY_GETSETDEF_STRUCT = CodeTemplate(
    """\
{(char*)"_saved_${name}", (getter)THP${op}_${name}_getter, nullptr, nullptr, nullptr}"""
)

// 定义 PY_RAW_GETSETDEF_STRUCT 代码模板，用于生成 Python 的 raw getset 结构
PY_RAW_GETSETDEF_STRUCT = CodeTemplate(
    """\
{(char*)"_raw_saved_${name}", (getter)THP${op}_${name}_raw_getter, nullptr, nullptr, nullptr}"""
)

// 定义 GETTER_DEFINITION 代码模板，用于生成 getter 方法
GETTER_DEFINITION = CodeTemplate(
    """\
PyObject* THP${op}_${name}_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  // 获取属性
  auto prop = static_cast<${op}*>(self->cdata.get())->${name};
  ${body}
  END_HANDLE_TH_ERRORS
}
"""
)

// 定义 GETTER_DEFINITION_SAVEDVAR 代码模板，用于生成保存变量的 getter 方法
GETTER_DEFINITION_SAVEDVAR = CodeTemplate(
    """\
PyObject* THP${op}_${name}_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  // 获取常量引用的属性
  const auto& prop = static_cast<${op}*>(self->cdata.get())->${name}_;
  ${body}
  END_HANDLE_TH_ERRORS
}
"""
)

// 定义 GETTER_DEFINITION_RAW_SAVEDVAR 代码模板，用于生成保存变量的 raw getter 方法
GETTER_DEFINITION_RAW_SAVEDVAR = CodeTemplate(
    """\
PyObject* THP${op}_${name}_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  // 获取常量引用的属性
  const auto& prop = static_cast<${op}*>(self->cdata.get())->${name}_;
  ${body}
  END_HANDLE_TH_ERRORS
}
"""
)

// 定义 GETTER_DEFINITION_VEC_SAVEDVAR 代码模板，用于生成保存向量变量的 getter 方法
GETTER_DEFINITION_VEC_SAVEDVAR = CodeTemplate(
    """\
PyObject* THP${op}_${name}_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  // 获取属性
  auto prop = static_cast<${op}*>(self->cdata.get())->${name};
  ${body}
  END_HANDLE_TH_ERRORS
}
"""
PyObject* THP${op}_${name}_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  // 将self的cdata强制转换为${op}*类型，并获取${name}_属性的引用
  const auto *node = static_cast<${op}*>(self->cdata.get());
  const auto& prop = node->${name}_;
  // 如果${name}_released_为真，则设置异常并返回空指针
  if (node->${name}_released_) {
    PyErr_SetString(PyExc_RuntimeError, ERR_BACKWARD_TWICE);
    return nullptr;
  }
  // ${body}是一个占位符，表示后续代码将被替换到此处
  // 此处应该包含具体的返回逻辑，根据${name}_的类型不同可能有不同的处理方式
  // 例如，可以根据不同的类型构建不同的PyObject*对象进行返回
  END_HANDLE_TH_ERRORS
}



PyObject* THP${op}_${name}_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  // 将self的cdata强制转换为${op}*类型，并获取${name}_属性的引用
  const auto *node = static_cast<${op}*>(self->cdata.get());
  const auto& prop = node->${name}_;
  // 如果${name}_released_为真，则设置异常并返回空指针
  if (node->${name}_released_) {
    PyErr_SetString(PyExc_RuntimeError, ERR_BACKWARD_TWICE);
    return nullptr;
  }
  // ${body}是一个占位符，表示后续代码将被替换到此处
  // 此处应该包含具体的返回逻辑，根据${name}_的类型不同可能有不同的处理方式
  // 在此模板中，${body}对应的是返回原始类型的getter逻辑
  END_HANDLE_TH_ERRORS
}



PyObject* THP${op}_${name}_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  // 将self的cdata强制转换为${op}*类型，并获取${name}属性的optional值
  auto opt_prop = static_cast<${op}*>(self->cdata.get())->${name};
  // 如果optional值不存在，则返回Py_None
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  // 获取optional值的实际内容
  auto prop = opt_prop.value();
  // ${body}是一个占位符，表示后续代码将被替换到此处
  END_HANDLE_TH_ERRORS
}



PyObject* THP${op}_${name}_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  // 将self的cdata强制转换为${op}*类型，并获取${name}属性的optional值
  auto opt_prop = static_cast<${op}*>(self->cdata.get())->${name};
  // 如果optional值中的list不存在，则返回Py_None
  if (!opt_prop.list.has_value()) {
    Py_RETURN_NONE;
  }
  // 获取optional值中的list内容
  auto prop = opt_prop.list.value();
  // ${body}是一个占位符，表示后续代码将被替换到此处
  END_HANDLE_TH_ERRORS
}



// Getter body
GETTER_BODY_SAVEDVAR = """\
// 将prop解包为THPVariable对象，并将其封装为PyObject*返回
return THPVariable_Wrap(prop.unpack(self->cdata));
"""

GETTER_BODY_RAW_SAVEDVAR = """\
// 将prop转换为pybind11::object对象，并释放其所有权后返回其底层指针
pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
return obj.release().ptr();
"""

GETTER_BODY_VEC_SAVEDVAR = """\
// 创建一个长度为prop.size()的元组对象
PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
// 遍历prop并将每个元素解包为THPVariable对象，存入元组中
for (auto i: c10::irange(prop.size())) {
  PyTuple_SetItem(tup, (Py_ssize_t) i, THPVariable_Wrap(prop[i].unpack(self->cdata)));
}
return tup;
"""

GETTER_BODY_RAW_VEC_SAVEDVAR = """\
// 创建一个长度为prop.size()的元组对象
PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
// 遍历prop并将每个元素转换为pybind11::object对象，并将其底层指针存入元组中
for (auto i : c10::irange(prop.size())) {
  pybind11::object obj = pybind11::cast(prop[i], pybind11::return_value_policy::reference);
  PyTuple_SetItem(tup, (Py_ssize_t) i, obj.release().ptr());
}
return tup;
"""

GETTER_BODY_ARRAYREF_LONG = """\
// 创建一个长度为prop.size()的元组对象
PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
// 遍历prop并将每个元素转换为PyLong对象，存入元组中
for (auto i : c10::irange(prop.size())) {
  PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
}
return tup;
"""

GETTER_BODY_ARRAYREF_SYMINT = """\
// 创建一个长度为prop.size()的元组对象
PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
// 遍历prop并将每个元素转换为PyLong对象或符号整数对象，存入元组中
for (auto i : c10::irange(prop.size())) {
    auto si = prop[i];
    if (auto m = si.maybe_as_int()) {
      PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong(*m));
    } else {
      auto py_symint = py::cast(si).release().ptr();
      PyTuple_SetItem(tup, (Py_ssize_t) i, py_symint);
    }
}
return tup;
"""

GETTER_BODY_ARRAYREF_DOUBLE = """\
// 创建一个长度为prop.size()的元组对象
PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());

// 此处省略了具体的遍历和处理逻辑，需要根据实际情况填充
// 请根据实际情况将具体的代码补充完整
// 此注释不应该超出代码块范围
for (auto i : c10::irange(prop.size())) {
  // 遍历prop向量的每个元素，使用auto关键字推断元素类型
  PyTuple_SetItem(tup, (Py_ssize_t) i, PyFloat_FromDouble((double) prop[i]));
  // 在Python元组tup中设置第i个元素，将prop[i]转换为Python的浮点数对象
}
return tup;
"""
// 返回Python元组tup

GETTER_BODY_INT64_T = """\
return PyLong_FromUnsignedLong((int64_t) prop);
"""
// 返回prop转换为Python的无符号长整型对象

GETTER_BODY_SYMINT = """\
if (auto m = prop.maybe_as_int()) {
  return PyLong_FromUnsignedLong(*m);
} else {
  return py::cast(prop).release().ptr();
}
"""
// 如果prop可以作为整数返回，将其转换为Python的无符号长整型对象；否则，释放prop的Python对象引用

GETTER_BODY_DOUBLE = """\
return PyFloat_FromDouble((double) prop);
"""
// 返回prop转换为Python的浮点数对象

GETTER_BODY_BOOL = """\
if (prop) {
  Py_RETURN_TRUE;
} else {
  Py_RETURN_FALSE;
}
"""
// 如果prop为真，则返回Python的True对象；否则返回False对象

GETTER_BODY_STRING = """\
return PyUnicode_FromStringAndSize(prop.data(), prop.size());
"""
// 返回prop作为字符串的Unicode对象

GETTER_BODY_SCALAR = """\
if (prop.isComplex()) {
  // 如果prop是复数类型
  auto cprop = prop.to<c10::complex<double>>();
  // 将prop转换为双精度复数类型
  return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  // 返回以cprop的实部和虚部创建的Python复数对象
} else if (prop.isFloatingPoint()) {
  // 如果prop是浮点数类型
  return PyFloat_FromDouble(prop.to<double>());
  // 返回prop转换为Python的浮点数对象
} else if (prop.isIntegral(/*includeBool=*/false)) {
  // 如果prop是整数类型（不包括布尔值）
  return PyLong_FromLong(prop.to<int64_t>());
  // 返回prop转换为Python的长整型对象
} else if (prop.isBoolean()) {
  // 如果prop是布尔值类型
  if (prop.to<bool>()) {
    // 如果prop为真
    Py_RETURN_TRUE;
    // 返回Python的True对象
  } else {
    Py_RETURN_FALSE;
    // 返回Python的False对象
  }
} else {
  PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
  // 抛出运行时错误，表明未知的标量类型
  return nullptr;
  // 返回空指针表示错误
}
"""


GETTER_BODY_VEC_SCALAR = """\
PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
// 创建Python元组对象tup，大小为prop的大小
for (auto i: c10::irange(prop.size())) {
  // 遍历prop的每个索引i
  if (prop[i].isComplex()) {
    // 如果prop[i]是复数类型
    auto cprop = prop[i].to<c10::complex<double>>();
    // 将prop[i]转换为双精度复数类型
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyComplex_FromDoubles(cprop.real(), cprop.imag()));
    // 在Python元组tup中设置第i个元素，将cprop的实部和虚部作为Python复数对象
  } else if (prop[i].isFloatingPoint()) {
    // 如果prop[i]是浮点数类型
    auto double_prop = prop[i].to<double>();
    // 将prop[i]转换为双精度浮点数
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyFloat_FromDouble(double_prop));
    // 在Python元组tup中设置第i个元素，将double_prop作为Python浮点数对象
  } else if (prop[i].isIntegral(/*includeBool=*/false)) {
    // 如果prop[i]是整数类型（不包括布尔值）
    auto long_prop = prop[i].to<int64_t>();
    // 将prop[i]转换为长整型
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromLong(long_prop));
    // 在Python元组tup中设置第i个元素，将long_prop作为Python长整型对象
  } else if (prop[i].isBoolean()) {
    // 如果prop[i]是布尔值类型
    if (prop[i].to<bool>()) {
      // 如果prop[i]为真
      PyTuple_SetItem(tup, (Py_ssize_t) i, Py_True);
      // 在Python元组tup中设置第i个元素为Python的True对象
    } else {
      PyTuple_SetItem(tup, (Py_ssize_t) i, Py_False);
      // 在Python元组tup中设置第i个元素为Python的False对象
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    // 抛出运行时错误，表明未知的标量类型
    return nullptr;
    // 返回空指针表示错误
  }
}
return tup;
// 返回Python元组tup
"""


MISC_GETTER_DEFS = {
    OptionalCType(BaseCType(longT)): (GETTER_DEFINITION_OPT, GETTER_BODY_INT64_T),
    // 对于可选的长整型基本类型，使用GETTER_DEFINITION_OPT定义，使用GETTER_BODY_INT64_T返回Python对象
    OptionalCType(BaseCType(SymIntT)): (GETTER_DEFINITION_OPT, GETTER_BODY_SYMINT),
    // 对于可选的符号整数基本类型，使用GETTER_DEFINITION_OPT定义，使用GETTER_BODY_SYMINT返回Python对象
    BaseCType(doubleT): (GETTER_DEFINITION, GETTER_BODY_DOUBLE),
    // 对于双精度浮点数基本类型，使用GETTER_DEFINITION定义，使用GETTER_BODY_DOUBLE返回Python对象
    OptionalCType(BaseCType(doubleT)): (GETTER_DEFINITION_OPT, GETTER_BODY_DOUBLE),
    // 对于可选的双精度浮点数基本类型，使用GETTER_DEFINITION_OPT定义，使用GETTER_BODY_DOUBLE返回Python对象
    BaseCType(boolT): (GETTER_DEFINITION, GETTER_BODY_BOOL),
    // 对于布尔值基本类型，使用GETTER_DEFINITION定义，使用GETTER_BODY_BOOL返回Python对象
    BaseCType(scalarT): (GETTER_DEFINITION, GETTER_BODY_SCALAR),
    // 对于标量基本类型，使用GETTER_DEFINITION定义，使用GETTER_BODY_SCALAR返回Python对象
    OptionalCType(BaseCType(scalarT)): (GETTER_DEFINITION_OPT, GETTER_BODY_SCALAR),
    // 对于可选的标量基本类型，使用GETTER_DEFINITION_OPT定义，使用GETTER_BODY_SCALAR返回Python对象
}

// These functions have backwards which cannot be traced, and so must have
// their backward functions traced opaquely.
// VIEW_FUNCTIONS are not traceable because they use as_strided, which
// has an untraceable backwards, see
// https://github.com/pytorch/pytorch/issues/4250
# 定义一个变量，用于存储不可追踪的函数列表，初始值等同于VIEW_FUNCTIONS
UNTRACEABLE_FUNCTIONS = VIEW_FUNCTIONS


def get_infos_with_derivatives_list(
    differentiability_infos: dict[FunctionSchema, dict[str, DifferentiabilityInfo]]
) -> list[DifferentiabilityInfo]:
    # 从给定的字典中提取所有的DifferentiabilityInfo对象，存入一维列表diff_info_list
    diff_info_list = [
        info
        for diffinfo_dict in differentiability_infos.values()  # 遍历不同FunctionSchema对应的dict
        for info in diffinfo_dict.values()  # 遍历每个dict中的DifferentiabilityInfo对象
    ]

    # 使用过滤器函数，从diff_info_list中筛选出具有args_with_derivatives属性的对象，存入新列表并返回
    return list(filter(lambda info: info.args_with_derivatives, diff_info_list))


def gen_autograd_functions_lib(
    out: str,
    differentiability_infos: dict[FunctionSchema, dict[str, DifferentiabilityInfo]],
    template_path: str,
) -> None:
    """Functions.h and Functions.cpp body

    These contain the auto-generated subclasses of torch::autograd::Node
    for each every differentiable torch function.
    """

    # 获取所有具有导数信息的DifferentiabilityInfo对象，存入infos列表
    infos = get_infos_with_derivatives_list(differentiability_infos)
    # 生成函数声明列表，对infos中的每个函数使用process_function函数生成声明
    declarations = [process_function(f, FUNCTION_DECLARATION) for f in infos]
    # 生成函数定义列表，对infos中的每个函数使用process_function函数生成定义
    definitions = [process_function(f, FUNCTION_DEFINITION) for f in infos]

    # 指定文件基名
    file_basename = "Functions"
    # 创建文件管理器实例
    fm = FileManager(install_dir=out, template_dir=template_path, dry_run=False)
    # 遍历.h和.cpp两种文件后缀
    for suffix in [".h", ".cpp"]:
        # 拼接文件名
        fname = file_basename + suffix
        # 使用模板写入文件，提供模板变量和生成的内容
        fm.write_with_template(
            fname,
            fname,
            lambda: {
                "generated_comment": "@"
                + f"generated from {fm.template_dir_for_comments()}/"
                + fname,
                "autograd_function_declarations": declarations,
                "autograd_function_definitions": definitions,
            },
        )


def gen_autograd_functions_python(
    out: str,
    differentiability_infos: dict[FunctionSchema, dict[str, DifferentiabilityInfo]],
    template_path: str,
) -> None:
    # 创建文件管理器实例
    fm = FileManager(install_dir=out, template_dir=template_path, dry_run=False)
    # 定义shards的数量
    num_shards = 5
    # 使用模板写入python_functions.h文件，提供模板变量和生成的内容
    fm.write(
        "python_functions.h",
        lambda: {
            "generated_comment": "@"
            + f"generated from {fm.template_dir_for_comments()}/python_functions.h",
            # 生成shard的前向声明列表
            "shard_forward_declare": [
                f"void initialize_autogenerated_functions_{i}(PyObject* module);"
                for i in range(num_shards)
            ],
            # 生成shard的调用列表
            "shard_call": [
                f"initialize_autogenerated_functions_{i}(module);"
                for i in range(num_shards)
            ],
        },
    )

    # 获取所有具有导数信息的DifferentiabilityInfo对象，存入infos列表
    infos = get_infos_with_derivatives_list(differentiability_infos)
    # 使用fm对象的write_sharded方法，将数据写入多个文件片段中
    fm.write_sharded(
        # 第一个参数：目标文件名为"python_functions.cpp"
        "python_functions.cpp",
        # 第二个参数：infos，传入的信息对象列表
        infos,
        # key_fn参数：用于从info对象中提取关键字的函数，这里使用info.name作为关键字
        key_fn=lambda info: info.name,
        # base_env参数：基础环境变量，包含一个生成的注释字符串
        base_env={
            "generated_comment": "@"
            + f"generated from {fm.template_dir_for_comments()}/python_functions.cpp",
        },
        # env_callable参数：可调用对象，接受info对象并返回一个包含py_function_initializers和py_function_props_and_getters键的字典
        env_callable=lambda info: {
            "py_function_initializers": [
                process_function(info, PY_FUNCTION_DEFINITION)
            ],
            "py_function_props_and_getters": [
                process_function(info, PY_FUNCTION_PROPS_AND_GETTERS)
            ],
        },
        # num_shards参数：指定要分片的数量
        num_shards=num_shards,
        # sharded_keys参数：指定在分片过程中使用的键集合
        sharded_keys={"py_function_initializers", "py_function_props_and_getters"},
    )
# 定义处理函数，接受不同iabilityInfo对象和CodeTemplate模板作为参数，并返回字符串结果
def process_function(info: DifferentiabilityInfo, template: CodeTemplate) -> str:
    # 保存变量列表，用于存储需要保留的变量名
    saved_variables: list[str] = []
    # 释放变量列表，用于存储需要释放的变量名
    release_variables: list[str] = []
    # 保存列表大小的列表，用于存储需要保存大小的变量名
    saved_list_sizes: list[str] = []
    # 解包列表，用于存储需要解包的变量名
    unpack: list[str] = []
    # 断言列表，用于存储需要断言的语句
    asserts: list[str] = []
    # 计算索引范围的列表，用于存储生成索引范围的语句
    compute_index_ranges: list[str] = []
    # 获取器定义列表，用于存储获取器的定义语句
    getter_definitions: list[str] = []
    # py_getsetdef结构体列表，用于存储结构体定义语句
    py_getsetdef_structs: list[str] = []
    # 编译参数列表，用于存储编译参数的语句
    compiled_args: list[str] = []
    # 应用前保存列表，用于存储应用前保存的语句
    apply_with_saved_before: list[str] = []
    # 应用后保存列表，用于存储应用后保存的语句
    apply_with_saved_after: list[str] = []

    # 遍历具有导数信息的参数列表
    for arg in info.args_with_derivatives:
        # 如果参数类型在TENSOR_LIST_LIKE_CTYPES中
        if arg.type in TENSOR_LIST_LIKE_CTYPES:
            # 生成变量名大小的字符串
            size = f"{arg.name}_size_"
            # 将size_t类型的变量名添加到保存列表大小的列表中
            saved_list_sizes.append(f"size_t {arg.name}_size_;")
        else:
            # 否则，大小为1
            size = "1"
        # 生成计算索引范围的语句并添加到计算索引范围的列表中
        compute_index_ranges.append(f"auto {arg.name}_ix = gen.range({size});")

# 定义PyObject类型的THP${op}_${name}_getter函数，接受THPCppFunction指针self和void指针_unused作为参数
PyObject* THP${op}_${name}_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  // 将self->cdata强制转换为${op}类型的指针node
  const auto *node = static_cast<${op}*>(self->cdata.get());
  // 获取node对象的${name}属性并赋值给prop
  const auto& prop = node->${name};
  // 如果node的${name}_released_属性为真
  if (node->${name}_released_) {
    // 设置运行时错误并返回空指针
    PyErr_SetString(PyExc_RuntimeError, ERR_BACKWARD_TWICE);
    return nullptr;
  }
  // ${body}是占位符，用于在处理函数调用时替换成具体的处理逻辑
  ${body}
  END_HANDLE_TH_ERRORS
    }
                            """
                ).substitute(
                    op=info.op,
                    name=name,
                    body=GETTER_BODY_VEC_SCALAR,
                )
            )
        else:
            # 检查是否将非拥有引用放入保存变量字段中。如果这个断言在错误触发，
            # 需要编辑这个字段。否则，可能需要在上面添加一个情况。
            assert (
                "ref" not in type.cpp_type().lower()
                and "view" not in type.cpp_type().lower()
                and "*" not in type.cpp_type()
                and "&" not in type.cpp_type()
            ), f"{type.cpp_type()} looks like it contains a non-owning reference"
            # 将带有类型和变量名的保存变量添加到列表中
            saved_variables.append(f"{type.cpp_type()} {name};")

            if type in MISC_GETTER_DEFS:
                getter_def, body = MISC_GETTER_DEFS[type]
                # 替换获取器定义中的占位符，并将结果添加到获取器定义列表中
                getter_definitions.append(
                    getter_def.substitute(op=info.op, name=name, body=body)
                )
            else:
                # 我们暂时不向 Python 绑定这些类型：
                #   TypeAndSize, at::ScalarType, TensorOptions, TensorGeometry,
                #   std::vector<std::vector<int64_t>>, std::vector<at::ScalarType>
                should_append_getsetdef = False

        # 如果应该附加 getsetdef 结构，则添加基于占位符替换的结构到 py_getsetdef_structs 列表中
        if should_append_getsetdef:
            py_getsetdef_structs.append(
                PY_GETSETDEF_STRUCT.substitute(op=info.op, name=name)
            )
        # 如果应该附加 raw_getsetdef 结构，则添加基于占位符替换的结构到 py_getsetdef_structs 列表中
        if should_append_raw_getsetdef:
            py_getsetdef_structs.append(
                PY_RAW_GETSETDEF_STRUCT.substitute(op=info.op, name=name)
            )

        # 将 collect 方法的调用代码添加到 compiled_args 列表中
        compiled_args.append(f"args.collect({visit_name});")
        # 将 before 方法的调用代码添加到 apply_with_saved_before 列表中
        apply_with_saved_before.append(f"saved.before({visit_name});")
        # 将 after 方法的调用代码添加到 apply_with_saved_after 列表中
        apply_with_saved_after.append(f"saved.after({visit_name});")

    # 对所有保存的输入变量按名称进行排序，并逐个调用 save_var 函数保存
    for var in sorted(info.all_saved_inputs, key=lambda sa: str(sa.nctype.name)):
        save_var(var, is_output=False)
    # 对所有保存的输出变量按名称进行排序，并逐个调用 save_var 函数保存
    for var in sorted(info.all_saved_outputs, key=lambda sa: str(sa.nctype.name)):
        save_var(var, is_output=True)

    # 当释放变量和在 Node::apply 中保护线程安全时，锁定互斥量
    # 参见 Note [Thread Safety on Autograd Node]
    if len(release_variables) > 0:
        thread_lock = "std::lock_guard<std::mutex> lock(mutex_);"
    else:
        thread_lock = ""

    # 如果使用 retain_variables，则生成将要释放变量的代码
    if uses_retain_variables(info):
        will_release_variables = WILL_RELEASE_VARIABLES.substitute()
    else:
        will_release_variables = ""

    # 初始化 body 列表
    body: list[str] = []

    # 如果只使用单一梯度，则添加该行代码到 body 中
    if uses_single_grad(info):
        body.append("const auto& grad = grads[0];")
    else:
        # 为返回值生成命名的梯度别名
        body.extend(
            f"const auto& {name} = grads[{info.available_named_gradients.index(name)}];"
            for name in sorted(info.used_named_gradients)
        )
    # 定义一个函数，用于生成导数的代码文本和检查是否需要定义任何梯度的布尔值
    def emit_derivative(
        derivative: Derivative,  # 导数对象
        args_with_derivatives: Sequence[Binding],  # 带有导数信息的参数序列
    ) -> tuple[bool, str]:  # 返回一个元组，包含布尔值和生成的导数代码文本
        formula = derivative.formula  # 获取导数的数学公式
        var_names = derivative.var_names  # 获取参与导数计算的变量名列表
        if len(var_names) == 1:  # 如果只有一个变量名
            checks_any_grad_defined = False  # 初始化是否检查任何梯度已定义的标志为False
            if "not_implemented" not in formula:  # 如果公式中不包含"not_implemented"
                # 查找匹配的参数，即与导数变量名相同的参数对象
                matching_args = [
                    arg for arg in args_with_derivatives if arg.name == var_names[0]
                ]
                if len(matching_args) == 1:  # 如果找到了唯一匹配的参数
                    # 如果参数是Tensor类型，并且需要支持未定义梯度的情况
                    arg = matching_args[0]
                    if isinstance(arg.argument, Argument) and str(
                        arg.argument.type
                    ) in ("Tensor", "Tensor?"):
                        # 将公式包装为在任何梯度定义时执行的表达式
                        formula = "any_grad_defined ? (" + formula + ") : Tensor()"
                        checks_any_grad_defined = True  # 设置检查任何梯度已定义的标志为True
            # 根据函数名前缀选择导数模板
            if info.name.startswith("_foreach_"):
                derivative_template = DERIVATIVE_SINGLE_FOREACH
            else:
                derivative_template = DERIVATIVE_SINGLE
            # 返回检查结果和生成的导数代码文本
            return (
                checks_any_grad_defined,
                derivative_template.substitute(name=var_names[0], derivative=formula),
            )
        else:  # 如果有多个变量名
            if "grad_input_mask" in formula:  # 如果公式中包含"grad_input_mask"
                # 为每个变量生成掩码以检查是否计算输出
                masks = [
                    f"task_should_compute_output({{ {n}_ix }})," for n in var_names
                ]
                grad_input_mask = GRAD_INPUT_MASK.substitute(
                    masks=masks, n=len(var_names)
                )
            else:
                grad_input_mask = ""  # 否则掩码为空字符串
            idx_ranges = ", ".join(f"{n}_ix" for n in var_names)  # 生成索引范围字符串
            copy_ranges: list[str] = []
            for i, n in enumerate(var_names):  # 遍历变量名列表
                # 为每个变量生成多重复制范围的代码文本
                copy_ranges.append(DERIVATIVE_MULTI_COPY_RANGE.substitute(name=n, i=i))
            # 返回False和生成的多变量导数代码文本
            return False, DERIVATIVE_MULTI.substitute(
                idx_ranges=idx_ranges,
                copy_ranges=copy_ranges,
                derivative=formula,
                grad_input_mask=grad_input_mask,
            )

    body.extend(unpack)  # 将解压缩的代码段扩展到函数体中
    need_any_grad_defined_var = False  # 初始化是否需要定义任何梯度的变量为False
    for derivative in info.derivatives:  # 遍历所有导数信息对象
        # 生成导数代码文本并检查是否需要定义任何梯度
        checks_any_grad_defined, derivative_text = emit_derivative(
            derivative, info.args_with_derivatives
        )
        body.append(derivative_text)  # 将生成的导数代码文本添加到函数体中
        need_any_grad_defined_var |= checks_any_grad_defined  # 更新是否需要定义任何梯度的变量
    # 如果有任何导数公式需要检查梯度是否定义，则在所有公式之前执行检查
    if need_any_grad_defined_var:
        body.insert(
            -len(info.derivatives),
            "bool any_grad_defined = any_variable_defined(grads);",
        )

    if info.name in UNTRACEABLE_FUNCTIONS:  # 如果函数名在不可跟踪函数列表中
        superclass = "Node"  # 设置超类为Node
    else:
        superclass = "TraceableFunction"  # 否则设置超类为TraceableFunction
    # 将 py_getsetdef_structs 列表转换为字符串，每个元素之间用逗号分隔，并在末尾加上逗号（如果列表不为空）
    all_getsetdef_structs = (
        ",\n".join(py_getsetdef_structs) + "," if len(py_getsetdef_structs) != 0 else ""
    )
    # 将 getter_definitions 列表转换为字符串，每个元素之间用换行符分隔
    all_getter_definitions = "\n".join(getter_definitions)

    # 使用字符串模板 template 进行格式化，替换其中的占位符
    return template.substitute(
        op=info.op,  # 替换操作符信息
        compute_index_ranges=compute_index_ranges,  # 替换计算索引范围的信息
        saved_variables=saved_variables,  # 替换保存的变量信息
        release_variables=release_variables,  # 替换释放的变量信息
        saved_list_sizes=saved_list_sizes,  # 替换保存的列表大小信息
        asserts=asserts,  # 替换断言信息
        thread_lock=thread_lock,  # 替换线程锁信息
        will_release_variables=will_release_variables,  # 替换将释放的变量信息
        body=body,  # 替换主体信息
        superclass=superclass,  # 替换超类信息
        all_getter_definitions=all_getter_definitions,  # 替换所有 getter 定义的字符串
        all_getsetdef_structs=all_getsetdef_structs,  # 替换所有 getsetdef 结构的字符串
        compiled_args=compiled_args,  # 替换编译参数信息
        apply_with_saved_before=apply_with_saved_before,  # 替换在保存之前应用的信息
        apply_with_saved_after=apply_with_saved_after,  # 替换在保存之后应用的信息
    )
```