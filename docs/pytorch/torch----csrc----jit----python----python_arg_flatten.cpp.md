# `.\pytorch\torch\csrc\jit\python\python_arg_flatten.cpp`

```py
// 包含 C++ 头文件：c10/util/irange.h、torch/csrc/jit/python/python_arg_flatten.h、torch/csrc/utils/python_strings.h、torch/csrc/utils/six.h
#include <c10/util/irange.h>
#include <torch/csrc/jit/python/python_arg_flatten.h>
#include <torch/csrc/utils/python_strings.h>
#include <torch/csrc/utils/six.h>

// 包含 torch/csrc/autograd/grad_mode.h 中的头文件
#include <torch/csrc/autograd/grad_mode.h>

// 进入 torch::jit::python 命名空间
namespace torch::jit::python {

// 使用命名空间 torch::autograd 和 at
using namespace torch::autograd;
using namespace at;

// 用于描述输入/输出结构的字符定义（D 为描述）
namespace D {
static constexpr char DictOpen = '<';    // '<' 字符代表字典的开始
static constexpr char DictClose = '>';   // '>' 字符代表字典的结束
static constexpr char ListOpen = '[';    // '[' 字符代表列表的开始
static constexpr char ListClose = ']';   // ']' 字符代表列表的结束
static constexpr char TupleOpen = '(';   // '(' 字符代表元组的开始
static constexpr char TupleClose = ')';  // ')' 字符代表元组的结束
static constexpr char Variable = 'v';    // 'v' 字符代表变量
static constexpr char Bool = 'b';        // 'b' 字符代表布尔值
static constexpr char Long = 'l';        // 'l' 字符代表长整型
static constexpr char Double = 'd';      // 'd' 字符代表双精度浮点数
static constexpr char String = 's';      // 's' 字符代表字符串
static constexpr char NoneType = 'n';    // 'n' 字符代表空值
} // namespace D

// 匿名命名空间，用于定义内部函数和变量
namespace {

// 检查 PyObject* 是否为 Py_None 的内联函数
inline bool PyNone_Check(PyObject* o) {
  return o == Py_None;
}

// 将 py::handle 向量 objs 转换为 T 类型序列的模板函数
template <typename T>
py::object cast_handle_sequence(std::vector<py::handle> objs) {
  auto num_objs = objs.size();
  T sequence{num_objs};
  for (const auto i : c10::irange(num_objs)) {
    sequence[i] = py::reinterpret_borrow<py::object>(objs[i]);
  }
  return sequence;
}

// 递归展开 Python 对象 obj 并填充 ParsedArgs args 结构的函数
void flatten_rec(PyObject* obj, ParsedArgs& args) {
  auto& structure = args.desc.structure;  // 引用结构描述的 vector
  if (six::isTuple(obj)) {  // 如果 obj 是元组
    structure.push_back(D::TupleOpen);    // 添加元组开始字符到结构描述
    for (auto item : py::reinterpret_borrow<py::tuple>(obj))
      flatten_rec(item.ptr(), args);     // 递归展开元组中的每个元素
    structure.push_back(D::TupleClose);   // 添加元组结束字符到结构描述
  } else if (PyList_Check(obj)) {         // 如果 obj 是列表
    structure.push_back(D::ListOpen);     // 添加列表开始字符到结构描述
    for (auto item : py::reinterpret_borrow<py::list>(obj))
      flatten_rec(item.ptr(), args);     // 递归展开列表中的每个元素
    structure.push_back(D::ListClose);    // 添加列表结束字符到结构描述
  } else if (PyDict_Check(obj)) {         // 如果 obj 是字典
    auto* dict_items = PyDict_Items(obj); // 获取字典的键值对列表
    structure.push_back(D::DictOpen);     // 添加字典开始字符到结构描述
    for (auto item : py::reinterpret_borrow<py::list>(dict_items)) {
      flatten_rec(item.ptr(), args);     // 递归展开字典中每个键值对
    }
    structure.push_back(D::DictClose);    // 添加字典结束字符到结构描述
    Py_DECREF(dict_items);                // 释放字典项列表的引用计数
  } else if (THPUtils_checkString(obj)) {  // 如果 obj 是字符串
    string str = THPUtils_unpackString(obj);  // 解包字符串为 C++ 字符串
    args.desc.strings.emplace_back(str);   // 将字符串添加到描述的字符串列表
    args.desc.structure.push_back(D::String);  // 添加字符串字符到结构描述
  } else if (THPVariable_Check(obj)) {     // 如果 obj 是 Torch 变量
    auto& var = THPVariable_Unpack(obj);  // 解包 Torch 变量
    args.vars.push_back(var);             // 将变量添加到变量列表
    args.desc.metadata.emplace_back(var); // 将变量添加到描述的元数据列表
    args.desc.structure.push_back(D::Variable);  // 添加变量字符到结构描述
  } else if (PyNone_Check(obj)) {          // 如果 obj 是 Py_None
    args.desc.structure.push_back(D::NoneType);  // 添加空值字符到结构描述
  } else if (PyBool_Check(obj)) {          // 如果 obj 是布尔值
    at::Tensor var = scalar_to_tensor(at::Scalar(THPUtils_unpackBool(obj)));  // 将布尔值封装为张量
    args.vars.push_back(var);             // 将张量添加到变量列表
    args.desc.metadata.emplace_back(var); // 将张量添加到描述的元数据列表
    args.desc.structure.push_back(D::Bool);  // 添加布尔值字符到结构描述
  } else if (PyLong_Check(obj)) {          // 如果 obj 是长整型
    at::Tensor var = scalar_to_tensor(
        at::Scalar(static_cast<int64_t>(THPUtils_unpackLong(obj))));  // 将长整型封装为张量
    args.vars.push_back(var);             // 将张量添加到变量列表
    args.desc.metadata.emplace_back(var); // 将张量添加到描述的元数据列表
    args.desc.structure.push_back(D::Long);  // 添加长整型字符到结构描述
  }
  // 未涵盖的情况可能会由于 Python/C++ API 的扩展导致错误
}
    // 如果对象是一个 Python 浮点数对象（PyFloat），则将其封装为双精度张量（Tensor）
    // 并将该张量添加到参数列表 args.vars 中
    // 同时将该张量添加到描述符的元数据中
    // 最后将类型标识 D::Double 添加到描述符的结构中
    } else if (PyFloat_Check(obj)) {
        at::Tensor var = scalar_to_tensor(THPUtils_unpackDouble(obj));
        args.vars.push_back(var);
        args.desc.metadata.emplace_back(var);
        args.desc.structure.push_back(D::Double);
    // 如果对象不是元组、列表或者 Variables（张量），则抛出运行时异常
    // 提示仅支持元组、列表、Variables作为JIT的输入/输出
    // 同时也支持字典和字符串，但不推荐使用
    // 抛出异常说明收到了不支持的类型的输入对象
    } else {
        std::string msg =
            "Only tuples, lists and Variables are supported as JIT inputs/outputs. "
            "Dictionaries and strings are also accepted, but their usage is not "
            "recommended. Here, received an input of unsupported type: ";
        msg += THPUtils_typename(obj);
        throw std::runtime_error(msg);
    }
} // 匿名命名空间的结束

} // anonymous namespace

// 将 Python 对象扁平化为 ParsedArgs 结构
ParsedArgs flatten(py::handle obj) {
  // 创建 ParsedArgs 结构
  ParsedArgs args;
  // 检查梯度是否启用，并将其设置到 args.desc.grad_enabled 中
  args.desc.grad_enabled = autograd::GradMode::is_enabled();
  // 递归调用 flatten_rec 函数处理 Python 对象
  flatten_rec(obj.ptr(), args);
  // 返回处理后的 args 结构
  return args;
}

namespace {

// 模板函数，将 py::object 对象的向量转换为 T 类型的序列
template <typename T>
py::object cast_sequence(std::vector<py::object> objs) {
  // 获取对象数量
  auto num_objs = objs.size();
  // 创建 T 类型的序列 sequence
  T sequence{num_objs};
  // 遍历对象向量，将每个对象移动到 sequence 中
  for (const auto i : c10::irange(num_objs)) {
    sequence[i] = std::move(objs[i]);
  }
  // 返回转换后的 sequence
  return std::move(sequence);
}

// 将 py::object 对象的向量转换为 py::dict 类型的对象
py::object cast_dict(std::vector<py::object> objs) {
  // 获取对象数量
  auto num_objs = objs.size();
  // 创建空的 py::dict 对象 sequence
  py::dict sequence = {};
  // 遍历对象向量，将每个对象解释为 py::tuple，并添加到 sequence 中
  for (const auto i : c10::irange(num_objs)) {
    py::tuple obj = py::reinterpret_borrow<py::tuple>(objs[i]);
    sequence[obj[0]] = obj[1];
  }
  // 返回转换后的 sequence
  return std::move(sequence);
}

// 递归函数，用于将扁平化的变量和描述信息 unflatten 为 py::object 对象
py::object unflatten_rec(
    ArrayRef<Variable>::iterator& var_it,
    ArrayRef<Variable>::iterator& var_it_end,
    std::string::const_iterator& desc_it,
    std::vector<string>::const_iterator& str_it,
    std::vector<string>::const_iterator& str_it_end) {
  // 获取描述符类型
  char type = *desc_it++;
  // 根据类型执行相应的解析和转换操作
  if (type == D::TupleOpen) {
    // 创建空的 py::tuple 对象序列 objs
    std::vector<py::object> objs;
    // 循环解析元组元素，直到遇到 D::TupleClose
    while (*desc_it != D::TupleClose)
      objs.push_back(
          unflatten_rec(var_it, var_it_end, desc_it, str_it, str_it_end));
    ++desc_it;
    // 返回转换后的 py::tuple 对象
    return cast_sequence<py::tuple>(objs);
  } else if (type == D::ListOpen) {
    // 创建空的 py::list 对象序列 objs
    std::vector<py::object> objs;
    // 循环解析列表元素，直到遇到 D::ListClose
    while (*desc_it != D::ListClose)
      objs.push_back(
          unflatten_rec(var_it, var_it_end, desc_it, str_it, str_it_end));
    ++desc_it;
    // 返回转换后的 py::list 对象
    return cast_sequence<py::list>(objs);
  } else if (type == D::DictOpen) {
    // 创建空的 py::dict 对象序列 objs
    std::vector<py::object> objs;
    // 循环解析字典元素，直到遇到 D::DictClose
    while (*desc_it != D::DictClose) {
      objs.push_back(
          unflatten_rec(var_it, var_it_end, desc_it, str_it, str_it_end));
    }
    ++desc_it;
    // 返回转换后的 py::dict 对象
    return cast_dict(objs);
  } else if (type == D::String) {
    // 检查字符串是否足够用于解析，如果不足则抛出异常
    if (str_it == str_it_end)
      throw std::runtime_error("Not enough Variables given to unflatten");
    auto str = *str_it++;
    // 将字符串封装为 Python 字符串对象返回
    return py::reinterpret_borrow<py::object>(THPUtils_packString(str));
  } else if (type == D::NoneType) {
    // 返回 Python 的 None 对象
    return py::reinterpret_borrow<py::object>(py::none());
  } else {
    // 处理变量类型的描述符，如果变量不足则抛出异常
    // 这里可能会涉及变量的解封装和跟踪器的使用
    if (var_it == var_it_end)
      throw std::runtime_error("Not enough Variables given to unflatten");
    auto var = *var_it++;
    // 将变量封装为 Python 对象返回
    return py::reinterpret_steal<py::object>(THPVariable_Wrap(var));
  }
}

} // 匿名命名空间的结束
// 根据变量列表和描述符解析并重建 PyObject 对象
PyObject* unflatten(ArrayRef<Variable> vars, const IODescriptor& desc) {
  // 注意：在描述符上不进行正确性检查。
  // 描述符必须是由 unflatten 函数生成的正确的字节对象。
  auto vars_it = vars.begin();  // 获取变量数组的起始迭代器
  auto vars_it_end = vars.end();  // 获取变量数组的结束迭代器
  auto desc_it = desc.structure.begin();  // 获取描述符结构的起始迭代器
  std::vector<std::string>::const_iterator str_it = desc.strings.begin();  // 获取描述符字符串的起始迭代器
  std::vector<std::string>::const_iterator str_end = desc.strings.end();  // 获取描述符字符串的结束迭代器
  // 调用递归函数 unflatten_rec 进行解析和重建
  auto output = unflatten_rec(vars_it, vars_it_end, desc_it, str_it, str_end);
  // 如果变量列表中还有未处理的变量，则抛出异常
  if (vars_it != vars_it_end)
    throw std::runtime_error("Too many Variables given to unflatten");
  // 返回解析和重建的 PyObject 对象的指针
  return output.release().ptr();
}

} // namespace torch::jit::python
```