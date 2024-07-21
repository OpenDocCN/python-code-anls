# `.\pytorch\torch\csrc\python_dimname.cpp`

```py
// 引入包含flat_hash_map的头文件
#include <c10/util/flat_hash_map.h>
// 引入torch异常处理头文件
#include <torch/csrc/Exceptions.h>
// 引入torch Python维度名处理头文件
#include <torch/csrc/python_dimname.h>
// 引入torch Python字符串处理头文件
#include <torch/csrc/utils/python_strings.h>

// 进入torch命名空间
namespace torch {

// 定义结构体InternedStringsTable
struct InternedStringsTable {
  // 默认构造函数
  InternedStringsTable() = default;
  // 析构函数声明，NOLINTNEXTLINE用于禁止特定的静态分析检查
  ~InternedStringsTable();
  // 删除拷贝构造函数和赋值运算符
  InternedStringsTable(const InternedStringsTable&) = delete;
  InternedStringsTable& operator=(InternedStringsTable const&) = delete;
  // 删除移动构造函数和移动赋值运算符
  InternedStringsTable(InternedStringsTable&&) = delete;
  InternedStringsTable& operator=(InternedStringsTable&&) = delete;

  // 查找Python对象对应的Dimname，返回optional类型
  at::optional<at::Dimname> lookup(PyObject* obj);
  // 添加Python对象和Dimname的映射关系
  // 前提条件：obj是一个interned的Python字符串
  void addMapping(PyObject* obj, at::Dimname dimname);

 private:
  // 使用flat_hash_map实现Python对象到Dimname的映射
  ska::flat_hash_map<PyObject*, at::Dimname> py_interned_string_to_dimname_;
};

// 实例化InternedStringsTable对象kPyInternedStringToDimname
InternedStringsTable kPyInternedStringToDimname;

// 析构函数的定义
// 如果Python已经退出，则释放包装的Python对象
InternedStringsTable::~InternedStringsTable() {
  if (Py_IsInitialized()) {
    // 获取全局解释器锁
    pybind11::gil_scoped_acquire gil;
    // 遍历py_interned_string_to_dimname_映射表
    for (auto it = py_interned_string_to_dimname_.begin();
         it != py_interned_string_to_dimname_.end();
         ++it) {
      // 见注释 [References to python interned strings]
      // 减少Python对象的引用计数
      Py_DECREF(it->first);
    }
  }
}

// 查找Python对象对应的Dimname
at::optional<at::Dimname> InternedStringsTable::lookup(PyObject* obj) {
  // 在映射表中查找Python对象
  auto it = py_interned_string_to_dimname_.find(obj);
  if (it == py_interned_string_to_dimname_.end()) {
    return at::nullopt; // 未找到则返回空optional
  }
  return it->second; // 返回找到的Dimname
}

// 添加Python对象和Dimname的映射关系
void InternedStringsTable::addMapping(PyObject* obj, at::Dimname dimname) {
  // 见注释 [References to python interned strings]
  // 增加Python对象的引用计数
  Py_INCREF(obj);
  // 插入映射关系到映射表中
  py_interned_string_to_dimname_.emplace(obj, dimname);
}

} // namespace torch

// 检查是否为Dimname类型的工具函数
bool THPUtils_checkDimname(PyObject* obj) {
  return obj == Py_None || THPUtils_checkString(obj);
}

// 如果obj是列表或元组并且其第一个元素是Dimname，则解析为DimnameList以避免与IntArrayRef的歧义
bool THPUtils_checkDimnameList(PyObject* obj) {
  auto tuple = PyTuple_Check(obj); // 检查是否为元组
  if (!tuple && !PyList_Check(obj)) {
    return false; // 不是元组也不是列表，则返回false
  }
  const auto size = tuple ? PyTuple_GET_SIZE(obj) : PyList_GET_SIZE(obj); // 获取列表或元组的大小
  if (size == 0) {
    return true; // 空列表或元组，直接返回true
  }
  PyObject* first_elt =
      tuple ? PyTuple_GET_ITEM(obj, 0) : PyList_GET_ITEM(obj, 0); // 获取第一个元素
  return THPUtils_checkDimname(first_elt); // 检查第一个元素是否为Dimname类型
}

// 解析Python对象为Dimname类型
at::Dimname THPDimname_parse(PyObject* obj) {
  if (obj == Py_None) {
    return at::Dimname::wildcard(); // 如果是None，则返回通配符Dimname
  }

  // 检查类型是否为字符串，否则抛出异常
  TORCH_CHECK_TYPE(
      THPUtils_checkString(obj),
      "expected None or string for Dimname but got ",
      Py_TYPE(obj)->tp_name);

  // 如果不是interned的字符串，则抛出异常
  if (!THPUtils_isInterned(obj)) {
    // 增加对象的引用计数，以确保在函数执行期间不会被销毁
    Py_INCREF(obj);
    // 将 Python 字符串对象在原地进行国际化，即将其变为唯一化的字符串对象
    THPUtils_internStringInPlace(&obj);
    // 减少对象的引用计数，恢复为函数调用前的状态
    Py_DECREF(obj);
  }

  // 尝试从已经映射的 Python 字符串到维度名的表中查找这个对象
  auto maybeDimname = torch::kPyInternedStringToDimname.lookup(obj);
  // 如果找到匹配的维度名，则直接返回
  if (maybeDimname) {
    return *maybeDimname;
  }

  // 如果未找到匹配的维度名，则解包 Python 字符串对象
  const auto name = THPUtils_unpackString(obj);
  // 创建新的维度名对象，并与 Python 字符串对象关联
  auto dimname = at::Dimname::fromSymbol(at::Symbol::dimname(name));
  // 将 Python 字符串对象与新创建的维度名对象建立映射关系
  torch::kPyInternedStringToDimname.addMapping(obj, dimname);
  // 返回新创建的维度名对象
  return dimname;
}


注释：


# 这是代码块的结束，表示前面的代码逻辑或者功能定义已经完成
```