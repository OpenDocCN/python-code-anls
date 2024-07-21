# `.\pytorch\torch\csrc\fx\node.cpp`

```py
#include <torch/csrc/fx/node.h>

#include <structmember.h>
#include <torch/csrc/utils/pythoncapi_compat.h>

////////////////////////////////
// NodeBase
///////////////////////////////

// 定义一个结构体 NodeBase，用于表示节点的基本信息
struct NodeBase {
  PyObject_HEAD bool _erased; // 节点是否被擦除的标志，布尔类型
  NodeBase* _prev; // 指向前一个节点的指针
  NodeBase* _next; // 指向下一个节点的指针
};

// 创建 NodeBase 对象的构造函数
static PyObject* NodeBase_new(
    PyTypeObject* type,
    PyObject* args,
    PyObject* kwds) {
  PyObject* self = type->tp_alloc(type, 0); // 分配内存以创建新的 NodeBase 对象
  if (!self)
    return nullptr;
  return self;
}

// NodeBase 对象的初始化函数
static int NodeBase_init_fn(NodeBase* self, PyObject* args, PyObject* kwds) {
  self->_erased = false; // 初始化 _erased 标志为 false
  Py_INCREF(self); // 增加自身的引用计数
  self->_prev = self; // 设置 _prev 指向自身
  Py_INCREF(self); // 再次增加自身的引用计数
  self->_next = self; // 设置 _next 指向自身
  return 0;
}

// 定义 NodeBase 的成员变量
// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
static struct PyMemberDef NodeBase_members[] = {
    {"_erased", T_BOOL, offsetof(NodeBase, _erased), 0, nullptr}, // _erased 成员的定义
    {"_prev", T_OBJECT_EX, offsetof(NodeBase, _prev), 0, nullptr}, // _prev 成员的定义
    {"_next", T_OBJECT_EX, offsetof(NodeBase, _next), 0, nullptr}, // _next 成员的定义
    {nullptr} /* Sentinel */
};

// NodeBase 对象的垃圾回收遍历函数
static int NodeBase_traverse(NodeBase* self, visitproc visit, void* arg) {
  Py_VISIT(self->_prev); // 访问并增加 _prev 成员的引用计数
  Py_VISIT(self->_next); // 访问并增加 _next 成员的引用计数
  return 0;
}

// NodeBase 对象的清理函数
static int NodeBase_clear(NodeBase* self) {
  Py_CLEAR(self->_prev); // 清理 _prev 成员的引用计数
  Py_CLEAR(self->_next); // 清理 _next 成员的引用计数
  return 0;
}

// NodeBase 对象的销毁函数
static void NodeBase_dealloc(PyObject* self) {
  PyObject_GC_UnTrack(self); // 停止跟踪垃圾回收
  (void)NodeBase_clear((NodeBase*)self); // 清理 NodeBase 对象的成员
  Py_TYPE(self)->tp_free(self); // 释放 NodeBase 对象的内存
}

// 定义 NodeBaseType 类型对象
static PyTypeObject NodeBaseType = {
    PyVarObject_HEAD_INIT(nullptr, 0) "torch._C._NodeBase", /* tp_name */
    sizeof(NodeBase), /* tp_basicsize */
    0, /* tp_itemsize */
    (destructor)NodeBase_dealloc, /* tp_dealloc */
    0, /* tp_vectorcall_offset */
    nullptr, /* tp_getattr */
    nullptr, /* tp_setattr */
    nullptr, /* tp_reserved */
    nullptr, /* tp_repr */
    nullptr, /* tp_as_number */
    nullptr, /* tp_as_sequence */
    nullptr, /* tp_as_mapping */
    nullptr, /* tp_hash  */
    nullptr, /* tp_call */
    nullptr, /* tp_str */
    nullptr, /* tp_getattro */
    nullptr, /* tp_setattro */
    nullptr, /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE |
        Py_TPFLAGS_HAVE_GC, /* tp_flags */
    nullptr, /* tp_doc */
    (traverseproc)NodeBase_traverse, /* tp_traverse */
    (inquiry)NodeBase_clear, /* tp_clear */
    nullptr, /* tp_richcompare */
    0, /* tp_weaklistoffset */
    nullptr, /* tp_iter */
    nullptr, /* tp_iternext */
    nullptr, /* tp_methods */
    NodeBase_members, /* tp_members */
    nullptr, /* tp_getset */
    nullptr, /* tp_base */
    nullptr, /* tp_dict */
    nullptr, /* tp_descr_get */
    nullptr, /* tp_descr_set */
    0, /* tp_dictoffset */
    (initproc)NodeBase_init_fn, /* tp_init */
    nullptr, /* tp_alloc */
    NodeBase_new, /* tp_new */
};

// 将 NodeBaseType 添加到给定的模块中
bool NodeBase_init(PyObject* module) {
  if (PyModule_AddType(module, &NodeBaseType) < 0) {
    return false;
  }
  return true;
}

////////////////////////////////
// NodeIter
////////////////////////////////
struct NodeIter {
  PyObject_HEAD bool _reversed;  // 布尔变量，指示迭代顺序是否为逆序
  NodeBase* _root;               // 指向根节点的指针
  NodeBase* _cur;                // 当前节点的指针
};

static PyObject* NodeIter_new(
    PyTypeObject* type,
    PyObject* args,
    PyObject* kwds) {
  PyObject* self = type->tp_alloc(type, 0);  // 分配新的 NodeIter 对象
  if (!self)
    return nullptr;  // 分配失败时返回空指针
  return self;       // 返回分配的对象
}

static int NodeIter_init_fn(NodeIter* self, PyObject* args, PyObject* kwargs) {
  NodeBase* root = nullptr;   // 根节点初始化为空指针
  bool reversed = false;      // 初始迭代顺序为正序
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
  constexpr const char* keywords[] = {"root", "reversed", nullptr};  // 关键字参数列表
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "Ob|",
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
          const_cast<char**>(keywords),
          &root,
          &reversed)) {
    return -1;  // 解析参数失败时返回 -1
  }
  self->_reversed = reversed;  // 设置迭代顺序
  Py_INCREF(root);             // 增加根节点的引用计数
  self->_root = root;          // 设置根节点指针
  Py_INCREF(root);             // 再次增加根节点的引用计数
  self->_cur = root;           // 设置当前节点指针为根节点
  return 0;                    // 初始化成功返回 0
}

template <bool reversed>
PyObject* NodeIter_iternext_helper(NodeIter* self) {
  // It should be possible to relax the ref counting here
  // but in practice, we do not have that many _erased Nodes,
  // so probably not worth it.
  if constexpr (reversed) {
    NodeBase* prev = (NodeBase*)Py_NewRef(self->_cur->_prev);  // 获取前一个节点
    Py_CLEAR(self->_cur);  // 清除当前节点的引用
    self->_cur = prev;     // 更新当前节点为前一个节点
  } else {
    NodeBase* next = (NodeBase*)Py_NewRef(self->_cur->_next);  // 获取后一个节点
    Py_CLEAR(self->_cur);  // 清除当前节点的引用
    self->_cur = next;     // 更新当前节点为后一个节点
  }
  while (self->_cur != self->_root) {  // 循环直到当前节点等于根节点
    if (!self->_cur->_erased) {        // 如果当前节点未被删除
      Py_INCREF(self->_cur);           // 增加当前节点的引用计数
      return (PyObject*)self->_cur;    // 返回当前节点的 PyObject 指针
    }
    if constexpr (reversed) {
      NodeBase* prev = (NodeBase*)Py_NewRef(self->_cur->_prev);  // 获取前一个节点
      Py_CLEAR(self->_cur);  // 清除当前节点的引用
      self->_cur = prev;     // 更新当前节点为前一个节点
    } else {
      NodeBase* next = (NodeBase*)Py_NewRef(self->_cur->_next);  // 获取后一个节点
      Py_CLEAR(self->_cur);  // 清除当前节点的引用
      self->_cur = next;     // 更新当前节点为后一个节点
    }
  }
  PyErr_SetNone(PyExc_StopIteration);  // 设置迭代结束异常
  return nullptr;                     // 返回空指针表示迭代结束
}

PyObject* NodeIter_iternext(PyObject* _self) {
  NodeIter* self = (NodeIter*)_self;  // 将 PyObject 指针转换为 NodeIter 指针
  if (self->_reversed) {
    return NodeIter_iternext_helper<true>(self);   // 如果逆序，则调用逆序迭代辅助函数
  } else {
    return NodeIter_iternext_helper<false>(self);  // 否则调用正序迭代辅助函数
  }
}

static int NodeIter_traverse(NodeIter* self, visitproc visit, void* arg) {
  Py_VISIT(self->_root);  // 增加根节点的引用计数
  Py_VISIT(self->_cur);   // 增加当前节点的引用计数
  return 0;                // 成功返回 0
}

static int NodeIter_clear(NodeIter* self) {
  Py_CLEAR(self->_root);  // 清除根节点的引用
  Py_CLEAR(self->_cur);   // 清除当前节点的引用
  return 0;                // 成功返回 0
}

static void NodeIter_dealloc(PyObject* self) {
  PyObject_GC_UnTrack(self);         // 取消跟踪对象
  (void)NodeIter_clear((NodeIter*)self);  // 清除对象
  Py_TYPE(self)->tp_free(self);      // 释放对象内存
}

static PyTypeObject NodeIterType = {
    PyVarObject_HEAD_INIT(nullptr, 0) "torch._C._NodeIter", /* tp_name */
    sizeof(NodeIter), /* tp_basicsize */
    0, /* tp_itemsize */
    (destructor)NodeIter_dealloc, /* tp_dealloc */
    0, /* tp_vectorcall_offset */
    nullptr, /* tp_getattr */
    nullptr, /* tp_setattr */
    nullptr, /* tp_reserved */
    nullptr, /* tp_repr */
    nullptr, /* tp_as_number */
    nullptr, /* tp_as_sequence */
    nullptr, /* tp_as_mapping */
    // 设置为nullptr，表示该类型对象不支持映射协议
    nullptr, /* tp_hash  */
    // 设置为nullptr，表示该类型对象没有定义哈希方法
    nullptr, /* tp_call */
    // 设置为nullptr，表示该类型对象不支持调用操作
    nullptr, /* tp_str */
    // 设置为nullptr，表示该类型对象没有定义字符串表示方法
    nullptr, /* tp_getattro */
    // 设置为nullptr，表示该类型对象不支持getattr操作
    nullptr, /* tp_setattro */
    // 设置为nullptr，表示该类型对象不支持setattr操作
    nullptr, /* tp_as_buffer */
    // 设置为nullptr，表示该类型对象不支持缓冲区协议
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC, /* tp_flags */
    // 定义类型对象的标志，此处默认标志加上支持垃圾回收的标志
    nullptr, /* tp_doc */
    // 设置为nullptr，表示该类型对象没有文档字符串
    (traverseproc)NodeIter_traverse, /* tp_traverse */
    // 设置对象的遍历函数为NodeIter_traverse，用于垃圾回收时遍历对象
    (inquiry)NodeIter_clear, /* tp_clear */
    // 设置对象的清理函数为NodeIter_clear，用于释放对象资源
    nullptr, /* tp_richcompare */
    // 设置为nullptr，表示该类型对象没有定义富比较方法
    0, /* tp_weaklistoffset */
    // 弱引用列表的偏移量，此处设置为0，表示不支持弱引用
    PyObject_SelfIter, /* tp_iter */
    // 设置迭代器函数为PyObject_SelfIter，用于返回对象自身的迭代器
    NodeIter_iternext, /* tp_iternext */
    // 设置迭代下一个元素的函数为NodeIter_iternext
    nullptr, /* tp_methods */
    // 设置为nullptr，表示该类型对象没有方法
    nullptr, /* tp_members */
    // 设置为nullptr，表示该类型对象没有成员
    nullptr, /* tp_getset */
    // 设置为nullptr，表示该类型对象没有属性获取/设置方法
    nullptr, /* tp_base */
    // 设置为nullptr，表示该类型对象没有基类
    nullptr, /* tp_dict */
    // 设置为nullptr，表示该类型对象没有字典
    nullptr, /* tp_descr_get */
    // 设置为nullptr，表示该类型对象没有描述符获取方法
    nullptr, /* tp_descr_set */
    // 设置为nullptr，表示该类型对象没有描述符设置方法
    0, /* tp_dictoffset */
    // 字典偏移量为0，表示该类型对象没有字典存储在对象中
    (initproc)NodeIter_init_fn, /* tp_init */
    // 初始化函数为NodeIter_init_fn，用于对象初始化
    nullptr, /* tp_alloc */
    // 设置为nullptr，表示使用默认的内存分配方式
    NodeIter_new, /* tp_new */
    // 设置创建新对象的函数为NodeIter_new
};

bool NodeIter_init(PyObject* module) {
    // 向 Python 模块中添加 NodeIterType 类型对象
    if (PyModule_AddType(module, &NodeIterType) < 0) {
        // 如果添加失败，返回 false
        return false;
    }
    // 添加成功，返回 true
    return true;
}
```