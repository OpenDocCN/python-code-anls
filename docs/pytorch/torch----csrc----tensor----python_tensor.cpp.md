# `.\pytorch\torch\csrc\tensor\python_tensor.cpp`

```
// 引入 Torch 的 C++ Tensor 头文件
#include <torch/csrc/tensor/python_tensor.h>

// 引入 Pybind11 库，用于 Python 和 C++ 之间的接口转换
#include <pybind11/pybind11.h>

// 引入结构成员定义文件
#include <structmember.h>

// 引入 Torch 的 Python 绑定工具函数
#include <torch/csrc/utils/pybind.h>

// 引入 Torch 的数据类型相关头文件
#include <torch/csrc/Dtype.h>
#include <torch/csrc/DynamicTypes.h>

// 引入 Torch 的异常处理头文件
#include <torch/csrc/Exceptions.h>

// 引入 Torch 的张量布局头文件
#include <torch/csrc/Layout.h>

// 引入 Torch 的自动微分相关头文件
#include <torch/csrc/autograd/generated/VariableType.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/autograd/utils/wrap_outputs.h>
#include <torch/csrc/autograd/variable.h>

// 引入 Torch 的 CUDA 及设备初始化相关头文件
#include <torch/csrc/utils/cuda_enabled.h>
#include <torch/csrc/utils/device_lazy_init.h>

// 引入 Torch 的 Python 字符串处理工具头文件
#include <torch/csrc/utils/python_strings.h>

// 引入 Torch 的张量创建及类型转换头文件
#include <torch/csrc/utils/tensor_new.h>
#include <torch/csrc/utils/tensor_types.h>

// 引入 ATen 的核心头文件
#include <ATen/ATen.h>

// 引入标准库头文件
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

// 定义 Torch 的张量命名空间
namespace torch::tensors {

// 使用 ATen 命名空间
using namespace at;

// 使用 Torch 的自动微分命名空间
using namespace torch::autograd;

// 定义 Python 中的张量类型结构体
struct PyTensorType {
  PyTypeObject py_type;  // Python 类型对象
  THPDtype* dtype;       // Torch 的数据类型指针
  THPLayout* layout;     // Torch 的布局指针
  bool is_cuda;          // 是否使用 CUDA
  bool is_xpu;           // 是否使用 XPU（扩展处理单元）
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,cppcoreguidelines-avoid-magic-numbers,modernize-avoid-c-arrays)
  char name[64];         // 张量类型名称
  int backend;           // 后端类型
  int scalar_type;       // 标量类型

  // 获取后端类型
  Backend get_backend() const {
    return static_cast<Backend>(backend);
  }

  // 获取分派键（Dispatch Key）
  DispatchKey get_dispatch_key() const {
    return backendToDispatchKey(static_cast<Backend>(backend));
  }

  // 获取标量类型
  ScalarType get_scalar_type() const {
    return static_cast<ScalarType>(scalar_type);
  }
};

// 静态断言：确保 PyTensorType 是标准布局
static_assert(
    std::is_standard_layout_v<PyTensorType>,
    "PyTensorType must be standard layout");

// 默认后端为 CPU
static Backend default_backend = Backend::CPU;

// Python 绑定函数：绑定张量类型
static void py_bind_tensor_types(
    const std::vector<PyTensorType*>& tensor_types);

// Python 中的张量对象构造函数
static PyObject* Tensor_new(
    PyTypeObject* type,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  auto& tensor_type = *((PyTensorType*)type);

  // 检查是否允许使用 CUDA，如果当前未启用 CUDA，则抛出错误
  TORCH_CHECK_TYPE(
      !tensor_type.is_cuda || torch::utils::cuda_enabled(),
      "type ",
      tensor_type.name,
      " not available. Torch not compiled with CUDA enabled.")

  // 如果张量类型是 CUDA 类型，发出警告建议使用更现代的创建方式
  if (tensor_type.is_cuda) {
    TORCH_WARN_ONCE(
        "The torch.cuda.*DtypeTensor constructors are no longer recommended. "
        "It's best to use methods such as torch.tensor(data, dtype=*, device='cuda') to create tensors.")
  }

  // 使用 Torch 的旧式张量构造函数包装变量并返回
  return THPVariable_Wrap(torch::utils::legacy_tensor_ctor(
      tensor_type.get_dispatch_key(),
      tensor_type.get_scalar_type(),
      args,
      kwargs));
  END_HANDLE_TH_ERRORS
}

// TODO: 废弃此实例检查函数。它用于使 instanceof(t, torch.FloatTensor) 起作用，
// 但我们不会为每个新张量类型都添加 torch.QuantizedIntTensor 类...
static PyObject* Tensor_instancecheck(PyObject* _self, PyObject* arg) {
  HANDLE_TH_ERRORS
  auto self = (PyTensorType*)_self;

  // 如果参数是 THPVariable 类型，进行类型检查
  if (THPVariable_Check(arg)) {
    const auto& var = THPVariable_Unpack(arg);
    // 注意：这里的处理方式有些不太幸运，如果进行 isinstance 检查
    // ...
    // 检查变量的调度键集合是否包含与当前对象的调度键相同，并且标量类型是否匹配。
    // 如果条件成立，则返回 True，表示变量与当前对象类型匹配。
    if (legacyExtractDispatchKey(var.key_set()) == self->get_dispatch_key() &&
        var.scalar_type() == static_cast<ScalarType>(self->scalar_type)) {
      // 返回 Python 中的 True
      Py_RETURN_TRUE;
    }
  }
  // 如果未满足上述条件，则返回 Python 中的 False，表示变量与当前对象类型不匹配。
  Py_RETURN_FALSE;
  // 处理 Torch 引发的任何错误，并结束错误处理过程。
  END_HANDLE_TH_ERRORS
}

// 返回张量的数据类型包装后的 Python 对象
static PyObject* Tensor_dtype(PyTensorType* self, void* unused) {
  return torch::autograd::utils::wrap(self->dtype);
}

// 返回张量的布局类型包装后的 Python 对象
static PyObject* Tensor_layout(PyTensorType* self, void* unused) {
  return torch::autograd::utils::wrap(self->layout);
}

// 返回张量是否在 CUDA 上的 Python 布尔值对象
static PyObject* Tensor_is_cuda(PyTensorType* self, void* unused) {
  if (self->is_cuda) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
}

// 返回张量是否在 XPU 上的 Python 布尔值对象
static PyObject* Tensor_is_xpu(PyTensorType* self, void* unused) {
  if (self->is_xpu) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
}

// 返回张量是否为稀疏张量的 Python 布尔值对象
static PyObject* Tensor_is_sparse(PyTensorType* self, void* unused) {
  if (self->layout->layout == at::Layout::Strided) {
    Py_RETURN_FALSE;
  } else {
    Py_RETURN_TRUE;
  }
}

// 返回张量是否为稀疏 CSR 格式的 Python 布尔值对象
static PyObject* Tensor_is_sparse_csr(PyTensorType* self, void* unused) {
  if (self->layout->layout == at::Layout::SparseCsr) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
}

// 定义元类的方法数组，用于定义 Python 类的特殊方法
// 包含 "__instancecheck__" 方法，使用 Tensor_instancecheck 函数，接收一个参数，无返回描述
static struct PyMethodDef metaclass_methods[] = {
    {"__instancecheck__", Tensor_instancecheck, METH_O, nullptr},
    {nullptr}
};

// 定义元类的属性数组，用于定义 Python 类的特殊属性
// 包含 "dtype", "layout", "is_cuda", "is_xpu", "is_sparse", "is_sparse_csr" 属性
// 每个属性使用对应的 getter 函数，不使用 setter 或 deleter 函数
static struct PyGetSetDef metaclass_properties[] = {
    {"dtype", (getter)Tensor_dtype, nullptr, nullptr, nullptr},
    {"layout", (getter)Tensor_layout, nullptr, nullptr, nullptr},
    {"is_cuda", (getter)Tensor_is_cuda, nullptr, nullptr, nullptr},
    {"is_xpu", (getter)Tensor_is_xpu, nullptr, nullptr, nullptr},
    {"is_sparse", (getter)Tensor_is_sparse, nullptr, nullptr, nullptr},
    {"is_sparse_csr", (getter)Tensor_is_sparse_csr, nullptr, nullptr, nullptr},
    {nullptr}
};

// 定义元类对象的结构体
// 设置元类对象的名称为 "torch.tensortype"，基本大小为 PyTypeObject 结构体的大小
static PyTypeObject metaclass = {
    PyVarObject_HEAD_INIT(nullptr, 0) "torch.tensortype", /* tp_name */
    sizeof(PyTypeObject) /* tp_basicsize */
};

// 初始化元类对象的函数
// 设置元类对象的标志、方法数组、属性数组、基类为 PyType_Type
// 如果初始化失败，则抛出 python_error 异常
static void py_initialize_metaclass(PyTypeObject& metaclass) {
  metaclass.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
  metaclass.tp_methods = metaclass_methods;
  metaclass.tp_getset = metaclass_properties;
  metaclass.tp_base = &PyType_Type;
  if (PyType_Ready(&metaclass) < 0) {
    throw python_error();
  }
}

// 定义张量类型原型对象的结构体
// 使用 metaclass 作为基类，基本大小为 PyTensorType 结构体的大小
static PyTypeObject tensor_type_prototype = {
    PyVarObject_HEAD_INIT(&metaclass, 0) nullptr, /* tp_name */
    sizeof(PyTensorType) /* tp_basicsize */
};

// 初始化张量类型对象的函数
// 设置张量类型对象的名称、基类、基本大小
static void py_initialize_tensor_type(
    PyTypeObject& type,
    const char* name,
    // NOTE: we don't use the typical static declaration of PyTypeObject because
    // we need to initialize as many types as there are VariableType instances.
    // We copy the basic object fields from a prototype definition and initialize
    // the remaining fields below.
    // 注意：我们不使用典型的静态声明 PyTypeObject，因为我们需要初始化与 VariableType 实例数量相同的类型。
    // 我们从原型定义复制基本对象字段，并在下面初始化其余字段。
    
    memcpy(&type, &tensor_type_prototype, sizeof(PyTypeObject));
    // 使用 memcpy 将 tensor_type_prototype 的基本对象字段复制到 type 中，大小为 PyTypeObject 的大小。
    
    // Subclassing from torch.<ScalarType>Tensor isn't supported.
    // (Py_TPFLAGS_BASETYPE omitted). Subclassing torch.Tensor still allowed.
    // 不支持从 torch.<ScalarType>Tensor 派生子类。
    // （忽略 Py_TPFLAGS_BASETYPE）。仍允许从 torch.Tensor 派生子类。
    type.tp_flags = Py_TPFLAGS_DEFAULT;
    // 将 type 的标志设置为 Py_TPFLAGS_DEFAULT。
    
    type.tp_name = name;
    // 设置 type 的名称为变量 name 所指向的名称。
    
    type.tp_new = Tensor_new;
    // 将 type 的 tp_new 指针设置为 Tensor_new 函数的地址，用于实例化新对象。
    
    if (PyType_Ready(&type) < 0) {
      throw python_error();
    }
    // 准备 type 类型，如果失败则抛出 python_error 异常。
    
    if (PyDict_Merge(type.tp_dict, tp_dict, 0) < 0) {
      throw python_error();
    }
    // 将 tp_dict 合并到 type.tp_dict 中，如果失败则抛出 python_error 异常。
// 结束静态成员函数 get_name 的定义
static std::string get_name(Backend backend, ScalarType scalarType) {
  // 创建一个字符串流对象
  std::ostringstream ss;
  // 将 backend 转换为字符串并追加到流中
  ss << torch::utils::backend_to_string(backend) << "." << toString(scalarType)
     << "Tensor";
  // 返回流中的字符串表示
  return ss.str();
}

// 返回存储对象的 Python 对象指针
static THPObjectPtr get_storage_obj(Backend backend, ScalarType dtype) {
  // 获取后端对应的模块名
  auto module_name = torch::utils::backend_to_string(backend);
  // 导入该模块
  auto module_obj = THPObjectPtr(PyImport_ImportModule(module_name));
  // 如果导入失败，则抛出 Python 异常
  if (!module_obj)
    throw python_error();

  // 获取存储类型的名称
  auto storage_name = std::string(toString(dtype)) + "Storage";
  // 获取该模块中的存储对象
  THPObjectPtr storage(
      PyObject_GetAttrString(module_obj.get(), storage_name.c_str()));
  // 检查获取的对象类型是否正确，否则抛出异常
  TORCH_CHECK_TYPE(
      storage.get(), "couldn't find storage object ", storage_name);
  // 返回存储对象的 Python 对象指针
  return storage;
}

// 设置 PyTensorType 对象的类型信息
static void set_type(
    PyTensorType& type_obj,
    Backend backend,
    ScalarType scalarType) {
  // 将后端转换为整数并赋值给对象的 backend 字段
  type_obj.backend = static_cast<int>(backend);
  // 将标量类型转换为整数并赋值给对象的 scalar_type 字段
  type_obj.scalar_type = static_cast<int>(scalarType);
  // 根据后端获取布局对象，并使用 Py_NewRef 来增加引用计数后赋值给对象的 layout 字段
  type_obj.layout =
      (THPLayout*)Py_NewRef(torch::getTHPLayout(layout_from_backend(backend)));
  // 根据标量类型获取数据类型对象，并使用 Py_NewRef 来增加引用计数后赋值给对象的 dtype 字段
  type_obj.dtype = (THPDtype*)Py_NewRef(torch::getTHPDtype(scalarType));
  // 检查是否是 CUDA 后端或者稀疏 CUDA 后端，并赋值给对象的 is_cuda 字段
  type_obj.is_cuda =
      (backend == at::Backend::CUDA || backend == at::Backend::SparseCUDA);
  // 检查是否是 XPU 后端或者稀疏 XPU 后端，并赋值给对象的 is_xpu 字段
  type_obj.is_xpu =
      (backend == at::Backend::XPU || backend == at::Backend::SparseXPU);
}

// 设置 PyTensorType 对象的名称
static void set_name(PyTensorType& type_obj, const std::string& name) {
  // 获取对象名称的最大长度
  size_t n = sizeof(type_obj.name);
  // 将给定名称拷贝到对象的 name 字段中，并确保以 '\0' 结尾
  strncpy(type_obj.name, name.c_str(), n);
  type_obj.name[n - 1] = '\0';
}

// 获取 Tensor 类型的字典对象
static THPObjectPtr get_tensor_dict() {
  // 导入 torch 模块
  auto torch = THPObjectPtr(PyImport_ImportModule("torch"));
  // 如果导入失败，则抛出 Python 异常
  if (!torch)
    throw python_error();

  // 获取 Tensor 类的 Python 对象
  auto tensor_class = THPObjectPtr(PyObject_GetAttrString(torch, "Tensor"));
  // 如果获取失败，则抛出 Python 异常
  if (!tensor_class)
    throw python_error();

  // 获取 Tensor 类的类型对象
  auto tensor_type = (PyTypeObject*)tensor_class.get();
  // 检查 Tensor 类型对象是否有基类，如果没有则抛出异常
  TORCH_CHECK(tensor_type->tp_base, "missing base type for Tensor");

  // 创建一个新的空字典对象
  auto res = THPObjectPtr(PyDict_New());
  // 如果创建失败，则抛出 Python 异常
  if (!res)
    throw python_error();

  // 将 Tensor 类的字典属性合并到 res 字典中
  if (PyDict_Merge(res.get(), tensor_type->tp_dict, 0) < 0) {
    throw python_error();
  }
  // 将 Tensor 类的基类字典属性合并到 res 字典中
  if (PyDict_Merge(res.get(), tensor_type->tp_base->tp_dict, 0) < 0) {
    throw python_error();
  }

  // 返回合并后的字典对象
  return res;
}

// 关于 PyTensorType 生命周期的注释
// PyTypeObject 实例通常是静态分配的，但我们希望在初始化时动态创建它们，
// 因为它们的确切数量取决于 torch::utils::all_declared_types()。
// 每个 PyTensorType 的内存由 initialize_aten_types() 分配，且永远不会释放：
// 从技术上讲是一个内存泄漏，但因为我们希望它们在整个进程生命周期内都保持活跃，
// 所以这不是问题。
//
// 另一种选择是使用 std::vector<PyTensorType>，并让 std::vector 管理其项的生命周期。
// 然而，这种做法有问题，因为它意味着 PyTensorType 的内存会在它们的生命周期结束时被释放。
// 全局静态变量，存储 PyTensorType 指针的向量，用于管理注册的张量类型
// 如果在程序退出时有其他全局析构函数或 atexit() 函数尝试访问 PyTensorTypes，
// 可能会导致 use-after-free 错误。例如，如果嵌入了 CPython 并在导入 torch 之前
// 调用 Py_Finalize()，而 atexit() 函数在注册时已经存在。
static std::vector<PyTensorType*> tensor_types;

// 设置默认的张量存储类型
static void set_default_storage_type(Backend backend, ScalarType dtype) {
  // 获取指定 backend 和 dtype 的存储对象
  THPObjectPtr storage = get_storage_obj(backend, dtype);

  // 导入 torch 模块
  auto torch_module = THPObjectPtr(PyImport_ImportModule("torch"));
  if (!torch_module)
    throw python_error();

  // 将 storage 对象设置为 torch 模块的属性 "Storage"
  if (PyObject_SetAttrString(torch_module.get(), "Storage", storage) != 0) {
    throw python_error();
  }
}

// 设置默认的张量类型
static void set_default_tensor_type(
    std::optional<Backend> backend,
    std::optional<ScalarType> dtype) {
  // 如果有指定 backend，则进行类型检查
  if (backend.has_value()) {
    TORCH_CHECK_TYPE(
        *backend != Backend::Undefined, "default type cannot be undefined");
    TORCH_CHECK_TYPE(
        !isSparse(*backend),
        "only dense types are supported as the default type");
  }
  // 如果有指定 dtype，则进行类型检查
  if (dtype.has_value()) {
    TORCH_CHECK_TYPE(
        at::isFloatingType(*dtype),
        "only floating-point types are supported as the default type");
  }

  // 首先尝试在 Python 环境中设置默认存储类型，因为这是唯一可能失败的操作
  set_default_storage_type(
      backend.value_or(default_backend),
      dtype.value_or(at::get_default_dtype_as_scalartype()));

  // 如果指定了 dtype，则设置为默认的张量数据类型
  if (dtype.has_value()) {
    at::set_default_dtype(scalarTypeToTypeMeta(*dtype));
  }
  // 如果指定了 backend，则更新默认的 backend
  if (backend.has_value()) {
    default_backend = *backend;
  }
}

// 初始化 ATen 类型，为每种声明的类型创建 PyTensorType 对象
static void initialize_aten_types(std::vector<PyTensorType*>& tensor_types) {
  // 获取所有声明的类型，包括即使在没有构建 CUDA 的情况下也包括 CUDA 类型
  auto declared_types = torch::utils::all_declared_types();
  // 调整 tensor_types 的大小以容纳所有声明的类型
  tensor_types.resize(declared_types.size());

  // 遍历所有声明的类型
  for (size_t i = 0, end = declared_types.size(); i != end; i++) {
    // 创建新的 PyTensorType 对象
    tensor_types[i] = new PyTensorType();
    auto& tensor_type = *tensor_types[i];
    Backend backend = declared_types[i].first;
    ScalarType scalar_type = declared_types[i].second;
    // 设置 PyTensorType 对象的类型和名称
    set_type(tensor_type, backend, scalar_type);
    set_name(tensor_type, get_name(backend, scalar_type));
  }

  // 设置默认的张量类型为 CPU 和 Float
  set_default_tensor_type(Backend::CPU, ScalarType::Float);
}
// 初始化 Python 绑定
void initialize_python_bindings() {
  // 初始化 at::Type* 指针、名称和 PyTensorType 向量的属性。
  // 此调用后，不能调整向量大小。
  initialize_aten_types(tensor_types);

  // 初始化 torch.FloatTensor 等类型的 Python 元类。
  // 元类处理 __instancecheck__ 检查，并绑定类型对象的 dtype 属性。
  py_initialize_metaclass(metaclass);

  // 获取 Variable 类的 tp_dict。我们将函数定义复制到每个 Tensor 类型对象上，
  // 以便可以通过例如 `torch.FloatTensor.add` 访问它们。
  auto tensor_dict = get_tensor_dict();

  // 初始化每个 Python 类型对象 torch.FloatTensor、torch.DoubleTensor 等。
  for (auto& tensor_type : tensor_types) {
    py_initialize_tensor_type(
        tensor_type->py_type, tensor_type->name, tensor_dict.get());
  }

  // 将类型对象添加到它们对应的模块中。例如 torch.FloatTensor
  // 作为 `FloatTensor` 添加到 `torch` 模块中。
  // 同时将所有类型对象添加到集合 torch._tensor_classes 中。
  py_bind_tensor_types(tensor_types);
}

// 将类型对象绑定到 Python
static void py_bind_tensor_types(
    const std::vector<PyTensorType*>& tensor_types) {
  // 导入 torch 模块
  auto torch_module = THPObjectPtr(PyImport_ImportModule("torch"));
  if (!torch_module)
    throw python_error();

  // 获取 torch._tensor_classes 集合
  auto tensor_classes = THPObjectPtr(
      PyObject_GetAttrString(torch_module.get(), "_tensor_classes"));
  if (!tensor_classes)
    throw python_error();

  // 遍历每个 PyTensorType 类型对象
  for (auto& tensor_type : tensor_types) {
    // 获取类型对象的名称和模块名
    auto name = std::string(tensor_type->name);
    auto idx = name.rfind('.');
    auto type_name = name.substr(idx + 1);
    auto module_name = name.substr(0, idx);

    // 导入模块对象
    auto module_obj = THPObjectPtr(PyImport_ImportModule(module_name.c_str()));
    if (!module_obj)
      throw python_error();

    // 增加类型对象的引用计数并将其添加到模块中
    PyObject* type_obj = (PyObject*)tensor_type;
    Py_INCREF(type_obj);
    if (PyModule_AddObject(module_obj.get(), type_name.c_str(), type_obj) < 0) {
      throw python_error();
    }
    // 将类型对象添加到 torch._tensor_classes 集合中
    if (PySet_Add(tensor_classes.get(), type_obj) < 0) {
      throw python_error();
    }
  }
}

// 检查 PyTensorType 对象
static bool PyTensorType_Check(PyObject* obj) {
  // 在 tensor_types 向量中查找与给定对象相匹配的 PyTensorType 指针
  auto it = std::find_if(
      tensor_types.begin(), tensor_types.end(), [obj](PyTensorType* x) {
        return (PyObject*)x == obj;
      });
  // 返回是否找到匹配的 PyTensorType 对象
  return it != tensor_types.end();
}
void py_set_default_tensor_type(PyObject* obj) {
``` 
// 定义一个名为 `py_set_default_tensor_type` 的函数，接受一个名为 `obj` 的 PyObject 指针参数。


  TORCH_WARN_ONCE(
      "torch.set_default_tensor_type() is deprecated as of PyTorch 2.1, "
      "please use torch.set_default_dtype() and torch.set_default_device() as alternatives.")
``` 
// 发出一次性警告，指出 `torch.set_default_tensor_type()` 自 PyTorch 2.1 起已弃用，建议使用 `torch.set_default_dtype()` 和 `torch.set_default_device()` 作为替代方案。


  TORCH_CHECK_TYPE(
      PyTensorType_Check(obj),
      "invalid type object: only floating-point types are supported as the default type");
``` 
// 检查 `obj` 是否为 `PyTensorType` 类型，如果不是，抛出类型错误，提示仅支持浮点类型作为默认类型。


  PyTensorType* type = (PyTensorType*)obj;
``` 
// 将 `obj` 强制转换为 `PyTensorType*` 类型，存储在 `type` 变量中。


  TORCH_CHECK_TYPE(
      !type->is_cuda || torch::utils::cuda_enabled(),
      "type ",
      type->name,
      " not available. Torch not compiled with CUDA enabled.")
``` 
// 检查 `type` 是否支持 CUDA，如果不支持且当前环境未启用 CUDA，则抛出错误，指明具体类型 `type->name` 在未启用 CUDA 的情况下不可用。


  set_default_tensor_type(type->get_backend(), type->get_scalar_type());
``` 
// 调用 `set_default_tensor_type` 函数，设置默认的张量类型，传入 `type` 的后端和标量类型信息。


}

void py_set_default_dtype(PyObject* obj) {
``` 
// 定义一个名为 `py_set_default_dtype` 的函数，接受一个名为 `obj` 的 PyObject 指针参数。


  TORCH_CHECK_TYPE(
      THPDtype_Check(obj),
      "invalid dtype object: only floating-point types are supported as the default type");
``` 
// 检查 `obj` 是否为 `THPDtype` 类型，如果不是，抛出类型错误，提示仅支持浮点类型作为默认类型。


  auto scalar_type = ((THPDtype*)obj)->scalar_type;
``` 
// 将 `obj` 强制转换为 `THPDtype*` 类型，并获取其标量类型信息，存储在 `scalar_type` 变量中。


  set_default_tensor_type(/*backend=*/c10::nullopt, scalar_type);
``` 
// 调用 `set_default_tensor_type` 函数，设置默认的张量类型，传入空的后端选项和 `scalar_type` 标量类型信息。


}

c10::DispatchKey get_default_dispatch_key() {
``` 
// 定义一个名为 `get_default_dispatch_key` 的函数，返回类型为 `c10::DispatchKey`。


  return backendToDispatchKey(default_backend);
``` 
// 调用 `backendToDispatchKey` 函数，将默认后端转换为 `DispatchKey` 并返回。


}

at::Device get_default_device() {
``` 
// 定义一个名为 `get_default_device` 的函数，返回类型为 `at::Device`。


  return at::Device(c10::backendToDeviceType(default_backend));
``` 
// 调用 `backendToDeviceType` 函数，将默认后端转换为设备类型并包装为 `at::Device` 对象返回。


}

ScalarType get_default_scalar_type() {
``` 
// 定义一个名为 `get_default_scalar_type` 的函数，返回类型为 `ScalarType`。


  return get_default_dtype_as_scalartype();
``` 
// 调用 `get_default_dtype_as_scalartype` 函数，获取默认的标量类型并返回。


}

} // namespace torch::tensors
``` 
// 结束 `torch::tensors` 命名空间的定义。
```