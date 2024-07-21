# `.\pytorch\torch\csrc\utils\tensor_new.cpp`

```py
// 包含 Torch 的 Python 头文件
#include <torch/csrc/python_headers.h>
// 包含 Torch 的张量创建工具函数头文件
#include <torch/csrc/utils/tensor_new.h>

// 包含 Pybind11 库的头文件
#include <pybind11/pybind11.h>
// Torch 的动态类型定义
#include <torch/csrc/DynamicTypes.h>
// Torch 的异常定义
#include <torch/csrc/Exceptions.h>
// Torch 的张量大小定义
#include <torch/csrc/Size.h>
// Torch 的自动求导变量工厂函数
#include <torch/csrc/autograd/generated/variable_factories.h>
// Torch 的自动求导变量定义
#include <torch/csrc/autograd/variable.h>
// Torch 的设备惰性初始化工具
#include <torch/csrc/utils/device_lazy_init.h>
// Torch 的 NumPy 兼容性工具
#include <torch/csrc/utils/numpy_stub.h>
// Torch 的 Python 工具
#include <torch/csrc/utils/pybind.h>
// Torch 的 Python 参数解析器
#include <torch/csrc/utils/python_arg_parser.h>
// Torch 的 Python 数字处理工具
#include <torch/csrc/utils/python_numbers.h>
// Torch 的 Python 标量处理工具
#include <torch/csrc/utils/python_scalars.h>
// Torch 的 Python 字符串处理工具
#include <torch/csrc/utils/python_strings.h>
// Torch 的张量与 NumPy 互操作工具
#include <torch/csrc/utils/tensor_numpy.h>

// ATen 张量库
#include <ATen/ATen.h>
// ATen 数据加载转换工具
#include <ATen/DLConvertor.h>
// ATen 初始张量选项
#include <ATen/InitialTensorOptions.h>
// ATen 命名张量工具
#include <ATen/NamedTensorUtils.h>
// ATen 原生功能函数
#include <ATen/NativeFunctions.h>
// ATen 稀疏 CSR 张量工具
#include <ATen/SparseCsrTensorUtils.h>
// ATen 追踪模式
#include <ATen/TracerMode.h>
// ATen 数据加载包装
#include <ATen/dlpack.h>
// C10 核心后端
#include <c10/core/Backend.h>
// C10 分发键集
#include <c10/core/DispatchKeySet.h>
// C10 张量布局
#include <c10/core/Layout.h>
// C10 异常处理
#include <c10/util/Exception.h>
// C10 可选类型
#include <c10/util/Optional.h>
// C10 整数范围
#include <c10/util/irange.h>

// 标准异常库
#include <stdexcept>
// 标准向量库
#include <vector>

// 使用 ATen 命名空间下的设备和整数数组引用
using at::Device;
using at::IntArrayRef;
// 使用 ATen 下的整型和长整型标量类型
using at::kInt;
using at::kLong;
using at::ScalarType;
using at::Storage;
using at::Tensor;
using at::TensorOptions;
// 使用 C10 下的可选类型
using std::optional;

// Torch 的工具命名空间
namespace torch::utils {
// 匿名命名空间中定义的常量，最大维度为 128
namespace {
const int MAX_DIMS = 128;

// 线程局部变量，标识是否仅提升 CPU 张量
thread_local bool kOnlyLiftCPUTensors = false;

// 构建张量选项函数，接受 ATen 的张量选项、标量类型和可选设备，返回张量选项
TensorOptions build_options(
    c10::TensorOptions options,
    at::ScalarType scalar_type,
    const std::optional<Device>& device = c10::nullopt) {
  // 设置选项的数据类型为给定的标量类型
  options = options.dtype(scalar_type);
  // 如果设备值存在，则设置选项的设备
  if (device.has_value()) {
    return options.device(device);
  }
  return options;
}

// 创建指定大小的新张量函数，接受 ATen 的张量选项、标量类型、可选设备和大小数组引用
Tensor new_with_sizes(
    c10::TensorOptions options,
    at::ScalarType scalar_type,
    const optional<Device>& device,
    c10::SymIntArrayRef sizes) {
  // 可能初始化设备
  maybe_initialize_device(options.device());
  // 释放全局解释器锁
  pybind11::gil_scoped_release no_gil;
  // 返回使用指定大小和选项创建的空张量
  return at::empty_symint(sizes, build_options(options, scalar_type, device));
}

// 使用给定存储创建新张量函数，接受 ATen 的张量选项、标量类型和存储
Tensor new_with_storage(
    c10::TensorOptions options,
    at::ScalarType scalar_type,
    Storage storage) {
  // 创建空张量
  auto tensor = at::empty({}, build_options(options, scalar_type));
  // 将给定存储设置为张量的存储
  tensor.set_(std::move(storage));
  // 返回设置了存储的张量
  return tensor;
}

// 计算给定 Python 对象序列的大小函数，接受 Python 对象指针和标量类型
std::vector<int64_t> compute_sizes(PyObject* seq, ScalarType scalar_type) {
  // 判断是否为存储对象
  bool is_storage = isStorage(seq);
  // 大小向量初始化
  std::vector<int64_t> sizes;
  // 注意：在第一次迭代之后，obj 是唯一保持 seq 原始指针有效的对象
  THPObjectPtr obj;
  // 循环检查是否为 Python 序列
  while (PySequence_Check(seq)) {
    // 获取序列长度
    auto length = PySequence_Length(seq);
    // 如果长度小于 0 抛出 Python 异常
    if (length < 0)
      throw python_error();


这里只展示了部分代码的注释，因为完整的代码可能会超出单个回答的长度限制。
    // 如果 is_storage 为真，则将 length 除以元素大小得到实际长度
    if (is_storage) {
      length /= static_cast<int64_t>(elementSize(scalar_type));
    }
    // 将计算得到的长度添加到 sizes 容器中
    sizes.push_back(length);
    // 检查维度数是否超过最大限制 MAX_DIMS，若超过则抛出异常
    TORCH_CHECK_VALUE(
        sizes.size() <= MAX_DIMS,
        "too many dimensions '",
        Py_TYPE(seq)->tp_name,
        "'");
    // 如果长度为 0，则跳出循环
    if (length == 0)
      break;
    // 获取序列对象 seq 的第一个元素，并将其赋给 new_obj
    PyObject* new_obj = PySequence_GetItem(seq, 0);
    // 检查获取的对象 new_obj 是否有效，若无效则抛出异常
    // 注意：在此行之前不能覆盖 obj 对象
    TORCH_CHECK_VALUE(
        new_obj,
        "could not determine the shape of object type '",
        Py_TYPE(seq)->tp_name,
        "'");
    // 使用 new_obj 构造 THPObjectPtr 对象，并将其赋给 obj
    obj = THPObjectPtr(new_obj);
    // 将 obj 对象的底层指针赋给 seq，准备下一次循环
    seq = obj.get();
  }

  // 返回存储各维度大小的 sizes 容器
  return sizes;
}

// 推断给定 Python 对象的标量类型
ScalarType infer_scalar_type(PyObject* obj) {
  // 如果是符号整数，返回长整型标量类型
  if (torch::is_symint(obj)) {
    return ScalarType::Long;
  }
  // 如果是符号浮点数，返回默认张量标量类型
  if (torch::is_symfloat(obj)) {
    return torch::tensors::get_default_scalar_type();
  }
#ifdef USE_NUMPY
  // 如果 NumPy 可用，检查是否为 NumPy 数组，转换为 ATen 标量类型
  if (is_numpy_available()) {
    if (PyArray_Check(obj)) {
      return numpy_dtype_to_aten(PyArray_TYPE((PyArrayObject*)obj));
    }
    // 如果是 NumPy 标量，从标量创建 NumPy 数组，并转换为 ATen 标量类型
    if (PyArray_CheckScalar(obj)) {
      THPObjectPtr arr(PyArray_FromScalar(obj, nullptr));
      return numpy_dtype_to_aten(PyArray_TYPE((PyArrayObject*)arr.get()));
    }
  }
#endif
  // 如果是 Python 浮点数，返回默认张量标量类型
  if (PyFloat_Check(obj)) {
    // 总是浮点类型，便于编写如 torch.tensor(0.) 的代码
    return torch::tensors::get_default_scalar_type();
  }
  // 如果是 Python 长整数，返回长整型标量类型
  if (THPUtils_checkLong(obj)) {
    return ScalarType::Long;
  }
  // 如果是 Python 布尔值，返回布尔型标量类型
  if (PyBool_Check(obj)) {
    return ScalarType::Bool;
  }
  // 如果是 Python 复数，根据默认张量类型返回对应的复数类型
  if (PyComplex_Check(obj)) {
    switch (torch::tensors::get_default_scalar_type()) {
      case ScalarType::Float:
        return ScalarType::ComplexFloat;
      case ScalarType::Double:
        return ScalarType::ComplexDouble;
      case ScalarType::Half:
        return ScalarType::ComplexHalf;
      default:
        TORCH_CHECK(false, "invalid default scalar type for complex");
    }
  }
  // 如果是 PyTorch 变量，返回其标量类型
  if (THPVariable_Check(obj)) {
    const auto& var = THPVariable_Unpack(obj);
    return var.scalar_type();
  }
  // 检查是否为字符串类型，抛出类型错误
  TORCH_CHECK_TYPE(
      !THPUtils_checkString(obj),
      "new(): invalid data type '",
      Py_TYPE(obj)->tp_name,
      "'");
  // 如果是 Python 序列类型，递归推断每个元素的标量类型，并返回最终的标量类型
  if (PySequence_Check(obj)) {
    std::optional<ScalarType> scalarType;
    auto length = PySequence_Length(obj);
    if (length < 0)
      throw python_error();
    // 遵循 NumPy 语义，使用默认张量类型而不是 double 类型
    if (length == 0)
      return torch::tensors::get_default_scalar_type();
    for (const auto i : c10::irange(length)) {
      THPObjectPtr handle(PySequence_GetItem(obj, i));
      if (!handle)
        throw python_error();
      auto cur_item = handle.get();
      // 检查是否为自引用列表，不允许自引用
      TORCH_CHECK_TYPE(
          cur_item != obj, "new(): self-referential lists are incompatible");
      // 推断当前元素的标量类型
      ScalarType item_scalarType = infer_scalar_type(cur_item);
      // 根据元素的标量类型推断整体的标量类型
      scalarType = (scalarType) ? at::promoteTypes(*scalarType, item_scalarType)
                                : item_scalarType;
      // 如果是复数类型，则直接返回复数类型
      if (scalarType == ScalarType::ComplexDouble) {
        return *scalarType;
      }
    }
    // 返回推断得到的标量类型
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    return *scalarType;
  }
  // 如果无法推断数据类型，则抛出错误
  AT_ERROR("Could not infer dtype of ", Py_TYPE(obj)->tp_name);
}

// 递归存储函数，存储多维数组数据
void recursive_store(
    char* data,
    IntArrayRef sizes,
    IntArrayRef strides,
    int64_t dim,
    ScalarType scalarType,
    size_t elementSize,
    // 对象不应为空的断言检查
    PyObject* obj) {
  // 使用 sizes 的大小来确定张量的维度数量
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(data != nullptr);

  // 将 sizes 的大小转换为 int64_t 类型的变量 ndim
  int64_t ndim = static_cast<int64_t>(sizes.size());
  // 检查对象是否为符号浮点数
  bool is_symfloat = torch::is_symfloat(obj);
  // 检查对象是否为符号整数
  bool is_symint = torch::is_symint(obj);
  // 如果 dim 等于 ndim，则进行处理
  if (dim == ndim) {
    // 如果对象是符号浮点数类型
    if (is_symfloat) {
      // 重新解释 obj 为 py::object 对象
      auto new_obj = py::reinterpret_borrow<py::object>(obj);
      // 将 obj 转换为 c10::SymFloat 类型的变量 val
      auto val = new_obj.cast<c10::SymFloat>();
      // 从 val 中获取受保护的浮点数值
      const double double_val = val.guard_float(__FILE__, __LINE__);
      // 根据 elementSize 的大小将 double_val 赋值给 data
      switch (elementSize) {
        case 8:
          *reinterpret_cast<double*>(data) = double_val;
          break;
        case 4:
          *reinterpret_cast<float*>(data) = static_cast<float>(double_val);
          break;
      }
      return;
    }
    // 如果对象是符号整数类型
    if (is_symint) {
      // 重新解释 obj 为 py::object 对象
      auto new_obj = py::reinterpret_borrow<py::object>(obj);
      // 将 obj 转换为 c10::SymInt 类型的变量 val
      auto val = new_obj.cast<c10::SymInt>();
      // 从 val 中获取受保护的整数值
      const auto int_val = val.guard_int(__FILE__, __LINE__);
      // 根据 elementSize 的大小将 int_val 赋值给 data
      switch (elementSize) {
        case 8:
          *reinterpret_cast<int64_t*>(data) = int_val;
          break;
        case 4:
          *reinterpret_cast<int32_t*>(data) = static_cast<int32_t>(int_val);
          break;
        case 2:
          *reinterpret_cast<int16_t*>(data) = static_cast<int16_t>(int_val);
          break;
        case 1:
          *reinterpret_cast<int8_t*>(data) = static_cast<int8_t>(int_val);
          break;
        default:
          // 如果 elementSize 不在预期范围内，则抛出异常
          TORCH_CHECK(false, "Unexpected elementSize ", elementSize);
      }
      return;
    }
    // 如果不是符号浮点数或符号整数，则使用 torch::utils::store_scalar 存储标量值到 data
    torch::utils::store_scalar(data, scalarType, obj);
    return;
  }

  // 获取 sizes[dim] 的值，并将其存储到变量 n 中
  auto n = sizes[dim];
  // 快速获取 obj 的 Python 序列对象的指针
  auto seq = THPObjectPtr(PySequence_Fast(obj, "not a sequence"));
  // 如果 seq 为空指针，则抛出 Python 错误异常
  if (!seq)
    throw python_error();
  // 获取序列的长度
  // NOLINTNEXTLINE(bugprone-branch-clone)
  auto seq_size = PySequence_Fast_GET_SIZE(seq.get());
  // 检查序列长度是否与 n 相等，否则抛出错误
  TORCH_CHECK_VALUE(
      seq_size == n,
      "expected sequence of length ",
      n,
      " at dim ",
      dim,
      " (got ",
      seq_size,
      ")");

  // 获取序列的所有项的指针
  PyObject** items = PySequence_Fast_ITEMS(seq.get());
  // 遍历序列中的每一项
  for (const auto i : c10::irange(n)) {
#ifdef USE_NUMPY
    // 检查是否使用了 NumPy，并且输入的数据项为 NumPy 数组
    if (is_numpy_available() && PyArray_Check(items[i])) {
      TORCH_WARN_ONCE(
          "Creating a tensor from a list of numpy.ndarrays is extremely slow. "
          "Please consider converting the list to a single numpy.ndarray with "
          "numpy.array() before converting to a tensor.");
    }
#endif
    // 递归存储数据，处理数据、尺寸、步长、维度等参数
    recursive_store(
        data, sizes, strides, dim + 1, scalarType, elementSize, items[i]);
    // 更新数据指针位置，以便下一个数据项
    data += strides[dim] * elementSize;
  }
}

// 从给定数据创建新的张量
Tensor internal_new_from_data(
    c10::TensorOptions options,                  // 张量选项
    at::ScalarType scalar_type,                  // 标量类型
    std::optional<Device> device_opt,            // 设备选项（可选）
    PyObject* data,                              // 输入的 Python 对象数据
    bool copy_variables,                         // 是否复制变量
    bool copy_numpy,                             // 是否从 NumPy 复制
    bool type_inference,                         // 是否进行类型推断
    bool pin_memory = false) {                   // 是否将张量固定在内存中（默认不固定）

  // 检查输入数据类型是否有效，不应为字符串
  TORCH_CHECK_TYPE(
      !THPUtils_checkString(data),
      "new(): invalid data type '",
      Py_TYPE(data)->tp_name,
      "'");

  // 如果输入数据是 PyTorch 变量
  if (THPVariable_Check(data)) {
    // 不允许从变量构建的张量固定在内存中
    TORCH_CHECK(!pin_memory, "Can't pin tensor constructed from a variable");
    // TODO: 使用 MaybeOwned
    auto var = THPVariable_Unpack(data);
    // 如果需要复制变量，则分离变量
    if (copy_variables) {
      var = var.detach();
    }
    // 推断标量类型和设备类型；不期望推断布局，因为这些构造函数是按布局类型定义的（例如张量 vs sparse_coo_tensor）
    const auto& inferred_scalar_type =
        type_inference ? var.scalar_type() : scalar_type;
    // 获取设备并初始化设备
    auto device = device_opt.has_value() ? *device_opt : var.device();
    pybind11::gil_scoped_release no_gil;
    maybe_initialize_device(device);
    // 返回变量转换为指定设备和标量类型的张量
    return var.to(
        device,
        inferred_scalar_type,
        /*non_blocking=*/false,
        /*copy=*/copy_variables);
  }

#ifdef USE_NUMPY
  // 如果数据具有 "__cuda_array_interface__" 属性
  if (PyObject_HasAttrString(data, "__cuda_array_interface__")) {
    // 不允许从 __cuda_array_interface__ 构建的张量固定在内存中
    TORCH_CHECK(
        !pin_memory,
        "Can't pin tensor constructed from __cuda_array_interface__");
    // 从 __cuda_array_interface__ 创建张量
    auto tensor = tensor_from_cuda_array_interface(data);
    // 推断标量类型和设备类型
    const auto& inferred_scalar_type =
        type_inference ? tensor.scalar_type() : scalar_type;
    // 获取设备并初始化设备
    auto device = device_opt.has_value() ? *device_opt : options.device();
    pybind11::gil_scoped_release no_gil;
    maybe_initialize_device(device);
    // 返回张量转换为指定设备和标量类型的结果
    return tensor.to(
        device,
        inferred_scalar_type,
        /*non_blocking=*/false,
        /*copy=*/copy_numpy);
  }

  // 如果 NumPy 可用且数据为 NumPy 数组
  if (is_numpy_available() && PyArray_Check(data)) {
    // 不允许从 NumPy 构建的张量固定在内存中
    TORCH_CHECK(!pin_memory, "Can't pin tensor constructed from numpy");
    // 从 NumPy 数组创建张量
    auto tensor =
        tensor_from_numpy(data, /*warn_if_not_writeable=*/!copy_numpy);
    // 推断标量类型和设备类型
    const auto& inferred_scalar_type =
        type_inference ? tensor.scalar_type() : scalar_type;
    // 获取设备并初始化设备
    auto device = device_opt.has_value() ? *device_opt : options.device();
    pybind11::gil_scoped_release no_gil;
    maybe_initialize_device(device);
    // 返回张量转换为指定设备和标量类型的结果
    return tensor.to(
        device,
        inferred_scalar_type,
        /*non_blocking=*/false,
        /*copy=*/copy_numpy);
  }
#endif
#endif

  // 如果设备可选值存在，则使用该值，否则使用选项中的设备
  auto device = device_opt.has_value() ? *device_opt : options.device();

  // 计算数据的大小
  auto sizes = compute_sizes(data, scalar_type);

  // 推断标量类型：如果开启类型推断，则推断数据的标量类型；否则使用指定的标量类型
  ScalarType inferred_scalar_type =
      type_inference ? infer_scalar_type(data) : scalar_type;

  // 这里存在的目的是防止我们追踪到 empty() 的调用。实际的 autograd 代码并不重要，
  // 因为 requires_grad 总是 false。
  // tensor_new() 的语义是什么？
  // 我们手动构造一个张量，并将其放置在正确的设备上，然后在某些情况下需要“提升”新构造的张量，
  // 比如在执行 functorch 变换或运行 functionalization 时。exclude guards 用于确保在构造原始张量时不运行额外的逻辑。
  Tensor tensor;
  {
    // 在以下范围内自动分派到 ADInplaceOrView 以下的分派器
    at::AutoDispatchBelowADInplaceOrView guard;
    // 用于排除 Python 分派模式的 guard
    c10::impl::ExcludeDispatchKeyGuard torchdispatchmode_guard(
        c10::DispatchKey::Python);
    // 用于排除 PythonTLS 快照的 guard
    c10::impl::ExcludeDispatchKeyGuard torchdispatchmode_snapshot_guard(
        c10::DispatchKey::PythonTLSSnapshot);
    // functorch 使用 FuncTorchDynamicLayerBackMode 作为模式键，用于将操作符返回的所有张量包装在特殊的 TensorWrapper 张量扩展中
    c10::impl::ExcludeDispatchKeyGuard functorch_front_guard(
        c10::DispatchKey::FuncTorchDynamicLayerFrontMode);
    // 类似地，用于排除 Fake 和 DeferredInit 处理程序
    c10::impl::ExcludeDispatchKeyGuard fake_and_deferred_init_guard(
        c10::DispatchKeySet{
            c10::DispatchKey::Fake, c10::DispatchKey::DeferredInit});
    // 注释 [Functionalization <> torch.Tensor constructor]
    // Functionalization 通过 aten::lift() 将新构造的张量“提升”为包装器
    c10::impl::ExcludeDispatchKeyGuard functionalize_guard(
        c10::DispatchKey::Functionalize);
    {
      // 追踪应该可能还要使用“lift”操作符将张量添加到跟踪中，但实际上这可能会破坏向后兼容性，因为我们当前在跟踪 .to() 调用。
      at::tracer::impl::NoTracerDispatchMode tracer_guard;
    
      if (isStorage(data)) {
        // 如果数据是存储器，则创建存储器并获取相关信息
        auto [storage, storage_scalar_type, is_typed_storage] =
            createStorageGetType(data);
    
        // 检查存储器是否是指定的标量类型或者未指定类型的存储器
        TORCH_CHECK(
            !is_typed_storage || storage_scalar_type == scalar_type,
            "Expected a Storage of type ",
            scalar_type,
            " or an UntypedStorage, but got ",
            storage_scalar_type);
        
        // 创建一个空张量，基于存储器的设备和类型
        tensor = at::empty(
            sizes,
            at::initialTensorOptions()
                .dtype(
                    is_typed_storage ? storage_scalar_type
                                     : inferred_scalar_type)
                .pinned_memory(pin_memory)
                .device(storage.device()));
        
        // 将存储器的内容拷贝到张量中
        tensor.set_(storage);
    
      } else {
        // 创建默认选项为推断的标量类型的张量
        TensorOptions opts =
            at::initialTensorOptions().dtype(inferred_scalar_type);
    
        // 如果设备是 Meta，则返回一个空张量，避免破坏 Meta 张量的契约
        if (device == at::kMeta) {
          return at::empty(sizes, opts.device(device));
        }
    
        // 创建一个空张量，可能会将数据递归存储到其中
        tensor = at::empty(sizes, opts.pinned_memory(pin_memory));
    
        // 如果张量的大小不为零，则递归存储数据
        if (c10::multiply_integers(tensor.sizes()) != 0) {
          recursive_store(
              (char*)tensor.data_ptr(),
              tensor.sizes(),
              tensor.strides(),
              0,
              inferred_scalar_type,
              tensor.dtype().itemsize(),
              data);
        }
      }
    }
    
    // 释放 GIL（全局解释器锁），允许 Python 线程并发执行
    pybind11::gil_scoped_release no_gil;
    
    // 可能初始化设备，确保设备的准备就绪
    maybe_initialize_device(device);
    
    // 但是，非常重要的是在这里跟踪 to() 调用（尽管原因是一个 hack）。
    // 如果在构造时没有跟踪到 *某个* 工厂函数调用，我们将认为张量常数来自于跟踪“外部”，
    // 如果直接尝试返回它，将会出现“没有可观察数据依赖”的错误。
    // 在理想情况下，我们不应该跟踪 to() 调用，但我需要更深入地考虑在这种情况下到底应该跟踪什么。
    if (only_lift_cpu_tensors()) {
      // 如果仅提升 CPU 张量，则将张量转换到推断的标量类型
      tensor = tensor.to(
          inferred_scalar_type, /*non_blocking=*/false, /*copy=*/false);
    
    } else {
      // 否则，将张量转换到指定设备和推断的标量类型
      tensor = tensor.to(
          device, inferred_scalar_type, /*non_blocking=*/false, /*copy=*/false);
    }
    // 设置自动分发以下自动微分不可变操作或视图的保护
    at::AutoDispatchBelowADInplaceOrView guard;
    // 对张量进行提升操作，返回新的张量
    tensor = at::lift_fresh(tensor);
  }
  // 如果仅提升 CPU 张量且设备类型不是 CPU
  if (only_lift_cpu_tensors() && device.type() != DeviceType::CPU) {
    // 如果设备没有索引且设备类型未初始化
    if (!device.has_index() &&
        !torch::utils::is_device_initialized(device.type())) {
      // 推断设备 0 以避免设备初始化
      device = c10::Device(device.type(), 0);
    }
    // 将张量移动到指定设备上，非异步操作，不进行复制
    tensor = tensor.to(device, /*non_blocking=*/false, /*copy=*/false);
  }
  // 返回处理后的张量
  return tensor;
// 定义一个函数，用于从给定数据创建新的张量，进行数据复制
Tensor new_from_data_copy(
    c10::TensorOptions options,  // 张量的选项
    at::ScalarType scalar_type,  // 张量的标量类型
    std::optional<Device> device,  // 设备的可选参数
    PyObject* data) {  // Python对象，作为张量数据输入
  // 调用内部函数，创建新张量，设置复制选项为true，复制numpy数组为true，类型推断为false
  return internal_new_from_data(
      options,
      scalar_type,
      device,
      data,
      /*copy_variables=*/true,  // 是否复制变量
      /*copy_numpy=*/true,  // 是否复制NumPy数组
      /*type_inference=*/false);  // 是否进行类型推断
}

// 用于从序列创建遗留张量
Tensor legacy_new_from_sequence(
    c10::TensorOptions options,  // 张量的选项
    at::ScalarType scalar_type,  // 张量的标量类型
    std::optional<Device> device,  // 设备的可选参数
    PyObject* data) {  // Python对象，作为张量数据输入
  // 检查数据是否是序列，否则抛出错误
  TORCH_CHECK_TYPE(
      PySequence_Check(data),
      "new(): data must be a sequence (got ",
      Py_TYPE(data)->tp_name,
      ")");
  // 调用内部函数，创建新张量，设置复制选项为false，复制numpy数组为false，类型推断为false
  return internal_new_from_data(
      options,
      scalar_type,
      device,
      data,
      /*copy_variables=*/false,  // 是否复制变量
      /*copy_numpy=*/false,  // 是否复制NumPy数组
      /*type_inference=*/false);  // 是否进行类型推断
}

// 检查遗留的new操作的基本类型和预期布局
void check_base_legacy_new(
    c10::DispatchKey dispatch_key,  // 分发键
    at::Layout expected_layout) {  // 预期的布局
  if (expected_layout == c10::kStrided) {
    // 如果预期的布局是strided，检查分发键是否在预定义的键集合中
    constexpr c10::DispatchKeySet expected_key_set({
        c10::DispatchKey::CPU,
        c10::DispatchKey::CUDA,
        c10::DispatchKey::HIP,
        c10::DispatchKey::XLA,
        c10::DispatchKey::Lazy,
        c10::DispatchKey::IPU,
        c10::DispatchKey::XPU,
        c10::DispatchKey::HPU,
        c10::DispatchKey::MPS,
        c10::DispatchKey::Meta,
        c10::DispatchKey::PrivateUse1,
    });
    // 检查分发键是否在预期键集合中，否则抛出错误
    TORCH_CHECK(
        expected_key_set.has(dispatch_key),
        "new(): expected key in ",
        expected_key_set,
        " but got: ",
        dispatch_key);
  } else if (expected_layout == c10::kSparse) {
    // 如果预期的布局是sparse，检查分发键是否在预定义的sparse键集合中
    // 注意：sparse布局不支持XLA和Lazy
    constexpr c10::DispatchKeySet expected_key_set({
        c10::DispatchKey::SparseCPU,
        c10::DispatchKey::SparseCUDA,
        c10::DispatchKey::SparseHIP,
        c10::DispatchKey::SparseXPU,
        c10::DispatchKey::SparsePrivateUse1,
    });
    // 检查分发键是否在预期sparse键集合中，否则抛出错误
    TORCH_CHECK(
        expected_key_set.has(dispatch_key),
        "new(): expected key in ",
        expected_key_set,
        " but got: ",
        dispatch_key);
  } else {
    // 如果布局既不是strided也不是sparse，抛出内部断言错误
    TORCH_INTERNAL_ASSERT(false, "unexpected layout");
  }
}

// TODO: 使用dispatchKeyToTensorOptions重写此函数
// 检查遗留构造函数的设备类型
void check_legacy_ctor_device(
    c10::DispatchKey dispatch_key,  // 分发键
    std::optional<Device> device) {  // 设备的可选参数
  if (device.has_value()) {
    // 如果设备值存在，检查分发键对应的设备类型是否匹配
    TORCH_CHECK(
        dispatchKeyToDeviceType(dispatch_key) == device.value().type(),
        "legacy constructor expects device type: ",
        dispatchKeyToDeviceType(dispatch_key),
        " but device type: ",
        device.value().type(),
        " was passed");
  }
}

// 构造器类型：基础构造器、构造器、新建
enum class CtorOrNew {
  BASE_CTOR,
  CTOR,
  NEW,
};

// 通用的遗留稀疏张量构造器和新建函数
Tensor legacy_sparse_tensor_generic_ctor_new(
    c10::DispatchKey dispatch_key,  // 分发键
    at::ScalarType scalar_type,  // 张量的标量类型
    PyObject* args,  // 参数
    PyObject* kwargs,  // 关键字参数
    // 根据不同的构造方法或者新建方法，创建稀疏张量
    auto options = dispatchKeyToTensorOptions(dispatch_key);
    // 定义静态的 Python 参数解析器，支持多种构造函数和新建函数
    static PythonArgParser parser({
        "new(*, Device? device=None)",
        "new(*, int64_t cdata)|hidden",
        "new(Tensor indices, Tensor values, *, Device? device=None)",
        "new(Tensor indices, Tensor values, IntArrayRef size, *, Device? device=None)",
        "new(SymIntArrayRef size, *, Device? device=None)",
    });
    // 如果是新建操作，检查是否符合基于 dispatch_key 的稀疏张量的旧API
    if (ctor_or_new == CtorOrNew::NEW)
      check_base_legacy_new(dispatch_key, c10::kSparse);
    // 解析输入的参数 args 和 kwargs
    ParsedArgs<4> parsed_args;
    auto r = parser.parse(args, kwargs, parsed_args);
    // 根据解析器的索引选择不同的构造方法
    if (r.idx == 0) {
      // 如果是构造函数并且选择了第一个签名，生成空的稀疏张量并返回
      if (ctor_or_new == CtorOrNew::CTOR) {
        TORCH_WARN_ONCE(
            "torch.sparse.SparseTensor() is deprecated."
            "  Please use torch.sparse_coo_tensor((0,), dtype=).");
      }
      auto deviceOptional = r.deviceOptional(0);
      // 检查构造函数的设备参数是否符合旧的API要求
      check_legacy_ctor_device(dispatch_key, deviceOptional);
      return at::empty({0}, build_options(options, scalar_type, deviceOptional));
    } else if (r.idx == 1) {
      // 如果是构造函数并且选择了第二个签名，从给定的 cdata 创建不安全的张量并返回
      if (ctor_or_new == CtorOrNew::CTOR) {
        TORCH_WARN_ONCE(
            "torch.sparse.SparseTensor(cdata=x._cdata) is deprecated."
            "  Please use torch.sparse_coo_tensor(x._indices(), x._values(), x.shape).");
      }
      // 从整数 cdata 转换为指针，并创建不安全的张量返回
      // NOLINTNEXTLINE(performance-no-int-to-ptr)
      auto cdata = reinterpret_cast<void*>(r.toInt64(0));
      return at::unsafeTensorFromTH(cdata, true);
    } else if (r.idx == 2) {
      // 如果是构造函数并且选择了第三个签名，使用给定的 indices 和 values 创建稀疏 COO 张量并返回
      if (ctor_or_new == CtorOrNew::CTOR) {
        TORCH_WARN_ONCE(
            "torch.sparse.SparseTensor(indices, values, *, device=) is deprecated."
            "  Please use torch.sparse_coo_tensor(indices, values, dtype=, device=).");
      }
      // 注意：这个签名没有 dtype，尽管有 device；可能不应该有 device（我们应该推断它）
      auto deviceOptional = r.deviceOptional(2);
      // 检查构造函数的设备参数是否符合旧的API要求
      check_legacy_ctor_device(dispatch_key, deviceOptional);
      // 创建稀疏 COO 张量，并使用设备可选对象保护设备上下文
      at::OptionalDeviceGuard device_guard(deviceOptional);
      return at::sparse_coo_tensor(r.tensor(0), r.tensor(1));
    } else if (r.idx == 3) {
      // 如果是构造函数并且选择了第四个签名，使用给定的 indices、values 和 size 创建稀疏 COO 张量并返回
      if (ctor_or_new == CtorOrNew::CTOR) {
        TORCH_WARN_ONCE(
            "torch.sparse.SparseTensor(indices, values, shape, *, device=) is deprecated."
            "  Please use torch.sparse_coo_tensor(indices, values, shape, dtype=, device=).");
      }
      // 注意：这个签名没有 dtype，尽管有 device；可能不应该有 device（我们应该推断它）
      auto deviceOptional = r.deviceOptional(3);
      // 检查构造函数的设备参数是否符合旧的API要求
      check_legacy_ctor_device(dispatch_key, deviceOptional);
      // 创建稀疏 COO 张量，并使用设备可选对象保护设备上下文
      at::OptionalDeviceGuard device_guard(deviceOptional);
      return at::sparse_coo_tensor(r.tensor(0), r.tensor(1), r.intlist(2));
    } else if (r.idx == 4) {
      // 如果是构造函数并且选择了第五个签名，从 Python 对象 arg 创建稀疏 COO 张量并返回
      PyObject* arg = r.pyobject(0);
      auto deviceOptional = r.deviceOptional(1);
      // 检查构造函数的设备参数是否符合旧的API要求
      check_legacy_ctor_device(dispatch_key, deviceOptional);
    // 检查是否 arg 不是 torch.Size 类型并且参数个数大于等于 1，并且 arg 与参数元组中第一个参数相同
    if (!THPSize_Check(arg) && PyTuple_GET_SIZE(args) >= 1 &&
        arg == PyTuple_GET_ITEM(args, 0)) {
      // 如果是通过 new(sequence) 调用，并且 sequence 不是 torch.Size 类型，抛出类型错误
      if (ctor_or_new == CtorOrNew::CTOR) {
        throw TypeError(
            "torch.sparse.SparseTensor(sequence) only accepts sizes.  Please use torch.sparse_coo_tensor() "
            "or construct a strided tensor and convert it to sparse via to_sparse.");
      } else {
        throw TypeError(
            "SparseTensor.new(sequence) only accepts sizes.  Please use torch.sparse_coo_tensor() "
            "or construct a strided tensor and convert it to sparse via to_sparse.");
      }
    }
    // 如果是通过构造函数调用（ctor_or_new == CtorOrNew::CTOR），发出一次性警告
    if (ctor_or_new == CtorOrNew::CTOR) {
      TORCH_WARN_ONCE(
          "torch.sparse.SparseTensor(shape, *, device=) is deprecated."
          "  Please use torch.sparse_coo_tensor(shape, dtype=, device=).");
    }
    // 调用 new_with_sizes 函数，传递给它 options、scalar_type、r.deviceOptional(1) 和 r.symintlist(0) 参数
    return new_with_sizes(
        options, scalar_type, r.deviceOptional(1), r.symintlist(0));
  }
  // 如果以上条件都不满足，则抛出运行时错误，表示 new() 函数的参数无效
  throw std::runtime_error("new(): invalid arguments");
} // 结束匿名命名空间

// 注意：这里的 device_idx 不是 DeviceIndex，而是 PythonArgs 的索引
c10::TensorOptions typeIdWithDefault(
    PythonArgs& r, // Python 参数解析器对象的引用
    int64_t device_idx, // 设备索引，用于选择设备
    c10::DispatchKey dispatch_key) { // 分发键，用于确定张量的后端和布局

  auto options = dispatchKeyToTensorOptions(dispatch_key); // 根据分发键获取张量选项

  if (!r.isNone(static_cast<int>(device_idx))) {
    // 如果设备索引对应的参数不为空
    // TODO: 这行代码在测试中似乎没有被执行到
    options = options.device(r.device(static_cast<int>(device_idx)).type()); // 根据参数设置张量选项的设备类型
  }

  return options; // 返回张量选项
}

} // namespace

// 创建遗留张量的通用构造函数
Tensor legacy_tensor_generic_ctor_new(
    c10::DispatchKey dispatch_key, // 分发键，用于选择张量的后端和布局
    at::ScalarType scalar_type, // 张量的标量类型
    PyObject* args, // Python 参数对象
    PyObject* kwargs, // Python 关键字参数对象
    CtorOrNew ctor_or_new) { // 构造函数或者新建函数的类型

  auto options = dispatchKeyToTensorOptions(dispatch_key); // 根据分发键获取张量选项

  static PythonArgParser parser({
      "new(*, Device? device=None)", // 构造函数签名：new(*, Device? device=None)
      "new(Storage storage)", // 构造函数签名：new(Storage storage)
      "new(*, int64_t cdata)|hidden", // 隐藏的构造函数签名
      // 以下构造函数不再是遗留构造函数，也可用于子类初始化
      "new(Tensor other)", // 构造函数签名：new(Tensor other)
      "new(Tensor other, *, Device? device=None)|hidden", // 隐藏的构造函数签名
      // 防止 Tensor 与 IntArrayRef、PyObject* 匹配
      "new(SymIntArrayRef size, *, Device? device=None)", // 构造函数签名：new(SymIntArrayRef size, *, Device? device=None)
      "new(PyObject* data, *, Device? device=None)", // 构造函数签名：new(PyObject* data, *, Device? device=None)
  });

  if (isSparse(dispatchKeyToBackend(dispatch_key))) {
    return legacy_sparse_tensor_generic_ctor_new(
        dispatch_key, scalar_type, args, kwargs, ctor_or_new); // 如果是稀疏张量，则调用稀疏张量的通用构造函数
  }

  if (ctor_or_new == CtorOrNew::NEW)
    check_base_legacy_new(dispatch_key, c10::kStrided); // 检查基础遗留构造函数是否符合要求

  ParsedArgs<2> parsed_args; // 解析的参数对象，最多包含两个参数
  auto r = parser.parse(args, kwargs, parsed_args); // 解析传入的 Python 参数和关键字参数
  if (r.idx == 0) {
    auto deviceOptional = r.deviceOptional(0); // 获取设备可选参数
    check_legacy_ctor_device(dispatch_key, deviceOptional); // 检查遗留构造函数的设备类型
    at::OptionalDeviceGuard device_guard(deviceOptional); // 设备保护器，根据设备可选参数创建
    return at::empty({0}, build_options(options, scalar_type)); // 返回一个空张量
  } else if (r.idx == 1) {
    at::ScalarType storage_scalar_type{at::ScalarType::Undefined}; // 存储的标量类型，默认未定义
    bool is_typed_storage = false; // 是否是类型化存储，默认为假
    at::Storage storage = r.storage(0, storage_scalar_type, is_typed_storage); // 获取存储对象
    if (storage_scalar_type != at::ScalarType::Undefined && is_typed_storage) {
      TORCH_CHECK(
          storage_scalar_type == scalar_type,
          "Expected a Storage of type ",
          scalar_type,
          " or an UntypedStorage, but got type ",
          storage_scalar_type,
          " for argument 1 'storage'");
    }
    return new_with_storage(options, scalar_type, storage); // 根据存储对象创建张量
  } else if (r.idx == 2) {
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    auto cdata = reinterpret_cast<void*>(r.toInt64(0)); // 将整数转换为指针类型
    return at::unsafeTensorFromTH(cdata, true); // 从 TH（TorchScript）创建不安全的张量
  } else if (r.idx == 3) {
    const auto& other = r.tensor(0); // 获取传入的张量参数
    // BASE_CTOR（即 torch.Tensor）现在放宽为接受任何 dtype；以前是“float”偏好的
    // 如果不是基础构造函数，设置选项为特定数据类型
    if (ctor_or_new != CtorOrNew::BASE_CTOR) {
      options = options.dtype(scalar_type);
      // 检查其他张量的选项是否与当前选项相同
      TORCH_CHECK_TYPE(
          other.options().type_equal(options),
          "expected ",
          options,
          " (got ",
          other.options(),
          ")");
    }
    // 返回其他张量的别名
    return other.alias();
  } else if (r.idx == 4) {
    // 如果索引为4，处理特定的构造函数或基础构造函数错误
    if (ctor_or_new == CtorOrNew::CTOR || ctor_or_new == CtorOrNew::BASE_CTOR) {
      // 抛出错误信息，提示不支持旧版本的张量构造方式
      TORCH_CHECK(
          false,
          "Legacy tensor constructor of the form torch.Tensor(tensor, device=device) "
          "is not supported.  Use torch.tensor(...) or torch.as_tensor(...) instead.");
    } else {
      // 抛出错误信息，提示不支持旧版本的张量新建方式
      TORCH_CHECK(
          false,
          "Legacy tensor new of the form tensor.new(tensor, device=device) "
          "is not supported.  Use torch.as_tensor(...) instead.");
    }
  } else if (r.idx == 5) {
    // 如果索引为5，处理特定的构造函数情况
    PyObject* arg = r.pyobject(0);
    auto deviceOptional = r.deviceOptional(1);
    // 检查使用旧版本构造函数的设备选项是否有效
    check_legacy_ctor_device(dispatch_key, deviceOptional);
    // 如果参数不是 torch.Size 且参数数量大于等于1且与第一个参数相同
    if (!THPSize_Check(arg) && PyTuple_GET_SIZE(args) >= 1 &&
        arg == PyTuple_GET_ITEM(args, 0)) {
      // 如果是以 sequence 形式传入的新建操作，但不是 torch.Size 类型的序列，特殊处理
      return legacy_new_from_sequence(
          options, scalar_type, deviceOptional, r.pyobject(0));
    }
    // 使用给定大小创建新张量
    return new_with_sizes(
        options, scalar_type, r.deviceOptional(1), r.symintlist(0));
  } else if (r.idx == 6) {
    // 如果索引为6，处理特定的构造函数情况
    auto deviceOptional = r.deviceOptional(1);
    // 检查使用旧版本构造函数的设备选项是否有效
    check_legacy_ctor_device(dispatch_key, deviceOptional);
    // 使用序列创建新张量
    return legacy_new_from_sequence(
        options, scalar_type, deviceOptional, r.pyobject(0));
  }
  // 抛出无效参数错误，表示构造函数不支持给定的参数
  throw std::runtime_error("new(): invalid arguments");
}

// 仅处理 torch.Tensor 类型的张量
// 与传统的 dtype/device 专用构造函数不同，此函数接受任何设备/dtype输入张量（即使它与默认值不匹配）
Tensor base_tensor_ctor(PyObject* args, PyObject* kwargs) {
  return legacy_tensor_generic_ctor_new(
      torch::tensors::get_default_dispatch_key(),
      torch::tensors::get_default_scalar_type(),
      args,
      kwargs,
      CtorOrNew::BASE_CTOR);
}

// 处理类似 torch.DoubleTensor、torch.cuda.FloatTensor、torch.sparse.FloatTensor 等的调用
Tensor legacy_tensor_ctor(
    c10::DispatchKey dispatch_key,
    at::ScalarType scalar_type,
    PyObject* args,
    PyObject* kwargs) {
  return legacy_tensor_generic_ctor_new(
      dispatch_key, scalar_type, args, kwargs, CtorOrNew::CTOR);
}

// 处理 tensor.new(...) 的情况
Tensor legacy_tensor_new(
    c10::DispatchKey dispatch_key,
    at::ScalarType scalar_type,
    PyObject* args,
    PyObject* kwargs) {
  return legacy_tensor_generic_ctor_new(
      dispatch_key, scalar_type, args, kwargs, CtorOrNew::NEW);
}

Tensor indexing_tensor_from_data(
    c10::TensorOptions options,
    at::ScalarType scalar_type,
    std::optional<Device> device,
    PyObject* data) {
  // 专门用于张量索引，将索引列表转换为索引张量（类型为 Byte 或 Long）
  ScalarType inferred_scalar_type = infer_scalar_type(data);
  if (inferred_scalar_type == ScalarType::Byte ||
      inferred_scalar_type == ScalarType::Bool) {
    return internal_new_from_data(
        options,
        inferred_scalar_type,
        device,
        data,
        /*copy_variables=*/false,
        /*copy_numpy=*/false,
        /*type_inference=*/false);
  } else {
    return internal_new_from_data(
        options,
        scalar_type,
        device,
        data,
        /*copy_variables=*/false,
        /*copy_numpy=*/false,
        /*type_inference=*/false);
  }
}

class CheckSparseTensorInvariantsContext {
 public:
  CheckSparseTensorInvariantsContext()
      : state{at::globalContext().checkSparseTensorInvariants()} {}
  ~CheckSparseTensorInvariantsContext() {
    at::globalContext().setCheckSparseTensorInvariants(state);
  }

 private:
  bool state;
};

static Tensor sparse_compressed_tensor_ctor_worker(
    const std::string& name,
    c10::DispatchKey dispatch_key,
    at::ScalarType scalar_type,
    PythonArgs& r,
    std::optional<c10::Layout> required_layout) {
  TORCH_INTERNAL_ASSERT(!isSparseCsr(dispatchKeyToBackend(dispatch_key)));
  TORCH_INTERNAL_ASSERT(!isSparse(dispatchKeyToBackend(dispatch_key)));
  enum {
    ARG_COMPRESSED_INDICES = 0,
    ARG_PLAIN_INDICES,
    ARG_VALUES,
    ARG_SIZE,
    ARG_TYPE,
    ARG_LAYOUT,
    ARG_DEVICE,
    ARG_PIN_MEMORY,
    ARG_REQUIRES_GRAD,
    ARG_CHECK_INVARIANTS,
    ARGS_COUNT
  };
  enum {
    ARG_VALUES1 = ARG_VALUES,
    ARG_TYPE1,
    ARG_LAYOUT1,
    ARG_DEVICE1,
    ARG_PIN_MEMORY1,
    ARG_REQUIRES_GRAD1,
    ARG_CHECK_INVARIANTS1,
  ARGS_COUNT1
};

// Lambda function to safely get a string attribute from a Python object
auto safe_get_attr_string = [](PyObject* o,
                               const char* attr_name) -> PyObject* {
  // Clear error indicator if attribute does not exist.
  // Otherwise subsequent Python C API calls might return bogus values.
  // See https://github.com/pytorch/pytorch/issues/58520 for more details
  auto rc = PyObject_GetAttrString(o, attr_name);
  if (!rc) {
    // If attribute error occurs, check if it's not an AttributeError exception
    if (!PyErr_ExceptionMatches(PyExc_AttributeError)) {
      throw python_error();
    }
    // Clear the error indicator for AttributeError to suppress the error
    PyErr_Clear();
  }
  return rc;
};

// Retrieve the dtype attribute for compressed indices from Python object
THPObjectPtr compressed_indices_dtype_attr(
    safe_get_attr_string(r.pyobject(ARG_COMPRESSED_INDICES), "dtype"));
// Retrieve the dtype attribute for plain indices from Python object
THPObjectPtr plain_indices_dtype_attr(
    safe_get_attr_string(r.pyobject(ARG_PLAIN_INDICES), "dtype"));

// Determine scalar types from dtype attributes or default to kInt
at::ScalarType compressed_indices_scalar_type = compressed_indices_dtype_attr
    ? reinterpret_cast<THPDtype*>(compressed_indices_dtype_attr.get())
          ->scalar_type
    : kInt;
at::ScalarType plain_indices_scalar_type = plain_indices_dtype_attr
    ? reinterpret_cast<THPDtype*>(plain_indices_dtype_attr.get())->scalar_type
    : kInt;

// Context to restore the global state of sparse tensor invariants check flag
CheckSparseTensorInvariantsContext restores_check_sparse_tensor_invariants_global_state{};

// Determine whether to check sparse tensor invariants globally
bool default_check_invariants =
    at::globalContext().checkSparseTensorInvariants();

// If processing the first index
if (r.idx == 0) {
  // Determine if type inference is required
  bool type_inference = r.isNone(ARG_TYPE);
  // Infer options for the tensor creation
  const auto inferred_options =
      typeIdWithDefault(r, ARG_DEVICE, dispatch_key);
  // Infer scalar type for the tensor creation
  const auto inferred_scalar_type =
      r.scalartypeWithDefault(ARG_TYPE, scalar_type);
  // Guard the optional device for tensor operations
  at::OptionalDeviceGuard device_guard(r.deviceOptional(ARG_DEVICE));

  // Set the global state of invariants check flag based on argument
  at::globalContext().setCheckSparseTensorInvariants(
      r.toBoolWithDefault(ARG_CHECK_INVARIANTS, default_check_invariants));

  // Create tensor for values based on provided data and options
  Tensor values = internal_new_from_data(
      inferred_options,
      inferred_scalar_type,
      r.deviceOptional(ARG_DEVICE),
      r.pyobject(ARG_VALUES),
      /*copy_variables=*/false,
      /*copy_numpy=*/true,
      /*type_inference=*/type_inference);

  // Create tensor for compressed indices based on provided data and options
  Tensor compressed_indices = internal_new_from_data(
      values.options(),
      compressed_indices_scalar_type,
      r.deviceOptional(ARG_DEVICE),
      r.pyobject(ARG_COMPRESSED_INDICES),
      /*copy_variables=*/false,
      /*copy_numpy=*/true,
      /*type_inference=*/true);

  // Create tensor for plain indices based on provided data and options
  Tensor plain_indices = internal_new_from_data(
      values.options(),
      plain_indices_scalar_type,
      r.deviceOptional(ARG_DEVICE),
      r.pyobject(ARG_PLAIN_INDICES),
      /*copy_variables=*/false,
      /*copy_numpy=*/true,
      /*type_inference=*/true);
    std::optional<c10::Layout> layout =
        // 如果 required_layout 存在，则使用指定的布局参数来获取布局；否则使用可选的布局参数获取布局
        (required_layout
             ? r.layoutWithDefault(ARG_LAYOUT, required_layout.value())
             : r.layoutOptional(ARG_LAYOUT));
    if (required_layout && layout) {
      // 检查布局是否符合要求，如果不符合则抛出错误
      TORCH_CHECK(
          layout.value() == required_layout.value(),
          name,
          ": layout must be ",
          required_layout.value(),
          " but got ",
          layout.value());
    }
    // 创建稀疏压缩张量对象，参数包括压缩索引、普通索引、数值、大小、布局和是否需要梯度
    return at::sparse_compressed_tensor(
               compressed_indices,
               plain_indices,
               values,
               r.intlist(ARG_SIZE),
               values.options().layout(layout))
        .set_requires_grad(r.toBool(ARG_REQUIRES_GRAD));
  } else if (r.idx == 1) {
    bool type_inference = r.isNone(ARG_TYPE1);
    // 推断设备类型并设置默认的类型 ID
    const auto inferred_options =
        typeIdWithDefault(r, ARG_DEVICE1, dispatch_key);
    // 推断标量类型并设置默认的标量类型
    const auto inferred_scalar_type =
        r.scalartypeWithDefault(ARG_TYPE1, scalar_type);
    // 设置设备守卫，用于管理可选的设备
    at::OptionalDeviceGuard device_guard(r.deviceOptional(ARG_DEVICE1));
    // 设置全局稀疏张量不变性检查的标志
    at::globalContext().setCheckSparseTensorInvariants(
        r.toBoolWithDefault(ARG_CHECK_INVARIANTS1, default_check_invariants));

    // 从给定数据创建张量对象，包括选项、标量类型、设备、数据、是否复制变量和是否复制 NumPy 数组的标志
    Tensor values = internal_new_from_data(
        inferred_options,
        inferred_scalar_type,
        r.deviceOptional(ARG_DEVICE1),
        r.pyobject(ARG_VALUES),
        /*copy_variables=*/false,
        /*copy_numpy=*/true,
        /*type_inference=*/type_inference);
    // 从给定数据创建压缩索引张量对象
    Tensor compressed_indices = internal_new_from_data(
        values.options(),
        compressed_indices_scalar_type,
        r.deviceOptional(ARG_DEVICE1),
        r.pyobject(ARG_COMPRESSED_INDICES),
        /*copy_variables=*/false,
        /*copy_numpy=*/true,
        /*type_inference=*/true);
    // 从给定数据创建普通索引张量对象
    Tensor plain_indices = internal_new_from_data(
        values.options(),
        plain_indices_scalar_type,
        r.deviceOptional(ARG_DEVICE1),
        r.pyobject(ARG_PLAIN_INDICES),
        /*copy_variables=*/false,
        /*copy_numpy=*/true,
        /*type_inference=*/true);
    std::optional<c10::Layout> layout =
        // 如果 required_layout 存在，则使用指定的布局参数来获取布局；否则使用可选的布局参数获取布局
        (required_layout
             ? r.layoutWithDefault(ARG_LAYOUT1, required_layout.value())
             : r.layoutOptional(ARG_LAYOUT1));
    if (required_layout && layout) {
      // 检查布局是否符合要求，如果不符合则抛出错误
      TORCH_CHECK(
          layout.value() == required_layout.value(),
          name,
          ": layout must be ",
          required_layout.value(),
          " but got ",
          layout.value());
    }
    // 创建稀疏压缩张量对象，参数包括压缩索引、普通索引、数值、布局和是否需要梯度
    return at::sparse_compressed_tensor(
               compressed_indices,
               plain_indices,
               values,
               values.options().layout(layout))
        .set_requires_grad(r.toBool(ARG_REQUIRES_GRAD1));
  }
  // 如果以上条件均不满足，则抛出运行时错误，表示参数无效
  throw std::runtime_error(name + ": invalid arguments");
// 结束 sparse_compressed_tensor_ctor 函数定义
}

// 定义 sparse_compressed_tensor_ctor 函数，用于创建稀疏压缩张量
Tensor sparse_compressed_tensor_ctor(
    c10::DispatchKey dispatch_key,     // 分发键，指定操作的分发方式
    at::ScalarType scalar_type,        // 标量类型，张量中元素的数据类型
    PythonArgs& r) {                   // Python 参数对象的引用
  std::optional<c10::Layout> required_layout{};  // 可选的张量布局，默认为空
  // 调用辅助函数进行实际的稀疏压缩张量构造，返回结果张量
  return sparse_compressed_tensor_ctor_worker(
      "sparse_compressed_tensor",     // 张量类型名称字符串
      dispatch_key,                   // 分发键
      scalar_type,                    // 标量类型
      r,                              // Python 参数对象引用
      required_layout);               // 可选的布局对象
}

// 定义 sparse_csr_tensor_ctor 函数，用于创建稀疏 CSR 格式张量
Tensor sparse_csr_tensor_ctor(
    c10::DispatchKey dispatch_key,     // 分发键，指定操作的分发方式
    at::ScalarType scalar_type,        // 标量类型，张量中元素的数据类型
    PythonArgs& r) {                   // Python 参数对象的引用
  std::optional<c10::Layout> required_layout(c10::Layout::SparseCsr);  // 指定稀疏 CSR 布局
  // 调用稀疏压缩张量构造函数进行实际的构造，返回结果张量
  return sparse_compressed_tensor_ctor_worker(
      "sparse_csr_tensor",            // 张量类型名称字符串
      dispatch_key,                   // 分发键
      scalar_type,                    // 标量类型
      r,                              // Python 参数对象引用
      required_layout);               // 指定的布局对象
}

// 定义 sparse_csc_tensor_ctor 函数，用于创建稀疏 CSC 格式张量
Tensor sparse_csc_tensor_ctor(
    c10::DispatchKey dispatch_key,     // 分发键，指定操作的分发方式
    at::ScalarType scalar_type,        // 标量类型，张量中元素的数据类型
    PythonArgs& r) {                   // Python 参数对象的引用
  std::optional<c10::Layout> required_layout(c10::Layout::SparseCsc);  // 指定稀疏 CSC 布局
  // 调用稀疏压缩张量构造函数进行实际的构造，返回结果张量
  return sparse_compressed_tensor_ctor_worker(
      "sparse_csc_tensor",            // 张量类型名称字符串
      dispatch_key,                   // 分发键
      scalar_type,                    // 标量类型
      r,                              // Python 参数对象引用
      required_layout);               // 指定的布局对象
}

// 定义 sparse_bsr_tensor_ctor 函数，用于创建稀疏 BSR 格式张量
Tensor sparse_bsr_tensor_ctor(
    c10::DispatchKey dispatch_key,     // 分发键，指定操作的分发方式
    at::ScalarType scalar_type,        // 标量类型，张量中元素的数据类型
    PythonArgs& r) {                   // Python 参数对象的引用
  std::optional<c10::Layout> required_layout(c10::Layout::SparseBsr);  // 指定稀疏 BSR 布局
  // 调用稀疏压缩张量构造函数进行实际的构造，返回结果张量
  return sparse_compressed_tensor_ctor_worker(
      "sparse_bsr_tensor",            // 张量类型名称字符串
      dispatch_key,                   // 分发键
      scalar_type,                    // 标量类型
      r,                              // Python 参数对象引用
      required_layout);               // 指定的布局对象
}

// 定义 sparse_bsc_tensor_ctor 函数，用于创建稀疏 BSC 格式张量
Tensor sparse_bsc_tensor_ctor(
    c10::DispatchKey dispatch_key,     // 分发键，指定操作的分发方式
    at::ScalarType scalar_type,        // 标量类型，张量中元素的数据类型
    PythonArgs& r) {                   // Python 参数对象的引用
  std::optional<c10::Layout> required_layout(c10::Layout::SparseBsc);  // 指定稀疏 BSC 布局
  // 调用稀疏压缩张量构造函数进行实际的构造，返回结果张量
  return sparse_compressed_tensor_ctor_worker(
      "sparse_bsc_tensor",            // 张量类型名称字符串
      dispatch_key,                   // 分发键
      scalar_type,                    // 标量类型
      r,                              // Python 参数对象引用
      required_layout);               // 指定的布局对象
}

// 注释段落开始
// 注意 [Ensuring sparse values and indices match devices]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// 在所有构造索引的地方，我们从值中读取选项（而不是使用推断的选项）。为什么？这处理了这样一种情况：
// 当值是 CUDA 张量时，但索引是非张量值（并且未设置设备参数）。例如：
//
//  torch.sparse_coo_tensor(([0, 1],), self.empty(2, 0).cuda(), (4, 0))
//
// 稀疏张量要求索引和值位于同一设备上。如果值位于 CUDA 上，我们可以推断索引应该位于的位置，
// 甚至应接受普通索引序列（只需确保将其写入正确的设备即可）。值是我们知道索引张量应该到达 CUDA 的唯一方法，
// 因此我们必须以某种方式获取信息。
//
// 这段代码有点粗糙。首先，选项中的 dtype 被内部的 internal_new_from_data 静默地忽略了。
// 此外，在经典的粗糙代码样式中，它过去不太正确：如果值位于 "cuda:1"，我们过去只是说“这应该是 CUDA”，
// 然后索引会分配到错误的张量上。选项更正确，可以正确处理这一点。
  PythonArgs& r) {
  // 断言稠密张量
  TORCH_INTERNAL_ASSERT(!isSparse(dispatchKeyToBackend(dispatch_key)));
  // 断言非 CSR 格式稀疏张量
  TORCH_INTERNAL_ASSERT(!isSparseCsr(dispatchKeyToBackend(dispatch_key)));

  // 定义第一个枚举
  enum {
    ARG_INDICES = 0,        // 索引参数位置
    ARG_VALUES,             // 值参数位置
    ARG_TYPE,               // 类型参数位置
    ARG_DEVICE,             // 设备参数位置
    ARG_REQUIRES_GRAD,      // 是否需要梯度参数位置
    ARG_CHECK_INVARIANTS,   // 检查不变性参数位置
    ARGS_COUNT              // 参数个数
  };

  // 定义第二个枚举
  enum {
    ARG_INDICES1 = 0,       // 索引参数位置
    ARG_VALUES1,            // 值参数位置
    ARG_SIZE1,              // 尺寸参数位置
    ARG_TYPE1,              // 类型参数位置
    ARG_DEVICE1,            // 设备参数位置
    ARG_REQUIRES_GRAD1,     // 是否需要梯度参数位置
    ARG_CHECK_INVARIANTS1,  // 检查不变性参数位置
    ARG_IS_COALESCED1,      // 是否紧缩参数位置
    ARGS_COUNT1             // 参数个数
  };

  // 定义第三个枚举
  enum {
    ARG_SIZE2 = 0,          // 尺寸参数位置
    ARG_TYPE2,              // 类型参数位置
    ARG_DEVICE2,            // 设备参数位置
    ARG_REQUIRES_GRAD2,     // 是否需要梯度参数位置
    ARG_CHECK_INVARIANTS2,  // 检查不变性参数位置
    ARGS_COUNT2             // 参数个数
  };

  // 恢复检查稀疏张量不变性上下文
  CheckSparseTensorInvariantsContext restores_check_sparse_tensor_invariants_global_state{};
  // 获取全局上下文中的稀疏张量不变性检查设置
  bool default_check_invariants = at::globalContext().checkSparseTensorInvariants();

  if (r.idx == 0) {
    // 是否进行类型推断
    bool type_inference = r.isNone(ARG_TYPE);
    // 推断出的选项
    const auto inferred_options = typeIdWithDefault(r, ARG_DEVICE, dispatch_key);
    // 推断出的标量类型
    const auto inferred_scalar_type = r.scalartypeWithDefault(ARG_TYPE, scalar_type);
    // 可选设备保护
    at::OptionalDeviceGuard device_guard(r.deviceOptional(ARG_DEVICE));
    // 设置全局上下文中的稀疏张量不变性检查
    at::globalContext().setCheckSparseTensorInvariants(
        r.toBoolWithDefault(ARG_CHECK_INVARIANTS, default_check_invariants));

    // 如果未提供 dtype，则根据值类型推断类型
    Tensor values = internal_new_from_data(
        inferred_options,
        inferred_scalar_type,
        r.deviceOptional(ARG_DEVICE),
        r.pyobject(ARG_VALUES),
        /*copy_variables=*/false,
        /*copy_numpy=*/true,
        /*type_inference=*/type_inference);
    // 见注释 [确保稀疏值和索引匹配设备]
    // 创建新的张量用于索引
    Tensor indices = internal_new_from_data(
        values.options(),
        kLong,
        r.deviceOptional(ARG_DEVICE),
        r.pyobject(ARG_INDICES),
        /*copy_variables=*/false,
        /*copy_numpy=*/true,
        /*type_inference=*/false);
    // 返回 COO 格式的稀疏张量，设置是否需要梯度属性
    return at::sparse_coo_tensor(
               indices, values, values.options().layout(at::kSparse))
        .set_requires_grad(r.toBool(ARG_REQUIRES_GRAD));
  } else if (r.idx == 1) {
    // 是否进行类型推断
    bool type_inference = r.isNone(ARG_TYPE1);
    // 推断出的选项
    const auto inferred_options = typeIdWithDefault(r, ARG_DEVICE1, dispatch_key);
    // 推断出的标量类型
    const auto inferred_scalar_type = r.scalartypeWithDefault(ARG_TYPE1, scalar_type);
    // 可选设备保护
    at::OptionalDeviceGuard device_guard(r.deviceOptional(ARG_DEVICE1));
    // 设置全局上下文中的稀疏张量不变性检查
    at::globalContext().setCheckSparseTensorInvariants(
        r.toBoolWithDefault(ARG_CHECK_INVARIANTS1, default_check_invariants));

    // 创建新的张量用于值
    Tensor values = internal_new_from_data(
        inferred_options,
        inferred_scalar_type,
        r.deviceOptional(ARG_DEVICE1),
        r.pyobject(ARG_VALUES1),
        /*copy_variables=*/false,
        /*copy_numpy=*/true,
        /*type_inference=*/type_inference);
    // 见注释 [确保稀疏值和索引匹配设备]
    // 创建新的张量用于索引
    // 创建稀疏张量的索引张量，从数据值张量的选项中获取类型信息
    Tensor indices = internal_new_from_data(
        values.options(),
        kLong,
        r.deviceOptional(ARG_DEVICE1),
        r.pyobject(ARG_INDICES1),
        /*copy_variables=*/false,
        /*copy_numpy=*/true,
        /*type_inference=*/false);
    
    // 返回一个新的 COO 格式的稀疏张量，使用提供的索引、值和大小参数
    return at::sparse_coo_tensor(
               indices,                                    // 使用上面创建的索引张量
               values,                                     // 提供的值张量
               r.intlist(ARG_SIZE1),                       // 提供的大小列表
               values.options().layout(at::kSparse),       // 使用稀疏布局
               r.toBoolOptional(ARG_IS_COALESCED1))       // 是否需要对张量进行合并
        .set_requires_grad(r.toBool(ARG_REQUIRES_GRAD1)); // 设置是否需要梯度跟踪
  } else if (r.idx == 2) {
    // 推断稀疏张量的选项
    const auto inferred_options =
        typeIdWithDefault(r, ARG_DEVICE2, dispatch_key);
    // 推断稀疏张量的标量类型
    const auto inferred_scalar_type =
        r.scalartypeWithDefault(ARG_TYPE2, scalar_type);
    // 可选的设备保护范围
    at::OptionalDeviceGuard device_guard(r.deviceOptional(ARG_DEVICE2));
    // 设置全局上下文中的稀疏张量不变量检查
    at::globalContext().setCheckSparseTensorInvariants(
        r.toBoolWithDefault(ARG_CHECK_INVARIANTS2, default_check_invariants));

    // 返回一个新的 COO 格式的稀疏张量，使用推断的选项和布局
    return at::sparse_coo_tensor(
               r.intlist(ARG_SIZE2),                               // 提供的大小列表
               inferred_options.dtype(inferred_scalar_type).layout(at::kSparse)) // 使用推断的选项和布局
        .set_requires_grad(r.toBool(ARG_REQUIRES_GRAD2));          // 设置是否需要梯度跟踪
  }
  // 若未匹配任何情况，则抛出运行时错误
  throw std::runtime_error("sparse_coo_tensor(): invalid arguments");
// 关闭上一个函数定义的花括号，结束函数定义
}

// 验证稀疏 COO 张量参数的有效性
void _validate_sparse_coo_tensor_args(
    c10::DispatchKey dispatch_key,  // 分发键，指定张量的分发方式
    at::ScalarType scalar_type,     // 标量类型，指定张量元素的数据类型
    PyObject* args,                 // Python 元组，包含函数参数
    PyObject* kwargs) {             // Python 字典，包含函数关键字参数
  auto options = dispatchKeyToTensorOptions(dispatch_key);  // 根据分发键获取张量选项
  static PythonArgParser parser({
      "_validate_sparse_coo_tensor(PyObject* indices, PyObject* values, IntArrayRef size)",
  });  // 静态 Python 参数解析器，解析函数参数和关键字参数

  ParsedArgs<3> parsed_args;  // 解析后的参数对象，预期包含三个参数
  auto r = parser.parse(args, kwargs, parsed_args);  // 解析函数调用的参数
  // 根据数据创建新的张量对象，用于存储稀疏 COO 张量的值
  Tensor values = internal_new_from_data(
      options,
      scalar_type,
      c10::nullopt,
      r.pyobject(1),
      /*copy_variables=*/false,
      /*copy_numpy=*/true,
      /*type_inference=*/true);
  // 创建稀疏 COO 张量的索引张量对象，确保与设备匹配
  Tensor indices = internal_new_from_data(
      values.options(),
      kLong,
      c10::nullopt,
      r.pyobject(0),
      /*copy_variables=*/false,
      /*copy_numpy=*/true,
      /*type_inference=*/false);
  // 调用底层函数，验证稀疏 COO 张量的参数
  at::native::_validate_sparse_coo_tensor_args(indices, values, r.intlist(2));
}

// 验证稀疏压缩张量参数的有效性
void _validate_sparse_compressed_tensor_args(
    c10::DispatchKey dispatch_key,  // 分发键，指定张量的分发方式
    at::ScalarType scalar_type,     // 标量类型，指定张量元素的数据类型
    PyObject* args,                 // Python 元组，包含函数参数
    PyObject* kwargs) {             // Python 字典，包含函数关键字参数
  auto options = dispatchKeyToTensorOptions(dispatch_key);  // 根据分发键获取张量选项
  enum {
    ARG_COMPRESSED_INDICES = 0,
    ARG_PLAIN_INDICES,
    ARG_VALUES,
    ARG_SIZE,
    ARG_LAYOUT,
    ARGS_COUNT
  };

  // 定义函数签名，描述 _validate_sparse_compressed_tensor 的参数
  const std::string signature =
      "_validate_sparse_compressed_tensor(PyObject* compressed_indices, PyObject* plain_indices, PyObject* values, IntArrayRef size, Layout layout)";
  static PythonArgParser parser({signature});  // 静态 Python 参数解析器，解析函数参数和关键字参数

  ParsedArgs<ARGS_COUNT> parsed_args;  // 解析后的参数对象，预期包含多个参数
  auto r = parser.parse(args, kwargs, parsed_args);  // 解析函数调用的参数
  // 根据数据创建新的张量对象，用于存储稀疏压缩张量的值
  Tensor values = internal_new_from_data(
      options,
      scalar_type,
      c10::nullopt,
      r.pyobject(ARG_VALUES),
      /*copy_variables=*/false,
      /*copy_numpy=*/true,
      /*type_inference=*/true);
  // 创建稀疏压缩张量的压缩索引张量对象，确保与设备匹配
  Tensor compressed_indices = internal_new_from_data(
      values.options(),
      kInt,
      c10::nullopt,
      r.pyobject(ARG_COMPRESSED_INDICES),
      /*copy_variables=*/false,
      /*copy_numpy=*/true,
      /*type_inference=*/true);
  // 创建稀疏压缩张量的普通索引张量对象，确保与设备匹配
  Tensor plain_indices = internal_new_from_data(
      values.options(),
      kInt,
      c10::nullopt,
      r.pyobject(ARG_PLAIN_INDICES),
      /*copy_variables=*/false,
      /*copy_numpy=*/true,
      /*type_inference=*/true);
  // 调用底层函数，验证稀疏压缩张量的参数
  at::native::_validate_sparse_compressed_tensor_args(
      compressed_indices,
      plain_indices,
      values,
      r.intlist(ARG_SIZE),
      r.layout(ARG_LAYOUT));
}

// 模板函数，验证稀疏压缩张量参数的有效性，根据布局要求
template <c10::Layout required_layout>
void _validate_sparse_compressed_tensor_args_template(
    c10::DispatchKey dispatch_key,  // 分发键，指定张量的分发方式
    at::ScalarType scalar_type,     // 标量类型，指定张量元素的数据类型
    PyObject* args,                 // Python 元组，包含函数参数
    PyObject* kwargs) {             // Python 字典，包含函数关键字参数
  auto options = dispatchKeyToTensorOptions(dispatch_key);  // 根据分发键获取张量选项
  enum {
    ARG_COMPRESSED_INDICES = 0,
    ARG_PLAIN_INDICES,
    ARG_VALUES,
    ARG_SIZE,
  ARGS_COUNT
};

static std::string sig;
switch (required_layout) {
  // 根据 required_layout 的值选择对应的签名字符串
  case c10::Layout::SparseCsr:
    sig =
        "_validate_sparse_csr_tensor(PyObject* crow_indices, PyObject* col_indices, PyObject* values, IntArrayRef size)";
    break;
  case c10::Layout::SparseCsc:
    sig =
        "_validate_sparse_csc_tensor(PyObject* ccol_indices, PyObject* row_indices, PyObject* values, IntArrayRef size)";
    break;
  case c10::Layout::SparseBsr:
    sig =
        "_validate_sparse_bsr_tensor(PyObject* crow_indices, PyObject* col_indices, PyObject* values, IntArrayRef size)";
    break;
  case c10::Layout::SparseBsc:
    sig =
        "_validate_sparse_bsc_tensor(PyObject* ccol_indices, PyObject* row_indices, PyObject* values, IntArrayRef size)";
    break;
  default:;
}

static PythonArgParser parser({sig});

ParsedArgs<ARGS_COUNT> parsed_args;
// 解析传入的参数，使用预定义的签名字符串
auto r = parser.parse(args, kwargs, parsed_args);

// 使用解析后的参数创建新的 Tensor 对象，从给定的数据中
// options - Tensor 的选项
// scalar_type - Tensor 的标量类型
// r.pyobject(ARG_VALUES) - 从参数中获取值数据
// copy_variables=false, copy_numpy=true, type_inference=true - 控制复制和类型推断行为
Tensor values = internal_new_from_data(
    options,
    scalar_type,
    c10::nullopt,
    r.pyobject(ARG_VALUES),
    /*copy_variables=*/false,
    /*copy_numpy=*/true,
    /*type_inference=*/true);

// 创建稀疏张量的压缩索引 Tensor 对象
Tensor compressed_indices = internal_new_from_data(
    values.options(),
    kInt,
    c10::nullopt,
    r.pyobject(ARG_COMPRESSED_INDICES),
    /*copy_variables=*/false,
    /*copy_numpy=*/true,
    /*type_inference=*/true);

// 创建稀疏张量的普通索引 Tensor 对象
Tensor plain_indices = internal_new_from_data(
    values.options(),
    kInt,
    c10::nullopt,
    r.pyobject(ARG_PLAIN_INDICES),
    /*copy_variables=*/false,
    /*copy_numpy=*/true,
    /*type_inference=*/true);

// 调用底层函数验证稀疏压缩张量的参数
at::native::_validate_sparse_compressed_tensor_args(
    compressed_indices, plain_indices, values, r.intlist(3), required_layout);
}

void _validate_sparse_csr_tensor_args(
    c10::DispatchKey dispatch_key,
    at::ScalarType scalar_type,
    PyObject* args,
    PyObject* kwargs) {
  // 调用模板函数，验证稀疏 CSR 格式的张量参数
  _validate_sparse_compressed_tensor_args_template<c10::Layout::SparseCsr>(
      dispatch_key, scalar_type, args, kwargs);
}

void _validate_sparse_csc_tensor_args(
    c10::DispatchKey dispatch_key,
    at::ScalarType scalar_type,
    PyObject* args,
    PyObject* kwargs) {
  // 调用模板函数，验证稀疏 CSC 格式的张量参数
  _validate_sparse_compressed_tensor_args_template<c10::Layout::SparseCsc>(
      dispatch_key, scalar_type, args, kwargs);
}

void _validate_sparse_bsr_tensor_args(
    c10::DispatchKey dispatch_key,
    at::ScalarType scalar_type,
    PyObject* args,
    PyObject* kwargs) {
  // 调用模板函数，验证稀疏 BSR 格式的张量参数
  _validate_sparse_compressed_tensor_args_template<c10::Layout::SparseBsr>(
      dispatch_key, scalar_type, args, kwargs);
}

void _validate_sparse_bsc_tensor_args(
    c10::DispatchKey dispatch_key,
    at::ScalarType scalar_type,
    PyObject* args,
    PyObject* kwargs) {
  // 调用模板函数，验证稀疏 BSC 格式的张量参数
  _validate_sparse_compressed_tensor_args_template<c10::Layout::SparseBsc>(
      dispatch_key, scalar_type, args, kwargs);
}

Tensor tensor_ctor(
    c10::DispatchKey dispatch_key,
    at::ScalarType scalar_type,
    PythonArgs& r) {
  if (r.idx == 0) {
    // 从 Python 参数中获取数据对象
    PyObject* data = r.pyobject(0);
    // 检查数据对象是否为 THPVariable 类型，给出相应警告
    if (THPVariable_Check(data)) {
      auto ret = PyErr_WarnEx(
          PyExc_UserWarning,
          "To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() "
          "or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).",
          1);
      if (ret != 0)
        throw python_error();
    }

    // 是否进行类型推断
    bool type_inference = r.isNone(1);
    // 是否将张量固定在内存中
    bool pin_memory = r.toBool(3);
    // 是否需要保留梯度信息
    bool args_requires_grad = r.toBool(4);
    // 使用数据创建新的张量对象
    auto new_tensor = internal_new_from_data(
        typeIdWithDefault(r, 2, dispatch_key),
        r.scalartypeWithDefault(1, scalar_type),
        r.deviceOptional(2),
        data,
        /*copy_variables=*/true,
        /*copy_numpy=*/true,
        /*type_inference=*/type_inference,
        pin_memory);
    // 获取维度名称列表
    auto names = r.toDimnameListOptional(5);
    // 若存在维度名称，则进行命名推断
    if (names) {
      at::namedinference::propagate_names(
          new_tensor, *names, /*validate_names=*/true);
    }
    // 将新张量设为叶子节点
    new_tensor.detach_();
    // 设置是否需要梯度
    new_tensor.set_requires_grad(args_requires_grad);
    // 返回新创建的张量对象
    return new_tensor;
  }
  // 抛出异常，表示参数无效
  throw std::runtime_error("tensor(): invalid arguments");
}

Tensor as_tensor(
    c10::DispatchKey dispatch_key,
    at::ScalarType scalar_type,
    PythonArgs& r) {
  // TODO: add requires_grad once we decide on semantics for sharing data.
  if (r.idx == 0) {
    // 是否进行类型推断
    bool type_inference = r.isNone(1);
    # 调用内部函数 `internal_new_from_data`，并返回其结果
    return internal_new_from_data(
        # 获取参数 `typeIdWithDefault` 的返回值作为第一个参数
        typeIdWithDefault(r, 2, dispatch_key),
        # 获取参数 `scalartypeWithDefault` 的返回值作为第二个参数
        r.scalartypeWithDefault(1, scalar_type),
        # 获取参数 `deviceOptional` 的返回值作为第三个参数
        r.deviceOptional(2),
        # 获取参数 `pyobject` 的返回值作为第四个参数
        r.pyobject(0),
        # 设置 `copy_variables` 参数为 false
        /*copy_variables=*/false,
        # 设置 `copy_numpy` 参数为 false
        /*copy_numpy=*/false,
        # 传递 `type_inference` 参数
        /*type_inference=*/type_inference);
  }
  # 如果前面的条件不满足，则抛出运行时错误
  throw std::runtime_error("tensor(): invalid arguments");
}

Tensor new_tensor(
    c10::DispatchKey dispatch_key,  // 分派键，用于确定张量的分派策略
    at::ScalarType scalar_type,     // 标量类型，指定张量的数据类型
    PyObject* args,                 // Python 参数元组
    PyObject* kwargs) {             // Python 关键字参数字典
  static PythonArgParser parser({
      "new_tensor(PyObject* data, *, ScalarType dtype=None, Device? device=None, bool requires_grad=False)",  // 定义 PythonArgParser 对象并初始化
  });

  ParsedArgs<4> parsed_args;        // 解析后的参数对象，最多支持4个参数
  auto r = parser.parse(args, kwargs, parsed_args);  // 解析 Python 参数
  if (r.idx == 0) {                 // 如果解析索引为0
    PyObject* data = r.pyobject(0); // 获取解析后的第一个 Python 对象作为数据
    if (THPVariable_Check(data)) {  // 检查数据是否为 THPVariable 类型
      auto ret = PyErr_WarnEx(
          PyExc_UserWarning,        // 发出用户警告
          "To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() "
          "or sourceTensor.clone().detach().requires_grad_(True), rather than tensor.new_tensor(sourceTensor).",  // 警告内容
          1);
      if (ret != 0)                // 如果发出警告时返回值不为0，抛出 Python 异常
        throw python_error();
    }

    bool args_requires_grad = r.toBool(3);  // 将第4个参数转换为布尔值
    auto new_tensor = new_from_data_copy(   // 调用 new_from_data_copy 函数创建新张量的拷贝
        typeIdWithDefault(r, 2, dispatch_key),     // 获取分派键的默认值
        r.scalartypeWithDefault(1, scalar_type),   // 获取标量类型的默认值
        r.deviceOptional(2),            // 获取可选设备参数
        data);
    new_tensor.detach_();             // 确保新张量为叶节点
    new_tensor.set_requires_grad(args_requires_grad);  // 设置是否需要梯度
    return new_tensor;                // 返回新张量
  }
  throw std::runtime_error("new_tensor(): invalid arguments");  // 抛出运行时错误，表示无效的参数
}

Tensor tensor_frombuffer(
    PyObject* buffer,                // Python 对象缓冲区
    ScalarType dtype,                // 标量类型，指定张量的数据类型
    int64_t count,                   // 张量的元素数量
    int64_t offset,                  // 缓冲区的偏移量
    bool requires_grad) {            // 是否需要梯度
  auto elsize = at::elementSize(dtype);  // 计算指定数据类型的元素大小
  size_t actual_count = 0;

  Py_buffer view;                   // Python 缓冲区视图结构体
  if (PyObject_GetBuffer(buffer, &view, PyBUF_WRITABLE) < 0) {  // 尝试获取可写缓冲区视图
    TORCH_CHECK(
        PyObject_GetBuffer(buffer, &view, PyBUF_SIMPLE) >= 0,  // 获取简单缓冲区视图
        "could not retrieve buffer from object");  // 抛出错误，无法从对象获取缓冲区
    TORCH_WARN_ONCE(
        "The given buffer is not writable, and PyTorch does "
        "not support non-writable tensors. This means you can write to the "
        "underlying (supposedly non-writable) buffer using the tensor. "
        "You may want to copy the buffer to protect its data or make it writable "
        "before converting it to a tensor. This type of warning will be "
        "suppressed for the rest of this program.");  // 发出一次性警告，指出不可写的缓冲区问题
    PyErr_Clear();                  // 清除 Python 异常状态
  }

  Py_INCREF(view.obj);              // 增加缓冲区对象的引用计数
  THPObjectPtr obj(view.obj);       // 将缓冲区对象封装成 THPObjectPtr 对象，确保在函数退出时自动释放

  auto len = view.len;              // 缓冲区的长度
  auto buf = view.buf;              // 缓冲区的指针
  PyBuffer_Release(&view);          // 释放缓冲区视图

  TORCH_CHECK_VALUE(
      len > 0 && count != 0,        // 检查缓冲区长度和元素数量是否有效
      "both buffer length (", len, ") and count (", count, ") must not be 0");
  TORCH_CHECK_VALUE(
      offset >= 0 && offset < len,  // 检查偏移量是否有效
      "offset (", offset, " bytes) must be non-negative and no greater than "
      "buffer length (", len, " bytes) minus 1");
  TORCH_CHECK_VALUE(
      count > 0 || (len - offset) % elsize == 0,  // 检查元素数量是否有效
      "buffer length (", len - offset, " bytes) after offset (", offset, " bytes) "
      "must be a multiple of element size (", elsize, ")");

  if (count < 0) {
    actual_count = (len - offset) / elsize;  // 计算实际元素数量
  } else {

    actual_count = count;           // 使用指定的元素数量
  }

  // 返回从缓冲区创建的张量，指定数据类型、元素数量、偏移量和是否需要梯度
  return at::empty_strided({actual_count}, {elsize}, buf + offset, {dispatchKeyToDeviceType(dispatch_key, torch::kCPU)}, dtype, requires_grad);
}
    // 将 count 转换为 size_t 类型，赋值给 actual_count
    actual_count = static_cast<size_t>(count);
  }

  // 使用 TORCH_CHECK_VALUE 宏检查请求的缓冲长度是否合法
  TORCH_CHECK_VALUE(
      // 检查请求的缓冲区长度是否不超过实际缓冲区长度
      static_cast<size_t>(offset) + actual_count * elsize <=
          static_cast<size_t>(len),
      "requested buffer length (",
      actual_count,
      " * ",
      elsize,
      " bytes) "
      "after offset (",
      offset,
      " bytes) must not be greater than actual "
      "buffer length (",
      len,
      " bytes)");

  // 将 buf 的指针向前偏移 offset 个字节，赋值给 offset_buf
  auto offset_buf = static_cast<char*>(buf) + offset;
  
  // 设置张量的选项，包括数据类型 dtype 和设备类型为 CPU
  auto options = TensorOptions().dtype(dtype).device(c10::kCPU);

  // 创建张量 tensor，使用 offset_buf 作为数据源，长度为 actual_count
  auto tensor = at::for_blob(offset_buf, static_cast<int64_t>(actual_count))
                    .options(options)
                    // 设置在张量销毁时释放 Python 对象 obj
                    .deleter([obj = obj.release()](void*) {
                      // 获取全局解释器锁（GIL），以安全释放 Python 对象
                      pybind11::gil_scoped_acquire gil;
                      Py_DECREF(obj);
                    })
                    .make_tensor();
  // 设置张量是否需要梯度
  tensor.set_requires_grad(requires_grad);
  // 返回创建的张量
  return tensor;
  // 定义从 Python 对象中获取 DLManagedTensor 的函数，返回一个 Torch 的 Tensor
Tensor tensor_fromDLPack(PyObject* data) {
  // 从 Python Capsule 中获取 DLManagedTensor 指针
  DLManagedTensor* dlMTensor =
      (DLManagedTensor*)PyCapsule_GetPointer(data, "dltensor");
  // 使用 TORCH_CHECK 进行断言，确保 dlMTensor 不为空，否则抛出异常
  TORCH_CHECK(
      dlMTensor,
      "from_dlpack received an invalid capsule. "
      "Note that DLTensor capsules can be consumed only once, "
      "so you might have already constructed a tensor from it once.");

  // 创建一个带有 GIL（全局解释器锁）的 lambda 函数，用于执行 dlMTensor 的释放函数
  auto deleter_with_gil = [dlMTensor](void*) {
    if (dlMTensor->deleter) {
      pybind11::gil_scoped_acquire gil;  // 获取 GIL
      dlMTensor->deleter(dlMTensor);     // 调用 dlMTensor 的释放函数
    }
  };

  // atensor 从 dlMTensor 中创建 Torch 的 Tensor 对象，并且使用指定的析构函数
  // 如果是由于 NumPy 的 bug 导致的问题，则确保在创建时持有 GIL
  auto atensor = torch::utils::is_numpy_dlpack_deleter_bugged()
      ? at::fromDLPack(dlMTensor, std::move(deleter_with_gil))
      : at::fromDLPack(dlMTensor);

  // 确保这个 Python Capsule 不会再次使用
  PyCapsule_SetName(data, "used_dltensor");

  // 如果这是第一次调用创建 Tensor 的函数，则在此处调用 _lazy_init 来初始化设备类型
  maybe_initialize_device(atensor.device());

  // 返回创建的 Tensor 对象
  return atensor;
}

// 将 Python 对象转换为 Torch 的 Tensor 对象
Tensor asarray(
    PyObject* obj,
    std::optional<ScalarType> dtype,
    std::optional<Device> device,
    std::optional<bool> copy,
    bool requires_grad) {
  // 初始化 Tensor 对象
  Tensor tensor;

  // 根据参数确定是否复制数据
  bool force_copy = copy.value_or(false);
  bool force_alias = !copy.value_or(true);
  bool should_warn_numpy_not_writable = false;

  // 确定数据类型，如果未指定则使用默认标量类型
  auto dtype_unwrapped =
      dtype.value_or(torch::tensors::get_default_scalar_type());

  // 检查 'obj' 是否为 'Tensor' 类型
  if (THPVariable_Check(obj)) {
    tensor = THPVariable_Unpack(obj);  // 拆包成 Torch 的 Tensor 对象
  }

#ifdef USE_NUMPY
  // 如果 NumPy 可用，则检查 'obj' 是否为 NumPy 数组或标量
  if (is_numpy_available()) {
    bool is_numpy_array = PyArray_Check(obj);       // 检查是否为数组
    bool is_numpy_scalar = PyArray_CheckScalar(obj);// 检查是否为标量
    // 检查对象是否为 NumPy 数组或标量
    if (is_numpy_array || is_numpy_scalar) {
      // 创建一个指向 Python 对象的智能指针
      THPObjectPtr ptr;
      // 将 obj 赋值给 arr
      auto arr = obj;

      // 如果是 NumPy 标量而不是数组
      if (is_numpy_scalar && !is_numpy_array) {
        // 检查是否禁用了强制别名选项
        TORCH_CHECK(
            !force_alias,
            "can't alias NumPy scalars. ",
            "Either remove copy=False or transform it in a ndarray. ")

        // 从标量创建一个新的 NumPy 数组
        ptr = PyArray_FromScalar(obj, nullptr);
        // 将 arr 更新为新创建的数组对象
        arr = ptr.get();
      }

      // 将 NumPy 数组转换为 PyTorch 张量
      tensor = tensor_from_numpy(arr, /*warn_if_not_writeable=*/false);
      // 检查 NumPy 数组是否可写
      should_warn_numpy_not_writable =
          !PyArray_ISWRITEABLE((PyArrayObject*)arr);

      // 如果是 NumPy 标量
      if (is_numpy_scalar) {
        // 使用克隆的存储而不是共享的存储
        // THPObjectPtr 将在前一个作用域结束时删除先前的存储
        tensor = tensor.clone();

        // 后续不需要再次克隆
        force_copy = false;
      }
    }
#endif

// 检查 'obj' 是否是一个 'DLPack' 的胶囊
if (!tensor.defined() && PyCapsule_IsValid(obj, "dltensor") != 0) {
  // 如果是有效的 DLPack 胶囊，则从中创建一个 PyTorch 张量
  tensor = tensor_fromDLPack(obj);
}

// 检查 'obj' 是否实现了缓冲区协议
if (!tensor.defined() && PyObject_CheckBuffer(obj) != 0) {
  // 如果实现了缓冲区协议，则从中创建一个 PyTorch 张量
  tensor = tensor_frombuffer(obj, dtype_unwrapped, -1, 0, requires_grad);
}

if (tensor.defined()) {
  // 对于可别名的张量，是否需要复制它？
  bool wrong_device = device.has_value() && device.value() != tensor.device();
  bool wrong_dtype =
      dtype.has_value() && dtype.value() != tensor.scalar_type();
  bool needs_copying = !copy.has_value() && (wrong_device || wrong_dtype);

  // 给定一个定义好的张量，如果需要复制（copy=True）或因设备或数据类型不匹配而需要复制（copy=None），则执行复制操作
  if (force_copy || needs_copying) {
    if (wrong_device || wrong_dtype) {
      // 如果设备或数据类型不匹配，则将张量转移到指定的设备和数据类型
      tensor = tensor.to(
          device.value_or(tensor.device()),
          dtype.value_or(tensor.scalar_type()),
          /*non_blocking=*/false,
          /*copy=*/force_copy);
    } else {
      // 否则执行克隆操作
      tensor = tensor.clone();
    }
  } else {
    // 如果不需要复制，则检查张量是否在正确的设备和具有正确的数据类型
    TORCH_CHECK_VALUE(
        !wrong_device,
        "can't alias tensor from device '",
        tensor.device(),
        "' to '",
        device.value(),
        "'.");
    TORCH_CHECK_VALUE(
        !wrong_dtype,
        "can't alias tensor with dtype '",
        tensor.scalar_type(),
        "' into dtype '",
        dtype.value(),
        "'.");
    // 如果张量是 NumPy 数组视图，则在必要时警告用户关于不可写数组
    if (should_warn_numpy_not_writable) {
      warn_numpy_not_writeable();
    }
  }

  // 当张量不是叶子节点且不需要梯度时，使用 'detach' 方法
  if (!tensor.is_leaf() && !requires_grad) {
    tensor = tensor.detach();
  } else {
    // 否则根据需求设置梯度追踪
    tensor.set_requires_grad(requires_grad);
  }
} else {
  // 未定义的张量意味着它既未实现 DLPack 也未实现缓冲区协议，最后一种情况是序列，此时必须复制（copy 不能为 false）
  TORCH_CHECK_VALUE(
      !force_alias, "can't alias arbitrary sequence into a tensor.");

  // 从序列创建张量，并推断其类型，然后将其转换为所需类型
  // 仅当未指定 dtype 时才激活类型推断，否则强制使用 unwrapped dtype
  tensor = internal_new_from_data(
      TensorOptions(),
      dtype_unwrapped,
      device,
      obj,
      /* copy_variables = */ false,
      /* copy_numpy = */ false,
      /* type_inference = */ !dtype.has_value());
    tensor.set_requires_grad(requires_grad);

设置张量的梯度计算需求。


  }

结束函数并返回张量。
# 结束命名空间
}

# 定义函数，返回是否只提升 CPU 张量
bool only_lift_cpu_tensors() {
  return kOnlyLiftCPUTensors;
}

# 定义函数，设置是否只提升 CPU 张量
void set_only_lift_cpu_tensors(bool value) {
  kOnlyLiftCPUTensors = value;
}

# 结束命名空间
} // namespace torch::utils
```