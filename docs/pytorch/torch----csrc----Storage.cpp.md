# `.\pytorch\torch\csrc\Storage.cpp`

```py
// 包含 Torch 的 Python 头文件
#include <torch/csrc/python_headers.h>
#ifdef _MSC_VER
// 包含 Windows 平台特定头文件
#include <c10/util/win32-headers.h>
#endif
// 包含 Python 中对结构成员的定义
#include <structmember.h>

// 包含 ATen 库中的 MPSDevice 类定义
#include <ATen/mps/MPSDevice.h>
// 包含 C10 库中的 CPUAllocator 类定义
#include <c10/core/CPUAllocator.h>
// 包含 C10 库中的 RefcountedDeleter 类定义
#include <c10/core/RefcountedDeleter.h>
// 包含 libshm 库的头文件
#include <libshm.h>
// 包含 Torch 中的 CUDA IPC 类型定义
#include <torch/csrc/CudaIPCTypes.h>
// 包含 Torch 中的 Device 类定义
#include <torch/csrc/Device.h>
// 包含 Torch 中的动态类型定义
#include <torch/csrc/DynamicTypes.h>
// 包含 Torch 中的存储方法定义
#include <torch/csrc/StorageMethods.h>
// 包含 Torch 中的存储共享定义
#include <torch/csrc/StorageSharing.h>
// 包含 Torch 中的 THP.h 头文件
#include <torch/csrc/THP.h>
// 包含 Torch 中的自动求导工具函数定义
#include <torch/csrc/autograd/utils/wrap_outputs.h>
// 包含 Torch 中的复制工具函数定义
#include <torch/csrc/copy_utils.h>
// 包含 Torch 中的 PyObject 保留工具函数定义
#include <torch/csrc/utils/pyobject_preservation.h>
// 包含 Torch 中的 Python 参数解析工具函数定义
#include <torch/csrc/utils/python_arg_parser.h>

// 包含 C10 库中的 intrusive_ptr 定义
#include <c10/util/intrusive_ptr.h>
// 包含 fmt 库中的格式化函数定义
#include <fmt/format.h>

// THPPointer 类模板特化，用于释放 c10::StorageImpl 类型指针资源
template <>
void THPPointer<c10::StorageImpl>::free() {
  if (ptr) {
    c10::raw::intrusive_ptr::decref(ptr);
  }
}

// THPStorageClass 的全局变量声明，初始化为空指针
PyTypeObject* THPStorageClass = nullptr;

// 创建一个新的 Storage 对象，并返回对应的 Python 对象
PyObject* THPStorage_NewWithStorage(
    PyTypeObject* type,
    c10::Storage _storage,
    c10::impl::PyInterpreterStatus status,
    bool allow_preexisting_pyobj) {
  // 检查传入的 type 是否是 THPStorageType 的子类
  TORCH_CHECK(
      PyType_IsSubtype(type, &THPStorageType),
      "Creating a Storage subclass from a class that does not inherit from ",
      "Storage is not possible. Make sure your class inherits from Storage.");

  // 检查是否已经存在与 Storage 关联的 Python 对象
  auto maybe_pyobj = _storage.unsafeGetStorageImpl()->pyobj_slot()->check_pyobj(
      getPyInterpreter(), /*ignore_hermetic_tls=*/false);
  if (maybe_pyobj.has_value() && maybe_pyobj.value()) {
    // 如果已存在 Python 对象，并且不允许使用已存在的 Python 对象，则报错
    TORCH_CHECK(
        allow_preexisting_pyobj,
        "Creating a new Storage subclass ",
        type->tp_name,
        " but the raw Storage object is already associated to a python object ",
        "of type ",
        maybe_pyobj.value()->ob_type->tp_name);
    // 获取已存在的 Python 对象，并进行类型检查
    PyObject* obj = *maybe_pyobj;
    PyTypeObject* obj_type = Py_TYPE(obj);
    TORCH_CHECK(
        obj_type == type || PyType_IsSubtype(obj_type, type),
        "Creating a new Storage subclass ",
        type->tp_name,
        " but the raw Storage object is already associated to a python object ",
        "of type ",
        maybe_pyobj.value()->ob_type->tp_name,
        " which is not a subclass of the "
        "requested type");
    // 包装并返回 Storage 对象
    return THPStorage_Wrap(std::move(_storage));
  }

  // 分配新的 Python 对象内存
  PyObject* obj = type->tp_alloc(type, 0);
  TORCH_CHECK(obj, "Failed to allocate a ", type->tp_name, " object");

  // 将分配的内存转换为 THPStorage 类型
  auto s = (THPStorage*)obj;

  // 在分配的内存上构造 c10::MaybeOwned<c10::Storage> 对象
  new (&s->cdata) c10::MaybeOwned<c10::Storage>();

  // 将传入的 Storage 对象移动到 cdata 成员中
  s->cdata = c10::MaybeOwned<c10::Storage>::owned(std::move(_storage));

  // 如果不是隔离模式，则初始化 Python 对象并关联到 Storage 上
  if (!c10::impl::HermeticPyObjectTLS::get_state()) {
    s->is_hermetic = false;
    const auto& storage = THPStorage_Unpack(s);
    storage.unsafeGetStorageImpl()->pyobj_slot()->init_pyobj(
        getPyInterpreter(), obj, status);
  } else {
    // 否则，标记为隔离模式
    s->is_hermetic = true;
  }

  // 返回创建的 Python 对象
  return obj;
}

// 使用 c10::Storage 封装一个 Storage 对象，并返回 PyObject 指针
PyObject* THPStorage_Wrap(c10::Storage storage) {
  // 获取 StorageImpl 指针
  c10::StorageImpl* storage_impl = storage.unsafeGetStorageImpl();
  // 如果处于隔离模式，则返回空指针
  if (c10::impl::HermeticPyObjectTLS::get_state()) {
    // 使用 THPStorage_NewWithStorage 创建一个新的 THPStorage 对象，并传入指定的参数
    return THPStorage_NewWithStorage(
        THPStorageClass,  // THPStorageClass 是 THPStorage 对象的类型对象
        std::move(storage),  // 使用 std::move 将 storage 移动到 THPStorage_NewWithStorage 函数中
        c10::impl::PyInterpreterStatus::DEFINITELY_UNINITIALIZED  // 使用 DEFINITELY_UNINITIALIZED 标记 PyInterpreter 的状态
    );
  }
  
  // 获取 storage_impl 的 pyobj_slot
  c10::impl::PyObjectSlot* pyobj_slot = storage_impl->pyobj_slot();

  // 如果 StorageImpl 具有由当前解释器以外的解释器管理的 PyObject，
  // 则创建一个指向相同数据的新 StorageImpl，并从此创建 Python 存储。
  // 注意：这仅应在 MultiPy 环境中发生
  if (pyobj_slot->has_pyobj_nonhermetic() &&
      !pyobj_slot->check_interpreter(getPyInterpreter())) {
    // 使用 newStorageImplFromRefcountedDataPtr 创建一个新的 StorageImpl，并传入 THPStorage_NewWithStorage
    return THPStorage_NewWithStorage(
        THPStorageClass,
        c10::newStorageImplFromRefcountedDataPtr(storage),
        c10::impl::PyInterpreterStatus::DEFINITELY_UNINITIALIZED
    );
  }
  
  // 检查 pyobj_slot 是否有相关的 PyObject
  std::optional<PyObject*> maybe_pyobj = pyobj_slot->check_pyobj(
      getPyInterpreter(), /*ignore_hermetic_tls=*/false);
  // 设置默认的 PyInterpreter 状态为 TAGGED_BY_US
  c10::impl::PyInterpreterStatus status =
      c10::impl::PyInterpreterStatus::TAGGED_BY_US;
  
  // 如果 maybe_pyobj 中有值
  if (maybe_pyobj.has_value()) {
    auto obj = *maybe_pyobj;
    if (obj) {
      // 检查 obj 是否为 THPStorage 类型的对象
      TORCH_CHECK(
          THPStorage_Check(obj),
          "Expected a storage type, but got ",
          Py_TYPE(obj)->tp_name);

      // 如果 pyobj_slot 拥有该 PyObject
      if (pyobj_slot->owns_pyobj()) {
        pyobj_slot->set_owns_pyobj(false);  // 设置 owns_pyobj 为 false
        reinterpret_cast<THPStorage*>(obj)->cdata =
            c10::MaybeOwned<c10::Storage>::owned(std::move(storage));  // 将 storage 移动到 cdata 中
        return obj;  // 返回 obj
      } else {
        Py_INCREF(obj);  // 增加 obj 的引用计数
        return obj;  // 返回 obj
      }
    }
    status = c10::impl::PyInterpreterStatus::TAGGED_BY_US;  // 设置状态为 TAGGED_BY_US
  } else {
    // 如果 storage 的 use_count() 小于等于 1
    if (storage.use_count() <= 1) {
      status = c10::impl::PyInterpreterStatus::DEFINITELY_UNINITIALIZED;  // 设置状态为 DEFINITELY_UNINITIALIZED
    } else {
      status = c10::impl::PyInterpreterStatus::MAYBE_UNINITIALIZED;  // 设置状态为 MAYBE_UNINITIALIZED
    }
  }
  
  // 使用 THPStorage_NewWithStorage 创建一个新的 THPStorage 对象，并传入指定的参数
  return THPStorage_NewWithStorage(
      THPStorageClass,  // THPStorageClass 是 THPStorage 对象的类型对象
      std::move(storage),  // 使用 std::move 将 storage 移动到 THPStorage_NewWithStorage 函数中
      status  // 使用 status 标记 PyInterpreter 的状态
  );
// 检查是否可以保留 THPStorage 对象
static bool THPStorage_isPreservable(THPStorage* self) {
  // 如果是借用状态的对象，不能保留
  if (self->cdata.unsafeIsBorrowed()) {
    return false;
  }
  // 解包 THPStorage 对象
  auto const& storage = THPStorage_Unpack(self);

  // 如果标记为 hermetic，不能保留
  if (self->is_hermetic) {
    return false;
  }

  // 检查存储实现的 PyObjectSlot 是否匹配，并且引用计数大于 1 才能保留
  if (storage.unsafeGetStorageImpl()->pyobj_slot()->check_pyobj(
          getPyInterpreter(), /*ignore_hermetic_tls=*/true) !=
      c10::make_optional((PyObject*)self)) {
    return false;
  }
  if (storage.use_count() <= 1) {
    return false;
  }
  // 可以保留
  return true;
}

// 尝试保留 THPStorage 对象
static bool THPStorage_tryPreserve(THPStorage* self) {
  // 如果不能保留，则返回 false
  if (!THPStorage_isPreservable(self)) {
    return false;
  }

  // 解包 THPStorage 对象
  const auto& storage = THPStorage_Unpack(self);
  c10::StorageImpl* storage_impl = storage.unsafeGetStorageImpl();

  // 获取存储实现的 PyObjectSlot
  auto maybe_pyobj = storage_impl->pyobj_slot()->check_pyobj(
      getPyInterpreter(),
      /*ignore_hermetic_tls=*/true);
  // 注意：这里可能只需设置 PyObjectSlot，但关键是我们应该在创建存储的 PyObject 时已经设置了 PyObjectSlot
  TORCH_INTERNAL_ASSERT(
      maybe_pyobj.has_value(),
      "Trying to preserve a Python storage whose PyObjectSlot does not have a PyObject");

  // 获取 PyObject
  PyObject* pyobj = *maybe_pyobj;

  // 检查 PyObject 是否是预期的存储类型
  TORCH_CHECK(
      THPStorage_Check(pyobj),
      "Expected a storage type, but got ",
      Py_TYPE(pyobj)->tp_name);

  // 确保 Python 存储和 PyObjectSlot 中的 PyObject 地址相同
  TORCH_INTERNAL_ASSERT(
      (void*)pyobj == (void*)self,
      "Python storage and the PyObject in the internal PyObjectSlot are not at the same address");

  // 确保 PyObjectSlot 不拥有 PyObject
  TORCH_INTERNAL_ASSERT(!storage_impl->pyobj_slot()->owns_pyobj());

  // 设置 PyObjectSlot 拥有 PyObject，并增加其引用计数
  storage_impl->pyobj_slot()->set_owns_pyobj(true);
  Py_INCREF(self);

  // 将存储数据赋值给 THPStorage 对象的 cdata
  self->cdata = c10::MaybeOwned<c10::Storage>::borrowed(storage);
  return true;
}

// 自定义子类析构函数，用于 THPStorage 对象
static void THPStorage_subclass_dealloc(PyObject* self) {
  THPStorage* _self = (THPStorage*)self;

  // 尝试保留 THPStorage 对象，如果成功则直接返回
  if (THPStorage_tryPreserve(_self)) {
    return;
  }

  // 检查是否为 GC-tracked 对象的子类
  auto* type = Py_TYPE(self);
  if (PyType_HasFeature(type, Py_TPFLAGS_HAVE_GC) != 0) {
    PyObject_GC_UnTrack(self);
  }

  // 检查是否有 finalizer
  bool has_finalizer = type->tp_finalize || type->tp_del;

  if (type->tp_finalize) {
    PyObject_GC_Track(self);
    // 调用 finalizer，如果返回负值表示 finalizer 使 PyObject 复活
    if (PyObject_CallFinalizerFromDealloc(self) < 0) {
      // finalizer 使 PyObject 复活，停止释放对象
      return;
    }
    PyObject_GC_UnTrack(self);
  }

  // 清除弱引用
  if (type->tp_weaklistoffset) {
    PyObject_ClearWeakRefs(self);
  }

  // 调用 tp_del 方法
  if (type->tp_del) {
    PyObject_GC_Track(self);
    type->tp_del(self);
    // 如果对象被引用，则停止释放对象
    if (Py_REFCNT(self) > 0) {
      return;
    }
    PyObject_GC_UnTrack(self);
  }

  // 如果有 finalizer，则执行以下代码
  if (has_finalizer) {
    /* New weakrefs could be created during the finalizer call.
       If this occurs, clear them out without calling their
       finalizers since they might rely on part of the object
       being finalized that has already been destroyed. */
    // 如果在析构函数调用期间创建了新的弱引用对象，
    // 则必须在不调用其析构函数的情况下清除它们，
    // 因为这些析构函数可能依赖于已经被销毁的对象的一部分。
    if (type->tp_weaklistoffset) {
      /* Modeled after GET_WEAKREFS_LISTPTR() */
      // 获取对象的弱引用列表指针
      PyWeakReference** list =
          (PyWeakReference**)PyObject_GET_WEAKREFS_LISTPTR(self);
      // 遍历弱引用列表，并清除引用
      while (*list)
        _PyWeakref_ClearRef(*list);
    }
  }

  // 清除槽位（slots）
  {
    // 从当前类型开始逐级清除父类的槽位
    PyTypeObject* base = type;
    while (base != &THPStorageType) {
      // 如果当前类型有槽位，执行清除操作
      if (Py_SIZE(base)) {
        clear_slots(base, self);
      }
      // 移动到父类
      base = base->tp_base;
      // 断言父类存在
      TORCH_INTERNAL_ASSERT(base);
    }
  }

  // 清除 __dict__ 属性
  if (C10_LIKELY(type->tp_dictoffset)) {
    // 获取对象的 __dict__ 指针
    PyObject** dictptr = _PyObject_GetDictPtr(self);
    if (dictptr != nullptr) {
      PyObject* dict = *dictptr;
      if (dict != nullptr) {
        // 减少字典的引用计数并置空指针
        Py_DECREF(dict);
        *dictptr = nullptr;
      }
    }
  }

  // 断言对象的类型与当前类型相符
  TORCH_INTERNAL_ASSERT(Py_TYPE(self) == type);

  // 销毁存储对象
  _self->cdata.~MaybeOwned<c10::Storage>();
  // 调用类型的 tp_free 方法释放对象
  Py_TYPE(_self)->tp_free(self);

  // 断言当前类型是堆类型
  TORCH_INTERNAL_ASSERT(type->tp_flags & Py_TPFLAGS_HEAPTYPE);
  // 减少类型的引用计数
  Py_DECREF(type);
// 定义 THPStorage_pynew 函数，用于创建新的 Torch Storage 对象
static PyObject* THPStorage_pynew(
    PyTypeObject* type,
    PyObject* args,
    PyObject* kwargs) {
  // 处理 Torch 异常
  HANDLE_TH_ERRORS
  // 检查是否直接构造了 StorageBase，应该通过子类化来构造
  TORCH_CHECK(
      type != &THPStorageType,
      "Cannot directly construct StorageBase; subclass it and then construct that");

  // 定义静态的 PythonArgParser 对象，并列出可能的构造函数参数
  static torch::PythonArgParser parser({
      THPStorageStr "(*, int64_t allocator=None, Device device=None)",
      THPStorageStr
      "(int64_t size, *, int64_t allocator=None, Device device=None)",
      THPStorageStr
      "(PyObject* sequence, *, int64_t allocator=None, Device device=None)",
  });

  // 解析参数并存储在 parsed_args 中
  torch::ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  // 初始化 allocator_arg_idx 和 device_arg_idx
  int allocator_arg_idx = 0;
  int device_arg_idx = 1;

  // 根据解析结果确定 allocator_arg_idx 和 device_arg_idx 的值
  if (r.idx > 0) {
    allocator_arg_idx = 1;
    device_arg_idx = 2;
  }

  // 将 allocator 和 device 参数解析为可选的值
  std::optional<int64_t> allocator_opt = r.toInt64Optional(allocator_arg_idx);
  std::optional<at::Device> device_opt = r.deviceOptional(device_arg_idx);

  // 检查 allocator 和 device 参数不能同时给定
  TORCH_CHECK(
      !allocator_opt.has_value() || !device_opt.has_value(),
      THPStorageStr,
      "(): only one or neither of 'allocator' or 'device' can ",
      "be given, but not both");

  // 初始化 self, allocator, device_guard
  PyObject* self = nullptr;
  c10::Allocator* allocator = nullptr;
  at::OptionalDeviceGuard device_guard;

  // 根据 allocator_opt 和 device_opt 的值选择合适的 allocator
  if (allocator_opt.has_value()) {
    // 将 int64_t 类型的 allocator 转换为 c10::Allocator 指针
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    allocator = reinterpret_cast<c10::Allocator*>(allocator_opt.value());
  } else if (device_opt.has_value()) {
    // 如果给定了 device，则根据其类型选择对应的 allocator
    at::Device device = device_opt.value();
    if (device.type() == at::kCPU) {
      allocator = c10::GetDefaultCPUAllocator();
#ifdef USE_CUDA
    } else if (device.type() == at::kCUDA) {
      // 如果是 CUDA 设备，则初始化 CUDA 上下文并获取 CUDA 分配器
      at::globalContext().lazyInitCUDA();
      allocator = c10::cuda::CUDACachingAllocator::get();
#endif
#ifdef USE_MPS
    } else if (device.type() == at::kMPS) {
      // 如果是 MPS 设备，则获取 MPS 分配器
      allocator = at::mps::GetMPSAllocator();
#endif
      // 处理其他特定设备类型
      // NOLINTBEGIN(bugprone-branch-clone)
    } else if (device.type() == at::DeviceType::XPU) {
      allocator = c10::GetAllocator(device.type());
    } else if (device.type() == at::DeviceType::HPU) {
      allocator = c10::GetAllocator(device.type());
    } else if (device.type() == at::DeviceType::Meta) {
      allocator = c10::GetAllocator(device.type());
    } else if (device.type() == at::DeviceType::PrivateUse1) {
      // 初始化 PrivateUse1 设备上下文并获取分配器
      at::globalContext().lazyInitPrivateUse1();
      allocator = c10::GetAllocator(device.type());
    } else if (device.type() == at::DeviceType::MAIA) {
      allocator = c10::GetAllocator(device.type());
    } else {
      // 如果设备类型未识别，则抛出异常
      // NOLINTEND(bugprone-branch-clone)
      TORCH_CHECK(
          false,
          THPStorageStr,
          "(): Storage device not recognized: ",
          device.type());
    }
    // 设置设备保护，确保在作用域结束时恢复原设备
    device_guard.reset_device(device);
  } else {
    // 默认情况下使用 CPU 默认分配器
    allocator = c10::GetDefaultCPUAllocator();
  }

  // 处理 torch.Storage(*, ...) 构造函数的情况
  if (r.idx == 0) {
    self = THPStorage_NewWithStorage(
        type,
        make_storage_impl(
            c10::StorageImpl::use_byte_size_t(),
            0,
            at::DataPtr(),
            allocator,
            /*resizable=*/true,
            device_opt),
        c10::impl::PyInterpreterStatus::DEFINITELY_UNINITIALIZED);

    // 如果返回值的索引是0，表明参数是一个类型对象
  } else if (r.idx == 1) {
    // 从返回结果中提取第一个整数作为存储空间的大小
    int64_t size = r.toInt64(0);
    // 使用给定的大小创建一个新的存储空间对象
    self = THPStorage_NewWithStorage(
        type,
        make_storage_impl(
            c10::StorageImpl::use_byte_size_t(),
            size,
            at::DataPtr(),
            allocator,
            /*resizable=*/true,
            device_opt),
        c10::impl::PyInterpreterStatus::DEFINITELY_UNINITIALIZED);

    // 如果返回值的索引是1，表明参数是一个整数大小的序列
  } else if (r.idx == 2) {
    // 获取 Python 对象作为序列
    PyObject* sequence = r.pyobject(0);
    // 获取序列的长度
    Py_ssize_t length = PySequence_Length(sequence);
    // 检查序列是否确实是一个序列类型
    TORCH_CHECK(
        PySequence_Check(sequence),
        THPStorageStr,
        "(): Expected a sequence type, but got ",
        THPUtils_typename(sequence));
    // 检查序列长度是否非负
    TORCH_CHECK(
        length >= 0,
        THPStorageStr,
        "(): Could not obtain the length of sequence of type ",
        THPUtils_typename(sequence));
    // 使用序列的长度创建一个新的存储空间对象
    self = THPStorage_NewWithStorage(
        type,
        make_storage_impl(
            c10::StorageImpl::use_byte_size_t(),
            length,
            at::DataPtr(),
            allocator,
            /*resizable=*/true,
            device_opt),
        c10::impl::PyInterpreterStatus::DEFINITELY_UNINITIALIZED);
    THPObjectPtr item;
    try {
      // 解包存储空间
      const auto& storage = THPStorage_Unpack(self);
      // 遍历序列中的每一个项
      for (Py_ssize_t i = 0; i < length; i++) {
        // 获取序列中的第 i 个项
        item = PySequence_GetItem(sequence, i);
        // 将 Python 对象解包为 uint8_t 类型的值
        uint8_t value = THPByteUtils_unpackReal(item.get());
        // 根据分配器类型将值写入存储空间
        if (allocator == c10::GetDefaultCPUAllocator()) {
          static_cast<uint8_t*>(storage.mutable_data())[i] = value;
        } else {
          // TODO: 可能较慢，考虑批量更新？
          storage_set(storage, i, value);
        }
      }
    } catch (const std::exception& e) {
      // 如果发生异常，报错并返回空指针
      TORCH_CHECK(
          THPStorageStr "(): tried to construct a storage from a sequence (",
          THPUtils_typename(sequence),
          "), ",
          "but one of the items was of type ",
          THPUtils_typename(item.get()),
          " instead of int");
      return nullptr;
    }
  }
  // 返回创建的存储空间对象或空指针
  return self;
  // 返回 None，表示函数执行完毕
  Py_RETURN_NONE;
  // 处理 Torch 错误的结尾标记
  END_HANDLE_TH_ERRORS
// THPStorage_length 函数实现，返回 THPStorage 对象的长度
static Py_ssize_t THPStorage_length(THPStorage* self) {
  HANDLE_TH_ERRORS
  // 确保 THPStorage 对象不为空
  THPStorage_assertNotNull(self);
  // 获取存储的字节大小，并转换为 Py_ssize_t 类型返回
  return static_cast<Py_ssize_t>(THPStorage_Unpack(self).nbytes());
  END_HANDLE_TH_ERRORS_RET(-1)
}

// THPStorage_get 函数实现，用于获取 THPStorage 对象中的元素
static PyObject* THPStorage_get(THPStorage* self, PyObject* index) {
  HANDLE_TH_ERRORS
  // 确保 THPStorage 对象不为空
  THPStorage_assertNotNull(self);
  // 获取存储对象的引用
  const auto& storage = THPStorage_Unpack(self);
  // 获取存储的长度
  int64_t len = static_cast<int64_t>(storage.nbytes());
  /* Integer index */
  // 检查是否为整数索引
  if (THPUtils_checkLong(index)) {
    // 解包整数索引
    int64_t nindex = THPUtils_unpackLong(index);
    // 处理负索引
    if (nindex < 0)
      nindex += len;
    // 检查索引是否有效
    if (nindex < 0 || nindex >= len) {
      // 抛出索引错误异常
      PyErr_SetString(
          PyExc_IndexError,
          fmt::format(
              "index {} out of range for storage of size {}", nindex, len));
      return nullptr;
    }
    // 获取存储中索引位置的值，并返回作为 Python 对象
    uint8_t value = storage_get(storage, nindex);
    return THPByteUtils_newReal(value);
    /* Slice index */
  } else if (PySlice_Check(index)) {
    // 处理切片索引
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    Py_ssize_t start, stop, slicelength, step;
    // 解包切片参数
    if (PySlice_Unpack(index, &start, &stop, &step) < 0) {
      return nullptr;
    }
    // 调整切片参数
    slicelength = PySlice_AdjustIndices(len, &start, &stop, step);
    // 检查步长是否为1
    if (step != 1) {
      TORCH_CHECK(
          "Trying to slice with a step of ",
          step,
          ", but only a step of "
          "1 is supported");
      return nullptr;
    }

    // 获取存储对象的引用
    const auto& storage = THPStorage_Unpack(self);
    // 获取存储对象的数据指针
    auto data = static_cast<uint8_t*>(storage.mutable_data());

    // 增加存储实现对象的引用计数
    at::StorageImpl* old_storage_impl = storage.unsafeGetStorageImpl();
    c10::raw::intrusive_ptr::incref(old_storage_impl);
    // 获取设备信息
    std::optional<at::Device> device_opt = old_storage_impl->device();
    // 创建新的存储实现对象
    auto new_storage_impl = make_storage_impl(
        c10::StorageImpl::use_byte_size_t(),
#ifdef THQUANTIZED
        slicelength * sizeof(quantized_t),
#else
        slicelength,
#endif
        at::DataPtr(
            static_cast<void*>(data + start),
            old_storage_impl,
            [](void* s) {
              c10::raw::intrusive_ptr::decref(static_cast<at::StorageImpl*>(s));
            },
            old_storage_impl->device()),
        old_storage_impl->allocator(),
        /* resizable */ false,
        device_opt);

    // 使用新的存储实现创建 THPStorage 对象，并返回
    PyObject* _ret = THPStorage_NewWithStorage(
        Py_TYPE(self),
        std::move(new_storage_impl),
        c10::impl::PyInterpreterStatus::DEFINITELY_UNINITIALIZED);

    return _ret;
  }
  // 抛出类型错误异常，表示无法用 index 对象索引 THPStorage
  PyErr_Format(
      PyExc_TypeError,
      "can't index a " THPStorageStr " with %s",
      THPUtils_typename(index));
  return nullptr;
  END_HANDLE_TH_ERRORS
}

// THPStorage_set 函数实现，用于设置 THPStorage 对象中的元素
static int THPStorage_set(THPStorage* self, PyObject* index, PyObject* value) {
  HANDLE_TH_ERRORS
  // 确保 THPStorage 对象不为空
  THPStorage_assertNotNull(self);
  // 检查值是否为整数类型
  if (!THPByteUtils_checkReal(value)) {
    TORCH_CHECK(
        "can only set storage content with a int types, but got ",
        THPUtils_typename(value),
        " instead");
    // 返回错误码 -1，表示操作失败
    return -1;
  }

  // 解包传入的 value 参数为实数类型
  uint8_t rvalue = THPByteUtils_unpackReal(value);
  // 解包 self 参数为 storage 对象
  const auto& storage = THPStorage_Unpack(self);

  // 检查 index 是否为长整型
  if (THPUtils_checkLong(index)) {
    // 解包 index 参数为 int64_t 类型
    int64_t nindex = THPUtils_unpackLong(index);
    // 调用 storage_set 函数设置 storage 对象的第 nindex 个位置为 rvalue
    storage_set(storage, nindex, rvalue);
    // 返回成功状态码 0
    return 0;
  } else if (PySlice_Check(index)) {
    // 如果 index 是切片对象
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    Py_ssize_t start, stop, step;
    // 获取 storage 的长度作为切片的边界
    Py_ssize_t len = static_cast<Py_ssize_t>(storage.nbytes());
    // 解包切片对象，获取 start, stop, step
    if (PySlice_Unpack(index, &start, &stop, &step) < 0) {
      // 解包失败时返回错误码 -1
      return -1;
    }
    // 调整切片的起始、终止位置，确保不超出边界
    PySlice_AdjustIndices(len, &start, &stop, step);
    // 如果步长不为 1，报错
    if (step != 1) {
      TORCH_CHECK(
          "Trying to slice with a step of ",
          step,
          ", but only a step of "
          "1 is supported");
      return 0;
    }
    // TODO: check the bounds only once
    // TODO: fill?
    // 遍历切片范围内的索引，将每个位置设置为 rvalue
    for (; start < stop; start++)
      storage_set(storage, start, rvalue);
    // 返回成功状态码 0
    return 0;
  }
  // 如果 index 类型无法处理，报错
  TORCH_CHECK(
      "can't index a " THPStorageStr " with ", THPUtils_typename(index));
  // 返回错误码 -1
  return -1;
  // 处理 Torch 错误并返回错误码 -1
  END_HANDLE_TH_ERRORS_RET(-1)
}

// 定义静态的 PyMappingMethods 结构体，用于映射操作
static PyMappingMethods THPStorage_mappingmethods = {
    (lenfunc)THPStorage_length,    // 获取对象长度的函数指针
    (binaryfunc)THPStorage_get,    // 获取映射值的函数指针
    (objobjargproc)THPStorage_set  // 设置映射值的函数指针
};

// 定义 THPStorageMeta 结构体，继承自 PyHeapTypeObject
struct THPStorageMeta {
  PyHeapTypeObject base;  // 基础类型为 PyHeapTypeObject
};

// 初始化函数声明，用于 THPStorageMetaType
int THPStorageMetaType_init(PyObject* cls, PyObject* args, PyObject* kwargs);

// 定义 PyTypeObject 结构体 THPStorageMetaType
PyTypeObject THPStorageMetaType = {
    PyVarObject_HEAD_INIT(
        DEFERRED_ADDRESS(&PyType_Type),  // 变量对象头初始化，延迟地址解析
        0) "torch._C._StorageMeta",  /* tp_name */  // 类型名称
    sizeof(THPStorageMeta),  /* tp_basicsize */  // 基本大小
    0,  /* tp_itemsize */  // 每个项大小
    nullptr,  /* tp_dealloc */  // 释放内存的函数指针
    0,  /* tp_vectorcall_offset */  // 矢量调用偏移量
    nullptr,  /* tp_getattr */  // 获取属性的函数指针
    nullptr,  /* tp_setattr */  // 设置属性的函数指针
    nullptr,  /* tp_reserved */  // 保留字段
    nullptr,  /* tp_repr */  // 获取对象字符串表示形式的函数指针
    nullptr,  /* tp_as_number */  // 数字类型协议
    nullptr,  /* tp_as_sequence */  // 序列类型协议
    nullptr,  /* tp_as_mapping */  // 映射类型协议
    nullptr,  /* tp_hash  */  // 哈希函数指针
    nullptr,  /* tp_call */  // 调用函数指针
    nullptr,  /* tp_str */  // 转换为字符串的函数指针
    nullptr,  /* tp_getattro */  // 获取属性的函数指针
    nullptr,  /* tp_setattro */  // 设置属性的函数指针
    nullptr,  /* tp_as_buffer */  // 缓冲区协议
    // NOLINTNEXTLINE(misc-redundant-expression)
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,  /* tp_flags */  // 类型标志
    nullptr,  /* tp_doc */  // 文档字符串
    nullptr,  /* tp_traverse */  // 遍历对象的函数指针
    nullptr,  /* tp_clear */  // 清除对象的函数指针
    nullptr,  /* tp_richcompare */  // 富比较函数指针
    0,  /* tp_weaklistoffset */  // 弱引用列表偏移量
    nullptr,  /* tp_iter */  // 迭代器协议
    nullptr,  /* tp_iternext */  // 迭代器的下一个元素
    nullptr,  /* tp_methods */  // 方法
    nullptr,  /* tp_members */  // 成员
    nullptr,  /* tp_getset */  // 获取和设置方法
    DEFERRED_ADDRESS(&PyType_Type),  /* tp_base */  // 基类类型
    nullptr,  /* tp_dict */  // 字典
    nullptr,  /* tp_descr_get */  // 获取描述符的函数指针
    nullptr,  /* tp_descr_set */  // 设置描述符的函数指针
    0,  /* tp_dictoffset */  // 字典偏移量
    THPStorageMetaType_init,  /* tp_init */  // 初始化函数指针
    nullptr,  /* tp_alloc */  // 分配函数指针
    nullptr  /* tp_new */  // 新建函数指针
};

// TODO: implement equality
// 定义 PyTypeObject 结构体 THPStorageType，表示 torch._C.StorageBase 类型
PyTypeObject THPStorageType = {
    PyVarObject_HEAD_INIT(
        &THPStorageMetaType,  // 变量对象头初始化，指向 THPStorageMetaType
        0) "torch._C.StorageBase",  /* tp_name */  // 类型名称
    sizeof(THPStorage),  /* tp_basicsize */  // 基本大小
    0,  /* tp_itemsize */  // 每个项大小
    nullptr,  /* tp_dealloc */  // 释放内存的函数指针
    0,  /* tp_vectorcall_offset */  // 矢量调用偏移量
    nullptr,  /* tp_getattr */  // 获取属性的函数指针
    nullptr,  /* tp_setattr */  // 设置属性的函数指针
    nullptr,  /* tp_reserved */  // 保留字段
    nullptr,  /* tp_repr */  // 获取对象字符串表示形式的函数指针
    nullptr,  /* tp_as_number */  // 数字类型协议
    nullptr,  /* tp_as_sequence */  // 序列类型协议
    &THPStorage_mappingmethods,  /* tp_as_mapping */  // 映射类型协议
    nullptr,  /* tp_hash  */  // 哈希函数指针
    nullptr,  /* tp_call */  // 调用函数指针
    nullptr,  /* tp_str */  // 转换为字符串的函数指针
    nullptr,  /* tp_getattro */  // 获取属性的函数指针
    nullptr,  /* tp_setattro */  // 设置属性的函数指针
    nullptr,  /* tp_as_buffer */  // 缓冲区协议
    // NOLINTNEXTLINE(misc-redundant-expression)
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,  /* tp_flags */  // 类型标志
    nullptr,  /* tp_doc */  // 文档字符串
    nullptr,  /* tp_traverse */  // 遍历对象的函数指针
    nullptr,  /* tp_clear */  // 清除对象的函数指针
    nullptr,  /* tp_richcompare */  // 富比较函数指针
    0,  /* tp_weaklistoffset */  // 弱引用列表偏移量
    nullptr,  /* tp_iter */  // 迭代器协议
    nullptr,  /* tp_iternext */  // 迭代器的下一个元素
    nullptr,  /* tp_methods */  // 方法
    nullptr,  /* tp_members */  // 成员
    nullptr,  /* tp_getset */  // 获取和设置方法
    nullptr,  /* tp_base */  // 基类类型
    nullptr,  /* tp_dict */  // 字典
    nullptr,  /* tp_descr_get */  // 获取描述符的函数指针
    nullptr,  /* tp_descr_set */  // 设置描述符的函数指针
    0  /* tp_dictoffset */  // 字典偏移量
};
    nullptr, /* tp_init */
    // 指向空的初始化函数指针，通常表示对象无需特定的初始化操作
    nullptr, /* tp_alloc */
    // 指向空的内存分配函数指针，表明对象的内存分配由Python解释器的默认方式处理
    THPStorage_pynew, /* tp_new */
    // 指向THPStorage_pynew函数的指针，用于创建新的对象实例
};

// 初始化 THPStorageMetaType 类型对象，设置其析构函数为 THPStorage_subclass_dealloc
int THPStorageMetaType_init(PyObject* cls, PyObject* args, PyObject* kwargs) {
  // 调用 PyType_Type 的初始化方法，如果失败返回 -1
  if (PyType_Type.tp_init(cls, args, kwargs) < 0) {
    return -1;
  }
  // 将 cls 对应的类型对象的析构函数设置为 THPStorage_subclass_dealloc
  ((PyTypeObject*)cls)->tp_dealloc = (destructor)THPStorage_subclass_dealloc;
  return 0;
}

// 返回 THPStorage 对象的设备信息作为 PyObject*
static PyObject* THPStorage_device(THPStorage* self, void* unused) {
  HANDLE_TH_ERRORS
  // 检查 self 是否为非空
  THPStorage_assertNotNull(self);
  // 返回 THPDevice_New 创建的设备信息对象
  return THPDevice_New(THPStorage_Unpack(self).device());
  END_HANDLE_TH_ERRORS
}

// 返回 THPStorage 对象的 _cdata 属性作为 PyLongObject*
PyObject* THPStorage_get_cdata(THPStorage* self, void* unused) {
  HANDLE_TH_ERRORS
  // 返回包装了 THPStorage_Unpack(self).unsafeGetStorageImpl() 的 PyLongObject*
  return PyLong_FromVoidPtr(THPStorage_Unpack(self).unsafeGetStorageImpl());
  END_HANDLE_TH_ERRORS
}

// 定义一个 getter 函数指针类型
typedef PyObject* (*getter)(PyObject*, void*);

// 定义 THPStorage_properties 结构体数组，描述 THPStorage 类的属性
static struct PyGetSetDef THPStorage_properties[] = {
    {"device", (getter)THPStorage_device, nullptr, nullptr, nullptr},
    {"_cdata", (getter)THPStorage_get_cdata, nullptr, nullptr, nullptr},
    {nullptr}};

// 初始化 THPStorage 类型相关的方法和属性
bool THPStorage_init(PyObject* module) {
  static std::vector<PyMethodDef> methods;
  // 向 methods 添加 THPStorage_getMethods() 和 THPStorage_getSharingMethods() 中定义的方法
  THPUtils_addPyMethodDefs(methods, THPStorage_getMethods());
  THPUtils_addPyMethodDefs(methods, THPStorage_getSharingMethods());

  // 设置 THPStorageMetaType 的基类为 PyType_Type，并完成类型的初始化
  THPStorageMetaType.tp_base = &PyType_Type;
  if (PyType_Ready(&THPStorageMetaType) < 0)
    return false;
  // 增加对 THPStorageMetaType 类型的引用计数，并将其添加到 module 中
  Py_INCREF(&THPStorageMetaType);
  PyModule_AddObject(module, "_StorageMeta", (PyObject*)&THPStorageMetaType);

  // 设置 THPStorageType 的方法和属性，并完成类型的初始化
  THPStorageType.tp_methods = methods.data();
  THPStorageType.tp_getset = THPStorage_properties;
  if (PyType_Ready(&THPStorageType) < 0)
    return false;
  // 增加对 THPStorageType 类型的引用计数，并将其添加到 module 中
  Py_INCREF(&THPStorageType);
  PyModule_AddObject(module, "StorageBase", (PyObject*)&THPStorageType);
  return true;
}

// 完成 THPStorage 类型的初始化后执行的后续操作
void THPStorage_postInit(PyObject* module) {
  // 获取 module 中名称为 "UntypedStorage" 的属性，并将其转换为 PyTypeObject*
  THPStorageClass =
      (PyTypeObject*)PyObject_GetAttrString(module, "UntypedStorage");
  // 如果获取失败，则抛出 python_error 异常
  if (!THPStorageClass)
    throw python_error();
}

// 确保 THPStorage 对象不为空，否则抛出 TORCH_CHECK 异常
void THPStorage_assertNotNull(THPStorage* storage) {
  TORCH_CHECK(
      THPStorage_Unpack(storage).unsafeGetStorageImpl(), "Got a null Storage");
}

// 确保 PyObject* 类型的对象不为空，调用 THPStorage_assertNotNull(THPStorage*) 方法
void THPStorage_assertNotNull(PyObject* obj) {
  THPStorage_assertNotNull((THPStorage*)obj);
}
```