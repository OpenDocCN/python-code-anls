# `.\pytorch\torch\csrc\jit\serialization\pickler.cpp`

```py
// 包含 ATen 库中的基本头文件
#include <ATen/ATen.h>
// 包含 ATen 库中的字典数据结构定义
#include <ATen/core/Dict.h>
// 根据编译配置，包含分布式 RPC 相关头文件
#ifdef USE_RPC
#include <torch/csrc/distributed/rpc/rref_context.h>
#endif
// 包含 ATen 库中的量化器定义
#include <ATen/quantized/Quantizer.h>
// 包含 C10 库中的工具函数，用于生成范围迭代器
#include <c10/util/irange.h>
// 包含 Torch 序列化与反序列化功能的相关头文件
#include <torch/csrc/jit/api/function_impl.h>
#include <torch/csrc/jit/serialization/pickler.h>
// 包含 Torch 中的字节顺序处理功能
#include <torch/csrc/utils/byte_order.h>
// 包含标准字符串库
#include <string>
// 包含 C++ 类型特性支持
#include <type_traits>

// 命名空间 torch::jit 中使用 IValue 类
namespace torch::jit {

// 使用 c10 中的 IValue 类
using ::c10::IValue;

// 定义 Pickler 类的常量协议版本为 2
constexpr static uint8_t PROTOCOL_VERSION = 2;

// 析构函数，用于 Pickler 类对象的销毁
// NOLINTNEXTLINE(bugprone-exception-escape)
Pickler::~Pickler() {
  flush();
}

// 设置 Pickler 对象的协议版本
void Pickler::protocol() {
  // 将协议操作码和协议版本压入 Pickler 对象的操作栈中
  push<PickleOpCode>(PickleOpCode::PROTO);
  push<uint8_t>(PROTOCOL_VERSION);
}

// 开始创建元组操作，将 MARK 操作码压入操作栈中
void Pickler::startTuple() {
  // 所有属性都被推送到元组中，并且它们的索引保存在模块定义中
  push<PickleOpCode>(PickleOpCode::MARK);
}

// 结束创建元组操作，将 TUPLE 操作码压入操作栈中
void Pickler::endTuple() {
  push<PickleOpCode>(PickleOpCode::TUPLE);
}

// 结束 Pickler 对象的序列化，将 STOP 操作码压入操作栈中并刷新缓冲区
void Pickler::stop() {
  push<PickleOpCode>(PickleOpCode::STOP);
  flush();
}

// 未缓存版本的 pushIValue 函数，被 pushIValue 调用
void Pickler::pushIValueImpl(const IValue& ivalue) {
  // 根据 IValue 的类型，分别执行相应的序列化操作
  if (ivalue.isTensor()) {
    pushTensor(ivalue);
  } else if (ivalue.isTuple()) {
    pushTuple(ivalue);
  } else if (ivalue.isDouble()) {
    pushDouble(ivalue.toDouble());
  } else if (ivalue.isComplexDouble()) {
    pushComplexDouble(ivalue);
  } else if (ivalue.isInt()) {
    pushInt(ivalue.toInt());
  } else if (ivalue.isBool()) {
    pushBool(ivalue.toBool());
  } else if (ivalue.isString()) {
    pushString(ivalue.toStringRef());
  } else if (ivalue.isGenericDict()) {
    pushDict(ivalue);
  } else if (ivalue.isNone()) {
    push<PickleOpCode>(PickleOpCode::NONE);
  } else if (ivalue.isIntList()) {
    // 处理整数列表类型的序列化操作
    pushSpecializedList(ivalue, "build_intlist", [this](const IValue& ivalue) {
      for (const int64_t item : ivalue.toIntVector()) {
        pushInt(item);
      }
    });
  } else if (ivalue.isTensorList()) {
    // 处理张量列表类型的序列化操作
    pushSpecializedList(
        ivalue, "build_tensorlist", [this](const IValue& ivalue) {
          for (const at::Tensor& item : ivalue.toTensorVector()) {
            pushIValue(item);
          }
        });
  } else if (ivalue.isDoubleList()) {
    // 处理双精度浮点数列表类型的序列化操作
    pushSpecializedList(
        ivalue, "build_doublelist", [this](const IValue& ivalue) {
          for (double item : ivalue.toDoubleVector()) {
            pushDouble(item);
          }
        });
  } else if (ivalue.isBoolList()) {
    // 处理布尔值列表类型的序列化操作
    pushSpecializedList(ivalue, "build_boollist", [this](const IValue& ivalue) {
      for (bool item : ivalue.toBoolList()) {
        pushBool(item);
      }
    });
    // 注意：必须在 isIntList 等之后检查 isList，因为 isList 对所有列表类型为真
  } else if (ivalue.isList()) {
    // 处理通用列表类型的序列化操作
    pushGenericList(ivalue);
  } else if (ivalue.isObject()) {
    // 处理对象类型的序列化操作
    auto obj = ivalue.toObject();
    auto type = obj->type();
    if (memoized_class_types_ != nullptr) {
      // 如果 memoized_class_types_ 不为空指针，则执行以下操作
      // 将 Pickler 遇到的每种类类型进行记忆化
      // 这用于确保我们捕捉所有的运行时类型，并正确地序列化它们以支持类/接口的多态性
      memoized_class_types_->emplace_back(type);
    }
    // 获取类型的名称作为字符串
    auto type_name = type->name().value();
    // 如果定义了类型重命名器，则使用它来重命名类型
    if (type_renamer_) {
      type_name = type_renamer_(type);
    }
    // 将类型名称推送到全局堆栈
    pushGlobal(type_name.prefix(), type_name.name());
    // 推送空元组操作码
    push<PickleOpCode>(PickleOpCode::EMPTY_TUPLE);
    // 推送新对象操作码
    push<PickleOpCode>(PickleOpCode::NEWOBJ);
    // 检查类型是否有有效的 __getstate__ 方法，并获取其状态
    if (checkHasValidSetGetState(type)) {
      // 获取 __getstate__ 方法并推送其返回值到堆栈
      Function& getstate = type->getMethod("__getstate__");
      pushIValue(getstate({obj}));
    } else {
      // 如果没有有效的 __getstate__ 方法，则推送空字典操作码和标记操作码
      push<PickleOpCode>(PickleOpCode::EMPTY_DICT);
      push<PickleOpCode>(PickleOpCode::MARK);
      // 遍历类型的属性，并推送属性名称和对应的值到堆栈
      for (size_t i = 0, n = type->numAttributes(); i < n; ++i) {
        pushString(type->getAttributeName(i));
        pushIValue(obj->getSlot(i));
      }
      // 推送设置项目操作码
      push<PickleOpCode>(PickleOpCode::SETITEMS);
    }
    // 推送构建操作码
    push<PickleOpCode>(PickleOpCode::BUILD);
  } else if (ivalue.isDevice()) {
    // 如果值是设备类型，则推送设备操作
    pushDevice(ivalue);
  } else if (ivalue.isCapsule()) {
    // 如果值是胶囊类型，则生成序列化错误信息
    std::stringstream err;
    err << "Cannot serialize custom bound C++ class";
    // 如果 memoized_class_types_ 不为空且不为空集合，则获取最后一个类类型的完全限定名
    if (memoized_class_types_ && !memoized_class_types_->empty()) {
      if (auto qualname = memoized_class_types_->back()->name()) {
        err << " " << qualname->qualifiedName();
      }
    }
    err << ". Please define serialization methods via def_pickle() for "
           "this class.";
    // 抛出带有错误信息的异常
    AT_ERROR(err.str());
  } else if (ivalue.isRRef()) {
#ifdef USE_RPC
    // 如果定义了 USE_RPC 宏，则执行以下内容
    TORCH_CHECK(
        torch::distributed::rpc::getAllowJitRRefPickle() == true,
        "RRef jit pickling is only allowed inside RPC calls.");
    // 检查是否允许在 RPC 调用内进行 RRef jit pickling
    pushRRef(ivalue);
    // 调用 pushRRef 函数，将 ivalue 推送到 RRef
#else
    // 如果未定义 USE_RPC 宏，则执行以下内容
    TORCH_CHECK(
        false, "RRef pickling is only supported with the distributed package");
    // 抛出错误，因为 RRef pickling 只能在分布式包中支持
#endif
  } else if (ivalue.isEnum()) {
    // 如果 ivalue 是枚举类型
    auto enum_holder = ivalue.toEnumHolder();
    const auto& qualified_class_name =
        enum_holder->type()->qualifiedClassName();
    // 获取枚举类型的完全限定类名
    pushGlobal(qualified_class_name.prefix(), qualified_class_name.name());
    // 将完全限定类名的前缀和名称推送到全局堆栈
    pushIValue(enum_holder->value());
    // 推送枚举值的 IValue
    push<PickleOpCode>(PickleOpCode::REDUCE);
    // 推送 REDUCE 操作码
  } else {
    // 对于未知的 IValue 类型，抛出错误
    AT_ERROR("Unknown IValue type for pickling: ", ivalue.tagKind());
  }
}

void Pickler::pushDevice(const IValue& ivalue) {
  // 推送设备信息到 Pickler
  auto device = ivalue.toDevice();
  auto deviceStr = device.str();
  auto it = memoized_devices_map_.find(deviceStr);
  if (it == memoized_devices_map_.end()) {
    // 如果设备信息未被 memoize 过
    pushGlobal("torch", "device");
    // 推送 torch.device 全局对象
    pushString(deviceStr);
    // 推送设备字符串
    push<PickleOpCode>(PickleOpCode::TUPLE1);
    // 推送 TUPLE1 操作码
    push<PickleOpCode>(PickleOpCode::REDUCE);
    // 推送 REDUCE 操作码
    memoized_devices_map_[deviceStr] = pushNextBinPut();
    // 将设备信息的索引加入 memoized_devices_map_
  } else {
    // 如果设备信息已经 memoize 过
    pushBinGet(it->second);
    // 使用之前 memoize 的索引推送设备信息
  }
}

#ifdef USE_RPC
void Pickler::pushRRef(const IValue& ivalue) {
  // 如果定义了 USE_RPC 宏，则执行以下内容
  // 这里的注释文本只是一个例子，并没有实际的注释内容
  auto rrefInterface = ivalue.toRRef();
  auto rref =
      c10::static_intrusive_pointer_cast<distributed::rpc::RRef>(rrefInterface);
  pushGlobal("torch.distributed.rpc", "rref");
  auto& ctx = distributed::rpc::RRefContext::getInstance();
  auto rrefForkData = ctx.prepareChildFork(rref);
  push<PickleOpCode>(PickleOpCode::MARK);
  pushInt(rrefForkData.ownerId_);
  pushInt(rrefForkData.rrefId_.createdOn_);
  pushInt(rrefForkData.rrefId_.localId_);
  pushInt(rrefForkData.forkId_.createdOn_);
  pushInt(rrefForkData.forkId_.localId_);
  pushInt(rrefForkData.parent_);
  pushString(rrefForkData.typeStr_);
  push<PickleOpCode>(PickleOpCode::TUPLE);
  push<PickleOpCode>(PickleOpCode::REDUCE);
}
#endif

void Pickler::pushIValue(const IValue& ivalue) {
  // 推送 IValue 到 Pickler
  bool shouldMemoizeByPointer =
      ivalue.isPtrType() && !ivalue.isString() && ivalue.use_count() > 1;
  // 决定是否通过指针进行 memoize

  // Mutable 类型的 IValue 根据指针相等进行 memoize，不可变类型则在 pushIValueImpl 内部处理
  if (shouldMemoizeByPointer) {
    const void* ptr = ivalue.internalToPointer();
    // 获取指向内部数据的指针
    TORCH_CHECK(
        ptr != nullptr,
        "Pickler cannot memoize ",
        ivalue.tagKind(),
        " IValue ",
        ivalue);
    // 如果指针为空，则抛出错误
    auto memo_entry = memoized_ivalue_map_.find(ptr);
    // 查找 memoized_ivalue_map_ 中是否已经有该指针的 memoize 记录
    if (memo_entry != memoized_ivalue_map_.end()) {
      // 如果找到了已有的记录，则通过 BINGET 操作获取该值
      pushBinGet(memo_entry->second);
      return;
    }

    pushIValueImpl(ivalue);
    // 否则，调用 pushIValueImpl 推送该 IValue

    memoized_ivalues_.push_back(ivalue);
    // 将该 IValue 添加到 memoized_ivalues_ 中
    memoized_ivalue_map_[ptr] = pushNextBinPut();
    // 记录该 IValue 的索引到 memoized_ivalue_map_
  } else {
    // 如果不通过指针进行 memoize
    pushIValueImpl(ivalue);


    # 调用名为 pushIValueImpl 的函数，传入参数 ivalue
}

// 将整数压入序列化流中
void Pickler::pushInt(int64_t n) {
  // 如果整数可以用一个字节表示
  if (n >= std::numeric_limits<uint8_t>::min() &&
      n <= std::numeric_limits<uint8_t>::max()) {
    // 压入操作码 BININT1
    push<PickleOpCode>(PickleOpCode::BININT1);
    // 压入一个字节的整数值
    push<uint8_t>(n);
  } else if (
      // 如果整数可以用两个字节表示
      n >= std::numeric_limits<uint16_t>::min() &&
      n <= std::numeric_limits<uint16_t>::max()) {
    // 压入操作码 BININT2
    push<PickleOpCode>(PickleOpCode::BININT2);
    // 压入两个字节的整数值（小端序）
    push<uint16_t>(to_le16(n));
  } else if (
      // 如果整数可以用四个字节表示
      n >= std::numeric_limits<int32_t>::min() &&
      n <= std::numeric_limits<int32_t>::max()) {
    // 压入操作码 BININT
    push<PickleOpCode>(PickleOpCode::BININT);
    // 压入四个字节的整数值（小端序）
    push<int32_t>(to_le32(n));
  } else {
    // 否则，压入 8 字节整数
    // Push 8 byte integer
    push<PickleOpCode>(PickleOpCode::LONG1);
    // 压入操作码 LONG1
    push<uint8_t>(8);
    // 压入八字节的整数值（小端序）
    push<int64_t>(to_le64(n));
  }
}

// 将布尔值压入序列化流中
void Pickler::pushBool(bool value) {
  // 根据布尔值压入操作码 NEWTRUE 或 NEWFALSE
  push<PickleOpCode>(value ? PickleOpCode::NEWTRUE : PickleOpCode::NEWFALSE);
}

// 将二进制数据索引压入序列化流中
void Pickler::pushBinGet(uint32_t memo_id) {
  // 如果 memo_id 在一个字节内可以表示
  if (memo_id <= std::numeric_limits<uint8_t>::max()) {
    // 压入操作码 BINGET
    push<PickleOpCode>(PickleOpCode::BINGET);
    // 压入一个字节的索引值
    push<uint8_t>(memo_id);
  } else {
    // 否则，索引值过大，使用 LONG_BINGET 操作码
    // Memoized too many items, issue a LONG_BINGET instead
    push<PickleOpCode>(PickleOpCode::LONG_BINGET);
    // 压入四字节的索引值（小端序）
    push<uint32_t>(memo_id);
  }
}

// 将字符串压入序列化流中（未记忆化）
// 如果字符串长度超过 UINT_MAX，则使用 BINUNICODE8 操作码
void Pickler::pushStringImpl(const std::string& string) {
  if (string.size() <= UINT_MAX) {
    // 压入操作码 BINUNICODE
    push<PickleOpCode>(PickleOpCode::BINUNICODE);
    // 压入四字节的字符串长度（小端序）
    push<uint32_t>(to_le32(string.size()));
    // 压入字符串内容
    pushBytes(string);
  } else {
    // 否则，使用 BINUNICODE8 操作码
    push<PickleOpCode>(PickleOpCode::BINUNICODE8);
    // 压入八字节的字符串长度（小端序）
    push<int64_t>(to_le64(string.size()));
    // 压入字符串内容
    pushBytes(string);
  }
}

// 将字符串压入序列化流中
void Pickler::pushString(const std::string& string) {
  // 查找字符串是否已记忆化
  auto it = memoized_strings_map_.find(string);
  if (it == memoized_strings_map_.end()) {
    // 如果字符串未记忆化，则执行字符串压入
    pushStringImpl(string);
    // 记录字符串并返回索引
    memoized_strings_map_[string] = pushNextBinPut();
  } else {
    // 如果字符串已记忆化，则压入索引值
    pushBinGet(it->second);
  }
}

// 将 Tensor 的存储压入序列化流中
void Pickler::pushStorageOfTensor(const at::Tensor& tensor) {
  const at::Storage& storage = tensor.storage();
  void* addr = storage.unsafeGetStorageImpl();
  // 查找存储是否已记忆化
  auto it = memoized_storage_map_.find(addr);
  if (it != memoized_storage_map_.end()) {
    // 如果存储已记忆化，则压入存储索引值
    pushBinGet(it->second);
    return;
  }

  // 创建持久加载的元组
  push<PickleOpCode>(PickleOpCode::MARK);
  // 类型名称
  pushString("storage");
  // 数据类型
  std::string data_type =
      std::string(toString(tensor.scalar_type())).append("Storage");
  pushGlobal("torch", data_type);
  // 根键
  std::string root_key = get_tensor_id_ != nullptr
      ? get_tensor_id_(tensor)
      : std::to_string(tensor_data_.size());
  pushString(root_key);
  // 位置
  pushString(tensor.device().str());
  // 大小
  pushInt(tensor.storage().nbytes() / tensor.element_size());

  // 元组结束标记
  push<PickleOpCode>(PickleOpCode::TUPLE);
  // BINPERSID 操作码
  push<PickleOpCode>(PickleOpCode::BINPERSID);

  // 如果不需要写入 Tensor，则跳过这部分
  memoized_storage_map_[addr] = pushNextBinPut();
  tensor_data_.push_back(tensor);
}
// 将字符串数据压入缓冲区，如果字符串长度小于等于 kSmallStr 并且加上当前缓冲区位置不超过缓冲区大小，则直接缓存数据
void Pickler::pushBytes(const std::string& string) {
  static const size_t kSmallStr = 32;
  if (string.size() <= kSmallStr &&
      bufferPos_ + string.size() <= buffer_.size()) {
    // Small string that fits: buffer the data.
    memcpy(buffer_.data() + bufferPos_, string.data(), string.size());
    bufferPos_ += string.size();
  } else {
    // Otherwise, first flush, then write directly.
    flush(); // 刷新缓冲区
    writer_(string.data(), string.size()); // 写入数据到输出流
  }
}

// 将全局变量的模块名和类名拼接成字符串键，并根据键查找或者写入 memoized_globals_map_ 中的全局变量
void Pickler::pushGlobal(
    c10::string_view module_name,
    c10::string_view class_name) {
  std::string key;
  key.reserve(module_name.size() + class_name.size() + 2);
  key.append(module_name.data(), module_name.size());
  key.push_back('\n');
  key.append(class_name.data(), class_name.size());
  key.push_back('\n');

  const auto memo_entry = memoized_globals_map_.find(key); // 在 memoized_globals_map_ 中查找键值对
  if (memo_entry == memoized_globals_map_.end()) {
    // 如果键不存在，将 GLOBAL 操作码推入堆栈
    push<PickleOpCode>(PickleOpCode::GLOBAL);
    // 将键值对应的字符串数据推入缓冲区
    pushBytes(key);
    // 推入下一个可用的 BINPUT 操作码，并将其映射到 memoized_globals_map_ 中
    size_t memo_id = pushNextBinPut();
    memoized_globals_map_.insert({key, memo_id});
  } else {
    // 如果键已存在，推入对应的 BINGET 操作码
    pushBinGet(memo_entry->second);
  }
}

// 将张量数据推入序列化数据流
void Pickler::pushTensor(const IValue& ivalue) {
  if (tensor_table_ == nullptr) {
    // 如果张量表为空，推入字面量张量数据
    pushLiteralTensor(ivalue);
  } else {
    // 否则，推入张量的引用
    pushTensorReference(ivalue);
  }
}

// 将稀疏张量的字面量数据推入序列化数据流
void Pickler::pushLiteralSparseTensor(const at::Tensor& tensor) {
  // 推入 torch._utils._rebuild_sparse_tensor 全局函数名
  pushGlobal("torch._utils", "_rebuild_sparse_tensor");
  // 推入 MARK 操作码
  push<PickleOpCode>(PickleOpCode::MARK);
  // 推入布局类型的整数表示
  auto layout = static_cast<int>(tensor.layout());
  pushInt(layout);
  switch (layout) {
    case static_cast<int>(c10::Layout::Sparse):
      // 如果布局是稀疏格式，推入 MARK 操作码并推入尺寸信息
      push<PickleOpCode>(PickleOpCode::MARK);
      for (auto size : tensor.sizes()) {
        pushInt(size);
      }
      push<PickleOpCode>(PickleOpCode::TUPLE);
      // 推入是否需要梯度的 IValue
      pushIValue(tensor.requires_grad());
      // 推入稀疏张量的索引数据
      pushTensor(tensor._indices());
      // 推入稀疏张量的值数据
      pushTensor(tensor._values());
      break;
    case static_cast<int>(c10::Layout::SparseCsr):
      // 如果布局是 CSR 稀疏格式，推入 MARK 操作码并推入尺寸信息
      push<PickleOpCode>(PickleOpCode::MARK);
      for (auto size : tensor.sizes()) {
        pushInt(size);
      }
      push<PickleOpCode>(PickleOpCode::TUPLE);
      // 推入是否需要梯度的 IValue
      pushIValue(tensor.requires_grad());
      // 推入 CSR 稀疏张量的行索引
      pushTensor(tensor.crow_indices());
      // 推入 CSR 稀疏张量的列索引
      pushTensor(tensor.col_indices());
      // 推入 CSR 稀疏张量的值数据
      pushTensor(tensor.values());
      break;
    default:
      // 如果布局类型不支持，抛出异常
      TORCH_CHECK(
          false,
          "Unsupported sparse tensor layout type in serialization ",
          static_cast<c10::Layout>(layout));
      break;
  }
  // 推入 collections.OrderedDict 的全局函数名和 EMPTY_TUPLE 操作码
  pushGlobal("collections", "OrderedDict");
  push<PickleOpCode>(PickleOpCode::EMPTY_TUPLE);
  // 构建 collections.OrderedDict 用于反向钩子
  push<PickleOpCode>(PickleOpCode::REDUCE);
  push<PickleOpCode>(PickleOpCode::TUPLE);
  // 调用 torch._utils._rebuild_sparse_coo_tensor 函数
  push<PickleOpCode>(PickleOpCode::REDUCE);
}
// 将字面量张量压入 Pickler 中
void Pickler::pushLiteralTensor(const IValue& ivalue) {
  // 与张量引用相比，字面量张量包含在 pickle 程序的二进制数据块中。
  // 它们在 STOP 操作码之后写入文件。由于字节字符串限制为4GB，不能直接包含在 pickle 程序中，否则需要大量额外的机制。
  //
  // 这里使用的格式与 `torch.save()` 中使用的格式相同。格式的代码可以在 `torch/serialization.py` 中找到。

  auto& tensor = ivalue.toTensor();

  if (tensor.is_sparse() || tensor.is_sparse_csr()) {
    // 如果张量是稀疏张量或CSR格式的稀疏张量，调用 pushLiteralSparseTensor 处理
    pushLiteralSparseTensor(tensor);
    return;
  }

  bool quantized = tensor.is_quantized();
  // 函数的参数包括:
  //    storage, storage_offset, size, stride, requires_grad, backward_hooks
  // 根据是否量化，推送全局函数名 "torch._utils._rebuild_qtensor" 或者 "torch._utils._rebuild_tensor_v2"
  pushGlobal(
      "torch._utils", quantized ? "_rebuild_qtensor" : "_rebuild_tensor_v2");

  // 压入 PICKLE 操作码 MARK
  push<PickleOpCode>(PickleOpCode::MARK);
  // 压入张量的存储信息
  pushStorageOfTensor(tensor);

  // 压入张量的存储偏移
  pushInt(tensor.storage_offset());

  // 压入张量的尺寸信息
  push<PickleOpCode>(PickleOpCode::MARK);
  for (auto size : tensor.sizes()) {
    pushInt(size);
  }
  push<PickleOpCode>(PickleOpCode::TUPLE);

  // 压入张量的步幅信息
  push<PickleOpCode>(PickleOpCode::MARK);
  for (auto stride : tensor.strides()) {
    pushInt(stride);
  }
  push<PickleOpCode>(PickleOpCode::TUPLE);

  if (quantized) {
    // 如果张量是量化的，压入相关信息：量化方案、缩放因子、零点等
    push<PickleOpCode>(PickleOpCode::MARK);
    pushGlobal("torch", toString(tensor.qscheme()));
    // 根据量化方案的不同，压入不同的数据信息
    switch (tensor.qscheme()) {
      case at::kPerTensorAffine:
        pushDouble(tensor.q_scale());
        pushInt(tensor.q_zero_point());
        break;
      case at::kPerChannelAffineFloatQParams:
      case at::kPerChannelAffine: {
        pushTensor(tensor.q_per_channel_scales());
        pushTensor(tensor.q_per_channel_zero_points());
        pushInt(tensor.q_per_channel_axis());
      } break;
      default:
        // 如果遇到不支持的量化类型，抛出错误
        TORCH_CHECK(
            false,
            "Unsupported tensor quantization type in serialization ",
            toString(tensor.qscheme()));
        break;
    }
    push<PickleOpCode>(PickleOpCode::TUPLE);
  }

  // 压入张量的 requires_grad 属性
  pushIValue(tensor.requires_grad());

  // 压入 collections.OrderedDict
  pushGlobal("collections", "OrderedDict");
  // 压入空元组 PICKLE 操作码 EMPTY_TUPLE
  push<PickleOpCode>(PickleOpCode::EMPTY_TUPLE);
  // 构建 collections.OrderedDict 用于 backward_hooks
  push<PickleOpCode>(PickleOpCode::REDUCE);

  if (!quantized) {
    // 如果张量不是量化的，且 metadata 不为空，则处理 metadata
    auto metadata = torch::jit::getTensorMetadata(tensor);
    if (!metadata.empty()) {
      // 因为基于 std::unordered_map<K, V> 的 IValues 速度慢且已弃用，
      // 所以将 metadata 转换为 c10::Dict，然后压入 Pickler
      c10::Dict<std::string, bool> math_bits_;
      for (const auto& pair : metadata) {
        math_bits_.insert(pair.first, pair.second);
      }
      pushDict(math_bits_);
    }
  }



  // 这里是函数或语句块的结尾，闭合了之前的代码块
  push<PickleOpCode>(PickleOpCode::TUPLE);



  // 将 PickleOpCode::TUPLE 推入堆栈，表示在 Pickle 序列化中创建一个元组
  push<PickleOpCode>(PickleOpCode::REDUCE);
// 结束 Pickler 类中 pushSpecializedList 方法的实现

void Pickler::pushSpecializedList(
    const IValue& ivalue,
    const char* list_name,
    const std::function<void(const IValue&)>& item_pusher) {
  // 调用 pushGlobal 方法，将 torch.jit._pickle 和 list_name 推入堆栈
  pushGlobal("torch.jit._pickle", list_name);

  // 将 PickleOpCode::MARK 推入堆栈，表示序列化的起始点
  push<PickleOpCode>(PickleOpCode::MARK);

  // 将 PickleOpCode::EMPTY_LIST 推入堆栈，表示将要创建一个空列表
  push<PickleOpCode>(PickleOpCode::EMPTY_LIST);

  // 将 PickleOpCode::MARK 推入堆栈，标记列表的开始
  push<PickleOpCode>(PickleOpCode::MARK);

  // 调用 item_pusher 函数，将 ivalue 中的元素添加到列表中
  item_pusher(ivalue);

  // 将 PickleOpCode::APPENDS 推入堆栈，表示列表的结束
  push<PickleOpCode>(PickleOpCode::APPENDS);

  // 将 PickleOpCode::TUPLE 推入堆栈，表示元组的结束
  push<PickleOpCode>(PickleOpCode::TUPLE);

  // 将 PickleOpCode::REDUCE 推入堆栈，表示调用 reduce 函数
  push<PickleOpCode>(PickleOpCode::REDUCE);
}

// 定义了一个静态内联函数 swapDouble，用于交换 double 类型的字节顺序
static inline double swapDouble(double value) {
  const char* bytes = reinterpret_cast<const char*>(&value);
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  double flipped;
  char* out_bytes = reinterpret_cast<char*>(&flipped);
  for (const auto i : c10::irange(sizeof(double))) {
    out_bytes[i] = bytes[sizeof(double) - i - 1];
  }
  return *reinterpret_cast<double*>(out_bytes);
}

// 实现 Pickler 类中 pushDouble 方法
void Pickler::pushDouble(double value) {
  // 将 PickleOpCode::BINFLOAT 推入堆栈，表示将一个浮点数压入 Pickle 流
  push<PickleOpCode>(PickleOpCode::BINFLOAT);
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  // 如果是小端字节序，则交换 double 值的字节顺序
  push<double>(swapDouble(value));
#elif __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  // 如果是大端字节序，直接推入原始的 double 值
  push<double>(value);
#else
#error Unexpected or undefined __BYTE_ORDER__
#endif
}

// 实现 Pickler 类中 pushComplexDouble 方法
void Pickler::pushComplexDouble(const IValue& value) {
  // 将 value 转换为复数类型
  c10::complex<double> d = value.toComplexDouble();
  // 将全局函数 builtins.complex 推入堆栈
  pushGlobal("builtins", "complex");
  // 将实部和虚部分别推入堆栈
  pushIValue(d.real());
  pushIValue(d.imag());
  // 将 PickleOpCode::TUPLE2 推入堆栈，表示一个包含两个元素的元组
  push<PickleOpCode>(PickleOpCode::TUPLE2);
  // 将 PickleOpCode::REDUCE 推入堆栈，表示调用 reduce 函数
  push<PickleOpCode>(PickleOpCode::REDUCE);
}

// 实现 Pickler 类中 pushLong 方法
void Pickler::pushLong(const std::string& data) {
  // 获取数据的大小
  uint64_t size = data.size();

  // 断言数据大小不超过 uint8_t 的最大值
  TORCH_INTERNAL_ASSERT(
      size <= std::numeric_limits<uint8_t>::max(),
      "Cannot pickle a long larger than 255 bytes");

  // 将 PickleOpCode::LONG1 推入堆栈，表示将一个长整型压入 Pickle 流
  push<PickleOpCode>(PickleOpCode::LONG1);
  // 将数据的大小作为 uint8_t 压入堆栈
  push<uint8_t>(size);
  // 将数据内容以字节形式压入堆栈
  pushBytes(data);
}

// 实现 Pickler 类中 pushTensorReference 方法
void Pickler::pushTensorReference(const IValue& ivalue) {
  // 将 torch.jit._pickle.build_tensor_from_id 推入堆栈
  pushGlobal("torch.jit._pickle", "build_tensor_from_id");
  // 将 ivalue 转换为 Tensor，并加入 tensor_table_
  tensor_table_->push_back(ivalue.toTensor());
  // 获取当前 Tensor 在 tensor_table_ 中的索引
  int64_t tensor_id = tensor_table_->size() - 1;
  // 将 PickleOpCode::MARK 推入堆栈，表示序列化的起始点
  push<PickleOpCode>(PickleOpCode::MARK);
  // 将 tensor_id 压入堆栈
  pushIValue(tensor_id);
  // 将 PickleOpCode::TUPLE 推入堆栈，表示元组的结束
  push<PickleOpCode>(PickleOpCode::TUPLE);
  // 将 PickleOpCode::REDUCE 推入堆栈，表示调用 reduce 函数
  push<PickleOpCode>(PickleOpCode::REDUCE);
}

// startTypeTag() 和 endTypeTag() 必须成对调用，中间需要推入一个参数到堆栈
// 它们将 ivalue 的容器类型作为字符串添加到堆栈，以便我们可以在序列化过程中保留类型标签
void Pickler::startTypeTag() {
  // 如果 tag_aggregates_ 为真，则将 torch.jit._pickle.restore_type_tag 推入堆栈
  if (tag_aggregates_) {
    pushGlobal("torch.jit._pickle", "restore_type_tag");
  }
}

// 定义一个匿名命名空间中的 type_printer 函数，用于打印类型信息
namespace {
std::optional<std::string> type_printer(const c10::Type& type) {
  if (auto dyn = type.castRaw<c10::DynamicType>()) {
    return dyn->fallback()->annotation_str(type_printer);


// 返回动态类型的回退类型的注释字符串，使用给定的类型打印器
  }


  return c10::nullopt;


// 如果没有找到动态类型的注释信息，则返回空的optional对象
} // namespace

// See startTypeTag
// 结束类型标签，处理传入的 IValue 参数
void Pickler::endTypeTag(const IValue& ivalue) {
  // 如果不是聚合类型，则直接返回
  if (!tag_aggregates_) {
    return;
  }
  // 断言传入的 ivalue 是 GenericDict 或 List 类型
  TORCH_INTERNAL_ASSERT(ivalue.isGenericDict() || ivalue.isList());

  // 将 ivalue 的类型推入堆栈
  auto type = ivalue.type();
  TORCH_INTERNAL_ASSERT(type);

  // 获取类型的字符串注解并推入堆栈
  auto annot_str = type->annotation_str(type_printer);
  pushString(annot_str);

  // 将字典或列表以及类型推入元组
  push<PickleOpCode>(PickleOpCode::TUPLE2);

  // 通过 REDUCE 调用函数
  push<PickleOpCode>(PickleOpCode::REDUCE);
}

// 将传入的 ivalue 转换为字典并推入堆栈
void Pickler::pushDict(const IValue& ivalue) {
  auto dict = ivalue.toGenericDict();

  // 开始类型标签处理
  startTypeTag();

  // 推入一个空字典的操作码
  push<PickleOpCode>(PickleOpCode::EMPTY_DICT);

  // 静态断言，确保字典大小为非负数
  static_assert(
      std::is_unsigned_v<decltype(dict.size())>,
      "Expected size to be non-negative.");
  // 推入 MARK 操作码
  push<PickleOpCode>(PickleOpCode::MARK);

  // 对字典进行排序以确保键的确定性顺序
  for (const auto& entry : dict) {
    // 推入键和值
    pushIValue(entry.key());
    pushIValue(entry.value());
  }

  // 推入 SETITEMS 操作码，表示所有键值对都已推入
  push<PickleOpCode>(PickleOpCode::SETITEMS);

  // 结束类型标签处理
  endTypeTag(ivalue);
}

// 将下一个 memo_id_ 的二进制标识符推入堆栈
size_t Pickler::pushNextBinPut() {
  if (memo_id_ <= std::numeric_limits<uint8_t>::max()) {
    // 如果 memo_id_ 在 uint8_t 范围内，推入 BINPUT 操作码及其值
    push<PickleOpCode>(PickleOpCode::BINPUT);
    push<uint8_t>(memo_id_);
  } else {
    // 如果 memo_id_ 超过 uint8_t 范围，推入 LONG_BINPUT 操作码及其值
    push<PickleOpCode>(PickleOpCode::LONG_BINPUT);
    push<uint32_t>(memo_id_);
  }
  // 断言 memo_id_ 仍在 uint32_t 范围内
  AT_ASSERT(memo_id_ <= std::numeric_limits<uint32_t>::max());
  // 递增 memo_id_ 并返回它之前的值
  ++memo_id_;
  return memo_id_ - 1;
}

// 将传入的 ivalue 转换为列表并推入堆栈
void Pickler::pushGenericList(const IValue& ivalue) {
  auto list = ivalue.toListRef();
  // 开始类型标签处理
  startTypeTag();

  // 推入一个空列表的操作码
  push<PickleOpCode>(PickleOpCode::EMPTY_LIST);
  // 推入 MARK 操作码
  push<PickleOpCode>(PickleOpCode::MARK);

  // 遍历列表并推入所有元素
  for (const IValue& item : list) {
    pushIValue(item);
  }

  // 推入 APPENDS 操作码，表示所有元素都已推入
  push<PickleOpCode>(PickleOpCode::APPENDS);

  // 结束类型标签处理
  endTypeTag(ivalue);
}

// 将传入的 ivalue 转换为元组并推入堆栈
void Pickler::pushTuple(const IValue& ivalue) {
  auto tuple = ivalue.toTuple();
  auto tuple_size = tuple->elements().size();

  switch (tuple_size) {
    case 0: {
      // 空元组
      push<PickleOpCode>(PickleOpCode::EMPTY_TUPLE);
    } break;
    case 1: {
      // 单元素元组
      pushIValue(tuple->elements()[0]);
      push<PickleOpCode>(PickleOpCode::TUPLE1);
    } break;
    case 2: {
      // 两元素元组
      pushIValue(tuple->elements()[0]);
      pushIValue(tuple->elements()[1]);
      push<PickleOpCode>(PickleOpCode::TUPLE2);
    } break;
    case 3: {
      // 三元素元组
      pushIValue(tuple->elements()[0]);
      pushIValue(tuple->elements()[1]);
      pushIValue(tuple->elements()[2]);
      push<PickleOpCode>(PickleOpCode::TUPLE3);
    } break;
    default: {
      // 多于三个元素的元组
      push<PickleOpCode>(PickleOpCode::MARK);
      for (const IValue& item : tuple->elements()) {
        pushIValue(item);
      }
      push<PickleOpCode>(PickleOpCode::TUPLE);
    } break;
  }
}
    // 定义一个函数，将给定的张量数据封装成 WriteableTensorData 结构
    WriteableTensorData toWriteableTensorData(
        // 输入参数 tensor 表示需要封装的张量数据
        at::Tensor tensor,
        // 输入参数 to_cpu 表示是否需要将张量数据转移到 CPU
        bool to_cpu) {
      // 创建 WriteableTensorData 结构的实例 result
      WriteableTensorData result;
      // 将输入的张量数据赋值给 result 结构中的 tensor_ 成员变量
      result.tensor_ = tensor;
      // 计算张量数据的总字节数并赋值给 result 结构中的 size_ 成员变量
      result.size_ = tensor.storage().nbytes();
      // TODO HIP support
      // 如果张量数据不在 CPU 设备上且需要转移到 CPU
      if (tensor.storage().device_type() != DeviceType::CPU && to_cpu) {
        // 创建一个新的空张量，使用与输入张量相同的选项
        // 设置新张量的存储为输入张量的存储，存储偏移为 0，大小为整个存储的大小
        // 步长为 {1}，然后将其转移到 CPU
        result.tensor_ =
            at::empty({0}, tensor.options())
                .set_(
                    tensor.storage(),
                    /* storage_offset = */ 0,
                    /* size = */
                    {static_cast<int64_t>(
                        tensor.storage().nbytes() / tensor.element_size())},
                    /* stride = */ {1})
                .cpu();
        // 检查新创建的 CPU 张量的存储大小是否与记录的 size_ 相匹配
        TORCH_CHECK(
            result.tensor_.storage().nbytes() == result.size_,
            "Storage tensor size did not match record size");
      }
      // 返回封装后的 WriteableTensorData 结构
      return result;
    }
// 检查给定类的 __getstate__ 和 __setstate__ 方法的有效性
bool checkHasValidSetGetState(const std::shared_ptr<c10::ClassType>& cls) {
  // 查找并获取 __getstate__ 方法的 schema
  auto getstate = cls->findMethod("__getstate__");
  // 如果找不到 __getstate__ 方法，返回 false
  if (getstate == nullptr) {
    return false;
  }
  auto get_schema = getstate->getSchema();

  // 检查 __getstate__ 方法的 schema
  //   __getstate__ 方法预期为 (self) -> T
  TORCH_CHECK(
      get_schema.arguments().size() == 1,
      "'__getstate__' 的参数必须只有 'self'，但找到了 ",
      get_schema.arguments().size(),
      " 个参数");
  TORCH_CHECK(
      get_schema.returns().size() == 1,
      "'__getstate__' 必须返回一个值，但找到了 ",
      get_schema.returns().size(),
      " 个返回值");

  // 检查 __setstate__ 方法是否存在
  //   如果不存在 __setstate__ 方法，返回 false
  auto setstate = cls->findMethod("__setstate__");
  if (!setstate) {
    return false;
  }
  auto set_schema = setstate->getSchema();

  // 检查 __setstate__ 方法的 schema
  //   __setstate__ 方法预期为 (self, T) -> None
  TORCH_CHECK(
      set_schema.arguments().size() == 2,
      "'__setstate__' 必须有 'self' 和状态作为它的唯一参数，但找到了 ",
      set_schema.arguments().size(),
      " 个参数");
  TORCH_CHECK(
      set_schema.returns().size() == 1,
      "'__setstate__' 必须返回 None，但找到了 ",
      set_schema.returns().size(),
      " 个返回值");
  TORCH_CHECK(
      set_schema.returns().at(0).type()->isSubtypeOf(*NoneType::get()),
      "'__setstate__' 必须返回 None，但返回了类型为 ",
      set_schema.returns().at(0).type()->annotation_str());

  // 检查 __getstate__ 的返回类型是否与 __setstate__ 的参数类型匹配
  auto get_type = get_schema.returns().at(0).type();
  auto set_type = set_schema.arguments().at(1).type();

  TORCH_CHECK(
      get_type->isSubtypeOf(*set_type),
      "'__getstate__' 的返回类型 (",
      get_type->annotation_str(),
      ") 与 '__setstate__' 的参数类型 (",
      set_type->annotation_str(),
      ") 不匹配");

  return true;
}

// torch::jit 命名空间的结束
} // namespace torch::jit
```