# `.\pytorch\torch\csrc\jit\serialization\pickler.h`

```py
#pragma once
// 预处理指令，确保头文件只包含一次

#include <ATen/core/qualified_name.h>
// 包含 ATen 库中的 qualified_name.h，用于限定名称处理

#include <string>
// 包含标准字符串库

#include <utility>
// 包含实用工具库

#include <vector>
// 包含标准向量库

#include <ATen/Utils.h>
// 包含 ATen 库中的 Utils.h，提供各种实用工具函数

#include <ATen/core/ivalue.h>
// 包含 ATen 库中的 ivalue.h，定义 IValue 类型

#include <ATen/core/jit_type.h>
// 包含 ATen 库中的 jit_type.h，定义 JIT 类型

#include <c10/util/ArrayRef.h>
// 包含 c10 库中的 ArrayRef.h，提供 ArrayRef 类型

#include <c10/util/FbcodeMaps.h>
// 包含 c10 库中的 FbcodeMaps.h，提供 FbcodeMaps 类型

#include <c10/util/intrusive_ptr.h>
// 包含 c10 库中的 intrusive_ptr.h，提供 intrusive_ptr 类型

#include <c10/util/string_view.h>
// 包含 c10 库中的 string_view.h，提供 string_view 类型

#include <torch/csrc/Export.h>
// 包含 Torch 库中的 Export.h，定义导出相关的宏

namespace torch::jit {

// 声明命名空间 torch::jit

// Python 的 pickletools.py 中每个代码的详细描述，请参考该文件
enum class PickleOpCode : char {
  MARK = '(',
  STOP = '.',
  POP = '0',
  POP_MARK = '1',
  DUP = '2',
  FLOAT = 'F',
  INT = 'I',
  BININT = 'J',
  BININT1 = 'K',
  LONG = 'L',
  BININT2 = 'M',
  NONE = 'N',
  PERSID = 'P',
  BINPERSID = 'Q',
  REDUCE = 'R',
  STRING = 'S',
  BINSTRING = 'T',
  SHORT_BINSTRING = 'U',
  UNICODE_ = 'V',
  BINUNICODE = 'X',
  APPEND = 'a',
  BUILD = 'b',
  GLOBAL = 'c',
  DICT = 'd',
  EMPTY_DICT = '}',
  APPENDS = 'e',
  GET = 'g',
  BINGET = 'h',
  INST = 'i',
  LONG_BINGET = 'j',
  LIST = 'l',
  EMPTY_LIST = ']',
  OBJ = 'o',
  PUT = 'p',
  BINPUT = 'q',
  LONG_BINPUT = 'r',
  SETITEM = 's',
  TUPLE = 't',
  EMPTY_TUPLE = ')',
  SETITEMS = 'u',
  BINFLOAT = 'G',

  PROTO = char('\x80'),
  NEWOBJ = '\x81',
  EXT1 = '\x82',
  EXT2 = '\x83',
  EXT4 = '\x84',
  TUPLE1 = '\x85',
  TUPLE2 = '\x86',
  TUPLE3 = '\x87',
  NEWTRUE = '\x88',
  NEWFALSE = '\x89',
  LONG1 = '\x8a',
  LONG4 = '\x8b',

  BINBYTES = 'B',
  SHORT_BINBYTES = 'C',

  SHORT_BINUNICODE = char('\x8c'),
  BINUNICODE8 = '\x8d',
  BINBYTES8 = '\x8e',
  EMPTY_SET = '\x8f',
  ADDITEMS = '\x90',
  FROZENSET = '\x91',
  NEWOBJ_EX = '\x92',
  STACK_GLOBAL = '\x93',
  MEMOIZE = '\x94',
  FRAME = '\x95'
};

// 定义 PickleOpCode 枚举，表示 Python 的 pickle 协议操作码

using ::c10::IValue;
// 使用 c10 命名空间中的 IValue 类型

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
struct WriteableTensorData {
  const char* data() const {
    return static_cast<const char*>(tensor_.storage().data());
  }
  // 返回可写张量数据的指针

  size_t sizeInBytes() const {
    return size_;
  }
  // 返回数据大小（以字节为单位）

  size_t nbytes() const {
    return tensor_.storage().nbytes();
  }
  // 返回数据大小（以字节为单位）

  bool storageHasDeleter() const {
    return tensor_.storage().data_ptr().get_context() != nullptr;
  }
  // 检查张量存储是否具有删除器

 private:
  friend TORCH_API WriteableTensorData
  getWriteableTensorData(const at::Tensor& tensor, bool to_cpu);
  // 声明友元函数 getWriteableTensorData，用于获取可写张量数据

  at::Tensor tensor_;
  // 定义张量对象

  uint64_t size_;
  // 定义数据大小
};

void setTypeTags(bool state);
// 声明函数 setTypeTags，用于设置类型标签状态

bool getTypeTags();
// 声明函数 getTypeTags，用于获取类型标签状态

}  // namespace torch::jit
// 命名空间结束声明
// 定义名为 Pickler 的类，用于序列化对象并将其写入给定的 writer 函数
class TORCH_API Pickler {
  AT_DISALLOW_COPY_AND_ASSIGN(Pickler);  // 禁止复制和赋值操作

 public:
  // 构造函数，接受一个写入函数作为参数，并使用默认值初始化其他参数
  Pickler(std::function<void(const char*, size_t)> writer)
      : Pickler(std::move(writer), nullptr, nullptr, nullptr) {}

  // 第二个构造函数，接受多个参数，包括写入函数、张量表、类型重命名函数、类类型表等，并设置默认值
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  Pickler(
      std::function<void(const char*, size_t)> writer,
      std::vector<at::Tensor>* tensor_table,
      std::function<c10::QualifiedName(const c10::ClassTypePtr&)> type_renamer,
      std::vector<c10::ClassTypePtr>* memoized_class_types,
      std::function<std::string(const at::Tensor&)> get_tensor_id = nullptr,
      bool tag_aggregates = true)
      : writer_(std::move(writer)),
        tensor_table_(tensor_table),
        type_renamer_(std::move(type_renamer)),
        memoized_class_types_(memoized_class_types),
        get_tensor_id_(std::move(get_tensor_id)),
        tag_aggregates_(tag_aggregates) {}

  // 析构函数，用于清理资源
  // NOLINTNEXTLINE(bugprone-exception-escape)
  ~Pickler();

  // 将协议推入堆栈
  void protocol();

  // 将 STOP PickleOpCode 推入堆栈
  void stop();

  // 将 IValue 对象推入堆栈
  void pushIValue(const IValue& ivalue);

  // 开始一个元组
  void startTuple();

  // 结束一个元组
  void endTuple();

  // 返回张量数据的常量引用
  const std::vector<at::Tensor>& tensorData() {
    return tensor_data_;
  }

  // 推入空字典
  void pushEmptyDict();

  // 推入字典
  void pushDict(const IValue& ivalue);

  // 推入整数
  void pushInt(int64_t value);

  // 推入长字符串
  void pushLong(const std::string& data);

 private:
  // 实际推送 IValue 对象到堆栈的实现函数
  void pushIValueImpl(const IValue& ivalue);

  // 开始类型标签
  void startTypeTag();

  // 结束类型标签
  void endTypeTag(const IValue& value);

  // 推入布尔值
  void pushBool(bool value);

  // 推入双精度浮点数
  void pushDouble(double value);

  // 推入复数双精度浮点数
  void pushComplexDouble(const IValue& value);

  // 推入通用列表
  void pushGenericList(const IValue& ivalue);

  // 推入整数列表
  void pushIntList(const IValue& ivalue);

  // 推入列表
  void pushList(const IValue& ivalue);

  // 推入张量
  void pushTensor(const IValue& ivalue);

  // 推入张量引用
  void pushTensorReference(const IValue& ivalue);

  // 推入字面量张量
  void pushLiteralTensor(const IValue& ivalue);

  // 推入字面量稀疏张量
  void pushLiteralSparseTensor(const at::Tensor& tensor);

  // 推入元组
  void pushTuple(const IValue& ivalue);

  // 推入字符串
  void pushString(const std::string& string);

  // 推入设备信息
  void pushDevice(const IValue& ivalue);

#ifdef USE_DISTRIBUTED
  // 推入远程引用
  void pushRRef(const IValue& ivalue);
#endif

  // 推入原始字符串数据到字节流
  void pushBytes(const std::string& string);

  // 推入张量数据
  void pushTensorData(const at::Tensor& tensor);

  // 添加 BINPUT 操作并返回使用的记忆化 ID
  size_t pushNextBinPut();

  // 获取 IValue 对象的指针
  const void* getPointer(const IValue& ivalue);

  // 当缓冲区不为空时刷新写入
  void flushNonEmpty() {
    writer_(buffer_.data(), bufferPos_);
    bufferPos_ = 0;
  }

  // 刷新写入
  void flush() {
    if (bufferPos_ != 0) {
      flushNonEmpty();
    }
  }
  }

  // 这些函数将值转换为字节并将其添加到堆栈中（注意：由于 T 位于 '::' 的左边，编译器无法推断其类型，因此必须显式实例化模板，例如 push<int>(int) 可行，push(int) 不可行）
  static CONSTEXPR_EXCEPT_WIN_CUDA size_t kBufferSize = 256;
  template <typename T>
  void push(std::common_type_t<T> value) {
    // 将值的内存地址解释为 const char* 类型的指针
    const char* begin = reinterpret_cast<const char*>(&value);
    // 如果当前位置加上 T 的大小超过了缓冲区的大小，则刷新非空缓冲区
    if (bufferPos_ + sizeof(T) > buffer_.size()) {
      flushNonEmpty();
    }
    // 静态断言，确保 T 的大小不超过 kBufferSize，用于缓冲区大小的假设
    static_assert(sizeof(T) <= kBufferSize, "Buffer size assumption");
    // 将值的字节复制到缓冲区中
    memcpy(buffer_.data() + bufferPos_, begin, sizeof(T));
    bufferPos_ += sizeof(T);
  }

  // 用于写入二进制数据的流
  // 在直接调用 writer_ 之前，代码应先调用 flush()
  std::function<void(const char*, size_t)> writer_;

  // 缓冲区，用于避免按每个字节调用 writer_
  std::array<char, kBufferSize> buffer_;
  size_t bufferPos_{0};

  // 操作码/数据的堆栈
  std::vector<char> stack_;

  // 要序列化的张量的外部表格。如果缺少此项，则张量将直接序列化到 pickle 中
  std::vector<at::Tensor>* tensor_table_;

  // TODO: 仅在必要时使用（添加一个 pass 来查找所有共享的 ivalues，并只对它们进行记忆）
  uint32_t memo_id_ = 0;

  // 已写入的 IValue 的记忆化（表中的索引用于 BINPUT 操作码），以启用共享引用
  c10::FastMap<const void*, uint32_t> memoized_ivalue_map_;

  // 由于上述映射中基于其原始指针地址对 ivalues 进行了去重，因此在 pickle 过程中需要保持所有已记忆的值保持活跃状态。
  // 否则，可能会出现原始地址被重新使用的情况，而我们会将其别名为旧对象的地址。
  std::vector<IValue> memoized_ivalues_;

  // 类型重命名器
  std::function<c10::QualifiedName(const c10::ClassTypePtr&)> type_renamer_;

  // 写入的所有类型的列表，可以从写入的 IValues 进行检查。
  std::vector<c10::ClassTypePtr>* memoized_class_types_;

  // 用于张量存储的下一个 id_name 的函数，该函数负责返回唯一的 id
  std::function<std::string(const at::Tensor&)> get_tensor_id_;

  // 要在 pickle 数据中序列化的张量存储列表，与 ivalues 类似，它们使用 BINPUT 进行记忆化
  std::vector<at::Tensor> tensor_data_;
  c10::FastMap<const void*, uint32_t> memoized_storage_map_;

  c10::FastMap<std::string, uint32_t> memoized_globals_map_;
  c10::FastMap<std::string, uint32_t> memoized_strings_map_;
  c10::FastMap<std::string, uint32_t> memoized_devices_map_;
  // 当为 true 时，List 和 Dict 对象将被包装在 torch.jit._pickle.restore_type_tag 调用中，以正确设置对象的动态 TorchScript 类型。
  // 当为 true 时，反序列化的对象必须安装有 torch。
  bool tag_aggregates_;
// 结构体 WriteableTensorData，用于封装可写入的张量数据以及记录大小
TORCH_API WriteableTensorData
getWriteableTensorData(const at::Tensor& tensor, bool to_cpu = true);

// 获取张量的存储指针的数值表示
uint64_t getStorageKey(const at::Tensor& tensor);

// 检查类是否具有 __getstate__/__setstate__ 方法，并验证其模式是否正确，返回 true 或 false
bool checkHasValidSetGetState(const std::shared_ptr<c10::ClassType>& cls);

// 声明 BackendMeta 的序列化和反序列化函数指针类型
using BackendMetaPtr = std::function<
    void(const at::Tensor&, std::unordered_map<std::string, bool>&)>;

// 获取后端设备类型的白名单集合，目前仅包含 PrivateUse1
inline std::unordered_set<c10::DeviceType>& GetBackendMetaAllowlist() {
  static std::unordered_set<c10::DeviceType> DeviceTypeAllowlist{
      c10::DeviceType::PrivateUse1};
  return DeviceTypeAllowlist;
}

// 动态获取需要相应后端的序列化函数对
inline std::array<
    std::optional<std::pair<BackendMetaPtr, BackendMetaPtr>>,
    at::COMPILE_TIME_MAX_DEVICE_TYPES>&
GetBackendMetaSerialization() {
  // 保存 BackendMeta 序列化函数指针的数组
  // 键为 DeviceType，值为 std::pair 对象
  // std::pair 的 first 表示获取函数，second 表示设置函数
  static std::array<
      std::optional<std::pair<BackendMetaPtr, BackendMetaPtr>>,
      at::COMPILE_TIME_MAX_DEVICE_TYPES>
      BackendMetaSerialization;
  return BackendMetaSerialization;
}

// 注册张量后端元数据的序列化函数指针
TORCH_API inline void TensorBackendMetaRegistry(
    c10::DeviceType t,
    const BackendMetaPtr& get_fptr,
    const BackendMetaPtr& set_fptr) {
  // 进行白名单验证
  // 只有当设备类型在白名单中时，才允许注册后端元数据的序列化扩展
  const auto& DeviceTypeAllowlist = GetBackendMetaAllowlist();
  TORCH_CHECK(
      DeviceTypeAllowlist.find(t) != DeviceTypeAllowlist.end(),
      "It is not allowed to register the serialization method ",
      "of backendMeta data for PrivateUse1. ",
      "If you have related serialization requirements, ",
      "please expand the allowlist");
  // 注册函数指针
  int device_type = static_cast<int>(t);
  auto& BackendMetaSerialization = GetBackendMetaSerialization();
  TORCH_CHECK(
      !BackendMetaSerialization[device_type].has_value(),
      "The tensor BackendMeta serialization function pointer for ",
      t,
      " has been registered.");
  BackendMetaSerialization[device_type] =
      std::optional<std::pair<BackendMetaPtr, BackendMetaPtr>>(
          std::make_pair(get_fptr, set_fptr));
}

// 返回包含后端元数据用于序列化的张量元数据的映射。目前仅处理 `conj` 和 `neg` 位。
// 获取张量的元数据，返回一个映射表，记录了张量的特性信息
inline std::unordered_map<std::string, bool> getTensorMetadata(
    const at::Tensor& t) {
  // 检查是否是 ZeroTensor，因为它尚未公开，不支持序列化
  TORCH_CHECK(
      !t._is_zerotensor(),
      "ZeroTensor is not serializable,",
      " please file an issue if required.");
  // 初始化空的元数据映射表
  std::unordered_map<std::string, bool> metadata{};

  // 只有在值不为默认值时才添加元数据
  if (t.is_conj()) {
    metadata["conj"] = true;
  }
  if (t.is_neg()) {
    metadata["neg"] = true;
  }

  // 如果注册了自定义后端的序列化函数指针，则添加 BackendMetaData
  int device_type = static_cast<int>(t.device().type());
  const auto& BackendMetaSerialization = GetBackendMetaSerialization();
  if (BackendMetaSerialization[device_type].has_value()) {
    // 获取自定义后端的序列化函数指针并调用，传入张量和元数据映射表的引用
    BackendMetaPtr fptr = BackendMetaSerialization[device_type].value().first;
    fptr(t, metadata);
  }

  // 返回构建好的元数据映射表
  return metadata;
}

// 根据传入的映射表设置张量的元数据。
// 参考: getTensorMetadata
inline void setTensorMetadata(
    const at::Tensor& t,
    std::unordered_map<std::string, bool> metadata) {
  // 迭代器定义，用于遍历映射表
  auto iter_end = metadata.end();
  auto iter_temp = metadata.find("conj");

  // 如果在映射表中找到 "conj"，则设置张量的共轭属性为 true，并从映射表中删除该项
  if (iter_temp != iter_end) {
    t._set_conj(true);
    metadata.erase(iter_temp);
  }

  // 类似地，如果在映射表中找到 "neg"，则设置张量的负属性为 true，并从映射表中删除该项
  iter_temp = metadata.find("neg");
  if (iter_temp != iter_end) {
    t._set_neg(true);
    metadata.erase(iter_temp);
  }

  // 如果注册了自定义后端的反序列化函数指针，则设置 BackendMetaData
  int device_type = static_cast<int>(t.device().type());
  const auto& BackendMetaSerialization = GetBackendMetaSerialization();
  if (BackendMetaSerialization[device_type].has_value()) {
    // 获取自定义后端的反序列化函数指针并调用，传入张量和元数据映射表的引用
    BackendMetaPtr fptr = BackendMetaSerialization[device_type].value().second;
    fptr(t, metadata);
  }
}

// 根据传入的字典设置张量的元数据。
// 注意: 这个重载函数是由 unpickler.cpp 需要的
inline void setTensorMetadata(
    const at::Tensor& t,
    const c10::Dict<c10::IValue, c10::IValue>& metadata_idict) {
  // 初始化空的元数据映射表
  std::unordered_map<std::string, bool> metadata;

  // 遍历传入的字典，将其转换为标准的元数据映射表
  for (auto& pair : metadata_idict) {
    auto key = *pair.key().toString();  // 获取键的字符串表示
    metadata[key] = pair.value().toBool();  // 获取值并转换为布尔类型，存入映射表
  }

  // 调用前面定义的函数，根据转换得到的元数据映射表设置张量的元数据
  setTensorMetadata(t, std::move(metadata));
}

} // namespace torch::jit
```