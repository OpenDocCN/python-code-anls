# `.\pytorch\torch\csrc\jit\serialization\unpickler.cpp`

```
// 包含 ATen 库的头文件，提供了张量操作的支持
#include <ATen/ATen.h>

// 包含 ATen 核心的字典类
#include <ATen/core/Dict.h>

#ifdef USE_RPC
// 如果定义了 USE_RPC，包含分布式 RPC 的远程引用上下文
#include <torch/csrc/distributed/rpc/rref_context.h>
#endif

// 包含 Torch JIT 的函数实现 API 头文件
#include <torch/csrc/jit/api/function_impl.h>

// 包含 Torch JIT 的移动设备类型解析器
#include <torch/csrc/jit/mobile/type_parser.h>

// 包含 Torch JIT 的 Pickler 序列化器
#include <torch/csrc/jit/serialization/pickler.h>

// 包含 Torch JIT 的存储上下文
#include <torch/csrc/jit/serialization/storage_context.h>

// 包含 Torch JIT 的 Unpickler 反序列化器
#include <torch/csrc/jit/serialization/unpickler.h>

// 包含 Torch 的字节顺序处理工具
#include <torch/csrc/utils/byte_order.h>

// 包含标准字符串库
#include <string>

// 包含实用工具的实用程序类
#include <utility>

// 命名空间 torch::jit 中的实现
namespace torch::jit {

// 使用 c10::IValue 进行简化命名
using ::c10::IValue;

// 用于恢复精确类型标签的静态函数，如果可能的话
static void restoreAccurateTypeTagsIfPossible(const IValue& root) {
  // 如果根值是对象类型，则调用 restoreAccurateTypeTags 函数恢复类型标签
  if (root.isObject()) {
    restoreAccurateTypeTags(root, root.type());
  }
}

// 恢复精确类型标签的函数，根据提供的根值和类型标签
void restoreAccurateTypeTags(const IValue& root, const TypePtr& type_tag) {
  // 定义一个工作结构体，用于追踪类型和值
  struct Work {
    TypePtr type;
    IValue value;
  };
  // 初始化工作队列，包含根类型和值
  std::vector<Work> to_process = {{type_tag, root}};
  // 用于存储已扫描过的地址，防止重复扫描
  std::unordered_set<const void*> scanned;
  // 循环处理工作队列中的每个工作项
  while (!to_process.empty()) {
    Work w = std::move(to_process.back());
    to_process.pop_back();
    // 如果值是指针类型，确保只扫描每个指针值一次
    if (w.value.isPtrType()) {
      const void* key = w.value.internalToPointer();
      auto it = scanned.find(key);
      if (it != scanned.end()) {
        continue;
      }
      scanned.emplace_hint(it, key);
    }
    // 获取类型的种类
    auto kind = w.type->kind();
    // 如果是动态类型，获取动态类型的种类
    if (auto dyn = w.type->castRaw<c10::DynamicType>()) {
      kind = dyn->dynamicKind();
    }
    // 根据类型种类执行相应操作，这里代码似乎缺少具体的处理逻辑
    // 可能需要补充处理逻辑以恢复精确类型标签
  }
}

// 匿名命名空间，定义了模板函数用于检查类型是否匹配
namespace {
template <typename T>
bool is(const Type& type) {
  // 如果类型的种类匹配 T 类型的种类，则返回 true
  if (type.kind() == T::Kind) {
    return true;
  }
  // 如果是动态类型，再检查动态类型的标签是否匹配
  if (auto dyn = type.castRaw<c10::DynamicType>()) {
    return dyn->tag() == c10::DynamicTypeTrait<T>::tagValue();
  }
  // 否则返回 false
  return false;
}
} // namespace

// 恢复容器类型标签的函数，根据提供的值和类型
static void restoreContainerTypeTags(
    const IValue& ivalue,
    const TypePtr& type) {
  // 如果是字典类型，则设置字典的键类型和值类型
  if (is<DictType>(*type)) {
    auto dict = ivalue.toGenericDict();
    dict.unsafeSetKeyType(type->containedType(0));
    dict.unsafeSetValueType(type->containedType(1));
  } else if (is<ListType>(*type)) {
    // 如果是列表类型，则设置列表的元素类型
    ivalue.toList().unsafeSetElementType(type->containedType(0));
  } else {
    // 如果类型未知，则抛出错误
    AT_ERROR("Unknown type for tag restoration: " + type->annotation_str());
  }
}
// 执行解析过程，可能会修改解析栈
IValue Unpickler::parse_ivalue() {
  run();  // 调用运行函数，开始解析过程
  // 检查解析栈是否仅有一个元素
  TORCH_CHECK(
      stack_.size() == 1,
      "Unpickler expected 1 element on the stack, but found ",
      stack_.size());
  // 如果版本小于等于2，恢复准确的类型标签（见[type tag serialization]）
  if (version_ <= 2) {
    restoreAccurateTypeTagsIfPossible(stack_[0]);  // 恢复可能的准确类型标签
  }
  // 返回解析栈顶的元素
  return stack_[0];
}

// 读取一个双精度浮点数，考虑大小端字节序
double Unpickler::readFloat() {
  AT_ASSERT(sizeof(double) == 8);  // 断言双精度浮点数占8字节
  double big_endian = read<double>();  // 读取大端序的双精度浮点数
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  // 如果是小端序，将字节反转以得到小端序的双精度浮点数
  double little_endian;
  auto big_endian_ptr = reinterpret_cast<const char*>(&big_endian);
  std::reverse_copy(
      big_endian_ptr,
      big_endian_ptr + sizeof(big_endian),
      reinterpret_cast<char*>(&little_endian));
  return little_endian;
#elif __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  // 如果是大端序，直接返回大端序的双精度浮点数
  return big_endian;
#else
#error Unexpected or undefined __BYTE_ORDER__
#endif
}

void Unpickler::run() {
  // 期望在数据块开头找到 PROTO 操作码和协议号
  auto opcode = readOpCode();
  TORCH_CHECK(
      opcode == PickleOpCode::PROTO,
      "Expected PROTO opcode at the start"
      " of pickle archive, found ",
      int(static_cast<uint8_t>(opcode)));
  uint8_t protocol = read<uint8_t>();
  TORCH_CHECK(
      protocol == 2,
      "Only Pickle protocol 2 is supported, found protocol = ",
      protocol);

  while (true) {
    // 读取下一个操作码
    PickleOpCode opcode = readInstruction();
    // 如果遇到 STOP 操作码，结束循环
    if (opcode == PickleOpCode::STOP) {
      return;
    }
  }
}

void Unpickler::setInput(size_t memo_id) {
  AT_ASSERT(!stack_.empty());  // 断言栈非空
  // 如果 memo_id 超出了 memo_table_ 的大小，进行插入操作
  if (memo_id >= memo_table_.size()) {
    memo_table_.insert(
        memo_table_.end(), memo_id - memo_table_.size(), IValue());
    memo_table_.push_back(stack_.back());
  } else {
    // 否则，直接赋值给 memo_table_ 中的相应位置
    memo_table_[memo_id] = stack_.back();
  }
}

// 在某些系统上，bool 向量上不存在 emplace_back 操作，通过 push_back 避免问题
template <typename T>
inline void append(std::vector<T>& a, T&& e) {
  a.emplace_back(std::forward<T>(e));
}

// 特化模板，处理 bool 类型的向量
template <>
inline void append<bool>(std::vector<bool>& a, bool&& e) {
  a.push_back(e);
}

// 将 IValue 转换为 int64_t 类型的向量
static std::vector<int64_t> tupleToIntList(const IValue& v) {
  return fmap(v.toTupleRef().elements(), [](const IValue& v) -> int64_t {
    return v.toInt();
  });
}

// 在反序列化时不能使用 toIntList、toDoubleList，因为列表尚未被标记
// 将 IValue 转换为指定类型 T 的向量
template <typename T>
static std::vector<T> convertList(const IValue& v) {
  return fmap(v.toListRef(), [](const IValue& elem) { return elem.to<T>(); });
}

// 读取指令操作码
PickleOpCode Unpickler::readInstruction() {
  auto opcode = readOpCode();
  switch (opcode) {
    case PickleOpCode::EMPTY_LIST: {
      // 如果操作码是 EMPTY_LIST，向栈中添加一个空的通用列表
      stack_.emplace_back(c10::impl::GenericList(AnyType::get()));
    } break;
    case PickleOpCode::EMPTY_TUPLE: {
      // 如果 empty_tuple_ 尚未初始化，则创建一个空元组对象
      if (empty_tuple_.isNone()) {
        empty_tuple_ = c10::ivalue::Tuple::create(std::vector<IValue>());
      }
      // 将 empty_tuple_ 压入栈中
      stack_.emplace_back(empty_tuple_);
    } break;
    case PickleOpCode::BINPUT: {
      // 读取一个 uint8_t 类型的 memo_id，并调用 setInput 设置输入
      size_t memo_id = read<uint8_t>();
      setInput(memo_id);
    } break;
    case PickleOpCode::LONG_BINPUT: {
      // 检查 size_t 是否足够大以解码 uint32_t 类型的 memo_id
      TORCH_CHECK(
          std::numeric_limits<size_t>::max() >=
              std::numeric_limits<uint32_t>::max(),
          "Found a LONG_BINPUT opcode, but size_t on this system is "
          "not big enough to decode it");
      // 读取一个 uint32_t 类型的 memo_id，并调用 setInput 设置输入
      size_t memo_id = read<uint32_t>();
      setInput(memo_id);
    } break;
    case PickleOpCode::MARK: {
      // 标记当前容器 ivalue 在栈中的位置
      marks_.push_back(stack_.size());
    } break;
    case PickleOpCode::NEWTRUE: {
      // 在栈中压入一个布尔值 true
      stack_.emplace_back(true);
    } break;
    case PickleOpCode::NEWFALSE: {
      // 在栈中压入一个布尔值 false
      stack_.emplace_back(false);
    } break;
    case PickleOpCode::NONE: {
      // 在栈中压入一个 None 对象
      stack_.emplace_back();
    } break;
    case PickleOpCode::BININT1: {
      // 读取一个 uint8_t 类型的整数值，并将其转换为 int64_t 后压入栈中
      uint8_t value = read<uint8_t>();
      stack_.emplace_back(int64_t(value));
    } break;
    case PickleOpCode::BININT2: {
      // 读取一个 uint16_t 类型的整数值，并将其转换为 int64_t 后压入栈中
      uint16_t value = from_le16(read<uint16_t>());
      stack_.emplace_back(int64_t(value));
    } break;
    case PickleOpCode::BININT: {
      // 读取一个 int32_t 类型的整数值，并将其转换为 int64_t 后压入栈中
      int32_t value = from_le32(read<int32_t>());
      stack_.emplace_back(int64_t(value));
    } break;
    case PickleOpCode::LONG1: {
      // 读取一个长度为 8 的 int64_t 类型的整数值，并压入栈中
      uint8_t length = read<uint8_t>();
      TORCH_CHECK(length == 8, "Expected length to be 8, got ", int(length));
      stack_.emplace_back(int64_t(from_le64(read<int64_t>())));
    } break;
    case PickleOpCode::BINUNICODE: {
      // 读取一个 uint32_t 类型的字符串长度，并根据长度读取相应字节数的字符串数据，压入栈中
      uint32_t length = from_le32(read<uint32_t>());
      stack_.emplace_back(readBytes(length));
    } break;
    case PickleOpCode::BINUNICODE8: {
      // 读取一个 int64_t 类型的字符串长度，并根据长度读取相应字节数的字符串数据，压入栈中
      int64_t length = from_le64(read<int64_t>());
      stack_.emplace_back(readBytes(length));
    } break;
    case PickleOpCode::BINFLOAT:
      // 读取一个浮点数，并将其压入栈中
      stack_.emplace_back(readFloat());
      break;
    case PickleOpCode::TUPLE: {
      // 检查 marks_ 是否为空，如果为空则抛出解析错误
      TORCH_CHECK(!marks_.empty(), "Parsing error: marks_ is empty");
      // 取出 marks_ 中最后一个标记的索引作为元组起始位置
      size_t start = marks_.back();
      marks_.pop_back();
      // 创建一个空的元素列表
      std::vector<IValue> elements;
      // 检查 stack_ 的大小是否大于等于起始索引 start
      TORCH_CHECK(
          stack_.size() >= start,
          "Parsing error: wrong start index ",
          start,
          " for stack_ of size ",
          stack_.size());
      // 计算元组的大小
      const auto tupleSize = stack_.size() - start;
      switch (tupleSize) {
        case 3: {
          // 弹出栈顶的三个元素
          auto e3 = pop(stack_);
          auto e2 = pop(stack_);
          auto e1 = pop(stack_);
          // 将三个元素创建为一个 Tuple 并压入栈中
          stack_.emplace_back(c10::ivalue::Tuple::create(
              std::move(e1), std::move(e2), std::move(e3)));
          break;
        }
        case 2: {
          // 弹出栈顶的两个元素
          auto e2 = pop(stack_);
          auto e1 = pop(stack_);
          // 将两个元素创建为一个 Tuple 并压入栈中
          stack_.emplace_back(
              c10::ivalue::Tuple::create(std::move(e1), std::move(e2)));
          break;
        }
        case 1:
          // 弹出栈顶的一个元素，创建为一个单元素的 Tuple 并压入栈中
          stack_.emplace_back(c10::ivalue::Tuple::create(pop(stack_)));
          break;
        default: {
          // 对于大于3个元素的情况，创建一个包含所有元素的 Tuple
          elements.reserve(stack_.size() - start);
          auto start_it = stack_.begin() + start;
          for (auto it = start_it; it != stack_.end(); ++it) {
            elements.emplace_back(std::move(*it));
          }
          stack_.erase(start_it, stack_.end());
          stack_.emplace_back(c10::ivalue::Tuple::create(std::move(elements)));
          break;
        }
      }
    } break;
    case PickleOpCode::TUPLE1: {
      // 检查 stack_ 是否为空，如果为空则抛出解析错误
      TORCH_CHECK(
          !stack_.empty(),
          "Parsing error: stack_ contains ",
          stack_.size(),
          " elements, at least 1 expected");
      // 弹出栈顶的一个元素，创建为一个单元素的 Tuple 并压入栈中
      stack_.emplace_back(c10::ivalue::Tuple::create(pop(stack_)));
    } break;
    case PickleOpCode::TUPLE2: {
      // 检查 stack_ 中是否至少有两个元素，如果没有则抛出解析错误
      TORCH_CHECK(
          stack_.size() > 1,
          "Parsing error: stack_ contains ",
          stack_.size(),
          " elements, at least 2 expected");
      // 弹出栈顶的两个元素，创建为一个两元素的 Tuple 并压入栈中
      auto e2 = pop(stack_);
      auto e1 = pop(stack_);
      stack_.emplace_back(
          c10::ivalue::Tuple::create(std::move(e1), std::move(e2)));
    } break;
    case PickleOpCode::TUPLE3: {
      // 检查 stack_ 中是否至少有三个元素，如果没有则抛出解析错误
      TORCH_CHECK(
          stack_.size() > 2,
          "Parsing error: stack_ contains ",
          stack_.size(),
          " elements, at least 3 expected");
      // 弹出栈顶的三个元素，创建为一个三元素的 Tuple 并压入栈中
      auto e3 = pop(stack_);
      auto e2 = pop(stack_);
      auto e1 = pop(stack_);
      stack_.emplace_back(c10::ivalue::Tuple::create(
          std::move(e1), std::move(e2), std::move(e3)));
    } break;
    case PickleOpCode::EMPTY_DICT:
      // 创建一个空的 GenericDict 并压入栈中
      stack_.emplace_back(
          c10::impl::GenericDict(AnyType::get(), AnyType::get()));
      break;
    case PickleOpCode::APPENDS: {
      // 检查 marks_ 不为空，否则抛出解析错误
      TORCH_CHECK(!marks_.empty(), "Parsing error: marks_ is empty");
      // 取得 marks_ 中最后一个位置作为起始点
      size_t start = marks_.back();
      // 检查起始点的有效性，必须大于 0 并且不超过 stack_ 的大小
      TORCH_CHECK(
          start > 0 && start <= stack_.size(),
          "Parsing error: wrong start index ",
          start,
          " for stack_ of size ",
          stack_.size());
      // 在 stack_ 中取得起始点的元素，应为一个列表类型的 IValue
      auto list_ivalue = stack_.at(start - 1);
      // 调用 readList 函数读取列表内容
      readList(list_ivalue);
    } break;
    case PickleOpCode::APPEND: {
      // 检查 stack_ 中至少有两个元素，否则抛出解析错误
      TORCH_CHECK(
          stack_.size() >= 2, "Parsing error: missing elements in stack_.");
      // 在 stack_ 中取得倒数第二个元素，应为一个列表类型的 IValue
      auto list_ivalue = stack_.at(stack_.size() - 2);
      // 调用 readListElements 函数读取列表元素，并传入最后一个元素的索引
      readListElements(list_ivalue, stack_.size() - 1);
    } break;
    case PickleOpCode::LIST: {
      // 创建一个空的通用列表类型的 IValue
      IValue list_ivalue = c10::impl::GenericList(AnyType::get());
      // 调用 readList 函数读取列表内容并填充到 list_ivalue 中
      readList(list_ivalue);
      // 将填充好的 list_ivalue 放入 stack_ 中
      stack_.push_back(std::move(list_ivalue));
    } break;
    case PickleOpCode::DICT: {
      // 检查 marks_ 不为空，否则抛出解析错误
      TORCH_CHECK(!marks_.empty(), "Parsing error: marks_ is empty");
      // 取得 marks_ 中最后一个位置作为起始点，并将其从 marks_ 中移除
      size_t start = marks_.back();
      marks_.pop_back();
      // 检查 stack_ 中有足够的元素来构建字典，起始点必须有效
      TORCH_CHECK(
          stack_.size() > start,
          "Parsing error: wrong start index ",
          start,
          " for stack_ which of size ",
          stack_.size());
      // 创建一个空的通用字典类型的对象
      auto dict = c10::impl::GenericDict(AnyType::get(), AnyType::get());
      // 检查 stack_ 中从起始点开始的元素数量为偶数，保证每个键值对都有配对
      TORCH_CHECK(
          (stack_.size() - start) % 2 == 0,
          "Parsing error: stack_ is of size ",
          stack_.size(),
          " and start index is ",
          start,
          ", but stack_ is iterated by two elements at a time");
      // 遍历 stack_ 中起始点开始的每对元素，插入到字典中
      for (size_t i = start; i < stack_.size(); i += 2) {
        dict.insert_or_assign(stack_[i], stack_[i + 1]);
      }
      // 删除 stack_ 中起始点之后的所有元素
      stack_.erase(stack_.begin() + start, stack_.end());
      // 将构建好的字典对象放入 stack_ 中
      stack_.emplace_back(std::move(dict));
    } break;
    case PickleOpCode::SETITEMS: {
      // 检查 marks_ 不为空，否则抛出解析错误
      TORCH_CHECK(!marks_.empty(), "Parsing error: marks_ is empty");
      // 取得 marks_ 中最后一个位置作为起始点，并将其从 marks_ 中移除
      size_t start = marks_.back();
      marks_.pop_back();
      // 检查起始点的有效性，必须大于 0 并且不超过 stack_ 的大小
      TORCH_CHECK(
          start > 0 && start <= stack_.size(),
          "Parsing error: wrong start index for stack_");
      // 在 stack_ 中取得起始点的元素，并转换为通用字典类型
      auto dict = stack_.at(start - 1).toGenericDict();
      // 检查 stack_ 中从起始点开始的元素数量为偶数，保证每个键值对都有配对
      TORCH_CHECK(
          (stack_.size() - start) % 2 == 0,
          "Parsing error: stack_ is of size ",
          stack_.size(),
          " and start index is ",
          start,
          ", but stack_ is iterated by two elemenst at a time");
      // 遍历 stack_ 中起始点开始的每对元素，插入到字典中
      for (size_t i = start; i < stack_.size(); i += 2) {
        dict.insert_or_assign(stack_[i], stack_[i + 1]);
      }
      // 删除 stack_ 中起始点之后的所有元素
      stack_.erase(stack_.begin() + start, stack_.end());
    } break;
    case PickleOpCode::BINGET: {
      // 读取一个字节作为 memo_table_ 的索引位置
      auto pos = read<uint8_t>();
      // 检查索引位置是否在 memo_table_ 的有效范围内
      TORCH_CHECK(
          memo_table_.size() > pos,
          "Parsing error: out of bounds access at ",
          (size_t)pos,
          " to memo_table_ which is of size ",
          memo_table_.size());
      // 将 memo_table_ 中的值压入 stack_ 中
      stack_.push_back(memo_table_.at(pos));
    } break;
    case PickleOpCode::LONG_BINGET: {
      // 从字节流中读取一个 uint32_t 类型的位置信息
      auto pos = read<uint32_t>();
      // 检查 memo_table_ 的大小，确保访问的位置在有效范围内
      TORCH_CHECK(
          memo_table_.size() > pos,
          "Parsing error: out of bounds access at ",
          (size_t)pos,
          " to memo_table_ which is of size ",
          memo_table_.size());
      // 将 memo_table_ 中对应位置的值压入栈中
      stack_.push_back(memo_table_.at(pos));
    } break;
    case PickleOpCode::STOP:
      // 停止操作，无需执行任何动作
      break;
    case PickleOpCode::GLOBAL: {
      // 读取模块名，这里没有实际使用需求
      auto module_name = readString();
      // 读取类名
      auto class_name = readString();
      // 根据模块名和类名读取全局对象
      readGlobal(module_name, class_name);
    } break;
    case PickleOpCode::NEWOBJ: {
      // 检查堆栈是否为空
      TORCH_CHECK(!stack_.empty(), "Parsing error: stack_ is empty");
      // 弹出空元组，实际操作在 globals_stack_ 中进行
      stack_.pop_back();
    } break;
    // 因为 NEWOBJ 不做任何操作，BUILD 和 REDUCE 实际上会执行相同的操作
    case PickleOpCode::BUILD:
    case PickleOpCode::REDUCE: {
      // 堆栈结构为: <functor_idx> <functor_arg>
      // 提取 <functor_idx> 并从堆栈中移除
      TORCH_CHECK(
          stack_.size() > 1,
          "Parsing error: stack_ contains ",
          stack_.size(),
          " elements, at least 2 expected");
      std::swap(*(stack_.end() - 2), *(stack_.end() - 1));
      // 获取 <functor_idx> 的值作为索引
      size_t idx = stack_.back().toInt();
      stack_.pop_back();
      // 堆栈结构为: <functor_arg>
      // 检查索引是否超出 globals_ 的范围
      TORCH_CHECK(
          idx < globals_.size(),
          "Parsing error: out of bounds access to globals_");
      // 调用 globals_ 中对应索引处的函数对象
      globals_.at(idx)();
    } break;
    case PickleOpCode::BINPERSID: {
      // 检查堆栈是否为空，如果为空则抛出解析错误
      TORCH_CHECK(!stack_.empty(), "Parsing error: stack_ is empty");
      // 从堆栈中弹出一个元组对象
      auto tuple = pop(stack_).toTuple();
      // 获取元组的所有元素
      const auto& args = tuple->elements();
      // 断言第一个参数是字符串 "storage"
      AT_ASSERT(
          args.at(0).toStringRef() == "storage",
          "unknown PERSID key ",
          args.at(0).toStringRef());
      // 第二个参数是标量类型
      at::ScalarType type = args.at(1).toScalarType();
      // 第三个参数是字符串类型的键
      const std::string& key = args.at(2).toStringRef();

      // 第四个参数是设备类型，根据字符串创建设备对象
      at::Device device(args.at(3).toStringRef());
      // 如果当前有设备对象且设备不是元设备，则使用当前设备
      if (device_ && !device.is_meta()) {
        device = *device_;
      }

      // 创建一个空的存储对象
      at::Storage storage;
      // 如果存储上下文不为空且包含指定键的存储，则从上下文中获取存储
      if (storage_context_ != nullptr && storage_context_->hasStorage(key)) {
        // 用于 torch.package 逻辑，可能已经加载了存储
        storage = storage_context_->getStorage(key);
      } else {
        // 获取张量中的元素数
        int64_t numel = args.at(4).toInt();
        // 获取数据类型的类型元信息
        caffe2::TypeMeta dtype = at::CPU(type).typeMeta();

        at::DataPtr storage_ptr;
        // 如果元素数大于 0，则读取存储记录
        if (numel > 0) {
          // 如果张量中没有元素，则无需从输入流中读取零字节文件
          storage_ptr = read_record_(key);
        }

        // 创建存储对象
        storage = at::Storage(
            c10::Storage::use_byte_size_t(),
            numel * dtype.itemsize(),
            std::move(storage_ptr),
            /*allocator=*/nullptr,
            /*resizable=*/false); // 注意：张量没有设置任何分配器
        // 如果存储上下文不为空，则将新创建的存储添加到上下文中
        if (storage_context_ != nullptr) {
          storage_context_->addStorage(key, storage);
        }
      }

      // 获取张量的选项对象
      auto options = at::CPU(type).options();
      // 如果使用存储设备选项，则更新选项对象的设备，并更新设备
      if (use_storage_device_) {
        options = options.device(storage.device());
        device = storage.device();
      }

      // 创建一个空张量对象
      at::Tensor tensor;
      // 如果选项的后端是 QuantizedCPU，则创建一个仿射量化的空张量，并设置存储
      if (options.backend() == c10::Backend::QuantizedCPU) {
        tensor = at::_empty_affine_quantized({}, options, 0, 0)
                     .set_(storage, 0, {}, {});
      } else {
        // 否则创建一个空张量，并设置存储
        tensor = at::empty({0}, options).set_(storage);
      }

      // 如果设备是 CUDA、XPU、元设备等，则将张量转移到指定设备
      if (device.is_cuda() || device.is_xpu() || device.is_meta() ||
          device.is_hpu() || device.is_mps() || device.is_privateuseone()) {
        tensor = tensor.to(device, tensor.scalar_type());
      } else if (device.type() != DeviceType::CPU) {
        // 如果设备类型不是 CPU，则抛出错误
        AT_ERROR(
            "supported devices include CPU, CUDA, HPU and ",
            c10::get_privateuse1_backend(),
            " however got ",
            DeviceTypeName(device.type(), false));
      }
      // 将创建的张量压入堆栈中
      stack_.emplace_back(std::move(tensor));
    } break;
    case PickleOpCode::SETITEM: {
      // 当遇到 SETITEM 操作码时，栈的结构如下：
      // | 栈底         |
      // | ......       |
      // | Dict         | -> (stack_size - 3)
      // | Key          | -> (stack_size - 2)
      // | Value        | -> (stack_size - 1)

      // 检查栈中是否至少有三个元素
      TORCH_CHECK(
          stack_.size() >= 3,
          "Parsing error: stack doesn't have enough elements");

      auto stack_size = stack_.size();
      auto dict_pos = stack_size - 3;
      auto key_pos = stack_size - 2;
      auto val_pos = stack_size - 1;

      // 检查索引是否在有效范围内，避免越界访问
      TORCH_CHECK(
          (dict_pos < stack_size) && (key_pos < stack_size) &&
              (val_pos < stack_size),
          "Parsing error: attempted out-of-bounds access while processing SETITEM opcode");

      // 从栈中取出字典对象，并将键值对插入或更新字典
      auto dict = stack_.at(dict_pos).toGenericDict();
      dict.insert_or_assign(stack_.at(key_pos), stack_.at(val_pos));

      // 删除栈中 SETITEM 操作后的所有元素（保留字典）
      stack_.erase(stack_.begin() + (key_pos), stack_.end());
    } break;
    default: {
      // 遇到未知的操作码时抛出错误
      AT_ERROR(
          "Unknown opcode for unpickling at ",
          reinterpret_cast<void*>(opcode),
          ": ",
          int(static_cast<uint8_t>(opcode)));
    } break;
  }
  // 返回当前处理的操作码
  return opcode;
}

void Unpickler::readGlobal(
    const std::string& module_name,
    const std::string& class_name) {
  if (this->skip_next_read_global) {
    // 如果 skip_next_read_global 不为零，则跳过当前的全局读取操作
    this->skip_next_read_global--;
    if (this->skip_next_read_global == 1) {
      // 对应于正确的处理程序
      // Pass through to the correct handler
    } else if (this->skip_next_read_global == 0) {
      // 对应于正在反序列化的 Tensor 类型
      // 如果模块名不是 "torch" 或者类名不是 "Tensor"，发出警告并转换为 at::Tensor
      if (module_name != "torch" || class_name != "Tensor") {
        TORCH_WARN(
            "Trying to load a Subclassed Tensor, it will be converted to at::Tensor in C++");
      }
      // 将当前全局的索引添加到堆栈中
      stack_.emplace_back(int64_t(globals_.size() - 1));
      return;
    } else {
      // 如果 skip_next_read_global 不是 0 或 1，则出现无效值错误
      TORCH_CHECK(false, "INVALID VALUES")
    }
  }
  // TODO [unpickler refactor] __main__ 不再由 pickler 使用，此处仅用于向后兼容性的原因
  // 如果模块名是 "__main__"
  if (module_name == "__main__") {
    // 如果类名是 "TensorID"
    if (class_name == "TensorID") {
      // 向全局变量列表中添加一个 lambda 函数，用于处理 TensorID
      globals_.emplace_back([this] {
        // 弹出栈顶的 setitem_data
        auto setitem_data = stack_.back();
        stack_.pop_back();
        // 确保 tensor_table_ 不为空，否则抛出内部断言错误
        TORCH_INTERNAL_ASSERT(
            !tensor_table_.empty(),
            "Pickler tried to write a tensor but had no tensor table to write to");
        // 将 stack_ 中的元素设置为 tensor_table_ 中对应的值
        stack_.emplace_back(tensor_table_.at(setitem_data.toInt()));
      });
    } else if (class_name == "IntList") {
      // 向全局变量列表中添加一个 lambda 函数，用于处理 IntList
      globals_.emplace_back([this] {
        // 将栈顶的元素转换为 IntList
        stack_.back().toList().unsafeSetElementType(IntType::get());
      });
    } else {
      // 如果类名未知，抛出错误
      AT_ERROR("Unknown pickler class id", class_name);
    }
  } else if (module_name == "torch.jit._pickle") {
    // 如果模块名是 "torch.jit._pickle"
    if (class_name == "build_tensor_from_id") {
      // 向全局变量列表中添加一个 lambda 函数，用于处理 build_tensor_from_id
      globals_.emplace_back([this] {
        // 弹出栈顶的 reduce arg
        auto data = stack_.back().toTupleRef().elements().at(0);
        stack_.pop_back();
        // 确保 tensor_table_ 不为空，否则抛出错误
        TORCH_CHECK(
            !tensor_table_.empty(),
            "Found a tensor table reference but Unpickler"
            " has no tensor table\n");
        // 将 stack_ 中的元素设置为 tensor_table_ 中对应的值
        stack_.emplace_back(tensor_table_.at(data.toInt()));
      });
    } else if (class_name == "restore_type_tag") {
      // 向全局变量列表中添加一个 lambda 函数，用于处理 restore_type_tag
      globals_.emplace_back([this] {
        // 将 stack_ 中栈顶的元素转换为 Tuple
        auto tuple = stack_.back().toTuple();
        const auto& data = tuple->elements();
        // 获取类型字符串
        auto type_str = data.at(1).toStringRef();
        stack_.pop_back();
        TypePtr type = nullptr;
        // 查找类型缓存中是否存在该类型字符串对应的类型
        auto entry = type_cache_.find(type_str);
        if (entry != type_cache_.end()) {
          type = entry->second;
        } else {
          // 如果类型缓存中不存在该类型字符串，则根据 type_resolver_ 或 type_parser_ 解析类型
          if (type_resolver_ == nullptr) {
            // 如果没有自定义的类型解析方式，则使用基本的类型解析器
            type = type_parser_(type_str);
          } else {
            // 否则使用自定义的类型解析器
            type = type_resolver_(type_str).type_;
          }
          // 将解析后的类型缓存起来
          type_cache_[type_str] = type;
        }
        // TODO: 使用前瞻避免在此处创建并立即销毁元组
        // 根据数据的第一个元素和解析得到的类型，恢复容器的类型标签
        restoreContainerTypeTags(data.at(0), type);
        // 将栈顶的元素设置为 data 的第一个元素
        stack_.emplace_back(data.at(0));
      });
  } else {
    // 确定元素类型的指针，默认为nullptr
    TypePtr elem_type = nullptr;
    // 根据类名选择合适的元素类型
    if (class_name == "build_intlist") {
      elem_type = IntType::get();
    } else if (class_name == "build_tensorlist") {
      elem_type = TensorType::get();
    } else if (class_name == "build_doublelist") {
      elem_type = FloatType::get();
    } else if (class_name == "build_boollist") {
      elem_type = BoolType::get();
    } else {
      // 抛出错误，如果类名未知
      AT_ERROR("Unknown pickler class id ", class_name);
    }
    // 将 lambda 函数添加到 globals_ 中，用于反序列化列表特化 (如 List[Tensor], List[int], ...)
    globals_.emplace_back([this, elem_type] {
      // 从栈顶弹出 reduce 参数
      auto data = stack_.back().toTupleRef().elements().at(0).toList();
      stack_.pop_back();
      // 设置列表数据的元素类型
      data.unsafeSetElementType(elem_type);
      // 将处理后的数据压回栈中
      stack_.emplace_back(std::move(data));
    });
  } else if (
      module_name == "torch._utils" &&
      (class_name == "_rebuild_tensor_v2" ||
       class_name == "_rebuild_qtensor")) {
    // 反序列化张量
    bool quantized = class_name == "_rebuild_qtensor";
    rebuildTensor(quantized);
  } else if (
      module_name == "torch._tensor" &&
      (class_name == "_rebuild_from_type_v2")) {
    // 反序列化带有 Python 属性或子类化张量
    rebuildTensorFromTypeV2();
  } else if (
      module_name == "torch._utils" && class_name == "_rebuild_sparse_tensor") {
    // 反序列化稀疏张量
    rebuildSparseTensor();
  } else if (module_name == "builtins" && class_name == "complex") {
    // 处理复数对象的反序列化
    globals_.emplace_back([this] {
      auto tuple = pop(stack_).toTuple();
      const auto& elems = tuple->elements();
      AT_ASSERT(elems.size() == 2);
      // 构造复数对象并压入栈中
      auto complex =
          c10::complex<double>(elems.at(0).toDouble(), elems.at(1).toDouble());
      stack_.emplace_back(complex);
    });
  } else if (module_name == "collections" && class_name == "OrderedDict") {
    // 处理 OrderedDict 的反序列化
    globals_.emplace_back([this] {
      // 弹出作为 OrderedDict 参数的 Tuple，并将其替换为 None
      stack_.back() = IValue();
    });
  } else if (module_name == "torch" && class_name == "device") {
    // 处理设备对象的反序列化
    globals_.emplace_back([this] {
      // 从栈顶获取设备字符串
      auto device_string = stack_.back().toTupleRef().elements().at(0);
      stack_.pop_back();
      // 构造设备对象并压入栈中
      stack_.emplace_back(c10::Device(device_string.toStringRef()));
    });
    // 将处理后的对象在 globals_ 中的索引作为整数压入栈中，并返回
    stack_.emplace_back(int64_t(globals_.size() - 1));
    return;
  } else if (module_name == "torch.distributed.rpc" && class_name == "rref") {
#ifdef USE_RPC
    // 如果定义了 USE_RPC 宏，则调用 rebuildRRef() 函数返回结果
    return rebuildRRef();
#else
    // 否则，断言失败并显示错误信息，因为 RRef 反序列化仅支持分布式包
    TORCH_INTERNAL_ASSERT(
        false,
        "RRef unpickling is only supported with the distributed package");
#endif
  } else if (module_name == "torch") {
    // 尝试手动解析几个全局枚举值

    // 检查是否为 ScalarType 枚举类型
    std::optional<c10::ScalarType> scalar_type;
#define CHECK_SCALAR(_, name)          \
  if (class_name == #name "Storage") { \
    scalar_type = c10::k##name;        \
  }
    AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(CHECK_SCALAR)
#undef CHECK_SCALAR

    // 如果找到匹配的 ScalarType，则将其压入堆栈并返回
    if (scalar_type.has_value()) {
      stack_.emplace_back(int64_t(*scalar_type));
      return;
    }

    // 检查是否为 QScheme 枚举类型
    std::optional<at::QScheme> qscheme;
    for (int i = 0; i < at::COMPILE_TIME_NUM_QSCHEMES; ++i) {
      if (class_name == toString(static_cast<at::QScheme>(i))) {
        qscheme = static_cast<at::QScheme>(i);
      }
    }

    // 如果找到匹配的 QScheme，则将其压入堆栈并返回
    if (qscheme.has_value()) {
      stack_.emplace_back(int64_t(*qscheme));
      return;
    }

    // 若既非 ScalarType 也非 QScheme，则报告找到未知的 torch 全局值
    TORCH_CHECK(
        false,
        "Unpickler found unknown torch global, 'torch.",
        class_name,
        "'");
  } else {
    // 对于非 torch 模块的全局对象/类类型，尝试解析其类型

    // 断言是否存在类型解析器
    TORCH_CHECK(
        type_resolver_,
        "Unpickler found unknown type ",
        module_name,
        ".",
        class_name);

    // 解析指定的类型并处理为 StrongTypePtr
    at::StrongTypePtr type =
        type_resolver_(c10::QualifiedName(module_name, class_name));

    // 如果是枚举类型，则处理为 EnumHolder 对象并压入堆栈
    if (auto enum_type = type.type_->cast<c10::EnumType>()) {
      globals_.emplace_back([this, enum_type] {
        auto val = stack_.back();
        stack_.pop_back();
        for (const auto& p : enum_type->enumNamesValues()) {
          if (p.second == val) {
            auto enum_holder = c10::make_intrusive<at::ivalue::EnumHolder>(
                enum_type, p.first, p.second);
            stack_.emplace_back(std::move(enum_holder));
            return;
          }
        }
      });
    } else {
      // 否则，全局对象是类/对象类型，使用对象加载器加载对象并压入堆栈
      globals_.emplace_back([this, type] {
        auto val = stack_.back();
        stack_.pop_back();
        auto obj = obj_loader_(type, val);
        stack_.emplace_back(std::move(obj));
      });
    }
  }
  // 压入当前全局对象的索引到堆栈中作为结果
  stack_.emplace_back(int64_t(globals_.size() - 1));
}

void Unpickler::rebuildSparseTensor() {
  // 重建稀疏张量对象的处理函数入口

  // 压入一个 lambda 函数，用于处理稀疏张量的反序列化过程
  globals_.emplace_back([this] {
    // 弹出堆栈上的元组对象，并获取其元素列表
    auto tup = pop(stack_).toTuple();
    const auto& elements = tup->elements();
    size_t idx = 0;

    // 解析元组的第一个元素作为稀疏张量的布局信息
    auto layout = elements.at(idx++).toInt();
    at::Tensor result;
    // 开始一个 switch 语句块，根据 layout 变量的值进行不同情况的处理
    switch (layout) {
      // 如果 layout 是稀疏格式
      case static_cast<int>(c10::Layout::Sparse): {
        // 从 elements 中获取 size 向量
        std::vector<int64_t> size = tupleToIntList(elements.at(idx++));
        // 从 elements 中获取 requires_grad 布尔值
        bool requires_grad = elements.at(idx++).toBool();
        // 从 elements 中获取 indices_tensor
        auto& indices_tensor = elements.at(idx++).toTensor();
        // 从 elements 中获取 values_tensor
        auto& values_tensor = elements.at(idx++).toTensor();
        // 设置 options，指定布局为 Sparse，并设置是否需要梯度
        auto options = values_tensor.options()
                           .layout(c10::Layout::Sparse)
                           .requires_grad(requires_grad);
        // 调用 _sparse_coo_tensor_unsafe 函数创建稀疏 COO 格式的 Tensor
        result = at::_sparse_coo_tensor_unsafe(
            indices_tensor, values_tensor, size, options);
        // 将 result 转换为需要梯度的变量，并重新赋给 result
        result = autograd::make_variable(result, options.requires_grad());
        // 退出 case 语句块
        break;
      }
      // 如果 layout 是稀疏 CSR 格式
      case static_cast<int>(c10::Layout::SparseCsr): {
        // 从 elements 中获取 size 向量
        std::vector<int64_t> size = tupleToIntList(elements.at(idx++));
        // 从 elements 中获取 requires_grad 布尔值
        bool requires_grad = elements.at(idx++).toBool();
        // 从 elements 中获取 crow_indices
        auto& crow_indices = elements.at(idx++).toTensor();
        // 从 elements 中获取 col_indices
        auto& col_indices = elements.at(idx++).toTensor();
        // 从 elements 中获取 values_tensor
        auto& values_tensor = elements.at(idx++).toTensor();
        // 设置 options，指定布局为 SparseCsr，并设置是否需要梯度
        auto options = values_tensor.options()
                           .layout(c10::Layout::SparseCsr)
                           .requires_grad(requires_grad);
        // 调用 _sparse_csr_tensor_unsafe 函数创建稀疏 CSR 格式的 Tensor
        result = at::_sparse_csr_tensor_unsafe(
            crow_indices, col_indices, values_tensor, size, options);
        // 将 result 转换为需要梯度的变量，并重新赋给 result
        result =
            autograd::make_variable(std::move(result), options.requires_grad());
        // 退出 case 语句块
        break;
      }
      // 如果 layout 为其他未支持的格式，抛出错误
      default:
        TORCH_CHECK(
            false,
            "Unsupported sparse tensor layout type in serialization ",
            static_cast<c10::Layout>(layout));
        // 退出 switch 语句块
        break;
    }
    // 将处理完的 result 压入 stack_ 的末尾
    stack_.emplace_back(std::move(result));
  });
// 在 `Unpickler` 类中定义的成员函数 `rebuildTensor`，用于重建张量对象
void Unpickler::rebuildTensor(bool quantized) {
  // 将 lambda 函数放入 `globals_` 向量中，该 lambda 函数用于重建张量
  globals_.emplace_back([this, quantized] {
    // 从堆栈中弹出一个元组对象
    auto tup = pop(stack_).toTuple();
    // 获取元组的元素列表
    const auto& elements = tup->elements();
    size_t idx = 0;
    // 提取存储张量的第一个元素并转换为 Tensor 对象
    auto& storage_tensor = elements.at(idx++).toTensor();
    // 提取存储偏移量
    int64_t storage_offset = elements.at(idx++).toInt();
    // 将元组转换为表示大小的整数向量
    std::vector<int64_t> size = tupleToIntList(elements.at(idx++));
    // 将元组转换为表示步长的整数向量
    std::vector<int64_t> stride = tupleToIntList(elements.at(idx++));
    at::Tensor result;
    // 如果是量化张量
    if (quantized) {
      // 提取量化参数元组
      auto qparams_tuple = elements.at(idx++).toTuple();
      const auto& qparams = qparams_tuple->elements();
      // 提取量化方案并转换为枚举类型
      auto qscheme = static_cast<at::QScheme>(qparams.at(0).toInt());
      switch (qscheme) {
        // 如果是每个张量的仿射量化
        case at::kPerTensorAffine: {
          // 提取量化比例因子和零点偏移量
          double q_scale = qparams.at(1).toDouble();
          int64_t q_zero_point = qparams.at(2).toInt();
          // 使用仿射量化创建空张量
          result = at::_empty_affine_quantized(
              {0}, storage_tensor.options(), q_scale, q_zero_point);
        } break;
        // 如果是每个通道的仿射量化
        case at::kPerChannelAffineFloatQParams:
        case at::kPerChannelAffine: {
          // 提取量化比例和零点偏移张量
          const auto& scales = qparams.at(1).toTensor();
          const auto& zero_points = qparams.at(2).toTensor();
          int64_t axis = qparams.at(3).toInt();
          // 使用每个通道仿射量化创建空张量
          result = at::_empty_per_channel_affine_quantized(
              {0}, scales, zero_points, axis, storage_tensor.options());
        } break;
        // 不支持的量化类型，抛出异常
        default:
          TORCH_CHECK(
              false,
              "Unsupported tensor quantization type in serialization ",
              toString(qscheme));
          break;
      }
    } else {
      // 如果不是量化张量，创建空张量
      result = at::empty({0}, storage_tensor.options());
    }
    // 提取是否需要梯度的标志
    bool requires_grad = elements.at(idx++).toBool();
    // 跳过反向钩子为空的情况
    idx++; // backwards hooks is empty
    // 获取张量实现对象
    at::TensorImpl* impl = result.unsafeGetTensorImpl();
    // 设置张量实现对象的存储，保持数据类型不变
    impl->set_storage_keep_dtype(storage_tensor.storage());
    // 设置张量实现对象的存储偏移量
    impl->set_storage_offset(storage_offset);
    // 设置张量实现对象的大小和步长
    impl->set_sizes_and_strides(size, stride);
    // 将张量包装成可变张量，设置是否需要梯度
    result = autograd::make_variable(result, requires_grad);

    // 处理是否已经序列化了数学位信息
    // 见 `_reduce_ex_internal` 函数的 `args` 参数说明
    // 对于正常张量（最终 else 情况），旧版本未存储数学位信息，此时不进行操作
    // 注意：`math_bits` 是第 7 个参数
    // 注意：此功能仅适用于常规张量，不适用于量化张量，后者也有 7 个序列化参数
    if (!quantized && elements.size() == 7) {
      // 提取泛型字典对象作为数学位信息
      auto math_bits = elements.at(idx++).toGenericDict();
      // 设置张量的元数据
      torch::jit::setTensorMetadata(result, math_bits);
    }

    // 将处理后的结果张量压入堆栈
    stack_.emplace_back(std::move(result));
  });
}
void Unpickler::rebuildTensorFromTypeV2() {
    // [NOTE] skip_next_read_global
    // 当重新构建带有 Python 属性或子类化 Tensor 的 Tensor 时，
    // 在 `rebuildTensorFromTypeV2` 中栈上会收到 `(func, type(self), args, state)`，
    // 因此下一个对 readGlobal 的调用对应于 `func`，它是重建基础张量的函数。
    // 在 `func` 后的调用 readGlobal 对应于 Tensor 的 `type`，
    // 如果类型不是 `torch.Tensor`，会发出警告。
    this->skip_next_read_global = 2;  // 设置跳过下两次 readGlobal 的标志

    auto curr_globals_idx = globals_.size();  // 当前全局变量索引
    globals_.emplace_back([this, curr_globals_idx] {
        // args 是一个元组，包含以下数据：
        //  (用于重建基础张量的函数, Tensor 的类型,
        //   构造基础张量的参数, Python 状态（作为字典）)
        auto args = pop(stack_).toTuple();  // 从栈中弹出元组 args
        size_t tup_idx = 0;
        const auto args_elems = args->elements();
        auto base_tensor_args = args_elems.at(tup_idx + 2).toTuple();  // 获取基础张量的参数
        auto py_state = args_elems.at(tup_idx + 3).toGenericDict();  // 获取 Python 状态作为字典
        if (!py_state.empty()) {
            TORCH_WARN(
                "Loading Tensor with Python attributes will return at::Tensor with Python attributes being discarded");
        }
        // 调用函数来重建基础张量
        // 例如 `rebuildTensor`, `rebuildSpareTensor`.
        stack_.emplace_back(base_tensor_args);  // 将基础张量的参数压入栈中
        globals_[curr_globals_idx + 1]();  // 调用下一个全局函数
        stack_.emplace_back(pop(stack_));  // 将结果压回栈中
    });
}

#ifdef USE_RPC
void Unpickler::rebuildRRef() {
    globals_.emplace_back([this] {
        // 它与 Python 中的 rref 反序列化方式相同，
        // 参见 PyRRef::unpickle
        auto tuple = std::move(stack_.back()).toTuple();  // 从栈中取出元组
        const auto& args = tuple->elements();  // 获取元组的元素
        stack_.pop_back();  // 弹出栈顶元素

        TORCH_INTERNAL_ASSERT(
            args.size() == distributed::rpc::RFD_TUPLE_SIZE,
            "Pickled RRefForkData must contain 7 numbers.");

        auto ownerId =
            static_cast<int16_t>(args.at(distributed::rpc::OWNER_IDX).toInt());  // 获取 ownerId
        // const 引用会延长临时变量的生命周期
        const auto& rrefId = distributed::rpc::RRefId(
            static_cast<int16_t>(args.at(distributed::rpc::RREFID_ON_IDX).toInt()),  // 获取 rrefId
            static_cast<int64_t>(args.at(distributed::rpc::RREFID_ID_IDX).toInt()));
        const auto& forkId = distributed::rpc::RRefId(
            static_cast<int16_t>(args.at(distributed::rpc::FORKID_ON_IDX).toInt()),  // 获取 forkId
            static_cast<int64_t>(args.at(distributed::rpc::FORKID_ID_IDX).toInt()));
        auto parent =
            static_cast<int16_t>(args.at(distributed::rpc::PARENT_IDX).toInt());  // 获取 parent
        const auto& typeStr = static_cast<std::string>(
            args.at(distributed::rpc::TYPE_IDX).toStringRef());  // 获取 typeStr
        auto rrefForkData = distributed::rpc::RRefForkData(
            ownerId, rrefId, forkId, parent, typeStr);  // 构造 RRefForkData

        auto& ctx = distributed::rpc::RRefContext::getInstance();  // 获取 RRefContext 实例
        c10::intrusive_ptr<distributed::rpc::RRef> rref;  // 创建 RRef 智能指针
        TORCH_INTERNAL_ASSERT(
            type_resolver_ != nullptr, "type_resolver_ is nullptr.");


注释：
    # 使用类型解析器解析类型字符串，并创建 StrongTypePtr 对象
    at::StrongTypePtr type = type_resolver_(c10::QualifiedName(typeStr));

    # 根据 rrefForkData 和类型创建或获取 RRef 对象，并赋值给 rref
    rref = ctx.getOrCreateRRef(rrefForkData, type.type_);

    # 通知所有者和父节点关于分支的信息：分支ID、父节点和 RRef 对象
    ctx.notifyOwnerAndParentOfFork(
        rrefForkData.forkId_, rrefForkData.parent_, rref);

    # 将 RRef 对象转换为 c10::RRefInterface 类型并压入栈中
    stack_.emplace_back(
        c10::static_intrusive_pointer_cast<c10::RRefInterface>(rref));
  });

  # 将当前全局变量的索引值（globals_.size() 减去 1）压入栈中
  stack_.emplace_back(int64_t(globals_.size() - 1));

  # 函数执行完毕，返回
  return;
}
#endif

// 从输入流中以缓冲区方式读取一定数量的字节
void Unpickler::readSlowWithBuffer(char* dest, size_t sz) {
  // 首先从缓冲区中读取任何部分（可能为0）。
  // 我们显式假设 sz > buffer_remaining_，
  // 并且 sz 永远不会大于 buffer_.size()。
  AT_ASSERT(sz > buffer_remaining_);
  const size_t from_old_buf = buffer_remaining_;
  if (from_old_buf != 0) {
    memcpy(dest, buffer_.data() + buffer_pos_, from_old_buf);
  }
  const size_t needed = sz - from_old_buf;
  // 将数据完全读入缓冲区。这里的调用都显式假设一个缓冲区足够大来容纳任何 sz。
  AT_ASSERT(sz <= buffer_.size());
  buffer_remaining_ = reader_(buffer_.data(), buffer_.size());
  if (buffer_remaining_ < needed) {
    AT_ERROR("Unexpected end of pickler archive.");
  }
  memcpy(dest + from_old_buf, buffer_.data(), needed);
  buffer_pos_ = needed; // 赋值（从读取操作中归零）
  buffer_remaining_ -= needed;
}

// 从输入流中读取指定长度的字节并返回字符串
std::string Unpickler::readBytes(size_t length) {
  std::string data;
  static const size_t kSmallString = 64;
  TORCH_CHECK(
      length <= data.max_size(),
      "Parsing error: can't read ",
      length,
      " bytes to a string");
  if (length <= buffer_remaining_) {
    // 快速路径：完全在缓冲区内。
    data.assign(buffer_.data() + buffer_pos_, length);
    buffer_pos_ += length;
    buffer_remaining_ -= length;
  } else if (length <= kSmallString) {
    // 如果字符串比较小，进行完整的缓冲区读取，
    // 然后从该缓冲区中读取数据。
    data.resize(length);
    readSlowWithBuffer(&data[0], length);
  } else {
    // 否则，对于较大的字符串，尽可能从缓冲区中读取，
    // 然后直接读取到目标位置。
    const size_t from_old_buf = buffer_remaining_;
    if (from_old_buf != 0) {
      data.reserve(length);
      data.append(buffer_.data() + buffer_pos_, from_old_buf);
    }
    data.resize(length);
    const size_t needed = length - from_old_buf;
    size_t nread = reader_(&data[from_old_buf], needed);
    if (nread != needed) {
      AT_ERROR("Unexpected end of pickler archive.");
    }
    buffer_remaining_ = 0;
    // 当 buffer_remaining_ == 0 时，buffer_pos_ 没有意义。
  }
  return data;
}

void Unpickler::readListElements(IValue list_ivalue, size_t start) {
  auto num_elements = stack_.size() - start;
  auto elements = c10::ArrayRef<IValue>(stack_).slice(start);
  if (list_ivalue.isIntList()) {
    auto list = std::move(list_ivalue).toIntList();
    list.reserve(num_elements);
    for (const auto& elem : elements) {
      list.emplace_back(elem.toInt());
    }
  } else if (list_ivalue.isTensorList()) {
    auto list = std::move(list_ivalue).toTensorList();
    list.reserve(num_elements);
    for (const auto& elem : elements) {
      list.emplace_back(elem.toTensor());
    }
  } else if (list_ivalue.isDoubleList()) {
    auto list = std::move(list_ivalue).toDoubleList();
    list.reserve(num_elements);
    // 如果 IValue 是一个 Double 类型的列表
    for (const auto& elem : elements) {
      // 将每个元素转换为 Double 类型并添加到列表末尾
      list.emplace_back(elem.toDouble());
    }
  } else if (list_ivalue.isBoolList()) {
    // 如果 IValue 是一个布尔值类型的列表
    auto list = std::move(list_ivalue).toBoolList();
    // 预留足够的空间以容纳 num_elements 个元素
    list.reserve(num_elements);
    for (const auto& elem : elements) {
      // 将每个元素转换为布尔值并添加到列表末尾
      list.push_back(elem.toBool());
    }
  } else if (list_ivalue.isList()) {
    // 如果 IValue 是一个通用的列表
    auto list = std::move(list_ivalue).toList();
    // 预留足够的空间以容纳 num_elements 个元素
    list.reserve(num_elements);
    for (const auto& elem : elements) {
      // 将每个元素直接添加到列表末尾
      list.emplace_back(elem);
    }
  } else {
    // 如果 IValue 的类型未知，则抛出错误并显示其标签类型
    AT_ERROR("Unknown IValue list kind: ", list_ivalue.tagKind());
  }
  // 从 stack_ 容器中删除从 start 索引开始到末尾的元素
  stack_.erase(stack_.begin() + start, stack_.end());
// 弹出栈顶的 MARK，并将所有列表项追加到相应的列表中
void Unpickler::readList(IValue list_ivalue) {
  // 检查 marks_ 栈不为空，否则抛出解析错误
  TORCH_CHECK(!marks_.empty(), "Parsing error: marks_ is empty");
  // 记录当前列表的起始位置
  size_t start = marks_.back();
  // 弹出栈顶的 MARK
  marks_.pop_back();
  // 读取列表元素，并将其加入到给定的列表中
  readListElements(std::move(list_ivalue), start);
}

// 内联函数，检查字符是否是有效的 Python 标识符字符
inline bool is_valid_python_id_char(char c) {
  return c == '_' || c == '.' || (c >= '0' && c <= '9') ||
      (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}

// 读取以换行符结束的字符串
std::string Unpickler::readString() {
  std::string ss;
  while (true) {
    // 缓冲区中剩余数据的起始位置
    auto* const bufferStart = buffer_.data() + buffer_pos_;
    // 缓冲区中剩余的字节数
    const auto bufferLeft = buffer_.size() - buffer_pos_;
    // 在剩余数据中查找换行符的位置
    char* const newlinePtr =
        static_cast<char*>(memchr(bufferStart, '\n', bufferLeft));
    if (newlinePtr) {
      // 找到换行符，读取到换行符位置并结束
      auto const charsRead = newlinePtr - bufferStart;
      ss.append(bufferStart, charsRead);
      // 更新剩余缓冲区大小和位置
      buffer_remaining_ -= charsRead + 1;
      buffer_pos_ += charsRead + 1;
      break;
    } else {
      // 没有找到换行符，读取整个缓冲区，并重新填充缓冲区
      for (const char* p = bufferStart; p < bufferStart + bufferLeft; ++p) {
        // 简单检查，防止没有结束符 '\n'
        TORCH_CHECK(
            is_valid_python_id_char(*p),
            "Found character '",
            int(uint8_t(*p)),
            "' in string, ",
            "strings must be qualified Python identifiers");
      }
      // 将整个缓冲区内容追加到字符串中
      ss.append(bufferStart, bufferLeft);
      // 重新从输入流读取数据并更新缓冲区状态
      buffer_remaining_ = reader_(buffer_.data(), buffer_.size());
      buffer_pos_ = 0;
    }
  }
  return ss;
}
```