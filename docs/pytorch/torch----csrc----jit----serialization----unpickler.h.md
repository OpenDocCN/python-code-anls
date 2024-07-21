# `.\pytorch\torch\csrc\jit\serialization\unpickler.h`

```
// 一次包含头文件#pragma once，确保本头文件只被编译一次
#include <ATen/core/ivalue.h>
#include <c10/util/ArrayRef.h>
#include <caffe2/serialize/inline_container.h>
#include <torch/csrc/Export.h>
#include <torch/csrc/jit/frontend/script_type_parser.h>
#include <torch/csrc/jit/serialization/pickler.h>

namespace torch::jit {

// 使用别名TypeResolver表示一个std::function，该函数接受一个QualifiedName参数并返回c10::StrongTypePtr
using TypeResolver =
    std::function<c10::StrongTypePtr(const c10::QualifiedName&)>;

// 使用别名ObjLoader表示一个std::function，该函数接受at::StrongTypePtr和IValue参数并返回c10::intrusive_ptr<c10::ivalue::Object>
using ObjLoader = std::function<
    c10::intrusive_ptr<c10::ivalue::Object>(const at::StrongTypePtr&, IValue)>;

class DeserializationStorageContext;

// [unpickler refactor] there is some cruft around PickleOpCode::BUILD,
// PickleOpCode::NEWOBJ, and the last_opcode_ member below that should be
// deleted at some point, the Pickler doesn't produce it and it's only around to
// support models saved before 1.1
// 未提供完整的代码内容，此处需要进一步补充以完善注释

// 设置私有成员version_为version_number，用于记录反序列化器的版本信息
version_ = version_number;
}

static c10::TypePtr defaultTypeParser(const std::string& str) {
  // 创建ScriptTypeParser对象parser
  ScriptTypeParser parser;
  // 调用parser的parseType方法，解析给定字符串str，返回解析后的类型对象
  return parser.parseType(str);
}

private:
// 没有参数确保模板参数必须被指定，以便明确读取的字节数/类型
template <typename T>
T read() {
  // 声明变量item
  T item;
  // 如果T的大小不超过buffer_remaining_，从缓冲区中快速读取
  if (sizeof(T) <= buffer_remaining_) {
    // 使用memcpy从buffer_的当前位置buffer_pos_复制sizeof(T)字节到item
    memcpy(&item, buffer_.data() + buffer_pos_, sizeof(T));
    // 减少buffer_remaining_以反映已读取的字节数
    buffer_remaining_ -= sizeof(T);
    // 增加buffer_pos_以指向下一个未读取的位置
    buffer_pos_ += sizeof(T);
  } else {
    // 否则调用readSlowWithBuffer方法，以缓慢方式从缓冲区读取数据
    readSlowWithBuffer(reinterpret_cast<char*>(&item), sizeof(T));
  }
  // 返回读取的数据item
  return item;
}

// 声明readSlowWithBuffer方法，用于从缓冲区读取指定大小的数据
void readSlowWithBuffer(char* dest, size_t sz);
// 声明readBytes方法，用于从缓冲区读取指定数量的字节，并返回读取的字节内容作为std::string
std::string readBytes(size_t num_bytes);

// 声明readFloat方法，用于从缓冲区读取一个double类型的浮点数，并返回读取的浮点数
double readFloat();

// 声明readGlobal方法，接受模块名和类名作为参数，用于重建全局对象
void readGlobal(
    const std::string& module_name,
    const std::string& class_name);

// 声明rebuildTensor方法，接受一个布尔值quantized，用于重建张量对象
void rebuildTensor(bool quantized);

// 声明rebuildTensorFromTypeV2方法，用于从类型V2重建张量对象
void rebuildTensorFromTypeV2();

// 声明rebuildSparseTensor方法，用于重建稀疏张量对象
void rebuildSparseTensor();

#ifdef USE_DISTRIBUTED
// 如果定义了USE_DISTRIBUTED宏，则声明rebuildRRef方法，用于重建RRef对象
void rebuildRRef();
#endif

// 声明readInstruction方法，用于从缓冲区读取并返回PickleOpCode指令
PickleOpCode readInstruction();

// 声明readOpCode方法，用于从缓冲区读取并返回PickleOpCode操作码
PickleOpCode readOpCode() {
    // 返回一个 static_cast 后的 PickleOpCode 类型，通过调用 read<uint8_t>() 读取的结果
    return static_cast<PickleOpCode>(read<uint8_t>());
    }
    
    // 声明一个 readString 函数，具体实现在其他地方
    std::string readString();
    
    // 声明一个 readList 函数，接受一个 IValue 参数 list_ivalue，具体实现在其他地方
    void readList(IValue list_ivalue);
    
    // 声明一个 readListElements 函数，接受一个 IValue 参数 list_ivalue 和一个 size_t 参数 start，具体实现在其他地方
    void readListElements(IValue list_ivalue, size_t start);
    
    // 设置输入的 memo_id，具体实现在其他地方
    void setInput(size_t memo_id);
    
    // 运行函数 run，具体实现在其他地方
    void run();
    
    // 声明一个 std::function，用于读取数据，返回读取的字节数，不直接调用 reader_
    std::function<size_t(char*, size_t)> reader_;
    
    // 用于避免每个字节都调用 reader_ 的小缓冲区
    std::array<char, 256> buffer_;
    
    // 缓冲区的当前位置
    size_t buffer_pos_{0};
    
    // 缓冲区中剩余的字节数
    size_t buffer_remaining_{0};
    
    // 栈，存储 IValue 元素
    std::vector<IValue> stack_;
    
    // 全局变量的函数列表，以 IValue 整数索引形式表示在栈中
    std::vector<std::function<void(void)>> globals_;
    
    // 备忘录表，存储 IValue 元素
    std::vector<IValue> memo_table_;
    
    // 标记列表，存储 size_t 类型
    std::vector<size_t> marks_;
    
    // tensor 表的常量引用
    c10::ArrayRef<at::Tensor> tensor_table_;
    
    // 在列表和字典类型反序列化时，缓存类型信息，避免多次解析相同类型
    std::unordered_map<std::string, c10::TypePtr> type_cache_;
    
    // 可选的类型解析器指针，用于创建类时需要存在
    TypeResolver type_resolver_;
    
    // ObjLoader 对象，用于加载对象
    ObjLoader obj_loader_;
    
    // 空元组的 IValue
    IValue empty_tuple_;
    
    // 读取记录的函数对象，返回 at::DataPtr，通过传递字符串参数
    std::function<at::DataPtr(const std::string&)> read_record_;
    
    // 可选的设备对象，用于指定数据的设备
    std::optional<at::Device> device_;
    
    // 当设置为 true 时，Unpickler 将忽略 pickled 设备，使用 read_record_ 函数返回的 DataPtr 的设备
    // 默认值为 false
    const bool use_storage_device_;
    
    // 类型解析器对象，用于解析类型标签
    TypeParserT type_parser_{defaultTypeParser};
    
    // 用于 torch.package，启用跨 ScriptModules 和 eager 模块共享存储的上下文对象
    std::shared_ptr<DeserializationStorageContext> storage_context_;
    
    // 版本号，用于类型标签序列化
    uint64_t version_;
    
    // 控制标志，当设置为 1 时，Unpickler 将跳过下一个 read_global 操作
    uint8_t skip_next_read_global = 0;
};

void restoreAccurateTypeTags(const IValue& root, const c10::TypePtr& type_tag);



// 结束命名空间 torch::jit
} // namespace torch::jit

// 声明函数 restoreAccurateTypeTags，用于恢复准确的类型标签
void restoreAccurateTypeTags(const IValue& root, const c10::TypePtr& type_tag);
```