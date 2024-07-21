# `.\pytorch\torch\csrc\jit\mobile\compatibility\backport_manager.cpp`

```
// 包含头文件：ATen 库中的 IValue 类定义
#include <ATen/core/ivalue.h>
// 包含头文件：C10 异常处理工具
#include <c10/util/Exception.h>
// 包含头文件：Caffe2 序列化的文件适配器
#include <caffe2/serialize/file_adapter.h>
// 包含头文件：Caffe2 序列化的内联容器
#include <caffe2/serialize/inline_container.h>
// 包含头文件：Torch 移动端兼容性管理器
#include <torch/csrc/jit/mobile/compatibility/backport_manager.h>
// 包含头文件：Torch 移动端模型兼容性检查
#include <torch/csrc/jit/mobile/compatibility/model_compatibility.h>
// 包含头文件：Torch 移动端模型导入
#include <torch/csrc/jit/mobile/import.h>
// 包含头文件：Torch 移动端模块定义
#include <torch/csrc/jit/mobile/module.h>
// 包含头文件：Torch JIT 序列化导出
#include <torch/csrc/jit/serialization/export.h>
// 包含头文件：Torch JIT 序列化导入
#include <torch/csrc/jit/serialization/import.h>
// 包含头文件：Torch JIT Pickler
#include <torch/csrc/jit/serialization/pickler.h>
// 包含标准库头文件：cstddef（C++ 标准定义的各种重要类型和常量）
#include <cstddef>
// 包含标准库头文件：sstream（字符串流处理）
#include <sstream>

// 命名空间：torch::jit
namespace torch {
namespace jit {

// 使用语句：使用 caffe2::serialize 命名空间中的 IStreamAdapter 类
using caffe2::serialize::IStreamAdapter;
// 使用语句：使用 caffe2::serialize 命名空间中的 PyTorchStreamReader 类
using caffe2::serialize::PyTorchStreamReader;
// 使用语句：使用 caffe2::serialize 命名空间中的 PyTorchStreamWriter 类
using caffe2::serialize::PyTorchStreamWriter;

// 匿名命名空间中定义的常量：字节码版本号常量
namespace {
// 定义常量：字节码版本 V4
constexpr int64_t kBytecodeVersionV4 = 0x4L;
// 定义常量：字节码版本 V5
constexpr int64_t kBytecodeVersionV5 = 0x5L;
// 定义常量：字节码版本 V6
constexpr int64_t kBytecodeVersionV6 = 0x6L;
// 定义常量：字节码版本 V7
constexpr int64_t kBytecodeVersionV7 = 0x7L;
// 定义常量：字节码版本 V8
constexpr int64_t kBytecodeVersionV8 = 0x8L;
// 定义常量：字节码版本 V9
constexpr int64_t kBytecodeVersionV9 = 0x9L;
} // namespace

/********************** Utility Functions **********************/

// 匿名命名空间中定义的实用函数：有选择性地复制文件
namespace {
// 函数定义：有选择性地从源到目标复制文件，排除指定的文件和目录
void selective_copy(
    PyTorchStreamReader& reader,                     // 读取器对象引用
    PyTorchStreamWriter& writer,                     // 写入器对象引用
    const std::unordered_set<std::string>& excluded_files,  // 排除的文件集合
    const std::unordered_set<std::string>& excluded_dirs) { // 排除的目录集合

  // 获取所有记录（文件名列表）
  auto records = reader.getAllRecords();
  // 遍历每个记录
  for (const auto& record : records) {
    // 判断是否在排除的文件列表中
    bool skip = excluded_files.count(record) > 0;

    // 如果不在排除的文件列表中，则判断是否在排除的目录列表中
    for (const auto& excluded_dir : excluded_dirs) {
      // 查找记录中最后一个 '/' 或 '\\' 的位置
      std::size_t found = record.find_last_of("/\\");
      // 获取路径名，并与排除的目录进行比较
      auto path = record.substr(0, found);
      if (excluded_dir == path) {
        skip = true;
        break;
      }
    }

    // 如果不需要跳过，则从读取器中读取记录数据，并写入到写入器中
    if (!skip) {
      auto data_ptr = reader.getRecord(record);
      auto data = std::get<0>(data_ptr).get();
      auto size = std::get<1>(data_ptr);
      writer.writeRecord(record, data, size);
    }
  }
}

// 函数定义：用于写入当前版本的归档数据（字节码从 v5 到 v7 的情况）
void write_archive_current() {
  // 该函数用于处理字节码版本从 v5 到 v7 的序列化数据，处理逻辑未完整给出
}
// 这个函数用于将模型以旧格式写入归档文件。注意，这个函数可能会在 export_module.cpp 中发生变化，
// 但是我们没有保留旧的导出函数在代码库中的方式。为了能够以旧格式导出模型，我们在这里记录了导出函数的信息。
void write_archive_current(
    PyTorchStreamWriter& writer,                   // PyTorch 的写入流对象的引用，用于写入数据
    const IValue& value,                           // 要写入的值
    const std::string& archive_name,               // 归档文件名
    const std::string& archive_dir,                // 归档文件目录
    const std::string& tensor_dir,                 // 张量数据目录
    bool use_storage_context,                      // 是否使用存储上下文
    SerializationStorageContext& storage_context)  // 序列化存储上下文对象的引用
{
    std::vector<char> data;  // 用于存储数据的字符向量
    std::vector<c10::ClassTypePtr> memoizedClassTypes;  // 运行时类类型的向量，用于记录在 IValues 的 pickling 过程中的类型信息
    std::vector<std::string> tensor_names;  // 张量的名称列表
    Pickler data_pickle(  // Pickler 对象，用于将数据序列化为字节流
        [&](const char* buf, size_t size) {
            data.insert(data.end(), buf, buf + size);  // 将序列化后的数据插入到 data 向量中
        },
        nullptr,
        nullptr,
        &memoizedClassTypes,
        [&](const at::Tensor& tensor) {  // lambda 函数，用于处理张量对象
            if (use_storage_context) {
                std::string string_id = std::to_string(reinterpret_cast<std::intptr_t>(
                    tensor.storage().unsafeGetStorageImpl()));  // 获取张量的存储实现的唯一标识符
                tensor_names.push_back(string_id + ".storage");  // 将张量存储标识符添加到张量名称列表中
                storage_context.getOrAddStorage(tensor.storage());  // 获取或添加张量的存储对象到序列化存储上下文中
            } else {
                tensor_names.push_back(std::to_string(tensor_names.size()));  // 将张量的索引添加到张量名称列表中
            }
            return tensor_names.back();  // 返回最后一个张量名称作为标识符
        });
    data_pickle.protocol();  // 设置 Pickler 使用的协议版本
    data_pickle.pushIValue(value);  // 将要序列化的 IValue 值推送到 Pickler 中
    data_pickle.stop();  // 结束 Pickler 的操作

    size_t i = 0;
    std::string prefix = archive_name + "/";  // 归档名称的前缀

    // 断言张量名称列表的大小与 Pickler 中的张量数据大小相等
    TORCH_INTERNAL_ASSERT(tensor_names.size() == data_pickle.tensorData().size());

    // 获取已经序列化的文件列表
    const std::unordered_set<std::string>& pre_serialized_files =
        writer.getAllWrittenRecords();

    // 遍历 Pickler 中的张量数据
    for (const auto& td : data_pickle.tensorData()) {
        WriteableTensorData writable_td = getWriteableTensorData(td);  // 获取可写的张量数据
        std::string fname = tensor_dir + tensor_names[i++];  // 构造张量数据文件名
        // 如果使用存储上下文并且文件已经被序列化过，跳过写入操作
        if (use_storage_context &&
            pre_serialized_files.find(fname) != pre_serialized_files.end()) {
            continue;
        }
        writer.writeRecord(fname, writable_td.data(), writable_td.sizeInBytes());  // 写入张量数据到流中
    }

    std::string fname = archive_dir + archive_name + ".pkl";  // 归档文件的名称
    writer.writeRecord(fname, data.data(), data.size());  // 写入归档数据到流中
}

/*
inputs: 1) bytecode tuple from bytecode.pkl 2) the output bytecode version,
return: A boolean to indicate whether bytecode tuple is updated successfully
*/
bool update_bytecode_version(
    std::vector<at::IValue>& bytecode_values,  // 存储 bytecode.pkl 中的值的向量
    const int64_t to_version)  // 要更新的字节码版本号
{
    if (!bytecode_values.empty() && bytecode_values[0].isInt()) {
        bytecode_values[0] = c10::IValue(to_version);  // 更新第一个值为指定的字节码版本号
        return true;  // 返回更新成功
    }
    return false;  // 返回更新失败
}
    (('instructions',             # 定义指令序列的元组，用于描述程序执行的具体操作步骤
      (('STOREN', 1, 2),          # 子元组描述一个指令，STOREN 操作，参数为 1 和 2
       ('DROPR', 1, 0),           # 子元组描述一个指令，DROPR 操作，参数为 1 和 0
       ('MOVE', 2, 0),            # 子元组描述一个指令，MOVE 操作，参数为 2 和 0
       ('OP', 0, 0),              # 子元组描述一个指令，OP 操作，参数为 0 和 0
       ('RET', 0, 0))),           # 子元组描述一个指令，RET 操作，参数为 0 和 0
     ('operators',                # 定义操作符信息的元组，描述程序中使用的运算符类型
      (('aten::Int', 'Tensor'),)),# 子元组指示程序中使用的运算符类型为 aten::Int，作用于 Tensor 类型的数据
     ('constants', ()),           # 定义常量信息的元组，此处为空元组，表示没有常量定义
     ('types', ()),               # 定义类型信息的元组，此处为空元组，表示没有类型定义
     ('register_size', 2))))      # 定义寄存器大小为 2，表示程序执行时的寄存器数量
/*
std::stringstream update_bytecode_version(
    std::stringstream& input_model,
    const int64_t to_version) {
  // 使用输入的模型字符串流创建 PyTorchStreamReader 对象
  PyTorchStreamReader reader_bytecode(&input_model);
  // 从读取器中获取常量值并转换为元组
  auto constants_values =
      std::move(*readArchive(kArchiveNameConstants, reader_bytecode).toTuple())
          .elements();

  // 从读取器中获取字节码 IValue 的向量
  std::vector<IValue> bytecode_values = get_bytecode_ivalues(reader_bytecode);

  // 定义要排除的文件名集合
  std::unordered_set<std::string> excluded_files{
      "constants.pkl", "bytecode.pkl"};

  // 定义要排除的目录集合
  std::unordered_set<std::string> excluded_dirs{
      "constants",
      "bytecode",
  };

  // 创建输出模型的字符串流
  std::stringstream output_model_stream;
  // 定义写入函数，将数据写入输出模型流中
  auto writer_func = [&](const void* buf, size_t nbytes) -> size_t {
    output_model_stream.write(static_cast<const char*>(buf), nbytes);
    return !output_model_stream ? 0 : nbytes;
  };

  // 使用写入函数创建 PyTorchStreamWriter 对象
  PyTorchStreamWriter writer_bytecode(writer_func);

  // 选择性地复制模型内容到输出流，排除指定的文件和目录
  selective_copy(
      reader_bytecode, writer_bytecode, excluded_files, excluded_dirs);

  // 更新字节码的版本号
  update_bytecode_version(bytecode_values, to_version);
  // 创建字节码的元组对象
  auto bytecode_tuple = c10::ivalue::Tuple::create(std::move(bytecode_values));
  // 创建序列化存储上下文
  SerializationStorageContext storage_context;
  // 将常量值写入存档
  write_archive_current(
      writer_bytecode,
      c10::ivalue::Tuple::create(std::move(constants_values)),
      /*archive_name=*/"constants",
      /*archive_dir=*/"",
      /*tensor_dir=*/"constants/",
      /*use_storage_context=*/true,
      storage_context);
  // 将字节码元组写入存档
  write_archive_current(
      writer_bytecode,
      bytecode_tuple,
      /*archive_name=*/"bytecode",
      /*archive_dir=*/"",
      /*tensor_dir=*/"constants/",
      /*use_storage_context=*/true,
      storage_context);

  // 返回输出模型的字符串流
  return output_model_stream;
}
} // namespace

/******************** backport_v{i}_to_v{i-1} Functions **********************/
/*
 To add next backport function, for example, backport_vn_to_vn-1, create an
 anonymous namespace with a backport_vn_to_vn-1 function + other necessary
 customized function. If a function can be reused by other backport functions,
 move it to the utility function group. It will be easier to split out
 backport_manager.cpp to smaller files when it grows too long.

 How to add backport_v{i}_to_v{i-1} ?
 There are two options:
 1) [Format change only, recommended] Constrcut a reader with the
 input_model_stream, modify the file, and use PyTorchWriter to write it to
 output_model_stream. See backport_v5_to_v4.

 2) [Both format and content change] ]Use torch.jit.load() to load the stream,
 and save it to output_model_stream.

 The first option is preferred, because it will be purely format change, and
 the model doesn't need to go through inline again and model content will
 remain the same.

 A note for manipulate stringstream, it's recommend to declare a new
 stringstream, tmp_stream, and swap it with the argument output_model_stream
 once it's ready, output_model_stream.swap(tmp_stream). Do not use
 output_model_stream.clear(). It only clears out error state flag
 (https://www.cplusplus.com/reference/ios/ios/c/clear/), while the content is the
 same. It's cleaner to just declare a new one and swap.

*/

namespace {

/*
The following functions needed for backport model from v5 to v4.
Backport function bytecode v5 that deduplicate constanst table.
Previously, in v4, constant table will be exported twice, in both archive
bytecode and archive constants, and majority (almost all) are duplicates.
Currently, in v5, JIT and mobile will share archive constants, and all
constant tensors will be exported in this archive. The bump was needed
because the v5 bytecode export the tensor storage path in the schema, since
the runtime code is now able to query which archive this tensor is stored at
and query the correct archive.
For example, Previously, in v4, we deserialize tensor as without archive
path, and mobile will always read tensor from bytecode archive:
(torch._utils._rebuild_tensor_v2(pers.obj(('storage', torch.DoubleStorage,
'0', 'cpu', 8),),
   0,
   (2, 4),
   (4, 1),
   False,
   collections.OrderedDict()),
 1)),
 So, if the program defines: torch.add(x, h, out=x)
Currently, in v5, we deserialize the bytecode with the archive path, and
mobile can read tensor from the given path:
(torch._utils._rebuild_tensor_v2(pers.obj(('storage', torch.DoubleStorage,
'constants/0', 'cpu', 8),),
   0,
   (2, 4),
   (4, 1),
   False,
   collections.OrderedDict()),
 1)),
Thus, the backport is necessary such that the runtime can read tensor from
the correct archive.
*/
// 1) 从输入的模型流中读取 PyTorch 数据流
std::stringstream backport_v5_to_v4(std::stringstream& input_model_stream) {
  // 创建 PyTorchStreamReader 对象，用于读取输入模型流
  PyTorchStreamReader reader(&input_model_stream);
  // 获取模型字节码的 IValue 列表
  std::vector<IValue> bytecode_values = get_bytecode_ivalues(reader);
  // 读取常量值并转换为元组
  auto constants_values =
      std::move(*readArchive(kArchiveNameConstants, reader).toTuple())
          .elements();

  // 2) 复制除特定文件和目录外的所有内容到新的输出流中
  // 需要排除的文件集合
  std::unordered_set<std::string> excluded_files{
      "constants.pkl", "bytecode.pkl"};

  // 需要排除的目录集合
  std::unordered_set<std::string> excluded_dirs{
      "constants",
      "bytecode",
  };

  // 创建输出模型流
  std::stringstream output_model_stream;
  // 写入数据的 Lambda 函数
  auto writer_func = [&](const void* buf, size_t nbytes) -> size_t {
    output_model_stream.write(static_cast<const char*>(buf), nbytes);
    return !output_model_stream ? 0 : nbytes;
  };

  // 创建 PyTorchStreamWriter 对象，用于写入输出流
  PyTorchStreamWriter writer(writer_func);

  // 从输入流中选择性地复制文件和目录到输出流中
  selective_copy(reader, writer, excluded_files, excluded_dirs);

  // 3) 写入 `bytecode` 存档
  // 更新字节码版本为 v4
  update_bytecode_version(bytecode_values, kBytecodeVersionV4);
  // 创建包含所有字节码值的元组
  auto bytecode_tuple = c10::ivalue::Tuple::create(std::move(bytecode_values));

  // 导出函数，用于生成版本 4 的 bytecode.pkl
  auto writeArchiveV4 = [](PyTorchStreamWriter& writer,
                           const std::string& archive_name,
                           const c10::IValue& value) {
    // 用于存储序列化数据的向量
    std::vector<char> data;

    // 用于存储序列化过程中的运行时类类型
    std::vector<c10::ClassTypePtr> memoizedClassTypes;
    // 创建 Pickler 对象，并将数据填充到 data 向量中
    Pickler data_pickle(
        [&](const char* buf, size_t size) {
          data.insert(data.end(), buf, buf + size);
        },
        nullptr,
        nullptr,
        &memoizedClassTypes);
    // 设置序列化协议
    data_pickle.protocol();
    // 序列化 IValue
    data_pickle.pushIValue(value);
    data_pickle.stop();
    size_t i = 0;
    std::string prefix = archive_name + "/";

    // 将张量数据写入记录
    for (const auto& td : data_pickle.tensorData()) {
      WriteableTensorData writable_td = getWriteableTensorData(td);
      std::string fname = prefix + std::to_string(i++);
      writer.writeRecord(fname, writable_td.data(), writable_td.sizeInBytes());
    }
    // 写入文件记录
    std::string fname = archive_name + ".pkl";
    writer.writeRecord(fname, data.data(), data.size());
  };

  // 写入 `bytecode` 存档
  writeArchiveV4(writer, kArchiveNameBytecode, bytecode_tuple);
  // 写入 `constants` 存档
  auto constants_tuple =
      c10::ivalue::Tuple::create(std::move(constants_values));
  writeArchiveV4(writer, kArchiveNameConstants, constants_tuple);

  // 返回输出模型流
  return output_model_stream;
}
/*
Backporting functionality from bytecode version 6 to version 5.
This function takes a stringstream containing a serialized model and performs
the necessary transformations to emulate the behavior of bytecode version 5.

The function starts by creating a shared pointer to an IStreamAdapter using
the input_model_stream.
*/
std::stringstream backport_v6_to_v5(std::stringstream& input_model_stream) {
  std::shared_ptr<IStreamAdapter> rai =
      std::make_shared<IStreamAdapter>(&input_model_stream);
  // Create a PyTorchStreamReader instance using the IStreamAdapter
  auto reader = std::make_shared<PyTorchStreamReader>(rai);

  // Check if the original model file contains debug information related to bytecode
  bool hasBytecodeDebug = reader->hasRecord("mobile_debug_handles.pkl");

  // Extract all records from the PyTorchStreamReader, which represent various parts of the model
  auto records = reader->getAllRecords();
  ExtraFilesMap extra_files;
  // Iterate through the records to identify and collect files stored under 'extra' path
  for (const auto& record : records) {
    std::size_t found = record.find_last_of("/\\");
    auto path = record.substr(0, found);
    if ("extra" == path) {
      extra_files.emplace(record.substr(found + 1), "");
    }
  }

  // Load the TorchScript module using torch::jit::load, incorporating any extra files found
  Module torch_script = torch::jit::load(rai, c10::nullopt, extra_files);

  // Prepare for bytecode emission: set up BytecodeEmitModeGuard to handle default argument values
  std::stringstream intermediate_model_stream;
  {
    BytecodeEmitModeGuard argNumGuard(
        true /*emit_default_input_instructions*/,
        false /*enable_defaults_args_with_out_args*/,
        false /*enable_emit_promoted_ops*/);
    // Serialize the TorchScript module to intermediate_model_stream, enabling bytecode generation
    torch_script._save_for_mobile(
        intermediate_model_stream, extra_files, hasBytecodeDebug);
  }

  // Convert the bytecode version from 6 to 5 using update_bytecode_version function
  std::stringstream output_model_stream =
      update_bytecode_version(intermediate_model_stream, kBytecodeVersionV5);
  
  // Return the stringstream containing the backported model bytecode version 5
  return output_model_stream;
}
/*
std::stringstream backport_v7_to_v6(std::stringstream& input_model_stream) {
  // 创建一个共享指针，用于管理输入模型流的适配器
  std::shared_ptr<IStreamAdapter> rai =
      std::make_shared<IStreamAdapter>(&input_model_stream);
  // 创建 PyTorch 的流阅读器
  auto reader = std::make_shared<PyTorchStreamReader>(rai);
  // 读取存档中的常量数值并移动到元组
  auto constants_values =
      std::move(*readArchive(kArchiveNameConstants, *reader.get()).toTuple())
          .elements();

  // 如果原始模型文件中有调试信息文件，则在回溯的模型中也应该出现
  bool hasBytecodeDebug = reader->hasRecord("mobile_debug_handles.pkl");

  // 提取额外的文件并保存到 extra_files 映射中
  auto records = reader->getAllRecords();
  ExtraFilesMap extra_files;
  for (const auto& record : records) {
    std::size_t found = record.find_last_of("/\\");
    auto path = record.substr(0, found);
    if ("extra" == path) {
      extra_files.emplace(record.substr(found + 1), "");
    }
  }

  // 加载 TorchScript 模块以便进行回溯，因为需要重新发射字节码
  Module torch_script = torch::jit::load(rai, c10::nullopt, extra_files);

  // RAII 守卫用于修改标志，将 emit_default_input_instructions 设为 false，以保持字节码版本 6 中的行为
  // 同时将 enable_defaults_args_with_out_args 设为 false，以便序列化特定操作的数量，允许包含 out 参数和默认参数到 #all_args 中
  std::stringstream intermediate_model_stream;
  {
    BytecodeEmitModeGuard argNumGuard(
        false /*emit_default_input_instructions*/,
        false /*enable_defaults_args_with_out_args*/,
        false /*enable_emit_promoted_ops*/);
    // 将 TorchScript 模块保存为移动端格式到 intermediate_model_stream 中，带有额外文件和调试信息标志
    torch_script._save_for_mobile(
        intermediate_model_stream, extra_files, hasBytecodeDebug);
  }

  // 更新字节码版本号（从 7 更新到 6）
  std::stringstream output_model_stream =
      update_bytecode_version(intermediate_model_stream, kBytecodeVersionV6);
  // 返回更新后的输出模型流
  return output_model_stream;
}
*/
// 将给定的模型流回退至较旧的版本 V8 到 V7

std::stringstream backport_v9_to_v8(std::stringstream& input_model_stream) {
  // 创建额外文件映射
  ExtraFilesMap extra_files;
  // 加载 TorchScript 模块
  Module torch_script =
      torch::jit::load(input_model_stream, c10::nullopt, extra_files);
  // 创建中间模型流
  std::stringstream intermediate_model_stream;
  // TODO(@pavithran) : 检查是否有调试信息，并在回退时使用 load/save，暂时硬编码为 false 直到支持为止。
  bool hasBytecodeDebug = false;
  {
    // 设置字节码发射模式
    BytecodeEmitModeGuard argNumGuard(
        false /*emit_default_input_instructions*/,
        true /*enable_defaults_args_with_out_args*/,
        true /*enable_emit_promoted_ops*/);
    // 将模型保存为移动端可用格式
    torch_script._save_for_mobile(
        intermediate_model_stream,
        extra_files,
        hasBytecodeDebug,
        /*use_flatbuffer=*/false);
  }
  // 更新字节码版本（从 9 到 8）
  std::stringstream output_model_stream =
      update_bytecode_version(intermediate_model_stream, kBytecodeVersionV8);

  return output_model_stream;
}

// 将给定的模型流回退至较旧的版本 V7 到 V6

std::stringstream backport_v8_to_v7(std::stringstream& input_model_stream) {
  // 创建流适配器
  std::shared_ptr<IStreamAdapter> rai =
      std::make_shared<IStreamAdapter>(&input_model_stream);
  // 创建 PyTorch 读取器
  auto reader = std::make_shared<PyTorchStreamReader>(rai);
  // 获取所有记录
  auto records = reader->getAllRecords();
  // 检查是否有字节码调试信息
  bool hasBytecodeDebug = reader->hasRecord("mobile_debug_handles.pkl");
  // 创建额外文件映射
  ExtraFilesMap extra_files;
  // 遍历记录，根据路径添加额外文件
  for (const auto& record : records) {
    std::size_t found = record.find_last_of("/\\");
    auto path = record.substr(0, found);
    if ("extra" == path) {
      extra_files.emplace(record.substr(found + 1), "");
    }
  }
  // 加载 TorchScript 模块
  Module torch_script = torch::jit::load(rai, c10::nullopt, extra_files);
  // 创建中间模型流
  std::stringstream intermediate_model_stream;
  {
    // 设置字节码发射模式
    BytecodeEmitModeGuard argNumGuard(
        false /*emit_default_input_instructions*/,
        true /*enable_defaults_args_with_out_args*/,
        false /*enable_emit_promoted_ops*/);
    // 将模型保存为移动端可用格式
    torch_script._save_for_mobile(
        intermediate_model_stream, extra_files, hasBytecodeDebug);
  }

  // 更新字节码版本（从 8 到 7）
  std::stringstream output_model_stream =
      update_bytecode_version(intermediate_model_stream, kBytecodeVersionV7);

  return output_model_stream;
}

} // namespace

/********************** BackportManager **********************/

// 用于向前向后退移动字节码版本的通用合约
// Args:
// * PyTorchStreamReader 能访问 N 版本的输入模型
// * PyTorchStreamWriter 能访问回退到前一个 N-1 版本的输出模型
// 如果成功返回 true，否则返回 false。
using BytecodeBackportFunction =
    std::function<std::stringstream(std::stringstream&)>;
// BackportManager 类的构造函数，在实例化时注册了一系列的字节码回退函数
BackportManager::BackportManager() {
    registerBytecodeBackportFunction(kBytecodeVersionV5, backport_v5_to_v4);
    registerBytecodeBackportFunction(kBytecodeVersionV6, backport_v6_to_v5);
    registerBytecodeBackportFunction(kBytecodeVersionV7, backport_v7_to_v6);
    registerBytecodeBackportFunction(kBytecodeVersionV8, backport_v8_to_v7);
    registerBytecodeBackportFunction(kBytecodeVersionV9, backport_v9_to_v8);
}

// 返回字节码回退函数的静态映射表的引用，键是字节码版本，值是对应的回退函数
std::unordered_map<
    int64_t,
    std::function<std::stringstream(std::stringstream&)>>&
BackportManager::bytecodeBackportFunctions() const {
  static std::unordered_map<
      int64_t,
      std::function<std::stringstream(std::stringstream&)>>
      backport_functions;
  return backport_functions;
}

// 检查是否存在特定版本的字节码回退函数
bool BackportManager::hasBytecodeBackportFunction(
    const int64_t from_version) const {
  return bytecodeBackportFunctions().count(from_version);
}

// 注册字节码回退函数，如果已经注册则抛出错误
void BackportManager::registerBytecodeBackportFunction(
    const int64_t from_version,
    const BytecodeBackportFunction& backport_function) {
  TORCH_CHECK(
      !hasBytecodeBackportFunction(from_version),
      "Backporting from version ",
      from_version,
      " is already registered.");
  bytecodeBackportFunctions()[from_version] = backport_function;
}

// 主要的回退函数，从给定的字节流（oss）中回退模型数据，版本从 from_version 到 to_version
// 中间结果存储在 final_writer 中
bool BackportManager::backport(
    std::istream& oss,
    PyTorchStreamWriter& final_writer,
    int64_t from_version,
    int64_t to_version) const {
  if (from_version <= to_version) {
    TORCH_WARN(
        "backport doesn't support backporting model to new version. It's trying to backport from version ",
        from_version,
        " to version ",
        to_version);
    return false;
  }
  int64_t bytecode_version = from_version;
  bool backport_success = true;

  // 1) 给定 istream_adapter（一个可以访问输入模型的适配器，可以是从 istream 或文件中获取的模型），将所有模型内容复制到 stringstream 中
  oss.seekg(0, std::ios::beg);
  std::stringstream input_model_stream;
  input_model_stream << oss.rdbuf();
  std::stringstream output_model_stream;

  // 2) 回退模型，backport_v{i}_to_v{i-1} 函数的参数是 (input_model_stream 和 output_model_stream)
  while (bytecode_version > to_version) {
    // 如果不是第一次回退且 output_model_stream 有值，则交换 input 和 output
    if (!output_model_stream.str().empty()) {
      input_model_stream.swap(output_model_stream);
      // 重置 output_model_stream
      output_model_stream.str("");
    }

    // 如果不存在当前字节码版本的回退函数，则返回失败
    if (!hasBytecodeBackportFunction(bytecode_version)) {
      return false;
    }

    input_model_stream.seekg(0, input_model_stream.beg);
    // 获取输入模型的字节码版本
    auto input_model_stream_version =
        _get_model_bytecode_version(input_model_stream);
    // 检查输入模型流的字节码版本是否与期望的字节码版本相匹配
    if (static_cast<int64_t>(input_model_stream_version) != bytecode_version) {
      TORCH_WARN(
          "The bytecode version of input model stream is supposed to be ",
          bytecode_version,
          ", but it gets ",
          input_model_stream_version);
      // 如果不匹配，发出警告并返回false
      return false;
    }

    // 从当前字节码版本回溯到请求的版本
    std::stringstream backport_model_stream =
        bytecodeBackportFunctions()[bytecode_version--](input_model_stream);

    // 交换输出模型流和回溯后的模型流
    output_model_stream.swap(backport_model_stream);
    // 将输出模型流的读取位置设为开头
    output_model_stream.seekg(0, output_model_stream.beg);
    // 获取输出模型流的字节码版本
    auto output_model_stream_version =
        _get_model_bytecode_version(output_model_stream);

    // 检查输出模型流的字节码版本是否与期望的字节码版本相匹配
    if (static_cast<int64_t>(output_model_stream_version) != bytecode_version) {
      TORCH_WARN(
          "The bytecode version of output model stream is supposed to be ",
          bytecode_version,
          ", but it gets ",
          output_model_stream_version);
      // 如果不匹配，发出警告并返回false
      return false;
    }
  }

  // 3) 将最终的 output_model_stream 写入到 final_writer，final_writer 可以访问最终的模型目的地（文件、ostream 等）
  // 如果 output_model_stream 为空，发出警告并返回false
  if (output_model_stream.str().empty()) {
    TORCH_WARN("No output model from backport.");
    return false;
  }

  // 使用 PyTorchStreamReader 从 output_model_stream 中创建最终的模型读取器
  PyTorchStreamReader last_model_reader(&output_model_stream);
  // 选择性地将模型数据从 last_model_reader 复制到 final_writer 中
  selective_copy(
      last_model_reader,
      final_writer,
      std::unordered_set<std::string>(),
      std::unordered_set<std::string>());

  // 返回回溯成功的标志
  return backport_success;
}

} // namespace jit
} // namespace torch
```