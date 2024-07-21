# `.\pytorch\torch\csrc\jit\mobile\train\export_data.cpp`

```
/**
 * 包含 Torch 库中移动端训练所需的头文件
 */
#include <torch/csrc/jit/mobile/train/export_data.h>

/**
 * 包含 Torch 库中移动端导入导出的通用函数和结构
 */
#include <torch/csrc/jit/mobile/import_export_common.h>

/**
 * 包含 Torch 库中移动端模块的定义和操作
 */
#include <torch/csrc/jit/mobile/module.h>

/**
 * 包含 Torch 库中 JIT 运行时指令的定义
 */
#include <torch/csrc/jit/runtime/instruction.h>

/**
 * 包含 Torch 库中的 FlatBuffer 序列化器
 */
#include <torch/csrc/jit/serialization/flatbuffer_serializer.h>

/**
 * 包含 Torch 库中的 Pickler 序列化器
 */
#include <torch/csrc/jit/serialization/pickler.h>

/**
 * 包含 Torch 库中类型名称唯一化的功能
 */
#include <torch/csrc/jit/serialization/type_name_uniquer.h>

/**
 * 包含 Caffe2 库中的内联容器功能
 */
#include <caffe2/serialize/inline_container.h>

/**
 * 包含 ATen 库中 IValue 类型的定义
 */
#include <ATen/core/ivalue.h>

/**
 * 包含 ATen 库中 JIT 类型的定义
 */
#include <ATen/core/jit_type.h>

/**
 * 包含标准字符串处理功能
 */
#include <string>

/**
 * 包含标准向量功能
 */
#include <vector>

namespace torch {
namespace jit {
namespace mobile {

/**
 * 将 OpCode 枚举转换为字符串表示
 */
char const* toString(OpCode op);

/**
 * 匿名命名空间，实现 IValuePickler 类，用于将 IValue 序列化为 Pickle 格式并存入 ZIP 包中的文件 "data.pkl"
 */
namespace {

/**
 * 实现 IValuePickler 类，用于将 IValue 序列化为 Pickle 格式并存入 ZIP 包中的文件 "data.pkl"
 */
class IValuePickler final {
 public:
  /**
   * 构造函数，接受文件名并初始化 PyTorchStreamWriter 对象
   */
  explicit IValuePickler(const std::string& filename) : writer_(filename) {}

  /**
   * 构造函数，接受写入函数并初始化 PyTorchStreamWriter 对象
   */
  explicit IValuePickler(
      const std::function<size_t(const void*, size_t)>& writer_func)
      : writer_(writer_func) {}

  /**
   * 序列化给定的 IValue 对象
   */
  void serialize(const IValue& object) {
    // 仅序列化数据部分
    writeArchive("data", object);
  }

 private:
  /**
   * 写入指定名称的存档及其对应的 IValue 数据
   */
  void writeArchive(const std::string& archive_name, const IValue& value) {
    std::vector<char> data;
    // 用于记录在 Pickling 过程中运行时类类型的向量
    std::vector<c10::ClassTypePtr> memoizedClassTypes;
    Pickler data_pickle(
        [&](const char* buf, size_t size) {
          data.insert(data.end(), buf, buf + size);
        },
        nullptr,
        [&](const c10::ClassTypePtr& t) {
          return type_name_uniquer_.getUniqueName(t);
        },
        &memoizedClassTypes);
    data_pickle.protocol();
    data_pickle.pushIValue(value);
    data_pickle.stop();
    size_t i = 0;
    std::string prefix = archive_name + "/";
    // 遍历数据中的张量数据并写入记录
    for (const auto& td : data_pickle.tensorData()) {
      WriteableTensorData writable_td = getWriteableTensorData(td);
      std::string fname = prefix + std::to_string(i++);
      writer_.writeRecord(fname, writable_td.data(), writable_td.sizeInBytes());
    }
    std::string fname = archive_name + ".pkl";
    writer_.writeRecord(fname, data.data(), data.size());
  }

  // PyTorchStreamWriter 对象，用于写入数据流
  caffe2::serialize::PyTorchStreamWriter writer_;

  // 类型名称唯一化工具对象
  TypeNameUniquer type_name_uniquer_;
};

} // namespace

/**
 * 将给定的命名张量映射转换为 c10::Dict 类型
 */
c10::Dict<std::string, at::Tensor> tensor_map_to_dict(
    const std::map<std::string, at::Tensor>& map) {
  c10::Dict<std::string, at::Tensor> dict;
  // 遍历映射，将每个命名张量插入字典
  for (const auto& e : map) {
    dict.insert(e.first, e.second);
  }
  return dict;
}

/**
 * 返回一个移动端模块，具有一个指定属性名的属性，其值为提供的字典
 */
mobile::Module tensor_dict_to_mobile(
    // 创建一个用于支持模块的对象，包含一个用于保存字典的属性
    auto cu = std::make_shared<torch::jit::CompilationUnit>();
    
    // 注意，类的名称并不重要，但必须以"__torch__."开头，才能在导入时被视为有效的类
    auto cls = c10::ClassType::create(
        "__torch__.SavedParameters", cu, /*is_module=*/true);
    
    // 向类添加一个属性，用于保存字典
    cls->addAttribute(
        internal::kSavedParametersAttributeName,
        c10::DictType::create(dict.keyType(), dict.valueType()));
    
    // 创建一个包含上述类的对象，这个对象将被包装在一个模块中
    auto object = c10::ivalue::Object::create(
        c10::StrongTypePtr(std::move(cu), std::move(cls)), /*numSlots=*/1);
    
    // 将字典作为一个属性添加到对象中
    object->setAttr(internal::kSavedParametersAttributeName, dict);
    
    // 用一个 CompilationUnit 对象包装这个对象，最终返回一个 mobile::Module 对象
    auto mcu = std::make_shared<mobile::CompilationUnit>();
    return mobile::Module(object, mcu);
} // namespace mobile



} // namespace mobile

// 结束 mobile 命名空间的声明


void (*_save_mobile_module_to)(
    const mobile::Module& module,
    const std::function<size_t(const void*, size_t)>& writer_func) = nullptr;


void (*_save_mobile_module_to)(
    const mobile::Module& module,
    const std::function<size_t(const void*, size_t)>& writer_func) = nullptr;

// 声明一个函数指针 `_save_mobile_module_to`，用于保存移动模块的操作函数，初始值为 `nullptr`


void _save_parameters(
    const std::map<std::string, at::Tensor>& map,
    std::ostream& out,
    bool use_flatbuffer) {


void _save_parameters(
    const std::map<std::string, at::Tensor>& map,
    std::ostream& out,
    bool use_flatbuffer) {

// 定义函数 `_save_parameters`，接受一个 tensor 的映射 `map`、一个输出流 `out` 和一个布尔值 `use_flatbuffer`


  auto dict = mobile::tensor_map_to_dict(map);


  auto dict = mobile::tensor_map_to_dict(map);

// 使用 `mobile` 命名空间中的函数 `tensor_map_to_dict` 将输入的 tensor 映射转换为字典 `dict`


  auto write_func = [&out](const void* buf, size_t nbytes) -> size_t {
    out.write(
        static_cast<const char*>(buf), static_cast<std::streamsize>(nbytes));
    return !out ? 0 : nbytes;
  };


  auto write_func = [&out](const void* buf, size_t nbytes) -> size_t {
    out.write(
        static_cast<const char*>(buf), static_cast<std::streamsize>(nbytes));
    return !out ? 0 : nbytes;
  };

// 定义一个 lambda 函数 `write_func`，用于将指定的字节数据写入输出流 `out`


  if (use_flatbuffer) {
    save_mobile_module_to_func(mobile::tensor_dict_to_mobile(dict), write_func);
  } else {


  if (use_flatbuffer) {
    save_mobile_module_to_func(mobile::tensor_dict_to_mobile(dict), write_func);
  } else {

// 如果 `use_flatbuffer` 为真，则调用 `save_mobile_module_to_func` 函数，将转换后的移动模块字典写入输出流


    // For Pickle, we only serialize the dict itself.
    mobile::IValuePickler pickler(write_func);
    pickler.serialize(dict);


    // For Pickle, we only serialize the dict itself.
    mobile::IValuePickler pickler(write_func);
    pickler.serialize(dict);

// 否则，使用 Pickle 方式，创建 `IValuePickler` 对象并序列化字典 `dict`


void _save_parameters(
    const std::map<std::string, at::Tensor>& map,
    const std::string& filename,
    bool use_flatbuffer) {


void _save_parameters(
    const std::map<std::string, at::Tensor>& map,
    const std::string& filename,
    bool use_flatbuffer) {

// 重载函数 `_save_parameters`，接受一个 tensor 映射 `map`、一个文件名 `filename` 和一个布尔值 `use_flatbuffer`


  std::ofstream ifile(filename);
  _save_parameters(map, ifile, use_flatbuffer);


  std::ofstream ifile(filename);
  _save_parameters(map, ifile, use_flatbuffer);

// 打开输出文件流 `ifile`，然后调用前面定义的 `_save_parameters` 函数，将数据保存到文件中


} // namespace jit
} // namespace torch


} // namespace jit
} // namespace torch

// 结束 `jit` 和 `torch` 命名空间的声明
```