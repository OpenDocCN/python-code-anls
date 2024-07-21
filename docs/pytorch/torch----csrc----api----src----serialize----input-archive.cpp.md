# `.\pytorch\torch\csrc\api\src\serialize\input-archive.cpp`

```py
// 包含 Torch 序列化输入存档所需的头文件
#include <torch/serialize/input-archive.h>

// 包含 Torch 的类型定义
#include <torch/types.h>
// 包含 Torch 的实用工具函数
#include <torch/utils.h>

// 包含 C10 库的异常处理工具
#include <c10/util/Exception.h>
// 包含 Caffe2 序列化读取适配器接口
#include <caffe2/serialize/read_adapter_interface.h>
// 包含 Torch JIT 模块 API 头文件
#include <torch/csrc/jit/api/module.h>
// 包含 Torch JIT 序列化导入工具
#include <torch/csrc/jit/serialization/import.h>

// 包含 C++ 标准库头文件
#include <istream>
#include <memory>
#include <string>
#include <utility>

// Torch 命名空间
namespace torch {
namespace serialize {

// 构造函数：创建一个名为 "Module" 的 JIT 编译单元的模块
InputArchive::InputArchive()
    : module_("Module", std::make_shared<jit::CompilationUnit>()) {}

// 读取指定键的属性值到给定的 c10::IValue 引用
void InputArchive::read(const std::string& key, c10::IValue& ivalue) {
  ivalue = module_.attr(key);
}

// 尝试读取指定键的属性值到给定的 c10::IValue 引用，返回是否成功
bool InputArchive::try_read(const std::string& key, c10::IValue& ivalue) {
  // 如果模块中不存在指定的键，则返回 false
  if (!module_.hasattr(key)) {
    return false;
  }
  // 从模块中读取指定键的属性值到给定的 c10::IValue 引用
  ivalue = module_.attr(key);
  return true;
}

// 尝试读取指定键的属性值到给定的 Tensor 引用，支持设备切换
bool InputArchive::try_read(
    const std::string& key,
    Tensor& tensor,
    bool is_buffer) {
  // 如果模块中不存在指定的键，则返回 false
  if (!module_.hasattr(key)) {
    return false;
  }
  // 从模块中读取指定键的属性值到临时变量 iv
  auto iv = module_.attr(key);
  // 如果属性值不是 Tensor 类型，则返回 false
  if (!iv.isTensor()) {
    return false;
  }
  // 将属性值转换为 Tensor 类型
  auto read_tensor = iv.toTensor();
  // 如果目标 Tensor 已经定义，则进行数据复制
  if (tensor.defined()) {
    torch::NoGradGuard guard;  // 禁用梯度计算
    // 如果目标 Tensor 的设备与读取的 Tensor 设备不同，则复制数据
    if (tensor.device() != read_tensor.device()) {
      tensor.set_data(read_tensor);
    } else {
      tensor.set_(read_tensor);  // 直接共享数据
    }
  } else {
    tensor = std::move(read_tensor);  // 直接移动数据
  }
  return true;
}

// 读取指定键的属性值到给定的 Tensor 引用，支持设备切换，否则抛出错误
void InputArchive::read(
    const std::string& key,
    Tensor& tensor,
    bool is_buffer) {
  // 尝试读取指定键的属性值到给定的 Tensor 引用，否则抛出错误信息
  TORCH_CHECK(
      try_read(key, tensor, is_buffer),
      "No such serialized tensor '",
      hierarchy_prefix_,
      key,
      "'");
}

// 尝试读取指定键的属性值为另一个 InputArchive 对象，如果不成功则返回 false
bool InputArchive::try_read(const std::string& key, InputArchive& archive) {
  // 如果模块中不存在指定的键，则返回 false
  if (!module_.hasattr(key)) {
    return false;
  }
  // 从模块中读取指定键的属性值到临时变量 iv
  auto iv = module_.attr(key);
  // 如果属性值不是 Module 类型，则返回 false
  if (!iv.isModule()) {
    return false;
  }
  // 将属性值转换为 Module 类型并赋值给 archive 的 module_
  archive.module_ = iv.toModule();
  // 更新子模块的层级前缀
  archive.hierarchy_prefix_ = hierarchy_prefix_ + key + ".";
  return true;
}

// 读取指定键的属性值为另一个 InputArchive 对象，如果不成功则抛出错误信息
void InputArchive::read(const std::string& key, InputArchive& archive) {
  // 尝试读取指定键的属性值为另一个 InputArchive 对象，否则抛出错误信息
  TORCH_CHECK(
      try_read(key, archive),
      "No such serialized submodule: '",
      hierarchy_prefix_,
      key,
      "'");
}

// 从文件加载模块的序列化数据，支持指定设备
void InputArchive::load_from(
    const std::string& filename,
    std::optional<torch::Device> device /*= c10::nullopt*/) {
  // 使用 Torch JIT 加载指定文件中的模块数据，可选指定设备
  module_ = torch::jit::load(filename, std::move(device));
}

// 从流中加载模块的序列化数据，支持指定设备
void InputArchive::load_from(
    std::istream& stream,
    std::optional<torch::Device> device /*= c10::nullopt*/) {
  // 使用 Torch JIT 加载流中的模块数据，可选指定设备
  module_ = torch::jit::load(stream, std::move(device));
}

// 从指定数据和大小加载模块的序列化数据，支持指定设备
void InputArchive::load_from(
    const char* data,
    size_t size,
    std::optional<torch::Device> device /*= c10::nullopt*/) {
  // 使用自定义的数据读取适配器加载模块的序列化数据
  using caffe2::serialize::ReadAdapterInterface;
  class OurAdapter : public ReadAdapterInterface {
   public:
    OurAdapter(const char* data, size_t size) : data_(data), size_(size) {}
    // 实现读取适配器接口中的数据大小函数
    size_t size() const override {
      return size_;
    }
    // 私有数据成员：指向数据和大小
   private:
    const char* data_;
    size_t size_;
  };
    size_t read(uint64_t pos, void* buf, size_t n, const char* what = "")
        const override {
      (void)what;  // 忽略未使用的参数 `what`
      if (pos >= size_) {  // 如果请求读取的位置超过数据大小
        return 0;  // 返回读取数据量为 0
      }
      // 计算实际可读取的数据量，避免越界访问
      size_t nread = std::min(static_cast<size_t>(pos) + n, size_) - pos;
      // 将数据从存储位置复制到目标缓冲区
      memcpy(buf, data_ + pos, nread);
      return nread;  // 返回实际读取的数据量
    }

   private:
    const char* data_;  // 数据的起始位置
    size_t size_;  // 数据的总大小
  };
  module_ = torch::jit::load(
      std::make_unique<OurAdapter>(data, size), std::move(device));  // 加载 Torch 模块
}

// 定义类 InputArchive 的成员函数 load_from，用于从自定义的读取和大小函数加载模型
void InputArchive::load_from(
    const std::function<size_t(uint64_t, void*, size_t)>& read_func,
    const std::function<size_t(void)>& size_func,
    std::optional<torch::Device> device /*= c10::nullopt*/) {
  // 使用 caffe2 序列化库中的 ReadAdapterInterface
  class OurAdapter : public ReadAdapterInterface {
   public:
    // 构造函数，初始化读取和大小函数
    OurAdapter(
        const std::function<size_t(uint64_t, void*, size_t)>& read_func,
        const std::function<size_t(void)>& size_func)
        : read_func_(read_func), size_func_(size_func) {}
    // 实现接口函数 size()，返回模型的大小
    size_t size() const override {
      return size_func_();
    }
    // 实现接口函数 read()，从指定位置读取数据到缓冲区
    size_t read(uint64_t pos, void* buf, size_t n, const char* what = "")
        const override {
      (void)what; // 防止未使用警告
      return read_func_(pos, buf, n);
    }

   private:
    const std::function<size_t(uint64_t, void*, size_t)>& read_func_;
    const std::function<size_t(void)>& size_func_;
  };
  // 调用 torch 库的 jit::load 函数加载模型，传入自定义的 OurAdapter 对象和设备信息
  module_ = torch::jit::load(
      std::make_unique<OurAdapter>(read_func, size_func), std::move(device));
}

// 定义类 InputArchive 的成员函数 keys，返回模型的所有属性名列表
std::vector<std::string> InputArchive::keys() {
  std::vector<std::string> all_keys;
  // 预留足够的空间以容纳模型的所有属性名
  all_keys.reserve(module_.named_attributes(/*recurse=*/false).size());

  // 遍历模型的属性名列表，将每个属性名加入到 all_keys 中
  for (const torch::jit::NameValue& s :
       module_.named_attributes(/*recurse=*/false)) {
    all_keys.push_back(s.name);
  }

  // 返回包含所有属性名的列表
  return all_keys;
}

// 命名空间 serialize 的结束标记
} // namespace serialize
// 命名空间 torch 的结束标记
} // namespace torch
```