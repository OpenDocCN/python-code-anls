# `.\pytorch\torch\csrc\jit\runtime\argument_spec.h`

```py
#pragma once

#include <ATen/core/jit_type.h>  // 包含 ATen 核心 JIT 类型定义
#include <ATen/core/stack.h>     // 包含 ATen 核心栈定义
#include <c10/util/hash.h>       // 包含 C10 实用工具中的哈希功能
#include <c10/util/irange.h>     // 包含 C10 实用工具中的范围迭代功能
#include <torch/csrc/Export.h>   // Torch 导出相关定义
#include <torch/csrc/autograd/variable.h>  // Torch 自动求导变量定义
#include <torch/csrc/jit/ir/ir.h>          // Torch JIT IR 相关定义
#include <ostream>               // 包含输出流定义
#include <vector>                // 包含向量容器定义

C10_CLANG_DIAGNOSTIC_PUSH()       // 使用 clang 编译器，推入诊断设置
#if C10_CLANG_HAS_WARNING("-Wshorten-64-to-32")
C10_CLANG_DIAGNOSTIC_IGNORE("-Wshorten-64-to-32")  // 忽略 64 位转 32 位的警告
#endif

namespace torch::jit {

// GraphExecutor 为不同维度和类型的输入创建 Graph 的特化版本。
struct ArgumentInfo {
  friend struct ArgumentSpec;
  using plain_data_type = uint64_t;  // 定义 plain_data_type 为 uint64_t 类型

  bool defined() const {  // 检查是否已定义
    return defined_;       // 返回 defined_ 成员的值
  }
  at::Device device() const {  // 获取设备信息
    return at::Device(DeviceType(dev_type_), device_);  // 使用设备类型和设备索引构造 Device 对象
  }
  // XXX: 保证在非张量参数上调用时返回 false
  bool requires_grad() const {  // 检查是否需要梯度
    return requires_grad_;      // 返回 requires_grad_ 成员的值
  }
  int dim() const {  // 获取张量维度
    // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
    return dim_;    // 返回 dim_ 成员的值（忽略类型转换警告）
  }
  at::ScalarType type() const {  // 获取张量标量类型
    return at::ScalarType(type_);  // 返回 type_ 成员的值
  }
  TypePtr toType() const {  // 转换为 TypePtr 类型
    if (!defined())           // 如果未定义
      return TensorType::get();  // 返回默认的 TensorType

    return TensorType::create(     // 创建具体的 TensorType 对象
        type(), device(), std::optional<size_t>(dim()), requires_grad());
  }
  operator TypePtr() const {  // 转换为 TypePtr 类型
    return toType();          // 调用 toType() 方法
  }

 private:
  unsigned defined_ : 1;    // 定义为 1 位的 defined_ 成员
  unsigned requires_grad_ : 1;  // 定义为 1 位的 requires_grad_ 成员
  unsigned : 5;             // 占用 5 位，未命名的位域
  unsigned dim_ : 8;        // 定义为 8 位的 dim_ 成员
  unsigned device_ : 8;     // 定义为 8 位的 device_ 成员
  unsigned type_ : 8;       // 定义为 8 位的 type_ 成员
  unsigned dev_type_ : 16;  // 定义为 16 位的 dev_type_ 成员
  unsigned : 16;            // 占用 16 位，未命名的位域
};

static_assert(
    std::is_standard_layout<ArgumentInfo>::value,
    "ArgumentInfo is to be a POD struct");  // 断言 ArgumentInfo 是标准布局的 POD 结构体
static_assert(
    sizeof(ArgumentInfo) == sizeof(ArgumentInfo::plain_data_type),
    "ArgumentInfo is expected to be a 32-bit struct");  // 断言 ArgumentInfo 应为 32 位结构体

struct ArgumentSpec {
  ArgumentSpec(size_t num_flat_tensor_inputs, size_t num_flat_optional_inputs)
      : hash_code(c10::hash_combine(
            num_flat_tensor_inputs,
            num_flat_optional_inputs)) {  // 构造函数，初始化 hash_code 和容器大小
    tensor_args.reserve(num_flat_tensor_inputs);         // 预留 tensor_args 容量
    optional_presence.reserve(num_flat_optional_inputs);  // 预留 optional_presence 容量
  }

  void addOptional(const IValue& input) {  // 添加可选参数
    bool is_present = !input.isNone();     // 检查输入是否为 None
    optional_presence.push_back(is_present);  // 将是否存在的信息加入容器
    hash_code = c10::hash_combine(hash_code, is_present);  // 更新哈希码
  }

  void addTensor(const IValue& input, bool with_grad) {  // 添加张量参数
    AT_ASSERT(input.isTensor(), "Expected Tensor but found ", input.tagKind());  // 断言输入为 Tensor
    tensor_args.emplace_back();               // 在 tensor_args 中添加新元素
    auto& arg = tensor_args.back();           // 获取最后一个元素的引用
    // 将所有字段初始化为 0。这很方便，因为例如 requires_grad() 可以在张量上检查，也会使填充位全部为 0。
    std::memset(&arg, 0, sizeof(ArgumentInfo));  // 使用 0 初始化 ArgumentInfo 结构体

    // [argspec refcounting] reinterpret the IValue to avoid having to refcount
    // the Tensor microbenchmarks
    // https://github.com/zdevito/pytorch/commit/21e7200a0a0fc456bea2f10e95b1781f83933d10
  // 将输入参数转换为 const at::Tensor 指针
  const at::Tensor* t = reinterpret_cast<const at::Tensor*>(&input);
  // 将参数的定义状态复制给 arg 对象
  arg.defined_ = t->defined();
  // 如果参数已定义
  if (arg.defined_) {
    // 根据需求设置 requires_grad_
    arg.requires_grad_ = with_grad && autograd::Variable(*t).requires_grad();
    // 设置参数的维度
    arg.dim_ = t->dim();
    // 获取参数的设备类型
    at::Device device = t->device();
    // 将设备类型转换为底层类型并赋值给 dev_type_
    arg.dev_type_ = static_cast<std::underlying_type<DeviceType>::type>(device.type());
    // 将设备索引赋值给 device_
    arg.device_ = device.index();
    // 将参数的数据类型转换为无符号整数并赋值给 type_
    arg.type_ = static_cast<unsigned>(t->scalar_type());
  }
  // 将 arg 对象与当前对象的哈希码组合
  combineHash(arg);
}

// 将 ArgumentInfo 结构体的数据复制到 plain_data_type 中，并更新哈希码
void combineHash(const ArgumentInfo& arg) {
  ArgumentInfo::plain_data_type arg_data;
  std::memcpy(&arg_data, &arg, sizeof(ArgumentInfo));
  hash_code = c10::hash_combine(hash_code, arg_data);
}

// 比较两个 ArgumentSpec 对象是否相等
bool operator==(const ArgumentSpec& spec) const {
  // 如果 optional_presence 不同，则对象不相等
  if (optional_presence != spec.optional_presence) {
    return false;
  }
  // 如果 tensor_args 的长度不同，则对象不相等
  if (tensor_args.size() != spec.tensor_args.size())
    return false;
  // 如果 tensor_args 为空，则认为相等
  if (tensor_args.empty())
    return true;
  // 使用 memcmp 比较两个 tensor_args 的数据是否相同
  return std::memcmp(
             tensor_args.data(),
             spec.tensor_args.data(),
             tensor_args.size() * sizeof(ArgumentInfo)) == 0;
}

// 比较两个 ArgumentSpec 对象是否不相等
bool operator!=(const ArgumentSpec& spec) const {
  return !(*this == spec);
}

// 返回 tensor_args 的长度，即 tensor 的数量
size_t numTensors() const {
  return tensor_args.size();
}

// 返回索引 i 处的 tensor 参数信息
const ArgumentInfo& tensorAt(size_t i) const {
  return tensor_args[i];
}

// 返回 optional_presence 的长度，即可选参数的数量
size_t numOptionals() const {
  return optional_presence.size();
}

// 返回索引 i 处的 optional_presence 值，表示该位置的可选参数是否存在
bool isPresent(size_t i) const {
  return optional_presence[i];
}

// 返回对象的哈希码
size_t hashCode() const {
  return hash_code;
}
};

// 匿名命名空间，用于限定 ARG_SPEC_DEPTH_LIMIT 的作用域范围，其值为 128
namespace {
static constexpr size_t ARG_SPEC_DEPTH_LIMIT = 128;
}

// ArgumentSpecCreator 结构体，用于创建 ArgumentSpec 的简单指令集
struct TORCH_API ArgumentSpecCreator {
  // 枚举类型 Inst，定义了 ArgumentSpecCreator 的指令集
  // instructs 操作一个输入 IValues 列表的堆栈
  // 初始时堆栈包含一个输入函数的列表
  // ENTER_ 指令用于进入子对象并将新列表推送到堆栈上
  enum Inst : char {
    ENTER_TUPLE, // 从顶部列表消耗一个元组 ivalue，并将其元素列表作为新列表推送到堆栈上
    ENTER_OBJECT, // 类似 ENTER_TUPLE，但输入是类
    LEAVE, // 从堆栈中弹出顶部列表
    SKIP, // 从顶部列表消耗一个元素，并丢弃
    SPECIALIZE_OPTIONAL_TENSOR, // 为顶部列表消耗一个可选张量，并将其添加到正在创建的 ArgSpec 键中
    SPECIALIZE_TENSOR, // 为顶部列表消耗一个张量，并将其添加到正在创建的 ArgSpec 键中
    SPECIALIZE_OPTIONAL,
    // 为顶部列表消耗一个非张量可选项，并将其添加到正在创建的 ArgSpec 键中
  };
  ArgumentSpecCreator(Graph& graph); // 构造函数，以图形对象初始化
  ArgumentSpec create(bool with_grad, const Stack& stack) const; // 创建 ArgumentSpec
  void specializeTypes(Graph& g, const ArgumentSpec& spec) const; // 根据特定的 ArgumentSpec 特化类型
  void dump() const; // 打印函数，输出 ArgumentSpecCreator 的内容
  using WrittenSlots = std::unordered_set<std::string>;

 private:
  void scan(
      const TypePtr& typ,
      size_t depth,
      const WrittenSlots& written_slots); // 扫描函数，用于处理类型信息
  size_t num_inputs_; // 输入数量
  size_t num_tensors_ = 0; // 张量数量
  size_t num_optionals_ = 0; // 可选项数量
  std::vector<Inst> instructions_; // 指令集
};

// CompleteArgumentSpec 结构体，表示一个特定的专业化
// 设计为能够快速创建、哈希和比较，通常用于 JIT 的热路径以检查已创建的代码是否对给定输入有效
// COmpleteArgumentInfoPOD 仅在 CompleteArgumentSpec 内部使用，API 用户应使用 ArgumentInfo
struct CompleteArgumentInfoPOD {
  // 总大小为 64 位
  unsigned is_tensor : 8; // 如果为 false，则所有其他字段均无效
  unsigned type : 8; // 标量类型
  unsigned defined : 1; // 是否已定义
  unsigned requires_grad : 1; // 是否需要梯度
  signed device : 14; // 设备编号
  unsigned dev_type : 16; // 设备类型
  unsigned
      total_dims : 16; // 所有 TensorInfoPOD 在 CompleteArgumentSpec 的 tensor_info() 数组中
                       // total_dims 是到目前为止在 tensor_info() 的所有先前成员中看到的维度总数
                       // 包括此张量在内 2*total_dims 成为 sizes_strides 列表中下一个张量的偏移量
                       // 对于张量 0，偏移量始终为 0
};

static_assert(
    sizeof(CompleteArgumentInfoPOD) == sizeof(int64_t),
    "CompleteArgumentInfoPOD must be 64-bit struct for CompleteArgumentSpec encoding to work");



// 检查 CompleteArgumentInfoPOD 结构体大小是否等于 int64_t 的大小
// 如果不相等，抛出错误信息，要求 CompleteArgumentInfoPOD 必须是 64 位结构体，以便 CompleteArgumentSpec 编码能够正常工作
struct CompleteArgumentInfo;  // 声明 CompleteArgumentInfo 结构体，稍后会定义其内容

struct CompleteArgumentSpec {
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  // 构造函数：初始化 CompleteArgumentSpec 对象，根据输入参数生成数据结构
  CompleteArgumentSpec(bool with_grad, at::ArrayRef<IValue> inputs)
      : hash_code(0), ninputs(inputs.size()) {
    int32_t all_dims = 0;  // 初始化 all_dims 变量为 0，用于统计所有张量的维度总和

    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    const int32_t num_inputs = inputs.size();  // 获取输入张量的数量

    // 遍历输入的每个元素
    for (const auto i : c10::irange(num_inputs)) {
      if (!inputs[i].isTensor())  // 如果当前输入不是张量，则跳过
        continue;

      auto& tensor = inputs[i].toTensor();  // 获取当前输入的张量引用
      all_dims += tensor.defined() ? tensor.ndimension() : 0;  // 如果张量已定义，则累加其维度数
    }

    // 分配足够空间用于所有 TensorPOD 和维度信息
    data.resize(ninputs + all_dims * 2);

    // 将数据数组解释为 CompleteArgumentInfoPOD 结构体数组
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    auto* pods = reinterpret_cast<CompleteArgumentInfoPOD*>(data.data());

    int64_t* next_dim = sizes_strides();  // 获取尺寸和步幅的起始位置
    int32_t total_dims = 0;  // 初始化总维度计数器为 0

    // 再次遍历输入的每个元素
    for (const auto i : c10::irange(num_inputs)) {
      auto& pod = pods[i];  // 获取当前 CompleteArgumentInfoPOD 结构体引用
      pod.is_tensor = static_cast<uint32_t>(inputs[i].isTensor());  // 设置是否为张量的标志位

      if (pod.is_tensor) {  // 如果当前是张量
        at::Tensor t = inputs[i].toTensor();  // 获取当前张量

        pod.defined = t.defined();  // 设置是否已定义的标志位

        if (pod.defined) {  // 如果张量已定义
          pod.type = static_cast<int>(t.scalar_type());  // 设置张量的数据类型
          // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
          at::Device device = t.device();  // 获取张量的设备
          // NOLINTNEXTLINE(bugprone-signed-char-misuse)
          pod.dev_type = static_cast<std::underlying_type<DeviceType>::type>(
              device.type());  // 获取设备类型并设置到结构体中
          // NOLINTNEXTLINE(bugprone-signed-char-misuse)
          pod.device = device.index();  // 获取设备索引并设置到结构体中
          pod.requires_grad = with_grad && t.requires_grad();  // 设置是否需要梯度的标志位

          total_dims += t.ndimension();  // 累加当前张量的维度数

          auto sizes = t.sizes();  // 获取张量的尺寸信息
          std::copy(sizes.begin(), sizes.end(), next_dim);  // 将尺寸信息复制到指定位置
          next_dim += sizes.size();  // 更新下一个维度信息的位置

          auto strides = t.strides();  // 获取张量的步幅信息
          std::copy(strides.begin(), strides.end(), next_dim);  // 将步幅信息复制到指定位置
          next_dim += strides.size();  // 更新下一个步幅信息的位置
        }
      }

      // 每个 POD 结构体记录所有维度的累积总和
      TORCH_CHECK(
          total_dims < std::numeric_limits<uint16_t>::max(),
          "The number of dims cannot be packed into CompleteArgumentSpec:",
          total_dims);
      pod.total_dims = total_dims;  // 将累积总维度数设置到当前 POD 结构体中
    }

    // 预先计算 hash_code，以减少哈希表操作中的时间开销
    hash_code = c10::hash_combine(0, ninputs);
    for (auto d : data) {
      hash_code = c10::hash_combine(hash_code, d);  // 使用数据数组计算 hash_code
    }
  }

  // 相等性比较函数：快速比较 ninputs 和数据数组，无需考虑尺寸和步幅的间接性
  bool operator==(const CompleteArgumentSpec& spec) const {
    return ninputs == spec.ninputs && data == spec.data;  // 比较 ninputs 和数据数组是否相等
  }
  bool operator!=(const CompleteArgumentSpec& spec) const {
    // 不等性比较函数：与相等性比较相反
    return !(*this == spec);  // 使用相等性比较函数来实现
  }
};
  // 返回与当前对象不相等的 spec 对象
  return !(*this == spec);
}
// CompleteArgumentInfo 结构体的友元结构
friend struct CompleteArgumentInfo;
// 返回索引为 i 的 CompleteArgumentInfo 对象
CompleteArgumentInfo at(size_t i) const;
// 返回对象的输入数量 ninputs
size_t size() const {
  return ninputs;
}
// 返回对象的哈希值 hash_code
size_t hashCode() const {
  return hash_code;
}

private:
// 返回 tensor_info 的 ArrayRef，该函数将 data 转换为 CompleteArgumentInfoPOD 数组
ArrayRef<CompleteArgumentInfoPOD> tensor_info() const {
  return ArrayRef<CompleteArgumentInfoPOD>(
      reinterpret_cast<const CompleteArgumentInfoPOD*>(data.data()), ninputs);
}
// 返回 sizes_strides 的指针，指向 data 中 CompleteArgumentInfoPOD 列表后的大小和步幅信息
const int64_t* sizes_strides() const {
  return data.data() + ninputs;
}
// 返回 sizes_strides 的指针，用于修改 data 中 CompleteArgumentInfoPOD 列表后的大小和步幅信息
int64_t* sizes_strides() {
  return data.data() + ninputs;
}
// 预先计算的对象哈希码，构造时已计算
size_t hash_code;
// 对象的输入数量
size_t ninputs;
// 布局是 ninputs 个 TensorPOD（每个为64位），后跟它们的三个张量的大小和步幅信息：
// [t0POD][t1POD][t2POD]...
// [t0 sizes][t0 strides][t1 sizes][t1 strides][t2 sizes][t2 strides]
std::vector<int64_t> data;
};

// 定义结构体 CompleteArgumentInfo，用于公开压缩的 CompleteArgumentInfo
struct CompleteArgumentInfo {
  // 构造函数，接受 CompleteArgumentSpec 和整数 i 作为参数
  CompleteArgumentInfo(const CompleteArgumentSpec& spec, const int i)
      : spec(spec), i(i) {}

  // 返回当前 Tensor 是否是张量
  bool isTensor() const {
    return pod(i).is_tensor;
  }

  // 返回当前 Tensor 的数据类型
  at::ScalarType type() const {
    return at::ScalarType(pod(i).type);
  }

  // 返回当前 Tensor 是否已定义
  bool defined() const {
    return pod(i).defined;
  }

  // 返回当前 Tensor 是否需要梯度
  bool requires_grad() const {
    return pod(i).requires_grad;
  }

  // 返回当前 Tensor 的设备类型和设备索引
  at::Device device() const {
    return at::Device(
        DeviceType(pod(i).dev_type),
        static_cast<c10::DeviceIndex>(pod(i).device));
  }

  // 返回当前 Tensor 的维度数
  int ndimension() const {
    // 查看有效范围，始终可以请求 (i + 1) 处的偏移量
    return (sizes_strides_offset(i + 1) - sizes_strides_offset(i)) / 2;
  }

  // 返回当前 Tensor 的尺寸数组
  at::IntArrayRef sizes() const {
    return at::IntArrayRef(
        spec.sizes_strides() + sizes_strides_offset(i), ndimension());
  }

  // 返回当前 Tensor 的步幅数组
  at::IntArrayRef strides() const {
    int ndim = ndimension();
    return at::IntArrayRef(
        spec.sizes_strides() + sizes_strides_offset(i) + ndim, ndim);
  }

  // 转换为 TypePtr 类型，如果未定义则返回 TensorType::get()
  operator TypePtr() const {
    if (!defined())
      return TensorType::get();
    return TensorType::create(
        type(),
        device(),
        c10::VaryingShape<int64_t>{sizes()},
        c10::VaryingShape<int64_t>{strides()},
        requires_grad());
  }

 private:
  // 返回 sizes_strides() 数组中用于 tensor j 大小开始的偏移量
  // 有效范围是 [0, ninputs]
  // (即使请求 ninputs 处的偏移量也是有效的，如果存在下一个 tensor 的话)
  int sizes_strides_offset(int j) const {
    if (j == 0)
      return 0;
    // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
    return 2 * pod(j - 1).total_dims;
  }

  // 返回第 j 个 Tensor 的 CompleteArgumentInfoPOD 对象
  const CompleteArgumentInfoPOD& pod(int j) const {
    return spec.tensor_info().at(j);
  }

  const CompleteArgumentSpec& spec;  // 引用指向 CompleteArgumentSpec 对象
  const int i;  // 整数 i
};

// 重载操作符 <<，输出 ArgumentInfo 对象信息到输出流
inline std::ostream& operator<<(std::ostream& out, const ArgumentInfo& info) {
  if (!info.defined()) {
    return out << "<undefined>";
  }
  out << "Tensor(device=" << info.device() << ", type=" << toString(info.type())
      << ", requires_grad=" << info.requires_grad() << ", dims=" << info.dim()
      << ")";
  return out;
}

// 重载操作符 <<，输出 ArgumentSpec 对象信息到输出流
inline std::ostream& operator<<(std::ostream& out, const ArgumentSpec& spec) {
  out << "{";
  for (const auto i : c10::irange(spec.numTensors())) {
    if (i > 0)
      out << ", ";
    out << spec.tensorAt(i);
  }
  out << "; ";
  for (const auto i : c10::irange(spec.numOptionals())) {
    if (i > 0)
      out << ", ";
    out << spec.isPresent(i);
  }
  out << "}";
  return out;
}

// 重载操作符 <<，输出 CompleteArgumentInfo 对象信息到输出流
inline std::ostream& operator<<(
    std::ostream& out,
    const CompleteArgumentInfo& info) {
  if (!info.defined()) {
    // 如果 Tensor 未定义，输出 "<undefined>"
    return out << "<undefined>";
  }
    # 如果函数执行到此处，表示输出流不是有效的对象，因此输出 "<undefined>"
    return out << "<undefined>";
  }
  
  # 构建一个描述张量信息的字符串，包括设备信息、类型、梯度需求、尺寸和步长
  out << "Tensor(device=" << info.device() << ", type=" << toString(info.type())
      << ", requires_grad=" << info.requires_grad()
      << ", sizes=" << info.sizes() << ", strides=" << info.strides() << ")";
  
  # 返回构建好的描述字符串
  return out;
} // 结束 torch::jit 命名空间的定义

inline std::ostream& operator<<(
    std::ostream& out,
    const CompleteArgumentSpec& spec) {
  out << "{"; // 输出左大括号开始表示对象的开始
  for (const auto i : c10::irange(spec.size())) { // 对于 spec 的索引范围进行迭代
    if (i > 0)
      out << ", "; // 如果不是第一个元素，输出逗号和空格分隔
    out << spec.at(i); // 输出 spec 中第 i 个元素的内容
  }
  out << "}"; // 输出右大括号表示对象的结束
  return out; // 返回输出流对象
}

inline CompleteArgumentInfo CompleteArgumentSpec::at(size_t i) const {
  return CompleteArgumentInfo(*this, i); // 返回索引为 i 的 CompleteArgumentInfo 对象
}

inline std::optional<int8_t> convertOptional(
    std::optional<c10::ScalarType> const& from) {
  return (from) ? std::optional<int8_t>(static_cast<int8_t>(*from)) // 如果 from 有值，转换为 int8_t 类型的 optional
                : std::optional<int8_t>{}; // 否则返回空的 int8_t optional
}

} // 结束 torch::jit 命名空间的定义

namespace std {

template <typename T>
struct hash<c10::VaryingShape<T>> {
  size_t operator()(const c10::VaryingShape<T>& vs) const {
    return c10::get_hash(
        vs.size(), // 获取 vs 的大小
        vs.size() ? vs.sizes().value() : std::vector<std::optional<T>>()); // 如果 vs 非空，返回 sizes 的值，否则返回空的 optional vector
  }
};

template <>
struct hash<c10::TensorType> {
  size_t operator()(const c10::TensorType& ptt) const {
    return c10::get_hash<
        std::optional<int8_t>, // 使用 convertOptional 转换的 optional int8_t
        c10::VaryingShape<int64_t>, // c10::VaryingShape<int64_t> 的哈希值
        c10::VaryingShape<int64_t>, // c10::VaryingShape<int64_t> 的哈希值
        std::optional<bool>>( // optional bool 类型的哈希值
        torch::jit::convertOptional(ptt.scalarType()), // 调用 convertOptional 转换 scalarType 的 optional int8_t
        ptt.sizes(), // TensorType 的 sizes 的哈希值
        ptt.strides(), // TensorType 的 strides 的哈希值
        ptt.requiresGrad()); // TensorType 的 requiresGrad 的哈希值
  }
};

template <>
struct hash<torch::jit::ArgumentSpec> {
  size_t operator()(const torch::jit::ArgumentSpec& spec) const {
    return spec.hashCode(); // 返回 ArgumentSpec 对象的哈希码
  }
};
template <>
struct hash<torch::jit::CompleteArgumentSpec> {
  size_t operator()(const torch::jit::CompleteArgumentSpec& spec) const {
    return spec.hashCode(); // 返回 CompleteArgumentSpec 对象的哈希码
  }
};
} // 结束 std 命名空间的定义

C10_CLANG_DIAGNOSTIC_POP() // 弹出 C10_CLANG 的诊断设置
```