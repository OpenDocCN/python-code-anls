# `.\pytorch\torch\csrc\dynamo\compiled_autograd.h`

```
#pragma once
// 包含 ATen 库的头文件，用于张量几何信息
#include <ATen/TensorGeometry.h>
// 包含 IValue 类定义的头文件，用于表示任意类型的值
#include <ATen/core/ivalue.h>
// 包含 TorchDispatchModeTLS 类定义的头文件，用于管理 Torch 分发模式的线程本地存储
#include <c10/core/impl/TorchDispatchModeTLS.h>
// 包含 flat_hash_map 类定义的头文件，用于实现高效的哈希映射
#include <c10/util/flat_hash_map.h>
// 包含 function 类定义的头文件，用于定义自动求导函数
#include <torch/csrc/autograd/function.h>
// 包含 input_metadata 类定义的头文件，用于存储输入元数据信息
#include <torch/csrc/autograd/input_metadata.h>
// 包含 saved_variable 类定义的头文件，用于保存变量的自动求导信息
#include <torch/csrc/autograd/saved_variable.h>
// 包含 variable_info 类定义的头文件，用于存储变量的信息
#include <torch/csrc/autograd/variable_info.h>
// 包含 python_stub 类定义的头文件，用于支持 Python 的辅助函数和类型
#include <torch/csrc/utils/python_stub.h>
// 包含 torch_dispatch_mode 类定义的头文件，用于定义 Torch 分发模式
#include <torch/csrc/utils/torch_dispatch_mode.h>
// 包含 typeindex 类定义的头文件，用于提供类型的索引信息
#include <typeindex>
// 包含 vector 类定义的头文件，用于实现动态数组
#include <vector>

// 见 [Note: Compiled Autograd]，表示此命名空间与编译自动求导相关
namespace torch::dynamo::autograd {
using namespace torch::autograd;

// 表示输入尺寸的结构体，支持动态和静态两种类型
struct SizeInput {
  enum DynType : uint8_t { STATIC = 0, DYNAMIC = 1 }; // 定义动态类型和静态类型的枚举
  SizeInput(DynType dt, int64_t v) : dyn_type(dt), value(v) {} // 构造函数，初始化动态类型和值
  DynType dyn_type; // 动态类型
  int64_t value;   // 尺寸值
};

// 缓存键的缓冲区结构体，用于存储字节数组的数据，并进行深拷贝
struct CacheKeyBuffer {
  CacheKeyBuffer(const uint8_t* key, uint16_t len) : data(new uint8_t[len]) { // 构造函数，进行深拷贝
    std::memcpy(data.get(), key, len); // 将传入的 key 数据拷贝到 data 中
  }
  const uint8_t* get() const { // 获取缓存键数据的方法
    return data.get(); // 返回缓存键数据
  }

 private:
  std::unique_ptr<uint8_t[]> data; // 存储缓存键数据的唯一指针
};

// 表示缓存键的结构体，用于在影子图中查找下一个节点的关键
struct CacheKey {
  CacheKey(const std::type_index& ntype, const uint8_t* key, uint16_t len) // 构造函数，初始化节点类型、键大小和键数据
      : node_type(ntype), key_size(len), key(key) {} // 初始化成员变量

  bool operator<(const CacheKey& other) const { // 比较运算符重载，用于排序
    if (node_type != other.node_type) { // 首先比较节点类型
      return node_type < other.node_type; // 如果类型不同，根据节点类型排序
    }
    if (key_size != other.key_size) { // 若节点类型相同，则比较键的大小
      return key_size < other.key_size; // 根据键大小排序
    }
    return std::memcmp(key, other.key, key_size) < 0; // 最后比较键的具体数据内容
  }

  bool operator==(const CacheKey& other) const { // 相等比较运算符重载，用于判断是否相等
    return node_type == other.node_type && key_size == other.key_size &&
        std::memcmp(key, other.key, key_size) == 0; // 判断节点类型、键大小和键数据是否完全相同
  }

  size_t hash() const { // 计算哈希值的方法
    return std::hash<std::type_index>()(node_type) ^ key_size; // 哈希节点类型和键大小
  }

  std::type_index node_type; // 节点类型索引
  uint16_t key_size; // 键的大小
  const uint8_t* key; // 键的数据
};

// 表示节点调用的结构体，用于管理节点的调用信息
struct NodeCall {
  NodeCall(uint32_t id_, std::shared_ptr<Node> node_) // 构造函数，初始化节点 ID 和节点指针
      : id(id_), node(std::move(node_)) {} // 初始化成员变量

  void mark_output(int input_nr, int output_idx) { // 标记输出的方法，用于记录图的输出信息
    graph_output.emplace_back(input_nr, output_idx); // 添加图输出的信息
  }

  uint32_t id; // 节点 ID
  std::shared_ptr<Node> node; // 节点指针
  std::vector<std::pair<int, int>> tensor_pre_hooks; // 张量预处理钩子的集合
  std::vector<int> pre_hooks; // 预处理钩子的集合
  std::vector<int> post_hooks; // 后处理钩子的集合
  std::vector<int> post_acc_grad_hooks; // 累积梯度后处理钩子的集合
  std::vector<std::pair<int, int>> graph_output; // 图输出的集合
  bool needed = true; // 是否需要的标志
};

// 表示节点调用集合的结构体，继承自无序映射，用于管理多个节点调用信息
struct NodeCalls : public std::unordered_map<Node*, NodeCall> {
  NodeCall& lookup(const std::shared_ptr<Node>& function) { // 查找节点调用的方法
    auto it = find(function.get()); // 在映射中查找节点指针
    if (it == end()) { // 如果找不到
      it = emplace(function.get(), NodeCall(_next_id++, function)).first; // 插入新的节点调用信息
    }
    return it->second; // 返回节点调用信息
  }

 private:
  uint32_t _next_id = 0; // 下一个节点 ID
};
// 表示将传递到图中的去重张量的结构体
struct TensorArg {
  // 构造函数，初始化张量标识符为指定值，默认为0
  TensorArg(uint32_t i = 0) : id(i) {}
  
  // 返回张量在集合中的索引，要求张量已定义
  uint32_t index() const {
    TORCH_INTERNAL_ASSERT(defined());
    return id - 1;
  }
  
  // 检查张量是否已定义
  bool defined() const {
    return id != 0;
  }
  
  // 张量标识符
  uint32_t id;
  
  // 代理张量
  at::Tensor proxy_tensor;
};

// 管理TensorArg集合及其与张量/保存变量的映射
struct TensorArgs {
  // 查找给定张量对应的TensorArg对象，如果不存在且create为true，则创建新的TensorArg
  TensorArg& lookup(const at::Tensor& tensor, bool create = false) {
    // 若张量未定义，返回未定义的TensorArg
    if (!tensor.defined()) {
      return _undefined;
    }
    // 获取张量实现
    auto impl = tensor.unsafeGetTensorImpl();
    // 在_args中查找张量实现对应的TensorArg对象
    auto it = _args.find(impl);
    // 如果不存在，根据create标志创建新的TensorArg对象，并添加到_args和inputs中
    if (it == _args.end()) {
      TORCH_INTERNAL_ASSERT(create && inputs.size() == _next_id - 1);
      it = _args.emplace(impl, TensorArg(_next_id++)).first;
      inputs.emplace_back(tensor);
    }
    // 返回找到或新创建的TensorArg对象
    return it->second;
  }

  // 查找给定SavedVariable对应的TensorArg对象
  TensorArg& lookup(const SavedVariable& sv) {
    auto it = _saved_variables.find(&sv);
    // 断言确保SavedVariable存在对应的TensorArg对象
    TORCH_INTERNAL_ASSERT(it != _saved_variables.end());
    return *it->second;
  }

  // 添加给定张量的TensorArg对象
  TensorArg& add(const at::Tensor& tensor) {
    return lookup(tensor, true);
  }

  // 添加给定SavedVariable的TensorArg对象，同时执行一次SavedVariable的解包操作
  TensorArg& add(const SavedVariable& sv, const std::shared_ptr<Node>& node) {
    // TODO(jansel): 在此处确保仅解包一次SavedVariable。可能会触发SavedTensor hooks。将来应将saved tensor hooks放入图中。
    // 解包SavedVariable获取张量，并添加其对应的TensorArg对象
    at::Tensor tensor = sv.unpack(node);
    TensorArg& arg = add(tensor);
    _saved_variables.emplace(&sv, &arg);
    return arg;
  }

  // 将传递到图中的具体张量输入
  std::vector<at::Tensor> inputs;

private:
  // 从此处获取的每个TensorArg实际上由_args（或_undefined）拥有，这就是我们在这里使用非拥有指针的原因。
  std::unordered_map<const c10::TensorImpl*, TensorArg> _args;
  
  // 每个SavedVariable对应的TensorArg对象的映射
  std::unordered_map<const SavedVariable*, TensorArg*> _saved_variables;
  
  // 未定义的TensorArg对象
  TensorArg _undefined;
  
  // 下一个可用的TensorArg对象的标识符，0被_undefined使用
  uint32_t _next_id = 1; // id=0 used by _undefined
};

// AutogradCompilerCall结构体
struct AutogradCompilerCall {
  // 添加大小输入
  void add_size_input(const c10::SymInt& s) {
    // 将大小输入添加到all_size_inputs中
    all_size_inputs.emplace_back(
        default_dyn_type, s.guard_int(__FILE__, __LINE__));
  }

  // 插入钩子函数
  size_t emplace_hook(c10::SafePyObject&& fn) {
    // 将钩子函数添加到hooks中，并返回其索引
    hooks.emplace_back(std::move(fn));
    return hooks.size() - 1;
  }

  // TensorArgs对象，管理要传递到图中的张量参数
  TensorArgs tensor_args;
  
  // 所有大小输入的集合
  std::vector<SizeInput> all_size_inputs;
  
  // 动态大小输入的集合
  std::vector<int64_t> dyn_size_inputs;
  
  // 钩子函数的集合
  std::vector<c10::SafePyObject> hooks;
  
  // NodeCalls对象
  NodeCalls node_calls;
  
  // 默认的动态类型
  SizeInput::DynType default_dyn_type = SizeInput::STATIC;
};
class CompiledNodeArgs {
  // CompiledNodeArgs类用于构建编译图中所有节点中找到的常量值的表示，
  // 通过'collect'重载方法进行收集。收集到的常量值会被串联起来形成缓存键。
  // 张量（Tensor）、符号整数（SymInt）参数会被转发给编译器，并不包含在键中。

 public:
  // 收集张量参数的方法
  void collect(const TensorArg& t) {
    // 收集张量参数的标识大小
    collect_size(t.id);
    // 如果张量已定义
    if (t.defined()) {
      // 获取张量对象
      const at::Tensor& tensor = _compiler.tensor_args.inputs[t.index()];
      // 将张量的设备信息包含在缓存键中，以便可以跳过Dynamo级别的张量保护
      collect(tensor.device());
      // 收集张量的数据类型信息
      collect(tensor.dtype());
      // 收集张量是否需要梯度信息
      collect(tensor.requires_grad());
    }
  }

  // 收集张量参数的方法
  void collect(const at::Tensor& t) {
    // 将张量参数添加到编译器中进行收集
    collect(_compiler.tensor_args.add(t));
  }

  // 收集SavedVariable参数的方法
  void collect(const SavedVariable& t) {
    // 将SavedVariable参数添加到编译器中进行收集，并关联到当前节点
    collect(_compiler.tensor_args.add(t, _node_call.node));
  }

  // 收集SymInt参数的方法
  void collect(const c10::SymInt& t) {
    // 将SymInt参数的大小添加到输入中
    _compiler.add_size_input(t);
  }

  // 收集标准库vector容器类型参数的方法
  template <typename T>
  void collect(const std::vector<T>& t) {
    // 收集vector的大小信息
    collect_size(t.size());
    // 遍历vector中的每个元素，并进行收集
    for (const T& i : t) {
      collect(i);
    }
  }

  // 收集C10库ArrayRef容器类型参数的方法
  template <typename T>
  void collect(const c10::ArrayRef<T>& t) {
    // 收集ArrayRef的大小信息
    collect_size(t.size());
    // 遍历ArrayRef中的每个元素，并进行收集
    for (const T& i : t) {
      collect(i);
    }
  }

  // 收集C10库OptionalArray容器类型参数的方法
  template <typename T>
  void collect(const c10::OptionalArray<T>& t) {
    // 收集OptionalArray中的列表信息
    collect(t.list);
  }

  // 收集标准库optional容器类型参数的方法
  template <typename T>
  void collect(const std::optional<T>& t) {
    // 如果optional对象有值
    if (cond(t.has_value())) {
      // 收集optional对象的值
      collect(*t);
    }
  }

  // 收集std::pair参数的方法
  template <typename A, typename B>
  void collect(const std::pair<A, B>& t) {
    // 收集pair中的第一个元素
    collect(t.first);
    // 收集pair中的第二个元素
    collect(t.second);
  }

  // 收集flat_hash_map容器类型参数的方法
  template <typename V>
  void collect(const ska::flat_hash_map<std::string, V>& m) {
    // 收集flat_hash_map的大小信息
    collect_size(m.size());

    // 获取flat_hash_map中所有键，并排序
    std::vector<std::string> keys;
    keys.reserve(m.size());
    std::transform(
        m.begin(), m.end(), std::back_inserter(keys), [](const auto& entry) {
          return entry.first;
        });
    std::sort(keys.begin(), keys.end());

    // 遍历排序后的键，依次收集键和对应的值
    for (const auto& k : keys) {
      collect(k);
      collect(m.at(k));
    }
  }

  // 收集at::IValue参数的方法
  void collect(const at::IValue& iv) {
    // 如果IValue对象是列表类型
    if (iv.isList()) {
      // 将IValue对象转换为列表
      c10::List<at::IValue> list = iv.toList();
      // 收集列表的大小信息
      collect_size(list.size());
      // 遍历列表中的每个值，并进行收集
      for (auto&& value : list) {
        collect(value);
      }
    } else if (iv.isGenericDict()) {  // 如果IValue对象是通用字典类型
      // 将IValue对象转换为通用字典
      c10::Dict<at::IValue, at::IValue> ordered_dict = iv.toGenericDict();
      // 收集通用字典的大小信息
      collect_size(ordered_dict.size());
      // 遍历通用字典中的每个键值对，并进行收集
      // NOLINTNEXTLINE(modernize-loop-convert)
      for (auto it = ordered_dict.begin(); it != ordered_dict.end(); it++) {
        collect(it->key());
        collect(it->value());
      }
    }
  } else {
    try {
      // 尝试将 IValue 对象的哈希转换为 uint64_t，并进行收集
      collect(static_cast<uint64_t>(at::IValue::hash(iv)));
    } catch (const std::runtime_error& e) {
      // 捕获运行时错误，构造错误消息并抛出 TORCH_CHECK_NOT_IMPLEMENTED 异常
      std::string msg =
          "Compiled autograd can not trace unhashable IValues, error: " +
          std::string(e.what());
      TORCH_CHECK_NOT_IMPLEMENTED(false, msg);
    }
  }
}
void collect(const c10::Scalar& t) {
  // 获取 Scalar 的类型信息，并根据类型进行特定的收集操作
  auto type = t.type();
  specialize_on_bytes(type);
  if (type == c10::ScalarType::Double) {
    collect(t.toDouble());  // 收集 double 类型的值
  } else if (type == c10::ScalarType::Long) {
    collect(t.toLong());    // 收集 long 类型的值
  } else if (type == c10::ScalarType::Bool) {
    collect(t.toBool());    // 收集 bool 类型的值
  } else if (type == c10::ScalarType::ComplexDouble) {
    auto c = t.toComplexDouble();
    collect(c.real());      // 收集复数的实部
    collect(c.imag());      // 收集复数的虚部
  } else {
    TORCH_INTERNAL_ASSERT(false);  // 断言，不应该执行到这里
  }
}
void collect(const c10::TensorOptions& t) {
  collect(t.device());            // 收集张量选项中的设备信息
  collect(t.dtype());             // 收集张量选项中的数据类型信息
  collect(t.layout());            // 收集张量选项中的布局信息
  collect(t.requires_grad());     // 收集张量选项中的梯度信息
  collect(t.pinned_memory());     // 收集张量选项中的固定内存信息
  collect(t.memory_format_opt()); // 收集张量选项中的内存格式信息
}
void collect(const at::TensorGeometry& t) {
  collect(t.sym_sizes());         // 收集张量几何属性中的符号尺寸信息
  collect(t.sym_strides());       // 收集张量几何属性中的符号步长信息
  collect(t.sym_storage_offset()); // 收集张量几何属性中的符号存储偏移信息
}
void collect(const torch::autograd::TypeAndSize& t) {
  collect(t.sym_sizes);           // 收集类型和尺寸中的符号尺寸信息
  collect(t.options);             // 收集类型和尺寸中的选项信息
}
void collect(const c10::Device& t) {
  collect(t.type());              // 收集设备对象中的类型信息
  collect(t.index());             // 收集设备对象中的索引信息
}
void collect(const std::string& t) {
  collect_size(t.size());         // 收集字符串的大小信息
  for (char c : t) {
    collect(c);                   // 遍历收集字符串中的每个字符
  }
}
void collect(const caffe2::TypeMeta& t) {
  specialize_on_bytes(t.id());    // 根据类型元信息的 ID 进行特定的收集操作
}
void collect(const std::shared_ptr<Node>& t) {
  // 注意：这里仅捕获节点的 ID，而不是节点内部的所有内容。
  // 用于跟踪节点之间的连接，节点本身的详细信息需要通过 `node->compiled_args()` 单独处理。
  if (cond((bool)t)) {
    collect(_compiler.node_calls.lookup(t));  // 收集节点的调用信息
  }
}
void collect(const NodeCall& t) {
  collect_size(t.id);              // 收集节点调用的 ID 信息
  collect(t.graph_output);         // 收集节点调用的图输出信息
  collect_hooks_from(t.node.get()); // 收集从节点获取的钩子信息
}
void collect(const Edge& t) {
  if (cond(t.is_valid())) {
    collect_size(_compiler.node_calls.lookup(t.function).id); // 收集边的函数 ID 信息
    collect_size(t.input_nr);     // 收集边的输入编号信息
    collect(t.function->input_metadata(t.input_nr)); // 收集函数的输入元数据信息（用于验证输出）
  }
}
void collect(const InputMetadata& t) {
  TORCH_CHECK(!t.is_nested_tensor(), "NestedTensor not implemented");
  collect(t.options());           // 收集输入元数据的选项信息
  collect(t.is_tensor_subclass()); // 收集是否为张量子类的信息
  collect(t.shape_as_dim_vector()); // 收集形状作为维度向量的信息
}
void collect(const VariableInfo& t) {
  collect(t.layout);              // 收集变量信息中的布局信息
  collect(t.device);              // 收集变量信息中的设备信息
  collect(t.scalar_type);         // 收集变量信息中的标量类型信息
  collect(t.size);                // 收集变量信息中的尺寸信息
  collect(t.requires_grad);       // 收集变量信息中的梯度要求信息
  collect(t.is_empty);            // 收集变量信息中是否为空的信息
}
bool cond(bool cond) {
  collect(cond);                  // 收集条件值
  return cond;                    // 返回条件值
}
#define COLLECT_AS_BYTES(T) \
  void collect(T t) {       \  // 宏定义，生成一个模板函数 collect，接受类型 T 作为参数
    specialize_on_bytes(t); \  // 调用 specialize_on_bytes 函数处理参数 t
  }
  COLLECT_AS_BYTES(c10::ScalarType);   // 实例化模板，生成 collect 函数，接受 c10::ScalarType 类型参数
  COLLECT_AS_BYTES(c10::DeviceType);   // 实例化模板，生成 collect 函数，接受 c10::DeviceType 类型参数
  COLLECT_AS_BYTES(c10::Layout);       // 实例化模板，生成 collect 函数，接受 c10::Layout 类型参数
  COLLECT_AS_BYTES(c10::MemoryFormat); // 实例化模板，生成 collect 函数，接受 c10::MemoryFormat 类型参数
  COLLECT_AS_BYTES(int8_t);            // 实例化模板，生成 collect 函数，接受 int8_t 类型参数
  COLLECT_AS_BYTES(int16_t);           // 实例化模板，生成 collect 函数，接受 int16_t 类型参数
  COLLECT_AS_BYTES(int32_t);           // 实例化模板，生成 collect 函数，接受 int32_t 类型参数
  COLLECT_AS_BYTES(int64_t);           // 实例化模板，生成 collect 函数，接受 int64_t 类型参数
  COLLECT_AS_BYTES(uint8_t);           // 实例化模板，生成 collect 函数，接受 uint8_t 类型参数
  COLLECT_AS_BYTES(uint16_t);          // 实例化模板，生成 collect 函数，接受 uint16_t 类型参数
  COLLECT_AS_BYTES(uint32_t);          // 实例化模板，生成 collect 函数，接受 uint32_t 类型参数
  COLLECT_AS_BYTES(uint64_t);          // 实例化模板，生成 collect 函数，接受 uint64_t 类型参数
  COLLECT_AS_BYTES(bool);              // 实例化模板，生成 collect 函数，接受 bool 类型参数
  COLLECT_AS_BYTES(float);             // 实例化模板，生成 collect 函数，接受 float 类型参数
  COLLECT_AS_BYTES(double);            // 实例化模板，生成 collect 函数，接受 double 类型参数
#undef COLLECT_AS_BYTES

void collect_hooks_from(Node* fn) {
  TORCH_CHECK(
      fn->retains_grad_hooks().empty(),
      "retains_grad_hooks not implemented for compiled autograd");  // 检查是否没有保留梯度钩子
  for (auto& i : fn->tensor_pre_hooks()) {
    i->compiled_args(*this);  // 对 tensor_pre_hooks 中的每个元素调用 compiled_args 方法
  }
  for (auto& i : fn->pre_hooks()) {
    i->compiled_args(*this);  // 对 pre_hooks 中的每个元素调用 compiled_args 方法
  }
  for (auto& i : fn->post_hooks()) {
    i->compiled_args(*this);  // 对 post_hooks 中的每个元素调用 compiled_args 方法
  }
  collect_size(_node_call.tensor_pre_hooks.size());  // 收集 tensor_pre_hooks 的大小
  collect_size(_node_call.pre_hooks.size());         // 收集 pre_hooks 的大小
  collect_size(_node_call.post_hooks.size());        // 收集 post_hooks 的大小
  for (const auto& h : _node_call.tensor_pre_hooks) {
    collect_size(static_cast<size_t>(h.second));     // 收集 tensor_pre_hooks 中每个元素的大小
  }
}

CacheKey key() const {
  Node* node = _node_call.node.get();  // 获取 _node_call 的节点指针
  return CacheKey(
      typeid(*node), _specialization_key, _specialization_key_size);  // 返回 CacheKey，包括节点类型、特化键和特化键大小
}

size_t add_backward(c10::SafePyObject&& obj) {
  return _compiler.emplace_hook(std::move(obj));  // 添加后向钩子，并返回钩子的索引
}

size_t add_backward_state(c10::SafePyObject&& obj) {
  return _compiler.emplace_hook(std::move(obj));  // 添加后向状态钩子，并返回钩子的索引
}

void add_tensor_pre_hook(c10::SafePyObject&& obj, int index) {
  auto fn_id = _compiler.emplace_hook(std::move(obj));  // 添加张量预钩子，并获取钩子的索引
  collect_size(fn_id);                                 // 收集钩子索引的大小
  _node_call.tensor_pre_hooks.emplace_back(fn_id, index);  // 将钩子索引和索引添加到 tensor_pre_hooks 中
}

void add_pre_hook(c10::SafePyObject&& obj) {
  auto fn_id = _compiler.emplace_hook(std::move(obj));  // 添加预钩子，并获取钩子的索引
  collect_size(fn_id);                                 // 收集钩子索引的大小
  _node_call.pre_hooks.emplace_back(fn_id);            // 将钩子索引添加到 pre_hooks 中
}

void add_post_hook(c10::SafePyObject&& obj) {
  auto fn_id = _compiler.emplace_hook(std::move(obj));  // 添加后钩子，并获取钩子的索引
  collect_size(fn_id);                                 // 收集钩子索引的大小
  _node_call.post_hooks.emplace_back(fn_id);           // 将钩子索引添加到 post_hooks 中
}

void add_post_acc_grad_hook(c10::SafePyObject&& obj) {
  auto fn_id = _compiler.emplace_hook(std::move(obj));   // 添加后累积梯度钩子，并获取钩子的索引
  collect_size(fn_id);                                  // 收集钩子索引的大小
  _node_call.post_acc_grad_hooks.emplace_back(fn_id);    // 将钩子索引添加到 post_acc_grad_hooks 中
}

// Need to template the size_t to silence internal 32-bit build errors due to
// a mix of -Werror, -Wtautological-type-limit-compare and
// -Wunknown-pragmas
template <typename T>
std::enable_if_t<std::is_unsigned_v<T>, void> collect_size(T s) {
  // we expect sizes to be small, so try to cram them into a single byte
  constexpr uint8_t encode_as_u64 = std::numeric_limits<uint8_t>::max();  // 将大小编码为 uint8_t 类型
  constexpr uint8_t encode_as_u32 = encode_as_u64 - 1;  // 用于编码为 uint32_t 的大小
  constexpr uint8_t encode_as_u16 = encode_as_u64 - 2;  // 用于编码为 uint16_t 的大小
    // 如果 s 大于等于 encode_as_u16，则需要特别处理
    if (C10_UNLIKELY(s >= encode_as_u16)) {
      // 首先写入一个字节表示所采用的路径，然后写入数据
      // 如果 s 不超过 uint16_t 的最大值，则需要 3 个字节
      if (s <= std::numeric_limits<uint16_t>::max()) {
        specialize_on_bytes(encode_as_u16);
        specialize_on_bytes(static_cast<uint16_t>(s));
      } else if (s <= std::numeric_limits<uint32_t>::max()) {
        // 如果 s 不超过 uint32_t 的最大值，则需要 5 个字节
        specialize_on_bytes(encode_as_u32);
        specialize_on_bytes(static_cast<uint32_t>(s));
      } else {
        // 否则需要 9 个字节
        specialize_on_bytes(encode_as_u64);
        specialize_on_bytes(s);
      }
    } else {
      // 正常情况下，只需要 1 个字节即可
      specialize_on_bytes(static_cast<uint8_t>(s));
    }
  }

  // 设置默认的动态类型，并返回之前的默认动态类型
  SizeInput::DynType set_default_dyn_type(SizeInput::DynType default_dyn_type) {
    return std::exchange(_compiler.default_dyn_type, default_dyn_type);
  }

  // 编译节点参数的构造函数，初始化各成员变量
  CompiledNodeArgs(AutogradCompilerCall& compiler, NodeCall& node_call)
      : _compiler(compiler),
        _node_call(node_call),
        // 分配 _specialization_key 的内存空间，大小为 _specialization_key_storage 字节
        _specialization_key(
            // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
            (uint8_t*)std::malloc(_specialization_key_storage)) {}

  // 析构函数，释放 _specialization_key 所分配的内存
  ~CompiledNodeArgs() {
    // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
    std::free(_specialization_key);
  }

  // 禁用复制构造函数
  CompiledNodeArgs(const CompiledNodeArgs&) = delete;

 private:
  // 向 _specialization_key 中写入 T 类型的数据，并根据需要重新分配内存
  template <typename T>
  void specialize_on_bytes(const T& t) {
    while (C10_UNLIKELY(
        _specialization_key_size + sizeof(T) > _specialization_key_storage)) {
      _specialization_key_storage *= 2;
      // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
      _specialization_key = (uint8_t*)std::realloc(
          _specialization_key, _specialization_key_storage);
    }
    std::memcpy(_specialization_key + _specialization_key_size, &t, sizeof(T));
    _specialization_key_size += sizeof(T);
  }

  // 编译器调用对象的引用
  AutogradCompilerCall& _compiler;
  // 节点调用对象的引用
  NodeCall& _node_call;
  // 用于存储专门化数据的缓冲区大小和当前使用量
  size_t _specialization_key_size{0};
  size_t _specialization_key_storage{1024};
  // 用于存储专门化数据的指针
  uint8_t* _specialization_key;
};

// 跟踪状态的结构体，用于保存符号整数的可选项和输出的数量
struct TraceState {
  TraceState(
      const std::vector<std::optional<c10::SymInt>>& ss,  // 构造函数，接受符号整数的可选项列表和输出数量
      size_t num_outputs)
      : sym_sizes(ss), outputs(num_outputs) {}

  // 断言检查函数，用于调试
  void debug_asserts() {
    TORCH_INTERNAL_ASSERT(sym_sizes_index == sym_sizes.size());  // 断言，确保符号整数索引的有效性
  }
  
  // 返回下一个符号整数的可选项
  std::optional<c10::SymInt> next_sym_size() {
    TORCH_INTERNAL_ASSERT(sym_sizes_index < sym_sizes.size());  // 断言，确保符号整数索引小于列表长度
    return sym_sizes[sym_sizes_index++];
  }

  size_t sym_sizes_index{0};  // 符号整数索引，默认为0
  std::vector<std::optional<c10::SymInt>> sym_sizes;  // 符号整数的可选项列表
  variable_list outputs;  // 变量列表
};

// 用于在跟踪/编译阶段进行缓存未命中后的变量交换
class SwapSavedVariables {
  // SwapSavedVariables 在缓存未命中后的跟踪/编译阶段使用。它交换任何 'lifted' 输入（张量，符号整数）到代理节点，
  // 允许跟踪发生，然后在之后再次交换它们回来。
 public:
  // 在张量之前进行处理
  void before(at::Tensor& t) {
    TensorArg& arg = compiler.tensor_args.lookup(t);  // 查找张量在编译器中的参数
    stashed_tensors.save(&t, std::move(t));  // 保存张量的备份
    if (arg.defined()) {
      TORCH_INTERNAL_ASSERT(arg.proxy_tensor.defined());  // 断言，确保代理张量已定义
      t = arg.proxy_tensor;  // 将张量替换为其代理张量
    }
  }
  // 在张量之后进行处理
  void after(at::Tensor& t) {
    stashed_tensors.restore(&t);  // 恢复张量的原始值
  }

  // 在 SavedVariable 之前进行处理
  void before(SavedVariable& t) {
    TensorArg& arg = compiler.tensor_args.lookup(t);  // 查找 SavedVariable 在编译器中的参数
    stashed_variables.save(&t, std::move(t));  // 保存 SavedVariable 的备份
    if (arg.defined()) {
      TORCH_INTERNAL_ASSERT(arg.proxy_tensor.defined());  // 断言，确保代理张量已定义
      t = SavedVariable(arg.proxy_tensor, false);  // 将 SavedVariable 替换为其代理张量的 SavedVariable
    }
  }
  // 在 SavedVariable 之后进行处理
  void after(SavedVariable& t) {
    stashed_variables.restore(&t);  // 恢复 SavedVariable 的原始值
  }

  // 在符号整数之前进行处理
  void before(c10::SymInt& t) {
    stashed_symints.save(&t, c10::SymInt(t));  // 保存符号整数的备份
    auto opt_value = state.next_sym_size();  // 获取下一个符号整数的可选项
    if (opt_value.has_value()) {
      t = *opt_value; // dynamic shape  // 将符号整数替换为下一个可选项的值（动态形状）
    }
  }
  // 在符号整数之后进行处理
  void after(c10::SymInt& t) {
    stashed_symints.restore(&t);  // 恢复符号整数的原始值
  }

  // 在 IValue 之前进行处理
  void before(at::IValue& t) {
    stashed_ivalues.save(&t, at::IValue(t));  // 保存 IValue 的备份
  }

  // 在 IValue 之后进行处理
  void after(at::IValue& t) {
    stashed_ivalues.restore(&t);  // 恢复 IValue 的原始值
  }

  // 在 Edge 之前进行处理
  void before(Edge& t) {
    if (t.is_valid()) {
      // need for symints used by validate_outputs
      before(t.function->mutable_input_metadata(t.input_nr));  // 处理 Edge 的输入元数据
    }
  }
  // 在 Edge 之后进行处理
  void after(Edge& t) {
    if (t.is_valid()) {
      after(t.function->mutable_input_metadata(t.input_nr));  // 处理 Edge 的输入元数据
    }
  }

  // 在 InputMetadata 之前进行处理
  void before(InputMetadata& t) {
    before(t.mutable_shape_as_dim_vector());  // 处理输入元数据的形状向量
  }
  // 在 InputMetadata 之后进行处理
  void after(InputMetadata& t) {
    after(t.mutable_shape_as_dim_vector());  // 处理输入元数据的形状向量
  }

  // 在 TensorGeometry 之前进行处理
  void before(at::TensorGeometry& t) {
    before(t.mutable_sizes());  // 处理张量几何的大小
    before(t.mutable_strides());  // 处理张量几何的步幅
    before(t.mutable_storage_offset());  // 处理张量几何的存储偏移
    t.recompute();  // 重新计算张量几何
  }
  // 在 TensorGeometry 之后进行处理
  void after(at::TensorGeometry& t) {
    after(t.mutable_sizes());  // 处理张量几何的大小
    after(t.mutable_strides());  // 处理张量几何的步幅
    after(t.mutable_storage_offset());  // 处理张量几何的存储偏移
    t.recompute();  // 重新计算张量几何
  }

  // 在 TypeAndSize 之前进行处理
  void before(torch::autograd::TypeAndSize& t) {
    before(t.sym_sizes);  // 处理 TypeAndSize 的符号大小
    before(t.options);  // 处理 TypeAndSize 的选项
  }
  // 在 TypeAndSize 之后进行处理
  void after(torch::autograd::TypeAndSize& t) {
    after(t.sym_sizes);  // 处理 TypeAndSize 的符号大小
    after(t.options);  // 处理 TypeAndSize 的选项
  }

  // 在 VariableInfo 之前进行处理
  void before(VariableInfo& t) {
    before(t.size);  // 处理变量信息的大小
  }
  // 在 VariableInfo 之后进行处理
  void after(VariableInfo& t) {
    // 留空，因为没有后处理
  }
};
    // 调用 after 函数来处理参数 t 的所有元素，其中 t 是一个标准库容器的引用
    template <typename T>
    void after(std::vector<T>& t) {
      // 遍历容器 t 中的每个元素，对每个元素调用 after 函数
      for (T& i : t) {
        after(i);
      }
    }
    
    // 调用 before 函数来处理参数 t 的所有元素，其中 t 是一个 c10::SmallVector 的引用，N 表示大小
    template <typename T, unsigned N>
    void before(c10::SmallVector<T, N>& t) {
      // 遍历 c10::SmallVector t 中的每个元素，对每个元素调用 before 函数
      for (T& i : t) {
        before(i);
      }
    }
    
    // 调用 after 函数来处理参数 t 的所有元素，其中 t 是一个 c10::SmallVector 的引用，N 表示大小
    template <typename T, unsigned N>
    void after(c10::SmallVector<T, N>& t) {
      // 遍历 c10::SmallVector t 中的每个元素，对每个元素调用 after 函数
      for (T& i : t) {
        after(i);
      }
    }
    
    // 调用 before 函数来处理参数 t 的所有元素，其中 t 是一个 c10::OptionalArray 的引用
    template <typename T>
    void before(c10::OptionalArray<T>& t) {
      // 调用 before 函数处理 c10::OptionalArray 的成员变量 list
      before(t.list);
    }
    
    // 调用 after 函数来处理参数 t 的所有元素，其中 t 是一个 c10::OptionalArray 的引用
    template <typename T>
    void after(c10::OptionalArray<T>& t) {
      // 调用 after 函数处理 c10::OptionalArray 的成员变量 list
      after(t.list);
    }
    
    // 调用 before 函数来处理参数 t 的所有元素，其中 t 是一个 std::optional 的引用
    template <typename T>
    void before(std::optional<T>& t) {
      // 如果 std::optional t 有值，则调用 before 函数处理其值
      if (t.has_value()) {
        before(*t);
      }
    }
    
    // 调用 after 函数来处理参数 t 的所有元素，其中 t 是一个 std::optional 的引用
    template <typename T>
    void after(std::optional<T>& t) {
      // 如果 std::optional t 有值，则调用 after 函数处理其值
      if (t.has_value()) {
        after(*t);
      }
    }
    
    // 调用 before 函数来处理参数 m 的所有元素，其中 m 是一个 ska::flat_hash_map 的引用，V 是值类型
    template <typename V>
    void before(ska::flat_hash_map<std::string, V>& m) {
      // 创建一个字符串向量 keys，用于存放 ska::flat_hash_map m 的键，并按字母顺序排序
      std::vector<std::string> keys;
      keys.reserve(m.size());
      std::transform(
          m.begin(), m.end(), std::back_inserter(keys), [](const auto& entry) {
            return entry.first;
          });
      std::sort(keys.begin(), keys.end());
      // 遍历排序后的键列表 keys，并对 ska::flat_hash_map m 中对应键的值调用 before 函数
      for (auto& k : keys) {
        before(m.at(k));
      }
    }
    
    // 调用 after 函数来处理参数 m 的所有元素，其中 m 是一个 ska::flat_hash_map 的引用，V 是值类型
    template <typename V>
    void after(ska::flat_hash_map<std::string, V>& m) {
      // 遍历 ska::flat_hash_map m 中的每个值，对每个值调用 after 函数
      for (auto& [_, v] : m) {
        after(v);
      }
    }
// 定义一个宏 NO_OP_VISIT，用于生成空的 before() 和 after() 函数模板
#define NO_OP_VISIT(T)     \
  void before(const T&) {} \  // 生成空的 before() 函数模板，接受常量引用参数
  void after(const T&) {}   // 生成空的 after() 函数模板，接受常量引用参数

// 依次使用宏 NO_OP_VISIT 生成一系列特化的 before() 和 after() 函数模板
NO_OP_VISIT(caffe2::TypeMeta);    // 特化 caffe2::TypeMeta 类型的函数模板
NO_OP_VISIT(c10::Device);         // 特化 c10::Device 类型的函数模板
NO_OP_VISIT(c10::DeviceType);     // 特化 c10::DeviceType 类型的函数模板
NO_OP_VISIT(c10::Layout);         // 特化 c10::Layout 类型的函数模板
NO_OP_VISIT(c10::MemoryFormat);   // 特化 c10::MemoryFormat 类型的函数模板
NO_OP_VISIT(c10::ScalarType);     // 特化 c10::ScalarType 类型的函数模板
NO_OP_VISIT(c10::Scalar);         // 特化 c10::Scalar 类型的函数模板
NO_OP_VISIT(c10::TensorOptions);  // 特化 c10::TensorOptions 类型的函数模板
NO_OP_VISIT(std::string);         // 特化 std::string 类型的函数模板
NO_OP_VISIT(int64_t);             // 特化 int64_t 类型的函数模板
NO_OP_VISIT(bool);                // 特化 bool 类型的函数模板
NO_OP_VISIT(double);              // 特化 double 类型的函数模板

// 取消之前定义的 NO_OP_VISIT 宏，避免在后续代码中继续使用
#undef NO_OP_VISIT

// SwapSavedVariables 类的构造函数，接受 AutogradCompilerCall、TraceState、PyObject* 和 NodeCall 参数
SwapSavedVariables(
    AutogradCompilerCall& c,
    TraceState& s,
    PyObject* p,
    const NodeCall& n)
    : compiler(c), state(s), py_compiler(p), curr_node_call(n) {}

// 返回成员变量 py_compiler，该变量为 PyObject* 类型
PyObject* get_py_compiler() {
  return py_compiler;
}

// 返回成员变量 curr_node_call，该变量为 const NodeCall& 类型的引用
const NodeCall& get_curr_node_call() {
  return curr_node_call;
}

// 执行断言检查函数，检查三个 StashedVars 结构体的内容是否为空
void debug_asserts() {
  stashed_variables.debug_assert();   // 检查 stashed_variables 是否为空
  stashed_tensors.debug_assert();     // 检查 stashed_tensors 是否为空
  stashed_symints.debug_assert();     // 检查 stashed_symints 是否为空
}

// SwapSavedVariables 类的私有部分开始
private:
// 定义一个模板结构体 Stashed，用于保存之前的值及其计数
template <typename T>
struct Stashed {
  Stashed(T&& v) : prior_value(std::move(v)) {}  // 构造函数，保存给定值的右值引用到 prior_value 中
  T prior_value;  // 保存的值
  // 注意：count 用于支持对 before() 的重复调用，当有多个 autograd::Edge 对象指向同一个 autograd::Node 时会发生
  int count = 1;  // 计数，默认为 1
};

// 定义一个模板结构体 StashedVars，继承自 std::unordered_map<const T*, Stashed<T>>
template <typename T>
struct StashedVars : public std::unordered_map<const T*, Stashed<T>> {
  // 保存键值对到映射中，如果已存在则保留先前的值
  void save(const T* key, T&& value) {
    auto [it, inserted] = this->try_emplace(key, std::move(value));
    if (!inserted) {
      // 如果插入失败，则增加计数
      it->second.count++;
    }
  }

  // 从映射中恢复值
  void restore(T* var) {
    auto it = this->find(var);
    TORCH_INTERNAL_ASSERT(it != this->end(), "missing before()");
    if (--it->second.count == 0) {
      // 当计数减少到 0 时，恢复之前保存的值，并从映射中移除
      *var = std::move(it->second.prior_value);
      this->erase(it);
    }
  }

  // 断言检查函数，确保映射为空，即所有保存的值都已经被恢复
  void debug_assert() {
    TORCH_INTERNAL_ASSERT(this->empty(), "missing call to after()");
  }
};

// 成员变量 compiler，引用自 AutogradCompilerCall 类型
AutogradCompilerCall& compiler;
// 成员变量 state，引用自 TraceState 类型
TraceState& state;
// 成员变量 py_compiler，这是一个 PyObject* 类型的借用引用，不改变其所有权或生命周期
PyObject* py_compiler;
// 成员变量 curr_node_call，引用自 const NodeCall 类型
const NodeCall& curr_node_call;

// 下面的结构用于保存在 before() 中被覆盖的值，在 after() 中用于清理
// 每个结构体用于不同类型的对象：SavedVariable、at::Tensor、c10::SymInt 和 at::IValue
StashedVars<SavedVariable> stashed_variables;  // 保存之前 SavedVariable 对象的值
StashedVars<at::Tensor> stashed_tensors;       // 保存之前 at::Tensor 对象的值
StashedVars<c10::SymInt> stashed_symints;      // 保存之前 c10::SymInt 对象的值
StashedVars<at::IValue> stashed_ivalues;       // 保存之前 at::IValue 对象的值
};

} // namespace torch::dynamo::autograd

// 为 torch::dynamo::autograd::CacheKey 类型定义 std::hash 特化，用于哈希操作
template <>
struct std::hash<torch::dynamo::autograd::CacheKey> {
  // 哈希运算符，调用 CacheKey 类的 hash() 方法
  size_t operator()(const torch::dynamo::autograd::CacheKey& k) const {
    return k.hash();
  }
};
```