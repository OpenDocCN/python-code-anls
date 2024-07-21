# `.\pytorch\torch\csrc\distributed\rpc\types.cpp`

```py
// 引用 Torch 分布式 RPC 框架中的类型定义头文件
#include <torch/csrc/distributed/rpc/types.h>

namespace torch {
namespace distributed {
namespace rpc {

// 线程本地变量，用于在 RPC 调用范围内允许或禁止对 JIT RRef 进行序列化
static thread_local bool allowJitRRefPickle = false;

// 返回当前线程的 allowJitRRefPickle 变量值
bool getAllowJitRRefPickle() {
  return allowJitRRefPickle;
}

// 允许当前线程对 JIT RRef 进行序列化
void enableJitRRefPickle() {
  allowJitRRefPickle = true;
}

// 禁止当前线程对 JIT RRef 进行序列化
void disableJitRRefPickle() {
  allowJitRRefPickle = false;
}

// 断言：本地 ID 类型的最大值不超过 int64_t 的最大值
static_assert(
    std::numeric_limits<local_id_t>::max() <=
        std::numeric_limits<int64_t>::max(),
    "The max value of local_id_t must be within the range of int64_t");
// 断言：工作节点 ID 类型的最大值不超过 int64_t 的最大值
static_assert(
    std::numeric_limits<worker_id_t>::max() <=
        std::numeric_limits<int64_t>::max(),
    "The max value of worker_id_t must be within the range of int64_t");

///////////////////////////  JitRRefPickleGuard   ///////////////////////////

// JitRRefPickleGuard 类的构造函数，启用 allowJitRRefPickle 标志
JitRRefPickleGuard::JitRRefPickleGuard() {
  allowJitRRefPickle = true;
}

// JitRRefPickleGuard 类的析构函数，禁用 allowJitRRefPickle 标志
JitRRefPickleGuard::~JitRRefPickleGuard() {
  allowJitRRefPickle = false;
}

///////////////////////////  GloballyUniqueId   ///////////////////////////

// GloballyUniqueId 类的构造函数，接受创建节点和本地 ID 作为参数
GloballyUniqueId::GloballyUniqueId(worker_id_t createdOn, local_id_t localId)
    : createdOn_(createdOn), localId_(localId) {}

// 比较两个 GloballyUniqueId 对象是否相等
bool GloballyUniqueId::operator==(const GloballyUniqueId& other) const {
  return createdOn_ == other.createdOn_ && localId_ == other.localId_;
}

// 比较两个 GloballyUniqueId 对象是否不相等
bool GloballyUniqueId::operator!=(const GloballyUniqueId& other) const {
  return createdOn_ != other.createdOn_ || localId_ != other.localId_;
}

// 将 GloballyUniqueId 转换为 IValue 对象
at::IValue GloballyUniqueId::toIValue() const {
  return c10::ivalue::Tuple::create(
      {static_cast<int64_t>(createdOn_), static_cast<int64_t>(localId_)});
}

// 从 IValue 对象构建 GloballyUniqueId
GloballyUniqueId GloballyUniqueId::fromIValue(const at::IValue& ivalue) {
  TORCH_INTERNAL_ASSERT(
      ivalue.isTuple(),
      "GloballyUniqueId::fromIValue expected ivalue to be a tuple.");
  const auto& ivalues = ivalue.toTupleRef().elements();
  TORCH_CHECK(
      ivalues.size() == 2,
      "Constructing GloballyUniqueId from ivalue "
      "expects a GenericList of two elements, but got ",
      ivalues.size());

  TORCH_CHECK(
      ivalues[0].toInt() <= std::numeric_limits<worker_id_t>::max(),
      "GloballyUniqueId createdOn out of range, got ",
      ivalues[0].toInt());
  worker_id_t createdOn = ivalues[0].toInt();

  TORCH_CHECK(
      ivalues[1].toInt() <= std::numeric_limits<local_id_t>::max(),
      "GloballyUniqueId localId out of range, got ",
      ivalues[1].toInt());
  local_id_t localId = ivalues[1].toInt();

  return GloballyUniqueId(createdOn, localId);
}

// 重载流输出操作符，用于输出 GloballyUniqueId 对象信息
std::ostream& operator<<(std::ostream& os, GloballyUniqueId const& globalId) {
  return os << "GloballyUniqueId(created_on=" << globalId.createdOn_
            << ", local_id=" << globalId.localId_ << ")";
}

///////////////////////////  SerializedPyObj   ///////////////////////////
// 将当前对象的数据转换为 std::vector<at::IValue> 类型的对象，并且是右值引用
std::vector<at::IValue> SerializedPyObj::toIValues() && {
  // 创建一个空的 std::vector<at::IValue> 对象
  std::vector<at::IValue> ivalues;
  // 预留足够的空间以容纳所有张量和一个附加数据
  ivalues.reserve(tensors_.size() + 1);
  // 遍历当前对象中的每一个张量
  for (auto& tensor : tensors_) {
    // 将每个张量移动到 ivalues 中
    ivalues.emplace_back(std::move(tensor));
  }
  // 将附加数据 payload_ 移动到 ivalues 中
  ivalues.emplace_back(std::move(payload_));
  // 返回包含所有移动值的 std::vector<at::IValue> 对象
  return ivalues;
}

// 根据给定的 std::vector<at::IValue> 对象创建 SerializedPyObj 对象的静态方法
SerializedPyObj SerializedPyObj::fromIValues(std::vector<at::IValue> values) {
  // 从 values 中提取最后一个元素作为 payload，并转换为 std::string
  std::string payload = values.back().toStringRef();
  // 移除 values 中的最后一个元素
  values.pop_back();
  // 创建一个空的 std::vector<at::Tensor> 对象
  std::vector<at::Tensor> tensors;
  // 预留足够的空间以容纳 values 中的所有张量
  tensors.reserve(values.size());
  // 遍历 values 中的每一个元素
  for (auto& value : values) {
    // 将每个元素转换为 at::Tensor，并移动到 tensors 中
    tensors.emplace_back(value.toTensor());
  }
  // 返回一个新的 SerializedPyObj 对象，使用移动语义传递 payload 和 tensors
  return SerializedPyObj(std::move(payload), std::move(tensors));
}

// 结束命名空间 rpc，并结束命名空间 distributed，最后结束命名空间 torch
} // namespace rpc
} // namespace distributed
} // namespace torch
```