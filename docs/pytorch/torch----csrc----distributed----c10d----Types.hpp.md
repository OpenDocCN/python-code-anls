# `.\pytorch\torch\csrc\distributed\c10d\Types.hpp`

```
#pragma once
// 预处理指令，确保本头文件只被编译一次

#include <torch/csrc/distributed/c10d/Store.hpp>
// 引入 Torch 分布式库的存储头文件

#include <chrono>
#include <cstdint>
// 引入用于时间计算和整数数据类型的标准库头文件

#include <ATen/core/Tensor.h>
#include <ATen/core/ivalue.h>
// 引入 ATen 张量和 IValue 类的核心头文件

#include <c10/macros/Macros.h>
#include <c10/util/intrusive_ptr.h>
// 引入 C10 库的宏和智能指针实用函数

namespace c10d {

// 为 ReduceOps 可能需要的补充数据定义基类
struct TORCH_API _SupplementBase : torch::CustomClassHolder {
  ~_SupplementBase() override = default;
};

// 特定于 NCCL PREMUL_SUM 的补充数据
// 在 ProcessGroupNCCL 中的使用方式已知如何解包
struct NCCLPreMulSumSupplement : _SupplementBase {
  double double_factor{0.0}; // 双精度浮点数因子
  at::Tensor tensor_factor; // ATen 张量因子
  NCCLPreMulSumSupplement(double f) : double_factor{f} {} // 构造函数，初始化双精度浮点数因子
  NCCLPreMulSumSupplement(at::Tensor t) : tensor_factor{std::move(t)} { // 构造函数，初始化 ATen 张量因子
    TORCH_CHECK_EQ(tensor_factor.numel(), 1); // 检查张量因子的元素数量为1
  }
};

// 其他 ReduceOps 可能需要不同补充数据，可以从 _SupplementBase 派生
struct TORCH_API ReduceOp : torch::CustomClassHolder {
  // 注释：RedOpType 可能会在 `ReduceOp` 外部定义
  enum RedOpType : uint8_t {
    SUM = 0, // 求和操作
    AVG = 1, // 平均值操作
    PRODUCT = 2, // 乘积操作
    MIN = 3, // 最小值操作
    MAX = 4, // 最大值操作
    BAND = 5, // 按位与操作
    BOR = 6, // 按位或操作
    BXOR = 7, // 按位异或操作
    PREMUL_SUM = 8, // 在求和前乘以用户提供的常数
    UNUSED = 9 // 未使用
  };

  ReduceOp() = default; // 默认构造函数

  ReduceOp(RedOpType op) : op_(op) {
    TORCH_INTERNAL_ASSERT(
        op_ != PREMUL_SUM,
        "Use `torch.distributed._make_nccl_premul_sum` to create an instance of ReduceOp with PREMUL_SUM");
  } // 构造函数，初始化操作类型，并检查不允许使用 PREMUL_SUM

  ReduceOp(
      RedOpType op,
      const c10::intrusive_ptr<_SupplementBase>& optional_supplement) {
    if (optional_supplement) {
      op_ = op; // 如果提供了可选补充数据，设置操作类型
    } else {
      supplement_ = optional_supplement; // 否则设置补充数据
    }
  } // 构造函数，根据是否提供可选补充数据设置操作类型或补充数据

  // 补充资源 supplement_ 如果存在，由 c10::intrusive_ptr 管理，因此构造函数和 operator= 可以简单处理
  ReduceOp(const ReduceOp& other) = default;
  ReduceOp& operator=(const ReduceOp& other) = default;

  ReduceOp(ReduceOp&& other) = default;
  ReduceOp& operator=(ReduceOp&& other) = default;

  operator RedOpType() const {
    return op_;
  } // 转换操作符，返回操作类型

  bool operator==(const std::uint8_t other) {
    TORCH_INTERNAL_ASSERT(other < 9, "Invalid other op value");
    return other == op_;
  } // 判断是否与给定的 std::uint8_t 类型的操作值相等

  bool operator==(const ReduceOp::RedOpType other) {
    return *this == static_cast<std::uint8_t>(other);
  } // 判断是否与给定的 RedOpType 类型的操作值相等

  // 注释：处理 `RedOpType::PREMUL_SUM` 以及其缩放因子
  bool operator==(const ReduceOp& other) {
    // 返回当前对象是否等于另一个对象的操作类型
    return *this == other.op_;
  }

  // 默认操作类型为SUM
  RedOpType op_ = SUM;

  // supplement_ 是用于存储可选补充数据的“类型擦除”存储空间，
  // 这些数据可能由操作类型需要。
  // 使用该点时将知道 supplement_ 实际上是什么派生类型，
  // 并向下转型其指针以提取所需类型的数据。
  // 目前，只有 PREMUL_SUM 需要补充数据，但相同的机制可以扩展到支持其他非平凡的归约操作，
  // 其中包含不同的补充有效负载。
  c10::intrusive_ptr<_SupplementBase> supplement_;
};

// 定义模板函数 makeNCCLPreMulSum，用于创建一个 ReduceOp 对象，类型为 PREMUL_SUM
template <typename T>
ReduceOp makeNCCLPreMulSum(const T& factor) {
  // 创建 ReduceOp 对象 rop
  ReduceOp rop;
  // 设置 rop 的操作类型为 PREMUL_SUM
  rop.op_ = ReduceOp::PREMUL_SUM;
  // 创建一个 NCCLPreMulSumSupplement 对象，并将其作为 rop 的附加信息
  rop.supplement_ = c10::make_intrusive<NCCLPreMulSumSupplement>(factor);
  // 返回创建的 ReduceOp 对象 rop
  return rop;
}

// 声明一个常量，表示超时时间为负数的情况
constexpr auto kUnsetTimeout = std::chrono::milliseconds(-1);

// 定义结构体 BroadcastOptions，用于设置广播操作的选项
struct BroadcastOptions {
  int64_t rootRank = 0; // 广播操作的根节点排名，默认为 0
  int64_t rootTensor = 0; // 广播操作的根张量，默认为 0
  std::chrono::milliseconds timeout = kUnsetTimeout; // 广播操作的超时时间，默认为 kUnsetTimeout
  bool asyncOp = true; // 是否异步操作，默认为 true
};

// 定义结构体 AllreduceOptions，用于设置全局归约操作的选项
struct AllreduceOptions {
  ReduceOp reduceOp = ReduceOp::SUM; // 归约操作的类型，默认为 SUM
  std::chrono::milliseconds timeout = kUnsetTimeout; // 归约操作的超时时间，默认为 kUnsetTimeout
  std::optional<at::Tensor> sparseIndices = c10::nullopt; // 稀疏索引，可选项，默认为空
};

// 定义结构体 AllreduceCoalescedOptions，继承自 AllreduceOptions，用于设置集成全局归约操作的选项
struct AllreduceCoalescedOptions : AllreduceOptions {};

// 定义结构体 ReduceOptions，用于设置归约操作的选项
struct ReduceOptions {
  ReduceOp reduceOp = ReduceOp::SUM; // 归约操作的类型，默认为 SUM
  int64_t rootRank = 0; // 归约操作的根节点排名，默认为 0
  int64_t rootTensor = 0; // 归约操作的根张量，默认为 0
  std::chrono::milliseconds timeout = kUnsetTimeout; // 归约操作的超时时间，默认为 kUnsetTimeout
};

// 定义结构体 AllgatherOptions，用于设置全收集操作的选项
struct AllgatherOptions {
  std::chrono::milliseconds timeout = kUnsetTimeout; // 全收集操作的超时时间，默认为 kUnsetTimeout
  bool asyncOp = true; // 是否异步操作，默认为 true
};

// 定义结构体 GatherOptions，用于设置收集操作的选项
struct GatherOptions {
  int64_t rootRank = 0; // 收集操作的根节点排名，默认为 0
  std::chrono::milliseconds timeout = kUnsetTimeout; // 收集操作的超时时间，默认为 kUnsetTimeout
};

// 定义结构体 ScatterOptions，用于设置分散操作的选项
struct ScatterOptions {
  int64_t rootRank = 0; // 分散操作的根节点排名，默认为 0
  std::chrono::milliseconds timeout = kUnsetTimeout; // 分散操作的超时时间，默认为 kUnsetTimeout
  bool asyncOp = true; // 是否异步操作，默认为 true
};

// 定义结构体 ReduceScatterOptions，用于设置归约分散操作的选项
struct ReduceScatterOptions {
  ReduceOp reduceOp = ReduceOp::SUM; // 归约分散操作的类型，默认为 SUM
  std::chrono::milliseconds timeout = kUnsetTimeout; // 归约分散操作的超时时间，默认为 kUnsetTimeout
  bool asyncOp = true; // 是否异步操作，默认为 true
};

// 定义结构体 AllToAllOptions，用于设置全对全操作的选项
struct AllToAllOptions {
  std::chrono::milliseconds timeout = kUnsetTimeout; // 全对全操作的超时时间，默认为 kUnsetTimeout
};

// 定义结构体 BarrierOptions，用于设置屏障同步操作的选项
struct BarrierOptions {
  std::vector<int64_t> device_ids; // 设备 ID 列表
  std::chrono::milliseconds timeout = kUnsetTimeout; // 屏障同步操作的超时时间，默认为 kUnsetTimeout
  std::optional<at::Device> device; // 设备，可选项
};

// 定义结构体 DistributedBackendOptions，用于设置分布式后端操作的选项
struct DistributedBackendOptions {
  c10::intrusive_ptr<::c10d::Store> store; // 分布式存储对象
  int group_rank; // 分组排名
  int group_size; // 分组大小
  std::chrono::duration<float> timeout; // 操作超时时间
  std::string group_id; // 分组 ID
  std::vector<int64_t> global_ranks_in_group; // 分组中的全局排名列表
};

} // namespace c10d
```