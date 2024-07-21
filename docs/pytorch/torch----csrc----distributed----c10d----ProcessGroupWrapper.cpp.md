# `.\pytorch\torch\csrc\distributed\c10d\ProcessGroupWrapper.cpp`

```py
#ifdef USE_C10D_GLOO
// 包含必要的头文件
#include <c10/core/Allocator.h>
#include <c10/core/DeviceType.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>
#include <c10/util/intrusive_ptr.h>
#include <c10/util/irange.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupGloo.hpp>
#include <stdexcept>
#include <utility>

namespace c10d {

namespace {
// 用于存储特定集合操作的信息，包括操作类型和输入张量（如果适用）的容器
struct CollectiveFingerPrint {
  // 当前集合操作的操作类型
  OpType op_type_;
  // 输入张量的数量
  std::size_t num_tensors_{};
  // 输入张量的数据类型
  std::vector<int8_t> tensor_dtypes_;
  // 输入张量的设备类型
  std::vector<int8_t> tensor_device_types_;
  // 输入张量的大小
  std::vector<std::vector<int64_t>> tensor_sizes_;
  // 序列号
  uint64_t sequence_number_;

  // 构造函数，根据输入张量创建集合指纹信息
  CollectiveFingerPrint(
      OpType op_type,
      const std::vector<at::Tensor>& input_tensors,
      uint64_t sequence_number)
      : op_type_(op_type),
        num_tensors_(input_tensors.size()),
        sequence_number_(sequence_number) {
    tensor_dtypes_.reserve(num_tensors_);
    tensor_device_types_.reserve(num_tensors_);
    tensor_sizes_.reserve(num_tensors_);
    for (const at::Tensor& t : input_tensors) {
      tensor_dtypes_.push_back(static_cast<int8_t>(t.dtype().toScalarType()));
      tensor_device_types_.push_back(static_cast<int8_t>(t.device().type()));
      tensor_sizes_.push_back(t.sizes().vec());
    }
  }

  // 从反序列化的指纹数据创建的构造函数
  CollectiveFingerPrint(
      OpType op_type,
      size_t num_tensors,
      std::vector<int8_t> tensor_dtypes,
      std::vector<int8_t> tensor_device_types,
      std::vector<std::vector<int64_t>> tensor_sizes,
      uint64_t sequence_number)
      : op_type_(op_type),
        num_tensors_(num_tensors),
        tensor_dtypes_(std::move(tensor_dtypes)),
        tensor_device_types_(std::move(tensor_device_types)),
        tensor_sizes_(std::move(tensor_sizes)),
        sequence_number_(sequence_number) {}

  // 在失败情况下记录集合信息的日志
  friend std::ostream& operator<<(
      std::ostream& output,
      const CollectiveFingerPrint& collective_fingerprint);

  // 执行和验证集合指纹
  void verify(c10::intrusive_ptr<Backend> backend) {
    // 序列化指纹信息为张量
    at::Tensor serialized_tensor = serialize_fingerprint();
    // 将序列化后的张量作为输入
    std::vector<at::Tensor> inp{serialized_tensor};
    // 首先验证张量的形状。这是必要的，因为如果张量的维度在进程间不匹配，
    // 直接验证张量将导致在allgather期间崩溃，我们实际上希望报告有关不一致性的描述。


这段代码定义了一个名为`CollectiveFingerPrint`的结构体，用于存储和验证分布式集合操作的信息，包括操作类型、输入张量的数据类型、设备类型和大小等。
    // 从输入张量获取形状信息，并验证张量的一致性
    std::vector<at::Tensor> sp = c10d::getTensorShapes(inp);
    // 验证张量列表的一致性
    verify_tensors(sp, backend);
    // 针对实际张量验证其一致性
    verify_tensors(inp, backend);
  }

  // 从 CollectiveFingerPrint::serialize_fingerprint 中获取序列化的指纹，
  // 并将其反序列化为 CollectiveFingerPrint 结构体
  CollectiveFingerPrint deserialize_fingerprint(
      const at::Tensor& serialized_tensor) {
    // 存储数据类型的向量
    auto dtypes = std::vector<int8_t>();
    // 存储设备类型的向量
    auto device_types = std::vector<int8_t>();
    // 存储张量形状的向量的向量
    auto sizes = std::vector<std::vector<int64_t>>();
    int index = 0;
    int64_t seq = 0;
    // 1. 操作类型
    auto optype = OpType(serialized_tensor[index].item<int>());
    index++;
    int num_tensors = 0;
    if (index < serialized_tensor.size(0)) {
      seq = serialized_tensor[index].item<int64_t>();
      index++;
      // 2. 张量数量
      num_tensors = serialized_tensor[index].item<int>();
      index++;
      dtypes.reserve(num_tensors);
      device_types.reserve(num_tensors);
      sizes.reserve(num_tensors);

      // 3. 张量数据类型
      for (int i = 0; i < num_tensors; i++) {
        dtypes.push_back(serialized_tensor[index].item<int8_t>());
        index++;
      }
      // 4. 设备类型
      for (int i = 0; i < num_tensors; i++) {
        device_types.push_back(serialized_tensor[index].item<int8_t>());
        index++;
      }
      // 5. 张量形状
      for (int i = 0; i < num_tensors; i++) {
        // 5a. 形状大小
        int size = serialized_tensor[index].item<int>();
        index++;
        // 5b. 形状
        auto shapeVec = std::vector<int64_t>();
        shapeVec.reserve(size);
        for (int j = 0; j < size; j++) {
          shapeVec.push_back(serialized_tensor[index].item<int64_t>());
          index++;
        }
        sizes.push_back(shapeVec);
      }
    }
    // 返回反序列化后的 CollectiveFingerPrint 结构体
    return CollectiveFingerPrint(
        optype, num_tensors, dtypes, device_types, sizes, seq);
  }

 private:
  // 验证张量的一致性
  void verify_tensors(
      std::vector<at::Tensor>& tensors_to_verify,
      c10::intrusive_ptr<Backend>& backend) {
    // 创建输出张量数据结构以传递给 allgather 函数
    std::vector<std::vector<at::Tensor>> output_tensors;
    // 输出张量列表: [<张量 0 的输出>, <张量 1 的输出>, ..., <张量 n 的输出>]
    output_tensors.reserve(tensors_to_verify.size());
    for (const auto& tensor_shape : tensors_to_verify) {
      // 每个秩级有自己的输出形状，例如
      // <张量 0 的输出>: [<秩 0 张量>, <秩 1 张量>, ..., <秩 n 张量>]
      std::vector<at::Tensor> outputs;
      outputs.reserve(backend->getSize());
      for (const auto i : c10::irange(backend->getSize())) {
        std::ignore = i; // 抑制未使用变量警告
        outputs.emplace_back(at::zeros_like(tensor_shape));
      }
      output_tensors.emplace_back(outputs);
    }
    // Allgather tensor shapes.
    backend->allgather(output_tensors, tensors_to_verify)->wait();
    // Verify equivalence
    for (const auto i : c10::irange(output_tensors.size())) {
      const std::vector<at::Tensor> gathered_tensors = output_tensors[i];
      const at::Tensor reference_tensor = tensors_to_verify[i];
      for (const auto rank : c10::irange(gathered_tensors.size())) {
        const auto& rank_tensor = gathered_tensors[rank];
        // 检查当前收集的张量与参考张量是否相等
        if (!rank_tensor.equal(reference_tensor)) {
          // 反序列化当前排名的指纹
          CollectiveFingerPrint rank_fingerprint =
              deserialize_fingerprint(rank_tensor);
          std::stringstream ss;
          ss << "Detected mismatch between collectives on ranks. Rank "
             << backend->getRank() << " is running collective: " << *this
             << ", but Rank " << rank
             << " is running collective: " << rank_fingerprint << ".";
          // 计算两个指纹之间的差异
          auto diff_result = compute_collective_diff(rank_fingerprint);
          // 如果存在差异，将其附加到错误信息中
          if (std::get<0>(diff_result)) {
            ss << std::get<1>(diff_result);
          }

          // 抛出 Torch 异常，指示不匹配
          TORCH_CHECK(false, ss.str());
        }
      }
    }
  }

  static std::vector<std::string> get_size_strs(
      const CollectiveFingerPrint& collective_fingerprint) {
    std::vector<std::string> size_strs;
    // 获取集合指纹中第一个张量大小的字符串表示
    if (!collective_fingerprint.tensor_sizes_.empty()) {
      for (const auto& single_tensor_shape_num :
           collective_fingerprint.tensor_sizes_[0]) {
        size_strs.emplace_back(std::to_string(single_tensor_shape_num));
      }
    }
    return size_strs;
  }

  static std::vector<std::string> get_dtype_strs(
      const CollectiveFingerPrint& collective_fingerprint) {
    std::vector<std::string> dtype_strs;
    // 获取集合指纹中张量数据类型的字符串表示
    dtype_strs.reserve(collective_fingerprint.tensor_dtypes_.size());
    for (const auto& tensor_dtype : collective_fingerprint.tensor_dtypes_) {
      dtype_strs.emplace_back(
          c10::toString(static_cast<at::ScalarType>(tensor_dtype)));
    }
    return dtype_strs;
  }

  static std::vector<std::string> get_device_type_strs(
      const CollectiveFingerPrint& collective_fingerprint) {
    std::vector<std::string> device_type_strs;
    // 获取集合指纹中张量设备类型的字符串表示
    device_type_strs.reserve(
        collective_fingerprint.tensor_device_types_.size());
    for (const auto& tensor_device_type :
         collective_fingerprint.tensor_device_types_) {
      device_type_strs.emplace_back(
          c10::toString(static_cast<at::DeviceType>(tensor_device_type)));
    }
    return device_type_strs;
  }

  std::pair<bool, std::string> compute_collective_diff(
      CollectiveFingerPrint& other) {
    // 计算两个集合的差异（序列号、张量大小、集合类型等），以便更容易理解在不同排名上不匹配的集合如何不同
    bool found_diff = false;
    std::stringstream ss;
    ss << "Collectives differ in the following aspects: ";
    // 检查序列号等是否相同
    // 在这里未完全展示
    // 如果其他对象的序列号与当前对象不同
    if (other.sequence_number_ != sequence_number_) {
      // 发现差异，设置标志为真
      found_diff = true;
      // 将差异信息添加到输出流中
      ss << c10::str(
          "\t Sequence number: ",
          sequence_number_,
          "vs ",
          other.sequence_number_);
    }
    
    // 检查操作类型
    auto other_op = opTypeToString(other.op_type_);
    auto this_op = opTypeToString(op_type_);
    // 如果其他对象的操作类型与当前对象不同
    if (other_op != this_op) {
      // 发现差异，设置标志为真
      found_diff = true;
      // 将差异信息添加到输出流中
      ss << c10::str("  Op type: ", this_op, "vs ", other_op);
    }

    // 定义一个函数 check，用于检查两个向量是否相同
    auto check = [&ss, &found_diff](
                     const char* arg,
                     std::vector<std::string> other,
                     std::vector<std::string> curr) {
      // 如果两个向量的大小不同
      if (other.size() != curr.size()) {
        // 发现差异，设置标志为真
        found_diff = true;
        // 将差异信息添加到输出流中
        ss << c10::str("  Tensor ", arg, ": ", curr, "vs ", other);
        return;
      }
      // 逐个比较向量中的元素
      for (size_t i = 0; i < other.size(); ++i) {
        // 如果发现不同的元素
        if (other[i] != curr[i]) {
          // 发现差异，设置标志为真
          found_diff = true;
          // 将差异信息添加到输出流中
          ss << c10::str("  Tensor ", arg, ": ", curr, "vs ", other);
          return;
        }
      }
    };

    // 检查张量的大小
    auto other_sizes = get_size_strs(other);
    auto this_sizes = get_size_strs(*this);
    check("Tensor shapes", other_sizes, this_sizes);

    // 检查张量的数据类型
    auto other_dtypes = get_dtype_strs(other);
    auto this_dtypes = get_dtype_strs(*this);
    check("Tensor dtypes", other_dtypes, this_dtypes);

    // 检查张量的设备类型
    auto other_devices = get_device_type_strs(other);
    auto this_devices = get_device_type_strs(*this);
    check("Tensor devices", other_devices, this_devices);
    
    // 如果没有发现差异
    if (!found_diff) {
      // 返回一个假值和当前输出流的字符串表示
      return std::make_pair(false, ss.str());
    } else {
      // 否则返回一个真值和当前输出流的字符串表示
      return std::make_pair(true, ss.str());
    }
  }

  // 将关于指纹的信息（操作类型、输入形状、数据类型、设备类型）序列化成张量
  at::Tensor serialize_fingerprint() {
    // 创建一个存储 int64_t 的独占指针
    auto data = std::make_unique<std::vector<int64_t>>();
    // 向数据向量中添加序列化后的信息

    // 1. OpType（操作类型）
    data->push_back(static_cast<int64_t>(op_type_));
    // sequence number（序列号）
    data->push_back(static_cast<int64_t>(sequence_number_));
    // 2. Num tensors（张量数量）
    data->push_back(static_cast<int64_t>(num_tensors_));
    // 3. Tensor dtypes（张量数据类型）
    for (const auto& type : tensor_dtypes_) {
      data->push_back(type);
    }
    // 4. Device types（设备类型）
    for (const auto& d : tensor_device_types_) {
      data->push_back(d);
    }
    // 5. Shapes（张量形状）
    for (const auto& sizes : tensor_sizes_) {
      data->push_back(static_cast<int64_t>(sizes.size()));
      for (const auto& s : sizes) {
        data->push_back(s);
      }
    }
    // 序列化数据为张量
    int64_t data_size = static_cast<int64_t>(data->size());
    // 由于 C++ 参数评估顺序，需要在这里释放并获取指针
    auto d = data.release();
    // 创建一个 ATen 的张量（Tensor），从给定的数据指针和数据大小开始
    at::Tensor serialized_tensor =
        at::for_blob(d->data(), {data_size})
            // 将上下文信息关联到张量，用于释放资源
            .context(
                d,
                [](void* ctx) {
                  delete static_cast<std::vector<int64_t>*>(ctx);
                })
            // 指定张量的选项，这里设置数据类型为长整型（kLong）
            .options(at::TensorOptions().dtype(at::kLong))
            // 根据上述配置生成张量对象
            .make_tensor();
    // 返回生成的张量对象
    return serialized_tensor;
}
};

// 重载输出流操作符，用于输出 CollectiveFingerPrint 对象信息
std::ostream& operator<<(
    std::ostream& output,
    const CollectiveFingerPrint& collective_fingerprint) {
  // 初始化一个空字符串用于存储 CollectiveFingerPrint 的信息
  std::string collectiveInfo;
  // 获取操作类型的字符串表示
  auto op_type_str = opTypeToString(collective_fingerprint.op_type_);
  // 如果张量数量不为零
  if (collective_fingerprint.num_tensors_ != 0) {
    // 获取张量数据类型的字符串表示
    std::vector<std::string> dtype_strs =
        CollectiveFingerPrint::get_dtype_strs(collective_fingerprint);
    // 获取张量设备类型的字符串表示
    std::vector<std::string> device_type_strs =
        CollectiveFingerPrint::get_device_type_strs(collective_fingerprint);
    // 获取张量尺寸的字符串表示
    std::vector<std::string> size_strs =
        CollectiveFingerPrint::get_size_strs(collective_fingerprint);

    // 构建包含所有信息的字符串
    collectiveInfo = c10::str(
        "CollectiveFingerPrint(",
        "SequenceNumber=",
        collective_fingerprint.sequence_number_,
        ", OpType=",
        op_type_str,
        ", TensorShape=[",
        c10::Join(", ", size_strs),
        "], TensorDtypes=",
        (dtype_strs),
        ", TensorDeviceTypes=",
        (device_type_strs),
        ")");
  } else {
    // 若没有张量信息，则构建简化的字符串
    collectiveInfo = c10::str(
        "CollectiveFingerPrint(",
        "SequenceNumber=",
        collective_fingerprint.sequence_number_,
        "OpType=",
        op_type_str,
        ")");
  }
  // 将构建好的信息字符串输出到流中
  return output << collectiveInfo;
}

// 检查输入张量集合是否具有相同的尺寸
bool check_same_size(const std::vector<at::Tensor>& input_tensors) {
  // 遍历所有输入张量
  for (const auto& input_tensor : input_tensors) {
    // 如果当前张量与第一个张量尺寸不同，则返回 false
    if (!input_tensors[0].is_same_size(input_tensor)) {
      return false;
    }
  }
  // 所有张量尺寸相同，返回 true
  return true;
}

// 实现 ProcessGroupWrapper 类的构造函数
ProcessGroupWrapper::ProcessGroupWrapper(
    const c10::intrusive_ptr<Backend>& backend,
    c10::intrusive_ptr<Backend> glooBackend)
    : Backend(backend->getRank(), backend->getSize()), // 调用基类构造函数进行初始化
      backend_(backend), // 初始化成员变量 backend_
      glooBackend_(std::move(glooBackend)) { // 初始化成员变量 glooBackend_
  // 为底层进程组设置序列号
  backend_->setSequenceNumberForGroup();
}

// 返回后端名称的函数实现
const std::string ProcessGroupWrapper::getBackendName() const {
  return backend_->getBackendName();
}

// 执行广播操作的函数实现
c10::intrusive_ptr<Work> ProcessGroupWrapper::broadcast(
    std::vector<at::Tensor>& data,
    const BroadcastOptions& opts) {
  // 运行集合通信操作的前置检查
  runCollectiveChecks(OpType::BROADCAST, data);
  // 调用底层 backend_ 的广播函数
  return backend_->broadcast(data, opts);
}

// 执行全reduce操作的函数实现
c10::intrusive_ptr<Work> ProcessGroupWrapper::allreduce(
    std::vector<at::Tensor>& data,
    const AllreduceOptions& opts) {
  // 运行集合通信操作的前置检查
  runCollectiveChecks(OpType::ALLREDUCE, data);
  // 调用底层 backend_ 的全reduce函数
  return backend_->allreduce(data, opts);
}

// 执行全reduce_coalesced操作的函数实现
c10::intrusive_ptr<Work> ProcessGroupWrapper::allreduce_coalesced(
    std::vector<at::Tensor>& tensors,
    const AllreduceCoalescedOptions& opts) {
  // 注意：对于 allreduce_coalesced 操作，我们不执行形状检查，因为其实现本身不强制要求，我们有测试用例使用不一致的形状，请参见 distributed_c10d 中的 Python 实现。
  // 运行集合通信操作的前置检查
  runCollectiveChecks(OpType::ALLREDUCE_COALESCED, {});
  // 调用底层 backend_ 的全reduce_coalesced函数
  return backend_->allreduce_coalesced(tensors, opts);
}

// 执行 reduce 操作的函数实现
c10::intrusive_ptr<Work> ProcessGroupWrapper::reduce(
    // 执行集合检查，确认操作类型为 REDUCE，并确保张量列表有效性
    runCollectiveChecks(OpType::REDUCE, tensors);
    // 调用后端实现的 reduce 函数，传入张量列表和选项参数，并返回结果
    return backend_->reduce(tensors, opts);
}

// 定义 ProcessGroupWrapper 类的 allgather 方法，执行收集操作并返回 Work 对象指针
c10::intrusive_ptr<Work> ProcessGroupWrapper::allgather(
    std::vector<std::vector<at::Tensor>>& outputTensors,  // 输出张量的二维向量
    std::vector<at::Tensor>& inputTensors,  // 输入张量向量
    const AllgatherOptions& opts) {  // allgather 操作的选项

  // 检查输出张量的最后一组是否具有相同的大小
  if (check_same_size(outputTensors.back())) {
    // 运行 OpType::ALLGATHER 类型的集体检查，使用输入张量作为参数
    runCollectiveChecks(OpType::ALLGATHER, inputTensors);
  } else {
    // 如果输出张量最后一组大小不同，则运行 OpType::ALLGATHER 类型的集体检查，不传递参数
    runCollectiveChecks(OpType::ALLGATHER, {});
  }

  // 调用 backend_ 的 allgather 方法执行实际的收集操作，并返回结果
  return backend_->allgather(outputTensors, inputTensors, opts);
}

// 定义 ProcessGroupWrapper 类的 _allgather_base 方法，执行基础的 allgather 操作并返回 Work 对象指针
c10::intrusive_ptr<Work> ProcessGroupWrapper::_allgather_base(
    at::Tensor& outputBuffer,  // 输出缓冲区张量
    at::Tensor& inputBuffer,   // 输入缓冲区张量
    const AllgatherOptions& opts) {  // allgather 操作的选项

  // 创建输入张量向量，其中包含唯一的输入缓冲区张量
  std::vector<at::Tensor> inputTensors({inputBuffer});

  // 运行 OpType::_ALLGATHER_BASE 类型的集体检查，使用输入张量向量作为参数
  runCollectiveChecks(OpType::_ALLGATHER_BASE, inputTensors);

  // 调用 backend_ 的 _allgather_base 方法执行基础的 allgather 操作，并返回结果
  return backend_->_allgather_base(outputBuffer, inputBuffer, opts);
}

// 定义 ProcessGroupWrapper 类的 allgather_coalesced 方法，执行聚合收集操作并返回 Work 对象指针
c10::intrusive_ptr<Work> ProcessGroupWrapper::allgather_coalesced(
    std::vector<std::vector<at::Tensor>>& outputTensorLists,  // 输出张量列表的二维向量
    std::vector<at::Tensor>& inputTensors,  // 输入张量向量
    const AllgatherOptions& opts) {  // allgather 操作的选项

  // 运行 OpType::ALLGATHER_COALESCED 类型的集体检查，不传递参数
  runCollectiveChecks(OpType::ALLGATHER_COALESCED, {});

  // 调用 backend_ 的 allgather_coalesced 方法执行实际的聚合收集操作，并返回结果
  return backend_->allgather_coalesced(outputTensorLists, inputTensors, opts);
}

// 定义 ProcessGroupWrapper 类的 gather 方法，执行聚集操作并返回 Work 对象指针
c10::intrusive_ptr<Work> ProcessGroupWrapper::gather(
    std::vector<std::vector<at::Tensor>>& outputTensors,  // 输出张量的二维向量
    std::vector<at::Tensor>& inputTensors,  // 输入张量向量
    const GatherOptions& opts) {  // gather 操作的选项

  // 运行 OpType::GATHER 类型的集体检查，使用输入张量向量作为参数
  runCollectiveChecks(OpType::GATHER, inputTensors);

  // 调用 backend_ 的 gather 方法执行实际的聚集操作，并返回结果
  return backend_->gather(outputTensors, inputTensors, opts);
}

// 定义 ProcessGroupWrapper 类的 scatter 方法，执行分散操作并返回 Work 对象指针
c10::intrusive_ptr<Work> ProcessGroupWrapper::scatter(
    std::vector<at::Tensor>& outputTensors,  // 输出张量向量
    std::vector<std::vector<at::Tensor>>& inputTensors,  // 输入张量的二维向量
    const ScatterOptions& opts) {  // scatter 操作的选项

  // 运行 OpType::SCATTER 类型的集体检查，使用输出张量向量作为参数
  runCollectiveChecks(OpType::SCATTER, outputTensors);

  // 调用 backend_ 的 scatter 方法执行实际的分散操作，并返回结果
  return backend_->scatter(outputTensors, inputTensors, opts);
}

// 定义 ProcessGroupWrapper 类的 reduce_scatter 方法，执行分散聚合操作并返回 Work 对象指针
c10::intrusive_ptr<Work> ProcessGroupWrapper::reduce_scatter(
    std::vector<at::Tensor>& outputTensors,  // 输出张量向量
    std::vector<std::vector<at::Tensor>>& inputTensors,  // 输入张量的二维向量
    const ReduceScatterOptions& opts) {  // reduce_scatter 操作的选项

  // 检查输入张量的最后一组是否具有相同的大小
  if (check_same_size(inputTensors.back())) {
    // 运行 OpType::REDUCE_SCATTER 类型的集体检查，使用输出张量向量作为参数
    runCollectiveChecks(OpType::REDUCE_SCATTER, outputTensors);
  } else {
    // 如果输入张量最后一组大小不同，则运行 OpType::REDUCE_SCATTER 类型的集体检查，不传递参数
    runCollectiveChecks(OpType::REDUCE_SCATTER, {});
  }

  // 调用 backend_ 的 reduce_scatter 方法执行实际的分散聚合操作，并返回结果
  return backend_->reduce_scatter(outputTensors, inputTensors, opts);
}

// 定义 ProcessGroupWrapper 类的 alltoall_base 方法，执行基础的 alltoall 操作并返回 Work 对象指针
c10::intrusive_ptr<Work> ProcessGroupWrapper::alltoall_base(
    at::Tensor& outputTensor,  // 输出张量
    at::Tensor& inputTensor,   // 输入张量
    std::vector<int64_t>& outputSplitSizes,  // 输出分割大小向量
    std::vector<int64_t>& inputSplitSizes,   // 输入分割大小向量
    const AllToAllOptions& opts) {  // alltoall 操作的选项

  // alltoall 支持不均匀的分割大小，因此不执行形状检查
  runCollectiveChecks(OpType::ALLTOALL_BASE, {});

  // 调用 backend_ 的 alltoall_base 方法执行实际的基础 alltoall 操作，并返回结果
  return backend_->alltoall_base(
      outputTensor, inputTensor, outputSplitSizes, inputSplitSizes, opts);
}
    // 调用 runCollectiveChecks 函数，验证操作类型为 ALLTOALL，但不进行任何形状检查
    void AllToAllStub::alltoall(
        std::vector<at::Tensor>& outputTensors,
        std::vector<at::Tensor>& inputTensors,
        const AllToAllOptions& opts) {
      runCollectiveChecks(OpType::ALLTOALL, {});
      // 调用后端 backend_ 的 alltoall 函数，执行 AllToAll 操作，并返回结果
      return backend_->alltoall(outputTensors, inputTensors, opts);
    }
}

// 定义 ProcessGroupWrapper 类的 monitoredBarrier 方法，实现对 barrier 的监控
void ProcessGroupWrapper::monitoredBarrier(
    const BarrierOptions& opts,  // 传入的 BarrierOptions 参数
    bool waitAllRanks) {         // 是否等待所有进程完成 barrier

  // 调用 backend_ 对象的 monitoredBarrier 方法
  return backend_->monitoredBarrier(opts, waitAllRanks);
}

// 定义 ProcessGroupWrapper 类的 setSequenceNumberForGroup 方法
void ProcessGroupWrapper::setSequenceNumberForGroup() {
  // 如果 underlying pg 的 sequence number 尚未设置为 0，则设置它
  if (backend_->getSequenceNumberForGroup() == 0) {
    // 调用 backend_ 对象的 setSequenceNumberForGroup 方法
    backend_->setSequenceNumberForGroup();
  }
}

// 定义 ProcessGroupWrapper 类的 getSequenceNumberForGroup 方法
uint64_t ProcessGroupWrapper::getSequenceNumberForGroup() {
  // 获取 underlying pg 的 sequence number
  return backend_->getSequenceNumberForGroup();
}

// 定义 ProcessGroupWrapper 类的 send 方法
c10::intrusive_ptr<Work> ProcessGroupWrapper::send(
    std::vector<at::Tensor>& tensors,  // 待发送的张量向量
    int dstRank,                       // 目标进程的排名
    int tag) {                         // 标签
  // 调用 backend_ 对象的 send 方法
  return backend_->send(tensors, dstRank, tag);
}

// 定义 ProcessGroupWrapper 类的 recv 方法
c10::intrusive_ptr<Work> ProcessGroupWrapper::recv(
    std::vector<at::Tensor>& tensors,  // 接收到的张量向量
    int srcRank,                       // 源进程的排名
    int tag) {                         // 标签
  // 调用 backend_ 对象的 recv 方法
  return backend_->recv(tensors, srcRank, tag);
}

// 定义 ProcessGroupWrapper 类的 recvAnysource 方法
c10::intrusive_ptr<Work> ProcessGroupWrapper::recvAnysource(
    std::vector<at::Tensor>& tensors,  // 接收到的张量向量
    int tag) {                         // 标签
  // 调用 backend_ 对象的 recvAnysource 方法
  return backend_->recvAnysource(tensors, tag);
}

// 定义 ProcessGroupWrapper 类的 barrier 方法
c10::intrusive_ptr<Work> ProcessGroupWrapper::barrier(
    const BarrierOptions& opts) {  // 传入的 BarrierOptions 参数
  // 运行 collective 检查
  runCollectiveChecks(OpType::BARRIER, {});
  // 调用 backend_ 对象的 barrier 方法
  return backend_->barrier(opts);
}

// 定义 ProcessGroupWrapper 类的 _reduce_scatter_base 方法
c10::intrusive_ptr<Work> ProcessGroupWrapper::_reduce_scatter_base(
    at::Tensor& outputBuffer,                // 输出缓冲区张量
    at::Tensor& inputBuffer,                 // 输入缓冲区张量
    const ReduceScatterOptions& opts) {      // 传入的 ReduceScatterOptions 参数
  // 运行 collective 检查
  runCollectiveChecks(
      OpType::_REDUCE_SCATTER_BASE, {inputBuffer, outputBuffer});
  // 调用 backend_ 对象的 _reduce_scatter_base 方法
  return backend_->_reduce_scatter_base(outputBuffer, inputBuffer, opts);
}

// 定义 ProcessGroupWrapper 类的 startCoalescing 方法
void ProcessGroupWrapper::startCoalescing() {
  // 调用 backend_ 对象的 startCoalescing 方法
  return backend_->startCoalescing();
}

// 定义 ProcessGroupWrapper 类的 endCoalescing 方法
c10::intrusive_ptr<Work> ProcessGroupWrapper::endCoalescing() {
  // 调用 backend_ 对象的 endCoalescing 方法
  return backend_->endCoalescing();
}

// 定义 ProcessGroupWrapper 类的 getWrappedPg 方法，返回 backend_ 对象
c10::intrusive_ptr<Backend> ProcessGroupWrapper::getWrappedPg() const {
  return backend_;
}

// 定义 ProcessGroupWrapper 类的 runCollectiveChecks 方法
void ProcessGroupWrapper::runCollectiveChecks(
    OpType op_type,                          // collective 操作类型
    const std::vector<at::Tensor>& tensors) {  // 待处理的张量向量
  // 执行 monitored barrier 来确保所有进程同步
  c10d::BarrierOptions options;
  // TODO: 应该在这里使用封装的 backend_ 的超时时间，但 C++ ProcessGroup API 并未暴露超时时间
  auto seq = getSequenceNumberForGroup();
  // 创建 CollectiveFingerPrint 对象来记录 collective 的指纹信息
  auto finger_print = CollectiveFingerPrint(op_type, tensors, seq);
  // 记录日志，显示当前正在运行的 collective 操作信息
  LOG(INFO) << "[Rank " << getRank() << "] "
            << "Running collective: " << finger_print;
  try {
    // 调用 glooBackend_ 对象的 monitoredBarrier 方法，确保所有进程同步
    glooBackend_->monitoredBarrier(options, /* waitAllRanks */ true);
  } catch (const std::runtime_error& e) {
    // 将 collective 信息附加到异常信息中，并重新抛出异常
    std::stringstream ss;
    ss << finger_print;
    auto collective_info = ss.str();
    auto err_msg = c10::str(
        "ProcessGroupWrapper: Monitored Barrier encountered error running collective: ",
        collective_info,
        ". Error: \n",
        e.what());
    TORCH_CHECK(false, err_msg);
  }
  // 如果检测到非法形式的 collective，将会抛出异常
  finger_print.verify(glooBackend_);
}
} // namespace c10d
```