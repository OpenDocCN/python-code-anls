# `.\pytorch\torch\csrc\profiler\collection.cpp`

```
// 包含 Torch 的性能分析模块中的相关头文件
#include <torch/csrc/profiler/collection.h>
#include <torch/csrc/profiler/orchestration/vulkan.h>

// 标准库头文件
#include <algorithm>
#include <functional>
#include <limits>
#include <memory>
#include <queue>
#include <type_traits>
#include <utility>

// 使用 fmt 库进行格式化输出
#include <fmt/format.h>

// 如果定义了 USE_KINETO 宏，则包含 kineto 库
#ifdef USE_KINETO
#include <libkineto.h>
#endif

// Torch 核心头文件
#include <ATen/Context.h>
#include <ATen/record_function.h>
#include <c10/util/Exception.h>
#include <c10/util/flat_hash_map.h>
#include <c10/util/overloaded.h>
#include <torch/csrc/jit/runtime/interpreter.h>
#include <torch/csrc/profiler/data_flow.h>
#include <torch/csrc/profiler/kineto_shim.h>

// 定义 Torch 的性能分析命名空间
namespace torch::profiler::impl {

// 使用 typedef 简化类型名称
using result_ptr_t = std::shared_ptr<Result>;
using trace_ptr_t = std::unique_ptr<torch::profiler::impl::kineto::ActivityTraceWrapper>;

// RawTensorMetadataBase 类的构造函数，初始化原始张量元数据
RawTensorMetadataBase::RawTensorMetadataBase(const at::Tensor& t)
    : data_{t.has_storage() ? t.storage().data() : nullptr},  // 存储数据指针
      dtype_{t.scalar_type()},                                // 标量类型
      layout_{t.layout()},                                    // 张量布局
      dim_{static_cast<uint32_t>(t.sizes().size())} {         // 张量维度数
  // 断言调试模式下张量维度不超过 uint32_t 的最大值
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      t.sizes().size() <= std::numeric_limits<uint32_t>::max(),
      "Cannot profile Tensors of size > uint32 max. Got dim: ",
      t.sizes().size());
}

// RawTensorMetadata 类的构造函数，初始化原始张量元数据
RawTensorMetadata::RawTensorMetadata(const at::Tensor& t)
    : RawTensorMetadataBase(t),                                // 调用基类构造函数
      weak_self_{WeakTensor(t)},                               // 弱引用到张量的自身
      device_type_{t.device().type()},                         // 设备类型
      device_index_{t.device().index()} {}                     // 设备索引

// TensorMetadata 类的构造函数，初始化张量元数据
TensorMetadata::TensorMetadata(
    const RawTensorMetadata& r,
    std::vector<int64_t> sizes,
    std::vector<int64_t> strides)
    // NOLINTNEXTLINE(cppcoreguidelines-slicing)
    : RawTensorMetadataBase(r),                                // 调用基类构造函数
      weak_self_{r.weak_self_.value_or(WeakTensor(at::Tensor()))},  // 弱引用到张量的自身
      device_{r.device_type_, r.device_index_},                // 设备类型和索引
      sizes_{std::move(sizes)},                                // 张量尺寸
      strides_{std::move(strides)} {                           // 张量步长
  // 断言弱引用到张量自身存在
  SOFT_ASSERT(r.weak_self_.has_value());
}

// ============================================================================
// == PyTorch 操作相关定义 =====================================================
// ============================================================================

// 匿名命名空间用于定义标签到 I/O 类型的映射关系
namespace {
struct TagToIOType {
  InputOutputEncoder::Tag tag;                                 // 输入输出编码器的标签
  InputOutputEncoder::IOType io_type;                          // 输入输出编码器的 I/O 类型
};

// 输入输出编码器标签数量
constexpr int tagCount = ((int)InputOutputEncoder::Tag::TERMINATOR) + 1;

// 标签到 I/O 类型的映射数组
constexpr std::array<TagToIOType, tagCount> tag_map = {{
    {InputOutputEncoder::Tag::Tensor, InputOutputEncoder::IOType::Shapes},  // 张量 -> 形状
    {InputOutputEncoder::Tag::UndefinedTensor, InputOutputEncoder::IOType::Shapes},  // 未定义张量 -> 形状
    {InputOutputEncoder::Tag::TensorListBegin, InputOutputEncoder::IOType::Shapes},  // 张量列表开始 -> 形状
    {InputOutputEncoder::Tag::ScalarList, InputOutputEncoder::IOType::ConcreteInputs},  // 标量列表 -> 具体输入
    {InputOutputEncoder::Tag::Scalar, InputOutputEncoder::IOType::Shapes},  // 标量 -> 形状
    {InputOutputEncoder::Tag::Other, InputOutputEncoder::IOType::Shapes},  // 其他 -> 形状
    {InputOutputEncoder::Tag::TERMINATOR, InputOutputEncoder::IOType::None},  // 终结符 -> 无
}};
// 检查是否所有的标签在tag_map数组中都有正确映射，直到遇到终止标记或idx超出数组边界
constexpr bool allTagsMapped(int idx = 0) {
  return tag_map[idx].tag == InputOutputEncoder::Tag::TERMINATOR ||
      ((idx == (int)tag_map[idx].tag) && allTagsMapped(idx + 1));
}

// 使用静态断言确保所有标签在tag_map数组中的映射都是有序的
static_assert(allTagsMapped(), "tag_map is out of order");

// 根据给定的标签从tag_map数组中获取对应的IO类型
constexpr InputOutputEncoder::IOType tagToIOType(InputOutputEncoder::Tag tag) {
  return tag_map[(int)tag].io_type;
}

} // namespace

// ----------------------------
// |  输入/输出编码器           |
// ----------------------------

// 将一组IValue值推入编码器中进行处理
void InputOutputEncoder::push(c10::ArrayRef<const c10::IValue> values) {
  for (const auto& value : values) {
    if (value.isTensor()) {
      push(value.toTensor());
    } else if (value.isScalar()) {
      tags_.emplace_back(Tag::Scalar);
      // 标量小到可以直接存储在ivalues中，无需额外的内存分配
      // TODO: 可以通过给Profiler访问IValue的内部来进一步优化此过程
      ivalues_.emplace_back(value);
    } else if (value.isTensorList()) {
      tags_.emplace_back(Tag::TensorListBegin);
      // 将Tensor列表的开始标记放入tags_
      for (const auto& t : value.toTensorList()) {
        push(t);
      }
      tags_.emplace_back(Tag::TERMINATOR);
      // 在处理完Tensor列表后，放入终止标记
    } else if (isSupportedScalarList(value)) {
      tags_.emplace_back(Tag::ScalarList);
      // 将标量列表的标记放入tags_
      ivalues_.emplace_back(value);
    } else {
      tags_.emplace_back(Tag::Other);
      // 若值类型无法识别，则将Other标记放入tags_
    }
  }
  tags_.emplace_back(Tag::TERMINATOR);
  // 在处理完所有值后，放入终止标记
}

// 将一个Tensor对象推入编码器中进行处理
void InputOutputEncoder::push(const at::Tensor& t) {
  // TODO: 修复嵌套和符号大小的问题
  if (t.defined() && !t.is_nested() &&
      !t.unsafeGetTensorImpl()->has_symbolic_sizes_strides()) {
    tags_.emplace_back(Tag::Tensor);
    // 将Tensor标记放入tags_
    tensor_metadata_.emplace_back(t);
    tensor_sizes_strides_.copy(t.sizes());
    if (t.layout() == at::kStrided) {
      // 只有Strided布局的Tensor才有strides
      tensor_sizes_strides_.copy(t.strides());
    }
  } else {
    tags_.emplace_back(Tag::UndefinedTensor);
    // 若Tensor未定义或具有嵌套或符号大小，则将UndefinedTensor标记放入tags_
  }
}

// 判断给定的IValue值是否是支持的标量列表
bool InputOutputEncoder::isSupportedScalarList(
    const c10::IValue& list_candidate) {
  // 标量列表可能非常长。如果列表过长，应当避免收集它们。该函数检查列表是否为标量列表以及长度是否适当。

  if (!get_record_concrete_inputs_enabled()) {
    return false;
  }

  if (!list_candidate.isList()) {
    return false;
  }
  auto list_ref = list_candidate.toListRef();
  if (C10_UNLIKELY(list_ref.empty())) {
    return true;
  }
  if (C10_UNLIKELY(!list_ref[0].isScalar())) {
    return false;
  }
  if (C10_UNLIKELY(list_ref.size() > SCALAR_LIST_LENGTH_LIMIT)) {
    return false;
  }
  return true;
}

// 该函数返回一个lambda函数，用作自定义迭代器样式的getter。
// 每次调用lambda函数时，返回一个操作的输入值。
// 根据输入的 io_type 返回一个函数对象，该函数对象用于生成对应类型的输入值
auto InputOutputEncoder::getIValueGenerator(const IOType& io_type) {
  // 返回一个 lambda 表达式，捕获了当前对象的成员变量和输入的 io_type
  return [this,
          // 使用 tags_ 容器的起始迭代器进行初始化
          tag_it = tags_.begin(),
          // 使用 tensor_metadata_ 容器的起始迭代器进行初始化
          tensor_metadata_it = tensor_metadata_.begin(),
          // 使用 tensor_sizes_strides_ 容器的起始迭代器进行初始化
          tensor_size_strides_it = tensor_sizes_strides_.begin(),
          // 使用 ivalues_ 容器的起始迭代器进行初始化
          ivals_it = ivalues_.begin(),
          // 捕获 io_type 参数
          io_type]() mutable {
    // 定义一个 lambda 函数 decode_tensor，用于解析张量的元数据
    auto decode_tensor = [&]() -> TensorMetadata {
      // 获取当前 tensor_metadata_it 指向的元数据
      const auto& raw_metadata = *tensor_metadata_it++;
      // 定义用于存储张量大小和步长的容器
      std::vector<int64_t> sizes;
      std::vector<int64_t> strides;
      // 根据元数据中的维度信息，填充 sizes 容器
      for (C10_UNUSED const auto _ : c10::irange(raw_metadata.dim_)) {
        sizes.push_back(*tensor_size_strides_it++);
      }
      // 如果元数据的布局是 kStrided，则填充 strides 容器
      if (raw_metadata.layout_ == at::kStrided) {
        for (C10_UNUSED const auto _ : c10::irange(raw_metadata.dim_)) {
          strides.push_back(*tensor_size_strides_it++);
        }
      }
      // 返回解析后的 TensorMetadata 对象，包含原始元数据、大小和步长信息
      return {raw_metadata, sizes, strides};
    };

    // 定义用于存储操作输入值的容器
    std::vector<op_input_t> out;
    // 定义一个 lambda 函数 push_value，根据标签将输入值推送到 out 容器中
    auto push_value = [&out, io_type](const Tag& tag, op_input_t input) {
      // 如果 io_type 与 tag 对应的输入输出类型匹配，则将 input 推送到 out 容器
      if (io_type == tagToIOType(tag)) {
        out.emplace_back(std::move(input));
      } else {
        // 否则，推送一个空值到 out 容器
        out.emplace_back(c10::nullopt);
      }
    };

    // 定义一个标志位，指示是否终止循环
    bool terminate = false;
    // 遍历 tags_ 容器，处理每一个 tag
    while (!terminate && tag_it != tags_.end()) {
      // 根据当前 tag 进行分支处理
      switch (*tag_it) {
        case Tag::Tensor:
          // 处理 Tensor 类型的 tag，将解析后的张量元数据推送到 out 容器
          push_value(*tag_it, decode_tensor());
          break;

        case Tag::TensorListBegin: {
          // 处理 TensorListBegin 类型的 tag
          std::vector<TensorMetadata> arg;
          // 标志位，指示是否找到 UndefinedTensor 类型的 tag
          bool found_undefined = false;
          // 遍历直到遇到 TERMINATOR 类型的 tag
          while (*(++tag_it) != Tag::TERMINATOR) {
            // 如果遇到 UndefinedTensor 类型的 tag，更新标志位并继续
            if (*tag_it == Tag::UndefinedTensor) {
              found_undefined = true;
              continue;
            }
            // 断言当前 tag 为 Tensor 类型
            TORCH_INTERNAL_ASSERT(*tag_it == Tag::Tensor, (int)(*tag_it));
            // 解析张量元数据并添加到 arg 容器中
            arg.emplace_back(decode_tensor());
          }
          // 根据是否找到 UndefinedTensor 类型的 tag，推送相应的值到 out 容器
          if (found_undefined) {
            push_value(*tag_it, c10::nullopt);
          } else {
            push_value(Tag::TensorListBegin, std::move(arg));
          }
        } break;

        case Tag::ScalarList:
        case Tag::Scalar:
          // 处理 ScalarList 和 Scalar 类型的 tag，推送当前 ivalues_ 容器中的值到 out
          push_value(*tag_it, *ivals_it++);
          break;

        case Tag::UndefinedTensor:
        case Tag::Other:
          // 处理 UndefinedTensor 和 Other 类型的 tag，推送空值到 out 容器
          push_value(*tag_it, c10::nullopt);
          break;

        case Tag::TERMINATOR:
          // 处理 TERMINATOR 类型的 tag，标志着操作的结束
          terminate = true;
          break;

        default:
          break;
      }
      // 更新 tag_it 迭代器
      ++tag_it;
    }
    // 返回存储操作输入值的 out 容器
    return out;
  };
}

// 返回 Shapes 类型输入值的生成器
auto InputOutputEncoder::getInputShapeGenerator() {
  return getIValueGenerator(IOType::Shapes);
}

// 返回 ConcreteInputs 类型输入值的生成器
auto InputOutputEncoder::getConcreteInputGenerator() {
  return getIValueGenerator(IOType::ConcreteInputs);
}

// 清空 tags_, tensor_metadata_, tensor_sizes_strides_, ivalues_ 容器
void InputOutputEncoder::clear() {
  tags_.clear();
  tensor_metadata_.clear();
  tensor_sizes_strides_.clear();
  ivalues_.clear();
}
// 构造函数 EventBlock<T, ChunkSize> 的实现
template <typename T, size_t ChunkSize>
ThreadLocalSubqueue::TorchOpStorage::EventBlock<T, ChunkSize>::EventBlock() {
  // 静态原子计数器，用于生成唯一的起始 ID
  static std::atomic<uint64_t> counter_{0};
  // 设置起始 ID，每次递增 ChunkSize
  id_start_ = 1 + ChunkSize * counter_++;
}

// 向 OpList 中添加新事件的函数模板实现
template <class... Args>
std::pair<KinetoObserverContext::Event*, uint64_t> ThreadLocalSubqueue::
    TorchOpStorage::OpList::emplace_back(Args&&... args) {
  // 调用基类的 emplace_back 函数，添加新事件
  auto event_ptr = AppendOnlyList::emplace_back(std::forward<Args>(args)...);
  // 获取新事件对应的关联 ID
  auto corr_id = buffer_last_->correlation_id(event_ptr);
  return {event_ptr, corr_id};
}

// 获取 OpList 迭代器指向事件的关联 ID
uint64_t ThreadLocalSubqueue::TorchOpStorage::OpList::correlationID(
    const OpList::Iterator& e) {
  return e.address().first->correlation_id(&*e);
}

// 获取 EventBlock<T, ChunkSize> 中指针 ptr 所指事件的关联 ID
template <typename T, size_t ChunkSize>
uint64_t ThreadLocalSubqueue::TorchOpStorage::EventBlock<T, ChunkSize>::
    correlation_id(const T* ptr) const {
  // 断言：确保 ptr 在 EventBlock 数据范围内
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      ptr >= this->data() && ptr < this->data() + ChunkSize);
  // 计算并返回事件的关联 ID
  return id_start_ + (ptr - this->data());
}

// ---------------------------------
// |  Collection (Observer logic)  |
// ---------------------------------

// 开始记录操作时创建 KinetoObserverContext 对象
std::unique_ptr<KinetoObserverContext> ThreadLocalSubqueue::begin_op(
    const at::RecordFunction& fn) {
  // 向 op_events_ 中添加新的 TorchOpBasicFields 对象
  auto [event, corr_id] = torch_ops_.op_events_.emplace_back(
      torch::profiler::impl::TorchOpBasicFields{
          fn.seqNr(),
          fn.forwardThreadId(),
          fn.scope(),
          fn.isAsync(),
          fn.handle(),
          fn.debugHandle(),
          fn.name()});
  
  // 如果配置中需要报告输入形状，则添加输入参数
  if (config_.report_input_shapes) {
    torch_ops_.inputs_outputs_.push(fn.inputs());
  }
  
  // 如果是用户定义的作用域，则使用用户关联 ID
  if (fn.scope() == at::RecordScope::USER_SCOPE) {
    torch::profiler::impl::kineto::pushUserCorrelationId(corr_id);
  } else { // 否则使用一般关联 ID
    torch::profiler::impl::kineto::pushCorrelationId(corr_id);
  }

#if !defined BUILD_LITE_INTERPRETER && !defined C10_MOBILE
  // 如果配置允许并且不是反向函数作用域，则记录调用栈信息
  if (config_.with_stack && fn.scope() != at::RecordScope::BACKWARD_FUNCTION) {
    auto cs = torch::profiler::impl::prepareCallstack(jit::currentCallstack());
    torch_ops_.jit_stack_.emplace_back(callstackStr(cs));
  }
  
  // 如果配置允许并且不是反向函数作用域，则记录模块层次信息
  if (config_.with_modules &&
      fn.scope() != at::RecordScope::BACKWARD_FUNCTION) {
    torch_ops_.jit_modules_.emplace_back(jit::currentModuleHierarchy());
  }
#endif

  // 如果配置中需要记录 FLOPs，则保存额外参数
  if (config_.with_flops) {
    torch_ops_.extra_args_.emplace_back(
        torch::profiler::impl::saveExtraArgs(fn));
  }

  // 如果是 NCCL 元数据相关的 CPU 操作，则记录 NCCL 元数据
  fn.isNcclMeta() ? torch_ops_.extra_meta_.emplace_back(
                        torch::profiler::impl::saveNcclMeta(fn))
                  : torch_ops_.extra_meta_.emplace_back();

  // 创建并返回 KinetoObserverContext 对象
  auto out = std::make_unique<KinetoObserverContext>(event);

  // 如果配置状态为 KINETO_GPU_FALLBACK，则执行下列操作
  if (config_.state == ProfilerState::KINETO_GPU_FALLBACK) {
    try {
      // 尝试记录 CUDA 事件，使用设备的回退机制
      out->fallback_ = torch_ops_.device_fallback_.emplace_back();
      torch::profiler::impl::cudaStubs()->record(
          nullptr, &out->fallback_->device_event_start_, nullptr);
    } catch (const std::exception& e) {
      // 捕获异常，记录错误日志并打印异常信息
      LOG(WARNING) << "Failed to record CUDA event. " << e.what();
    }
  } else if (config_.state == ProfilerState::KINETO_PRIVATEUSE1_FALLBACK) {
    // 如果配置状态表明使用私有备用1回退机制，则记录私有1类型的事件
    out->fallback_ = torch_ops_.device_fallback_.emplace_back();
    torch::profiler::impl::privateuse1Stubs()->record(
        nullptr, &out->fallback_->device_event_start_, nullptr);
  }

  // 设置事件的起始时间为近似时间
  event->start_time_ = c10::getApproximateTime();
  // 允许使用 TF32 CuBLAS，根据全局上下文的配置
  event->allow_tf32_cublas_ = at::globalContext().allowTF32CuBLAS();
  // 如果性能事件配置非空，则创建性能计数器并启用性能分析器
  if (!config_.experimental_config.performance_events.empty()) {
    const size_t n = config_.experimental_config.performance_events.size();
    // 创建性能计数器对象，数量为配置中指定的事件个数，初始化为0
    event->counters_ = std::make_unique<perf_counters_t>(n, 0);
    // 启用性能分析器
    perf_profiler_->Enable();
  }
  // 返回处理后的事件对象
  return out;
// 结构模板，用于从给定容器中"偷取"或默认获取元素
namespace {
template <typename T>
struct StealOrDefault {
  StealOrDefault(T& container)
      : container_{container}, it_{container.begin()} {}  // 初始化，使用容器的起始迭代器

  ~StealOrDefault() {
    container_.get().clear();  // 析构函数，清空容器
  }

  // 函数调用操作符重载，返回容器中的元素值类型
  typename T::Iterator::value_type operator()() {
    if (it_.exhausted()) {  // 如果迭代器已经耗尽
      return typename T::Iterator::value_type();  // 返回默认构造的值类型对象
    } else {
      auto result = std::move(*it_);  // 移动当前迭代器指向的元素
      ++it_;  // 迭代器递增
      return result;  // 返回移动后的元素
    }
  }

  std::reference_wrapper<T> container_;  // 引用包装器，引用给定的容器
  typename T::Iterator it_;  // 迭代器对象
};
} // namespace

// 实现 ThreadLocalSubqueue::TorchOpStorage 类的 materialize 方法
void ThreadLocalSubqueue::TorchOpStorage::materialize(
    std::vector<std::shared_ptr<Result>>& out,  // 输出结果向量的引用
    const std::function<c10::time_t(c10::approx_time_t)>& time_converter,  // 时间转换函数对象
    const uint64_t tid,  // 线程 ID
    const kineto::DeviceAndResource& kineto_info) {  // 设备和资源信息对象

  // 将 Autograd 信息传递给顶层注释
  auto it = op_events_.begin();  // 获取操作事件的起始迭代器
  for (C10_UNUSED const auto _ :
       c10::irange(static_cast<int64_t>(op_events_.size()) - 1)) {  // 循环遍历操作事件
    auto& first = it->basic_fields_;  // 获取第一个事件的基本字段引用
    auto& second = (++it)->basic_fields_;  // 获取下一个事件的基本字段引用
    if (first.scope_ == at::RecordScope::FUNCTION &&  // 如果第一个事件的作用域为 FUNCTION
        second.scope_ == at::RecordScope::BACKWARD_FUNCTION &&  // 且第二个事件的作用域为 BACKWARD_FUNCTION
        first.name_.rfind("autograd::engine::evaluate_function: ", 0) == 0) {  // 并且第一个事件的名称以指定前缀开头
      first.sequence_number_ = second.sequence_number_;  // 设置第一个事件的序列号为第二个事件的序列号
      first.forward_tid_ = second.forward_tid_;  // 设置第一个事件的前向线程 ID 为第二个事件的前向线程 ID
    }
  }

  // "AccumulateGrad" 是性能分析的重要标记；然而，该注释依赖于平台相关的 c10::demangle 函数
  const std::string accumulate_grad = "torch::autograd::AccumulateGrad";  // 定义积累梯度的字符串
  const std::string windows_pattern = std::string("struct ") + accumulate_grad;  // 在 Windows 平台上添加 "struct " 前缀
  for (auto& event : op_events_) {  // 遍历操作事件
    auto& name = event.basic_fields_.name_;  // 获取事件的名称引用
    auto position = name.find(windows_pattern);  // 查找名称中是否包含 Windows 模式
    if (position != std::string::npos) {  // 如果找到匹配位置
      name.replace(position, windows_pattern.size(), accumulate_grad);  // 替换为积累梯度的字符串
    }
  }

  auto input_shape_getter = inputs_outputs_.getInputShapeGenerator();  // 获取输入形状生成器
  auto concrete_input_getter = inputs_outputs_.getConcreteInputGenerator();  // 获取具体输入生成器

  // 使用 StealOrDefault 模板实例化对象，"偷取"或默认获取各种成员变量
  auto jit_stack = StealOrDefault<decltype(jit_stack_)>(jit_stack_);
  auto jit_module = StealOrDefault<decltype(jit_modules_)>(jit_modules_);
  auto extra_args = StealOrDefault<decltype(extra_args_)>(extra_args_);
  auto extra_meta = StealOrDefault<decltype(extra_meta_)>(extra_meta_);
  auto gpu_fallback =
      StealOrDefault<decltype(device_fallback_)>(device_fallback_);

  for (auto event = op_events_.begin(); event != op_events_.end(); ++event) {
    // 循环遍历操作事件
    ExtraFields<EventType::TorchOp> e{
        std::move(event->basic_fields_),    // 使用event的基本字段创建ExtraFields对象，并进行移动语义转移
        ThreadLocalSubqueue::TorchOpStorage::OpList::correlationID(event),    // 调用correlationID方法获取event的相关ID
        time_converter(event->end_time_),   // 使用time_converter函数转换event的结束时间
        input_shape_getter(),               // 调用input_shape_getter函数获取输入形状信息
        concrete_input_getter(),            // 调用concrete_input_getter函数获取具体输入信息
        jit_stack(),                        // 创建jit_stack对象
        jit_module(),                       // 创建jit_module对象
        extra_args(),                       // 创建extra_args对象
        extra_meta(),                       // 创建extra_meta对象
        gpu_fallback(),                     // 创建gpu_fallback对象
        event->allow_tf32_cublas_,          // 设置tf32_cublas允许标志
        std::move(event->counters_)         // 使用event的计数器字段创建对象，并进行移动语义转移
    };

    out.emplace_back(Result::create(
        time_converter(event->start_time_),  // 使用time_converter函数转换event的开始时间
        tid,                                 // 传入tid作为线程ID
        kineto_info,                         // 传入kineto_info对象
        std::move(e)));                      // 将e对象移动构造并放入out容器的末尾

  }

  op_events_.clear();                       // 清空op_events_容器
  inputs_outputs_.clear();                  // 清空inputs_outputs_容器
// 使用模板定义一个函数 materialize_vulkan，该函数接受以下参数：
//   - out：存储结果的共享指针向量
//   - raw_events：具有模板大小 BlockSize 的附加字段列表，其中包含 Vulkan 事件类型的原始事件
//   - time_converter：将 c10::approx_time_t 转换为 c10::time_t 的函数对象
//   - tid：线程 ID
//   - kineto_info：设备和资源信息
template <size_t BlockSize>
void materialize_vulkan(
    std::vector<std::shared_ptr<Result>>& out,
    AppendOnlyList<ExtraFields<EventType::Vulkan>::raw_event_t, BlockSize>& raw_events,
    const std::function<c10::time_t(c10::approx_time_t)>& time_converter,
    const uint64_t tid,
    const kineto::DeviceAndResource& kineto_info) {
  
  // 遍历 raw_events 中的每个元素 i
  for (const auto& i : raw_events) {
    // 获取 Vulkan 事件 i.second 的着色器名称和持续时间（单位：纳秒）
    const auto name_and_duration_ns = torch::profiler::impl::vulkan::getShaderNameAndDurationNs(i.second);

    // 将结果添加到 out 中，使用 Result::create 方法创建 Result 对象，传递以下参数：
    //   - start_time_ns_：通过 time_converter 转换 i.first 得到的开始时间（单位：纳秒）
    //   - start_tid_：tid，即线程 ID
    //   - kineto_info_：kineto_info，设备和资源信息
    //   - extra_fields_：使用 Vulkan 事件类型的额外字段对象 ExtraFields<EventType::Vulkan> 初始化，包含以下内容：
    //       - name_：着色器名称
    //       - duration_ns_：持续时间的整数表示（单位：纳秒）
    //       - in_tree_building_：false，表示不在树构建中
    out.emplace_back(Result::create(
        /*start_time_ns_=*/time_converter(i.first),
        /*start_tid_=*/tid,
        /*kineto_info_=*/kineto_info,
        /*extra_fields_=*/
        ExtraFields<EventType::Vulkan>{
            /*name_=*/std::get<0>(name_and_duration_ns),
            /*duration_ns_=*/
            static_cast<int64_t>(std::get<1>(name_and_duration_ns)),
            /*in_tree_building_=*/false}));
  }
}

// 匿名命名空间，用于定义局部静态变量和函数，限制其作用域在当前编译单元内使用

// 结构体 SubQueueThreadCache，用于缓存子队列的线程本地变量
struct SubQueueThreadCache {
  uint32_t key_;           // 缓存的键值
  ThreadLocalSubqueue* ref_;  // 指向线程本地子队列的指针
};

// 原子变量 queue_id_，用于存储队列 ID，初始值为 0
std::atomic<uint32_t> queue_id_{0};

// 线程局部变量 sub_queue_cache_，存储 SubQueueThreadCache 结构体的实例
thread_local SubQueueThreadCache sub_queue_cache_{0, nullptr};

// toString 函数，根据 EventType::PyCall 类型的 ExtraFields<EventType::PyCall> 对象 e 返回其字符串表示
std::string toString(const ExtraFields<EventType::PyCall>& e) {
  if (e.module_.has_value()) {
    // 如果 e 中包含模块值，则返回格式化字符串，显示模块名称和 ID
    return fmt::format(
        "nn.Module: {}_{}", e.module_->cls_name_.str(), e.module_->id_);
  }
  // 否则，返回调用站点的文件名、行号和函数名格式化字符串
  return fmt::format(
      "{}({}): {}",
      e.callsite_.filename_.str(),
      e.callsite_.line_no_,
      e.callsite_.funcname_.str());
}

// scopeToType 函数，根据 at::RecordScope 的值 scope 返回对应的 libkineto::ActivityType 枚举值
auto scopeToType(at::RecordScope scope) {
  return scope == at::RecordScope::USER_SCOPE
      ? libkineto::ActivityType::USER_ANNOTATION
      : libkineto::ActivityType::CPU_OP;
}

// torchOpEndNS 函数，根据 EventType::TorchOp 类型的 ExtraFields<EventType::TorchOp> 对象 e，
// 完成标志 finished，以及父 Result 对象的弱引用 parent，返回 Torch 操作的结束时间（单位：纳秒）
int64_t torchOpEndNS(
    const ExtraFields<EventType::TorchOp>& e,
    const bool finished,
    const std::weak_ptr<Result>& parent) {
  if (finished && e.end_time_ns_ == std::numeric_limits<c10::time_t>::min()) {
    auto p = parent.lock();
    if (p) {
      // 如果 finished 为 true，且 end_time_ns_ 为默认最小值，返回父 Result 对象的结束时间
      return p->endTimeNS();
    }
  }
  // 否则，返回 e 的结束时间 end_time_ns_
  return e.end_time_ns_;
}

// kinetoEventCorrelationID 函数，根据 EventType::Kineto 类型的 ExtraFields<EventType::Kineto> 对象 e，
// 以及父 Result 对象的弱引用 parent，返回 Kineto 事件的相关 ID
auto kinetoEventCorrelationID(
    const ExtraFields<EventType::Kineto>& e,
    const std::weak_ptr<Result>& parent) {
  if (e.correlation_id_) {
    // 如果 e 中包含 correlation_id_，则返回该值作为相关 ID
    return e.correlation_id_;
  }
  auto p = parent.lock();
  // 否则，返回父 Result 对象的相关 ID，如果不存在父对象，则返回 0
  return p ? p->correlationID() : 0;
}

// 定义宏 ATTRIBUTE，用于生成函数对象，参数为 event_type 和 expr
#define ATTRIBUTE(event_type, expr)                  \
  [&](const ExtraFields<EventType::event_type>& e) { \
    (void)e;                                         \
    // 注释结束，expr 为函数体的部分未注释代码
    return expr;                                     \
  }


注释：

// 返回表达式的结果并结束当前函数
return expr;
// 结束当前的代码块
}
// 返回 Result 对象的名称。根据不同的事件类型调用对应的访问者函数，获取名称信息。
std::string Result::name() const {
  return visit(c10::overloaded(
      // 如果事件类型是 Vulkan，则返回事件名称字符串
      ATTRIBUTE(Vulkan, std::string(e.name_)),
      // 如果事件类型是 Allocation，则返回固定的内存字符串
      ATTRIBUTE(Allocation, std::string("[memory]")),
      // 如果事件类型是 OutOfMemory，则返回固定的 OutOfMemory 字符串
      ATTRIBUTE(OutOfMemory, std::string("[OutOfMemory]")),
      // 如果事件类型是 PyCall，则将事件转换为字符串
      ATTRIBUTE(PyCall, toString(e)),
      // 如果事件类型是 PyCCall，则返回函数名字符串
      ATTRIBUTE(PyCCall, std::string(e.function_name_.str())),
      // 对于其他未明确指定的事件类型，返回事件名称字符串
      [](const auto& e) -> std::string { return e.name_; }));
}

// 返回 Result 对象的 Kineto 类型。根据不同的事件类型调用对应的访问者函数，获取 Kineto 类型信息。
libkineto::ActivityType Result::kinetoType() const {
  return visit(c10::overloaded(
      // 如果事件类型是 TorchOp，则将作用域转换为对应的 ActivityType
      ATTRIBUTE(TorchOp, scopeToType(e.scope_)),
      // 如果事件类型是 Backend，则将作用域转换为对应的 ActivityType
      ATTRIBUTE(Backend, scopeToType(e.scope_)),
      // 如果事件类型是 Vulkan，则返回 CPU_OP 类型
      ATTRIBUTE(Vulkan, libkineto::ActivityType::CPU_OP),
      // 如果事件类型是 Allocation 或 OutOfMemory，则返回 CPU_INSTANT_EVENT 类型
      ATTRIBUTE(Allocation, libkineto::ActivityType::CPU_INSTANT_EVENT),
      ATTRIBUTE(OutOfMemory, libkineto::ActivityType::CPU_INSTANT_EVENT),
      // 如果事件类型是 PyCall 或 PyCCall，则返回 PYTHON_FUNCTION 类型
      ATTRIBUTE(PyCall, libkineto::ActivityType::PYTHON_FUNCTION),
      ATTRIBUTE(PyCCall, libkineto::ActivityType::PYTHON_FUNCTION),
      // 如果事件类型是 Kineto，则返回具体的 ActivityType
      ATTRIBUTE(Kineto, e.activity_type_)));
}

// 返回 Result 对象的相关 ID。根据不同的事件类型调用对应的访问者函数，获取相关 ID 信息。
uint64_t Result::correlationID() const {
  return visit(c10::overloaded(
      // 如果事件类型是 TorchOp，则返回相关 ID
      ATTRIBUTE(TorchOp, e.correlation_id_),
      // 如果事件类型是 Kineto，则调用函数获取相关 ID
      ATTRIBUTE(Kineto, kinetoEventCorrelationID(e, parent_)),
      // 对于其他事件类型，返回默认值 0
      [&](const auto&) -> uint64_t { return 0; }));
}

// 返回 Result 对象的结束时间。根据不同的事件类型调用对应的访问者函数，获取结束时间信息。
int64_t Result::endTimeNS() const {
  auto end_time_ns = visit(c10::overloaded(
      // 如果事件类型是 TorchOp，则计算结束时间
      ATTRIBUTE(TorchOp, torchOpEndNS(e, finished_, parent_)),
      // 如果事件类型是 Backend，则将微秒级别的结束时间转换为纳秒
      ATTRIBUTE(Backend, e.end_time_us_ * 1000),
      // 如果事件类型是 Vulkan，则根据是否在构建树中决定结束时间
      ATTRIBUTE(
          Vulkan, start_time_ns_ + (e.in_tree_building_ ? 0 : e.duration_ns_)),
      // 如果事件类型是 Allocation 或 OutOfMemory，则返回起始时间
      ATTRIBUTE(Allocation, start_time_ns_),
      ATTRIBUTE(OutOfMemory, start_time_ns_),
      // 如果事件类型是 Kineto，则返回起始时间加上持续时间
      ATTRIBUTE(Kineto, start_time_ns_ + e.duration_ns_),
      // 对于其他事件类型，返回默认的结束时间
      [&](const auto& e) -> int64_t { return e.end_time_ns_; }));

  // 在极少数情况下，允许操作没有明确的结束时间，此时可能需要借用父事件的结束时间。
  // 这会导致 `endTimeNS` 直到构建树完成前可能无法得到有效值。
  auto end_time_is_valid =
      !finished_ || SOFT_ASSERT(end_time_ns >= start_time_ns_, name());
  return end_time_is_valid ? end_time_ns : start_time_ns_;
}

// 返回 Result 对象的结束线程 ID。根据不同的事件类型调用对应的访问者函数，获取结束线程 ID 信息。
uint64_t Result::endTID() const {
  return visit(c10::overloaded(
      // 如果事件类型是 TorchOp，则返回结束线程 ID
      ATTRIBUTE(TorchOp, e.end_tid_),
      // 对于其他事件类型，返回起始线程 ID
      [&](const auto&) -> uint64_t { return start_tid_; }));
}

// 返回 Result 对象的设备类型。根据不同的事件类型调用对应的访问者函数，获取设备类型信息。
c10::DeviceType Result::deviceType() const {
  using torch::autograd::profiler::deviceTypeFromActivity;
  return visit(c10::overloaded(
      // 如果事件类型是 Vulkan，则返回 Vulkan 设备类型
      ATTRIBUTE(Vulkan, c10::DeviceType::Vulkan),
      // 如果事件类型是 Allocation 或 OutOfMemory，则返回设备类型
      ATTRIBUTE(Allocation, e.device_type_),
      ATTRIBUTE(OutOfMemory, e.device_type_),
      // 如果事件类型是 Kineto，则根据 ActivityType 获取设备类型
      ATTRIBUTE(Kineto, deviceTypeFromActivity(e.activity_type_)),
      // 对于其他事件类型，默认返回 CPU 设备类型
      [&](const auto&) { return c10::DeviceType::CPU; }));
}
#undef ATTRIBUTE

// 构造函数：初始化 ThreadLocalSubqueue 对象，设置线程 ID 和分析器配置。
ThreadLocalSubqueue::ThreadLocalSubqueue(
    const uint64_t tid,
    ProfilerConfig config)
    : tid_{tid},
      config_{std::move(config)},
      kineto_info_{kineto::kineto_ids()} {


    // 初始化对象的成员变量，使用给定的参数进行初始化
    // tid_{tid}: 使用 tid 参数初始化对象的 tid_ 成员变量
    // config_{std::move(config)}: 使用 std::move 将 config 参数移动到 config_ 成员变量中
    // kineto_info_{kineto::kineto_ids()}: 使用 kineto::kineto_ids() 函数初始化 kineto_info_ 成员变量
    // {} 中是构造函数的初始化列表，用于初始化对象的成员变量
    // 注意：这是 C++ 中的构造函数初始化列表语法



  torch::profiler::impl::kineto::recordThreadInfo();


  // 调用 torch::profiler::impl::kineto::recordThreadInfo() 函数
  // 这个函数的作用是记录线程的信息，通常用于性能分析或调试



  if (!config_.experimental_config.performance_events.empty()) {


  // 检查 config_ 对象的 experimental_config 成员变量中的 performance_events 是否为空
  // 如果不为空，则进入下面的条件语句块



    perf_profiler_ =
        std::make_unique<torch::profiler::impl::linux_perf::PerfProfiler>();


    // 使用 std::make_unique 创建一个名为 perf_profiler_ 的独占指针
    // 它指向一个 torch::profiler::impl::linux_perf::PerfProfiler 类的新实例
    // 这行代码用于创建性能分析器对象



    perf_profiler_->Configure(config_.experimental_config.performance_events);


    // 调用 perf_profiler_ 指向对象的 Configure 方法，传入 performance_events 参数
    // 这行代码用于配置性能分析器对象，根据 config_ 中的 performance_events 设置性能分析器的参数



  }


  // 结束条件语句块，表示如果 performance_events 为空则不执行以上两行代码



}


// 构造函数的结束，表示对象初始化完毕，进入对象的正常使用状态
// 在 C++ 中，构造函数的右括号表示初始化列表和函数体的结束
}

`
}

// RecordQueue 类的构造函数，初始化成员变量和数据成员
RecordQueue::RecordQueue(
    ProfilerConfig config,                    // 传入的配置参数
    std::set<ActivityType> activities)        // 传入的活动类型集合
    : id_(++queue_id_),                       // 初始化队列 ID，使用静态成员变量自增
      config_{std::move(config)},             // 移动构造配置参数
      activities_{std::move(activities)} {    // 移动构造活动类型集合

  // 如果配置要求跟踪 Python 并且活动类型包含 CPU，创建 Python 跟踪器
  if (tracePython()) {
    python_tracer_ = python_tracer::PythonTracerBase::make(this);
  }
}

// 判断是否需要跟踪 Python 的辅助函数
bool RecordQueue::tracePython() const {
  return config_.with_stack && activities_.count(ActivityType::CPU);
}

// 获取子队列的方法
ThreadLocalSubqueue* RecordQueue::getSubqueue() {
  // 如果当前队列 ID 与缓存中的 ID 相同，直接返回缓存的子队列指针
  if (id_ == sub_queue_cache_.key_) {
    return sub_queue_cache_.ref_;
  }

  // 否则，获取当前线程 ID，并加锁保护访问子队列的操作
  const auto tid = at::RecordFunction::currentThreadId();
  std::lock_guard<std::mutex> guard(sub_queue_mutex_);

  // 在子队列映射中查找当前线程的子队列，如果不存在则创建一个新的子队列
  auto it = sub_queues_.find(tid);
  if (it == sub_queues_.end()) {
    it = sub_queues_
             .emplace(tid, std::make_unique<ThreadLocalSubqueue>(tid, config_))
             .first;
  }

  // 更新子队列缓存并返回对应的子队列指针
  sub_queue_cache_ = SubQueueThreadCache{id_, it->second.get()};
  return it->second.get();
}

// 停止函数，停止 Python 跟踪器（如果存在）
void RecordQueue::stop() {
  if (python_tracer_) {
    python_tracer_->stop();
  }
}

// 命名空间内部的函数，用于标记结果为完成状态
void mark_finished(std::shared_ptr<Result>& r) {
  TORCH_INTERNAL_ASSERT(!r->finished_, r->name());
  r->finished_ = true;
  TORCH_INTERNAL_ASSERT(r->endTimeNS() >= r->start_time_ns_, r->name());
}

// 内联函数，根据线程 ID 和序列号生成前向线程键值
static inline uint64_t getForwardThreadKey(uint64_t tid, uint64_t seqNr) {
  return (((tid) << 48) | ((seqNr) & (((uint64_t)1 << 48) - 1)));
}

#ifdef USE_KINETO
// 如果使用 Kineto，则生成前向后向链接的函数
void generateForwardBackwardLink(
    const Result& profiler_result,                      // Profiler 结果的引用
    uint64_t& fwd_bwd_link_id,                          // 前向后向链接 ID 的引用
    libkineto::GenericTraceActivity& activity,           // Kineto 的活动对象引用
    std::unordered_map<uint64_t, libkineto::GenericTraceActivity*>&
        tidSeq2activity) {                              // 线程 ID 序列号到活动对象的pe::TorchOp>>(profiler_result.extra_fields_);
  if (extra_fields.forward_tid_ > 0) {
    // 如果存在前向线程 ID，则生成前向线程键
    uint64_t key = getForwardThreadKey(
        extra_fields.forward_tid_, extra_fields.sequence_number_);
    auto iter = tidSeq2activity.find(key);
    // 如果在 tidSeq2activity 中找到与当前迭代器相匹配的项
    if (iter != tidSeq2activity.end()) {
      // 获取找到的活动对象的指针 fwd
      libkineto::GenericTraceActivity* fwd = iter->second;
      // 启动流的标志设置为 true
      fwd->flow.start = true;
      // 设置当前活动对象的流 ID，并将其设置为 fwd_bwd_link_id
      activity.flow.id = fwd->flow.id = fwd_bwd_link_id;
      // 设置当前活动对象的流类型，并将其设置为双向链接类型
      activity.flow.type = fwd->flow.type = libkineto::kLinkFwdBwd;
      // 增加 fwd_bwd_link_id，以便为下一个流分配唯一的 ID
      ++fwd_bwd_link_id;

      // 如果存在多个与此 sequence/tid 组合匹配的事件，应该从映射中删除该条目，
      // 避免插入多个 "end" 流事件。
      tidSeq2activity.erase(iter);
    }
  } else if (profiler_result.start_tid_ != 0) {
    // 当前操作是正向操作
    uint64_t key = getForwardThreadKey(
        profiler_result.start_tid_, extra_fields.sequence_number_);
    // 假设：在所有具有相同序列号的操作中，
    // 具有最大起始时间的操作很可能是启动反向操作。
    auto iter = tidSeq2activity.find(key);
    // 如果在映射中未找到与 key 匹配的项
    if (iter == tidSeq2activity.end()) {
      // 将当前活动对象指针存储到 tidSeq2activity 中
      tidSeq2activity[key] = &activity;
    } else {
      // 现在假定序列号仅在创建 "Node" 对象用于反向传播时递增，
      // 通过调用 "at::sequence_number::get_and_increment()"。在所有具有相同序列号的操作中，
      // 具有最大 startTime 的操作将启动反向操作。
      if (activity.startTime >= iter->second->startTime) {
        // 更新 tidSeq2activity 中的活动对象指针为当前活动对象
        tidSeq2activity[key] = &activity;
      }
    }
  }
// 如果未定义 USE_KINETO 宏，则执行以下代码块
#ifndef USE_KINETO
{
}
#else // 如果定义了 USE_KINETO 宏，则执行以下代码块
    // 确保 CPU 跟踪活动数与结果数相同
    TORCH_INTERNAL_ASSERT(cpu_trace->activities.size() == results.size());

    // 创建一个无序映射表，将线程 ID 和序列号组合成 uint64_t 作为键，映射到活动对象的指针
    std::unordered_map<uint64_t, libkineto::GenericTraceActivity*> tidSeq2activity;
    // 初始化前向后向链接 ID
    uint64_t fwd_bwd_link_id = 1;

    // 使用 result_activity_t 类型表示结果与活动的对
    using result_activity_t = std::pair<Result*, libkineto::GenericTraceActivity*>;
    // 存储 Torch 事件的容器
    std::vector<result_activity_t> torch_events;

    // 遍历 CPU 跟踪活动列表
    for (const auto idx : c10::irange(cpu_trace->activities.size())) {
      auto& profiler_result = results[idx];
      auto& activity = cpu_trace->activities[idx];

      // 如果结果具有 TorchOp 类型的额外字段，则访问它
      profiler_result->visit_if_base<ExtraFields<EventType::TorchOp>>(
          [&](const auto& e) {
            // 如果序列号大于等于 0，则添加到 Torch 事件列表中
            if (e.sequence_number_ >= 0) {
              torch_events.emplace_back(profiler_result.get(), activity.get());
            }
          });
    }

    // 按照事件的结束时间排序 Torch 事件列表，以便按时间顺序处理
    std::sort(
        torch_events.begin(),
        torch_events.end(),
        [](const result_activity_t& left, const result_activity_t& right) {
          auto left_end_time =
              std::get<ExtraFields<EventType::TorchOp>>(left.first->extra_fields_)
                  .end_time_ns_;
          auto right_end_time =
              std::get<ExtraFields<EventType::TorchOp>>(right.first->extra_fields_)
                  .end_time_ns_;
          return left_end_time < right_end_time;
        });

    // 遍历排序后的 Torch 事件列表，为每个事件生成前向后向链接
    for (auto& [profiler_result, activity] : torch_events) {
      generateForwardBackwardLink(
          *profiler_result, fwd_bwd_link_id, *activity, tidSeq2activity);
    }
}
#endif // 结束 USE_KINETO 块

// 定义静态常量字符串 indexKey 为 "Ev Idx"
static constexpr const char* indexKey = "Ev Idx";

// 将结果传递给 Kineto，生成事件跟踪数据
void passEventsToKineto(
    const std::vector<std::shared_ptr<Result>>& results,
    uint64_t start_time_ns,
    uint64_t end_time_ns,
    const ProfilerConfig& config) {
  using namespace torch::profiler::impl::kineto;
  // 创建 CPU 跟踪对象，指定开始时间和描述
  TraceWrapper cpu_trace(
      static_cast<int64_t>(start_time_ns), "PyTorch Profiler");

  // 遍历每个记录的事件结果
  for (const auto i : c10::irange(results.size())) {
    const auto& e = results[i];
    // 选择事件的结束时间，避免使用 int64 最小值作为 Kineto 的结束时间，因为会导致溢出
    int64_t act_end_time = std::max(e->endTimeNS(), e->start_time_ns_);
    // 将 CPU 活动添加到 CPU 跟踪对象中
    auto* activity = cpu_trace.addCPUActivity(
        e->name(),                       // 使用事件名称
        e->kinetoType(),                 // 获取 Kineto 类型
        e->kineto_info_,                 // Kineto 信息
        e->correlationID(),              // 关联 ID
        e->start_time_ns_,               // 开始时间（纳秒）
        act_end_time);                   // 结束时间

    // 断言确保 activity 不为空，或者 Kineto 不可用
    TORCH_INTERNAL_ASSERT(activity || !kKinetoAvailable);
    if (activity) {
      addMetadata(activity, indexKey, std::to_string(i));

      // 如果全局配置开启
      if (config.global()) {
        e->kineto_activity_ = activity;  // 将 activity 赋给事件的 kineto_activity_
      }
    }
  }

  // 如果启用了前向/后向链接
  if (get_fwd_bwd_enabled()) {
    generateForwardBackwardLinks(cpu_trace.get(), results);  // 生成前向/后向链接
  }

  // 将 CPU 跟踪对象中收集的事件转移
  cpu_trace.transferCpuTrace(static_cast<int64_t>(end_time_ns));
}

#ifdef USE_KINETO
// 使用Kineto的两种机制连接分析器和Kineto事件。
// 第一种是关联ID。分析器在操作开始时推送一个唯一整数，并在结束时弹出。
// Kineto然后将其收集的事件与该关联ID相关联，并将其收集的事件的关联活动设置为指向分析器操作。
//
// 但这并不足以描述Kineto操作之间的依赖关系。考虑调用`torch.add`的情况。
// 会收集三个事件：
//   `aten::add`          (由分析器收集的Torch操作)
//   `cudaLaunchKernel`   (由Kineto收集的CUDA运行时事件)
//   `at::vectorized_...` (由Kineto收集的GPU内核)
// 如果仅依赖于关联ID，我们会将两个Kineto事件都设置为`at::add`的子节点，而不是正确的
//   `at::add -> cudaLaunchKernel -> at::vectorized_...`
//
// Kineto通过第二个概念“流”展示这些信息。
// 在本例中，`cudaLaunchKernel`事件是流的起始，GPU内核具有相同的流ID但不是起始事件。
// 因此，在将Kineto事件合并到调用树时，我们首先添加所有作为流起始节点的事件。
// 然后合并其余事件，尝试将它们与流起始事件配对，必要时回退到关联ID。
// 对于没有关联事件的任何节点，使用常规树构建算法确定其调用者。
class TransferEvents {
  using itrace_t = libkineto::ITraceActivity;
  using activity_t = torch::profiler::impl::kineto::activity_t;

 public:
  // 构造函数，初始化结果向量和跟踪指针
  TransferEvents(
      std::vector<std::shared_ptr<Result>>& results,
      trace_ptr_t& trace)
      : results_{results} {
    // 获取跟踪中的活动列表指针
    auto* trace_activities_ptr = trace->get()->activities();
    // 内部断言确保活动列表指针非空
    TORCH_INTERNAL_ASSERT(trace_activities_ptr != nullptr);
    // 解引用并存储活动列表
    trace_activities_ = *trace_activities_ptr;
    // 重新关联事件
    reassociate();
    // 从跟踪中提取事件
    extractEventsFromTrace();
    // 设置父节点
    setParents();
  }

 private:
  // 从元数据JSON中提取索引
  static long long extractIndex(const std::string& metadata_json) {
    // 前缀是索引键的格式化字符串
    static const auto prefix = fmt::format("\"{}\": ", indexKey);
    // 查找前缀在元数据JSON中的位置
    auto pos = metadata_json.find(prefix);
    // 如果找不到前缀，则返回未匹配的索引
    return (pos == std::string::npos) ? unmatchedIndex : [&]() {
      // 查找逗号的位置，结束位置是元数据JSON的末尾或逗号处
      auto end = metadata_json.find(',', pos);
      end = (end == std::string::npos) ? metadata_json.size() : end;
      // 将字符串转换为长整数
      return std::stoll(metadata_json.substr(pos + prefix.size(), end));
    }();
  }

  // 查找与关键字匹配的结果指针
  std::shared_ptr<Result> lookup(const itrace_t* key) {
    // 如果关键字为空指针，则返回空指针
    if (key == nullptr) {
      return nullptr;
    }

    // 首先检查映射中是否存在
    auto it = kineto_events_.find(key);
    // 如果在映射中找到，则返回相应的结果指针
    if (it != kineto_events_.end()) {
      return it->second;
    }

    // 否则，回退到编码的元数据
    const auto index = extractIndex(key ? key->metadataJson() : "");
    // 如果找到匹配的索引，则返回对应结果向量中的结果
    if (index != unmatchedIndex) {
      auto out = results_.get().at(index);
      // 将关键字和结果存入映射
      kineto_events_[key] = out;
      return out;
    }
    // 最终放弃，返回空指针。
    return nullptr;
  }

  void reassociate() {
    // 将分析器事件与对应的Kineto事件重新关联。Kineto可能已经移动或复制了这些活动，因此我们需要恢复
    // `libkineto::ITraceActivity` 和 `Result` 之间的关系。
    for (const auto* activity : trace_activities_) {
      TORCH_INTERNAL_ASSERT(activity != nullptr);
      auto e = lookup(activity);
      if (e != nullptr) {
        TORCH_INTERNAL_ASSERT(e->kineto_activity_ == nullptr);
        e->kineto_activity_ = static_cast<const activity_t*>(activity);
      }
    }
    // 如果结果集的大小与Kineto事件的大小不一致，则发出警告。
    if (results_.get().size() != kineto_events_.size()) {
      TORCH_WARN(fmt::format(
          "Failed to recover relationship between all profiler and kineto events: "
          "{} vs. {}  reassociated.",
          results_.get().size(),
          kineto_events_.size()));
    }
  }

  std::shared_ptr<Result> resultFromActivity(const itrace_t* activity) {
    TORCH_INTERNAL_ASSERT(activity != nullptr);

    // Kineto的类型不一致，因此我们必须将其转换为int32。
    torch::profiler::impl::kineto::DeviceAndResource device_and_resource{
        static_cast<int32_t>(activity->deviceId()),
        static_cast<int32_t>(activity->resourceId())};

    // 创建一个Result对象，根据给定的活动。
    auto event = Result::create(
        activity->timestamp(),
        noTID, // 占位符
        device_and_resource,
        ExtraFields<EventType::Kineto>{
            activity->name(),
            activity->duration(),
            static_cast<uint64_t>(activity->correlationId()),
            activity->type(),
            {/*id=*/static_cast<uint32_t>(activity->flowId()),
             /*type=*/static_cast<uint32_t>(activity->flowType()),
             /*start=*/activity->flowStart()}});

    // 注意：设置`event->kineto_activity_`是很诱人的，但我们只能保证我们传递给Kineto的事件是`GenericTraceActivity`类型的。
    // 其他可能是从ITraceActivity派生的类型，因此无法安全地转换。
    return event;
  }

  std::shared_ptr<Result> toResult(const itrace_t* activity) {
    auto e = lookup(activity);

    // 在非常确定可以重新关联Kineto和分析器事件之前，我们需要非常谨慎。
    const auto type = activity->type();
    if (e == nullptr &&
        (type == libkineto::ActivityType::CPU_OP ||
         type == libkineto::ActivityType::CPU_INSTANT_EVENT ||
         type == libkineto::ActivityType::USER_ANNOTATION ||
         type == libkineto::ActivityType::PYTHON_FUNCTION)) {
      TORCH_WARN_ONCE(
          "Detected an event which was likely passed to kineto by the PyTorch "
          "profiler, but is not present in the set of known events: ",
          activity->name(),
          " This most likely means that Kineto has not "
          "maintained address stability for this event. Please report this to "
          "the PyTorch team.");
      return nullptr;
    }
    // 如果当前事件指针为空指针，则根据活动创建结果，并将结果添加到结果列表和映射中
    if (e == nullptr) {
      e = resultFromActivity(activity);
      results_.get().push_back(e);
      kineto_events_[activity] = e;
    }
    // 返回事件指针
    return e;
  }

  void extractEventsFromTrace() {
    // 遍历跟踪活动列表中的每个活动
    for (const auto* activity : trace_activities_) {
      // 将活动转换为结果对象
      auto e = toResult(activity);
      // 获取与当前活动关联的链接活动指针
      const auto* linked_activity = activity->linkedActivity();
      // 如果当前事件和链接活动都存在
      if (e && linked_activity) {
        // 访问事件，根据事件类型执行不同的操作
        e->visit(c10::overloaded(
            // 如果是 Kineto 类型的附加字段，设置其链接活动结果
            [&](ExtraFields<EventType::Kineto>& i) {
              i.linked_activity_ = toResult(linked_activity);
            },
            // 否则，断言错误，这种情况不应该发生
            [](auto&) { TORCH_INTERNAL_ASSERT(false); }));
      }
    }
  }

  void setKinetoTID(
      std::shared_ptr<Result>& r,
      std::shared_ptr<Result> parent) {
    // 访问结果对象，根据其类型执行不同的操作
    r->visit(c10::overloaded(
        // 如果是 Kineto 类型的附加字段
        [&](ExtraFields<EventType::Kineto>& i) {
          // 断言起始线程 ID 尚未设置
          TORCH_INTERNAL_ASSERT(r->start_tid_ == noTID);
          // 如果父结果存在，设置起始线程 ID 为父结果的起始线程 ID；否则设置为当前线程 ID
          r->start_tid_ = parent ? parent->start_tid_
                                 : at::RecordFunction::currentThreadId();
        },
        // 如果不是 Kineto 类型的附加字段，什么都不做
        [](auto&) {}));

    // 递归处理结果对象的子对象
    for (auto& child : r->children_) {
      setKinetoTID(child, r);
    }
  }

  void setParents() {
    // 第一遍扫描：收集起始事件，并将父事件设置为链接事件
    ska::flat_hash_map<uint32_t, std::shared_ptr<Result>> flow_map;
    // 遍历结果列表中的每个结果对象
    for (auto& e : results_.get()) {
      // 断言结果对象不为空
      TORCH_INTERNAL_ASSERT(e != nullptr);
      // 访问结果对象，根据其类型执行不同的操作
      e->visit(c10::overloaded(
          // 如果是 Kineto 类型的附加字段
          [&](const ExtraFields<EventType::Kineto>& i) {
            // 如果流的类型是异步 CPU-GPU 链接且为起始事件
            if (i.flow.type == libkineto::kLinkAsyncCpuGpu && i.flow.start) {
              // 将当前结果对象插入流映射中
              auto inserted = flow_map.insert({i.flow.id, e});
#ifdef USE_ROCM
              // 如果使用 ROCm，检查是否插入了重复的流开始事件
              if (inserted.second) {
                // 如果插入了重复的流开始事件，则发出警告
                TORCH_WARN_ONCE(
                    "ROCTracer produced duplicate flow start: ", i.flow.id);
              }
#else // USE_ROCM
              // 如果不使用 ROCm，内部断言插入操作成功
              TORCH_INTERNAL_ASSERT(inserted.second);
#endif // USE_ROCM
            }
            // 内部断言当前事件的父事件已经过期
            TORCH_INTERNAL_ASSERT(e->parent_.expired());
            // 将当前事件的父事件设置为指定的关联活动
            e->parent_ = i.linked_activity_;
          },
          [](const auto&) {}));
    }

    // 第二遍扫描
    for (auto& e : results_.get()) {
      e->visit(c10::overloaded(
          [&](const ExtraFields<EventType::Kineto>& i) {
            // 流事件优先于关联事件
            const auto it = flow_map.find(i.flow.id);
            if (it != flow_map.end() &&
                i.flow.type == libkineto::kLinkAsyncCpuGpu && !i.flow.start) {
              // 如果找到流事件且符合条件，则将当前事件的父事件设置为流事件的关联事件
              e->parent_ = it->second;
            }

            // 如果设置了父事件，进行一些记录工作
            auto parent = e->parent_.lock();
            if (parent) {
              // 将当前事件添加到父事件的子事件列表中
              parent->children_.push_back(e);
              // 标记当前事件为已完成状态
              mark_finished(e);
            }
          },
          [](const auto&) {}));
    }

    // 确定了事件的谱系后，设置线程标识
    for (auto& e : results_.get()) {
      if (e->parent_.expired()) {
        // 如果父事件已过期，设置事件的 Kineto 线程标识为 nullptr
        setKinetoTID(e, nullptr);
      }
    }
  }

  // 未匹配索引的常量定义
  static constexpr long long unmatchedIndex = -1;
  // 无效线程标识的常量定义
  static constexpr auto noTID = std::numeric_limits<uint64_t>::max();
  // 对结果列表的引用包装器
  std::reference_wrapper<std::vector<std::shared_ptr<Result>>> results_;
  // 追踪活动的指针列表
  std::vector<const itrace_t*> trace_activities_;
  // Kineto 事件映射
  ska::flat_hash_map<const itrace_t*, std::shared_ptr<Result>> kineto_events_;
};
#else
class TransferEvents {
 public:
  // 可变参数模板构造函数
  template <class... Args>
  TransferEvents(Args&&...) {}
};
#endif

// 添加 Kineto 事件到结果列表，并返回追踪指针
trace_ptr_t addKinetoEvents(
    std::vector<std::shared_ptr<Result>>& results,
    uint64_t start_time_ns,
    uint64_t end_time_ns,
    const ProfilerConfig& config) {
  using namespace torch::profiler::impl::kineto;
  // 将事件传递给 Kineto 进行处理
  passEventsToKineto(results, start_time_ns, end_time_ns, config);

  // 在按需模式下，Kineto 直接由其他机制控制
  if (config.global()) {
    // 如果全局配置为真，则返回空指针
    return nullptr;
  }

  // 停止追踪并创建活动追踪包装器
  auto trace = std::make_unique<ActivityTraceWrapper>(stopTrace());
  // 内部断言追踪不为空或 Kineto 不可用
  TORCH_INTERNAL_ASSERT(trace || !kKinetoAvailable);
  // 创建 TransferEvents 对象，传递结果列表和追踪指针
  TransferEvents transfer{results, trace};
  // 返回追踪指针
  return trace;
}

// 结果比较器，按结束时间降序排列
struct ResultGreater {
  bool operator()(const result_ptr_t& a, const result_ptr_t& b) const {
    return a->endTimeNS() > b->endTimeNS();
  }
};

// 设置在树中构建状态的函数
void set_in_tree_building(
    std::vector<result_ptr_t>& results,
    const bool value) {
  // 遍历结果指针列表
  for (result_ptr_t& r : results) {
    // 对结果进行访问，根据事件类型进行不同处理
    r->visit(c10::overloaded(
        [value](ExtraFields<EventType::Vulkan>& i) {
          // 设置 Vulkan 类型事件的在树中构建状态
          i.in_tree_building_ = value;
        },
        [&](auto&) {
          // 对于其他事件类型，不做任何处理
        }));
  }
}
// 构建事件树的函数，输入是按时间排序的事件列表
void build_tree(std::vector<std::shared_ptr<Result>>& sorted_events) {
  // 标记正在构建树结构中，传入已排序事件列表和标志 true
  set_in_tree_building(sorted_events, true);

  // 使用 TorchOp 类型的额外字段定义 ska::flat_hash_map 和优先队列
  using op_fields = ExtraFields<EventType::TorchOp>;
  ska::flat_hash_map<uint64_t, std::shared_ptr<Result>> stacks;
  std::priority_queue<result_ptr_t, std::vector<result_ptr_t>, ResultGreater> end_events_;

  // lambda 函数用于向树结构中推入事件
  auto push_event = [&stacks, &end_events_](std::shared_ptr<Result>& event) {
    // Kineto 使用关联 ID 和流来构建子树，因此某些 Kineto 事件在主树构建算法之前已标记为完成。
    // 可以忽略它们；这些子树的根事件不是 Kineto 操作，将会正常处理。
    if (std::holds_alternative<ExtraFields<EventType::Kineto>>(
            event->extra_fields_) &&
        event->finished_) {
      return;
    }

    // 断言事件的父事件不存在（即未被引用）
    TORCH_INTERNAL_ASSERT(event->parent_.expired());
    // 断言所有子事件已完成
    for (const auto& child : event->children_) {
      TORCH_INTERNAL_ASSERT(child->finished_);
    }
    // 断言当前事件未完成
    TORCH_INTERNAL_ASSERT(!event->finished_);

    // 查找起始线程 ID 对应的父事件
    auto parent_it = stacks.find(event->start_tid_);
    if (parent_it == stacks.end()) {
      // 如果未找到，尝试使用 forward_tid 查找父事件
      auto fwd_tid = event->visit(c10::overloaded(
          [](const op_fields& i) { return i.forward_tid_; },
          [](const auto&) -> uint64_t { return 0; }));
      if (fwd_tid) {
        parent_it = stacks.find(fwd_tid);
      }
    }

    // 如果找到父事件，则建立父子关系
    if (parent_it != stacks.end()) {
      event->parent_ = parent_it->second;
      parent_it->second->children_.push_back(event);
    }

    // 如果事件的结束时间晚于开始时间，则加入堆栈和优先队列
    if (event->endTimeNS() > event->start_time_ns_) {
      stacks[event->start_tid_] = event;
      end_events_.push(event);
    } else if (event->endTimeNS() == std::numeric_limits<c10::time_t>::min()) {
      // 如果结束时间为最小时间，表示缺少终止事件，不加入 end_events_
      stacks[event->start_tid_] = event;
    } else {
      // 否则标记事件为已完成
      mark_finished(event);
    }
  };

  // lambda 函数用于从树结构中弹出事件
  auto pop_event = [&stacks](std::shared_ptr<Result> event) {
    if (event->finished_) {
      // 如果事件已经被前一个 `pop_event` 调用标记为完成，则直接返回
      return;
    }

    // 获取起始线程 ID 对应的事件帧
    auto start_tid = event->start_tid_;
    auto frame = stacks.at(start_tid);

    // 标记事件及其所有父事件为完成状态
    while (frame.get() != event.get()) {
      TORCH_INTERNAL_ASSERT(frame != nullptr);
      mark_finished(frame);
      TORCH_INTERNAL_ASSERT(!frame->parent_.expired());
      frame = frame->parent_.lock();
    }

    // 标记当前事件为完成状态，并从堆栈中移除
    mark_finished(event);
    stacks.erase(start_tid);
    // 如果存在新的父事件，则将其重新加入堆栈
    auto new_frame = event->parent_.lock();
    if (new_frame != nullptr) {
      stacks[start_tid] = new_frame;
    }
  };

  // 回放堆栈中的事件循环
  for (auto& event : sorted_events) {
    while (!end_events_.empty() &&
           end_events_.top()->endTimeNS() < event->start_time_ns_) {
      pop_event(end_events_.top());
      end_events_.pop();
    }
    push_event(event);
  }

  // 清理剩余的退出事件
  while (!end_events_.empty()) {
    pop_event(end_events_.top());
    # 从列表 end_events_ 中移除最后一个元素
    end_events_.pop();
    # 调用函数 set_in_tree_building，传入参数 sorted_events 和 false
    set_in_tree_building(sorted_events, false);
}

/**
 * Adjust r's duration to be the max of its current duration and the sum of all
 * of its children's adjusted durations (keeping its start time the same)
 * (adjust all child durations recursively)
 */
int64_t adjust_durations_dfs(std::shared_ptr<Result>& r) {
  // 如果 r 不为空，执行以下操作
  if (SOFT_ASSERT(r != nullptr)) {
    // 计算 r 当前的持续时间
    int64_t original_duration = r->endTimeNS() - r->start_time_ns_;
    // 计算所有子节点调整后持续时间的总和
    int64_t children_total_duration = std::accumulate(
        r->children_.begin(),
        r->children_.end(),
        0,
        [](int64_t acc, std::shared_ptr<Result>& child) {
          return acc + adjust_durations_dfs(child);
        });

    // 如果子节点调整后的总持续时间大于当前持续时间，进行调整
    if (children_total_duration > original_duration) {
      // 根据事件类型访问 r，更新结束时间或持续时间
      r->visit(c10::overloaded(
          [&r, &children_total_duration](ExtraFields<EventType::TorchOp>& i) {
            i.end_time_ns_ = r->start_time_ns_ + children_total_duration;
          },
          [&children_total_duration](ExtraFields<EventType::Vulkan>& i) {
            i.duration_ns_ = children_total_duration;
          },
          [](ExtraFields<EventType::Allocation>& _) {
            // Pass- 分配事件无需处理子节点
          },
          [&](auto&) {
            SOFT_ASSERT(
                false,
                "unexpected event type in mobile profiler adjust_durations_dfs: ",
                r->name());
          }));
      // 返回子节点调整后的总持续时间
      return children_total_duration;
    } else {
      // 如果子节点调整后的总持续时间不大于当前持续时间，保持原持续时间不变
      return original_duration;
    }
  } else {
    // 如果 r 为空，返回 0
    return 0;
  }
}

/**
 * 1) Adjust r's start time to be [new_start_time] (also adjusting end time and
      keeping duration the same)
 * 2) Recursively adjust r's children's start times, making them line up such
      that the last one ends at the same time as r
 * 3) Return r's final end time
 */
int64_t adjust_timestamps_dfs(
    std::shared_ptr<Result>& r,
    int64_t new_start_time) {
  // 如果 r 不为空，执行以下操作
  if (SOFT_ASSERT(r != nullptr)) {
    // 如果 r 的开始时间不等于新的开始时间，调整开始时间（保持持续时间不变）
    if (r->start_time_ns_ != new_start_time) {
      // 根据事件类型访问 r，更新结束时间或持续时间
      r->visit(c10::overloaded(
          [&r, &new_start_time](ExtraFields<EventType::TorchOp>& i) {
            i.end_time_ns_ =
                new_start_time + (i.end_time_ns_ - r->start_time_ns_);
          },
          [](ExtraFields<EventType::Vulkan>& i) {
            // Pass- Vulkan 事件无需手动调整结束时间
          },
          [](ExtraFields<EventType::Allocation>& _) {
            // Pass- 分配事件无需调整持续时间或结束时间
          },
          [&](auto&) {
            SOFT_ASSERT(
                false,
                "unexpected event type in mobile profiler adjust_timestamps_dfs: ",
                r->name());
          }));
      // 更新 r 的开始时间为新的开始时间
      r->start_time_ns_ = new_start_time;
    }
    // 计算所有子节点的总持续时间
    int64_t children_total_duration = std::accumulate(
        r->children_.begin(),
        r->children_.end(),
        0,
        [](int64_t acc, std::shared_ptr<Result>& child) {
          return acc + (child->endTimeNS() - child->start_time_ns_);
        });
    # 计算子任务的起始时间，通过父任务的结束时间减去所有子任务总持续时间得到
    int64_t child_start_time = r->endTimeNS() - children_total_duration;

    # 对于每一个子任务，递归调用 adjust_timestamps_dfs 函数来调整时间戳，并更新子任务的起始时间
    for (std::shared_ptr<Result>& child : r->children_) {
      child_start_time = adjust_timestamps_dfs(child, child_start_time);
    }

  }

  # 返回当前任务（r）的结束时间戳
  return r->endTimeNS();
/**
 * Adjust timestamps and durations of nodes in [out] such that
 *  - Vulkan event timelines are synchronized with CPU event times
 *  - Parent event timelines fully contain their child timelines
 *  - No overlaps in timelines for nodes at the same depth
 */
void adjust_timestamps(std::vector<std::shared_ptr<Result>>& out) {
  // 如果输出向量为空，则直接返回，无需调整时间戳
  if (out.empty()) {
    return;
  }

  // 初始化最小开始时间为第一个节点的开始时间
  int64_t min_start_time = out[0]->start_time_ns_;
  
  // 遍历输出向量中的所有节点
  for (std::shared_ptr<Result>& r : out) {
    // 只对根节点进行遍历
    if (r->parent_.expired()) {
      // 调用深度优先搜索函数，调整节点的持续时间
      adjust_durations_dfs(r);
      // 调用深度优先搜索函数，调整节点的时间戳，并更新最小开始时间
      min_start_time = adjust_timestamps_dfs(
          r,
          std::max(
              // 如果节点类型不是 Vulkan 事件，则使用节点的开始时间作为比较值
              r->tag() != EventType::Vulkan
                  ? r->start_time_ns_
                  // 如果是 Vulkan 事件，则使用 int64_t 类型的最小值作为比较值
                  : std::numeric_limits<int64_t>::min(),
              min_start_time));
    }
  }
}
    try {
      // 尝试从 python_tracer_ 获取事件
      ev = python_tracer_->getEvents(
          converter, python_enters, static_cast<c10::time_t>(end_time_ns));
    } catch (std::exception& e) {
      // 如果在获取事件过程中抛出异常，确保停止追踪，然后重新抛出异常
      torch::profiler::impl::kineto::stopTrace();
      throw;
    }
    // 将获取的事件加入到 out 中
    for (const auto& i : ev) {
      out.push_back(i);
    }
    // 重置 python_tracer_
    python_tracer_.reset();
  }

  // 如果启用了调整时间戳的实验性配置
  if (config_.experimental_config.adjust_timestamps) {
    // 根据事件的开始时间对 out 进行稳定排序
    std::stable_sort(out.begin(), out.end(), [](const auto& a, const auto& b) {
      return a->start_time_ns_ < b->start_time_ns_;
    });
    // 构建事件树
    build_tree(out);
    // 调整时间戳
    adjust_timestamps(out);
    // 清理每个事件的父指针和子节点列表，为第二次构建树做准备
    for (auto& r : out) {
      r->parent_.reset();
      r->finished_ = false;
      r->children_.clear();
    }
  }

  // 将 Kineto 事件添加到跟踪中，并返回跟踪结果
  auto trace = addKinetoEvents(out, start_time_ns, end_time_ns, config_);

  // 根据事件的开始时间再次对 out 进行稳定排序
  std::stable_sort(out.begin(), out.end(), [](const auto& a, const auto& b) {
    return a->start_time_ns_ < b->start_time_ns_;
  });

  // 如果配置要求报告输入形状并且启用了内存分析
  if (config_.report_input_shapes && config_.profile_memory) {
    // 计算唯一张量 ID
    calculateUniqueTensorIDs(out);
  }

  // 再次构建事件树
  build_tree(out);
  // 返回包含 out 和 trace 的结果
  return {out, std::move(trace)};
} // 结束命名空间

namespace {
// 返回一个引用静态函数对象，该对象始终返回 true
std::function<bool()>& record_concrete_inputs_enabled_fn() {
    // 定义静态函数对象 fn，并初始化为返回 true 的 lambda 函数
    static std::function<bool()> fn = []() { return true; };
    return fn; // 返回静态函数对象 fn 的引用
}
} // 匿名命名空间结束

// 返回 record_concrete_inputs_enabled_fn() 的执行结果
bool get_record_concrete_inputs_enabled() {
    return record_concrete_inputs_enabled_fn()();
}

// 将传入的函数对象 fn 移动赋值给 record_concrete_inputs_enabled_fn()
void set_record_concrete_inputs_enabled_fn(std::function<bool()> fn) {
    record_concrete_inputs_enabled_fn() = std::move(fn);
}

// 将 val 值捕获并作为 lambda 函数返回，然后移动赋值给 record_concrete_inputs_enabled_fn()
void set_record_concrete_inputs_enabled_val(bool val) {
    record_concrete_inputs_enabled_fn() = [val]() { return val; };
}

namespace {
// 返回一个引用静态函数对象，该对象始终返回 true
std::function<bool()>& fwd_bwd_enabled_fn() {
    // 定义静态函数对象 fn，并初始化为返回 true 的 lambda 函数
    static std::function<bool()> fn = []() { return true; };
    return fn; // 返回静态函数对象 fn 的引用
}
} // 匿名命名空间结束

// 返回 fwd_bwd_enabled_fn() 的执行结果
bool get_fwd_bwd_enabled() {
    return fwd_bwd_enabled_fn()();
}

// 将传入的函数对象 fn 移动赋值给 fwd_bwd_enabled_fn()
void set_fwd_bwd_enabled_fn(std::function<bool()> fn) {
    fwd_bwd_enabled_fn() = std::move(fn);
}

// 将 val 值捕获并作为 lambda 函数返回，然后移动赋值给 fwd_bwd_enabled_fn()
void set_fwd_bwd_enabled_val(bool val) {
    fwd_bwd_enabled_fn() = [val]() { return val; };
}

namespace {
// 返回一个引用静态函数对象，该对象始终返回 false
std::function<bool()>& cuda_sync_enabled_fn() {
    // 定义静态函数对象 fn，并初始化为返回 false 的 lambda 函数
    static std::function<bool()> fn = []() { return false; };
    return fn; // 返回静态函数对象 fn 的引用
}
} // 匿名命名空间结束

// 返回 cuda_sync_enabled_fn() 的执行结果
bool get_cuda_sync_enabled() {
    return cuda_sync_enabled_fn()();
}

// 将传入的函数对象 fn 移动赋值给 cuda_sync_enabled_fn()
void set_cuda_sync_enabled_fn(std::function<bool()> fn) {
    cuda_sync_enabled_fn() = std::move(fn);
}

// 将 val 值捕获并作为 lambda 函数返回，然后移动赋值给 cuda_sync_enabled_fn()
void set_cuda_sync_enabled_val(bool val) {
    cuda_sync_enabled_fn() = [val]() { return val; };
}

} // namespace torch::profiler::impl 结束
```