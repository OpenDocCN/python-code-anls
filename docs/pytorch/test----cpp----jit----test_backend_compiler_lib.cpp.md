# `.\pytorch\test\cpp\jit\test_backend_compiler_lib.cpp`

```py
// 包含ATen库中的实用工具、TensorImpl核心、ApproximateClock等头文件
#include <ATen/Utils.h>
#include <c10/core/TensorImpl.h>
#include <c10/util/ApproximateClock.h>
#include <torch/csrc/jit/backends/backend.h>
#include <torch/csrc/jit/backends/backend_exception.h>

// 如果未定义NO_PROFILING，则包含移动设备上的分析边缘头文件
#ifndef NO_PROFILING
#include <torch/csrc/jit/mobile/profiler_edge.h>
#endif

// 定义torch命名空间下的jit命名空间
namespace torch {
namespace jit {

// PyTorch后端的实现，用于处理、编译和执行由'add'和'sub'操作符组成的TorchScript模块。
// 该后端仅支持实现两个输入的加法或减法（即in1 + in2或in1 - in2）的模块。
// 因此，模型的方法期望恰好是两个Tensor类型的输入。
// 此后端用于演示编译和执行的流程，但不打算作为可用于实际推理的实用后端。

// 实现细节：
//
// 编译
// 1. 添加了一个具有最小编译功能的后端，“backend_with_compiler_demo”。
// 2. 编译发生在注册到此后端的预处理函数中。
// 3. 编译结果以字符串blob的形式存储在每个方法中，它们通过__getstate__函数序列化到降阶模块中。
// 4. 对于后端编译器无法处理的功能，抛出包含模型源代码的错误消息。

// 运行时
// 1. 编译的blob在__setstate__方法中加载。
// 2. 后端的编译函数：将预处理的blob解析为后端能理解的格式（一系列令牌）。
// 3. 后端的执行函数执行指定的方法（handle）。

namespace {
// 解析方法句柄的函数，输入为blob字符串，输出为包含指令和调试句柄的元组向量
std::vector<std::tuple<std::string, int64_t>> parseMethodHandle(
    const std::string& blob) {
  std::vector<std::tuple<std::string, int64_t>> result;
  std::stringstream s_stream(blob);
  constexpr char debug_handle_token[] = "<debug_handle>";
  // 逐行解析blob中的内容
  while (s_stream.good()) {
    std::string substr;
    getline(s_stream, substr, ',');
    auto debug_handle_pos = substr.find(debug_handle_token);
    int64_t debug_handle{-1};
    auto instruction = substr.substr(0);
    // 如果找到调试句柄标记，则解析调试句柄和指令
    if (debug_handle_pos != std::string::npos) {
      instruction = substr.substr(0, debug_handle_pos);
      debug_handle = stoi(substr.substr(debug_handle_pos + 14));
    }
    // 将解析的指令和调试句柄添加到结果向量中
    result.push_back(std::make_tuple(instruction, debug_handle));
  }
  return result;
}

// 返回Tensor的浮点数据指针的函数
float* float_data_ptr(const at::Tensor& t) {
  return t.data_ptr<float>();
}
} // namespace

// 继承自PyTorchBackendInterface的BackendWithCompiler类
class BackendWithCompiler : public PyTorchBackendInterface {
 public:
  // 构造函数
  // NOLINTNEXTLINE(modernize-use-equals-default)
  explicit BackendWithCompiler() {}
  // 虚析构函数，声明为默认
  virtual ~BackendWithCompiler() override = default;

  // 判断后端是否可用的函数
  bool is_available() override {
  // 返回 true，表示编译成功
  return true;
}

// 由于实际的编译在这个后端是AOT完成的，所以编译函数简单地将所有内容转发。在非玩具设置中，这可能会从运行时获取一些可能对执行有用的信息，例如构建标志、设备摄像头的分辨率，或者基本上任何在预处理被调用的服务器端无法获取的运行时特定信息。

c10::impl::GenericDict compile(
    c10::IValue processed,
    c10::impl::GenericDict method_compile_spec) override {
  // 将输入的 processed 转换为通用字典
  auto dict = processed.toGenericDict();
  // 初始化存储方法句柄的数据结构
  auto handles =
      c10::Dict<std::string, std::vector<std::tuple<std::string, int64_t>>>();
  // 遍历字典中的每一对键值对
  for (const auto& kv : dict) {
    // 解析方法句柄的 tokens，并插入到 handles 中
    auto tokens = parseMethodHandle(kv.value().toStringRef());
    handles.insert(kv.key().toStringRef(), tokens);
  }
  // 将 handles 转换为通用字典并返回
  return c10::impl::toGenericDict(handles);
}

// 实际在后端执行模型的函数。这里没有需要分发的内容，因此后端在 execute 函数内实现，并且仅支持 add、subtract 和 constant。在非玩具后端中，可以想象这个函数如何用于将输入实际分发到相关的后端/设备上。

c10::impl::GenericList execute(
    c10::IValue
        handle, // 例如：[('prim::Constant#1', 14), ('aten::add', 15)]
    c10::impl::GenericList inputs) override {
  // 断言输入列表的大小为 2
  TORCH_INTERNAL_ASSERT(inputs.size() == 2);
  // 获取第一个输入的值并转换为 Tensor
  c10::IValue val0 = inputs[0];
  at::Tensor x = val0.toTensor();
  // 获取第二个输入的值并转换为 Tensor
  c10::IValue val1 = inputs[1];
  at::Tensor h = val1.toTensor();
  // 初始化用于存储操作运行时间的 vector
  std::vector<std::tuple<int64_t, int64_t, std::string>> op_runtimes_us;
  op_runtimes_us.reserve(handle.toList().size());

  // 初始化用于存储输出 Tensor 的列表
  c10::List<at::Tensor> output_list;
#ifndef NO_PROFILING
    // 记录开始时间（微秒级别），用于性能分析
    auto start_us = c10::getTime() / 1000;
#endif
    // 遍历处理器句柄的每个令牌
    for (const auto& token : handle.toList()) {
      // 将令牌转换为IValue类型
      IValue val = token;
      // 获取指令字符串
      auto instruction = val.toTupleRef().elements()[0].toStringRef();
      // 获取调试句柄
      auto debug_handle = val.toTupleRef().elements()[1].toInt();
#ifndef NO_PROFILING
      // 记录每个指令的开始时间（微秒级别），用于性能分析
      auto start_time_us = c10::getTime() / 1000;
#endif
      try {
        // 检查指令是否以"prim::Constant"开头
        if (instruction.rfind("prim::Constant", 0) == 0) {
          // 15是字符串"prim::Constant#"的长度，获取常量值的子字符串
          TORCH_CHECK(
              instruction.size() > 15,
              "Constant value is expected in ",
              instruction);
          // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
          auto sub = instruction.substr(15);
        } else if (instruction == "aten::add" || instruction == "aten::sub") {
          // 检查张量x和h的尺寸是否相同
          TORCH_CHECK(x.sizes() == h.sizes());
          // 检查张量x的维度
          if (x.dim() > 1 || (x.dim() == 1 && x.size(0) > 1)) {
            // 发出警告，仅添加或减去张量的第一个元素
            TORCH_WARN(
                "Only the first elements of the tensors are added or subbed.");
          }
          // 检查张量x和h的数据类型是否为Float
          TORCH_CHECK(
              (x.scalar_type() == c10::ScalarType::Float &&
               h.scalar_type() == c10::ScalarType::Float),
              "Only float tensors are compatible for add and sub.");
          // 创建一个与张量x大小相同的空张量y
          at::Tensor y = at::detail::empty_cpu(x.sizes(), at::kFloat);
          // 获取张量x、h和y的数据指针
          auto x_ptr = float_data_ptr(x);
          auto h_ptr = float_data_ptr(h);
          auto y_ptr = float_data_ptr(y);
#ifndef NO_PROFILING
          // 记录内存事件到性能分析器（Edge Profiler）
          RECORD_BACKEND_MEMORY_EVENT_TO_EDGE_PROFILER(
              x_ptr,
              x.numel() * sizeof(float),
              x.numel() * sizeof(float),
              x.numel() * sizeof(float) + y.numel() * sizeof(float) +
                  h.numel() * sizeof(float),
              c10::Device(c10::kCPU));
#endif
          // 根据指令执行加法或减法操作
          if (instruction == "aten::add") {
            y_ptr[0] = x_ptr[0] + h_ptr[0];
          } else {
            y_ptr[0] = x_ptr[0] - h_ptr[0];
          }
          // 将计算结果张量y添加到输出列表中
          output_list.emplace_back(y);
        } else {
          // 若指令不支持，则抛出异常
          TORCH_CHECK(
              false,
              "Instruction, ",
              instruction,
              " is not supported. ",
              "Contact the backend POC for details. ");
        }
      } catch (c10::Error& e) {
        // 捕获C10库的错误并抛出自定义后端异常
        TORCH_DELEGATED_BACKEND_THROW(false, e.what(), debug_handle);
      }
#ifndef NO_PROFILING
      // 计算操作执行时间（微秒级别）并记录到操作运行时间列表
      auto end_time_us = c10::getTime() / 1000;
      auto duration = end_time_us - start_time_us;
      op_runtimes_us.emplace_back(duration, debug_handle, instruction);
#endif
    }
#ifndef NO_PROFILING
    // 遍历操作运行时间列表，记录每个操作的事件到性能分析器（Edge Profiler）
    for (const auto& tup : op_runtimes_us) {
      RECORD_BACKEND_EVENT_TO_EDGE_PROFILER(
          start_us,
          start_us + std::get<0>(tup),
          std::get<1>(tup),
          std::get<2>(tup),
          "test_backend");
      start_us = start_us + std::get<0>(tup);
    }
#endif
    // 将输出列表转换为IValue并返回
    return c10::impl::toList(output_list);
  }
};

namespace {
// 后端名称常量
constexpr auto backend_name = "backend_with_compiler_demo";
// 定义静态变量 cls，并使用 torch::jit::backend<BackendWithCompiler> 来初始化它
static auto cls = torch::jit::backend<BackendWithCompiler>(backend_name);
// 结束命名空间 jit
} // namespace jit
// 结束命名空间 torch
} // namespace torch
```