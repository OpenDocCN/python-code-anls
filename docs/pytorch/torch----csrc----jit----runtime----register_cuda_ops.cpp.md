# `.\pytorch\torch\csrc\jit\runtime\register_cuda_ops.cpp`

```
// 引入头文件，包含实现 PyTorch CUDA API 在 TorchScript 中使用的特殊 JIT 运算符的声明
#include <torch/csrc/api/include/torch/utils.h>
#include <torch/csrc/jit/cuda/cuda.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/runtime/operator.h>

// 定义命名空间 torch::jit
namespace torch::jit {

// 匿名命名空间，用于隐藏内部实现细节
namespace {

// 定义一个函数，返回从模式中获取的别名分析类型
c10::AliasAnalysisKind aliasAnalysisFromSchema() {
  return c10::AliasAnalysisKind::FROM_SCHEMA;
}

// 定义一个函数，同步指定设备的辅助 API
void _device_synchronize(int64_t device_index) {
  // 获取当前设备的索引
  auto current_device_index = c10::cuda::current_device();
  // 如果当前设备索引与要同步的设备索引不同，则设置设备为要同步的设备索引
  if (current_device_index != device_index) {
    c10::cuda::set_device(device_index);
  }
  // 执行 CUDA 设备同步操作
  c10::cuda::device_synchronize();

  // 在同步之前恢复到当前设备
  if (current_device_index != device_index) {
    c10::cuda::set_device(current_device_index);
  }
}

// 注册自定义运算符
RegisterOperators const reg({
    // 注册 cuda::current_stream.device 运算符
    Operator(
        "cuda::current_stream.device(Device? device) -> __torch__.torch.classes.cuda.Stream",
        [](Stack& stack) {
          // 从堆栈中获取可选的设备对象
          auto device = pop(stack).toOptional<c10::Device>();
          // 如果设备对象存在，则获取其索引；否则使用当前设备索引
          c10::DeviceIndex device_index = device.has_value()
              ? device->index()
              : c10::cuda::current_device();
          // 获取当前设备的 CUDA 流对象
          auto s = c10::cuda::getCurrentCUDAStream(device_index);
          // 创建自定义的 CUDAStream 类对象
          auto st = make_custom_class<torch::jit::CUDAStream>(s);
          // 将对象推送回堆栈
          push(stack, IValue(st));
        },
        aliasAnalysisFromSchema()),

    // 注册 cuda::current_stream.int 运算符
    Operator(
        "cuda::current_stream.int(int? val) -> __torch__.torch.classes.cuda.Stream",
        [](Stack& stack) {
          // 从堆栈中获取可选的设备索引
          auto idx = pop(stack).toOptional<c10::DeviceIndex>();
          // 如果索引存在，则使用它；否则使用当前设备索引
          c10::DeviceIndex device_index =
              idx.has_value() ? idx.value() : c10::cuda::current_device();
          // 获取当前设备的 CUDA 流对象
          auto s = c10::cuda::getCurrentCUDAStream(device_index);
          // 创建自定义的 CUDAStream 类对象
          auto st = make_custom_class<torch::jit::CUDAStream>(s);
          // 将对象推送回堆栈
          push(stack, IValue(st));
        },
        aliasAnalysisFromSchema()),

    // 注册 cuda::default_stream.device 运算符
    Operator(
        "cuda::default_stream.device(Device? device) -> __torch__.torch.classes.cuda.Stream",
        [](Stack& stack) {
          // 从堆栈中获取可选的设备对象
          auto device = pop(stack).toOptional<c10::Device>();
          // 如果设备对象存在，则获取其索引；否则使用当前设备索引
          c10::DeviceIndex device_index = device.has_value()
              ? device->index()
              : c10::cuda::current_device();
          // 获取默认的 CUDA 流对象
          auto s = c10::cuda::getDefaultCUDAStream(device_index);
          // 创建自定义的 CUDAStream 类对象
          auto st = make_custom_class<torch::jit::CUDAStream>(s);
          // 将对象推送回堆栈
          push(stack, IValue(st));
        },
        aliasAnalysisFromSchema()),
    Operator(
        "cuda::default_stream.int(int? val) -> __torch__.torch.classes.cuda.Stream",
        [](Stack& stack) {
          auto idx = pop(stack).toOptional<c10::DeviceIndex>();  // 从栈中弹出一个可能为空的设备索引
          c10::DeviceIndex device_index =
              idx.has_value() ? idx.value() : c10::cuda::current_device();  // 如果索引有值则使用其值，否则使用当前 CUDA 设备索引
          auto s = c10::cuda::getDefaultCUDAStream(device_index);  // 获取指定设备上的默认 CUDA 流
          auto st = make_custom_class<torch::jit::CUDAStream>(s);  // 创建一个自定义的 CUDA 流类
          push(stack, IValue(st));  // 将创建的流类推送回栈中
        },
        aliasAnalysisFromSchema()),  // 根据模式进行别名分析

    Operator(
        "cuda::_current_device() -> int",
        [](Stack& stack) {
          auto v = c10::cuda::current_device();  // 获取当前 CUDA 设备索引
          push(stack, static_cast<int>(v));  // 将设备索引推送回栈中
        },
        aliasAnalysisFromSchema()),  // 根据模式进行别名分析

    Operator(
        "cuda::_exchange_device(int64_t index) -> int",
        [](Stack& stack) {
          int64_t idx = -1;  // 初始化索引为负一
          pop(stack, idx);  // 从栈中弹出索引值
          if (idx < 0) {
            push(stack, -1);  // 如果索引小于零，则推送负一回栈中并返回
            return;
          }
          auto prev_idx = c10::cuda::current_device();  // 获取当前 CUDA 设备索引
          c10::cuda::set_device(static_cast<c10::DeviceIndex>(idx));  // 设置当前 CUDA 设备索引为给定索引
          push(stack, static_cast<int>(prev_idx));  // 将先前的设备索引推送回栈中
        },
        // cuda::set_device has side effects.
        c10::AliasAnalysisKind::CONSERVATIVE),  // 使用保守的别名分析

    Operator(
        "cuda::_maybe_exchange_device(int64_t index) -> int",
        [](Stack& stack) {
          int64_t idx = -1;  // 初始化索引为负一
          pop(stack, idx);  // 从栈中弹出索引值
          if (idx < 0) {
            push(stack, -1);  // 如果索引小于零，则推送负一回栈中并返回
            return;
          }
          int prev_idx = c10::cuda::MaybeExchangeDevice(static_cast<int>(idx));  // 尝试设置当前 CUDA 设备索引为给定索引，并返回先前的索引
          push(stack, prev_idx);  // 将先前的设备索引推送回栈中
        },
        c10::AliasAnalysisKind::CONSERVATIVE),  // 使用保守的别名分析

    Operator(
        "cuda::_set_device(int64_t val) -> ()",
        [](Stack& stack) {
          int64_t idx = -1;  // 初始化索引为负一
          pop(stack, idx);  // 从栈中弹出索引值
          c10::cuda::set_device(static_cast<c10::DeviceIndex>(idx));  // 设置当前 CUDA 设备索引为给定索引
        },
        aliasAnalysisFromSchema()),  // 根据模式进行别名分析

    Operator(
        "cuda::device_index(Device device) -> int",
        [](Stack& stack) {
          auto device = pop(stack);  // 从栈中弹出设备
          auto idx = device.toDevice().index();  // 获取设备的索引
          push(stack, idx);  // 将设备索引推送回栈中
        },
        aliasAnalysisFromSchema()),  // 根据模式进行别名分析

    Operator(
        "cuda::device_count() -> int",
        [](Stack& stack) { push(stack, at::cuda::device_count()); },  // 将 CUDA 设备的数量推送回栈中
        aliasAnalysisFromSchema()),  // 根据模式进行别名分析
    // 定义一个名为 "cuda::set_stream(__torch__.torch.classes.cuda.Stream stream) -> ()" 的运算符
    Operator(
        "cuda::set_stream(__torch__.torch.classes.cuda.Stream stream) -> ()",
        [](Stack& stack) {
          // 从堆栈中弹出一个值，这里应该是从 Torch 的堆栈中弹出一个自定义类对象
          auto v = pop(stack);
          // 将弹出的对象转换为 torch::jit::CUDAStream 类型
          auto s = v.toCustomClass<torch::jit::CUDAStream>();
          // 获取流的设备索引
          auto stream_device_idx = s->device_index();
          // 获取当前设备的索引
          auto cur_device_idx = c10::cuda::current_device();
          // 如果流不在当前设备上，则切换设备到流所在的设备
          if (cur_device_idx != stream_device_idx) {
            c10::cuda::set_device(stream_device_idx);
          }
          // 要使用 c10::cuda::setCurrentCUDAStream 设置当前 CUDA 流，
          // 需要将 jit::CUDAStream 对象转换为 c10::cuda::CUDAStream。
          // 由于后者无法从通过 TorchBind 注册的类中返回，只能通过打包
          // jit::CUDAStream 对象内包含的 c10::cuda::CUDAStream 实例到一个结构体表示，
          // 然后在此操作符内解包。解包后的流用于设置当前的 CUDA 流。
          auto unpacked = c10::cuda::CUDAStream::unpack3(
              s->id(), stream_device_idx, c10::DeviceType::CUDA);
          c10::cuda::setCurrentCUDAStream(unpacked);
        },
        aliasAnalysisFromSchema()),
    
    // 定义一个名为 "cuda::synchronize() -> ()" 的运算符
    Operator(
        "cuda::synchronize() -> ()",
        [](Stack& stack) { c10::cuda::device_synchronize(); },
        aliasAnalysisFromSchema()),
    
    // 定义一个名为 "cuda::synchronize.device(Device? device) -> ()" 的运算符
    Operator(
        "cuda::synchronize.device(Device? device) -> ()",
        [](Stack& stack) {
          // 从堆栈中弹出一个值，此处应该是一个可选的设备类型
          auto device = pop(stack).toOptional<c10::Device>();
          // 获取设备索引，如果没有提供设备，则使用当前 CUDA 设备的索引
          c10::DeviceIndex device_index = device.has_value()
              ? device->index()
              : c10::cuda::current_device();
          // 同步特定设备上的所有流
          _device_synchronize(device_index);
        },
        aliasAnalysisFromSchema()),
    
    // 定义一个名为 "cuda::synchronize.int(int? val) -> ()" 的运算符
    Operator(
        "cuda::synchronize.int(int? val) -> ()",
        [](Stack& stack) {
          // 从堆栈中弹出一个值，这里应该是一个可选的设备索引
          auto idx = pop(stack).toOptional<c10::DeviceIndex>();
          // 获取设备索引，如果没有提供设备索引，则使用当前 CUDA 设备的索引
          c10::DeviceIndex device_index =
              idx.has_value() ? idx.value() : c10::cuda::current_device();
          // 同步特定设备上的所有流
          _device_synchronize(device_index);
        },
        aliasAnalysisFromSchema()),
});
// 关闭 torch::jit 命名空间
} // namespace torch::jit
// 关闭外层命名空间
} // namespace
```