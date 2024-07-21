# `.\pytorch\torch\csrc\jit\backends\nnapi\nnapi_backend_lib.cpp`

```py
// 包含标准库头文件
#include <memory>

// 包含 PyTorch NNAPI 绑定头文件
#include <ATen/nnapi/nnapi_bind.h>

// 包含 PyTorch 后端接口头文件
#include <torch/csrc/jit/backends/backend.h>

// 包含 PyTorch 后端异常处理头文件
#include <torch/csrc/jit/backends/backend_exception.h>

// 包含 PyTorch 移动端导入头文件
#include <torch/csrc/jit/mobile/import.h>

// 包含 PyTorch 移动端模块头文件
#include <torch/csrc/jit/mobile/module.h>

// 定义 torch 命名空间下的 jit 命名空间
namespace torch {
namespace jit {

// 实现 Android NNAPI 后端代理

// Android Neural Networks API (NNAPI) 是一个专为在 Android 设备上运行
// 机器学习计算密集型操作而设计的 Android C API。此 API 可用于运行 Android 8.1
// (API 级别 27) 或更高版本的所有 Android 设备。

// 此实现反映了 caffe2/torch/backends/_nnapi/prepare.py 中的 NnapiModule.forward() 方法
class NnapiBackend : public PyTorchBackendInterface {
 public:
  // 构造函数
  explicit NnapiBackend() = default;

  // 虚析构函数
  ~NnapiBackend() override = default;

  // 检查 NNAPI 后端是否可用
  bool is_available() override {
    return true;
  }

  // 编译处理后的输入和方法编译规范
  c10::impl::GenericDict compile(
      c10::IValue processed,
      c10::impl::GenericDict method_compile_spec) override {
    // 将 processed 封装在字典中: {"forward": processed}
    auto dict = processed.toGenericDict();
    c10::Dict<c10::IValue, c10::IValue> handles(
        c10::StringType::get(), c10::AnyType::get());
    handles.insert("forward", dict);
    return c10::impl::toGenericDict(handles);
  }

  // 执行处理后的句柄和输入列表
  c10::impl::GenericList execute(
      c10::IValue handle,
      c10::impl::GenericList inputs) override {
    // 将输入转换为张量
    c10::List<at::Tensor> tensorInp;
    for (c10::IValue element : inputs) {
      tensorInp.push_back(element.toTensor());
    }

    // 如果 comp_ 为空，则延迟调用 init()
    if (comp_ == nullptr) {
      init(handle, tensorInp);
    }
    TORCH_CHECK(comp_ != nullptr)

    // 创建输出张量列表
    c10::List<at::Tensor> outputs;
    for (at::Tensor out : out_templates_) {
      outputs.push_back(at::empty_like(out));
    }

    // 调整输入内存格式
    auto dict = handle.toGenericDict();
    auto inp_mem_fmts = dict.at("inp_mem_fmts").toIntList();
    TORCH_CHECK(tensorInp.size() == inp_mem_fmts.size());
    std::vector<at::Tensor> fixed_inputs;
    for (auto i = 0U; i < tensorInp.size(); i++) {
      int fmt = inp_mem_fmts[i];
      // 这些常量与 serializer.py 中的 DimOrder 中的值相匹配
      // 0: NCHW, 1: NHWC
      // TODO: 看看是否可以直接使用这些值
      if (fmt == 0) {
        fixed_inputs.push_back(tensorInp.get(i).contiguous());
      } else if (fmt == 1) {
        fixed_inputs.push_back(
            tensorInp.get(i).permute({0, 2, 3, 1}).contiguous());
      } else {
        TORCH_CHECK(false, "Invalid mem_fmt");
      }
    }

    // 运行组件的计算
    comp_->run(fixed_inputs, outputs.vec());

    // 调整输出内存格式
    auto out_mem_fmts = dict.at("out_mem_fmts").toIntList();
    TORCH_CHECK(outputs.size() == out_mem_fmts.size());
    for (auto i = 0U; i < outputs.size(); i++) {
      int fmt = out_mem_fmts[i];
      // These constants match the values in DimOrder in serializer.py
      // 0: NCHW, 1: NHWC
      // TODO: See if it's possible to use those directly.
      // 检查当前输出的内存格式是否为NHWC（1）
      if (fmt == 1) {
        // 如果是NHWC格式，重新排列第i个输出张量的维度顺序为NCHW
        outputs.set(i, outputs.get(i).permute({0, 3, 1, 2}));
      } else {
        // 如果不是NHWC格式，验证其是否为NCHW格式（0），否则抛出异常
        TORCH_CHECK(fmt == 0, "Invalid mem_fmt");
      }
    }

    // 将输出列表转换为C++标准库的列表类型并返回
    return c10::impl::toList(outputs);
  }

 private:
  // The following variables are modified by init() during execution,
  // and cannot be passed through the handles dictionary
  // 下列变量在执行init()期间被修改，不能通过handles字典传递
  std::unique_ptr<torch::nnapi::bind::NnapiCompilation> comp_;
  c10::List<at::Tensor> out_templates_;

  // Runs once per model initialization
  // Cannot be moved to compile(), because init() requires actual inputs
  // 模型初始化时运行一次，不能移到compile()中，因为init()需要实际输入
  void init(c10::IValue handle, c10::List<at::Tensor> inputs) {
    // 确保comp_为nullptr，即未初始化状态
    TORCH_CHECK(comp_ == nullptr);
    // 从handle中获取泛型字典
    auto dict = handle.toGenericDict();

    // Get ser_model
    // 获取序列化模型ser_model
    auto ser_model = dict.at("ser_model").toTensor();
    // Load shape computation module
    // 加载形状计算模块
    std::stringstream ss;
    // 获取形状计算模块的指针并转换为字符串流
    auto shape_ptr = dict.at("shape_compute_module").toString();
    ss.str(*shape_ptr);
    // 使用_load_for_mobile加载模块
    auto shape_compute_module = _load_for_mobile(ss);
    // 运行shape_compute_module的prepare方法，生成输出模板列表
    out_templates_ =
        shape_compute_module.run_method("prepare", ser_model, inputs)
            .toTensorList();

    // Create and initialize NnapiComilation object
    // 创建并初始化NnapiComilation对象
    comp_ = std::make_unique<torch::nnapi::bind::NnapiCompilation>();
    // 获取权重列表并将其传递给comp_的init方法进行初始化
    auto weights = dict.at("weights").toTensorVector();
    comp_->init(ser_model, weights);
  }
};

namespace {
constexpr auto backend_name = "nnapi";
static auto cls = torch::jit::backend<NnapiBackend>(backend_name);
} // namespace

} // namespace jit
} // namespace torch


注释：


// 结束了一个匿名命名空间的定义

namespace {
// 定义了一个 constexpr 字符串常量，表示后端名称为 "nnapi"
constexpr auto backend_name = "nnapi";
// 使用 nnapi 后端名称注册了一个名为 NnapiBackend 的 Torch JIT 后端
static auto cls = torch::jit::backend<NnapiBackend>(backend_name);
} // namespace

// 结束了命名空间 jit
} // namespace torch
```