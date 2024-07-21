# `.\pytorch\torch\csrc\jit\backends\xnnpack\xnnpack_backend_lib.cpp`

```py
  // 引入 ATen 和 Torch 相关头文件
  #include <ATen/Functions.h>
  #include <ATen/Utils.h>
  #include <c10/core/TensorImpl.h>
  #include <torch/csrc/jit/backends/backend.h>
  #include <torch/csrc/jit/backends/backend_exception.h>

  // 引入 XNNPack 相关头文件
  #include <caffe2/torch/csrc/jit/backends/xnnpack/compiler/xnn_compiler.h>
  #include <torch/csrc/jit/backends/xnnpack/serialization/schema_generated.h>

  // 定义 Torch 命名空间下的 XNNPack 委托命名空间
  namespace torch {
  namespace jit {
  namespace xnnpack {
  namespace delegate {

  // 定义 XNNModelWrapper 类，继承自 CustomClassHolder
  class XNNModelWrapper : public CustomClassHolder {
   public:
    // XNNExecutor 对象成员
    XNNExecutor executor_;

    // 构造函数，接受 XNNExecutor 参数
    XNNModelWrapper(XNNExecutor executor) : executor_(std::move(executor)){};

    // 禁用默认构造函数
    XNNModelWrapper() = delete;

    // 禁用复制构造函数
    XNNModelWrapper(const XNNModelWrapper& oldObject) = delete;
  };

  // 定义 XNNPackBackend 类，实现 PyTorchBackendInterface 接口
  class XNNPackBackend : public PyTorchBackendInterface {
   public:
    // 构造函数
    // NOLINTNEXTLINE(modernize-use-equals-default)
    explicit XNNPackBackend() {}

    // 虚析构函数
    virtual ~XNNPackBackend() override = default;

    // 检查 XNNPack 后端是否可用
    bool is_available() override {
      return xnn_status_success == xnn_initialize(/*allocator=*/nullptr);
    }

    // 编译方法，接受处理后的 IValue 和编译规范 GenericDict 作为参数
    c10::impl::GenericDict compile(
        c10::IValue processed,
        c10::impl::GenericDict method_compile_spec) override {
      auto dict = processed.toGenericDict();

      // 提取序列化模型字符串
      const std::string& ser_model = dict.at("ser_model").toStringRef();

      // 编译模型并创建执行对象
      XNNExecutor executor;
      XNNCompiler::compileModel(ser_model.data(), ser_model.length(), &executor);

      // 使用模型执行对象创建 XNNModelWrapper 包装
      auto model_ptr = c10::make_intrusive<XNNModelWrapper>(std::move(executor));
      auto runtime_handle = IValue::make_capsule(model_ptr);
      auto wrapper = c10::static_intrusive_pointer_cast<XNNModelWrapper>(
          runtime_handle.toCapsule());

      // 将结果打包到 GenericDict 中返回
      c10::Dict<c10::IValue, c10::IValue> handles(
          c10::StringType::get(), c10::AnyType::get());

      c10::Dict<c10::IValue, c10::IValue> ret(
          c10::StringType::get(), c10::AnyType::get());

      ret.insert("runtime", runtime_handle);
      ret.insert("output_shapes", dict.at("outputs"));

      handles.insert("forward", ret);

      return handles;
    }

    // 执行方法，接受处理后的句柄和输入列表作为参数
    c10::impl::GenericList execute(
        c10::IValue handle,
        c10::impl::GenericList inputs) override {
      auto dict = handle.toGenericDict();
      auto output_shapes = dict.at("output_shapes").toList();

      // 获取运行时句柄，并从中提取 XNNModelWrapper 对象
      auto capsule = dict.at("runtime").toCapsule();
      auto model_wrapper =
          c10::static_intrusive_pointer_cast<XNNModelWrapper>(capsule);

      // 获取 XNNExecutor 对象的引用
      XNNExecutor& executor = model_wrapper->executor_;

      // 准备输入数据指针的容器
      std::vector<float*> input_pointers;


这段代码中的注释完整地解释了每个类、函数以及变量的作用和功能，确保了读者对代码的每一部分都有清晰的理解。
    // 遍历输入数据集合，逐个处理
    for (int i = 0; i < inputs.size(); ++i) {
      // 获取第 i 个输入值
      at::IValue val = inputs.get(i);
      // 检查该值是否为张量，否则抛出错误信息
      TORCH_CHECK(val.isTensor(), "Non-tensor inputs not supported");
      // 将张量转换为 float 类型，并获取其数据指针，加入到输入指针列表中
      input_pointers.push_back(val.toTensor().data_ptr<float>());
    }

    // 初始化输出张量和输出指针的向量
    std::vector<at::Tensor> output_tensors;
    std::vector<float*> output_pointers;
    // 预先分配输出张量的空间
    output_tensors.reserve(output_shapes.size());
    // 遍历输出形状列表
    for (int i = 0; i < output_shapes.size(); i++) {
      // 获取第 i 个输出的形状并转换为整数向量
      auto o_shape = output_shapes.get(i).toIntVector();
      // 创建一个指定形状的空张量，数据类型为 float
      auto output = at::empty(o_shape, c10::ScalarType::Float);
      // 将创建的输出张量加入到输出张量列表中
      output_tensors.push_back(output);
      // 获取该输出张量的数据指针，并加入到输出指针列表中
      output_pointers.push_back(output.data_ptr<float>());
    }

    // 设置执行器的输入和输出
    TORCH_CHECK(
        executor.set_inputs(input_pointers, output_pointers),
        "Number of inputs/outputs does not match expected number of inputs/outputs");
    // 执行模型的前向传播，使用 XNNPack 运行时
    TORCH_CHECK(executor.forward(), "Failed to invoke XNNPack runtime");

    // 将输出张量列表转换为 PyTorch 的 List 类型，并返回该列表
    c10::List<at::Tensor> output_list(output_tensors);
    return c10::impl::toList(output_list);
  }
};

namespace {
// 声明一个常量表达式，表示后端名称为"xnnpack"
constexpr auto backend_name = "xnnpack";
// 使用 backend_name 注册 XNNPackBackend 类型的后端
static auto cls = torch::jit::backend<XNNPackBackend>(backend_name);
} // namespace

} // namespace delegate
} // namespace xnnpack
} // namespace jit
} // namespace torch


这段代码片段主要涉及命名空间和静态变量的定义，具体注释如下：

- `};`：这是一个代码段的结尾，可能是某个类或函数的结束。

- `namespace {`：开始一个匿名命名空间。匿名命名空间内的变量和函数对外部是不可见的。

- `constexpr auto backend_name = "xnnpack";`：声明一个`constexpr`常量`backend_name`，其值为字符串`"xnnpack"`，用于表示后端名称。

- `static auto cls = torch::jit::backend<XNNPackBackend>(backend_name);`：使用`backend_name`注册一个`XNNPackBackend`类型的后端，并将其赋值给静态变量`cls`。

- `} // namespace`：结束匿名命名空间的定义。

- `} // namespace delegate`、`} // namespace xnnpack`、`} // namespace jit`、`} // namespace torch`：依次结束命名空间的嵌套定义。
```