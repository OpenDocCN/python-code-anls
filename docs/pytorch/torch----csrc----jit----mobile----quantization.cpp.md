# `.\pytorch\torch\csrc\jit\mobile\quantization.cpp`

```
namespace torch {
namespace jit {
namespace mobile {
namespace quantization {

// PTQQuanizationHelper 类的 quantize_dynamic 方法，用于动态量化模块
void PTQQuanizationHelper::quantize_dynamic(
    torch::jit::mobile::Module& m,
    const std::string& method_name) {

  // 设置在预打包时不释放权重
  at::globalContext().setReleaseWeightsWhenPrepacking(false);

  // 构建方法名字符串
  std::string reset_observers_method_name = "reset_observers_" + method_name;
  std::string observe_method_name = "observe_" + method_name;
  std::string quantize_method_name = "quantize_" + method_name;
  std::string quantized_method_name = "quantized_" + method_name;

  // 检查模块是否包含必需的方法
  TORCH_CHECK(
      m.find_method(reset_observers_method_name).has_value(),
      "PTQ ready module must have",
      reset_observers_method_name,
      " method.");
  TORCH_CHECK(
      m.find_method(observe_method_name),
      "PTQ ready module must have",
      observe_method_name,
      " method.");
  TORCH_CHECK(
      m.find_method(quantize_method_name),
      "PTQ ready module must have",
      quantize_method_name,
      " method.");
  TORCH_CHECK(
      m.find_method(quantized_method_name),
      "PTQ ready module must have",
      quantized_method_name,
      " method.");
  TORCH_CHECK(
      m.find_method("get_all_bundled_inputs"),
      "PTQ ready module must have get_all_bundled_inputs method.");

  // 获取模块的输入，运行必要的方法进行动态量化
  auto inputs = m.run_method("get_all_bundled_inputs")
                    .toList()
                    .get(0)
                    .toTupleRef()
                    .elements()
                    .vec();
  m.get_method(reset_observers_method_name)({});
  m.get_method(observe_method_name)(inputs);
  m.get_method(quantize_method_name)(inputs);

  // 比较方法的模式并复制必要的方法
  m.compareMethodSchemas(method_name, quantized_method_name);
  m.unsafeRemoveMethod(method_name);
  const Function& to_be_copied =
      m.find_method(quantized_method_name).value().function();
  m.unsafeCopyMethod(method_name, to_be_copied);
  m.unsafeRemoveMethod(quantized_method_name);
  m.unsafeRemoveMethod(quantize_method_name);
  m.unsafeRemoveMethod(observe_method_name);
  m.unsafeRemoveMethod(reset_observers_method_name);
}

} // namespace quantization
} // namespace mobile
} // namespace jit
} // namespace torch
```