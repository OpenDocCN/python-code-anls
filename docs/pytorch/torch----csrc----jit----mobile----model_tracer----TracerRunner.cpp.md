# `.\pytorch\torch\csrc\jit\mobile\model_tracer\TracerRunner.cpp`

```
/**
 * 包含必要的头文件以及命名空间声明
 */
#include <ATen/Functions.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/dispatch/ObservedOperators.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>
#include <torch/csrc/autograd/grad_mode.h>
#include <torch/csrc/jit/mobile/compatibility/runtime_compatibility.h>
#include <torch/csrc/jit/mobile/model_tracer/KernelDTypeTracer.h>
#include <torch/csrc/jit/mobile/model_tracer/MobileModelRunner.h>
#include <torch/csrc/jit/mobile/model_tracer/OperatorCallTracer.h>
#include <torch/csrc/jit/mobile/model_tracer/TensorUtils.h>
#include <torch/csrc/jit/mobile/model_tracer/TracerRunner.h>
#include <torch/csrc/jit/mobile/parse_operators.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <torch/script.h>

namespace torch {
namespace jit {
namespace mobile {

/**
 * 存储需要追踪的 GPU Metal 操作符的字符串列表
 */
const std::vector<std::string> gpu_metal_operators = {
    "aten::conv2d",
    "aten::add.Tensor",
    "aten::add_.Tensor",
    "aten::addmm",
    "aten::empty.memory_format",
    "aten::empty_strided",
    "aten::log_softmax.int",
    "aten::max_pool2d",
    "aten::mul.Tensor",
    "aten::relu",
    "aten::relu_",
    "aten::sigmoid",
    "aten::sub.Tensor",
    "aten::upsample_nearest2d.vec",
    "aten::view",
    "aten::adaptive_avg_pool2d",
    "aten::hardtanh_",
    "aten::reshape",
    "aten::flatten.using_ints",
};

/**
 * 这些是一些常见的 ATen 方法的集合，通常在模型的 forward() 运行之外调用，
 * 需要进行追踪以确保使用的操作符包含在构建中。
 * 如果/当此列表变得过长，可以考虑将其作为每个模型的列表。
 */
void call_setup_methods() {
  // 创建一个2x2的全零张量
  at::zeros({2, 2});
  // 创建一个2x2的全1张量
  at::ones({2, 2});
  // 创建一个空张量，并填充为3
  at::Tensor t1 = at::empty({7, 7});
  at::Tensor t2 = t1.fill_(3);
  // 根据指定步长创建一个新的空张量
  at::Tensor t3 = t1.new_empty_strided(
      {2, 3},
      {3,
       1}); // TODO investigate how this is different from normal empty_strided
  // 对张量进行窄化操作
  at::narrow(t2, 1, 0, 1);
  // 比较两个张量是否相等
  at::eq(t1, t2);
  // 检查全零张量是否非零
  const volatile bool nz = at::native::is_nonzero(at::zeros({1}));
  (void)nz;

  // 创建一个字节类型的张量并进行复制
  auto zb = at::zeros({10}, at::kByte);
  auto zf = at::zeros({10}, at::kFloat);
  zb.copy_(zf);
  // 张量t2除以1
  t2.div(1);

  // 典型情况下，失败会出现在CopyKernel.cpp中，因此列举可能出现的常见数据类型。
  const auto all_dtypes_for_copy = {
      at::kBool,
      at::kByte,
      at::kFloat,
      at::kInt,
      at::kChar,
      at::kDouble,
      at::kShort,
      at::kLong};
  // 遍历所有的数据类型并创建对应类型的空张量进行复制
  for (const auto dtype : all_dtypes_for_copy) {
    auto tensor1 = at::empty({10}, dtype);
    tensor1.copy_(at::zeros({10}, at::kBool));
    tensor1.copy_(at::zeros({10}, at::kFloat));
    // 使用张量 tensor1 的 copy_ 方法将一个大小为 10 的整数张量拷贝到 tensor1 中
    tensor1.copy_(at::zeros({10}, at::kInt));
  }

  // 创建一个大小为 0x0 的浮点数张量，但此处未分配给任何变量或使用
  torch::zeros({0, 0}, torch::ScalarType::Float);

  // 创建一个包含 20 个元素，每个元素初始化为 1.0 的浮点数向量 storage
  std::vector<float> storage(20, 1.0);

  // 创建一个大小为 {2, 10} 的整数向量 sizes
  std::vector<int64_t> sizes({2, 10});

  // 从存储器 storage 中创建一个张量，并指定其形状为 sizes，数据类型为浮点数
  torch::from_blob(storage.data(), at::IntArrayRef(sizes), at::kFloat);
/**
 * 类似于设置方法，这里有一套函数，通常在特定条件下出现，但由于输入束缚的狭窄性可能不会在跟踪中被调用
 */
void call_dependent_methods(std::set<std::string>& root_ops) {
  // 是否处于训练模式
  bool is_training = false;
  // 是否包含批归一化操作
  bool has_batchnorm = false;
  // 是否包含dropout操作
  bool has_dropout = false;

  // 遍历根操作集合
  for (const std::string& op : root_ops) {
    // 如果操作名称中包含"backward"或"requires_grad_"，则视为处于训练模式
    if (op.find("backward") != std::string::npos ||
        op.find("requires_grad_") != std::string::npos) {
      is_training = true;
    }
    // 如果操作名称中包含"batch_norm"，则设置包含批归一化标志
    if (op.find("batch_norm") != std::string::npos) {
      has_batchnorm = true;
    }
    // 如果操作名称中包含"dropout"，则设置包含dropout标志
    if (op.find("dropout") != std::string::npos) {
      has_dropout = true;
    }
  }

  // 如果处于训练模式且包含批归一化操作，则调用批归一化方法
  if (is_training && has_batchnorm) {
    at::batch_norm(
        at::ones({2, 2}),
        c10::nullopt,
        c10::nullopt,
        c10::nullopt,
        c10::nullopt,
        true,
        0.1,
        0.1,
        false);
  }

  // 如果处于训练模式且包含dropout操作，则调用dropout方法
  if (is_training && has_dropout) {
    at::dropout(at::ones({20, 20, 20}), 0.2, true);
  }
}

/**
 * 调用Tensor对象上的方法，预期这些方法在该Tensor在生产环境中会被调用。
 */
void consume_tensor(const at::Tensor& t) {
  // 常量引用c指向输入Tensor t
  const at::Tensor& c = t;
  // 在CPU上复制Tensor t，并赋值给常量引用c
  c.copy_(t.cpu());
}

/**
 * 获取运行时操作及其模式的映射表。
 */
std::unordered_map<std::string, c10::FunctionSchema>
_get_runtime_ops_and_schema() {
  // 结果映射表，用于存储操作名称到函数模式的映射
  std::unordered_map<std::string, c10::FunctionSchema> result;

  // 获取所有非调度操作的操作符
  auto nonDispatcherOperators = torch::jit::getAllOperators();
  // 遍历每个操作符的模式
  for (const auto& full_op : nonDispatcherOperators) {
    // 获取操作的模式
    auto op = full_op->schema();
    // 获取操作名称
    auto op_name = op.name();
    // 如果存在重载名称，则在操作名称后添加重载名称
    if (!op.overload_name().empty()) {
      op_name += ("." + op.overload_name());
    }
    // 将操作名称及其模式添加到结果映射表中
    result.emplace(op_name, op);
  }

  // 获取所有调度操作的操作符
  auto dispatcherOperators = c10::Dispatcher::singleton().getAllOpNames();
  // 遍历每个操作的名称
  for (auto& op : dispatcherOperators) {
    // 查找操作的句柄
    const auto op_handle = c10::Dispatcher::singleton().findOp(op);
    // 如果操作句柄包含模式，则获取操作名称
    if (op_handle->hasSchema()) {
      auto op_name = op.name;
      // 如果存在重载名称，则在操作名称后添加重载名称
      if (!op.overload_name.empty()) {
        op_name += ("." + op.overload_name);
      }
      // 将操作名称及其模式添加到结果映射表中
      result.emplace(op_name, op_handle->schema());
    }
  }

  // 返回结果映射表
  return result;
}

/**
 * 从操作模式中记录自定义类。
 */
void recordCustomClassesFromOpSchemas(
    std::set<std::string>& root_ops,
    std::set<std::string>& traced_ops,
    std::unordered_map<std::string, c10::FunctionSchema>& custom_classes) {
  // 将 root_ops 和 traced_ops 中的所有元素插入 ops 集合中
  std::set<std::string> ops;
  ops.insert(root_ops.begin(), root_ops.end());
  ops.insert(traced_ops.begin(), traced_ops.end());

  // 获取运行时操作和模式的信息，存储在 ops_and_schemas 中
  auto ops_and_schemas = _get_runtime_ops_and_schema();

  // 定义 lambda 函数 record_if_class，用于记录自定义类的类名到 loaded_classes 集合中
  auto record_if_class = [&](std::string type_name) {
    // 如果类型名包含 "__torch__"，则认为是自定义类
    if (type_name.find("__torch__") != std::string::npos) {
      // 从完全限定名提取类名
      auto class_name = type_name.substr(type_name.find_last_of('.') + 1);
      // 仅保留字母、数字和下划线字符，去除其他类型指示符号
      class_name = class_name.substr(
          0,
          class_name.find_first_not_of(
              "aAbBcCdDeEfFgGhHiIjJkKlLmMnNoOpPqQrRsStTuUvVwWxXyYzZ_1234567890"));
      // 将类名插入 loaded_classes 集合中
      loaded_classes.insert(class_name);
    }
  };

  // 遍历 ops 中的每个操作名 op_name
  for (auto& op_name : ops) {
    // 如果 ops_and_schemas 中存在该操作名
    if (ops_and_schemas.find(op_name) != ops_and_schemas.end()) {
      // 获取操作对应的 schema
      auto& schema = ops_and_schemas.at(op_name);
      
      // 遍历 schema 的所有参数，记录参数类型中的自定义类名
      for (auto& arg : schema.arguments()) {
        record_if_class(arg.type()->annotation_str());
      }
      
      // 遍历 schema 的所有返回值，记录返回值类型中的自定义类名
      for (auto& ret : schema.returns()) {
        record_if_class(ret.type()->annotation_str());
      }
    }
  }
}

// 运行模型的函数，加载指定路径的模型并进行初始化
void run_model(
    const std::string& input_module_path,   // 输入的模型文件路径
    std::set<std::string>& root_ops,        // 根操作符的集合，通过引用传递
    std::set<std::string>& enabled_backends,    // 启用的后端集合，通过引用传递
    KernelDTypeTracer::kernel_tags_type& called_kernel_tags) {    // 调用的内核标签映射，通过引用传递

  // 使用 MobileModelRunner 类加载模型，标记为在 CPU 上运行并跳过操作符存在性检查
  torch::jit::mobile::MobileModelRunner module_runner(input_module_path, 0);
  
  // 获取模型的根操作符集合
  root_ops = module_runner.get_root_operators();
  
  // 输出根操作符集合的大小信息
  std::cout << "Got " << root_ops.size() << " Root Operators." << std::endl;

  // 检测是否模型含有 Metal GPU 操作符
  if (torch::jit::mobile::MobileModelRunner::set_has_metal_gpu_operators(
          root_ops)) {
    // 如果模型包含 Metal GPU 操作符，输出推断为 Metal GPU 模型的信息
    std::cout << "Inferred Metal GPU Model." << std::endl;

    // 将 GPU Metal 操作符插入根操作符集合
    root_ops.insert(gpu_metal_operators.begin(), gpu_metal_operators.end());

    // 设置调用的内核标签为 "__unused__" 对应 "Float"
    called_kernel_tags["__unused__"] = {"Float"};

    // 将启用的后端集合加入 "Metal GPU"
    enabled_backends.insert("Metal GPU");

    // 遍历模型中捆绑输入的每个张量，并应用 consume_tensor 函数
    module_runner.for_each_tensor_in_bundled_inputs(consume_tensor);
  } else {
    // 如果模型不包含 Metal GPU 操作符，输出推断为 CPU 模型的信息
    std::cout << "Inferred CPU Model." << std::endl;

    // 将启用的后端集合加入 "CPU"
    enabled_backends.insert("CPU");

    // 使用 MobileModelRunner 类加载模型，默认在 CPU 上运行
    torch::jit::mobile::MobileModelRunner mobile_module_runner(
        input_module_path);

    // 遍历模型中捆绑输入的每个张量，并应用 consume_tensor 函数
    module_runner.for_each_tensor_in_bundled_inputs(consume_tensor);

    // 对于捆绑输入的功能的更新，需要调用 get_bundled_inputs_functions_and_info 函数以获取集合
    // 因为在 tracer.cpp 中不知道哪些函数使用了捆绑输入，必须调用此函数获取集合信息
    // 如果存在的话，即使仅在 forward 函数中捆绑输入，也会使用新的捆绑输入样式
  }
}
    // 检查是否存在新样式的捆绑输入
    if (mobile_module_runner.has_new_style_bundled_inputs()) {
      // 获取多个函数的捆绑输入映射
      auto bundled_inputs_mapping =
          mobile_module_runner.get_many_functions_bundled_inputs();
      // 遍历捆绑输入映射
      for (auto& entry : bundled_inputs_mapping) {
        // 提取函数名
        std::string function_name = entry.first;
        // 提取捆绑输入的向量
        std::vector<std::vector<at::IValue>> bundled_inputs = entry.second;
        // 输出捆绑输入的数量和函数名
        std::cout << "Got " << bundled_inputs.size() << " bundled input(s) for "
                  << function_name << "\n\n";
        // 运行带有捆绑输入的函数，并获取结果
        std::vector<at::IValue> results =
            mobile_module_runner.run_with_inputs(function_name, bundled_inputs);

        // 遍历结果，处理每个张量结果（在 CPU 跟踪时需要消耗结果张量）
        for (auto& result : results) {
          // 使用 consume_tensor 函数消耗结果张量，因为 Android/Java JNI 绑定会执行相同操作
          torch::jit::mobile::for_each_tensor_in_ivalue(result, consume_tensor);
        }
      }
      // 如果没有 get_bundled_inputs_functions_and_info 函数，则默认为在此更改之前进行捆绑。
      // 如果在此处找不到捆绑输入，将抛出错误。
    } else {
      // 获取所有捆绑输入
      std::vector<std::vector<at::IValue>> bundled_inputs =
          mobile_module_runner.get_all_bundled_inputs();
      // 输出捆绑输入的数量
      std::cout << "Got " << bundled_inputs.size() << " bundled input(s)\n\n";
      // 运行带有所有捆绑输入的函数，并获取结果
      std::vector<at::IValue> results =
          mobile_module_runner.run_with_inputs(bundled_inputs);

      // 遍历结果，处理每个张量结果（在 CPU 跟踪时需要消耗结果张量）
      for (auto& result : results) {
        // 使用 consume_tensor 函数消耗结果张量，因为 Android/Java JNI 绑定会执行相同操作
        torch::jit::mobile::for_each_tensor_in_ivalue(result, consume_tensor);
      }
    }
  }
}

TracerResult trace_run(const std::string& input_module_path) {
  return trace_run(std::vector<std::string>(1, input_module_path));
}

// 使用给定的模块路径列表来执行追踪，并返回追踪结果
TracerResult trace_run(const std::vector<std::string>& input_module_paths) {
  // 设置全局的量化引擎为 QNNPACK
  at::globalContext().setQEngine(at::QEngine::QNNPACK);
  // 清空未观察到的操作符列表
  c10::ObservedOperators::getUnobservedOperatorList().clear();

  // 创建用于追踪操作符调用的对象
  torch::jit::mobile::OperatorCallTracer op_tracer;
  // 创建用于追踪内核数据类型的对象
  torch::jit::mobile::KernelDTypeTracer kdtype_tracer;
  // 创建用于追踪自定义类的对象
  torch::jit::mobile::CustomClassTracer custom_class_tracer;
  // 创建用于追踪构建特征的对象
  torch::jit::mobile::BuildFeatureTracer build_feature_tracer;

  // 调用设置方法来进行初始化设置
  call_setup_methods();

  // 创建存储各种信息的集合
  std::set<std::string> root_ops, traced_operators, enabled_backends,
      loaded_classes, build_features;
  // 创建用于存储调用的内核标签的集合
  torch::jit::mobile::KernelDTypeTracer::kernel_tags_type called_kernel_tags;

  // 使用输入的模块路径列表进行迭代
  for (auto& input_module_path : input_module_paths) {
    // 使用 QNNPACK 引擎运行模型
    at::globalContext().setQEngine(at::QEngine::QNNPACK);

    // 运行模型并收集根操作符、启用的后端和调用的内核标签
    run_model(
        input_module_path, root_ops, enabled_backends, called_kernel_tags);

    // 尝试在 FBGEMM 模式下运行模型，以扩展对超优化 QNNPack 路径的追踪范围
    try {
      at::globalContext().setQEngine(at::QEngine::FBGEMM);
      run_model(
          input_module_path, root_ops, enabled_backends, called_kernel_tags);
    } catch (std::exception& ex) {
      // 如果遇到异常，打印错误信息并跳过 FBGEMM 执行
      std::cerr
          << "ModelTracer encountered an error while attempting to run the model in FBGEMM mode"
          << ex.what() << "\n Skipping FBGEMM execution" << std::endl;
    }

    // 尝试在 QNNPACK 模式下运行模型，并设置推断模式
    try {
      at::globalContext().setQEngine(at::QEngine::QNNPACK);
      c10::InferenceMode guard(true);
      run_model(
          input_module_path, root_ops, enabled_backends, called_kernel_tags);
    } catch (std::exception& ex) {
      // 如果遇到异常，打印错误信息并跳过推断模式执行
      std::cerr
          << "ModelTracer encountered an error while attempting to run the model under an inference guard"
          << ex.what() << "\n Skipping inference guard execution" << std::endl;
  }
  // 调用依赖的方法，传入根操作列表
  call_dependent_methods(root_ops);

  // 使用操作追踪器获取被调用的运算符集合，并加锁以安全访问
  op_tracer.getCalledOperators().withLock(
      [&](std::set<std::string>& called_operators) {
        // 将获取到的被调用运算符集合赋给traced_operators变量
        traced_operators = called_operators;
      });

  // 记录从操作模式中的自定义类信息到已加载的类集合中
  recordCustomClassesFromOpSchemas(root_ops, traced_operators, loaded_classes);

  // 使用数据类型跟踪器获取调用的内核标签集合，并加锁以安全访问
  kdtype_tracer.getCalledKernelTags().withLock(
      [&](KernelDTypeTracer::kernel_tags_type& kernel_tags) {
        // 将获取到的内核标签集合插入到called_kernel_tags集合中
        called_kernel_tags.insert(kernel_tags.begin(), kernel_tags.end());
      });

  // 将始终包含在traced_operators集合中的始终包含的追踪操作插入
  traced_operators.insert(
      always_included_traced_ops.begin(), always_included_traced_ops.end());

  // 使用自定义类追踪器获取已加载的类集合，并加锁以安全访问
  custom_class_tracer.getLoadedClasses().withLock(
      [&](CustomClassTracer::custom_classes_type& custom_classes) {
        // 将获取到的自定义类集合插入到loaded_classes集合中
        loaded_classes.insert(custom_classes.begin(), custom_classes.end());
      });

  // 使用构建特征追踪器获取构建特征集合，并加锁以安全访问
  build_feature_tracer.getBuildFeatures().withLock(
      [&](BuildFeatureTracer::build_feature_type& bf) {
        // 将获取到的构建特征集合插入到build_features集合中
        build_features.insert(bf.begin(), bf.end());
      });

  // 创建追踪结果对象，包含根操作、追踪的运算符、调用的内核标签、加载的类、构建特征和启用的后端
  TracerResult tracer_result = {
      root_ops,
      traced_operators,
      called_kernel_tags,
      loaded_classes,
      build_features,
      enabled_backends};

  // 返回追踪结果对象
  return tracer_result;
}

} // namespace mobile
} // namespace jit
} // namespace torch
```