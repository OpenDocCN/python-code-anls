# `.\pytorch\tools\test\test_executorch_gen.py`

```py
# 从未来导入注释，使得代码在较旧的 Python 版本中也能使用注释功能
from __future__ import annotations

# 导入标准库中的模块
import os
import tempfile
import unittest

# 导入 PyYAML 库，用于处理 YAML 格式的数据
import yaml

# 导入自定义模块中的类和函数
from torchgen.executorch.model import ETKernelIndex, ETKernelKey
from torchgen.gen import LineLoader
from torchgen.gen_executorch import (
    ComputeCodegenUnboxedKernels,
    gen_functions_declarations,
    parse_yaml_files,
    translate_native_yaml,
)
from torchgen.model import (
    BackendIndex,
    BackendMetadata,
    DispatchKey,
    Location,
    NativeFunction,
    OperatorName,
)
from torchgen.selective_build.selector import SelectiveBuilder

# 定义一个多行字符串，包含测试用的 YAML 数据
TEST_YAML = """
- func: add.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
  device_check: NoCheck   # TensorIterator
  structured: True
  structured_inherits: TensorIteratorBase
  ufunc_inner_loop:
    Generic: add (AllAndComplex, BFloat16, Half, ComplexHalf)
    ScalarOnly: add (Bool)
  dispatch:
    SparseCPU: add_out_sparse_cpu
    SparseCUDA: add_out_sparse_cuda
    SparseCsrCPU: add_out_sparse_csr_cpu
    SparseCsrCUDA: add_out_sparse_csr_cuda
    MkldnnCPU: mkldnn_add_out
    MPS: add_out_mps

- func: add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
  device_check: NoCheck   # TensorIterator
  structured_delegate: add.out
  variants: function, method
  dispatch:
    SparseCPU, SparseCUDA: add_sparse
    SparseCsrCPU, SparseCsrCUDA: add_sparse_csr
    MkldnnCPU: mkldnn_add
    ZeroTensor: add_zerotensor
    NestedTensorCPU, NestedTensorCUDA: NestedTensor_add_Tensor
  tags: core

- func: mul.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
  device_check: NoCheck   # TensorIterator
  structured: True
  structured_inherits: TensorIteratorBase
  dispatch:
    CPU, CUDA: mul_out
    MPS: mul_out_mps
    SparseCPU: mul_out_sparse_cpu
    SparseCUDA: mul_out_sparse_cuda
    SparseCsrCPU, SparseCsrCUDA: mul_out_sparse_csr
    MkldnnCPU: mkldnn_mul_out

- func: mul.Tensor(Tensor self, Tensor other) -> Tensor
  device_check: NoCheck   # TensorIterator
  structured_delegate: mul.out
  variants: function, method
  dispatch:
    SparseCPU, SparseCUDA: mul_sparse
    SparseCsrCPU, SparseCsrCUDA: mul_sparse_csr
    MkldnnCPU: mkldnn_mul
    ZeroTensor: mul_zerotensor
    NestedTensorCPU, NestedTensorCUDA: NestedTensor_mul_Tensor
  tags: core

"""

# 定义一个多行字符串，包含测试用的内核 YAML 数据
TEST_KERNEL_YAML = """
- func: add.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
  device_check: NoCheck   # TensorIterator
  structured: True
  structured_inherits: TensorIteratorBase
  ufunc_inner_loop:
    Generic: add (AllAndComplex, BFloat16, Half, ComplexHalf)
    ScalarOnly: add (Bool)
  type_alias:
    T0: [Float, Double]
    T1: [Double, Int]
  dim_order_alias:
    D0: [0, 1, 2, 3]
    D1: [0, 3, 2, 1]
  kernels:
    - arg_meta: null
      kernel_name: default_impl
    - arg_meta:
        self: [T0, D0]
        other: [T1, D0]
        out: [T0, D0]
      kernel_name: test_impl
    - arg_meta:
        # 定义一个参数元数据的字典，描述函数参数的类型和维度信息
        self: [T1, D0]
        # 参数名为self，类型为T1，维度为D0
        other: [T1, D1]
        # 参数名为other，类型为T1，维度为D1
        out: [T0, D1]
        # 参数名为out，类型为T0，维度为D1
      kernel_name: test_impl_2
        # 指定内核的名称为test_impl_2
class TestParseNativeYaml(unittest.TestCase):
    # 设置测试前的准备工作，创建临时目录用于测试
    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()

        # 创建 ATen YAML 文件并写入测试内容
        self.aten_yaml_path = os.path.join(self.temp_dir, "test_native_functions.yaml")
        with open(self.aten_yaml_path, "w") as f:
            f.write(TEST_YAML)

        # 创建操作 YAML 文件并写入测试内容
        self.ops_yaml_path = os.path.join(self.temp_dir, "test.yaml")
        with open(self.ops_yaml_path, "w") as f:
            f.write(
                """
- op: add.out
  device_check: NoCheck   # TensorIterator
  dispatch:
    CPU: torch::executor::add_out_kernel

- op: mul.out
  device_check: NoCheck   # TensorIterator
  dispatch:
    CPU: torch::executor::mul_out_kernel
                """
            )

        # 创建标签 YAML 文件并写入测试内容
        self.tags_yaml_path = os.path.join(self.temp_dir, "tags.yaml")
        with open(self.tags_yaml_path, "w") as f:
            f.write(
                """
- tag: core
  desc: test
                """
            )

    # 测试函数：将本地 YAML 文件翻译成正确的数据格式并写入输出文件
    def test_translate_native_yaml_writes_correct_data(self) -> None:
        # 准备输出的 YAML 文件路径
        out_yaml_path = os.path.join(self.temp_dir, "out.yaml")

        # 调用翻译函数，将本地 YAML 转换为输出文件
        with open(out_yaml_path, "w") as out_file:
            translate_native_yaml(
                tags_yaml_path=self.tags_yaml_path,
                aten_yaml_path=self.aten_yaml_path,
                native_yaml_path=self.ops_yaml_path,
                use_aten_lib=False,
                out_file=out_file,
            )

        # 从输出文件中加载 YAML 数据
        with open(out_yaml_path) as out_file:
            es = yaml.load(out_file, Loader=LineLoader)

        # 断言：所有条目中均包含 "func" 字段
        self.assertTrue(all("func" in e for e in es))

        # 断言：所有条目中 "variants" 字段的值为 "function"
        self.assertTrue(all(e.get("variants") == "function" for e in es))

        # 检查：确保 YAML 中没有引入内核字段
        for e in es:
            self.assertFalse({"kernels", "type_alias", "dim_order_alias"} < e.keys())
    # 定义测试方法，用于测试解析 YAML 文件的功能
    def test_parse_yaml_files(self) -> None:
        # 设置自定义操作的 YAML 文件路径为 None
        custom_ops_yaml_path = None
        # 获取一个空选择器对象
        selector = SelectiveBuilder.get_nop_selector()
        # 设置是否使用 ATen 库为 False
        use_aten_lib = False

        # 调用解析 YAML 文件的函数，解析 ATen、标签、本地操作和自定义操作的 YAML 文件
        parsed_yaml, custom_ops_parsed_yaml = parse_yaml_files(
            aten_yaml_path=self.aten_yaml_path,
            tags_yaml_path=self.tags_yaml_path,
            native_yaml_path=self.ops_yaml_path,
            custom_ops_yaml_path=custom_ops_yaml_path,
            selector=selector,
            use_aten_lib=use_aten_lib,
        )

        # 预期的默认内核条目
        expected_kernel_entry = {"add.out": 1, "mul.out": 1}
        # 断言解析后的 YAML 中的本地函数数量等于预期内核条目的数量
        self.assertTrue(len(parsed_yaml.native_functions) == len(expected_kernel_entry))

        # 获取操作条目的内核映射索引
        op_entries = parsed_yaml.kernel_index.index
        for op_name, kernel_mapping in op_entries.items():
            # 断言每个操作的内核映射数量等于预期内核条目中该操作的数量
            self.assertTrue(
                len(kernel_mapping) == expected_kernel_entry.pop(str(op_name))
            )

        # 最终确认预期内核条目已全部匹配
        self.assertTrue(len(expected_kernel_entry) == 0)

    # 在测试结束时执行的方法，用于清理临时目录
    def tearDown(self) -> None:
        import shutil

        try:
            # 尝试递归删除临时目录
            shutil.rmtree(self.temp_dir)
        except OSError:
            # 如果发生 OSError，忽略错误
            pass
# 定义一个单元测试类，用于测试解析内核 YAML 文件的功能
class TestParseKernelYamlFiles(unittest.TestCase):
    
    # 在每个测试方法运行前设置临时目录和测试所需的 YAML 文件
    def setUp(self) -> None:
        # 创建临时目录用于存放测试文件
        self.temp_dir = tempfile.mkdtemp()
        
        # 创建一个包含测试内核函数的 YAML 文件
        self.aten_kernel_yaml_path = os.path.join(
            self.temp_dir, "test_kernel_native_functions.yaml"
        )
        with open(self.aten_kernel_yaml_path, "w") as f:
            f.write(TEST_KERNEL_YAML)
        
        # 创建测试用的操作和标签的 YAML 文件
        self.ops_yaml_path = os.path.join(self.temp_dir, "test.yaml")
        self.tags_yaml_path = os.path.join(self.temp_dir, "tags.yaml")
        
        # 写入标签 YAML 文件的内容
        with open(self.tags_yaml_path, "w") as f:
            f.write(
                """
- tag: core
  desc: test
                """
            )
        
        # 写入操作 YAML 文件的内容
        with open(self.ops_yaml_path, "w") as f:
            f.write(
                """
- op: add.out
  device_check: NoCheck   # TensorIterator
  dispatch:
    CPU: torch::executor::add_out_kernel

- op: mul.out
  device_check: NoCheck   # TensorIterator
  dispatch:
    CPU: torch::executor::mul_out_kernel
                """
            )

    # 测试函数：验证 translate_native_yaml 函数是否能正确写入目标 YAML 文件
    def test_translate_kernel_native_yaml_writes_correct_data(self) -> None:
        # 设置输出的目标 YAML 文件路径
        out_yaml_path = os.path.join(self.temp_dir, "out2.yaml")
        
        # 调用 translate_native_yaml 函数进行转换和写入
        with open(out_yaml_path, "w") as out_file:
            translate_native_yaml(
                tags_yaml_path=self.tags_yaml_path,
                aten_yaml_path=self.aten_kernel_yaml_path,
                native_yaml_path=self.ops_yaml_path,
                use_aten_lib=False,
                out_file=out_file,
            )
        
        # 读取输出的目标 YAML 文件，并使用 LineLoader 加载 YAML 数据
        with open(out_yaml_path) as out_file:
            es = yaml.load(out_file, Loader=LineLoader)
        
        # 断言所有条目中都包含 "func" 字段
        self.assertTrue(all("func" in e for e in es))
        # 断言所有条目中的 "variants" 字段都等于 "function"
        self.assertTrue(all(e.get("variants") == "function" for e in es))

        # 检查 YAML 中内核字段的持久性
        for e in es:
            # 断言每个条目都包含 "kernels", "type_alias", "dim_order_alias" 字段
            self.assertTrue({"kernels", "type_alias", "dim_order_alias"} < e.keys())

    # 测试函数：验证 parse_yaml_files 函数是否能正确解析 YAML 文件
    def test_parse_yaml_files(self) -> None:
        # 初始化自定义操作 YAML 文件路径为空
        custom_ops_yaml_path = None
        # 使用 SelectiveBuilder 类的 nop 选择器
        selector = SelectiveBuilder.get_nop_selector()
        # 设置是否使用 Aten 库为 False
        use_aten_lib = False

        # 调用 parse_yaml_files 函数解析 YAML 文件
        parsed_yaml, custom_ops_parsed_yaml = parse_yaml_files(
            aten_yaml_path=self.aten_kernel_yaml_path,
            tags_yaml_path=self.tags_yaml_path,
            native_yaml_path=self.ops_yaml_path,
            custom_ops_yaml_path=custom_ops_yaml_path,
            selector=selector,
            use_aten_lib=use_aten_lib,
        )

        # 预期的内核条目字典
        expected_kernel_entry = {"add.out": 9, "mul.out": 2}
        
        # 断言解析后的 native_functions 字段长度与预期的内核条目字典长度相等
        self.assertTrue(len(parsed_yaml.native_functions) == len(expected_kernel_entry))

        # 获取内核索引的条目
        op_entries = parsed_yaml.kernel_index.index
        
        # 遍历内核索引条目，验证每个操作的内核映射数与预期值匹配
        for op_name, kernel_mapping in op_entries.items():
            self.assertTrue(
                len(kernel_mapping) == expected_kernel_entry.pop(str(op_name))
            )

        # 验证预期的内核条目字典长度为 0，即所有预期条目已验证完毕
        self.assertTrue(len(expected_kernel_entry) == 0)

    # 在每个测试方法运行结束后，清理临时目录及其内容
    def tearDown(self) -> None:
        import shutil

        try:
            # 尝试递归删除临时目录
            shutil.rmtree(self.temp_dir)
        except OSError:
            pass
class TestGenFunctionsDeclarations(unittest.TestCase):
    # 设置测试前的准备工作
    def setUp(self) -> None:
        # 从 YAML 加载第一个自定义函数和其后端索引
        (
            self.custom_1_native_function,
            custom_1_backend_index,
        ) = NativeFunction.from_yaml(
            {"func": "custom_1::op_1() -> bool", "dispatch": {"CPU": "kernel_1"}},
            loc=Location(__file__, 1),
            valid_tags=set(),
        )
        # 从 YAML 加载第二个自定义函数和其后端索引
        (
            self.custom_2_native_function,
            custom_2_backend_index,
        ) = NativeFunction.from_yaml(
            {
                "func": "custom_2::op_2() -> bool",
                "dispatch": {"CPU": "kernel_2"},
            },
            loc=Location(__file__, 1),
            valid_tags=set(),
        )
        # 从 YAML 加载第三个自定义函数和其后端索引
        (
            self.custom_3_native_function,
            custom_3_backend_index,
        ) = NativeFunction.from_yaml(
            {
                "func": "custom_3::op_3(Tensor(a!) self, Tensor x) -> Tensor(a!)",
                "dispatch": {"CPU": "kernel_3"},
                "variants": "method",
            },
            loc=Location(__file__, 1),
            valid_tags=set(),
        )

        # 初始化后端索引字典，包括 CPU 和 QuantizedCPU 的分发键
        backend_indices: dict[DispatchKey, dict[OperatorName, BackendMetadata]] = {
            DispatchKey.CPU: {},
            DispatchKey.QuantizedCPU: {},
        }
        # 向后端索引字典添加第一个自定义函数的后端索引
        BackendIndex.grow_index(backend_indices, custom_1_backend_index)
        # 向后端索引字典添加第二个自定义函数的后端索引
        BackendIndex.grow_index(backend_indices, custom_2_backend_index)
        # 创建静态分发索引列表，包含 CPU 和 QuantizedCPU 的索引信息
        self.static_dispatch_idx = [
            BackendIndex(
                dispatch_key=k,
                use_out_as_primary=True,
                external=False,
                device_guard=False,
                index=backend_indices[k],
            )
            for k in backend_indices
        ]
        # 基于后端索引创建核心索引对象
        self.kernel_index = ETKernelIndex.from_backend_indices(backend_indices)

    # 测试不同命名空间的操作符是否正确分组
    def test_operators_with_different_namespaces_are_grouped_correctly(self) -> None:
        # 生成函数声明，包括第一个和第二个自定义函数
        declarations = gen_functions_declarations(
            native_functions=[
                self.custom_1_native_function,
                self.custom_2_native_function,
            ],
            kernel_index=self.kernel_index,
            selector=SelectiveBuilder.get_nop_selector(),
            use_aten_lib=False,
        )
        # 断言第一个自定义函数的声明在生成的声明中
        self.assertTrue(
            """
namespace custom_1 {

// custom_1::op_1() -> bool
TORCH_API inline bool op_1(torch::executor::KernelRuntimeContext & context) {
    return ::at::native::kernel_1(context);
}

} // namespace custom_1
"""
            in declarations
        )

        # 断言第二个自定义函数的声明在生成的声明中
        self.assertTrue(
            """
namespace custom_2 {

// custom_2::op_2() -> bool
TORCH_API inline bool op_2(torch::executor::KernelRuntimeContext & context) {
    return ::at::native::kernel_2(context);
}

} // namespace custom_2
        """
            in declarations
        )
        # 定义测试函数，验证 `aten_lib` 是否具有上下文参数
        declarations = gen_functions_declarations(
            # 生成函数声明，传入自定义的本地函数和内核索引
            native_functions=[
                self.custom_1_native_function,
            ],
            # 指定内核索引
            kernel_index=self.kernel_index,
            # 使用特定选择器获取 NOP（空操作）选择器
            selector=SelectiveBuilder.get_nop_selector(),
            # 启用 ATen 库
            use_aten_lib=True,
        )
        # 断言检查：确保测试通过
        self.assertTrue(
            """
// 在 custom_1 命名空间内定义的函数 op_1，接收一个 KernelRuntimeContext 引用并返回一个布尔值
TORCH_API inline bool op_1(torch::executor::KernelRuntimeContext & context) {
    // 调用 at::op_1 函数并返回其结果
    return at::op_1();
}
    [](torch::executor::KernelRuntimeContext & context, EValue** stack) {
        """
        这是一个匿名的 lambda 函数，接受两个参数：context 和 stack
        
        创建一个事件追踪器作用域，用于记录 "native_call_op_1" 操作的事件追踪信息
        """
        internal::EventTracerProfileScope event_tracer_scope(context.internal_event_tracer(), "native_call_op_1");
        
        // 在执行器范围内启用性能分析，记录 "native_call_op_1" 的性能信息
        EXECUTORCH_SCOPE_PROF("native_call_op_1");
        
        // 调用 at::native::default_kernel 函数执行某些操作，将结果存储在 result_ 中
        bool result_ = at::native::default_kernel(context, /* 参数可能被省略或是其他的代码 */);
        
        // 记录 *stack[0] 的值到事件追踪器日志中
        internal::event_tracer_log_evalue(context.internal_event_tracer(), *stack[0]);
        
        // 将 result_ 封装成 EValue 对象，并赋值给 *stack[0]
        *stack[0] = EValue(result_);
    }
    def test_codegen_unboxed_default(self) -> None:
        """
        This test checks that if there is no specialized kernel, the default kernel is used.
        """
        selector = SelectiveBuilder.from_yaml_dict(
            {
                "include_all_operators": True,
                "et_kernel_metadata": {
                    "custom_1::op_1": ["v1/7;0,1,2,3|7;0,1,2,3|7;0,1,2,3"]
                },
            }
        )
        use_aten_lib = False
        entry = (self.native_function_no_kern, self.default_kernel_entry)

        # 调用 ComputeCodegenUnboxedKernels 对象，并传入 selector 和 use_aten_lib 参数
        result = ComputeCodegenUnboxedKernels(selector, use_aten_lib)(entry)
        # 创建预期字符串，用于与结果进行比较
        expected_str = (
            """
Kernel(
    "custom_1::op_1",
    [](torch::executor::KernelRuntimeContext & context, EValue** stack) {
        // 匿名 lambda 函数，接受一个 KernelRuntimeContext 引用和一个 EValue 类型的双重指针作为参数
        // 开始一个代码块
    
        // 创建一个 EventTracerProfileScope 对象，用于跟踪事件，指定事件名称为 "native_call_op_1"
        internal::EventTracerProfileScope event_tracer_scope(context.internal_event_tracer(), "native_call_op_1");
        
        // 在 ExecutorProfilingScope 中进行性能分析，标记为 "native_call_op_1"
        EXECUTORCH_SCOPE_PROF("native_call_op_1");
    
        // 调用 at::native::default_kernel 函数，传入 context 作为参数，结果存储在 result_ 中
        bool result_ = at::native::default_kernel(context, );
    
        // 使用内部事件跟踪器记录 *stack[0] 的当前状态
        internal::event_tracer_log_evalue(context.internal_event_tracer(), *stack[0]);
    
        // 将 result_ 转换为 EValue 对象，并将其赋值给 *stack[0]
        *stack[0] = EValue(result_);
    
        // 结束 lambda 函数
    }
    ),
"""
        )
        
        # 断言检查生成的代码与预期的字符串是否一致
        self.assertEqual(expected_str, result)

    def test_codegen_unboxed_default_kernel_key_selected(self) -> None:
        """
        This test checks that if there is no specialized kernel, the default kernel is used, when the selector only has default key.
        """
        # 创建一个 SelectiveBuilder 对象，从给定的 YAML 字典中构建
        selector = SelectiveBuilder.from_yaml_dict(
            {
                "include_all_operators": True,
                "et_kernel_metadata": {"custom_1::op_1": ["default"]},
            }
        )
        # 是否使用 Aten 库的标志
        use_aten_lib = False
        # 定义一个输入元组
        entry = (self.native_function_no_kern, self.default_kernel_entry)

        # 调用 ComputeCodegenUnboxedKernels 对象的 __call__ 方法，生成代码
        result = ComputeCodegenUnboxedKernels(selector, use_aten_lib)(entry)
        # Concat used to prevent whitespace stripping
        # 预期的字符串，包含生成代码的文本形式
        expected_str = (
            """
Kernel(
    "custom_1::op_1",
    [](torch::executor::KernelRuntimeContext & context, EValue** stack) {
        """
            + """

        internal::EventTracerProfileScope event_tracer_scope(context.internal_event_tracer(), "native_call_op_1");
        EXECUTORCH_SCOPE_PROF("native_call_op_1");
        # 调用 at::native::default_kernel 函数，并传入上下文对象作为参数
        bool result_ = at::native::default_kernel(context, );
        # 记录当前操作的 evalue 到内部事件追踪器
        internal::event_tracer_log_evalue(context.internal_event_tracer(), *stack[0]);

        # 将结果包装为 EValue 类型并存入堆栈
        *stack[0] = EValue(result_);
    }
),
"""
        )

        # 断言检查生成的代码与预期的字符串是否一致
        self.assertEqual(expected_str, result)
```