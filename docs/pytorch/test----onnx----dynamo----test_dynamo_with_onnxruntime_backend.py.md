# `.\pytorch\test\onnx\dynamo\test_dynamo_with_onnxruntime_backend.py`

```py
# Owner(s): ["module: onnx"]
from __future__ import annotations  # 导入 Python 未来版本支持的注解特性

import contextlib  # 导入上下文管理相关模块
import copy  # 导入复制相关模块
import dataclasses  # 导入数据类支持模块
import os  # 导入操作系统相关模块
import sys  # 导入系统相关模块
import unittest  # 导入单元测试模块
from typing import Tuple  # 导入类型提示模块

import onnxruntime  # 导入 ONNX 运行时模块
from parameterized import parameterized  # 导入参数化测试支持模块

import torch  # 导入 PyTorch 模块
import torch._dynamo.backends.registry  # 导入 PyTorch 动态编译后端注册模块
from torch import nn  # 导入 PyTorch 神经网络模块
from torch.onnx import (  # 导入 PyTorch 的 ONNX 模块及相关功能
    _OrtBackend as OrtBackend,
    _OrtBackendOptions as OrtBackendOptions,
    ExportOptions,
)

from torch.testing._internal import common_utils  # 导入 PyTorch 内部测试工具
from torch.testing._internal.common_utils import skipIfNNModuleInlined  # 导入跳过条件测试装饰器

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 将父目录添加到系统路径中，以便导入文件模块
import onnx_test_common  # 导入 ONNX 测试常用工具模块


def make_aot_ort(dynamic: bool = False):
    ort_backend = OrtBackend(
        options=OrtBackendOptions(
            export_options=ExportOptions(
                dynamic_shapes=dynamic,  # 设置导出选项中的动态形状参数
            )
        )
    )
    return ort_backend, ort_backend  # 返回创建的 ONNX 运行时后端实例


class TestDynamoWithONNXRuntime(onnx_test_common._TestONNXRuntime):
    def setUp(self):
        super().setUp()  # 调用父类的 setUp 方法
        torch._dynamo.reset()  # 重置 PyTorch 动态编译后端
        OrtBackend.clear_cached_instances()  # 清除 ONNX 运行时后端的缓存实例

    def tearDown(self):
        super().tearDown()  # 调用父类的 tearDown 方法
        torch._dynamo.reset()  # 重置 PyTorch 动态编译后端
        OrtBackend.clear_cached_instances()  # 清除 ONNX 运行时后端的缓存实例

    def test_get_ort_device_type(self):
        self.assertEqual(
            torch.onnx._internal.onnxruntime._get_ort_device_type("cuda"),  # 获取 ORT 设备类型为 CUDA
            torch.onnx._internal.onnxruntime.ORTC.OrtDevice.cuda(),  # 断言为 CUDA 设备
        )
        self.assertEqual(
            torch.onnx._internal.onnxruntime._get_ort_device_type("cpu"),  # 获取 ORT 设备类型为 CPU
            torch.onnx._internal.onnxruntime.ORTC.OrtDevice.cpu(),  # 断言为 CPU 设备
        )
        self.assertEqual(
            torch.onnx._internal.onnxruntime._get_ort_device_type("maia"),  # 获取 ORT 设备类型为 maia
            torch.onnx._internal.onnxruntime.ORTC.OrtDevice.npu(),  # 断言为 NPU 设备
        )

    def test_torch_compile_backend_registration(self):
        self.assertIn(
            "onnxrt", torch._dynamo.backends.registry.list_backends()
        )  # 断言 "onnxrt" 在动态编译后端注册表中
        backend = torch._dynamo.backends.registry.lookup_backend("onnxrt")  # 查找名为 "onnxrt" 的后端
        self.assertEqual(
            backend.__module__, "torch.onnx._internal.onnxruntime"
        )  # 断言后端模块为 "torch.onnx._internal.onnxruntime"

    def _test_torch_compile_backend_caching_assert_reused(
        self, options: OrtBackendOptions
    ):
        self.assertFalse(OrtBackend.get_cached_instances())  # 断言未缓存实例
        new_backend = OrtBackend.get_cached_instance_for_options(options)  # 获取针对选项的新实例
        reused_backend = OrtBackend.get_cached_instance_for_options(options)  # 获取重复使用的实例
        self.assertEqual(len(OrtBackend.get_cached_instances()), 1)  # 断言缓存实例数量为1
        self.assertIs(reused_backend, new_backend)  # 断言重复使用的实例与新实例相同
        if options is None or options.ort_session_options is None:
            # OrtBackendOptions.ort_session_options 是一个无法通过 dataclasses.asdict 序列化的 pybind11 对象
            self.assertEqual(
                new_backend,
                OrtBackend.get_cached_instance_for_options(
                    dataclasses.asdict(options) if options else None
                ),  # 获取使用序列化选项的缓存实例
            )
    @parameterized.expand(
        [
            (None,),  # 测试用例：使用默认参数
            (OrtBackendOptions(),),  # 测试用例：使用空的 OrtBackendOptions 对象
            (OrtBackendOptions(use_aot_autograd=True),),  # 测试用例：开启 AOT 自动微分
            (OrtBackendOptions(use_aot_autograd=False),),  # 测试用例：关闭 AOT 自动微分
            (OrtBackendOptions(preallocate_output=True),),  # 测试用例：预分配输出
            (OrtBackendOptions(preallocate_output=False),),  # 测试用例：不预分配输出
            (OrtBackendOptions(infer_execution_providers=True),),  # 测试用例：推断执行提供者
            (OrtBackendOptions(infer_execution_providers=False),),  # 测试用例：不推断执行提供者
            (OrtBackendOptions(preferred_execution_providers=["A", "B", "C"]),),  # 测试用例：设置优选执行提供者列表
            (
                OrtBackendOptions(
                    preferred_execution_providers=["A", "B", ("C", {"option": "value"})]
                ),  # 测试用例：设置带有选项的优选执行提供者列表
            ),
            (OrtBackendOptions(default_execution_providers=["Something"]),),  # 测试用例：设置默认执行提供者
            (
                OrtBackendOptions(
                    export_options=ExportOptions(
                        dynamic_shapes=True,
                    )  # 测试用例：导出选项设置为动态形状
                ),
            ),
            (
                OrtBackendOptions(
                    use_aot_autograd=False,
                    export_options=ExportOptions(
                        op_level_debug=True,
                        dynamic_shapes=True,
                    ),  # 测试用例：关闭 AOT 自动微分，同时设置导出选项
                ),
            ),
        ]
    )
    def test_torch_compile_backend_caching_assert_reused(
        self, options: OrtBackendOptions
    ):
        self._test_torch_compile_backend_caching_assert_reused(options)

    @parameterized.expand(
        [
            (OrtBackendOptions(ort_session_options=onnxruntime.SessionOptions()),),  # 测试用例：设置 ort_session_options 的 OrtBackendOptions 对象
        ]
    )
    def test_torch_compile_backend_caching_assert_not_reused(
        self, options: OrtBackendOptions
    ):
        with self.assertRaises(AssertionError):  # 测试用例：断言抛出 AssertionError
            self._test_torch_compile_backend_caching_assert_reused(options)

    def _test_model_numerically(
        self,
        model,
        dynamo_backend,
        example_args_collection,
        fullgraph: bool = False,
        test_backward: bool = False,
        atol: float = 1e-5,
        rtol: float = 1e-6,
    ):
        """运行原始模型和编译后模型，并比较结果。

        Args:
            model: 要测试的模型。
            dynamo_backend: 使用的动态后端。这里可以是字符串 'onnxrt' 或者
              调用 `make_aot_ort(dynamic=True)` 后返回的第一个值。
            example_args_collection: 用于测试的示例参数的元组。例如，
                (
                  (torch.randn(2), torch.randn(2)),
                  (torch.randn(4), torch.randn(4)),
                )
              如果想测试
                model(torch.randn(2), torch.randn(2)) 和
                model(torch.randn(4), torch.randn(4))
              。

        """
        # 编译模型，使用指定的后端和动态模式
        compiled_model = torch.compile(
            model if not isinstance(model, torch.nn.Module) else copy.deepcopy(model),
            backend=dynamo_backend,
            dynamic=True,
            fullgraph=fullgraph,
        )

        # 对于每组示例参数，运行原始模型和编译后模型，并比较结果
        for example_args in example_args_collection:
            # 运行原始模型
            baseline_result = model(*example_args)
            # 运行编译后模型
            result = compiled_model(*example_args)
            
            # 如果基准结果是 torch.Tensor 类型
            if isinstance(baseline_result, torch.Tensor):
                # 比较结果是否相近
                torch.testing.assert_close(
                    baseline_result, result, atol=atol, rtol=rtol
                )
                
                # 如果需要测试反向传播
                if test_backward:
                    # 计算基准模型和编译后模型的梯度
                    baseline_result.sum().backward()
                    result.sum().backward()
                    
                    # 对比每个参数的梯度是否相近
                    for baseline_param, param in zip(
                        model.parameters(), compiled_model.parameters()
                    ):
                        torch.testing.assert_close(
                            baseline_param.grad, param.grad, atol=atol, rtol=rtol
                        )
            
            # 如果基准结果不是 torch.Tensor 类型
            else:
                # 多输出情况下不支持计算反向传播
                assert (
                    test_backward is False
                ), "Calculating backward with multiple outputs is not supported yet."
                
                # 对比每个元素是否相近
                for baseline_elem, result_elem in zip(baseline_result, result):
                    torch.testing.assert_close(
                        baseline_elem, result_elem, atol=atol, rtol=rtol
                    )

    def _assert_counting_information(
        self,
        ort_backend: OrtBackend,
        # 会话运行的次数。
        # 如果没有图中断，这应该与前向调用的总数相同。
        expected_execution_count: int,
        # 缓存的 GraphModule 数量。
        # 一个图中断会使得模型映射到两个 GraphModule。
        number_of_cached_graph_modules: int,
        # 每个 GraphModule 缓存的 ONNX 模型数量。
        # number_of_exported_onnx_models[i] 包含 OrtBackend._all_ort_execution_info.execution_info_per_graph_module.values() 中第 i 个元素（类型为 torch.fx.GraphModule）导出的 ONNX 模型数量。
        number_of_exported_onnx_models_for_all_graph_modules: Tuple[int, ...],
    ):
        # 断言实际执行次数与预期执行次数相等
        self.assertEqual(expected_execution_count, ort_backend.execution_count)
        # 断言缓存的图模块数量与预期数量相等
        self.assertEqual(
            len(ort_backend._all_ort_execution_info.execution_info_per_graph_module),
            number_of_cached_graph_modules,
        )
        # 断言导出的ONNX模型数量与预期数量相等
        self.assertEqual(
            len(ort_backend._all_ort_execution_info.execution_info_per_graph_module),
            len(number_of_exported_onnx_models_for_all_graph_modules),
        )
        # 遍历每个图模块的执行信息和预期导出的ONNX模型数量
        for (
            onnx_info,
            expected_number_of_onnx_models,
        ) in zip(
            ort_backend._all_ort_execution_info.execution_info_per_graph_module.values(),
            number_of_exported_onnx_models_for_all_graph_modules,
        ):
            # 断言每个图模块的实际ONNX模型数量与预期数量相等
            self.assertEqual(len(onnx_info), expected_number_of_onnx_models)

    # 断言所有ONNX模型中动态输入和输出形状
    def _assert_dynamic_input_and_output_shapes_in_all_onnx_models(self, backend):
        # 遍历每个图模块的ONNX会话信息
        for (
            onnx_session_infos
        ) in backend._all_ort_execution_info.execution_info_per_graph_module.values():
            for onnx_session_info in onnx_session_infos:
                inputs_have_dynamic_shapes = False
                # 检查输入是否有动态形状
                for input in onnx_session_info.input_value_infos:
                    if hasattr(input.type, "tensor_type") and hasattr(
                        input.type.tensor_type, "shape"
                    ):
                        for dim in input.type.tensor_type.shape.dim:
                            inputs_have_dynamic_shapes = (
                                inputs_have_dynamic_shapes or hasattr(dim, "dim_param")
                            )
                output_have_dynamic_shapes = False
                # 检查输出是否有动态形状
                for output in onnx_session_info.output_value_infos:
                    if hasattr(output.type, "tensor_type") and hasattr(
                        output.type.tensor_type, "shape"
                    ):
                        for dim in output.type.tensor_type.shape.dim:
                            output_have_dynamic_shapes = (
                                output_have_dynamic_shapes or hasattr(dim, "dim_param")
                            )
                # 断言输入和输出都有动态形状
                self.assertTrue(inputs_have_dynamic_shapes)
                self.assertTrue(output_have_dynamic_shapes)

    # 参数化测试扩展
    @parameterized.expand(
        [
            (True,),  # 参数为True
            (False,),  # 参数为False
        ]
    )
    def test_elementwise_function_single_output(self, test_local_backend: bool):
        # 定义一个包含不同批次随机张量的示例参数集合
        example_args_collection = tuple(
            (torch.randn(batch, dtype=torch.float32),) for batch in (2, 4, 6, 8, 10)
        )

        # 定义一个元素级模型函数，接收一个张量 x 作为输入，应用 ReLU 和 sigmoid 激活函数后返回结果
        def elementwise_model(x: torch.Tensor):
            y = x.relu()  # 对输入张量应用 ReLU 激活函数
            z = y.sigmoid()  # 对中间结果 y 应用 sigmoid 激活函数
            return z  # 返回处理后的结果张量 z

        # 根据 test_local_backend 参数决定是否使用本地后端，获取相关的 AOT 编译器和 ONNXRuntime 实例
        if test_local_backend:
            local_aot_ort, local_ort = make_aot_ort(dynamic=True)
        else:
            # 如果不使用本地后端，则使用全局注册的 ONNXRuntime 后端来编译测试模型
            local_aot_ort, local_ort = "onnxrt", None

        # 使用数值方法测试定义的元素级模型，传入模型函数、本地后端信息和示例参数集合
        self._test_model_numerically(
            elementwise_model,
            local_aot_ort,
            example_args_collection,
        )

        # 当使用本地后端时，验证本地 ORT 实例的计数信息
        # 由于有 5 个不同的批次大小进行测试，OrtBackend._ort_acclerated_call 应该被调用 5 次
        if test_local_backend:
            assert local_ort is not None
            self._assert_counting_information(
                local_ort,
                expected_execution_count=len(example_args_collection),
                # 由于该本地 ORT 实例只编译了一个函数，其缓存中应该只有一个 GraphModule
                number_of_cached_graph_modules=1,
                # 动态形状启用时，应该只有一个导出的 ONNX 模型来支持不同的批次大小
                number_of_exported_onnx_models_for_all_graph_modules=(1,),
            )

    @parameterized.expand(
        [
            (True,),
            (False,),
        ]
    )
    def test_elementwise_function_multiple_output(self, test_local_backend: bool):
        # 定义一个包含不同批次随机张量的示例参数集合
        example_args_collection = tuple(
            (torch.randn(batch, dtype=torch.float32),) for batch in (2, 4, 8)
        )

        # 定义一个具有多个输出的元素级模型函数，接收一个张量 w 作为输入，执行多个操作后返回三个张量 x, y, z
        def elementwise_model_with_multiple_outputs(w: torch.Tensor):
            x = w + w  # 计算输入张量的加法操作
            y = x.relu()  # 对中间结果 x 应用 ReLU 激活函数
            z = y * y  # 对中间结果 y 进行元素级乘法操作
            return x, y, z  # 返回计算后的三个张量 x, y, z

        # 根据 test_local_backend 参数决定是否使用本地后端，获取相关的 AOT 编译器和 ONNXRuntime 实例
        if test_local_backend:
            local_aot_ort, local_ort = make_aot_ort(dynamic=True)
        else:
            local_aot_ort, local_ort = "onnxrt", None

        # 使用数值方法测试定义的具有多个输出的元素级模型，传入模型函数、本地后端信息和示例参数集合
        self._test_model_numerically(
            elementwise_model_with_multiple_outputs,
            local_aot_ort,
            example_args_collection,
        )

        # 当使用本地后端时，验证本地 ORT 实例的计数信息
        # 由于有 3 个不同的批次大小进行测试，OrtBackend._ort_acclerated_call 应该被调用 3 次
        if test_local_backend:
            assert local_ort is not None
            self._assert_counting_information(
                local_ort,
                expected_execution_count=len(example_args_collection),
                # 由于该本地 ORT 实例只编译了一个函数，其缓存中应该只有一个 GraphModule
                number_of_cached_graph_modules=1,
                # 动态形状启用时，应该只有一个导出的 ONNX 模型来支持不同的批次大小
                number_of_exported_onnx_models_for_all_graph_modules=(1,),
            )
    # 定义一个测试方法，用于测试使用本地后端的多层感知机模型
    def test_mlp_with_local_backend(self, test_local_backend: bool):
        # 创建一个包含不同大小随机张量的元组作为示例参数集合
        example_args_collection = tuple(
            (torch.randn(batch, 2, dtype=torch.float32),) for batch in (1, 2, 4, 6, 8)
        )

        # 定义一个多层感知机模型类，继承自nn.Module
        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                # 定义两个线性层，输入维度为2，输出维度为4和2
                self.fc1 = nn.Linear(2, 4, bias=True)
                self.fc2 = nn.Linear(4, 2, bias=True)

            # 定义前向传播方法
            def forward(self, tensor_x: torch.Tensor):
                tensor_x = self.fc1(tensor_x)  # 第一线性层计算
                tensor_x = torch.sigmoid(tensor_x)  # sigmoid激活函数
                tensor_x = self.fc2(tensor_x)  # 第二线性层计算
                tensor_x = torch.sigmoid(tensor_x)  # sigmoid激活函数
                return tensor_x

        # 根据测试标志选择本地后端测试环境
        if test_local_backend:
            local_aot_ort, local_ort = make_aot_ort(dynamic=True)  # 调用函数创建本地AOT编译的ONNX运行时
        else:
            local_aot_ort, local_ort = "onnxrt", None

        # 使用定义的MLP模型测试其数值计算的正确性
        self._test_model_numerically(
            MLP(),
            local_aot_ort,
            example_args_collection,
        )

        # 如果测试本地后端，断言本地运行时对象不为空
        if test_local_backend:
            assert local_ort is not None
            # 断言本地运行时对象的计数信息是否符合预期
            self._assert_counting_information(
                local_ort,
                # 期望OrtBackend._ort_acclerated_call被调用5次，因为有5种不同的批处理大小进行测试
                expected_execution_count=len(example_args_collection),
                # 由于本地运行时仅编译了一个函数，其缓存中应该只有两个GraphModule，
                # 一个用于批处理大小2、4、6、8，另一个用于批处理大小1
                number_of_cached_graph_modules=2,
                # 由于启用了动态形状，应该只有一个ONNX模型来支持不同的批处理大小
                number_of_exported_onnx_models_for_all_graph_modules=(1, 1),
            )
        ):
            # 导入需要的库，禁止检查未使用的变量
            from transformers import LlamaConfig  # noqa: F811
            from transformers.models.llama.modeling_llama import LlamaModel  # noqa: F811

            # 设置 LLAMA 模型的配置参数
            config = LlamaConfig(
                num_hidden_layers=1,
                vocab_size=1024,
                hidden_size=16,
                intermediate_size=16,
                max_position_embeddings=256,
                num_attention_heads=2,
                hidden_dropout_prob=0.0,
                attention_dropout_prob=0.0,
            )

            # 设置 LLAMA 模型的注意力实现方式为 "eager"
            config._attn_implementation = "eager"

            # 定义一个包装器类，用于包装 LLAMA 模型
            class LlamaModelWrapper(torch.nn.Module):
                def __init__(self, config):
                    super().__init__()
                    self.llama = LlamaModel(config)

                # 定义模型的前向传播函数
                def forward(self, input_ids, attention_mask, position_ids):
                    # 调用 LLAMA 模型的前向传播
                    decoder_output = self.llama(
                        input_ids, attention_mask, position_ids, return_dict=False
                    )
                    return decoder_output[0]

            # 定义生成示例输入的函数
            def generate_example_inputs(batch: int, seq: int):
                # shape: batch x seq x hidden_size
                input_ids = torch.randint(0, 7, size=(batch, seq), dtype=torch.int64)
                # 通常情况下，attention_mask 的形状是 batch x seq x seq 的张量。
                # 但为了绕过模型中的某些控制流，这里使用 None。
                attention_mask = None
                position_ids = torch.arange(0, seq, dtype=torch.int64)
                position_ids = position_ids.unsqueeze(0).view(-1, seq)
                return input_ids, attention_mask, position_ids

            # 设置用于测试的多个示例参数组的集合
            example_args_collection = (
                generate_example_inputs(2, 8),
                generate_example_inputs(4, 7),
                generate_example_inputs(9, 15),
            )

            # 根据 test_local_backend 的值选择本地后端测试或使用 ONNX 运行时
            if test_local_backend:
                local_aot_ort, local_ort = make_aot_ort(dynamic=True)
            else:
                local_aot_ort, local_ort = "onnxrt", None

            # 创建并评估 LLAMA 模型
            model = LlamaModelWrapper(config).eval()

            # 使用数值方法测试模型的输出
            self._test_model_numerically(
                model,
                local_aot_ort,
                example_args_collection,
                fullgraph=True,
                test_backward=test_backward,
                atol=1e-4,
                rtol=1e-4,
            )

            # 如果使用本地后端测试，则进行断言检查
            if test_local_backend:
                assert local_ort is not None
                number_of_captured_graphs = 2 if test_backward else 1
                execution_count = len(example_args_collection) * number_of_captured_graphs
                # 断言检查本地运行时的计数信息
                self._assert_counting_information(
                    local_ort,
                    expected_execution_count=execution_count,
                    number_of_cached_graph_modules=number_of_captured_graphs,
                    number_of_exported_onnx_models_for_all_graph_modules=(1,)
                    * number_of_captured_graphs,
                )
                # 断言检查所有 ONNX 模型中的动态输入和输出形状
                self._assert_dynamic_input_and_output_shapes_in_all_onnx_models(local_ort)
    @parameterized.expand(
        [
            (True,),  # 参数化测试，使用True和False两种情况进行测试
            (False,),
        ]
    )
    def test_dump_model(self, test_local_backend: bool):
        @contextlib.contextmanager
        def onnxrt_dump_path(path):
            key = "ONNXRT_DUMP_PATH"
            before = os.environ.get(key, None)
            os.environ[key] = path  # 设置环境变量ONNXRT_DUMP_PATH为指定路径
            yield  # 生成器的yield，用于返回上下文管理器
            if before is None:
                del os.environ[key]
            else:
                os.environ[key] = before  # 恢复环境变量ONNXRT_DUMP_PATH的原值

        example_args_collection = tuple(
            (torch.randn(batch, 2, dtype=torch.float32),) for batch in (1, 2, 4, 6, 8)  # 创建包含不同批次大小的随机张量的元组
        )

        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(2, 4, bias=True)  # 创建第一个全连接层，输入维度为2，输出维度为4
                self.fc2 = nn.Linear(4, 2, bias=True)  # 创建第二个全连接层，输入维度为4，输出维度为2

            def forward(self, tensor_x: torch.Tensor):
                tensor_x = self.fc1(tensor_x)  # 第一全连接层的前向传播
                tensor_x = torch.sigmoid(tensor_x)  # 使用sigmoid激活函数
                tensor_x = self.fc2(tensor_x)  # 第二全连接层的前向传播
                tensor_x = torch.sigmoid(tensor_x)  # 使用sigmoid激活函数
                return tensor_x

        if test_local_backend:
            local_aot_ort, local_ort = make_aot_ort(dynamic=True)  # 根据测试标志选择本地AOT或ORT后端
        else:
            local_aot_ort, local_ort = "onnxrt", None

        prefix = f"test_dump_model_{'local' if test_local_backend else 'onnxrt'}_"  # 根据测试标志确定文件名前缀
        expected = f"{prefix}0.onnx"  # 期望的ONNX模型文件名
        expected_graph = f"{prefix}0.txt"  # 期望的图形化文件名
        if os.path.exists(expected):
            os.remove(expected)  # 如果存在期望的ONNX模型文件，删除它
        if os.path.exists(expected_graph):
            os.remove(expected_graph)  # 如果存在期望的图形化文件，删除它
        not_expected = f"{prefix}1.onnx"  # 不期望存在的ONNX模型文件名
        self.assertFalse(os.path.exists(not_expected))  # 断言不应存在名为not_expected的文件

        model = MLP()  # 创建MLP模型实例
        compiled_model = torch.compile(
            model if not isinstance(model, torch.nn.Module) else copy.deepcopy(model),  # 编译模型为指定后端的格式
            backend=local_aot_ort,
            dynamic=True,
        )

        self.assertFalse(os.path.exists(expected))  # 断言不应存在名为expected的文件
        self.assertFalse(os.path.exists(not_expected))  # 断言不应存在名为not_expected的文件

        with onnxrt_dump_path(prefix):  # 使用上下文管理器设置ONNXRT_DUMP_PATH环境变量
            example_args = example_args_collection[0]  # 获取示例参数集合中的第一个参数
            result = compiled_model(*example_args)  # 使用编译后的模型执行前向传播
            self.assertTrue(os.path.exists(expected))  # 断言应存在名为expected的文件
            self.assertTrue(os.path.exists(expected_graph))  # 断言应存在名为expected_graph的文件
            self.assertFalse(os.path.exists(not_expected))  # 断言不应存在名为not_expected的文件

            result = compiled_model(*example_args)  # 再次使用编译后的模型执行前向传播
            self.assertTrue(os.path.exists(expected))  # 断言应存在名为expected的文件
            self.assertFalse(os.path.exists(not_expected))  # 断言不应存在名为not_expected的文件
    def test_mix_device_inputs(self):
        # 创建一个在 CUDA 设备上随机生成数据的张量
        data = torch.randn(4, 8, device="cuda")
        # 创建一个在 CPU 上随机生成数据的参考张量
        ref_data = torch.randn(8, 4, device="cpu")

        def reshape_wrapper(data, ref_cpu_data):
            # 为了确保 ref_cpu_data 被捕捉到计算图中，增加一个虚拟的操作
            ref_cpu_data += 1
            # 获取 ref_cpu_data 的形状
            shape = ref_cpu_data.shape
            # 使用 GPU 和 CPU 输入调用 torch.reshape
            return torch.reshape(data, shape)

        # 编译 reshape_wrapper 函数为 ONNX 运行时模型
        compiled_model = torch.compile(
            reshape_wrapper,
            backend="onnxrt",
            dynamic=True,
        )

        # 使用 data 和 ref_data 作为输入调用编译后的模型
        result = compiled_model(data, ref_data)

        # 断言编译模型的输出与使用 data.view(ref_data.shape) 的结果近似相等
        self.assertTrue(torch.allclose(result, data.view(ref_data.shape)))

    def test_no_input(self):
        def reshape_wrapper():
            # 一个没有输入的模型
            ones = torch.ones(4, 8)
            zeros = torch.zeros(4, 8)
            return ones + zeros

        recorded_models = []

        def record_onnx_model_transform(onnx_model):
            # 记录转换过程中的 ONNX 模型
            recorded_models.append(onnx_model)

        # 编译 reshape_wrapper 函数为 ONNX 运行时模型，并记录转换过程中的模型
        compiled_model = torch.compile(
            reshape_wrapper,
            backend="onnxrt",
            dynamic=True,
            options=torch.onnx._OrtBackendOptions(
                pre_ort_model_transforms=[
                    record_onnx_model_transform,
                ]
            ),
        )

        # 使用编译后的模型进行推理，无需输入
        result = compiled_model()

        # 断言记录的模型数量为1
        self.assertEqual(len(recorded_models), 1)
        # 注意：被优化器常量折叠
        self.assertTrue(
            "Constant" in [node.op_type for node in recorded_models[0].graph.node]
        )

        # 断言结果与 torch.ones(4, 8) 相等
        self.assertEqual(result, torch.ones(4, 8))
    def test_custom_onnx_transform(self):
        # 这个测试包括两个部分：
        # 1. 调用并记录一个已注册的 ONNX 转换的模型。
        # 2. 调用并改变一个已注册的 ONNX 转换的模型。

        # Part 1: 记录被转换函数看到的 ONNX 模型。
        # 这个列表包含了由 record_onnx_model_transform 记录的模型。
        recorded_models = []

        def record_onnx_model_transform(onnx_model):
            # 记录被转换函数看到的 ONNX 模型。
            recorded_models.append(onnx_model)

        def example_model(x: torch.Tensor):
            y = torch.sigmoid(x)
            z = x + y
            return z

        compiled_model = torch.compile(
            example_model,
            backend="onnxrt",
            dynamic=True,
            options=torch.onnx._OrtBackendOptions(
                pre_ort_model_transforms=[record_onnx_model_transform]
            ),
        )

        x = torch.randn(2)
        assert len(recorded_models) == 0
        y = compiled_model(x)
        assert len(recorded_models) == 1

        # Part 2: 改变被转换函数看到的 ONNX 模型，以便 ORT 收到不同的模型。
        # 注意：此函数会被优化器优化掉。
        def replace_relu_with_sigmoid(onnx_model):
            for node in onnx_model.graph.node:
                if node.op_type == "Relu":
                    node.op_type = "Sigmoid"

        def another_example_model(x: torch.Tensor):
            y = torch.relu(x)
            z = x + y
            return z

        another_compiled = torch.compile(
            another_example_model,
            backend="onnxrt",
            dynamic=True,
            options=torch.onnx._OrtBackendOptions(
                pre_ort_model_transforms=[
                    replace_relu_with_sigmoid,
                    record_onnx_model_transform,
                ]
            ),
        )

        another_y = another_compiled(x)
        # 我们通过上面的两个 torch.compile 调用记录了 2 个模型，由 `record_onnx_model_transform` 记录。
        assert len(recorded_models) == 2
        # 因为在 replace_sigmoid_with_relu 中将 "Relu" 改为 "Sigmoid"，
        # 所以结果应该与之前的 y 相同。
        torch.testing.assert_close(y, another_y)
        # another_example_model 仍然使用 "Relu"，所以结果应该与 y 不同。
        self.assertFalse(torch.allclose(y, another_example_model(x)))
# 如果当前脚本作为主程序执行（而不是被导入为模块），则执行以下代码块
if __name__ == "__main__":
    # 导入的模块或包中的常用工具，运行其测试函数
    common_utils.run_tests()
```