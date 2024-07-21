# `.\pytorch\test\onnx\dynamo\test_exporter_api.py`

```
# 导入必要的模块和库
import io  # 提供了用于读写的核心工具
import os  # 提供了与操作系统进行交互的功能

import onnx  # 用于处理ONNX模型的库
from beartype import roar  # 用于类型检查的库

import torch  # PyTorch深度学习库
from torch.onnx import dynamo_export, ExportOptions, ONNXProgram  # PyTorch ONNX导出相关模块
from torch.onnx._internal import exporter, io_adapter  # PyTorch ONNX导出内部使用的工具
from torch.onnx._internal.exporter import (
    LargeProtobufONNXProgramSerializer,  # 用于大型模型序列化的类
    ONNXProgramSerializer,  # ONNX程序序列化的类
    ProtobufONNXProgramSerializer,  # 使用Protobuf进行ONNX程序序列化的类
    ResolvedExportOptions,  # 解析后的导出选项类
)
from torch.onnx._internal.fx import diagnostics  # ONNX导出时的诊断工具

from torch.testing._internal import common_utils  # PyTorch内部测试工具

# 定义一个简单的神经网络模型，用于示例
class SampleModel(torch.nn.Module):
    def forward(self, x):
        y = x + 1  # 输入加1
        z = y.relu()  # 对y进行ReLU操作
        return (y, z)  # 返回y和z

# 定义一个带两个输入的神经网络模型，用于示例
class SampleModelTwoInputs(torch.nn.Module):
    def forward(self, x, b):
        y = x + b  # 输入x和b相加
        z = y.relu()  # 对y进行ReLU操作
        return (y, z)  # 返回y和z

# 定义一个支持动态形状的神经网络模型，用于示例
class SampleModelForDynamicShapes(torch.nn.Module):
    def forward(self, x, b):
        return x.relu(), b.sigmoid()  # 对x进行ReLU操作，对b进行sigmoid操作，并返回结果

# 定义一个非常大的神经网络模型，用于示例
class _LargeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.randn(2**28))  # 初始化一个大小为1GB的参数
        self.param2 = torch.nn.Parameter(torch.randn(2**28))  # 初始化另一个大小为1GB的参数

    def forward(self, x):
        return self.param + self.param2 + x  # 返回参数之和与输入x之和

# 测试导出选项API的单元测试类
class TestExportOptionsAPI(common_utils.TestCase):
    # 测试在传入无效参数类型时是否抛出异常
    def test_raise_on_invalid_argument_type(self):
        expected_exception_type = roar.BeartypeException  # 预期的异常类型为BeartypeException
        with self.assertRaises(expected_exception_type):
            ExportOptions(dynamic_shapes=2)  # 尝试使用整数作为dynamic_shapes参数，应该抛出异常
        with self.assertRaises(expected_exception_type):
            ExportOptions(diagnostic_options="DEBUG")  # 尝试使用字符串作为diagnostic_options参数，应该抛出异常
        with self.assertRaises(expected_exception_type):
            ResolvedExportOptions(options=12)  # 尝试使用整数作为options参数，应该抛出异常

    # 测试动态形状选项的默认行为
    def test_dynamic_shapes_default(self):
        options = ResolvedExportOptions(ExportOptions())  # 创建默认的导出选项
        self.assertFalse(options.dynamic_shapes)  # 检查dynamic_shapes是否为False

    # 测试显式设置动态形状选项后的行为
    def test_dynamic_shapes_explicit(self):
        options = ResolvedExportOptions(ExportOptions(dynamic_shapes=None))  # 创建动态形状为None的导出选项
        self.assertFalse(options.dynamic_shapes)  # 检查dynamic_shapes是否为False
        options = ResolvedExportOptions(ExportOptions(dynamic_shapes=True))  # 创建动态形状为True的导出选项
        self.assertTrue(options.dynamic_shapes)  # 检查dynamic_shapes是否为True
        options = ResolvedExportOptions(ExportOptions(dynamic_shapes=False))  # 创建动态形状为False的导出选项
        self.assertFalse(options.dynamic_shapes)  # 检查dynamic_shapes是否为False

# 测试动态导出API的单元测试类
class TestDynamoExportAPI(common_utils.TestCase):
    # 测试默认导出行为
    def test_default_export(self):
        output = dynamo_export(SampleModel(), torch.randn(1, 1, 2))  # 导出SampleModel模型
        self.assertIsInstance(output, ONNXProgram)  # 检查输出是否为ONNXProgram类型
        self.assertIsInstance(output.model_proto, onnx.ModelProto)  # 检查输出的model_proto是否为onnx.ModelProto类型

    # 测试使用自定义选项进行导出
    def test_export_with_options(self):
        self.assertIsInstance(
            dynamo_export(
                SampleModel(),
                torch.randn(1, 1, 2),
                export_options=ExportOptions(
                    dynamic_shapes=True,
                ),
            ),
            ONNXProgram,
        )  # 使用动态形状为True的选项导出SampleModel模型，并检查输出类型是否为ONNXProgram
    # 测试保存至文件，默认使用序列化器
    def test_save_to_file_default_serializer(self):
        # 使用临时文件名上下文管理器创建临时文件路径
        with common_utils.TemporaryFileName() as path:
            # 导出模型并保存至临时文件路径
            dynamo_export(SampleModel(), torch.randn(1, 1, 2)).save(path)
            # 从临时文件加载 ONNX 模型
            onnx.load(path)

    # 测试保存至已存在缓冲区，默认使用序列化器
    def test_save_to_existing_buffer_default_serializer(self):
        # 创建一个字节流缓冲区
        buffer = io.BytesIO()
        # 导出模型并保存至字节流缓冲区
        dynamo_export(SampleModel(), torch.randn(1, 1, 2)).save(buffer)
        # 从字节流缓冲区加载 ONNX 模型
        onnx.load(buffer)

    # 测试使用指定的序列化器保存至文件
    def test_save_to_file_using_specified_serializer(self):
        expected_buffer = "I am not actually ONNX"

        # 定义自定义序列化器类
        class CustomSerializer(ONNXProgramSerializer):
            # 实现序列化方法，将预期的缓冲区内容编码写入目标缓冲区
            def serialize(
                self, onnx_program: ONNXProgram, destination: io.BufferedIOBase
            ) -> None:
                destination.write(expected_buffer.encode())

        # 使用临时文件名上下文管理器创建临时文件路径
        with common_utils.TemporaryFileName() as path:
            # 导出模型并保存至临时文件路径，指定自定义序列化器
            dynamo_export(SampleModel(), torch.randn(1, 1, 2)).save(
                path, serializer=CustomSerializer()
            )
            # 打开临时文件，断言其内容与预期缓冲区内容相同
            with open(path) as fp:
                self.assertEqual(fp.read(), expected_buffer)

    # 测试使用未继承自 ONNXProgramSerializer 的指定序列化器保存至文件
    def test_save_to_file_using_specified_serializer_without_inheritance(self):
        expected_buffer = "I am not actually ONNX"

        # 注意：不需要从 `ONNXProgramSerializer` 继承，因为它是一个协议类。
        # `beartype` 不会报错。

        # 定义自定义序列化器类
        class CustomSerializer:
            # 实现序列化方法，将预期的缓冲区内容编码写入目标缓冲区
            def serialize(
                self, onnx_program: ONNXProgram, destination: io.BufferedIOBase
            ) -> None:
                destination.write(expected_buffer.encode())

        # 使用临时文件名上下文管理器创建临时文件路径
        with common_utils.TemporaryFileName() as path:
            # 导出模型并保存至临时文件路径，指定自定义序列化器
            dynamo_export(SampleModel(), torch.randn(1, 1, 2)).save(
                path, serializer=CustomSerializer()
            )
            # 打开临时文件，断言其内容与预期缓冲区内容相同
            with open(path) as fp:
                self.assertEqual(fp.read(), expected_buffer)

    # 测试保存成功，当模型大于2GB且目标为字符串路径时
    def test_save_succeeds_when_model_greater_than_2gb_and_destination_is_str(self):
        # 使用临时文件名上下文管理器创建临时文件路径
        with common_utils.TemporaryFileName() as path:
            # 导出大型模型并保存至临时文件路径
            dynamo_export(_LargeModel(), torch.randn(1)).save(path)

    # 测试保存失败，当模型大于2GB且目标不是字符串路径时
    def test_save_raises_when_model_greater_than_2gb_and_destination_is_not_str(self):
        # 断言抛出值错误异常，提示当保存大于2GB的模型时，应提供路径字符串作为目标
        with self.assertRaisesRegex(
            ValueError,
            "'destination' should be provided as a path-like string when saving a model larger than 2GB. ",
        ):
            # 尝试导出大型模型并保存至字节流缓冲区，应抛出异常
            dynamo_export(_LargeModel(), torch.randn(1)).save(io.BytesIO())

    # 测试将 SARIF 日志保存至文件，成功导出模型时
    def test_save_sarif_log_to_file_with_successful_export(self):
        # 使用以 .sarif 结尾的临时文件名上下文管理器创建临时文件路径
        with common_utils.TemporaryFileName(suffix=".sarif") as path:
            # 导出模型并保存诊断信息至临时文件路径
            dynamo_export(SampleModel(), torch.randn(1, 1, 2)).save_diagnostics(path)
            # 断言临时文件路径存在，即保存成功
            self.assertTrue(os.path.exists(path))
    # 定义一个测试函数，验证在导出失败时是否正确保存 SARIF 日志文件
    def test_save_sarif_log_to_file_with_failed_export(self):
        # 定义一个会导致导出错误的模型类
        class ModelWithExportError(torch.nn.Module):
            def forward(self, x):
                raise RuntimeError("Export error")

        # 断言在执行导出时会抛出 RuntimeError 异常
        with self.assertRaises(RuntimeError):
            dynamo_export(ModelWithExportError(), torch.randn(1, 1, 2))
        
        # 断言导出失败后，默认的 SARIF 日志文件路径存在
        self.assertTrue(os.path.exists(exporter._DEFAULT_FAILED_EXPORT_SARIF_LOG_PATH))

    # 测试当导出失败时，能否访问到 ONNX 程序并正确处理异常
    def test_onnx_program_accessible_from_exception_when_export_failed(self):
        # 定义一个会导致导出错误的模型类
        class ModelWithExportError(torch.nn.Module):
            def forward(self, x):
                raise RuntimeError("Export error")

        # 断言在执行导出时会抛出 torch.onnx.OnnxExporterError 异常
        with self.assertRaises(torch.onnx.OnnxExporterError) as cm:
            dynamo_export(ModelWithExportError(), torch.randn(1, 1, 2))
        
        # 断言异常对象是 torch.onnx.OnnxExporterError 类型
        self.assertIsInstance(cm.exception, torch.onnx.OnnxExporterError)
        # 断言异常对象中的 onnx_program 属性是 ONNXProgram 的实例
        self.assertIsInstance(cm.exception.onnx_program, ONNXProgram)

    # 测试当 ONNX 程序在导出失败时，访问 model_proto 属性是否会引发 RuntimeError 异常
    def test_access_onnx_program_model_proto_raises_when_onnx_program_is_emitted_from_failed_export(
        self,
    ):
        # 定义一个会导致导出错误的模型类
        class ModelWithExportError(torch.nn.Module):
            def forward(self, x):
                raise RuntimeError("Export error")

        # 断言在执行导出时会抛出 torch.onnx.OnnxExporterError 异常
        with self.assertRaises(torch.onnx.OnnxExporterError) as cm:
            dynamo_export(ModelWithExportError(), torch.randn(1, 1, 2))
        
        # 获取异常对象中的 onnx_program 属性
        onnx_program = cm.exception.onnx_program
        # 断言访问 onnx_program.model_proto 属性时会抛出 RuntimeError 异常
        with self.assertRaises(RuntimeError):
            onnx_program.model_proto

    # 测试在设置 diagnostic_options 中 warnings_as_errors 为 True 时是否正确抛出异常
    def test_raise_from_diagnostic_warning_when_diagnostic_option_warning_as_error_is_true(
        self,
    ):
        # 断言在导出过程中会抛出 torch.onnx.OnnxExporterError 异常
        with self.assertRaises(torch.onnx.OnnxExporterError):
            dynamo_export(
                SampleModel(),
                torch.randn(1, 1, 2),
                export_options=ExportOptions(
                    diagnostic_options=torch.onnx.DiagnosticOptions(
                        warnings_as_errors=True
                    )
                ),
            )

    # 测试当保存参数类型无效时是否正确抛出异常
    def test_raise_on_invalid_save_argument_type(self):
        # 断言在创建 ONNXProgram 对象时会抛出 roar.BeartypeException 异常
        with self.assertRaises(roar.BeartypeException):
            ONNXProgram(torch.nn.Linear(2, 3))  # type: ignore[arg-type]
        
        # 创建一个 ONNXProgram 对象
        onnx_program = ONNXProgram(
            onnx.ModelProto(),
            io_adapter.InputAdapter(),
            io_adapter.OutputAdapter(),
            diagnostics.DiagnosticContext("test", "1.0"),
            fake_context=None,
        )
        
        # 断言在调用 save 方法时会抛出 roar.BeartypeException 异常
        with self.assertRaises(roar.BeartypeException):
            onnx_program.save(None)  # type: ignore[arg-type]
# 定义一个测试类 TestProtobufONNXProgramSerializerAPI，继承自 common_utils.TestCase
class TestProtobufONNXProgramSerializerAPI(common_utils.TestCase):

    # 定义测试方法 test_raise_on_invalid_argument_type，验证传入无效参数类型时是否抛出异常
    def test_raise_on_invalid_argument_type(self):
        # 使用 assertRaises 断言捕获 BeartypeException 异常
        with self.assertRaises(roar.BeartypeException):
            # 实例化 ProtobufONNXProgramSerializer 对象
            serializer = ProtobufONNXProgramSerializer()
            # 调用序列化方法，传入 None 和 None 参数，声明类型为 ignore[arg-type] 以忽略类型检查
            serializer.serialize(None, None)  # type: ignore[arg-type]

    # 定义测试方法 test_serialize_raises_when_model_greater_than_2gb，验证当模型大于2GB时是否抛出异常
    def test_serialize_raises_when_model_greater_than_2gb(self):
        # 生成大型模型的 ONNX 表示
        onnx_program = torch.onnx.dynamo_export(_LargeModel(), torch.randn(1))
        # 实例化 ProtobufONNXProgramSerializer 对象
        serializer = ProtobufONNXProgramSerializer()
        # 使用 assertRaisesRegex 断言捕获 ValueError 异常，并检查异常信息中是否包含指定文本
        with self.assertRaisesRegex(ValueError, "exceeds maximum protobuf size of 2GB"):
            # 调用序列化方法，传入 onnx_program 和 io.BytesIO() 对象
            serializer.serialize(onnx_program, io.BytesIO())


# 定义一个测试类 TestLargeProtobufONNXProgramSerializerAPI，继承自 common_utils.TestCase
class TestLargeProtobufONNXProgramSerializerAPI(common_utils.TestCase):

    # 定义测试方法 test_serialize_succeeds_when_model_greater_than_2gb，验证当模型大于2GB时序列化是否成功
    def test_serialize_succeeds_when_model_greater_than_2gb(self):
        # 生成大型模型的 ONNX 表示
        onnx_program = torch.onnx.dynamo_export(_LargeModel(), torch.randn(1))
        # 使用 common_utils.TemporaryFileName 创建临时文件名，并在 with 语句中使用
        with common_utils.TemporaryFileName() as path:
            # 实例化 LargeProtobufONNXProgramSerializer 对象，传入临时文件路径
            serializer = LargeProtobufONNXProgramSerializer(path)
            # 调用序列化方法，传入 onnx_program 和 io.BytesIO() 对象
            serializer.serialize(onnx_program, io.BytesIO())


# 定义一个测试类 TestONNXExportWithDynamo，继承自 common_utils.TestCase
class TestONNXExportWithDynamo(common_utils.TestCase):

    # 定义测试方法 test_args_normalization_with_no_kwargs，验证在无关键字参数的情况下参数是否正常化
    def test_args_normalization_with_no_kwargs(self):
        # 导出 SampleModelTwoInputs 模型的程序
        exported_program = torch.export.export(
            SampleModelTwoInputs(),
            (
                torch.randn(1, 1, 2),
                torch.randn(1, 1, 2),
            ),
        )
        # 使用 torch.onnx.dynamo_export 生成新导出程序的 ONNX 表示
        onnx_program_from_new_exporter = torch.onnx.dynamo_export(
            exported_program, torch.randn(1, 1, 2), torch.randn(1, 1, 2)
        )
        # 使用 torch.onnx.export 生成旧导出程序的 ONNX 表示
        onnx_program_from_old_exporter = torch.onnx.export(
            SampleModelTwoInputs(),
            (torch.randn(1, 1, 2), torch.randn(1, 1, 2)),
            dynamo=True,
        )
        # 使用 self.assertEqual 断言比较两个 ONNX 表示的模型协议是否相等
        self.assertEqual(
            onnx_program_from_new_exporter.model_proto,
            onnx_program_from_old_exporter.model_proto,
        )

    # 定义测试方法 test_args_is_tensor_not_tuple，验证当参数为张量而不是元组时的行为
    def test_args_is_tensor_not_tuple(self):
        # 导出 SampleModel 模型的程序
        exported_program = torch.export.export(SampleModel(), (torch.randn(1, 1, 2),))
        # 使用 torch.onnx.dynamo_export 生成新导出程序的 ONNX 表示
        onnx_program_from_new_exporter = torch.onnx.dynamo_export(
            exported_program, torch.randn(1, 1, 2)
        )
        # 使用 torch.onnx.export 生成旧导出程序的 ONNX 表示
        onnx_program_from_old_exporter = torch.onnx.export(
            SampleModel(), torch.randn(1, 1, 2), dynamo=True
        )
        # 使用 self.assertEqual 断言比较两个 ONNX 表示的模型协议是否相等
        self.assertEqual(
            onnx_program_from_new_exporter.model_proto,
            onnx_program_from_old_exporter.model_proto,
        )
    # 测试函数：test_args_normalization_with_kwargs
    def test_args_normalization_with_kwargs(self):
        # 使用 torch.export.export 导出 SampleModelTwoInputs 模型，传入一个张量和一个包含张量的字典作为参数
        exported_program = torch.export.export(
            SampleModelTwoInputs(), (torch.randn(1, 1, 2),), {"b": torch.randn(1, 1, 2)}
        )
        # 使用 torch.onnx.dynamo_export 从导出的程序创建 ONNX 程序，传入张量和关键字参数 b
        onnx_program_from_new_exporter = torch.onnx.dynamo_export(
            exported_program, torch.randn(1, 1, 2), b=torch.randn(1, 1, 2)
        )
        # 使用 torch.onnx.export 从 SampleModelTwoInputs 导出 ONNX 程序，传入一个张量、包含张量的字典和一个空字典作为参数
        onnx_program_from_old_exporter = torch.onnx.export(
            SampleModelTwoInputs(),
            (torch.randn(1, 1, 2), {"b": torch.randn(1, 1, 2)}),
            dynamo=True,
        )
        # 断言两个导出的 ONNX 程序的 model_proto 是否相同
        self.assertEqual(
            onnx_program_from_new_exporter.model_proto,
            onnx_program_from_old_exporter.model_proto,
        )

    # 测试函数：test_args_normalization_with_empty_dict_at_the_tail
    def test_args_normalization_with_empty_dict_at_the_tail(self):
        # 使用 torch.export.export 导出 SampleModelTwoInputs 模型，传入一个张量和一个包含张量的字典作为参数
        exported_program = torch.export.export(
            SampleModelTwoInputs(), (torch.randn(1, 1, 2),), {"b": torch.randn(1, 1, 2)}
        )
        # 使用 torch.onnx.dynamo_export 从导出的程序创建 ONNX 程序，传入张量和关键字参数 b
        onnx_program_from_new_exporter = torch.onnx.dynamo_export(
            exported_program, torch.randn(1, 1, 2), b=torch.randn(1, 1, 2)
        )
        # 使用 torch.onnx.export 从 SampleModelTwoInputs 导出 ONNX 程序，传入一个张量、包含张量的字典和一个空字典作为参数
        onnx_program_from_old_exporter = torch.onnx.export(
            SampleModelTwoInputs(),
            (torch.randn(1, 1, 2), {"b": torch.randn(1, 1, 2)}, {}),
            dynamo=True,
        )
        # 断言两个导出的 ONNX 程序的 model_proto 是否相同
        self.assertEqual(
            onnx_program_from_new_exporter.model_proto,
            onnx_program_from_old_exporter.model_proto,
        )

    # 测试函数：test_dynamic_axes_enable_dynamic_shapes_with_fully_specified_axes
    def test_dynamic_axes_enable_dynamic_shapes_with_fully_specified_axes(self):
        # 使用 torch.export.export 导出 SampleModelForDynamicShapes 模型，传入两个张量和动态形状的字典作为参数
        exported_program = torch.export.export(
            SampleModelForDynamicShapes(),
            (
                torch.randn(2, 2, 3),
                torch.randn(2, 2, 3),
            ),
            dynamic_shapes={
                "x": {
                    0: torch.export.Dim("customx_dim_0"),
                    1: torch.export.Dim("customx_dim_1"),
                    2: torch.export.Dim("customx_dim_2"),
                },
                "b": {
                    0: torch.export.Dim("customb_dim_0"),
                    1: torch.export.Dim("customb_dim_1"),
                    2: torch.export.Dim("customb_dim_2"),
                },
            },
        )
        # 使用 torch.onnx.dynamo_export 从导出的程序创建 ONNX 程序，传入两个张量作为参数
        onnx_program_from_new_exporter = torch.onnx.dynamo_export(
            exported_program,
            torch.randn(2, 2, 3),
            b=torch.randn(2, 2, 3),
        )
        # 使用 torch.onnx.export 从 SampleModelForDynamicShapes 导出 ONNX 程序，传入两个张量、包含张量的字典和一个空字典作为参数
        onnx_program_from_old_exporter = torch.onnx.export(
            SampleModelForDynamicShapes(),
            (torch.randn(2, 2, 3), {"b": torch.randn(2, 2, 3)}, {}),
            dynamic_axes={
                "x": {0: "customx_dim_0", 1: "customx_dim_1", 2: "customx_dim_2"},
                "b": {0: "customb_dim_0", 1: "customb_dim_1", 2: "customb_dim_2"},
            },
            dynamo=True,
        )
        # 断言两个导出的 ONNX 程序的 model_proto 是否相同
        self.assertEqual(
            onnx_program_from_new_exporter.model_proto,
            onnx_program_from_old_exporter.model_proto,
        )
    # 测试动态轴是否启用动态形状以及默认轴名称
    def test_dynamic_axes_enable_dynamic_shapes_with_default_axe_names(self):
        # 导出模型为 ONNX 格式，使用新的导出器
        exported_program = torch.export.export(
            SampleModelForDynamicShapes(),
            (
                torch.randn(2, 2, 3),  # 输入张量 x，形状为 (2, 2, 3)
                torch.randn(2, 2, 3),  # 输入张量 b，形状为 (2, 2, 3)
            ),
            dynamic_shapes={
                "x": {  # 定义 x 的动态形状
                    0: torch.export.Dim("customx_dim_0"),  # x 的第一个维度自定义为 customx_dim_0
                    1: torch.export.Dim("customx_dim_1"),  # x 的第二个维度自定义为 customx_dim_1
                    2: torch.export.Dim("customx_dim_2"),  # x 的第三个维度自定义为 customx_dim_2
                },
                "b": {  # 定义 b 的动态形状
                    0: torch.export.Dim("customb_dim_0"),  # b 的第一个维度自定义为 customb_dim_0
                    1: torch.export.Dim("customb_dim_1"),  # b 的第二个维度自定义为 customb_dim_1
                    2: torch.export.Dim("customb_dim_2"),  # b 的第三个维度自定义为 customb_dim_2
                },
            },
        )
        # 使用新导出的程序创建 ONNX 模型，传入相同的输入张量
        onnx_program_from_new_exporter = torch.onnx.dynamo_export(
            exported_program,
            torch.randn(2, 2, 3),  # 输入张量 x，形状为 (2, 2, 3)
            b=torch.randn(2, 2, 3),  # 输入张量 b，形状为 (2, 2, 3)
        )
        # 使用旧的导出器创建 ONNX 模型，传入相同的输入张量和动态轴定义
        onnx_program_from_old_exporter = torch.onnx.export(
            SampleModelForDynamicShapes(),
            (torch.randn(2, 2, 3), {"b": torch.randn(2, 2, 3)}, {}),  # 输入张量和动态轴定义
            dynamic_axes={
                "x": [0, 1, 2],  # x 的三个维度定义为动态轴
                "b": [0, 1, 2],  # b 的三个维度定义为动态轴
            },
            dynamo=True,  # 使用动态轴功能
        )
        # 断言新旧导出的 ONNX 模型的模型定义是否一致
        self.assertEqual(
            onnx_program_from_new_exporter.model_proto,
            onnx_program_from_old_exporter.model_proto,
        )

    # 测试动态轴是否支持部分动态形状
    def test_dynamic_axes_supports_partial_dynamic_shapes(self):
        # 导出模型为 ONNX 格式，使用新的导出器
        exported_program = torch.export.export(
            SampleModelForDynamicShapes(),
            (
                torch.randn(2, 2, 3),  # 输入张量 x，形状为 (2, 2, 3)
                torch.randn(2, 2, 3),  # 输入张量 b，形状为 (2, 2, 3)
            ),
            dynamic_shapes={
                "x": None,  # x 的形状为静态（不变）
                "b": {  # 定义 b 的动态形状
                    0: torch.export.Dim("customb_dim_0"),  # b 的第一个维度自定义为 customb_dim_0
                    1: torch.export.Dim("customb_dim_1"),  # b 的第二个维度自定义为 customb_dim_1
                    2: torch.export.Dim("customb_dim_2"),  # b 的第三个维度自定义为 customb_dim_2
                },
            },
        )
        # 使用新导出的程序创建 ONNX 模型，传入相同的输入张量
        onnx_program_from_new_exporter = torch.onnx.dynamo_export(
            exported_program,
            torch.randn(2, 2, 3),  # 输入张量 x，形状为 (2, 2, 3)
            b=torch.randn(2, 2, 3),  # 输入张量 b，形状为 (2, 2, 3)
        )
        # 使用旧的导出器创建 ONNX 模型，传入相同的输入张量和动态轴定义
        onnx_program_from_old_exporter = torch.onnx.export(
            SampleModelForDynamicShapes(),
            (torch.randn(2, 2, 3), {"b": torch.randn(2, 2, 3)}, {}),  # 输入张量和动态轴定义
            dynamic_axes={
                "b": [0, 1, 2],  # b 的三个维度定义为动态轴
            },
            dynamo=True,  # 使用动态轴功能
        )
        # 断言新旧导出的 ONNX 模型的模型定义是否一致
        self.assertEqual(
            onnx_program_from_new_exporter.model_proto,
            onnx_program_from_old_exporter.model_proto,
        )
    # 测试：在使用不相关参数时触发警告
    def test_raises_unrelated_parameters_warning(self):
        # 定义警告信息
        message = (
            "f, export_params, verbose, training, input_names, output_names, operator_export_type, opset_version, "
            "do_constant_folding, keep_initializers_as_inputs, custom_opsets, export_modules_as_functions, and "
            "autograd_inlining are not supported for dynamo export at the moment."
        )

        # 断言捕获 UserWarning 并检查其消息是否包含特定文本
        with self.assertWarnsOnceRegex(UserWarning, message):
            # 导出模型到 ONNX 格式，设置 dynamo=True 触发警告
            _ = torch.onnx.export(
                SampleModel(),
                (torch.randn(1, 1, 2),),
                dynamo=True,
            )

    # 测试：动态轴中不支持设置新的输入名称
    def test_input_names_are_not_yet_supported_in_dynamic_axes(self):
        # 断言捕获 ValueError 并检查其消息是否包含特定文本
        with self.assertRaisesRegex(
            ValueError,
            "Assinging new input names is not supported yet. Please use model forward signature "
            "to specify input names in dynamix_axes.",
        ):
            # 导出模型到 ONNX 格式，设置 input_names 和 dynamic_axes，触发异常
            _ = torch.onnx.export(
                SampleModelForDynamicShapes(),
                (
                    torch.randn(2, 2, 3),
                    torch.randn(2, 2, 3),
                ),
                input_names=["input"],
                dynamic_axes={"input": [0, 1]},
                dynamo=True,
            )

    # 测试：在 dynamo 模式下动态形状受到约束
    def test_dynamic_shapes_hit_constraints_in_dynamo(self):
        # 断言捕获 torch._dynamo.exc.UserError 并检查其消息是否包含特定文本
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            "Constraints violated",
        ):
            # 导出模型到 ONNX 格式，设置 dynamic_axes，触发异常
            _ = torch.onnx.export(
                SampleModelTwoInputs(),
                (torch.randn(2, 2, 3), torch.randn(2, 2, 3)),
                dynamic_axes={
                    "x": {0: "x_dim_0", 1: "x_dim_1", 2: "x_dim_2"},
                    "b": {0: "b_dim_0", 1: "b_dim_1", 2: "b_dim_2"},
                },
                dynamo=True,
            )

    # 测试：导出后检查生成的 ONNX 文件是否存在
    def test_saved_f_exists_after_export(self):
        # 使用临时文件名后缀为 .onnx 来保存导出的模型
        with common_utils.TemporaryFileName(suffix=".onnx") as path:
            # 导出模型到 ONNX 格式，并断言生成的文件路径存在
            _ = torch.onnx.export(
                SampleModel(), torch.randn(1, 1, 2), path, dynamo=True
            )
            self.assertTrue(os.path.exists(path))

    # 测试：当输入为 ScriptModule 或 ScriptFunction 时抛出错误
    def test_raises_error_when_input_is_script_module(self):
        # 定义一个继承自 torch.jit.ScriptModule 的类
        class ScriptModule(torch.jit.ScriptModule):
            def forward(self, x):
                return x

        # 断言捕获 TypeError 并检查其消息是否包含特定文本
        with self.assertRaisesRegex(
            TypeError,
            "Dynamo export does not support ScriptModule or ScriptFunction.",
        ):
            # 导出 ScriptModule 到 ONNX 格式，触发异常
            _ = torch.onnx.export(ScriptModule(), torch.randn(1, 1, 2), dynamo=True)
# 如果当前脚本作为主程序运行
if __name__ == "__main__":
    # 调用 common_utils 模块中的 run_tests 函数，用于运行测试
    common_utils.run_tests()
```