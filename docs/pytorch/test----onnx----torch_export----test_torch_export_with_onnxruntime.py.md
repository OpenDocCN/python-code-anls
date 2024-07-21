# `.\pytorch\test\onnx\torch_export\test_torch_export_with_onnxruntime.py`

```py
# Owner(s): ["module: onnx"]
# 导入未来版本的注解功能，用于类型注解
from __future__ import annotations

# 导入系统相关的操作模块
import os
import sys

# 导入PyTorch及其ONNX相关的模块
import torch
import torch.onnx

# 导入PyTorch内部测试相关的工具模块
from torch.testing._internal import common_utils
# 导入PyTorch的_pytree模块
from torch.utils import _pytree as torch_pytree

# 将当前文件的父目录添加到系统路径中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 导入自定义的ONNX测试公共函数库
import onnx_test_common

# 继承自自定义的ONNX运行时测试类
class TestFxToOnnxWithOnnxRuntime(onnx_test_common._TestONNXRuntime):
    
    # 比较ONNX和Torch导出程序的方法
    def _compare_onnx_and_torch_exported_program(
        self,
        torch_exported_program,  # Torch导出的程序对象
        onnx_exported_program,   # ONNX导出的程序对象
        input_args,              # 输入参数
        input_kwargs=None,       # 输入关键字参数，默认为None
        rtol=1e-03,              # 相对误差容差
        atol=1e-07,              # 绝对误差容差
    ):
        # 避免可变默认参数
        if input_kwargs is None:
            input_kwargs = {}

        # 注意：ONNXProgram保存对原始ref_model的引用（而不是副本），包括其state_dict。
        # 因此，必须在ref_model()之前运行ONNXProgram()，以防止ref_model.forward()更改state_dict。
        # 否则，ref_model可能会更改state_dict上的缓冲区，这些缓冲区将被ONNXProgram.__call__()使用。
        onnx_outputs = onnx_exported_program(*input_args, **input_kwargs)
        if isinstance(torch_exported_program, torch.export.ExportedProgram):
            torch_outputs = torch_exported_program.module()(*input_args, **input_kwargs)
        else:
            torch_outputs = torch_exported_program(*input_args, **input_kwargs)
        
        # 将Torch格式的输出调整为ONNX格式
        torch_outputs_onnx_format = onnx_exported_program.adapt_torch_outputs_to_onnx(
            torch_outputs
        )
        
        # 检查输出的数量是否一致，如果不一致则抛出断言错误
        if len(torch_outputs_onnx_format) != len(onnx_outputs):
            raise AssertionError(
                f"Expected {len(torch_outputs_onnx_format)} outputs, got {len(onnx_outputs)}"
            )
        
        # 使用torch.testing.assert_close函数比较每个输出
        for torch_output, onnx_output in zip(torch_outputs_onnx_format, onnx_outputs):
            torch.testing.assert_close(
                torch_output, torch.tensor(onnx_output), rtol=rtol, atol=atol
            )

    # 测试带动态输入的导出程序
    def test_exported_program_with_dynamic_input(self):
        # 定义一个简单的模型，用于测试
        class Model(torch.nn.Module):
            def forward(self, x):
                return x + 1.0

        # 创建一个随机张量作为输入
        x = torch.randn(2, 3, 4, dtype=torch.float)
        # 创建一个动态维度对象
        dim0 = torch.export.Dim("dim0")
        # 导出模型为Torch导出程序对象
        exported_program = torch.export.export(
            Model(), (x,), dynamic_shapes={"x": {0: dim0}}
        )
        # 使用Torch导出程序对象导出为ONNX程序对象
        onnx_program = torch.onnx.dynamo_export(exported_program, x)

        # 创建不同维度的输入张量
        y = torch.randn(3, 3, 4, dtype=torch.float)
        # 调用_compare_onnx_and_torch_exported_program方法比较输出
        self._compare_onnx_and_torch_exported_program(
            exported_program, onnx_program, input_args=(y,)
        )
    # 定义一个测试方法，用于测试从文件中导入导出的程序
    def test_exported_program_as_input_from_file(self):
        # 导入临时文件模块
        import tempfile

        # 定义一个简单的神经网络模型
        class Model(torch.nn.Module):
            # 前向传播函数，将输入张量加上1.0并返回
            def forward(self, x):
                return x + 1.0

        # 生成一个形状为(1, 1, 2)的随机张量x，数据类型为float
        x = torch.randn(1, 1, 2, dtype=torch.float)
        # 导出模型的程序
        exported_program = torch.export.export(Model(), args=(x,))
        # 转换为ONNX程序
        onnx_program = torch.onnx.dynamo_export(exported_program, x)

        # 使用临时命名的文件对象f，后缀为.pte，保存导出的程序
        with tempfile.NamedTemporaryFile(suffix=".pte") as f:
            # 将导出的程序保存到临时文件中
            torch.export.save(exported_program, f.name)
            # 删除导出的程序对象，确保后续加载的是文件中的内容
            del exported_program
            # 加载保存在文件中的导出程序
            loaded_exported_program = torch.export.load(f.name)

        # 调用比较函数，比较加载的导出程序和ONNX程序
        self._compare_onnx_and_torch_exported_program(
            loaded_exported_program, onnx_program, input_args=(x,)
        )

    # 定义一个测试方法，测试在跟踪期间使用特定输入导出的程序
    def test_exported_program_with_specialized_input_during_tracing(self):
        # 定义一个简单的神经网络模型
        class Foo(torch.nn.Module):
            # 前向传播函数，将两个输入张量相加并返回
            def forward(self, x, y):
                return x + y

        # 创建模型实例f
        f = Foo()

        # 创建一个大小为(7, 5)的全1张量作为输入
        tensor_input = torch.ones(7, 5)
        # 创建一个维度对象，表示在跟踪期间对"x"维度0进行特化，最小值为6
        dim0_x = torch.export.Dim("dim0_x", min=6)
        # 定义动态形状字典，"x"维度0使用dim0_x特化，"y"维度不特化
        dynamic_shapes = {"x": {0: dim0_x}, "y": None}
        # 在跟踪期间使用特定的输入5导出程序
        exported_program = torch.export.export(
            f, (tensor_input, 5), dynamic_shapes=dynamic_shapes
        )
        # 转换为ONNX程序
        onnx_program = torch.onnx.dynamo_export(exported_program, tensor_input, 5)

        # 创建另一个输入张量，大小为(8, 5)，全1
        additional_tensor_input = torch.ones(8, 5)
        # 调用比较函数，比较导出的程序和ONNX程序
        self._compare_onnx_and_torch_exported_program(
            exported_program, onnx_program, input_args=(additional_tensor_input, 5)
        )
    def test_onnx_program_supports_retraced_graph(self):
        class Bar(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("buf", torch.ones(1))  # 在模块初始化时注册名为"buf"的缓冲区，初始值为1

            def forward(self, x):
                self.buf.add_(1)  # 缓冲区"buf"的值加1
                return x.sum() + self.buf.sum()  # 返回输入张量x的总和加上缓冲区"buf"的总和

        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("buf", torch.zeros(1))  # 在模块初始化时注册名为"buf"的缓冲区，初始值为0
                self.bar = Bar()  # 创建Bar类的实例self.bar作为Foo类的属性

            def forward(self, x):
                self.buf.add_(1)  # 缓冲区"buf"的值加1
                bar = self.bar(x)  # 调用self.bar的forward方法，并传入输入张量x
                self.bar.buf.add_(2)  # self.bar的缓冲区"buf"的值加2
                return bar.sum() + self.buf.sum()  # 返回bar的总和加上缓冲区"buf"的总和

        tensor_input = torch.ones(5, 5)  # 创建一个5x5全1张量tensor_input
        exported_program = torch.export.export(Foo(), (tensor_input,))  # 导出Foo类实例的模型，并传入tensor_input作为输入参数

        dim0_x = torch.export.Dim("dim0_x")  # 创建一个维度描述对象dim0_x
        # 注意：如果输入是ExportedProgram，我们需要指定dynamic_shapes作为一个元组。
        reexported_program = torch.export.export(
            exported_program.module(), (tensor_input,), dynamic_shapes=({0: dim0_x},)
        )
        reexported_onnx_program = torch.onnx.dynamo_export(
            reexported_program, tensor_input
        )

        additional_tensor_input = torch.ones(7, 5)  # 创建一个7x5全1张量additional_tensor_input
        self._compare_onnx_and_torch_exported_program(
            reexported_program,
            reexported_onnx_program,
            input_args=(additional_tensor_input,),
        )

    def test_onnx_program_supports_none_arg_name_in_dynamic(self):
        class Foo(torch.nn.Module):
            def forward(self, a, b):
                return a.sum() + b.sum()

        foo = Foo()  # 创建Foo类的实例foo

        dim = torch.export.Dim("dim")  # 创建一个维度描述对象dim
        exported_program = torch.export.export(
            foo, (torch.randn(4, 4), torch.randn(4, 4)), dynamic_shapes=(None, {0: dim})
        )
        onnx_program = torch.onnx.dynamo_export(
            exported_program, torch.randn(4, 4), torch.randn(4, 4)
        )

        test_inputs = (
            torch.randn(4, 4),  # 创建一个4x4随机张量
            torch.randn(7, 4),  # 创建一个7x4随机张量
        )
        self._compare_onnx_and_torch_exported_program(
            exported_program, onnx_program, test_inputs
        )
    # 定义一个测试方法，用于验证 ONNX 程序在支持带关键字参数名的情况下的行为
    def test_onnx_program_suppors_non_arg_name_with_kwarg(self):
        # 定义一个继承自 torch.nn.Module 的子类 Foo
        class Foo(torch.nn.Module):
            # 定义 forward 方法，接受四个参数 a, b, kw1, kw2，并返回它们的求和结果
            def forward(self, a, b, kw1, kw2):
                return a.sum() + b.sum() + kw1.sum() - kw2.sum()

        # 创建 Foo 类的实例 foo
        foo = Foo()

        # 创建两个维度对象 dim 和 dim_for_kw1
        dim = torch.export.Dim("dim")
        dim_for_kw1 = torch.export.Dim("dim_for_kw1")

        # 导出模型程序，传入 foo 实例，两个张量作为位置参数，
        # 以及一个字典作为关键字参数，指定了 kw2 和 kw1 的值，同时通过 dynamic_shapes
        # 指定了第一个关键字参数的动态性，尽管用户传入的顺序不同
        exported_program = torch.export.export(
            foo,
            (torch.randn(4, 4), torch.randn(4, 4)),
            {"kw2": torch.ones(4, 4), "kw1": torch.zeros(4, 4)},
            dynamic_shapes=(None, {0: dim}, {0: dim_for_kw1}, None),
        )

        # 使用 torch.onnx.dynamo_export 方法将导出的程序转换为 ONNX 程序，
        # 并传入与导出时相同的参数
        onnx_program = torch.onnx.dynamo_export(
            exported_program,
            torch.randn(4, 4),
            torch.randn(4, 4),
            kw2=torch.ones(4, 4),
            kw1=torch.zeros(4, 4),
        )

        # 创建测试输入 test_inputs 和 test_kwargs，分别包含两个张量和一个不同维度的张量
        test_inputs = (torch.randn(4, 4), torch.randn(7, 4))
        test_kwargs = {"kw2": torch.ones(4, 4), "kw1": torch.zeros(9, 4)}

        # 调用自定义的 _compare_onnx_and_torch_exported_program 方法，
        # 用于比较导出的程序和 ONNX 程序的行为是否一致，传入测试输入和测试关键字参数
        self._compare_onnx_and_torch_exported_program(
            exported_program, onnx_program, test_inputs, test_kwargs
        )

    # 定义一个测试方法，验证导出程序作为输入时，缓冲区的变异行为
    def test_exported_program_as_input_lifting_buffers_mutation(self):
        # 遍历 persistent 参数为 True 和 False 的两种情况
        for persistent in (True, False):

            # 定义一个自定义模块 CustomModule，继承自 torch.nn.Module
            class CustomModule(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    # 注册一个名为 "my_buffer" 的缓冲区，持久性根据 persistent 参数决定
                    self.register_buffer(
                        "my_buffer", torch.tensor(4.0), persistent=persistent
                    )

                # 定义 forward 方法，接受输入 x 和 b，并返回它们的和
                def forward(self, x, b):
                    output = x + b
                    # 通过原位加法修改缓冲区 self.my_buffer
                    self.my_buffer.add_(1.0) + 3.0
                    return output

            # 创建输入张量 input_x 和 input_b
            input_x = torch.rand((3, 3), dtype=torch.float32)
            input_b = torch.randn(3, 3)
            # 创建 CustomModule 的实例 model
            model = CustomModule()

            # 创建维度对象 dim
            dim = torch.export.Dim("dim")

            # 导出模型程序，传入 model 实例和输入张量作为参数，
            # 并通过 dynamic_shapes 指定输入张量的动态性
            exported_program = torch.export.export(
                model,
                (
                    input_x,
                    input_b,
                ),
                dynamic_shapes=({0: dim}, {0: dim}),
            )

            # 使用 torch.onnx.dynamo_export 方法将导出的程序转换为 ONNX 程序，
            # 并传入与导出时相同的输入张量
            onnx_program = torch.onnx.dynamo_export(exported_program, input_x, input_b)

            # 创建额外的输入张量 additional_inputs_x 和 additional_inputs_b，
            # 与原始输入形状不同
            additional_inputs_x = torch.rand((4, 3), dtype=torch.float32)
            additional_inputs_b = torch.randn(4, 3)

            # 调用自定义的 _compare_onnx_and_torch_exported_program 方法，
            # 用于比较导出的程序和 ONNX 程序的行为是否一致，传入额外的输入张量
            self._compare_onnx_and_torch_exported_program(
                exported_program,
                onnx_program,
                (
                    additional_inputs_x,
                    additional_inputs_b,
                ),
            )
    def test_onnx_program_supports_non_arg_name_with_container_type(self):
        # 定义一个名为 test_onnx_program_supports_non_arg_name_with_container_type 的测试方法
        class Foo(torch.nn.Module):
            # 定义一个名为 Foo 的类，继承自 torch.nn.Module
            def forward(self, a, b):
                # 定义类的前向传播方法，接受参数 a 和 b
                return a[0].sum() + a[1].sum() + b.sum()

        foo = Foo()  # 创建 Foo 类的实例对象

        inp_a = (torch.randn(4, 4), torch.randn(4, 4))  # 创建输入元组 inp_a，包含两个形状为 (4, 4) 的张量
        inp_b = torch.randn(4, 4)  # 创建输入张量 inp_b，形状为 (4, 4)
        inp = (inp_a, inp_b)  # 创建输入元组 inp，包含 inp_a 和 inp_b

        count = 0  # 初始化计数器 count

        def dynamify_inp(x):
            # 定义一个内部函数 dynamify_inp，接受参数 x
            # 标记第二个输入 a[1] 为动态输入
            nonlocal count  # 声明 count 为非局部变量
            if count == 1:
                dim = torch.export.Dim("dim", min=3)  # 创建一个维度对象 dim，最小值为 3
                count += 1  # 增加 count 计数
                return {0: dim}  # 返回包含动态维度信息的字典
            count += 1  # 增加 count 计数
            return None  # 如果没有动态维度要求，返回 None

        dynamic_shapes = torch_pytree.tree_map(dynamify_inp, inp)  # 使用 dynamify_inp 处理输入 inp，得到动态形状信息
        exported_program = torch.export.export(foo, inp, dynamic_shapes=dynamic_shapes)
        # 导出 Torch 模型为可执行程序，传入模型 foo、输入 inp 和动态形状信息 dynamic_shapes
        onnx_program = torch.onnx.dynamo_export(exported_program, inp_a, inp_b)
        # 使用动态形状信息导出 ONNX 程序，传入导出的程序 exported_program 和输入 inp_a、inp_b

        # 注意：输入格式需要谨慎处理，必须与模型导出时保持一致。
        test_inputs = ((torch.randn(4, 4), torch.randn(6, 4)), torch.randn(4, 4))
        # 定义测试输入 test_inputs，包含两个元组，第一个元组中第二个张量形状为 (6, 4)，第二个张量形状为 (4, 4)
        self._compare_onnx_and_torch_exported_program(
            exported_program, onnx_program, test_inputs
        )
        # 调用内部方法 _compare_onnx_and_torch_exported_program，比较导出的 Torch 程序和 ONNX 程序的结果

    def test_onnx_program_supports_lazy_module_kwargs(self):
        # 定义一个名为 test_onnx_program_supports_lazy_module_kwargs 的测试方法
        class LazyModule(torch.nn.modules.lazy.LazyModuleMixin, torch.nn.Module):
            # 定义一个名为 LazyModule 的类，混入 LazyModuleMixin 和继承自 torch.nn.Module
            def initialize_parameters(self, *args, **kwargs):
                # 定义初始化参数的方法，接受任意位置参数和关键字参数
                pass

            def forward(self, x, y):
                # 定义类的前向传播方法，接受参数 x 和 y
                return x + y  # 返回 x 和 y 的求和结果

        m = LazyModule()  # 创建 LazyModule 类的实例对象
        dim = torch.export.Dim("dim")  # 创建一个维度对象 dim
        dynamic_shapes = ({0: dim}, {0: dim})  # 创建动态形状信息，分别应用于输入 x 和 y
        exported_program = torch.export.export(
            m,
            (),  # 空元组，因为 LazyModule 的 forward 方法不接受位置参数
            {"x": torch.randn(3, 3), "y": torch.randn(3, 3)},  # 关键字参数，指定输入 x 和 y 的张量
            dynamic_shapes=dynamic_shapes,  # 传入动态形状信息
        )
        onnx_program = torch.onnx.dynamo_export(
            exported_program, x=torch.randn(3, 3), y=torch.randn(3, 3)
        )
        # 使用动态形状信息导出 ONNX 程序，传入导出的程序 exported_program 和输入 x、y 的张量

        # 注意：模型应该使用与其导出时相匹配的输入格式。
        inputs = {"x": torch.randn(6, 3), "y": torch.randn(6, 3)}
        # 定义输入字典 inputs，包含键为 "x" 和 "y" 的张量，形状分别为 (6, 3)
        self._compare_onnx_and_torch_exported_program(
            exported_program, onnx_program, input_args=(), input_kwargs=inputs
        )
        # 调用内部方法 _compare_onnx_and_torch_exported_program，比较导出的 Torch 程序和 ONNX 程序的结果
# 如果当前脚本作为主程序运行（而不是作为模块导入），则执行以下代码块
if __name__ == "__main__":
    # 调用 common_utils 模块中的 run_tests 函数，用于执行测试
    common_utils.run_tests()
```