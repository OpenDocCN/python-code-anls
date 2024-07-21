# `.\pytorch\test\inductor\test_aot_inductor_utils.py`

```py
# Owner(s): ["module: inductor"]

# 导入PyTorch相关模块
import torch
import torch._export
import torch._inductor
import torch.export._trace
import torch.fx._pytree as fx_pytree

# 导入测试工具相关模块
from torch.testing._internal.common_utils import IS_FBCODE

# 导入pytree工具模块
from torch.utils import _pytree as pytree


# 包装器模块，用于将传入的模型包装为Module子类
class WrapperModule(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


# AOTIRunnerUtil类，提供AOT编译器相关功能
class AOTIRunnerUtil:
    @classmethod
    def compile(
        cls,
        model,
        example_inputs,
        options=None,
        dynamic_shapes=None,
        disable_constraint_solver=False,
    ):
        # 如果model不是torch.nn.Module的实例，将其包装为WrapperModule
        if not isinstance(model, torch.nn.Module):
            model = WrapperModule(model)
        
        # 根据配置是否进行预调度导出模型
        if torch._inductor.config.is_predispatch:
            # 使用_export函数进行导出，并返回导出的模块
            ep = torch.export._trace._export(
                model, example_inputs, dynamic_shapes=dynamic_shapes, pre_dispatch=True
            )
            gm = ep.module()  # 获取导出模块
        else:
            # 将模型导出到Torch IR中
            gm = torch.export._trace._export_to_torch_ir(
                model,
                example_inputs,
                dynamic_shapes=dynamic_shapes,
                disable_constraint_solver=disable_constraint_solver,
                # 禁用restore_fqn标志，因为可以依赖于来自Dynamo的映射
                restore_fqn=False,
            )

        # 使用torch.no_grad()上下文管理器，进行AOT编译
        with torch.no_grad():
            # 调用_aot_compile函数编译gm，返回编译后的动态链接库路径
            so_path = torch._inductor.aot_compile(gm, example_inputs, options=options)  # type: ignore[arg-type]

        return so_path  # 返回动态链接库路径

    @classmethod
    def load_runner(cls, device, so_path):
        # 如果是FBCODE环境
        if IS_FBCODE:
            # 导入fb中的test_aot_inductor_model_runner_pybind模块
            from .fb import test_aot_inductor_model_runner_pybind

            # 返回Runner对象，根据device是否为"cpu"进行选择不同的初始化方法
            return test_aot_inductor_model_runner_pybind.Runner(
                so_path, device == "cpu"
            )
        else:
            # 如果不是FBCODE环境，返回AOTIModelContainerRunnerCpu或AOTIModelContainerRunnerCuda对象
            return (
                torch._C._aoti.AOTIModelContainerRunnerCpu(so_path, 1)
                if device == "cpu"
                else torch._C._aoti.AOTIModelContainerRunnerCuda(so_path, 1, device)
            )

    @classmethod
    def load(cls, device, so_path):
        # TODO: 统一FBCODE和开源社区行为，只使用torch._export.aot_load

        # 如果是FBCODE环境
        if IS_FBCODE:
            # 加载runner
            runner = AOTIRunnerUtil.load_runner(device, so_path)

            # 定义优化函数optimized，根据runner获取调用规范，并根据in_spec将输入展平
            def optimized(*args, **kwargs):
                call_spec = runner.get_call_spec()
                in_spec = pytree.treespec_loads(call_spec[0])
                out_spec = pytree.treespec_loads(call_spec[1])
                flat_inputs = fx_pytree.tree_flatten_spec((args, kwargs), in_spec)
                flat_outputs = runner.run(flat_inputs)
                return pytree.tree_unflatten(flat_outputs, out_spec)

            return optimized  # 返回优化后的函数
        else:
            # 如果不是FBCODE环境，使用torch._export.aot_load加载AOT模型
            return torch._export.aot_load(so_path, device)

    @classmethod
    # 定义一个类方法 `run`，接受多个参数用于模型优化和执行
    def run(
        cls,
        device,
        model,
        example_inputs,
        options=None,
        dynamic_shapes=None,
        disable_constraint_solver=False,
    ):
        # 调用 AOTIRunnerUtil 的 compile 方法编译模型，返回优化后的动态链接库路径
        so_path = AOTIRunnerUtil.compile(
            model,
            example_inputs,
            options=options,
            dynamic_shapes=dynamic_shapes,
            disable_constraint_solver=disable_constraint_solver,
        )
        # 调用 AOTIRunnerUtil 的 load 方法加载优化后的模型
        optimized = AOTIRunnerUtil.load(device, so_path)
        # 使用优化后的模型执行，并返回结果
        return optimized(*example_inputs)

    # 定义一个类方法 `run_multiple`，用于多次运行模型并返回结果列表
    @classmethod
    def run_multiple(
        cls,
        device,
        model,
        list_example_inputs,
        options=None,
        dynamic_shapes=None,
    ):
        # 调用 AOTIRunnerUtil 的 compile 方法编译模型，以第一个示例输入作为参数
        so_path = AOTIRunnerUtil.compile(
            model,
            list_example_inputs[0],
            options=options,
            dynamic_shapes=dynamic_shapes,
        )
        # 调用 AOTIRunnerUtil 的 load 方法加载优化后的模型
        optimized = AOTIRunnerUtil.load(device, so_path)
        # 初始化一个空列表，用于存储每次运行模型后的输出张量
        list_output_tensors = []
        # 遍历输入的多个示例输入，每次使用优化后的模型执行，并将结果添加到输出列表中
        for example_inputs in list_example_inputs:
            list_output_tensors.append(optimized(*example_inputs))
        # 返回所有运行结果组成的列表
        return list_output_tensors
```