# `.\pytorch\torch\onnx\_internal\fx\patcher.py`

```py
# mypy: allow-untyped-defs
# 导入必要的模块和函数
import copy  # 复制对象的模块
import functools  # 函数装饰器和其他高阶功能的模块
import io  # 用于处理文件流的模块
from typing import List, Union  # 引入类型提示的模块和Union类型

import torch  # PyTorch深度学习框架


# TODO: Remove after https://github.com/huggingface/safetensors/pull/318
@functools.lru_cache(None)
# 判断是否同时引入了safetensors和transformers
def has_safetensors_and_transformers():
    try:
        # safetensors不是导出器要求的一部分，但一些huggingface模型需要它
        import safetensors  # type: ignore[import]  # noqa: F401
        import transformers  # type: ignore[import]  # noqa: F401

        from safetensors import torch as safetensors_torch  # noqa: F401

        return True
    except ImportError:
        return False


class ONNXTorchPatcher:
    """Context manager to temporarily patch PyTorch during FX-to-ONNX export.

    This class is a collection of "patches" required by FX-to-ONNX exporter.

    This context overrides several torch functions to support symbolic
    export of large scale models.

    torch.load:
        This function is patched to record the files PyTorch stores model
        parameters and buffers. Downstream FX-to-ONNX exporter can create
        initializers from these files.
    torch.fx._symbolic_trace._wrapped_methods_to_patch:
        This list is extended with (torch.Tensor, "__getitem__") so that
        weight[x, :, y] becomes exportable with torch.fx.symbolic_trace.
    safetensors.torch.load_file:
        This function is patched to allow safetensors to be loaded within
        FakeTensorMode. Remove after https://github.com/huggingface/safetensors/pull/318

    Search for ONNXTorchPatcher in test_fx_to_onnx_with_onnxruntime.py for
    example usage.

    TODO: Should this really be a global patcher? Can we make it a local patcher?
        A reason for splitting this into several patchers is to patch one part of the code
        as a collateral damage of patching another part of the code. For example, we
        for tracing model with torch._dynamo.export, we don't need to patch
        `torch.fx._symbolic_trace._wrapped_methods_to_patch`
    """
    def __init__(self):
        # List of file paths processed by torch.load.
        self.paths: List[Union[str, io.BufferedIOBase]] = []
        
        def torch_load_wrapper(f, *args, **kwargs):
            # Record path for later serialization into ONNX proto
            self.paths.append(f)
            # Then, call the original torch.load.
            return self.torch_load(f, *args, **kwargs)
        
        # Original version of torch.load.
        self.torch_load = torch.load
        
        # Wrapper or modified version of torch functions.
        self.torch_load_wrapper = torch_load_wrapper
        
        if has_safetensors_and_transformers():
            import safetensors
            import transformers
            
            def safetensors_load_file_wrapper(filename, device="cpu"):
                # Record path for later serialization into ONNX proto
                self.paths.append(filename)
                result = {}
                with safetensors.torch.safe_open(  # type: ignore[attr-defined]
                    filename, framework="pt", device=device
                ) as f:
                    for k in f.keys():
                        fake_mode = torch._guards.detect_fake_mode()
                        if not fake_mode:
                            result[k] = f.get_tensor(k)
                        else:
                            empty_tensor = f.get_slice(k)
                            result[k] = torch.empty(
                                tuple(empty_tensor.get_shape()),
                                dtype=safetensors.torch._getdtype(
                                    empty_tensor.get_dtype()
                                ),
                            )
                return result
            
            # Assigning wrapped functions to attributes.
            self.safetensors_torch_load_file = safetensors.torch.load_file
            self.safetensors_torch_load_file_wrapper = safetensors_load_file_wrapper
            self.transformers_modeling_utils_safe_load_file = (
                transformers.modeling_utils.safe_load_file
            )
    # 进入上下文管理器时调用的方法，用于设置环境以支持 Torch 加载和 FX 符号化跟踪
    def __enter__(self):
        # 替换 Torch 的加载函数为自定义的包装器
        torch.load = self.torch_load_wrapper

        # 备份原始的被包装方法列表
        self.torch_fx__symbolic_trace__wrapped_methods_to_patch = (
            torch.fx._symbolic_trace._wrapped_methods_to_patch
        )

        # 深拷贝被包装方法列表，以确保不会影响原始列表
        desired_wrapped_methods = copy.deepcopy(
            torch.fx._symbolic_trace._wrapped_methods_to_patch
        )

        # 如果缺少 '__getitem__' 方法的追踪，则添加它到包装方法列表中
        if (torch.Tensor, "__getitem__") not in desired_wrapped_methods:
            # 通过在 patching 列表中添加 `__getitem__` 方法，使得 tensor 的索引可以通过
            # torch.fx.symbolic_trace 进行追踪。否则，`tensor[x, :, y]` 无法被追踪。
            # 这是因为 `__getitem__` 方法既不在 torch 领域下，也不是 aten 运算符，
            # 因此 patching 或类似的代理生成机制不会自动发生。
            # 注意，torch.fx.symbolic_trace 为启用 FX_PATCH_GETITEM 环境变量，
            # 可以通过以下行进行 patching。
            desired_wrapped_methods.append((torch.Tensor, "__getitem__"))

        # 更新 Torch FX 符号化跟踪中的被包装方法列表
        torch.fx._symbolic_trace._wrapped_methods_to_patch = desired_wrapped_methods

        # 如果安装了 SafeTensors 和 Transformers，进行相应的配置
        if has_safetensors_and_transformers():
            import safetensors
            import transformers

            # 替换 SafeTensors 中的 torch.load_file 方法为自定义的包装器
            safetensors.torch.load_file = self.safetensors_torch_load_file_wrapper
            # 替换 Transformers 中的 safe_load_file 方法为自定义的包装器
            transformers.modeling_utils.safe_load_file = (
                self.safetensors_torch_load_file_wrapper
            )

    # 退出上下文管理器时调用的方法，用于恢复原始环境设置
    def __exit__(self, exc_type, exc_value, traceback):
        # 恢复 Torch 的加载函数为原始的函数
        torch.load = self.torch_load

        # 恢复 Torch FX 符号化跟踪中的被包装方法列表为原始值
        torch.fx._symbolic_trace._wrapped_methods_to_patch = (
            self.torch_fx__symbolic_trace__wrapped_methods_to_patch
        )

        # 如果安装了 SafeTensors 和 Transformers，进行相应的恢复配置
        if has_safetensors_and_transformers():
            import safetensors
            import transformers

            # 恢复 SafeTensors 中的 torch.load_file 方法为原始的函数
            safetensors.torch.load_file = self.safetensors_torch_load_file
            # 恢复 Transformers 中的 safe_load_file 方法为原始的函数
            transformers.modeling_utils.safe_load_file = (
                self.transformers_modeling_utils_safe_load_file
            )
```