# `.\pytorch\benchmarks\dynamo\torchao_backend.py`

```
from typing import Any, Callable

import torch


# 设置基线模型的优化参数
def setup_baseline():
    # 禁用动态形状自动调整
    torch._dynamo.epilogue_fusion = False
    torch._dynamo.config.automatic_dynamic_shapes = False
    # 禁用强制参数静态形状
    torch._dynamo.config.force_parameter_static_shapes = False
    # 设置缓存大小限制
    torch._dynamo.config.cache_size_limit = 10000
    # 强制整数乘法与矩阵乘法融合
    torch._inductor.config.force_fuse_int_mm_with_mul = True
    # 使用混合乘法
    torch._inductor.config.use_mixed_mm = True


# 创建 TorchAO 优化上下文
def torchao_optimize_ctx(quantization: str):
    import torchao
    from torchao.quantization import (
        change_linear_weights_to_int4_woqtensors,
        change_linear_weights_to_int8_dqtensors,
        change_linear_weights_to_int8_woqtensors,
    )

    # 内部函数，应用 TorchAO 优化到模型
    def inner(model_iter_fn: Callable):
        def _torchao_apply(module: torch.nn.Module, example_inputs: Any):
            # 如果模块未量化
            if getattr(module, "_quantized", None) is None:
                # 根据量化策略选择对应的量化方法
                if quantization == "int8dynamic":
                    change_linear_weights_to_int8_dqtensors(module)
                elif quantization == "int8weightonly":
                    change_linear_weights_to_int8_woqtensors(module)
                elif quantization == "int4weightonly":
                    change_linear_weights_to_int4_woqtensors(module)
                elif quantization == "autoquant":
                    # 自动量化模块
                    torchao.autoquant(module, error_on_unseen=False)
                    # 如果输入是字典，则使用键值对作为参数调用模块
                    if isinstance(example_inputs, dict):
                        module(**example_inputs)
                    else:
                        module(*example_inputs)
                    from torchao.quantization.autoquant import AUTOQUANT_CACHE

                    # 检查自动量化缓存是否有内容
                    assert (
                        len(AUTOQUANT_CACHE) > 0
                    ), f"Err: found no autoquantizable layers in model {type(module)}, stopping autoquantization"
                elif quantization == "noquant":
                    pass
                else:
                    raise AssertionError(
                        f"Unsupposed quantization mode {quantization}."
                    )
                setattr(module, "_quantized", True)  # noqa: B010
            # 应用模型迭代函数到模块
            model_iter_fn(module, example_inputs)

        return _torchao_apply

    return inner
```