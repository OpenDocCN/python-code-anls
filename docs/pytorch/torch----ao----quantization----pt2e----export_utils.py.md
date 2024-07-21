# `.\pytorch\torch\ao\quantization\pt2e\export_utils.py`

```py
# mypy: allow-untyped-defs
# 引入 types 模块，用于处理类型相关操作
import types

# 引入 PyTorch 库
import torch
import torch.nn.functional as F

# 从 PyTorch 的量化工具中引入特定函数
from torch.ao.quantization.utils import _assert_and_get_unique_device

# 定义模块中可以导出的变量
__all__ = [
    "model_is_exported",
]

# 定义一个用于包装可调用对象的类，使其成为 torch.nn.Module 的子类
class _WrapperModule(torch.nn.Module):
    """Class to wrap a callable in an :class:`torch.nn.Module`. Use this if you
    are trying to export a callable.
    """

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, *args, **kwargs):
        """Simple forward that just calls the ``fn`` provided to :meth:`WrapperModule.__init__`."""
        return self.fn(*args, **kwargs)


# 检查给定的 torch.nn.Module 是否已导出
def model_is_exported(m: torch.nn.Module) -> bool:
    """
    Return True if the `torch.nn.Module` was exported, False otherwise
    (e.g. if the model was FX symbolically traced or not traced at all).
    """
    return isinstance(m, torch.fx.GraphModule) and any(
        "val" in n.meta for n in m.graph.nodes
    )


# 在模型的图表示中替换 dropout 模式以在训练和评估模式之间切换
def _replace_dropout(m: torch.fx.GraphModule, train_to_eval: bool):
    """
    Switch dropout patterns in the model between train and eval modes.

    Dropout has different behavior in train vs eval mode. For exported models,
    however, calling `model.train()` or `model.eval()` does not automatically switch
    the dropout behavior between the two modes, so here we need to rewrite the aten
    dropout patterns manually to achieve the same effect.

    See https://github.com/pytorch/pytorch/issues/103681.
    """
    # 避免循环依赖
    from .utils import _get_aten_graph_module_for_pattern

    # 确保子图匹配是自包含的
    m.graph.eliminate_dead_code()
    m.recompile()

    # 针对不同 inplace 设置，定义在训练模式下的 dropout 函数和在评估模式下的 dropout 函数
    for inplace in [False, True]:

        def dropout_train(x):
            return F.dropout(x, p=0.5, training=True, inplace=inplace)

        def dropout_eval(x):
            return F.dropout(x, p=0.5, training=False, inplace=inplace)

        example_inputs = (torch.randn(1),)
        if train_to_eval:
            match_pattern = _get_aten_graph_module_for_pattern(
                _WrapperModule(dropout_train), example_inputs
            )
            replacement_pattern = _get_aten_graph_module_for_pattern(
                _WrapperModule(dropout_eval), example_inputs
            )
        else:
            match_pattern = _get_aten_graph_module_for_pattern(
                _WrapperModule(dropout_eval), example_inputs
            )
            replacement_pattern = _get_aten_graph_module_for_pattern(
                _WrapperModule(dropout_train), example_inputs
            )

        # 引入替换模式的函数
        from torch.fx.subgraph_rewriter import replace_pattern_with_filters

        # 使用指定的模式替换子图
        replace_pattern_with_filters(
            m,
            match_pattern,
            replacement_pattern,
            match_filters=[],
            ignore_literals=True,
        )
        m.recompile()


def _replace_batchnorm(m: torch.fx.GraphModule, train_to_eval: bool):
    """
    """
    Switch batchnorm patterns in the model between train and eval modes.
    
    Batchnorm has different behavior in train vs eval mode. For exported models,
    however, calling `model.train()` or `model.eval()` does not automatically switch
    the batchnorm behavior between the two modes, so here we need to rewrite the aten
    batchnorm patterns manually to achieve the same effect.
    """
    # TODO(Leslie): This function still fails to support custom momentum and eps value.
    # Enable this support in future updates.
    
    # Avoid circular dependencies by importing necessary function
    from .utils import _get_aten_graph_module_for_pattern
    
    # Ensure that any dead code in the graph is eliminated
    m.graph.eliminate_dead_code()
    # Recompile the model to reflect any changes made
    m.recompile()
    
    # Define a function to simulate batch normalization in training mode
    def bn_train(
        x: torch.Tensor,
        bn_weight: torch.Tensor,
        bn_bias: torch.Tensor,
        bn_running_mean: torch.Tensor,
        bn_running_var: torch.Tensor,
    ):
        return F.batch_norm(
            x, bn_running_mean, bn_running_var, bn_weight, bn_bias, training=True
        )
    
    # Define a function to simulate batch normalization in evaluation mode
    def bn_eval(
        x: torch.Tensor,
        bn_weight: torch.Tensor,
        bn_bias: torch.Tensor,
        bn_running_mean: torch.Tensor,
        bn_running_var: torch.Tensor,
    ):
        return F.batch_norm(
            x, bn_running_mean, bn_running_var, bn_weight, bn_bias, training=False
        )
    
    # Example inputs used for pattern matching
    example_inputs = (
        torch.randn(1, 1, 3, 3),  # x
        torch.randn(1),          # bn_weight
        torch.randn(1),          # bn_bias
        torch.randn(1),          # bn_running_mean
        torch.randn(1),          # bn_running_var
    )
    
    # Ensure the model operates on a single device, preventing errors
    device = _assert_and_get_unique_device(m)
    # Determine if the device is CUDA capable for GPU processing
    is_cuda = device is not None and device.type == "cuda"
    
    # Obtain the ATen graph module corresponding to the training batchnorm pattern
    bn_train_aten = _get_aten_graph_module_for_pattern(
        _WrapperModule(bn_train),
        example_inputs,
        is_cuda,
    )
    
    # Obtain the ATen graph module corresponding to the evaluation batchnorm pattern
    bn_eval_aten = _get_aten_graph_module_for_pattern(
        _WrapperModule(bn_eval),
        example_inputs,
        is_cuda,
    )
    
    # Decide which pattern to match and which one to replace based on mode switch
    if train_to_eval:
        match_pattern = bn_train_aten
        replacement_pattern = bn_eval_aten
    else:
        match_pattern = bn_eval_aten
        replacement_pattern = bn_train_aten
    
    # Import the function to replace subgraph patterns with filters
    from torch.fx.subgraph_rewriter import replace_pattern_with_filters
    
    # Replace the identified pattern in the model with the corresponding replacement
    replace_pattern_with_filters(
        m,
        match_pattern,
        replacement_pattern,
        match_filters=[],    # No specific filters applied during matching
        ignore_literals=True,  # Ignore literal values during matching
    )
    
    # Recompile the modified model to reflect the pattern replacements
    m.recompile()
# 将导出的图模块切换到评估模式的辅助函数
def _move_exported_model_to_eval(model: torch.fx.GraphModule):
    """
    Move an exported GraphModule to eval mode.

    This function modifies the model to behave as if it's in evaluation mode,
    but only for specific operations like dropout and batch normalization.

    Args:
    - model (torch.fx.GraphModule): The exported model to modify.

    Returns:
    - torch.fx.GraphModule: The modified model in eval mode.
    """
    # 调用内部函数，将特定操作（如 dropout、batchnorm）从训练模式切换到评估模式
    _replace_dropout(model, train_to_eval=True)
    _replace_batchnorm(model, train_to_eval=True)
    return model


# 将导出的图模块切换到训练模式的辅助函数
def _move_exported_model_to_train(model: torch.fx.GraphModule):
    """
    Move an exported GraphModule to train mode.

    This function modifies the model to behave as if it's in training mode,
    but only for specific operations like dropout and batch normalization.

    Args:
    - model (torch.fx.GraphModule): The exported model to modify.

    Returns:
    - torch.fx.GraphModule: The modified model in train mode.
    """
    # 调用内部函数，将特定操作（如 dropout、batchnorm）从评估模式切换到训练模式
    _replace_dropout(model, train_to_eval=False)
    _replace_batchnorm(model, train_to_eval=False)
    return model


# 允许在导出模型上调用 `model.train()` 和 `model.eval()` 的辅助函数
def _allow_exported_model_train_eval(model: torch.fx.GraphModule):
    """
    Allow users to call `model.train()` and `model.eval()` on an exported model,
    but limit the effect to changing behavior for specific operations like dropout and batch normalization.

    Note: This approximation does not fully replicate the behavior of `model.train()` and `model.eval()`
    in eager models. Code that depends on the `training` flag may not function correctly due to
    specialization at export time. Operations beyond dropout and batch normalization with different
    train/eval behaviors will also not be converted properly.

    Args:
    - model (torch.fx.GraphModule): The exported model to modify.

    Returns:
    - torch.fx.GraphModule: The modified model with `train` and `eval` methods added.
    """
    
    # 定义内部 `_train` 函数，通过修改 `training` 模式来调用 `_move_exported_model_to_train` 或 `_move_exported_model_to_eval`
    def _train(self, mode: bool = True):
        if mode:
            _move_exported_model_to_train(self)
        else:
            _move_exported_model_to_eval(self)

    # 定义内部 `_eval` 函数，调用 `_move_exported_model_to_eval`
    def _eval(self):
        _move_exported_model_to_eval(self)

    # 将 `_train` 和 `_eval` 方法绑定到 `model` 上，使得可以调用 `model.train()` 和 `model.eval()`
    model.train = types.MethodType(_train, model)  # type: ignore[method-assign]
    model.eval = types.MethodType(_eval, model)  # type: ignore[method-assign]
    return model
```