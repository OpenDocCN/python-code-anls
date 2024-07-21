# `.\pytorch\torch\__future__.py`

```
# 是否在转换 nn.Module 时覆盖模块参数，默认为 False
_overwrite_module_params_on_conversion: bool = False

# 是否在转换 nn.Module 时交换模块参数，默认为 False
_swap_module_params_on_conversion: bool = False


def set_overwrite_module_params_on_conversion(value: bool) -> None:
    """
    设置在转换 nn.Module 时是否将新张量分配给参数，而不是原地修改现有参数。

    当启用时，以下方法将会为模块分配新的参数：

    1. ``module.{device}()``（例如：:meth:`nn.Module.cuda()`）用于在设备之间移动模块
    2. ``module.{dtype}()``（例如：:meth:`nn.Module.float()`）用于将模块转换为不同的数据类型
    3. :meth:`nn.Module.to`
    4. :meth:`nn.Module.to_empty`

    Args:
        value (bool): 是否分配新张量。

    """
    global _overwrite_module_params_on_conversion
    _overwrite_module_params_on_conversion = value


def get_overwrite_module_params_on_conversion() -> bool:
    """
    返回在转换 :class:`torch.nn.Module` 时是否将新张量分配给参数，而不是原地修改现有参数。默认为 ``False``。

    有关更多信息，请参阅 :func:`~torch.__future__.set_overwrite_module_params_on_conversion`。
    """
    return _overwrite_module_params_on_conversion


def set_swap_module_params_on_conversion(value: bool) -> None:
    """
    设置是否使用 :func:`~torch.utils.swap_tensors` 而不是将 .data 设置为修改现有参数时在转换 ``nn.Module`` 时使用，
    并在将状态字典加载到 ``nn.Module`` 时使用 ``param.copy_(state_dict[key])``。

    .. 注意::
        此函数优先于 :func:`~torch.__future__.get_overwrite_module_params_on_conversion`

    当启用时，以下方法将会原地交换现有参数：

    1. ``module.{device}()``（例如：:meth:`nn.Module.cuda()`）用于在设备之间移动模块
    2. ``module.{dtype}()``（例如：:meth:`nn.Module.float()`）用于将模块转换为不同的数据类型
    3. :meth:`nn.Module.to`
    4. :meth:`nn.Module.to_empty`
    5. :meth:`nn.Module.load_state_dict`

    当设置此选项时，:meth:`~nn.Module.load_state_dict` 的语义如下：

    1. 对于每个参数/缓冲区，其对应的 ``state_dict['key']`` 将通过 :meth:`~torch.Tensor.module_load` 进行转换
       （即 ``res = param.module_load(state_dict['key'])``）
    2. 如果需要，``res`` 将被包装在 :class:`~nn.Parameter` 中
    3. 使用 :func:`~torch.utils.swap_tensors` 将模块中的参数/缓冲区与 ``res`` 原地交换

    Args:
        value (bool): 是否使用 :func:`~torch.utils.swap_tensors`。

    """
    global _swap_module_params_on_conversion
    _swap_module_params_on_conversion = value


def get_swap_module_params_on_conversion() -> bool:
    """
    返回是否使用 :func:`~torch.utils.swap_tensors` 而不是将 .data 设置为

    """
    return _swap_module_params_on_conversion
    # 返回一个函数对象 _swap_module_params_on_conversion，该函数用于在转换 nn.Module 时是否原地修改现有参数，默认为 False
    # 查看 torch.__future__.set_swap_module_params_on_conversion 函数获取更多信息。
    """
    返回 _swap_module_params_on_conversion 函数对象。
    """
    return _swap_module_params_on_conversion
```