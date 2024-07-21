# `.\pytorch\torch\nn\utils\memory_format.py`

```py
# 添加类型检查的声明，允许未类型化的函数定义
# 该模块用于深度学习框架 PyTorch
import torch

# 定义函数，用于将 nn.Conv2d.weight 的内存格式转换为指定的 memory_format
def convert_conv2d_weight_memory_format(module, memory_format):
    r"""Convert ``memory_format`` of ``nn.Conv2d.weight`` to ``memory_format``.

    The conversion recursively applies to nested ``nn.Module``, including ``module``.
    Note that it only changes the memory_format, but not the semantics of each dimensions.
    This function is used to facilitate the computation to adopt NHWC kernels, which
    provides considerable speed up for fp16 data on CUDA devices with compute capability >= 7.0

    .. note::
        调用 ``model.to(memory_format=torch.channels_last)`` 比 ``convert_conv2d_weight_memory_format`` 更激进。
        任何具有 4 维权重的层都会受到 ``model.to`` 的影响，但并不一定从指定的 ``memory_format`` 转换中受益。
        我们有信心的一个地方是，cuDNN 中卷积的 NHWC（channels_last）转换是有益的，
        即使在需要对输入张量进行置换的情况下。

        因此，我们的策略是仅将卷积的权重转换为 channels_last。这确保：
        1. 将使用快速卷积核，其优势可能超过置换的开销（如果输入格式不同）。
        2. 不会对不受 memory_format 转换益处的层应用不必要的置换。

        最佳情况是，卷积层之间的层都兼容 channels_last。输入张量遇到第一个卷积层时会被置换为 channels_last，并保持该内存格式。
        因此，后续的卷积操作将无需对输入张量进行置换。

        如果在卷积层之间存在不兼容 channels_last 的层，我们需要将输入张量置换回连续格式。
        输入张量将在剩余的层中以连续格式传递，并在遇到另一个卷积层时置换为 channels_last。没有必要将该置换传播到更早的层，因为大多数层对 ``memory_format`` 都不太敏感。

        当 PyTorch 支持置换融合时，这个说法可能会改变，因为可能有更好的位置来融合置换，而不是在卷积层之前立即执行。

    Args:
        module (nn.Module): ``nn.Conv2d`` 和 ``nn.ConvTranspose2d`` 或包含它们的容器 ``nn.Module``
        memory_format: 用户指定的 ``memory_format``，例如 ``torch.channels_last`` 或 ``torch.contiguous_format``

    Returns:
        The original module with updated ``nn.Conv2d``
    # TODO: 当 channels_last 支持超出仅限于 4 维张量时，将此扩展为 `_ConvNd`
    def convert_conv2d_weight_memory_format(module, memory_format):
        # 检查模块是否是 Conv2d 或 ConvTranspose2d 类型的实例
        if isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
            # 复制模块的权重数据并确保是连续的，使用指定的内存格式
            weight_data = (
                module.weight.detach().clone().contiguous(memory_format=memory_format)
            )
            # 调整模块的权重数据形状，使用指定的内存格式
            module.weight.data = weight_data.resize_(
                weight_data.size(), memory_format=memory_format
            )
        # 递归调用每个子模块，以便对整个模型的权重进行内存格式转换
        for child in module.children():
            convert_conv2d_weight_memory_format(child, memory_format)
        # 返回转换后的模块
        return module
# 定义函数 convert_conv3d_weight_memory_format，用于转换 nn.Conv3d.weight 的内存格式为指定的 memory_format
def convert_conv3d_weight_memory_format(module, memory_format):
    r"""Convert ``memory_format`` of ``nn.Conv3d.weight`` to ``memory_format``
    The conversion recursively applies to nested ``nn.Module``, including ``module``.
    Note that it only changes the memory_format, but not the semantics of each dimensions.
    This function is used to facilitate the computation to adopt NHWC kernels, which
    provides considerable speed up for fp16 data on CUDA devices with compute capability >= 7.0

    .. note::
        Calling ``model.to(memory_format=torch.channels_last)`` is more aggressive
        than the utility function ``convert_conv3d_weight_memory_format``. Any
        layer with 4d weight will be affected by ``model.to``, which does not
        necessarily benefit from conversion to specified ``memory_format``.
        One place we are confident in is that NHWC(channels_last) conversion for
        convolution in cuDNN, As it is beneficial to run convolution in NHWC,
        even in cases where we have to apply permutation to input tensors.

        Hence our strategy here is to convert only the weight of convolution to
        channels_last. This ensures that;
        1. Fast convolution kernels will be used, the benefit of which could
        outweigh overhead of permutation (if input is not in the same format)
        2. No unnecessary permutations are applied on layers that do not benefit
        from memory_format conversion.

        The optimal case is that, layers between convolution layers are channels
        last compatible. Input tensor would be permuted to channels last when it
        encounters the first convolution layer and stay in that memory format.
        Hence following convolutions will not need to permute its input tensor.

        In case where a channels last incompatible layer is between convolution
        layers, we need to permute the input tensor back to contiguous format
        for that layer. The input tensor will go through the remaining layers in
        contiguous format and be permuted to channels last when it encounters
        another convolution layer. There's no point in propagating that
        permutation to an earlier layer, as most layers are quite agnostic to
        ``memory_format``.

        This claim might change when PyTorch supports fusion of permutation, as
        there might have been a better spot to fuse the permutation other than
        immediately before a convolution.

    Args:
        module (nn.Module): ``nn.Conv3d`` & ``nn.ConvTranspose3d`` or container
                            ``nn.Module``
        memory_format: user specified ``memory_format``,
            e.g. ``torch.channels_last`` or ``torch.contiguous_format``

    Returns:
        The original module with updated ``nn.Conv3d``
    # 如果 channels_last 支持被扩展到非4D张量时，将此TODO扩展为 `_ConvNd`
    if isinstance(module, (torch.nn.Conv3d, torch.nn.ConvTranspose3d)):
        # 获取模块的权重数据，进行分离、克隆，并保证数据连续性和内存格式
        weight_data = (
            module.weight.detach().clone().contiguous(memory_format=memory_format)
        )
        # 调整模块的权重数据的大小和内存格式
        module.weight.data = weight_data.resize_(
            weight_data.size(), memory_format=memory_format
        )
    
    # 遍历模块的子模块，并对每个子模块递归调用 `convert_conv3d_weight_memory_format`
    for child in module.children():
        convert_conv3d_weight_memory_format(child, memory_format)
    
    # 返回已更新的模块
    return module
```