# `.\pytorch\torch\distributed\_shard\sharded_tensor\_ops\init.py`

```
# mypy: allow-untyped-defs
# 引入 PyTorch 库
import torch
# 引入分布式张量的相关模块
import torch.distributed._shard.sharded_tensor as sharded_tensor
from torch.distributed._shard.sharded_tensor import _sharded_op_impl


# 参数验证函数，确保参数不为 None
def validate_param(param, param_name):
    if param is None:
        raise ValueError(f"param: {param_name} shouldn't be None!")


# 装饰器函数，实现对 uniform_ 函数的扩展
@_sharded_op_impl(torch.nn.init.uniform_)
def uniform_(types, args=(), kwargs=None, pg=None):
    r"""
    Fills the Tensor in tensor.local_shards with values drawn from the uniform
    distribution :math:`\mathcal{U}(a, b)`.
    Args:
        tensor: tensor sharded across devices
        a: the lower bound of the uniform distribution
        b: the upper bound of the uniform distribution
    """
    validate_param(kwargs, "kwargs")
    # 获取关键字参数中的 tensor，并验证其不为 None
    sharded_tensor = kwargs["tensor"]
    validate_param(sharded_tensor, "tensor")
    # 获取均匀分布的上下界参数 a 和 b，并验证它们不为 None
    a = kwargs["a"]
    validate_param(a, "a")
    b = kwargs["b"]
    validate_param(b, "b")

    # 对每个本地分片中的张量执行均匀分布的初始化
    for shard in sharded_tensor.local_shards():
        torch.nn.init.uniform_(shard.tensor, a=a, b=b)
    # 返回分片张量对象
    return sharded_tensor


# 装饰器函数，实现对 normal_ 函数的扩展
@_sharded_op_impl(torch.nn.init.normal_)
def normal_(types, args=(), kwargs=None, pg=None):
    r"""
    Fills the Tensors in tensor.local_shards with values drawn from the normal
    distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`.
    Args:
        tensor: tensor sharded across devices
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
    """
    validate_param(kwargs, "kwargs")
    # 获取关键字参数中的 tensor，并验证其不为 None
    sharded_tensor = kwargs["tensor"]
    validate_param(sharded_tensor, "tensor")
    # 获取正态分布的均值 mean 和标准差 std 参数，并验证它们不为 None
    mean = kwargs["mean"]
    validate_param(mean, "mean")
    std = kwargs["std"]
    validate_param(std, "std")

    # 对每个本地分片中的张量执行正态分布的初始化
    for shard in sharded_tensor.local_shards():
        torch.nn.init.normal_(shard.tensor, mean=mean, std=std)
    # 返回分片张量对象
    return sharded_tensor


# 装饰器函数，实现对 kaiming_uniform_ 函数的扩展
@_sharded_op_impl(torch.nn.init.kaiming_uniform_)
def kaiming_uniform_(types, args=(), kwargs=None, pg=None):
    r"""
    Fills the Tensors in tensor.local_shards with values according to the method
    described in `Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification` - He, K. et al. (2015), using a
    uniform distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{U}(-\text{bound}, \text{bound})` where
    .. math::
        \text{bound} = \text{gain} \times \sqrt{\frac{3}{\text{fan\_mode}}}
    Also known as He initialization.
    """
    # 验证关键字参数不为 None
    validate_param(kwargs, "kwargs")
    # 获取关键字参数中的 tensor，并验证其不为 None
    sharded_tensor = kwargs["tensor"]
    validate_param(sharded_tensor, "tensor")
    # 未对 gain 和 fan_mode 参数进行验证，可能是由于未在代码中使用这些参数

    # 对每个本地分片中的张量执行 He 初始化的均匀分布
    for shard in sharded_tensor.local_shards():
        torch.nn.init.kaiming_uniform_(shard.tensor)
    # 返回分片张量对象
    return sharded_tensor
    def init_sharded_tensor(**kwargs):
        """
        Initialize a sharded tensor across devices using Kaiming uniform initialization.
    
        Args:
            tensor: tensor sharded across devices
            a: the negative slope of the rectifier used after this layer (only
                used with ``'leaky_relu'``)
            mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing ``'fan_in'``
                preserves the magnitude of the variance of the weights in the
                forward pass. Choosing ``'fan_out'`` preserves the magnitudes in the
                backwards pass.
            nonlinearity: the non-linear function (`nn.functional` name),
                recommended to use only with ``'relu'`` or ``'leaky_relu'`` (default).
        """
        # Validate the input parameters using a helper function
        validate_param(kwargs, "kwargs")
        
        # Extract tensor from kwargs and validate it
        sharded_tensor = kwargs["tensor"]
        validate_param(sharded_tensor, "tensor")
        
        # Extract 'a' from kwargs and validate it
        a = kwargs["a"]
        validate_param(a, "a")
        
        # Extract 'mode' from kwargs and validate it
        mode = kwargs["mode"]
        validate_param(mode, "mode")
        
        # Extract 'nonlinearity' from kwargs and validate it
        nonlinearity = kwargs["nonlinearity"]
        validate_param(nonlinearity, "nonlinearity")
        
        # Initialize each shard of the sharded tensor using Kaiming uniform initialization
        for shard in sharded_tensor.local_shards():
            torch.nn.init.kaiming_uniform_(
                shard.tensor, a=a, mode=mode, nonlinearity=nonlinearity
            )
        
        # Return the initialized sharded tensor
        return sharded_tensor
# 定义装饰器函数，用于实现被装饰函数 `_sharded_op_impl`
@_sharded_op_impl(torch.nn.init.constant_)
def constant_(types, args=(), kwargs=None, pg=None):
    """
    使用常数填充输入的 ShardedTensor。
    Args:
        types: 类型参数，可能用于类型验证
        args: 位置参数，通常包含 ShardedTensor 和填充值
        kwargs: 关键字参数，应包含 'tensor' 和 'val'
        pg: 参数组，可能用于分布式训练中的参数组
    """
    # 验证关键字参数 kwargs 是否有效
    validate_param(kwargs, "kwargs")
    # 获取关键字参数中的 'tensor'
    sharded_tensor = kwargs["tensor"]
    # 验证 'tensor' 是否有效
    validate_param(sharded_tensor, "tensor")
    # 获取关键字参数中的 'val'
    val = kwargs["val"]
    # 验证 'val' 是否有效
    validate_param(val, "val")
    
    # 遍历 ShardedTensor 中的本地分片
    for shard in sharded_tensor.local_shards():
        # 使用 torch.nn.init.constant_ 将分片中的张量填充为指定的常数值
        torch.nn.init.constant_(shard.tensor, val=val)
    
    # 返回填充后的 ShardedTensor
    return sharded_tensor


# 定义字典，将 torch 的 tensor 创建操作映射到 sharded_tensor 模块中的对应方法
tensor_like_creation_op_map = {
    torch.full_like: sharded_tensor.full,
    torch.empty_like: sharded_tensor.empty,
    torch.zeros_like: sharded_tensor.zeros,
    torch.ones_like: sharded_tensor.ones,
    torch.rand_like: sharded_tensor.rand,
    torch.randn_like: sharded_tensor.randn,
}


# 注册与默认张量行为相同的张量创建操作
def register_tensor_creation_op(op):
    @_sharded_op_impl(op)
    def tensor_creation_op(types, args=(), kwargs=None, pg=None):
        """
        处理 ``__torch_function__`` 分发，用于接受 ShardedTensor 作为参数的张量创建操作，
        如 ``torch.zeros_like`` 或 ``torch.full_like``。
        """
        # 获取对应的 sharded_tensor 模块中的创建操作
        creation_op = tensor_like_creation_op_map.get(op, None)
        if creation_op is None:
            raise RuntimeError(f"Tensor creation {op} not supported!")
        # 如果 kwargs 为空，则初始化为空字典
        if kwargs is None:
            kwargs = {}
        
        # 获取位置参数中的第一个参数，通常为 ShardedTensor
        st = args[0]
        
        # 调用相应的创建操作函数，使用 ShardedTensor 的分片规范和大小，以及其余参数和 kwargs
        new_st = creation_op(st.sharding_spec(), st.size(), *args[1:], **kwargs)  # type: ignore[operator]
        
        # 返回新创建的 ShardedTensor
        return new_st


# 注册多个张量创建操作到 tensor_creation_op 函数
register_tensor_creation_op(torch.full_like)
register_tensor_creation_op(torch.empty_like)
register_tensor_creation_op(torch.zeros_like)
register_tensor_creation_op(torch.ones_like)
register_tensor_creation_op(torch.rand_like)
register_tensor_creation_op(torch.randn_like)
```