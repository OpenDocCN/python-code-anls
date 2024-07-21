# `.\pytorch\torch\nn\utils\prune.py`

```py
# mypy: allow-untyped-defs
r"""Pruning methods."""
import numbers  # 导入用于数字处理的模块
from abc import ABC, abstractmethod  # 导入用于抽象基类和抽象方法的模块
from collections.abc import Iterable  # 导入用于可迭代对象的抽象基类
from typing import Tuple  # 导入用于类型提示的元组类型

import torch  # 导入 PyTorch 库


class BasePruningMethod(ABC):
    r"""Abstract base class for creation of new pruning techniques.

    Provides a skeleton for customization requiring the overriding of methods
    such as :meth:`compute_mask` and :meth:`apply`.
    """

    _tensor_name: str  # 类属性，用于存储张量的名称字符串

    def __call__(self, module, inputs):
        r"""Multiply the mask into original tensor and store the result.

        Multiplies the mask (stored in ``module[name + '_mask']``)
        into the original tensor (stored in ``module[name + '_orig']``)
        and stores the result into ``module[name]`` by using :meth:`apply_mask`.

        Args:
            module (nn.Module): module containing the tensor to prune
            inputs: not used.
        """
        setattr(module, self._tensor_name, self.apply_mask(module))

    @abstractmethod
    def compute_mask(self, t, default_mask):
        r"""Compute and returns a mask for the input tensor ``t``.

        Starting from a base ``default_mask`` (which should be a mask of ones
        if the tensor has not been pruned yet), generate a random mask to
        apply on top of the ``default_mask`` according to the specific pruning
        method recipe.

        Args:
            t (torch.Tensor): tensor representing the importance scores of the
            parameter to prune.
            default_mask (torch.Tensor): Base mask from previous pruning
            iterations, that need to be respected after the new mask is
            applied. Same dims as ``t``.

        Returns:
            mask (torch.Tensor): mask to apply to ``t``, of same dims as ``t``
        """
        pass

    def apply_mask(self, module):
        r"""Simply handles the multiplication between the parameter being pruned and the generated mask.

        Fetches the mask and the original tensor from the module
        and returns the pruned version of the tensor.

        Args:
            module (nn.Module): module containing the tensor to prune

        Returns:
            pruned_tensor (torch.Tensor): pruned version of the input tensor
        """
        # to carry out the multiplication, the mask needs to have been computed,
        # so the pruning method must know what tensor it's operating on
        assert (
            self._tensor_name is not None
        ), f"Module {module} has to be pruned"  # this gets set in apply()
        mask = getattr(module, self._tensor_name + "_mask")  # 获取存储在模块中的掩码
        orig = getattr(module, self._tensor_name + "_orig")  # 获取存储在模块中的原始张量
        pruned_tensor = mask.to(dtype=orig.dtype) * orig  # 应用掩码到原始张量上并返回修剪后的张量
        return pruned_tensor

    @classmethod
    def prune(self, t, default_mask=None, importance_scores=None):
        r"""Compute and returns a pruned version of input tensor ``t``.

        According to the pruning rule specified in :meth:`compute_mask`.

        Args:
            t (torch.Tensor): tensor to prune (of same dimensions as
                ``default_mask``).
            importance_scores (torch.Tensor): tensor of importance scores (of
                same shape as ``t``) used to compute mask for pruning ``t``.
                The values in this tensor indicate the importance of the
                corresponding elements in the ``t`` that is being pruned.
                If unspecified or None, the tensor ``t`` will be used in its place.
            default_mask (torch.Tensor, optional): mask from previous pruning
                iteration, if any. To be considered when determining what
                portion of the tensor that pruning should act on. If None,
                default to a mask of ones.

        Returns:
            pruned version of tensor ``t``.
        """
        # 如果指定了 importance_scores，则确保其形状与 t 相同
        if importance_scores is not None:
            assert (
                importance_scores.shape == t.shape
            ), "importance_scores should have the same shape as tensor t"
        else:
            # 如果未指定 importance_scores，则将 t 自身作为重要性分数
            importance_scores = t
        # 如果未指定 default_mask，则创建一个全为 1 的与 t 相同形状的张量作为默认掩码
        default_mask = default_mask if default_mask is not None else torch.ones_like(t)
        # 返回经过掩码处理的 t，使用 compute_mask 方法生成掩码
        return t * self.compute_mask(importance_scores, default_mask=default_mask)

    def remove(self, module):
        r"""Remove the pruning reparameterization from a module.

        The pruned parameter named ``name`` remains permanently pruned,
        and the parameter named ``name+'_orig'`` is removed from the parameter list.
        Similarly, the buffer named ``name+'_mask'`` is removed from the buffers.

        Note:
            Pruning itself is NOT undone or reversed!
        """
        # 在从模块中移除修剪之前，必须先应用修剪
        assert (
            self._tensor_name is not None
        ), f"Module {module} has to be pruned before pruning can be removed"  # 这个在 apply() 中被设置

        # 获取应用了掩码的权重
        weight = self.apply_mask(module)  # 被掩码处理的权重

        # 删除和重置操作
        if hasattr(module, self._tensor_name):
            delattr(module, self._tensor_name)
        # 将原始权重数据更新为最新的训练后权重
        orig = module._parameters[self._tensor_name + "_orig"]
        orig.data = weight.data
        del module._parameters[self._tensor_name + "_orig"]
        # 删除模块的缓冲区中的掩码
        del module._buffers[self._tensor_name + "_mask"]
        # 将更新后的原始权重重新设置到模块中
        setattr(module, self._tensor_name, orig)
class PruningContainer(BasePruningMethod):
    """Container holding a sequence of pruning methods for iterative pruning.

    Keeps track of the order in which pruning methods are applied and handles
    combining successive pruning calls.

    Accepts as argument an instance of a BasePruningMethod or an iterable of
    them.
    """

    def __init__(self, *args):
        # Initialize an empty tuple to hold pruning methods
        self._pruning_methods: Tuple[BasePruningMethod, ...] = tuple()
        # If only one argument is provided and it's not iterable, treat it as a single method
        if not isinstance(args, Iterable):  # only 1 item
            # Extract the tensor name from the single method
            self._tensor_name = args._tensor_name
            # Add the single method to the container
            self.add_pruning_method(args)
        # If only one item in a tuple, treat it as a single method
        elif len(args) == 1:  # only 1 item in a tuple
            # Extract the tensor name from the method in the tuple
            self._tensor_name = args[0]._tensor_name
            # Add the single method to the container
            self.add_pruning_method(args[0])
        else:  # manual construction from list or other iterable (or no args)
            # Iterate over each method in the arguments and add them to the container
            for method in args:
                self.add_pruning_method(method)

    def add_pruning_method(self, method):
        r"""Add a child pruning ``method`` to the container.

        Args:
            method (subclass of BasePruningMethod): child pruning method
                to be added to the container.
        """
        # Check if the provided method is an instance of BasePruningMethod or None
        if not isinstance(method, BasePruningMethod) and method is not None:
            # Raise an error if the method is not a subclass of BasePruningMethod
            raise TypeError(f"{type(method)} is not a BasePruningMethod subclass")
        # Check if the tensor name of the method matches the container's tensor name
        elif method is not None and self._tensor_name != method._tensor_name:
            # Raise a ValueError if the tensor names do not match
            raise ValueError(
                "Can only add pruning methods acting on "
                f"the parameter named '{self._tensor_name}' to PruningContainer {self}."
                + f" Found '{method._tensor_name}'"
            )
        # If all checks passed, add the method to the _pruning_methods tuple
        self._pruning_methods += (method,)  # type: ignore[operator]

    def __len__(self):
        # Return the number of pruning methods in the container
        return len(self._pruning_methods)

    def __iter__(self):
        # Return an iterator over the pruning methods in the container
        return iter(self._pruning_methods)

    def __getitem__(self, idx):
        # Return a pruning method from the container by index
        return self._pruning_methods[idx]


class Identity(BasePruningMethod):
    r"""Utility pruning method that does not prune any units but generates the pruning parametrization with a mask of ones."""

    PRUNING_TYPE = "unstructured"

    def compute_mask(self, t, default_mask):
        # Simply return the default mask unchanged
        mask = default_mask
        return mask

    @classmethod
    def apply(cls, module, name):
        r"""Add pruning on the fly and reparametrization of a tensor.

        Adds the forward pre-hook that enables pruning on the fly and
        the reparametrization of a tensor in terms of the original tensor
        and the pruning mask.

        Args:
            module (nn.Module): module containing the tensor to prune
            name (str): parameter name within ``module`` on which pruning
                will act.
        """
        # Call the apply method of the superclass BasePruningMethod
        return super().apply(module, name)


class RandomUnstructured(BasePruningMethod):
    r"""Prune (currently unpruned) units in a tensor at random.
    Args:
        name (str): 模块内的参数名称，将在其中进行剪枝操作。
        amount (int or float): 要剪枝的参数数量。
            如果是 ``float``，应该在 0.0 到 1.0 之间，并表示要剪掉的参数比例。
            如果是 ``int``，则表示要剪掉的绝对参数数量。
    """

    PRUNING_TYPE = "unstructured"

    def __init__(self, amount):
        # 验证剪枝量的有效范围
        _validate_pruning_amount_init(amount)
        self.amount = amount

    def compute_mask(self, t, default_mask):
        # 检查要剪掉的单元数是否不大于张量 t 中的参数数目
        tensor_size = t.nelement()
        # 计算要剪掉的单元数：如果是 int，则为 amount；如果是 float，则为 amount * tensor_size
        nparams_toprune = _compute_nparams_toprune(self.amount, tensor_size)
        # 如果要剪掉的单元数大于张量中的单元数，则会引发错误
        _validate_pruning_amount(nparams_toprune, tensor_size)

        mask = default_mask.clone(memory_format=torch.contiguous_format)

        if nparams_toprune != 0:  # torch.kthvalue 不支持 k=0
            prob = torch.rand_like(t)
            topk = torch.topk(prob.view(-1), k=nparams_toprune)
            mask.view(-1)[topk.indices] = 0

        return mask

    @classmethod
    def apply(cls, module, name, amount):
        r"""在运行时添加剪枝，并重新参数化张量。

        添加前向预钩子，使得能够在运行时进行剪枝，
        并在原始张量和剪枝掩码的基础上重新参数化张量。

        Args:
            module (nn.Module): 包含要进行剪枝的张量的模块
            name (str): ``module`` 内的参数名称，将在其中进行剪枝操作。
            amount (int or float): 要剪枝的参数数量。
                如果是 ``float``，应该在 0.0 到 1.0 之间，并表示要剪掉的参数比例。
                如果是 ``int``，则表示要剪掉的绝对参数数量。
        """
        return super().apply(module, name, amount=amount)
class L1Unstructured(BasePruningMethod):
    r"""Prune (currently unpruned) units in a tensor by zeroing out the ones with the lowest L1-norm.

    Args:
        amount (int or float): quantity of parameters to prune.
            If ``float``, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If ``int``, it represents the
            absolute number of parameters to prune.
    """

    PRUNING_TYPE = "unstructured"

    def __init__(self, amount):
        # Check range of validity of pruning amount
        _validate_pruning_amount_init(amount)
        self.amount = amount

    def compute_mask(self, t, default_mask):
        # Check that the amount of units to prune is not > than the number of
        # parameters in t
        tensor_size = t.nelement()
        # Compute number of units to prune: amount if int,
        # else amount * tensor_size
        nparams_toprune = _compute_nparams_toprune(self.amount, tensor_size)
        # This should raise an error if the number of units to prune is larger
        # than the number of units in the tensor
        _validate_pruning_amount(nparams_toprune, tensor_size)

        mask = default_mask.clone(memory_format=torch.contiguous_format)

        if nparams_toprune != 0:  # k=0 not supported by torch.kthvalue
            # largest=True --> top k; largest=False --> bottom k
            # Prune the smallest k
            topk = torch.topk(torch.abs(t).view(-1), k=nparams_toprune, largest=False)
            # topk will have .indices and .values
            mask.view(-1)[topk.indices] = 0

        return mask

    @classmethod
    def apply(cls, module, name, amount, importance_scores=None):
        r"""Add pruning on the fly and reparametrization of a tensor.

        Adds the forward pre-hook that enables pruning on the fly and
        the reparametrization of a tensor in terms of the original tensor
        and the pruning mask.

        Args:
            module (nn.Module): module containing the tensor to prune
            name (str): parameter name within ``module`` on which pruning
                will act.
            amount (int or float): quantity of parameters to prune.
                If ``float``, should be between 0.0 and 1.0 and represent the
                fraction of parameters to prune. If ``int``, it represents the
                absolute number of parameters to prune.
            importance_scores (torch.Tensor): tensor of importance scores (of same
                shape as module parameter) used to compute mask for pruning.
                The values in this tensor indicate the importance of the corresponding
                elements in the parameter being pruned.
                If unspecified or None, the module parameter will be used in its place.
        """
        # Delegate to parent class's apply method
        return super().apply(
            module, name, amount=amount, importance_scores=importance_scores
        )


class RandomStructured(BasePruningMethod):
    """
    Prune entire (currently unpruned) channels in a tensor at random.
    
    Args:
        amount (int or float): quantity of parameters to prune.
            If ``float``, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If ``int``, it represents the
            absolute number of parameters to prune.
        dim (int, optional): index of the dim along which we define
            channels to prune. Default: -1.
    """
    
    # 定义剪枝类型为结构化剪枝
    PRUNING_TYPE = "structured"
    
    class Pruner:
        def __init__(self, amount, dim=-1):
            # 检查剪枝量的有效范围
            _validate_pruning_amount_init(amount)
            # 设置剪枝量和剪枝维度
            self.amount = amount
            self.dim = dim
    def compute_mask(self, t, default_mask):
        r"""Compute and returns a mask for the input tensor ``t``.

        Starting from a base ``default_mask`` (which should be a mask of ones
        if the tensor has not been pruned yet), generate a random mask to
        apply on top of the ``default_mask`` by randomly zeroing out channels
        along the specified dim of the tensor.

        Args:
            t (torch.Tensor): tensor representing the parameter to prune
            default_mask (torch.Tensor): Base mask from previous pruning
                iterations, that need to be respected after the new mask is
                applied. Same dims as ``t``.

        Returns:
            mask (torch.Tensor): mask to apply to ``t``, of same dims as ``t``

        Raises:
            IndexError: if ``self.dim >= len(t.shape)``
        """
        # 检查张量是否有结构（即至少有1个维度），以便“通道”的概念有意义
        _validate_structured_pruning(t)

        # 检查 self.dim 是否是有效的索引维度，否则引发 IndexError
        _validate_pruning_dim(t, self.dim)

        # 检查要修剪的通道数是否不超过沿着要修剪的维度的张量 t 中的通道数
        tensor_size = t.shape[self.dim]
        # 计算要修剪的单元数：如果是整数，则为 amount；否则为 amount * tensor_size
        nparams_toprune = _compute_nparams_toprune(self.amount, tensor_size)
        # 如果要修剪的单元数大于张量中的单元数，则引发错误
        _validate_pruning_amount(nparams_toprune, tensor_size)

        # 通过初始化为全0然后根据 topk.indices 填充1来计算二进制掩码，沿着 self.dim
        # mask 具有与张量 t 相同的形状
        def make_mask(t, dim, nchannels, nchannels_toprune):
            # 为每个通道生成一个在 [0, 1] 范围内的随机数
            prob = torch.rand(nchannels)
            # 通过将概率值中最小的 k = nchannels_toprune 个设为0来生成每个通道的掩码
            threshold = torch.kthvalue(prob, k=nchannels_toprune).values
            channel_mask = prob > threshold

            mask = torch.zeros_like(t)
            slc = [slice(None)] * len(t.shape)
            slc[dim] = channel_mask
            mask[slc] = 1
            return mask

        if nparams_toprune == 0:  # k=0 不被 torch.kthvalue 支持
            mask = default_mask
        else:
            # 将新的结构化掩码应用于之前（可能是非结构化的）掩码之上
            mask = make_mask(t, self.dim, tensor_size, nparams_toprune)
            mask *= default_mask.to(dtype=mask.dtype)
        return mask
    # 定义一个类方法 `apply`，用于在张量上应用动态剪枝和重新参数化操作

    r"""Add pruning on the fly and reparametrization of a tensor.
    
    在张量上添加动态剪枝和重新参数化的前向预处理钩子。
    """

    # 调用父类的 `apply` 方法，将剪枝和重新参数化应用到指定模块的指定参数上
    return super().apply(module, name, amount=amount, dim=dim)
class LnStructured(BasePruningMethod):
    r"""Prune entire (currently unpruned) channels in a tensor based on their L\ ``n``-norm.

    Args:
        amount (int or float): quantity of channels to prune.
            If ``float``, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If ``int``, it represents the
            absolute number of parameters to prune.
        n (int, float, inf, -inf, 'fro', 'nuc'): See documentation of valid
            entries for argument ``p`` in :func:`torch.norm`.
        dim (int, optional): index of the dim along which we define
            channels to prune. Default: -1.
    """

    PRUNING_TYPE = "structured"

    def __init__(self, amount, n, dim=-1):
        # Check range of validity of amount
        _validate_pruning_amount_init(amount)
        self.amount = amount
        self.n = n
        self.dim = dim

    @classmethod
    def apply(cls, module, name, amount, n, dim, importance_scores=None):
        r"""Add pruning on the fly and reparametrization of a tensor.

        Adds the forward pre-hook that enables pruning on the fly and
        the reparametrization of a tensor in terms of the original tensor
        and the pruning mask.

        Args:
            module (nn.Module): module containing the tensor to prune
            name (str): parameter name within ``module`` on which pruning
                will act.
            amount (int or float): quantity of parameters to prune.
                If ``float``, should be between 0.0 and 1.0 and represent the
                fraction of parameters to prune. If ``int``, it represents the
                absolute number of parameters to prune.
            n (int, float, inf, -inf, 'fro', 'nuc'): See documentation of valid
                entries for argument ``p`` in :func:`torch.norm`.
            dim (int): index of the dim along which we define channels to
                prune.
            importance_scores (torch.Tensor): tensor of importance scores (of same
                shape as module parameter) used to compute mask for pruning.
                The values in this tensor indicate the importance of the corresponding
                elements in the parameter being pruned.
                If unspecified or None, the module parameter will be used in its place.
        """
        # Call the superclass's apply method with specified parameters
        return super().apply(
            module,
            name,
            amount=amount,
            n=n,
            dim=dim,
            importance_scores=importance_scores,
        )


class CustomFromMask(BasePruningMethod):
    PRUNING_TYPE = "global"

    def __init__(self, mask):
        self.mask = mask

    def compute_mask(self, t, default_mask):
        assert default_mask.shape == self.mask.shape
        # Element-wise multiplication of default mask and custom mask
        mask = default_mask * self.mask.to(dtype=default_mask.dtype)
        return mask

    @classmethod
    # 定义一个类方法 `apply`，用于应用剪枝和重新参数化张量
    def apply(cls, module, name, mask):
        # 添加前向预钩子，实现动态剪枝和张量的重新参数化
        # Args:
        #   module (nn.Module): 包含需要剪枝的张量的模块
        #   name (str): 在 `module` 中的参数名称，剪枝将作用在该参数上
        # 返回调用父类 `apply` 方法的结果，传入模块、参数名称和剪枝掩码
        return super().apply(module, name, mask=mask)
# 应用修剪重参数化，但不修剪任何单元
# 对模块中名为“name”的参数应用修剪重参数化，但实际上不修剪任何单元。
# 在原地修改模块（并返回修改后的模块）：
# 1) 添加名为“name+'_mask'”的命名缓冲区，对应于修剪方法应用于参数“name”的二进制掩码。
# 2) 用其修剪版本替换参数“name”，同时原始（未修剪）参数存储在名为“name+'_orig'”的新参数中。
def identity(module, name):
    Identity.apply(module, name)
    return module

# 通过随机移除（当前未修剪的）单元修剪张量
# 通过随机移除选择的指定“amount”数量（当前未修剪的）单元，修剪名为“name”的参数对应的张量。
# 在原地修改模块（并返回修改后的模块）：
# 1) 添加名为“name+'_mask'”的命名缓冲区，对应于修剪方法应用于参数“name”的二进制掩码。
# 2) 用其修剪版本替换参数“name”，同时原始（未修剪）参数存储在名为“name+'_orig'”的新参数中。
# 参数：
# - module (nn.Module): 包含要修剪的张量的模块
# - name (str): 在“module”中的参数名称，修剪将作用于此处
# - amount (int or float): 要修剪的参数量。如果是“float”，应在0.0和1.0之间，表示要修剪的参数比例。如果是“int”，表示要修剪的绝对参数数目。
def random_unstructured(module, name, amount):
    RandomUnstructured.apply(module, name, amount)
    return module

# 通过删除具有最低L1范数的单元来修剪张量
# 通过删除具有最低L1范数的指定“amount”数量（当前未修剪的）单元，修剪名为“name”的参数对应的张量。
# 在原地修改模块（并返回修改后的模块）：
# 1) 添加名为“name+'_mask'”的命名缓冲区，对应于修剪方法应用于参数“name”的二进制掩码。
# 2) 用其修剪版本替换参数“name”，同时原始（未修剪）参数存储在名为“name+'_orig'”的新参数中。
# 参数：
# - module (nn.Module): 包含要修剪的张量的模块
# - name (str): 在“module”中的参数名称，修剪将作用于此处
# - amount (int or float): 要修剪的参数量。如果是“float”，应在0.0和1.0之间，表示要修剪的参数比例。如果是“int”，表示要修剪的绝对参数数目。
# - importance_scores (tensor): 可选参数，指定了单位的重要性分数，以便按L1范数进行修剪选择。
def l1_unstructured(module, name, amount, importance_scores=None):
    Modifies module in place (and also return the modified module)
    by:

    1) adding a named buffer called ``name+'_mask'`` corresponding to the
       binary mask applied to the parameter ``name`` by the pruning method.
    2) replacing the parameter ``name`` by its pruned version, while the
       original (unpruned) parameter is stored in a new parameter named
       ``name+'_orig'``.

    Args:
        module (nn.Module): module containing the tensor to prune
        name (str): parameter name within ``module`` on which pruning
                will act.
        amount (int or float): quantity of parameters to prune.
            If ``float``, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If ``int``, it represents the
            absolute number of parameters to prune.
        importance_scores (torch.Tensor): tensor of importance scores (of same
            shape as module parameter) used to compute mask for pruning.
            The values in this tensor indicate the importance of the corresponding
            elements in the parameter being pruned.
            If unspecified or None, the module parameter will be used in its place.

    Returns:
        module (nn.Module): modified (i.e. pruned) version of the input module

    Examples:
        >>> # xdoctest: +SKIP
        >>> m = prune.l1_unstructured(nn.Linear(2, 3), 'weight', amount=0.2)
        >>> m.state_dict().keys()
        odict_keys(['bias', 'weight_orig', 'weight_mask'])
    """
    # 调用 L1Unstructured 类的 apply 方法，对指定模块的指定参数进行 L1 剪枝
    L1Unstructured.apply(
        module, name, amount=amount, importance_scores=importance_scores
    )
    # 返回被修改后的模块
    return module
# 使用随机结构化方法对指定模块的参数进行剪枝，剪除沿指定维度随机选择的指定数量的通道
def random_structured(module, name, amount, dim):
    # 调用 RandomStructured 类的 apply 方法，对模块中名为 name 的参数进行剪枝操作
    RandomStructured.apply(module, name, amount, dim)
    # 返回修改后的模块
    return module


# 使用最低 L_n 范数方法对指定模块的参数进行剪枝，剪除沿指定维度具有最低 L_n 范数的指定数量的通道
def ln_structured(module, name, amount, n, dim, importance_scores=None):
    # 待补充


在 ln_structured 函数中，注释需要补充完整，按照同样的格式进行解释。
    def prune_ln_structured(module, name, amount, n, dim, importance_scores=None):
        """
        Apply structured L_n norm based pruning to a specified parameter of a module.
    
        Args:
            module (nn.Module): module containing the tensor to prune
            name (str): parameter name within ``module`` on which pruning
                    will act.
            amount (int or float): quantity of parameters to prune.
                If ``float``, should be between 0.0 and 1.0 and represent the
                fraction of parameters to prune. If ``int``, it represents the
                absolute number of parameters to prune.
            n (int, float, inf, -inf, 'fro', 'nuc'): See documentation of valid
                entries for argument ``p`` in :func:`torch.norm`.
            dim (int): index of the dim along which we define channels to prune.
            importance_scores (torch.Tensor): tensor of importance scores (of same
                shape as module parameter) used to compute mask for pruning.
                The values in this tensor indicate the importance of the corresponding
                elements in the parameter being pruned.
                If unspecified or None, the module parameter will be used in its place.
    
        Returns:
            module (nn.Module): modified (i.e. pruned) version of the input module
    
        Examples:
            >>> from torch.nn.utils import prune
            >>> m = prune.ln_structured(
            ...     nn.Conv2d(5, 3, 2), 'weight', amount=0.3, dim=1, n=float('-inf')
            ... )
        """
        # 调用 LnStructured 类的 apply 方法，对指定模块的参数进行结构化 L_n 范数剪枝
        LnStructured.apply(
            module, name, amount, n, dim, importance_scores=importance_scores
        )
        # 返回经过剪枝后的模块
        return module
# 确保参数 parameters 是一个元组的列表或生成器
def global_unstructured(parameters, pruning_method, importance_scores=None, **kwargs):
    r"""
    Globally prunes tensors corresponding to all parameters in ``parameters`` by applying the specified ``pruning_method``.

    Modifies modules in place by:

    1) adding a named buffer called ``name+'_mask'`` corresponding to the
       binary mask applied to the parameter ``name`` by the pruning method.
    2) replacing the parameter ``name`` by its pruned version, while the
       original (unpruned) parameter is stored in a new parameter named
       ``name+'_orig'``.

    Args:
        parameters (Iterable of (module, name) tuples): parameters of
            the model to prune in a global fashion, i.e. by aggregating all
            weights prior to deciding which ones to prune. module must be of
            type :class:`nn.Module`, and name must be a string.
        pruning_method (function): a valid pruning function from this module,
            or a custom one implemented by the user that satisfies the
            implementation guidelines and has ``PRUNING_TYPE='unstructured'``.
        importance_scores (dict): a dictionary mapping (module, name) tuples to
            the corresponding parameter's importance scores tensor. The tensor
            should be the same shape as the parameter, and is used for computing
            mask for pruning.
            If unspecified or None, the parameter will be used in place of its
            importance scores.
        kwargs: other keyword arguments such as:
            amount (int or float): quantity of parameters to prune across the
            specified parameters.
            If ``float``, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If ``int``, it represents the
            absolute number of parameters to prune.

    Raises:
        TypeError: if ``PRUNING_TYPE != 'unstructured'``

    Note:
        Since global structured pruning doesn't make much sense unless the
        norm is normalized by the size of the parameter, we now limit the
        scope of global pruning to unstructured methods.

    Examples:
        >>> from torch.nn.utils import prune
        >>> from collections import OrderedDict
        >>> net = nn.Sequential(OrderedDict([
        ...     ('first', nn.Linear(10, 4)),
        ...     ('second', nn.Linear(4, 1)),
        ... ]))
        >>> parameters_to_prune = (
        ...     (net.first, 'weight'),
        ...     (net.second, 'weight'),
        ... )
        >>> prune.global_unstructured(
        ...     parameters_to_prune,
        ...     pruning_method=prune.L1Unstructured,
        ...     amount=10,
        ... )
        >>> print(sum(torch.nn.utils.parameters_to_vector(net.buffers()) == 0))
        tensor(10)

    """
    # ensure parameters is a list or generator of tuples
    if not isinstance(parameters, Iterable):
        raise TypeError("global_unstructured(): parameters is not an Iterable")
    # 如果 importance_scores 已经定义，则保留其当前值，否则设置为空字典
    importance_scores = importance_scores if importance_scores is not None else {}
    # 检查 importance_scores 是否为字典类型，如果不是则引发类型错误
    if not isinstance(importance_scores, dict):
        raise TypeError("global_unstructured(): importance_scores must be of type dict")

    # 将重要性分数扁平化，以便在全局修剪中一次性考虑所有参数
    relevant_importance_scores = torch.nn.utils.parameters_to_vector(
        [
            # 获取 (module, name) 对应的重要性分数，如果不存在则使用参数本身的值
            importance_scores.get((module, name), getattr(module, name))
            for (module, name) in parameters
        ]
    )
    
    # 同样地，扁平化掩码（如果存在），否则使用与参数相同维度的全1向量
    default_mask = torch.nn.utils.parameters_to_vector(
        [
            # 获取 (module, name + "_mask") 对应的掩码，如果不存在则使用参数本身的全1向量
            getattr(module, name + "_mask", torch.ones_like(getattr(module, name)))
            for (module, name) in parameters
        ]
    )

    # 使用经典的修剪方法计算新的掩码，即使参数现在是 `parameters` 的扁平化版本
    container = PruningContainer()
    container._tensor_name = "temp"  # 以便与 `method` 的名称匹配
    method = pruning_method(**kwargs)
    method._tensor_name = "temp"  # 以便与 `container` 的名称匹配
    # 检查修剪方法的 PRUNING_TYPE 是否为 "unstructured"，否则引发类型错误
    if method.PRUNING_TYPE != "unstructured":
        raise TypeError(
            'Only "unstructured" PRUNING_TYPE supported for '
            f"the `pruning_method`. Found method {pruning_method} of type {method.PRUNING_TYPE}"
        )

    # 将修剪方法添加到容器中
    container.add_pruning_method(method)

    # 使用 `PruningContainer` 的 `compute_mask` 方法将新方法计算的掩码与预先存在的掩码结合
    final_mask = container.compute_mask(relevant_importance_scores, default_mask)

    # 指针，用于切片掩码以匹配每个参数的形状
    pointer = 0
    for module, name in parameters:
        param = getattr(module, name)
        # 参数的长度（即元素个数）
        num_param = param.numel()
        # 切片掩码，并将其重新形状为与参数相同的形状
        param_mask = final_mask[pointer : pointer + num_param].view_as(param)
        # 将正确的预计算掩码分配给每个参数，并像任何其他修剪方法一样添加到 forward_pre_hooks 中
        custom_from_mask(module, name, mask=param_mask)

        # 增加指针，继续切片 final_mask
        pointer += num_param
# 使用自定义的掩码对模块中名为 `name` 的参数进行修剪，修改模块本身并返回修改后的模块
def custom_from_mask(module, name, mask):
    # 调用自定义的掩码应用方法，实现参数修剪
    CustomFromMask.apply(module, name, mask)
    # 返回被修改（即被修剪）后的模块
    return module


# 从模块中移除修剪重新参数化和前向钩子中的修剪方法
def remove(module, name):
    # 遍历模块的前向钩子，查找与指定参数名相关的修剪方法
    for k, hook in module._forward_pre_hooks.items():
        # 如果找到与指定参数名相关的修剪方法
        if isinstance(hook, BasePruningMethod) and hook._tensor_name == name:
            # 移除修剪方法及其前向钩子
            hook.remove(module)
            del module._forward_pre_hooks[k]
            # 返回移除修剪后的模块
            return module
    
    # 若未找到指定参数名相关的修剪方法，则抛出异常
    raise ValueError(
        f"Parameter '{name}' of module {module} has to be pruned before pruning can be removed"
    )


# 检查模块是否被修剪，通过查找前向钩子中的修剪预钩子
def is_pruned(module):
    # 遍历模块中的前向钩子，检查是否存在继承自 BasePruningMethod 的修剪预钩子
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, BasePruningMethod):
            return True
    
    # 若未找到任何修剪预钩子，则模块未被修剪
    return False
    # 遍历模块中的命名子模块
    for _, submodule in module.named_modules():
        # 遍历子模块中的前向预钩子
        for hook in submodule._forward_pre_hooks.values():
            # 检查钩子是否是基于 BasePruningMethod 的实例
            if isinstance(hook, BasePruningMethod):
                # 如果是，则返回 True
                return True
    # 如果没有找到符合条件的钩子，返回 False
    return False
# 验证辅助函数，检查参数修剪量在初始化时的范围是否有效。
def _validate_pruning_amount_init(amount):
    # 检查amount是否为实数（整数或浮点数）
    if not isinstance(amount, numbers.Real):
        raise TypeError(f"Invalid type for amount: {amount}. Must be int or float.")

    # 如果amount是整数且为负数，或者是浮点数且不在[0, 1]范围内，引发值错误异常
    if (isinstance(amount, numbers.Integral) and amount < 0) or (
        not isinstance(amount, numbers.Integral)  # 这时amount应为浮点数
        and (float(amount) > 1.0 or float(amount) < 0.0)
    ):
        raise ValueError(
            f"amount={amount} should either be a float in the range [0, 1] or a non-negative integer"
        )


# 验证函数，检查修剪量相对于数据大小是否有效。
def _validate_pruning_amount(amount, tensor_size):
    # 如果amount是整数且大于tensor_size，引发值错误异常
    if isinstance(amount, numbers.Integral) and amount > tensor_size:
        raise ValueError(
            f"amount={amount} should be smaller than the number of parameters to prune={tensor_size}"
        )


# 验证函数，确保待修剪的张量至少是二维的。
def _validate_structured_pruning(t):
    # 获取张量的形状
    shape = t.shape
    # 如果张量的维度少于等于1，引发值错误异常
    if len(shape) <= 1:
        raise ValueError(
            "Structured pruning can only be applied to "
            "multidimensional tensors. Found tensor of shape "
            f"{shape} with {len(shape)} dims"
        )


# 计算函数，用于确定应修剪的参数数量
def _compute_nparams_toprune(amount, tensor_size):
    # 将修剪量从百分比转换为绝对值。

    # 由于修剪量可以表示为绝对值或张量中单元/通道数量的百分比，
    # 此实用函数将百分比转换为绝对值，以标准化修剪的处理方式。

    Args:
        amount (int or float): 要修剪的参数数量。
            如果是 float，应介于 0.0 和 1.0 之间，表示要修剪的参数比例。
            如果是 int，则表示要修剪的参数的绝对数量。
        tensor_size (int): 张量中要修剪的参数的绝对数量。

    Returns:
        int: 张量中要修剪的单元数量
    """
    # 错误的类型已在 _validate_pruning_amount_init 中检查过
    if isinstance(amount, numbers.Integral):
        return amount
    else:
        return round(amount * tensor_size)
# 验证修剪维度是否在张量维度的范围内
def _validate_pruning_dim(t, dim):
    r"""Validate that the pruning dimension is within the bounds of the tensor dimension.

    Args:
        t (torch.Tensor): tensor representing the parameter to prune
        dim (int): index of the dim along which we define channels to prune
    """
    # 如果指定的dim超过了张量t的维度数，则引发索引错误
    if dim >= t.dim():
        raise IndexError(f"Invalid index {dim} for tensor of size {t.shape}")


# 计算张量沿指定维度之外的所有维度上的L_n范数
def _compute_norm(t, n, dim):
    r"""Compute the L_n-norm of a tensor along all dimensions except for the specified dimension.

    The L_n-norm will be computed across all entries in tensor `t` along all dimension
    except for the one identified by dim.
    Example: if `t` is of shape, say, 3x2x4 and dim=2 (the last dim),
    then norm will have Size [4], and each entry will represent the
    `L_n`-norm computed using the 3x2=6 entries for each of the 4 channels.

    Args:
        t (torch.Tensor): tensor representing the parameter to prune
        n (int, float, inf, -inf, 'fro', 'nuc'): See documentation of valid
            entries for argument p in torch.norm
        dim (int): dim identifying the channels to prune

    Returns:
        norm (torch.Tensor): L_n norm computed across all dimensions except
            for `dim`. By construction, `norm.shape = t.shape[-1]`.
    """
    # dims = 所有轴的列表，除了由`dim`标识的轴
    dims = list(range(t.dim()))
    # 将负索引转换为正索引
    if dim < 0:
        dim = dims[dim]
    # 从dims列表中移除dim标识的维度
    dims.remove(dim)

    # 计算张量t在指定维度dim之外的所有维度上的L_n范数
    norm = torch.norm(t, p=n, dim=dims)
    return norm
```