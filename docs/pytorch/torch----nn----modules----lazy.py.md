# `.\pytorch\torch\nn\modules\lazy.py`

```
# mypy: allow-untyped-defs
# 导入 itertools 库，用于迭代操作
import itertools
# 导入类型提示相关的模块
from typing import Any, Optional, Protocol, Type

# 导入 PyTorch 库
import torch
# 从 torch.nn.parameter 模块中导入 is_lazy 函数
from torch.nn.parameter import is_lazy


# 定义 __all__ 列表，用于模块导入时指定可导出的符号
__all__ = ["LazyModuleMixin"]


# 定义一个私有的 _LazyProtocol 协议类，用于类型检查和属性声明
class _LazyProtocol(Protocol):
    """This class is used to avoid errors with mypy checks for the attributes in a mixin.

    https://mypy.readthedocs.io/en/latest/more_types.html#mixin-classes
    """

    # 声明一个未实现的方法 _register_load_state_dict_pre_hook，用于加载状态字典前的钩子注册
    def _register_load_state_dict_pre_hook(self, hook):
        ...

    # 声明 register_forward_pre_hook 方法，用于注册前向传播前的钩子函数
    def register_forward_pre_hook(self, hook, *, prepend=False, with_kwargs=False):
        ...

    # 声明 _lazy_load_hook 方法，用于懒加载的钩子函数
    def _lazy_load_hook(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        ...

    # 声明 _get_name 方法，用于获取对象的名称
    def _get_name(self):
        ...

    # 声明 _infer_parameters 方法，用于推断参数
    def _infer_parameters(self, module, input):
        ...

    # 声明 _parameters 属性，用于获取参数
    @property
    def _parameters(self):
        ...

    # 声明 _buffers 属性，用于获取缓冲区
    @property
    def _buffers(self):
        ...

    # 声明 _non_persistent_buffers_set 属性，用于获取非持久性缓冲区集合
    @property
    def _non_persistent_buffers_set(self):
        ...

    # 声明 _load_hook 属性，用于获取加载钩子
    @property
    def _load_hook(self):
        ...

    # 声明 _initialize_hook 属性，用于获取初始化钩子
    @property
    def _initialize_hook(self):
        ...


# 定义 LazyModuleMixin 类，作为延迟初始化参数的模块混合类
class LazyModuleMixin:
    r"""A mixin for modules that lazily initialize parameters, also known as "lazy modules".

    .. warning:
        Lazy modules are an experimental new feature under active development,
        and their API is likely to change.

    Modules that lazily initialize parameters, or "lazy modules",
    derive the shapes of their parameters from the first input(s)
    to their forward method. Until that first forward they contain
    :class:`torch.nn.UninitializedParameter` s that should not be accessed
    or used, and afterward they contain regular :class:`torch.nn.Parameter` s.
    Lazy modules are convenient since they don't require computing some
    module arguments, like the :attr:`in_features` argument of a
    typical :class:`torch.nn.Linear`.

    After construction, networks with lazy modules should first
    be converted to the desired dtype and placed on the expected device.
    This is because lazy modules only perform shape inference so the usual dtype
    and device placement behavior applies.
    The lazy modules should then perform "dry runs" to initialize all the components in the module.
    These "dry runs" send inputs of the correct size, dtype, and device through
    the network and to each one of its lazy modules. After this the network can be used as usual.

    >>> # xdoctest: +SKIP
    >>> class LazyMLP(torch.nn.Module):
    ...    def __init__(self):
    ...        super().__init__()
    ...        self.fc1 = torch.nn.LazyLinear(10)
    ...        self.relu1 = torch.nn.ReLU()
    ...        self.fc2 = torch.nn.LazyLinear(1)
    ...        self.relu2 = torch.nn.ReLU()
    ...
    ...    def forward(self, input):
    ...        x = self.relu1(self.fc1(input))
    ...        y = self.relu2(self.fc2(x))
    ...        return y
    >>> # constructs a network with lazy modules
    >>> lazy_mlp = LazyMLP()
    # 创建一个 LazyMLP 的实例对象 lazy_mlp
    
    >>> # transforms the network's device and dtype
    # 转换网络的设备和数据类型
    
    >>> # NOTE: these transforms can and should be applied after construction and before any 'dry runs'
    # 注意：这些转换应该在构建后和任何“dry runs”（干跑）之前应用
    
    >>> lazy_mlp = lazy_mlp.cuda().double()
    # 将 lazy_mlp 移动到 CUDA 设备，并将数据类型转换为 double（双精度浮点数）
    
    >>> lazy_mlp
    # 打印 lazy_mlp 实例，显示其结构及其包含的模块
    
    LazyMLP( (fc1): LazyLinear(in_features=0, out_features=10, bias=True)
      (relu1): ReLU()
      (fc2): LazyLinear(in_features=0, out_features=1, bias=True)
      (relu2): ReLU()
    )
    
    >>> # performs a dry run to initialize the network's lazy modules
    # 执行一个“dry run”（干跑），初始化网络中的 lazy 模块
    
    >>> lazy_mlp(torch.ones(10,10).cuda())
    # 对 lazy_mlp 输入全为1的大小为(10, 10)的张量，这会初始化 LazyLinear 模块
    
    >>> # after initialization, LazyLinear modules become regular Linear modules
    # 初始化后，LazyLinear 模块变成常规的 Linear 模块
    
    >>> lazy_mlp
    # 打印 lazy_mlp 实例，显示现在每个模块的类型已经变化为 Linear
    
    LazyMLP(
      (fc1): Linear(in_features=10, out_features=10, bias=True)
      (relu1): ReLU()
      (fc2): Linear(in_features=10, out_features=1, bias=True)
      (relu2): ReLU()
    )
    
    >>> # attaches an optimizer, since parameters can now be used as usual
    # 现在可以像往常一样使用参数，因此附加一个优化器
    
    >>> optim = torch.optim.SGD(mlp.parameters(), lr=0.01)
    # 使用随机梯度下降（SGD）优化器来优化 lazy_mlp 中的参数
    
    A final caveat when using lazy modules is that the order of initialization of a network's
    parameters may change, since the lazy modules are always initialized after other modules.
    For example, if the LazyMLP class defined above had a :class:`torch.nn.LazyLinear` module
    first and then a regular :class:`torch.nn.Linear` second, the second module would be
    initialized on construction and the first module would be initialized during the first dry run.
    This can cause the parameters of a network using lazy modules to be initialized differently
    than the parameters of a network without lazy modules as the order of parameter initializations,
    which often depends on a stateful random number generator, is different.
    Check :doc:`/notes/randomness` for more details.
    
    Lazy modules can be serialized with a state dict like other modules. For example:
    
    >>> lazy_mlp = LazyMLP()
    # 创建一个新的 LazyMLP 实例对象 lazy_mlp
    
    >>> # The state dict shows the uninitialized parameters
    # 状态字典显示未初始化的参数
    
    >>> lazy_mlp.state_dict()
    # 显示 lazy_mlp 实例对象的状态字典，展示出未初始化的参数
    
    OrderedDict([('fc1.weight', Uninitialized parameter),
                 ('fc1.bias',
                  tensor([-1.8832e+25,  4.5636e-41, -1.8832e+25,  4.5636e-41, -6.1598e-30,
                           4.5637e-41, -1.8788e+22,  4.5636e-41, -2.0042e-31,  4.5637e-41])),
                 ('fc2.weight', Uninitialized parameter),
                 ('fc2.bias', tensor([0.0019]))])
    
    Lazy modules can load regular :class:`torch.nn.Parameter` s (i.e. you can serialize/deserialize
    initialized LazyModules and they will remain initialized)
    
    >>> full_mlp = LazyMLP()
    # 创建一个新的 LazyMLP 实例对象 full_mlp
    
    >>> # Dry run to initialize another module
    # 运行干跑以初始化另一个模块
    
    >>> full_mlp.forward(torch.ones(10, 1))
    # 对 full_mlp 输入全为1的大小为(10, 1)的张量，这会初始化 LazyLinear 模块
    
    >>> # Load an initialized state into a lazy module
    # 将一个已初始化的状态加载到一个 lazy 模块中
    
    >>> lazy_mlp.load_state_dict(full_mlp.state_dict())
    # 使用 full_mlp 的状态字典来加载 lazy_mlp 的状态，使其保持初始化状态
    
    >>> # The state dict now holds valid values
    # 现在状态字典包含有效的值
    
    >>> lazy_mlp.state_dict()
    # 显示 lazy_mlp 实例对象的状态字典，现在所有参数都已经初始化
    OrderedDict([('fc1.weight',
                  tensor([[-0.3837],
                          [ 0.0907],
                          [ 0.6708],
                          [-0.5223],
                          [-0.9028],
                          [ 0.2851],
                          [-0.4537],
                          [ 0.6813],
                          [ 0.5766],
                          [-0.8678]])),
                 ('fc1.bias',
                  tensor([-1.8832e+25,  4.5636e-41, -1.8832e+25,  4.5636e-41, -6.1598e-30,
                           4.5637e-41, -1.8788e+22,  4.5636e-41, -2.0042e-31,  4.5637e-41])),
                 ('fc2.weight',
                  tensor([[ 0.1320,  0.2938,  0.0679,  0.2793,  0.1088, -0.1795, -0.2301,  0.2807,
                            0.2479,  0.1091]])),
                 ('fc2.bias', tensor([0.0019]))])

    Note, however, that the loaded parameters will not be replaced when doing a "dry run" if they are initialized
    when the state is loaded. This prevents using initialized modules in different contexts.
    """

    # 定义一个可选的类类型，初始化为 None，用于指定初始化完成后需要变成的类
    cls_to_become: Optional[Type[Any]] = None

    def __init__(self: _LazyProtocol, *args, **kwargs):
        # 调用父类的初始化方法，传递所有参数
        super().__init__(*args, **kwargs)  # type: ignore[misc]
        # 注册一个状态字典加载前的钩子函数，用于懒加载
        self._load_hook = self._register_load_state_dict_pre_hook(self._lazy_load_hook)
        # 注册一个前向传播前的钩子函数，用于推断参数
        self._initialize_hook = self.register_forward_pre_hook(
            self._infer_parameters, with_kwargs=True
        )

    def _save_to_state_dict(self: _LazyProtocol, destination, prefix, keep_vars):
        # 将对象的参数保存到状态字典中
        for name, param in self._parameters.items():
            if param is not None:
                # 如果参数不是懒加载或者需要保持变量，则将其分离（detach）
                if not (is_lazy(param) or keep_vars):
                    param = param.detach()
                destination[prefix + name] = param
        # 将对象的缓冲区保存到状态字典中
        for name, buf in self._buffers.items():
            if buf is not None and name not in self._non_persistent_buffers_set:
                # 如果缓冲区不是懒加载或者需要保持变量，则将其分离（detach）
                if not (is_lazy(buf) or keep_vars):
                    buf = buf.detach()
                destination[prefix + name] = buf

    def _lazy_load_hook(
        self: _LazyProtocol,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
        ):
        # 懒加载钩子函数，在加载状态字典时调用
    ):
        """
        load_state_dict 的预钩子函数，用于懒加载的缓冲区和参数。

        该钩子的目的是调整当前状态和/或正在加载的 `state_dict`，以便可以将序列化为未初始化和已初始化状态的模块实例反序列化为未初始化和已初始化的模块实例。
        详细的钩子规范，请参见 `torch.nn.Module._register_load_state_dict_pre_hook` 中的注释。

        参数:
            prefix: 用于调整 `state_dict` 中键名的前缀。
            state_dict: 包含待加载状态的字典。
        """
        for name, param in itertools.chain(
            self._parameters.items(), self._buffers.items()
        ):
            key = prefix + name
            if key in state_dict and param is not None:
                input_param = state_dict[key]
                if is_lazy(param):
                    # 当前参数未初始化，但加载的参数已初始化
                    # 基于未初始化的参数创建一个新的参数
                    if not is_lazy(input_param):
                        with torch.no_grad():
                            param.materialize(input_param.shape)

    def initialize_parameters(self: _LazyProtocol, *args, **kwargs):
        """
        根据输入批处理属性初始化参数。

        这提供了一个接口，将参数初始化与前向传递分离，用于参数形状推断。
        """
        raise NotImplementedError(
            f"initialize_parameters is not implemented for {self.__class__.__name__}"
        )

    def has_uninitialized_params(self: _LazyProtocol):
        """
        检查模块是否具有未初始化的参数。

        返回:
            bool: 如果有未初始化的参数返回 True，否则返回 False。
        """
        # 避免 JIT 跟踪此参数并强制自定义模块的 `__setstate__` 添加它
        params = self._parameters.values()
        buffers = self._buffers.values()
        for param in itertools.chain(params, buffers):
            if is_lazy(param):
                return True
        return False

    # torchrec 测试代码一致性
    # fmt: off
    def _infer_parameters(self: _LazyProtocol, module, args, kwargs=None):
        r"""Infers the size and initializes the parameters according to the provided input batch.

        Given a module that contains parameters that were declared inferrable
        using :class:`torch.nn.parameter.ParameterMode.Infer`, runs a forward pass
        in the complete module using the provided input to initialize all the parameters
        as needed.
        The module is set into evaluation mode before running the forward pass in order
        to avoid saving statistics or calculating gradients
        """
        # 如果 kwargs 为 None，则将其初始化为空字典
        kwargs = kwargs if kwargs else {}
        # 调用模块的 initialize_parameters 方法，使用给定的 args 和 kwargs 初始化参数
        module.initialize_parameters(*args, **kwargs)
        # 如果模块中还有未初始化的参数，则抛出 RuntimeError 异常
        if module.has_uninitialized_params():
            raise RuntimeError(f'module {self._get_name()} has not been fully initialized')
        # 移除模块的 _initialize_hook 和 _load_hook 属性
        module._initialize_hook.remove()
        module._load_hook.remove()
        # 删除模块的 _initialize_hook 和 _load_hook 属性
        delattr(module, '_initialize_hook')
        delattr(module, '_load_hook')
        # 如果模块有 cls_to_become 属性，则将模块的类更改为 cls_to_become
        if module.cls_to_become is not None:
            module.__class__ = module.cls_to_become
    # fmt: on

    def _replicate_for_data_parallel(self: _LazyProtocol):
        # 抛出 RuntimeError 异常，说明具有未初始化参数的模块无法使用 DataParallel
        raise RuntimeError(
            "Modules with uninitialized parameters can't be used with `DataParallel`. "
            "Run a dummy forward pass to correctly initialize the modules"
        )
```